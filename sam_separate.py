"""Separate the active sound from cabin/road noise with SAM Audio (large) on Replicate.

    export REPLICATE_API_TOKEN=r8_...
    .venv/bin/python sam_separate.py                      # both rev takes, default descriptions
    .venv/bin/python sam_separate.py FILE.wav -d "synthesizer" -d "engine hum"
    .venv/bin/python sam_separate.py --residual           # also save "everything else"

Writes analysis/sam_audio/<take>__<description>.wav (and __RESIDUAL.wav). Runs that
already have an output file are skipped, so re-running only pays for new ones.
Each run uploads the take to Replicate for processing (not published anywhere).
"""
from __future__ import annotations

import argparse
import io
import math
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np
import replicate

MODEL = ("geopti/sam-audio-large:"
         "d8a8a4fcdcbf0bdc863f6d98cd2117ec0bc02224b576c7b98b2a009a8a1f83fa")
HERE = Path(__file__).parent
TAKES = HERE / "analysis/boom_ioniq5n/opposite_decode"
DEFAULT_FILES = [TAKES / "evolution_rev.wav", TAKES / "supersonic_rev.wav"]
DEFAULT_DESCRIPTIONS = [
    "electric car sound",
    "synthesizer",
    "engine humming",
    "electric motor whine",
    "road noise",          # the background itself: its residual is the car sound
]
OUT = HERE / "analysis/sam_audio"
# Replicate allows 6 predictions a minute (burst 1) while credit is under $5:
# start at most one every 11 s, and back off and retry if throttled anyway.
MIN_GAP_SEC = 11.0
RETRIES = 5
CHUNK_SEC = 30.0       # longer takes are split (85 s ran the hosted GPU out of memory)
OVERLAP_SEC = 2.0      # pieces overlap by this much and are crossfaded
_last_start = 0.0


def predict(inputs: dict):
    global _last_start
    for attempt in range(RETRIES):
        wait = _last_start + MIN_GAP_SEC - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_start = time.monotonic()
        if hasattr(inputs.get("audio"), "seek"):
            inputs["audio"].seek(0)              # a retry must upload the whole file again
        try:
            return replicate.run(MODEL, input=inputs)
        except Exception as e:
            if "throttled" not in str(e).lower() and "429" not in str(e):
                raise
            delay = MIN_GAP_SEC * (attempt + 1)
            print(f"  throttled; retrying in {delay:.0f} s", flush=True)
            time.sleep(delay)
    raise RuntimeError("still throttled after retries")


def slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")


def fetch(item) -> bytes:
    """An output may be a FileOutput (has .read()) or a plain URL string."""
    if hasattr(item, "read"):
        return item.read()
    import urllib.request
    with urllib.request.urlopen(str(item)) as r:
        return r.read()


def read_wav(data: bytes):
    """-> (samples float [n, channels], rate). 16/24/32-bit PCM."""
    with wave.open(io.BytesIO(data)) as w:
        rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
        x = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8)
             | (b[:, 2].astype(np.int8).astype(np.int32) << 16)) / 2 ** 23
    else:
        x = np.frombuffer(raw, {1: np.uint8, 2: "<i2", 4: "<i4"}[width]).astype(np.float64)
        x = (x - 128) / 128 if width == 1 else x / 2 ** (8 * width - 1)
    return x.reshape(-1, ch), rate


def write_wav(path: Path, x, rate: int):
    x = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    x = x.reshape(len(x), -1)
    with wave.open(path if hasattr(path, "write") else str(path), "wb") as w:
        w.setnchannels(x.shape[1])
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((x * 32767).round().astype("<i2").tobytes())


def level_db(x) -> float:
    return 10 * np.log10(np.mean(np.square(x)) + 1e-12)


def separate(data: bytes, description: str, residual: bool, predict_spans: bool):
    """One prediction on one WAV (bytes) -> (target, residual or None, rate)."""
    output = predict({
        "audio": io.BytesIO(data),
        "description": description,
        "span_anchors": "[]",
        "predict_spans": predict_spans,
        "output_residual": residual,
        "use_span_prompting": False,
    })
    items = output if isinstance(output, (list, tuple)) else [output]
    tgt, rate = read_wav(fetch(items[0]))
    res = read_wav(fetch(items[1]))[0] if residual and len(items) > 1 else None
    return tgt, res, rate


def run(audio: Path, description: str, residual: bool, predict_spans: bool):
    stem = f"{audio.stem}__{slug(description)}"
    target = OUT / f"{stem}.wav"
    resid = OUT / f"{stem}__RESIDUAL.wav"
    if target.exists() and (resid.exists() or not residual):
        print(f"skip {stem} (already done)")
        return
    src, src_rate = read_wav(audio.read_bytes())
    sec = len(src) / src_rate
    # long takes run the hosted GPU out of memory: split into overlapping pieces
    n_chunks = max(1, math.ceil((sec - OVERLAP_SEC) / (CHUNK_SEC - OVERLAP_SEC)))
    print(f"{audio.name} ({sec:.0f} s{f', {n_chunks} pieces' if n_chunks > 1 else ''}): "
          f"'{description}' ...", flush=True)
    hop = int((CHUNK_SEC - OVERLAP_SEC) * src_rate)
    size = int(CHUNK_SEC * src_rate)
    tgt_parts, res_parts, rate = [], [], None
    for k in range(n_chunks):
        piece = src[k * hop:k * hop + size] if n_chunks > 1 else src
        buf = io.BytesIO()
        write_wav(buf, piece, src_rate)
        if n_chunks > 1:
            print(f"  piece {k + 1}/{n_chunks}", flush=True)
        t, r, rate = separate(buf.getvalue(), description, residual, predict_spans)
        tgt_parts.append(t.mean(axis=1))
        res_parts.append(r.mean(axis=1) if r is not None else None)
    out_hop = int((CHUNK_SEC - OVERLAP_SEC) * rate)
    fade = int(OVERLAP_SEC * rate)

    def join(parts):
        """Overlap-add with equal-power crossfades across each overlap."""
        total = out_hop * (len(parts) - 1) + len(parts[-1])
        y = np.zeros(total)
        for k, part in enumerate(parts):
            g = np.ones(len(part))
            ramp = np.sin(np.linspace(0, np.pi / 2, fade)) ** 2
            if k > 0:
                g[:fade] = ramp[:len(g[:fade])]
            if k < len(parts) - 1:
                g[-fade:] = ramp[::-1][-len(g[-fade:]):]
            y[k * out_hop:k * out_hop + len(part)] += part * g
        return y

    y = join(tgt_parts)
    write_wav(target, y, rate)
    print(f"  wrote {target.relative_to(HERE)} (level {level_db(y) - level_db(src):+.1f} dB vs input)")
    if level_db(y) < -60:
        print("  ^ almost silent: the model found nothing matching this description;"
              " try another (-d)")
    if residual:
        if any(r is None for r in res_parts):
            print("  (no residual returned)")
        else:
            write_wav(resid, join(res_parts), rate)
            print(f"  wrote {resid.relative_to(HERE)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path, default=DEFAULT_FILES)
    ap.add_argument("-d", "--description", action="append",
                    help="what to keep (repeatable); default: a set of car-sound phrases")
    ap.add_argument("--residual", action="store_true", help="also save everything else")
    ap.add_argument("--predict-spans", action="store_true",
                    help="let the model find when the sound occurs (for intermittent sounds)")
    args = ap.parse_args()
    if not os.environ.get("REPLICATE_API_TOKEN"):
        sys.exit("set REPLICATE_API_TOKEN first: export REPLICATE_API_TOKEN=r8_...")
    OUT.mkdir(parents=True, exist_ok=True)
    for audio in args.files:
        for d in args.description or DEFAULT_DESCRIPTIONS:
            try:
                run(audio, d, args.residual, args.predict_spans)
            except Exception as e:          # keep going with the other runs
                print(f"  failed: {e}")


if __name__ == "__main__":
    main()
