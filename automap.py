#!/usr/bin/env python3
"""Map a recording to speed automatically, from its pitch -- a starting point for mapper.py.

    .venv/bin/python automap.py some_rev.wav --band 55 165 --top-kmh 110
    .venv/bin/python mapper.py some_rev.wav        # then refine it by ear

Works for tonal recordings whose pitch follows the motor (EV sound generators, motors,
turbines). It follows the strongest continuous line in --band through the whole file
(a Viterbi path, so it doesn't hop between harmonics), then picks:
  accel    the rising passage covering the most pitch     -> used while speeding up
  decel    the falling passage covering the most pitch    -> used while slowing down
  plateau  the steadiest stretch near the top pitch       -> held past the top speed
Speed is taken as proportional to pitch above idle: idle pitch = 0 km/h, the accel
passage's top pitch = --top-kmh. Writes maps/<file>.json (main.py lists it as a voice)
and maps/<file>.png: the spectrogram with the pitch track and the chosen passages.

Pick --band so it contains ONE harmonic line through its whole travel: look at the png
of a first run (default band 50-400 Hz) and narrow it if the track jumps between lines.
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import zlib
from pathlib import Path

import numpy as np

MAPS_DIR = Path(__file__).parent / "maps"


def _load_wav(path, sr):
    """Plain PCM WAV without ffmpeg: mono mix, then band-limited (FFT) resampling."""
    import wave
    with wave.open(str(path)) as w:
        ch, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        x = (b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16))
        x = np.where(x >= 1 << 23, x - (1 << 24), x) / float(1 << 23)
    elif width in (1, 2, 4):
        x = np.frombuffer(raw, {1: np.uint8, 2: "<i2", 4: "<i4"}[width]).astype(np.float64)
        x = (x - 128.0) / 128.0 if width == 1 else x / float(1 << (8 * width - 1))
    else:
        raise ValueError(f"unsupported WAV sample width {width}")
    x = x.reshape(-1, ch).sum(axis=1) / np.sqrt(ch)     # same mono mix as ffmpeg -ac 1
    if rate != sr and len(x):
        n2 = int(round(len(x) * sr / rate))
        X = np.fft.rfft(x)
        Y = np.zeros(n2 // 2 + 1, dtype=complex)
        k = min(len(X), len(Y))
        Y[:k] = X[:k]
        x = np.fft.irfft(Y, n2) * (n2 / len(x))
    return x


def load(path, sr):
    """Any audio file as mono float64 at `sr`. WAV files are read directly (no ffmpeg
    needed -- Raspberry Pi OS Lite doesn't ship it); other formats go through ffmpeg."""
    if str(path).lower().endswith(".wav"):
        try:
            return _load_wav(path, sr)
        except Exception:
            pass                       # e.g. float or extensible WAV: let ffmpeg read it
    import shutil
    if not shutil.which("ffmpeg"):
        raise RuntimeError(f"ffmpeg is needed to read {Path(path).name}: sudo apt install ffmpeg")
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, "<f4").astype(np.float64)


def log_spectrogram(x, sr, fmin, fmax, hop=0.02, nfft=4096, nb=240):
    win = np.hanning(nfft)
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    bins = np.geomspace(fmin, fmax, nb)
    h = int(hop * sr)
    frames = []
    for i in range(0, max(1, len(x) - nfft), h):
        seg = x[i:i + nfft]
        if len(seg) < nfft:
            seg = np.pad(seg, (0, nfft - len(seg)))
        frames.append(20 * np.log10(np.interp(bins, freqs, np.abs(np.fft.rfft(seg * win))) + 1e-9))
    times = (np.arange(len(frames)) * h + nfft / 2) / sr
    return times, bins, np.array(frames)


def ridge(times, bins, spec, jump_db=0.6):
    """Strongest continuous line: maximise salience minus a cost per bin of jump."""
    e = spec - np.median(spec, axis=1, keepdims=True)
    n_t, n_b = e.shape
    idx = np.arange(n_b)
    pen = jump_db * np.abs(idx[:, None] - idx[None, :])
    cost = e[0].copy()
    back = np.zeros((n_t, n_b), dtype=int)
    for t in range(1, n_t):
        tot = cost[None, :] - pen
        back[t] = tot.argmax(1)
        cost = tot.max(1) + e[t]
    path = np.zeros(n_t, dtype=int)
    path[-1] = int(cost.argmax())
    for t in range(n_t - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return bins[path]


def harmonic_salience(x, sr, lo, hi, harmonics=8, nb=240):
    """For each candidate pitch in [lo, hi]: the summed (whitened) level at its first few
    harmonics. Follows the fundamental of sounds whose harmonics are all similar in
    strength, where the single strongest line hops from one harmonic to another."""
    top = min(hi * harmonics, sr / 2 - 100)
    t, b, s = log_spectrogram(x, sr, lo * 0.9, top, nb=4 * nb)
    e = s - np.median(s, axis=0, keepdims=True)          # constant lines drop out
    e = np.clip(e - np.median(e, axis=1, keepdims=True), 0, 30)
    cand = np.geomspace(lo, hi, nb)
    lb = np.log(b)
    sal = np.zeros((len(t), nb))
    for k in range(1, harmonics + 1):
        sal += np.array([np.interp(np.log(k * cand), lb, row, right=0.0) for row in e]) / k ** 0.3
    return t, cand, sal


def runs(f0, rising, slack=0.03, min_len=6):
    """Monotonic passages (allowing small wobbles): (start, end, pitch ratio covered)."""
    out = []
    n = len(f0)
    i = 0
    while i < n - 1:
        j, best, best_j = i, f0[i], i
        while j + 1 < n:
            nxt = f0[j + 1]
            if rising:
                if nxt < best * (1 - slack):
                    break
                if nxt >= best:
                    best, best_j = nxt, j + 1
            else:
                if nxt > best * (1 + slack):
                    break
                if nxt <= best:
                    best, best_j = nxt, j + 1
            j += 1
        if best_j - i >= min_len:
            ratio = f0[best_j] / f0[i] if rising else f0[i] / f0[best_j]
            out.append((i, best_j, ratio))
        i = max(i + 1, best_j)
    return out


def plateau(f0, lo_pitch, min_frames=15, tol=0.03):
    """Longest stretch with pitch within +-tol of its mean, at or above lo_pitch."""
    best = None
    for i in range(len(f0)):
        if f0[i] < lo_pitch:
            continue
        j = i
        while j + 1 < len(f0) and abs(f0[j + 1] / f0[i] - 1) < tol and f0[j + 1] >= lo_pitch:
            j += 1
        if j - i >= min_frames and (best is None or j - i > best[1] - best[0]):
            best = (i, j)
    return best


def steady(times, f0, min_sec=0.15, tol=0.025, same=0.04):
    """Steady stretches, the longest one per distinct pitch: [(start, end, hz)]."""
    segs, i = [], 0
    while i < len(f0):
        j = i
        while j + 1 < len(f0) and abs(f0[j + 1] / f0[i] - 1) < tol:
            j += 1
        if times[j] - times[i] >= min_sec:
            segs.append((float(times[i]), float(times[j]), float(np.median(f0[i:j + 1]))))
        i = j + 1
    best = []
    for seg in sorted(segs, key=lambda s: s[1] - s[0], reverse=True):
        if all(abs(seg[2] / b[2] - 1) > same for b in best):
            best.append(seg)
    return sorted(best, key=lambda s: s[2])


def refine(x, sr, seg, search=0.08, min_prominence_db=9.0):
    """Exact pitch of a steady stretch from one long, zero-padded spectrum (the line
    tracker only resolves ~0.5 % steps). Drops stretches without a clear peak there."""
    a, b, f = seg
    y = x[int(a * sr):int(b * sr)]
    if len(y) < sr * 0.1:
        return None
    n = 1 << 18
    spec = 20 * np.log10(np.abs(np.fft.rfft(y * np.hanning(len(y)), n)) + 1e-12)
    fr = np.fft.rfftfreq(n, 1 / sr)
    m = (fr > f * (1 - search)) & (fr < f * (1 + search))
    k = np.flatnonzero(m)[np.argmax(spec[m])]
    if 0 < k < len(spec) - 1:                       # parabolic peak interpolation
        al, be, ga = spec[k - 1], spec[k], spec[k + 1]
        k = k + 0.5 * (al - ga) / (al - 2 * be + ga)
    peak = float(np.interp(k, np.arange(len(fr)), fr))
    around = (fr > f * 0.5) & (fr < f * 1.5)
    prominence = spec[m].max() - np.median(spec[around])
    if prominence < min_prominence_db:
        return None
    return (a, b, peak)


def enhance(path, times, f0, out_path, cut_db=12.0, width=0.03, max_hz=6000.0, hp_hz=40.0):
    """Bring the tonal sound forward: keep each STFT frame's energy at the harmonics of
    the tracked pitch and turn down the noise between them by cut_db (soft mask, so it
    stays natural). Also removes rumble below hp_hz. Writes a 48 kHz mono WAV."""
    import wave
    sr = 48000
    x = load(path, sr)
    n, hop = 8192, 1024
    win = np.hanning(n)
    pad = np.concatenate([np.zeros(n), x, np.zeros(n)])
    fr = np.fft.rfftfreq(n, 1 / sr)
    floor = 10 ** (-cut_db / 20)
    out = np.zeros(len(pad))
    norm = np.zeros(len(pad))
    for i in range(0, len(pad) - n, hop):
        t = (i + n / 2 - n) / sr                                  # centre time in x
        f = float(np.interp(t, times, f0))
        spec = np.fft.rfft(pad[i:i + n] * win)
        k = fr / f                                                 # position in harmonics
        nearest = np.maximum(1.0, np.round(k))
        dist_hz = np.abs(k - np.round(k)) * f                      # to the nearest harmonic
        # lobe a little wider for higher harmonics (a wobbling pitch moves them further),
        # never narrower than the window's own main lobe
        sigma = np.maximum(1.5 * sr / n, width * f * np.sqrt(nearest))
        lobe = np.exp(-0.5 * (dist_hz / sigma) ** 2)
        gain = floor + (1 - floor) * lobe
        gain[fr > max_hz] = floor ** 0.5                          # top end: gentler
        gain[fr < hp_hz] = 0.0
        gain[k < 0.5] = np.minimum(gain[k < 0.5], floor)          # below the fundamental
        out[i:i + n] += np.fft.irfft(spec * gain, n) * win
        norm[i:i + n] += win ** 2
    y = out / np.maximum(norm, 1e-6)
    y = y[n:n + len(x)]
    y = y / (np.abs(y).max() + 1e-9) * 0.9
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((y * 32767).astype("<i2").tobytes())
    return out_path


def points(times, f0, i0, i1, f_idle, f_top, top_kmh, rising, step_kmh=5.0):
    """(km/h, seconds) along a passage: first moment its pitch reaches each speed's pitch."""
    seg_t, seg_f = times[i0:i1 + 1], f0[i0:i1 + 1]
    if not rising:
        seg_t, seg_f = seg_t[::-1], seg_f[::-1]          # walk the fall from the bottom up
    seg_f = np.maximum.accumulate(seg_f)                 # one-way: tracker glitches can't
                                                         # send the map backwards
    pts = []
    kmh = 0.0
    while kmh <= top_kmh + 1e-6:
        target = f_idle + (f_top - f_idle) * kmh / top_kmh
        k = np.argmax(seg_f >= target) if np.any(seg_f >= target) else None
        if k is None:
            break
        pts.append((round(kmh, 1), round(float(seg_t[k]), 3)))
        kmh += step_kmh
    # keep positions moving one way along the passage (drop stalls from wobbles)
    clean = [pts[0]] if pts else []
    for p in pts[1:]:
        if (p[1] > clean[-1][1]) if rising else (p[1] < clean[-1][1]):
            clean.append(p)
    return clean


def write_png(path, times, bins_img, img_db, f0, marks):
    h, w = img_db.shape[1], img_db.shape[0]
    top = np.percentile(img_db, 99.7)
    s = np.clip((img_db.T[::-1] - (top - 70)) / 70, 0, 1)            # rows: high f at top
    rgb = np.stack([np.clip(1.6 * s, 0, 1), np.clip(1.6 * s - 0.6, 0, 1),
                    np.clip(np.where(s < 0.5, 1.2 * s, 1.2 - 1.2 * s), 0, 1) * 0.8], -1)
    img = (rgb * 255).astype(np.uint8)
    rows = bins_img[::-1]
    for (i0, i1), color in marks:                                   # passages: bar on top
        img[:8, i0:i1 + 1] = color
    for t_i, f in enumerate(f0):                                     # pitch track: cyan
        y = int(np.argmin(np.abs(rows - f)))
        img[max(0, y - 1):y + 2, t_i] = (0, 255, 255)
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + \
            struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--band", nargs=2, type=float, default=(50.0, 400.0), metavar=("LO", "HI"),
                    help="frequency band (Hz) holding the one harmonic line to follow")
    ap.add_argument("--top-kmh", type=float, default=110.0,
                    help="speed at the top of the accel passage's pitch (default 110)")
    ap.add_argument("--name", default=None, help="voice name (default 'Recording: <file>')")
    ap.add_argument("--samples", action="store_true",
                    help="instrument mode: play only the steady stretches, retuned to the "
                         "speed's pitch (best for revs and sweeps; see SampledRecording)")
    ap.add_argument("--min-sec", type=float, default=0.17,
                    help="with --samples: shortest steady stretch to use (default 0.17 s)")
    ap.add_argument("--max-hz", type=float, default=None,
                    help="with --samples: ignore steady stretches above this pitch")
    ap.add_argument("--enhance", type=float, default=None, metavar="DB",
                    help="write a cleaned copy to sounds/ (noise between the harmonics cut "
                         "by DB, rumble removed) and map that instead; 10-15 is natural")
    ap.add_argument("--harmonic", action="store_true",
                    help="--band is the FUNDAMENTAL's range; track it from all its harmonics "
                         "together (for sounds whose harmonics are all similar in strength)")
    ap.add_argument("--gate-db", type=float, default=35.0,
                    help="frames this far below the loud parts count as silent (default 35)")
    for opt, what in (("--accel", "rising passage"), ("--decel", "falling passage"),
                      ("--plateau", "held top-pitch stretch")):
        ap.add_argument(opt, nargs=2, type=float, metavar=("START", "END"),
                        help=f"use this {what} (seconds) instead of the automatic pick")
    args = ap.parse_args()

    src = Path(args.audio).resolve()
    sr = 16000
    x = load(src, sr)
    times, bins, spec = log_spectrogram(x, sr, *args.band)
    if args.harmonic:
        _, cand, sal = harmonic_salience(x, sr, *args.band)
        f0 = ridge(times, cand, sal[:len(times)] + np.median(sal, axis=1, keepdims=True)[:len(times)])
    else:
        f0 = ridge(times, bins, spec)
    f0 = np.array([np.median(f0[max(0, i - 2):i + 3]) for i in range(len(f0))])  # de-speckle
    # silent frames (e.g. a separated stem that is silent at idle): the tracker wanders
    # there, so nothing is picked from them
    level = spec.max(axis=1)
    quiet = level < np.percentile(level, 95) - args.gate_db
    loud = lambda i0, i1: not quiet[i0:i1 + 1].any()
    if quiet.any():
        print(f"ignoring {quiet.mean() * 100:.0f}% of frames as silent "
              f"(> {args.gate_db:.0f} dB below the loud parts)")

    near = lambda sec: int(np.argmin(np.abs(times - sec)))
    ups = [r for r in runs(f0, True) if loud(r[0], r[1])]
    downs = [r for r in runs(f0, False) if loud(r[0], r[1])]
    if args.accel:
        a0, a1 = near(args.accel[0]), near(args.accel[1])
    elif ups:
        a0, a1, _ = max(ups, key=lambda r: (r[2], r[1] - r[0]))
    else:
        raise SystemExit("no rising passage found -- try a different --band, or --accel")
    if args.decel:
        downs = [(near(args.decel[0]), near(args.decel[1]), 0.0)]
    f_idle = float(np.percentile(f0[~quiet], 10))
    f_top = float(f0[a0:a1 + 1].max())
    a_ratio = f_top / f_idle
    accel = points(times, f0, a0, a1, f_idle, f_top, args.top_kmh, True)
    decel, d_span = [], None
    if downs:
        d0, d1, _ = max(downs, key=lambda r: (r[2], r[1] - r[0]))
        decel = points(times, f0, d0, d1, f_idle, f_top, args.top_kmh, False)
        d_span = (d0, d1)
    plat = (near(args.plateau[0]), near(args.plateau[1])) if args.plateau else \
        plateau(np.where(quiet, 0.0, f0), 0.85 * f_top)

    if args.enhance:
        clean_dir = Path(__file__).parent / "sounds" / "enhanced"
        clean_dir.mkdir(parents=True, exist_ok=True)
        clean = enhance(src, times, f0, clean_dir / f"{src.stem}.wav", cut_db=args.enhance)
        print(f"enhanced copy: {clean} (inter-harmonic noise -{args.enhance:.0f} dB)")
        src = clean
    MAPS_DIR.mkdir(exist_ok=True)
    out = MAPS_DIR / f"{src.stem}.json"
    data = {"file": str(src), "name": args.name or f"Recording: {src.stem}",
            "accel": accel, "decel": decel, "pulse_until": 0.0, "hum_from": 1.5,
            "automap": {"band": list(args.band), "idle_hz": round(f_idle, 1),
                        "top_hz": round(f_top, 1), "top_kmh": args.top_kmh}}
    if plat:
        data["plateau"] = [round(float(times[plat[0]]), 3), round(float(times[plat[1]]), 3)]
    if args.samples:
        sm = [s for s in steady(times, f0, min_sec=args.min_sec)
              if (args.max_hz is None or s[2] <= args.max_hz)
              and loud(near(s[0]), near(s[1]))]
        sm = [r for r in (refine(x, sr, s) for s in sm) if r]
        if len(sm) < 2:
            raise SystemExit("fewer than two steady stretches found -- try another --band")
        data["samples"] = [[round(a, 3), round(b, 3), round(f, 2)] for a, b, f in sm]
        data["idle_hz"] = round(sm[0][2], 2)
        data["hz_per_kmh"] = round((sm[-1][2] - sm[0][2]) / args.top_kmh, 4)
        print("samples:", ", ".join(f"{f:.0f} Hz @ {a:.2f}-{b:.2f}s" for a, b, f in sm))
    out.write_text(json.dumps(data, indent=1))

    # picture: full-range spectrogram (40 Hz - 4 kHz) with the track and passages
    t_img, b_img, s_img = log_spectrogram(load(src, 16000), 16000, 40.0, 4000.0, nb=420)
    f0_img = np.interp(t_img, times, f0)
    idx = lambda t: int(np.argmin(np.abs(t_img - t)))
    marks = [((idx(times[a0]), idx(times[a1])), (60, 220, 120))]
    if d_span:
        marks.append(((idx(times[d_span[0]]), idx(times[d_span[1]])), (255, 160, 70)))
    if plat:
        marks.append(((idx(times[plat[0]]), idx(times[plat[1]])), (255, 255, 255)))
    write_png(out.with_suffix(".png"), t_img, b_img, s_img, f0_img, marks)

    print(f"pitch: idle {f_idle:.0f} Hz -> top {f_top:.0f} Hz ({a_ratio:.2f}x) = 0 -> {args.top_kmh:.0f} km/h")
    print(f"accel  {times[a0]:.2f}-{times[a1]:.2f} s, {len(accel)} points")
    if d_span:
        print(f"decel  {times[d_span[0]]:.2f}-{times[d_span[1]]:.2f} s, {len(decel)} points")
    if plat:
        print(f"plateau {data['plateau'][0]:.2f}-{data['plateau'][1]:.2f} s")
    print(f"wrote {out} and {out.with_suffix('.png')} (green = accel, orange = decel, white = plateau)")


if __name__ == "__main__":
    main()
