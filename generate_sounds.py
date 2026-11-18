"""Generate source recordings with Stable Audio Open 1.0 on Replicate.

    export REPLICATE_API_TOKEN=r8_...
    .venv/bin/python generate_sounds.py                 # the whole jet set, 3 takes each
    .venv/bin/python generate_sounds.py --only idle cruise --takes 2
    .venv/bin/python generate_sounds.py --list

Instead of synthesizing from oscillators, a voice plays real-sounding recordings: steady
loops at several thrust levels, blended and pitch-shifted by the spool / gearbox
(the way game engines do vehicle sound). This makes those recordings.

Stable Audio Open was trained largely on Freesound, so the prompts are written like
Freesound descriptions; the negative prompt keeps it away from music. Every take uses a
fixed seed, so a set can be regenerated exactly. Files go to
sounds/generated/<set>/<state>__s<seed>.wav; existing files are skipped (no re-billing).
Replicate allows 6 predictions a minute on low credit: runs are spaced 11 s apart.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

MODEL = ("stackadoc/stable-audio-open-1.0:"
         "9aff84a639f96d0f7e6081cdea002d15133d0043727f849c40abdd166b7c75a8")
HERE = Path(__file__).parent
OUT = HERE / "sounds" / "generated"
MIN_GAP_SEC = 11.0
RETRIES = 5

NEGATIVE = ("music, melody, song, chords, drums, beat, rhythm, synthesizer, vocals, speech, "
            "talking, flyby, doppler, passing by, wind gusts, birds, traffic")

# state: (seconds, prompt). Steady states become loops; spool clips are transitions.
SETS = {
    "jet": {
        "idle": (12, "Fighter jet engine idling, turbine whine and low rumble, steady, "
                     "constant, recorded inside the cockpit, close, no movement"),
        "cruise": (12, "Fighter jet engine at steady mid throttle, turbine roar and whine, "
                       "constant, recorded inside the cockpit, close, no movement"),
        "high": (12, "Fighter jet engine at full military power, loud deep turbine roar, "
                     "steady, constant, recorded inside the cockpit, close"),
        "afterburner": (12, "Fighter jet afterburner, massive deep roar with crackle, steady, "
                            "constant, recorded inside the cockpit, close"),
        "spool_up": (8, "Fighter jet engine spooling up from idle to full power, turbine "
                        "whine rising, recorded inside the cockpit, close"),
        "spool_down": (8, "Fighter jet engine spooling down from full power to idle, turbine "
                          "whine falling, recorded inside the cockpit, close"),
    },
    # futuristic electric car: sound design, not a recording of a real machine. Kept
    # deep and smooth -- the negative prompt rules out whine, beeps and music.
    "ev": {
        "_negative": ("music, melody, song, chords, drums, beat, rhythm, vocals, speech, "
                      "high pitched whine, squeal, whistle, beeps, alarm, siren, gasoline engine, "
                      "exhaust, combustion, motorcycle, flyby, doppler, traffic, distortion"),
        "_cfg": 5.0,
        "idle": (12, "Futuristic electric car idling, deep smooth low hum, soft energy "
                     "pulse, sci-fi vehicle, steady, constant, interior, clean"),
        "cruise": (12, "Futuristic electric car cruising, deep powerful smooth electric hum, "
                       "rich low tones, sci-fi vehicle drive, steady, constant, interior, clean"),
        "power": (12, "Futuristic electric sports car at full power, deep rich powerful "
                      "electric drive, low growl and energy, sci-fi, steady, constant, "
                      "interior, clean"),
        "accelerate": (8, "Futuristic electric sports car accelerating, deep powerful electric "
                          "drive sound rising smoothly in pitch, sci-fi vehicle, interior, clean"),
        "decelerate": (8, "Futuristic electric car slowing down, deep electric drive sound "
                          "falling smoothly in pitch, sci-fi vehicle, interior, clean"),
    },
    # sci-fi drive: the word "car" pulls Freesound-trained output toward real (often
    # combustion) cars, so this set describes designed sci-fi sounds instead, in the warm
    # mid range a small speaker plays well
    "scifi": {
        "_negative": ("car, automobile, gasoline engine, piston, exhaust, motorcycle, truck, "
                      "traffic, music, melody, song, chords, drums, beat, rhythm, vocals, speech, "
                      "high pitched whine, squeal, whistle, beeps, alarm, siren, sub bass rumble, "
                      "distortion, flyby, doppler"),
        "_cfg": 5.0,
        "drive_hum": (12, "Sci-fi spaceship drive hum, smooth warm mid-range drone with rich "
                          "harmonics, futuristic, steady, constant, close, clean, sound design"),
        "power_core": (12, "Sci-fi power core humming, warm resonant energy drone with slow "
                           "pulsing, futuristic machine, steady, constant, close, clean, sound "
                           "design"),
        "hover": (12, "Futuristic hover vehicle engine, smooth magnetic energy hum with "
                      "shimmering harmonics, sci-fi, steady, constant, close, clean, sound design"),
        "thrust": (12, "Sci-fi starship engine at full thrust, powerful warm energy roar with "
                       "rich harmonics, futuristic, steady, constant, close, clean, sound design"),
        "power_up": (8, "Sci-fi spaceship engine powering up, warm energy hum rising smoothly "
                        "in pitch, futuristic, close, clean, sound design"),
        "power_down": (8, "Sci-fi spaceship engine powering down, warm energy hum falling "
                          "smoothly in pitch, futuristic, close, clean, sound design"),
    },
    # race car, heard onboard: steady clips at several rev levels become loops (blended and
    # pitch-shifted by the virtual gearbox); the two runs are transitions / references
    "race": {
        "_negative": ("crowd, cheering, commentary, announcer, speech, music, melody, drums, "
                      "beat, tire squeal, crash, horn, flyby, doppler, passing by, wind, rain, "
                      "distortion, clipping"),
        "_cfg": 5.0,
        "idle": (12, "Race car engine idling, onboard camera audio, lumpy aggressive idle, "
                     "steady, constant, close, clean"),
        "low_rpm": (12, "Race car engine held at low rpm, onboard camera audio, deep steady "
                        "engine note, constant, close, clean"),
        "mid_rpm": (12, "Race car engine held at steady mid rpm, onboard camera audio, "
                        "powerful constant engine note, close, clean"),
        "high_rpm": (12, "Race car engine held at high rpm, onboard camera audio, loud "
                         "aggressive constant engine note, close, clean"),
        "full_throttle": (12, "Race car accelerating at full throttle through the gears, "
                              "onboard camera audio, fast upshifts, close, clean"),
        "braking": (10, "Race car braking hard into a corner, onboard camera audio, engine "
                        "downshifts with throttle blips and exhaust pops, close, clean"),
    },
}
SEEDS = (11, 23, 47, 71, 97)

_last = 0.0


def predict(inputs: dict):
    global _last
    import replicate
    for attempt in range(RETRIES):
        wait = _last + MIN_GAP_SEC - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last = time.monotonic()
        try:
            # create + poll (rather than run) so the wait is visible: the first call may
            # cold-start the model on Replicate, which takes a few minutes
            pred = replicate.predictions.create(version=MODEL.split(":", 1)[1], input=inputs)
            t0, shown = time.monotonic(), None
            while pred.status not in ("succeeded", "failed", "canceled"):
                if pred.status != shown or int(time.monotonic() - t0) % 30 < 3:
                    print(f"  {pred.status} ({time.monotonic() - t0:.0f} s)"
                          + ("  -- cold start, can take a few minutes" if pred.status == "starting"
                             else ""), flush=True)
                    shown = pred.status
                time.sleep(3)
                pred.reload()
            if pred.status != "succeeded":
                raise RuntimeError(f"prediction {pred.status}: {pred.error}")
            print(f"  done in {time.monotonic() - t0:.0f} s", flush=True)
            return pred.output
        except Exception as e:
            if "throttled" not in str(e).lower() and "429" not in str(e):
                raise
            print(f"  throttled; retrying in {MIN_GAP_SEC * (attempt + 1):.0f} s", flush=True)
            time.sleep(MIN_GAP_SEC * (attempt + 1))
    raise RuntimeError("still throttled after retries")


def fetch(item) -> bytes:
    if hasattr(item, "read"):
        return item.read()
    with urllib.request.urlopen(str(item)) as r:
        return r.read()


def trimmed_copy(path: Path) -> Path:
    """The model always returns a ~47 s file with only the requested seconds filled: write
    <set>/trimmed/<name>.wav without the silence, mono, peak-normalized for listening."""
    import subprocess
    import wave
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ar", "48000", "-ac", "1",
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, "<f4").astype(float)
    e = np.array([np.mean(x[i:i + 2400] ** 2) for i in range(0, len(x), 2400)]) + 1e-15
    on = np.flatnonzero(10 * np.log10(e) > 10 * np.log10(e.max()) - 45)
    y = x[on[0] * 2400:(on[-1] + 1) * 2400]
    y = y / (np.abs(y).max() + 1e-9) * 0.89
    out = path.parent / "trimmed" / (path.stem + ".wav")
    out.parent.mkdir(exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes((y * 32767).astype("<i2").tobytes())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="jet", choices=sorted(SETS))
    ap.add_argument("--only", nargs="+", help="just these states")
    ap.add_argument("--takes", type=int, default=3, help="takes (seeds) per state, max 5")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cfg", type=float, default=None,
                    help="how literally to follow the prompt (set default: jet 7, others 5; "
                         "high values overshoot and clip)")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    states = {k: v for k, v in SETS[a.set].items() if not k.startswith("_")}
    negative = SETS[a.set].get("_negative", NEGATIVE)
    cfg = a.cfg if a.cfg is not None else SETS[a.set].get("_cfg", 7.0)
    if a.list:
        for k, (sec, p) in states.items():
            print(f"{k:12s} {sec:2d} s  {p}")
        return
    if not os.environ.get("REPLICATE_API_TOKEN"):
        sys.exit("set REPLICATE_API_TOKEN first: export REPLICATE_API_TOKEN=r8_...")
    out_dir = OUT / a.set
    out_dir.mkdir(parents=True, exist_ok=True)
    for state, (sec, prompt) in states.items():
        if a.only and state not in a.only:
            continue
        for seed in SEEDS[:a.takes]:
            path = out_dir / f"{state}__s{seed}.wav"
            if path.exists():
                print(f"skip {path.name} (already made)")
                continue
            print(f"{a.set}/{state} seed {seed} ({sec} s) ...", flush=True)
            try:
                out = predict({"prompt": prompt, "negative_prompt": negative,
                               "seconds_total": sec, "steps": a.steps, "cfg_scale": cfg,
                               "seed": seed})
                item = out[0] if isinstance(out, (list, tuple)) else out
                data = fetch(item)
                if data[:4] != b"RIFF":                      # not wav: keep its own format
                    path = path.with_suffix("." + str(item).rsplit(".", 1)[-1][:4])
                path.write_bytes(data)
                print(f"  wrote {path.relative_to(HERE)} ({len(data) // 1024} KB)")
                print(f"  trimmed: {trimmed_copy(path).relative_to(HERE)}")
            except Exception as e:
                print(f"  failed: {e}")


if __name__ == "__main__":
    main()
