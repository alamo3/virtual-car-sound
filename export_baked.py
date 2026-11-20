"""Pre-render ("bake") synthesizer voices into loop sets for Open Blue.

    .venv/bin/python export_baked.py ~/open-blue-android/app/src/main/assets/loopsets

The cinematic voices (and the Jetsons i3 recording voice) are live signalflow patches on
the Pi. Rather than rewrite each in Kotlin, this renders each one -- with the Pi's own
code -- at a grid of steady states: SPEEDS km/h x {cruise, full pedal}, LOOP_SEC each,
made seamless. Open Blue's LoopVoice then crossfades them by speed and pedal.

Between two speeds the pitch is interpolated, not just crossfaded: the frequency ratio
between neighbouring renders is measured (log-spectrum alignment) and stored as each
loop's "ref", so the player stretches both loops onto the same pitch while blending.

Output per voice: <id>/set.json + mono 16-bit WAVs at SR. set.json has
  "key": "speed", "speeds": [...], "on": full-pedal loops, "light": cruise loops,
  "tone": false, "raw_level": true, "coast_is_light": true, "sample_rate": SR
"""
from __future__ import annotations

import json
import re
import sys
import wave
from pathlib import Path

import numpy as np

SR = 24000                 # these voices have little above 10 kHz: half the size of 48k
GRAPH_SR = 48000
SPEEDS = [2, 8, 15, 25, 35, 50, 65, 80, 100, 125, 150]
LOADS = {"light": 0.15, "on": 1.0}       # cruise, full pedal
WARM_SEC = 2.5
LOOP_SEC = 4.0
XFADE_SEC = 0.25
VOICES = ["Cinematic Drive", "Cinematic Chord", "Action Pulse", "Trailer Brass",
          "Strings & Choir", "Cathedral", "Grand Touring", "Dreadnought",
          "Jetsons i3 (recording)"]
# chord / tempo based voices: speed changes the chord and tempo, not one sliding pitch, so
# the spectral pitch estimate is meaningless -- plain crossfade between speeds instead
NO_PITCH = {"Action Pulse", "Trailer Brass", "Strings & Choir"}


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def seamless(x: np.ndarray, n: int) -> np.ndarray:
    """Crossfade the last n samples into the first n (equal power) and drop the tail."""
    r = np.sin(np.linspace(0, np.pi / 2, n)) ** 2
    y = x[:-n].copy()
    y[:n] = x[:n] * r + x[-n:] * (1 - r)
    return y


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    n2 = int(round(len(x) * sr_out / sr_in))
    X = np.fft.rfft(x)
    Y = np.zeros(n2 // 2 + 1, dtype=complex)
    k = min(len(X), len(Y))
    Y[:k] = X[:k]
    return np.fft.irfft(Y, n2) * (n2 / len(x))


def log_spectrum(x: np.ndarray, sr: int, bins_per_oct: int = 48):
    n = 1 << 14
    P = np.mean([np.abs(np.fft.rfft(x[i:i + n] * np.hanning(n))) ** 2
                 for i in range(0, max(1, len(x) - n), n // 2)], axis=0)
    fr = np.fft.rfftfreq(n, 1 / sr)
    grid = 60.0 * 2 ** (np.arange(int(np.log2(4000 / 60) * bins_per_oct)) / bins_per_oct)
    L = np.log(np.interp(grid, fr, P) + 1e-12)
    return L - L.mean()


def pitch_ratio(a: np.ndarray, b: np.ndarray, sr: int, bpo: int = 48) -> float:
    """How much higher b's spectrum sits than a's (frequency ratio), from aligning their
    log-frequency spectra. 1.0 when there's no clear shift (noise-like sounds)."""
    La, Lb = log_spectrum(a, sr, bpo), log_spectrum(b, sr, bpo)
    best, best_k = -1e9, 0
    for k in range(-bpo // 2, bpo + 1):           # -1/2 .. +1 octave
        if k >= 0:
            c = np.dot(La[:len(La) - k], Lb[k:]) / (len(La) - k)
        else:
            c = np.dot(La[-k:], Lb[:len(Lb) + k]) / (len(La) + k)
        if c > best:
            best, best_k = c, k
    flat = np.dot(La, Lb) / len(La)
    if best < flat * 1.05 + 1e-9:                 # no clear alignment: don't shift
        return 1.0
    return float(2 ** (best_k / bpo))


def render_states(name: str) -> dict:
    """{(load_key, speed): mono float array at GRAPH_SR}, rendered with the Pi's code."""
    from signalflow import AudioGraph, AudioGraphConfig
    cfg = AudioGraphConfig()
    cfg.backend_name = "null"
    cfg.output_buffer_size = 480
    g = AudioGraph(cfg)
    g.stop()
    import main
    import voices
    v = main.VOICES[name]()
    v.set_small_speaker(False)
    v.output.play()
    dt = 480 / GRAPH_SR
    out = {}

    def run(sec, st, keep):
        blocks = []
        for _ in range(int(sec / dt)):
            v.update(st)
            b = g.render_to_new_buffer(480)
            if keep:
                blocks.append(np.array(b.data[0], dtype=np.float64))
        return np.concatenate(blocks) if keep else None

    for key, pedal in LOADS.items():
        for kmh in SPEEDS:
            rpm = kmh * main.RPM_PER_KMH
            st = voices.State(motor_ratio=min(1.0, rpm / main.CAR_RPM_MAX), pedal=pedal,
                              speed_kmh=float(kmh), swell=0.0, active=True, dt=dt)
            run(WARM_SEC, st, False)
            out[(key, kmh)] = run(LOOP_SEC + XFADE_SEC, st, True)
    v.output.stop()
    return out


def write(path: Path, x: np.ndarray):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).round().astype("<i2").tobytes())


def export(name: str, out: Path):
    sid = slug(name)
    d = out / sid
    d.mkdir(parents=True, exist_ok=True)
    states = render_states(name)
    loops = {k: seamless(resample(x, GRAPH_SR, SR), int(XFADE_SEC * SR)) for k, x in states.items()}
    info = {"name": name, "key": "speed", "speeds": SPEEDS, "cylinders": 2, "idle_rpm": 0,
            "redline_rpm": 1, "sample_rate": SR, "tone": False, "raw_level": True,
            "coast_is_light": True, "off": [], "shift_up": None, "shift_down": None, "pops": []}
    for key in LOADS:
        refs = [1.0]
        for a, b in zip(SPEEDS, SPEEDS[1:]):          # cumulative pitch from speed to speed
            r = 1.0 if name in NO_PITCH else \
                float(np.clip(pitch_ratio(loops[(key, a)], loops[(key, b)], SR), 0.8, 1.5))
            refs.append(refs[-1] * r)
        entries = []
        for kmh, ref in zip(SPEEDS, refs):
            f = f"{key}_{kmh:03d}.wav"
            write(d / f, loops[(key, kmh)])
            entries.append({"file": f, "ref_hz": ref})
        info[key] = entries
    (d / "set.json").write_text(json.dumps(info, indent=1))
    size = sum(p.stat().st_size for p in d.iterdir()) / 1e6
    pitch_span = info["on"][-1]["ref_hz"] / info["on"][0]["ref_hz"]
    peak = max(float(np.abs(x).max()) for x in loops.values())
    print(f"{name}: {len(loops)} loops, pitch x{pitch_span:.2f} over {SPEEDS[0]}-{SPEEDS[-1]} km/h, "
          f"peak {peak:.2f}, {size:.1f} MB", flush=True)
    return sid


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "loopsets").expanduser()
    only = sys.argv[2:] or VOICES
    out.mkdir(parents=True, exist_ok=True)
    for name in only:
        export(name, out)


if __name__ == "__main__":
    main()
