"""Synthesize the hydraulic gearshift one-shots Open Blue plays on every shift.

    .venv/bin/python shift_sound.py ~/open-blue-android/app/src/main/assets/effects

Writes shift_hyd_up.wav and shift_hyd_down.wav (mono 16-bit 48 kHz, ~0.3 s): an
electro-hydraulic sequential gearbox -- the valve's hiss, the dog ring's clunk, a short
actuator whirr, and the second clunk as the gear engages. Mid-frequency on purpose (most
energy 150 Hz - 4 kHz), so a small speaker plays it and it never turns into a whistle.
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

SR = 48000


def bandnoise(n, lo, hi, rng):
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    X *= (f >= lo) & (f <= hi)
    x = np.fft.irfft(X, n)
    return x / (np.abs(x).max() + 1e-9)


def decay(n, tau, attack=0.002):
    t = np.arange(n) / SR
    return np.minimum(1.0, t / attack) * np.exp(-t / tau)


def clunk(n, f0, tau, rng, bright=1.0):
    """Metal on metal: a low thump, two body resonances and a click."""
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * f0 * t) * decay(n, tau)
    x += 0.5 * np.sin(2 * np.pi * f0 * 2.07 * t) * decay(n, tau * 0.6)
    for fr, a in ((1150, 0.35), (2350, 0.2)):
        x += bright * a * np.sin(2 * np.pi * fr * t + rng.uniform(0, 6)) * decay(n, 0.012)
    x += bright * 0.4 * bandnoise(n, 800, 5000, rng) * decay(n, 0.004, attack=0.0005)
    return x


def shift(up: bool, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(0.32 * SR)
    out = np.zeros(n)

    def add(x, at, gain):
        i = int(at * SR)
        out[i:i + len(x)] += gain * x[:n - i]

    # valve: a short burst of hydraulic pressure (hiss, kept under ~4 kHz)
    add(bandnoise(int(0.12 * SR), 900, 3800, rng) * decay(int(0.12 * SR), 0.035, 0.003), 0.0, 0.45)
    # dog ring disengages
    add(clunk(int(0.12 * SR), 200 if up else 180, 0.035, rng), 0.012, 0.8)
    # actuator whirr: a short pitch glide (down on upshifts, up on downshifts)
    m = int(0.11 * SR)
    t = np.arange(m) / SR
    f = np.linspace(420, 260, m) if up else np.linspace(260, 420, m)
    ph = 2 * np.pi * np.cumsum(f) / SR
    whirr = sum(np.sin(k * ph) / k for k in range(1, 7))
    add(whirr * np.sin(np.pi * t / t[-1]) ** 2, 0.03, 0.35)
    # the gear engages: second, slightly duller clunk
    add(clunk(int(0.15 * SR), 220 if up else 165, 0.05, rng, bright=0.7), 0.105, 0.6)
    out = np.tanh(2.6 * out / np.abs(out).max())   # grit and density: audible over an engine
    fade = int(0.02 * SR)
    out[-fade:] *= np.linspace(1, 0, fade)
    return 0.85 * out / np.abs(out).max()


def write(path: Path, x: np.ndarray):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).round().astype("<i2").tobytes())


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "effects").expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for name, up, seed in (("shift_hyd_up", True, 3), ("shift_hyd_down", False, 7)):
        x = shift(up, seed)
        write(out / f"{name}.wav", x)
        P = np.abs(np.fft.rfft(x)) ** 2
        f = np.fft.rfftfreq(len(x), 1 / SR)
        mid = P[(f >= 150) & (f <= 4000)].sum() / P.sum()
        print(f"{name}: {len(x) / SR:.2f} s, {100 * mid:.0f}% of energy in 150 Hz - 4 kHz, "
              f"above 6 kHz {100 * P[f > 6000].sum() / P.sum():.1f}%")


if __name__ == "__main__":
    main()
