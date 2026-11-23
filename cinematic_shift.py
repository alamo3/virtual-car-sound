"""Cinematic gearshift one-shots for Open Blue: three takes to choose from.

    .venv/bin/python cinematic_shift.py <out dir>

  A  impact:    a quick rising swell into a deep pitched hit, short rumbling tail
  B  braam:     a short, dark brass-like chord stab over a sub thump
  C  metal hit: the impact plus a low metallic ring
Each as shift_<take>_up.wav / _down.wav (down: lower, a touch darker). Harmonics kept in
150 Hz - 3 kHz so small speakers carry them; nothing whistly.
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

SR = 48000
rng = np.random.default_rng(5)


def env(n, attack, tau):
    t = np.arange(n) / SR
    return np.minimum(1, t / max(attack, 1e-4)) * np.exp(-np.maximum(0, t - attack) / tau)


def bandnoise(n, lo, hi):
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    X *= (f >= lo) & (f <= hi)
    x = np.fft.irfft(X, n)
    return x / (np.abs(x).max() + 1e-9)


def lowpass(x, fc):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X /= np.sqrt(1 + (f / fc) ** 4)
    return np.fft.irfft(X, len(x))


def glide(n, f0, f1, harmonics=6, curve=3.0):
    """A pitched tone gliding f0 -> f1 (exponential), with decaying harmonics."""
    t = np.linspace(0, 1, n)
    f = f1 + (f0 - f1) * np.exp(-curve * t)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return sum(np.sin(k * ph) * 0.7 ** (k - 1) for k in range(1, harmonics + 1))


def swell(n):
    """Reversed-cymbal style rise: band noise growing into the hit."""
    t = np.linspace(0, 1, n)
    return bandnoise(n, 400, 2500) * t ** 3


def impact(up: bool):
    n = int(0.75 * SR)
    hit = np.tanh(2.2 * glide(n, 170 if up else 140, 55 if up else 45) * env(n, 0.003, 0.16))
    rumble = lowpass(bandnoise(n, 40, 600), 300) * env(n, 0.01, 0.25) * 0.5
    return hit + rumble


def braam(up: bool):
    n = int(0.6 * SR)
    root = 98.0 if up else 82.4                     # G2 / E2
    chord = (1.0, 1.5, 2.0, 3.0)                   # root, fifth, octave, twelfth
    t = np.arange(n) / SR
    x = np.zeros(n)
    for i, r in enumerate(chord):
        ph = 2 * np.pi * root * r * t * (1 + 0.004 * (i - 1.5))   # slight detune: width
        x += sum(np.sin(k * ph) / k for k in range(1, 9)) * (0.9 ** i)
    # brassy: a filter that opens on the attack and closes as it fades
    X = np.fft.rfft(x * env(n, 0.015, 0.22))
    f = np.fft.rfftfreq(n, 1 / SR)
    X /= np.sqrt(1 + (f / 1400) ** 4)
    x = np.fft.irfft(X, n)
    sub = np.sin(2 * np.pi * np.cumsum(np.linspace(90, 45, n)) / SR) * env(n, 0.002, 0.12)
    return np.tanh(1.6 * x / np.abs(x).max()) + 0.7 * sub


def metal(up: bool):
    x = impact(up)
    n = len(x)
    t = np.arange(n) / SR
    base = 260.0 if up else 220.0
    ring = sum(a * np.sin(2 * np.pi * base * r * t + rng.uniform(0, 6)) * env(n, 0.002, d)
               for r, a, d in ((1.0, 0.5, 0.35), (2.76, 0.3, 0.22), (4.2, 0.15, 0.12)))
    return x + 0.45 * ring


def finish(x, lead):
    if lead:
        s = swell(int(0.12 * SR))
        x = np.concatenate([0.35 * s, x])
    fade = int(0.05 * SR)
    x[-fade:] *= np.linspace(1, 0, fade)
    return 0.85 * x / np.abs(x).max()


def write(path: Path, x: np.ndarray):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).round().astype("<i2").tobytes())


TAKES = {"A": (impact, True), "B": (braam, False), "C": (metal, True)}


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".").expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for take, (fn, lead) in TAKES.items():
        for up in (True, False):
            x = finish(fn(up), lead)
            write(out / f"shift_{take}_{'up' if up else 'down'}.wav", x)
        P = np.abs(np.fft.rfft(x)) ** 2
        f = np.fft.rfftfreq(len(x), 1 / SR)
        print(f"{take}: {len(x) / SR:.2f} s, {100 * P[(f >= 150) & (f <= 3000)].sum() / P.sum():.0f}% in 150 Hz - 3 kHz, "
              f"above 5 kHz {100 * P[f > 5000].sum() / P.sum():.1f}%")


if __name__ == "__main__":
    main()
