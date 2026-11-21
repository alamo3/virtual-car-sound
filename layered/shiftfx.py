"""The regular (hydraulic) shift sound, as Open Blue plays it: its two takes level-matched,
the "down" take on upshifts and the "up" take on downshifts, and only on shifts made while
driving hard (DriveEffects.kt)."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

import dsp

ASSETS = Path(__file__).resolve().parents[2] / "open-blue-android" / "app" / "src" / "main" / "assets" / "effects"


def _load(name):
    with wave.open(str(ASSETS / f"{name}.wav")) as w:
        assert w.getframerate() == dsp.SR, "shift takes are 48 kHz"
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(float) / 32768
        return x.reshape(-1, w.getnchannels()) if w.getnchannels() == 2 else np.repeat(x[:, None], 2, 1)


def takes():
    up, down = _load("shift_cin_down"), _load("shift_cin_up")      # crossed on purpose, as the app
    rms = [np.sqrt((t ** 2).mean()) for t in (up, down)]
    target = max(rms)
    return up * target / rms[0], down * target / rms[1]


FROM_PEDAL, FULL_PEDAL = 0.55, 0.85      # as DriveEffects.kt: only when driven hard


def effort_level(pedal: float) -> float:
    x = min(1.0, max(0.0, (pedal - FROM_PEDAL) / (FULL_PEDAL - FROM_PEDAL)))
    return x * x * (3 - 2 * x)


def mix(out: np.ndarray, c: dict, level: float = 0.5) -> np.ndarray:
    """Adds the shift sound at the gearbox's shifts made while driving hard: none below 55 %
    pedal, full by 85 % (c: control signals at audio rate)."""
    up, down = takes()
    n = len(out)
    for counter, take in (("shifts", up), ("downshifts", down)):
        for i in dsp.events(c[counter]):
            g = effort_level(c["pedal"][i])
            if g <= 0:
                continue
            m = min(len(take), n - i)
            out[i:i + m] += take[:m] * level * g
    return out
