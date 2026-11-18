"""Experimental IONIQ voices from locally supplied BOOM interior recordings.

The audio has no RPM or throttle labels: the regions below are provisional,
chosen from the recorded spectral contours, and need listening-based tuning.
"""
from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
from signalflow import Buffer, BufferPlayer, SineOscillator, Smooth, SVFilter

import voices

SOURCE_DIR = Path(__file__).resolve().parent / "analysis/boom_ioniq5n/opposite_decode"


def _loop(path: Path, start: float, end: float) -> Buffer:
    """Read a stereo PCM region and overlap its ends to make a continuous loop."""
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 2 or wf.getsampwidth() != 2:
            raise ValueError(f"Expected stereo 16-bit PCM: {path}")
        sr = wf.getframerate()
        wf.setpos(round(start * sr))
        frames = wf.readframes(round((end - start) * sr))
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32).reshape(-1, 2) / 32768.0
    fade = round(0.12 * sr)
    if len(audio) <= 2 * fade:
        raise ValueError(f"Invalid loop region {start}..{end} in {path}")
    mix = np.linspace(0, 1, fade, endpoint=False, dtype=np.float32)[:, None]
    seam = audio[-fade:] * (1 - mix) + audio[:fade] * mix
    audio = np.concatenate((audio[fade:-fade], seam), axis=0)
    # Match typical region levels, but limit amplification and retain headroom.
    rms = float(np.sqrt(np.mean(audio * audio)))
    peak = float(np.max(np.abs(audio)))
    if rms < 1e-5 or peak < 1e-5:
        raise ValueError(f"Silent loop region in {path}")
    audio *= min(3.0, 0.10 / rms, 0.8 / peak)
    buf = Buffer(2, len(audio), [audio[:, 0].tolist(), audio[:, 1].tolist()])
    buf.sample_rate = sr
    return buf


class BoomVoice(voices.Voice):
    """Three overlapping source regions, plus a provisional limiter modulation."""

    REGIONS = ()

    @classmethod
    def available(cls):
        return all((SOURCE_DIR / filename).is_file() for filename, _, _ in cls.REGIONS)

    def __init__(self):
        self._buffers = [_loop(SOURCE_DIR / f, start, end) for f, start, end in self.REGIONS]
        self._tach = 0.0
        self._pedal = 0.0
        self._limiting = False
        super().__init__()

    def build(self):
        t = {name: voices._target(value) for name, value in (
            ("low", 1.0), ("mid", 0.0), ("high", 0.0), ("rate", 1.0),
            ("cutoff", 2500.0), ("gain", 0.0), ("limiter", 0.0))}
        rate = Smooth(t["rate"], 0.9997)
        players = [BufferPlayer(buf, rate=rate, loop=1) for buf in self._buffers]
        self._players = players
        signal = sum(player * Smooth(t[weight], 0.9995)
                     for player, weight in zip(players, ("low", "mid", "high")))
        signal = SVFilter(signal, "low_pass", cutoff=Smooth(t["cutoff"], 0.9995), resonance=0.0)
        # A subtle rhythmic cut on the sampled sound: an approximation, not an
        # identified recording of Hyundai's actual limiter.
        pulse = (SineOscillator(8.0) + 1.0) * 0.5
        limiter = Smooth(t["limiter"], 0.9995)
        signal *= 1.0 - limiter * 0.42 * pulse
        return signal * Smooth(t["gain"], 0.9995), t

    def update(self, st):
        dt = min(0.2, max(0.0, st.dt))
        target_tach = st.virtual_tach if st.virtual_tach is not None else st.motor_ratio
        target_tach = max(0.0, min(1.0, target_tach))
        pedal = max(0.0, min(1.0, st.pedal))
        self._tach += (target_tach - self._tach) * (1.0 - math.exp(-dt / 0.12))
        self._pedal += (pedal - self._pedal) * (1.0 - math.exp(-dt / 0.10))
        # Adjacent regions crossfade; mild playback-rate movement avoids asking
        # one static recording to span the full range.
        u = self._tach * 2.0
        low, high = max(0.0, 1.0 - u), max(0.0, u - 1.0)
        for name, value in (("low", low), ("mid", 1.0 - low - high), ("high", high)):
            self.targets[name].set_value(value)
        self.targets["rate"].set_value(0.93 + 0.14 * self._tach)
        self.targets["cutoff"].set_value(1800.0 + 3200.0 * self._pedal)
        self.targets["gain"].set_value((0.55 + 0.45 * self._pedal) if st.active else 0.0)
        # Only enable the limiter with virtual gears, above redline and under
        # substantial pedal. Hysteresis stops it chattering at the threshold.
        if st.virtual_tach is None or not st.active or pedal < 0.5:
            self._limiting = False
        elif self._limiting:
            self._limiting = target_tach >= 0.93
        else:
            self._limiting = target_tach >= 0.97
        self.targets["limiter"].set_value(1.0 if self._limiting else 0.0)
        return f"sample tach {target_tach:.0%}" + (" | limiter" if self._limiting else "")


class BoomEvolution(BoomVoice):
    name = "BOOM: Evolution"
    REGIONS = (("evolution_rev.wav", 0.25, 1.80),
               ("evolution_drive_02.wav", 2.35, 3.65),
               ("evolution_drive_02.wav", 4.40, 5.75))


class BoomSupersonic(BoomVoice):
    name = "BOOM: Supersonic"
    REGIONS = (("supersonic_rev.wav", 0.20, 1.45),
               ("supersonic_rev.wav", 4.60, 6.00),
               ("supersonic_rev.wav", 7.20, 8.65))
