"""Experimental two-state voices using a locally supplied IONIQ recording.

The source is a video mix and the two excerpts are not labelled pedal states.
These voices deliberately hold playback rate fixed while testing the transition.
"""
from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np
from signalflow import Buffer, BufferPlayer, Smooth

import voices


REFERENCE = Path(__file__).with_name("아이오닉5N, N active sound plus 리뷰.mp3")
SAMPLE_RATE = 48000


def _load_loops(start, intervals):
    """Decode just the four-second reference and build the auditioned loops in RAM."""
    decoded = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(start), "-t", "4", "-i", str(REFERENCE),
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"],
        check=True, stdout=subprocess.PIPE,
    ).stdout
    x = np.frombuffer(decoded, dtype="<f4").astype(np.float64)
    if len(x) < 3.9 * SAMPLE_RATE:
        raise ValueError(f"Reference recording is too short for {start}s section")
    fade = int(0.08 * SAMPLE_RATE)
    buffers = []
    for first, last in intervals:
        segment = x[round((first - start) * SAMPLE_RATE):round((last - start) * SAMPLE_RATE)]
        if len(segment) <= 2 * fade:
            raise ValueError("Reference loop is too short")
        phase = np.linspace(0.0, 1.0, fade, endpoint=False)
        crossfade = segment[-fade:] * (1.0 - phase) + segment[:fade] * phase
        cycle = np.concatenate((segment[fade:-fade], crossfade))
        rms = float(np.sqrt(np.mean(cycle * cycle)))
        if rms < 1e-5:
            raise ValueError("Reference loop is silent")
        cycle *= min(2.0, 0.12 / rms)
        buf = Buffer(1, len(cycle), [cycle.tolist()])
        buf.sample_rate = SAMPLE_RATE
        buffers.append(buf)
    return buffers


class ReferenceVoice(voices.Voice):
    """Crossfade two naturally timed loops in response to pedal demand."""

    START = 0
    INTERVALS = ()

    def __init__(self):
        self.load = 0.0
        self._loops = _load_loops(self.START, self.INTERVALS)
        super().__init__()

    def build(self):
        targets = {"blend": voices._target(0.0), "gain": voices._target(0.0)}
        blend = Smooth(targets["blend"], 0.9997)
        first = BufferPlayer(self._loops[0], loop=1)
        second = BufferPlayer(self._loops[1], loop=1)
        return (first * (1.0 - blend) + second * blend) * Smooth(targets["gain"], 0.999), targets

    def update(self, st):
        pedal = min(1.0, max(0.0, st.pedal))
        target = min(1.0, max(0.0, (pedal - 0.2) / 0.6))
        tau = 0.18 if target > self.load else 0.40
        self.load += (target - self.load) * (1.0 - math.exp(-max(0.0, st.dt) / tau))
        self.targets["blend"].set_value(self.load)
        self.targets["gain"].set_value((1.0 + 0.25 * pedal) if st.active else 0.0)
        return f"recording blend {self.load:.0%}"


class EvolutionReference(ReferenceVoice):
    name = "Reference: Evolution"
    START = 175
    INTERVALS = ((175.86, 176.58), (177.59, 178.59))


class SupersonicReference(ReferenceVoice):
    name = "Reference: Supersonic"
    START = 241
    INTERVALS = ((241.54, 242.26), (243.79, 244.41))
