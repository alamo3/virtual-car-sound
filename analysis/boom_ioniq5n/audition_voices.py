"""Offline scripted audition of the BOOM voices; no audio device required.

Run from the project root: .venv/bin/python analysis/boom_ioniq5n/audition_voices.py
"""
from pathlib import Path
import sys
import wave

import numpy as np
from signalflow import AudioGraph, AudioGraphConfig

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from boom_voices import BoomEvolution, BoomSupersonic
from voices import State


def inputs(t):
    # Low, acceleration, hold, release, then a brief redline and lift.
    if t < 2:
        return 0.12, 0.22
    if t < 9:
        return 0.12 + (t - 2) / 7 * 0.70, 0.75
    if t < 12:
        return 0.82, 0.45
    if t < 15:
        return 0.82 - (t - 12) / 3 * 0.35, 0.08
    if t < 19:
        return 0.47 + (t - 15) / 4 * 0.52, 0.9
    if t < 21:
        return 0.99, 0.9
    return 0.72, 0.12


def render(cls):
    cfg = AudioGraphConfig()
    cfg.backend_name = "null"
    graph = AudioGraph(cfg, start=False)
    sr, block = int(graph.sample_rate), 480
    voice = cls()
    voice.output.play()
    blocks = []
    saw_limiter = False
    for start in range(0, sr * 24, block):
        n = min(block, sr * 24 - start)
        tach, pedal = inputs(start / sr)
        status = voice.update(State(motor_ratio=tach, pedal=pedal, speed_kmh=100 * tach,
                                    swell=0, active=True, dt=n / sr, virtual_tach=tach))
        saw_limiter |= "limiter" in status
        graph.render(n)
        blocks.append(np.array(voice.output.output_buffer)[:, :n].copy())
    audio = np.concatenate(blocks, axis=1).T
    assert audio.shape == (sr * 24, 2) and np.isfinite(audio).all()
    assert 0.01 < np.max(np.abs(audio)) < 1.0 and saw_limiter
    # A stale vehicle input must silence the voice and disarm the limiter.
    voice.update(State(motor_ratio=1, pedal=1, speed_kmh=100, swell=0,
                       active=False, dt=.02, virtual_tach=1))
    assert not voice._limiting and voice.targets["gain"].value == 0
    # Without virtual gears, ordinary high motor RPM never engages the limiter.
    voice.update(State(motor_ratio=1, pedal=1, speed_kmh=100, swell=0,
                       active=True, dt=.02))
    assert not voice._limiting
    path = ROOT / "demos" / ("boom_evolution_scripted.wav" if cls is BoomEvolution
                             else "boom_supersonic_scripted.wav")
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    print(f"{cls.name}: {path}, peak {np.max(np.abs(audio)):.3f}")
    graph.destroy()


if __name__ == "__main__":
    for voice_class in (BoomEvolution, BoomSupersonic):
        render(voice_class)
