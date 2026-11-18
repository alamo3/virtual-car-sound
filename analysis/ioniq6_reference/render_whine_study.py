#!/usr/bin/env python3
"""Compare two conservative EQ changes to the slow continuous reference.

The frequency causing the reported whine is not yet known. These are exploratory
A/Bs, not automatic source separation and not a live driving implementation.
"""
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'evolution_slow_rubberband.wav'
OPTIONS = (
    ('evolution_slow_soft_top.wav', 'treble=f=1700:g=-5'),
    ('evolution_slow_mid_notch.wav', 'equalizer=f=1125:t=q:w=1.5:g=-5'),
)

for name, audio_filter in OPTIONS:
    out = HERE / name
    if out.exists():
        raise SystemExit(f'Output already exists: {out}')
    subprocess.run(['ffmpeg', '-v', 'error', '-n', '-i', str(SOURCE), '-af', audio_filter,
                    '-c:a', 'pcm_s16le', str(out)], check=True)
    print(out)
