#!/usr/bin/env python3
"""Render one offline drive sketch from a single, continuous reference.

This is an audition, not a vehicle-controlled voice. The reference only contains a
launch: cruising and lift-off are approximated by time stretch and volume, and the
source continues moving forward throughout. No loop or jump is introduced.
"""
from pathlib import Path
import argparse
import subprocess
import wave
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RATE = 48000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=HERE / 'hyundai_evolution_launch.wav')
    parser.add_argument('--output', type=Path, default=ROOT / 'demos/evolution_continuous_drive_sketch.wav')
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'Output already exists: {args.output}')
    # Commands are in source time, not car telemetry. All rates are <= 1 because
    # faster stretching sounded worse in the user's earlier audition.
    filt = ("asendcmd=c='1.8 rubberband tempo 0.4;3.0 rubberband tempo 0.65;"
            "4.2 rubberband tempo 0.8',rubberband=tempo=0.8:channels=together")
    cmd = ['ffmpeg', '-v', 'error', '-i', str(args.source), '-af', filt,
           '-ac', '2', '-ar', str(RATE), '-f', 'f32le', 'pipe:1']
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    samples = np.frombuffer(raw, '<f4').astype(np.float64)
    if len(samples) % 2 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError('Invalid rendered audio')
    samples = samples.reshape(-1, 2)
    times = np.arange(len(samples)) / RATE
    # Approximate output-time phases: acceleration, cruise, lift, acceleration.
    # Linear ramps prevent gain clicks. They do not represent recorded torque.
    points = [0, 0.2, 2.25, 2.8, 5.25, 5.8, 6.7, 7.1, 7.7, 11.5, len(samples)/RATE]
    levels = [0, 1, 1, .74, .74, .38, .38, .70, 1, 1, .7]
    gain = np.interp(times, points, levels)
    samples *= gain[:, None]
    peak = float(np.max(np.abs(samples)))
    headroom = min(1.0, 0.90 / max(peak, 1e-12))
    samples *= headroom
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.output), 'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(RATE)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype('<i2').tobytes())
    print(f'{args.output}: {len(samples)/RATE:.2f} s, peak {np.max(np.abs(samples)):.3f}, headroom {headroom:.3f}')


if __name__ == '__main__':
    main()
