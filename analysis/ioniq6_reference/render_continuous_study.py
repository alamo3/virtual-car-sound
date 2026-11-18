#!/usr/bin/env python3
"""Render whole-reference time-stretch studies, not a live driving voice.

Each output is a single pass through the input. FFmpeg's time stretch keeps the
nominal pitch fixed; it may introduce artifacts and does not add driving states.
"""
from pathlib import Path
import argparse
import subprocess
import wave
import numpy as np

HERE = Path(__file__).resolve().parent


def decode_filtered(path, audio_filter):
    cmd = ['ffmpeg', '-v', 'error', '-i', str(path), '-af', audio_filter,
           '-ac', '2', '-ar', '48000', '-f', 'f32le', 'pipe:1']
    result = subprocess.run(cmd, stdout=subprocess.PIPE, check=True)
    raw = np.frombuffer(result.stdout, dtype='<f4').astype(np.float64)
    if len(raw) % 2 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError('Invalid audio from ffmpeg')
    return raw.reshape(-1, 2)


def write_wav(path, samples):
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype('<i2').tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=HERE / 'hyundai_evolution_launch.wav')
    parser.add_argument('--output-dir', type=Path, default=HERE)
    args = parser.parse_args()
    versions = [('evolution_slow_rubberband.wav', 'rubberband=tempo=0.75:channels=together'),
                ('evolution_slow_atempo.wav', 'atempo=0.75'),
                ('evolution_fast_rubberband.wav', 'rubberband=tempo=1.25:channels=together')]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, _ in versions:
        path = args.output_dir / filename
        if path.exists():
            raise SystemExit(f'Output already exists: {path}; choose another output directory')
    rendered = [(filename, decode_filtered(args.source, f)) for filename, f in versions]
    # A common attenuation keeps comparisons level-consistent and prevents clipping.
    peak = max(float(np.max(np.abs(samples))) for _, samples in rendered)
    gain = min(1.0, 0.95 / peak)
    for filename, samples in rendered:
        output = args.output_dir / filename
        write_wav(output, samples * gain)
        print(f'{output}: {len(samples)/48000:.2f}s, peak {np.max(np.abs(samples*gain)):.3f}, common gain {gain:.3f}')


if __name__ == '__main__':
    main()
