#!/usr/bin/env python3
"""Audition short, fixed-position grains from the user's IONIQ 5 N recording.

This is a source-based listening study, not an isolated OEM sample or a finished voice.
The source remains in the project root; no original source audio is redistributed here.
"""
from pathlib import Path
import argparse
import subprocess
import sys
import tempfile
import wave
import numpy as np
from signalflow import AudioGraph, AudioGraphConfig

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import voices


def render(source, start, destination):
    with tempfile.TemporaryDirectory(prefix='ioniq-texture-') as temp:
        sample = Path(temp) / 'segment.wav'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', str(start), '-t', '4',
                        '-i', str(source), '-ar', '48000', '-ac', '1', str(sample)], check=True)
        cfg = AudioGraphConfig()
        cfg.backend_name = 'null'
        graph = AudioGraph(cfg, start=False)
        sr = int(graph.sample_rate)
        voice = voices.RecordingVoice(str(sample), name='Texture study')
        voice.output.play()
        # Hold the same position and playback rate. This does not map pedal to RPM.
        voice.audition = 1.25
        block = 480
        chunks = []
        for n in range(0, 12 * sr, block):
            t = n / sr
            pedal = 0.2 if t < 4 or t >= 8 else 0.8
            state = voices.State(motor_ratio=.5, pedal=pedal, speed_kmh=30,
                                 swell=0, active=True, dt=block / sr)
            voice.update(state)
            graph.render(block)
            chunks.append(np.array(voice.output.output_buffer)[0, :block].copy())
        graph.destroy()
    x = np.concatenate(chunks)
    if not np.isfinite(x).all() or np.max(np.abs(x)) > 1:
        raise RuntimeError('Invalid or clipped output')
    with wave.open(str(destination), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x * 32767).astype('<i2').tobytes())
    print(destination, 'peak', round(float(np.max(np.abs(x))), 3),
          'rms', round(float(np.sqrt(np.mean(x*x))), 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'demos')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for label, start in [('evolution', 175), ('supersonic', 241)]:
        dest = args.output_dir / f'ioniq_reference_{label}_grain_study.wav'
        if dest.exists():
            raise SystemExit(f'File exists: {dest}')
        render(args.source.resolve(), start, dest)


if __name__ == '__main__':
    main()
