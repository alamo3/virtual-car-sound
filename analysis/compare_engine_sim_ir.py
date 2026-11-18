"""Apply Engine Simulator's impulse responses to our V8 demo for a controlled audition.

Usage: .venv/bin/python analysis/compare_engine_sim_ir.py /path/to/engine-sim
This is an offline filter comparison, not a render from Engine Simulator.
"""
from pathlib import Path
import argparse
import wave
import numpy as np


def read_mono(path):
    with wave.open(str(path), 'rb') as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError(f'Expected mono 16-bit PCM: {path}')
        sr = w.getframerate()
        return np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float64) / 32768, sr


def write_mono(path, x, sr):
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype('<i2').tobytes())


def rms(x):
    return float(np.sqrt(np.mean(x * x)))


def apply_ir(x, sr, path):
    ir, ir_sr = read_mono(path)
    # Engine Simulator's synthesizer caps the IR at 10,000 samples.
    ir = ir[:10000]
    ir = np.interp(np.arange(round(len(ir) * sr / ir_sr)) / sr,
                   np.arange(len(ir)) / ir_sr, ir)
    length = len(x) + len(ir) - 1
    fft_n = 1 << (length - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, fft_n) * np.fft.rfft(ir, fft_n), fft_n)[:len(x)]
    if rms(y) < 1e-10:
        raise ValueError(f'Empty impulse response: {path}')
    return y * (rms(x) / rms(y))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('engine_sim', type=Path)
    args = parser.parse_args()
    source, sr = read_mono(Path('demos/v8_muscle.wav'))
    root = args.engine_sim / 'es/sound-library'
    clips = {'baseline': source,
             'smooth': apply_ir(source, sr, root / 'smooth/smooth_39.wav'),
             'mild_exhaust': apply_ir(source, sr, root / 'new/mild_exhaust.wav')}
    # One common attenuation preserves the RMS match and keeps every version unclipped.
    peak = max(float(np.max(np.abs(x))) for x in clips.values())
    gain = min(1.0, 0.95 / peak)
    for name, x in clips.items():
        x *= gain
        path = Path('demos') / f'v8_ir_comparison_{name}.wav'
        write_mono(path, x, sr)
        print(f'{path}: RMS {rms(x):.3f}; peak {np.max(np.abs(x)):.3f}')


if __name__ == '__main__':
    main()
