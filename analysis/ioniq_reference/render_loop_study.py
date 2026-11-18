#!/usr/bin/env python3
"""Audition longer crossfaded loops from two user-specified reference sections.

Unlike the grain study, most of each segment plays through unchanged. These are
reference-based experiments, not isolated OEM sounds or production-ready loops.
"""
from pathlib import Path
import argparse
import subprocess
import tempfile
import wave
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def make_loop(source, start, output):
    with tempfile.TemporaryDirectory(prefix='ioniq-loop-') as tmp:
        wav = Path(tmp) / 'excerpt.wav'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', str(start), '-t', '4',
                        '-i', str(source), '-ar', '48000', '-ac', '1', str(wav)], check=True)
        with wave.open(str(wav), 'rb') as w:
            sr = w.getframerate()
            x = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float64) / 32768

    # Compare short regions on either side of each proposed overlap. This gives us
    # a seam candidate; it cannot assess artistic suitability or repeated motion.
    step = sr // 50
    fade = int(sr * .08)
    length_min, length_max = int(sr * .60), int(sr * 1.10)
    best = None
    for left in range(int(sr*.15), len(x)-length_max-fade, step):
        a = x[left:left+fade]
        ar = np.sqrt(np.mean(a*a)) + 1e-9
        for length in range(length_min, length_max+1, step):
            right = left + length - fade
            b = x[right:right+fade]
            br = np.sqrt(np.mean(b*b)) + 1e-9
            corr = np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b)+1e-12)
            score = corr - .2 * abs(np.log(ar/br))
            if best is None or score > best[0]:
                best = (score, left, left+length)
    _, left, right = best
    seg = x[left:right]
    phase = np.linspace(0, 1, fade, endpoint=False)
    # Starts after the overlap; ends with the final part blended into the start.
    blend = seg[-fade:] * (1-phase) + seg[:fade] * phase
    cycle = np.concatenate((seg[fade:-fade], blend))
    out = np.tile(cycle, int(np.ceil(8*sr/len(cycle))))[:8*sr]
    ramp = int(.15*sr)
    out[:ramp] *= np.linspace(0,1,ramp)
    out[-ramp:] *= np.linspace(1,0,ramp)
    peak = float(np.max(np.abs(out)))
    if peak > .85: out *= .85/peak
    with wave.open(str(output), 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(out,-1,1)*32767).astype('<i2').tobytes())
    print(f'{output.name}: source {start+left/sr:.2f}-{start+right/sr:.2f}s, cycle {len(cycle)/sr:.2f}s, peak {np.max(np.abs(out)):.3f}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('--output-dir', type=Path, default=ROOT/'demos')
    args = p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for name,start in [('evolution',175),('supersonic',241)]:
        dest=args.output_dir/f'ioniq_reference_{name}_loop_study.wav'
        if dest.exists(): raise SystemExit(f'File exists: {dest}')
        make_loop(args.source.resolve(),start,dest)

if __name__ == '__main__': main()
