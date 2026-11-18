#!/usr/bin/env python3
"""Test switching between two longer snippets of a supplied recording.

Experimental offline audition only. The two snippets are earlier/later source
positions, not validated pedal or speed states. No pitch shifting is used.
"""
from pathlib import Path
import argparse
import subprocess
import tempfile
import wave
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def find_loop(x, sr, begin, end):
    fade = int(.08 * sr)
    step = sr // 50
    best = None
    for left in range(int(begin*sr), int(end*sr)-int(.60*sr), step):
        a = x[left:left+fade]
        ar = np.sqrt(np.mean(a*a))+1e-9
        for length in range(int(.60*sr), min(int(1.05*sr),int(end*sr)-left)+1,step):
            b=x[left+length-fade:left+length]
            br=np.sqrt(np.mean(b*b))+1e-9
            score=np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12)-.2*abs(np.log(ar/br))
            if best is None or score>best[0]: best=(score,left,left+length)
    if best is None: raise ValueError('No loop candidate')
    _,left,right=best
    seg=x[left:right]
    p=np.linspace(0,1,fade,endpoint=False)
    loop=np.concatenate((seg[fade:-fade],seg[-fade:]*(1-p)+seg[:fade]*p))
    return loop, (left/sr,right/sr)


def run_one(source,start,dest):
    with tempfile.TemporaryDirectory(prefix='ioniq-states-') as temp:
        wav=Path(temp)/'source.wav'
        subprocess.run(['ffmpeg','-v','error','-y','-ss',str(start),'-t','4','-i',str(source),
                        '-ar','48000','-ac','1',str(wav)],check=True)
        with wave.open(str(wav),'rb') as w:
            sr=w.getframerate(); x=np.frombuffer(w.readframes(w.getnframes()),'<i2').astype(float)/32768
    loops=[]
    for a,b in [(.12,1.75),(2.05,3.90)]:
        y,interval=find_loop(x,sr,a,b)
        rms=np.sqrt(np.mean(y*y))
        if rms < 1e-5: raise ValueError('Silent source selection')
        # Equal RMS keeps the change in texture audible without automatic gain jumps.
        loops.append((y*min(2.0,.12/rms),interval))
    total=16*sr
    a=np.resize(loops[0][0],total);b=np.resize(loops[1][0],total)
    # Change state every four seconds, over 500ms; no playback-rate modulation.
    t=np.arange(total)/sr
    w=np.clip((t-3.75)/.5,0,1)-np.clip((t-7.75)/.5,0,1)+np.clip((t-11.75)/.5,0,1)
    out=a*(1-w)+b*w
    ramp=int(.15*sr);out[:ramp]*=np.linspace(0,1,ramp);out[-ramp:]*=np.linspace(1,0,ramp)
    peak=np.max(np.abs(out))
    if peak>.9: out*=.9/peak
    with wave.open(str(dest),'wb') as wfile:
        wfile.setnchannels(1);wfile.setsampwidth(2);wfile.setframerate(sr)
        wfile.writeframes((out*32767).astype('<i2').tobytes())
    print(dest.name, 'source intervals:',[(round(start+a,2),round(start+b,2)) for _,(a,b) in loops], 'peak',round(float(np.max(np.abs(out))),3))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('--output-dir',type=Path,default=ROOT/'demos')
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    for name,start in [('evolution',175),('supersonic',241)]:
        dest=args.output_dir/f'ioniq_{name}_two_state_study.wav'
        if dest.exists():raise SystemExit(f'File exists: {dest}')
        run_one(args.source.resolve(),start,dest)

if __name__=='__main__':main()
