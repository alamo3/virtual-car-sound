#!/usr/bin/env python3
"""Inspect the official reference mixes and a generated demo, without altering audio."""
from pathlib import Path
import wave
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
FILES=[('Evolution (Hyundai)',ROOT/'analysis/ioniq6_reference/hyundai_evolution_launch.wav'),
       ('Lightspeed (Hyundai)',ROOT/'analysis/ioniq6_reference/hyundai_lightspeed_launch.wav'),
       ('Hypercar Response (our demo)',ROOT/'demos/hypercar_response_drive.wav')]


def measure(path):
    with wave.open(str(path),'rb') as w:
        sr=w.getframerate();channels=w.getnchannels()
        if w.getsampwidth()!=2: raise ValueError('Expected 16-bit PCM')
        raw=np.frombuffer(w.readframes(w.getnframes()),'<i2').astype(np.float64).reshape(-1,channels)/32768
    # Average channel powers, not waveforms: this avoids phase cancellation in stereo.
    n=2048;hop=480
    frames=np.lib.stride_tricks.sliding_window_view(raw,n,axis=0)[::hop]
    frames=frames[:int(8*sr/hop)] * np.hanning(n)
    power=np.abs(np.fft.rfft(frames,axis=-1))**2
    power=power.mean(axis=1)
    freq=np.fft.rfftfreq(n,1/sr); times=(np.arange(len(frames))*hop+n/2)/sr
    energies=power.sum(axis=1)
    bands=np.stack([power[:,(freq>=lo)&(freq<hi)].sum(axis=1) for lo,hi in [(40,180),(180,500),(500,1500),(1500,5000)]])
    bands=bands/np.maximum(energies,1e-20)
    rms=np.sqrt(np.mean(frames**2,axis=(1,2)) / np.mean(np.hanning(n)**2))
    return freq,times,power,bands,rms,raw,sr


def main():
    data=[(label,measure(path)) for label,path in FILES]
    fig,axes=plt.subplots(3,2,figsize=(14,10),constrained_layout=True)
    for row,(label,(f,t,p,b,rms,raw,sr)) in enumerate(data):
        ax,ax2=axes[row];mask=(f>=80)&(f<=5000)
        db=10*np.log10(p[:,mask]+1e-12);db-=db.max()
        mesh=ax.pcolormesh(t,f[mask],db.T,shading='auto',cmap='magma',vmin=-65,vmax=0,rasterized=True)
        ax.set_yscale('log');ax.set_ylim(80,5000);ax.set_yticks([100,200,500,1000,2000,5000]);ax.set_yticklabels(['100','200','500','1k','2k','5k']);ax.set_title(label);ax.set_ylabel('Frequency (Hz)');ax.set_xlabel('Seconds')
        labels=('<180 Hz','180–500 Hz','500–1500 Hz','1.5–5 kHz')
        for i in range(4): ax2.plot(t,b[i],label=labels[i],lw=1.2)
        ax2.set_ylim(0,1);ax2.set_xlim(t[0],t[-1]);ax2.set_xlabel('Seconds');ax2.set_ylabel('Share of spectral energy');ax2.set_title('Energy distribution, normalized per moment')
        if row==0: ax2.legend(loc='upper right',fontsize=8)
        mean=b[:,(t>=1)&(t<=7)].mean(axis=1)
        print(label,'average share 1–7 s:',', '.join(f'{x:.0%}' for x in mean), 'RMS range:',f'{np.percentile(rms,10):.3f}–{np.percentile(rms,90):.3f}')
    fig.suptitle('Reference recordings and existing demo: first 8 seconds\nDifferent drive scripts; this is not a matched driving comparison',fontsize=13)
    out=ROOT/'analysis/ioniq6_reference/launch_comparison.png';fig.savefig(out,dpi=145);print('wrote',out)

if __name__=='__main__':main()
