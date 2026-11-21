"""Render the layered voices over the standard test drive: WAVs to listen to, and a picture of
each (spectrogram over the drive's revs, speed and pedal).

    python layered/render.py [out_dir] [voice ...]
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np

import drive
import dsp

VOICES = ["flat6", "hypercar", "jet", "warp"]


def render_voice(name: str, out_dir: Path):
    mod = importlib.import_module(name)
    ctrl = drive.run(idle=getattr(mod, "IDLE", 900.0), redline=getattr(mod, "REDLINE", 7000.0))
    c = drive.to_audio_rate(ctrl, dsp.SR)
    audio = mod.render(c)
    wav = out_dir / f"{name}_test_drive.wav"
    dsp.write_wav(wav, audio)
    picture(name, audio, ctrl, out_dir / f"{name}_test_drive.png")
    return wav


def picture(name, audio, ctrl, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mono = audio.mean(axis=1)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, constrained_layout=True,
                                 gridspec_kw=dict(height_ratios=[3, 1.3]))
    a1.specgram(mono, NFFT=4096, Fs=dsp.SR, noverlap=3072, cmap="magma", vmin=-110)
    a1.set_ylim(0, 6000)
    a1.set_ylabel("Hz")
    a1.set_title(f"{name}: spectrogram over the standard test drive", loc="left", fontsize=10)
    t = ctrl["t"]
    a2.plot(t, ctrl["rpm"], color="#2a6fdb", lw=1.2, label="gearbox rpm")
    a2.set_ylabel("rpm")
    b = a2.twinx()
    b.plot(t, ctrl["kmh"], color="#555", lw=1, label="km/h")
    b.plot(t, ctrl["pedal"] * 100, color="#e07b00", lw=1, ls="--", label="pedal %")
    b.set_ylabel("km/h · %")
    a2.set_xlabel("seconds")
    h1, l1 = a2.get_legend_handles_labels()
    h2, l2 = b.get_legend_handles_labels()
    a2.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, ncol=3)
    fig.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("layered/out")
    out.mkdir(parents=True, exist_ok=True)
    for v in (sys.argv[2:] or VOICES):
        print("rendered", render_voice(v, out))
