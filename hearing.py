"""How a sound will come across on the USB speaker -- a check before anyone listens.

    .venv/bin/python hearing.py demos/a.wav demos/b.wav          # as heard on the USB speaker
    .venv/bin/python hearing.py --flat demos/a.wav               # the file itself

The speaker's response was measured with a sine sweep (analysis/speaker/): it plays
almost nothing below ~120 Hz and has a +6 dB bump at 200 Hz. Every number below is
computed on the sound *after* that response.

  sharpness   approximate DIN 45692 sharpness (acum), from Bark-band loudness: how
              shrill / whiny vs warm a sound is. Car interiors sit around 1.0-1.5;
              above ~1.8 people start calling it whiny or thin.
  low/mid/high  share of loudness below 300 Hz / 300 Hz-1.5 kHz / above 1.5 kHz.
              "Body" lives in low; a sound that is mostly mid reads as a distant whine.
  tonality    how much of the loudness sits in narrow peaks (tones) rather than
              broadband: high values read as whistle / whine.
  lost        how much level the speaker throws away (content below its range).
The measures are approximations (no masking or level-dependent thresholds), meant for
comparing versions of a sound, not as certified figures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RESPONSE = HERE / "analysis" / "speaker" / "usb_speaker_response.json"
SR = 48000
# critical-band (Bark) edges, Hz
BARK_EDGES = [20, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320,
              2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500]


def load(path: str) -> np.ndarray:
    import automap
    return automap.load(path, SR)


def speaker_gain_db(freqs: np.ndarray) -> np.ndarray:
    r = json.loads(RESPONSE.read_text())
    hz, db = np.array(r["hz"]), np.array(r["db"])
    g = np.interp(np.log(np.maximum(freqs, 1.0)), np.log(hz), db, left=db[0], right=db[-1])
    return np.minimum(g, 6.0)


def power_spectrum(x: np.ndarray, n: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    w = np.hanning(n)
    frames = [np.abs(np.fft.rfft(x[i:i + n] * w)) ** 2 for i in range(0, len(x) - n, n // 2)]
    return np.fft.rfftfreq(n, 1 / SR), np.mean(frames, axis=0) if frames else np.zeros(n // 2 + 1)


def analyse(x: np.ndarray, as_heard: bool = True) -> dict:
    fr, P = power_spectrum(x)
    total_in = P.sum()
    if as_heard:
        P = P * 10 ** (speaker_gain_db(fr) / 10)
    # loudness per Bark band (Zwicker's 0.23 power law; relative units)
    z = np.arange(len(BARK_EDGES) - 1) + 0.5
    Pb = np.array([P[(fr >= a) & (fr < b)].sum() for a, b in zip(BARK_EDGES[:-1], BARK_EDGES[1:])])
    N = (Pb / (Pb.max() + 1e-30) + 1e-9) ** 0.23
    N = np.maximum(N - (1e-6) ** 0.23, 0.0)            # crude threshold: ~60 dB below the peak
    g = np.where(z < 15.8, 1.0, 0.066 * np.exp(0.171 * z))
    sharp = 0.11 * float((N * g * z).sum() / (N.sum() + 1e-12))
    # loudness shares
    c = np.sqrt(np.array(BARK_EDGES[:-1]) * np.array(BARK_EDGES[1:]))
    low, mid, high = (float(N[m].sum() / N.sum()) for m in (c < 300, (c >= 300) & (c < 1500), c >= 1500))
    # tonality: per 0.17 s frame, the share of energy (100 Hz - 5 kHz) in bins standing
    # >10 dB above their local median, averaged -- per frame, so moving tones still count
    n = 8192
    fr2 = np.fft.rfftfreq(n, 1 / SR)
    band = (fr2 > 100) & (fr2 < 5000)
    gain = 10 ** (speaker_gain_db(fr2) / 10) if as_heard else 1.0
    shares = []
    for i in range(0, len(x) - n, n):
        Pf = (np.abs(np.fft.rfft(x[i:i + n] * np.hanning(n))) ** 2 * gain)[band]
        if Pf.sum() < 1e-12:
            continue
        Pd = 10 * np.log10(Pf + 1e-30)
        med = np.array([np.median(Pd[max(0, j - 31):j + 31]) for j in range(0, len(Pd))])
        shares.append(Pf[Pd > med + 10].sum() / Pf.sum())
    tonal = float(np.mean(shares)) if shares else 0.0
    return {"sharpness": round(sharp, 2), "low": round(low, 2), "mid": round(mid, 2),
            "high": round(high, 2), "tonality": round(tonal, 2),
            "lost_db": round(10 * np.log10(P.sum() / (total_in + 1e-30) + 1e-30), 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--flat", action="store_true", help="the file itself, not as heard")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    a = ap.parse_args()
    print(f"{'file':42s} sharp   low  mid  high  tonal  lost")
    for f in a.files:
        x = load(f)
        x = x[int(a.start * SR):int(a.end * SR) if a.end else None]
        r = analyse(x, not a.flat)
        print(f"{Path(f).name[:42]:42s} {r['sharpness']:5.2f}  {r['low']:.2f} {r['mid']:.2f} "
              f"{r['high']:.2f}  {r['tonality']:.2f}  {r['lost_db']:+.1f} dB")


if __name__ == "__main__":
    main()
