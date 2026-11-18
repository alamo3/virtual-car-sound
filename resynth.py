#!/usr/bin/env python3
"""Recreate a recording as a synthesizer that matches its spectrogram.

    .venv/bin/python resynth.py analyse ref.wav --band 55 165 --name "Evolution (synth)"
    .venv/bin/python resynth.py fit "profiles/Evolution (synth).json"   # calibrate + report

Model (measured per pitch and per driving state -- load / coast / idle):
  partials  sine oscillators on half-harmonic steps of the tracked pitch (0.5x, 1x, 1.5x,
            ... ). Generator sounds often carry a sub-octave and in-between partials,
            so whole-number harmonics alone lose most of the body.
  noise     everything else -- the noise between partials, fixed resonances, cabin --
            as noise through a bank of 1/3-octave band filters, each set to the
            recording's measured level in that band.
`analyse` measures the recording frame by frame (20 ms) and tabulates both against pitch.
`fit` re-synthesizes the recording itself -- the synth driven by the recording's own pitch
and state over time -- compares the two spectrograms band by band, corrects the noise
band calibration and repeats, then reports how closely they match.
In the voice, NOISE_MIX scales the whole noise layer: turn it down to strip cabin noise.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

PROFILE_DIR = Path(__file__).parent / "profiles"
SR = 48000
N_FFT, HOP = 8192, 960
STEP = 0.5                                     # partial spacing, in multiples of the pitch
N_PART = 48                                    # 0.5x ... 24x
BANDS = [40.0 * 2 ** (i / 3) for i in range(25)]   # 1/3-octave centres, 40 Hz - 10 kHz
MAX_HZ = 9000.0
STATES = ("load", "coast", "idle")
PEAK_Q = 4.32                                  # 1/3-octave peaking bands
EQ_MIN_HZ = 100.0          # no peaking bands below this: narrow low-frequency biquads go
                           # unstable in 32-bit float; tone + tilt cover the bottom
PEAK_RES = PEAK_Q * math.sqrt(10.0)            # signalflow's "peak" BiquadFilter: Q = res / sqrt(10)
BUTTERWORTH_Q = 0.7071                         # ...but for low/high-pass, resonance IS the Q
SPLIT_HZ, SPLIT_N = 85.0, 4    # below EQ_MIN_HZ the noise comes from its own parallel bands;
                               # the main (graphic EQ) path is high-passed here, N sections


def peak_db(f, f0, gain_db, q=PEAK_Q):
    """Magnitude (dB) of a peaking biquad (RBJ cookbook) -- matches signalflow exactly."""
    a = 10 ** (np.asarray(gain_db) / 40)
    w0 = 2 * np.pi * np.asarray(f0) / SR
    al = np.sin(w0) / (2 * q)
    z = np.exp(-1j * 2 * np.pi * np.asarray(f) / SR)
    c = -2 * np.cos(w0)
    num = (1 + al * a) + c * z + (1 - al * a) * z ** 2
    den = (1 + al / a) + c * z + (1 - al / a) * z ** 2
    return 20 * np.log10(np.abs(num / den))


def lp_db(f, fc, q=0.7071):
    """Magnitude (dB) of a 2-pole low-pass biquad (RBJ), as signalflow's BiquadFilter."""
    w0 = 2 * np.pi * fc / SR
    al = np.sin(w0) / (2 * q)
    c = np.cos(w0)
    z = np.exp(-1j * 2 * np.pi * np.asarray(f) / SR)
    b0 = b2 = (1 - c) / 2
    b1 = 1 - c
    num = b0 + b1 * z + b2 * z ** 2
    den = (1 + al) - 2 * c * z + (1 - al) * z ** 2
    return 20 * np.log10(np.abs(num / den))


def hp_db(f, fc, q=0.7071):
    """Magnitude (dB) of a 2-pole high-pass biquad (RBJ), as signalflow's BiquadFilter."""
    w0 = 2 * np.pi * fc / SR
    al = np.sin(w0) / (2 * q)
    c = np.cos(w0)
    z = np.exp(-1j * 2 * np.pi * np.asarray(f) / SR)
    b0 = b2 = (1 + c) / 2
    b1 = -(1 + c)
    num = b0 + b1 * z + b2 * z ** 2
    den = (1 + al) - 2 * c * z + (1 - al) * z ** 2
    return 20 * np.log10(np.abs(num / den))


def bp_db(f, fc, q=PEAK_Q):
    """Magnitude (dB) of a band-pass biquad (RBJ, 0 dB peak); signalflow: resonance = Q."""
    w0 = 2 * np.pi * fc / SR
    al = np.sin(w0) / (2 * q)
    c = np.cos(w0)
    z = np.exp(-1j * 2 * np.pi * np.asarray(f) / SR)
    num = al - al * z ** 2
    den = (1 + al) - 2 * c * z + (1 - al) * z ** 2
    return 20 * np.log10(np.abs(num / den) + 1e-30)


LOW_SECTIONS = 2


def low_band_db(fc, f):
    """Response of one low band's filter: two 1/3-octave band-pass sections."""
    return LOW_SECTIONS * bp_db(f, fc)


def _nnls(M, y, iters=60):
    """Non-negative x with M x ~= y (small systems; projected Gauss-Seidel)."""
    x = np.maximum(np.linalg.lstsq(M, y, rcond=None)[0], 0.0)
    d = np.einsum("ii->i", M.T @ M)
    for _ in range(iters):
        for i in range(len(x)):
            r = y - M @ x
            x[i] = max(0.0, x[i] + (M[:, i] @ r) / d[i])
    return x


def _body_weight(f):
    """1 below 60 Hz, fading to 0 at 120 Hz (log scale): where BODY_DB applies."""
    f = np.maximum(np.asarray(f, dtype=float), 1.0)
    return np.clip(np.log(120.0 / f) / np.log(2.0), 0.0, 1.0)


def fit_tilt(prof):
    """Corner of the 2-pole low-pass that best matches the average noise slope."""
    d = np.concatenate([np.array(prof[s]["dens"]) for s in STATES])
    m = 10 * np.log10(np.mean(d, axis=0) + 1e-30)
    bands = np.array(prof["bands_hz"])
    best = None
    for fc in np.geomspace(80, 8000, 200):
        r = lp_db(bands, fc)
        off = np.mean(m - r)
        err = np.mean((m - r - off) ** 2)
        if best is None or err < best[0]:
            best = (err, fc)
    return float(best[1])


def solve_peaks(target_db, centres, iters=4):
    """Gains for a cascade of peaking filters so its response at each centre hits the
    target (dB). Responses of a cascade add in dB; neighbours overlap, so iterate."""
    c = np.asarray(centres)
    g = np.array(target_db, dtype=float)
    for _ in range(iters):
        resp = peak_db(c[:, None], c[None, :], g[None, :]).sum(axis=1)
        g = g + 0.85 * (target_db - resp)
        g = np.clip(g, -80.0, 80.0)
    return g


def _band_edges():
    return [(c * 2 ** (-1 / 6), c * 2 ** (1 / 6)) for c in BANDS]


def _sine_power_cal():
    """sum |X|^2 over a partial's window, for a unit-amplitude sine (Hann, N_FFT)."""
    tt = np.arange(N_FFT) / SR
    f = 1000.3
    X = np.abs(np.fft.rfft(np.sin(2 * np.pi * f * tt) * np.hanning(N_FFT)))
    fr = np.fft.rfftfreq(N_FFT, 1 / SR)
    w = max(2.5 * SR / N_FFT, 0.03 * f)
    return float((X[(fr > f - w) & (fr < f + w)] ** 2).sum())


_SINE_CAL = None


def measure_frames(x, f0_of_t):
    """Per STFT frame: time, pitch, partial amplitudes (linear), residual noise density
    per band (mean |X|^2 of bins away from partials)."""
    global _SINE_CAL
    if _SINE_CAL is None:
        _SINE_CAL = _sine_power_cal()
    win = np.hanning(N_FFT)
    fr = np.fft.rfftfreq(N_FFT, 1 / SR)
    band_idx = [np.flatnonzero((fr >= lo) & (fr < hi)) for lo, hi in _band_edges()]
    out = []
    for i in range(0, len(x) - N_FFT, HOP):
        t = (i + N_FFT / 2) / SR
        f0 = float(f0_of_t(t))
        X = np.abs(np.fft.rfft(x[i:i + N_FFT] * win))
        amps = np.zeros(N_PART)
        near = np.zeros(len(fr), dtype=bool)
        for j in range(N_PART):
            fk = (j + 1) * STEP * f0
            if fk > MAX_HZ:
                break
            w = min(0.2 * f0, max(2.5 * SR / N_FFT, 0.03 * fk))
            m = (fr > fk - w) & (fr < fk + w)
            # total power in the partial's window: a partial smeared by a moving pitch
            # spreads over many bins, and its peak bin alone would read far too low
            amps[j] = math.sqrt((X[m] ** 2).sum() / _SINE_CAL)
            near |= m
        P = X ** 2
        dens = np.array([P[b[~near[b]]].mean() if (~near[b]).any() else P[b].mean()
                         for b in band_idx])
        out.append((t, f0, amps, dens))
    return out


WARP_CYCLES, WARP_PTS = 32, 128        # analysis window: 32 pitch cycles, 128 points each
WARP_MAX_HZ = 3200.0                   # warped analysis below this; plain STFT above


def measure_frames_warped(x, f0_of_t, rows):
    """Re-measure partials and the noise below WARP_MAX_HZ in pitch-warped frames.

    Each frame is resampled on a grid uniform in pitch phase, so a sweeping pitch becomes
    constant: every partial lands exactly on one FFT bin (ratio r -> bin r*WARP_CYCLES),
    nothing smears into the gaps, and the gaps hold only true noise. Noise is converted
    to the same units as the plain STFT measurement (equivalent mean |X|^2)."""
    # resampling at ~WARP_PTS x pitch folds anything above half that rate back down:
    # remove it first (brick-wall, just above the warped analysis range)
    X = np.fft.rfft(x)
    X[np.fft.rfftfreq(len(x), 1 / SR) > WARP_MAX_HZ * 1.15] = 0.0
    x = np.fft.irfft(X, len(x))
    t_s = np.arange(len(x)) / SR
    f_s = np.maximum(np.asarray([f0_of_t(t) for t in t_s[::480]]), 1.0)
    f_s = np.interp(t_s, t_s[::480], f_s)
    phase = np.cumsum(f_s) / SR                               # in pitch cycles
    L = WARP_CYCLES * WARP_PTS
    # Blackman-Harris: sidelobes at -92 dB, so a loud partial can't leak into the gaps
    n_ = np.arange(L) / (L - 1)
    w = (0.35875 - 0.48829 * np.cos(2 * np.pi * n_) + 0.14128 * np.cos(4 * np.pi * n_)
         - 0.01168 * np.cos(6 * np.pi * n_))
    sw2 = (w ** 2).sum()
    std_sw2 = (np.hanning(N_FFT) ** 2).sum()
    edges = _band_edges()
    out = []
    for (t, f0, amps_std, dens_std) in rows:
        pc = float(np.interp(t, t_s, phase))
        half = L / 2 / WARP_PTS                                # near the file's ends,
        pc = min(max(pc, phase[0] + half), phase[-1] - half)   # slide the window inward
        grid = pc + (np.arange(L) - L / 2) / WARP_PTS          # phase points
        if grid[0] < phase[0] or grid[-1] > phase[-1]:         # file shorter than a window
            out.append((t, f0, amps_std, dens_std))
            continue
        tt = np.interp(grid, phase, t_s)
        seg = np.interp(tt, t_s, x)
        X = np.fft.rfft(seg * w)
        P = np.abs(X) ** 2
        fs_local = WARP_PTS * f0                               # local sample rate
        amps = amps_std.copy()
        partial_bins = []
        for j in range(N_PART):
            r = (j + 1) * STEP
            k = int(round(r * WARP_CYCLES))
            if r * f0 > WARP_MAX_HZ or k + 1 >= len(P):
                break
            partial_bins.append((j, k))
        mask = np.ones(len(P), dtype=bool)
        for _, k in partial_bins:
            mask[max(0, k - 5):k + 6] = False
        mask[:2] = False
        gap_idx = np.flatnonzero(mask)
        for j, k in partial_bins:
            # power in the main lobe minus the noise that sits under it: a "partial" that
            # is only noise must come out as nothing, or the synth turns noise into tones
            lobe = P[max(0, k - 4):k + 5]
            near = gap_idx[np.abs(gap_idx - k) <= 24]
            floor = float(np.mean(P[near])) if len(near) else 0.0
            excess = lobe.sum() - floor * len(lobe)
            amps[j] = math.sqrt(4.0 * excess / (L * sw2)) if excess > floor * len(lobe) else 0.0
        fr_w = np.arange(len(P)) / WARP_CYCLES * f0
        dens = dens_std.copy()
        got, missing = {}, []
        for b, (lo, hi) in enumerate(edges):
            if hi > WARP_MAX_HZ:
                continue
            m = mask & (fr_w >= lo) & (fr_w < hi)
            if m.any():
                psd = P[m].mean() / (fs_local * sw2)           # per Hz
                got[b] = psd * SR * std_sw2                    # -> plain-STFT units
            else:
                missing.append(b)                              # band fully covered by partials
        for b, v in got.items():
            dens[b] = v
        if got and missing:                                    # fill from measured neighbours
            kb = np.array(sorted(got))
            lv = np.log([got[k] for k in kb])
            for b in missing:
                dens[b] = float(np.exp(np.interp(b, kb, lv)))
        out.append((t, f0, amps, dens))
    return out


def load_span(path, sr, span=None):
    """The source audio, cut to `span` = (start, end) seconds if given."""
    import automap
    x = automap.load(path, sr)
    if span:
        x = x[int(span[0] * sr):int(span[1] * sr) if span[1] else None]
    return x


def track(path, band, whiten=False, span=None):
    """Pitch over time: the strongest continuous line in `band`. With `whiten`, each
    frequency's level over the whole take is subtracted first, so constant lines (cabin
    hum, fans) drop out and a line that moves wins even if it is quieter."""
    import automap
    x16 = load_span(path, 16000, span)
    t, b, s = automap.log_spectrogram(x16, 16000, *band)
    if whiten:
        s = s - np.median(s, axis=0, keepdims=True)
        s = s + np.median(s, axis=1, keepdims=True)
    f = automap.ridge(t, b, s)
    f = np.array([np.median(f[max(0, i - 2):i + 3]) for i in range(len(f))])
    f = np.exp(np.convolve(np.log(f), np.ones(5) / 5, "same"))   # smooth the tracker steps
    f[:2], f[-2:] = f[2], f[-3]
    return t, f


def states(t, f0):
    """load / coast / idle masks per frame, from the pitch slope."""
    lf = np.log(f0)
    slope = np.gradient(np.convolve(lf, np.ones(5) / 5, "same"), t)
    idle_hz = float(np.percentile(f0, 10))
    idle = (np.abs(slope) <= 0.25) & (f0 <= idle_hz * 1.08)
    load = ~idle & (slope > -0.25)
    coast = ~idle & (slope <= -0.25)
    return {"load": load, "coast": coast, "idle": idle}, idle_hz


def analyse(path, band, name, bins=12, top_kmh=100.0, whiten=False, span=None,
            broad=0.0, broad_hz=20.0):
    t_tr, f_tr = track(path, band, whiten, span)
    x = load_span(path, SR, span)
    f_of = lambda t: np.interp(t, t_tr, f_tr)
    rows = measure_frames_warped(x, f_of, measure_frames(x, f_of))
    t = np.array([r[0] for r in rows])
    f0 = np.array([r[1] for r in rows])
    A = np.array([r[2] for r in rows])
    D = np.array([r[3] for r in rows])
    masks, idle_hz = states(t, f0)
    edges = np.geomspace(f0.min(), f0.max() * 1.0001, bins + 1)
    centres = np.sqrt(edges[:-1] * edges[1:])

    def table(mask):
        a_t, d_t, n_t = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = mask & (f0 >= lo) & (f0 < hi)
            n_t.append(int(m.sum()))
            if m.sum() < 2:
                a_t.append(None)
                d_t.append(None)
                continue
            # power average: what a spectrogram (and the ear) integrates
            a_t.append(np.sqrt(np.mean(A[m] ** 2, axis=0)).tolist())
            d_t.append(np.mean(D[m], axis=0).tolist())
        have = [i for i, a in enumerate(a_t) if a is not None]
        if not have:
            return None
        for i in range(len(a_t)):
            if a_t[i] is None:
                j = min(have, key=lambda k: abs(k - i))
                a_t[i], d_t[i] = a_t[j], d_t[j]
        return {"amp": a_t, "dens": d_t, "frames": n_t}

    lf = np.log(f0)
    smooth = np.convolve(lf, np.ones(15) / 15, "same")
    steady = np.abs(np.gradient(smooth, t)) < 0.1
    dev = (lf - smooth)[steady]
    wobble = float(min(0.01, max(0.002, 1.4826 * np.median(np.abs(dev - np.median(dev))))))
    load_t = table(masks["load"])
    prof = {
        "name": name, "source": str(Path(path).resolve()), "band": list(band),
        "whiten": bool(whiten), "span": list(span) if span else None,
        "broad": broad, "broad_hz": broad_hz,
        "pitch_hz": centres.round(3).tolist(), "idle_hz": round(idle_hz, 2),
        "top_hz": round(float(np.percentile(f0, 98)), 2), "top_kmh": top_kmh,
        "wobble": round(wobble, 5), "step": STEP, "bands_hz": [round(b, 2) for b in BANDS],
        "noise_cal": None,
        "load": load_t, "coast": table(masks["coast"]) or load_t,
        "idle": table(masks["idle"]) or load_t,
        "frames": {k: int(v.sum()) for k, v in masks.items()},
    }
    prof["tilt_hz"] = fit_tilt(prof)
    PROFILE_DIR.mkdir(exist_ok=True)
    out = PROFILE_DIR / (name.replace("/", "_") + ".json")
    out.write_text(json.dumps(prof))
    return out, prof


# --- the voice ----------------------------------------------------------------------------

def _voice_class():
    import voices
    from signalflow import (BiquadFilter, Impulse, RandomUniform, SineOscillator, Smooth,
                            WhiteNoise)

    class ResynthVoice(voices.Voice):
        """Partials + band noise, set every tick from the measured profile."""
        LEVEL = 1.0
        NOISE_MIX = 1.0            # 1 = as recorded; turn down to remove cabin noise
        BROAD_MIX = 1.0            # line width as measured; 0 = clean, pure partials
        BODY_DB = 0.0              # extra sub-bass (tones and noise under ~100 Hz)
        PARTIALS = N_PART
        BAND_Q = 0.55              # SVF resonance for the 1/3-octave noise bands

        def __init__(self, profile_path):
            self.path = Path(profile_path)
            self.prof = json.loads(self.path.read_text())
            self.name = self.prof["name"]
            p = self.prof
            self.lp = np.log(np.array(p["pitch_hz"]))
            self.tabs = {k: (np.array(p[k]["amp"]), np.array(p[k]["dens"])) for k in STATES}
            self.cal = np.array(p["noise_cal"]) if p.get("noise_cal") else \
                np.full(len(p["bands_hz"]), 0.004)
            self.eq = np.array(p.get("eq") or [1.0] * len(p["bands_hz"]))
            # partial correction: only ever turns partials down, where they alone would
            # exceed the recording's band (e.g. a fixed hum mistaken for a partial)
            self.peq = np.array(p.get("partial_eq") or [1.0] * len(p["bands_hz"]))
            self.lb = np.log(np.array(p["bands_hz"]))
            self.bands = np.array(p["bands_hz"])
            self.tilt_hz = p.get("tilt_hz") or fit_tilt(p)
            self.tilt_db = lp_db(self.bands, self.tilt_hz) + SPLIT_N * hp_db(self.bands, SPLIT_HZ)
            # each low band's own filter, power-averaged over its band -> level correction
            # low bands overlap a lot, so their levels are solved together:
            # mix[i, j] = power of band j's filter, averaged over band i
            self.low = np.flatnonzero(self.bands < EQ_MIN_HZ)
            ffs = [np.geomspace(c * 2 ** (-1 / 6), c * 2 ** (1 / 6), 32) for c in self.bands[self.low]]
            self.low_mix = np.array([[np.mean(10 ** (low_band_db(cj, ff) / 10))
                                      for cj in self.bands[self.low]] for ff in ffs])
            self.hz_per_kmh = (p["top_hz"] - p["idle_hz"]) / p["top_kmh"]
            self.load = 0.0
            ref = np.median(self.tabs["load"][0].max(axis=1))   # typical strongest partial
            self.norm = 0.3 / ref                                # recording -> synth level
            super().__init__()

        def build(self):
            # fixed seeds: a render is repeatable, so the fit's partials-only and full
            # renders differ only by the noise layer
            seeds = iter(range(1, 10_000))

            def seeded(node):
                node.set_seed(next(seeds))
                return node

            t = {"f": voices._target(self.prof["idle_hz"]), "gain": voices._target(0.0),
                 "noise": voices._target(self.NOISE_MIX)}
            wob = self.prof["wobble"]
            drift = Smooth(seeded(RandomUniform(1.0 - 2 * wob, 1.0 + 2 * wob, clock=Impulse(6.0))),
                           0.9995)
            f = Smooth(t["f"], 0.9995) * drift
            # line width: each partial's amplitude carries a share of slow noise, which
            # spreads it into a band ~2 x BROAD_HZ wide (power unchanged). A few
            # independent noises, dealt out in turn, keep neighbouring partials unrelated.
            broad = min(0.95, float(self.prof.get("broad", 0.0)) * self.BROAD_MIX)
            bw = float(self.prof.get("broad_hz", 30.0))
            mods = []
            if broad > 0:
                scale = math.sqrt(broad) / math.sqrt((1 / 3) * math.pi * bw / SR)
                for _ in range(4):
                    nb = BiquadFilter(seeded(WhiteNoise()), "low_pass", cutoff=bw, resonance=BUTTERWORTH_Q)
                    mods.append(math.sqrt(1 - broad) + nb * scale)
            part = None
            for j in range(self.PARTIALS):
                k = (j + 1) * self.prof["step"]
                t[f"p{j}"] = voices._target(0.0)
                a = Smooth(t[f"p{j}"], 0.998)
                if j < 12:          # shimmer on the strong low partials, as measured
                    spread = 10 ** (min(3.0, 1.0 + 0.25 * j) / 20) - 1.0
                    a = a * Smooth(seeded(RandomUniform(1 - spread, 1 + spread,
                                                 clock=Impulse(2.5 + 0.37 * j))), 0.9997)
                if mods:
                    a = a * mods[j % len(mods)]
                p = SineOscillator(f * k) * a
                part = p if part is None else part + p
            # noise: white noise through a series graphic EQ (one peaking band per 1/3
            # octave). In series the gains multiply, so the steep spectrum of a recording
            # is reachable -- a parallel band-pass bank leaks low bands into high ones.
            # fixed tilt: the recording's average slope, so the EQ only shapes deviations
            # (signalflow: for low/high-pass, resonance IS the Q -- unlike "peak")
            noise = BiquadFilter(seeded(WhiteNoise()), "low_pass", cutoff=self.tilt_hz,
                                 resonance=BUTTERWORTH_Q)
            for b, fc in enumerate(self.prof["bands_hz"]):
                t[f"n{b}"] = voices._target(0.0)
                if fc < EQ_MIN_HZ:
                    continue
                noise = BiquadFilter(noise, "peak", cutoff=fc, resonance=PEAK_RES,
                                     peak_gain=Smooth(t[f"n{b}"], 0.999))
            # outside the bands the EQ is flat, so plain white noise would pass at the
            # average band level: close the range off steeply at both ends
            top = self.prof["bands_hz"][-1] * 2 ** (1 / 6)
            bottom = self.prof["bands_hz"][0] * 2 ** (-1 / 6)
            for _ in range(5):
                noise = BiquadFilter(noise, "low_pass", cutoff=top, resonance=BUTTERWORTH_Q)
                noise = BiquadFilter(noise, "high_pass", cutoff=bottom, resonance=BUTTERWORTH_Q)
            for _ in range(SPLIT_N):
                noise = BiquadFilter(noise, "high_pass", cutoff=SPLIT_HZ, resonance=BUTTERWORTH_Q)
            t["nlev"] = voices._target(0.0)
            noise = noise * Smooth(t["nlev"], 0.999)
            # the bottom bands: each its own noise through a plain band-pass, level set
            # directly (peaking filters this low are unstable in 32-bit float)
            for b, fc in enumerate(self.prof["bands_hz"]):
                if fc >= EQ_MIN_HZ:
                    continue
                band = seeded(WhiteNoise())
                for _ in range(LOW_SECTIONS):          # resonance IS the Q for band-pass
                    band = BiquadFilter(band, "band_pass", cutoff=fc, resonance=PEAK_Q)
                noise = noise + band * Smooth(t[f"n{b}"], 0.999)
            self.raw = part + noise * Smooth(t["noise"], 0.999)
            return self.raw * Smooth(t["gain"], 0.999), t

        def _interp(self, which, lf):
            amp, dens = self.tabs[which]
            x = float(np.clip(lf, self.lp[0], self.lp[-1]))
            i = int(np.clip(np.searchsorted(self.lp, x) - 1, 0, len(self.lp) - 2))
            w = (x - self.lp[i]) / (self.lp[i + 1] - self.lp[i])
            return amp[i] * (1 - w) + amp[i + 1] * w, dens[i] * (1 - w) + dens[i + 1] * w

        def set_controls(self, f, weights):
            """f: pitch (Hz); weights: {state: weight} summing to 1."""
            self.targets["f"].set_value(f)
            lf = math.log(f)
            amp = np.zeros(N_PART)
            dens = np.zeros(len(self.cal))
            for s, w in weights.items():
                if w > 0:
                    a, d = self._interp(s, lf)
                    amp += w * a
                    dens += w * d
            ks = (np.arange(N_PART) + 1) * self.prof["step"] * f
            amp[ks > MAX_HZ] = 0.0
            amp = amp * np.interp(np.log(np.maximum(ks, 1.0)), self.lb, self.peq)
            amp = amp * 10 ** (self.BODY_DB * _body_weight(ks) / 20)
            for j in range(self.PARTIALS):
                self.targets[f"p{j}"].set_value(float(amp[j] * self.norm))
            # band level targets (dB re white noise), then the EQ gains that produce them
            tgt = 10 * np.log10(dens + 1e-30) + 20 * np.log10(self.cal * self.eq * self.norm)
            dev = tgt - self.tilt_db                       # what the tilt doesn't cover
            eqb = self.bands >= EQ_MIN_HZ
            level = float(np.mean(dev[eqb]))
            gains = np.zeros(len(dev))
            gains[eqb] = solve_peaks(dev[eqb] - level, self.bands[eqb])
            # low bands: linear levels for their own noises, solved jointly (non-negative)
            want = 10 ** (tgt[self.low] / 10) - 10 ** ((self.tilt_db[self.low] + level) / 10)
            gains[self.low] = np.sqrt(_nnls(self.low_mix, np.maximum(want, 0.0)))
            gains[self.low] *= 10 ** (self.BODY_DB * _body_weight(self.bands[self.low]) / 20)
            for b, v in enumerate(gains):
                self.targets[f"n{b}"].set_value(float(v))
            self.targets["nlev"].set_value(10 ** (level / 20))

        def update(self, st):
            f = (self.prof["idle_hz"] + self.hz_per_kmh * max(0.0, st.speed_kmh)) * st.pitch_mul
            target = min(1.0, st.pedal + 0.5 * st.swell)
            tau = 0.15 if target > self.load else 0.4
            self.load += (target - self.load) * min(1.0, st.dt / tau)
            crawl = max(0.0, 1.0 - st.speed_kmh / 8.0) * (1.0 - self.load)
            w = {"idle": crawl, "load": (1 - crawl) * self.load,
                 "coast": (1 - crawl) * (1 - self.load)}
            self.set_controls(f, w)
            self.targets["gain"].set_value(self.LEVEL if st.active else 0.0)
            state = "idle" if crawl > 0.5 else ("load" if self.load > 0.5 else "coast")
            return f"{f:4.0f} Hz  {state} {self.load * 100:3.0f}%"

    return ResynthVoice


_CLS = None


def voice_class():
    global _CLS
    if _CLS is None:
        _CLS = _voice_class()
    return _CLS


def profiles():
    if not PROFILE_DIR.is_dir():
        return []
    out = []
    for p in sorted(PROFILE_DIR.glob("*.json")):
        try:
            out.append((json.loads(p.read_text())["name"], p))
        except (OSError, ValueError, KeyError):
            pass
    return out


# --- copy synthesis: does the synth reproduce the recording? -------------------------------

def band_spectrogram(x, frame=0.05):
    """1/3-octave band levels (dB) per frame. Bands under 200 Hz are only a few Hz wide,
    so they are measured with a longer FFT (same frame centres)."""
    h = int(frame * SR)
    specs = []
    for n in (4096, 32768):
        win = np.hanning(n)
        fr = np.fft.rfftfreq(n, 1 / SR)
        idx = [np.flatnonzero((fr >= lo) & (fr < hi)) for lo, hi in _band_edges()]
        xp = np.pad(x, (n // 2, n // 2))
        rows = []
        for c in range(4096 // 2, len(x) - 4096 // 2, h):
            seg = xp[c:c + n]                       # centred on sample c of x
            P = np.abs(np.fft.rfft(seg * win)) ** 2 / (win ** 2).sum()
            rows.append([10 * np.log10(P[b].sum() * fr[1] + 1e-20) for b in idx])
        specs.append(np.array(rows))
    low = np.array(BANDS) < 200.0
    return np.where(low[None, :], specs[1], specs[0])


def copy_synth(profile_path, noise=True):
    """Render the synth driven by the recording's own pitch and states, frame by frame."""
    import automap
    from signalflow import AudioGraph, AudioGraphConfig
    prof = json.loads(Path(profile_path).read_text())
    t_tr, f_tr = track(prof["source"], prof["band"], prof.get("whiten", False),
                       prof.get("span"))
    x = load_span(prof["source"], SR, prof.get("span"))
    rows = measure_frames(x, lambda t: np.interp(t, t_tr, f_tr))
    t = np.array([r[0] for r in rows])
    f0 = np.array([r[1] for r in rows])
    masks, _ = states(t, f0)
    g = AudioGraph.get_shared_graph()
    if g is None:
        cfg = AudioGraphConfig()
        cfg.backend_name = "null"
        cfg.output_buffer_size = HOP
        g = AudioGraph(cfg)
    g.stop()
    v = voice_class()(profile_path)
    if not noise:
        v.targets["noise"].set_value(0.0)
    v.raw.play()
    lead = int(N_FFT / 2 / HOP)          # first frame's centre is N_FFT/2 into the file
    sched = [rows[0]] * lead + rows
    smask = {s: np.concatenate([[masks[s][0]] * lead, masks[s]]) for s in STATES}
    out = []
    for i, r in enumerate(sched):
        v.set_controls(r[1], {s: float(smask[s][i]) for s in STATES})
        b = g.render_to_new_buffer(HOP)
        out.append(b.data[0].copy())
    v.raw.stop()
    y = np.concatenate(out).astype(np.float64)[:len(x)]
    return x, y


def compare(x, y):
    X, Y = band_spectrogram(x), band_spectrogram(y)
    n = min(len(X), len(Y))
    X, Y = X[:n], Y[:n]
    # one overall gain, from total power (the synth runs at its own level)
    off = 10 * np.log10((10 ** (X / 10)).sum() / ((10 ** (Y / 10)).sum() + 1e-30))
    Y = Y + off
    X, Y = X[5:], Y[5:]                            # skip the synth's start-up
    # per band: time-averaged power, as a spectrogram (and the ear) adds it up
    pmean = lambda A: 10 * np.log10(np.mean(10 ** (A / 10), axis=0) + 1e-30)
    mx, my = pmean(X), pmean(Y)
    per_band = np.where(mx > mx.max() - 50, my - mx, 0.0)
    per_band -= np.median(per_band[mx > mx.max() - 50])
    n = len(X)
    k = 5                                          # 0.25 s: noise fluctuates per frame
    Xs = np.array([X[i:i + k].mean(0) for i in range(0, n - k, k)])
    Ys = np.array([Y[i:i + k].mean(0) for i in range(0, n - k, k)])
    act = Xs > Xs.max() - 50
    mae = float(np.mean(np.abs(Ys - Xs)[act]))
    return per_band, mae


def copy_synth_isolated(profile_path, noise=True):
    """copy_synth in a fresh process: signalflow can crash rendering a second graph."""
    import subprocess
    import sys
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "xy.npz"
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "_copysynth",
                        str(profile_path), str(out)] + ([] if noise else ["--no-noise"]),
                       check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        z = np.load(out)
        return z["x"], z["y"]


def noise_filter_cal(profile_path):
    """Render white noise through each band filter alone and measure its density in its
    own band with the same measure the analysis uses -> amplitude calibration."""
    from signalflow import AudioGraph, AudioGraphConfig, SVFilter, WhiteNoise
    prof = json.loads(Path(profile_path).read_text())
    cfg = AudioGraphConfig()
    cfg.backend_name = "null"
    cfg.output_buffer_size = HOP
    g = AudioGraph(cfg)
    g.stop()
    fr = np.fft.rfftfreq(N_FFT, 1 / SR)
    cal = []
    src = WhiteNoise()
    win = np.hanning(N_FFT)
    for (lo, hi), fc in zip(_band_edges(), prof["bands_hz"]):
        node = src * 1.0                     # white noise itself: the EQ shapes it
        node.play()
        blocks = []
        for _ in range(150):
            buf = g.render_to_new_buffer(HOP)       # keep the Buffer alive while copying
            blocks.append(buf.data[0].copy())
        y = np.concatenate(blocks).astype(np.float64)
        node.stop()
        b = (fr >= lo) & (fr < hi)
        dens = np.mean([(np.abs(np.fft.rfft(y[i:i + N_FFT] * win)) ** 2)[b].mean()
                        for i in range(0, len(y) - N_FFT, N_FFT // 2)])
        cal.append(1.0 / math.sqrt(dens))
    return cal


def _band_power(a):
    """Time-averaged power per 1/3-octave band (linear), skipping the start-up."""
    B = band_spectrogram(a)[5:]
    return np.mean(10 ** (B / 10), axis=0)


def fit(profile_path, rounds=4):
    """Partials are kept as measured; each band's noise is solved for: the recording's
    band power minus the synth's partial power there."""
    import subprocess
    import sys
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "_noisecal",
                    str(profile_path)], check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    prof = json.loads(Path(profile_path).read_text())
    prof["eq"] = [1.0] * len(prof["bands_hz"])
    prof["partial_eq"] = [1.0] * len(prof["bands_hz"])
    Path(profile_path).write_text(json.dumps(prof))
    x, yp = copy_synth_isolated(profile_path, noise=False)
    R, Fp = _band_power(x), _band_power(yp)
    lo_res = R / (Fp + 1e-30)
    tonal = Fp > 0.3 * np.median(Fp)
    alpha = float(np.median(lo_res[tonal & (R > R.max() * 1e-5)]))
    over = alpha * Fp > 0.8 * R
    if over.any():
        peq = np.where(over, np.sqrt(0.8 * R / (alpha * Fp + 1e-30)), 1.0)
        prof["partial_eq"] = np.clip(peq, 0.01, 1.0).tolist()
        Path(profile_path).write_text(json.dumps(prof))
        print("partials turned down in", ", ".join(f"{b:.0f} Hz ({20 * np.log10(p):.0f} dB)"
                                                for b, p, o in zip(prof["bands_hz"], peq, over) if o))
        x, yp = copy_synth_isolated(profile_path, noise=False)
        Fp = _band_power(yp)
    # overall scale: from bands where the recording is mostly tone (partials only, so it
    # doesn't depend on the noise we are about to solve for)
    lo_res = R / (Fp + 1e-30)
    tonal = Fp > 0.3 * np.median(Fp)
    alpha = float(np.median(lo_res[tonal & (R > R.max() * 1e-5)]))
    for r in range(rounds + 1):
        x, y = copy_synth_isolated(profile_path)
        F = _band_power(y)
        per_band, mae = compare(x, y)
        prof = json.loads(Path(profile_path).read_text())
        avg = float(np.mean(np.abs(per_band[per_band != 0])))
        worst = int(np.abs(per_band).argmax())
        tag = "final" if r == rounds else f"round {r + 1}"
        print(f"{tag}: average spectrum within {avg:.2f} dB (~{(10 ** (avg / 20) - 1) * 100:.0f}% "
              f"amplitude), worst band {per_band[worst]:+.1f} dB at {prof['bands_hz'][worst]:.0f} Hz"
              f" | moment-to-moment {mae:.2f} dB", flush=True)
        if r == rounds:
            print("per-band offset (dB):", " ".join(
                f"{b:.0f}:{e:+.1f}" for b, e in zip(prof["bands_hz"], per_band)))
            return x, y
        if not np.all(np.isfinite(F)):
            raise SystemExit("copy-synthesis went unstable (NaN) -- not saving this round")
        n_cur = np.maximum(alpha * (F - Fp), R * 1e-4)
        n_need = np.maximum(R - alpha * Fp, R * 0.02)          # never ask for < 2 % of band
        step_db = np.clip(10 * np.log10(n_need / n_cur) * 0.9, -6.0, 6.0)   # per round
        eq_db = np.clip(20 * np.log10(np.array(prof["eq"])) + step_db, -60.0, 30.0)
        prof["eq"] = (10 ** (eq_db / 20)).tolist()
        Path(profile_path).write_text(json.dumps(prof))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyse")
    a.add_argument("audio")
    a.add_argument("--band", nargs=2, type=float, default=(55.0, 165.0))
    a.add_argument("--name", required=True)
    a.add_argument("--top-kmh", type=float, default=100.0)
    a.add_argument("--whiten", action="store_true",
                   help="ignore constant lines when tracking (for tones under cabin hum)")
    a.add_argument("--broad", nargs=2, type=float, default=(0.0, 20.0), metavar=("SHARE", "HZ"),
                   help="line width: share of each partial spread into a band ~2xHZ wide")
    a.add_argument("--span", nargs=2, type=float, default=None, metavar=("START", "END"),
                   help="use only this part of the recording (seconds)")
    f = sub.add_parser("fit")
    f.add_argument("profile")
    f.add_argument("--rounds", type=int, default=4)
    f.add_argument("--wav", default=None, help="also write the copy-synthesis here")
    nc = sub.add_parser("_noisecal")
    nc.add_argument("profile")
    c = sub.add_parser("_copysynth")
    c.add_argument("profile")
    c.add_argument("out")
    c.add_argument("--no-noise", action="store_true")
    args = ap.parse_args()
    if args.cmd == "_noisecal":
        prof = json.loads(Path(args.profile).read_text())
        prof["noise_cal"] = noise_filter_cal(args.profile)
        Path(args.profile).write_text(json.dumps(prof))
        return
    if args.cmd == "_copysynth":
        x, y = copy_synth(args.profile, noise=not args.no_noise)
        np.savez(args.out, x=x, y=y)
        return
    if args.cmd == "analyse":
        out, p = analyse(args.audio, args.band, args.name, top_kmh=args.top_kmh,
                         whiten=args.whiten, span=args.span, broad=args.broad[0],
                         broad_hz=args.broad[1])
        print(f"{out}: pitch {p['idle_hz']:.0f}-{p['top_hz']:.0f} Hz, frames {p['frames']}")
    else:
        x, y = fit(args.profile, args.rounds)
        if args.wav:
            import wave
            yy = y / (np.abs(y).max() + 1e-9) * 0.9
            with wave.open(args.wav, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SR)
                w.writeframes((yy * 32767).astype("<i2").tobytes())


if __name__ == "__main__":
    main()
