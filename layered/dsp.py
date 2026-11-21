"""Small DSP kit for the layered voice prototypes: vectorised oscillators, time-varying RBJ
biquads (coefficients updated every BLOCK samples), noise, smoothing."""
from __future__ import annotations

import wave

import numpy as np
from scipy.signal import lfilter

SR = 48000
BLOCK = 64


def phase(freq: np.ndarray, sr: int = SR, start: float = 0.0) -> np.ndarray:
    """Running phase in cycles for a frequency track (Hz per sample)."""
    return start + np.cumsum(freq) / sr


def sine(freq, sr=SR):
    return np.sin(2 * np.pi * phase(np.asarray(freq, float) * np.ones(1), sr))


def white(n, seed=0):
    return np.random.default_rng(seed).standard_normal(n)


def pink(n, seed=0):
    # Paul Kellet's economy filter, as one IIR
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    return lfilter(b, a, white(n, seed)) * 3.5


def one_pole(x: np.ndarray, tau_sec: float, sr=SR) -> np.ndarray:
    k = 1 - np.exp(-1 / (tau_sec * sr))
    return lfilter([k], [1, -(1 - k)], x)


def _rbj(kind, f, q, gain_db, sr):
    w = 2 * np.pi * np.clip(f, 10, sr * 0.45) / sr
    cw, sw = np.cos(w), np.sin(w)
    alpha = sw / (2 * q)
    A = 10 ** (gain_db / 40)
    if kind == "lp":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]; a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "hp":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]; a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "bp":
        b = [alpha, 0, -alpha]; a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "peak":
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]; a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    elif kind == "lowshelf":
        s = 2 * np.sqrt(A) * alpha
        b = [A * ((A + 1) - (A - 1) * cw + s), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - s)]
        a = [(A + 1) + (A - 1) * cw + s, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - s]
    elif kind == "highshelf":
        s = 2 * np.sqrt(A) * alpha
        b = [A * ((A + 1) + (A - 1) * cw + s), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - s)]
        a = [(A + 1) - (A - 1) * cw + s, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - s]
    else:
        raise ValueError(kind)
    b = np.array(b) / a[0]
    a = np.array(a) / a[0]
    return b, a


def biquad(x: np.ndarray, kind: str, f, q=0.707, gain_db=0.0, sr=SR) -> np.ndarray:
    """Biquad whose frequency / gain may be arrays (one value per sample): coefficients are
    taken at the start of each BLOCK, state carried across."""
    f = np.broadcast_to(np.asarray(f, float), x.shape)
    g = np.broadcast_to(np.asarray(gain_db, float), x.shape)
    if np.ptp(f) == 0 and np.ptp(g) == 0:
        b, a = _rbj(kind, f[0], q, g[0], sr)
        return lfilter(b, a, x)
    y = np.empty_like(x)
    zi = np.zeros(2)
    for i in range(0, len(x), BLOCK):
        b, a = _rbj(kind, f[i], q, g[i], sr)
        y[i:i + BLOCK], zi = lfilter(b, a, x[i:i + BLOCK], zi=zi)
    return y


def lerp(a, b, t):
    return a + (b - a) * np.clip(t, 0, 1)


def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def events(counter: np.ndarray) -> np.ndarray:
    """Sample indices where a counter (shifts, lift-offs...) goes up."""
    return np.nonzero(np.diff(counter) > 0)[0] + 1


def write_wav(path, stereo: np.ndarray, sr=SR, peak_db=-1.0):
    """Normalise to peak_db and write 16-bit stereo."""
    x = stereo / (np.abs(stereo).max() + 1e-9) * 10 ** (peak_db / 20)
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _blep(t, dt):
    """PolyBLEP residual for a unit step at phase 0 (t in [0, 1), dt = f / sr)."""
    r = np.zeros_like(t)
    a = t < dt
    u = t[a] / dt[a]
    r[a] = u + u - u * u - 1
    b = t > 1 - dt
    u = (t[b] - 1) / dt[b]
    r[b] = u * u + u + u + 1
    return r


def saw_bl(freq, sr=SR, offset=0.0):
    """Band-limited sawtooth (PolyBLEP) for a frequency track; offset shifts its phase."""
    freq = np.asarray(freq, float)
    dt = np.clip(freq / sr, 1e-9, 0.5)
    t = (phase(freq, sr) + offset) % 1.0
    return 2 * t - 1 - _blep(t, dt)


def pulse_bl(freq, duty=0.5, sr=SR):
    """Band-limited pulse: the difference of two saws duty apart."""
    return 0.5 * (saw_bl(freq, sr) - saw_bl(freq, sr, offset=duty))
