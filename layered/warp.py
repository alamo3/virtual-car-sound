"""Warp bubble: a spaceship's field drive. Continuous (no gears, no spool): its motion comes
from acceleration, its size from speed.

Layers
  drone     a Shepard-Risset glissando: octave-spaced partials under a fixed bell-shaped
            spectral envelope. Its glide follows ACCELERATION -- pressing on sounds like an
            endless climb, braking an endless descent, cruising holds still -- while the
            envelope's centre rises a little with speed
  bubble    the field's pressure: a deep sine whose level throbs at a rate that rises with
            speed (slow heartbeat at a crawl, a rapid flutter at speed)
  shimmer   the field's edge: noise through a comb tuned to the drone, sparkling under load
            and with speed
  charge    a deep, fast press winds the field up: a rising swell over about a second
  collapse  a quick lift lets it go: a falling whoosh
  width     the stereo field opens with speed (slow opposite-phase drift between the sides)
"""
from __future__ import annotations

import numpy as np

import dsp
from dsp import SR, lerp

BASE_HZ = 55.0            # the drone's lowest octave
OCTAVES = 7
GLIDE_OCT_PER_MS2 = 0.06  # octaves per second of glide per m/s^2 of acceleration


def render(c: dict, seed: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(c["rpm"])
    kmh = c["kmh"]
    speed_n = np.clip(kmh / 160, 0, 1)
    pedal = np.clip(c["pedal"], 0, 1)
    load = np.clip(c["load"], 0, 1)
    accel = dsp.one_pole(c["accel"], 0.3)
    out = np.zeros((n, 2))

    # --- drone: Shepard-Risset glide driven by acceleration ---------------------------------
    glide = np.cumsum(accel * GLIDE_OCT_PER_MS2) / SR          # octaves travelled
    centre = lerp(3.0, 4.2, speed_n)                            # envelope centre (octave index)
    drone = np.zeros((n, 2))
    for k in range(OCTAVES):
        pos = (k + glide) % OCTAVES                             # this partial's octave position
        f = BASE_HZ * 2 ** pos
        env = np.exp(-0.5 * ((pos - centre) / 1.25) ** 2)       # fixed bell over log-frequency
        ph = dsp.phase(f)
        s = np.sin(2 * np.pi * ph) + 0.15 * np.sin(2 * np.pi * 3 * ph)   # a hint of fifth above the octave
        # each partial drifts slowly between the sides
        lr = 0.5 + 0.25 * np.sin(2 * np.pi * (0.05 + 0.02 * k) * np.arange(n) / SR + k)
        drone[:, 0] += s * env * lr
        drone[:, 1] += s * env * (1 - lr)
    drone_level = 0.10 + 0.08 * speed_n + 0.10 * load
    out += drone * dsp.one_pole(drone_level, 0.2)[:, None] * 0.55

    # --- bubble: deep throbbing pressure ---------------------------------------------------
    throb_hz = lerp(0.8, 7.0, speed_n)
    throb = 0.55 + 0.45 * np.sin(2 * np.pi * dsp.phase(throb_hz)) ** 2
    bubble = np.sin(2 * np.pi * dsp.phase(lerp(38.0, 52.0, speed_n))) * throb
    bubble *= dsp.one_pole(0.25 + 0.35 * load + 0.2 * speed_n, 0.2) * 0.35
    out += bubble[:, None] * 0.5

    # --- shimmer: comb-filtered noise tuned to the drone -------------------------------------
    noise = dsp.biquad(dsp.white(n, seed + 1), "hp", 2500.0)
    comb_hz = BASE_HZ * 2 ** (3 + (glide % 1.0))
    shimmer = np.zeros(n)
    for h in (8, 12, 16, 24):                                    # resonances on the drone's harmonics
        shimmer += dsp.biquad(noise, "bp", comb_hz * h, 18.0)
    sparkle = np.abs(dsp.biquad(dsp.white(n, seed + 2), "lp", 12.0)) * 4
    shimmer *= sparkle * dsp.one_pole(0.02 + 0.10 * speed_n + 0.10 * load, 0.3) * 0.5
    out[:, 0] += shimmer * 0.6
    out[:, 1] += dsp.biquad(shimmer, "peak", 5000.0, 1.0, 2.0) * 0.4

    # --- charge on a deep, fast press; collapse on a quick lift ----------------------------
    rate = np.gradient(dsp.one_pole(pedal, 0.04)) * SR
    sw_len = int(1.1 * SR)
    ts = np.arange(sw_len) / SR
    charge = (np.sin(2 * np.pi * np.cumsum(lerp(90.0, 900.0, (ts / 1.1) ** 2)) / SR) * 0.5
              + dsp.biquad(dsp.white(sw_len, seed + 3), "bp", 1200.0, 1.0) * 0.6) * np.sin(np.pi * ts / 1.1) ** 2
    collapse = dsp.biquad(dsp.white(sw_len, seed + 4), "bp", 1400.0, 0.9)
    collapse = dsp.biquad(collapse, "lp", lerp(6000.0, 300.0, ts / 1.1), 0.7) * np.exp(-ts / 0.35)
    last = -SR
    # a press counts as it passes 60 % while still moving fast
    for i in np.nonzero((pedal[1:] >= 0.6) & (pedal[:-1] < 0.6) & (rate[1:] > 2.0))[0]:
        if i - last < SR:
            continue
        last = i
        m = min(sw_len, n - i)
        out[i:i + m, 0] += charge[:m] * 0.22
        out[i:i + m, 1] += charge[:m] * 0.22
    last = -SR
    for i in np.nonzero((rate[1:] < -2.5) & (rate[:-1] >= -2.5))[0]:
        if i - last < SR:
            continue
        last = i
        m = min(sw_len, n - i)
        out[i:i + m, 0] += collapse[:m] * 0.5 * np.linspace(1, 0.3, m)
        out[i:i + m, 1] += collapse[:m] * 0.5 * np.linspace(0.3, 1, m)

    # --- width: the field opens with speed ------------------------------------------------
    mid = out.mean(axis=1)
    side = (out[:, 0] - out[:, 1]) / 2
    width = lerp(0.8, 1.8, speed_n)
    out = np.stack([mid + side * width, mid - side * width], 1)
    for ch in range(2):
        out[:, ch] = dsp.biquad(out[:, ch], "hp", 28.0)
    return np.tanh(out * 1.4) / 1.4
