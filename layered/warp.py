"""Warp bubble: a spaceship's field drive, heard from inside the ship. Continuous (no gears, no
spool): its motion comes from acceleration, its size from speed, and the space around it is
alive -- events arrive at random, more of them the faster and harder you go.

Continuous layers
  drone     a Shepard-Risset glissando: octave-spaced partials under a fixed bell-shaped
            spectral envelope, gliding with ACCELERATION (an endless climb pressing on, an
            endless descent braking, still when cruising); the envelope's centre rises a
            little with speed. The field is unstable: its pitch and level wander, and now
            and then it flickers
  bubble    the field's pressure: a deep sine throbbing faster with speed
  tunnel    the warp tunnel's rush: turbulent, decorrelated noise rising with speed
  shimmer   the field's edge: noise through a comb tuned to the drone, sparkling with speed

Random events (rates rise with speed / acceleration / load)
  streaks   particles and debris streaking past: a whistling band that falls in pitch as
            it passes, swept across the stereo field
  hull      the hull groaning under stress: deep metallic resonances bending down, when
            accelerating or braking hard
  waves     subspace pressure waves: a deep, slow boom rolling through
  arcs      energy arcing across the field under heavy load: brief sparse crackle
Driving events
  jump      a deep, fast press winds the field up (a rising swell) and jumps: a boom, a flash
            of noise, a long tail
  collapse  a quick lift lets the field go: a falling whoosh
  drop out  braking down through ~20 km/h drops out of warp: a falling sweep and a thump
  width     the stereo field opens with speed
"""
from __future__ import annotations

import numpy as np

import dsp
from dsp import SR, lerp

BASE_HZ = 55.0            # the drone's lowest octave
OCTAVES = 7
GLIDE_OCT_PER_MS2 = 0.06  # octaves per second of glide per m/s^2 of acceleration


def smooth_noise(n, hz, rng):
    """Random wander, roughly -1..1, changing at about hz."""
    x = dsp.biquad(rng.standard_normal(n), "lp", hz, 0.7)
    return x / (np.std(x) * 2.5 + 1e-9)


def schedule(rate: np.ndarray, rng, min_gap=0.05):
    """Event start indices for a time-varying rate (events per second, at audio rate)."""
    times = []
    i = 0
    n = len(rate)
    while i < n:
        r = rate[i]
        if r <= 1e-3:
            i += int(0.1 * SR)
            continue
        i += int(max(min_gap, rng.exponential(1 / r)) * SR)
        if i < n:
            times.append(i)
    return times


def add(out, i, sig_l, sig_r):
    m = min(len(sig_l), len(out) - i)
    if m > 0:
        out[i:i + m, 0] += sig_l[:m]
        out[i:i + m, 1] += sig_r[:m]


def render(c: dict, seed: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(c["rpm"])
    kmh = c["kmh"]
    speed_n = np.clip(kmh / 160, 0, 1)
    pedal = np.clip(c["pedal"], 0, 1)
    load = np.clip(c["load"], 0, 1)
    accel = dsp.one_pole(c["accel"], 0.3)
    out = np.zeros((n, 2))

    # --- an unstable field: wander and flicker -------------------------------------------
    wander_cents = 18 * smooth_noise(n, 0.6, rng)
    wander_level = 1 + 0.22 * smooth_noise(n, 0.4, rng)
    flicker = np.ones(n)
    for i in schedule(0.05 + 0.25 * speed_n * (0.4 + 0.6 * load), rng, 1.5):
        d = int(rng.uniform(0.06, 0.18) * SR)
        depth = rng.uniform(0.4, 0.8)
        flicker[i:i + d] *= 1 - depth * np.sin(np.pi * np.arange(min(d, n - i)) / d) ** 2
    field = wander_level * flicker

    # --- drone: Shepard-Risset glide driven by acceleration ---------------------------------
    glide = np.cumsum(accel * GLIDE_OCT_PER_MS2) / SR
    centre = lerp(3.0, 4.2, speed_n)
    wobble = 2 ** (wander_cents / 1200)
    drone = np.zeros((n, 2))
    for k in range(OCTAVES):
        pos = (k + glide) % OCTAVES
        f = BASE_HZ * 2 ** pos * wobble
        env = np.exp(-0.5 * ((pos - centre) / 1.25) ** 2)
        ph = dsp.phase(f)
        s = np.sin(2 * np.pi * ph) + 0.15 * np.sin(2 * np.pi * 3 * ph)
        lr = 0.5 + 0.25 * np.sin(2 * np.pi * (0.05 + 0.02 * k) * np.arange(n) / SR + k)
        drone[:, 0] += s * env * lr
        drone[:, 1] += s * env * (1 - lr)
    drone_level = (0.10 + 0.08 * speed_n + 0.10 * load) * field
    out += drone * dsp.one_pole(drone_level, 0.05)[:, None] * 0.5

    # --- bubble -----------------------------------------------------------------------------
    throb = 0.55 + 0.45 * np.sin(2 * np.pi * dsp.phase(lerp(0.8, 7.0, speed_n) * (1 + 0.15 * smooth_noise(n, 0.3, rng)))) ** 2
    bubble = np.sin(2 * np.pi * dsp.phase(lerp(38.0, 52.0, speed_n))) * throb
    bubble *= dsp.one_pole(0.25 + 0.35 * load + 0.2 * speed_n, 0.2) * 0.32 * field
    out += bubble[:, None] * 0.5

    # --- tunnel rush ---------------------------------------------------------------------
    turb = np.clip(1 + 0.45 * smooth_noise(n, 1.2, rng), 0.2, 2)
    tunnel = np.stack([dsp.biquad(dsp.pink(n, seed + s), "bp", lerp(300.0, 1400.0, speed_n) * (1 + 0.2 * smooth_noise(n, 0.5, rng)), 0.5)
                       for s in (11, 12)], 1)
    out += tunnel * (turb * (0.01 + 0.10 * speed_n ** 1.3))[:, None]

    # --- shimmer ---------------------------------------------------------------------------
    noise = dsp.biquad(dsp.white(n, seed + 1), "hp", 2500.0)
    comb_hz = BASE_HZ * 2 ** (3 + (glide % 1.0))
    shimmer = sum(dsp.biquad(noise, "bp", comb_hz * h, 18.0) for h in (8, 12, 16, 24))
    sparkle = np.abs(dsp.biquad(dsp.white(n, seed + 2), "lp", 12.0)) * 4
    shimmer *= sparkle * dsp.one_pole(0.02 + 0.10 * speed_n + 0.08 * load, 0.3) * 0.45
    out[:, 0] += shimmer * 0.6
    out[:, 1] += shimmer * 0.4

    # --- streaks: debris passing ---------------------------------------------------------
    for i in schedule(0.15 + 2.2 * speed_n ** 1.2, rng, 0.08):
        dur = rng.uniform(0.25, 0.9) * (1.25 - 0.6 * speed_n[i])
        m = int(dur * SR)
        t = np.arange(m) / m
        fc = rng.uniform(700, 3800) * lerp(1.3, 0.7, t)                 # falls as it passes
        band = dsp.biquad(dsp.white(m, int(rng.integers(1 << 30))), "bp", fc, 7.0)
        tone = np.sin(2 * np.pi * np.cumsum(fc * 0.5) / SR) * 0.25
        env = np.sin(np.pi * t) ** 3 * rng.uniform(0.03, 0.08) * (0.4 + 0.6 * speed_n[i])
        p0 = rng.uniform(0.0, 0.3) if rng.random() < 0.5 else rng.uniform(0.7, 1.0)
        pan = lerp(p0, 1 - p0, t)
        sig = (band + tone) * env
        add(out, i, sig * pan, sig * (1 - pan))

    # --- hull: groans under stress -------------------------------------------------------
    stress = np.clip(np.abs(accel) / 3.0, 0, 1)
    for i in schedule(0.02 + 0.5 * stress ** 1.5, rng, 0.6):
        dur = rng.uniform(0.6, 1.6)
        m = int(dur * SR)
        t = np.arange(m) / SR
        exc = dsp.biquad(dsp.white(m, int(rng.integers(1 << 30))), "lp", 600.0) * (1 - np.exp(-t / 0.05)) * np.exp(-t / (dur * 0.4))
        groan = np.zeros(m)
        for _ in range(3):
            f0 = rng.uniform(70, 380)
            groan += dsp.biquad(exc, "bp", f0 * lerp(1.0, 0.9, t / dur), 25.0)
        groan *= rng.uniform(0.15, 0.3) * (0.4 + 0.6 * stress[i])
        p = rng.uniform(0.3, 0.7)
        add(out, i, groan * p, groan * (1 - p))

    # --- waves: subspace pressure booms ----------------------------------------------------
    for i in schedule(0.03 + 0.10 * speed_n, rng, 4.0):
        m = int(2.4 * SR)
        t = np.arange(m) / SR
        env = (1 - np.exp(-t / 0.12)) * np.exp(-t / 0.7)
        boom = (np.sin(2 * np.pi * rng.uniform(28, 44) * t) * 0.8
                + dsp.biquad(dsp.white(m, int(rng.integers(1 << 30))), "lp", 140.0) * 0.6) * env * 0.5
        add(out, i, boom, boom)

    # --- arcs: energy crackle under heavy load ---------------------------------------------
    heavy = np.clip((load - 0.5) / 0.5, 0, 1) * (0.3 + 0.7 * speed_n)
    for i in schedule(1.8 * heavy, rng, 0.15):
        m = int(rng.uniform(0.08, 0.35) * SR)
        clicks = (rng.random(m) < rng.uniform(0.002, 0.01)).astype(float)
        arc = dsp.biquad(dsp.biquad(clicks, "bp", rng.uniform(3500, 7000), 3.0) + 0.15 * dsp.white(m, int(rng.integers(1 << 30))),
                         "hp", 2500.0) * np.sin(np.pi * np.arange(m) / m) * 0.25
        p = rng.uniform(0.2, 0.8)
        add(out, i, arc * p, arc * (1 - p))

    # --- jump on a deep fast press; collapse on a quick lift; drop out braking --------------
    rate = np.gradient(dsp.one_pole(pedal, 0.04)) * SR
    ch_len = int(1.3 * SR)
    ts = np.arange(ch_len) / SR
    swell = (np.sin(2 * np.pi * np.cumsum(lerp(70.0, 700.0, (ts / 1.3) ** 2)) / SR) * 0.5
             + dsp.biquad(dsp.white(ch_len, seed + 3), "bp", lerp(400.0, 3000.0, ts / 1.3), 1.0) * 0.6) * (ts / 1.3) ** 2
    jump_len = int(2.5 * SR)
    tj = np.arange(jump_len) / SR
    flash = dsp.biquad(dsp.white(jump_len, seed + 5), "bp", 2500.0, 0.5) * np.exp(-tj / 0.15)
    tail = dsp.biquad(dsp.pink(jump_len, seed + 6), "lp", lerp(1800.0, 200.0, tj / 2.5)) * np.exp(-tj / 0.8)
    thud = np.sin(2 * np.pi * 36 * tj) * (1 - np.exp(-tj / 0.01)) * np.exp(-tj / 0.45)
    jump = flash * 0.35 + tail * 0.35 + thud * 0.9
    last = -10 * SR
    for i in np.nonzero((pedal[1:] >= 0.6) & (pedal[:-1] < 0.6) & (rate[1:] > 2.0))[0]:
        if i - last < 3 * SR:
            continue
        last = i
        add(out, i, swell * 0.22, swell * 0.22)
        j = i + ch_len
        if j < n and pedal[j] > 0.8:                     # still pressing: it jumps
            jl = np.roll(jump, 0)
            add(out, j, jl * 0.6, dsp.biquad(jl, "peak", 1800.0, 1.0, -3.0) * 0.6)
    collapse = dsp.biquad(dsp.white(ch_len, seed + 4), "bp", 1400.0, 0.9)
    collapse = dsp.biquad(collapse, "lp", lerp(6000.0, 300.0, ts / 1.3), 0.7) * np.exp(-ts / 0.35)
    last = -10 * SR
    for i in np.nonzero((rate[1:] < -2.5) & (rate[:-1] >= -2.5))[0]:
        if i - last < SR:
            continue
        last = i
        m = min(ch_len, n - i)
        add(out, i, collapse[:m] * 0.5 * np.linspace(1, 0.3, m), collapse[:m] * 0.5 * np.linspace(0.3, 1, m))
    drop_len = int(1.6 * SR)
    td = np.arange(drop_len) / SR
    drop = (np.sin(2 * np.pi * np.cumsum(lerp(600.0, 50.0, td / 1.6)) / SR) * np.exp(-td / 0.5) * 0.4
            + np.sin(2 * np.pi * 40 * td) * np.exp(-((td - 0.9) / 0.12) ** 2) * 0.8)
    braking = (accel < -1.5)
    for i in np.nonzero((kmh[1:] < 20) & (kmh[:-1] >= 20) & braking[1:])[0]:
        add(out, i, drop * 0.5, drop * 0.5)

    # --- width -----------------------------------------------------------------------------
    mid = out.mean(axis=1)
    side = (out[:, 0] - out[:, 1]) / 2
    width = lerp(0.8, 1.8, speed_n)
    out = np.stack([mid + side * width, mid - side * width], 1)
    for ch in range(2):
        out[:, ch] = dsp.biquad(out[:, ch], "hp", 25.0)
    return np.tanh(out * 1.4) / 1.4
