"""Flat-6: a layered, physically-minded synthesis of a naturally aspirated boxer six, driven by
the DCT's revs and load. Each cylinder's exhaust pulse is synthesised and run through the
exhaust; intake, mechanical and overrun layers ride on top.

Layers
  exhaust   one pulse per firing (order 3), firing order 1-6-2-4-3-5. Each pulse is a sum of
            decaying resonances (formants) whose brightness rises with load. The two banks
            differ slightly (gain, formant tuning): that imbalance puts energy at order 1.5,
            the boxer's rasp. Cycle-to-cycle variation keeps it alive. Pipe resonances and a
            muffler that opens with load and revs shape it.
  intake    induction roar: pink noise in a band that climbs with the revs, pulsed by the
            firing, coming in with load -- the howl behind a flat-6 at high revs
  mech      valvetrain and timing-chain detail: a faint high order whine and ticking
  overrun   off the throttle: the exhaust softens and darkens; after lifting off a hard pull,
            crackles and pops
  shifts    a DCT upshift under load cuts the ignition (the pulses drop out for the cut);
            the regular hydraulic shift sound plays on every shift
"""
from __future__ import annotations

import numpy as np

import dsp
import shiftfx
from dsp import SR, lerp

IDLE, REDLINE = 900.0, 8500.0
FIRING_ORDER = [0, 5, 1, 3, 2, 4]          # cylinders 1-6-2-4-3-5
BANK = [0, 1, 0, 1, 0, 1]                  # cylinders 1,3,5 left; 2,4,6 right
# per-cylinder gain spread (manufacturing / header differences)
CYL_GAIN = np.array([1.00, 0.93, 1.05, 0.97, 1.02, 0.95])
# formants of one exhaust pulse: (Hz, decay s, level at light load, level at full load)
FORMANTS = [(170.0, 0.0045, 1.00, 1.00),
            (480.0, 0.0018, 0.45, 0.85),
            (1250.0, 0.0009, 0.10, 0.55),
            (3100.0, 0.00035, 0.02, 0.22)]
BANK_TUNE = [1.0, 1.10]                    # right bank's pulses ring a touch higher...
# ...and are weaker: order-1.5 rasp and the low-rev burble, evening out toward the redline
BANK_GAIN_LOW_REVS, BANK_GAIN_HIGH_REVS = 0.66, 0.86


def pulses(t_since, f_scale, bright):
    """One exhaust pulse evaluated at times since firing (s)."""
    y = np.zeros_like(t_since)
    attack = 1 - np.exp(-t_since / 0.00025)
    for f, d, lo, hi in FORMANTS:
        level = lo + (hi - lo) * bright
        y += level * np.exp(-t_since / d) * np.sin(2 * np.pi * f * f_scale * t_since)
    return y * attack


def render(c: dict, seed: int = 1) -> np.ndarray:
    """c: control signals at audio rate (drive.to_audio_rate). Returns stereo (n, 2)."""
    rng = np.random.default_rng(seed)
    n = len(c["rpm"])
    rpm = np.maximum(c["rpm"], IDLE * 0.95)      # the DCT runs over this engine's rev range
    rpm_n = np.clip((rpm - IDLE) / (REDLINE - IDLE), 0, 1)
    load = np.clip(c["load"], 0, 1)
    overrun = np.clip(c["overrun"], 0, 1)
    cut = c["limiter"] > 0.5

    # --- exhaust ---------------------------------------------------------------------
    fire_hz = rpm / 20.0                     # 6 cylinders, 4-stroke: 3 firings per rev
    x = dsp.phase(fire_hz)                   # firings so far
    k = np.floor(x).astype(np.int64)
    frac = x - k
    t_cur = frac / fire_hz
    t_prev = t_cur + 1.0 / fire_hz           # the previous pulse is still ringing out
    cyl_cur = np.array(FIRING_ORDER)[k % 6]
    cyl_prev = np.array(FIRING_ORDER)[(k - 1) % 6]
    n_fire = int(k.max()) + 2
    cycle_var = 1 + 0.10 * rng.standard_normal(n_fire)           # cycle-to-cycle variation
    weak_bank = lerp(BANK_GAIN_LOW_REVS, BANK_GAIN_HIGH_REVS, rpm_n)
    # the ignition cut: an upshift's torque handover (deep at load), the limiter
    fire_amp = 1.0 - 0.85 * np.clip(c["shifting"], 0, 1) * (load > 0.3)
    fire_amp = np.where(cut & (np.sin(2 * np.pi * dsp.phase(np.full(n, 11.0))) > 0), 0.15, fire_amp)
    bright = np.clip(0.15 + 0.85 * load * (0.5 + 0.5 * rpm_n), 0, 1) * (1 - 0.6 * overrun)

    out = np.zeros((n, 2))
    for which, t_s, cyl, kk in ((0, t_cur, cyl_cur, k), (1, t_prev, cyl_prev, k - 1)):
        bank = np.array(BANK)[cyl]
        tune = np.where(bank == 0, BANK_TUNE[0], BANK_TUNE[1])
        g = CYL_GAIN[cyl] * np.where(bank == 0, 1.0, weak_bank) * cycle_var[np.maximum(kk, 0)]
        p = pulses(t_s, tune, bright) * g
        # each bank's own tailpipe: a little left / right
        out[:, 0] += p * np.where(bank == 0, 0.62, 0.38)
        out[:, 1] += p * np.where(bank == 0, 0.38, 0.62)
    # combustion turbulence: grit on every pulse, more under load
    grit = dsp.biquad(dsp.white(n, seed + 3), "bp", 2200.0, 0.8)
    out *= (1 + 0.35 * load[:, None] * grit[:, None])
    out *= fire_amp[:, None]
    # pipes and muffler
    for ch in range(2):
        y = out[:, ch]
        y = dsp.biquad(y, "peak", 92.0, 2.5, 5.0)              # tailpipe mode
        y = dsp.biquad(y, "peak", 270.0, 2.0, 3.0)             # header / collector
        mute = lerp(1400.0, 8000.0, load * (0.35 + 0.65 * rpm_n)) * (1 - 0.45 * overrun)
        y = dsp.biquad(y, "lp", mute, 0.8)
        out[:, ch] = y
    # loudness: mostly load (a light cruise sits ~12 dB under full throttle); off the
    # throttle the exhaust falls right back to a burble
    level = (0.15 + 0.85 * load ** 1.3) * (0.55 + 0.45 * rpm_n)
    level = level * (1 - overrun) + 0.07 * overrun * (0.4 + 0.6 * rpm_n)
    out *= dsp.one_pole(level, 0.02)[:, None]

    # --- intake --------------------------------------------------------------------------
    pulse_env = np.exp(-t_cur / np.maximum(0.25 / fire_hz, 0.0006))
    roar = dsp.biquad(dsp.pink(n, seed + 5), "bp", lerp(700.0, 2600.0, rpm_n), 1.1)
    roar *= (0.45 + 0.55 * pulse_env)
    roar *= dsp.one_pole(load ** 1.4 * (0.25 + 0.75 * rpm_n ** 1.3), 0.05) * 0.55
    out += roar[:, None] * np.array([0.5, 0.5])

    # --- mechanical ---------------------------------------------------------------------
    chain = np.sin(2 * np.pi * dsp.phase(rpm / 60 * 21)) * 0.012 * (0.3 + rpm_n)
    tick = dsp.biquad(dsp.white(n, seed + 7), "bp", 5200.0, 2.0) * (pulse_env ** 3) * 0.05 * (0.2 + rpm_n)
    out += (chain + tick)[:, None] * 0.5

    # --- overrun crackle after lifting off a hard pull -------------------------------------
    crackle = np.zeros(n)
    active = np.zeros(n)
    for i in dsp.events(c["lift_offs"]):
        active[i:i + int(1.6 * SR)] = np.linspace(1, 0, len(active[i:i + int(1.6 * SR)]))
    rate = active * overrun * (8 + 30 * rpm_n)            # pops per second
    pops = np.nonzero(rng.random(n) < rate / SR)[0]
    burst = dsp.biquad(dsp.white(int(0.03 * SR), seed + 9), "bp", 1800.0, 0.7)
    burst *= np.exp(-np.arange(len(burst)) / (0.006 * SR))
    thump = np.sin(2 * np.pi * 85 * np.arange(len(burst)) / SR) * np.exp(-np.arange(len(burst)) / (0.012 * SR))
    for i in pops:
        m = min(len(burst), n - i)
        crackle[i:i + m] += (burst[:m] * rng.uniform(0.4, 1.0) + thump[:m] * 0.8) * 0.35
    pan = 0.5 + 0.2 * np.sin(np.arange(n) / SR * 3.0)
    out[:, 0] += crackle * pan
    out[:, 1] += crackle * (1 - pan)

    # --- the regular hydraulic shift sound on every shift ---------------------------------
    shiftfx.mix(out, c, level=0.30)

    # --- in the cabin: a little weight, the top softened -------------------------------
    for ch in range(2):
        out[:, ch] = dsp.biquad(out[:, ch], "lowshelf", 140.0, 0.7, 3.0)
        out[:, ch] = dsp.biquad(out[:, ch], "hp", 28.0, 0.7)
    return np.tanh(out * 1.6) / 1.6
