"""Fast plane: a fighter-style turbofan heard from the cockpit, run through the car's virtual
gearbox. The gearbox's revs set the target spool, so the engine winds up through each gear
and spools down a little at every shift; the turbine's lag rounds the steps off. (GEARED =
False: no gears, the pedal sets the spool directly.) Floored, the afterburner lights and
stays lit through the shifts; the hard-shift sound and the rev limiter's stutter come from
the gearbox as for the other sounds.

Layers
  fan       the fan's blade-pass tone (shaft rate x blades): a clean whine at low spool that
            turns to "buzzsaw" at high spool -- the shaft-rate harmonics below the blade
            tone, from supersonic blade tips
  core      the high-pressure compressor's thin whine, above the fan
  rumble    combustion rumble: low, rough noise, growing with spool
  roar      jet mixing noise: broad and heavy, rising steeply with spool (jet noise grows
            roughly with the 8th power of exhaust speed; here a gentler power for the cabin)
  burner    afterburner: a thump as it lights, then a crackling, pulsing low roar
  air       airframe rush with road speed
"""
from __future__ import annotations

import numpy as np

import dsp
import shiftfx
from dsp import SR, lerp

GEARED = True
IDLE, REDLINE = 1000.0, 7000.0           # the gearbox's rev range for the jet (as FmodVoice)
GEARED_UP_SEC, GEARED_DOWN_SEC = 0.35, 0.25   # spool lag following the gearbox: quick, but rounded

IDLE_SPOOL = 0.22          # fraction of max fan speed at idle
FAN_MAX_RPM = 10500.0
FAN_BLADES = 18
CORE_RATIO = 1.6           # HP shaft runs this much faster than the fan
SPOOL_UP_SEC, SPOOL_DOWN_SEC = 1.6, 1.0
BURNER_PEDAL, BURNER_SPOOL = 0.9, 0.55


def spool_track(pedal, kmh, rpm=None, sr=SR):
    """The engine's speed (0..1): a lagged follow of what's asked -- the gearbox's revs when
    geared, else the pedal (with a floor rising with road speed)."""
    if rpm is not None:
        target = IDLE_SPOOL + (1 - IDLE_SPOOL) * np.clip((rpm - IDLE) / (REDLINE - IDLE), 0, 1)
        up, down = GEARED_UP_SEC, GEARED_DOWN_SEC
    else:
        target = np.maximum(IDLE_SPOOL + (1 - IDLE_SPOOL) * pedal ** 1.3,
                            IDLE_SPOOL + 0.35 * np.clip(kmh / 150, 0, 1))
        up, down = SPOOL_UP_SEC, SPOOL_DOWN_SEC
    s = np.empty_like(target)
    cur = IDLE_SPOOL
    step = 16                                    # control rate: every 16 samples
    for i in range(0, len(target), step):
        tgt = target[i]
        tau = up if tgt > cur else down
        # spool-up accelerates as it winds up (like a real turbine), so it's not a plain lag
        k = (step / sr) / tau * (0.6 + 0.8 * cur)
        cur += (tgt - cur) * min(1.0, k)
        s[i:i + step] = cur
    return s


def render(c: dict, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(c["rpm"])
    pedal = np.clip(c["pedal"], 0, 1)
    kmh = c["kmh"]
    speed_n = np.clip(kmh / 160, 0, 1)
    spool = spool_track(pedal, kmh, c["rpm"] if GEARED else None)
    sp_n = np.clip((spool - IDLE_SPOOL) / (1 - IDLE_SPOOL), 0, 1)
    out = np.zeros((n, 2))

    # --- fan: blade-pass tone, buzzsaw at high spool ----------------------------------------
    shaft = spool * FAN_MAX_RPM / 60                    # Hz
    bpf = shaft * FAN_BLADES
    fan = np.sin(2 * np.pi * dsp.phase(bpf)) + 0.3 * np.sin(2 * np.pi * dsp.phase(bpf * 2))
    buzz = np.zeros(n)
    for h in range(6, FAN_BLADES):                      # shaft harmonics below the blade tone
        buzz += rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * dsp.phase(shaft * h) + rng.uniform(0, 6.28))
    buzz_amt = np.clip((sp_n - 0.55) / 0.35, 0, 1)
    fan_mix = fan * 0.09 * (0.5 + 0.5 * sp_n) + buzz * 0.012 * buzz_amt
    fan_mix = dsp.biquad(fan_mix, "lp", 6000.0)
    out[:, 0] += fan_mix * 0.55
    out[:, 1] += fan_mix * 0.45

    # --- core whine --------------------------------------------------------------------------
    core = np.sin(2 * np.pi * dsp.phase(shaft * CORE_RATIO * 29)) * 0.02 * sp_n ** 1.5
    out += core[:, None] * 0.5

    # --- rumble and roar ---------------------------------------------------------------------
    rumble = dsp.biquad(dsp.pink(n, seed + 1), "lp", lerp(90.0, 220.0, sp_n), 0.9)
    rumble = dsp.biquad(rumble, "hp", 40.0)
    out += (rumble * (0.10 + 0.25 * sp_n ** 2))[:, None] * np.array([0.5, 0.5])
    roar = np.stack([dsp.biquad(dsp.pink(n, seed + s), "bp", lerp(250.0, 900.0, sp_n), 0.5) for s in (2, 3)], 1)
    out += roar * (0.08 + 0.37 * sp_n ** 3)[:, None]

    # --- afterburner -----------------------------------------------------------------------
    lit = ((pedal > BURNER_PEDAL) & (spool > BURNER_SPOOL)).astype(float)
    burner = dsp.one_pole(lit, 0.25)
    if GEARED:                       # stays lit through the shifts' brief spool-down
        lit = (dsp.one_pole(lit, 0.6) > 0.5).astype(float)
        burner = dsp.one_pole(lit, 0.25)
    crackle_am = np.abs(dsp.biquad(dsp.white(n, seed + 4), "lp", 25.0, 0.7)) * 6     # slow ragged pulsing
    fire = dsp.biquad(dsp.pink(n, seed + 5), "bp", 160.0, 0.6) * (0.6 + crackle_am)
    out += (fire * burner * 0.5)[:, None] * np.array([0.5, 0.5])
    lights = np.nonzero(np.diff(lit) > 0)[0] + 1
    thump_len = int(0.5 * SR)
    tt = np.arange(thump_len) / SR
    thump = (np.sin(2 * np.pi * 48 * tt) * 0.9 + dsp.biquad(dsp.white(thump_len, seed + 6), "lp", 400.0)) * np.exp(-tt / 0.12)
    for i in lights:
        m = min(thump_len, n - i)
        out[i:i + m] += (thump[:m] * 0.6)[:, None]

    # --- air -------------------------------------------------------------------------------
    air = np.stack([dsp.biquad(dsp.pink(n, seed + s), "bp", lerp(600.0, 2200.0, speed_n), 0.5) for s in (7, 8)], 1)
    out += air * (speed_n ** 1.7 * 0.10)[:, None]

    if GEARED:
        shiftfx.mix(out, c, level=0.35)                                   # hard shifts only
        out *= dsp.one_pole(np.where(c["cut"] > 0.5, 0.35, 1.0), 0.004)[:, None]   # rev limiter

    for ch in range(2):
        out[:, ch] = dsp.biquad(out[:, ch], "hp", 28.0)
    return np.tanh(out * 1.3) / 1.3
