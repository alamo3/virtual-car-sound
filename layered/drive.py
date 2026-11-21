"""A standard test drive for auditioning sounds: a rough IONIQ 5 driven by a script, its data
handled as the app handles the Pi's (whole km/h, smoothed), through the DCT. Gives every
control signal a sound layer might follow, at CONTROL_HZ."""
from __future__ import annotations

import numpy as np

from dct import Dct

CONTROL_HZ = 200          # control signals are computed this often, then interpolated to audio
PI_HZ = 25                # the car's data arrives this often


def car_accel(v, pedal, brake):
    """m/s^2: traction- then power-limited drive, drag, one-pedal regen, brakes."""
    drive = pedal ** 0.8 * min(3.9, 160000.0 / 2100.0 / max(v, 1.0))
    regen = 1.7 * max(0.0, 1 - pedal / 0.06) if v > 0.5 else 0.0
    return drive - 0.15 - 0.0004 * v * v - regen - brake


def standard_script(t):
    """~50 s: idle, gentle pull and cruise, kickdown and a hard pull, lift, brake, stop, idle,
    then a hard launch from rest and lift. (pedal 0..1, brake m/s^2)"""
    steps = [(3, 0, 0), (11, 0.25, 0), (17, 0.15, 0), (25, 1.0, 0), (27, 0.0, 0),
             (32, 0.0, 3.5), (34, 0, 0), (41, 1.0, 0), (44, 0.0, 0), (49, 0.0, 3.0), (52, 0, 0)]
    for end, p, b in steps:
        if t < end:
            return p, b
    return 0.0, 0.0


def run(script=standard_script, seconds=52.0, idle=900.0, redline=7000.0):
    dt = 1.0 / CONTROL_HZ
    box = Dct(idle=idle, redline=redline)
    n = int(seconds * CONTROL_HZ)
    keys = ("t", "pedal", "kmh", "accel", "motor_rpm", "rpm", "gear", "load", "throttle", "shifting",
            "overrun", "sport", "limiter", "cut", "engaged", "shifts", "downshifts", "lift_offs", "brake")
    out = {k: np.zeros(n) for k in keys}
    v = 0.0
    pedal = 0.0
    last_raw = smooth = -1.0
    acc = 0.0
    kmh = 0.0
    per_pi = CONTROL_HZ // PI_HZ
    for i in range(n):
        t = i * dt
        want, brake = script(t)
        pedal += max(-5 * dt, min(5 * dt, want - pedal))     # a foot moves in ~0.2 s
        v = max(0.0, v + car_accel(v, pedal, brake) * dt)
        if i % per_pi == 0:                                   # a new reading from the Pi
            raw = round(v * 3.6)
            pdt = per_pi * dt
            if last_raw >= 0:
                acc += ((raw - last_raw) / 3.6 / pdt - acc) * min(1, pdt / 0.3)
                smooth += (raw - smooth) * min(1, pdt / 0.15)
            else:
                smooth = raw
            last_raw = raw
            kmh = 0.0 if raw < 0.5 else max(0.0, smooth + acc * 3.6 * 0.15)
        st = box.step(dt, kmh, pedal, motor_rpm=int(v * 3.6 * 77.5))
        row = dict(t=t, pedal=pedal, kmh=kmh, accel=acc, motor_rpm=v * 3.6 * 77.5, brake=brake, **{
            k: float(st[k]) for k in ("rpm", "gear", "load", "throttle", "shifting", "overrun", "sport",
                                      "limiter", "cut", "engaged", "shifts", "downshifts", "lift_offs")})
        for k in keys:
            out[k][i] = row[k]
    return out


def to_audio_rate(ctrl: dict, sr: int) -> dict:
    """Linear interpolation of every control signal to audio rate (counters stay stepped)."""
    n = int(len(ctrl["t"]) * sr / CONTROL_HZ)
    ta = np.arange(n) / sr
    res = {}
    for k, v in ctrl.items():
        if k in ("shifts", "downshifts", "lift_offs", "gear", "cut", "limiter"):
            idx = np.minimum((ta * CONTROL_HZ).astype(int), len(v) - 1)
            res[k] = v[idx]
        else:
            res[k] = np.interp(ta, ctrl["t"], v)
    return res
