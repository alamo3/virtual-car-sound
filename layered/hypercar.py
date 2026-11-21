"""Futuristic hypercar: an electric performance voice in layers. The aggressive tone follows the
virtual DCT's revs (so it shifts); underneath, the car's REAL motor sings its own orders and
never shifts -- the two together read as an electric hypercar with a gearbox, not a sample.

Layers
  core      the voice: a detuned stack (saw + pulse) an octave pair apart, through a resonant
            low-pass that opens with load and revs; slight stereo detune for width
  edge      FM metal under load: a carrier with an inharmonic modulator, index rising with
            load and revs -- the snarl when it's pushed, gone when cruising
  sub       a sine an octave under the core, for weight (cabin-friendly, not boomy)
  motor     the real motor's orders (from the car's own motor rpm): a pure high whine and
            its neighbour order, louder under torque and under regen
  spool     a high, breathy whistle that rises with revs at high load (turbine-like)
  air       road / wind rush rising with speed
  shift     the regular hydraulic shift sound on every shift; the core dips through it
  regen     off the pedal while slowing: the motor whine descends with speed, a soft pad
            underneath
"""
from __future__ import annotations

import numpy as np

import dsp
import shiftfx
from dsp import SR, lerp

IDLE, REDLINE = 1000.0, 9000.0
CORE_ORDER = 1.0          # core pitch = revs/60 * this (Hz): 17 Hz idle .. 150 Hz redline...
CORE_OCTAVE = 2.0         # ...played an octave up: 33 .. 300 Hz
MOTOR_ORDERS = (8.0, 12.0)


def render(c: dict, seed: int = 2) -> np.ndarray:
    n = len(c["rpm"])
    rpm = np.maximum(c["rpm"], IDLE)
    rpm_n = np.clip((rpm - IDLE) / (REDLINE - IDLE), 0, 1)
    load = np.clip(c["load"], 0, 1)
    overrun = np.clip(c["overrun"], 0, 1)
    kmh = c["kmh"]
    speed_n = np.clip(kmh / 160, 0, 1)
    out = np.zeros((n, 2))

    # --- core ----------------------------------------------------------------------------
    f = rpm / 60 * CORE_ORDER * CORE_OCTAVE
    voice = np.zeros((n, 2))
    for detune, pan in ((-6, 0.75), (0, 0.5), (+7, 0.25)):          # cents, L share
        ratio = 2 ** (detune / 1200)
        tone = 0.6 * dsp.saw_bl(f * ratio) + 0.8 * dsp.pulse_bl(f * ratio * 2.0, 0.35)   # saw + pulse an octave up
        voice[:, 0] += tone * pan
        voice[:, 1] += tone * (1 - pan)
    cutoff = lerp(350.0, 4200.0, 0.25 * rpm_n + 0.75 * load * (0.4 + 0.6 * rpm_n))
    for ch in range(2):
        voice[:, ch] = dsp.biquad(voice[:, ch], "lp", cutoff, 2.2)
    core_level = (0.12 + 0.88 * load ** 1.2) * (0.6 + 0.4 * rpm_n) * (1 - 0.85 * overrun)
    core_level *= 1 - 0.7 * np.clip(c["shifting"], 0, 1)
    out += voice * dsp.one_pole(core_level, 0.015)[:, None] * 0.45

    # --- edge: FM metal under load -------------------------------------------------------
    mod = np.sin(2 * np.pi * dsp.phase(f * 3.41))
    index = 4.0 * load ** 1.5 * (0.3 + 0.7 * rpm_n)
    edge = np.sin(2 * np.pi * dsp.phase(f * 2.0) + index * mod)
    edge = dsp.biquad(edge, "hp", 300.0) * dsp.one_pole(load ** 2 * (0.2 + 0.8 * rpm_n), 0.03) * 0.22
    out[:, 0] += edge * 0.55
    out[:, 1] += edge * 0.45

    # --- sub -------------------------------------------------------------------------------
    sub = np.sin(2 * np.pi * dsp.phase(f / 2)) * dsp.one_pole((0.25 + 0.75 * load) * (1 - 0.6 * overrun), 0.05) * 0.35
    out += sub[:, None] * 0.5

    # --- the real motor -----------------------------------------------------------------
    motor_hz = c["motor_rpm"] / 60
    torque = np.clip(load + 0.8 * overrun * speed_n, 0, 1)           # drive or regen
    whine = np.zeros(n)
    for order, lvl in zip(MOTOR_ORDERS, (1.0, 0.45)):
        whine += lvl * np.sin(2 * np.pi * dsp.phase(motor_hz * order))
    whine *= dsp.one_pole(np.clip(kmh / 8, 0, 1) * (0.25 + 0.75 * torque), 0.05) * 0.06
    out[:, 0] += whine * 0.5
    out[:, 1] += whine * 0.5

    # --- spool whistle ---------------------------------------------------------------------
    whistle = np.sin(2 * np.pi * dsp.phase(f * 11.0)) + 0.4 * dsp.biquad(dsp.white(n, seed), "bp", f * 11.0, 6.0)
    whistle *= dsp.one_pole(rpm_n ** 2 * load, 0.08) * 0.05
    out[:, 0] += whistle * 0.4
    out[:, 1] += whistle * 0.6

    # --- air -------------------------------------------------------------------------------
    air = np.stack([dsp.biquad(dsp.pink(n, seed + s), "bp", lerp(500.0, 1800.0, speed_n), 0.6) for s in (1, 2)], 1)
    out += air * (speed_n ** 1.6 * 0.12)[:, None]

    # --- shifts: the regular hydraulic shift sound (as the app plays it) -------------------
    shiftfx.mix(out, c, level=0.35)

    # --- regen: a soft pad while slowing off the pedal --------------------------------------
    pad_f = lerp(110.0, 220.0, speed_n)
    pad = sum(np.sin(2 * np.pi * dsp.phase(pad_f * r)) for r in (1.0, 1.5, 2.0)) / 3
    pad *= dsp.one_pole(overrun * np.clip(kmh / 20, 0, 1), 0.3) * 0.06
    out += pad[:, None] * 0.5

    for ch in range(2):
        out[:, ch] = dsp.biquad(out[:, ch], "hp", 30.0)
    return np.tanh(out * 1.3) / 1.3
