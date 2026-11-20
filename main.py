import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
from signalflow import *
import functools
import signal
import threading
from dataclasses import dataclass

import fx
import geared
import jet
import loopengine
import voices

MAX_PEDAL = 1.0
MAX_MOTOR_RPM = 5000

SOURCE_FILE = "jetson.mp3"   # steady ~136 Hz Jetson hum
LOOP_START_SEC = 0.5         # skip the fade-in at the start of the file
LOOP_SEARCH_SEC = (13.0, 13.8)
CROSSFADE_SEC = 0.3

# RPM -> playback rate (1.0 = original pitch). Rate changes pitch AND wobble speed together,
# which is what a real motor sound does as it spins faster.
#
# There is no separate vehicle-speed input: with a single-speed reduction gear, shaft RPM
# and road speed are locked together by a fixed ratio, so RPM already *is* the speed cue.
#
# 1.0 is the ceiling, so nothing ever plays faster than the recording does. Pitch only climbs
# from rate 0.62, which keeps the whine (and its wobble) unhurried across the whole range.
MIN_RATE = 0.62  # at 0 RPM
MAX_RATE = 1.0   # at MAX_MOTOR_RPM
PITCH_CURVE = 0.9  # <1 lifts low-RPM pitch, >1 flattens it; 1.0 is linear in RPM

# --- Powertrain (desk mode): pedal is a torque request, road speed is the result ------
# Modelled in real units, roughly an IONIQ 5 RWD: traction-limited pull at low speed, then
# power-limited, against rolling resistance, road load and aero. Top speed is independent
# of the pitch range (it used to be tied to MAX_MOTOR_RPM, which capped the sim near
# 60 km/h). Off the pedal, regen adds a near-constant braking force, so the car comes to a
# stop in finite time instead of crawling through the low-speed end of every sound.
LAUNCH_ACCEL_KMH_S = 14.0    # traction limit at full pedal (~3.9 m/s^2)
POWER_KMH2_S = 1040.0        # power limit: accel = this / speed (~160 kW for ~2 t)
VMAX_KMH = 185.0             # speed limiter
ROAD_LOAD_PER_SEC = 0.02     # speed-proportional load (tyres, driveline), 1/s
AERO_KMH_S_AT_100 = 1.2      # aero deceleration at 100 km/h, grows with speed squared
ROLLING_DECEL_KMH_S = 0.5    # constant rolling resistance, always present
REGEN_DECEL_KMH_S = 6.0      # constant regen braking off-throttle (~1.7 m/s^2, one-pedal feel)
REGEN_PER_SEC = 0.10         # extra speed-proportional regen off-throttle
REGEN_FADE = 0.35            # pedal travel over which regen fades out

# Load layer: an octave-up copy of the hum blended in by motor current (not raw pedal)
LOAD_LAYER_PITCH = 2.0   # 2.0 = one octave above the base layer
LOAD_LAYER_LEVEL = 1.15  # mix level at full pedal, mid-range (current falloff shapes the ends)
CURRENT_FALLOFF = 0.45   # current at full pedal drops this much from 0 RPM to MAX_MOTOR_RPM

# Gear mesh: the reduction gear's tooth-mesh tone, a high order of shaft speed. This is the
# "there is a gearbox in here" cue -- a discrete tone rather than another harmonic of the
# hum, so it reads as machinery instead of timbre.
GEAR_MESH_TEETH = 24     # 3000 RPM -> 1.2 kHz
GEAR_MESH_BASE = 0.012   # level while spinning unloaded
GEAR_MESH_LOAD = 0.035   # extra level at full current
GEAR_MESH_H2 = 0.35      # 2nd-order (2x mesh) level; real mesh tones are rich in it
GEAR_MESH_LEVEL = 1.0    # layer master gain; set to 0 to disable

# --- Wobble ---------------------------------------------------------------------------
# Resampling ties warble speed to pitch: play the loop faster and its flutter speeds up too.
# So the recording's own modulation is flattened out of the loop and re-applied here as an
# LFO at a rate we choose, which holds still no matter what the motor is doing.
WOBBLE_FLATTEN = 0.9    # 0 = keep the recording's flutter, 1 = perfectly steady
WOBBLE_HZ = 0.5         # warble rate (2 s period), independent of RPM
WOBBLE_AMP = 0.10       # tremolo depth (+-10% level)
WOBBLE_PITCH = 0.004    # vibrato depth (+-0.4% rate = about +-7 cents)

# Tip-in swell: brief boost when the pedal is pressed quickly
SWELL_THRESHOLD = 1.0    # pedal speed (full travel per second) below which nothing happens
SWELL_FULL_AT = 4.0      # pedal speed that gives the maximum swell
SWELL_DECAY_SEC = 1.2  # how long the swell takes to fade (time constant)
SWELL_GAIN = 0.4         # +40% volume at full swell
SWELL_CUTOFF_HZ = 6000.0 # extra brightness at full swell
SWELL_LOAD = 0.6         # extra load-layer mix at full swell

MASTER_GAIN = 1.0

# --- "Car move" voice (default) ---------------------------------------------------------
# Synthesised from measurements of jetson-car-move.mp3 rather than resampled from a loop,
# because that clip sweeps nearly two octaves -- far more than a looped recording can be
# sped up or slowed down without its flutter turning into a blur or a crawl.
# Its recipe, constant across the whole glide:
#   fundamental, plus harmonics 2-5 at -15/-19/-25/-22 dB
#   a "whistle" partial at 5.88x the fundamental (slightly flat of the 6th), -12 dB,
#     warbling hard: +-74 cents at ~13.5 Hz, level +-58 % at ~12.6 Hz
#   the body wobbles less: +-34 cents at ~14 Hz, level +-28 % at ~8 Hz
#   pitch: ~136 Hz at the bottom of its glide, ~520 Hz while flying; glide is ~linear in Hz
CM_F_IDLE = 136.0            # Hz at standstill
CM_F_TOP = 520.0             # Hz at MAX_MOTOR_RPM
# Levels as set here are ~4 dB above the measured peaks: vibrato smears the upper
# partials, so they need more level going in to measure like the clip coming out.
CM_HARMONICS = (1.0, 0.28, 0.18, 0.09, 0.13)       # measured h1..h5: 0, -15, -19, -25, -22 dB
CM_WHISTLE_RATIO = 5.88
CM_WHISTLE_LEVEL = 0.42      # measures -12 dB re the fundamental once smeared
CM_WHISTLE_LOAD = 0.25       # extra whistle level with motor current (pedal)
CM_VIB_HZ, CM_VIB_CENTS = 14.0, 34.0
CM_TREM_HZ, CM_TREM_DEPTH = 8.0, 0.2   # clip: 28 % while holding, 13 % in the glide
CM_WVIB_HZ, CM_WVIB_CENTS = 13.5, 74.0
CM_WTREM_HZ, CM_WTREM_DEPTH = 12.6, 0.58
CM_CUTOFF_MIN = 4500.0       # keeps the whistle audible even when coasting

# Idle hum: at standstill the motor is "ready", not off -- it plays at the bottom of the
# pitch range, quieter, and the pedal still brightens it (e.g. pressing it on the brake).
MUTE_BELOW_RPM = 30.0    # below this (and no pedal) the car counts as standing still
# Standstill: instead of cutting off the moment the car stops, the voice keeps idling for
# STANDSTILL_HOLD_SEC, then fades over STANDSTILL_FADE_SEC to the idle-hum level (0 =
# silence, the default). Moving or touching the pedal brings it back in START_FADE_SEC.
AUDIO_CHECK_SEC = 5.0     # headless: how often to check the speaker is still there
TICK_ERRORS_SWITCH_VOICE = 50    # ~1 s of failing ticks: fall back to the default voice
TICK_ERRORS_RESTART = 500        # ~10 s: give up and let systemd restart the program
STANDSTILL_HOLD_SEC = 3.0
STANDSTILL_FADE_SEC = 2.5
START_FADE_SEC = 0.15
IDLE_LEVEL = 0.5         # standstill volume relative to the normal coasting level
IDLE_RAMP_RPM = 500.0    # idle level blends into the normal level over this much rpm
MESH_FADE_RPM = 300.0    # gear-mesh tone fades out below this: stationary gears don't whine

# --- Live car input (--car) -------------------------------------------------------------
# Pedal and speed arrive together at ~30 Hz from the VCU (obd.IoniqReader). Speed is whole
# km/h, so it is smoothed into a glide before becoming pitch; the pedal gets a lighter touch
# so tip-ins still register as fast presses.
CAR_RPM_MAX = 10000          # pitch-range top in car mode (~130 km/h at the ratio below)
RPM_PER_KMH = 77.0           # starting guess: 10.65:1 reduction, ~2.29 m tyre circumference.
                             # Refined live from the battery module's motor rpm.
RATIO_LEARN = 0.1            # how far each rpm sample moves the ratio estimate
RATIO_MIN_KMH = 15.0         # below this, whole-km/h rounding makes the ratio too noisy
CAR_SPEED_SMOOTH_SEC = 0.15
CAR_PEDAL_SMOOTH_SEC = 0.05



def _modulation_envelope(mono, sr, window_sec=0.012):
    """Slow amplitude envelope of the loop, normalised to mean 1.

    Wrapped circularly so the envelope is loop-periodic and can't taper at the seam.
    12 ms tracks flutter down to ~80 Hz without flattening the 137 Hz hum itself.
    """
    k = max(1, int(window_sec * sr))
    padded = np.concatenate([mono[-k:], mono, mono[:k]])
    env = np.convolve(np.abs(padded), np.ones(k) / k, mode="same")[k:k + len(mono)]
    return env / max(env.mean(), 1e-9)


def flatten_wobble(loop, sr, amount=WOBBLE_FLATTEN):
    """Divide out the recording's amplitude modulation, keeping the mean level.

    `amount` blends between untouched (0) and perfectly steady (1), so a trace of the
    original texture can be kept for character.
    """
    env = np.maximum(_modulation_envelope(loop.mean(axis=0), sr), 0.2)
    rms_before = np.sqrt(np.mean(loop ** 2))
    out = loop / env ** amount
    return out * (rms_before / max(np.sqrt(np.mean(out ** 2)), 1e-9))


def build_loop_buffer(path=SOURCE_FILE):
    """Cut a seamless loop out of the steady part of the file.

    Picks a loop end whose waveform best matches the loop start, then blends the
    seam with an equal-power crossfade so there's no click or level dip.
    """
    src = Buffer(path)
    sr = int(src.sample_rate)
    data = np.array(src.data)
    mono = data.mean(axis=0)

    start = int(LOOP_START_SEC * sr)
    win = int(0.05 * sr)
    ref = mono[start:start + win]
    best_end, best_corr = None, -2.0
    for end in range(int(LOOP_SEARCH_SEC[0] * sr), int(LOOP_SEARCH_SEC[1] * sr), 4):
        seg = mono[end:end + win]
        corr = np.dot(ref, seg) / (np.linalg.norm(ref) * np.linalg.norm(seg) + 1e-12)
        if corr > best_corr:
            best_end, best_corr = end, corr

    fade = int(CROSSFADE_SEC * sr)
    loop = data[:, start:best_end].copy()
    tail = data[:, best_end:best_end + fade]
    t = np.linspace(0.0, np.pi / 2, fade)
    loop[:, :fade] = loop[:, :fade] * np.sin(t) + tail * np.cos(t)

    if WOBBLE_FLATTEN > 0:
        loop = flatten_wobble(loop, sr)

    buf = Buffer(loop.shape[0], loop.shape[1], loop.tolist())
    buf.sample_rate = sr  # must be set, otherwise it plays at the graph rate (wrong pitch)
    return buf


def create_ev_sound(loop_buffer):
    """Looping base layer + load layer + gear-mesh tone, driven by smoothed targets."""
    targets = {
        "rate": Constant(1.0),
        "cutoff": Constant(4000.0),
        "load_mix": Constant(0.0),
        "gain": Constant(0.0),
        "mesh_hz": Constant(1200.0),
        "mesh_mix": Constant(0.0),
    }
    rate = Smooth(targets["rate"], smooth=0.9995)
    cutoff = Smooth(targets["cutoff"], smooth=0.999)
    load_mix = Smooth(targets["load_mix"], smooth=0.9995)
    gain = Smooth(targets["gain"], smooth=0.999)
    # Mesh tracks shaft speed directly, so it glides rather than steps between frames
    mesh_hz = Smooth(targets["mesh_hz"], smooth=0.998)
    mesh_mix = Smooth(targets["mesh_mix"], smooth=0.999)

    # Wobble: a slow, deliberate warble at a fixed rate -- independent of playback rate, so
    # pitch and warble speed no longer move together. Tremolo for level, a touch of vibrato
    # for pitch, slightly out of phase so it doesn't read as a plain sine pump.
    tremolo = SineLFO(WOBBLE_HZ, min=1.0 - WOBBLE_AMP, max=1.0 + WOBBLE_AMP)
    vibrato = SineLFO(WOBBLE_HZ, min=1.0 - WOBBLE_PITCH, max=1.0 + WOBBLE_PITCH, phase=0.25)

    # Base layer: the hum at RPM-driven pitch
    base = BufferPlayer(loop_buffer, rate=rate * vibrato, loop=1)

    # Load layer: same loop an octave up (locked to the base pitch, so it reads as a
    # harmonic rather than a second sound). High-passed so it doesn't double the bass;
    # it fades in with motor current to make the motor sound like it's working harder.
    load = BufferPlayer(loop_buffer, rate=rate * LOAD_LAYER_PITCH * vibrato, loop=1)
    load = SVFilter(load, "high_pass", cutoff=250.0, resonance=0.0)

    # Gear mesh: sine at the tooth-mesh order plus a quieter 2nd order, so it sits outside
    # the hum's harmonic series and its pitch rises with speed like a real gearbox.
    mesh = SineOscillator(mesh_hz) + GEAR_MESH_H2 * SineOscillator(mesh_hz * 2.0)

    mix = base + load * load_mix + mesh * mesh_mix
    filtered = SVFilter(mix, "low_pass", cutoff=cutoff, resonance=0.0)
    output = filtered * gain * tremolo
    return output, targets


def create_carmove_sound():
    """Additive voice measured from jetson-car-move.mp3 (see the CM_* constants)."""
    targets = {
        "freq": Constant(CM_F_IDLE),
        "cutoff": Constant(CM_CUTOFF_MIN),
        "load_mix": Constant(0.0),
        "gain": Constant(0.0),
    }
    freq = Smooth(targets["freq"], smooth=0.9995)
    cutoff = Smooth(targets["cutoff"], smooth=0.999)
    load_mix = Smooth(targets["load_mix"], smooth=0.9995)
    gain = Smooth(targets["gain"], smooth=0.999)

    def cents(c):
        return 2.0 ** (c / 1200.0)

    # Body: harmonics 1-5 sharing one vibrato and a slower tremolo
    vib = SineLFO(CM_VIB_HZ, min=1.0 / cents(CM_VIB_CENTS), max=cents(CM_VIB_CENTS))
    trem = SineLFO(CM_TREM_HZ, min=1.0 - CM_TREM_DEPTH, max=1.0 + CM_TREM_DEPTH)
    body = None
    for h, amp in enumerate(CM_HARMONICS, start=1):
        osc = SineOscillator(freq * vib * float(h)) * amp
        body = osc if body is None else body + osc
    body = body * trem

    # Whistle: the slightly-flat 6th partial with its own, much stronger warble. The two
    # LFO rates (13.5 vs 14 Hz) drift against each other, so the flutter never sounds
    # like a plain repeating pulse.
    wvib = SineLFO(CM_WVIB_HZ, min=1.0 / cents(CM_WVIB_CENTS), max=cents(CM_WVIB_CENTS),
                   phase=0.3)
    wtrem = SineLFO(CM_WTREM_HZ, min=1.0 - CM_WTREM_DEPTH, max=1.0 + CM_WTREM_DEPTH,
                    phase=0.6)
    whistle = SineOscillator(freq * wvib * CM_WHISTLE_RATIO) * wtrem * (CM_WHISTLE_LEVEL + load_mix)

    mix = (body + whistle) * 0.33       # ~same loudness as the other voices, no clipping
    filtered = SVFilter(mix, "low_pass", cutoff=cutoff, resonance=0.0)
    return filtered * gain, targets


class StandstillFade:
    """Level 0..1 for the voice: 1 while moving; after stopping, held for a moment, then
    faded to `hum` (0 = silent). Starts silent, so a parked car powering on stays quiet."""

    def __init__(self):
        self.level = 0.0
        self.stopped_for = 1e9

    def step(self, moving: bool, dt: float, hum: float = 0.0) -> float:
        if moving:
            self.stopped_for = 0.0
            self.level += (1.0 - self.level) * min(1.0, dt / START_FADE_SEC)
        else:
            self.stopped_for += dt
            if self.stopped_for > STANDSTILL_HOLD_SEC and self.level > hum:
                self.level = max(hum, self.level - dt / STANDSTILL_FADE_SEC)
        return self.level


class TipInSwell:
    """Turns quick pedal presses into a short 0..1 swell that fades out.

    Only pressing triggers it (lifting off doesn't), and slow pedal movement below
    SWELL_THRESHOLD is ignored so steady driving isn't affected.
    """

    def __init__(self):
        self.prev_pedal = None
        self.value = 0.0

    def update(self, pedal, dt):
        pedal_ratio = max(0.0, min(1.0, pedal / MAX_PEDAL))
        if self.prev_pedal is None or dt <= 0:
            self.prev_pedal = pedal_ratio
            return self.value

        speed = (pedal_ratio - self.prev_pedal) / dt  # full travel per second, + = pressing
        self.prev_pedal = pedal_ratio

        # Fade out the current swell, then let a new fast press push it back up
        self.value *= np.exp(-dt / SWELL_DECAY_SEC)
        if speed > SWELL_THRESHOLD:
            hit = min(1.0, (speed - SWELL_THRESHOLD) / (SWELL_FULL_AT - SWELL_THRESHOLD))
            self.value = max(self.value, hit)
        return self.value


class Powertrain:
    """Desk-mode car: integrates pedal into road speed, so pitch is a consequence.

    Full pedal reaches 100 km/h in ~8.5 s and tops out around 155 km/h; a steady 30 %
    pedal settles near 90 km/h. Lift off and regen pulls it down to a stop in a few
    seconds. `rpm` is motor rpm (speed x RPM_PER_KMH), which is what the voices follow.
    """

    def __init__(self, rpm=0.0):
        self.speed = float(rpm) / RPM_PER_KMH

    @property
    def rpm(self):
        return self.speed * RPM_PER_KMH

    @rpm.setter
    def rpm(self, value):
        self.speed = max(0.0, float(value)) / RPM_PER_KMH

    def accel(self, pedal_ratio, v):
        """km/h per second at speed v for a pedal position 0..1."""
        drive = min(LAUNCH_ACCEL_KMH_S, POWER_KMH2_S / max(v, 1.0)) * pedal_ratio
        drive *= min(1.0, max(0.0, (VMAX_KMH - v) / 8.0))          # limiter taper
        regen = 0.0 if REGEN_FADE <= 0 else max(0.0, 1.0 - pedal_ratio / REGEN_FADE)
        drag = ROAD_LOAD_PER_SEC * v + AERO_KMH_S_AT_100 * (v / 100.0) ** 2
        if v > 0:
            drag += ROLLING_DECEL_KMH_S + regen * (REGEN_DECEL_KMH_S + REGEN_PER_SEC * v)
        return drive - drag

    def step(self, pedal, dt):
        if dt <= 0.0:
            return self.rpm
        pedal_ratio = max(0.0, min(1.0, pedal / MAX_PEDAL))
        # Sub-step so a stalled frame can't overshoot into a discontinuous pitch jump
        steps = max(1, int(np.ceil(dt / 0.008)))
        h = dt / steps
        for _ in range(steps):
            self.speed = max(0.0, self.speed + self.accel(pedal_ratio, self.speed) * h)
        return self.rpm


class CarInput:
    """Turns the reader's latest samples into smooth (rpm, pedal) for the sound.

    Pitch comes from speed x rpm-per-km/h rather than from the battery module's rpm
    directly: speed updates ~6x more often. The rpm samples only calibrate the ratio.
    If the link goes quiet, the pedal falls to 0 so the sound coasts instead of sticking.
    """

    def __init__(self, reader, ratio=RPM_PER_KMH):
        self.reader = reader
        self.ratio = ratio
        self.speed = 0.0
        self.pedal = 0.0
        self._rpm_seen = 0.0

    def step(self, dt):
        r = self.reader
        fresh = r.fresh and r.speed is not None and r.pedal is not None
        target_speed = r.speed if fresh else self.speed
        target_pedal = (r.pedal / 100.0) * MAX_PEDAL if fresh else 0.0
        if dt > 0:
            self.speed += (target_speed - self.speed) * (1.0 - np.exp(-dt / CAR_SPEED_SMOOTH_SEC))
            self.pedal += (target_pedal - self.pedal) * (1.0 - np.exp(-dt / CAR_PEDAL_SMOOTH_SEC))

        # New motor-rpm sample taken while moving with fresh speed -> nudge the ratio
        if (r.rpm_at != self._rpm_seen and r.motor_rpm and fresh
                and r.speed >= RATIO_MIN_KMH and abs(r.rpm_at - r.updated_at) < 0.15):
            self.ratio += RATIO_LEARN * (r.motor_rpm / r.speed - self.ratio)
        self._rpm_seen = r.rpm_at
        return self.speed * self.ratio, self.pedal, fresh


def apply_controls(targets, rpm, pedal, swell=0.0, active=True):
    """Map motor RPM, pedal position and tip-in swell (0..1) to sound targets.

    active=False silences everything (car not answering); otherwise the motor hums even
    at standstill.
    """
    motor_ratio = max(0.0, min(1.0, rpm / MAX_MOTOR_RPM))
    pedal_ratio = max(0.0, min(1.0, pedal / MAX_PEDAL))
    curved_rpm = motor_ratio ** PITCH_CURVE  # perceptually smoother pitch rise

    # Motor current: pedal is the request, but back-EMF means the same pedal pulls less
    # current as speed rises. This, not pedal demand, is what loads the sound.
    current = pedal_ratio * (1.0 - CURRENT_FALLOFF * curved_rpm)

    if "freq" in targets:
        _apply_carmove(targets, rpm, pedal_ratio, curved_rpm, current, swell, active,
                       targets.get("pitch_scale", 1.0))
        return

    # RPM -> pitch
    targets["rate"].set_value(MIN_RATE + (MAX_RATE - MIN_RATE) * curved_rpm)

    # Pedal -> brightness: coasting is muffled, accelerating opens up the full sound
    targets["cutoff"].set_value(1200.0 + 14000.0 * pedal_ratio ** 1.5 + SWELL_CUTOFF_HZ * swell)

    # Current -> load layer: fades in progressively, and relaxes at high speed where a
    # real motor is under less load for the same pedal
    targets["load_mix"].set_value(LOAD_LAYER_LEVEL * current ** 1.5 + SWELL_LOAD * swell)

    # Gear mesh tone follows shaft speed exactly, and speaks loudest under load
    targets["mesh_hz"].set_value(max(40.0, (rpm / 60.0) * GEAR_MESH_TEETH))
    spinning = min(1.0, max(0.0, rpm) / MESH_FADE_RPM)
    targets["mesh_mix"].set_value(
        GEAR_MESH_LEVEL * (GEAR_MESH_BASE + GEAR_MESH_LOAD * current) * spinning)

    # Pedal + RPM -> volume, with an idle hum at standstill that blends into the normal
    # level as the car pulls away; swell adds a brief push on top
    if not active:
        targets["gain"].set_value(0.0)
    else:
        coast_floor = 0.35
        idle = IDLE_LEVEL + (1.0 - IDLE_LEVEL) * min(1.0, max(0.0, rpm) / IDLE_RAMP_RPM)
        targets["gain"].set_value(
            (coast_floor + (1.0 - coast_floor) * pedal_ratio) * (0.5 + 0.5 * curved_rpm)
            * idle * (1.0 + SWELL_GAIN * swell) * MASTER_GAIN
        )


def _volume(rpm, pedal_ratio, curved_rpm, swell, active):
    """Shared loudness law: idle hum at standstill, rising with pedal and speed."""
    if not active:
        return 0.0
    coast_floor = 0.35
    idle = IDLE_LEVEL + (1.0 - IDLE_LEVEL) * min(1.0, max(0.0, rpm) / IDLE_RAMP_RPM)
    return ((coast_floor + (1.0 - coast_floor) * pedal_ratio) * (0.5 + 0.5 * curved_rpm)
            * idle * (1.0 + SWELL_GAIN * swell) * MASTER_GAIN)


def _apply_carmove(targets, rpm, pedal_ratio, curved_rpm, current, swell, active,
                   pitch_scale=1.0):
    # Speed -> pitch across the clip's own range, linear in Hz like its glide
    targets["freq"].set_value((CM_F_IDLE + (CM_F_TOP - CM_F_IDLE) * curved_rpm) * pitch_scale)
    # Pedal opens the top end and pushes the whistle; never muffled below the whistle
    targets["cutoff"].set_value(CM_CUTOFF_MIN + 10000.0 * pedal_ratio ** 1.5
                                + SWELL_CUTOFF_HZ * swell)
    targets["load_mix"].set_value(CM_WHISTLE_LOAD * current + 0.5 * SWELL_LOAD * swell)
    targets["gain"].set_value(_volume(rpm, pedal_ratio, curved_rpm, swell, active))


# --- Voice registry ----------------------------------------------------------------------
# The two original voices wrapped to the same interface as voices.py, so the GUI can list
# them all together and switch live.

class JetsonsVoice(voices.Voice):
    name = "Jetsons"
    PITCH = 1.0

    def build(self):
        out, t = create_carmove_sound()
        t["pitch_scale"] = self.PITCH
        return out, t

    def update(self, st):
        apply_controls(self.targets, st.motor_ratio * MAX_MOTOR_RPM, st.pedal * MAX_PEDAL,
                       st.swell, st.active)
        for key in ("freq", "rate"):            # virtual gears
            if key in self.targets:
                self.targets[key].set_value(self.targets[key].value * st.pitch_mul)
        return ""


class JetsonsLowVoice(JetsonsVoice):
    """Same recipe one octave down: 68 Hz at a standstill, 260 Hz flying."""
    name = "Jetsons Low"
    PITCH = 0.5


class HumVoice(voices.Voice):
    name = "Hum (jetson.mp3 loop)"

    def build(self):
        out, t = create_ev_sound(build_loop_buffer())
        return out * 2.0, t          # match the other voices' loudness

    def update(self, st):
        apply_controls(self.targets, st.motor_ratio * MAX_MOTOR_RPM, st.pedal * MAX_PEDAL,
                       st.swell, st.active)
        for key in ("freq", "rate"):            # virtual gears
            if key in self.targets:
                self.targets[key].set_value(self.targets[key].value * st.pitch_mul)
        return ""


VOICES = {v.name: v for v in (jet.SupersonicJet, geared.GearedEngine, voices.HypercarResponse, voices.HypercarDrive, voices.Slipstream, voices.PrecisionDrive, voices.RecordingI3, voices.JetsonsI3, voices.DeepDrone, voices.CinematicChord,
                              voices.CinematicDrive, voices.ActionPulse, voices.TrailerBrass,
                              voices.StringsChoir, voices.Cathedral, voices.GrandTouring, voices.Dreadnought, voices.JetTurbine, voices.InverterWhine, voices.Maglev,
                              voices.WarpCore, voices.Theremin, voices.HoverTurbine,
                              JetsonsLowVoice,
                              JetsonsVoice, HumVoice,
                              voices.V8Muscle, voices.Inline4Sport)}
import reference_voices as _reference_voices
if _reference_voices.REFERENCE.is_file():
    VOICES.update({v.name: v for v in (_reference_voices.EvolutionReference,
                                       _reference_voices.SupersonicReference)})
import boom_voices as _boom_voices
for _voice in (_boom_voices.BoomEvolution, _boom_voices.BoomSupersonic):
    if _voice.available():
        VOICES[_voice.name] = _voice
# every recording mapped with mapper.py (maps/*.json) becomes a voice too
for _name, _path in voices.RecordingVoice.saved_maps():
    VOICES[_name] = functools.partial(voices.RecordingVoice.from_map, _path)
import resynth as _resynth
for _name, _path in _resynth.profiles():           # resynth.py recreations of recordings
    VOICES[_name] = functools.partial(_resynth.voice_class(), _path)
import presets as _presets
# game-style loop engines, only where their (local, unshareable) sound sets exist
for _cls in (loopengine.LoopEngineIoniq, loopengine.KitBattalion, loopengine.LoopEngineInterior, loopengine.LoopEngine,
             loopengine.LoopEngineAudi):
    if loopengine.available(_cls.SET):
        VOICES[_cls.name] = _cls
_presets.register(VOICES)          # sound-lab presets (presets/*.json) become voices too
DEFAULT_VOICE = "Jetsons i3 (recording)"




def find_voice(name):
    """Exact voice name, else a unique case-insensitive substring ("v8", "i3 rec")."""
    if name in VOICES:
        return name
    words = name.lower().split()
    hits = [v for v in VOICES if all(w in v.lower() for w in words)]
    if len(hits) == 1:
        return hits[0]
    raise SystemExit(f"voice {name!r} " + ("is ambiguous: " + ", ".join(hits) if hits else
                     "not found") + "\nvoices: " + ", ".join(VOICES))


# --- Car link -----------------------------------------------------------------------------

class CarLink:
    """Owns the OBD adapter: connects in the background, and reconnects if it goes away.

    Nothing here blocks the caller, so audio can start (and the chime play) while the
    adapter is still being found and initialised. `reader` is None until connected.
    """
    RETRY_SEC = 2.0          # between connection attempts
    DEAD_AFTER_SEC = 10.0    # connected but no fresh data this long -> reopen the adapter

    def __init__(self, port=None, rpm_every=4, log=print):
        self.port = port
        self.rpm_every = rpm_every
        self.log = log
        self.reader = None
        self.elm = None
        self.state = "connecting"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="car-link", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _connect(self):
        import obd
        elm = obd.Elm327(self.port).open()
        try:
            elm.init(fast=True, protocol=6)
        except Exception:
            elm.close()
            raise
        reader = obd.IoniqReader(elm, rpm_every=self.rpm_every)
        reader.start()
        self.elm, self.reader = elm, reader
        self.log(f"car: adapter on {elm.port}, low-latency USB "
                 f"{'on' if elm.low_latency else 'OFF'}")

    def _drop(self):
        reader, elm = self.reader, self.elm
        self.reader = self.elm = None
        if reader:
            reader.stop()
            reader.join(timeout=1.0)
        if elm:
            try:
                elm.close()
            except Exception:
                pass

    def _run(self):
        while not self._stop.is_set():
            try:
                self._run_once()
            except Exception as e:          # a bug here must not end reconnecting for good
                self.log(f"car: link thread error {e!r}; restarting the link")
                self._drop()
                self._stop.wait(self.RETRY_SEC)
        self._drop()

    def _run_once(self):
        last_err = None
        while not self._stop.is_set():
            if self.reader is None:
                try:
                    self._connect()
                    self.state, last_err = "connected", None
                    seen = time.monotonic()
                except Exception as e:      # no adapter yet, permissions, car asleep...
                    self.state = "connecting"
                    if str(e) != last_err:  # log each distinct problem once
                        self.log(f"car: {e}; retrying every {self.RETRY_SEC:.0f} s")
                        last_err = str(e)
                    self._stop.wait(self.RETRY_SEC)
                    continue
            if self.reader.fresh:
                seen = time.monotonic()
                self.state = "live"
            else:
                self.state = "no data"
                if time.monotonic() - seen > self.DEAD_AFTER_SEC:
                    self.log(f"car: no data for {self.DEAD_AFTER_SEC:.0f} s "
                             f"({self.reader.last_error or 'no reply'}); reopening the adapter")
                    self._drop()
                    continue
            self._stop.wait(0.25)

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)


class _NoData:
    """Stands in for the reader while the adapter is (re)connecting."""
    fresh = False
    speed = pedal = motor_rpm = motor_rpm_signed = battery_current = soc = None
    rpm_at = updated_at = 0.0


class _LinkReader:
    """Lets CarInput read whichever reader the link currently has."""

    def __init__(self, link):
        self.link = link

    def __getattr__(self, name):
        return getattr(self.link.reader or _NoData, name)


# --- Engine: everything between inputs and the speaker, shared by GUI and headless --------

@dataclass
class Settings:
    """What a user can change while it runs (GUI today, a phone app later)."""
    voice: str = DEFAULT_VOICE
    small_speaker: bool = voices.SMALL_SPEAKER
    effects: bool = True
    cruise_quiet: bool = True
    virtual_gears: bool = True
    reverse_test: bool = False
    idle_hum: float = 0.0          # level kept at a standstill after the fade (0 = silent)
    volume: float = 1.0            # master volume (the phone app's slider)


# Settings the phone changes are saved here and restored at the next start, so the car
# comes up the way it was left. Only the fields below are stored.
SETTINGS_FILE = Path.home() / ".config" / "virtual-car-sound" / "settings.json"
SAVED_FIELDS = ("voice", "volume", "effects", "small_speaker", "virtual_gears", "cruise_quiet",
                "idle_hum")


def sd_notify(message: str):
    """Tell systemd how we are (READY=1, WATCHDOG=1, STATUS=...). No-op outside systemd."""
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return
    import socket
    if addr.startswith("@"):
        addr = "\0" + addr[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.connect(addr)
            s.sendall(message.encode())
    except OSError:
        pass


def save_settings(settings, path=None):
    path = Path(path or SETTINGS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps({k: getattr(settings, k) for k in SAVED_FIELDS}, indent=1))
        f.flush()
        os.fsync(f.fileno())               # on the card before the swap (power cuts)
    tmp.replace(path)                      # atomic: a power cut never leaves half a file


def load_settings(settings, path=None, log=print):
    """Apply a saved settings file over `settings` (unknown voices are ignored)."""
    path = Path(path or SETTINGS_FILE)
    if not path.exists():
        return settings
    try:
        saved = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        log(f"settings: ignoring {path}: {e}")
        return settings
    for k in SAVED_FIELDS:
        if k in saved and (k != "voice" or saved[k] in VOICES):
            setattr(settings, k, type(getattr(settings, k))(saved[k]))
    log(f"settings: restored from {path}")
    return settings


class Engine:
    """One tick() per ~20 ms: reads the car (or the desk inputs), drives voice + effects.

    Desk mode inputs: `pedal_position` (0..MAX_PEDAL) and `motor_rpm`; with `rpm_coupled`
    the pedal drives a simulated powertrain and motor_rpm is an output.
    """

    def __init__(self, settings, car_link=None, log=print):
        self.settings = settings
        self.log = log
        self.link = car_link
        self.reader = _LinkReader(car_link) if car_link else None
        self.car = CarInput(self.reader) if car_link else None
        self.pedal_position = 0.5
        self.motor_rpm = 1000.0
        self.rpm_coupled = True
        self.powertrain = Powertrain(self.motor_rpm)
        self.swell = TipInSwell()
        self.standstill = StandstillFade()
        self.events = fx.DriveEvents()
        self.effects = fx.Effects()
        self.built = {}
        self.current = None
        self.speed = 0.0
        self.fresh = True
        self.flags = []
        self.info = ""
        self.message = ""                # last problem worth showing the phone ("" = none)
        self.mute_power_on_until = 0.0   # an early chime was already played (headless boot)
        self._applied = None
        self._last = None

    def chime_now(self, hold_off=30.0):
        """Play the power-on chime immediately and skip the data-triggered one."""
        self.effects.power_on()
        self.mute_power_on_until = time.monotonic() + hold_off

    def select_voice(self, name):
        """Swap voices live. Each is built once and kept, so switching back is instant."""
        if self.current is not None and self.current.name == name:
            return
        if name not in self.built:
            t = time.monotonic()
            try:
                self.built[name] = VOICES[name]()
            except Exception as e:         # missing file, bad preset...: keep playing
                self.log(f"voice {name!r} failed to build: {e!r}")
                self.message = f"Couldn't load {name}: {e}"
                if self.current is not None:
                    self.settings.voice = self.current.name      # stay on what works
                    return
                fallback = DEFAULT_VOICE if name != DEFAULT_VOICE else next(
                    n for n in VOICES if n != name)
                self.settings.voice = fallback
                return self.select_voice(fallback)
            self.built[name].set_small_speaker(self.settings.small_speaker)
            self.log(f"voice: {name} (built in {time.monotonic() - t:.1f} s)")
        self.message = ""
        if self.current is not None:
            self.current.output.stop()
        self.current = self.built[name]
        self.current.output.play()

    def _apply_settings(self):
        s = self.settings
        if s.voice != (self.current.name if self.current else None):
            self.select_voice(s.voice)
        if s.small_speaker != self._applied:
            for v in self.built.values():
                v.set_small_speaker(s.small_speaker)
            self._applied = s.small_speaker

    def tick(self, now=None):
        now = time.monotonic() if now is None else now
        dt = 0.02 if self._last is None else now - self._last
        self._last = now
        s = self.settings
        self._apply_settings()

        fresh = True
        if self.car:
            self.motor_rpm, self.pedal_position, fresh = self.car.step(dt)
            self.powertrain.rpm = self.motor_rpm
        elif self.rpm_coupled:
            self.motor_rpm = self.powertrain.step(self.pedal_position, dt)
        else:
            self.powertrain.rpm = self.motor_rpm
        rpm, r = self.motor_rpm, self.reader
        speed = self.car.speed if self.car else rpm / RPM_PER_KMH
        pedal_n = max(0.0, min(1.0, self.pedal_position / MAX_PEDAL))
        cur = self.current
        ev_voice = cur.kind == "ev"
        # shared virtual gears, for voices that don't run their own gearbox
        fx_gears = s.virtual_gears and ev_voice and not getattr(cur, "own_gears", False)

        # drive events: from the car when it can tell us, from pedal/speed otherwise
        triggers = self.events.step(
            dt, speed, pedal_n, fresh,
            current=r.battery_current if r else None,
            signed_rpm=r.motor_rpm_signed if r else None,
            data_age=(now - r.rpm_at) if r else 1e9,
            reverse_test=s.reverse_test, virtual_gears=fx_gears,
            regen_fade=REGEN_FADE)
        if s.effects:
            for trig in triggers:
                if trig == "power_on":
                    if now >= self.mute_power_on_until:
                        self.effects.power_on()
                elif trig == "launch" and ev_voice:     # engines have their own launch
                    self.effects.launch()
                elif trig in ("up", "down"):
                    self.effects.shift(up=trig == "up")
        ev = self.events
        self.effects.update(dt, regen=ev.regen if ev_voice else 0.0, reverse=ev.reverse,
                            enabled=s.effects and fresh,
                            speed_ratio=max(0.0, min(1.0, rpm / MAX_MOTOR_RPM)))
        # standstill: idle on for a moment after stopping, then fade (not a hard cut)
        moving = rpm > MUTE_BELOW_RPM or pedal_n > 0.02
        presence = self.standstill.step(moving, dt, s.idle_hum)
        cur.set_master((ev.duck if s.effects else 1.0)
                       * (ev.cruise_gain if s.cruise_quiet else 1.0) * presence
                       * max(0.0, min(1.0, s.volume)))

        tach = None
        if fx_gears:
            lo = fx.GEAR_SHIFT_KMH[ev.gear - 2] if ev.gear > 1 else 0.0
            hi = fx.GEAR_SHIFT_KMH[ev.gear - 1] if ev.gear <= len(fx.GEAR_SHIFT_KMH) else lo + 40.0
            tach = max(0.0, min(1.0, (speed - lo) / (hi - lo)))
        st = voices.State(motor_ratio=max(0.0, min(1.0, rpm / MAX_MOTOR_RPM)),
                          pedal=pedal_n, speed_kmh=speed,
                          swell=self.swell.update(self.pedal_position, dt),
                          active=fresh and presence > 0.001, dt=dt,
                          pitch_mul=ev.pitch_mul, virtual_tach=tach)
        self.info = cur.update(st)
        self.speed, self.pedal_n, self.fresh = speed, pedal_n, fresh

        flags = []
        if fx_gears:
            flags.append(f"gear {ev.gear}")
        if ev.regen > 0.05:
            flags.append(f"regen {ev.regen * 100:3.0f}%")
        if ev.reverse:
            flags.append("REVERSE")
        if s.cruise_quiet and ev.cruise_gain < 0.95:
            flags.append(f"cruise {20 * np.log10(ev.cruise_gain):+.0f} dB")
        self.flags = flags

    def status(self):
        return (f"{self.speed:5.1f} km/h   pedal {self.pedal_n * 100:3.0f} %   {self.info}   "
                + "   ".join(self.flags))

    def car_summary(self):
        """One line about the car link, or None in desk mode."""
        if not self.link:
            return None
        r = self.link.reader
        if r is None:
            return f"car: {self.link.state}"
        if r.fresh:
            amps = "?" if r.battery_current is None else f"{r.battery_current:+.0f}"
            rpm_s = "?" if r.motor_rpm_signed is None else f"{r.motor_rpm_signed:+.0f}"
            soc = "" if r.soc is None else f" | SOC {r.soc:.0f}%"
            return (f"car {r.rate_hz:4.1f} Hz | {r.speed:3.0f} km/h | pedal {r.pedal:5.1f} % "
                    f"| battery {amps} A | motor {rpm_s} rpm{soc}")
        return f"car: no data ({r.errors} errors) {r.last_error[:40]}"


class DataEngine:
    """Data-only stand-in for Engine: reads the car, plays nothing.

    For when the sound comes from the phone (Open Blue over Android Auto) and the Pi only
    relays the car's data -- so it runs on a board with no audio output at all (Pi Zero 2 W).
    Gives the Bluetooth server the same fields as Engine (speed, pedal_n, fresh, reader,
    events, settings, message, info), smoothed the same way, with no audio graph or voices.
    """
    current = None

    def __init__(self, settings, car_link, log=print):
        self.settings = settings
        self.log = log
        self.link = car_link
        self.reader = _LinkReader(car_link)
        self.car = CarInput(self.reader)
        self.events = fx.DriveEvents()
        self.speed = 0.0
        self.pedal_n = 0.0
        self.fresh = False
        self.info = "data only: the sound plays on the phone"
        self.message = ""
        self._last = None

    def tick(self, now=None):
        now = time.monotonic() if now is None else now
        dt = 0.02 if self._last is None else now - self._last
        self._last = now
        _, pedal, fresh = self.car.step(dt)
        r = self.reader
        speed = self.car.speed
        pedal_n = max(0.0, min(1.0, pedal / MAX_PEDAL))
        # reverse (and regen) detection, the same as when the Pi plays the sound
        self.events.step(dt, speed, pedal_n, fresh, current=r.battery_current,
                         signed_rpm=r.motor_rpm_signed, data_age=now - r.rpm_at,
                         reverse_test=self.settings.reverse_test, virtual_gears=False,
                         regen_fade=REGEN_FADE)
        self.speed, self.pedal_n, self.fresh = speed, pedal_n, fresh

    def status(self):
        return (f"{self.speed:5.1f} km/h   pedal {self.pedal_n * 100:3.0f} %"
                + ("   REVERSE" if self.events.reverse else ""))

    car_summary = Engine.car_summary


# --- Front ends ----------------------------------------------------------------------------

PI_BUFFER_FRAMES = 512


def make_graph(args):
    """AudioGraph on the chosen device: --output-device NAME, else a USB device if one is
    plugged in (the car speaker), else the system default. "--output-device default"
    skips the USB search. A NAME may be any unique part of the device name."""
    config = AudioGraphConfig()
    want = args.output_device
    names = AudioGraph.get_output_device_names()
    if want is None:
        usb = [n for n in names if "usb" in n.lower()]
        if usb:
            config.output_device_name = usb[0]
    elif want.lower() != "default":
        hits = [n for n in names if want.lower() in n.lower()]
        if not hits:
            raise SystemExit(f"output device {want!r} not found; devices:\n  "
                             + "\n  ".join(names))
        config.output_device_name = want if want in names else hits[0]
    # 256 frames (5 ms) is fine on a laptop; a Pi's slower cores need slack for the odd
    # slow block (Python, USB, a voice build), so ARM boards default to 512 (10.7 ms).
    import platform
    size = getattr(args, "buffer_size", None) or (
        PI_BUFFER_FRAMES if platform.machine() in ("aarch64", "armv7l", "armv6l") else None)
    if size:
        config.output_buffer_size = size
    return AudioGraph(config)


def run_gui(args, settings, link):
    from tkinter import (Tk, Scale, Checkbutton, BooleanVar, StringVar, OptionMenu, Label,
                         Button, HORIZONTAL)

    graph = make_graph(args)
    print(f"audio: {graph.output_device_name}")
    if link:
        link.start()
    window = Tk()
    window.title("Pedal and Motor Control")
    window.geometry("470x455")
    engine = Engine(settings, link)
    engine.select_voice(settings.voice)

    def bind(var, field):
        var.trace_add("write", lambda *_: setattr(settings, field, var.get()))
        return var

    voice_var = bind(StringVar(value=settings.voice), "voice")
    OptionMenu(window, voice_var, *VOICES).pack(fill="x", padx=8, pady=(6, 0))
    status = Label(window, text="", anchor="w")
    status.pack(fill="x", padx=8)
    for field, text in (("small_speaker", "Small speaker mode (laptop / 2-3\" speaker)"),
                        ("effects", "Drive effects (launch, regen, reverse, power-on chime)"),
                        ("cruise_quiet", "Quieter when cruising"),
                        ("virtual_gears", "Virtual gears (EV voices)")):
        Checkbutton(window, text=text,
                    variable=bind(BooleanVar(value=getattr(settings, field)), field)).pack()
    test_row = Label(window)
    test_row.pack()
    Checkbutton(test_row, text="Reverse (desk test)",
                variable=bind(BooleanVar(value=False), "reverse_test")).pack(side="left")
    Button(test_row, text="Power-on chime",
           command=lambda: engine.effects.power_on()).pack(side="left")
    Button(test_row, text="Launch", command=lambda: engine.effects.launch()).pack(side="left")

    def set_pedal(v):
        engine.pedal_position = float(v)

    def set_rpm(v):
        engine.motor_rpm = float(v)

    pedal_scale = Scale(window, from_=0, to=MAX_PEDAL, resolution=0.01, orient=HORIZONTAL,
                        label="Pedal Position", command=set_pedal)
    pedal_scale.set(engine.pedal_position)
    pedal_scale.pack()
    motor_scale = Scale(window, from_=0, to=int(max(MAX_MOTOR_RPM, VMAX_KMH * RPM_PER_KMH)),
                        resolution=10, orient=HORIZONTAL, label="Motor RPM", command=set_rpm)
    motor_scale.set(engine.motor_rpm)
    motor_scale.pack()
    coupled_var = BooleanVar(value=engine.rpm_coupled)
    Checkbutton(window, text="Pedal drives RPM (powertrain)", variable=coupled_var,
                command=lambda: setattr(engine, "rpm_coupled", coupled_var.get())).pack()

    def close():
        if link:
            link.stop()
        window.destroy()

    window.protocol("WM_DELETE_WINDOW", close)
    last_title = [0.0]

    def update_audio():
        engine.tick()
        if engine.car:      # sliders become live readouts; the car is in charge
            pedal_scale.set(round(engine.pedal_position, 2))
            motor_scale.set(int(engine.motor_rpm))
            now = time.monotonic()
            if now - last_title[0] >= 0.25:
                last_title[0] = now
                window.title(engine.car_summary())
        elif engine.rpm_coupled:
            motor_scale.set(int(engine.motor_rpm))
        status.config(text=engine.status())
        window.after(20, update_audio)

    update_audio()
    window.mainloop()


def demo_pedal(t):
    """A 40 s scripted drive for --headless without --car (testing / Pi CPU check)."""
    t %= 40.0
    for end, pedal in ((3, 0.0), (11, 1.0), (19, 0.3), (27, 0.0), (33, 0.6), (40, 0.0)):
        if t < end:
            return pedal
    return 0.0


def run_headless(args, settings, link):
    """No window: for the Pi. Sound first, chime, then the car connects in the background."""
    t0 = time.monotonic()

    def log(msg):
        print(f"[{time.monotonic() - t0:6.1f}s] {msg}", flush=True)

    if not link and not args.demo:
        raise SystemExit("--headless needs --car (or --demo for a scripted desk drive)")
    if link:
        link.log = log
        link.start()                 # adapter init runs alongside the audio start-up

    graph = make_graph(args)
    log(f"audio: {graph.output_device_name or 'default device'}, "
        f"{graph.sample_rate} Hz, buffer {graph.output_buffer_size}")

    engine = Engine(settings, link, log=log)
    if settings.effects and not args.no_boot_chime:
        engine.chime_now()           # before the voice is built: the slow part comes after
    engine.select_voice(settings.voice)
    if args.demo:
        engine.rpm_coupled = True
        engine.motor_rpm = engine.powertrain.rpm = 0.0

    control = None
    if args.bluetooth:                 # phone app control (ble_control.py, android/)
        import ble_control
        control = ble_control.ControlServer(engine, list(VOICES), save_settings, log=log)
        control.start()

    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())

    # audio device watchdog: the program can't move a running graph to another device,
    # so if the speaker in use disappears -- or a USB speaker appears while we are on the
    # fallback -- it exits and systemd (Restart=always) starts it again on the right one
    using = graph.output_device_name or ""
    on_fallback = args.output_device is None and "usb" not in using.lower()
    if on_fallback:
        log("no USB speaker found: using the default output until one is plugged in")

    def audio_device_problem():
        names = AudioGraph.get_output_device_names()
        if using and using not in names:
            return f"audio device {using!r} disappeared"
        if on_fallback and any("usb" in n.lower() for n in names):
            return "a USB speaker was plugged in"
        return None

    log(f"running: voice {settings.voice!r}, "
        f"{'car' if link else 'demo drive'}; Ctrl+C / SIGTERM to stop")
    period, next_t, last_log, worst_cpu = 0.02, time.monotonic(), 0.0, 0.0
    last_check, restart_reason = time.monotonic(), None
    errors, last_ping = 0, 0.0
    sd_notify("READY=1")
    sd_notify(f"STATUS=playing {settings.voice}")
    try:
        while not stop.is_set():
            now = time.monotonic()
            if args.demo:
                engine.pedal_position = demo_pedal(now - t0) * MAX_PEDAL
            try:
                engine.tick(now)
                errors = 0
            except Exception as e:      # one bad tick must not take the sound down
                errors += 1
                if errors == 1 or errors % 250 == 0:
                    log(f"engine error ({errors} in a row): {e!r}\n{traceback.format_exc()}")
                if errors == TICK_ERRORS_SWITCH_VOICE and settings.voice != DEFAULT_VOICE:
                    log(f"voice {settings.voice!r} keeps failing: switching to {DEFAULT_VOICE!r}")
                    settings.voice = DEFAULT_VOICE
                if errors >= TICK_ERRORS_RESTART:
                    restart_reason = f"engine keeps failing ({e!r})"
                    break
            if now - last_ping >= 2.0:  # systemd kills and restarts us if these stop
                last_ping = now
                sd_notify("WATCHDOG=1")
            worst_cpu = max(worst_cpu, graph.cpu_usage)
            if now - last_check >= AUDIO_CHECK_SEC:
                last_check = now
                restart_reason = audio_device_problem()
                if restart_reason:
                    break
            if args.log_every and now - last_log >= args.log_every:
                last_log = now
                car = engine.car_summary()
                fxs = ",".join(engine.effects.playing)
                log(f"{engine.status().strip()}   | audio cpu {worst_cpu * 100:3.0f}% (peak)"
                    + (f" fx {fxs}" if fxs else "")
                    + (f"   | {car}" if car else ""))
                worst_cpu = 0.0
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                stop.wait(delay)
            else:
                next_t = time.monotonic()     # fell behind (e.g. a voice build); resync
    finally:
        log("stopping")
        sd_notify("STOPPING=1")
        if control:
            control.stop()
        if link:
            link.stop()
        graph.stop()
    if restart_reason:              # non-zero exit: systemd restarts us on the right device
        log(f"{restart_reason}: restarting")
        raise SystemExit(75)


def run_data_only(args, settings, link):
    """No sound: read the car and send its data to the phone over Bluetooth LE."""
    t0 = time.monotonic()

    def log(msg):
        print(f"[{time.monotonic() - t0:6.1f}s] {msg}", flush=True)

    link.log = log
    link.start()
    engine = DataEngine(settings, link, log=log)
    import ble_control
    control = ble_control.ControlServer(engine, list(VOICES), save_settings, log=log)
    control.start()

    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())

    log("running data only (no sound): car -> phone over Bluetooth; Ctrl+C / SIGTERM to stop")
    period, next_t, last_log, last_ping, errors = 0.02, time.monotonic(), 0.0, 0.0, 0
    restart_reason = None
    sd_notify("READY=1")
    sd_notify("STATUS=data only: sending car data to the phone")
    try:
        while not stop.is_set():
            now = time.monotonic()
            try:
                engine.tick(now)
                errors = 0
            except Exception as e:      # one bad tick must not stop the data
                errors += 1
                if errors == 1 or errors % 250 == 0:
                    log(f"error ({errors} in a row): {e!r}\n{traceback.format_exc()}")
                if errors >= TICK_ERRORS_RESTART:
                    restart_reason = f"keeps failing ({e!r})"
                    break
            if now - last_ping >= 2.0:  # systemd kills and restarts us if these stop
                last_ping = now
                sd_notify("WATCHDOG=1")
            if args.log_every and now - last_log >= args.log_every:
                last_log = now
                log(f"{engine.status().strip()}   | {engine.car_summary()}")
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                stop.wait(delay)
            else:
                next_t = time.monotonic()
    finally:
        log("stopping")
        sd_notify("STOPPING=1")
        control.stop()
        link.stop()
    if restart_reason:
        log(f"{restart_reason}: restarting")
        raise SystemExit(75)


def main():
    global MAX_MOTOR_RPM
    import argparse

    ap = argparse.ArgumentParser(description="Jetson/EV motor sound synthesis")
    ap.add_argument("--rpm-max", type=float, default=None,
                    help="top of the pitch range in motor rpm (default %d, ~130 km/h). "
                         "A real EV motor spins far faster than the synthetic Jetson (an "
                         "E-GMP car is around 15000), and the pitch mapping stretches over "
                         "whatever you give it here" % CAR_RPM_MAX)
    ap.add_argument("--voice", default=DEFAULT_VOICE, metavar="NAME",
                    help="starting voice; a unique part of the name is enough, e.g. 'v8' "
                         "(see --list-voices)")
    ap.add_argument("--list-voices", action="store_true", help="print the voices and exit")
    ap.add_argument("--small-speaker", action=argparse.BooleanOptionalAction,
                    default=voices.SMALL_SPEAKER,
                    help="voice for small speakers: cut the unplayable bass and rebuild it "
                         "as harmonics (on by default; --no-small-speaker for headphones)")
    ap.add_argument("--effects", action=argparse.BooleanOptionalAction, default=True,
                    help="drive effects: launch, regen shimmer, reverse tone, power-on chime")
    ap.add_argument("--idle-hum", type=float, default=0.0, metavar="LEVEL",
                    help="at a standstill, fade to this level instead of silence (0-1, "
                         "default 0: idle for 3 s after stopping, then fade out)")
    ap.add_argument("--cruise-quiet", action=argparse.BooleanOptionalAction, default=True,
                    help="ease the voice down 6 dB once speed holds steady (+-5 km/h) for 4 s")
    ap.add_argument("--virtual-gears", action=argparse.BooleanOptionalAction, default=True,
                    help="virtual gear shifts: pitch climbs per gear, drops on upshift "
                         "(on by default; --no-virtual-gears for a smooth single gear)")
    ap.add_argument("--car", action="store_true",
                    help="drive pedal and pitch from the car over OBD (IONIQ 5 / E-GMP)")
    ap.add_argument("--port", default=None, help="OBD adapter port (default: autodetect)")
    ap.add_argument("--rpm-every", type=int, default=None,
                    help="with --car: read motor rpm every Nth cycle to calibrate the "
                         "rpm/km/h ratio (0 = never; pedal+speed then run at ~33 Hz). "
                         "Default 4, or 8 with --data-only: each rpm read holds up the "
                         "pedal ~95 ms, and the phone only needs it to spot reversing")
    hl = ap.add_argument_group("headless (Raspberry Pi)")
    hl.add_argument("--bluetooth", action="store_true",
                    help="accept control from the phone app over Bluetooth LE (ble_control.py)")
    hl.add_argument("--headless", action="store_true",
                    help="no window: chime at start-up, connect to the car in the background "
                         "and keep reconnecting; logs status lines to stdout")
    hl.add_argument("--data-only", action="store_true",
                    help="no sound: just read the car and send its data to the phone over "
                         "Bluetooth LE (Open Blue plays the sound). Implies --headless --car "
                         "--bluetooth; needs no speaker, so it runs on a Pi Zero 2 W")
    hl.add_argument("--demo", action="store_true",
                    help="with --headless and no --car: loop a scripted 40 s drive")
    ap.add_argument("--output-device", default=None,
                    help="audio output: any unique part of a device name (see --list-devices). "
                         "Default: the first USB audio device, else the system default; "
                         "'default' forces the system default")
    ap.add_argument("--list-devices", action="store_true",
                    help="print the audio output devices and exit")
    ap.add_argument("--buffer-size", type=int, default=None,
                    help="audio buffer in frames (default 256, or %d on a Pi); raise it "
                         "(1024) if it crackles" % PI_BUFFER_FRAMES)
    hl.add_argument("--no-boot-chime", action="store_true",
                    help="headless: don't chime at start-up (it still chimes when car data "
                         "first appears)")
    hl.add_argument("--log-every", type=float, default=5.0,
                    help="headless: seconds between status lines (0 = quiet)")
    args = ap.parse_args()

    if args.list_voices:
        print("\n".join(VOICES))
        return
    if args.list_devices:
        print("\n".join(AudioGraph.get_output_device_names()))
        return

    # module global: apply_controls() reads it per call
    # same pitch range at the desk as in the car; the desk sim's top speed is separate
    MAX_MOTOR_RPM = args.rpm_max or CAR_RPM_MAX
    settings = Settings(voice=find_voice(args.voice), small_speaker=args.small_speaker,
                        effects=args.effects, cruise_quiet=args.cruise_quiet,
                        idle_hum=max(0.0, min(1.0, args.idle_hum)),
                        virtual_gears=args.virtual_gears)
    if args.data_only:
        args.headless = args.car = args.bluetooth = True
    if args.rpm_every is None:
        args.rpm_every = 8 if args.data_only else 4
    if args.bluetooth or args.headless:   # the phone's last settings win over the defaults
        load_settings(settings)
    link = CarLink(args.port, args.rpm_every) if args.car else None
    if args.data_only:
        run_data_only(args, settings, link)
    elif args.headless:
        run_headless(args, settings, link)
    else:
        run_gui(args, settings, link)


if __name__ == "__main__":
    main()
