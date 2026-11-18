"""Synthesised voices for the sound engine: futuristic EV tones and combustion engines.

Every voice takes the same inputs each GUI tick (see `State`) and owns its own signalflow
graph. The EV voices follow motor speed directly. The combustion voices put a virtual
gearbox in between, because an engine's sound is about engine rpm -- which climbs, drops on
each upshift and idles at a standstill -- not about road speed.

Kept deliberately low: fundamentals stay under ~200 Hz for the EV voices, and every voice
is low-passed so nothing gets shrill at speed.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from signalflow import (Abs, ASREnvelope, Buffer, Clip, CombDelay, Constant, EnvelopeBuffer,
                        FeedbackDelayNetwork,
                        Granulator, Impulse, PinkNoise, Pow, RandomUniform, SawOscillator, SineLFO, SineOscillator, Smooth,
                        SquareOscillator, SVFilter, Tanh, TriangleOscillator, WhiteNoise)


@dataclass
class State:
    motor_ratio: float   # 0..1 of the motor rpm range
    pedal: float         # 0..1
    speed_kmh: float
    swell: float         # 0..1 tip-in swell
    active: bool         # False = no data from the car: go silent
    dt: float
    pitch_mul: float = 1.0   # virtual gears: <1 just after an upshift, climbing to 1


def _target(value):
    c = Constant(value)
    return c


def _cents(c):
    return 2.0 ** (c / 1200.0)


def _ev_gain(st: State, level: float, idle_level: float = 0.5) -> float:
    """Shared EV loudness law: quiet hum at standstill, rising with pedal and speed."""
    if not st.active:
        return 0.0
    moving = min(1.0, st.motor_ratio / 0.05)
    idle = idle_level + (1.0 - idle_level) * moving
    return (0.35 + 0.65 * st.pedal) * (0.5 + 0.5 * st.motor_ratio) * idle \
        * (1.0 + 0.4 * st.swell) * level


# --- Output stage shared by every voice -------------------------------------------------
# Small speakers (laptops, a 2-3" driver) can't move air below ~120 Hz, so a low voice
# loses its body and much of its loudness. Small-speaker mode (a) high-passes what the
# speaker can't play anyway, so it stops wasting power and distorting on it, and
# (b) regenerates the bass as its own 2nd/4th harmonics ("virtual bass"): the ear hears the
# missing fundamental from its overtones, so the note still sounds as deep as it is.
# Full-wave rectification makes those harmonics in proportion to the bass level.
SMALL_SPEAKER = False
SMALL_HP_HZ = 110.0          # below this the small-speaker path is cut
VBASS_FROM_HZ = 140.0        # bass below this is turned into harmonics
VBASS_LEVEL = 1.6

# Loudness trims in dB, from A-weighted measurements of each voice through the same drive
# (see demos/). They make switching voices keep the same perceived level.
TRIM_DB: dict[str, float] = {
    'Jetsons i3 (recording)': +3.8,
    'Jetsons i3': -5.7,
    'Deep Drone': +11.8,
    'Cinematic Chord': +6.2,
    'Cinematic Drive': +3.6,
    'Inverter Whine': +8.1,
    'Maglev': +8.9,
    'Warp Core': +12.0,
    'Theremin': +5.4,
    'Hover Turbine': +12.0,
    'Jetsons Low': +2.3,
    'Jetsons': -0.2,
    'Hum (jetson.mp3 loop)': -1.0,
    'V8 Muscle': +3.1,
    'Inline-4 Sport': +6.6,
    'Action Pulse': +5.1,
    'Trailer Brass': +6.6,
    'Strings & Choir': +2.9,
    'Cathedral': +3.6,
    'Grand Touring': +8.0,
    'Dreadnought': +9.0,
    'Jet Turbine': +9.8,
}


def output_stage(node, small_ctrl, master=None):
    sm = Smooth(small_ctrl, 0.9995)
    low = SVFilter(node, "low_pass", cutoff=VBASS_FROM_HZ, resonance=0.0)
    vbass = SVFilter(SVFilter(Abs(low), "high_pass", cutoff=SMALL_HP_HZ, resonance=0.0),
                     "low_pass", cutoff=800.0, resonance=0.0) * VBASS_LEVEL
    small = SVFilter(node, "high_pass", cutoff=SMALL_HP_HZ, resonance=0.0) + vbass
    mixed = node * (1.0 - sm) + small * sm
    if master is not None:
        mixed = mixed * Smooth(master, 0.998)   # ducking for effects (~10 ms glide)
    return Tanh(mixed)       # soft limiter: transparent at normal levels, never clips


class Voice:
    name = "voice"
    kind = "ev"

    def __init__(self):
        out, self.targets = self.build()
        # presets (see presets.py) keep the loudness trim of the voice they are based on
        trim = 10.0 ** (TRIM_DB.get(getattr(self, "trim_name", None) or self.name, 0.0) / 20.0)
        self.targets["small"] = Constant(1.0 if SMALL_SPEAKER else 0.0)
        self.targets["master"] = Constant(1.0)
        self.output = output_stage(out * trim, self.targets["small"], self.targets["master"])

    def set_master(self, level: float):
        """Duck the voice (0..1) under effects such as reverse or a virtual-gear shift."""
        self.targets["master"].set_value(level)

    def set_small_speaker(self, on: bool):
        self.targets["small"].set_value(1.0 if on else 0.0)

    def build(self):
        raise NotImplementedError

    def update(self, st: State) -> str:
        """Set the targets for this tick; returns a short status for the GUI."""
        raise NotImplementedError


# ---------------------------------------------------------------------------------------
# Futuristic EV voices
# ---------------------------------------------------------------------------------------

class Slipstream(Voice):
    """Electric performance sound with a slowly responding, load-driven rotor.

    A low motor note anchors a higher, slightly inharmonic rotor. The rotor spools up
    with pedal and settles more slowly on release, adding motion independent of speed.
    """
    name = "Slipstream"

    def __init__(self):
        self.rotor = 0.0
        super().__init__()

    def build(self):
        t = {key: _target(value) for key, value in {
            "base": 105.0, "rotor": 360.0, "load": 0.0, "surge": 0.0, "gain": 0.0,
        }.items()}
        base = Smooth(t["base"], 0.9995)
        rotor = Smooth(t["rotor"], 0.9995)
        load = Smooth(t["load"], 0.999)
        surge = Smooth(t["surge"], 0.999)

        # The 1.5-order component gives the motor a mechanical edge without adding
        # a major/minor chord. The fixed resonator gives the sweep a changing color.
        motor = TriangleOscillator(base) * 0.12 + SawOscillator(base * 1.5) * 0.055
        motor = SVFilter(motor, "low_pass", cutoff=850.0, resonance=0.25)
        rotor_tone = (SineOscillator(rotor) * 0.050
                      + SineOscillator(rotor * 2.03) * 0.014)
        rotor_tone *= 0.45 + 0.65 * load + 0.35 * surge
        # A soft saturating load layer changes timbre at tip-in, then recedes.
        growl = SVFilter(Tanh(SawOscillator(base * 0.75) * 1.5), "low_pass",
                         cutoff=720.0, resonance=0.15)
        growl *= 0.025 + 0.055 * load + 0.04 * surge
        return (motor + rotor_tone + growl) * Smooth(t["gain"], 0.999), t

    def update(self, st):
        r = max(0.0, min(1.0, st.motor_ratio))
        p = max(0.0, min(1.0, st.pedal))
        target = min(1.0, 0.35 * r + 0.65 * p)
        tau = 0.5 if target > self.rotor else 1.8
        self.rotor += (target - self.rotor) * (1.0 - math.exp(-max(0.0, st.dt) / tau))
        self.targets["base"].set_value((105.0 + 100.0 * r) * st.pitch_mul)
        self.targets["rotor"].set_value((350.0 + 300.0 * r + 230.0 * self.rotor) * st.pitch_mul)
        self.targets["load"].set_value(p)
        self.targets["surge"].set_value(st.swell)
        self.targets["gain"].set_value((1.4 + 0.3 * p) if st.active else 0.0)
        return "spooling" if p > 0.5 else "cruise"


class PrecisionDrive(Voice):
    """A restrained electric drive sound for sustained use.

    A soft body and a quiet mechanical overtone respond smoothly to speed and pedal.
    Lift-off is inferred from low pedal, not a measurement of regen current.
    """
    name = "Precision Drive"

    def build(self):
        t = {key: _target(value) for key, value in {
            "body": 130.0, "mesh": 500.0,
            "load": 0.0, "coast": 0.0, "attack": 0.0, "gain": 0.0,
        }.items()}
        body_f = Smooth(t["body"], 0.9995)
        mesh_f = Smooth(t["mesh"], 0.9995)
        load = Smooth(t["load"], 0.999)
        coast = Smooth(t["coast"], 0.999)
        attack = Smooth(t["attack"], 0.9995)

        body = SineOscillator(body_f) * 0.09 + TriangleOscillator(body_f * 2.0) * 0.035
        # A resonant, slightly irregular harmonic body becomes more present on load.
        # Its fixed resonance changes the tone as the harmonics pass through it.
        motor = SawOscillator(body_f) * 0.5 + TriangleOscillator(body_f * 1.5) * 0.25
        motor = SVFilter(motor, "low_pass", cutoff=620.0, resonance=0.45)
        motor *= 0.045 + 0.07 * load + 0.015 * attack
        mesh = SineOscillator(mesh_f) * (0.008 + 0.018 * load)
        overrun = SineOscillator(mesh_f * 1.48) * coast * 0.006
        return (body + motor + mesh + overrun) * Smooth(t["gain"], 0.999), t

    def update(self, st):
        r = max(0.0, min(1.0, st.motor_ratio))
        p = max(0.0, min(1.0, st.pedal))
        self.targets["body"].set_value((125.0 + 95.0 * r) * st.pitch_mul)
        self.targets["mesh"].set_value((390.0 + 440.0 * r) * st.pitch_mul)
        self.targets["load"].set_value(p)
        self.targets["coast"].set_value(max(0.0, 1.0 - p / 0.3) * min(1.0, r / 0.08))
        self.targets["attack"].set_value(st.swell)
        self.targets["gain"].set_value((1.65 + 0.3 * p) if st.active else 0.0)
        return "coast" if p < 0.15 else "drive"


class DeepDrone(Voice):
    """Detuned saw pad with a sub-octave: a big, low sci-fi hum. 45 Hz idle -> 110 Hz."""
    name = "Deep Drone"
    F_IDLE, F_TOP = 45.0, 110.0
    LEVEL = 0.55

    def build(self):
        t = {"freq": _target(self.F_IDLE), "cutoff": _target(300.0), "gain": _target(0.0)}
        f = Smooth(t["freq"], 0.9995)
        drift = SineLFO(0.13, 1.0 / _cents(6), _cents(6))     # slow chorus drift
        saws = (SawOscillator(f * drift) + SawOscillator(f * 1.006)
                + SawOscillator(f / drift * 0.994)) * 0.33
        sub = SineOscillator(f * 0.5) * 0.6
        body = SVFilter(saws, "low_pass", cutoff=Smooth(t["cutoff"], 0.999), resonance=0.25)
        out = (body + sub) * Smooth(t["gain"], 0.999)
        return out, t

    def update(self, st):
        r = st.motor_ratio
        self.targets["freq"].set_value((self.F_IDLE + (self.F_TOP - self.F_IDLE) * r ** 0.8)
                                       * st.pitch_mul)
        # starts at 650 Hz: below that the saws are all fundamental, which small speakers
        # can't play, so the drone was nearly inaudible on them
        self.targets["cutoff"].set_value(650.0 + 900.0 * r + 1800.0 * st.pedal ** 1.3
                                         + 800.0 * st.swell)
        self.targets["gain"].set_value(_ev_gain(st, self.LEVEL))
        return ""


class HoverTurbine(Voice):
    """Spooling air: band-passed noise whose pitch rises with speed, over a soft tone."""
    name = "Hover Turbine"
    LEVEL = 0.9

    def build(self):
        t = {"band": _target(300.0), "tone": _target(90.0), "air": _target(0.3),
             "gain": _target(0.0)}
        band = Smooth(t["band"], 0.9995)
        tone_f = Smooth(t["tone"], 0.9995)
        noise = PinkNoise()
        whoosh = SVFilter(noise, "band_pass", cutoff=band, resonance=0.82) * 1.6
        whoosh2 = SVFilter(noise, "band_pass", cutoff=band * 1.5, resonance=0.75) * 0.6
        hum = (TriangleOscillator(tone_f) * 0.5 + SineOscillator(tone_f * 2.0) * 0.12)
        air = Smooth(t["air"], 0.999)
        mix = (whoosh + whoosh2) * air + hum
        out = SVFilter(mix, "low_pass", cutoff=2800.0, resonance=0.0) * Smooth(t["gain"], 0.999)
        return out, t

    def update(self, st):
        r = st.motor_ratio
        self.targets["band"].set_value((220.0 + 1300.0 * r ** 0.9) * st.pitch_mul)
        self.targets["tone"].set_value((70.0 + 120.0 * r) * st.pitch_mul)
        self.targets["air"].set_value(0.35 + 0.65 * st.pedal + 0.4 * st.swell)
        self.targets["gain"].set_value(_ev_gain(st, self.LEVEL, idle_level=0.45))
        return ""


class CinematicChord(Voice):
    """Root + fifth + octave in soft triangles, slowly beating: a calm, 'concept car' tone.

    Pedal brings in more octave and opens the filter, so pressing on reads as a swell
    rather than a whine. 65 Hz idle -> 165 Hz. The filter starts well above the chord so
    its upper harmonics carry it on small speakers, where a 65 Hz root alone is inaudible.
    """
    name = "Cinematic Chord"
    F_IDLE, F_TOP = 65.0, 165.0
    LEVEL = 1.5

    def build(self):
        t = {"freq": _target(self.F_IDLE), "cutoff": _target(500.0), "oct": _target(0.2),
             "gain": _target(0.0)}
        f = Smooth(t["freq"], 0.9995)
        root = TriangleOscillator(f) + TriangleOscillator(f * 1.003) * 0.6
        fifth = TriangleOscillator(f * 1.5 * 0.998) * 0.55
        octave = TriangleOscillator(f * 2.0 * 1.002) * Smooth(t["oct"], 0.9995)
        # the 12th (octave + fifth): still inside the chord, but where hearing is sensitive
        twelfth = TriangleOscillator(f * 3.0 * 0.999) * 0.3
        shimmer = SineLFO(0.21, 0.85, 1.15)
        mix = (root + fifth + octave + twelfth) * shimmer * 0.4
        filtered = SVFilter(mix, "low_pass", cutoff=Smooth(t["cutoff"], 0.999), resonance=0.15)
        out = Tanh(filtered * Smooth(t["gain"], 0.999))   # soft ceiling: louder, never clips
        return out, t

    def update(self, st):
        r = st.motor_ratio
        self.targets["freq"].set_value((self.F_IDLE + (self.F_TOP - self.F_IDLE) * r ** 0.85)
                                       * st.pitch_mul)
        self.targets["cutoff"].set_value(1100.0 + 900.0 * r + 2500.0 * st.pedal ** 1.4
                                         + 1000.0 * st.swell)
        self.targets["oct"].set_value(0.35 + 0.5 * st.pedal)
        self.targets["gain"].set_value(_ev_gain(st, self.LEVEL))
        return ""


class CinematicDrive(Voice):
    """An original cinematic drive sound: layered, swelling, always-rising.

    Layers, all following speed (motor ratio) and pedal:
      pad      a chord (root, fifth, octave, tenth) of detuned saws; root rises one
               octave over the speed range, and the filter opens with speed and pedal
               so it swells rather than whines
      rise     a Shepard tone: octave-spaced partials that keep sliding up as speed
               rises, under a fixed bell-shaped level curve -- an endless climb that
               never gets shrill (one full cycle per RISE_KMH)
      brass    saws through a resonant band-pass, fading in with pedal (load)
      pulse    a soft sub 'heartbeat' at half the root, quickening with speed
      air      band-passed noise that rises with speed
    """
    name = "Cinematic Drive"
    ROOT_IDLE, ROOT_TOP = 55.0, 110.0            # pad root, Hz
    CHORD = (1.0, 1.5, 2.0, 2.5)                 # root, fifth, octave, major tenth
    CHORD_LEVEL = (1.0, 0.7, 0.55, 0.35)
    RISE_PARTIALS = 6
    RISE_BASE = 41.0                             # lowest Shepard partial, Hz
                                                 # (window peaks 3 octaves up: 330 Hz)
    RISE_KMH = 25.0                              # km/h per full octave of climb
    LEVEL = 0.8

    def build(self):
        t = {"root": _target(self.ROOT_IDLE), "cutoff": _target(500.0),
             "brass": _target(0.0), "brass_f": _target(900.0), "pulse_rate": _target(0.8),
             "air": _target(0.03), "air_f": _target(900.0), "rise": _target(0.5),
             "gain": _target(0.0)}
        root = Smooth(t["root"], 0.9995)

        pad = None
        for ratio, lvl in zip(self.CHORD, self.CHORD_LEVEL):
            f = root * ratio
            note = (SawOscillator(f * 0.997) + SawOscillator(f * 1.003)
                    + SawOscillator(f) * 0.6) * (lvl * 0.22)
            pad = note if pad is None else pad + note
        pad = SVFilter(pad, "low_pass", cutoff=Smooth(t["cutoff"], 0.999), resonance=0.3)

        # Shepard partials: frequency and level set per tick (see update)
        rise = None
        for k in range(self.RISE_PARTIALS):
            t[f"rf{k}"] = _target(self.RISE_BASE * 2 ** k)
            t[f"ra{k}"] = _target(0.0)
            osc = (SineOscillator(Smooth(t[f"rf{k}"], 0.995)) * 0.8
                   + TriangleOscillator(Smooth(t[f"rf{k}"], 0.995)) * 0.2)
            osc = osc * Smooth(t[f"ra{k}"], 0.999)
            rise = osc if rise is None else rise + osc
        rise = SVFilter(rise, "low_pass", cutoff=800.0, resonance=0.0) * Smooth(t["rise"], 0.999)

        brass_src = SawOscillator(root) + SawOscillator(root * 1.5 * 1.002) * 0.7
        brass = SVFilter(brass_src, "band_pass", cutoff=Smooth(t["brass_f"], 0.999),
                         resonance=0.55) * Smooth(t["brass"], 0.999)

        heart = SineLFO(Smooth(t["pulse_rate"], 0.999), 0.75, 1.0)
        pulse = SineOscillator(root * 0.5) * 0.45 * heart

        air = SVFilter(PinkNoise(), "band_pass", cutoff=Smooth(t["air_f"], 0.999),
                       resonance=0.5) * Smooth(t["air"], 0.999)

        mix = pad + rise * 0.35 + brass * 0.5 + pulse + air
        out = Tanh(mix * Smooth(t["gain"], 0.999))
        return out, t

    def update(self, st):
        r, p = st.motor_ratio, st.pedal
        self.targets["root"].set_value(self.ROOT_IDLE * (self.ROOT_TOP / self.ROOT_IDLE)
                                       ** (r ** 0.85) * st.pitch_mul)
        self.targets["cutoff"].set_value(450.0 + 1600.0 * r + 2600.0 * p ** 1.3
                                         + 1500.0 * st.swell)
        self.targets["brass"].set_value(0.9 * p ** 1.5 + 0.6 * st.swell)
        self.targets["brass_f"].set_value(700.0 + 900.0 * r + 900.0 * p)
        self.targets["pulse_rate"].set_value(0.8 + 2.4 * r)
        self.targets["air"].set_value(0.03 + 0.12 * r + 0.05 * p)
        self.targets["air_f"].set_value(700.0 + 2500.0 * r)

        # Shepard climb: phase advances with speed; levels from a fixed bell in log-freq,
        # so each partial fades in at the bottom and out at the top as it slides up
        # each partial wraps on its own, top -> bottom, where the window is silent
        # (wrapping all at once swept every audible partial down an octave: a chirp)
        n, climb = self.RISE_PARTIALS, st.speed_kmh / self.RISE_KMH
        for k in range(n):
            pos = (k + climb) % n
            self.targets[f"rf{k}"].set_value(self.RISE_BASE * 2 ** pos * st.pitch_mul)
            self.targets[f"ra{k}"].set_value(math.sin(math.pi * pos / n) ** 4)
        self.targets["rise"].set_value(0.25 + 0.75 * min(1.0, r * 3) + 0.4 * st.swell)

        self.targets["gain"].set_value(_ev_gain(st, self.LEVEL))
        return ""


class InverterWhine(Voice):
    """A real performance EV, exaggerated: what you hear inside one under hard throttle.

    Motor electromagnetic whine at orders 8 and 12 of motor speed (pole-pair harmonics),
    reduction-gear mesh whine, a faint inverter switching whistle that is most audible at
    low speed, and road/air noise that grows with the square of speed. Whine strength
    follows torque (pedal). Kept under ~1.3 kHz at 130 km/h so it isn't shrill.
    """
    name = "Inverter Whine"
    RPM_PER_KMH = 77.0
    ORDERS = ((8.0, 1.0), (12.0, 0.45), (16.0, 0.18))
    GEAR_ORDER = 2.6              # gear mesh relative to motor electrical order 8, damped
    LEVEL = 0.5

    def build(self):
        t = {"motor_hz": _target(1.0), "whine": _target(0.0), "gear": _target(0.0),
             "pwm": _target(0.0), "road": _target(0.0), "road_f": _target(300.0),
             "gain": _target(0.0)}
        m = Smooth(t["motor_hz"], 0.999)
        whine = None
        for order, amp in self.ORDERS:
            osc = SineOscillator(m * order) * amp
            whine = osc if whine is None else whine + osc
        whine = whine * Smooth(t["whine"], 0.999)
        gear = (TriangleOscillator(m * 8.0 * self.GEAR_ORDER) * 0.5) * Smooth(t["gear"], 0.999)
        # switching whistle: carrier with sidebands at +-2x electrical frequency
        pwm = (SineOscillator(m * 8.0 * 0.5 + 1900.0) + SineOscillator(1900.0 - m * 8.0 * 0.5))
        pwm = pwm * 0.5 * Smooth(t["pwm"], 0.999)
        road = SVFilter(PinkNoise(), "low_pass", cutoff=Smooth(t["road_f"], 0.999),
                        resonance=0.0) * Smooth(t["road"], 0.999)
        body = SineOscillator(m * 4.0) * 0.25            # low electrical hum underneath
        mix = whine * 0.35 + gear * 0.2 + pwm * 0.06 + road * 0.9 + body * 0.3
        out = SVFilter(mix, "low_pass", cutoff=3200.0, resonance=0.0) * Smooth(t["gain"], 0.999)
        return out, t

    def update(self, st):
        rpm = st.speed_kmh * self.RPM_PER_KMH
        hz = max(1.0, rpm / 60.0 * st.pitch_mul)
        self.targets["motor_hz"].set_value(hz)
        torque = 0.25 + 0.75 * st.pedal + 0.4 * st.swell
        self.targets["whine"].set_value(torque)
        self.targets["gear"].set_value(0.3 + 0.7 * st.pedal)
        self.targets["pwm"].set_value(max(0.0, 1.0 - st.speed_kmh / 40.0) * (0.3 + st.pedal))
        r = min(1.0, st.speed_kmh / 130.0)
        self.targets["road"].set_value(0.05 + 0.4 * r * r)
        self.targets["road_f"].set_value(400.0 + 1600.0 * r)
        self.targets["gain"].set_value(_ev_gain(st, self.LEVEL))
        return ""


class Maglev(Voice):
    """A futuristic train pulling away: a clean sine with soft bell-like overtones that
    beat slowly, a rhythmic shimmer as it passes 'magnet segments' (faster with speed),
    and rising air. Pedal brightens it."""
    name = "Maglev"
    F_IDLE, F_TOP = 70.0, 300.0
    BELL = ((2.0, 0.3), (2.76, 0.18), (5.4, 0.07))
    SEGMENTS_PER_KMH = 0.22       # shimmer pulses per second per km/h
    LEVEL = 0.6

    def build(self):
        t = {"freq": _target(self.F_IDLE), "seg": _target(1.0), "bright": _target(800.0),
             "air": _target(0.0), "air_f": _target(400.0), "gain": _target(0.0)}
        f = Smooth(t["freq"], 0.9995)
        tone = SineOscillator(f)
        bells = None
        for ratio, amp in self.BELL:
            osc = SineOscillator(f * ratio * 1.002) * amp
            bells = osc if bells is None else bells + osc
        bells = SVFilter(bells, "low_pass", cutoff=Smooth(t["bright"], 0.999), resonance=0.1)
        seg = Pow(SineLFO(Smooth(t["seg"], 0.999), 0.0, 1.0), 4.0)    # short swells
        shimmer = (tone * 0.15 + bells) * seg * 0.6
        air = SVFilter(PinkNoise(), "band_pass", cutoff=Smooth(t["air_f"], 0.999),
                       resonance=0.4) * Smooth(t["air"], 0.999)
        mix = tone * 0.6 + bells * 0.5 + shimmer + air
        return Tanh(mix) * Smooth(t["gain"], 0.999), t

    def update(self, st):
        r = st.motor_ratio
        self.targets["freq"].set_value((self.F_IDLE + (self.F_TOP - self.F_IDLE) * r ** 0.8)
                                       * st.pitch_mul)
        self.targets["seg"].set_value(max(0.3, st.speed_kmh * self.SEGMENTS_PER_KMH))
        self.targets["bright"].set_value(700.0 + 2200.0 * st.pedal ** 1.2 + 800.0 * st.swell)
        self.targets["air"].set_value(0.05 + 0.35 * r * r)
        self.targets["air_f"].set_value(350.0 + 1400.0 * r)
        self.targets["gain"].set_value(_ev_gain(st, self.LEVEL))
        return ""


class WarpCore(Voice):
    """A deep drone through a resonant filter that throbs -- quickening and opening up
    with speed and pedal -- over a sub. Root 38 Hz idle -> 76 Hz."""
    name = "Warp Core"
    F_IDLE, F_TOP = 38.0, 76.0
    LEVEL = 0.6

    def build(self):
        t = {"freq": _target(self.F_IDLE), "throb": _target(0.6), "lo": _target(180.0),
             "hi": _target(600.0), "gain": _target(0.0)}
        f = Smooth(t["freq"], 0.9995)
        src = SawOscillator(f) + SawOscillator(f * 1.5 * 1.003) * 0.6 + SawOscillator(f * 2.004) * 0.4
        lo, hi = Smooth(t["lo"], 0.999), Smooth(t["hi"], 0.999)
        lfo = SineLFO(Smooth(t["throb"], 0.999), 0.0, 1.0)
        cutoff = lo + (hi - lo) * lfo
        core = SVFilter(src * 0.3, "low_pass", cutoff=cutoff, resonance=0.72)
        sub = SineOscillator(f) * 0.5
        hiss = SVFilter(PinkNoise(), "band_pass", cutoff=cutoff * 2.0, resonance=0.5) * 0.04
        return Tanh((core + sub + hiss) * Smooth(t["gain"], 0.999)), t

    def update(self, st):
        r, p = st.motor_ratio, st.pedal
        self.targets["freq"].set_value((self.F_IDLE + (self.F_TOP - self.F_IDLE) * r ** 0.85)
                                       * st.pitch_mul)
        self.targets["throb"].set_value(0.6 + 6.5 * r + 2.0 * p)
        self.targets["lo"].set_value(140.0 + 200.0 * r)
        self.targets["hi"].set_value(450.0 + 900.0 * r + 1800.0 * p ** 1.3 + 800.0 * st.swell)
        self.targets["gain"].set_value(_ev_gain(st, self.LEVEL))
        return ""


class Theremin(Voice):
    """Classic 1950s sci-fi: a gliding, slightly vocal sine with vibrato that deepens with
    the pedal. Kept low (110 -> 330 Hz) and glides slowly between pitches."""
    name = "Theremin"
    F_IDLE, F_TOP = 110.0, 330.0
    LEVEL = 0.5

    def build(self):
        t = {"freq": _target(self.F_IDLE), "depth": _target(1.0), "gain": _target(0.0)}
        f = Smooth(t["freq"], 0.99985)                     # slow, singing glide
        depth = Smooth(t["depth"], 0.999)
        vib = SineLFO(5.6, 0.0, 1.0) * (depth * 0.03) + (1.0 - depth * 0.015)
        tone = Tanh(SineOscillator(f * vib) * 1.6) * 0.7    # soft odd harmonics
        voice = SVFilter(tone, "band_pass", cutoff=900.0, resonance=0.35) * 0.5
        low = SineOscillator(f * vib * 0.5) * 0.25
        mix = tone + voice + low
        mix = mix + CombDelay(mix, 0.11, 0.25) * 0.3       # a little room
        return mix * Smooth(t["gain"], 0.999), t

    def update(self, st):
        r = st.motor_ratio
        self.targets["freq"].set_value((self.F_IDLE + (self.F_TOP - self.F_IDLE) * r ** 0.75)
                                       * st.pitch_mul)
        self.targets["depth"].set_value(0.5 + 1.2 * st.pedal + 0.8 * st.swell)
        self.targets["gain"].set_value(_ev_gain(st, self.LEVEL))
        return ""


# --- Cinematic score voices ------------------------------------------------------------
# All in D minor. Speed walks a chord progression -- Dm, Bb, C, then Dm an octave up -- so
# accelerating plays music instead of just raising pitch. A shared concert-hall reverb
# (feedback delay network) gives them the space that makes a sound read as "film score".
D_MINOR_PROGRESSION = (
    (0.0, "Dm", (73.42, 87.31, 110.00)),        # D2 F2 A2
    (25.0, "Bb", (58.27, 73.42, 87.31)),        # Bb1 D2 F2
    (50.0, "C", (65.41, 82.41, 98.00)),         # C2 E2 G2
    (80.0, "Dm", (146.83, 174.61, 220.00)),     # D3 F3 A3: the lift
)


class _Progression:
    """Chord index from speed, with hysteresis so hovering at a boundary can't flip-flop."""

    def __init__(self, chords=D_MINOR_PROGRESSION, hysteresis_kmh=3.0):
        self.chords, self.hyst, self.i = chords, hysteresis_kmh, 0

    def step(self, speed):
        while self.i + 1 < len(self.chords) and speed >= self.chords[self.i + 1][0]:
            self.i += 1
        while self.i > 0 and speed < self.chords[self.i][0] - self.hyst:
            self.i -= 1
        return self.chords[self.i]


def _hall(node, wet=0.3):
    """Big, dark concert-hall reverb mixed under the dry signal."""
    verb = FeedbackDelayNetwork(node, num_delays=8, feedback=0.86, damping=0.45, wet=1.0,
                                dry=0.0, max_delay_time=0.2)
    return node + verb * wet


class _ScoreVoice(Voice):
    """Shared plumbing: chord-note targets that follow the progression."""
    GLIDE = 0.995                     # note smoothing: near-instant chord changes

    def __init__(self):
        self.prog = _Progression()
        super().__init__()

    def _notes(self, t):
        t["n0"], t["n1"], t["n2"] = (_target(f) for f in D_MINOR_PROGRESSION[0][2])
        return [Smooth(t[k], self.GLIDE) for k in ("n0", "n1", "n2")]

    def _set_chord(self, st):
        _, name, notes = self.prog.step(st.speed_kmh)
        for k, f in zip(("n0", "n1", "n2"), notes):
            self.targets[k].set_value(f * st.pitch_mul)
        return name


class ActionPulse(_ScoreVoice):
    """Action-film ostinato: a plucky 8th-note pulse alternating root and octave, over a
    low drone. Tempo follows speed (80 -> 150 BPM); pedal brings in a sustained chord
    layer and opens the pulse's filter."""
    name = "Action Pulse"
    BPM_IDLE, BPM_TOP = 80.0, 150.0
    LEVEL = 1.6

    def build(self):
        t = {"rate": _target(self.BPM_IDLE / 30.0), "cutoff": _target(900.0),
             "layer": _target(0.2), "gain": _target(0.0)}
        n0, n1, n2 = self._notes(t)
        rate = Smooth(t["rate"], 0.999)                     # 8th notes per second
        clock = Impulse(rate)
        env = ASREnvelope(0.002, 0.03, 0.13, curve=2.0, clock=clock)
        octave = SquareOscillator(rate * 0.5) * 0.5 + 1.5   # 1, 2, 1, 2 ... (root, octave)
        f = n0 * octave
        cutoff = Smooth(t["cutoff"], 0.999)
        pulse = SVFilter(SawOscillator(f) + SawOscillator(f * 1.006) * 0.7, "low_pass",
                         cutoff=cutoff * (env * 0.8 + 0.35), resonance=0.25) * env * 0.5
        drone = SVFilter(SawOscillator(n0 * 0.5) * 0.4 + SineOscillator(n0 * 0.5) * 0.7,
                         "low_pass", cutoff=320.0, resonance=0.0) * 0.45
        pad = None
        for i, n in enumerate((n0, n1, n2)):
            voice = SawOscillator(n * 2.0 * (1.0 + 0.002 * (i - 1))) * 0.18
            pad = voice if pad is None else pad + voice
        pad = SVFilter(pad, "low_pass", cutoff=1400.0, resonance=0.1) * Smooth(t["layer"], 0.9995)
        out = _hall(pulse + drone + pad, wet=0.25)
        return Tanh(out * Smooth(t["gain"], 0.999)), t

    def update(self, st):
        chord = self._set_chord(st)
        r = st.motor_ratio
        bpm = self.BPM_IDLE + (self.BPM_TOP - self.BPM_IDLE) * r
        self.targets["rate"].set_value(bpm / 30.0)
        self.targets["cutoff"].set_value(700.0 + 2500.0 * st.pedal + 1500.0 * r)
        self.targets["layer"].set_value(0.15 + 0.85 * st.pedal ** 1.2 + 0.5 * st.swell)
        self.targets["gain"].set_value(_ev_gain(st, self.LEVEL))
        return f"{chord} · {bpm:.0f} BPM"


class TrailerBrass(_ScoreVoice):
    """A low brass ensemble: a quiet bed when cruising that blasts -- filter opening, drive
    rising -- when you press. Power chord (root, fifth, octave) plus the chord's third."""
    name = "Trailer Brass"
    LEVEL = 1.8

    def build(self):
        t = {"blast": _target(0.0), "gain": _target(0.0)}
        n0, n1, n2 = self._notes(t)
        blast = Smooth(t["blast"], 0.9995)
        brass = None
        for f, amp in ((n0 * 0.5, 1.0), (n0 * 0.75, 0.75), (n0, 0.6), (n1 * 0.5, 0.35)):
            sec = (SawOscillator(f) + SawOscillator(f * 1.004) + SawOscillator(f * 0.996)) * amp
            brass = sec if brass is None else brass + sec
        brass = SVFilter(brass * 0.12, "low_pass", cutoff=blast * 2300.0 + 160.0, resonance=0.3)
        brass = Tanh(brass * (blast * 2.5 + 1.0)) * 0.7
        sub = SineOscillator(n0 * 0.5) * 0.35
        out = _hall(brass + sub, wet=0.35)
        return Tanh(out * Smooth(t["gain"], 0.999)), t

    def update(self, st):
        chord = self._set_chord(st)
        blast = min(1.0, max(st.pedal ** 1.3, st.swell))
        self.targets["blast"].set_value(blast)
        g = _ev_gain(st, self.LEVEL) * (0.45 + 0.55 * blast)
        self.targets["gain"].set_value(g)
        return f"{chord} · {'BLAST' if blast > 0.6 else 'bed'}"


class StringsChoir(_ScoreVoice):
    """Lush string section with an 'aah' choir. Strings: three detuned saws per note with
    their own vibrato; choir: saws through 'ah' formants. Pedal crescendos and brightens;
    the choir grows with speed."""
    name = "Strings & Choir"
    GLIDE = 0.999                     # a little portamento between chords
    VOWEL_AH = ((700.0, 1.0), (1220.0, 0.6), (2600.0, 0.25))
    LEVEL = 1.6

    def build(self):
        t = {"bright": _target(1500.0), "swell": _target(0.3), "choir": _target(0.2),
             "gain": _target(0.0)}
        n0, n1, n2 = self._notes(t)
        strings = None
        for i, (n, mult) in enumerate(((n0, 2.0), (n1, 2.0), (n2, 2.0), (n0, 4.0))):
            vib = SineLFO(4.8 + 0.37 * i, 0.9975, 1.0025)
            f = n * mult * vib
            sec = SawOscillator(f) + SawOscillator(f * 1.003) * 0.8 + SawOscillator(f * 0.997) * 0.8
            strings = sec if strings is None else strings + sec
        strings = SVFilter(strings * 0.07, "low_pass", cutoff=Smooth(t["bright"], 0.999),
                           resonance=0.1) * Smooth(t["swell"], 0.9995)
        voices_src = (SawOscillator(n0 * 2.0 * SineLFO(5.3, 0.998, 1.002))
                      + SawOscillator(n2 * 2.0 * SineLFO(4.9, 0.998, 1.002)) * 0.8) * 0.2
        choir = None
        for fc, amp in self.VOWEL_AH:
            band = SVFilter(voices_src, "band_pass", cutoff=fc, resonance=0.6) * amp
            choir = band if choir is None else choir + band
        choir = choir * Smooth(t["choir"], 0.9995)
        out = _hall(strings + choir, wet=0.45)
        return Tanh(out * Smooth(t["gain"], 0.999)), t

    def update(self, st):
        chord = self._set_chord(st)
        r = st.motor_ratio
        self.targets["bright"].set_value(1200.0 + 2600.0 * st.pedal ** 1.2 + 800.0 * r)
        self.targets["swell"].set_value(0.35 + 0.65 * st.pedal + 0.4 * st.swell)
        self.targets["choir"].set_value(0.15 + 0.85 * min(1.0, r * 1.5))
        self.targets["gain"].set_value(_ev_gain(st, self.LEVEL))
        return chord


class Cathedral(Voice):
    """A pipe organ in a big stone hall: serious and cinematic, built very differently from
    the saw-pad voices -- pure flue-pipe partials (sines), no filter sweeps.

    Speed draws stops, the way an organist builds a crescendo: a soft 8' and 16' at a crawl,
    then the 4', the quint (2 2/3'), the 2' and finally a mixture as speed climbs, so the
    sound grows taller and brighter rather than just higher. A celeste rank tuned a few
    cents sharp beats against the 8' -- the slow shimmer of a large organ. Pedal adds a
    reed rank (trumpet-like, the "full organ" weight) and the mixture sooner.
    The pitch still follows speed, a held note gliding like a motor: C2 (65 Hz) -> D3.
    """
    name = "Cathedral"
    F_IDLE, F_TOP = 65.41, 146.83
    LEVEL = 1.4
    # (rank, harmonic of the 8' pitch, level, speed (0..1) at which it is fully drawn)
    STOPS = (("16'", 0.5, 0.3, 0.0), ("8'", 1.0, 0.7, 0.0), ("4'", 2.0, 0.55, 0.25),
             ("2 2/3'", 3.0, 0.32, 0.45), ("2'", 4.0, 0.3, 0.6), ("mix 1 1/3'", 6.0, 0.18, 0.8),
             ("mix 1'", 8.0, 0.12, 0.9))

    def build(self):
        t = {"freq": _target(self.F_IDLE), "reed": _target(0.0), "gain": _target(0.0)}
        f = Smooth(t["freq"], 0.9995)
        ranks = None
        for i, (_, h, amp, _) in enumerate(self.STOPS):
            t[f"stop{i}"] = _target(1.0 if i < 2 else 0.0)
            pipe = SineOscillator(f * h) * (amp * 0.5) * Smooth(t[f"stop{i}"], 0.9997)
            ranks = pipe if ranks is None else ranks + pipe
        # a fifth above, on the 8' and 4' only: an open, unresolved interval
        quint = (SineOscillator(f * 1.5) * 0.25 + SineOscillator(f * 3.0) * 0.08) * 0.5
        celeste = SineOscillator(f * _cents(4.0)) * 0.3        # beats ~0.15-0.3 Hz
        reed_src = SawOscillator(f) + SawOscillator(f * 2.0 * _cents(-2.0)) * 0.5
        reed = SVFilter(reed_src, "low_pass", cutoff=f * 9.0, resonance=0.2) * 0.12 \
            * Smooth(t["reed"], 0.9995)
        wind = SVFilter(PinkNoise(), "band_pass", cutoff=f * 4.0, resonance=0.5) * 0.015
        out = _hall(ranks + quint + celeste + reed + wind, wet=0.45)
        return Tanh(out * Smooth(t["gain"], 0.999)), t

    def update(self, st):
        r, p = st.motor_ratio, st.pedal
        self.targets["freq"].set_value(self.F_IDLE * (self.F_TOP / self.F_IDLE) ** (r ** 0.85)
                                       * st.pitch_mul)
        drawn = min(1.0, r * 1.4 + 0.35 * p + 0.3 * st.swell)   # pedal draws stops sooner
        stops = 0
        for i, (_, _, _, at) in enumerate(self.STOPS):
            level = 1.0 if at == 0.0 else min(1.0, max(0.0, (drawn - at) / 0.15 + 1.0))
            self.targets[f"stop{i}"].set_value(level)
            stops += level > 0.5
        reed = min(1.0, max(0.0, (p - 0.3) / 0.6)) ** 1.3 + 0.4 * st.swell
        self.targets["reed"].set_value(reed)
        self.targets["gain"].set_value(_ev_gain(st, self.LEVEL))
        return f"{stops} stops" + (" + reed" if reed > 0.3 else "")


class GrandTouring(Voice):
    """Three layers, built the way production EV interior sounds are:

      base    order synthesis, pitched by speed. Integer and half orders of a "motor"
              frequency, with a separate harmonic recipe per load band (0-25, 25-50,
              50-75, 75-100 %), crossfaded -- so the pedal changes the timbre itself:
              round and calm when cruising, dense and metallic under full power.
      torque  the load layer: a sub-octave and 1.5-order growl with roughness -- a
              40-65 Hz amplitude flutter that hearing reads as strain -- and some grit.
              Gone when cruising, heavy when you press.
      shepard sines an octave apart sliding up a fixed bell, so speed keeps sounding like
              it is rising while the base stays low ("flowing energy").
    Load = smoothed pedal, plus a kick from quick presses. Motor frequency 55 -> 140 Hz.
    """
    name = "Grand Touring"
    F_IDLE, F_TOP = 55.0, 140.0
    LEVEL = 1.5
    ORDERS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)
    # amplitude of each order, per load band (cruise, light, firm, full)
    LOAD_MAPS = (
        (0.15, 1.0, 0.05, 0.50, 0.00, 0.25, 0.12, 0.00, 0.00, 0.00),
        (0.25, 1.0, 0.15, 0.60, 0.05, 0.35, 0.20, 0.08, 0.00, 0.00),
        (0.40, 1.0, 0.30, 0.75, 0.20, 0.60, 0.45, 0.30, 0.20, 0.10),
        (0.55, 1.0, 0.50, 0.90, 0.40, 0.80, 0.70, 0.55, 0.40, 0.28),
    )
    STRETCH = 0.0015          # high orders run slightly sharp: a faint metallic shimmer
    ROUGH_HZ = (40.0, 65.0)   # roughness flutter rate, light -> full load
    SHEP_PARTIALS, SHEP_BASE, SHEP_KMH = 6, 50.0, 30.0   # window peaks at 400 Hz

    def __init__(self):
        self.load = 0.0
        super().__init__()

    def build(self):
        t = {"freq": _target(self.F_IDLE), "torque": _target(0.0),
             "rough_hz": _target(self.ROUGH_HZ[0]), "rough": _target(0.0), "grit": _target(1.0),
             "shep": _target(0.0), "gain": _target(0.0)}
        f = Smooth(t["freq"], 0.9995)

        # base: one sine per order, amplitudes set per tick from the load maps
        drift = SineLFO(0.17, 0.92, 1.0)                    # slow life in the upper orders
        low = high = None
        for i, o in enumerate(self.ORDERS):
            t[f"o{i}"] = _target(self.LOAD_MAPS[0][i])
            stretch = 1.0 + self.STRETCH * max(0.0, o - 2.0)
            p = SineOscillator(f * (o * stretch)) * Smooth(t[f"o{i}"], 0.9995)
            if o > 2.0:
                high = p if high is None else high + p
            else:
                low = p if low is None else low + p
        # roughness: under load the upper orders flutter at 40-65 Hz along with the growl
        flutter = SineLFO(Smooth(t["rough_hz"], 0.999), 0.0, 1.0)
        rough = Smooth(t["rough"], 0.999)
        base = (low + high * drift * ((flutter - 1.0) * rough + 1.0)) * 0.22

        # torque: half-order growl, rough and gritty under load. Distortion comes BEFORE
        # the low-pass: a distorted waveform's edges are sharp, and an unfiltered 30 Hz
        # sub would tick 30 times a second -- heard as crackle.
        src = TriangleOscillator(f * 0.5) + SawOscillator(f * 1.5 * 1.003) * 0.5 \
            + TriangleOscillator(f * 1.002) * 0.5
        src = SVFilter(src, "low_pass", cutoff=f * 6.0, resonance=0.0)
        growl = SVFilter(Tanh(src * Smooth(t["grit"], 0.999)), "low_pass",
                         cutoff=f * 7.0, resonance=0.2)
        growl = growl * (flutter * 0.65 + 0.35)
        torque = growl * 0.4 * Smooth(t["torque"], 0.9993)

        # shepard: partial frequencies and levels set per tick (see update)
        shep = None
        for k in range(self.SHEP_PARTIALS):
            t[f"sf{k}"] = _target(self.SHEP_BASE * 2 ** k)
            t[f"sa{k}"] = _target(0.0)
            p = SineOscillator(Smooth(t[f"sf{k}"], 0.995)) * Smooth(t[f"sa{k}"], 0.999)
            shep = p if shep is None else shep + p
        # the top partials pass through 1-2 kHz on their way round: keep them out of earshot
        shep = SVFilter(shep, "low_pass", cutoff=900.0, resonance=0.0) * 0.07 \
            * Smooth(t["shep"], 0.999)

        out = _hall(base + torque + shep, wet=0.15)
        return Tanh(out * Smooth(t["gain"], 0.999)), t

    def update(self, st):
        r, tg = st.motor_ratio, self.targets
        f = self.F_IDLE * (self.F_TOP / self.F_IDLE) ** (r ** 0.85) * st.pitch_mul
        tg["freq"].set_value(f)

        target = min(1.0, st.pedal + 0.5 * st.swell)
        tau = 0.12 if target > self.load else 0.35          # builds fast, eases off slower
        self.load += (target - self.load) * min(1.0, st.dt / tau)
        load = self.load

        # base: crossfade the two load maps either side of the current load
        x = load * (len(self.LOAD_MAPS) - 1)
        lo = min(int(x), len(self.LOAD_MAPS) - 2)
        w = x - lo
        for i in range(len(self.ORDERS)):
            a = self.LOAD_MAPS[lo][i] * (1.0 - w) + self.LOAD_MAPS[lo + 1][i] * w
            tg[f"o{i}"].set_value(a)

        heavy = min(1.0, max(0.0, (load - 0.2) / 0.8)) ** 1.3
        tg["torque"].set_value(heavy)
        tg["rough_hz"].set_value(self.ROUGH_HZ[0] + (self.ROUGH_HZ[1] - self.ROUGH_HZ[0]) * heavy)
        tg["grit"].set_value(1.0 + 2.5 * heavy)
        tg["rough"].set_value(0.6 * heavy)

        # each partial climbs on its own and wraps from the top back to the bottom where the
        # window is silent; wrapping them all at once would sweep every audible partial
        # down an octave in a few ms -- a chirp each time
        n, climb = self.SHEP_PARTIALS, st.speed_kmh / self.SHEP_KMH
        for k in range(n):
            pos = (k + climb) % n
            tg[f"sf{k}"].set_value(self.SHEP_BASE * 2 ** pos)
            tg[f"sa{k}"].set_value(math.sin(math.pi * pos / n) ** 4)   # narrow: no high whine
        tg["shep"].set_value(min(1.0, st.speed_kmh / 15.0) * (0.6 + 0.4 * load))

        tg["gain"].set_value(_ev_gain(st, self.LEVEL))
        band = ("cruise", "light", "firm", "full")[min(3, int(load * 4))]
        return f"load {load * 100:3.0f}% ({band})"


class Dreadnought(Voice):
    """Dark sci-fi machine: sound design, not music -- a huge craft's drive, no chords.

      hull hum    a low hum with inharmonic partials (1, 2.03, 3.17, 4.41, 5.93): metal,
                  not an instrument. Each partial wanders in level on its own, slowly and
                  irregularly, like big machinery. 34 -> 85 Hz with speed.
      resonance   noise ringing two comb resonators tuned to clashing, non-harmonic
                  pitches -- the hull itself singing. Load tightens their feedback, so
                  pressing makes the metal ring harder and longer.
      servo       a resonant whine that follows ACCELERATION, not speed: rising while
                  you speed up, sagging as you slow, silent when steady.
      surge       a throb under load (6 -> 11 Hz pulsing), the drive straining.
      air         dark air handling that grows with speed.
    """
    name = "Dreadnought"
    F_IDLE, F_TOP = 34.0, 85.0
    LEVEL = 1.6
    HUM = ((1.0, 0.5), (2.03, 0.7), (3.17, 0.5), (4.41, 0.4), (5.93, 0.3))
    RES_RATIOS = (2.71, 4.13)            # comb pitches relative to the hum: clashing metal

    def __init__(self):
        self._prev_speed = None
        self.accel = 0.0
        self.load = 0.0
        super().__init__()

    def build(self):
        t = {"freq": _target(self.F_IDLE), "hum": _target(1.0), "res_fb": _target(0.85),
             "res": _target(0.3),
             "servo_f": _target(300.0), "servo": _target(0.0), "surge": _target(0.0),
             "throb": _target(6.0), "air": _target(0.05), "air_f": _target(300.0),
             "gain": _target(0.0)}
        f = Smooth(t["freq"], 0.9995)

        hum = None
        for i, (ratio, amp) in enumerate(self.HUM):
            # irregular, slow level wander: a new random target every ~0.3-0.7 s, glided
            wander = Smooth(RandomUniform(0.45, 1.0, clock=Impulse(1.4 + 0.53 * i)), 0.99995)
            p = SineOscillator(f * ratio) * (amp * wander)
            hum = p if hum is None else hum + p
        hum = hum * 0.3 * Smooth(t["hum"], 0.999)

        exc = SVFilter(WhiteNoise(), "band_pass", cutoff=f * 10.0, resonance=0.2) * 0.17
        fb = Smooth(t["res_fb"], 0.999)
        res = None
        for ratio in self.RES_RATIOS:
            c = CombDelay(exc, delay_time=1.0 / (f * ratio), feedback=fb, max_delay_time=0.05)
            res = c if res is None else res + c
        res = SVFilter(SVFilter(res, "low_pass", cutoff=1800.0, resonance=0.0),
                       "low_pass", cutoff=1800.0, resonance=0.0) * Smooth(t["res"], 0.999)

        servo_f = Smooth(t["servo_f"], 0.9995)
        servo = SVFilter(SawOscillator(servo_f) + SawOscillator(servo_f * 1.007), "band_pass",
                         cutoff=servo_f * 2.0, resonance=0.6)
        servo = SVFilter(servo, "low_pass", cutoff=1500.0, resonance=0.0) \
            * 0.05 * Smooth(t["servo"], 0.999)

        throb = SineLFO(Smooth(t["throb"], 0.999), 0.2, 1.0)
        # pitched where it is heard (2nd/3rd harmonic), not as sub that only eats headroom
        surge = (TriangleOscillator(f * 2.0) + SineOscillator(f * 3.0) * 0.5) * throb \
            * 0.3 * Smooth(t["surge"], 0.9995)

        air = SVFilter(PinkNoise(), "low_pass", cutoff=Smooth(t["air_f"], 0.999),
                       resonance=0.1) * 18.0 * Smooth(t["air"], 0.999)

        # a dark machine: nothing above ~3 kHz, so no hiss or edge at full load
        mix = SVFilter(hum + res + servo + surge + air, "low_pass", cutoff=3000.0, resonance=0.0)
        verb = FeedbackDelayNetwork(mix, num_delays=8, feedback=0.88, damping=0.7, wet=1.0,
                                    dry=0.0, max_delay_time=0.25)
        return Tanh((mix + verb * 0.25) * Smooth(t["gain"], 0.999)), t

    def update(self, st):
        r, tg = st.motor_ratio, self.targets
        f = self.F_IDLE * (self.F_TOP / self.F_IDLE) ** (r ** 0.85) * st.pitch_mul
        tg["freq"].set_value(f)

        # acceleration in km/h per s, smoothed; the servo answers to this
        if self._prev_speed is not None and st.dt > 0:
            a = (st.speed_kmh - self._prev_speed) / st.dt
            self.accel += (a - self.accel) * min(1.0, st.dt / 0.3)
        self._prev_speed = st.speed_kmh
        acc = max(-1.0, min(1.0, self.accel / 12.0))

        target = min(1.0, st.pedal + 0.5 * st.swell)
        self.load += (target - self.load) * min(1.0, st.dt / (0.12 if target > self.load else 0.4))
        load = self.load

        tg["res_fb"].set_value(0.86 + 0.1 * load)
        tg["res"].set_value(0.25 + 0.5 * load + 0.2 * r)
        tg["servo_f"].set_value(max(60.0, 180.0 + 120.0 * r + 160.0 * acc))
        tg["servo"].set_value(min(1.0, abs(acc) * 1.3))
        heavy = min(1.0, max(0.0, (load - 0.3) / 0.7))
        tg["surge"].set_value(heavy)
        tg["throb"].set_value(6.0 + 5.0 * heavy)
        tg["air"].set_value(0.03 + 0.14 * r)
        tg["air_f"].set_value(250.0 + 900.0 * r)
        tg["gain"].set_value(_ev_gain(st, self.LEVEL))
        mood = "surging" if heavy > 0.4 else ("spooling" if acc > 0.25 else
                                               "winding down" if acc < -0.25 else "humming")
        return f"{mood}  {f:3.0f} Hz"


class JetTurbine(Voice):
    """A turbofan, scaled down to stay out of whine territory. No music in it at all.

    The engine has its own spool speed (N1), which lags the throttle like a real turbine:
    pressing winds it up over a second or so, lifting off lets it run down slowly. Idles
    while stopped. Everything is driven by the spool, not directly by the pedal:
      shaft     the low hum of the rotating spool, with two spools (N1/N2) beating
      blades    a soft fan blade-pass tone, kept around 400-1000 Hz and low in level
      buzzsaw   near full power, a rough chord of shaft harmonics -- the ripping,
                saw-like sound of a fan at takeoff thrust
      intake    resonant air howl that rises with the spool
      exhaust   broadband roar with slow turbulent swirl, plus the low popping crackle
                of a jet exhaust under high thrust
    Dry -- no hall: it is outdoors and mechanical.
    """
    name = "Jet Turbine"
    LEVEL = 1.5
    IDLE_N1 = 0.22
    SPOOL_UP_SEC, SPOOL_DOWN_SEC = 0.9, 2.2
    BLADES = 11
    BUZZ = ((2.0, 0.9), (3.0, 0.6), (4.0, 0.75), (5.0, 0.45), (6.0, 0.5), (7.0, 0.3), (8.0, 0.35))

    def __init__(self):
        self.n1 = 0.0
        super().__init__()

    def build(self):
        t = {"shaft": _target(40.0), "hum": _target(1.0), "blade": _target(0.1), "buzz": _target(0.0),
             "intake_f": _target(500.0), "intake": _target(0.2), "roar_f": _target(250.0),
             "roar": _target(0.2), "crackle_p": _target(0.0), "crackle": _target(0.0),
             "gain": _target(0.0)}
        shaft = Smooth(t["shaft"], 0.9995)

        n2 = shaft * 1.37                                   # the core spool runs faster
        hum = (SineOscillator(shaft) + SineOscillator(shaft * 2.0) * 0.5
               + TriangleOscillator(n2) * 0.35 + SineOscillator(n2 * 2.01) * 0.2) * 0.12 \
            * Smooth(t["hum"], 0.999)

        bpf = shaft * self.BLADES                           # blade-pass, low and soft
        blades = (SineOscillator(bpf) + SineOscillator(bpf * 1.006) * 0.8
                  + SineOscillator(n2 * 7.0) * 0.4) * 0.009 * Smooth(t["blade"], 0.999)

        buzz = None
        for h, a in self.BUZZ:
            p = SawOscillator(shaft * h * (1.0 + 0.0013 * h)) * a
            buzz = p if buzz is None else buzz + p
        buzz = SVFilter(Tanh(SVFilter(buzz * 0.25, "low_pass", cutoff=shaft * 9.0, resonance=0.0) * 2.0),
                        "low_pass", cutoff=1400.0, resonance=0.0) * 0.3 * Smooth(t["buzz"], 0.999)

        intake = SVFilter(WhiteNoise(), "band_pass", cutoff=Smooth(t["intake_f"], 0.999),
                          resonance=0.5) * 0.22 * Smooth(t["intake"], 0.999)

        swirl = Smooth(RandomUniform(0.55, 1.0, clock=Impulse(7.0)), 0.9997)
        roar = SVFilter(PinkNoise(), "low_pass", cutoff=Smooth(t["roar_f"], 0.999),
                        resonance=0.15) * swirl * 9.0 * Smooth(t["roar"], 0.999)

        chance = RandomUniform(0.0, 1.0, clock=Impulse(35.0))
        gate = Clip((chance - (Smooth(t["crackle_p"], 0.99) * -1.0 + 1.0)) * 1000.0, 0.0, 1.0)
        crackle = SVFilter(WhiteNoise(), "band_pass", cutoff=520.0, resonance=0.3) \
            * ASREnvelope(0.0005, 0.003, 0.03, curve=2.0, clock=gate) * 1.4 \
            * Smooth(t["crackle"], 0.999)

        mix = hum + blades + buzz + intake + roar + crackle
        mix = SVFilter(mix, "low_pass", cutoff=3000.0, resonance=0.0)
        return Tanh(mix * Smooth(t["gain"], 0.999)), t

    def update(self, st):
        r, tg = st.motor_ratio, self.targets
        load = min(1.0, st.pedal + 0.4 * st.swell)
        target = self.IDLE_N1 + (1.0 - self.IDLE_N1) * min(1.0, 0.75 * load + 0.35 * r)
        tau = self.SPOOL_UP_SEC if target > self.n1 else self.SPOOL_DOWN_SEC
        self.n1 += (target - self.n1) * min(1.0, st.dt / tau)
        n1 = self.n1

        tg["shaft"].set_value((32.0 + 58.0 * n1) * st.pitch_mul)        # 39 -> 90 Hz
        tg["blade"].set_value(0.4 + 0.6 * n1)
        thrust = min(1.0, max(0.0, (n1 - 0.65) / 0.3))                  # takeoff region
        tg["buzz"].set_value(thrust ** 1.3)
        tg["intake_f"].set_value(380.0 + 650.0 * n1)
        tg["intake"].set_value(0.3 + 0.7 * n1)
        tg["roar_f"].set_value(160.0 + 520.0 * n1 + 250.0 * r)
        tg["roar"].set_value(0.15 + 0.55 * n1 ** 1.5 + 0.3 * r)
        tg["crackle_p"].set_value(0.25 * thrust)
        tg["crackle"].set_value(thrust)
        tg["gain"].set_value(_ev_gain(st, self.LEVEL, idle_level=0.6))
        return f"N1 {n1 * 100:3.0f}%" + ("  TAKEOFF" if thrust > 0.5 else "")


def _interp_log(x, points):
    """Piecewise interpolation of log(y) over x; points = ((x0, y0), (x1, y1), ...)."""
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            u = (x - x0) / (x1 - x0)
            return math.exp(math.log(y0) + u * (math.log(y1) - math.log(y0)))
    return points[-1][1]


def _interp(x, points):
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            return y0 + (x - x0) / (x1 - x0) * (y1 - y0)
    return points[-1][1]


class JetsonsI3(Voice):
    """The Jetsons car as a BMW i3 sound mod plays it, modelled on that recording.

    Its character is a PULSE TRAIN whose rate climbs with speed, so the sound changes
    kind, not just pitch (measured regimes in the recording):
        stopped     ~2 Hz   separate "blip ... blip" of a ~530 Hz tone, fully gated
        creeping    ~5 Hz   faster blips
        slow        ~10 Hz  a putter
        ~18 km/h    ~34 Hz  the pulses fuse into a low buzz -- rhythm has become pitch
        cruising    75-103+ a hum whose 7th harmonic is loudest, with a gentle ripple
    Once fused, the blip tone locks to 6.84x the pulse rate. Being just under the 7th
    harmonic, it makes h7 the strongest partial and beats against it at 0.16x the rate,
    which is the warble -- quickening as speed rises. A harmonic body (h1..h4) fades in
    as the pulses become a pitch. The wandering 1.3-2.5 kHz whistle comes in on quick
    pedal presses and when slowing down, where the recording has it.
    Speeds are a mapping choice (the recording has none); the regimes and their order are
    measured.
    """
    name = "Jetsons i3"
    # speed km/h -> pulse rate Hz (log-interpolated)
    RATE = ((0, 2.3), (2, 3.5), (5, 5.0), (9, 9.0), (14, 20.0), (18, 34.0), (24, 55.0),
            (30, 75.0), (45, 103.0), (70, 125.0), (100, 140.0))
    # pulse rate -> gating depth (1 = silent between pulses) and pulse sharpness
    DEPTH = ((3, 1.0), (4, 0.75), (9, 0.45), (20, 0.4), (34, 0.35), (75, 0.3))
    SHARP = ((5, 5.0), (9, 3.5), (34, 1.5), (75, 1.0))
    BLIP_HZ = ((2.3, 530.0), (5, 530.0), (34, 232.0))   # blip tone floor before fusing
    BEAT_RATIO = 6.84
    BODY = (0.25, 0.55, 0.5, 0.32, 0.15, 0.25)          # h1..h6 under the blip tone
    LEVEL = 0.5

    def __init__(self):
        super().__init__()
        self._prev_speed = None
        self._decel = 0.0

    def build(self):
        t = {"rate": _target(2.3), "blip": _target(530.0), "depth": _target(1.0),
             "sharp": _target(5.0), "body": _target(0.0), "cutoff": _target(4000.0),
             "whistle": _target(0.0), "gain": _target(0.0)}
        rate = Smooth(t["rate"], 0.9995)
        drift = SineLFO(3.0, 1.0 / _cents(15), _cents(15))       # slight irregularity
        pulse = Pow(SineLFO(rate, 0.0, 1.0), Smooth(t["sharp"], 0.999))
        depth = Smooth(t["depth"], 0.999)
        gate = (1.0 - depth) + depth * pulse

        # Blip tone: a slightly bright tone that each pulse sounds. A small upward chirp
        # inside each pulse gives the 'bwip' rather than a flat beep.
        blip_f = Smooth(t["blip"], 0.9995) * drift * (pulse * 0.03 + 1.0)
        blip = SineOscillator(blip_f) + SineOscillator(blip_f * 2.0) * 0.25
        # Body: harmonics of the pulse rate itself, only once that rate is a pitch
        body = None
        for h, amp in enumerate(self.BODY, start=1):
            osc = SineOscillator(rate * drift * float(h)) * amp
            body = osc if body is None else body + osc
        body = body * Smooth(t["body"], 0.999)

        wander = SineLFO(6.5, 0.75, 1.3) * SineLFO(2.3, 0.9, 1.1)
        whistle = SVFilter(SineOscillator(wander * 1900.0), "low_pass", cutoff=3000.0,
                           resonance=0.0) * Smooth(t["whistle"], 0.9995)
        mix = (blip + body) * gate + whistle
        out = SVFilter(mix, "low_pass", cutoff=Smooth(t["cutoff"], 0.999), resonance=0.0) \
            * Smooth(t["gain"], 0.999)
        return out, t

    def update(self, st):
        speed = max(0.0, st.speed_kmh)
        rate = _interp_log(speed, self.RATE)
        self.targets["rate"].set_value(rate * st.pitch_mul)
        self.targets["blip"].set_value(st.pitch_mul * max(self.BEAT_RATIO * rate,
                                           _interp_log(rate, self.BLIP_HZ)))
        self.targets["depth"].set_value(_interp(rate, self.DEPTH))
        self.targets["sharp"].set_value(_interp(rate, self.SHARP))
        self.targets["body"].set_value(min(1.0, max(0.0, (rate - 18.0) / 16.0)))
        self.targets["cutoff"].set_value(3500.0 + 4000.0 * st.pedal ** 1.3)

        # deceleration (km/h per s), smoothed: the recording whistles while slowing
        if self._prev_speed is not None and st.dt > 0:
            d = max(0.0, (self._prev_speed - speed) / st.dt)
            self._decel += (d - self._decel) * min(1.0, st.dt / 0.3)
        self._prev_speed = speed
        self.targets["whistle"].set_value(0.22 * st.swell + 0.18 * min(1.0, self._decel / 8.0))

        # quieter while blipping at a standstill, like the recording (~9 dB under cruise);
        # the gating already removes most of the energy there, so this lifts it back up
        moving = min(1.0, speed / 10.0)
        g = 0.0 if not st.active else (0.35 + 0.65 * st.pedal) * (1.75 - 0.75 * moving) \
            * (1.0 + 0.4 * st.swell) * self.LEVEL
        self.targets["gain"].set_value(g)
        return f"pulse {rate:5.1f} Hz"


PROJECT_DIR = Path(__file__).parent
MAPS_DIR = PROJECT_DIR / "maps"          # one <audio name>.json per mapped recording
SOUNDS_DIR = PROJECT_DIR / "sounds"      # audio converted by ffmpeg lands here
AUDIO_TYPES = (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".aiff", ".aif")


def load_audio_buffer(path):
    """Load any audio file as a signalflow Buffer; converts via ffmpeg if needed."""
    path = Path(path)
    try:
        return Buffer(str(path)), path
    except Exception:
        import shutil
        import subprocess
        if not shutil.which("ffmpeg"):
            raise
        SOUNDS_DIR.mkdir(exist_ok=True)
        out = SOUNDS_DIR / (path.stem + ".wav")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-ac", "2", str(out)],
                       check=True)
        return Buffer(str(out)), out


class RecordingVoice(Voice):
    """Plays any recording, with the position in it chosen by speed (see mapper.py).

    A map is two lists of (km/h, seconds into the file): one used while speeding up, one
    while slowing down. Holding a speed holds the position; overlapping grains keep it
    sounding at the recording's own pitch. Past the top of the map the grains pitch up.
    Maps live in maps/<audio name>.json and every mapped file becomes a voice in main.py.
    """
    TARGET_RMS = 0.1458      # every file is normalised to the i3 recording's loudness
    LEVEL = 4.0              # grains are centre-panned (-6 dB) and recordings are peaky
    DIR_WINDOW_SEC, DIR_THRESHOLD = 0.8, 2.0   # see RecordingI3
    GROUPS: tuple = ()       # putter pulse onsets (only measured for the i3 recording)
    GROUP_KMH: tuple = ()

    def __init__(self, file=None, name=None):
        self._hist = []              # (time, speed) over the direction window
        self._t = 0.0
        self._decel = 0.0
        self.audition = None         # file position to hold (mapper), or None
        self.cursor = 0.0            # file position currently heard (for the mapper)
        self._buf, self.file = load_audio_buffer(file or self.FILE)
        self.source = Path(file or self.FILE)
        self.name = name or getattr(type(self), "NAME", None) or f"Recording: {self.source.stem}"
        self.map_path = MAPS_DIR / f"{self.source.stem}.json"
        self._analyse()
        self.set_map(*self.default_map())
        self.load_map()
        super().__init__()

    def default_map(self):
        """A plain ramp through the file, for audio that hasn't been mapped yet."""
        d = self.duration
        return ((0.0, 0.05 * d), (50.0, 0.9 * d)), (), 0.0, 1.5

    @classmethod
    def from_map(cls, map_path):
        d = json.loads(Path(map_path).read_text())
        return cls(file=d["file"], name=d.get("name"))

    @staticmethod
    def saved_maps():
        """(name, map path) for every mapped recording except the built-in i3 one."""
        out = []
        for p in sorted(MAPS_DIR.glob("*.json")) if MAPS_DIR.exists() else []:
            try:
                d = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            if Path(d.get("file", "")).exists() and \
                    Path(d["file"]).resolve() != Path(RecordingI3.FILE).resolve():
                out.append((d.get("name") or f"Recording: {Path(d['file']).stem}", p))
        return out

    # -- map ---------------------------------------------------------------------------
    def set_map(self, accel, decel, pulse_until=None, hum_from=None):
        """accel/decel: iterables of (km/h, seconds into the file)."""
        self.accel = sorted((float(k), float(s)) for k, s in accel)
        self.decel = sorted((float(k), float(s)) for k, s in decel)
        if pulse_until is not None:
            self.pulse_until = float(pulse_until) if self.GROUPS else 0.0
        if hum_from is not None:
            self.hum_from = max(float(hum_from), self.pulse_until + 0.1)
        self.top_kmh = self.accel[-1][0]
        self.decel_top = self.decel[-1][0] if self.decel else 0.0
        end = self.accel[-1][1]
        self.plateau = (end, min(self.duration - 0.05, end + 1.0))

    def map_dict(self):
        return {"file": str(self.source.resolve()), "name": self.name,
                "accel": self.accel, "decel": self.decel,
                "pulse_until": self.pulse_until, "hum_from": self.hum_from}

    def load_map(self, path=None):
        path = Path(path or self.map_path)
        if path.exists():
            d = json.loads(path.read_text())
            self.set_map(d["accel"], d.get("decel", []), d.get("pulse_until"),
                         d.get("hum_from"))
            return True
        return False

    def save_map(self, path=None):
        path = Path(path or self.map_path)
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(self.map_dict(), indent=1))
        return path

    def _analyse(self):
        """Mono copy normalised to TARGET_RMS, plus a 50 ms level track."""
        sr = int(self._buf.sample_rate)
        mono = np.array(self._buf.data).mean(axis=0)
        self.duration = len(mono) / sr
        hop = sr // 20
        frames = mono[:len(mono) // hop * hop].reshape(-1, hop)
        rms = np.sqrt(np.mean(frames ** 2, axis=1))
        typical = float(np.median(rms[rms > 1e-3])) if np.any(rms > 1e-3) else 1.0
        mono = mono * (self.TARGET_RMS / typical)
        self._mono = Buffer(1, len(mono), [mono.tolist()])
        self._mono.sample_rate = sr
        self._rms = rms * (self.TARGET_RMS / typical) + 1e-4
        self._rms_ref = self.TARGET_RMS

    def _level_at(self, pos):
        i = min(len(self._rms) - 1, max(0, int(pos * 20)))
        return float(self._rms[i])

    def build(self):
        buf = self._mono
        t = {"pos_a": _target(6.0), "pos_d": _target(4.0), "dur": _target(0.2),
             "density": _target(15.0), "rate": _target(1.0), "w_dec": _target(0.0),
             "gain": _target(0.0)}
        clock = Impulse(Smooth(t["density"], 0.999))
        jitter = RandomUniform(-0.02, 0.02, clock=clock)
        dur = Smooth(t["dur"], 0.999)
        rate = Smooth(t["rate"], 0.999)
        acc = Granulator(buf, clock=clock, pos=Smooth(t["pos_a"], 0.9995) + jitter,
                         duration=dur, rate=rate)
        dec = Granulator(buf, clock=clock, pos=Smooth(t["pos_d"], 0.9995) + jitter,
                         duration=dur, rate=rate)
        w = Smooth(t["w_dec"], 0.9995)
        # three grains overlap at any moment, so scale back to the recording's level
        cloud = (acc * (1.0 - w) + dec * w) * 0.45

        # Triplet mode: one flat-topped grain per blip group, fired at the group's own
        # period, so the rhythm is replayed intact instead of smeared by overlapping grains.
        # Position is not smoothed: each grain must start exactly at a group onset.
        t["p_pos"], t["p_rate"], t["p_dur"], t["w_pulse"] = (
            _target(self.GROUPS[0][0] if self.GROUPS else 0.0),
            _target(1.0 / self.GROUPS[0][1] if self.GROUPS else 1.0),
            _target(self.GROUPS[0][1] if self.GROUPS else 0.1), _target(0.0))
        n = 512
        env = np.ones(n)
        ramp = int(n * 0.04)
        env[:ramp] = np.linspace(0.0, 1.0, ramp)
        env[-ramp:] = np.linspace(1.0, 0.0, ramp)
        pulse = Granulator(buf, clock=Impulse(t["p_rate"]), pos=t["p_pos"],
                           duration=t["p_dur"])
        pulse.set_buffer("envelope", EnvelopeBuffer(env.tolist()))
        wp = Smooth(t["w_pulse"], 0.999)
        out = (pulse * 0.9 * wp + cloud * (1.0 - wp)) * Smooth(t["gain"], 0.999)
        self._pulse_node = pulse            # keep a reference alive
        return out, t

    def update(self, st):
        speed = max(0.0, st.speed_kmh)
        # Direction from the speed change over the last 0.6 s: a steady crawl wobbling by
        # a km/h or two (or whole-km/h steps from the car) mustn't flip between stretches.
        self._t += st.dt
        self._hist.append((self._t, speed))
        while self._hist and self._t - self._hist[0][0] > self.DIR_WINDOW_SEC:
            self._hist.pop(0)
        span = self._t - self._hist[0][0]
        dv = (speed - self._hist[0][1]) / span if span > 0.2 else 0.0
        if dv < -self.DIR_THRESHOLD:
            want = 1.0
        elif dv > self.DIR_THRESHOLD:
            want = 0.0
        else:
            want = 1.0 if self._decel > 0.5 else 0.0     # hold the current stretch
        if speed < 0.5:
            want = 0.0      # stopped (and muted): pulling away always starts accelerating
        want *= min(1.0, max(0.0, (self.decel_top + 2.0 - speed) / 2.0)) if self.decel else 0.0
        self._decel += (want - self._decel) * min(1.0, st.dt / 0.3)

        if speed <= self.top_kmh:
            pos_a, rate = _interp(speed, self.accel), 1.0
        else:                                   # past the recording: wander its plateau, pitch up
            u = min(1.0, (speed - self.top_kmh) / 25.0)
            pos_a = self.plateau[0] + u * (self.plateau[1] - self.plateau[0])
            rate = _interp_log(speed, JetsonsI3.RATE) / _interp_log(self.top_kmh, JetsonsI3.RATE)
        pos_d = _interp(min(speed, self.decel_top), self.decel) if self.decel else pos_a
        # longer grains for the putter, short ones once it is a buzz/hum (from ~4 km/h)
        dur = 0.2 if speed < 3 else (0.12 if speed > 8 else 0.2 - (speed - 3) / 5 * 0.08)

        self.targets["pos_a"].set_value(pos_a)
        self.targets["pos_d"].set_value(pos_d)
        self.targets["rate"].set_value(rate * st.pitch_mul)
        self.targets["dur"].set_value(dur)
        self.targets["density"].set_value(3.0 / dur)
        self.targets["w_dec"].set_value(self._decel)

        if self.GROUPS:
            gi = max(i for i, k in enumerate(self.GROUP_KMH) if speed >= k)
            start, period = self.GROUPS[gi]
        else:
            start, period = 0.0, 0.1
        self.targets["p_pos"].set_value(max(0.0, start - 0.012))
        self.targets["p_rate"].set_value(1.0 / period)
        self.targets["p_dur"].set_value(period)
        wp = 0.0 if self.pulse_until <= 0 else \
            min(1.0, max(0.0, (self.hum_from - speed) / (self.hum_from - self.pulse_until)))

        if self.audition is not None:           # mapper: hold one spot of the file
            pos_a, pos_d, wp, rate, dur = self.audition, self.audition, 0.0, 1.0, 0.15
            self._decel = 0.0
            self.targets["pos_a"].set_value(pos_a)
            self.targets["pos_d"].set_value(pos_d)
            self.targets["rate"].set_value(1.0)
            self.targets["dur"].set_value(dur)
            self.targets["density"].set_value(3.0 / dur)
            self.targets["w_dec"].set_value(0.0)
        self.targets["w_pulse"].set_value(wp)
        self.cursor = pos_a if self._decel <= 0.5 else pos_d

        # keep the recording's dynamics but halve its swings (in dB)
        lvl = self._level_at(pos_a) * (1 - self._decel) + self._level_at(pos_d) * self._decel
        corr = min(2.0, max(0.5, (self._rms_ref / lvl) ** 0.5))
        active = st.active or self.audition is not None
        g = 0.0 if not active else (0.6 + 0.4 * st.pedal) * corr * self.LEVEL
        self.targets["gain"].set_value(g)
        return f"{'slowing' if self._decel > 0.5 else 'speeding'} @ {pos_a if self._decel <= 0.5 else pos_d:4.2f}s"


class RecordingI3(RecordingVoice):
    """The i3 recording with its hand-measured defaults (the original recording voice).

    The recording contains a clean slowing-down stretch and a clean speeding-up stretch
    (measured pitch/pulse track below). Speeding up reads from the accelerating stretch,
    slowing down from the decelerating one, with a crossfade when you switch; holding a
    speed holds the position. Overlapping grains keep it sounding while held, and play
    at their recorded pitch, so pitch, pulsing and timbre at each speed are the
    recording's own -- including the blip -> putter -> buzz -> hum progression.
    Above the recording's top (103 Hz, mapped to 45 km/h) the grains are pitched up.
    """
    name = NAME = "Jetsons i3 (recording)"
    FILE = str(Path(__file__).with_name("i3 with Jetsons Sound.mp3"))
    LEGACY_MAP = Path(__file__).with_name("recording_map.json")   # pre-maps/ location
    # speed km/h -> seconds into the file. Pulling away and slow driving is 6-8 s of the
    # recording (putter -> fusing buzz -> rising), per listening; the 4.8-6 s
    # "bip-bip-bip" section is not part of it. Measured along the accelerating stretch:
    #   6.0 s putter 8.5 Hz, 6.2 putter 10.5 Hz, 6.35 buzz ~34 Hz, 6.5 44 Hz, 6.7 50 Hz,
    #   7.0 60 Hz, 7.3 70 Hz, 7.6 77.5 Hz, 8.0 87 Hz, 8.5 93 Hz, 9.0 97.5 Hz, 10.1 103 Hz
    # The speeds themselves are an estimate (the recording carries no speed data).
    ACCEL = ((0, 6.0), (2, 6.2), (4, 6.35), (7, 6.5), (10, 6.7), (14, 7.0), (18, 7.3),
             (22, 7.6), (27, 8.0), (33, 8.5), (40, 9.0), (50, 10.1))
    # Slowing-down stretch, aligned so each speed has the same pitch as when speeding up:
    #   1.6 s 72.5 Hz, 2.0 66, 2.3 60.5, 2.6 55.5, 3.0 50, 3.3 44, 3.5 40, 3.7 34 (buzz),
    #   3.95 ~29, 4.05 putter 9 Hz, 4.2 6 Hz, 4.35 slowing blips, 4.5 stop
    DECEL = ((0, 4.5), (0.5, 4.35), (1, 4.2), (2, 4.05), (3, 3.95), (4, 3.7), (5.8, 3.5),
             (7, 3.3), (10, 3.0), (12.2, 2.6), (14.2, 2.3), (16.4, 2.0), (19.3, 1.6))
    DECEL_TOP_KMH = 19.3       # the recording only decelerates from ~73 Hz
    TOP_KMH, TOP_PULSE_HZ, PLATEAU = 50.0, 103.0, (10.1, 11.1)
    # Putter at a crawl: single pulses, measured onsets (s) and spacing (s); one
    # flat-topped grain per pulse keeps the rhythm intact (overlapping grains smear it).
    GROUPS = ((6.045, 0.071), (6.116, 0.097), (6.213, 0.087))
    GROUP_KMH = (0.0, 1.0, 2.0)
    PULSE_UNTIL, HUM_FROM = 2.5, 4.0       # crossfade from pulse mode to grain cloud
    # km/h per s over the window: one whole-km/h step from the car reads as 1.25,
    # gentle braking is ~3+
    DIR_WINDOW_SEC, DIR_THRESHOLD = 0.8, 2.0
    def __init__(self, file=None, name=None):
        super().__init__(file, name)

    def default_map(self):
        return self.ACCEL, self.DECEL, self.PULSE_UNTIL, self.HUM_FROM

    def load_map(self, path=None):
        if path is None and not self.map_path.exists() and self.LEGACY_MAP.exists():
            path = self.LEGACY_MAP
        return super().load_map(path)

# ---------------------------------------------------------------------------------------
# Combustion engines
# ---------------------------------------------------------------------------------------

class VirtualGearbox:
    """Road speed + pedal -> engine rpm and gear, like a 6-speed automatic.

    Upshifts later the harder you press, kicks down when you floor it, slips a clutch
    (torque converter) at low speed and revs freely at a standstill.
    """
    RPM_PER_KMH = (120.0, 72.0, 50.0, 38.0, 31.0, 26.0)
    SHIFT_CUT_SEC = 0.14      # load is cut briefly on upshift: the audible gap

    def __init__(self, idle=780.0, redline=6200.0):
        self.idle, self.redline = idle, redline
        self.gear = 1
        self.rpm = idle
        self.cut = 0.0
        self.load = 0.0

    def step(self, speed, pedal, dt):
        up_at = min(self.redline - 400.0, 2300.0 + 3600.0 * pedal ** 1.2)
        down_at = 1150.0 + 2200.0 * pedal ** 1.5
        geared = speed * self.RPM_PER_KMH[self.gear - 1]
        # no upshifts while coasting off the pedal: like a real automatic it holds the gear
        # (engine braking, overrun crackle) and only downshifts as the car slows
        if self.gear < 6 and geared > up_at and pedal > 0.08:
            self.gear += 1
            self.cut = self.SHIFT_CUT_SEC
        elif self.gear > 1 and geared < down_at:
            self.gear -= 1
        geared = speed * self.RPM_PER_KMH[self.gear - 1]

        if speed < 1.0:
            target = self.idle + pedal * 3200.0                 # neutral-ish: free revs
        elif self.gear == 1 and speed < 14.0:
            target = max(geared, self.idle + pedal * 2200.0)    # converter slip
        else:
            target = max(self.idle, geared)
        target = min(target, self.redline)

        tau = 0.08 if target > self.rpm else 0.16
        if self.cut > 0:
            tau = 0.07                                          # rpm drops fast into the gear
        self.rpm += (target - self.rpm) * (1.0 - math.exp(-dt / tau))

        self.cut = max(0.0, self.cut - dt)
        load_target = 0.0 if self.cut > 0 else pedal
        self.load += (load_target - self.load) * (1.0 - math.exp(-dt / 0.05))
        return self.rpm, self.gear, self.load


class _Engine(Voice):
    kind = "ice"
    IDLE, REDLINE = 800.0, 6500.0
    CYL = 4

    def __init__(self):
        super().__init__()
        self.box = VirtualGearbox(self.IDLE, self.REDLINE)

    def firing_hz(self, rpm):
        return rpm / 60.0 * self.CYL / 2.0          # 4-stroke: each cylinder fires every 2 revs

    def status(self, rpm, gear, speed):
        return f"gear {'N' if speed < 1 else gear}  {rpm:5.0f} rpm"


class _PulseEngine(_Engine):
    """Combustion engine built from individual cylinder firings, not a buzzing oscillator.

    Each cylinder is its own pressure pulse at its crank angle (sharp rise, fast decay,
    a few ms long), with its own cycle-to-cycle strength variation -- real engines never
    fire two cycles the same, and that roughness is most of what makes them sound real.
    The pulse train then goes through an exhaust: a low-pass "body" that opens with load,
    a resonant muffler chamber, and a pipe resonance (a short comb delay). On top: intake
    roar under load, faint valvetrain ticking, and crackle on the overrun (off the pedal at
    high rpm). Firing angles, per-cylinder strengths and exhaust sizes set the character.
    """
    FIRING_DEG: tuple = ()          # crank angle of each firing within the 720 deg cycle
    CYL_AMPS: tuple = ()            # per-cylinder strength (bank/header differences)
    JITTER = 0.1                    # cycle-to-cycle strength variation (+-)
    PULSE_SEC = 0.003               # combustion pressure pulse length
    MUFFLER_HZ, MUFFLER_Q = 110.0, 0.6
    PIPE_SEC, PIPE_FEEDBACK = 0.005, 0.45
    INTAKE_HZ = 600.0
    BODY_HZ = 350.0                 # exhaust low-pass at idle, no load
    LEVEL = 0.6

    def __init__(self):
        super().__init__()
        self._prev_rpm = None

    def build(self):
        t = {"cycle": _target(self.IDLE / 120.0), "body": _target(self.BODY_HZ),
             "drive": _target(1.2), "intake": _target(0.0), "mech": _target(0.0),
             "pop_p": _target(0.0), "pop_lvl": _target(0.0), "gain": _target(0.0)}
        cycle = Smooth(t["cycle"], 0.999)              # 4-stroke cycle rate = crank / 2
        width = cycle * self.PULSE_SEC                 # pulse length as a cycle fraction
        clock = Impulse(cycle)
        pulses = None
        for deg, amp in zip(self.FIRING_DEG, self.CYL_AMPS):
            saw = SawOscillator(cycle, phase_offset=(deg / 720.0) % 1.0)
            ramp = Clip((saw * -1.0 - (width * -2.0 + 1.0)) / (width * 2.0), 0.0, 1.0)
            strength = RandomUniform(1.0 - self.JITTER, 1.0 + self.JITTER, clock=clock)
            p = ramp * ramp * (amp * strength)         # sharp rise, quick decay
            pulses = p if pulses is None else pulses + p
        pulses = SVFilter(pulses, "high_pass", cutoff=25.0, resonance=0.0)   # no DC

        body = SVFilter(pulses, "low_pass", cutoff=Smooth(t["body"], 0.999), resonance=0.2)
        muffler = SVFilter(pulses, "band_pass", cutoff=self.MUFFLER_HZ,
                           resonance=self.MUFFLER_Q)
        pipe = CombDelay(body, self.PIPE_SEC, self.PIPE_FEEDBACK)
        exhaust = body * 0.6 + muffler * 0.7 + pipe * 0.35

        envelope = SVFilter(Abs(pulses), "low_pass", cutoff=40.0, resonance=0.0)
        intake = SVFilter(WhiteNoise(), "band_pass", cutoff=self.INTAKE_HZ, resonance=0.5) \
            * envelope * Smooth(t["intake"], 0.999)
        mech = SVFilter(WhiteNoise(), "high_pass", cutoff=3500.0, resonance=0.0) \
            * envelope * Smooth(t["mech"], 0.999)

        # overrun crackle: random pops, more likely the higher the rpm off the pedal
        chance = RandomUniform(0.0, 1.0, clock=Impulse(22.0))
        pop_gate = Clip((chance - (Smooth(t["pop_p"], 0.99) * -1.0 + 1.0)) * 1000.0, 0.0, 1.0)
        pops = SVFilter(WhiteNoise(), "band_pass", cutoff=900.0, resonance=0.35) \
            * ASREnvelope(0.001, 0.004, 0.05, clock=pop_gate) * Smooth(t["pop_lvl"], 0.99)

        drive = Smooth(t["drive"], 0.999)
        out = Tanh(exhaust * drive + intake + mech + pops) * 0.8 * Smooth(t["gain"], 0.999)
        return out, t

    def update(self, st):
        rpm, gear, load = self.box.step(st.speed_kmh, st.pedal, st.dt)
        rr = (rpm - self.IDLE) / (self.REDLINE - self.IDLE)
        self.targets["cycle"].set_value(rpm / 120.0)
        self.targets["body"].set_value(self.BODY_HZ + self.firing_hz(rpm) * 2.0
                                       + 1800.0 * load + 500.0 * st.swell)
        self.targets["drive"].set_value(1.1 + 2.4 * load)
        self.targets["intake"].set_value(0.4 * load * (0.3 + 0.7 * rr))
        self.targets["mech"].set_value(0.04 + 0.08 * rr)
        falling = self._prev_rpm is not None and rpm < self._prev_rpm - 1.0
        self._prev_rpm = rpm
        overrun = st.pedal < 0.05 and falling and rpm > 2400
        self.targets["pop_p"].set_value(min(0.3, 0.06 + (rpm - 2400) / 12000) if overrun else 0.0)
        self.targets["pop_lvl"].set_value(0.5 if overrun else 0.0)
        g = 0.0 if not st.active else (0.45 + 0.55 * load) * (0.6 + 0.4 * rr) * self.LEVEL
        self.targets["gain"].set_value(g)
        return self.status(rpm, gear, st.speed_kmh)


class V8Muscle(_PulseEngine):
    """Cross-plane V8: eight firings per cycle at 90 deg, but the two banks' exhausts
    merge unevenly (firing order 1-8-4-3-6-5-7-2), so successive pulses differ in
    strength -- the half-order 'burble' -- through a big, low muffler."""
    name = "V8 Muscle"
    IDLE, REDLINE, CYL = 750.0, 6200.0, 8
    FIRING_DEG = tuple(i * 90.0 for i in range(8))
    CYL_AMPS = (1.0, 0.72, 0.95, 0.84, 1.0, 0.68, 0.9, 0.8)
    JITTER = 0.12
    PULSE_SEC = 0.0035
    MUFFLER_HZ, MUFFLER_Q = 105.0, 0.65
    PIPE_SEC, PIPE_FEEDBACK = 0.0055, 0.45
    INTAKE_HZ = 520.0
    BODY_HZ = 320.0
    LEVEL = 0.7


class Inline4Sport(_PulseEngine):
    """Revvy inline-4: four even firings at 180 deg (no burble), sharper pulses, a short
    raspy pipe and a lot more intake roar; higher idle and redline, shorter gearing."""
    name = "Inline-4 Sport"
    IDLE, REDLINE, CYL = 850.0, 7400.0, 4
    FIRING_DEG = (0.0, 180.0, 360.0, 540.0)
    CYL_AMPS = (1.0, 0.95, 1.0, 0.93)
    JITTER = 0.07
    PULSE_SEC = 0.0025
    MUFFLER_HZ, MUFFLER_Q = 190.0, 0.55
    PIPE_SEC, PIPE_FEEDBACK = 0.003, 0.5
    INTAKE_HZ = 900.0
    BODY_HZ = 450.0
    LEVEL = 0.8

    def __init__(self):
        super().__init__()
        self.box.RPM_PER_KMH = (135.0, 85.0, 60.0, 46.0, 37.0, 31.0)   # shorter gearing


SYNTH_VOICES = (DeepDrone, HoverTurbine, CinematicChord, V8Muscle, Inline4Sport)
