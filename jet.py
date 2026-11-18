"""Supersonic fighter-jet voice, physically modelled like geared.py: a turbofan's parts,
driven by a simulated spool that steps with virtual gear shifts, plus what changes when
the "aircraft" goes supersonic.

  spool       fan speed N1 follows the virtual drivetrain's rpm (drivetrain.py): it
              climbs through each gear and steps down at every upshift, like the
              N e-shift; quick to spool up, slow to wind down after lifting off.
  core        the body of the sound: a turbofan has two spools -- low pressure (N1) and
              high pressure (N2, CORE_N2_RATIO faster, with its own lag) -- and their
              rotation orders make a dense, harmonically rich tone (CORE_IDLE_HZ ->
              CORE_TOP_HZ). Pitched so its weight lands at 150-400 Hz, where the car's
              small speaker is strongest (its fundamental is implied by the harmonics).
  fan tones   quiet colour on top: blade tones rising with N1 (TONE_IDLE_HZ -> TONE_TOP_HZ), breathy: each
              partial carries slow noise, spreading it into a narrow band like the broad
              ridges in Hyundai's Supersonic recording.
  roar        jet mixing noise; its pitch centre rises with exhaust speed and its level
              climbs steeply with thrust.
  shock crackle  at high thrust the exhaust noise steepens into shock waves: one-sided
              spikes, the tearing crackle of a fighter at full power.
  buzzsaw     fan blade tips going supersonic: a gritty rasp at the shaft orders.
  afterburner lights at the top of the pedal: thump, deep roar and pops.
  Mach        road speed / MACH1_KMH is the "Mach number". Crossing Mach 1 sends a low
              double thump (the N-wave of a sonic boom); above it the exhaust roar falls
              behind and rushing airflow over the airframe takes over.
  cabin       heard from inside the car, not from the runway: the body passes low
              frequencies and blocks high ones (transmission loss climbs with frequency),
              so the roar arrives as weight rather than hiss; tones and structure-borne
              vibration come through; the cabin adds its own low resonances.
"""
from __future__ import annotations

import dataclasses

from signalflow import (Abs, ASREnvelope, BiquadFilter, Impulse, RandomImpulse, RandomUniform,
                        SawOscillator, SineOscillator, Smooth, Tanh, WhiteNoise)

import drivetrain
from voices import State, Voice, _target

Q = 0.7071                    # signalflow: low/high/band-pass resonance is the Q itself


def _smoothstep(x: float, lo: float, hi: float) -> float:
    u = min(1.0, max(0.0, (x - lo) / (hi - lo)))
    return u * u * (3.0 - 2.0 * u)


class SupersonicJet(Voice):
    name = "Supersonic Jet"
    own_gears = True          # runs its own gearbox: main.py leaves its shifts alone
    LEVEL = 1.0
    # --- spool ----------------------------------------------------------------------
    IDLE_N1 = 0.3
    SPOOL_UP_SEC = 0.2        # follows the climb through a gear
    SHIFT_SPOOL_SEC = 0.1     # how fast it steps down at an upshift
    SPOOL_DOWN_SEC = 1.5      # winding down after lifting off
    # --- core (two spools) -------------------------------------------------------------
    CORE = 1.5
    CORE_IDLE_HZ, CORE_TOP_HZ = 70.0, 140.0
    CORE_N2_RATIO = 1.37      # high-pressure spool order relative to the low one
    CORE_N2_LAG_SEC = 0.35    # N2 trails N1
    CORE_BODY_HZ = 260.0      # resonance the core rings through
    CORE_BUZZ = 0.0           # 0 = smooth spool hum (sine-like), 1 = buzzy (saw: reads as
                              # a piston engine)
    ROAR_FLOOR = 0.35         # roar already present at idle (a jet is never silent)
    # --- fan tones -------------------------------------------------------------------
    TONE_IDLE_HZ = 165.0
    TONE_TOP_HZ = 380.0
    TONES = 0.15
    TONE_WIDTH_HZ = 30.0      # how breathy: each partial spread over ~2x this
    # --- roar / crackle ---------------------------------------------------------------
    ROAR = 1.0
    ROAR_LO_HZ, ROAR_HI_HZ = 140.0, 650.0
    CRACKLE = 0.7             # shock crackle at high thrust
    CRACKLE_FROM = 0.55       # thrust where it starts
    # --- buzzsaw ---------------------------------------------------------------------
    BUZZ = 0.5
    BUZZ_FROM = 0.7
    BLADES = 7
    # --- afterburner -----------------------------------------------------------------
    AB_FROM = 0.85
    AB_ROAR = 0.8
    AB_POPS = 0.5
    AB_POP_RATE = 30.0
    # --- Mach ------------------------------------------------------------------------
    MACH1_KMH = 90.0
    BOOM = 0.6                # the double thump crossing Mach 1 (0 = off)
    AIRFLOW = 0.3             # airframe buffeting above Mach 1 (low rumble, not hiss)
    # --- shifts ----------------------------------------------------------------------
    SHIFT_DIP = 0.12
    STAGE_THUMP = 0.3         # small thump on a hard upshift
    CABIN_HZ = 2200.0         # what gets through the body: roll-off above this
    CABIN_BOOM_HZ, CABIN_BOOM_DB = 85.0, 4.0     # the cabin's low resonance
    # drivetrain constants (GEARS, GEAR_STEP, ...) are added below the class

    def __init__(self):
        self.dt = drivetrain.Drivetrain()
        self.n1 = self.IDLE_N1
        self.n2 = self.IDLE_N1
        self.ab = 0.0
        self._ab_was = False
        self._sup = 0.0
        self._boom_armed = True
        self._thumps: list[float] = []     # per-tick thump levels still to play
        super().__init__()

    def build(self):
        t = {k: _target(v) for k, v in {
            "tone": self.TONE_IDLE_HZ, "tones": 0.0, "roar": 0.0, "roar_hz": self.ROAR_LO_HZ,
            "crackle": 0.0, "buzz": 0.0, "ab": 0.0, "pop_rate": 0.0, "pops": 0.0,
            "air": 0.0, "air_hz": 500.0, "thump": 0.0, "gain": 0.0,
            "core_f1": self.CORE_IDLE_HZ, "core_f2": self.CORE_IDLE_HZ * self.CORE_N2_RATIO,
            "core": 0.0, "core_drive": 1.0, "core_lp": 600.0}.items()}
        f = Smooth(t["tone"], 0.9995)

        # core: two spools' rotation orders, saturated together (they intermodulate),
        # through the body resonance; brighter and harder with thrust
        f1, f2 = Smooth(t["core_f1"], 0.9995), Smooth(t["core_f2"], 0.9995)
        cdrv = Smooth(t["core_drive"], 0.999)
        b = self.CORE_BUZZ
        src1 = SawOscillator(f1) * b + SineOscillator(f1) * (1.0 - b) * 1.6
        src2 = SawOscillator(f2) * b + SineOscillator(f2) * (1.0 - b) * 1.6
        core = Tanh((src1 + src2 * 0.7 + SineOscillator(f1 * 0.5) * 0.4) * cdrv)
        core = BiquadFilter(core, "peak", cutoff=self.CORE_BODY_HZ, resonance=1.2 * 3.1623,
                            peak_gain=5.0)
        clp = Smooth(t["core_lp"], 0.999)
        for _ in range(2):
            core = BiquadFilter(core, "low_pass", cutoff=clp, resonance=Q)
        core = BiquadFilter(core, "high_pass", cutoff=60.0, resonance=Q) * Smooth(t["core"], 0.999)

        # fan tones: partials 1..3, each carrying its own slow noise (breathy bands)
        tones = None
        for k, a in ((1.0, 1.0), (2.0, 0.5), (3.0, 0.22)):
            wob = BiquadFilter(WhiteNoise(), "low_pass", cutoff=self.TONE_WIDTH_HZ, resonance=Q)
            p = SineOscillator(f * k) * (0.45 + wob * 25.0) * a
            tones = p if tones is None else tones + p
        tones = tones * Smooth(t["tones"], 0.999)

        # roar: broadband turbulence, band centre following exhaust speed
        rhz = Smooth(t["roar_hz"], 0.999)
        roar = BiquadFilter(WhiteNoise(), "band_pass", cutoff=rhz, resonance=0.6) \
            + BiquadFilter(WhiteNoise(), "band_pass", cutoff=rhz * 2.3, resonance=0.8) * 0.15
        roar = BiquadFilter(roar, "high_pass", cutoff=70.0, resonance=Q) * Smooth(t["roar"], 0.999)

        # shock crackle: one-sided (steepened) spikes from the same kind of turbulence
        nb = BiquadFilter(WhiteNoise(), "band_pass", cutoff=rhz * 1.5, resonance=0.7) * 4.0
        half = (Abs(nb) + nb) * 0.5
        crack = BiquadFilter(half * half * half, "high_pass", cutoff=250.0, resonance=Q)
        crack = Tanh(crack * 3.0) * Smooth(t["crackle"], 0.999)

        # buzzsaw: shaft-order rasp
        shaft = f / self.BLADES
        jitter = Smooth(RandomUniform(0.6, 1.4, clock=Impulse(shaft)), 0.98)
        buzz = Tanh(SawOscillator(shaft) * 3.0) * jitter
        buzz = BiquadFilter(buzz, "band_pass", cutoff=f * 1.6, resonance=0.9) \
            * Smooth(t["buzz"], 0.999)

        # afterburner: deep roar and pops
        ab = Smooth(t["ab"], 0.9995)
        deep = BiquadFilter(WhiteNoise(), "low_pass", cutoff=220.0, resonance=Q)
        deep = BiquadFilter(deep, "high_pass", cutoff=60.0, resonance=Q) * ab * self.AB_ROAR * 2.5
        env = ASREnvelope(0.0003, 0.0, 0.012, clock=RandomImpulse(Smooth(t["pop_rate"], 0.99)))
        pops = Tanh(BiquadFilter(WhiteNoise(), "high_pass", cutoff=400.0, resonance=Q)
                    * env * 6.0) * Smooth(t["pops"], 0.99) * 0.6

        # airflow over the airframe: broad rushing noise, rising with speed
        air = BiquadFilter(WhiteNoise(), "band_pass", cutoff=Smooth(t["air_hz"], 0.999),
                           resonance=0.5) * Smooth(t["air"], 0.999)

        # thumps (boom, burner light, hard upshift): low and short
        thump = (SineOscillator(70.0) + BiquadFilter(WhiteNoise(), "low_pass", cutoff=250.0,
                                                     resonance=Q)) * Smooth(t["thump"], 0.996)

        # outside sources through the car body (steep high cut: no hiss), then the cabin
        outside = roar + crack + deep + pops + air
        for _ in range(2):
            outside = BiquadFilter(outside, "low_pass", cutoff=self.CABIN_HZ, resonance=Q)
        inside = tones + buzz + core   # tonal, structure-borne: comes through, gently cut
        inside = BiquadFilter(inside, "low_pass", cutoff=self.CABIN_HZ * 2.0, resonance=Q)
        mix = outside + inside + thump * 0.8
        mix = BiquadFilter(mix, "peak", cutoff=self.CABIN_BOOM_HZ, resonance=1.5 * 3.1623,
                           peak_gain=self.CABIN_BOOM_DB)
        out = Tanh(mix * 0.6) * Smooth(t["gain"], 0.999)     # rounds only the loudest peaks
        return out, t

    def _thump(self, *pattern: float):
        """Queue thump levels, one per control tick (merged with any already queued)."""
        for i, v in enumerate(pattern):
            if i < len(self._thumps):
                self._thumps[i] = max(self._thumps[i], v)
            else:
                self._thumps.append(v)

    def update(self, st: State) -> str:
        for fld in dataclasses.fields(drivetrain.DrivetrainConfig):   # live-tunable
            setattr(self.dt.cfg, fld.name, getattr(self, fld.name))
        d = self.dt.step(st.speed_kmh, st.pedal, st.dt)
        on = max(0.0, d.load)
        if d.shifting > 0:
            on = max(on, max(0.0, self.dt._pedal_s) * (1.0 - 0.3 * d.shifting))

        # spool follows the gearbox: climbs through a gear, steps down on an upshift
        target = self.IDLE_N1 + (1.0 - self.IDLE_N1) * (0.85 * d.rpm_norm + 0.15 * on)
        if target > self.n1:
            tau = self.SPOOL_UP_SEC
        else:
            tau = self.SHIFT_SPOOL_SEC if (d.shifting > 0 or on > 0.1) else self.SPOOL_DOWN_SEC
        self.n1 += (target - self.n1) * min(1.0, st.dt / tau)
        n = (self.n1 - self.IDLE_N1) / (1.0 - self.IDLE_N1)
        thrust = max(0.0, n) ** 1.3 * (0.4 + 0.6 * on)
        tg = self.targets

        # Mach
        mach = st.speed_kmh / self.MACH1_KMH
        self._sup += (_smoothstep(mach, 0.98, 1.15) - self._sup) * min(1.0, st.dt / 0.4)
        if mach >= 1.0 and self._boom_armed:
            self._boom_armed = False
            b = self.BOOM
            self._thump(b, b, 0, 0, 0, b * 0.8, b * 0.8)   # N-wave: two bangs ~0.1 s apart
        elif mach < 0.9:
            self._boom_armed = True

        tg["tone"].set_value(self.TONE_IDLE_HZ + (self.TONE_TOP_HZ - self.TONE_IDLE_HZ) * n)
        self.n2 += (self.n1 - self.n2) * min(1.0, st.dt / self.CORE_N2_LAG_SEC)
        n2 = (self.n2 - self.IDLE_N1) / (1.0 - self.IDLE_N1)
        span = self.CORE_TOP_HZ - self.CORE_IDLE_HZ
        tg["core_f1"].set_value(self.CORE_IDLE_HZ + span * n)
        tg["core_f2"].set_value((self.CORE_IDLE_HZ + span * n2) * self.CORE_N2_RATIO)
        tg["core"].set_value(self.CORE * (0.5 + 0.5 * n) * (1.0 - self.SHIFT_DIP * d.shifting))
        tg["core_drive"].set_value(1.0 + 2.5 * on)
        tg["core_lp"].set_value(450.0 + 900.0 * on * max(0.0, n))
        tg["tones"].set_value(self.TONES * (0.35 + 0.65 * n) * (1.0 - self.SHIFT_DIP * d.shifting))
        roar_amt = self.ROAR_FLOOR + (1.0 - self.ROAR_FLOOR) * thrust ** 2
        tg["roar"].set_value(self.ROAR * roar_amt * 2.5 * (1.0 - 0.35 * self._sup))
        tg["roar_hz"].set_value(self.ROAR_LO_HZ + (self.ROAR_HI_HZ - self.ROAR_LO_HZ) * thrust)
        tg["crackle"].set_value(self.CRACKLE * _smoothstep(thrust, self.CRACKLE_FROM, 1.0))
        tg["buzz"].set_value(self.BUZZ * _smoothstep(self.n1, self.BUZZ_FROM, 1.0))
        speed2 = min(1.5, mach) ** 2
        tg["air"].set_value(self.AIRFLOW * speed2 * (0.25 + 0.75 * self._sup) * 1.5)
        tg["air_hz"].set_value(120.0 + 180.0 * min(1.5, mach))     # buffeting, not hiss

        # afterburner
        lit = st.pedal >= self.AB_FROM and self.n1 > 0.75
        self.ab += ((1.0 if lit else 0.0) - self.ab) * min(1.0, st.dt / (0.25 if lit else 0.6))
        if lit and not self._ab_was:
            self._thump(0.8, 0.8, 0.8)
        self._ab_was = lit
        tg["ab"].set_value(self.ab)
        tg["pop_rate"].set_value(self.AB_POP_RATE * self.ab if self.ab > 0.05 else 0.0)
        tg["pops"].set_value(self.AB_POPS * self.ab)

        if d.event == "up" and self.dt._pedal_s > 0.6:
            self._thump(self.STAGE_THUMP, self.STAGE_THUMP)
        tg["thump"].set_value(self._thumps.pop(0) if self._thumps else 0.0)

        tg["gain"].set_value(self.LEVEL if st.active else 0.0)
        tag = "  AFTERBURNER" if self.ab > 0.5 else ""
        return f"gear {d.gear}  N1 {self.n1 * 100:3.0f}%  Mach {mach:.2f}{tag}"


for _f in dataclasses.fields(drivetrain.DrivetrainConfig):     # tunable in the lab
    setattr(SupersonicJet, _f.name, _f.default)
