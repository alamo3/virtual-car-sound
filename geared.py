"""Physically modelled combustion-style voice on a virtual drivetrain (drivetrain.py).

Built from the parts that make a real engine sound like one, each driven by the virtual
RPM, load and gear the drivetrain produces:

  cylinders  each firing is a pressure pulse. A 4-cylinder fires in turn, and no two
             cylinders are identical: a per-cylinder strength pattern (CYL_LEVELS) plus
             cycle-to-cycle combustion variation. That repeating unevenness is the
             "burble" -- the sub-orders below the firing frequency.
  rasp       each pulse also carries a burst of turbulent noise (gas rushing past the
             valve), shaped by the same pulses: the grain between the harmonics.
  exhaust    after engine-sim (AngeTheGreat): each cylinder's pulse reaches the collector
             through its own header pipe (HEADER_MS: unequal lengths = uneven arrival,
             the boxer / unequal-header burble), then the whole exhaust system acts as
             one impulse response -- a measured one from engine-sim's library
             (sounds/ir/, EXHAUST_IR). Every pulse plays its own copy of that response,
             scaled by the pulse's strength: convolution, done as overlapping grains so
             it stays cheap. Turbulence (AIR_NOISE) modulates the flow, as in
             engine-sim. The older model -- the pulses and rasp ringing through two pipes
             of different length (comb resonators with fixed resonances in Hz) -- is
             blended in by 1 - IR_MIX. As rpm rises the firing harmonics sweep
             across those fixed resonances, so the timbre changes with revs instead of
             just transposing -- the main difference from an oscillator. A muffler
             (low-pass, opening with load) and soft clipping under load follow.
  intake     induction roar: air noise chopped by the firing pulses (the engine
             breathes in bursts), through the airbox resonance; mostly under load.
  cabin      a low body boom and the cabin's lift around PEAK1_HZ.
  overrun    lift off and unburnt fuel pops in the exhaust: random crackle, strongest
             in the first second or two after lifting and at higher revs.
  shift      torque cut (level and brightness dip), the rev swing to the new gear,
             a driveline clunk, and on hard upshifts a single exhaust pop.
The drivetrain's constants (gears, shift points, shift time...) are class constants too,
so the sound lab can tune them live.
"""
from __future__ import annotations

import dataclasses
import math
import wave
from pathlib import Path

import numpy as np

from signalflow import (ASREnvelope, BiquadFilter, Buffer, CombDelay, Counter, EnvelopeBuffer,
                        Granulator, Impulse, Index, OneTapDelay,
                        RandomImpulse, RandomUniform, SawOscillator, SineOscillator, Smooth,
                        Tanh, WhiteNoise)

import drivetrain
from voices import State, Voice, _target

SR = 48000
IR_DIR = Path(__file__).parent / "sounds" / "ir"
BUTTERWORTH_Q = 0.7071        # signalflow: low/high/band-pass resonance is the Q itself
PEAK_RES = math.sqrt(10.0)    # ... but a peaking filter's Q is resonance / sqrt(10)


_FLAT_ENV = [1.0] * 480 + list(np.linspace(1.0, 0.0, 32))   # grains play unfaded


def _exhaust_grain(name: str, pulse_ms: float) -> np.ndarray:
    """The exhaust system's response to one pressure pulse: the impulse response
    convolved with a smooth pulse, trimmed where it has died away (-60 dB)."""
    with wave.open(str(IR_DIR / f"{name}.wav")) as w:
        raw = w.readframes(w.getnframes())
        width = w.getsampwidth()
    ir = np.frombuffer(raw, "<f4" if width == 4 else "<i2").astype(np.float64)
    if width == 2:
        ir = ir / 32768.0
    n = max(3, int(pulse_ms * SR / 1000.0))
    g = np.convolve(ir, np.hanning(n) / np.hanning(n).sum())
    e = np.cumsum(g[::-1] ** 2)[::-1]
    g = g[:max(64, int(np.searchsorted(-e, -e[0] * 1e-6)))]
    g = g / (np.abs(g).max() + 1e-12)                     # unit peak: a pulse of 1 -> 1
    return g.astype(np.float32)


class GearedEngine(Voice):
    name = "Geared Engine"
    own_gears = True          # runs its own gearbox: main.py leaves its shifts alone
    LEVEL = 1.0
    # --- cylinders ----------------------------------------------------------------------
    F_REDLINE = 81.0          # firing frequency (Hz) at the redline
    CYL_LEVELS = (1.0, 0.6, 0.85, 0.5)     # per-cylinder pulse strength, in firing order
    COMBUSTION_VAR = 0.35     # random cycle-to-cycle variation of each pulse
    PULSE_HZ = 1400.0         # pulse shape: how sharp each pressure pulse is
    RASP = 4.0                # turbulent noise riding on each pulse (more under load)
    # --- exhaust ------------------------------------------------------------------------
    EXHAUST_IR = "mild_exhaust"   # sounds/ir/<name>.wav: mild_exhaust, smooth_39, sharp_01,
                                  # minimal_muffling_01, mild_exhaust_reverb
    IR_MIX = 0.85             # 1 = all impulse-response exhaust, 0 = the comb pipes
    IR_GAIN = 0.17
    HEADER_MS = (0.0, 0.7, 1.4, 0.35)     # extra header length per cylinder, as delay
    PULSE_MS = 2.0            # how long each exhaust pressure pulse lasts
    AIR_NOISE = 0.5           # turbulence riding on the exhaust flow (engine-sim's air noise)
    PIPE1_MS, PIPE2_MS = 4.6, 6.3          # two pipe lengths (resonances at k / delay)
    PIPE_FEEDBACK = 0.62      # how long the pipes ring
    MUFFLER_HZ = 1100.0       # low-pass at full load; lower cruising and on overrun
    DRIVE = 2.0               # extra soft clipping at full load
    # --- intake / cabin -----------------------------------------------------------------
    INTAKE = 0.35
    AIRBOX_HZ = 480.0
    BOOM_HZ, BOOM_DB = 58.0, 2.0
    PEAK1_HZ, PEAK1_DB = 265.0, 4.0
    # --- overrun / shift ----------------------------------------------------------------
    CRACKLE = 0.5             # overrun pops (0 = off)
    CRACKLE_RATE = 9.0        # pops per second at high rpm
    SHIFT_POP = 0.5           # one pop on a hard upshift
    CUT_DIP = 0.15            # level drop during the torque cut (a hint, not a gap)
    CLUNK = 0.25
    CLUNK_HZ = 48.0
    OVERRUN_LEVEL = 0.6       # engine level on overrun relative to light load
    # drivetrain constants (GEARS, GEAR_STEP, ...) are added below the class

    def __init__(self):
        self.dt = drivetrain.Drivetrain()
        self._clunk_ticks = 0
        self._clunk_level = 0.0
        self._pop = 0
        self._lift = 0.0          # 1 just after lifting off, decaying
        self._was_on = False
        super().__init__()

    def build(self):
        t = {k: _target(v) for k, v in {
            "f": self.F_REDLINE * self.IDLE_RPM / self.REDLINE_RPM, "drive": 1.0,
            "muffler": self.MUFFLER_HZ * 0.6, "engine": 0.0, "intake": 0.0,
            "intake_hz": 700.0, "crackle_rate": 0.0, "crackle": 0.0, "pop_trig": 0.0,
            "pop": 0.0, "clunk": 0.0, "rasp": self.RASP * 0.5, "gain": 0.0}.items()}
        f = Smooth(t["f"], 0.999)

        # cylinders: one pulse per firing, strength from the cylinder pattern, varied
        # randomly cycle to cycle
        fire = Impulse(f)
        cyl = Index(list(self.CYL_LEVELS), Counter(fire, 0, len(self.CYL_LEVELS)))
        var = 1.0 + self.COMBUSTION_VAR * RandomUniform(-1.0, 1.0, clock=fire)
        pulses = fire * cyl * var
        pulses = BiquadFilter(pulses, "low_pass", cutoff=self.PULSE_HZ, resonance=BUTTERWORTH_Q)
        pulses = BiquadFilter(pulses, "high_pass", cutoff=25.0, resonance=BUTTERWORTH_Q)

        # per-firing envelope: 1 at each firing, decaying until the next (from the firing
        # phase, so it is unit height at any rpm)
        phase = (SawOscillator(f) + 1.0) * 0.5
        decay = 1.0 - phase
        env = decay * decay * decay * decay
        # rasp: turbulent noise in each firing's envelope, as strong as that cylinder
        rasp = BiquadFilter(WhiteNoise(), "high_pass", cutoff=300.0, resonance=BUTTERWORTH_Q) \
            * env * cyl * var * Smooth(t["rasp"], 0.999) * 0.05
        src = pulses + rasp

        # exhaust (older model): two pipes ringing at their own fixed resonances
        pipe1 = CombDelay(src, self.PIPE1_MS / 1000.0, self.PIPE_FEEDBACK, 0.05)
        pipe2 = CombDelay(src, self.PIPE2_MS / 1000.0, self.PIPE_FEEDBACK * 0.9, 0.05)
        combs = (pipe1 + pipe2 * 0.8) * 6.0

        # exhaust (engine-sim model): per-cylinder header delays, then each pulse plays a
        # copy of the exhaust system's impulse response, scaled by its strength
        n_cyl = len(self.CYL_LEVELS)
        count = Counter(fire, 0, n_cyl)
        clock = None
        for i in range(n_cyl):
            gate = fire * Index([1.0 if j == i else 0.0 for j in range(n_cyl)], count) \
                * self.CYL_LEVELS[i] * var
            ms = self.HEADER_MS[i % len(self.HEADER_MS)]
            if ms > 0:                 # whole samples: an impulse must stay one sample
                gate = OneTapDelay(gate, round(ms * SR / 1000.0) / SR, 0.05)
            clock = gate if clock is None else clock + gate
        grain = _exhaust_grain(self.EXHAUST_IR, self.PULSE_MS)
        flow = Granulator(Buffer(np.array([grain])), clock=clock, pos=0.0,
                          duration=len(grain) / SR, use_trigger_amplitude=True)
        flow.set_buffer("envelope", EnvelopeBuffer(_FLAT_ENV))
        turb = BiquadFilter(WhiteNoise(), "low_pass", cutoff=2000.0, resonance=BUTTERWORTH_Q)
        flow = flow * (1.0 - self.AIR_NOISE + self.AIR_NOISE * turb * 4.7)
        ex = flow * (2.0 * self.IR_GAIN * self.IR_MIX) + combs * (1.0 - self.IR_MIX)
        drive = Smooth(t["drive"], 0.999)
        ex = Tanh(ex * drive)
        muffler = Smooth(t["muffler"], 0.999)
        for _ in range(2):
            ex = BiquadFilter(ex, "low_pass", cutoff=muffler, resonance=BUTTERWORTH_Q)
        ex = BiquadFilter(ex, "peak", cutoff=self.BOOM_HZ, resonance=1.2 * PEAK_RES,
                          peak_gain=self.BOOM_DB)
        ex = BiquadFilter(ex, "peak", cutoff=self.PEAK1_HZ, resonance=1.4 * PEAK_RES,
                          peak_gain=self.PEAK1_DB)
        ex = BiquadFilter(ex, "high_pass", cutoff=35.0, resonance=BUTTERWORTH_Q)
        engine = ex * Smooth(t["engine"], 0.999)

        # intake: noise chopped by the (smoothed) firing pulses, through the airbox
        air = WhiteNoise() * (0.3 + 1.5 * env * cyl)
        air = BiquadFilter(air, "band_pass", cutoff=self.AIRBOX_HZ, resonance=2.0) * 0.7 \
            + BiquadFilter(air, "band_pass", cutoff=Smooth(t["intake_hz"], 0.999),
                           resonance=1.5) * 0.5
        intake = air * Smooth(t["intake"], 0.999)

        # overrun crackle and the shift pop: short noise bursts through a pipe-ish band
        crackle_env = ASREnvelope(0.0005, 0.0, 0.035,
                                  clock=RandomImpulse(Smooth(t["crackle_rate"], 0.99)))
        pop_env = ASREnvelope(0.0005, 0.01, 0.07, clock=t["pop_trig"])
        burst = WhiteNoise() * (crackle_env * Smooth(t["crackle"], 0.99) + pop_env * t["pop"])
        burst = Tanh(BiquadFilter(burst, "band_pass", cutoff=320.0, resonance=1.3) * 3.0) * 0.5
        burst = CombDelay(burst, self.PIPE1_MS / 1000.0, 0.4, 0.05)

        thump = Smooth(t["clunk"], 0.997)
        clunk = (SineOscillator(self.CLUNK_HZ) * 0.8
                 + BiquadFilter(WhiteNoise(), "low_pass", cutoff=180.0,
                                resonance=BUTTERWORTH_Q) * 0.6) * thump

        # gentle soft clip: only the loudest full-throttle peaks are touched
        out = Tanh((engine + intake + burst + clunk) * 6.0) / 6.0 * Smooth(t["gain"], 0.999)
        return out, t

    def update(self, st: State) -> str:
        for fld in dataclasses.fields(drivetrain.DrivetrainConfig):   # live-tunable
            setattr(self.dt.cfg, fld.name, getattr(self, fld.name))
        d = self.dt.step(st.speed_kmh, st.pedal, st.dt)
        tg = self.targets
        tg["f"].set_value(self.F_REDLINE * d.rpm / self.REDLINE_RPM)

        on = max(0.0, d.load)
        if d.shifting > 0:        # torque is cut, but the sound only eases off a little
            on = max(on, max(0.0, self.dt._pedal_s) * (1.0 - 0.3 * d.shifting))
        overrun = max(0.0, -d.load)
        drive = 1.0 + self.DRIVE * on
        tg["drive"].set_value(drive)
        tg["muffler"].set_value(self.MUFFLER_HZ * (0.5 + 0.5 * on) * (1.0 - 0.3 * overrun)
                                * (1.0 - 0.15 * d.shifting))
        rpm_level = 0.6 + 0.4 * d.rpm_norm
        level = (0.7 + 0.3 * on) if d.load >= 0 else self.OVERRUN_LEVEL
        tg["engine"].set_value(rpm_level * level * (1.0 - self.CUT_DIP * d.shifting)
                               / math.sqrt(drive))      # clipping adds level: undo it
        tg["intake"].set_value(self.INTAKE * (0.15 + 0.85 * on) * (0.3 + 0.7 * d.rpm_norm))
        tg["intake_hz"].set_value(700.0 + 1300.0 * d.rpm_norm)

        tg["rasp"].set_value(self.RASP * (0.4 + 0.6 * on))
        # crackle: on overrun, strongest just after lifting off and at higher revs
        if d.load < 0 and self._was_on:
            self._lift = 1.0
        self._was_on = d.load > 0.15
        self._lift *= math.exp(-st.dt / 1.2)
        pops = self.CRACKLE * min(1.0, overrun * 2.0) * (0.15 + 0.85 * self._lift) \
            * (0.3 + 0.7 * d.rpm_norm)
        tg["crackle_rate"].set_value(self.CRACKLE_RATE * min(1.0, pops * 2.0) if pops > 0 else 0.0)
        tg["crackle"].set_value(pops)

        # shift: clunk scaled by the torque it interrupts; a pop on hard upshifts
        if d.event:
            self._clunk_ticks = 2
            self._clunk_level = self.CLUNK * (0.15 + 0.85 * max(0.0, self.dt._pedal_s))
            if d.event == "up" and self.dt._pedal_s > 0.7:
                self._pop = 2
        tg["clunk"].set_value(self._clunk_level if self._clunk_ticks > 0 else 0.0)
        self._clunk_ticks = max(0, self._clunk_ticks - 1)
        tg["pop"].set_value(self.SHIFT_POP)
        tg["pop_trig"].set_value(1.0 if self._pop == 2 else 0.0)   # rising edge fires it
        self._pop = max(0, self._pop - 1)

        tg["gain"].set_value(self.LEVEL if st.active else 0.0)
        tag = " LIMITER" if d.limiter else (" shift" if d.shifting > 0 else "")
        return f"gear {d.gear}  {d.rpm:4.0f} rpm  load {d.load * 100:+4.0f}%{tag}"


for _f in dataclasses.fields(drivetrain.DrivetrainConfig):     # tunable in the lab
    setattr(GearedEngine, _f.name, _f.default)
