"""Drive-event sound effects layered over any voice, and the logic that detects the events.

Effects (Effects):
  launch     flooring it from low speed: air whoosh sweeping up, a rising tone, a sub boom
  regen      the motor working as a generator: a warm low hum that follows road speed
             (so it falls as regen slows the car) plus a soft low breath, both scaled by
             regen current
  reverse    the drive voice ducks; a calm pulsing tone with a gentle sonar ping plays
  shift      virtual gears: a short low thump on upshift (lighter on downshift)
  power-on   a soft rising bell arpeggio over a swell when the car first answers

Events (DriveEvents) come from the car when it can tell us (battery current for regen,
signed motor rpm for reverse) and from pedal/speed otherwise, so everything also works with
the sliders at a desk.
"""
from __future__ import annotations


from signalflow import (ASREnvelope, Clip, CombDelay, Constant, Impulse, Line, PinkNoise, Pow,
                        RandomUniform, SawOscillator, SineLFO,
                        SineOscillator, Smooth, SVFilter, Tanh, TriangleOscillator, WhiteNoise)

# --- detection settings -------------------------------------------------------------------
# Sign conventions on the battery-module block: confirm with a short drive (the window title
# shows both live). Usual Hyundai convention: current < 0 while regenerating.
REGEN_CURRENT_SIGN = -1.0      # multiply current by this: positive = regenerating
REGEN_FULL_A = 80.0            # regen current that gives the full regen sound
REVERSE_RPM_SIGN = -1.0        # multiply motor rpm by this: positive = reversing
REVERSE_MIN_RPM = 60.0
LAUNCH_PEDAL_RATE = 2.0        # pedal travel per second (0..1 per s) that counts as a stab
LAUNCH_MIN_PEDAL = 0.65
LAUNCH_MAX_KMH = 50.0
LAUNCH_COOLDOWN_SEC = 4.0
POWER_ON_AFTER_STALE_SEC = 5.0
# Virtual gears: upshift speeds (km/h). Pitch climbs GEAR_PITCH_DEPTH through each gear and
# drops back on the upshift, with a brief torque-cut duck and a thump.
GEAR_SHIFT_KMH = (18.0, 38.0, 60.0, 85.0, 115.0)
GEAR_HYSTERESIS_KMH = 2.5
GEAR_PITCH_DEPTH = 0.08        # 8 % (~1.3 semitones) climb per gear
SHIFT_DUCK, SHIFT_DUCK_SEC = 0.55, 0.13
REVERSE_DUCK = 0.2
# Cruising: speed has stayed within CRUISE_BAND_KMH for CRUISE_WINDOW_SEC -> the voice
# eases down to CRUISE_GAIN, and comes straight back when speed moves or on a quick press.
CRUISE_BAND_KMH = 10.0
CRUISE_WINDOW_SEC = 4.0
CRUISE_GAIN = 0.5              # -6 dB
CRUISE_FADE_DOWN_SEC, CRUISE_FADE_UP_SEC = 1.5, 0.3


class DriveEvents:
    """Turns pedal/speed/car readings into effect levels and one-shot triggers."""

    def __init__(self):
        self.regen = 0.0
        self.reverse = False
        self.gear = 1
        self.pitch_mul = 1.0
        self.duck = 1.0
        self._prev_pedal = None
        self._pedal_rate = 0.0
        self._cooldown = 0.0
        self._duck_t = 0.0
        self._stale_for = 1e9          # so the first fresh data counts as a power-on
        self.cruising = False
        self.cruise_gain = 1.0
        self._speeds = []              # (time, speed) over the cruise window
        self._t = 0.0

    def step(self, dt, speed, pedal, fresh, *, current=None, signed_rpm=None, data_age=1e9,
             reverse_test=False, virtual_gears=False, regen_fade=0.35):
        """Advance one tick; returns a list of triggers: 'launch', 'up', 'down', 'power_on'."""
        triggers = []

        # power-on: first fresh data, or fresh again after a long silence
        if fresh:
            if self._stale_for >= POWER_ON_AFTER_STALE_SEC:
                triggers.append("power_on")
            self._stale_for = 0.0
        else:
            self._stale_for += dt

        live = data_age < 1.5
        # regen: battery current when we have it, else "off the pedal while moving"
        if current is not None and live:
            target = min(1.0, max(0.0, REGEN_CURRENT_SIGN * current / REGEN_FULL_A))
        else:
            off = max(0.0, 1.0 - pedal / regen_fade) if regen_fade > 0 else 0.0
            target = off * min(1.0, max(0.0, (speed - 3.0) / 25.0))
        self.regen += (target - self.regen) * min(1.0, dt / 0.3)

        # reverse: negative motor rpm (car) or the test switch (desk)
        if signed_rpm is not None and live:
            rev = REVERSE_RPM_SIGN * signed_rpm
            if rev > REVERSE_MIN_RPM:
                self.reverse = True
            elif rev < REVERSE_MIN_RPM / 2:
                self.reverse = False
        else:
            self.reverse = bool(reverse_test)

        # launch: a fast, deep pedal stab from low speed
        if self._prev_pedal is not None and dt > 0:
            rate = (pedal - self._prev_pedal) / dt
            self._pedal_rate += (rate - self._pedal_rate) * min(1.0, dt / 0.08)
        self._prev_pedal = pedal
        self._cooldown = max(0.0, self._cooldown - dt)
        if (self._pedal_rate > LAUNCH_PEDAL_RATE and pedal > LAUNCH_MIN_PEDAL
                and speed < LAUNCH_MAX_KMH and self._cooldown == 0.0 and not self.reverse):
            triggers.append("launch")
            self._cooldown = LAUNCH_COOLDOWN_SEC

        # virtual gears
        if virtual_gears and not self.reverse:
            up_at = GEAR_SHIFT_KMH[self.gear - 1] if self.gear <= len(GEAR_SHIFT_KMH) else 1e9
            down_at = (GEAR_SHIFT_KMH[self.gear - 2] - GEAR_HYSTERESIS_KMH
                       if self.gear > 1 else -1.0)
            if speed > up_at:
                self.gear += 1
                triggers.append("up")
                self._duck_t = SHIFT_DUCK_SEC
            elif speed < down_at:
                self.gear -= 1
                triggers.append("down")
            lo = GEAR_SHIFT_KMH[self.gear - 2] if self.gear > 1 else 0.0
            hi = GEAR_SHIFT_KMH[self.gear - 1] if self.gear <= len(GEAR_SHIFT_KMH) else lo + 40
            frac = min(1.0, max(0.0, (speed - lo) / (hi - lo)))
            self.pitch_mul = 1.0 - GEAR_PITCH_DEPTH + GEAR_PITCH_DEPTH * frac
        else:
            self.gear, self.pitch_mul = 1, 1.0

        # cruising: speed held within the band for the whole window
        self._t += dt
        self._speeds.append((self._t, speed))
        while self._speeds and self._t - self._speeds[0][0] > CRUISE_WINDOW_SEC:
            self._speeds.pop(0)
        window_full = self._t - self._speeds[0][0] >= CRUISE_WINDOW_SEC - 0.05
        spread = max(v for _, v in self._speeds) - min(v for _, v in self._speeds)
        tip_in = self._pedal_rate > LAUNCH_PEDAL_RATE * 0.5
        if tip_in:
            self._speeds = [(self._t, speed)]          # restart the window
        self.cruising = window_full and spread < CRUISE_BAND_KMH and speed > 1.0 and not tip_in
        target = CRUISE_GAIN if self.cruising else 1.0
        tau = CRUISE_FADE_DOWN_SEC if target < self.cruise_gain else CRUISE_FADE_UP_SEC
        self.cruise_gain += (target - self.cruise_gain) * min(1.0, dt / tau)

        self._duck_t = max(0.0, self._duck_t - dt)
        self.duck = (REVERSE_DUCK if self.reverse else 1.0) * \
            (SHIFT_DUCK if self._duck_t > 0 else 1.0)
        return triggers


class _OneShot:
    """An envelope that stays silent until first triggered (signalflow envelopes start on
    creation), plus any sweeps that restart with it."""

    def __init__(self, env, *lines):
        self.env, self.lines = env, lines
        self.armed = Constant(0.0)
        self.node = env * self.armed

    def fire(self):
        self.armed.set_value(1.0)
        self.env.trigger()
        for line in self.lines:
            line.trigger()


class _Section:
    """One effect branch, connected to the output only until its hold time runs out."""

    def __init__(self, node):
        self.node, self.on, self.until = node, False, 0.0

    def hold(self, now, sec):
        self.until = max(self.until, now + sec)
        if not self.on:
            self.node.play()
            self.on = True

    def expire(self, now):
        if self.on and now >= self.until:
            self.node.stop()
            self.on = False


class Effects:
    LEVEL = 0.8
    REGEN_HZ_STOPPED, REGEN_HZ_TOP = 80.0, 160.0   # generator hum follows speed
    REGEN_TONE_CUTOFF = 650.0                      # keeps it warm, never whistly
    CHIME = (220.0, 329.63, 440.0, 554.37, 659.26)       # A3 E4 A4 C#5 E5
    CHIME_STEP_SEC = 0.13

    def __init__(self):
        self.targets = {"regen": Constant(0.0), "reverse": Constant(0.0),
                        "regen_hz": Constant(self.REGEN_HZ_STOPPED),
                        "master": Constant(1.0), "shift_amp": Constant(1.0)}
        self._clock = 0.0                  # audio-side time, advanced by update(dt)
        self._pending = []                 # (due time, callable) for staggered chime notes
        # Each effect is its own branch that only plays while it can be heard: a branch
        # that isn't playing costs no CPU. Running them all silently cost ~3x a voice,
        # which a Pi 3B+ can't afford.
        self._sections = {name: _Section(node) for name, node in self._build().items()}

    def _build(self):
        t = self.targets
        # launch: impact + an FM spool-up that pulses faster and faster -------------------
        # Fun comes from motion (accelerating pulses fusing into a growl, a timbre that gets
        # richer as it charges); "not cartoonish" comes from avoiding a single pure tone
        # sliding a long way -- the pitch only rises an octave and it is an FM timbre.
        thud = Line(58.0, 38.0, 0.35)
        self.impact_env = _OneShot(ASREnvelope(0.003, 0.05, 0.6, curve=2.0), thud)
        self.body_env = _OneShot(ASREnvelope(0.002, 0.03, 0.25, curve=2.0))
        impact = SineOscillator(thud) * self.impact_env.node * 0.9 + \
            SVFilter(PinkNoise(), "low_pass", cutoff=220.0, resonance=0.0) * self.body_env.node * 1.4

        carrier = Line(98.0, 196.0, 1.3)                  # G2 -> G3: one octave, no further
        index = Line(0.6, 3.2, 1.3)                       # FM depth: richer as it charges
        pulse_rate = Line(6.0, 26.0, 1.2)                 # spins up, then fuses into a growl
        pulse_depth = Line(0.85, 0.15, 1.2)
        # slower attack than the impact: the spool builds up out of the hit
        self.spool_env = _OneShot(ASREnvelope(0.25, 0.8, 0.8, curve=1.5),
                                  carrier, index, pulse_rate, pulse_depth)
        mod = SineOscillator(carrier * 1.5) * (carrier * 1.5 * index)
        fm = SineOscillator(carrier + mod) + SineOscillator(carrier * 0.5 + mod * 0.5) * 0.6
        spin = Pow(SineLFO(pulse_rate, 0.0, 1.0), 1.5) * pulse_depth + (pulse_depth * -1.0 + 1.0)
        spool = SVFilter(fm, "low_pass", cutoff=2600.0, resonance=0.0) * spin * \
            self.spool_env.node * 0.6

        # electric crackle: sparse arcs in a mid band, a little denser as it charges
        arc_rate = Line(14.0, 40.0, 1.2)
        self.arc_env = _OneShot(ASREnvelope(0.05, 0.6, 0.5), arc_rate)
        arc_gate = Clip((RandomUniform(0.0, 1.0, clock=Impulse(arc_rate)) - 0.7) * 1000.0, 0.0, 1.0)
        arcs = SVFilter(WhiteNoise(), "band_pass", cutoff=1700.0, resonance=0.45) * \
            ASREnvelope(0.0005, 0.004, 0.03, clock=arc_gate) * self.arc_env.node * 0.35

        air_open = Line(300.0, 1500.0, 0.6)
        self.air_env = _OneShot(ASREnvelope(0.08, 0.15, 1.1, curve=1.5), air_open)
        air = SVFilter(PinkNoise(), "low_pass", cutoff=air_open, resonance=0.0) * \
            self.air_env.node * 0.45
        launch = impact + spool + arcs + air

        # regen: generator hum + low breath ------------------------------------------------
        # Triangle + a gently saturated copy give overtones up to ~500 Hz, so a small
        # speaker still carries it, while the low-pass keeps it from turning whistly.
        hz = Smooth(t["regen_hz"], 0.999)
        wobble = SineLFO(0.7, 0.985, 1.015)                 # slow, slight drift
        # saw gives every overtone below the cutoff, so a small speaker that drops the
        # fundamental still conveys the low pitch; the triangle adds body underneath
        hum = SawOscillator(hz * wobble) * 0.55 + TriangleOscillator(hz * wobble) * 0.5
        hum = SVFilter(hum, "low_pass", cutoff=self.REGEN_TONE_CUTOFF, resonance=0.15)
        breath = SVFilter(PinkNoise(), "band_pass", cutoff=320.0, resonance=0.3) * \
            SineLFO(0.35, 0.6, 1.0)
        regen = (hum * 0.45 + breath * 0.5) * Smooth(t["regen"], 0.9995)

        # reverse ------------------------------------------------------------------------
        pad = (TriangleOscillator(220.0) + TriangleOscillator(330.0) * 0.6) * \
            SineLFO(1.6, 0.55, 1.0) * 0.18
        pad = SVFilter(pad, "low_pass", cutoff=1400.0, resonance=0.0)
        ping = SineOscillator(1250.0) * ASREnvelope(0.002, 0.0, 0.45, clock=Impulse(0.9)) * 0.2
        ping = ping + CombDelay(ping, 0.19, 0.35) * 0.5
        reverse = (pad + ping) * Smooth(t["reverse"], 0.999)

        # virtual-gear thump ----------------------------------------------------------------
        thump_f = Line(85.0, 42.0, 0.12)
        self.thump_env = _OneShot(ASREnvelope(0.002, 0.02, 0.22), thump_f)
        self.click_env = _OneShot(ASREnvelope(0.001, 0.004, 0.04))
        thump = (SineOscillator(thump_f) * self.thump_env.node * 0.7
                 + SVFilter(WhiteNoise(), "low_pass", cutoff=900.0, resonance=0.0)
                 * self.click_env.node * 0.15) * t["shift_amp"]

        # power-on chime --------------------------------------------------------------------
        self.chime_notes = []
        chime = None
        for f in self.CHIME:
            env = _OneShot(ASREnvelope(0.004, 0.05, 1.8, curve=2.0))
            bell = (SineOscillator(f) + SineOscillator(f * 2.76) * 0.18 +
                    SineOscillator(f * 5.4) * 0.05) * env.node * 0.32
            self.chime_notes.append(env)
            chime = bell if chime is None else chime + bell
        self.swell_env = _OneShot(ASREnvelope(0.5, 0.4, 1.6))
        swell = (TriangleOscillator(110.0) + TriangleOscillator(164.81) * 0.7) * \
            self.swell_env.node * 0.22
        chime = chime + swell + CombDelay(chime, 0.23, 0.3) * 0.4

        master = Smooth(t["master"], 0.999) * self.LEVEL
        return {name: Tanh(node * master) for name, node in
                (("launch", launch), ("regen", regen), ("reverse", reverse),
                 ("shift", thump), ("chime", chime))}

    # -- control --------------------------------------------------------------------------
    def update(self, dt, regen=0.0, reverse=False, enabled=True, speed_ratio=0.0):
        self._clock += dt
        now = self._clock
        for due, fn in [p for p in self._pending if p[0] <= now]:
            fn()
        self._pending = [p for p in self._pending if p[0] > now]
        self.targets["master"].set_value(1.0 if enabled else 0.0)
        self.targets["regen"].set_value(regen if enabled else 0.0)
        self.targets["reverse"].set_value(1.0 if (reverse and enabled) else 0.0)
        self.targets["regen_hz"].set_value(
            self.REGEN_HZ_STOPPED + (self.REGEN_HZ_TOP - self.REGEN_HZ_STOPPED)
            * min(1.0, max(0.0, speed_ratio)))
        # continuous effects run while asked for, plus a moment to fade out
        sec = self._sections
        if enabled and regen > 0.001:
            sec["regen"].hold(now, 0.5)
        if enabled and reverse:
            sec["reverse"].hold(now, 0.5)
        for s in sec.values():
            s.expire(now)

    @property
    def playing(self):
        """Names of the effect branches currently running (for CPU checks)."""
        return [n for n, s in self._sections.items() if s.on]

    def launch(self):
        self._sections["launch"].hold(self._clock, 2.2)
        for shot in (self.impact_env, self.body_env, self.spool_env, self.arc_env,
                     self.air_env):
            shot.fire()

    def shift(self, up=True):
        self.targets["shift_amp"].set_value(1.0 if up else 0.35)
        self._sections["shift"].hold(self._clock, 0.4)
        self.thump_env.fire()
        self.click_env.fire()

    def power_on(self):
        now = self._clock
        self._sections["chime"].hold(now, 4.5)
        self.swell_env.fire()
        for i, note in enumerate(self.chime_notes):
            self._pending.append((now + 0.25 + i * self.CHIME_STEP_SEC, note.fire))
