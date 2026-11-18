"""A virtual combustion-style drivetrain: turns the car's real speed and pedal into the
engine state a sound designer would drive a game engine sound with.

The EV has one fixed reduction gear, so its motor speed is just road speed. This models
what a multi-gear car would be doing at the same road speed and pedal:

  gears       GEARS ratios, each GEAR_STEP of the one before (a 6-speed's spread; Hyundai's
              N e-shift sound drops by 0.78 per shift, which felt too busy). Gear 1
              reaches the redline at GEAR1_TOP_KMH.
  shifting    upshift rpm rises with pedal (short-shift when cruising, near the redline
              at full throttle); downshift when rpm sags, kickdown when the pedal is
              stabbed. Shifts take SHIFT_SEC (quicker at full throttle) during which
              torque is cut and the engine speed swings to the new gear's speed.
  clutch      at a standstill / walking pace the engine can't drop below idle: it slips,
              held at a launch speed that rises with pedal until the wheels catch up.
  inertia     engine speed follows its target with a short lag (a flywheel).
  load        -1 .. 1: positive = driving torque (follows pedal, cut during shifts),
              negative = overrun / engine braking (pedal off while rolling).
  limiter     at the redline the fuel is cut and restored ~12 times a second (LIMITER_HZ),
              the revs dipping a little each cut. Flat out (pedal above LIMITER_PEDAL) every
              gear runs into it and bounces for LIMITER_HOLD_SEC before the upshift, like a
              race driver holding a gear to the limiter; in top gear it just stays there.

Everything here is plain Python, ticked at the control rate (~50 Hz).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class DrivetrainConfig:
    GEARS: int = 6
    GEAR_STEP: float = 0.68           # next gear's rpm / this gear's rpm at the same speed
    GEAR1_TOP_KMH: float = 50.0       # gear 1 hits the redline here (2nd ~74, 3rd ~108 km/h)
    IDLE_RPM: float = 2200.0
    REDLINE_RPM: float = 7000.0
    LAUNCH_RPM: float = 3800.0        # clutch-slip engine speed at full pedal from rest
    # the first upshifts happen at these road speeds (km/h), whatever the pedal: 1->2, 2->3.
    # Higher gears shift on revs (below). Set an entry to 0 to use revs for that gear too.
    SHIFT_KMH: tuple = (30.0, 60.0)
    UPSHIFT_LIGHT: float = 0.50       # upshift point (fraction of redline) at light pedal
    UPSHIFT_FULL: float = 0.97        # ... and at full pedal
    DOWNSHIFT_MARGIN: float = 0.85    # downshift once rpm falls this far under the point
                                      # the last upshift would land on (hysteresis)
    KICKDOWN_STAB: float = 0.30       # pedal rise within ~0.25 s that asks for a kickdown
    SHIFT_SEC_FULL: float = 0.25      # shift time at full pedal
    SHIFT_SEC_LIGHT: float = 0.45     # ... and when cruising
    MIN_GEAR_SEC: float = 1.0         # no new shift sooner than this after the last one
    INERTIA_SEC: float = 0.07         # flywheel lag of the engine speed
    PEDAL_SMOOTH_SEC: float = 0.25    # how the gearbox reads the pedal (ignores jitter)
    OVERRUN_KMH: float = 6.0          # below this, lifting off doesn't count as overrun
    LIMITER_HZ: float = 12.0
    LIMITER_PEDAL: float = 0.85       # above this, each gear runs into the limiter
    LIMITER_HOLD_SEC: float = 0.35    # ... and bounces off it this long before upshifting
    LIMITER_DIP: float = 0.04         # revs drop this fraction during each fuel cut
    DOWNSHIFT_BLIP: float = 0.07      # rev-matching blip: revs overshoot this much (and the
                                      # throttle opens briefly) on a downshift off the pedal


@dataclass
class DriveState:
    rpm: float                # engine speed
    rpm_norm: float           # 0 at idle .. 1 at the redline
    gear: int
    load: float               # -1 (overrun) .. 1 (full torque)
    shifting: float           # 0 .. 1: how deep into a torque cut we are
    limiter: bool
    event: str | None = None  # "up" / "down" on the tick a shift starts
    cut: bool = False         # the limiter has the fuel cut on this tick


@dataclass
class Drivetrain:
    cfg: DrivetrainConfig = field(default_factory=DrivetrainConfig)
    gear: int = 1
    rpm: float = 0.0
    _pedal_s: float = 0.0
    _pedal_slow: float = 0.0
    _since_shift: float = 1e9
    _shift_t: float = 0.0       # time left in the current shift
    _shift_len: float = 0.0
    _limiter_t: float = 0.0
    _shift_up: bool = True

    def __post_init__(self):
        self.rpm = self.cfg.IDLE_RPM

    # --- gear geometry ---------------------------------------------------------------
    def top_kmh(self, gear: int) -> float:
        """Road speed at which `gear` reaches the redline."""
        return self.cfg.GEAR1_TOP_KMH / self.cfg.GEAR_STEP ** (gear - 1)

    def wheel_rpm(self, gear: int, kmh: float) -> float:
        """Engine speed the wheels impose in `gear` (clutch closed)."""
        return self.cfg.REDLINE_RPM * max(0.0, kmh) / self.top_kmh(gear)

    def upshift_rpm(self, pedal: float) -> float:
        c = self.cfg
        frac = c.UPSHIFT_LIGHT + (c.UPSHIFT_FULL - c.UPSHIFT_LIGHT) * min(1.0, pedal) ** 1.3
        # never upshift into a gear that would land below idle
        return max(frac * c.REDLINE_RPM, c.IDLE_RPM * 1.15 / c.GEAR_STEP)

    # --- one control tick ------------------------------------------------------------
    def step(self, kmh: float, pedal: float, dt: float) -> DriveState:
        c = self.cfg
        pedal = min(1.0, max(0.0, pedal))
        self._pedal_s += (pedal - self._pedal_s) * min(1.0, dt / c.PEDAL_SMOOTH_SEC)
        self._pedal_slow += (pedal - self._pedal_slow) * min(1.0, dt / 0.25)
        self._since_shift += dt
        event = None

        # gear choice (not while a shift is still in progress)
        if self._shift_t <= 0.0 and self._since_shift >= c.MIN_GEAR_SEC:
            up_at = self.upshift_rpm(self._pedal_s)
            w = self.wheel_rpm(self.gear, kmh)
            flat_out = self._pedal_s >= c.LIMITER_PEDAL
            fixed = c.SHIFT_KMH[self.gear - 1] if self.gear <= len(c.SHIFT_KMH) else 0.0
            if fixed > 0:             # low gears: shift at a set road speed
                ready = kmh >= fixed or self._limiter_t >= c.LIMITER_HOLD_SEC
            elif flat_out:            # hold the gear into the limiter, then shift
                ready = self._limiter_t >= c.LIMITER_HOLD_SEC
            else:
                ready = w >= up_at
            if self.gear < c.GEARS and ready:
                event = self._shift(self.gear + 1, pedal)
            elif self.gear > 1:
                down_at = max(c.IDLE_RPM * 1.1,
                              self.upshift_rpm(self._pedal_s) * c.GEAR_STEP * c.DOWNSHIFT_MARGIN)
                lower = self.wheel_rpm(self.gear - 1, kmh)
                stab = pedal - self._pedal_slow > c.KICKDOWN_STAB
                below = self.gear - 2 < len(c.SHIFT_KMH) and c.SHIFT_KMH[self.gear - 2] > 0
                if below:             # gears with set shift speeds come back 20 % below them
                    down = kmh < 0.8 * c.SHIFT_KMH[self.gear - 2]
                else:
                    down = w < down_at
                if down or (stab and lower < 0.9 * c.REDLINE_RPM):
                    event = self._shift(self.gear - 1, pedal)
            if kmh < 1.0:
                self.gear = 1

        # engine speed target: the wheels, but never below idle; clutch slip from rest
        target = max(self.wheel_rpm(self.gear, kmh), c.IDLE_RPM)
        if self.gear == 1:
            target = max(target, c.IDLE_RPM + (c.LAUNCH_RPM - c.IDLE_RPM) * pedal)
        limiter = False
        if target >= c.REDLINE_RPM:
            target = c.REDLINE_RPM
            limiter = pedal > 0.5
        cut = False
        if limiter:
            self._limiter_t += dt
            cut = bool(int(self._limiter_t * c.LIMITER_HZ * 2) % 2)
            if cut:
                target = c.REDLINE_RPM * (1.0 - c.LIMITER_DIP)
        else:
            self._limiter_t = 0.0
        blip = 0.0
        if self._shift_t > 0.0 and not self._shift_up and self._pedal_s < 0.3:
            frac = 1.0 - self._shift_t / self._shift_len
            if frac < 0.6:
                blip = math.sin(math.pi * frac / 0.6)
                target *= 1.0 + c.DOWNSHIFT_BLIP * blip
        tau = c.INERTIA_SEC
        if self._shift_t > 0.0:
            tau = max(tau, self._shift_len / 3.0)     # the swing to the new gear's speed
        self.rpm += (target - self.rpm) * min(1.0, dt / tau)

        # torque: follows pedal, cut through a shift, negative on overrun
        shifting = 0.0
        if self._shift_t > 0.0:
            # deepest at the start, torque back in over the last 40 % of the shift
            frac = 1.0 - self._shift_t / self._shift_len
            shifting = 1.0 if frac < 0.6 else max(0.0, (1.0 - frac) / 0.4)
            self._shift_t -= dt
        if cut:
            shifting = max(shifting, 1.0)
        if self._pedal_s > 0.04 or kmh < c.OVERRUN_KMH:
            load = self._pedal_s * (1.0 - shifting)
        else:
            load = -min(1.0, (self.rpm - c.IDLE_RPM) / (c.REDLINE_RPM - c.IDLE_RPM) + 0.2)

        if blip > 0.0:                # the blip is a throttle stab: briefly on load
            load = max(load, 0.4 * blip)
        rpm_norm = (self.rpm - c.IDLE_RPM) / (c.REDLINE_RPM - c.IDLE_RPM)
        return DriveState(self.rpm, max(0.0, min(1.0, rpm_norm)), self.gear, load, shifting,
                          limiter, event, cut)

    def _shift(self, new_gear: int, pedal: float) -> str:
        c = self.cfg
        up = new_gear > self.gear
        self._shift_up = up
        self.gear = new_gear
        self._since_shift = 0.0
        self._shift_len = c.SHIFT_SEC_LIGHT + (c.SHIFT_SEC_FULL - c.SHIFT_SEC_LIGHT) * pedal
        self._shift_t = self._shift_len
        return "up" if up else "down"
