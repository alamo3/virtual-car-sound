"""Python twin of Open Blue's Dct.kt (7-speed dual-clutch gearbox + control unit), kept in step
with it so offline renders shift exactly as the app does."""
import math

RATIOS = [3.91, 2.29, 1.65, 1.29, 1.08, 0.88, 0.62]   # 7-speed DCT (sports-car spread)
EV_REDUCTION = 10.65          # IONIQ 5: motor -> wheels
MOTOR_RPM_PER_KMH = 77.0      # starting guess, learned from the Pi's motor rpm
GEAR1_TOP_KMH = 62.0          # 1st gear reaches the redline here (sets the final drive)


def lerp(a, b, t):
    return a + (b - a) * max(0.0, min(1.0, t))


def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


class Dct:
    MIN_GEAR_SEC = 0.5
    FAST_OFF_HOLD_SEC = 2.5
    KICKDOWN_JUMP = 0.25          # pedal rise over ~0.3 s that counts as a kickdown request
    BRAKE_DECEL = 2.5             # m/s^2 off the pedal that counts as braking (regen alone is ~1.7)
    CLUTCH_CLOSE_SEC = 1.2
    LIMITER_HZ = 12.0
    LIMITER_PEDAL = 0.9           # floored: run into the limiter before upshifting...
    LIMITER_HOLD_SEC = 0.3        # ...and bounce off it this long
    LIMITER_DIP = 0.04

    def __init__(self, idle=900.0, redline=7000.0):
        self.idle, self.redline = idle, redline
        self.motor_per_kmh = MOTOR_RPM_PER_KMH
        self.gear = 1
        self.rpm = idle
        self.sport = 0.0
        self.pedal_t = 0.0        # torque pedal (fast)
        self.pedal_avg = 0.0      # ~0.3 s average, for kickdown / fast-off detection
        self.since_shift = 1e9
        self.shift_t = 0.0
        self.shift_len = 0.0
        self.shift_from_rpm = 0.0
        self.shift_up = True
        self.hold = 0.0           # fast-off: upshifts held this long
        self.last_kmh = 0.0
        self.accel = 0.0
        self.shifts = self.downshifts = self.lift_offs = 0
        self.pull_peak = 0.0
        self.last_motor = 0
        self.last_pedal = 0.0
        self.clutch_locked = False
        self.slip_t = 0.0
        self.limiter_t = 0.0
        self.pedal_rate = 0.0

    # --- gear train --------------------------------------------------------------------
    def final_drive(self):
        # chosen so 1st gear tops out at GEAR1_TOP_KMH at the redline
        wheel_per_kmh = self.motor_per_kmh / EV_REDUCTION
        return (self.redline / GEAR1_TOP_KMH) / (wheel_per_kmh * RATIOS[0])

    def gear_rpm(self, g, kmh):
        wheel_rpm = kmh * self.motor_per_kmh / EV_REDUCTION    # = motor rpm / reduction
        return wheel_rpm * self.final_drive() * RATIOS[g - 1]

    # --- shift schedule ----------------------------------------------------------------
    def up_rpm(self, p):
        s = self.sport
        light = lerp(0.22, 0.50, s)
        frac = light + (0.97 - light) * p ** lerp(1.6, 1.0, s)
        return frac * self.redline

    def lug_rpm(self, p):
        """Lowest rpm the box will run at for this pedal (it won't labour the engine)."""
        return lerp(self.idle * 1.25, 0.40 * self.redline, p) * lerp(1.0, 1.25, self.sport)

    def step(self, dt, kmh, pedal, motor_rpm=0):
        R, idle = self.redline, self.idle
        # learn the car's own motor-rpm per km/h (whenever the Pi sends a new reading)
        if motor_rpm and motor_rpm != self.last_motor and kmh > 15:
            self.motor_per_kmh += (abs(motor_rpm) / kmh - self.motor_per_kmh) * 0.1
        self.last_motor = motor_rpm
        a = (kmh - self.last_kmh) / 3.6 / dt if dt > 0 else 0.0
        self.accel += (a - self.accel) * min(1, dt / 0.25)
        self.last_kmh = kmh
        prev_avg = self.pedal_avg
        self.pedal_t += (pedal - self.pedal_t) * min(1, dt / 0.04)
        self.pedal_avg += (pedal - self.pedal_avg) * min(1, dt / 0.3)
        self.since_shift += dt
        self.hold = max(0.0, self.hold - dt)

        # driving style: quick to rise on hard use, slow to settle back. Signs of hard use: a
        # deep pedal, a pedal stamped on (fast, and far), hard acceleration
        r = (pedal - self.last_pedal) / dt if dt > 0 else 0.0
        self.last_pedal = pedal
        self.pedal_rate += (r - self.pedal_rate) * min(1, dt / 0.1)
        inst = max((pedal - 0.35) / 0.55, (self.pedal_rate - 1.5) / 3.0 * pedal, (self.accel - 1.0) / 3.0)
        inst = max(0.0, min(1.0, inst))
        self.sport += (inst - self.sport) * min(1, dt / (1.5 if inst > self.sport else 30.0))

        # fast-off: a quick lift holds the gear (no upshift mid-corner / before re-accelerating)
        if prev_avg > 0.45 and pedal < 0.1 and prev_avg - pedal > 0.35:
            self.hold = self.FAST_OFF_HOLD_SEC
        if pedal > 0.2:
            self.hold = min(self.hold, 0.0) if pedal > prev_avg + 0.05 else self.hold

        braking = pedal < 0.04 and self.accel < -self.BRAKE_DECEL
        g = self.gear
        if self.shift_t <= 0 and kmh > 2:
            here = self.gear_rpm(g, kmh)
            # kickdown: a deep, fast press -- or floored and well down the revs (not just after a
            # shift, or it would undo the upshift it just made)
            kick = pedal > 0.8 and pedal - self.pedal_avg > self.KICKDOWN_JUMP * 0.5 or \
                   (pedal > 0.9 and here < 0.55 * R and g > 1 and self.since_shift >= self.MIN_GEAR_SEC)
            target = g
            if kick:
                # the lowest gear that lands in the power band: under 75 % of the redline and
                # under the upshift point (may skip gears)
                limit = min(0.75 * R, 0.9 * self.up_rpm(pedal))
                for cand in range(1, g):
                    if self.gear_rpm(cand, kmh) < limit:
                        target = cand
                        break
            elif self.since_shift >= self.MIN_GEAR_SEC:
                # no upshifts while slowing down (it would only come back down): coasting and
                # braking stay in gear or go down
                slowing = self.accel < -0.5 and pedal < 0.2
                lifting = self.pedal_rate < -1.5            # mid-lift: never upshift on the way off
                # floored, it holds the gear into the rev limiter and bounces off it first
                up_ready = self.limiter_t >= self.LIMITER_HOLD_SEC if pedal >= self.LIMITER_PEDAL \
                    else here >= self.up_rpm(self.pedal_avg)
                if g < len(RATIOS) and self.hold <= 0 and not slowing and not lifting and up_ready \
                        and self.gear_rpm(g + 1, kmh) >= self.lug_rpm(self.pedal_avg) * 0.9:
                    target = g + 1
                elif g > 1:
                    lower = self.gear_rpm(g - 1, kmh)
                    want = self.lug_rpm(self.pedal_avg)
                    if braking:   # downshift on the brakes, earlier the sportier
                        want = max(want, lerp(0.30, 0.55, self.sport) * R)
                    if here < want and lower < min(0.88 * R, 0.92 * self.up_rpm(self.pedal_avg) + 0.2 * R):
                        target = g - 1
            if self.hold > 0 and here > 0.97 * R and g < len(RATIOS):
                target = g + 1   # held, but not into the limiter
            if target != g:
                self.begin_shift(target, pedal)
        if kmh <= 2 and self.shift_t <= 0:
            if self.gear != 1:
                self.gear = 1
        g = self.gear

        # engine speed: wheels through the gear train, clutch slip pulling away
        wheel = self.gear_rpm(g, kmh)
        target = max(wheel, idle)
        if g == 1:
            # pulling away: the clutch slips, holding the engine at a launch rpm (higher the harder
            # you press) until the wheels catch up; then it's locked until the car stops again
            if not self.clutch_locked and pedal > 0.02:
                launch = lerp(idle * 1.3, lerp(0.45, 0.60, self.sport) * R, pedal ** 1.2)
                # the clutch closes over CLUTCH_CLOSE_SEC: the engine comes down to meet the wheels
                if kmh > 2:
                    self.slip_t += dt
                closing = min(1.0, self.slip_t / self.CLUTCH_CLOSE_SEC)
                # Gently, the revs settle toward a little over idle as it bites; under power it
                # holds the launch rpm and the car catches up (it never stalls the engine)
                bite = max(wheel, idle * 1.3)
                held = launch - (launch - bite) * smoothstep(closing) * (1 - pedal)
                target = max(target, held)
                if wheel >= 0.98 * held:
                    self.clutch_locked = True
        if kmh < 2:
            self.clutch_locked = False
            self.slip_t = 0.0
        limiter = cut = False
        if target >= R:
            target, limiter = R, pedal > 0.5
        if limiter:
            self.limiter_t += dt
            cut = int(self.limiter_t * self.LIMITER_HZ * 2) % 2 == 1
            if cut:
                target = R * (1 - self.LIMITER_DIP)
        else:
            self.limiter_t = 0.0
        shifting = 0.0
        blip = 0.0
        if self.shift_t > 0:
            frac = 1 - self.shift_t / self.shift_len
            k = smoothstep(frac)
            if self.shift_up:
                depth = lerp(0.25, 0.9, self.pedal_t)
                shifting = depth * (1 - abs(frac - 0.35) / 0.65 if frac > 0.35 else frac / 0.35)
                self.rpm = self.shift_from_rpm + (target - self.shift_from_rpm) * k
            else:
                blip = math.sin(math.pi * min(1, frac / 0.7))
                over = 1 + lerp(0.03, 0.08, self.sport) * blip * (1 - self.pedal_t)
                self.rpm = (self.shift_from_rpm + (target - self.shift_from_rpm) * k) * over
            self.shift_t -= dt
        else:
            self.rpm += (target - self.rpm) * min(1, dt / 0.06)

        load = self.pedal_t * (1 - shifting)
        if blip and pedal < 0.3:
            load = max(load, 0.45 * blip)    # the rev-matching blip is a throttle stab
        engaged = self.pedal_t > 0.04 or kmh < 6
        if engaged:
            self.pull_peak = max(self.pull_peak, load)
        elif self.pull_peak > 0.45 and self.rpm > 0.5 * R:
            self.lift_offs += 1
            self.pull_peak = 0.0
        else:
            self.pull_peak = 0.0
        overrun = 0.0 if engaged else min(1.0, (self.rpm - idle) / (R - idle) + 0.2)
        return dict(rpm=self.rpm, gear=g, load=max(0.0, load), shifting=shifting, limiter=limiter, cut=cut,
                    overrun=overrun, sport=self.sport, throttle=self.pedal_t, shifts=self.shifts,
                    downshifts=self.downshifts, lift_offs=self.lift_offs, engaged=engaged,
                    up=self.shift_up and self.shift_t > 0)

    def begin_shift(self, to, pedal):
        up = to > self.gear
        steps = abs(to - self.gear)
        if up:
            length = lerp(0.30, 0.10, max(pedal, 0.6 * self.sport))
            self.shifts += 1
        else:
            length = lerp(0.35, 0.16, self.sport) + 0.05 * (steps - 1)
            self.downshifts += 1
        self.shift_from_rpm = self.rpm
        self.shift_up = up
        self.gear = to
        self.shift_t = self.shift_len = length
        self.since_shift = 0.0
