"""Game-style engine sound: recorded loops at several rpm points, blended and
pitch-shifted by the virtual drivetrain (drivetrain.py) -- how racing games do it.

  loops     on-load loops, each recorded at a known firing frequency (REF_HZ). The
            drivetrain's rpm gives the firing frequency now; each loop plays at
            rate = now / REF_HZ, and the two loops either side of it crossfade
            (equal power, on a log-frequency scale), so no loop is ever pitched far.
  light     optional light-throttle loop: pedal load crossfades it with the on-load loops
            (how many game sound sets separate low and high load).
  off-load  a coasting loop blends in when you lift off (engine braking).
  events    the sound set's own one-shots: a shift sound on each up/downshift; exhaust
            pops just after lifting off from high revs, then overrun crackle that keeps
            popping while the revs stay up (denser at higher revs); and the rev limiter
            (drivetrain.py: flat out, every gear bounces off it before the upshift) chops
            the sound with each fuel cut.
  playback  pitch-synchronous granular (GRANULAR = 1): each loop is cut into grains of
            two engine cycles (two crank turns each -- we know its firing frequency and
            cylinder count), and grains are retriggered at
            the target firing frequency while the read position moves through the loop at
            normal speed. Pitch and rhythm follow the revs but the recording's resonances
            stay where they are -- no sped-up-tape / slowed-tape sound, even when a loop is
            stretched far from the revs it was recorded at. Grain positions snap to whole
            engine cycles so overlapping grains line up. GRANULAR = 0 plays loops like tape.
  life      grains jitter randomly around the read position, and each loop's pitch (+-0.3 %)
            and level wander slowly, so a short loop never audibly repeats.
  tone      load shapes the timbre, not just the level: under load the sound is driven
            harder (soft saturation), opens up (low-pass rises) and gains presence; light
            throttle is softer; the overrun is dark. A quick stab of the pedal (tip-in)
            gives a short extra surge.
Loops are made seamless at load time (authored loop points, or tail crossfaded into head).

The only sound set so far comes from an Assetto Corsa car mod (extracted with
fsb_extract.py into sounds/extracted/, git-ignored): copyrighted, personal use only.
A set is used only if its files are present.
"""
from __future__ import annotations

import dataclasses
import math
import random
from pathlib import Path

import numpy as np
from signalflow import (BiquadFilter, Buffer, BufferPlayer, EnvelopeBuffer, Granulator, Impulse,
                        Modulo, RandomUniform, Round, SawLFO, Smooth, Tanh)

import drivetrain
from voices import State, Voice, _target

HERE = Path(__file__).parent
SR = 48000

# sound sets: loop file -> firing frequency it was recorded at (measured from its
# harmonic spacing, see the session notes), plus one-shots
SETS = {
    "bmw_m3_gtr": {
        "dir": HERE / "sounds" / "extracted" / "gue_bmw_m3_e46_gtr",
        "cylinders": 8,
        "on": [("m3_onidle_ex", 143.0), ("m3_onlow_ex", 282.8), ("m3_onmid_ex", 426.5)],
        "off": ("BMW_V8_ex_offhigh", 267.5),
        "shift_up": "991_gt2_clubsport_shift_up5",
        "shift_down": "gue_cayman_rsr_shiftdown1_ext",
        "pops": ["991_gt2_clubsport_pop1", "991_gt2_clubsport_pop3", "991_gt2_clubsport_pop6",
                 "991_gt2_clubsport_pop8"],
    },
    # the same car from inside the cabin: interior loops are recorded near ~4000 rpm
    # (names describe load, not rpm), so they are pitch-shifted across the range, with
    # the ~5400 rpm loop taking over at the top
    "bmw_m3_gtr_int": {
        "dir": HERE / "sounds" / "extracted" / "gue_bmw_m3_e46_gtr",
        "cylinders": 8,
        # interior loops only cover ~4000-5400 rpm, so the exterior idle (~2150) and mid
        # (~6400) loops fill the ends -- EQ-matched to the interior loops' tone ("match")
        # so they sound heard from inside; all loops level-matched for even crossfades
        "on": [("m3_onidle_ex", 143.0, {"match": True}), ("e46_gtr_int_high", 265.0),
               ("e46_gtr_int_low2", 358.5), ("m3_onmid_ex", 426.5, {"match": True})],
        "level_match": True,
        "light": ("e46_gtr_int1", 271.0),
        "off": ("e46_gtr_int_off_high", 258.0),
        "shift_up": "991_gt2_clubsport_shift_up5",
        "shift_down": "gue_cayman_rsr_shiftdown1_ext",
        "pops": ["991_gt2_clubsport_pop1", "991_gt2_clubsport_pop3", "991_gt2_clubsport_pop6",
                 "991_gt2_clubsport_pop8"],
    },
    # Audi R8 GT (V10) mod: the bank has no sample names; these four were picked by ear
    # as the interior loops (an evenly spaced rev series). No off-throttle loop was found,
    # and its one-shots are unnamed, so shifts and pops borrow the BMW mod's.
    "audi_r8_int": {
        "dir": HERE / "sounds" / "extracted" / "fx_audi_r8_gt_evo",
        "cylinders": 10,
        "on": [("sample_050", 179.5), ("sample_087", 206.2), ("sample_035", 231.0),
               ("sample_074", 282.8)],
        "shift_up": "gue_bmw_m3_e46_gtr/991_gt2_clubsport_shift_up5",
        "shift_down": "gue_bmw_m3_e46_gtr/gue_cayman_rsr_shiftdown1_ext",
        "pops": ["gue_bmw_m3_e46_gtr/991_gt2_clubsport_pop1",
                 "gue_bmw_m3_e46_gtr/991_gt2_clubsport_pop3",
                 "gue_bmw_m3_e46_gtr/991_gt2_clubsport_pop6"],
    },
}
# Epic Stock Media "Game Engines - Futuristic Vehicles Kit": designed sci-fi engine loops
# at four intensities (Idle / Low / Medium / High), licensed for use inside our own app
# (not for redistribution as raw files: sounds/kit/ is git-ignored). Not rpm recordings,
# so they are spread evenly over the rev range ("cylinders": 2 makes the reference
# "firing frequency" simply rpm / 60) and pitch-shifted only gently (PITCH_DEPTH).
_KIT = HERE / "sounds" / "kit" / "futuristic_vehicles"
SETS["battalion"] = {
    "dir": _KIT, "cylinders": 2,
    "on": [("battalion_idle", 2200 / 60), ("battalion_low", 3800 / 60),
           ("battalion_medium", 5400 / 60), ("battalion_high", 7000 / 60)],
}
# Assetto Corsa "Hyundai IONIQ 5 N" mod (bsk_): a recreation of the N's combustion-style
# sound mode, built from petrol-engine recordings. Every loop is named with its rpm, so the
# references are exact ("cylinders": 2 -> reference = rpm / 60). Interior = stereo "500ss".
_IONIQ = HERE / "sounds" / "extracted" / "bsk_hyundai_ioniq5_n"
SETS["ioniq5n_int"] = {
    "dir": _IONIQ, "cylinders": 2,
    "on": [("500ss_idle", 1000 / 60), ("500ss_on_3000", 3000 / 60), ("500ss_on_4000", 4000 / 60),
           ("500ss_on_5500", 5500 / 60), ("500ss_on_7500", 7500 / 60)],
    "off_series": [("500ss_off_2500", 2500 / 60), ("500ss_off_4000", 4000 / 60),
                   ("500ss_off_7500", 7500 / 60)],
    "shift_up": "gearup",
    "shift_down": "geardnEXT",
    "pops": ["500_backfire3", "500_backfire4", "500_backfire5", "500_backfire6", "500_backfire7"],
}
EXTRACTED = HERE / "sounds" / "extracted"


def _file(s: dict, name: str) -> Path:
    """A sample of this set, or "<other set dir>/<name>" to borrow from another one."""
    return (EXTRACTED / f"{name}.wav") if "/" in name else (s["dir"] / f"{name}.wav")


def _load(path: Path, seamless: bool = False) -> np.ndarray:
    import json
    import automap
    x = automap.load(str(path), SR).astype(np.float32)
    meta = path.parent / "loops.json"            # authored loop points (fsb_extract.py)
    if seamless and meta.exists():
        m = json.loads(meta.read_text()).get(path.stem)
        if m and m.get("loop"):
            k = SR / m["rate"]
            a, b = int(round(m["loop"][0] * k)), int(round((m["loop"][1] + 1) * k))
            return x[a:b].copy()
    if seamless and len(x) > SR // 2:
        n = int(0.08 * SR)                       # crossfade the tail into the head
        head, tail = x[:n].copy(), x[-n:]
        ramp = np.sin(np.linspace(0, np.pi / 2, n)) ** 2
        x = x[:-n].copy()
        x[:n] = head * ramp + tail * (1.0 - ramp)
    return x


_HANN = list(np.hanning(512))


def _periods(x: np.ndarray, ref: float, cylinders: int) -> tuple[float, float]:
    """Exact (firing period, engine-cycle length) of a loop in seconds, from its
    autocorrelation: the firing period near 1/ref, the full cycle near `cylinders` of those."""
    y = x[: min(len(x), 4 * SR)].astype(np.float64)
    y = y - y.mean()
    n = 1 << int(np.ceil(np.log2(2 * len(y))))
    ac = np.fft.irfft(np.abs(np.fft.rfft(y, n)) ** 2, n)[: len(y)]
    ac /= ac[0] + 1e-12

    def peak_near(lag: float, tol: float) -> float:
        a, b = int(lag * (1 - tol)), int(lag * (1 + tol)) + 1
        if b + 1 >= len(ac) or a < 1:
            return lag
        k = a + int(np.argmax(ac[a:b]))
        al, be, ga = ac[k - 1], ac[k], ac[k + 1]          # parabolic refinement
        den = al - 2 * be + ga
        return k + (0.5 * (al - ga) / den if den != 0 else 0.0)

    fire = peak_near(SR / ref, 0.03)
    cycle = peak_near(fire * cylinders, 0.01)
    return fire / SR, cycle / SR


def _eq_match(x: np.ndarray, refs: list[np.ndarray], max_db: float = 15.0) -> np.ndarray:
    """Give loop x the average tone of `refs` (e.g. an exterior loop heard from inside):
    the ratio of their 1/3-octave spectra, smoothed and limited, applied as a zero-phase
    filter. Circular (whole-loop FFT), so a seamless loop stays seamless."""
    def bands_db(y):
        n = 1 << 14
        P = np.mean([np.abs(np.fft.rfft(y[i:i + n] * np.hanning(n))) ** 2
                     for i in range(0, max(1, len(y) - n), n // 2)], axis=0)
        fr = np.fft.rfftfreq(n, 1 / SR)
        c = 40.0 * 2 ** (np.arange(28) / 3)               # 40 Hz .. ~20 kHz
        return c, np.array([10 * np.log10(P[(fr >= f * 2 ** -(1 / 6)) & (fr < f * 2 ** (1 / 6))].mean()
                                          + 1e-20) for f in c])
    c, own = bands_db(x)
    target = np.mean([bands_db(r)[1] for r in refs], axis=0)
    diff = np.clip(target - own, -max_db, max_db)
    diff = np.convolve(np.pad(diff, 1, mode="edge"), np.ones(3) / 3, "valid")   # smooth
    X = np.fft.rfft(x.astype(np.float64))
    fr = np.fft.rfftfreq(len(x), 1 / SR)
    g = 10 ** (np.interp(np.log(np.maximum(fr, 1.0)), np.log(c), diff) / 20)
    return np.fft.irfft(X * g, len(x)).astype(np.float32)


def _xfade(lf: float, lr: list[float]) -> list[float]:
    """Equal-power weights for the two loops either side of log-frequency lf."""
    w = [0.0] * len(lr)
    if lf <= lr[0]:
        w[0] = 1.0
    elif lf >= lr[-1]:
        w[-1] = 1.0
    else:
        i = max(k for k in range(len(lr) - 1) if lr[k] <= lf)
        u = (lf - lr[i]) / (lr[i + 1] - lr[i])
        w[i], w[i + 1] = math.cos(u * math.pi / 2), math.sin(u * math.pi / 2)
    return w


def available(name: str) -> bool:
    s = SETS[name]
    return all(_file(s, e[0]).exists() for e in s["on"])


class LoopEngine(Voice):
    name = "Loop Engine (BMW M3 GTR)"
    own_gears = True
    SET = "bmw_m3_gtr"
    LEVEL = 1.0
    OFF_LEVEL = 0.8           # coasting loop level on lift-off
    PITCH = 1.0               # overall pitch: the set's absolute revs can be a guess
    PITCH_DEPTH = 1.0         # how far loops are pitch-shifted (1 = follow revs exactly,
                              # 0.5 = half as far: for designed loops, not rpm recordings)
    ON_MIN = 0.25             # on-load loops never drop below this (light throttle)
    SHIFT = 0.7               # shift one-shot level
    POPS = 0.6                # lift-off pops level (0 = off)
    SHIFT_DIP = 0.2
    LIMITER_CUT = 0.65        # how much each limiter fuel cut drops the level
    CRACKLE_RATE = 5.0        # overrun pops per second at the redline (0 = off)
    # playback
    GRANULAR = 0.0            # 1 = pitch-synchronous grains (sounded broken by ear), 0 = tape-style
    GRAIN_CYCLES = 2.0        # grain length, in periods of GRAIN_SYNC
    GRAIN_SYNC = "firing"     # snap grains to the firing period ("firing") or full cycle
    JITTER_SEC = 0.01         # random scatter of grain positions (anti-repetition)
    WANDER = 0.003            # slow random pitch drift (+-0.3 %)
    LEVEL_WANDER = 0.1        # slow random level drift per loop (+-10 %)
    # tone (a stronger setting -- DRIVE 3, BITE 8 dB -- distorted at high volume)
    TONE_FULL_HZ = 12000.0    # low-pass at full load
    TONE_LIGHT_HZ = 4500.0    # ... at light load
    TONE_OVERRUN_HZ = 2200.0  # ... on the overrun
    DRIVE = 0.8               # extra saturation at full load
    BITE_HZ, BITE_DB = 1800.0, 3.0   # presence lift at full load
    SWELL_BITE = 0.6          # extra load on a quick pedal stab (tip-in)
    # drivetrain constants (GEARS, GEAR_STEP, ...) are added below the class

    def __init__(self):
        self.dt = drivetrain.Drivetrain()
        self.set = SETS[self.SET]
        self._was_on = False
        self._pop_queue: list[int] = []
        self._trig_ticks: dict[str, int] = {}
        super().__init__()

    def build(self):
        s = self.set
        t = {"gain": _target(0.0), "off": _target(0.0)}
        mix = None
        self.refs = []
        loops = [_load(_file(s, e[0]), seamless=True) for e in s["on"]]
        opts = [e[2] if len(e) > 2 else {} for e in s["on"]]
        native = [x for x, o in zip(loops, opts) if not o.get("match")]
        loops = [_eq_match(x, native) if o.get("match") else x for x, o in zip(loops, opts)]
        if s.get("level_match"):
            rms = [float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) for x in loops]
            ref_rms = float(np.median(rms))
            loops = [(x * (ref_rms / (r + 1e-9))).astype(np.float32) for x, r in zip(loops, rms)]
        for i, (entry, x) in enumerate(zip(s["on"], loops)):
            ref = entry[1]
            t[f"r{i}"], t[f"w{i}"] = _target(1.0), _target(0.0)
            p = self._loop(x, ref, t[f"r{i}"]) * Smooth(t[f"w{i}"], 0.999)
            mix = p if mix is None else mix + p
            self.refs.append(ref)
        self.light_ref = None
        if s.get("light"):
            fname, self.light_ref = s["light"]
            t["light_rate"], t["light"] = _target(1.0), _target(0.0)
            mix = mix + self._loop(_load(_file(s, fname), seamless=True), self.light_ref,
                                   t["light_rate"]) * Smooth(t["light"], 0.999)
        # off-throttle: one loop ("off") or a series by rpm ("off_series"), crossfaded
        # by revs like the on-load loops, all under the shared "off" level
        offs = s.get("off_series") or ([s["off"]] if s.get("off") else [])
        self.off_refs = []
        off_level = Smooth(t["off"], 0.998)
        for j, (fname, ref) in enumerate(offs):
            t[f"offr{j}"], t[f"offw{j}"] = _target(1.0), _target(0.0)
            mix = mix + self._loop(_load(_file(s, fname), seamless=True), ref, t[f"offr{j}"]) \
                * Smooth(t[f"offw{j}"], 0.999) * off_level
            self.off_refs.append(ref)
        # tone: load drives the engine harder, opens it up and adds presence
        t.update({"drive": _target(1.0), "drive_inv": _target(1.0),
                  "tone_lp": _target(self.TONE_LIGHT_HZ), "bite": _target(0.0)})
        mix = Tanh(mix * Smooth(t["drive"], 0.999)) * Smooth(t["drive_inv"], 0.999)
        mix = BiquadFilter(mix, "low_pass", cutoff=Smooth(t["tone_lp"], 0.999), resonance=0.7071)
        mix = BiquadFilter(mix, "peak", cutoff=self.BITE_HZ, resonance=1.0 * math.sqrt(10.0),
                           peak_gain=Smooth(t["bite"], 0.999))
        # one-shots: retriggered by a clock target going 0 -> 1
        for key in [k for k in ("shift_up", "shift_down") if s.get(k)]:
            t[f"{key}_clk"] = _target(0.0)
            buf = Buffer(np.array([_load(_file(s, s[key]))]))
            mix = mix + BufferPlayer(buf, clock=t[f"{key}_clk"]) * self.SHIFT
        for j, fname in enumerate(s.get("pops", [])):
            t[f"pop{j}_clk"] = _target(0.0)
            buf = Buffer(np.array([_load(_file(s, fname))]))
            mix = mix + BufferPlayer(buf, clock=t[f"pop{j}_clk"]) * self.POPS
        return mix * Smooth(t["gain"], 0.999), t

    def _loop(self, x: np.ndarray, ref: float, rate_t):
        """One loop, played at `rate_t` (target firing freq / ref): granular or tape."""
        buf = Buffer(np.array([x]))
        wander = Smooth(RandomUniform(1.0 - self.WANDER, 1.0 + self.WANDER,
                                      clock=Impulse(0.6)), 0.9999)
        level = Smooth(RandomUniform(1.0 - self.LEVEL_WANDER, 1.0 + self.LEVEL_WANDER,
                                     clock=Impulse(0.4)), 0.9999)
        rate = Smooth(rate_t, 0.999) * wander
        if self.GRANULAR <= 0:
            return BufferPlayer(buf, rate=rate, loop=1) * level
        L = len(x) / SR
        # grains snap to whole firing periods (measured exactly from the loop) and one is
        # triggered per firing. (Snapping to full engine cycles keeps the cylinder-to-
        # cylinder pattern, but grain-to-grain differences then repeat at cycle rate and
        # show up as false half orders -- worse than losing the loops' weak real ones.)
        fire, cycle = _periods(x, ref, self.set["cylinders"])
        T0 = cycle if self.GRAIN_SYNC == "cycle" else fire
        clock = Impulse(rate * (1.0 / T0))
        ptr = SawLFO(1.0 / L, 0.0, L) + random.uniform(0.0, L)
        jit = RandomUniform(-self.JITTER_SEC, self.JITTER_SEC, clock=clock)
        pos = Modulo(Round((ptr + jit) / T0) * T0 + L, L)   # snap to whole cycles
        g = Granulator(buf, clock=clock, pos=pos, duration=self.GRAIN_CYCLES * T0, wrap=True)
        g.set_buffer("envelope", EnvelopeBuffer(_HANN))
        # overlapping Hann grains add up to GRAIN_CYCLES/2 * rate: undo that
        return g * (2.0 / self.GRAIN_CYCLES) / rate * level

    def _fire(self, key: str):
        self._trig_ticks[key] = 2              # high for 2 ticks, then low (re-armable)

    def update(self, st: State) -> str:
        for fld in dataclasses.fields(drivetrain.DrivetrainConfig):   # live-tunable
            setattr(self.dt.cfg, fld.name, getattr(self, fld.name))
        d = self.dt.step(st.speed_kmh, st.pedal, st.dt)
        tg = self.targets
        f = d.rpm / 60.0 * self.set["cylinders"] / 2.0 * self.PITCH   # firing freq (4-stroke)

        # crossfade the two loops either side of f (log scale), equal power
        lf = math.log(f)
        w = _xfade(lf, [math.log(r) for r in self.refs])
        on = max(0.0, d.load)
        if d.shifting > 0:
            on = max(on, max(0.0, self.dt._pedal_s) * (1.0 - 0.3 * d.shifting))
        on_level = (self.ON_MIN + (1.0 - self.ON_MIN) * on) * (1.0 - self.SHIFT_DIP * d.shifting)
        heavy = on_level
        if self.light_ref is not None:          # pedal crossfades light <-> heavy load
            heavy = on_level * math.sin(on * math.pi / 2)
            tg["light_rate"].set_value((f / self.light_ref) ** self.PITCH_DEPTH)
            tg["light"].set_value(on_level * math.cos(on * math.pi / 2) * (d.load >= -0.05))
        for k, ref in enumerate(self.refs):
            tg[f"r{k}"].set_value((f / ref) ** self.PITCH_DEPTH)
            tg[f"w{k}"].set_value(w[k] * heavy)
        if self.off_refs:
            w_off = _xfade(lf, [math.log(r) for r in self.off_refs])
            for j, ref in enumerate(self.off_refs):
                tg[f"offr{j}"].set_value((f / ref) ** self.PITCH_DEPTH)
                tg[f"offw{j}"].set_value(w_off[j])
        tg["off"].set_value(0.0 if not self.off_refs else self.OFF_LEVEL * max(0.0, -d.load) ** 0.7
                            + 0.3 * self.OFF_LEVEL * (1.0 - on) * (d.load >= 0))

        # tone: from load, plus a short surge on a quick pedal stab
        tone_on = min(1.0, on + self.SWELL_BITE * st.swell) if d.load >= 0 else 0.0
        if d.load < 0:
            lp = self.TONE_OVERRUN_HZ
        else:
            lp = self.TONE_LIGHT_HZ * (self.TONE_FULL_HZ / self.TONE_LIGHT_HZ) ** tone_on
        drive = 1.0 + self.DRIVE * tone_on
        tg["drive"].set_value(drive)
        tg["drive_inv"].set_value(1.0 / drive)
        tg["tone_lp"].set_value(lp)
        tg["bite"].set_value(self.BITE_DB * tone_on)

        # events
        if d.event == "up" and self.set.get("shift_up"):
            self._fire("shift_up")
        elif d.event == "down" and self.set.get("shift_down"):
            self._fire("shift_down")
        pops = self.POPS > 0 and bool(self.set.get("pops"))
        if d.load < 0 and self._was_on and d.rpm_norm > 0.35 and pops:
            n = random.randint(1, 3)            # a few pops over the next second
            self._pop_queue = sorted(random.sample(range(2, 50), n))
        self._was_on = d.load > 0.2
        if self._pop_queue and self._pop_queue[0] <= 0:
            self._pop_queue.pop(0)
            self._fire(f"pop{random.randrange(len(self.set['pops']))}")
        self._pop_queue = [q - 1 for q in self._pop_queue]
        for key in [k for k in tg if k.endswith("_clk")]:
            name = key[:-4]
            left = self._trig_ticks.get(name, 0)
            tg[key].set_value(1.0 if left > 0 else 0.0)
            self._trig_ticks[name] = max(0, left - 1)

        # overrun crackle: random pops while coasting with revs up
        if d.load < -0.05 and d.rpm_norm > 0.25 and not self._pop_queue and pops:
            if random.random() < self.CRACKLE_RATE * st.dt * d.rpm_norm ** 1.5:
                self._fire(f"pop{random.randrange(len(self.set['pops']))}")
        chop = 1.0 - self.LIMITER_CUT * d.cut
        tg["gain"].set_value(self.LEVEL * (0.6 + 0.4 * d.rpm_norm) * chop if st.active else 0.0)
        tag = "  LIMITER" if d.limiter else ("  shift" if d.shifting > 0 else "")
        return f"gear {d.gear}  {d.rpm:4.0f} rpm  load {d.load * 100:+4.0f}%{tag}"


for _f in dataclasses.fields(drivetrain.DrivetrainConfig):     # tunable in the lab
    setattr(LoopEngine, _f.name, _f.default)


class LoopEngineInterior(LoopEngine):
    """The same car heard from inside the cabin."""
    name = "Loop Engine (BMW M3 GTR interior)"
    SET = "bmw_m3_gtr_int"


class LoopEngineAudi(LoopEngine):
    """Audi R8 GT mod, interior loops (chosen by ear: no names in the bank). Without an
    off-throttle loop, lifting off keeps the interior loops at a lower level."""
    name = "Loop Engine (Audi R8 interior)"
    SET = "audi_r8_int"
    ON_MIN = 0.45
    PITCH = 0.75              # the loops span a narrow range: split the stretch evenly


class KitBattalion(LoopEngine):
    """Futuristic Vehicles Kit, "Battleship Battalion": four designed intensity loops."""
    name = "Battleship Battalion (kit)"
    SET = "battalion"
    PITCH_DEPTH = 0.5
    ON_MIN = 0.5              # a spacecraft drive doesn't die away off throttle


class LoopEngineIoniq(LoopEngine):
    """IONIQ 5 N Assetto Corsa mod, interior: rpm-labelled loops (idle ~1000 rpm, on-load
    3000-7500, off-load 2500-7500), the mod's own gear and backfire sounds."""
    name = "Loop Engine (IONIQ 5 N mod interior)"
    SET = "ioniq5n_int"
    IDLE_RPM = 1000.0         # the loops start at a ~1000 rpm idle and run to 7500
    REDLINE_RPM = 7500.0
    LAUNCH_RPM = 3500.0
