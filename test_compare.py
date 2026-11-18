"""Offline checks for the jetson.mp3-based EV sound.

1. Loop is seamless: renders past the loop point and checks for level dips / clicks.
2. At the RPM where rate == 1.0 with full pedal, it matches jetson.mp3 (pitch + timbre).
3. Pitch follows RPM.
4. The load layer audibly enriches the sound, and relaxes at high speed (back-EMF).
5. Pedal drives shaft speed: full pedal settles near MAX_MOTOR_RPM, lift-off sags.
6. End to end: a throttle press raises the pitch, lift-off makes it sag.
7. The recording's fast flutter is flattened out of the loop.
8. Warble speed is set by WOBBLE_HZ, not by playback rate.
Dumps WAVs for listening. Usage: .venv/bin/python test_compare.py
"""
import wave
import numpy as np
from signalflow import AudioGraph, AudioGraphConfig, Buffer

import main


def write_wav(path, samples, sample_rate):
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())


def render(rpm_fn, pedal_fn, duration_sec, block=480):
    cfg = AudioGraphConfig()
    cfg.backend_name = "null"
    graph = AudioGraph(cfg, start=False)
    sr = int(graph.sample_rate)
    output, targets = main.create_ev_sound(main.build_loop_buffer())
    output.play()
    total, done, out = int(duration_sec * sr), 0, []
    while done < total:
        prog = done / total
        main.apply_controls(targets, rpm_fn(prog), pedal_fn(prog))
        n = min(block, total - done)
        graph.render(n)
        out.append(np.array(output.output_buffer)[0, :n].copy())
        done += n
    graph.destroy()
    return np.concatenate(out), sr


def render_drive(pedal_fn, duration_sec, block=480):
    """Like render(), but shaft speed comes from the powertrain instead of being given."""
    cfg = AudioGraphConfig()
    cfg.backend_name = "null"
    graph = AudioGraph(cfg, start=False)
    sr = int(graph.sample_rate)
    output, targets = main.create_ev_sound(main.build_loop_buffer())
    output.play()
    pt = main.Powertrain(0.0)
    total, done, out, rpms = int(duration_sec * sr), 0, [], []
    while done < total:
        pedal = pedal_fn(done / sr)
        rpm = pt.step(pedal, block / sr)
        main.apply_controls(targets, rpm, pedal)
        n = min(block, total - done)
        graph.render(n)
        out.append(np.array(output.output_buffer)[0, :n].copy())
        rpms.append(rpm)
        done += n
    graph.destroy()
    return np.concatenate(out), sr, np.array(rpms)


def low_peak_hz(x, sr):
    mag = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    m = (f > 60) & (f < 400)
    return float(f[m][np.argmax(mag[m])])


def band_profile_db(x, sr, edges=np.geomspace(60, 8000, 19)):
    mag = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    b = np.array([10 * np.log10(np.mean(mag[(f >= lo) & (f < hi)]) + 1e-20)
                  for lo, hi in zip(edges[:-1], edges[1:])])
    return b - b.max()


def rpm_for_rate(rate):
    curved = (rate - main.MIN_RATE) / (main.MAX_RATE - main.MIN_RATE)
    return main.MAX_MOTOR_RPM * curved ** (1 / main.PITCH_CURVE)


MAX_TEST_RPM = main.MAX_MOTOR_RPM
MID_RPM = 0.5 * main.MAX_MOTOR_RPM  # load layer is tuned around mid-range


if __name__ == "__main__":
    failures = []
    src = Buffer(main.SOURCE_FILE)
    ref = np.array(src.data).mean(axis=0)
    ref_sr = int(src.sample_rate)
    ref = ref[int(1 * ref_sr):int(13 * ref_sr)]

    # --- 1 + 2: hold rate 1.0, full pedal, load layer OFF, for longer than one loop ---
    unity_rpm = rpm_for_rate(1.0)
    saved_level = main.LOAD_LAYER_LEVEL
    main.LOAD_LAYER_LEVEL = 0.0
    x, sr = render(lambda p: unity_rpm, lambda p: 1.0, 20.0)
    main.LOAD_LAYER_LEVEL = saved_level
    x = x[int(0.5 * sr):]  # skip smoothing warm-up
    write_wav("synth_steady.wav", x, sr)

    hop = int(0.05 * sr)
    rms = np.array([np.sqrt(np.mean(x[i:i + hop] ** 2)) for i in range(0, len(x) - hop, hop)])
    dip_db = 20 * np.log10(rms.min() / np.median(rms))
    jumps = np.abs(np.diff(x))
    click_ratio = jumps.max() / np.percentile(jumps, 99.9)
    print(f"Loop: level range {dip_db:+.1f} dB below median over 20 s; "
          f"max sample step = {click_ratio:.2f}x the 99.9th percentile")
    if dip_db < -6:
        failures.append(f"level dips {dip_db:.1f} dB somewhere (loop seam?)")
    if click_ratio > 2.0:
        failures.append("possible click (sample jump far above normal)")

    f_ref, f_syn = low_peak_hz(ref, ref_sr), low_peak_hz(x, sr)
    p_ref, p_syn = band_profile_db(ref, ref_sr), band_profile_db(x, sr)
    lsd = float(np.mean(np.abs(np.maximum(p_ref, -50) - np.maximum(p_syn, -50))))
    print(f"Pitch at rate 1.0 ({unity_rpm:.0f} RPM): ref {f_ref:.1f} Hz | output {f_syn:.1f} Hz")
    print(f"Timbre difference vs {main.SOURCE_FILE}: {lsd:.1f} dB mean")
    if abs(f_syn - f_ref) > 3:
        failures.append("pitch at rate 1.0 doesn't match the source file")
    if lsd > 2.0:
        failures.append("timbre at full pedal doesn't match the source file")

    # --- 3: pitch follows RPM (full pedal = worst case for clipping) ---
    print("RPM -> pitch (full pedal):")
    last = 0
    for rpm in (300, 1500, 3000, 5000):
        y, _ = render(lambda p: rpm, lambda p: 1.0, 2.0)
        f = low_peak_hz(y[int(0.7 * sr):], sr)
        print(f"  {rpm:5d} RPM -> {f:6.1f} Hz, peak level {np.abs(y).max():.2f}")
        if f <= last:
            failures.append("pitch does not rise with RPM")
        if np.abs(y).max() > 1.0:
            failures.append(f"clipping at {rpm} RPM")
        last = f

    # --- 4: the load layer audibly enriches the sound at full pedal --------------------
    # Measured against the same render with the load layer muted, so the base hum's own
    # 2nd harmonic doesn't mask what the layer contributes.
    def octave_vs_fundamental_db(y, f0):
        mag = np.abs(np.fft.rfft(y * np.hanning(len(y))))
        f = np.fft.rfftfreq(len(y), 1.0 / sr)
        level = lambda fc: np.max(mag[(f > fc * 0.95) & (f < fc * 1.05)])
        return 20 * np.log10(level(2 * f0) / level(f0))

    def centroid(y):
        mag = np.abs(np.fft.rfft(y * np.hanning(len(y))))
        f = np.fft.rfftfreq(len(y), 1.0 / sr)
        return float(np.sum(f * mag) / np.sum(mag))

    saved_level = main.LOAD_LAYER_LEVEL
    light, _ = render(lambda p: MID_RPM, lambda p: 0.2, 3.0)
    full, _ = render(lambda p: MID_RPM, lambda p: 1.0, 3.0)
    main.LOAD_LAYER_LEVEL = 0.0
    no_load_light, _ = render(lambda p: MID_RPM, lambda p: 0.2, 3.0)
    no_load_full, _ = render(lambda p: MID_RPM, lambda p: 1.0, 3.0)
    main.LOAD_LAYER_LEVEL = saved_level
    light, full = light[int(1.0 * sr):], full[int(1.0 * sr):]
    no_load_light, no_load_full = no_load_light[int(1.0 * sr):], no_load_full[int(1.0 * sr):]
    f0 = low_peak_hz(full, sr)
    gain_full = octave_vs_fundamental_db(full, f0) - octave_vs_fundamental_db(no_load_full, f0)
    gain_light = octave_vs_fundamental_db(light, f0) - octave_vs_fundamental_db(no_load_light, f0)
    c_light, c_full = centroid(light), centroid(full)
    print(f"Load layer @ {MID_RPM:.0f} RPM: adds {gain_light:+.1f} dB at the octave (pedal 0.2), "
          f"{gain_full:+.1f} dB (pedal 1.0); centroid {c_light:.0f} -> {c_full:.0f} Hz")
    if gain_full < 3.0:
        failures.append("load layer adds < 3 dB of octave content at full pedal")
    if c_full <= c_light * 1.2:
        failures.append("full pedal isn't clearly brighter than light pedal")

    # --- 5: current falls with speed, so full pedal at the top isn't the loudest load ---
    print("Load layer vs speed at full pedal:")
    load_at = {}
    for rpm in (300, 3000, MAX_TEST_RPM):
        y, _ = render(lambda p: rpm, lambda p: 1.0, 2.0)
        load_at[rpm] = octave_vs_fundamental_db(y[int(0.7 * sr):], low_peak_hz(y[int(0.7 * sr):], sr))
        print(f"  {rpm:5d} RPM -> {load_at[rpm]:+.1f} dB at the octave")
    if not load_at[300] > load_at[MAX_TEST_RPM] + 1.0:
        failures.append("load layer doesn't relax at high speed for the same pedal")

    # --- 6: pedal drives shaft speed through the powertrain ---------------------------
    pt = main.Powertrain(0.0)
    for _ in range(300):
        pt.step(main.MAX_PEDAL, 0.02)
    cruise = pt.rpm
    print(f"Powertrain: 6 s of full pedal from rest -> {pt.speed:.0f} km/h ({cruise:.0f} RPM)")
    # real-units sim: must get well past the old ~60 km/h cap, at a believable rate
    if not 60.0 < pt.speed < 120.0:
        failures.append(f"6 s of full pedal reached {pt.speed:.0f} km/h (expected 60-120)")
    for _ in range(100):  # 2 s off the pedal
        pt.step(0.0, 0.02)
    print(f"            lift-off for 2 s -> {pt.rpm:.0f} RPM ({20 * np.log10(max(pt.rpm, 1) / cruise):+.1f} dB pitch)")
    # realistic regen (~1.7 m/s^2 plus drag): at least a quarter of the speed in 2 s
    if pt.rpm > 0.75 * cruise:
        failures.append("lift-off doesn't drop shaft speed (no pitch sag)")
    if pt.rpm < 0.0:
        failures.append("shaft speed went negative")

    sweep, _ = render(lambda p: 300 + p * 4700, lambda p: 0.3 + 0.7 * p, 8.0)
    write_wav("synth_sweep.wav", sweep, sr)

    # --- 7: end to end -- a throttle press should make the whine climb, then sag ------
    # 0-1 s idle, 1-6 s full pedal, 6-9 s off the pedal.
    y, sr_d, rpms = render_drive(lambda t: 1.0 if 1.0 <= t < 6.0 else 0.0, 9.0)
    write_wav("synth_drive.wav", y, sr_d)
    win = lambda a, b: y[int(a * sr_d):int(b * sr_d)]
    f_early, f_cruise, f_sag = (low_peak_hz(win(1.2, 1.7), sr_d),
                                low_peak_hz(win(4.0, 5.5), sr_d),
                                low_peak_hz(win(7.5, 8.9), sr_d))
    print(f"Throttle drive: {f_early:.1f} Hz just after the press -> {f_cruise:.1f} Hz "
          f"at {rpms[int(5.0 / (480 / sr_d))]:.0f} RPM -> {f_sag:.1f} Hz 2.5 s after lift-off")
    if f_cruise <= f_early:
        failures.append("throttle doesn't raise pitch")
    # this voice's pitch range tops out at MAX_MOTOR_RPM, below the sim's cruise speed,
    # so the audible sag is smaller than the speed drop; require it to be clearly there
    if f_sag >= f_cruise * 0.97:
        failures.append("pitch doesn't sag after lift-off")
    if np.abs(y).max() > 1.0:
        failures.append("clipping during a throttle press")
    if rpms.min() < 0.0:
        failures.append("shaft speed went negative during the drive")

    # --- 8: the recording's fast flutter is flattened out of the loop ------------------
    def flutter_db(x, sr_):
        k = max(1, int(0.012 * sr_))
        env = np.convolve(np.abs(x), np.ones(k) / k, mode="same")
        env = env / env.mean() - 1.0
        E = np.abs(np.fft.rfft(env * np.hanning(len(env)))) / len(env) * 4
        f = np.fft.rfftfreq(len(env), 1.0 / sr_)
        return 20 * np.log10(np.sqrt(np.sum(E[(f > 4) & (f < 40)] ** 2)) + 1e-12)

    loop_buf = main.build_loop_buffer()
    loop_mono = np.array(loop_buf.data).mean(axis=0)
    raw_mono = ref * 1.0
    f_raw, f_flat = flutter_db(raw_mono, ref_sr), flutter_db(loop_mono, int(loop_buf.sample_rate))
    print(f"Fast flutter (4-40 Hz): source {f_raw:.0f} dB -> loop {f_flat:.0f} dB "
          f"({f_flat - f_raw:+.0f} dB, flatten={main.WOBBLE_FLATTEN})")
    if f_flat > f_raw - 6:
        failures.append("loop still has the recording's fast flutter")

    # --- 9: warble speed is independent of playback rate ------------------------------
    def env_period(y, sr_):
        env = np.abs(y)
        k = int(0.01 * sr_)
        env = np.convolve(env, np.ones(k) / k, mode="same")
        env -= env.mean()
        E = np.abs(np.fft.rfft(env * np.hanning(len(env))))
        f = np.fft.rfftfreq(len(env), 1.0 / sr_)
        band = (f > 0.15) & (f < 3.0)
        return 1.0 / f[band][np.argmax(E[band])]

    slow, _ = render(lambda p: rpm_for_rate(0.65), lambda p: 0.7, 16.0)
    fast, _ = render(lambda p: MAX_TEST_RPM, lambda p: 0.7, 16.0)
    p_slow, p_fast = env_period(slow[int(0.5 * sr):], sr), env_period(fast[int(0.5 * sr):], sr)
    target = 1.0 / main.WOBBLE_HZ
    print(f"Warble period: rate 0.65 -> {p_slow:.2f} s | rate 1.0 -> {p_fast:.2f} s "
          f"| target {target:.2f} s")
    if abs(p_slow - p_fast) > 0.15 * target:
        failures.append("warble period still follows playback rate")
    if abs(p_fast - target) > 0.15 * target:
        failures.append(f"warble period {p_fast:.2f}s doesn't match WOBBLE_HZ ({target:.2f}s)")

    print("Dumped: synth_steady.wav (20 s at original pitch), synth_sweep.wav (RPM + pedal ramp), "
          "synth_drive.wav (throttle press and lift-off)")

    if failures:
        raise SystemExit("FAIL:\n  " + "\n  ".join(failures))
    print("PASS")
