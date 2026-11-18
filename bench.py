"""Measure how much CPU the synth graph costs, so you can check a target board.

Renders offline through the null backend and reports the realtime factor: how many
seconds of audio are produced per second of CPU. Anything above ~3x realtime leaves
comfortable headroom for OBD polling and OS work on a small single-board computer.

Usage: python bench.py [seconds]
"""
import platform
import resource
import sys
import time

import numpy as np
from signalflow import AudioGraph, AudioGraphConfig

import main


def peak_rss_mb():
    # ru_maxrss is KB on Linux, bytes on macOS
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / 1024.0 if platform.system() != "Darwin" else v / (1024.0 * 1024.0)


def bench(duration_sec, block, rpm=3000.0, pedal=0.6):
    cfg = AudioGraphConfig()
    cfg.backend_name = "null"
    graph = AudioGraph(cfg, start=False)
    sr = int(graph.sample_rate)
    output, targets = main.create_ev_sound(main.build_loop_buffer())
    output.play()

    for _ in range(50):  # warm up caches / branch prediction
        main.apply_controls(targets, rpm, pedal)
        graph.render(block)

    total, done = duration_sec * sr, 0
    t0 = time.perf_counter()
    while done < total:
        main.apply_controls(targets, rpm, pedal)
        graph.render(block)
        done += block
    elapsed = time.perf_counter() - t0
    graph.destroy()
    return duration_sec / elapsed, 1000.0 * elapsed / duration_sec, sr


if __name__ == "__main__":
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0

    t0 = time.perf_counter()
    main.build_loop_buffer()
    print(f"loop build (one-off):      {time.perf_counter() - t0:6.3f} s")

    print(f"machine: {platform.machine()} | python {platform.python_version()} | "
          f"{peak_rss_mb():.0f} MB peak RSS")
    print(f"\n{'block':>6}  {'realtime':>9}  {'ms CPU / s audio':>17}")
    for block in (256, 480, 1024):
        rt, ms, sr = bench(dur, block)
        print(f"{block:6d}  {rt:8.1f}x  {ms:17.1f}")
    print(f"\naudible?  rate is the graph's sample rate ({sr} Hz); the 44.1 kHz loop is resampled.")
    print(f"peak RSS after rendering: {peak_rss_mb():.0f} MB")
