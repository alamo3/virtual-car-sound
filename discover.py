#!/usr/bin/env python3
"""Find fast, short Mode 22 records that carry speed / motor rpm / pedal.

A scan of the non-safety modules (read-only, service 0x22 only) found many records of
8-23 bytes that answer in 10-50 ms, vs 120 ms for the VCU's pedal+speed record. Which of
their bytes (if any) track speed can only be learned while moving, so:

  python discover.py record --seconds 180     # drive: accelerate, cruise, brake, repeat
  python discover.py analyze discover.jsonl   # rank candidate fields against VSS_HD/pedal

`record` cycles through the candidates and reads the VCU reference (E004: pedal + 1/64 km/h
speed) after every few of them, logging raw payloads with timestamps. `analyze` decodes
every byte and 16-bit pair (both endiannesses, signed and unsigned) and correlates it with
the reference interpolated to the same instant.

Safety: never add ABS (7D1), steering (7D4) or ADAS (730) here. Have a passenger run it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

import obd

# (header, DID) -- from the scan; replies are 7-23 bytes. The cluster (7C6) is left out:
# it takes 65-170 ms per read, which is no faster than the VCU itself.
CANDIDATES = [
    # VCU single-frame pedal D + speed via the OBD mirror DIDs: fit pedal D's scale
    # against the E004 pedal (b0 = pedal D raw, b3 = speed km/h).
    (0x7E2, "F449F40D"),
    (0x7E5, "E001"), (0x7E5, "E003"), (0x7E5, "E004"), (0x7E5, "E005"), (0x7E5, "E007"),
    (0x7A0, "B003"), (0x7A0, "B004"), (0x7A0, "B005"), (0x7A0, "B006"), (0x7A0, "B007"),
    (0x7A0, "B008"), (0x7A0, "B009"), (0x7A0, "B00A"), (0x7A0, "C002"), (0x7A0, "C010"),
    (0x7A0, "C011"), (0x7A0, "C012"), (0x7A0, "C013"), (0x7A0, "C014"), (0x7A0, "E002"),
    (0x7A0, "E003"), (0x7A0, "E005"), (0x7A0, "E006"), (0x7A0, "E00A"),
    (0x770, "010B"), (0x770, "010E"), (0x770, "0117"), (0x770, "0118"), (0x770, "0119"),
    (0x770, "011A"), (0x770, "011B"), (0x770, "011C"), (0x770, "011D"), (0x770, "011E"),
    (0x770, "011F"),
    (0x7B3, "0103"),
]
SAFETY = {0x7D0, 0x7D1, 0x7D2, 0x7D4, 0x730}
REF_EVERY = 4          # read the VCU reference after this many candidates


def cmd_record(args):
    assert not any(h in SAFETY for h, _ in CANDIDATES)
    n = 0
    with obd.Elm327(args.port).open() as elm, open(args.out, "w") as f:
        elm.init(fast=True, protocol=6)
        print(f"low-latency USB: {'on' if elm.low_latency else 'OFF'}; "
              f"logging {len(CANDIDATES)} candidates for {args.seconds:.0f} s -> {args.out}")
        print("drive: pull away, accelerate hard and gently, cruise, brake to a stop. Repeat.")
        t0 = time.monotonic()
        i = 0
        while time.monotonic() - t0 < args.seconds:
            if i % REF_EVERY == 0:
                ref = elm.read_hyundai("vcu", "22E004")
                f.write(json.dumps({"t": time.monotonic() - t0, "ref": ref}) + "\n")
                spd = ref.get("IONIQ5_VSS_HD")
                ped = ref.get("IONIQ5_ACCELERATOR_PEDAL_POSITION")
                print(f"\r{time.monotonic() - t0:6.1f} s  speed {spd!s:>8} km/h  "
                      f"pedal {ped!s:>6} %  samples {n}", end="", flush=True)
            hdr, did = CANDIDATES[i % len(CANDIDATES)]
            t = time.perf_counter()
            p = elm.read_mode22(hdr, "22" + did, hdr + 8, timeout=0.3)
            ms = (time.perf_counter() - t) * 1000
            f.write(json.dumps({"t": time.monotonic() - t0, "hdr": f"{hdr:03X}", "did": did,
                                "ms": round(ms, 1), "hex": p.hex().upper()}) + "\n")
            n += 1
            i += 1
    print(f"\n{n} candidate reads written to {args.out}")


def fields(payload: bytes):
    """Every byte and byte-aligned 16-bit pair, as (label, value) in all plausible codings."""
    data = payload[3:]                              # skip '62 <did>'
    for i in range(len(data)):
        yield f"b{i} u8", data[i]
        yield f"b{i} s8", data[i] - 256 if data[i] > 127 else data[i]
    for i in range(len(data) - 1):
        for end in ("little", "big"):
            for signed in (False, True):
                v = int.from_bytes(data[i:i + 2], end, signed=signed)
                yield f"b{i}-{i + 1} {'le' if end == 'little' else 'be'} {'s' if signed else 'u'}16", v


def interp(ts, vs, t):
    """Linear interpolation of the reference at time t (None outside the recorded span)."""
    if not ts or t < ts[0] or t > ts[-1]:
        return None
    lo, hi = 0, len(ts) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ts[mid] <= t:
            lo = mid
        else:
            hi = mid
    if ts[hi] == ts[lo]:
        return vs[lo]
    return vs[lo] + (vs[hi] - vs[lo]) * (t - ts[lo]) / (ts[hi] - ts[lo])


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return 0.0, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy), sxy / sxx      # r, slope (ref units per count)


def cmd_analyze(args):
    refs, cands = [], {}
    for line in open(args.file):
        r = json.loads(line)
        if "ref" in r:
            refs.append(r)
        elif r["hex"]:
            cands.setdefault((r["hdr"], r["did"]), []).append(r)
    targets = {"speed": "IONIQ5_VSS_HD", "pedal": "IONIQ5_ACCELERATOR_PEDAL_POSITION"}
    series = {}
    for name, key in targets.items():
        pts = [(r["t"], r["ref"][key]) for r in refs if key in r["ref"]]
        series[name] = ([t for t, _ in pts], [v for _, v in pts])
        vals = series[name][1]
        print(f"reference {name}: {len(vals)} samples, "
              f"range {min(vals, default=0):.1f}..{max(vals, default=0):.1f}")
    if max(series["speed"][1], default=0) < 10:
        print("\nthe car barely moved in this log -- speed correlation needs a real drive")

    for name in targets:
        ts, vs = series[name]
        hits = []
        for (hdr, did), rows in cands.items():
            ms = sorted(r["ms"] for r in rows)[len(rows) // 2]
            per_field: dict[str, list[tuple[float, float]]] = {}
            for r in rows:
                y = interp(ts, vs, r["t"])
                if y is None:
                    continue
                for label, x in fields(bytes.fromhex(r["hex"])):
                    per_field.setdefault(label, []).append((x, y))
            for label, pairs in per_field.items():
                if len(pairs) < 10:
                    continue
                rr, slope = pearson([p[0] for p in pairs], [p[1] for p in pairs])
                hits.append((abs(rr), rr, slope, hdr, did, label, ms, len(pairs)))
        hits.sort(reverse=True)
        print(f"\nbest matches for {name} (|r| near 1 = same signal; slope = {name} per count):")
        seen = set()
        shown = 0
        for a, rr, slope, hdr, did, label, ms, n in hits:
            key = (hdr, did, label.split()[0])
            if key in seen:
                continue
            seen.add(key)
            print(f"  r={rr:+.3f}  {hdr} 22{did} {label:16s} slope {slope:+.5f}  "
                  f"read {ms:5.1f} ms  n={n}")
            shown += 1
            if shown >= args.top:
                break


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("record")
    p.add_argument("--seconds", type=float, default=180)
    p.add_argument("--out", default="discover.jsonl")
    p.set_defaults(func=cmd_record)
    p = sub.add_parser("analyze")
    p.add_argument("file", nargs="?", default="discover.jsonl")
    p.add_argument("--top", type=int, default=12)
    p.set_defaults(func=cmd_analyze)
    args = ap.parse_args()
    try:
        args.func(args)
    except obd.ObdError as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
