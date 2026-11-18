#!/usr/bin/env python3
"""Fast OBD-II polling over an ELM327-compatible adapter (developed against a vLinker FS).

Everything here exists to shave milliseconds off a request/response round trip, because
OBD-II PID polling is latency-bound, not bandwidth-bound:

  1. FTDI latency_timer (16 ms default!)  -- set to 1 ms on open, no sudo needed
  2. ATE0/ATS0/ATH0/ATL0                  -- less to transmit and parse
  3. ATSP6 + ATCFC0                       -- no protocol search, no flow-control frame
  4. ATAT1                                -- let the adapter adapt its own timeout

Usage:
  python obd.py scan                      # find adapters and report permissions
  python obd.py info                      # adapter identity, protocol, latency timer
  python obd.py pids                      # which PIDs this vehicle supports
  python obd.py bench                     # measure round-trip latency and sustained rate
  python obd.py sweep                     # try several adapter configs, keep the fastest
  python obd.py monitor                   # live RPM + pedal
  python obd.py selftest                  # parser/init checks with a fake adapter
"""
from __future__ import annotations

import argparse
import glob
import os
import statistics
import sys
import threading
import time
from dataclasses import dataclass

import serial
from serial.tools import list_ports

# ---------------------------------------------------------------------------------------
# PIDs and decoding
# ---------------------------------------------------------------------------------------

RPM = 0x0C
SPEED = 0x0D
THROTTLE = 0x11          # absolute throttle position, effectively always supported
LOAD = 0x04              # calculated engine load %, also mandated
# Accelerator pedal D/E/F, then absolute/relative throttle. 0x49 is NOT part of the
# legislated PID set, so on a gateway-filtered car it may be absent -- the last two
# fallbacks are emissions-mandated and always answer.
PEDAL_CANDIDATES = (0x49, 0x4A, 0x4B, 0x47, 0x45, THROTTLE, LOAD)


def decode_pid(pid: int, data: bytes):
    """Decode a Mode 01 payload into engineering units (None if too short)."""
    if not data:
        return None
    a = data[0]
    b = data[1] if len(data) > 1 else 0
    if pid == RPM:
        return ((a << 8) | b) / 4.0
    if pid == SPEED:
        return float(a)
    if pid in (LOAD, 0x11, 0x45, 0x47, 0x49, 0x4A, 0x4B, 0x4C):
        return a * 100.0 / 255.0
    return float(a)


# ---------------------------------------------------------------------------------------
# Adapter access
# ---------------------------------------------------------------------------------------

ELM_ERRORS = (
    "NODATA", "STOPPED", "BUFFERFULL", "UNABLETOCONNECT", "CANERROR", "BUSERROR",
    "DATAERROR", "ERROR", "ERR", "?", "FBERROR", "LPERROR", "RXERROR", "ACTALERT",
)


class ObdError(RuntimeError):
    pass


class Elm327:
    """Minimal, low-latency ELM327 driver."""

    def __init__(self, port: str | None = None, baudrate: int = 115200,
                 timeout: float = 0.25, echo: bool = False):
        self.port = port or find_adapter()
        # 115200 is what these adapters expect. Measured on a vLinker FS: 38400 and 500000
        # baud both drop to one reply per 250 ms (garbled -> timeout path), 16x slower.
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser: serial.Serial | None = None
        self.version = ""
        self.config: dict[str, object] = {}
        # Header/receive-address setup is only re-sent when the target ECU changes.
        self._mode22_ctx: int | None = None
        self.low_latency = False
        self.last_did_payload = b""
        self.last_raw = b""
        # After the last frame of a reply the adapter keeps listening for other ECUs until
        # its timeout expires (~96 ms measured). Once the ISO-TP length says the reply is
        # complete, one byte cuts that wait short ("STOPPED"), saving ~64 ms per read.
        # Neither the "22E0041" count suffix nor STPX R:1 can do this: the STN counts
        # single CAN frames and abandons a multi-frame reply after the first one.
        self.early_abort = True
        self.last_t0 = 0.0
        self.last_trace: list[tuple[float, bytes]] = []   # (arrival time, chunk)

    # -- lifecycle ---------------------------------------------------------------------
    def open(self, retries: int = 3, delay: float = 0.4) -> "Elm327":
        if self.ser is not None and self.ser.is_open:
            return self
        if self.port is None:
            raise ObdError("no OBD adapter found (see: python obd.py scan)")
        check_permissions(self.port)
        for attempt in range(retries):
            try:
                self.ser = serial.Serial(
                    self.port, self.baudrate, timeout=0.002, write_timeout=0.5,
                    exclusive=True,  # fail loudly if something else already owns the port
                )
                # The FTDI chip holds short replies for its latency timer (16 ms default)
                # before sending them over USB: dead time on every single line. The
                # ASYNC_LOW_LATENCY flag makes ftdi_sio drop the timer to 1 ms, and unlike
                # writing sysfs it needs no root. Measured: ATI round trip 16.0 -> 3.0 ms.
                try:
                    self.ser.set_low_latency_mode(True)
                    self.low_latency = True
                except (AttributeError, NotImplementedError, OSError, ValueError):
                    self.low_latency = False
                return self
            except serial.SerialException as e:
                # ModemManager / another tool can hold the port for a moment during a probe
                self.ser = None
                if attempt + 1 == retries:
                    raise ObdError(
                        f"could not open {self.port}: {e}\n"
                        f"  If this persists, modem probing is holding it -- stop it with:\n"
                        f"    sudo systemctl stop ModemManager\n"
                        f"  or make it ignore this adapter permanently:\n"
                        f"    echo 'SUBSYSTEM==\"usb\", ATTRS{{idVendor}}==\"0403\", "
                        f"ATTRS{{idProduct}}==\"6015\", ENV{{ID_MM_DEVICE_IGNORE}}=\"1\"' | "
                        f"sudo tee /etc/udev/rules.d/99-vlinker-mm.rules"
                    ) from e
                time.sleep(delay)
        return self

    def close(self):
        if self.ser:
            self.ser.close()
            self.ser = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    # -- transport ---------------------------------------------------------------------
    def _transact(self, payload: str, timeout: float | None = None,
                  done=None) -> list[str]:
        """Send one command and read up to the '>' prompt.

        done: optional predicate on the bytes received so far. When it first returns True,
        the adapter is interrupted so it stops waiting for more replies. Anything written
        while it is still answering is discarded, so the next command still has to wait
        for the prompt.
        """
        if not self.ser:
            raise ObdError("port not open")
        timeout = self.timeout if timeout is None else timeout
        cmd = payload.encode("ascii")
        self.ser.reset_input_buffer()
        trace = []
        self.last_t0 = time.perf_counter()
        self.ser.write(cmd + b"\r")

        buf = bytearray()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            n = self.ser.in_waiting
            chunk = self.ser.read(n or 1)
            if chunk:
                trace.append((time.perf_counter(), chunk))
                buf += chunk
                if b">" in buf:
                    break
                if done is not None and done(buf):
                    # A space is harmless if the prompt beat us to it: the adapter
                    # ignores spaces at the start of the next command.
                    self.ser.write(b" ")
                    done = None
                    deadline = max(deadline, time.monotonic() + 0.2)
        self.last_trace = trace
        self.last_raw = bytes(buf)
        return self._parse(buf, cmd)

    def _parse(self, buf: bytes, cmd: bytes = b"") -> list[str]:
        text = buf.decode("ascii", "replace").replace(">", "\r")
        echo = cmd.decode("ascii").strip().replace(" ", "").upper()
        lines = []
        for raw in text.split("\r"):
            ln = raw.strip().replace(" ", "").upper()
            if not ln or ln == echo:
                continue
            if ln.startswith(("SEARCHING", "BUSINIT", "TRYING")):
                continue
            lines.append(ln)
        return lines

    # -- commands ----------------------------------------------------------------------
    def at(self, cmd: str) -> list[str]:
        return self._transact(cmd if cmd.upper().startswith("AT") else "AT" + cmd)

    def at_ok(self, cmd: str) -> bool:
        """Send an AT command, tolerate '?' for unsupported ones."""
        try:
            return any(r == "OK" for r in self.at(cmd))
        except (serial.SerialException, OSError):
            return False

    def init(self, fast: bool = True, protocol: int | None = 6, adaptive: bool = True,
             flow_control: bool = True, warm: bool = False) -> "Elm327":
        """Bring the adapter up. `fast` = the low-latency settings that don't lose data.

        protocol: 6 = ISO 15765-4 CAN 11-bit/500k (skips auto-search), None = ATSP0 auto.
        adaptive: True = ATAT1 (adapter learns the ECU's response time).
        flow_control: True = ATCFC1. Leave it on: a Mode 22 answer over ~7 bytes is
        multi-frame ISO-TP and the ECU will not send consecutive frames without a flow
        control reply. Turning this off truncates every long response to its first frame.
        """
        self._mode22_ctx = None          # protocol reset invalidates the cached addressing
        banner = self._transact("ATWS" if warm else "ATZ", timeout=1.5)
        self.version = " ".join(banner) or "unknown"
        self.config = {"fast": fast, "protocol": protocol, "adaptive": adaptive,
                       "flow_control": flow_control}

        self.at("E0")                       # echo off
        if fast:
            self.at("L0")                   # no linefeeds
            self.at("S0")                   # no spaces in hex
            self.at("H0")                   # no headers
        self.at("AL")                       # allow long messages
        if protocol is not None:
            self.at(f"SP{protocol}")
        else:
            self.at("SP0")
        if fast:
            self.at("ATAT1" if adaptive else "ATAT0")
        if protocol in (6, 7, 8, 9):
            self.at("CFC1" if flow_control else "CFC0")
            self.at("V0")                   # no variable DLC (CAN)
        return self

    # -- STN extensions (the fast path) -------------------------------------------------
    def stn_identity(self) -> str:
        """The real chip behind the ELM327 persona, e.g. 'STN1170V4.3.2'. '' if not STN."""
        lines = self._transact("STI")
        if lines and lines[0].startswith("STN"):
            self.is_stn = True
            self.stn_version = lines[0]
            return self.stn_version
        return ""

    def stn_setup(self, protocol_timeout_ms: int = 20) -> bool:
        """Tune the STN for low latency. Returns False if this isn't an STN chip."""
        if not self.stn_identity():
            return False
        # not self.at(): that would send "ATSTPTO", which the chip rejects with '?'
        self._transact(f"STPTO {protocol_timeout_ms}")
        return True

    def stpx(self, data: bytes, header: int = 0x7DF, replies: int = 1,
             timeout_ms: int = 20) -> list[bytes]:
        """Send a raw CAN message and collect up to `replies` answers.

        Bypasses the ELM327 text protocol: no PID formatting, explicit response count and
        timeout. `data` is the message payload (e.g. b'\x02\x01\x0c' for RPM).
        """
        cmd = (f"STPX H:{header:03X}, D:{data.hex().upper()}, "
               f"R:{max(1, replies)}, T:{max(1, timeout_ms)}")
        frames = []
        for ln in self._transact(cmd, timeout=timeout_ms / 1000.0 + 0.2):
            if ln in ELM_ERRORS:
                continue
            try:
                frames.append(bytes.fromhex(ln))
            except ValueError:
                continue
        return frames

    def monitor_start(self, filters: tuple[int, ...] = ()):
        """Begin streaming every CAN frame the bus carries (STMA).

        While streaming, the FTDI chip's 16 ms latency timer stops mattering: frames arrive
        faster than the 62-byte USB flush threshold, so there's no per-transaction wait.
        """
        if not self.ser:
            raise ObdError("port not open")
        self.at("H1")            # show CAN IDs
        self.at("S0")
        self.at("L0")
        self.ser.reset_input_buffer()
        for can_id in filters:
            self._transact(f"STFAP {can_id:03X}, 7FF")
        self.ser.write(b"STMA\r")
        self._monitoring = True

    def monitor_read(self, duration: float) -> list[tuple[int, bytes]]:
        """Collect frames for `duration` seconds. Stops the stream when done."""
        if not self.ser:
            raise ObdError("port not open")
        out: list[tuple[int, bytes]] = []
        buf = bytearray()
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            n = self.ser.in_waiting
            if n:
                buf += self.ser.read(n)
                while b"\r" in buf:
                    line, _, buf = buf.partition(b"\r")
                    frame = parse_can_line(line.decode("ascii", "replace"))
                    if frame:
                        out.append(frame)
            else:
                time.sleep(0.001)
        if buf:  # trailing partial line
            frame = parse_can_line(buf.decode("ascii", "replace"))
            if frame:
                out.append(frame)
        self.monitor_stop()
        return out

    def monitor_stop(self):
        if self.ser and getattr(self, "_monitoring", False):
            self.ser.write(b"\r")      # any character stops STMA
            deadline = time.monotonic() + 0.3
            buf = bytearray()
            while b">" not in buf and time.monotonic() < deadline:
                n = self.ser.in_waiting
                buf += self.ser.read(n or 1)
            self._monitoring = False

    # -- Mode 22 / manufacturer reads ---------------------------------------------------
    def read_mode22(self, hdr: int, cmd_hex: str, rax: int | None = None,
                    timeout: float = 0.8) -> bytes:
        """Read a manufacturer DID by physical address; returns the reassembled payload.

        Read-only by construction: send only service 0x22/0x21 requests here. Never send
        session control, routines or anything else to safety modules (ABS 7D1, steering 7D4,
        ADAS 730): probing ABS has lit a warning lamp on this car before.
        """
        if self._mode22_ctx != hdr:
            # The receive filter is global adapter state: set it explicitly on every switch,
            # otherwise one module's CRA silently filters out the next module's replies.
            self.at(f"SH {hdr:03X}")
            self.at(f"CRA {rax:03X}" if rax is not None else "AR")
            self._mode22_ctx = hdr
        payload = self._did_read(cmd_hex, timeout)
        self.last_did_payload = payload
        return payload

    def _did_read(self, cmd_hex: str, timeout: float) -> bytes:
        chunks = []
        declared = 0
        done = did_reply_complete if self.early_abort else None
        for ln in self._transact(cmd_hex, timeout=timeout, done=done):
            if ln.startswith("7F") and ln.endswith("78"):
                continue                    # "response pending": the real answer follows
            if ln in ELM_ERRORS or ln.startswith(("7F", "NO")):
                break
            # Leading 3-hex-digit line = total length the adapter is expecting.
            if not chunks and len(ln) == 3:
                try:
                    declared = int(ln, 16)
                    continue
                except ValueError:
                    pass
            # Long replies are split into segments labelled "0:", "1:", ...
            if ":" in ln:
                ln = ln.split(":", 1)[1]
            try:
                chunks.append(bytes.fromhex(ln))
            except ValueError:
                continue
        payload = reassemble_iso_tp(chunks)
        # The adapter displays whole CAN frames, so the last one drags in 0xAA padding.
        return payload[:declared] if declared and len(payload) > declared else payload

    def read_obd_dids(self, hdr: int, pids: tuple[int, ...], rax: int | None = None,
                      timeout: float = 0.3) -> dict[int, float]:
        """Read legislated PIDs through their UDS mirror DIDs (0xF400 + pid), in one request.

        The IONIQ 5 VCU refuses Mode 01 (7F 01 11) yet answers '22 F449 F40D' with a
        single CAN frame: pedal + speed in ~30 ms, which is the VCU's own response time.
        Up to 2 one-byte PIDs fit in a single frame; a third makes it multi-frame (~60 ms).
        """
        if len(pids) * 2 > 254:
            raise ObdError("too many DIDs in one request")
        cmd = "22" + "".join(f"F4{p:02X}" for p in pids)
        payload = self.read_mode22(hdr, cmd, hdr + 8 if rax is None else rax, timeout)
        return {pid: decode_pid(pid, data) for pid, data in parse_multi_did(payload).items()}

    def read_vcu_fast(self) -> dict[str, float]:
        """Pedal (0-100 %, calibrated) + speed (km/h, whole) from the VCU in one ~30 ms frame.

        Speed confirmed on a drive against VSS_HD: slope 0.9997, r = 0.999.
        """
        hdr, rax = HYUNDAI_ECUS["vcu"]
        payload = self.read_mode22(hdr, "22F449F40D", rax, timeout=0.3)
        got = parse_multi_did(payload)
        out = {}
        if 0x49 in got:
            out["pedal"] = pedal_d_percent(got[0x49][0])
        if SPEED in got:
            out["speed"] = float(got[SPEED][0])
        return out

    def read_hyundai(self, ecu: str, cmd_hex: str) -> dict[str, float]:
        """Read one mode-22 command and decode every signal defined on it."""
        want = {(s["cmd"], s["ecu"]) for s in HYUNDAI_SIGNALS.values()}
        if (cmd_hex, ecu) not in want:
            raise ObdError(f"no signals defined for {ecu}/{cmd_hex}")
        hdr, rax = HYUNDAI_ECUS[ecu]
        payload = self.read_mode22(hdr, cmd_hex, rax)
        out = {}
        for name, spec in HYUNDAI_SIGNALS.items():
            if spec["ecu"] != ecu or spec["cmd"] != cmd_hex:
                continue
            skew = MOTOR_SKEW if spec["ecu"] == "motor" and cmd_hex == "22E001" else 0
            v = decode_hyundai(payload, spec["bix"], spec["length"], spec["div"], skew=skew,
                               signed=spec.get("signed", False),
                               big_endian=spec.get("big", False))
            if v is not None:
                out[name] = v
        return out

    # -- PIDs --------------------------------------------------------------------------

    
    def query(self, pid: int, mode: int = 1) -> tuple[bytes | None, str]:
        """One PID round trip. Returns (data bytes, status)."""
        want = 0x40 + mode
        return self._match("\r".join(self._transact(f"{mode:02X}{pid:02X}")), want, pid)

    def _read_prompts(self, count: int, timeout: float) -> bytes:
        """Read until `count` '>' prompts have arrived (or timeout)."""
        buf = bytearray()
        deadline = time.monotonic() + timeout
        while buf.count(b">") < count and time.monotonic() < deadline:
            n = self.ser.in_waiting
            chunk = self.ser.read(n or 1)
            if chunk:
                buf += chunk
        self.last_raw = bytes(buf)
        return bytes(buf)

    def query_pids(self, pids: tuple[int, ...], mode: int = 1) -> dict[int, tuple[bytes | None, str]]:
        """Poll several PIDs, one round trip each.

        Not pipelined on purpose: an ELM327 buffers a single command line, so writing two
        requests in one USB transfer corrupts the second one (measured: second PID came back
        as '?'). One round trip at a time is the only correct option on this architecture.
        """
        return {pid: self.query(pid, mode=mode) for pid in pids}

    def _match(self, block: str, want: int, pid: int) -> tuple[bytes | None, str]:
        lines = [ln.strip().replace(" ", "").upper() for ln in block.replace("\r", "\n").split("\n")]
        for ln in filter(None, lines):
            if ln in ELM_ERRORS or ln.startswith(("7F", "UNABLE", "BUS")):
                return None, ln
            try:
                raw = bytes.fromhex(ln)
            except ValueError:
                continue
            for j in range(len(raw) - 1):
                if raw[j] == want and raw[j + 1] == pid:
                    return raw[j + 2:], "ok"
        return None, "nodata"

    def supported(self, base: int) -> set[int]:
        """PIDs advertised by the vehicle in the 0x00/0x20/0x40/0x60 bitmask block."""
        data, status = self.query(base)
        if status != "ok" or len(data) < 4:
            return set()
        found = set()
        for i in range(32):
            if data[i // 8] & (1 << (7 - i % 8)):
                found.add(base + 1 + i)
        return found

    def supported_all(self) -> set[int]:
        pids: set[int] = set()
        for base in (0x00, 0x20, 0x40, 0x60):
            block = self.supported(base)
            pids |= block
            if not block:  # vehicle doesn't advertise further blocks
                break
        return pids

    def best_pedal_pid(self) -> int | None:
        """Accelerator pedal if the car exposes it, else throttle plate."""
        pids = self.supported_all()
        for pid in PEDAL_CANDIDATES:
            if pid in pids:
                return pid
        return THROTTLE if (THROTTLE in pids or not pids) else None


def parse_can_line(line: str) -> tuple[int, bytes] | None:
    """Parse one monitor-mode line into (can_id, payload).

    Handles the shapes STN/ELM monitor output can take:
        '7E8 02 41 0C 1A F8'           (id, then payload)
        '7E8 8 02 41 0C 1A F8 00 00'   (id, then DLC, then payload)
        '>7E8 ...'                     (prompt prefix)
    A leading token is treated as the DLC only when the counts match exactly, because the
    first payload byte of an OBD response is also a length byte.
    """
    t = line.strip().lstrip(">").strip()
    if not t or t.upper().replace(" ", "") in ("STOPPED", "BUFFERFULL", "SEARCHING..."):
        return None
    toks = [x for x in t.replace("\t", " ").split() if x]
    if len(toks) < 2:
        return None
    try:
        can_id = int(toks[0], 16)
    except ValueError:
        return None
    data = toks[1:]
    if len(data[0]) <= 2:
        try:
            dlc = int(data[0], 16)
        except ValueError:
            dlc = None
        if dlc is not None and 0 <= dlc <= 8 and len(data) - 1 == dlc:
            data = data[1:]
    payload = bytearray()
    for tok in data:
        try:
            payload += bytes.fromhex(tok if len(tok) % 2 == 0 else "0" + tok)
        except ValueError:
            break
    return (can_id, bytes(payload)) if payload else None


# ---------------------------------------------------------------------------------------
# Manufacturer (Mode 22) signals
# ---------------------------------------------------------------------------------------
# Hyundai E-GMP (IONIQ 5 / EV6 / GV60) keeps the interesting values off the legislated
# 0x7DF broadcast entirely -- there is no ECM at 0x7E0. They come back as Mode 22 reads
# to physically-addressed modules, as multi-frame ISO-TP responses.
#
# Layout, from OBDb's IONIQ 5 signalset: the payload is '62 <did_hi> <did_lo>' followed by
# the data block and the bit index counts from the start of the data block. Byte order is
# per module: VCU and motor are little-endian, BMS is big-endian.
#
# How fast each one is on this car (STN1170, low-latency USB, early abort):
#   vcu   22F449F40D 7 B  ~30 ms   pedal D + VSS (1 km/h)  single frame, read_obd_dids()
#   hvac  220100   39 B   ~51 ms   VSS, whole km/h       (frames back to back)
#   bms   220101   62 B   ~95 ms   drive motor rpm       (frames back to back)
#   vcu   22E004   34 B  ~121 ms   pedal + VSS_HD        (ECU paces frames ~20 ms)
#   motor 22E001  203 B  ~620 ms   rpm, torque, temps    (ECU paces frames ~20 ms)
# Mode 01 is refused (7F 01 11), but the VCU answers the same PIDs as UDS DIDs 0xF4xx.

HYUNDAI_ECUS = {
    "motor": (0x7E3, 0x7EB),   # inverter / motor control
    "vcu": (0x7E2, 0x7EA),     # vehicle control unit
    "bms": (0x7E4, 0x7EC),     # battery management
    "hvac": (0x7B3, 0x7BB),    # climate control (gets road speed for blower scheduling)
}

HYUNDAI_SIGNALS = {
    "IONIQ5_MOTOR_RPM": dict(
        ecu="motor", cmd="22E001", bix=136, length=16, div=1, signed=True, unit="rpm"),
    "IONIQ5_MOTOR_ACTUAL_TORQUE": dict(
        ecu="motor", cmd="22E001", bix=168, length=16, div=10, signed=True, unit="Nm"),
    "IONIQ5_NOMINAL_TORQUE_VALUE": dict(
        ecu="motor", cmd="22E001", bix=152, length=16, div=10, signed=True, unit="Nm"),
    "IONIQ5_ACCELERATOR_PEDAL_POSITION": dict(
        ecu="vcu", cmd="22E004", bix=64, length=16, div=512, unit="%"),
    # OBDb scale is 156.25/10000 = 1/64 km/h per bit
    "IONIQ5_VSS_HD": dict(
        ecu="vcu", cmd="22E004", bix=80, length=16, div=64, signed=True, unit="km/h"),
    "IONIQ5_DRIVE_MOTOR_SPEED_1": dict(
        ecu="bms", cmd="220101", bix=424, length=16, div=1, signed=True, big=True,
        unit="rpm"),
    "IONIQ5_DRIVE_MOTOR_SPEED_2": dict(
        ecu="bms", cmd="220101", bix=440, length=16, div=1, signed=True, big=True,
        unit="rpm"),
    "IONIQ5_HVBAT_SOC": dict(                       # plausibility witness for the BMS layout
        ecu="bms", cmd="220101", bix=32, length=8, div=2, unit="%"),
    "IONIQ5_BATTERY_CURRENT": dict(
        ecu="bms", cmd="220101", bix=80, length=16, div=10, signed=True, big=True,
        unit="A"),
    "IONIQ5_VSS": dict(
        ecu="hvac", cmd="220100", bix=232, length=8, div=1, unit="km/h"),
    # extra E001 fields, useful as the plausibility witnesses during calibration
    "IONIQ5_IGN_VOLTAGE": dict(
        ecu="motor", cmd="22E001", bix=120, length=16, div=1000, unit="V"),
    "IONIQ5_MOTOR_T": dict(
        ecu="motor", cmd="22E001", bix=200, length=16, div=100, unit="C"),
    "IONIQ5_INVERTER_T": dict(
        ecu="motor", cmd="22E001", bix=216, length=16, div=100, unit="C"),
    "IONIQ5_MOTOR_PHASE_CURRENT_RMS": dict(
        ecu="motor", cmd="22E001", bix=184, length=16, div=10, unit="A"),
}

# Pedal D through the OBD mirror DID (22F449, raw 0-255) mapped onto the VCU's own 0-100 %
# pedal (22E004). Fitted on a drive: r = 0.9987, max error 1.1 % with the pedal held. Raw
# reads 38 at rest and the VCU reports 0 % until ~43.6 (a small dead band). The drive only
# reached ~80 %, so the top end (100 % at raw ~197) is extrapolated.
PEDAL_D_ZERO_RAW = 43.6
PEDAL_D_PCT_PER_COUNT = 0.6504


def pedal_d_percent(raw: int) -> float:
    """Raw pedal D byte -> the VCU's 0-100 % accelerator position."""
    return min(100.0, max(0.0, (raw - PEDAL_D_ZERO_RAW) * PEDAL_D_PCT_PER_COUNT))


MOTOR_SKEW = -1  # this car's E001 block sits one byte earlier than OBDb's 2024 capture


def did_reply_complete(buf: bytes) -> bool:
    """True once a Mode 22 reply is complete, so the adapter's timeout wait can be cut.

    Single frame: one line starting '62' (or a final '7F' negative response).
    Multi frame: '022' (total length, hex) then '0:...', '1:...' segments, complete once
    the declared byte count has arrived. Only CR-terminated lines count, so a half-received
    line is never mistaken for the end. '7F xx 78' (response pending) is not an answer.
    """
    lines = [ln.strip().replace(b" ", b"") for ln in bytes(buf).split(b"\r")[:-1]]
    lines = [ln for ln in lines if ln and not (ln.startswith(b"7F") and ln.endswith(b"78"))]
    if not lines:
        return False
    first = lines[0]
    if len(first) != 3:
        return first.startswith((b"62", b"7F"))
    try:
        declared = int(first, 16)
    except ValueError:
        return False
    got = sum(len(ln.split(b":", 1)[1]) // 2 for ln in lines[1:] if b":" in ln)
    return got >= declared


# Data length of the legislated PIDs we read through their UDS mirror (DID 0xF4xx).
OBD_PID_LEN = {0x04: 1, 0x0C: 2, 0x0D: 1, 0x11: 1, 0x1C: 1, 0x1F: 2, 0x31: 2, 0x45: 1,
               0x47: 1, 0x49: 1, 0x4A: 1, 0x4B: 1, 0x5B: 1}


def parse_multi_did(payload: bytes) -> dict[int, bytes]:
    """Split '62 F449 26 F40D 00' into {0x49: b'\x26', 0x0D: b'\x00'}.

    The ECU silently leaves out DIDs it doesn't support, so the reply is walked
    DID by DID using the known data length of each one.
    """
    out: dict[int, bytes] = {}
    if not payload or payload[0] != 0x62:
        return out
    i = 1
    while i + 2 <= len(payload):
        hi, pid = payload[i], payload[i + 1]
        n = OBD_PID_LEN.get(pid)
        if hi != 0xF4 or n is None or i + 2 + n > len(payload):
            break
        out[pid] = payload[i + 2:i + 2 + n]
        i += 2 + n
    return out


def reassemble_iso_tp(chunks: list[bytes]) -> bytes:
    """Join ISO-TP frames, or pass through a payload the adapter already reassembled."""
    if not chunks:
        return b""
    if chunks[0] and chunks[0][0] == 0x62:
        return b"".join(chunks)
    first = chunks[0]
    total = ((first[0] & 0x0F) << 8) | first[1]
    out = bytearray(first[2:])
    for c in chunks[1:]:
        if c and (c[0] >> 4) == 2:          # consecutive frame
            out += c[1:]
    return bytes(out[:total]) if total else bytes(out)


def decode_hyundai(payload: bytes, bix: int, length: int = 16, div: float = 1.0,
                   offset: float = 0.0, skew: int = 0, signed: bool = False,
                   big_endian: bool = False) -> float | None:
    """Extract a byte-aligned field at `bix` bits into the data block.

    `skew` shifts the whole block by n bytes. ECU software revisions don't always agree on
    where the block starts; `obd.py calibrate` picks it by plausibility instead of guessing.
    """
    start = 3 + skew + bix // 8               # +3 skips the '62 <did>' echo
    n = max(1, length // 8)
    if len(payload) < start + n:
        return None
    raw = int.from_bytes(payload[start:start + n], "big" if big_endian else "little",
                         signed=signed)
    return raw / div + offset


# ---------------------------------------------------------------------------------------
# Port discovery / diagnostics
# ---------------------------------------------------------------------------------------

def find_adapter() -> str | None:
    """Locate a vLinker/ELM-style USB serial adapter."""
    for p in list_ports.comports():
        blob = " ".join(str(x) for x in (p.product, p.description, p.manufacturer, p.hwid)
                        if x).lower()
        if (p.vid, p.pid) in ((0x0403, 0x6015), (0x0403, 0x6001), (0x067B, 0x2303)) \
                or "vlinker" in blob or "obd" in blob or "elm327" in blob:
            return p.device
    for pattern in ("/dev/ttyUSB*", "/dev/ttyACM*"):
        found = sorted(glob.glob(pattern))
        if found:
            return found[0]
    return None


def check_permissions(port: str):
    if not os.access(port, os.R_OK | os.W_OK):
        raise ObdError(
            f"{port} is not readable by this user. Pick one:\n"
            f"    sudo chmod 666 {port}\n"
            f"    sudo usermod -aG dialout $USER    (then log out and back in)\n"
            f"  or a permanent udev rule:\n"
            f"    echo 'SUBSYSTEM==\"tty\", ATTRS{{idVendor}}==\"0403\", "
            f"ATTRS{{idProduct}}==\"6015\", MODE=\"0666\"' | "
            f"sudo tee /etc/udev/rules.d/99-vlinker.rules"
        )


def latency_timer_path(port: str = "/dev/ttyUSB0") -> str:
    return f"/sys/bus/usb-serial/devices/{os.path.basename(port)}/latency_timer"


def latency_timer(port: str) -> int | None:
    try:
        with open(latency_timer_path(port)) as f:
            return int(f.read().strip())
    except OSError:
        return None


def latency_timer_hint(port: str) -> str:
    ms = latency_timer(port)
    if ms is None:
        return "latency_timer: not found (not an FTDI-family device?)"
    if ms <= 2:
        return (f"latency_timer: sysfs says {ms} ms -- clones can lie about this; "
                f"`obd.py info` measures the real round trip")
    return (f"latency_timer: {ms} ms while closed -- Elm327.open() drops it to 1 ms "
            f"(ASYNC_LOW_LATENCY), no sudo needed")


# ---------------------------------------------------------------------------------------
# Polling / measurement
# ---------------------------------------------------------------------------------------

@dataclass
class Stats:
    n: int
    ok: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    cycle_hz: float
    cycle_ms: float = 0.0

    def __str__(self):
        return (f"{self.ok}/{self.n} ok | per-PID median {self.median_ms:5.1f} ms "
                f"(p95 {self.p95_ms:5.1f}) | cycle {self.cycle_ms:5.1f} ms | "
                f"{self.cycle_hz:5.1f} cycles/s")


def measure(elm: Elm327, pids=tuple, cycles: int = 100) -> Stats:
    """Time whole poll cycles; per-PID latency is the cycle divided across the PID set."""
    cyc, ok = [], 0
    t0 = time.perf_counter()
    for _ in range(cycles):
        a = time.perf_counter()
        for pid in pids:
            _, st = elm.query(pid)
            ok += 1 if st == "ok" else 0
        cyc.append((time.perf_counter() - a) * 1000.0)
    elapsed = time.perf_counter() - t0
    per_pid = sorted(c / max(len(pids), 1) for c in cyc)
    cyc.sort()
    return Stats(
        n=len(cyc) * len(pids), ok=ok,
        mean_ms=statistics.fmean(per_pid) if per_pid else 0.0,
        median_ms=per_pid[len(per_pid) // 2] if per_pid else 0.0,
        p95_ms=per_pid[int(len(per_pid) * 0.95) - 1] if per_pid else 0.0,
        cycle_hz=cycles / elapsed if elapsed else 0.0,
        cycle_ms=cyc[len(cyc) // 2] if cyc else 0.0,
    )


class ObdReader(threading.Thread):
    """Polls a small PID set in the background and keeps the latest values.

    Plain attribute assignment is atomic under the GIL, so the audio/control side can read
    `reader.rpm` / `reader.pedal` without locking. `stale_after` lets the caller notice a
    dropped link and fall back to a model.
    """

    def __init__(self, elm: Elm327, pids: tuple[int, ...] = (RPM, THROTTLE),
                 stale_after: float = 0.5, pedal_every: int = 2):
        super().__init__(daemon=True, name="obd-reader")
        self.elm = elm
        self.pids = tuple(pids)
        self.stale_after = stale_after
        # RPM drives pitch, so it gets every slot; slower-moving values get every Nth slot.
        self.pedal_every = max(1, pedal_every)
        self._cycle = 0
        self.rpm: float | None = None
        self.pedal: float | None = None
        self.values: dict[int, float | None] = {p: None for p in self.pids}
        self.updated_at = 0.0
        self.rate_hz = 0.0
        self.errors = 0
        self._stop = threading.Event()

    @property
    def fresh(self) -> bool:
        return (time.monotonic() - self.updated_at) < self.stale_after

    def _due(self) -> tuple[int, ...]:
        """Which PIDs to poll this cycle: RPM always, the rest on a slower rotation."""
        due = [p for p in self.pids if p == RPM]
        others = [p for p in self.pids if p != RPM]
        if others and self._cycle % self.pedal_every == 0:
            due.extend(others)
        self._cycle += 1
        return tuple(due)

    def poll_once(self) -> dict[int, tuple[bytes | None, str]]:
        results = self.elm.query_pids(self._due())
        for pid, (data, status) in results.items():
            if status != "ok":
                self.errors += 1
                continue
            value = decode_pid(pid, data)
            self.values[pid] = value
            if pid == RPM:
                self.rpm = value
            elif pid != SPEED:
                self.pedal = value
            self.updated_at = time.monotonic()
        return results

    def run(self):
        count, t0 = 0, time.perf_counter()
        while not self._stop.is_set():
            self.poll_once()
            count += 1
            el = time.perf_counter() - t0
            if el >= 1.0:
                self.rate_hz, count, t0 = count / el, 0, time.perf_counter()

    def stop(self):
        self._stop.set()


class IoniqReader(threading.Thread):
    """Background poller for the fast E-GMP sources; the audio side reads the attributes.

    Every cycle: VCU pedal (0-100 %) + speed (whole km/h) in one ~30 ms frame.
    Every `rpm_every`-th cycle: the battery-module block (~95 ms). From it: motor rpm (to
    calibrate rpm-per-km/h), its SIGN (reverse), battery current (regen) and SOC.
    Plain attribute writes are atomic under the GIL, so no locking is needed.
    """

    def __init__(self, elm: Elm327, rpm_every: int = 4, stale_after: float = 0.5):
        super().__init__(daemon=True, name="ioniq-reader")
        self.elm = elm
        self.rpm_every = rpm_every
        self.stale_after = stale_after
        self.pedal: float | None = None
        self.speed: float | None = None
        self.motor_rpm: float | None = None       # magnitude, for the speed ratio
        self.motor_rpm_signed: float | None = None   # sign shows direction (reverse)
        self.battery_current: float | None = None    # A, signed (regen vs drive)
        self.soc: float | None = None
        self.updated_at = 0.0
        self.rpm_at = 0.0
        self.rate_hz = 0.0
        self.errors = 0
        self.last_error = ""
        self._cycle = 0
        self._stop = threading.Event()

    @property
    def fresh(self) -> bool:
        return (time.monotonic() - self.updated_at) < self.stale_after

    def poll_once(self):
        got = self.elm.read_vcu_fast()
        if "pedal" in got and "speed" in got:
            self.speed = got["speed"]
            self.pedal = got["pedal"]
            self.updated_at = time.monotonic()
        else:
            self.errors += 1
        if self.rpm_every and self._cycle % self.rpm_every == self.rpm_every - 1:
            vals = self.elm.read_hyundai("bms", "220101")
            # SPEED_1/2 are the two motor slots; a single-motor car leaves one at 0.
            rpms = [vals[k] for k in ("IONIQ5_DRIVE_MOTOR_SPEED_1",
                                      "IONIQ5_DRIVE_MOTOR_SPEED_2") if k in vals]
            if rpms:
                signed = max(rpms, key=abs)
                self.motor_rpm_signed = signed
                self.motor_rpm = abs(signed)
                self.battery_current = vals.get("IONIQ5_BATTERY_CURRENT")
                self.soc = vals.get("IONIQ5_HVBAT_SOC")
                self.rpm_at = time.monotonic()
        self._cycle += 1

    def run(self):
        count, t0 = 0, time.perf_counter()
        while not self._stop.is_set():
            try:
                self.poll_once()
            except (ObdError, serial.SerialException, OSError) as e:
                self.errors += 1
                self.last_error = str(e)
                time.sleep(0.1)
            count += 1
            el = time.perf_counter() - t0
            if el >= 1.0:
                self.rate_hz, count, t0 = count / el, 0, time.perf_counter()

    def stop(self):
        self._stop.set()


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------

def cmd_scan(_args):
    ports = [p for p in list_ports.comports()
             if not p.device.startswith(("/dev/ttyS", "/dev/ttyAMA"))]
    if not ports:
        print("no serial ports found")
    for p in ports:
        access = "rw" if os.access(p.device, os.R_OK | os.W_OK) else "NO ACCESS"
        ids = f"{p.vid:04X}:{p.pid:04X}" if p.vid is not None else "    :    "
        print(f"  {p.device:16s} {ids} {(p.product or p.description or ''):24s} [{access}]")
    port = find_adapter()
    print(f"\nbest candidate: {port or 'none'}")
    if port:
        print(latency_timer_hint(port))
        try:
            check_permissions(port)
            print("permissions: OK")
        except ObdError as e:
            print(f"permissions: {e}")


def cmd_info(args):
    with Elm327(args.port, args.baud).open() as elm:
        elm.init(fast=True, protocol=args.protocol, adaptive=True, flow_control=False)
        print(f"adapter:  {elm.port} @ {elm.baudrate}")
        print(f"version:  {elm.version}")
        ms = [0.0] * 20
        for i in range(20):
            a = time.perf_counter()
            elm._transact("ATI")
            ms[i] = (time.perf_counter() - a) * 1000.0
        eff = statistics.median(ms)
        claimed = latency_timer(elm.port)
        print(f"transport: {eff:.2f} ms per round trip (median) -- sysfs claims {claimed} ms")
        if claimed is not None and eff > claimed + 4:
            print("  note: round trip is well above the latency timer -- check the adapter")
        us, st = elm.query(0x00)
        print(f"supported PID bitmap (0100): {st}")
        pids = sorted(elm.supported_all())
        print(f"vehicle supports {len(pids)} PIDs: "
              f"{' '.join(f'{p:02X}' for p in pids[:32])}{' ...' if len(pids) > 32 else ''}")
        print(f"pedal PID choice: {_pid_name(elm.best_pedal_pid())}")


def _pid_name(pid):
    if pid is None:
        return "none found (car may only expose throttle)"
    return {RPM: "0x0C RPM", THROTTLE: "0x11 throttle plate", LOAD: "0x04 calculated load",
            0x45: "0x45 relative throttle",
            0x47: "0x47 absolute throttle B", 0x49: "0x49 pedal D",
            0x4A: "0x4A pedal E", 0x4B: "0x4B pedal F"}.get(pid, f"0x{pid:02X}")


def cmd_pids(args):
    with Elm327(args.port, args.baud).open() as elm:
        elm.init(protocol=args.protocol)
        pids = sorted(elm.supported_all())
        print(f"{len(pids)} supported PIDs: {' '.join(f'{p:02X}' for p in pids)}")
        for pid in (RPM, THROTTLE, *PEDAL_CANDIDATES):
            data, st = elm.query(pid)
            val = decode_pid(pid, data) if st == "ok" else None
            print(f"  {pid:02X} {_pid_name(pid):26s} {st:8s} {'' if val is None else f'{val:8.2f}'}")


def _pids_from(args):
    pids = tuple(int(x, 16) for x in args.pids.split(","))
    return pids


def cmd_bench(args):
    with Elm327(args.port, args.baud).open() as elm:
        elm.init(fast=True, protocol=args.protocol, adaptive=True, flow_control=False)
        ms = [0.0] * 20
        for i in range(20):
            a = time.perf_counter()
            elm._transact("ATI")
            ms[i] = (time.perf_counter() - a) * 1000.0
        eff = statistics.median(ms)
        claimed = latency_timer(elm.port)
        print(f"transport: {eff:.2f} ms per round trip (median) -- sysfs claims {claimed} ms")
        if claimed is not None and eff > claimed + 4:
            print("  note: round trip is well above the latency timer -- check the adapter")
        pedal = elm.best_pedal_pid() or THROTTLE
        for label, pids in (("RPM only", (RPM,)), ("RPM + pedal", (RPM, pedal))):
            st = measure(elm, pids, cycles=args.cycles)
            print(f"{label:12s} {st}")
            if not st.ok:
                print("             (no ECU answering -- transport time only)")


def cmd_sweep(args):
    """Try the knobs that matter and report which combination is fastest here."""
    configs = [
        ("ATAT1 CFC0 SP6 (default fast)", dict(adaptive=True, flow_control=False, protocol=6)),
        ("ATAT1 CFC1 SP6", dict(adaptive=True, flow_control=True, protocol=6)),
        ("ATAT0 CFC0 SP6", dict(adaptive=False, flow_control=False, protocol=6)),
        ("ATAT1 CFC0 SP0 (auto)", dict(adaptive=True, flow_control=False, protocol=None)),
    ]
    results = []
    for label, kw in configs:
        with Elm327(args.port, args.baud).open() as elm:
            elm.init(fast=True, **kw)
            pedal = elm.best_pedal_pid() or THROTTLE
            st = measure(elm, (RPM, pedal), cycles=args.cycles)
            results.append((st.cycle_hz, label, st))
            print(f"  {st.cycle_hz:5.1f} cyc/s  {label:30s} {st}")
    best = max(results)
    print(f"\nfastest: {best[1]}  ({best[0]:.1f} cycles/s)")


def cmd_sniff(args):
    """Stream the whole bus (STMA) and summarise which frames carry changing data.

    Polling can't beat one round trip per PID; streaming has no round trips at all. This is
    the tool for finding the vehicle's own RPM frame so the sound can follow it directly.
    """
    with Elm327(args.port, args.baud).open() as elm:
        elm.init(fast=True, protocol=args.protocol)
        if not elm.stn_setup(protocol_timeout_ms=args.proto_timeout):
            print("warning: not an STN chip, monitor mode may be unavailable")
        filters = tuple(int(x, 16) for x in args.filter.split(",")) if args.filter else ()
        print(f"streaming {args.seconds:.0f} s"
              f"{' (filters: ' + args.filter + ')' if filters else ''} ...")
        frames = elm.monitor_read(args.seconds)
        elm.init(fast=True, protocol=args.protocol)

    if not frames:
        print("no frames received -- ignition on and adapter on the OBD port?")
        return

    from collections import defaultdict
    stats = defaultdict(list)
    for can_id, data in frames:
        stats[can_id].append(data)

    print(f"\n{len(frames)} frames from {len(stats)} IDs in {args.seconds:.0f} s\n")
    print(f"{'CAN ID':8s} {'Hz':>7s} {'DLC':>3s}  bytes that change (min..max)")
    for can_id, rows in sorted(stats.items(), key=lambda kv: -len(kv[1]))[:args.top]:
        n = len(rows)
        dlc = max(len(r) for r in rows)
        varying = []
        for i in range(dlc):
            vals = [r[i] for r in rows if len(r) > i]
            if len(set(vals)) > 1:
                varying.append(f"[{i}] {min(vals):02X}-{max(vals):02X} ({len(set(vals))} vals)")
        if not varying:
            varying = ["(constant)"]
        print(f"0x{can_id:03X}   {n / args.seconds:7.1f} {dlc:3d}  {' '.join(varying[:4])}")

    # Interpret the capture: legislated OBD is a legal guarantee, everything else isn't.
    legislated = {i for i in stats if 0x7E8 <= i <= 0x7EF or i == 0x7DF}
    other = {i: v for i, v in stats.items() if i not in legislated}
    hotspot = [i for i, v in other.items()
               if len(v) / args.seconds > 20 and len({tuple(r[:2]) for r in v}) > 1]
    print()
    if not other:
        print("VERDICT: only legislated OBD traffic visible (0x7DF / 0x7E8-0x7EF).")
        print("  This port is filtered, almost certainly by a security gateway. Streaming gets")
        print("  you nothing here -- poll the legislated PIDs instead (that path is mandated).")
    else:
        print(f"VERDICT: {len(other)} non-legislated IDs visible -> this port is NOT filtered")
        print("  to legislated traffic only. Look for a frame carrying RPM." +
              (f" Fast candidates (>20 Hz, changing): "
               f"{' '.join(f'0x{i:03X}' for i in hotspot[:6])}" if hotspot else ""))
        print("  Blip the throttle and re-run: the RPM bytes will move with it.")
    if args.save:
        with open(args.save, "w") as f:
            f.write("t_s,can_id,data\n")
            step = 1.0 / max(len(frames) / args.seconds, 1)
            for i, (can_id, data) in enumerate(frames):
                f.write(f"{i * step:.4f},{can_id:03X},{data.hex().upper()}\n")
        print(f"\nraw frames written to {args.save}")
    print("\nnext: find the ID whose varying bytes track RPM, then decode it. "
          "RPM usually sits in 2 adjacent bytes and scales linearly with engine speed.")


def cmd_record(args):
    """Log RPM + speed to CSV and summarise the range -- the calibration run for the synth.

    Prints percentiles and checks that motor RPM tracks road speed linearly, which is what
    a single-speed EV drivetrain must do (and what lets the sound be mapped from either).
    """
    pids = tuple(int(x, 16) for x in args.pids.split(","))
    rows = []
    with Elm327(args.port, args.baud).open() as elm:
        elm.init(fast=True, protocol=args.protocol)
        pedal = elm.best_pedal_pid()
        print(f"logging {', '.join(f'0x{p:02X}' for p in pids)} for {args.seconds:.0f} s "
              f"(pedal source: {_pid_name(pedal)}) -- drive now")
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.seconds:
            t = time.monotonic() - t0
            vals = {}
            for pid in pids:
                data, status = elm.query(pid)
                vals[pid] = decode_pid(pid, data) if status == "ok" else None
            if any(v is not None for v in vals.values()):
                rows.append((t, vals))

    got = [r for r in rows if any(v is not None for v in r[1].values())]
    if not got:
        print("\nno decodable answers. If the car is on and awake, this port is probably not")
        print("speaking classic CAN 500k -- try --protocol 0, then a CAN-FD capable interface.")
        return

    print(f"\n{len(got)} samples over {args.seconds:.0f} s "
          f"({len(got)/args.seconds:.1f} cycles/s)")
    for pid in pids:
        vals = [r[1][pid] for r in got if r[1][pid] is not None]
        if not vals:
            print(f"  0x{pid:02X} {_pid_name(pid):22s} -> no data")
            continue
        vals.sort()
        print(f"  0x{pid:02X} {_pid_name(pid):22s} min {vals[0]:8.1f}  "
              f"p50 {vals[len(vals)//2]:8.1f}  max {vals[-1]:8.1f}  n={len(vals)}")

    ratios = [r[1][RPM] / r[1][SPEED] for r in got
              if r[1].get(RPM) and r[1].get(SPEED) and r[1][SPEED] > 5]
    if ratios:
        ratios.sort()
        mid = ratios[len(ratios)//2]
        spread = (ratios[-1] - ratios[0]) / mid * 100
        print(f"\nRPM per km/h: median {mid:.1f} (spread {spread:+.1f}% across the run)")
        if spread < 8:
            print("  -> tracks speed linearly: fixed reduction gear, as expected for an EV.")
        else:
            print("  -> not proportional; likely gear/slip or a different scaling.")

    if args.save:
        with open(args.save, "w") as f:
            f.write("t_s," + ",".join(f"pid_{p:02X}" for p in pids) + "\n")
            for t, vals in got:
                f.write(f"{t:.3f}," + ",".join("" if vals[p] is None else f"{vals[p]:.2f}"
                                               for p in pids) + "\n")
        print(f"written: {args.save}")


def cmd_diag(args):
    """Probe the link layer by layer and say where it stops."""
    with Elm327(args.port, args.baud).open() as elm:
        elm.init(fast=True, protocol=args.protocol)
        chip = elm.stn_identity()
        print(f"adapter : {elm.version}" + (f"  chip {chip}" if chip else ""))
        print(f"protocol: {elm._transact('ATDPN')} {elm._transact('ATDP')}")

        data, status = elm.query(0x00)
        print(f"mode 01 : 0100 -> {status}"
              f"{'  (legislated OBD answers)' if data else '  (no answer)'}")

        for label, hdr, cmd in (("motor", 0x7E3, "22E001"), ("vcu", 0x7E2, "22E004")):
            for mode in ("header only", "with CAF1+CRA"):
                elm._mode22_ctx = None
                elm.at(f"SH {hdr:03X}")
                if mode == "with CAF1+CRA":
                    elm.at("CAF1")
                    elm.at(f"CRA {0x7EA if hdr == 0x7E2 else 0x7EB:03X}")
                payload = elm._did_read(cmd, 0.8)
                print(f"mode 22 : {label:5s} {cmd} to 0x{hdr:03X} [{mode:13s}] -> "
                      f"{len(payload):3d} B {payload[:16].hex().upper()}")
        elm._mode22_ctx = None

        print()
        mode22_ok = False   # recomputed below for the verdict
        print("verdict :", end=" ")
        ok01 = elm.query(0x00)[1] == "ok"
        if not ok01:
            print("nothing answers at all.")
            print("  -> modules asleep (car not in READY?), or the port isn't classic CAN.")
        else:
            print("legislated mode 01 answers, so the bus is alive.")
            print("  -> if mode 22 above returned nothing, the gateway is filtering it.")
        print("  next   : put the car in READY (brake + start, READY lamp on), then re-run.")
        print("           motor RPM stays 0 in Park -- it needs the car to be moving.")


def cmd_calibrate(args):
    """Pick the byte skew whose decodes are physically plausible -- no guessing."""
    with Elm327(args.port, args.baud).open() as elm:
        elm.init(fast=True, protocol=args.protocol)
        payload = elm.read_mode22(0x7E3, "22E001", 0x7EB)
    if not payload:
        print("no data from 0x7E3 / 22 E001")
        return
    print(f"E001 payload: {len(payload)} B  {payload[:24].hex().upper()} ...\n")
    print(f"{'skew':>4}  {'V':>8}  {'motor T':>8}  {'inv T':>8}  {'torque':>8}  plausible")
    best = None
    for skew in (-1, 0, 1, 2):
        v = decode_hyundai(payload, 120, div=1000, skew=skew)
        mt = decode_hyundai(payload, 200, div=100, skew=skew)
        it = decode_hyundai(payload, 216, div=100, skew=skew)
        tq = decode_hyundai(payload, 168, div=10, skew=skew)
        ok = (v is not None and 9.0 <= v <= 16.5 and
              mt is not None and -40 <= mt <= 120 and
              it is not None and -40 <= it <= 120 and
              tq is not None and abs(tq) <= 2000)
        print(f"{skew:>4}  {v!s:>8}  {mt!s:>8}  {it!s:>8}  {tq!s:>8}  {'YES' if ok else '-'}")
        if ok and best is None:
            best = skew
    print(f"\nbest skew: {best if best is not None else 'none (module may not be in READY)'}")
    if best not in (None, MOTOR_SKEW):
        print(f"set MOTOR_SKEW = {best} in obd.py")


def cmd_ioniq(args):
    """Read motor RPM / torque and accelerator pedal from an E-GMP car (IONIQ 5 family)."""
    with Elm327(args.port, args.baud).open() as elm:
        elm.init(fast=True, protocol=args.protocol)
        chip = elm.stn_identity()
        print(f"adapter: {elm.version}" + (f"  (chip {chip})" if chip else ""))
        print("press ctrl-c to stop\n")
        counts = {"motor": 0, "vcu": 0}
        got_any = False
        motor: dict[str, float] = {}
        motor_payload = vcu_payload = b""
        cycle = 0
        t0 = time.perf_counter()
        try:
            while True:
                c0 = time.perf_counter()
                # Pedal costs ~128 ms (5 frames); the 203-byte motor block ~620 ms because
                # the ECU paces consecutive frames ~20 ms apart. Poll pedal every cycle, motor every
                # Nth, and let the caller interpolate in between.
                vcu = elm.read_hyundai("vcu", "22E004")
                vcu_payload = elm.last_did_payload
                counts["vcu"] += int("IONIQ5_ACCELERATOR_PEDAL_POSITION" in vcu)
                if cycle % max(1, args.motor_every) == 0:
                    motor = elm.read_hyundai("motor", "22E001")
                    motor_payload = elm.last_did_payload
                    counts["motor"] += int("IONIQ5_MOTOR_RPM" in motor)
                cycle += 1
                got_any = got_any or bool(motor) or bool(vcu)
                el = time.perf_counter() - t0
                def fmt(d, key, spec):
                    v = d.get(key)
                    return "N/A" if v is None else format(v, spec)
                part = (f"RPM {fmt(motor, 'IONIQ5_MOTOR_RPM', '7.0f'):>7s}  "
                        f"torque {fmt(motor, 'IONIQ5_MOTOR_ACTUAL_TORQUE', '7.1f'):>8s} Nm  "
                        f"pedal {fmt(vcu, 'IONIQ5_ACCELERATOR_PEDAL_POSITION', '5.1f'):>6s} %  "
                        f"cycle {1000*(time.perf_counter()-c0):5.0f} ms")
                if args.raw:
                    part += (f"  motor={len(motor_payload)}B "
                             f"vcu={len(vcu_payload)}B")
                if el > 0.5:
                    print(f"\r{part}  [RPM {counts['motor']/el:4.1f} Hz  "
                          f"pedal {counts['vcu']/el:4.1f} Hz]", end="", flush=True)
        except KeyboardInterrupt:
            print()
        if not got_any:
            print(f"no decoded values. raw payloads: motor {len(motor_payload)} B, "
                  f"vcu {len(vcu_payload)} B")
            if motor_payload:
                print(f"  motor hex: {motor_payload[:40].hex().upper()}")
            if vcu_payload:
                print(f"  vcu hex:   {vcu_payload[:40].hex().upper()}")
            if not motor_payload and not vcu_payload:
                print("  nothing came back at all -> run `obd.py diag`")
            else:
                print("  payloads arrived but didn't decode -> DID/layout mismatch "
                      "(2025+ uses E009 for motor data)")


def cmd_monitor(args):
    with Elm327(args.port, args.baud).open() as elm:
        elm.init(fast=True, protocol=args.protocol)
        pedal_pid = elm.best_pedal_pid() or THROTTLE
        print(f"pedal source: {_pid_name(pedal_pid)}   (ctrl-c to stop)")
        reader = ObdReader(elm, (RPM, pedal_pid))
        reader.start()
        try:
            t_print = 0.0
            while True:
                time.sleep(0.05)
                now = time.perf_counter()
                if now - t_print < 0.1:
                    continue
                t_print = now
                rpm = "" if reader.rpm is None else f"{reader.rpm:7.1f}"
                ped = "" if reader.pedal is None else f"{reader.pedal:6.1f}"
                print(f"\rRPM {rpm}  pedal {ped} %   {reader.rate_hz:5.1f} cyc/s   "
                      f"errors {reader.errors}   ", end="", flush=True)
        except KeyboardInterrupt:
            print()
        finally:
            reader.stop()


def cmd_selftest(_args):
    """Exercise init/parsing against a fake adapter -- no hardware needed."""
    fake = FakeSerial({
        "ATZ": ["\r\rvLinker FS v2.2\r\rELM327 v1.5\r\r>"],
        "ATSP6": ["\rOK\r>"], "ATCFC1": ["\rOK\r>"], "ATAT1": ["\rOK\r>"],
        "ATV0": ["\rOK\r>"], "ATE0": ["\rOK\r>"], "ATL0": ["\rOK\r>"],
        "ATS0": ["\rOK\r>"], "ATH0": ["\rOK\r>"], "ATAL": ["\rOK\r>"],
        "010C": ["\r410C1AF8\r>"],       # 0x1AF8 / 4 = 1726.0 rpm
        "0111": ["\r41117F\r>"],         # 0x7F * 100/255 = 49.8 %
        "0100": ["\r4100BE3FA800\r>"],
        "0120": ["\r412080001000\r>"],
        "0140": ["\r414000000000\r>"],
        "01FF": ["\rNO DATA\r>"],
        "010D": ["\rSTOPPED\r>"],
    })
    elm = Elm327("/dev/fake")
    elm.ser = fake
    elm.init(fast=True, protocol=6)

    assert [c for c in fake.written][:3] == ["ATZ", "ATE0", "ATL0"], fake.written
    # flow control must be ON -- without it multi-frame Mode 22 answers truncate
    assert "ATSP6" in fake.written and "ATCFC1" in fake.written and "ATAT1" in fake.written
    assert "ATCFC0" not in fake.written

    data, st = elm.query(RPM)
    assert st == "ok" and decode_pid(RPM, data) == 1726.0, (st, data)
    data, st = elm.query(THROTTLE)
    assert st == "ok" and abs(decode_pid(THROTTLE, data) - 49.8) < 0.2

    piped = elm.query_pids((RPM, THROTTLE))
    assert piped[RPM][1] == "ok" and decode_pid(RPM, piped[RPM][0]) == 1726.0, piped
    assert piped[THROTTLE][1] == "ok", piped

    data, st = elm.query(0xFF)
    assert st == "NODATA", st
    data, st = elm.query(SPEED)
    assert st == "STOPPED", st

    # 01 00 BE 3F A8 00 -> PIDs 01,03-07,0B-11,13,15; 01 20 80 00 10 00 -> 21,34
    assert elm.supported(0x00) == {0x01, 0x03, 0x04, 0x05, 0x06, 0x07, 0x0B, 0x0C,
                                   0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x13, 0x15}, \
        sorted(elm.supported(0x00))
    assert elm.supported(0x20) == {0x21, 0x34}, sorted(elm.supported(0x20))
    assert elm.best_pedal_pid() == THROTTLE, elm.best_pedal_pid()
    # reader schedule: RPM every cycle, pedal every 2nd
    fake.written.clear()
    reader = ObdReader(elm, (RPM, THROTTLE), pedal_every=2)
    for _ in range(4):
        reader.poll_once()
    assert fake.written.count("010C") == 4, fake.written
    assert fake.written.count("0111") == 2, fake.written
    assert parse_can_line("7E8 02 41 0C 1A F8") == (0x7E8, bytes.fromhex("02410C1AF8"))
    assert parse_can_line(">7E8 8 02 41 0C 1A F8 00 00 00") == \
        (0x7E8, bytes.fromhex("02410C1AF8000000")), parse_can_line(">7E8 8 02 41 0C 1A F8 00 00 00")
    assert parse_can_line("STOPPED") is None
    assert parse_can_line("") is None
    # Mode 22 decoding, checked against OBDb's recorded IONIQ 5 captures
    rpm_capture = [bytes.fromhex(x) for x in
                   ("10CB62E001FFFFFF", "21FF6009408E9337", "2200B0072C05FC38",
                    "23EA0F1901190107", "240187071D07FDFF")]
    p_rpm = reassemble_iso_tp(rpm_capture)
    assert decode_hyundai(p_rpm, 136) == 4074, decode_hyundai(p_rpm, 136)
    assert abs(decode_hyundai(p_rpm, 168, div=10) - 28.1) < 0.01
    assert abs(decode_hyundai(p_rpm, 152, div=10) - 28.1) < 0.01
    ped_capture = [bytes.fromhex(x) for x in
                   ("102262E004FC0000", "21000006F6020000", "220000217 5000000".replace(" ", ""))]
    p_ped = reassemble_iso_tp(ped_capture)
    assert decode_hyundai(p_ped, 64, div=512) == 0.0
    assert abs(decode_hyundai(p_ped, 96, div=2048) - 14.64168) < 0.01
    # adapter output as actually seen: length line, then "0:"-labelled segments
    assert reassemble_iso_tp([b"\x62\xE0\x04" + b"\x00" * 31])[:3] == b"\x62\xE0\x04"
    # and an already-reassembled payload must pass through untouched
    raw = bytes.fromhex("62E004" + "0000" * 20)
    assert reassemble_iso_tp([raw]) == raw
    # early-abort trigger: only when every declared byte is in a CR-terminated segment
    full = (b"022\r0:62E004FFFF00\r1:00EC05F6020000\r2:0000C371001000\r"
            b"3:00000004000000\r4:000000FFF80000\r")
    assert did_reply_complete(full)
    assert not did_reply_complete(full[:-1])            # last segment not terminated
    assert not did_reply_complete(full[:full.index(b"4:")])
    assert not did_reply_complete(b"NO DATA\r")
    assert did_reply_complete(b"7F2231\r")                 # negative answer: done
    assert not did_reply_complete(b"7F2278\r")             # response pending: keep waiting
    assert did_reply_complete(b"7F2278\r62F44926F40D00\r")
    assert did_reply_complete(b"62F44926F40D00\r")         # single frame
    assert not did_reply_complete(b"62F44926F4")            # not terminated yet
    # multi-DID reply (as read from the car, parked): pedal D 0x26, speed 0
    md = parse_multi_did(bytes.fromhex("62F44926F40D00"))
    assert md == {0x49: b"\x26", 0x0D: b"\x00"}, md
    assert abs(decode_pid(0x49, md[0x49]) - 14.9) < 0.1
    # pedal D calibration (from the drive): rest and dead band -> 0 %, then linear
    assert pedal_d_percent(38) == 0.0 and pedal_d_percent(43) == 0.0
    assert abs(pedal_d_percent(118) - 48.4) < 1.0        # logged: raw 118 <-> 48.0 %
    assert abs(pedal_d_percent(91) - 30.8) < 1.0         # logged: raw 91 <-> 31.4 %
    assert pedal_d_percent(255) == 100.0
    assert parse_multi_did(bytes.fromhex("62F40C1AF8F40D32")) == \
        {0x0C: b"\x1a\xf8", 0x0D: b"\x32"}
    # BMS 220101 is big-endian; this payload was read from the car (parked, SOC 65 %)
    bms = bytes.fromhex(
        "620101EFFBE7EF82000000000000101D250A0909090909090030C216C21900008A00000CD100"
        "000C3200000971000008C60010B4EF0002E90000000006A3")
    assert decode_hyundai(bms, 32, 8, div=2) == 65.0
    assert decode_hyundai(bms, 80, 16, div=10, signed=True, big_endian=True) == 1.6
    assert decode_hyundai(bms, 424, 16, signed=True, big_endian=True) == 0
    # signed fields go negative (reverse), e.g. -100 rpm big-endian
    neg = bms[:56] + (-100).to_bytes(2, "big", signed=True) + bms[58:]
    assert decode_hyundai(neg, 424, 16, signed=True, big_endian=True) == -100

    print("selftest OK: init order, parsing, errors, PID bitmask, schedule, monitor lines,")
    print(f"              Mode 22 decode vs OBDb captures (RPM {decode_hyundai(p_rpm, 136):.0f}, "
          f"torque {decode_hyundai(p_rpm, 168, div=10):.1f} Nm, pedal "
          f"{decode_hyundai(p_ped, 64, div=512):.0f} %), decode -> {reader.rpm} rpm")


class FakeSerial:
    """Just enough of pyserial.Serial to drive Elm327 in tests."""

    def __init__(self, responses: dict[str, list[str]]):
        self.responses = {k.replace(" ", ""): v for k, v in responses.items()}
        self.written: list[str] = []
        self._buf = b""

    def reset_input_buffer(self):
        self._buf = b""

    def write(self, data: bytes):
        for cmd in filter(None, data.decode("ascii").split("\r")):
            self.written.append(cmd)
            for txt in self.responses.get(cmd, ["\r?\r>"]):
                self._buf += txt.encode("ascii")

    @property
    def in_waiting(self):
        return len(self._buf)

    def read(self, n: int) -> bytes:
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def close(self):
        pass


def format_trace(t0: float, trace: list[tuple[float, bytes]]) -> str:
    """One line per USB read: ms since the request was written, gap, and the bytes."""
    rows, prev = [], t0
    for t, chunk in trace:
        txt = chunk.decode("ascii", "replace").replace("\r", "|")
        rows.append(f"    {(t - t0) * 1000:7.1f} ms  +{(t - prev) * 1000:6.1f}  "
                    f"{len(chunk):4d} B  {txt[:70]}")
        prev = t
    return "\n".join(rows) if rows else "    (nothing received)"


def cmd_ioniq_bench(args):
    """Measure every speed/pedal source, alone and interleaved. Read-only requests only."""
    if args.cycles < 1:
        raise ObdError("--cycles must be at least 1")
    sources = {
        "vcu obd pedal+speed": ("vcu", "obd", None),
        "hvac speed": ("hvac", "220100", "IONIQ5_VSS"),
        "bms motor rpm": ("bms", "220101", "IONIQ5_DRIVE_MOTOR_SPEED_1"),
        "vcu pedal+speed": ("vcu", "22E004", "IONIQ5_ACCELERATOR_PEDAL_POSITION"),
        "motor block": ("motor", "22E001", "IONIQ5_MOTOR_RPM"),
    }
    schedules = [(name,) for name in sources] + [
        ("vcu obd pedal+speed", "vcu obd pedal+speed", "vcu obd pedal+speed",
         "bms motor rpm"),
    ]
    if args.only:
        schedules = [s for s in schedules if any(args.only.lower() in n for n in s)]
    with Elm327(args.port, args.baud).open() as elm:
        elm.init(fast=True, protocol=args.protocol)
        chip = elm.stn_identity()
        print(f"adapter: {elm.version}" + (f"  (chip {chip})" if chip else "") +
              f"   low-latency USB: {'on' if elm.low_latency else 'OFF'}")
        print("Rates count decoded responses, not attempts.\n")
        for schedule in schedules:
            durations = {n: [] for n in schedule}
            ok = {n: 0 for n in schedule}
            last = {}
            start = time.perf_counter()
            for _ in range(args.cycles):
                for n in schedule:
                    ecu, did, key = sources[n]
                    t = time.perf_counter()
                    if did == "obd":
                        got = elm.read_vcu_fast()
                        values = {"OBD_PEDAL": got["pedal"]} if "pedal" in got else {}
                        if "speed" in got:
                            values["OBD_SPEED"] = got["speed"]
                        hit = len(values) == 2
                    else:
                        values = elm.read_hyundai(ecu, did)
                        hit = key in values
                    durations[n].append(time.perf_counter() - t)
                    ok[n] += int(hit)
                    last[n] = values
            elapsed = time.perf_counter() - start
            print(" + ".join(schedule) + f"  ({elapsed:.2f} s)")
            for n in dict.fromkeys(schedule):
                vals = "  ".join(f"{k.replace('IONIQ5_', '')}={v:g}" for k, v in last[n].items()
                                 if k in ("IONIQ5_VSS", "IONIQ5_VSS_HD", "IONIQ5_MOTOR_RPM",
                                         "OBD_PEDAL", "OBD_SPEED",
                                          "IONIQ5_DRIVE_MOTOR_SPEED_1",
                                          "IONIQ5_ACCELERATOR_PEDAL_POSITION"))
                print(f"  {n:16s} {ok[n]:>3d}/{args.cycles * schedule.count(n)} ok  "
                      f"{ok[n] / elapsed:6.2f} Hz  "
                      f"median {statistics.median(durations[n]) * 1000:5.1f} ms   {vals}")
                if args.trace:
                    print(format_trace(elm.last_t0, elm.last_trace))


def trace_summary(t0: float, trace: list[tuple[float, bytes]]) -> str:
    """First frame, last frame, prompt, and the dead wait between the last two."""
    data_t = [t for t, c in trace if b"STOPPED" not in c
              and any(ch in b"0123456789ABCDEF" for ch in c)]
    prompt_t = next((t for t, c in trace if b">" in c), None)
    if not data_t or prompt_t is None:
        return "no data"
    ms = lambda t: (t - t0) * 1000
    return (f"first {ms(data_t[0]):5.1f} ms, last {ms(data_t[-1]):5.1f} ms, "
            f"prompt {ms(prompt_t):5.1f} ms (tail wait {ms(prompt_t) - ms(data_t[-1]):5.1f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=None, help="serial port (default: autodetect)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--protocol", type=int, default=6,
                    help="6 = CAN 11/500 (default), 0 = auto")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("scan", cmd_scan), ("info", cmd_info), ("pids", cmd_pids),
                     ("selftest", cmd_selftest), ("monitor", cmd_monitor),
                     ("bench", cmd_bench), ("sweep", cmd_sweep), ("sniff", cmd_sniff), ("record", cmd_record),
                     ("ioniq", cmd_ioniq), ("ioniq-bench", cmd_ioniq_bench), ("diag", cmd_diag),
                     ("calibrate", cmd_calibrate)):
        p = sub.add_parser(name)
        p.set_defaults(func=fn)
        if name in ("bench", "sweep"):
            p.add_argument("--cycles", type=int, default=100)
        if name == "ioniq-bench":
            p.add_argument("--cycles", type=int, default=10,
                           help="reads per signal in each test (default: 10)")
            p.add_argument("--trace", action="store_true",
                           help="print per-chunk arrival times for one read of each DID")
            p.add_argument("--only", default="",
                           help="run only schedules with a source containing this text")
        if name == "bench":
            p.add_argument("--pids", default="0C")
        if name == "ioniq":
            p.add_argument("--raw", action="store_true", help="show payload sizes")
            p.add_argument("--motor-every", type=int, default=4,
                           help="poll the 203-byte motor block every Nth cycle")
        if name == "record":
            p.add_argument("--seconds", type=float, default=60.0)
            p.add_argument("--pids", default="0C,0D")
            p.add_argument("--save", default="rpm_log.csv")
        if name == "sniff":
            p.add_argument("--seconds", type=float, default=10.0)
            p.add_argument("--filter", default="", help="comma-separated CAN IDs, e.g. 1A0,2F1")
            p.add_argument("--top", type=int, default=15)
            p.add_argument("--save", default="")
            p.add_argument("--proto-timeout", type=int, default=20)
    args = ap.parse_args()
    try:
        args.func(args)
    except ObdError as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
