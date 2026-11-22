"""The Pi playing the sound with the native engine (soundcore/): runs sc_player and feeds it.

    main.py --native-sound [--output-device NAME]

The Python side keeps doing what it does well -- reading the car over OBD, serving the phone
over Bluetooth -- and hands the sound to the C++ player (soundcore/build/sc_player), which plays
the shared sound engine straight to an ALSA device: a USB sound card or a soundbar's USB input.
Each new reading of the car goes to the player the moment it arrives, as the same 10-byte
packet the phone gets (ble_control.py), over a local socket; so do setting changes (the sound,
the volume, gearshifts, a test drive). The player is restarted if it ever exits.
"""
from __future__ import annotations

import os
import socket
import subprocess
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAYER = HERE / "soundcore" / "build" / "sc_player"
EFFECTS = HERE / "soundcore" / "effects"
SOCKET = "/tmp/vcs-sound.sock"

# the native sounds, as the phone app and settings name them -> the player's names
SOUNDS = {"Flat-6": "flat6", "Hypercar": "hypercar", "Turbofan": "turbofan", "Warp bubble": "warp"}
DEFAULT_SOUND = "Flat-6"


def usb_card() -> str | None:
    """ALSA device for the first USB sound card (a USB DAC or a soundbar's USB input), if any."""
    try:
        cards = Path("/proc/asound/cards").read_text()
    except OSError:
        return None
    for line in cards.splitlines():
        # " 1 [Device         ]: USB-Audio - USB Audio Device"
        if "USB-Audio" in line and "[" in line:
            name = line.split("[", 1)[1].split("]", 1)[0].strip()
            return f"plughw:CARD={name}"
    return None


class NativeSound:
    def __init__(self, device: str | None = None, log=print, gearshifts: bool = True):
        self.device = device
        # the gearbox: on unless --no-virtual-gears. (Not the saved "virtual gears" setting: that
        # belongs to the Python voices, and an old saved "off" would switch the gearbox off here.)
        self.gearshifts = gearshifts
        self.log = log
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.proc: subprocess.Popen | None = None
        self._stop = threading.Event()
        self._sent: dict[str, str] = {}          # last value sent per setting
        self._playing_on: str | None = None      # the ALSA device the player was started on
        self.state: dict = {}                    # the player's latest state (sc_player's stdout)
        self.state_at = 0.0
        self._last_card_check = 0.0
        self._thread = threading.Thread(target=self._keep_running, name="native-sound", daemon=True)

    def start(self):
        if not PLAYER.exists():
            raise SystemExit(f"{PLAYER} isn't built: run  cmake -S soundcore -B soundcore/build && "
                             f"cmake --build soundcore/build  (install_service.sh --native-sound does it)")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def _keep_running(self):
        while not self._stop.is_set():
            device = self.device or usb_card() or "default"
            self._playing_on = device
            cmd = [str(PLAYER), "--socket", SOCKET, "--device", device, "--effects", str(EFFECTS)]
            self.log(f"native sound: starting the player on {device}")
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self._sent.clear()                   # a fresh player: send every setting again
            threading.Thread(target=self._read_state, args=(self.proc,), name="native-state", daemon=True).start()
            for line in self.proc.stderr:        # its log lines, into ours
                self.log(f"player: {line.rstrip()}")
            code = self.proc.wait()
            if self._stop.is_set():
                break
            self.log(f"native sound: the player exited ({code}); restarting in 2 s")
            self._stop.wait(2.0)

    def _read_state(self, proc):
        """The player's state, a JSON line on stdout a few times a second: for the phone's remote."""
        import json
        for line in proc.stdout:
            try:
                self.state = json.loads(line)
                self.state_at = time.monotonic()
            except ValueError:
                pass

    def status(self) -> dict | None:
        """The player's latest state, if it's recent (gear, rpm, km/h, limiter, test drive, underruns, cpu)."""
        return self.state if time.monotonic() - self.state_at < 2.0 else None

    def _check_card(self, now: float):
        """A USB sound card plugged in (or found) after the player started: move the player to it."""
        if self.device or now - self._last_card_check < 5.0:
            return
        self._last_card_check = now
        card = usb_card()
        if card and card != self._playing_on and self.proc and self.proc.poll() is None:
            self.log(f"native sound: found {card}; moving the player to it")
            self.proc.terminate()                # _keep_running starts it again, on the card

    # --- feeding it -------------------------------------------------------------------
    def _send(self, data: bytes) -> bool:
        try:
            self.sock.sendto(data, SOCKET)
            return True
        except OSError:
            return False                         # not up yet (or restarting)

    def telemetry(self, packet: bytes):
        """A reading of the car: the 10-byte packet the phone gets."""
        self._send(packet)

    def command(self, text: str):
        """A one-off command (a test drive, revving): held until the player can take it."""
        if not self._flush() or not self._send(text.encode("utf-8")):
            self._pending = (getattr(self, "_pending", []) + [text])[-5:]

    def _flush(self) -> bool:
        pending = getattr(self, "_pending", [])
        while pending:
            if not self._send(pending[0].encode("utf-8")):
                return False
            pending.pop(0)
        return True

    def settings(self, s):
        """Sends the settings that changed (the sound, volume, gearshifts) -- cheap to call often."""
        want = {
            "sound": SOUNDS.get(s.native_sound, SOUNDS[DEFAULT_SOUND]),
            "volume": f"{max(0.0, min(1.0, s.volume)):.2f}",
            "gearshifts": "1" if (self.gearshifts and s.native_gears) else "0",
        }
        # on a change -- and every couple of seconds anyway, so a player that has just
        # (re)started and missed the first ones catches up
        now = time.monotonic()
        self._check_card(now)
        self._flush()
        resend = now - getattr(self, "_last_full", 0.0) >= 2.0
        if resend:
            self._last_full = now
        for key, val in want.items():
            if resend or self._sent.get(key) != val:
                self.command(f"{key} {val}")
                self._sent[key] = val
