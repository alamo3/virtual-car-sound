#!/usr/bin/env bash
# Install the virtual car sound engine as a systemd service that starts at boot.
#
#   sudo ./install_service.sh            install (or update) and start it
#   sudo ./install_service.sh --data-only
#                                        install in data-only mode: no sound on the Pi, it just
#                                        reads the car and sends the data to the phone (Open Blue
#                                        plays the sound). Needs no speaker: fits a Pi Zero 2 W
#   sudo ./install_service.sh --native-sound
#                                        the Pi plays the sound with the native engine
#                                        (soundcore/: the same sounds as Open Blue) to a USB sound
#                                        card or soundbar, and still sends the car's data to the
#                                        phone. Builds the player (needs cmake, g++, ALSA headers)
#   sudo ./install_service.sh --with-sound
#                                        switch back to the Pi playing the sound itself
#   sudo ./install_service.sh uninstall  stop it and remove the service
#   ./install_service.sh status          is it running?
#   ./install_service.sh logs            follow its log (Ctrl+C to stop following)
#
# What install does:
#   - makes the Python environment (.venv) if it is missing, and installs requirements
#   - adds the user to the bluetooth / audio / dialout groups (phone app, speaker, OBD)
#   - switches Bluetooth on at boot
#   - writes /etc/default/virtual-car-sound with the start options (edit to change them)
#   - writes /etc/systemd/system/virtual-car-sound.service, enables and starts it
# Before each start the service runs update.sh: if the Pi is online it pulls the latest code
# from GitHub (fast-forward only, never over local changes); offline it just starts.
# The service runs as the user who ran sudo and logs to the system journal. It restarts
# whenever it exits or hangs (systemd watchdog), with no limit on restarts, and the Pi's
# hardware watchdog reboots the board if the whole system ever freezes. The phone app's settings are kept in ~/.config/virtual-car-sound/.
set -euo pipefail

NAME=virtual-car-sound
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT=/etc/systemd/system/$NAME.service
HW_WATCHDOG=/etc/systemd/system.conf.d/90-$NAME-watchdog.conf
DEFAULTS=/etc/default/$NAME
RUN_USER="${SUDO_USER:-$(id -un)}"
# start options: headless (no window), read the car over OBD, accept the phone app
SOUND_ARGS="--headless --car --bluetooth --log-every 30"
# data only: no sound, just the car's data to the phone over Bluetooth
DATA_ARGS="--data-only --log-every 30"
# native sound: the C++ player (soundcore/) makes the sound, Python reads the car and serves the phone
NATIVE_ARGS="--native-sound --log-every 30"
MODE=""        # "", "data", "native" or "sound": set by --data-only / --native-sound / --with-sound

need_root() {
    if [ "$(id -u)" -ne 0 ]; then
        echo "Run this with sudo:  sudo $0 ${1:-}" >&2
        exit 1
    fi
}

install() {
    need_root install
    if [ "$RUN_USER" = root ]; then
        echo "Run it with sudo from your normal user (the service runs as that user)." >&2
        exit 1
    fi
    echo "== $NAME: installing for user '$RUN_USER' in $DIR"

    # Python environment, as the user (not root). The venv carries its own pip, so no
    # system-wide pip is needed -- only Debian's venv module to create it the first time.
    if [ ! -x "$DIR/.venv/bin/python" ]; then
        if ! python3 -c "import ensurepip, venv" 2>/dev/null; then
            echo "-- installing python3-venv"
            apt-get install -y -qq python3-venv >/dev/null
        fi
        echo "-- creating .venv"
        sudo -u "$RUN_USER" python3 -m venv "$DIR/.venv"
    fi
    echo "-- installing Python requirements into .venv (with the venv's own pip)"
    sudo -u "$RUN_USER" "$DIR/.venv/bin/python" -m pip install -q -r "$DIR/requirements.txt"

    # the mode it will run in: the one asked for, else the one it's already in
    local mode="$MODE"
    if [ -z "$mode" ] && [ -f "$DEFAULTS" ] && grep -q -- '^VCS_ARGS=.*--data-only' "$DEFAULTS"; then
        mode=data
    fi
    if [ -z "$mode" ] && [ -f "$DEFAULTS" ] && grep -q -- '^VCS_ARGS=.*--native-sound' "$DEFAULTS"; then
        mode=native
    fi

    # native sound: build the player (as the user) -- cmake, a C++ compiler, ALSA headers
    if [ "$mode" = native ]; then
        echo "-- native sound: installing build tools and building soundcore/"
        apt-get install -y -qq cmake g++ libasound2-dev >/dev/null || echo "   (couldn't install the build tools: no network?)"
        sudo -u "$RUN_USER" cmake -S "$DIR/soundcore" -B "$DIR/soundcore/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
        sudo -u "$RUN_USER" cmake --build "$DIR/soundcore/build" -j2
        sudo -u "$RUN_USER" "$DIR/soundcore/build/sc_tests" "$DIR/soundcore/effects" | tail -1
    fi

    # ffmpeg: reads the mp3 recordings some voices use (Raspberry Pi OS Lite lacks it);
    # not needed in data-only mode, where the Pi plays no sound
    if [ "$mode" = data ] || [ "$mode" = native ]; then
        echo "-- $mode mode: skipping ffmpeg (no Python voices)"
    elif ! command -v ffmpeg >/dev/null; then
        echo "-- installing ffmpeg"
        apt-get install -y -qq ffmpeg >/dev/null || echo "   (couldn't install ffmpeg: no network?)"
    fi

    # permissions: Bluetooth (phone app), audio (USB speaker), serial (OBD adapter)
    for g in bluetooth audio dialout; do
        if getent group "$g" >/dev/null; then usermod -aG "$g" "$RUN_USER"; fi
    done

    # Bluetooth on at boot
    systemctl enable --now bluetooth >/dev/null 2>&1 || echo "   (no bluetooth service found)"
    command -v rfkill >/dev/null && rfkill unblock bluetooth || true

    # start options: kept if the file already exists, so your edits survive updates --
    # unless --data-only / --with-sound asks for a different mode
    local args="$SOUND_ARGS"
    [ "$MODE" = data ] && args="$DATA_ARGS"
    [ "$MODE" = native ] && args="$NATIVE_ARGS"
    if [ ! -f "$DEFAULTS" ]; then
        cat > "$DEFAULTS" <<EOF
# Options for $NAME (see: $DIR/.venv/bin/python $DIR/main.py --help)
# Sound on the Pi:  VCS_ARGS="$SOUND_ARGS"
#   (e.g. add  --voice "Loop Engine (BMW M3 GTR interior)"  or  --output-device usb)
# Data only (the phone plays the sound):  VCS_ARGS="$DATA_ARGS"
# Native sound on the Pi (soundcore/):  VCS_ARGS="$NATIVE_ARGS"  (add --output-device NAME)
VCS_ARGS="$args"
EOF
        echo "-- wrote $DEFAULTS"
    elif [ -n "$MODE" ]; then
        sed -i "s|^VCS_ARGS=.*|VCS_ARGS=\"$args\"|" "$DEFAULTS"
        echo "-- set $DEFAULTS to $MODE mode: VCS_ARGS=\"$args\""
    else
        echo "-- keeping existing $DEFAULTS"
    fi
    echo "-- mode: $(grep -q -- '--data-only' "$DEFAULTS" && echo 'data only (no sound on the Pi)' || \
        (grep -q -- '--native-sound' "$DEFAULTS" && echo 'native sound on the Pi' || echo 'sound on the Pi'))"

    cat > "$UNIT" <<EOF
[Unit]
Description=Virtual car sound engine
After=bluetooth.target sound.target
Wants=bluetooth.target
# never give up: keep restarting however often it fails (default: stop after 5 in 10 s)
StartLimitIntervalSec=0

[Service]
# the program tells systemd when it is ready and pings it every 2 s while running;
# if the pings stop (a hang), systemd kills and restarts it
Type=notify
NotifyAccess=main
WatchdogSec=60
# long enough for update.sh to install new Python packages before the engine starts
TimeoutStartSec=900
TimeoutStopSec=10
User=$RUN_USER
SupplementaryGroups=bluetooth audio dialout
WorkingDirectory=$DIR
EnvironmentFile=-$DEFAULTS
Environment=PYTHONUNBUFFERED=1
# update from GitHub first if online (never fails the start: the leading '-')
ExecStartPre=-$DIR/update.sh
ExecStart=$DIR/.venv/bin/python $DIR/main.py \$VCS_ARGS
Restart=always
RestartSec=3
# audio first: a little more CPU priority than ordinary background jobs, and the last
# thing the kernel kills if memory runs out
Nice=-5
OOMScoreAdjust=-500
# the native player may run its audio thread at real-time priority (fewer dropouts under load)
LimitRTPRIO=50

[Install]
WantedBy=multi-user.target
EOF
    echo "-- wrote $UNIT"

    # hardware watchdog (Raspberry Pi has one): if the whole system freezes, the board
    # reboots itself and the service starts again
    if [ -e /dev/watchdog ]; then
        mkdir -p "$(dirname "$HW_WATCHDOG")"
        printf '%s\n' "# written by $NAME's install_service.sh: reboot if the system hangs" \
            "[Manager]" "RuntimeWatchdogSec=15" "RebootWatchdogSec=2min" > "$HW_WATCHDOG"
        systemctl daemon-reexec
        echo "-- hardware watchdog on (reboots the Pi if it ever freezes)"
    fi

    systemctl daemon-reload
    systemctl enable "$NAME" >/dev/null
    systemctl restart "$NAME"
    sleep 2
    systemctl --no-pager --lines=8 status "$NAME" || true
    echo
    echo "Installed. It starts at every boot. Useful commands:"
    echo "  $0 logs                     follow the log"
    echo "  sudo systemctl restart $NAME   restart after changing $DEFAULTS"
    echo "  sudo $0 uninstall          remove it"
}

uninstall() {
    need_root uninstall
    systemctl disable --now "$NAME" 2>/dev/null || true
    rm -f "$UNIT"
    if [ -f "$HW_WATCHDOG" ]; then rm -f "$HW_WATCHDOG"; systemctl daemon-reexec; fi
    systemctl daemon-reload
    echo "Removed the $NAME service (kept $DEFAULTS and the project files)."
}

CMD=install
for a in "$@"; do
    case "$a" in
        --data-only) MODE=data ;;
        --with-sound) MODE=sound ;;
        --native-sound) MODE=native ;;
        -*) echo "unknown option $a" >&2; exit 2 ;;
        *) CMD="$a" ;;
    esac
done

case "$CMD" in
    install) install ;;
    uninstall|remove) uninstall ;;
    status) systemctl --no-pager status "$NAME" ;;
    logs) journalctl -u "$NAME" -f -n 50 ;;
    *) echo "usage: sudo $0 [install] [--data-only|--native-sound|--with-sound] | sudo $0 uninstall | $0 [status|logs]" >&2; exit 2 ;;
esac
