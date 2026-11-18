#!/usr/bin/env bash
# Install the virtual car sound engine as a systemd service that starts at boot.
#
#   sudo ./install_service.sh            install (or update) and start it
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
# The service runs as the user who ran sudo, restarts itself if it ever exits, and logs
# to the system journal. The phone app's settings are kept in ~/.config/virtual-car-sound/.
set -euo pipefail

NAME=virtual-car-sound
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT=/etc/systemd/system/$NAME.service
DEFAULTS=/etc/default/$NAME
RUN_USER="${SUDO_USER:-$(id -un)}"
# start options: headless (no window), read the car over OBD, accept the phone app
DEFAULT_ARGS="--headless --car --bluetooth --log-every 30"

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

    # Python environment, as the user (not root)
    if [ ! -x "$DIR/.venv/bin/python" ]; then
        echo "-- creating .venv"
        sudo -u "$RUN_USER" python3 -m venv "$DIR/.venv"
    fi
    echo "-- installing Python requirements"
    sudo -u "$RUN_USER" "$DIR/.venv/bin/pip" install -q -r "$DIR/requirements.txt"

    # permissions: Bluetooth (phone app), audio (USB speaker), serial (OBD adapter)
    for g in bluetooth audio dialout; do
        if getent group "$g" >/dev/null; then usermod -aG "$g" "$RUN_USER"; fi
    done

    # Bluetooth on at boot
    systemctl enable --now bluetooth >/dev/null 2>&1 || echo "   (no bluetooth service found)"
    command -v rfkill >/dev/null && rfkill unblock bluetooth || true

    # start options: kept if the file already exists, so your edits survive updates
    if [ ! -f "$DEFAULTS" ]; then
        cat > "$DEFAULTS" <<EOF
# Options for $NAME (see: $DIR/.venv/bin/python $DIR/main.py --help)
# e.g. add  --voice "Loop Engine (BMW M3 GTR interior)"  or  --output-device usb
VCS_ARGS="$DEFAULT_ARGS"
EOF
        echo "-- wrote $DEFAULTS"
    else
        echo "-- keeping existing $DEFAULTS"
    fi

    cat > "$UNIT" <<EOF
[Unit]
Description=Virtual car sound engine
After=bluetooth.target sound.target
Wants=bluetooth.target

[Service]
Type=simple
User=$RUN_USER
SupplementaryGroups=bluetooth audio dialout
WorkingDirectory=$DIR
EnvironmentFile=-$DEFAULTS
Environment=PYTHONUNBUFFERED=1
ExecStart=$DIR/.venv/bin/python $DIR/main.py \$VCS_ARGS
Restart=always
RestartSec=3
# audio first: a little more CPU priority than ordinary background jobs
Nice=-5

[Install]
WantedBy=multi-user.target
EOF
    echo "-- wrote $UNIT"

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
    systemctl daemon-reload
    echo "Removed the $NAME service (kept $DEFAULTS and the project files)."
}

case "${1:-install}" in
    install) install ;;
    uninstall|remove) uninstall ;;
    status) systemctl --no-pager status "$NAME" ;;
    logs) journalctl -u "$NAME" -f -n 50 ;;
    *) echo "usage: sudo $0 [install|uninstall] | $0 [status|logs]" >&2; exit 2 ;;
esac
