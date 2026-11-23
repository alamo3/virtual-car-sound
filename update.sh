#!/usr/bin/env bash
# Update the code from GitHub before the engine starts (run by the service: ExecStartPre).
#
#   ./update.sh          update if online; always exits 0 so the engine starts regardless
#
# - offline (GitHub unreachable within a few seconds): skip, start with what's here
# - local edits or local commits not on GitHub: skip (never overwrites your changes)
# - otherwise: fast-forward to origin's branch; if requirements.txt changed, install the
#   new Python packages into .venv
# - either way, where the native player is built, bring it up to date (incremental)
# Changes to install_service.sh itself (the service definition) still need
#   sudo ./install_service.sh   to take effect.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR" || exit 0
log() { echo "update: $*"; }

# The native player (only where it's been built: install_service.sh --native-sound): brought up
# to date on every run, whatever happened above -- an incremental build, a no-op when nothing
# changed, so a build that failed or was cut short (power off mid-build) is finished next time.
rebuild_player() {
    [ -d "$DIR/soundcore/build" ] || return 0
    local out
    if out=$(timeout 900 cmake --build "$DIR/soundcore/build" -j2 2>&1); then
        if echo "$out" | grep -q "Linking CXX executable sc_player"; then
            log "rebuilt the native player (it plays from the next start)"
        fi
    else
        log "rebuilding the native player failed (keeping the old one; tries again next time):"
        echo "$out" | grep -iE "error|fatal" | head -5 | sed 's/^/update:   /'
    fi
}
trap rebuild_player EXIT

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    log "not a git checkout, skipping"; exit 0
fi
branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
if [ -z "$branch" ] || [ "$branch" = HEAD ]; then
    log "not on a branch, skipping"; exit 0
fi

# online? (DNS + reaching the remote, quickly; no internet in the car is the normal case)
if ! timeout 12 git ls-remote --heads origin "$branch" >/dev/null 2>&1; then
    log "offline (or GitHub unreachable), starting without updating"; exit 0
fi

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    log "local changes in the checkout, not updating (commit or discard them to resume)"; exit 0
fi

if ! timeout 60 git fetch --quiet origin "$branch"; then
    log "fetch failed, starting without updating"; exit 0
fi
old=$(git rev-parse HEAD)
new=$(git rev-parse "origin/$branch")
if [ "$old" = "$new" ]; then
    log "up to date ($(git log -1 --format='%h %s'))"; exit 0
fi
# only ever a fast-forward: local commits not on GitHub (HEAD not behind origin) are left alone
if ! git merge-base --is-ancestor HEAD "origin/$branch" || ! git merge --ff-only --quiet "origin/$branch" 2>/dev/null; then
    log "local commits not on GitHub, not updating (push them to resume)"; exit 0
fi
log "updated $(git rev-parse --short "$old") -> $(git log -1 --format='%h %s')"

if ! git diff --quiet "$old" "$new" -- requirements.txt; then
    log "requirements.txt changed: installing Python packages"
    timeout 600 "$DIR/.venv/bin/python" -m pip install -q -r requirements.txt \
        || log "pip install failed (will retry next start)"
fi
if ! git diff --quiet "$old" "$new" -- install_service.sh; then
    log "install_service.sh changed: run 'sudo ./install_service.sh' to apply the new service settings"
fi
exit 0
