#!/usr/bin/env bash
# Update the code from GitHub before the engine starts (run by the service: ExecStartPre).
#
#   ./update.sh          update if online; always exits 0 so the engine starts regardless
#
# - offline (GitHub unreachable within a few seconds): skip, start with what's here
# - local edits or local commits not on GitHub: skip (never overwrites your changes)
# - otherwise: fast-forward to origin's branch; if requirements.txt changed, install the
#   new Python packages into .venv
# Changes to install_service.sh itself (the service definition) still need
#   sudo ./install_service.sh   to take effect.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR" || exit 0
log() { echo "update: $*"; }

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
if ! git merge --ff-only --quiet "origin/$branch" 2>/dev/null; then
    log "local commits not on GitHub, not updating"; exit 0
fi
log "updated $(git rev-parse --short "$old") -> $(git log -1 --format='%h %s')"

# the native player: rebuild it when its code changed (only where it's been built)
if [ -d "$DIR/soundcore/build" ] && ! git diff --quiet "$old" "$new" -- soundcore; then
    log "soundcore changed: rebuilding the native player"
    timeout 900 cmake --build "$DIR/soundcore/build" -j2 >/dev/null 2>&1 || log "  (rebuild failed: keeping the old player)"
fi

if ! git diff --quiet "$old" "$new" -- requirements.txt; then
    log "requirements.txt changed: installing Python packages"
    timeout 600 "$DIR/.venv/bin/python" -m pip install -q -r requirements.txt \
        || log "pip install failed (will retry next start)"
fi
if ! git diff --quiet "$old" "$new" -- install_service.sh; then
    log "install_service.sh changed: run 'sudo ./install_service.sh' to apply the new service settings"
fi
exit 0
