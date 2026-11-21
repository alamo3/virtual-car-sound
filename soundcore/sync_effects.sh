#!/bin/sh
# Copy the drive-effect sounds from Open Blue (where they're chosen) into soundcore/effects, so
# the Pi's player plays the same ones. Run after changing them there.
#   ./sync_effects.sh [path/to/open-blue-android]
set -e
src="${1:-../../open-blue-android}/app/src/main/assets/effects"
dst="$(dirname "$0")/effects"
mkdir -p "$dst"
for n in chime launch_build launch_bang boom shift_cin_up shift_cin_down reverse; do
    cp "$src/$n.wav" "$dst/"
done
echo "copied the effects from $src"
