#!/bin/sh
# Copy the sound engine's sources into Open Blue (app/src/main/cpp/soundcore), which builds them
# into libsoundcore_jni.so. This directory is the one to edit; run this afterwards.
#   ./sync_to_app.sh [path/to/open-blue-android]
set -e
here="$(cd "$(dirname "$0")" && pwd)"
dst="${1:-$here/../../open-blue-android}/app/src/main/cpp/soundcore"
mkdir -p "$dst"
rm -f "$dst"/*.h "$dst"/*.cpp
for f in "$here"/src/*.h "$here"/src/*.cpp; do
    { echo "// Copied from virtual-car-sound/soundcore/src by sync_to_app.sh -- edit it there, not here."; cat "$f"; } > "$dst/$(basename "$f")"
done
echo "copied soundcore/src -> $dst"
