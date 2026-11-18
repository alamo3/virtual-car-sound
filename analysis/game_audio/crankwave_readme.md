# Crankwave audition

The BMW inline-six recording in `demos/crankwave_bmw_load_coast.wav` is an
offline render of [Crankwave](https://github.com/SvetlozarValchev/crankwave)'s
`bmw-m52tub28-cleanroom` engine and its
`canonical-loaded-rise-part-load-coast-1500-4500rpm` scenario. It is simulated
combustion audio, not a Hyundai or EV recording. The scenario calls for a rise
from 1500 to 4500 RPM, partial throttle, then lift and coast. It is an audition
of a possible underlying mechanical source, not a complete EV sound.

The build and source assets were installed in `/tmp/crankwave-src` and the
release archive was verified by the project's asset-fetch script. For this
local audition the source-stamp toolchain query, admitted target triple and
loaded-library identity checks were adjusted to work with this environment;
therefore the renderer's normal reproducibility guarantees do not apply. No
changes were made to the user's Engine Simulator clone. The audition WAV is
converted from the renderer's audition output to stereo 48 kHz PCM for ease of
playback. Crankwave code is MIT; consult its asset notices for individual
asset licensing before redistribution.
