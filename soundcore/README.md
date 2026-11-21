# soundcore: the shared native sound engine

One C++ implementation of the car's sound for every host: the Raspberry Pi's player
(`sc_player`), Open Blue (through JNI), and offline rendering. Ported from Open Blue's Kotlin and
checked against it (within 0.2 dB(A) over the standard test drive); the Kotlin copies are gone now.

| file | what |
| --- | --- |
| `src/dsp.h` | the real-time DSP kit (biquads, resonators, PolyBLEP, table sine, noise) |
| `src/dct.*` | the 7-speed dual-clutch gearbox and its control unit; revving in place |
| `src/flat6.cpp` `hypercar.cpp` `jet.cpp` `warp.cpp` | the four layered voices |
| `src/tires.*` | the tires in hard corners: scrub, then a squeal, from the cornering force (speed x yaw rate) at speed |
| `src/effects.*` | chime, launches, the hard-shift sound, reverse (WAVs from `effects/`) |
| `src/engine.*` | the one entry point: car inputs in, stereo out, per block (plus hill effort, the test drive, the output stage) |
| `tools/player.cpp` | `sc_player`: the engine to an ALSA device, fed over a local socket |
| `tools/render.cpp` | `sc_render`: a drive to WAV, with levels and speed |
| `tests/test_core.cpp` | `sc_tests` |

    cmake -S soundcore -B soundcore/build && cmake --build soundcore/build
    soundcore/build/sc_tests soundcore/effects
    soundcore/build/sc_render hypercar out.wav soundcore/effects

On the Pi, `sudo ./install_service.sh --native-sound` builds it and runs `main.py
--native-sound`: Python reads the car and serves the phone, and sends each reading and setting
change to `sc_player` (`native_sound.py`). `effects/` is copied from Open Blue's assets by
`sync_effects.sh` -- run it after changing them there.

Open Blue builds `src/` into `libsoundcore_jni.so` from its own copy in
`app/src/main/cpp/soundcore` (with `soundcore_jni.cpp` and `SoundCore.kt` as the bridge). This
directory is the one to edit; `./sync_to_app.sh [path/to/open-blue-android]` copies it over.

Player socket (`/tmp/vcs-sound.sock`, UNIX datagrams): the 10-byte car-data packet of
`ble_control.py`, or a text command: `sound flat6|hypercar|turbofan|warp`, `volume 0..1`,
`gearshifts 0|1`, `testdrive start|stop`, `rev 0..1|off`.
