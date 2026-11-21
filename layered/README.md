# Layered voices (prototypes)

Four EV voices built from layers that each follow their own control (revs, load, speed,
acceleration, shifts, the real motor), instead of one pitched loop. Prototyped here in
Python, rendered offline over a standard test drive, then ported to Open Blue.

| voice | what drives it | layers |
| --- | --- | --- |
| `flat6.py` | DCT revs and load | cylinder-by-cylinder exhaust pulses (firing order 1-6-2-4-3-5, bank imbalance for the boxer rasp), pipes and a muffler that opens with load, intake roar, mechanical detail, overrun crackle, ignition cut on shifts |
| `hypercar.py` | DCT revs + the car's real motor rpm | detuned saw/pulse core through a load-opened resonant filter, FM edge under load, sub, real motor orders, spool whistle, air, shift zap, regen pad |
| `jet.py` | a lagged turbine spool (pedal + speed) | fan blade-pass tone turning to buzzsaw, core whine, combustion rumble, jet roar, afterburner (thump + crackling roar), air |
| `warp.py` | acceleration and speed | Shepard-Risset drone gliding with acceleration, throbbing field bubble, comb shimmer, charge on a hard press, collapse on a lift, stereo width with speed |

`dct.py` is the Python twin of Open Blue's `Dct.kt` (keep them in step); `drive.py` is the
standard test drive (a rough IONIQ 5, its data smoothed as the app does).

    python layered/render.py layered/out            # all four: WAV + spectrogram each
    python layered/render.py layered/out flat6      # just one

Needs numpy, scipy (and matplotlib for the pictures).
