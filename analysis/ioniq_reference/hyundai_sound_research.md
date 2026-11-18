# IONIQ 5 N active sound: public evidence

## Primary source

Hyundai Motor Group developer-conference presentation, “Development of high-performance EV sound: the differentiated IONIQ 5 N sound development case,” credited to Taegeon Yun, Hyundai/Kia Sound Design Research Lab:
https://aw-download-file.hmg-corp.io/3m5b8c7d9k/01HESPDAP67HYSEHT61Z28Q824

The presentation is in Korean. Its system-comparison table identifies seven driving sound-design layers for IONIQ 5 N e-ASD N versus two for the IONIQ 6 e-ASD. Its N-system diagram lists engine sound logic, FM modulation, wave playback, acceleration effect, launch-control sound, and shift-sound logic; it also shows an external speaker. This is a list of depicted features, not a verified one-to-one list of the seven layers or a routing diagram for either named theme.

The slides describe ordinary EV e-ASD in terms of granular synthesis using short samples, and describe the N system as expanding the driving-sound layers with engine-order sound design and waveform playback. They say the engine-sound design changes timbre using information such as torque, and show separate sound processing for EV versus virtual-gear-shift driving. This does not establish the precise granular algorithm, source waveforms, order curves, or routing used in Evolution or Supersonic.

The theme slides identify Evolution with the RN22e and N 2025 Gran Turismo concept sounds and say lateral acceleration controls left-right panning. Supersonic has a sonic-boom effect associated with gear shifts. Those effect notes do not reveal the source of the continuous driving sounds.

## Related, not proof of product implementation

Hyundai/Kia patent US20240326688A1 describes a possible acoustic-design processor that uses granular synthesis of stored sound sources and varies its output with vehicle speed, torque, acceleration/deceleration, steering, and gear. It is not evidence that either IONIQ 5 N theme uses that exact patent embodiment:
https://patents.google.com/patent/US20240326688A1/en

Hyundai Mobis patent US20230311756A1 describes another possible system using virtual RPM, harmonic control, and tone or wavetable synthesis. It similarly does not prove that the IONIQ 5 N uses that implementation:
https://patents.google.com/patent/US20230311756A1/en

## What we have and do not have

The user's recording is a cabin-video mix containing vehicle sound, possible road/cabin noise and commentary, not the original isolated layers or driving telemetry. The long-loop study preserves more source character by user report; the earlier arbitrary frozen-grain study did not. This is feedback on those implementations, not proof that granular synthesis is unsuitable.

The current `RecordingVoice` makes overlapping 120–200 ms grains with random start jitter and holds or maps one recording position by speed. The fixed-pitch study held a single position and combined grains from an already mixed, changing recording. Hyundai's presentation does not disclose enough detail to claim their grain selection, phase alignment, or layer architecture is identical or different.
