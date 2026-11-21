// The standard test drive used to audition and compare the sounds (the same script, car model
// and handling of the car's data as Open Blue's LayeredVoiceTest and virtual-car-sound's
// layered/drive.py): idle, a gentle pull and cruise, a kickdown and hard pull, a lift, braking
// to a stop, a hard launch from rest, a lift, a stop.
#pragma once
#include <algorithm>
#include <cmath>
#include "engine.h"

namespace sc {

inline void driveScript(float t, float& pedal, float& brake) {
    struct Step { float end, p, b; };
    static const Step steps[] = {{3, 0, 0}, {11, 0.25f, 0}, {17, 0.15f, 0}, {25, 1, 0}, {27, 0, 0}, {32, 0, 3.5f},
                                 {34, 0, 0}, {41, 1, 0}, {44, 0, 0}, {49, 0, 3}, {52, 0, 0}};
    for (const Step& s : steps) if (t < s.end) { pedal = s.p; brake = s.b; return; }
    pedal = brake = 0;
}

// the car, and its data as the app sees it (whole km/h, smoothed)
struct DriveSim {
    float v = 0, pedal = 0, lastRaw = -1, smooth = 0, acc = 0;
    Inputs step(float t, float dt) {
        float want, brake;
        driveScript(t, want, brake);
        pedal += std::clamp(want - pedal, -5.f * dt, 5.f * dt);
        float push = std::pow(pedal, 0.8f) * std::min(3.9f, 160000.f / 2100.f / std::max(v, 1.f));
        float regen = v > 0.5f ? 1.7f * std::max(0.f, 1.f - pedal / 0.06f) : 0.f;
        v = std::max(0.f, v + (push - 0.15f - 0.0004f * v * v - regen - brake) * dt);
        float raw = std::round(v * 3.6f);
        if (lastRaw >= 0) {
            acc += ((raw - lastRaw) / 3.6f / dt - acc) * std::min(1.f, dt / 0.3f);
            smooth += (raw - smooth) * std::min(1.f, dt / 0.15f);
        } else {
            smooth = raw;
        }
        lastRaw = raw;
        float kmh = raw < 0.5f ? 0.f : std::max(0.f, smooth + acc * 3.6f * 0.15f);
        Inputs in;
        in.pedal = pedal; in.speedMs = kmh / 3.6f; in.accel = acc; in.motorRpm = (int)(v * 3.6f * 77.5f);
        return in;
    }
};

}  // namespace sc
