// Checks for the native sound core (the same behaviour Open Blue's DctTest / LayeredVoiceTest /
// TestDriveTest check on the Kotlin): the gearbox's shifting, limiter and idle; every sound plays
// cleanly, within the limiter's ceiling, louder pulling than cruising, comfortably in real time.
//
//   sc_tests [effects_dir]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../tools/drive.h"
#include "engine.h"

using namespace sc;

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { std::printf("FAIL %s:%d  ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); failures++; } } while (0)

static void gearboxGentleNeverHuntsOverAnyRevRange() {
    const float ranges[][2] = {{900, 6900}, {900, 7000}, {1000, 7000}, {2200, 7000}, {1000, 7500}, {900, 8500}, {1000, 9000}};
    for (auto& rr : ranges) for (float p : {0.15f, 0.25f, 0.4f, 0.6f}) {
        Dct box(rr[0], rr[1]);
        float v = 0, dt = 0.04f;
        std::vector<int> seq;
        for (float t = 0; t < 20; t += dt) {
            float push = std::pow(p, 0.8f) * std::fmin(3.9f, 160000.f / 2100.f / std::fmax(v, 1.f));
            v = std::fmax(0.f, v + (push - 0.15f - 0.0004f * v * v) * dt);
            int g = box.step(dt, v * 3.6f, p).gear;
            if (seq.empty() || seq.back() != g) seq.push_back(g);
        }
        bool monotone = true;
        for (size_t i = 1; i < seq.size(); ++i) monotone &= seq[i] > seq[i - 1];
        CHECK(monotone, "%.0f-%.0f rpm at %.0f %% pedal hunts", rr[0], rr[1], p * 100);
    }
}

static void gearboxFlooredBouncesOffTheLimiter() {
    Dct box(900, 7000);
    float v = 0, dt = 0.04f;
    int ups = 0, lastShifts = 0, cutsBeforeUp = 0, cuts = 0;
    bool wasCut = false;
    for (float t = 0; t < 14; t += dt) {
        float p = t < 1 ? 0.f : 1.f;
        float push = std::pow(p, 0.8f) * std::fmin(3.9f, 160000.f / 2100.f / std::fmax(v, 1.f));
        v = std::fmax(0.f, v + (push - 0.15f - 0.0004f * v * v) * dt);
        EngineState s = box.step(dt, v * 3.6f, p);
        if (s.cut && !wasCut) cuts++;
        wasCut = s.cut;
        if (s.shifts != lastShifts) { ups++; cutsBeforeUp += cuts; cuts = 0; lastShifts = s.shifts; }
    }
    CHECK(ups >= 2, "floored: only %d upshifts", ups);
    CHECK(cutsBeforeUp >= 3 * ups, "floored: %d limiter cuts over %d upshifts", cutsBeforeUp, ups);
}

static void gearboxColdIdleSettles() {
    Dct box(900, 7000);
    EngineState s;
    for (int i = 0; i < 50; ++i) s = box.step(0.04f, 0, 0);
    CHECK(s.rpm > 1100 && s.rpm < 1300, "cold idle %.0f", s.rpm);
    for (int i = 0; i < 2500; ++i) s = box.step(0.04f, 0, 0);
    CHECK(s.rpm > 860 && s.rpm < 950, "warm idle %.0f", s.rpm);
}

static void testDriveReaches100AndStops() {
    TestDrive d;
    float t = 0, top = 0, left = 0, right = 0;
    while (d.step(256.f / 48000)) { t += 256.f / 48000; top = std::fmax(top, d.kmh()); left = std::fmax(left, d.yawRate * d.speed); right = std::fmin(right, d.yawRate * d.speed); }
    CHECK(top >= 100 && top < 106, "test drive top %.0f km/h", top);
    CHECK(t > 15 && t < 40, "test drive %.1f s", t);
    CHECK(d.speed == 0.f, "test drive ends moving");
    CHECK(left > 1.5f && right < -3.f, "test drive corners %.1f / %.1f", left, right);
}

static void everySoundPlaysCleanly(const std::string& fx) {
    for (int k = 0; k < 4; ++k) {
        Sound snd = (Sound)k;
        Engine e(48000, fx);
        e.setSound(snd);
        DriveSim sim;
        const int frames = 256;
        float dt = frames / 48000.f;
        std::vector<float> l(frames), r(frames), all;
        double cpu = 0;
        for (float t = 0; t < 52; t += dt) {
            Inputs in = sim.step(t, dt);
            auto t0 = std::chrono::steady_clock::now();
            e.render(in, l.data(), r.data(), frames);
            cpu += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            for (int i = 0; i < frames; ++i) { all.push_back(l[i]); all.push_back(r[i]); }
        }
        bool finite = true;
        float peak = 0;
        for (float v : all) { finite &= std::isfinite(v); peak = std::fmax(peak, std::fabs(v)); }
        auto rms = [&](float a, float b) { double en = 0; size_t i0 = (size_t)(a * 48000) * 2, i1 = (size_t)(b * 48000) * 2;
            for (size_t i = i0; i < i1; ++i) en += all[i] * all[i]; return 10 * std::log10(en / (i1 - i0) + 1e-12); };
        double cruise = rms(12, 16), pull = rms(19, 24), realtime = 52.0 / cpu;
        std::printf("%-12s peak %.2f, cruise %.1f, hard pull %.1f dBFS, %.0fx real time\n", info(snd).name, peak, cruise, pull, realtime);
        CHECK(finite, "%s: not finite", info(snd).name);
        CHECK(peak > 0.05f && peak <= 0.891f, "%s: peak %.3f", info(snd).name, peak);
        CHECK(pull > cruise + 2, "%s: pull %.1f vs cruise %.1f", info(snd).name, pull, cruise);
        CHECK(realtime > 15, "%s: only %.0fx real time", info(snd).name, realtime);
    }
}

// hosts resend their settings: the same sound again must not reset its gearbox
static void sameSoundAgainKeepsTheGearbox() {
    Engine e(48000);
    e.setSound(Sound::HYPERCAR);
    std::vector<float> l(256), r(256);
    Inputs in;
    in.pedal = 0.3f;
    for (int b = 0; b < 2000; ++b) { in.speedMs = std::fmin(20.f, b * 0.02f); e.render(in, l.data(), r.data(), 256); }
    int gear = e.readout().gear;
    e.setSound(Sound::HYPERCAR);
    e.render(in, l.data(), r.data(), 256);
    CHECK(gear > 1 && e.readout().gear == gear, "same sound reset the gearbox (%d -> %d)", gear, e.readout().gear);
}

int main(int argc, char** argv) {
    std::string fx = argc > 1 ? argv[1] : "";
    gearboxGentleNeverHuntsOverAnyRevRange();
    gearboxFlooredBouncesOffTheLimiter();
    gearboxColdIdleSettles();
    testDriveReaches100AndStops();
    sameSoundAgainKeepsTheGearbox();
    everySoundPlaysCleanly(fx);
    std::printf(failures ? "%d FAILED\n" : "all passed\n", failures);
    return failures ? 1 : 0;
}
