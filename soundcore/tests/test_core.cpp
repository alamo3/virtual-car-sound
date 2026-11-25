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
#include "speedtracker.h"

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
        // the player needs 1x; 4x leaves room for the rest of the Pi (a Pi 3B+ runs these ~6-15x)
        CHECK(realtime > 4, "%s: only %.1fx real time", info(snd).name, realtime);
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

// the tires: silent driving straight, at a parking-lot crawl and in an ordinary bend; scrub and
// then a squeal as the cornering force nears the limit at speed, loudest on the outside of the turn
static void tiresSquealOnlyHardAndFast() {
    auto measure = [](float kmh, float lateral, double& outL, double& outR) {
        Tires t(48000);
        std::vector<float> l(256), r(256);
        double el = 0, er = 0;
        for (int b = 0; b < 400; ++b) {
            std::fill(l.begin(), l.end(), 0.f); std::fill(r.begin(), r.end(), 0.f);
            t.render(l.data(), r.data(), 256, lateral, kmh / 3.6f, 0.8f);
            if (b >= 200) for (int i = 0; i < 256; ++i) { el += l[i] * l[i]; er += r[i] * r[i]; }
        }
        outL = 10 * std::log10(el / (200 * 256) + 1e-12); outR = 10 * std::log10(er / (200 * 256) + 1e-12);
        return t.squeal();
    };
    double l, r;
    measure(80, 0, l, r);          CHECK(l < -100 && r < -100, "tires straight: %.0f dB", l);
    measure(10, 7, l, r);          CHECK(l < -100, "tires at a crawl: %.0f dB", l);
    measure(60, 2.5f, l, r);       CHECK(l < -100, "tires in an ordinary bend (0.25 g): %.0f dB", l);
    float sq = measure(60, 3.9f, l, r);
    std::printf("tires 0.4 g at 60: squeal %.2f, %.1f / %.1f dB\n", sq, l, r);
    sq = measure(80, 7.f, l, r);
    std::printf("tires 0.7 g at 80 left: squeal %.2f, %.1f / %.1f dB\n", sq, l, r);
    CHECK(sq > 0.9f, "tires at 0.7 g: squeal only %.2f", sq);
    CHECK(r > l + 3, "cornering left, the outside (right) tires should be louder: %.1f / %.1f", l, r);
    CHECK(r > -16 && r < -8, "tire squeal level %.1f dB", r);
}

// the speed tracker: whole-km/h readings 30 a second in, a smooth speed out (as SpeedTrackerTest.kt)
static void speedTrackerIsSmooth() {
    auto truth = [](float t) { if (t < 1) return 0.f; float v = 3.6f * 3.f * 6.f * (1.f - std::exp(-(t - 1.f) / 6.f));
                               if (t > 10) v -= 3.6f * 1.5f * (t - 10.f); return std::max(0.f, v); };
    SpeedTracker tr;
    float blk = 256.f / 48000, t = 0, next = 0, prev = 0, prevTrue = 0, worst = 0, err = 0, last = -1; int n = 0;
    while (t < 28) {
        if (t >= next) { tr.reading(std::round(truth(t)), t); next += 0.035f; }
        float s = tr.next(t, blk), tv = truth(t);
        if (t > 2 && t < 13) { worst = std::max(worst, std::fabs((s - prev) - (tv - prevTrue))); err += s - tv; n++; }
        prev = s; prevTrue = tv; last = s; t += blk;
    }
    std::printf("speed tracker: worst step %.3f km/h, mean error %+.2f km/h\n", worst, err / n);
    CHECK(worst < 0.06f, "speed tracker steps %.3f km/h", worst);
    CHECK(std::fabs(err / n) < 0.4f, "speed tracker error %.2f", err / n);
    CHECK(last == 0.f, "speed tracker doesn't stop at zero (%.2f)", last);
}

// cruise settle: a steady cruise eases the engine down ~6 dB; a press brings it straight back
static void cruiseSettles() {
    auto run = [](bool on, std::vector<float>& level) {
        Engine e(48000); e.setSound(Sound::FLAT6); e.skipChime(); e.setCruiseSettle(on);
        std::vector<float> l(256), r(256);
        float v = 25.f;                                     // 90 km/h
        for (int b = 0; b < 3300; ++b) {                    // 16 s cruise, then a press at 16 s
            Inputs in; float t = b * 256.f / 48000;
            in.pedal = t < 16.f ? 0.25f : 0.6f; in.speedMs = v; in.accel = t < 16.f ? 0.f : 1.5f;
            e.render(in, l.data(), r.data(), 256);
            double en = 0; for (int i = 0; i < 256; ++i) en += l[i] * l[i] + r[i] * r[i];
            level.push_back((float)(10 * std::log10(en / 512 + 1e-12)));
        }
        return e.readout().settled;
    };
    std::vector<float> on, off;
    run(true, on); run(false, off);
    auto avg = [](const std::vector<float>& x, float a, float b) { double s = 0; int n = 0;
        for (size_t i = (size_t)(a * 187.5f); i < (size_t)(b * 187.5f) && i < x.size(); ++i) { s += x[i]; n++; } return s / n; };
    double settled = avg(on, 12, 16) - avg(off, 12, 16), early = avg(on, 1, 3) - avg(off, 1, 3), back = avg(on, 16.4f, 16.8f) - avg(off, 16.4f, 16.8f);
    std::printf("cruise settle: %+.1f dB at a steady cruise, %+.1f dB in the first seconds, %+.1f dB 0.4 s after a press\n", settled, early, back);
    CHECK(settled < -6.5 && settled > -9.5, "cruise settle %.1f dB", settled);
    CHECK(std::fabs(early) < 0.5, "settled too early (%.1f dB)", early);
    CHECK(back > -1.0, "not back after a press (%.1f dB)", back);
}

// idle settle: stopped off the pedal, the engine eases ~9 dB down; touching the pedal brings it back
static void idleSettles() {
    auto run = [](bool on, std::vector<float>& level) {
        Engine e(48000); e.setSound(Sound::HYPERCAR); e.skipChime(); e.setCruiseSettle(on);
        std::vector<float> l(256), r(256);
        for (int b = 0; b < 2500; ++b) {                    // 12 s stopped, then the pedal at 12 s
            Inputs in; float t = b * 256.f / 48000;
            in.pedal = t < 12.f ? 0.f : 0.3f; in.speedMs = 0.f;
            e.render(in, l.data(), r.data(), 256);
            double en = 0; for (int i = 0; i < 256; ++i) en += l[i] * l[i] + r[i] * r[i];
            level.push_back((float)(10 * std::log10(en / 512 + 1e-12)));
        }
    };
    std::vector<float> on, off;
    run(true, on); run(false, off);
    auto avg = [](const std::vector<float>& x, float a, float b) { double s = 0; int n = 0;
        for (size_t i = (size_t)(a * 187.5f); i < (size_t)(b * 187.5f) && i < x.size(); ++i) { s += x[i]; n++; } return s / n; };
    double settled = avg(on, 9, 12) - avg(off, 9, 12), early = avg(on, 0.5f, 2.5f) - avg(off, 0.5f, 2.5f), back = avg(on, 12.3f, 12.6f) - avg(off, 12.3f, 12.6f);
    std::printf("idle settle: %+.1f dB stopped, %+.1f dB in the first seconds, %+.1f dB 0.3 s after the pedal\n", settled, early, back);
    CHECK(settled < -7.5 && settled > -10.5, "idle settle %.1f dB", settled);
    CHECK(std::fabs(early) < 0.5, "idle settled too early (%.1f dB)", early);
    CHECK(back > -1.0, "not back after the pedal (%.1f dB)", back);
}

int main(int argc, char** argv) {
    std::string fx = argc > 1 ? argv[1] : "";
    gearboxGentleNeverHuntsOverAnyRevRange();
    gearboxFlooredBouncesOffTheLimiter();
    gearboxColdIdleSettles();
    testDriveReaches100AndStops();
    sameSoundAgainKeepsTheGearbox();
    tiresSquealOnlyHardAndFast();
    speedTrackerIsSmooth();
    cruiseSettles();
    idleSettles();
    everySoundPlaysCleanly(fx);
    std::printf(failures ? "%d FAILED\n" : "all passed\n", failures);
    return failures ? 1 : 0;
}
