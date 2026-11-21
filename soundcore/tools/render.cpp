// Renders a sound over the standard test drive (or the in-app test drive) to a 16-bit WAV and
// reports its levels and how much faster than real time it ran.
//
//   sc_render <flat6|hypercar|turbofan|warp> out.wav [effects_dir] [--testdrive]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "drive.h"
#include "engine.h"

using namespace sc;

static void writeWav(const char* path, const std::vector<float>& x, int sr) {
    std::ofstream f(path, std::ios::binary);
    auto u32 = [&](uint32_t v) { f.write((const char*)&v, 4); };
    auto u16 = [&](uint16_t v) { f.write((const char*)&v, 2); };
    f.write("RIFF", 4); u32(36 + x.size() * 2); f.write("WAVEfmt ", 8);
    u32(16); u16(1); u16(2); u32(sr); u32(sr * 4); u16(4); u16(16);
    f.write("data", 4); u32(x.size() * 2);
    for (float s : x) { int16_t v = (int16_t)std::lround(std::fmax(-1.f, std::fmin(1.f, s)) * 32767); f.write((const char*)&v, 2); }
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <flat6|hypercar|turbofan|warp> out.wav [effects_dir] [--testdrive]\n", argv[0]); return 2; }
    std::string name = argv[1];
    Sound s = name == "hypercar" ? Sound::HYPERCAR : name == "turbofan" ? Sound::TURBOFAN : name == "warp" ? Sound::WARP : Sound::FLAT6;
    std::string fx = argc > 3 && argv[3][0] != '-' ? argv[3] : "";
    bool testDrive = argc > 3 && std::strcmp(argv[argc - 1], "--testdrive") == 0;
    const int sr = 48000, frames = 256;
    Engine e(sr, fx);
    e.setSound(s);
    if (testDrive) e.startTestDrive();
    DriveSim sim;
    const float dt = (float)frames / sr, seconds = testDrive ? 30.f : 52.f;
    std::vector<float> out;
    std::vector<float> l(frames), r(frames);
    double cpu = 0;
    for (float t = 0; t < seconds; t += dt) {
        Inputs in = sim.step(t, dt);
        auto t0 = std::chrono::steady_clock::now();
        e.render(in, l.data(), r.data(), frames);
        cpu += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        for (int i = 0; i < frames; ++i) { out.push_back(l[i]); out.push_back(r[i]); }
        if (testDrive && !e.testDriving() && t > 2) break;
    }
    writeWav(argv[2], out, sr);
    auto rms = [&](float a, float b) {
        size_t i0 = (size_t)(a * sr) * 2, i1 = std::min(out.size(), (size_t)(b * sr) * 2); double en = 0;
        for (size_t i = i0; i < i1; ++i) en += out[i] * out[i];
        return 10 * std::log10(en / std::max<size_t>(1, i1 - i0) + 1e-12);
    };
    float peak = 0;
    for (float v : out) peak = std::fmax(peak, std::fabs(v));
    std::printf("%s: peak %.2f | idle %.1f, cruise %.1f, hard pull %.1f dBFS | %.0fx real time\n", info(s).name, peak,
                rms(0.5f, 2.5f), rms(12, 16), rms(19, 24), (out.size() / 2.0 / sr) / cpu);
    return 0;
}
