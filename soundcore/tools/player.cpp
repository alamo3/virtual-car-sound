// The Pi's sound player: the native sound engine straight to an ALSA device (a USB sound card or
// a soundbar's USB input), fed by the Python process that reads the car.
//
//   sc_player [--socket /tmp/vcs-sound.sock] [--device default] [--effects DIR]
//             [--sound flat6|hypercar|turbofan|warp] [--period 256] [--periods 3]
//
// On the socket (UNIX datagrams) it takes
//   - the car's data: the same 10-byte packet the Pi sends the phone over Bluetooth
//     (ble_control.py: version 1, flags, speed x100, pedal x200, motor rpm, current x10, seq)
//   - commands, as text: "sound <name>", "volume <0..1>", "gearshifts <0|1>",
//     "testdrive start|stop", "rev <0..1>" (revving in place; "rev off" to stop)
// The speed arrives in whole km/h: it's smoothed, and the lag that adds taken back out with the
// acceleration, as Open Blue does (PiLink). With no fresh data it fades out (car off, adapter
// gone) and comes back when the data does.
#include <alsa/asoundlib.h>
#include <sched.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "engine.h"

using namespace sc;
using Clock = std::chrono::steady_clock;

static volatile sig_atomic_t g_stop = 0;
static void onSignal(int) { g_stop = 1; }

static bool soundByName(const std::string& n, Sound& out) {
    if (n == "flat6" || n == "Flat-6") { out = Sound::FLAT6; return true; }
    if (n == "hypercar" || n == "Hypercar") { out = Sound::HYPERCAR; return true; }
    if (n == "turbofan" || n == "Turbofan") { out = Sound::TURBOFAN; return true; }
    if (n == "warp" || n == "Warp bubble") { out = Sound::WARP; return true; }
    return false;
}

// the car's data as Open Blue's PiLink sees it: whole km/h smoothed, lag taken back out
struct Car {
    static constexpr float SPEED_SMOOTH_SEC = 0.15f, STALE_SEC = 0.5f;
    float speedKmh = 0, pedal = 0, accel = 0, smooth = -1;
    int motorRpm = 0;
    bool reversing = false, piFresh = false;
    float lastRaw = -1;
    Clock::time_point lastAt{}, lastPacket{};

    void packet(const uint8_t* b) {
        auto now = Clock::now();
        uint8_t flags = b[1];
        float raw = (uint16_t)(b[2] | b[3] << 8) / 100.f;
        pedal = std::min(1.f, b[4] / 200.f);
        motorRpm = (int16_t)(b[5] | b[6] << 8);
        piFresh = flags & 1;
        reversing = flags & 2;
        if (lastRaw >= 0) {
            float dt = std::chrono::duration<float>(now - lastAt).count();
            if (dt > 0) {
                float a = (raw - lastRaw) / 3.6f / dt;
                accel += (a - accel) * std::min(1.f, dt / 0.3f);
                smooth += (raw - smooth) * std::min(1.f, dt / SPEED_SMOOTH_SEC);
            }
        } else {
            smooth = raw;
        }
        lastRaw = raw;
        lastAt = lastPacket = now;
        speedKmh = raw < 0.5f ? 0.f : std::max(0.f, smooth + accel * 3.6f * SPEED_SMOOTH_SEC);
    }
    bool fresh() const { return piFresh && std::chrono::duration<float>(Clock::now() - lastPacket).count() < STALE_SEC; }
};

static int openSocket(const std::string& path) {
    int fd = socket(AF_UNIX, SOCK_DGRAM | SOCK_NONBLOCK, 0);
    if (fd < 0) { perror("socket"); return -1; }
    sockaddr_un a{};
    a.sun_family = AF_UNIX;
    std::snprintf(a.sun_path, sizeof a.sun_path, "%s", path.c_str());
    unlink(path.c_str());
    if (bind(fd, (sockaddr*)&a, sizeof a) < 0) { perror("bind"); close(fd); return -1; }
    return fd;
}

static snd_pcm_t* openPcm(const std::string& dev, unsigned& rate, snd_pcm_uframes_t& period, unsigned periods) {
    snd_pcm_t* pcm = nullptr;
    int err = snd_pcm_open(&pcm, dev.c_str(), SND_PCM_STREAM_PLAYBACK, 0);
    if (err < 0) { std::fprintf(stderr, "can't open %s: %s\n", dev.c_str(), snd_strerror(err)); return nullptr; }
    snd_pcm_hw_params_t* hw;
    snd_pcm_hw_params_alloca(&hw);
    snd_pcm_hw_params_any(pcm, hw);
    snd_pcm_hw_params_set_access(pcm, hw, SND_PCM_ACCESS_RW_INTERLEAVED);
    snd_pcm_hw_params_set_format(pcm, hw, SND_PCM_FORMAT_S16_LE);
    snd_pcm_hw_params_set_channels(pcm, hw, 2);
    snd_pcm_hw_params_set_rate_near(pcm, hw, &rate, nullptr);
    snd_pcm_hw_params_set_period_size_near(pcm, hw, &period, nullptr);
    snd_pcm_uframes_t buffer = period * periods;
    snd_pcm_hw_params_set_buffer_size_near(pcm, hw, &buffer);
    if ((err = snd_pcm_hw_params(pcm, hw)) < 0) {
        std::fprintf(stderr, "can't set up %s: %s\n", dev.c_str(), snd_strerror(err));
        snd_pcm_close(pcm);
        return nullptr;
    }
    std::fprintf(stderr, "audio: %s, %u Hz, %lu-frame periods, %lu-frame buffer (%.1f ms)\n", dev.c_str(), rate,
                 (unsigned long)period, (unsigned long)buffer, 1000.0 * buffer / rate);
    return pcm;
}

int main(int argc, char** argv) {
    std::string sockPath = "/tmp/vcs-sound.sock", dev = "default", fx, soundName = "flat6";
    snd_pcm_uframes_t period = 256;
    unsigned periods = 3;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--socket") sockPath = v;
        else if (k == "--device") dev = v;
        else if (k == "--effects") fx = v;
        else if (k == "--sound") soundName = v;
        else if (k == "--period") period = std::stoul(v);
        else if (k == "--periods") periods = (unsigned)std::stoul(v);
        else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); return 2; }
    }
    if (fx.empty()) {                                        // effects/ next to the build directory
        std::string exe = argv[0];
        fx = exe.substr(0, exe.find_last_of('/') + 1) + "../effects";
    }
    signal(SIGINT, onSignal);
    signal(SIGTERM, onSignal);

    unsigned rate = 48000;
    snd_pcm_t* pcm = openPcm(dev, rate, period, periods);
    if (!pcm) return 1;
    int fd = openSocket(sockPath);
    if (fd < 0) return 1;
    sched_param sp{};
    sp.sched_priority = 50;
    if (sched_setscheduler(0, SCHED_FIFO, &sp) != 0) std::fprintf(stderr, "(no real-time priority: running at normal priority)\n");

    Engine engine((int)rate, fx);
    Sound snd;
    if (soundByName(soundName, snd)) engine.setSound(snd);
    std::fprintf(stderr, "sound: %s, effects from %s\n", info(engine.sound()).name, fx.c_str());

    Car car;
    float rev = -1.f;
    const int frames = (int)period;
    std::vector<float> l(frames), r(frames);
    std::vector<int16_t> pcmBuf(frames * 2);
    uint8_t msg[512];
    std::string lastCmd[8];             // the last of each command, so repeats aren't logged
    auto lastLog = Clock::now();
    double cpu = 0, audio = 0;

    while (!g_stop) {
        // drain the socket: the latest car data, any commands
        ssize_t n;
        while ((n = recv(fd, msg, sizeof msg - 1, 0)) > 0) {
            if (n == 10 && msg[0] == 1) { car.packet(msg); continue; }
            msg[n] = 0;
            std::string text((char*)msg);
            std::string cmd = text.substr(0, text.find(' ')), arg = text.find(' ') == std::string::npos ? "" : text.substr(text.find(' ') + 1);
            if (cmd == "sound" && soundByName(arg, snd)) engine.setSound(snd);
            else if (cmd == "volume") engine.setVolume(std::clamp(std::stof(arg), 0.f, 1.f));
            else if (cmd == "gearshifts") engine.setGearshifts(arg == "1" || arg == "on" || arg == "true");
            else if (cmd == "testdrive") arg == "start" ? engine.startTestDrive() : engine.stopTestDrive();
            else if (cmd == "rev") rev = arg == "off" ? -1.f : std::clamp(std::stof(arg), 0.f, 1.f);
            else { std::fprintf(stderr, "unknown command: %s\n", text.c_str()); continue; }
            int slot = (int)(std::hash<std::string>{}(cmd) % 8);
            if (lastCmd[slot] != text) { lastCmd[slot] = text; std::fprintf(stderr, "command: %s\n", text.c_str()); }
        }
        // the car's data while it's fresh; stale (car off, adapter gone), standing still
        const bool fresh = car.fresh();
        Inputs in;
        if (fresh) {
            in.pedal = car.reversing ? 0.f : car.pedal;
            in.speedMs = car.speedKmh / 3.6f;
            in.accel = car.reversing ? 0.f : car.accel;
            in.motorRpm = car.motorRpm;
            in.reversing = car.reversing;
        }
        in.revThrottle = rev;
        engine.setActive(fresh || engine.testDriving() || rev >= 0.f);

        auto t0 = Clock::now();
        engine.render(in, l.data(), r.data(), frames);
        for (int i = 0; i < frames; ++i) {
            pcmBuf[2 * i] = (int16_t)std::lround(std::clamp(l[i], -1.f, 1.f) * 32767);
            pcmBuf[2 * i + 1] = (int16_t)std::lround(std::clamp(r[i], -1.f, 1.f) * 32767);
        }
        cpu += std::chrono::duration<double>(Clock::now() - t0).count();
        audio += (double)frames / rate;
        snd_pcm_sframes_t w = snd_pcm_writei(pcm, pcmBuf.data(), frames);
        if (dev == "null") {                     // the null device doesn't wait: keep real time (a test bench)
            static auto start = Clock::now();
            static double played = 0;
            played += (double)frames / rate;
            auto due = start + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(played));
            std::this_thread::sleep_until(due);
        }
        if (w < 0) {
            if (w == -EPIPE) std::fprintf(stderr, "underrun\n");
            if (snd_pcm_recover(pcm, (int)w, 1) < 0) { std::fprintf(stderr, "audio device lost: %s\n", snd_strerror((int)w)); break; }
        }
        if (std::chrono::duration<float>(Clock::now() - lastLog).count() >= 5.f) {
            lastLog = Clock::now();
            const Readout& ro = engine.readout();
            std::fprintf(stderr, "%s | %3.0f km/h pedal %3.0f %% | gear %d %5.0f rpm%s%s | car %s | cpu %.1f %%\n",
                         info(engine.sound()).name, ro.kmh, car.pedal * 100, ro.gear, ro.rpm, ro.limiter ? " LIMITER" : "",
                         ro.testDrive ? (std::string(" | test drive: ") + ro.testDrive).c_str() : "",
                         car.fresh() ? "live" : "no data", 100.0 * cpu / std::max(audio, 1e-9));
            cpu = audio = 0;
        }
    }
    // fade out, then stop
    engine.setActive(false);
    for (int k = 0; k < 20; ++k) {
        engine.render(Inputs{}, l.data(), r.data(), frames);
        for (int i = 0; i < frames; ++i) { pcmBuf[2 * i] = (int16_t)(l[i] * 32767); pcmBuf[2 * i + 1] = (int16_t)(r[i] * 32767); }
        if (snd_pcm_writei(pcm, pcmBuf.data(), frames) < 0) break;
    }
    snd_pcm_drain(pcm);
    snd_pcm_close(pcm);
    close(fd);
    unlink(sockPath.c_str());
    return 0;
}
