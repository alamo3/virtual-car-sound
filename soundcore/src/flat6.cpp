// Flat-6: a layered boxer six. Each firing is a smooth pressure pulse (a half-sine about half the
// firing interval wide, as Open Blue's piston engines: no sharp clicks, so no saw-wave buzz) into
// its bank's exhaust -- a low-pass body that opens with load, an exhaust and a pipe resonance,
// turbulence riding on each pulse -- the right bank weaker and tuned apart (the boxer's lumpy
// half-order note). Then a muffler that opens with load; intake howl, valvetrain tick, overrun
// crackle; the gearbox's torque handover and limiter cuts drop the firing out.
#include <algorithm>
#include <cmath>
#include <vector>

#include "dsp.h"
#include "voices.h"

namespace sc {

namespace {
constexpr int FIRING_ORDER[6] = {0, 5, 1, 3, 2, 4};
constexpr float CYL_GAIN[6] = {1.00f, 0.93f, 1.05f, 0.97f, 1.02f, 0.95f};
// the exhaust (per bank): body low-pass, exhaust and pipe resonances, mixed as the piston engines
constexpr float EXHAUST_HZ = 105.f, PIPE_HZ = 330.f, BODY_GAIN = 1.1f, EXHAUST_GAIN = 6.f, PIPE_GAIN = 2.2f;
// each pulse: a half-sine this wide (PULSE_WIDTH of the firing interval, within PULSE_MIN..MAX s),
// varying in strength by ROUGHNESS, landing early or late by FIRE_JITTER (of an interval)
constexpr float PULSE_WIDTH = 0.5f, PULSE_MIN = 0.0012f, PULSE_MAX = 0.006f, ROUGHNESS = 0.22f;
constexpr float FIRE_JITTER = 0.006f, MUFFLE_LO = 2000.f, MUFFLE_HI = 5000.f;
constexpr float BANK_LEVEL = 0.115f;      // the banks into the rest at the level the old exhaust had
constexpr float SAT_IN = 0.35f;          // how hot the exhaust runs into its saturation
constexpr float RIGHT_TUNE = 1.10f, WEAK_LOW = 0.66f, WEAK_HIGH = 0.86f;
constexpr float TRIM = 5.7f, POP_SEC = 0.03f;
constexpr float RASP = 0.06f;            // the turbulent rasp on each pulse, at full load
}  // namespace

class Flat6Voice : public LayeredVoice {
public:
    static constexpr float IDLE = 900.f, REDLINE = 8500.f;

    explicit Flat6Voice(int sr)
        : LayeredVoice(sr), noise_(11), pink_(12), level_(0.02f, sr), intakeLevel_(0.05f, sr) {
        turbDecay_ = std::exp(-1.f / (0.0030f * sr));
        for (int c = 0; c < 2; ++c) { shelf_[c].lowShelf(140.f, 0.7f, 1.f, sr); hp_[c].highPass(28.f, 0.707f, sr); }
        int n = (int)(POP_SEC * sr);
        for (int k = 0; k < 6; ++k) {
            pops_[k].resize(n);
            Biq bp; bp.bandPass(1800.f, 0.7f, sr);
            Noise nz(100 + k);
            for (int i = 0; i < n; ++i) {
                float t = (float)i / sr;
                pops_[k][i] = bp.process(nz.gauss()) * std::exp(-t / 0.006f) + sin1(85.0 * t) * std::exp(-t / 0.012f) * 0.8f;
            }
            popPos_[k] = -1;
        }
    }

    void render(float* out, float* outR, int frames) override {
        if (!has_) { std::fill(out, out + frames, 0.f); std::fill(outR, outR + frames, 0.f); return; }
        const EngineState& s = st_;
        float rpm = std::max(s.rpm, IDLE * 0.95f);
        float rpmN = clamp01((rpm - IDLE) / (REDLINE - IDLE));
        float load = clamp01(s.load);
        float overrun = clamp01(s.overrun);
        float fireHz = rpm / 20.f;                               // 6 cylinders, 3 firings a rev
        float bright = clamp01(0.15f + 0.85f * load * (0.5f + 0.5f * rpmN)) * (1.f - 0.6f * overrun);
        float weak = lerp(WEAK_LOW, WEAK_HIGH, rpmN);
        float fireAmp = 1.f - 0.85f * clamp01(s.shifting) * (load > 0.3f ? 1.f : 0.f);
        if (s.cut) fireAmp = 0.12f;
        for (int b = 0; b < 2; ++b) {                          // the right bank tuned apart
            float tune = b == 1 ? RIGHT_TUNE : 1.f;
            body_[b].lowPass((300.f + 2400.f * load * bright + rpm * 0.25f) * tune, 0.7f, sr_);
            exh_[b].bandPass(EXHAUST_HZ * tune, 0.9f, sr_);
            pipe_[b].bandPass(PIPE_HZ * tune, 2.2f, sr_);
        }
        const float pulseW = clampf(PULSE_WIDTH / fireHz, PULSE_MIN, PULSE_MAX);
        const float drive = 1.2f + 1.8f * load;                // pushed harder, a little harsher
        const float driveNorm = 1.f / std::tanh(drive);
        grit_.bandPass(2200.f, 0.8f, sr_);
        float mute = lerp(MUFFLE_LO, MUFFLE_HI, load * (0.35f + 0.65f * rpmN)) * (1.f - 0.45f * overrun);
        for (int c = 0; c < 2; ++c) {
            tailpipe_[c].peak(92.f, 2.5f, 3.f, sr_);
            header_[c].peak(270.f, 2.f, 3.f, sr_);
            muffler_[c].lowPass(mute, 0.8f, sr_);
        }
        float lvlTarget = (0.45f + 0.55f * std::pow(load, 1.3f)) * (0.55f + 0.45f * rpmN) * (1.f - overrun) +
                          0.10f * overrun * (0.4f + 0.6f * rpmN);
        intakeBp_.bandPass(lerp(700.f, 2600.f, rpmN), 1.1f, sr_);
        float intakeTarget = std::pow(load, 1.4f) * (0.25f + 0.75f * std::pow(rpmN, 1.3f)) * 0.55f;
        tickBp_.bandPass(5200.f, 2.f, sr_);
        float pulseTau = std::max(0.25f / fireHz, 0.0006f);

        if (liftOffsSeen_ >= 0 && s.liftOffs != liftOffsSeen_) crackleLeft_ = 1.6f;   // crackle after a hard lift
        liftOffsSeen_ = s.liftOffs;
        float popRate = (crackleLeft_ / 1.6f) * overrun * (8.f + 30.f * rpmN);
        crackleLeft_ = std::max(0.f, crackleLeft_ - (float)frames / sr_);

        float vol = volume * TRIM;
        for (int i = 0; i < frames; ++i) {
            // firing: start a pressure pulse in that cylinder's bank
            firePhase_ += fireHz / sr_;
            if (firePhase_ >= fireAt_) {
                firePhase_ -= fireAt_;
                fireAt_ = 1.0 + FIRE_JITTER * noise_.gauss();       // when the next one lands
                firing_++;
                sinceFire_ = 0.f;
                int cyl = FIRING_ORDER[firing_ % 6];
                bool right = cyl % 2 == 1;
                int b = right ? 1 : 0;
                pulseAmp_[b] = CYL_GAIN[cyl] * (right ? weak : 1.f) * (1.f + ROUGHNESS * noise_.white() * (0.6f + load)) * fireAmp *
                               (0.7f + 0.3f * load);   // (the level below follows load too)
                pulseT_[b] = 0.f;
                turb_[b] = pulseAmp_[b] * (0.08f + 0.92f * load);   // turbulence: mostly when it's working
            } else {
                sinceFire_ += 1.f / sr_;
            }
            float bankL = bank(0, pulseW, drive, driveNorm);
            float bankR = bank(1, pulseW, drive, driveNorm);
            // the rasp: turbulent flow riding on each exhaust pulse, added to it. (It used to
            // multiply the whole exhaust by noise -- heard as a distorted, rattling buzz.)
            float pulseEnv = std::exp(-sinceFire_ / pulseTau);
            float rasp = grit_.process(noise_.gauss()) * pulseEnv * RASP * load;
            float l = bankL * 0.62f + bankR * 0.38f + rasp;
            float r = bankL * 0.38f + bankR * 0.62f + rasp;
            l = muffler_[0].process(header_[0].process(tailpipe_[0].process(l)));
            r = muffler_[1].process(header_[1].process(tailpipe_[1].process(r)));
            float lv = level_.next(lvlTarget);
            l *= lv; r *= lv;
            // intake
            float roar = intakeBp_.process(pink_.next()) * (0.45f + 0.55f * pulseEnv) * intakeLevel_.next(intakeTarget);
            l += roar * 0.5f; r += roar * 0.5f;
            // mechanical: valvetrain tick on each pulse. (A pure 'timing chain' tone at 21x the
            // revs used to sit here: at a cruise it stood ~4 dB over the engine's own harmonics
            // around it -- heard as an electric-motor whine. A real chain is texture, not a tone.)
            float mech = tickBp_.process(noise_.gauss()) * pulseEnv * pulseEnv * pulseEnv * 0.05f * (0.2f + rpmN);
            l += mech * 0.5f; r += mech * 0.5f;
            // overrun crackle
            if (popRate > 0.f && noise_.chance(popRate / sr_)) {
                for (int k = 0; k < 6; ++k) if (popPos_[k] < 0) { popPos_[k] = 0; popGain_[k] = noise_.uniform(0.4f, 1.f) * 0.35f; break; }
            }
            popPan_ += 3.0 / sr_ / (2 * M_PI);
            for (int k = 0; k < 6; ++k) if (popPos_[k] >= 0) {
                float v = pops_[k][popPos_[k]] * popGain_[k];
                float pan = 0.5f + 0.2f * sin1(popPan_);
                l += v * pan; r += v * (1.f - pan);
                if (++popPos_[k] >= (int)pops_[k].size()) popPos_[k] = -1;
            }
            // in the cabin
            l = hp_[0].process(shelf_[0].process(l));
            r = hp_[1].process(shelf_[1].process(r));
            out[i] = softClip(l, 1.6f) * vol;
            outR[i] = softClip(r, 1.6f) * vol;
        }
    }

private:
    // one bank's exhaust: its current pulse (a half-sine) plus turbulence, through the body,
    // exhaust and pipe, driven a little harder with load
    float bank(int b, float pulseW, float drive, float driveNorm) {
        float e = pulseT_[b] < pulseW ? pulseAmp_[b] * sin1(0.5 * pulseT_[b] / pulseW) : 0.f;
        pulseT_[b] += 1.f / sr_;
        turb_[b] *= turbDecay_;
        float x = e + noise_.white() * turb_[b] * 0.35f;
        float y = body_[b].process(x) * BODY_GAIN + exh_[b].process(x) * EXHAUST_GAIN + pipe_[b].process(x) * PIPE_GAIN;
        // into the saturation at SAT_IN of its level (gentle at a cruise, biting when pushed), back
        // up after it: the rest of the voice sees the level it always did
        return std::tanh(y * SAT_IN * drive) * driveNorm / SAT_IN * BANK_LEVEL;
    }

    Noise noise_;
    Pink pink_;
    Biq body_[2], exh_[2], pipe_[2];
    float pulseT_[2] = {1.f, 1.f}, pulseAmp_[2] = {0, 0}, turb_[2] = {0, 0};
    float turbDecay_ = 0.f;                                        // turbulence dies in ~3 ms
    Biq grit_, tailpipe_[2], header_[2], muffler_[2], intakeBp_, tickBp_, shelf_[2], hp_[2];
    Smooth level_, intakeLevel_;
    double firePhase_ = 0, popPan_ = 0, fireAt_ = 1.0;
    long long firing_ = 0;
    float sinceFire_ = 1e9f;
    int liftOffsSeen_ = -1;
    float crackleLeft_ = 0;
    std::vector<float> pops_[6];
    int popPos_[6];
    float popGain_[6] = {0};
};

std::unique_ptr<LayeredVoice> makeFlat6(int sr) { return std::make_unique<Flat6Voice>(sr); }

}  // namespace sc
