#include "scripting/include/compiler.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstring>

static std::vector<int16_t> read_wav(const std::string &path) {
    std::ifstream f(path, std::ios::binary); if (!f) throw std::runtime_error("Missing impulse response: " + path);
    char id[4]; uint32_t size; f.read(id,4); f.read((char*)&size,4); f.read(id,4);
    while (f.read(id,4) && f.read((char*)&size,4)) {
        if (!std::memcmp(id,"data",4)) { std::vector<int16_t> data(size/2); f.read((char*)data.data(),size); return data; }
        f.seekg(size+(size&1),std::ios::cur);
    } throw std::runtime_error("No WAV data: " + path);
}
static void write_wav(const std::string &path, const std::vector<int16_t> &data) {
    std::ofstream f(path,std::ios::binary); uint32_t ds=data.size()*2, size=36+ds, fmt=16, sr=44100, rate=88200; uint16_t pcm=1,ch=1,align=2,bits=16;
    f.write("RIFF",4); f.write((char*)&size,4); f.write("WAVEfmt ",8); f.write((char*)&fmt,4); f.write((char*)&pcm,2); f.write((char*)&ch,2); f.write((char*)&sr,4); f.write((char*)&rate,4); f.write((char*)&align,2); f.write((char*)&bits,2); f.write("data",4); f.write((char*)&ds,4); f.write((char*)data.data(),ds);
}
int main(int argc, char **argv) {
    if (argc != 2) { std::cerr << "Usage: headless OUTPUT.wav\n"; return 2; }
    es_script::Compiler c; c.initialize();
    if (!c.compile("assets/main.mr")) { std::cerr << "Script compile failed; see error_log.log\n"; return 1; }
    auto result=c.execute(); auto *e=result.engine; if (!e) return 2;
    auto *v=result.vehicle; if(!v) { Vehicle::Parameters p; p.mass=units::mass(1597,units::kg); p.diffRatio=3.42; p.tireRadius=units::distance(10,units::inch); p.dragCoefficient=.25; p.crossSectionArea=3.3; p.rollingResistance=2000; v=new Vehicle; v->initialize(p); }
    auto *tr=result.transmission; if(!tr) { const double ratios[]={2.97,2.07,1.43,1,.84,.56}; Transmission::Parameters p; p.GearCount=6; p.GearRatios=ratios; p.MaxClutchTorque=units::torque(1000,units::ft_lb); tr=new Transmission; tr->initialize(p); }
    auto *sim=e->createSimulator(v,tr); sim->setSimulationFrequency(e->getSimulationFrequency());
    auto ap=sim->synthesizer().getAudioParameters(); ap.volume=0.25; ap.inputSampleNoise=e->getInitialJitter(); ap.airNoise=e->getInitialNoise(); ap.dF_F_mix=e->getInitialHighFrequencyGain(); sim->synthesizer().setAudioParameters(ap);
    for(int i=0;i<e->getExhaustSystemCount();++i) { auto *ir=e->getExhaustSystem(i)->getImpulseResponse(); auto raw=read_wav(ir->getFilename()); sim->synthesizer().initializeImpulseResponse(raw.data(), raw.size(), ir->getVolume(), i); std::cout << "IR " << ir->getFilename() << "\n"; }
    e->getIgnitionModule()->m_enabled=true;
    std::vector<int16_t> flush(44100); sim->readAudioOutput(flush.size(),flush.data());
    std::vector<int16_t> out; int16_t chunk[2048];
    for(int frame=0;frame<1200;++frame) {
        double t=frame*.01; sim->m_starterMotor.m_enabled=t<1.5; double pedal=(t<2? .0 : t<4? .12 : t<6? .22 : t<8? .02 : t<10? .15 : .0); e->setSpeedControl(pedal);
        sim->startFrame(.01); while(sim->simulateStep()){} sim->endFrame(); sim->synthesizer().renderAudio();
        int n=sim->readAudioOutput(2048,chunk); out.insert(out.end(),chunk,chunk+n);
        if(frame%100==0) std::cout << "t=" << t << " rpm=" << e->getRpm() << " samples=" << out.size() << std::endl;
    }
    write_wav(argv[1],out); std::cout << "Wrote " << out.size() << " samples\n";
    return 0;
}
