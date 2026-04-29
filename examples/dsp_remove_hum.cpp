#include <DSP/dsp.h>

#include <iostream>

int main() {
    constexpr double sampleRate = 48000.0;
    auto wanted = SharedMath::DSP::sineWave(1000.0, sampleRate, 4096, 0.8);
    auto hum = SharedMath::DSP::sineWave(60.0, sampleRate, 4096, 0.2);

    for (size_t i = 0; i < wanted.size(); ++i) wanted[i] += hum[i] + 0.1;

    auto noDC = SharedMath::DSP::removeDC(wanted);
    auto notch = SharedMath::DSP::designRBJNotchHz(60.0, sampleRate, 30.0);
    auto clean = SharedMath::DSP::applyIIR(noDC, {notch});

    std::cout << "Input RMS: " << SharedMath::DSP::rms(wanted) << "\n";
    std::cout << "Clean RMS: " << SharedMath::DSP::rms(clean) << "\n";
    return 0;
}
