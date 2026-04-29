#include <DSP/dsp.h>

#include <iostream>

int main() {
    constexpr double sampleRate = 48000.0;
    auto low = SharedMath::DSP::sineWave(200.0, sampleRate, 4096, 0.6);
    auto mid = SharedMath::DSP::sineWave(1200.0, sampleRate, 4096, 1.0);
    auto high = SharedMath::DSP::sineWave(8000.0, sampleRate, 4096, 0.6);

    for (size_t i = 0; i < mid.size(); ++i) mid[i] += low[i] + high[i];

    auto sections = SharedMath::DSP::designButterworthBandPassHz(
        4, 800.0, 2000.0, sampleRate);
    auto filtered = SharedMath::DSP::applyIIR(mid, sections);

    std::cout << "Filtered samples: " << filtered.size() << "\n";
    std::cout << "Filtered RMS: " << SharedMath::DSP::rms(filtered) << "\n";
    return 0;
}
