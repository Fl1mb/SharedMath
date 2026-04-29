#include <DSP/dsp.h>

#include <algorithm>
#include <iostream>

int main() {
    constexpr double sampleRate = 48000.0;
    auto signal = SharedMath::DSP::sineWave(1200.0, sampleRate, 4096);
    auto psd = SharedMath::DSP::welchPSD(signal, sampleRate, 1024);

    auto peakIt = std::max_element(psd.psd.begin() + 1, psd.psd.end());
    const size_t peakBin = static_cast<size_t>(peakIt - psd.psd.begin());

    std::cout << "Dominant frequency: "
              << psd.frequencies[peakBin] << " Hz\n";
    return 0;
}
