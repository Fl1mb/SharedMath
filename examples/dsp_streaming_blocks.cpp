#include <DSP/dsp.h>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <vector>

int main() {
    constexpr double sampleRate = 48000.0;
    auto signal = SharedMath::DSP::sineWave(1000.0, sampleRate, 4096);
    auto coeffs = SharedMath::DSP::designFIRLowPassHz(64, 3000.0, sampleRate);

    SharedMath::DSP::FIRFilter filter(coeffs);
    std::vector<double> output;
    output.reserve(signal.size());

    constexpr size_t blockSize = 256;
    for (size_t start = 0; start < signal.size(); start += blockSize) {
        const size_t end = std::min(start + blockSize, signal.size());
        std::vector<double> block(signal.begin() + static_cast<std::ptrdiff_t>(start),
                                  signal.begin() + static_cast<std::ptrdiff_t>(end));
        auto y = filter.processBlock(block);
        output.insert(output.end(), y.begin(), y.end());
    }

    std::cout << "Processed samples: " << output.size() << "\n";
    return 0;
}
