#include <DSP/dsp.h>

#include <iostream>

int main() {
    auto signal = SharedMath::DSP::sineWave(440.0, 48000.0, 4800);
    auto resampled = SharedMath::DSP::resampleTo(signal, 48000, 16000);

    std::cout << "Input samples: " << signal.size() << "\n";
    std::cout << "Output samples: " << resampled.size() << "\n";
    return 0;
}
