#include <gtest/gtest.h>
#include "DSP/FilterResponse.h"
#include "DSP/FIR.h"
#include "DSP/FFT.h"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>

using namespace SharedMath::DSP;

static constexpr size_t kNFFT = 512;
static constexpr double kFs   = 1000.0;

// ─────────────────────────────────────────────────────────────────────────────
// frequencyResponseFIR

TEST(FrequencyResponseFIR, OutputSizeMatchesNFFT) {
    // order=64 → 65 taps; nfft=512 → nextPow2(512)=512 → 257 bins
    auto h = designFIRLowPass(64, 0.3);
    auto H = frequencyResponseFIR(h, kNFFT, kFs);
    EXPECT_EQ(H.size(), kNFFT / 2 + 1);
}

TEST(FrequencyResponseFIR, ImpulseResponseIsFlat) {
    // H(ω) of a unit impulse at n=0 equals 1.0 everywhere
    std::vector<double> h = {1.0};
    auto H = frequencyResponseFIR(h, 64, kFs);
    for (const auto& c : H)
        EXPECT_NEAR(std::abs(c), 1.0, 1e-12);
}

TEST(FrequencyResponseFIR, FrequenciesMatchHelper) {
    auto h     = designFIRLowPass(32, 0.4);
    auto H     = frequencyResponseFIR(h, kNFFT, kFs);
    auto freqs = firResponseFrequencies(h, kNFFT, kFs);
    EXPECT_EQ(H.size(), freqs.size());
    EXPECT_DOUBLE_EQ(freqs.front(), 0.0);
    EXPECT_NEAR(freqs.back(), kFs / 2.0, 1.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// magnitudeResponseFIR — low-pass filter behaviour

TEST(MagnitudeResponseFIR, LowPassDCGainNearOne) {
    // Low-pass cutoff at 0.3 (normalized) → gain at DC should be ≈ 1
    auto h   = designFIRLowPass(64, 0.3);
    auto mag = magnitudeResponseFIR(h, kNFFT, kFs);
    EXPECT_NEAR(mag[0], 1.0, 0.01);
}

TEST(MagnitudeResponseFIR, LowPassNyquistIsAttenuated) {
    // Gain at Nyquist (last bin) should be very small for a low-pass
    auto h   = designFIRLowPass(64, 0.3);
    auto mag = magnitudeResponseFIR(h, kNFFT, kFs);
    EXPECT_LT(mag.back(), 0.01);
}

TEST(MagnitudeResponseFIR, HighPassPassbandNearNyquist) {
    // High-pass fc=0.7 → gain near Nyquist should be ≈ 1
    auto h   = designFIRHighPass(64, 0.7);
    auto mag = magnitudeResponseFIR(h, kNFFT, kFs);
    EXPECT_NEAR(mag.back(), 1.0, 0.05);
}

TEST(MagnitudeResponseFIR, HighPassDCIsAttenuated) {
    auto h   = designFIRHighPass(64, 0.7);
    auto mag = magnitudeResponseFIR(h, kNFFT, kFs);
    EXPECT_LT(mag[0], 0.01);
}

TEST(MagnitudeResponseFIR, BandPassPeakInBand) {
    auto h    = designFIRBandPass(64, 0.3, 0.5);
    auto mag  = magnitudeResponseFIR(h, kNFFT, kFs);
    auto freqs = firResponseFrequencies(h, kNFFT, kFs);

    // Peak should be between the two cutoffs (in Hz: 150..250 Hz at 1000 Hz)
    size_t pk = static_cast<size_t>(
        std::max_element(mag.begin(), mag.end()) - mag.begin());
    EXPECT_GT(freqs[pk], 100.0);
    EXPECT_LT(freqs[pk], 300.0);
}

TEST(MagnitudeResponseFIR, AllValuesNonNegative) {
    auto h   = designFIRLowPass(32, 0.4);
    auto mag = magnitudeResponseFIR(h, kNFFT, kFs);
    for (double v : mag) EXPECT_GE(v, 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// magnitudeResponseFIRDB

TEST(MagnitudeResponseFIRDB, DCGainNearZeroDBForLowPass) {
    auto h  = designFIRLowPass(64, 0.3);
    auto db = magnitudeResponseFIRDB(h, kNFFT, kFs);
    EXPECT_NEAR(db[0], 0.0, 0.1);   // ≈ 0 dB at DC
}

TEST(MagnitudeResponseFIRDB, StopbandIsNegativeDB) {
    auto h  = designFIRLowPass(64, 0.3);
    auto db = magnitudeResponseFIRDB(h, kNFFT, kFs);
    EXPECT_LT(db.back(), -20.0);    // well below −20 dB at Nyquist
}

// ─────────────────────────────────────────────────────────────────────────────
// phaseResponseFIR

TEST(PhaseResponseFIR, OutputSize) {
    auto h  = designFIRLowPass(32, 0.3);
    auto ph = phaseResponseFIR(h, kNFFT, kFs);
    EXPECT_EQ(ph.size(), kNFFT / 2 + 1);
}

TEST(PhaseResponseFIR, ImpulseResponseHasZeroPhase) {
    // unit impulse at n=0: H(ω)=1 → phase=0 everywhere
    std::vector<double> h = {1.0};
    auto ph = phaseResponseFIR(h, 64, kFs);
    for (double v : ph) EXPECT_NEAR(v, 0.0, 1e-12);
}

TEST(PhaseResponseFIR, SymmetricFIRLinearPhase) {
    // Symmetric FIR has linear phase in the passband.
    // Check that values are in [-π, π]
    auto h  = designFIRLowPass(64, 0.3);
    auto ph = phaseResponseFIR(h, kNFFT, kFs);
    for (double v : ph) {
        EXPECT_GE(v, -3.1416);
        EXPECT_LE(v,  3.1416);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// groupDelayFIR

TEST(GroupDelayFIR, OutputSize) {
    auto h  = designFIRLowPass(64, 0.3);
    auto gd = groupDelayFIR(h, kNFFT, kFs);
    EXPECT_EQ(gd.size(), kNFFT / 2 + 1);
}

TEST(GroupDelayFIR, SymmetricFIRPassbandDelay) {
    // Linear-phase symmetric FIR of order M → group delay = M/2 samples
    size_t order = 64;  // forced even inside designFIRLowPass
    auto h       = designFIRLowPass(order, 0.3);
    auto gd      = groupDelayFIR(h, kNFFT, kFs);
    auto freqs   = firResponseFrequencies(h, kNFFT, kFs);

    // Average group delay in the passband (0..cutoff = 0.3×500 = 150 Hz)
    double sum = 0.0;
    size_t cnt = 0;
    for (size_t k = 0; k < gd.size(); ++k) {
        if (freqs[k] <= 150.0) { sum += gd[k]; ++cnt; }
    }
    double meanGD = (cnt > 0) ? sum / cnt : 0.0;

    // Expected group delay = order/2 = 32 samples (tolerance ±1 sample)
    EXPECT_NEAR(meanGD, static_cast<double>(order) / 2.0, 1.0);
}

TEST(GroupDelayFIR, ImpulseResponseGroupDelayIsZero) {
    // Unit impulse at n=0: no delay
    std::vector<double> h = {1.0};
    auto gd = groupDelayFIR(h, 64, kFs);
    for (double v : gd) EXPECT_NEAR(v, 0.0, 1e-10);
}

TEST(GroupDelayFIR, DelayedImpulseHasConstantDelay) {
    // Impulse at position d has group delay = d samples everywhere
    size_t d = 5;
    std::vector<double> h(16, 0.0);
    h[d] = 1.0;
    auto gd = groupDelayFIR(h, 512, kFs);
    // Ignore bins where |H|≈0 (they're set to 0); all non-zero bins should be ≈ d
    // For a pure delay the magnitude is flat=1 everywhere, so all bins are valid
    for (double v : gd) EXPECT_NEAR(v, static_cast<double>(d), 1e-6);
}
