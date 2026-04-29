#include <gtest/gtest.h>
#include "DSP/FilterDesign.h"
#include "DSP/FilterResponse.h"

#include <cmath>
#include <vector>
#include <stdexcept>

using namespace SharedMath::DSP;

static constexpr double kFs   = 8000.0;
static constexpr size_t kNFFT = 1024;

// ─────────────────────────────────────────────────────────────────────────────
// designFIRLowPassHz

TEST(FIRLowPassHz, DCGainNearOne) {
    auto h   = designFIRLowPassHz(64, 1000.0, kFs);
    auto mag = magnitudeResponseFIR(h, kNFFT, kFs);
    EXPECT_NEAR(mag[0], 1.0, 0.02);
}

TEST(FIRLowPassHz, NyquistAttenuated) {
    auto h   = designFIRLowPassHz(64, 1000.0, kFs);
    auto mag = magnitudeResponseFIR(h, kNFFT, kFs);
    EXPECT_LT(mag.back(), 0.01);
}

TEST(FIRLowPassHz, InvalidCutoffThrows) {
    EXPECT_THROW(designFIRLowPassHz(32, 0.0,    kFs), std::invalid_argument);
    EXPECT_THROW(designFIRLowPassHz(32, kFs,     kFs), std::invalid_argument);
    EXPECT_THROW(designFIRLowPassHz(32, kFs/2.0, kFs), std::invalid_argument);
}

TEST(FIRLowPassHz, MatchesNormalizedAPI) {
    double cutHz  = 1000.0;
    double normFc = cutHz / (kFs / 2.0);
    auto h1 = designFIRLowPassHz(64, cutHz, kFs);
    auto h2 = designFIRLowPass(64, normFc);
    ASSERT_EQ(h1.size(), h2.size());
    for (size_t i = 0; i < h1.size(); ++i)
        EXPECT_NEAR(h1[i], h2[i], 1e-12);
}

// ─────────────────────────────────────────────────────────────────────────────
// designFIRHighPassHz

TEST(FIRHighPassHz, NyquistGainNearOne) {
    auto h   = designFIRHighPassHz(64, 3000.0, kFs);
    auto mag = magnitudeResponseFIR(h, kNFFT, kFs);
    EXPECT_NEAR(mag.back(), 1.0, 0.05);
}

TEST(FIRHighPassHz, DCAttenuated) {
    auto h   = designFIRHighPassHz(64, 3000.0, kFs);
    auto mag = magnitudeResponseFIR(h, kNFFT, kFs);
    EXPECT_LT(mag[0], 0.01);
}

// ─────────────────────────────────────────────────────────────────────────────
// designFIRBandPassHz / BandStopHz

TEST(FIRBandPassHz, PeakInBand) {
    auto h     = designFIRBandPassHz(64, 1000.0, 2000.0, kFs);
    auto mag   = magnitudeResponseFIR(h, kNFFT, kFs);
    auto freqs = firResponseFrequencies(h, kNFFT, kFs);
    size_t pk  = static_cast<size_t>(
        std::max_element(mag.begin(), mag.end()) - mag.begin());
    EXPECT_GT(freqs[pk], 800.0);
    EXPECT_LT(freqs[pk], 2200.0);
}

TEST(FIRBandPassHz, InvalidBandThrows) {
    EXPECT_THROW(designFIRBandPassHz(32, 2000.0, 1000.0, kFs), std::invalid_argument);
    EXPECT_THROW(designFIRBandPassHz(32, 0.0,    1000.0, kFs), std::invalid_argument);
}

TEST(FIRBandStopHz, DCGainPreserved) {
    auto h   = designFIRBandStopHz(64, 1000.0, 2000.0, kFs);
    auto mag = magnitudeResponseFIR(h, kNFFT, kFs);
    EXPECT_NEAR(mag[0], 1.0, 0.05);
}

// ─────────────────────────────────────────────────────────────────────────────
// designButterworthLowPassHz / HighPassHz

TEST(ButterworthLowPassHz, ReturnsSections) {
    auto sections = designButterworthLowPassHz(4, 1000.0, kFs);
    EXPECT_EQ(sections.size(), 2u);  // order 4 → 2 biquads
}

TEST(ButterworthHighPassHz, ReturnsSections) {
    auto sections = designButterworthHighPassHz(4, 1000.0, kFs);
    EXPECT_EQ(sections.size(), 2u);
}

TEST(ButterworthBandPassHz, ReturnsSections) {
    // band-pass doubles order → 4 sections for order=4
    auto sections = designButterworthBandPassHz(4, 500.0, 2000.0, kFs);
    EXPECT_FALSE(sections.empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// designNotchHz

TEST(NotchHz, SupressesTargetFrequency) {
    double notchHz = 1000.0;
    auto bq = designNotchHz(notchHz, kFs, 10.0);

    // Apply to a pure sine at notchHz and check RMS is near zero
    const size_t N = 4096;
    const double omega = 2.0 * 3.14159265358979323846 * notchHz / kFs;
    std::vector<double> sig(N);
    for (size_t i = 0; i < N; ++i) sig[i] = std::sin(omega * i);

    // Run through biquad (warm up with first half, measure second half)
    double s1 = 0.0, s2 = 0.0;
    double rms_out = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < N; ++i) {
        double y = bq.process(sig[i], s1, s2);
        if (i >= N / 2) { rms_out += y * y; ++count; }
    }
    rms_out = std::sqrt(rms_out / count);
    EXPECT_LT(rms_out, 0.01);
}

TEST(NotchHz, PassesOtherFrequencies) {
    double notchHz = 1000.0;
    double passHz  = 200.0;
    auto bq = designNotchHz(notchHz, kFs, 10.0);

    const size_t N = 4096;
    const double omega = 2.0 * 3.14159265358979323846 * passHz / kFs;
    std::vector<double> sig(N);
    for (size_t i = 0; i < N; ++i) sig[i] = std::sin(omega * i);

    double s1 = 0.0, s2 = 0.0;
    double rms_out = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < N; ++i) {
        double y = bq.process(sig[i], s1, s2);
        if (i >= N / 2) { rms_out += y * y; ++count; }
    }
    rms_out = std::sqrt(rms_out / count);
    // 200 Hz is far from 1000 Hz notch; amplitude should remain ~1/sqrt(2)
    EXPECT_GT(rms_out, 0.5);
}

TEST(NotchHz, InvalidFrequencyThrows) {
    EXPECT_THROW(designNotchHz(0.0,    kFs, 1.0), std::invalid_argument);
    EXPECT_THROW(designNotchHz(kFs,    kFs, 1.0), std::invalid_argument);
    EXPECT_THROW(designNotchHz(500.0,  kFs, 0.0), std::invalid_argument);
}
