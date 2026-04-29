#include <gtest/gtest.h>
#include "DSP/Spectral.h"
#include "DSP/SignalGenerator.h"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>

using namespace SharedMath::DSP;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers

static size_t peakBin(const std::vector<double>& v) {
    return static_cast<size_t>(
        std::max_element(v.begin(), v.end()) - v.begin());
}

static bool allNonNegative(const std::vector<double>& v) {
    for (double x : v) if (x < 0.0) return false;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// periodogram

TEST(Periodogram, SinePeakAtExpectedFrequency) {
    // 100 Hz sine at 1000 Hz sample rate — peak should land within 2 bins of 100 Hz
    double fs = 1000.0, f = 100.0;
    size_t N  = 1024;
    auto sig  = sineWave(f, fs, N);

    auto [freqs, psd] = periodogram(sig, fs);  // default Hann window

    size_t pk  = peakBin(psd);
    double res = fs / static_cast<double>(N);  // frequency resolution ≈ 0.977 Hz
    EXPECT_NEAR(freqs[pk], f, 2.0 * res);
}

TEST(Periodogram, PSDNonNegative) {
    auto sig = sineWave(200.0, 1000.0, 512);
    auto [freqs, psd] = periodogram(sig, 1000.0);
    EXPECT_TRUE(allNonNegative(psd));
}

TEST(Periodogram, FrequencyAxisSize) {
    // N=1024 (already power of 2) → Nfft=1024 → 513 bins
    auto sig = sineWave(50.0, 1000.0, 1024);
    auto [freqs, psd] = periodogram(sig, 1000.0);
    EXPECT_EQ(freqs.size(), 513u);
    EXPECT_EQ(psd.size(),   513u);
    EXPECT_DOUBLE_EQ(freqs.front(), 0.0);
    EXPECT_NEAR(freqs.back(), 500.0, 1.0);   // Nyquist ≈ 500 Hz
}

TEST(Periodogram, FrequenciesAndPSDSameSize) {
    auto sig = sineWave(100.0, 1000.0, 600);  // non-power-of-2 length
    auto [freqs, psd] = periodogram(sig, 1000.0);
    EXPECT_EQ(freqs.size(), psd.size());
    EXPECT_GT(freqs.size(), 0u);
}

TEST(Periodogram, DCSignalPeakAtBinZero) {
    std::vector<double> dc(1024, 1.0);
    auto [freqs, psd] = periodogram(dc, 1000.0);
    EXPECT_EQ(peakBin(psd), 0u);
}

TEST(Periodogram, RectangularWindowGivesSameSizeResult) {
    WindowParams rect; rect.type = WindowType::Rectangular;
    auto sig = sineWave(100.0, 1000.0, 1024);
    auto [freqs, psd] = periodogram(sig, 1000.0, rect);
    EXPECT_EQ(freqs.size(), psd.size());
}

TEST(Periodogram, SpectrumScalingPSDSmallerThanDensity) {
    // Density scales by 1/(fs·Σw²) which is smaller than 1/(Σw)² for Hann at N=1024
    // so density values at the peak should differ from spectrum values
    auto sig = sineWave(100.0, 1000.0, 1024);
    auto [f1, psd_density]  = periodogram(sig, 1000.0, {}, PSDScaling::Density);
    auto [f2, psd_spectrum] = periodogram(sig, 1000.0, {}, PSDScaling::Spectrum);
    // Peak bins should be at the same frequency
    EXPECT_EQ(peakBin(psd_density), peakBin(psd_spectrum));
    // Values should differ (different normalization)
    EXPECT_NE(psd_density[peakBin(psd_density)],
              psd_spectrum[peakBin(psd_spectrum)]);
}

// ─────────────────────────────────────────────────────────────────────────────
// welchPSD

TEST(WelchPSD, SinePeakAtExpectedFrequency) {
    double fs = 1000.0, f = 100.0;
    auto sig  = sineWave(f, fs, 4096);

    // frame=256 → resolution ≈ 3.9 Hz
    auto [freqs, psd] = welchPSD(sig, fs, 256, 128);

    size_t pk  = peakBin(psd);
    double res = fs / 256.0;
    EXPECT_NEAR(freqs[pk], f, 2.0 * res);
}

TEST(WelchPSD, PSDNonNegative) {
    auto sig = sineWave(300.0, 1000.0, 2048);
    auto [freqs, psd] = welchPSD(sig, 1000.0, 256, 128);
    EXPECT_TRUE(allNonNegative(psd));
}

TEST(WelchPSD, FrequencyAxisMatchesFrameSize) {
    size_t frame = 256;                        // nextPow2(256)=256 → 129 bins
    auto sig = sineWave(100.0, 1000.0, 4096);
    auto [freqs, psd] = welchPSD(sig, 1000.0, frame, 128);
    EXPECT_EQ(freqs.size(), psd.size());
    EXPECT_GT(freqs.size(), 0u);
    EXPECT_DOUBLE_EQ(freqs.front(), 0.0);
    EXPECT_EQ(freqs.size(), 129u);             // 256/2+1
}

TEST(WelchPSD, MoreFramesMeansLessVariance) {
    // Not a variance test per se — just checks that the function runs and
    // the peak is at the right place for many overlapping frames.
    double fs = 1000.0, f = 200.0;
    auto sig  = sineWave(f, fs, 8192);
    auto [freqs, psd] = welchPSD(sig, fs, 256, 64);
    size_t pk  = peakBin(psd);
    double res = fs / 256.0;
    EXPECT_NEAR(freqs[pk], f, 2.0 * res);
}

// ─────────────────────────────────────────────────────────────────────────────
// powerSpectralDensityDB

TEST(PowerSpectralDensityDB, MonotonicWithLinearPSD) {
    std::vector<double> psd = {0.001, 0.01, 0.1, 1.0, 10.0};
    auto db = powerSpectralDensityDB(psd);
    ASSERT_EQ(db.size(), 5u);
    for (size_t i = 1; i < db.size(); ++i)
        EXPECT_GT(db[i], db[i - 1]);
}

TEST(PowerSpectralDensityDB, UnitPowerIsZeroDB) {
    std::vector<double> psd = {1.0, 1.0, 1.0};
    auto db = powerSpectralDensityDB(psd);
    for (double v : db) EXPECT_NEAR(v, 0.0, 1e-10);
}

TEST(PowerSpectralDensityDB, TenTimesMoreIsTenDB) {
    std::vector<double> psd = {1.0, 10.0};
    auto db = powerSpectralDensityDB(psd);
    EXPECT_NEAR(db[1] - db[0], 10.0, 1e-10);
}

TEST(PowerSpectralDensityDB, CustomRefPower) {
    std::vector<double> psd = {100.0};
    auto db = powerSpectralDensityDB(psd, 100.0);
    EXPECT_NEAR(db[0], 0.0, 1e-10);
}

// ─────────────────────────────────────────────────────────────────────────────
// crossPowerSpectralDensity

TEST(CrossPSD, AutoPSDMatchesWelchPSD) {
    double fs  = 1000.0;
    auto sig   = sineWave(100.0, fs, 4096);

    auto cpsd_res = crossPowerSpectralDensity(sig, sig, fs, 256, 128);
    auto psd_res  = welchPSD(sig, fs, 256, 128);

    ASSERT_EQ(cpsd_res.cpsd.size(), psd_res.psd.size());
    for (size_t k = 0; k < psd_res.psd.size(); ++k) {
        EXPECT_NEAR(cpsd_res.cpsd[k].real(), psd_res.psd[k], 1e-10);
        EXPECT_NEAR(cpsd_res.cpsd[k].imag(), 0.0,            1e-10);
    }
}

TEST(CrossPSD, FrequenciesNonNegative) {
    auto sig  = sineWave(100.0, 1000.0, 4096);
    auto res  = crossPowerSpectralDensity(sig, sig, 1000.0, 256, 128);
    for (double f : res.frequencies) EXPECT_GE(f, 0.0);
}

TEST(CrossPSD, FrequenciesAndCPSDSameSize) {
    auto sig = sineWave(150.0, 1000.0, 2048);
    auto res = crossPowerSpectralDensity(sig, sig, 1000.0, 256, 128);
    EXPECT_EQ(res.frequencies.size(), res.cpsd.size());
}
