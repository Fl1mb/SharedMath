#include <gtest/gtest.h>
#include "DSP/SignalEstimation.h"

#include <complex>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace SharedMath::DSP;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<std::complex<double>>
makeToneE(double f, double fs, size_t N, double A = 1.0)
{
    std::vector<std::complex<double>> iq(N);
    const double ph = 2.0 * M_PI * f / fs;
    for (size_t i = 0; i < N; ++i)
        iq[i] = A * std::polar(1.0, ph * static_cast<double>(i));
    return iq;
}

static std::vector<std::complex<double>>
makeNoiseE(size_t N, double A = 0.01, uint64_t seed = 42)
{
    std::vector<std::complex<double>> out(N);
    uint64_t s = seed;
    auto rng = [&]() -> double {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return (static_cast<double>(s >> 33) /
                static_cast<double>(uint64_t{1} << 31)) - 1.0;
    };
    for (auto& v : out) v = A * std::complex<double>(rng(), rng());
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Empty IQ
// ─────────────────────────────────────────────────────────────────────────────

TEST(SignalEstimation, EmptyIQ_ReturnsSafeEstimate)
{
    SignalEstimationParams p;
    auto e = estimateSignal({}, p);
    EXPECT_DOUBLE_EQ(e.centerFrequencyHz,   0.0);
    EXPECT_DOUBLE_EQ(e.durationSec,         0.0);
    EXPECT_DOUBLE_EQ(e.occupiedBandwidthHz, 0.0);
}

TEST(SignalEstimation, EmptyIQ_CenterFreq_ReturnsZero)
{
    EXPECT_DOUBLE_EQ(estimateCenterFrequencyHz({}, 1000.0), 0.0);
}

TEST(SignalEstimation, EmptyIQ_OccupiedBW_ReturnsZero)
{
    EXPECT_DOUBLE_EQ(estimateOccupiedBandwidthHz({}, 1000.0), 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Invalid parameters
// ─────────────────────────────────────────────────────────────────────────────

TEST(SignalEstimation, InvalidSampleRate_Throws)
{
    SignalEstimationParams p;
    p.sampleRate = 0.0;
    std::vector<std::complex<double>> iq = {{1.0, 0.0}};
    EXPECT_THROW(estimateSignal(iq, p), std::invalid_argument);
}

TEST(SignalEstimation, NegativeSampleRate_Throws)
{
    SignalEstimationParams p;
    p.sampleRate = -1.0;
    std::vector<std::complex<double>> iq = {{1.0, 0.0}};
    EXPECT_THROW(estimateSignal(iq, p), std::invalid_argument);
}

TEST(SignalEstimation, ZeroFftSize_Throws)
{
    SignalEstimationParams p;
    p.fftSize = 0;
    std::vector<std::complex<double>> iq = {{1.0, 0.0}};
    EXPECT_THROW(estimateSignal(iq, p), std::invalid_argument);
}

TEST(SignalEstimation, ZeroOccupiedRatio_Throws)
{
    SignalEstimationParams p;
    p.occupiedPowerRatio = 0.0;
    std::vector<std::complex<double>> iq = {{1.0, 0.0}};
    EXPECT_THROW(estimateSignal(iq, p), std::invalid_argument);
}

TEST(SignalEstimation, OccupiedRatioAboveOne_Throws)
{
    SignalEstimationParams p;
    p.occupiedPowerRatio = 1.1;
    std::vector<std::complex<double>> iq = {{1.0, 0.0}};
    EXPECT_THROW(estimateSignal(iq, p), std::invalid_argument);
}

// ─────────────────────────────────────────────────────────────────────────────
// Centre frequency
// ─────────────────────────────────────────────────────────────────────────────

TEST(SignalEstimation, CenterFrequency_NearTone_Positive)
{
    const double fs   = 10'000.0;
    const double tone = 2'000.0;
    auto iq = makeToneE(tone, fs, 4096);

    SignalEstimationParams p;
    p.sampleRate = fs;
    p.fftSize    = 1024;
    auto e = estimateSignal(iq, p);

    const double binHz = fs / static_cast<double>(p.fftSize);
    EXPECT_NEAR(e.centerFrequencyHz, tone, 3.0 * binHz)
        << "Expected cf near " << tone << " Hz, got " << e.centerFrequencyHz;
}

TEST(SignalEstimation, CenterFrequency_NearTone_Negative)
{
    const double fs   = 10'000.0;
    const double tone = -1'500.0;
    auto iq = makeToneE(tone, fs, 4096);

    SignalEstimationParams p;
    p.sampleRate = fs;
    p.fftSize    = 1024;
    auto e = estimateSignal(iq, p);

    const double binHz = fs / static_cast<double>(p.fftSize);
    EXPECT_NEAR(e.centerFrequencyHz, tone, 3.0 * binHz);
}

TEST(SignalEstimation, StandaloneEstimateCenterFreq_NearTone)
{
    const double fs   = 8'000.0;
    const double tone = 1'000.0;
    auto iq = makeToneE(tone, fs, 2048);

    const double cf    = estimateCenterFrequencyHz(iq, fs, 512);
    const double binHz = fs / 512.0;
    EXPECT_NEAR(cf, tone, 3.0 * binHz);
}

// ─────────────────────────────────────────────────────────────────────────────
// Occupied bandwidth
// ─────────────────────────────────────────────────────────────────────────────

TEST(SignalEstimation, OccupiedBandwidth_Positive_LessThanSampleRate)
{
    const double fs = 10'000.0;
    auto iq = makeToneE(1'000.0, fs, 4096);

    SignalEstimationParams p;
    p.sampleRate = fs;
    p.fftSize    = 1024;
    auto e = estimateSignal(iq, p);

    EXPECT_GT(e.occupiedBandwidthHz, 0.0);
    EXPECT_LT(e.occupiedBandwidthHz, fs);
}

TEST(SignalEstimation, OccupiedBandwidth_99pct_LessThanFull)
{
    const double fs = 10'000.0;
    auto iq = makeToneE(1'000.0, fs, 4096);

    const double bw99  = estimateOccupiedBandwidthHz(iq, fs, 0.99,  1024);
    const double bw100 = estimateOccupiedBandwidthHz(iq, fs, 1.00,  1024);

    EXPECT_LE(bw99, bw100);
    EXPECT_GT(bw99, 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Power and duration
// ─────────────────────────────────────────────────────────────────────────────

TEST(SignalEstimation, AveragePowerDb_UnitAmplitudeTone)
{
    const double fs = 1'000.0;
    auto iq = makeToneE(100.0, fs, 2048, /*A=*/1.0);

    SignalEstimationParams p;
    p.sampleRate = fs;
    auto e = estimateSignal(iq, p);

    EXPECT_NEAR(e.averagePowerDb, 0.0, 0.5);  // |exp(j*...)|² = 1 → 0 dBFS
}

TEST(SignalEstimation, DurationSec_MatchesIQLength)
{
    const double fs   = 1'000.0;
    const size_t N    = 500;
    auto iq = makeToneE(100.0, fs, N);

    SignalEstimationParams p;
    p.sampleRate = fs;
    auto e = estimateSignal(iq, p);

    EXPECT_NEAR(e.durationSec, static_cast<double>(N) / fs, 1e-9);
}

TEST(SignalEstimation, SNR_PositiveForToneInNoise)
{
    const double fs = 10'000.0;
    auto iq  = makeToneE(1'000.0, fs, 4096, /*A=*/1.0);
    auto noise = makeNoiseE(4096, 0.001, 7);
    for (size_t i = 0; i < 4096; ++i) iq[i] += noise[i];

    SignalEstimationParams p;
    p.sampleRate = fs;
    p.fftSize    = 1024;
    auto e = estimateSignal(iq, p);

    EXPECT_GT(e.snrDb, 0.0);
}
