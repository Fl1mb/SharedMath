#include <gtest/gtest.h>
#include "DSP/SignalMetrics.h"

#include <complex>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

using namespace SharedMath::DSP;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<std::complex<double>>
makeToneSM(double f, double fs, size_t N, double A = 1.0)
{
    std::vector<std::complex<double>> iq(N);
    const double ph = 2.0 * M_PI * f / fs;
    for (size_t i = 0; i < N; ++i)
        iq[i] = A * std::polar(1.0, ph * static_cast<double>(i));
    return iq;
}

// ─────────────────────────────────────────────────────────────────────────────
// averagePowerDb
// ─────────────────────────────────────────────────────────────────────────────

TEST(SignalMetrics, AveragePowerDb_EmptyIQ_ReturnsNegInf)
{
    const double v = averagePowerDb({});
    EXPECT_TRUE(std::isinf(v) && v < 0.0);
}

TEST(SignalMetrics, AveragePowerDb_UnitAmplitudeConstant_IsZeroDB)
{
    // Constant IQ = (1, 0): |x|² = 1 → 0 dBFS
    std::vector<std::complex<double>> iq(1024, {1.0, 0.0});
    EXPECT_NEAR(averagePowerDb(iq), 0.0, 1e-9);
}

TEST(SignalMetrics, AveragePowerDb_UnitTone_IsZeroDB)
{
    // Unit-amplitude tone: |x|² = 1 → 0 dBFS
    auto iq = makeToneSM(100.0, 1000.0, 1024, 1.0);
    EXPECT_NEAR(averagePowerDb(iq), 0.0, 1e-9);
}

TEST(SignalMetrics, AveragePowerDb_AmplitudeTen_IsTwentyDB)
{
    // |10·exp(j*...)|² = 100 → 10·log10(100) = 20 dBFS
    auto iq = makeToneSM(100.0, 1000.0, 512, 10.0);
    EXPECT_NEAR(averagePowerDb(iq), 20.0, 1e-6);
}

TEST(SignalMetrics, AveragePowerDb_HalfAmplitude_IsMinusSixDB)
{
    // |0.5|² = 0.25 → 10·log10(0.25) ≈ −6.02 dBFS
    std::vector<std::complex<double>> iq(512, {0.5, 0.0});
    EXPECT_NEAR(averagePowerDb(iq), 10.0 * std::log10(0.25), 1e-9);
}

// ─────────────────────────────────────────────────────────────────────────────
// peakPowerDb
// ─────────────────────────────────────────────────────────────────────────────

TEST(SignalMetrics, PeakPowerDb_EmptyIQ_ReturnsNegInf)
{
    const double v = peakPowerDb({});
    EXPECT_TRUE(std::isinf(v) && v < 0.0);
}

TEST(SignalMetrics, PeakPowerDb_KnownAmplitude)
{
    // Constant IQ amplitude = 3 → peak power = 10·log10(9) ≈ 9.54 dBFS
    std::vector<std::complex<double>> iq(256, {3.0, 0.0});
    EXPECT_NEAR(peakPowerDb(iq), 10.0 * std::log10(9.0), 1e-9);
}

TEST(SignalMetrics, PeakPowerDb_GePeakEqualOrGreaterThanAverage)
{
    auto iq = makeToneSM(50.0, 1000.0, 512, 1.0);
    EXPECT_GE(peakPowerDb(iq), averagePowerDb(iq));
}

// ─────────────────────────────────────────────────────────────────────────────
// paprDb
// ─────────────────────────────────────────────────────────────────────────────

TEST(SignalMetrics, PaprDb_EmptyIQ_ReturnsZero)
{
    EXPECT_DOUBLE_EQ(paprDb({}), 0.0);
}

TEST(SignalMetrics, PaprDb_ConstantAmplitude_IsNearZero)
{
    // Constant amplitude: peak = average → PAPR ≈ 0 dB
    std::vector<std::complex<double>> iq(512, {2.0, 0.0});
    EXPECT_NEAR(paprDb(iq), 0.0, 1e-9);
}

TEST(SignalMetrics, PaprDb_NonNegative)
{
    // PAPR is always ≥ 0 by definition
    auto iq = makeToneSM(100.0, 1000.0, 1024, 1.0);
    EXPECT_GE(paprDb(iq), 0.0);
}

TEST(SignalMetrics, PaprDb_EqualsPeakMinusAverage)
{
    auto iq = makeToneSM(100.0, 1000.0, 512, 1.5);
    EXPECT_NEAR(paprDb(iq), peakPowerDb(iq) - averagePowerDb(iq), 1e-9);
}

// ─────────────────────────────────────────────────────────────────────────────
// evmRmsPercent
// ─────────────────────────────────────────────────────────────────────────────

TEST(SignalMetrics, EvmRmsPercent_IdenticalVectors_IsZero)
{
    std::vector<std::complex<double>> v = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
    EXPECT_NEAR(evmRmsPercent(v, v), 0.0, 1e-10);
}

TEST(SignalMetrics, EvmRmsPercent_EmptyMeasured_Throws)
{
    std::vector<std::complex<double>> ref = {{1, 0}};
    EXPECT_THROW(evmRmsPercent({}, ref), std::invalid_argument);
}

TEST(SignalMetrics, EvmRmsPercent_EmptyReference_Throws)
{
    std::vector<std::complex<double>> meas = {{1, 0}};
    EXPECT_THROW(evmRmsPercent(meas, {}), std::invalid_argument);
}

TEST(SignalMetrics, EvmRmsPercent_MismatchedSizes_Throws)
{
    std::vector<std::complex<double>> a(4, {1, 0});
    std::vector<std::complex<double>> b(5, {1, 0});
    EXPECT_THROW(evmRmsPercent(a, b), std::invalid_argument);
}

TEST(SignalMetrics, EvmRmsPercent_KnownError_CorrectValue)
{
    // reference: unit vector (1,0); measured: (1+ε, 0) → |error|/|ref| = ε → ε×100 %
    const double eps = 0.1;
    std::vector<std::complex<double>> ref  = {{1.0, 0.0}};
    std::vector<std::complex<double>> meas = {{1.0 + eps, 0.0}};
    EXPECT_NEAR(evmRmsPercent(meas, ref), eps * 100.0, 1e-8);
}

TEST(SignalMetrics, EvmRmsPercent_NonNegative)
{
    std::vector<std::complex<double>> ref  = {{1, 0}, {0, 1}};
    std::vector<std::complex<double>> meas = {{0.9, 0.1}, {-0.1, 0.9}};
    EXPECT_GE(evmRmsPercent(meas, ref), 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// evmRmsDb
// ─────────────────────────────────────────────────────────────────────────────

TEST(SignalMetrics, EvmRmsDb_IdenticalVectors_IsNegInf)
{
    std::vector<std::complex<double>> v = {{1, 0}, {0, 1}};
    const double db = evmRmsDb(v, v);
    EXPECT_TRUE(std::isinf(db) && db < 0.0);
}

TEST(SignalMetrics, EvmRmsDb_TenPercentEVM_IsMinusTwentyDB)
{
    // 10 % EVM → 20·log10(0.1) = −20 dB
    const double eps = 0.1;
    std::vector<std::complex<double>> ref  = {{1.0, 0.0}};
    std::vector<std::complex<double>> meas = {{1.0 + eps, 0.0}};
    EXPECT_NEAR(evmRmsDb(meas, ref), 20.0 * std::log10(eps), 0.1);
}

// ─────────────────────────────────────────────────────────────────────────────
// estimateSnrDb
// ─────────────────────────────────────────────────────────────────────────────

TEST(SignalMetrics, EstimateSnrDb_EmptyIQ_ReturnsZero)
{
    EXPECT_DOUBLE_EQ(estimateSnrDb({}, 1000.0), 0.0);
}

TEST(SignalMetrics, EstimateSnrDb_ZeroSampleRate_Throws)
{
    std::vector<std::complex<double>> iq = {{1.0, 0.0}};
    EXPECT_THROW(estimateSnrDb(iq, 0.0), std::invalid_argument);
}

TEST(SignalMetrics, EstimateSnrDb_ZeroFftSize_Throws)
{
    std::vector<std::complex<double>> iq = {{1.0, 0.0}};
    EXPECT_THROW(estimateSnrDb(iq, 1000.0, 0), std::invalid_argument);
}

TEST(SignalMetrics, EstimateSnrDb_PositiveForToneInNoise)
{
    const double fs = 10'000.0;
    auto iq = makeToneSM(1'000.0, fs, 4096, 1.0);
    // Add tiny noise
    uint64_t s = 17;
    for (auto& v : iq) {
        s = s * 6364136223846793005ULL + 1;
        v += 0.001 * std::complex<double>(
            (static_cast<double>(s >> 33) / static_cast<double>(uint64_t{1} << 31)) - 1.0,
            (static_cast<double>((s >> 1) & 0xFFFF) / 32768.0) - 1.0);
    }
    const double snr = estimateSnrDb(iq, fs, 1024);
    EXPECT_GT(snr, 10.0);  // tone should be clearly above noise
}
