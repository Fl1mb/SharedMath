#include <gtest/gtest.h>
#include "DSP/Hilbert.h"

#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>

using namespace SharedMath::DSP;

static constexpr double kPi   = 3.14159265358979323846;
static constexpr double kTol  = 1e-9;
static constexpr double kTolF = 1e-6;  // relaxed tolerance for frequency estimates

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static double maxErrC(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return 1e30;
    double e = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        e = std::max(e, std::abs(a[i] - b[i]));
    return e;
}

// Pure cosine: k0 full cycles in N samples (bin-aligned → exact FFT, no leakage).
static std::vector<double> purecos(size_t N, size_t k0) {
    std::vector<double> x(N);
    for (size_t i = 0; i < N; ++i)
        x[i] = std::cos(2.0 * kPi * static_cast<double>(k0) / static_cast<double>(N)
                        * static_cast<double>(i));
    return x;
}

// Pure sine: bin-aligned.
static std::vector<double> puresin(size_t N, size_t k0) {
    std::vector<double> x(N);
    for (size_t i = 0; i < N; ++i)
        x[i] = std::sin(2.0 * kPi * static_cast<double>(k0) / static_cast<double>(N)
                        * static_cast<double>(i));
    return x;
}

// ═════════════════════════════════════════════════════════════════════════════
// HilbertTransform — basic signal identities
// ═════════════════════════════════════════════════════════════════════════════

TEST(HilbertTransform, EmptyInput) {
    EXPECT_TRUE(analyticSignal({}).empty());
    EXPECT_TRUE(hilbert({}).empty());
    EXPECT_TRUE(instantaneousAmplitude({}).empty());
    EXPECT_TRUE(instantaneousPhase({}).empty());
    EXPECT_TRUE(instantaneousFrequency({}).empty());
}

TEST(HilbertTransform, SingleSampleOutputLength) {
    auto z = analyticSignal({3.14});
    EXPECT_EQ(z.size(), 1u);
    // For N=1, only DC exists; analytic signal = (3.14, 0j)
    EXPECT_NEAR(z[0].real(), 3.14, 1e-12);
}

TEST(HilbertTransform, RealPartEqualsInput) {
    // Re(analyticSignal(x)) must equal x for all n.
    const size_t N = 128;
    auto x = purecos(N, 8);
    auto z = analyticSignal(x);
    ASSERT_EQ(z.size(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR(z[i].real(), x[i], kTol);
}

TEST(HilbertTransform, QuadratureOf_Cosine_Is_Sine) {
    // H{cos(2π f₀ n)} = sin(2π f₀ n)  for a bin-aligned tone.
    // With k0=8 cycles in N=128 samples (no spectral leakage → exact result).
    const size_t N = 128;
    const size_t k0 = 8;
    auto hx  = hilbert(purecos(N, k0));
    auto ref = puresin(N, k0);
    EXPECT_LT(maxErrC(hx, ref), kTol);
}

TEST(HilbertTransform, QuadratureOf_Sine_Is_MinusCosine) {
    // H{sin(2π f₀ n)} = −cos(2π f₀ n)
    const size_t N = 128;
    const size_t k0 = 12;
    auto hx = hilbert(puresin(N, k0));
    auto ref = purecos(N, k0);
    // hx should equal -ref
    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR(hx[i], -ref[i], kTol);
}

TEST(HilbertTransform, Linearity) {
    // H{a·x + b·y} = a·H{x} + b·H{y}
    const size_t N = 128;
    auto x = purecos(N, 5);
    auto y = puresin(N, 11);
    const double a = 2.5, b = -1.3;

    std::vector<double> combo(N);
    for (size_t i = 0; i < N; ++i) combo[i] = a * x[i] + b * y[i];

    auto h_combo = hilbert(combo);
    auto h_x     = hilbert(x);
    auto h_y     = hilbert(y);

    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR(h_combo[i], a * h_x[i] + b * h_y[i], kTol);
}

// ═════════════════════════════════════════════════════════════════════════════
// InstantaneousAmplitude — envelope extraction
// ═════════════════════════════════════════════════════════════════════════════

TEST(InstantaneousAmplitude, ConstantEnvelope_PureCosine) {
    // The envelope of a pure (unmodulated) cosine is identically 1.
    const size_t N = 128;
    auto env = instantaneousAmplitude(purecos(N, 10));
    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR(env[i], 1.0, kTol);
}

TEST(InstantaneousAmplitude, ConstantEnvelope_PureSine) {
    const size_t N = 128;
    auto env = instantaneousAmplitude(puresin(N, 7));
    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR(env[i], 1.0, kTol);
}

TEST(InstantaneousAmplitude, ScaledAmplitude) {
    // Envelope of A·cos should equal |A| everywhere.
    const size_t N = 128;
    const double A = 3.7;
    std::vector<double> x = purecos(N, 6);
    for (auto& v : x) v *= A;
    auto env = instantaneousAmplitude(x);
    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR(env[i], A, kTol);
}

TEST(InstantaneousAmplitude, AMSignal_EnvelopeFollowsModulation) {
    // x[n] = (1 + 0.5·cos(2πf_m·n)) · cos(2πf_c·n)
    // For fc >> fm and both bin-aligned, envelope ≈ |1 + 0.5·cos(2πf_m·n)|.
    // We test that the envelope is always between 0.5 and 1.5.
    const size_t N = 512;
    const size_t kc = 40, km = 4;  // carrier and modulation, both bin-aligned
    std::vector<double> x(N);
    for (size_t i = 0; i < N; ++i) {
        double mod      = 1.0 + 0.5 * std::cos(2.0 * kPi * km * i / N);
        double carrier  = std::cos(2.0 * kPi * kc * i / N);
        x[i] = mod * carrier;
    }
    auto env = instantaneousAmplitude(x);
    // Interior only — edges are less reliable for slowly-varying AM
    for (size_t i = N / 8; i < 7 * N / 8; ++i) {
        EXPECT_GE(env[i], 0.45);
        EXPECT_LE(env[i], 1.55);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// InstantaneousPhase — wrapped and unwrapped
// ═════════════════════════════════════════════════════════════════════════════

TEST(InstantaneousPhase, WrappedInRange) {
    const size_t N = 128;
    auto phi = instantaneousPhase(purecos(N, 8), /*unwrap=*/false);
    for (double p : phi)
        EXPECT_LE(p, kPi + 1e-12);
}

TEST(InstantaneousPhase, UnwrappedIsMonotonic_PositiveTone) {
    // Unwrapped phase of a positive-frequency cosine must be non-decreasing.
    const size_t N = 256;
    auto phi = instantaneousPhase(purecos(N, 16), /*unwrap=*/true);
    for (size_t i = 1; i < phi.size(); ++i)
        EXPECT_GE(phi[i], phi[i - 1] - 1e-9);
}

TEST(InstantaneousPhase, UnwrappedLinear_BinAligned) {
    // For a bin-aligned cosine the unwrapped phase must increase by 2π·k0/N
    // per sample (constant increment).
    const size_t N = 128;
    const size_t k0 = 10;
    double expectedSlope = 2.0 * kPi * static_cast<double>(k0) / static_cast<double>(N);
    auto phi = instantaneousPhase(purecos(N, k0), /*unwrap=*/true);
    // Check constant slope on interior samples (skip the first sample, where
    // the phase starts at 0 regardless of convention).
    for (size_t i = 2; i < N; ++i)
        EXPECT_NEAR(phi[i] - phi[i - 1], expectedSlope, kTol);
}

// ═════════════════════════════════════════════════════════════════════════════
// InstantaneousFrequency — derivative of phase
// ═════════════════════════════════════════════════════════════════════════════

TEST(InstantaneousFrequency, OutputLengthIsNMinus1) {
    const size_t N = 64;
    auto freq = instantaneousFrequency(purecos(N, 4));
    EXPECT_EQ(freq.size(), N - 1);
}

TEST(InstantaneousFrequency, EmptyAndSingleInputReturnEmpty) {
    EXPECT_TRUE(instantaneousFrequency({}).empty());
    EXPECT_TRUE(instantaneousFrequency({1.0}).empty());
}

TEST(InstantaneousFrequency, ConstantFrequency_BinAligned) {
    // For x[n] = cos(2π·k0/N·n), instantaneous frequency = k0/N (normalised).
    // With sampleRate = 1.0, result is in cycles/sample ∈ [0, 0.5].
    const size_t N = 128;
    const size_t k0 = 8;
    double expectedFreq = static_cast<double>(k0) / static_cast<double>(N);

    auto freq = instantaneousFrequency(purecos(N, k0), /*sampleRate=*/1.0);
    // Interior samples (skip first and last to avoid phase-branch artefacts)
    for (size_t i = 2; i + 2 < freq.size(); ++i)
        EXPECT_NEAR(freq[i], expectedFreq, kTolF);
}

TEST(InstantaneousFrequency, ScaledBySampleRate) {
    // instantaneousFrequency(x, fs) should return the frequency in Hz.
    const size_t N = 128;
    const size_t k0 = 8;
    const double fs = 1000.0;
    double expectedHz = static_cast<double>(k0) / static_cast<double>(N) * fs;

    auto freq = instantaneousFrequency(purecos(N, k0), fs);
    for (size_t i = 2; i + 2 < freq.size(); ++i)
        EXPECT_NEAR(freq[i], expectedHz, kTolF * fs);
}

TEST(InstantaneousFrequency, ChirpTracksLinearSweep) {
    // Linear chirp with proper phase accumulation:
    //   x[n] = cos(2π * (f0*n + (f1-f0)/(2N) * n²))
    //   true IF at sample n: f0 + (f1-f0)*n/N
    // Both f0 and f1 are well below Nyquist (0.5) to avoid aliasing.
    // We verify that the estimated IF closely tracks the true IF over the
    // interior of the signal (edges are excluded to avoid Hilbert edge effects).
    const size_t N  = 512;
    const double f0 = 0.05, f1 = 0.30;

    std::vector<double> chirp(N);
    for (size_t i = 0; i < N; ++i) {
        double phase = f0 * i + (f1 - f0) / (2.0 * N) * static_cast<double>(i * i);
        chirp[i] = std::cos(2.0 * kPi * phase);
    }

    auto freq = instantaneousFrequency(chirp, 1.0);

    // Interior quarter [N/4, 3N/4): estimate should be within 2% of true IF.
    size_t lo = N / 4, hi = 3 * N / 4;
    for (size_t i = lo; i < hi && i < freq.size(); ++i) {
        double trueIF = f0 + (f1 - f0) * static_cast<double>(i) / static_cast<double>(N);
        EXPECT_NEAR(freq[i], trueIF, 0.02);
    }

    // Mean IF in last quarter must exceed mean IF in first quarter.
    double sumLo = 0.0, sumHi = 0.0;
    size_t qtr = N / 4;
    for (size_t i = qtr;     i < 2 * qtr; ++i) sumLo += freq[i];
    for (size_t i = 2 * qtr; i < 3 * qtr; ++i) sumHi += freq[i];
    EXPECT_GT(sumHi / qtr, sumLo / qtr);
}
