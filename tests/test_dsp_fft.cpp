#include <gtest/gtest.h>
#include <complex>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "DSP/DSP.h"

using namespace SharedMath::DSP;
using cx = std::complex<double>;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
namespace {

constexpr double kTight = 1e-10;   // Cooley-Tukey round-trip tolerance
constexpr double kLoose = 1e-8;    // Bluestein / multi-pass tolerance

// Maximum absolute element-wise error between two complex vectors
double maxErr(const std::vector<cx>& a, const std::vector<cx>& b) {
    double e = 0.0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i)
        e = std::max(e, std::abs(a[i] - b[i]));
    return e;
}

// Build a mixed test signal: 1 + cos(2π·k1·n/N) + 0.5·sin(2π·k2·n/N)
std::vector<cx> makeMixed(size_t N, size_t k1 = 3, size_t k2 = 7) {
    std::vector<cx> x(N);
    for (size_t n = 0; n < N; ++n) {
        double t = 2.0 * M_PI * static_cast<double>(n) / static_cast<double>(N);
        x[n] = 1.0 + std::cos(static_cast<double>(k1) * t)
                    + 0.5 * std::sin(static_cast<double>(k2) * t);
    }
    return x;
}

// O(N²) DFT — reference implementation
std::vector<cx> naiveDFT(const std::vector<cx>& x, bool inverse = false) {
    size_t N = x.size();
    double sign = inverse ? 1.0 : -1.0;
    std::vector<cx> X(N, {0.0, 0.0});
    for (size_t k = 0; k < N; ++k)
        for (size_t n = 0; n < N; ++n) {
            double ang = sign * 2.0 * M_PI * static_cast<double>(k * n)
                                           / static_cast<double>(N);
            X[k] += x[n] * cx{std::cos(ang), std::sin(ang)};
        }
    return X;
}

// O(N²) linear convolution — reference
std::vector<double> naiveConvolve(const std::vector<double>& a,
                                   const std::vector<double>& b) {
    size_t n = a.size() + b.size() - 1;
    std::vector<double> out(n, 0.0);
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j)
            out[i + j] += a[i] * b[j];
    return out;
}

// O(N²) cross-correlation — reference: corr[k] = Σ conj(a[n])·b[n+k]
std::vector<double> naiveCorrelate(const std::vector<double>& a,
                                    const std::vector<double>& b) {
    // Full cross-correlation, length = a.size() + b.size() - 1
    // lag runs from -(a.size()-1) to +(b.size()-1)
    size_t n = a.size() + b.size() - 1;
    std::vector<double> out(n, 0.0);
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j)
            out[b.size() - 1 - j + i] += a[i] * b[j]; // off by j direction
    // Actually just compute via naive convolution with reversed a
    std::vector<double> ar(a.rbegin(), a.rend());
    return naiveConvolve(ar, b);
}

} // namespace

// ═════════════════════════════════════════════════════════════════════════════
// FFTPlan — construction and metadata
// ═════════════════════════════════════════════════════════════════════════════

TEST(FFTPlanTest, CreateDefaultCPU) {
    auto plan = FFTPlan::create(256);
    EXPECT_EQ(plan.size(), 256u);
    EXPECT_FALSE(plan.isInverse());
    EXPECT_NE(plan.backendName(), nullptr);
    EXPECT_NE(plan.backend(), nullptr);
}

TEST(FFTPlanTest, CreateWithConfig) {
    FFTConfig cfg{FFTDirection::Inverse, FFTNorm::ByN};
    auto plan = FFTPlan::create(512, cfg);
    EXPECT_EQ(plan.size(), 512u);
    EXPECT_TRUE(plan.isInverse());
    EXPECT_EQ(plan.config().norm, FFTNorm::ByN);
}

TEST(FFTPlanTest, ZeroSizeThrows) {
    EXPECT_THROW(FFTPlan::create(0), std::invalid_argument);
}

TEST(FFTPlanTest, CooleyTukeyNonPow2Throws) {
    EXPECT_THROW(
        FFTPlan::create(100, {FFTDirection::Forward, FFTNorm::None,
                              FFTAlgorithm::CooleyTukey}),
        std::invalid_argument);
}

TEST(FFTPlanTest, CooleyTukeyPow2Succeeds) {
    EXPECT_NO_THROW(FFTPlan::create(1024, {FFTDirection::Forward, FFTNorm::None,
                                           FFTAlgorithm::CooleyTukey}));
}

TEST(FFTPlanTest, BluesteinArbitraryN) {
    EXPECT_NO_THROW(FFTPlan::create(97,  {}));
    EXPECT_NO_THROW(FFTPlan::create(100, {}));
    EXPECT_NO_THROW(FFTPlan::create(997, {}));
}

TEST(FFTPlanTest, ExecuteSizeMismatchThrows) {
    auto plan = FFTPlan::create(64);
    std::vector<cx> x(32, {1.0, 0.0});
    EXPECT_THROW(plan.execute(x), std::invalid_argument);
}

TEST(FFTPlanTest, InversePlanIsInverse) {
    auto fwd  = FFTPlan::create(128);
    auto inv  = fwd.inversePlan();
    EXPECT_EQ(inv.size(), 128u);
    EXPECT_TRUE(inv.isInverse());
    EXPECT_EQ(inv.config().norm, FFTNorm::ByN);
}

TEST(FFTPlanTest, ExecuteConst) {
    auto plan = FFTPlan::create(16);
    std::vector<cx> x(16, {1.0, 0.0});
    auto original = x;
    auto result   = plan.executeConst(x);
    EXPECT_EQ(x, original);          // original unchanged
    EXPECT_NE(result[0], x[0]);      // result is different (DC bin = 16)
    EXPECT_NEAR(result[0].real(), 16.0, kTight);
}

TEST(FFTPlanTest, BackendNameCooleyTukey) {
    auto plan = FFTPlan::create(64, {FFTDirection::Forward, FFTNorm::None,
                                     FFTAlgorithm::CooleyTukey});
    EXPECT_NE(std::string(plan.backendName()).find("Cooley"), std::string::npos);
}

TEST(FFTPlanTest, BackendNameBluestein) {
    auto plan = FFTPlan::create(100, {FFTDirection::Forward, FFTNorm::None,
                                      FFTAlgorithm::Bluestein});
    EXPECT_NE(std::string(plan.backendName()).find("Bluestein"), std::string::npos);
}

// ═════════════════════════════════════════════════════════════════════════════
// FFT mathematical correctness — Cooley-Tukey (power-of-2)
// ═════════════════════════════════════════════════════════════════════════════

TEST(FFTMathTest, DCSignalPow2) {
    // x[n] = 1 → X[0] = N, X[k≠0] = 0
    const size_t N = 64;
    std::vector<cx> x(N, {1.0, 0.0});
    fft(x, FFTNorm::None);
    EXPECT_NEAR(x[0].real(), static_cast<double>(N), kTight);
    EXPECT_NEAR(x[0].imag(), 0.0, kTight);
    for (size_t k = 1; k < N; ++k)
        EXPECT_NEAR(std::abs(x[k]), 0.0, kTight) << "bin " << k;
}

TEST(FFTMathTest, ImpulsePow2) {
    // x = δ[0] → X[k] = 1 for all k
    const size_t N = 128;
    std::vector<cx> x(N, {0.0, 0.0});
    x[0] = {1.0, 0.0};
    fft(x, FFTNorm::None);
    for (size_t k = 0; k < N; ++k)
        EXPECT_NEAR(std::abs(x[k]), 1.0, kTight) << "bin " << k;
}

TEST(FFTMathTest, ImpulseAtN4) {
    // δ[n₀] → X[k] = exp(-2πi·k·n₀/N)
    const size_t N = 32, n0 = 5;
    std::vector<cx> x(N, {0.0, 0.0});
    x[n0] = {1.0, 0.0};
    fft(x, FFTNorm::None);
    for (size_t k = 0; k < N; ++k) {
        double ang = -2.0 * M_PI * static_cast<double>(k * n0) / static_cast<double>(N);
        cx expected{std::cos(ang), std::sin(ang)};
        EXPECT_NEAR(x[k].real(), expected.real(), kTight) << "bin " << k;
        EXPECT_NEAR(x[k].imag(), expected.imag(), kTight) << "bin " << k;
    }
}

TEST(FFTMathTest, CosineSpike) {
    // x[n] = cos(2π·k₀·n/N) → |X[k₀]| = |X[N-k₀]| = N/2, all others ≈ 0
    const size_t N = 64, k0 = 5;
    std::vector<cx> x(N);
    for (size_t n = 0; n < N; ++n)
        x[n] = std::cos(2.0 * M_PI * k0 * n / N);
    fft(x, FFTNorm::None);
    EXPECT_NEAR(std::abs(x[k0]),     N / 2.0, kTight);
    EXPECT_NEAR(std::abs(x[N - k0]), N / 2.0, kTight);
    for (size_t k = 1; k < N; ++k)
        if (k != k0 && k != N - k0)
            EXPECT_NEAR(std::abs(x[k]), 0.0, kTight) << "bin " << k;
}

TEST(FFTMathTest, ParsevalTheoremPow2) {
    // Σ|x[n]|² = (1/N)·Σ|X[k]|²
    const size_t N = 256;
    auto x = makeMixed(N);
    double ex = 0.0;
    for (auto& v : x) ex += std::norm(v);

    auto X = x;
    fft(X, FFTNorm::None);
    double eX = 0.0;
    for (auto& v : X) eX += std::norm(v);

    EXPECT_NEAR(ex, eX / static_cast<double>(N), kLoose);
}

TEST(FFTMathTest, RoundtripPow2) {
    // IFFT(FFT(x)) ≈ x
    const size_t N = 256;
    auto original = makeMixed(N);
    auto x        = original;
    fft(x);
    ifft(x);  // default norm = ByN
    EXPECT_LT(maxErr(x, original), kTight);
}

TEST(FFTMathTest, RoundtripLargePow2) {
    const size_t N = 4096;
    std::vector<cx> x(N);
    for (size_t i = 0; i < N; ++i)
        x[i] = {static_cast<double>(i % 17) - 8.0, static_cast<double>(i % 13)};
    auto original = x;
    fft(x);
    ifft(x);
    EXPECT_LT(maxErr(x, original), 1e-9);
}

TEST(FFTMathTest, Linearity) {
    // FFT(a·x + b·y) = a·FFT(x) + b·FFT(y)
    const size_t N = 64;
    auto x = makeMixed(N, 3, 7);
    auto y = makeMixed(N, 5, 11);
    cx a{2.0, 1.0}, b{-1.0, 0.5};

    std::vector<cx> lhs(N);
    for (size_t i = 0; i < N; ++i) lhs[i] = a * x[i] + b * y[i];
    fft(lhs);

    auto Fx = x; fft(Fx);
    auto Fy = y; fft(Fy);
    std::vector<cx> rhs(N);
    for (size_t i = 0; i < N; ++i) rhs[i] = a * Fx[i] + b * Fy[i];

    EXPECT_LT(maxErr(lhs, rhs), kTight);
}

TEST(FFTMathTest, ZeroSignal) {
    std::vector<cx> x(64, {0.0, 0.0});
    fft(x);
    for (auto& v : x) EXPECT_NEAR(std::abs(v), 0.0, kTight);
}

TEST(FFTMathTest, AgainstNaiveDFT) {
    // Compare FFT output with O(N²) naive DFT on small N
    const size_t N = 16;
    auto x = makeMixed(N, 2, 5);
    auto ref = naiveDFT(x);

    fft(x);
    EXPECT_LT(maxErr(x, ref), kTight);
}

// ═════════════════════════════════════════════════════════════════════════════
// Normalization modes
// ═════════════════════════════════════════════════════════════════════════════

TEST(FFTNormTest, NoneNoScaling) {
    const size_t N = 32;
    std::vector<cx> x(N, {1.0, 0.0});
    fft(x, FFTNorm::None);
    EXPECT_NEAR(x[0].real(), static_cast<double>(N), kTight);
}

TEST(FFTNormTest, ByNDividesN) {
    const size_t N = 32;
    std::vector<cx> x(N, {1.0, 0.0});
    fft(x, FFTNorm::ByN);
    EXPECT_NEAR(x[0].real(), 1.0, kTight);
    for (size_t k = 1; k < N; ++k)
        EXPECT_NEAR(std::abs(x[k]), 0.0, kTight);
}

TEST(FFTNormTest, BySqrtNScales) {
    const size_t N = 64;
    std::vector<cx> x(N, {1.0, 0.0});
    fft(x, FFTNorm::BySqrtN);
    EXPECT_NEAR(x[0].real(), std::sqrt(static_cast<double>(N)), kTight);
}

TEST(FFTNormTest, BySqrtNUnitaryParseval) {
    // Unitary DFT: Σ|x|² = Σ|X_unit|²  (no 1/N factor)
    const size_t N = 128;
    auto x = makeMixed(N);
    double ex = 0.0;
    for (auto& v : x) ex += std::norm(v);

    fft(x, FFTNorm::BySqrtN);
    double eX = 0.0;
    for (auto& v : x) eX += std::norm(v);

    EXPECT_NEAR(ex, eX, kLoose);
}

TEST(FFTNormTest, RoundtripBySqrtN) {
    const size_t N = 64;
    auto original = makeMixed(N);
    auto x = original;
    fft(x,  FFTNorm::BySqrtN);
    ifft(x, FFTNorm::BySqrtN);  // unitary: apply same norm on inverse
    EXPECT_LT(maxErr(x, original), kTight);
}

// ═════════════════════════════════════════════════════════════════════════════
// Bluestein (non-power-of-2 and explicit Bluestein on pow-of-2)
// ═════════════════════════════════════════════════════════════════════════════

TEST(FFTBluesteinTest, DCArbitraryN) {
    for (size_t N : {7u, 10u, 15u, 100u, 997u}) {
        std::vector<cx> x(N, {1.0, 0.0});
        FFTPlan::create(N).execute(x);
        EXPECT_NEAR(x[0].real(), static_cast<double>(N), kLoose) << "N=" << N;
        for (size_t k = 1; k < N; ++k)
            EXPECT_NEAR(std::abs(x[k]), 0.0, kLoose) << "N=" << N << " k=" << k;
    }
}

TEST(FFTBluesteinTest, ImpulseArbitraryN) {
    for (size_t N : {7u, 11u, 50u, 200u}) {
        std::vector<cx> x(N, {0.0, 0.0});
        x[0] = {1.0, 0.0};
        FFTPlan::create(N).execute(x);
        for (size_t k = 0; k < N; ++k)
            EXPECT_NEAR(std::abs(x[k]), 1.0, kLoose) << "N=" << N << " k=" << k;
    }
}

TEST(FFTBluesteinTest, RoundtripArbitraryN) {
    for (size_t N : {7u, 13u, 100u, 997u}) {
        auto original = makeMixed(N, 2, 4);
        auto x = original;
        FFTPlan::create(N, {FFTDirection::Forward, FFTNorm::None}).execute(x);
        FFTPlan::create(N, {FFTDirection::Inverse, FFTNorm::ByN}).execute(x);
        EXPECT_LT(maxErr(x, original), kLoose) << "N=" << N;
    }
}

TEST(FFTBluesteinTest, MatchesCooleyTukeyOnPow2) {
    // Both algorithms must produce identical results for power-of-2 sizes
    const size_t N = 128;
    auto signal = makeMixed(N);
    auto x_ct = signal, x_bl = signal;

    FFTPlan::create(N, {FFTDirection::Forward, FFTNorm::None,
                        FFTAlgorithm::CooleyTukey}).execute(x_ct);
    FFTPlan::create(N, {FFTDirection::Forward, FFTNorm::None,
                        FFTAlgorithm::Bluestein}).execute(x_bl);

    EXPECT_LT(maxErr(x_ct, x_bl), kLoose);
}

TEST(FFTBluesteinTest, AgainstNaiveDFTPrimeN) {
    const size_t N = 17;  // prime
    auto x = makeMixed(N, 2, 5);
    auto ref = naiveDFT(x);
    FFTPlan::create(N).execute(x);
    EXPECT_LT(maxErr(x, ref), kLoose);
}

TEST(FFTBluesteinTest, ParsevalArbitraryN) {
    for (size_t N : {13u, 100u, 300u}) {
        auto x = makeMixed(N, 1, 3);
        double ex = 0.0;
        for (auto& v : x) ex += std::norm(v);

        FFTPlan::create(N).execute(x);
        double eX = 0.0;
        for (auto& v : x) eX += std::norm(v);

        EXPECT_NEAR(ex, eX / static_cast<double>(N), kLoose * 10) << "N=" << N;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// rfft / irfft
// ═════════════════════════════════════════════════════════════════════════════

TEST(RFFTTest, BinCount) {
    for (size_t N : {8u, 16u, 64u, 127u, 128u}) {
        auto bins = rfft(std::vector<double>(N, 1.0));
        EXPECT_EQ(bins.size(), N / 2 + 1) << "N=" << N;
    }
}

TEST(RFFTTest, DCSignal) {
    const size_t N = 64;
    std::vector<double> x(N, 1.0);
    auto bins = rfft(x);
    EXPECT_NEAR(bins[0].real(), static_cast<double>(N), kTight);
    EXPECT_NEAR(bins[0].imag(), 0.0, kTight);
    for (size_t k = 1; k < bins.size(); ++k)
        EXPECT_NEAR(std::abs(bins[k]), 0.0, kTight) << "bin " << k;
}

TEST(RFFTTest, RoundtripEven) {
    const size_t N = 128;
    std::vector<double> x(N);
    for (size_t i = 0; i < N; ++i) x[i] = std::sin(2.0 * M_PI * 5 * i / N);
    auto bins = rfft(x);
    auto back = irfft(bins, N);
    ASSERT_EQ(back.size(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR(back[i], x[i], kTight) << "i=" << i;
}

TEST(RFFTTest, RoundtripOdd) {
    const size_t N = 65;
    std::vector<double> x(N);
    for (size_t i = 0; i < N; ++i) x[i] = static_cast<double>(i % 7) - 3.0;
    auto bins = rfft(x);
    auto back = irfft(bins, N);
    ASSERT_EQ(back.size(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR(back[i], x[i], kLoose) << "i=" << i;
}

TEST(RFFTTest, HermitianSymmetry) {
    // For real x: full FFT satisfies X[N-k] = conj(X[k])
    const size_t N = 64;
    std::vector<double> x(N);
    for (size_t i = 0; i < N; ++i) x[i] = std::cos(2.0 * M_PI * 3 * i / N);

    // Compute full FFT
    std::vector<cx> cx_x(N);
    for (size_t i = 0; i < N; ++i) cx_x[i] = x[i];
    fft(cx_x);

    // rfft bins must match the first N/2+1 bins of the full FFT
    auto bins = rfft(x);
    for (size_t k = 0; k < bins.size(); ++k) {
        EXPECT_NEAR(bins[k].real(), cx_x[k].real(), kTight) << "k=" << k;
        EXPECT_NEAR(bins[k].imag(), cx_x[k].imag(), kTight) << "k=" << k;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// convolve
// ═════════════════════════════════════════════════════════════════════════════

TEST(ConvolveTest, OutputLength) {
    std::vector<double> a(5, 1.0), b(3, 1.0);
    auto r = convolve(a, b);
    EXPECT_EQ(r.size(), a.size() + b.size() - 1);
}

TEST(ConvolveTest, AgainstNaive) {
    std::vector<double> a = {1, 2, 3, 4, 5};
    std::vector<double> b = {1, -1, 2};
    auto fft_r   = convolve(a, b);
    auto naive_r = naiveConvolve(a, b);
    ASSERT_EQ(fft_r.size(), naive_r.size());
    for (size_t i = 0; i < fft_r.size(); ++i)
        EXPECT_NEAR(fft_r[i], naive_r[i], kLoose) << "i=" << i;
}

TEST(ConvolveTest, AgainstNaiveLarge) {
    const size_t Na = 200, Nb = 150;
    std::vector<double> a(Na), b(Nb);
    for (size_t i = 0; i < Na; ++i) a[i] = std::sin(0.1 * i);
    for (size_t i = 0; i < Nb; ++i) b[i] = std::cos(0.07 * i);
    auto fft_r   = convolve(a, b);
    auto naive_r = naiveConvolve(a, b);
    for (size_t i = 0; i < fft_r.size(); ++i)
        EXPECT_NEAR(fft_r[i], naive_r[i], kLoose) << "i=" << i;
}

TEST(ConvolveTest, Commutativity) {
    std::vector<double> a = {1, 2, 3};
    std::vector<double> b = {4, 5, 6, 7};
    auto ab = convolve(a, b);
    auto ba = convolve(b, a);
    ASSERT_EQ(ab.size(), ba.size());
    for (size_t i = 0; i < ab.size(); ++i)
        EXPECT_NEAR(ab[i], ba[i], kLoose) << "i=" << i;
}

TEST(ConvolveTest, ConvolveWithDelta) {
    // Convolving with unit impulse should return the original signal
    std::vector<double> x = {3, 1, -2, 4, 0.5};
    std::vector<double> delta = {1.0};
    auto r = convolve(x, delta);
    ASSERT_EQ(r.size(), x.size());
    for (size_t i = 0; i < x.size(); ++i)
        EXPECT_NEAR(r[i], x[i], kLoose) << "i=" << i;
}

TEST(ConvolveTest, EmptyInputReturnsEmpty) {
    EXPECT_TRUE(convolve({}, {1.0, 2.0}).empty());
    EXPECT_TRUE(convolve({1.0}, {}).empty());
}

// ═════════════════════════════════════════════════════════════════════════════
// correlate
// ═════════════════════════════════════════════════════════════════════════════

TEST(CorrelateTest, OutputLength) {
    std::vector<double> a(5), b(3);
    auto r = correlate(a, b);
    EXPECT_EQ(r.size(), a.size() + b.size() - 1);
}

TEST(CorrelateTest, AgainstNaive) {
    std::vector<double> a = {1, 2, 3};
    std::vector<double> b = {1, 0, -1};
    auto fft_r   = correlate(a, b);
    auto naive_r = naiveCorrelate(a, b);
    ASSERT_EQ(fft_r.size(), naive_r.size());
    for (size_t i = 0; i < fft_r.size(); ++i)
        EXPECT_NEAR(fft_r[i], naive_r[i], kLoose) << "i=" << i;
}

TEST(CorrelateTest, AutoCorrelationPeak) {
    // Auto-correlation: max at lag=0 (centre of output)
    std::vector<double> x(32);
    for (size_t i = 0; i < 32; ++i) x[i] = std::sin(2.0 * M_PI * 4 * i / 32);
    auto r = correlate(x, x);
    size_t centre = x.size() - 1;  // zero-lag index
    double peak = r[centre];
    for (size_t i = 0; i < r.size(); ++i)
        if (i != centre)
            EXPECT_LE(std::abs(r[i]), std::abs(peak) + kLoose) << "i=" << i;
}

// ═════════════════════════════════════════════════════════════════════════════
// Spectral helper functions
// ═════════════════════════════════════════════════════════════════════════════

TEST(SpectralHelperTest, MagnitudeImpulse) {
    const size_t N = 16;
    std::vector<cx> x(N, {0.0, 0.0});
    x[0] = {1.0, 0.0};
    fft(x);
    auto mag = magnitude(x);
    for (double v : mag) EXPECT_NEAR(v, 1.0, kTight);
}

TEST(SpectralHelperTest, PowerSpectrumIsSquaredMagnitude) {
    std::vector<cx> x = {{3.0, 4.0}, {1.0, -2.0}, {0.0, 5.0}};
    auto mag = magnitude(x);
    auto ps  = powerSpectrum(x);
    for (size_t i = 0; i < x.size(); ++i)
        EXPECT_NEAR(ps[i], mag[i] * mag[i], kTight) << "i=" << i;
}

TEST(SpectralHelperTest, PhaseRange) {
    std::vector<cx> x = {{1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}};
    auto ph = phase(x);
    EXPECT_NEAR(ph[0],  0.0,        kTight);
    EXPECT_NEAR(ph[1],  M_PI / 2,   kTight);
    EXPECT_NEAR(std::abs(ph[2]), M_PI, kTight);
    EXPECT_NEAR(ph[3], -M_PI / 2,   kTight);
}

TEST(SpectralHelperTest, PowerSpectrumDBFloorAtMinusInf) {
    // Zero-magnitude bin should not produce -inf (guarded by 1e-300)
    std::vector<cx> x = {{0.0, 0.0}};
    auto db = powerSpectrumDB(x);
    EXPECT_TRUE(std::isfinite(db[0]));
    EXPECT_LT(db[0], -2900.0);  // 10·log10(1e-300) ≈ -3000
}

TEST(SpectralHelperTest, MagnitudeDB) {
    // 20·log10(1) = 0, 20·log10(10) = 20
    std::vector<cx> x = {{1.0, 0.0}, {10.0, 0.0}};
    auto db = magnitudeDB(x);
    EXPECT_NEAR(db[0],  0.0, kTight);
    EXPECT_NEAR(db[1], 20.0, kTight);
}

TEST(SpectralHelperTest, FFTFrequenciesLength) {
    auto f = fftFrequencies(256, 44100.0);
    EXPECT_EQ(f.size(), 256u);
}

TEST(SpectralHelperTest, FFTFrequenciesDC) {
    auto f = fftFrequencies(256, 44100.0);
    EXPECT_DOUBLE_EQ(f[0], 0.0);
}

TEST(SpectralHelperTest, FFTFrequenciesNyquist) {
    const size_t N = 256;
    double fs = 44100.0;
    auto f = fftFrequencies(N, fs);
    // Bin N/2 = Nyquist
    EXPECT_NEAR(f[N / 2], fs / 2.0, 1e-6);
}

TEST(SpectralHelperTest, RFFTFrequenciesLength) {
    auto f = rfftFrequencies(128, 8000.0);
    EXPECT_EQ(f.size(), 128u / 2 + 1);
}

TEST(SpectralHelperTest, RFFTFrequenciesMonotone) {
    auto f = rfftFrequencies(64, 1000.0);
    for (size_t i = 1; i < f.size(); ++i)
        EXPECT_GT(f[i], f[i - 1]);
}

// ═════════════════════════════════════════════════════════════════════════════
// fftShift / ifftShift
// ═════════════════════════════════════════════════════════════════════════════

TEST(FFTShiftTest, ShiftDCToCenter) {
    // After fftShift, the DC bin (index 0) should move to index N/2
    const size_t N = 8;
    std::vector<cx> x(N);
    for (size_t i = 0; i < N; ++i) x[i] = static_cast<double>(i);
    auto shifted = fftShift(x);
    // DC component (value 0) is now at position N/2
    EXPECT_NEAR(shifted[N / 2].real(), 0.0, kTight);
}

TEST(FFTShiftTest, RoundtripEven) {
    const size_t N = 8;
    std::vector<cx> x(N);
    for (size_t i = 0; i < N; ++i) x[i] = static_cast<double>(i);
    auto shifted   = fftShift(x);
    auto restored  = ifftShift(shifted);
    EXPECT_LT(maxErr(x, restored), kTight);
}

TEST(FFTShiftTest, RoundtripOdd) {
    const size_t N = 7;
    std::vector<cx> x(N);
    for (size_t i = 0; i < N; ++i) x[i] = static_cast<double>(i * i);
    auto r = ifftShift(fftShift(x));
    EXPECT_LT(maxErr(x, r), kTight);
}

TEST(FFTShiftTest, EmptyInput) {
    std::vector<cx> empty;
    EXPECT_TRUE(fftShift(empty).empty());
    EXPECT_TRUE(ifftShift(empty).empty());
}

// ═════════════════════════════════════════════════════════════════════════════
// applyWindow / windowedFFT
// ═════════════════════════════════════════════════════════════════════════════

TEST(WindowedFFTTest, ApplyWindowScales) {
    std::vector<double> sig(8, 1.0);
    std::vector<double> win = {0, 0.5, 1, 0.5, 0, 0.5, 1, 0.5};
    auto r = applyWindow(sig, win);
    for (size_t i = 0; i < sig.size(); ++i)
        EXPECT_NEAR(r[i], win[i], kTight) << "i=" << i;
}

TEST(WindowedFFTTest, ApplyWindowSizeMismatch) {
    EXPECT_THROW(applyWindow({1.0, 2.0}, {1.0}), std::invalid_argument);
}

TEST(WindowedFFTTest, WindowedFFTOutputSize) {
    const size_t N = 64;
    std::vector<double> sig(N, 1.0);
    auto win = windowHann(N);
    auto r   = windowedFFT(sig, win);
    EXPECT_EQ(r.size(), N);
}

TEST(WindowedFFTTest, WindowedFFTReducesDCForHann) {
    // Hann window is zero at endpoints → reduces overall energy vs rectangular
    const size_t N = 64;
    std::vector<double> dc(N, 1.0);

    std::vector<cx> rect_fft(N);
    for (size_t i = 0; i < N; ++i) rect_fft[i] = 1.0;
    fft(rect_fft);

    auto hann = windowHann(N, false);  // periodic for spectral analysis
    auto hann_fft = windowedFFT(dc, hann);

    double rect_dc = std::abs(rect_fft[0]);
    double hann_dc = std::abs(hann_fft[0]);
    EXPECT_LT(hann_dc, rect_dc);
}
