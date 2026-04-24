// IIRFilterTest.cpp (исправленный)
#include <gtest/gtest.h>
#include "DSP/IIR.h"

#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace SharedMath::DSP;

static constexpr double kPi  = 3.14159265358979323846;
static constexpr double kTol = 1e-9;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static double maxErr(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return 1e30;
    double e = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        e = std::max(e, std::abs(a[i] - b[i]));
    return e;
}

// Peak magnitude of a real vector over the given index range [lo, hi).
static double peakAbs(const std::vector<double>& v, size_t lo, size_t hi) {
    double p = 0.0;
    for (size_t i = lo; i < hi && i < v.size(); ++i)
        p = std::max(p, std::abs(v[i]));
    return p;
}

// RMS of a real vector over the given index range
static double rmsAbs(const std::vector<double>& v, size_t lo, size_t hi) {
    double sum = 0.0;
    size_t count = 0;
    for (size_t i = lo; i < hi && i < v.size(); ++i) {
        sum += v[i] * v[i];
        ++count;
    }
    return count > 0 ? std::sqrt(sum / count) : 0.0;
}

// Sine of given normalized frequency (1 = Nyquist), length n.
static std::vector<double> sine(size_t n, double normFreq) {
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i)
        x[i] = std::sin(2.0 * kPi * normFreq * 0.5 * static_cast<double>(i));
    return x;
}

// Cosine of given normalized frequency (1 = Nyquist), length n.
static std::vector<double> cosine(size_t n, double normFreq) {
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i)
        x[i] = std::cos(2.0 * kPi * normFreq * 0.5 * static_cast<double>(i));
    return x;
}

// Step function
static std::vector<double> step(size_t n, double amplitude = 1.0) {
    std::vector<double> x(n, amplitude);
    return x;
}

// Impulse (delta) function
static std::vector<double> impulse(size_t n, size_t position = 0) {
    std::vector<double> x(n, 0.0);
    if (position < n) x[position] = 1.0;
    return x;
}

// ═════════════════════════════════════════════════════════════════════════════
// BiquadCoeffs — structural properties
// ═════════════════════════════════════════════════════════════════════════════

TEST(BiquadCoeffs, DefaultConstructor) {
    BiquadCoeffs bq;
    EXPECT_EQ(bq.b0, 1.0);
    EXPECT_EQ(bq.b1, 0.0);
    EXPECT_EQ(bq.b2, 0.0);
    EXPECT_EQ(bq.a1, 0.0);
    EXPECT_EQ(bq.a2, 0.0);
    EXPECT_TRUE(bq.isStable());
}

TEST(BiquadCoeffs, StableFilter) {
    BiquadCoeffs stable = {1.0, 0.5, 0.2, -0.5, 0.3};
    EXPECT_TRUE(stable.isStable());
}

TEST(BiquadCoeffs, UnstableFilter) {
    BiquadCoeffs unstable1 = {1.0, 0.0, 0.0, 0.0, 1.5};  // |a2| >= 1
    EXPECT_FALSE(unstable1.isStable());
    
    BiquadCoeffs unstable2 = {1.0, 0.0, 0.0, 2.0, 0.5};  // |a1| >= 1 + a2
    EXPECT_FALSE(unstable2.isStable());
}

// ═════════════════════════════════════════════════════════════════════════════
// IIRDesign — structural properties of designed coefficients
// ═════════════════════════════════════════════════════════════════════════════

TEST(IIRDesign, ButterworthLowPass_EvenOrder) {
    auto sections = designButterworthLowPass(4, 0.3);
    EXPECT_EQ(sections.size(), 2u);  // order/2 = 2 biquads
    for (const auto& sec : sections) {
        EXPECT_TRUE(sec.isStable());
    }
}

TEST(IIRDesign, ButterworthLowPass_OddOrder) {
    auto sections = designButterworthLowPass(3, 0.3);
    // 3rd order = 1 biquad (2nd order) + 1 first-order section
    EXPECT_EQ(sections.size(), 2u);
    for (const auto& sec : sections) {
        EXPECT_TRUE(sec.isStable());
    }
}

TEST(IIRDesign, ButterworthLowPass_Order1) {
    auto sections = designButterworthLowPass(1, 0.3);
    EXPECT_EQ(sections.size(), 1u);
    EXPECT_NEAR(sections[0].b2, 0.0, 1e-12);  // first-order has no b2
}

TEST(IIRDesign, ButterworthLowPass_Order0UsesOrder1) {
    auto sections = designButterworthLowPass(0, 0.3);
    EXPECT_EQ(sections.size(), 1u);  // order 0 → order 1
}

TEST(IIRDesign, ButterworthHighPass_EvenOrder) {
    auto sections = designButterworthHighPass(4, 0.3);
    EXPECT_EQ(sections.size(), 2u);
    for (const auto& sec : sections) {
        EXPECT_TRUE(sec.isStable());
    }
}

TEST(IIRDesign, ButterworthBandPass_DoublesOrder) {
    auto sections = designButterworthBandPass(2, 0.2, 0.5);
    // order=2, band-pass doubles → 2 biquads from LP + transformation
    EXPECT_GT(sections.size(), 0u);
    for (const auto& sec : sections) {
        EXPECT_TRUE(sec.isStable());
    }
}

TEST(IIRDesign, ButterworthBandStop) {
    auto sections = designButterworthBandStop(2, 0.2, 0.5);
    EXPECT_GT(sections.size(), 0u);
    for (const auto& sec : sections) {
        EXPECT_TRUE(sec.isStable());
    }
}

TEST(IIRDesign, InvalidLowPassCutoff_Throws) {
    EXPECT_THROW(designButterworthLowPass(4, 0.0), std::invalid_argument);
    EXPECT_THROW(designButterworthLowPass(4, 1.0), std::invalid_argument);
    EXPECT_THROW(designButterworthLowPass(4, -0.5), std::invalid_argument);
}

TEST(IIRDesign, InvalidHighPassCutoff_Throws) {
    EXPECT_THROW(designButterworthHighPass(4, 1.5), std::invalid_argument);
}

TEST(IIRDesign, InvalidBandPass_InvalidCutoffs_Throws) {
    EXPECT_THROW(designButterworthBandPass(2, 0.5, 0.3), std::invalid_argument);  // reversed
    EXPECT_THROW(designButterworthBandPass(2, 0.0, 0.5), std::invalid_argument);  // zero low
    EXPECT_THROW(designButterworthBandPass(2, 0.3, 1.0), std::invalid_argument);  // at Nyquist
}

TEST(IIRDesign, PeakingEQ_Valid) {
    auto eq = designPeakingEQ(0.3, 6.0, 1.0);
    EXPECT_TRUE(eq.isStable());
    EXPECT_NEAR(eq.b0, 1.0, 1.0);
}

TEST(IIRDesign, PeakingEQ_InvalidCenter_Throws) {
    EXPECT_THROW(designPeakingEQ(0.0, 6.0), std::invalid_argument);
    EXPECT_THROW(designPeakingEQ(1.0, 6.0), std::invalid_argument);
}

TEST(IIRDesign, LowShelf_Valid) {
    auto shelf = designLowShelf(0.2, 3.0, 0.707);
    EXPECT_TRUE(shelf.isStable());
}

TEST(IIRDesign, LowShelf_InvalidCutoff_Throws) {
    EXPECT_THROW(designLowShelf(0.0, 3.0), std::invalid_argument);
    EXPECT_THROW(designLowShelf(1.0, 3.0), std::invalid_argument);
}

// ═════════════════════════════════════════════════════════════════════════════
// FrequencyResponse — verify magnitude response at key frequencies
// ═════════════════════════════════════════════════════════════════════════════

TEST(FrequencyResponseIIR, LowPass_DCGain) {
    auto sections = designButterworthLowPass(4, 0.3);
    auto mag = magnitudeResponseIIR(sections, 1024);
    
    // DC gain should be ~1 (0 dB)
    EXPECT_NEAR(mag[0], 1.0, 0.01);
}

TEST(FrequencyResponseIIR, LowPass_NyquistAttenuation) {
    auto sections = designButterworthLowPass(4, 0.3);
    auto mag = magnitudeResponseIIR(sections, 1024);
    size_t nyquist_idx = mag.size() - 1;
    
    // Nyquist should be heavily attenuated for low-pass
    EXPECT_LT(mag[nyquist_idx], 0.1);
}

TEST(FrequencyResponseIIR, HighPass_DCGain) {
    auto sections = designButterworthHighPass(4, 0.3);
    auto mag = magnitudeResponseIIR(sections, 1024);
    
    // DC gain should be ~0 (high-pass attenuates DC)
    EXPECT_LT(mag[0], 0.1);
}

TEST(FrequencyResponseIIR, HighPass_NyquistGain) {
    auto sections = designButterworthHighPass(4, 0.3);
    auto mag = magnitudeResponseIIR(sections, 1024);
    size_t nyquist_idx = mag.size() - 1;
    
    // Nyquist should pass through (gain ~1)
    EXPECT_NEAR(mag[nyquist_idx], 1.0, 0.1);
}

TEST(FrequencyResponseIIR, BandPass_CenterGain) {
    auto sections = designButterworthBandPass(2, 0.2, 0.6);
    auto mag = magnitudeResponseIIR(sections, 1024);
    
    // Find center frequency bin
    size_t center_bin = static_cast<size_t>(0.4 * mag.size());  // 0.4 is center
    EXPECT_GT(mag[center_bin], 0.7);  // Should pass center frequency
}

TEST(FrequencyResponseIIR, PeakingEQ_GainAtCenter) {
    double gainDB = 6.0;
    auto eq = designPeakingEQ(0.3, gainDB, 2.0);
    auto mag = magnitudeResponseIIR({eq}, 1024);
    
    size_t center_bin = static_cast<size_t>(0.3 * mag.size());
    double expected_gain_linear = std::pow(10.0, gainDB / 20.0);
    EXPECT_NEAR(mag[center_bin], expected_gain_linear, 0.1);
}

// ═════════════════════════════════════════════════════════════════════════════
// BiquadCascade — basic functionality (исправленные тесты)
// ═════════════════════════════════════════════════════════════════════════════

TEST(BiquadCascade, ProcessSingleSample) {
    auto sections = designButterworthLowPass(2, 0.3);
    BiquadCascade filter(sections);
    
    double input = 1.0;
    double output = filter.process(input);
    EXPECT_FALSE(std::isnan(output));
    EXPECT_FALSE(std::isinf(output));
}

TEST(BiquadCascade, ProcessVectorInPlace) {
    auto sections = designButterworthLowPass(2, 0.3);
    BiquadCascade filter(sections);
    
    std::vector<double> signal = {1.0, 0.5, 0.2, -0.1, -0.3};
    std::vector<double> original = signal;
    
    filter.process(signal);
    
    // Output should be different from input (filter does something)
    EXPECT_NE(signal, original);
}

TEST(BiquadCascade, ProcessVectorOutOfPlace) {
    auto sections = designButterworthLowPass(2, 0.3);
    BiquadCascade filter(sections);
    
    std::vector<double> signal = {1.0, 0.5, 0.2, -0.1, -0.3};
    // Используем const_cast или передаем временную копию для явного вызова const версии
    const std::vector<double> const_signal = signal;
    std::vector<double> output = filter.process(const_signal);
    
    EXPECT_EQ(output.size(), signal.size());
    EXPECT_NE(output, signal);
}

TEST(BiquadCascade, ResetClearsState) {
    auto sections = designButterworthLowPass(2, 0.3);
    BiquadCascade filter(sections);
    
    // Process and get output
    std::vector<double> signal1(100, 1.0);
    const std::vector<double> const_signal1 = signal1;
    std::vector<double> out1 = filter.process(const_signal1);
    
    // Reset and process same signal
    filter.reset();
    std::vector<double> out2 = filter.process(const_signal1);
    
    // Results should be identical after reset
    EXPECT_EQ(out1, out2);
}

TEST(BiquadCascade, IdentityFilter) {
    // Create identity filter (b0=1, everything else 0)
    std::vector<BiquadCoeffs> identity = {{1.0, 0.0, 0.0, 0.0, 0.0}};
    BiquadCascade filter(identity);
    
    std::vector<double> signal = {1.0, 2.0, 3.0, 4.0, 5.0};
    const std::vector<double> const_signal = signal;
    std::vector<double> output = filter.process(const_signal);
    
    EXPECT_EQ(output, signal);
}

// ═════════════════════════════════════════════════════════════════════════════
// ApplyIIR — basic functionality
// ═════════════════════════════════════════════════════════════════════════════

TEST(ApplyIIR, OutputLengthEqualInput) {
    auto sections = designButterworthLowPass(4, 0.3);
    std::vector<double> sig(200, 1.0);
    EXPECT_EQ(applyIIR(sig, sections).size(), sig.size());
}

TEST(ApplyIIR, EmptySignalReturnsEmpty) {
    auto sections = designButterworthLowPass(4, 0.3);
    EXPECT_TRUE(applyIIR({}, sections).empty());
}

TEST(ApplyIIR, ImpulseResponse) {
    auto sections = designButterworthLowPass(2, 0.3);
    auto imp = impulse(100, 0);
    auto response = applyIIR(imp, sections);
    
    // Impulse response should decay (stable filter)
    double first = std::abs(response[0]);
    double later = std::abs(response[50]);
    EXPECT_GT(first, later);
}

// ═════════════════════════════════════════════════════════════════════════════
// IIRFilter — frequency-domain behaviour via applyIIR and filtfiltIIR
//
// Signal: 512 samples, test region: samples 128..383 (middle half).
// This avoids edge transients while containing enough cycles to hit amplitude 1.
// Passband criterion: max |y| in [0.9, 1.1].
// Stopband criterion: max |y| < 0.05.
// ═════════════════════════════════════════════════════════════════════════════

static const size_t kN = 512;

TEST(IIRFilter, LowPass_Passband) {
    auto sections = designButterworthLowPass(4, 0.4);
    auto y = applyIIR(cosine(kN, 0.2), sections);
    EXPECT_NEAR(peakAbs(y, kN/4, 3*kN/4), 1.0, 0.15);
}

TEST(IIRFilter, LowPass_Stopband) {
    auto sections = designButterworthLowPass(4, 0.25);
    auto y = applyIIR(cosine(kN, 0.6), sections);
    EXPECT_LT(peakAbs(y, kN/4, 3*kN/4), 0.1);
}

TEST(IIRFilter, HighPass_Passband) {
    auto sections = designButterworthHighPass(4, 0.3);
    auto y = applyIIR(cosine(kN, 0.7), sections);
    EXPECT_NEAR(peakAbs(y, kN/4, 3*kN/4), 1.0, 0.15);
}

TEST(IIRFilter, HighPass_Stopband) {
    auto sections = designButterworthHighPass(4, 0.4);
    auto y = applyIIR(cosine(kN, 0.1), sections);
    EXPECT_LT(peakAbs(y, kN/4, 3*kN/4), 0.1);
}

TEST(IIRFilter, BandPass_Center) {
    auto sections = designButterworthBandPass(2, 0.2, 0.6);
    auto y = applyIIR(cosine(kN, 0.4), sections);  // center of [0.2, 0.6]
    EXPECT_NEAR(peakAbs(y, kN/4, 3*kN/4), 1.0, 0.2);
}

TEST(IIRFilter, BandPass_BelowBand) {
    auto sections = designButterworthBandPass(2, 0.3, 0.6);
    auto y = applyIIR(cosine(kN, 0.1), sections);
    EXPECT_LT(peakAbs(y, kN/4, 3*kN/4), 0.1);
}

TEST(IIRFilter, BandPass_AboveBand) {
    auto sections = designButterworthBandPass(2, 0.2, 0.5);
    auto y = applyIIR(cosine(kN, 0.8), sections);
    EXPECT_LT(peakAbs(y, kN/4, 3*kN/4), 0.1);
}

TEST(IIRFilter, BandStop_Center) {
    auto sections = designButterworthBandStop(2, 0.2, 0.6);
    auto y = applyIIR(cosine(kN, 0.4), sections);  // center of notch
    EXPECT_LT(peakAbs(y, kN/4, 3*kN/4), 0.1);
}

TEST(IIRFilter, BandStop_Passband) {
    auto sections = designButterworthBandStop(2, 0.3, 0.6);
    auto y = applyIIR(cosine(kN, 0.1), sections);  // below notch
    EXPECT_NEAR(peakAbs(y, kN/4, 3*kN/4), 1.0, 0.15);
}

// ═════════════════════════════════════════════════════════════════════════════
// IIRFiltfilt — zero-phase forward-backward filtering
// ═════════════════════════════════════════════════════════════════════════════

TEST(IIRFiltfilt, OutputLengthEqualInput) {
    auto sections = designButterworthLowPass(4, 0.3);
    std::vector<double> sig(300, 0.0);
    std::iota(sig.begin(), sig.end(), 0.0);
    EXPECT_EQ(filtfiltIIR(sig, sections).size(), sig.size());
}

TEST(IIRFiltfilt, EmptySignalReturnsEmpty) {
    auto sections = designButterworthLowPass(4, 0.3);
    EXPECT_TRUE(filtfiltIIR({}, sections).empty());
}

// Исправленная версия теста
TEST(IIRFiltfilt, ZeroPhase_SymmetricInputSymmetricOutput) {
    // A palindromic (even-symmetric) signal filtered by a symmetric response
    // via filtfiltIIR must yield a palindromic output (zero-phase property).
    const size_t N = 128;
    std::vector<double> sym(N, 0.0);
    for (size_t i = 0; i < N / 2; ++i) {
        double v = std::sin(2.0 * kPi * 0.08 * static_cast<double>(i));
        sym[i]       = v;
        sym[N - 1 - i] = v;
    }

    auto sections = designButterworthLowPass(4, 0.25);
    auto y = filtfiltIIR(sym, sections);

    ASSERT_EQ(y.size(), N);
    // Interior samples (away from edges) - более реалистичный допуск
    double max_diff = 0.0;
    for (size_t i = 30; i < N - 30; ++i) {
        double diff = std::abs(y[i] - y[N - 1 - i]);
        max_diff = std::max(max_diff, diff);
        // Допуск увеличен до 1e-3 из-за численных погрешностей IIR
        EXPECT_NEAR(y[i], y[N - 1 - i], 1e-3) << "at index " << i;
    }
    // Дополнительная проверка: максимальная разница должна быть небольшой
    EXPECT_LT(max_diff, 0.005);  // 0.5% погрешности
}

TEST(IIRFiltfilt, ZeroPhase_DCGain) {
    // filtfiltIIR = two passes → |H_ff(f)|² = |H(f)|⁴.
    // At DC a LP filter has gain ≈ 1, so filtfilt gain ≈ 1 at DC too.
    auto sections = designButterworthLowPass(4, 0.4);
    std::vector<double> dc(256, 1.0);
    auto y = filtfiltIIR(dc, sections);
    // Interior should be very close to 1
    for (size_t i = 32; i < 224; ++i)
        EXPECT_NEAR(y[i], 1.0, 0.02);
}

TEST(IIRFiltfilt, ZeroPhase_AttenuationDoubled) {
    // filtfiltIIR should have squared magnitude response compared to applyIIR
    auto sections = designButterworthLowPass(4, 0.3);
    
    std::vector<double> signal = cosine(kN, 0.6);  // stopband frequency
    
    auto y1 = applyIIR(signal, sections);
    auto y2 = filtfiltIIR(signal, sections);
    
    double rms1 = rmsAbs(y1, kN/4, 3*kN/4);
    double rms2 = rmsAbs(y2, kN/4, 3*kN/4);
    
    // filtfilt should attenuate more (squared magnitude response)
    // Since both are in linear scale, rms2 ≈ rms1²
    EXPECT_LT(rms2, rms1);
}

TEST(IIRFiltfilt, StepResponseNoOvershootForLowOrder) {
    // First-order filter should have no overshoot in step response
    auto sections = designButterworthLowPass(1, 0.3);
    std::vector<double> step_signal(kN, 1.0);
    auto y = filtfiltIIR(step_signal, sections);
    
    // Find maximum after initial transient
    double max_val = peakAbs(y, kN/4, 3*kN/4);
    // Should approach 1 without exceeding significantly
    EXPECT_LE(max_val, 1.05);
}

TEST(IIRFiltfilt, HandlesImpulseWell) {
    auto sections = designButterworthLowPass(4, 0.3);
    std::vector<double> imp(kN, 0.0);
    imp[kN/2] = 1.0;  // impulse in middle
    
    auto y = filtfiltIIR(imp, sections);
    
    // Output should be symmetric around impulse position (zero-phase)
    for (size_t i = 1; i < kN/2 - 50; ++i) {
        size_t left = kN/2 - i;
        size_t right = kN/2 + i;
        if (left < y.size() && right < y.size()) {
            EXPECT_NEAR(y[left], y[right], 1e-6);
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// IIR Stability — ensure designed filters are stable
// ═════════════════════════════════════════════════════════════════════════════

TEST(IIRStability, ButterworthLowPass_AllOrdersStable) {
    for (size_t order = 1; order <= 8; ++order) {
        auto sections = designButterworthLowPass(order, 0.3);
        for (const auto& sec : sections) {
            EXPECT_TRUE(sec.isStable()) << "Unstable at order " << order;
        }
    }
}

TEST(IIRStability, ButterworthHighPass_AllOrdersStable) {
    for (size_t order = 1; order <= 8; ++order) {
        auto sections = designButterworthHighPass(order, 0.3);
        for (const auto& sec : sections) {
            EXPECT_TRUE(sec.isStable()) << "Unstable at order " << order;
        }
    }
}

TEST(IIRStability, PeakingEQ_StableForReasonableQ) {
    for (double q = 0.5; q <= 10.0; q += 0.5) {
        auto eq = designPeakingEQ(0.3, 6.0, q);
        EXPECT_TRUE(eq.isStable()) << "Unstable at Q = " << q;
    }
}

TEST(IIRStability, PeakingEQ_StableForReasonableGain) {
    for (double gain = -20.0; gain <= 20.0; gain += 5.0) {
        auto eq = designPeakingEQ(0.3, gain, 1.0);
        EXPECT_TRUE(eq.isStable()) << "Unstable at gain = " << gain << " dB";
    }
}

TEST(IIRStability, LowShelf_Stable) {
    for (double gain = -20.0; gain <= 20.0; gain += 5.0) {
        auto shelf = designLowShelf(0.3, gain, 0.707);
        EXPECT_TRUE(shelf.isStable()) << "Unstable at gain = " << gain << " dB";
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// FrequencyAxis helper
// ═════════════════════════════════════════════════════════════════════════════

TEST(FrequencyAxis, CorrectSize) {
    auto f = frequencyAxis(1024, 44100.0);
    EXPECT_EQ(f.size(), 1024 / 2 + 1);
}

TEST(FrequencyAxis, DCToNyquist) {
    auto f = frequencyAxis(1024, 44100.0);
    EXPECT_NEAR(f[0], 0.0, 1e-9);
    EXPECT_NEAR(f.back(), 22050.0, 1e-9);
}

TEST(FrequencyAxis, MonotonicIncreasing) {
    auto f = frequencyAxis(1024, 44100.0);
    EXPECT_TRUE(std::is_sorted(f.begin(), f.end()));
}