#include <gtest/gtest.h>
#include <cmath>
#include "NumericalMethods/Integration.h"

using namespace SharedMath::NumericalMethods;

// ─────────────────────────────────────────────────────────────────────────────
// Shared test functions and their exact integrals
// ─────────────────────────────────────────────────────────────────────────────

namespace {
    // ∫₀¹ x dx = 0.5
    auto f_linear = [](double x){ return x; };

    // ∫₀¹ x² dx = 1/3
    auto f_quad   = [](double x){ return x * x; };

    // ∫₀¹ eˣ dx = e - 1
    auto f_exp    = [](double x){ return std::exp(x); };

    // ∫₀^π sin(x) dx = 2
    auto f_sin    = [](double x){ return std::sin(x); };

    // ∫₁² ln(x) dx = 2*ln(2) - 1
    auto f_log    = [](double x){ return std::log(x); };
} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// integrate_rect
// ─────────────────────────────────────────────────────────────────────────────

TEST(Integration, RectLinearExact) {
    // Midpoint rule is exact for constants and affine functions
    EXPECT_NEAR(integrate_rect(f_linear, 0.0, 1.0, 1000), 0.5, 1e-6);
}

TEST(Integration, RectQuadratic) {
    EXPECT_NEAR(integrate_rect(f_quad, 0.0, 1.0, 10000), 1.0 / 3.0, 1e-5);
}

TEST(Integration, RectSin) {
    EXPECT_NEAR(integrate_rect(f_sin, 0.0, M_PI, 10000), 2.0, 1e-5);
}

// ─────────────────────────────────────────────────────────────────────────────
// integrate_trap
// ─────────────────────────────────────────────────────────────────────────────

TEST(Integration, TrapLinearExact) {
    // Trapezoidal rule is exact for linear functions
    EXPECT_NEAR(integrate_trap(f_linear, 0.0, 1.0, 10), 0.5, 1e-14);
}

TEST(Integration, TrapExp) {
    EXPECT_NEAR(integrate_trap(f_exp, 0.0, 1.0, 10000), std::exp(1.0) - 1.0, 1e-5);
}

TEST(Integration, TrapSin) {
    EXPECT_NEAR(integrate_trap(f_sin, 0.0, M_PI, 10000), 2.0, 1e-5);
}

TEST(Integration, TrapLog) {
    double exact = 2.0 * std::log(2.0) - 1.0;
    EXPECT_NEAR(integrate_trap(f_log, 1.0, 2.0, 10000), exact, 1e-5);
}

// ─────────────────────────────────────────────────────────────────────────────
// integrate_simpson
// ─────────────────────────────────────────────────────────────────────────────

TEST(Integration, SimpsonQuadraticExact) {
    // Simpson's 1/3 is exact for polynomials up to degree 3
    EXPECT_NEAR(integrate_simpson(f_quad, 0.0, 1.0, 2), 1.0 / 3.0, 1e-14);
}

TEST(Integration, SimpsonCubic) {
    // ∫₀¹ x³ dx = 1/4
    auto f = [](double x){ return x*x*x; };
    EXPECT_NEAR(integrate_simpson(f, 0.0, 1.0, 2), 0.25, 1e-14);
}

TEST(Integration, SimpsonExp) {
    EXPECT_NEAR(integrate_simpson(f_exp, 0.0, 1.0, 1000), std::exp(1.0) - 1.0, 1e-8);
}

TEST(Integration, SimpsonSin) {
    EXPECT_NEAR(integrate_simpson(f_sin, 0.0, M_PI, 1000), 2.0, 1e-8);
}

TEST(Integration, SimpsonOddNAutoCorrected) {
    // Odd n must be silently rounded up to even — result should still be correct
    EXPECT_NEAR(integrate_simpson(f_quad, 0.0, 1.0, 3), 1.0 / 3.0, 1e-10);
}

// ─────────────────────────────────────────────────────────────────────────────
// integrate_gauss
// ─────────────────────────────────────────────────────────────────────────────

TEST(Integration, GaussOrder1Constant) {
    // Order-1 GL is midpoint rule, exact for constants
    EXPECT_NEAR(integrate_gauss([](double){ return 3.0; }, 0.0, 2.0, 1), 6.0, 1e-14);
}

TEST(Integration, GaussOrder2Linear) {
    // Order-2 GL is exact for degree-3 polynomials
    EXPECT_NEAR(integrate_gauss(f_linear, 0.0, 1.0, 2), 0.5, 1e-14);
}

TEST(Integration, GaussOrder5Exp) {
    EXPECT_NEAR(integrate_gauss(f_exp, 0.0, 1.0, 5), std::exp(1.0) - 1.0, 1e-7);
}

TEST(Integration, GaussOrder5Sin) {
    EXPECT_NEAR(integrate_gauss(f_sin, 0.0, M_PI, 5), 2.0, 1e-5);
}

TEST(Integration, GaussInvalidOrderThrows) {
    EXPECT_THROW(integrate_gauss(f_sin, 0.0, 1.0, 0), std::invalid_argument);
    EXPECT_THROW(integrate_gauss(f_sin, 0.0, 1.0, 6), std::invalid_argument);
}

// ─────────────────────────────────────────────────────────────────────────────
// integrate_adaptive
// ─────────────────────────────────────────────────────────────────────────────

TEST(Integration, AdaptiveSin) {
    EXPECT_NEAR(integrate_adaptive(f_sin, 0.0, M_PI, 1e-9), 2.0, 1e-8);
}

TEST(Integration, AdaptiveExp) {
    EXPECT_NEAR(integrate_adaptive(f_exp, 0.0, 1.0, 1e-9), std::exp(1.0) - 1.0, 1e-8);
}

TEST(Integration, AdaptiveSharpeningFunction) {
    // ∫₀¹ 1/(1+25x²) dx = arctan(5)/5 ≈ 0.27416...
    auto f = [](double x){ return 1.0 / (1.0 + 25.0 * x * x); };
    double exact = std::atan(5.0) / 5.0;
    EXPECT_NEAR(integrate_adaptive(f, 0.0, 1.0, 1e-8), exact, 1e-7);
}

TEST(Integration, AdaptiveLog) {
    double exact = 2.0 * std::log(2.0) - 1.0;
    EXPECT_NEAR(integrate_adaptive(f_log, 1.0, 2.0, 1e-9), exact, 1e-8);
}

// ─────────────────────────────────────────────────────────────────────────────
// integrate2d
// ─────────────────────────────────────────────────────────────────────────────

TEST(Integration, TwoDConstantBounds) {
    // ∫₀¹ ∫₀¹ (x+y) dy dx = 1.0
    auto f  = [](double x, double y){ return x + y; };
    auto ay = [](double){ return 0.0; };
    auto by = [](double){ return 1.0; };
    EXPECT_NEAR(integrate2d(f, 0.0, 1.0, ay, by, 200, 200), 1.0, 1e-3);
}

TEST(Integration, TwoDUnitCircleArea) {
    // Area = ∫₋₁¹ ∫_{-√(1-x²)}^{√(1-x²)} 1 dy dx = π
    auto f  = [](double, double){ return 1.0; };
    auto ay = [](double x){ return -std::sqrt(1.0 - x*x); };
    auto by = [](double x){ return  std::sqrt(1.0 - x*x); };
    EXPECT_NEAR(integrate2d(f, -1.0, 1.0, ay, by, 500, 500), M_PI, 1e-2);
}

TEST(Integration, TwoDTriangularDomain) {
    // ∫₀¹ ∫₀ˣ y dy dx = ∫₀¹ x²/2 dx = 1/6
    auto f  = [](double, double y){ return y; };
    auto ay = [](double){ return 0.0; };
    auto by = [](double x){ return x; };
    EXPECT_NEAR(integrate2d(f, 0.0, 1.0, ay, by, 500, 500), 1.0 / 6.0, 1e-3);
}
