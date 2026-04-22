#include <gtest/gtest.h>
#include "functions/functions.h"

#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace SharedMath::Functions;

static constexpr double kEps = 1e-9;   // tight tolerance (exact formulas)
static constexpr double kMed = 1e-6;   // medium tolerance (polynomial approx)

// ═════════════════════════════════════════════════════════════════════════════
// GAMMA FAMILY
// ═════════════════════════════════════════════════════════════════════════════

TEST(Gamma, KnownValues) {
    // std::tgamma is the portable Gamma function; our library adds digamma etc.
    EXPECT_NEAR(std::tgamma(1.0), 1.0, kEps);
    EXPECT_NEAR(std::tgamma(2.0), 1.0, kEps);
    EXPECT_NEAR(std::tgamma(4.0), 6.0, kEps);
    EXPECT_NEAR(std::tgamma(0.5), std::sqrt(M_PI), kEps);
}

TEST(Gamma, PositiveIntegersAreFactorials) {
    for (int n = 1; n <= 10; ++n) {
        double fact = 1.0;
        for (int k = 1; k < n; ++k) fact *= k;
        EXPECT_NEAR(std::tgamma(static_cast<double>(n)), fact, kEps * fact + kEps);
    }
}

TEST(Digamma, KnownValues) {
    // ψ(1) = −γ_EM ≈ −0.5772156649
    EXPECT_NEAR(digamma(1.0), -0.5772156649015329, 1e-10);
    // Recurrence: ψ(x+1) = ψ(x) + 1/x
    for (double x : {1.0, 2.0, 3.5, 7.0})
        EXPECT_NEAR(digamma(x + 1.0), digamma(x) + 1.0 / x, 1e-12) << "x=" << x;
}

TEST(Digamma, ReflectionFormula) {
    for (double x : {0.3, 0.1, 0.7}) {
        double lhs = digamma(1.0 - x);
        double rhs = digamma(x) + M_PI / std::tan(M_PI * x);
        EXPECT_NEAR(lhs, rhs, 1e-10) << "x=" << x;
    }
}

TEST(Digamma, InvalidInput) {
    EXPECT_THROW(digamma(0.0),  std::domain_error);
    EXPECT_THROW(digamma(-1.0), std::domain_error);
}

TEST(Trigamma, KnownValue) {
    // ψ'(1) = π²/6
    EXPECT_NEAR(trigamma(1.0), M_PI * M_PI / 6.0, 1e-10);
}

TEST(Trigamma, Recurrence) {
    for (double x : {1.0, 2.0, 5.0})
        EXPECT_NEAR(trigamma(x + 1.0), trigamma(x) - 1.0 / (x * x), 1e-10) << "x=" << x;
}

TEST(Beta, SymmetryAndKnownValues) {
    EXPECT_NEAR(beta(2.0, 3.0), beta(3.0, 2.0), kEps);
    EXPECT_NEAR(beta(1.0, 1.0), 1.0, kEps);
    EXPECT_NEAR(beta(3.0, 4.0), 1.0 / 60.0, kEps);
}

TEST(Betainc, BoundaryAndMonotone) {
    EXPECT_NEAR(betainc(0.0, 2.0, 3.0), 0.0, kEps);
    EXPECT_NEAR(betainc(1.0, 2.0, 3.0), 1.0, kEps);
    EXPECT_NEAR(betainc(0.5, 1.0, 1.0), 0.5, kEps);
    // Symmetry: I_x(a,b) + I_{1-x}(b,a) = 1
    for (double x : {0.2, 0.5, 0.8})
        EXPECT_NEAR(betainc(x, 3.0, 5.0) + betainc(1.0 - x, 5.0, 3.0), 1.0, 1e-10) << "x=" << x;
}

TEST(GammaInc, BoundaryAndMonotone) {
    EXPECT_NEAR(gammainc(1.0, 0.0), 0.0, kEps);
    EXPECT_NEAR(gammainc(2.0, 20.0), 1.0, 1e-8);
    double prev = 0.0;
    for (double x : {0.5, 1.0, 2.0, 4.0, 8.0}) {
        double cur = gammainc(3.0, x);
        EXPECT_GT(cur, prev) << "x=" << x;
        prev = cur;
    }
}


// ═════════════════════════════════════════════════════════════════════════════
// ERROR FUNCTIONS
// ═════════════════════════════════════════════════════════════════════════════

TEST(ErfFuncs, KnownValues) {
    EXPECT_NEAR(std::erf(0.0),  0.0, kEps);
    EXPECT_NEAR(std::erfc(0.0), 1.0, kEps);
    EXPECT_NEAR(std::erf(1.0) + std::erfc(1.0), 1.0, kEps);
}

TEST(ErfInv, RoundTrip) {
    for (double y : {-0.9, -0.5, 0.0, 0.3, 0.7, 0.95})
        EXPECT_NEAR(std::erf(erfinv(y)), y, 1e-10) << "y=" << y;
}

TEST(ErfInv, Symmetry) {
    EXPECT_NEAR(erfinv(-0.6), -erfinv(0.6), 1e-10);
}


// ═════════════════════════════════════════════════════════════════════════════
// BESSEL FUNCTIONS
// ═════════════════════════════════════════════════════════════════════════════

TEST(BesselJ0, KnownAndEven) {
    EXPECT_NEAR(besselJ0(0.0),   1.0,        kEps);
    EXPECT_NEAR(besselJ0(1.0),   0.7651977,  kMed);
    EXPECT_NEAR(besselJ0(-2.0),  besselJ0(2.0), kEps);
}

TEST(BesselJ1, KnownAndOdd) {
    EXPECT_NEAR(besselJ1(0.0),  0.0,       kEps);
    EXPECT_NEAR(besselJ1(1.0),  0.4400506, kMed);
    EXPECT_NEAR(besselJ1(-1.0), -besselJ1(1.0), kEps);
}

TEST(BesselJn, RecurrenceRelation) {
    // J_{n-1}(x) + J_{n+1}(x) = (2n/x) J_n(x)
    double x = 3.0;
    for (int n = 1; n <= 5; ++n) {
        double lhs = besselJn(n-1, x) + besselJn(n+1, x);
        double rhs = (2.0 * n / x) * besselJn(n, x);
        EXPECT_NEAR(lhs, rhs, kMed) << "n=" << n;
    }
}

TEST(BesselI0K0, WronskianRelation) {
    // I0(x)·K1(x) + I1(x)·K0(x) = 1/x
    for (double x : {0.5, 1.0, 2.0, 5.0})
        EXPECT_NEAR(besselI0(x) * besselK1(x) + besselI1(x) * besselK0(x),
                    1.0 / x, kMed) << "x=" << x;
}

TEST(BesselI0, KnownValues) {
    EXPECT_NEAR(besselI0(0.0), 1.0,       kEps);
    EXPECT_NEAR(besselI0(1.0), 1.2660658, kMed);
}

TEST(BesselK0, ThrowsAtZero) {
    EXPECT_THROW(besselK0(0.0), std::domain_error);
}


// ═════════════════════════════════════════════════════════════════════════════
// ORTHOGONAL POLYNOMIALS
// ═════════════════════════════════════════════════════════════════════════════

TEST(LegendreP, LowDegree) {
    EXPECT_NEAR(legendreP(0, 0.5),  1.0,    kEps);
    EXPECT_NEAR(legendreP(1, 0.5),  0.5,    kEps);
    EXPECT_NEAR(legendreP(2, 0.5), -0.125,  kEps);
    EXPECT_NEAR(legendreP(3, 0.0),  0.0,    kEps);
}

TEST(LegendreP, Orthogonality) {
    // ∫_{-1}^{1} P_0 P_2 dx = 0  (trapezoidal rule)
    int N = 1000; double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        double x = -1.0 + (2.0*i + 1.0) / N;
        sum += legendreP(0, x) * legendreP(2, x) * (2.0 / N);
    }
    EXPECT_NEAR(sum, 0.0, 1e-6);
}

TEST(ChebyshevT, CosineRelation) {
    // T_n(cos θ) = cos(n θ)
    double theta = 0.7, x = std::cos(theta);
    for (int n : {2, 4, 6})
        EXPECT_NEAR(chebyshevT(n, x), std::cos(n * theta), kEps) << "n=" << n;
}

TEST(HermiteH, LowDegree) {
    EXPECT_NEAR(hermiteH(2, 1.0),  2.0,  kEps);   // 4x²-2 at x=1
    EXPECT_NEAR(hermiteH(2, 0.0), -2.0,  kEps);
}

TEST(LaguerreL, LowDegree) {
    EXPECT_NEAR(laguerreL(1, 1.0),  0.0, kEps);
    EXPECT_NEAR(laguerreL(2, 2.0), -1.0, kEps);
}


// ═════════════════════════════════════════════════════════════════════════════
// ELLIPTIC INTEGRALS
// ═════════════════════════════════════════════════════════════════════════════

TEST(EllipticK, SpecialValues) {
    EXPECT_NEAR(ellipticK(0.0), M_PI / 2.0, kEps);
}

TEST(EllipticE, SpecialValues) {
    EXPECT_NEAR(ellipticE(0.0), M_PI / 2.0, kEps);
    EXPECT_NEAR(ellipticE(1.0), 1.0,        kEps);
}

TEST(EllipticE, LegendreRelation) {
    double k = 0.5, kp = std::sqrt(1.0 - k*k);
    double lhs = ellipticK(k)*ellipticE(kp) + ellipticK(kp)*ellipticE(k)
               - ellipticK(k)*ellipticK(kp);
    EXPECT_NEAR(lhs, M_PI / 2.0, 1e-10);
}


// ═════════════════════════════════════════════════════════════════════════════
// OTHER FUNCTIONS
// ═════════════════════════════════════════════════════════════════════════════

TEST(Sinc, KnownValues) {
    EXPECT_NEAR(sinc(0.0), 1.0,       kEps);
    EXPECT_NEAR(sinc(1.0), 0.0,       kEps);
    EXPECT_NEAR(sinc(0.5), 2.0/M_PI,  kEps);
}

TEST(LambertW, Identity) {
    for (double x : {0.0, 0.5, 1.0, 2.0, 5.0}) {
        double w = lambertW(x * std::exp(x));
        EXPECT_NEAR(w, x, 1e-12) << "x=" << x;
    }
}

TEST(RiemannZeta, KnownValues) {
    EXPECT_NEAR(riemannZeta(2), M_PI*M_PI/6.0,         1e-6);
    EXPECT_NEAR(riemannZeta(4), std::pow(M_PI,4)/90.0, 1e-6);
}


// ═════════════════════════════════════════════════════════════════════════════
// ML ACTIVATIONS
// ═════════════════════════════════════════════════════════════════════════════

TEST(Sigmoid, KnownAndSymmetry) {
    EXPECT_NEAR(sigmoid(0.0), 0.5, kEps);
    for (double x : {-3.0, -1.0, 0.5, 2.0})
        EXPECT_NEAR(sigmoid(x) + sigmoid(-x), 1.0, kEps);
}

TEST(Sigmoid, DerivativeNumerical) {
    double x = 1.5, h = 1e-6;
    EXPECT_NEAR(sigmoidPrime(x),
                (sigmoid(x+h) - sigmoid(x-h)) / (2.0*h), 1e-8);
}

TEST(ReLU, Basic) {
    EXPECT_NEAR(relu(2.0),  2.0, kEps);
    EXPECT_NEAR(relu(-3.0), 0.0, kEps);
}

TEST(GELU, DerivativeNumerical) {
    double x = 0.7, h = 1e-6;
    EXPECT_NEAR(geluPrime(x),
                (gelu(x+h) - gelu(x-h)) / (2.0*h), 1e-7);
}

TEST(Swish, DerivativeNumerical) {
    double x = 1.3, h = 1e-6;
    EXPECT_NEAR(swishPrime(x),
                (swish(x+h) - swish(x-h)) / (2.0*h), 1e-7);
}

TEST(Softmax, SumsToOneAndOrdered) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto out = softmax(x);
    EXPECT_NEAR(std::accumulate(out.begin(), out.end(), 0.0), 1.0, kEps);
    EXPECT_LT(out[0], out[1]);
    EXPECT_LT(out[1], out[2]);
}

TEST(LogSoftmax, ExpSumsToOne) {
    std::vector<double> x = {0.5, -0.5, 1.5};
    auto ls = logSoftmax(x);
    double s = 0.0;
    for (double v : ls) s += std::exp(v);
    EXPECT_NEAR(s, 1.0, kMed);
}


// ═════════════════════════════════════════════════════════════════════════════
// LOSS FUNCTIONS
// ═════════════════════════════════════════════════════════════════════════════

TEST(MSE, ZeroAndKnown) {
    std::vector<double> y    = {1.0, 2.0, 3.0};
    std::vector<double> zero = {0.0, 0.0, 0.0};
    EXPECT_NEAR(mse(y, y),    0.0,      kEps);
    EXPECT_NEAR(mse(y, zero), 14.0/3.0, kEps);
}

TEST(MAE, ZeroAndKnown) {
    std::vector<double> y    = {1.0, 2.0, 3.0};
    std::vector<double> zero = {0.0, 0.0, 0.0};
    EXPECT_NEAR(mae(y, y),    0.0, kEps);
    EXPECT_NEAR(mae(y, zero), 2.0, kEps);
}

TEST(Huber, EquivalentToHalfMSEForSmallErrors) {
    std::vector<double> p = {0.1, -0.2, 0.3};
    std::vector<double> t = {0.0,  0.0, 0.0};
    EXPECT_NEAR(huber(p, t, 10.0), mse(p, t) / 2.0, kEps);
}

TEST(HingeLoss, CorrectAndWrong) {
    std::vector<double> p1 = {2.0},  l1 = {1.0};
    std::vector<double> p2 = {-2.0}, l2 = {1.0};
    EXPECT_NEAR(hingeLoss(p1, l1), 0.0, kEps);
    EXPECT_NEAR(hingeLoss(p2, l2), 3.0, kEps);
}

TEST(R2Score, PerfectAndMean) {
    std::vector<double> y    = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> mean = {2.0, 2.0, 2.0};
    std::vector<double> ref  = {1.0, 2.0, 3.0};
    EXPECT_NEAR(r2Score(y, y),      1.0, kEps);
    EXPECT_NEAR(r2Score(mean, ref), 0.0, kEps);
}

TEST(KLDivergence, ZeroForIdentical) {
    std::vector<double> p = {0.3, 0.4, 0.3};
    std::vector<double> q = {0.2, 0.5, 0.3};
    std::vector<double> r = {0.5, 0.3, 0.2};
    EXPECT_NEAR(klDivergence(p, p), 0.0, kEps);
    EXPECT_GE(klDivergence(q, r), 0.0);
}
