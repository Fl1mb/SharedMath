#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "NumericalMethods/IntegralEquations.h"

using namespace SharedMath::NumericalMethods;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

namespace {
    // Check that every solution value is close to the given exact function
    void checkSolution(const std::pair<std::vector<double>, std::vector<double>>& result,
                       std::function<double(double)> exact, double tol)
    {
        const auto& [nodes, y] = result;
        ASSERT_EQ(nodes.size(), y.size());
        ASSERT_FALSE(nodes.empty());
        for (size_t i = 0; i < nodes.size(); ++i)
            EXPECT_NEAR(y[i], exact(nodes[i]), tol)
                << "  at x = " << nodes[i];
    }
} // namespace

// ═════════════════════════════════════════════════════════════════════════════
// Fredholm 2nd kind
// ═════════════════════════════════════════════════════════════════════════════

TEST(Fredholm2, ConstantSolution) {
    // y(x) = 1 + 0.5 * ∫₀¹ 1 * y(t) dt
    // If y = c (const): c = 1 + 0.5*c  →  c = 2
    // So exact solution: y(x) = 2 for all x ∈ [0,1]
    auto f = [](double){ return 1.0; };
    auto K = [](double, double){ return 1.0; };
    auto result = fredholm2(f, K, 0.0, 1.0, 0.5, 200);
    checkSolution(result, [](double){ return 2.0; }, 1e-5);
}

TEST(Fredholm2, ZeroLambda) {
    // λ=0 → y(x) = f(x) exactly
    auto f = [](double x){ return std::sin(x); };
    auto K = [](double, double){ return 1.0; };
    auto result = fredholm2(f, K, 0.0, M_PI, 0.0, 100);
    checkSolution(result, f, 1e-12);
}

TEST(Fredholm2, SeparableKernel) {
    // K(x,t) = x * t, f(x) = x, λ = 1 on [0,1]
    // y(x) = x + x * ∫₀¹ t * y(t) dt
    //      = x (1 + ∫₀¹ t*y(t)dt)
    // Let c = ∫₀¹ t*y(t)dt.  Then y(x) = x*(1+c).
    // c = ∫₀¹ t*(1+c)*t dt = (1+c)/3  →  c - c/3 = 1/3  →  c = 1/2
    // Exact: y(x) = 3x/2
    auto f = [](double x){ return x; };
    auto K = [](double x, double t){ return x * t; };
    auto result = fredholm2(f, K, 0.0, 1.0, 1.0, 300);
    checkSolution(result, [](double x){ return 1.5 * x; }, 1e-5);
}

TEST(Fredholm2, NodeBoundsCorrect) {
    // First and last nodes must equal a and b
    auto result = fredholm2([](double){ return 1.0; },
                             [](double, double){ return 0.0; },
                             -1.0, 1.0, 0.0, 50);
    EXPECT_NEAR(result.first.front(), -1.0, 1e-14);
    EXPECT_NEAR(result.first.back(),   1.0, 1e-14);
}

TEST(Fredholm2, SmallNThrows) {
    EXPECT_THROW(
        fredholm2([](double){ return 1.0; },
                   [](double, double){ return 1.0; },
                   0.0, 1.0, 0.5, 1),
        std::invalid_argument);
}

TEST(Fredholm2, NodeCount) {
    auto result = fredholm2([](double){ return 1.0; },
                             [](double, double){ return 0.0; },
                             0.0, 1.0, 0.0, 75);
    EXPECT_EQ(result.first.size(), 75u);
    EXPECT_EQ(result.second.size(), 75u);
}

// ═════════════════════════════════════════════════════════════════════════════
// Volterra 2nd kind
// ═════════════════════════════════════════════════════════════════════════════

TEST(Volterra2, ExponentialSolution) {
    // y(x) = 1 + ∫₀ˣ y(t) dt
    // Differentiate: y' = y, y(0) = 1  →  y(x) = eˣ
    auto f = [](double){ return 1.0; };
    auto K = [](double, double){ return 1.0; };
    auto result = volterra2(f, K, 0.0, 1.0, 1.0, 500);
    checkSolution(result, [](double x){ return std::exp(x); }, 1e-4);
}

TEST(Volterra2, ZeroLambda) {
    // λ=0 → y(x) = f(x)
    auto f = [](double x){ return std::cos(x); };
    auto K = [](double, double){ return 1.0; };
    auto result = volterra2(f, K, 0.0, M_PI, 0.0, 100);
    checkSolution(result, f, 1e-12);
}

TEST(Volterra2, LinearSolution) {
    // y(x) = 1 + ∫₀ˣ (x - t) * y(t) dt
    // K(x,t) = x - t, f(x) = 1, λ = 1
    // Differentiating twice gives y'' = y, y(0)=1, y'(0)=0 → y = cosh(x)
    auto f = [](double){ return 1.0; };
    auto K = [](double x, double t){ return x - t; };
    auto result = volterra2(f, K, 0.0, 1.5, 1.0, 500);
    checkSolution(result, [](double x){ return std::cosh(x); }, 1e-4);
}

TEST(Volterra2, InitialValueExact) {
    // y(0) = f(0) always (integral from 0 to 0 is 0)
    auto result = volterra2([](double x){ return std::sin(x) + 1.0; },
                             [](double, double){ return 1.0; },
                             0.0, 1.0, 0.5, 100);
    EXPECT_NEAR(result.second.front(), 1.0, 1e-14);  // sin(0)+1 = 1
}

TEST(Volterra2, SmallNThrows) {
    EXPECT_THROW(
        volterra2([](double){ return 1.0; },
                   [](double, double){ return 1.0; },
                   0.0, 1.0, 1.0, 1),
        std::invalid_argument);
}

TEST(Volterra2, NodeCount) {
    auto result = volterra2([](double){ return 1.0; },
                             [](double, double){ return 0.0; },
                             0.0, 2.0, 0.0, 80);
    EXPECT_EQ(result.first.size(), 80u);
    EXPECT_EQ(result.second.size(), 80u);
}
