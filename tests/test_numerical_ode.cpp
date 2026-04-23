#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "NumericalMethods/ODE.h"

using namespace SharedMath::NumericalMethods;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

namespace {
    // Compare last value of solution against exact result
    void checkFinal(const ScalarODESolution& sol, double exact, double tol) {
        ASSERT_FALSE(sol.t.empty());
        EXPECT_NEAR(sol.y.back(), exact, tol);
    }

    // Check that t values are strictly increasing
    void checkMonotonicT(const std::vector<double>& t) {
        for (size_t i = 1; i < t.size(); ++i)
            EXPECT_LT(t[i - 1], t[i]);
    }
} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Euler — scalar
// ─────────────────────────────────────────────────────────────────────────────

TEST(ODEEuler, ExponentialGrowth) {
    // dy/dt = y, y(0)=1 → y(1) = e
    // Euler has O(h) error, h=1e-4 → ~1e-4 accuracy
    auto sol = euler([](double, double y){ return y; }, 1.0, 0.0, 1.0, 1e-4);
    checkFinal(sol, std::exp(1.0), 1e-3);
    checkMonotonicT(sol.t);
}

TEST(ODEEuler, ExponentialDecay) {
    // dy/dt = -y, y(0)=1 → y(2) = e^{-2}
    auto sol = euler([](double, double y){ return -y; }, 1.0, 0.0, 2.0, 1e-4);
    checkFinal(sol, std::exp(-2.0), 1e-4);
}

TEST(ODEEuler, ConstantRHS) {
    // dy/dt = 3, y(0)=0 → y(2) = 6
    auto sol = euler([](double, double){ return 3.0; }, 0.0, 0.0, 2.0, 1e-3);
    checkFinal(sol, 6.0, 1e-10);
}

TEST(ODEEuler, InitialConditionPreserved) {
    auto sol = euler([](double, double y){ return y; }, 5.0, 0.0, 1.0, 0.1);
    EXPECT_NEAR(sol.y.front(), 5.0, 1e-14);
    EXPECT_NEAR(sol.t.front(), 0.0, 1e-14);
}

// ─────────────────────────────────────────────────────────────────────────────
// RK4 — scalar
// ─────────────────────────────────────────────────────────────────────────────

TEST(ODERK4, ExponentialGrowth) {
    // RK4 O(h^4), h=1e-2 → very tight
    auto sol = rk4([](double, double y){ return y; }, 1.0, 0.0, 1.0, 1e-2);
    checkFinal(sol, std::exp(1.0), 1e-7);
    checkMonotonicT(sol.t);
}

TEST(ODERK4, SinusoidalSolution) {
    // dy/dt = cos(t), y(0)=0 → y(pi) = sin(pi) = 0
    auto sol = rk4([](double t, double){ return std::cos(t); }, 0.0, 0.0, M_PI, 1e-3);
    checkFinal(sol, 0.0, 1e-6);
}

TEST(ODERK4, LogisticEquation) {
    // dy/dt = y*(1-y), y(0)=0.5 → y(t) = 1/(1+e^{-t})
    // At t=3: y = 1/(1+e^{-3})
    auto sol = rk4([](double, double y){ return y * (1.0 - y); }, 0.5, 0.0, 3.0, 1e-3);
    double exact = 1.0 / (1.0 + std::exp(-3.0));
    checkFinal(sol, exact, 1e-7);
}

TEST(ODERK4, BetterThanEuler) {
    // RK4 should be far more accurate than Euler for the same step
    double h = 0.1;
    auto euler_sol = euler([](double, double y){ return y; }, 1.0, 0.0, 1.0, h);
    auto rk4_sol   = rk4([](double, double y){ return y; }, 1.0, 0.0, 1.0, h);
    double exact = std::exp(1.0);
    double err_euler = std::abs(euler_sol.y.back() - exact);
    double err_rk4   = std::abs(rk4_sol.y.back()   - exact);
    EXPECT_LT(err_rk4, err_euler);
}

// ─────────────────────────────────────────────────────────────────────────────
// RK45 — scalar (adaptive)
// ─────────────────────────────────────────────────────────────────────────────

TEST(ODERK45, ExponentialGrowth) {
    auto sol = rk45([](double, double y){ return y; }, 1.0, 0.0, 1.0, 1e-8);
    checkFinal(sol, std::exp(1.0), 1e-6);
    checkMonotonicT(sol.t);
}

TEST(ODERK45, ExponentialDecay) {
    auto sol = rk45([](double, double y){ return -y; }, 1.0, 0.0, 3.0, 1e-8);
    checkFinal(sol, std::exp(-3.0), 1e-6);
}

TEST(ODERK45, StiffVariation) {
    // dy/dt = -10*y, y(0)=1 → y(1) = e^{-10}
    auto sol = rk45([](double, double y){ return -10.0 * y; }, 1.0, 0.0, 1.0, 1e-8, 1e-4);
    checkFinal(sol, std::exp(-10.0), 1e-5);
}

TEST(ODERK45, AdaptiveStepCount) {
    // Smooth problem → adaptive should need fewer steps than a fixed fine-grid RK4
    auto sol = rk45([](double, double y){ return y; }, 1.0, 0.0, 1.0, 1e-6);
    // Expect well under 10 000 steps for a simple exponential
    EXPECT_LT(sol.t.size(), 10000u);
}

// ─────────────────────────────────────────────────────────────────────────────
// Euler — system
// ─────────────────────────────────────────────────────────────────────────────

TEST(ODESystem_Euler, HarmonicOscillator) {
    // y1' = y2,  y2' = -y1,  y(0) = [1, 0]
    // Exact: y1(t) = cos(t), y2(t) = -sin(t)
    SystemODE f = [](double, const std::vector<double>& y) -> std::vector<double> {
        return { y[1], -y[0] };
    };
    auto sol = euler_system(f, {1.0, 0.0}, 0.0, M_PI, 1e-4);
    EXPECT_NEAR(sol.y.back()[0], std::cos(M_PI),   1e-3);
    EXPECT_NEAR(sol.y.back()[1], -std::sin(M_PI),  1e-3);
}

// ─────────────────────────────────────────────────────────────────────────────
// RK4 — system
// ─────────────────────────────────────────────────────────────────────────────

TEST(ODESystem_RK4, HarmonicOscillator) {
    // Same SHO — much tighter accuracy
    SystemODE f = [](double, const std::vector<double>& y) -> std::vector<double> {
        return { y[1], -y[0] };
    };
    auto sol = rk4_system(f, {1.0, 0.0}, 0.0, M_PI, 1e-3);
    EXPECT_NEAR(sol.y.back()[0],  std::cos(M_PI),  1e-6);
    EXPECT_NEAR(sol.y.back()[1], -std::sin(M_PI),  1e-6);
}

TEST(ODESystem_RK4, ExponentialDecoupled) {
    // y1' = -y1,  y2' = -2*y2
    // y1(t) = e^{-t},  y2(t) = e^{-2t}
    SystemODE f = [](double, const std::vector<double>& y) -> std::vector<double> {
        return { -y[0], -2.0 * y[1] };
    };
    auto sol = rk4_system(f, {1.0, 1.0}, 0.0, 1.0, 1e-3);
    EXPECT_NEAR(sol.y.back()[0], std::exp(-1.0),   1e-7);
    EXPECT_NEAR(sol.y.back()[1], std::exp(-2.0),   1e-7);
}

TEST(ODESystem_RK4, InitialConditionPreserved) {
    SystemODE f = [](double, const std::vector<double>& y) -> std::vector<double> {
        return { y[1], -y[0] };
    };
    auto sol = rk4_system(f, {3.0, 7.0}, 0.0, 1.0, 0.1);
    EXPECT_NEAR(sol.y.front()[0], 3.0, 1e-14);
    EXPECT_NEAR(sol.y.front()[1], 7.0, 1e-14);
}

// ─────────────────────────────────────────────────────────────────────────────
// RK45 — system (adaptive)
// ─────────────────────────────────────────────────────────────────────────────

TEST(ODESystem_RK45, HarmonicOscillator) {
    SystemODE f = [](double, const std::vector<double>& y) -> std::vector<double> {
        return { y[1], -y[0] };
    };
    auto sol = rk45_system(f, {1.0, 0.0}, 0.0, 2.0 * M_PI, 1e-8);
    // After one full period: y1 ≈ 1, y2 ≈ 0
    EXPECT_NEAR(sol.y.back()[0], 1.0, 1e-5);
    EXPECT_NEAR(sol.y.back()[1], 0.0, 1e-5);
}

TEST(ODESystem_RK45, ExponentialDecoupled) {
    SystemODE f = [](double, const std::vector<double>& y) -> std::vector<double> {
        return { -y[0], -3.0 * y[1] };
    };
    auto sol = rk45_system(f, {1.0, 1.0}, 0.0, 1.0, 1e-8);
    EXPECT_NEAR(sol.y.back()[0], std::exp(-1.0),   1e-6);
    EXPECT_NEAR(sol.y.back()[1], std::exp(-3.0),   1e-6);
}
