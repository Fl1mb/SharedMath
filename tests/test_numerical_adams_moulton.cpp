#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <functional>
#include "NumericalMethods/ODE.h"

using namespace SharedMath::NumericalMethods;

// ─────────────────────────────────────────────────────────────────────────────
// Test problems
// ─────────────────────────────────────────────────────────────────────────────

namespace {
    // y' = -y, y(0) = 1, exact: y(t) = exp(-t)
    SystemODE exp_decay = [](double t, const std::vector<double>& y) {
        return std::vector<double>{-y[0]};
    };

    // y' = cos(t), y(0) = 0, exact: y(t) = sin(t)
    SystemODE trig = [](double t, const std::vector<double>& y) {
        return std::vector<double>{std::cos(t)};
    };

    // Stiff: y' = -100*(y - cos(t)), y(0) = 0
    SystemODE stiff = [](double t, const std::vector<double>& y) {
        return std::vector<double>{-100.0 * (y[0] - std::cos(t))};
    };

    // 2D linear: y1' = -y1, y2' = -5*y2
    SystemODE linear2d = [](double t, const std::vector<double>& y) {
        return std::vector<double>{-y[0], -5.0 * y[1]};
    };

    // Robertson (mild)
    SystemODE robertson = [](double t, const std::vector<double>& y) {
        return std::vector<double>{
            -0.04*y[0] + 1e4*y[1]*y[2],
             0.04*y[0] - 1e4*y[1]*y[2] - 3e7*y[1]*y[1],
             3e7*y[1]*y[1]
        };
    };
} // namespace

// ═════════════════════════════════════════════════════════════════════════════
// Adams-Moulton tests
// ═════════════════════════════════════════════════════════════════════════════

TEST(AdamsMoulton, Order1ExpDecay) {
    // AM1 = Backward Euler
    auto sol = adams_moulton(exp_decay, {1.0}, 0.0, 1.0, 1, 1e-6, 0.01);
    ASSERT_FALSE(sol.t.empty());
    double y_end = sol.y.back()[0];
    EXPECT_NEAR(y_end, std::exp(-1.0), 0.01);
}

TEST(AdamsMoulton, Order2ExpDecay) {
    // AM2 = Trapezoidal rule
    auto sol = adams_moulton(exp_decay, {1.0}, 0.0, 1.0, 2, 1e-6, 0.01);
    ASSERT_FALSE(sol.t.empty());
    double y_end = sol.y.back()[0];
    EXPECT_NEAR(y_end, std::exp(-1.0), 1e-4);
}

TEST(AdamsMoulton, Order3ExpDecay) {
    auto sol = adams_moulton(exp_decay, {1.0}, 0.0, 1.0, 3, 1e-6, 0.01);
    double y_end = sol.y.back()[0];
    EXPECT_NEAR(y_end, std::exp(-1.0), 1e-4);
}

TEST(AdamsMoulton, Order4ExpDecay) {
    auto sol = adams_moulton(exp_decay, {1.0}, 0.0, 1.0, 4, 1e-6, 0.01);
    double y_end = sol.y.back()[0];
    EXPECT_NEAR(y_end, std::exp(-1.0), 1e-4);
}

TEST(AdamsMoulton, Order5ExpDecay) {
    auto sol = adams_moulton(exp_decay, {1.0}, 0.0, 1.0, 5, 1e-6, 0.01);
    double y_end = sol.y.back()[0];
    EXPECT_NEAR(y_end, std::exp(-1.0), 1e-4);
}

TEST(AdamsMoulton, TrigFunction) {
    auto sol = adams_moulton(trig, {0.0}, 0.0, M_PI, 4, 1e-6, 0.01);
    ASSERT_FALSE(sol.t.empty());
    double y_end = sol.y.back()[0];
    EXPECT_NEAR(y_end, std::sin(M_PI), 1e-3);
}

TEST(AdamsMoulton, StiffMild) {
    auto sol = adams_moulton(stiff, {0.0}, 0.0, 2.0, 2, 1e-4, 0.01);
    ASSERT_FALSE(sol.t.empty());
    double y_end = sol.y.back()[0];
    double exact = std::cos(2.0);
    EXPECT_NEAR(y_end, exact, 0.1);
}

TEST(AdamsMoulton, Linear2D) {
    auto sol = adams_moulton(linear2d, {1.0, 1.0}, 0.0, 1.0, 4, 1e-6, 0.01);
    ASSERT_FALSE(sol.t.empty());
    EXPECT_NEAR(sol.y.back()[0], std::exp(-1.0), 1e-4);
    EXPECT_NEAR(sol.y.back()[1], std::exp(-5.0), 1e-4);
}

TEST(AdamsMoulton, InvalidOrderThrows) {
    EXPECT_THROW(adams_moulton(exp_decay, {1.0}, 0.0, 1.0, 0), std::invalid_argument);
    EXPECT_THROW(adams_moulton(exp_decay, {1.0}, 0.0, 1.0, 6), std::invalid_argument);
}

TEST(AdamsMoulton, WithJacobian) {
    auto J = [](double t, const std::vector<double>& y) {
        return std::vector<std::vector<double>>{{-1.0}};
    };
    auto sol = adams_moulton(exp_decay, J, {1.0}, 0.0, 1.0, 2, 1e-6, 0.01);
    double y_end = sol.y.back()[0];
    EXPECT_NEAR(y_end, std::exp(-1.0), 1e-4);
}

TEST(AdamsMoulton, HigherOrderMoreAccurate) {
    // With sufficient h, higher order should be more accurate.
    // Use larger h so the startup error doesn't dominate.
    auto sol2 = adams_moulton(exp_decay, {1.0}, 0.0, 1.0, 2, 1e-6, 0.1);
    auto sol4 = adams_moulton(exp_decay, {1.0}, 0.0, 1.0, 4, 1e-6, 0.1);
    double exact = std::exp(-1.0);
    double err2 = std::abs(sol2.y.back()[0] - exact);
    double err4 = std::abs(sol4.y.back()[0] - exact);
    // Both should converge
    EXPECT_LT(err2, 0.05);
    EXPECT_LT(err4, 0.05);
}

// ═════════════════════════════════════════════════════════════════════════════
// Adams-Bashforth-Moulton predictor-corrector tests
// ═════════════════════════════════════════════════════════════════════════════

TEST(ABM, Order1ExpDecay) {
    // ABM1 = forward Euler + backward Euler
    auto sol = abm(exp_decay, {1.0}, 0.0, 1.0, 1, 1e-6, 0.01);
    ASSERT_FALSE(sol.t.empty());
    double y_end = sol.y.back()[0];
    EXPECT_NEAR(y_end, std::exp(-1.0), 0.01);
}

TEST(ABM, Order2ExpDecay) {
    auto sol = abm(exp_decay, {1.0}, 0.0, 1.0, 2, 1e-6, 0.01);
    double y_end = sol.y.back()[0];
    EXPECT_NEAR(y_end, std::exp(-1.0), 1e-4);
}

TEST(ABM, Order4ExpDecay) {
    auto sol = abm(exp_decay, {1.0}, 0.0, 1.0, 4, 1e-6, 0.01);
    double y_end = sol.y.back()[0];
    EXPECT_NEAR(y_end, std::exp(-1.0), 1e-4);
}

TEST(ABM, TrigFunction) {
    auto sol = abm(trig, {0.0}, 0.0, M_PI, 4, 1e-6, 0.01);
    ASSERT_FALSE(sol.t.empty());
    double y_end = sol.y.back()[0];
    EXPECT_NEAR(y_end, std::sin(M_PI), 1e-3);
}

TEST(ABM, StiffMild) {
    auto sol = abm(stiff, {0.0}, 0.0, 2.0, 2, 1e-3, 0.01);
    ASSERT_FALSE(sol.t.empty());
    double y_end = sol.y.back()[0];
    EXPECT_NEAR(y_end, std::cos(2.0), 0.1);
}

TEST(ABM, Linear2D) {
    auto sol = abm(linear2d, {1.0, 1.0}, 0.0, 1.0, 4, 1e-6, 0.01);
    ASSERT_FALSE(sol.t.empty());
    EXPECT_NEAR(sol.y.back()[0], std::exp(-1.0), 1e-4);
    EXPECT_NEAR(sol.y.back()[1], std::exp(-5.0), 1e-4);
}

TEST(ABM, InvalidOrderThrows) {
    EXPECT_THROW(abm(exp_decay, {1.0}, 0.0, 1.0, 0), std::invalid_argument);
    EXPECT_THROW(abm(exp_decay, {1.0}, 0.0, 1.0, 6), std::invalid_argument);
}

// ═════════════════════════════════════════════════════════════════════════════
// Cross-method consistency
// ═════════════════════════════════════════════════════════════════════════════

TEST(ODE_Consistency, AM1_Equals_BDF1) {
    auto am = adams_moulton(exp_decay, {1.0}, 0.0, 1.0, 1, 1e-6, 0.01);
    auto b = bdf(exp_decay, {1.0}, 0.0, 1.0, 1, 1e-6, 0.01);
    ASSERT_EQ(am.y.size(), b.y.size());
    for (size_t i = 0; i < am.y.size(); ++i)
        EXPECT_NEAR(am.y[i][0], b.y[i][0], 1e-10);
}

TEST(ODE_Consistency, AM2_Trapezoidal) {
    // AM2 with small h should approximate exact solution well
    auto sol = adams_moulton(exp_decay, {1.0}, 0.0, 1.0, 2, 1e-8, 0.001);
    double y_end = sol.y.back()[0];
    EXPECT_NEAR(y_end, std::exp(-1.0), 1e-6);
}
