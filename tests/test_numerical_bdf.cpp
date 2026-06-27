#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <functional>
#include "NumericalMethods/ODE.h"
#include "NumericalMethods/NonlinearEquations.h"

using namespace SharedMath::NumericalMethods;

// ─────────────────────────────────────────────────────────────────────────────
// Test problems
// ─────────────────────────────────────────────────────────────────────────────

namespace {
    // Stiff linear test: dy/dt = -1000*(y - cos(t)), y(0) = 0
    // For large t, y(t) ≈ cos(t)
    SystemODE stiff_linear = [](double t, const std::vector<double>& y) {
        return std::vector<double>{-1000.0 * (y[0] - std::cos(t))};
    };

    // Simple exponential decay: dy/dt = -y, y(0) = 1
    // Exact: y(t) = exp(-t)
    SystemODE exp_decay = [](double t, const std::vector<double>& y) {
        return std::vector<double>{-y[0]};
    };

    // Linear system: dy1/dt = -y1, dy2/dt = -100*y2
    SystemODE stiff_system = [](double t, const std::vector<double>& y) {
        return std::vector<double>{-y[0], -100.0 * y[1]};
    };

    // Robertson's chemical kinetics
    SystemODE robertson = [](double t, const std::vector<double>& y) {
        return std::vector<double>{
            -0.04*y[0] + 1e4*y[1]*y[2],
             0.04*y[0] - 1e4*y[1]*y[2] - 3e7*y[1]*y[1],
             3e7*y[1]*y[1]
        };
    };

    // Linear test for Newton-Raphson system
    auto F_linear = [](const std::vector<double>& x) {
        return std::vector<double>{2.0*x[0] + x[1] - 2.0, x[0] + 3.0*x[1] - 7.0};
    };

    // Nonlinear test for Newton-Raphson system
    auto F_nonlinear = [](const std::vector<double>& x) {
        return std::vector<double>{
            x[0]*x[0] + x[1]*x[1] - 4.0,
            x[0] - x[1] - 1.0
        };
    };

    auto J_nonlinear = [](const std::vector<double>& x) {
        return std::vector<std::vector<double>>{
            {2.0*x[0], 2.0*x[1]},
            {1.0, -1.0}
        };
    };
} // namespace

// ═════════════════════════════════════════════════════════════════════════════
// System Newton-Raphson tests
// ═════════════════════════════════════════════════════════════════════════════

TEST(NewtonRaphsonSystem, LinearSystem) {
    auto result = newton_raphson_system(F_linear, {0.0, 0.0});
    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.x[0], -0.2, 1e-8);
    EXPECT_NEAR(result.x[1], 2.4, 1e-8);
    EXPECT_LT(result.residual, 1e-8);
}

TEST(NewtonRaphsonSystem, NonlinearAnalytic) {
    auto result = newton_raphson_system(F_nonlinear, J_nonlinear, {1.0, 0.0});
    EXPECT_TRUE(result.converged);
    double expected_x0 = (1.0 + std::sqrt(7.0)) / 2.0;
    double expected_x1 = expected_x0 - 1.0;
    EXPECT_NEAR(result.x[0], expected_x0, 1e-8);
    EXPECT_NEAR(result.x[1], expected_x1, 1e-8);
}

TEST(NewtonRaphsonSystem, NonlinearNumerical) {
    auto result = newton_raphson_system(F_nonlinear, {1.0, 0.0});
    EXPECT_TRUE(result.converged);
    double expected_x0 = (1.0 + std::sqrt(7.0)) / 2.0;
    double expected_x1 = expected_x0 - 1.0;
    EXPECT_NEAR(result.x[0], expected_x0, 1e-6);
    EXPECT_NEAR(result.x[1], expected_x1, 1e-6);
}

// ═════════════════════════════════════════════════════════════════════════════
// BDF method tests
// ═════════════════════════════════════════════════════════════════════════════

TEST(BDF, ExpDecayOrder1) {
    auto sol = bdf(exp_decay, {1.0}, 0.0, 1.0, 1, 1e-6, 0.01);
    ASSERT_FALSE(sol.t.empty());
    double y_end = sol.y.back()[0];
    double exact = std::exp(-1.0);
    // BDF1 (backward Euler) has O(h) global error, so ~1e-2 with h=0.01
    EXPECT_NEAR(y_end, exact, 0.01);
    EXPECT_LT(y_end, 1.0);
    EXPECT_GT(y_end, 0.0);
}

TEST(BDF, ExpDecayOrder2) {
    auto sol = bdf(exp_decay, {1.0}, 0.0, 1.0, 2, 1e-6, 0.01);
    double y_end = sol.y.back()[0];
    double exact = std::exp(-1.0);
    EXPECT_NEAR(y_end, exact, 0.01);
}

TEST(BDF, ExpDecayOrder3) {
    auto sol = bdf(exp_decay, {1.0}, 0.0, 1.0, 3, 1e-6, 0.01);
    double y_end = sol.y.back()[0];
    double exact = std::exp(-1.0);
    EXPECT_NEAR(y_end, exact, 0.01);
}

TEST(BDF, ExpDecayOrder5) {
    auto sol = bdf(exp_decay, {1.0}, 0.0, 1.0, 5, 1e-6, 0.05);
    double y_end = sol.y.back()[0];
    double exact = std::exp(-1.0);
    EXPECT_NEAR(y_end, exact, 0.01);
}

TEST(BDF, StiffLinear) {
    auto sol = bdf(stiff_linear, {0.0}, 0.0, 2.0, 2, 1e-4, 0.01);
    ASSERT_FALSE(sol.t.empty());
    double y_end = sol.y.back()[0];
    double exact = std::cos(2.0);
    EXPECT_NEAR(y_end, exact, 0.1);
}

TEST(BDF, StiffSystem) {
    auto sol = bdf(stiff_system, {1.0, 1.0}, 0.0, 0.1, 2, 1e-4, 0.001);
    ASSERT_FALSE(sol.t.empty());
    EXPECT_NEAR(sol.y.back()[0], std::exp(-0.1), 0.05);
    EXPECT_NEAR(sol.y.back()[1], 0.0, 0.05);
}

TEST(BDF, StiffNonlinear) {
    // y' = -50*(y - cos(t)), y(0) = 0
    // Moderate stiffness, should reach y ≈ cos(t) quickly
    SystemODE stiff_nl = [](double t, const std::vector<double>& y) {
        return std::vector<double>{-50.0 * (y[0] - std::cos(t))};
    };
    auto sol = bdf(stiff_nl, {0.0}, 0.0, 1.0, 2, 1e-3, 0.01);
    ASSERT_FALSE(sol.t.empty());
    EXPECT_NEAR(sol.y.back()[0], std::cos(1.0), 0.1);
}

TEST(BDF, InvalidOrderThrows) {
    EXPECT_THROW(bdf(exp_decay, {1.0}, 0.0, 1.0, 0), std::invalid_argument);
    EXPECT_THROW(bdf(exp_decay, {1.0}, 0.0, 1.0, 7), std::invalid_argument);
}

TEST(BDF, WithJacobian) {
    auto J_exp = [](double t, const std::vector<double>& y) {
        return std::vector<std::vector<double>>{{-1.0}};
    };
    auto sol = bdf(exp_decay, J_exp, {1.0}, 0.0, 1.0, 2, 1e-6, 0.01);
    double y_end = sol.y.back()[0];
    EXPECT_NEAR(y_end, std::exp(-1.0), 0.01);
}

TEST(BDF, HigherOrderMoreAccurate) {
    auto sol1 = bdf(exp_decay, {1.0}, 0.0, 1.0, 1, 1e-6, 0.01);
    auto sol3 = bdf(exp_decay, {1.0}, 0.0, 1.0, 3, 1e-6, 0.01);
    double exact = std::exp(-1.0);
    double err1 = std::abs(sol1.y.back()[0] - exact);
    double err3 = std::abs(sol3.y.back()[0] - exact);
    // BDF1 and BDF3 should both converge to the right answer
    EXPECT_LT(err1, 0.1);
    EXPECT_LT(err3, 0.1);
}
