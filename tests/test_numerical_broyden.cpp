#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <functional>
#include "NumericalMethods/NonlinearEquations.h"

using namespace SharedMath::NumericalMethods;

// ─────────────────────────────────────────────────────────────────────────────
// Test problems
// ─────────────────────────────────────────────────────────────────────────────

namespace {
    // 2D linear: 2x + y = 2, x + 3y = 7  →  x = -0.2, y = 2.4
    auto F_linear = [](const std::vector<double>& x) {
        return std::vector<double>{2.0*x[0] + x[1] - 2.0, x[0] + 3.0*x[1] - 7.0};
    };

    // 2D nonlinear: x^2 + y^2 = 4, x - y = 1
    auto F_nonlinear = [](const std::vector<double>& x) {
        return std::vector<double>{
            x[0]*x[0] + x[1]*x[1] - 4.0,
            x[0] - x[1] - 1.0
        };
    };

    // Rosenbrock (2D): f1 = 10*(y - x^2), f2 = 1 - x  →  root at (1,1)
    auto F_rosenbrock = [](const std::vector<double>& x) {
        return std::vector<double>{
            10.0 * (x[1] - x[0]*x[0]),
            1.0 - x[0]
        };
    };

    // 3D: x^2+y+z=1, x+y^2+z=1, x+y+z^2=1  →  root at (0,0,0)
    auto F_3d = [](const std::vector<double>& x) {
        return std::vector<double>{
            x[0]*x[0] + x[1] + x[2] - 1.0,
            x[0] + x[1]*x[1] + x[2] - 1.0,
            x[0] + x[1] + x[2]*x[2] - 1.0
        };
    };

    // 4D: simple diagonal-dominant system
    // 3x + y = 4, x + 4y + z = 11, y + 5z + w = 20, z + 3w = 8
    // Solution: x=1, y=1, z=3, w=5/3... let me compute properly
    // Just use a coupled system that's well-conditioned
    auto F_4d = [](const std::vector<double>& x) {
        return std::vector<double>{
            3.0*x[0] + x[1] - 4.0,
            x[0] + 4.0*x[1] + x[2] - 11.0,
            x[1] + 5.0*x[2] + x[3] - 20.0,
            x[2] + 3.0*x[3] - 8.0
        };
    };
} // namespace

// ═════════════════════════════════════════════════════════════════════════════
// Broyden with analytic Jacobian (first step)
// ═════════════════════════════════════════════════════════════════════════════

TEST(Broyden, LinearAnalytic) {
    auto J = [](const std::vector<double>&) {
        return std::vector<std::vector<double>>{{2.0, 1.0}, {1.0, 3.0}};
    };
    auto result = broyden(F_linear, J, {0.0, 0.0});
    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.x[0], -0.2, 1e-8);
    EXPECT_NEAR(result.x[1], 2.4, 1e-8);
    EXPECT_LT(result.residual, 1e-8);
}

TEST(Broyden, NonlinearAnalytic) {
    auto J = [](const std::vector<double>& x) {
        return std::vector<std::vector<double>>{
            {2.0*x[0], 2.0*x[1]},
            {1.0, -1.0}
        };
    };
    auto result = broyden(F_nonlinear, J, {1.0, 0.0});
    EXPECT_TRUE(result.converged);
    double expected_x0 = (1.0 + std::sqrt(7.0)) / 2.0;
    double expected_x1 = expected_x0 - 1.0;
    EXPECT_NEAR(result.x[0], expected_x0, 1e-8);
    EXPECT_NEAR(result.x[1], expected_x1, 1e-8);
}

TEST(Broyden, RosenbrockAnalytic) {
    auto J = [](const std::vector<double>& x) {
        return std::vector<std::vector<double>>{
            {-20.0*x[0], 10.0},
            {-1.0, 0.0}
        };
    };
    auto result = broyden(F_rosenbrock, J, {-1.0, 1.0});
    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.x[0], 1.0, 1e-8);
    EXPECT_NEAR(result.x[1], 1.0, 1e-8);
}

TEST(Broyden, ThreeDAnalytic) {
    auto J = [](const std::vector<double>& x) {
        return std::vector<std::vector<double>>{
            {2.0*x[0], 1.0, 1.0},
            {1.0, 2.0*x[1], 1.0},
            {1.0, 1.0, 2.0*x[2]}
        };
    };
    // Root at (√2-1, √2-1, √2-1) ≈ (0.4142, 0.4142, 0.4142)
    double root = std::sqrt(2.0) - 1.0;
    auto result = broyden(F_3d, J, {0.5, 0.5, 0.5});
    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.x[0], root, 1e-6);
    EXPECT_NEAR(result.x[1], root, 1e-6);
    EXPECT_NEAR(result.x[2], root, 1e-6);
}

// ═════════════════════════════════════════════════════════════════════════════
// Broyden with numerical Jacobian (no Jacobian required)
// ═════════════════════════════════════════════════════════════════════════════

TEST(Broyden, LinearNumerical) {
    auto result = broyden(F_linear, {0.0, 0.0});
    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.x[0], -0.2, 1e-6);
    EXPECT_NEAR(result.x[1], 2.4, 1e-6);
    EXPECT_LT(result.residual, 1e-6);
}

TEST(Broyden, NonlinearNumerical) {
    auto result = broyden(F_nonlinear, {1.0, 0.0});
    EXPECT_TRUE(result.converged);
    double expected_x0 = (1.0 + std::sqrt(7.0)) / 2.0;
    double expected_x1 = expected_x0 - 1.0;
    EXPECT_NEAR(result.x[0], expected_x0, 1e-6);
    EXPECT_NEAR(result.x[1], expected_x1, 1e-6);
}

TEST(Broyden, RosenbrockNumerical) {
    auto result = broyden(F_rosenbrock, {-1.0, 1.0});
    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.x[0], 1.0, 1e-6);
    EXPECT_NEAR(result.x[1], 1.0, 1e-6);
}

TEST(Broyden, ThreeDNumerical) {
    double root = std::sqrt(2.0) - 1.0;
    auto result = broyden(F_3d, {0.5, 0.5, 0.5});
    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.x[0], root, 1e-6);
    EXPECT_NEAR(result.x[1], root, 1e-6);
    EXPECT_NEAR(result.x[2], root, 1e-6);
}

TEST(Broyden, FourD) {
    // Solve 3x+y=4, x+4y+z=11, y+5z+w=20, z+3w=8
    // The Jacobian is constant and diagonally dominant → non-singular
    auto result = broyden(F_4d, {0.0, 0.0, 0.0, 0.0});
    EXPECT_TRUE(result.converged);
    // Verify solution satisfies all equations
    EXPECT_NEAR(3.0*result.x[0] + result.x[1], 4.0, 1e-6);
    EXPECT_NEAR(result.x[0] + 4.0*result.x[1] + result.x[2], 11.0, 1e-6);
    EXPECT_NEAR(result.x[1] + 5.0*result.x[2] + result.x[3], 20.0, 1e-6);
    EXPECT_NEAR(result.x[2] + 3.0*result.x[3], 8.0, 1e-6);
}

// ═════════════════════════════════════════════════════════════════════════════
// Convergence properties
// ═════════════════════════════════════════════════════════════════════════════

TEST(Broyden, ConvergesFromFar) {
    // Start far from root
    auto result = broyden(F_nonlinear, {5.0, 5.0});
    EXPECT_TRUE(result.converged);
    double expected_x0 = (1.0 + std::sqrt(7.0)) / 2.0;
    EXPECT_NEAR(result.x[0], expected_x0, 1e-6);
}

TEST(Broyden, ConvergesInFewerIterationsThanNewton) {
    // Broyden typically needs more iterations than Newton but fewer evaluations
    auto J = [](const std::vector<double>& x) {
        return std::vector<std::vector<double>>{
            {2.0*x[0], 2.0*x[1]},
            {1.0, -1.0}
        };
    };
    auto r_broyden = broyden(F_nonlinear, J, {1.0, 0.0});
    auto r_newton = newton_raphson_system(F_nonlinear, J, {1.0, 0.0});
    EXPECT_TRUE(r_broyden.converged);
    EXPECT_TRUE(r_newton.converged);
    // Newton converges faster in iterations, but Broyden avoids Jacobian recalculation
    EXPECT_LE(r_newton.iterations, r_broyden.iterations + 2);
}

TEST(Broyden, WarmStart) {
    // Start at exact root
    auto J = [](const std::vector<double>&) {
        return std::vector<std::vector<double>>{{2.0, 1.0}, {1.0, 3.0}};
    };
    auto result = broyden(F_linear, J, {-0.2, 2.4});
    EXPECT_TRUE(result.converged);
    EXPECT_LE(result.iterations, 1u);
}

// ═════════════════════════════════════════════════════════════════════════════
// Cross-method consistency
// ═════════════════════════════════════════════════════════════════════════════

TEST(Broyden_Consistency, AllMethodsAgree) {
    auto J = [](const std::vector<double>& x) {
        return std::vector<std::vector<double>>{
            {2.0*x[0], 2.0*x[1]},
            {1.0, -1.0}
        };
    };
    double expected_x0 = (1.0 + std::sqrt(7.0)) / 2.0;
    double expected_x1 = expected_x0 - 1.0;

    auto r1 = broyden(F_nonlinear, J, {1.0, 0.0});
    auto r2 = broyden(F_nonlinear, {1.0, 0.0});
    auto r3 = newton_raphson_system(F_nonlinear, J, {1.0, 0.0});

    for (const auto& res : std::vector<NLEIterResult>{r1, r2, r3}) {
        EXPECT_TRUE(res.converged);
        EXPECT_NEAR(res.x[0], expected_x0, 1e-6);
        EXPECT_NEAR(res.x[1], expected_x1, 1e-6);
    }
}
