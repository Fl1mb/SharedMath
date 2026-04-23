#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "NumericalMethods/Differentiation.h"

using namespace SharedMath::NumericalMethods;

static constexpr double kTight = 1e-7;   // central-difference O(h²), h=1e-5
static constexpr double kLoose = 1e-4;   // higher-order / multivariate

// ─────────────────────────────────────────────────────────────────────────────
// derivative (central difference, 1st order)
// ─────────────────────────────────────────────────────────────────────────────

TEST(Differentiation, DerivativeSin) {
    // d/dx sin(x)|_{x=0} = cos(0) = 1
    EXPECT_NEAR(derivative([](double x){ return std::sin(x); }, 0.0), 1.0, kTight);
}

TEST(Differentiation, DerivativeCos) {
    // d/dx cos(x)|_{x=pi/2} = -sin(pi/2) = -1
    EXPECT_NEAR(derivative([](double x){ return std::cos(x); }, M_PI / 2.0), -1.0, kTight);
}

TEST(Differentiation, DerivativePolynomial) {
    // d/dx x^3|_{x=2} = 3*4 = 12
    EXPECT_NEAR(derivative([](double x){ return x*x*x; }, 2.0), 12.0, kTight);
}

TEST(Differentiation, DerivativeExp) {
    // d/dx e^x = e^x
    double x = 1.5;
    EXPECT_NEAR(derivative([](double x){ return std::exp(x); }, x), std::exp(x), kTight);
}

// ─────────────────────────────────────────────────────────────────────────────
// derivative2 (second derivative)
// ─────────────────────────────────────────────────────────────────────────────

TEST(Differentiation, Derivative2Sin) {
    // d²/dx² sin(x)|_{x=0} = -sin(0) = 0
    EXPECT_NEAR(derivative2([](double x){ return std::sin(x); }, 0.0), 0.0, kTight);
}

TEST(Differentiation, Derivative2Polynomial) {
    // d²/dx² x^4|_{x=1} = 12
    EXPECT_NEAR(derivative2([](double x){ return x*x*x*x; }, 1.0), 12.0, 1e-5);
}

TEST(Differentiation, Derivative2Exp) {
    // d²/dx² e^x = e^x
    double x = 0.7;
    EXPECT_NEAR(derivative2([](double x){ return std::exp(x); }, x), std::exp(x), 1e-5);
}

// ─────────────────────────────────────────────────────────────────────────────
// derivativeN
// ─────────────────────────────────────────────────────────────────────────────

TEST(Differentiation, DerivativeNZeroOrder) {
    // 0-th derivative is the function itself
    EXPECT_NEAR(derivativeN([](double x){ return std::sin(x); }, M_PI / 6.0, 0), 0.5, 1e-12);
}

TEST(Differentiation, DerivativeNFirstOrder) {
    // Should agree with derivative() for n=1
    double x = 0.4;
    double d1   = derivativeN([](double x){ return std::exp(x); }, x, 1, 1e-4);
    EXPECT_NEAR(d1, std::exp(x), kLoose);
}

TEST(Differentiation, DerivativeNThirdOrderCubic) {
    // d³/dx³ (x^3) = 6  everywhere
    EXPECT_NEAR(derivativeN([](double x){ return x*x*x; }, 1.0, 3, 1e-3), 6.0, 1e-3);
}

TEST(Differentiation, DerivativeNThrowsNegative) {
    EXPECT_THROW(derivativeN([](double x){ return x; }, 0.0, -1), std::invalid_argument);
}

// ─────────────────────────────────────────────────────────────────────────────
// partial derivative
// ─────────────────────────────────────────────────────────────────────────────

TEST(Differentiation, PartialX) {
    // f(x,y) = x^2 + y^2,  ∂f/∂x = 2x
    auto f = [](const std::vector<double>& v){ return v[0]*v[0] + v[1]*v[1]; };
    EXPECT_NEAR(partial(f, {3.0, 4.0}, 0), 6.0, kTight);
}

TEST(Differentiation, PartialY) {
    // f(x,y) = x^2 + y^2,  ∂f/∂y = 2y
    auto f = [](const std::vector<double>& v){ return v[0]*v[0] + v[1]*v[1]; };
    EXPECT_NEAR(partial(f, {3.0, 4.0}, 1), 8.0, kTight);
}

TEST(Differentiation, PartialMixed) {
    // f(x,y,z) = x*y*z,  ∂f/∂z|_{(1,2,3)} = x*y = 2
    auto f = [](const std::vector<double>& v){ return v[0]*v[1]*v[2]; };
    EXPECT_NEAR(partial(f, {1.0, 2.0, 3.0}, 2), 2.0, kTight);
}

// ─────────────────────────────────────────────────────────────────────────────
// gradient
// ─────────────────────────────────────────────────────────────────────────────

TEST(Differentiation, GradientQuadratic) {
    // f(x,y,z) = x^2 + 2y^2 + 3z^2
    // ∇f = [2x, 4y, 6z]
    auto f = [](const std::vector<double>& v){
        return v[0]*v[0] + 2*v[1]*v[1] + 3*v[2]*v[2];
    };
    std::vector<double> x = {1.0, 2.0, 3.0};
    auto g = gradient(f, x);
    ASSERT_EQ(g.size(), 3u);
    EXPECT_NEAR(g[0], 2.0,  kTight);
    EXPECT_NEAR(g[1], 8.0,  kTight);
    EXPECT_NEAR(g[2], 18.0, kTight);
}

TEST(Differentiation, GradientAtMinimum) {
    // f(x,y) = (x-1)^2 + (y-2)^2,  minimum at (1,2), gradient = [0,0]
    auto f = [](const std::vector<double>& v){
        return (v[0]-1)*(v[0]-1) + (v[1]-2)*(v[1]-2);
    };
    auto g = gradient(f, {1.0, 2.0});
    EXPECT_NEAR(g[0], 0.0, kTight);
    EXPECT_NEAR(g[1], 0.0, kTight);
}

// ─────────────────────────────────────────────────────────────────────────────
// jacobian
// ─────────────────────────────────────────────────────────────────────────────

TEST(Differentiation, JacobianLinear) {
    // f(x,y) = [x*y, x+y]
    // J = [[y, x], [1, 1]]  at (2, 3): [[3,2],[1,1]]
    auto f = [](const std::vector<double>& v) -> std::vector<double> {
        return {v[0]*v[1], v[0]+v[1]};
    };
    auto J = jacobian(f, {2.0, 3.0});
    ASSERT_EQ(J.rows(), 2u);
    ASSERT_EQ(J.cols(), 2u);
    EXPECT_NEAR(J(0, 0), 3.0, kTight);  // ∂(xy)/∂x = y = 3
    EXPECT_NEAR(J(0, 1), 2.0, kTight);  // ∂(xy)/∂y = x = 2
    EXPECT_NEAR(J(1, 0), 1.0, kTight);  // ∂(x+y)/∂x = 1
    EXPECT_NEAR(J(1, 1), 1.0, kTight);  // ∂(x+y)/∂y = 1
}

TEST(Differentiation, JacobianIdentityMap) {
    // f(x,y,z) = [x, y, z] → J = I_3
    auto f = [](const std::vector<double>& v) -> std::vector<double> {
        return {v[0], v[1], v[2]};
    };
    auto J = jacobian(f, {1.0, 2.0, 3.0});
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_NEAR(J(i, j), i == j ? 1.0 : 0.0, kTight);
}

// ─────────────────────────────────────────────────────────────────────────────
// hessian
// ─────────────────────────────────────────────────────────────────────────────

TEST(Differentiation, HessianQuadratic) {
    // f(x,y) = x^2 + x*y + y^2
    // H = [[2, 1], [1, 2]]
    auto f = [](const std::vector<double>& v){
        return v[0]*v[0] + v[0]*v[1] + v[1]*v[1];
    };
    auto H = hessian(f, {1.0, 1.0});
    ASSERT_EQ(H.rows(), 2u);
    EXPECT_NEAR(H(0, 0), 2.0, kLoose);
    EXPECT_NEAR(H(0, 1), 1.0, kLoose);
    EXPECT_NEAR(H(1, 0), 1.0, kLoose);
    EXPECT_NEAR(H(1, 1), 2.0, kLoose);
}

TEST(Differentiation, HessianSymmetric) {
    // Hessian of any smooth f must be symmetric
    auto f = [](const std::vector<double>& v){
        return std::sin(v[0]*v[1]) + v[0]*v[0]*v[1];
    };
    auto H = hessian(f, {0.5, 0.7});
    EXPECT_NEAR(H(0, 1), H(1, 0), 1e-8);
}

TEST(Differentiation, HessianUnitSphere) {
    // f(x,y,z) = x^2+y^2+z^2,  H = 2*I
    auto f = [](const std::vector<double>& v){
        return v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
    };
    auto H = hessian(f, {1.0, 1.0, 1.0});
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_NEAR(H(i, j), i == j ? 2.0 : 0.0, kLoose);
}
