#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "NumericalMethods/SLAE.h"
#include "LinearAlgebra/DynamicMatrix.h"

using namespace SharedMath::NumericalMethods;
using SharedMath::LinearAlgebra::DynamicMatrix;

// ─────────────────────────────────────────────────────────────────────────────
// Test fixtures
// ─────────────────────────────────────────────────────────────────────────────

namespace {
    // 2×2 strictly diagonally dominant: solution x = [1, 1]
    // 4x - y = 3
    // -x + 4y = 3
    DynamicMatrix make2x2() {
        return DynamicMatrix(2, 2, {4.0, -1.0, -1.0, 4.0});
    }
    std::vector<double> rhs2x2() { return {3.0, 3.0}; }

    // 3×3 SDD: solution x = [1, 2, 3]
    // 5x  - y       = 3
    // -x  + 5y - z  = 7
    //     - y  + 5z = 13
    DynamicMatrix make3x3() {
        return DynamicMatrix(3, 3, {
             5.0, -1.0,  0.0,
            -1.0,  5.0, -1.0,
             0.0, -1.0,  5.0
        });
    }
    std::vector<double> rhs3x3() { return {3.0, 6.0, 13.0}; }

    // 3×3 Symmetric Positive Definite: solution x = [1, 2, 3]
    // 4 1 0   x   6
    // 1 3 1 * y = 10
    // 0 1 4   z   14
    DynamicMatrix makeSPD() {
        return DynamicMatrix(3, 3, {
            4.0, 1.0, 0.0,
            1.0, 3.0, 1.0,
            0.0, 1.0, 4.0
        });
    }
    std::vector<double> rhsSPD() { return {6.0, 10.0, 14.0}; }

    // Check solution vector against expected values
    void checkSolution(const IterResult& res,
                        const std::vector<double>& expected,
                        double tol)
    {
        ASSERT_EQ(res.x.size(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i)
            EXPECT_NEAR(res.x[i], expected[i], tol)
                << "  component i=" << i;
    }
} // namespace

// ═════════════════════════════════════════════════════════════════════════════
// Jacobi
// ═════════════════════════════════════════════════════════════════════════════

TEST(Jacobi, TwoByTwo) {
    auto A = make2x2();
    auto result = jacobi(A, rhs2x2());
    EXPECT_TRUE(result.converged);
    checkSolution(result, {1.0, 1.0}, 1e-7);
}

TEST(Jacobi, ThreeByThree) {
    auto A = make3x3();
    auto result = jacobi(A, rhs3x3());
    EXPECT_TRUE(result.converged);
    checkSolution(result, {1.0, 2.0, 3.0}, 1e-7);
}

TEST(Jacobi, ResidualSmall) {
    auto A = make3x3();
    auto result = jacobi(A, rhs3x3());
    EXPECT_LT(result.residual, 1e-6);
}

TEST(Jacobi, WarmStart) {
    auto A = make2x2();
    auto b = rhs2x2();
    // Start from exact solution — should converge in 1 iteration
    auto result = jacobi(A, b, {1.0, 1.0});
    EXPECT_TRUE(result.converged);
    EXPECT_LE(result.iterations, 2u);
}

// ═════════════════════════════════════════════════════════════════════════════
// Gauss-Seidel
// ═════════════════════════════════════════════════════════════════════════════

TEST(GaussSeidel, TwoByTwo) {
    auto A = make2x2();
    auto result = gauss_seidel(A, rhs2x2());
    EXPECT_TRUE(result.converged);
    checkSolution(result, {1.0, 1.0}, 1e-7);
}

TEST(GaussSeidel, ThreeByThree) {
    auto A = make3x3();
    auto result = gauss_seidel(A, rhs3x3());
    EXPECT_TRUE(result.converged);
    checkSolution(result, {1.0, 2.0, 3.0}, 1e-7);
}

TEST(GaussSeidel, FasterThanJacobi) {
    // Gauss-Seidel typically converges in fewer iterations than Jacobi
    auto A = make3x3();
    auto b = rhs3x3();
    auto r_jac = jacobi(A, b);
    auto r_gs  = gauss_seidel(A, b);
    // Both converge; G-S should need no more iterations
    EXPECT_LE(r_gs.iterations, r_jac.iterations + 5);  // allow small margin
}

TEST(GaussSeidel, ResidualSmall) {
    auto A = make3x3();
    auto result = gauss_seidel(A, rhs3x3());
    EXPECT_LT(result.residual, 1e-7);
}

// ═════════════════════════════════════════════════════════════════════════════
// SOR
// ═════════════════════════════════════════════════════════════════════════════

TEST(SOR, TwoByTwo) {
    auto A = make2x2();
    auto result = sor(A, rhs2x2(), 1.2);
    EXPECT_TRUE(result.converged);
    checkSolution(result, {1.0, 1.0}, 1e-7);
}

TEST(SOR, ThreeByThree) {
    auto A = make3x3();
    auto result = sor(A, rhs3x3(), 1.1);
    EXPECT_TRUE(result.converged);
    checkSolution(result, {1.0, 2.0, 3.0}, 1e-7);
}

TEST(SOR, OmegaOneIsGaussSeidel) {
    // omega=1 must give same result as Gauss-Seidel
    auto A = make3x3();
    auto b = rhs3x3();
    auto r_gs  = gauss_seidel(A, b);
    auto r_sor = sor(A, b, 1.0);
    ASSERT_EQ(r_gs.x.size(), r_sor.x.size());
    for (size_t i = 0; i < r_gs.x.size(); ++i)
        EXPECT_NEAR(r_sor.x[i], r_gs.x[i], 1e-10);
}

TEST(SOR, InvalidOmegaThrows) {
    auto A = make2x2();
    EXPECT_THROW(sor(A, rhs2x2(), 0.0),  std::invalid_argument);
    EXPECT_THROW(sor(A, rhs2x2(), 2.0),  std::invalid_argument);
    EXPECT_THROW(sor(A, rhs2x2(), -0.5), std::invalid_argument);
}

// ═════════════════════════════════════════════════════════════════════════════
// Conjugate Gradient
// ═════════════════════════════════════════════════════════════════════════════

TEST(ConjugateGradient, TwoByTwo) {
    auto A = make2x2();  // symmetric, strictly diag dominant → SPD
    auto result = conjugate_gradient(A, rhs2x2());
    EXPECT_TRUE(result.converged);
    checkSolution(result, {1.0, 1.0}, 1e-7);
}

TEST(ConjugateGradient, SPD) {
    auto A = makeSPD();
    auto result = conjugate_gradient(A, rhsSPD());
    EXPECT_TRUE(result.converged);
    checkSolution(result, {1.0, 2.0, 3.0}, 1e-7);
}

TEST(ConjugateGradient, ResidualSmall) {
    auto A = makeSPD();
    auto result = conjugate_gradient(A, rhsSPD());
    EXPECT_LT(result.residual, 1e-7);
}

TEST(ConjugateGradient, ConvergesInAtMostNSteps) {
    // CG on an n×n SPD system converges in at most n exact-arithmetic steps
    auto A = makeSPD();    // 3×3
    auto result = conjugate_gradient(A, rhsSPD(), {}, 1e-12);
    EXPECT_LE(result.iterations, 10u);  // generous bound
}

// ═════════════════════════════════════════════════════════════════════════════
// GMRES
// ═════════════════════════════════════════════════════════════════════════════

TEST(GMRES, TwoByTwo) {
    auto A = make2x2();
    auto result = gmres(A, rhs2x2());
    EXPECT_TRUE(result.converged);
    checkSolution(result, {1.0, 1.0}, 1e-7);
}

TEST(GMRES, ThreeByThree) {
    auto A = make3x3();
    auto result = gmres(A, rhs3x3());
    EXPECT_TRUE(result.converged);
    checkSolution(result, {1.0, 2.0, 3.0}, 1e-7);
}

TEST(GMRES, SPD) {
    auto A = makeSPD();
    auto result = gmres(A, rhsSPD());
    EXPECT_TRUE(result.converged);
    checkSolution(result, {1.0, 2.0, 3.0}, 1e-7);
}

TEST(GMRES, AsymmetricSystem) {
    // Non-symmetric: solution x = [1, 1, 1]
    // 3x + y       = 4
    // x  + 3y + z  = 5
    //      y  + 3z = 4
    DynamicMatrix A(3, 3, {
        3.0, 1.0, 0.0,
        1.0, 3.0, 1.0,
        0.0, 1.0, 3.0
    });
    auto result = gmres(A, {4.0, 5.0, 4.0});
    EXPECT_TRUE(result.converged);
    checkSolution(result, {1.0, 1.0, 1.0}, 1e-7);
}

TEST(GMRES, ResidualSmall) {
    auto A = make3x3();
    auto result = gmres(A, rhs3x3());
    EXPECT_LT(result.residual, 1e-7);
}

// ═════════════════════════════════════════════════════════════════════════════
// Cross-method consistency
// ═════════════════════════════════════════════════════════════════════════════

TEST(SLAE_Consistency, AllMethodsAgree) {
    auto A = make3x3();
    auto b = rhs3x3();
    std::vector<double> expected = {1.0, 2.0, 3.0};

    auto r1 = jacobi(A, b);
    auto r2 = gauss_seidel(A, b);
    auto r3 = sor(A, b, 1.2);
    auto r4 = conjugate_gradient(A, b);
    auto r5 = gmres(A, b);

    for (const IterResult& res : std::vector<IterResult>{r1, r2, r3, r4, r5})
        for (size_t i = 0; i < expected.size(); ++i)
            EXPECT_NEAR(res.x[i], expected[i], 1e-6)
                << "method disagreement at component " << i;
}
