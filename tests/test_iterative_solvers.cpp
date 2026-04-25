#include <gtest/gtest.h>
#include "LinearAlgebra/IterativeSolvers.h"
#include "LinearAlgebra/MatrixFunctions.h"
#include "LinearAlgebra/DynamicMatrix.h"
#include "LinearAlgebra/Tensor.h"

#include <cmath>
#include <vector>

using namespace SharedMath::LinearAlgebra;

static constexpr double kTol = 1e-6;

// ════════════════════════════════════════════════════════════════════════════
// Conjugate Gradient
// ════════════════════════════════════════════════════════════════════════════

TEST(CG, TwoByTwoSPD) {
    // [[4, 1], [1, 3]] x = [1, 2]  → x = [1/11, 7/11]
    DynamicMatrix A(2, 2);
    A.set(0,0,4); A.set(0,1,1);
    A.set(1,0,1); A.set(1,1,3);
    std::vector<double> b = {1.0, 2.0};

    auto x = cg(A, b);

    // Verify Ax = b
    EXPECT_NEAR(4*x[0] + x[1], 1.0, kTol);
    EXPECT_NEAR(  x[0] + 3*x[1], 2.0, kTol);
}

TEST(CG, ThreeByThreeSPD) {
    // A = diag(1, 2, 3), b = [1, 2, 3] → x = [1, 1, 1]
    DynamicMatrix A = diag({1.0, 2.0, 3.0});
    std::vector<double> b = {1.0, 2.0, 3.0};

    auto x = cg(A, b);
    EXPECT_NEAR(x[0], 1.0, kTol);
    EXPECT_NEAR(x[1], 1.0, kTol);
    EXPECT_NEAR(x[2], 1.0, kTol);
}

TEST(CG, Identity) {
    DynamicMatrix I = eye(3);
    std::vector<double> b = {2.0, -1.0, 5.0};
    auto x = cg(I, b);
    for (size_t i = 0; i < b.size(); ++i)
        EXPECT_NEAR(x[i], b[i], kTol);
}

TEST(CG, ZeroRhsGivesZero) {
    DynamicMatrix A = diag({2.0, 3.0});
    std::vector<double> b(2, 0.0);
    auto x = cg(A, b);
    for (double xi : x) EXPECT_NEAR(xi, 0.0, kTol);
}

TEST(CG, NonSquareThrows) {
    DynamicMatrix A(2, 3);
    EXPECT_THROW(cg(A, {1.0, 2.0}), std::invalid_argument);
}

// ════════════════════════════════════════════════════════════════════════════
// LSQR
// ════════════════════════════════════════════════════════════════════════════

TEST(LSQR, SquareSystem) {
    // Exact solution exists: same as solve()
    DynamicMatrix A(2, 2);
    A.set(0,0,2); A.set(0,1,1);
    A.set(1,0,1); A.set(1,1,3);
    std::vector<double> b = {5.0, 10.0};

    auto x = lsqr(A, b);
    EXPECT_NEAR(x[0], 1.0, kTol);
    EXPECT_NEAR(x[1], 3.0, kTol);
}

TEST(LSQR, OverdeterminedExact) {
    // 3 equations, 1 unknown: A=[1;1;1], b=[2;2;2] → x=[2]
    DynamicMatrix A(3, 1);
    A.set(0,0,1); A.set(1,0,1); A.set(2,0,1);
    std::vector<double> b = {2.0, 2.0, 2.0};

    auto x = lsqr(A, b);
    ASSERT_EQ(x.size(), 1u);
    EXPECT_NEAR(x[0], 2.0, kTol);
}

TEST(LSQR, OverdeterminedLeastSquares) {
    // 3 equations, 2 unknowns; b not in range of A — just check residual decreases
    DynamicMatrix A(3, 2);
    A.set(0,0,1); A.set(0,1,1);
    A.set(1,0,1); A.set(1,1,2);
    A.set(2,0,1); A.set(2,1,3);
    std::vector<double> b = {1.0, 2.0, 3.5};

    auto x = lsqr(A, b);
    ASSERT_EQ(x.size(), 2u);
    EXPECT_TRUE(std::isfinite(x[0]));
    EXPECT_TRUE(std::isfinite(x[1]));

    // Normal equations: A^T A x = A^T b
    // Check residual is roughly minimal by comparing to lstsq
    auto x_ref = lstsq(A, b);
    for (size_t i = 0; i < 2; ++i)
        EXPECT_NEAR(x[i], x_ref[i], 1e-4);
}

TEST(LSQR, Identity) {
    DynamicMatrix I = eye(3);
    std::vector<double> b = {3.0, -1.0, 2.0};
    auto x = lsqr(I, b);
    for (size_t i = 0; i < b.size(); ++i)
        EXPECT_NEAR(x[i], b[i], kTol);
}

TEST(LSQR, DimMismatchThrows) {
    DynamicMatrix A(3, 2);
    EXPECT_THROW(lsqr(A, {1.0, 2.0}), std::invalid_argument);
}

// ════════════════════════════════════════════════════════════════════════════
// CPU fallback when CUDA is unavailable
// ════════════════════════════════════════════════════════════════════════════

// Tensor::cuda() must be safe to call even without a GPU.
// If CUDA is not available, it returns a CPU tensor and all ops stay on CPU.
TEST(CUDAFallback, CudaCallNeverCrashes) {
    Tensor A = Tensor::from_matrix(2, 2, {1.0, 2.0, 3.0, 4.0});
    // This should either move to GPU (if available) or stay on CPU silently.
    Tensor A_dev = A.cuda();
    EXPECT_TRUE(A_dev.device() == Device::CPU ||
                A_dev.device() == Device::CUDA);
}

TEST(CUDAFallback, MatmulOnCudaOrCpuGivesSameResult) {
    Tensor A = Tensor::from_matrix(2, 2, {1.0, 0.0, 0.0, 1.0}); // identity
    Tensor B = Tensor::from_matrix(2, 2, {3.0, 4.0, 5.0, 6.0});

    // Compute on whatever device is available
    Tensor C = A.cuda().matmul(B.cuda()).cpu();
    EXPECT_NEAR(C(0, 0), 3.0, 1e-9);
    EXPECT_NEAR(C(0, 1), 4.0, 1e-9);
    EXPECT_NEAR(C(1, 0), 5.0, 1e-9);
    EXPECT_NEAR(C(1, 1), 6.0, 1e-9);
}

TEST(CUDAFallback, UnaryOpsWorkOnAnyDevice) {
    Tensor A({4}, {0.0, 1.0, 2.0, 3.0});
    Tensor E = A.cuda().exp().cpu();
    EXPECT_NEAR(E.flat(0), 1.0,           1e-6);
    EXPECT_NEAR(E.flat(1), std::exp(1.0), 1e-6);
    EXPECT_NEAR(E.flat(2), std::exp(2.0), 1e-6);
}

TEST(CUDAFallback, DeviceRoundTrip) {
    // .cuda().cpu() must reproduce original data exactly.
    std::vector<double> orig = {1.5, -2.3, 3.7, 0.0};
    Tensor A({4}, orig);
    Tensor B = A.cuda().cpu();
    ASSERT_EQ(B.size(), orig.size());
    for (size_t i = 0; i < orig.size(); ++i)
        EXPECT_DOUBLE_EQ(B.flat(i), orig[i]);
}
