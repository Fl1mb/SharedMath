#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "NumericalMethods/NumericalMethodsGPU.h"

using namespace SharedMath::NumericalMethods;

// ═════════════════════════════════════════════════════════════════════════════
// GPUNumVec tests
// ═════════════════════════════════════════════════════════════════════════════

TEST(GPUNumVec, RoundTrip) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    GPUNumVec v(data);
    EXPECT_EQ(v.size(), 5u);
    EXPECT_EQ(v.device(), Device::CPU);

    auto vg = v.toGPU();
    EXPECT_EQ(vg.device(), Device::CUDA);
    EXPECT_EQ(vg.size(), 5u);

    auto vc = vg.toCPU();
    EXPECT_EQ(vc.device(), Device::CPU);
    ASSERT_EQ(vc.host().size(), 5u);
    for (size_t i = 0; i < 5; ++i)
        EXPECT_NEAR(vc.host()[i], data[i], 1e-15);
}

TEST(GPUNumVec, AddCPU) {
    GPUNumVec a({1.0, 2.0, 3.0});
    GPUNumVec b({4.0, 5.0, 6.0});
    auto r = GPUNumVec::add(a, b);
    EXPECT_EQ(r.device(), Device::CPU);
    EXPECT_NEAR(r.host()[0], 5.0, 1e-15);
    EXPECT_NEAR(r.host()[1], 7.0, 1e-15);
    EXPECT_NEAR(r.host()[2], 9.0, 1e-15);
}

TEST(GPUNumVec, AddGPU) {
    GPUNumVec a({1.0, 2.0, 3.0});
    GPUNumVec b({4.0, 5.0, 6.0});
    auto ag = a.toGPU();
    auto bg = b.toGPU();
    auto rg = GPUNumVec::add(ag, bg);
    EXPECT_EQ(rg.device(), Device::CUDA);
    auto rc = rg.toCPU();
    EXPECT_NEAR(rc.host()[0], 5.0, 1e-15);
    EXPECT_NEAR(rc.host()[1], 7.0, 1e-15);
    EXPECT_NEAR(rc.host()[2], 9.0, 1e-15);
}

TEST(GPUNumVec, ScaleCPU) {
    GPUNumVec v({1.0, 2.0, 3.0});
    auto r = GPUNumVec::scale(2.5, v);
    EXPECT_NEAR(r.host()[0], 2.5, 1e-15);
    EXPECT_NEAR(r.host()[1], 5.0, 1e-15);
    EXPECT_NEAR(r.host()[2], 7.5, 1e-15);
}

TEST(GPUNumVec, ScaleGPU) {
    GPUNumVec v({1.0, 2.0, 3.0});
    auto vg = v.toGPU();
    auto rg = GPUNumVec::scale(2.5, vg);
    auto rc = rg.toCPU();
    EXPECT_NEAR(rc.host()[0], 2.5, 1e-15);
    EXPECT_NEAR(rc.host()[1], 5.0, 1e-15);
    EXPECT_NEAR(rc.host()[2], 7.5, 1e-15);
}

TEST(GPUNumVec, DotCPU) {
    GPUNumVec a({1.0, 2.0, 3.0});
    GPUNumVec b({4.0, 5.0, 6.0});
    double d = GPUNumVec::dot(a, b);
    EXPECT_NEAR(d, 32.0, 1e-12);  // 1*4 + 2*5 + 3*6 = 32
}

TEST(GPUNumVec, DotGPU) {
    GPUNumVec a({1.0, 2.0, 3.0});
    GPUNumVec b({4.0, 5.0, 6.0});
    auto ag = a.toGPU();
    auto bg = b.toGPU();
    double d = GPUNumVec::dot(ag, bg);
    EXPECT_NEAR(d, 32.0, 1e-12);
}

TEST(GPUNumVec, Nrm2CPU) {
    GPUNumVec v({3.0, 4.0});
    double n = GPUNumVec::nrm2(v);
    EXPECT_NEAR(n, 5.0, 1e-12);  // sqrt(9+16) = 5
}

TEST(GPUNumVec, Nrm2GPU) {
    GPUNumVec v({3.0, 4.0});
    auto vg = v.toGPU();
    double n = GPUNumVec::nrm2(vg);
    EXPECT_NEAR(n, 5.0, 1e-12);
}

TEST(GPUNumVec, AxpypassCPU) {
    GPUNumVec x({1.0, 2.0, 3.0});
    GPUNumVec y({10.0, 20.0, 30.0});
    auto r = GPUNumVec::axpy(2.0, x, y);
    // r = 2*x + y = {12, 24, 36}
    EXPECT_NEAR(r.host()[0], 12.0, 1e-12);
    EXPECT_NEAR(r.host()[1], 24.0, 1e-12);
    EXPECT_NEAR(r.host()[2], 36.0, 1e-12);
}

TEST(GPUNumVec, AxpypassGPU) {
    GPUNumVec x({1.0, 2.0, 3.0});
    GPUNumVec y({10.0, 20.0, 30.0});
    auto xg = x.toGPU();
    auto yg = y.toGPU();
    auto rg = GPUNumVec::axpy(2.0, xg, yg);
    auto rc = rg.toCPU();
    EXPECT_NEAR(rc.host()[0], 12.0, 1e-12);
    EXPECT_NEAR(rc.host()[1], 24.0, 1e-12);
    EXPECT_NEAR(rc.host()[2], 36.0, 1e-12);
}

TEST(GPUNumVec, SizeMismatch) {
    GPUNumVec a({1.0, 2.0});
    GPUNumVec b({1.0, 2.0, 3.0});
    EXPECT_THROW(GPUNumVec::add(a, b), std::invalid_argument);
    EXPECT_THROW(GPUNumVec::dot(a, b), std::invalid_argument);
}

// ═════════════════════════════════════════════════════════════════════════════
// GPUNumMat tests
// ═════════════════════════════════════════════════════════════════════════════

TEST(GPUNumMat, RoundTrip) {
    // 2x3 matrix: [1 2 3; 4 5 6]
    std::vector<double> data = {1, 2, 3, 4, 5, 6};
    GPUNumMat m(2, 3, data);
    EXPECT_EQ(m.rows(), 2u);
    EXPECT_EQ(m.cols(), 3u);

    auto mg = m.toGPU();
    EXPECT_EQ(mg.device(), Device::CUDA);

    auto mc = mg.toCPU();
    EXPECT_EQ(mc.device(), Device::CPU);
    for (size_t i = 0; i < 6; ++i)
        EXPECT_NEAR(mc.host()[i], data[i], 1e-15);
}

TEST(GPUNumMat, GemvCPU) {
    // A = [1 2; 3 4], x = [5; 6]
    // A*x = [1*5+2*6; 3*5+4*6] = [17; 39]
    std::vector<double> A = {1, 2, 3, 4};
    GPUNumMat m(2, 2, A);
    GPUNumVec x({5.0, 6.0});
    auto r = m.gemv(x);
    EXPECT_NEAR(r.host()[0], 17.0, 1e-12);
    EXPECT_NEAR(r.host()[1], 39.0, 1e-12);
}

TEST(GPUNumMat, GemvGPU) {
    std::vector<double> A = {1, 2, 3, 4};
    GPUNumMat m(2, 2, A);
    GPUNumVec x({5.0, 6.0});
    auto mg = m.toGPU();
    auto xg = x.toGPU();
    auto rg = mg.gemv(xg);
    EXPECT_EQ(rg.device(), Device::CUDA);
    auto rc = rg.toCPU();
    EXPECT_NEAR(rc.host()[0], 17.0, 1e-12);
    EXPECT_NEAR(rc.host()[1], 39.0, 1e-12);
}

TEST(GPUNumMat, GerCPU) {
    // A = [1 0; 0 1], u = [2; 3], v = [4, 5]
    // A += u*v^T = [1+8, 0+10; 0+12, 1+15] = [9, 10; 12, 16]
    std::vector<double> A = {1, 0, 0, 1};
    GPUNumMat m(2, 2, A);
    GPUNumVec u({2.0, 3.0});
    GPUNumVec v({4.0, 5.0});
    m.ger(1.0, u, v);
    EXPECT_NEAR(m.host()[0], 9.0, 1e-12);
    EXPECT_NEAR(m.host()[1], 10.0, 1e-12);
    EXPECT_NEAR(m.host()[2], 12.0, 1e-12);
    EXPECT_NEAR(m.host()[3], 16.0, 1e-12);
}

TEST(GPUNumMat, GerGPU) {
    std::vector<double> A = {1, 0, 0, 1};
    GPUNumMat m(2, 2, A);
    GPUNumVec u({2.0, 3.0});
    GPUNumVec v({4.0, 5.0});
    auto mg = m.toGPU();
    auto ug = u.toGPU();
    auto vg = v.toGPU();
    mg.ger(1.0, ug, vg);
    auto mc = mg.toCPU();
    EXPECT_NEAR(mc.host()[0], 9.0, 1e-12);
    EXPECT_NEAR(mc.host()[1], 10.0, 1e-12);
    EXPECT_NEAR(mc.host()[2], 12.0, 1e-12);
    EXPECT_NEAR(mc.host()[3], 16.0, 1e-12);
}

TEST(GPUNumMat, SolveCPU) {
    // [2 1; 1 3] * x = [5; 7]
    // 2x + y = 5, x + 3y = 7 → x = 8/5 = 1.6, y = 9/5 = 1.8
    std::vector<double> A = {2, 1, 1, 3};
    GPUNumMat m(2, 2, A);
    GPUNumVec b({5.0, 7.0});
    auto x = m.solve(b);
    EXPECT_NEAR(x.host()[0], 1.6, 1e-10);
    EXPECT_NEAR(x.host()[1], 1.8, 1e-10);
}

TEST(GPUNumMat, SolveGPU) {
    std::vector<double> A = {2, 1, 1, 3};
    GPUNumMat m(2, 2, A);
    GPUNumVec b({5.0, 7.0});
    auto mg = m.toGPU();
    auto bg = b.toGPU();
    auto xg = mg.solve(bg);
    EXPECT_EQ(xg.device(), Device::CUDA);
    auto xc = xg.toCPU();
    EXPECT_NEAR(xc.host()[0], 1.6, 1e-10);
    EXPECT_NEAR(xc.host()[1], 1.8, 1e-10);
}

TEST(GPUNumMat, Solve3x3) {
    // 3x3 identity-like system: I * x = b → x = b
    std::vector<double> A = {
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    };
    GPUNumMat m(3, 3, A);
    GPUNumVec b({4.0, 11.0, 17.0});

    auto mg = m.toGPU();
    auto bg = b.toGPU();
    auto xg = mg.solve(bg);
    auto xc = xg.toCPU();

    EXPECT_NEAR(xc.host()[0], 4.0, 1e-10);
    EXPECT_NEAR(xc.host()[1], 11.0, 1e-10);
    EXPECT_NEAR(xc.host()[2], 17.0, 1e-10);
}

TEST(GPUNumMat, DimMismatch) {
    GPUNumMat m(2, 2, {1, 0, 0, 1});
    GPUNumVec x({1.0, 2.0, 3.0});
    EXPECT_THROW(m.gemv(x), std::invalid_argument);
}

TEST(GPUNumMat, NonSquareSolve) {
    GPUNumMat m(2, 3, {1, 0, 0, 0, 1, 0});
    GPUNumVec b({1.0, 2.0});
    EXPECT_THROW(m.solve(b), std::invalid_argument);
}

// ═════════════════════════════════════════════════════════════════════════════
// GPU-accelerated solver tests
// ═════════════════════════════════════════════════════════════════════════════

#include "NumericalMethods/SLAE.h"
#include "NumericalMethods/NonlinearEquations.h"
#include "LinearAlgebra/DynamicMatrix.h"

using SharedMath::LinearAlgebra::DynamicMatrix;

namespace {
    // SPD matrix for CG: [4 1; 1 3]
    DynamicMatrix makeSPD() {
        return DynamicMatrix(2, 2, {4.0, 1.0, 1.0, 3.0});
    }

    // General non-symmetric matrix for GMRES: [3 1; 1 3]
    DynamicMatrix makeSym() {
        return DynamicMatrix(2, 2, {3.0, 1.0, 1.0, 3.0});
    }

    void checkSolution(const IterResult& res,
                        const std::vector<double>& expected,
                        double tol)
    {
        ASSERT_EQ(res.x.size(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i)
            EXPECT_NEAR(res.x[i], expected[i], tol)
                << "  component i=" << i;
    }
}

// ── GPU Jacobi ──────────────────────────────────────────────────────────────

TEST(GPUSolver, Jacobi_CUDA) {
    // 4x - y = 3, -x + 4y = 3  →  x=1, y=1
    DynamicMatrix A(2, 2, {4.0, -1.0, -1.0, 4.0});
    auto result = jacobi_cuda(A, {3.0, 3.0});
    EXPECT_TRUE(result.converged);
    checkSolution(result, {1.0, 1.0}, 1e-7);
}

TEST(GPUSolver, Jacobi_CPU_vs_CUDA) {
    DynamicMatrix A(2, 2, {4.0, -1.0, -1.0, 4.0});
    auto r_cpu = jacobi(A, {3.0, 3.0});
    auto r_gpu = jacobi_cuda(A, {3.0, 3.0});
    EXPECT_TRUE(r_cpu.converged);
    EXPECT_TRUE(r_gpu.converged);
    for (size_t i = 0; i < r_cpu.x.size(); ++i)
        EXPECT_NEAR(r_cpu.x[i], r_gpu.x[i], 1e-10);
}

// ── GPU Conjugate Gradient ──────────────────────────────────────────────────

TEST(GPUSolver, CG_CUDA) {
    auto A = makeSPD();
    // Solution: x = [1, 1]  →  b = [5, 4]
    auto result = conjugate_gradient_cuda(A, {5.0, 4.0});
    EXPECT_TRUE(result.converged);
    checkSolution(result, {1.0, 1.0}, 1e-7);
}

TEST(GPUSolver, CG_CPU_vs_CUDA) {
    auto A = makeSPD();
    std::vector<double> b = {5.0, 4.0};
    auto r_cpu = conjugate_gradient(A, b);
    auto r_gpu = conjugate_gradient_cuda(A, b);
    EXPECT_TRUE(r_cpu.converged);
    EXPECT_TRUE(r_gpu.converged);
    for (size_t i = 0; i < r_cpu.x.size(); ++i)
        EXPECT_NEAR(r_cpu.x[i], r_gpu.x[i], 1e-7);
}

// ── GPU GMRES ───────────────────────────────────────────────────────────────

TEST(GPUSolver, GMRES_CUDA) {
    auto A = makeSym();
    // Solution: x = [1, 1]  →  b = [4, 4]
    auto result = gmres_cuda(A, {4.0, 4.0});
    EXPECT_TRUE(result.converged);
    checkSolution(result, {1.0, 1.0}, 1e-7);
}

TEST(GPUSolver, GMRES_CPU_vs_CUDA) {
    DynamicMatrix A(3, 3, {
        3.0, 1.0, 0.0,
        1.0, 3.0, 1.0,
        0.0, 1.0, 3.0
    });
    std::vector<double> b = {4.0, 5.0, 4.0};
    auto r_cpu = gmres(A, b);
    auto r_gpu = gmres_cuda(A, b);
    EXPECT_TRUE(r_cpu.converged);
    EXPECT_TRUE(r_gpu.converged);
    for (size_t i = 0; i < r_cpu.x.size(); ++i)
        EXPECT_NEAR(r_cpu.x[i], r_gpu.x[i], 1e-6);
}

// ── GPU Broyden ─────────────────────────────────────────────────────────────

TEST(GPUSolver, Broyden_CUDA) {
    // F(x) = [2*x0 + x1 - 2, x0 + 3*x1 - 7]  →  x = [-0.2, 2.4]
    auto F = [](const std::vector<double>& x) {
        return std::vector<double>{2.0*x[0] + x[1] - 2.0, x[0] + 3.0*x[1] - 7.0};
    };
    auto result = broyden_cuda(F, {0.0, 0.0});
    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.x[0], -0.2, 1e-6);
    EXPECT_NEAR(result.x[1], 2.4, 1e-6);
}

TEST(GPUSolver, Broyden_CPU_vs_CUDA) {
    auto F = [](const std::vector<double>& x) {
        return std::vector<double>{2.0*x[0] + x[1] - 2.0, x[0] + 3.0*x[1] - 7.0};
    };
    auto r_cpu = broyden(F, {0.0, 0.0});
    auto r_gpu = broyden_cuda(F, {0.0, 0.0});
    EXPECT_TRUE(r_cpu.converged);
    EXPECT_TRUE(r_gpu.converged);
    for (size_t i = 0; i < r_cpu.x.size(); ++i)
        EXPECT_NEAR(r_cpu.x[i], r_gpu.x[i], 1e-6);
}
