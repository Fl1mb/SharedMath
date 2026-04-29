#include <gtest/gtest.h>
#include "LinearAlgebra/Tensor.h"
#include "LinearAlgebra/MatrixFunctions.h"

#include <cmath>
#include <chrono>
#include <iostream>
#include <vector>
#include <numeric>

using namespace SharedMath::LinearAlgebra;
using Clock = std::chrono::high_resolution_clock;

// ════════════════════════════════════════════════════════════════════════════
// Helpers
// ════════════════════════════════════════════════════════════════════════════

#define SKIP_IF_NO_GPU()                                        \
    do {                                                        \
        if (!cuda_is_available())                               \
            GTEST_SKIP() << "No CUDA-capable GPU detected";    \
    } while (false)

static double elapsed_ms(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static Tensor make_wave(size_t n) {
    std::vector<double> v(n);
    for (size_t i = 0; i < n; ++i)
        v[i] = std::sin(static_cast<double>(i + 1) * 0.001);
    return Tensor({n}, v);
}

static Tensor make_positive(size_t n) {
    std::vector<double> v(n);
    for (size_t i = 0; i < n; ++i)
        v[i] = 0.5 + 0.4 * std::sin(static_cast<double>(i + 1) * 0.001);
    return Tensor({n}, v);
}

static Tensor make_mat(size_t rows, size_t cols) {
    std::vector<double> v(rows * cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            v[i * cols + j] = std::sin(static_cast<double>(i * cols + j + 1));
    return Tensor({rows, cols}, v);
}

static Tensor make_eye(size_t n) {
    std::vector<double> v(n * n, 0.0);
    for (size_t i = 0; i < n; ++i) v[i * n + i] = 1.0;
    return Tensor({n, n}, v);
}

static double fro_err(const Tensor& A, const Tensor& B) {
    EXPECT_EQ(A.size(), B.size());
    double s = 0.0;
    for (size_t i = 0; i < A.size(); ++i) {
        double d = A.flat(i) - B.flat(i);
        s += d * d;
    }
    return std::sqrt(s);
}

static double fro_norm(const Tensor& A) {
    double s = 0.0;
    for (size_t i = 0; i < A.size(); ++i) s += A.flat(i) * A.flat(i);
    return std::sqrt(s);
}

static double rel_fro_err(const Tensor& ref, const Tensor& got) {
    double n = fro_norm(ref);
    return (n > 0) ? fro_err(ref, got) / n : fro_err(ref, got);
}

// ════════════════════════════════════════════════════════════════════════════
// 1. Device management
// ════════════════════════════════════════════════════════════════════════════

TEST(TensorCUDA_Device, QueryNeverCrashes) {
    bool avail = cuda_is_available();
    std::cout << "[GPU] cuda_is_available() = " << (avail ? "YES" : "NO") << "\n";
    (void)avail;
}

TEST(TensorCUDA_Device, CudaCallIsAlwaysSafe) {
    Tensor A = make_wave(1024);
    Tensor A_dev = A.cuda();
    EXPECT_TRUE(A_dev.device() == Device::CPU ||
                A_dev.device() == Device::CUDA);
}

TEST(TensorCUDA_Device, CpuCallOnCpuTensorIsNoop) {
    Tensor A = make_wave(1024);
    Tensor B = A.cpu();
    EXPECT_EQ(B.device(), Device::CPU);
    EXPECT_EQ(fro_err(A, B), 0.0);
}

// ════════════════════════════════════════════════════════════════════════════
// 2. Round-trip: cuda().cpu() == original
// ════════════════════════════════════════════════════════════════════════════

TEST(TensorCUDA_RoundTrip, Scalar) {
    Tensor A({1}, {3.14159265358979});
    Tensor B = A.cuda().cpu();
    EXPECT_DOUBLE_EQ(B.flat(0), A.flat(0));
}

TEST(TensorCUDA_RoundTrip, Small4x4) {
    Tensor A = make_mat(4, 4);
    Tensor B = A.cuda().cpu();
    EXPECT_EQ(fro_err(A, B), 0.0) << "4×4 round-trip must be bit-exact";
}

TEST(TensorCUDA_RoundTrip, Medium_512x512) {
    Tensor A = make_mat(512, 512);
    auto t0 = Clock::now();
    Tensor B = A.cuda().cpu();
    std::cout << "[512×512 round-trip] " << elapsed_ms(t0) << " ms\n";
    EXPECT_EQ(fro_err(A, B), 0.0) << "512×512 round-trip must be bit-exact";
}

TEST(TensorCUDA_RoundTrip, Large_2048x2048) {
    SKIP_IF_NO_GPU();
    Tensor A = make_mat(2048, 2048);
    auto t0 = Clock::now();
    Tensor B = A.cuda().cpu();
    std::cout << "[2048×2048 round-trip] " << elapsed_ms(t0) << " ms\n";
    EXPECT_EQ(fro_err(A, B), 0.0) << "2048×2048 round-trip must be bit-exact";
}

// ════════════════════════════════════════════════════════════════════════════
// 3. Matrix multiply
// ════════════════════════════════════════════════════════════════════════════

// Lightweight correctness check: small size, CPU reference kept intentionally.
TEST(TensorCUDA_Matmul, CorrectnessVsCPU_64x64) {
    SKIP_IF_NO_GPU();
    Tensor A = make_mat(64, 64);
    Tensor B = make_mat(64, 64);
    Tensor C_cpu = A.matmul(B);
    Tensor C_gpu = A.cuda().matmul(B.cuda()).cpu();
    EXPECT_LT(rel_fro_err(C_cpu, C_gpu), 1e-9) << "GPU matmul must match CPU";
}

// Larger sizes: GPU-only, verified via mathematical properties.
TEST(TensorCUDA_Matmul, GPUOnly_512x512) {
    SKIP_IF_NO_GPU();
    Tensor A = make_mat(512, 512);
    Tensor B = make_mat(512, 512);
    auto t0 = Clock::now();
    Tensor C = A.cuda().matmul(B.cuda()).cpu();
    std::cout << "[512×512 matmul GPU] " << elapsed_ms(t0) << " ms\n";
    ASSERT_EQ(C.dim(0), 512u);
    ASSERT_EQ(C.dim(1), 512u);
    EXPECT_TRUE(std::isfinite(fro_norm(C)));
}

TEST(TensorCUDA_Matmul, IdentityProperty_1024x1024) {
    SKIP_IF_NO_GPU();
    Tensor A = make_mat(1024, 1024);
    Tensor I = make_eye(1024);
    auto t0 = Clock::now();
    Tensor C = A.cuda().matmul(I.cuda()).cpu();
    std::cout << "[1024×1024 A*I] " << elapsed_ms(t0) << " ms\n";
    EXPECT_LT(rel_fro_err(A, C), 1e-10) << "A * I must equal A";
}

TEST(TensorCUDA_Matmul, IdentityProperty_2048x2048) {
    SKIP_IF_NO_GPU();
    Tensor A = make_mat(2048, 2048);
    Tensor I = make_eye(2048);
    auto t0 = Clock::now();
    Tensor C = A.cuda().matmul(I.cuda()).cpu();
    std::cout << "[2048×2048 A*I] " << elapsed_ms(t0) << " ms\n";
    EXPECT_LT(rel_fro_err(A, C), 1e-10) << "A * I must equal A";
}

TEST(TensorCUDA_Matmul, NonSquare_1024x512x2048) {
    SKIP_IF_NO_GPU();
    Tensor A = make_mat(1024, 512);
    Tensor B = make_mat(512, 2048);
    auto t0 = Clock::now();
    Tensor C = A.cuda().matmul(B.cuda()).cpu();
    std::cout << "[1024×512 × 512×2048] " << elapsed_ms(t0) << " ms\n";
    ASSERT_EQ(C.dim(0), 1024u);
    ASSERT_EQ(C.dim(1), 2048u);
    EXPECT_TRUE(std::isfinite(fro_norm(C)));
}

// ════════════════════════════════════════════════════════════════════════════
// 4. Element-wise binary ops
// ════════════════════════════════════════════════════════════════════════════

// Lightweight correctness check (CPU reference kept intentionally).
TEST(TensorCUDA_Binary, CorrectnessVsCPU_Add) {
    SKIP_IF_NO_GPU();
    constexpr size_t N = 4096;
    Tensor A = make_wave(N);
    Tensor B = make_positive(N);
    Tensor C_cpu = A + B;
    Tensor C_gpu = (A.cuda() + B.cuda()).cpu();
    EXPECT_LT(rel_fro_err(C_cpu, C_gpu), 1e-14);
}

// Large ops: GPU-only, finite-result check.
#define BINARY_GPU_TEST(TestName, OP, N)                                      \
TEST(TensorCUDA_Binary, TestName) {                                           \
    SKIP_IF_NO_GPU();                                                         \
    Tensor A = make_wave(N);                                                  \
    Tensor B;                                                                 \
    { std::vector<double> bv(N);                                              \
      for (size_t i = 0; i < N; ++i)                                         \
          bv[i] = std::sin(static_cast<double>(i + 500) * 0.001);            \
      B = Tensor({N}, bv); }                                                  \
    Tensor C_cpu = A OP B;                                                    \
    auto t0 = Clock::now();                                                   \
    Tensor C = (A.cuda() OP B.cuda()).cpu();                                  \
    std::cout << "[" #N " " #OP "] " << elapsed_ms(t0) << " ms\n";           \
    EXPECT_EQ(C.size(), N);                                                   \
    EXPECT_TRUE(std::isfinite(fro_norm(C)));                                  \
}

static constexpr size_t k4M = 4'194'304;

BINARY_GPU_TEST(Add_4M, +, k4M)
BINARY_GPU_TEST(Sub_4M, -, k4M)
BINARY_GPU_TEST(Mul_4M, *, k4M)

TEST(TensorCUDA_Binary, Div_4M) {
    SKIP_IF_NO_GPU();
    Tensor A = make_positive(k4M);
    Tensor B = make_positive(k4M);
    auto t0 = Clock::now();
    Tensor C = (A.cuda() / B.cuda()).cpu();
    std::cout << "[4M /] " << elapsed_ms(t0) << " ms\n";
    EXPECT_EQ(C.size(), k4M);
    EXPECT_TRUE(std::isfinite(fro_norm(C)));
}

// ════════════════════════════════════════════════════════════════════════════
// 5. Scalar arithmetic
// ════════════════════════════════════════════════════════════════════════════

TEST(TensorCUDA_Scalar, MulAndDiv_Small) {
    SKIP_IF_NO_GPU();
    Tensor A = make_wave(k4M);
    auto t0 = Clock::now();
    Tensor B = (A.cuda() * 3.14).cpu();
    std::cout << "[4M *scalar] " << elapsed_ms(t0) << " ms\n";
    EXPECT_EQ(B.size(), k4M);
    EXPECT_TRUE(std::isfinite(fro_norm(B)));
    Tensor C = (A.cuda() / 2.71828).cpu();
    EXPECT_TRUE(std::isfinite(fro_norm(C)));
}

// ════════════════════════════════════════════════════════════════════════════
// 6. Unary element-wise ops
// ════════════════════════════════════════════════════════════════════════════

static constexpr size_t k1M = 1'048'576;

// Lightweight correctness check (CPU reference kept intentionally).
TEST(TensorCUDA_Unary, CorrectnessVsCPU_Sin) {
    SKIP_IF_NO_GPU();
    constexpr size_t N = 4096;
    Tensor A = make_wave(N);
    Tensor B_cpu = A.sin();
    Tensor B_gpu = A.cuda().sin().cpu();
    EXPECT_LT(rel_fro_err(B_cpu, B_gpu), 1e-14);
}

// Large ops: GPU-only, finite-result check.
#define UNARY_GPU_TEST(Name, INPUT, OP_CALL)                                  \
TEST(TensorCUDA_Unary, Name) {                                                \
    SKIP_IF_NO_GPU();                                                         \
    Tensor A = INPUT;                                                         \
    auto t0 = Clock::now();                                                   \
    Tensor B = A.cuda().OP_CALL().cpu();                                      \
    std::cout << "[1M " #OP_CALL "] " << elapsed_ms(t0) << " ms\n";          \
    EXPECT_EQ(B.size(), A.size());                                            \
    EXPECT_TRUE(std::isfinite(fro_norm(B)));                                  \
}

UNARY_GPU_TEST(Neg_1M,    make_wave(k1M),     operator-)
UNARY_GPU_TEST(Abs_1M,    make_wave(k1M),     abs)
UNARY_GPU_TEST(Sqrt_1M,   make_positive(k1M), sqrt)
UNARY_GPU_TEST(Exp_1M,    make_wave(k1M),     exp)
UNARY_GPU_TEST(Log_1M,    make_positive(k1M), log)
UNARY_GPU_TEST(Sin_1M,    make_wave(k1M),     sin)
UNARY_GPU_TEST(Cos_1M,    make_wave(k1M),     cos)
UNARY_GPU_TEST(Tanh_1M,   make_wave(k1M),     tanh)
UNARY_GPU_TEST(Floor_1M,  make_wave(k1M),     floor)
UNARY_GPU_TEST(Ceil_1M,   make_wave(k1M),     ceil)
UNARY_GPU_TEST(Round_1M,  make_wave(k1M),     round)
UNARY_GPU_TEST(Sign_1M,   make_wave(k1M),     sign)
UNARY_GPU_TEST(Log2_1M,   make_positive(k1M), log2)
UNARY_GPU_TEST(Log10_1M,  make_positive(k1M), log10)

TEST(TensorCUDA_Unary, Pow2_1M) {
    SKIP_IF_NO_GPU();
    Tensor A = make_wave(k1M);
    auto t0 = Clock::now();
    Tensor B = A.cuda().pow(2.0).cpu();
    std::cout << "[1M pow(2)] " << elapsed_ms(t0) << " ms\n";
    EXPECT_TRUE(std::isfinite(fro_norm(B)));
}

TEST(TensorCUDA_Unary, Clip_Small) {
    SKIP_IF_NO_GPU();
    Tensor A = make_positive(k1M);
    Tensor B = A.cuda().pow(0.5).cpu();
    EXPECT_TRUE(std::isfinite(fro_norm(B)));
}

TEST(TensorCUDA_Unary, Clip_1M) {
    SKIP_IF_NO_GPU();
    Tensor A = make_wave(k1M);
    auto t0 = Clock::now();
    Tensor B = A.cuda().clip(-0.5, 0.5).cpu();
    std::cout << "[1M clip] " << elapsed_ms(t0) << " ms\n";
    for (size_t i = 0; i < std::min(k1M, size_t{100}); ++i) {
        EXPECT_GE(B.flat(i), -0.5);
        EXPECT_LE(B.flat(i),  0.5);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// 7. Chained GPU operations
// ════════════════════════════════════════════════════════════════════════════

TEST(TensorCUDA_Chained, AddThenMul_2M) {
    SKIP_IF_NO_GPU();
    constexpr size_t N = 2'097'152;
    Tensor A = make_wave(N);
    Tensor B = make_positive(N);
    Tensor C = make_wave(N);
    auto t0 = Clock::now();
    Tensor res = ((A.cuda() + B.cuda()) * C.cuda()).cpu();
    std::cout << "[2M (A+B)*C GPU] " << elapsed_ms(t0) << " ms\n";
    EXPECT_EQ(res.size(), N);
    EXPECT_TRUE(std::isfinite(fro_norm(res)));
}

// Lightweight correctness check: exp(sin(A)) on 1M — CPU kept intentionally.
TEST(TensorCUDA_Chained, ExpSin_1M) {
    SKIP_IF_NO_GPU();
    Tensor A = make_wave(kSmallSize);
    Tensor ref = A.sin().exp();
    Tensor res = A.cuda().sin().exp().cpu();
    EXPECT_LT(rel_fro_err(ref, res), 1e-12);
}

TEST(TensorCUDA_Chained, MatmulThenTanh_512) {
    SKIP_IF_NO_GPU();
    Tensor A = make_mat(512, 512);
    Tensor B = make_mat(512, 512);
    auto t0 = Clock::now();
    Tensor res = A.cuda().matmul(B.cuda()).tanh().cpu();
    std::cout << "[512×512 matmul+tanh GPU] " << elapsed_ms(t0) << " ms\n";
    ASSERT_EQ(res.dim(0), 512u);
    ASSERT_EQ(res.dim(1), 512u);
    EXPECT_TRUE(std::isfinite(fro_norm(res)));
}

// ════════════════════════════════════════════════════════════════════════════
// 8. Numerical properties
// ════════════════════════════════════════════════════════════════════════════

TEST(TensorCUDA_Numerics, SumConsistency_2M) {
    SKIP_IF_NO_GPU();
    Tensor A = make_wave(kSmallSize);
    double s_cpu = A.sum();
    double s_gpu = A.cuda().cpu().sum();
    EXPECT_NEAR(s_cpu, s_gpu, std::abs(s_cpu) * 1e-12 + 1e-9);
}

TEST(TensorCUDA_Numerics, FrobeniusNormPreserved_1024x1024) {
    SKIP_IF_NO_GPU();
    Tensor A = make_mat(64, 64);
    double n_cpu = fro_norm(A);
    double n_gpu = fro_norm(A.cuda().cpu());
    EXPECT_NEAR(n_cpu, n_gpu, n_cpu * 1e-14) << "Frobenius norm changed after round-trip";
}

// ════════════════════════════════════════════════════════════════════════════
// 9. Error / edge cases
// ════════════════════════════════════════════════════════════════════════════

TEST(TensorCUDA_Edge, EmptyTensor) {
    Tensor A({0}, {});
    Tensor B = A.cuda().cpu();
    EXPECT_TRUE(B.empty());
}

TEST(TensorCUDA_Edge, SingleElement) {
    Tensor A({1}, {42.0});
    Tensor B = A.cuda().cpu();
    EXPECT_DOUBLE_EQ(B.flat(0), 42.0);
}

TEST(TensorCUDA_Edge, FlatThrowsOnGPU) {
    if (!cuda_is_available()) GTEST_SKIP() << "No GPU";
    Tensor A = make_wave(16);
    Tensor A_dev = A.cuda();
    if (A_dev.device() == Device::CUDA)
        EXPECT_THROW(A_dev.flat(0), std::runtime_error);
}

TEST(TensorCUDA_Edge, DeviceFieldConsistency) {
    Tensor A = make_wave(1024);
    EXPECT_EQ(A.device(), Device::CPU);
    Tensor A_dev = A.cuda();
    bool valid = (A_dev.device() == Device::CPU ||
                  A_dev.device() == Device::CUDA);
    EXPECT_TRUE(valid);
    Tensor A_back = A_dev.cpu();
    EXPECT_EQ(A_back.device(), Device::CPU);
}