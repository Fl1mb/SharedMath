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

// Skip the current test when no CUDA-capable GPU is present.
// Tests that don't skip verify the graceful CPU fallback path.
#define SKIP_IF_NO_GPU()                                        \
    do {                                                        \
        if (!cuda_is_available())                               \
            GTEST_SKIP() << "No CUDA-capable GPU detected";    \
    } while (false)

// Print elapsed ms (shows in --verbose output and helps compare CPU vs GPU).
static double elapsed_ms(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ── Data generators ───────────────────────────────────────────────────────────
// All use deterministic, analytically-known formulas so we can verify results
// without a separate reference RNG.

// Flat 1-D tensor of n elements with values in (-1, 1) via sin.
static Tensor make_wave(size_t n) {
    std::vector<double> v(n);
    for (size_t i = 0; i < n; ++i)
        v[i] = std::sin(static_cast<double>(i + 1) * 0.001);
    return Tensor({n}, v);
}

// Values in (0, 1) — safe for log and sqrt.
static Tensor make_positive(size_t n) {
    std::vector<double> v(n);
    for (size_t i = 0; i < n; ++i)
        v[i] = 0.5 + 0.4 * std::sin(static_cast<double>(i + 1) * 0.001);
    return Tensor({n}, v);
}

// 2-D row-major tensor with values sin(i*cols+j) — range (-1, 1).
static Tensor make_mat(size_t rows, size_t cols) {
    std::vector<double> v(rows * cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            v[i * cols + j] = std::sin(static_cast<double>(i * cols + j + 1));
    return Tensor({rows, cols}, v);
}

// n×n identity tensor.
static Tensor make_eye(size_t n) {
    std::vector<double> v(n * n, 0.0);
    for (size_t i = 0; i < n; ++i) v[i * n + i] = 1.0;
    return Tensor({n, n}, v);
}

// ── Numerical error metrics ───────────────────────────────────────────────────

// Frobenius / L2 error between two same-shape tensors (both on CPU).
static double fro_err(const Tensor& A, const Tensor& B) {
    EXPECT_EQ(A.size(), B.size());
    double s = 0.0;
    for (size_t i = 0; i < A.size(); ++i) {
        double d = A.flat(i) - B.flat(i);
        s += d * d;
    }
    return std::sqrt(s);
}

// Frobenius norm of A.
static double fro_norm(const Tensor& A) {
    double s = 0.0;
    for (size_t i = 0; i < A.size(); ++i) s += A.flat(i) * A.flat(i);
    return std::sqrt(s);
}

// Relative Frobenius error (handles near-zero matrices gracefully).
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
    (void)avail;   // result itself is informational
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

// ════════════════════════════════════════════════════════════════════════════
// 3. Matrix multiply (matmul) - GPU only correctness checks
// ════════════════════════════════════════════════════════════════════════════

// Identity property: A * I == A
TEST(TensorCUDA_Matmul, IdentityProperty_1024x1024) {
    SKIP_IF_NO_GPU();
    Tensor A = make_mat(1024, 1024);
    Tensor I = make_eye(1024);

    auto t0 = Clock::now();
    Tensor C = A.cuda().matmul(I.cuda()).cpu();
    std::cout << "[1024×1024 A*I] " << elapsed_ms(t0) << " ms\n";

    double rel = rel_fro_err(A, C);
    EXPECT_LT(rel, 1e-10) << "A * I must equal A; rel err=" << rel;
}

// Non-square matmul: check that dimensions are correct and result is finite
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
// 4. Element-wise binary ops (GPU correctness vs CPU on small size)
// ════════════════════════════════════════════════════════════════════════════

static constexpr size_t kSmallSize = 16384;  // 16K elements - quick test

#define BINARY_CORRECTNESS_TEST(TestName, OP)                                 \
TEST(TensorCUDA_Binary, TestName) {                                           \
    SKIP_IF_NO_GPU();                                                         \
    Tensor A = make_wave(kSmallSize);                                         \
    Tensor B = make_wave(kSmallSize);                                         \
    Tensor C_cpu = A OP B;                                                    \
    auto t0 = Clock::now();                                                   \
    Tensor C_gpu = (A.cuda() OP B.cuda()).cpu();                              \
    std::cout << "[" #OP "] " << elapsed_ms(t0) << " ms\n";                  \
    double rel = rel_fro_err(C_cpu, C_gpu);                                   \
    EXPECT_LT(rel, 1e-12) << "Relative error=" << rel;                       \
}

BINARY_CORRECTNESS_TEST(Add_Small, +)
BINARY_CORRECTNESS_TEST(Sub_Small, -)
BINARY_CORRECTNESS_TEST(Mul_Small, *)

TEST(TensorCUDA_Binary, Div_Small) {
    SKIP_IF_NO_GPU();
    Tensor A = make_positive(kSmallSize);
    Tensor B = make_positive(kSmallSize);
    Tensor C_cpu = A / B;
    auto t0 = Clock::now();
    Tensor C_gpu = (A.cuda() / B.cuda()).cpu();
    std::cout << "[/] " << elapsed_ms(t0) << " ms\n";
    EXPECT_LT(rel_fro_err(C_cpu, C_gpu), 1e-12);
}

// ════════════════════════════════════════════════════════════════════════════
// 5. Scalar arithmetic
// ════════════════════════════════════════════════════════════════════════════

TEST(TensorCUDA_Scalar, MulAndDiv_Small) {
    SKIP_IF_NO_GPU();
    Tensor A = make_wave(kSmallSize);

    Tensor B_cpu = A * 3.14;
    auto t0 = Clock::now();
    Tensor B_gpu = (A.cuda() * 3.14).cpu();
    std::cout << "[*scalar] " << elapsed_ms(t0) << " ms\n";
    EXPECT_LT(rel_fro_err(B_cpu, B_gpu), 1e-13);

    Tensor C_cpu = A / 2.71828;
    Tensor C_gpu = (A.cuda() / 2.71828).cpu();
    EXPECT_LT(rel_fro_err(C_cpu, C_gpu), 1e-13);
}

// ════════════════════════════════════════════════════════════════════════════
// 6. Unary element-wise ops (small size)
// ════════════════════════════════════════════════════════════════════════════

#define UNARY_TEST(Name, INPUT, OP_CALL, TOL)                                 \
TEST(TensorCUDA_Unary, Name) {                                                \
    SKIP_IF_NO_GPU();                                                         \
    Tensor A = INPUT;                                                         \
    Tensor B_cpu = A.OP_CALL();                                               \
    auto t0 = Clock::now();                                                   \
    Tensor B_gpu = A.cuda().OP_CALL().cpu();                                  \
    std::cout << "[" #OP_CALL "] " << elapsed_ms(t0) << " ms\n";              \
    EXPECT_LT(rel_fro_err(B_cpu, B_gpu), TOL);                               \
}

UNARY_TEST(Neg_Small,    make_wave(kSmallSize), operator-, 1e-14)
UNARY_TEST(Abs_Small,    make_wave(kSmallSize), abs,       1e-14)
UNARY_TEST(Sqrt_Small,   make_positive(kSmallSize), sqrt,  1e-14)
UNARY_TEST(Exp_Small,    make_wave(kSmallSize), exp,       1e-13)
UNARY_TEST(Log_Small,    make_positive(kSmallSize), log,   1e-13)
UNARY_TEST(Sin_Small,    make_wave(kSmallSize), sin,       1e-14)
UNARY_TEST(Cos_Small,    make_wave(kSmallSize), cos,       1e-14)
UNARY_TEST(Tanh_Small,   make_wave(kSmallSize), tanh,      1e-14)
UNARY_TEST(Floor_Small,  make_wave(kSmallSize), floor,     1e-14)
UNARY_TEST(Ceil_Small,   make_wave(kSmallSize), ceil,      1e-14)
UNARY_TEST(Round_Small,  make_wave(kSmallSize), round,     1e-14)
UNARY_TEST(Sign_Small,   make_wave(kSmallSize), sign,      1e-14)

TEST(TensorCUDA_Unary, Pow2_Small) {
    SKIP_IF_NO_GPU();
    Tensor A = make_wave(kSmallSize);
    Tensor B_cpu = A.pow(2.0);
    auto t0 = Clock::now();
    Tensor B_gpu = A.cuda().pow(2.0).cpu();
    std::cout << "[pow(2)] " << elapsed_ms(t0) << " ms\n";
    EXPECT_LT(rel_fro_err(B_cpu, B_gpu), 1e-13);
}

TEST(TensorCUDA_Unary, Clip_Small) {
    SKIP_IF_NO_GPU();
    Tensor A = make_wave(kSmallSize);
    Tensor B_cpu = A.clip(-0.5, 0.5);
    auto t0 = Clock::now();
    Tensor B_gpu = A.cuda().clip(-0.5, 0.5).cpu();
    std::cout << "[clip] " << elapsed_ms(t0) << " ms\n";
    EXPECT_LT(rel_fro_err(B_cpu, B_gpu), 1e-14);
    // Bounds check: every element in [-0.5, 0.5]
    for (size_t i = 0; i < std::min(kSmallSize, size_t{100}); ++i) {
        EXPECT_GE(B_gpu.flat(i), -0.5);
        EXPECT_LE(B_gpu.flat(i),  0.5);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// 7. Chained GPU operations (no intermediate round-trips to CPU)
// ════════════════════════════════════════════════════════════════════════════

TEST(TensorCUDA_Chained, AddThenMul_Small) {
    SKIP_IF_NO_GPU();
    constexpr size_t N = 32768;
    Tensor A = make_wave(N);
    Tensor B = make_positive(N);
    Tensor C = make_wave(N);

    Tensor ref = (A + B) * C;   // CPU

    auto t0 = Clock::now();
    Tensor gpu = (A.cuda() + B.cuda()) * C.cuda();   // stays on GPU
    Tensor res = gpu.cpu();
    std::cout << "[(A+B)*C GPU] " << elapsed_ms(t0) << " ms\n";

    EXPECT_LT(rel_fro_err(ref, res), 1e-13);
}

TEST(TensorCUDA_Chained, ExpSin_Small) {
    SKIP_IF_NO_GPU();
    Tensor A = make_wave(kSmallSize);
    Tensor ref = A.sin().exp();
    Tensor res = A.cuda().sin().exp().cpu();
    EXPECT_LT(rel_fro_err(ref, res), 1e-12);
}

// Matmul followed by element-wise op: (A*B).tanh()
TEST(TensorCUDA_Chained, MatmulThenTanh_Small) {
    SKIP_IF_NO_GPU();
    constexpr size_t N = 128;  // Small size for quick test
    Tensor A = make_mat(N, N);
    Tensor B = make_mat(N, N);
    Tensor ref = A.matmul(B).tanh();

    auto t0 = Clock::now();
    Tensor res = A.cuda().matmul(B.cuda()).tanh().cpu();
    std::cout << "[" << N << "×" << N << " matmul+tanh] " << elapsed_ms(t0) << " ms\n";

    EXPECT_LT(rel_fro_err(ref, res), 1e-9);
}

// ════════════════════════════════════════════════════════════════════════════
// 8. Numerical properties
// ════════════════════════════════════════════════════════════════════════════

// GPU sum should match CPU sum to relative machine-epsilon scale.
TEST(TensorCUDA_Numerics, SumConsistency_Small) {
    SKIP_IF_NO_GPU();
    Tensor A = make_wave(kSmallSize);
    double s_cpu = A.sum();
    double s_gpu = A.cuda().cpu().sum();   // round-trip then sum
    EXPECT_NEAR(s_cpu, s_gpu, std::abs(s_cpu) * 1e-12 + 1e-9);
}

// ||A||_F should be preserved by the GPU round-trip.
TEST(TensorCUDA_Numerics, FrobeniusNormPreserved_Small) {
    SKIP_IF_NO_GPU();
    Tensor A = make_mat(64, 64);
    double n_cpu = fro_norm(A);
    double n_gpu = fro_norm(A.cuda().cpu());
    EXPECT_NEAR(n_cpu, n_gpu, n_cpu * 1e-14) << "Frobenius norm changed after round-trip";
}

// ════════════════════════════════════════════════════════════════════════════
// 9. Error / edge cases
// ════════════════════════════════════════════════════════════════════════════

// Empty tensor: cuda().cpu() must return an empty tensor.
TEST(TensorCUDA_Edge, EmptyTensor) {
    Tensor A({0}, {});
    Tensor B = A.cuda().cpu();
    EXPECT_TRUE(B.empty());
}

// Single element.
TEST(TensorCUDA_Edge, SingleElement) {
    Tensor A({1}, {42.0});
    Tensor B = A.cuda().cpu();
    EXPECT_DOUBLE_EQ(B.flat(0), 42.0);
}

// flat() must throw on a GPU tensor.
TEST(TensorCUDA_Edge, FlatThrowsOnGPU) {
    if (!cuda_is_available()) GTEST_SKIP() << "No GPU";
    Tensor A = make_wave(16);
    Tensor A_dev = A.cuda();
    if (A_dev.device() == Device::CUDA)
        EXPECT_THROW(A_dev.flat(0), std::runtime_error);
}

// device() is correct in all states.
TEST(TensorCUDA_Edge, DeviceFieldConsistency) {
    Tensor A = make_wave(1024);
    EXPECT_EQ(A.device(), Device::CPU);

    Tensor A_dev = A.cuda();
    // Either CPU (no GPU) or CUDA (GPU present)
    bool valid = (A_dev.device() == Device::CPU ||
                  A_dev.device() == Device::CUDA);
    EXPECT_TRUE(valid);

    Tensor A_back = A_dev.cpu();
    EXPECT_EQ(A_back.device(), Device::CPU);
}