#include <gtest/gtest.h>
#include "LinearAlgebra/DynamicMatrix.h"
#include "LinearAlgebra/MatrixFunctions.h"

#include <cmath>
#include <chrono>
#include <iostream>
#include <vector>

using namespace SharedMath::LinearAlgebra;
using Clock = std::chrono::high_resolution_clock;

// ════════════════════════════════════════════════════════════════════════════
// Helpers
// ════════════════════════════════════════════════════════════════════════════

#define SKIP_IF_NO_GPU()                                         \
    do {                                                         \
        if (!cuda_is_available())                                \
            GTEST_SKIP() << "No CUDA-capable GPU detected";     \
    } while (false)

static double elapsed_ms(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ── Data generators ───────────────────────────────────────────────────────────

// Deterministic pseudo-random values in (-1, 1) via sin.
static DynamicMatrix make_wave(size_t rows, size_t cols) {
    DynamicMatrix A(rows, cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            A(i, j) = std::sin(static_cast<double>(i * cols + j + 1) * 0.001);
    return A;
}

// Values in (0.1, 1) — safe for division and log.
static DynamicMatrix make_positive(size_t rows, size_t cols) {
    DynamicMatrix A(rows, cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            A(i, j) = 0.55 + 0.45 * std::sin(static_cast<double>(i * cols + j + 1) * 0.001);
    return A;
}

// n×n identity matrix.
static DynamicMatrix make_eye_dyn(size_t n) {
    DynamicMatrix I(n, n, 0.0);
    for (size_t i = 0; i < n; ++i) I(i, i) = 1.0;
    return I;
}

// ── Numerical error metrics ───────────────────────────────────────────────────

static double fro_norm(const DynamicMatrix& A) {
    double s = 0.0;
    for (size_t i = 0; i < A.rows(); ++i)
        for (size_t j = 0; j < A.cols(); ++j)
            s += A(i, j) * A(i, j);
    return std::sqrt(s);
}

static double fro_err(const DynamicMatrix& ref, const DynamicMatrix& got) {
    EXPECT_EQ(ref.rows(), got.rows());
    EXPECT_EQ(ref.cols(), got.cols());
    double s = 0.0;
    for (size_t i = 0; i < ref.rows(); ++i)
        for (size_t j = 0; j < ref.cols(); ++j) {
            double d = ref(i, j) - got(i, j);
            s += d * d;
        }
    return std::sqrt(s);
}

static double rel_fro_err(const DynamicMatrix& ref, const DynamicMatrix& got) {
    double n = fro_norm(ref);
    return (n > 0.0) ? fro_err(ref, got) / n : fro_err(ref, got);
}

// ════════════════════════════════════════════════════════════════════════════
// 1. Device management
// ════════════════════════════════════════════════════════════════════════════

TEST(DynMatCUDA_Device, QueryNeverCrashes) {
    bool avail = cuda_is_available();
    std::cout << "[GPU] cuda_is_available() = " << (avail ? "YES" : "NO") << "\n";
    (void)avail;
}

TEST(DynMatCUDA_Device, CudaIsAlwaysSafe) {
    DynamicMatrix A = make_wave(32, 32);
    DynamicMatrix A_dev = A.cuda();
    EXPECT_TRUE(A_dev.device() == Device::CPU ||
                A_dev.device() == Device::CUDA);
}

TEST(DynMatCUDA_Device, CpuOnCpuMatrixIsNoop) {
    DynamicMatrix A = make_wave(32, 32);
    DynamicMatrix B = A.cpu();
    EXPECT_EQ(B.device(), Device::CPU);
    EXPECT_EQ(fro_err(A, B), 0.0);
}

TEST(DynMatCUDA_Device, DeviceFieldLifecycle) {
    DynamicMatrix A = make_wave(64, 64);
    EXPECT_EQ(A.device(), Device::CPU);

    DynamicMatrix A_d = A.cuda();
    bool ok = (A_d.device() == Device::CPU || A_d.device() == Device::CUDA);
    EXPECT_TRUE(ok);

    DynamicMatrix A_back = A_d.cpu();
    EXPECT_EQ(A_back.device(), Device::CPU);
}

// ════════════════════════════════════════════════════════════════════════════
// 2. Round-trip: cuda().cpu() is bit-exact
// ════════════════════════════════════════════════════════════════════════════

TEST(DynMatCUDA_RoundTrip, Small_4x4) {
    DynamicMatrix A = make_wave(4, 4);
    DynamicMatrix B = A.cuda().cpu();
    ASSERT_EQ(B.rows(), 4u);
    ASSERT_EQ(B.cols(), 4u);
    EXPECT_EQ(fro_err(A, B), 0.0);
}

// 256×256 = 65 536 doubles = 512 KB
TEST(DynMatCUDA_RoundTrip, Medium_256x256) {
    DynamicMatrix A = make_wave(256, 256);
    auto t0 = Clock::now();
    DynamicMatrix B = A.cuda().cpu();
    std::cout << "[256×256 round-trip] " << elapsed_ms(t0) << " ms\n";
    EXPECT_EQ(fro_err(A, B), 0.0);
}

// 1024×1024 = 1 048 576 doubles = 8 MB
TEST(DynMatCUDA_RoundTrip, Large_1024x1024) {
    DynamicMatrix A = make_wave(1024, 1024);
    auto t0 = Clock::now();
    DynamicMatrix B = A.cuda().cpu();
    std::cout << "[1024×1024 round-trip] " << elapsed_ms(t0) << " ms\n";
    EXPECT_EQ(fro_err(A, B), 0.0) << "Must be bit-exact after H→D→H";
}

// 2048×2048 = 4 194 304 doubles = 32 MB — exercises PCIe seriously
TEST(DynMatCUDA_RoundTrip, Huge_2048x2048) {
    SKIP_IF_NO_GPU();
    DynamicMatrix A = make_wave(2048, 2048);
    auto t0 = Clock::now();
    DynamicMatrix B = A.cuda().cpu();
    std::cout << "[2048×2048 round-trip] " << elapsed_ms(t0) << " ms\n";
    EXPECT_EQ(fro_err(A, B), 0.0) << "Must be bit-exact after H→D→H";
}

// ════════════════════════════════════════════════════════════════════════════
// 3. Matrix multiply (operator*)
// ════════════════════════════════════════════════════════════════════════════

// 128×128 — tiny, full element-wise comparison
TEST(DynMatCUDA_Matmul, CorrectnessVsCPU_128x128) {
    SKIP_IF_NO_GPU();
    DynamicMatrix A = make_wave(128, 128);
    DynamicMatrix B = make_wave(128, 128);

    DynamicMatrix C_cpu = A * B;

    auto t0 = Clock::now();
    DynamicMatrix C_gpu = (A.cuda() * B.cuda()).cpu();
    std::cout << "[128×128 matmul GPU] " << elapsed_ms(t0) << " ms\n";

    double rel = rel_fro_err(C_cpu, C_gpu);
    EXPECT_LT(rel, 1e-9) << "Relative Frobenius error=" << rel;
}

// 512×512 — GPU-only performance test (correctness already covered by 128×128 above)
TEST(DynMatCUDA_Matmul, GPUOnly_512x512) {
    SKIP_IF_NO_GPU();
    DynamicMatrix A = make_wave(512, 512);
    DynamicMatrix B = make_wave(512, 512);

    auto t0 = Clock::now();
    DynamicMatrix C_gpu = (A.cuda() * B.cuda()).cpu();
    std::cout << "[512×512 matmul GPU] " << elapsed_ms(t0) << " ms\n";

    ASSERT_EQ(C_gpu.rows(), 512u);
    ASSERT_EQ(C_gpu.cols(), 512u);
    double nrm = fro_norm(C_gpu);
    EXPECT_TRUE(std::isfinite(nrm)) << "GPU matmul result is not finite";
    EXPECT_GT(nrm, 0.0) << "GPU matmul result is zero";
}

// 1024×1024 via A*I=A (avoids slow CPU matmul reference)
TEST(DynMatCUDA_Matmul, IdentityProperty_1024x1024) {
    SKIP_IF_NO_GPU();
    DynamicMatrix A = make_wave(1024, 1024);
    DynamicMatrix I = make_eye_dyn(1024);

    auto t0 = Clock::now();
    DynamicMatrix C = (A.cuda() * I.cuda()).cpu();
    std::cout << "[1024×1024 A*I GPU] " << elapsed_ms(t0) << " ms\n";

    double rel = rel_fro_err(A, C);
    EXPECT_LT(rel, 1e-10) << "A * I must equal A; rel err=" << rel;
}

// 2048×2048 via identity — the heavyweight
TEST(DynMatCUDA_Matmul, IdentityProperty_2048x2048) {
    SKIP_IF_NO_GPU();
    DynamicMatrix A = make_wave(2048, 2048);
    DynamicMatrix I = make_eye_dyn(2048);

    auto t0 = Clock::now();
    DynamicMatrix C = (A.cuda() * I.cuda()).cpu();
    std::cout << "[2048×2048 A*I GPU] " << elapsed_ms(t0) << " ms\n";

    double rel = rel_fro_err(A, C);
    EXPECT_LT(rel, 1e-10) << "A * I must equal A; rel err=" << rel;
}

// Non-square: (512×1024) * (1024×512) = (512×512)
// No CPU reference — the inner dimension 1024 makes CPU matmul very slow.
TEST(DynMatCUDA_Matmul, NonSquare_512x1024x512) {
    SKIP_IF_NO_GPU();
    DynamicMatrix A = make_wave(512, 1024);
    DynamicMatrix B = make_wave(1024, 512);

    auto t0 = Clock::now();
    DynamicMatrix C_gpu = (A.cuda() * B.cuda()).cpu();
    std::cout << "[512×1024 × 1024×512 GPU] " << elapsed_ms(t0) << " ms\n";

    ASSERT_EQ(C_gpu.rows(), 512u);
    ASSERT_EQ(C_gpu.cols(), 512u);
    double nrm = fro_norm(C_gpu);
    EXPECT_TRUE(std::isfinite(nrm)) << "GPU matmul result is not finite";
    EXPECT_GT(nrm, 0.0);
}

// Tall × fat: (2048×64) * (64×2048)
TEST(DynMatCUDA_Matmul, TallFat_2048x64x2048) {
    SKIP_IF_NO_GPU();
    DynamicMatrix A = make_wave(2048, 64);
    DynamicMatrix B = make_wave(64, 2048);

    DynamicMatrix C_cpu = A * B;   // fast on CPU (n_inner=64)
    auto t0 = Clock::now();
    DynamicMatrix C_gpu = (A.cuda() * B.cuda()).cpu();
    std::cout << "[2048×64 × 64×2048 GPU] " << elapsed_ms(t0) << " ms\n";

    ASSERT_EQ(C_gpu.rows(), 2048u);
    ASSERT_EQ(C_gpu.cols(), 2048u);
    EXPECT_LT(rel_fro_err(C_cpu, C_gpu), 1e-9);
}

// ════════════════════════════════════════════════════════════════════════════
// 4. Element-wise add / subtract
// ════════════════════════════════════════════════════════════════════════════

TEST(DynMatCUDA_Binary, Add_1024x1024) {
    SKIP_IF_NO_GPU();
    DynamicMatrix A = make_wave(1024, 1024);
    DynamicMatrix B = make_positive(1024, 1024);

    DynamicMatrix C_cpu = A + B;
    auto t0 = Clock::now();
    DynamicMatrix C_gpu = (A.cuda() + B.cuda()).cpu();
    std::cout << "[1024×1024 add GPU] " << elapsed_ms(t0) << " ms\n";

    EXPECT_LT(rel_fro_err(C_cpu, C_gpu), 1e-13);
}

TEST(DynMatCUDA_Binary, Sub_1024x1024) {
    SKIP_IF_NO_GPU();
    DynamicMatrix A = make_wave(1024, 1024);
    DynamicMatrix B = make_positive(1024, 1024);

    DynamicMatrix C_cpu = A - B;
    auto t0 = Clock::now();
    DynamicMatrix C_gpu = (A.cuda() - B.cuda()).cpu();
    std::cout << "[1024×1024 sub GPU] " << elapsed_ms(t0) << " ms\n";

    EXPECT_LT(rel_fro_err(C_cpu, C_gpu), 1e-13);
}

TEST(DynMatCUDA_Binary, Add_2048x2048) {
    SKIP_IF_NO_GPU();
    DynamicMatrix A = make_wave(2048, 2048);
    DynamicMatrix B = make_wave(2048, 2048);

    DynamicMatrix C_cpu = A + B;
    auto t0 = Clock::now();
    DynamicMatrix C_gpu = (A.cuda() + B.cuda()).cpu();
    std::cout << "[2048×2048 add GPU] " << elapsed_ms(t0) << " ms\n";

    EXPECT_LT(rel_fro_err(C_cpu, C_gpu), 1e-13);
}

TEST(DynMatCUDA_Binary, Sub_2048x2048) {
    SKIP_IF_NO_GPU();
    DynamicMatrix A = make_wave(2048, 2048);
    DynamicMatrix B = make_wave(2048, 2048);

    DynamicMatrix C_cpu = A - B;
    auto t0 = Clock::now();
    DynamicMatrix C_gpu = (A.cuda() - B.cuda()).cpu();
    std::cout << "[2048×2048 sub GPU] " << elapsed_ms(t0) << " ms\n";

    EXPECT_LT(rel_fro_err(C_cpu, C_gpu), 1e-13);
}

// ════════════════════════════════════════════════════════════════════════════
// 5. In-place operators (+=, -=)
// ════════════════════════════════════════════════════════════════════════════

TEST(DynMatCUDA_InPlace, AddAssign_1024x1024) {
    SKIP_IF_NO_GPU();
    DynamicMatrix A = make_wave(1024, 1024);
    DynamicMatrix B = make_positive(1024, 1024);

    // CPU reference
    DynamicMatrix R_cpu = A;
    R_cpu += B;

    // GPU
    DynamicMatrix A_d = A.cuda();
    DynamicMatrix B_d = B.cuda();
    auto t0 = Clock::now();
    A_d += B_d;
    DynamicMatrix R_gpu = A_d.cpu();
    std::cout << "[1024×1024 += GPU] " << elapsed_ms(t0) << " ms\n";

    EXPECT_LT(rel_fro_err(R_cpu, R_gpu), 1e-13);
}

TEST(DynMatCUDA_InPlace, SubAssign_1024x1024) {
    SKIP_IF_NO_GPU();
    DynamicMatrix A = make_wave(1024, 1024);
    DynamicMatrix B = make_positive(1024, 1024);

    DynamicMatrix R_cpu = A;
    R_cpu -= B;

    DynamicMatrix A_d = A.cuda();
    DynamicMatrix B_d = B.cuda();
    auto t0 = Clock::now();
    A_d -= B_d;
    DynamicMatrix R_gpu = A_d.cpu();
    std::cout << "[1024×1024 -= GPU] " << elapsed_ms(t0) << " ms\n";

    EXPECT_LT(rel_fro_err(R_cpu, R_gpu), 1e-13);
}

// ════════════════════════════════════════════════════════════════════════════
// 6. Scalar multiply / divide
// ════════════════════════════════════════════════════════════════════════════

TEST(DynMatCUDA_Scalar, Mul_2048x2048) {
    SKIP_IF_NO_GPU();
    DynamicMatrix A = make_wave(2048, 2048);
    const double s = 3.14159265358979;

    DynamicMatrix B_cpu = A * s;
    auto t0 = Clock::now();
    DynamicMatrix B_gpu = (A.cuda() * s).cpu();
    std::cout << "[2048×2048 *scalar GPU] " << elapsed_ms(t0) << " ms\n";

    EXPECT_LT(rel_fro_err(B_cpu, B_gpu), 1e-14);
}

TEST(DynMatCUDA_Scalar, Div_2048x2048) {
    SKIP_IF_NO_GPU();
    DynamicMatrix A = make_wave(2048, 2048);
    const double s = 2.71828182845905;

    DynamicMatrix B_cpu = A / s;
    auto t0 = Clock::now();
    DynamicMatrix B_gpu = (A.cuda() / s).cpu();
    std::cout << "[2048×2048 /scalar GPU] " << elapsed_ms(t0) << " ms\n";

    EXPECT_LT(rel_fro_err(B_cpu, B_gpu), 1e-14);
}

TEST(DynMatCUDA_Scalar, MulAssign_1024x1024) {
    SKIP_IF_NO_GPU();
    DynamicMatrix A = make_wave(1024, 1024);
    const double s = -2.5;

    DynamicMatrix R_cpu = A;
    R_cpu *= s;

    DynamicMatrix A_d = A.cuda();
    auto t0 = Clock::now();
    A_d *= s;
    DynamicMatrix R_gpu = A_d.cpu();
    std::cout << "[1024×1024 *=scalar GPU] " << elapsed_ms(t0) << " ms\n";

    EXPECT_LT(rel_fro_err(R_cpu, R_gpu), 1e-14);
}

TEST(DynMatCUDA_Scalar, DivAssign_1024x1024) {
    SKIP_IF_NO_GPU();
    DynamicMatrix A = make_positive(1024, 1024);
    const double s = 0.125;

    DynamicMatrix R_cpu = A;
    R_cpu /= s;

    DynamicMatrix A_d = A.cuda();
    auto t0 = Clock::now();
    A_d /= s;
    DynamicMatrix R_gpu = A_d.cpu();
    std::cout << "[1024×1024 /=scalar GPU] " << elapsed_ms(t0) << " ms\n";

    EXPECT_LT(rel_fro_err(R_cpu, R_gpu), 1e-13);
}

// left-scalar: s * A
TEST(DynMatCUDA_Scalar, LeftScalarMul_1024x1024) {
    SKIP_IF_NO_GPU();
    DynamicMatrix A = make_wave(1024, 1024);
    DynamicMatrix B_cpu = 5.0 * A;
    DynamicMatrix B_gpu = (5.0 * A.cuda()).cpu();
    EXPECT_LT(rel_fro_err(B_cpu, B_gpu), 1e-14);
}

// ════════════════════════════════════════════════════════════════════════════
// 7. Chained GPU operations
// ════════════════════════════════════════════════════════════════════════════

// (A + B) * 2  — one binary then one scalar, fully on GPU
TEST(DynMatCUDA_Chained, AddThenScale_2048x2048) {
    SKIP_IF_NO_GPU();
    DynamicMatrix A = make_wave(2048, 2048);
    DynamicMatrix B = make_positive(2048, 2048);

    DynamicMatrix R_cpu = (A + B) * 2.0;

    auto t0 = Clock::now();
    DynamicMatrix R_gpu = ((A.cuda() + B.cuda()) * 2.0).cpu();
    std::cout << "[(A+B)*2 2048×2048 GPU] " << elapsed_ms(t0) << " ms\n";

    EXPECT_LT(rel_fro_err(R_cpu, R_gpu), 1e-13);
}

// A*B + C  — matmul then add, fully on GPU (no CPU matmul reference)
TEST(DynMatCUDA_Chained, MatmulPlusAdd_512x512) {
    SKIP_IF_NO_GPU();
    DynamicMatrix A = make_wave(512, 512);
    DynamicMatrix B = make_wave(512, 512);
    DynamicMatrix C = make_positive(512, 512);

    auto t0 = Clock::now();
    DynamicMatrix R_gpu = (A.cuda() * B.cuda() + C.cuda()).cpu();
    std::cout << "[A*B+C 512×512 GPU] " << elapsed_ms(t0) << " ms\n";

    ASSERT_EQ(R_gpu.rows(), 512u);
    ASSERT_EQ(R_gpu.cols(), 512u);
    double nrm = fro_norm(R_gpu);
    EXPECT_TRUE(std::isfinite(nrm)) << "Chained GPU result is not finite";
    EXPECT_GT(nrm, 0.0);
}

// A*B - A*B == 0  (numerical zero check)
TEST(DynMatCUDA_Chained, MatmulMinusSelf_256x256) {
    SKIP_IF_NO_GPU();
    DynamicMatrix A = make_wave(256, 256);
    DynamicMatrix B = make_wave(256, 256);

    DynamicMatrix A_d = A.cuda();
    DynamicMatrix B_d = B.cuda();

    DynamicMatrix C = A_d * B_d;
    DynamicMatrix Z = (C - C).cpu();   // must be exactly zero

    EXPECT_EQ(fro_norm(Z), 0.0) << "C - C on GPU must be exactly zero";
}

// ════════════════════════════════════════════════════════════════════════════
// 8. Error handling
// ════════════════════════════════════════════════════════════════════════════

TEST(DynMatCUDA_Errors, MixedDeviceAddThrows) {
    if (!cuda_is_available()) GTEST_SKIP() << "No GPU";
    DynamicMatrix A = make_wave(64, 64);
    DynamicMatrix B_gpu = A.cuda();
    if (B_gpu.device() == Device::CUDA)
        EXPECT_THROW(A + B_gpu, std::invalid_argument);
}

TEST(DynMatCUDA_Errors, MixedDeviceSubThrows) {
    if (!cuda_is_available()) GTEST_SKIP() << "No GPU";
    DynamicMatrix A = make_wave(64, 64);
    DynamicMatrix B_gpu = A.cuda();
    if (B_gpu.device() == Device::CUDA)
        EXPECT_THROW(A - B_gpu, std::invalid_argument);
}

TEST(DynMatCUDA_Errors, MixedDeviceMulThrows) {
    if (!cuda_is_available()) GTEST_SKIP() << "No GPU";
    DynamicMatrix A = make_wave(64, 64);
    DynamicMatrix B_gpu = A.cuda();
    if (B_gpu.device() == Device::CUDA)
        EXPECT_THROW(A * B_gpu, std::invalid_argument);
}

TEST(DynMatCUDA_Errors, GetOnGPUThrows) {
    if (!cuda_is_available()) GTEST_SKIP() << "No GPU";
    DynamicMatrix A = make_wave(64, 64);
    DynamicMatrix A_d = A.cuda();
    if (A_d.device() == Device::CUDA)
        EXPECT_THROW(A_d.get(0, 0), std::runtime_error);
}

TEST(DynMatCUDA_Errors, SetOnGPUThrows) {
    if (!cuda_is_available()) GTEST_SKIP() << "No GPU";
    DynamicMatrix A = make_wave(64, 64);
    DynamicMatrix A_d = A.cuda();
    if (A_d.device() == Device::CUDA)
        EXPECT_THROW(A_d.set(0, 0, 1.0), std::runtime_error);
}

TEST(DynMatCUDA_Errors, TransposedOnGPUThrows) {
    if (!cuda_is_available()) GTEST_SKIP() << "No GPU";
    DynamicMatrix A = make_wave(64, 64);
    DynamicMatrix A_d = A.cuda();
    if (A_d.device() == Device::CUDA)
        EXPECT_THROW(A_d.transposed(), std::runtime_error);
}

TEST(DynMatCUDA_Errors, ShapeMismatchThrowsOnGPU) {
    if (!cuda_is_available()) GTEST_SKIP() << "No GPU";
    DynamicMatrix A = make_wave(128, 64);
    DynamicMatrix B = make_wave(64, 128);  // different shape
    DynamicMatrix A_d = A.cuda();
    DynamicMatrix B_d = B.cuda();
    EXPECT_THROW(A_d + B_d, std::invalid_argument);
}

TEST(DynMatCUDA_Errors, InnerDimMismatchThrowsOnGPU) {
    if (!cuda_is_available()) GTEST_SKIP() << "No GPU";
    DynamicMatrix A = make_wave(64, 32);
    DynamicMatrix B = make_wave(64, 64);  // inner dim mismatch
    EXPECT_THROW(A.cuda() * B.cuda(), std::invalid_argument);
}

// ════════════════════════════════════════════════════════════════════════════
// 9. Numerical stress — large random-ish workloads
// ════════════════════════════════════════════════════════════════════════════

// Verify ||A||_F is preserved by the GPU round-trip
TEST(DynMatCUDA_Numerics, FrobeniusNormPreserved_2048) {
    SKIP_IF_NO_GPU();
    DynamicMatrix A = make_wave(2048, 2048);
    double n_cpu = fro_norm(A);
    double n_gpu = fro_norm(A.cuda().cpu());
    EXPECT_NEAR(n_cpu, n_gpu, n_cpu * 1e-14)
        << "Frobenius norm changed after GPU round-trip";
}

// Repeated += does not accumulate error beyond expected floating-point bounds
TEST(DynMatCUDA_Numerics, RepeatedAddAccumulation_512x512) {
    SKIP_IF_NO_GPU();
    DynamicMatrix A = make_wave(512, 512);
    DynamicMatrix B = make_positive(512, 512);

    DynamicMatrix R_cpu = A;
    DynamicMatrix R_gpu = A.cuda();

    for (int k = 0; k < 16; ++k) {
        R_cpu += B;
        DynamicMatrix B_d = B.cuda();
        R_gpu += B_d;
    }

    DynamicMatrix R_gpu_back = R_gpu.cpu();
    // 16 iterations → expect at most 16× machine epsilon error per element
    EXPECT_LT(rel_fro_err(R_cpu, R_gpu_back), 1e-12);
}

// A*I_n*A should equal A*A if the matmul is accurate
TEST(DynMatCUDA_Numerics, MatmulAssociativity_256) {
    SKIP_IF_NO_GPU();
    DynamicMatrix A = make_wave(256, 256);
    DynamicMatrix I = make_eye_dyn(256);

    // CPU: A*(I*A) = A*A
    DynamicMatrix C1_cpu = A * (I * A);
    DynamicMatrix C2_cpu = A * A;

    // GPU
    DynamicMatrix A_d = A.cuda(), I_d = I.cuda();
    DynamicMatrix C1_gpu = (A_d * (I_d * A_d)).cpu();
    DynamicMatrix C2_gpu = (A_d * A_d).cpu();

    // GPU vs CPU
    EXPECT_LT(rel_fro_err(C1_cpu, C1_gpu), 1e-9);
    EXPECT_LT(rel_fro_err(C2_cpu, C2_gpu), 1e-9);
    // Associativity: A*(I*A) == A*A
    EXPECT_LT(rel_fro_err(C1_gpu, C2_gpu), 1e-9);
}
