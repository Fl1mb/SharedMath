#pragma once

/**
 * @file NumericalMethodsGPU.h
 * @brief GPU-accelerated vector and matrix types for numerical methods.
 *
 * @defgroup NumericalMethods_GPU GPU Acceleration
 * @ingroup NumericalMethods
 *
 * Provides GPUNumVec and GPUNumMat — thin wrappers around device memory
 * that support BLAS-1/BLAS-2 operations via cuBLAS and dense linear
 * solves via cuSOLVER.  When CUDA is not compiled in, all methods fall
 * back to CPU implementations using std::vector.
 *
 * Usage:
 * @code
 *   GPUNumVec v({1.0, 2.0, 3.0});   // from host
 *   auto vg = v.toGPU();             // H→D copy
 *   auto dot = GPUNumVec::dot(vg, vg);  // GPU dot product
 *   auto vc = vg.toCPU();            // D→H copy
 * @endcode
 */

#include <vector>
#include <memory>
#include <cstddef>
#include <sharedmath_numericalmethods_export.h>
#include "LinearAlgebra/Tensor.h"  // Device enum

namespace SharedMath::NumericalMethods {

using SharedMath::LinearAlgebra::Device;

/// GPU-backed dense vector (contiguous double* on device).
///
/// CPU fallback: when CUDA is unavailable, operations execute on host
/// using std::vector<double>. The API is identical in both cases.
class SHAREDMATH_NUMERICALMETHODS_EXPORT GPUNumVec {
public:
    /// Forward-declare GPU buffer (completed only in .cu file).
    struct CUDABuffer;

    /// Construct from host data (CPU vector).
    explicit GPUNumVec(const std::vector<double>& host);

    /// Construct an uninitialized vector of size n (on CPU).
    explicit GPUNumVec(size_t n);

    /// Default: empty vector.
    GPUNumVec() = default;

    ~GPUNumVec();
    GPUNumVec(const GPUNumVec&);
    GPUNumVec& operator=(const GPUNumVec&);
    GPUNumVec(GPUNumVec&&) noexcept;
    GPUNumVec& operator=(GPUNumVec&&) noexcept;

    /// ── Transfer ─────────────────────────────────────────────────────────
    GPUNumVec toGPU() const;
    GPUNumVec toCPU() const;

    /// ── Access ───────────────────────────────────────────────────────────
    Device device() const noexcept;
    size_t size() const noexcept;
    bool   empty() const noexcept;

    const std::vector<double>& host() const;
    std::vector<double>&       host();

    double* devicePtr();
    const double* devicePtr() const;

    /// ── BLAS-1 operations ────────────────────────────────────────────────

    /// r = a*x + y
    static GPUNumVec axpy(double a, const GPUNumVec& x, const GPUNumVec& y);

    /// r = a + b (element-wise)
    static GPUNumVec add(const GPUNumVec& a, const GPUNumVec& b);

    /// r = s*x (element-wise scale)
    static GPUNumVec scale(double s, const GPUNumVec& x);

    /// dot product: sum(a[i]*b[i])
    static double dot(const GPUNumVec& a, const GPUNumVec& b);

    /// L2 norm: sqrt(sum(x[i]*x[i]))
    static double nrm2(const GPUNumVec& x);

    /// Create a GPU vector from a raw device pointer (takes ownership via copy).
    static GPUNumVec from_device(const double* d_ptr, size_t n);

    friend class GPUNumMat;

private:
    std::vector<double> m_host;
    size_t m_size = 0;
    Device m_device = Device::CPU;
    std::shared_ptr<CUDABuffer> m_buf;

    void requireCPU(const char* op) const;
    void requireGPU(const char* op) const;
};

/// GPU-backed dense matrix (contiguous row-major double* on device).
///
/// For use in Broyden's method (rank-1 update), CG/GMRES (matvec),
/// and BDF/AM (linear system solve).
class SHAREDMATH_NUMERICALMETHODS_EXPORT GPUNumMat {
public:
    struct CUDABuffer;

    GPUNumMat(size_t rows, size_t cols, const std::vector<double>& host);
    GPUNumMat(size_t rows, size_t cols);
    GPUNumMat() = default;

    ~GPUNumMat();
    GPUNumMat(const GPUNumMat&);
    GPUNumMat& operator=(const GPUNumMat&);
    GPUNumMat(GPUNumMat&&) noexcept;
    GPUNumMat& operator=(GPUNumMat&&) noexcept;

    GPUNumMat toGPU() const;
    GPUNumMat toCPU() const;

    Device device() const noexcept;
    size_t rows() const noexcept;
    size_t cols() const noexcept;
    size_t size() const noexcept;

    const std::vector<double>& host() const;
    std::vector<double>&       host();

    double* devicePtr();
    const double* devicePtr() const;

    /// y = A * x (matrix-vector product via cuBLAS)
    GPUNumVec gemv(const GPUNumVec& x) const;

    /// Rank-1 update: A += alpha * u * v^T (via cuBLAS cublasDger)
    void ger(double alpha, const GPUNumVec& u, const GPUNumVec& v);

    /// Solve Ax = b via LU factorization (cuSOLVER)
    GPUNumVec solve(const GPUNumVec& b) const;

    friend class GPUNumVec;

private:
    std::vector<double> m_host;
    size_t m_rows = 0, m_cols = 0;
    Device m_device = Device::CPU;
    std::shared_ptr<CUDABuffer> m_buf;

    void requireCPU(const char* op) const;
    void requireGPU(const char* op) const;
};

} // namespace SharedMath::NumericalMethods
