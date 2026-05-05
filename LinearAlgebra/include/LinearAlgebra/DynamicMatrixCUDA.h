/// DynamicMatrixCUDA.h — internal CUDA accessor and dispatch declarations for
/// DynamicMatrix.
///
/// THIS HEADER IS NOT PART OF THE PUBLIC API.
/// It is included only by DynamicMatrix.cpp and DynamicMatrixCUDA.cu.
/// Do not install it or include it from user-facing code.
///
/// Architecture mirrors TensorCUDA.h:
///   • DynamicMatrixCUDAImpl   — grants the .cu TU access to private fields
///                               via the friend declaration in DynamicMatrix.h
///   • dm_cuda_*               — free-function dispatch points called from
///                               DynamicMatrix.cpp under the SHAREDMATH_CUDA guard
#pragma once

#include "DynamicMatrix.h"   // resolved via the PRIVATE include path

namespace SharedMath::LinearAlgebra {
namespace detail {

/// ── Dispatch functions (defined in DynamicMatrixCUDA.cu) ──────────────────────

/// Matrix multiply: C = A * B  (both must be on GPU, row-major)
DynamicMatrix dm_cuda_matmul(const DynamicMatrix& A, const DynamicMatrix& B);

/// Element-wise add / subtract (same shape, both on GPU)
DynamicMatrix dm_cuda_add(const DynamicMatrix& A, const DynamicMatrix& B);
DynamicMatrix dm_cuda_sub(const DynamicMatrix& A, const DynamicMatrix& B);

/// Element-wise multiply by scalar
DynamicMatrix dm_cuda_scale(const DynamicMatrix& A, double scalar);

/// ── Private-member accessor ───────────────────────────────────────────────────
/// DynamicMatrix declares `friend struct detail::DynamicMatrixCUDAImpl;`
/// so this struct can reach into private storage without breaking encapsulation.

struct DynamicMatrixCUDAImpl {
    /// Raw device pointer to the GPU buffer
    static double* cuda_ptr(const DynamicMatrix& m);
    static double* buffer_ptr(
        const std::shared_ptr<DynamicMatrix::CUDABuffer>& buf);
    static std::shared_ptr<DynamicMatrix::CUDABuffer> make_buffer(size_t n);
    static std::shared_ptr<DynamicMatrix::CUDABuffer> make_buffer(
        const double* host_src, size_t n);

    static size_t nrows(const DynamicMatrix& m) { return m.rows_; }
    static size_t ncols(const DynamicMatrix& m) { return m.cols_; }

    // Wrap a freshly-allocated CUDABuffer into a DynamicMatrix
    static DynamicMatrix make(size_t rows, size_t cols,
                              std::shared_ptr<DynamicMatrix::CUDABuffer> buf) {
        return DynamicMatrix::from_cuda(rows, cols, std::move(buf));
    }
};

} // namespace detail
} // namespace SharedMath::LinearAlgebra
