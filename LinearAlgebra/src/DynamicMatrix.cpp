// DynamicMatrix.cpp — out-of-line method implementations for DynamicMatrix.
//
// All arithmetic operators live here (not inline in the header) so that the
// CUDA dispatch macros below can include DynamicMatrixCUDA.h without creating
// a circular dependency between the public header and the internal CUDA header.

#include "LinearAlgebra/DynamicMatrix.h"

#ifdef SHAREDMATH_CUDA
#  include "DynamicMatrixCUDA.h"   // dispatch declarations (internal, not installed)
#endif

#include <stdexcept>
#include <string>

namespace SharedMath::LinearAlgebra {

// ─── Device management CPU stubs ──────────────────────────────────────────────
// When CUDA support is not compiled in, cuda()/cpu() are no-ops and from_cuda()
// is a linker stub that should never be reached.

#ifndef SHAREDMATH_CUDA

DynamicMatrix DynamicMatrix::cuda() const { return *this; }
DynamicMatrix DynamicMatrix::cpu()  const { return *this; }

DynamicMatrix DynamicMatrix::from_cuda(size_t /*rows*/, size_t /*cols*/,
                                        std::shared_ptr<CUDABuffer> /*buf*/) {
    throw std::runtime_error(
        "DynamicMatrix::from_cuda: CUDA support not compiled in");
}

#endif // !SHAREDMATH_CUDA

// ─── CUDA dispatch helpers ────────────────────────────────────────────────────
// These macros inject a CUDA fast-path at the top of each arithmetic operator.
// On CPU-only builds they expand to nothing.

#ifdef SHAREDMATH_CUDA
#  define DM_CUDA_SCALE(A, s)    return detail::dm_cuda_scale((A), (s))
#  define DM_CUDA_BINARY(fn,A,B) return detail::fn((A), (B))
#  define DM_CUDA_MATMUL(A,B)    return detail::dm_cuda_matmul((A), (B))
#else
#  define DM_CUDA_SCALE(A, s)
#  define DM_CUDA_BINARY(fn,A,B)
#  define DM_CUDA_MATMUL(A,B)
#endif

// ─── Scalar arithmetic ────────────────────────────────────────────────────────

DynamicMatrix DynamicMatrix::operator*(double s) const {
#ifdef SHAREDMATH_CUDA
    if (m_device == Device::CUDA) { DM_CUDA_SCALE(*this, s); }
#endif
    auto r = *this;
    for (double& v : r.data_) v *= s;
    return r;
}

DynamicMatrix DynamicMatrix::operator/(double s) const {
#ifdef SHAREDMATH_CUDA
    if (m_device == Device::CUDA) { DM_CUDA_SCALE(*this, 1.0 / s); }
#endif
    double inv = 1.0 / s;
    auto r = *this;
    for (double& v : r.data_) v *= inv;
    return r;
}

DynamicMatrix& DynamicMatrix::operator*=(double s) {
#ifdef SHAREDMATH_CUDA
    if (m_device == Device::CUDA) { *this = detail::dm_cuda_scale(*this, s); return *this; }
#endif
    for (double& v : data_) v *= s;
    return *this;
}

DynamicMatrix& DynamicMatrix::operator/=(double s) {
#ifdef SHAREDMATH_CUDA
    if (m_device == Device::CUDA) { *this = detail::dm_cuda_scale(*this, 1.0 / s); return *this; }
#endif
    double inv = 1.0 / s;
    for (double& v : data_) v *= inv;
    return *this;
}

// ─── Matrix arithmetic ────────────────────────────────────────────────────────

DynamicMatrix DynamicMatrix::operator+(const DynamicMatrix& o) const {
    checkSameShape(o);
    checkSameDevice(o);
#ifdef SHAREDMATH_CUDA
    if (m_device == Device::CUDA) { DM_CUDA_BINARY(dm_cuda_add, *this, o); }
#endif
    auto r = *this;
    for (size_t i = 0; i < data_.size(); ++i) r.data_[i] += o.data_[i];
    return r;
}

DynamicMatrix DynamicMatrix::operator-(const DynamicMatrix& o) const {
    checkSameShape(o);
    checkSameDevice(o);
#ifdef SHAREDMATH_CUDA
    if (m_device == Device::CUDA) { DM_CUDA_BINARY(dm_cuda_sub, *this, o); }
#endif
    auto r = *this;
    for (size_t i = 0; i < data_.size(); ++i) r.data_[i] -= o.data_[i];
    return r;
}

DynamicMatrix& DynamicMatrix::operator+=(const DynamicMatrix& o) {
    checkSameShape(o);
    checkSameDevice(o);
#ifdef SHAREDMATH_CUDA
    if (m_device == Device::CUDA) { *this = detail::dm_cuda_add(*this, o); return *this; }
#endif
    for (size_t i = 0; i < data_.size(); ++i) data_[i] += o.data_[i];
    return *this;
}

DynamicMatrix& DynamicMatrix::operator-=(const DynamicMatrix& o) {
    checkSameShape(o);
    checkSameDevice(o);
#ifdef SHAREDMATH_CUDA
    if (m_device == Device::CUDA) { *this = detail::dm_cuda_sub(*this, o); return *this; }
#endif
    for (size_t i = 0; i < data_.size(); ++i) data_[i] -= o.data_[i];
    return *this;
}

// Cache-friendly matrix multiply (i-k-j loop order).
// Dispatches to cuBLAS when both matrices are on GPU.
DynamicMatrix DynamicMatrix::operator*(const DynamicMatrix& B) const {
    if (cols_ != B.rows_)
        throw std::invalid_argument(
            "DynamicMatrix: inner dimensions mismatch (" +
            std::to_string(cols_) + " vs " + std::to_string(B.rows_) + ")");
    checkSameDevice(B);
#ifdef SHAREDMATH_CUDA
    if (m_device == Device::CUDA) { DM_CUDA_MATMUL(*this, B); }
#endif
    DynamicMatrix C(rows_, B.cols_, 0.0);
    for (size_t i = 0; i < rows_; ++i) {
        const double* Ai = row_ptr(i);
        double*       Ci = C.row_ptr(i);
        for (size_t k = 0; k < cols_; ++k) {
            const double  a  = Ai[k];
            const double* Bk = B.row_ptr(k);
            for (size_t j = 0; j < B.cols_; ++j)
                Ci[j] += a * Bk[j];
        }
    }
    return C;
}

} // namespace SharedMath::LinearAlgebra
