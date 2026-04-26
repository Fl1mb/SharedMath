#pragma once

#include "AbstractMatrix.h"
#include "Tensor.h"

#include <vector>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <sharedmath_linearalgebra_export.h>

namespace SharedMath::LinearAlgebra {

// Forward-declare the CUDA accessor so DynamicMatrix can grant it friendship.
// The struct is fully defined in DynamicMatrixCUDA.h (internal, never installed).
namespace detail { struct DynamicMatrixCUDAImpl; }

// Row-major dense matrix backed by a single contiguous heap allocation.
//
// Layout: element (r, c) lives at data_[r * cols_ + c].
//
// GPU acceleration (Variant A — transparent dispatch):
//   Call .cuda() to move a matrix to the GPU, .cpu() to bring it back.
//   When CUDA is not compiled in, both calls are no-ops.
//   Operations between two GPU matrices are dispatched to cuBLAS / custom
//   CUDA kernels automatically; no user-visible API changes required.
//
class SHAREDMATH_LINEARALGEBRA_EXPORT DynamicMatrix : public AbstractMatrix {
public:
    // ── Construction ──────────────────────────────────────────────────────
    DynamicMatrix() = default;

    DynamicMatrix(size_t rows, size_t cols, double fill = 0.0)
        : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

    // Construct from pre-built flat row-major data
    DynamicMatrix(size_t rows, size_t cols, std::vector<double> flat)
        : rows_(rows), cols_(cols), data_(std::move(flat))
    {
        if (data_.size() != rows_ * cols_)
            throw std::invalid_argument(
                "DynamicMatrix: data size (" + std::to_string(data_.size()) +
                ") does not match rows*cols (" + std::to_string(rows_ * cols_) + ")");
    }

    // Deep-copy from any AbstractMatrix (goes through virtual interface)
    explicit DynamicMatrix(const AbstractMatrix& src)
        : rows_(src.rows()), cols_(src.cols()),
          data_(src.rows() * src.cols())
    {
        for (size_t i = 0; i < rows_; ++i)
            for (size_t j = 0; j < cols_; ++j)
                at_unsafe(i, j) = src.get(i, j);
    }

    explicit DynamicMatrix(std::shared_ptr<AbstractMatrix> src)
        : DynamicMatrix(*src) {}

    // Construct from 2-D CPU Tensor (zero-copy move)
    explicit DynamicMatrix(Tensor t) {
        if (t.ndim() != 2)
            throw std::invalid_argument(
                "DynamicMatrix: Tensor must be 2-D, got ndim=" +
                std::to_string(t.ndim()));
        rows_ = t.dim(0);
        cols_ = t.dim(1);
        data_ = std::move(t.data());
    }

    DynamicMatrix(const DynamicMatrix&)            = default;
    DynamicMatrix(DynamicMatrix&&) noexcept        = default;
    DynamicMatrix& operator=(const DynamicMatrix&) = default;
    DynamicMatrix& operator=(DynamicMatrix&&) noexcept = default;

    // Destructor is non-trivial when CUDA buffer is active.
    // shared_ptr<CUDABuffer> handles cleanup via its own deleter — no manual
    // CUDA calls needed here, and CUDABuffer can remain an incomplete type.
    ~DynamicMatrix() override = default;

    // ── AbstractMatrix interface ──────────────────────────────────────────
    size_t rows() const noexcept override { return rows_; }
    size_t cols() const noexcept override { return cols_; }

    // Element access is only valid for CPU matrices.
    double get(size_t r, size_t c) const override {
        requireCPU("get");
        return at_unsafe(r, c);
    }
    double& get(size_t r, size_t c) override {
        requireCPU("get");
        return at_unsafe(r, c);
    }
    void set(size_t r, size_t c, double v) override {
        requireCPU("set");
        at_unsafe(r, c) = v;
    }

    // toPtr() returns the host data pointer; null / dangling for GPU matrices.
    double*       toPtr()       noexcept override { return data_.data(); }
    const double* toPtr() const noexcept override { return data_.data(); }

    // ── Element access ────────────────────────────────────────────────────

    // Unchecked (r, c) syntax — only safe on CPU matrices
    double& operator()(size_t r, size_t c)       noexcept { return at_unsafe(r, c); }
    double  operator()(size_t r, size_t c) const noexcept { return at_unsafe(r, c); }

    // Bounds-checked access — CPU only
    double& at(size_t r, size_t c) {
        requireCPU("at");
        checkBounds(r, c);
        return at_unsafe(r, c);
    }
    double at(size_t r, size_t c) const {
        requireCPU("at");
        checkBounds(r, c);
        return at_unsafe(r, c);
    }

    // Pointer to the first element of row r — CPU only
    double*       row_ptr(size_t r)       noexcept { return data_.data() + r * cols_; }
    const double* row_ptr(size_t r) const noexcept { return data_.data() + r * cols_; }

    // Flat (linearised row-major) access — CPU only, no bounds check
    double& flat(size_t i)       noexcept { return data_[i]; }
    double  flat(size_t i) const noexcept { return data_[i]; }

    const std::vector<double>& data()  const noexcept { return data_; }
    std::vector<double>&       data()        noexcept { return data_; }

    // ── Metadata ──────────────────────────────────────────────────────────

    // size() is computed from shape so it is correct for both CPU and GPU matrices.
    size_t size()     const noexcept { return rows_ * cols_; }
    bool   empty()    const noexcept { return rows_ * cols_ == 0; }
    bool   isSquare() const noexcept { return rows_ == cols_; }

    // ── Device management ─────────────────────────────────────────────────

    Device device() const noexcept { return m_device; }

    // Move data to GPU (host → device copy).  Returns *this if already on GPU
    // or if no CUDA-capable device is present (graceful CPU fallback).
    // On CPU-only builds this is always a no-op.
    DynamicMatrix cuda() const;

    // Move data back to CPU (device → host copy).  No-op if already on CPU.
    DynamicMatrix cpu() const;

    // ── Comparison ────────────────────────────────────────────────────────
    bool operator==(const DynamicMatrix& o) const noexcept {
        return rows_ == o.rows_ && cols_ == o.cols_ && data_ == o.data_;
    }
    bool operator!=(const DynamicMatrix& o) const noexcept { return !(*this == o); }

    // ── Scalar arithmetic — CUDA-aware ────────────────────────────────────
    DynamicMatrix  operator* (double s) const;
    DynamicMatrix  operator/ (double s) const;
    DynamicMatrix& operator*=(double s);
    DynamicMatrix& operator/=(double s);

    friend DynamicMatrix operator*(double s, const DynamicMatrix& m) { return m * s; }

    // ── Matrix arithmetic — CUDA-aware ────────────────────────────────────
    DynamicMatrix  operator+ (const DynamicMatrix& o) const;
    DynamicMatrix  operator- (const DynamicMatrix& o) const;
    DynamicMatrix& operator+=(const DynamicMatrix& o);
    DynamicMatrix& operator-=(const DynamicMatrix& o);

    // Matrix multiply — dispatches to cuBLAS when both operands are on GPU
    DynamicMatrix  operator* (const DynamicMatrix& B) const;

    // ── Transpose ─────────────────────────────────────────────────────────
    DynamicMatrix transposed() const {
        requireCPU("transposed");
        DynamicMatrix T(cols_, rows_);
        for (size_t i = 0; i < rows_; ++i) {
            const double* Ai = row_ptr(i);
            for (size_t j = 0; j < cols_; ++j)
                T.at_unsafe(j, i) = Ai[j];
        }
        return T;
    }

    // ── Utilities ─────────────────────────────────────────────────────────
    void clear()        noexcept { std::fill(data_.begin(), data_.end(), 0.0); }
    void fill(double v) noexcept { std::fill(data_.begin(), data_.end(), v); }

    // ── Named constructors ────────────────────────────────────────────────
    static DynamicMatrix zeros(size_t r, size_t c) { return DynamicMatrix(r, c, 0.0); }
    static DynamicMatrix ones (size_t r, size_t c) { return DynamicMatrix(r, c, 1.0); }
    static DynamicMatrix eye  (size_t n) {
        DynamicMatrix M(n, n, 0.0);
        for (size_t i = 0; i < n; ++i) M(i, i) = 1.0;
        return M;
    }

    // ── Tensor interop — CPU matrices only ───────────────────────────────

    // Copy data into a 2-D CPU Tensor.
    // For GPU matrices call .cpu().toTensor().
    Tensor toTensor() const {
        requireCPU("toTensor");
        return Tensor({rows_, cols_}, data_);
    }

    static DynamicMatrix fromTensor(const Tensor& t) {
        if (t.ndim() != 2)
            throw std::invalid_argument(
                "DynamicMatrix::fromTensor: Tensor must be 2-D");
        return DynamicMatrix(t.dim(0), t.dim(1), t.data());
    }

private:
    size_t rows_ = 0;
    size_t cols_ = 0;
    std::vector<double> data_; // flat row-major: index = r * cols_ + c
                               // empty when matrix is on GPU

    // ── GPU storage (PIMPL — no CUDA headers leak into this public header) //
    struct CUDABuffer;                           // defined in DynamicMatrixCUDA.cu
    std::shared_ptr<CUDABuffer> m_cuda_buf;      // null → CPU matrix
    Device m_device = Device::CPU;

    // Private factory: wraps a freshly-allocated GPU buffer into a DynamicMatrix.
    // Only called from DynamicMatrixCUDA.cu.
    static DynamicMatrix from_cuda(size_t rows, size_t cols,
                                   std::shared_ptr<CUDABuffer> buf);

    // Grants the CUDA translation unit access to private members without
    // exposing raw pointers in the public header.
    friend struct detail::DynamicMatrixCUDAImpl;

    // ── Private helpers ───────────────────────────────────────────────────

    double& at_unsafe(size_t r, size_t c) noexcept {
        return data_[r * cols_ + c];
    }
    double at_unsafe(size_t r, size_t c) const noexcept {
        return data_[r * cols_ + c];
    }

    void requireCPU(const char* op) const {
        if (m_device != Device::CPU)
            throw std::runtime_error(
                std::string("DynamicMatrix::") + op +
                ": matrix is on GPU — call .cpu() first to access elements");
    }

    void checkBounds(size_t r, size_t c) const {
        if (r >= rows_ || c >= cols_)
            throw std::out_of_range(
                "DynamicMatrix::at: index (" + std::to_string(r) + ", " +
                std::to_string(c) + ") out of range for matrix (" +
                std::to_string(rows_) + "x" + std::to_string(cols_) + ")");
    }

    void checkSameShape(const DynamicMatrix& o) const {
        if (rows_ != o.rows_ || cols_ != o.cols_)
            throw std::invalid_argument(
                "DynamicMatrix: shape mismatch (" +
                std::to_string(rows_) + "x" + std::to_string(cols_) +
                " vs " + std::to_string(o.rows_) + "x" + std::to_string(o.cols_) + ")");
    }

    void checkSameDevice(const DynamicMatrix& o) const {
        if (m_device != o.m_device)
            throw std::invalid_argument(
                "DynamicMatrix: operands are on different devices — "
                "align them with .cuda() or .cpu() before this operation");
    }
};

} // namespace SharedMath::LinearAlgebra
