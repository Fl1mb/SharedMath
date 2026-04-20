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

// Row-major dense matrix backed by a single contiguous heap allocation.
//
// Layout: element (r, c) lives at data_[r * cols_ + c].
//
// Benefits over the old vector<vector<double>> design:
//  - Single allocation → cache-friendly row traversal
//  - toPtr() is correct and non-UB
//  - Cache-friendly matmul (i-k-j loop keeps B rows hot)
//  - Zero-cost conversion to/from Tensor
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

    // Construct from 2-D Tensor (zero-copy move)
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
    ~DynamicMatrix() override = default;

    // ── AbstractMatrix interface ──────────────────────────────────────────
    size_t rows() const noexcept override { return rows_; }
    size_t cols() const noexcept override { return cols_; }

    double  get(size_t r, size_t c) const override { return at_unsafe(r, c); }
    double& get(size_t r, size_t c)       override { return at_unsafe(r, c); }
    void    set(size_t r, size_t c, double v) override { at_unsafe(r, c) = v; }

    // Single allocation → toPtr() is always correct and contiguous
    double*       toPtr()       noexcept override { return data_.data(); }
    const double* toPtr() const noexcept override { return data_.data(); }

    // ── Element access ────────────────────────────────────────────────────

    // Natural (r, c) syntax — no bounds check (use at() for checked access)
    double& operator()(size_t r, size_t c)       noexcept { return at_unsafe(r, c); }
    double  operator()(size_t r, size_t c) const noexcept { return at_unsafe(r, c); }

    // Bounds-checked access
    double& at(size_t r, size_t c) {
        checkBounds(r, c);
        return at_unsafe(r, c);
    }
    double at(size_t r, size_t c) const {
        checkBounds(r, c);
        return at_unsafe(r, c);
    }

    // Pointer to the first element of row r (useful for vectorised code / BLAS)
    double*       row_ptr(size_t r)       noexcept { return data_.data() + r * cols_; }
    const double* row_ptr(size_t r) const noexcept { return data_.data() + r * cols_; }

    // Flat (linearised row-major) access
    double& flat(size_t i)       noexcept { return data_[i]; }
    double  flat(size_t i) const noexcept { return data_[i]; }

    const std::vector<double>& data()  const noexcept { return data_; }
    std::vector<double>&       data()        noexcept { return data_; }

    // ── Metadata ──────────────────────────────────────────────────────────
    bool   empty() const noexcept { return data_.empty(); }
    size_t size()  const noexcept { return data_.size();  }
    bool   isSquare() const noexcept { return rows_ == cols_; }

    // ── Comparison ────────────────────────────────────────────────────────
    bool operator==(const DynamicMatrix& o) const noexcept {
        return rows_ == o.rows_ && cols_ == o.cols_ && data_ == o.data_;
    }
    bool operator!=(const DynamicMatrix& o) const noexcept { return !(*this == o); }

    // ── Scalar arithmetic ─────────────────────────────────────────────────
    DynamicMatrix operator*(double s) const { auto r = *this; r *= s; return r; }
    DynamicMatrix operator/(double s) const { auto r = *this; r /= s; return r; }

    DynamicMatrix& operator*=(double s) noexcept {
        for (double& v : data_) v *= s;
        return *this;
    }
    DynamicMatrix& operator/=(double s) {
        double inv = 1.0 / s;
        for (double& v : data_) v *= inv;
        return *this;
    }

    friend DynamicMatrix operator*(double s, const DynamicMatrix& m) { return m * s; }

    // ── Matrix arithmetic ─────────────────────────────────────────────────
    DynamicMatrix operator+(const DynamicMatrix& o) const { auto r = *this; r += o; return r; }
    DynamicMatrix operator-(const DynamicMatrix& o) const { auto r = *this; r -= o; return r; }

    DynamicMatrix& operator+=(const DynamicMatrix& o) {
        checkSameShape(o);
        for (size_t i = 0; i < data_.size(); ++i) data_[i] += o.data_[i];
        return *this;
    }
    DynamicMatrix& operator-=(const DynamicMatrix& o) {
        checkSameShape(o);
        for (size_t i = 0; i < data_.size(); ++i) data_[i] -= o.data_[i];
        return *this;
    }

    // Cache-friendly matrix multiply.
    //
    // Loop order i-k-j:
    //   - A rows accessed sequentially (row i stays hot in cache)
    //   - B rows accessed sequentially (row k stays hot in cache)
    //   - C rows written sequentially
    // This is ~3-10× faster than naïve i-j-k on large matrices.
    DynamicMatrix operator*(const DynamicMatrix& B) const {
        if (cols_ != B.rows_)
            throw std::invalid_argument(
                "DynamicMatrix: inner dimensions mismatch (" +
                std::to_string(cols_) + " vs " + std::to_string(B.rows_) + ")");

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

    // ── Transpose ─────────────────────────────────────────────────────────
    DynamicMatrix transposed() const {
        DynamicMatrix T(cols_, rows_);
        for (size_t i = 0; i < rows_; ++i) {
            const double* Ai = row_ptr(i);
            for (size_t j = 0; j < cols_; ++j)
                T.at_unsafe(j, i) = Ai[j];
        }
        return T;
    }

    // ── Utilities ─────────────────────────────────────────────────────────
    void clear()       noexcept { std::fill(data_.begin(), data_.end(), 0.0); }
    void fill(double v) noexcept { std::fill(data_.begin(), data_.end(), v); }

    // ── Named constructors ────────────────────────────────────────────────
    static DynamicMatrix zeros(size_t r, size_t c) { return DynamicMatrix(r, c, 0.0); }
    static DynamicMatrix ones (size_t r, size_t c) { return DynamicMatrix(r, c, 1.0); }
    static DynamicMatrix eye  (size_t n) {
        DynamicMatrix M(n, n, 0.0);
        for (size_t i = 0; i < n; ++i) M(i, i) = 1.0;
        return M;
    }

    // ── Tensor interop ────────────────────────────────────────────────────

    // Copy data into a 2-D Tensor
    Tensor toTensor() const {
        return Tensor({rows_, cols_}, data_);
    }

    // Build a DynamicMatrix from a 2-D Tensor (copies data)
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

    // Unchecked access — callers guarantee valid indices
    double& at_unsafe(size_t r, size_t c) noexcept {
        return data_[r * cols_ + c];
    }
    double at_unsafe(size_t r, size_t c) const noexcept {
        return data_[r * cols_ + c];
    }

    void checkBounds(size_t r, size_t c) const {
        if (r >= rows_ || c >= cols_)
            throw std::out_of_range(
                "DynamicMatrix::at: index (" + std::to_string(r) + ", " +
                std::to_string(c) + ") out of range for matrix (" +
                std::to_string(rows_) + " x " + std::to_string(cols_) + ")");
    }

    void checkSameShape(const DynamicMatrix& o) const {
        if (rows_ != o.rows_ || cols_ != o.cols_)
            throw std::invalid_argument(
                "DynamicMatrix: shape mismatch (" +
                std::to_string(rows_) + "x" + std::to_string(cols_) +
                " vs " + std::to_string(o.rows_) + "x" + std::to_string(o.cols_) + ")");
    }
};

} // namespace SharedMath::LinearAlgebra
