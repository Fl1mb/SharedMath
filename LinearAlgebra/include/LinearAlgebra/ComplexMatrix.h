#pragma once

#include <vector>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <sharedmath_linearalgebra_export.h>

namespace SharedMath::LinearAlgebra {

using Complex = std::complex<double>;

/// Row-major dense complex matrix backed by a single contiguous heap allocation.
///
/// Layout: element (r, c) lives at data_[r * cols_ + c].
///
/// Used for AC analysis in circuit simulation:
///   Y · V = I
/// where Y is the complex admittance matrix.
///
class SHAREDMATH_LINEARALGEBRA_EXPORT ComplexMatrix {
public:
    // ── Construction ──────────────────────────────────────────────────────
    ComplexMatrix() = default;

    ComplexMatrix(size_t rows, size_t cols, Complex fill = Complex(0.0, 0.0))
        : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

    ComplexMatrix(size_t rows, size_t cols, std::vector<Complex> flat)
        : rows_(rows), cols_(cols), data_(std::move(flat))
    {
        if (data_.size() != rows_ * cols_)
            throw std::invalid_argument(
                "ComplexMatrix: data size (" + std::to_string(data_.size()) +
                ") does not match rows*cols (" + std::to_string(rows_ * cols_) + ")");
    }

    ComplexMatrix(const ComplexMatrix&)            = default;
    ComplexMatrix(ComplexMatrix&&) noexcept        = default;
    ComplexMatrix& operator=(const ComplexMatrix&) = default;
    ComplexMatrix& operator=(ComplexMatrix&&) noexcept = default;
    ~ComplexMatrix()                               = default;

    // ── Metadata ──────────────────────────────────────────────────────────
    size_t rows()     const noexcept { return rows_; }
    size_t cols()     const noexcept { return cols_; }
    size_t size()     const noexcept { return rows_ * cols_; }
    bool   empty()    const noexcept { return rows_ * cols_ == 0; }
    bool   isSquare() const noexcept { return rows_ == cols_; }

    // ── Element access ────────────────────────────────────────────────────
    Complex& operator()(size_t r, size_t c)       noexcept { return at_unsafe(r, c); }
    Complex  operator()(size_t r, size_t c) const noexcept { return at_unsafe(r, c); }

    Complex& at(size_t r, size_t c) {
        checkBounds(r, c);
        return at_unsafe(r, c);
    }
    Complex at(size_t r, size_t c) const {
        checkBounds(r, c);
        return at_unsafe(r, c);
    }

    Complex*       row_ptr(size_t r)       noexcept { return data_.data() + r * cols_; }
    const Complex* row_ptr(size_t r) const noexcept { return data_.data() + r * cols_; }

    Complex*       data()       noexcept { return data_.data(); }
    const Complex* data() const noexcept { return data_.data(); }

    const std::vector<Complex>& vec() const noexcept { return data_; }
    std::vector<Complex>&       vec()       noexcept { return data_; }

    // ── Comparison ────────────────────────────────────────────────────────
    bool operator==(const ComplexMatrix& o) const noexcept {
        return rows_ == o.rows_ && cols_ == o.cols_ && data_ == o.data_;
    }
    bool operator!=(const ComplexMatrix& o) const noexcept { return !(*this == o); }

    // ── Scalar arithmetic ─────────────────────────────────────────────────
    ComplexMatrix  operator* (Complex s) const;
    ComplexMatrix& operator*=(Complex s);

    friend ComplexMatrix operator*(Complex s, const ComplexMatrix& m) { return m * s; }

    // ── Matrix arithmetic ─────────────────────────────────────────────────
    ComplexMatrix  operator+ (const ComplexMatrix& o) const;
    ComplexMatrix  operator- (const ComplexMatrix& o) const;
    ComplexMatrix& operator+=(const ComplexMatrix& o);
    ComplexMatrix& operator-=(const ComplexMatrix& o);

    ComplexMatrix  operator* (const ComplexMatrix& B) const;

    // ── Utilities ─────────────────────────────────────────────────────────
    void clear()                   noexcept { std::fill(data_.begin(), data_.end(), Complex(0.0, 0.0)); }
    void fill(Complex v)           noexcept { std::fill(data_.begin(), data_.end(), v); }

    // ── Named constructors ────────────────────────────────────────────────
    static ComplexMatrix zeros(size_t r, size_t c) { return ComplexMatrix(r, c, Complex(0.0, 0.0)); }
    static ComplexMatrix eye  (size_t n) {
        ComplexMatrix M(n, n, Complex(0.0, 0.0));
        for (size_t i = 0; i < n; ++i) M(i, i) = Complex(1.0, 0.0);
        return M;
    }

private:
    size_t rows_ = 0;
    size_t cols_ = 0;
    std::vector<Complex> data_;

    Complex& at_unsafe(size_t r, size_t c) noexcept { return data_[r * cols_ + c]; }
    Complex at_unsafe(size_t r, size_t c) const noexcept { return data_[r * cols_ + c]; }

    void checkBounds(size_t r, size_t c) const {
        if (r >= rows_ || c >= cols_)
            throw std::out_of_range(
                "ComplexMatrix::at: index (" + std::to_string(r) + ", " +
                std::to_string(c) + ") out of range for matrix (" +
                std::to_string(rows_) + "x" + std::to_string(cols_) + ")");
    }

    void checkSameShape(const ComplexMatrix& o) const {
        if (rows_ != o.rows_ || cols_ != o.cols_)
            throw std::invalid_argument(
                "ComplexMatrix: shape mismatch (" +
                std::to_string(rows_) + "x" + std::to_string(cols_) +
                " vs " + std::to_string(o.rows_) + "x" + std::to_string(o.cols_) + ")");
    }
};

/// ── Free functions ───────────────────────────────────────────────────────

/// Element-wise real part extraction
SHAREDMATH_LINEARALGEBRA_EXPORT
std::vector<double> real(const ComplexMatrix& m);

/// Element-wise imaginary part extraction
SHAREDMATH_LINEARALGEBRA_EXPORT
std::vector<double> imag(const ComplexMatrix& m);

} // namespace SharedMath::LinearAlgebra
