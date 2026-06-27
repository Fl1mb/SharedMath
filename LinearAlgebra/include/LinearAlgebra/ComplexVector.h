#pragma once

#include <vector>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <string>
#include <ostream>
#include <limits>
#include <sharedmath_linearalgebra_export.h>

namespace SharedMath::LinearAlgebra {

using Complex = std::complex<double>;

/// Heap-allocated dense complex vector.
/// Complements ComplexMatrix: supports the same arithmetic patterns and
/// provides first-class matrix–vector multiply as free functions.
///
class SHAREDMATH_LINEARALGEBRA_EXPORT ComplexVector {
public:
    // ── Construction ──────────────────────────────────────────────────────
    ComplexVector() = default;

    explicit ComplexVector(size_t n, Complex fill = Complex(0.0, 0.0))
        : data_(n, fill) {}

    explicit ComplexVector(std::vector<Complex> data)
        : data_(std::move(data)) {}

    ComplexVector(const ComplexVector&)            = default;
    ComplexVector(ComplexVector&&) noexcept        = default;
    ComplexVector& operator=(const ComplexVector&) = default;
    ComplexVector& operator=(ComplexVector&&) noexcept = default;
    ~ComplexVector()                               = default;

    // ── Element access ────────────────────────────────────────────────────
    Complex& operator[](size_t i)       noexcept { return data_[i]; }
    Complex  operator[](size_t i) const noexcept { return data_[i]; }

    Complex& at(size_t i) {
        if (i >= data_.size())
            throw std::out_of_range("ComplexVector::at: index " +
                                    std::to_string(i) + " out of range " +
                                    std::to_string(data_.size()));
        return data_[i];
    }
    Complex at(size_t i) const {
        if (i >= data_.size())
            throw std::out_of_range("ComplexVector::at: index " +
                                    std::to_string(i) + " out of range " +
                                    std::to_string(data_.size()));
        return data_[i];
    }

    // ── Iterators / raw pointer ───────────────────────────────────────────
    Complex*       begin()        noexcept { return data_.data(); }
    const Complex* begin()  const noexcept { return data_.data(); }
    Complex*       end()          noexcept { return data_.data() + data_.size(); }
    const Complex* end()    const noexcept { return data_.data() + data_.size(); }
    const Complex* cbegin() const noexcept { return data_.data(); }
    const Complex* cend()   const noexcept { return data_.data() + data_.size(); }

    Complex*       data()       noexcept { return data_.data(); }
    const Complex* data() const noexcept { return data_.data(); }

    const std::vector<Complex>& vec() const noexcept { return data_; }
    std::vector<Complex>&       vec()       noexcept { return data_; }

    // ── Metadata ──────────────────────────────────────────────────────────
    size_t size()  const noexcept { return data_.size(); }
    bool   empty() const noexcept { return data_.empty(); }

    // ── Comparison ────────────────────────────────────────────────────────
    bool operator==(const ComplexVector& o) const noexcept { return data_ == o.data_; }
    bool operator!=(const ComplexVector& o) const noexcept { return data_ != o.data_; }

    // ── Arithmetic ────────────────────────────────────────────────────────
    ComplexVector operator-() const {
        ComplexVector r(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) r.data_[i] = -data_[i];
        return r;
    }

    ComplexVector operator+(const ComplexVector& o) const {
        checkSize(o);
        ComplexVector r(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) r.data_[i] = data_[i] + o.data_[i];
        return r;
    }
    ComplexVector operator-(const ComplexVector& o) const {
        checkSize(o);
        ComplexVector r(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) r.data_[i] = data_[i] - o.data_[i];
        return r;
    }
    ComplexVector operator*(Complex s) const {
        ComplexVector r(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) r.data_[i] = data_[i] * s;
        return r;
    }
    ComplexVector operator/(Complex s) const {
        Complex inv = Complex(1.0, 0.0) / s;
        ComplexVector r(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) r.data_[i] = data_[i] * inv;
        return r;
    }

    ComplexVector& operator+=(const ComplexVector& o) {
        checkSize(o);
        for (size_t i = 0; i < data_.size(); ++i) data_[i] += o.data_[i];
        return *this;
    }
    ComplexVector& operator-=(const ComplexVector& o) {
        checkSize(o);
        for (size_t i = 0; i < data_.size(); ++i) data_[i] -= o.data_[i];
        return *this;
    }
    ComplexVector& operator*=(Complex s) {
        for (Complex& v : data_) v *= s;
        return *this;
    }
    ComplexVector& operator/=(Complex s) {
        Complex inv = Complex(1.0, 0.0) / s;
        for (Complex& v : data_) v *= inv;
        return *this;
    }

    friend ComplexVector operator*(Complex s, const ComplexVector& v) { return v * s; }

    // ── Linear algebra ────────────────────────────────────────────────────

    /// Hermitian dot product: conj(this)^T * o
    Complex dot(const ComplexVector& o) const {
        checkSize(o);
        Complex s(0.0, 0.0);
        for (size_t i = 0; i < data_.size(); ++i)
            s += std::conj(data_[i]) * o.data_[i];
        return s;
    }

    /// Euclidean norm (L2)
    double norm() const {
        double s = 0.0;
        for (const auto& v : data_) s += std::norm(v);
        return std::sqrt(s);
    }

    ComplexVector normalized() const {
        double n = norm();
        if (n < 1e-300)
            throw std::runtime_error("ComplexVector::normalized: zero vector");
        return *this * Complex(1.0 / n, 0.0);
    }

    // ── Named constructors ────────────────────────────────────────────────
    static ComplexVector zeros(size_t n) { return ComplexVector(n, Complex(0.0, 0.0)); }
    static ComplexVector ones (size_t n) { return ComplexVector(n, Complex(1.0, 0.0)); }

    // ── Display ───────────────────────────────────────────────────────────
    SHAREDMATH_LINEARALGEBRA_EXPORT
    friend std::ostream& operator<<(std::ostream& os, const ComplexVector& v);

private:
    std::vector<Complex> data_;

    void checkSize(const ComplexVector& o) const {
        if (data_.size() != o.data_.size())
            throw std::invalid_argument(
                "ComplexVector: size mismatch (" +
                std::to_string(data_.size()) + " vs " +
                std::to_string(o.data_.size()) + ")");
    }
};

} // namespace SharedMath::LinearAlgebra
