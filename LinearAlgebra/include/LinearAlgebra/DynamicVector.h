#pragma once

#include "Tensor.h"         // Device enum
#include <sharedmath_linearalgebra_export.h>

#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <string>
#include <ostream>
#include <limits>

namespace SharedMath::LinearAlgebra {

class DynamicMatrix;   // forward declaration — avoids circular include

// Heap-allocated dense vector of doubles.
// Complements DynamicMatrix: supports the same arithmetic patterns and
// provides first-class matrix–vector multiply (A*x, x*A) as free functions.
//
// Design mirrors DynamicMatrix: all storage is a flat std::vector<double>,
// element access is O(1), the class is value-semantic (copy/move are cheap).
//
class SHAREDMATH_LINEARALGEBRA_EXPORT DynamicVector {
public:
    // ── Construction ──────────────────────────────────────────────────────
    DynamicVector() = default;

    explicit DynamicVector(size_t n, double fill = 0.0)
        : data_(n, fill) {}

    explicit DynamicVector(std::vector<double> data)
        : data_(std::move(data)) {}

    DynamicVector(std::initializer_list<double> init)
        : data_(init) {}

    DynamicVector(const DynamicVector&)            = default;
    DynamicVector(DynamicVector&&) noexcept        = default;
    DynamicVector& operator=(const DynamicVector&) = default;
    DynamicVector& operator=(DynamicVector&&) noexcept = default;
    ~DynamicVector()                               = default;

    // ── Element access ────────────────────────────────────────────────────
    double& operator[](size_t i)       noexcept { return data_[i]; }
    double  operator[](size_t i) const noexcept { return data_[i]; }

    double& at(size_t i) {
        if (i >= data_.size())
            throw std::out_of_range("DynamicVector::at: index " +
                                    std::to_string(i) + " out of range " +
                                    std::to_string(data_.size()));
        return data_[i];
    }
    double at(size_t i) const {
        if (i >= data_.size())
            throw std::out_of_range("DynamicVector::at: index " +
                                    std::to_string(i) + " out of range " +
                                    std::to_string(data_.size()));
        return data_[i];
    }

    // ── Iterators / raw pointer ───────────────────────────────────────────
    double*       begin()        noexcept { return data_.data(); }
    const double* begin()  const noexcept { return data_.data(); }
    double*       end()          noexcept { return data_.data() + data_.size(); }
    const double* end()    const noexcept { return data_.data() + data_.size(); }
    const double* cbegin() const noexcept { return data_.data(); }
    const double* cend()   const noexcept { return data_.data() + data_.size(); }

    double*       data()         noexcept { return data_.data(); }
    const double* data()   const noexcept { return data_.data(); }

    const std::vector<double>& vec() const noexcept { return data_; }
    std::vector<double>&       vec()       noexcept { return data_; }

    // ── Metadata ──────────────────────────────────────────────────────────
    size_t size()  const noexcept { return data_.size(); }
    bool   empty() const noexcept { return data_.empty(); }

    // ── Comparison ────────────────────────────────────────────────────────
    bool operator==(const DynamicVector& o) const noexcept { return data_ == o.data_; }
    bool operator!=(const DynamicVector& o) const noexcept { return data_ != o.data_; }

    // ── Arithmetic ────────────────────────────────────────────────────────
    DynamicVector operator-() const {
        DynamicVector r(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) r.data_[i] = -data_[i];
        return r;
    }

    DynamicVector operator+(const DynamicVector& o) const {
        checkSize(o);
        DynamicVector r(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) r.data_[i] = data_[i] + o.data_[i];
        return r;
    }
    DynamicVector operator-(const DynamicVector& o) const {
        checkSize(o);
        DynamicVector r(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) r.data_[i] = data_[i] - o.data_[i];
        return r;
    }
    DynamicVector operator*(double s) const {
        DynamicVector r(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) r.data_[i] = data_[i] * s;
        return r;
    }
    DynamicVector operator/(double s) const {
        double inv = 1.0 / s;
        DynamicVector r(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) r.data_[i] = data_[i] * inv;
        return r;
    }

    DynamicVector& operator+=(const DynamicVector& o) {
        checkSize(o);
        for (size_t i = 0; i < data_.size(); ++i) data_[i] += o.data_[i];
        return *this;
    }
    DynamicVector& operator-=(const DynamicVector& o) {
        checkSize(o);
        for (size_t i = 0; i < data_.size(); ++i) data_[i] -= o.data_[i];
        return *this;
    }
    DynamicVector& operator*=(double s) {
        for (double& v : data_) v *= s;
        return *this;
    }
    DynamicVector& operator/=(double s) {
        double inv = 1.0 / s;
        for (double& v : data_) v *= inv;
        return *this;
    }

    friend DynamicVector operator*(double s, const DynamicVector& v) { return v * s; }

    // ── Element-wise multiply / divide ────────────────────────────────────
    DynamicVector hadamard(const DynamicVector& o) const {
        checkSize(o);
        DynamicVector r(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) r.data_[i] = data_[i] * o.data_[i];
        return r;
    }

    // ── Linear algebra ────────────────────────────────────────────────────

    double dot(const DynamicVector& o) const {
        checkSize(o);
        double s = 0.0;
        for (size_t i = 0; i < data_.size(); ++i) s += data_[i] * o.data_[i];
        return s;
    }

    double norm_sq() const { return dot(*this); }

    // p-norm: p=1 → L1, p=2 → L2 (default), p=inf → max|xi|
    double norm(double p = 2.0) const {
        if (data_.empty()) return 0.0;
        if (p == std::numeric_limits<double>::infinity() || p < 0) {
            double m = 0.0;
            for (double v : data_) m = std::max(m, std::abs(v));
            return m;
        }
        if (p == 1.0) {
            double s = 0.0;
            for (double v : data_) s += std::abs(v);
            return s;
        }
        if (p == 2.0) return std::sqrt(norm_sq());
        // General p-norm
        double s = 0.0;
        for (double v : data_) s += std::pow(std::abs(v), p);
        return std::pow(s, 1.0 / p);
    }

    double norm_inf() const { return norm(std::numeric_limits<double>::infinity()); }

    DynamicVector normalized() const {
        double n = norm();
        if (n < 1e-300)
            throw std::runtime_error("DynamicVector::normalized: zero vector");
        return *this * (1.0 / n);
    }

    // ── Reductions ────────────────────────────────────────────────────────
    double sum()  const { return std::accumulate(data_.begin(), data_.end(), 0.0); }
    double mean() const { return empty() ? 0.0 : sum() / static_cast<double>(data_.size()); }
    double min()  const { return *std::min_element(data_.begin(), data_.end()); }
    double max()  const { return *std::max_element(data_.begin(), data_.end()); }

    // ── Named constructors ────────────────────────────────────────────────
    static DynamicVector zeros(size_t n) { return DynamicVector(n, 0.0); }
    static DynamicVector ones (size_t n) { return DynamicVector(n, 1.0); }

    // Standard basis vector e_k (all zeros except position k = 1)
    static DynamicVector unit(size_t n, size_t k) {
        if (k >= n)
            throw std::out_of_range("DynamicVector::unit: k >= n");
        DynamicVector v(n, 0.0);
        v.data_[k] = 1.0;
        return v;
    }

    // ── Conversion to/from matrix ─────────────────────────────────────────
    // Defined in DynamicVector.cpp (includes DynamicMatrix.h to avoid cycle)
    DynamicMatrix to_column() const;   // n×1 DynamicMatrix
    DynamicMatrix to_row()    const;   // 1×n DynamicMatrix

    static DynamicVector from_column(const DynamicMatrix& A);
    static DynamicVector from_row   (const DynamicMatrix& A);

    // ── Display ───────────────────────────────────────────────────────────
    SHAREDMATH_LINEARALGEBRA_EXPORT
    friend std::ostream& operator<<(std::ostream& os, const DynamicVector& v);

private:
    std::vector<double> data_;

    void checkSize(const DynamicVector& o) const {
        if (data_.size() != o.data_.size())
            throw std::invalid_argument(
                "DynamicVector: size mismatch (" +
                std::to_string(data_.size()) + " vs " +
                std::to_string(o.data_.size()) + ")");
    }
};

// ── Matrix–vector free functions ──────────────────────────────────────────────
// Declared here, implemented in DynamicVector.cpp (which includes both headers).

// y = A * x  (rows(A) × 1 result)
SHAREDMATH_LINEARALGEBRA_EXPORT
DynamicVector matvec(const DynamicMatrix& A, const DynamicVector& x);

// y = A^T * x  (cols(A) × 1 result)
SHAREDMATH_LINEARALGEBRA_EXPORT
DynamicVector rmatvec(const DynamicMatrix& A, const DynamicVector& x);

// Outer product: u ⊗ v → DynamicMatrix (n×m)
SHAREDMATH_LINEARALGEBRA_EXPORT
DynamicMatrix outer(const DynamicVector& u, const DynamicVector& v);

SHAREDMATH_LINEARALGEBRA_EXPORT
DynamicVector operator*(const DynamicMatrix& A, const DynamicVector& x);

SHAREDMATH_LINEARALGEBRA_EXPORT
DynamicVector operator*(const DynamicVector& x, const DynamicMatrix& A);

} // namespace SharedMath::LinearAlgebra
