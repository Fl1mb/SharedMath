#pragma once

#include "constans.h"
#include <cmath>
#include <array>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <ostream>

namespace SharedMath::LinearAlgebra {

// Fixed-size N-dimensional vector with stack-allocated contiguous storage.
// Fully STL-compatible (iterators, data(), size()).
template<size_t N>
class Vector {
    static_assert(N > 0, "Vector size must be greater than zero");
public:
    using value_type      = double;
    using size_type       = size_t;
    using iterator        = double*;
    using const_iterator  = const double*;

    /// ── Construction ──────────────────────────────────────────────────────
    Vector() noexcept { data_.fill(0.0); }

    explicit Vector(double fill) noexcept { data_.fill(fill); }

    Vector(std::initializer_list<double> values) {
        if (values.size() != N)
            throw std::invalid_argument("Vector: initializer list size mismatch");
        std::copy(values.begin(), values.end(), data_.begin());
    }

    Vector(const Vector&)            = default;
    Vector(Vector&&) noexcept        = default;
    Vector& operator=(const Vector&) = default;
    Vector& operator=(Vector&&) noexcept = default;
    ~Vector() = default;

    // ── Element access ────────────────────────────────────────────────────
    double& operator[](size_t i) noexcept { return data_[i]; }
    const double& operator[](size_t i) const noexcept { return data_[i]; }

    double& at(size_t i) {
        if (i >= N) throw std::out_of_range("Vector::at: index out of range");
        return data_[i];
    }
    const double& at(size_t i) const {
        if (i >= N) throw std::out_of_range("Vector::at: index out of range");
        return data_[i];
    }

    double*       data() noexcept       { return data_.data(); }
    const double* data() const noexcept { return data_.data(); }

    /// ── STL iterators ─────────────────────────────────────────────────────
    iterator       begin()  noexcept       { return data_.data(); }
    iterator       end()    noexcept       { return data_.data() + N; }
    const_iterator begin()  const noexcept { return data_.data(); }
    const_iterator end()    const noexcept { return data_.data() + N; }
    const_iterator cbegin() const noexcept { return data_.data(); }
    const_iterator cend()   const noexcept { return data_.data() + N; }

    /// ── Metadata ──────────────────────────────────────────────────────────
    static constexpr size_t size()     noexcept { return N; }
    static constexpr bool   empty()    noexcept { return N == 0; }

    // ── Comparison ────────────────────────────────────────────────────────
    bool operator==(const Vector& o) const noexcept { return data_ == o.data_; }
    bool operator!=(const Vector& o) const noexcept { return data_ != o.data_; }

    // ── Arithmetic ────────────────────────────────────────────────────────
    Vector operator-() const noexcept {
        Vector r;
        for (size_t i = 0; i < N; ++i) r.data_[i] = -data_[i];
        return r;
    }

    Vector operator+(const Vector& o) const noexcept {
        Vector r;
        for (size_t i = 0; i < N; ++i) r.data_[i] = data_[i] + o.data_[i];
        return r;
    }
    Vector operator-(const Vector& o) const noexcept {
        Vector r;
        for (size_t i = 0; i < N; ++i) r.data_[i] = data_[i] - o.data_[i];
        return r;
    }
    Vector operator*(double s) const noexcept {
        Vector r;
        for (size_t i = 0; i < N; ++i) r.data_[i] = data_[i] * s;
        return r;
    }
    Vector operator/(double s) const {
        if (std::abs(s) < Epsilon)
            throw std::runtime_error("Vector: division by zero");
        double inv = 1.0 / s;
        Vector r;
        for (size_t i = 0; i < N; ++i) r.data_[i] = data_[i] * inv;
        return r;
    }

    Vector& operator+=(const Vector& o) noexcept {
        for (size_t i = 0; i < N; ++i) data_[i] += o.data_[i];
        return *this;
    }
    Vector& operator-=(const Vector& o) noexcept {
        for (size_t i = 0; i < N; ++i) data_[i] -= o.data_[i];
        return *this;
    }
    Vector& operator*=(double s) noexcept {
        for (double& v : data_) v *= s;
        return *this;
    }
    Vector& operator/=(double s) {
        *this = *this / s;
        return *this;
    }

    /// ── Linear algebra ────────────────────────────────────────────────────
    double dot(const Vector& o) const noexcept {
        double r = 0.0;
        for (size_t i = 0; i < N; ++i) r += data_[i] * o.data_[i];
        return r;
    }

    double norm_sq() const noexcept { return dot(*this); }
    double norm()    const noexcept { return std::sqrt(norm_sq()); }

    Vector normalized() const {
        double len = norm();
        if (len < Epsilon)
            throw std::runtime_error("Vector: cannot normalize zero vector");
        return (*this) * (1.0 / len);
    }

    // Cross product — available only for 3-D vectors (compile-time check)
    template<size_t M = N, typename = std::enable_if_t<M == 3>>
    Vector cross(const Vector& o) const noexcept {
        return { data_[1]*o.data_[2] - data_[2]*o.data_[1],
                 data_[2]*o.data_[0] - data_[0]*o.data_[2],
                 data_[0]*o.data_[1] - data_[1]*o.data_[0] };
    }

    /// ── Reductions ────────────────────────────────────────────────────────
    double sum()  const noexcept {
        return std::accumulate(data_.begin(), data_.end(), 0.0);
    }
    double mean() const noexcept { return sum() / static_cast<double>(N); }
    double min()  const noexcept {
        return *std::min_element(data_.begin(), data_.end());
    }
    double max()  const noexcept {
        return *std::max_element(data_.begin(), data_.end());
    }

    // ── Display ───────────────────────────────────────────────────────────
    friend std::ostream& operator<<(std::ostream& os, const Vector& v) {
        os << "[";
        for (size_t i = 0; i < N; ++i) {
            if (i) os << ", ";
            os << v.data_[i];
        }
        return os << "]";
    }

private:
    std::array<double, N> data_;
};

// ── Free functions ────────────────────────────────────────────────────────────
template<size_t N>
Vector<N> operator*(double s, const Vector<N>& v) noexcept { return v * s; }

} // namespace SharedMath::LinearAlgebra
