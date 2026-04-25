#include "Tensor.h"

#ifdef SHAREDMATH_CUDA
#  include "TensorCUDA.h"   // CUDA dispatch declarations (internal header)
#endif

#include <sstream>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <algorithm>
#include <stdexcept>

namespace SharedMath::LinearAlgebra {

// ─── size / flat (must work for both CPU and GPU tensors) ────────────────────

size_t Tensor::size() const noexcept {
    if (m_shape.empty()) return 0;
    size_t s = 1;
    for (size_t d : m_shape) s *= d;
    return s;
}

double& Tensor::flat(size_t i) {
    if (m_device != Device::CPU)
        throw std::runtime_error(
            "Tensor::flat: cannot access elements of a GPU tensor directly; "
            "call .cpu() first");
    return m_data[i];
}

double Tensor::flat(size_t i) const {
    if (m_device != Device::CPU)
        throw std::runtime_error(
            "Tensor::flat: cannot access elements of a GPU tensor directly; "
            "call .cpu() first");
    return m_data[i];
}

// ─── CPU stubs for cuda() / cpu() ────────────────────────────────────────────
// When CUDA is disabled these are no-ops.  The real implementations live in
// TensorCUDA.cu and are compiled only with -DSHAREDMATH_ENABLE_CUDA=ON.

#ifndef SHAREDMATH_CUDA
Tensor Tensor::cuda() const { return *this; }
Tensor Tensor::cpu()  const { return *this; }

// Private factory — only needed in TensorCUDA.cu; stub here to satisfy the
// linker on CPU-only builds (it should never actually be called).
Tensor Tensor::from_cuda(Shape /*shape*/,
                          std::shared_ptr<CUDABuffer> /*buf*/) {
    throw std::runtime_error("Tensor::from_cuda: CUDA support not compiled in");
}
#endif

// ─── construction ─────────────────────────────────────────────────────────────

Tensor::Tensor(Shape shape, double fill)
    : m_shape(std::move(shape))
{
    size_t total = 1;
    for (size_t d : m_shape) total *= d;
    m_data.assign(total, fill);
    computeStrides();
}

Tensor::Tensor(Shape shape, std::vector<double> data)
    : m_shape(std::move(shape)), m_data(std::move(data))
{
    size_t total = 1;
    for (size_t d : m_shape) total *= d;
    if (m_data.size() != total)
        throw std::invalid_argument("Tensor: data size does not match shape");
    computeStrides();
}

// ─── static factories ─────────────────────────────────────────────────────────

Tensor Tensor::zeros(Shape shape) { return Tensor(std::move(shape), 0.0); }
Tensor Tensor::ones(Shape shape)  { return Tensor(std::move(shape), 1.0); }

Tensor Tensor::eye(size_t n) {
    Tensor t({n, n}, 0.0);
    for (size_t i = 0; i < n; ++i) t(i, i) = 1.0;
    return t;
}

Tensor Tensor::arange(double start, double stop, double step) {
    if (step == 0.0)
        throw std::invalid_argument("Tensor::arange: step must not be zero");
    std::vector<double> v;
    v.reserve(static_cast<size_t>(std::abs((stop - start) / step)) + 1);
    for (double x = start; step > 0 ? x < stop : x > stop; x += step)
        v.push_back(x);
    size_t n = v.size();
    return Tensor({n}, std::move(v));
}

Tensor Tensor::linspace(double start, double stop, size_t num) {
    if (num == 0) return Tensor({0}, {});
    if (num == 1) return Tensor({1}, {start});
    std::vector<double> v(num);
    double step = (stop - start) / static_cast<double>(num - 1);
    for (size_t i = 0; i < num; ++i) v[i] = start + step * static_cast<double>(i);
    v.back() = stop;
    return Tensor({num}, std::move(v));
}

Tensor Tensor::from_vector(const std::vector<double>& v) {
    return Tensor({v.size()}, v);
}

Tensor Tensor::from_matrix(size_t rows, size_t cols,
                            const std::vector<double>& flat) {
    if (flat.size() != rows * cols)
        throw std::invalid_argument("Tensor::from_matrix: flat size != rows*cols");
    return Tensor({rows, cols}, flat);
}

// ─── stride helpers ───────────────────────────────────────────────────────────

void Tensor::computeStrides() {
    m_strides.resize(m_shape.size());
    if (m_shape.empty()) return;
    
    size_t stride = 1;
    // Идем с конца к началу
    for (int i = static_cast<int>(m_shape.size()) - 1; i >= 0; --i) {
        m_strides[i] = stride;      // Текущий stride
        stride *= m_shape[i];        // Умножаем для следующего измерения
    }
}

size_t Tensor::flatIndex(const std::vector<size_t>& idx) const {
    if (idx.size() != m_shape.size())
        throw std::invalid_argument(
            "Tensor: index rank " + std::to_string(idx.size()) +
            " does not match tensor rank " + std::to_string(m_shape.size()));
    size_t flat = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (idx[i] >= m_shape[i])
            throw std::out_of_range(
                "Tensor: index " + std::to_string(idx[i]) +
                " out of range for dim " + std::to_string(i) +
                " (size " + std::to_string(m_shape[i]) + ")");
        flat += idx[i] * m_strides[i];
    }
    return flat;
}

std::vector<size_t> Tensor::unravel(size_t flat) const {
    std::vector<size_t> idx(m_shape.size());
    for (size_t i = 0; i < m_shape.size(); ++i) {
        idx[i] = flat / m_strides[i];
        flat   %= m_strides[i];
    }
    return idx;
}

size_t Tensor::dim(size_t axis) const {
    if (axis >= m_shape.size())
        throw std::out_of_range("Tensor::dim: axis out of range");
    return m_shape[axis];
}

// ─── element access ──────────────────────────────────────────────────────────

double& Tensor::at(const std::vector<size_t>& idx)       { return m_data[flatIndex(idx)]; }
double  Tensor::at(const std::vector<size_t>& idx) const  { return m_data[flatIndex(idx)]; }

// ─── shape operations ────────────────────────────────────────────────────────

Tensor Tensor::reshape(Shape new_shape) const {
    size_t total = 1;
    for (size_t d : new_shape) total *= d;
    if (total != m_data.size())
        throw std::invalid_argument("Tensor::reshape: size mismatch");
    return Tensor(std::move(new_shape), m_data);
}

Tensor Tensor::flatten() const {
    return Tensor({m_data.size()}, m_data);
}

Tensor Tensor::squeeze() const {
    Shape s;
    for (size_t d : m_shape) if (d != 1) s.push_back(d);
    if (s.empty()) s.push_back(1);
    return Tensor(std::move(s), m_data);
}

Tensor Tensor::expand_dims(size_t axis) const {
    if (axis > m_shape.size())
        throw std::out_of_range("Tensor::expand_dims: axis out of range");
    Shape s = m_shape;
    s.insert(s.begin() + axis, 1);
    return Tensor(std::move(s), m_data);
}

Tensor Tensor::transpose(std::vector<size_t> axes) const {
    if (axes.size() != ndim())
        throw std::invalid_argument("Tensor::transpose: wrong number of axes");
    {
        std::vector<size_t> sorted = axes;
        std::sort(sorted.begin(), sorted.end());
        for (size_t i = 0; i < sorted.size(); ++i)
            if (sorted[i] != i)
                throw std::invalid_argument("Tensor::transpose: invalid permutation");
    }
    Shape new_shape(ndim());
    for (size_t i = 0; i < ndim(); ++i) new_shape[i] = m_shape[axes[i]];
    Tensor result(new_shape);
    for (size_t flat = 0; flat < size(); ++flat) {
        auto src = unravel(flat);
        std::vector<size_t> dst(ndim());
        for (size_t i = 0; i < ndim(); ++i) dst[i] = src[axes[i]];
        result.at(dst) = m_data[flat];
    }
    return result;
}

Tensor Tensor::transpose() const {
    if (ndim() < 2) return *this;
    std::vector<size_t> axes(ndim());
    std::iota(axes.begin(), axes.end(), 0);
    std::reverse(axes.begin(), axes.end());
    return transpose(axes);
}

Tensor Tensor::slice(size_t axis, size_t start, size_t end) const {
    if (axis >= ndim())
        throw std::out_of_range("Tensor::slice: axis out of range");
    if (start >= end || end > m_shape[axis])
        throw std::invalid_argument("Tensor::slice: invalid range");
    Shape new_shape = m_shape;
    new_shape[axis] = end - start;
    Tensor result(new_shape);
    for (size_t flat = 0; flat < result.size(); ++flat) {
        auto dst = result.unravel(flat);
        auto src = dst;
        src[axis] += start;
        result.m_data[flat] = at(src);
    }
    return result;
}

// ─── broadcasting ────────────────────────────────────────────────────────────

Tensor::Shape Tensor::broadcastShape(const Shape& a, const Shape& b) {
    size_t n = std::max(a.size(), b.size());
    Shape result(n);
    for (size_t i = 0; i < n; ++i) {
        size_t da = (i < n - a.size()) ? 1 : a[i - (n - a.size())];
        size_t db = (i < n - b.size()) ? 1 : b[i - (n - b.size())];
        if (da != db && da != 1 && db != 1)
            throw std::invalid_argument("Tensor: shapes are not broadcastable");
        result[i] = std::max(da, db);
    }
    return result;
}

Tensor Tensor::broadcastOp(const Tensor& other,
                            std::function<double(double, double)> op) const
{
    Shape rshape = broadcastShape(m_shape, other.m_shape);
    Tensor result(rshape);
    size_t nr = rshape.size();
    size_t na = ndim(), nb = other.ndim();

    for (size_t flat = 0; flat < result.size(); ++flat) {
        auto ridx = result.unravel(flat);

        std::vector<size_t> aidx(na), bidx(nb);
        for (size_t i = 0; i < na; ++i) {
            size_t ri = ridx[nr - na + i];
            aidx[i] = (m_shape[i] == 1) ? 0 : ri;
        }
        for (size_t i = 0; i < nb; ++i) {
            size_t ri = ridx[nr - nb + i];
            bidx[i] = (other.m_shape[i] == 1) ? 0 : ri;
        }
        result.m_data[flat] = op(m_data[flatIndex(aidx)],
                                 other.m_data[other.flatIndex(bidx)]);
    }
    return result;
}

// ─── arithmetic operators ─────────────────────────────────────────────────────

// ── CUDA binary dispatch macro ────────────────────────────────────────────────
// Same-shape → CUDA kernel.  Different shapes (broadcast) → CPU fallback.
// Mixed device → error.
#ifdef SHAREDMATH_CUDA
#define CUDA_BINARY_DISPATCH(OP_ENUM, CPU_EXPR)                             \
    if (m_device != o.m_device)                                             \
        throw std::runtime_error(                                           \
            "Tensor: binary op between tensors on different devices");      \
    if (m_device == Device::CUDA) {                                         \
        if (m_shape == o.m_shape)                                           \
            return detail::cuda_binary(*this, o, detail::BinaryOp::OP_ENUM);\
        return cpu().CPU_EXPR(o.cpu()); /* broadcast → CPU fallback */      \
    }
    /* If cuda() returned CPU (no GPU present) — fall through to CPU path */
#else
#define CUDA_BINARY_DISPATCH(OP_ENUM, CPU_EXPR)
#endif

Tensor Tensor::operator+(const Tensor& o) const {
    CUDA_BINARY_DISPATCH(Add, operator+)
    return broadcastOp(o, [](double a, double b) { return a + b; });
}
Tensor Tensor::operator-(const Tensor& o) const {
    CUDA_BINARY_DISPATCH(Sub, operator-)
    return broadcastOp(o, [](double a, double b) { return a - b; });
}
Tensor Tensor::operator*(const Tensor& o) const {
    CUDA_BINARY_DISPATCH(Mul, operator*)
    return broadcastOp(o, [](double a, double b) { return a * b; });
}
Tensor Tensor::operator/(const Tensor& o) const {
    CUDA_BINARY_DISPATCH(Div, operator/)
    return broadcastOp(o, [](double a, double b) { return a / b; });
}

#undef CUDA_BINARY_DISPATCH

Tensor Tensor::operator+(double s) const { return apply([s](double x){ return x + s; }); }
Tensor Tensor::operator-(double s) const { return apply([s](double x){ return x - s; }); }
Tensor Tensor::operator*(double s) const { return apply([s](double x){ return x * s; }); }
Tensor Tensor::operator/(double s) const { return apply([s](double x){ return x / s; }); }
Tensor Tensor::operator-()         const { return apply([](double x){ return -x;     }); }

Tensor& Tensor::operator+=(const Tensor& o) { *this = *this + o; return *this; }
Tensor& Tensor::operator-=(const Tensor& o) { *this = *this - o; return *this; }
Tensor& Tensor::operator*=(double s) { for (auto& v : m_data) v *= s; return *this; }
Tensor& Tensor::operator/=(double s) { for (auto& v : m_data) v /= s; return *this; }

bool Tensor::operator==(const Tensor& o) const {
    return m_shape == o.m_shape && m_data == o.m_data;
}
bool Tensor::operator!=(const Tensor& o) const { return !(*this == o); }

// ─── global reductions ────────────────────────────────────────────────────────

double Tensor::sum() const {
    return std::accumulate(m_data.begin(), m_data.end(), 0.0);
}
double Tensor::product() const {
    return std::accumulate(m_data.begin(), m_data.end(), 1.0,
                           [](double a, double b){ return a * b; });
}
double Tensor::min() const {
    if (m_data.empty()) throw std::runtime_error("Tensor::min: empty tensor");
    return *std::min_element(m_data.begin(), m_data.end());
}
double Tensor::max() const {
    if (m_data.empty()) throw std::runtime_error("Tensor::max: empty tensor");
    return *std::max_element(m_data.begin(), m_data.end());
}
double Tensor::mean() const {
    return sum() / static_cast<double>(size());
}
double Tensor::var(bool ddof) const {
    double m = mean();
    double acc = 0.0;
    for (double v : m_data) acc += (v - m) * (v - m);
    double denom = static_cast<double>(size() - (ddof ? 1 : 0));
    if (denom <= 0.0) throw std::runtime_error("Tensor::var: not enough elements");
    return acc / denom;
}
double Tensor::stddev(bool ddof) const { return std::sqrt(var(ddof)); }

size_t Tensor::argmin() const {
    if (m_data.empty()) throw std::runtime_error("Tensor::argmin: empty tensor");
    return static_cast<size_t>(
        std::min_element(m_data.begin(), m_data.end()) - m_data.begin());
}
size_t Tensor::argmax() const {
    if (m_data.empty()) throw std::runtime_error("Tensor::argmax: empty tensor");
    return static_cast<size_t>(
        std::max_element(m_data.begin(), m_data.end()) - m_data.begin());
}

// ─── axis reductions ─────────────────────────────────────────────────────────

Tensor Tensor::axisReduce(size_t axis,
                           std::function<double(double, double)> reducer,
                           double init) const
{
    if (axis >= ndim()) throw std::out_of_range("Tensor: axis out of range");
    Shape rshape = m_shape;
    rshape.erase(rshape.begin() + axis);
    if (rshape.empty()) rshape.push_back(1);
    Tensor result(rshape, init);
    for (size_t flat = 0; flat < size(); ++flat) {
        auto idx = unravel(flat);
        std::vector<size_t> ridx = idx;
        ridx.erase(ridx.begin() + axis);
        if (ridx.empty()) ridx.push_back(0);
        result.at(ridx) = reducer(result.at(ridx), m_data[flat]);
    }
    return result;
}

Tensor Tensor::sum(size_t axis) const {
    return axisReduce(axis, [](double a, double b){ return a + b; }, 0.0);
}
Tensor Tensor::min(size_t axis) const {
    return axisReduce(axis,
        [](double a, double b){ return std::min(a, b); },
        std::numeric_limits<double>::infinity());
}
Tensor Tensor::max(size_t axis) const {
    return axisReduce(axis,
        [](double a, double b){ return std::max(a, b); },
        -std::numeric_limits<double>::infinity());
}
Tensor Tensor::mean(size_t axis) const {
    Tensor s = sum(axis);
    s /= static_cast<double>(m_shape[axis]);
    return s;
}

// ─── element-wise math ────────────────────────────────────────────────────────

// apply(f) is CPU-only (std::function can't be passed to a CUDA kernel).
// If the tensor is on GPU, it is first brought to CPU.
Tensor Tensor::apply(std::function<double(double)> f) const {
#ifdef SHAREDMATH_CUDA
    if (m_device == Device::CUDA) return cpu().apply(f);
#endif
    Tensor result(m_shape);
    for (size_t i = 0; i < m_data.size(); ++i) result.m_data[i] = f(m_data[i]);
    return result;
}

// Named unary ops: GPU-accelerated via dedicated CUDA kernels when on device.
#ifdef SHAREDMATH_CUDA
#define CUDA_UNARY(OP_ENUM, CPU_EXPR)                                       \
    if (m_device == Device::CUDA)                                           \
        return detail::cuda_unary(*this, detail::UnaryOp::OP_ENUM);        \
    return CPU_EXPR;
#else
#define CUDA_UNARY(OP_ENUM, CPU_EXPR) return CPU_EXPR;
#endif

Tensor Tensor::abs()   const { CUDA_UNARY(Abs,   apply([](double x){ return std::abs(x);   })) }
Tensor Tensor::sqrt()  const { CUDA_UNARY(Sqrt,  apply([](double x){ return std::sqrt(x);  })) }
Tensor Tensor::exp()   const { CUDA_UNARY(Exp,   apply([](double x){ return std::exp(x);   })) }
Tensor Tensor::log()   const { CUDA_UNARY(Log,   apply([](double x){ return std::log(x);   })) }
Tensor Tensor::log2()  const { CUDA_UNARY(Log2,  apply([](double x){ return std::log2(x);  })) }
Tensor Tensor::log10() const { CUDA_UNARY(Log10, apply([](double x){ return std::log10(x); })) }
Tensor Tensor::sin()   const { CUDA_UNARY(Sin,   apply([](double x){ return std::sin(x);   })) }
Tensor Tensor::cos()   const { CUDA_UNARY(Cos,   apply([](double x){ return std::cos(x);   })) }
Tensor Tensor::tanh()  const { CUDA_UNARY(Tanh,  apply([](double x){ return std::tanh(x);  })) }
Tensor Tensor::floor() const { CUDA_UNARY(Floor, apply([](double x){ return std::floor(x); })) }
Tensor Tensor::ceil()  const { CUDA_UNARY(Ceil,  apply([](double x){ return std::ceil(x);  })) }
Tensor Tensor::round() const { CUDA_UNARY(Round, apply([](double x){ return std::round(x); })) }
Tensor Tensor::sign()  const {
    CUDA_UNARY(Sign, apply([](double x){ return x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0); }))
}

#undef CUDA_UNARY

Tensor Tensor::pow(double e) const {
#ifdef SHAREDMATH_CUDA
    if (m_device == Device::CUDA) return detail::cuda_pow(*this, e);
#endif
    return apply([e](double x){ return std::pow(x, e); });
}
Tensor Tensor::clip(double lo, double hi) const {
#ifdef SHAREDMATH_CUDA
    if (m_device == Device::CUDA) return detail::cuda_clip(*this, lo, hi);
#endif
    return apply([lo, hi](double x){ return std::clamp(x, lo, hi); });
}

// ─── linear algebra ──────────────────────────────────────────────────────────

Tensor Tensor::matmul(const Tensor& other) const {
    if (ndim() != 2 || other.ndim() != 2)
        throw std::invalid_argument("Tensor::matmul: both tensors must be 2-D");
    size_t m = m_shape[0], k = m_shape[1], n = other.m_shape[1];
    if (k != other.m_shape[0])
        throw std::invalid_argument(
            "Tensor::matmul: inner dimensions mismatch (" +
            std::to_string(k) + " vs " + std::to_string(other.m_shape[0]) + ")");

#ifdef SHAREDMATH_CUDA
    if (m_device != other.m_device)
        throw std::runtime_error("Tensor::matmul: tensors must be on the same device");
    if (m_device == Device::CUDA)
        return detail::cuda_matmul(*this, other);
#endif

    // CPU path — cache-friendly i-l-j loop
    Tensor result({m, n}, 0.0);
    for (size_t i = 0; i < m; ++i)
        for (size_t l = 0; l < k; ++l) {
            double a = (*this)(i, l);
            for (size_t j = 0; j < n; ++j)
                result(i, j) += a * other(l, j);
        }
    return result;
}

double Tensor::trace() const {
    if (ndim() != 2 || m_shape[0] != m_shape[1])
        throw std::invalid_argument("Tensor::trace: requires square 2-D tensor");
    double s = 0.0;
    for (size_t i = 0; i < m_shape[0]; ++i) s += (*this)(i, i);
    return s;
}

Tensor Tensor::diag() const {
    if (ndim() == 1) {
        size_t n = m_shape[0];
        Tensor result({n, n}, 0.0);
        for (size_t i = 0; i < n; ++i) result(i, i) = m_data[i];
        return result;
    }
    if (ndim() == 2) {
        size_t n = std::min(m_shape[0], m_shape[1]);
        Tensor result({n});
        for (size_t i = 0; i < n; ++i) result(i) = (*this)(i, i);
        return result;
    }
    throw std::invalid_argument("Tensor::diag: requires 1-D or 2-D tensor");
}

// ─── string representation ────────────────────────────────────────────────────

std::string Tensor::str() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);

    oss << "Tensor(shape=[";
    for (size_t i = 0; i < m_shape.size(); ++i) {
        if (i) oss << ", ";
        oss << m_shape[i];
    }
    oss << "])";

    if (empty()) return oss.str();

    if (ndim() == 1) {
        oss << "\n[";
        for (size_t i = 0; i < m_data.size(); ++i) {
            if (i) oss << ", ";
            oss << m_data[i];
        }
        oss << "]";
    } else if (ndim() == 2) {
        size_t rows = m_shape[0], cols = m_shape[1];
        oss << "\n[";
        for (size_t i = 0; i < rows; ++i) {
            if (i) oss << " ";
            oss << "[";
            for (size_t j = 0; j < cols; ++j) {
                if (j) oss << ", ";
                oss << (*this)(i, j);
            }
            oss << "]";
            if (i + 1 < rows) oss << "\n";
        }
        oss << "]";
    } else {
        // N-D: print flat with shape info
        oss << "\n[";
        for (size_t i = 0; i < m_data.size(); ++i) {
            if (i) oss << ", ";
            oss << m_data[i];
        }
        oss << "]";
    }
    return oss.str();
}

// ─── scalar-on-left operators ────────────────────────────────────────────────

Tensor operator+(double s, const Tensor& t) { return t + s; }
Tensor operator-(double s, const Tensor& t) { return (-t) + s; }
Tensor operator*(double s, const Tensor& t) { return t * s; }
Tensor operator/(double s, const Tensor& t) {
    return t.apply([s](double x){ return s / x; });
}

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    return os << t.str();
}

} // namespace SharedMath::LinearAlgebra
