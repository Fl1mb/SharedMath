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
#include <random>
#include <cstdint>

namespace SharedMath::LinearAlgebra {

namespace {

size_t shapeSize(const Tensor::Shape& shape) {
    if (shape.empty()) return 0;
    size_t total = 1;
    for (size_t d : shape) total *= d;
    return total;
}

Tensor::Shape shapeWithoutAxis(const Tensor::Shape& shape, size_t axis) {
    Tensor::Shape out = shape;
    out.erase(out.begin() + static_cast<std::ptrdiff_t>(axis));
    if (out.empty()) out.push_back(1);
    return out;
}

std::mt19937_64 makeGenerator(std::uint64_t seed) {
    if (seed != 0) return std::mt19937_64(seed);
    std::random_device rd;
    return std::mt19937_64((static_cast<std::uint64_t>(rd()) << 32) ^ rd());
}

size_t mlOutDim(size_t input, size_t kernel, size_t stride,
                size_t padding, const char* name) {
    if (kernel == 0 || stride == 0)
        throw std::invalid_argument(std::string(name) + ": kernel and stride must be > 0");
    size_t padded = input + 2 * padding;
    if (padded < kernel)
        throw std::invalid_argument(std::string(name) + ": kernel larger than padded input");
    return (padded - kernel) / stride + 1;
}

} // namespace

// ─── TensorView ─────────────────────────────────────────────────────────────

TensorView::TensorView(Tensor* base, Shape shape, Shape strides, size_t offset)
    : m_base(base),
      m_shape(std::move(shape)),
      m_strides(std::move(strides)),
      m_offset(offset)
{}

size_t TensorView::size() const noexcept {
    if (m_shape.empty()) return 0;
    size_t total = 1;
    for (size_t d : m_shape) total *= d;
    return total;
}

size_t TensorView::dim(size_t axis) const {
    if (axis >= m_shape.size())
        throw std::out_of_range("TensorView::dim: axis out of range");
    return m_shape[axis];
}

std::vector<size_t> TensorView::unravel(size_t flat) const {
    std::vector<size_t> idx(m_shape.size());
    for (int i = static_cast<int>(m_shape.size()) - 1; i >= 0; --i) {
        idx[static_cast<size_t>(i)] = flat % m_shape[static_cast<size_t>(i)];
        flat /= m_shape[static_cast<size_t>(i)];
    }
    return idx;
}

size_t TensorView::physicalIndex(const std::vector<size_t>& idx) const {
    if (!m_base)
        throw std::runtime_error("TensorView: empty view");
    if (idx.size() != m_shape.size())
        throw std::invalid_argument("TensorView: index rank does not match view rank");
    size_t flat = m_offset;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (idx[i] >= m_shape[i])
            throw std::out_of_range("TensorView: index out of range");
        flat += idx[i] * m_strides[i];
    }
    return flat;
}

double& TensorView::at(const std::vector<size_t>& idx) {
    return m_base->m_data[physicalIndex(idx)];
}

double TensorView::at(const std::vector<size_t>& idx) const {
    return m_base->m_data[physicalIndex(idx)];
}

double& TensorView::flat(size_t logical_flat) {
    if (logical_flat >= size())
        throw std::out_of_range("TensorView::flat: index out of range");
    return at(unravel(logical_flat));
}

double TensorView::flat(size_t logical_flat) const {
    if (logical_flat >= size())
        throw std::out_of_range("TensorView::flat: index out of range");
    return at(unravel(logical_flat));
}

Tensor TensorView::to_tensor() const {
    Tensor out(m_shape);
    for (size_t i = 0; i < out.size(); ++i)
        out.flat(i) = flat(i);
    return out;
}

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
Tensor Tensor::cuda(int /*device_id*/) const { return *this; }
Tensor Tensor::cuda_auto() const { return *this; }
Tensor Tensor::cpu()  const { return *this; }
Tensor Tensor::to(Device /*device*/, int /*device_id*/) const { return *this; }

// Private factory — only needed in TensorCUDA.cu; stub here to satisfy the
// linker on CPU-only builds (it should never actually be called).
Tensor Tensor::from_cuda(Shape /*shape*/,
                          std::shared_ptr<CUDABuffer> /*buf*/,
                          int /*device_id*/) {
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

Tensor::Tensor(Shape shape, std::vector<double> data, TensorDType dtype)
    : Tensor(std::move(shape), std::move(data))
{
    m_dtype = dtype;
    if (m_dtype == TensorDType::Float32) {
        for (double& v : m_data)
            v = static_cast<double>(static_cast<float>(v));
    }
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

Tensor Tensor::uniform(Shape shape, double low, double high, std::uint64_t seed) {
    if (low > high)
        throw std::invalid_argument("Tensor::uniform: low must be <= high");
    std::vector<double> values(shapeSize(shape));
    auto gen = makeGenerator(seed);
    std::uniform_real_distribution<double> dist(low, high);
    for (double& v : values) v = dist(gen);
    return Tensor(std::move(shape), std::move(values));
}

Tensor Tensor::normal(Shape shape, double mean, double stddev, std::uint64_t seed) {
    if (stddev < 0.0)
        throw std::invalid_argument("Tensor::normal: stddev must be non-negative");
    std::vector<double> values(shapeSize(shape));
    auto gen = makeGenerator(seed);
    std::normal_distribution<double> dist(mean, stddev);
    for (double& v : values) v = dist(gen);
    return Tensor(std::move(shape), std::move(values));
}

Tensor Tensor::randn(Shape shape, std::uint64_t seed) {
    return normal(std::move(shape), 0.0, 1.0, seed);
}

Tensor Tensor::bernoulli(Shape shape, double p, std::uint64_t seed) {
    if (p < 0.0 || p > 1.0)
        throw std::invalid_argument("Tensor::bernoulli: p must be in [0, 1]");
    std::vector<double> values(shapeSize(shape));
    auto gen = makeGenerator(seed);
    std::bernoulli_distribution dist(p);
    for (double& v : values) v = dist(gen) ? 1.0 : 0.0;
    return Tensor(std::move(shape), std::move(values));
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

Tensor Tensor::concat(const std::vector<Tensor>& tensors, size_t axis) {
    if (tensors.empty())
        throw std::invalid_argument("Tensor::concat: no tensors");
    const Tensor& first = tensors.front();
    if (axis >= first.ndim())
        throw std::out_of_range("Tensor::concat: axis out of range");

    Shape out_shape = first.shape();
    out_shape[axis] = 0;
    for (const Tensor& t : tensors) {
        if (t.ndim() != first.ndim())
            throw std::invalid_argument("Tensor::concat: ranks differ");
        for (size_t i = 0; i < first.ndim(); ++i) {
            if (i != axis && t.dim(i) != first.dim(i))
                throw std::invalid_argument("Tensor::concat: non-concat dimensions differ");
        }
        out_shape[axis] += t.dim(axis);
    }

    Tensor out(out_shape);
    size_t axis_offset = 0;
    for (const Tensor& t : tensors) {
        for (size_t flat = 0; flat < t.size(); ++flat) {
            auto idx = t.unravel(flat);
            auto out_idx = idx;
            out_idx[axis] += axis_offset;
            out.at(out_idx) = t.flat(flat);
        }
        axis_offset += t.dim(axis);
    }
    return out;
}

Tensor Tensor::stack(const std::vector<Tensor>& tensors, size_t axis) {
    if (tensors.empty())
        throw std::invalid_argument("Tensor::stack: no tensors");
    const Shape& base_shape = tensors.front().shape();
    if (axis > base_shape.size())
        throw std::out_of_range("Tensor::stack: axis out of range");
    for (const Tensor& t : tensors) {
        if (t.shape() != base_shape)
            throw std::invalid_argument("Tensor::stack: all shapes must match");
    }

    Shape out_shape = base_shape;
    out_shape.insert(out_shape.begin() + static_cast<std::ptrdiff_t>(axis),
                     tensors.size());
    Tensor out(out_shape);
    for (size_t n = 0; n < tensors.size(); ++n) {
        const Tensor& t = tensors[n];
        for (size_t flat = 0; flat < t.size(); ++flat) {
            auto idx = t.unravel(flat);
            idx.insert(idx.begin() + static_cast<std::ptrdiff_t>(axis), n);
            out.at(idx) = t.flat(flat);
        }
    }
    return out;
}

Tensor Tensor::where(const Tensor& condition, const Tensor& x, const Tensor& y) {
    Shape out_shape = broadcastShape(broadcastShape(condition.shape(), x.shape()), y.shape());
    Tensor c = condition.broadcast_to(out_shape);
    Tensor xb = x.broadcast_to(out_shape);
    Tensor yb = y.broadcast_to(out_shape);
    Tensor out(out_shape);
    for (size_t i = 0; i < out.size(); ++i)
        out.flat(i) = c.flat(i) != 0.0 ? xb.flat(i) : yb.flat(i);
    return out;
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
    if (total != size())
        throw std::invalid_argument("Tensor::reshape: size mismatch");
#ifdef SHAREDMATH_CUDA
    if (m_device == Device::CUDA)
        return from_cuda(std::move(new_shape), m_cuda_buf, m_device_id);
#endif
    return Tensor(std::move(new_shape), m_data);
}

Tensor Tensor::view(Shape new_shape) const {
    return reshape(std::move(new_shape));
}

Tensor Tensor::flatten() const {
#ifdef SHAREDMATH_CUDA
    if (m_device == Device::CUDA)
        return from_cuda({size()}, m_cuda_buf, m_device_id);
#endif
    return Tensor({m_data.size()}, m_data);
}

Tensor Tensor::squeeze() const {
    Shape s;
    for (size_t d : m_shape) if (d != 1) s.push_back(d);
    if (s.empty()) s.push_back(1);
    return Tensor(std::move(s), m_data);
}

Tensor Tensor::squeeze(size_t axis) const {
    if (axis >= ndim())
        throw std::out_of_range("Tensor::squeeze: axis out of range");
    if (m_shape[axis] != 1)
        throw std::invalid_argument("Tensor::squeeze: selected axis is not size 1");
    Shape s = m_shape;
    s.erase(s.begin() + static_cast<std::ptrdiff_t>(axis));
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

Tensor Tensor::unsqueeze(size_t axis) const {
    return expand_dims(axis);
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

Tensor Tensor::permute(std::vector<size_t> axes) const {
    return transpose(std::move(axes));
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

TensorView Tensor::slice_view(size_t axis, size_t start, size_t end) {
    if (m_device != Device::CPU)
        throw std::runtime_error("Tensor::slice_view: only CPU tensors can be viewed");
    if (axis >= ndim())
        throw std::out_of_range("Tensor::slice_view: axis out of range");
    if (start >= end || end > m_shape[axis])
        throw std::invalid_argument("Tensor::slice_view: invalid range");
    Shape view_shape = m_shape;
    view_shape[axis] = end - start;
    size_t offset = start * m_strides[axis];
    return TensorView(this, std::move(view_shape), m_strides, offset);
}

Tensor Tensor::broadcast_to(Shape target_shape) const {
#ifdef SHAREDMATH_CUDA
    if (m_device == Device::CUDA) return cpu().broadcast_to(std::move(target_shape));
#endif
    Shape actual = broadcastShape(m_shape, target_shape);
    if (actual != target_shape)
        throw std::invalid_argument("Tensor::broadcast_to: target shape is not compatible");

    Tensor result(target_shape);
    const size_t nr = target_shape.size();
    const size_t ns = ndim();
    for (size_t flat = 0; flat < result.size(); ++flat) {
        auto ridx = result.unravel(flat);
        std::vector<size_t> sidx(ns);
        for (size_t i = 0; i < ns; ++i) {
            size_t ri = ridx[nr - ns + i];
            sidx[i] = (m_shape[i] == 1) ? 0 : ri;
        }
        result.flat(flat) = at(sidx);
    }
    return result;
}

std::vector<Tensor> Tensor::split(size_t axis, const std::vector<size_t>& sections) const {
    if (axis >= ndim())
        throw std::out_of_range("Tensor::split: axis out of range");
    size_t total = std::accumulate(sections.begin(), sections.end(), size_t{0});
    if (total != m_shape[axis])
        throw std::invalid_argument("Tensor::split: sections do not sum to axis length");

    std::vector<Tensor> parts;
    parts.reserve(sections.size());
    size_t start = 0;
    for (size_t len : sections) {
        parts.push_back(slice(axis, start, start + len));
        start += len;
    }
    return parts;
}

std::vector<Tensor> Tensor::split(size_t axis, size_t chunk_size) const {
    if (axis >= ndim())
        throw std::out_of_range("Tensor::split: axis out of range");
    if (chunk_size == 0)
        throw std::invalid_argument("Tensor::split: chunk_size must be positive");
    std::vector<Tensor> parts;
    for (size_t start = 0; start < m_shape[axis]; start += chunk_size) {
        size_t end = std::min(start + chunk_size, m_shape[axis]);
        parts.push_back(slice(axis, start, end));
    }
    return parts;
}

Tensor Tensor::astype(TensorDType dtype) const {
#ifdef SHAREDMATH_CUDA
    if (m_device == Device::CUDA) return cpu().astype(dtype);
#endif
    Tensor out = *this;
    out.m_dtype = dtype;
    if (dtype == TensorDType::Float32) {
        for (double& v : out.m_data)
            v = static_cast<double>(static_cast<float>(v));
    }
    return out;
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
    if (m_device == Device::CUDA && m_device_id != o.m_device_id)           \
        throw std::runtime_error(                                           \
            "Tensor: binary op between tensors on different CUDA devices"); \
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

#ifdef SHAREDMATH_CUDA
#define CUDA_SCALAR_DISPATCH(OP_ENUM, CPU_EXPR)                             \
    if (m_device == Device::CUDA)                                           \
        return detail::cuda_scalar(*this, s, detail::ScalarOp::OP_ENUM);    \
    return CPU_EXPR;
#else
#define CUDA_SCALAR_DISPATCH(OP_ENUM, CPU_EXPR) return CPU_EXPR;
#endif

Tensor Tensor::operator+(double s) const {
    CUDA_SCALAR_DISPATCH(Add, apply([s](double x){ return x + s; }))
}
Tensor Tensor::operator-(double s) const {
    CUDA_SCALAR_DISPATCH(Sub, apply([s](double x){ return x - s; }))
}
Tensor Tensor::operator*(double s) const {
    CUDA_SCALAR_DISPATCH(Mul, apply([s](double x){ return x * s; }))
}
Tensor Tensor::operator/(double s) const {
    CUDA_SCALAR_DISPATCH(Div, apply([s](double x){ return x / s; }))
}

#undef CUDA_SCALAR_DISPATCH

Tensor Tensor::operator-() const {
#ifdef SHAREDMATH_CUDA
    if (m_device == Device::CUDA)
        return detail::cuda_unary(*this, detail::UnaryOp::Neg);
#endif
    return apply([](double x){ return -x; });
}

Tensor& Tensor::operator+=(const Tensor& o) { *this = *this + o; return *this; }
Tensor& Tensor::operator-=(const Tensor& o) { *this = *this - o; return *this; }
Tensor& Tensor::operator*=(double s) { *this = *this * s; return *this; }
Tensor& Tensor::operator/=(double s) { *this = *this / s; return *this; }

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

Tensor Tensor::var(int axis, bool ddof) const {
    if (axis < 0)
        throw std::out_of_range("Tensor::var: axis out of range");
    return var(static_cast<size_t>(axis), ddof);
}

Tensor Tensor::var(size_t axis, bool ddof) const {
    if (axis >= ndim()) throw std::out_of_range("Tensor::var: axis out of range");
    size_t axis_len = m_shape[axis];
    size_t correction = ddof ? 1 : 0;
    if (axis_len <= correction)
        throw std::runtime_error("Tensor::var: not enough elements");

    Shape rshape = shapeWithoutAxis(m_shape, axis);
    Tensor means = mean(axis);
    Tensor out(rshape, 0.0);
    for (size_t flat = 0; flat < size(); ++flat) {
        auto idx = unravel(flat);
        auto ridx = idx;
        ridx.erase(ridx.begin() + static_cast<std::ptrdiff_t>(axis));
        if (ridx.empty()) ridx.push_back(0);
        double diff = m_data[flat] - means.at(ridx);
        out.at(ridx) += diff * diff;
    }
    out /= static_cast<double>(axis_len - correction);
    return out;
}

Tensor Tensor::argmin(size_t axis) const {
    if (axis >= ndim()) throw std::out_of_range("Tensor::argmin: axis out of range");
    Shape rshape = shapeWithoutAxis(m_shape, axis);
    Tensor values(rshape, std::numeric_limits<double>::infinity());
    Tensor indices(rshape, 0.0);
    for (size_t flat = 0; flat < size(); ++flat) {
        auto idx = unravel(flat);
        auto ridx = idx;
        size_t axis_idx = idx[axis];
        ridx.erase(ridx.begin() + static_cast<std::ptrdiff_t>(axis));
        if (ridx.empty()) ridx.push_back(0);
        if (m_data[flat] < values.at(ridx)) {
            values.at(ridx) = m_data[flat];
            indices.at(ridx) = static_cast<double>(axis_idx);
        }
    }
    return indices;
}

Tensor Tensor::argmax(size_t axis) const {
    if (axis >= ndim()) throw std::out_of_range("Tensor::argmax: axis out of range");
    Shape rshape = shapeWithoutAxis(m_shape, axis);
    Tensor values(rshape, -std::numeric_limits<double>::infinity());
    Tensor indices(rshape, 0.0);
    for (size_t flat = 0; flat < size(); ++flat) {
        auto idx = unravel(flat);
        auto ridx = idx;
        size_t axis_idx = idx[axis];
        ridx.erase(ridx.begin() + static_cast<std::ptrdiff_t>(axis));
        if (ridx.empty()) ridx.push_back(0);
        if (m_data[flat] > values.at(ridx)) {
            values.at(ridx) = m_data[flat];
            indices.at(ridx) = static_cast<double>(axis_idx);
        }
    }
    return indices;
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
Tensor Tensor::relu()  const { CUDA_UNARY(Relu,  apply([](double x){ return x > 0.0 ? x : 0.0; })) }
Tensor Tensor::sigmoid() const {
    CUDA_UNARY(Sigmoid, apply([](double x){ return 1.0 / (1.0 + std::exp(-x)); }))
}
Tensor Tensor::gelu() const {
    CUDA_UNARY(Gelu, apply([](double x){
        return x * 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    }))
}
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

Tensor Tensor::softmax(size_t axis) const {
#ifdef SHAREDMATH_CUDA
    if (m_device == Device::CUDA) return detail::cuda_softmax(*this, axis);
#endif
    if (axis >= ndim())
        throw std::out_of_range("Tensor::softmax: axis out of range");
    Shape rshape = shapeWithoutAxis(m_shape, axis);
    Tensor maxes(rshape, -std::numeric_limits<double>::infinity());
    for (size_t flat = 0; flat < size(); ++flat) {
        auto idx = unravel(flat);
        auto ridx = idx;
        ridx.erase(ridx.begin() + static_cast<std::ptrdiff_t>(axis));
        if (ridx.empty()) ridx.push_back(0);
        maxes.at(ridx) = std::max(maxes.at(ridx), m_data[flat]);
    }

    Tensor sums(rshape, 0.0);
    Tensor out(m_shape);
    for (size_t flat = 0; flat < size(); ++flat) {
        auto idx = unravel(flat);
        auto ridx = idx;
        ridx.erase(ridx.begin() + static_cast<std::ptrdiff_t>(axis));
        if (ridx.empty()) ridx.push_back(0);
        double e = std::exp(m_data[flat] - maxes.at(ridx));
        out.flat(flat) = e;
        sums.at(ridx) += e;
    }
    for (size_t flat = 0; flat < out.size(); ++flat) {
        auto idx = out.unravel(flat);
        auto ridx = idx;
        ridx.erase(ridx.begin() + static_cast<std::ptrdiff_t>(axis));
        if (ridx.empty()) ridx.push_back(0);
        out.flat(flat) /= sums.at(ridx);
    }
    return out;
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
    if (m_device == Device::CUDA && m_device_id != other.m_device_id)
        throw std::runtime_error("Tensor::matmul: tensors must be on the same CUDA device");
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

Tensor Tensor::conv2d(const Tensor& weight, const Tensor* bias,
                      size_t stride, size_t padding) const {
    if (ndim() != 4 || weight.ndim() != 4)
        throw std::invalid_argument("Tensor::conv2d: input and weight must be 4-D");
    if (dim(1) != weight.dim(1))
        throw std::invalid_argument("Tensor::conv2d: input channel mismatch");
    if (bias && (bias->ndim() != 1 || bias->dim(0) != weight.dim(0)))
        throw std::invalid_argument("Tensor::conv2d: bias must have shape [out_channels]");
    if (stride == 0)
        throw std::invalid_argument("Tensor::conv2d: stride must be > 0");

#ifdef SHAREDMATH_CUDA
    if (m_device != weight.m_device || (bias && bias->m_device != m_device))
        throw std::runtime_error("Tensor::conv2d: tensors must be on the same device");
    if (m_device == Device::CUDA &&
        (m_device_id != weight.m_device_id || (bias && bias->m_device_id != m_device_id)))
        throw std::runtime_error("Tensor::conv2d: tensors must be on the same CUDA device");
    if (m_device == Device::CUDA)
        return detail::cuda_conv2d(*this, weight, bias, stride, padding);
#endif

    const size_t N = dim(0), C = dim(1), H = dim(2), W = dim(3);
    const size_t OC = weight.dim(0), K = weight.dim(2);
    if (weight.dim(3) != K)
        throw std::invalid_argument("Tensor::conv2d: only square kernels are supported");
    const size_t OH = mlOutDim(H, K, stride, padding, "Tensor::conv2d");
    const size_t OW = mlOutDim(W, K, stride, padding, "Tensor::conv2d");

    Tensor out({N, OC, OH, OW}, 0.0);
    for (size_t n = 0; n < N; ++n)
        for (size_t oc = 0; oc < OC; ++oc)
            for (size_t oh = 0; oh < OH; ++oh)
                for (size_t ow = 0; ow < OW; ++ow) {
                    double sum = bias ? bias->flat(oc) : 0.0;
                    for (size_t ic = 0; ic < C; ++ic)
                        for (size_t kh = 0; kh < K; ++kh) {
                            long ih = static_cast<long>(oh * stride + kh) -
                                      static_cast<long>(padding);
                            if (ih < 0 || ih >= static_cast<long>(H)) continue;
                            for (size_t kw = 0; kw < K; ++kw) {
                                long iw = static_cast<long>(ow * stride + kw) -
                                          static_cast<long>(padding);
                                if (iw < 0 || iw >= static_cast<long>(W)) continue;
                                sum += (*this)(n, ic, static_cast<size_t>(ih),
                                               static_cast<size_t>(iw)) *
                                       weight(oc, ic, kh, kw);
                            }
                        }
                    out(n, oc, oh, ow) = sum;
                }
    return out;
}

Tensor Tensor::conv2d_backward_input(const Tensor& grad_out,
                                     const Tensor& weight,
                                     Shape input_shape,
                                     size_t stride,
                                     size_t padding) {
    if (grad_out.ndim() != 4 || weight.ndim() != 4 || input_shape.size() != 4)
        throw std::invalid_argument("Tensor::conv2d_backward_input: invalid ranks");
    if (stride == 0)
        throw std::invalid_argument("Tensor::conv2d_backward_input: stride must be > 0");

#ifdef SHAREDMATH_CUDA
    if (grad_out.m_device != weight.m_device)
        throw std::runtime_error("Tensor::conv2d_backward_input: tensors must be on the same device");
    if (grad_out.m_device == Device::CUDA && grad_out.m_device_id != weight.m_device_id)
        throw std::runtime_error("Tensor::conv2d_backward_input: tensors must be on the same CUDA device");
    if (grad_out.m_device == Device::CUDA)
        return detail::cuda_conv2d_backward_input(grad_out, weight, std::move(input_shape), stride, padding);
#endif

    const size_t N = input_shape[0], C = input_shape[1], H = input_shape[2], W = input_shape[3];
    const size_t OC = weight.dim(0), K = weight.dim(2);
    const size_t OH = grad_out.dim(2), OW = grad_out.dim(3);
    Tensor dx(input_shape, 0.0);
    for (size_t n = 0; n < N; ++n)
        for (size_t oc = 0; oc < OC; ++oc)
            for (size_t oh = 0; oh < OH; ++oh)
                for (size_t ow = 0; ow < OW; ++ow) {
                    double g = grad_out(n, oc, oh, ow);
                    for (size_t ic = 0; ic < C; ++ic)
                        for (size_t kh = 0; kh < K; ++kh) {
                            long ih = static_cast<long>(oh * stride + kh) -
                                      static_cast<long>(padding);
                            if (ih < 0 || ih >= static_cast<long>(H)) continue;
                            for (size_t kw = 0; kw < K; ++kw) {
                                long iw = static_cast<long>(ow * stride + kw) -
                                          static_cast<long>(padding);
                                if (iw < 0 || iw >= static_cast<long>(W)) continue;
                                dx(n, ic, static_cast<size_t>(ih), static_cast<size_t>(iw)) +=
                                    g * weight(oc, ic, kh, kw);
                            }
                        }
                }
    return dx;
}

Tensor Tensor::conv2d_backward_weight(const Tensor& input,
                                      Shape weight_shape,
                                      size_t stride,
                                      size_t padding) const {
    if (ndim() != 4 || input.ndim() != 4 || weight_shape.size() != 4)
        throw std::invalid_argument("Tensor::conv2d_backward_weight: invalid ranks");
    if (stride == 0)
        throw std::invalid_argument("Tensor::conv2d_backward_weight: stride must be > 0");

#ifdef SHAREDMATH_CUDA
    if (m_device != input.m_device)
        throw std::runtime_error("Tensor::conv2d_backward_weight: tensors must be on the same device");
    if (m_device == Device::CUDA && m_device_id != input.m_device_id)
        throw std::runtime_error("Tensor::conv2d_backward_weight: tensors must be on the same CUDA device");
    if (m_device == Device::CUDA)
        return detail::cuda_conv2d_backward_weight(*this, input, std::move(weight_shape), stride, padding);
#endif

    const size_t N = input.dim(0), C = input.dim(1), H = input.dim(2), W = input.dim(3);
    const size_t OC = weight_shape[0], K = weight_shape[2];
    const size_t OH = dim(2), OW = dim(3);
    Tensor dw(weight_shape, 0.0);
    for (size_t n = 0; n < N; ++n)
        for (size_t oc = 0; oc < OC; ++oc)
            for (size_t oh = 0; oh < OH; ++oh)
                for (size_t ow = 0; ow < OW; ++ow) {
                    double g = (*this)(n, oc, oh, ow);
                    for (size_t ic = 0; ic < C; ++ic)
                        for (size_t kh = 0; kh < K; ++kh) {
                            long ih = static_cast<long>(oh * stride + kh) -
                                      static_cast<long>(padding);
                            if (ih < 0 || ih >= static_cast<long>(H)) continue;
                            for (size_t kw = 0; kw < K; ++kw) {
                                long iw = static_cast<long>(ow * stride + kw) -
                                          static_cast<long>(padding);
                                if (iw < 0 || iw >= static_cast<long>(W)) continue;
                                dw(oc, ic, kh, kw) += g * input(n, ic, static_cast<size_t>(ih),
                                                               static_cast<size_t>(iw));
                            }
                        }
                }
    return dw;
}

Tensor Tensor::conv2d_backward_bias() const {
    if (ndim() != 4)
        throw std::invalid_argument("Tensor::conv2d_backward_bias: grad_out must be 4-D");
#ifdef SHAREDMATH_CUDA
    if (m_device == Device::CUDA)
        return detail::cuda_conv2d_backward_bias(*this);
#endif
    Tensor db({dim(1)}, 0.0);
    for (size_t n = 0; n < dim(0); ++n)
        for (size_t oc = 0; oc < dim(1); ++oc)
            for (size_t oh = 0; oh < dim(2); ++oh)
                for (size_t ow = 0; ow < dim(3); ++ow)
                    db.flat(oc) += (*this)(n, oc, oh, ow);
    return db;
}

Tensor Tensor::max_pool2d(size_t kernel_size, size_t stride, size_t padding) const {
    if (ndim() != 4)
        throw std::invalid_argument("Tensor::max_pool2d: input must be 4-D NCHW");
    if (stride == 0) stride = kernel_size;
#ifdef SHAREDMATH_CUDA
    if (m_device == Device::CUDA)
        return detail::cuda_max_pool2d(*this, kernel_size, stride, padding);
#endif
    const size_t N = dim(0), C = dim(1), H = dim(2), W = dim(3);
    const size_t OH = mlOutDim(H, kernel_size, stride, padding, "Tensor::max_pool2d");
    const size_t OW = mlOutDim(W, kernel_size, stride, padding, "Tensor::max_pool2d");
    Tensor out({N, C, OH, OW}, 0.0);
    for (size_t n = 0; n < N; ++n)
        for (size_t c = 0; c < C; ++c)
            for (size_t oh = 0; oh < OH; ++oh)
                for (size_t ow = 0; ow < OW; ++ow) {
                    double best = -std::numeric_limits<double>::infinity();
                    for (size_t kh = 0; kh < kernel_size; ++kh) {
                        long ih = static_cast<long>(oh * stride + kh) - static_cast<long>(padding);
                        if (ih < 0 || ih >= static_cast<long>(H)) continue;
                        for (size_t kw = 0; kw < kernel_size; ++kw) {
                            long iw = static_cast<long>(ow * stride + kw) - static_cast<long>(padding);
                            if (iw < 0 || iw >= static_cast<long>(W)) continue;
                            best = std::max(best, (*this)(n, c, static_cast<size_t>(ih), static_cast<size_t>(iw)));
                        }
                    }
                    out(n, c, oh, ow) = best;
                }
    return out;
}

Tensor Tensor::max_pool2d_backward(const Tensor& input, size_t kernel_size,
                                   size_t stride, size_t padding) const {
    if (ndim() != 4 || input.ndim() != 4)
        throw std::invalid_argument("Tensor::max_pool2d_backward: tensors must be 4-D");
    if (stride == 0) stride = kernel_size;
#ifdef SHAREDMATH_CUDA
    if (m_device != input.m_device)
        throw std::runtime_error("Tensor::max_pool2d_backward: tensors must be on the same device");
    if (m_device == Device::CUDA && m_device_id != input.m_device_id)
        throw std::runtime_error("Tensor::max_pool2d_backward: tensors must be on the same CUDA device");
    if (m_device == Device::CUDA)
        return detail::cuda_max_pool2d_backward(*this, input, kernel_size, stride, padding);
#endif
    const size_t N = input.dim(0), C = input.dim(1), H = input.dim(2), W = input.dim(3);
    const size_t OH = dim(2), OW = dim(3);
    Tensor dx(input.shape(), 0.0);
    for (size_t n = 0; n < N; ++n)
        for (size_t c = 0; c < C; ++c)
            for (size_t oh = 0; oh < OH; ++oh)
                for (size_t ow = 0; ow < OW; ++ow) {
                    double best = -std::numeric_limits<double>::infinity();
                    size_t bi = 0, bj = 0;
                    for (size_t kh = 0; kh < kernel_size; ++kh) {
                        long ih = static_cast<long>(oh * stride + kh) - static_cast<long>(padding);
                        if (ih < 0 || ih >= static_cast<long>(H)) continue;
                        for (size_t kw = 0; kw < kernel_size; ++kw) {
                            long iw = static_cast<long>(ow * stride + kw) - static_cast<long>(padding);
                            if (iw < 0 || iw >= static_cast<long>(W)) continue;
                            double v = input(n, c, static_cast<size_t>(ih), static_cast<size_t>(iw));
                            if (v > best) { best = v; bi = static_cast<size_t>(ih); bj = static_cast<size_t>(iw); }
                        }
                    }
                    dx(n, c, bi, bj) += (*this)(n, c, oh, ow);
                }
    return dx;
}

Tensor Tensor::avg_pool2d(size_t kernel_size, size_t stride, size_t padding) const {
    if (ndim() != 4)
        throw std::invalid_argument("Tensor::avg_pool2d: input must be 4-D NCHW");
    if (stride == 0) stride = kernel_size;
#ifdef SHAREDMATH_CUDA
    if (m_device == Device::CUDA)
        return detail::cuda_avg_pool2d(*this, kernel_size, stride, padding);
#endif
    const size_t N = dim(0), C = dim(1), H = dim(2), W = dim(3);
    const size_t OH = mlOutDim(H, kernel_size, stride, padding, "Tensor::avg_pool2d");
    const size_t OW = mlOutDim(W, kernel_size, stride, padding, "Tensor::avg_pool2d");
    Tensor out({N, C, OH, OW}, 0.0);
    for (size_t n = 0; n < N; ++n)
        for (size_t c = 0; c < C; ++c)
            for (size_t oh = 0; oh < OH; ++oh)
                for (size_t ow = 0; ow < OW; ++ow) {
                    double sum = 0.0, count = 0.0;
                    for (size_t kh = 0; kh < kernel_size; ++kh) {
                        long ih = static_cast<long>(oh * stride + kh) - static_cast<long>(padding);
                        if (ih < 0 || ih >= static_cast<long>(H)) continue;
                        for (size_t kw = 0; kw < kernel_size; ++kw) {
                            long iw = static_cast<long>(ow * stride + kw) - static_cast<long>(padding);
                            if (iw < 0 || iw >= static_cast<long>(W)) continue;
                            sum += (*this)(n, c, static_cast<size_t>(ih), static_cast<size_t>(iw));
                            count += 1.0;
                        }
                    }
                    out(n, c, oh, ow) = count > 0.0 ? sum / count : 0.0;
                }
    return out;
}

Tensor Tensor::avg_pool2d_backward(Shape input_shape, size_t kernel_size,
                                   size_t stride, size_t padding) const {
    if (ndim() != 4 || input_shape.size() != 4)
        throw std::invalid_argument("Tensor::avg_pool2d_backward: invalid ranks");
    if (stride == 0) stride = kernel_size;
#ifdef SHAREDMATH_CUDA
    if (m_device == Device::CUDA)
        return detail::cuda_avg_pool2d_backward(*this, std::move(input_shape), kernel_size, stride, padding);
#endif
    const size_t N = input_shape[0], C = input_shape[1], H = input_shape[2], W = input_shape[3];
    const size_t OH = dim(2), OW = dim(3);
    Tensor dx(input_shape, 0.0);
    for (size_t n = 0; n < N; ++n)
        for (size_t c = 0; c < C; ++c)
            for (size_t oh = 0; oh < OH; ++oh)
                for (size_t ow = 0; ow < OW; ++ow) {
                    double count = 0.0;
                    for (size_t kh = 0; kh < kernel_size; ++kh) {
                        long ih = static_cast<long>(oh * stride + kh) - static_cast<long>(padding);
                        if (ih < 0 || ih >= static_cast<long>(H)) continue;
                        for (size_t kw = 0; kw < kernel_size; ++kw) {
                            long iw = static_cast<long>(ow * stride + kw) - static_cast<long>(padding);
                            if (iw < 0 || iw >= static_cast<long>(W)) continue;
                            count += 1.0;
                        }
                    }
                    double share = count > 0.0 ? (*this)(n, c, oh, ow) / count : 0.0;
                    for (size_t kh = 0; kh < kernel_size; ++kh) {
                        long ih = static_cast<long>(oh * stride + kh) - static_cast<long>(padding);
                        if (ih < 0 || ih >= static_cast<long>(H)) continue;
                        for (size_t kw = 0; kw < kernel_size; ++kw) {
                            long iw = static_cast<long>(ow * stride + kw) - static_cast<long>(padding);
                            if (iw < 0 || iw >= static_cast<long>(W)) continue;
                            dx(n, c, static_cast<size_t>(ih), static_cast<size_t>(iw)) += share;
                        }
                    }
                }
    return dx;
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
#ifdef SHAREDMATH_CUDA
    if (t.device() == Device::CUDA)
        return detail::cuda_scalar(t, s, detail::ScalarOp::RDiv);
#endif
    return t.apply([s](double x){ return s / x; });
}

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    return os << t.str();
}

} // namespace SharedMath::LinearAlgebra
