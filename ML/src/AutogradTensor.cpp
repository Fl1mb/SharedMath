#include "AutogradTensor.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace SharedMath::ML {

using Tensor = AutoTensor::Tensor;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers (file-local)
// ─────────────────────────────────────────────────────────────────────────────

// Reduce gradient `g` to match `target_shape` by summing over broadcast dims.
static Tensor reduceGradToShape(const Tensor& g,
                                 const Tensor::Shape& target)
{
    if (g.shape() == target) return g;

    Tensor result = g;

    // Remove extra leading dimensions by summing axis 0 repeatedly.
    while (result.ndim() > target.size()) {
        result = result.sum(0);
    }

    // Sum singleton-broadcast dimensions within the same rank.
    for (size_t i = 0; i < target.size(); ++i) {
        if (i < result.ndim() && target[i] == 1 && result.dim(i) > 1) {
            result = result.sum(i).expand_dims(i);
        }
    }

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// AutoTensor::Impl
// ─────────────────────────────────────────────────────────────────────────────

void AutoTensor::Impl::accumulate_grad(const Tensor& g) {
    if (!has_grad_) {
        grad     = Tensor::zeros(data.shape());
        has_grad_ = true;
    }
    grad += g;
}

void AutoTensor::Impl::propagate(const Tensor& upstream) {
    if (is_leaf) {
        if (requires_grad)
            accumulate_grad(upstream);
    } else {
        if (grad_fn)
            grad_fn(upstream);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AutoTensor construction
// ─────────────────────────────────────────────────────────────────────────────

AutoTensor::AutoTensor()
    : m_impl(std::make_shared<Impl>())
{}

AutoTensor::AutoTensor(Tensor data, bool requires_grad)
    : m_impl(std::make_shared<Impl>())
{
    m_impl->data          = std::move(data);
    m_impl->requires_grad = requires_grad;
}

AutoTensor AutoTensor::from(Tensor data, bool requires_grad) {
    return AutoTensor(std::move(data), requires_grad);
}

AutoTensor AutoTensor::make_result(Tensor data,
                                   bool requires_grad,
                                   std::function<void(const Tensor&)> grad_fn)
{
    AutoTensor t(std::move(data), requires_grad);
    t.m_impl->is_leaf  = false;
    t.m_impl->grad_fn  = std::move(grad_fn);
    return t;
}

// ─────────────────────────────────────────────────────────────────────────────
// Data & gradient accessors
// ─────────────────────────────────────────────────────────────────────────────

const Tensor& AutoTensor::data() const noexcept { return m_impl->data; }
Tensor&       AutoTensor::data()       noexcept { return m_impl->data; }

bool AutoTensor::requires_grad() const noexcept { return m_impl->requires_grad; }
void AutoTensor::set_requires_grad(bool v) noexcept { m_impl->requires_grad = v; }

bool AutoTensor::has_grad() const noexcept { return m_impl && m_impl->has_grad_; }

const Tensor& AutoTensor::grad() const {
    if (!m_impl->has_grad_)
        throw std::runtime_error("AutoTensor::grad: gradient is not set");
    return m_impl->grad;
}

Tensor& AutoTensor::grad() {
    if (!m_impl->has_grad_)
        throw std::runtime_error("AutoTensor::grad: gradient is not set");
    return m_impl->grad;
}

void AutoTensor::set_grad(Tensor g) {
    if (g.shape() != m_impl->data.shape())
        throw std::invalid_argument("AutoTensor::set_grad: gradient shape mismatch");
    m_impl->grad      = std::move(g);
    m_impl->has_grad_ = true;
}

void AutoTensor::zero_grad() {
    m_impl->grad      = Tensor::zeros(m_impl->data.shape());
    m_impl->has_grad_ = true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Backward
// ─────────────────────────────────────────────────────────────────────────────

void AutoTensor::backward() {
    if (m_impl->data.size() != 1)
        throw std::runtime_error(
            "AutoTensor::backward(): tensor must be scalar (size == 1). "
            "Call backward(upstream_grad) for non-scalar outputs.");
    backward(Tensor::ones({1}));
}

void AutoTensor::backward(Tensor upstream) {
    m_impl->propagate(upstream);
}

// ─────────────────────────────────────────────────────────────────────────────
// Arithmetic — tensor × tensor
// ─────────────────────────────────────────────────────────────────────────────

AutoTensor AutoTensor::operator+(const AutoTensor& o) const {
    bool ng = m_impl->requires_grad || o.m_impl->requires_grad;
    auto out_data = m_impl->data + o.m_impl->data;
    if (!ng) return AutoTensor(out_data);

    auto si = m_impl, oi = o.m_impl;
    return make_result(std::move(out_data), true,
        [si, oi](const Tensor& g) {
            si->propagate(reduceGradToShape(g, si->data.shape()));
            oi->propagate(reduceGradToShape(g, oi->data.shape()));
        });
}

AutoTensor AutoTensor::operator-(const AutoTensor& o) const {
    bool ng = m_impl->requires_grad || o.m_impl->requires_grad;
    auto out_data = m_impl->data - o.m_impl->data;
    if (!ng) return AutoTensor(out_data);

    auto si = m_impl, oi = o.m_impl;
    return make_result(std::move(out_data), true,
        [si, oi](const Tensor& g) {
            si->propagate(reduceGradToShape(g, si->data.shape()));
            oi->propagate(reduceGradToShape(-g, oi->data.shape()));
        });
}

AutoTensor AutoTensor::operator*(const AutoTensor& o) const {
    bool ng = m_impl->requires_grad || o.m_impl->requires_grad;
    auto out_data = m_impl->data * o.m_impl->data;
    if (!ng) return AutoTensor(out_data);

    auto si = m_impl, oi = o.m_impl;
    return make_result(std::move(out_data), true,
        [si, oi](const Tensor& g) {
            si->propagate(reduceGradToShape(g * oi->data, si->data.shape()));
            oi->propagate(reduceGradToShape(g * si->data, oi->data.shape()));
        });
}

AutoTensor AutoTensor::operator/(const AutoTensor& o) const {
    bool ng = m_impl->requires_grad || o.m_impl->requires_grad;
    auto out_data = m_impl->data / o.m_impl->data;
    if (!ng) return AutoTensor(out_data);

    auto si = m_impl, oi = o.m_impl;
    return make_result(std::move(out_data), true,
        [si, oi](const Tensor& g) {
            // d/dx (x/y) = 1/y
            si->propagate(reduceGradToShape(g / oi->data, si->data.shape()));
            // d/dy (x/y) = -x/y²
            Tensor gy = (g * si->data) / (oi->data * oi->data);
            oi->propagate(reduceGradToShape(-gy, oi->data.shape()));
        });
}

AutoTensor AutoTensor::operator-() const {
    auto out_data = -m_impl->data;
    if (!m_impl->requires_grad) return AutoTensor(out_data);

    auto si = m_impl;
    return make_result(std::move(out_data), true,
        [si](const Tensor& g) { si->propagate(-g); });
}

// ─────────────────────────────────────────────────────────────────────────────
// Arithmetic — tensor × scalar
// ─────────────────────────────────────────────────────────────────────────────

AutoTensor AutoTensor::operator+(double s) const {
    auto out_data = m_impl->data + s;
    if (!m_impl->requires_grad) return AutoTensor(out_data);
    auto si = m_impl;
    return make_result(std::move(out_data), true,
        [si](const Tensor& g) { si->propagate(g); });
}

AutoTensor AutoTensor::operator-(double s) const {
    return *this + (-s);
}

AutoTensor AutoTensor::operator*(double s) const {
    auto out_data = m_impl->data * s;
    if (!m_impl->requires_grad) return AutoTensor(out_data);
    auto si = m_impl;
    return make_result(std::move(out_data), true,
        [si, s](const Tensor& g) { si->propagate(g * s); });
}

AutoTensor AutoTensor::operator/(double s) const {
    return *this * (1.0 / s);
}

AutoTensor operator+(double s, const AutoTensor& t) { return t + s; }
AutoTensor operator*(double s, const AutoTensor& t) { return t * s; }

// ─────────────────────────────────────────────────────────────────────────────
// Linear algebra
// ─────────────────────────────────────────────────────────────────────────────

AutoTensor AutoTensor::matmul(const AutoTensor& o) const {
    if (m_impl->data.ndim() != 2 || o.m_impl->data.ndim() != 2)
        throw std::invalid_argument(
            "AutoTensor::matmul: both tensors must be 2-D");

    bool ng = m_impl->requires_grad || o.m_impl->requires_grad;
    auto out_data = m_impl->data.matmul(o.m_impl->data);
    if (!ng) return AutoTensor(out_data);

    auto si = m_impl, oi = o.m_impl;
    return make_result(std::move(out_data), true,
        [si, oi](const Tensor& g) {
            // g: [M,P], A: [M,N], B: [N,P]
            // dA = g @ B^T → [M,N]
            // dB = A^T @ g → [N,P]
            if (si->requires_grad || !si->is_leaf)
                si->propagate(g.matmul(oi->data.transpose()));
            if (oi->requires_grad || !oi->is_leaf)
                oi->propagate(si->data.transpose().matmul(g));
        });
}

AutoTensor AutoTensor::T() const {
    if (m_impl->data.ndim() != 2)
        throw std::invalid_argument("AutoTensor::T(): tensor must be 2-D");
    auto out_data = m_impl->data.transpose();
    if (!m_impl->requires_grad) return AutoTensor(out_data);
    auto si = m_impl;
    return make_result(std::move(out_data), true,
        [si](const Tensor& g) { si->propagate(g.transpose()); });
}

// ─────────────────────────────────────────────────────────────────────────────
// Reductions → scalar node (shape {1})
// ─────────────────────────────────────────────────────────────────────────────

AutoTensor AutoTensor::sum() const {
    double s = m_impl->data.sum();
    Tensor out_data = Tensor::from_vector({s});
    if (!m_impl->requires_grad) return AutoTensor(out_data);

    auto si = m_impl;
    return make_result(std::move(out_data), true,
        [si](const Tensor& g) {
            double upstream = g.flat(0);
            si->propagate(Tensor(si->data.shape(), upstream));
        });
}

AutoTensor AutoTensor::mean() const {
    double m   = m_impl->data.mean();
    double n   = static_cast<double>(m_impl->data.size());
    Tensor out_data = Tensor::from_vector({m});
    if (!m_impl->requires_grad) return AutoTensor(out_data);

    auto si = m_impl;
    return make_result(std::move(out_data), true,
        [si, n](const Tensor& g) {
            double upstream = g.flat(0);
            si->propagate(Tensor(si->data.shape(), upstream / n));
        });
}

// ─────────────────────────────────────────────────────────────────────────────
// Element-wise activations
// ─────────────────────────────────────────────────────────────────────────────

AutoTensor AutoTensor::relu() const {
    auto out_data = m_impl->data.apply([](double v) { return v > 0.0 ? v : 0.0; });
    if (!m_impl->requires_grad) return AutoTensor(out_data);

    auto si = m_impl;
    return make_result(std::move(out_data), true,
        [si](const Tensor& g) {
            Tensor mask = si->data.apply([](double v) { return v > 0.0 ? 1.0 : 0.0; });
            si->propagate(g * mask);
        });
}

AutoTensor AutoTensor::sigmoid() const {
    auto out_data = m_impl->data.apply(
        [](double v) { return 1.0 / (1.0 + std::exp(-v)); });
    if (!m_impl->requires_grad) return AutoTensor(out_data);

    Tensor s_copy = out_data; // capture sigmoid output for backward
    auto si = m_impl;
    return make_result(std::move(out_data), true,
        [si, s_copy](const Tensor& g) {
            // σ'(x) = σ(x)(1 − σ(x))
            Tensor dsig = s_copy * s_copy.apply([](double v) { return 1.0 - v; });
            si->propagate(g * dsig);
        });
}

AutoTensor AutoTensor::tanh() const {
    auto out_data = m_impl->data.tanh();
    if (!m_impl->requires_grad) return AutoTensor(out_data);

    Tensor t_copy = out_data;
    auto si = m_impl;
    return make_result(std::move(out_data), true,
        [si, t_copy](const Tensor& g) {
            // tanh'(x) = 1 − tanh²(x)
            Tensor dtanh = t_copy.apply([](double v) { return 1.0 - v * v; });
            si->propagate(g * dtanh);
        });
}

AutoTensor AutoTensor::exp() const {
    auto out_data = m_impl->data.exp();
    if (!m_impl->requires_grad) return AutoTensor(out_data);

    Tensor e_copy = out_data; // exp(x) needed for backward
    auto si = m_impl;
    return make_result(std::move(out_data), true,
        [si, e_copy](const Tensor& g) { si->propagate(g * e_copy); });
}

AutoTensor AutoTensor::log() const {
    auto out_data = m_impl->data.log();
    if (!m_impl->requires_grad) return AutoTensor(out_data);

    auto si = m_impl;
    return make_result(std::move(out_data), true,
        [si](const Tensor& g) {
            // d log(x)/dx = 1/x
            si->propagate(g / si->data);
        });
}

AutoTensor AutoTensor::pow(double exponent) const {
    auto out_data = m_impl->data.pow(exponent);
    if (!m_impl->requires_grad) return AutoTensor(out_data);

    auto si = m_impl;
    return make_result(std::move(out_data), true,
        [si, exponent](const Tensor& g) {
            // d x^n/dx = n * x^(n-1)
            si->propagate(g * si->data.pow(exponent - 1.0) * exponent);
        });
}

} // namespace SharedMath::ML
