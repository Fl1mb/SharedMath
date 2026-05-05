/**
 * @file Activations.h
 * @brief ML activation functions (sigmoid, ReLU, GELU, Swish, softmax, …).
 * @ingroup Functions_Activations
 *
 * All activations are templated on @p T (`float` or `double`).
 * Each activation also provides a `*Prime` variant (analytical derivative).
 * Vector versions of softmax and log-softmax operate on `std::vector<T>`.
 *
 * References:
 *   - Agostinelli et al. (2014)
 *   - Clevert et al. (2016) — ELU
 *   - Hendrycks & Gimpel (2016) — GELU
 *   - Ramachandran et al. (2017) — Swish / SiLU
 *   - Misra (2019) — Mish
 */

/**
 * @defgroup Functions_Activations Activation Functions
 * @ingroup Functions
 * @brief Neural-network activation functions templated on `float` or `double`.
 */
#pragma once

#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <limits>

namespace SharedMath::Functions {

/// ─────────────────────────────────────────────────────────────────────────────
/// Helper constants
/// ─────────────────────────────────────────────────────────────────────────────
namespace detail {
template<typename T> constexpr T ACT_PI   = static_cast<T>(3.14159265358979323846);
template<typename T> constexpr T ACT_SQRT2 = static_cast<T>(1.41421356237309504880);
// √(2/π) — used in GELU approximation
template<typename T> constexpr T SQRT2OPI = static_cast<T>(0.79788456080286535588);
} // namespace detail


/// ═════════════════════════════════════════════════════════════════════════════
/// SIGMOID  σ(x) = 1 / (1 + e^{−x})
/// ═════════════════════════════════════════════════════════════════════════════

template<typename T> inline T sigmoid(T x) {
    return T(1) / (T(1) + std::exp(-x));
}
/// σ'(x) = σ(x)(1 − σ(x))
template<typename T> inline T sigmoidPrime(T x) {
    T s = sigmoid(x);
    return s * (T(1) - s);
}
/// Numerically stable log-sigmoid: log(σ(x)) = −softplus(−x)
template<typename T> inline T logSigmoid(T x) {
    return (x >= T(0)) ? -std::log(T(1) + std::exp(-x))
                       :  x - std::log(T(1) + std::exp(x));
}


/// ═════════════════════════════════════════════════════════════════════════════
/// ReLU  max(0, x)
/// ═════════════════════════════════════════════════════════════════════════════

template<typename T> inline T relu(T x) { return x > T(0) ? x : T(0); }
/// Subgradient at 0 → 0
template<typename T> inline T reluPrime(T x) { return x > T(0) ? T(1) : T(0); }


/// ═════════════════════════════════════════════════════════════════════════════
/// Leaky ReLU  max(alpha·x, x)
/// ═════════════════════════════════════════════════════════════════════════════

template<typename T> inline T leakyRelu(T x, T alpha = T(0.01)) {
    return x > T(0) ? x : alpha * x;
}
template<typename T> inline T leakyReluPrime(T x, T alpha = T(0.01)) {
    return x > T(0) ? T(1) : alpha;
}


/// ═════════════════════════════════════════════════════════════════════════════
/// ELU — Exponential Linear Unit
/// f(x) = x          if x > 0
///       = α(e^x − 1) if x ≤ 0
/// ═════════════════════════════════════════════════════════════════════════════

template<typename T> inline T elu(T x, T alpha = T(1)) {
    return x > T(0) ? x : alpha * (std::exp(x) - T(1));
}
template<typename T> inline T eluPrime(T x, T alpha = T(1)) {
    return x > T(0) ? T(1) : alpha * std::exp(x);
}


/// ═════════════════════════════════════════════════════════════════════════════
/// SELU — Scaled ELU  (Klambauer et al., 2017)
/// Self-normalising: preserves mean≈0, var≈1 for standard-normal inputs.
/// ═════════════════════════════════════════════════════════════════════════════

namespace detail {
constexpr double SELU_ALPHA  = 1.6732632423543772848170429916717;
constexpr double SELU_LAMBDA = 1.0507009873554804934193349852946;
}

template<typename T> inline T selu(T x) {
    constexpr T alpha  = static_cast<T>(detail::SELU_ALPHA);
    constexpr T lambda = static_cast<T>(detail::SELU_LAMBDA);
    return lambda * (x > T(0) ? x : alpha * (std::exp(x) - T(1)));
}
template<typename T> inline T seluPrime(T x) {
    constexpr T alpha  = static_cast<T>(detail::SELU_ALPHA);
    constexpr T lambda = static_cast<T>(detail::SELU_LAMBDA);
    return lambda * (x > T(0) ? T(1) : alpha * std::exp(x));
}


/// ═════════════════════════════════════════════════════════════════════════════
/// GELU — Gaussian Error Linear Unit  (Hendrycks & Gimpel, 2016)
/// Exact:          gelu(x) = x · Φ(x) = x/2 · (1 + erf(x/√2))
/// Fast approx:    gelu(x) ≈ x/2 · (1 + tanh(√(2/π) · (x + 0.044715·x³)))
/// ═════════════════════════════════════════════════════════════════════════════

template<typename T> inline T gelu(T x) {
    return x * T(0.5) * (T(1) + std::erf(x / detail::ACT_SQRT2<T>));
}
/// Approximation — slightly faster, avoids erf
template<typename T> inline T geluApprox(T x) {
    constexpr T c = static_cast<T>(0.044715);
    T inner = detail::SQRT2OPI<T> * (x + c * x * x * x);
    return x * T(0.5) * (T(1) + std::tanh(inner));
}
/// Derivative of exact GELU
template<typename T> inline T geluPrime(T x) {
    constexpr T inv_sqrt2    = T(1) / detail::ACT_SQRT2<T>;
    constexpr T inv_sqrt2pi  = static_cast<T>(0.39894228040143267794);
    T cdf = T(0.5) * (T(1) + std::erf(x * inv_sqrt2));
    T pdf = inv_sqrt2pi * std::exp(T(-0.5) * x * x);
    return cdf + x * pdf;
}


/// ═════════════════════════════════════════════════════════════════════════════
/// Swish / SiLU — x · σ(x)  (Ramachandran et al., 2017)
/// ═════════════════════════════════════════════════════════════════════════════

template<typename T> inline T swish(T x) { return x * sigmoid(x); }
template<typename T> inline T swishPrime(T x) {
    T s = sigmoid(x);
    return s * (T(1) + x * (T(1) - s));
}


/// ═════════════════════════════════════════════════════════════════════════════
/// Mish — x · tanh(softplus(x)) = x · tanh(ln(1 + e^x))  (Misra, 2019)
/// ═════════════════════════════════════════════════════════════════════════════

template<typename T> inline T mish(T x) {
    T sp = (x > T(20)) ? x : std::log(T(1) + std::exp(x));  // stable softplus
    return x * std::tanh(sp);
}
template<typename T> inline T mishPrime(T x) {
    T ex   = std::exp(x);
    T ex2  = ex * ex;
    T ex3  = ex2 * ex;
    T sp   = (x > T(20)) ? x : std::log(T(1) + ex);
    T th   = std::tanh(sp);
    T s    = sigmoid(x);
    return th + x * s * (T(1) - th * th);
}


/// ═════════════════════════════════════════════════════════════════════════════
/// Hard Sigmoid — piecewise linear approximation to σ(x)
/// f(x) = clamp(x/6 + 0.5, 0, 1)
/// ═════════════════════════════════════════════════════════════════════════════

template<typename T> inline T hardSigmoid(T x) {
    return std::max(T(0), std::min(T(1), x / T(6) + T(0.5)));
}
template<typename T> inline T hardSigmoidPrime(T x) {
    return (x > T(-3) && x < T(3)) ? T(1) / T(6) : T(0);
}


/// ═════════════════════════════════════════════════════════════════════════════
/// Hard Swish — x · hardSigmoid(x)  (Howard et al., MobileNetV3)
/// ═════════════════════════════════════════════════════════════════════════════

template<typename T> inline T hardSwish(T x) {
    return x * hardSigmoid(x);
}
template<typename T> inline T hardSwishPrime(T x) {
    if (x <= T(-3)) return T(0);
    if (x >= T(3))  return T(1);
    return (T(2) * x + T(3)) / T(6);
}


/// ═════════════════════════════════════════════════════════════════════════════
/// Softplus — smooth approximation to ReLU: log(1 + e^x)
/// ═════════════════════════════════════════════════════════════════════════════

template<typename T> inline T softplus(T x) {
    /// Numerically stable: for large x, log(1+e^x) ≈ x
    return (x > T(20)) ? x : std::log(T(1) + std::exp(x));
}
template<typename T> inline T softplusPrime(T x) { return sigmoid(x); }


/// ═════════════════════════════════════════════════════════════════════════════
/// Softsign — x / (1 + |x|)
/// ═════════════════════════════════════════════════════════════════════════════

template<typename T> inline T softsign(T x) {
    return x / (T(1) + std::abs(x));
}
template<typename T> inline T softsignPrime(T x) {
    T d = T(1) + std::abs(x);
    return T(1) / (d * d);
}


/// ═════════════════════════════════════════════════════════════════════════════
/// Bent Identity — (√(x²+1) − 1)/2 + x  (smooth, nearly-linear)
/// ═════════════════════════════════════════════════════════════════════════════

template<typename T> inline T bentIdentity(T x) {
    return (std::sqrt(x * x + T(1)) - T(1)) / T(2) + x;
}
template<typename T> inline T bentIdentityPrime(T x) {
    return x / (T(2) * std::sqrt(x * x + T(1))) + T(1);
}


// ═════════════════════════════════════════════════════════════════════════════
// VECTOR ACTIVATIONS
// ═════════════════════════════════════════════════════════════════════════════

// ── Softmax: σ(x)_i = e^{x_i} / Σ_j e^{x_j} ────────────────────────────────
// Numerically stable: subtract max before exponentiating.
template<typename T>
inline std::vector<T> softmax(const std::vector<T>& x) {
    if (x.empty()) return {};
    T xmax = *std::max_element(x.begin(), x.end());
    std::vector<T> out(x.size());
    T sum = T(0);
    for (size_t i = 0; i < x.size(); ++i) { out[i] = std::exp(x[i] - xmax); sum += out[i]; }
    for (T& v : out) v /= sum;
    return out;
}

// ── Log-softmax: log(σ(x)_i) = x_i − log(Σ e^{x_j})  (numerically stable) ──
template<typename T>
inline std::vector<T> logSoftmax(const std::vector<T>& x) {
    if (x.empty()) return {};
    T xmax = *std::max_element(x.begin(), x.end());
    T lse  = T(0);
    for (const T& v : x) lse += std::exp(v - xmax);
    lse = xmax + std::log(lse);
    std::vector<T> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) out[i] = x[i] - lse;
    return out;
}

// ── Element-wise clip ─────────────────────────────────────────────────────────
template<typename T>
inline T clip(T x, T lo, T hi) { return std::max(lo, std::min(hi, x)); }

template<typename T>
inline std::vector<T> clip(const std::vector<T>& x, T lo, T hi) {
    std::vector<T> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) out[i] = clip(x[i], lo, hi);
    return out;
}

} // namespace SharedMath::Functions
