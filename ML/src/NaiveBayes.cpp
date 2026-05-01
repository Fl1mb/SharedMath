#include "NaiveBayes.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace SharedMath::ML {

GaussianNB::GaussianNB(double var_smoothing)
    : m_var_smoothing(var_smoothing)
{
    if (var_smoothing < 0.0)
        throw std::invalid_argument("GaussianNB: var_smoothing must be >= 0");
}

void GaussianNB::fit(const Tensor& X, const Tensor& y) {
    if (X.ndim() != 2)
        throw std::invalid_argument("GaussianNB::fit: X must be 2-D [N, D]");
    if (y.ndim() != 1 || y.size() != X.dim(0))
        throw std::invalid_argument("GaussianNB::fit: y must be 1-D [N]");

    const size_t N = X.dim(0);
    const size_t D = X.dim(1);

    // Discover classes
    m_n_classes = 0;
    for (size_t i = 0; i < N; ++i) {
        size_t c = static_cast<size_t>(y.flat(i)) + 1;
        if (c > m_n_classes) m_n_classes = c;
    }
    m_n_features = D;

    std::vector<size_t> counts(m_n_classes, 0);
    m_theta      = Tensor::zeros({m_n_classes, D});
    m_sigma      = Tensor::zeros({m_n_classes, D});
    m_log_prior  = Tensor::zeros({m_n_classes});

    // Accumulate sums for means
    for (size_t i = 0; i < N; ++i) {
        size_t c = static_cast<size_t>(y.flat(i));
        ++counts[c];
        for (size_t d = 0; d < D; ++d)
            m_theta(c, d) += X(i, d);
    }
    for (size_t c = 0; c < m_n_classes; ++c) {
        if (counts[c] == 0) continue;
        for (size_t d = 0; d < D; ++d)
            m_theta(c, d) /= static_cast<double>(counts[c]);
        m_log_prior.flat(c) = std::log(static_cast<double>(counts[c]) / static_cast<double>(N));
    }

    // Compute variances
    for (size_t i = 0; i < N; ++i) {
        size_t c = static_cast<size_t>(y.flat(i));
        for (size_t d = 0; d < D; ++d) {
            double diff = X(i, d) - m_theta(c, d);
            m_sigma(c, d) += diff * diff;
        }
    }
    // Compute max variance across all features for smoothing scale
    double max_var = 0.0;
    for (size_t c = 0; c < m_n_classes; ++c) {
        if (counts[c] == 0) continue;
        for (size_t d = 0; d < D; ++d) {
            m_sigma(c, d) /= static_cast<double>(counts[c]);
            if (m_sigma(c, d) > max_var) max_var = m_sigma(c, d);
        }
    }
    // Add smoothing: var_smoothing * max_var (or absolute if max_var == 0)
    double smoothing = m_var_smoothing * (max_var > 0.0 ? max_var : 1.0);
    for (size_t c = 0; c < m_n_classes; ++c)
        for (size_t d = 0; d < D; ++d)
            m_sigma(c, d) += smoothing;

    m_fitted = true;
}

Tensor GaussianNB::predict_proba(const Tensor& X) const {
    if (!m_fitted)
        throw std::runtime_error("GaussianNB::predict_proba: model not fitted");
    if (X.ndim() != 2 || X.dim(1) != m_n_features)
        throw std::invalid_argument("GaussianNB::predict_proba: feature dim mismatch");

    const size_t N  = X.dim(0);
    const size_t C  = m_n_classes;
    const double half_log2pi = 0.5 * std::log(2.0 * M_PI);
    Tensor proba = Tensor::zeros({N, C});

    for (size_t i = 0; i < N; ++i) {
        double max_log = -std::numeric_limits<double>::infinity();
        std::vector<double> log_p(C);
        for (size_t c = 0; c < C; ++c) {
            double lp = m_log_prior.flat(c);
            for (size_t d = 0; d < m_n_features; ++d) {
                double var = m_sigma(c, d);
                double diff = X(i, d) - m_theta(c, d);
                lp -= half_log2pi + 0.5 * std::log(var) + 0.5 * diff * diff / var;
            }
            log_p[c] = lp;
            if (lp > max_log) max_log = lp;
        }
        // Softmax-normalise
        double denom = 0.0;
        for (size_t c = 0; c < C; ++c) {
            log_p[c] = std::exp(log_p[c] - max_log);
            denom += log_p[c];
        }
        for (size_t c = 0; c < C; ++c)
            proba(i, c) = log_p[c] / denom;
    }
    return proba;
}

Tensor GaussianNB::predict(const Tensor& X) const {
    Tensor proba = predict_proba(X);
    const size_t N = X.dim(0);
    const size_t C = m_n_classes;
    Tensor result = Tensor::zeros({N});
    for (size_t i = 0; i < N; ++i) {
        size_t best = 0;
        double best_p = proba(i, 0);
        for (size_t c = 1; c < C; ++c)
            if (proba(i, c) > best_p) { best_p = proba(i, c); best = c; }
        result.flat(i) = static_cast<double>(best);
    }
    return result;
}

size_t GaussianNB::n_classes()  const noexcept { return m_n_classes; }
size_t GaussianNB::n_features() const noexcept { return m_n_features; }
bool   GaussianNB::fitted()     const noexcept { return m_fitted; }
const Tensor& GaussianNB::class_prior() const noexcept { return m_log_prior; }
const Tensor& GaussianNB::theta()       const noexcept { return m_theta; }
const Tensor& GaussianNB::sigma()       const noexcept { return m_sigma; }

} // namespace SharedMath::ML
