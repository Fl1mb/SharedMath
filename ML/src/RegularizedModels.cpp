#include "RegularizedModels.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace SharedMath::ML {

namespace {

void check2DSupervised(const Tensor& X, const Tensor& y, const char* fn) {
    if (X.ndim() != 2)
        throw std::invalid_argument(std::string(fn) + ": X must be 2-D");
    if (y.ndim() != 1)
        throw std::invalid_argument(std::string(fn) + ": y must be 1-D");
    if (X.dim(0) != y.dim(0))
        throw std::invalid_argument(std::string(fn) + ": X and y row-count mismatch");
}

void checkFitted(bool f, const char* fn) {
    if (!f) throw std::runtime_error(std::string(fn) + ": model not fitted");
}

double dot_row(const Tensor& X, size_t i, const Tensor& theta, size_t D) {
    double v = 0.0;
    for (size_t d = 0; d < D; ++d) v += X(i, d) * theta.flat(d);
    return v;
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// RidgeRegression
// ─────────────────────────────────────────────────────────────────────────────

RidgeRegression::RidgeRegression(double alpha, double lr, size_t max_iter)
    : m_alpha(alpha), m_lr(lr), m_max_iter(max_iter)
{
    if (alpha < 0.0) throw std::invalid_argument("RidgeRegression: alpha must be >= 0");
    if (lr <= 0.0)   throw std::invalid_argument("RidgeRegression: lr must be > 0");
}

void RidgeRegression::fit(const Tensor& X, const Tensor& y) {
    check2DSupervised(X, y, "RidgeRegression::fit");
    const size_t N = X.dim(0);
    const size_t D = X.dim(1);

    m_theta = Tensor::zeros({D});
    m_bias  = 0.0;

    double max_col_norm = 0.0;
    for (size_t d = 0; d < D; ++d) {
        double col_norm = 0.0;
        for (size_t i = 0; i < N; ++i)
            col_norm += X(i, d) * X(i, d);
        if (col_norm > max_col_norm) max_col_norm = col_norm;
    }
    const double smoothness =
        2.0 * (max_col_norm / static_cast<double>(N) + m_alpha);
    const double effective_lr =
        std::min(m_lr, 0.5 / std::max(smoothness, 1e-12));

    for (size_t iter = 0; iter < m_max_iter; ++iter) {
        // gradient of MSE + alpha * ||theta||^2
        // dL/dtheta = (2/N) * X^T (Xθ + b - y) + 2*alpha*theta
        Tensor grad_theta = Tensor::zeros({D});
        double grad_bias  = 0.0;
        for (size_t i = 0; i < N; ++i) {
            double residual = dot_row(X, i, m_theta, D) + m_bias - y.flat(i);
            for (size_t d = 0; d < D; ++d)
                grad_theta.flat(d) += X(i, d) * residual;
            grad_bias += residual;
        }
        double scale = 2.0 / static_cast<double>(N);
        for (size_t d = 0; d < D; ++d)
            m_theta.flat(d) -= effective_lr * (scale * grad_theta.flat(d) + 2.0 * m_alpha * m_theta.flat(d));
        m_bias -= effective_lr * scale * grad_bias;
    }
    m_fitted = true;
}

Tensor RidgeRegression::predict(const Tensor& X) const {
    checkFitted(m_fitted, "RidgeRegression::predict");
    if (X.ndim() != 2 || X.dim(1) != m_theta.size())
        throw std::invalid_argument("RidgeRegression::predict: feature dim mismatch");
    const size_t N = X.dim(0);
    const size_t D = X.dim(1);
    Tensor out = Tensor::zeros({N});
    for (size_t i = 0; i < N; ++i)
        out.flat(i) = dot_row(X, i, m_theta, D) + m_bias;
    return out;
}

const Tensor& RidgeRegression::coef()     const { return m_theta; }
double RidgeRegression::intercept()  const noexcept { return m_bias; }
bool   RidgeRegression::fitted()     const noexcept { return m_fitted; }

// ─────────────────────────────────────────────────────────────────────────────
// LassoRegression  (coordinate descent with soft-thresholding)
// ─────────────────────────────────────────────────────────────────────────────

LassoRegression::LassoRegression(double alpha, size_t max_iter, double tol)
    : m_alpha(alpha), m_max_iter(max_iter), m_tol(tol)
{
    if (alpha < 0.0) throw std::invalid_argument("LassoRegression: alpha must be >= 0");
}

void LassoRegression::fit(const Tensor& X, const Tensor& y) {
    check2DSupervised(X, y, "LassoRegression::fit");
    const size_t N = X.dim(0);
    const size_t D = X.dim(1);

    m_theta = Tensor::zeros({D});
    m_bias  = 0.0;

    // Precompute column norms squared
    std::vector<double> col_norm2(D, 0.0);
    for (size_t d = 0; d < D; ++d)
        for (size_t i = 0; i < N; ++i)
            col_norm2[d] += X(i, d) * X(i, d);

    auto soft_threshold = [](double v, double thresh) -> double {
        if (v > thresh)  return v - thresh;
        if (v < -thresh) return v + thresh;
        return 0.0;
    };

    for (size_t iter = 0; iter < m_max_iter; ++iter) {
        double max_change = 0.0;
        // Update bias (intercept, no regularisation)
        double r_sum = 0.0;
        for (size_t i = 0; i < N; ++i)
            r_sum += y.flat(i) - dot_row(X, i, m_theta, D);
        m_bias = r_sum / static_cast<double>(N);

        for (size_t d = 0; d < D; ++d) {
            // Partial residual: r_i + X[i,d] * theta[d]
            double rho = 0.0;
            for (size_t i = 0; i < N; ++i) {
                double partial_res = y.flat(i) - m_bias - dot_row(X, i, m_theta, D)
                                     + X(i, d) * m_theta.flat(d);
                rho += X(i, d) * partial_res;
            }
            double old = m_theta.flat(d);
            if (col_norm2[d] < 1e-15) {
                m_theta.flat(d) = 0.0;
            } else {
                m_theta.flat(d) = soft_threshold(rho / col_norm2[d],
                                                  m_alpha / col_norm2[d] * static_cast<double>(N));
            }
            double change = std::abs(m_theta.flat(d) - old);
            if (change > max_change) max_change = change;
        }
        if (max_change < m_tol) break;
    }
    m_fitted = true;
}

Tensor LassoRegression::predict(const Tensor& X) const {
    checkFitted(m_fitted, "LassoRegression::predict");
    if (X.ndim() != 2 || X.dim(1) != m_theta.size())
        throw std::invalid_argument("LassoRegression::predict: feature dim mismatch");
    const size_t N = X.dim(0);
    const size_t D = X.dim(1);
    Tensor out = Tensor::zeros({N});
    for (size_t i = 0; i < N; ++i)
        out.flat(i) = dot_row(X, i, m_theta, D) + m_bias;
    return out;
}

const Tensor& LassoRegression::coef()     const { return m_theta; }
double LassoRegression::intercept()  const noexcept { return m_bias; }
bool   LassoRegression::fitted()     const noexcept { return m_fitted; }

// ─────────────────────────────────────────────────────────────────────────────
// ElasticNet  (coordinate descent)
// ─────────────────────────────────────────────────────────────────────────────

ElasticNet::ElasticNet(double alpha, double l1_ratio, size_t max_iter, double tol)
    : m_alpha(alpha), m_l1_ratio(l1_ratio), m_max_iter(max_iter), m_tol(tol)
{
    if (alpha < 0.0)           throw std::invalid_argument("ElasticNet: alpha must be >= 0");
    if (l1_ratio < 0.0 || l1_ratio > 1.0)
        throw std::invalid_argument("ElasticNet: l1_ratio must be in [0, 1]");
}

void ElasticNet::fit(const Tensor& X, const Tensor& y) {
    check2DSupervised(X, y, "ElasticNet::fit");
    const size_t N = X.dim(0);
    const size_t D = X.dim(1);

    m_theta = Tensor::zeros({D});
    m_bias  = 0.0;

    std::vector<double> col_norm2(D, 0.0);
    for (size_t d = 0; d < D; ++d)
        for (size_t i = 0; i < N; ++i)
            col_norm2[d] += X(i, d) * X(i, d);

    const double l1 = m_alpha * m_l1_ratio;
    const double l2 = m_alpha * (1.0 - m_l1_ratio);

    auto soft_threshold = [](double v, double thresh) -> double {
        if (v > thresh)  return v - thresh;
        if (v < -thresh) return v + thresh;
        return 0.0;
    };

    for (size_t iter = 0; iter < m_max_iter; ++iter) {
        double max_change = 0.0;
        double r_sum = 0.0;
        for (size_t i = 0; i < N; ++i)
            r_sum += y.flat(i) - dot_row(X, i, m_theta, D);
        m_bias = r_sum / static_cast<double>(N);

        for (size_t d = 0; d < D; ++d) {
            double rho = 0.0;
            for (size_t i = 0; i < N; ++i) {
                double partial_res = y.flat(i) - m_bias - dot_row(X, i, m_theta, D)
                                     + X(i, d) * m_theta.flat(d);
                rho += X(i, d) * partial_res;
            }
            double denom = col_norm2[d] + l2 * static_cast<double>(N);
            double old = m_theta.flat(d);
            if (denom < 1e-15) {
                m_theta.flat(d) = 0.0;
            } else {
                m_theta.flat(d) = soft_threshold(rho / denom,
                                                  l1 * static_cast<double>(N) / denom);
            }
            double change = std::abs(m_theta.flat(d) - old);
            if (change > max_change) max_change = change;
        }
        if (max_change < m_tol) break;
    }
    m_fitted = true;
}

Tensor ElasticNet::predict(const Tensor& X) const {
    checkFitted(m_fitted, "ElasticNet::predict");
    if (X.ndim() != 2 || X.dim(1) != m_theta.size())
        throw std::invalid_argument("ElasticNet::predict: feature dim mismatch");
    const size_t N = X.dim(0);
    const size_t D = X.dim(1);
    Tensor out = Tensor::zeros({N});
    for (size_t i = 0; i < N; ++i)
        out.flat(i) = dot_row(X, i, m_theta, D) + m_bias;
    return out;
}

const Tensor& ElasticNet::coef()     const { return m_theta; }
double ElasticNet::intercept()  const noexcept { return m_bias; }
bool   ElasticNet::fitted()     const noexcept { return m_fitted; }

} // namespace SharedMath::ML
