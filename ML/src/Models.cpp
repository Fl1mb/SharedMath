#include "Models.h"

#include "functions/Activations.h"

#include <algorithm>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace SharedMath::ML {
namespace {

void checkSupervised(const Tensor& X, const Tensor& y, const char* fn) {
    if (X.ndim() != 2)
        throw std::invalid_argument(std::string(fn) + ": X must be 2-D [N, D]");
    if (y.ndim() != 1)
        throw std::invalid_argument(std::string(fn) + ": y must be 1-D [N]");
    if (X.dim(0) != y.dim(0))
        throw std::invalid_argument(std::string(fn) + ": X and y row-count mismatch");
}

void checkFitted(bool fitted, const char* fn) {
    if (!fitted)
        throw std::runtime_error(std::string(fn) + ": model not fitted yet");
}

} // namespace

LinearRegression::LinearRegression(double lr, size_t max_iter)
    : m_lr(lr),
      m_max_iter(max_iter)
{}

void LinearRegression::fit(const Tensor& X, const Tensor& y) {
    checkSupervised(X, y, "LinearRegression::fit");

    const size_t N = X.dim(0);
    const size_t D = X.dim(1);

    m_theta = Tensor::zeros({D});
    m_bias  = 0.0;

    for (size_t iter = 0; iter < m_max_iter; ++iter) {
        Tensor y_hat = Tensor::zeros({N});
        for (size_t i = 0; i < N; ++i) {
            double val = m_bias;
            for (size_t d = 0; d < D; ++d)
                val += X(i, d) * m_theta.flat(d);
            y_hat.flat(i) = val;
        }

        Tensor residual = Tensor::zeros({N});
        for (size_t i = 0; i < N; ++i)
            residual.flat(i) = y_hat.flat(i) - y.flat(i);

        double scale = 2.0 / static_cast<double>(N);
        for (size_t d = 0; d < D; ++d) {
            double g = 0.0;
            for (size_t i = 0; i < N; ++i)
                g += X(i, d) * residual.flat(i);
            m_theta.flat(d) -= m_lr * scale * g;
        }

        double gb = 0.0;
        for (size_t i = 0; i < N; ++i) gb += residual.flat(i);
        m_bias -= m_lr * scale * gb;
    }
    m_fitted = true;
}

Tensor LinearRegression::predict(const Tensor& X) const {
    checkFitted(m_fitted, "LinearRegression::predict");
    if (X.ndim() != 2 || X.dim(1) != m_theta.dim(0))
        throw std::invalid_argument("LinearRegression::predict: shape mismatch");

    const size_t N = X.dim(0);
    const size_t D = X.dim(1);
    Tensor y_hat = Tensor::zeros({N});
    for (size_t i = 0; i < N; ++i) {
        double val = m_bias;
        for (size_t d = 0; d < D; ++d)
            val += X(i, d) * m_theta.flat(d);
        y_hat.flat(i) = val;
    }
    return y_hat;
}

const Tensor& LinearRegression::coef() const { return m_theta; }
double LinearRegression::intercept() const { return m_bias; }

LogisticRegression::LogisticRegression(double lr, size_t max_iter)
    : m_lr(lr),
      m_max_iter(max_iter)
{}

void LogisticRegression::fit(const Tensor& X, const Tensor& y) {
    checkSupervised(X, y, "LogisticRegression::fit");

    const size_t N = X.dim(0);
    const size_t D = X.dim(1);

    m_theta = Tensor::zeros({D});
    m_bias  = 0.0;

    for (size_t iter = 0; iter < m_max_iter; ++iter) {
        Tensor p = Tensor::zeros({N});
        for (size_t i = 0; i < N; ++i) {
            double z = m_bias;
            for (size_t d = 0; d < D; ++d)
                z += X(i, d) * m_theta.flat(d);
            p.flat(i) = SharedMath::Functions::sigmoid(z);
        }

        double scale = 1.0 / static_cast<double>(N);
        for (size_t d = 0; d < D; ++d) {
            double g = 0.0;
            for (size_t i = 0; i < N; ++i)
                g += X(i, d) * (p.flat(i) - y.flat(i));
            m_theta.flat(d) -= m_lr * scale * g;
        }
        double gb = 0.0;
        for (size_t i = 0; i < N; ++i) gb += p.flat(i) - y.flat(i);
        m_bias -= m_lr * scale * gb;
    }
    m_fitted = true;
}

Tensor LogisticRegression::predict_proba(const Tensor& X) const {
    checkFitted(m_fitted, "LogisticRegression::predict_proba");
    if (X.ndim() != 2 || X.dim(1) != m_theta.dim(0))
        throw std::invalid_argument("LogisticRegression::predict_proba: shape mismatch");

    const size_t N = X.dim(0);
    const size_t D = X.dim(1);
    Tensor proba = Tensor::zeros({N});
    for (size_t i = 0; i < N; ++i) {
        double z = m_bias;
        for (size_t d = 0; d < D; ++d)
            z += X(i, d) * m_theta.flat(d);
        proba.flat(i) = SharedMath::Functions::sigmoid(z);
    }
    return proba;
}

Tensor LogisticRegression::predict(const Tensor& X, double threshold) const {
    Tensor proba = predict_proba(X);
    Tensor labels = Tensor::zeros({proba.dim(0)});
    for (size_t i = 0; i < proba.dim(0); ++i)
        labels.flat(i) = proba.flat(i) >= threshold ? 1.0 : 0.0;
    return labels;
}

const Tensor& LogisticRegression::coef() const { return m_theta; }
double LogisticRegression::intercept() const { return m_bias; }

KMeans::KMeans(size_t k, size_t max_iter, uint64_t seed)
    : m_k(k),
      m_max_iter(max_iter),
      m_seed(seed)
{
    if (k == 0)
        throw std::invalid_argument("KMeans: k must be > 0");
}

void KMeans::fit(const Tensor& X) {
    if (X.ndim() != 2)
        throw std::invalid_argument("KMeans::fit: X must be 2-D [N, D]");

    const size_t N = X.dim(0);
    const size_t D = X.dim(1);

    if (N < m_k)
        throw std::invalid_argument("KMeans::fit: fewer samples than clusters");

    std::mt19937_64 rng(m_seed);
    std::uniform_int_distribution<size_t> dist(0, N - 1);

    m_centroids = Tensor::zeros({m_k, D});
    for (size_t c = 0; c < m_k; ++c) {
        size_t idx = dist(rng);
        for (size_t d = 0; d < D; ++d)
            m_centroids(c, d) = X(idx, d);
    }

    std::vector<size_t> labels(N, 0);

    for (size_t iter = 0; iter < m_max_iter; ++iter) {
        bool changed = false;

        for (size_t i = 0; i < N; ++i) {
            size_t best = 0;
            double best_dist = std::numeric_limits<double>::infinity();
            for (size_t c = 0; c < m_k; ++c) {
                double d2 = 0.0;
                for (size_t d = 0; d < D; ++d) {
                    double diff = X(i, d) - m_centroids(c, d);
                    d2 += diff * diff;
                }
                if (d2 < best_dist) { best_dist = d2; best = c; }
            }
            if (labels[i] != best) { labels[i] = best; changed = true; }
        }

        if (!changed) break;

        Tensor new_centroids = Tensor::zeros({m_k, D});
        std::vector<size_t> counts(m_k, 0);
        for (size_t i = 0; i < N; ++i) {
            size_t c = labels[i];
            ++counts[c];
            for (size_t d = 0; d < D; ++d)
                new_centroids(c, d) += X(i, d);
        }
        for (size_t c = 0; c < m_k; ++c) {
            if (counts[c] == 0) continue;
            for (size_t d = 0; d < D; ++d)
                new_centroids(c, d) /= static_cast<double>(counts[c]);
        }
        m_centroids = new_centroids;
    }

    m_labels = labels;
    m_fitted = true;
}

Tensor KMeans::predict(const Tensor& X) const {
    checkFitted(m_fitted, "KMeans::predict");
    if (X.ndim() != 2 || X.dim(1) != m_centroids.dim(1))
        throw std::invalid_argument("KMeans::predict: feature dim mismatch");

    const size_t N = X.dim(0);
    const size_t D = X.dim(1);
    Tensor result = Tensor::zeros({N});
    for (size_t i = 0; i < N; ++i) {
        size_t best = 0;
        double best_dist = std::numeric_limits<double>::infinity();
        for (size_t c = 0; c < m_k; ++c) {
            double d2 = 0.0;
            for (size_t d = 0; d < D; ++d) {
                double diff = X(i, d) - m_centroids(c, d);
                d2 += diff * diff;
            }
            if (d2 < best_dist) { best_dist = d2; best = c; }
        }
        result.flat(i) = static_cast<double>(best);
    }
    return result;
}

const Tensor& KMeans::centroids() const { return m_centroids; }
const std::vector<size_t>& KMeans::labels() const { return m_labels; }
size_t KMeans::k() const noexcept { return m_k; }

KNNClassifier::KNNClassifier(size_t k)
    : m_k(k)
{
    if (k == 0)
        throw std::invalid_argument("KNNClassifier: k must be > 0");
}

void KNNClassifier::fit(const Tensor& X, const Tensor& y) {
    checkSupervised(X, y, "KNNClassifier::fit");
    m_X_train = X;
    m_y_train = y;
    m_fitted  = true;
}

Tensor KNNClassifier::predict(const Tensor& X) const {
    checkFitted(m_fitted, "KNNClassifier::predict");
    if (X.ndim() != 2 || X.dim(1) != m_X_train.dim(1))
        throw std::invalid_argument("KNNClassifier::predict: feature dim mismatch");

    const size_t N_test  = X.dim(0);
    const size_t N_train = m_X_train.dim(0);
    const size_t D       = X.dim(1);
    const size_t k       = std::min(m_k, N_train);

    Tensor result = Tensor::zeros({N_test});

    for (size_t i = 0; i < N_test; ++i) {
        std::vector<std::pair<double, size_t>> dists(N_train);
        for (size_t j = 0; j < N_train; ++j) {
            double d2 = 0.0;
            for (size_t d = 0; d < D; ++d) {
                double diff = X(i, d) - m_X_train(j, d);
                d2 += diff * diff;
            }
            dists[j] = {d2, j};
        }

        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());

        std::vector<std::pair<int, size_t>> votes;
        for (size_t n = 0; n < k; ++n) {
            int lbl = static_cast<int>(m_y_train.flat(dists[n].second));
            bool found = false;
            for (auto& vote : votes) {
                if (vote.first == lbl) {
                    ++vote.second;
                    found = true;
                    break;
                }
            }
            if (!found) votes.push_back({lbl, 1});
        }

        auto best = std::max_element(votes.begin(), votes.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        result.flat(i) = static_cast<double>(best->first);
    }
    return result;
}

size_t KNNClassifier::k() const noexcept { return m_k; }

} // namespace SharedMath::ML
