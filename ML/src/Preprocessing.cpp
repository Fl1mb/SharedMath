#include "Preprocessing.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace SharedMath::ML {

namespace {

void checkFitted2D(bool fitted, const char* fn) {
    if (!fitted)
        throw std::runtime_error(std::string(fn) + ": scaler not fitted yet");
}

void check2D(const Tensor& X, const char* fn) {
    if (X.ndim() != 2)
        throw std::invalid_argument(std::string(fn) + ": X must be 2-D [N, D]");
    if (X.dim(0) == 0 || X.dim(1) == 0)
        throw std::invalid_argument(std::string(fn) + ": X must be non-empty");
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// StandardScaler
// ─────────────────────────────────────────────────────────────────────────────

void StandardScaler::fit(const Tensor& X) {
    check2D(X, "StandardScaler::fit");
    const size_t N = X.dim(0);
    const size_t D = X.dim(1);

    m_mean  = Tensor::zeros({D});
    m_scale = Tensor::zeros({D});

    for (size_t d = 0; d < D; ++d) {
        double sum = 0.0;
        for (size_t i = 0; i < N; ++i) sum += X(i, d);
        m_mean.flat(d) = sum / static_cast<double>(N);

        double var = 0.0;
        for (size_t i = 0; i < N; ++i) {
            double diff = X(i, d) - m_mean.flat(d);
            var += diff * diff;
        }
        var /= static_cast<double>(N);
        m_scale.flat(d) = (var > 0.0) ? std::sqrt(var) : 1.0;
    }
    m_fitted = true;
}

Tensor StandardScaler::transform(const Tensor& X) const {
    checkFitted2D(m_fitted, "StandardScaler::transform");
    check2D(X, "StandardScaler::transform");
    if (X.dim(1) != m_mean.size())
        throw std::invalid_argument("StandardScaler::transform: feature dimension mismatch");

    const size_t N = X.dim(0);
    const size_t D = X.dim(1);
    Tensor out = Tensor::zeros({N, D});
    for (size_t i = 0; i < N; ++i)
        for (size_t d = 0; d < D; ++d)
            out(i, d) = (X(i, d) - m_mean.flat(d)) / m_scale.flat(d);
    return out;
}

Tensor StandardScaler::fit_transform(const Tensor& X) {
    fit(X);
    return transform(X);
}

Tensor StandardScaler::inverse_transform(const Tensor& X) const {
    checkFitted2D(m_fitted, "StandardScaler::inverse_transform");
    check2D(X, "StandardScaler::inverse_transform");
    if (X.dim(1) != m_mean.size())
        throw std::invalid_argument("StandardScaler::inverse_transform: feature dimension mismatch");

    const size_t N = X.dim(0);
    const size_t D = X.dim(1);
    Tensor out = Tensor::zeros({N, D});
    for (size_t i = 0; i < N; ++i)
        for (size_t d = 0; d < D; ++d)
            out(i, d) = X(i, d) * m_scale.flat(d) + m_mean.flat(d);
    return out;
}

const Tensor& StandardScaler::mean()  const noexcept { return m_mean; }
const Tensor& StandardScaler::scale() const noexcept { return m_scale; }
bool StandardScaler::fitted() const noexcept { return m_fitted; }

// ─────────────────────────────────────────────────────────────────────────────
// MinMaxScaler
// ─────────────────────────────────────────────────────────────────────────────

MinMaxScaler::MinMaxScaler(double feature_min, double feature_max)
    : m_feature_min(feature_min),
      m_feature_max(feature_max)
{
    if (feature_min >= feature_max)
        throw std::invalid_argument("MinMaxScaler: feature_min must be < feature_max");
}

void MinMaxScaler::fit(const Tensor& X) {
    check2D(X, "MinMaxScaler::fit");
    const size_t N = X.dim(0);
    const size_t D = X.dim(1);

    m_data_min   = Tensor(Tensor::Shape{D},  std::numeric_limits<double>::infinity());
    m_data_max   = Tensor(Tensor::Shape{D}, -std::numeric_limits<double>::infinity());

    for (size_t i = 0; i < N; ++i)
        for (size_t d = 0; d < D; ++d) {
            if (X(i, d) < m_data_min.flat(d)) m_data_min.flat(d) = X(i, d);
            if (X(i, d) > m_data_max.flat(d)) m_data_max.flat(d) = X(i, d);
        }

    m_data_range = Tensor::zeros({D});
    for (size_t d = 0; d < D; ++d) {
        double r = m_data_max.flat(d) - m_data_min.flat(d);
        m_data_range.flat(d) = (r > 0.0) ? r : 1.0;
    }
    m_fitted = true;
}

Tensor MinMaxScaler::transform(const Tensor& X) const {
    checkFitted2D(m_fitted, "MinMaxScaler::transform");
    check2D(X, "MinMaxScaler::transform");
    if (X.dim(1) != m_data_min.size())
        throw std::invalid_argument("MinMaxScaler::transform: feature dimension mismatch");

    const size_t N = X.dim(0);
    const size_t D = X.dim(1);
    const double scale = m_feature_max - m_feature_min;
    Tensor out = Tensor::zeros({N, D});
    for (size_t i = 0; i < N; ++i)
        for (size_t d = 0; d < D; ++d)
            out(i, d) = (X(i, d) - m_data_min.flat(d)) / m_data_range.flat(d) * scale + m_feature_min;
    return out;
}

Tensor MinMaxScaler::fit_transform(const Tensor& X) {
    fit(X);
    return transform(X);
}

Tensor MinMaxScaler::inverse_transform(const Tensor& X) const {
    checkFitted2D(m_fitted, "MinMaxScaler::inverse_transform");
    check2D(X, "MinMaxScaler::inverse_transform");
    if (X.dim(1) != m_data_min.size())
        throw std::invalid_argument("MinMaxScaler::inverse_transform: feature dimension mismatch");

    const size_t N = X.dim(0);
    const size_t D = X.dim(1);
    const double scale = m_feature_max - m_feature_min;
    Tensor out = Tensor::zeros({N, D});
    for (size_t i = 0; i < N; ++i)
        for (size_t d = 0; d < D; ++d)
            out(i, d) = (X(i, d) - m_feature_min) / scale * m_data_range.flat(d) + m_data_min.flat(d);
    return out;
}

const Tensor& MinMaxScaler::data_min()   const noexcept { return m_data_min; }
const Tensor& MinMaxScaler::data_max()   const noexcept { return m_data_max; }
const Tensor& MinMaxScaler::data_range() const noexcept { return m_data_range; }
bool MinMaxScaler::fitted() const noexcept { return m_fitted; }

// ─────────────────────────────────────────────────────────────────────────────
// train_test_split
// ─────────────────────────────────────────────────────────────────────────────

TrainTestSplit train_test_split(const Tensor& X,
                                const Tensor& y,
                                double   test_size,
                                bool     shuffle,
                                uint64_t seed)
{
    if (X.ndim() != 2)
        throw std::invalid_argument("train_test_split: X must be 2-D");
    if (y.ndim() != 1)
        throw std::invalid_argument("train_test_split: y must be 1-D");
    if (X.dim(0) != y.dim(0))
        throw std::invalid_argument("train_test_split: X and y row-count mismatch");
    if (test_size <= 0.0 || test_size >= 1.0)
        throw std::invalid_argument("train_test_split: test_size must be in (0, 1)");

    const size_t N     = X.dim(0);
    const size_t D     = X.dim(1);
    const size_t n_test  = std::max<size_t>(1, static_cast<size_t>(std::round(N * test_size)));
    const size_t n_train = N - n_test;

    std::vector<size_t> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    if (shuffle) {
        std::mt19937_64 rng(seed);
        std::shuffle(idx.begin(), idx.end(), rng);
    }

    Tensor X_train = Tensor::zeros({n_train, D});
    Tensor y_train = Tensor::zeros({n_train});
    Tensor X_test  = Tensor::zeros({n_test, D});
    Tensor y_test  = Tensor::zeros({n_test});

    for (size_t i = 0; i < n_train; ++i) {
        y_train.flat(i) = y.flat(idx[i]);
        for (size_t d = 0; d < D; ++d)
            X_train(i, d) = X(idx[i], d);
    }
    for (size_t i = 0; i < n_test; ++i) {
        y_test.flat(i) = y.flat(idx[n_train + i]);
        for (size_t d = 0; d < D; ++d)
            X_test(i, d) = X(idx[n_train + i], d);
    }

    return {X_train, X_test, y_train, y_test};
}

} // namespace SharedMath::ML
