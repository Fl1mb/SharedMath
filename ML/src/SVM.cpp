#include "SVM.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace SharedMath::ML {

LinearSVM::LinearSVM(double C, double lr, size_t max_iter)
    : m_C(C), m_lr(lr), m_max_iter(max_iter)
{
    if (C <= 0.0)    throw std::invalid_argument("LinearSVM: C must be > 0");
    if (lr <= 0.0)   throw std::invalid_argument("LinearSVM: lr must be > 0");
}

void LinearSVM::fit(const Tensor& X, const Tensor& y) {
    if (X.ndim() != 2)
        throw std::invalid_argument("LinearSVM::fit: X must be 2-D");
    if (y.ndim() != 1 || y.size() != X.dim(0))
        throw std::invalid_argument("LinearSVM::fit: y must be 1-D [N]");

    const size_t N = X.dim(0);
    const size_t D = X.dim(1);

    // Convert labels to {-1, +1}
    // Detect if labels are {0,1} — if so, remap 0 -> -1, 1 -> +1
    bool has_zero = false, has_neg = false;
    for (size_t i = 0; i < N; ++i) {
        double v = y.flat(i);
        if (v == 0.0)       has_zero = true;
        if (v < 0.0)        has_neg  = true;
    }
    bool remap = has_zero && !has_neg;  // {0,1} -> {-1,+1}

    std::vector<double> labels(N);
    for (size_t i = 0; i < N; ++i)
        labels[i] = remap ? (y.flat(i) == 0.0 ? -1.0 : 1.0) : y.flat(i);

    m_w = Tensor::zeros({D});
    m_b = 0.0;

    // Subgradient descent: L = (1/2)||w||² + C * Σ max(0, 1 - y*(w·x + b))
    for (size_t iter = 0; iter < m_max_iter; ++iter) {
        Tensor dw = Tensor::zeros({D});
        double db = 0.0;

        for (size_t i = 0; i < N; ++i) {
            double yi = labels[i];
            double dot = m_b;
            for (size_t d = 0; d < D; ++d) dot += m_w.flat(d) * X(i, d);
            double margin = yi * dot;

            if (margin < 1.0) {
                // Hinge is active: subgradient = -y * x
                for (size_t d = 0; d < D; ++d)
                    dw.flat(d) -= yi * X(i, d);
                db -= yi;
            }
        }

        // dL/dw = w + C * dw_hinge,  dL/db = C * db_hinge
        for (size_t d = 0; d < D; ++d)
            m_w.flat(d) -= m_lr * (m_w.flat(d) + m_C * dw.flat(d) / static_cast<double>(N));
        m_b -= m_lr * m_C * db / static_cast<double>(N);
    }
    m_fitted = true;
}

Tensor LinearSVM::decision_function(const Tensor& X) const {
    if (!m_fitted)
        throw std::runtime_error("LinearSVM::decision_function: model not fitted");
    if (X.ndim() != 2 || X.dim(1) != m_w.size())
        throw std::invalid_argument("LinearSVM::decision_function: feature dim mismatch");
    const size_t N = X.dim(0);
    const size_t D = X.dim(1);
    Tensor out = Tensor::zeros({N});
    for (size_t i = 0; i < N; ++i) {
        double v = m_b;
        for (size_t d = 0; d < D; ++d) v += m_w.flat(d) * X(i, d);
        out.flat(i) = v;
    }
    return out;
}

Tensor LinearSVM::predict(const Tensor& X) const {
    Tensor df  = decision_function(X);
    Tensor out = Tensor::zeros({df.size()});
    for (size_t i = 0; i < df.size(); ++i)
        out.flat(i) = df.flat(i) >= 0.0 ? 1.0 : 0.0;
    return out;
}

const Tensor& LinearSVM::coef()      const { return m_w; }
double        LinearSVM::intercept() const noexcept { return m_b; }
bool          LinearSVM::fitted()    const noexcept { return m_fitted; }

} // namespace SharedMath::ML
