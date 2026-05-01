#include "Loss.h"

#include <cmath>
#include <stdexcept>

namespace SharedMath::ML {

AutoTensor Loss::operator()(const AutoTensor& y_pred, const AutoTensor& y_true) {
    return forward(y_pred, y_true);
}

AutoTensor MSELoss::forward(const AutoTensor& y_pred, const AutoTensor& y_true) {
    auto diff = y_pred - y_true;
    return (diff * diff).mean();
}

CrossEntropyLoss::CrossEntropyLoss(double eps)
    : m_eps(eps)
{
    if (eps <= 0.0)
        throw std::invalid_argument("CrossEntropyLoss: eps must be > 0");
}

AutoTensor CrossEntropyLoss::forward(const AutoTensor& logits, const Tensor& labels) {
    if (logits.data().ndim() != 2)
        throw std::invalid_argument("CrossEntropyLoss::forward: logits must be [N, C]");
    if (labels.ndim() != 1 || labels.dim(0) != logits.data().dim(0))
        throw std::invalid_argument("CrossEntropyLoss::forward: labels must be [N]");

    const size_t N = logits.data().dim(0);
    const size_t C = logits.data().dim(1);
    Tensor probs = logits.data().softmax(1);
    Tensor probs_cpu = probs.cpu();
    Tensor labels_cpu = labels.cpu();

    double loss_value = 0.0;
    Tensor grad = probs_cpu;
    for (size_t n = 0; n < N; ++n) {
        int cls = static_cast<int>(labels_cpu.flat(n));
        if (cls < 0 || cls >= static_cast<int>(C))
            throw std::out_of_range("CrossEntropyLoss::forward: class id out of range");
        loss_value -= std::log(std::max(m_eps, probs_cpu(n, static_cast<size_t>(cls))));
        grad(n, static_cast<size_t>(cls)) -= 1.0;
    }
    loss_value /= static_cast<double>(N);
    grad /= static_cast<double>(N);
    grad = grad.to(logits.data().device(), logits.data().device_id());

    auto li = logits.impl();
    Tensor loss_data = Tensor::from_vector({loss_value});
    return AutoTensor::make_result(std::move(loss_data), logits.requires_grad(),
        [li, grad](const Tensor& upstream) {
            double scale = upstream.cpu().flat(0);
            li->propagate(grad * scale);
        });
}

AutoTensor CrossEntropyLoss::operator()(const AutoTensor& logits, const Tensor& labels) {
    return forward(logits, labels);
}

// ─── HuberLoss ──────────────────────────────────────────────────────────────

HuberLoss::HuberLoss(double delta)
    : m_delta(delta)
{
    if (delta <= 0.0)
        throw std::invalid_argument("HuberLoss: delta must be > 0");
}

AutoTensor HuberLoss::forward(const AutoTensor& y_pred, const AutoTensor& y_true) {
    const double d = m_delta;
    auto diff = y_pred - y_true;

    // Element-wise Huber: 0.5*err^2 if |err|<=delta, else delta*(|err|-0.5*delta)
    Tensor out_data = diff.data().apply([d](double v) {
        return std::abs(v) <= d ? 0.5 * v * v : d * (std::abs(v) - 0.5 * d);
    });
    AutoTensor huber = AutoTensor::make_result(
        std::move(out_data), diff.requires_grad(),
        [di = diff.impl(), d](const Tensor& g) {
            Tensor grad = di->data.apply([d](double v) {
                if (v > d)  return d;
                if (v < -d) return -d;
                return v;
            });
            di->propagate(g * grad);
        });
    return huber.mean();
}

double HuberLoss::delta() const noexcept { return m_delta; }

// ─── BCELoss ────────────────────────────────────────────────────────────────

BCELoss::BCELoss(double eps)
    : m_eps(eps)
{
    if (eps <= 0.0)
        throw std::invalid_argument("BCELoss: eps must be > 0");
}

AutoTensor BCELoss::forward(const AutoTensor& y_pred, const AutoTensor& y_true) {
    const double eps = m_eps;
    const size_t N = y_pred.data().size();
    if (N != y_true.data().size())
        throw std::invalid_argument("BCELoss: size mismatch between y_pred and y_true");

    // loss = -mean(y*log(p) + (1-y)*log(1-p))
    Tensor p  = y_pred.data();
    Tensor yt = y_true.data();
    double loss_val = 0.0;
    Tensor grad_data(p.shape());
    for (size_t i = 0; i < N; ++i) {
        double pi = std::max(eps, std::min(1.0 - eps, p.flat(i)));
        double yi = yt.flat(i);
        loss_val -= (yi * std::log(pi) + (1.0 - yi) * std::log(1.0 - pi));
        grad_data.flat(i) = (pi - yi) / (pi * (1.0 - pi));
    }
    loss_val /= static_cast<double>(N);
    grad_data /= static_cast<double>(N);

    auto pi_impl = y_pred.impl();
    return AutoTensor::make_result(
        Tensor::from_vector({loss_val}), y_pred.requires_grad(),
        [pi_impl, grad_data](const Tensor& upstream) {
            double s = upstream.cpu().flat(0);
            pi_impl->propagate(grad_data * s);
        });
}

double BCELoss::eps() const noexcept { return m_eps; }

// ─── BCEWithLogitsLoss ──────────────────────────────────────────────────────

AutoTensor BCEWithLogitsLoss::forward(const AutoTensor& logits, const AutoTensor& y_true) {
    const size_t N = logits.data().size();
    if (N != y_true.data().size())
        throw std::invalid_argument("BCEWithLogitsLoss: size mismatch");

    // numerically stable: max(z,0) - z*y + log(1+exp(-|z|))
    Tensor z  = logits.data();
    Tensor yt = y_true.data();
    double loss_val = 0.0;
    Tensor grad_data(z.shape());
    for (size_t i = 0; i < N; ++i) {
        double zi = z.flat(i);
        double yi = yt.flat(i);
        loss_val += std::max(0.0, zi) - zi * yi + std::log(1.0 + std::exp(-std::abs(zi)));
        double sig = 1.0 / (1.0 + std::exp(-zi));
        grad_data.flat(i) = (sig - yi);
    }
    loss_val /= static_cast<double>(N);
    grad_data /= static_cast<double>(N);

    auto li = logits.impl();
    return AutoTensor::make_result(
        Tensor::from_vector({loss_val}), logits.requires_grad(),
        [li, grad_data](const Tensor& upstream) {
            double s = upstream.cpu().flat(0);
            li->propagate(grad_data * s);
        });
}

} // namespace SharedMath::ML
