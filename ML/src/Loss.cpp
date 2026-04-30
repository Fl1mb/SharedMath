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

} // namespace SharedMath::ML
