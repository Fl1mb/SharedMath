#include "Trainer.h"

namespace SharedMath::ML {

Trainer::Trainer(Module& model, Optimizer& optimizer, Loss& loss)
    : m_model(model), m_optimizer(optimizer), m_loss(loss)
{}

std::vector<double> Trainer::fit(DataLoader& loader, size_t epochs) {
    std::vector<double> losses;
    losses.reserve(epochs);
    m_model.train();
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        loader.resetEpoch();
        losses.push_back(train_epoch(loader));
    }
    return losses;
}

double Trainer::train_epoch(DataLoader& loader) {
    double total = 0.0;
    size_t batches = 0;
    for (const auto& batch : loader) {
        m_optimizer.zero_grad();
        auto x = AutoTensor::from(batch.X);
        auto y = AutoTensor::from(batch.y);
        auto pred = m_model.forward(x);
        auto loss = m_loss.forward(pred, y);
        total += loss.data().cpu().flat(0);
        ++batches;
        loss.backward();
        m_optimizer.step();
    }
    return batches == 0 ? 0.0 : total / static_cast<double>(batches);
}

} // namespace SharedMath::ML
