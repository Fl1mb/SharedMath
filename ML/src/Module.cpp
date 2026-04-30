#include "Module.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace SharedMath::ML {

AutoTensor Module::operator()(const AutoTensor& x) {
    return forward(x);
}

std::vector<AutoTensor*> Module::parameters() {
    return {};
}

void Module::train(bool mode) {
    m_training = mode;
}

void Module::eval() {
    train(false);
}

bool Module::is_training() const noexcept {
    return m_training;
}

void Module::zero_grad() {
    for (auto* p : parameters()) p->zero_grad();
}

Linear::Linear(size_t in_features, size_t out_features, bool use_bias)
    : m_in(in_features),
      m_out(out_features),
      m_use_bias(use_bias)
{
    if (in_features == 0 || out_features == 0)
        throw std::invalid_argument("Linear: feature dimensions must be > 0");

    double limit = 1.0 / std::sqrt(static_cast<double>(in_features));
    m_weight = AutoTensor::from(
        Tensor::uniform({in_features, out_features}, -limit, limit, 42),
        true);

    if (use_bias)
        m_bias = AutoTensor::from(Tensor::zeros({1, out_features}), true);
}

AutoTensor Linear::forward(const AutoTensor& x) {
    if (x.data().ndim() != 2)
        throw std::invalid_argument(
            "Linear::forward: input must be 2-D [batch, in_features]");
    if (x.data().dim(1) != m_in)
        throw std::invalid_argument("Linear::forward: input feature dim mismatch");

    auto out = x.matmul(m_weight);
    if (m_use_bias)
        out = out + m_bias;
    return out;
}

std::vector<AutoTensor*> Linear::parameters() {
    if (m_use_bias) return {&m_weight, &m_bias};
    return {&m_weight};
}

AutoTensor& Linear::weight() { return m_weight; }
AutoTensor& Linear::bias() { return m_bias; }
const AutoTensor& Linear::weight() const { return m_weight; }
const AutoTensor& Linear::bias() const { return m_bias; }

size_t Linear::in_features() const noexcept { return m_in; }
size_t Linear::out_features() const noexcept { return m_out; }

Sequential::Sequential() = default;

Sequential::Sequential(std::vector<std::shared_ptr<Module>> layers)
    : m_layers(std::move(layers))
{}

void Sequential::add(std::shared_ptr<Module> layer) {
    m_layers.push_back(std::move(layer));
}

AutoTensor Sequential::forward(const AutoTensor& x) {
    AutoTensor out = x;
    for (auto& layer : m_layers) {
        layer->train(m_training);
        out = layer->forward(out);
    }
    return out;
}

std::vector<AutoTensor*> Sequential::parameters() {
    std::vector<AutoTensor*> all;
    for (auto& layer : m_layers) {
        auto p = layer->parameters();
        all.insert(all.end(), p.begin(), p.end());
    }
    return all;
}

void Sequential::train(bool mode) {
    m_training = mode;
    for (auto& layer : m_layers) layer->train(mode);
}

size_t Sequential::size() const noexcept {
    return m_layers.size();
}

Dropout::Dropout(double p)
    : m_p(p)
{
    if (p < 0.0 || p >= 1.0)
        throw std::invalid_argument("Dropout: p must be in [0, 1)");
}

AutoTensor Dropout::forward(const AutoTensor& x) {
    if (!m_training || m_p == 0.0) return x;

    const double keep = 1.0 - m_p;
    const double scale = 1.0 / keep;

    Tensor mask = Tensor::bernoulli(x.data().shape(), keep, m_seed++);
    Tensor out_data = x.data() * mask * scale;

    if (!x.requires_grad())
        return AutoTensor::from(std::move(out_data));

    auto xi = x.impl();
    return AutoTensor::make_result(std::move(out_data), true,
        [xi, mask, scale](const Tensor& g) {
            xi->propagate(g * mask * scale);
        });
}

double Dropout::drop_prob() const noexcept {
    return m_p;
}

} // namespace SharedMath::ML
