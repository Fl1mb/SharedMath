#include "AutogradTensor.h"

#include <stdexcept>
#include <utility>

namespace SharedMath::ML {

AutoTensor::AutoTensor(Tensor data, bool requires_grad)
    : m_data(std::move(data)),
      m_requires_grad(requires_grad)
{}

AutoTensor AutoTensor::from(Tensor data, bool requires_grad) {
    return AutoTensor(std::move(data), requires_grad);
}

const AutoTensor::Tensor& AutoTensor::data() const noexcept {
    return m_data;
}

AutoTensor::Tensor& AutoTensor::data() noexcept {
    return m_data;
}

bool AutoTensor::requires_grad() const noexcept {
    return m_requires_grad;
}

void AutoTensor::set_requires_grad(bool value) noexcept {
    m_requires_grad = value;
}

bool AutoTensor::has_grad() const noexcept {
    return m_grad.has_value();
}

const AutoTensor::Tensor& AutoTensor::grad() const {
    if (!m_grad)
        throw std::runtime_error("AutoTensor::grad: gradient is not set");
    return *m_grad;
}

AutoTensor::Tensor& AutoTensor::grad() {
    if (!m_grad)
        throw std::runtime_error("AutoTensor::grad: gradient is not set");
    return *m_grad;
}

void AutoTensor::set_grad(Tensor grad) {
    if (grad.shape() != m_data.shape())
        throw std::invalid_argument("AutoTensor::set_grad: gradient shape mismatch");
    m_grad = std::move(grad);
}

void AutoTensor::zero_grad() {
    m_grad = Tensor::zeros(m_data.shape());
}

} // namespace SharedMath::ML
