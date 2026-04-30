#include "Optimizer.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace SharedMath::ML {

Optimizer::Optimizer(std::vector<AutoTensor*> params)
    : m_params(std::move(params))
{}

void Optimizer::zero_grad() {
    for (auto* p : m_params) p->zero_grad();
}

SGD::SGD(std::vector<AutoTensor*> params, double lr, double momentum)
    : Optimizer(std::move(params)),
      m_lr(lr),
      m_momentum(momentum)
{
    if (lr <= 0.0)
        throw std::invalid_argument("SGD: lr must be > 0");
    if (momentum < 0.0 || momentum >= 1.0)
        throw std::invalid_argument("SGD: momentum must be in [0, 1)");

    if (m_momentum > 0.0) {
        m_velocity.reserve(m_params.size());
        for (auto* p : m_params)
            m_velocity.push_back(Tensor::zeros(p->data().shape()));
    }
}

void SGD::step() {
    for (size_t i = 0; i < m_params.size(); ++i) {
        auto* p = m_params[i];
        if (!p->has_grad()) continue;

        const Tensor& g = p->grad();

        if (m_momentum > 0.0) {
            m_velocity[i] = m_velocity[i] * m_momentum + g;
            p->data() -= m_velocity[i] * m_lr;
        } else {
            p->data() -= g * m_lr;
        }
    }
}

double SGD::lr() const noexcept { return m_lr; }
double SGD::momentum() const noexcept { return m_momentum; }

AdaGrad::AdaGrad(std::vector<AutoTensor*> params, double lr, double eps)
    : Optimizer(std::move(params)),
      m_lr(lr),
      m_eps(eps)
{
    if (lr <= 0.0)
        throw std::invalid_argument("AdaGrad: lr must be > 0");
    if (eps <= 0.0)
        throw std::invalid_argument("AdaGrad: eps must be > 0");

    m_accumulator.reserve(m_params.size());
    for (auto* p : m_params)
        m_accumulator.push_back(Tensor::zeros(p->data().shape()));
}

void AdaGrad::step() {
    for (size_t i = 0; i < m_params.size(); ++i) {
        auto* p = m_params[i];
        if (!p->has_grad()) continue;

        const Tensor& g = p->grad();
        m_accumulator[i] += g * g;

        Tensor& param = p->data();
        const size_t sz = param.size();
        for (size_t j = 0; j < sz; ++j) {
            const double adjusted_lr =
                m_lr / (std::sqrt(m_accumulator[i].flat(j)) + m_eps);
            param.flat(j) -= adjusted_lr * g.flat(j);
        }
    }
}

double AdaGrad::lr() const noexcept { return m_lr; }
double AdaGrad::eps() const noexcept { return m_eps; }

RMSProp::RMSProp(std::vector<AutoTensor*> params,
                 double lr,
                 double alpha,
                 double eps)
    : Optimizer(std::move(params)),
      m_lr(lr),
      m_alpha(alpha),
      m_eps(eps)
{
    if (lr <= 0.0)
        throw std::invalid_argument("RMSProp: lr must be > 0");
    if (alpha < 0.0 || alpha >= 1.0)
        throw std::invalid_argument("RMSProp: alpha must be in [0, 1)");
    if (eps <= 0.0)
        throw std::invalid_argument("RMSProp: eps must be > 0");

    m_square_avg.reserve(m_params.size());
    for (auto* p : m_params)
        m_square_avg.push_back(Tensor::zeros(p->data().shape()));
}

void RMSProp::step() {
    for (size_t i = 0; i < m_params.size(); ++i) {
        auto* p = m_params[i];
        if (!p->has_grad()) continue;

        const Tensor& g = p->grad();
        m_square_avg[i] =
            m_square_avg[i] * m_alpha + (g * g) * (1.0 - m_alpha);

        Tensor& param = p->data();
        const size_t sz = param.size();
        for (size_t j = 0; j < sz; ++j) {
            const double denom = std::sqrt(m_square_avg[i].flat(j)) + m_eps;
            param.flat(j) -= m_lr * g.flat(j) / denom;
        }
    }
}

double RMSProp::lr() const noexcept { return m_lr; }
double RMSProp::alpha() const noexcept { return m_alpha; }
double RMSProp::eps() const noexcept { return m_eps; }

Adam::Adam(std::vector<AutoTensor*> params,
           double lr,
           double beta1,
           double beta2,
           double eps)
    : Optimizer(std::move(params)),
      m_lr(lr),
      m_beta1(beta1),
      m_beta2(beta2),
      m_eps(eps)
{
    if (lr <= 0.0)  throw std::invalid_argument("Adam: lr must be > 0");
    if (eps <= 0.0) throw std::invalid_argument("Adam: eps must be > 0");

    m_m.reserve(m_params.size());
    m_v.reserve(m_params.size());
    for (auto* p : m_params) {
        m_m.push_back(Tensor::zeros(p->data().shape()));
        m_v.push_back(Tensor::zeros(p->data().shape()));
    }
}

void Adam::step() {
    ++m_t;
    const double bc1 = 1.0 - std::pow(m_beta1, static_cast<double>(m_t));
    const double bc2 = 1.0 - std::pow(m_beta2, static_cast<double>(m_t));

    for (size_t i = 0; i < m_params.size(); ++i) {
        auto* p = m_params[i];
        if (!p->has_grad()) continue;

        const Tensor& g = p->grad();
        m_m[i] = m_m[i] * m_beta1 + g * (1.0 - m_beta1);
        m_v[i] = m_v[i] * m_beta2 + (g * g) * (1.0 - m_beta2);

        Tensor& param = p->data();
        const size_t sz = param.size();
        for (size_t j = 0; j < sz; ++j) {
            double mh = m_m[i].flat(j) / bc1;
            double vh = m_v[i].flat(j) / bc2;
            param.flat(j) -= m_lr * mh / (std::sqrt(vh) + m_eps);
        }
    }
}

double Adam::lr() const noexcept { return m_lr; }
double Adam::beta1() const noexcept { return m_beta1; }
double Adam::beta2() const noexcept { return m_beta2; }

} // namespace SharedMath::ML
