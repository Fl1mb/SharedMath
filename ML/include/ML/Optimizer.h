#pragma once

/**
 * @file Optimizer.h
 * @brief Gradient-descent optimizers.
 *
 * @defgroup ML_Optimizers Optimizers
 * @ingroup ML
 * @{
 *
 * All optimizers inherit from @ref SharedMath::ML::Optimizer.  Usage:
 * @code{.cpp}
 * Adam opt(model.parameters(), 1e-3);
 * for (auto& batch : loader) {
 *     opt.zero_grad();
 *     auto loss = model(batch.X);
 *     loss.backward();
 *     opt.step();
 * }
 * @endcode
 *
 * | Class | Algorithm |
 * |-------|-----------|
 * | SGD | Stochastic gradient descent (optional momentum) |
 * | AdaGrad | Adaptive per-parameter learning rates |
 * | RMSProp | Exponentially-smoothed adaptive rates |
 * | Adam | Adaptive moment estimation (Kingma & Ba, 2015) |
 * | AdamW | Adam with decoupled weight decay |
 *
 * @}
 */

#include <sharedmath_ml_export.h>

#include "AutogradTensor.h"

#include <vector>

namespace SharedMath::ML {

using Tensor = SharedMath::LinearAlgebra::Tensor;

class SHAREDMATH_ML_EXPORT Optimizer {
public:
    explicit Optimizer(std::vector<AutoTensor*> params);
    virtual ~Optimizer() = default;

    virtual void step() = 0;
    void zero_grad();

protected:
    std::vector<AutoTensor*> m_params;
};

class SHAREDMATH_ML_EXPORT SGD : public Optimizer {
public:
    SGD(std::vector<AutoTensor*> params,
        double lr,
        double momentum = 0.0);

    void step() override;

    double lr()       const noexcept;
    double momentum() const noexcept;

private:
    double              m_lr;
    double              m_momentum;
    std::vector<Tensor> m_velocity;
};

class SHAREDMATH_ML_EXPORT AdaGrad : public Optimizer {
public:
    AdaGrad(std::vector<AutoTensor*> params,
            double lr  = 1e-2,
            double eps = 1e-8);

    void step() override;

    double lr()  const noexcept;
    double eps() const noexcept;

private:
    double              m_lr;
    double              m_eps;
    std::vector<Tensor> m_accumulator;
};

class SHAREDMATH_ML_EXPORT RMSProp : public Optimizer {
public:
    RMSProp(std::vector<AutoTensor*> params,
            double lr    = 1e-3,
            double alpha = 0.99,
            double eps   = 1e-8);

    void step() override;

    double lr()    const noexcept;
    double alpha() const noexcept;
    double eps()   const noexcept;

private:
    double              m_lr;
    double              m_alpha;
    double              m_eps;
    std::vector<Tensor> m_square_avg;
};

class SHAREDMATH_ML_EXPORT Adam : public Optimizer {
public:
    Adam(std::vector<AutoTensor*> params,
         double lr    = 1e-3,
         double beta1 = 0.9,
         double beta2 = 0.999,
         double eps   = 1e-8);

    void step() override;

    double lr()    const noexcept;
    double beta1() const noexcept;
    double beta2() const noexcept;

private:
    double              m_lr;
    double              m_beta1;
    double              m_beta2;
    double              m_eps;
    size_t              m_t = 0;
    std::vector<Tensor> m_m;
    std::vector<Tensor> m_v;
};

class SHAREDMATH_ML_EXPORT AdamW : public Optimizer {
public:
    AdamW(std::vector<AutoTensor*> params,
          double lr           = 1e-3,
          double beta1        = 0.9,
          double beta2        = 0.999,
          double eps          = 1e-8,
          double weight_decay = 1e-2);

    void step() override;

    double lr()           const noexcept;
    double beta1()        const noexcept;
    double beta2()        const noexcept;
    double weight_decay() const noexcept;

private:
    double              m_lr;
    double              m_beta1;
    double              m_beta2;
    double              m_eps;
    double              m_weight_decay;
    size_t              m_t = 0;
    std::vector<Tensor> m_m;
    std::vector<Tensor> m_v;
};

} // namespace SharedMath::ML
