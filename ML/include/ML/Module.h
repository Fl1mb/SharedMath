#pragma once

// SharedMath::ML — Module / Layer API
//
// Module       — abstract base class for all layers
// Linear       — fully-connected layer  (y = x @ W + b)
// Sequential   — ordered container of modules
// Dropout      — training-time dropout with inverted scaling

#include <sharedmath_ml_export.h>

#include "AutogradTensor.h"

#include <memory>
#include <vector>
#include <cstdint>

namespace SharedMath::ML {

using Tensor = SharedMath::LinearAlgebra::Tensor;

class SHAREDMATH_ML_EXPORT Module {
public:
    virtual ~Module() = default;

    virtual AutoTensor forward(const AutoTensor& x) = 0;
    AutoTensor operator()(const AutoTensor& x);

    virtual std::vector<AutoTensor*> parameters();

    virtual void train(bool mode = true);
    void eval();
    bool is_training() const noexcept;

    void zero_grad();

protected:
    bool m_training = true;
};

class SHAREDMATH_ML_EXPORT Linear : public Module {
public:
    Linear(size_t in_features, size_t out_features, bool use_bias = true);

    AutoTensor forward(const AutoTensor& x) override;
    std::vector<AutoTensor*> parameters() override;

    AutoTensor& weight();
    AutoTensor& bias();
    const AutoTensor& weight() const;
    const AutoTensor& bias() const;

    size_t in_features()  const noexcept;
    size_t out_features() const noexcept;

private:
    size_t     m_in;
    size_t     m_out;
    bool       m_use_bias;
    AutoTensor m_weight;
    AutoTensor m_bias;
};

class SHAREDMATH_ML_EXPORT Sequential : public Module {
public:
    Sequential();
    explicit Sequential(std::vector<std::shared_ptr<Module>> layers);

    void add(std::shared_ptr<Module> layer);

    AutoTensor forward(const AutoTensor& x) override;
    std::vector<AutoTensor*> parameters() override;
    void train(bool mode = true) override;

    size_t size() const noexcept;

private:
    std::vector<std::shared_ptr<Module>> m_layers;
};

class SHAREDMATH_ML_EXPORT Dropout : public Module {
public:
    explicit Dropout(double p = 0.5);

    AutoTensor forward(const AutoTensor& x) override;
    double drop_prob() const noexcept;

private:
    double        m_p;
    std::uint64_t m_seed = 12345;
};

} // namespace SharedMath::ML
