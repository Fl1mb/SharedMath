#pragma once

#include <sharedmath_ml_export.h>

#include "LinearAlgebra/Tensor.h"

#include <optional>

namespace SharedMath::ML {

// First building block for the ML module. For now this class is a small
// autograd-ready wrapper over LinearAlgebra::Tensor: it stores data, optional
// gradient, and the requires_grad flag. Graph construction/backward ops can be
// added here without making LinearAlgebra::Tensor heavier.
class SHAREDMATH_ML_EXPORT AutoTensor {
public:
    using Tensor = SharedMath::LinearAlgebra::Tensor;

    AutoTensor() = default;
    explicit AutoTensor(Tensor data, bool requires_grad = false);

    static AutoTensor from(Tensor data, bool requires_grad = false);

    const Tensor& data() const noexcept;
    Tensor& data() noexcept;

    bool requires_grad() const noexcept;
    void set_requires_grad(bool value) noexcept;

    bool has_grad() const noexcept;
    const Tensor& grad() const;
    Tensor& grad();
    void set_grad(Tensor grad);
    void zero_grad();

private:
    Tensor m_data;
    std::optional<Tensor> m_grad;
    bool m_requires_grad = false;
};

} // namespace SharedMath::ML
