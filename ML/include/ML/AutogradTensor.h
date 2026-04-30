#pragma once

#include <sharedmath_ml_export.h>
#include "LinearAlgebra/Tensor.h"

#include <functional>
#include <memory>
#include <stdexcept>
#include <cmath>

namespace SharedMath::ML {

// ─────────────────────────────────────────────────────────────────────────────
// AutoTensor — reference-typed autograd node
//
// Internally uses shared_ptr<Impl> so that copies of an AutoTensor share the
// same gradient storage, and backward closures captured by operations can
// accumulate gradients into the correct leaf nodes.
//
// Usage:
//   auto x = AutoTensor::from(Tensor::ones({3}), /*requires_grad=*/true);
//   auto y = (x * x).mean();  // builds computation graph
//   y.backward();             // populates x.grad()
// ─────────────────────────────────────────────────────────────────────────────
class SHAREDMATH_ML_EXPORT AutoTensor {
public:
    using Tensor = SharedMath::LinearAlgebra::Tensor;

    // ── Internal node (public for shared capture in backward closures) ── //
    struct Impl {
        Tensor data;
        Tensor grad;
        bool   has_grad_     = false;
        bool   requires_grad = false;
        bool   is_leaf       = true;

        // Called by operations to propagate upstream gradient into this node.
        std::function<void(const Tensor&)> grad_fn;

        void accumulate_grad(const Tensor& g);

        // Routes upstream gradient: accumulate if leaf, recurse via grad_fn otherwise.
        void propagate(const Tensor& upstream);
    };

    // ── Construction ──────────────────────────────────────────────────── //

    AutoTensor();
    explicit AutoTensor(Tensor data, bool requires_grad = false);

    static AutoTensor from(Tensor data, bool requires_grad = false);

    // Low-level factory used by operation implementations to build non-leaf
    // nodes with a custom backward closure.
    static AutoTensor make_result(Tensor data,
                                  bool   requires_grad,
                                  std::function<void(const Tensor&)> grad_fn);

    // ── Data & gradient access ────────────────────────────────────────── //

    const Tensor& data() const noexcept;
    Tensor&       data()       noexcept;

    bool requires_grad()              const noexcept;
    void set_requires_grad(bool value)      noexcept;

    bool          has_grad()   const noexcept;
    const Tensor& grad()       const;
    Tensor&       grad();
    void          set_grad(Tensor g);
    void          zero_grad();

    // ── Backward ─────────────────────────────────────────────────────── //

    // Start backpropagation from a scalar output (size == 1).
    void backward();

    // Start backpropagation with an explicit upstream gradient.
    void backward(Tensor upstream_grad);

    // ── Arithmetic operations (build computation graph) ───────────────── //

    AutoTensor operator+(const AutoTensor& o) const;
    AutoTensor operator-(const AutoTensor& o) const;
    AutoTensor operator*(const AutoTensor& o) const;
    AutoTensor operator/(const AutoTensor& o) const;
    AutoTensor operator-()                    const;

    AutoTensor operator+(double s) const;
    AutoTensor operator-(double s) const;
    AutoTensor operator*(double s) const;
    AutoTensor operator/(double s) const;

    friend AutoTensor operator+(double s, const AutoTensor& t);
    friend AutoTensor operator*(double s, const AutoTensor& t);

    // ── Linear algebra ────────────────────────────────────────────────── //

    // Both tensors must be 2-D. Gradient flows to both operands.
    AutoTensor matmul(const AutoTensor& other) const;

    // Transpose for 2-D tensors.
    AutoTensor T() const;

    // ── Reductions → scalar node (shape {1}) ─────────────────────────── //

    AutoTensor sum()  const;
    AutoTensor mean() const;

    // ── Element-wise activations ──────────────────────────────────────── //

    AutoTensor relu()    const;
    AutoTensor sigmoid() const;
    AutoTensor tanh()    const;
    AutoTensor exp()     const;
    AutoTensor log()     const;
    AutoTensor pow(double exponent) const;

    // ── Internal shared state (needed by Module/Optimizer) ────────────── //

    std::shared_ptr<Impl> impl() const noexcept { return m_impl; }

private:
    std::shared_ptr<Impl> m_impl;
};

// Scalar-on-left helpers
SHAREDMATH_ML_EXPORT AutoTensor operator+(double s, const AutoTensor& t);
SHAREDMATH_ML_EXPORT AutoTensor operator*(double s, const AutoTensor& t);

} // namespace SharedMath::ML
