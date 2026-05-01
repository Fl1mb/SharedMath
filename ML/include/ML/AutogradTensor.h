#pragma once

/**
 * @file AutogradTensor.h
 * @brief Automatic differentiation tensor — the core of the ML autograd engine.
 *
 * @defgroup ML_Autograd Autograd Engine
 * @ingroup ML
 * @{
 *
 * AutoTensor wraps a `LinearAlgebra::Tensor` in a reference-counted node that
 * participates in a dynamic computation graph.  When `requires_grad` is
 * `true` for at least one operand of an operation, the result node stores a
 * backward closure (`grad_fn`) that propagates the upstream gradient to the
 * inputs when `backward()` is called.
 *
 * ### Quick example
 * @code{.cpp}
 * using namespace SharedMath::ML;
 *
 * auto x = AutoTensor::from(Tensor::from_vector({2.0, 3.0}), true);
 * auto loss = (x * x).mean();   // builds graph: x -> x*x -> mean
 * loss.backward();              // populates x.grad()
 * // x.grad() == {4.0, 6.0}  (d/dx x² = 2x)
 * @endcode
 *
 * @}
 */

#include <sharedmath_ml_export.h>
#include "LinearAlgebra/Tensor.h"

#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>

namespace SharedMath::ML {

/**
 * @brief Reference-typed autograd node.
 *
 * Internally stores a `shared_ptr<Impl>` so all copies of the same AutoTensor
 * share gradient storage.  Backward closures captured by arithmetic operations
 * hold `shared_ptr<Impl>` pointers to inputs, keeping the graph alive until
 * `backward()` has finished propagating.
 *
 * @ingroup ML_Autograd
 */
class SHAREDMATH_ML_EXPORT AutoTensor {
public:
    using Tensor = SharedMath::LinearAlgebra::Tensor; ///< Underlying dense tensor type.

    // ── Internal node (public for capture in backward closures) ─────────── //

    /**
     * @brief Internal graph node — do not use directly.
     *
     * Holds the forward value, the accumulated gradient, and the backward
     * closure for non-leaf nodes.
     */
    struct Impl {
        Tensor data;          ///< Forward value of this node.
        Tensor grad;          ///< Accumulated gradient (valid after backward).
        bool   has_grad_     = false; ///< True once a gradient has been set.
        bool   requires_grad = false; ///< Track gradient for this node?
        bool   is_leaf       = true;  ///< True for user-created (non-result) nodes.

        /// Backward closure: propagates upstream gradient to inputs.
        std::function<void(const Tensor&)> grad_fn;

        /// Accumulate @p g into `grad`, initialising to zeros on first call.
        void accumulate_grad(const Tensor& g);

        /**
         * @brief Route @p upstream to the correct destination.
         *
         * - **Leaf node, requires_grad=true** → accumulate_grad().
         * - **Non-leaf node** → call grad_fn().
         */
        void propagate(const Tensor& upstream);
    };

    /// ── Construction ──────────────────────────────────────────────────────── //

    /// Default constructor — creates an empty, detached node.
    AutoTensor();

    /**
     * @brief Wrap an existing tensor.
     * @param data         The forward value.
     * @param requires_grad If true this leaf participates in gradient tracking.
     */
    explicit AutoTensor(Tensor data, bool requires_grad = false);

    /**
     * @brief Named constructor — equivalent to the explicit constructor.
     * @see AutoTensor(Tensor, bool)
     */
    static AutoTensor from(Tensor data, bool requires_grad = false);

    /**
     * @brief Low-level factory for operation outputs.
     *
     * Creates a non-leaf node whose gradient is computed by @p grad_fn.
     * Intended for use inside operation implementations, not by end users.
     *
     * @param data         Forward value of the result.
     * @param requires_grad Whether the result participates in the graph.
     * @param grad_fn      Backward closure; receives the upstream gradient.
     */
    static AutoTensor make_result(Tensor data,
                                  bool   requires_grad,
                                  std::function<void(const Tensor&)> grad_fn);

    /// ── Data & gradient access ─────────────────────────────────────────────── //

    const Tensor& data() const noexcept; ///< Read the forward value.
    Tensor&       data()       noexcept; ///< Modify the forward value (e.g. from an optimizer).

    bool requires_grad()               const noexcept; ///< True if gradient tracking is on.
    void set_requires_grad(bool value)       noexcept; ///< Enable/disable gradient tracking.

    bool          has_grad()   const noexcept; ///< True after at least one backward pass.
    const Tensor& grad()       const;          ///< Accumulated gradient (throws if not set).
    Tensor&       grad();                      ///< Mutable gradient access.
    void          set_grad(Tensor g);          ///< Overwrite the gradient.
    void          zero_grad();                 ///< Reset gradient to zeros.

    // ── Backward ──────────────────────────────────────────────────────────── //

    /**
     * @brief Start backpropagation from a **scalar** output.
     *
     * The tensor must have exactly one element (size == 1).  Seed gradient
     * is `Tensor::ones({1})`.
     *
     * @throws std::runtime_error if the tensor is not scalar.
     */
    void backward();

    /**
     * @brief Start backpropagation with an explicit upstream gradient.
     * @param upstream_grad Gradient of the scalar loss w.r.t. this tensor.
     */
    void backward(Tensor upstream_grad);

    // ── Arithmetic (builds computation graph) ─────────────────────────────── //

    AutoTensor operator+(const AutoTensor& o) const; ///< Element-wise addition.
    AutoTensor operator-(const AutoTensor& o) const; ///< Element-wise subtraction.
    AutoTensor operator*(const AutoTensor& o) const; ///< Element-wise multiplication.
    AutoTensor operator/(const AutoTensor& o) const; ///< Element-wise division.
    AutoTensor operator-()                    const; ///< Unary negation.

    AutoTensor operator+(double s) const; ///< Add scalar @p s to every element.
    AutoTensor operator-(double s) const; ///< Subtract scalar @p s from every element.
    AutoTensor operator*(double s) const; ///< Multiply every element by scalar @p s.
    AutoTensor operator/(double s) const; ///< Divide every element by scalar @p s.

    friend AutoTensor operator+(double s, const AutoTensor& t); ///< @p s + t
    friend AutoTensor operator*(double s, const AutoTensor& t); ///< @p s * t

    // ── Linear algebra ──────────────────────────────────────────────────────── //

    /**
     * @brief Matrix multiplication.
     *
     * Both tensors must be 2-D.  Gradient flows through both operands:
     * \f$ \nabla A = G B^\top \f$, \f$ \nabla B = A^\top G \f$.
     *
     * @throws std::invalid_argument if either tensor is not 2-D.
     */
    AutoTensor matmul(const AutoTensor& other) const;

    /**
     * @brief Transpose (2-D tensors only).
     * @throws std::invalid_argument if the tensor is not 2-D.
     */
    AutoTensor T() const;

    /// Reshape with gradient passthrough.
    AutoTensor reshape(Tensor::Shape new_shape) const;

    /// ── Reductions → scalar node (shape {1}) ──────────────────────────────── //

    AutoTensor sum()  const; ///< Sum all elements → scalar with gradient 1 everywhere.
    AutoTensor mean() const; ///< Mean of all elements → scalar with gradient 1/N everywhere.

    /// ── Element-wise activations ───────────────────────────────────────────── //

    AutoTensor relu()    const; ///< Rectified linear unit: max(0, x).
    AutoTensor sigmoid() const; ///< Logistic sigmoid: 1/(1+exp(-x)).
    AutoTensor tanh()    const; ///< Hyperbolic tangent.
    AutoTensor exp()     const; ///< Element-wise exponential.
    AutoTensor log()     const; ///< Element-wise natural logarithm.
    AutoTensor pow(double exponent) const; ///< Element-wise power x^n.

    /// ── Internal ─────────────────────────────────────────────────────────────── //

    /// Access the shared internal node (used by layers and optimizers).
    std::shared_ptr<Impl> impl() const noexcept { return m_impl; }

private:
    std::shared_ptr<Impl> m_impl; ///< Shared graph node.
};

/// @related AutoTensor
SHAREDMATH_ML_EXPORT AutoTensor operator+(double s, const AutoTensor& t);
/// @related AutoTensor
SHAREDMATH_ML_EXPORT AutoTensor operator*(double s, const AutoTensor& t);

} // namespace SharedMath::ML
