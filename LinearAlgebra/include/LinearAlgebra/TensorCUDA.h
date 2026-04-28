#pragma once
// Internal header — NOT included by LinearAlgebra.h.
// Only included by Tensor.cpp (behind #ifdef SHAREDMATH_CUDA) and TensorCUDA.cu.

#include "Tensor.h"

namespace SharedMath::LinearAlgebra::detail {

// ── Op codes ──────────────────────────────────────────────────────────────────

enum class UnaryOp {
    Abs, Sqrt, Exp, Log, Log2, Log10,
    Sin, Cos, Tanh,
    Floor, Ceil, Round, Sign, Neg
};

enum class BinaryOp { Add, Sub, Mul, Div };

// ── CUDA dispatch functions ───────────────────────────────────────────────────
// All functions assume their Tensor arguments are already on GPU (Device::CUDA).

// Matrix multiply via cuBLAS (both tensors must be 2-D).
Tensor cuda_matmul(const Tensor& a, const Tensor& b);

// Element-wise binary op — a and b must have identical shapes.
Tensor cuda_binary(const Tensor& a, const Tensor& b, BinaryOp op);

// Element-wise unary op.
Tensor cuda_unary(const Tensor& a, UnaryOp op);

// pow(a, exponent) element-wise.
Tensor cuda_pow(const Tensor& a, double exponent);

// clip(a, lo, hi) element-wise.
Tensor cuda_clip(const Tensor& a, double lo, double hi);

// TensorCUDAImpl is defined in TensorCUDA.cu (where CUDABuffer is complete).
// Forward-declare it here so Tensor.h's friend declaration resolves correctly.
struct TensorCUDAImpl;

} // namespace SharedMath::LinearAlgebra::detail
