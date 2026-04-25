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

// ── CUDA accessor (friend of Tensor) ─────────────────────────────────────────
// Provides controlled access to Tensor private members for TensorCUDA.cu.
struct TensorCUDAImpl {
    // Raw pointer to GPU buffer (nullptr if on CPU or CUDA disabled).
    static double* cuda_ptr(const Tensor& t) {
        return t.m_cuda_buf ? t.m_cuda_buf->ptr : nullptr;
    }

    // Wrap an existing GPU buffer into a new GPU Tensor.
    static Tensor make(Tensor::Shape shape,
                       std::shared_ptr<Tensor::CUDABuffer> buf) {
        return Tensor::from_cuda(std::move(shape), std::move(buf));
    }

    // Host data vector (empty for GPU tensors).
    static const std::vector<double>& host_data(const Tensor& t) {
        return t.m_data;
    }

    // Shape accessor (same as t.shape(), just here for symmetry).
    static const Tensor::Shape& shape(const Tensor& t) {
        return t.m_shape;
    }
};

} // namespace SharedMath::LinearAlgebra::detail
