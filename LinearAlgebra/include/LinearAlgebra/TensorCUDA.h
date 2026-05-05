#pragma once
/// Internal header — NOT included by LinearAlgebra.h.
/// Only included by Tensor.cpp (behind the SHAREDMATH_CUDA guard) and TensorCUDA.cu.

#include "Tensor.h"

namespace SharedMath::LinearAlgebra::detail {

/// ── Op codes ──────────────────────────────────────────────────────────────────

enum class UnaryOp {
    Abs, Sqrt, Exp, Log, Log2, Log10,
    Sin, Cos, Tanh,
    Floor, Ceil, Round, Sign, Neg,
    Relu, Sigmoid, Gelu
};

enum class BinaryOp { Add, Sub, Mul, Div };
enum class ScalarOp { Add, Sub, Mul, Div, RSub, RDiv };

/// ── CUDA dispatch functions ───────────────────────────────────────────────────
/// All functions assume their Tensor arguments are already on GPU (Device::CUDA).

/// Matrix multiply via cuBLAS (both tensors must be 2-D).
Tensor cuda_matmul(const Tensor& a, const Tensor& b);

/// Element-wise binary op — a and b must have identical shapes.
Tensor cuda_binary(const Tensor& a, const Tensor& b, BinaryOp op);

/// Element-wise tensor/scalar op.
Tensor cuda_scalar(const Tensor& a, double scalar, ScalarOp op);

/// Element-wise unary op.
Tensor cuda_unary(const Tensor& a, UnaryOp op);

/// pow(a, exponent) element-wise.
Tensor cuda_pow(const Tensor& a, double exponent);

/// clip(a, lo, hi) element-wise.
Tensor cuda_clip(const Tensor& a, double lo, double hi);

/// Softmax along an arbitrary axis.
Tensor cuda_softmax(const Tensor& a, size_t axis);

// ML-oriented NCHW kernels.
Tensor cuda_conv2d(const Tensor& input, const Tensor& weight, const Tensor* bias,
                   size_t stride, size_t padding);
Tensor cuda_conv2d_backward_input(const Tensor& grad_out, const Tensor& weight,
                                  Tensor::Shape input_shape,
                                  size_t stride, size_t padding);
Tensor cuda_conv2d_backward_weight(const Tensor& grad_out, const Tensor& input,
                                   Tensor::Shape weight_shape,
                                   size_t stride, size_t padding);
Tensor cuda_conv2d_backward_bias(const Tensor& grad_out);
Tensor cuda_max_pool2d(const Tensor& input, size_t kernel_size,
                       size_t stride, size_t padding);
Tensor cuda_max_pool2d_backward(const Tensor& grad_out, const Tensor& input,
                                size_t kernel_size, size_t stride,
                                size_t padding);
Tensor cuda_avg_pool2d(const Tensor& input, size_t kernel_size,
                       size_t stride, size_t padding);
Tensor cuda_avg_pool2d_backward(const Tensor& grad_out,
                                Tensor::Shape input_shape,
                                size_t kernel_size, size_t stride,
                                size_t padding);

/// ── CUDA accessor (friend of Tensor) ─────────────────────────────────────────
/// Provides controlled access to Tensor private members for TensorCUDA.cu.
struct TensorCUDAImpl {
    /// Raw pointer to GPU buffer (nullptr if on CPU or CUDA disabled).
    static double* cuda_ptr(const Tensor& t);
    static double* buffer_ptr(const std::shared_ptr<Tensor::CUDABuffer>& buf);
    static int buffer_device_id(const std::shared_ptr<Tensor::CUDABuffer>& buf);
    static std::shared_ptr<Tensor::CUDABuffer> make_buffer(size_t n);
    static std::shared_ptr<Tensor::CUDABuffer> make_buffer(size_t n,
                                                           int device_id);
    static std::shared_ptr<Tensor::CUDABuffer> make_buffer(const double* host_src,
                                                           size_t n);
    static std::shared_ptr<Tensor::CUDABuffer> make_buffer(const double* host_src,
                                                           size_t n,
                                                           int device_id);

    // Wrap an existing GPU buffer into a new GPU Tensor.
    static Tensor make(Tensor::Shape shape,
                       std::shared_ptr<Tensor::CUDABuffer> buf) {
        int dev = buffer_device_id(buf);
        return Tensor::from_cuda(std::move(shape), std::move(buf), dev);
    }

    /// Host data vector (empty for GPU tensors).
    static const std::vector<double>& host_data(const Tensor& t) {
        return t.m_data;
    }

    /// Shape accessor (same as t.shape(), just here for symmetry).
    static const Tensor::Shape& shape(const Tensor& t) {
        return t.m_shape;
    }

    static int device_id(const Tensor& t) {
        return t.m_device_id;
    }
};

} // namespace SharedMath::LinearAlgebra::detail
