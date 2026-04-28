#pragma once

#include "constans.h"
#include <sharedmath_linearalgebra_export.h>

#include <vector>
#include <functional>
#include <string>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <limits>
#include <cmath>
#include <memory>
#include <ostream>

namespace SharedMath::LinearAlgebra {

// Compute device. CUDA is only functional when the library is built with
// -DSHAREDMATH_ENABLE_CUDA=ON. On CPU-only builds, .cuda() is a no-op.
enum class SHAREDMATH_LINEARALGEBRA_EXPORT Device { CPU, CUDA };

namespace detail { struct TensorCUDAImpl; }   // forward-declared CUDA accessor

// N-dimensional dense tensor with row-major (C-contiguous) storage.
// Supports NumPy-style broadcasting, axis reductions, and element-wise math.
// When built with CUDA support, tensors can be moved to GPU with .cuda() and
// back with .cpu(). Operations between two GPU tensors are dispatched to cuBLAS
// / custom CUDA kernels automatically.
class SHAREDMATH_LINEARALGEBRA_EXPORT Tensor {
public:
    using Shape = std::vector<size_t>;

    // ------------------------------------------------------------------ //
    // Construction
    // ------------------------------------------------------------------ //

    Tensor() = default;
    explicit Tensor(Shape shape, double fill = 0.0);
    Tensor(Shape shape, std::vector<double> data);

    // Static factories
    static Tensor zeros(Shape shape);
    static Tensor ones(Shape shape);
    static Tensor eye(size_t n);
    static Tensor arange(double start, double stop, double step = 1.0);
    static Tensor linspace(double start, double stop, size_t num);
    static Tensor from_vector(const std::vector<double>& v);
    static Tensor from_matrix(size_t rows, size_t cols,
                              const std::vector<double>& flat_row_major);

    // ------------------------------------------------------------------ //
    // Shape & metadata
    // ------------------------------------------------------------------ //

    const Shape& shape()           const noexcept { return m_shape; }
    size_t        ndim()           const noexcept { return m_shape.size(); }
    // Total element count — valid for both CPU and GPU tensors.
    size_t        size()           const noexcept;
    size_t        dim(size_t axis) const;
    bool          empty()          const noexcept { return size() == 0; }

    // ── Device management ─────────────────────────────────────────────── //

    Device device() const noexcept { return m_device; }

    // Transfer to GPU (copy host→device). No-op if already on GPU.
    // When built without CUDA support, returns *this unchanged.
    Tensor cuda() const;

    // Transfer to CPU (copy device→host). No-op if already on CPU.
    Tensor cpu()  const;

    // Convert flat index to multi-index (row-major)
    std::vector<size_t> unravel(size_t flat_idx) const;

    // ------------------------------------------------------------------ //
    // Element access
    // ------------------------------------------------------------------ //

    // t(i, j, k) — variadic, bounds-checked
    template<typename... Idx>
    double& operator()(Idx... indices) {
        return m_data[flatIndex({ static_cast<size_t>(indices)... })];
    }
    template<typename... Idx>
    double operator()(Idx... indices) const {
        return m_data[flatIndex({ static_cast<size_t>(indices)... })];
    }

    double& at(const std::vector<size_t>& idx);
    double  at(const std::vector<size_t>& idx) const;

    // Raw flat access — only valid for CPU tensors; throws for GPU tensors.
    double& flat(size_t i);
    double  flat(size_t i) const;

    // Host data vector (empty for GPU tensors — use .cpu().data() to fetch).
    const std::vector<double>& data()  const noexcept { return m_data; }
    std::vector<double>&       data()        noexcept { return m_data; }

    // ------------------------------------------------------------------ //
    // Shape operations
    // ------------------------------------------------------------------ //

    Tensor reshape(Shape new_shape)            const;
    Tensor flatten()                           const;
    Tensor squeeze()                           const;  // remove all size-1 dims
    Tensor expand_dims(size_t axis)            const;
    Tensor transpose()                         const;  // reverse all axes
    Tensor transpose(std::vector<size_t> axes) const;  // custom permutation
    Tensor slice(size_t axis, size_t start, size_t end) const;

    // ------------------------------------------------------------------ //
    // Arithmetic — element-wise with NumPy broadcasting
    // ------------------------------------------------------------------ //

    Tensor operator+(const Tensor& o) const;
    Tensor operator-(const Tensor& o) const;
    Tensor operator*(const Tensor& o) const;
    Tensor operator/(const Tensor& o) const;

    Tensor operator+(double s) const;
    Tensor operator-(double s) const;
    Tensor operator*(double s) const;
    Tensor operator/(double s) const;
    Tensor operator-()         const;

    Tensor& operator+=(const Tensor& o);
    Tensor& operator-=(const Tensor& o);
    Tensor& operator*=(double s);
    Tensor& operator/=(double s);

    bool operator==(const Tensor& o) const;
    bool operator!=(const Tensor& o) const;

    // ------------------------------------------------------------------ //
    // Global reductions
    // ------------------------------------------------------------------ //

    double sum()                    const;
    double product()                const;
    double min()                    const;
    double max()                    const;
    double mean()                   const;
    double var(bool ddof = false)   const;    // ddof=true → sample variance
    double stddev(bool ddof = false) const;
    size_t argmin()                 const;
    size_t argmax()                 const;

    // Reductions along one axis (output rank = ndim - 1)
    Tensor sum(size_t axis)  const;
    Tensor min(size_t axis)  const;
    Tensor max(size_t axis)  const;
    Tensor mean(size_t axis) const;

    // ------------------------------------------------------------------ //
    // Element-wise math
    // ------------------------------------------------------------------ //

    Tensor apply(std::function<double(double)> f) const;
    Tensor abs()                      const;
    Tensor sqrt()                     const;
    Tensor exp()                      const;
    Tensor log()                      const;
    Tensor log2()                     const;
    Tensor log10()                    const;
    Tensor pow(double exponent)       const;
    Tensor clip(double lo, double hi) const;
    Tensor sign()                     const;
    Tensor floor()                    const;
    Tensor ceil()                     const;
    Tensor round()                    const;
    Tensor sin()                      const;
    Tensor cos()                      const;
    Tensor tanh()                     const;

    // ------------------------------------------------------------------ //
    // Linear algebra (primarily for 2-D tensors)
    // ------------------------------------------------------------------ //

    // Matrix multiply (both must be 2-D)
    Tensor matmul(const Tensor& other) const;

    // Sum of main-diagonal elements (square 2-D tensor)
    double trace() const;

    // 1-D → diagonal matrix; 2-D → extract main diagonal as 1-D tensor
    Tensor diag() const;

    // ------------------------------------------------------------------ //
    // Display
    // ------------------------------------------------------------------ //

    std::string str() const;

    // ── GPU buffer type ───────────────────────────────────────────────── //
    // Forward-declared public so that TensorCUDA.cu free functions can use
    // std::make_shared<CUDABuffer>.  The struct itself is completed only in
    // TensorCUDA.cu — no CUDA headers are needed here.
    struct CUDABuffer;

private:
    // ── CPU storage ───────────────────────────────────────────────────── //
    Shape               m_shape;
    std::vector<double> m_data;      // empty when tensor is on GPU
    std::vector<size_t> m_strides;

    // ── GPU storage ──────────────────────────────────────────────────────//
    std::shared_ptr<CUDABuffer> m_cuda_buf;     // null → CPU tensor
    Device m_device = Device::CPU;

    // ── Helpers ───────────────────────────────────────────────────────── //
    void   computeStrides();
    size_t flatIndex(const std::vector<size_t>& idx) const;

    static Shape broadcastShape(const Shape& a, const Shape& b);
    Tensor broadcastOp(const Tensor& other,
                       std::function<double(double, double)> op) const;
    Tensor axisReduce(size_t axis,
                      std::function<double(double, double)> reducer,
                      double init) const;

    // Private factory used by TensorCUDA.cu to wrap a GPU buffer
    static Tensor from_cuda(Shape shape, std::shared_ptr<CUDABuffer> buf);

    friend struct detail::TensorCUDAImpl;   // CUDA implementation accessor
};

// Scalar-on-left arithmetic
SHAREDMATH_LINEARALGEBRA_EXPORT Tensor operator+(double s, const Tensor& t);
SHAREDMATH_LINEARALGEBRA_EXPORT Tensor operator-(double s, const Tensor& t);
SHAREDMATH_LINEARALGEBRA_EXPORT Tensor operator*(double s, const Tensor& t);
SHAREDMATH_LINEARALGEBRA_EXPORT Tensor operator/(double s, const Tensor& t);
SHAREDMATH_LINEARALGEBRA_EXPORT std::ostream& operator<<(std::ostream& os, const Tensor& t);

} // namespace SharedMath::LinearAlgebra
