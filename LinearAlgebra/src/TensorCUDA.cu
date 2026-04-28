// TensorCUDA.cu — CUDA backend for Tensor.
// Compiled only when -DSHAREDMATH_ENABLE_CUDA=ON.
//
// Accelerated operations:
//   matmul   → cuBLAS dgemm  (row-major trick: C = A*B ↔ Cᵀ = Bᵀ*Aᵀ)
//   binary   → custom element-wise kernels (+, -, *, /)
//   unary    → custom element-wise kernels (exp, log, sqrt, tanh, …)
//   pow/clip → custom element-wise kernels

#include "TensorCUDA.h"   // internal — includes Tensor.h

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <memory>

// ── have_gpu forward declaration ─────────────────────────────────────────────
// Defined in the anonymous namespace below; declared here so it can be called
// by Tensor::cuda() before the anonymous namespace is parsed.
namespace { bool have_gpu() noexcept; }

namespace SharedMath::LinearAlgebra {

// ─── CUDABuffer ───────────────────────────────────────────────────────────────
// Defined here (translation unit where CUDA is available).

struct Tensor::CUDABuffer {
    double* ptr   = nullptr;
    size_t  count = 0;        // number of double elements

    // Allocate and copy from host.
    explicit CUDABuffer(const double* host_src, size_t n) : count(n) {
        cudaError_t err = cudaMalloc(&ptr, n * sizeof(double));
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("CUDABuffer: cudaMalloc failed: ") +
                cudaGetErrorString(err));
        if (host_src && n > 0) {
            err = cudaMemcpy(ptr, host_src, n * sizeof(double),
                             cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
                throw std::runtime_error(
                    std::string("CUDABuffer: cudaMemcpy H→D failed: ") +
                    cudaGetErrorString(err));
        }
    }

    // Allocate without copying (output buffer).
    explicit CUDABuffer(size_t n) : CUDABuffer(nullptr, n) {}

    ~CUDABuffer() { if (ptr) cudaFree(ptr); }

    // Non-copyable.
    CUDABuffer(const CUDABuffer&)            = delete;
    CUDABuffer& operator=(const CUDABuffer&) = delete;

    void to_host(double* dst) const {
        cudaError_t err = cudaMemcpy(dst, ptr, count * sizeof(double),
                                     cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("CUDABuffer: cudaMemcpy D→H failed: ") +
                cudaGetErrorString(err));
    }
};

// ─── TensorCUDAImpl ───────────────────────────────────────────────────────────
// Defined here (CUDABuffer is complete at this point).
// Declared friend of Tensor in Tensor.h so it can access private members.

namespace detail {

struct TensorCUDAImpl {
    // Raw pointer to the GPU buffer (nullptr if CPU or disabled).
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

    // Shape accessor.
    static const Tensor::Shape& shape(const Tensor& t) {
        return t.m_shape;
    }
};

} // namespace detail

// ─── Private factory ──────────────────────────────────────────────────────────

Tensor Tensor::from_cuda(Shape shape, std::shared_ptr<CUDABuffer> buf) {
    Tensor t;
    t.m_shape    = std::move(shape);
    t.m_cuda_buf = std::move(buf);
    t.m_device   = Device::CUDA;
    t.computeStrides();
    // m_data intentionally empty — GPU buffer is authoritative
    return t;
}

// ─── Transfer: .cuda() / .cpu() ──────────────────────────────────────────────

Tensor Tensor::cuda() const {
    if (m_device == Device::CUDA) return *this;   // already on GPU

    // Graceful fallback: no GPU available → stay on CPU silently.
    if (!have_gpu()) return *this;

    auto buf = std::make_shared<CUDABuffer>(m_data.data(), m_data.size());
    return from_cuda(m_shape, std::move(buf));
}

Tensor Tensor::cpu() const {
    if (m_device == Device::CPU) return *this;    // already on CPU

    size_t n = size();
    std::vector<double> host(n);
    m_cuda_buf->to_host(host.data());
    return Tensor(m_shape, std::move(host));
}

} // namespace SharedMath::LinearAlgebra

// ─── Runtime GPU availability ─────────────────────────────────────────────────

namespace SharedMath::LinearAlgebra {

bool cuda_is_available() noexcept {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

} // namespace SharedMath::LinearAlgebra

// ─── cuBLAS handle (thread-local singleton) ───────────────────────────────────

namespace {

// Returns false if no GPU is present — used to gate handle creation.
bool have_gpu() noexcept {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

cublasHandle_t& cublas_handle() {
    thread_local cublasHandle_t handle = []{
        cublasHandle_t h;
        cublasStatus_t s = cublasCreate(&h);
        if (s != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cuBLAS: cublasCreate failed — no GPU?");
        return h;
    }();
    return handle;
}

inline void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(msg) + ": " +
                                 cudaGetErrorString(err));
}

} // anonymous namespace

// ─── Kernel helpers ───────────────────────────────────────────────────────────

namespace {

constexpr int kBlock = 256;

// ── Unary kernels ─────────────────────────────────────────────────────────────

__global__ void k_neg  (const double* in, double* out, size_t n)
{ size_t i = blockIdx.x*kBlock+threadIdx.x; if(i<n) out[i]=-in[i]; }
__global__ void k_abs  (const double* in, double* out, size_t n)
{ size_t i = blockIdx.x*kBlock+threadIdx.x; if(i<n) out[i]=::fabs(in[i]); }
__global__ void k_sqrt (const double* in, double* out, size_t n)
{ size_t i = blockIdx.x*kBlock+threadIdx.x; if(i<n) out[i]=::sqrt(in[i]); }
__global__ void k_exp  (const double* in, double* out, size_t n)
{ size_t i = blockIdx.x*kBlock+threadIdx.x; if(i<n) out[i]=::exp(in[i]); }
__global__ void k_log  (const double* in, double* out, size_t n)
{ size_t i = blockIdx.x*kBlock+threadIdx.x; if(i<n) out[i]=::log(in[i]); }
__global__ void k_log2 (const double* in, double* out, size_t n)
{ size_t i = blockIdx.x*kBlock+threadIdx.x; if(i<n) out[i]=::log2(in[i]); }
__global__ void k_log10(const double* in, double* out, size_t n)
{ size_t i = blockIdx.x*kBlock+threadIdx.x; if(i<n) out[i]=::log10(in[i]); }
__global__ void k_sin  (const double* in, double* out, size_t n)
{ size_t i = blockIdx.x*kBlock+threadIdx.x; if(i<n) out[i]=::sin(in[i]); }
__global__ void k_cos  (const double* in, double* out, size_t n)
{ size_t i = blockIdx.x*kBlock+threadIdx.x; if(i<n) out[i]=::cos(in[i]); }
__global__ void k_tanh (const double* in, double* out, size_t n)
{ size_t i = blockIdx.x*kBlock+threadIdx.x; if(i<n) out[i]=::tanh(in[i]); }
__global__ void k_floor(const double* in, double* out, size_t n)
{ size_t i = blockIdx.x*kBlock+threadIdx.x; if(i<n) out[i]=::floor(in[i]); }
__global__ void k_ceil (const double* in, double* out, size_t n)
{ size_t i = blockIdx.x*kBlock+threadIdx.x; if(i<n) out[i]=::ceil(in[i]); }
__global__ void k_round(const double* in, double* out, size_t n)
{ size_t i = blockIdx.x*kBlock+threadIdx.x; if(i<n) out[i]=::round(in[i]); }
__global__ void k_sign (const double* in, double* out, size_t n)
{ size_t i = blockIdx.x*kBlock+threadIdx.x;
  if(i<n) out[i]=(in[i]>0.0)?1.0:((in[i]<0.0)?-1.0:0.0); }
__global__ void k_pow  (const double* in, double* out, size_t n, double e)
{ size_t i = blockIdx.x*kBlock+threadIdx.x; if(i<n) out[i]=::pow(in[i],e); }
__global__ void k_clip (const double* in, double* out, size_t n, double lo, double hi)
{ size_t i = blockIdx.x*kBlock+threadIdx.x;
  if(i<n) out[i]=in[i]<lo?lo:(in[i]>hi?hi:in[i]); }

// ── Binary kernels (element-wise, same shape) ─────────────────────────────────

__global__ void k_add(const double* a, const double* b, double* c, size_t n)
{ size_t i = blockIdx.x*kBlock+threadIdx.x; if(i<n) c[i]=a[i]+b[i]; }
__global__ void k_sub(const double* a, const double* b, double* c, size_t n)
{ size_t i = blockIdx.x*kBlock+threadIdx.x; if(i<n) c[i]=a[i]-b[i]; }
__global__ void k_mul(const double* a, const double* b, double* c, size_t n)
{ size_t i = blockIdx.x*kBlock+threadIdx.x; if(i<n) c[i]=a[i]*b[i]; }
__global__ void k_div(const double* a, const double* b, double* c, size_t n)
{ size_t i = blockIdx.x*kBlock+threadIdx.x; if(i<n) c[i]=a[i]/b[i]; }

} // anonymous namespace

// ─── CUDA dispatch implementations ───────────────────────────────────────────

namespace SharedMath::LinearAlgebra::detail {

using Acc = TensorCUDAImpl;   // shorthand for the accessor

// ── cuda_unary ────────────────────────────────────────────────────────────────

Tensor cuda_unary(const Tensor& a, UnaryOp op) {
    size_t n    = a.size();
    int    grid = static_cast<int>((n + kBlock - 1) / kBlock);

    const double* in  = Acc::cuda_ptr(a);
    auto out_buf = std::make_shared<Tensor::CUDABuffer>(n);
    double* out  = out_buf->ptr;

    switch (op) {
    case UnaryOp::Neg:   k_neg  <<<grid,kBlock>>>(in, out, n); break;
    case UnaryOp::Abs:   k_abs  <<<grid,kBlock>>>(in, out, n); break;
    case UnaryOp::Sqrt:  k_sqrt <<<grid,kBlock>>>(in, out, n); break;
    case UnaryOp::Exp:   k_exp  <<<grid,kBlock>>>(in, out, n); break;
    case UnaryOp::Log:   k_log  <<<grid,kBlock>>>(in, out, n); break;
    case UnaryOp::Log2:  k_log2 <<<grid,kBlock>>>(in, out, n); break;
    case UnaryOp::Log10: k_log10<<<grid,kBlock>>>(in, out, n); break;
    case UnaryOp::Sin:   k_sin  <<<grid,kBlock>>>(in, out, n); break;
    case UnaryOp::Cos:   k_cos  <<<grid,kBlock>>>(in, out, n); break;
    case UnaryOp::Tanh:  k_tanh <<<grid,kBlock>>>(in, out, n); break;
    case UnaryOp::Floor: k_floor<<<grid,kBlock>>>(in, out, n); break;
    case UnaryOp::Ceil:  k_ceil <<<grid,kBlock>>>(in, out, n); break;
    case UnaryOp::Round: k_round<<<grid,kBlock>>>(in, out, n); break;
    case UnaryOp::Sign:  k_sign <<<grid,kBlock>>>(in, out, n); break;
    }
    check_cuda(cudaDeviceSynchronize(), "cuda_unary");
    return Acc::make(Acc::shape(a), std::move(out_buf));
}

Tensor cuda_pow(const Tensor& a, double exponent) {
    size_t n    = a.size();
    int    grid = static_cast<int>((n + kBlock - 1) / kBlock);
    auto out_buf = std::make_shared<Tensor::CUDABuffer>(n);
    k_pow<<<grid, kBlock>>>(Acc::cuda_ptr(a), out_buf->ptr, n, exponent);
    check_cuda(cudaDeviceSynchronize(), "cuda_pow");
    return Acc::make(Acc::shape(a), std::move(out_buf));
}

Tensor cuda_clip(const Tensor& a, double lo, double hi) {
    size_t n    = a.size();
    int    grid = static_cast<int>((n + kBlock - 1) / kBlock);
    auto out_buf = std::make_shared<Tensor::CUDABuffer>(n);
    k_clip<<<grid, kBlock>>>(Acc::cuda_ptr(a), out_buf->ptr, n, lo, hi);
    check_cuda(cudaDeviceSynchronize(), "cuda_clip");
    return Acc::make(Acc::shape(a), std::move(out_buf));
}

// ── cuda_binary ───────────────────────────────────────────────────────────────

Tensor cuda_binary(const Tensor& a, const Tensor& b, BinaryOp op) {
    size_t n    = a.size();
    int    grid = static_cast<int>((n + kBlock - 1) / kBlock);

    const double* pa = Acc::cuda_ptr(a);
    const double* pb = Acc::cuda_ptr(b);
    auto out_buf = std::make_shared<Tensor::CUDABuffer>(n);
    double* pc   = out_buf->ptr;

    switch (op) {
    case BinaryOp::Add: k_add<<<grid,kBlock>>>(pa, pb, pc, n); break;
    case BinaryOp::Sub: k_sub<<<grid,kBlock>>>(pa, pb, pc, n); break;
    case BinaryOp::Mul: k_mul<<<grid,kBlock>>>(pa, pb, pc, n); break;
    case BinaryOp::Div: k_div<<<grid,kBlock>>>(pa, pb, pc, n); break;
    }
    check_cuda(cudaDeviceSynchronize(), "cuda_binary");
    return Acc::make(Acc::shape(a), std::move(out_buf));
}

// ── cuda_matmul ───────────────────────────────────────────────────────────────
// Row-major A (m×k) * B (k×n) = C (m×n).
// cuBLAS is column-major, so we compute Cᵀ = Bᵀ * Aᵀ:
//   treat row-major A as column-major Aᵀ (k×m), lda = k
//   treat row-major B as column-major Bᵀ (n×k), ldb = n
//   result is column-major Cᵀ (n×m) = row-major C (m×n), ldc = n

Tensor cuda_matmul(const Tensor& a, const Tensor& b) {
    size_t m = Acc::shape(a)[0];
    size_t k = Acc::shape(a)[1];
    size_t n = Acc::shape(b)[1];

    auto out_buf = std::make_shared<Tensor::CUDABuffer>(m * n);

    const double alpha = 1.0, beta = 0.0;
    cublasStatus_t st = cublasDgemm(
        cublas_handle(),
        CUBLAS_OP_N, CUBLAS_OP_N,
        static_cast<int>(n),   // rows of op(B^T) = cols of result
        static_cast<int>(m),   // cols of op(A^T) = rows of result
        static_cast<int>(k),   // inner dimension
        &alpha,
        Acc::cuda_ptr(b), static_cast<int>(n),   // B row-major → B^T col-major, ldb = n
        Acc::cuda_ptr(a), static_cast<int>(k),   // A row-major → A^T col-major, lda = k
        &beta,
        out_buf->ptr,     static_cast<int>(n)    // C^T col-major, ldc = n
    );
    if (st != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuda_matmul: cublasDgemm failed");

    check_cuda(cudaDeviceSynchronize(), "cuda_matmul");
    return Acc::make({m, n}, std::move(out_buf));
}

} // namespace SharedMath::LinearAlgebra::detail
