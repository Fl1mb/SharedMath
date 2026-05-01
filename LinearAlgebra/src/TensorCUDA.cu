// TensorCUDA.cu — CUDA backend for Tensor.
// Compiled only when -DSHAREDMATH_ENABLE_CUDA=ON.
//
// Accelerated operations:
//   matmul   → cuBLAS dgemm  (row-major trick: C = A*B ↔ Cᵀ = Bᵀ*Aᵀ)
//   binary   → custom element-wise kernels (+, -, *, /)
//   unary    → custom element-wise kernels (exp, log, sqrt, tanh, …)
//   pow/clip → custom element-wise kernels

#include "TensorCUDA.h"   // internal — includes Tensor.h
#include "core/CudaDeviceManager.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <memory>

// ─── GPU availability check (used by Tensor::cuda() below) ───────────────────
namespace {

bool have_gpu() noexcept {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

} // anonymous namespace

namespace SharedMath::LinearAlgebra {

// ─── CUDABuffer ───────────────────────────────────────────────────────────────
// Defined here (translation unit where CUDA is available).

struct Tensor::CUDABuffer {
    double* ptr   = nullptr;
    size_t  count = 0;        // number of double elements
    int     device_id = 0;

    // Allocate and copy from host.
    explicit CUDABuffer(const double* host_src, size_t n, int dev = 0)
        : count(n), device_id(dev) {
        cudaSetDevice(device_id);
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
    explicit CUDABuffer(size_t n, int dev = 0) : CUDABuffer(nullptr, n, dev) {}

    ~CUDABuffer() {
        if (ptr) {
            cudaSetDevice(device_id);
            cudaFree(ptr);
        }
    }

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

namespace detail {

double* TensorCUDAImpl::cuda_ptr(const Tensor& t) {
    return t.m_cuda_buf ? t.m_cuda_buf->ptr : nullptr;
}

double* TensorCUDAImpl::buffer_ptr(
    const std::shared_ptr<Tensor::CUDABuffer>& buf)
{
    return buf ? buf->ptr : nullptr;
}

int TensorCUDAImpl::buffer_device_id(
    const std::shared_ptr<Tensor::CUDABuffer>& buf)
{
    return buf ? buf->device_id : -1;
}

std::shared_ptr<Tensor::CUDABuffer> TensorCUDAImpl::make_buffer(size_t n) {
    int dev = 0;
    cudaGetDevice(&dev);
    return std::make_shared<Tensor::CUDABuffer>(n, dev);
}

std::shared_ptr<Tensor::CUDABuffer> TensorCUDAImpl::make_buffer(size_t n,
                                                                int device_id) {
    return std::make_shared<Tensor::CUDABuffer>(n, device_id);
}

std::shared_ptr<Tensor::CUDABuffer> TensorCUDAImpl::make_buffer(
    const double* host_src, size_t n)
{
    int dev = 0;
    cudaGetDevice(&dev);
    return std::make_shared<Tensor::CUDABuffer>(host_src, n, dev);
}

std::shared_ptr<Tensor::CUDABuffer> TensorCUDAImpl::make_buffer(
    const double* host_src, size_t n, int device_id)
{
    return std::make_shared<Tensor::CUDABuffer>(host_src, n, device_id);
}

} // namespace detail

// ─── Private factory ──────────────────────────────────────────────────────────

Tensor Tensor::from_cuda(Shape shape, std::shared_ptr<CUDABuffer> buf,
                         int device_id) {
    Tensor t;
    t.m_shape    = std::move(shape);
    t.m_cuda_buf = std::move(buf);
    t.m_device   = Device::CUDA;
    t.m_device_id = device_id;
    t.computeStrides();
    // m_data intentionally empty — GPU buffer is authoritative
    return t;
}

// ─── Transfer: .cuda() / .cpu() ──────────────────────────────────────────────

Tensor Tensor::cuda() const {
    return cuda_auto();
}

Tensor Tensor::cuda_auto() const {
    int dev = Core::CudaDeviceManager::instance().leastLoadedDevice();
    if (dev < 0) return *this;
    return cuda(dev);
}

Tensor Tensor::cuda(int device_id) const {
    if (m_device == Device::CUDA) {
        if (device_id < 0 || device_id == m_device_id) return *this;
        return cpu().cuda(device_id);
    }

    // Graceful fallback: no GPU available → stay on CPU silently.
    if (!have_gpu()) return *this;

    auto& mgr = Core::CudaDeviceManager::instance();
    if (device_id < 0) device_id = mgr.leastLoadedDevice();
    if (device_id < 0 || device_id >= mgr.deviceCount())
        throw std::out_of_range("Tensor::cuda: invalid CUDA device id");

    auto buf = detail::TensorCUDAImpl::make_buffer(m_data.data(), m_data.size(),
                                                   device_id);
    return from_cuda(m_shape, std::move(buf), device_id);
}

Tensor Tensor::cpu() const {
    if (m_device == Device::CPU) return *this;    // already on CPU

    size_t n = size();
    std::vector<double> host(n);
    m_cuda_buf->to_host(host.data());
    return Tensor(m_shape, std::move(host));
}

Tensor Tensor::to(Device device, int device_id) const {
    if (device == Device::CPU) return cpu();
    return device_id >= 0 ? cuda(device_id) : cuda_auto();
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
__global__ void k_relu (const double* in, double* out, size_t n)
{ size_t i = blockIdx.x*kBlock+threadIdx.x; if(i<n) out[i]=in[i]>0.0?in[i]:0.0; }
__global__ void k_sigmoid(const double* in, double* out, size_t n)
{ size_t i = blockIdx.x*kBlock+threadIdx.x; if(i<n) out[i]=1.0/(1.0+::exp(-in[i])); }
__global__ void k_gelu(const double* in, double* out, size_t n)
{ size_t i = blockIdx.x*kBlock+threadIdx.x;
  if(i<n) out[i]=in[i]*0.5*(1.0+::erf(in[i]/::sqrt(2.0))); }
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

__global__ void k_scalar(const double* a, double* out, size_t n, double s, int op)
{
    size_t i = blockIdx.x*kBlock+threadIdx.x;
    if (i >= n) return;
    double v = a[i];
    switch (op) {
    case 0: out[i] = v + s; break;
    case 1: out[i] = v - s; break;
    case 2: out[i] = v * s; break;
    case 3: out[i] = v / s; break;
    case 4: out[i] = s - v; break;
    case 5: out[i] = s / v; break;
    }
}

__global__ void k_softmax(const double* in, double* out, size_t outer,
                          size_t axis_dim, size_t inner)
{
    size_t idx = blockIdx.x*kBlock + threadIdx.x;
    size_t total = outer * inner;
    if (idx >= total) return;

    size_t o = idx / inner;
    size_t inr = idx % inner;
    size_t base = o * axis_dim * inner + inr;

    double m = -INFINITY;
    for (size_t a = 0; a < axis_dim; ++a)
        m = ::fmax(m, in[base + a * inner]);

    double sum = 0.0;
    for (size_t a = 0; a < axis_dim; ++a) {
        double e = ::exp(in[base + a * inner] - m);
        out[base + a * inner] = e;
        sum += e;
    }
    for (size_t a = 0; a < axis_dim; ++a)
        out[base + a * inner] /= sum;
}

__device__ inline size_t flat4d(size_t n, size_t c, size_t h, size_t w,
                                size_t C, size_t H, size_t W)
{
    return ((n * C + c) * H + h) * W + w;
}

__global__ void k_conv2d_forward(const double* x, const double* w,
                                 const double* b, double* out,
                                 size_t N, size_t C, size_t H, size_t W,
                                 size_t OC, size_t K, size_t OH, size_t OW,
                                 size_t stride, size_t padding, bool use_bias)
{
    size_t idx = blockIdx.x*kBlock + threadIdx.x;
    size_t total = N * OC * OH * OW;
    if (idx >= total) return;

    size_t ow = idx % OW;
    size_t oh = (idx / OW) % OH;
    size_t oc = (idx / (OW * OH)) % OC;
    size_t n  = idx / (OW * OH * OC);

    double sum = use_bias ? b[oc] : 0.0;
    for (size_t ic = 0; ic < C; ++ic) {
        for (size_t kh = 0; kh < K; ++kh) {
            long ih = static_cast<long>(oh * stride + kh) - static_cast<long>(padding);
            if (ih < 0 || ih >= static_cast<long>(H)) continue;
            for (size_t kw = 0; kw < K; ++kw) {
                long iw = static_cast<long>(ow * stride + kw) - static_cast<long>(padding);
                if (iw < 0 || iw >= static_cast<long>(W)) continue;
                size_t xidx = flat4d(n, ic, static_cast<size_t>(ih), static_cast<size_t>(iw), C, H, W);
                size_t widx = ((oc * C + ic) * K + kh) * K + kw;
                sum += x[xidx] * w[widx];
            }
        }
    }
    out[idx] = sum;
}

__global__ void k_conv2d_backward_input(const double* go, const double* w,
                                        double* dx,
                                        size_t N, size_t C, size_t H, size_t W,
                                        size_t OC, size_t K, size_t OH, size_t OW,
                                        size_t stride, size_t padding)
{
    size_t idx = blockIdx.x*kBlock + threadIdx.x;
    size_t total = N * C * H * W;
    if (idx >= total) return;

    size_t iw = idx % W;
    size_t ih = (idx / W) % H;
    size_t ic = (idx / (W * H)) % C;
    size_t n  = idx / (W * H * C);

    double sum = 0.0;
    for (size_t oc = 0; oc < OC; ++oc) {
        for (size_t kh = 0; kh < K; ++kh) {
            long oh_num = static_cast<long>(ih + padding) - static_cast<long>(kh);
            if (oh_num < 0 || oh_num % static_cast<long>(stride) != 0) continue;
            size_t oh = static_cast<size_t>(oh_num / static_cast<long>(stride));
            if (oh >= OH) continue;
            for (size_t kw = 0; kw < K; ++kw) {
                long ow_num = static_cast<long>(iw + padding) - static_cast<long>(kw);
                if (ow_num < 0 || ow_num % static_cast<long>(stride) != 0) continue;
                size_t ow = static_cast<size_t>(ow_num / static_cast<long>(stride));
                if (ow >= OW) continue;
                size_t goidx = ((n * OC + oc) * OH + oh) * OW + ow;
                size_t widx = ((oc * C + ic) * K + kh) * K + kw;
                sum += go[goidx] * w[widx];
            }
        }
    }
    dx[idx] = sum;
}

__global__ void k_conv2d_backward_weight(const double* go, const double* x,
                                         double* dw,
                                         size_t N, size_t C, size_t H, size_t W,
                                         size_t OC, size_t K, size_t OH, size_t OW,
                                         size_t stride, size_t padding)
{
    size_t idx = blockIdx.x*kBlock + threadIdx.x;
    size_t total = OC * C * K * K;
    if (idx >= total) return;

    size_t kw = idx % K;
    size_t kh = (idx / K) % K;
    size_t ic = (idx / (K * K)) % C;
    size_t oc = idx / (K * K * C);

    double sum = 0.0;
    for (size_t n = 0; n < N; ++n)
        for (size_t oh = 0; oh < OH; ++oh) {
            long ih = static_cast<long>(oh * stride + kh) - static_cast<long>(padding);
            if (ih < 0 || ih >= static_cast<long>(H)) continue;
            for (size_t ow = 0; ow < OW; ++ow) {
                long iw = static_cast<long>(ow * stride + kw) - static_cast<long>(padding);
                if (iw < 0 || iw >= static_cast<long>(W)) continue;
                size_t goidx = ((n * OC + oc) * OH + oh) * OW + ow;
                size_t xidx = flat4d(n, ic, static_cast<size_t>(ih), static_cast<size_t>(iw), C, H, W);
                sum += go[goidx] * x[xidx];
            }
        }
    dw[idx] = sum;
}

__global__ void k_conv2d_backward_bias(const double* go, double* db,
                                       size_t N, size_t OC, size_t OH, size_t OW)
{
    size_t oc = blockIdx.x*kBlock + threadIdx.x;
    if (oc >= OC) return;
    double sum = 0.0;
    for (size_t n = 0; n < N; ++n)
        for (size_t oh = 0; oh < OH; ++oh)
            for (size_t ow = 0; ow < OW; ++ow)
                sum += go[((n * OC + oc) * OH + oh) * OW + ow];
    db[oc] = sum;
}

__global__ void k_max_pool2d(const double* x, double* out,
                             size_t N, size_t C, size_t H, size_t W,
                             size_t K, size_t OH, size_t OW,
                             size_t stride, size_t padding)
{
    size_t idx = blockIdx.x*kBlock + threadIdx.x;
    size_t total = N * C * OH * OW;
    if (idx >= total) return;
    size_t ow = idx % OW;
    size_t oh = (idx / OW) % OH;
    size_t c = (idx / (OW * OH)) % C;
    size_t n = idx / (OW * OH * C);
    double best = -INFINITY;
    for (size_t kh = 0; kh < K; ++kh) {
        long ih = static_cast<long>(oh * stride + kh) - static_cast<long>(padding);
        if (ih < 0 || ih >= static_cast<long>(H)) continue;
        for (size_t kw = 0; kw < K; ++kw) {
            long iw = static_cast<long>(ow * stride + kw) - static_cast<long>(padding);
            if (iw < 0 || iw >= static_cast<long>(W)) continue;
            best = ::fmax(best, x[flat4d(n, c, static_cast<size_t>(ih), static_cast<size_t>(iw), C, H, W)]);
        }
    }
    out[idx] = best;
}

__global__ void k_avg_pool2d(const double* x, double* out,
                             size_t N, size_t C, size_t H, size_t W,
                             size_t K, size_t OH, size_t OW,
                             size_t stride, size_t padding)
{
    size_t idx = blockIdx.x*kBlock + threadIdx.x;
    size_t total = N * C * OH * OW;
    if (idx >= total) return;
    size_t ow = idx % OW;
    size_t oh = (idx / OW) % OH;
    size_t c = (idx / (OW * OH)) % C;
    size_t n = idx / (OW * OH * C);
    double sum = 0.0, count = 0.0;
    for (size_t kh = 0; kh < K; ++kh) {
        long ih = static_cast<long>(oh * stride + kh) - static_cast<long>(padding);
        if (ih < 0 || ih >= static_cast<long>(H)) continue;
        for (size_t kw = 0; kw < K; ++kw) {
            long iw = static_cast<long>(ow * stride + kw) - static_cast<long>(padding);
            if (iw < 0 || iw >= static_cast<long>(W)) continue;
            sum += x[flat4d(n, c, static_cast<size_t>(ih), static_cast<size_t>(iw), C, H, W)];
            count += 1.0;
        }
    }
    out[idx] = count > 0.0 ? sum / count : 0.0;
}

__global__ void k_avg_pool2d_backward(const double* go, double* dx,
                                      size_t N, size_t C, size_t H, size_t W,
                                      size_t K, size_t OH, size_t OW,
                                      size_t stride, size_t padding)
{
    size_t idx = blockIdx.x*kBlock + threadIdx.x;
    size_t total = N * C * H * W;
    if (idx >= total) return;
    size_t iw = idx % W;
    size_t ih = (idx / W) % H;
    size_t c = (idx / (W * H)) % C;
    size_t n = idx / (W * H * C);
    double sum = 0.0;
    for (size_t oh = 0; oh < OH; ++oh)
        for (size_t ow = 0; ow < OW; ++ow) {
            long kh = static_cast<long>(ih + padding) - static_cast<long>(oh * stride);
            long kw = static_cast<long>(iw + padding) - static_cast<long>(ow * stride);
            if (kh < 0 || kw < 0 || kh >= static_cast<long>(K) || kw >= static_cast<long>(K)) continue;
            double count = 0.0;
            for (size_t rkh = 0; rkh < K; ++rkh) {
                long rih = static_cast<long>(oh * stride + rkh) - static_cast<long>(padding);
                if (rih < 0 || rih >= static_cast<long>(H)) continue;
                for (size_t rkw = 0; rkw < K; ++rkw) {
                    long riw = static_cast<long>(ow * stride + rkw) - static_cast<long>(padding);
                    if (riw < 0 || riw >= static_cast<long>(W)) continue;
                    count += 1.0;
                }
            }
            sum += go[((n * C + c) * OH + oh) * OW + ow] / count;
        }
    dx[idx] = sum;
}

__global__ void k_max_pool2d_backward(const double* go, const double* x, double* dx,
                                      size_t N, size_t C, size_t H, size_t W,
                                      size_t K, size_t OH, size_t OW,
                                      size_t stride, size_t padding)
{
    size_t idx = blockIdx.x*kBlock + threadIdx.x;
    size_t total = N * C * OH * OW;
    if (idx >= total) return;
    size_t ow = idx % OW;
    size_t oh = (idx / OW) % OH;
    size_t c = (idx / (OW * OH)) % C;
    size_t n = idx / (OW * OH * C);
    double best = -INFINITY;
    size_t best_idx = 0;
    for (size_t kh = 0; kh < K; ++kh) {
        long ih = static_cast<long>(oh * stride + kh) - static_cast<long>(padding);
        if (ih < 0 || ih >= static_cast<long>(H)) continue;
        for (size_t kw = 0; kw < K; ++kw) {
            long iw = static_cast<long>(ow * stride + kw) - static_cast<long>(padding);
            if (iw < 0 || iw >= static_cast<long>(W)) continue;
            size_t xidx = flat4d(n, c, static_cast<size_t>(ih), static_cast<size_t>(iw), C, H, W);
            if (x[xidx] > best) { best = x[xidx]; best_idx = xidx; }
        }
    }
    atomicAdd(&dx[best_idx], go[idx]);
}

} // anonymous namespace

// ─── CUDA dispatch implementations ───────────────────────────────────────────

namespace SharedMath::LinearAlgebra::detail {

using Acc = TensorCUDAImpl;   // shorthand for the accessor

// ── cuda_unary ────────────────────────────────────────────────────────────────

Tensor cuda_unary(const Tensor& a, UnaryOp op) {
    cudaSetDevice(Acc::device_id(a));
    size_t n    = a.size();
    int    grid = static_cast<int>((n + kBlock - 1) / kBlock);

    const double* in  = Acc::cuda_ptr(a);
    auto out_buf = Acc::make_buffer(n);
    double* out  = Acc::buffer_ptr(out_buf);

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
    case UnaryOp::Relu:  k_relu <<<grid,kBlock>>>(in, out, n); break;
    case UnaryOp::Sigmoid: k_sigmoid<<<grid,kBlock>>>(in, out, n); break;
    case UnaryOp::Gelu:  k_gelu <<<grid,kBlock>>>(in, out, n); break;
    case UnaryOp::Floor: k_floor<<<grid,kBlock>>>(in, out, n); break;
    case UnaryOp::Ceil:  k_ceil <<<grid,kBlock>>>(in, out, n); break;
    case UnaryOp::Round: k_round<<<grid,kBlock>>>(in, out, n); break;
    case UnaryOp::Sign:  k_sign <<<grid,kBlock>>>(in, out, n); break;
    }
    check_cuda(cudaDeviceSynchronize(), "cuda_unary");
    return Acc::make(Acc::shape(a), std::move(out_buf));
}

Tensor cuda_pow(const Tensor& a, double exponent) {
    cudaSetDevice(Acc::device_id(a));
    size_t n    = a.size();
    int    grid = static_cast<int>((n + kBlock - 1) / kBlock);
    auto out_buf = Acc::make_buffer(n);
    k_pow<<<grid, kBlock>>>(Acc::cuda_ptr(a), Acc::buffer_ptr(out_buf), n, exponent);
    check_cuda(cudaDeviceSynchronize(), "cuda_pow");
    return Acc::make(Acc::shape(a), std::move(out_buf));
}

Tensor cuda_clip(const Tensor& a, double lo, double hi) {
    cudaSetDevice(Acc::device_id(a));
    size_t n    = a.size();
    int    grid = static_cast<int>((n + kBlock - 1) / kBlock);
    auto out_buf = Acc::make_buffer(n);
    k_clip<<<grid, kBlock>>>(Acc::cuda_ptr(a), Acc::buffer_ptr(out_buf), n, lo, hi);
    check_cuda(cudaDeviceSynchronize(), "cuda_clip");
    return Acc::make(Acc::shape(a), std::move(out_buf));
}

// ── cuda_binary ───────────────────────────────────────────────────────────────

Tensor cuda_binary(const Tensor& a, const Tensor& b, BinaryOp op) {
    cudaSetDevice(Acc::device_id(a));
    size_t n    = a.size();
    int    grid = static_cast<int>((n + kBlock - 1) / kBlock);

    const double* pa = Acc::cuda_ptr(a);
    const double* pb = Acc::cuda_ptr(b);
    auto out_buf = Acc::make_buffer(n);
    double* pc   = Acc::buffer_ptr(out_buf);

    switch (op) {
    case BinaryOp::Add: k_add<<<grid,kBlock>>>(pa, pb, pc, n); break;
    case BinaryOp::Sub: k_sub<<<grid,kBlock>>>(pa, pb, pc, n); break;
    case BinaryOp::Mul: k_mul<<<grid,kBlock>>>(pa, pb, pc, n); break;
    case BinaryOp::Div: k_div<<<grid,kBlock>>>(pa, pb, pc, n); break;
    }
    check_cuda(cudaDeviceSynchronize(), "cuda_binary");
    return Acc::make(Acc::shape(a), std::move(out_buf));
}

Tensor cuda_scalar(const Tensor& a, double scalar, ScalarOp op) {
    cudaSetDevice(Acc::device_id(a));
    size_t n = a.size();
    int grid = static_cast<int>((n + kBlock - 1) / kBlock);
    auto out_buf = Acc::make_buffer(n);
    int op_code = 0;
    switch (op) {
    case ScalarOp::Add:  op_code = 0; break;
    case ScalarOp::Sub:  op_code = 1; break;
    case ScalarOp::Mul:  op_code = 2; break;
    case ScalarOp::Div:  op_code = 3; break;
    case ScalarOp::RSub: op_code = 4; break;
    case ScalarOp::RDiv: op_code = 5; break;
    }
    k_scalar<<<grid, kBlock>>>(Acc::cuda_ptr(a), Acc::buffer_ptr(out_buf), n,
                               scalar, op_code);
    check_cuda(cudaDeviceSynchronize(), "cuda_scalar");
    return Acc::make(Acc::shape(a), std::move(out_buf));
}

Tensor cuda_softmax(const Tensor& a, size_t axis) {
    cudaSetDevice(Acc::device_id(a));
    const auto& shape = Acc::shape(a);
    if (axis >= shape.size())
        throw std::out_of_range("cuda_softmax: axis out of range");
    size_t outer = 1;
    for (size_t i = 0; i < axis; ++i) outer *= shape[i];
    size_t axis_dim = shape[axis];
    size_t inner = 1;
    for (size_t i = axis + 1; i < shape.size(); ++i) inner *= shape[i];

    auto out_buf = Acc::make_buffer(a.size());
    size_t work = outer * inner;
    int grid = static_cast<int>((work + kBlock - 1) / kBlock);
    k_softmax<<<grid, kBlock>>>(Acc::cuda_ptr(a), Acc::buffer_ptr(out_buf),
                                outer, axis_dim, inner);
    check_cuda(cudaDeviceSynchronize(), "cuda_softmax");
    return Acc::make(shape, std::move(out_buf));
}

namespace {

size_t out_dim(size_t input, size_t kernel, size_t stride, size_t padding,
               const char* name) {
    if (kernel == 0 || stride == 0)
        throw std::invalid_argument(std::string(name) + ": kernel and stride must be > 0");
    size_t padded = input + 2 * padding;
    if (padded < kernel)
        throw std::invalid_argument(std::string(name) + ": kernel larger than padded input");
    return (padded - kernel) / stride + 1;
}

} // anonymous namespace

Tensor cuda_conv2d(const Tensor& input, const Tensor& weight, const Tensor* bias,
                   size_t stride, size_t padding) {
    cudaSetDevice(Acc::device_id(input));
    const auto& xs = Acc::shape(input);
    const auto& ws = Acc::shape(weight);
    size_t N = xs[0], C = xs[1], H = xs[2], W = xs[3];
    size_t OC = ws[0], K = ws[2];
    size_t OH = out_dim(H, K, stride, padding, "cuda_conv2d");
    size_t OW = out_dim(W, K, stride, padding, "cuda_conv2d");
    auto out_buf = Acc::make_buffer(N * OC * OH * OW);
    size_t total = N * OC * OH * OW;
    int grid = static_cast<int>((total + kBlock - 1) / kBlock);
    k_conv2d_forward<<<grid, kBlock>>>(
        Acc::cuda_ptr(input), Acc::cuda_ptr(weight),
        bias ? Acc::cuda_ptr(*bias) : nullptr,
        Acc::buffer_ptr(out_buf),
        N, C, H, W, OC, K, OH, OW, stride, padding, bias != nullptr);
    check_cuda(cudaDeviceSynchronize(), "cuda_conv2d");
    return Acc::make({N, OC, OH, OW}, std::move(out_buf));
}

Tensor cuda_conv2d_backward_input(const Tensor& grad_out, const Tensor& weight,
                                  Tensor::Shape input_shape,
                                  size_t stride, size_t padding) {
    cudaSetDevice(Acc::device_id(grad_out));
    const auto& ws = Acc::shape(weight);
    const auto& gs = Acc::shape(grad_out);
    size_t N = input_shape[0], C = input_shape[1], H = input_shape[2], W = input_shape[3];
    size_t OC = ws[0], K = ws[2], OH = gs[2], OW = gs[3];
    auto out_buf = Acc::make_buffer(N * C * H * W);
    size_t total = N * C * H * W;
    int grid = static_cast<int>((total + kBlock - 1) / kBlock);
    k_conv2d_backward_input<<<grid, kBlock>>>(
        Acc::cuda_ptr(grad_out), Acc::cuda_ptr(weight), Acc::buffer_ptr(out_buf),
        N, C, H, W, OC, K, OH, OW, stride, padding);
    check_cuda(cudaDeviceSynchronize(), "cuda_conv2d_backward_input");
    return Acc::make(std::move(input_shape), std::move(out_buf));
}

Tensor cuda_conv2d_backward_weight(const Tensor& grad_out, const Tensor& input,
                                   Tensor::Shape weight_shape,
                                   size_t stride, size_t padding) {
    cudaSetDevice(Acc::device_id(grad_out));
    const auto& xs = Acc::shape(input);
    const auto& gs = Acc::shape(grad_out);
    size_t N = xs[0], C = xs[1], H = xs[2], W = xs[3];
    size_t OC = weight_shape[0], K = weight_shape[2], OH = gs[2], OW = gs[3];
    auto out_buf = Acc::make_buffer(OC * C * K * K);
    size_t total = OC * C * K * K;
    int grid = static_cast<int>((total + kBlock - 1) / kBlock);
    k_conv2d_backward_weight<<<grid, kBlock>>>(
        Acc::cuda_ptr(grad_out), Acc::cuda_ptr(input), Acc::buffer_ptr(out_buf),
        N, C, H, W, OC, K, OH, OW, stride, padding);
    check_cuda(cudaDeviceSynchronize(), "cuda_conv2d_backward_weight");
    return Acc::make(std::move(weight_shape), std::move(out_buf));
}

Tensor cuda_conv2d_backward_bias(const Tensor& grad_out) {
    cudaSetDevice(Acc::device_id(grad_out));
    const auto& gs = Acc::shape(grad_out);
    size_t N = gs[0], OC = gs[1], OH = gs[2], OW = gs[3];
    auto out_buf = Acc::make_buffer(OC);
    int grid = static_cast<int>((OC + kBlock - 1) / kBlock);
    k_conv2d_backward_bias<<<grid, kBlock>>>(Acc::cuda_ptr(grad_out),
                                             Acc::buffer_ptr(out_buf),
                                             N, OC, OH, OW);
    check_cuda(cudaDeviceSynchronize(), "cuda_conv2d_backward_bias");
    return Acc::make({OC}, std::move(out_buf));
}

Tensor cuda_max_pool2d(const Tensor& input, size_t kernel_size,
                       size_t stride, size_t padding) {
    cudaSetDevice(Acc::device_id(input));
    const auto& xs = Acc::shape(input);
    size_t N = xs[0], C = xs[1], H = xs[2], W = xs[3];
    size_t OH = out_dim(H, kernel_size, stride, padding, "cuda_max_pool2d");
    size_t OW = out_dim(W, kernel_size, stride, padding, "cuda_max_pool2d");
    auto out_buf = Acc::make_buffer(N * C * OH * OW);
    size_t total = N * C * OH * OW;
    int grid = static_cast<int>((total + kBlock - 1) / kBlock);
    k_max_pool2d<<<grid, kBlock>>>(Acc::cuda_ptr(input), Acc::buffer_ptr(out_buf),
                                   N, C, H, W, kernel_size, OH, OW, stride, padding);
    check_cuda(cudaDeviceSynchronize(), "cuda_max_pool2d");
    return Acc::make({N, C, OH, OW}, std::move(out_buf));
}

Tensor cuda_max_pool2d_backward(const Tensor& grad_out, const Tensor& input,
                                size_t kernel_size, size_t stride,
                                size_t padding) {
    cudaSetDevice(Acc::device_id(grad_out));
    const auto& xs = Acc::shape(input);
    const auto& gs = Acc::shape(grad_out);
    size_t N = xs[0], C = xs[1], H = xs[2], W = xs[3], OH = gs[2], OW = gs[3];
    auto out_buf = Acc::make_buffer(N * C * H * W);
    check_cuda(cudaMemset(Acc::buffer_ptr(out_buf), 0, N * C * H * W * sizeof(double)),
               "cuda_max_pool2d_backward memset");
    size_t total = N * C * OH * OW;
    int grid = static_cast<int>((total + kBlock - 1) / kBlock);
    k_max_pool2d_backward<<<grid, kBlock>>>(Acc::cuda_ptr(grad_out),
                                            Acc::cuda_ptr(input),
                                            Acc::buffer_ptr(out_buf),
                                            N, C, H, W, kernel_size, OH, OW,
                                            stride, padding);
    check_cuda(cudaDeviceSynchronize(), "cuda_max_pool2d_backward");
    return Acc::make(xs, std::move(out_buf));
}

Tensor cuda_avg_pool2d(const Tensor& input, size_t kernel_size,
                       size_t stride, size_t padding) {
    cudaSetDevice(Acc::device_id(input));
    const auto& xs = Acc::shape(input);
    size_t N = xs[0], C = xs[1], H = xs[2], W = xs[3];
    size_t OH = out_dim(H, kernel_size, stride, padding, "cuda_avg_pool2d");
    size_t OW = out_dim(W, kernel_size, stride, padding, "cuda_avg_pool2d");
    auto out_buf = Acc::make_buffer(N * C * OH * OW);
    size_t total = N * C * OH * OW;
    int grid = static_cast<int>((total + kBlock - 1) / kBlock);
    k_avg_pool2d<<<grid, kBlock>>>(Acc::cuda_ptr(input), Acc::buffer_ptr(out_buf),
                                   N, C, H, W, kernel_size, OH, OW, stride, padding);
    check_cuda(cudaDeviceSynchronize(), "cuda_avg_pool2d");
    return Acc::make({N, C, OH, OW}, std::move(out_buf));
}

Tensor cuda_avg_pool2d_backward(const Tensor& grad_out,
                                Tensor::Shape input_shape,
                                size_t kernel_size, size_t stride,
                                size_t padding) {
    cudaSetDevice(Acc::device_id(grad_out));
    const auto& gs = Acc::shape(grad_out);
    size_t N = input_shape[0], C = input_shape[1], H = input_shape[2], W = input_shape[3];
    size_t OH = gs[2], OW = gs[3];
    auto out_buf = Acc::make_buffer(N * C * H * W);
    size_t total = N * C * H * W;
    int grid = static_cast<int>((total + kBlock - 1) / kBlock);
    k_avg_pool2d_backward<<<grid, kBlock>>>(Acc::cuda_ptr(grad_out),
                                            Acc::buffer_ptr(out_buf),
                                            N, C, H, W, kernel_size, OH, OW,
                                            stride, padding);
    check_cuda(cudaDeviceSynchronize(), "cuda_avg_pool2d_backward");
    return Acc::make(std::move(input_shape), std::move(out_buf));
}

// ── cuda_matmul ───────────────────────────────────────────────────────────────
// Row-major A (m×k) * B (k×n) = C (m×n).
// cuBLAS is column-major, so we compute Cᵀ = Bᵀ * Aᵀ:
//   treat row-major A as column-major Aᵀ (k×m), lda = k
//   treat row-major B as column-major Bᵀ (n×k), ldb = n
//   result is column-major Cᵀ (n×m) = row-major C (m×n), ldc = n

Tensor cuda_matmul(const Tensor& a, const Tensor& b) {
    cudaSetDevice(Acc::device_id(a));
    size_t m = Acc::shape(a)[0];
    size_t k = Acc::shape(a)[1];
    size_t n = Acc::shape(b)[1];

    auto out_buf = Acc::make_buffer(m * n);

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
        Acc::buffer_ptr(out_buf), static_cast<int>(n) // C^T col-major, ldc = n
    );
    if (st != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuda_matmul: cublasDgemm failed");

    check_cuda(cudaDeviceSynchronize(), "cuda_matmul");
    return Acc::make({m, n}, std::move(out_buf));
}

} // namespace SharedMath::LinearAlgebra::detail
