// DynamicMatrixCUDA.cu — CUDA backend for DynamicMatrix.
// Compiled only when -DSHAREDMATH_ENABLE_CUDA=ON.
//
// Accelerated operations
//   matmul  → cuBLAS dgemm  (row-major trick: C = A·B  ↔  Cᵀ = Bᵀ·Aᵀ)
//   add/sub → custom element-wise kernels
//   scale   → custom element-wise kernel (multiply every element by a scalar)

#include "DynamicMatrixCUDA.h"   // internal header; includes DynamicMatrix.h

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <memory>

namespace SharedMath::LinearAlgebra {

// ─── CUDABuffer ───────────────────────────────────────────────────────────────
// Defined here (the only translation unit that sees CUDA headers).
// DynamicMatrix.h forward-declares the struct so shared_ptr<CUDABuffer> compiles
// with an incomplete type; the real destructor lives here where cudaFree is
// visible, so the shared_ptr control block calls the right cleanup code.

struct DynamicMatrix::CUDABuffer {
    double* ptr   = nullptr;
    size_t  count = 0;          // number of double elements

    // Allocate and copy from host (pass nullptr to skip the copy).
    explicit CUDABuffer(const double* host_src, size_t n) : count(n) {
        cudaError_t err = cudaMalloc(&ptr, n * sizeof(double));
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("DM CUDABuffer: cudaMalloc failed: ") +
                cudaGetErrorString(err));
        if (host_src && n > 0) {
            err = cudaMemcpy(ptr, host_src, n * sizeof(double),
                             cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
                throw std::runtime_error(
                    std::string("DM CUDABuffer: cudaMemcpy H→D failed: ") +
                    cudaGetErrorString(err));
        }
    }

    // Allocate output buffer without copying.
    explicit CUDABuffer(size_t n) : CUDABuffer(nullptr, n) {}

    ~CUDABuffer() { if (ptr) cudaFree(ptr); }

    // Non-copyable — ownership is managed by shared_ptr.
    CUDABuffer(const CUDABuffer&)            = delete;
    CUDABuffer& operator=(const CUDABuffer&) = delete;

    void to_host(double* dst) const {
        cudaError_t err = cudaMemcpy(dst, ptr, count * sizeof(double),
                                     cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("DM CUDABuffer: cudaMemcpy D→H failed: ") +
                cudaGetErrorString(err));
    }
};

// ─── DynamicMatrixCUDAImpl ────────────────────────────────────────────────────
// Defined here (CUDABuffer is complete at this point).
// Declared friend of DynamicMatrix in DynamicMatrix.h.

namespace detail {

struct DynamicMatrixCUDAImpl {
    static double* cuda_ptr(const DynamicMatrix& m) {
        return m.m_cuda_buf->ptr;
    }
    static size_t nrows(const DynamicMatrix& m) { return m.rows_; }
    static size_t ncols(const DynamicMatrix& m) { return m.cols_; }

    static DynamicMatrix make(size_t rows, size_t cols,
                              std::shared_ptr<DynamicMatrix::CUDABuffer> buf) {
        return DynamicMatrix::from_cuda(rows, cols, std::move(buf));
    }
};

} // namespace detail

// ─── Private factory ──────────────────────────────────────────────────────────
// Wraps a GPU buffer into a DynamicMatrix without any host allocation.

DynamicMatrix DynamicMatrix::from_cuda(size_t rows, size_t cols,
                                        std::shared_ptr<CUDABuffer> buf) {
    DynamicMatrix m;
    m.rows_      = rows;
    m.cols_      = cols;
    m.m_cuda_buf = std::move(buf);
    m.m_device   = Device::CUDA;
    // m.data_ stays empty — the GPU buffer is authoritative
    return m;
}

// ─── Runtime GPU availability ─────────────────────────────────────────────────

namespace {
bool have_gpu() noexcept {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}
} // anonymous

// ─── Transfer: .cuda() / .cpu() ──────────────────────────────────────────────

DynamicMatrix DynamicMatrix::cuda() const {
    if (m_device == Device::CUDA) return *this;   // already on GPU

    // Graceful fallback: if no CUDA-capable device is present, stay on CPU.
    if (!have_gpu()) return *this;

    auto buf = std::make_shared<CUDABuffer>(data_.data(), data_.size());
    return from_cuda(rows_, cols_, std::move(buf));
}

DynamicMatrix DynamicMatrix::cpu() const {
    if (m_device == Device::CPU) return *this;    // already on CPU

    size_t n = rows_ * cols_;
    std::vector<double> host(n);
    m_cuda_buf->to_host(host.data());
    return DynamicMatrix(rows_, cols_, std::move(host));
}

} // namespace SharedMath::LinearAlgebra

// ─── cuBLAS handle (thread-local singleton) ───────────────────────────────────
// Separate from TensorCUDA.cu's handle — each translation unit owns its own so
// there are no cross-TU ODR issues.  Two handles per thread is negligible.

namespace {

cublasHandle_t& dm_cublas_handle() {
    thread_local cublasHandle_t handle = [] {
        cublasHandle_t h;
        if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error(
                "DynamicMatrix CUDA: cublasCreate failed — no GPU available?");
        return h;
    }();
    return handle;
}

inline void dm_check(cudaError_t err, const char* where) {
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string(where) + ": " + cudaGetErrorString(err));
}

} // anonymous

// ─── Element-wise CUDA kernels ────────────────────────────────────────────────

namespace {

constexpr int kDMBlock = 256;

__global__ void k_dm_add(const double* __restrict__ a,
                          const double* __restrict__ b,
                          double*       __restrict__ c, size_t n)
{
    size_t i = blockIdx.x * kDMBlock + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void k_dm_sub(const double* __restrict__ a,
                          const double* __restrict__ b,
                          double*       __restrict__ c, size_t n)
{
    size_t i = blockIdx.x * kDMBlock + threadIdx.x;
    if (i < n) c[i] = a[i] - b[i];
}

__global__ void k_dm_scale(const double* __restrict__ a,
                            double*       __restrict__ c, size_t n, double s)
{
    size_t i = blockIdx.x * kDMBlock + threadIdx.x;
    if (i < n) c[i] = a[i] * s;
}

} // anonymous

// ─── Dispatch implementations ─────────────────────────────────────────────────

namespace SharedMath::LinearAlgebra::detail {

using Acc = DynamicMatrixCUDAImpl;   // shorthand

// ── dm_cuda_add ───────────────────────────────────────────────────────────────

DynamicMatrix dm_cuda_add(const DynamicMatrix& A, const DynamicMatrix& B) {
    size_t n    = A.size();
    int    grid = static_cast<int>((n + kDMBlock - 1) / kDMBlock);
    auto   out  = std::make_shared<DynamicMatrix::CUDABuffer>(n);

    k_dm_add<<<grid, kDMBlock>>>(Acc::cuda_ptr(A), Acc::cuda_ptr(B),
                                  out->ptr, n);
    dm_check(cudaDeviceSynchronize(), "dm_cuda_add");
    return Acc::make(Acc::nrows(A), Acc::ncols(A), std::move(out));
}

// ── dm_cuda_sub ───────────────────────────────────────────────────────────────

DynamicMatrix dm_cuda_sub(const DynamicMatrix& A, const DynamicMatrix& B) {
    size_t n    = A.size();
    int    grid = static_cast<int>((n + kDMBlock - 1) / kDMBlock);
    auto   out  = std::make_shared<DynamicMatrix::CUDABuffer>(n);

    k_dm_sub<<<grid, kDMBlock>>>(Acc::cuda_ptr(A), Acc::cuda_ptr(B),
                                  out->ptr, n);
    dm_check(cudaDeviceSynchronize(), "dm_cuda_sub");
    return Acc::make(Acc::nrows(A), Acc::ncols(A), std::move(out));
}

// ── dm_cuda_scale ─────────────────────────────────────────────────────────────

DynamicMatrix dm_cuda_scale(const DynamicMatrix& A, double scalar) {
    size_t n    = A.size();
    int    grid = static_cast<int>((n + kDMBlock - 1) / kDMBlock);
    auto   out  = std::make_shared<DynamicMatrix::CUDABuffer>(n);

    k_dm_scale<<<grid, kDMBlock>>>(Acc::cuda_ptr(A), out->ptr, n, scalar);
    dm_check(cudaDeviceSynchronize(), "dm_cuda_scale");
    return Acc::make(Acc::nrows(A), Acc::ncols(A), std::move(out));
}

// ── dm_cuda_matmul ────────────────────────────────────────────────────────────
// Row-major A (m×k) * B (k×n) = C (m×n).
//
// cuBLAS is column-major, so we exploit the identity:
//     C  (row-major, m×n) = A · B
//  ↔  Cᵀ (col-major, n×m) = Bᵀ · Aᵀ
//
// Treating row-major A as column-major Aᵀ:  lda = k  (number of cols of A)
// Treating row-major B as column-major Bᵀ:  ldb = n  (number of cols of B)
// Result Cᵀ stored column-major with ldc = n, which IS the row-major C.

DynamicMatrix dm_cuda_matmul(const DynamicMatrix& A, const DynamicMatrix& B) {
    const size_t m = Acc::nrows(A);
    const size_t k = Acc::ncols(A);
    const size_t n = Acc::ncols(B);

    auto out = std::make_shared<DynamicMatrix::CUDABuffer>(m * n);

    const double alpha = 1.0, beta = 0.0;
    cublasStatus_t st = cublasDgemm(
        dm_cublas_handle(),
        CUBLAS_OP_N, CUBLAS_OP_N,
        static_cast<int>(n),        // rows    of op(Bᵀ) = cols of result
        static_cast<int>(m),        // columns of op(Aᵀ) = rows of result
        static_cast<int>(k),        // inner dimension
        &alpha,
        Acc::cuda_ptr(B), static_cast<int>(n),  // B row-major → Bᵀ col-major, ldb=n
        Acc::cuda_ptr(A), static_cast<int>(k),  // A row-major → Aᵀ col-major, lda=k
        &beta,
        out->ptr,         static_cast<int>(n)   // Cᵀ col-major = C row-major, ldc=n
    );
    if (st != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("dm_cuda_matmul: cublasDgemm failed");

    dm_check(cudaDeviceSynchronize(), "dm_cuda_matmul");
    return Acc::make(m, n, std::move(out));
}

} // namespace SharedMath::LinearAlgebra::detail
