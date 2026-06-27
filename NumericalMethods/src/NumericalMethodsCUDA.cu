// NumericalMethodsCUDA.cu — CUDA backend for GPUNumVec / GPUNumMat.
// Compiled only when -DSHAREDMATH_ENABLE_CUDA=ON.
//
// Accelerated operations
//   BLAS-1  → cuBLAS (axpy, dot, nrm2, scal)
//   gemv    → cuBLAS cublasDgemv
//   ger     → cuBLAS cublasDger (rank-1 update)
//   solve   → cuSOLVER LU (cusolverDnDgetrf + cusolverDnDgetrs)

#ifdef SHAREDMATH_CUDA

#include "NumericalMethods/NumericalMethodsGPU.h"

#include <cublas_v2.h>
#include <cusolver_common.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <memory>
#include <algorithm>

namespace SharedMath::NumericalMethods {

// ── CUDABuffer definitions (RAII, same pattern as DynamicMatrixCUDA.cu) ────

struct GPUNumVec::CUDABuffer {
    double* ptr   = nullptr;
    size_t  count = 0;

    explicit CUDABuffer(const double* host_src, size_t n) : count(n) {
        // Reset any prior GPU error
        cudaGetLastError();

        cudaError_t err = cudaMalloc(&ptr, n * sizeof(double));
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("GPUNumVec CUDABuffer: cudaMalloc failed: ") +
                cudaGetErrorString(err));
        if (host_src && n > 0) {
            err = cudaMemcpy(ptr, host_src, n * sizeof(double),
                             cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
                throw std::runtime_error(
                    std::string("GPUNumVec CUDABuffer: cudaMemcpy H→D failed: ") +
                    cudaGetErrorString(err));
        }
    }

    explicit CUDABuffer(size_t n) : CUDABuffer(nullptr, n) {}

    ~CUDABuffer() { if (ptr) cudaFree(ptr); }

    CUDABuffer(const CUDABuffer&)            = delete;
    CUDABuffer& operator=(const CUDABuffer&) = delete;

    void to_host(double* dst) const {
        cudaError_t err = cudaMemcpy(dst, ptr, count * sizeof(double),
                                     cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("GPUNumVec CUDABuffer: cudaMemcpy D→H failed: ") +
                cudaGetErrorString(err));
    }
};

struct GPUNumMat::CUDABuffer {
    double* ptr   = nullptr;
    size_t  count = 0;

    explicit CUDABuffer(const double* host_src, size_t n) : count(n) {
        // Reset any prior GPU error
        cudaGetLastError();

        cudaError_t err = cudaMalloc(&ptr, n * sizeof(double));
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("GPUNumMat CUDABuffer: cudaMalloc failed: ") +
                cudaGetErrorString(err));
        if (host_src && n > 0) {
            err = cudaMemcpy(ptr, host_src, n * sizeof(double),
                             cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
                throw std::runtime_error(
                    std::string("GPUNumMat CUDABuffer: cudaMemcpy H→D failed: ") +
                    cudaGetErrorString(err));
        }
    }

    explicit CUDABuffer(size_t n) : CUDABuffer(nullptr, n) {}

    ~CUDABuffer() { if (ptr) cudaFree(ptr); }

    CUDABuffer(const CUDABuffer&)            = delete;
    CUDABuffer& operator=(const CUDABuffer&) = delete;

    void to_host(double* dst) const {
        cudaError_t err = cudaMemcpy(dst, ptr, count * sizeof(double),
                                     cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("GPUNumMat CUDABuffer: cudaMemcpy D→H failed: ") +
                cudaGetErrorString(err));
    }
};

// ── cuBLAS / cuSOLVER handles (thread-local singletons) ────────────────────

namespace {
cublasHandle_t& nm_cublas_handle() {
    thread_local cublasHandle_t handle = [] {
        cublasHandle_t h;
        if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error(
                "NumericalMethods CUDA: cublasCreate failed — no GPU available?");
        return h;
    }();
    return handle;
}

cusolverDnHandle_t& nm_cusolver_handle() {
    thread_local cusolverDnHandle_t handle = [] {
        cusolverDnHandle_t h;
        if (cusolverDnCreate(&h) != CUSOLVER_STATUS_SUCCESS)
            throw std::runtime_error(
                "NumericalMethods CUDA: cusolverDnCreate failed");
        return h;
    }();
    return handle;
}

inline void nm_check(cudaError_t err, const char* where) {
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string(where) + ": " + cudaGetErrorString(err));
}

constexpr int kNMBlock = 256;

bool have_gpu() noexcept {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}
} // anonymous namespace

// ── GPUNumVec implementation ───────────────────────────────────────────────

GPUNumVec::GPUNumVec(const std::vector<double>& host)
    : m_host(host), m_size(host.size()), m_device(Device::CPU) {}

GPUNumVec::GPUNumVec(size_t n)
    : m_host(n, 0.0), m_size(n), m_device(Device::CPU) {}

GPUNumVec::~GPUNumVec() = default;
GPUNumVec::GPUNumVec(const GPUNumVec&) = default;
GPUNumVec& GPUNumVec::operator=(const GPUNumVec&) = default;
GPUNumVec::GPUNumVec(GPUNumVec&&) noexcept = default;
GPUNumVec& GPUNumVec::operator=(GPUNumVec&&) noexcept = default;

GPUNumVec GPUNumVec::toGPU() const {
    if (m_device == Device::CUDA || m_size == 0) return *this;
    if (!have_gpu()) return *this;

    GPUNumVec result = *this;
    result.m_buf = std::make_shared<CUDABuffer>(m_host.data(), m_size);
    result.m_device = Device::CUDA;
    return result;
}

GPUNumVec GPUNumVec::toCPU() const {
    if (m_device == Device::CPU) return *this;

    GPUNumVec result;
    result.m_host.resize(m_size);
    m_buf->to_host(result.m_host.data());
    result.m_size = m_size;
    result.m_device = Device::CPU;
    return result;
}

GPUNumVec GPUNumVec::from_device(const double* d_ptr, size_t n) {
    GPUNumVec v(n);
    if (d_ptr && n > 0) {
        v.m_buf = std::make_shared<CUDABuffer>(n);
        nm_check(cudaMemcpy(v.m_buf->ptr, d_ptr, n * sizeof(double),
                            cudaMemcpyDeviceToDevice), "from_device");
        v.m_device = Device::CUDA;
    }
    return v;
}

Device GPUNumVec::device() const noexcept { return m_device; }
size_t GPUNumVec::size() const noexcept { return m_size; }
bool   GPUNumVec::empty() const noexcept { return m_size == 0; }

const std::vector<double>& GPUNumVec::host() const {
    requireCPU("host");
    return m_host;
}

std::vector<double>& GPUNumVec::host() {
    requireCPU("host");
    return m_host;
}

double* GPUNumVec::devicePtr() {
    requireGPU("devicePtr");
    return m_buf->ptr;
}

const double* GPUNumVec::devicePtr() const {
    requireGPU("devicePtr");
    return m_buf->ptr;
}

void GPUNumVec::requireCPU(const char* op) const {
    if (m_device != Device::CPU)
        throw std::runtime_error(
            std::string("GPUNumVec::") + op +
            ": vector is on GPU — call .toCPU() first");
}

void GPUNumVec::requireGPU(const char* op) const {
    if (m_device != Device::CUDA)
        throw std::runtime_error(
            std::string("GPUNumVec::") + op +
            ": vector is on CPU — call .toGPU() first");
}

// ── BLAS-1 GPU kernels ────────────────────────────────────────────────────

namespace {
__global__ void k_add(const double* a, const double* b, double* c, size_t n) {
    size_t i = blockIdx.x * kNMBlock + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void k_scale(double s, const double* x, double* r, size_t n) {
    size_t i = blockIdx.x * kNMBlock + threadIdx.x;
    if (i < n) r[i] = s * x[i];
}
} // anonymous

GPUNumVec GPUNumVec::axpy(double a, const GPUNumVec& x, const GPUNumVec& y) {
    if (x.size() != y.size())
        throw std::invalid_argument("GPUNumVec::axpy: size mismatch");

    if (x.device() == Device::CUDA && y.device() == Device::CUDA) {
        // r = a*x + y
        GPUNumVec result(x.size());
        result.m_buf = std::make_shared<CUDABuffer>(x.size());
        result.m_device = Device::CUDA;

        // First copy y to result
        cublasStatus_t st = cublasDcopy(nm_cublas_handle(),
            static_cast<int>(y.size()),
            y.devicePtr(), 1,
            result.devicePtr(), 1);
        if (st != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("GPUNumVec::axpy: cublasDcopy failed");

        // Then result += a*x
        st = cublasDaxpy(nm_cublas_handle(),
            static_cast<int>(x.size()), &a,
            x.devicePtr(), 1, result.devicePtr(), 1);
        if (st != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("GPUNumVec::axpy: cublasDaxpy failed");

        return result;
    }

    // CPU fallback
    size_t n = x.size();
    std::vector<double> r(n);
    const auto& xh = x.host();
    const auto& yh = y.host();
    for (size_t i = 0; i < n; ++i)
        r[i] = a * xh[i] + yh[i];
    return GPUNumVec(r);
}

GPUNumVec GPUNumVec::add(const GPUNumVec& a, const GPUNumVec& b) {
    if (a.size() != b.size())
        throw std::invalid_argument("GPUNumVec::add: size mismatch");

    if (a.device() == Device::CUDA && b.device() == Device::CUDA) {
        GPUNumVec result(a.size());
        result.m_buf = std::make_shared<CUDABuffer>(a.size());
        result.m_device = Device::CUDA;

        int grid = static_cast<int>((a.size() + kNMBlock - 1) / kNMBlock);
        k_add<<<grid, kNMBlock>>>(a.devicePtr(), b.devicePtr(),
                                   result.devicePtr(), a.size());
        nm_check(cudaDeviceSynchronize(), "GPUNumVec::add");
        return result;
    }

    size_t n = a.size();
    std::vector<double> r(n);
    const auto& ah = a.host();
    const auto& bh = b.host();
    for (size_t i = 0; i < n; ++i)
        r[i] = ah[i] + bh[i];
    return GPUNumVec(r);
}

GPUNumVec GPUNumVec::scale(double s, const GPUNumVec& x) {
    if (x.device() == Device::CUDA) {
        GPUNumVec result(x.size());
        result.m_buf = std::make_shared<CUDABuffer>(x.size());
        result.m_device = Device::CUDA;

        int grid = static_cast<int>((x.size() + kNMBlock - 1) / kNMBlock);
        k_scale<<<grid, kNMBlock>>>(s, x.devicePtr(), result.devicePtr(), x.size());
        nm_check(cudaDeviceSynchronize(), "GPUNumVec::scale");
        return result;
    }

    size_t n = x.size();
    std::vector<double> r(n);
    const auto& xh = x.host();
    for (size_t i = 0; i < n; ++i)
        r[i] = s * xh[i];
    return GPUNumVec(r);
}

double GPUNumVec::dot(const GPUNumVec& a, const GPUNumVec& b) {
    if (a.size() != b.size())
        throw std::invalid_argument("GPUNumVec::dot: size mismatch");

    if (a.device() == Device::CUDA && b.device() == Device::CUDA) {
        double result = 0.0;
        cublasStatus_t st = cublasDdot(
            nm_cublas_handle(),
            static_cast<int>(a.size()),
            a.devicePtr(), 1,
            b.devicePtr(), 1,
            &result);
        if (st != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("GPUNumVec::dot: cublasDdot failed");
        return result;
    }

    size_t n = a.size();
    const auto& ah = a.host();
    const auto& bh = b.host();
    double s = 0.0;
    for (size_t i = 0; i < n; ++i)
        s += ah[i] * bh[i];
    return s;
}

double GPUNumVec::nrm2(const GPUNumVec& x) {
    if (x.device() == Device::CUDA) {
        double result = 0.0;
        cublasStatus_t st = cublasDnrm2(
            nm_cublas_handle(),
            static_cast<int>(x.size()),
            x.devicePtr(), 1,
            &result);
        if (st != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("GPUNumVec::nrm2: cublasDnrm2 failed");
        return result;
    }

    size_t n = x.size();
    const auto& xh = x.host();
    double s = 0.0;
    for (size_t i = 0; i < n; ++i)
        s += xh[i] * xh[i];
    return std::sqrt(s);
}

// ── GPUNumMat implementation ───────────────────────────────────────────────

GPUNumMat::GPUNumMat(size_t rows, size_t cols, const std::vector<double>& host)
    : m_host(host), m_rows(rows), m_cols(cols), m_device(Device::CPU)
{
    if (host.size() != rows * cols)
        throw std::invalid_argument("GPUNumMat: data size mismatch");
}

GPUNumMat::GPUNumMat(size_t rows, size_t cols)
    : m_host(rows * cols, 0.0), m_rows(rows), m_cols(cols), m_device(Device::CPU) {}

GPUNumMat::~GPUNumMat() = default;
GPUNumMat::GPUNumMat(const GPUNumMat&) = default;
GPUNumMat& GPUNumMat::operator=(const GPUNumMat&) = default;
GPUNumMat::GPUNumMat(GPUNumMat&&) noexcept = default;
GPUNumMat& GPUNumMat::operator=(GPUNumMat&&) noexcept = default;

GPUNumMat GPUNumMat::toGPU() const {
    if (m_device == Device::CUDA || size() == 0) return *this;
    if (!have_gpu()) return *this;

    GPUNumMat result = *this;
    result.m_buf = std::make_shared<CUDABuffer>(m_host.data(), m_rows * m_cols);
    result.m_device = Device::CUDA;
    return result;
}

GPUNumMat GPUNumMat::toCPU() const {
    if (m_device == Device::CPU) return *this;

    GPUNumMat result;
    result.m_host.resize(m_rows * m_cols);
    m_buf->to_host(result.m_host.data());
    result.m_rows = m_rows;
    result.m_cols = m_cols;
    result.m_device = Device::CPU;
    return result;
}

Device GPUNumMat::device() const noexcept { return m_device; }
size_t GPUNumMat::rows() const noexcept { return m_rows; }
size_t GPUNumMat::cols() const noexcept { return m_cols; }
size_t GPUNumMat::size() const noexcept { return m_rows * m_cols; }

const std::vector<double>& GPUNumMat::host() const {
    requireCPU("host");
    return m_host;
}

std::vector<double>& GPUNumMat::host() {
    requireCPU("host");
    return m_host;
}

double* GPUNumMat::devicePtr() {
    requireGPU("devicePtr");
    return m_buf->ptr;
}

const double* GPUNumMat::devicePtr() const {
    requireGPU("devicePtr");
    return m_buf->ptr;
}

void GPUNumMat::requireCPU(const char* op) const {
    if (m_device != Device::CPU)
        throw std::runtime_error(
            std::string("GPUNumMat::") + op +
            ": matrix is on GPU — call .toCPU() first");
}

void GPUNumMat::requireGPU(const char* op) const {
    if (m_device != Device::CUDA)
        throw std::runtime_error(
            std::string("GPUNumMat::") + op +
            ": matrix is on CPU — call .toGPU() first");
}

// ── Matrix-vector product (cuBLAS gemv) ────────────────────────────────────

GPUNumVec GPUNumMat::gemv(const GPUNumVec& x) const {
    if (x.size() != m_cols)
        throw std::invalid_argument("GPUNumMat::gemv: dimension mismatch");

    if (m_device == Device::CUDA && x.device() == Device::CUDA) {
        GPUNumVec result(m_rows);
        result.m_buf = std::make_shared<GPUNumVec::CUDABuffer>(m_rows);
        result.m_device = Device::CUDA;

        // y = A*x, A row-major m×n
        // cuBLAS column-major: y = op(A)*x where op(A) = A^T (row-major ↔ col-major)
        const double alpha = 1.0, beta = 0.0;
        cublasStatus_t st = cublasDgemv(
            nm_cublas_handle(),
            CUBLAS_OP_T,                         // transpose because row-major
            static_cast<int>(m_cols),             // rows of op(A) = cols of A
            static_cast<int>(m_rows),             // cols of op(A) = rows of A
            &alpha,
            m_buf->ptr, static_cast<int>(m_cols), // A row-major, lda = cols
            x.devicePtr(), 1,
            &beta,
            result.devicePtr(), 1);
        if (st != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("GPUNumMat::gemv: cublasDgemv failed");

        nm_check(cudaDeviceSynchronize(), "GPUNumMat::gemv");
        return result;
    }

    // CPU fallback
    std::vector<double> result(m_rows, 0.0);
    const auto& xh = x.host();
    for (size_t i = 0; i < m_rows; ++i)
        for (size_t j = 0; j < m_cols; ++j)
            result[i] += m_host[i * m_cols + j] * xh[j];
    return GPUNumVec(result);
}

// ── Rank-1 update: A += alpha * u * v^T (cuBLAS ger) ──────────────────────

void GPUNumMat::ger(double alpha, const GPUNumVec& u, const GPUNumVec& v) {
    if (u.size() != m_rows || v.size() != m_cols)
        throw std::invalid_argument("GPUNumMat::ger: dimension mismatch");

    if (m_device == Device::CUDA && u.device() == Device::CUDA && v.device() == Device::CUDA) {
        // cublasDger on column-major: A_cm += alpha*x*y^T
        // For row-major A_rm: swap roles → A_rm += alpha*u*v^T ↔ A_cm += alpha*v*u^T
        cublasStatus_t st = cublasDger(
            nm_cublas_handle(),
            static_cast<int>(m_cols),   // rows of A_cm = cols of A_rm
            static_cast<int>(m_rows),   // cols of A_cm = rows of A_rm
            &alpha,
            v.devicePtr(), 1,           // x = v
            u.devicePtr(), 1,           // y = u
            m_buf->ptr, static_cast<int>(m_cols));  // A_rm, lda = cols
        if (st != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("GPUNumMat::ger: cublasDger failed");

        nm_check(cudaDeviceSynchronize(), "GPUNumMat::ger");
        return;
    }

    // CPU fallback
    const auto& uh = u.host();
    const auto& vh = v.host();
    for (size_t i = 0; i < m_rows; ++i)
        for (size_t j = 0; j < m_cols; ++j)
            m_host[i * m_cols + j] += alpha * uh[i] * vh[j];
}

// ── Solve Ax = b via LU factorization (cuSOLVER) ──────────────────────────

GPUNumVec GPUNumMat::solve(const GPUNumVec& b) const {
    if (m_rows != m_cols)
        throw std::invalid_argument("GPUNumMat::solve: matrix must be square");
    if (b.size() != m_rows)
        throw std::invalid_argument("GPUNumMat::solve: dimension mismatch");

    size_t n = m_rows;

    if (m_device == Device::CUDA && b.device() == Device::CUDA) {
        // cuSOLVER operates in-place, so copy A to workspace
        double* d_A = nullptr;
        nm_check(cudaMalloc(&d_A, n * n * sizeof(double)), "solve: cudaMalloc A");
        nm_check(cudaMemcpy(d_A, m_buf->ptr, n * n * sizeof(double),
                            cudaMemcpyDeviceToDevice), "solve: copy A to workspace");

        int* d_ipiv = nullptr;
        nm_check(cudaMalloc(&d_ipiv, n * sizeof(int)), "solve: cudaMalloc ipiv");

        int* d_info = nullptr;
        nm_check(cudaMalloc(&d_info, sizeof(int)), "solve: cudaMalloc info");

        int lwork = 0;
        cusolverDnDgetrf_bufferSize(nm_cusolver_handle(),
            n, n, d_A, n, &lwork);

        double* d_work = nullptr;
        nm_check(cudaMalloc(&d_work, lwork * sizeof(double)), "solve: cudaMalloc work");

        cusolverStatus_t cs = cusolverDnDgetrf(
            nm_cusolver_handle(), n, n, d_A, n, d_work, d_ipiv, d_info);
        if (cs != CUSOLVER_STATUS_SUCCESS)
            throw std::runtime_error("GPUNumMat::solve: cusolverDnDgetrf failed");

        // Copy b to device
        double* d_b = nullptr;
        nm_check(cudaMalloc(&d_b, n * sizeof(double)), "solve: cudaMalloc b");
        nm_check(cudaMemcpy(d_b, b.devicePtr(), n * sizeof(double),
                            cudaMemcpyDeviceToDevice), "solve: copy b to workspace");

        int info = 0;
        cs = cusolverDnDgetrs(
            nm_cusolver_handle(), CUBLAS_OP_T, n, 1,
            d_A, n, d_ipiv, d_b, n, d_info);
        if (cs != CUSOLVER_STATUS_SUCCESS)
            throw std::runtime_error("GPUNumMat::solve: cusolverDnDgetrs failed");

        nm_check(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost),
                 "solve: copy info");

        // Copy result
        GPUNumVec result(n);
        result.m_buf = std::make_shared<GPUNumVec::CUDABuffer>(n);
        result.m_device = Device::CUDA;
        nm_check(cudaMemcpy(result.devicePtr(), d_b, n * sizeof(double),
                            cudaMemcpyDeviceToDevice), "solve: memcpy result");
        nm_check(cudaDeviceSynchronize(), "GPUNumMat::solve");

        cudaFree(d_A);
        cudaFree(d_ipiv);
        cudaFree(d_info);
        cudaFree(d_work);
        cudaFree(d_b);

        return result;
    }

    // CPU fallback — Gaussian elimination with partial pivoting
    std::vector<std::vector<double>> A(n, std::vector<double>(n));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            A[i][j] = m_host[i * n + j];

    std::vector<double> x = b.host();

    for (size_t k = 0; k < n; ++k) {
        size_t pivot = k;
        double max_val = std::abs(A[k][k]);
        for (size_t i = k + 1; i < n; ++i) {
            if (std::abs(A[i][k]) > max_val) {
                max_val = std::abs(A[i][k]);
                pivot = i;
            }
        }
        if (max_val < 1e-15)
            throw std::runtime_error("GPUNumMat::solve: singular matrix");

        if (pivot != k) {
            std::swap(A[k], A[pivot]);
            std::swap(x[k], x[pivot]);
        }

        for (size_t i = k + 1; i < n; ++i) {
            double factor = A[i][k] / A[k][k];
            for (size_t j = k; j < n; ++j)
                A[i][j] -= factor * A[k][j];
            x[i] -= factor * x[k];
        }
    }

    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        for (size_t j = i + 1; j < n; ++j)
            x[i] -= A[i][j] * x[j];
        x[i] /= A[i][i];
    }

    return GPUNumVec(x);
}

} // namespace SharedMath::NumericalMethods

#endif // SHAREDMATH_CUDA
