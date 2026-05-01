#include "core/CudaDeviceManager.h"

#include <algorithm>
#include <memory>
#include <stdexcept>

#ifdef SHAREDMATH_CUDA
#  include <cuda_runtime.h>
#endif

namespace SharedMath::Core {

// ── Singleton ─────────────────────────────────────────────────────────────────

CudaDeviceManager& CudaDeviceManager::instance() {
    static CudaDeviceManager inst;
    return inst;
}

// ── Constructor — device discovery ────────────────────────────────────────────

CudaDeviceManager::CudaDeviceManager() {
#ifdef SHAREDMATH_CUDA
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0)
        return;   // no CUDA devices — vectors stay empty

    devices_.reserve(count);
    // std::atomic<int> is not movable in all standard versions, so resize
    // in-place and default-initialise (value = 0).
    activeTasks_.reserve(count);

    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, i);

        std::size_t freeMem = 0, totalMem = 0;
        cudaSetDevice(i);
        cudaMemGetInfo(&freeMem, &totalMem);

        CudaDeviceInfo info;
        info.id                     = i;
        info.name                   = prop.name;
        info.totalMemoryBytes       = prop.totalGlobalMem;
        info.freeMemoryBytes        = freeMem;
        info.computeCapabilityMajor = prop.major;
        info.computeCapabilityMinor = prop.minor;
        info.multiprocessorCount    = prop.multiProcessorCount;
        info.maxThreadsPerBlock     = prop.maxThreadsPerBlock;
        info.warpSize               = prop.warpSize;
        for (int d = 0; d < 3; ++d) {
            info.maxThreadsDim[d] = prop.maxThreadsDim[d];
            info.maxGridSize[d]   = prop.maxGridSize[d];
        }
        devices_.push_back(std::move(info));
        activeTasks_.push_back(std::make_unique<std::atomic<int>>(0));
    }
#endif // SHAREDMATH_CUDA
}

// ── Device enumeration ────────────────────────────────────────────────────────

int CudaDeviceManager::deviceCount() const noexcept {
    return static_cast<int>(devices_.size());
}

const CudaDeviceInfo& CudaDeviceManager::deviceInfo(int id) const {
    if (id < 0 || id >= static_cast<int>(devices_.size()))
        throw std::out_of_range("CudaDeviceManager::deviceInfo — invalid id");
    return devices_[id];
}

const std::vector<CudaDeviceInfo>& CudaDeviceManager::devices() const noexcept {
    return devices_;
}

// ── Load balancing ────────────────────────────────────────────────────────────

int CudaDeviceManager::leastLoadedDevice() const noexcept {
    if (devices_.empty()) return -1;

    int best    = 0;
    int minLoad = activeTasks_[0]->load(std::memory_order_relaxed);

    for (int i = 1; i < static_cast<int>(activeTasks_.size()); ++i) {
        int load = activeTasks_[i]->load(std::memory_order_relaxed);
        if (load < minLoad) {
            minLoad = load;
            best    = i;
        }
    }
    return best;
}

int CudaDeviceManager::activeTaskCount(int deviceId) const noexcept {
    if (deviceId < 0 || deviceId >= static_cast<int>(activeTasks_.size()))
        return 0;
    return activeTasks_[deviceId]->load(std::memory_order_relaxed);
}

void CudaDeviceManager::incrementLoad(int deviceId) noexcept {
    if (deviceId >= 0 && deviceId < static_cast<int>(activeTasks_.size()))
        activeTasks_[deviceId]->fetch_add(1, std::memory_order_relaxed);
}

void CudaDeviceManager::decrementLoad(int deviceId) noexcept {
    if (deviceId >= 0 && deviceId < static_cast<int>(activeTasks_.size()))
        activeTasks_[deviceId]->fetch_sub(1, std::memory_order_relaxed);
}

} // namespace SharedMath::Core
