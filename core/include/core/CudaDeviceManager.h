#pragma once

#include "CudaDeviceInfo.h"
#include <vector>
#include <atomic>
#include <stdexcept>

namespace SharedMath::Core {

// ─────────────────────────────────────────────────────────────────────────────
// CudaDeviceManager  —  singleton
//
// Enumerates all available CUDA GPUs at first use and keeps an atomic
// per-device "active task" counter that the dispatcher uses for load
// balancing.  Works at compile time without CUDA — reports 0 devices.
// ─────────────────────────────────────────────────────────────────────────────
class CudaDeviceManager {
public:
    // ── Singleton access ─────────────────────────────────────────────────────
    static CudaDeviceManager& instance();

    CudaDeviceManager(const CudaDeviceManager&)            = delete;
    CudaDeviceManager& operator=(const CudaDeviceManager&) = delete;

    // ── Device enumeration ───────────────────────────────────────────────────

    /// Number of CUDA-capable devices found at startup.
    int deviceCount() const noexcept;

    /// Returns true when at least one GPU is available.
    bool hasDevices() const noexcept { return deviceCount() > 0; }

    /// Static properties of a single device.
    /// Throws std::out_of_range for invalid id.
    const CudaDeviceInfo& deviceInfo(int id) const;

    /// Reference to the full device list (read-only).
    const std::vector<CudaDeviceInfo>& devices() const noexcept;

    // ── Load balancing ───────────────────────────────────────────────────────

    /// Index of the GPU with the fewest currently active tasks.
    /// Returns -1 when no devices are available.
    int leastLoadedDevice() const noexcept;

    /// Number of tasks currently executing (or queued) on a device.
    int activeTaskCount(int deviceId) const noexcept;

    // Called internally by CudaDispatcher — not intended for direct use.
    void incrementLoad(int deviceId) noexcept;
    void decrementLoad(int deviceId) noexcept;

private:
    CudaDeviceManager();   // discovers devices; safe to call without CUDA

    std::vector<CudaDeviceInfo>   devices_;
    // One atomic counter per device; value = #tasks in-flight.
    // std::vector<std::atomic> is non-copyable, so we store by value and
    // never copy the vector after construction.
    std::vector<std::atomic<int>> activeTasks_;
};

} // namespace SharedMath::Core
