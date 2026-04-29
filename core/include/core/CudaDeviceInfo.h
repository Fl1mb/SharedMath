#pragma once

#include <string>
#include <cstddef>

namespace SharedMath::Core {

// ─────────────────────────────────────────────────────────────────────────────
// Snapshot of a single CUDA device's static properties.
// Populated once at startup by CudaDeviceManager.
// ─────────────────────────────────────────────────────────────────────────────
struct CudaDeviceInfo {
    int         id                     = -1;
    std::string name;

    // Memory
    std::size_t totalMemoryBytes       = 0;
    std::size_t freeMemoryBytes        = 0;   // sampled at startup

    // Compute capability
    int         computeCapabilityMajor = 0;
    int         computeCapabilityMinor = 0;

    // Hardware
    int         multiprocessorCount    = 0;
    int         maxThreadsPerBlock     = 0;
    int         warpSize               = 0;
    int         maxThreadsDim[3]       = {};
    int         maxGridSize[3]         = {};

    // ── Convenience ──────────────────────────────────────────────────────────

    double totalMemoryGB() const noexcept {
        return static_cast<double>(totalMemoryBytes) / (1024.0 * 1024.0 * 1024.0);
    }

    double freeMemoryGB() const noexcept {
        return static_cast<double>(freeMemoryBytes) / (1024.0 * 1024.0 * 1024.0);
    }

    // Returns "8.6", "9.0", etc.
    std::string computeCapabilityStr() const {
        return std::to_string(computeCapabilityMajor) + "." +
               std::to_string(computeCapabilityMinor);
    }

    bool isValid() const noexcept { return id >= 0; }
};

} // namespace SharedMath::Core
