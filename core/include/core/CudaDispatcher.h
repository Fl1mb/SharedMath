#pragma once

#include "CudaDeviceManager.h"

#include <functional>
#include <future>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace SharedMath::Core {

/// ─────────────────────────────────────────────────────────────────────────────
/// CudaDispatcher  —  singleton
///
/// Manages one dedicated worker thread per GPU.  Each worker calls
/// cudaSetDevice once at start so every job it runs is automatically on the
/// right device.
///
/// Usage:
///
///   auto& disp = CudaDispatcher::instance();
///
///   // Submit to the least-loaded GPU (returns std::future).
///   auto fut = disp.submit([](int deviceId) -> float {
///       // your CUDA work here — device is already set
///       return computeOnGPU(deviceId);
///   });
///
///   float result = fut.get();   // blocks until done
///
///   // Or submit to a specific GPU:
///   auto fut2 = disp.submitTo(2, myKernel);
///
///   // Wait for ALL pending tasks across every GPU:
///   disp.sync();
///
/// ─────────────────────────────────────────────────────────────────────────────
class CudaDispatcher {
public:
    /// ── Singleton access ─────────────────────────────────────────────────────
    static CudaDispatcher& instance();

    CudaDispatcher(const CudaDispatcher&)            = delete;
    CudaDispatcher& operator=(const CudaDispatcher&) = delete;
    ~CudaDispatcher();

    // ── Task submission ───────────────────────────────────────────────────────

    /// Submit a callable to the least-loaded GPU.
    ///
    /// The callable must accept a single int (the device id it runs on)
    /// and may return any type (including void).
    ///
    ///   auto fut = dispatcher.submit([](int dev) -> double { ... });
    ///   double val = fut.get();
    ///
    /// Throws std::runtime_error when no CUDA device is available.
    template<typename F>
    auto submit(F&& task) -> std::future<std::invoke_result_t<F, int>>;

    /// Submit to a specific device.
    /// Throws std::out_of_range for an invalid deviceId.
    template<typename F>
    auto submitTo(int deviceId, F&& task)
        -> std::future<std::invoke_result_t<F, int>>;

    /// ── Synchronisation ───────────────────────────────────────────────────────

    /// Block the calling thread until every submitted task has completed
    /// on every GPU.
    void sync();

    /// ── Queries ───────────────────────────────────────────────────────────────

    int deviceCount() const noexcept;

    /// Snapshot of pending-task counts, indexed by device id.
    std::vector<int> pendingCounts() const;

private:
    CudaDispatcher();   // spawns one worker thread per available GPU

    /// Forward-declared so the .cpp owns the full definition.
    struct Worker;
    std::vector<std::unique_ptr<Worker>> workers_;

    /// Post a pre-bound void() job to a specific device queue.
    void enqueue(int deviceId, std::function<void()> job);
};

// ── Template implementations ─────────────────────────────────────────────────
// Kept in the header because they depend on the return type of the callable.

template<typename F>
auto CudaDispatcher::submit(F&& task)
    -> std::future<std::invoke_result_t<F, int>>
{
    if (workers_.empty())
        throw std::runtime_error(
            "CudaDispatcher::submit — no CUDA devices available");

    int dev = CudaDeviceManager::instance().leastLoadedDevice();
    return submitTo(dev, std::forward<F>(task));
}

template<typename F>
auto CudaDispatcher::submitTo(int deviceId, F&& task)
    -> std::future<std::invoke_result_t<F, int>>
{
    using R = std::invoke_result_t<F, int>;

    if (deviceId < 0 || deviceId >= static_cast<int>(workers_.size()))
        throw std::out_of_range(
            "CudaDispatcher::submitTo — invalid device id");

    // Package the task.  shared_ptr so the lambda below can copy it safely.
    auto pt = std::make_shared<std::packaged_task<R()>>(
        [f = std::forward<F>(task), deviceId]() mutable {
            return f(deviceId);
        });

    auto fut = pt->get_future();

    CudaDeviceManager::instance().incrementLoad(deviceId);

    enqueue(deviceId, [pt, deviceId]() mutable {
        (*pt)();
        CudaDeviceManager::instance().decrementLoad(deviceId);
    });

    return fut;
}

} // namespace SharedMath::Core
