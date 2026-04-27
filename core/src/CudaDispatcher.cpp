#include "core/CudaDispatcher.h"

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>

#ifdef SHAREDMATH_CUDA
#  include <cuda_runtime.h>
#endif

namespace SharedMath::Core {

// ─────────────────────────────────────────────────────────────────────────────
// Worker  —  one per GPU
//
// The worker thread calls cudaSetDevice(deviceId) exactly once at startup so
// every job it executes runs on the right GPU with no extra overhead.
// ─────────────────────────────────────────────────────────────────────────────
struct CudaDispatcher::Worker {
    int deviceId = -1;

    std::thread               thread;
    std::queue<std::function<void()>> queue;
    std::mutex                mutex;
    std::condition_variable   taskCv;   // wakes worker when queue non-empty

    // Pending-task counter + condition to let sync() wait efficiently.
    std::atomic<int>          pending{0};
    std::mutex                doneMutex;
    std::condition_variable   doneCv;   // notified when pending reaches 0

    std::atomic<bool>         stop{false};

    Worker() = default;
    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;
};

// ── Singleton ─────────────────────────────────────────────────────────────────

CudaDispatcher& CudaDispatcher::instance() {
    static CudaDispatcher inst;
    return inst;
}

// ── Constructor — spawn one thread per GPU ────────────────────────────────────

CudaDispatcher::CudaDispatcher() {
    auto& mgr = CudaDeviceManager::instance();
    const int n = mgr.deviceCount();
    workers_.reserve(n);

    for (int i = 0; i < n; ++i) {
        auto w = std::make_unique<Worker>();
        w->deviceId = i;

        Worker* wp = w.get();

        w->thread = std::thread([wp]() {
            // Bind this OS thread to the assigned GPU once and for all.
#ifdef SHAREDMATH_CUDA
            cudaSetDevice(wp->deviceId);
#endif
            while (true) {
                std::function<void()> job;

                // ── Wait for a job or shutdown signal ─────────────────────────
                {
                    std::unique_lock<std::mutex> lock(wp->mutex);
                    wp->taskCv.wait(lock, [wp] {
                        return wp->stop.load(std::memory_order_relaxed) ||
                               !wp->queue.empty();
                    });

                    if (wp->stop.load(std::memory_order_relaxed) &&
                        wp->queue.empty())
                        break;

                    job = std::move(wp->queue.front());
                    wp->queue.pop();
                }

                // ── Execute ───────────────────────────────────────────────────
                job();

                // ── Update pending counter and wake sync() if all done ────────
                {
                    std::unique_lock<std::mutex> dLock(wp->doneMutex);
                    if (wp->pending.fetch_sub(1, std::memory_order_acq_rel) == 1)
                        wp->doneCv.notify_all();
                }
            }
        });

        workers_.push_back(std::move(w));
    }
}

// ── Destructor — graceful shutdown ────────────────────────────────────────────

CudaDispatcher::~CudaDispatcher() {
    for (auto& w : workers_) {
        {
            std::unique_lock<std::mutex> lock(w->mutex);
            w->stop.store(true, std::memory_order_relaxed);
        }
        w->taskCv.notify_one();
        if (w->thread.joinable())
            w->thread.join();
    }
}

// ── enqueue (called from submitTo template) ───────────────────────────────────

void CudaDispatcher::enqueue(int deviceId, std::function<void()> job) {
    auto& w = *workers_[deviceId];
    {
        std::unique_lock<std::mutex> lock(w.mutex);
        // Increment pending *inside* the mutex so sync() cannot observe
        // pending==0 between enqueue and the worker picking up the job.
        w.pending.fetch_add(1, std::memory_order_relaxed);
        w.queue.push(std::move(job));
    }
    w.taskCv.notify_one();
}

// ── sync ──────────────────────────────────────────────────────────────────────

void CudaDispatcher::sync() {
    for (auto& w : workers_) {
        std::unique_lock<std::mutex> lock(w->doneMutex);
        w->doneCv.wait(lock, [&w] {
            return w->pending.load(std::memory_order_acquire) == 0;
        });
    }
}

// ── Queries ───────────────────────────────────────────────────────────────────

int CudaDispatcher::deviceCount() const noexcept {
    return static_cast<int>(workers_.size());
}

std::vector<int> CudaDispatcher::pendingCounts() const {
    std::vector<int> counts;
    counts.reserve(workers_.size());
    for (const auto& w : workers_)
        counts.push_back(w->pending.load(std::memory_order_relaxed));
    return counts;
}

} // namespace SharedMath::Core
