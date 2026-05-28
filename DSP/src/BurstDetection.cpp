/**
 * @file BurstDetection.cpp
 * @brief Implementation of burst detection algorithms.
 */

#include "BurstDetection.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace SharedMath::DSP {

namespace detail {

double medianBD(std::vector<double> v)
{
    if (v.empty()) return 0.0;
    const size_t n = v.size();
    std::nth_element(v.begin(), v.begin() + n / 2, v.end());
    if (n % 2 == 1) return v[n / 2];
    const double hi = v[n / 2];
    std::nth_element(v.begin(), v.begin() + n / 2 - 1, v.end());
    return 0.5 * (v[n / 2 - 1] + hi);
}

} // namespace detail

// ─────────────────────────────────────────────────────────────────────────────
// detectBursts
// ─────────────────────────────────────────────────────────────────────────────

std::vector<Burst> detectBursts(
    const std::vector<std::complex<double>>& iq,
    const BurstDetectionParams&              params)
{
    if (params.sampleRate <= 0.0)
        throw std::invalid_argument("detectBursts: sampleRate must be > 0");
    if (params.windowSize == 0)
        throw std::invalid_argument("detectBursts: windowSize must be > 0");
    if (params.overlap < 0.0 || params.overlap >= 1.0)
        throw std::invalid_argument("detectBursts: overlap must be in [0, 1)");

    if (iq.empty()) return {};

    const size_t N    = iq.size();
    const size_t wLen = params.windowSize;
    const size_t step = std::max<size_t>(1,
        static_cast<size_t>(std::round(static_cast<double>(wLen) * (1.0 - params.overlap))));
    const double fs   = params.sampleRate;

    // ── Per-window power in dBFS ──────────────────────────────────────────────
    std::vector<double> winPower;
    std::vector<size_t> winStart;
    winPower.reserve((N + step - 1) / step);
    winStart.reserve(winPower.capacity());

    for (size_t s = 0; s + wLen <= N; s += step) {
        double p = 0.0;
        for (size_t i = 0; i < wLen; ++i) p += std::norm(iq[s + i]);
        p /= static_cast<double>(wLen);
        winPower.push_back(10.0 * std::log10(std::max(p, 1e-300)));
        winStart.push_back(s);
    }
    if (winPower.empty()) return {};

    const double noiseFloor = detail::medianBD(winPower);
    const double threshold  = noiseFloor + params.thresholdDb;

    // ── Merge consecutive above-threshold windows into raw bursts ─────────────
    std::vector<Burst> raw;
    bool   inBurst = false;
    double peak    = 0.0;
    double sumPwr  = 0.0;
    size_t count   = 0;
    size_t bStart  = 0, bEnd = 0;

    auto pushBurst = [&]() {
        Burst b;
        b.startSample    = bStart;
        b.endSample      = bEnd;
        b.startTimeSec   = static_cast<double>(bStart) / fs;
        b.endTimeSec     = static_cast<double>(bEnd)   / fs;
        b.durationSec    = b.endTimeSec - b.startTimeSec;
        b.peakPowerDb    = peak;
        b.averagePowerDb = (count > 0) ? sumPwr / static_cast<double>(count) : peak;
        b.snrDb          = peak - noiseFloor;
        raw.push_back(b);
    };

    for (size_t wi = 0; wi < winPower.size(); ++wi) {
        const bool   above = (winPower[wi] >= threshold);
        const size_t wEnd  = winStart[wi] + wLen - 1;
        if (above) {
            if (!inBurst) {
                inBurst = true;
                bStart  = winStart[wi];
                peak    = winPower[wi];
                sumPwr  = 0.0;
                count   = 0;
            }
            bEnd = wEnd;
            if (winPower[wi] > peak) peak = winPower[wi];
            sumPwr += winPower[wi];
            ++count;
        } else {
            if (inBurst) { pushBurst(); inBurst = false; }
        }
    }
    if (inBurst) pushBurst();

    // ── Gap merging ───────────────────────────────────────────────────────────
    const size_t maxGapSamples =
        (params.maxGapSec > 0.0)
        ? static_cast<size_t>(std::ceil(params.maxGapSec * fs))
        : 0;

    std::vector<Burst> merged;
    for (auto& b : raw) {
        if (merged.empty()) {
            merged.push_back(b);
        } else {
            Burst& last = merged.back();
            const size_t gap =
                (b.startSample > last.endSample) ? b.startSample - last.endSample : 0;
            if (maxGapSamples > 0 && gap <= maxGapSamples) {
                last.endSample    = b.endSample;
                last.endTimeSec   = b.endTimeSec;
                last.durationSec  = last.endTimeSec - last.startTimeSec;
                last.peakPowerDb  = std::max(last.peakPowerDb, b.peakPowerDb);
                last.averagePowerDb =
                    0.5 * (last.averagePowerDb + b.averagePowerDb);  // approximate
                last.snrDb = last.peakPowerDb - noiseFloor;
            } else {
                merged.push_back(b);
            }
        }
    }

    // ── Filter by minimum duration ────────────────────────────────────────────
    std::vector<Burst> result;
    for (const auto& b : merged)
        if (b.durationSec >= params.minDurationSec)
            result.push_back(b);

    return result;
}

} // namespace SharedMath::DSP