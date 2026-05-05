#pragma once

/**
 * @file Channelization.h
 * @brief Digital down-conversion and channel extraction for wideband IQ.
 *
 * @defgroup DSP_Channelization Channelization
 * @ingroup DSP
 * @{
 *
 * extractChannel() implements a three-step DDC:
 *  1. Frequency shift to baseband.
 *  2. Windowed-sinc low-pass FIR filter.
 *  3. Optional integer-ratio decimation.
 *
 * ### Example
 * @code{.cpp}
 * SharedMath::DSP::ChannelizerParams p;
 * p.sampleRate        = 20e6;
 * p.centerFrequencyHz = 5e6;
 * p.bandwidthHz       = 200e3;
 * p.outputSampleRate  = 400e3;
 * p.filterOrder       = 128;
 * auto ch = SharedMath::DSP::extractChannel(iq, p);
 * @endcode
 *
 * @}
 */

#include "Window.h"

#include <algorithm>
#include <complex>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace SharedMath::DSP {

// ─────────────────────────────────────────────────────────────────────────────
// Data types
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Configuration for extractChannel().
 * @ingroup DSP_Channelization
 */
struct ChannelizerParams {
    double sampleRate        = 1.0;  ///< Input sample rate in Hz.  Must be > 0.
    double centerFrequencyHz = 0.0;  ///< Channel centre frequency in Hz.
    double bandwidthHz       = 0.0;  ///< Channel bandwidth in Hz.  Must be > 0.
    double outputSampleRate  = 0.0;  ///< Desired output sample rate (0 = no decimation).
    size_t filterOrder       = 128;  ///< FIR filter order (length = order + 1).  Must be > 0.
};

/**
 * @brief Output of extractChannel().
 * @ingroup DSP_Channelization
 */
struct ChannelizedSignal {
    std::vector<std::complex<double>> iq;    ///< Baseband IQ samples.
    double sampleRate        = 1.0;          ///< Actual output sample rate in Hz.
    double centerFrequencyHz = 0.0;          ///< Always 0 (signal is at DC).
    double bandwidthHz       = 0.0;          ///< Requested channel bandwidth in Hz.
};

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

namespace detail {

/**
 * @brief Design a windowed-sinc low-pass FIR filter.
 *
 * Generates a length-(order+1) FIR using a Hann window.
 *
 * @param order       Filter order; number of coefficients = order + 1.
 * @param cutoffNorm  Normalised cutoff frequency in (0, 0.5).
 * @return FIR coefficient vector of length order + 1.
 */
inline std::vector<double> designLpFIR(size_t order, double cutoffNorm)
{
    const size_t len = order + 1;
    const double wc  = 2.0 * M_PI * cutoffNorm;
    std::vector<double> h(len);
    for (size_t i = 0; i < len; ++i) {
        const double n = static_cast<double>(i) - static_cast<double>(order) * 0.5;
        h[i] = (std::abs(n) < 1e-10)
            ? 2.0 * cutoffNorm
            : std::sin(wc * n) / (M_PI * n);
    }
    auto win = windowHann(len, /*symmetric=*/true);
    for (size_t i = 0; i < len; ++i) h[i] *= win[i];
    return h;
}

/**
 * @brief Apply real FIR coefficients to complex IQ via direct (causal) convolution.
 *
 * For each output sample the filter sums `h[k] · x[n-k]` for k = 0…M-1.
 * The real and imaginary parts are processed jointly to avoid two passes.
 *
 * @param x IQ input.
 * @param h Real FIR coefficients.
 * @return Filtered IQ (same length as `x`).
 */
inline std::vector<std::complex<double>> applyComplexFIR(
    const std::vector<std::complex<double>>& x,
    const std::vector<double>&               h)
{
    const size_t N = x.size();
    const size_t M = h.size();
    std::vector<std::complex<double>> y(N, {0.0, 0.0});
    for (size_t n = 0; n < N; ++n) {
        const size_t kMax = std::min(M, n + 1);
        for (size_t k = 0; k < kMax; ++k)
            y[n] += h[k] * x[n - k];
    }
    return y;
}

/**
 * @brief Decimate a complex IQ vector by an integer factor.
 *
 * Keeps every `factor`-th sample starting at index 0.
 * Anti-aliasing must be applied before calling this function.
 *
 * @param x      Input IQ (pre-filtered).
 * @param factor Decimation factor.  factor ≤ 1 → copy.
 * @return Decimated IQ vector of length ⌈N/factor⌉.
 */
inline std::vector<std::complex<double>> decimateComplex(
    const std::vector<std::complex<double>>& x, size_t factor)
{
    if (factor <= 1) return x;
    std::vector<std::complex<double>> y;
    y.reserve((x.size() + factor - 1) / factor);
    for (size_t i = 0; i < x.size(); i += factor)
        y.push_back(x[i]);
    return y;
}

} // namespace detail

// ─────────────────────────────────────────────────────────────────────────────
// extractChannel
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Extract and down-convert a single channel from a wideband IQ stream.
 *
 * Three sequential operations:
 *  -# **Frequency shift** — multiply each sample by
 *     `exp(−j·2π·centerFrequencyHz·n / sampleRate)` to move the channel to DC.
 *  -# **Low-pass filtering** — apply a windowed-sinc FIR with cutoff
 *     `bandwidthHz / 2`, suppressing all energy outside the channel.
 *  -# **Decimation** (optional) — if `outputSampleRate > 0` and smaller than
 *     `sampleRate`, decimate by the nearest integer ratio.
 *
 * @param iq     Wideband input IQ.  Empty → returns empty ChannelizedSignal.
 * @param params Channel extraction parameters.
 * @return ChannelizedSignal with baseband IQ (`centerFrequencyHz = 0`).
 *
 * @throws std::invalid_argument if `sampleRate ≤ 0`, `bandwidthHz ≤ 0`,
 *         `filterOrder == 0`, or `centerFrequencyHz` is outside
 *         [−sampleRate/2, +sampleRate/2].
 *
 * @ingroup DSP_Channelization
 */
inline ChannelizedSignal extractChannel(
    const std::vector<std::complex<double>>& iq,
    const ChannelizerParams&                 params)
{
    if (params.sampleRate <= 0.0)
        throw std::invalid_argument("extractChannel: sampleRate must be > 0");
    if (params.bandwidthHz <= 0.0)
        throw std::invalid_argument("extractChannel: bandwidthHz must be > 0");
    if (params.filterOrder == 0)
        throw std::invalid_argument("extractChannel: filterOrder must be > 0");
    const double nyq = params.sampleRate * 0.5;
    if (params.centerFrequencyHz < -nyq || params.centerFrequencyHz > nyq)
        throw std::invalid_argument(
            "extractChannel: centerFrequencyHz out of [-sampleRate/2, sampleRate/2]");

    // Determine output sample rate
    const double outFs =
        (params.outputSampleRate > 0.0 && params.outputSampleRate < params.sampleRate)
        ? params.outputSampleRate
        : params.sampleRate;

    ChannelizedSignal result;
    result.centerFrequencyHz = 0.0;
    result.bandwidthHz       = params.bandwidthHz;
    result.sampleRate        = outFs;

    if (iq.empty()) return result;

    // ── 1. Frequency shift to baseband ────────────────────────────────────────
    const size_t N        = iq.size();
    const double phaseInc = -2.0 * M_PI * params.centerFrequencyHz / params.sampleRate;
    std::vector<std::complex<double>> shifted(N);
    for (size_t n = 0; n < N; ++n)
        shifted[n] = iq[n] * std::polar(1.0, phaseInc * static_cast<double>(n));

    // ── 2. Low-pass FIR filter ────────────────────────────────────────────────
    // Normalised cutoff in (0, 0.5) relative to the input sample rate
    const double cutoffNorm = std::min(
        std::max((params.bandwidthHz * 0.5) / params.sampleRate, 1e-4),
        0.49);

    const auto h        = detail::designLpFIR(params.filterOrder, cutoffNorm);
    auto       filtered = detail::applyComplexFIR(shifted, h);

    // ── 3. Decimation ─────────────────────────────────────────────────────────
    if (outFs < params.sampleRate) {
        const size_t factor = std::max<size_t>(1,
            static_cast<size_t>(std::round(params.sampleRate / outFs)));
        result.iq        = detail::decimateComplex(filtered, factor);
        result.sampleRate = params.sampleRate / static_cast<double>(factor);
    } else {
        result.iq        = std::move(filtered);
        result.sampleRate = params.sampleRate;
    }

    return result;
}

} // namespace SharedMath::DSP

/// @} // DSP_Channelization
