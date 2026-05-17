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

#include <complex>
#include <cstddef>
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
// Public API
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
ChannelizedSignal extractChannel(
    const std::vector<std::complex<double>>& iq,
    const ChannelizerParams&                 params);

} // namespace SharedMath::DSP

/// @} // DSP_Channelization