#pragma once

/**
 * @file SignalDetection.h
 * @brief IQ-based signal detection: time-domain energy, spectral, and matched-filter methods.
 *
 * @defgroup DSP_SignalDetection Signal Detection
 * @ingroup DSP
 * @{
 *
 * Three orthogonal detection strategies are provided, all operating on
 * `std::vector<std::complex<double>>` IQ data:
 *
 * | Method              | Best for                                              |
 * |---------------------|-------------------------------------------------------|
 * | EnergyTimeDomain    | Wideband bursts, on/off-keyed signals                 |
 * | EnergySpectral      | Narrowband CW / known-frequency signals               |
 * | MatchedFilter       | Preambles and chirps with a known reference waveform  |
 *
 * ### Quick start
 * @code{.cpp}
 * #include <DSP/dsp.h>
 *
 * std::vector<std::complex<double>> iq = ...;
 *
 * SharedMath::DSP::SignalDetectionParams params;
 * params.sampleRate  = 2e6;
 * params.fftSize     = 4096;
 * params.thresholdDb = 10.0;
 * params.bandwidthHz = 50e3;
 *
 * auto result = SharedMath::DSP::detectSignals(
 *     iq, params, SharedMath::DSP::DetectionMethod::EnergySpectral);
 *
 * for (const auto& d : result.detections)
 *     std::cout << "SNR " << d.snrDb << " dB  cf=" << d.centerFrequencyHz << " Hz\n";
 * @endcode
 *
 * @}
 */

#include <complex>
#include <vector>
#include <cstddef>

namespace SharedMath::DSP {

// ─────────────────────────────────────────────────────────────────────────────
// DetectionMethod
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Selects the algorithm used by detectSignals().
 * @ingroup DSP_SignalDetection
 */
enum class DetectionMethod {
    EnergyTimeDomain, ///< Window-by-window mean power threshold in the time domain.
    EnergySpectral,   ///< Averaged, FFT-shifted power spectrum threshold.
    MatchedFilter     ///< Normalised cross-correlation with a known reference waveform.
};

// ─────────────────────────────────────────────────────────────────────────────
// SignalDetectionParams
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Configuration shared by all detection algorithms.
 * @ingroup DSP_SignalDetection
 */
struct SignalDetectionParams {
    double sampleRate         = 1.0;    ///< Sample rate in Hz.  Must be > 0.
    double bandwidthHz        = 0.0;    ///< Expected signal bandwidth (0 = full band).
    double centerFrequencyHz  = 0.0;    ///< Expected centre frequency in Hz (0 = DC).
    size_t fftSize            = 1024;   ///< Analysis window / FFT length.  Must be > 0.
    double overlap            = 0.5;    ///< Window overlap fraction in [0, 1).
    double thresholdDb        = 10.0;   ///< Required excess above noise floor in dB.
    bool   estimateNoiseFloor = true;   ///< If true, noise floor is estimated via median.
    size_t guardBins          = 2;      ///< Guard bins excluded from spectral region edges.
    double minDurationSec     = 0.0;    ///< Minimum detection duration (0 = no minimum).
};

// ─────────────────────────────────────────────────────────────────────────────
// SignalDetection
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Describes a single detected signal event.
 * @ingroup DSP_SignalDetection
 */
struct SignalDetection {
    bool   detected          = false; ///< Always `true` when returned inside a DetectionResult.
    double confidence        = 0.0;   ///< Normalised detection confidence in [0, 1].
    double snrDb             = 0.0;   ///< Estimated SNR (signal power minus noise floor) in dB.
    double powerDb           = 0.0;   ///< Peak signal power in dBFS.
    double noiseFloorDb      = 0.0;   ///< Estimated local noise floor in dBFS.
    double centerFrequencyHz = 0.0;   ///< Estimated centre frequency in Hz.
    double bandwidthHz       = 0.0;   ///< Estimated bandwidth in Hz.
    size_t startSample       = 0;     ///< First sample index of the detection.
    size_t endSample         = 0;     ///< Last sample index of the detection (inclusive).
    double startTimeSec      = 0.0;   ///< Corresponding start time in seconds.
    double endTimeSec        = 0.0;   ///< Corresponding end time in seconds.
};

// ─────────────────────────────────────────────────────────────────────────────
// DetectionResult
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Aggregate output returned by detectSignals() and the individual detectors.
 * @ingroup DSP_SignalDetection
 */
struct DetectionResult {
    std::vector<SignalDetection> detections;      ///< All detected signal events.
    std::vector<double>          frequencyAxisHz;  ///< Frequency axis for `spectrumDb` in Hz.
    std::vector<double>          spectrumDb;       ///< Averaged two-sided power spectrum in dBFS.
    double                       noiseFloorDb = 0.0; ///< Estimated global noise floor in dBFS.
};

// ─────────────────────────────────────────────────────────────────────────────
// detectEnergyTimeDomain
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Detect signal bursts via time-domain window energy.
 *
 * The IQ stream is divided into overlapping windows of length `params.fftSize`.
 * The step between windows is `round(fftSize · (1 − overlap))` samples.
 * For each window the mean instantaneous power is computed and converted to dBFS.
 * The noise floor is the median of all per-window powers when
 * `params.estimateNoiseFloor` is `true`; otherwise it is 0 dBFS.
 * Consecutive windows whose power exceeds `noiseFloor + params.thresholdDb`
 * are merged into a single `SignalDetection`.
 *
 * @param iq     Complex IQ samples.  May be empty.
 * @param params Detection configuration (see SignalDetectionParams).
 * @return DetectionResult with `detections` and `noiseFloorDb` filled in.
 *         `frequencyAxisHz` and `spectrumDb` are always empty.
 *
 * @throws std::invalid_argument if any parameter is out of range.
 *
 * @ingroup DSP_SignalDetection
 */
DetectionResult detectEnergyTimeDomain(
    const std::vector<std::complex<double>>& iq,
    const SignalDetectionParams&             params);

// ─────────────────────────────────────────────────────────────────────────────
// detectSpectral
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Detect signals by thresholding an averaged, FFT-shifted power spectrum.
 *
 * The IQ stream is divided into overlapping windows of length `params.fftSize`.
 * Each window is multiplied by a periodic Hann window (via `windowHann()`),
 * forward-FFT'd, and its squared magnitude is added to a running accumulator.
 * After averaging over all frames the spectrum is FFT-shifted so that the
 * frequency axis runs from −fs/2 to +fs/2 with DC at the centre.
 *
 * Contiguous spectral bins whose averaged power (in dBFS) exceeds
 * `noiseFloor + params.thresholdDb` are collected into `SignalDetection`
 * records.  The reported `centerFrequencyHz` and `bandwidthHz` of each record
 * are derived from the first and last bin indices of the region.
 *
 * @param iq     Complex IQ samples.  May be empty.
 * @param params Detection configuration.
 * @return DetectionResult with `detections`, `frequencyAxisHz`, `spectrumDb`,
 *         and `noiseFloorDb` filled in.
 *
 * @throws std::invalid_argument if any parameter is out of range.
 *
 * @ingroup DSP_SignalDetection
 */
DetectionResult detectSpectral(
    const std::vector<std::complex<double>>& iq,
    const SignalDetectionParams&             params);

// ─────────────────────────────────────────────────────────────────────────────
// detectMatchedFilter
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Detect a known reference waveform via a complex matched filter.
 *
 * Performs FFT-based complex cross-correlation between `iq` and `reference`:
 * @code
 *   corr[n] = IFFT( conj(FFT(reference)) · FFT(iq) )
 * @endcode
 * The correlation magnitude is normalised by the square-root of the reference
 * energy so that a perfect match yields a peak amplitude of 1.0 (confidence 1).
 *
 * Peaks in `|corr|` that exceed `noiseFloor + params.thresholdDb` (where the
 * noise floor is the median of `20·log10(|corr[n]|)`) are returned as
 * `SignalDetection` records.  A simple peak-search with a guard interval of
 * `ceil(reference.size() / 2)` samples prevents double-counting.
 *
 * @param iq        Complex IQ signal to search.  May be empty.
 * @param reference Known reference waveform.  Must not be empty and must not
 *                  be longer than `iq`.
 * @param params    Detection parameters.  `fftSize` is not used; the transform
 *                  length is chosen automatically as the next power of two
 *                  above `iq.size() + reference.size() − 1`.
 * @return DetectionResult with `detections` and `noiseFloorDb` filled in.
 *         `frequencyAxisHz` and `spectrumDb` are always empty.
 *
 * @throws std::invalid_argument if any parameter is out of range, `reference`
 *         is empty, or `reference` is longer than `iq`.
 *
 * @ingroup DSP_SignalDetection
 */
DetectionResult detectMatchedFilter(
    const std::vector<std::complex<double>>& iq,
    const std::vector<std::complex<double>>& reference,
    const SignalDetectionParams&             params);

// ─────────────────────────────────────────────────────────────────────────────
// detectSignals — unified dispatcher
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Unified signal-detection entry point.
 *
 * Dispatches to detectEnergyTimeDomain(), detectSpectral(), or
 * detectMatchedFilter() based on `method`.
 *
 * @param iq        Complex IQ samples.
 * @param params    Detection parameters.
 * @param method    Algorithm selector (@ref DetectionMethod).
 * @param reference Reference waveform required for DetectionMethod::MatchedFilter;
 *                  ignored for the other two methods.
 * @return DetectionResult appropriate for the chosen method.
 *
 * @throws std::invalid_argument if params are invalid, or if method is
 *         MatchedFilter and `reference` is empty or longer than `iq`.
 *
 * @ingroup DSP_SignalDetection
 */
DetectionResult detectSignals(
    const std::vector<std::complex<double>>& iq,
    const SignalDetectionParams&             params,
    DetectionMethod                          method,
    const std::vector<std::complex<double>>& reference = {});

} // namespace SharedMath::DSP

/// @} // DSP_SignalDetection