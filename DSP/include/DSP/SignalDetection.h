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

#include "FFTPlan.h"
#include "FFTConfig.h"
#include "Window.h"

#include <algorithm>
#include <complex>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

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
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

namespace detail {

/**
 * @brief Compute the median of a vector without modifying the original.
 * @param v Input values (copied internally).
 * @return Median value, or 0 if `v` is empty.
 */
inline double median(std::vector<double> v)
{
    if (v.empty()) return 0.0;
    const size_t n = v.size();
    std::nth_element(v.begin(), v.begin() + n / 2, v.end());
    if (n % 2 == 1) return v[n / 2];
    const double hi = v[n / 2];
    std::nth_element(v.begin(), v.begin() + n / 2 - 1, v.end());
    return 0.5 * (v[n / 2 - 1] + hi);
}

/**
 * @brief Convert linear power to dBFS: `10·log10(p)`.
 * @param p Linear power value (clamped away from zero to avoid −∞).
 * @return Power in dBFS.
 */
inline double toDb(double p) noexcept
{
    return 10.0 * std::log10(std::max(p, 1e-300));
}

/**
 * @brief Validate SignalDetectionParams and throw on invalid values.
 *
 * @param p Parameters to validate.
 * @throws std::invalid_argument if any field is out of range.
 */
inline void validateParams(const SignalDetectionParams& p)
{
    if (p.sampleRate <= 0.0)
        throw std::invalid_argument("SignalDetection: sampleRate must be > 0");
    if (p.fftSize == 0)
        throw std::invalid_argument("SignalDetection: fftSize must be > 0");
    if (p.overlap < 0.0 || p.overlap >= 1.0)
        throw std::invalid_argument("SignalDetection: overlap must be in [0, 1)");
    if (p.bandwidthHz < 0.0)
        throw std::invalid_argument("SignalDetection: bandwidthHz must be >= 0");
    const double nyq = p.sampleRate * 0.5;
    if (p.centerFrequencyHz < -nyq || p.centerFrequencyHz > nyq)
        throw std::invalid_argument(
            "SignalDetection: centerFrequencyHz out of [-sampleRate/2, sampleRate/2]");
}

} // namespace detail

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
inline DetectionResult detectEnergyTimeDomain(
    const std::vector<std::complex<double>>& iq,
    const SignalDetectionParams&             params)
{
    detail::validateParams(params);
    DetectionResult result;
    if (iq.empty()) return result;

    const size_t N    = iq.size();
    const size_t wLen = params.fftSize;
    const size_t step = std::max<size_t>(1,
        static_cast<size_t>(std::round(static_cast<double>(wLen) * (1.0 - params.overlap))));

    // ── Compute per-window mean power in dBFS ─────────────────────────────────
    std::vector<double> winPowerDb;
    std::vector<size_t> winStart;
    winPowerDb.reserve((N + step - 1) / step);
    winStart.reserve(winPowerDb.capacity());

    for (size_t s = 0; s + wLen <= N; s += step) {
        double pwr = 0.0;
        for (size_t i = 0; i < wLen; ++i)
            pwr += std::norm(iq[s + i]);        // |x|²
        pwr /= static_cast<double>(wLen);
        winPowerDb.push_back(detail::toDb(pwr));
        winStart.push_back(s);
    }

    if (winPowerDb.empty()) return result;

    // ── Noise floor ───────────────────────────────────────────────────────────
    const double noiseFloor = params.estimateNoiseFloor
        ? detail::median(winPowerDb)
        : 0.0;
    result.noiseFloorDb = noiseFloor;

    const double threshold = noiseFloor + params.thresholdDb;
    const double fs        = params.sampleRate;

    // ── Merge consecutive above-threshold windows into detections ─────────────
    bool   inDetection    = false;
    double detPeakPower   = 0.0;
    size_t detStartSample = 0;
    size_t detEndSample   = 0;

    auto finalise = [&]() {
        const double dur =
            static_cast<double>(detEndSample - detStartSample) / fs;
        if (dur < params.minDurationSec) return;

        SignalDetection d;
        d.detected     = true;
        d.startSample  = detStartSample;
        d.endSample    = detEndSample;
        d.startTimeSec = static_cast<double>(detStartSample) / fs;
        d.endTimeSec   = static_cast<double>(detEndSample)   / fs;
        d.powerDb      = detPeakPower;
        d.noiseFloorDb = noiseFloor;
        d.snrDb        = detPeakPower - noiseFloor;
        // Confidence: linearly ramps from 0 at threshold to 1 at threshold+3×thresholdDb
        d.confidence   = std::min(1.0, d.snrDb / (params.thresholdDb * 3.0));
        result.detections.push_back(d);
    };

    for (size_t wi = 0; wi < winPowerDb.size(); ++wi) {
        const bool   above = (winPowerDb[wi] >= threshold);
        const size_t wEnd  = winStart[wi] + wLen - 1;

        if (above) {
            if (!inDetection) {
                inDetection    = true;
                detStartSample = winStart[wi];
                detPeakPower   = winPowerDb[wi];
            }
            detEndSample = wEnd;
            if (winPowerDb[wi] > detPeakPower) detPeakPower = winPowerDb[wi];
        } else {
            if (inDetection) {
                finalise();
                inDetection = false;
            }
        }
    }
    if (inDetection) finalise();

    return result;
}

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
inline DetectionResult detectSpectral(
    const std::vector<std::complex<double>>& iq,
    const SignalDetectionParams&             params)
{
    detail::validateParams(params);
    DetectionResult result;
    if (iq.empty()) return result;

    const size_t N    = iq.size();
    const size_t M    = params.fftSize;
    const size_t step = std::max<size_t>(1,
        static_cast<size_t>(std::round(static_cast<double>(M) * (1.0 - params.overlap))));
    const double fs   = params.sampleRate;

    // Periodic (spectral-analysis) Hann window of length M
    const auto win = windowHann(M, /*symmetric=*/false);

    double winSumSq = 0.0;
    for (double w : win) winSumSq += w * w;

    // ── Accumulate two-sided power spectrum ───────────────────────────────────
    std::vector<double> psdAccum(M, 0.0);
    size_t numFrames = 0;

    for (size_t s = 0; s + M <= N; s += step) {
        std::vector<std::complex<double>> frame(M);
        for (size_t i = 0; i < M; ++i)
            frame[i] = iq[s + i] * win[i];

        FFTPlan::create(M, {FFTDirection::Forward, FFTNorm::None}).execute(frame);

        for (size_t k = 0; k < M; ++k)
            psdAccum[k] += std::norm(frame[k]);

        ++numFrames;
    }

    if (numFrames == 0) return result;

    const double nf    = static_cast<double>(numFrames);
    const double scale = 1.0 / (winSumSq * nf);

    // ── FFT-shift: bin k → shifted index (k + M/2) % M ───────────────────────
    std::vector<double> spectrumLin(M);
    for (size_t k = 0; k < M; ++k)
        spectrumLin[(k + M / 2) % M] = psdAccum[k] * scale;

    // ── Build frequency axis: from −fs/2 to just below +fs/2 ─────────────────
    const double binHz = fs / static_cast<double>(M);
    result.frequencyAxisHz.resize(M);
    for (size_t k = 0; k < M; ++k)
        result.frequencyAxisHz[k] =
            (static_cast<double>(k) - static_cast<double>(M / 2)) * binHz;

    // ── Convert accumulated spectrum to dBFS ──────────────────────────────────
    result.spectrumDb.resize(M);
    for (size_t k = 0; k < M; ++k)
        result.spectrumDb[k] = detail::toDb(spectrumLin[k]);

    // ── Noise floor ───────────────────────────────────────────────────────────
    const double noiseFloor = params.estimateNoiseFloor
        ? detail::median(result.spectrumDb)
        : 0.0;
    result.noiseFloorDb = noiseFloor;

    const double threshold = noiseFloor + params.thresholdDb;

    // ── Detect contiguous above-threshold spectral regions ────────────────────
    bool   inRegion   = false;
    size_t regStart   = 0;
    double regPeakPwr = 0.0;

    auto finaliseRegion = [&](size_t regEnd) {
        // Apply optional guard bins for a more conservative bandwidth estimate.
        const size_t guardedStart =
            (regStart > params.guardBins) ? regStart - params.guardBins : 0;
        const size_t guardedEnd =
            std::min(M - 1, regEnd + params.guardBins);

        if (params.bandwidthHz > 0.0) {
            const double expectedLow  = params.centerFrequencyHz - 0.5 * params.bandwidthHz;
            const double expectedHigh = params.centerFrequencyHz + 0.5 * params.bandwidthHz;
            const double regionLow    = result.frequencyAxisHz[guardedStart];
            const double regionHigh   = result.frequencyAxisHz[guardedEnd];
            if (regionHigh < expectedLow || regionLow > expectedHigh) return;
        }

        const double cfHz =
            0.5 * (result.frequencyAxisHz[guardedStart] + result.frequencyAxisHz[guardedEnd]);
        const double bwHz =
            result.frequencyAxisHz[guardedEnd] - result.frequencyAxisHz[guardedStart] + binHz;

        SignalDetection d;
        d.detected          = true;
        d.powerDb           = regPeakPwr;
        d.noiseFloorDb      = noiseFloor;
        d.snrDb             = regPeakPwr - noiseFloor;
        d.centerFrequencyHz = cfHz;
        d.bandwidthHz       = bwHz;
        d.confidence        = std::min(1.0, d.snrDb / (params.thresholdDb * 3.0));
        // Spectral detector reports the whole IQ block as the time extent
        d.startSample  = 0;
        d.endSample    = N - 1;
        d.startTimeSec = 0.0;
        d.endTimeSec   = static_cast<double>(N - 1) / fs;
        result.detections.push_back(d);
    };

    const bool restrictBand = params.bandwidthHz > 0.0;
    const double expectedLow =
        params.centerFrequencyHz - 0.5 * params.bandwidthHz;
    const double expectedHigh =
        params.centerFrequencyHz + 0.5 * params.bandwidthHz;

    for (size_t k = 0; k < M; ++k) {
        const bool inSearchBand = !restrictBand ||
            (result.frequencyAxisHz[k] >= expectedLow &&
             result.frequencyAxisHz[k] <= expectedHigh);
        const bool above = inSearchBand && (result.spectrumDb[k] >= threshold);
        if (above) {
            if (!inRegion) {
                inRegion   = true;
                regStart   = k;
                regPeakPwr = result.spectrumDb[k];
            }
            if (result.spectrumDb[k] > regPeakPwr) regPeakPwr = result.spectrumDb[k];
        } else {
            if (inRegion) {
                finaliseRegion(k - 1);
                inRegion = false;
            }
        }
    }
    if (inRegion) finaliseRegion(M - 1);

    return result;
}

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
inline DetectionResult detectMatchedFilter(
    const std::vector<std::complex<double>>& iq,
    const std::vector<std::complex<double>>& reference,
    const SignalDetectionParams&             params)
{
    detail::validateParams(params);
    DetectionResult result;
    if (iq.empty()) return result;

    if (reference.empty())
        throw std::invalid_argument(
            "detectMatchedFilter: reference must not be empty");
    if (reference.size() > iq.size())
        throw std::invalid_argument(
            "detectMatchedFilter: reference must not be longer than iq");

    const size_t La     = iq.size();
    const size_t Lb     = reference.size();
    const size_t outLen = La + Lb - 1;
    const double fs     = params.sampleRate;

    // Next power of two ≥ outLen
    size_t fftN = 1;
    while (fftN < outLen) fftN <<= 1;

    // Zero-pad both signals to fftN
    std::vector<std::complex<double>> A(fftN, {0.0, 0.0});
    std::vector<std::complex<double>> B(fftN, {0.0, 0.0});
    for (size_t i = 0; i < La; ++i) A[i] = iq[i];
    for (size_t i = 0; i < Lb; ++i) B[i] = reference[i];

    {
        auto fwd = FFTPlan::create(fftN, {FFTDirection::Forward, FFTNorm::None});
        fwd.execute(A);
        fwd.execute(B);
    }

    // Cross-correlation: IFFT( conj(B) · A )
    for (size_t i = 0; i < fftN; ++i) A[i] = std::conj(B[i]) * A[i];
    FFTPlan::create(fftN, {FFTDirection::Inverse, FFTNorm::ByN}).execute(A);

    // Normalise by reference energy so a perfect full-length match yields 1.
    double refEnergy = 0.0;
    for (const auto& s : reference) refEnergy += std::norm(s);
    if (refEnergy < 1e-300) {
        // Zero-energy reference → no useful detection possible
        result.noiseFloorDb = detail::toDb(0.0);
        return result;
    }
    const double refScale = 1.0 / refEnergy;

    // Only the first La samples carry physically meaningful information
    std::vector<double> corrMag(La);
    for (size_t i = 0; i < La; ++i)
        corrMag[i] = std::abs(A[i]) * refScale;

    // ── Convert to dB for thresholding ────────────────────────────────────────
    std::vector<double> corrDb(La);
    for (size_t i = 0; i < La; ++i)
        corrDb[i] = detail::toDb(corrMag[i] * corrMag[i]);

    const double noiseFloor = params.estimateNoiseFloor
        ? detail::median(corrDb)
        : 0.0;
    result.noiseFloorDb = noiseFloor;

    const double threshold = noiseFloor + params.thresholdDb;

    // ── Peak search with half-reference guard interval ────────────────────────
    const size_t guard = Lb / 2;
    size_t       skip  = 0;

    for (size_t i = 0; i < La; ++i) {
        if (i < skip)              continue;
        if (corrDb[i] < threshold) continue;

        // Find the peak within the next Lb samples
        size_t peakIdx  = i;
        double peakVal  = corrDb[i];
        const size_t end = std::min(La, i + Lb);
        for (size_t j = i + 1; j < end; ++j) {
            if (corrDb[j] > peakVal) { peakVal = corrDb[j]; peakIdx = j; }
        }

        const size_t startSample = peakIdx;
        const size_t endSample   = std::min(La - 1, peakIdx + Lb - 1);
        const double dur =
            static_cast<double>(endSample - startSample) / fs;

        if (dur >= params.minDurationSec) {
            SignalDetection d;
            d.detected     = true;
            d.startSample  = startSample;
            d.endSample    = endSample;
            d.startTimeSec = static_cast<double>(startSample) / fs;
            d.endTimeSec   = static_cast<double>(endSample)   / fs;
            d.powerDb      = peakVal;
            d.noiseFloorDb = noiseFloor;
            d.snrDb        = peakVal - noiseFloor;
            d.confidence   = std::min(1.0, corrMag[peakIdx]);   // normalised ∈ [0,1]
            result.detections.push_back(d);
        }

        skip = peakIdx + guard + 1;
    }

    return result;
}

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
inline DetectionResult detectSignals(
    const std::vector<std::complex<double>>& iq,
    const SignalDetectionParams&             params,
    DetectionMethod                          method,
    const std::vector<std::complex<double>>& reference = {})
{
    switch (method) {
        case DetectionMethod::EnergyTimeDomain:
            return detectEnergyTimeDomain(iq, params);
        case DetectionMethod::EnergySpectral:
            return detectSpectral(iq, params);
        case DetectionMethod::MatchedFilter:
            return detectMatchedFilter(iq, reference, params);
    }
    return {};   // unreachable — silences compiler warning
}

} // namespace SharedMath::DSP

/// @} // DSP_SignalDetection
