#pragma once

/**
 * @file Signal.h
 * @brief Unified signal container and high-level helpers for real and IQ signals.
 *
 * @defgroup DSP_Signal Signal Container
 * @ingroup DSP
 * @{
 *
 * `Signal` is a value-type wrapper around either:
 *  - `std::vector<double>` for real-valued signals,
 *  - `std::vector<std::complex<double>>` for complex baseband / IQ signals, or
 *  - a raw binary file that is decoded lazily on demand.
 *
 * In addition to raw samples, the class stores derived metadata such as
 * duration, RMS level, peak amplitude, average and peak power, PAPR,
 * dominant frequency, occupied bandwidth, and SNR estimate.
 *
 * ### Example
 * @code{.cpp}
 * auto iq = SharedMath::DSP::frequencyShift(rawIq, -12500.0, 48000.0);
 *
 * SharedMath::DSP::Signal s(iq, 48000.0, 0.0, {2048, 0.99});
 * auto cleaned = s.removeDC().normalizePeak(1.0);
 *
 * std::cout << "duration = " << cleaned.durationSec() << " s\n";
 * std::cout << "peak     = " << cleaned.characteristics().peakAmplitude << '\n';
 * std::cout << "cf       = " << cleaned.characteristics().estimatedCenterFrequencyHz << " Hz\n";
 * @endcode
 *
 * @}
 */

#include <complex>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace SharedMath::DSP {

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Storage kind of the signal samples.
 * @ingroup DSP_Signal
 */
enum class SignalStorage {
    Real,      ///< Real-valued time-domain samples.
    ComplexIQ  ///< Complex baseband / IQ samples.
};

/**
 * @brief Physical backing store used by a Signal instance.
 * @ingroup DSP_Signal
 */
enum class SignalSourceKind {
    Memory, ///< Samples are stored directly in in-memory vectors.
    File    ///< Samples are stored in an external raw binary file.
};

/**
 * @brief Raw binary sample encoding used by file-backed signals.
 * @ingroup DSP_Signal
 */
enum class SignalFileFormat {
    RealU8,               ///< One unsigned 8-bit real sample per file byte.
    RealI8,               ///< One signed 8-bit real sample per file byte.
    RealF32,              ///< One 32-bit floating-point real sample.
    RealF64,              ///< One 64-bit floating-point real sample.
    ComplexU8Interleaved, ///< Interleaved unsigned 8-bit IQ: I0,Q0,I1,Q1,...
    ComplexI8Interleaved, ///< Interleaved signed 8-bit IQ: I0,Q0,I1,Q1,...
    ComplexF32Interleaved,///< Interleaved float32 IQ: I0,Q0,I1,Q1,...
    ComplexF64Interleaved ///< Interleaved float64 IQ: I0,Q0,I1,Q1,...
};

/**
 * @brief Parameters used to estimate spectral characteristics of a signal.
 * @ingroup DSP_Signal
 */
struct SignalAnalysisParams {
    size_t fftSize            = 1024; ///< FFT size used by spectral estimators. Must be > 0.
    double occupiedPowerRatio = 0.99; ///< Power fraction used for occupied-bandwidth estimation, in (0, 1].
};

/**
 * @brief Configuration of a file-backed raw signal source.
 *
 * The file is interpreted as a raw binary stream with no header. Samples are
 * decoded according to @ref SignalFileFormat. For integer formats, `scale` and
 * `bias` are applied per decoded component as:
 * @code
 * decoded = scale * (raw + bias)
 * @endcode
 *
 * This is useful, for example, for large unsigned 8-bit ADC dumps where one
 * may set `bias = -128.0` to recenter the data around zero without loading the
 * whole file into memory.
 *
 * @ingroup DSP_Signal
 */
struct SignalFileParams {
    std::string      path;               ///< Path to the raw binary sample file.
    SignalFileFormat format = SignalFileFormat::RealU8; ///< On-disk sample encoding.
    size_t           byteOffset = 0;     ///< Number of bytes to skip before the first sample.
    size_t           sampleCount = 0;    ///< Number of logical samples; 0 means infer from file size.
    double           scale = 1.0;        ///< Multiplicative scale applied after decoding.
    double           bias = 0.0;         ///< Additive bias applied before scaling.
};

/**
 * @brief Derived characteristics stored together with a signal.
 *
 * These values are computed from the currently stored samples and updated
 * either on construction or by calling updateCharacteristics().
 *
 * @ingroup DSP_Signal
 */
struct SignalCharacteristics {
    SignalStorage storage = SignalStorage::Real; ///< Underlying sample storage kind.

    size_t sampleCount = 0; ///< Number of stored samples.
    double sampleRate  = 1.0; ///< Sampling rate in Hz.
    double durationSec = 0.0; ///< Signal duration in seconds.

    std::complex<double> meanValue{0.0, 0.0}; ///< Arithmetic mean of the samples.
    double rms            = 0.0; ///< Root-mean-square amplitude.
    double peakAmplitude  = 0.0; ///< Maximum absolute sample magnitude.
    double energy         = 0.0; ///< Total signal energy: Σ|x[n]|².
    double averagePower   = 0.0; ///< Mean instantaneous power.
    double averagePowerDb = -std::numeric_limits<double>::infinity(); ///< Mean power in dB.
    double peakPower      = 0.0; ///< Maximum instantaneous power.
    double peakPowerDb    = -std::numeric_limits<double>::infinity(); ///< Peak power in dB.
    double paprDb         = 0.0; ///< Peak-to-average power ratio in dB.

    double nominalCenterFrequencyHz    = 0.0; ///< User-assigned nominal RF / IF center frequency in Hz.
    double estimatedCenterFrequencyHz  = 0.0; ///< Estimated spectral center frequency relative to the stored signal band.
    double dominantFrequencyHz         = 0.0; ///< Dominant spectral peak frequency relative to the stored signal band.
    double absoluteDominantFrequencyHz = 0.0; ///< Dominant frequency plus nominalCenterFrequencyHz.
    double occupiedBandwidthHz         = 0.0; ///< Estimated occupied bandwidth in Hz.
    double noiseFloorDb                = -std::numeric_limits<double>::infinity(); ///< Estimated spectral noise floor in dB.
    double snrDb                       = 0.0; ///< Estimated signal-to-noise ratio in dB.
};

/**
 * @brief Value-type container for real or complex signals with cached metadata.
 *
 * The class stores the raw samples and immediately computes the main time-domain
 * and spectral characteristics. Transforming operations such as removeDC(),
 * normalizePeak(), resample(), and frequencyShift() return a new Signal with
 * refreshed characteristics.
 *
 * File-backed instances keep only metadata and file access parameters in
 * memory. Vector-returning accessors and transforming operations are disabled
 * for them to avoid accidental materialization of a very large recording. Use
 * load() to explicitly decode only the required block of samples.
 *
 * @ingroup DSP_Signal
 */
class Signal {
public:
    /// @brief Construct an empty real-valued signal with default analysis parameters.
    Signal();

    /**
     * @brief Construct a real-valued signal.
     * @param samples Time-domain real samples.
     * @param sampleRate Sampling rate in Hz. Must be > 0.
     * @param analysisParams Spectral-analysis settings used to populate characteristics().
     */
    explicit Signal(std::vector<double> samples,
                    double sampleRate = 1.0,
                    SignalAnalysisParams analysisParams = {});

    /**
     * @brief Construct a complex IQ signal.
     * @param samples Complex baseband / IQ samples.
     * @param sampleRate Sampling rate in Hz. Must be > 0.
     * @param nominalCenterFrequencyHz Optional nominal carrier / RF center frequency in Hz.
     * @param analysisParams Spectral-analysis settings used to populate characteristics().
     */
    explicit Signal(std::vector<std::complex<double>> samples,
                    double sampleRate = 1.0,
                    double nominalCenterFrequencyHz = 0.0,
                    SignalAnalysisParams analysisParams = {});

    /**
     * @brief Construct a file-backed signal without loading all samples into memory.
     * @param fileParams Raw file description.
     * @param sampleRate Sampling rate in Hz. Must be > 0.
     * @param nominalCenterFrequencyHz Optional nominal carrier / RF center frequency in Hz.
     * @param analysisParams Spectral-analysis settings used to populate characteristics().
     *
     * The resulting object keeps only metadata and file access parameters. Use
     * load() to materialize a block of samples into an in-memory Signal.
     *
     * @throws std::invalid_argument if the file path is empty, the file cannot
     *         be opened, the raw payload is inconsistent with the selected
     *         format, or sampleRate is not positive.
     */
    explicit Signal(const SignalFileParams& fileParams,
                    double sampleRate,
                    double nominalCenterFrequencyHz = 0.0,
                    SignalAnalysisParams analysisParams = {});

    /// @brief Return the current sample storage kind.
    SignalStorage storage() const noexcept;
    /// @brief Return whether the signal is memory-backed or file-backed.
    SignalSourceKind sourceKind() const noexcept;
    /// @brief Return `true` if the signal reads samples from a file on demand.
    bool isFileBacked() const noexcept;
    /// @brief Return `true` if the signal stores real-valued samples.
    bool isReal() const noexcept;
    /// @brief Return `true` if the signal stores complex IQ samples.
    bool isComplex() const noexcept;
    /// @brief Return `true` if the signal contains no samples.
    bool empty() const noexcept;
    /// @brief Return the logical number of samples in the signal.
    size_t size() const noexcept;

    /// @brief Return the sampling rate in Hz.
    double sampleRate() const noexcept;
    /// @brief Return the signal duration in seconds.
    double durationSec() const noexcept;
    /// @brief Return the user-assigned nominal center frequency in Hz.
    double nominalCenterFrequencyHz() const noexcept;

    /// @brief Return the analysis parameters currently used by updateCharacteristics().
    const SignalAnalysisParams& analysisParams() const noexcept;
    /// @brief Return the cached time-domain and spectral characteristics.
    const SignalCharacteristics& characteristics() const noexcept;
    /// @brief Return the raw-file descriptor for a file-backed signal.
    const SignalFileParams& fileParams() const;

    /**
     * @brief Access the underlying real-valued sample storage.
     * @return Const reference to the real sample vector.
     * @throws std::invalid_argument if the signal stores complex IQ samples or is file-backed.
     */
    const std::vector<double>& realSamples() const;

    /**
     * @brief Access the underlying complex IQ sample storage.
     * @return Const reference to the complex sample vector.
     * @throws std::invalid_argument if the signal stores real-valued samples or is file-backed.
     */
    const std::vector<std::complex<double>>& complexSamples() const;

    /**
     * @brief Materialize a block of samples into an in-memory Signal.
     * @param startSample Zero-based logical starting sample.
     * @param count Number of samples to load. The default loads until the end.
     * @return In-memory Signal containing the decoded block.
     *
     * For memory-backed signals this behaves like slice().
     */
    Signal load(size_t startSample = 0,
                size_t count = std::numeric_limits<size_t>::max()) const;

    /// @brief Return the samples as a complex vector (`imag = 0` for real-valued signals).
    /// @throws std::invalid_argument if the signal is file-backed.
    std::vector<std::complex<double>> asComplex() const;
    /// @brief Return the real component of each stored sample.
    /// @throws std::invalid_argument if the signal is file-backed.
    std::vector<double> realPart() const;
    /// @brief Return the imaginary component of each stored sample (all zeros for real-valued signals).
    /// @throws std::invalid_argument if the signal is file-backed.
    std::vector<double> imagPart() const;
    /// @brief Return the magnitude `|x[n]|` of each sample.
    /// @throws std::invalid_argument if the signal is file-backed.
    std::vector<double> magnitude() const;
    /// @brief Return the instantaneous power `|x[n]|²` of each sample.
    /// @throws std::invalid_argument if the signal is file-backed.
    std::vector<double> power() const;
    /// @brief Return the wrapped sample phase `arg(x[n])` in radians.
    /// @throws std::invalid_argument if the signal is file-backed.
    std::vector<double> phase() const;
    /// @brief Return the time axis `[0, 1/fs, 2/fs, ...]` in seconds.
    /// @throws std::invalid_argument if the signal is file-backed.
    std::vector<double> timeAxis() const;

    /**
     * @brief Recompute all cached characteristics.
     * @param analysisParams Spectral-analysis configuration.
     * @return Const reference to the refreshed characteristic cache.
     *
     * This is useful after changing analysis settings such as `fftSize` or
     * `occupiedPowerRatio`.
     */
    const SignalCharacteristics& updateCharacteristics(
        SignalAnalysisParams analysisParams = {});

    /**
     * @brief Change the sampling rate and recompute cached characteristics.
     * @param sampleRate New sample rate in Hz. Must be > 0.
     * @param analysisParams Spectral-analysis configuration used for recomputation.
     */
    void setSampleRate(double sampleRate,
                       SignalAnalysisParams analysisParams = {});

    /**
     * @brief Set the nominal carrier / RF center frequency without modifying samples.
     * @param nominalCenterFrequencyHz New nominal center frequency in Hz.
     */
    void setNominalCenterFrequencyHz(double nominalCenterFrequencyHz) noexcept;

    /**
     * @brief Extract a contiguous subrange of samples.
     * @param startSample Zero-based starting index.
     * @param count Number of samples to copy.
     * @return New Signal containing the requested slice.
     *
     * If `startSample` lies outside the signal or `count == 0`, an empty signal
     * with the same metadata basis is returned. For file-backed signals the
     * returned object remains lazy and references the same file with an updated
     * offset.
     */
    Signal slice(size_t startSample, size_t count) const;

    /**
     * @brief Remove the DC component from the signal.
     * @return New signal with its mean subtracted.
     * @throws std::invalid_argument if the signal is file-backed.
     */
    Signal removeDC() const;

    /**
     * @brief Scale the signal so that its peak amplitude becomes `targetPeak`.
     * @param targetPeak Target peak magnitude. Must be > 0.
     * @return New normalized signal.
     * @throws std::invalid_argument if `targetPeak <= 0`, the signal is all-zero,
     *         or the signal is file-backed.
     */
    Signal normalizePeak(double targetPeak = 1.0) const;

    /**
     * @brief Scale the signal so that its RMS amplitude becomes `targetRMS`.
     * @param targetRMS Target RMS amplitude. Must be > 0.
     * @return New normalized signal.
     * @throws std::invalid_argument if `targetRMS <= 0`, the signal RMS is zero,
     *         or the signal is file-backed.
     */
    Signal normalizeRMS(double targetRMS = 1.0) const;

    /**
     * @brief Resample the signal to a new sampling rate.
     * @param outputSampleRate Target sampling rate in Hz. Must be > 0.
     * @param analysisParams Spectral-analysis configuration for the output signal.
     * @return New signal resampled to `outputSampleRate`.
     *
     * Real-valued signals are resampled directly. Complex IQ signals are
     * resampled component-wise on their real and imaginary parts.
     *
     * @throws std::invalid_argument if the signal is file-backed.
     */
    Signal resample(double outputSampleRate,
                    SignalAnalysisParams analysisParams = {}) const;

    /**
     * @brief Frequency-shift a complex IQ signal.
     * @param shiftHz Frequency shift in Hz.
     * @param analysisParams Spectral-analysis configuration for the output signal.
     * @return New frequency-shifted signal.
     * @throws std::invalid_argument if called on a real-valued or file-backed signal.
     */
    Signal frequencyShift(double shiftHz,
                          SignalAnalysisParams analysisParams = {}) const;

private:
    SignalSourceKind sourceKind_ = SignalSourceKind::Memory;
    SignalStorage storage_ = SignalStorage::Real;
    std::vector<double> realSamples_;
    std::vector<std::complex<double>> complexSamples_;
    SignalFileParams fileParams_;
    double sampleRate_ = 1.0;
    double nominalCenterFrequencyHz_ = 0.0;
    SignalAnalysisParams analysisParams_{};
    SignalCharacteristics characteristics_{};
};

} // namespace SharedMath::DSP

/// @} // DSP_Signal
