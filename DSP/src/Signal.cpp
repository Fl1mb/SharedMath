/**
 * @file Signal.cpp
 * @brief Implementation of the unified signal container.
 */

#include "Signal.h"

#include "FFT.h"
#include "FrequencyCorrection.h"
#include "Resampling.h"
#include "SignalEstimation.h"
#include "SignalProcessing.h"
#include "Window.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace SharedMath::DSP {

namespace {

constexpr double kTinyPower = 1e-300;
constexpr double kRateScale = 1e6;

struct SpectralSummary {
    double estimatedCenterFrequencyHz = 0.0;
    double dominantFrequencyHz        = 0.0;
    double occupiedBandwidthHz        = 0.0;
    double noiseFloorDb               = -std::numeric_limits<double>::infinity();
    double snrDb                      = 0.0;
};

SignalStorage storageForFormat(SignalFileFormat format)
{
    switch (format) {
        case SignalFileFormat::RealU8:
        case SignalFileFormat::RealI8:
        case SignalFileFormat::RealF32:
        case SignalFileFormat::RealF64:
            return SignalStorage::Real;
        case SignalFileFormat::ComplexU8Interleaved:
        case SignalFileFormat::ComplexI8Interleaved:
        case SignalFileFormat::ComplexF32Interleaved:
        case SignalFileFormat::ComplexF64Interleaved:
            return SignalStorage::ComplexIQ;
    }
    return SignalStorage::Real;
}

size_t bytesPerSample(SignalFileFormat format)
{
    switch (format) {
        case SignalFileFormat::RealU8:
        case SignalFileFormat::RealI8:
            return 1;
        case SignalFileFormat::RealF32:
            return sizeof(float);
        case SignalFileFormat::RealF64:
            return sizeof(double);
        case SignalFileFormat::ComplexU8Interleaved:
        case SignalFileFormat::ComplexI8Interleaved:
            return 2;
        case SignalFileFormat::ComplexF32Interleaved:
            return 2 * sizeof(float);
        case SignalFileFormat::ComplexF64Interleaved:
            return 2 * sizeof(double);
    }
    return 1;
}

void validateSampleRate(double sampleRate, const char* fn)
{
    if (sampleRate <= 0.0)
        throw std::invalid_argument(std::string(fn) + ": sampleRate must be > 0");
}

void validateAnalysisParams(const SignalAnalysisParams& params, const char* fn)
{
    if (params.fftSize == 0)
        throw std::invalid_argument(std::string(fn) + ": fftSize must be > 0");
    if (params.occupiedPowerRatio <= 0.0 || params.occupiedPowerRatio > 1.0)
        throw std::invalid_argument(
            std::string(fn) + ": occupiedPowerRatio must be in (0, 1]");
}

void validateFileParams(const SignalFileParams& params, const char* fn)
{
    if (params.path.empty())
        throw std::invalid_argument(std::string(fn) + ": file path must not be empty");
    if (params.scale == 0.0)
        throw std::invalid_argument(std::string(fn) + ": scale must not be 0");
}

size_t inferSampleCountFromFile(const SignalFileParams& params)
{
    std::ifstream stream(params.path, std::ios::binary | std::ios::ate);
    if (!stream)
        throw std::invalid_argument("Signal file open failed: " + params.path);

    const auto endPos = stream.tellg();
    if (endPos < 0)
        throw std::invalid_argument("Signal file size query failed: " + params.path);

    const size_t fileSize = static_cast<size_t>(endPos);
    if (params.byteOffset > fileSize)
        throw std::invalid_argument("Signal file byteOffset exceeds file size");

    const size_t sampleBytes = bytesPerSample(params.format);
    const size_t payloadSize = fileSize - params.byteOffset;
    if (payloadSize % sampleBytes != 0)
        throw std::invalid_argument(
            "Signal file payload size is not aligned to the selected sample format");
    return payloadSize / sampleBytes;
}

template<typename T>
T readPOD(const unsigned char* data)
{
    T value{};
    std::memcpy(&value, data, sizeof(T));
    return value;
}

double applyFileScaling(double raw, const SignalFileParams& params)
{
    return params.scale * (raw + params.bias);
}

double decodeRealSample(const unsigned char* data, const SignalFileParams& params)
{
    switch (params.format) {
        case SignalFileFormat::RealU8:
            return applyFileScaling(static_cast<double>(data[0]), params);
        case SignalFileFormat::RealI8:
            return applyFileScaling(static_cast<double>(readPOD<int8_t>(data)), params);
        case SignalFileFormat::RealF32:
            return applyFileScaling(static_cast<double>(readPOD<float>(data)), params);
        case SignalFileFormat::RealF64:
            return applyFileScaling(readPOD<double>(data), params);
        default:
            throw std::invalid_argument("decodeRealSample: file format is not real-valued");
    }
}

std::complex<double> decodeComplexSample(const unsigned char* data,
                                         const SignalFileParams& params)
{
    switch (params.format) {
        case SignalFileFormat::ComplexU8Interleaved:
            return {
                applyFileScaling(static_cast<double>(data[0]), params),
                applyFileScaling(static_cast<double>(data[1]), params)
            };
        case SignalFileFormat::ComplexI8Interleaved:
            return {
                applyFileScaling(static_cast<double>(readPOD<int8_t>(data)), params),
                applyFileScaling(static_cast<double>(readPOD<int8_t>(data + 1)), params)
            };
        case SignalFileFormat::ComplexF32Interleaved:
            return {
                applyFileScaling(static_cast<double>(readPOD<float>(data)), params),
                applyFileScaling(static_cast<double>(readPOD<float>(data + sizeof(float))), params)
            };
        case SignalFileFormat::ComplexF64Interleaved:
            return {
                applyFileScaling(readPOD<double>(data), params),
                applyFileScaling(readPOD<double>(data + sizeof(double)), params)
            };
        default:
            throw std::invalid_argument("decodeComplexSample: file format is not complex-valued");
    }
}

std::vector<unsigned char> readFileBytes(const SignalFileParams& params,
                                         size_t startSample,
                                         size_t count)
{
    const size_t sampleBytes = bytesPerSample(params.format);
    const size_t byteCount = count * sampleBytes;
    const size_t byteStart = params.byteOffset + startSample * sampleBytes;

    std::ifstream stream(params.path, std::ios::binary);
    if (!stream)
        throw std::invalid_argument("Signal file open failed: " + params.path);

    stream.seekg(static_cast<std::streamoff>(byteStart), std::ios::beg);
    if (!stream)
        throw std::invalid_argument("Signal file seek failed: " + params.path);

    std::vector<unsigned char> bytes(byteCount);
    if (byteCount == 0) return bytes;

    stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(byteCount));
    if (stream.gcount() != static_cast<std::streamsize>(byteCount))
        throw std::invalid_argument("Signal file read failed: " + params.path);

    return bytes;
}

std::vector<double> readRealSamplesFromFile(const SignalFileParams& params,
                                            size_t startSample,
                                            size_t count)
{
    if (count == 0) return {};

    const size_t sampleBytes = bytesPerSample(params.format);
    const auto bytes = readFileBytes(params, startSample, count);

    std::vector<double> out(count);
    for (size_t i = 0; i < count; ++i)
        out[i] = decodeRealSample(bytes.data() + i * sampleBytes, params);
    return out;
}

std::vector<std::complex<double>> readComplexSamplesFromFile(
    const SignalFileParams& params,
    size_t startSample,
    size_t count)
{
    if (count == 0) return {};

    const size_t sampleBytes = bytesPerSample(params.format);
    const auto bytes = readFileBytes(params, startSample, count);

    std::vector<std::complex<double>> out(count);
    for (size_t i = 0; i < count; ++i)
        out[i] = decodeComplexSample(bytes.data() + i * sampleBytes, params);
    return out;
}

void ensureMemoryBacked(const class Signal& signal, const char* fn);

double medianValue(std::vector<double> values)
{
    if (values.empty()) return -std::numeric_limits<double>::infinity();
    const size_t n = values.size();
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(n / 2),
                     values.end());
    if (n % 2 == 1) return values[n / 2];
    const double hi = values[n / 2];
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(n / 2 - 1),
                     values.end());
    return 0.5 * (values[n / 2 - 1] + hi);
}

std::vector<std::complex<double>> toComplexSamples(const std::vector<double>& samples)
{
    std::vector<std::complex<double>> out(samples.size());
    for (size_t i = 0; i < samples.size(); ++i)
        out[i] = std::complex<double>(samples[i], 0.0);
    return out;
}

std::complex<double> meanComplex(const std::vector<std::complex<double>>& samples)
{
    if (samples.empty()) return {0.0, 0.0};
    std::complex<double> sum{0.0, 0.0};
    for (const auto& sample : samples) sum += sample;
    return sum / static_cast<double>(samples.size());
}

double rmsComplex(const std::vector<std::complex<double>>& samples)
{
    if (samples.empty()) return 0.0;
    double energy = 0.0;
    for (const auto& sample : samples) energy += std::norm(sample);
    return std::sqrt(energy / static_cast<double>(samples.size()));
}

double peakAmplitudeComplex(const std::vector<std::complex<double>>& samples)
{
    double peak = 0.0;
    for (const auto& sample : samples) peak = std::max(peak, std::abs(sample));
    return peak;
}

std::vector<std::complex<double>> removeDCComplex(
    const std::vector<std::complex<double>>& samples)
{
    if (samples.empty()) return {};
    const std::complex<double> mu = meanComplex(samples);
    std::vector<std::complex<double>> out(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) out[i] = samples[i] - mu;
    return out;
}

std::vector<std::complex<double>> normalizePeakComplex(
    const std::vector<std::complex<double>>& samples,
    double targetPeak)
{
    if (samples.empty()) return {};
    if (targetPeak <= 0.0)
        throw std::invalid_argument("normalizePeak: targetPeak must be > 0");

    const double peak = peakAmplitudeComplex(samples);
    if (peak <= 0.0)
        throw std::invalid_argument("normalizePeak: signal is all-zero");

    const double scale = targetPeak / peak;
    std::vector<std::complex<double>> out(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) out[i] = samples[i] * scale;
    return out;
}

std::vector<std::complex<double>> normalizeRMSComplex(
    const std::vector<std::complex<double>>& samples,
    double targetRMS)
{
    if (samples.empty()) return {};
    if (targetRMS <= 0.0)
        throw std::invalid_argument("normalizeRMS: targetRMS must be > 0");

    const double value = rmsComplex(samples);
    if (value <= 0.0)
        throw std::invalid_argument("normalizeRMS: signal RMS is zero");

    const double scale = targetRMS / value;
    std::vector<std::complex<double>> out(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) out[i] = samples[i] * scale;
    return out;
}

SpectralSummary analyzeRealSpectrum(const std::vector<double>& samples,
                                    double sampleRate,
                                    const SignalAnalysisParams& params)
{
    SpectralSummary summary;
    if (samples.empty()) return summary;

    const size_t fftSize = std::max<size_t>(2, params.fftSize);
    const size_t winLen  = std::min(samples.size(), fftSize);

    auto win = windowHann(winLen, /*symmetric=*/false);
    double winSumSq = 0.0;
    for (double w : win) winSumSq += w * w;

    std::vector<double> frame(fftSize, 0.0);
    for (size_t i = 0; i < winLen; ++i)
        frame[i] = samples[i] * win[i];

    auto spectrum = rfft(frame, FFTNorm::None);
    auto freqs    = rfftFrequencies(fftSize, sampleRate);

    std::vector<double> power(spectrum.size(), 0.0);
    const double scale = 1.0 / std::max(winSumSq, kTinyPower);
    for (size_t k = 0; k < spectrum.size(); ++k) {
        power[k] = std::norm(spectrum[k]) * scale;
        if (k > 0 && k + 1 < spectrum.size()) power[k] *= 2.0;
    }

    std::vector<double> powerDb(power.size());
    for (size_t k = 0; k < power.size(); ++k)
        powerDb[k] = 10.0 * std::log10(std::max(power[k], kTinyPower));

    summary.noiseFloorDb = medianValue(powerDb);

    const double totalPower = std::accumulate(power.begin(), power.end(), 0.0);
    if (totalPower <= kTinyPower) return summary;

    const auto peakIt = std::max_element(power.begin(), power.end());
    const size_t peakIndex =
        static_cast<size_t>(peakIt - power.begin());

    summary.dominantFrequencyHz = freqs[peakIndex];
    summary.snrDb = powerDb[peakIndex] - summary.noiseFloorDb;

    double weightedFrequency = 0.0;
    for (size_t k = 0; k < power.size(); ++k)
        weightedFrequency += freqs[k] * power[k];
    summary.estimatedCenterFrequencyHz = weightedFrequency / totalPower;

    std::vector<size_t> indices(power.size());
    std::iota(indices.begin(), indices.end(), 0u);
    std::sort(indices.begin(), indices.end(),
              [&](size_t lhs, size_t rhs) { return power[lhs] > power[rhs]; });

    const double target = totalPower * params.occupiedPowerRatio;
    double accum = 0.0;
    size_t minBin = power.size();
    size_t maxBin = 0;
    for (size_t index : indices) {
        accum += power[index];
        minBin = std::min(minBin, index);
        maxBin = std::max(maxBin, index);
        if (accum >= target) break;
    }

    if (minBin <= maxBin) {
        const double binHz =
            (freqs.size() > 1) ? (freqs[1] - freqs[0]) : 0.0;
        summary.occupiedBandwidthHz =
            freqs[maxBin] - freqs[minBin] + binHz;
    }

    return summary;
}

std::pair<size_t, size_t> buildResampleRatio(double inputRate,
                                             double outputRate)
{
    validateSampleRate(inputRate, "Signal::resample");
    validateSampleRate(outputRate, "Signal::resample");

    const auto scaledInput =
        static_cast<unsigned long long>(std::llround(inputRate * kRateScale));
    const auto scaledOutput =
        static_cast<unsigned long long>(std::llround(outputRate * kRateScale));

    if (scaledInput == 0 || scaledOutput == 0)
        throw std::invalid_argument("Signal::resample: invalid sample-rate ratio");

    const auto gcdValue = std::gcd(scaledInput, scaledOutput);
    return {
        static_cast<size_t>(scaledOutput / gcdValue),
        static_cast<size_t>(scaledInput / gcdValue)
    };
}

Signal makeRealSignalWithMetadata(std::vector<double> samples,
                                  double sampleRate,
                                  double nominalCenterFrequencyHz,
                                  const SignalAnalysisParams& analysisParams)
{
    Signal signal(std::move(samples), sampleRate, analysisParams);
    signal.setNominalCenterFrequencyHz(nominalCenterFrequencyHz);
    return signal;
}

Signal makeEmptySignal(SignalStorage storage,
                       double sampleRate,
                       double nominalCenterFrequencyHz,
                       const SignalAnalysisParams& analysisParams)
{
    if (storage == SignalStorage::Real) {
        return makeRealSignalWithMetadata(
            {},
            sampleRate,
            nominalCenterFrequencyHz,
            analysisParams);
    }

    return Signal(std::vector<std::complex<double>>{},
                  sampleRate,
                  nominalCenterFrequencyHz,
                  analysisParams);
}

} // namespace

Signal::Signal()
{
    updateCharacteristics();
}

Signal::Signal(std::vector<double> samples,
               double sampleRate,
               SignalAnalysisParams analysisParams)
    : storage_(SignalStorage::Real),
      realSamples_(std::move(samples)),
      sampleRate_(sampleRate),
      analysisParams_(analysisParams)
{
    validateSampleRate(sampleRate_, "Signal");
    updateCharacteristics(analysisParams_);
}

Signal::Signal(std::vector<std::complex<double>> samples,
               double sampleRate,
               double nominalCenterFrequencyHz,
               SignalAnalysisParams analysisParams)
    : sourceKind_(SignalSourceKind::Memory),
      storage_(SignalStorage::ComplexIQ),
      complexSamples_(std::move(samples)),
      sampleRate_(sampleRate),
      nominalCenterFrequencyHz_(nominalCenterFrequencyHz),
      analysisParams_(analysisParams)
{
    validateSampleRate(sampleRate_, "Signal");
    updateCharacteristics(analysisParams_);
}

Signal::Signal(const SignalFileParams& fileParams,
               double sampleRate,
               double nominalCenterFrequencyHz,
               SignalAnalysisParams analysisParams)
    : sourceKind_(SignalSourceKind::File),
      storage_(storageForFormat(fileParams.format)),
      fileParams_(fileParams),
      sampleRate_(sampleRate),
      nominalCenterFrequencyHz_(nominalCenterFrequencyHz),
      analysisParams_(analysisParams)
{
    validateSampleRate(sampleRate_, "Signal");
    validateFileParams(fileParams_, "Signal");
    if (fileParams_.sampleCount == 0) {
        fileParams_.sampleCount = inferSampleCountFromFile(fileParams_);
    } else {
        const size_t inferred = inferSampleCountFromFile(fileParams_);
        if (fileParams_.sampleCount > inferred)
            throw std::invalid_argument("Signal: sampleCount exceeds file capacity");
    }
    updateCharacteristics(analysisParams_);
}

SignalSourceKind Signal::sourceKind() const noexcept
{
    return sourceKind_;
}

bool Signal::isFileBacked() const noexcept
{
    return sourceKind_ == SignalSourceKind::File;
}

namespace {

void ensureMemoryBacked(const Signal& signal, const char* fn)
{
    if (signal.isFileBacked())
        throw std::invalid_argument(std::string(fn) +
                                    ": operation is not available for file-backed signals; use load()");
}
} // namespace

SignalStorage Signal::storage() const noexcept
{
    return storage_;
}

bool Signal::isReal() const noexcept
{
    return storage_ == SignalStorage::Real;
}

bool Signal::isComplex() const noexcept
{
    return storage_ == SignalStorage::ComplexIQ;
}

bool Signal::empty() const noexcept
{
    return size() == 0;
}

size_t Signal::size() const noexcept
{
    if (isFileBacked()) return fileParams_.sampleCount;
    return isReal() ? realSamples_.size() : complexSamples_.size();
}

double Signal::sampleRate() const noexcept
{
    return sampleRate_;
}

double Signal::durationSec() const noexcept
{
    return characteristics_.durationSec;
}

double Signal::nominalCenterFrequencyHz() const noexcept
{
    return nominalCenterFrequencyHz_;
}

const SignalAnalysisParams& Signal::analysisParams() const noexcept
{
    return analysisParams_;
}

const SignalCharacteristics& Signal::characteristics() const noexcept
{
    return characteristics_;
}

const SignalFileParams& Signal::fileParams() const
{
    if (!isFileBacked())
        throw std::invalid_argument("Signal::fileParams: signal is memory-backed");
    return fileParams_;
}

const std::vector<double>& Signal::realSamples() const
{
    ensureMemoryBacked(*this, "Signal::realSamples");
    if (!isReal())
        throw std::invalid_argument("Signal::realSamples: signal stores complex IQ data");
    return realSamples_;
}

const std::vector<std::complex<double>>& Signal::complexSamples() const
{
    ensureMemoryBacked(*this, "Signal::complexSamples");
    if (!isComplex())
        throw std::invalid_argument("Signal::complexSamples: signal stores real samples");
    return complexSamples_;
}

Signal Signal::load(size_t startSample, size_t count) const
{
    if (!isFileBacked()) return slice(startSample, count);

    if (startSample >= size() || count == 0) {
        return makeEmptySignal(
            storage_,
            sampleRate_,
            nominalCenterFrequencyHz_,
            analysisParams_);
    }

    const size_t actualCount =
        std::min(count, size() - startSample);

    if (isReal()) {
        return makeRealSignalWithMetadata(
            readRealSamplesFromFile(fileParams_, startSample, actualCount),
            sampleRate_,
            nominalCenterFrequencyHz_,
            analysisParams_);
    }

    return Signal(readComplexSamplesFromFile(fileParams_, startSample, actualCount),
                  sampleRate_,
                  nominalCenterFrequencyHz_,
                  analysisParams_);
}

std::vector<std::complex<double>> Signal::asComplex() const
{
    ensureMemoryBacked(*this, "Signal::asComplex");
    return isComplex() ? complexSamples_ : toComplexSamples(realSamples_);
}

std::vector<double> Signal::realPart() const
{
    ensureMemoryBacked(*this, "Signal::realPart");
    if (isReal()) return realSamples_;

    std::vector<double> out(complexSamples_.size());
    for (size_t i = 0; i < complexSamples_.size(); ++i)
        out[i] = complexSamples_[i].real();
    return out;
}

std::vector<double> Signal::imagPart() const
{
    ensureMemoryBacked(*this, "Signal::imagPart");
    std::vector<double> out(size(), 0.0);
    if (isReal()) return out;

    for (size_t i = 0; i < complexSamples_.size(); ++i)
        out[i] = complexSamples_[i].imag();
    return out;
}

std::vector<double> Signal::magnitude() const
{
    ensureMemoryBacked(*this, "Signal::magnitude");
    std::vector<double> out(size(), 0.0);
    if (isReal()) {
        for (size_t i = 0; i < realSamples_.size(); ++i)
            out[i] = std::abs(realSamples_[i]);
        return out;
    }

    for (size_t i = 0; i < complexSamples_.size(); ++i)
        out[i] = std::abs(complexSamples_[i]);
    return out;
}

std::vector<double> Signal::power() const
{
    ensureMemoryBacked(*this, "Signal::power");
    std::vector<double> out(size(), 0.0);
    if (isReal()) {
        for (size_t i = 0; i < realSamples_.size(); ++i)
            out[i] = realSamples_[i] * realSamples_[i];
        return out;
    }

    for (size_t i = 0; i < complexSamples_.size(); ++i)
        out[i] = std::norm(complexSamples_[i]);
    return out;
}

std::vector<double> Signal::phase() const
{
    ensureMemoryBacked(*this, "Signal::phase");
    std::vector<double> out(size(), 0.0);
    if (isReal()) {
        for (size_t i = 0; i < realSamples_.size(); ++i)
            out[i] = std::arg(std::complex<double>(realSamples_[i], 0.0));
        return out;
    }

    for (size_t i = 0; i < complexSamples_.size(); ++i)
        out[i] = std::arg(complexSamples_[i]);
    return out;
}

std::vector<double> Signal::timeAxis() const
{
    ensureMemoryBacked(*this, "Signal::timeAxis");
    std::vector<double> axis(size());
    if (axis.empty()) return axis;

    const double dt = 1.0 / sampleRate_;
    for (size_t i = 0; i < axis.size(); ++i)
        axis[i] = static_cast<double>(i) * dt;
    return axis;
}

const SignalCharacteristics& Signal::updateCharacteristics(
    SignalAnalysisParams analysisParams)
{
    validateSampleRate(sampleRate_, "Signal::updateCharacteristics");
    validateAnalysisParams(analysisParams, "Signal::updateCharacteristics");
    analysisParams_ = analysisParams;

    SignalCharacteristics updated;
    updated.storage                  = storage_;
    updated.sampleCount              = size();
    updated.sampleRate               = sampleRate_;
    updated.durationSec              = static_cast<double>(size()) / sampleRate_;
    updated.nominalCenterFrequencyHz = nominalCenterFrequencyHz_;

    if (empty()) {
        updated.absoluteDominantFrequencyHz = nominalCenterFrequencyHz_;
        characteristics_ = updated;
        return characteristics_;
    }

    if (isFileBacked()) {
        const size_t chunkSize = 65536;

        if (isReal()) {
            double sum = 0.0;
            for (size_t start = 0; start < size(); start += chunkSize) {
                const size_t count = std::min(chunkSize, size() - start);
                const auto block = readRealSamplesFromFile(fileParams_, start, count);
                for (double sample : block) {
                    sum += sample;
                    updated.energy += sample * sample;
                    updated.peakAmplitude = std::max(updated.peakAmplitude, std::abs(sample));
                }
            }

            updated.meanValue = std::complex<double>(
                sum / static_cast<double>(size()), 0.0);
            updated.averagePower = updated.energy / static_cast<double>(size());
            updated.rms = std::sqrt(updated.averagePower);
            updated.averagePowerDb = 10.0 * std::log10(std::max(updated.averagePower, kTinyPower));
            updated.peakPower = updated.peakAmplitude * updated.peakAmplitude;
            updated.peakPowerDb = 10.0 * std::log10(std::max(updated.peakPower, kTinyPower));
            updated.paprDb = updated.peakPowerDb - updated.averagePowerDb;

            const auto block = readRealSamplesFromFile(
                fileParams_, 0, std::min(size(), std::max<size_t>(size_t{2}, analysisParams_.fftSize)));
            const auto spectral = analyzeRealSpectrum(block, sampleRate_, analysisParams_);
            updated.estimatedCenterFrequencyHz = spectral.estimatedCenterFrequencyHz;
            updated.dominantFrequencyHz = spectral.dominantFrequencyHz;
            updated.absoluteDominantFrequencyHz =
                nominalCenterFrequencyHz_ + updated.dominantFrequencyHz;
            updated.occupiedBandwidthHz = spectral.occupiedBandwidthHz;
            updated.noiseFloorDb = spectral.noiseFloorDb;
            updated.snrDb = spectral.snrDb;
        } else {
            std::complex<double> sum{0.0, 0.0};
            for (size_t start = 0; start < size(); start += chunkSize) {
                const size_t count = std::min(chunkSize, size() - start);
                const auto block = readComplexSamplesFromFile(fileParams_, start, count);
                for (const auto& sample : block) {
                    sum += sample;
                    updated.energy += std::norm(sample);
                    updated.peakAmplitude = std::max(updated.peakAmplitude, std::abs(sample));
                }
            }

            updated.meanValue = sum / static_cast<double>(size());
            updated.averagePower = updated.energy / static_cast<double>(size());
            updated.rms = std::sqrt(updated.averagePower);
            updated.averagePowerDb = 10.0 * std::log10(std::max(updated.averagePower, kTinyPower));
            updated.peakPower = updated.peakAmplitude * updated.peakAmplitude;
            updated.peakPowerDb = 10.0 * std::log10(std::max(updated.peakPower, kTinyPower));
            updated.paprDb = updated.peakPowerDb - updated.averagePowerDb;

            const auto block = readComplexSamplesFromFile(
                fileParams_, 0, std::min(size(), std::max<size_t>(size_t{2}, analysisParams_.fftSize)));

            SignalEstimationParams estimationParams;
            estimationParams.sampleRate = sampleRate_;
            estimationParams.fftSize = analysisParams_.fftSize;
            estimationParams.occupiedPowerRatio = analysisParams_.occupiedPowerRatio;

            const auto estimate = estimateSignal(block, estimationParams);
            updated.estimatedCenterFrequencyHz = estimate.centerFrequencyHz;
            updated.dominantFrequencyHz =
                estimateFrequencyOffsetFromPeak(block, sampleRate_, analysisParams_.fftSize);
            updated.absoluteDominantFrequencyHz =
                nominalCenterFrequencyHz_ + updated.dominantFrequencyHz;
            updated.occupiedBandwidthHz = estimate.occupiedBandwidthHz;
            updated.noiseFloorDb = estimate.noiseFloorDb;
            updated.snrDb = estimate.snrDb;
        }
    } else if (isReal()) {
        updated.meanValue = std::complex<double>(mean(realSamples_), 0.0);
        updated.rms = DSP::rms(realSamples_);
        updated.peakAmplitude = peakAbs(realSamples_);

        for (double sample : realSamples_)
            updated.energy += sample * sample;

        updated.averagePower   = updated.energy / static_cast<double>(realSamples_.size());
        updated.averagePowerDb = 10.0 * std::log10(std::max(updated.averagePower, kTinyPower));
        updated.peakPower      = updated.peakAmplitude * updated.peakAmplitude;
        updated.peakPowerDb    = 10.0 * std::log10(std::max(updated.peakPower, kTinyPower));
        updated.paprDb         = updated.peakPowerDb - updated.averagePowerDb;

        const auto spectral = analyzeRealSpectrum(realSamples_, sampleRate_, analysisParams_);
        updated.estimatedCenterFrequencyHz = spectral.estimatedCenterFrequencyHz;
        updated.dominantFrequencyHz        = spectral.dominantFrequencyHz;
        updated.absoluteDominantFrequencyHz =
            nominalCenterFrequencyHz_ + updated.dominantFrequencyHz;
        updated.occupiedBandwidthHz = spectral.occupiedBandwidthHz;
        updated.noiseFloorDb        = spectral.noiseFloorDb;
        updated.snrDb               = spectral.snrDb;
    } else {
        updated.meanValue     = meanComplex(complexSamples_);
        updated.rms           = rmsComplex(complexSamples_);
        updated.peakAmplitude = peakAmplitudeComplex(complexSamples_);

        for (const auto& sample : complexSamples_)
            updated.energy += std::norm(sample);

        updated.averagePower   = updated.energy / static_cast<double>(complexSamples_.size());
        updated.averagePowerDb = 10.0 * std::log10(std::max(updated.averagePower, kTinyPower));
        updated.peakPower      = updated.peakAmplitude * updated.peakAmplitude;
        updated.peakPowerDb    = 10.0 * std::log10(std::max(updated.peakPower, kTinyPower));
        updated.paprDb         = updated.peakPowerDb - updated.averagePowerDb;

        SignalEstimationParams estimationParams;
        estimationParams.sampleRate         = sampleRate_;
        estimationParams.fftSize            = analysisParams_.fftSize;
        estimationParams.occupiedPowerRatio = analysisParams_.occupiedPowerRatio;

        const auto estimate = estimateSignal(complexSamples_, estimationParams);
        updated.estimatedCenterFrequencyHz = estimate.centerFrequencyHz;
        updated.dominantFrequencyHz =
            estimateFrequencyOffsetFromPeak(complexSamples_, sampleRate_, analysisParams_.fftSize);
        updated.absoluteDominantFrequencyHz =
            nominalCenterFrequencyHz_ + updated.dominantFrequencyHz;
        updated.occupiedBandwidthHz = estimate.occupiedBandwidthHz;
        updated.noiseFloorDb        = estimate.noiseFloorDb;
        updated.snrDb               = estimate.snrDb;
    }

    characteristics_ = updated;
    return characteristics_;
}

void Signal::setSampleRate(double sampleRate,
                           SignalAnalysisParams analysisParams)
{
    validateSampleRate(sampleRate, "Signal::setSampleRate");
    sampleRate_ = sampleRate;
    updateCharacteristics(analysisParams);
}

void Signal::setNominalCenterFrequencyHz(double nominalCenterFrequencyHz) noexcept
{
    nominalCenterFrequencyHz_ = nominalCenterFrequencyHz;
    characteristics_.nominalCenterFrequencyHz = nominalCenterFrequencyHz_;
    characteristics_.absoluteDominantFrequencyHz =
        nominalCenterFrequencyHz_ + characteristics_.dominantFrequencyHz;
}

Signal Signal::slice(size_t startSample, size_t count) const
{
    if (startSample >= size() || count == 0) {
        return makeEmptySignal(
            storage_,
            sampleRate_,
            nominalCenterFrequencyHz_,
            analysisParams_);
    }

    const size_t endSample = std::min(size(), startSample + count);
    if (isFileBacked()) {
        SignalFileParams sliceParams = fileParams_;
        sliceParams.byteOffset += startSample * bytesPerSample(fileParams_.format);
        sliceParams.sampleCount = endSample - startSample;
        return Signal(sliceParams,
                      sampleRate_,
                      nominalCenterFrequencyHz_,
                      analysisParams_);
    }

    if (isReal()) {
        return makeRealSignalWithMetadata(
            std::vector<double>(
                realSamples_.begin() + static_cast<std::ptrdiff_t>(startSample),
                realSamples_.begin() + static_cast<std::ptrdiff_t>(endSample)),
            sampleRate_,
            nominalCenterFrequencyHz_,
            analysisParams_);
    }

    return Signal(
        std::vector<std::complex<double>>(
            complexSamples_.begin() + static_cast<std::ptrdiff_t>(startSample),
            complexSamples_.begin() + static_cast<std::ptrdiff_t>(endSample)),
        sampleRate_,
        nominalCenterFrequencyHz_,
        analysisParams_);
}

Signal Signal::removeDC() const
{
    ensureMemoryBacked(*this, "Signal::removeDC");
    if (isReal())
        return makeRealSignalWithMetadata(
            DSP::removeDC(realSamples_),
            sampleRate_,
            nominalCenterFrequencyHz_,
            analysisParams_);

    return Signal(removeDCComplex(complexSamples_),
                  sampleRate_,
                  nominalCenterFrequencyHz_,
                  analysisParams_);
}

Signal Signal::normalizePeak(double targetPeak) const
{
    ensureMemoryBacked(*this, "Signal::normalizePeak");
    if (isReal())
        return makeRealSignalWithMetadata(
            DSP::normalizePeak(realSamples_, targetPeak),
            sampleRate_,
            nominalCenterFrequencyHz_,
            analysisParams_);

    return Signal(normalizePeakComplex(complexSamples_, targetPeak),
                  sampleRate_,
                  nominalCenterFrequencyHz_,
                  analysisParams_);
}

Signal Signal::normalizeRMS(double targetRMS) const
{
    ensureMemoryBacked(*this, "Signal::normalizeRMS");
    if (isReal())
        return makeRealSignalWithMetadata(
            DSP::normalizeRMS(realSamples_, targetRMS),
            sampleRate_,
            nominalCenterFrequencyHz_,
            analysisParams_);

    return Signal(normalizeRMSComplex(complexSamples_, targetRMS),
                  sampleRate_,
                  nominalCenterFrequencyHz_,
                  analysisParams_);
}

Signal Signal::resample(double outputSampleRate,
                        SignalAnalysisParams analysisParams) const
{
    ensureMemoryBacked(*this, "Signal::resample");
    validateSampleRate(outputSampleRate, "Signal::resample");
    if (std::abs(outputSampleRate - sampleRate_) < 1e-12 * std::max(sampleRate_, 1.0))
        return *this;

    const auto [L, M] = buildResampleRatio(sampleRate_, outputSampleRate);

    if (isReal()) {
        return makeRealSignalWithMetadata(
            resamplePolyphaseAligned(realSamples_, L, M),
            outputSampleRate,
            nominalCenterFrequencyHz_,
            analysisParams);
    }

    std::vector<double> re = realPart();
    std::vector<double> im = imagPart();
    re = resamplePolyphaseAligned(re, L, M);
    im = resamplePolyphaseAligned(im, L, M);

    std::vector<std::complex<double>> out(std::min(re.size(), im.size()));
    for (size_t i = 0; i < out.size(); ++i)
        out[i] = std::complex<double>(re[i], im[i]);

    return Signal(std::move(out),
                  outputSampleRate,
                  nominalCenterFrequencyHz_,
                  analysisParams);
}

Signal Signal::frequencyShift(double shiftHz,
                              SignalAnalysisParams analysisParams) const
{
    ensureMemoryBacked(*this, "Signal::frequencyShift");
    if (!isComplex())
        throw std::invalid_argument(
            "Signal::frequencyShift: only complex IQ signals can be frequency shifted");

    return Signal(DSP::frequencyShift(complexSamples_, shiftHz, sampleRate_),
                  sampleRate_,
                  nominalCenterFrequencyHz_,
                  analysisParams);
}

} // namespace SharedMath::DSP
