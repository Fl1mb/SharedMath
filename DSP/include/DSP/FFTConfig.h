#pragma once

#include <cstddef>

namespace SharedMath::DSP {

/// ── Transform direction ───────────────────────────────────────────────────────
///
///   Forward : X[k] = Σ_{n=0}^{N-1} x[n] · exp(−2πi·nk/N)
///   Inverse : x[n] = Σ_{k=0}^{N-1} X[k] · exp(+2πi·nk/N)   (unnormalized)
///
enum class FFTDirection {
    Forward,
    Inverse
};

/// ── Normalization convention ──────────────────────────────────────────────────
///
///   None    — raw sum, no scaling  (matches FFTW_ESTIMATE, fastest)
///   ByN     — divide by N after inverse  (standard mathematical convention)
///   BySqrtN — divide by √N on both forward and inverse  (unitary / energy-preserving)
///
enum class FFTNorm {
    None,
    ByN,
    BySqrtN
};

/// ── Algorithm selection hint ──────────────────────────────────────────────────
///
///   Auto         — Cooley-Tukey when N is a power of 2, Bluestein otherwise
///   CooleyTukey  — radix-2 DIT; throws if N is not a power of 2
///   Bluestein    — chirp-z; works for any N (incl. prime sizes)
///
enum class FFTAlgorithm {
    Auto,
    CooleyTukey,
    Bluestein
};

/// ── Aggregate plan configuration ─────────────────────────────────────────────
///
/// All fields carry sensible defaults so you can write:
///
///   FFTPlan::create(1024)                              // forward, no norm, auto
///   FFTPlan::create(1024, {FFTDirection::Inverse, FFTNorm::ByN})
///   FFTPlan::create(100,  {FFTDirection::Forward, FFTNorm::None, FFTAlgorithm::Bluestein})
///
struct FFTConfig {
    FFTDirection direction  = FFTDirection::Forward;
    FFTNorm      norm       = FFTNorm::None;
    FFTAlgorithm algorithm  = FFTAlgorithm::Auto;
};

} // namespace SharedMath::DSP
