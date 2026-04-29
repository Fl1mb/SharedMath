# DSP (Digital Signal Processing) module — header-only INTERFACE library.
#
# Contents:
#   FFT / RFFT / IFFT         (FFT.h, FFTPlan.h, FFTBackend.h, CPUBackend.h, FFTConfig.h)
#   Windowing functions       (Window.h)
#   Convolution / correlation (Convolution.h)
#   FIR filter design         (FIR.h)
#   IIR / biquad filter design(IIR.h)
#   STFT / ISTFT              (STFT.h)
#   Hilbert transform         (Hilbert.h)
#   Power spectral density    (Spectral.h)
#   FIR frequency response    (FilterResponse.h)
#   Signal generators         (SignalGenerator.h)

add_library(SharedMath_DSP INTERFACE)
add_library(SharedMath::DSP ALIAS SharedMath_DSP)

target_include_directories(SharedMath_DSP INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(SharedMath_DSP INTERFACE
    SharedMath_Core
    SharedMath_LinearAlgebra
)
