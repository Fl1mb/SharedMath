# DSP (Digital Signal Processing) module — placeholder.
# Planned: FFT, convolution, FIR/IIR filters, windowing functions.
# Add source files and compiled-library rules here when implementing.

add_library(SharedMath_DSP INTERFACE)
add_library(SharedMath::DSP ALIAS SharedMath_DSP)

target_link_libraries(SharedMath_DSP INTERFACE
    SharedMath_Core
    SharedMath_LinearAlgebra
)
