# DSP (Digital Signal Processing) module — placeholder.
# Planned: FFT, convolution, FIR/IIR filters, windowing functions.

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
