option(SharedMath_ENABLE_CUDA_DSP
    "Enable CUDA-accelerated DSP primitives (requires CUDAToolkit)"
    OFF)

set(SHAREDMATH_DSP_SOURCES
    src/BurstDetection.cpp
    src/CPUBackend.cpp
    src/Channelization.cpp
    src/Convolution.cpp
    src/FFT.cpp
    src/FFTPlan.cpp
    src/FIR.cpp
    src/FilterDesign.cpp
    src/FilterResponse.cpp
    src/FrequencyCorrection.cpp
    src/Hilbert.cpp
    src/IIR.cpp
    src/PulseShaping.cpp
    src/Resampling.cpp
    src/STFT.cpp
    src/SignalDetection.cpp
    src/SignalEstimation.cpp
    src/SignalGenerator.cpp
    src/Signal.cpp
    src/SignalMetrics.cpp
    src/SignalProcessing.cpp
    src/Spectral.cpp
    src/Streaming.cpp
    src/Waterfall.cpp
    src/Window.cpp
)

add_library(SharedMath_DSP ${SHAREDMATH_DSP_SOURCES})
add_library(SharedMath::DSP ALIAS SharedMath_DSP)

target_include_directories(SharedMath_DSP
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/DSP
)

target_link_libraries(SharedMath_DSP PUBLIC
    SharedMath_Core
    SharedMath_LinearAlgebra
)

target_compile_features(SharedMath_DSP PUBLIC cxx_std_17)

if(SharedMath_ENABLE_CUDA_DSP)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(SharedMath_DSP PUBLIC SHAREDMATH_CUDA_DSP)
    message(STATUS "SharedMath DSP: CUDA backend enabled")
endif()

sharedmath_configure_module(SharedMath_DSP)
