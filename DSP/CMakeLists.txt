include(GenerateExportHeader)

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

generate_export_header(SharedMath_DSP
    EXPORT_MACRO_NAME SHAREDMATH_DSP_EXPORT
    EXPORT_FILE_NAME  ${PROJECT_BINARY_DIR}/include/sharedmath_dsp_export.h
    STATIC_DEFINE     SHAREDMATH_DSP_STATIC_DEFINE
)

target_include_directories(SharedMath_DSP
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/DSP
)

target_link_libraries(SharedMath_DSP
    PUBLIC
        SharedMath_Core
        SharedMath_LinearAlgebra
)

target_compile_definitions(SharedMath_DSP PUBLIC
    $<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:SHAREDMATH_DSP_STATIC_DEFINE>
)

target_compile_features(SharedMath_DSP PUBLIC cxx_std_17)

if(SharedMath_ENABLE_CUDA_DSP)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(SharedMath_DSP PUBLIC SHAREDMATH_CUDA_DSP)
    message(STATUS "SharedMath DSP: CUDA backend enabled")
endif()

sharedmath_configure_module(SharedMath_DSP)


set_target_properties(SharedMath_DSP PROPERTIES
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF
)

if(WIN32 AND BUILD_SHARED_LIBS)
    set_target_properties(SharedMath_DSP PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
endif()
