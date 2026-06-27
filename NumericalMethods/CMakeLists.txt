include(GenerateExportHeader)

# ── Optional CUDA support ──────────────────────────────────────────────────
find_package(CUDAToolkit QUIET)
option(SHAREDMATH_ENABLE_CUDA "Enable CUDA acceleration for NumericalMethods" ${CUDAToolkit_FOUND})

if(SHAREDMATH_ENABLE_CUDA)
    if(NOT CUDAToolkit_FOUND)
        find_package(CUDAToolkit REQUIRED)
    endif()
    enable_language(CUDA)
    message(STATUS "SharedMath NumericalMethods: CUDA support enabled (toolkit: ${CUDAToolkit_VERSION})")
endif()

# ── Library sources ────────────────────────────────────────────────────────
set(SHAREDMATH_NUM_METHODS_SOURCES
    src/Differentiation.cpp
    src/Integration.cpp
    src/ODE.cpp
    src/IntegralEquations.cpp
    src/SLAE.cpp
    src/NonlinearEquations.cpp
)

if(SHAREDMATH_ENABLE_CUDA)
    list(APPEND SHAREDMATH_NUM_METHODS_SOURCES
        src/NumericalMethodsCUDA.cu
        src/NumericalMethodsSLAECUDA.cu
        src/NumericalMethodsNLECUDA.cu
    )
endif()

add_library(SharedMath_NumericalMethods ${SHAREDMATH_NUM_METHODS_SOURCES})
add_library(SharedMath::NumericalMethods ALIAS SharedMath_NumericalMethods)

generate_export_header(SharedMath_NumericalMethods
    EXPORT_MACRO_NAME SHAREDMATH_NUMERICALMETHODS_EXPORT
    EXPORT_FILE_NAME  ${PROJECT_BINARY_DIR}/include/sharedmath_numericalmethods_export.h
    STATIC_DEFINE     SHAREDMATH_NUMERICALMETHODS_STATIC_DEFINE
)

target_include_directories(SharedMath_NumericalMethods
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/NumericalMethods
)

target_link_libraries(SharedMath_NumericalMethods PUBLIC SharedMath_LinearAlgebra)

target_compile_definitions(SharedMath_NumericalMethods PUBLIC
    $<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:SHAREDMATH_NUMERICALMETHODS_STATIC_DEFINE>
)

# ── CUDA-specific settings ─────────────────────────────────────────────────
if(SHAREDMATH_ENABLE_CUDA)
    target_compile_definitions(SharedMath_NumericalMethods PUBLIC SHAREDMATH_CUDA)

    set_target_properties(SharedMath_NumericalMethods PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_ARCHITECTURES          "75;80;86;89"
    )

    target_link_libraries(SharedMath_NumericalMethods PUBLIC
        CUDA::cublas
        CUDA::cudart
        CUDA::cusolver
    )
endif()

sharedmath_configure_module(SharedMath_NumericalMethods)
