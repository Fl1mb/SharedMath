include(GenerateExportHeader)

# ── Optional CUDA support ──────────────────────────────────────────────────────
# SHAREDMATH_ENABLE_CUDA is set by the root CMakeLists via check_language(CUDA).
# We declare the option here only so it is visible in cmake-gui / ccmake;
# the default OFF is overridden to ON by the root when nvcc is found.
option(SHAREDMATH_ENABLE_CUDA "Enable CUDA acceleration for Tensor and LinearAlgebra" OFF)

if(SHAREDMATH_ENABLE_CUDA)
    # CUDAToolkit and enable_language(CUDA) are already handled by the root
    # CMakeLists before add_subdirectory(LinearAlgebra) is called.
    # We just verify the toolkit target is available.
    if(NOT TARGET CUDA::cublas)
        find_package(CUDAToolkit REQUIRED)
    endif()
    message(STATUS "SharedMath LinearAlgebra: CUDA support ON (toolkit ${CUDAToolkit_VERSION})")
endif()

# ── Library sources ────────────────────────────────────────────────────────────
set(SHAREDMATH_LINALG_SOURCES
    src/MatrixView.cpp
    src/MatrixOperationFactory.cpp
    src/MatrixOperations.cpp
    src/LUDecomposition.cpp
    src/Tensor.cpp
    src/DynamicMatrix.cpp
    src/DynamicVector.cpp
    src/SparseMatrix.cpp
    src/LinearSolver.cpp
    src/PCA.cpp
    src/MatrixFunctions.cpp
    src/IterativeSolvers.cpp
)

if(SHAREDMATH_ENABLE_CUDA)
    list(APPEND SHAREDMATH_LINALG_SOURCES
        src/TensorCUDA.cu
        src/DynamicMatrixCUDA.cu
    )
endif()

add_library(SharedMath_LinearAlgebra ${SHAREDMATH_LINALG_SOURCES})
add_library(SharedMath::LinearAlgebra ALIAS SharedMath_LinearAlgebra)

# ── CUDA-specific settings ─────────────────────────────────────────────────────
if(SHAREDMATH_ENABLE_CUDA)
    target_compile_definitions(SharedMath_LinearAlgebra PUBLIC SHAREDMATH_CUDA)

    set_target_properties(SharedMath_LinearAlgebra PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON          # needed for device code in .cu
        CUDA_ARCHITECTURES          "75;86;89" # Turing / Ampere / Ada — adjust to your GPU
    )

    target_link_libraries(SharedMath_LinearAlgebra PUBLIC
        CUDA::cublas
        CUDA::cudart
    )
endif()

# ── Export header ──────────────────────────────────────────────────────────────
generate_export_header(SharedMath_LinearAlgebra
    EXPORT_MACRO_NAME SHAREDMATH_LINEARALGEBRA_EXPORT
    EXPORT_FILE_NAME  ${PROJECT_BINARY_DIR}/include/sharedmath_linearalgebra_export.h
    STATIC_DEFINE     SHAREDMATH_LINEARALGEBRA_STATIC_DEFINE
)

target_include_directories(SharedMath_LinearAlgebra
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/LinearAlgebra
)

target_link_libraries(SharedMath_LinearAlgebra PUBLIC SharedMath_Core)

target_compile_definitions(SharedMath_LinearAlgebra PUBLIC
    $<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:SHAREDMATH_LINEARALGEBRA_STATIC_DEFINE>
)

sharedmath_configure_module(SharedMath_LinearAlgebra)
