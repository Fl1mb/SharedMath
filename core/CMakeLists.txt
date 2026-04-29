include(GNUInstallDirs)

# ── Sources ───────────────────────────────────────────────────────────────────

set(CORE_SOURCES
    src/CudaDeviceManager.cpp
    src/CudaDispatcher.cpp
)

# ── Target ────────────────────────────────────────────────────────────────────

add_library(SharedMath_Core STATIC ${CORE_SOURCES})
add_library(SharedMath::Core ALIAS SharedMath_Core)

target_include_directories(SharedMath_Core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(SharedMath_Core PUBLIC cxx_std_20)

target_compile_definitions(SharedMath_Core PUBLIC
    _USE_MATH_DEFINES
    $<$<BOOL:${WIN32}>:WIN32>
)

# ── Optional CUDA support ─────────────────────────────────────────────────────

if(SharedMath_USE_CUDA)
    target_link_libraries(SharedMath_Core PUBLIC CUDA::cudart)
    target_compile_definitions(SharedMath_Core PUBLIC SHAREDMATH_CUDA=1)
endif()

# ── Threads (needed by CudaDispatcher worker threads) ─────────────────────────

find_package(Threads REQUIRED)
target_link_libraries(SharedMath_Core PUBLIC Threads::Threads)
