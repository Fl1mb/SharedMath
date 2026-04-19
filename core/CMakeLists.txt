include(GNUInstallDirs)

add_library(SharedMath_Core INTERFACE)
add_library(SharedMath::Core ALIAS SharedMath_Core)

target_include_directories(SharedMath_Core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(SharedMath_Core INTERFACE cxx_std_20)

target_compile_definitions(SharedMath_Core INTERFACE
    _USE_MATH_DEFINES
    $<$<BOOL:${WIN32}>:WIN32>
)
