# SharedMath::Functions — header-only INTERFACE library
# All implementations are inline/template functions in .h files.

add_library(SharedMath_Functions INTERFACE)
add_library(SharedMath::Functions ALIAS SharedMath_Functions)

target_include_directories(SharedMath_Functions INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(SharedMath_Functions INTERFACE SharedMath_Core)
