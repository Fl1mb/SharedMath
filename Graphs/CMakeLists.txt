add_library(SharedMath_Graphs INTERFACE)
add_library(SharedMath::Graphs ALIAS SharedMath_Graphs)

# Header-only module (BinaryTree, BaseGraph are templates)
target_include_directories(SharedMath_Graphs INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(SharedMath_Graphs INTERFACE SharedMath_Core)
