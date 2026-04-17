add_library(SharedMath_Graphs INTERFACE)
add_library(SharedMath::Graphs ALIAS SharedMath_Graphs)

# Graphs module is header-only (BinaryTree, BaseGraph are templates).
target_link_libraries(SharedMath_Graphs INTERFACE SharedMath_Core)
