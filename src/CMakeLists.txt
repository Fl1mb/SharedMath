add_subdirectory(core)
add_subdirectory(LinearAlgebra)
add_subdirectory(geometry)
add_subdirectory(functions)
add_subdirectory(Graphs)
add_subdirectory(DSP)

# Umbrella target — links every module so users can do:
#   target_link_libraries(myapp PRIVATE SharedMath::SharedMath)
add_library(SharedMath INTERFACE)
add_library(SharedMath::SharedMath ALIAS SharedMath)

target_link_libraries(SharedMath INTERFACE
    SharedMath_Core
    SharedMath_LinearAlgebra
    SharedMath_Geometry
    SharedMath_Functions
    SharedMath_Graphs
    SharedMath_DSP
)
