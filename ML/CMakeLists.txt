include(GenerateExportHeader)

add_library(SharedMath_ML
    src/AutogradTensor.cpp
    src/DataLoader.cpp
    src/Models.cpp
    src/Module.cpp
    src/Optimizer.cpp
)
add_library(SharedMath::ML ALIAS SharedMath_ML)

generate_export_header(SharedMath_ML
    EXPORT_MACRO_NAME SHAREDMATH_ML_EXPORT
    EXPORT_FILE_NAME  ${PROJECT_BINARY_DIR}/include/sharedmath_ml_export.h
    STATIC_DEFINE     SHAREDMATH_ML_STATIC_DEFINE
)

target_include_directories(SharedMath_ML
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        # Generated export header
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/ML
)

target_link_libraries(SharedMath_ML
    PUBLIC
        SharedMath_LinearAlgebra
        SharedMath_Functions
)

target_compile_definitions(SharedMath_ML PUBLIC
    $<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:SHAREDMATH_ML_STATIC_DEFINE>
)

sharedmath_configure_module(SharedMath_ML)
