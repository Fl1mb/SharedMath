include(GenerateExportHeader)

add_library(SharedMath_NumericalMethods
    src/Differentiation.cpp
    src/Integration.cpp
    src/ODE.cpp
    src/IntegralEquations.cpp
    src/SLAE.cpp
)
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

sharedmath_configure_module(SharedMath_NumericalMethods)
