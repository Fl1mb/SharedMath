include(GenerateExportHeader)

add_library(SharedMath_LinearAlgebra
    MatrixView.cpp
    MatrixOperationFactory.cpp
    MatrixOperations.cpp
    LUDecomposition.cpp
)
add_library(SharedMath::LinearAlgebra ALIAS SharedMath_LinearAlgebra)

generate_export_header(SharedMath_LinearAlgebra
    EXPORT_MACRO_NAME SHAREDMATH_LINEARALGEBRA_EXPORT
    EXPORT_FILE_NAME  ${PROJECT_BINARY_DIR}/include/sharedmath_linearalgebra_export.h
    STATIC_DEFINE     SHAREDMATH_LINEARALGEBRA_STATIC_DEFINE
)

target_link_libraries(SharedMath_LinearAlgebra PUBLIC SharedMath_Core)

target_compile_definitions(SharedMath_LinearAlgebra PUBLIC
    $<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:SHAREDMATH_LINEARALGEBRA_STATIC_DEFINE>
)

sharedmath_configure_module(SharedMath_LinearAlgebra)
