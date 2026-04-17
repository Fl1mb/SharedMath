include(GenerateExportHeader)

add_library(SharedMath_Functions
    GammaFunc.cpp
)
add_library(SharedMath::Functions ALIAS SharedMath_Functions)

generate_export_header(SharedMath_Functions
    EXPORT_MACRO_NAME SHAREDMATH_FUNCTIONS_EXPORT
    EXPORT_FILE_NAME  ${PROJECT_BINARY_DIR}/include/sharedmath_functions_export.h
    STATIC_DEFINE     SHAREDMATH_FUNCTIONS_STATIC_DEFINE
)

target_link_libraries(SharedMath_Functions PUBLIC SharedMath_Core)

target_compile_definitions(SharedMath_Functions PUBLIC
    $<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:SHAREDMATH_FUNCTIONS_STATIC_DEFINE>
)

sharedmath_configure_module(SharedMath_Functions)
