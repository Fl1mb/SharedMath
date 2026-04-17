include(GenerateExportHeader)

add_library(SharedMath_Geometry
    Parallelogram.cpp
    Rectangle.cpp
    Triangle.cpp
)
add_library(SharedMath::Geometry ALIAS SharedMath_Geometry)

generate_export_header(SharedMath_Geometry
    EXPORT_MACRO_NAME SHAREDMATH_GEOMETRY_EXPORT
    EXPORT_FILE_NAME  ${PROJECT_BINARY_DIR}/include/sharedmath_geometry_export.h
    STATIC_DEFINE     SHAREDMATH_GEOMETRY_STATIC_DEFINE
)

target_link_libraries(SharedMath_Geometry PUBLIC SharedMath_Core)

target_compile_definitions(SharedMath_Geometry PUBLIC
    $<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:SHAREDMATH_GEOMETRY_STATIC_DEFINE>
)

sharedmath_configure_module(SharedMath_Geometry)
