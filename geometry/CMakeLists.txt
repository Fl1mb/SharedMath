include(GenerateExportHeader)

add_library(SharedMath_Geometry
    src/Parallelogram.cpp
    src/Rectangle.cpp
    src/Triangle.cpp
)
add_library(SharedMath::Geometry ALIAS SharedMath_Geometry)

generate_export_header(SharedMath_Geometry
    EXPORT_MACRO_NAME SHAREDMATH_GEOMETRY_EXPORT
    EXPORT_FILE_NAME  ${PROJECT_BINARY_DIR}/include/sharedmath_geometry_export.h
    STATIC_DEFINE     SHAREDMATH_GEOMETRY_STATIC_DEFINE
)

target_include_directories(SharedMath_Geometry
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/geometry
)

target_link_libraries(SharedMath_Geometry PUBLIC SharedMath_Core)

target_compile_definitions(SharedMath_Geometry PUBLIC
    $<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:SHAREDMATH_GEOMETRY_STATIC_DEFINE>
)

sharedmath_configure_module(SharedMath_Geometry)
