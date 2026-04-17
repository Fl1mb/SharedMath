include(GNUInstallDirs)
include(GenerateExportHeader)

add_library(${PROJECT_NAME}
    geometry/Parallelogram.cpp
    geometry/Rectangle.cpp
    geometry/Triangle.cpp
    LinearAlgebra/MatrixView.cpp
    LinearAlgebra/MatrixOperationFactory.cpp
    LinearAlgebra/MatrixOperations.cpp
    LinearAlgebra/LUDecomposition.cpp
    functions/GammaFunc.cpp
    SharedMath.cpp
)

add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

# Generate the export header (SHAREDMATH_EXPORT macro)
generate_export_header(${PROJECT_NAME}
    EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/${PROJECT_NAME_LOWER}_export.h)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

set_target_properties(${PROJECT_NAME} PROPERTIES
    VERSION                   ${PROJECT_VERSION}
    SOVERSION                 ${PROJECT_VERSION_MAJOR}
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET     hidden
    VISIBILITY_INLINES_HIDDEN ON
)

target_compile_definitions(${PROJECT_NAME}
    PUBLIC
        # consumers of the static lib get SHAREDMATH_STATIC_DEFINE
        $<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:${PROJECT_NAME_UPPER}_STATIC_DEFINE>
        $<$<BOOL:${WIN32}>:WIN32 _USE_MATH_DEFINES>
    PRIVATE
        $<$<NOT:$<BOOL:${WIN32}>>:_USE_MATH_DEFINES>
)

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /EHsc /W3)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(UNIX)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${CMAKE_DL_LIBS})
endif()
