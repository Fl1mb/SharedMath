cmake_minimum_required(VERSION 3.16)

string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" is_top_level)

if(NOT DEFINED CMAKE_DEBUG_POSTFIX)
    set(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Debug postfix")
endif()

project(SharedMath
    VERSION 1.0.0
    DESCRIPTION "Math library for shared computations"
    LANGUAGES CXX)

string(TOLOWER ${PROJECT_NAME} PROJECT_NAME_LOWER)
string(TOUPPER ${PROJECT_NAME} PROJECT_NAME_UPPER)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils.cmake)
include(GNUInstallDirs)

# --------------------------------------------------------------------------- #
# CUDA Configuration (Optional & Auto-Detect)
# --------------------------------------------------------------------------- #
option(${PROJECT_NAME}_USE_CUDA "Enable CUDA support in LinearAlgebra" ON)

if(${PROJECT_NAME}_USE_CUDA)
    find_package(CUDA QUIET)
    
    if(CUDA_FOUND)
        enable_language(CUDA)

        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89" CACHE STRING "CUDA architectures")
        endif()

        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -lineinfo")
        else()
            set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")
        endif()

        message(STATUS "[${PROJECT_NAME}] CUDA enabled: ON")
        message(STATUS "[${PROJECT_NAME}] CUDA version:      ${CMAKE_CUDA_COMPILER_VERSION}")
        message(STATUS "[${PROJECT_NAME}] CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    else()
        message(STATUS "[${PROJECT_NAME}] CUDA not found. Disabling CUDA support.")
        set(${PROJECT_NAME}_USE_CUDA OFF CACHE BOOL "Enable CUDA support" FORCE)
    endif()
else()
    message(STATUS "[${PROJECT_NAME}] CUDA enabled: OFF (disabled by user)")
endif()

# --------------------------------------------------------------------------- #
# Options
# --------------------------------------------------------------------------- #
option(${PROJECT_NAME}_BUILD_TESTS    "Build ${PROJECT_NAME} unit tests"       ON)
option(${PROJECT_NAME}_INSTALL        "Generate install rules"                 ${is_top_level})
option(${PROJECT_NAME}_SHARED_LIBS    "Build ${PROJECT_NAME} as a shared lib"  OFF)
option(${PROJECT_NAME}_BUILD_EXAMPLES "Build ${PROJECT_NAME} examples"         ON)

set_if_undefined(${PROJECT_NAME}_INSTALL_CMAKEDIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
    CACHE STRING "Install destination for ${PROJECT_NAME} CMake package files")

if(DEFINED ${PROJECT_NAME}_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ${${PROJECT_NAME}_SHARED_LIBS})
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY
        STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

message(STATUS "[${PROJECT_NAME}] Version:      ${PROJECT_VERSION}")
message(STATUS "[${PROJECT_NAME}] Build type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "[${PROJECT_NAME}] Shared libs:  ${BUILD_SHARED_LIBS}")
message(STATUS "[${PROJECT_NAME}] Top-level:    ${is_top_level}")
message(STATUS "[${PROJECT_NAME}] CUDA enabled: ${${PROJECT_NAME}_USE_CUDA}")

# --------------------------------------------------------------------------- #
# Subdirectories
# --------------------------------------------------------------------------- #
add_subdirectory(core)
add_subdirectory(LinearAlgebra)
add_subdirectory(geometry)
add_subdirectory(functions)
add_subdirectory(Graphs)
add_subdirectory(DSP)
add_subdirectory(NumericalMethods)

# Umbrella target
add_library(SharedMath INTERFACE)
add_library(SharedMath::SharedMath ALIAS SharedMath)

target_compile_features(SharedMath INTERFACE cxx_std_17)

# CUDA-фичи добавляем только если язык успешно включён
if(${PROJECT_NAME}_USE_CUDA)
    target_compile_features(SharedMath INTERFACE cuda_std_17)
endif()

target_link_libraries(SharedMath INTERFACE
    SharedMath_Core
    SharedMath_LinearAlgebra
    SharedMath_Geometry
    SharedMath_Functions
    SharedMath_Graphs
    SharedMath_DSP
    SharedMath_NumericalMethods
)

if(${PROJECT_NAME}_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(${PROJECT_NAME}_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# --------------------------------------------------------------------------- #
# Install
# --------------------------------------------------------------------------- #
if(${PROJECT_NAME}_INSTALL AND NOT CMAKE_SKIP_INSTALL_RULES)
    include(CMakePackageConfigHelpers)

    install(TARGETS
        SharedMath
        SharedMath_Core
        SharedMath_Functions
        SharedMath_Graphs
        SharedMath_DSP
        EXPORT ${PROJECT_NAME}_export
    )

    install(TARGETS
        SharedMath_LinearAlgebra
        SharedMath_Geometry
        SharedMath_NumericalMethods
        EXPORT   ${PROJECT_NAME}_export
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}    COMPONENT bin
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}    COMPONENT bin
                 NAMELINK_COMPONENT devel
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}    COMPONENT devel
    )

    if(MSVC AND BUILD_SHARED_LIBS)
        foreach(_mod LinearAlgebra Geometry NumericalMethods)
            install(FILES $<TARGET_PDB_FILE:SharedMath_${_mod}>
                DESTINATION ${CMAKE_INSTALL_BINDIR}
                COMPONENT   dbg
                OPTIONAL)
        endforeach()
    endif()

    foreach(_mod core functions LinearAlgebra geometry Graphs DSP NumericalMethods)
        install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${_mod}/include/
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            COMPONENT   devel
            FILES_MATCHING PATTERN "*.h")
    endforeach()

    install(DIRECTORY ${PROJECT_BINARY_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT   devel
        FILES_MATCHING PATTERN "*_export.h")

    install(EXPORT ${PROJECT_NAME}_export
        FILE        ${PROJECT_NAME}-targets.cmake
        NAMESPACE   ${PROJECT_NAME}::
        DESTINATION "${${PROJECT_NAME}_INSTALL_CMAKEDIR}"
        COMPONENT   devel)

    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/project-config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake
        INSTALL_DESTINATION "${${PROJECT_NAME}_INSTALL_CMAKEDIR}"
        PATH_VARS CMAKE_INSTALL_INCLUDEDIR)

    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake
        COMPATIBILITY SameMajorVersion)

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
        DESTINATION "${${PROJECT_NAME}_INSTALL_CMAKEDIR}"
        COMPONENT   devel)

    if(is_top_level)
        set(CPACK_VERBATIM_VARIABLES YES)
        set(CPACK_PACKAGE_CONTACT    "D.Rylov <danila.rylov2005@mail.ru>")
        set(CPACK_PACKAGE_VENDOR     "SharedMath Team")
        set(CPACK_PACKAGE_NAME       ${PROJECT_NAME})
        set(CPACK_PACKAGE_VERSION    ${PROJECT_VERSION})
        set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Shared Math Library with CUDA Support")
        set(CPACK_PACKAGE_FILE_NAME
            "${PROJECT_NAME}_${PROJECT_VERSION}_${CMAKE_SYSTEM_NAME}")
        set(CPACK_STRIP_FILES ON)

        if(UNIX)
            set(CPACK_GENERATOR "TBZ2;DEB")
            set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
            execute_process(
                COMMAND dpkg --print-architecture
                OUTPUT_VARIABLE CPACK_DEBIAN_PACKAGE_ARCHITECTURE
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)
            set(CPACK_PACKAGE_FILE_NAME
                "${CPACK_PACKAGE_FILE_NAME}_${CMAKE_SYSTEM_PROCESSOR}")
        else()
            set(CPACK_GENERATOR "ZIP")
            find_program(NSIS_PROGRAM makensis.exe)
            if(NSIS_PROGRAM)
                list(APPEND CPACK_GENERATOR "NSIS")
            endif()
        endif()

        include(CPack)
    endif()
endif()