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

# --------------------------------------------------------------------------- #
# Subdirectories
# --------------------------------------------------------------------------- #
add_subdirectory(src)

if(${PROJECT_NAME}_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(${PROJECT_NAME}_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# --------------------------------------------------------------------------- #
# Install
# --------------------------------------------------------------------------- #
if(${PROJECT_NAME}_INSTALL AND NOT CMAKE_SKIP_INSTALL_RULES)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    install(TARGETS ${PROJECT_NAME}
        EXPORT  ${PROJECT_NAME}_export
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}   COMPONENT bin
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}   COMPONENT bin
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}   COMPONENT bin
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    if(MSVC AND BUILD_SHARED_LIBS)
        install(FILES $<TARGET_PDB_FILE:${PROJECT_NAME}>
            DESTINATION ${CMAKE_INSTALL_BINDIR}
            COMPONENT   dbg
            OPTIONAL)
    endif()

    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT   devel
        FILES_MATCHING PATTERN "*.h")

    install(FILES ${PROJECT_BINARY_DIR}/include/${PROJECT_NAME_LOWER}_export.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT   devel)

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

    # ----------------------------------------------------------------------- #
    # CPack (top-level only)
    # ----------------------------------------------------------------------- #
    if(is_top_level)
        set(CPACK_VERBATIM_VARIABLES YES)
        set(CPACK_PACKAGE_CONTACT    "D.Rylov <danila.rylov2005@mail.ru>")
        set(CPACK_PACKAGE_VENDOR     "SharedMath Team")
        set(CPACK_PACKAGE_NAME       ${PROJECT_NAME})
        set(CPACK_PACKAGE_VERSION    ${PROJECT_VERSION})
        set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Shared Math Library")
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
