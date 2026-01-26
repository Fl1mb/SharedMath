cmake_minimum_required(VERSION 3.16)

set(PROJECT_TITLE SharedMath)

MESSAGE("CMake version ${CMAKE_VERSION}")

string(TOLOWER ${PROJECT_TITLE} PROJECT_NAME_LOWER)
string(TOUPPER ${PROJECT_TITLE} PROJECT_NAME_UPPER)

string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" is_top_level)
message(STATUS "[INFO]: is_top_level ${is_top_level}")

if(NOT DEFINED CMAKE_DEBUG_POSTFIX)
    set(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Debug postfix")
endif()

project(${PROJECT_TITLE}
    VERSION 1.0.0
    DESCRIPTION "Math library for shared computations"
    LANGUAGES CXX)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils.cmake)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/version_utils.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/version_utils.cmake)
    make_version_from_git()
endif()

option(${PROJECT_NAME}_BUILD_TESTS "Build ${PROJECT_NAME} tests as separate executable" ON)
option(${PROJECT_NAME}_RUN_TESTS "Run ${PROJECT_NAME} tests" ON)
option(${PROJECT_NAME}_INSTALL "Generate target for installing ${PROJECT_NAME}" ${is_top_level})
option(${PROJECT_NAME}_SHARED_LIBS "Generate shared lib ${PROJECT_NAME}" OFF)
option(${PROJECT_NAME}_BUILD_EXAMPLES "Build ${PROJECT_NAME} examples" ON)

set_if_undefined(${PROJECT_NAME}_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}" CACHE STRING
    "Install path for ${PROJECT_NAME} package-related CMake files")

if(DEFINED ${PROJECT_NAME}_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ${${PROJECT_NAME}_SHARED_LIBS})
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set_if_undefined(CMAKE_CXX_VISIBILITY_PRESET hidden)
set_if_undefined(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

include(CheckCXXCompilerFlag)

if(MSVC)
    CHECK_CXX_COMPILER_FLAG("/std:c++20" COMPILER_SUPPORTS_CXX20)
    CHECK_CXX_COMPILER_FLAG("/std:c++latest" COMPILER_SUPPORTS_CXXLATEST)
    if(COMPILER_SUPPORTS_CXX20)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++20")
    elseif(COMPILER_SUPPORTS_CXXLATEST)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++latest")
    else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++20 support.")
    endif()
else()
    CHECK_CXX_COMPILER_FLAG("-std=c++20" COMPILER_SUPPORTS_CXX20)
    if(COMPILER_SUPPORTS_CXX20)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
    else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++20 support. Please use a different C++ compiler.")
    endif()
endif()

message(STATUS "[INFO]: CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE}")
message(STATUS "[INFO]: CMAKE_CONFIGURATION_TYPES ${CMAKE_CONFIGURATION_TYPES}")

add_subdirectory(include)
add_subdirectory(src)

include(GenerateExportHeader)
set(export_file_name "export_shared.h")

if(NOT BUILD_SHARED_LIBS)
    set(export_file_name "export_static.h")
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/export.h.in "${CMAKE_CURRENT_BINARY_DIR}/include/public/include/${PROJECT_NAME}/export.h")
generate_export_header(${PROJECT_NAME} EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/include/public/include/${PROJECT_NAME}/${export_file_name})

if(WIN32)
    set(DLL_VER_VENDOR "SharedMath Developer")
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/version_utils.cmake)
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/version_utils.cmake)
        make_dll_version(${DLL_VER_VENDOR})
        set(RESOURCES ${CMAKE_CURRENT_BINARY_DIR}/version.rc)
    endif()
endif()

if(WIN32)
    target_compile_definitions(${PROJECT_NAME} PUBLIC WIN32)
endif()

target_compile_definitions(${PROJECT_NAME} PUBLIC "$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:${PROJECT_NAME_UPPER}_STATIC_DEFINE>")

add_definitions(-D_USE_MATH_DEFINES)

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /EHsc)
endif()

if(UNIX)
    set(LIBS ${LIBS} ${CMAKE_DL_LIBS})
    message("CMAKE_DL_LIBS: ${CMAKE_DL_LIBS}")
endif(UNIX)

target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBS})

set(${PROJECT_NAME}_STATIC_SUFFIX "")
if(NOT BUILD_SHARED_LIBS)
    set(${PROJECT_NAME}_STATIC_SUFFIX "_static")

    if(CMAKE_VERSION VERSION_GREATER 2.8.8)
        set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
    elseif(CMAKE_COMPILER_IS_GNUCXX OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    endif()
endif()

if(${PROJECT_NAME}_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(${PROJECT_NAME}_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(${PROJECT_NAME}_INSTALL AND NOT CMAKE_SKIP_INSTALL_RULES)
    string(COMPARE EQUAL ${CMAKE_BUILD_TYPE} "Release" is_release_build)
    message(STATUS "[INFO]: is_release_build ${is_release_build}")

    include(CMakePackageConfigHelpers)
    include(GNUInstallDirs)
    
    configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/project-config.cmake.in 
        ${PROJECT_NAME}-config.cmake
        INSTALL_DESTINATION "${${PROJECT_NAME}_INSTALL_CMAKEDIR}")

    write_basic_package_version_file(${PROJECT_NAME}-config-version.cmake
        COMPATIBILITY SameMajorVersion)

    if(is_release_build)
        install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}_export
                RUNTIME COMPONENT bin
                LIBRARY COMPONENT bin
                ARCHIVE COMPONENT bin CONFIGURATIONS Release
                INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

        set(targets_file "${PROJECT_NAME}-shared-targets.cmake")

        if(NOT BUILD_SHARED_LIBS)
            set(targets_file "${PROJECT_NAME}-static-targets.cmake")
        endif()

        install(EXPORT ${PROJECT_NAME}_export
                COMPONENT devel
                FILE "${targets_file}"
                DESTINATION "${${PROJECT_NAME}_INSTALL_CMAKEDIR}"
                NAMESPACE ${PROJECT_NAME}::
                CONFIGURATIONS Release OPTIONAL)

        install(FILES
                "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
                "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
                COMPONENT devel
                DESTINATION "${${PROJECT_NAME}_INSTALL_CMAKEDIR}"
                CONFIGURATIONS Release OPTIONAL)

    else(is_release_build)
        install(TARGETS ${PROJECT_NAME}
                RUNTIME COMPONENT dbg
                LIBRARY COMPONENT dbg
                ARCHIVE COMPONENT dbg
                CONFIGURATIONS Debug RelWithDebInfo)

        if(MSVC)
            set(pdb_file "")
            set(pdb_file_destination "")

            if(BUILD_SHARED_LIBS)
                set(pdb_file "$<TARGET_PDB_FILE:${PROJECT_NAME}>")
                set(pdb_file_destination "${CMAKE_INSTALL_BINDIR}")
            else(BUILD_SHARED_LIBS)
                set(pdb_file "$<TARGET_FILE_DIR:${PROJECT_NAME}>/$<TARGET_FILE_PREFIX:${PROJECT_NAME}>$<TARGET_FILE_BASE_NAME:${PROJECT_NAME}>.pdb")
                set(pdb_file_destination "${CMAKE_INSTALL_LIBDIR}")
            endif(BUILD_SHARED_LIBS)

            install(FILES "${pdb_file}"
                COMPONENT dbg
                CONFIGURATIONS Debug RelWithDebInfo
                DESTINATION "${pdb_file_destination}"
                OPTIONAL)
        endif(MSVC)
    endif(is_release_build)
    
    if(is_top_level)
        set(CPACK_VERBATIM_VARIABLES YES)
        set(CPACK_SET_DESTDIR TRUE)
        set(CPACK_PACKAGE_RELOCATABLE ON)

        set(CPACK_PACKAGE_CONTACT "D.Rylov <danila.rylov2005@mail.ru>")
        set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Shared Math Library")
        set(CPACK_PACKAGE_VENDOR "SharedMath Team")

        set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
        set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})

        set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}_${PROJECT_VERSION}_${CMAKE_SYSTEM_NAME}")
        set(CPACK_STRIP_FILES is_release_build)
        set(CPACK_INSTALL_PREFIX "/")

        if(UNIX)
            set(CPACK_GENERATOR "TBZ2;DEB")
            set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS On)
            set(CPACK_STRIP_FILES ON)
            execute_process(
                COMMAND dpkg --print-architecture
                OUTPUT_VARIABLE CPACK_DEBIAN_PACKAGE_ARCHITECTURE
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            
            if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
                set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE arm64)
            endif()
            
            set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}_${CMAKE_SYSTEM_PROCESSOR}${${PROJECT_NAME}_STATIC_SUFFIX}")
        else()
            set(CPACK_GENERATOR "ZIP")
            find_program(NSIS_PROGRAM makensis.exe
                PATHS
                "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\NSIS;InstallLocation]"
                "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Wow6432Node\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\NSIS;InstallLocation]")
            
            if(NSIS_PROGRAM)
                set(CPACK_SET_DESTDIR FALSE)
                set(CPACK_GENERATOR "${CPACK_GENERATOR};NSIS")
            endif()
            
            set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}${${PROJECT_NAME}_STATIC_SUFFIX}")
        endif()
        
        message(STATUS "[INFO]: CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_FILE_NAME}")
        include(CPack)
    endif(is_top_level)
endif()