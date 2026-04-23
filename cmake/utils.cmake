macro(set_if_undefined variable)
    if(NOT DEFINED ${variable})
        set(${variable} ${ARGN})
    endif()
endmacro()

# Apply standard build properties to a compiled SharedMath module target.
function(sharedmath_configure_module target)
    # Require C++17 from every consumer that links this target
    target_compile_features(${target} PUBLIC cxx_std_17)

    set_target_properties(${target} PROPERTIES
        VERSION                   ${PROJECT_VERSION}
        SOVERSION                 ${PROJECT_VERSION_MAJOR}
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET     hidden
        VISIBILITY_INLINES_HIDDEN ON
        DEBUG_POSTFIX             "${CMAKE_DEBUG_POSTFIX}"
        # After installation shared libs will find each other via RPATH
        INSTALL_RPATH             "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
        BUILD_WITH_INSTALL_RPATH  FALSE
        INSTALL_RPATH_USE_LINK_PATH ON
    )
    if(MSVC)
        target_compile_options(${target} PRIVATE /EHsc /W3)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    if(UNIX)
        target_link_libraries(${target} PRIVATE ${CMAKE_DL_LIBS})
    endif()
endfunction()
