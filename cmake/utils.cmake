macro(set_if_undefined variable)
    if(NOT DEFINED ${variable})
        set(${variable} ${ARGN})
    endif()
endmacro()

# Apply standard build properties to a compiled SharedMath module target.
function(sharedmath_configure_module target)
    set_target_properties(${target} PROPERTIES
        VERSION                   ${PROJECT_VERSION}
        SOVERSION                 ${PROJECT_VERSION_MAJOR}
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET     hidden
        VISIBILITY_INLINES_HIDDEN ON
        DEBUG_POSTFIX             "${CMAKE_DEBUG_POSTFIX}"
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
