
if(BUILD_EXAMPLES)
    
    add_executable(demo_example example.cpp)
    target_link_libraries(demo_example PRIVATE SharedMath)



endif()