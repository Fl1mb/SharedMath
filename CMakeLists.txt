cmake_minimum_required(VERSION 3.10)
project(SharedMath VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_SHARED_LIBS "Build shared library" OFF)
option(BUILD_EXAMPLES "Make Examples" ON)

add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(tests)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

