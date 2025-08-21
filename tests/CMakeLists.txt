cmake_minimum_required(VERSION 3.10)
project(SharedMathTests)

find_package(GTest REQUIRED)
include(GoogleTest)

link_directories(${CMAKE_SOURCE_DIR}/../build)

add_executable(SharedMathTests
    main.cpp
    test_geometry_lines.cpp
    test_geometry_polygon.cpp
    test_geometry_rect.cpp
)
target_link_libraries(SharedMathTests PRIVATE GTest::GTest GTest::Main SharedMath)
target_include_directories(SharedMathTests PRIVATE ${CMAKE_SOURCE_DIR}/include)

enable_testing()
gtest_discover_tests(SharedMathTests)
add_custom_command(
    TARGET SharedMathTests POST_BUILD
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    COMMENT "Running tests..."
)