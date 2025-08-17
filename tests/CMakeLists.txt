cmake_minimum_required(VERSION 3.10)
project(SharedMathTests)

find_package(GTest REQUIRED)
add_executable(SharedMathTests SharedMathTests.cxx)
target_link_libraries(SharedMathTests PRIVATE GTest::GTest GTest::Main)

enable_testing()
add_test(NAME SharedMathTests COMMAND SharedMathTests)

add_custom_command(
    TARGET SharedMathTests POST_BUILD
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    COMMENT "Tests on..."
)