find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(${PROJECT_NAME}Tests
    main.cpp
    test_binary_tree.cpp
    test_tensor.cpp
    test_matrixfunctions.cpp
    # test_geometry_lines.cpp
    # test_geometry_polygon.cpp
    # test_geometry_rect.cpp
    # test_geometry_triangle.cpp
    # test_geometry_circle.cpp
    # test_linAl_matrix.cpp
    # test_linAl_matrixOperations.cpp
    # test_functions.cpp
)

target_link_libraries(${PROJECT_NAME}Tests
    PRIVATE
        SharedMath::SharedMath
        GTest::GTest
        GTest::Main
)

enable_testing()
gtest_discover_tests(${PROJECT_NAME}Tests)

add_custom_command(
    TARGET ${PROJECT_NAME}Tests POST_BUILD
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    COMMENT "Running ${PROJECT_NAME} tests...")
