add_executable(demo_example example.cpp)
target_link_libraries(demo_example PRIVATE ${PROJECT_NAME}::${PROJECT_NAME})
