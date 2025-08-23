

add_library(
    SharedMath
    #Add source Files 
    geometry/Parallelogram.cpp
    geometry/Polygon.cpp
    geometry/Rectangle.cpp
    geometry/Triangle.cpp

    SharedMath.cpp
)

target_include_directories(SharedMath PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

install(TARGETS SharedMath DESTINATION SharedMathLib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/SharedMath DESTINATION include)
