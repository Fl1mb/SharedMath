

add_library(
    SharedMath
    #Add source Files 
    algebra.cpp 
)

target_include_directories(SharedMath PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

install(TARGETS SharedMath DESTINATION SharedMathLib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/SharedMath DESTINATION include)
