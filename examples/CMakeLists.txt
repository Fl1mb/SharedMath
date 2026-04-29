add_executable(demo_example example.cpp)
target_link_libraries(demo_example PRIVATE ${PROJECT_NAME}::${PROJECT_NAME})

add_executable(dsp_find_dominant_frequency dsp_find_dominant_frequency.cpp)
target_link_libraries(dsp_find_dominant_frequency PRIVATE ${PROJECT_NAME}::${PROJECT_NAME})

add_executable(dsp_remove_hum dsp_remove_hum.cpp)
target_link_libraries(dsp_remove_hum PRIVATE ${PROJECT_NAME}::${PROJECT_NAME})

add_executable(dsp_bandpass_filter dsp_bandpass_filter.cpp)
target_link_libraries(dsp_bandpass_filter PRIVATE ${PROJECT_NAME}::${PROJECT_NAME})

add_executable(dsp_resample dsp_resample.cpp)
target_link_libraries(dsp_resample PRIVATE ${PROJECT_NAME}::${PROJECT_NAME})

add_executable(dsp_streaming_blocks dsp_streaming_blocks.cpp)
target_link_libraries(dsp_streaming_blocks PRIVATE ${PROJECT_NAME}::${PROJECT_NAME})
