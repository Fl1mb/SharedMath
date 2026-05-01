find_package(GTest QUIET)

if(NOT GTest_FOUND)
    message(STATUS "System GTest not found, fetching from GitHub...")
    include(FetchContent)
    
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    FetchContent_MakeAvailable(googletest)

    set(GTEST_MAIN_TARGET gtest_main)
else()
    message(STATUS "Using system GTest")
    include(GoogleTest)
    set(GTEST_MAIN_TARGET GTest::gtest_main)
endif()

add_executable(${PROJECT_NAME}Tests
    main.cpp
    test_binary_tree.cpp
    test_union_find.cpp
    test_avl_tree.cpp
    test_adjacency_list_graph.cpp
    test_graph_algorithms.cpp
    test_dsp_fft.cpp
    test_dsp_windows.cpp
    test_dsp_convolution.cpp
    test_dsp_fir.cpp
    test_dsp_iir.cpp
    test_dsp_stft.cpp
    test_dsp_hilbert.cpp
    test_tensor.cpp
    test_matrixfunctions.cpp
    test_geometry_lines.cpp
    test_geometry_polygon.cpp
    test_geometry_rect.cpp
    test_geometry_triangle.cpp
    test_geometry_circle.cpp
    test_linAl_matrix.cpp
    test_linAl_matrixOperations.cpp
    test_functions.cpp
    test_numerical_differentiation.cpp
    test_numerical_integration.cpp
    test_numerical_ode.cpp
    test_numerical_integral_equations.cpp
    test_numerical_slae.cpp
    test_iterative_solvers.cpp
    test_dsp_spectral.cpp
    test_dsp_signal_generator.cpp
    test_dsp_filter_response.cpp
    test_dsp_filter_design_hz.cpp
    test_dsp_signal_processing.cpp
    test_dsp_streaming.cpp
    test_dsp_resampling.cpp
    test_dsp_rbj_filters.cpp
    test_ml_module.cpp
    test_ml_autograd.cpp
    test_ml_layers.cpp
    test_ml_optimizers.cpp
    test_ml_training.cpp
    test_ml_models.cpp
    test_ml_dataloader.cpp
    test_ml_metrics.cpp
    test_ml_preprocessing.cpp
    test_ml_tree_models.cpp
    test_ml_forest_models.cpp
    test_ml_regularized_models.cpp
    test_ml_model_selection.cpp
)

if(SHAREDMATH_ENABLE_CUDA)
    target_sources(${PROJECT_NAME}Tests PRIVATE
        test_tensor_cuda.cpp
        test_dynamicmatrix_cuda.cpp
    )
endif()

target_link_libraries(${PROJECT_NAME}Tests
    PRIVATE
        SharedMath::SharedMath
        SharedMath::ML
        ${GTEST_MAIN_TARGET}
)

enable_testing()

if(GTest_FOUND)
    gtest_discover_tests(${PROJECT_NAME}Tests)
else()
    add_test(NAME ${PROJECT_NAME}Tests COMMAND ${PROJECT_NAME}Tests)
endif()
