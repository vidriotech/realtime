cmake_minimum_required(VERSION 3.17)
project(rtstests CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 17)

set(HEADER_FILES
        test_utilities/test_utilities.h test_utilities/test_probe.h test_utilities/test_exception.h)
set(SOURCE_FILES
        test_utilities/test_probe.cpp
        test_utilities/test_utilities.cpp
        kernels_test.cu unique_ptr_cuda_test.cu
        ../src/structures/distance_matrix.cpp distance_matrix_test.cpp
        median_tree_node_test.cpp median_tree_test.cpp
        ../src/probe/probe.cpp probe_test.cpp
        ../src/acquisition/file_reader.cpp file_reader_test.cpp
        ../src/acquisition/socket_reader.cpp socket_reader_test.cpp
        ../src/detection/detector.cpp detector_test.cpp
        ../src/detection/threshold_computer.cpp threshold_computer_test.cpp
        ../src/extraction/snippet.cpp snippet_test.cpp
        ../src/extraction/snippet_extractor.cpp snippet_extractor_test.cpp
        ../src/extraction/extractor.cpp extractor_test.cpp
        ../src/pipeline/pipeline.cpp pipeline_test.cpp
        ../src/pipeline/pipeline_thread_pool.cpp pipeline_thread_pool_test.cpp utilities_test.cpp params_test.cpp extractor_test.cpp)
add_executable(rtstests ${SOURCE_FILES} ${HEADER_FILES})

add_subdirectory(lib)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

target_link_libraries(rtstests RTSKernels)
target_link_libraries(rtstests gtest gtest_main)

set_target_properties(
        rtstests
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
