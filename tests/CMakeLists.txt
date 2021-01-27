cmake_minimum_required(VERSION 3.17)
project(rtstests CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 17)

set(HEADER_FILES
        test_utilities/test_utilities.h test_utilities/test_probe.h test_utilities/test_exception.h)
set(SOURCE_FILES
        test_utilities/test_probe.cpp
        kernel_test.cu
        distance_matrix_test.cpp
        ../src/probe/probe.cpp probe_test.cpp
        file_reader_test.cpp
        SnippetTests.cpp SnippetExtractorTests.cpp
        median_tree_node_test.cpp median_tree_test.cpp threshold_computer_test.cpp detector_test.cpp test_utilities/test_utilities.h test_utilities/test_utilities.cpp)
add_executable(rtstests ${SOURCE_FILES} ${HEADER_FILES})

add_subdirectory(lib)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

target_link_libraries(rtstests RTSKernels)
target_link_libraries(rtstests gtest gtest_main)

set_target_properties(
        rtstests
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
