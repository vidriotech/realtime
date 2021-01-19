cmake_minimum_required(VERSION 3.17)
project(rtstests CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 17)

set(HEADER_FILES utilities/TestProbe.h utilities/TestException.h)
set(SOURCE_FILES utilities/TestProbe.cpp utilities/TestException.cpp
        KernelTests.cu
        DistanceMatrixTests.cpp
        ../src/Probe.cpp ProbeTests.cpp
        FileReaderTests.cpp
        SnippetTests.cpp
        SnippetExtractorTests.cpp
        median_tree_node_test.cpp median_tree_test.cpp)
add_executable(rtstests ${SOURCE_FILES} ${HEADER_FILES})

add_subdirectory(lib)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

target_link_libraries(rtstests RTSKernels)
target_link_libraries(rtstests gtest gtest_main)

set_target_properties(
        rtstests
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
