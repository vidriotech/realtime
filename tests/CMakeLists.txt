cmake_minimum_required(VERSION 3.17)
project(rtstests CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 17)

set(HEADER_FILES TestProbe.h
        ../DistanceMatrix.h ../Probe.h)
set(SOURCE_FILES KernelTests.cu
        ../DistanceMatrix.cpp DistanceMatrixTests.cpp
        ../Probe.cpp ProbeTests.cpp
        SocketReaderTests.cpp)
add_executable(rtstests ${SOURCE_FILES} ${HEADER_FILES})

add_subdirectory(lib)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

target_link_libraries(rtstests RTSKernels)
target_link_libraries(rtstests gtest gtest_main)

set_target_properties(
        rtstests
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
