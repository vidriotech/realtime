cmake_minimum_required(VERSION 3.17)
project(RTSKernels CUDA)

set(CMAKE_CUDA_STANDARD 14)

set(HEADER_FILES ChanAdd.cuh Filters.cuh)
set(SOURCE_FILES ChanAdd.cu Filters.cu)

add_library(RTSKernels STATIC ${SOURCE_FILES} ${HEADER_FILES})

set_target_properties(
        RTSKernels
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)