cmake_minimum_required(VERSION 3.17)
project(RTSKernels CUDA)

set(CMAKE_CUDA_STANDARD 14)

set(HEADER_FILES filters.cuh thresholds.cuh dev_array.cuh stats.cuh)
set(SOURCE_FILES filters.cu thresholds.cu kernels.cuh dev_array.cu stats.cu)

add_library(RTSKernels STATIC ${SOURCE_FILES} ${HEADER_FILES})

set_target_properties(
        RTSKernels
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)