cmake_minimum_required(VERSION 3.17)
project(RTSKernels CUDA)

set(CMAKE_CUDA_STANDARD 14)

set(HEADER_FILES ChanAdd.cuh filters.cuh thresholds.cuh dev_array.cuh)
set(SOURCE_FILES ChanAdd.cu filters.cu thresholds.cu kernels.cuh dev_array.cu)

add_library(RTSKernels STATIC ${SOURCE_FILES} ${HEADER_FILES})

set_target_properties(
        RTSKernels
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)