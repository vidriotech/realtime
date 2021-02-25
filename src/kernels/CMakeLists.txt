cmake_minimum_required(VERSION 3.17)
project(RTSKernels CUDA)

set(CMAKE_CUDA_STANDARD 14)

set(HEADER_FILES kernels.cuh filters.cuh thresholds.cuh pca.cuh operators.cuh)
set(SOURCE_FILES filters.cu thresholds.cu pca.cu)

add_library(RTSKernels STATIC ${SOURCE_FILES} ${HEADER_FILES})

set_target_properties(
        RTSKernels
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(RTSKernels cublas cublasLt cusolver)