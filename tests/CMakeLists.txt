cmake_minimum_required(VERSION 3.17)
project(rtstests CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 17)

set(HEADER_FILES
        test_utilities/test_utilities.cuh test_utilities/test_probe.cuh test_utilities/test_exception.cuh)
set(SOURCE_FILES
        test_utilities/test_probe.cuh
        test_utilities/test_utilities.cuh
        kernels_test.cu unique_ptr_cuda_test.cu
        ../src/structures/distance_matrix.cuh distance_matrix_test.cu
        median_tree_node_test.cu median_tree_test.cu
        ../src/probe/probe.cuh probe_test.cu
        ../src/acquisition/file_reader.cuh file_reader_test.cu
        ../src/acquisition/socket_reader.cuh socket_reader_test.cu
        ../src/detection/detector.cuh detector_test.cu
        ../src/detection/threshold_computer.cuh threshold_computer_test.cu
        ../src/extraction/snippet.cuh snippet_test.cu
        ../src/extraction/snippet_extractor.cuh snippet_extractor_test.cu
        ../src/extraction/extractor.cuh extractor_test.cu
        ../src/extraction/feature_extractor.cuh
        ../src/pipeline/pipeline.cuh pipeline_test.cu
        ../src/pipeline/pipeline_thread_pool.cuh pipeline_thread_pool_test.cu utilities_test.cu params_test.cu extractor_test.cu)

add_executable(rtstests ${SOURCE_FILES} ${HEADER_FILES})

add_subdirectory(lib)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

target_link_libraries(rtstests RTSKernels cublas)
target_link_libraries(rtstests gtest gtest_main)

set_target_properties(
        rtstests
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
