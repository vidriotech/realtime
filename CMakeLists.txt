cmake_minimum_required(VERSION 3.17)
message(WARNING, "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
project(rts CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 17)

# RTS
set(HEADER_FILES src/utilities.cuh
        src/structures/median_tree_node.cuh src/structures/median_tree.cuh src/structures/distance_matrix.cuh
        src/pipeline/pipeline_thread_pool.cuh
        src/probe/probe.cuh src/probe/probe_reader.cuh
        src/params/device_params.cuh src/params/acquire_params.cuh src/params/detect_params.cuh
        src/params/extract_params.cuh src/params/classify_params.cuh src/params/params.cuh
        src/acquisition/reader.cuh src/acquisition/file_reader.cuh src/acquisition/socket_reader.cuh
        src/detection/detector.cuh src/detection/threshold_computer.cuh
        src/extraction/snippet.cuh src/extraction/snippet_extractor.cuh
        src/extraction/extractor.cuh
        src/pipeline/pipeline.cuh src/extraction/feature_extractor.cu src/extraction/feature_extractor.cuh
        src/clustering/dp_clustering.cuh)
set(SOURCE_FILES src/main.cu
        src/structures/median_tree_node.cu src/structures/median_tree.cu src/structures/distance_matrix.cu
        src/pipeline/pipeline_thread_pool.cu
        src/probe/probe.cu src/probe/probe_reader.cu
        src/acquisition/reader.cu src/acquisition/file_reader.cu src/acquisition/socket_reader.cu
        src/detection/detector.cu src/detection/threshold_computer.cu
        src/extraction/snippet.cu src/extraction/snippet_extractor.cu
        src/extraction/extractor.cu
        src/pipeline/pipeline.cu src/extraction/feature_extractor.cu src/clustering/dp_clustering.cu src/utilities.cu)

add_executable(rts ${SOURCE_FILES} ${HEADER_FILES})

set_target_properties(
        rts
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

# CUDA
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

include_directories(src/kernels)
add_subdirectory(src/kernels)

target_link_libraries(rts RTSKernels)

# tests
add_subdirectory(tests)
