cmake_minimum_required(VERSION 3.17)
message(WARNING, "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
project(rts_2 CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 17)

# RTS
set(HEADER_FILES src/utilities.h
        src/structures/median_tree_node.h src/structures/median_tree.h src/structures/distance_matrix.h
        src/pipeline/pipeline_thread_pool.h
        src/probe/probe.h src/probe/probe_reader.h
        src/params/device_params.h src/params/acquire_params.h src/params/detect_params.h
        src/params/extract_params.h src/params/classify_params.h src/params/params.h
        src/acquisition/reader.h src/acquisition/file_reader.h src/acquisition/socket_reader.h
        src/detection/detector.cuh src/detection/threshold_computer.cuh
        src/extraction/snippet.h src/extraction/snippet_extractor.h
        src/extraction/extractor.cuh
        src/pipeline/pipeline.cuh src/extraction/feature_extractor.cu src/extraction/feature_extractor.cuh)
set(SOURCE_FILES src/main.cu
        src/structures/median_tree_node.cpp src/structures/median_tree.cpp src/structures/distance_matrix.cpp
        src/pipeline/pipeline_thread_pool.cpp
        src/probe/probe.cpp src/probe/probe_reader.cpp
        src/acquisition/reader.cpp src/acquisition/file_reader.cpp src/acquisition/socket_reader.cpp
        src/detection/detector.cu src/detection/threshold_computer.cu
        src/extraction/snippet.cpp src/extraction/snippet_extractor.cpp
        src/extraction/extractor.cu
        src/pipeline/pipeline.cu src/extraction/feature_extractor.cu src/extraction/feature_extractor.cuh)

add_executable(rts_2 ${SOURCE_FILES} ${HEADER_FILES})

set_target_properties(
        rts_2
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

# CUDA
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

include_directories(src/kernels)
add_subdirectory(src/kernels)

target_link_libraries(rts_2 RTSKernels cublas)

# tests
add_subdirectory(tests)
