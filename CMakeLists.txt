cmake_minimum_required(VERSION 3.17)
message(WARNING, "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
project(rts_2 CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 17)

# RTS
set(HEADER_FILES src/probe/probe.h src/probe/probe_reader.h
        src/structures/median_tree_node.h src/structures/median_tree.h src/structures/distance_matrix.h
        src/acquisition/file_reader.h
        src/detection/threshold_computer.h
        src/extraction/Snippet.h src/extraction/SnippetExtractor.h)
set(SOURCE_FILES src/main.cpp
        src/probe/probe.cpp src/probe/probe_reader.cpp
        src/structures/median_tree_node.cpp src/structures/median_tree.cpp src/structures/distance_matrix.cpp
        src/acquisition/file_reader.cpp
        src/detection/threshold_computer.cpp
        src/extraction/Snippet.cpp src/extraction/SnippetExtractor.cpp)

add_executable(rts_2 ${SOURCE_FILES} ${HEADER_FILES})

set_target_properties(
        rts_2
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

# CUDA
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

include_directories(src/kernels)
add_subdirectory(src/kernels)

target_link_libraries(rts_2 RTSKernels)

# QT
find_package(Qt5Widgets REQUIRED)
target_link_libraries(rts_2 Qt5::Widgets)

# tests
add_subdirectory(tests)
