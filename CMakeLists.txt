cmake_minimum_required(VERSION 3.17)
message(WARNING, "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
project(rts_2 CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 17)

# RTS
set(HEADER_FILES src/probe.h src/DistanceMatrix.h
        src/acquisition/file_reader.h src/extraction/SnippetExtractor.h src/extraction/Snippet.h
        src/structures/median_tree.h)
set(SOURCE_FILES src/main.cpp src/probe.cpp src/DistanceMatrix.cpp
        src/acquisition/file_reader.cpp src/extraction/SnippetExtractor.cpp src/extraction/Snippet.cpp
        src/structures/median_tree_node.h src/structures/median_tree_node.cpp src/structures/median_tree.cpp)

#set(HEADER_FILES)
#set(SOURCE_FILES main.cpp)
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
