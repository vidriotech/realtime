cmake_minimum_required(VERSION 3.17)
project(rts_2 CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 17)

# RTS
set(HEADER_FILES Probe.h DistanceMatrix.h  SocketReader.h
        tests/TestSocketServer.h)
set(SOURCE_FILES main.cpp Probe.cpp DistanceMatrix.cpp SocketReader.cpp
        tests/TestSocketServer.cpp)

#set(HEADER_FILES)
#set(SOURCE_FILES main.cpp)
add_executable(rts_2 ${SOURCE_FILES} ${HEADER_FILES})

set_target_properties(
        rts_2
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

# CUDA
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

include_directories(kernels)
add_subdirectory(kernels)

target_link_libraries(rts_2 RTSKernels)

# QT
find_package(Qt5Widgets REQUIRED)
target_link_libraries(rts_2 Qt5::Widgets)

# tests
add_subdirectory(tests)
