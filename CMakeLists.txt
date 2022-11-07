cmake_minimum_required(VERSION 3.10)

project(BinaryTreeSummation)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

find_package(benchmark REQUIRED)

add_compile_options(-Wall -O0 -ggdb -mavx)

add_library(binarytreesummation SHARED src/binarytreesummation.cpp)
target_include_directories(binarytreesummation PUBLIC src)

add_executable(benchmark src/benchmark.cpp)
target_link_libraries(benchmark benchmark::benchmark binarytreesummation MPI::MPI_C MPI::MPI_CXX)

add_executable(sum src/main.cpp src/io.cpp)
target_link_libraries(sum binarytreesummation MPI::MPI_C MPI::MPI_CXX)
