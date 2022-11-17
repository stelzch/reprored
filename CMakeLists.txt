cmake_minimum_required(VERSION 3.10)

project(BinaryTreeSummation)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MPI REQUIRED)

find_package(benchmark REQUIRED)


add_definitions(-DAVX)
add_library(binarytreesummation STATIC src/binary_tree.cpp src/c_wrapper.cpp)

target_compile_options(binarytreesummation PRIVATE -Wall -O0 -ggdb -mavx)
target_include_directories(binarytreesummation 
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> ${MPI_INCLUDE_PATH}
    PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)

 #add_executable(benchmark src/benchmark.cpp)
 #target_link_libraries(benchmark benchmark::benchmark binarytreesummation MPI::MPI_C MPI::MPI_CXX)
 #
add_executable(sum src/main.cpp src/io.cpp)
target_link_libraries(sum binarytreesummation MPI::MPI_C MPI::MPI_CXX)
target_compile_options(sum PRIVATE -Wall -O0 -ggdb -mavx)
