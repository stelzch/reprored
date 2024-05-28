cmake_minimum_required(VERSION 3.10)

project(BinaryTreeSummation)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MPI REQUIRED)

find_package(benchmark REQUIRED)


if(${REPRODUCIBLE} STREQUAL "REPROBLAS")
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(reproblas REQUIRED IMPORTED_TARGET reproblas)
    list(APPEND WRAPPER_SRC src/c_wrapper_reproblas.cpp)
elseif(${REPRODUCIBLE} STREQUAL "BINARYTREESUMMATION")
    list(APPEND WRAPPER_SRC src/c_wrapper.cpp src/binary_tree.cpp)
else()
    message(FATAL_ERROR "Unsupported reproducibility mode: ${REPRODUCIBLE}. Must be either REPROBLAS or BINARYTREESUMMATION")
endif()


add_definitions(-DAVX)
add_library(binarytreesummation SHARED)
target_sources(binarytreesummation
    PRIVATE ${WRAPPER_SRC}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/binary_tree_summation.h>)

target_compile_options(binarytreesummation PRIVATE -Wall -O0 -ggdb -mavx)
target_include_directories(binarytreesummation 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(binarytreesummation PUBLIC MPI::MPI_C MPI::MPI_CXX)

if(BUILD_EXAMPLE_EXECUTABLE)
    add_executable(sum src/main.cpp src/io.cpp)
    target_link_libraries(sum PRIVATE binarytreesummation MPI::MPI_C MPI::MPI_CXX)
    target_compile_options(sum PRIVATE -Wall -O0 -ggdb -mavx)
endif()
