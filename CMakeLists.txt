cmake_minimum_required(VERSION 3.10)

if (USE_SCOREP)
    set(CMAKE_CXX_COMPILER ${CMAKE_CURRENT_SOURCE_DIR}/contrib/scorep_compiler.sh)
    message("Using SCOREP, ${CMAKE_C_COMPILER}")
endif ()

project(BinaryTreeSummation)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MPI REQUIRED)
find_package(PkgConfig REQUIRED)
add_subdirectory(libs/reproblas)

option(USE_SCOREP "Enable instrumentation with ScoreP" OFF)
option(ENABLE_LTO "Enable link-time-optimizations" OFF)
option(BUILD_BENCHMARK "Build benchmark executable" ON)
option(BUILD_TESTS "Build unit tests" ON)


list(APPEND WRAPPER_SRC src/c_wrapper.cpp src/binary_tree_summation.cpp src/message_buffer.cpp src/reproblas_summation.cpp src/kgather_summation.cpp src/util.cpp src/allreduce_summation.cpp src/dual_tree_summation.cpp src/summation.cpp)


add_definitions(-DAVX -DREPRODUCIBLE=${REPRODUCIBLE})
add_library(binarytreesummation STATIC
        include/dual_tree_topology.hpp
        include/MAryTree.hpp)
target_sources(binarytreesummation
        PRIVATE ${WRAPPER_SRC}
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/binary_tree_summation.h>)

target_compile_options(binarytreesummation PRIVATE -Wall $<$<CONFIG:RELEASE>:-O3> -ggdb -mavx -mavx2 -Wextra -pedantic)

target_compile_definitions(binarytreesummation PRIVATE $<$<CONFIG:DEBUG>:DEBUG_TRACE>)


if (USE_GPROF)
    target_compile_options(binarytreesummation PRIVATE -pg)
endif ()

if (NOT USE_SCOREP AND ENABLE_LTO)
    target_compile_options(binarytreesummation PRIVATE -flto)
endif ()
target_include_directories(binarytreesummation
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

find_package(TBB)
target_link_libraries(binarytreesummation
    PUBLIC MPI::MPI_C MPI::MPI_CXX $<$<TARGET_EXISTS:TBB::tbb>:TBB::tbb>
    ReproBLAS)
#target_include_directories(binarytreesummation PUBLIC ${reproblas_INCLUDE_DIRS})
#target_compile_options(binarytreesummation PUBLIC ${reproblas_CFLAGS_OTHER})

add_executable(dualtreecalculator src/dual_tree_calculator.cpp)
target_link_libraries(dualtreecalculator PRIVATE binarytreesummation)
target_include_directories(dualtreecalculator
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

if (USE_SCOREP)
    target_compile_definitions(binarytreesummation PRIVATE -DSCOREP)
endif ()

if (BUILD_EXAMPLE_EXECUTABLE)
    add_executable(sum src/main.cpp src/io.cpp)
    target_link_libraries(sum PRIVATE binarytreesummation MPI::MPI_C MPI::MPI_CXX)
    target_compile_options(sum PRIVATE -Wall -O3 -ggdb -mavx)
endif ()

if (BUILD_BENCHMARK)
    add_executable(benchmark src/benchmark.cpp)
    target_link_libraries(benchmark PRIVATE binarytreesummation MPI::MPI_C MPI::MPI_CXX)
    target_compile_options(benchmark PRIVATE -Wall -O3 -ggdb -mavx)
    if (USE_GPROF)
        target_compile_options(benchmark PRIVATE -pg)
        target_link_options(benchmark PRIVATE -pg)
    endif ()
    if (USE_SCOREP)
        target_compile_definitions(benchmark PRIVATE -DSCOREP)
    endif ()
endif ()

if (BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest
            GIT_TAG v1.17.0 # 52eb8108c5bdec04579160ae17225d66034bd723
            GIT_SHALLOW true
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_executable(test test/main.cpp test/reproducibility_test.cpp test/chunked_array_test.cpp
            test/dual_tree_test.cpp
            test/m_ary_tree.cpp)
    target_link_libraries(test GTest::gtest_main GTest::gmock MPI::MPI_CXX binarytreesummation)

    include(GoogleTest)
    gtest_discover_tests(test)

endif ()
