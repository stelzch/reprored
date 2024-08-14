cmake_minimum_required(VERSION 3.10)

project(BinaryTreeSummation)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MPI REQUIRED)

if("${REPRODUCIBLE}" STREQUAL "REPROBLAS")
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(reproblas REQUIRED IMPORTED_TARGET reproblas)
    list(APPEND WRAPPER_SRC src/c_wrapper_reproblas.cpp)
elseif("${REPRODUCIBLE}" STREQUAL "ALLREDUCE")
    list(APPEND WRAPPER_SRC src/c_wrapper_allreduce.cpp)
elseif("${REPRODUCIBLE}" STREQUAL "BINARYTREESUMMATION")
    list(APPEND WRAPPER_SRC src/c_wrapper.cpp src/binary_tree.cpp)
else()
    message(FATAL_ERROR "Unsupported reproducibility mode: ${REPRODUCIBLE}. Must be either REPROBLAS, ALLREDUCE or BINARYTREESUMMATION")
endif()


add_definitions(-DAVX)
add_library(binarytreesummation STATIC)
target_sources(binarytreesummation
    PRIVATE ${WRAPPER_SRC}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/binary_tree_summation.h>)

target_compile_options(binarytreesummation PRIVATE -Wall -O3 -ggdb -mavx)
target_include_directories(binarytreesummation 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if("${REPRODUCIBLE}" STREQUAL "REPROBLAS")
    target_link_directories(binarytreesummation PUBLIC ${reproblas_LIBRARY_DIRS})
    target_link_libraries(binarytreesummation PUBLIC MPI::MPI_C MPI::MPI_CXX ${reproblas_LIBRARIES})
    target_include_directories(binarytreesummation PUBLIC ${reproblas_INCLUDE_DIRS})
    target_compile_options(binarytreesummation PUBLIC ${reproblas_CFLAGS_OTHER})
else()
    target_link_libraries(binarytreesummation PUBLIC MPI::MPI_C MPI::MPI_CXX)
endif()

if(BUILD_EXAMPLE_EXECUTABLE)
    add_executable(sum src/main.cpp src/io.cpp)
    target_link_libraries(sum PRIVATE binarytreesummation MPI::MPI_C MPI::MPI_CXX)
    target_compile_options(sum PRIVATE -Wall -O3 -ggdb -mavx)
endif()

if(BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
          googletest
            URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    FetchContent_Declare(
        kassert
        GIT_REPOSITORY https://github.com/kamping-site/kassert
        GIT_TAG f0873f85ff046c6dee35a85148a51bfab73af44a
    )
    FetchContent_MakeAvailable(kassert)

    add_executable(test test/main.cpp test/reproducibility_test.cpp)
    target_link_libraries(test GTest::gtest_main GTest::gmock MPI::MPI_CXX kassert binarytreesummation)

    include(GoogleTest)
    gtest_discover_tests(test)

endif()
