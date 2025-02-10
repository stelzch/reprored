cmake_minimum_required(VERSION 3.10)

include(ExternalProject)

if(NOT REPROBLAS_URL)
    set(REPROBLAS_URL "https://bebop.cs.berkeley.edu/reproblas/include/download.php?filetype=tar.gz")
endif()

set(REPROBLAS_VERSION 2.1.0)
set(REPROBLAS_HASH "f0c6b219691b3c5a07c442e22ca0485f37454c2e4847a8872ba4e46190d439c2")

# Build ReproBLAS
ExternalProject_Add(
    reproblas
    URL ${REPROBLAS_URL}
    URL_HASH SHA256=${REPROBLAS_HASH}
    DOWNLOAD_NAME reproblas.tar.gz
    PATCH_COMMAND echo 'CFLAGS += -fPIC' >> config.mk &&
                  sed -i "s/CC = cc/CC = gcc/" config.mk && 
                  sed -i "s/BUILD_MPI = false/BUILD_MPI = true/" config.mk &&
                  patch --forward -p1 scripts/cogapp/cogapp.py < ${CMAKE_CURRENT_SOURCE_DIR}/reproblas-deprecated-imp.patch
    CONFIGURE_COMMAND make -C <SOURCE_DIR> update
    BUILD_COMMAND make -C <SOURCE_DIR>
    INSTALL_COMMAND make -C <SOURCE_DIR> BIN_DIR=<INSTALL_DIR>/bin LIB_DIR=<INSTALL_DIR>/lib INC_DIR=<INSTALL_DIR>/include DOC_DIR=<INSTALL_DIR>/doc install
    TEST_COMMAND make -C <SOURCE_DIR> check
)

ExternalProject_Get_property(reproblas INSTALL_DIR)

add_library(ReproBLAS INTERFACE)
target_include_directories(ReproBLAS INTERFACE ${INSTALL_DIR}/include)
target_link_libraries(ReproBLAS
    INTERFACE ${INSTALL_DIR}/lib/libbinned.a
              ${INSTALL_DIR}/lib/libbinnedblas.a
              ${INSTALL_DIR}/lib/libbinnedmpi.a
              ${INSTALL_DIR}/lib/libreproblas.a)
add_dependencies(ReproBLAS reproblas)
