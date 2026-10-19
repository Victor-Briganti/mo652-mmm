cmake_minimum_required(VERSION 3.18)

project(MMM C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(MPI REQUIRED)

set(BLA_VENDOR OpenBLAS)
find_package(BLAS REQUIRED)

find_path(CBLAS_INCLUDE_DIR cblas.h PATH_SUFFIXES openblas REQUIRED)
message(STATUS "Found cblas.h in: ${CBLAS_INCLUDE_DIR}")

foreach(t mms mmm mmmb)
  add_executable(${t} src/${t}.c)
  target_include_directories(${t} PRIVATE ${CBLAS_INCLUDE_DIR})
  target_link_libraries(${t} PRIVATE MPI::MPI_C BLAS::BLAS)
endforeach()