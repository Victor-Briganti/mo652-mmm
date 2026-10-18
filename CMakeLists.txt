cmake_minimum_required(VERSION 3.10)

project(MMM C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(MPI REQUIRED)

set(BLA_VENDOR OpenBLAS)
find_package(BLAS REQUIRED)

add_executable(mms src/mms.c)
target_link_libraries(mms PRIVATE MPI::MPI_C)

add_executable(mmm src/mmm.c)
target_link_libraries(mmm PRIVATE MPI::MPI_C)

add_executable(mmmb src/mmmb.c)
target_link_libraries(mmmb PRIVATE MPI::MPI_C BLAS::BLAS)
