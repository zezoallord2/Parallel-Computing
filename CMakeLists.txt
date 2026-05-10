cmake_minimum_required(VERSION 3.20)
project(parallel_mpi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED)

add_executable(parallel_mpi
    main.cpp
    communication.cpp
    utils.cpp
    algorithms/heat_diffusion.cpp
    algorithms/matrix_mult.cpp
)

target_include_directories(parallel_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(parallel_mpi PRIVATE MPI::MPI_CXX)

target_compile_options(parallel_mpi PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
)
