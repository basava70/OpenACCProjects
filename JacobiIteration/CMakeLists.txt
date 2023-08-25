cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(run_jacobi_iteration)
set(CMAKE_CXX_COMPILER
    "/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/compilers/bin/nvc++")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "-std=c++11 -acc -Minfo=accel")
add_executable(run_jacobi_iteration_sequential jacobi_iteration_sequential.cc)
