cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(noise noise.cpp rice_code.cpp)
target_link_libraries(noise "${TORCH_LIBRARIES}" sndfile) 
set_property(TARGET noise PROPERTY CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_RELEASE "-O3" "avx2")
