set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

cmake_minimum_required(VERSION 3.28.1 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)
find_package(SndFile REQUIRED)

add_executable(compressor compressor.cpp rice_code.cpp read_write_func.cpp)
target_link_libraries(compressor "${TORCH_LIBRARIES}" SndFile::sndfile) 
set_property(TARGET compressor PROPERTY CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3" "avx2")
