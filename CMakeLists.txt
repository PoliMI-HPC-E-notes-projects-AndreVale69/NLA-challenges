cmake_minimum_required(VERSION 3.29)
project(nla_challenges)

set(CMAKE_CXX_STANDARD 17)

# mk
include_directories(
        /u/sw/toolchains/gcc-glibc/11.2.0/base/include
        /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/eigen/3.3.9/include/eigen3
        /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/lis/2.0.30/include
        /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/dealii/9.5.1/include
)

add_executable(
        main challenge-1/src/main.cpp
        challenge-1/src/utils/convert_utils.cpp
        challenge-1/src/utils/image_manipulation.cpp
)
