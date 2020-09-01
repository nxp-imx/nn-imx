# Platform defines.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_CROSSCOMPILING 1)

set(CROSS_COMPILE_ENV "/home/software/Linux/rockchips/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu")
set(ROOTFS "${CROSS_COMPILE_ENV}")

# Toolchain/compiler base.
set(CROSS_COMPILE "${CROSS_COMPILE_ENV}/bin/aarch64-linux-gnu-" CACHE STRING "Cross compiler prefix")
set(CMAKE_C_COMPILER "${CROSS_COMPILE}gcc")
set(CMAKE_CXX_COMPILER "${CROSS_COMPILE}g++")

set(CMAKE_FIND_ROOT_PATH
        "${ROOTFS}/lib"
        "${ROOTFS}/lib/gcc/aarch64-linux-gnu/6.3.1"
        )

message(CMAKE_FIND_ROOT_PATH "${CMAKE_FIND_ROOT_PATH}")

# Search libs and include files (but not programs) from toolchain dir.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY FIRST)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

add_definitions(-mtune=cortex-a35)
