#/****************************************************************************
#*
#*    Copyright (c) 2019 Vivante Corporation
#*
#*    Permission is hereby granted, free of charge, to any person obtaining a
#*    copy of this software and associated documentation files (the "Software"),
#*    to deal in the Software without restriction, including without limitation
#*    the rights to use, copy, modify, merge, publish, distribute, sublicense,
#*    and/or sell copies of the Software, and to permit persons to whom the
#*    Software is furnished to do so, subject to the following conditions:
#*
#*    The above copyright notice and this permission notice shall be included in
#*    all copies or substantial portions of the Software.
#*
#*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#*    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#*    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#*    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#*    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#*    DEALINGS IN THE SOFTWARE.
#*
#*****************************************************************************/

list(APPEND armnnNputBackend_sources
        arm_nn_interpreter.cpp
        arm_nn_interpreter.hpp
        NpuBackendContext.cpp
        NpuBackendContext.hpp
        NpuBackend.cpp
        NpuBackend.hpp
        NpuBackendId.hpp
        NpuLayerSupport.cpp
        NpuLayerSupport.hpp
        NpuModelShell.cpp
        NpuModelShell.hpp
        NpuTensorAllocator.cpp
        NpuTensorAllocator.hpp
        NpuTensorHandler.cpp
        NpuTensorHandler.hpp
        NpuTensorHandleFactory.cpp
        NpuTensorHandleFactory.hpp
        NpuWorkloadFactory.cpp
        NpuWorkloadFactory.hpp
        NpuStaticRegistry.cpp
    )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wno-old-style-cast \
        -Wno-sign-conversion -Wno-conversion -Wno-unused-variable")
add_library(armnnNpuBackend OBJECT ${armnnNputBackend_sources})
#target_link_libraries(armnnNpuBackend PUBLIC ovxlib)
#TARGET_LINK_LIBRARIES(armnnNpuBackend libovxlib.so libnnrt.so)

if(NOT DEFINED ENV{VIVANTE_SDK_DIR})
    message(FATAL_ERROR "please set ENV: VIVANTE_SDK_DIR")
    else()
set(OVX_DRIVER_ROOT $ENV{VIVANTE_SDK_DIR})
set(OVX_DRIVER_INCLUDE ${OVX_DRIVER_ROOT}/include)
set(OVX_DRIVER_LIB     ${OVX_DRIVER_ROOT}/drivers)
endif()

if(NOT DEFINED ENV{OVXLIB_DIR})
    message(FATAL_ERROR "please set ENV: OVXLIB_DIR")
    else()
set(OVXLIB_DIR $ENV{OVXLIB_DIR})
set(OVXLIB_INCLUDE ${OVXLIB_DIR}/include)
endif()

if(NOT DEFINED ENV{NNRT_ROOT})
    message(FATAL_ERROR "please set ENV: NNRT_ROOT")
else()
    set(NNRT_ROOT $ENV{NNRT_ROOT})
    set(NNRT_INCLUDE ${NNRT_ROOT} ${NNRT_ROOT}/nnrt)
endif()

include_directories(
        ${OVXLIB_INCLUDE}
        ${OVX_DRIVER_INCLUDE}
        ${NNRT_INCLUDE}
        )

link_directories(${OVX_DRIVER_LIB})
    message("ovxlib include: ${OVXLIB_INCLUDE}")
    message("driver include: ${OVX_DRIVER_INCLUDE}")
    message("nnrt include: ${NNRT_INCLUDE}")

target_include_directories(armnnNpuBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnNpuBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnNpuBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_include_directories(armnnNpuBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/backends/vsi_npu)
target_include_directories(armnnNpuBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/profiling)
target_include_directories(armnnNpuBackend PRIVATE ${PROJECT_SOURCE_DIR}/profiling/common/include)

add_subdirectory(workloads)

if(BUILD_UNIT_TESTS)
    add_subdirectory(test)
endif()
