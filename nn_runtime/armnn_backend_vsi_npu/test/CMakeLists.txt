#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnNpuBackendUnitTests_sources
    NpuWorkloadFactoryHelper.hpp
    NpuLayerTests.cpp
    NpuCreateWorkloadTests.cpp
    NpuEndToEndTests.cpp
    NpuLayerSupportTests.cpp
    #NpuRuntimeTests.cpp
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wno-old-style-cast \
                    -Wno-sign-conversion -Wno-conversion -Wno-unused-variable\
                    -Wno-unused-function \
                    ")

add_library(armnnNpuBackendUnitTests OBJECT ${armnnNpuBackendUnitTests_sources})
set(OVXLIB_DIR $ENV{OVXLIB_DIR})
set(OVXLIB_INCLUDE ${OVXLIB_DIR}/include)
set(OVX_DRIVER_ROOT $ENV{VIVANTE_SDK_DIR})
set(OVX_DRIVER_INCLUDE ${OVX_DRIVER_ROOT}/include)
set(OVX_DRIVER_LIB     ${OVX_DRIVER_ROOT}/drivers)
set(NNRT_ROOT $ENV{NNRT_ROOT})
set(NNRT_INCLUDE ${NNRT_ROOT} ${NNRT_ROOT}/nnrt)
include_directories(${OVXLIB_INCLUDE}
    ${OVX_DRIVER_INCLUDE}
    ${NNRT_INCLUDE}
    ${PROJECT_SOURCE_DIR}/src/backends/vsi_npu
)
target_include_directories(armnnNpuBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnNpuBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnNpuBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_include_directories(armnnNpuBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/backends/dynamic/vsi_npu)
