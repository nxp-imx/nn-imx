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

list(APPEND armnnNpuBackendWorkloads_sources
    CMakeLists.txt
    NpuActivationWorkload.hpp
    NpuBatchNormalizationWorkload.hpp
    NpuBatchToSpaceNdWorkload.hpp
    NpuConcatWorkload.hpp
    NpuConvolution2dWorkload.hpp
    NpuDepthWiseConvolution2dWorkload.hpp
    NpuElementwiseWorkload.hpp
    NpuFullyConnectedWorkload.hpp
    NpuL2NormalizationWorkload.hpp
    NpuLstmWorkload.hpp
    NpuMeanWorkload.hpp
    NpuPadWorkload.hpp
    NpuPermuteWorkload.hpp
    NpuPooling2dWorkload.hpp
    NpuPreluWorkload.hpp
    NpuReshapeWorkload.hpp
    NpuSoftmaxWorkload.hpp
    NpuComparisonWorkload.hpp
    NpuSpaceToBatchNdWorkload.hpp
    NpuSpaceToDepthWorkload.hpp
    NpuSplitterWorkload.hpp
    NpuStridedSliceWorkload.hpp
    NpuTensorCopyWorkload.hpp
    NpuTransposeConvolution2dWorkload.hpp
    NpuWorkloadUtils.hpp
    TNpuWorkloads.hpp
    FakeBiasSelector.hpp
)

add_library(armnnNpuBackendWorkloads OBJECT ${armnnNpuBackendWorkloads_sources})
set(OVXLIB_DIR $ENV{OVXLIB_DIR})
set(OVXLIB_INCLUDE ${OVXLIB_DIR}/include)
set(OVX_DRIVER_ROOT $ENV{VIVANTE_SDK_DIR})
set(OVX_DRIVER_INCLUDE ${OVX_DRIVER_ROOT}/include)
set(OVX_DRIVER_LIB     ${OVX_DRIVER_ROOT}/drivers)
set(NNRT_ROOT $ENV{NNRT_ROOT})
set(NNRT_INCLUDE ${NNRT_ROOT}/include)
include_directories(${OVXLIB_INCLUDE}
    ${OVX_DRIVER_INCLUDE}
    ${NNRT_INCLUDE}
    ${NNRT_INCLUDE}/graph_transformations
    ${PROJECT_SOURCE_DIR}/src/backends/vsi_npu
)
target_include_directories(armnnNpuBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnNpuBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnNpuBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_include_directories(armnnNpuBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/src/backends/vsi_npu)
