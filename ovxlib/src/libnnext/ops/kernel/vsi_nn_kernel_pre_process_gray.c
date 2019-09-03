/****************************************************************************
*
*    Copyright (c) 2019 Vivante Corporation
*
*    Permission is hereby granted, free of charge, to any person obtaining a
*    copy of this software and associated documentation files (the "Software"),
*    to deal in the Software without restriction, including without limitation
*    the rights to use, copy, modify, merge, publish, distribute, sublicense,
*    and/or sell copies of the Software, and to permit persons to whom the
*    Software is furnished to do so, subject to the following conditions:
*
*    The above copyright notice and this permission notice shall be included in
*    all copies or substantial portions of the Software.
*
*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
*    DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "vsi_nn_platform.h"

#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_test.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_math.h"
#include "client/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"

vx_status VX_CALLBACK vxPre_process_grayInitializer
    (
    vx_node node,
    const vx_reference *paramObj,
    vx_uint32 paraNum
    )
{
    // Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
    vx_kernel_execution_parameters_t shaderParam = {
        2,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {1, 1, 1},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread
    vx_status       status          = VX_SUCCESS;

    vx_tensor       output             = (vx_tensor)paramObj[1];
    vx_scalar       xRatio_s           = (vx_scalar)paramObj[2];
    vx_scalar       yRatio_s           = (vx_scalar)paramObj[3];
    uint32_t        width              = 0;
    uint32_t        height             = 0;
    vx_int8         dstFixedPointPos   = 0;
    vsi_nn_type_e   dstFormat          = VSI_NN_TYPE_FLOAT16;
    vx_float32      outputScale        = 1.0f;
    vx_int32        xRatio             = 0;
    vx_int32        yRatio             = 0;
    vx_bool         enable_copy        = vx_false_e;
    vsi_nn_tensor_attr_t attr;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));

    status |= vsi_nn_vxGetTensorAttr(output, &attr);
    status |= vxCopyScalar(xRatio_s, (void*)&xRatio, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    status |= vxCopyScalar(yRatio_s, (void*)&yRatio, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    width = attr.size[0];
    height = attr.size[1];
    dstFormat = attr.dtype.vx_type;
    dstFixedPointPos = attr.dtype.fl;

    enable_copy = (vx_bool)(xRatio == (1 << 15) && yRatio == (1 << 15));

    shaderParam.globalWorkScale[0]  = enable_copy ? 16 : 4;
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.globalWorkSize[0]   = gcmALIGN((width + shaderParam.globalWorkScale[0] - 1)
        / shaderParam.globalWorkScale[0], 4);
    shaderParam.globalWorkSize[1]   = height;

    if (enable_copy)
    {
        vx_uint32 uniDataMeanStddevLo_2x8[16] = {
            0x99999999, // TCfg
            0x44444444, // ASelt
            0x03020100, 0x07060504, // ABin
            0x99999999, // BSelt
            0x06060606, 0x06060606, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000 // Constant
        };
        vx_uint32 uniDataMeanStddevHi_2x8[16] = {
            0x99999999, // TCfg
            0x44444444, // ASelt
            0x0b0a0908, 0x0f0e0d0c, // ABin
            0x99999999, // BSelt
            0x06060606, 0x06060606, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000 // Constant
        };

        if (dstFormat == VSI_NN_TYPE_INT8 || dstFormat == VSI_NN_TYPE_INT16)
        {
            if(dstFixedPointPos > 0)
                outputScale = (vx_float32) (1 << dstFixedPointPos);
            else
            {
                outputScale = 1.0f;
                uniDataMeanStddevLo_2x8[7] |= ((-dstFixedPointPos) & 0x1F);
                uniDataMeanStddevHi_2x8[7] |= ((-dstFixedPointPos) & 0x1F);
            }
        }
        else if (dstFormat == VSI_NN_TYPE_UINT8)
        {
            vx_float32 outputZP = (vx_float32)attr.dtype.zero_point;

            outputScale = 1.0f / attr.dtype.scale;

            vxSetNodeUniform(node, "outputZP", 1, &outputZP);
        }

        vxSetNodeUniform(node, "uniDataMeanStddevLo_2x8", 1, uniDataMeanStddevLo_2x8);
        vxSetNodeUniform(node, "uniDataMeanStddevHi_2x8", 1, uniDataMeanStddevHi_2x8);
        status |= vxSetNodeUniform(node, "outputScale", 1, &outputScale);
    }
    else
    {
        vx_uint32 uniVecShift10[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00020000, 0x00060004, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000 // Constant
        };
        vx_uint32 uniAddRShift[16] = {
            0x0f0f0f0f, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002405, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniGetTempVal[16] = {
            0x09090909, // TCfg
            0x00000000, // ASelt
            0x00230001, 0x00670045, // ABin
            0x05050505, // BSelt
            0x00110000, 0x00330022, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniExtractBytes[16] = {
            0x0f0f0f0f, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002414, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniDataMulAlpha_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x01010101, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniDataSubMean_4x4[16] = {
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00007100, // AccumType, ConstantType, and PostShift
            0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000 // Constant
        };
        vx_uint32 uniConvertIntergetoF32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        };
        vx_uint32 uniExtactInteger_2x8[16] = {
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002300, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };

        if (dstFormat == VSI_NN_TYPE_FLOAT16)
        {
            status |= vxSetNodeUniform(node, "uniDataMulAlpha_4x4", 1, uniDataMulAlpha_4x4);
            status |= vxSetNodeUniform(node, "uniDataSubMean_4x4", 1, uniDataSubMean_4x4);
        }

        status |= vxSetNodeUniform(node, "uniVecShift10", 1, uniVecShift10);
        status |= vxSetNodeUniform(node, "uniAddRShift", 1, uniAddRShift);
        status |= vxSetNodeUniform(node, "uniGetTempVal", 1, uniGetTempVal);
        status |= vxSetNodeUniform(node, "uniExtractBytes", 1, uniExtractBytes);

        if (dstFormat == VSI_NN_TYPE_INT8 || dstFormat == VSI_NN_TYPE_INT16)
        {
            if(dstFixedPointPos > 0)
                outputScale *= (vx_float32) (1 << dstFixedPointPos);
            else
                outputScale *= 1.0f / (vx_float32) (1 << -dstFixedPointPos);

            status |= vxSetNodeUniform(node, "uniConvertIntergetoF32_4x4", 1, uniConvertIntergetoF32_4x4);
            status |= vxSetNodeUniform(node, "outputScale", 1, &outputScale);
            status |= vxSetNodeUniform(node, "uniExtactInteger_2x8", 1, uniExtactInteger_2x8);
        }
        else if (dstFormat == VSI_NN_TYPE_UINT8)
        {
            vx_float32 outputZP = (vx_float32)attr.dtype.zero_point;

            outputScale = 1.0f / attr.dtype.scale;

            status |= vxSetNodeUniform(node, "uniConvertIntergetoF32_4x4", 1, uniConvertIntergetoF32_4x4);
            status |= vxSetNodeUniform(node, "outputZP", 1, &outputZP);
            status |= vxSetNodeUniform(node, "outputScale", 1, &outputScale);
            status |= vxSetNodeUniform(node, "uniExtactInteger_2x8", 1, uniExtactInteger_2x8);
        }
    }

    status |= vxSetNodeAttribute(node, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));

    return status;
}


#ifdef __cplusplus
extern "C" {
#endif

static vx_param_description_t vxPre_process_grayKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};


#define PRE_PROCESS_GRAY_KERNELS(DST_TYPE) \
    vx_kernel_description_t vxPre_Process_Gray_##DST_TYPE##_Kernel = \
{ \
    VX_KERNEL_ENUM_PRE_PROCESS_GRAY, \
    VX_KERNEL_NAME_PRE_PROCESS_GRAY_##DST_TYPE, \
    NULL, \
    vxPre_process_grayKernelParam, \
    (sizeof(vxPre_process_grayKernelParam) / sizeof(vxPre_process_grayKernelParam[0])), \
    vsi_nn_KernelValidator, \
    NULL, \
    NULL, \
    vxPre_process_grayInitializer, \
    vsi_nn_KernelDeinitializer \
};

#define PRE_PROCESS_GRAY_KERNELS_COPY(DST_TYPE) \
    vx_kernel_description_t vxPre_Process_Gray_##DST_TYPE##_COPY_Kernel = \
{ \
    VX_KERNEL_ENUM_PRE_PROCESS_GRAY, \
    VX_KERNEL_NAME_PRE_PROCESS_GRAY_##DST_TYPE##_COPY, \
    NULL, \
    vxPre_process_grayKernelParam, \
    (sizeof(vxPre_process_grayKernelParam) / sizeof(vxPre_process_grayKernelParam[0])), \
    vsi_nn_KernelValidator, \
    NULL, \
    NULL, \
    vxPre_process_grayInitializer, \
    vsi_nn_KernelDeinitializer \
};

PRE_PROCESS_GRAY_KERNELS(F16)
PRE_PROCESS_GRAY_KERNELS(I16)
PRE_PROCESS_GRAY_KERNELS(I8)
PRE_PROCESS_GRAY_KERNELS(U8)

PRE_PROCESS_GRAY_KERNELS_COPY(F16)
PRE_PROCESS_GRAY_KERNELS_COPY(I16)
PRE_PROCESS_GRAY_KERNELS_COPY(I8)
PRE_PROCESS_GRAY_KERNELS_COPY(U8)

#define PRE_PROCESS_GRAY_KERNELS_NAME(DST_TYPE, COPY) \
    &vxPre_Process_Gray_##DST_TYPE##COPY##_Kernel,

vx_kernel_description_t * vx_kernel_PRE_PROCESS_GRAY_list[] =
{
    NULL,
    PRE_PROCESS_GRAY_KERNELS_NAME(F16, )
    PRE_PROCESS_GRAY_KERNELS_NAME(I16, )
    PRE_PROCESS_GRAY_KERNELS_NAME(I8, )
    PRE_PROCESS_GRAY_KERNELS_NAME(U8, )
    PRE_PROCESS_GRAY_KERNELS_NAME(F16, _COPY)
    PRE_PROCESS_GRAY_KERNELS_NAME(I16,  _COPY)
    PRE_PROCESS_GRAY_KERNELS_NAME(I8,   _COPY)
    PRE_PROCESS_GRAY_KERNELS_NAME(U8,   _COPY)
    NULL
};
#ifdef __cplusplus
}
#endif
