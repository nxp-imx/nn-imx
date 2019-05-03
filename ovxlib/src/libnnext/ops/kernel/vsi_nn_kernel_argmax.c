/****************************************************************************
*
*    Copyright (c) 2018 Vivante Corporation
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
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_math.h"
#include "vsi_nn_log.h"
#include "client/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"

vsi_status VX_CALLBACK vxargMaxInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    uint32_t paraNum
    )
{
// Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
    vx_kernel_execution_parameters_t shaderParam = {
        2,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vx_status status = VX_SUCCESS;

    vx_tensor tensor = (vx_tensor)paramObj[0];
    vx_tensor outTensor = (vx_tensor)paramObj[1];
    vx_uint32 dims = 0;
    vx_uint32 size[DIM_SIZE] = {0};
    vx_enum dataType, outDataType;

    status = vxQueryTensor(tensor, VX_TENSOR_NUM_OF_DIMS, &dims, sizeof(vx_uint32));
    status |= vxQueryTensor(tensor, VX_TENSOR_DIMS, size, sizeof(vx_uint32) * dims);
    status |= vxQueryTensor(tensor, VX_TENSOR_DATA_TYPE, &dataType, sizeof(dataType));
    status |= vxQueryTensor(outTensor, VX_TENSOR_DATA_TYPE, &outDataType, sizeof(outDataType));

    if(VX_SUCCESS != status)
        return status;

    if((dataType == VX_TYPE_INT8 && outDataType == VX_TYPE_INT8)
        || (dataType == VX_TYPE_UINT8 && outDataType == VX_TYPE_INT16 && size[2] <= 256)
        || (dataType == VX_TYPE_INT8 && outDataType == VX_TYPE_INT16 && size[2] <= 256)
        || (dataType == VX_TYPE_UINT8 && outDataType == VX_TYPE_UINT8))
    {
        shaderParam.globalWorkOffset[0] = 0;
        shaderParam.globalWorkOffset[1] = 0;
        shaderParam.globalWorkScale[0]  = 16;
        shaderParam.globalWorkScale[1]  = 1;
        shaderParam.globalWorkSize[0]   = gcmALIGN((size[0] + shaderParam.globalWorkScale[0] - 1)
            / shaderParam.globalWorkScale[0], 4);
        shaderParam.globalWorkSize[1]   = (size[1] + shaderParam.globalWorkScale[1] - 1)
            / shaderParam.globalWorkScale[1];
    }
    else
    {
        shaderParam.globalWorkOffset[0] = 0;
        shaderParam.globalWorkOffset[1] = 0;
        shaderParam.globalWorkScale[0]  = 8;
        shaderParam.globalWorkScale[1]  = 1;
        shaderParam.globalWorkSize[0]   = gcmALIGN((size[0] + shaderParam.globalWorkScale[0] - 1)
            / shaderParam.globalWorkScale[0], 4);
        shaderParam.globalWorkSize[1]   = (size[1] + shaderParam.globalWorkScale[1] - 1)
            / shaderParam.globalWorkScale[1];
    }

    if ((dataType == VX_TYPE_UINT8 && outDataType == VX_TYPE_INT16 && size[2] <= 256)
      ||(dataType == VX_TYPE_INT8 && outDataType == VX_TYPE_INT16 && size[2] <= 256))
    {
        vx_uint32 uniPacekedU8toI16Lo_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };
        vx_uint32 uniPacekedU8toI16Hi_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x0b0a0908, 0x0f0e0d0c, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };
        vx_uint32 depthsub1 = size[2] - 1;
        vx_uint32 packedD = (size[2] << 24) | (size[2] << 16) | (size[2] << 8) | size[2];
        vx_uint32 packedDepth[4] = {packedD, packedD, packedD, packedD};

        status |= vxSetNodeUniform(nodObj, "uniPacekedU8toI16Lo_2x8", 1, uniPacekedU8toI16Lo_2x8);
        status |= vxSetNodeUniform(nodObj, "uniPacekedU8toI16Hi_2x8", 1, uniPacekedU8toI16Hi_2x8);
        status |= vxSetNodeUniform(nodObj, "packedDepth", 1, packedDepth);
        status |= vxSetNodeUniform(nodObj, "depthsub1", 1, &depthsub1);
    }
    else
    {
        // uniforms
        vx_uint32 intToShort8[16] = {
            0x33333333, // TCfg
            0x00000000, // ASelt
            0x00000000, 0x00000000, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        if(dataType == VX_TYPE_UINT8 || dataType == VX_TYPE_INT16)
            status |= vxSetNodeUniform(nodObj, "intToShort8_2", 1, intToShort8);
        else
            status |= vxSetNodeUniform(nodObj, "intToShort8", 1, intToShort8);

        if(dataType == VX_TYPE_UINT8 || dataType == VX_TYPE_INT16)
            status |= vxSetNodeUniform(nodObj, "depth2", 1, &size[2]);
        else
            status |= vxSetNodeUniform(nodObj, "depth", 1, &size[2]);
    }

    if(dataType == VX_TYPE_FLOAT16 && outDataType == VX_TYPE_INT16)
    {
        vx_uint32 uniExtractHalfMax_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x00000000, 0x00000000, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
        };
        status |= vxSetNodeUniform(nodObj, "uniExtractHalfMax_2x8", 1, uniExtractHalfMax_2x8);
    }

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));

    if(status < 0)
    {
        VSILOGE("Initializer failure!(argMax)\n");
    }
    return status;
}

static vx_param_description_t vxargMaxKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED}
};
#ifdef __cpluplus
extern "C" {
#endif
vx_kernel_description_t vxargMaxKernelInfo =
{
    VX_KERNEL_ENUM_ARGMAX,
    VX_KERNEL_NAME_ARGMAX,
    NULL,
    vxargMaxKernelParam,
    (sizeof(vxargMaxKernelParam) / sizeof(vxargMaxKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxargMaxInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxargMaxKernelInfoInt8 =
{
    VX_KERNEL_ENUM_ARGMAX,
    VX_KERNEL_NAME_ARGMAX_INT8,
    NULL,
    vxargMaxKernelParam,
    (sizeof(vxargMaxKernelParam) / sizeof(vxargMaxKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxargMaxInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxargMaxKernelInfoUint8 =
{
    VX_KERNEL_ENUM_ARGMAX,
    VX_KERNEL_NAME_ARGMAX_UINT8,
    NULL,
    vxargMaxKernelParam,
    (sizeof(vxargMaxKernelParam) / sizeof(vxargMaxKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxargMaxInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxargMaxKernelInfoInt16 =
{
    VX_KERNEL_ENUM_ARGMAX,
    VX_KERNEL_NAME_ARGMAX_INT16,
    NULL,
    vxargMaxKernelParam,
    (sizeof(vxargMaxKernelParam) / sizeof(vxargMaxKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxargMaxInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxargMaxKernelInfoUint8_Int16 =
{
    VX_KERNEL_ENUM_ARGMAX,
    VX_KERNEL_NAME_ARGMAX_UINT8_INT16,
    NULL,
    vxargMaxKernelParam,
    (sizeof(vxargMaxKernelParam) / sizeof(vxargMaxKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxargMaxInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxargMaxKernelInfoU8_Int16_WXHX256 =
{
    VX_KERNEL_ENUM_ARGMAX,
    VX_KERNEL_NAME_ARGMAX_U8_I16_WXHX256,
    NULL,
    vxargMaxKernelParam,
    (sizeof(vxargMaxKernelParam) / sizeof(vxargMaxKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxargMaxInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxargMaxKernelInfoI8_I16_WXHX256 =
{
    VX_KERNEL_ENUM_ARGMAX,
    VX_KERNEL_NAME_ARGMAX_I8_I16_WXHX256,
    NULL,
    vxargMaxKernelParam,
    (sizeof(vxargMaxKernelParam) / sizeof(vxargMaxKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxargMaxInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxargMaxKernelInfoInt8_Int16 =
{
    VX_KERNEL_ENUM_ARGMAX,
    VX_KERNEL_NAME_ARGMAX_INT8_INT16,
    NULL,
    vxargMaxKernelParam,
    (sizeof(vxargMaxKernelParam) / sizeof(vxargMaxKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxargMaxInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_ARGMAX_list[] =
{
    NULL,
    &vxargMaxKernelInfo,
    &vxargMaxKernelInfoInt8,
    &vxargMaxKernelInfoUint8,
    &vxargMaxKernelInfoInt16,
    &vxargMaxKernelInfoUint8_Int16,
    &vxargMaxKernelInfoInt8_Int16,
    &vxargMaxKernelInfoU8_Int16_WXHX256,
    &vxargMaxKernelInfoI8_I16_WXHX256,
    NULL
};
#ifdef __cpluplus
}
#endif
