/****************************************************************************
*
*    Copyright 2012 - 2019 Vivante Corporation, Santa Clara, California.
*    All Rights Reserved.
*
*    Permission is hereby granted, free of charge, to any person obtaining
*    a copy of this software and associated documentation files (the
*    'Software'), to deal in the Software without restriction, including
*    without limitation the rights to use, copy, modify, merge, publish,
*    distribute, sub license, and/or sell copies of the Software, and to
*    permit persons to whom the Software is furnished to do so, subject
*    to the following conditions:
*
*    The above copyright notice and this permission notice (including the
*    next paragraph) shall be included in all copies or substantial
*    portions of the Software.
*
*    THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
*    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
*    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
*    IN NO EVENT SHALL VIVANTE AND/OR ITS SUPPLIERS BE LIABLE FOR ANY
*    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
*    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
*    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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

vx_status VX_CALLBACK vxTensorExpInitializer
    (
    vx_node node,
    const vx_reference *paramObj,
    vx_uint32 paraNum
    )
{
    // Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
    vx_kernel_execution_parameters_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {1, 1, 1},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread
    vx_uint32 uniExtractInteger_2x8[16] = {
        0x33333333, // TCfg
        0x11110000, // ASelt
        0x03020100, 0x03020100, // ABin
        0x00000000, // BSelt
        0x00000000, 0x00000000, // BBin
        0x00002400, // AccumType, ConstantType, and PostShift
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
    };
    vx_uint32 uniExtractHalf8_2x8[16] = {
        0x11111111, // TCfg
        0x11110000, // ASelt
        0x06040200, 0x06040200, // ABin
        0x22222222, // BSelt
        0x00000000, 0x00000000, // BBin
        0x00000100, // AccumType, ConstantType, and PostShift
        0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00,
        0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
    };
    vx_uint32 uniDatatoFp32Part0_4x4[16] = {
        0x01010101, // TCfg
        0x00000000, // ASelt
        0x00010000, 0x00030002, // ABin
        0x02020202, // BSelt
        0x00000000, 0x00000000, // BBin
        0x00000100, // AccumType, ConstantType, and PostShift
        0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
        0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
    };
    vx_uint32 uniDatatoFp32Part1_4x4[16] = {
        0x01010101, // TCfg
        0x00000000, // ASelt
        0x00050004, 0x00070006, // ABin
        0x02020202, // BSelt
        0x00000000, 0x00000000, // BBin
        0x00000100, // AccumType, ConstantType, and PostShift
        0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
        0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
    };
    vx_status       status          = VX_SUCCESS;

    vx_tensor       input              = (vx_tensor)paramObj[0];
    vx_tensor       output             = (vx_tensor)paramObj[1];
    uint32_t        width              = 0;
    uint32_t        height             = 0;
    uint32_t        depth              = 0;
    vx_int8         srcFixPointPos     = 0;
    vx_int8         dstFixPointPos     = 0;
    vsi_nn_type_e   srcFormat          = VSI_NN_TYPE_FLOAT16;
    vsi_nn_type_e   dstFormat          = VSI_NN_TYPE_FLOAT16;
    vx_float32      inputScale         = 1.0f;
    vx_float32      inputTail          = 0;
    vx_float32      outputScale        = 1.0f;
    vx_float32      outputZP           = 0;
    vx_float32      logE               = (vx_float32)(log10(exp(1.0f)) / log10(2.0f));
    vsi_nn_tensor_attr_t attr[2];

    memset(&attr[0], 0, sizeof(vsi_nn_tensor_attr_t));
    memset(&attr[1], 0, sizeof(vsi_nn_tensor_attr_t));

    status  = vsi_nn_vxGetTensorAttr(input, &attr[0]);
    status |= vsi_nn_vxGetTensorAttr(output, &attr[1]);

    width = attr[1].size[0];
    height = attr[1].size[1];
    depth = attr[1].dim_num > 2 ? attr[1].size[2] : 1;
    srcFormat = attr[0].dtype.vx_type;
    dstFormat = attr[1].dtype.vx_type;
    srcFixPointPos = attr[0].dtype.fl;
    dstFixPointPos = attr[1].dtype.fl;

    shaderParam.globalWorkScale[0]  = 8;
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.globalWorkScale[2]  = 1;
    shaderParam.globalWorkSize[0]   = gcmALIGN((width + shaderParam.globalWorkScale[0] - 1)
        / shaderParam.globalWorkScale[0], 4);
    shaderParam.globalWorkSize[1]   = height;
    shaderParam.globalWorkSize[2]   = depth;

    status |= vxSetNodeAttribute(node, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));

    if (attr[0].dtype.qnt_type == VSI_NN_QNT_TYPE_DFP &&
        (srcFormat == VSI_NN_TYPE_INT8 || srcFormat == VSI_NN_TYPE_INT16))
    {
        if (srcFixPointPos > 0)
            inputScale = 1.0f / (vx_float32) (1 << srcFixPointPos);
        else
            inputScale = (vx_float32)(1 << -srcFixPointPos);
    }
    else if (attr[0].dtype.qnt_type == VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC &&
        srcFormat == VSI_NN_TYPE_UINT8)
    {
        inputScale = attr[0].dtype.scale;
        inputTail = 0 - attr[0].dtype.zero_point * inputScale;
    }

    if (attr[1].dtype.qnt_type == VSI_NN_QNT_TYPE_DFP &&
        (dstFormat == VSI_NN_TYPE_INT8 || dstFormat == VSI_NN_TYPE_INT16))
    {
        if (dstFixPointPos > 0)
            outputScale = (vx_float32) (1 << dstFixPointPos);
        else
            outputScale = 1.0f / (vx_float32) (1 << -dstFixPointPos);
    }
    else if (attr[1].dtype.qnt_type == VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC &&
        dstFormat == VSI_NN_TYPE_UINT8)
    {
        outputScale = 1.0f / attr[1].dtype.scale;
        outputZP = (vx_float32)attr[1].dtype.zero_point;
    }

    if (dstFormat == VSI_NN_TYPE_FLOAT16)
        status |= vxSetNodeUniform(node, "uniExtract8Data_2x8", 1, uniExtractHalf8_2x8);
    else
        status |= vxSetNodeUniform(node, "uniExtract8Data_2x8", 1, uniExtractInteger_2x8);

    status |= vxSetNodeUniform(node, "logE", 1, &logE);
    status |= vxSetNodeUniform(node, "inputScale", 1, &inputScale);
    status |= vxSetNodeUniform(node, "inputTail", 1, &inputTail);
    status |= vxSetNodeUniform(node, "outputScale", 1, &outputScale);
    status |= vxSetNodeUniform(node, "outputZP", 1, &outputZP);
    status |= vxSetNodeUniform(node, "uniDatatoFp32Part0_4x4", 1, uniDatatoFp32Part0_4x4);
    status |= vxSetNodeUniform(node, "uniDatatoFp32Part1_4x4", 1, uniDatatoFp32Part1_4x4);

    return VX_SUCCESS;
}

static vx_param_description_t vxTensorExpKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED}
};


#ifdef __cplusplus
extern "C" {
#endif


#define TENSOR_EXP_KERNELS(SRC_TYPE, DST_TYPE) \
    vx_kernel_description_t vxTensorExp_##SRC_TYPE##to##DST_TYPE##_Kernel = \
{ \
    VX_KERNEL_ENUM_TENSOR_EXP, \
    VX_KERNEL_NAME_TENSOR_EXP_##SRC_TYPE##TO##DST_TYPE, \
    NULL, \
    vxTensorExpKernelParam, \
    (sizeof(vxTensorExpKernelParam) / sizeof(vxTensorExpKernelParam[0])), \
    vsi_nn_KernelValidator, \
    NULL, \
    NULL, \
    vxTensorExpInitializer, \
    vsi_nn_KernelDeinitializer \
};

#define TENSOR_EXP_KERNELS_2D(SRC_TYPE, DST_TYPE) \
    vx_kernel_description_t vxTensorExp_##SRC_TYPE##to##DST_TYPE##_2D_Kernel = \
{ \
    VX_KERNEL_ENUM_TENSOR_EXP, \
    VX_KERNEL_NAME_TENSOR_EXP_##SRC_TYPE##TO##DST_TYPE##_2D, \
    NULL, \
    vxTensorExpKernelParam, \
    (sizeof(vxTensorExpKernelParam) / sizeof(vxTensorExpKernelParam[0])), \
    vsi_nn_KernelValidator, \
    NULL, \
    NULL, \
    vxTensorExpInitializer, \
    vsi_nn_KernelDeinitializer \
};

TENSOR_EXP_KERNELS(F16, F16)
TENSOR_EXP_KERNELS(F16, I8)
TENSOR_EXP_KERNELS(F16, I16)
TENSOR_EXP_KERNELS(F16, U8)
TENSOR_EXP_KERNELS(I8, I8)
TENSOR_EXP_KERNELS(I8, F16)
TENSOR_EXP_KERNELS(I16, I16)
TENSOR_EXP_KERNELS(I16, F16)
TENSOR_EXP_KERNELS(U8, U8)
TENSOR_EXP_KERNELS(U8, F16)
TENSOR_EXP_KERNELS_2D(F16, F16)
TENSOR_EXP_KERNELS_2D(F16, I8)
TENSOR_EXP_KERNELS_2D(F16, I16)
TENSOR_EXP_KERNELS_2D(F16, U8)
TENSOR_EXP_KERNELS_2D(I8, I8)
TENSOR_EXP_KERNELS_2D(I8, F16)
TENSOR_EXP_KERNELS_2D(I16, I16)
TENSOR_EXP_KERNELS_2D(I16, F16)
TENSOR_EXP_KERNELS_2D(U8, U8)
TENSOR_EXP_KERNELS_2D(U8, F16)

#define TENSOR_EXP_KERENLS_NAME(SRC_TYPE, DST_TYPE, INSTR) \
    &vxTensorExp_##SRC_TYPE##to##DST_TYPE##_##INSTR##Kernel,

vx_kernel_description_t * vx_kernel_EXP_list[] =
{
    NULL,
    TENSOR_EXP_KERENLS_NAME(F16, F16, )
    TENSOR_EXP_KERENLS_NAME(F16, I16, )
    TENSOR_EXP_KERENLS_NAME(F16, I8, )
    TENSOR_EXP_KERENLS_NAME(F16, U8, )
    TENSOR_EXP_KERENLS_NAME(I16, I16, )
    TENSOR_EXP_KERENLS_NAME(I16, F16, )
    TENSOR_EXP_KERENLS_NAME(I8,  I8, )
    TENSOR_EXP_KERENLS_NAME(I8,  F16, )
    TENSOR_EXP_KERENLS_NAME(U8,  U8, )
    TENSOR_EXP_KERENLS_NAME(U8,  F16, )

    TENSOR_EXP_KERENLS_NAME(F16, F16, 2D_)
    TENSOR_EXP_KERENLS_NAME(F16, I16, 2D_)
    TENSOR_EXP_KERENLS_NAME(F16, I8, 2D_)
    TENSOR_EXP_KERENLS_NAME(F16, U8, 2D_)
    TENSOR_EXP_KERENLS_NAME(I16, I16, 2D_)
    TENSOR_EXP_KERENLS_NAME(I16, F16, 2D_)
    TENSOR_EXP_KERENLS_NAME(I8,  I8, 2D_)
    TENSOR_EXP_KERENLS_NAME(I8,  F16, 2D_)
    TENSOR_EXP_KERENLS_NAME(U8,  U8, 2D_)
    TENSOR_EXP_KERENLS_NAME(U8,  F16, 2D_)

    NULL
};
#ifdef __cplusplus
}
#endif
