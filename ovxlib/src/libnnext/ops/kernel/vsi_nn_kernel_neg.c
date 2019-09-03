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

static void getFP32M0AndN(vx_float32 mult, vx_uint16 *M0, vx_int8 *N)
{
    vx_uint32 uintMult          = *((vx_uint32*)(&mult));
    vx_uint32 tmpMultiply       = 0;
    vx_int32  exp               = 0;
    vx_uint32 postShiftBit6to5  = 0;
    vx_uint32 postShift         = 0;
    vx_int8   tmpPostShift      = 0;

    tmpMultiply         = (uintMult & 0x7FFFFF) >> 8;
    *M0                 = (vx_uint16)((1U << 15) + tmpMultiply);

    exp                 = (uintMult & 0x7F800000) >> 23; /* postShift is Scale's exp*/
    tmpPostShift        = 15 - ((vx_int8)exp - 127);
    postShift           = tmpPostShift & 0x1F;
    tmpPostShift        = tmpPostShift >> 5;
    postShiftBit6to5    = tmpPostShift & 3;

    *N = (vx_int8)(((postShiftBit6to5 << 5) | (postShift & 0x1F)));
    *N = (((vx_int32)*N << 25) >> 25);
}

vx_status VX_CALLBACK vxTensorNegInitializer(vx_node nodObj, const vx_reference *paramObj, vx_uint32 paraNum)
{
// Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
#define gcmMIN(x, y)            (((x) <= (y)) ?  (x) :  (y))
#define gcmMAX(x, y)            (((x) >= (y)) ?  (x) :  (y))
#define MAX_MULTIPLIER_NUM      (65535)
#define MAX_POST_SHIFT_BITS     (31)
    vx_kernel_execution_parameters_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vx_status    status             = VX_SUCCESS;
    vx_tensor    input              = (vx_tensor)paramObj[0];
    vx_tensor    output             = (vx_tensor)paramObj[1];
    vx_uint16    M0                 = 0;
    vx_int8      postShift          = 0;
    vx_int8      srcFixPointPos     = 0;
    vx_int8      dstFixPointPos     = 0;
    vx_enum      srcFormat          = VX_TYPE_FLOAT16;
    vsi_nn_tensor_attr_t attr[2];

    memset(&attr[0], 0, sizeof(vsi_nn_tensor_attr_t));
    memset(&attr[1], 0, sizeof(vsi_nn_tensor_attr_t));

    status  = vsi_nn_vxGetTensorAttr(input, &attr[0]);
    status |= vsi_nn_vxGetTensorAttr(output, &attr[1]);

    if(status < 0)
        printf("error-%s,%d\n",__FILE__,__LINE__);

    srcFormat = attr[0].dtype.vx_type;
    srcFixPointPos = attr[0].dtype.fl;
    dstFixPointPos = attr[1].dtype.fl;

    if (srcFormat == VX_TYPE_FLOAT16 || srcFormat == VX_TYPE_INT16)
    {
        shaderParam.globalWorkScale[0]  = 8;
        shaderParam.globalWorkScale[1]  = 1;
        shaderParam.globalWorkScale[2]  = 1;
    }
    else if (srcFormat == VX_TYPE_INT8 || srcFormat == VX_TYPE_UINT8)
    {
        shaderParam.globalWorkScale[0]  = 16;
        shaderParam.globalWorkScale[1]  = 1;
        shaderParam.globalWorkScale[2]  = 1;
    }
    shaderParam.globalWorkSize[0]   = gcmALIGN((attr[1].size[0] + shaderParam.globalWorkScale[0] - 1) / shaderParam.globalWorkScale[0], 4);
    shaderParam.globalWorkSize[1]   = (attr[1].size[1] + shaderParam.globalWorkScale[1] - 1) / shaderParam.globalWorkScale[1];
    shaderParam.globalWorkSize[2]   = attr[1].size[2];


    if (srcFormat == VX_TYPE_FLOAT16)
    {
        vx_uint32 multAndoutZP[2]    = {0};
        vx_uint32 uniDataMulAndPostShift_2x8[16] = {
            0xdddddddd, // TCfg
            0x44444444, // ASelt
            0x13121110, 0x17161514, // ABin
            0x11111119, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 UniF16Neg_2x8[16] = {
            0x22222222, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };

        if (attr[1].dtype.qnt_type == VSI_NN_QNT_TYPE_DFP)
        {
            if (dstFixPointPos <= 0)
            {
                UniF16Neg_2x8[7] |= gcmMIN((-dstFixPointPos) & 0x1F, MAX_POST_SHIFT_BITS);
            }
            else
            {
                vx_uint32 lo_part    = gcmMIN((1 << dstFixPointPos), MAX_MULTIPLIER_NUM);
                vx_uint32 multiplier = lo_part;
                vx_uint32 i          = 0;

                for (i = 0; i < 8; i++)
                {
                    UniF16Neg_2x8[i + 8] = multiplier;
                }
            }
        }
        else if (attr[1].dtype.qnt_type == VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC)
        {
            if (attr[0].dtype.qnt_type != VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC)
            {
                attr[0].dtype.scale = 1.0f;
                attr[0].dtype.zero_point = 0;
            }

            getFP32M0AndN(attr[0].dtype.scale / attr[1].dtype.scale, &M0, &postShift);

            multAndoutZP[0] = (vx_uint32)(M0);
            multAndoutZP[1] = (vx_uint32)(attr[1].dtype.zero_point << postShift );

            uniDataMulAndPostShift_2x8[7] |= (postShift & 0x1F);
            status |= vxSetNodeUniform(nodObj, "multAndoutZP", 1, multAndoutZP);
            status |= vxSetNodeUniform(nodObj, "uniDataMulAndPostShift_2x8", 1, uniDataMulAndPostShift_2x8);
        }

        status |= vxSetNodeUniform(nodObj, "UniF16Neg_2x8", 1, UniF16Neg_2x8);
    }
    else if (attr[0].dtype.qnt_type == VSI_NN_QNT_TYPE_DFP)
    {
        vx_uint32 UniNegI8Hi_2x8[16] = {
            0x22222222, // TCfg
            0x00000000, // ASelt
            0x0b0a0908, 0x0f0e0d0c, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };
        vx_uint32 UniNegI8Lo_2x8[16] = {
            0x22222222, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };
        vx_uint32 UniNegI16_2x8[16] = {
            0x22222222, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };

        if (srcFixPointPos >= dstFixPointPos)
        {
            UniNegI8Lo_2x8[7] |= gcmMIN((srcFixPointPos - dstFixPointPos) & 0x3F, MAX_POST_SHIFT_BITS);
            UniNegI8Hi_2x8[7] |= gcmMIN((srcFixPointPos - dstFixPointPos) & 0x3F, MAX_POST_SHIFT_BITS);
            UniNegI16_2x8[7]  |= gcmMIN((srcFixPointPos - dstFixPointPos) & 0x3F, MAX_POST_SHIFT_BITS);
        }
        else
        {
            vx_uint32 multiplier = gcmMIN(1 << (dstFixPointPos - srcFixPointPos), MAX_MULTIPLIER_NUM);
            vx_uint32 i          = 0;

            for (i = 0; i < 8; i++)
            {
                UniNegI8Lo_2x8[i + 8] = multiplier;
                UniNegI8Hi_2x8[i + 8] = multiplier;
                UniNegI16_2x8[i + 8]  = multiplier;
            }
        }
        status |= vxSetNodeUniform(nodObj, "UniNegI8Lo_2x8", 1, UniNegI8Lo_2x8);
        status |= vxSetNodeUniform(nodObj, "UniNegI8Hi_2x8", 1, UniNegI8Hi_2x8);
        status |= vxSetNodeUniform(nodObj, "UniNegI16_2x8", 1, UniNegI16_2x8);
    }
    else if (attr[0].dtype.qnt_type == VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC)
    {
        vx_uint32 inputZP = attr[0].dtype.zero_point;
        vx_uint32 multAndoutZP[2]    = {0};
        vx_uint32 uniDataMulAndPostShift_2x8[16] = {
            0xdddddddd, // TCfg
            0x44444444, // ASelt
            0x13121110, 0x17161514, // ABin
            0x11111119, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniU8toI16Lo_2x8[16] = {
            0x99999999, // TCfg
            0x44444444, // ASelt
            0x03020100, 0x07060504, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
        };
        vx_uint32 uniU8toI16Hi_2x8[16] = {
            0x99999999, // TCfg
            0x44444444, // ASelt
            0x0b0a0908, 0x0f0e0d0c, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
        };

        if (attr[1].dtype.qnt_type != VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC)
        {
            attr[1].dtype.scale = 1.0f;
            attr[1].dtype.zero_point = 0;
        }

        getFP32M0AndN(attr[0].dtype.scale / attr[1].dtype.scale, &M0, &postShift);

        multAndoutZP[0] = (vx_uint32)(M0);
        multAndoutZP[1] = (vx_uint32)(attr[1].dtype.zero_point << postShift );

        uniDataMulAndPostShift_2x8[7] |= (postShift & 0x1F);

        status |= vxSetNodeUniform(nodObj, "inputZP", 1, &inputZP);
        status |= vxSetNodeUniform(nodObj, "multAndoutZP", 1, multAndoutZP);
        status |= vxSetNodeUniform(nodObj, "uniDataMulAndPostShift_2x8", 1, uniDataMulAndPostShift_2x8);
        status |= vxSetNodeUniform(nodObj, "uniU8toI16Lo_2x8", 1, uniU8toI16Lo_2x8);
        status |= vxSetNodeUniform(nodObj, "uniU8toI16Hi_2x8", 1, uniU8toI16Hi_2x8);
    }

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS, &shaderParam, sizeof(vx_kernel_execution_parameters_t));

    return status;
}

static vx_param_description_t vxTensorNegKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED}
};

#ifdef __cplusplus
extern "C" {
#endif

#define TENSOR_NEG_KERNELS(SRC_TYPE, DST_TYPE) \
vx_kernel_description_t vxTensorNeg_##SRC_TYPE##to##DST_TYPE##_Kernel = \
{ \
    VX_KERNEL_ENUM_NEG, \
    VX_KERNEL_NAME_TENSOR_NEG_##SRC_TYPE##TO##DST_TYPE, \
    NULL, \
    vxTensorNegKernelParam, \
    (sizeof(vxTensorNegKernelParam) / sizeof(vxTensorNegKernelParam[0])), \
    vsi_nn_KernelValidator, \
    NULL, \
    NULL, \
    vxTensorNegInitializer, \
    vsi_nn_KernelDeinitializer \
};

#define TENSOR_NEG_KERNELS_2D(SRC_TYPE, DST_TYPE) \
    vx_kernel_description_t vxTensorNeg_##SRC_TYPE##to##DST_TYPE##_2D_Kernel = \
{ \
    VX_KERNEL_ENUM_NEG, \
    VX_KERNEL_NAME_TENSOR_NEG_##SRC_TYPE##TO##DST_TYPE##_2D, \
    NULL, \
    vxTensorNegKernelParam, \
    (sizeof(vxTensorNegKernelParam) / sizeof(vxTensorNegKernelParam[0])), \
    vsi_nn_KernelValidator, \
    NULL, \
    NULL, \
    vxTensorNegInitializer, \
    vsi_nn_KernelDeinitializer \
};

TENSOR_NEG_KERNELS(F16, F16)
TENSOR_NEG_KERNELS(F16, I8)
TENSOR_NEG_KERNELS(F16, I16)
TENSOR_NEG_KERNELS(F16, U8)
TENSOR_NEG_KERNELS(I8, I8)
TENSOR_NEG_KERNELS(I8, F16)
TENSOR_NEG_KERNELS(I16, I16)
TENSOR_NEG_KERNELS(I16, F16)
TENSOR_NEG_KERNELS(U8, U8)
TENSOR_NEG_KERNELS(U8, F16)

TENSOR_NEG_KERNELS_2D(F16, F16)
TENSOR_NEG_KERNELS_2D(F16, I8)
TENSOR_NEG_KERNELS_2D(F16, I16)
TENSOR_NEG_KERNELS_2D(F16, U8)
TENSOR_NEG_KERNELS_2D(I8, I8)
TENSOR_NEG_KERNELS_2D(I8, F16)
TENSOR_NEG_KERNELS_2D(I16, I16)
TENSOR_NEG_KERNELS_2D(I16, F16)
TENSOR_NEG_KERNELS_2D(U8, U8)
TENSOR_NEG_KERNELS_2D(U8, F16)

#define TENSOR_NEG_KERENLS_NAME(SRC_TYPE, DST_TYPE, INSTR) \
    &vxTensorNeg_##SRC_TYPE##to##DST_TYPE##_##INSTR##Kernel,

vx_kernel_description_t* vx_kernel_NEG_list[] =
{
    NULL,
    TENSOR_NEG_KERENLS_NAME(F16, F16, )
    TENSOR_NEG_KERENLS_NAME(F16, I16, )
    TENSOR_NEG_KERENLS_NAME(F16, I8, )
    TENSOR_NEG_KERENLS_NAME(F16, U8, )
    TENSOR_NEG_KERENLS_NAME(I16, I16, )
    TENSOR_NEG_KERENLS_NAME(I16, F16, )
    TENSOR_NEG_KERENLS_NAME(I8,  I8, )
    TENSOR_NEG_KERENLS_NAME(I8,  F16, )
    TENSOR_NEG_KERENLS_NAME(U8,  U8, )
    TENSOR_NEG_KERENLS_NAME(U8,  F16, )
    TENSOR_NEG_KERENLS_NAME(F16, F16, 2D_)
    TENSOR_NEG_KERENLS_NAME(F16, I16, 2D_)
    TENSOR_NEG_KERENLS_NAME(F16, I8, 2D_)
    TENSOR_NEG_KERENLS_NAME(F16, U8, 2D_)
    TENSOR_NEG_KERENLS_NAME(I16, I16, 2D_)
    TENSOR_NEG_KERENLS_NAME(I16, F16, 2D_)
    TENSOR_NEG_KERENLS_NAME(I8,  I8, 2D_)
    TENSOR_NEG_KERENLS_NAME(I8,  F16, 2D_)
    TENSOR_NEG_KERENLS_NAME(U8,  U8, 2D_)
    TENSOR_NEG_KERENLS_NAME(U8,  F16, 2D_)
    NULL
};

#ifdef __cplusplus
}
#endif
