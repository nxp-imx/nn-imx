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

vsi_status VX_CALLBACK vxeltwiseMaxValidator
    (
    vx_node node,
    const vx_reference parameters[],
    uint32_t num,
    vx_meta_format metas[]
    )
{
    vsi_status status = VX_SUCCESS;
    uint32_t index = 0;
#ifndef UINT16_MAX
#define UINT16_MAX  ((unsigned short)0xffff)
#endif
    for(index = 0; index < num; index++)
    {
        // Validator
        if(index == 0) //tensor
        {
            vx_tensor tensor = (vx_tensor)parameters[index];
            if(tensor != NULL)
            {
                vsi_enum       data_format;

                // data_format
                status |= vxQueryTensor(tensor, VX_TENSOR_DATA_TYPE, &data_format,
                    sizeof(data_format));
                if (data_format != VX_TYPE_FLOAT16 && data_format != VX_TYPE_INT8
                    && data_format != VX_TYPE_INT16 && data_format != VX_TYPE_UINT8)
                    status |= VX_ERROR_INVALID_TYPE;

            }
            else
            {
                VSILOGW("[%s : %d] Warning parameter %d is NULL \n", __FILE__,__LINE__,index);
            }

        }
        else if(index == 1) //tensor
        {
            vx_tensor tensor = (vx_tensor)parameters[index];
            if(tensor != NULL)
            {
                vsi_enum       data_format;

                // data_format
                status |= vxQueryTensor(tensor, VX_TENSOR_DATA_TYPE, &data_format,
                    sizeof(data_format));
                if (data_format != VX_TYPE_FLOAT16 && data_format != VX_TYPE_INT8
                    && data_format != VX_TYPE_INT16 && data_format != VX_TYPE_UINT8)
                    status |= VX_ERROR_INVALID_TYPE;

            }
            else
            {
                VSILOGW("[%s : %d] Warning parameter %d is NULL \n", __FILE__,__LINE__,index);
            }

        }
        else if(index == 2) //tensor
        {
            vx_tensor tensor = (vx_tensor)parameters[index];
            if(tensor != NULL)
            {
                vsi_enum      data_format;

                // data_format
                status |= vxQueryTensor(tensor, VX_TENSOR_DATA_TYPE, &data_format,
                    sizeof(data_format));
                if (data_format != VX_TYPE_FLOAT16 && data_format != VX_TYPE_INT8
                    && data_format != VX_TYPE_INT16 && data_format != VX_TYPE_UINT8)
                    status |= VX_ERROR_INVALID_TYPE;

            }
            else
            {
                VSILOGW("[%s : %d] Warning parameter %d is NULL \n", __FILE__,__LINE__,index);
            }
        }

        else
        {
            VSILOGE("[%s : %d] Validator  failure! invalid index = %d\n", __FILE__,__LINE__,index);
        }

        if(status < 0)
        {
            VSILOGE("[%s : %d] Validator  failure! index = %d\n",__FILE__, __LINE__,index);
            break;
        }
    }
    return status;
}

#define MAX_MULTIPLIER_NUM      (65535)
#define MAX_POST_SHIFT_BITS     (31)

vsi_status VX_CALLBACK vxeltwiseMaxInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    uint32_t paraNum
    )
{
    vsi_status status = VX_SUCCESS;
    // Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
    vx_kernel_execution_parameters_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vx_tensor input0    = (vx_tensor)paramObj[0];
    vx_tensor input1    = (vx_tensor)paramObj[1];
    vx_tensor output    = (vx_tensor)paramObj[2];

    uint32_t size[DIM_SIZE] = {0};
    vsi_enum src0Type;
    vsi_enum src1Type;
    vsi_enum dstType;
    int8_t src0FixPointPos  = 0;
    int8_t src1FixPointPos  = 0;
    int8_t dstFixPointPos   = 0;
    int32_t output_ZP = 0;
    int32_t input0_ZP = 0;
    int32_t input1_ZP = 0;
    float scaleIn0 = 1.0;
    float scaleIn1 = 1.0;
    float scaleOut = 1.0;

    status |= vxQueryTensor(input0, VX_TENSOR_DIMS, size, sizeof(size));
    status |= vxQueryTensor(input0, VX_TENSOR_DATA_TYPE, &src0Type, sizeof(src0Type));
    status |= vxQueryTensor(input1, VX_TENSOR_DATA_TYPE, &src1Type, sizeof(src1Type));
    status |= vxQueryTensor(output, VX_TENSOR_DATA_TYPE, &dstType, sizeof(dstType));
    status |= vxQueryTensor(input0, VX_TENSOR_FIXED_POINT_POS,
        &src0FixPointPos, sizeof(src0FixPointPos));
    status |= vxQueryTensor(input1, VX_TENSOR_FIXED_POINT_POS,
        &src1FixPointPos, sizeof(src1FixPointPos));
    status |= vxQueryTensor(output, VX_TENSOR_FIXED_POINT_POS,
        &dstFixPointPos, sizeof(dstFixPointPos));
    status |= vxQueryTensor(input0, VX_TENSOR_ZERO_POINT, &input0_ZP, sizeof(input0_ZP));
    status |= vxQueryTensor(input1, VX_TENSOR_ZERO_POINT, &input1_ZP, sizeof(input1_ZP));
    status |= vxQueryTensor(output, VX_TENSOR_ZERO_POINT, &output_ZP, sizeof(output_ZP));
    status |= vxQueryTensor(input0, VX_TENSOR_SCALE, &scaleIn0, sizeof(scaleIn0));
    status |= vxQueryTensor(input1, VX_TENSOR_SCALE, &scaleIn1, sizeof(scaleIn1));
    status |= vxQueryTensor(output, VX_TENSOR_SCALE, &scaleOut, sizeof(scaleOut));

    if(VX_SUCCESS != status)
        return status;

    if (size[2] == 0)
    {
        size[2] = 1;
    }

    if ((src0Type == VX_TYPE_UINT8 && src1Type == VX_TYPE_UINT8 && dstType == VX_TYPE_UINT8)
     || (src0Type == VX_TYPE_INT8 && src1Type == VX_TYPE_INT8 && dstType == VX_TYPE_INT8)
     || (src0Type == VX_TYPE_INT8 && src1Type == VX_TYPE_FLOAT16 && dstType == VX_TYPE_INT8)
     || (src0Type == VX_TYPE_UINT8 && src1Type == VX_TYPE_FLOAT16 && dstType == VX_TYPE_UINT8))
    {
        shaderParam.globalWorkScale[0]  = 16;
        shaderParam.globalWorkScale[1]  = 1;
        shaderParam.globalWorkScale[2]  = 1;
    }
    else
    {
        shaderParam.globalWorkScale[0]  = 8;
        shaderParam.globalWorkScale[1]  = 1;
        shaderParam.globalWorkScale[2]  = 1;
    }

    shaderParam.globalWorkSize[0]   = gcmALIGN((size[0] + shaderParam.globalWorkScale[0] - 1) /
        shaderParam.globalWorkScale[0], 4);
    shaderParam.globalWorkSize[1]   = (size[1] + shaderParam.globalWorkScale[1] - 1) /
        shaderParam.globalWorkScale[1];
    shaderParam.globalWorkSize[2]   = size[2];

    if((src0Type == VX_TYPE_INT8 && src1Type == VX_TYPE_INT8 && dstType == VX_TYPE_INT8
        && (!(src0FixPointPos == dstFixPointPos && src1FixPointPos == dstFixPointPos)))
        || (src0Type == VX_TYPE_INT8 && src1Type == VX_TYPE_FLOAT16 && dstType == VX_TYPE_INT8))
    {
        vx_uint32 idx = 0;
        vx_uint32 uniDFP8toDFP8Lo_2x8_0[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };
        vx_uint32 uniDFP8toDFP8Hi_2x8_0[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x0b0a0908, 0x0f0e0d0c, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };
        vx_uint32 uniDFP8toDFP8Lo_2x8_1[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };
        vx_uint32 uniDFP8toDFP8Hi_2x8_1[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x0b0a0908, 0x0f0e0d0c, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };

        if (src0FixPointPos >= dstFixPointPos)
        {
            vx_uint8 postshift = vsi_nn_min(src0FixPointPos - dstFixPointPos, MAX_POST_SHIFT_BITS);

            uniDFP8toDFP8Lo_2x8_0[7] |= (postshift & 0x1F);
            uniDFP8toDFP8Hi_2x8_0[7] |= (postshift & 0x1F);
        }
        else
        {
            for (idx = 8; idx < 16; idx++)
            {
                uniDFP8toDFP8Lo_2x8_0[idx] = vsi_nn_min(1 << (dstFixPointPos - src0FixPointPos), MAX_MULTIPLIER_NUM);
                uniDFP8toDFP8Hi_2x8_0[idx] = vsi_nn_min(1 << (dstFixPointPos - src0FixPointPos), MAX_MULTIPLIER_NUM);
            }
        }

        if (src1FixPointPos >= dstFixPointPos)
        {
            vx_uint8 postshift = vsi_nn_min(src1FixPointPos - dstFixPointPos, MAX_POST_SHIFT_BITS);

            uniDFP8toDFP8Lo_2x8_1[7] |= (postshift & 0x1F);
            uniDFP8toDFP8Hi_2x8_1[7] |= (postshift & 0x1F);
        }
        else
        {
            for (idx = 8; idx < 16; idx++)
            {
                uniDFP8toDFP8Lo_2x8_1[idx] = vsi_nn_min(1 << (dstFixPointPos - src1FixPointPos), MAX_MULTIPLIER_NUM);
                uniDFP8toDFP8Hi_2x8_1[idx] = vsi_nn_min(1 << (dstFixPointPos - src1FixPointPos), MAX_MULTIPLIER_NUM);
            }
        }

        status |= vxSetNodeUniform(nodObj, "uniDFP8toDFP8Lo_2x8_0", 1, uniDFP8toDFP8Lo_2x8_0);
        status |= vxSetNodeUniform(nodObj, "uniDFP8toDFP8Hi_2x8_0", 1, uniDFP8toDFP8Hi_2x8_0);

        if(src1Type == VX_TYPE_FLOAT16)
        {
            vx_uint32 uinConvertFp16ToInt8_2x8[16] = {
                0x11111111, // TCfg
                0x00000000, // ASelt
                0x03020100, 0x07060504, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000300, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001,
                0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
            };

            if (0 >= dstFixPointPos)
            {
                vx_uint8 postshift = vsi_nn_min(0 - dstFixPointPos, MAX_POST_SHIFT_BITS);

                uinConvertFp16ToInt8_2x8[7] |= (postshift & 0x1F);
            }
            else
            {
                for (idx = 8; idx < 16; idx++)
                {
                    uinConvertFp16ToInt8_2x8[idx] = vsi_nn_min(1 << (dstFixPointPos - 0), MAX_MULTIPLIER_NUM);
                }
            }

            status |= vxSetNodeUniform(nodObj, "uinConvertFp16ToInt8_2x8", 1, uinConvertFp16ToInt8_2x8);
        }
        else
        {
            status |= vxSetNodeUniform(nodObj, "uniDFP8toDFP8Lo_2x8_1", 1, uniDFP8toDFP8Lo_2x8_1);
            status |= vxSetNodeUniform(nodObj, "uniDFP8toDFP8Hi_2x8_1", 1, uniDFP8toDFP8Hi_2x8_1);
        }
    }
    else if ((src0Type == VX_TYPE_UINT8 && src1Type == VX_TYPE_UINT8 && dstType == VX_TYPE_UINT8)
            || (src0Type == VX_TYPE_UINT8 && src1Type == VX_TYPE_FLOAT16 && dstType == VX_TYPE_UINT8))
    {
        vx_uint16   M0                      = 0;
        vx_int8     postShift0              = 0;
        vx_uint16   M1                      = 0;
        vx_int8     postShift1              = 0;
        vx_uint32  multAndoutZP0[2]         = {0};
        vx_uint32  multAndoutZP1[2]         = {0};

        vx_uint32 uniU8MulAndPostShift_Lo_2x8[16] = {
            0xdddddddd, // TCfg
            0x44444444, // ASelt
            0x13121110, 0x17161514, // ABin
            0x11111111, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniU8MulAndPostShift_Hi_2x8[16] = {
            0xdddddddd, // TCfg
            0x44444444, // ASelt
            0x1b1a1918, 0x1f1e1d1c, // ABin
            0x11111111, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };

        vsi_nn_GetFP32MultiAndPostShift(scaleIn0 / scaleOut, &M0, &postShift0);
        vsi_nn_GetFP32MultiAndPostShift(scaleIn1 / scaleOut, &M1, &postShift1);

        multAndoutZP0[0] = (vx_uint32)(M0);
        multAndoutZP0[1] = (vx_uint32)((output_ZP << postShift0) - input0_ZP * M0);
        multAndoutZP1[0] = (vx_uint32)(M1);
        multAndoutZP1[1] = (vx_uint32)((output_ZP << postShift1) - input1_ZP * M1);

        uniU8MulAndPostShift_Lo_2x8[7] = 0x00002600 | (postShift0 & 0x1F);
        uniU8MulAndPostShift_Hi_2x8[7] = 0x00002600 | (postShift0 & 0x1F);

        status |= vxSetNodeUniform(nodObj, "uniU8MulAndPostShift0_Lo_2x8",  1, uniU8MulAndPostShift_Lo_2x8);
        status |= vxSetNodeUniform(nodObj, "uniU8MulAndPostShift0_Hi_2x8",  1, uniU8MulAndPostShift_Hi_2x8);
        status |= vxSetNodeUniform(nodObj, "multAndoutZP0",  1, multAndoutZP0);
        status |= vxSetNodeUniform(nodObj, "multAndoutZP1",  1, multAndoutZP1);

        if(src1Type == VX_TYPE_FLOAT16)
        {
            vx_uint32 uniConvertFp16toU8_2x8[16] = {
                0xdddddddd, // TCfg
                0x44444444, // ASelt
                0x13121110, 0x17161514, // ABin
                0x11111111, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002600, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            };

            uniConvertFp16toU8_2x8[7] = 0x00002600 | (postShift1 & 0x1F);
            status |= vxSetNodeUniform(nodObj, "uniConvertFp16toU8_2x8",  1, uniConvertFp16toU8_2x8);
        }
        else
        {
            uniU8MulAndPostShift_Lo_2x8[7] = 0x00002600 | (postShift1 & 0x1F);
            uniU8MulAndPostShift_Hi_2x8[7] = 0x00002600 | (postShift1 & 0x1F);
            status |= vxSetNodeUniform(nodObj, "uniU8MulAndPostShift1_Lo_2x8",  1, uniU8MulAndPostShift_Lo_2x8);
            status |= vxSetNodeUniform(nodObj, "uniU8MulAndPostShift1_Hi_2x8",  1, uniU8MulAndPostShift_Hi_2x8);
        }
    }
    else if (src0Type == VX_TYPE_INT16 && src1Type == VX_TYPE_INT16 && dstType == VX_TYPE_INT16)
    {
        vx_uint32 idx = 0;
        vx_uint32 uniDFP16toDFP16_2x8_0[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };
        vx_uint32 uniDFP16toDFP16_2x8_1[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };

        if (src0FixPointPos >= dstFixPointPos)
        {
            uniDFP16toDFP16_2x8_0[7] |= ((src0FixPointPos - dstFixPointPos) & 0x1F);
        }
        else
        {
            for (idx = 8; idx < 16; idx++)
            {
                uniDFP16toDFP16_2x8_0[idx] = vsi_nn_min(1 << (dstFixPointPos - src0FixPointPos), MAX_MULTIPLIER_NUM);
            }
        }

        if (src1FixPointPos >= dstFixPointPos)
        {
            vx_uint8 postshift = vsi_nn_min(src0FixPointPos - dstFixPointPos, MAX_POST_SHIFT_BITS);

            uniDFP16toDFP16_2x8_1[7] |= (postshift & 0x1F);
        }
        else
        {
            for (idx = 8; idx < 16; idx++)
            {
                uniDFP16toDFP16_2x8_1[idx] = vsi_nn_min(1 << (dstFixPointPos - src1FixPointPos), MAX_MULTIPLIER_NUM);
            }
        }

        status |= vxSetNodeUniform(nodObj, "uniDFP16toDFP16_2x8_0", 1, uniDFP16toDFP16_2x8_0);
        status |= vxSetNodeUniform(nodObj, "uniDFP16toDFP16_2x8_1", 1, uniDFP16toDFP16_2x8_1);
    }
    else if(src0Type == VX_TYPE_INT16 && src1Type == VX_TYPE_FLOAT16 && dstType == VX_TYPE_INT16)
    {
        vx_uint32 uniConvertI16toI16_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };
        vx_uint32 uinConvertFp16ToInt16_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };

        if (src0FixPointPos > dstFixPointPos)
        {
            vx_uint8  postshift = vsi_nn_min(src0FixPointPos - dstFixPointPos, MAX_POST_SHIFT_BITS);

            uniConvertI16toI16_2x8[7] |= (postshift & 0x1F);
        }
        else
        {
            vx_uint32 multiplier = vsi_nn_min(1 << (dstFixPointPos - src0FixPointPos), MAX_MULTIPLIER_NUM);
            vx_uint32 i          = 0;

            for (i = 0; i < 8; i++)
            {
                uniConvertI16toI16_2x8[i + 8] = multiplier;
            }
        }

        if (0 > dstFixPointPos)
        {
            vx_uint8  postshift      = vsi_nn_min(0 - dstFixPointPos, MAX_POST_SHIFT_BITS);
            uinConvertFp16ToInt16_2x8[7] |= (postshift & 0x1F);
        }
        else
        {
            vx_uint32 multiplier = vsi_nn_min(1 << (dstFixPointPos - 0), MAX_MULTIPLIER_NUM);
            vx_uint32 i          = 0;

            for (i = 0; i < 8; i++)
            {
                uinConvertFp16ToInt16_2x8[i + 8] = multiplier;
            }
        }

        status |= vxSetNodeUniform(nodObj, "uniConvertI16toI16_2x8", 1, uniConvertI16toI16_2x8);
        status |= vxSetNodeUniform(nodObj, "uinConvertFp16ToInt16_2x8", 1, uinConvertFp16ToInt16_2x8);
    }
    else if(src0Type == VX_TYPE_INT16 && src1Type == VX_TYPE_FLOAT16 && dstType == VX_TYPE_FLOAT16)
    {
        vx_uint32 uniConvertInt16toFp16_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };

        if (src0FixPointPos > 0)
        {
            vx_uint8  postshift = vsi_nn_min(src0FixPointPos, MAX_POST_SHIFT_BITS);
            uniConvertInt16toFp16_2x8[7] |= (postshift & 0x1F);
        }
        else
        {
            vx_uint32 multiplier = vsi_nn_min(1 << (0 - src0FixPointPos), MAX_MULTIPLIER_NUM);
            vx_uint32 i          = 0;

            for (i = 0; i < 8; i++)
            {
                uniConvertInt16toFp16_2x8[i + 8] = multiplier;
            }
        }

        status |= vxSetNodeUniform(nodObj, "uniConvertInt16toFp16_2x8", 1, uniConvertInt16toFp16_2x8);
    }

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));
    if(status < 0)
    {
        VSILOGE("Initializer  failure! \n");
    }
    return status;
}

static vx_param_description_t vxeltwiseMaxKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED}
};

#ifdef __cplusplus
extern "C" {
#endif
vx_kernel_description_t vxeltwiseMaxKernelInfo =
{
    VX_KERNEL_ENUM_ELTWISE_MAX,
    VX_KERNEL_NAME_ELTWISE_MAX,
    NULL,
    vxeltwiseMaxKernelParam,
    (sizeof(vxeltwiseMaxKernelParam) / sizeof(vxeltwiseMaxKernelParam[0])),
    vxeltwiseMaxValidator,
    NULL,
    NULL,
    vxeltwiseMaxInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxeltwiseMaxKernelInfo_int8 =
{
    VX_KERNEL_ENUM_ELTWISE_MAX,
    VX_KERNEL_NAME_ELTWISE_MAX_INT8,
    NULL,
    vxeltwiseMaxKernelParam,
    (sizeof(vxeltwiseMaxKernelParam) / sizeof(vxeltwiseMaxKernelParam[0])),
    vxeltwiseMaxValidator,
    NULL,
    NULL,
    vxeltwiseMaxInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxeltwiseMaxKernelInfo_int8_nofl =
{
    VX_KERNEL_ENUM_ELTWISE_MAX,
    VX_KERNEL_NAME_ELTWISE_MAX_INT8_NOFL,
    NULL,
    vxeltwiseMaxKernelParam,
    (sizeof(vxeltwiseMaxKernelParam) / sizeof(vxeltwiseMaxKernelParam[0])),
    vxeltwiseMaxValidator,
    NULL,
    NULL,
    vxeltwiseMaxInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxeltwiseMaxKernelInfo_int16 =
{
    VX_KERNEL_ENUM_ELTWISE_MAX,
    VX_KERNEL_NAME_ELTWISE_MAX_INT16,
    NULL,
    vxeltwiseMaxKernelParam,
    (sizeof(vxeltwiseMaxKernelParam) / sizeof(vxeltwiseMaxKernelParam[0])),
    vxeltwiseMaxValidator,
    NULL,
    NULL,
    vxeltwiseMaxInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxeltwiseMaxKernelInfo_uint8 =
{
    VX_KERNEL_ENUM_ELTWISE_MAX,
    VX_KERNEL_NAME_ELTWISE_MAX_UINT8,
    NULL,
    vxeltwiseMaxKernelParam,
    (sizeof(vxeltwiseMaxKernelParam) / sizeof(vxeltwiseMaxKernelParam[0])),
    vxeltwiseMaxValidator,
    NULL,
    NULL,
    vxeltwiseMaxInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxeltwiseMaxKernelInfo_i8fp16_i8 =
{
    VX_KERNEL_ENUM_ELTWISE_MAX,
    VX_KERNEL_NAME_ELTWISE_MAX_INT8FP16_INT8,
    NULL,
    vxeltwiseMaxKernelParam,
    (sizeof(vxeltwiseMaxKernelParam) / sizeof(vxeltwiseMaxKernelParam[0])),
    vxeltwiseMaxValidator,
    NULL,
    NULL,
    vxeltwiseMaxInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxeltwiseMaxKernelInfo_u8fp16_u8 =
{
    VX_KERNEL_ENUM_ELTWISE_MAX,
    VX_KERNEL_NAME_ELTWISE_MAX_UINT8FP16_UINT8,
    NULL,
    vxeltwiseMaxKernelParam,
    (sizeof(vxeltwiseMaxKernelParam) / sizeof(vxeltwiseMaxKernelParam[0])),
    vxeltwiseMaxValidator,
    NULL,
    NULL,
    vxeltwiseMaxInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxeltwiseMaxKernelInfo_i16fp16_i16 =
{
    VX_KERNEL_ENUM_ELTWISE_MAX,
    VX_KERNEL_NAME_ELTWISE_MAX_INT16FP16_INT16,
    NULL,
    vxeltwiseMaxKernelParam,
    (sizeof(vxeltwiseMaxKernelParam) / sizeof(vxeltwiseMaxKernelParam[0])),
    vxeltwiseMaxValidator,
    NULL,
    NULL,
    vxeltwiseMaxInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxeltwiseMaxKernelInfo_i16fp16_fp16 =
{
    VX_KERNEL_ENUM_ELTWISE_MAX,
    VX_KERNEL_NAME_ELTWISE_MAX_INT16FP16_FP16,
    NULL,
    vxeltwiseMaxKernelParam,
    (sizeof(vxeltwiseMaxKernelParam) / sizeof(vxeltwiseMaxKernelParam[0])),
    vxeltwiseMaxValidator,
    NULL,
    NULL,
    vxeltwiseMaxInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_ELTWISEMAX_list[] =
{
    NULL,
    &vxeltwiseMaxKernelInfo,
    &vxeltwiseMaxKernelInfo_int8,
    &vxeltwiseMaxKernelInfo_int8_nofl,
    &vxeltwiseMaxKernelInfo_int16,
    &vxeltwiseMaxKernelInfo_uint8,
    &vxeltwiseMaxKernelInfo_i8fp16_i8,
    &vxeltwiseMaxKernelInfo_u8fp16_u8,
    &vxeltwiseMaxKernelInfo_i16fp16_i16,
    &vxeltwiseMaxKernelInfo_i16fp16_fp16,
    NULL
};
#ifdef __cplusplus
}
#endif
