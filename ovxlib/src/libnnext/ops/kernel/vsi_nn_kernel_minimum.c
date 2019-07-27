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
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_math.h"
#include "client/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"

#define _VX_KERNEL_ID           (VX_KERNEL_ENUM_MINIMUM)

static vx_param_description_t vxMinimumKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED}
};

vx_status VX_CALLBACK vxMinimumInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    vx_uint32 paraNum
    )
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
    vx_tensor    input0             = (vx_tensor)paramObj[0];
    vx_tensor    input1             = (vx_tensor)paramObj[1];
    vx_tensor    output             = (vx_tensor)paramObj[2];
    vx_enum      src0Format         = VX_TYPE_FLOAT16;
    vx_enum      src1Format         = VX_TYPE_FLOAT16;
    vx_enum      dstFormat          = VX_TYPE_FLOAT16;
    vx_enum      src0QuantType      = 0;
    vx_int8      src0FixPointPos    = 0;
    vx_int32     src0ZP             = 0;
    vx_float32   src0Scale          = 0;
    vx_enum      src1QuantType      = 0;
    vx_int8      src1FixPointPos    = 0;
    vx_int32     src1ZP             = 0;
    vx_float32   src1Scale          = 0;
    vx_enum      dstQuantType       = 0;
    vx_int8      dstFixPointPos     = 0;
    vx_int32     dstZP              = 0;
    vx_float32   dstScale           = 0;
    vx_bool      isDymFixPoint      = vx_false_e;

    vx_uint32 output_size[4] = {0, 0, 0, 0};

    status = vxQueryTensor(input0, VX_TENSOR_DATA_TYPE, &src0Format, sizeof(src0Format));
    status |= vxQueryTensor(input0, VX_TENSOR_QUANT_FORMAT, &src0QuantType, sizeof(src0QuantType));
    status |= vxQueryTensor(input0, VX_TENSOR_FIXED_POINT_POSITION, &src0FixPointPos, sizeof(src0FixPointPos));
    status |= vxQueryTensor(input0, VX_TENSOR_ZERO_POINT, &src0ZP, sizeof(src0ZP));
    status |= vxQueryTensor(input0, VX_TENSOR_SCALE, &src0Scale, sizeof(src0Scale));
    status |= vxQueryTensor(input1, VX_TENSOR_DATA_TYPE, &src1Format, sizeof(src1Format));
    status |= vxQueryTensor(input1, VX_TENSOR_QUANT_FORMAT, &src1QuantType, sizeof(src1QuantType));
    status |= vxQueryTensor(input1, VX_TENSOR_FIXED_POINT_POSITION, &src1FixPointPos, sizeof(src1FixPointPos));
    status |= vxQueryTensor(input1, VX_TENSOR_ZERO_POINT, &src1ZP, sizeof(src1ZP));
    status |= vxQueryTensor(input1, VX_TENSOR_SCALE, &src1Scale, sizeof(src1Scale));
    status |= vxQueryTensor(output, VX_TENSOR_DATA_TYPE, &dstFormat, sizeof(dstFormat));
    status |= vxQueryTensor(output, VX_TENSOR_DIMS, output_size, sizeof(output_size));
    status |= vxQueryTensor(output, VX_TENSOR_QUANT_FORMAT, &dstQuantType, sizeof(dstQuantType));
    status |= vxQueryTensor(output, VX_TENSOR_FIXED_POINT_POSITION, &dstFixPointPos, sizeof(dstFixPointPos));
    status |= vxQueryTensor(output, VX_TENSOR_ZERO_POINT, &dstZP, sizeof(dstZP));
    status |= vxQueryTensor(output, VX_TENSOR_SCALE, &dstScale, sizeof(dstScale));

    if(status < 0)
        printf("error-%s,%d\n",__FILE__,__LINE__);

    isDymFixPoint = (vx_bool)(src0QuantType == src1QuantType && src0QuantType == VX_QUANT_DYNAMIC_FIXED_POINT
                           && dstQuantType == VX_QUANT_DYNAMIC_FIXED_POINT);

    if (dstFormat == VX_TYPE_FLOAT16 || dstFormat == VX_TYPE_INT16)
    {
        shaderParam.globalWorkScale[0]  = 8;
        shaderParam.globalWorkScale[1]  = 1;
        shaderParam.globalWorkScale[2]  = 1;
    }
    else
    {
        shaderParam.globalWorkScale[0]  = 16;
        shaderParam.globalWorkScale[1]  = 1;
        shaderParam.globalWorkScale[2]  = 1;
    }

    shaderParam.globalWorkSize[0]   = gcmALIGN((output_size[0] + shaderParam.globalWorkScale[0] - 1)
                                        / shaderParam.globalWorkScale[0], 4);
    shaderParam.globalWorkSize[1]   = (output_size[1] + shaderParam.globalWorkScale[1] - 1)
                                        / shaderParam.globalWorkScale[1];
    shaderParam.globalWorkSize[2]   = output_size[2];

    if ((isDymFixPoint && src0Format == VX_TYPE_INT8 && src1Format == VX_TYPE_INT8 && dstFormat == VX_TYPE_INT8)
        || (src0Format == VX_TYPE_INT8 && src1Format == VX_TYPE_FLOAT16 && dstFormat == VX_TYPE_INT8))
    {
        vx_uint32 uniConvertI8toI8_0_part0_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };
        vx_uint32 uniConvertI8toI8_0_part1_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x0b0a0908, 0x0f0e0d0c, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };
        vx_uint32 uniConvertI8toI8_1_part0_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };
        vx_uint32 uniConvertI8toI8_1_part1_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x0b0a0908, 0x0f0e0d0c, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };

        if (src0FixPointPos > dstFixPointPos)
        {
            vx_uint8  postshift = gcmMIN(src0FixPointPos - dstFixPointPos, MAX_POST_SHIFT_BITS);

            uniConvertI8toI8_0_part0_2x8[7] |= (postshift & 0x1F);
            uniConvertI8toI8_0_part1_2x8[7] |= (postshift & 0x1F);
        }
        else
        {
            vx_uint32 multiplier = gcmMIN(1 << (dstFixPointPos - src0FixPointPos), MAX_MULTIPLIER_NUM);
            vx_uint32 i          = 0;

            for (i = 0; i < 8; i++)
            {
                uniConvertI8toI8_0_part0_2x8[i + 8] = multiplier;
                uniConvertI8toI8_0_part1_2x8[i + 8] = multiplier;
            }
        }

        status |= vxSetNodeUniform(nodObj, "uniConvertI8toI8_0_part0_2x8", 1, uniConvertI8toI8_0_part0_2x8);
        status |= vxSetNodeUniform(nodObj, "uniConvertI8toI8_0_part1_2x8", 1, uniConvertI8toI8_0_part1_2x8);

        if(src1Format == VX_TYPE_FLOAT16)
        {
            vx_uint32 uinConvertFp16ToInt8_2x8[16] = {
                0x11111111, // TCfg
                0x00000000, // ASelt
                0x03020100, 0x07060504, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001,
                0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
            };

            if (0 > dstFixPointPos)
            {
                vx_uint8  postshift      = gcmMIN(0 - dstFixPointPos, MAX_POST_SHIFT_BITS);

                uinConvertFp16ToInt8_2x8[7] |= (postshift & 0x1F);
            }
            else
            {
                vx_uint32 multiplier = gcmMIN(1 << (dstFixPointPos - 0), MAX_MULTIPLIER_NUM);
                vx_uint32 i          = 0;

                for (i = 0; i < 8; i++)
                {
                    uinConvertFp16ToInt8_2x8[i + 8] = multiplier;
                }
            }

            status |= vxSetNodeUniform(nodObj, "uinConvertFp16ToInt8_2x8", 1, uinConvertFp16ToInt8_2x8);
        }
        else
        {
            if (src1FixPointPos > dstFixPointPos)
            {
                vx_uint8  postshift      = gcmMIN(src1FixPointPos - dstFixPointPos, MAX_POST_SHIFT_BITS);

                uniConvertI8toI8_1_part0_2x8[7] |= (postshift & 0x1F);
                uniConvertI8toI8_1_part1_2x8[7] |= (postshift & 0x1F);
            }
            else
            {
                vx_uint32 multiplier = gcmMIN(1 << (dstFixPointPos - src1FixPointPos), MAX_MULTIPLIER_NUM);
                vx_uint32 i          = 0;

                for (i = 0; i < 8; i++)
                {
                    uniConvertI8toI8_1_part0_2x8[i + 8] = multiplier;
                    uniConvertI8toI8_1_part1_2x8[i + 8] = multiplier;
                }
            }

            status |= vxSetNodeUniform(nodObj, "uniConvertI8toI8_1_part0_2x8", 1, uniConvertI8toI8_1_part0_2x8);
            status |= vxSetNodeUniform(nodObj, "uniConvertI8toI8_1_part1_2x8", 1, uniConvertI8toI8_1_part1_2x8);
        }
    }
    else if (isDymFixPoint && src0Format == VX_TYPE_INT16 && src1Format == VX_TYPE_INT16 && dstFormat == VX_TYPE_INT16)
    {
        vx_uint32 uniConvertI16toI16_0_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };
        vx_uint32 uniConvertI16toI16_1_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };

        if (src0FixPointPos > dstFixPointPos)
        {
            vx_uint8  postshift      = gcmMIN(src0FixPointPos - dstFixPointPos, MAX_POST_SHIFT_BITS);

            uniConvertI16toI16_0_2x8[7] |= (postshift & 0x1F);
        }
        else
        {
            vx_uint32 multiplier = gcmMIN(1 << (dstFixPointPos - src0FixPointPos), MAX_MULTIPLIER_NUM);
            vx_uint32 i          = 0;

            for (i = 0; i < 8; i++)
            {
                uniConvertI16toI16_0_2x8[i + 8] = multiplier;
            }
        }

        status |= vxSetNodeUniform(nodObj, "uniConvertI16toI16_0_2x8", 1, uniConvertI16toI16_0_2x8);

        if (src1FixPointPos > dstFixPointPos)
        {
            vx_uint8  postshift      = gcmMIN(src1FixPointPos - dstFixPointPos, MAX_POST_SHIFT_BITS);

            uniConvertI16toI16_1_2x8[7] |= (postshift & 0x1F);
        }
        else
        {
            vx_uint32 multiplier = gcmMIN(1 << (dstFixPointPos - src1FixPointPos), MAX_MULTIPLIER_NUM);
            vx_uint32 i          = 0;

            for (i = 0; i < 8; i++)
            {
                uniConvertI16toI16_1_2x8[i + 8] = multiplier;
            }
        }

        status |= vxSetNodeUniform(nodObj, "uniConvertI16toI16_1_2x8", 1, uniConvertI16toI16_1_2x8);
    }
    else if ((src0Format == VX_TYPE_UINT8 && src1Format == VX_TYPE_UINT8 && dstFormat == VX_TYPE_UINT8)
            || (src0Format == VX_TYPE_UINT8 && src1Format == VX_TYPE_FLOAT16 && dstFormat == VX_TYPE_UINT8))
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

        vsi_nn_GetFP32MultiAndPostShift(src0Scale / dstScale, &M0, &postShift0);
        vsi_nn_GetFP32MultiAndPostShift(src1Scale / dstScale, &M1, &postShift1);

        multAndoutZP0[0] = (vx_uint32)(M0);
        multAndoutZP0[1] = (vx_uint32)((dstZP << postShift0) - src0ZP * M0);
        multAndoutZP1[0] = (vx_uint32)(M1);
        multAndoutZP1[1] = (vx_uint32)((dstZP << postShift1) - src1ZP * M1);

        uniU8MulAndPostShift_Lo_2x8[7] = 0x00002600 | (postShift0 & 0x1F);
        uniU8MulAndPostShift_Hi_2x8[7] = 0x00002600 | (postShift0 & 0x1F);

        status |= vxSetNodeUniform(nodObj, "uniU8MulAndPostShift0_Lo_2x8",  1, uniU8MulAndPostShift_Lo_2x8);
        status |= vxSetNodeUniform(nodObj, "uniU8MulAndPostShift0_Hi_2x8",  1, uniU8MulAndPostShift_Hi_2x8);
        status |= vxSetNodeUniform(nodObj, "multAndoutZP0",  1, multAndoutZP0);
        status |= vxSetNodeUniform(nodObj, "multAndoutZP1",  1, multAndoutZP1);

        if(src1Format == VX_TYPE_FLOAT16)
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
    else if(src0Format == VX_TYPE_INT8 && src1Format == VX_TYPE_FLOAT16 && dstFormat == VX_TYPE_FLOAT16)
    {
        vx_uint32 uniConvertInt8toFp16_2x8[16] = {
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
            vx_uint8  postshift = gcmMIN(src0FixPointPos, MAX_POST_SHIFT_BITS);

            uniConvertInt8toFp16_2x8[7] |= (postshift & 0x1F);
        }
        else
        {
            vx_uint32 multiplier = gcmMIN(1 << (0 - src0FixPointPos), MAX_MULTIPLIER_NUM);
            vx_uint32 i          = 0;

            for (i = 0; i < 8; i++)
            {
                uniConvertInt8toFp16_2x8[i + 8] = multiplier;
            }
        }

        status |= vxSetNodeUniform(nodObj, "uniConvertInt8toFp16_2x8", 1, uniConvertInt8toFp16_2x8);
    }
    else if(src0Format == VX_TYPE_UINT8 && src1Format == VX_TYPE_FLOAT16 && dstFormat == VX_TYPE_FLOAT16)
    {
        vx_uint16  M0                   = 0;
        vx_int8    postShift            = 0;
        vx_uint32    multAndoutZP0[2]   = {0};
        vx_uint32 uniU8MulAndPostShift_0_Lo_2x8[16] = {
            0xdddddddd, // TCfg
            0x44444444, // ASelt
            0x13121110, 0x17161514, // ABin
            0x11111111, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };

        vsi_nn_GetFP32MultiAndPostShift(src0Scale / dstScale, &M0, &postShift);
        multAndoutZP0[0] = (vx_uint32)(M0);
        multAndoutZP0[1] = (vx_uint32)((dstZP << postShift) - src0ZP * M0);

        uniU8MulAndPostShift_0_Lo_2x8[7] |= (postShift & 0x1F);
        status |= vxSetNodeUniform(nodObj, "uniU8MulAndPostShift_0_Lo_2x8", 1, uniU8MulAndPostShift_0_Lo_2x8);
        status |= vxSetNodeUniform(nodObj, "multAndoutZP0", 1, multAndoutZP0);
    }
    else if(src0Format == VX_TYPE_INT16 && src1Format == VX_TYPE_FLOAT16 && dstFormat == VX_TYPE_INT16)
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
            vx_uint8  postshift = gcmMIN(src0FixPointPos - dstFixPointPos, MAX_POST_SHIFT_BITS);

            uniConvertI16toI16_2x8[7] |= (postshift & 0x1F);
        }
        else
        {
            vx_uint32 multiplier = gcmMIN(1 << (dstFixPointPos - src0FixPointPos), MAX_MULTIPLIER_NUM);
            vx_uint32 i          = 0;

            for (i = 0; i < 8; i++)
            {
                uniConvertI16toI16_2x8[i + 8] = multiplier;
            }
        }

        if (0 > dstFixPointPos)
        {
            vx_uint8  postshift      = gcmMIN(0 - dstFixPointPos, MAX_POST_SHIFT_BITS);
            uinConvertFp16ToInt16_2x8[7] |= (postshift & 0x1F);
        }
        else
        {
            vx_uint32 multiplier = gcmMIN(1 << (dstFixPointPos - 0), MAX_MULTIPLIER_NUM);
            vx_uint32 i          = 0;

            for (i = 0; i < 8; i++)
            {
                uinConvertFp16ToInt16_2x8[i + 8] = multiplier;
            }
        }

        status |= vxSetNodeUniform(nodObj, "uniConvertI16toI16_2x8", 1, uniConvertI16toI16_2x8);
        status |= vxSetNodeUniform(nodObj, "uinConvertFp16ToInt16_2x8", 1, uinConvertFp16ToInt16_2x8);
    }
    else if(src0Format == VX_TYPE_INT16 && src1Format == VX_TYPE_FLOAT16 && dstFormat == VX_TYPE_FLOAT16)
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
            vx_uint8  postshift = gcmMIN(src0FixPointPos, MAX_POST_SHIFT_BITS);
            uniConvertInt16toFp16_2x8[7] |= (postshift & 0x1F);
        }
        else
        {
            vx_uint32 multiplier = gcmMIN(1 << (0 - src0FixPointPos), MAX_MULTIPLIER_NUM);
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

    return status;
}


#ifdef __cplusplus
extern "C" {
#endif


#define TENSOR_MIN_KERNELS(SRC0_TYPE, SRC1_TYPE, DST_TYPE) \
    vx_kernel_description_t vxTensorMinimum_##SRC0_TYPE##SRC1_TYPE##to##DST_TYPE##_Kernel = \
{ \
    _VX_KERNEL_ID, \
    VX_KERNEL_NAME_MINIMUM_##SRC0_TYPE##SRC1_TYPE##TO##DST_TYPE, \
    NULL, \
    vxMinimumKernelParam, \
    (sizeof(vxMinimumKernelParam) / sizeof(vxMinimumKernelParam[0])), \
    vsi_nn_KernelValidator, \
    NULL, \
    NULL, \
    vxMinimumInitializer, \
    vsi_nn_KernelDeinitializer \
};

#define TENSOR_MIN_KERNELS_2D(SRC0_TYPE, SRC1_TYPE, DST_TYPE) \
    vx_kernel_description_t vxTensorMinimum_##SRC0_TYPE##SRC1_TYPE##to##DST_TYPE##_2D_Kernel = \
{ \
    _VX_KERNEL_ID, \
    VX_KERNEL_NAME_MINIMUM_##SRC0_TYPE##SRC1_TYPE##TO##DST_TYPE##_2D, \
    NULL, \
    vxMinimumKernelParam, \
    (sizeof(vxMinimumKernelParam) / sizeof(vxMinimumKernelParam[0])), \
    vsi_nn_KernelValidator, \
    NULL, \
    NULL, \
    vxMinimumInitializer, \
    vsi_nn_KernelDeinitializer \
};


TENSOR_MIN_KERNELS(F16, F16, F16)
TENSOR_MIN_KERNELS(I8,  F16, I8)
TENSOR_MIN_KERNELS(I8,  F16, F16)
TENSOR_MIN_KERNELS(U8,  F16, U8)
TENSOR_MIN_KERNELS(U8,  F16, F16)
TENSOR_MIN_KERNELS(I8,  I8, I8)
TENSOR_MIN_KERNELS(U8,  U8, U8)
TENSOR_MIN_KERNELS(I16, I16, I16)
TENSOR_MIN_KERNELS(I16, F16, I16)
TENSOR_MIN_KERNELS(I16, F16, F16)
TENSOR_MIN_KERNELS(F16, F16, U8)

TENSOR_MIN_KERNELS_2D(F16, F16, F16)
TENSOR_MIN_KERNELS_2D(I8,  F16, I8)
TENSOR_MIN_KERNELS_2D(I8,  F16, F16)
TENSOR_MIN_KERNELS_2D(U8,  F16, U8)
TENSOR_MIN_KERNELS_2D(U8,  F16, F16)
TENSOR_MIN_KERNELS_2D(I8,  I8, I8)
TENSOR_MIN_KERNELS_2D(U8,  U8, U8)
TENSOR_MIN_KERNELS_2D(I16, I16, I16)
TENSOR_MIN_KERNELS_2D(I16, F16, I16)
TENSOR_MIN_KERNELS_2D(I16, F16, F16)
TENSOR_MIN_KERNELS_2D(F16, F16, U8)

#define TENSOR_MIN_KERENLS_NAME(SRC0_TYPE, SRC1_TYPE, DST_TYPE, INSTR) \
    &vxTensorMinimum_##SRC0_TYPE##SRC1_TYPE##to##DST_TYPE##_##INSTR##Kernel,


vx_kernel_description_t * vx_kernel_MINIMUM_list[] =
{
    NULL,
    TENSOR_MIN_KERENLS_NAME(F16, F16, F16, )
    TENSOR_MIN_KERENLS_NAME(I8,  F16, I8, )
    TENSOR_MIN_KERENLS_NAME(I8,  F16, F16, )
    TENSOR_MIN_KERENLS_NAME(U8,  F16, U8, )
    TENSOR_MIN_KERENLS_NAME(U8,  F16, F16, )
    TENSOR_MIN_KERENLS_NAME(I8,  I8, I8, )
    TENSOR_MIN_KERENLS_NAME(U8,  U8, U8, )
    TENSOR_MIN_KERENLS_NAME(I16, I16, I16, )
    TENSOR_MIN_KERENLS_NAME(I16, F16, I16, )
    TENSOR_MIN_KERENLS_NAME(I16, F16, F16, )
    TENSOR_MIN_KERENLS_NAME(F16, F16, U8, )

    TENSOR_MIN_KERENLS_NAME(F16, F16, F16, 2D_)
    TENSOR_MIN_KERENLS_NAME(I8,  F16, I8, 2D_)
    TENSOR_MIN_KERENLS_NAME(I8,  F16, F16, 2D_)
    TENSOR_MIN_KERENLS_NAME(U8,  F16, U8, 2D_)
    TENSOR_MIN_KERENLS_NAME(U8,  F16, F16, 2D_)
    TENSOR_MIN_KERENLS_NAME(I8,  I8, I8, 2D_)
    TENSOR_MIN_KERENLS_NAME(U8,  U8, U8, 2D_)
    TENSOR_MIN_KERENLS_NAME(I16, I16, I16, 2D_)
    TENSOR_MIN_KERENLS_NAME(I16, F16, I16, 2D_)
    TENSOR_MIN_KERENLS_NAME(I16, F16, F16, 2D_)
    TENSOR_MIN_KERENLS_NAME(F16, F16, U8, 2D_)
    NULL
};
#ifdef __cplusplus
}
#endif
