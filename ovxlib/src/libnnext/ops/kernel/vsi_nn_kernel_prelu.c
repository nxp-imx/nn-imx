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

vsi_status VX_CALLBACK vxParametricReluInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    uint32_t paraNum
    )
{
// Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
    vx_kernel_execution_parameters_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread
    uint32_t UniFP16Mul_dp2x8[16] = {
        0x11111111, // TCfg
        0x00000000, // ASelt
        0x03020100, 0x07060504, // ABin
        0x11111111, // BSelt
        0x00000000, 0x00000000, // BBin
        0x00000100, // AccumType, ConstantType, and PostShift
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
    };
    uint32_t UniS8xFp16_dp2x8[16] = {
        0x11111111, // TCfg
        0x00000000, // ASelt
        0x03020100, 0x07060504, // ABin
        0x11111111, // BSelt
        0x00000000, 0x00000000, // BBin
        0x00000400, // AccumType, ConstantType, and PostShift
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
    };

    vsi_status status = VX_SUCCESS;

    vx_tensor   input                   = (vx_tensor)paramObj[0];
    vx_tensor   output                  = (vx_tensor)paramObj[2];
    uint32_t    input_size[DIM_SIZE];
    int8_t      input_fixPointPos       = 0;
    int8_t      output_fixPointPos      = 0;
    vx_float32  in_scale                = 1.0f;
    vx_float32  out_scale               = 1.0f;
    vx_float32  scale_inOut             = 1.0f;
    vsi_enum    inDataType, outDataType;
    vx_float32  scaleIn                 = 0;
    vx_float32  scaleOut                = 0;
    int32_t     output_ZP               = 0;
    int32_t     input_ZP                = 0;
    vx_float32  scaleInInt16            = 0;
    vx_float32  scaleOutInt16           = 0;
    vx_float32  reScaleOut_u8           = 0;
    vx_bool     optFlg                  = vx_false_e;
    vx_uint16   M0                      = 0;
    vx_int8     postShift               = 0;
    vx_bool     enable_image_2d         = vx_false_e;

    status = vxQueryTensor(input, VX_TENSOR_DIMS, input_size, sizeof(input_size));
    status |= vxQueryTensor(input, VX_TENSOR_FIXED_POINT_POS, &input_fixPointPos, sizeof(input_fixPointPos));
    status |= vxQueryTensor(output, VX_TENSOR_FIXED_POINT_POS, &output_fixPointPos, sizeof(output_fixPointPos));
    status |= vxQueryTensor(input, VX_TENSOR_DATA_TYPE, &inDataType, sizeof(inDataType));
    status |= vxQueryTensor(output, VX_TENSOR_DATA_TYPE, &outDataType, sizeof(outDataType));
    status |= vxQueryTensor(input, VX_TENSOR_ZERO_POINT, &input_ZP, sizeof(input_ZP));
    status |= vxQueryTensor(input, VX_TENSOR_SCALE, &scaleIn, sizeof(scaleIn));
    status |= vxQueryTensor(output, VX_TENSOR_ZERO_POINT, &output_ZP, sizeof(output_ZP));
    status |= vxQueryTensor(output, VX_TENSOR_SCALE, &scaleOut, sizeof(scaleOut));

    if(outDataType == VX_TYPE_UINT8)
        reScaleOut_u8 = 1 / scaleOut;
    else
        scaleOut = 1.0f;

    enable_image_2d = (vx_bool)(input_size[2] == 1);

    if (status != VX_SUCCESS)
    {
        VSILOGE("vxQueryTensor FIXED_POINT_POS failure!\n");
        return status;
    }
    if (input_fixPointPos >= 0)
    {
        in_scale = 1.0f / (vx_float32) (1 << input_fixPointPos);
    }
    else if (input_fixPointPos < 0)
    {
        in_scale = (vx_float32) (1 << -input_fixPointPos);
    }
    if (output_fixPointPos >= 0)
    {
        out_scale = (vx_float32) (1 << output_fixPointPos);
    }
    else if (output_fixPointPos < 0)
    {
        out_scale = 1.0f / (vx_float32) (1 << -output_fixPointPos);
    }

    if (input_fixPointPos > 0)
    {
        scaleInInt16 = (1.0f / ((vx_float32) (1 << input_fixPointPos)));
    }
    else
    {
        scaleInInt16 = ((vx_float32) (1 << -input_fixPointPos));
    }

    if (output_fixPointPos > 0)
    {
        scaleOutInt16 = (vx_float32)(1 << output_fixPointPos);
    }
    else
    {
        scaleOutInt16 = (1.0f / (vx_float32)(1 << -output_fixPointPos));
    }
    scale_inOut = in_scale * out_scale;

    if(((inDataType == VX_TYPE_INT8 && outDataType == VX_TYPE_INT8)
        || (inDataType == VX_TYPE_INT16 && outDataType == VX_TYPE_INT16))
        && (((input_fixPointPos >= output_fixPointPos) && (input_fixPointPos - output_fixPointPos < 32))
        || ((input_fixPointPos < output_fixPointPos) && (output_fixPointPos - input_fixPointPos < 16))) && enable_image_2d)
    {
        optFlg = vx_true_e;
    }
    else if (enable_image_2d && inDataType == VX_TYPE_UINT8
        && (outDataType == VX_TYPE_UINT8 || outDataType == VX_TYPE_FLOAT16))
    {
        vsi_nn_GetFP32MultiAndPostShift(scaleIn / scaleOut, &M0, &postShift);
        optFlg = vx_true_e;
    }

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkOffset[2] = 0;
    shaderParam.globalWorkScale[0]  = 8;
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.globalWorkScale[2]  = 1;
    shaderParam.localWorkSize[0]    = 8;
    shaderParam.localWorkSize[1]    = 1;
    shaderParam.localWorkSize[2]    = 1;

    if (inDataType == VX_TYPE_INT8 && outDataType == VX_TYPE_INT8)
        shaderParam.globalWorkScale[0]  = 16;
    else if (inDataType == VX_TYPE_UINT8 && enable_image_2d
        && (outDataType == VX_TYPE_UINT8 || outDataType == VX_TYPE_FLOAT16))
        shaderParam.globalWorkScale[0]  = 16;

    if(optFlg)
    {
        shaderParam.workDim = 2;
        shaderParam.globalWorkSize[0]   = gcmALIGN((input_size[0] + shaderParam.globalWorkScale[0] - 1)
            / shaderParam.globalWorkScale[0], shaderParam.localWorkSize[0]);
        shaderParam.globalWorkSize[1]   = gcmALIGN((input_size[1] + shaderParam.globalWorkScale[1] - 1)
            / shaderParam.globalWorkScale[1], shaderParam.localWorkSize[1]);
    }
    else
    {
        shaderParam.globalWorkSize[0]   = gcmALIGN((input_size[0] + shaderParam.globalWorkScale[0] - 1)
            / shaderParam.globalWorkScale[0], shaderParam.localWorkSize[0]);
        shaderParam.globalWorkSize[1]   = gcmALIGN((input_size[1] + shaderParam.globalWorkScale[1] - 1)
            / shaderParam.globalWorkScale[1], shaderParam.localWorkSize[1]);
        shaderParam.globalWorkSize[2]   = input_size[2];
    }

    if(inDataType == VX_TYPE_UINT8 && outDataType == VX_TYPE_UINT8)
    {
        if (enable_image_2d)
        {
            vx_uint32 idx = 0;
            vx_uint32 uniU8SubZP_MulM_PStoF16Lo_2x8[16] = {
                0x99999999, // TCfg
                0x44444444, // ASelt
                0x03020100, 0x07060504, // ABin
                0xaaaaaaaa, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
            };
            vx_uint32 uniU8SubZP_MulM_PStoF16Hi_2x8[16] = {
                0x99999999, // TCfg
                0x44444444, // ASelt
                0x0b0a0908, 0x0f0e0d0c, // ABin
                0xaaaaaaaa, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
            };
            vx_uint32 uniF16MulF16_2x8[16] = {
                0x11111111, // TCfg
                0x00000000, // ASelt
                0x03020100, 0x07060504, // ABin
                0x11111111, // BSelt
                0x03020100, 0x07060504, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            };
            vx_uint32 uniS16AddZP_2x8[16] = {
                0x55555555, // TCfg
                0x44444444, // ASelt
                0x03020100, 0x07060504, // ABin
                0xaaaaaaaa, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
            };
            uniU8SubZP_MulM_PStoF16Lo_2x8[7] |= postShift;
            uniU8SubZP_MulM_PStoF16Hi_2x8[7] |= postShift;

            for (idx = 8; idx < 16; idx ++)
            {
                uniU8SubZP_MulM_PStoF16Hi_2x8[idx] = uniU8SubZP_MulM_PStoF16Lo_2x8[idx] = (vx_uint32)(M0 << 16) | M0;
            }

            status |= vxSetNodeUniform(nodObj, "uniU8SubZP_MulM_PStoF16Lo_2x8", 1, uniU8SubZP_MulM_PStoF16Lo_2x8);
            status |= vxSetNodeUniform(nodObj, "uniU8SubZP_MulM_PStoF16Hi_2x8", 1, uniU8SubZP_MulM_PStoF16Hi_2x8);
            status |= vxSetNodeUniform(nodObj, "uniF16MulF16_2x8", 1, uniF16MulF16_2x8);
            status |= vxSetNodeUniform(nodObj, "uniS16AddZP_2x8", 1, uniS16AddZP_2x8);
            status |= vxSetNodeUniform(nodObj, "outputZP", 1, &output_ZP);
            status |= vxSetNodeUniform(nodObj, "inputZP", 1, &input_ZP);
        }
        else
        {
            vx_uint32 uniConvertUint8SubZpToFp32_4x4[16] = {
                0x05050505, // TCfg
                0x04040404, // ASelt
                0x00010000, 0x00030002, // ABin
                0x0a0a0a0a, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002100, // AccumType, ConstantType, and PostShift
                0xbc003c00, 0x00000000, 0xbc003c00, 0x00000000, 0xbc003c00, 0x00000000, 0xbc003c00, 0x00000000 // Constant
            };

            vx_uint32 uniConvertSecUint8SubZpToFp32_4x4[16] = {
                0x05050505, // TCfg
                0x04040404, // ASelt
                0x00050004, 0x00070006, // ABin
                0x0a0a0a0a, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0xbc003c00, 0x00000000, 0xbc003c00, 0x00000000, 0xbc003c00, 0x00000000, 0xbc003c00, 0x00000000 // Constant
            };
            uint32_t uniConvertInt32toUint8_2x8[16] = {
                0x33333333, // TCfg
                0x11110000, // ASelt
                0x03020100, 0x03020100, // ABin
                0x00000000, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            };

            vx_float32 scale_inOut_u8 = scaleIn * reScaleOut_u8;

            status |= vxSetNodeUniform(nodObj, "uniConvertInt32toUint8_2x8",
                1, uniConvertInt32toUint8_2x8);
            status |= vxSetNodeUniform(nodObj, "uniConvertUint8SubZpToFp32_4x4",
                1, uniConvertUint8SubZpToFp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvertSecUint8SubZpToFp32_4x4",
                1, uniConvertSecUint8SubZpToFp32_4x4);
            status |= vxSetNodeUniform(nodObj, "scale_inOut_u8", 1, &scale_inOut_u8);
            status |= vxSetNodeUniform(nodObj, "input_ZP", 1, &input_ZP);
            status |= vxSetNodeUniform(nodObj, "output_ZP", 1, &output_ZP);
        }
    }
    else if(inDataType == VX_TYPE_INT16 && outDataType == VX_TYPE_INT16)
    {
        uint32_t uniConvertInt32toUint8_2x8[16] = {
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        uint32_t uniConvertDirInt16Fp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };
        uint32_t uniConvertEndInt16Fp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };

        ///
        vx_uint32 uniPreluInt16_2x8b[16] = {
            0x77777777, // TCfg
            0x44444444, // ASelt
            0x31211000, 0x73635242, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00003000, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };

        vx_uint32 uniPreluInt16_4x4[16] = {
            0x05050505, // TCfg
            0x00000000, // ASelt
            0x00510040, 0x00730062, // ABin
            0x06060606, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        };
        // output fl > input fl
        vx_uint32 uniMergeMultiplier_2x8[16] = {
            0x00000011, // TCfg
            0x00000010, // ASelt
            0x00000000, 0x00000000, // ABin
            0x00000021, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        if(optFlg)
        {
            if (input_fixPointPos >= output_fixPointPos)
            {
                uniPreluInt16_2x8b[7] = uniPreluInt16_2x8b[7] | (input_fixPointPos - output_fixPointPos);
                uniPreluInt16_4x4[7]  = uniPreluInt16_4x4[7] | (input_fixPointPos - output_fixPointPos);
                vxSetNodeUniform(nodObj, "uniPreluInt16_2x8b", 1, uniPreluInt16_2x8b);
                vxSetNodeUniform(nodObj, "uniPreluInt16_4x4", 1, uniPreluInt16_4x4);
            }
            else
            {
                vx_uint32 uniPreluInt16Mul_2x8b[16] = {
                    0x55555555, // TCfg
                    0x44444444, // ASelt
                    0x33221100, 0x77665544, // ABin
                    0x00000000, // BSelt
                    0x01010101, 0x01010101, // BBin
                    0x00000000, // AccumType, ConstantType, and PostShift
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
                };
                vx_int32 multiplier = 1 << (output_fixPointPos - input_fixPointPos);
                vxSetNodeUniform(nodObj, "uniPreluInt16_2x8b", 1, uniPreluInt16Mul_2x8b);
                vxSetNodeUniform(nodObj, "uniPreluInt16_4x4", 1, uniPreluInt16_4x4);
                vxSetNodeUniform(nodObj, "uniMergeMultiplier_2x8", 1, uniMergeMultiplier_2x8);
                vxSetNodeUniform(nodObj, "multiplier", 1, &multiplier);
            }
        }
        else
        {
            status |= vxSetNodeUniform(nodObj, "uniConvertDirInt16Fp32_4x4",
                1, uniConvertDirInt16Fp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvertEndInt16Fp32_4x4",
                1, uniConvertEndInt16Fp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvertInt32toUint8_2x8",
                1, uniConvertInt32toUint8_2x8);
        }
    }
    else if(inDataType == VX_TYPE_FLOAT16 && outDataType == VX_TYPE_UINT8)
    {
        uint32_t uniConvertInt32toUint8_2x8[16] = {
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        uint32_t uniConvertDirInt16Fp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };
        uint32_t uniConvertEndInt16Fp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };

        status |= vxSetNodeUniform(nodObj, "uniConvertDirInt16Fp32_4x4",
            1, uniConvertDirInt16Fp32_4x4);
        status |= vxSetNodeUniform(nodObj, "uniConvertEndInt16Fp32_4x4",
            1, uniConvertEndInt16Fp32_4x4);
        status |= vxSetNodeUniform(nodObj, "uniConvertInt32toUint8_2x8",
            1, uniConvertInt32toUint8_2x8);
        status |= vxSetNodeUniform(nodObj, "outputScale", 1, &reScaleOut_u8);
        status |= vxSetNodeUniform(nodObj, "output_ZP", 1, &output_ZP);
    }
    else if(inDataType == VX_TYPE_FLOAT16 && outDataType == VX_TYPE_INT16)
    {
        uint32_t uniConvertInt32toUint8_2x8[16] = {
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        uint32_t uniConvertDirInt16Fp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };
        uint32_t uniConvertEndInt16Fp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };

        status |= vxSetNodeUniform(nodObj, "uniConvertDirInt16Fp32_4x4",
            1, uniConvertDirInt16Fp32_4x4);
        status |= vxSetNodeUniform(nodObj, "uniConvertEndInt16Fp32_4x4",
            1, uniConvertEndInt16Fp32_4x4);
        status |= vxSetNodeUniform(nodObj, "uniConvertInt32toUint8_2x8",
            1, uniConvertInt32toUint8_2x8);
        status |= vxSetNodeUniform(nodObj, "outScaleInt16", 1, &scaleOutInt16);
    }
    else if(inDataType == VX_TYPE_INT16 && outDataType == VX_TYPE_FLOAT16)
    {
        uint32_t uniConvertInt32toUint8_2x8[16] = {
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        uint32_t uniConvertDirInt16Fp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };
        uint32_t uniConvertEndInt16Fp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };

        status |= vxSetNodeUniform(nodObj, "uniConvertDirInt16Fp32_4x4",
            1, uniConvertDirInt16Fp32_4x4);
        status |= vxSetNodeUniform(nodObj, "uniConvertEndInt16Fp32_4x4",
            1, uniConvertEndInt16Fp32_4x4);
        status |= vxSetNodeUniform(nodObj, "uniConvertInt32toUint8_2x8",
            1, uniConvertInt32toUint8_2x8);
        status |= vxSetNodeUniform(nodObj, "inScaleInt16", 1, &scaleInInt16);
    }
    else if(inDataType == VX_TYPE_INT8 && outDataType == VX_TYPE_INT8)
    {
        vx_uint32 uniConvertInt8FstFp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        };
        vx_uint32 uniConvertInt8SecFp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        };
        vx_uint32 uniConvertInt8TrdFp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00090008, 0x000b000a, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        };
        vx_uint32 uniConvertInt8ForFp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x000d000c, 0x000f000e, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        };
        uint32_t uniConvertInt32toUint8_2x8[16] = {
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };

        ///
        vx_uint32 uniPreluInt8Lo_2x8b[16] = {
            0x77777777, // TCfg
            0x44444444, // ASelt
            0x33221100, 0x77665544, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00004000, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniPreluInt8Hi_2x8b[16] = {
            0x77777777, // TCfg
            0x44444444, // ASelt
            0xbbaa9988, 0xffeeddcc, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00004000, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniPreluInt8_2x8[16] = {
            0x55555555, // TCfg
            0x00000000, // ASelt
            0xb3a29180, 0xf7e6d5c4, // ABin
            0x66666666, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };
        // output fl > input fl
        vx_uint32 uniMergeMultiplier_2x8[16] = {
            0x00000011, // TCfg
            0x00000010, // ASelt
            0x00000000, 0x00000000, // ABin
            0x00000021, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };

        if(optFlg)
        {
            if (input_fixPointPos >= output_fixPointPos)
            {
                uniPreluInt8Lo_2x8b[7] = uniPreluInt8Lo_2x8b[7] | (input_fixPointPos - output_fixPointPos);
                uniPreluInt8Hi_2x8b[7] = uniPreluInt8Hi_2x8b[7] | (input_fixPointPos - output_fixPointPos);
                uniPreluInt8_2x8[7]    = uniPreluInt8_2x8[7] | (input_fixPointPos - output_fixPointPos);

                vxSetNodeUniform(nodObj, "uniPreluInt8Lo_2x8b", 1, uniPreluInt8Lo_2x8b);
                vxSetNodeUniform(nodObj, "uniPreluInt8Hi_2x8b", 1, uniPreluInt8Hi_2x8b);
                vxSetNodeUniform(nodObj, "uniPreluInt8_2x8", 1, uniPreluInt8_2x8);
            }
            else
            {
                vx_uint32 uniPreluInt8Lo_2x8b[16] = {
                    0x55555555, // TCfg
                    0x44444444, // ASelt
                    0x33221100, 0x77665544, // ABin
                    0x00000000, // BSelt
                    0x01010101, 0x01010101, // BBin
                    0x00000000, // AccumType, ConstantType, and PostShift
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
                };
                vx_uint32 uniPreluInt8Hi_2x8b[16] = {
                    0x55555555, // TCfg
                    0x44444444, // ASelt
                    0xbbaa9988, 0xffeeddcc, // ABin
                    0x00000000, // BSelt
                    0x01010101, 0x01010101, // BBin
                    0x00000000, // AccumType, ConstantType, and PostShift
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
                };
                vx_int32 multiplier = 1 << (output_fixPointPos - input_fixPointPos);

                vxSetNodeUniform(nodObj, "uniPreluInt8Lo_2x8b", 1, uniPreluInt8Lo_2x8b);
                vxSetNodeUniform(nodObj, "uniPreluInt8Hi_2x8b", 1, uniPreluInt8Hi_2x8b);
                vxSetNodeUniform(nodObj, "uniMergeMultiplier_2x8", 1, uniMergeMultiplier_2x8);
                vxSetNodeUniform(nodObj, "uniPreluInt8_2x8", 1, uniPreluInt8_2x8);
                vxSetNodeUniform(nodObj, "multiplier", 1, &multiplier);
            }
        }
        else
        {
            status = vxSetNodeUniform(nodObj, "uniConvertInt8FstFp32_4x4", 1, uniConvertInt8FstFp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvertInt8SecFp32_4x4", 1, uniConvertInt8SecFp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvertInt8TrdFp32_4x4", 1, uniConvertInt8TrdFp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvertInt8ForFp32_4x4", 1, uniConvertInt8ForFp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvertInt32toUint8_2x8", 1, uniConvertInt32toUint8_2x8);
        }
    }
    else if(inDataType == VX_TYPE_UINT8 && outDataType == VX_TYPE_FLOAT16)
    {
        if (enable_image_2d)
        {
            vx_uint32 idx = 0;
            vx_uint32 uniU8SubZP_MulM_PStoF16Lo_2x8[16] = {
                0x99999999, // TCfg
                0x44444444, // ASelt
                0x03020100, 0x07060504, // ABin
                0xaaaaaaaa, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
            };
            vx_uint32 uniU8SubZP_MulM_PStoF16Hi_2x8[16] = {
                0x99999999, // TCfg
                0x44444444, // ASelt
                0x0b0a0908, 0x0f0e0d0c, // ABin
                0xaaaaaaaa, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
            };
            vx_uint32 uniF16MulF16_2x8[16] = {
                0x11111111, // TCfg
                0x00000000, // ASelt
                0x03020100, 0x07060504, // ABin
                0x11111111, // BSelt
                0x03020100, 0x07060504, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            };
            vx_uint32 uniS16AddZP_2x8[16] = {
                0x55555555, // TCfg
                0x44444444, // ASelt
                0x03020100, 0x07060504, // ABin
                0xaaaaaaaa, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
            };
            uniU8SubZP_MulM_PStoF16Lo_2x8[7] |= postShift;
            uniU8SubZP_MulM_PStoF16Hi_2x8[7] |= postShift;

            for (idx = 8; idx < 16; idx ++)
            {
                uniU8SubZP_MulM_PStoF16Hi_2x8[idx] = uniU8SubZP_MulM_PStoF16Lo_2x8[idx] = (vx_uint32)(M0 << 16) | M0;
            }

            status |= vxSetNodeUniform(nodObj, "uniU8SubZP_MulM_PStoF16Lo_2x8", 1, uniU8SubZP_MulM_PStoF16Lo_2x8);
            status |= vxSetNodeUniform(nodObj, "uniU8SubZP_MulM_PStoF16Hi_2x8", 1, uniU8SubZP_MulM_PStoF16Hi_2x8);
            status |= vxSetNodeUniform(nodObj, "uniF16MulF16_2x8", 1, uniF16MulF16_2x8);
            status |= vxSetNodeUniform(nodObj, "uniS16AddZP_2x8", 1, uniS16AddZP_2x8);
            status |= vxSetNodeUniform(nodObj, "inputZP", 1, &input_ZP);
        }
        else
        {
            vx_uint32 uniConvertUint8SubZpToFp32_4x4[16] = {
                0x05050505, // TCfg
                0x04040404, // ASelt
                0x00010000, 0x00030002, // ABin
                0x0a0a0a0a, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002100, // AccumType, ConstantType, and PostShift
                0xbc003c00, 0x00000000, 0xbc003c00, 0x00000000, 0xbc003c00, 0x00000000, 0xbc003c00, 0x00000000 // Constant
            };

            vx_uint32 uniConvertSecUint8SubZpToFp32_4x4[16] = {
                0x05050505, // TCfg
                0x04040404, // ASelt
                0x00050004, 0x00070006, // ABin
                0x0a0a0a0a, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0xbc003c00, 0x00000000, 0xbc003c00, 0x00000000, 0xbc003c00, 0x00000000, 0xbc003c00, 0x00000000 // Constant
            };

            uint32_t uniConvertInt32toUint8_2x8[16] = {
                0x33333333, // TCfg
                0x11110000, // ASelt
                0x03020100, 0x03020100, // ABin
                0x00000000, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            };

            status = vxSetNodeUniform(nodObj, "uniConvertUint8SubZpToFp32_4x4", 1, uniConvertUint8SubZpToFp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvertSecUint8SubZpToFp32_4x4", 1, uniConvertSecUint8SubZpToFp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvertInt32toUint8_2x8", 1, uniConvertInt32toUint8_2x8);
            status |= vxSetNodeUniform(nodObj, "inputScale", 1, &scaleIn);
            status |= vxSetNodeUniform(nodObj, "input_ZP", 1, &input_ZP);
        }
    }
    else if(inDataType == VX_TYPE_FLOAT16 && outDataType == VX_TYPE_INT8)
    {
        uint32_t uniConvertDirInt16Fp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };
        uint32_t uniConvertEndInt16Fp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };
        uint32_t uniConvertInt32toUint8_2x8[16] = {
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };

        status |= vxSetNodeUniform(nodObj, "uniConvertDirFp16Fp32_4x4",
            1, uniConvertDirInt16Fp32_4x4);
        status |= vxSetNodeUniform(nodObj, "uniConvertEndFp16Fp32_4x4",
            1, uniConvertEndInt16Fp32_4x4);
        status |= vxSetNodeUniform(nodObj, "uniConvertInt32toInt8_2x8",
            1, uniConvertInt32toUint8_2x8);
        status |= vxSetNodeUniform(nodObj, "outputFl_i8", 1, &out_scale);
    }

    if(!optFlg && !(inDataType == VX_TYPE_FLOAT16 && outDataType == VX_TYPE_INT8))
    {
        status = vxSetNodeUniform(nodObj, "UniFP16Mul_dp2x8", 1, UniFP16Mul_dp2x8);
        status |= vxSetNodeUniform(nodObj, "UniS8xFp16_dp2x8", 1, UniS8xFp16_dp2x8);
        status |= vxSetNodeUniform(nodObj, "in_scale_prelu", 1, &in_scale);
        //status |= vxSetNodeUniform(nodObj, "out_scale_prelu", 1, &out_scale);
        status |= vxSetNodeUniform(nodObj, "scale_inOut", 1, &scale_inOut);
    }

    if(status != VX_SUCCESS)
    {
        printf("Set uniform failed(prelu).\n");
        return status;
    }

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));
    if(status != VX_SUCCESS)
    {
        printf("Set node attribute failed(prelu).\n");
        return status;
    }

    return VX_SUCCESS;
}

static vx_param_description_t vxParametricReluKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED}
};

#ifdef __cpluplus
extern "C" {
#endif
vx_kernel_description_t vxParametricReluKernelInfo =
{
    VX_KERNEL_ENUM_PARAMETRICRELU,
    VX_KERNEL_NAME_PARAMETRICRELU,
    NULL,
    vxParametricReluKernelParam,
    (sizeof(vxParametricReluKernelParam) / sizeof(vxParametricReluKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxParametricReluInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxParametricReluKernelInfo_int8 =
{
    VX_KERNEL_ENUM_PARAMETRICRELU,
    VX_KERNEL_NAME_PARAMETRICRELU_INT8,
    NULL,
    vxParametricReluKernelParam,
    (sizeof(vxParametricReluKernelParam) / sizeof(vxParametricReluKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxParametricReluInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxParametricReluKernelInfo_int8_opt =
{
    VX_KERNEL_ENUM_PARAMETRICRELU,
    VX_KERNEL_NAME_PARAMETRICRELU_INT8_OPT,
    NULL,
    vxParametricReluKernelParam,
    (sizeof(vxParametricReluKernelParam) / sizeof(vxParametricReluKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxParametricReluInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxParametricReluKernelInfo_int8_opt1 =
{
    VX_KERNEL_ENUM_PARAMETRICRELU,
    VX_KERNEL_NAME_PARAMETRICRELU_INT8_OPT1,
    NULL,
    vxParametricReluKernelParam,
    (sizeof(vxParametricReluKernelParam) / sizeof(vxParametricReluKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxParametricReluInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxParametricReluKernelInfo_int8_fp16 =
{
    VX_KERNEL_ENUM_PARAMETRICRELU,
    VX_KERNEL_NAME_PARAMETRICRELU_INT8_FP16,
    NULL,
    vxParametricReluKernelParam,
    (sizeof(vxParametricReluKernelParam) / sizeof(vxParametricReluKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxParametricReluInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxParametricReluKernelInfo_uint8_uint8 =
{
    VX_KERNEL_ENUM_PARAMETRICRELU,
    VX_KERNEL_NAME_PARAMETRICRELU_UINT8_UINT8,
    NULL,
    vxParametricReluKernelParam,
    (sizeof(vxParametricReluKernelParam) / sizeof(vxParametricReluKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxParametricReluInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxParametricReluKernelInfo_uint8_opt =
{
    VX_KERNEL_ENUM_PARAMETRICRELU,
    VX_KERNEL_NAME_PARAMETRICRELU_UINT8_2D,
    NULL,
    vxParametricReluKernelParam,
    (sizeof(vxParametricReluKernelParam) / sizeof(vxParametricReluKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxParametricReluInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxParametricReluKernelInfo_int16_int16 =
{
    VX_KERNEL_ENUM_PARAMETRICRELU,
    VX_KERNEL_NAME_PARAMETRICRELU_INT16_INT16,
    NULL,
    vxParametricReluKernelParam,
    (sizeof(vxParametricReluKernelParam) / sizeof(vxParametricReluKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxParametricReluInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxParametricReluKernelInfo_int16_int16_opt =
{
    VX_KERNEL_ENUM_PARAMETRICRELU,
    VX_KERNEL_NAME_PARAMETRICRELU_INT16_INT16_OPT,
    NULL,
    vxParametricReluKernelParam,
    (sizeof(vxParametricReluKernelParam) / sizeof(vxParametricReluKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxParametricReluInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxParametricReluKernelInfo_int16_int16_opt1 =
{
    VX_KERNEL_ENUM_PARAMETRICRELU,
    VX_KERNEL_NAME_PARAMETRICRELU_INT16_INT16_OPT1,
    NULL,
    vxParametricReluKernelParam,
    (sizeof(vxParametricReluKernelParam) / sizeof(vxParametricReluKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxParametricReluInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxParametricReluKernelInfo_fp16_uint8 =
{
    VX_KERNEL_ENUM_PARAMETRICRELU,
    VX_KERNEL_NAME_PARAMETRICRELU_FP16_UINT8,
    NULL,
    vxParametricReluKernelParam,
    (sizeof(vxParametricReluKernelParam) / sizeof(vxParametricReluKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxParametricReluInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxParametricReluKernelInfo_fp16_int16 =
{
    VX_KERNEL_ENUM_PARAMETRICRELU,
    VX_KERNEL_NAME_PARAMETRICRELU_FP16_INT16,
    NULL,
    vxParametricReluKernelParam,
    (sizeof(vxParametricReluKernelParam) / sizeof(vxParametricReluKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxParametricReluInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxParametricReluKernelInfo_int16_fp16 =
{
    VX_KERNEL_ENUM_PARAMETRICRELU,
    VX_KERNEL_NAME_PARAMETRICRELU_INT16_FP16,
    NULL,
    vxParametricReluKernelParam,
    (sizeof(vxParametricReluKernelParam) / sizeof(vxParametricReluKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxParametricReluInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxParametricReluKernelInfo_uint8_fp16 =
{
    VX_KERNEL_ENUM_PARAMETRICRELU,
    VX_KERNEL_NAME_PARAMETRICRELU_UINT8_FP16,
    NULL,
    vxParametricReluKernelParam,
    (sizeof(vxParametricReluKernelParam) / sizeof(vxParametricReluKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxParametricReluInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxParametricReluKernelInfo_uint8_fp16_2d =
{
    VX_KERNEL_ENUM_PARAMETRICRELU,
    VX_KERNEL_NAME_PARAMETRICRELU_UINT8_FP16_2D,
    NULL,
    vxParametricReluKernelParam,
    (sizeof(vxParametricReluKernelParam) / sizeof(vxParametricReluKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxParametricReluInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxParametricReluKernelInfo_fp16_int8 =
{
    VX_KERNEL_ENUM_PARAMETRICRELU,
    VX_KERNEL_NAME_PARAMETRICRELU_FP16_INT8,
    NULL,
    vxParametricReluKernelParam,
    (sizeof(vxParametricReluKernelParam) / sizeof(vxParametricReluKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxParametricReluInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_PRELU_list[] =
{
    NULL,
    &vxParametricReluKernelInfo,
    &vxParametricReluKernelInfo_int8,
    &vxParametricReluKernelInfo_int8_fp16,
    &vxParametricReluKernelInfo_uint8_uint8,
    &vxParametricReluKernelInfo_int16_int16,
    &vxParametricReluKernelInfo_fp16_uint8,
    &vxParametricReluKernelInfo_fp16_int16,
    &vxParametricReluKernelInfo_int16_fp16,
    &vxParametricReluKernelInfo_uint8_fp16,
    &vxParametricReluKernelInfo_int8_opt,
    &vxParametricReluKernelInfo_int16_int16_opt,
    &vxParametricReluKernelInfo_int8_opt1,
    &vxParametricReluKernelInfo_int16_int16_opt1,
    &vxParametricReluKernelInfo_uint8_opt,
    &vxParametricReluKernelInfo_uint8_fp16_2d,
    &vxParametricReluKernelInfo_fp16_int8,
    NULL
};
#ifdef __cpluplus
}
#endif
