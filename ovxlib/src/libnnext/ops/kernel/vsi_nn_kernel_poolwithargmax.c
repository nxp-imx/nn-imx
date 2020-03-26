/****************************************************************************
*
*    Copyright (c) 2020 Vivante Corporation
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
#include "vsi_nn_log.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_math.h"
#include "client/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"

vsi_status VX_CALLBACK vxpoolingWithArgmaxInitializer
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

    vsi_status status = VX_SUCCESS;
    vx_tensor tensorIn = (vx_tensor)paramObj[0];
    vsi_nn_type_e inDataType, outDataType, axsFormat;

    vx_tensor  tensorOut        = (vx_tensor)paramObj[1];
    vx_tensor  tensorAxis       = (vx_tensor)paramObj[2];
    int8_t     infixpoint       = 0;
    int8_t     outfixpoint      = 0;
    vx_float32 scaleSF          = 0.0f;
    vx_float32 factorIn         = 0.0f;
    vx_float32 factorOut        = 0.0f;

    vx_float32 scaleIn          = 0;
    vx_float32 scaleOut         = 0;
    int32_t    output_ZP        = 0;
    int32_t    input_ZP         = 0;
    vx_uint16  M0               = 0;
    vx_int8    postShift        = 0;
    vsi_bool   enable_image_2d  = FALSE;
    vsi_nn_tensor_attr_t attr[3];

    memset(&attr[0], 0, sizeof(vsi_nn_tensor_attr_t));
    memset(&attr[1], 0, sizeof(vsi_nn_tensor_attr_t));
    memset(&attr[2], 0, sizeof(vsi_nn_tensor_attr_t));

    status  = vsi_nn_vxGetTensorAttr(tensorIn, &attr[0]);
    status |= vsi_nn_vxGetTensorAttr(tensorAxis, &attr[1]);
    status |= vsi_nn_vxGetTensorAttr(tensorOut, &attr[2]);

    if(VX_SUCCESS != status)
        return status;

    inDataType = attr[0].dtype.vx_type;
    infixpoint = attr[0].dtype.fl;
    if (attr[0].dtype.qnt_type == VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC)
    {
        input_ZP = attr[0].dtype.zero_point;
        scaleIn = attr[0].dtype.scale;
    }

    outDataType = attr[2].dtype.vx_type;
    outfixpoint = attr[2].dtype.fl;
    if (attr[2].dtype.qnt_type == VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC)
    {
        output_ZP = attr[2].dtype.zero_point;
        scaleOut = attr[2].dtype.scale;
    }

    axsFormat = attr[1].dtype.vx_type;

    if (inDataType == VSI_NN_TYPE_BFLOAT16 && outDataType == VSI_NN_TYPE_BFLOAT16)
    {
        inDataType = VSI_NN_TYPE_FLOAT16;
        outDataType = VSI_NN_TYPE_FLOAT16;
    }

    if ( inDataType == VSI_NN_TYPE_UINT8 && axsFormat == VSI_NN_TYPE_UINT8
        && (outDataType == VSI_NN_TYPE_UINT8/* || outDataType == VSI_NN_TYPE_FLOAT16*/))
    {
        vsi_nn_GetFP32MultiAndPostShift(scaleIn / scaleOut, &M0, &postShift);

        if (attr[0].size[2] == 1  || attr[0].dim_num < 3)
        {
            attr[0].size[2] = 1;
            enable_image_2d = TRUE;
        }
    }

    if (infixpoint > 0)
    {
        factorIn = (1.0f / ((vx_float32) (1 << infixpoint)));
    }
    else
    {
        factorIn = ((vx_float32) (1 << -infixpoint));
    }

    if (outfixpoint > 0)
    {
        factorOut = (vx_float32)(1 << outfixpoint);
    }
    else
    {
        factorOut = (1.0f / (vx_float32)(1 << -outfixpoint));
    }

    scaleSF = factorIn * factorOut;

    if(inDataType == VSI_NN_TYPE_INT8 || inDataType == VSI_NN_TYPE_UINT8)
    {
        if (enable_image_2d)
        {
            shaderParam.workDim             = 2;
            shaderParam.globalWorkOffset[0] = 0;
            shaderParam.globalWorkOffset[1] = 0;
            shaderParam.globalWorkScale[0]  = 16;
            shaderParam.globalWorkScale[1]  = 2;
            shaderParam.globalWorkSize[0]   = gcmALIGN((attr[0].size[0] + shaderParam.globalWorkScale[0] - 1)
                / shaderParam.globalWorkScale[0], 4);
            shaderParam.globalWorkSize[1]   = (attr[0].size[1] + shaderParam.globalWorkScale[1] - 1)
                / shaderParam.globalWorkScale[1];
        }
        else
        {
            shaderParam.globalWorkOffset[0] = 0;
            shaderParam.globalWorkOffset[1] = 0;
            shaderParam.globalWorkOffset[2] = 0;
            shaderParam.globalWorkScale[0]  = 16;
            shaderParam.globalWorkScale[1]  = 2;
            shaderParam.globalWorkScale[2]  = 1;
            shaderParam.globalWorkSize[0]   = gcmALIGN((attr[0].size[0] + shaderParam.globalWorkScale[0] - 1)
                / shaderParam.globalWorkScale[0], 4);
            shaderParam.globalWorkSize[1]   = (attr[0].size[1] + shaderParam.globalWorkScale[1] - 1)
                / shaderParam.globalWorkScale[1];
            shaderParam.globalWorkSize[2]   = attr[0].size[2];
        }

        if(inDataType == VSI_NN_TYPE_INT8 &&
            (outDataType == VSI_NN_TYPE_INT8 || outDataType == VSI_NN_TYPE_FLOAT16))
        {
            vx_uint32 poolingEncodeInt8_0[16] = {
                0x55555555, // TCfg
                0x50505050, // ASelt
                0x32321010, 0x76765454, // ABin
                0xaaaaaaaa, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000700, // AccumType, ConstantType, and PostShift
                0x00400080, 0x00100020, 0x00400080, 0x00100020,
                0x00400080, 0x00100020, 0x00400080, 0x00100020 // Constant
            };
            vx_uint32 poolingEncodeInt8_1[16] = {
                0x55555555, // TCfg
                0x50505050, // ASelt
                0xbaba9898, 0xfefedcdc, // ABin
                0xaaaaaaaa, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000700, // AccumType, ConstantType, and PostShift
                0x00400080, 0x00100020, 0x00400080, 0x00100020,
                0x00400080, 0x00100020, 0x00400080, 0x00100020 // Constant
            };
            vx_uint32 uniConvertInt8FstFp32_4x4[16] = {
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00010000, 0x00030002, // ABin
                0x02020202, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000000, 0x00000001, 0x00000000,
                0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
            };
            vx_uint32 uniConvertInt8SecFp32_4x4[16] = {
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00050004, 0x00070006, // ABin
                0x02020202, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000000, 0x00000001, 0x00000000,
                0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
            };
            vx_uint32 uniConvertInt32toInt8_2x8[16] = {
                0x33333333, // TCfg
                0x11110000, // ASelt
                0x03020100, 0x03020100, // ABin
                0x00000000, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            };
            vx_uint32 uniPoolQuantInt8_2x8[16] = {
                0x11111111, // TCfg
                0x00000000, // ASelt
                0x06040200, 0x0e0c0a08, // ABin
                0x11111111, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            };
            vx_uint32 uniQuantInOutInt8Even_2x8[16] = {
                0x11111111, // TCfg
                0x00000000, // ASelt
                0x06040200, 0x0e0c0a08, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000300, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001,
                0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
            };

            if(outDataType == VSI_NN_TYPE_INT8 && infixpoint != outfixpoint)
            {
                if(infixpoint > outfixpoint)
                {
                    uniQuantInOutInt8Even_2x8[7] = uniQuantInOutInt8Even_2x8[7] | (infixpoint - outfixpoint);
                }
                else
                {
                    vx_uint32 multiply       = (1 << (outfixpoint - infixpoint));
                    vx_uint32 i              = 0;

                    for (i = 8; i < 16; i++)
                    {
                        uniQuantInOutInt8Even_2x8[i] = multiply;
                    }
                }
                status = vxSetNodeUniform(nodObj, "uniQuantInOutInt8Even_2x8", 1, uniQuantInOutInt8Even_2x8);
                status |= vxSetNodeUniform(nodObj, "poolingEncodeInt8_0_opt", 1, poolingEncodeInt8_0);
                status |= vxSetNodeUniform(nodObj, "poolingEncodeInt8_1_opt", 1, poolingEncodeInt8_1);
            }
            else
            {
                status = vxSetNodeUniform(nodObj, "uniConvertInt8FstFp32_4x4", 1, uniConvertInt8FstFp32_4x4);
                status |= vxSetNodeUniform(nodObj, "uniConvertInt8SecFp32_4x4", 1, uniConvertInt8SecFp32_4x4);
                status |= vxSetNodeUniform(nodObj, "uniConvertInt32toInt8_2x8", 1, uniConvertInt32toInt8_2x8);
                status |= vxSetNodeUniform(nodObj, "poolingEncodeInt8_0", 1, poolingEncodeInt8_0);
                status |= vxSetNodeUniform(nodObj, "poolingEncodeInt8_1", 1, poolingEncodeInt8_1);
                if(outDataType == VSI_NN_TYPE_INT8)
                {
                    status |= vxSetNodeUniform(nodObj, "scaleSF_i8", 1, &scaleSF);
                }
                else if(outDataType == VSI_NN_TYPE_FLOAT16)
                {
                    status |= vxSetNodeUniform(nodObj, "uniPoolQuantInt8_2x8", 1,
                        uniPoolQuantInt8_2x8);
                    status |= vxSetNodeUniform(nodObj, "inputfl_scale_i8", 1, &factorIn);
                }
            }
        }
        else
        {
            // uniforms
            uint32_t poolingEncodeInt8_0[16] = {
                0x55555555, // TCfg
                0x50505050, // ASelt
                0x32321010, 0x76765454, // ABin
                0xaaaaaaaa, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000700, // AccumType, ConstantType, and PostShift
                0x00400080, 0x00100020, 0x00400080, 0x00100020,
                0x00400080, 0x00100020, 0x00400080, 0x00100020 // Constant
            };
            uint32_t poolingEncodeInt8_1[16] = {
                0x55555555, // TCfg
                0x50505050, // ASelt
                0xbaba9898, 0xfefedcdc, // ABin
                0xaaaaaaaa, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000700, // AccumType, ConstantType, and PostShift
                0x00400080, 0x00100020, 0x00400080, 0x00100020,
                0x00400080, 0x00100020, 0x00400080, 0x00100020 // Constant
            };
            vx_uint32 uniU8EvenBinSubZP_MulM_2x8[16] = {
                0x99999999, // TCfg
                0x44444444, // ASelt
                0x06040200, 0x0e0c0a08, // ABin
                0xaaaaaaaa, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00020001, 0x00020001, 0x00020001, 0x00020001,
                0x00020001, 0x00020001, 0x00020001, 0x00020001 // Constant
            };
            vx_uint32 uniEncodeUint8_4x8[16] = {
                0x55555555, 0x55555555, // TCfg
                0x8628c020, 0x6ad0a49c, 0xe128bd8e, 0xacde96ac, 0xff9eeef1, // BinSelect
                0x00000700, // AccumType, ConstantType, and PostShift
                0x10204080, 0x10204080, 0x10204080, 0x10204080,
                0x10204080, 0x10204080, 0x10204080, 0x10204080 // Constant
            };
            vx_uint32 uniS16AddOutZP_2x8[16] = {
                0x55555555, // TCfg
                0x44444444, // ASelt
                0x03020100, 0x07060504, // ABin
                0xaaaaaaaa, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00010001, 0x00010001, 0x00010001, 0x00010001,
                0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
            };

            if(inDataType == VSI_NN_TYPE_UINT8 && outDataType == VSI_NN_TYPE_UINT8
                && axsFormat == VSI_NN_TYPE_UINT8)
            {
                vx_uint32 idx                   = 0;
                vx_uint32 packed_outputZP[4]    = {0};

                for (idx = 0; idx < 4; idx ++)
                {
                    vx_uint8  zp = (vx_uint8)(output_ZP & 0xFF);
                    packed_outputZP[idx] = (zp << 24) | (zp << 16) | (zp << 8) | zp;
                }

                uniU8EvenBinSubZP_MulM_2x8[7] |= postShift;

                for (idx = 8; idx < 16; idx ++)
                {
                    uniU8EvenBinSubZP_MulM_2x8[idx] = (vx_uint32)((M0 << 16) | M0);
                }

                status |= vxSetNodeUniform(nodObj, "uniU8EvenBinSubZP_MulM_2x8",
                    1, uniU8EvenBinSubZP_MulM_2x8);
                status |= vxSetNodeUniform(nodObj, "uniEncodeUint8_4x8",
                    1, uniEncodeUint8_4x8);
                status |= vxSetNodeUniform(nodObj, "uniS16AddOutZP_2x8",
                    1, uniS16AddOutZP_2x8);
                status |= vxSetNodeUniform(nodObj, "packed_outputZP", 1, packed_outputZP);
                status |= vxSetNodeUniform(nodObj, "input_ZP", 1, &input_ZP);
            }
            else if(inDataType == VSI_NN_TYPE_UINT8 && !enable_image_2d)
            {
                status |= vxSetNodeUniform(nodObj, "uniEncodeUint8_4x8",
                    1, uniEncodeUint8_4x8);
            }
            else if(!enable_image_2d)
            {
                status |= vxSetNodeUniform(nodObj, "poolingEncodeInt8_0",
                    1, poolingEncodeInt8_0);
                status |= vxSetNodeUniform(nodObj, "poolingEncodeInt8_1",
                    1, poolingEncodeInt8_1);
            }

            if(outDataType == VSI_NN_TYPE_FLOAT16)
            {
                uint32_t uniConvertUint8ToFp32_4x4[16] = {
                    0x09090909, // TCfg
                    0x04040404, // ASelt
                    0x00010000, 0x00030002, // ABin
                    0x0a0a0a0a, // BSelt
                    0x00000000, 0x00000000, // BBin
                    0x00000100, // AccumType, ConstantType, and PostShift
                    0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000,
                    0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000 // Constant
                };
                uint32_t uniConvertSubZpUint8Fp32_4x4[16] = {
                    0x09090905, // TCfg
                    0x04040404, // ASelt
                    0x00050004, 0x00070006, // ABin
                    0x0a0a0a0a, // BSelt
                    0x00000000, 0x00000000, // BBin
                    0x00000100, // AccumType, ConstantType, and PostShift
                    0xbc003c00, 0x00000000, 0x3c003c00, 0x00000000,
                    0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000 // Constant
                };
                uint32_t uniPackHalf8_2x8[16] = {
                    0x11111111, // TCfg
                    0x11110000, // ASelt
                    0x06040200, 0x06040200, // ABin
                    0x22222222, // BSelt
                    0x00000000, 0x00000000, // BBin
                    0x00002100, // AccumType, ConstantType, and PostShift
                    0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00,
                    0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
                };
                vx_uint32 uniConvertEvenU8ToFp32_4x4[16] = {
                    0x09090905, // TCfg
                    0x04040404, // ASelt
                    0x00020000, 0x00060004, // ABin
                    0x0a0a0a0a, // BSelt
                    0x00000000, 0x00000000, // BBin
                    0x00000400, // AccumType, ConstantType, and PostShift
                    0xffff0001, 0x00000000, 0x00010001, 0x00000000,
                    0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
                };
                vx_uint32 uniConvertEvenU8SubZpToFp32_4x4[16] = {
                    0x09090909, // TCfg
                    0x04040404, // ASelt
                    0x000a0008, 0x000e000c, // ABin
                    0x0a0a0a0a, // BSelt
                    0x00000000, 0x00000000, // BBin
                    0x00000400, // AccumType, ConstantType, and PostShift
                    0x00010001, 0x00000000, 0x00010001, 0x00000000,
                    0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
                };

                status |= vxSetNodeUniform(nodObj, "uniPackHalf8_2x8",
                    1, uniPackHalf8_2x8);
                status |= vxSetNodeUniform(nodObj, "uniConvertUint8ToFp32_4x4",
                    1, uniConvertUint8ToFp32_4x4);
                status |= vxSetNodeUniform(nodObj, "uniConvertSubZpUint8Fp32_4x4",
                    1, uniConvertSubZpUint8Fp32_4x4);

                status |= vxSetNodeUniform(nodObj, "uniConvertEvenU8ToFp32_4x4",
                    1, uniConvertEvenU8ToFp32_4x4);
                status |= vxSetNodeUniform(nodObj, "uniConvertEvenU8SubZpToFp32_4x4",
                    1, uniConvertEvenU8SubZpToFp32_4x4);

                status |= vxSetNodeUniform(nodObj, "inputScale", 1, &scaleIn);
                status |= vxSetNodeUniform(nodObj, "input_ZP", 1, &input_ZP);
            }
        }
    }
    else
    {
        shaderParam.globalWorkOffset[0] = 0;
        shaderParam.globalWorkOffset[1] = 0;
        shaderParam.globalWorkOffset[2] = 0;
        shaderParam.globalWorkScale[0]  = 8;
        shaderParam.globalWorkScale[1]  = 2;
        shaderParam.globalWorkScale[2]  = 1;
        shaderParam.globalWorkSize[0]   = gcmALIGN((attr[0].size[0] + shaderParam.globalWorkScale[0] - 1)
            / shaderParam.globalWorkScale[0], 4);
        shaderParam.globalWorkSize[1]   = (attr[0].size[1] + shaderParam.globalWorkScale[1] - 1)
            / shaderParam.globalWorkScale[1];
        shaderParam.globalWorkSize[2]   = attr[0].size[2];
        {
            // uniforms
            uint32_t poolingEncode[16] = {
                0x55555555, // TCfg
                0x50505050, // ASelt
                0x32321010, 0x76765454, // ABin
                0xaaaaaaaa, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000700, // AccumType, ConstantType, and PostShift
                0x00400080, 0x00100020, 0x00400080, 0x00100020,
                0x00400080, 0x00100020, 0x00400080, 0x00100020 // Constant
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
            uint32_t uniPackHalf8_2x8[16] = {
                0x11111111, // TCfg
                0x11110000, // ASelt
                0x06040200, 0x06040200, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002100, // AccumType, ConstantType, and PostShift
                0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00,
                0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
            };
            vx_uint32 uniQuantInOutInt16Even_4x4[16] = {
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00020000, 0x00060004, // ABin
                0x02020202, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000300, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000000, 0x00000001, 0x00000000,
                0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
            };

            if(inDataType == VSI_NN_TYPE_FLOAT16)
            {
                status |= vxSetNodeUniform(nodObj, "poolingEncode", 1, poolingEncode);
            }
            else
            {
                if(inDataType == VSI_NN_TYPE_FLOAT16 && outDataType == VSI_NN_TYPE_FLOAT16
                    && infixpoint != outfixpoint)
                {
                    if(infixpoint > outfixpoint)
                    {
                        uniQuantInOutInt16Even_4x4[7] = uniQuantInOutInt16Even_4x4[7] | (infixpoint - outfixpoint);
                    }
                    else
                    {
                        vx_uint32 multiply       = (1 << (outfixpoint - infixpoint));
                        vx_uint32 i              = 0;

                        for (i = 8; i < 16; i+=2)
                        {
                            uniQuantInOutInt16Even_4x4[i] = multiply;
                        }
                    }

                    status |= vxSetNodeUniform(nodObj, "uniQuantInOutInt16Even_4x4", 1, uniQuantInOutInt16Even_4x4);
                    status |= vxSetNodeUniform(nodObj, "poolingEncode_opt", 1, poolingEncode);
                }
                else
                {
                    if(outDataType == VSI_NN_TYPE_FLOAT16)
                    {
                        status |= vxSetNodeUniform(nodObj, "uniPackHalf8_2x8_2",
                            1, uniPackHalf8_2x8);
                        status |= vxSetNodeUniform(nodObj, "input_fl_scale_i16", 1, &factorIn);
                    }
                    status |= vxSetNodeUniform(nodObj, "uniConvertDirInt16Fp32_4x4",
                        1, uniConvertDirInt16Fp32_4x4);
                    status |= vxSetNodeUniform(nodObj, "uniConvertEndInt16Fp32_4x4",
                        1, uniConvertEndInt16Fp32_4x4);
                    status |= vxSetNodeUniform(nodObj, "uniConvertInt32toInt16_2x8",
                        1, uniConvertInt32toUint8_2x8);
                    status |= vxSetNodeUniform(nodObj, "poolingEncode2", 1, poolingEncode);
                    status |= vxSetNodeUniform(nodObj, "scaleSF", 1, &scaleSF);
                }
            }
        }
    }

    if(status != VX_SUCCESS)
    {
        VSILOGE("Set uniform failed(poolwithargmax).\n");
        return status;
    }
    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));
    if(status != VX_SUCCESS)
    {
        VSILOGE("Set node attribute failed(poolwithargmax).\n");
        return status;
    }

    return VX_SUCCESS;
}

static vx_param_description_t vxpoolingWithArgmaxKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

#ifdef __cplusplus
extern "C" {
#endif
vx_kernel_description_t vxPoolingWithArgmaxInfo =
{
    VX_KERNEL_ENUM_POOLING_WITH_ARGMAX,
    VX_KERNEL_NAME_POOLING_WITH_ARGMAX,
    NULL,
    vxpoolingWithArgmaxKernelParam,
    (sizeof(vxpoolingWithArgmaxKernelParam) / sizeof(vxpoolingWithArgmaxKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxpoolingWithArgmaxInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxPoolingWithArgmaxInfoInt8 =
{
    VX_KERNEL_ENUM_POOLING_WITH_ARGMAX,
    VX_KERNEL_NAME_POOLING_WITH_ARGMAX_INT8,
    NULL,
    vxpoolingWithArgmaxKernelParam,
    (sizeof(vxpoolingWithArgmaxKernelParam) / sizeof(vxpoolingWithArgmaxKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxpoolingWithArgmaxInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxPoolingWithArgmaxInfoInt8_opt =
{
    VX_KERNEL_ENUM_POOLING_WITH_ARGMAX,
    VX_KERNEL_NAME_POOLING_WITH_ARGMAX_INT8_OPT,
    NULL,
    vxpoolingWithArgmaxKernelParam,
    (sizeof(vxpoolingWithArgmaxKernelParam) / sizeof(vxpoolingWithArgmaxKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxpoolingWithArgmaxInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxPoolingWithArgmaxInfoUint8 =
{
    VX_KERNEL_ENUM_POOLING_WITH_ARGMAX,
    VX_KERNEL_NAME_POOLING_WITH_ARGMAX_UINT8,
    NULL,
    vxpoolingWithArgmaxKernelParam,
    (sizeof(vxpoolingWithArgmaxKernelParam) / sizeof(vxpoolingWithArgmaxKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxpoolingWithArgmaxInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxPoolingWithArgmaxInfoUint8_fp16 =
{
    VX_KERNEL_ENUM_POOLING_WITH_ARGMAX,
    VX_KERNEL_NAME_POOLING_WITH_ARGMAX_UINT8_FP16,
    NULL,
    vxpoolingWithArgmaxKernelParam,
    (sizeof(vxpoolingWithArgmaxKernelParam) / sizeof(vxpoolingWithArgmaxKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxpoolingWithArgmaxInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxPoolingWithArgmaxInfoInt16 =
{
    VX_KERNEL_ENUM_POOLING_WITH_ARGMAX,
    VX_KERNEL_NAME_POOLING_WITH_ARGMAX_INT16,
    NULL,
    vxpoolingWithArgmaxKernelParam,
    (sizeof(vxpoolingWithArgmaxKernelParam) / sizeof(vxpoolingWithArgmaxKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxpoolingWithArgmaxInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxPoolingWithArgmaxInfoInt16_int16 =
{
    VX_KERNEL_ENUM_POOLING_WITH_ARGMAX,
    VX_KERNEL_NAME_POOLING_WITH_ARGMAX_INT16_INT16,
    NULL,
    vxpoolingWithArgmaxKernelParam,
    (sizeof(vxpoolingWithArgmaxKernelParam) / sizeof(vxpoolingWithArgmaxKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxpoolingWithArgmaxInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxPoolingWithArgmaxInfoInt16_opt =
{
    VX_KERNEL_ENUM_POOLING_WITH_ARGMAX,
    VX_KERNEL_NAME_POOLING_WITH_ARGMAX_INT16_OPT,
    NULL,
    vxpoolingWithArgmaxKernelParam,
    (sizeof(vxpoolingWithArgmaxKernelParam) / sizeof(vxpoolingWithArgmaxKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxpoolingWithArgmaxInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxPoolingWithArgmaxInfoInt16_fp16 =
{
    VX_KERNEL_ENUM_POOLING_WITH_ARGMAX,
    VX_KERNEL_NAME_POOLING_WITH_ARGMAX_INT16_FP16,
    NULL,
    vxpoolingWithArgmaxKernelParam,
    (sizeof(vxpoolingWithArgmaxKernelParam) / sizeof(vxpoolingWithArgmaxKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxpoolingWithArgmaxInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxPoolingWithArgmaxInfoInt16_axInt16 =
{
    VX_KERNEL_ENUM_POOLING_WITH_ARGMAX,
    VX_KERNEL_NAME_POOLING_WITH_ARGMAX_INT16_AXINT16,
    NULL,
    vxpoolingWithArgmaxKernelParam,
    (sizeof(vxpoolingWithArgmaxKernelParam) / sizeof(vxpoolingWithArgmaxKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxpoolingWithArgmaxInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxPoolingWithArgmaxInfoUint8_fp16_fp16 =
{
    VX_KERNEL_ENUM_POOLING_WITH_ARGMAX,
    VX_KERNEL_NAME_POOLING_WITH_ARGMAX_UINT8_FP16_FP16,
    NULL,
    vxpoolingWithArgmaxKernelParam,
    (sizeof(vxpoolingWithArgmaxKernelParam) / sizeof(vxpoolingWithArgmaxKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxpoolingWithArgmaxInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxPoolingWithArgmaxInfoInt8_int8 =
{
    VX_KERNEL_ENUM_POOLING_WITH_ARGMAX,
    VX_KERNEL_NAME_POOLING_WITH_ARGMAX_INT8_INT8,
    NULL,
    vxpoolingWithArgmaxKernelParam,
    (sizeof(vxpoolingWithArgmaxKernelParam) / sizeof(vxpoolingWithArgmaxKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxpoolingWithArgmaxInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxPoolingWithArgmaxInfoInt8_fp16 =
{
    VX_KERNEL_ENUM_POOLING_WITH_ARGMAX,
    VX_KERNEL_NAME_POOLING_WITH_ARGMAX_INT8_FP16,
    NULL,
    vxpoolingWithArgmaxKernelParam,
    (sizeof(vxpoolingWithArgmaxKernelParam) / sizeof(vxpoolingWithArgmaxKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxpoolingWithArgmaxInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxPoolingWithArgmaxInfoUint8_2D =
{
    VX_KERNEL_ENUM_POOLING_WITH_ARGMAX,
    VX_KERNEL_NAME_POOLING_WITH_ARGMAX_UINT8_2D,
    NULL,
    vxpoolingWithArgmaxKernelParam,
    (sizeof(vxpoolingWithArgmaxKernelParam) / sizeof(vxpoolingWithArgmaxKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxpoolingWithArgmaxInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_POOLWITHARGMAX_list[] =
{
    NULL,
    &vxPoolingWithArgmaxInfo,
    &vxPoolingWithArgmaxInfoInt8,
    &vxPoolingWithArgmaxInfoUint8,
    &vxPoolingWithArgmaxInfoUint8_fp16,
    &vxPoolingWithArgmaxInfoInt16,
    &vxPoolingWithArgmaxInfoUint8_fp16_fp16,
    &vxPoolingWithArgmaxInfoInt8_int8,
    &vxPoolingWithArgmaxInfoInt8_fp16,
    &vxPoolingWithArgmaxInfoInt16_axInt16,
    &vxPoolingWithArgmaxInfoInt16_fp16,
    &vxPoolingWithArgmaxInfoInt16_int16,
    &vxPoolingWithArgmaxInfoInt16_opt,
    &vxPoolingWithArgmaxInfoInt8_opt,
    &vxPoolingWithArgmaxInfoUint8_2D,
    NULL
};
#ifdef __cplusplus
}
#endif
