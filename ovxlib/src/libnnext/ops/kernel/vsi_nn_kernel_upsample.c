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

vsi_status VX_CALLBACK vxunpoolingInitializer
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
    vx_tensor tensorAx = (vx_tensor)paramObj[1];
    vsi_nn_type_e inDataType, axDataType, outDataType;
    vx_tensor   tensorOut       = (vx_tensor)paramObj[2];
    int8_t     infixpoint       = 0;
    int8_t     outfixpoint      = 0;
    vx_float32 scaleSF          = 0.0f;
    vx_float32 factorIn         = 0.0f;
    vx_float32 factorOut        = 0.0f;

    vx_float32 scaleIn          = 1.0f;
    vx_float32 scaleOut         = 1.0f;
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
    status |= vsi_nn_vxGetTensorAttr(tensorAx, &attr[1]);
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
    else if (attr[0].dtype.qnt_type == VSI_NN_QNT_TYPE_DFP)
    {
        input_ZP = 0;
        if (infixpoint > 0)
        {
            scaleIn = (1.0f / ((vx_float32) (1 << infixpoint)));
        }
        else
        {
            scaleIn = ((vx_float32) (1 << -infixpoint));
        }
    }

    outDataType = attr[2].dtype.vx_type;
    outfixpoint = attr[2].dtype.fl;

    if (attr[2].dtype.qnt_type == VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC)
    {
        output_ZP = attr[2].dtype.zero_point;
        scaleOut = attr[2].dtype.scale;
    }
    else if (attr[2].dtype.qnt_type == VSI_NN_QNT_TYPE_DFP)
    {
        output_ZP = 0;
        if (outfixpoint > 0)
        {
            scaleOut = (1.0f / ((vx_float32) (1 << outfixpoint)));
        }
        else
        {
            scaleOut = ((vx_float32) (1 << -outfixpoint));
        }
    }

    axDataType = attr[1].dtype.vx_type;

    if (inDataType == VSI_NN_TYPE_BFLOAT16 && outDataType == VSI_NN_TYPE_BFLOAT16)
    {
        inDataType = VSI_NN_TYPE_FLOAT16;
        outDataType = VSI_NN_TYPE_FLOAT16;
    }

    if ( (inDataType == VSI_NN_TYPE_UINT8 || inDataType == VSI_NN_TYPE_FLOAT16)
        && outDataType == VSI_NN_TYPE_UINT8 && axDataType == VSI_NN_TYPE_UINT8)
    {
        vsi_nn_GetFP32MultiAndPostShift(scaleIn / scaleOut, &M0, &postShift);

        if (attr[0].size[2] == 1 || attr[0].dim_num < 3)
        {
            attr[0].size[2] = 1;
            enable_image_2d = TRUE;
        }
    }

    if ( (inDataType == VSI_NN_TYPE_INT8 || inDataType == VSI_NN_TYPE_FLOAT16)
        && outDataType == VSI_NN_TYPE_INT8 )
    {
        vsi_nn_GetFP32MultiAndPostShift(scaleIn / scaleOut, &M0, &postShift);
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

    if((inDataType == VSI_NN_TYPE_INT8 && outDataType == VSI_NN_TYPE_INT8)
    || (inDataType == VSI_NN_TYPE_UINT8 && outDataType == VSI_NN_TYPE_UINT8)
    || (inDataType == VSI_NN_TYPE_UINT8 && outDataType == VSI_NN_TYPE_FLOAT16)
    || (inDataType == VSI_NN_TYPE_FLOAT16 && outDataType == VSI_NN_TYPE_UINT8 && axDataType == VSI_NN_TYPE_UINT8)
        )
    {
        vx_bool is_same_quant_type = (vx_bool)((attr[0].dtype.fl == attr[2].dtype.fl
                                        && attr[0].dtype.qnt_type  == VSI_NN_QNT_TYPE_DFP
                                        && attr[2].dtype.qnt_type == VSI_NN_QNT_TYPE_DFP)
                                      || ((attr[0].dtype.zero_point == attr[2].dtype.zero_point)
                                         && (attr[0].dtype.scale == attr[2].dtype.scale)
                                         && attr[0].dtype.qnt_type  == VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC
                                         && attr[2].dtype.qnt_type == VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC));
        if (enable_image_2d)
        {
            shaderParam.workDim             = 2;
            shaderParam.globalWorkOffset[0] = 0;
            shaderParam.globalWorkOffset[1] = 0;
            shaderParam.globalWorkScale[0]  = 8;
            shaderParam.globalWorkScale[1]  = 1;
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
            shaderParam.globalWorkScale[0]  = 8;
            shaderParam.globalWorkScale[1]  = 1;
            shaderParam.globalWorkScale[2]  = 1;
            shaderParam.globalWorkSize[0]   = gcmALIGN((attr[0].size[0] + shaderParam.globalWorkScale[0] - 1)
                / shaderParam.globalWorkScale[0], 4);
            shaderParam.globalWorkSize[1]   = (attr[0].size[1] + shaderParam.globalWorkScale[1] - 1)
                / shaderParam.globalWorkScale[1];
            shaderParam.globalWorkSize[2]   = attr[0].size[2];
        }

        if (inDataType == VSI_NN_TYPE_INT8 && outDataType == VSI_NN_TYPE_INT8 && is_same_quant_type)
        {
            status |= vxSetNodeUniform(nodObj, "input_ZP", 1, &input_ZP);
        }
        else if((inDataType == VSI_NN_TYPE_INT8 && outDataType == VSI_NN_TYPE_INT8) && (!is_same_quant_type))
        {
            vx_uint32 uniU8SubZP_MulM_2x8[16] = {
                0x99999999, // TCfg
                0x44444444, // ASelt
                0x03020100, 0x07060504, // ABin
                0xaaaaaaaa, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00010001, 0x00010001, 0x00010001, 0x00010001,
                0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
            };
            vx_uint32 uniU8SubZP_MulM_Hi_2x8[16] = {
                0x99999999, // TCfg
                0x44444444, // ASelt
                0x0b0a0908, 0x0f0e0d0c, // ABin
                0xaaaaaaaa, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00010001, 0x00010001, 0x00010001, 0x00010001,
                0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
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
            vx_uint32 uniS16AddOutZP_Hi_2x8[16] = {
                0x55555555, // TCfg
                0x44444444, // ASelt
                0x0b0a0908, 0x0f0e0d0c, // ABin
                0xaaaaaaaa, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00010001, 0x00010001, 0x00010001, 0x00010001,
                0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
            };

            vx_uint32 idx                   = 0;
            vx_uint32 packed_outputZP[4]    = {0};

            for (idx = 0; idx < 4; idx ++)
            {
                vx_uint16  zp = (vx_uint16)(output_ZP & 0xFFFF);
                packed_outputZP[idx] = (zp << 16) | zp;
            }

            uniU8SubZP_MulM_2x8[7]    |= postShift;
            uniU8SubZP_MulM_Hi_2x8[7] |= postShift;

            for (idx = 8; idx < 16; idx ++)
            {
                uniU8SubZP_MulM_2x8[idx]    = (vx_uint32)((M0 << 16) | M0);
                uniU8SubZP_MulM_Hi_2x8[idx] = (vx_uint32)((M0 << 16) | M0);
            }

            status |= vxSetNodeUniform(nodObj, "uniU8SubZP_MulM_2x8",
                1, uniU8SubZP_MulM_2x8);
            status |= vxSetNodeUniform(nodObj, "uniU8SubZP_MulM_Hi_2x8",
                1, uniU8SubZP_MulM_Hi_2x8);
            status |= vxSetNodeUniform(nodObj, "uniS16AddOutZP_2x8",
                1, uniS16AddOutZP_2x8);
            status |= vxSetNodeUniform(nodObj, "uniS16AddOutZP_Hi_2x8",
                1, uniS16AddOutZP_Hi_2x8);
            status |= vxSetNodeUniform(nodObj, "packed_outputZP", 1, packed_outputZP);
            status |= vxSetNodeUniform(nodObj, "input_ZP", 1, &input_ZP);

        }
        else if (!is_same_quant_type)
        {
            // uniforms
            uint32_t uniConvertDirUint8Fp32_4x4[16] = {
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00010000, 0x00030002, // ABin
                0x02020202, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
            };
            uint32_t uniConvertEndUint8Fp32_4x4[16] = {
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00050004, 0x00070006, // ABin
                0x02020202, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
            };
            uint32_t uniConvertTrdUint8Fp32_4x4[16] = {
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00090008, 0x000b000a, // ABin
                0x02020202, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
            };
            uint32_t uniConvertFthUint8Fp32_4x4[16] = {
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x000d000c, 0x000f000e, // ABin
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
            vx_uint32 uniMulMinusZpUint8_4x4[16] = {
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00010000, 0x00030002, // ABin
                0x01010101, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            };
            vx_uint32 uniMulMinusZp2Uint8_4x4[16] = {
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00050004, 0x00070006, // ABin
                0x01010101, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            };
            vx_uint32 uniMulMinusZp3Uint8_4x4[16] = {
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00090008, 0x000b000a, // ABin
                0x01010101, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            };
            vx_uint32 uniMulMinusZp4Uint8_4x4[16] = {
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x000d000c, 0x000f000e, // ABin
                0x01010101, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            };
            vx_uint32 uniF16MulMultipiler_PostShft_2x8[16] = {
                0x11111111, // TCfg
                0x00000000, // ASelt
                0x03020100, 0x07060504, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001,
                0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
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

            if(inDataType == VSI_NN_TYPE_FLOAT16 && axDataType == VSI_NN_TYPE_UINT8
                && outDataType == VSI_NN_TYPE_UINT8)
            {
                vx_uint32 idx                   = 0;
                vx_uint32 packed_outputZP[4]    = {0};

                for (idx = 0; idx < 4; idx ++)
                {
                    vx_uint8  zp = (vx_uint8)(output_ZP & 0xFF);
                    packed_outputZP[idx] = (zp << 24) | (zp << 16) | (zp << 8) | zp;
                }

                uniF16MulMultipiler_PostShft_2x8[7] |= postShift;

                for (idx = 8; idx < 16; idx ++)
                {
                    uniF16MulMultipiler_PostShft_2x8[idx] = (vx_uint32)(M0);
                }

                status |= vxSetNodeUniform(nodObj, "uniF16MulMultipiler_PostShft_2x8",
                    1, uniF16MulMultipiler_PostShft_2x8);
                status |= vxSetNodeUniform(nodObj, "uniS16AddOutZP_2x8",
                    1, uniS16AddOutZP_2x8);
                status |= vxSetNodeUniform(nodObj, "packed_outputZP", 1, packed_outputZP);
            }
            else if(inDataType == VSI_NN_TYPE_UINT8 && outDataType == VSI_NN_TYPE_FLOAT16)
            {
                status |= vxSetNodeUniform(nodObj, "uniConvertDirUint8Fp32_4x4",
                    1, uniConvertDirUint8Fp32_4x4);
                status |= vxSetNodeUniform(nodObj, "uniConvertEndUint8Fp32_4x4",
                    1, uniConvertEndUint8Fp32_4x4);
                status |= vxSetNodeUniform(nodObj, "uniConvertTrdUint8Fp32_4x4",
                    1, uniConvertTrdUint8Fp32_4x4);
                status |= vxSetNodeUniform(nodObj, "uniConvertFthUint8Fp32_4x4",
                    1, uniConvertFthUint8Fp32_4x4);
                status |= vxSetNodeUniform(nodObj, "uniConvertInt32toInt16_2x8",
                    1, uniConvertInt32toUint8_2x8);

                status |= vxSetNodeUniform(nodObj, "scaleU8Fp16", 1, &scaleIn);
                status |= vxSetNodeUniform(nodObj, "zpU8Fp16", 1, &input_ZP);

                status |= vxSetNodeUniform(nodObj, "uniMulMinusZpUint8_4x4", 1, uniMulMinusZpUint8_4x4);
                status |= vxSetNodeUniform(nodObj, "uniMulMinusZp2Uint8_4x4", 1, uniMulMinusZp2Uint8_4x4);
                status |= vxSetNodeUniform(nodObj, "uniMulMinusZp3Uint8_4x4", 1, uniMulMinusZp3Uint8_4x4);
                status |= vxSetNodeUniform(nodObj, "uniMulMinusZp4Uint8_4x4", 1, uniMulMinusZp4Uint8_4x4);
            }
        }
    }
    else if((inDataType == VSI_NN_TYPE_INT16 && outDataType == VSI_NN_TYPE_INT16
            && (axDataType == VSI_NN_TYPE_INT16 || axDataType == VSI_NN_TYPE_UINT8))
            || (inDataType == VSI_NN_TYPE_INT16 && outDataType == VSI_NN_TYPE_FLOAT16
                && axDataType == VSI_NN_TYPE_INT16))
    {
        shaderParam.globalWorkOffset[0] = 0;
        shaderParam.globalWorkOffset[1] = 0;
        shaderParam.globalWorkOffset[2] = 0;
        shaderParam.globalWorkScale[0]  = 4;
        shaderParam.globalWorkScale[1]  = 1;
        shaderParam.globalWorkScale[2]  = 1;
        shaderParam.globalWorkSize[0]   = gcmALIGN((attr[0].size[0] + shaderParam.globalWorkScale[0] - 1)
            / shaderParam.globalWorkScale[0], 4);
        shaderParam.globalWorkSize[1]   = (attr[0].size[1] + shaderParam.globalWorkScale[1] - 1)
            / shaderParam.globalWorkScale[1];
        shaderParam.globalWorkSize[2]   = (attr[0].size[2] + shaderParam.globalWorkScale[2] - 1)
            / shaderParam.globalWorkScale[2];
        {
            // uniforms
            uint32_t ucharMulShort_8x8[16] = {
                0x11111111, // TCfg
                0x00000000, // ASelt
                0x03020100, 0x07060504, // ABin
                0x11111111, // BSelt
                0x03020100, 0x07060504, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
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
            vx_uint32 uniQuantInOutInt16_2x8[16] = {
                0x11111111, // TCfg
                0x00000000, // ASelt
                0x03020100, 0x07060504, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000300, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001,
                0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
            };

            if(inDataType == VSI_NN_TYPE_INT16 && outDataType == VSI_NN_TYPE_INT16
                && axDataType == VSI_NN_TYPE_UINT8 && infixpoint != outfixpoint)
            {
                if(infixpoint > outfixpoint)
                {
                    uniQuantInOutInt16_2x8[7] = uniQuantInOutInt16_2x8[7] | (infixpoint - outfixpoint);
                }
                else
                {
                    vx_uint32 multiply       = (1 << (outfixpoint - infixpoint));
                    vx_uint32 i              = 0;

                    for (i = 8; i < 16; i++)
                    {
                        uniQuantInOutInt16_2x8[i] = multiply;
                    }
                }

                status = vxSetNodeUniform(nodObj, "uniQuantInOutInt16_2x8",
                    1, uniQuantInOutInt16_2x8);
                status |= vxSetNodeUniform(nodObj, "ucharMulShort_8x8_opt", 1, ucharMulShort_8x8);
            }
            else
            {
                status |= vxSetNodeUniform(nodObj, "uniConvertDirInt16Fp32_4x4",
                    1, uniConvertDirInt16Fp32_4x4);
                status |= vxSetNodeUniform(nodObj, "uniConvertEndInt16Fp32_4x4",
                    1, uniConvertEndInt16Fp32_4x4);
                status |= vxSetNodeUniform(nodObj, "uniConvertInt32toInt16_2x8",
                    1, uniConvertInt32toUint8_2x8);
                status |= vxSetNodeUniform(nodObj, "ucharMulShort_8x8", 1, ucharMulShort_8x8);
                status |= vxSetNodeUniform(nodObj, "scaleSF", 1, &scaleSF);
                status |= vxSetNodeUniform(nodObj, "inScaleInt16", 1, &factorIn);
            }
        }
    }
    else if(((inDataType == VSI_NN_TYPE_FLOAT16)
        && (outDataType == VSI_NN_TYPE_FLOAT16)
        && axDataType != VSI_NN_TYPE_INT16) )
    {
        shaderParam.globalWorkOffset[0] = 0;
        shaderParam.globalWorkOffset[1] = 0;
        shaderParam.globalWorkOffset[2] = 0;
        shaderParam.globalWorkScale[0]  = 4;
        shaderParam.globalWorkScale[1]  = 1;
        shaderParam.globalWorkScale[2]  = 1;
        shaderParam.globalWorkSize[0]   = gcmALIGN((attr[0].size[0] + shaderParam.globalWorkScale[0] - 1)
            / shaderParam.globalWorkScale[0], 4);
        shaderParam.globalWorkSize[1]   = (attr[0].size[1] + shaderParam.globalWorkScale[1] - 1)
            / shaderParam.globalWorkScale[1];
        shaderParam.globalWorkSize[2]   = (attr[0].size[2] + shaderParam.globalWorkScale[2] - 1)
            / shaderParam.globalWorkScale[2];
        {
            // uniforms
            uint32_t ucharMulShort_8x8[16] = {
                0x11111111, // TCfg
                0x00000000, // ASelt
                0x03020100, 0x07060504, // ABin
                0x11111111, // BSelt
                0x03020100, 0x07060504, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
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

            status |= vxSetNodeUniform(nodObj, "uniConvertDirInt16Fp32_4x4",
                1, uniConvertDirInt16Fp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvertEndInt16Fp32_4x4",
                1, uniConvertEndInt16Fp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvertInt32toInt16_2x8",
                1, uniConvertInt32toUint8_2x8);
            status |= vxSetNodeUniform(nodObj, "ucharMulShort_8x8", 1, ucharMulShort_8x8);
        }
    }
    else if(inDataType == VSI_NN_TYPE_FLOAT16
            && (outDataType == VSI_NN_TYPE_UINT8 || outDataType == VSI_NN_TYPE_INT8
              || outDataType == VSI_NN_TYPE_INT16))
    {
        shaderParam.globalWorkOffset[0] = 0;
        shaderParam.globalWorkOffset[1] = 0;
        shaderParam.globalWorkOffset[2] = 0;
        shaderParam.globalWorkScale[0]  = 4;
        shaderParam.globalWorkScale[1]  = 1;
        shaderParam.globalWorkScale[2]  = 1;
        shaderParam.globalWorkSize[0]   = gcmALIGN((attr[0].size[0] + shaderParam.globalWorkScale[0] - 1)
            / shaderParam.globalWorkScale[0], 4);
        shaderParam.globalWorkSize[1]   = (attr[0].size[1] + shaderParam.globalWorkScale[1] - 1)
            / shaderParam.globalWorkScale[1];
        shaderParam.globalWorkSize[2]   = (attr[0].size[2] + shaderParam.globalWorkScale[2] - 1)
            / shaderParam.globalWorkScale[2];
        {
            // uniforms
            uint32_t shortMulShort_8x8[16] = {
                0x11111111, // TCfg
                0x00000000, // ASelt
                0x03020100, 0x07060504, // ABin
                0x11111111, // BSelt
                0x03020100, 0x07060504, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            };
            vx_uint32 ucharMulShort_8x8[16] = {
                0x11111111, // TCfg
                0x00000000, // ASelt
                0x03020100, 0x07060504, // ABin
                0x11111111, // BSelt
                0x03020100, 0x07060504, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            };
            vx_uint32 uniConvertFstFp16Fp32_4x4[16] = {
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00010000, 0x00030002, // ABin
                0x02020202, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
                0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
            };
            vx_uint32 uniConvertSecFp16Fp32_4x4[16] = {
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
            vx_uint32 uniMulMinusZpUint8_4x4[16] = {
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00010000, 0x00030002, // ABin
                0x01010101, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            };
            vx_uint32 uniMulMinusZp2Uint8_4x4[16] = {
                0x01010101, // TCfg
                0x00000000, // ASelt
                0x00050004, 0x00070006, // ABin
                0x01010101, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            };

            status |= vxSetNodeUniform(nodObj, "uniConvertInt32toUint8_2x8",
                1, uniConvertInt32toUint8_2x8);
            status |= vxSetNodeUniform(nodObj, "uniConvertFstFp16Fp32_4x4",
                1, uniConvertFstFp16Fp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvertSecFp16Fp32_4x4",
                1, uniConvertSecFp16Fp32_4x4);
            status |= vxSetNodeUniform(nodObj, "shortMulShort_8x8", 1, shortMulShort_8x8);
            status |= vxSetNodeUniform(nodObj, "ucharMulShort_8x8_2", 1, ucharMulShort_8x8);
            if(outDataType == VSI_NN_TYPE_UINT8)
            {
                vx_float32 reOutScale_u8 = 1 / scaleOut;
                status |= vxSetNodeUniform(nodObj, "upOutput_Scale", 1, &scaleOut);
                status |= vxSetNodeUniform(nodObj, "reUpOutScale_u8", 1, &reOutScale_u8);
                status |= vxSetNodeUniform(nodObj, "upOutput_ZP", 1, &output_ZP);

                status |= vxSetNodeUniform(nodObj, "uniMulZpUint8_4x4_1",
                    1, uniMulMinusZpUint8_4x4);
                status |= vxSetNodeUniform(nodObj, "uniMulZpUint8_4x4_2",
                    1, uniMulMinusZp2Uint8_4x4);
            }
            else if(outDataType == VSI_NN_TYPE_INT8)
            {
                float scale_out_f = 1.0f / scaleOut;
                float output_ZP_f = (float)output_ZP;
                status |= vxSetNodeUniform(nodObj, "scaleOut", 1, &scale_out_f);
                status |= vxSetNodeUniform(nodObj, "outputZp", 1, &output_ZP_f);
            }
            else if(outDataType == VSI_NN_TYPE_INT16)
            {
                status |= vxSetNodeUniform(nodObj, "up_outFlScale_i16", 1, &factorOut);
            }
        }
    }
    else if(((inDataType == VSI_NN_TYPE_INT8)
        && (outDataType == VSI_NN_TYPE_FLOAT16)
        && axDataType != VSI_NN_TYPE_INT16) )
    {
        uint32_t uniConvertDirUint8Fp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };
        uint32_t uniConvertEndUint8Fp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };
        uint32_t uniConvertTrdUint8Fp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00090008, 0x000b000a, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };
        uint32_t uniConvertFthUint8Fp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x000d000c, 0x000f000e, // ABin
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
        float inputTail = (float)input_ZP * scaleIn * (-1.0f);

        shaderParam.globalWorkOffset[0] = 0;
        shaderParam.globalWorkOffset[1] = 0;
        shaderParam.globalWorkOffset[2] = 0;
        shaderParam.globalWorkScale[0]  = 8;
        shaderParam.globalWorkScale[1]  = 1;
        shaderParam.globalWorkScale[2]  = 1;
        shaderParam.globalWorkSize[0]   = gcmALIGN((attr[0].size[0] + shaderParam.globalWorkScale[0] - 1)
            / shaderParam.globalWorkScale[0], 4);
        shaderParam.globalWorkSize[1]   = (attr[0].size[1] + shaderParam.globalWorkScale[1] - 1)
            / shaderParam.globalWorkScale[1];
        shaderParam.globalWorkSize[2]   = attr[0].size[2];

        status |= vxSetNodeUniform(nodObj, "uniConvertDirUint8Fp32_4x4_2", 1, uniConvertDirUint8Fp32_4x4);
        status |= vxSetNodeUniform(nodObj, "uniConvertEndUint8Fp32_4x4_2", 1, uniConvertEndUint8Fp32_4x4);
        status |= vxSetNodeUniform(nodObj, "uniConvertTrdUint8Fp32_4x4_2", 1, uniConvertTrdUint8Fp32_4x4);
        status |= vxSetNodeUniform(nodObj, "uniConvertFthUint8Fp32_4x4_2", 1, uniConvertFthUint8Fp32_4x4);
        status |= vxSetNodeUniform(nodObj, "uniConvertInt32toUint8_2x8_2", 1, uniConvertInt32toUint8_2x8);
        status |= vxSetNodeUniform(nodObj, "scaleIn", 1, &scaleIn);
        status |= vxSetNodeUniform(nodObj, "inputTail", 1, &inputTail);
    }
    else if(((inDataType == VSI_NN_TYPE_INT16)
        && (outDataType == VSI_NN_TYPE_FLOAT16)
        && axDataType == VSI_NN_TYPE_UINT8) )
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
        vx_uint32 ucharMulShort_8x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x11111111, // BSelt
            0x03020100, 0x07060504, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
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

        shaderParam.globalWorkOffset[0] = 0;
        shaderParam.globalWorkOffset[1] = 0;
        shaderParam.globalWorkOffset[2] = 0;
        shaderParam.globalWorkScale[0]  = 4;
        shaderParam.globalWorkScale[1]  = 1;
        shaderParam.globalWorkScale[2]  = 1;
        shaderParam.globalWorkSize[0]   = gcmALIGN((attr[0].size[0] + shaderParam.globalWorkScale[0] - 1)
            / shaderParam.globalWorkScale[0], 4);
        shaderParam.globalWorkSize[1]   = (attr[0].size[1] + shaderParam.globalWorkScale[1] - 1)
            / shaderParam.globalWorkScale[1];
        shaderParam.globalWorkSize[2]   = attr[0].size[2];

        status |= vxSetNodeUniform(nodObj, "uniConvertFstInt16Fp32_4x4", 1, uniConvertDirInt16Fp32_4x4);
        status |= vxSetNodeUniform(nodObj, "uniConvertSecInt16Fp32_4x4", 1, uniConvertEndInt16Fp32_4x4);
        status |= vxSetNodeUniform(nodObj, "ucharMulShort_8x8_3", 1, ucharMulShort_8x8);
        status |= vxSetNodeUniform(nodObj, "uniConvertInt32toUint8_2x8_2", 1, uniConvertInt32toUint8_2x8);
        status |= vxSetNodeUniform(nodObj, "upInFl_i16", 1, &factorIn);
    }
    if(status != VX_SUCCESS)
    {
        VSILOGE("Set uniform failed(unpooling).\n");
        return status;
    }

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));
    if(status != VX_SUCCESS)
    {
        VSILOGE("Set node attribute failed(unpooling).\n");
        return status;
    }

    return VX_SUCCESS;
}

static vx_param_description_t vxunpoolingKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED}
};

#ifdef __cplusplus
extern "C" {
#endif
vx_kernel_description_t vxUnpoolingInfo =
{
    VX_KERNEL_ENUM_UNPOOLING,
    VX_KERNEL_NAME_UNPOOLING,
    NULL,
    vxunpoolingKernelParam,
    (sizeof(vxunpoolingKernelParam) / sizeof(vxunpoolingKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxunpoolingInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxUnpoolingInfoInt8 =
{
    VX_KERNEL_ENUM_UNPOOLING,
    VX_KERNEL_NAME_UNPOOLING_INT8,
    NULL,
    vxunpoolingKernelParam,
    (sizeof(vxunpoolingKernelParam) / sizeof(vxunpoolingKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxunpoolingInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxUnpoolingInfoUint8 =
{
    VX_KERNEL_ENUM_UNPOOLING,
    VX_KERNEL_NAME_UNPOOLING_UINT8,
    NULL,
    vxunpoolingKernelParam,
    (sizeof(vxunpoolingKernelParam) / sizeof(vxunpoolingKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxunpoolingInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxUnpoolingInfoInt16_int16 =
{
    VX_KERNEL_ENUM_UNPOOLING,
    VX_KERNEL_NAME_UNPOOLING_INT16_INT16,
    NULL,
    vxunpoolingKernelParam,
    (sizeof(vxunpoolingKernelParam) / sizeof(vxunpoolingKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxunpoolingInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxUnpoolingInfoInt16_int16_opt =
{
    VX_KERNEL_ENUM_UNPOOLING,
    VX_KERNEL_NAME_UNPOOLING_INT16_INT16_OPT,
    NULL,
    vxunpoolingKernelParam,
    (sizeof(vxunpoolingKernelParam) / sizeof(vxunpoolingKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxunpoolingInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxUnpoolingInfoInt16_int16_axInt16 =
{
    VX_KERNEL_ENUM_UNPOOLING,
    VX_KERNEL_NAME_UNPOOLING_INT16_INT16_AXINT16,
    NULL,
    vxunpoolingKernelParam,
    (sizeof(vxunpoolingKernelParam) / sizeof(vxunpoolingKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxunpoolingInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxUnpoolingInfoInt16_fp16_axI16 =
{
    VX_KERNEL_ENUM_UNPOOLING,
    VX_KERNEL_NAME_UNPOOLING_INT16_FP16_AXINT16,
    NULL,
    vxunpoolingKernelParam,
    (sizeof(vxunpoolingKernelParam) / sizeof(vxunpoolingKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxunpoolingInitializer,
    vsi_nn_KernelDeinitializer
};
vx_kernel_description_t vxUnpoolingInfoFp16_uint8 =
{
    VX_KERNEL_ENUM_UNPOOLING,
    VX_KERNEL_NAME_UNPOOLING_FP16_UINT8,
    NULL,
    vxunpoolingKernelParam,
    (sizeof(vxunpoolingKernelParam) / sizeof(vxunpoolingKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxunpoolingInitializer,
    vsi_nn_KernelDeinitializer
};
vx_kernel_description_t vxUnpoolingInfoUint8_fp16 =
{
    VX_KERNEL_ENUM_UNPOOLING,
    VX_KERNEL_NAME_UNPOOLING_UINT8_FP16,
    NULL,
    vxunpoolingKernelParam,
    (sizeof(vxunpoolingKernelParam) / sizeof(vxunpoolingKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxunpoolingInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxUnpoolingInfoFp16Fp16_uint8 =
{
    VX_KERNEL_ENUM_UNPOOLING,
    VX_KERNEL_NAME_UNPOOLING_FP16FP16_UINT8,
    NULL,
    vxunpoolingKernelParam,
    (sizeof(vxunpoolingKernelParam) / sizeof(vxunpoolingKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxunpoolingInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxUnpoolingInfoInt8_int8 =
{
    VX_KERNEL_ENUM_UNPOOLING,
    VX_KERNEL_NAME_UNPOOLING_INT8_INT8,
    NULL,
    vxunpoolingKernelParam,
    (sizeof(vxunpoolingKernelParam) / sizeof(vxunpoolingKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxunpoolingInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxUnpoolingInfoInt8_int8_opt =
{
    VX_KERNEL_ENUM_UNPOOLING,
    VX_KERNEL_NAME_UNPOOLING_INT8_INT8_OPT,
    NULL,
    vxunpoolingKernelParam,
    (sizeof(vxunpoolingKernelParam) / sizeof(vxunpoolingKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxunpoolingInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxUnpoolingInfoFp16_int8 =
{
    VX_KERNEL_ENUM_UNPOOLING,
    VX_KERNEL_NAME_UNPOOLING_FP16_INT8,
    NULL,
    vxunpoolingKernelParam,
    (sizeof(vxunpoolingKernelParam) / sizeof(vxunpoolingKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxunpoolingInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxUnpoolingInfoFp16_int16 =
{
    VX_KERNEL_ENUM_UNPOOLING,
    VX_KERNEL_NAME_UNPOOLING_FP16_INT16,
    NULL,
    vxunpoolingKernelParam,
    (sizeof(vxunpoolingKernelParam) / sizeof(vxunpoolingKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxunpoolingInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxUnpoolingInfoInt8_fp16 =
{
    VX_KERNEL_ENUM_UNPOOLING,
    VX_KERNEL_NAME_UNPOOLING_INT8_FP16,
    NULL,
    vxunpoolingKernelParam,
    (sizeof(vxunpoolingKernelParam) / sizeof(vxunpoolingKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxunpoolingInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxUnpoolingInfoInt16_fp16 =
{
    VX_KERNEL_ENUM_UNPOOLING,
    VX_KERNEL_NAME_UNPOOLING_INT16_FP16,
    NULL,
    vxunpoolingKernelParam,
    (sizeof(vxunpoolingKernelParam) / sizeof(vxunpoolingKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxunpoolingInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxUnpoolingInfoUint8_2D =
{
    VX_KERNEL_ENUM_UNPOOLING,
    VX_KERNEL_NAME_UNPOOLING_UINT8_2D,
    NULL,
    vxunpoolingKernelParam,
    (sizeof(vxunpoolingKernelParam) / sizeof(vxunpoolingKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxunpoolingInitializer,
    vsi_nn_KernelDeinitializer
};
vx_kernel_description_t vxUnpoolingInfoFp16_uint8_2D =
{
    VX_KERNEL_ENUM_UNPOOLING,
    VX_KERNEL_NAME_UNPOOLING_FP16_UINT8_2D,
    NULL,
    vxunpoolingKernelParam,
    (sizeof(vxunpoolingKernelParam) / sizeof(vxunpoolingKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxunpoolingInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_UPSAMPLE_list[] =
{
    NULL,
    &vxUnpoolingInfo,
    &vxUnpoolingInfoInt8,
    &vxUnpoolingInfoUint8,
    &vxUnpoolingInfoInt16_int16,
    &vxUnpoolingInfoInt16_fp16_axI16,
    &vxUnpoolingInfoFp16_uint8,
    &vxUnpoolingInfoUint8_fp16,
    &vxUnpoolingInfoFp16Fp16_uint8,
    &vxUnpoolingInfoInt8_int8,
    &vxUnpoolingInfoFp16_int8,
    &vxUnpoolingInfoFp16_int16,
    &vxUnpoolingInfoInt16_int16_axInt16,
    &vxUnpoolingInfoInt16_int16_opt,
    &vxUnpoolingInfoInt8_int8_opt,
    &vxUnpoolingInfoInt8_fp16,
    &vxUnpoolingInfoInt16_fp16,
    &vxUnpoolingInfoUint8_2D,
    &vxUnpoolingInfoFp16_uint8_2D,
    NULL
};
#ifdef __cplusplus
}
#endif
