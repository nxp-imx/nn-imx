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

vx_status VX_CALLBACK vxPre_process_rgbInitializer
    (
    vx_node node,
    const vx_reference *paramObj,
    vx_uint32 paraNum
    )
{
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
    vx_kernel_execution_parameters_t shaderParam = {
        2,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vx_status status            = VX_SUCCESS;
    vx_tensor output            = (vx_tensor)paramObj[1];
    vx_scalar xRatio_s          = (vx_scalar)paramObj[2];
    vx_scalar yRatio_s          = (vx_scalar)paramObj[3];
    vx_scalar reorder_s         = (vx_scalar)paramObj[10];
    vx_scalar trans_s           = (vx_scalar)paramObj[11];
    vx_int32   xRatio           = 0;
    vx_int32   yRatio           = 0;
    vx_int32   reverse_channel  = 0;
    vx_int32   enable_trans     = 0;
    vx_int8    dstFixedPointPos = 0;
    vsi_nn_type_e dstFormat;
    vx_float32 outputScale      = 1.0;
    vx_int32   output_ZP        = 0;
    vx_float32 outputZP         = 0;
    vx_int32   r_order          = 2;
    vx_int32   b_order          = 0;
    vx_uint32  width            = 0;
    vx_uint32  height           = 0;
    vx_bool    enable_copy      = vx_false_e;
    vsi_nn_tensor_attr_t attr;

    vxCopyScalar(xRatio_s, (void*)&xRatio, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar(yRatio_s, (void*)&yRatio, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar(reorder_s, (void*)&reverse_channel, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar(trans_s, (void*)&enable_trans, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    if (reverse_channel)
    {
        r_order          = 2;
        b_order          = 0;
    }

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));

    status |= vsi_nn_vxGetTensorAttr(output, &attr);

    width = attr.size[0];
    height = attr.size[1];
    dstFormat = attr.dtype.vx_type;
    dstFixedPointPos = attr.dtype.fl;
    output_ZP = attr.dtype.zero_point;
    outputScale = attr.dtype.scale;

    if(status < 0)
        printf("error-%s,%d\n",__FILE__,__LINE__);

    enable_copy = (vx_bool)(xRatio == (1 << 15) && yRatio == (1 << 15));

    if (enable_trans && enable_copy)
    {
        vx_uint32 uniNormilizationLo_2x8[16] = {
            0x99999999, // TCfg
            0x44444444, // ASelt
            0x45002142, 0x27480324, // ABin
            0x99999999, // BSelt
            0x06060606, 0x06060606, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000,
            0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000 // Constant
        };
        vx_uint32 uniNormilizationHi_2x8[16] = {
            0x09999999, // TCfg
            0x04444444, // ASelt
            0x092a4b06, 0x000c2d4e, // ABin
            0x09999999, // BSelt
            0x06060606, 0x00060606, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000,
            0x3c000000, 0x3c000000, 0x3c000000, 0x00000000 // Constant
        };
        vx_uint32 uniNormilizationLo_NHWC_2x8[16] = {
            0x99999999, // TCfg
            0x44444444, // ASelt
            0x03422100, 0x27064524, // ABin
            0x99999999, // BSelt
            0x06060606, 0x06060606, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000,
            0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000 // Constant
        };
        vx_uint32 uniNormilizationHi_NHWC_2x8[16] = {
            0x09999999, // TCfg
            0x04444444, // ASelt
            0x4b2a0948, 0x004e2d0c, // ABin
            0x09999999, // BSelt
            0x06060606, 0x00060606, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000,
            0x3c000000, 0x3c000000, 0x3c000000, 0x00000000 // Constant
        };

        shaderParam.globalWorkOffset[0] = 0;
        shaderParam.globalWorkOffset[1] = 0;
        shaderParam.globalWorkScale[0]  = 15;
        shaderParam.globalWorkScale[1]  = 1;
        shaderParam.globalWorkSize[0]   = gcmALIGN((width + shaderParam.globalWorkScale[0] - 1)
            / shaderParam.globalWorkScale[0], 4);
        shaderParam.globalWorkSize[1]   = height;

        if (dstFormat == VSI_NN_TYPE_INT8 || dstFormat == VSI_NN_TYPE_INT16)
        {
            if(dstFixedPointPos > 0)
                outputScale = (vx_float32) (1 << dstFixedPointPos);
            else
            {
                outputScale = 1.0f;
                uniNormilizationLo_2x8[7] |= ((-dstFixedPointPos) & 0x1F);
                uniNormilizationHi_2x8[7] |= ((-dstFixedPointPos) & 0x1F);
                uniNormilizationLo_NHWC_2x8[7] |= ((-dstFixedPointPos) & 0x1F);
                uniNormilizationHi_NHWC_2x8[7] |= ((-dstFixedPointPos) & 0x1F);
            }

            status |= vxSetNodeUniform(node, "outputZP", 1, &outputZP);
        }
        else if (dstFormat == VSI_NN_TYPE_UINT8)
        {
            outputZP = (vx_float32)output_ZP;

            outputScale = 1.0f / outputScale;

            status |= vxSetNodeUniform(node, "outputZP", 1, &outputZP);
        }
        else
        {
            outputScale = 1.0f;
        }

        if (reverse_channel)
        {
            status |= vxSetNodeUniform(node, "uniNormilizationLo_2x8", 1, uniNormilizationLo_2x8);
            status |= vxSetNodeUniform(node, "uniNormilizationHi_2x8", 1, uniNormilizationHi_2x8);
        }
        else
        {
            status |= vxSetNodeUniform(node, "uniNormilizationLo_2x8", 1, uniNormilizationLo_NHWC_2x8);
            status |= vxSetNodeUniform(node, "uniNormilizationHi_2x8", 1, uniNormilizationHi_NHWC_2x8);
        }
        status |= vxSetNodeUniform(node, "outputScale", 1, &outputScale);
    }
    else if (enable_copy)
    {
        vx_uint32 uniExtractR_2x8[16] = {
            0x00099999, // TCfg
            0x00044444, // ASelt
            0x09060300, 0x0000000c, // ABin
            0x00099999, // BSelt
            0x06060606, 0x00000006, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000,
            0x3c000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniExtractG_2x8[16] = {
            0x00099999, // TCfg
            0x00044444, // ASelt
            0x2a272421, 0x0000002d, // ABin
            0x00099999, // BSelt
            0x06060606, 0x00000006, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000,
            0x3c000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniExtractB_2x8[16] = {
            0x00099999, // TCfg
            0x00044444, // ASelt
            0x4b484542, 0x0000004e, // ABin
            0x00099999, // BSelt
            0x06060606, 0x00000006, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000,
            0x3c000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };

        shaderParam.globalWorkOffset[0] = 0;
        shaderParam.globalWorkOffset[1] = 0;
        if (dstFormat == VSI_NN_TYPE_FLOAT16 || dstFormat == VSI_NN_TYPE_INT16)
            shaderParam.globalWorkScale[0]  = 8;
        else if (dstFormat == VSI_NN_TYPE_INT8 || dstFormat == VSI_NN_TYPE_UINT8)
            shaderParam.globalWorkScale[0]  = 10;

        shaderParam.globalWorkScale[1]  = 1;
        shaderParam.globalWorkSize[0]   = gcmALIGN((width + shaderParam.globalWorkScale[0] - 1)
            / shaderParam.globalWorkScale[0], 4);
        shaderParam.globalWorkSize[1]   = height;

        if (dstFormat == VSI_NN_TYPE_INT8 || dstFormat == VSI_NN_TYPE_INT16)
        {
            if(dstFixedPointPos > 0)
                outputScale = (vx_float32) (1 << dstFixedPointPos);
            else
            {
                outputScale = 1.0f;
                uniExtractR_2x8[7] |= ((-dstFixedPointPos) & 0x1F);
                uniExtractG_2x8[7] |= ((-dstFixedPointPos) & 0x1F);
                uniExtractB_2x8[7] |= ((-dstFixedPointPos) & 0x1F);
            }

            status |= vxSetNodeUniform(node, "outputZP", 1, &outputZP);
        }
        else if (dstFormat == VSI_NN_TYPE_UINT8)
        {
            outputZP = (vx_float32)output_ZP;

            outputScale = 1.0f / outputScale;

            status |= vxSetNodeUniform(node, "outputZP", 1, &outputZP);
        }
        else
        {
            outputScale = 1.0f;
        }

        status |= vxSetNodeUniform(node, "uniExtractR_2x8", 1, uniExtractR_2x8);
        status |= vxSetNodeUniform(node, "uniExtractG_2x8", 1, uniExtractG_2x8);
        status |= vxSetNodeUniform(node, "uniExtractB_2x8", 1, uniExtractB_2x8);
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
            0x00000400, 0x00000000, 0x00000400, 0x00000000,
            0x00000400, 0x00000000, 0x00000400, 0x00000000 // Constant
        };
        vx_uint32 uniAddRShift[16] = {
            0x0f0f0f0f, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002405, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniGetTempVal[16] = {
            0x09090909, // TCfg
            0x00000000, // ASelt
            0x00230001, 0x00670045, // ABin
            0x05050505, // BSelt
            0x00110000, 0x00330022, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniExtractBytes[16] = {
            0x0f0f0f0f, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002414, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniUnpackToR[16] = {
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x09060300, 0x09060300, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00007400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniUnpackToG[16] = {
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x0a070401, 0x0a070401, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00007400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniUnpackToB[16] = {
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x0b080502, 0x0b080502, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00007400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniConvertIntergetoF32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        };
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
        vx_uint32 uniRePackRGBLo_2x8[16] = {
            0x00111111, // TCfg
            0x00001001, // ASelt
            0x01000400, 0x00000105, // ABin
            0x00222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniRePackRGBHi_2x8[16] = {
            0x00111111, // TCfg
            0x00001001, // ASelt
            0x03020602, 0x00000307, // ABin
            0x00222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniRePackRGBLo_NHWC_2x8[16] = {
            0x00111111, // TCfg
            0x00100100, // ASelt
            0x01000400, 0x00000105, // ABin
            0x00222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniRePackRGBHi_NHWC_2x8[16] = {
            0x00111111, // TCfg
            0x00100100, // ASelt
            0x03020602, 0x00000307, // ABin
            0x00222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000000, 0x00000000 // Constant
        };

        shaderParam.globalWorkScale[0]  = 4;
        shaderParam.globalWorkScale[1]  = 1;
        if (enable_trans)
            shaderParam.globalWorkSize[0]   = gcmALIGN((width / 3 + shaderParam.globalWorkScale[0] - 1)
            / shaderParam.globalWorkScale[0], 4);
        else
            shaderParam.globalWorkSize[0]   = gcmALIGN((width + shaderParam.globalWorkScale[0] - 1)
            / shaderParam.globalWorkScale[0], 4);
        shaderParam.globalWorkSize[1]   = height;

        status |= vxSetNodeUniform(node, "uniUnpackToR", 1, uniUnpackToR);
        status |= vxSetNodeUniform(node, "uniUnpackToG", 1, uniUnpackToG);
        status |= vxSetNodeUniform(node, "uniUnpackToB", 1, uniUnpackToB);
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

            output_ZP = 0;
        }
        else if (dstFormat == VSI_NN_TYPE_UINT8)
        {
            outputScale = 1.0f / outputScale;

            outputZP = (vx_float32)output_ZP;
        }
        else if (dstFormat == VSI_NN_TYPE_FLOAT16)
        {
            outputScale = 1.0f;

            output_ZP = 0;
        }

        if (dstFormat == VSI_NN_TYPE_FLOAT16)
            status |= vxSetNodeUniform(node, "uniExtract8Data_2x8", 1, uniExtractHalf8_2x8);
        else
            status |= vxSetNodeUniform(node, "uniExtract8Data_2x8", 1, uniExtractInteger_2x8);

        status |= vxSetNodeUniform(node, "uniConvertIntergetoF32_4x4", 1, uniConvertIntergetoF32_4x4);
        status |= vxSetNodeUniform(node, "outputZP", 1, &outputZP);
        status |= vxSetNodeUniform(node, "outputScale", 1, &outputScale);

        if (enable_trans)
        {
            if (reverse_channel)
            {
                status |= vxSetNodeUniform(node, "uniRePackRGBLo_2x8", 1, uniRePackRGBLo_2x8);
                status |= vxSetNodeUniform(node, "uniRePackRGBHi_2x8", 1, uniRePackRGBHi_2x8);
            }
            else
            {
                status |= vxSetNodeUniform(node, "uniRePackRGBLo_2x8", 1, uniRePackRGBLo_NHWC_2x8);
                status |= vxSetNodeUniform(node, "uniRePackRGBHi_2x8", 1, uniRePackRGBHi_NHWC_2x8);
            }
        }
    }

    if (!enable_trans)
    {
        status |= vxSetNodeUniform(node, "r_order", 1, &r_order);
        status |= vxSetNodeUniform(node, "b_order", 1, &b_order);
    }

    status |= vxSetNodeAttribute(node, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));

    return status;
}


#ifdef __cplusplus
extern "C" {
#endif

static vx_param_description_t vxPre_process_rgbKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};


#define PRE_RPOCESS_RGB_KERNELS(DST_TYPE, COPY) \
vx_kernel_description_t vxPre_Process_RGB_##DST_TYPE##COPY##_Kernel =  \
{ \
    VX_KERNEL_ENUM_PRE_PROCESS_RGB, \
    VX_KERNEL_NAME_PRE_PROCESS_RGB_##DST_TYPE##COPY, \
    NULL, \
    vxPre_process_rgbKernelParam, \
    (sizeof(vxPre_process_rgbKernelParam) / sizeof(vxPre_process_rgbKernelParam[0])), \
    vsi_nn_KernelValidator, \
    NULL, \
    NULL, \
    vxPre_process_rgbInitializer, \
    vsi_nn_KernelDeinitializer \
};

PRE_RPOCESS_RGB_KERNELS(F16, )
PRE_RPOCESS_RGB_KERNELS(I16, )
PRE_RPOCESS_RGB_KERNELS(I8, )
PRE_RPOCESS_RGB_KERNELS(U8, )
PRE_RPOCESS_RGB_KERNELS(F16, _COPY)
PRE_RPOCESS_RGB_KERNELS(I16,  _COPY)
PRE_RPOCESS_RGB_KERNELS(I8,   _COPY)
PRE_RPOCESS_RGB_KERNELS(U8,   _COPY)

PRE_RPOCESS_RGB_KERNELS(F16, _NHWC)
PRE_RPOCESS_RGB_KERNELS(I16, _NHWC)
PRE_RPOCESS_RGB_KERNELS(I8, _NHWC)
PRE_RPOCESS_RGB_KERNELS(U8, _NHWC)
PRE_RPOCESS_RGB_KERNELS(F16, _COPY_NHWC)
PRE_RPOCESS_RGB_KERNELS(I16,  _COPY_NHWC)
PRE_RPOCESS_RGB_KERNELS(I8,   _COPY_NHWC)
PRE_RPOCESS_RGB_KERNELS(U8,   _COPY_NHWC)

#define PRE_RPOCESS_RGB_KERNELS_NAME(DST_TYPE, COPY) \
    &vxPre_Process_RGB_##DST_TYPE##COPY##_Kernel,

vx_kernel_description_t * vx_kernel_PRE_PROCESS_RGB_list[] =
{
    NULL,
    PRE_RPOCESS_RGB_KERNELS_NAME(F16, )
    PRE_RPOCESS_RGB_KERNELS_NAME(I16, )
    PRE_RPOCESS_RGB_KERNELS_NAME(I8, )
    PRE_RPOCESS_RGB_KERNELS_NAME(U8, )
    PRE_RPOCESS_RGB_KERNELS_NAME(F16, _COPY)
    PRE_RPOCESS_RGB_KERNELS_NAME(I16,  _COPY)
    PRE_RPOCESS_RGB_KERNELS_NAME(I8,   _COPY)
    PRE_RPOCESS_RGB_KERNELS_NAME(U8,   _COPY)

    PRE_RPOCESS_RGB_KERNELS_NAME(F16, _NHWC)
    PRE_RPOCESS_RGB_KERNELS_NAME(I16, _NHWC)
    PRE_RPOCESS_RGB_KERNELS_NAME(I8, _NHWC)
    PRE_RPOCESS_RGB_KERNELS_NAME(U8, _NHWC)
    PRE_RPOCESS_RGB_KERNELS_NAME(F16, _COPY_NHWC)
    PRE_RPOCESS_RGB_KERNELS_NAME(I16,  _COPY_NHWC)
    PRE_RPOCESS_RGB_KERNELS_NAME(I8,   _COPY_NHWC)
    PRE_RPOCESS_RGB_KERNELS_NAME(U8,   _COPY_NHWC)
    NULL
};
#ifdef __cplusplus
}
#endif
