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
#include <math.h>

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

#define _VX_KERNEL_FUNC_KERNEL  (vxClipKernel)

#define clipMIN(x, y)            (((x) <= (y)) ?  (x) :  (y))
#define clipMAX(x, y)            (((x) >= (y)) ?  (x) :  (y))
#define gcmCLIP(x, minVal, maxVal)  (clipMIN(clipMAX((x), (minVal)), (maxVal)))

#define MAX_MULTIPLIER_NUM      (65535)
#define MAX_POST_SHIFT_BITS     (31)

static void clipGetFP32M0AndN(vx_float32 mult, vx_uint16 *M0, vx_int8 *N)
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

static vsi_status VX_CALLBACK vxClipKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
#define ARG_NUM            (2)
#define TENSOR_NUM_INPUT (1)
#define TENSOR_NUM_OUTPUT (1)
#define TENSOR_NUM (TENSOR_NUM_INPUT+TENSOR_NUM_OUTPUT)

    vsi_status status = VSI_FAILURE;
    vx_context context = NULL;
    vx_tensor input[TENSOR_NUM_INPUT] = {0};
    vx_tensor output[TENSOR_NUM_OUTPUT] = {0};
    float *f32_in_buffer[TENSOR_NUM_INPUT] = {0};
    float *f32_out_buffer[TENSOR_NUM_OUTPUT] = {0};
    vsi_nn_tensor_attr_t in_attr[TENSOR_NUM_INPUT];
    vsi_nn_tensor_attr_t out_attr[TENSOR_NUM_OUTPUT];
    uint32_t in_elements[TENSOR_NUM_INPUT] = {0};
    uint32_t out_elements[TENSOR_NUM_OUTPUT]= {0};

    float minf, maxf;

    int32_t i;

    /* prepare data */
    context = vxGetContext((vx_reference)node);

    for(i = 0; i < TENSOR_NUM_INPUT; i ++)
    {
        input[i] = (vx_tensor)paramObj[i];
        status = vsi_nn_vxGetTensorAttr(input[i], &in_attr[i]);
        TEST_CHECK_STATUS(status, final);
        in_elements[i] = vsi_nn_vxGetTensorElementNum(&in_attr[i]);
        f32_in_buffer[i] = (float *)malloc(in_elements[i] * sizeof(float));
        status = vsi_nn_vxConvertTensorToFloat32Data(
            context, input[i], &in_attr[i], f32_in_buffer[i],
            in_elements[i] * sizeof(float));
        TEST_CHECK_STATUS(status, final);
    }
    for(i = 0; i < TENSOR_NUM_OUTPUT; i ++)
    {
        output[i] = (vx_tensor)paramObj[i + TENSOR_NUM_INPUT];
        status = vsi_nn_vxGetTensorAttr(output[i], &out_attr[i]);
        TEST_CHECK_STATUS(status, final);
        out_elements[i] = vsi_nn_vxGetTensorElementNum(&out_attr[i]);
        f32_out_buffer[i]= (float *)malloc(out_elements[i] * sizeof(float));
        memset(f32_out_buffer[i], 0, out_elements[i] * sizeof(float));
    }
    vxCopyScalar((vx_scalar)paramObj[ARG_NUM], &(minf),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[ARG_NUM + 1], &(maxf),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    /* TODO: Add CPU kernel implement */
    /* example code : copy data form input tensor to output tensor*/

    {
        uint32_t n, c, h, w;
        uint32_t batch = in_attr[0].size[3];
        uint32_t channel = in_attr[0].size[2];
        uint32_t height = in_attr[0].size[1];
        uint32_t width = in_attr[0].size[0];
        for(n = 0; n < batch; ++n)
        {
            for(c = 0; c < channel; ++c)
            {
                for(h = 0; h < height; ++h)
                {
                    for(w = 0; w < width; ++w)
                    {
                        uint32_t index = w + h * width + c * width * height
                            + n * width * height * channel;
                        f32_out_buffer[0][index] = gcmCLIP(f32_in_buffer[0][index], minf, maxf);
                    }
                }
            }
        }
    }


    /* save data */
    for(i = 0; i < TENSOR_NUM_OUTPUT; i++)
    {
        status = vsi_nn_vxConvertFloat32DataToTensor(
            context, output[i], &out_attr[i], f32_out_buffer[i],
            out_elements[i] * sizeof(float));
        TEST_CHECK_STATUS(status, final);
    }

final:
    for (i = 0; i < TENSOR_NUM_INPUT; i++)
    {
        if (f32_in_buffer[i]) free(f32_in_buffer[i]);
    }
    for(i = 0; i < TENSOR_NUM_OUTPUT; i++)
    {
        if (f32_out_buffer[i]) free(f32_out_buffer[i]);
    }
    return status;
} /* _VX_KERNEL_FUNC_KERNEL() */

vx_status VX_CALLBACK vxClipInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    vx_uint32 paraNum
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

    vx_tensor     input           = (vx_tensor)paramObj[0];
    vx_tensor     output          = (vx_tensor)paramObj[1];
    vx_scalar     minScl          = (vx_scalar)paramObj[2];
    vx_scalar     maxScl          = (vx_scalar)paramObj[3];
    uint32_t      input_size[4]   = {1, 1, 1, 1};
    vx_uint32     output_size[4] = {1, 1, 1, 1};
    uint32_t      input_dims      = 0;
    uint32_t      output_dims     = 0;
    vsi_enum inputDataFormat = VSI_NN_TYPE_FLOAT16;
    vsi_enum outputDataFormat = VSI_NN_TYPE_FLOAT16;
    vx_float32 scaleIn = 0;
    vx_float32 scaleOut = 0;
    int32_t output_ZP = 0;
    int32_t input_ZP = 0;
    int8_t srcFixPointPos  = 0;
    int8_t dstFixPointPos   = 0;
    vx_float32 minVal, maxVal;
    vsi_nn_tensor_attr_t attr[2];
    vx_uint32 i = 0;

    memset(&attr[0], 0, sizeof(vsi_nn_tensor_attr_t));
    memset(&attr[1], 0, sizeof(vsi_nn_tensor_attr_t));
    status  = vsi_nn_vxGetTensorAttr(input, &attr[0]);
    status |= vsi_nn_vxGetTensorAttr(output, &attr[1]);
    if (status != VX_SUCCESS)
    {
        VSILOGE("vsi_nn_vxGetTensorAttr  failure! at line %d\n", __LINE__);
        return status;
    }

    input_dims  = attr[0].dim_num;
    for (i = 0; i < input_dims; i++)
    {
        input_size[i] = attr[0].size[i];
    }
    inputDataFormat  = attr[0].dtype.vx_type;
    input_ZP         = attr[0].dtype.zero_point;
    scaleIn          = attr[0].dtype.scale;
    srcFixPointPos   = attr[0].dtype.fl;
    dstFixPointPos   = attr[1].dtype.fl;
    output_ZP        = attr[1].dtype.zero_point;
    scaleOut         = attr[1].dtype.scale;
    outputDataFormat = attr[1].dtype.vx_type;
    output_dims      = attr[1].dim_num;
    for (i = 0; i < output_dims; i++)
    {
        output_size[i] = attr[1].size[i];
    }

    status |= vxCopyScalar(minScl, &minVal, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    status |= vxCopyScalar(maxScl, &maxVal, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    if(VX_SUCCESS != status)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
        return status;
    }

    input_size[2] = (input_dims <= 2)?1:input_size[2];

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkOffset[2] = 0;
    if (outputDataFormat == VSI_NN_TYPE_FLOAT16 || outputDataFormat == VSI_NN_TYPE_INT16)
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

    if (inputDataFormat == VSI_NN_TYPE_FLOAT16 && outputDataFormat == VSI_NN_TYPE_FLOAT16)
    {
        vx_uint16 minTmp   = 0;
        vx_uint16 maxTmp   = 0;
        vx_int32 packedMin = 0;
        vx_int32 packedMax = 0;
        vx_int32 packedMinData_FP16[4];
        vx_int32 packedMaxData_FP16[4];
        vx_int32 i;

        minTmp = vsi_nn_Fp32toFp16(minVal);
        maxTmp = vsi_nn_Fp32toFp16(maxVal);
        packedMin = (minTmp << 16) | (minTmp);
        packedMax = (maxTmp << 16) | (maxTmp);

        for (i = 0;i < 4; i++)
        {
            packedMinData_FP16[i] = packedMin;
            packedMaxData_FP16[i] = packedMax;
        }

        status |= vxSetNodeUniform(nodObj, "packedMinData_FP16", 1, packedMinData_FP16);
        status |= vxSetNodeUniform(nodObj, "packedMaxData_FP16", 1, packedMaxData_FP16);
    }
    else if (inputDataFormat == VSI_NN_TYPE_INT8 && outputDataFormat == VSI_NN_TYPE_INT8)
    {
        vx_uint8 minData   = 0;
        vx_uint8 maxData   = 0;
        vx_int32 packedMin = (minData << 24) | (minData << 16) | (minData << 8) | (minData);
        vx_int32 packedMax = (maxData << 24) | (maxData << 16) | (maxData << 8) | (maxData);
        vx_int32 packedMinData[4];
        vx_int32 packedMaxData[4];
        vx_uint32 uniConvertIntegerLo_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };
        vx_uint32 uniConvertIntegerHi_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x0b0a0908, 0x0f0e0d0c, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };

        if (srcFixPointPos > dstFixPointPos)
        {
            vx_uint8  postshift      = vsi_nn_min(srcFixPointPos - dstFixPointPos, MAX_POST_SHIFT_BITS);

            uniConvertIntegerLo_2x8[7] |= (postshift & 0x1F);
            uniConvertIntegerHi_2x8[7] |= (postshift & 0x1F);
        }
        else
        {
            vx_uint32 multiplier = vsi_nn_min(1 << (dstFixPointPos - srcFixPointPos), MAX_MULTIPLIER_NUM);
            vx_uint32 i          = 0;

            for (i = 0; i < 8; i++)
            {
                uniConvertIntegerLo_2x8[i + 8] = multiplier;
                uniConvertIntegerHi_2x8[i + 8] = multiplier;
            }
        }

        minData =  vsi_nn_Fp32ToDFP(minVal, dstFixPointPos, VSI_NN_TYPE_INT8);
        maxData =  vsi_nn_Fp32ToDFP(maxVal, dstFixPointPos, VSI_NN_TYPE_INT8);

        packedMin = (minData << 24) | (minData << 16) | (minData << 8) | (minData);
        packedMax = (maxData << 24) | (maxData << 16) | (maxData << 8) | (maxData);

        packedMinData[0] = packedMinData[1] = packedMinData[2] = packedMinData[3] = packedMin;
        packedMaxData[0] = packedMaxData[1] = packedMaxData[2] = packedMaxData[3] = packedMax;

        status |= vxSetNodeUniform(nodObj, "uniConvertIntegerLo_2x8", 1, uniConvertIntegerLo_2x8);
        status |= vxSetNodeUniform(nodObj, "uniConvertIntegerHi_2x8", 1, uniConvertIntegerHi_2x8);

        status |= vxSetNodeUniform(nodObj, "packedMinData", 1, packedMinData);
        status |= vxSetNodeUniform(nodObj, "packedMaxData", 1, packedMaxData);
    }
    else if (inputDataFormat == VSI_NN_TYPE_INT16 && outputDataFormat == VSI_NN_TYPE_INT16)
    {
        vx_uint16 minData  = 0;
        vx_uint16 maxData  = 0;
        vx_int32 packedMin = (minData << 16) | (minData);
        vx_int32 packedMax = (maxData << 16) | (maxData);
        vx_int32 packedMinData[4];
        vx_int32 packedMaxData[4];
        vx_uint32 uniConvertIntegerLo_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };
        vx_uint32 uniConvertIntegerHi_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x0b0a0908, 0x0f0e0d0c, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };

        minData =  vsi_nn_Fp32ToDFP(minVal, dstFixPointPos, VSI_NN_TYPE_INT16);
        maxData =  vsi_nn_Fp32ToDFP(maxVal, dstFixPointPos, VSI_NN_TYPE_INT16);

        packedMin = (minData << 16) | (minData);
        packedMax = (maxData << 16) | (maxData);

        packedMinData[0] = packedMinData[1] = packedMinData[2] = packedMinData[3] = packedMin;
        packedMaxData[0] = packedMaxData[1] = packedMaxData[2] = packedMaxData[3] = packedMax;

        if (srcFixPointPos > dstFixPointPos)
        {
            vx_uint8  postshift      = vsi_nn_min(srcFixPointPos - dstFixPointPos, MAX_POST_SHIFT_BITS);

            uniConvertIntegerLo_2x8[7] |= (postshift & 0x1F);
            uniConvertIntegerHi_2x8[7] |= (postshift & 0x1F);
        }
        else
        {
            vx_uint32 multiplier = vsi_nn_min(1 << (dstFixPointPos - srcFixPointPos), MAX_MULTIPLIER_NUM);
            vx_uint32 i          = 0;

            for (i = 0; i < 8; i++)
            {
                uniConvertIntegerLo_2x8[i + 8] = multiplier;
                uniConvertIntegerHi_2x8[i + 8] = multiplier;
            }
        }

        status |= vxSetNodeUniform(nodObj, "uniConvertIntegerLo_2x8", 1, uniConvertIntegerLo_2x8);
        status |= vxSetNodeUniform(nodObj, "uniConvertIntegerHi_2x8", 1, uniConvertIntegerHi_2x8);
        status |= vxSetNodeUniform(nodObj, "packedMinData", 1, packedMinData);
        status |= vxSetNodeUniform(nodObj, "packedMaxData", 1, packedMaxData);
    }
    else if (inputDataFormat == VSI_NN_TYPE_UINT8 && outputDataFormat == VSI_NN_TYPE_UINT8)
    {
        vx_uint8   minData          = 0;
        vx_uint8   maxData          = 0;
        vx_int32   packedMin        = 0;
        vx_int32   packedMax        = 0;
        vx_int32   packedMinData[4];
        vx_int32   packedMaxData[4];
        vx_float32 uint8Scale = scaleIn / scaleOut;
        vx_uint16  M0                   = 0;
        vx_int8    postShift            = 0;
        vx_uint32    multAndoutZP[2]    = {0};
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

        clipGetFP32M0AndN(uint8Scale, &M0, &postShift);
        multAndoutZP[0] = (vx_uint32)(M0);
        multAndoutZP[1] = (vx_uint32)((output_ZP << postShift) - input_ZP * M0);

        uniU8MulAndPostShift_Lo_2x8[7] |= (postShift & 0x1F);
        uniU8MulAndPostShift_Hi_2x8[7] |= (postShift & 0x1F);

        minData = vsi_nn_Fp32ToAffine(minVal, scaleOut, output_ZP, VSI_NN_TYPE_UINT8);
        maxData = vsi_nn_Fp32ToAffine(maxVal, scaleOut, output_ZP, VSI_NN_TYPE_UINT8);

        //calculateActivationRangeUInt8(funcType, scaleOut, output_ZP, &minData, &maxData, minVal, maxVal);
        packedMin = (minData << 24) | (minData << 16) | (minData << 8) | (minData);
        packedMax = (maxData << 24) | (maxData << 16) | (maxData << 8) | (maxData);

        packedMinData[0] = packedMinData[1] = packedMinData[2] = packedMinData[3] = packedMin;
        packedMaxData[0] = packedMaxData[1] = packedMaxData[2] = packedMaxData[3] = packedMax;

        status |= vxSetNodeUniform(nodObj, "uniU8MulAndPostShift_Lo_2x8", 1, uniU8MulAndPostShift_Lo_2x8);
        status |= vxSetNodeUniform(nodObj, "uniU8MulAndPostShift_Hi_2x8", 1, uniU8MulAndPostShift_Hi_2x8);
        status |= vxSetNodeUniform(nodObj, "multAndoutZP", 1, multAndoutZP);

        status |= vxSetNodeUniform(nodObj, "packedMinData", 1, packedMinData);
        status |= vxSetNodeUniform(nodObj, "packedMaxData", 1, packedMaxData);
    }

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));
    if(status < 0)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
    }

    return status;
}

static vx_param_description_t vxClipKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

#ifdef __cplusplus
extern "C" {
#endif
vx_kernel_description_t vxClip_CPU =
{
    VX_KERNEL_ENUM_CLIP,
    VX_KERNEL_NAME_CLIP_FP16,
    _VX_KERNEL_FUNC_KERNEL,
    vxClipKernelParam,
    _cnt_of_array( vxClipKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxClipKernelInfo_fp16 =
{
    VX_KERNEL_ENUM_CLIP,
    VX_KERNEL_NAME_CLIP_FP16,
    NULL,
    vxClipKernelParam,
    (sizeof(vxClipKernelParam) / sizeof(vxClipKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxClipInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxClipKernelInfo_int16 =
{
    VX_KERNEL_ENUM_CLIP,
    VX_KERNEL_NAME_CLIP_INT16,
    NULL,
    vxClipKernelParam,
    (sizeof(vxClipKernelParam) / sizeof(vxClipKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxClipInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxClipKernelInfo_int8 =
{
    VX_KERNEL_ENUM_CLIP,
    VX_KERNEL_NAME_CLIP_INT8,
    NULL,
    vxClipKernelParam,
    (sizeof(vxClipKernelParam) / sizeof(vxClipKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxClipInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxClipKernelInfo_uint8 =
{
    VX_KERNEL_ENUM_CLIP,
    VX_KERNEL_NAME_CLIP_UINT8,
    NULL,
    vxClipKernelParam,
    (sizeof(vxClipKernelParam) / sizeof(vxClipKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxClipInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_CLIP_list[] =
{
    &vxClip_CPU,
    &vxClipKernelInfo_fp16,
    &vxClipKernelInfo_int16,
    &vxClipKernelInfo_int8,
    &vxClipKernelInfo_uint8,
    NULL
};
#ifdef __cplusplus
}
#endif
