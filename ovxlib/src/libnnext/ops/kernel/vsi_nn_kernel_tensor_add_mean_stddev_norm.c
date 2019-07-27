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
#include <math.h>

#include "vsi_nn_platform.h"

#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_math.h"
#include "client/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"

#define _VX_KERNEL_VAR          (vx_kernel_TENSOR_ADD_MEAN_STDDEV_NORM)
#define _VX_KERNEL_FUNC_KERNEL  (vxTensor_add_mean_stddev_normKernel)

void myTensorAddMeanStddevNormalization_u8_fp16(vx_uint8* input_vector, uint8_t in_zp, vx_float32 inScale,
                                            vx_uint8* input_vector1, uint8_t in_zp1, vx_float32 inScale1,
    vx_int32 v_size, vx_int32 n_batch, vx_float32 normalization_epsilon, vx_uint8* output_vector, vx_int8 out_fpp)
{
    vx_int32 batch = 0, i = 0;
    vx_float32 mean = .0f, stddev_inv = .0f, variance = .0f, input_d = .0f, data = .0f;
    vx_uint16* output = (vx_uint16*)output_vector;

    for (batch = 0; batch < n_batch; ++batch)
    {
        vx_float32 sum = 0.0f;
        vx_float32 sum_sq = 0.0f;
        for (i = 0; i < v_size; ++i)
        {
            input_d = vsi_nn_AffineToFp32(input_vector[i], inScale, in_zp, VSI_NN_TYPE_UINT8) +\
                vsi_nn_AffineToFp32(input_vector1[i], inScale1, in_zp1, VSI_NN_TYPE_UINT8);
            sum += input_d;
            sum_sq += input_d * input_d;
        }

        mean = sum / v_size;
        stddev_inv = 0.0f;
        variance = sum_sq / v_size - mean * mean;

        if (variance == 0)
        {
            stddev_inv = (vx_float32)(1.0f / sqrt(normalization_epsilon));
        }
        else
        {
            stddev_inv = (vx_float32)(1.0f / sqrt(variance));
        }

        for (i = 0; i < v_size; ++i)
        {
            input_d = vsi_nn_AffineToFp32(input_vector[i], inScale, in_zp, VSI_NN_TYPE_UINT8) +\
                vsi_nn_AffineToFp32(input_vector1[i], inScale1, in_zp1, VSI_NN_TYPE_UINT8);
            /*output_vector[i] = (input_d - mean) * stddev_inv;*/
            data = (input_d - mean) * stddev_inv;
            output[i] = vsi_nn_Fp32ToFp16(data);
        }
        input_vector += v_size;
        input_vector1 += v_size;
        output += v_size;
    }
}

void myTensorAddMeanStddevNormalization(vx_uint8* input_vector, vx_uint8* input_vector1,
                                        int8_t inFl0, int8_t inFl1, vsi_nn_type_e type0,
    vx_int32 v_size, vx_int32 n_batch, vx_float32 normalization_epsilon, vx_uint8* output_vector, vx_int8 out_fpp)
{
    vx_int32 batch = 0, i = 0;
    vx_float32 mean = .0f, stddev_inv = .0f, variance = .0f, input_d = .0f, data = .0f;
    vx_int16* input = (vx_int16*)input_vector;
    vx_int16* input1 = (vx_int16*)input_vector1;
    vx_int16* output = (vx_int16*)output_vector;

    if(type0 == VSI_NN_TYPE_INT16)
    {
        for (batch = 0; batch < n_batch; ++batch)
        {
            vx_float32 sum = 0.0f;
            vx_float32 sum_sq = 0.0f;
            for (i = 0; i < v_size; ++i)
            {
                //input_d = vsi_nn_Fp16ToFp32(input[i]) + vsi_nn_Fp16ToFp32(input1[i]);
                input_d = vsi_nn_DFPToFp32(input[i], inFl0, type0) + vsi_nn_DFPToFp32(input1[i], inFl1, type0);
                sum += input_d;
                sum_sq += input_d * input_d;
            }

            mean = sum / v_size;
            stddev_inv = 0.0f;
            variance = sum_sq / v_size - mean * mean;

            if (variance == 0)
            {
                stddev_inv = (vx_float32)(1.0f / sqrt(normalization_epsilon));
            }
            else
            {
                stddev_inv = (vx_float32)(1.0f / sqrt(variance));
            }

            for (i = 0; i < v_size; ++i)
            {
                //input_d = vsi_nn_Fp16ToFp32(input[i]) + vsi_nn_Fp16ToFp32(input1[i]);
                input_d = vsi_nn_DFPToFp32(input[i], inFl0, type0) + vsi_nn_DFPToFp32(input1[i], inFl1, type0);
                /*output_vector[i] = (input_d - mean) * stddev_inv;*/
                data = (input_d - mean) * stddev_inv;
                output[i] = vsi_nn_Fp32ToFp16(data);
            }
            input += v_size;
            input1 += v_size;
            output += v_size;
        }
    }
    else if(type0 == VSI_NN_TYPE_FLOAT16)
    {
        for (batch = 0; batch < n_batch; ++batch)
        {
            vx_float32 sum = 0.0f;
            vx_float32 sum_sq = 0.0f;
            for (i = 0; i < v_size; ++i)
            {
                input_d = vsi_nn_Fp16ToFp32(input[i]) + vsi_nn_Fp16ToFp32(input1[i]);
                sum += input_d;
                sum_sq += input_d * input_d;
            }

            mean = sum / v_size;
            stddev_inv = 0.0f;
            variance = sum_sq / v_size - mean * mean;

            if (variance == 0)
            {
                stddev_inv = (vx_float32)(1.0f / sqrt(normalization_epsilon));
            }
            else
            {
                stddev_inv = (vx_float32)(1.0f / sqrt(variance));
            }

            for (i = 0; i < v_size; ++i)
            {
                input_d = vsi_nn_Fp16ToFp32(input[i]) + vsi_nn_Fp16ToFp32(input1[i]);
                /*output_vector[i] = (input_d - mean) * stddev_inv;*/
                data = (input_d - mean) * stddev_inv;
                output[i] = vsi_nn_Fp32ToFp16(data);
            }
            input += v_size;
            input1 += v_size;
            output += v_size;
        }
    }
}

static vsi_status VX_CALLBACK vxTensor_add_mean_stddev_normKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
    vsi_status status = VX_ERROR_INVALID_PARAMETERS;

    if(paramNum == 4)
    {
        vx_context context = NULL;
        // tensor
        vx_tensor imgObj[3] = { NULL };

        uint8_t *input = NULL;
        uint8_t *input1 = NULL;
        uint8_t *output = NULL;

        uint32_t input_size[DIM_SIZE] = {0}, output_size[DIM_SIZE] = {0};
        uint32_t input_stride_size[4]  = {0};
        uint32_t output_stride_size[4] = {0};

        vx_tensor_addressing input_user_addr = NULL;
        vx_tensor_addressing input_user_addr1 = NULL;
        vx_tensor_addressing output_user_addr = NULL;

        vsi_nn_type_e inputFormat = VSI_NN_TYPE_FLOAT16, outputFormat = VSI_NN_TYPE_FLOAT16;
        uint32_t input_dims = 0, output_dims = 0;
        uint32_t i;
        int32_t in_zp, in_zp1, out_zp;
        float in_scale, in_scale1, out_scale;
        int8_t in_fixpoint = 0, in_fixpoint1, out_fixpoint = 0;
        // scalar
        vx_scalar scalar[1] = { NULL };
        float eps = .0f;

        status = VX_SUCCESS;

        imgObj[0] = (vx_tensor)paramObj[0];
        imgObj[1] = (vx_tensor)paramObj[1];
        imgObj[2] = (vx_tensor)paramObj[2];
        scalar[0] = (vx_scalar)paramObj[3];
        context = vxGetContext((vx_reference)node);
        if (context == NULL)
        {
            VSILOGE("vxGetContext failure! at line %d\n", __LINE__);
            return status;
        }
        //input
        status = vxQueryTensor(imgObj[0], VX_TENSOR_NUM_OF_DIMS, &input_dims, sizeof(input_dims));
        status |= vxQueryTensor(imgObj[0], VX_TENSOR_DATA_TYPE, &inputFormat, sizeof(inputFormat));
        status |= vxQueryTensor(imgObj[0], VX_TENSOR_DIMS, input_size, sizeof(input_size));
        status |= vxQueryTensor(imgObj[0], VX_TENSOR_ZERO_POINT, &in_zp, sizeof(in_zp));
        status |= vxQueryTensor(imgObj[0], VX_TENSOR_SCALE, &in_scale, sizeof(in_scale));
        status |= vxQueryTensor(imgObj[0], VX_TENSOR_FIXED_POINT_POS, &in_fixpoint, sizeof(in_fixpoint));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor input failure! at line %d\n", __LINE__);
            return status;
        }

        //input1
        status = vxQueryTensor(imgObj[1], VX_TENSOR_ZERO_POINT, &in_zp1, sizeof(in_zp1));
        status |= vxQueryTensor(imgObj[1], VX_TENSOR_SCALE, &in_scale1, sizeof(in_scale1));
        status |= vxQueryTensor(imgObj[1], VX_TENSOR_FIXED_POINT_POS, &in_fixpoint1, sizeof(in_fixpoint1));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor input1 failure! at line %d\n", __LINE__);
            return status;
        }

        //output
        status  = vxQueryTensor(imgObj[2], VX_TENSOR_DATA_TYPE, &outputFormat, sizeof(outputFormat));
        status |= vxQueryTensor(imgObj[2], VX_TENSOR_NUM_OF_DIMS, &output_dims, sizeof(output_dims));
        status |= vxQueryTensor(imgObj[2], VX_TENSOR_DIMS, output_size, sizeof(output_size));
        status |= vxQueryTensor(imgObj[2], VX_TENSOR_ZERO_POINT, &out_zp, sizeof(out_zp));
        status |= vxQueryTensor(imgObj[2], VX_TENSOR_SCALE, &out_scale, sizeof(out_scale));
        status |= vxQueryTensor(imgObj[2], VX_TENSOR_FIXED_POINT_POS, &out_fixpoint, sizeof(out_fixpoint));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor output failure! at line %d\n", __LINE__);
            return status;
        }

        input_size[2] = (input_dims <= 2)?1:input_size[2];
        input_size[3] = (input_dims <= 3)?1:input_size[3];

        input_stride_size[0]  = vsi_nn_GetTypeBytes(inputFormat);
        output_stride_size[0] = vsi_nn_GetTypeBytes(outputFormat);
        for (i=1; i< input_dims; i++)
        {
            input_stride_size[i]  = input_stride_size[i-1] * input_size[i-1];
            output_stride_size[i] = output_stride_size[i-1] * output_size[i-1];
        }

        if(inputFormat == VSI_NN_TYPE_FLOAT16 || inputFormat == VSI_NN_TYPE_INT16)
        {
            input  = (uint8_t*)malloc(input_size[0]*input_size[1]*input_size[2]*sizeof(int16_t));
            input1  = (uint8_t*)malloc(input_size[0]*input_size[1]*input_size[2]*sizeof(int16_t));
        }
        else if(inputFormat == VSI_NN_TYPE_UINT8)
        {
            input  = (uint8_t*)malloc(input_size[0]*input_size[1]*input_size[2]*sizeof(uint8_t));
            input1  = (uint8_t*)malloc(input_size[0]*input_size[1]*input_size[2]*sizeof(uint8_t));
        }

        if(outputFormat == VSI_NN_TYPE_FLOAT16)
        {
            output = (uint8_t*)malloc(output_size[0]*output_size[1]*output_size[2]*sizeof(int16_t));
        }
        else if(outputFormat == VSI_NN_TYPE_UINT8)
        {
            output = (uint8_t*)malloc(output_size[0]*output_size[1]*output_size[2]*sizeof(uint8_t));
        }

        input_user_addr = vxCreateTensorAddressing(context, input_size, input_stride_size, input_dims);
        vxCopyTensorPatch(imgObj[0], NULL, input_user_addr, input, VX_READ_ONLY, 0);

        input_user_addr1 = vxCreateTensorAddressing(context, input_size, input_stride_size, input_dims);
        vxCopyTensorPatch(imgObj[1], NULL, input_user_addr1, input1, VX_READ_ONLY, 0);

        // scalar
        status = vxCopyScalar(scalar[0], &eps, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxCopyScalar failure! at line %d\n", __LINE__);
            return status;
        }

        // Call C Prototype
        if((inputFormat == VSI_NN_TYPE_FLOAT16 || inputFormat == VSI_NN_TYPE_INT16)
            && outputFormat == VSI_NN_TYPE_FLOAT16)
        {
            myTensorAddMeanStddevNormalization(input, input1, in_fixpoint, in_fixpoint1,
                inputFormat,
                input_size[0], input_size[1], eps, output, out_fixpoint);
        }
        else if(inputFormat == VSI_NN_TYPE_UINT8 && outputFormat == VSI_NN_TYPE_FLOAT16)
        {
            myTensorAddMeanStddevNormalization_u8_fp16(input, in_zp, in_scale, input1,
                in_zp1, in_scale1, input_size[0], input_size[1], eps, output, out_fixpoint);
        }
        else
        {
            VSILOGE("Unsupport data type! at line %d\n", __LINE__);
            return status;
        }

        //output tensor
        output_user_addr = vxCreateTensorAddressing(context, output_size,
            output_stride_size, output_dims);
        vxCopyTensorPatch(imgObj[2], NULL, output_user_addr, output, VX_WRITE_ONLY, 0);

        if(input) free(input);
        if(input1) free(input1);
        if(output) free(output);
        if(input_user_addr) vxReleaseTensorAddressing(&input_user_addr);
        if(input_user_addr1) vxReleaseTensorAddressing(&input_user_addr1);
        if(output_user_addr) vxReleaseTensorAddressing(&output_user_addr);
    }

    return status;
} /* _VX_KERNEL_FUNC_KERNEL() */

static vx_param_description_t vxTensor_add_mean_stddev_normKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED}
};

vx_status VX_CALLBACK vxTensor_add_mean_stddev_normInitializer
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
        2,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vx_tensor     input           = (vx_tensor)paramObj[0];
    vx_tensor     input1          = (vx_tensor)paramObj[1];
    vx_tensor     output          = (vx_tensor)paramObj[2];
    vx_scalar     scalar          = (vx_scalar)paramObj[3];
    uint32_t      input_size[DIM_SIZE]   = {0};
    uint32_t      input_dims      = 0;
    vsi_nn_type_e inputDataFormat = VSI_NN_TYPE_FLOAT16, outputDataFormat = VSI_NN_TYPE_FLOAT16;
    vx_float32 scaleIn  = 0;
    int32_t input_ZP    = 0;
    vx_float32 scaleIn1 = 0;
    int32_t input_ZP1   = 0;
    vx_float32 scaleOut = 0;
    int32_t output_ZP   = 0;
    vx_int8 fixpoint = 0, fixpoint1 = 0;
    vx_float32 inScale_dfp, inScale_dfp1;

    vx_float32 eps = 0;
    vx_float32 rsEps = 0;
    vx_float32 dimRatio = 0;
    status  = vxQueryTensor(input, VX_TENSOR_DIMS, input_size, sizeof(input_size));
    status |= vxQueryTensor(input, VX_TENSOR_NUM_OF_DIMS, &input_dims, sizeof(input_dims));
    status |= vxQueryTensor(input, VX_TENSOR_DATA_TYPE, &inputDataFormat, sizeof(inputDataFormat));
    status |= vxQueryTensor(input, VX_TENSOR_ZERO_POINT, &input_ZP, sizeof(input_ZP));
    status |= vxQueryTensor(input, VX_TENSOR_SCALE, &scaleIn, sizeof(scaleIn));
    status |= vxQueryTensor(input, VX_TENSOR_FIXED_POINT_POSITION, &fixpoint, sizeof(fixpoint));
    if(VX_SUCCESS != status)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
        return status;
    }
    status |= vxQueryTensor(input1, VX_TENSOR_ZERO_POINT, &input_ZP1, sizeof(input_ZP1));
    status |= vxQueryTensor(input1, VX_TENSOR_SCALE, &scaleIn1, sizeof(scaleIn1));
    status |= vxQueryTensor(input1, VX_TENSOR_FIXED_POINT_POSITION, &fixpoint1, sizeof(fixpoint1));
    status |= vxQueryTensor(output, VX_TENSOR_DATA_TYPE, &outputDataFormat, sizeof(outputDataFormat));
    status |= vxQueryTensor(output, VX_TENSOR_ZERO_POINT, &output_ZP, sizeof(output_ZP));
    status |= vxQueryTensor(output, VX_TENSOR_SCALE, &scaleOut, sizeof(scaleOut));

    status = vxCopyScalar(scalar, &eps, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    rsEps = (vx_float32)(1.0f / sqrtf(eps));

    dimRatio = (vx_float32)(1.0 / (input_size[0]));

    if (fixpoint >= 0)
    {
        inScale_dfp = 1.0f / (vx_float32) (1 << fixpoint);
    }
    else
    {
        inScale_dfp = (vx_float32) (1 << -fixpoint);
    }

    if (fixpoint1 >= 0)
    {
        inScale_dfp1 = 1.0f / (vx_float32) (1 << fixpoint1);
    }
    else
    {
        inScale_dfp1 = (vx_float32) (1 << -fixpoint1);
    }

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkScale[0]  = 8;
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.localWorkSize[0]    = 16;
    shaderParam.localWorkSize[1]    = 1;
    shaderParam.globalWorkSize[0]   = 16;
    shaderParam.globalWorkSize[1]   = gcmALIGN((input_size[1] + shaderParam.globalWorkScale[1] - 1)
        / shaderParam.globalWorkScale[1], 4);

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));
    if(status < 0)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
    }

    {
        vx_uint32 uniAddFp16_2x8[16] = {
            0x55555555, // TCfg
            0x44444444, // ASelt
            0x33221100, 0x77665544, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00 // Constant
        };
        vx_uint32 uniFp16SumSqr_dp8x2[16] = {
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x76543210, 0x76543210, // ABin
            0x5555aaaa, // BSelt
            0x00000000, 0x76543210, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniAddFp16toFp32Lo_4x4[16] = {
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00110000, 0x00330022, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000 // Constant
        };
        vx_uint32 uniAddFp16toFp32Hi_4x4[16] = {
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00550044, 0x00770066, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000 // Constant
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
        status |= vxSetNodeUniform(nodObj, "uniAddFp16_2x8", 1, uniAddFp16_2x8);
        status |= vxSetNodeUniform(nodObj, "uniFp16SumSqr_dp8x2", 1, uniFp16SumSqr_dp8x2);
        status |= vxSetNodeUniform(nodObj, "uniAddFp16toFp32Lo_4x4", 1, uniAddFp16toFp32Lo_4x4);
        status |= vxSetNodeUniform(nodObj, "uniAddFp16toFp32Hi_4x4", 1, uniAddFp16toFp32Hi_4x4);
        status |= vxSetNodeUniform(nodObj, "uniConvertInt32toUint8_2x8", 1, uniConvertInt32toUint8_2x8);
    }

    if(inputDataFormat == VSI_NN_TYPE_UINT8 && outputDataFormat == VSI_NN_TYPE_FLOAT16)
    {
        vx_uint16  M0                   = 0;
        vx_int8    postShift            = 0;
        vx_uint32    multAndoutZP0[2]   = {0};
        vx_uint32    multAndoutZP1[2]   = {0};

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
        vx_uint32 uniU8MulAndPostShift_1_Lo_2x8[16] = {
            0xdddddddd, // TCfg
            0x44444444, // ASelt
            0x13121110, 0x17161514, // ABin
            0x11111111, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };

        vsi_nn_GetFP32MultiAndPostShift(scaleIn / scaleOut, &M0, &postShift);
        multAndoutZP0[0] = (vx_uint32)(M0);
        multAndoutZP0[1] = (vx_uint32)((output_ZP << postShift) - input_ZP * M0);
        uniU8MulAndPostShift_0_Lo_2x8[7] |= (postShift & 0x1F);

        vsi_nn_GetFP32MultiAndPostShift(scaleIn1 / scaleOut, &M0, &postShift);
        multAndoutZP1[0] = (vx_uint32)(M0);
        multAndoutZP1[1] = (vx_uint32)((output_ZP << postShift) - input_ZP1 * M0);
        uniU8MulAndPostShift_1_Lo_2x8[7] |= (postShift & 0x1F);

        status |=vxSetNodeUniform(nodObj, "uniU8MulAndPostShift_0_Lo_2x8", 1, uniU8MulAndPostShift_0_Lo_2x8);
        status |=vxSetNodeUniform(nodObj, "multAndoutZP0", 1, multAndoutZP0);
        status |=vxSetNodeUniform(nodObj, "uniU8MulAndPostShift_1_Lo_2x8", 1, uniU8MulAndPostShift_1_Lo_2x8);
        status |=vxSetNodeUniform(nodObj, "multAndoutZP1", 1, multAndoutZP1);
    }
    else if(inputDataFormat == VSI_NN_TYPE_INT16 && outputDataFormat == VSI_NN_TYPE_FLOAT16)
    {
        vx_uint32 uniConvertInt16ScaleToFp32Fst_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x01010101, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniConvertInt16ScaleToFp32Sec_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x01010101, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        status |=vxSetNodeUniform(nodObj, "uniConvertInt16ScaleToFp32Fst_4x4", 1, uniConvertInt16ScaleToFp32Fst_4x4);
        status |=vxSetNodeUniform(nodObj, "uniConvertInt16ScaleToFp32Sec_4x4", 1, uniConvertInt16ScaleToFp32Sec_4x4);
        status |= vxSetNodeUniform(nodObj, "inScale_i16", 1, &inScale_dfp);
        status |= vxSetNodeUniform(nodObj, "inScale1_i16", 1, &inScale_dfp1);
    }
    status |= vxSetNodeUniform(nodObj, "width", 1, &input_size[0]);
    status |= vxSetNodeUniform(nodObj, "dimRatio", 1, &dimRatio);
    status |= vxSetNodeUniform(nodObj, "rsEps", 1, &rsEps);

    if(status < 0)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
    }
    return status;
}


#ifdef __cplusplus
extern "C" {
#endif
vx_kernel_description_t vxTensorAddMeanStddevNorm_CPU =
{
    VX_KERNEL_ENUM_TENSOR_ADD_MEAN_STDDEV_NORM,
    VX_KERNEL_NAME_TENSORADD_MEAN_STDDEV_NORM_FP16,
    _VX_KERNEL_FUNC_KERNEL,
    vxTensor_add_mean_stddev_normKernelParam,
    _cnt_of_array( vxTensor_add_mean_stddev_normKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxTensorAddMeanStddevNormInfo_Fp16 =
{
    VX_KERNEL_ENUM_TENSOR_ADD_MEAN_STDDEV_NORM,
    VX_KERNEL_NAME_TENSORADD_MEAN_STDDEV_NORM_FP16,
    NULL,
    vxTensor_add_mean_stddev_normKernelParam,
    _cnt_of_array( vxTensor_add_mean_stddev_normKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxTensor_add_mean_stddev_normInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxTensorAddMeanStddevNormInfoU8_Fp16 =
{
    VX_KERNEL_ENUM_TENSOR_ADD_MEAN_STDDEV_NORM,
    VX_KERNEL_NAME_TENSORADD_MEAN_STDDEV_NORM_U8_FP16,
    NULL,
    vxTensor_add_mean_stddev_normKernelParam,
    _cnt_of_array( vxTensor_add_mean_stddev_normKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxTensor_add_mean_stddev_normInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxTensorAddMeanStddevNormInfoI16_Fp16 =
{
    VX_KERNEL_ENUM_TENSOR_ADD_MEAN_STDDEV_NORM,
    VX_KERNEL_NAME_TENSORADD_MEAN_STDDEV_NORM_I16_FP16,
    NULL,
    vxTensor_add_mean_stddev_normKernelParam,
    _cnt_of_array( vxTensor_add_mean_stddev_normKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxTensor_add_mean_stddev_normInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_TENSOR_ADD_MEAN_STDDEV_NORM_list[] =
{
    &vxTensorAddMeanStddevNorm_CPU,
    &vxTensorAddMeanStddevNormInfo_Fp16,
    &vxTensorAddMeanStddevNormInfoU8_Fp16,
    &vxTensorAddMeanStddevNormInfoI16_Fp16,
    NULL
};
#ifdef __cplusplus
}
#endif
