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
#include <math.h>
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

/*************************************L2NormalizeScale_CPU**************************************/
vsi_status VX_CALLBACK vxL2NormalizeScaleValidator
    (
    vx_node node,
    const vx_reference parameters[],
    uint32_t num,
    vx_meta_format metas[]
)
{
    vsi_status status = VX_SUCCESS;
    vx_parameter param = NULL;
    uint32_t index = 0;
#ifndef UINT16_MAX
#define UINT16_MAX ((unsigned short)0xffff)
#endif
    for(index = 0; index < num; index++)
    {
        // Validator
        if(index == 0) //tensor
        {
            vx_tensor input_tensor;
            param = vxGetParameterByIndex(node, index);
            if(param != NULL)
            {
                uint32_t     num_of_dim;
                uint32_t     input_size[6];
                vsi_enum       data_format;

                status |= vxQueryParameter(param, VX_PARAMETER_REF, &input_tensor,
                    sizeof(input_tensor));
                if (status != VX_SUCCESS)
                {
                    VSILOGE("vxQueryTensor failure! at line %d\n", __LINE__);
                }
                // num_of_dim
                status |= vxQueryTensor(input_tensor, VX_TENSOR_NUM_OF_DIMS, &num_of_dim,
                    sizeof(num_of_dim));
                if (status != VX_SUCCESS)
                {
                    VSILOGE("vxQueryTensor num_of_dim failure! at line %d\n", __LINE__);
                }
                // input_size
                status |= vxQueryTensor(input_tensor, VX_TENSOR_DIMS, input_size,
                    sizeof(input_size));
                if (status != VX_SUCCESS)
                {
                    VSILOGE("vxQueryTensor input_size failure! at line %d\n", __LINE__);
                }
                // data_format
                status |= vxQueryTensor(input_tensor, VX_TENSOR_DATA_TYPE, &data_format,
                    sizeof(data_format));
                if (status != VX_SUCCESS)
                {
                    VSILOGE("vxQueryTensor data_format failure! at line %d\n", __LINE__);
                }
                if (data_format != VX_TYPE_FLOAT16)
                    status |= VX_ERROR_INVALID_TYPE;

                status |= vxReleaseTensor(&input_tensor);

                status |= vxReleaseParameter(&param);
            }
            else
            {
                status |= VX_ERROR_INVALID_VALUE;
            }

        }
        else if(index == 1) //tensor
        {
            vx_tensor input_tensor;
            param = vxGetParameterByIndex(node, index);
            if(param != NULL)
            {
                uint32_t     num_of_dim;
                uint32_t     input_size[6];
                vsi_enum       data_format;

                status |= vxQueryParameter(param, VX_PARAMETER_REF, &input_tensor,
                    sizeof(input_tensor));
                if (status != VX_SUCCESS)
                {
                    VSILOGE("vxQueryTensor failure! at line %d\n", __LINE__);
                }
                // num_of_dim
                status |= vxQueryTensor(input_tensor, VX_TENSOR_NUM_OF_DIMS,
                    &num_of_dim, sizeof(num_of_dim));
                if (status != VX_SUCCESS)
                {
                    VSILOGE("vxQueryTensor num_of_dim failure! at line %d\n", __LINE__);
                }
                // input_size
                status |= vxQueryTensor(input_tensor, VX_TENSOR_DIMS,
                    input_size, sizeof(input_size));
                if (status != VX_SUCCESS)
                {
                    VSILOGE("vxQueryTensor input_size failure! at line %d\n", __LINE__);
                }
                // data_format
                status |= vxQueryTensor(input_tensor, VX_TENSOR_DATA_TYPE,
                    &data_format, sizeof(data_format));
                if (status != VX_SUCCESS)
                {
                    VSILOGE("vxQueryTensor data_format failure! at line %d\n", __LINE__);
                }
                if (data_format != VX_TYPE_FLOAT16)
                    status |= VX_ERROR_INVALID_TYPE;

                status |= vxReleaseTensor(&input_tensor);

                status |= vxReleaseParameter(&param);
            }
            else
            {
                status |= VX_ERROR_INVALID_VALUE;
            }

        }
        else if(index == 2) //tensor
        {
        }
        else if(index == 3) //scalar
        {
            param = vxGetParameterByIndex(node, index);
            if(param != NULL)
            {
                vx_scalar scalar = NULL;
                status |= vxQueryParameter(param, VX_PARAMETER_REF, &scalar, sizeof(scalar));
                if(status == VX_SUCCESS)
                {
                    // VX_SCALAR_TYPE
                    vsi_enum type = 0;
                    status |= vxQueryScalar(scalar, VX_SCALAR_TYPE, &type, sizeof(type));
                    if (type != VX_TYPE_INT32)
                        status = VX_ERROR_INVALID_TYPE;

                    status |= vxReleaseScalar(&scalar);
                }
                else
                {
                    status |= VX_ERROR_INVALID_VALUE;
                }
                status |= vxReleaseParameter(&param);
            }
            else
            {
                status |= VX_ERROR_INVALID_VALUE;
            }
        }
        else
        {
            VSILOGE("Validator  failure! at line %d,invalid index = %d\n", __LINE__,index);
        }

        if(status < 0)
        {
            VSILOGE("Validator  failure! at line %d,index = %d, status = %d\n",
                __LINE__,index,status);
        }
    }
    return status;
}
void myL2NormalizeScaleFunc
    (
    int16_t* imgIn,
    int16_t* scale,
    int16_t* imgOut,
    uint32_t dim,
    uint32_t width,
    uint32_t height,
    uint32_t depth
    )
{
    uint32_t  w, h, c, n;
    float scale_val = 0.0f;
    float sum = 0.0f;
    float epsilon = (float)10e-12;
    float rsqrt = 0.0f;
    int32_t   inputStridec = width * height;
    VSILOGE("Hello myL2NormalizeScaleFunc!\n");
    if (dim == 3)
    {
        for (c = 0; c < depth; c++)
        {
            scale_val = vsi_nn_Fp16toFp32(scale[c]);

            for (h = 0; h < height; h++)
            {
                for (w = 0; w < width; w++)
                {
                    float data = 0.0f;
                    uint32_t index = 0;
                    sum = 0.0f;
                    for (n = 0; n < depth; n++)
                    {
                        index = n * inputStridec + width * h + w;
                        data = vsi_nn_Fp16toFp32(imgIn[index]);
                        sum += data * data;
                    }
                    rsqrt = 1.0f / sqrtf(gcoMATH_MAX(sum, epsilon));
                    index = c * inputStridec + width * h + w;
                    data = vsi_nn_Fp16toFp32(imgIn[index]);
                    data = data * rsqrt * scale_val;
                    imgOut[index] = vsi_nn_Fp32ToFp16(data);
                }
            }
        }
    }
    else if (dim == 0)
    {
        for (c = 0; c < depth; c++)
        {
            for (h = 0; h < height; h++)
            {
                for (w = 0; w < width; w++)
                {
                    uint32_t index = w + width * (h + c * height);
                    float data = 0.0f;

                    data = vsi_nn_Fp16toFp32(imgIn[index]);
                    sum += data * data;
                }
            }
        }
        rsqrt = 1.0f / sqrtf(gcoMATH_MAX(sum, epsilon));
        scale_val = vsi_nn_Fp16toFp32(scale[0]);
        for (c = 0; c < depth; c++)
        {
            for (h = 0; h < height; h++)
            {
                for (w = 0; w < width; w++)
                {
                    uint32_t index = w + width * (h + c * height);
                    float data = 0.0f;

                    data = vsi_nn_Fp16toFp32(imgIn[index]);
                    data = data * rsqrt * scale_val;
                    imgOut[index] = vsi_nn_Fp32ToFp16(data);
                }
            }
        }
    }
    else
    {
        VSILOGE("Not support the dim number dim = %d, at line %d\n", dim, __LINE__);
    }
}
vsi_status VX_CALLBACK vxL2NormalizeScaleKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
    vsi_status status = VX_ERROR_INVALID_PARAMETERS;

    if(paramNum == 4)
    {
        vx_scalar scalar[1] = { NULL };
        vx_context context = NULL;
        uint32_t dim = 0;
        vx_tensor imgObj[3] = { NULL };
        int16_t *input, *scale, *output;
        uint32_t input_size[DIM_SIZE], scale_size[DIM_SIZE], output_size[DIM_SIZE];
        uint32_t input_stride_size[4];
        uint32_t scale_stride_size[4];
        uint32_t output_stride_size[4];
        vx_tensor_addressing input_user_addr = NULL;
        vx_tensor_addressing scale_user_addr = NULL;
        vx_tensor_addressing output_user_addr = NULL;
        vsi_enum inputFormat = VX_TYPE_FLOAT16, scaleFormat = VX_TYPE_FLOAT16;
        vsi_enum outputFormat = VX_TYPE_FLOAT16;
        uint32_t input_dims = 0, scale_dims = 0;
        uint32_t i;

        status = VX_SUCCESS;

        // tensor
        imgObj[0] = (vx_tensor)paramObj[0];
        imgObj[1] = (vx_tensor)paramObj[1];
        imgObj[2] = (vx_tensor)paramObj[2];

        // scalar
        scalar[0] = (vx_scalar)paramObj[3];
        status = vxCopyScalar(scalar[0], &dim, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor output_size failure! at line %d\n", __LINE__);
            return status;
        }

        context = vxGetContext((vx_reference)node);
        if (context == NULL)
        {
            VSILOGE("vxGetContext failure! at line %d\n", __LINE__);
            return status;
        }
        status = vxQueryTensor(imgObj[0], VX_TENSOR_NUM_OF_DIMS, &input_dims, sizeof(input_dims));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor output_size failure! at line %d\n", __LINE__);
            return status;
        }
        status = vxQueryTensor(imgObj[0], VX_TENSOR_DATA_TYPE, &inputFormat, sizeof(inputFormat));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor output_size failure! at line %d\n", __LINE__);
            return status;
        }
        status = vxQueryTensor(imgObj[0], VX_TENSOR_DIMS, input_size, sizeof(input_size));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor output_size failure! at line %d\n", __LINE__);
            return status;
        }
        status = vxQueryTensor(imgObj[1], VX_TENSOR_NUM_OF_DIMS, &scale_dims, sizeof(scale_dims));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor output_size failure! at line %d\n", __LINE__);
            return status;
        }
        status = vxQueryTensor(imgObj[1], VX_TENSOR_DATA_TYPE, &scaleFormat, sizeof(scaleFormat));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor output_size failure! at line %d\n", __LINE__);
            return status;
        }
        status = vxQueryTensor(imgObj[1], VX_TENSOR_DIMS, scale_size, sizeof(scale_size));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor output_size failure! at line %d\n", __LINE__);
            return status;
        }
        status = vxQueryTensor(imgObj[2], VX_TENSOR_DATA_TYPE,
            &outputFormat, sizeof(outputFormat));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor output_size failure! at line %d\n", __LINE__);
            return status;
        }
        status = vxQueryTensor(imgObj[2], VX_TENSOR_DIMS, output_size, sizeof(output_size));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor output_size failure! at line %d\n", __LINE__);
            return status;
        }

        input_stride_size[0] = vsi_nn_GetTypeBytes(inputFormat);
        scale_stride_size[0] = vsi_nn_GetTypeBytes(scaleFormat);
        output_stride_size[0] = vsi_nn_GetTypeBytes(outputFormat);
        for (i=1; i<4; i++)
        {
            input_stride_size[i] = input_stride_size[i-1] * input_size[i-1];
            scale_stride_size[i] = scale_stride_size[i-1] * scale_size[i-1];
            output_stride_size[i] = output_stride_size[i-1] * output_size[i-1];
        }
        input = (int16_t*)malloc(input_size[0] * input_size[1] * input_size[2] *
            sizeof(int16_t));
        scale = (int16_t*)malloc(scale_size[0] * sizeof(int16_t));
        output = (int16_t*)malloc(output_size[0] * output_size[1] * output_size[2] *
            sizeof(int16_t));

        input_user_addr = vxCreateTensorAddressing(context, input_size,
            input_stride_size, input_dims);
        vxCopyTensorPatch(imgObj[0], NULL, input_user_addr, input, VX_READ_ONLY, 0);

        scale_user_addr = vxCreateTensorAddressing(context, scale_size,
            scale_stride_size, scale_dims);
        vxCopyTensorPatch(imgObj[1], NULL, scale_user_addr, scale, VX_READ_ONLY, 0);

        // Call C Prototype
        myL2NormalizeScaleFunc(input, scale, output, dim,
            input_size[0], input_size[1], input_size[2]);

        //output tensor
        output_user_addr = vxCreateTensorAddressing(context, output_size,
            output_stride_size, input_dims);
        vxCopyTensorPatch(imgObj[2], NULL, output_user_addr, output, VX_WRITE_ONLY, 0);

        free(input);
        free(scale);
        free(output);
        vxReleaseTensorAddressing(&input_user_addr);
        vxReleaseTensorAddressing(&scale_user_addr);
        vxReleaseTensorAddressing(&output_user_addr);
    }

    return status;
}

static vx_param_description_t vxL2NormalizeScaleKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED}
};

/*************************************L2NormalizeScale_VX**********************************/
vsi_status VX_CALLBACK vxL2NormScale_SumRsqrtInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    uint32_t paraNum
    )
{
    // Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
    vx_kernel_execution_parameters_t shaderParam = {
        2,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread
    uint32_t UniFp16MulLo_dp4x4[16] = {
        0x01010101, // TCfg
        0x00000000, // ASelt
        0x00010000, 0x00030002, // ABin
        0x01010101, // BSelt
        0x00010000, 0x00030002, // BBin
        0x00000400, // AccumType, ConstantType, and PostShift
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
    };
    uint32_t UniFp16MulHi_dp4x4[16] = {
        0x01010101, // TCfg
        0x00000000, // ASelt
        0x00050004, 0x00070006, // ABin
        0x01010101, // BSelt
        0x00050004, 0x00070006, // BBin
        0x00000400, // AccumType, ConstantType, and PostShift
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
    };
    uint32_t uniIntegerSquareLo_4x4[16] = {
        0x01010101, // TCfg
        0x00000000, // ASelt
        0x00010000, 0x00030002, // ABin
        0x00000000, // BSelt
        0x00010000, 0x00030002, // BBin
        0x00000400, // AccumType, ConstantType, and PostShift
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
    };
    uint32_t uniIntegerSquareHi_4x4[16] = {
        0x01010101, // TCfg
        0x00000000, // ASelt
        0x00050004, 0x00070006, // ABin
        0x00000000, // BSelt
        0x00050004, 0x00070006, // BBin
        0x00000400, // AccumType, ConstantType, and PostShift
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
    };
    vx_uint32 uniDataSquareAddU32Lo_4x4[16] = {
        0x0d0d0d0d, // TCfg
        0x04040404, // ASelt
        0x00110000, 0x00330022, // ABin
        0x00000000, // BSelt
        0x00010000, 0x00030002, // BBin
        0x00005400, // AccumType, ConstantType, and PostShift
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
    };
    vx_uint32 uniDataSquareAddU32Hi_4x4[16] = {
        0x0d0d0d0d, // TCfg
        0x04040404, // ASelt
        0x00150004, 0x00370026, // ABin
        0x00000000, // BSelt
        0x00050004, 0x00070006, // BBin
        0x00005400, // AccumType, ConstantType, and PostShift
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
    };
    vx_uint32 uniUInt8SquareLo_4x4[16] = {
        0x69696969, // TCfg
        0x40404040, // ASelt
        0x01110000, 0x03330222, // ABin
        0x54545454, // BSelt
        0x00010000, 0x00030002, // BBin
        0x00000400, // AccumType, ConstantType, and PostShift
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
    };
    vx_uint32 uniUInt8SquareHi_4x4[16] = {
        0x69696969, // TCfg
        0x40404040, // ASelt
        0x05550444, 0x07770666, // ABin
        0x54545454, // BSelt
        0x00050004, 0x00070006, // BBin
        0x00000400, // AccumType, ConstantType, and PostShift
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
    };

    vsi_status status = VX_SUCCESS;

    vx_tensor input         = (vx_tensor)paramObj[0];
    int32_t   input_size[4] = {0};
    vsi_enum  dataFormat;
    int8_t    fixPointPos   = 0;
    int32_t   inputZP       = 0;
    float     inputScale    = 1.0f;
    float     r_inputScale  = 1.0f;

    status = vxQueryTensor(input, VX_TENSOR_DIMS, input_size, sizeof(input_size));
    status |= vxQueryTensor(input, VX_TENSOR_DATA_TYPE, &dataFormat, sizeof(dataFormat));
    status |= vxQueryTensor(input, VX_TENSOR_FIXED_POINT_POS, &fixPointPos, sizeof(fixPointPos));
    status |= vxQueryTensor(input, VX_TENSOR_SCALE, &inputScale, sizeof(inputScale));
    status |= vxQueryTensor(input, VX_TENSOR_ZERO_POINT, &inputZP, sizeof(inputZP));
    if(VX_SUCCESS != status)
        return status;

    if(dataFormat == VX_TYPE_INT8 || dataFormat == VX_TYPE_INT16)
    {
        if (fixPointPos >= 0)
            inputScale = 1.0f / (float) (1 << fixPointPos);
        else
            inputScale = (float) (1 << -fixPointPos);
    }

    r_inputScale = 1.0f / inputScale;

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkScale[0] = 8;
    shaderParam.globalWorkScale[1] = 1;
    shaderParam.globalWorkScale[2] = 1;
    shaderParam.globalWorkSize[0] = gcmALIGN((input_size[0] + shaderParam.globalWorkScale[0] - 1)
        / shaderParam.globalWorkScale[0], 4);
    shaderParam.globalWorkSize[1] = 1;


    vxSetNodeUniform(nodObj, "L2NorS_depth", 1, &input_size[1]);
    if(dataFormat == VX_TYPE_FLOAT16)
    {
        vxSetNodeUniform(nodObj, "UniFp16MulLo_dp4x4", 1, UniFp16MulLo_dp4x4);
        vxSetNodeUniform(nodObj, "UniFp16MulHi_dp4x4", 1, UniFp16MulHi_dp4x4);
    }
    else if(dataFormat == VX_TYPE_INT8)
    {
        vxSetNodeUniform(nodObj, "r_inputScale", 1, &r_inputScale);
        vxSetNodeUniform(nodObj, "uniDataSquareAddU32Lo_4x4", 1, uniDataSquareAddU32Lo_4x4);
        vxSetNodeUniform(nodObj, "uniDataSquareAddU32Hi_4x4", 1, uniDataSquareAddU32Hi_4x4);
    }
    else if(dataFormat == VX_TYPE_INT16)
    {
        vxSetNodeUniform(nodObj, "r_inputScale", 1, &r_inputScale);
        vxSetNodeUniform(nodObj, "uniIntegerSquareLo_4x4", 1, uniIntegerSquareLo_4x4);
        vxSetNodeUniform(nodObj, "uniIntegerSquareHi_4x4", 1, uniIntegerSquareHi_4x4);
    }
    else if(dataFormat == VX_TYPE_UINT8)
    {
        vxSetNodeUniform(nodObj, "r_inputScale", 1, &r_inputScale);
        vxSetNodeUniform(nodObj, "inputZP", 1, &inputZP);
        vxSetNodeUniform(nodObj, "uniUInt8SquareLo_4x4", 1, uniUInt8SquareLo_4x4);
        vxSetNodeUniform(nodObj, "uniUInt8SquareHi_4x4", 1, uniUInt8SquareHi_4x4);
    }

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));
    return VX_SUCCESS;
}

static vx_param_description_t vxL2NormScale_SumRsqrtKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED}
};

vsi_status VX_CALLBACK vxL2NormScale_MulScaleInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    uint32_t paraNum
    )
{
    // Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
    vx_kernel_execution_parameters_t shaderParam = {
        2,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vsi_status status = VX_SUCCESS;

    vsi_enum    inputFormat;
    vsi_enum    outputFormat;
    vx_tensor   input           = (vx_tensor)paramObj[0];
    vx_tensor   output          = (vx_tensor)paramObj[3];
    int32_t     input_size[DIM_SIZE]   = {0};
    int8_t      srcFixPointPos  = 0;
    int32_t     inputZP         = 0;
    int8_t      dstFixPointPos  = 0;
    int32_t     outputZP        = 0;
    float       inputScale      = 1.0f;
    float       outputScale     = 1.0f;

    status = vxQueryTensor(input, VX_TENSOR_DIMS, input_size, sizeof(input_size));
    status |= vxQueryTensor(input, VX_TENSOR_DATA_TYPE, &inputFormat, sizeof(inputFormat));
    status |= vxQueryTensor(input, VX_TENSOR_FIXED_POINT_POS, &srcFixPointPos, sizeof(srcFixPointPos));
    status |= vxQueryTensor(input, VX_TENSOR_SCALE, &inputScale, sizeof(inputScale));
    status |= vxQueryTensor(input, VX_TENSOR_ZERO_POINT, &inputZP, sizeof(inputZP));
    status |= vxQueryTensor(output, VX_TENSOR_DATA_TYPE, &outputFormat, sizeof(outputFormat));
    status |= vxQueryTensor(output, VX_TENSOR_FIXED_POINT_POS,&dstFixPointPos, sizeof(dstFixPointPos));
    status |= vxQueryTensor(output, VX_TENSOR_SCALE, &outputScale, sizeof(outputScale));
    status |= vxQueryTensor(output, VX_TENSOR_ZERO_POINT, &outputZP, sizeof(outputZP));
    if(VX_SUCCESS != status)
        return status;
    if(inputFormat == VX_TYPE_INT8 || inputFormat == VX_TYPE_INT16)
    {
        if (srcFixPointPos >= 0)
            inputScale = 1.0f / (float) (1 << srcFixPointPos);
        else
            inputScale = (float) (1 << -srcFixPointPos);

        inputZP = 0;
    }
    else if(inputFormat == VX_TYPE_FLOAT16)
    {
        inputScale     = 1.0f;
        inputZP        = 0;
    }

    if(outputFormat == VX_TYPE_INT8 || outputFormat == VX_TYPE_INT16)
    {
        if (dstFixPointPos >= 0)
            outputScale = (float) (1 << dstFixPointPos);
        else
            outputScale = 1.0f / (float) (1 << -dstFixPointPos);

        outputZP = 0;
    }
    else if(outputFormat == VX_TYPE_FLOAT16)
    {
        outputScale    = 1.0f;
        outputZP       = 0;
    }

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkScale[0]  = 8;
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.globalWorkSize[0]   = gcmALIGN((input_size[0] + shaderParam.globalWorkScale[0] - 1)
        / shaderParam.globalWorkScale[0], 4);
    shaderParam.globalWorkSize[1]   = 1;


    {
        vx_float32 IntergerScale = inputScale;
        vx_float32 output_ZP      = (vx_float32)outputZP;
        vx_uint32 uniExtact8Bin_2x8[16] = {
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniDataSubZPtoFp32Part0_4x4[16] = {
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000 // Constant
        };
        vx_uint32 uniDataSubZPtoFp32Part1_4x4[16] = {
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00050004, 0x00070006, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000 // Constant
        };
        vx_uint32 uniExtractHalf8_2x8[16] = {
            0x11111111, // TCfg
            0x11110000, // ASelt
            0x06040200, 0x06040200, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
        };
        vx_uint32 uniFp16toFp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };

        if (outputFormat == VX_TYPE_UINT8)
            IntergerScale = IntergerScale / outputScale;
        else
            IntergerScale = IntergerScale * outputScale;

        vxSetNodeUniform(nodObj, "IntergerScale", 1, &IntergerScale);
        vxSetNodeUniform(nodObj, "inputZP", 1, &inputZP);
        vxSetNodeUniform(nodObj, "output_ZP", 1, &output_ZP);
        vxSetNodeUniform(nodObj, "L2NorS_depth", 1, &input_size[1]);
        vxSetNodeUniform(nodObj, "uniDataSubZPtoFp32Part0_4x4", 1, uniDataSubZPtoFp32Part0_4x4);
        vxSetNodeUniform(nodObj, "uniDataSubZPtoFp32Part1_4x4", 1, uniDataSubZPtoFp32Part1_4x4);
        vxSetNodeUniform(nodObj, "uniFp16toFp32_4x4", 1, uniFp16toFp32_4x4);

        if(outputFormat == VX_TYPE_FLOAT16)
            vxSetNodeUniform(nodObj, "uniExtact8Bin_2x8", 1, uniExtractHalf8_2x8);
        else
            vxSetNodeUniform(nodObj, "uniExtact8Bin_2x8", 1, uniExtact8Bin_2x8);
    }

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));
    return VX_SUCCESS;
}

static vx_param_description_t vxL2NormScale_MulScaleKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED}
};

#ifdef __cplusplus
extern "C" {
#endif
vx_kernel_description_t vxL2NormalizeScaleKernelInfo_CPU =
{
    VX_KERNEL_ENUM_L2NORMALIZESCALE,
    VX_KERNEL_NAME_L2NORMALIZESCALE,
    vxL2NormalizeScaleKernel,
    vxL2NormalizeScaleKernelParam,
    (sizeof(vxL2NormalizeScaleKernelParam) / sizeof(vxL2NormalizeScaleKernelParam[0])),
    vxL2NormalizeScaleValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};
vx_kernel_description_t vxL2NormScale_SumRsqrtKernelInfo =
{
    VX_KERNEL_ENUM_L2NORMSCALE_SUMRSQRT,
    VX_KERNEL_NAME_L2NORMSCALE_SUMRSQRT,
    NULL,
    vxL2NormScale_SumRsqrtKernelParam,
    (sizeof(vxL2NormScale_SumRsqrtKernelParam) / sizeof(vxL2NormScale_SumRsqrtKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxL2NormScale_SumRsqrtInitializer,
    vsi_nn_KernelDeinitializer
};
vx_kernel_description_t vxL2NormScale_SumRsqrtKernelInfoInt8 =
{
    VX_KERNEL_ENUM_L2NORMSCALE_SUMRSQRT,
    VX_KERNEL_NAME_L2NORMSCALE_SUMRSQRT_INT8,
    NULL,
    vxL2NormScale_SumRsqrtKernelParam,
    (sizeof(vxL2NormScale_SumRsqrtKernelParam) / sizeof(vxL2NormScale_SumRsqrtKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxL2NormScale_SumRsqrtInitializer,
    vsi_nn_KernelDeinitializer
};
vx_kernel_description_t vxL2NormScale_SumRsqrtKernelInfoUInt8 =
{
    VX_KERNEL_ENUM_L2NORMSCALE_SUMRSQRT,
    VX_KERNEL_NAME_L2NORMSCALE_SUMRSQRT_UINT8,
    NULL,
    vxL2NormScale_SumRsqrtKernelParam,
    (sizeof(vxL2NormScale_SumRsqrtKernelParam) / sizeof(vxL2NormScale_SumRsqrtKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxL2NormScale_SumRsqrtInitializer,
    vsi_nn_KernelDeinitializer
};
vx_kernel_description_t vxL2NormScale_SumRsqrtKernelInfoInt16 =
{
    VX_KERNEL_ENUM_L2NORMSCALE_SUMRSQRT,
    VX_KERNEL_NAME_L2NORMSCALE_SUMRSQRT_INT16,
    NULL,
    vxL2NormScale_SumRsqrtKernelParam,
    (sizeof(vxL2NormScale_SumRsqrtKernelParam) / sizeof(vxL2NormScale_SumRsqrtKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxL2NormScale_SumRsqrtInitializer,
    vsi_nn_KernelDeinitializer
};
vx_kernel_description_t vxL2NormScale_MulScaleKernelInfo_Fp16toFp16 =
{
    VX_KERNEL_ENUM_L2NORMSCALE_MULSCALE,
    VX_KERNEL_NAME_L2NORMSCALE_FP16TOFP16,
    NULL,
    vxL2NormScale_MulScaleKernelParam,
    (sizeof(vxL2NormScale_MulScaleKernelParam) / sizeof(vxL2NormScale_MulScaleKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxL2NormScale_MulScaleInitializer,
    vsi_nn_KernelDeinitializer
};
vx_kernel_description_t vxL2NormScale_MulScaleKernelInfoInt8toInt8 =
{
    VX_KERNEL_ENUM_L2NORMSCALE_MULSCALE,
    VX_KERNEL_NAME_L2NORMSCALE_INT8TOINT8,
    NULL,
    vxL2NormScale_MulScaleKernelParam,
    (sizeof(vxL2NormScale_MulScaleKernelParam) / sizeof(vxL2NormScale_MulScaleKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxL2NormScale_MulScaleInitializer,
    vsi_nn_KernelDeinitializer
};
vx_kernel_description_t vxL2NormScale_MulScaleKernelInfoInt8toFp16 =
{
    VX_KERNEL_ENUM_L2NORMSCALE_MULSCALE,
    VX_KERNEL_NAME_L2NORMSCALE_INT8TOFP16,
    NULL,
    vxL2NormScale_MulScaleKernelParam,
    (sizeof(vxL2NormScale_MulScaleKernelParam) / sizeof(vxL2NormScale_MulScaleKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxL2NormScale_MulScaleInitializer,
    vsi_nn_KernelDeinitializer
};
vx_kernel_description_t vxL2NormScale_MulScaleKernelInfoUInt8toUInt8 =
{
    VX_KERNEL_ENUM_L2NORMSCALE_MULSCALE,
    VX_KERNEL_NAME_L2NORMSCALE_UINT8TOUINT8,
    NULL,
    vxL2NormScale_MulScaleKernelParam,
    (sizeof(vxL2NormScale_MulScaleKernelParam) / sizeof(vxL2NormScale_MulScaleKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxL2NormScale_MulScaleInitializer,
    vsi_nn_KernelDeinitializer
};
vx_kernel_description_t vxL2NormScale_MulScaleKernelInfoUInt8toFp16 =
{
    VX_KERNEL_ENUM_L2NORMSCALE_MULSCALE,
    VX_KERNEL_NAME_L2NORMSCALE_UINT8TOFP16,
    NULL,
    vxL2NormScale_MulScaleKernelParam,
    (sizeof(vxL2NormScale_MulScaleKernelParam) / sizeof(vxL2NormScale_MulScaleKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxL2NormScale_MulScaleInitializer,
    vsi_nn_KernelDeinitializer
};
vx_kernel_description_t vxL2NormScale_MulScaleKernelInfoInt16toInt16 =
{
    VX_KERNEL_ENUM_L2NORMSCALE_MULSCALE,
    VX_KERNEL_NAME_L2NORMSCALE_INT16TOINT16,
    NULL,
    vxL2NormScale_MulScaleKernelParam,
    (sizeof(vxL2NormScale_MulScaleKernelParam) / sizeof(vxL2NormScale_MulScaleKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxL2NormScale_MulScaleInitializer,
    vsi_nn_KernelDeinitializer
};
vx_kernel_description_t vxL2NormScale_MulScaleKernelInfoInt16toFp16 =
{
    VX_KERNEL_ENUM_L2NORMSCALE_MULSCALE,
    VX_KERNEL_NAME_L2NORMSCALE_INT16TOFP16,
    NULL,
    vxL2NormScale_MulScaleKernelParam,
    (sizeof(vxL2NormScale_MulScaleKernelParam) / sizeof(vxL2NormScale_MulScaleKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxL2NormScale_MulScaleInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_L2NORMALIZESCALE_list[] =
{
    &vxL2NormalizeScaleKernelInfo_CPU,
    &vxL2NormScale_SumRsqrtKernelInfo,
    &vxL2NormScale_SumRsqrtKernelInfoInt8,
    &vxL2NormScale_SumRsqrtKernelInfoUInt8,
    &vxL2NormScale_SumRsqrtKernelInfoInt16,
    &vxL2NormScale_MulScaleKernelInfo_Fp16toFp16,
    &vxL2NormScale_MulScaleKernelInfoInt8toInt8,
    &vxL2NormScale_MulScaleKernelInfoInt8toFp16,
    &vxL2NormScale_MulScaleKernelInfoUInt8toUInt8,
    &vxL2NormScale_MulScaleKernelInfoUInt8toFp16,
    &vxL2NormScale_MulScaleKernelInfoInt16toInt16,
    &vxL2NormScale_MulScaleKernelInfoInt16toFp16,
    NULL
};
#ifdef __cplusplus
}
#endif
