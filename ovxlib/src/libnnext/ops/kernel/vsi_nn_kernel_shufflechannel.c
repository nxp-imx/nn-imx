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
#include "vsi_nn_log.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_math.h"
#include "client/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"

void myShuffleChannelFunc
    (
    int16_t* imgIn,
    int32_t group_number,
    int16_t* imgOut,
    uint32_t width,
    uint32_t height,
    uint32_t channel,
    uint32_t batch
    )
{
    uint32_t group_row = group_number;
    uint32_t group_column = channel / group_number;
    uint32_t feature_map_size = width * height * channel;
    uint32_t len = width * height;
    uint32_t num = (batch <= 0 ? 1 : batch);
    uint32_t n = 0, i = 0, j = 0;

    printf("Hello myShuffleChannelFunc!\n");

    for (n = 0; n < num; n++)
    {
        for (i = 0; i < group_row; i++)
        {
            for (j = 0; j < group_column; j++)
            {
                int16_t *input_ptr  = imgIn + n * feature_map_size + (i * group_column + j) * len;
                int16_t *output_ptr = imgOut + n * feature_map_size + (j * group_row + i) * len;
                memcpy(output_ptr, input_ptr, len * sizeof(int16_t));
            }
        }
    }

    return;
}
vsi_status VX_CALLBACK vxShuffleChannelKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
    vsi_status status = VX_ERROR_INVALID_PARAMETERS;
    int16_t *input = NULL, *output = NULL;
    vx_tensor_addressing input_user_addr = NULL;
    vx_tensor_addressing output_user_addr = NULL;

    if(paramNum == 3)
    {
        vx_context context = NULL;
        // tensor
        vx_tensor imgObj[2] = { NULL };
        uint32_t input_size[4] = {0}, output_size[4] = {0};
        uint32_t input_stride_size[4]  = {0};
        uint32_t output_stride_size[4] = {0};
        vsi_enum inputFormat = VX_TYPE_FLOAT16, outputFormat = VX_TYPE_FLOAT16;
        uint32_t input_dims = 0, output_dims = 0;
        uint32_t i;
        // scalar
        vx_scalar scalar[1] = { NULL };
        int32_t group_number = 0;

        status = VX_SUCCESS;

        imgObj[0] = (vx_tensor)paramObj[0];
        imgObj[1] = (vx_tensor)paramObj[1];
        scalar[0] = (vx_scalar)paramObj[2];

        context = vxGetContext((vx_reference)node);
        if (context == NULL)
        {
            printf("vxGetContext failure! at line %d\n", __LINE__);
            return status;
        }
        status = vxQueryTensor(imgObj[0], VX_TENSOR_NUM_OF_DIMS, &input_dims, sizeof(input_dims));
        if (status != VX_SUCCESS)
        {
            printf("vxQueryTensor input_dims failure! at line %d\n", __LINE__);
            return status;
        }
        status = vxQueryTensor(imgObj[0], VX_TENSOR_DATA_TYPE, &inputFormat, sizeof(inputFormat));
        if (status != VX_SUCCESS)
        {
            printf("vxQueryTensor inputFormat failure! at line %d\n", __LINE__);
            return status;
        }
        status = vxQueryTensor(imgObj[0], VX_TENSOR_DIMS, input_size, sizeof(input_size));
        if (status != VX_SUCCESS)
        {
            printf("vxQueryTensor input_size failure! at line %d\n", __LINE__);
            return status;
        }
        status = vxQueryTensor(imgObj[1], VX_TENSOR_NUM_OF_DIMS,
            &output_dims, sizeof(output_dims));
        if (status != VX_SUCCESS)
        {
            printf("vxQueryTensor output_dims failure! at line %d\n", __LINE__);
            return status;
        }
        status = vxQueryTensor(imgObj[1], VX_TENSOR_DATA_TYPE,
            &outputFormat, sizeof(outputFormat));
        if (status != VX_SUCCESS)
        {
            printf("vxQueryTensor outputFormat failure! at line %d\n", __LINE__);
            return status;
        }
        status = vxQueryTensor(imgObj[1], VX_TENSOR_DIMS, output_size, sizeof(output_size));
        if (status != VX_SUCCESS)
        {
            printf("vxQueryTensor output_size failure! at line %d\n", __LINE__);
            return status;
        }

        input_stride_size[0]  = vsi_nn_GetTypeBytes(inputFormat);
        output_stride_size[0] = vsi_nn_GetTypeBytes(outputFormat);
        for (i=1; i< input_dims; i++)
        {
            input_stride_size[i]  = input_stride_size[i-1] * input_size[i-1];
            output_stride_size[i] = output_stride_size[i-1] * output_size[i-1];
        }
        input  = (int16_t*)malloc(input_size[0]*input_size[1]*input_size[2]*sizeof(int16_t));
        output = (int16_t*)malloc(output_size[0]*output_size[1]*output_size[2]*sizeof(int16_t));

        input_user_addr = vxCreateTensorAddressing(context, input_size,
            input_stride_size, input_dims);
        vxCopyTensorPatch(imgObj[0], NULL, input_user_addr, input, VX_READ_ONLY, 0);

        // scalar
        status = vxCopyScalar(scalar[0], &group_number, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
        if (status != VX_SUCCESS)
        {
            printf("vxCopyScalar failure! at line %d\n", __LINE__);
            goto OnError;
        }
        if (input_size[2] % group_number)
        {
            printf("input channel can't be exact divided by group number! at line %d\n", __LINE__);
            status =VX_ERROR_INVALID_PARAMETERS;
            goto OnError;
        }
        // Call C Prototype
        myShuffleChannelFunc(input, group_number, output, input_size[0],
            input_size[1], input_size[2], input_size[3]);

        //output tensor
        output_user_addr = vxCreateTensorAddressing(context, output_size,
            output_stride_size, output_dims);
        vxCopyTensorPatch(imgObj[1], NULL, output_user_addr, output, VX_WRITE_ONLY, 0);

        goto OnError;
    }

OnError:
    if(input) free(input);
    if(output) free(output);
    if(input_user_addr) vxReleaseTensorAddressing(&input_user_addr);
    if(output_user_addr) vxReleaseTensorAddressing(&output_user_addr);

    return status;
}
vsi_status VX_CALLBACK vxShuffleChannelInitializer
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

    vx_tensor     input           = (vx_tensor)paramObj[0];
    vx_scalar     group_numbers   = (vx_scalar)paramObj[2];
    uint32_t      input_size[4]   = {0};
    vsi_nn_type_e inputDataFormat = VSI_NN_TYPE_FLOAT16;
    int32_t       group_number    = 0;
    int32_t       group_column    = 0;
    float         rgroup_column   = 0.0f;

    status  = vxQueryTensor(input, VX_TENSOR_DIMS, input_size, sizeof(input_size));
    status |= vxQueryTensor(input, VX_TENSOR_DATA_TYPE, &inputDataFormat, sizeof(inputDataFormat));
    status |= vxCopyScalar(group_numbers, &group_number, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    if(VX_SUCCESS != status)
    {
        printf("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
        return status;
    }
    if (input_size[2] % group_number)
    {
        printf("input channel can't be exact divided by group number! at line %d\n", __LINE__);
        return VX_FAILURE;
    }

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkOffset[2] = 0;
    if (inputDataFormat == VSI_NN_TYPE_FLOAT16 || inputDataFormat == VSI_NN_TYPE_INT16)
        shaderParam.globalWorkScale[0]  = 8;
    else
        shaderParam.globalWorkScale[0]  = 16;
    shaderParam.globalWorkScale[1]  = 4;
    shaderParam.globalWorkScale[2]  = 1;

    shaderParam.globalWorkSize[0]   = gcmALIGN((input_size[0] + shaderParam.globalWorkScale[0] - 1)
        / shaderParam.globalWorkScale[0], 4);
    shaderParam.globalWorkSize[1]   = (input_size[1] + shaderParam.globalWorkScale[1] - 1)
        / shaderParam.globalWorkScale[1];
    shaderParam.globalWorkSize[2]   = input_size[2];

    group_column = input_size[2] / group_number;
    rgroup_column = 1.0f / group_column;

    status |= vxSetNodeUniform(nodObj, "group_column", 1, &group_column);
    status |= vxSetNodeUniform(nodObj, "rgroup_column", 1, &rgroup_column);
    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));

    if(status < 0)
    {
        printf("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
    }
    return status;
}
static vx_param_description_t vxShuffleChannelKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED}
};
#ifdef __cplusplus
extern "C" {
#endif
vx_kernel_description_t vxShuffleChannelKernelInfo =
{
    VX_KERNEL_ENUM_SHUFFLECHANNEL,
    VX_KERNEL_NAME_SHUFFLECHANNEL,
    NULL,
    vxShuffleChannelKernelParam,
    (sizeof(vxShuffleChannelKernelParam) / sizeof(vxShuffleChannelKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxShuffleChannelInitializer,
    vsi_nn_KernelDeinitializer
};
vx_kernel_description_t vxShuffleChannelKernelInfo8Bits =
{
    VX_KERNEL_ENUM_SHUFFLECHANNEL,
    VX_KERNEL_NAME_SHUFFLECHANNEL8BITS,
    NULL,
    vxShuffleChannelKernelParam,
    (sizeof(vxShuffleChannelKernelParam) / sizeof(vxShuffleChannelKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxShuffleChannelInitializer,
    vsi_nn_KernelDeinitializer
};
vx_kernel_description_t vxShuffleChannelKernelInfo_CPU =
{
    VX_KERNEL_ENUM_SHUFFLECHANNEL,
    VX_KERNEL_NAME_SHUFFLECHANNEL,
    vxShuffleChannelKernel,
    vxShuffleChannelKernelParam,
    (sizeof(vxShuffleChannelKernelParam) / sizeof(vxShuffleChannelKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_SHUFFLECHANNEL_list[] =
{
    &vxShuffleChannelKernelInfo_CPU,
    &vxShuffleChannelKernelInfo,
    &vxShuffleChannelKernelInfo8Bits,
    NULL
};
#ifdef __cplusplus
}
#endif
