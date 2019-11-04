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

#define _VX_KERNEL_NAME         (VX_KERNEL_NAME_GATHER)
#define _VX_KERNEL_FUNC_KERNEL  (vxGatherKernel)

static vsi_status VX_CALLBACK vxGatherKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
#define ARG_NUM            (1)
#define TENSOR_NUM_INPUT (2)
#define TENSOR_NUM_OUTPUT (1)
#define TENSOR_NUM (TENSOR_NUM_INPUT+TENSOR_NUM_OUTPUT)

    vsi_status status = VSI_FAILURE;
    vx_context context = NULL;
    vx_tensor input[TENSOR_NUM_INPUT] = {0};
    vx_tensor output[TENSOR_NUM_OUTPUT] = {0};
    uint8_t *in_buffer[TENSOR_NUM_INPUT] = {0};
    uint8_t *out_buffer[TENSOR_NUM_OUTPUT] = {0};
    vsi_nn_tensor_attr_t in_attr[TENSOR_NUM_INPUT];
    vsi_nn_tensor_attr_t out_attr[TENSOR_NUM_OUTPUT];
    //uint32_t in_elements[TENSOR_NUM_INPUT] = {0};
    uint32_t out_elements[TENSOR_NUM_OUTPUT]= {0};

    int32_t axis;

    int32_t i;
    for(i = 0; i < TENSOR_NUM_INPUT; i++)
    {
        memset(&in_attr[i], 0x0, sizeof(vsi_nn_tensor_attr_t));
    }
    for(i = 0; i < TENSOR_NUM_OUTPUT; i++)
    {
        memset(&out_attr[i], 0x0, sizeof(vsi_nn_tensor_attr_t));
    }
    /* prepare data */
    context = vxGetContext((vx_reference)node);

    for(i = 0; i < TENSOR_NUM_INPUT; i ++)
    {
        input[i] = (vx_tensor)paramObj[i];
        status = vsi_nn_vxGetTensorAttr(input[i], &in_attr[i]);
        TEST_CHECK_STATUS(status, final);
        //in_elements[i] = vsi_nn_vxGetTensorElementNum(&in_attr[i]);
        in_buffer[i] = vsi_nn_vxCopyTensorToData(context, input[i], &in_attr[i]);
    }
    for(i = 0; i < TENSOR_NUM_OUTPUT; i ++)
    {
        uint32_t stride;
        output[i] = (vx_tensor)paramObj[i + TENSOR_NUM_INPUT];
        status = vsi_nn_vxGetTensorAttr(output[i], &out_attr[i]);
        TEST_CHECK_STATUS(status, final);
        stride = vsi_nn_TypeGetBytes(out_attr[i].dtype.vx_type);
        out_elements[i] = vsi_nn_vxGetTensorElementNum(&out_attr[i]);
        out_buffer[i] = (uint8_t *)malloc(out_elements[i] * stride);
    }
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM], &(axis),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    /* TODO: Add CPU kernel implement */
    {
        uint32_t block_size = 1, block_num = 1, indices_num = 1, axis_num;
        uint32_t i, j;
        uint32_t stride = vsi_nn_TypeGetBytes(in_attr[0].dtype.vx_type);

        for(i = 0; i < (uint32_t)axis; ++i)
        {
            block_size *= in_attr[0].size[i];
        }

        for(i = 0; i < in_attr[1].dim_num; ++i)
        {
            indices_num *= in_attr[1].size[i];
        }

        for(i = axis + 1; i < in_attr[0].dim_num; ++i)
        {
            block_num *= in_attr[0].size[i];
        }
        axis_num = in_attr[0].size[axis];

        for(i = 0; i < block_num; i++)
        {
            for(j = 0; j < indices_num; j++)
            {
                uint32_t indice = *((uint32_t *)&(in_buffer[1][j * sizeof(uint32_t)]));
                uint32_t in_index = (i * axis_num + indice) * block_size;
                uint32_t out_index = (i * indices_num + j) * block_size;

                memcpy(&(out_buffer[0][out_index * stride]), &(in_buffer[0][in_index * stride]),
                    block_size * stride);
            }
        }
    }
    /* save data */
    for(i = 0; i < TENSOR_NUM_OUTPUT; i++)
    {
        vsi_nn_vxCopyDataToTensor(context, output[i], &out_attr[i], out_buffer[i]);
    }

final:
    for (i = 0; i < TENSOR_NUM_INPUT; i++)
    {
        if (in_buffer[i]) free(in_buffer[i]);
    }
    for(i = 0; i < TENSOR_NUM_OUTPUT; i++)
    {
        if (out_buffer[i]) free(out_buffer[i]);
    }
    return status;
} /* _VX_KERNEL_FUNC_KERNEL() */

static vx_param_description_t vxGatherKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

static vx_param_description_t vxGatherCpuParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

vx_status VX_CALLBACK vxGatherInitializer
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

    vx_scalar     scalar[2];
    vx_tensor     input0          = (vx_tensor)paramObj[0];
    vx_tensor     input1          = (vx_tensor)paramObj[1];

    uint32_t      input1_size[DIM_SIZE]   = {0};
    int32_t       block_size = 0;
    int32_t       block_num = 0;
    int32_t       indices_num = 0;
    uint32_t      input_dims1      = 0;
    vsi_nn_type_e inputDataFormat = VSI_NN_TYPE_FLOAT16;

    scalar[0]            = (vx_scalar)paramObj[3];
    scalar[1]            = (vx_scalar)paramObj[4];

    status  = vxQueryTensor(input0, VX_TENSOR_DATA_TYPE, &inputDataFormat, sizeof(inputDataFormat));
    status |= vxQueryTensor(input1, VX_TENSOR_DIMS, input1_size, sizeof(input1_size));
    status |= vxQueryTensor(input1, VX_TENSOR_NUM_OF_DIMS, &input_dims1, sizeof(input_dims1));
    if(VX_SUCCESS != status)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
        return status;
    }

    indices_num = input1_size[0] * input1_size[1];
    if(input_dims1 == 3)
        indices_num *= input1_size[2];

    status = vxCopyScalar(scalar[0], &block_size, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    status |= vxCopyScalar(scalar[1], &block_num, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    if(VX_SUCCESS != status)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
        return status;
    }

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkOffset[2] = 0;
    shaderParam.globalWorkScale[0]  = 16;
    if(inputDataFormat == VSI_NN_TYPE_FLOAT16 || inputDataFormat == VSI_NN_TYPE_INT16)
        shaderParam.globalWorkScale[0]  = 8;
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.globalWorkScale[2]  = 1;
    shaderParam.globalWorkSize[0]   = gcmALIGN((block_size + shaderParam.globalWorkScale[0] - 1)
        / shaderParam.globalWorkScale[0], 4);
    shaderParam.globalWorkSize[1]   = indices_num;
    shaderParam.globalWorkSize[2]   = block_num;

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));
    if(status < 0)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
    }

    status = vxSetNodeUniform(nodObj, "indices_num", 1, &indices_num);
    if(status < 0)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
    }

    return status;
}


#ifdef __cplusplus
extern "C" {
#endif
vx_kernel_description_t vxGather_CPU =
{
    VX_KERNEL_ENUM_GATHER,
    _VX_KERNEL_NAME,
    _VX_KERNEL_FUNC_KERNEL,
    vxGatherCpuParam,
    _cnt_of_array( vxGatherCpuParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxGather_int8 =
{
    VX_KERNEL_ENUM_GATHER,
    VX_KERNEL_NAME_GATHER_INT8,
    NULL,
    vxGatherKernelParam,
    _cnt_of_array( vxGatherKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxGatherInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxGather_int16 =
{
    VX_KERNEL_ENUM_GATHER,
    VX_KERNEL_NAME_GATHER_INT16,
    NULL,
    vxGatherKernelParam,
    _cnt_of_array( vxGatherKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxGatherInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_GATHER_list[] =
{
    &vxGather_CPU,
    &vxGather_int8,
    &vxGather_int16,
    NULL
};
#ifdef __cplusplus
}
#endif
