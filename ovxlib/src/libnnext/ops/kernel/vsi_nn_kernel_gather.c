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

#define _VX_KERNEL_VAR          (vx_kernel_GATHER)
#define _VX_KERNEL_ID           (VX_KERNEL_ENUM_GATHER)
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
    vsi_nn_tensor_attr_t in_attr[TENSOR_NUM_INPUT] = {0};
    vsi_nn_tensor_attr_t out_attr[TENSOR_NUM_OUTPUT] = {0};
    //uint32_t in_elements[TENSOR_NUM_INPUT] = {0};
    uint32_t out_elements[TENSOR_NUM_OUTPUT]= {0};

    int32_t axis;

    int32_t i;

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
};

vx_status VX_CALLBACK vxGatherInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    vx_uint32 paraNum
    )
{
    vx_status status = VX_SUCCESS;
    /*TODO: Add initial code for VX program*/

    return status;
}


#ifdef __cplusplus
extern "C" {
#endif
vx_kernel_description_t vxGather_CPU =
{
    _VX_KERNEL_ID,
    _VX_KERNEL_NAME,
    _VX_KERNEL_FUNC_KERNEL,
    vxGatherKernelParam,
    _cnt_of_array( vxGatherKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxGather_VX =
{
    _VX_KERNEL_ID,
    _VX_KERNEL_NAME,
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
    &vxGather_VX,
    NULL
};
#ifdef __cplusplus
}
#endif
