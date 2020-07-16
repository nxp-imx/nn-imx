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

#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_test.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_math.h"
#include "client/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"

#define _VX_KERNEL_VAR          (vx_kernel_STACK)
#define _VX_KERNEL_ID           (VX_KERNEL_ENUM_STACK)
#define _VX_KERNEL_NAME         (VX_KERNEL_NAME_STACK)
#define _VX_KERNEL_FUNC_KERNEL  (vxStackKernel)

static vsi_status VX_CALLBACK vxStackKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
    vsi_status status = VSI_FAILURE;
    vx_context context = NULL;
    vx_object_array input_array = (vx_object_array)paramObj[0];
    vx_tensor input[VSI_NN_STACK_MAX_INPUTS] = {0};
    vx_tensor output = NULL;
    uint8_t *in_buffer[VSI_NN_STACK_MAX_INPUTS] = {0};
    uint8_t *out_buffer = NULL;
    vsi_nn_tensor_attr_t in_attr[VSI_NN_STACK_MAX_INPUTS];
    vsi_nn_tensor_attr_t out_attr;
    //uint32_t in_elements[VSI_NN_STACK_MAX_INPUTS];
    uint32_t out_elements;
    uint32_t i, j;
    vx_size itemCount;
    uint32_t axis;

    /* prepare data */
    context = vxGetContext((vx_reference)node);
    status = vxQueryObjectArray(input_array, VX_OBJECT_ARRAY_NUMITEMS,
        &itemCount, sizeof(itemCount));
    for (i = 0; i < (uint32_t)itemCount; i++)
    {
        input[i] = (vx_tensor)vxGetObjectArrayItem(input_array, i);
    }
    output = (vx_tensor)paramObj[1];

    vxCopyScalar((vx_scalar)paramObj[2], &axis,VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    for (i = 0; i < (uint32_t)itemCount; i++)
    {
        status = vsi_nn_vxGetTensorAttr(input[i], &in_attr[i]);
        TEST_CHECK_STATUS(status, final);
        //in_elements[i] = vsi_nn_vxGetTensorElementNum(&in_attr[i]);
        in_buffer[i] = vsi_nn_vxCopyTensorToData(context, input[i], &in_attr[i]);
    }
    {
        uint32_t stride;
        status = vsi_nn_vxGetTensorAttr(output, &out_attr);
        TEST_CHECK_STATUS(status, final);
        stride = vsi_nn_TypeGetBytes(out_attr.dtype.vx_type);
        out_elements = vsi_nn_vxGetTensorElementNum(&out_attr);
        out_buffer = (uint8_t *)malloc(out_elements * stride);
    }

    /* TODO: Add CPU kernel implement */
    {
        uint32_t block_size = 1;
        uint32_t block_num = 1;
        uint32_t stride = vsi_nn_TypeGetBytes(out_attr.dtype.vx_type);
        for (i = 0; i < axis; i++)
        {
            block_size *= in_attr[0].size[i];
        }
        for (i = axis; i < in_attr[0].dim_num; i++)
        {
            block_num *= in_attr[0].size[i];
        }

        for (i = 0; i < block_num; i++)
        {
            for (j = 0; j < itemCount; j++)
            {
                uint32_t in_index = i * block_size;
                uint32_t out_index = (uint32_t)((i * itemCount + j) * block_size);
                memcpy(&(out_buffer[out_index * stride]), &(in_buffer[j][in_index * stride]),
                    block_size * stride);
            }
        }
    }

    /* save output data*/
    status = vsi_nn_vxCopyDataToTensor(context, output, &out_attr, out_buffer);
final:
    for (i = 0; i < VSI_NN_STACK_MAX_INPUTS; i++)
    {
        if (in_buffer[i]) free(in_buffer[i]);
    }
    if (out_buffer) free(out_buffer);
    return status;
} /* _VX_KERNEL_FUNC_KERNEL() */

static vx_param_description_t vxStackKernelParam[] =
{
    {VX_INPUT, VX_TYPE_OBJECT_ARRAY, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

vx_status VX_CALLBACK vxStackInitializer
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
vx_kernel_description_t vxStack_CPU =
{
    _VX_KERNEL_ID,
    _VX_KERNEL_NAME,
    _VX_KERNEL_FUNC_KERNEL,
    vxStackKernelParam,
    _cnt_of_array( vxStackKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxStack_VX =
{
    _VX_KERNEL_ID,
    _VX_KERNEL_NAME,
    NULL,
    vxStackKernelParam,
    _cnt_of_array( vxStackKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxStackInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_STACK_list[] =
{
    &vxStack_CPU,
    &vxStack_VX,
    NULL
};
#ifdef __cplusplus
}
#endif
