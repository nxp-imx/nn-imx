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


#define _VX_KERNEL_VAR          (vx_kernel_REVERSE)
#define _VX_KERNEL_ID           (VX_KERNEL_ENUM_RESIZE)
#define _VX_KERNEL_NAME         ("REVERSE")
#define _VX_KERNEL_FUNC_KERNEL  (vxReverseKernel)

static vsi_status VX_CALLBACK vxReverseKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
    vsi_status status = VX_ERROR_INVALID_PARAMETERS;
    if( 3 == paramNum )
    {
        vx_context  context = NULL;
        vx_tensor   input_tensor = NULL;
        vx_tensor   output_tensor = NULL;
        uint8_t *  input_buffer = NULL;
        uint8_t *  output_buffer = NULL;
        vx_scalar   axis_scalar = NULL;
        uint32_t   axis = 1;
        uint32_t   input_dims = 0;
        uint32_t   output_dims = 0;
        vsi_enum     inputFormat = VSI_NN_TYPE_FLOAT16;
        vsi_enum     outputFormat = VSI_NN_TYPE_FLOAT16;
        uint32_t   input_size[VSI_NN_MAX_DIM_NUM] = { 0 };
        uint32_t   output_size[VSI_NN_MAX_DIM_NUM] = { 0 };
        uint32_t   input_stride_size[VSI_NN_MAX_DIM_NUM] = { 0 };
        uint32_t   output_stride_size[VSI_NN_MAX_DIM_NUM] = { 0 };
        vx_tensor_addressing input_user_addr = NULL;
        vx_tensor_addressing output_user_addr = NULL;
        vsi_nn_tensor_attr_t out_attr;

        status = VX_SUCCESS;

        memset(&out_attr, 0x0, sizeof(vsi_nn_tensor_attr_t));

        input_tensor = (vx_tensor)paramObj[0];
        output_tensor = (vx_tensor)paramObj[1];
        axis_scalar = (vx_scalar)paramObj[2];

        context = vxGetContext((vx_reference)node);
        if( NULL == context)
        {
            VSILOGE("vxGetContext failure!\n");
            status = VX_FAILURE;
            goto OnError;
        }

        input_buffer = vsi_nn_ConvertRawTensorToData(context, input_tensor,
            &input_dims, &inputFormat, input_size, input_stride_size,
            &input_user_addr, VX_READ_ONLY);
        if( NULL == input_buffer )
        {
            VSILOGE("vsi_nn_ConvertRawTensorToData failure!\n");
            status = VX_ERROR_NO_MEMORY;
            goto OnError;
        }

        output_buffer = vsi_nn_ConvertRawTensorToData(context, output_tensor,
            &output_dims, &outputFormat, output_size, output_stride_size,
            &output_user_addr, VX_WRITE_ONLY);
        if( NULL == output_buffer )
        {
            VSILOGE("vsi_nn_ConvertRawTensorToData failure!\n");
            status = VX_ERROR_NO_MEMORY;
            goto OnError;
        }

        status = vsi_nn_vxGetTensorAttr(output_tensor, &out_attr);
        if (status != VX_SUCCESS)
        {
            VSILOGE("vsi_nn_vxGetTensorAttr failure! at line %d\n", __LINE__);
            goto OnError;
        }

        status = vxCopyScalar(axis_scalar, &axis, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
        if( VX_SUCCESS != status)
        {
            VSILOGE("vxCopyScalar axis failure! status:%d\n", status);
            goto OnError;
        }

        if( input_dims != output_dims || axis >= input_dims )
        {
            VSILOGE("Invalid parameters, input_dims(%d) output_dims(%d) axis(%d) \
                    must match 'input_dims == output_dims and axis < input_dims'\n",
                    input_dims, output_dims, axis);
            status = VX_ERROR_INVALID_PARAMETERS;
            goto OnError;
        }

        {
            uint32_t i = 0;
            uint32_t j = 0;
            uint32_t fixed_num = 1;
            uint32_t changed_num = 1;
            uint32_t data_bytes = vsi_nn_TypeGetBytes(inputFormat);
            uint32_t cur_axis_sz = 0;
            uint32_t changed_bytes = 0;

            cur_axis_sz = input_size[input_dims - 1 - axis];

            for( i = 0; i < axis; i++ )
            {
                fixed_num *= input_size[input_dims - 1 - i];
            }

            for( i = axis + 1; i < input_dims; i++ )
            {
                changed_num *= input_size[input_dims - 1 - i];
            }

            changed_bytes = changed_num * data_bytes;

            for(i = 0; i < fixed_num; i++ )
            {
                for( j = 0; j < cur_axis_sz; j++ )
                {
                    uint8_t* src_addr = input_buffer + i * cur_axis_sz * changed_bytes +
                        j * changed_bytes;

                    uint8_t* dest_addr = output_buffer + i * cur_axis_sz * changed_bytes +
                        (cur_axis_sz - j - 1) * changed_bytes;

                    memcpy(dest_addr, src_addr, changed_bytes);
                }
            }

#if defined(_SAVE_TENSOR)
            {
                static int count = 0;
                char fname[256] = { 0 };
                sprintf(fname, "reverse_output_tensor.%d.axis.%d.txt", count, axis);
                vsi_nn_SaveDataToText(fname, output_buffer,
                    vsi_nn_ShapeProduct(output_size, output_dims), VSI_NN_TYPE_FLOAT16, NULL);
                count++;
            }
#endif
        }
        status = vsi_nn_vxCopyDataToTensor(context, output_tensor, &out_attr, output_buffer);
        TEST_CHECK_STATUS(status, OnError);

OnError:
        if( NULL != input_buffer )
        {
            free( input_buffer );
            input_buffer = NULL;
        }
        if( NULL != output_buffer )
        {
            free( output_buffer );
            output_buffer = NULL;
        }

        if (input_user_addr)
        {
            vxReleaseTensorAddressing(&input_user_addr);
        }
        if (output_user_addr)
        {
            vxReleaseTensorAddressing(&output_user_addr);
        }
    }

    return status;
} /* _VX_KERNEL_FUNC_KERNEL() */

static vx_param_description_t s_params[] =
{
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },
    { VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
};

vsi_status VX_CALLBACK vxTensorReverseInitializer
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

    vx_tensor input     = (vx_tensor)paramObj[0];
    uint32_t input_size[DIM_SIZE] = {1, 1, 1, 1};
    uint32_t cur_axis_sz_sub1 = 0;
    vx_uint32 i = 0;
    vsi_nn_tensor_attr_t attr;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    status = vsi_nn_vxGetTensorAttr(input, &attr);
    if (status != VX_SUCCESS)
    {
        VSILOGE("vsi_nn_vxGetTensorAttr  failure! at line %d\n", __LINE__);
        return status;
    }
    for (i = 0; i < attr.dim_num; i++)
    {
        input_size[i] = attr.size[i];
    }

    cur_axis_sz_sub1 = input_size[1] - 1;

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkScale[0]  = 8;
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.globalWorkSize[0]   = gcmALIGN((input_size[0] + shaderParam.globalWorkScale[0] - 1)
        / shaderParam.globalWorkScale[0], 4);
    shaderParam.globalWorkSize[1]   = (input_size[1] + shaderParam.globalWorkScale[1] - 1)
        / shaderParam.globalWorkScale[1];

    vxSetNodeUniform(nodObj, "cur_axis_sz_sub1", 1, &cur_axis_sz_sub1);
    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));

    return VX_SUCCESS;
}

static vx_param_description_t vxTensorReverseKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED}
};

#ifdef __cplusplus
extern "C" {
#endif
vx_kernel_description_t _VX_KERNEL_VAR =
{
    _VX_KERNEL_ID,
    _VX_KERNEL_NAME,
    _VX_KERNEL_FUNC_KERNEL,
    s_params,
    _cnt_of_array( s_params ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxTensorReverseKernelInfo =
{
    VX_KERNEL_ENUM_TENSORREVERSE,
    VX_KERNEL_NAME_TENSORREVERSE,
    NULL,
    vxTensorReverseKernelParam,
    (sizeof(vxTensorReverseKernelParam) / sizeof(vxTensorReverseKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxTensorReverseInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_REVERSE_list[] =
{
    &_VX_KERNEL_VAR,
    &vxTensorReverseKernelInfo,
    NULL
};
#ifdef __cplusplus
}
#endif
