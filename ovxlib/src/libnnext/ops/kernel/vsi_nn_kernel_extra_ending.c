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

#define _VX_KERNEL_VAR          (vx_kernel_EXTRA_ENDING)
#define _VX_KERNEL_ID           (VX_KERNEL_ENUM_EXTRA_ENDING)
#define _VX_KERNEL_FUNC_KERNEL  (vxExtra_endingKernel)

static vx_uint32 vxcExtraGetTypeSize(vx_enum format)
{
    switch(format)
    {
    case VX_TYPE_INT8:
    case VX_TYPE_UINT8:
        return 1;
    case VX_TYPE_INT16:
    case VX_TYPE_UINT16:
        return 2;
    case VX_TYPE_INT32:
    case VX_TYPE_UINT32:
        return 4;
    case VX_TYPE_INT64:
    case VX_TYPE_UINT64:
        return 8;
    case VX_TYPE_FLOAT32:
        return 4;
    case VX_TYPE_FLOAT64:
        return 8;
    case VX_TYPE_ENUM:
        return 4;
    case VX_TYPE_FLOAT16:
        return 2;
    }

    return 4;
}

vx_status extraTensorRead(vx_context context, vx_tensor ts, void *buf)
{
    vx_uint32  ts_size[4];
    vx_uint32 input_stride_size[4];
    vx_uint32 output_num,num_of_dim;
    vx_tensor_addressing input_user_addr = NULL;
    vx_status status = VX_FAILURE;
    vx_uint32 i = 0;
    void *dataBuf = (vx_uint16 *)buf;
    vx_uint32 dataFormat;

    status = vxQueryTensor(ts, VX_TENSOR_NUM_OF_DIMS, &num_of_dim, sizeof(num_of_dim));
    status |= vxQueryTensor(ts, VX_TENSOR_DIMS, ts_size, sizeof(ts_size));
    status |= vxQueryTensor(ts, VX_TENSOR_DATA_TYPE, &dataFormat, sizeof(dataFormat));

    input_stride_size[0] = vxcExtraGetTypeSize(dataFormat);
    output_num = ts_size[0];
    for (i=1; i< num_of_dim; i++)
    {
        input_stride_size[i] = input_stride_size[i - 1] * ts_size[i - 1];
        output_num *= ts_size[i];
    }

    if(dataBuf == NULL)
    {
        printf("TensorRead fail! input empty \n");
        return VX_FAILURE;
    }

    input_user_addr = vxCreateTensorAddressing(
        context,
        ts_size,
        input_stride_size,
        num_of_dim
        );
    status = vxCopyTensorPatch(
        ts,
        NULL,
        input_user_addr,
        dataBuf,
        VX_READ_ONLY,
        0
        );
    vxReleaseTensorAddressing(&input_user_addr);
    if(status < 0)
    {
        free(dataBuf);
        dataBuf = NULL;
        printf("TensorRead fail! status = %d\n",status);
        return status;
    }

    return VX_SUCCESS;
}

static vsi_status VX_CALLBACK vxExtra_endingKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
#define TENSOR_NUM_INPUT (2)
#define TENSOR_NUM_OUTPUT (1)

    vsi_status status = VSI_FAILURE;
    vx_context context = NULL;
    vx_tensor intput = NULL;
    vx_tensor output[TENSOR_NUM_OUTPUT] = {0};
    uint8_t *u8_in_buffer[1] = {0};
    uint8_t *u8_out_buffer[TENSOR_NUM_OUTPUT] = {0};
    vsi_nn_tensor_attr_t out_attr[TENSOR_NUM_OUTPUT] = {0};
    uint32_t out_elements[TENSOR_NUM_OUTPUT]= {0};
    vx_uint32  dst_size[4]   = {0, 0, 0, 0};
    vx_tensor_addressing output_user_addr = NULL;
    uint32_t output_stride_size[4] = {0};
    int32_t output_dims = 0;
    vsi_nn_type_e outputFormat;

    int32_t i;

    /* prepare data */
    context = vxGetContext((vx_reference)node);

    intput = (vx_tensor)paramObj[1];

    for(i = 0; i < 1; i ++)
    {
        output[i] = (vx_tensor)paramObj[i + TENSOR_NUM_INPUT];
        status = vsi_nn_vxGetTensorAttr(output[i], &out_attr[i]);
        TEST_CHECK_STATUS(status, final);
        out_elements[i] = vsi_nn_vxGetTensorElementNum(&out_attr[i]);
        u8_out_buffer[i]= (uint8_t *)malloc(out_elements[i] * sizeof(uint8_t));
        memset(u8_out_buffer[i], 0, out_elements[i] * sizeof(uint8_t));

        status |= vxQueryTensor(output[i], VX_TENSOR_NUMBER_OF_DIMS, &output_dims, sizeof(output_dims));
        status |= vxQueryTensor(output[i], VX_TENSOR_DIMS, dst_size, sizeof(dst_size));
        status |= vxQueryTensor(output[i], VX_TENSOR_DATA_TYPE, &outputFormat, sizeof(outputFormat));

        u8_in_buffer[i]= (uint8_t *)malloc(out_elements[i] * sizeof(uint8_t));

        status = extraTensorRead(context, intput, u8_in_buffer[0]);
        memcpy(u8_out_buffer[0], u8_in_buffer[0], out_elements[i] * sizeof(uint8_t));
    }

    output_stride_size[0] = vsi_nn_GetTypeBytes(outputFormat);
    for (i=1; i< output_dims; i++)
    {
        output_stride_size[i] = output_stride_size[i-1] * dst_size[i-1];
    }

    /* save data */
    output_user_addr = vxCreateTensorAddressing(context, dst_size,
        output_stride_size, output_dims);
    vxCopyTensorPatch(output[0], NULL, output_user_addr, u8_out_buffer[0], VX_WRITE_ONLY, 0);

final:
    for(i = 0; i < TENSOR_NUM_OUTPUT; i++)
    {
        if (u8_out_buffer[i]) free(u8_out_buffer[i]);
    }
    if (u8_in_buffer[0]) free(u8_in_buffer[0]);
    if(output_user_addr) vxReleaseTensorAddressing(&output_user_addr);
    return status;
} /* _VX_KERNEL_FUNC_KERNEL() */

static vx_param_description_t vxExtra_endingKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
};

vx_status VX_CALLBACK vxExtra_endingInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    vx_uint32 paraNum
    )
{
    vx_status status = VX_SUCCESS;
// Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
    vx_kernel_execution_parameters_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vx_tensor  output           = (vx_tensor)paramObj[2];

    vx_uint32 width             = 0;
    vx_uint32 height            = 0;
    vx_uint32 channel           = 0;

    vx_uint32  dst_size[4]   = {0, 0, 0, 0};

    status = vxQueryTensor(output, VX_TENSOR_DIMS, dst_size, sizeof(dst_size));

    width = dst_size[0];
    height = dst_size[1];
    channel = dst_size[2];

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkOffset[2] = 0;
    shaderParam.globalWorkScale[0]  = 8;
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.globalWorkScale[2]  = 1;
    shaderParam.localWorkSize[0]    = 16;
    shaderParam.localWorkSize[1]    = 1;
    shaderParam.localWorkSize[2]    = 1;
    shaderParam.globalWorkSize[0]   = gcmALIGN((width + shaderParam.globalWorkScale[0] - 1)
                                        / shaderParam.globalWorkScale[0], shaderParam.localWorkSize[0]);
    shaderParam.globalWorkSize[1]   = gcmALIGN((height + shaderParam.globalWorkScale[1] - 1)
                                        / shaderParam.globalWorkScale[1], shaderParam.localWorkSize[1]);
    shaderParam.globalWorkSize[2]   = channel;

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
                                    &shaderParam, sizeof(vx_kernel_execution_parameters_t));

    if(status < 0)
        printf("error-%s,%d\n",__FILE__,__LINE__);

    return status;
}


#ifdef __cplusplus
extern "C" {
#endif
vx_kernel_description_t vxExtra_ending_CPU =
{
    _VX_KERNEL_ID,
    VX_KERNEL_NAME_EXTRA_ENDING_I16,
    _VX_KERNEL_FUNC_KERNEL,
    vxExtra_endingKernelParam,
    _cnt_of_array( vxExtra_endingKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxExtra_ending_i16 =
{
    _VX_KERNEL_ID,
    VX_KERNEL_NAME_EXTRA_ENDING_I16,
    NULL,
    vxExtra_endingKernelParam,
    _cnt_of_array( vxExtra_endingKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxExtra_endingInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxExtra_ending_i8 =
{
    _VX_KERNEL_ID,
    VX_KERNEL_NAME_EXTRA_ENDING_I8,
    NULL,
    vxExtra_endingKernelParam,
    _cnt_of_array( vxExtra_endingKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxExtra_endingInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxExtra_ending_u8 =
{
    _VX_KERNEL_ID,
    VX_KERNEL_NAME_EXTRA_ENDING_U8,
    NULL,
    vxExtra_endingKernelParam,
    _cnt_of_array( vxExtra_endingKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxExtra_endingInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_EXTRA_ENDING_list[] =
{
    &vxExtra_ending_CPU,
    &vxExtra_ending_i16,
    &vxExtra_ending_i8,
    &vxExtra_ending_u8,
    NULL
};
#ifdef __cplusplus
}
#endif
