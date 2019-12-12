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
#include "vsi_nn_test.h"
#include "vsi_nn_log.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_math.h"
#include "client/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"

#define _VX_KERNEL_VAR          (vx_kernel_BATCH2SPACE)
#define _VX_KERNEL_ID           (VX_KERNEL_ENUM_BATCH2SPACE)
#define _VX_KERNEL_NAME         ("vsi_nn_kernel_batch2space")
#define _VX_KERNEL_FUNC_KERNEL  (vxBatch2SpaceKernel)

//static uint32_t layerNum = 0;

static vsi_status VX_CALLBACK vxBatch2SpaceKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
    /* TODO: */
#define ARG_NUM            (5)
#define TENSOR_NUM_INPUT (1)
#define TENSOR_NUM_OUTPUT (1)
#define TENSOR_NUM (TENSOR_NUM_INPUT+TENSOR_NUM_OUTPUT)

    vsi_status status = VX_SUCCESS;
    uint32_t  i = 0;
    vx_context context = NULL;
    vx_tensor input[TENSOR_NUM_INPUT] = {0};
    vx_tensor output[TENSOR_NUM_OUTPUT] = {0};
    float *f32_in_buffer[TENSOR_NUM_INPUT] = {0};
    float *f32_out_buffer[TENSOR_NUM_OUTPUT] = {0};
    vsi_nn_tensor_attr_t in_attr[TENSOR_NUM_INPUT];
    vsi_nn_tensor_attr_t out_attr[TENSOR_NUM_OUTPUT];
    uint32_t in_elements[TENSOR_NUM_INPUT] = {0};
    uint32_t out_elements[TENSOR_NUM_OUTPUT]= {0};

    int32_t block_size = 0, pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    int32_t output_batch = 0, output_depth = 0, output_height = 0, output_width = 0;
    int32_t input_batch = 0, input_depth = 0, input_height = 0, input_width = 0;
    int32_t in_b = 0;

    for(i = 0; i < TENSOR_NUM_INPUT; i++)
    {
        memset(&in_attr[i], 0x0, sizeof(vsi_nn_tensor_attr_t));
    }
    for(i = 0; i < TENSOR_NUM_OUTPUT; i ++)
    {
        memset(&out_attr[i], 0x0, sizeof(vsi_nn_tensor_attr_t));
    }
    //prepare data
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

    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM], &(block_size),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 1], &(pad_t),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 2], &(pad_b),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 3], &(pad_l),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 3], &(pad_r),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    //op calc

    output_batch = out_attr[0].size[3];
    output_depth = out_attr[0].size[2];
    output_height = out_attr[0].size[1];
    output_width = out_attr[0].size[0];

    input_batch = in_attr[0].size[3];
    input_depth = in_attr[0].size[2];
    input_height = in_attr[0].size[1];
    input_width = in_attr[0].size[0];

    for (in_b = 0; in_b < input_batch; ++in_b) {
        int32_t d = 0;
        int32_t out_b = in_b % output_batch;
        int32_t offset_w = (in_b / output_batch) % block_size;
        int32_t offset_h = (in_b / output_batch) / block_size;
        for (d = 0; d < input_depth; ++d) {
            int32_t in_h = 0;
            for (in_h = 0; in_h < input_height; ++in_h) {
                int32_t in_w = 0;
                int32_t out_h = in_h * block_size + offset_h - pad_t;
                for (in_w = 0; in_w < input_width; ++in_w) {
                    int32_t out_w = in_w * block_size + offset_w - pad_l;
                    if (out_h >= 0 && out_w >= 0 && out_h < output_height &&
                        out_w < output_width) {
                            float fval;
                            int32_t output_offset =
                                ((out_b * output_depth + d) * output_height + out_h) *
                                output_width +
                                out_w;
                            int32_t input_offset =
                                ((in_b * input_depth + d) * input_height + in_h) * input_width +
                                in_w;
                            fval = f32_in_buffer[0][input_offset];
                            f32_out_buffer[0][output_offset] = fval;
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
        if (f32_in_buffer[i])
        {
            free(f32_in_buffer[i]);
            f32_in_buffer[i] = NULL;
        }
    }
    for(i = 0; i < TENSOR_NUM_OUTPUT; i++)
    {
        if (f32_out_buffer[i])
        {
            free(f32_out_buffer[i]);
            f32_out_buffer[i] = NULL;
        }
    }

    return status;
} /* _VX_KERNEL_FUNC_KERNEL() */

static vx_param_description_t s_params[] =
{
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },
    { VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED },
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

vx_kernel_description_t * vx_kernel_BATCH2SPACE_list[] =
{
    &_VX_KERNEL_VAR,
    NULL
};
#ifdef __cplusplus
}
#endif
