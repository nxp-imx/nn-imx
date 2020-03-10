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
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_math.h"
#include "client/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"

#define _VX_KERNEL_VAR          (vx_kernel_SPACE2BATCH)
#define _VX_KERNEL_ID           (VX_KERNEL_ENUM_SPACE2BATCH)
#define _VX_KERNEL_NAME         ("vsi_nn_kernel_space2batch")
#define _VX_KERNEL_FUNC_KERNEL  (vxSpace2BatchKernel)

//static uint32_t layerNum = 0;

static vsi_status VX_CALLBACK vxSpace2BatchKernel
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
    vsi_nn_tensor_attr_t attr[TENSOR_NUM];
    uint32_t stride_size[TENSOR_NUM][VSI_NN_MAX_DIM_NUM];
    vx_tensor_addressing user_addr[TENSOR_NUM] = {NULL};
    uint8_t *buffer_ptr[TENSOR_NUM] = {NULL};
    vx_tensor tensor[TENSOR_NUM] = {NULL};

    int32_t block_size = 0, pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    int32_t output_batch = 0, output_depth = 0, output_height = 0, output_width = 0;
    int32_t input_batch = 0, input_depth = 0, input_height = 0, input_width = 0;
    int32_t out_b = 0;

    for(i = 0; i < TENSOR_NUM; i++)
    {
        memset(&attr[i], 0x0, sizeof(vsi_nn_tensor_attr_t));
    }

    //prepare data
    context = vxGetContext((vx_reference)node);

    for( i = 0; i < TENSOR_NUM_INPUT; i ++ )
    {
        tensor[i] = (vx_tensor)paramObj[i];
        buffer_ptr[i] = vsi_nn_ConvertRawTensorToData2(context, tensor[i],
            &(attr[i]), stride_size[i], &(user_addr[i]), VX_READ_ONLY);
    }
    for( i = TENSOR_NUM_INPUT; i < TENSOR_NUM; i ++ )
    {
        tensor[i] = (vx_tensor)paramObj[i];
        buffer_ptr[i] = vsi_nn_ConvertRawTensorToData2(context, tensor[i],
            &(attr[i]), stride_size[i], &(user_addr[i]), VX_WRITE_ONLY);
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

    output_batch = attr[1].size[3];
    output_depth = attr[1].size[2];
    output_height = attr[1].size[1];
    output_width = attr[1].size[0];

    input_batch = attr[0].size[3];
    input_depth = attr[0].size[2];
    input_height = attr[0].size[1];
    input_width = attr[0].size[0];

    for (out_b = 0; out_b < output_batch; ++out_b) {
        int32_t d;
        int32_t in_b = out_b % input_batch;
        int32_t offset_w = (out_b / input_batch) % block_size;
        int32_t offset_h = (out_b / input_batch) / block_size;
        for (d = 0; d < input_depth; ++d) {
            int32_t out_h;
            for (out_h = 0; out_h < output_height; ++out_h) {
                int32_t out_w;
                int32_t in_h = out_h * block_size + offset_h - pad_t;
                for (out_w = 0; out_w < output_width; ++out_w) {
                    float fval;
                    int32_t in_w = out_w * block_size + offset_w - pad_l;
                    int32_t output_offset =
                        ((out_b * output_depth + d) * output_height + out_h) *
                        output_width +
                        out_w;
                    int32_t input_offset =
                        ((in_b * input_depth + d) * input_height + in_h) * input_width +
                        in_w;

                    if (in_h >= 0 && in_w >= 0 && in_h < input_height &&
                        in_w < input_width) {
                            vsi_nn_DtypeToFloat32(&buffer_ptr[0][stride_size[0][0] * input_offset],
                                &fval, &attr[0].dtype);
                    } else {
                        fval = 0.0;
                    }
                    vsi_nn_Float32ToDtype(fval, &buffer_ptr[1][stride_size[1][0] * output_offset],
                        &attr[1].dtype);
                }
            }
        }
    }

    //save data
    for( i = TENSOR_NUM_INPUT; i < TENSOR_NUM; i ++ )
    {
        vsi_nn_copy_tensor_patch(tensor[i], &attr[i], buffer_ptr[i], VX_WRITE_ONLY);
    }
    for( i = 0; i < TENSOR_NUM; i ++ )
    {
        if (user_addr[i]) vxReleaseTensorAddressing(&(user_addr[i]));
        if (buffer_ptr[i]) free(buffer_ptr[i]);
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

vx_kernel_description_t * vx_kernel_SPACE2BATCH_list[] =
{
    &_VX_KERNEL_VAR,
    NULL
};
#ifdef __cplusplus
}
#endif
