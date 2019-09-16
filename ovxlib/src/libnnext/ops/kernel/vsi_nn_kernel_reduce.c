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

#include "vsi_nn_platform.h"

#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_math.h"
#include "client/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"

#define _VX_KERNEL_VAR          (vx_kernel_REDUCE)
#define _VX_KERNEL_ID           (VX_KERNEL_ENUM_REDUCE)
#define _VX_KERNEL_NAME         ("vsi_nn_kernel_reduce")
#define _VX_KERNEL_FUNC_KERNEL  (vxReduceKernel)

static vx_status VX_CALLBACK vxReduceKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    vx_uint32 paramNum
    )
{
    /* TODO: */
#define ARG_NUM            (6)
#define TENSOR_NUM_INPUT (1)
#define TENSOR_NUM_OUTPUT (1)
#define TENSOR_NUM (TENSOR_NUM_INPUT+TENSOR_NUM_OUTPUT)

    vx_status status = VX_SUCCESS;
    vx_context context = NULL;
    vsi_nn_tensor_attr_t attr[TENSOR_NUM];
    vx_uint32 stride_size[TENSOR_NUM][VSI_NN_MAX_DIM_NUM];
    vx_tensor_addressing user_addr[TENSOR_NUM] = {NULL};
    vx_uint8 *buffer_ptr[TENSOR_NUM] = {NULL};
    vx_tensor tensor[TENSOR_NUM];

    vx_float32 factor0;
    vx_int32 factor;
    vx_uint32 batch, c, h, w;
    vx_uint32 i, j, k, b;

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

    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM], &(factor0), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    //op calc
    if (factor0 > 1)
    {
        factor = (vx_int32)(factor0 + 0.5);
        w = attr[0].size[0];
        h = attr[0].size[1];
        c = attr[0].size[2];
        batch = 1;
        for(b = 0; b < batch; ++b){
            for(k = 0; k < c; ++k){
                for(j = 0; j < h*factor; ++j){
                    for(i = 0; i < w*factor; ++i){
                        vx_int32 in_index = b*w*h*c + k*w*h + (j/factor)*w + i/factor;
                        vx_int32 out_index = b*w*h*c*factor*factor + k*w*h*factor*factor +
                            j*w*factor + i;
                        vx_float32 fval;
                        //out[out_index] = in[in_index];
                        vsi_nn_DtypeToFloat32(&buffer_ptr[0][stride_size[0][0] * in_index],
                            &fval, &attr[0].dtype);
                        vsi_nn_Float32ToDtype(fval, &buffer_ptr[1][stride_size[1][0] * out_index],
                            &attr[1].dtype);
                    }
                }
            }
        }
    }
    else
    {
        factor = (vx_int32)(1 / factor0 + 0.5);
        w = attr[1].size[0];
        h = attr[1].size[1];
        c = attr[1].size[2];
        batch = 1;
        for(b = 0; b < batch; ++b){
            for(k = 0; k < c; ++k){
                for(j = 0; j < h; ++j){
                    for(i = 0; i < w; ++i){
                        vx_int32 in_index = b*w*h*c*factor*factor +
                            k*w*h*factor*factor + j*w*factor*factor + i*factor;
                        vx_int32 out_index = b*w*h*c + k*w*h + j * w + i;
                        vx_float32 fval;
                        //out[out_index] = in[in_index];
                        vsi_nn_DtypeToFloat32(&buffer_ptr[0][stride_size[0][0] * in_index], &fval,
                            &attr[0].dtype);
                        vsi_nn_Float32ToDtype(fval, &buffer_ptr[1][stride_size[1][0] * out_index],
                            &attr[1].dtype);
                    }
                }
            }
        }
    }

    //save data
    for( i = TENSOR_NUM_INPUT; i < TENSOR_NUM; i ++ )
    {
        status = vxCopyTensorPatch(
            tensor[i],
            NULL,
            user_addr[i],
            buffer_ptr[i],
            VX_WRITE_ONLY,
            0
            );
        if (user_addr[i]) vxReleaseTensorAddressing(&(user_addr[i]));
    }
    for( i = 0; i < TENSOR_NUM; i ++ )
    {
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
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL },
    { VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_OPTIONAL },
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

vx_kernel_description_t * vx_kernel_REDUCE_list[] =
{
    &_VX_KERNEL_VAR,
    NULL
};
#ifdef __cplusplus
}
#endif
