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

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "vsi_nn_types.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_log.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_error.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "kernel/vsi_nn_kernel_eltwise.h"
#include "client/vsi_nn_vxkernel.h"

__BEGIN_DECLS

#define _CPU_ARG_NUM            (0)
#define _CPU_INPUT_NUM          (2)
#define _CPU_OUTPUT_NUM         (1)
#define _CPU_IO_NUM             (_CPU_INPUT_NUM + _CPU_OUTPUT_NUM)
#define _CPU_PARAM_NUM          (_CPU_ARG_NUM + _CPU_IO_NUM)
#define _KERNEL_NAME            CVIVANTE_NAMESPACE("maximum_sw")

static uint32_t getExpandTensorOffset(uint32_t index, uint32_t num_of_dims, uint32_t * in_dims,
                                       uint32_t *strides, uint32_t * out_dims)
{
    uint32_t offset = 0;
    uint32_t i;

    for(i = 0; i < num_of_dims; i++)
    {
        if(in_dims[i] == out_dims[i])
            offset += strides[i] * (index % out_dims[i]);

        index /= out_dims[i];
    }

    return offset;
}

static vsi_status VX_CALLBACK _maximum_kernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
    vsi_status status = VX_SUCCESS;
    vx_tensor            input0             = NULL;
    vx_tensor            input1             = NULL;
    vx_tensor            output             = NULL;
    float                *f32_in0_buffer    = NULL;
    float                *f32_in1_buffer    = NULL;
    float                *f32_out_buffer    = NULL;
    uint32_t             in0_elements       = 0;
    uint32_t             in1_elements       = 0;
    uint32_t             out_elements       = 0;
    vsi_nn_tensor_attr_t attr[_CPU_IO_NUM];
    uint32_t    stride_size[_CPU_IO_NUM][VSI_NN_MAX_DIM_NUM];

    vx_context   context                     = vxGetContext((vx_reference)node);
    vx_uint32    i                           = 0;

    for (i = 0; i < _CPU_IO_NUM; i++)
    {
        memset(&attr[i], 0, sizeof(vsi_nn_tensor_attr_t));
    }

    input0  = (vx_tensor)paramObj[0];
    input1  = (vx_tensor)paramObj[1];
    output = (vx_tensor)paramObj[2];

    /* Fill input & output attribute data struct */
    status = vsi_nn_vxGetTensorAttr(input0, &attr[0]);
    CHECK_STATUS_FAIL_GOTO(status, final);
    status = vsi_nn_vxGetTensorAttr(input1, &attr[1]);
    CHECK_STATUS_FAIL_GOTO(status, final);
    status = vsi_nn_vxGetTensorAttr(output, &attr[2]);
    CHECK_STATUS_FAIL_GOTO(status, final);

    vsi_nn_GetStrideSize( &attr[0], stride_size[0] );
    vsi_nn_GetStrideSize( &attr[1], stride_size[1] );
    vsi_nn_GetStrideSize( &attr[2], stride_size[2] );

    in0_elements = vsi_nn_vxGetTensorElementNum(&attr[0]);
    in1_elements = vsi_nn_vxGetTensorElementNum(&attr[1]);
    out_elements = vsi_nn_vxGetTensorElementNum(&attr[2]);

    /* alloc the float32 data buffer */
    f32_in0_buffer = (float *)malloc(in0_elements * sizeof(float));
    VSI_CHECK_PTR(f32_in0_buffer, "malloc data failed", final);
    f32_in1_buffer = (float *)malloc(in1_elements * sizeof(float));
    VSI_CHECK_PTR(f32_in1_buffer, "malloc data failed", final);
    f32_out_buffer = (float *)malloc(out_elements * sizeof(float));
    VSI_CHECK_PTR(f32_out_buffer, "malloc data failed", final);
    memset(f32_in0_buffer, 0, in0_elements * sizeof(float));
    memset(f32_in1_buffer, 0, in1_elements * sizeof(float));
    memset(f32_out_buffer, 0, out_elements * sizeof(float));

    /* Copy tensor to buffer, and convert bufer to float32 format */
    status = vsi_nn_vxConvertTensorToFloat32Data(
        context, input0, &attr[0], f32_in0_buffer, in0_elements * sizeof(float));
    CHECK_STATUS_FAIL_GOTO(status, final);
    status = vsi_nn_vxConvertTensorToFloat32Data(
        context, input1, &attr[1], f32_in1_buffer, in1_elements * sizeof(float));
    CHECK_STATUS_FAIL_GOTO(status, final);

    for (i = 0; i < out_elements; i++)
    {
        uint32_t  in0offset = 0;
        uint32_t  in1offset = 0;
        float   *in0_ptr  = NULL;
        float   *in1_ptr  = NULL;
        vx_float32 in0Data   = 0;
        vx_float32 in1Data   = 0;

        in0offset = getExpandTensorOffset(i, attr[0].dim_num, attr[0].size, stride_size[0], attr[2].size);
        in1offset = getExpandTensorOffset(i, attr[1].dim_num, attr[1].size, stride_size[1], attr[2].size);

        in0_ptr = f32_in0_buffer + in0offset / stride_size[0][0];
        in1_ptr = f32_in1_buffer + in1offset / stride_size[1][0];

        in0Data = in0_ptr[0];
        in1Data = in1_ptr[0];

        f32_out_buffer[i] = vsi_nn_max(in0Data, in1Data);

    }

    status = vsi_nn_vxConvertFloat32DataToTensor(
        context, output, &attr[2], f32_out_buffer, out_elements * sizeof(float));
    CHECK_STATUS_FAIL_GOTO(status, final);

final:
    if(f32_in0_buffer)free(f32_in0_buffer);
    if(f32_in1_buffer)free(f32_in1_buffer);
    if(f32_out_buffer)free(f32_out_buffer);

    return status;
} /* _minimum_kernel() */

static vx_param_description_t kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED}
};


const static vx_kernel_description_t _kernel_info =
{
    KERNEL_ID_PLACEHOLDER,
    _KERNEL_NAME,
    _maximum_kernel,
    kernel_param_def,
    _cnt_of_array( kernel_param_def ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_nn_kernel_t* kernel
    )
{
    memmove( &kernel->info, &_kernel_info, sizeof(vx_kernel_description_t) );
    return VSI_SUCCESS;
} /* _query_kernel() */

static vsi_nn_kernel_node_t _setup
    (
    vsi_nn_graph_t              * graph,
    vsi_nn_tensor_t            ** inputs,
    size_t                        input_num,
    vsi_nn_tensor_t            ** outputs,
    size_t                        output_num,
    const vsi_nn_kernel_param_t * params,
    vsi_nn_kernel_t             * kernel
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_kernel_node_param_t backend_params[_CPU_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;

    status = _query_kernel( inputs, outputs, kernel );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( backend_params, _CPU_PARAM_NUM,
                    inputs, _CPU_INPUT_NUM, outputs, _CPU_OUTPUT_NUM );
            /* Pass parameters to node. */
            status = vsi_nn_kernel_node_pass_param( node, backend_params, _CPU_PARAM_NUM );
        }
        else
        {
            status = VSI_FAILURE;
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( maximum, _setup )

