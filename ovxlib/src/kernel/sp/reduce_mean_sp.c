/****************************************************************************
*
*    Copyright (c) 2021 Vivante Corporation
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

#include "vsi_nn_types.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_node.h"
#include "vsi_nn_log.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_error.h"
#include "vsi_nn_tensor_util_prv.h"
#include "vsi_nn_kernel_prv.h"
#include "kernel/vsi_nn_kernel.h"
#include "kernel/vsi_nn_sp_unit_operation.h"
#include "kernel/vsi_nn_sp_lut.h"

#if defined(VX_STREAM_PROCESSOR_SUPPORT) && defined(VX_REDUCE_OPS_VX_SUPPORT)

#define REGISTER_REDUCE_MEAN_STREAM_PROCESSOR_KERNEL( kernel_name )   \
    static vsi_nn_kernel_node_t _##kernel_name##setup \
        ( \
        vsi_nn_graph_t              * graph, \
        vsi_nn_tensor_t            ** inputs, \
        size_t                        input_num, \
        vsi_nn_tensor_t            ** outputs, \
        size_t                        output_num,\
        const vsi_nn_kernel_param_t * params, \
        vsi_nn_kernel_t             * kernel \
        ); \
    REGISTER_BACKEND_STREAM_PROCESSOR( kernel_name, _##kernel_name##setup ) \
    static vsi_nn_kernel_node_t _##kernel_name##setup \
        ( \
        vsi_nn_graph_t              * graph, \
        vsi_nn_tensor_t            ** inputs, \
        size_t                        input_num, \
        vsi_nn_tensor_t            ** outputs, \
        size_t                        output_num,\
        const vsi_nn_kernel_param_t * params, \
        vsi_nn_kernel_t             * kernel \
        )

REGISTER_REDUCE_MEAN_STREAM_PROCESSOR_KERNEL( reduce_mean )
{
    vsi_nn_kernel_node_t node = NULL;
    int32_t axis_num = vsi_nn_kernel_param_get_int32(params, "axis_num");
    const int32_t* axis =
        (const int32_t*)vsi_nn_kernel_param_get_str(params, "axis");
    vsi_nn_tensor_attr_t attr;
    vx_nn_mean_params_t reduce_param;
    int32_t i = 0;
    vsi_nn_tensor_t* reduce_axis = NULL;
    int32_t data[VSI_NN_MAX_DIM_NUM] = {0};

    for (i = 0; i < axis_num; i++) {
        data[i] = axis[i];
    }
    memset(&attr, 0, sizeof(attr));
    attr.dim_num = 1;
    attr.size[0] = 2;

    attr.dtype.vx_type = VSI_NN_TYPE_INT32;
    attr.is_const = TRUE;
    attr.vtl = FALSE;
    reduce_axis = vsi_nn_CreateTensorFromData(graph, (uint8_t*)data, &attr);
    reduce_param.keep_dims = TRUE;
    reduce_param.axis = reduce_axis->t;

    VSI_UNREFERENCED(input_num);
    VSI_UNREFERENCED(output_num);
    VSI_UNREFERENCED(kernel);
    node = vxTensorMeanNode(graph->g,
                            inputs[0]->t,
                            &reduce_param,
                            sizeof(reduce_param),
                            outputs[0]->t);
    CHECK_PTR_FAIL_GOTO(node, "Create reduce mean node fail.", final);
final:
    vsi_safe_release_tensor(reduce_axis);
    return (vsi_nn_kernel_node_t)node;
} /* reduce_mean() */

#undef REGISTER_REDUCE_MEAN_STREAM_PROCESSOR_KERNEL

#endif