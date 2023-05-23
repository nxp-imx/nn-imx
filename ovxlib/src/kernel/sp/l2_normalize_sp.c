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

#if (VX_STREAM_PROCESSOR_SUPPORT)

vsi_nn_kernel_node_t l2_norm_y_direction
    (
    vsi_nn_graph_t              * graph,
    vsi_nn_tensor_t            ** inputs,
    vsi_nn_tensor_t            ** outputs,
    const vsi_nn_kernel_param_t * params
    )
{
    vsi_nn_kernel_node_t node[3] = {NULL};
    vsi_nn_tensor_attr_t attr;
    vsi_size_t shape[VSI_NN_MAX_DIM_NUM] = { 0 };
    vsi_nn_tensor_t * trans_tensor[2] = {NULL};
    uint32_t perm[2][VSI_NN_MAX_DIM_NUM] = {{1, 0, 2, 3, 4, 5, 6, 7}, {1, 0, 2, 3, 4, 5, 6, 7}};
    uint32_t i = 0;
    vx_nn_l2norm_params_t param;

    uint32_t axis = vsi_nn_kernel_param_get_int32( params, "axis" );
    param.axis = axis;
    for ( i = 0; i < inputs[0]->attr.dim_num; i++)
    {
        shape[i] = inputs[0]->attr.size[perm[0][i]];
    }

    memcpy( &attr, &inputs[0]->attr, sizeof(vsi_nn_tensor_attr_t) );
    memcpy( attr.size, shape, sizeof(shape) );
    attr.is_const = FALSE;
    attr.vtl = TRUE;
    trans_tensor[0] = vsi_nn_CreateTensor( graph, &attr );

    memcpy( &attr, &outputs[0]->attr, sizeof(vsi_nn_tensor_attr_t) );
    memcpy( attr.size, shape, sizeof(shape) );
    attr.is_const = FALSE;
    attr.vtl = TRUE;
    trans_tensor[1] = vsi_nn_CreateTensor( graph, &attr );

    node[0] = vxTensorPermuteNode
        (
        graph->g,
        inputs[0]->t,
        trans_tensor[0]->t,
        perm[0],
        inputs[0]->attr.dim_num
        );
    CHECK_PTR_FAIL_GOTO( node[0], "Create vxTensorPermuteNode fail.", final );

    node[1] = vxL2NormalizeLayer2(
        graph->g,
        trans_tensor[0]->t,
        &param,
        sizeof(vx_nn_l2norm_params_t),
        trans_tensor[1]->t
        );
    CHECK_PTR_FAIL_GOTO( node[1], "Create vxL2NormalizeLayer2 fail.", final );

    node[2] = vxTensorPermuteNode
        (
        graph->g,
        trans_tensor[1]->t,
        outputs[0]->t,
        perm[1],
        outputs[0]->attr.dim_num
        );
    CHECK_PTR_FAIL_GOTO( node[2], "Create vxTensorPermuteNode fail.", final );

final:
    vsi_safe_release_node(node[0]);
    vsi_safe_release_node(node[1]);
    vsi_safe_release_tensor(trans_tensor[0]);
    vsi_safe_release_tensor(trans_tensor[1]);

    return node[2];
} /* l2_norm_y_direction() */

#define REGISTER_L2_NORMALIZE_STREAM_PROCESSOR_KERNEL( kernel_name )   \
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

REGISTER_L2_NORMALIZE_STREAM_PROCESSOR_KERNEL( l2_norm )
{
    vsi_nn_kernel_node_t node = NULL;
    vx_nn_l2norm_params_t param;

    uint32_t axis = vsi_nn_kernel_param_get_int32( params, "axis" );
    param.axis = axis;

    VSI_UNREFERENCED(input_num);
    VSI_UNREFERENCED(output_num);
    VSI_UNREFERENCED(kernel);

    if(axis == 1)
    {
        node = l2_norm_y_direction(graph, inputs, outputs, params);
        CHECK_PTR_FAIL_GOTO( node, "Create l2_norm_y_direction fail.", final );
    }
    else
    {
        node= vxL2NormalizeLayer2(
        graph->g,
        inputs[0]->t,
        &param,
        sizeof(vx_nn_l2norm_params_t),
        outputs[0]->t
        );
        CHECK_PTR_FAIL_GOTO( node, "Create vxL2NormalizeLayer2 fail.", final );
    }
final:
    return node;
} /* l2_norm() */

#undef REGISTER_L2_NORMALIZE_STREAM_PROCESSOR_KERNEL

#endif