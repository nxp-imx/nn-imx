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
#include "kernel/vsi_nn_kernel.h"
#include "kernel/vsi_nn_sp_unit_operation.h"
#include "kernel/vsi_nn_sp_lut.h"
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"

#if (VX_STREAM_PROCESSOR_SUPPORT)

vsi_nn_kernel_node_t vsi_nn_sp_moments_sums_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input,
        vsi_nn_tensor_t             * output0,
        vsi_nn_tensor_t             * output1
    );

vsi_nn_kernel_node_t vsi_nn_sp_moments_means_node
    (
        vsi_nn_graph_t  * graph,
        vsi_nn_tensor_t * input0,
        vsi_nn_tensor_t * input1,
        vsi_nn_tensor_t * output0,
        vsi_nn_tensor_t * output1,
        float             inv_m,
        float             const_a,
        float             s,
        float             eps
    );

vsi_nn_kernel_node_t vsi_nn_sp_bn_mov_weight_bias_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * weight,
        vsi_nn_tensor_t             * bias,
        vsi_nn_tensor_t             * dummy_output0,
        vsi_nn_tensor_t             * dummy_output1
    );

vsi_nn_kernel_node_t vsi_nn_sp_bn_in_times_v11_plus_v12_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input,
        vsi_nn_tensor_t             * dummy_tensor0,
        vsi_nn_tensor_t             * dummy_tensor1,
        vsi_nn_tensor_t             * output
    );

vsi_nn_kernel_node_t vsi_nn_sp_a_minus_v11_times_v12_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input,
        vsi_nn_tensor_t             * dummy0_tensor,
        vsi_nn_tensor_t             * dummy1_tensor,
        vsi_nn_tensor_t             * output
    )
{
    const int32_t spLoopInstsNum = 1;
    const int32_t spInstsNum = spLoopInstsNum;

    const uint32_t input_count = 3;
    const uint32_t output_count = 1;
    vx_tensor inputs_tensor[3] = {NULL};
    vx_tensor outputs_tensor[1] = {NULL};
    vx_node node = NULL;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[1];
    vsi_nn_spinst_attr_t attr;

    vsi_status status = VSI_FAILURE;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    memset(&attr, 0, sizeof(vsi_nn_spinst_attr_t));

    /* loop inst0: r1 = r1 * v12 || out = in - v11 */
    status  = vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_SR1, VSI_NN_SP_VR12, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_sub(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_VR11, VSI_NN_SP_SROUT);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;

    attr.input_setup = VSI_NN_SP_INPUT_SETUP_SINGLE_INPUT;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.ignored_leading_outputs = 3;
    attr.ignored_leading_v11_rd = 0;
    attr.ignored_leading_v12_rd = 3;
    attr.flush_cycle_num = 3;
    attr.v11_push_pop_config = VSI_NN_SP_PUSH_POP_EVERY_ROW;
    attr.v12_push_pop_config = VSI_NN_SP_PUSH_POP_EVERY_ROW;

    spinst = vsi_nn_create_spinst(graph);
    CHECK_PTR_FAIL_GOTO( spinst, "Create spInst fail.", final );
    status  = vsi_nn_add_spinst_insts(spinst, sp_insts_param, spInstsNum);
    status |= vsi_nn_set_spinst_attr(spinst, attr);
    CHECK_STATUS_FAIL_GOTO(status, final );

    inputs_tensor[0] = input->t;
    inputs_tensor[1] = dummy0_tensor->t;
    inputs_tensor[2] = dummy1_tensor->t;
    outputs_tensor[0] = output->t;
    node = vxStreamProcessorNode(
        graph->g,
        inputs_tensor,
        input_count,
        outputs_tensor,
        output_count,
        spinst->sp,
        NULL);

final:
    if (spinst)
    {
        vsi_nn_release_spinst(&spinst);
    }

    return (vsi_nn_kernel_node_t)node;
}

#define REGISTER_GROUP_NORM_STREAM_PROCESSOR_KERNEL( kernel_name )   \
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

REGISTER_GROUP_NORM_STREAM_PROCESSOR_KERNEL( group_norm )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_t * reshape_tensors[4] = {NULL};
    vsi_nn_tensor_t * dummy_tensor[6] = {NULL};
    vsi_nn_tensor_t * gamma = NULL;
    vsi_nn_tensor_t * beta = NULL;
    vsi_nn_tensor_t * norm_tensor = NULL;
    vsi_size_t shapes[2][VSI_NN_MAX_DIM_NUM] = { {1, 1, 1, 1}, {1, 1, 1, 1} };
    int32_t group_num  = vsi_nn_kernel_param_get_int32( params, "group_num" );
    int32_t group_size = outputs[0]->attr.size[2] / group_num;
    float output_scale = 1.0f / vsi_nn_get_tensor_scale(outputs[0]);
    float eps = vsi_nn_kernel_param_get_float32( params, "eps" );
    float inv_m = 1.0f / (float)(outputs[0]->attr.size[0] * outputs[0]->attr.size[1] * group_size);
    float s = inv_m * inv_m;
    float const_a = (float)(outputs[0]->attr.size[0] * outputs[0]->attr.size[1] * group_size);

    status =  vsi_nn_kernel_optimize_group_norm_shape( (const vsi_size_t*)inputs[0]->attr.size,
        inputs[0]->attr.dim_num, group_num, 0, shapes[0]);
    CHECK_STATUS_FAIL_GOTO(status, final);
    reshape_tensors[0] = vsi_nn_reshape_tensor(graph, inputs[0], shapes[0], 4);
    shapes[1][2] = inputs[0]->attr.size[2];
    reshape_tensors[2] = vsi_nn_reshape_tensor(graph, inputs[1], shapes[1], 4);
    reshape_tensors[3] = vsi_nn_reshape_tensor(graph, inputs[2], shapes[1], 4);

    memcpy( &attr, &outputs[0]->attr, sizeof(vsi_nn_tensor_attr_t) );
    memcpy( attr.size, shapes[0], VSI_NN_MAX_DIM_NUM * sizeof(vsi_size_t) );
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    attr.is_const = FALSE;
    attr.vtl = TRUE;
    norm_tensor = vsi_nn_CreateTensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( norm_tensor, "Create ifco_tensor fail.", final );
    reshape_tensors[1] = vsi_nn_reshape_tensor(graph, norm_tensor, outputs[0]->attr.size, outputs[0]->attr.dim_num);

    memcpy( &attr, &reshape_tensors[1]->attr, sizeof(vsi_nn_tensor_attr_t) );
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    attr.is_const = FALSE;
    attr.vtl = TRUE;
    attr.is_dummy = TRUE;
    attr.size[0] = 1;
    attr.size[1] = 1;
    dummy_tensor[0] = vsi_nn_CreateTensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[0], "Create dummy_tensor fail.", final );
    dummy_tensor[1] = vsi_nn_CreateTensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[1], "Create dummy_tensor fail.", final );
    dummy_tensor[2] = vsi_nn_CreateTensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[2], "Create dummy_tensor fail.", final );
    dummy_tensor[3] = vsi_nn_CreateTensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[3], "Create dummy_tensor fail.", final );
    dummy_tensor[4] = vsi_nn_CreateTensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[4], "Create dummy_tensor fail.", final );
    dummy_tensor[5] = vsi_nn_CreateTensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[5], "Create dummy_tensor fail.", final );

    gamma = vsi_nn_dropout_tensor(graph, reshape_tensors[3], output_scale);
    beta = vsi_nn_dropout_tensor(graph, reshape_tensors[2], output_scale);

    node = vsi_nn_sp_moments_sums_node(graph, reshape_tensors[0], dummy_tensor[0], dummy_tensor[1]);
    CHECK_PTR_FAIL_GOTO( node, "Create moments_sums fail.", final );
    node = vsi_nn_sp_moments_means_node(graph, dummy_tensor[0], dummy_tensor[1],
        dummy_tensor[2], dummy_tensor[3], inv_m, const_a, s, eps);
    CHECK_PTR_FAIL_GOTO( node, "Create moments_means fail.", final );
    node = vsi_nn_sp_a_minus_v11_times_v12_node(graph, reshape_tensors[0], dummy_tensor[2], dummy_tensor[3],
        norm_tensor);
    CHECK_PTR_FAIL_GOTO( node, "Create a_minus_v11_times_v12 fail.", final );
    node = vsi_nn_sp_bn_mov_weight_bias_node(graph, gamma, beta, dummy_tensor[4], dummy_tensor[5]);
    CHECK_PTR_FAIL_GOTO( node, "Create mov_weight_bias fail.", final );
    node = vsi_nn_sp_bn_in_times_v11_plus_v12_node(graph, reshape_tensors[1], dummy_tensor[4],
        dummy_tensor[5], outputs[0]);
    CHECK_PTR_FAIL_GOTO( node, "Create in_times_v11_plus_v12 fail.", final );

final:
    vsi_safe_release_tensor(dummy_tensor[0]);
    vsi_safe_release_tensor(dummy_tensor[1]);
    vsi_safe_release_tensor(dummy_tensor[2]);
    vsi_safe_release_tensor(dummy_tensor[3]);
    vsi_safe_release_tensor(dummy_tensor[4]);
    vsi_safe_release_tensor(dummy_tensor[5]);
    vsi_safe_release_tensor(gamma);
    vsi_safe_release_tensor(beta);
    vsi_safe_release_tensor(norm_tensor);
    vsi_safe_release_tensor(reshape_tensors[0]);
    vsi_safe_release_tensor(reshape_tensors[1]);
    vsi_safe_release_tensor(reshape_tensors[2]);
    vsi_safe_release_tensor(reshape_tensors[3]);

    return node;
} /* group_norm() */

#undef REGISTER_GROUP_NORM_STREAM_PROCESSOR_KERNEL

#endif
