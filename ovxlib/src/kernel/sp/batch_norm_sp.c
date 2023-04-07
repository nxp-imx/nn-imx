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

static vsi_status vsi_nn_get_bn_weight_bias
    (
    vsi_nn_graph_t *graph,
    vsi_nn_tensor_t **inputs,
    float            eps,
    float            input_scale,
    float            output_scale,
    int32_t          axis,
    vsi_size_t       input_element_count,
    vsi_nn_tensor_t  **weight_t,
    vsi_nn_tensor_t  **bias_t
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_tensor_attr_t attr;
    float* f32_in_buffer[4] = {NULL};
    float* f32_out_buffer[2]  = {NULL};
    vsi_size_t i = 0;
    vsi_size_t element_count = vsi_nn_vxGetTensorElementNum(&inputs[0]->attr);
    vsi_size_t batch = axis == 0 ? input_element_count / element_count : 1;

    for (i = 0; i < 4; i++)
    {
        f32_in_buffer[i] = (float *)malloc(element_count * sizeof(float));
        CHECK_PTR_FAIL_GOTO( f32_in_buffer[i], "Create tensor buffer fail.", final );
    }

    for (i = 0; i < 2; i++)
    {
        vsi_size_t o_size = axis == 0 ? input_element_count : element_count;
        f32_out_buffer[i] = (float *)malloc(o_size * sizeof(float));
        CHECK_PTR_FAIL_GOTO( f32_out_buffer[i], "Create tensor buffer fail.", final );
    }

    f32_in_buffer[0] = vsi_nn_ConvertTensorToFloat32Data(graph, inputs[0]);
    f32_in_buffer[1] = vsi_nn_ConvertTensorToFloat32Data(graph, inputs[1]);
    f32_in_buffer[2] = vsi_nn_ConvertTensorToFloat32Data(graph, inputs[2]);
    f32_in_buffer[3] = vsi_nn_ConvertTensorToFloat32Data(graph, inputs[3]);

    for (i = 0; i < element_count; i++)
    {
        float mean = f32_in_buffer[0][i];
        float var = f32_in_buffer[1][i];
        float gamma = f32_in_buffer[2][i];
        float beta = f32_in_buffer[3][i];

        float w = gamma / sqrtf(var + eps);
        float b = beta - mean * w;

        w = w * input_scale / output_scale;
        b = b / output_scale;

        f32_out_buffer[0][i] = w;
        f32_out_buffer[1][i] = b;
    }

    for (i = 1; i < batch; i++)
    {
        float *dst0_ptr = f32_out_buffer[0] + element_count * i;
        float *dst1_ptr = f32_out_buffer[1] + element_count * i;

        memcpy(dst0_ptr, f32_out_buffer[0], element_count * sizeof(float));
        memcpy(dst1_ptr, f32_out_buffer[1], element_count * sizeof(float));
    }

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    attr.size[0] = 1;
    attr.size[1] = axis == 0 ? input_element_count / element_count : 1;
    attr.size[2] = 1;
    attr.size[axis] = element_count;
    attr.dim_num = axis == 0 ? 2 : 3;
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    *weight_t = vsi_nn_CreateTensor(graph, &attr);
    CHECK_PTR_FAIL_GOTO( *weight_t, "Create tensor fail.", final );
    *bias_t = vsi_nn_CreateTensor(graph, &attr);
    CHECK_PTR_FAIL_GOTO( *bias_t, "Create tensor fail.", final );
    status = vsi_nn_CopyRawDataToTensor (graph,
        (uint8_t*)f32_out_buffer[0], &attr.dtype, *weight_t);
    status |= vsi_nn_CopyRawDataToTensor (graph,
        (uint8_t*)f32_out_buffer[1], &attr.dtype, *bias_t);

final:
    for (i = 0; i < 4; i++)
    {
        vsi_nn_safe_free(f32_in_buffer[i]);
    }

    for (i = 0; i < 2; i++)
    {
        vsi_nn_safe_free(f32_out_buffer[i]);
    }

    return status;
}

vsi_nn_kernel_node_t vsi_nn_sp_bn_mov_weight_bias_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * weight,
        vsi_nn_tensor_t             * bias,
        vsi_nn_tensor_t             * dummy_output0,
        vsi_nn_tensor_t             * dummy_output1,
        char                        * kernel_name
    )
{
    const int32_t spLoopInstsNum = 2;
    const int32_t spInstsNum = spLoopInstsNum;

    const uint32_t input_count = 2;
    const uint32_t output_count = 1;
    vx_tensor inputs_tensor[2] = {NULL};
    vx_tensor outputs_tensor[2] = {NULL};
    vx_node node = NULL;
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth /
        graph->ctx->config.sp_exec_count;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[2];
    vsi_nn_spinst_attr_t attr;

    vsi_status status = VSI_FAILURE;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    /* loop inst0: v11 = in*/
    status  = vsi_nn_sp_move(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_VR11);
    /* loop inst0: v12 = in*/
    status |= vsi_nn_sp_move(&sp_insts_param[1], VSI_NN_SP_SRIN, VSI_NN_SP_VR12);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;
    attr.input_setup = VSI_NN_SP_INPUT_SETUP_INTERLEAVE_TWO_INPUT;

    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.ignored_leading_outputs = 0;
    attr.flush_cycle_num = 0;
    attr.ignored_leading_v11_rd = 0;
    attr.ignored_leading_v11_wr = 0;
    attr.ignored_leading_v12_rd = 0;
    attr.ignored_leading_v12_wr = 0;
    attr.v11_reset_at_start = VSI_NN_SP_V_RESET_AT_START_RESET;
    attr.v12_reset_at_start = VSI_NN_SP_V_RESET_AT_START_RESET;
    attr.ch0_post_redistribute = VSI_NN_SP_CH_POST_REDISTRIBUTE_VECTOR_GATHER;
    attr.ch1_post_redistribute = VSI_NN_SP_CH_POST_REDISTRIBUTE_VECTOR_GATHER;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_Z;
    attr.split_max_vector_depth = max_vector_depth;

    attr.input0_reshape = VX_SP_ATTRIBUTE_RESHAPE_CHW2HWC;
    attr.input1_reshape = VX_SP_ATTRIBUTE_RESHAPE_CHW2HWC;
    attr.output_reshape = VX_SP_ATTRIBUTE_RESHAPE_CHW2HWC;

    spinst = vsi_nn_create_spinst(graph);
    CHECK_PTR_FAIL_GOTO( spinst, "Create spInst fail.", final );
    status  = vsi_nn_add_spinst_insts(spinst, sp_insts_param, spInstsNum);
    status |= vsi_nn_set_spinst_attr(spinst, attr);
    CHECK_STATUS_FAIL_GOTO(status, final );

    inputs_tensor[0] = weight->t;
    inputs_tensor[1] = bias->t;
    outputs_tensor[0] = dummy_output0->t;
    outputs_tensor[1] = dummy_output1->t;

    node = vxStreamProcessorNode(
        graph->g,
        inputs_tensor,
        input_count,
        outputs_tensor,
        output_count,
        spinst->sp,
        NULL);

    status = vsi_nn_set_sp_kernel_name(node, kernel_name);
    CHECK_STATUS_FAIL_GOTO(status, final );
final:
    if (spinst)
    {
        vsi_nn_release_spinst(&spinst);
    }

    return (vsi_nn_kernel_node_t)node;
}

vsi_nn_kernel_node_t vsi_nn_sp_bn_in_times_v11_plus_v12_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input,
        vsi_nn_tensor_t             * dummy_tensor0,
        vsi_nn_tensor_t             * dummy_tensor1,
        vsi_nn_tensor_t             * output,
        char                        * kernel_name
    )
{
    const int32_t spLoopInstsNum = 1;
    const int32_t spInstsNum = spLoopInstsNum;

    const uint32_t input_count = 2;
    const uint32_t output_count = 1;
    vx_tensor inputs_tensor[3] = {NULL};
    vx_tensor outputs_tensor[1] = {NULL};
    vx_node node = NULL;
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth /
        graph->ctx->config.sp_exec_count;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[1];
    vsi_nn_spinst_attr_t attr;

    vsi_status status = VSI_FAILURE;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    /* loop inst0: r1 = in * v11 || out = r1 + v12 */
    status  = vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_VR11, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_add(&sp_insts_param[0], VSI_NN_SP_SR1, VSI_NN_SP_VR12, VSI_NN_SP_SROUT);
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

    attr.num_of_v11_rd_in_flush_cycle = 0;
    attr.num_of_v12_rd_in_flush_cycle = 3;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_Z;
    attr.split_max_vector_depth = max_vector_depth;

    spinst = vsi_nn_create_spinst(graph);
    CHECK_PTR_FAIL_GOTO( spinst, "Create spInst fail.", final );
    status  = vsi_nn_add_spinst_insts(spinst, sp_insts_param, spInstsNum);
    status |= vsi_nn_set_spinst_attr(spinst, attr);
    CHECK_STATUS_FAIL_GOTO(status, final );

    inputs_tensor[0] = input->t;
    inputs_tensor[1] = dummy_tensor0->t;
    inputs_tensor[2] = dummy_tensor1->t;
    outputs_tensor[0] = output->t;
    node = vxStreamProcessorNode(
        graph->g,
        inputs_tensor,
        input_count,
        outputs_tensor,
        output_count,
        spinst->sp,
        NULL);

    status = vsi_nn_set_sp_kernel_name(node, kernel_name);
    CHECK_STATUS_FAIL_GOTO(status, final );

final:
    if (spinst)
    {
        vsi_nn_release_spinst(&spinst);
    }

    return (vsi_nn_kernel_node_t)node;
}

vsi_nn_kernel_node_t vsi_nn_sp_bn_a_times_b_to_v11_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input,
        vsi_nn_tensor_t             * weight,
        vsi_nn_tensor_t             * dummy_output,
        char                        * kernel_name
    )
{
    const int32_t spLoopInstsNum = 2;
    const int32_t spInstsNum = spLoopInstsNum;

    const uint32_t input_count = 2;
    const uint32_t output_count = 1;
    vx_tensor inputs_tensor[2] = {NULL};
    vx_tensor outputs_tensor[1] = {NULL};
    vx_node node = NULL;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[2];
    vsi_nn_spinst_attr_t attr;

    vsi_status status = VSI_FAILURE;
    float clamp_min = 0;
    float clamp_max = 0;

    vsi_nn_get_tensor_clamp_min_max(input, &clamp_min, &clamp_max);

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    /* loop inst0: v12 = clamp(in * r3, r6, r7) */
    status  = vsi_nn_sp_mul_clamp(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_SR3, VSI_NN_SP_VR12);
    /* loop inst1: v11 = v12 * r2 || r2 = in*/
    status |= vsi_nn_sp_mul(&sp_insts_param[1], VSI_NN_SP_VR12, VSI_NN_SP_SR2, VSI_NN_SP_VR11);
    status |= vsi_nn_sp_move(&sp_insts_param[1], VSI_NN_SP_SRIN, VSI_NN_SP_SR2);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;
    attr.input_setup = VSI_NN_SP_INPUT_SETUP_INTERLEAVE_TWO_INPUT;

    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.ignored_leading_outputs = 0;
    attr.flush_cycle_num = 4;
    attr.ignored_leading_v11_rd = 0;
    attr.ignored_leading_v11_wr = 2;
    attr.ignored_leading_v12_rd = 2;
    attr.ignored_leading_v12_wr = 0;
    attr.v11_reset_at_start = VSI_NN_SP_V_RESET_AT_START_RESET;
    attr.v12_reset_at_start = VSI_NN_SP_V_RESET_AT_START_RESET;

    attr.num_of_v11_rd_in_flush_cycle = 0;
    attr.num_of_v12_rd_in_flush_cycle = 2;
    attr.num_of_v11_wr_in_flush_cycle = 2;
    attr.num_of_v12_wr_in_flush_cycle = 0;

    VSI_NN_SP_ATTR_SET_CONST_TO_SR3(attr, 1.0f);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR6(attr, clamp_max);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR7(attr, clamp_min);

    spinst = vsi_nn_create_spinst(graph);
    CHECK_PTR_FAIL_GOTO( spinst, "Create spInst fail.", final );
    status  = vsi_nn_add_spinst_insts(spinst, sp_insts_param, spInstsNum);
    status |= vsi_nn_set_spinst_attr(spinst, attr);
    CHECK_STATUS_FAIL_GOTO(status, final );

    inputs_tensor[0] = input->t;
    inputs_tensor[1] = weight->t;
    outputs_tensor[0] = dummy_output->t;

    node = vxStreamProcessorNode(
        graph->g,
        inputs_tensor,
        input_count,
        outputs_tensor,
        output_count,
        spinst->sp,
        NULL);

    status = vsi_nn_set_sp_kernel_name(node, kernel_name);
    CHECK_STATUS_FAIL_GOTO(status, final );

final:
    if (spinst)
    {
        vsi_nn_release_spinst(&spinst);
    }

    return (vsi_nn_kernel_node_t)node;
}

vsi_nn_kernel_node_t vsi_nn_sp_bn_a_plus_v11_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input,
        vsi_nn_tensor_t             * dummy_tensor,
        vsi_nn_tensor_t             * output,
        char                        * kernel_name
    )
{
    const int32_t spLoopInstsNum = 1;
    const int32_t spInstsNum = spLoopInstsNum;

    const uint32_t input_count = 2;
    const uint32_t output_count = 1;
    vx_tensor inputs_tensor[2] = {NULL};
    vx_tensor outputs_tensor[1] = {NULL};
    vx_node node = NULL;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[1];
    vsi_nn_spinst_attr_t attr;

    vsi_status status = VSI_FAILURE;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    /* loop inst0: out = in + v11 */
    status  = vsi_nn_sp_add(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_VR11, VSI_NN_SP_SROUT);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;

    attr.input_setup = VSI_NN_SP_INPUT_SETUP_SINGLE_INPUT;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.ignored_leading_outputs = 0;
    attr.ignored_leading_v11_rd = 0;
    attr.flush_cycle_num = 0;

    spinst = vsi_nn_create_spinst(graph);
    CHECK_PTR_FAIL_GOTO( spinst, "Create spInst fail.", final );
    status  = vsi_nn_add_spinst_insts(spinst, sp_insts_param, spInstsNum);
    status |= vsi_nn_set_spinst_attr(spinst, attr);
    CHECK_STATUS_FAIL_GOTO(status, final );

    inputs_tensor[0] = input->t;
    inputs_tensor[1] = dummy_tensor->t;
    outputs_tensor[0] = output->t;
    node = vxStreamProcessorNode(
        graph->g,
        inputs_tensor,
        input_count,
        outputs_tensor,
        output_count,
        spinst->sp,
        NULL);

    status = vsi_nn_set_sp_kernel_name(node, kernel_name);
    CHECK_STATUS_FAIL_GOTO(status, final );

final:
    if (spinst)
    {
        vsi_nn_release_spinst(&spinst);
    }

    return (vsi_nn_kernel_node_t)node;
}

#define REGISTER_BATCH_NORM_STREAM_PROCESSOR_KERNEL( kernel_name )   \
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

REGISTER_BATCH_NORM_STREAM_PROCESSOR_KERNEL( batch_norm )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_node_t node[2] = {NULL};
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_t * weight = NULL;
    vsi_nn_tensor_t * bias = NULL;
    vsi_nn_tensor_t * dummy_tensor[2] = {NULL};
    float input_scale = vsi_nn_get_tensor_scale(inputs[0]);
    float output_scale = vsi_nn_get_tensor_scale(outputs[0]);
    float eps = vsi_nn_kernel_param_get_float32( params, "eps" );
    vsi_size_t input_element_count = vsi_nn_vxGetTensorElementNum(&inputs[0]->attr);
    vsi_size_t element_count = vsi_nn_vxGetTensorElementNum(&inputs[1]->attr);
    uint32_t axis = inputs[0]->attr.dim_num < 3 ? 0 : 2;

    memcpy( &attr, &inputs[0], sizeof(vsi_nn_tensor_attr_t) );
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    attr.is_const = FALSE;
    attr.vtl = TRUE;
    attr.size[0] = 1;
    attr.size[1] = 1;
    attr.size[2] = 1;
    attr.size[axis] = element_count;
    attr.dim_num = axis == 0 ? 2 : 3;
    dummy_tensor[0] = vsi_nn_create_dummy_tensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[0], "Create dummy_tensor fail.", final );
    dummy_tensor[1] = vsi_nn_create_dummy_tensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[1], "Create dummy_tensor fail.", final );

    status = vsi_nn_get_bn_weight_bias(graph, &inputs[1], eps, input_scale, output_scale,
        axis, input_element_count, &weight, &bias);
    CHECK_STATUS_FAIL_GOTO( status, final );
    if (axis == 2)
    {
        node[0] = vsi_nn_sp_bn_mov_weight_bias_node(graph, weight, bias,
            dummy_tensor[0], dummy_tensor[1], "batchnorm_0");
        CHECK_PTR_FAIL_GOTO( node[0], "Create mov_weight_bias fail.", final );
        node[1] = vsi_nn_sp_bn_in_times_v11_plus_v12_node(graph, inputs[0], dummy_tensor[0],
            dummy_tensor[1], outputs[0], "batchnorm_1");
        CHECK_PTR_FAIL_GOTO( node[1], "Create in_times_v11_plus_v12 fail.", final );
    }
    else
    {
        node[0] = vsi_nn_sp_bn_a_times_b_to_v11_node(graph, inputs[0], weight, dummy_tensor[0], "batchnorm_0");
        CHECK_PTR_FAIL_GOTO( node[0], "Create a_times_b_to_v11 fail.", final );
        node[1] = vsi_nn_sp_bn_a_plus_v11_node(graph, bias, dummy_tensor[0], outputs[0], "batchnorm_1");
        CHECK_PTR_FAIL_GOTO( node[1], "Create a_plus_v11 fail.", final );
    }
final:
    vsi_safe_release_node(node[0]);
    vsi_safe_release_tensor(weight);
    vsi_safe_release_tensor(bias);
    vsi_safe_release_tensor(dummy_tensor[0]);
    vsi_safe_release_tensor(dummy_tensor[1]);

    return node[1];
} /* batch_norm() */

#undef REGISTER_BATCH_NORM_STREAM_PROCESSOR_KERNEL

#endif
