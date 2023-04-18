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

vsi_nn_kernel_node_t layer_norm_y_direction
    (
    vsi_nn_graph_t              * graph,
    vsi_nn_tensor_t            ** inputs,
    vsi_nn_tensor_t            ** outputs,
    const vsi_nn_kernel_param_t * params
    );

vsi_nn_kernel_node_t layer_norm_z_direction
    (
    vsi_nn_graph_t              * graph,
    vsi_nn_tensor_t            ** inputs,
    vsi_nn_tensor_t            ** outputs,
    const vsi_nn_kernel_param_t * params
    );

vsi_nn_kernel_node_t vsi_nn_sp_sums_axis0_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input,
        vsi_nn_tensor_t             * output0,
        vsi_nn_tensor_t             * output1,
        char                        * kernel_name
    )
{
    const int32_t spInitInstsNum = 0;
    const int32_t spLoopInstsNum = 2;
    const int32_t spInstsNum = spInitInstsNum + spLoopInstsNum;

    const uint32_t input_count = 1;
    const uint32_t output_count = 2;
    vx_tensor inputs_tensor[1] = {NULL};
    vx_tensor outputs_tensor[2] = {NULL};
    vx_node node = NULL;
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth /
        graph->ctx->config.sp_exec_count;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[2];
    vsi_nn_spinst_attr_t attr;
    float input_scale = vsi_nn_get_tensor_scale(input);
    float clamp_min = 0;
    float clamp_max = 0;

    vsi_status status = VSI_FAILURE;

    vsi_nn_get_tensor_clamp_min_max(input, &clamp_min, &clamp_max);
    clamp_min = clamp_min * input_scale;
    clamp_max = clamp_max * input_scale;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    /* loop inst0: r1 = clamp(r3 * in, r6, r7) | acc0 = r2 + r4 | out = r2 */
    status  = vsi_nn_sp_mul_clamp(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_SR3, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_add(&sp_insts_param[0], VSI_NN_SP_SR2, VSI_NN_SP_SR4, VSI_NN_SP_ACC);
    status |= vsi_nn_sp_move(&sp_insts_param[0], VSI_NN_SP_SR2, VSI_NN_SP_SROUT);
    /* loop inst1: acc1 = r2 * r2 | r2 = r1 */
    status |= vsi_nn_sp_mul(&sp_insts_param[1], VSI_NN_SP_SR2, VSI_NN_SP_SR2, VSI_NN_SP_ACC);
    status |= vsi_nn_sp_move(&sp_insts_param[1], VSI_NN_SP_SR1, VSI_NN_SP_SR2);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_YZMERGE;
    attr.input_setup = VSI_NN_SP_INPUT_SETUP_SINGLE_INPUT;

    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.ignored_leading_acc_out = 6;
    attr.ignored_leading_outputs = 3;
    attr.flush_cycle_num = 7;
    attr.accelerator_input_select = VSI_NN_SP_ACCELERATOR_IN_FROM_ACCEL;
    attr.sum_engine_reset = VSI_NN_SP_SUM_ENGINE_RESET_START_FROM_ZERO;
    attr.sum_engine_control = VSI_NN_SP_ACCUM_2D;
    attr.sum_engine_num_ch_minus_one = VSI_NN_SP_SUM_ENGINE_NUM_CH_TWO_CH;
    attr.sum_engine_2d_accum_storeage = VSI_NN_SP_ACCM_STOREAGE_DIFFERENT;
    attr.v11_reset_at_start = VSI_NN_SP_V_RESET_AT_START_RESET;
    attr.v12_reset_at_start = VSI_NN_SP_V_RESET_AT_START_RESET;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_YZ;
    attr.split_max_vector_depth = max_vector_depth;

    VSI_NN_SP_ATTR_SET_CONST_TO_SR3(attr, input_scale);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR4(attr, 0);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR6(attr, clamp_max);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR7(attr, clamp_min);

    spinst = vsi_nn_create_spinst(graph);
    CHECK_PTR_FAIL_GOTO( spinst, "Create spInst fail.", final );
    status  = vsi_nn_add_spinst_insts(spinst, sp_insts_param, spInstsNum);
    status |= vsi_nn_set_spinst_attr(spinst, attr);
    CHECK_STATUS_FAIL_GOTO(status, final );

    inputs_tensor[0] = input->t;
    outputs_tensor[0] = output0->t;
    outputs_tensor[1] = output1->t;
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

vsi_nn_kernel_node_t vsi_nn_sp_layer_norm_means_node
    (
        vsi_nn_graph_t  * graph,
        vsi_nn_tensor_t * input,
        vsi_nn_tensor_t * output,
        float             inv_m,
        float             eps,
        float             output_scale,
        char            * kernel_name
    )
{
    const int32_t spInitInstsNum = 0;
    const int32_t spLoopInstsNum = output_scale == 1 ? 5 : 6;
    const int32_t spInstsNum = spInitInstsNum + spLoopInstsNum;

    const uint32_t input_count = 1;
    const uint32_t output_count = 1;
    vx_tensor inputs_tensor[1] = {NULL};
    vx_tensor outputs_tensor[1] = {NULL};
    vx_node node = NULL;
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth /
        graph->ctx->config.sp_exec_count;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[6];
    vsi_nn_spinst_attr_t attr;
    vx_lut_params_s vx_lut_params;

    vsi_status status = VSI_FAILURE;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);
    memset(&vx_lut_params, 0, sizeof(vx_lut_params_s));

    if (output_scale == 1)
    {
        /* loop inst0: r5 = v11 * r3 */
        status  = vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_VR11, VSI_NN_SP_SR3, VSI_NN_SP_SR5);
        /* loop inst1: r5 = v12 * r3 | r6 = r5 - r4 | r10 = r1 */
        status |= vsi_nn_sp_mul(&sp_insts_param[1], VSI_NN_SP_VR12, VSI_NN_SP_SR3, VSI_NN_SP_SR5);
        status |= vsi_nn_sp_sub(&sp_insts_param[1], VSI_NN_SP_SR5, VSI_NN_SP_SR4, VSI_NN_SP_SR6);
        status |= vsi_nn_sp_move(&sp_insts_param[1], VSI_NN_SP_SR1, VSI_NN_SP_SR10);
        /* loop inst2: r9 = pwlMul() | r7 = pwlAdd() */
        status |= vsi_nn_sp_mul(&sp_insts_param[2], VSI_NN_SP_PWLMUL, VSI_NN_SP_PWLMUL, VSI_NN_SP_SR9);
        status |= vsi_nn_sp_sub(&sp_insts_param[2], VSI_NN_SP_PWLADD, VSI_NN_SP_PWLADD, VSI_NN_SP_SR7);
        /* loop inst3: r4 = r5 * r5 | v12 = r8 + r10 | v11 = r5 */
        status |= vsi_nn_sp_mul(&sp_insts_param[3], VSI_NN_SP_SR5, VSI_NN_SP_SR5, VSI_NN_SP_SR4);
        status |= vsi_nn_sp_add(&sp_insts_param[3], VSI_NN_SP_SR8, VSI_NN_SP_SR10, VSI_NN_SP_VR12);
        status |= vsi_nn_sp_move(&sp_insts_param[3], VSI_NN_SP_SR5, VSI_NN_SP_VR11);
        /* loop inst4: r1 = setup(r6) | r8 = r9 * r7 */
        status |= vsi_nn_sp_pwl_setup0(&sp_insts_param[4], VSI_NN_SP_SR6, VSI_NN_SP_SR1);
        status |= vsi_nn_sp_mul(&sp_insts_param[4], VSI_NN_SP_SR9, VSI_NN_SP_SR7, VSI_NN_SP_SR8);
        CHECK_STATUS_FAIL_GOTO(status, final );

        attr.ignored_leading_v11_wr = 0;
        attr.ignored_leading_v12_wr = 4;
        attr.flush_cycle_num = 22;

        attr.num_of_v11_rd_in_flush_cycle = 0;
        attr.num_of_v12_rd_in_flush_cycle = 0;
        attr.num_of_v11_wr_in_flush_cycle = 1;
        attr.num_of_v12_wr_in_flush_cycle = 5;
    }
    else
    {
        /* loop inst0: r5 = v11 * r3 | r6 = r5 - r10 */
        status  = vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_VR11, VSI_NN_SP_SR3, VSI_NN_SP_SR5);
        status |= vsi_nn_sp_sub(&sp_insts_param[0], VSI_NN_SP_SR5, VSI_NN_SP_SR10, VSI_NN_SP_SR6);
        /* loop inst1: r8 = pwlMul() | r7 = pwlAdd() */
        status |= vsi_nn_sp_mul(&sp_insts_param[1], VSI_NN_SP_PWLMUL, VSI_NN_SP_PWLMUL, VSI_NN_SP_SR8);
        status |= vsi_nn_sp_sub(&sp_insts_param[1], VSI_NN_SP_PWLADD, VSI_NN_SP_PWLADD, VSI_NN_SP_SR7);
        /* loop inst2: r5 = v12 * r3 */
        status |= vsi_nn_sp_mul(&sp_insts_param[2], VSI_NN_SP_VR12, VSI_NN_SP_SR3, VSI_NN_SP_SR5);
        /* loop inst3: r10 = r5 * r5 | v11 = r5 */
        status |= vsi_nn_sp_mul(&sp_insts_param[3], VSI_NN_SP_SR5, VSI_NN_SP_SR5, VSI_NN_SP_SR10);
        status |= vsi_nn_sp_move(&sp_insts_param[3], VSI_NN_SP_SR5, VSI_NN_SP_VR11);
        /* loop inst4: r1 = setup(r6) | r7 = r7 * r8 */
        status |= vsi_nn_sp_pwl_setup0(&sp_insts_param[4], VSI_NN_SP_SR6, VSI_NN_SP_SR1);
        status |= vsi_nn_sp_mul(&sp_insts_param[4], VSI_NN_SP_SR7, VSI_NN_SP_SR8, VSI_NN_SP_SR7);
        /* loop inst5: v12 = r9 * r4 */
        status |= vsi_nn_sp_mul(&sp_insts_param[5], VSI_NN_SP_SR9, VSI_NN_SP_SR4, VSI_NN_SP_VR12);
        CHECK_STATUS_FAIL_GOTO(status, final );

        attr.ignored_leading_v11_wr = 0;
        attr.ignored_leading_v12_wr = 3;
        attr.flush_cycle_num = 21;

        attr.num_of_v11_rd_in_flush_cycle = 0;
        attr.num_of_v12_rd_in_flush_cycle = 0;
        attr.num_of_v11_wr_in_flush_cycle = 1;
        attr.num_of_v12_wr_in_flush_cycle = 4;
    }

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_YZMERGE;

    attr.input_setup = VSI_NN_SP_INPUT_SETUP_V12;
    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.ignored_leading_outputs = 0;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_YZ;
    attr.split_max_vector_depth = max_vector_depth;
    attr.ch0_post_redistribute = VSI_NN_SP_CH_POST_REDISTRIBUTE_VECTOR_GATHER;
    attr.ch1_post_redistribute = VSI_NN_SP_CH_POST_REDISTRIBUTE_VECTOR_GATHER;

    VSI_NN_SP_ATTR_SET_CONST_TO_SR3(attr, inv_m);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR4(attr, output_scale);

    spinst = vsi_nn_create_spinst(graph);
    CHECK_PTR_FAIL_GOTO( spinst, "Create spInst fail.", final );
    status  = vsi_nn_add_spinst_insts(spinst, sp_insts_param, spInstsNum);
    status |= vsi_nn_set_spinst_attr(spinst, attr);
    CHECK_STATUS_FAIL_GOTO(status, final );

    inputs_tensor[0] = input->t;
    outputs_tensor[0] = output->t;

    vx_lut_params.lut_function = VX_NN_ACTIVATION_RSQRT;
    vx_lut_params.float_values[0] = 0;
    vx_lut_params.float_values[1] = eps;
    vx_lut_params.fvalues_count = 2;

    node = vxStreamProcessorNode(
        graph->g,
        inputs_tensor,
        input_count,
        outputs_tensor,
        output_count,
        spinst->sp,
        &vx_lut_params);

    status = vsi_nn_set_sp_kernel_name(node, kernel_name);
    CHECK_STATUS_FAIL_GOTO(status, final );

final:
    if (spinst)
    {
        vsi_nn_release_spinst(&spinst);
    }

    return (vsi_nn_kernel_node_t)node;
}

vsi_nn_kernel_node_t vsi_nn_sp_layer_norm_scale_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input,
        vsi_nn_tensor_t             * alpha,
        vsi_nn_tensor_t             * dummy_tensor,
        vsi_nn_tensor_t             * output,
        char                        * kernel_name
    )
{
    const int32_t spLoopInstsNum = 2;
    const int32_t spInstsNum = spLoopInstsNum;

    const uint32_t input_count = 3;
    const uint32_t output_count = 1;
    vx_tensor inputs_tensor[3] = {NULL};
    vx_tensor outputs_tensor[1] = {NULL};
    vx_node node = NULL;
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth /
        graph->ctx->config.sp_exec_count;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[2];
    vsi_nn_spinst_attr_t attr;

    vsi_status status = VSI_FAILURE;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    /* loop inst0: out = r1 * r2 | r1 = in - v11 */
    status  = vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_SR1, VSI_NN_SP_SR2, VSI_NN_SP_SROUT);
    status |= vsi_nn_sp_sub(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_VR11, VSI_NN_SP_SR1);
    /* loop inst1: r2 = r1 * r2 */
    status |= vsi_nn_sp_mul(&sp_insts_param[1], VSI_NN_SP_SRIN, VSI_NN_SP_VR12, VSI_NN_SP_SR2);

    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_YZMERGE;

    attr.input_setup = VSI_NN_SP_INPUT_SETUP_INTERLEAVE_TWO_INPUT;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.ignored_leading_outputs = 2;
    attr.ignored_leading_v11_rd = 0;
    attr.ignored_leading_v12_rd = 0;
    attr.flush_cycle_num = 3;
    attr.v11_push_pop_config = VSI_NN_SP_PUSH_POP_EVERY_ROW;
    attr.v12_push_pop_config = VSI_NN_SP_PUSH_POP_EVERY_ROW;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_YZ;
    attr.split_max_vector_depth = max_vector_depth;

    spinst = vsi_nn_create_spinst(graph);
    CHECK_PTR_FAIL_GOTO( spinst, "Create spInst fail.", final );
    status  = vsi_nn_add_spinst_insts(spinst, sp_insts_param, spInstsNum);
    status |= vsi_nn_set_spinst_attr(spinst, attr);
    CHECK_STATUS_FAIL_GOTO(status, final );

    inputs_tensor[0] = input->t;
    inputs_tensor[1] = alpha->t;
    inputs_tensor[2] = dummy_tensor->t;
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

vsi_nn_kernel_node_t vsi_nn_sp_in_minus_v11_times_v12_node
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
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth /
        graph->ctx->config.sp_exec_count;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[1];
    vsi_nn_spinst_attr_t attr;

    vsi_status status = VSI_FAILURE;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    /* loop inst0: out = r1 * r2 | r1 = in - v11 | r2 = v12 */
    status  = vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_SR1, VSI_NN_SP_SR2, VSI_NN_SP_SROUT);
    status |= vsi_nn_sp_sub(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_VR11, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_move(&sp_insts_param[0], VSI_NN_SP_VR12, VSI_NN_SP_SR2);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_YZMERGE;

    attr.input_setup = VSI_NN_SP_INPUT_SETUP_SINGLE_INPUT;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.ignored_leading_outputs = 3;
    attr.flush_cycle_num = 3;
    attr.v11_push_pop_config = VSI_NN_SP_PUSH_POP_EVERY_ROW;
    attr.v12_push_pop_config = VSI_NN_SP_PUSH_POP_EVERY_ROW;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_YZ;
    attr.split_max_vector_depth = max_vector_depth;

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

vsi_nn_spinst_t * vsi_nn_sp_times_v11_plus_v12_inst
    (
        vx_context                context,
        int32_t                   fifo_depth,
        int32_t                   max_vector_depth
    )
{
    vsi_status status = VSI_FAILURE;
    const int32_t spInitInstsNum = fifo_depth == 1 ? 3 : 0;
    const int32_t spLoopInstsNum = fifo_depth == 1 ? 1 : ( fifo_depth > 2 ? 2 : 3 );
    const int32_t spInstsNum = spInitInstsNum + spLoopInstsNum;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[8];
    vsi_nn_spinst_attr_t attr;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    if (fifo_depth == 1)
    {
        /* init inst0: r1 = v11 */
        status = vsi_nn_sp_move(&sp_insts_param[0], VSI_NN_SP_VR11, VSI_NN_SP_SR1);
        /* init inst1: r2 = v12 */
        status |= vsi_nn_sp_move(&sp_insts_param[1], VSI_NN_SP_VR12, VSI_NN_SP_SR2);
        /* init inst2: nop */
        status |= vsi_nn_sp_nop(&sp_insts_param[2]);
        /* loop inst0: r3 = in * r1 | out = r2 + r3 */
        status |= vsi_nn_sp_mul(&sp_insts_param[3], VSI_NN_SP_SRIN, VSI_NN_SP_SR1, VSI_NN_SP_SR3);
        status |= vsi_nn_sp_add(&sp_insts_param[3], VSI_NN_SP_SR2, VSI_NN_SP_SR3, VSI_NN_SP_SROUT);
        CHECK_STATUS_FAIL_GOTO(status, final );

        attr.flush_cycle_num = 3;
        attr.ignored_leading_outputs = 3;
    }
    else if (fifo_depth == 2)
    {
        /* loop inst0: r1 = in * v11 | v11 = v11 */
        status  = vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_VR11, VSI_NN_SP_SR1);
        status |= vsi_nn_sp_move(&sp_insts_param[0], VSI_NN_SP_VR11, VSI_NN_SP_VR11);
        /* loop inst1: r2 = v12 + r4 | v12 = v12 */
        status |= vsi_nn_sp_add(&sp_insts_param[1], VSI_NN_SP_VR12, VSI_NN_SP_SR4, VSI_NN_SP_SR2);
        status |= vsi_nn_sp_move(&sp_insts_param[1], VSI_NN_SP_VR12, VSI_NN_SP_VR12);
        /* loop inst2: out = r1 + r2 */
        status |= vsi_nn_sp_add(&sp_insts_param[2], VSI_NN_SP_SR1, VSI_NN_SP_SR2, VSI_NN_SP_SROUT);
        CHECK_STATUS_FAIL_GOTO(status, final );

        attr.flush_cycle_num = 5;
        attr.ignored_leading_outputs = 1;

        attr.ignored_leading_v11_rd = 0;
        attr.ignored_leading_v12_rd = 0;
        attr.ignored_leading_v11_wr = 0;
        attr.ignored_leading_v12_wr = 0;

        attr.num_of_v11_rd_in_flush_cycle = 0;
        attr.num_of_v12_rd_in_flush_cycle = 1;
        attr.num_of_v11_wr_in_flush_cycle = 0;
        attr.num_of_v12_wr_in_flush_cycle = 1;
    }
    else
    {
         /* loop inst0: r1 = in * v11 | v11 = v11 */
        status  = vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_VR11, VSI_NN_SP_SR1);
        status |= vsi_nn_sp_move(&sp_insts_param[0], VSI_NN_SP_VR11, VSI_NN_SP_VR11);
        /* loop inst2: out = r1 + v12 | v12 = v12 */
        status |= vsi_nn_sp_add(&sp_insts_param[1], VSI_NN_SP_SR1, VSI_NN_SP_VR12, VSI_NN_SP_SROUT);
        status |= vsi_nn_sp_move(&sp_insts_param[1], VSI_NN_SP_VR12, VSI_NN_SP_VR12);
        CHECK_STATUS_FAIL_GOTO(status, final );

        attr.flush_cycle_num = 3;
        attr.ignored_leading_outputs = 1;

        attr.ignored_leading_v11_rd = 0;
        attr.ignored_leading_v12_rd = 1;
        attr.ignored_leading_v11_wr = 0;
        attr.ignored_leading_v12_wr = 1;

        attr.num_of_v11_rd_in_flush_cycle = 0;
        attr.num_of_v12_rd_in_flush_cycle = 2;
        attr.num_of_v11_wr_in_flush_cycle = 0;
        attr.num_of_v12_wr_in_flush_cycle = 2;
    }

    VSI_NN_SP_ATTR_SET_CONST_TO_SR4(attr, 0);

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_YZMERGE;
    attr.input_setup = VSI_NN_SP_INPUT_SETUP_SINGLE_INPUT;

    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_X;
    attr.split_tilex_equal_imgx = TRUE;
    attr.split_max_vector_depth = max_vector_depth;

    spinst = vsi_nn_create_spinst_by_context(context);
    CHECK_PTR_FAIL_GOTO( spinst, "Create spInst fail.", final );
    status  = vsi_nn_add_spinst_insts(spinst, sp_insts_param, spInstsNum);
    status |= vsi_nn_set_spinst_attr(spinst, attr);
    CHECK_STATUS_FAIL_GOTO(status, final );

final:
    return spinst;
}

DEF_SP_KERNEL_QUERY(times_v11_plus_v12_query)
    (
    vsi_nn_kernel_node_t        node
    )
{
    vsi_status status = VSI_FAILURE;
    vx_size index = 0;
    vx_size tile_size[2] = {0};
    vsi_nn_spinst_t *spinst = NULL;
    int32_t fifo_depth = 0;
    int32_t max_vector_depth = 0;
    vsi_nn_spinst_t pre_spinst;
    vx_context  ctx = vxGetContext((vx_reference)node);
    vx_hardware_caps_params_ext2_t hw_param;

    memset(&pre_spinst, 0, sizeof(pre_spinst));
    memset(&hw_param, 0, sizeof(vx_hardware_caps_params_ext2_t));
    status = vxQueryHardwareCaps(ctx, (vx_hardware_caps_params_t*)(&hw_param), sizeof(vx_hardware_caps_params_ext2_t));
    CHECK_STATUS_FAIL_GOTO( status, final );

    status = vxQueryNode(node, VX_NODE_SWTILING_TILE_XY, tile_size, sizeof(tile_size));
    CHECK_STATUS_FAIL_GOTO( status, final );
    status = vxQueryNode(node, VX_NODE_SPINST_INDEX, &index, sizeof(index));
    CHECK_STATUS_FAIL_GOTO( status, final );
    status = vxQueryNode(node, VX_NODE_SPINST, &pre_spinst.sp, sizeof(pre_spinst.sp));
    CHECK_STATUS_FAIL_GOTO( status, final );

    fifo_depth = (int32_t)ceil((float)(tile_size[0] * tile_size[1]) / (float)hw_param.streamProcessorExecCount);
    max_vector_depth = hw_param.streamProcessorVectorSize;

    spinst = vsi_nn_sp_times_v11_plus_v12_inst(ctx, fifo_depth, max_vector_depth);

    status = vxSetParameterByIndex( node, (uint32_t)index, (vx_reference)spinst->sp );
    CHECK_STATUS_FAIL_GOTO( status, final );

final:
    if (spinst)
    {
        vsi_nn_release_spinst(&spinst);
    }

    vsi_nn_release_vxspinst(&pre_spinst);

    return status;
}

static vsi_nn_kernel_node_t vsi_nn_sp_in_times_v11_plus_v12_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input0,
        vsi_nn_tensor_t             * input1,
        vsi_nn_tensor_t             * output,
        char                        * kernel_name
    )
{
    const uint32_t input_count = 2;
    const uint32_t output_count = 1;
    vx_tensor inputs_tensor[2] = {NULL};
    vx_tensor outputs_tensor[1] = {NULL};
    vx_node node = NULL;
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth;
    int32_t fifo_depth = 4;
    vsi_nn_spinst_t *spinst = NULL;
    vsi_status status = VSI_FAILURE;

    spinst = vsi_nn_sp_times_v11_plus_v12_inst(graph->ctx->c, fifo_depth, max_vector_depth);

    inputs_tensor[0] = input0->t;
    inputs_tensor[1] = input1->t;
    outputs_tensor[0] = output->t;
    node = vxStreamProcessorNode(
        graph->g,
        inputs_tensor,
        input_count,
        outputs_tensor,
        output_count,
        spinst->sp,
        NULL);

    if (node)
    {
        vxAssignNodeQueryCallback(node, times_v11_plus_v12_query);
    }

    status = vsi_nn_set_sp_kernel_name(node, kernel_name);
    CHECK_STATUS_FAIL_GOTO(status, final );

final:
    if (spinst)
    {
        vsi_nn_release_spinst(&spinst);
    }

    return (vsi_nn_kernel_node_t)node;
}

vsi_nn_kernel_node_t vsi_nn_sp_a_to_v11_b_to_v12_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input0,
        vsi_nn_tensor_t             * input1,
        vsi_nn_tensor_t             * output,
        float                         output_scale,
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
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth;

    vsi_status status = VSI_FAILURE;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    /* loop inst0: v11 = in * r4 */
    status  = vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_SR4, VSI_NN_SP_VR11);
    /* loop inst1: v12 = in * r4 */
    status |= vsi_nn_sp_mul(&sp_insts_param[1], VSI_NN_SP_SRIN, VSI_NN_SP_SR4, VSI_NN_SP_VR12);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_YZMERGE;
    attr.input_setup = VSI_NN_SP_INPUT_SETUP_INTERLEAVE_TWO_INPUT;

    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.ignored_leading_outputs = 0;
    attr.flush_cycle_num = 0;
    attr.v11_reset_at_start = VSI_NN_SP_V_RESET_AT_START_RESET;
    attr.v12_reset_at_start = VSI_NN_SP_V_RESET_AT_START_RESET;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_X;
    attr.split_tilex_equal_imgx = TRUE;
    attr.split_max_vector_depth = max_vector_depth;

    VSI_NN_SP_ATTR_SET_CONST_TO_SR4(attr, output_scale);

    spinst = vsi_nn_create_spinst(graph);
    CHECK_PTR_FAIL_GOTO( spinst, "Create spInst fail.", final );
    status  = vsi_nn_add_spinst_insts(spinst, sp_insts_param, spInstsNum);
    status |= vsi_nn_set_spinst_attr(spinst, attr);
    CHECK_STATUS_FAIL_GOTO(status, final );

    inputs_tensor[0] = input0->t;
    inputs_tensor[1] = input1->t;
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

vsi_bool _check_const_tensor_value
    (
        vsi_nn_graph_t            * graph,
        vsi_nn_tensor_t           * input,
        vx_bool                   * is_all_one,
        vx_bool                   * is_all_zero
    )
{
    vsi_size_t j = 0;
    vsi_size_t elements = 1;
    vx_bool is_all_one_value = FALSE;
    vx_bool is_all_zero_value = FALSE;
    float* data   = NULL;

    if (NULL == input || NULL == graph)
    {
        return FALSE;
    }

    elements = vsi_nn_GetElementNum(input);

    data = vsi_nn_ConvertTensorToFloat32Data(graph, input);
    CHECK_PTR_FAIL_GOTO( data, "Create input buffer fail.", final );

    for (j = 0; j < elements; j++)
    {
        if (is_all_one && data[j] != 1)
        {
            is_all_one_value = FALSE;
            break;
        }
        else if (is_all_zero && data[j] != 0)
        {
            is_all_zero_value = FALSE;
            break;
        }
    }

    if (is_all_one)
    {
        *is_all_one = is_all_one_value;
    }
    else if (is_all_zero)
    {
        *is_all_zero = is_all_zero_value;
    }

final:
    vsi_nn_safe_free(data);

    return TRUE;
}

vsi_nn_kernel_node_t layer_norm_x_direction
    (
    vsi_nn_graph_t              * graph,
    vsi_nn_tensor_t            ** inputs,
    vsi_nn_tensor_t            ** outputs,
    const vsi_nn_kernel_param_t * params
    )
{
    vsi_nn_kernel_node_t rt_node = NULL;
    vsi_nn_kernel_node_t node[4] = {NULL};
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_t * dummy_tensor[3] = {NULL};
    vsi_nn_tensor_t * output_tensor[2] = {NULL};
    vx_bool is_one_value = FALSE;
    vx_bool is_zero_value = FALSE;
    uint32_t i = 0;
    float output_scale = 1.0f / vsi_nn_get_tensor_scale(outputs[0]);
    float eps = vsi_nn_kernel_param_get_float32( params, "eps" );
    float inv_m = 1.0f / (float)(outputs[0]->attr.size[0]);

    memset( &attr, 0, sizeof(attr) );
    memcpy( attr.size, outputs[0]->attr.size, sizeof(attr.size) );
    attr.dim_num = outputs[0]->attr.dim_num;
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    attr.is_const = FALSE;
    attr.vtl = TRUE;
    attr.size[0] = 1;
    dummy_tensor[0] = vsi_nn_create_dummy_tensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[0], "Create dummy_tensor fail.", final );
    dummy_tensor[1] = vsi_nn_create_dummy_tensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[1], "Create dummy_tensor fail.", final );

    memcpy( attr.size, outputs[0]->attr.size, sizeof(attr.size) );
    for ( i = 1; i < outputs[0]->attr.dim_num; i++ )
    {
        attr.size[i] = 1;
    }
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    attr.is_const = TRUE;
    attr.vtl = FALSE;
    dummy_tensor[2] = vsi_nn_create_dummy_tensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[2], "Create dummy_tensor fail.", final );

    _check_const_tensor_value(graph, inputs[2], &is_one_value, NULL);
    _check_const_tensor_value(graph, inputs[1], NULL, &is_zero_value);

    if (!is_one_value || !is_zero_value)
    {
        output_scale = 1;
    }

    memcpy( attr.size, outputs[0]->attr.size, sizeof(attr.size) );
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    attr.is_const = FALSE;
    attr.vtl = TRUE;
    output_tensor[0] = vsi_nn_CreateTensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( output_tensor[0], "Create tensor fail.", final );
    output_tensor[1] = vsi_nn_CreateTensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( output_tensor[1], "Create tensor fail.", final );

    node[0] = vsi_nn_sp_sums_axis0_node(graph, inputs[0], output_tensor[0], dummy_tensor[0], "layernorm_0");
    CHECK_PTR_FAIL_GOTO( node[0], "Create sp_sums_axis0 fail.", final );

    node[1] = vsi_nn_sp_layer_norm_means_node(graph, dummy_tensor[0], dummy_tensor[1],
                                            inv_m, eps, output_scale, "layernorm_1");
    CHECK_PTR_FAIL_GOTO( node[1], "Create sp_sums_norm_means  fail.", final );

    if (is_one_value && is_zero_value)
    {
        rt_node = vsi_nn_sp_in_minus_v11_times_v12_node(graph, output_tensor[0],
            dummy_tensor[1], outputs[0], "layernorm_2");
        CHECK_PTR_FAIL_GOTO( rt_node, "Create in_times_v11_plus_v12 fail.", final );
    }
    else
    {
        output_scale = 1.0f / vsi_nn_get_tensor_scale(outputs[0]);

        node[2] = vsi_nn_sp_in_minus_v11_times_v12_node(graph, output_tensor[0],
            dummy_tensor[1], output_tensor[1], "layernorm_2");
        CHECK_PTR_FAIL_GOTO( node[2], "Create in_times_v11_plus_v12 fail.", final );

        node[3] = vsi_nn_sp_a_to_v11_b_to_v12_node(graph, inputs[2], inputs[1],
            dummy_tensor[2], output_scale, "layernorm_3");
        CHECK_PTR_FAIL_GOTO( node[3], "Create a_times_b_to_v11  fail.", final );
        rt_node = vsi_nn_sp_in_times_v11_plus_v12_node(graph, output_tensor[1], dummy_tensor[2],
            outputs[0], "layernorm_4");
        CHECK_PTR_FAIL_GOTO( rt_node, "Create a_plus_v11  fail.", final );
    }

final:
    vsi_safe_release_node(node[0]);
    vsi_safe_release_node(node[1]);
    vsi_safe_release_node(node[2]);
    vsi_safe_release_node(node[3]);
    vsi_safe_release_tensor(dummy_tensor[0]);
    vsi_safe_release_tensor(dummy_tensor[1]);
    vsi_safe_release_tensor(dummy_tensor[2]);
    vsi_safe_release_tensor(output_tensor[0]);
    vsi_safe_release_tensor(output_tensor[1]);

    return rt_node;
} /* layer_norm_x_direction() */

#define REGISTER_LAYER_NORM_STREAM_PROCESSOR_KERNEL( kernel_name )   \
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

REGISTER_LAYER_NORM_STREAM_PROCESSOR_KERNEL( layer_norm )
{
    vsi_nn_kernel_node_t node = NULL;
    int32_t axis = vsi_nn_kernel_param_get_int32( params, "axis" );

    VSI_UNREFERENCED(input_num);
    VSI_UNREFERENCED(output_num);
    VSI_UNREFERENCED(kernel);

    if (axis == 0)
    {
        node = layer_norm_x_direction(graph, inputs, outputs, params);
    }
    else if ( axis == 1 &&
             (inputs[0]->attr.dim_num < 3 || inputs[0]->attr.size[2] == 1) )
    {
        node = layer_norm_y_direction(graph, inputs, outputs, params);
    }
    else if (axis == 2)
    {
        node = layer_norm_z_direction(graph, inputs, outputs, params);
    }

    return node;
} /* layer_norm() */

#undef REGISTER_LAYER_NORM_STREAM_PROCESSOR_KERNEL

#endif
