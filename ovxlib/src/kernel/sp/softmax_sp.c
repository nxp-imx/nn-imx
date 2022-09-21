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

#if (VX_STREAM_PROCESSOR_SUPPORT)

vsi_nn_kernel_node_t vsi_nn_sp_softmax_max_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input,
        vsi_nn_tensor_t             * output0,
        vsi_nn_tensor_t             * output1
    )
{
    const int32_t spInitInstsNum = 0;
    const int32_t spLoopInstsNum = 1;
    const int32_t spInstsNum = spInitInstsNum + spLoopInstsNum;

    const uint32_t input_count = 1;
    const uint32_t output_count = 2;
    vx_tensor inputs_tensor[1] = {NULL};
    vx_tensor outputs_tensor[2] = {NULL};
    vx_node node = NULL;
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth /
        graph->ctx->config.sp_exec_count;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[1];
    vsi_nn_spinst_attr_t attr;
    float input_scale = vsi_nn_get_tensor_scale(input);
    float scale0 = input_scale;
    float clamp_min = 0;
    float clamp_max = 0;

    vsi_status status = VSI_FAILURE;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    vsi_nn_get_tensor_clamp_min_max(input, &clamp_min, &clamp_max);
    clamp_min = clamp_min * scale0;
    clamp_max = clamp_max * scale0;

    /* loop inst0: r1 = clamp(r3 * in, r6, r7) | out = r1 */
    status  = vsi_nn_sp_mul_clamp(&sp_insts_param[0], VSI_NN_SP_SR3, VSI_NN_SP_SRIN, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_move(&sp_insts_param[0], VSI_NN_SP_SR1, VSI_NN_SP_SROUT);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_YZMERGE;
    attr.input_setup = VSI_NN_SP_INPUT_SETUP_SINGLE_INPUT;

    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.ignored_leading_outputs = 3;
    attr.flush_cycle_num = 3;
    attr.accelerator_input_select = VSI_NN_SP_ACCELERATOR_IN_FROM_OUTPUT;
    attr.sum_engine_reset = VSI_NN_SP_SUM_ENGINE_RESET_START_FROM_MINIMUM;
    attr.sum_engine_control = VSI_NN_SP_ACCUM_2D;
    attr.sum_engine_num_ch_minus_one = VSI_NN_SP_SUM_ENGINE_NUM_CH_ONE_CH;
    attr.sum_engine_2d_accum_storeage = VSI_NN_SP_ACCM_STOREAGE_DIFFERENT;
    attr.sum_engine_op_select = VSI_NN_SP_MAX_OP;
    attr.v11_reset_at_start = VSI_NN_SP_V_RESET_AT_START_RESET;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_YZ;
    attr.split_max_vector_depth = max_vector_depth;

    VSI_NN_SP_ATTR_SET_CONST_TO_SR3(attr, input_scale);
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

final:
    if (spinst)
    {
        vsi_nn_release_spinst(&spinst);
    }

    return (vsi_nn_kernel_node_t)node;
}

vsi_nn_kernel_node_t vsi_nn_sp_softmax_move_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input,
        vsi_nn_tensor_t             * output
    )
{
    const int32_t spInitInstsNum = 0;
    const int32_t spLoopInstsNum = 1;
    const int32_t spInstsNum = spInitInstsNum + spLoopInstsNum;

    const uint32_t input_count = 1;
    const uint32_t output_count = 1;
    vx_tensor inputs_tensor[1] = {NULL};
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

    /* loop inst0: v12 = v11 */
    status = vsi_nn_sp_move(&sp_insts_param[0], VSI_NN_SP_VR11, VSI_NN_SP_VR12);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_YZMERGE;
    attr.input_setup = VSI_NN_SP_INPUT_SETUP_V11;

    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.ignored_leading_outputs = 0;
    attr.flush_cycle_num = 0;
    attr.ch0_post_redistribute = VSI_NN_SP_CH_POST_REDISTRIBUTE_VECTOR_GATHER;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_YZ;
    attr.split_max_vector_depth = max_vector_depth;

    spinst = vsi_nn_create_spinst(graph);
    CHECK_PTR_FAIL_GOTO( spinst, "Create spInst fail.", final );
    status  = vsi_nn_add_spinst_insts(spinst, sp_insts_param, spInstsNum);
    status |= vsi_nn_set_spinst_attr(spinst, attr);
    CHECK_STATUS_FAIL_GOTO(status, final );

    inputs_tensor[0] = input->t;
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

vsi_nn_kernel_node_t vsi_nn_sp_softmax_exp_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input0,
        vsi_nn_tensor_t             * input1,
        vsi_nn_tensor_t             * output0,
        vsi_nn_tensor_t             * output1,
        float                         beta
    )
{
    const int32_t spLoopInstsNum = 3;
    const int32_t spInstsNum = spLoopInstsNum;

    const uint32_t input_count = 2;
    const uint32_t output_count = 2;
    vx_tensor inputs_tensor[2] = {NULL};
    vx_tensor outputs_tensor[2] = {NULL};
    vx_node node = NULL;
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth /
        graph->ctx->config.sp_exec_count;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[3];
    vsi_nn_spinst_attr_t attr;

    vsi_nn_sp_lut_params sp_lut_params;
    vx_lut_params_s vx_lut_params;

    vsi_status status = VSI_FAILURE;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);
    memset(&sp_lut_params, 0, sizeof(vsi_nn_sp_lut_params));
    memset(&vx_lut_params, 0, sizeof(vx_lut_params_s));

    /* loop inst0: r8 = in - v12 | r6 = r1 */
    status  = vsi_nn_sp_sub(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_VR12, VSI_NN_SP_SR8);
    status |= vsi_nn_sp_move(&sp_insts_param[0], VSI_NN_SP_SR1, VSI_NN_SP_SR6);
    /* loop inst1: r1 = pwlSetup(r8) | r5 = pwlMult() | r3 = pwlAdd() | r10 = r6 */
    status |= vsi_nn_sp_pwl_setup0(&sp_insts_param[1], VSI_NN_SP_SR8, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_mul(&sp_insts_param[1], VSI_NN_SP_PWLMUL, VSI_NN_SP_PWLMUL, VSI_NN_SP_SR5);
    status |= vsi_nn_sp_sub(&sp_insts_param[1], VSI_NN_SP_PWLADD, VSI_NN_SP_PWLADD, VSI_NN_SP_SR3);
    status |= vsi_nn_sp_move(&sp_insts_param[1], VSI_NN_SP_SR6, VSI_NN_SP_SR10);
    /* loop inst2: r9 = r5 * r3 | out = r9 + r10 */
    status |= vsi_nn_sp_mul(&sp_insts_param[2], VSI_NN_SP_SR5, VSI_NN_SP_SR3, VSI_NN_SP_SR9);
    status |= vsi_nn_sp_add(&sp_insts_param[2], VSI_NN_SP_SR9, VSI_NN_SP_SR10, VSI_NN_SP_SROUT);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_YZMERGE;
    attr.input_setup = VSI_NN_SP_INPUT_SETUP_SINGLE_INPUT;

    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.ignored_leading_outputs = 4;
    attr.flush_cycle_num = 14;
    attr.accelerator_input_select = VSI_NN_SP_ACCELERATOR_IN_FROM_OUTPUT;
    attr.sum_engine_reset = VSI_NN_SP_SUM_ENGINE_RESET_START_FROM_ZERO;
    attr.sum_engine_control = VSI_NN_SP_ACCUM_2D;
    attr.sum_engine_num_ch_minus_one = VSI_NN_SP_SUM_ENGINE_NUM_CH_ONE_CH;
    attr.sum_engine_2d_accum_storeage = VSI_NN_SP_ACCM_STOREAGE_DIFFERENT;
    attr.v11_reset_at_start = VSI_NN_SP_V_RESET_AT_START_RESET;
    attr.v12_push_pop_config = VSI_NN_SP_PUSH_POP_EVERY_ROW;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_YZ;
    attr.split_max_vector_depth = max_vector_depth;

    spinst = vsi_nn_create_spinst(graph);
    CHECK_PTR_FAIL_GOTO( spinst, "Create spInst fail.", final );
    status  = vsi_nn_add_spinst_insts(spinst, sp_insts_param, spInstsNum);
    status |= vsi_nn_set_spinst_attr(spinst, attr);
    CHECK_STATUS_FAIL_GOTO(status, final );

    inputs_tensor[0] = input0->t;
    inputs_tensor[1] = input1->t;
    outputs_tensor[0] = output0->t;
    outputs_tensor[1] = output1->t;

    vx_lut_params.lut_function = VX_NN_ACTIVATION_CUSTOM;
    vx_lut_params.in_lut = vxCreateLUT( graph->ctx->c, VX_TYPE_FLOAT32, VSI_NN_SP_LUT_MAX_SIZE);
    vx_lut_params.out_lut = vxCreateLUT( graph->ctx->c, VX_TYPE_FLOAT32, VSI_NN_SP_LUT_MAX_SIZE);

    sp_lut_params.act_type = VSI_NN_SP_ACT_LINEAR_EXP;
    sp_lut_params.params[0] = beta;
    sp_lut_params.params[1] = 0;
    vsi_nn_sp_lut(vx_lut_params.in_lut, vx_lut_params.out_lut, &sp_lut_params);

    node = vxStreamProcessorNode(
        graph->g,
        inputs_tensor,
        input_count,
        outputs_tensor,
        output_count,
        spinst->sp,
        &vx_lut_params);

final:
    if (spinst)
    {
        vsi_nn_release_spinst(&spinst);
    }

    if (vx_lut_params.in_lut)
    {
        vxReleaseLUT(&vx_lut_params.in_lut);
        vx_lut_params.in_lut = NULL;
    }
    if (vx_lut_params.out_lut)
    {
        vxReleaseLUT(&vx_lut_params.out_lut);
        vx_lut_params.out_lut = NULL;
    }

    return (vsi_nn_kernel_node_t)node;
}

vsi_nn_kernel_node_t vsi_nn_sp_softmax_rcp_node
    (
        vsi_nn_graph_t  * graph,
        vsi_nn_tensor_t * input,
        vsi_nn_tensor_t * output,
        float             output_scale
    )
{
    const int32_t spLoopInstsNum = output_scale == 1 ? 2 : 3;
    const int32_t spInstsNum = spLoopInstsNum;

    const uint32_t input_count = 1;
    const uint32_t output_count = 1;
    vx_tensor inputs_tensor[1] = {NULL};
    vx_tensor outputs_tensor[1] = {NULL};
    vx_node node = NULL;
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth /
        graph->ctx->config.sp_exec_count;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[3];
    vsi_nn_spinst_attr_t attr;

    vsi_nn_sp_lut_params sp_lut_params;
    vx_lut_params_s vx_lut_params;

    vsi_status status = VSI_FAILURE;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);
    memset(&sp_lut_params, 0, sizeof(vsi_nn_sp_lut_params));
    memset(&vx_lut_params, 0, sizeof(vx_lut_params_s));

    if (output_scale == 1.0f)
    {
        /* loop inst0: r1 = pwlSetup(v11) | r6 = r5 * r2 | v11 = r4 + r6 | r3 = r1 */
        status  = vsi_nn_sp_pwl_setup0(&sp_insts_param[0], VSI_NN_SP_VR11, VSI_NN_SP_SR1);
        status |= vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_SR5, VSI_NN_SP_SR2, VSI_NN_SP_SR6);
        status |= vsi_nn_sp_add(&sp_insts_param[0], VSI_NN_SP_SR4, VSI_NN_SP_SR6, VSI_NN_SP_VR11);
        status |= vsi_nn_sp_move(&sp_insts_param[0], VSI_NN_SP_SR1, VSI_NN_SP_SR3);
        /* loop inst1: r5 = pwlMul() | r2 = pwlAdd() | r4 = r3*/
        status |= vsi_nn_sp_mul(&sp_insts_param[1], VSI_NN_SP_PWLMUL, VSI_NN_SP_PWLMUL, VSI_NN_SP_SR5);
        status |= vsi_nn_sp_sub(&sp_insts_param[1], VSI_NN_SP_PWLADD, VSI_NN_SP_PWLADD, VSI_NN_SP_SR2);
        status |= vsi_nn_sp_move(&sp_insts_param[1], VSI_NN_SP_SR3, VSI_NN_SP_SR4);
        CHECK_STATUS_FAIL_GOTO(status, final );

        attr.ignored_leading_v11_wr = 5;
        attr.flush_cycle_num = 10;
    }
    else
    {
        /* loop inst0: r1 = pwlSetup(v11) | r5 = pwlMul() | r2 = pwlAdd() | r8 = r1 */
        status  = vsi_nn_sp_pwl_setup0(&sp_insts_param[0], VSI_NN_SP_VR11, VSI_NN_SP_SR1);
        status |= vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_PWLMUL, VSI_NN_SP_PWLMUL, VSI_NN_SP_SR5);
        status |= vsi_nn_sp_sub(&sp_insts_param[0], VSI_NN_SP_PWLADD, VSI_NN_SP_PWLADD, VSI_NN_SP_SR2);
        status |= vsi_nn_sp_move(&sp_insts_param[0], VSI_NN_SP_SR1, VSI_NN_SP_SR8);
        /* loop inst1: r6 = r5 * r2 | r7 = r4 + r6 | r4 = r8 */
        status |= vsi_nn_sp_mul(&sp_insts_param[1], VSI_NN_SP_SR5, VSI_NN_SP_SR2, VSI_NN_SP_SR6);
        status |= vsi_nn_sp_add(&sp_insts_param[1], VSI_NN_SP_SR4, VSI_NN_SP_SR6, VSI_NN_SP_SR7);
        status |= vsi_nn_sp_move(&sp_insts_param[1], VSI_NN_SP_SR8, VSI_NN_SP_SR4);
        /* loop inst2: v11 = r7 * r3 */
        status |= vsi_nn_sp_mul(&sp_insts_param[2], VSI_NN_SP_SR7, VSI_NN_SP_SR3, VSI_NN_SP_VR11);
        CHECK_STATUS_FAIL_GOTO(status, final );

        attr.ignored_leading_v11_wr = 4;
        attr.flush_cycle_num = 14;
    }

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_YZMERGE;
    attr.input_setup = VSI_NN_SP_INPUT_SETUP_V11;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.ch0_post_redistribute = VSI_NN_SP_CH_POST_REDISTRIBUTE_VECTOR_GATHER;

    attr.num_of_v11_wr_in_flush_cycle = 5;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_YZ;
    attr.split_max_vector_depth = max_vector_depth;

    VSI_NN_SP_ATTR_SET_CONST_TO_SR3(attr, 1.0f / output_scale);

    spinst = vsi_nn_create_spinst(graph);
    CHECK_PTR_FAIL_GOTO( spinst, "Create spInst fail.", final );
    status  = vsi_nn_add_spinst_insts(spinst, sp_insts_param, spInstsNum);
    status |= vsi_nn_set_spinst_attr(spinst, attr);
    CHECK_STATUS_FAIL_GOTO(status, final );

    inputs_tensor[0] = input->t;
    outputs_tensor[0] = output->t;

    vx_lut_params.lut_function = VX_NN_ACTIVATION_CUSTOM;
    vx_lut_params.in_lut = vxCreateLUT( graph->ctx->c, VX_TYPE_FLOAT32, VSI_NN_SP_LUT_MAX_SIZE);
    vx_lut_params.out_lut = vxCreateLUT( graph->ctx->c, VX_TYPE_FLOAT32, VSI_NN_SP_LUT_MAX_SIZE);

    sp_lut_params.act_type = VSI_NN_SP_ACT_RCP;
    vsi_nn_sp_lut(vx_lut_params.in_lut, vx_lut_params.out_lut, &sp_lut_params);

    node = vxStreamProcessorNode(
        graph->g,
        inputs_tensor,
        input_count,
        outputs_tensor,
        output_count,
        spinst->sp,
        &vx_lut_params);

final:
    if (spinst)
    {
        vsi_nn_release_spinst(&spinst);
    }

    if (vx_lut_params.in_lut)
    {
        vxReleaseLUT(&vx_lut_params.in_lut);
        vx_lut_params.in_lut = NULL;
    }
    if (vx_lut_params.out_lut)
    {
        vxReleaseLUT(&vx_lut_params.out_lut);
        vx_lut_params.out_lut = NULL;
    }

    return (vsi_nn_kernel_node_t)node;
}

vsi_nn_kernel_node_t vsi_nn_sp_softmax_times_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * exp_input,
        vsi_nn_tensor_t             * dummy_input,
        vsi_nn_tensor_t             * output
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

    /* loop inst0: r0 = r0 * v11 */
    status  = vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_VR11, VSI_NN_SP_SROUT);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_YZMERGE;

    attr.input_setup = VSI_NN_SP_INPUT_SETUP_SINGLE_INPUT;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.ignored_leading_v11_rd = 0;
    attr.flush_cycle_num = 0;
    attr.v11_push_pop_config = VSI_NN_SP_PUSH_POP_EVERY_ROW;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_YZ;
    attr.split_max_vector_depth = max_vector_depth;

    spinst = vsi_nn_create_spinst(graph);
    CHECK_PTR_FAIL_GOTO( spinst, "Create spInst fail.", final );
    status  = vsi_nn_add_spinst_insts(spinst, sp_insts_param, spInstsNum);
    status |= vsi_nn_set_spinst_attr(spinst, attr);
    CHECK_STATUS_FAIL_GOTO(status, final );

    inputs_tensor[0] = exp_input->t;
    inputs_tensor[1] = dummy_input->t;
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

vsi_nn_kernel_node_t softmax_x_direction
    (
    vsi_nn_graph_t              * graph,
    vsi_nn_tensor_t            ** inputs,
    vsi_nn_tensor_t            ** outputs,
    const vsi_nn_kernel_param_t * params
    )
{
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_t * dummy_tensor[4] = {NULL};
    vsi_nn_tensor_t * output_tensor[2] = {NULL};
    int32_t axis = 0;
    float output_scale = vsi_nn_get_tensor_scale(outputs[0]);
    float beta = vsi_nn_kernel_param_get_float32( params, "beta" );

    memcpy( &attr, &outputs[0]->attr, sizeof(vsi_nn_tensor_attr_t) );
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    attr.is_const = FALSE;
    attr.vtl = TRUE;
    attr.is_dummy = TRUE;
    attr.size[axis] = 1;
    dummy_tensor[0] = vsi_nn_CreateTensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[0], "Create dummy_tensor fail.", final );
    dummy_tensor[1] = vsi_nn_CreateTensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[1], "Create dummy_tensor fail.", final );
    dummy_tensor[2] = vsi_nn_CreateTensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[2], "Create dummy_tensor fail.", final );
    dummy_tensor[3] = vsi_nn_CreateTensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[3], "Create dummy_tensor fail.", final );

    memcpy( &attr, &outputs[0]->attr, sizeof(vsi_nn_tensor_attr_t) );
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    attr.is_const = FALSE;
    attr.vtl = TRUE;
    output_tensor[0] = vsi_nn_CreateTensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( output_tensor[0], "Create tensor fail.", final );
    output_tensor[1] = vsi_nn_CreateTensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( output_tensor[1], "Create tensor fail.", final );

    node = vsi_nn_sp_softmax_max_node(graph, inputs[0], output_tensor[0], dummy_tensor[0]);
    CHECK_PTR_FAIL_GOTO( node, "Create sp softmax node fail.", final );
    node = vsi_nn_sp_softmax_move_node(graph, dummy_tensor[0], dummy_tensor[1]);
    CHECK_PTR_FAIL_GOTO( node, "Create sp softmax node fail.", final );

    node = vsi_nn_sp_softmax_exp_node(graph, output_tensor[0], dummy_tensor[1],
                        output_tensor[1], dummy_tensor[2], beta);
    CHECK_PTR_FAIL_GOTO( node, "Create sp softmax node fail.", final );
    node = vsi_nn_sp_softmax_rcp_node(graph, dummy_tensor[2], dummy_tensor[3], output_scale);
    CHECK_PTR_FAIL_GOTO( node, "Create sp_softmax_rcp fail.", final );
    node = vsi_nn_sp_softmax_times_node(graph, output_tensor[1], dummy_tensor[3], outputs[0]);
    CHECK_PTR_FAIL_GOTO( node, "Create softmax_times fail.", final );

final:
    vsi_safe_release_tensor(dummy_tensor[0]);
    vsi_safe_release_tensor(dummy_tensor[1]);
    vsi_safe_release_tensor(dummy_tensor[2]);
    vsi_safe_release_tensor(dummy_tensor[3]);
    vsi_safe_release_tensor(output_tensor[0]);
    vsi_safe_release_tensor(output_tensor[1]);

    return node;
} /* softmax_x_direction() */

vsi_nn_kernel_node_t softmax_y_direction
    (
    vsi_nn_graph_t              * graph,
    vsi_nn_tensor_t            ** inputs,
    vsi_nn_tensor_t            ** outputs,
    const vsi_nn_kernel_param_t * params
    )
{
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_tensor_attr_t attr;
    vsi_size_t shape[VSI_NN_MAX_DIM_NUM] = { 0 };
    vsi_nn_tensor_t * trans_tensor[2] = {NULL};
    uint32_t perm[2][VSI_NN_MAX_DIM_NUM] = {{1, 0, 2, 3, 4, 5, 6, 7}, {1, 0, 2, 3, 4, 5, 6, 7}};
    uint32_t i = 0;

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

    node = vxTensorPermuteNode
        (
        graph->g,
        inputs[0]->t,
        trans_tensor[0]->t,
        perm[0],
        inputs[0]->attr.dim_num
        );
    CHECK_PTR_FAIL_GOTO( node, "Create vxTensorPermuteNode fail.", final );

    softmax_x_direction(graph, trans_tensor, &trans_tensor[1], params);

    node = vxTensorPermuteNode
        (
        graph->g,
        trans_tensor[1]->t,
        outputs[0]->t,
        perm[1],
        outputs[0]->attr.dim_num
        );
    CHECK_PTR_FAIL_GOTO( node, "Create vxTensorPermuteNode fail.", final );

final:
    vsi_safe_release_tensor(trans_tensor[0]);
    vsi_safe_release_tensor(trans_tensor[1]);

    return node;
} /* softmax_y_direction() */

vsi_nn_kernel_node_t softmax_z_direction
    (
    vsi_nn_graph_t              * graph,
    vsi_nn_tensor_t            ** inputs,
    vsi_nn_tensor_t            ** outputs,
    const vsi_nn_kernel_param_t * params
    );

#define REGISTER_SOFTMAX_STREAM_PROCESSOR_KERNEL( kernel_name )   \
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

REGISTER_SOFTMAX_STREAM_PROCESSOR_KERNEL( softmax )
{
    vsi_nn_kernel_node_t node = NULL;
    uint32_t axis = vsi_nn_kernel_param_get_int32( params, "axis" );

    if (axis == 0)
    {
        node = softmax_x_direction(graph, inputs, outputs, params);
    }
    else if (axis == 1)
    {
        node = softmax_y_direction(graph, inputs, outputs, params);
    }
    else if (axis == 2)
    {
        node = softmax_z_direction(graph, inputs, outputs, params);
    }

    return node;
} /* softmax() */

#undef REGISTER_SOFTMAX_STREAM_PROCESSOR_KERNEL

#endif
