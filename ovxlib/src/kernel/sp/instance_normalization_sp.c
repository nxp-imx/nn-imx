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

vsi_nn_kernel_node_t vsi_nn_sp_moments_sums_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input,
        vsi_nn_tensor_t             * output0,
        vsi_nn_tensor_t             * output1,
        char                        * kernel_name
    )
{
    const int32_t spInitInstsNum = 1;
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
    vsi_nn_spinst_inst_param sp_insts_param[3];
    vsi_nn_spinst_attr_t attr;

    vsi_status status = VSI_FAILURE;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    /* init inst0: r2 = 1 */
    status  = vsi_nn_sp_move_constant(&sp_insts_param[0], 1, VSI_NN_SP_SR2);
    /* loop inst0: acc0 = r1 * r1 || r1 = r0*/
    status  = vsi_nn_sp_mul(&sp_insts_param[1], VSI_NN_SP_SR1, VSI_NN_SP_SR1, VSI_NN_SP_ACC);
    status |= vsi_nn_sp_move(&sp_insts_param[1], VSI_NN_SP_SRIN, VSI_NN_SP_SR1);
    /* loop inst1: acc1 = r1 * r2 */
    status |= vsi_nn_sp_mul(&sp_insts_param[2], VSI_NN_SP_SR1, VSI_NN_SP_SR2, VSI_NN_SP_ACC);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;
    attr.input_setup = VSI_NN_SP_INPUT_SETUP_SINGLE_INPUT;
    attr.output_collapse_x = VSI_NN_SP_ATTR_OUTPUT_COLLAPSE_ENABLED;
    attr.output_collapse_y = VSI_NN_SP_ATTR_OUTPUT_COLLAPSE_ENABLED;

    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.ignored_leading_acc_out = 3;
    attr.flush_cycle_num = 4;
    attr.accelerator_input_select = VSI_NN_SP_ACCELERATOR_IN_FROM_ACCEL;
    attr.sum_engine_reset = VSI_NN_SP_SUM_ENGINE_RESET_START_FROM_ZERO;
    attr.sum_engine_control = VSI_NN_SP_ACCUM_2D;
    attr.sum_engine_num_ch_minus_one = VSI_NN_SP_SUM_ENGINE_NUM_CH_TWO_CH;
    attr.sum_engine_2d_accum_storeage = VSI_NN_SP_ACCM_STOREAGE_DIFFERENT;
    attr.v11_reset_at_start = VSI_NN_SP_V_RESET_AT_START_RESET;
    attr.v12_reset_at_start = VSI_NN_SP_V_RESET_AT_START_RESET;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_Z;
    attr.split_max_vector_depth = max_vector_depth;

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
        float             eps,
        char            * kernel_name
    )
{
    const int32_t spInitInstsNum = 2;
    const int32_t spLoopInstsNum = 5;
    const int32_t spInstsNum = spInitInstsNum + spLoopInstsNum;

    const uint32_t input_count = 2;
    const uint32_t output_count = 2;
    vx_tensor inputs_tensor[2] = {NULL};
    vx_tensor outputs_tensor[2] = {NULL};
    vx_node node = NULL;
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth /
        graph->ctx->config.sp_exec_count;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[7];
    vsi_nn_spinst_attr_t attr;
    vsi_nn_sp_lut_params sp_lut_params;
    vx_lut_params_s vx_lut_params;

    vsi_status status = VSI_FAILURE;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);
    memset(&sp_lut_params, 0, sizeof(vsi_nn_sp_lut_params));
    memset(&vx_lut_params, 0, sizeof(vx_lut_params_s));

    /* init inst0: r2 = const_a */
    status  = vsi_nn_sp_move_constant(&sp_insts_param[0], const_a, VSI_NN_SP_SR2);
    /* init inst1: r3 = inv_m */
    status  = vsi_nn_sp_move_constant(&sp_insts_param[1], inv_m, VSI_NN_SP_SR3);
    /* loop inst0: acc0 = v11 * v11 || r6 = r4 + r5 || r5 = v11*/
    status  = vsi_nn_sp_mul(&sp_insts_param[2], VSI_NN_SP_VR11, VSI_NN_SP_VR11, VSI_NN_SP_SR4);
    status |= vsi_nn_sp_add(&sp_insts_param[2], VSI_NN_SP_SR4, VSI_NN_SP_SR5, VSI_NN_SP_SR6);
    status |= vsi_nn_sp_move(&sp_insts_param[2], VSI_NN_SP_VR11, VSI_NN_SP_SR5);
    /* loop inst1: r1 = pwlMul() || r7 = pwlAdd() */
    status |= vsi_nn_sp_mul(&sp_insts_param[3], VSI_NN_SP_PWLMUL, VSI_NN_SP_PWLMUL, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_sub(&sp_insts_param[3], VSI_NN_SP_PWLADD, VSI_NN_SP_PWLADD, VSI_NN_SP_SR7);
    /* loop inst2: r5 = r2 * v12 || v12 = r8 + r7 */
    status |= vsi_nn_sp_mul(&sp_insts_param[4], VSI_NN_SP_SR2, VSI_NN_SP_VR12, VSI_NN_SP_SR5);
    status |= vsi_nn_sp_add(&sp_insts_param[4], VSI_NN_SP_SR8, VSI_NN_SP_SR7, VSI_NN_SP_VR12);
    /* loop inst3: r1 = setup(r6) || v11 = r3 * r5 || r7 = r1 */
    status |= vsi_nn_sp_pwl_setup0(&sp_insts_param[5], VSI_NN_SP_SR6, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_mul(&sp_insts_param[5], VSI_NN_SP_SR3, VSI_NN_SP_SR5, VSI_NN_SP_VR11);
    status |= vsi_nn_sp_move(&sp_insts_param[5], VSI_NN_SP_SR1, VSI_NN_SP_SR7);
    /* loop inst3: r8 = r1 * r7 */
    status |= vsi_nn_sp_mul(&sp_insts_param[6], VSI_NN_SP_SR1, VSI_NN_SP_SR7, VSI_NN_SP_SR8);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;

    attr.input_setup = VSI_NN_SP_INPUT_SETUP_V11;
    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.ignored_leading_outputs = 0;
    attr.ignored_leading_v11_wr = 0;
    attr.ignored_leading_v12_wr = 3;
    attr.ignored_leading_v11_rd = 0;
    attr.flush_cycle_num = 17;

    attr.num_of_v11_rd_in_flush_cycle = 0;
    attr.num_of_v12_rd_in_flush_cycle = 1;
    attr.num_of_v11_wr_in_flush_cycle = 1;
    attr.num_of_v12_wr_in_flush_cycle = 4;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_Z;
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

    sp_lut_params.act_type = VSI_NN_SP_ACT_LINEAR_RSQRT;
    sp_lut_params.params[0] = s;
    sp_lut_params.params[1] = eps;
    sp_lut_params.params[2] = 1.0f;
    vsi_nn_sp_lut(vx_lut_params.in_lut, vx_lut_params.out_lut, &sp_lut_params);

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

vsi_nn_kernel_node_t vsi_nn_sp_instance_norm_alpha_beta_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * beta,
        vsi_nn_tensor_t             * alpha,
        vsi_nn_tensor_t             * dummy0,
        vsi_nn_tensor_t             * dummy1,
        vsi_nn_tensor_t             * output0,
        vsi_nn_tensor_t             * output1,
        char                        * kernel_name
    )
{
    const int32_t spInitInstsNum = 1;
    const int32_t spLoopInstsNum = 2;
    const int32_t spInstsNum = spInitInstsNum + spLoopInstsNum;

    const uint32_t input_count = 4;
    const uint32_t output_count = 2;
    vx_tensor inputs_tensor[4] = {NULL};
    vx_tensor outputs_tensor[2] = {NULL};
    vx_node node = NULL;
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth /
        graph->ctx->config.sp_exec_count;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[3];
    vsi_nn_spinst_attr_t attr;

    vsi_status status = VSI_FAILURE;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    /* init inst0: r5 = 0 */
    status  = vsi_nn_sp_move_constant(&sp_insts_param[0], 0, VSI_NN_SP_SR5);
    /* loop inst0: r3 = r1 * v11 || v12 = r1 + r5 || r2 = in0 */
    status  = vsi_nn_sp_mul(&sp_insts_param[1], VSI_NN_SP_SR1, VSI_NN_SP_VR11, VSI_NN_SP_SR3);
    status |= vsi_nn_sp_add(&sp_insts_param[1], VSI_NN_SP_SR1, VSI_NN_SP_SR5, VSI_NN_SP_VR12);
    status |= vsi_nn_sp_move(&sp_insts_param[1], VSI_NN_SP_SRIN, VSI_NN_SP_SR2);
    /* loop inst1: r1 = v12 * in1 || v11 = r4 - r3 || r4 = r2 */
    status  = vsi_nn_sp_mul(&sp_insts_param[2], VSI_NN_SP_VR12, VSI_NN_SP_SRIN, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_sub(&sp_insts_param[2], VSI_NN_SP_SR4, VSI_NN_SP_SR3, VSI_NN_SP_VR11);
    status |= vsi_nn_sp_move(&sp_insts_param[2], VSI_NN_SP_SR2, VSI_NN_SP_SR4);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;

    attr.input_setup = VSI_NN_SP_INPUT_SETUP_INTERLEAVE_TWO_INPUT;
    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.ignored_leading_acc_out = 3;
    attr.ignored_leading_v11_wr = 3;
    attr.ignored_leading_v12_wr = 2;
    attr.ignored_leading_v11_rd = 2;
    attr.flush_cycle_num = 6;
    attr.ch0_post_redistribute = VSI_NN_SP_CH_POST_REDISTRIBUTE_VECTOR_GATHER;
    attr.ch1_post_redistribute = VSI_NN_SP_CH_POST_REDISTRIBUTE_VECTOR_GATHER;

    attr.num_of_v11_rd_in_flush_cycle = 2;
    attr.num_of_v12_rd_in_flush_cycle = 0;
    attr.num_of_v11_wr_in_flush_cycle = 3;
    attr.num_of_v12_wr_in_flush_cycle = 2;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_Z;
    attr.split_max_vector_depth = max_vector_depth;

    spinst = vsi_nn_create_spinst(graph);
    CHECK_PTR_FAIL_GOTO( spinst, "Create spInst fail.", final );
    status  = vsi_nn_add_spinst_insts(spinst, sp_insts_param, spInstsNum);
    status |= vsi_nn_set_spinst_attr(spinst, attr);
    CHECK_STATUS_FAIL_GOTO(status, final );

    inputs_tensor[0] = beta->t;
    inputs_tensor[1] = alpha->t;
    inputs_tensor[2] = dummy0->t;
    inputs_tensor[3] = dummy1->t;
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

vsi_nn_kernel_node_t vsi_nn_sp_instance_norm_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input,
        vsi_nn_tensor_t             * dummy0_tensor,
        vsi_nn_tensor_t             * dummy1_tensor,
        vsi_nn_tensor_t             * output,
        char                        * kernel_name
    )
{
    const int32_t spLoopInstsNum = 1;
    const int32_t spInstsNum = spLoopInstsNum;

    const uint32_t input_count = 3;
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

    /* loop inst0: r1 = in * v12 || out = r1 + v11 */
    status  = vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_VR12, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_add(&sp_insts_param[0], VSI_NN_SP_SR1, VSI_NN_SP_VR11, VSI_NN_SP_SROUT);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;

    attr.input_setup = VSI_NN_SP_INPUT_SETUP_SINGLE_INPUT;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.ignored_leading_outputs = 3;
    attr.ignored_leading_v11_rd = 3;
    attr.ignored_leading_v12_rd = 0;
    attr.flush_cycle_num = 3;
    attr.v11_push_pop_config = VSI_NN_SP_PUSH_POP_EVERY_ROW;
    attr.v12_push_pop_config = VSI_NN_SP_PUSH_POP_EVERY_ROW;

    attr.num_of_v11_rd_in_flush_cycle = 3;
    attr.num_of_v11_wr_in_flush_cycle = 0;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_Z;
    attr.split_max_vector_depth = max_vector_depth;

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

    status = vsi_nn_set_sp_kernel_name(node, kernel_name);
    CHECK_STATUS_FAIL_GOTO(status, final );

final:
    if (spinst)
    {
        vsi_nn_release_spinst(&spinst);
    }

    return (vsi_nn_kernel_node_t)node;
}

#define REGISTER_INSTANCE_NORM_STREAM_PROCESSOR_KERNEL( kernel_name )   \
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

REGISTER_INSTANCE_NORM_STREAM_PROCESSOR_KERNEL( instance_norm )
{
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_t * reshape_tensors[2] = {NULL};
    vsi_size_t shape[VSI_NN_MAX_DIM_NUM] = { 0 };
    vsi_nn_tensor_t * dummy_tensor[6] = {NULL};
    vsi_nn_tensor_t * gamma = NULL;
    vsi_nn_tensor_t * beta = NULL;
    float output_scale = 1.0f / vsi_nn_get_tensor_scale(outputs[0]);
    float eps = vsi_nn_kernel_param_get_float32( params, "eps" );
    float inv_m = 1.0f / (float)(outputs[0]->attr.size[0] * outputs[0]->attr.size[1]);
    float s = inv_m * inv_m;
    float const_a = (float)(outputs[0]->attr.size[0] * outputs[0]->attr.size[1]);

    memcpy( &attr, &outputs[0]->attr, sizeof(vsi_nn_tensor_attr_t) );
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

    gamma = vsi_nn_dropout_tensor(graph, inputs[2], output_scale);
    beta = vsi_nn_dropout_tensor(graph, inputs[1], output_scale);

    shape[0] = 1;
    shape[1] = 1;
    shape[2] = outputs[0]->attr.size[2];
    reshape_tensors[0] = vsi_nn_reshape_tensor( graph,
                beta, shape, 3 );
    reshape_tensors[1] = vsi_nn_reshape_tensor( graph,
                gamma, shape, 3 );

    node = vsi_nn_sp_moments_sums_node(graph, inputs[0], dummy_tensor[0], dummy_tensor[1], "instancenorm_0");
    CHECK_PTR_FAIL_GOTO( node, "Create sp_instance_norm_sums  fail.", final );
    node = vsi_nn_sp_moments_means_node(graph, dummy_tensor[0], dummy_tensor[1],
        dummy_tensor[2], dummy_tensor[3], inv_m, const_a, s, eps, "instancenorm_1");
    CHECK_PTR_FAIL_GOTO( node, "Create sp_instance_norm_means  fail.", final );
    node = vsi_nn_sp_instance_norm_alpha_beta_node(graph, reshape_tensors[0], reshape_tensors[1],
        dummy_tensor[2], dummy_tensor[3], dummy_tensor[4], dummy_tensor[5], "instancenorm_2");
    CHECK_PTR_FAIL_GOTO( node, "Create sp_instance_norm_alpha_beta  fail.", final );
    node = vsi_nn_sp_instance_norm_node(graph, inputs[0], dummy_tensor[4],
        dummy_tensor[5], outputs[0], "instancenorm_3");
    CHECK_PTR_FAIL_GOTO( node, "Create sp_instance_norm  fail.", final );

final:
    vsi_safe_release_tensor(dummy_tensor[0]);
    vsi_safe_release_tensor(dummy_tensor[1]);
    vsi_safe_release_tensor(dummy_tensor[2]);
    vsi_safe_release_tensor(dummy_tensor[3]);
    vsi_safe_release_tensor(dummy_tensor[4]);
    vsi_safe_release_tensor(dummy_tensor[5]);
    vsi_safe_release_tensor(gamma);
    vsi_safe_release_tensor(beta);
    vsi_safe_release_tensor(reshape_tensors[0]);
    vsi_safe_release_tensor(reshape_tensors[1]);

    return node;
} /* instance_norm() */

#undef REGISTER_INSTANCE_NORM_STREAM_PROCESSOR_KERNEL

#endif
