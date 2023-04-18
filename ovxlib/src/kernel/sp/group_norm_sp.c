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
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"

#if (VX_STREAM_PROCESSOR_SUPPORT)

vsi_nn_kernel_node_t vsi_nn_sp_moments_sums_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input,
        vsi_nn_tensor_t             * output0,
        vsi_nn_tensor_t             * output1,
        char                        * kernel_name
    );

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

vsi_nn_kernel_node_t vsi_nn_sp_a_minus_v11_times_v12_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input0,
        vsi_nn_tensor_t             * input1,
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

    /* loop inst0: out = r1 * v12 | r1 = in - v11 */
    status  = vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_SR1, VSI_NN_SP_VR12, VSI_NN_SP_SROUT);
    status |= vsi_nn_sp_sub(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_VR11, VSI_NN_SP_SR1);
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

    attr.num_of_v12_rd_in_flush_cycle = 3;
    attr.num_of_v12_wr_in_flush_cycle = 0;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_Z;
    attr.split_max_vector_depth = max_vector_depth;

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

vsi_nn_kernel_node_t vsi_nn_sp_gn_means_node
    (
        vsi_nn_graph_t  * graph,
        vsi_nn_tensor_t * input,
        vsi_nn_tensor_t * output,
        float             inv_m,
        float             eps,
        char            * kernel_name
    )
{
    const int32_t spInitInstsNum = 0;
    const int32_t spLoopInstsNum = 3;
    const int32_t spInstsNum = spInitInstsNum + spLoopInstsNum;

    const uint32_t input_count = 1;
    const uint32_t output_count = 1;
    vx_tensor inputs_tensor[1] = {NULL};
    vx_tensor outputs_tensor[1] = {NULL};
    vx_node node = NULL;
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth /
        graph->ctx->config.sp_exec_count;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[5];
    vsi_nn_spinst_attr_t attr;

    vsi_status status = VSI_FAILURE;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    /* loop inst0: r1 = v11 * r3 | v11 = r1 */
    status  = vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_VR11, VSI_NN_SP_SR3, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_move(&sp_insts_param[0], VSI_NN_SP_SR1, VSI_NN_SP_VR11);
    /* loop inst1: r5 = r1 * r1 | r6 = r2 - r5 */
    status |= vsi_nn_sp_mul(&sp_insts_param[1], VSI_NN_SP_SR1, VSI_NN_SP_SR1, VSI_NN_SP_SR5);
    status |= vsi_nn_sp_sub(&sp_insts_param[1], VSI_NN_SP_SR2, VSI_NN_SP_SR5, VSI_NN_SP_SR6);
    /* loop inst2: r2 = r3 * v12 | v12 = r4 + r6 */
    status |= vsi_nn_sp_mul(&sp_insts_param[2], VSI_NN_SP_SR3, VSI_NN_SP_VR12, VSI_NN_SP_SR2);
    status |= vsi_nn_sp_add(&sp_insts_param[2], VSI_NN_SP_SR4, VSI_NN_SP_SR6, VSI_NN_SP_VR12);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;

    attr.input_setup = VSI_NN_SP_INPUT_SETUP_V12;
    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.ignored_leading_outputs = 0;
    attr.ignored_leading_v11_wr = 1;
    attr.ignored_leading_v12_wr = 3;
    attr.flush_cycle_num = 9;
    attr.ch0_post_redistribute = VSI_NN_SP_CH_POST_REDISTRIBUTE_VECTOR_GATHER;

    attr.num_of_v11_rd_in_flush_cycle = 0;
    attr.num_of_v12_rd_in_flush_cycle = 0;
    attr.num_of_v11_wr_in_flush_cycle = 1;
    attr.num_of_v12_wr_in_flush_cycle = 3;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_Z;
    attr.split_max_vector_depth = max_vector_depth;

    VSI_NN_SP_ATTR_SET_CONST_TO_SR3(attr, inv_m);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR4(attr, eps);

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

    status = vsi_nn_set_sp_kernel_name(node, kernel_name);
    CHECK_STATUS_FAIL_GOTO(status, final );

final:
    if (spinst)
    {
        vsi_nn_release_spinst(&spinst);
    }

    return (vsi_nn_kernel_node_t)node;
}

vsi_nn_kernel_node_t vsi_nn_sp_gn_rsqrt_node
    (
        vsi_nn_graph_t  * graph,
        vsi_nn_tensor_t * input,
        vsi_nn_tensor_t * output,
        char            * kernel_name
    )
{
    const int32_t spInitInstsNum = 0;
    const int32_t spLoopInstsNum = 14;
    const int32_t spInstsNum = spInitInstsNum + spLoopInstsNum;

    const uint32_t input_count = 1;
    const uint32_t output_count = 1;
    vx_tensor inputs_tensor[1] = {NULL};
    vx_tensor outputs_tensor[1] = {NULL};
    vx_node node = NULL;
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth /
        graph->ctx->config.sp_exec_count;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[14];
    vsi_nn_spinst_attr_t attr;
    vx_lut_params_s vx_lut_params;

    vsi_status status = VSI_FAILURE;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);
    memset(&vx_lut_params, 0, sizeof(vx_lut_params_s));

    /* loop inst0: r1 = setup(v12) | r2 = r4 * v12 */
    status = vsi_nn_sp_pwl_setup0(&sp_insts_param[0], VSI_NN_SP_VR12, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_SR4, VSI_NN_SP_VR12, VSI_NN_SP_SR2);
    /* loop inst1: r6 = r5 * r6 */
    status |= vsi_nn_sp_mul(&sp_insts_param[1], VSI_NN_SP_SR5, VSI_NN_SP_SR6, VSI_NN_SP_SR6);
    /* loop inst2: r7 = r3 - r7 | r10 = r2 */
    status |= vsi_nn_sp_sub(&sp_insts_param[2], VSI_NN_SP_SR3, VSI_NN_SP_SR7, VSI_NN_SP_SR7);
    status |= vsi_nn_sp_move(&sp_insts_param[2], VSI_NN_SP_SR2, VSI_NN_SP_SR10);
    /* loop inst3: r8 = pwlMul() | r9 = pwlAdd() */
    status |= vsi_nn_sp_mul(&sp_insts_param[3], VSI_NN_SP_PWLMUL, VSI_NN_SP_PWLMUL, VSI_NN_SP_SR8);
    status |= vsi_nn_sp_sub(&sp_insts_param[3], VSI_NN_SP_PWLADD, VSI_NN_SP_PWLADD, VSI_NN_SP_SR9);
    /* loop inst4: r6 = r3 - r6 */
    status |= vsi_nn_sp_sub(&sp_insts_param[4], VSI_NN_SP_SR3, VSI_NN_SP_SR6, VSI_NN_SP_SR6);
    /* loop inst5: v12 = r8 * r7 */
    status |= vsi_nn_sp_mul(&sp_insts_param[5], VSI_NN_SP_SR8, VSI_NN_SP_SR7, VSI_NN_SP_VR12);
    /* loop inst6: r5 = r8 * r9 */
    status |= vsi_nn_sp_mul(&sp_insts_param[6], VSI_NN_SP_SR8, VSI_NN_SP_SR9, VSI_NN_SP_SR5);
    /* loop inst7: r6 = r5 * r6 */
    status |= vsi_nn_sp_mul(&sp_insts_param[7], VSI_NN_SP_SR5, VSI_NN_SP_SR6, VSI_NN_SP_SR6);
    /* loop inst8: nop */
    status |= vsi_nn_sp_nop(&sp_insts_param[8]);
    /* loop inst9: r5 = r1 + r5 */
    status |= vsi_nn_sp_add(&sp_insts_param[9], VSI_NN_SP_SR1, VSI_NN_SP_SR5, VSI_NN_SP_SR5);
    /* loop inst10: r7 = r6 * r10 */
    status |= vsi_nn_sp_mul(&sp_insts_param[10], VSI_NN_SP_SR6, VSI_NN_SP_SR10, VSI_NN_SP_SR7);
    /* loop inst11: nop */
    status |= vsi_nn_sp_nop(&sp_insts_param[11]);
    /* loop inst12: r6 = r2 * r5 */
    status |= vsi_nn_sp_mul(&sp_insts_param[12], VSI_NN_SP_SR2, VSI_NN_SP_SR5, VSI_NN_SP_SR6);
    /* loop inst13: r7 = r6 * r7 | r8 = r6 */
    status |= vsi_nn_sp_mul(&sp_insts_param[13], VSI_NN_SP_SR6, VSI_NN_SP_SR7, VSI_NN_SP_SR7);
    status |= vsi_nn_sp_move(&sp_insts_param[13], VSI_NN_SP_SR6, VSI_NN_SP_SR8);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;

    attr.input_setup = VSI_NN_SP_INPUT_SETUP_V12;
    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.ignored_leading_outputs = 0;
    attr.ignored_leading_v12_wr = 2;
    attr.flush_cycle_num = 33;
    attr.ch1_post_redistribute = VSI_NN_SP_CH_POST_REDISTRIBUTE_VECTOR_GATHER;

    attr.num_of_v12_rd_in_flush_cycle = 0;
    attr.num_of_v12_wr_in_flush_cycle = 3;

    VSI_NN_SP_ATTR_SET_CONST_TO_SR3(attr, 1.5f);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR4(attr, 0.5f);

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_Z;
    attr.split_max_vector_depth = max_vector_depth;

    spinst = vsi_nn_create_spinst(graph);
    CHECK_PTR_FAIL_GOTO( spinst, "Create spInst fail.", final );
    status  = vsi_nn_add_spinst_insts(spinst, sp_insts_param, spInstsNum);
    status |= vsi_nn_set_spinst_attr(spinst, attr);
    CHECK_STATUS_FAIL_GOTO(status, final );

    inputs_tensor[0] = input->t;
    outputs_tensor[0] = output->t;

    vx_lut_params.lut_function = VX_NN_ACTIVATION_RSQRT;
    vx_lut_params.float_values[0] = 0;
    vx_lut_params.float_values[1] = 0;
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
    vsi_nn_kernel_node_t node[6] = {NULL};
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_t * reshape_tensors[4] = {NULL};
    vsi_nn_tensor_t * dummy_tensor[5] = {NULL};
    vsi_nn_tensor_t * output_tensors[2] = {NULL};
    vsi_nn_tensor_t * gamma = NULL;
    vsi_nn_tensor_t * beta = NULL;
    vsi_size_t shapes[2][VSI_NN_MAX_DIM_NUM] = { {1, 1, 1, 1}, {1, 1, 1, 1} };
    int32_t group_num  = vsi_nn_kernel_param_get_int32( params, "group_num" );
    int32_t group_size = (int32_t)outputs[0]->attr.size[2] / group_num;
    float output_scale = 1.0f / vsi_nn_get_tensor_scale(outputs[0]);
    float eps = vsi_nn_kernel_param_get_float32( params, "eps" );
    float inv_m = 1.0f / (float)(outputs[0]->attr.size[0] * outputs[0]->attr.size[1] * group_size);

    VSI_UNREFERENCED(input_num);
    VSI_UNREFERENCED(output_num);
    VSI_UNREFERENCED(kernel);

    status =  vsi_nn_kernel_optimize_group_norm_shape( (const vsi_size_t*)inputs[0]->attr.size,
        inputs[0]->attr.dim_num, group_num, 1, shapes[0]);
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
    output_tensors[0] = vsi_nn_CreateTensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( output_tensors[0], "Create dummy_tensor fail.", final );
    output_tensors[1] = vsi_nn_CreateTensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( output_tensors[1], "Create ifco_tensor fail.", final );
    reshape_tensors[1] = vsi_nn_reshape_tensor(graph, output_tensors[1],
        outputs[0]->attr.size, outputs[0]->attr.dim_num);

    memcpy( &attr, &reshape_tensors[0]->attr, sizeof(vsi_nn_tensor_attr_t) );
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    attr.is_const = FALSE;
    attr.vtl = TRUE;
    attr.size[0] = 1;
    attr.size[1] = 1;
    dummy_tensor[0] = vsi_nn_create_dummy_tensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[0], "Create dummy_tensor fail.", final );
    dummy_tensor[1] = vsi_nn_create_dummy_tensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[1], "Create dummy_tensor fail.", final );
    dummy_tensor[4] = vsi_nn_create_dummy_tensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[1], "Create dummy_tensor fail.", final );
    memcpy( &attr, &reshape_tensors[1]->attr, sizeof(vsi_nn_tensor_attr_t) );
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    attr.is_const = FALSE;
    attr.vtl = TRUE;
    attr.size[0] = 1;
    attr.size[1] = 1;
    dummy_tensor[2] = vsi_nn_create_dummy_tensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[2], "Create dummy_tensor fail.", final );
    dummy_tensor[3] = vsi_nn_create_dummy_tensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[3], "Create dummy_tensor fail.", final );

    gamma = vsi_nn_dropout_tensor(graph, reshape_tensors[3], output_scale);
    beta = vsi_nn_dropout_tensor(graph, reshape_tensors[2], output_scale);

    node[0] = vsi_nn_sp_moments_sums_node(graph, reshape_tensors[0],
        output_tensors[0], dummy_tensor[0], "groupnorm_0");
    CHECK_PTR_FAIL_GOTO( node[0], "Create moments_sums fail.", final );
    node[1] = vsi_nn_sp_gn_means_node(graph, dummy_tensor[0], dummy_tensor[1],
        inv_m, eps, "groupnorm_1");
    CHECK_PTR_FAIL_GOTO( node[1], "Create moments_means fail.", final );
    node[2] = vsi_nn_sp_gn_rsqrt_node(graph, dummy_tensor[1], dummy_tensor[4], "groupnorm_2");
    CHECK_PTR_FAIL_GOTO( node[2], "Create moments_means fail.", final );
    node[3] = vsi_nn_sp_a_minus_v11_times_v12_node(graph, output_tensors[0], dummy_tensor[4],
        output_tensors[1], "groupnorm_3");
    CHECK_PTR_FAIL_GOTO( node[3], "Create a_minus_v11_times_v12 fail.", final );
    node[4] = vsi_nn_sp_bn_mov_weight_bias_node(graph, gamma, beta, dummy_tensor[2], dummy_tensor[3], "groupnorm_4");
    CHECK_PTR_FAIL_GOTO( node[4], "Create mov_weight_bias fail.", final );
    node[5] = vsi_nn_sp_bn_in_times_v11_plus_v12_node(graph, reshape_tensors[1], dummy_tensor[2],
        dummy_tensor[3], outputs[0], "groupnorm_5");
    CHECK_PTR_FAIL_GOTO( node[5], "Create in_times_v11_plus_v12 fail.", final );

final:
    vsi_safe_release_node(node[0]);
    vsi_safe_release_node(node[1]);
    vsi_safe_release_node(node[2]);
    vsi_safe_release_node(node[3]);
    vsi_safe_release_node(node[4]);
    vsi_safe_release_tensor(dummy_tensor[0]);
    vsi_safe_release_tensor(dummy_tensor[1]);
    vsi_safe_release_tensor(dummy_tensor[2]);
    vsi_safe_release_tensor(dummy_tensor[3]);
    vsi_safe_release_tensor(dummy_tensor[4]);
    vsi_safe_release_tensor(gamma);
    vsi_safe_release_tensor(beta);
    vsi_safe_release_tensor(output_tensors[0]);
    vsi_safe_release_tensor(output_tensors[1]);
    vsi_safe_release_tensor(reshape_tensors[0]);
    vsi_safe_release_tensor(reshape_tensors[1]);
    vsi_safe_release_tensor(reshape_tensors[2]);
    vsi_safe_release_tensor(reshape_tensors[3]);

    return node[5];
} /* group_norm() */

#undef REGISTER_GROUP_NORM_STREAM_PROCESSOR_KERNEL

#endif
