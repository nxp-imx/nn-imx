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

vsi_nn_spinst_t * vsi_nn_sp_l2norm_z_direction_square_inst
    (
        vx_context                context,
        vsi_nn_spinst_t         * prev_spinst,
        int32_t                   fifo_depth,
        int32_t                   max_vector_depth
    )
{
    vsi_status status = VSI_FAILURE;
    const int32_t spInitInstsNum = fifo_depth > 3 ? 1 : fifo_depth > 1 ? 0 : 2;
    const int32_t spLoopInstsNum = fifo_depth > 3 ? 2 : fifo_depth > 1 ? 4 : 3;
    const int32_t spCompleteInstsNum = fifo_depth > 1 ? 0 : 3;
    const int32_t spInstsNum = spInitInstsNum + spLoopInstsNum + spCompleteInstsNum;
    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[9];
    vsi_nn_spinst_attr_t attr;
    float constant[5] = {0};

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    if (fifo_depth > 3)
    {
        /* init inst0: r2 = 0 */
        status  = vsi_nn_sp_move_constant(&sp_insts_param[0], 0, VSI_NN_SP_SR2);
        /* loop inst0: r1 = clamp(r3 * in, r6, r7) | v11 = r4 + r5 | out = r8 */
        status |= vsi_nn_sp_mul_clamp(&sp_insts_param[1], VSI_NN_SP_SR3, VSI_NN_SP_SRIN, VSI_NN_SP_SR1);
        status |= vsi_nn_sp_add(&sp_insts_param[1], VSI_NN_SP_SR4, VSI_NN_SP_SR5, VSI_NN_SP_VR11);
        status |= vsi_nn_sp_move(&sp_insts_param[1], VSI_NN_SP_SR8, VSI_NN_SP_SROUT);
        /* loop inst1: r4 = r1 * r1 | r8 = r1 + r2 | r5 = v11 */
        status |= vsi_nn_sp_mul(&sp_insts_param[2], VSI_NN_SP_SR1, VSI_NN_SP_SR1, VSI_NN_SP_SR4);
        status |= vsi_nn_sp_add(&sp_insts_param[2], VSI_NN_SP_SR1, VSI_NN_SP_SR2, VSI_NN_SP_SR8);
        status |= vsi_nn_sp_move(&sp_insts_param[2], VSI_NN_SP_VR11, VSI_NN_SP_SR5);
        CHECK_STATUS_FAIL_GOTO(status, final );

        attr.flush_cycle_num = 6;

        attr.ignored_leading_outputs = 3;
        attr.ignored_leading_v11_rd = fifo_depth + 1;
        attr.ignored_leading_v11_wr = 3;

        attr.num_of_v11_rd_in_flush_cycle = 2;
        attr.num_of_v11_wr_in_flush_cycle = 3;
    }
    else if (fifo_depth > 1)
    {
        /* loop inst0: r1 = clamp(r3 * in, r6, r7) | out = r1 */
        status  = vsi_nn_sp_mul_clamp(&sp_insts_param[0], VSI_NN_SP_SR3, VSI_NN_SP_SRIN, VSI_NN_SP_SR1);
        status |= vsi_nn_sp_move(&sp_insts_param[0], VSI_NN_SP_SR1, VSI_NN_SP_SROUT);
        /* loop inst1: nop */
        status |= vsi_nn_sp_nop(&sp_insts_param[1]);
        /* loop inst2: v11 = r4 + r5 */
        status |= vsi_nn_sp_add(&sp_insts_param[2], VSI_NN_SP_SR4, VSI_NN_SP_SR5, VSI_NN_SP_VR11);
        /* loop inst3: r4 = r1 * r1 | r5 = v11 */
        status |= vsi_nn_sp_mul(&sp_insts_param[3], VSI_NN_SP_SR1, VSI_NN_SP_SR1, VSI_NN_SP_SR4);
        status |= vsi_nn_sp_move(&sp_insts_param[3], VSI_NN_SP_VR11, VSI_NN_SP_SR5);
        CHECK_STATUS_FAIL_GOTO(status, final );

        attr.flush_cycle_num = 6;

        attr.ignored_leading_outputs = 1;
        attr.ignored_leading_v11_rd = fifo_depth;
        attr.ignored_leading_v11_wr = 1;

        attr.num_of_v11_rd_in_flush_cycle = 1;
        attr.num_of_v11_wr_in_flush_cycle = 2;
    }
    else
    {
        /* init inst0: r1 = 0 */
        status  = vsi_nn_sp_move_constant(&sp_insts_param[0], 0, VSI_NN_SP_SR1);
        /* init inst1: nop */
        status |= vsi_nn_sp_nop(&sp_insts_param[1]);
        /* loop inst0: r1 = clamp(r3 * in, r6, r7) | r2 = r1 */
        status |= vsi_nn_sp_mul_clamp(&sp_insts_param[2], VSI_NN_SP_SR3, VSI_NN_SP_SRIN, VSI_NN_SP_SR1);
        status |= vsi_nn_sp_move(&sp_insts_param[2], VSI_NN_SP_SR1, VSI_NN_SP_SR2);
        /* loop inst1: r4 = r1 * r1 | r5 = r4 + r5 | out = r2 */
        status |= vsi_nn_sp_mul(&sp_insts_param[3], VSI_NN_SP_SR1, VSI_NN_SP_SR1, VSI_NN_SP_SR4);
        status |= vsi_nn_sp_add(&sp_insts_param[3], VSI_NN_SP_SR4, VSI_NN_SP_SR5, VSI_NN_SP_SR5);
        status |= vsi_nn_sp_move(&sp_insts_param[3], VSI_NN_SP_SR2, VSI_NN_SP_SROUT);
        /* loop inst2: nop */
        status |= vsi_nn_sp_nop(&sp_insts_param[4]);
        /* complete inst0: nop */
        status |= vsi_nn_sp_nop(&sp_insts_param[5]);
        /* complete inst1: nop */
        status |= vsi_nn_sp_nop(&sp_insts_param[6]);
        /* complete inst2: v11 = r5 */
        status |= vsi_nn_sp_move(&sp_insts_param[7], VSI_NN_SP_SR5, VSI_NN_SP_VR11);
        CHECK_STATUS_FAIL_GOTO(status, final );

        attr.flush_cycle_num = 7;

        attr.ignored_leading_outputs = 2;
    }

    status = vsi_nn_get_constant_from_spinst(prev_spinst, constant);
    CHECK_STATUS_FAIL_GOTO(status, final );

    VSI_NN_SP_ATTR_SET_CONST_TO_SR3(attr, constant[0]);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR4(attr, constant[1]);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR5_LOW_PRECISION(attr, constant[2]);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR6(attr, constant[3]);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR7(attr, constant[4]);

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;
    attr.input_setup = VSI_NN_SP_INPUT_SETUP_SINGLE_INPUT;

    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.prog_complete_instr_num = spCompleteInstsNum;
    attr.v11_reset_at_start = VSI_NN_SP_V_RESET_AT_START_RESET;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_XY;
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

DEF_SP_KERNEL_QUERY(l2norm_z_direction_square_query)
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

    spinst = vsi_nn_sp_l2norm_z_direction_square_inst(ctx, &pre_spinst, fifo_depth, max_vector_depth);

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

vsi_nn_kernel_node_t vsi_nn_sp_l2norm_z_direction_square_node
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

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[3];
    vsi_nn_spinst_attr_t attr;

    vsi_status status = VSI_FAILURE;
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth;
    float input_scale = vsi_nn_get_tensor_scale(input);
    float clamp_min = 0;
    float clamp_max = 0;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    vsi_nn_get_tensor_clamp_min_max(input, &clamp_min, &clamp_max);
    clamp_min = clamp_min * input_scale;
    clamp_max = clamp_max * input_scale;

    /* loop inst0: r2 = 0 */
    status  = vsi_nn_sp_move_constant(&sp_insts_param[0], 0, VSI_NN_SP_SR2);
    /* loop inst0: r1 = clamp(r3 * in, r6, r7) | v11 = r4 + r5 | out = r8 */
    status  = vsi_nn_sp_mul_clamp(&sp_insts_param[1], VSI_NN_SP_SR3, VSI_NN_SP_SRIN, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_add(&sp_insts_param[1], VSI_NN_SP_SR4, VSI_NN_SP_SR5, VSI_NN_SP_VR11);
    status |= vsi_nn_sp_move(&sp_insts_param[1], VSI_NN_SP_SR8, VSI_NN_SP_SROUT);
    /* loop inst1: r4 = r1 * r1 | r8 = r1 + r2 | r5 = v11 */
    status |= vsi_nn_sp_mul(&sp_insts_param[2], VSI_NN_SP_SR1, VSI_NN_SP_SR1, VSI_NN_SP_SR4);
    status |= vsi_nn_sp_add(&sp_insts_param[2], VSI_NN_SP_SR1, VSI_NN_SP_SR2, VSI_NN_SP_SR8);
    status |= vsi_nn_sp_move(&sp_insts_param[2], VSI_NN_SP_VR11, VSI_NN_SP_SR5);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;
    attr.input_setup = VSI_NN_SP_INPUT_SETUP_SINGLE_INPUT;

    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.v11_reset_at_start = VSI_NN_SP_V_RESET_AT_START_RESET;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_XY;
    attr.split_tilex_equal_imgx = TRUE;
    attr.split_max_vector_depth = max_vector_depth;

    attr.flush_cycle_num = 6;

    attr.ignored_leading_outputs = 3;
    attr.ignored_leading_v11_rd = 5;
    attr.ignored_leading_v11_wr = 1;

    attr.num_of_v11_rd_in_flush_cycle = 2;
    attr.num_of_v11_wr_in_flush_cycle = 3;

    VSI_NN_SP_ATTR_SET_CONST_TO_SR3(attr, input_scale);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR4(attr, 0);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR5_LOW_PRECISION(attr, 0);
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

    if (node)
    {
        vxAssignNodeQueryCallback(node, l2norm_z_direction_square_query);
    }

    if (spinst)
    {
        vsi_nn_release_spinst(&spinst);
    }

    return (vsi_nn_kernel_node_t)node;
}

vsi_nn_kernel_node_t vsi_nn_sp_l2norm_z_direction_rsqrt_node
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
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[14];
    vsi_nn_spinst_attr_t attr;
    vx_lut_params_s vx_lut_params;

    vsi_status status = VSI_FAILURE;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);
    memset(&vx_lut_params, 0, sizeof(vx_lut_params_s));

    /* loop inst0: r1 = setup(v11) | r2 = r4 * v11 */
    status = vsi_nn_sp_pwl_setup0(&sp_insts_param[0], VSI_NN_SP_VR11, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_SR4, VSI_NN_SP_VR11, VSI_NN_SP_SR2);
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
    /* loop inst5: v11 = r8 * r7 */
    status |= vsi_nn_sp_mul(&sp_insts_param[5], VSI_NN_SP_SR8, VSI_NN_SP_SR7, VSI_NN_SP_VR11);
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

    attr.input_setup = VSI_NN_SP_INPUT_SETUP_V11;
    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.ignored_leading_outputs = 0;
    attr.ignored_leading_v11_wr = 2;
    attr.flush_cycle_num = 33;

    attr.num_of_v11_rd_in_flush_cycle = 0;
    attr.num_of_v11_wr_in_flush_cycle = 3;

    VSI_NN_SP_ATTR_SET_CONST_TO_SR3(attr, 1.5f);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR4(attr, 0.5f);

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_XY;
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

vsi_nn_kernel_node_t vsi_nn_sp_l2norm_z_direction_mul_node
    (
        vsi_nn_graph_t  * graph,
        vsi_nn_tensor_t * input,
        vsi_nn_tensor_t * output,
        float             output_scale,
        char            * kernel_name
    )
{
    const int32_t spInitInstsNum = 0;
    int32_t spLoopInstsNum = 1;
    int32_t spInstsNum = spInitInstsNum + spLoopInstsNum;

    const uint32_t input_count = 1;
    const uint32_t output_count = 1;
    vx_tensor inputs_tensor[1] = {NULL};
    vx_tensor outputs_tensor[1] = {NULL};
    vx_node node = NULL;
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[1];
    vsi_nn_spinst_attr_t attr;

    vsi_status status = VSI_FAILURE;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    if (output_scale == 1.0f)
    {
        spLoopInstsNum = 0;
        spInstsNum = spInitInstsNum + spLoopInstsNum;
    }
    else
    {
        /* loop inst0: v11 = r3 * v11 */
        status = vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_SR3, VSI_NN_SP_VR11, VSI_NN_SP_VR11);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;

    attr.input_setup = VSI_NN_SP_INPUT_SETUP_V11;
    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.ignored_leading_outputs = 0;
    attr.ignored_leading_v11_wr = 0;
    attr.flush_cycle_num = 0;

    attr.num_of_v11_rd_in_flush_cycle = 0;
    attr.num_of_v11_wr_in_flush_cycle = 0;

    VSI_NN_SP_ATTR_SET_CONST_TO_SR3(attr, 1 / output_scale);

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_XY;
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

    status = vsi_nn_set_sp_kernel_name(node, kernel_name);
    CHECK_STATUS_FAIL_GOTO(status, final );

final:
    if (spinst)
    {
        vsi_nn_release_spinst(&spinst);
    }

    return (vsi_nn_kernel_node_t)node;
}

vsi_nn_spinst_t * vsi_nn_sp_l2norm_z_direction_times_inst
    (
        vx_context                context,
        int32_t                   fifo_depth,
        int32_t                   max_vector_depth
    )
{
    vsi_status status = VSI_FAILURE;
    const int32_t spInitInstsNum = fifo_depth > 1 ? 0 : 3;
    const int32_t spLoopInstsNum = (fifo_depth > 4 || fifo_depth == 1) ? 1 : 3;
    const int32_t spInstsNum = spInitInstsNum + spLoopInstsNum;
    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[4];
    vsi_nn_spinst_attr_t attr;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    if (fifo_depth > 4)
    {
        /* loop inst0: out = v11 * in | v11 = v11 */
        status  = vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_VR11, VSI_NN_SP_SRIN, VSI_NN_SP_SROUT);
        status |= vsi_nn_sp_move(&sp_insts_param[0], VSI_NN_SP_VR11, VSI_NN_SP_VR11);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    else if (fifo_depth > 1)
    {
        /* loop inst0: out = v11 * in | v11 = v11 */
        status  = vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_VR11, VSI_NN_SP_SRIN, VSI_NN_SP_SROUT);
        status |= vsi_nn_sp_move(&sp_insts_param[0], VSI_NN_SP_VR11, VSI_NN_SP_VR11);
        /* loop inst1: nop */
        status |= vsi_nn_sp_nop(&sp_insts_param[1]);
        /* loop inst2: nop */
        status |= vsi_nn_sp_nop(&sp_insts_param[2]);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }
    else
    {
        /* init inst0: r2 = v11 */
        status  = vsi_nn_sp_move(&sp_insts_param[0], VSI_NN_SP_VR11, VSI_NN_SP_SR2);
        /* init inst1: nop */
        status |= vsi_nn_sp_nop(&sp_insts_param[1]);
        /* init inst2: nop */
        status |= vsi_nn_sp_nop(&sp_insts_param[2]);

        /* loop inst0: out = r2 * in */
        status |= vsi_nn_sp_mul(&sp_insts_param[3], VSI_NN_SP_SR2, VSI_NN_SP_SRIN, VSI_NN_SP_SROUT);
        CHECK_STATUS_FAIL_GOTO(status, final );
    }

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;
    attr.input_setup = VSI_NN_SP_INPUT_SETUP_SINGLE_INPUT;

    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_XY;
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

DEF_SP_KERNEL_QUERY(l2norm_z_direction_times_query)
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
    vx_context  ctx = vxGetContext((vx_reference)node);
    vx_hardware_caps_params_ext2_t hw_param;

    memset(&hw_param, 0, sizeof(vx_hardware_caps_params_ext2_t));
    status = vxQueryHardwareCaps(ctx, (vx_hardware_caps_params_t*)(&hw_param), sizeof(vx_hardware_caps_params_ext2_t));
    CHECK_STATUS_FAIL_GOTO( status, final );

    status = vxQueryNode(node, VX_NODE_SWTILING_TILE_XY, tile_size, sizeof(tile_size));
    CHECK_STATUS_FAIL_GOTO( status, final );
    status = vxQueryNode(node, VX_NODE_SPINST_INDEX, &index, sizeof(index));
    CHECK_STATUS_FAIL_GOTO( status, final );

    fifo_depth = (int32_t)ceil((float)(tile_size[0] * tile_size[1]) / (float)hw_param.streamProcessorExecCount);
    max_vector_depth = hw_param.streamProcessorVectorSize;

    spinst = vsi_nn_sp_l2norm_z_direction_times_inst(ctx, fifo_depth, max_vector_depth);

    status = vxSetParameterByIndex( node, (uint32_t)index, (vx_reference)spinst->sp );
    CHECK_STATUS_FAIL_GOTO( status, final );

final:
    if (spinst)
    {
        vsi_nn_release_spinst(&spinst);
    }

    return status;
}

vsi_nn_kernel_node_t vsi_nn_sp_l2norm_z_direction_times_node
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
    vx_tensor inputs_tensor[2] = {NULL, NULL};
    vx_tensor outputs_tensor[1] = {NULL};
    vx_node node = NULL;
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth;
    int32_t fifo_depth = 5;
    vsi_status status = VSI_FAILURE;

    vsi_nn_spinst_t *spinst = NULL;

    spinst = vsi_nn_sp_l2norm_z_direction_times_inst(graph->ctx->c, fifo_depth, max_vector_depth);

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
        vxAssignNodeQueryCallback(node, l2norm_z_direction_times_query);
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

/*
** This program requires sum operation in the z dimension.
** Instead of using the SUM Engine, the sum needs to be performed
** by Stream Processor instructions.
*/
vsi_nn_kernel_node_t l2_norm_z_direction
    (
    vsi_nn_graph_t              * graph,
    vsi_nn_tensor_t            ** inputs,
    vsi_nn_tensor_t            ** outputs,
    const vsi_nn_kernel_param_t * params
    )
{
    vsi_nn_kernel_node_t nodes[3] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_t * dummy_tensor[3] = {NULL};
    vsi_nn_tensor_t * output_tensor[1] = {NULL};
    int32_t axis = 2;
    float output_scale = vsi_nn_get_tensor_scale(outputs[0]);

    memcpy( &attr, &outputs[0]->attr, sizeof(vsi_nn_tensor_attr_t) );
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    attr.is_const = FALSE;
    attr.vtl = TRUE;
    attr.size[axis] = 1;
    dummy_tensor[0] = vsi_nn_create_dummy_tensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[0], "Create dummy_tensor fail.", final );
    dummy_tensor[1] = vsi_nn_create_dummy_tensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[1], "Create dummy_tensor fail.", final );
    dummy_tensor[2] = vsi_nn_create_dummy_tensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[2], "Create dummy_tensor fail.", final );

    memcpy( &attr, &outputs[0]->attr, sizeof(vsi_nn_tensor_attr_t) );
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    attr.is_const = FALSE;
    attr.vtl = TRUE;
    output_tensor[0] = vsi_nn_CreateTensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( output_tensor[0], "Create tensor fail.", final );

    nodes[0] = vsi_nn_sp_l2norm_z_direction_square_node(graph, inputs[0],
        output_tensor[0], dummy_tensor[0], "l2norm_0");
    CHECK_PTR_FAIL_GOTO( nodes[0], "Create l2norm_z_direction_square fail.", final );
    nodes[1] = vsi_nn_sp_l2norm_z_direction_rsqrt_node(graph, dummy_tensor[0],
        dummy_tensor[1], "l2norm_1");
    CHECK_PTR_FAIL_GOTO( nodes[1], "Create l2norm_z_direction_rsqrt fail.", final );
    nodes[2] = vsi_nn_sp_l2norm_z_direction_mul_node(graph, dummy_tensor[1],
        dummy_tensor[2], output_scale, "l2norm_2");
    CHECK_PTR_FAIL_GOTO( nodes[2], "Create l2norm_z_direction_mul fail.", final );
    node = vsi_nn_sp_l2norm_z_direction_times_node(graph, output_tensor[0],
        dummy_tensor[1], outputs[0], "l2norm_3");
    CHECK_PTR_FAIL_GOTO( node, "Create l2norm_z_direction_times fail.", final );

final:
    vsi_safe_release_node(nodes[0]);
    vsi_safe_release_node(nodes[1]);
    vsi_safe_release_node(nodes[2]);
    vsi_safe_release_tensor(dummy_tensor[0]);
    vsi_safe_release_tensor(dummy_tensor[1]);
    vsi_safe_release_tensor(dummy_tensor[2]);
    vsi_safe_release_tensor(output_tensor[0]);

    return node;
} /* softmax_z_direction() */

#endif
