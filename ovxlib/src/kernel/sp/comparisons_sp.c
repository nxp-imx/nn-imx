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
__BEGIN_DECLS

vsi_nn_kernel_node_t vsi_nn_sp_greater_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input0,
        vsi_nn_tensor_t             * input1,
        vsi_nn_tensor_t             * output
    )
{
    const int32_t spInitInstsNum = 3;
    const int32_t spLoopInstsNum = 2;
    const int32_t spInstsNum = spInitInstsNum + spLoopInstsNum;

    const uint32_t input_count = 2;
    const uint32_t output_count = 1;
    vx_tensor inputs_tensor[2] = {NULL};
    vx_tensor outputs_tensor[1] = {NULL};
    vx_node node = NULL;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[5];
    vsi_nn_spinst_attr_t attr;

    vsi_status status = VSI_FAILURE;
    float input0_scale = vsi_nn_get_tensor_scale(input0);
    float input1_scale = vsi_nn_get_tensor_scale(input1);
    float scale0 = input0_scale;
    float scale1 = input1_scale;
    float const2 = scale1 * (float)vsi_nn_get_tensor_zero_point(input1);
    float clamp_min = 0;
    float clamp_max = 0;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    vsi_nn_get_tensor_clamp_min_max(input0, &clamp_min, &clamp_max);
    clamp_min = clamp_min * scale0;
    clamp_max = clamp_max * scale0;

    /* init inst0: r5 = const2 */
    status  = vsi_nn_sp_move_constant(&sp_insts_param[0], const2, VSI_NN_SP_SR5);
    /* init inst1: r9 = 1 */
    status |= vsi_nn_sp_move_constant(&sp_insts_param[1], 1, VSI_NN_SP_SR9);
    /* init inst1: r10 = 0 */
    status |= vsi_nn_sp_move_constant(&sp_insts_param[2], 0, VSI_NN_SP_SR10);
    /* loop inst0: r1 = clamp(in * r3, r7, r6) | r8 = r1 - r2 | out = r8 ? r9 : r10 */
    status |= vsi_nn_sp_mul_clamp(&sp_insts_param[3], VSI_NN_SP_SRIN, VSI_NN_SP_SR3, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_sub(&sp_insts_param[3], VSI_NN_SP_SR1, VSI_NN_SP_SR2, VSI_NN_SP_SR8);
    status |= vsi_nn_sp_move_sel0(&sp_insts_param[3], VSI_NN_SP_SR8, VSI_NN_SP_SR9, VSI_NN_SP_SROUT);
    /* loop inst1: r2 = in * r4 | r8 = r8 + r5 */
    status |= vsi_nn_sp_mul(&sp_insts_param[4], VSI_NN_SP_SRIN, VSI_NN_SP_SR4, VSI_NN_SP_SR2);
    status |= vsi_nn_sp_add(&sp_insts_param[4], VSI_NN_SP_SR8, VSI_NN_SP_SR5, VSI_NN_SP_SR8);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.ignored_leading_outputs = 5;
    attr.flush_cycle_num = 9;

    VSI_NN_SP_ATTR_SET_CONST_TO_SR3(attr, scale0);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR4(attr, scale1);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR6(attr, clamp_max);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR7(attr, clamp_min);

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;
    attr.input_setup = VSI_NN_SP_INPUT_SETUP_INTERLEAVE_TWO_INPUT;

    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;

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

final:
    if (spinst)
    {
        vsi_nn_release_spinst(&spinst);
    }

    return (vsi_nn_kernel_node_t)node;
}

vsi_nn_kernel_node_t vsi_nn_sp_greater_equal_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input0,
        vsi_nn_tensor_t             * input1,
        vsi_nn_tensor_t             * output
    )
{
    const int32_t spInitInstsNum = 3;
    const int32_t spLoopInstsNum = 2;
    const int32_t spInstsNum = spInitInstsNum + spLoopInstsNum;

    const uint32_t input_count = 2;
    const uint32_t output_count = 1;
    vx_tensor inputs_tensor[2] = {NULL};
    vx_tensor outputs_tensor[1] = {NULL};
    vx_node node = NULL;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[5];
    vsi_nn_spinst_attr_t attr;

    vsi_status status = VSI_FAILURE;
    float input0_scale = vsi_nn_get_tensor_scale(input0);
    float input1_scale = vsi_nn_get_tensor_scale(input1);
    float scale0 = input0_scale;
    float scale1 = input1_scale;
    float const2 = scale1 * (float)vsi_nn_get_tensor_zero_point(input1);
    float clamp_min = 0;
    float clamp_max = 0;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    vsi_nn_get_tensor_clamp_min_max(input0, &clamp_min, &clamp_max);
    clamp_min = clamp_min * scale0;
    clamp_max = clamp_max * scale0;

    /* init inst0: r5 = const2 */
    status  = vsi_nn_sp_move_constant(&sp_insts_param[0], const2, VSI_NN_SP_SR5);
    /* init inst1: r9 = 0 */
    status |= vsi_nn_sp_move_constant(&sp_insts_param[1], 0, VSI_NN_SP_SR9);
    /* init inst1: r10 = 1 */
    status |= vsi_nn_sp_move_constant(&sp_insts_param[2], 1, VSI_NN_SP_SR10);
    /* loop inst0: r1 = clamp(in * r3, r7, r6) | r8 = r2 - r1 | out = r8 ? r9 : r10 */
    status |= vsi_nn_sp_mul_clamp(&sp_insts_param[3], VSI_NN_SP_SRIN, VSI_NN_SP_SR3, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_sub(&sp_insts_param[3], VSI_NN_SP_SR2, VSI_NN_SP_SR1, VSI_NN_SP_SR8);
    status |= vsi_nn_sp_move_sel0(&sp_insts_param[3], VSI_NN_SP_SR8, VSI_NN_SP_SR9, VSI_NN_SP_SROUT);
    /* loop inst1: r2 = in * r4 | r8 = r8 - r5 */
    status |= vsi_nn_sp_mul(&sp_insts_param[4], VSI_NN_SP_SRIN, VSI_NN_SP_SR4, VSI_NN_SP_SR2);
    status |= vsi_nn_sp_sub(&sp_insts_param[4], VSI_NN_SP_SR8, VSI_NN_SP_SR5, VSI_NN_SP_SR8);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.ignored_leading_outputs = 5;
    attr.flush_cycle_num = 9;

    VSI_NN_SP_ATTR_SET_CONST_TO_SR3(attr, scale0);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR4(attr, scale1);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR6(attr, clamp_max);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR7(attr, clamp_min);

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;
    attr.input_setup = VSI_NN_SP_INPUT_SETUP_INTERLEAVE_TWO_INPUT;

    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;

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

final:
    if (spinst)
    {
        vsi_nn_release_spinst(&spinst);
    }

    return (vsi_nn_kernel_node_t)node;
}

vsi_nn_kernel_node_t vsi_nn_sp_not_equal_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input0,
        vsi_nn_tensor_t             * input1,
        vsi_nn_tensor_t             * output
    )
{
    const int32_t spInitInstsNum = 3;
    const int32_t spLoopInstsNum = 2;
    const int32_t spInstsNum = spInitInstsNum + spLoopInstsNum;

    const uint32_t input_count = 2;
    const uint32_t output_count = 1;
    vx_tensor inputs_tensor[2] = {NULL};
    vx_tensor outputs_tensor[1] = {NULL};
    vx_node node = NULL;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[5];
    vsi_nn_spinst_attr_t attr;

    vsi_status status = VSI_FAILURE;
    float input0_scale = vsi_nn_get_tensor_scale(input0);
    float input1_scale = vsi_nn_get_tensor_scale(input1);
    float scale0 = input0_scale;
    float scale1 = input1_scale;
    float const2 = scale1 * (float)vsi_nn_get_tensor_zero_point(input1);
    float clamp_min = 0;
    float clamp_max = 0;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    vsi_nn_get_tensor_clamp_min_max(input0, &clamp_min, &clamp_max);
    clamp_min = clamp_min * scale0;
    clamp_max = clamp_max * scale0;

    /* init inst0: r5 = const2 */
    status  = vsi_nn_sp_move_constant(&sp_insts_param[0], const2, VSI_NN_SP_SR5);
    /* init inst1: r9 = 1 */
    status |= vsi_nn_sp_move_constant(&sp_insts_param[1], 1, VSI_NN_SP_SR9);
    /* init inst1: r10 = 0 */
    status |= vsi_nn_sp_move_constant(&sp_insts_param[2], 0, VSI_NN_SP_SR10);
    /* loop inst0: v11 = clamp(in * r3, r7, r6) | r8 = r2 - r5 | r1 = abs(r8) */
    status |= vsi_nn_sp_mul_clamp(&sp_insts_param[3], VSI_NN_SP_SRIN, VSI_NN_SP_SR3, VSI_NN_SP_VR11);
    status |= vsi_nn_sp_sub(&sp_insts_param[3], VSI_NN_SP_SR2, VSI_NN_SP_SR5, VSI_NN_SP_SR8);
    status |= vsi_nn_sp_abs(&sp_insts_param[3], VSI_NN_SP_SR8, VSI_NN_SP_SR1);
    /* loop inst1: r2 = in * r4 | r8 = v11 - r8 | out = r1 ? r9 : r10 */
    status |= vsi_nn_sp_mul(&sp_insts_param[4], VSI_NN_SP_SRIN, VSI_NN_SP_SR4, VSI_NN_SP_SR2);
    status |= vsi_nn_sp_sub(&sp_insts_param[4], VSI_NN_SP_VR11, VSI_NN_SP_SR8, VSI_NN_SP_SR8);
    status |= vsi_nn_sp_move_sel0(&sp_insts_param[4], VSI_NN_SP_SR1, VSI_NN_SP_SR9, VSI_NN_SP_SROUT);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.ignored_leading_outputs = 6;
    attr.flush_cycle_num = 12;
    attr.v11_reset_at_start = VSI_NN_SP_V_RESET_AT_START_RESET;
    attr.ignored_leading_v11_rd = 3;
    attr.num_of_v11_rd_in_flush_cycle = 3;

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;
    attr.input_setup = VSI_NN_SP_INPUT_SETUP_INTERLEAVE_TWO_INPUT;

    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;

    VSI_NN_SP_ATTR_SET_CONST_TO_SR3(attr, scale0);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR4(attr, scale1);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR6(attr, clamp_max);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR7(attr, clamp_min);

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

final:
    if (spinst)
    {
        vsi_nn_release_spinst(&spinst);
    }

    return (vsi_nn_kernel_node_t)node;
}

vsi_nn_kernel_node_t vsi_nn_sp_equal_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input0,
        vsi_nn_tensor_t             * input1,
        vsi_nn_tensor_t             * output
    )
{
    const int32_t spInitInstsNum = 3;
    const int32_t spLoopInstsNum = 2;
    const int32_t spInstsNum = spInitInstsNum + spLoopInstsNum;

    const uint32_t input_count = 2;
    const uint32_t output_count = 1;
    vx_tensor inputs_tensor[2] = {NULL};
    vx_tensor outputs_tensor[1] = {NULL};
    vx_node node = NULL;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[5];
    vsi_nn_spinst_attr_t attr;

    vsi_status status = VSI_FAILURE;
    float input0_scale = vsi_nn_get_tensor_scale(input0);
    float input1_scale = vsi_nn_get_tensor_scale(input1);
    float scale0 = input0_scale;
    float scale1 = input1_scale;
    float const2 = scale1 * (float)vsi_nn_get_tensor_zero_point(input1);
    float clamp_min = 0;
    float clamp_max = 0;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    vsi_nn_get_tensor_clamp_min_max(input0, &clamp_min, &clamp_max);
    clamp_min = clamp_min * scale0;
    clamp_max = clamp_max * scale0;

    /* init inst0: r5 = const2 */
    status  = vsi_nn_sp_move_constant(&sp_insts_param[0], const2, VSI_NN_SP_SR5);
    /* init inst1: r9 = 0 */
    status |= vsi_nn_sp_move_constant(&sp_insts_param[1], 0, VSI_NN_SP_SR9);
    /* init inst1: r10 = 1 */
    status |= vsi_nn_sp_move_constant(&sp_insts_param[2], 1, VSI_NN_SP_SR10);
    /* loop inst0: v11 = clamp(in * r3, r7, r6) | r8 = r2 - r5 | r1 = abs(r8) */
    status |= vsi_nn_sp_mul_clamp(&sp_insts_param[3], VSI_NN_SP_SRIN, VSI_NN_SP_SR3, VSI_NN_SP_VR11);
    status |= vsi_nn_sp_sub(&sp_insts_param[3], VSI_NN_SP_SR2, VSI_NN_SP_SR5, VSI_NN_SP_SR8);
    status |= vsi_nn_sp_abs(&sp_insts_param[3], VSI_NN_SP_SR8, VSI_NN_SP_SR1);
    /* loop inst1: r2 = in * r4 | r8 = v11 - r8 | out = r1 ? r9 : r10 */
    status |= vsi_nn_sp_mul(&sp_insts_param[4], VSI_NN_SP_SRIN, VSI_NN_SP_SR4, VSI_NN_SP_SR2);
    status |= vsi_nn_sp_sub(&sp_insts_param[4], VSI_NN_SP_VR11, VSI_NN_SP_SR8, VSI_NN_SP_SR8);
    status |= vsi_nn_sp_move_sel0(&sp_insts_param[4], VSI_NN_SP_SR1, VSI_NN_SP_SR9, VSI_NN_SP_SROUT);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.ignored_leading_outputs = 6;
    attr.flush_cycle_num = 12;
    attr.v11_reset_at_start = VSI_NN_SP_V_RESET_AT_START_RESET;
    attr.ignored_leading_v11_rd = 3;
    attr.num_of_v11_rd_in_flush_cycle = 3;

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;
    attr.input_setup = VSI_NN_SP_INPUT_SETUP_INTERLEAVE_TWO_INPUT;

    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;

    VSI_NN_SP_ATTR_SET_CONST_TO_SR3(attr, scale0);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR4(attr, scale1);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR6(attr, clamp_max);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR7(attr, clamp_min);

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

final:
    if (spinst)
    {
        vsi_nn_release_spinst(&spinst);
    }

    return (vsi_nn_kernel_node_t)node;
}

#define REGISTER_COMPARISONS_STREAM_PROCESSOR_KERNEL( kernel_name )   \
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

REGISTER_COMPARISONS_STREAM_PROCESSOR_KERNEL( relational_ops )
{
    int32_t operation = vsi_nn_kernel_param_get_int32( params, "operation" );
    vsi_nn_kernel_node_t node = NULL;

    if ( vsi_nn_is_broadcast_operaton(inputs, input_num, outputs[0]) )
    {
        return NULL;
    }

    switch ( operation )
    {
    case VSI_NN_RELATIONAL_OPS_GREAT:
        node = vsi_nn_sp_greater_node(graph, inputs[0], inputs[1], outputs[0]);
        break;
    case VSI_NN_RELATIONAL_OPS_GREAT_EQUAL:
        node = vsi_nn_sp_greater_equal_node(graph, inputs[0], inputs[1], outputs[0]);
        break;
    case VSI_NN_RELATIONAL_OPS_LESS:
        node = vsi_nn_sp_greater_node(graph, inputs[1], inputs[0], outputs[0]);
        break;
    case VSI_NN_RELATIONAL_OPS_LESS_EQUAL:
        node = vsi_nn_sp_greater_equal_node(graph, inputs[1], inputs[0], outputs[0]);
        break;
    case VSI_NN_RELATIONAL_OPS_NOT_EQUAL:
        node = vsi_nn_sp_not_equal_node(graph, inputs[1], inputs[0], outputs[0]);
        break;
    case VSI_NN_RELATIONAL_OPS_EQUAL:
        node = vsi_nn_sp_equal_node(graph, inputs[1], inputs[0], outputs[0]);
        break;
    default:
        return NULL;
    }

    CHECK_PTR_FAIL_GOTO( node, "Create sp_comparisons node fail.", final );

final:

    return node;
} /* add() */


__END_DECLS

#undef REGISTER_COMPARISONS_STREAM_PROCESSOR_KERNEL

#endif
