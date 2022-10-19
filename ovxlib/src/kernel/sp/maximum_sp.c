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

#if (VX_STREAM_PROCESSOR_SUPPORT)

vsi_nn_kernel_node_t vsi_nn_sp_maximum_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input0,
        vsi_nn_tensor_t             * input1,
        vsi_nn_tensor_t             * output,
        char                        * kernel_name
    )
{
    const int32_t input_zp1 = vsi_nn_get_tensor_zero_point(input1);
    const int32_t spInitInstsNum = input_zp1 == 0 ? 0 : 1;
    const int32_t spLoopInstsNum = input_zp1 == 0 ? 2 : 3;
    const int32_t spInstsNum = spInitInstsNum + spLoopInstsNum;

    const uint32_t input_count = 2;
    const uint32_t output_count = 1;
    vx_tensor inputs_tensor[2] = {NULL};
    vx_tensor outputs_tensor[1] = {NULL};
    vx_node node = NULL;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[4];
    vsi_nn_spinst_attr_t attr;
    float clamp_min = 0;
    float clamp_max = 0;

    float input0_scale = vsi_nn_get_tensor_scale(input0);
    float input1_scale = vsi_nn_get_tensor_scale(input1);
    float output_scale = vsi_nn_get_tensor_scale(output);
    float scale0 = input0_scale / output_scale;
    float scale1 = input1_scale / output_scale;
    float const2 = -scale1 * (float)vsi_nn_get_tensor_zero_point(input1);

    vsi_status status = VSI_FAILURE;

    vsi_nn_get_tensor_clamp_min_max(input0, &clamp_min, &clamp_max);
    clamp_min = clamp_min * scale0;
    clamp_max = clamp_max * scale0;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    if (input_zp1 == 0)
    {
        /* loop inst0: v11 = clamp(in0 * r3) | r10 = r1 + r5 | out = r8 > 0 ? r9 : r10 */
        status  = vsi_nn_sp_mul_clamp(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_SR3, VSI_NN_SP_VR11);
        status |= vsi_nn_sp_add(&sp_insts_param[0], VSI_NN_SP_SR1, VSI_NN_SP_SR5, VSI_NN_SP_SR10);
        status |= vsi_nn_sp_move_sel0(&sp_insts_param[0], VSI_NN_SP_SR8, VSI_NN_SP_SR9, VSI_NN_SP_SROUT);
        /* loop inst1: r1 = in1 * r4 | r8 = v11 - r1 | r9 = v11 */
        status |= vsi_nn_sp_mul(&sp_insts_param[1], VSI_NN_SP_SRIN, VSI_NN_SP_SR4, VSI_NN_SP_SR1);
        status |= vsi_nn_sp_sub(&sp_insts_param[1], VSI_NN_SP_VR11, VSI_NN_SP_SR1, VSI_NN_SP_SR8);
        status |= vsi_nn_sp_move(&sp_insts_param[1], VSI_NN_SP_VR11, VSI_NN_SP_SR9);
        CHECK_STATUS_FAIL_GOTO(status, final );

        attr.ignored_leading_outputs = 4;
        attr.flush_cycle_num = 7;
        attr.v11_reset_at_start = VSI_NN_SP_V_RESET_AT_START_RESET;
        attr.ignored_leading_v11_rd = 2;
        attr.ignored_leading_v11_wr = 0;

        attr.num_of_v11_rd_in_flush_cycle = 2;
        attr.num_of_v11_wr_in_flush_cycle = 0;
    }
    else
    {
        /* init inst0: r5 = -input_scale1 * input_zp1 */
        status  = vsi_nn_sp_move_constant(&sp_insts_param[0], const2, VSI_NN_SP_SR5);
        /* loop inst0: v12 = clamp(in0 * r3) | r1 = v11 + r5 | r10 = r1 */
        status |= vsi_nn_sp_mul_clamp(&sp_insts_param[1], VSI_NN_SP_SRIN, VSI_NN_SP_SR3, VSI_NN_SP_VR12);
        status |= vsi_nn_sp_add(&sp_insts_param[1], VSI_NN_SP_VR11, VSI_NN_SP_SR5, VSI_NN_SP_SR1);
        status |= vsi_nn_sp_move(&sp_insts_param[1], VSI_NN_SP_SR1, VSI_NN_SP_SR10);
        /* loop inst1: v11 = in1 * r4 | r8 = v12 - r1 | r9 = v12 */
        status |= vsi_nn_sp_mul(&sp_insts_param[2], VSI_NN_SP_SRIN, VSI_NN_SP_SR4, VSI_NN_SP_VR11);
        status |= vsi_nn_sp_sub(&sp_insts_param[2], VSI_NN_SP_VR12, VSI_NN_SP_SR1, VSI_NN_SP_SR8);
        status |= vsi_nn_sp_move(&sp_insts_param[2], VSI_NN_SP_VR12, VSI_NN_SP_SR9);
        /* loop inst2: out = r8 > 0 ? r9 : r10 */
        status |= vsi_nn_sp_move_sel0(&sp_insts_param[3], VSI_NN_SP_SR8, VSI_NN_SP_SR9, VSI_NN_SP_SROUT);
        CHECK_STATUS_FAIL_GOTO(status, final );

        attr.ignored_leading_outputs = 4;
        attr.flush_cycle_num = 13;
        attr.v11_reset_at_start = VSI_NN_SP_V_RESET_AT_START_RESET;
        attr.ignored_leading_v11_rd = 2;
        attr.ignored_leading_v11_wr = 0;
        attr.v12_reset_at_start = VSI_NN_SP_V_RESET_AT_START_RESET;
        attr.ignored_leading_v12_rd = 3;
        attr.ignored_leading_v12_wr = 0;

        attr.num_of_v11_rd_in_flush_cycle = 2;
        attr.num_of_v11_wr_in_flush_cycle = 0;
        attr.num_of_v12_rd_in_flush_cycle = 3;
        attr.num_of_v12_wr_in_flush_cycle = 0;
    }

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;
    attr.input_setup = VSI_NN_SP_INPUT_SETUP_INTERLEAVE_TWO_INPUT;
    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;

    VSI_NN_SP_ATTR_SET_CONST_TO_SR3(attr, scale0);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR4(attr, scale1);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR5_LOW_PRECISION(attr, 0);
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

    status = vsi_nn_set_sp_kernel_name(node, kernel_name);
    CHECK_STATUS_FAIL_GOTO(status, final );

final:
    if (spinst)
    {
        vsi_nn_release_spinst(&spinst);
    }

    return (vsi_nn_kernel_node_t)node;
}

#define REGISTER_MAXIMUM_STREAM_PROCESSOR_KERNEL( kernel_name )   \
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

REGISTER_MAXIMUM_STREAM_PROCESSOR_KERNEL( maximum )
{
    vsi_nn_kernel_node_t node = NULL;

    if (vsi_nn_is_broadcast_operaton(inputs, input_num, outputs[0]))
    {
        return NULL;
    }

    node = vsi_nn_sp_maximum_node(graph, inputs[0], inputs[1], outputs[0], "maximum");

    return node;
} /* maximum() */

#undef REGISTER_MAXIMUM_STREAM_PROCESSOR_KERNEL

#endif
