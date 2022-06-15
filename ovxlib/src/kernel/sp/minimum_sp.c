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

vsi_nn_kernel_node_t vsi_nn_sp_minimum_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input0,
        vsi_nn_tensor_t             * input1,
        vsi_nn_tensor_t             * output
    )
{
    const int32_t spInitInstsNum = 0;
    const int32_t spLoopInstsNum = 3;
    const int32_t spInstsNum = spInitInstsNum + spLoopInstsNum;

    const uint32_t input_count = 2;
    const uint32_t output_count = 1;
    vx_tensor inputs_tensor[2] = {NULL};
    vx_tensor outputs_tensor[1] = {NULL};
    vx_node node = NULL;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[3];
    vsi_nn_spinst_attr_t attr;

    float input0_scale = vsi_nn_get_tensor_scale(input0);
    float input1_scale = vsi_nn_get_tensor_scale(input1);
    float output_scale = vsi_nn_get_tensor_scale(output);
    float scale0 = input0_scale / output_scale;
    float scale1 = input1_scale / output_scale;
    float const2 = -scale1 * (float)vsi_nn_get_tensor_zero_point(input1);

    vsi_status status = VSI_FAILURE;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    memset(&attr, 0, sizeof(vsi_nn_spinst_attr_t));

    /* loop inst0: v12 = in0 * r3 || r2 = v11 + r6 || r10 = r2 */
    status  = vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_SR3, VSI_NN_SP_VR12);
    status |= vsi_nn_sp_add(&sp_insts_param[0], VSI_NN_SP_VR11, VSI_NN_SP_SR6, VSI_NN_SP_SR2);
    status |= vsi_nn_sp_move(&sp_insts_param[0], VSI_NN_SP_SR2, VSI_NN_SP_SR10);
    /* loop inst1: v11 = in1 * r4 || r7 = r2 - v12 || r9 = v12 */
    status |= vsi_nn_sp_mul(&sp_insts_param[1], VSI_NN_SP_SRIN, VSI_NN_SP_SR4, VSI_NN_SP_VR11);
    status |= vsi_nn_sp_sub(&sp_insts_param[1], VSI_NN_SP_SR2, VSI_NN_SP_VR12, VSI_NN_SP_SR7);
    status |= vsi_nn_sp_move(&sp_insts_param[1], VSI_NN_SP_VR12, VSI_NN_SP_SR9);
    /* loop inst2: out = r7 > 0 ? r9 : r10 */
    status |= vsi_nn_sp_move_sel0(&sp_insts_param[2], VSI_NN_SP_SR7, VSI_NN_SP_SR9, VSI_NN_SP_SROUT);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;
    attr.input_setup = VSI_NN_SP_INPUT_SETUP_INTERLEAVE_TWO_INPUT;
    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.ignored_leading_outputs = 4;
    attr.flush_cycle_num = 13;
    attr.v11_reset_at_start = VSI_NN_SP_V_RESET_AT_START_RESET;
    attr.ignored_leading_v11_rd = 2;
    attr.ignored_leading_v11_wr = 0;
    attr.v12_reset_at_start = VSI_NN_SP_V_RESET_AT_START_RESET;
    attr.ignored_leading_v12_rd = 3;
    attr.ignored_leading_v12_wr = 0;

    VSI_NN_SP_ATTR_SET_CONST_TO_SR3(attr, scale0);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR4(attr, scale1);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR5_LOW_PRECISION(attr, 0);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR6(attr, const2);

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

#define REGISTER_MINIMUM_STREAM_PROCESSOR_KERNEL( kernel_name )   \
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

REGISTER_MINIMUM_STREAM_PROCESSOR_KERNEL( minimum )
{
    vsi_nn_kernel_node_t node = NULL;

    if (vsi_nn_is_broadcast_operaton(inputs, input_num, outputs[0]))
    {
        return NULL;
    }

    node = vsi_nn_sp_minimum_node(graph, inputs[0], inputs[1], outputs[0]);

    return node;
} /* minimum() */

#undef REGISTER_MINIMUM_STREAM_PROCESSOR_KERNEL

#endif
