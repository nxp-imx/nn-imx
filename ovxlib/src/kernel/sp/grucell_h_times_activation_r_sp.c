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

vsi_nn_kernel_node_t vsi_nn_sp_grucell_r_times_h_node
    (
        vsi_nn_graph_t  * graph,
        vsi_nn_tensor_t * input0,
        vsi_nn_tensor_t * input1,
        vsi_nn_tensor_t * output
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

    vsi_status status = VSI_FAILURE;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    /* loop inst0: r1 = pwlSetup(In) || r5 = pwlMul() || r2 = pwlAdd() */
    status  = vsi_nn_sp_pwl_sigmoid(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_PWLMUL, VSI_NN_SP_PWLMUL, VSI_NN_SP_SR5);
    status |= vsi_nn_sp_sub(&sp_insts_param[0], VSI_NN_SP_PWLADD, VSI_NN_SP_PWLADD, VSI_NN_SP_SR2);
    /* loop inst1: r6 = r5 * r2 || r7 = v12 + r6 || v11 = in */
    status |= vsi_nn_sp_mul(&sp_insts_param[1], VSI_NN_SP_SR5, VSI_NN_SP_SR2, VSI_NN_SP_SR6);
    status |= vsi_nn_sp_add(&sp_insts_param[1], VSI_NN_SP_VR12, VSI_NN_SP_SR6, VSI_NN_SP_SR7);
    status |= vsi_nn_sp_move(&sp_insts_param[1], VSI_NN_SP_SRIN, VSI_NN_SP_VR11);
    /* loop inst2: out = v11 * r7 || v12 = r1*/
    status |= vsi_nn_sp_mul(&sp_insts_param[2], VSI_NN_SP_VR11, VSI_NN_SP_SR7, VSI_NN_SP_SROUT);
    status |= vsi_nn_sp_move(&sp_insts_param[2], VSI_NN_SP_SR1, VSI_NN_SP_VR12);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;

    attr.input_setup = VSI_NN_SP_INPUT_SETUP_INTERLEAVE_TWO_INPUT;
    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.flush_cycle_num = 13;
    attr.ignored_leading_outputs = 4;
    attr.ignored_leading_v11_wr = 0;
    attr.ignored_leading_v11_rd = 4;
    attr.ignored_leading_v12_wr = 1;
    attr.ignored_leading_v12_rd = 3;
    attr.v11_reset_at_start = VX_SP_ATTRIBUTE_V_RESET_AT_START_RESET;
    attr.v12_reset_at_start = VX_SP_ATTRIBUTE_V_RESET_AT_START_RESET;

    attr.num_of_v11_rd_in_flush_cycle = 5;
    attr.num_of_v11_wr_in_flush_cycle = 0;
    attr.num_of_v12_rd_in_flush_cycle = 3;
    attr.num_of_v12_wr_in_flush_cycle = 2;

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

vsi_nn_kernel_node_t vsi_nn_sp_grucell_r_times_h_qnt_node
    (
        vsi_nn_graph_t  * graph,
        vsi_nn_tensor_t * input0,
        vsi_nn_tensor_t * input1,
        vsi_nn_tensor_t * output
    )
{
    const int32_t spInitInstsNum = 0;
    const int32_t spLoopInstsNum = 4;
    const int32_t spInstsNum = spInitInstsNum + spLoopInstsNum;

    const uint32_t input_count = 2;
    const uint32_t output_count = 1;
    vx_tensor inputs_tensor[2] = {NULL};
    vx_tensor outputs_tensor[1] = {NULL};
    vx_node node = NULL;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[4];
    vsi_nn_spinst_attr_t attr;
    float hstate_scale = vsi_nn_get_tensor_scale(input1);
    float const2 = -hstate_scale * (float)vsi_nn_get_tensor_zero_point(input1);

    vsi_status status = VSI_FAILURE;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);
    /* loop inst0: r1 = pwlSetup(In) || out = v11 * r9 || v11 = r8 + r7 || v12 = r1 */
    status  = vsi_nn_sp_pwl_sigmoid(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_VR11, VSI_NN_SP_SR9, VSI_NN_SP_SROUT);
    status |= vsi_nn_sp_add(&sp_insts_param[0], VSI_NN_SP_SR8, VSI_NN_SP_SR7, VSI_NN_SP_VR11);
    status |= vsi_nn_sp_move(&sp_insts_param[0], VSI_NN_SP_SR1, VSI_NN_SP_VR12);
    /* loop inst1: r8 = in * r4 || r9 = v12 + r6 */
    status |= vsi_nn_sp_mul(&sp_insts_param[1], VSI_NN_SP_SRIN, VSI_NN_SP_SR4, VSI_NN_SP_SR8);
    status |= vsi_nn_sp_add(&sp_insts_param[1], VSI_NN_SP_VR12, VSI_NN_SP_SR6, VSI_NN_SP_SR9);
    /* loop inst2: out = r6 = r5 * r2 */
    status |= vsi_nn_sp_mul(&sp_insts_param[2], VSI_NN_SP_SR5, VSI_NN_SP_SR2, VSI_NN_SP_SR6);
    /* loop inst3: r5 = pwlMul() || r2 = pwlAdd() */
    status |= vsi_nn_sp_mul(&sp_insts_param[3], VSI_NN_SP_PWLMUL, VSI_NN_SP_PWLMUL, VSI_NN_SP_SR5);
    status |= vsi_nn_sp_sub(&sp_insts_param[3], VSI_NN_SP_PWLADD, VSI_NN_SP_PWLADD, VSI_NN_SP_SR2);

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;

    attr.input_setup = VSI_NN_SP_INPUT_SETUP_INTERLEAVE_TWO_INPUT;
    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.flush_cycle_num = 11;
    attr.ignored_leading_outputs = 3;
    attr.ignored_leading_v11_wr = 1;
    attr.ignored_leading_v11_rd = 3;
    attr.ignored_leading_v12_wr = 1;
    attr.ignored_leading_v12_rd = 2;
    attr.v11_reset_at_start = VX_SP_ATTRIBUTE_V_RESET_AT_START_RESET;
    attr.v12_reset_at_start = VX_SP_ATTRIBUTE_V_RESET_AT_START_RESET;

    attr.num_of_v11_rd_in_flush_cycle = 3;
    attr.num_of_v11_wr_in_flush_cycle = 1;
    attr.num_of_v12_rd_in_flush_cycle = 2;
    attr.num_of_v12_wr_in_flush_cycle = 1;

    VSI_NN_SP_ATTR_SET_CONST_TO_SR4(attr, hstate_scale);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR7(attr, const2);

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

vsi_nn_kernel_node_t vsi_nn_sp_add_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input0,
        vsi_nn_tensor_t             * input1,
        vsi_nn_tensor_t             * output
    );

#define REGISTER_GRUCELL_ACTIVATION_STREAM_PROCESSOR_KERNEL( kernel_name )   \
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

REGISTER_GRUCELL_ACTIVATION_STREAM_PROCESSOR_KERNEL( grucell_h_times_activation_r )
{
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_tensor_attr_t attr;
    int32_t  recurrent_activation;
    vsi_nn_tensor_t *gate_r = NULL;
    float hstate_scale = vsi_nn_get_tensor_scale(inputs[0]);
    float const2 = (float)vsi_nn_get_tensor_zero_point(inputs[0]);

    recurrent_activation = vsi_nn_kernel_param_get_int32( params, "recurrent_activation" );

    if ( recurrent_activation != VSI_NN_ACT_SIGMOID )
    {
        return NULL;
    }

    memcpy( &attr, &inputs[0]->attr, sizeof(vsi_nn_tensor_attr_t) );
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    attr.is_const = FALSE;
    attr.vtl = TRUE;
    gate_r = vsi_nn_CreateTensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( gate_r, "Create tensor fail.", final );

    if (hstate_scale == 1 && const2 == 0)
    {
        node = vsi_nn_sp_grucell_r_times_h_node(graph, gate_r, inputs[0], outputs[0]);
    }
    else
    {
        node = vsi_nn_sp_grucell_r_times_h_qnt_node(graph, gate_r, inputs[0], outputs[0]);
    }
    CHECK_PTR_FAIL_GOTO( node, "Create grucell activation sp add node fail.", final );

final:
    vsi_safe_release_tensor(gate_r);
    return node;
} /* grucell_h_times_activation_r() */

#undef REGISTER_GRUCELL_ACTIVATION_STREAM_PROCESSOR_KERNEL

#endif
