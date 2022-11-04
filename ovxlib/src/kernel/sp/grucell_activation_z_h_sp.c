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

vsi_nn_kernel_node_t vsi_nn_sp_add_sigmoid_node
    (
        vsi_nn_graph_t  * graph,
        vsi_nn_tensor_t * input0,
        vsi_nn_tensor_t * input1,
        vsi_nn_tensor_t * output,
        uint8_t           dst_vr,
        char            * kernel_name
    );

vsi_nn_kernel_node_t vsi_nn_sp_add_tanh_node
    (
        vsi_nn_graph_t  * graph,
        vsi_nn_tensor_t * input0,
        vsi_nn_tensor_t * input1,
        vsi_nn_tensor_t * output,
        uint8_t           dst_vr,
        char            * kernel_name
    );

vsi_nn_kernel_node_t vsi_nn_sp_grucell_activation_z_h_node
    (
        vsi_nn_graph_t  * graph,
        vsi_nn_tensor_t * hstate_in,
        vsi_nn_tensor_t * dummy_in0,
        vsi_nn_tensor_t * dummy_in1,
        vsi_nn_tensor_t * output,
        char            * kernel_name
    )
{
    const int32_t spInitInstsNum = 0;
    const int32_t spLoopInstsNum = 3;
    const int32_t spInstsNum = spInitInstsNum + spLoopInstsNum;

    const uint32_t input_count = 3;
    const uint32_t output_count = 1;
    vx_tensor inputs_tensor[3] = {NULL};
    vx_tensor outputs_tensor[1] = {NULL};
    vx_node node = NULL;
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[3];
    vsi_nn_spinst_attr_t attr;
    float input_scale = vsi_nn_get_tensor_scale(hstate_in);
    float output_scale = vsi_nn_get_tensor_scale(output);

    vsi_status status = VSI_FAILURE;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    /* loop inst0: r1 = in * r3 | r2 = r1 - v12 | r5 = v12 */
    status  = vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_SR3, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_sub(&sp_insts_param[0], VSI_NN_SP_SR1, VSI_NN_SP_VR12, VSI_NN_SP_SR2);
    status |= vsi_nn_sp_move(&sp_insts_param[0], VSI_NN_SP_VR12, VSI_NN_SP_SR5);
    /* loop inst1: r6 = v11 * r2 | r7 = r6 + r8 | r8 = r5 */
    status |= vsi_nn_sp_mul(&sp_insts_param[1], VSI_NN_SP_VR11, VSI_NN_SP_SR2, VSI_NN_SP_SR6);
    status |= vsi_nn_sp_add(&sp_insts_param[1], VSI_NN_SP_SR6, VSI_NN_SP_SR8, VSI_NN_SP_SR7);
    status |= vsi_nn_sp_move(&sp_insts_param[1], VSI_NN_SP_SR5, VSI_NN_SP_SR8);
    /* loop inst2: out = r7 * r4 */
    status |= vsi_nn_sp_mul(&sp_insts_param[2], VSI_NN_SP_SR7, VSI_NN_SP_SR4, VSI_NN_SP_SROUT);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;

    attr.input_setup = VSI_NN_SP_INPUT_SETUP_SINGLE_INPUT;
    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.flush_cycle_num = 14;
    attr.ignored_leading_outputs = 4;
    attr.ignored_leading_v11_rd = 2;
    attr.ignored_leading_v12_rd = 1;

    attr.num_of_v11_rd_in_flush_cycle = 3;
    attr.num_of_v11_wr_in_flush_cycle = 0;
    attr.num_of_v12_rd_in_flush_cycle = 1;
    attr.num_of_v12_wr_in_flush_cycle = 0;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_XYZ;
    attr.split_max_vector_depth = max_vector_depth;

    spinst = vsi_nn_create_spinst(graph);
    CHECK_PTR_FAIL_GOTO( spinst, "Create spInst fail.", final );
    status  = vsi_nn_add_spinst_insts(spinst, sp_insts_param, spInstsNum);
    status |= vsi_nn_set_spinst_attr(spinst, attr);
    CHECK_STATUS_FAIL_GOTO(status, final );

    VSI_NN_SP_ATTR_SET_CONST_TO_SR3(attr, input_scale);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR4(attr, 1.0f / output_scale);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR5_LOW_PRECISION(attr, 1.0f);

    inputs_tensor[0] = hstate_in->t;
    inputs_tensor[1] = dummy_in0->t;
    inputs_tensor[2] = dummy_in1->t;
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

REGISTER_GRUCELL_ACTIVATION_STREAM_PROCESSOR_KERNEL( grucell_activation_z_h )
{
    vsi_nn_kernel_node_t node[4] = {NULL};
    vsi_nn_tensor_attr_t attr;
    int32_t recurrent_activation = vsi_nn_kernel_param_get_int32( params, "recurrent_activation" );
    int32_t activation = vsi_nn_kernel_param_get_int32( params, "activation" );
    vsi_nn_tensor_t * dummy_tensor[2] = {NULL};

    if ( recurrent_activation != VSI_NN_ACT_SIGMOID ||
         activation != VSI_NN_ACT_TANH )
    {
        return NULL;
    }

    memcpy( &attr, &inputs[0]->attr, sizeof(vsi_nn_tensor_attr_t) );
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    attr.is_const = FALSE;
    attr.vtl = TRUE;
    dummy_tensor[0] = vsi_nn_create_dummy_tensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[0], "Create tensor fail.", final );
    dummy_tensor[1] = vsi_nn_create_dummy_tensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[1], "Create tensor fail.", final );

    node[0] = vsi_nn_sp_add_sigmoid_node(graph, inputs[GRUCELL_ACT_Z_H_I_FC_Z], inputs[GRUCELL_ACT_Z_H_H_FC_Z],
        dummy_tensor[0], VSI_NN_SP_VR11, "grucell_activation_z_h_0" );
    CHECK_PTR_FAIL_GOTO( node[0], "Create grucell sp add sigmoid node fail.", final );
    node[1] = vsi_nn_sp_add_tanh_node(graph, inputs[GRUCELL_ACT_Z_H_I_FC_H], inputs[GRUCELL_ACT_Z_H_H_FC_H],
        dummy_tensor[1], VSI_NN_SP_VR12, "grucell_activation_z_h_1" );
    CHECK_PTR_FAIL_GOTO( node[1], "Create grucell sp add sigmoid node fail.", final );

    node[2] = vsi_nn_sp_grucell_activation_z_h_node(graph, inputs[GRUCELL_ACT_Z_H_HSTATE], dummy_tensor[0],
        dummy_tensor[1], outputs[GRUCELL_ACT_Z_H_OUT_OUTPUT], "grucell_activation_z_h_2");
    CHECK_PTR_FAIL_GOTO( node[2], "Create grucell sp z h node fail.", final );

    node[3] = vxTensorCopyNode( graph->g, outputs[0]->t, outputs[1]->t);
    CHECK_PTR_FAIL_GOTO( node[3], "Create grucell dataconvert node fail.", final );

final:
    vsi_safe_release_node(node[0]);
    vsi_safe_release_node(node[1]);
    vsi_safe_release_node(node[2]);
    vsi_safe_release_tensor(dummy_tensor[0]);
    vsi_safe_release_tensor(dummy_tensor[1]);

    return node[3];
} /* grucell_activation_z_h() */

#undef REGISTER_GRUCELL_ACTIVATION_STREAM_PROCESSOR_KERNEL

#endif
