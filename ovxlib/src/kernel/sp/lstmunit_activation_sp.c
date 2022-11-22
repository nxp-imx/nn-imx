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

vsi_nn_kernel_node_t vsi_nn_sp_sigmoid_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input,
        vsi_nn_tensor_t             * output,
        uint8_t                       dst_vr,
        char                        * kernel_name
    )
{
    const int32_t spLoopInstsNum = 2;
    const int32_t spInstsNum = spLoopInstsNum;

    const uint32_t input_count = 1;
    const uint32_t output_count = 1;
    vx_tensor inputs_tensor[1] = {NULL};
    vx_tensor outputs_tensor[1] = {NULL};
    vx_node node = NULL;
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[2];
    vsi_nn_spinst_attr_t attr;

    vsi_status status = VSI_FAILURE;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    /* loop inst0: r1 = sigmid(in) | r6 = r5 * r2 | vector = r4 + r6 | r3 = r1 */
    status  = vsi_nn_sp_pwl_sigmoid(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_SR5, VSI_NN_SP_SR2, VSI_NN_SP_SR6);
    status |= vsi_nn_sp_add(&sp_insts_param[0], VSI_NN_SP_SR4, VSI_NN_SP_SR6, dst_vr);
    status |= vsi_nn_sp_move(&sp_insts_param[0], VSI_NN_SP_SR1, VSI_NN_SP_SR3);
    /* loop inst1: r5 = pwlMul() | r2 = pwlAdd() | r4 = r3 */
    status |= vsi_nn_sp_mul(&sp_insts_param[1], VSI_NN_SP_PWLMUL, VSI_NN_SP_PWLMUL, VSI_NN_SP_SR5);
    status |= vsi_nn_sp_sub(&sp_insts_param[1], VSI_NN_SP_PWLADD, VSI_NN_SP_PWLADD, VSI_NN_SP_SR2);
    status |= vsi_nn_sp_move(&sp_insts_param[1], VSI_NN_SP_SR3, VSI_NN_SP_SR4);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;

    attr.input_setup = VSI_NN_SP_INPUT_SETUP_SINGLE_INPUT;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.flush_cycle_num = 10;
    if (dst_vr == VSI_NN_SP_VR11)
    {
        attr.ignored_leading_v11_wr = 5;
        attr.v11_reset_at_start = VX_SP_ATTRIBUTE_V_RESET_AT_START_RESET;

        attr.num_of_v11_wr_in_flush_cycle = 5;
    }
    else
    {
        attr.ignored_leading_v12_wr = 5;
        attr.v12_reset_at_start = VX_SP_ATTRIBUTE_V_RESET_AT_START_RESET;

        attr.num_of_v12_wr_in_flush_cycle = 5;
    }

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_XYZ;
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

vsi_nn_kernel_node_t vsi_nn_sp_linear_sigmoid_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input,
        vsi_nn_tensor_t             * output,
        float                         forget_bias,
        char                        * kernel_name
    )
{
    const int32_t spLoopInstsNum = 2;
    const int32_t spInstsNum = spLoopInstsNum;

    const uint32_t input_count = 1;
    const uint32_t output_count = 1;
    vx_tensor inputs_tensor[1] = {NULL};
    vx_tensor outputs_tensor[1] = {NULL};
    vx_node node = NULL;
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[2];
    vsi_nn_spinst_attr_t attr;

    vsi_nn_sp_lut_params sp_lut_params;
    vx_lut_params_s vx_lut_params;
    vsi_status status = VSI_FAILURE;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);
    memset(&sp_lut_params, 0, sizeof(vsi_nn_sp_lut_params));
    memset(&vx_lut_params, 0, sizeof(vx_lut_params_s));

    /* loop inst0: r1 = pwlSetup(in) | r6 = r5 * r2 | v12 = r4 + r6 | r3 = r1 */
    status  = vsi_nn_sp_pwl_setup0(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_SR5, VSI_NN_SP_SR2, VSI_NN_SP_SR6);
    status |= vsi_nn_sp_add(&sp_insts_param[0], VSI_NN_SP_SR4, VSI_NN_SP_SR6, VSI_NN_SP_VR12);
    status |= vsi_nn_sp_move(&sp_insts_param[0], VSI_NN_SP_SR1, VSI_NN_SP_SR3);
    /* loop inst1: r5 = pwlMul() | r2 = pwlAdd() | r4 = r3 */
    status |= vsi_nn_sp_mul(&sp_insts_param[1], VSI_NN_SP_PWLMUL, VSI_NN_SP_PWLMUL, VSI_NN_SP_SR5);
    status |= vsi_nn_sp_sub(&sp_insts_param[1], VSI_NN_SP_PWLADD, VSI_NN_SP_PWLADD, VSI_NN_SP_SR2);
    status |= vsi_nn_sp_move(&sp_insts_param[1], VSI_NN_SP_SR3, VSI_NN_SP_SR4);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;

    attr.input_setup = VSI_NN_SP_INPUT_SETUP_SINGLE_INPUT;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.ignored_leading_v12_wr = 5;
    attr.ignored_leading_v12_rd = 0;
    attr.flush_cycle_num = 10;
    attr.v12_reset_at_start = VX_SP_ATTRIBUTE_V_RESET_AT_START_RESET;

    attr.num_of_v12_wr_in_flush_cycle = 5;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_XYZ;
    attr.split_max_vector_depth = max_vector_depth;

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

    sp_lut_params.act_type = VSI_NN_SP_ACT_LINEAR_SIGMOID;
    sp_lut_params.params[0] = 1;
    sp_lut_params.params[1] = forget_bias;
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

vsi_nn_kernel_node_t vsi_nn_sp_add_sigmoid_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input0,
        vsi_nn_tensor_t             * input1,
        vsi_nn_tensor_t             * input2,
        vsi_nn_tensor_t             * dummy_output,
        uint8_t                       dst_vr,
        char                        * kernel_name
    )
{
    const int32_t spInitInstsNum = 0;
    const int32_t spLoopInstsNum = 3;
    const int32_t spInstsNum = spInitInstsNum + spLoopInstsNum;

    const uint32_t input_count = input2 == NULL ? 2 : 3;
    const uint32_t output_count = 1;
    vx_tensor inputs_tensor[3] = {NULL};
    vx_tensor outputs_tensor[1] = {NULL};
    vx_node node = NULL;
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[5];
    vsi_nn_spinst_attr_t attr;

    vsi_status status = VSI_FAILURE;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    /* loop inst0: r8 = in * r3 | v11/v12 = r4 + r6 | r7 = r1 */
    status  = vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_SR3, VSI_NN_SP_SR8);
    status |= vsi_nn_sp_add(&sp_insts_param[0], VSI_NN_SP_SR4, VSI_NN_SP_SR6, dst_vr);
    status |= vsi_nn_sp_move(&sp_insts_param[0], VSI_NN_SP_SR1, VSI_NN_SP_SR7);
    /* loop inst1: r1 = sigmoid(r10) | r6 = r5 * r2 | r10 = r8 + r9 | r9 = in*/
    status |= vsi_nn_sp_pwl_sigmoid(&sp_insts_param[1], VSI_NN_SP_SR10, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_mul(&sp_insts_param[1], VSI_NN_SP_SR5, VSI_NN_SP_SR2, VSI_NN_SP_SR6);
    status |= vsi_nn_sp_add(&sp_insts_param[1], VSI_NN_SP_SR8, VSI_NN_SP_SR9, VSI_NN_SP_SR10);
    status |= vsi_nn_sp_move(&sp_insts_param[1], VSI_NN_SP_SRIN, VSI_NN_SP_SR9);
    /* loop inst2: r5 = pwlMul() | r2 = pwlAdd() | r4 = r7 */
    status |= vsi_nn_sp_mul(&sp_insts_param[2], VSI_NN_SP_PWLMUL, VSI_NN_SP_PWLMUL, VSI_NN_SP_SR5);
    status |= vsi_nn_sp_sub(&sp_insts_param[2], VSI_NN_SP_PWLADD, VSI_NN_SP_PWLADD, VSI_NN_SP_SR2);
    status |= vsi_nn_sp_move(&sp_insts_param[2], VSI_NN_SP_SR7, VSI_NN_SP_SR4);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;

    attr.input_setup = VSI_NN_SP_INPUT_SETUP_INTERLEAVE_TWO_INPUT;
    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.flush_cycle_num = 20;
    VSI_NN_SP_ATTR_SET_CONST_TO_SR3(attr, 1.0f);
    if (dst_vr == VSI_NN_SP_VR11)
    {
        attr.ignored_leading_v11_wr = 7;
        attr.v11_reset_at_start = VX_SP_ATTRIBUTE_V_RESET_AT_START_RESET;

        attr.num_of_v11_wr_in_flush_cycle = 7;
    }
    else
    {
        attr.ignored_leading_v12_wr = 7;
        attr.v12_reset_at_start = VX_SP_ATTRIBUTE_V_RESET_AT_START_RESET;

        attr.num_of_v12_wr_in_flush_cycle = 7;
    }

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_XYZ;
    attr.split_max_vector_depth = max_vector_depth;

    spinst = vsi_nn_create_spinst(graph);
    CHECK_PTR_FAIL_GOTO( spinst, "Create spInst fail.", final );
    status  = vsi_nn_add_spinst_insts(spinst, sp_insts_param, spInstsNum);
    status |= vsi_nn_set_spinst_attr(spinst, attr);
    CHECK_STATUS_FAIL_GOTO(status, final );

    inputs_tensor[0] = input0->t;
    inputs_tensor[1] = input1->t;
    if (input2)
        inputs_tensor[2] = input2->t;
    outputs_tensor[0] = dummy_output->t;

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

vsi_nn_kernel_node_t vsi_nn_sp_add_tanh_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input0,
        vsi_nn_tensor_t             * input1,
        vsi_nn_tensor_t             * dummy_output,
        uint8_t                       dst_vr,
        char                        * kernel_name
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
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[5];
    vsi_nn_spinst_attr_t attr;

    vsi_status status = VSI_FAILURE;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    /* loop inst0: r8 = in * r3 || v11/v12 = r4 + r6 || r7 = r1 */
    status  = vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_SR3, VSI_NN_SP_SR8);
    status |= vsi_nn_sp_add(&sp_insts_param[0], VSI_NN_SP_SR4, VSI_NN_SP_SR6, dst_vr);
    status |= vsi_nn_sp_move(&sp_insts_param[0], VSI_NN_SP_SR1, VSI_NN_SP_SR7);
    /* loop inst1: r1 = pwl_tanh(r10) || r6 = r5 * r2 || r10 = r8 + r9 || r9 = in*/
    status |= vsi_nn_sp_pwl_tanh(&sp_insts_param[1], VSI_NN_SP_SR10, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_mul(&sp_insts_param[1], VSI_NN_SP_SR5, VSI_NN_SP_SR2, VSI_NN_SP_SR6);
    status |= vsi_nn_sp_add(&sp_insts_param[1], VSI_NN_SP_SR8, VSI_NN_SP_SR9, VSI_NN_SP_SR10);
    status |= vsi_nn_sp_move(&sp_insts_param[1], VSI_NN_SP_SRIN, VSI_NN_SP_SR9);
    /* loop inst2: r5 = pwlMul() || r2 = pwlAdd() || r7 = r4*/
    status |= vsi_nn_sp_mul(&sp_insts_param[2], VSI_NN_SP_PWLMUL, VSI_NN_SP_PWLMUL, VSI_NN_SP_SR5);
    status |= vsi_nn_sp_sub(&sp_insts_param[2], VSI_NN_SP_PWLADD, VSI_NN_SP_PWLADD, VSI_NN_SP_SR2);
    status |= vsi_nn_sp_move(&sp_insts_param[2], VSI_NN_SP_SR4, VSI_NN_SP_SR7);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;

    attr.input_setup = VSI_NN_SP_INPUT_SETUP_INTERLEAVE_TWO_INPUT;
    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.flush_cycle_num = 20;
    VSI_NN_SP_ATTR_SET_CONST_TO_SR3(attr, 1.0f);
    if (dst_vr == VSI_NN_SP_VR11)
    {
        attr.ignored_leading_v11_wr = 7;
        attr.v11_reset_at_start = VX_SP_ATTRIBUTE_V_RESET_AT_START_RESET;

        attr.num_of_v11_wr_in_flush_cycle = 7;
    }
    else
    {
        attr.ignored_leading_v12_wr = 7;
        attr.v12_reset_at_start = VX_SP_ATTRIBUTE_V_RESET_AT_START_RESET;

        attr.num_of_v12_wr_in_flush_cycle = 7;
    }

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_XYZ;
    attr.split_max_vector_depth = max_vector_depth;

    spinst = vsi_nn_create_spinst(graph);
    CHECK_PTR_FAIL_GOTO( spinst, "Create spInst fail.", final );
    status  = vsi_nn_add_spinst_insts(spinst, sp_insts_param, spInstsNum);
    status |= vsi_nn_set_spinst_attr(spinst, attr);
    CHECK_STATUS_FAIL_GOTO(status, final );

    inputs_tensor[0] = input0->t;
    inputs_tensor[1] = input1->t;
    outputs_tensor[0] = dummy_output->t;

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

vsi_nn_kernel_node_t vsi_nn_sp_add_sigmoid_ext_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input0,
        vsi_nn_tensor_t             * input1,
        vsi_nn_tensor_t             * dummy_output,
        float                         forget_bias,
        char                        * kernel_name
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
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth;
    float input0_scale = vsi_nn_get_tensor_scale(input0);

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[5];
    vsi_nn_spinst_attr_t attr;

    vsi_status status = VSI_FAILURE;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    /* loop inst0: r1 = sigmoid(r10) | r2 = in * r3 | r10 = r2 + r5 */
    status  = vsi_nn_sp_pwl_sigmoid(&sp_insts_param[0], VSI_NN_SP_SR10, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_SR3, VSI_NN_SP_SR2);
    status |= vsi_nn_sp_add(&sp_insts_param[0], VSI_NN_SP_SR2, VSI_NN_SP_SR5, VSI_NN_SP_SR10);
    /* loop inst1: r5 = in + r4 */
    status |= vsi_nn_sp_add(&sp_insts_param[1], VSI_NN_SP_SRIN, VSI_NN_SP_SR4, VSI_NN_SP_SR5);
    /* loop inst2: r9 = r6 * r7 | v12 = r8 + r9 | r8 = r1 */
    status |= vsi_nn_sp_mul(&sp_insts_param[2], VSI_NN_SP_SR6, VSI_NN_SP_SR7, VSI_NN_SP_SR9);
    status |= vsi_nn_sp_add(&sp_insts_param[2], VSI_NN_SP_SR8, VSI_NN_SP_SR9, VSI_NN_SP_VR12);
    status |= vsi_nn_sp_move(&sp_insts_param[2], VSI_NN_SP_SR1, VSI_NN_SP_SR8);
    /* loop inst3: r6 = pwlMul() | r7 = pwlAdd() */
    status |= vsi_nn_sp_mul(&sp_insts_param[3], VSI_NN_SP_PWLMUL, VSI_NN_SP_PWLMUL, VSI_NN_SP_SR6);
    status |= vsi_nn_sp_sub(&sp_insts_param[3], VSI_NN_SP_PWLADD, VSI_NN_SP_PWLADD, VSI_NN_SP_SR7);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;

    attr.input_setup = VSI_NN_SP_INPUT_SETUP_INTERLEAVE_TWO_INPUT;
    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.flush_cycle_num = 17;
    attr.ignored_leading_v12_wr = 4;
    attr.v12_reset_at_start = VX_SP_ATTRIBUTE_V_RESET_AT_START_RESET;

    attr.num_of_v12_wr_in_flush_cycle = 5;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_XYZ;
    attr.split_max_vector_depth = max_vector_depth;

    VSI_NN_SP_ATTR_SET_CONST_TO_SR3(attr, input0_scale);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR4(attr, forget_bias);

    spinst = vsi_nn_create_spinst(graph);
    CHECK_PTR_FAIL_GOTO( spinst, "Create spInst fail.", final );
    status  = vsi_nn_add_spinst_insts(spinst, sp_insts_param, spInstsNum);
    status |= vsi_nn_set_spinst_attr(spinst, attr);
    CHECK_STATUS_FAIL_GOTO(status, final );

    inputs_tensor[0] = input0->t;
    inputs_tensor[1] = input1->t;
    outputs_tensor[0] = dummy_output->t;

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

vsi_nn_kernel_node_t vsi_nn_sp_get_cell_status_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * c_gate_in,
        vsi_nn_tensor_t             * cell_status_in,
        vsi_nn_tensor_t             * dummy0_input,
        vsi_nn_tensor_t             * dummy1_input,
        vsi_nn_tensor_t             * dummy2_input,
        vsi_nn_tensor_t             * cell_status_out,
        vsi_nn_tensor_t             * dummy_output,
        char                        * kernel_name
    )
{
    const int32_t spInitInstsNum = 1;
    const int32_t spLoopInstsNum = 4;
    const int32_t spInstsNum = spInitInstsNum + spLoopInstsNum;

    const uint32_t input_count = dummy2_input == NULL ? 4 : 5;
    const uint32_t output_count = 2;
    vx_tensor inputs_tensor[5] = {NULL};
    vx_tensor outputs_tensor[2] = {NULL};
    vx_node node = NULL;
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[5];
    vsi_nn_spinst_attr_t attr;

    vsi_status status = VSI_FAILURE;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    /* init inst0: r10 = 0 */
    status  = vsi_nn_sp_move_constant(&sp_insts_param[0], 0, VSI_NN_SP_SR10);
    /* loop inst0: r1 = pwlSetup(in) | r7 = v12 * r6 | r9 = r5 + r8 | r4 = r1 */
    status |= vsi_nn_sp_pwl_tanh(&sp_insts_param[1], VSI_NN_SP_SRIN, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_mul(&sp_insts_param[1], VSI_NN_SP_VR12, VSI_NN_SP_SR6, VSI_NN_SP_SR7);
    status |= vsi_nn_sp_add(&sp_insts_param[1], VSI_NN_SP_SR5, VSI_NN_SP_SR8, VSI_NN_SP_SR9);
    status |= vsi_nn_sp_move(&sp_insts_param[1], VSI_NN_SP_SR1, VSI_NN_SP_SR4);
    /* loop inst1: r5 = r5 * v11 | out = r9 + r10 | v12 = r9 */
    status |= vsi_nn_sp_mul(&sp_insts_param[2], VSI_NN_SP_SR5, VSI_NN_SP_VR11, VSI_NN_SP_SR5);
    status |= vsi_nn_sp_add(&sp_insts_param[2], VSI_NN_SP_SR9, VSI_NN_SP_SR10, VSI_NN_SP_SROUT);
    status |= vsi_nn_sp_move(&sp_insts_param[2], VSI_NN_SP_SR9, VSI_NN_SP_VR12);
    /* loop inst2: r3 = r1 * r2 | r5 = r3 + r4 | r6 = in */
    status |= vsi_nn_sp_mul(&sp_insts_param[3], VSI_NN_SP_SR1, VSI_NN_SP_SR2, VSI_NN_SP_SR3);
    status |= vsi_nn_sp_add(&sp_insts_param[3], VSI_NN_SP_SR3, VSI_NN_SP_SR4, VSI_NN_SP_SR5);
    status |= vsi_nn_sp_move(&sp_insts_param[3], VSI_NN_SP_SRIN, VSI_NN_SP_SR6);
    /* loop inst3: r8 = pwlMul * pwlMul | r2 = pwlAdd + pwlAdd | r8 = r7 */
    status |= vsi_nn_sp_mul(&sp_insts_param[4], VSI_NN_SP_PWLMUL, VSI_NN_SP_PWLMUL, VSI_NN_SP_SR8);
    status |= vsi_nn_sp_sub(&sp_insts_param[4], VSI_NN_SP_PWLADD, VSI_NN_SP_PWLADD, VSI_NN_SP_SR2);
    status |= vsi_nn_sp_move(&sp_insts_param[4], VSI_NN_SP_SR7, VSI_NN_SP_SR8);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;

    attr.input_setup = VSI_NN_SP_INPUT_SETUP_INTERLEAVE_TWO_INPUT;
    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.ignored_leading_outputs = 5;
    attr.ignored_leading_v11_rd = 3;
    attr.ignored_leading_v12_rd = 2;
    attr.ignored_leading_v12_wr = 5;
    attr.flush_cycle_num = 19;

    attr.num_of_v11_rd_in_flush_cycle = 3;
    attr.num_of_v11_wr_in_flush_cycle = 0;
    attr.num_of_v12_rd_in_flush_cycle = 2;
    attr.num_of_v12_wr_in_flush_cycle = 5;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_XYZ;
    attr.split_max_vector_depth = max_vector_depth;

    spinst = vsi_nn_create_spinst(graph);
    CHECK_PTR_FAIL_GOTO( spinst, "Create spInst fail.", final );
    status  = vsi_nn_add_spinst_insts(spinst, sp_insts_param, spInstsNum);
    status |= vsi_nn_set_spinst_attr(spinst, attr);
    CHECK_STATUS_FAIL_GOTO(status, final );

    inputs_tensor[0] = c_gate_in->t;
    inputs_tensor[1] = cell_status_in->t;
    inputs_tensor[2] = dummy0_input->t;
    inputs_tensor[3] = dummy1_input->t;
    if (dummy2_input)
    {
        inputs_tensor[4] = dummy2_input->t;
    }

    outputs_tensor[0] = cell_status_out->t;
    outputs_tensor[1] = dummy_output->t;

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

vsi_nn_kernel_node_t vsi_nn_sp_get_float_output_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * dummy0_input,
        vsi_nn_tensor_t             * dummy1_input,
        vsi_nn_tensor_t             * output,
        char                        * kernel_name
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
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[3];
    vsi_nn_spinst_attr_t attr;

    vsi_status status = VSI_FAILURE;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    /* loop inst0: r1 = pwlSetup(v12) || r2 = pwlMul * pwlMul || r3 = pwlAdd + pwlAdd */
    status  = vsi_nn_sp_pwl_tanh(&sp_insts_param[0], VSI_NN_SP_VR12, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_PWLMUL, VSI_NN_SP_PWLMUL, VSI_NN_SP_SR2);
    status |= vsi_nn_sp_sub(&sp_insts_param[0], VSI_NN_SP_PWLADD, VSI_NN_SP_PWLADD, VSI_NN_SP_SR3);
    /* loop inst1: r5 = r2 * r3 || r6 = r5 + r4 */
    status |= vsi_nn_sp_mul(&sp_insts_param[1], VSI_NN_SP_SR2, VSI_NN_SP_SR3, VSI_NN_SP_SR5);
    status |= vsi_nn_sp_add(&sp_insts_param[1], VSI_NN_SP_SR5, VSI_NN_SP_SR4, VSI_NN_SP_SR6);
    /* loop inst2: out = r6 * v11 || r4 = r1 */
    status |= vsi_nn_sp_mul(&sp_insts_param[2], VSI_NN_SP_SR6, VSI_NN_SP_VR11, VSI_NN_SP_SROUT);
    status |= vsi_nn_sp_move(&sp_insts_param[2], VSI_NN_SP_SR1, VSI_NN_SP_SR4);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;

    attr.input_setup = VSI_NN_SP_INPUT_SETUP_V12;
    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.ignored_leading_outputs = 4;
    attr.ignored_leading_v11_rd = 4;
    attr.ignored_leading_v12_rd = 0;
    attr.flush_cycle_num = 14;

    attr.num_of_v11_rd_in_flush_cycle = 5;
    attr.num_of_v11_wr_in_flush_cycle = 0;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_XYZ;
    attr.split_max_vector_depth = max_vector_depth;

    spinst = vsi_nn_create_spinst(graph);
    CHECK_PTR_FAIL_GOTO( spinst, "Create spInst fail.", final );
    status  = vsi_nn_add_spinst_insts(spinst, sp_insts_param, spInstsNum);
    status |= vsi_nn_set_spinst_attr(spinst, attr);
    CHECK_STATUS_FAIL_GOTO(status, final );

    inputs_tensor[0] = dummy0_input->t;
    inputs_tensor[1] = dummy1_input->t;
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

vsi_nn_kernel_node_t vsi_nn_sp_get_quantized_output_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * dummy0_input,
        vsi_nn_tensor_t             * dummy1_input,
        vsi_nn_tensor_t             * output,
        char                        * kernel_name
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
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[4];
    vsi_nn_spinst_attr_t attr;
    float output_scale = vsi_nn_get_tensor_scale(output);

    vsi_status status = VSI_FAILURE;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    /* loop inst0: r1 = pwlSetup(v12) || r2 = pwlMul * pwlMul || r3 = pwlAdd - pwlAdd */
    status  = vsi_nn_sp_pwl_tanh(&sp_insts_param[0], VSI_NN_SP_VR12, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_PWLMUL, VSI_NN_SP_PWLMUL, VSI_NN_SP_SR2);
    status |= vsi_nn_sp_sub(&sp_insts_param[0], VSI_NN_SP_PWLADD, VSI_NN_SP_PWLADD, VSI_NN_SP_SR3);
    /* loop inst1: out = r6 * r8 */
    status |= vsi_nn_sp_mul(&sp_insts_param[1], VSI_NN_SP_SR6, VSI_NN_SP_SR8, VSI_NN_SP_SROUT);
    /* loop inst2: r8 = v11 * r7 || r6 = r5 + r4 || r4 = r1 */
    status |= vsi_nn_sp_mul(&sp_insts_param[2], VSI_NN_SP_VR11, VSI_NN_SP_SR7, VSI_NN_SP_SR8);
    status |= vsi_nn_sp_add(&sp_insts_param[2], VSI_NN_SP_SR5, VSI_NN_SP_SR4, VSI_NN_SP_SR6);
    status |= vsi_nn_sp_move(&sp_insts_param[2], VSI_NN_SP_SR1, VSI_NN_SP_SR4);
    /* loop inst3: r5 = r2 * r3 */
    status |= vsi_nn_sp_mul(&sp_insts_param[3], VSI_NN_SP_SR2, VSI_NN_SP_SR3, VSI_NN_SP_SR5);

    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;

    attr.input_setup = VSI_NN_SP_INPUT_SETUP_V12;
    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.ignored_leading_outputs = 3;
    attr.ignored_leading_v11_rd = 2;
    attr.ignored_leading_v12_rd = 0;
    attr.flush_cycle_num = 13;
    attr.num_of_v11_rd_in_flush_cycle = 3;
    attr.num_of_v11_wr_in_flush_cycle = 0;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_XYZ;
    attr.split_max_vector_depth = max_vector_depth;

    VSI_NN_SP_ATTR_SET_CONST_TO_SR7(attr, 1.0f / output_scale);

    spinst = vsi_nn_create_spinst(graph);
    CHECK_PTR_FAIL_GOTO( spinst, "Create spInst fail.", final );
    status  = vsi_nn_add_spinst_insts(spinst, sp_insts_param, spInstsNum);
    status |= vsi_nn_set_spinst_attr(spinst, attr);
    CHECK_STATUS_FAIL_GOTO(status, final );

    inputs_tensor[0] = dummy0_input->t;
    inputs_tensor[1] = dummy1_input->t;
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

vsi_nn_kernel_node_t vsi_nn_sp_get_output_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * dummy0_input,
        vsi_nn_tensor_t             * dummy1_input,
        vsi_nn_tensor_t             * output,
        char                        * kernel_name
    )
{
    float output_scale = vsi_nn_get_tensor_scale(output);
    vsi_nn_kernel_node_t node = NULL;

    if (output_scale == 1.0f)
    {
        node = vsi_nn_sp_get_float_output_node(graph, dummy0_input, dummy1_input, output, kernel_name);
    }
    else
    {
        node = vsi_nn_sp_get_quantized_output_node(graph, dummy0_input, dummy1_input, output, kernel_name);
    }

    return node;
}

vsi_nn_kernel_node_t vsi_nn_sp_cifg_linear_sigmoid_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input0,
        vsi_nn_tensor_t             * input1,
        vsi_nn_tensor_t             * output0,
        vsi_nn_tensor_t             * output1,
        float                         forget_bias,
        char                        * kernel_name
    )
{
    const int32_t spInitInstsNum = 1;
    const int32_t spLoopInstsNum = 3;
    const int32_t spInstsNum = spInitInstsNum + spLoopInstsNum;

    const uint32_t input_count = 2;
    const uint32_t output_count = 2;
    vx_tensor inputs_tensor[2] = {NULL};
    vx_tensor outputs_tensor[2] = {NULL};
    vx_node node = NULL;
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[4];
    vsi_nn_spinst_attr_t attr;

    vsi_nn_sp_lut_params sp_lut_params;
    vx_lut_params_s vx_lut_params;
    vsi_status status = VSI_FAILURE;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);
    memset(&sp_lut_params, 0, sizeof(vsi_nn_sp_lut_params));
    memset(&vx_lut_params, 0, sizeof(vx_lut_params_s));

    /* init inst0: r8 = 1 */
    status  = vsi_nn_sp_move_constant(&sp_insts_param[0], 1, VSI_NN_SP_SR8);
    /* loop inst0: r1 = pwlSetup(in) || r5 = pwlMul * pwlMul || r2 = pwlAdd + pwlAdd || v12 = r7*/
    status |= vsi_nn_sp_pwl_setup0(&sp_insts_param[1], VSI_NN_SP_SRIN, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_mul(&sp_insts_param[1], VSI_NN_SP_PWLMUL, VSI_NN_SP_PWLMUL, VSI_NN_SP_SR5);
    status |= vsi_nn_sp_sub(&sp_insts_param[1], VSI_NN_SP_PWLADD, VSI_NN_SP_PWLADD, VSI_NN_SP_SR2);
    status |= vsi_nn_sp_move(&sp_insts_param[1], VSI_NN_SP_SR7, VSI_NN_SP_VR12);
    /* loop inst1: r6 = r5 * r2 || v11 = r8 - r7 || r3 = r1*/
    status |= vsi_nn_sp_mul(&sp_insts_param[2], VSI_NN_SP_SR5, VSI_NN_SP_SR2, VSI_NN_SP_SR6);
    status |= vsi_nn_sp_sub(&sp_insts_param[2], VSI_NN_SP_SR8, VSI_NN_SP_SR7, VSI_NN_SP_VR11);
    status |= vsi_nn_sp_move(&sp_insts_param[2], VSI_NN_SP_SR1, VSI_NN_SP_SR3);
    /* loop inst1: r7 = r4 + r6 || r4 = r3*/
    status |= vsi_nn_sp_add(&sp_insts_param[3], VSI_NN_SP_SR4, VSI_NN_SP_SR6, VSI_NN_SP_SR7);
    status |= vsi_nn_sp_move(&sp_insts_param[3], VSI_NN_SP_SR3, VSI_NN_SP_SR4);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;

    attr.input_setup = VSI_NN_SP_INPUT_SETUP_SINGLE_INPUT;
    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.ignored_leading_v11_wr = 5;
    attr.ignored_leading_v12_wr = 5;
    attr.flush_cycle_num = 16;
    attr.v11_reset_at_start = VX_SP_ATTRIBUTE_V_RESET_AT_START_RESET;
    attr.v12_reset_at_start = VX_SP_ATTRIBUTE_V_RESET_AT_START_RESET;

    attr.num_of_v11_rd_in_flush_cycle = 0;
    attr.num_of_v11_wr_in_flush_cycle = 6;
    attr.num_of_v12_rd_in_flush_cycle = 0;
    attr.num_of_v12_wr_in_flush_cycle = 5;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_XYZ;
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

    sp_lut_params.act_type = VSI_NN_SP_ACT_LINEAR_SIGMOID;
    sp_lut_params.params[0] = 1;
    sp_lut_params.params[1] = forget_bias;
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

vsi_nn_kernel_node_t vsi_nn_sp_lstm_add_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input0,
        vsi_nn_tensor_t             * input1,
        vsi_nn_tensor_t             * output0,
        vsi_nn_tensor_t             * output1,
        char                        * kernel_name
    )
{
    const int32_t spInitInstsNum = 0;
    const int32_t spLoopInstsNum = 2;
    const int32_t spInstsNum = spInitInstsNum + spLoopInstsNum;

    const uint32_t input_count = 2;
    const uint32_t output_count = 2;
    vx_tensor inputs_tensor[2] = {NULL};
    vx_tensor outputs_tensor[2] = {NULL};
    vx_node node = NULL;
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[2];
    vsi_nn_spinst_attr_t attr;

    vsi_status status = VSI_FAILURE;
    float input0_scale = vsi_nn_get_tensor_scale(input0);
    float input1_scale = vsi_nn_get_tensor_scale(input1);
    float output_scale = vsi_nn_get_tensor_scale(output0);
    float scale0 = input0_scale / output_scale;
    float scale1 = input1_scale / output_scale;
    float const2 = -scale1 * (float)vsi_nn_get_tensor_zero_point(input1);
    float clamp_min = 0;
    float clamp_max = 0;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    vsi_nn_get_tensor_clamp_min_max(input0, &clamp_min, &clamp_max);
    clamp_min = clamp_min * scale0;
    clamp_max = clamp_max * scale0;

    /* loop inst0: r1 = clamp(in * r3, r7, r6) || r2 = r1 + r2 */
    status  = vsi_nn_sp_mul_clamp(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_SR3, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_add(&sp_insts_param[0], VSI_NN_SP_SR1, VSI_NN_SP_SR2, VSI_NN_SP_SR2);
    /* loop inst1: r2 = in * r4 || out = r2 + r5 */
    status |= vsi_nn_sp_mul(&sp_insts_param[1], VSI_NN_SP_SRIN, VSI_NN_SP_SR4, VSI_NN_SP_SR2);
    status |= vsi_nn_sp_add(&sp_insts_param[1], VSI_NN_SP_SR2, VSI_NN_SP_SR5, VSI_NN_SP_SROUT);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.ignored_leading_outputs = 3;
    attr.flush_cycle_num = 6;

    VSI_NN_SP_ATTR_SET_CONST_TO_SR3(attr, scale0);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR4(attr, scale1);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR5_LOW_PRECISION(attr, const2);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR6(attr, clamp_max);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR7(attr, clamp_min);

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;
    attr.input_setup = VSI_NN_SP_INPUT_SETUP_INTERLEAVE_TWO_INPUT;

    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_XYZ;
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

vsi_nn_kernel_node_t vsi_nn_sp_lstm_cell_times_v11_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input0,
        vsi_nn_tensor_t             * input1,
        vsi_nn_tensor_t             * output,
        char                        * kernel_name
    )
{
    const int32_t spInitInstsNum = 0;
    const int32_t spLoopInstsNum = 2;
    const int32_t spInstsNum = spInitInstsNum + spLoopInstsNum;

    const uint32_t input_count = 2;
    const uint32_t output_count = 1;
    vx_tensor inputs_tensor[2] = {NULL};
    vx_tensor outputs_tensor[1] = {NULL};
    vx_node node = NULL;
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[2];
    vsi_nn_spinst_attr_t attr;

    vsi_status status = VSI_FAILURE;
    float input0_scale = vsi_nn_get_tensor_scale(input0);
    float scale0 = input0_scale;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    /* loop inst0: r1 = in * r3 */
    status  = vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_SR3, VSI_NN_SP_SR1);
    /* loop inst1: v12 = r1 * v12 */
    status |= vsi_nn_sp_mul(&sp_insts_param[1], VSI_NN_SP_SR1, VSI_NN_SP_VR12, VSI_NN_SP_VR12);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.flush_cycle_num = 3;

    VSI_NN_SP_ATTR_SET_CONST_TO_SR3(attr, scale0);

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;
    attr.input_setup = VSI_NN_SP_INPUT_SETUP_SINGLE_INPUT;

    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;

    attr.ignored_leading_v12_rd = 1;
    attr.ignored_leading_v12_wr = 1;
    attr.num_of_v12_rd_in_flush_cycle = 2;
    attr.num_of_v12_wr_in_flush_cycle = 2;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_XYZ;
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

vsi_nn_kernel_node_t vsi_nn_sp_lstm_cell_gate_times_v11_plus_v12_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input0,
        vsi_nn_tensor_t             * input1,
        vsi_nn_tensor_t             * input2,
        vsi_nn_tensor_t             * input3,
        vsi_nn_tensor_t             * output0,
        vsi_nn_tensor_t             * output1,
        char                        * kernel_name
    )
{
    const int32_t spInitInstsNum = 0;
    const int32_t spLoopInstsNum = 4;
    const int32_t spInstsNum = spInitInstsNum + spLoopInstsNum;

    const uint32_t input_count = 4;
    const uint32_t output_count = 2;
    vx_tensor inputs_tensor[4] = {NULL};
    vx_tensor outputs_tensor[2] = {NULL};
    vx_node node = NULL;
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[4];
    vsi_nn_spinst_attr_t attr;

    vsi_status status = VSI_FAILURE;
    float input0_scale = vsi_nn_get_tensor_scale(input0);
    float scale0 = input0_scale;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    /* loop inst0: r2 = in * r3 | r4 = r2 + r4 | r6 = r1 */
    status  = vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_SR3, VSI_NN_SP_SR2);
    status |= vsi_nn_sp_add(&sp_insts_param[0], VSI_NN_SP_SR2, VSI_NN_SP_SR4, VSI_NN_SP_SR4);
    status |= vsi_nn_sp_move(&sp_insts_param[0], VSI_NN_SP_SR1, VSI_NN_SP_SR6);
    /* loop inst1: r5 = r1 * r4 | r7 = r5 + r6 | r4 = in */
    status |= vsi_nn_sp_mul(&sp_insts_param[1], VSI_NN_SP_SR1, VSI_NN_SP_SR4, VSI_NN_SP_SR5);
    status |= vsi_nn_sp_add(&sp_insts_param[1], VSI_NN_SP_SR5, VSI_NN_SP_SR6, VSI_NN_SP_SR7);
    status |= vsi_nn_sp_move(&sp_insts_param[1], VSI_NN_SP_SRIN, VSI_NN_SP_SR4);
    /* loop inst2:r1 = pwlMul * pwlMul | r4 = pwlAdd + pwlAdd | v12 = r9 */
    status |= vsi_nn_sp_mul(&sp_insts_param[2], VSI_NN_SP_PWLMUL, VSI_NN_SP_PWLMUL, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_sub(&sp_insts_param[2], VSI_NN_SP_PWLADD, VSI_NN_SP_PWLADD, VSI_NN_SP_SR4);
    status |= vsi_nn_sp_move(&sp_insts_param[2], VSI_NN_SP_SR9, VSI_NN_SP_VR12);
    /* loop inst2: r1 = tanh(r4) | r8 = v11 * r7 | r9 = r8 + v12 | out = r9 */
    status |= vsi_nn_sp_pwl_tanh(&sp_insts_param[3], VSI_NN_SP_SR4, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_mul(&sp_insts_param[3], VSI_NN_SP_VR11, VSI_NN_SP_SR7, VSI_NN_SP_SR8);
    status |= vsi_nn_sp_add(&sp_insts_param[3], VSI_NN_SP_SR8, VSI_NN_SP_VR12, VSI_NN_SP_SR9);
    status |= vsi_nn_sp_move(&sp_insts_param[3], VSI_NN_SP_SR9, VSI_NN_SP_SROUT);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;

    attr.input_setup = VSI_NN_SP_INPUT_SETUP_INTERLEAVE_TWO_INPUT;
    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.ignored_leading_outputs = 7;
    attr.ignored_leading_v11_rd = 5;
    attr.ignored_leading_v12_rd = 6;
    attr.ignored_leading_v12_wr = 7;
    attr.flush_cycle_num = 30;

    attr.num_of_v11_rd_in_flush_cycle = 6;
    attr.num_of_v11_wr_in_flush_cycle = 0;
    attr.num_of_v12_rd_in_flush_cycle = 7;
    attr.num_of_v12_wr_in_flush_cycle = 8;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_XYZ;
    attr.split_max_vector_depth = max_vector_depth;

    VSI_NN_SP_ATTR_SET_CONST_TO_SR3(attr, scale0);

    spinst = vsi_nn_create_spinst(graph);
    CHECK_PTR_FAIL_GOTO( spinst, "Create spInst fail.", final );
    status  = vsi_nn_add_spinst_insts(spinst, sp_insts_param, spInstsNum);
    status |= vsi_nn_set_spinst_attr(spinst, attr);
    CHECK_STATUS_FAIL_GOTO(status, final );

    inputs_tensor[0] = input0->t;
    inputs_tensor[1] = input1->t;
    inputs_tensor[2] = input2->t;
    inputs_tensor[3] = input3->t;
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

vsi_nn_kernel_node_t vsi_nn_sp_lstm_one_minus_v12_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input,
        vsi_nn_tensor_t             * output,
        char                        * kernel_name
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
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[4];
    vsi_nn_spinst_attr_t attr;

    vsi_status status = VSI_FAILURE;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    /* loop inst0: v11 = r3 - v12 | v12 = v12 */
    status  = vsi_nn_sp_sub(&sp_insts_param[0], VSI_NN_SP_SR3, VSI_NN_SP_VR12, VSI_NN_SP_VR11);
    status |= vsi_nn_sp_move(&sp_insts_param[0], VSI_NN_SP_VR12, VSI_NN_SP_VR12);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;

    attr.input_setup = VSI_NN_SP_INPUT_SETUP_V12;
    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.flush_cycle_num = 0;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_XYZ;
    attr.split_max_vector_depth = max_vector_depth;

    VSI_NN_SP_ATTR_SET_CONST_TO_SR3(attr, 1);

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

vsi_nn_kernel_node_t vsi_nn_sp_lstmunit_ln_activation
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input_i,
        vsi_nn_tensor_t             * input_f,
        vsi_nn_tensor_t             * input_c,
        vsi_nn_tensor_t             * input_o,
        vsi_nn_tensor_t             * cell_state_in,
        vsi_nn_tensor_t             * output,
        vsi_nn_tensor_t             * cell_state_out,
        vsi_nn_tensor_t             * hstate_out,
        int32_t                      _is_cifg,
        int32_t                      _is_proj,
        float                        forget_bias
    )
{
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_kernel_node_t temp_node[5] = {NULL};
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_t * dummy_tensor[4] = {NULL};

    memcpy( &attr, &cell_state_in->attr, sizeof(vsi_nn_tensor_attr_t) );
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    attr.is_const = FALSE;
    attr.vtl = TRUE;
    dummy_tensor[0] = vsi_nn_create_dummy_tensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[0], "Create dummy_tensor fail.", final );
    dummy_tensor[1] = vsi_nn_create_dummy_tensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[1], "Create dummy_tensor fail.", final );
    dummy_tensor[2] = vsi_nn_create_dummy_tensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[2], "Create dummy_tensor fail.", final );
    dummy_tensor[3] = vsi_nn_create_dummy_tensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[3], "Create dummy_tensor fail.", final );

    if (!_is_cifg)
    {
        temp_node[0] = vsi_nn_sp_sigmoid_node(graph, input_i,
            dummy_tensor[0], VSI_NN_SP_VR11, "lstmunit_activation_ln_0" );
        CHECK_PTR_FAIL_GOTO( temp_node[0], "Create lstmunit_activation sp node fail.", final );

        if (forget_bias == 0)
        {
            temp_node[1] = vsi_nn_sp_sigmoid_node(graph, input_f,
                dummy_tensor[1], VSI_NN_SP_VR12, "lstmunit_activation_ln_1");
        }
        else
        {
            temp_node[1] = vsi_nn_sp_linear_sigmoid_node(graph, input_f,
                dummy_tensor[1], forget_bias, "lstmunit_activation_ln_1");
        }
        CHECK_PTR_FAIL_GOTO( temp_node[1], "Create lstmunit_activation sp sigmoid node fail.", final );
    }
    else
    {
        temp_node[0] = vsi_nn_sp_cifg_linear_sigmoid_node(graph, input_f, NULL, dummy_tensor[0], dummy_tensor[1],
            forget_bias, "lstmunit_activation_ln_cifg_0");
        CHECK_PTR_FAIL_GOTO( temp_node[0], "Create lstmunit_activation sp cifg_linear_sigmoid node fail.", final );
    }

    temp_node[2] = vsi_nn_sp_get_cell_status_node(graph, input_c, cell_state_in,
        dummy_tensor[0], dummy_tensor[1], NULL, cell_state_out, dummy_tensor[2], "lstmunit_activation_ln_2");
    CHECK_PTR_FAIL_GOTO( temp_node[2], "Create lstmunit_activation sp get_cell_status node fail.", final );

    temp_node[3] = vsi_nn_sp_sigmoid_node( graph, input_o,
        dummy_tensor[3], VSI_NN_SP_VR11, "lstmunit_activation_ln_3");
    CHECK_PTR_FAIL_GOTO( temp_node[3], "Create lstmunit_activation sp sigmoid node fail.", final );

    node = vsi_nn_sp_get_output_node(graph, dummy_tensor[2], dummy_tensor[3], output, "lstmunit_activation_ln_4");
    CHECK_PTR_FAIL_GOTO( node, "Create lstmunit_activation sp sigmoid node fail.", final );

    if (!_is_proj)
    {
        temp_node[4] = node;

        node = vxTensorCopyNode( graph->g, output->t, hstate_out->t);
        CHECK_PTR_FAIL_GOTO( node, "Create lstmunit_activation dataconvert node fail.", final );
    }

final:
    vsi_safe_release_node(temp_node[0]);
    vsi_safe_release_node(temp_node[1]);
    vsi_safe_release_node(temp_node[2]);
    vsi_safe_release_node(temp_node[3]);
    vsi_safe_release_node(temp_node[4]);
    vsi_safe_release_tensor(dummy_tensor[0]);
    vsi_safe_release_tensor(dummy_tensor[1]);
    vsi_safe_release_tensor(dummy_tensor[2]);
    vsi_safe_release_tensor(dummy_tensor[3]);

    return node;
}

vsi_nn_kernel_node_t vsi_nn_sp_lstmunit_float_activation
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input_i,
        vsi_nn_tensor_t             * input_f,
        vsi_nn_tensor_t             * input_c,
        vsi_nn_tensor_t             * input_o,
        vsi_nn_tensor_t             * cell_state_in,
        vsi_nn_tensor_t             * hstate_i,
        vsi_nn_tensor_t             * hstate_f,
        vsi_nn_tensor_t             * hstate_c,
        vsi_nn_tensor_t             * hstate_o,
        vsi_nn_tensor_t             * output,
        vsi_nn_tensor_t             * cell_state_out,
        vsi_nn_tensor_t             * hstate_out,
        int32_t                      _is_cifg,
        int32_t                      _is_proj,
        float                        forget_bias
    )
{
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_kernel_node_t temp_node[6] = {NULL};
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_t * dummy_tensor[5] = {NULL};

    memcpy( &attr, &cell_state_in->attr, sizeof(vsi_nn_tensor_attr_t) );
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    attr.is_const = FALSE;
    attr.vtl = TRUE;
    dummy_tensor[0] = vsi_nn_create_dummy_tensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[0], "Create dummy_tensor fail.", final );
    dummy_tensor[1] = vsi_nn_create_dummy_tensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[1], "Create dummy_tensor fail.", final );
    dummy_tensor[2] = vsi_nn_create_dummy_tensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[2], "Create dummy_tensor fail.", final );
    dummy_tensor[3] = vsi_nn_create_dummy_tensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[3], "Create dummy_tensor fail.", final );
    dummy_tensor[4] = vsi_nn_create_dummy_tensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[4], "Create dummy_tensor fail.", final );

    if (!_is_cifg)
    {
        temp_node[0] = vsi_nn_sp_add_sigmoid_node(graph, input_i, hstate_i, NULL,
            dummy_tensor[0], VSI_NN_SP_VR11, "lstmunit_activation_0");
        CHECK_PTR_FAIL_GOTO( temp_node[0], "Create lstmunit_activation sp add node fail.", final );

        if (forget_bias == 0)
        {
            temp_node[1] = vsi_nn_sp_add_sigmoid_node(graph, input_f, hstate_f, NULL,
                dummy_tensor[1], VSI_NN_SP_VR12, "lstmunit_activation_1");
        }
        else
        {
            temp_node[1] = vsi_nn_sp_add_sigmoid_ext_node(graph, input_f, hstate_f,
                dummy_tensor[1], forget_bias, "lstmunit_activation_1");
        }
        CHECK_PTR_FAIL_GOTO( temp_node[1], "Create lstmunit_activation sp add sigmoid node fail.", final );
    }
    else
    {
        if (forget_bias == 0)
        {
            temp_node[0] = vsi_nn_sp_add_sigmoid_node(graph, input_f, hstate_f, NULL,
                dummy_tensor[0], VSI_NN_SP_VR12, "lstmunit_activation_0");
        }
        else
        {
            temp_node[0] = vsi_nn_sp_add_sigmoid_ext_node(graph, input_f, hstate_f,
                dummy_tensor[0], forget_bias, "lstmunit_activation_0");
        }
        CHECK_PTR_FAIL_GOTO( temp_node[0], "Create lstmunit_activation sp add node fail.", final );

        temp_node[1] = vsi_nn_sp_lstm_one_minus_v12_node(graph, dummy_tensor[0],
            dummy_tensor[1], "lstmunit_activation_1");
        CHECK_PTR_FAIL_GOTO( temp_node[1], "Create lstmunit_activation sp one_minus_v12 node fail.", final );
    }

    temp_node[2] = vsi_nn_sp_lstm_cell_times_v11_node(graph, cell_state_in, dummy_tensor[1],
        dummy_tensor[2], "lstmunit_activation_2");
    CHECK_PTR_FAIL_GOTO( temp_node[2], "Create lstmunit_activation sp node fail.", final );
    temp_node[3] = vsi_nn_sp_lstm_cell_gate_times_v11_plus_v12_node(graph, input_c, hstate_c,
        dummy_tensor[0], dummy_tensor[2], cell_state_out, dummy_tensor[3], "lstmunit_activation_3");

    temp_node[4] = vsi_nn_sp_add_sigmoid_node(graph, input_o, hstate_o,
        dummy_tensor[3], dummy_tensor[4], VSI_NN_SP_VR11,"lstmunit_activation_4");
    CHECK_PTR_FAIL_GOTO( temp_node[4], "Create lstmunit_activation sp add sigmoid node fail.", final );

    node = vsi_nn_sp_get_output_node(graph, dummy_tensor[2], dummy_tensor[4], output, "lstmunit_activation_5");
    CHECK_PTR_FAIL_GOTO( node, "Create lstmunit_activation sp sigmoid node fail.", final );

    if (!_is_proj)
    {
        temp_node[5] = node;

        node = vxTensorCopyNode( graph->g, output->t, hstate_out->t);
        CHECK_PTR_FAIL_GOTO( node, "Create lstmunit_activation dataconvert node fail.", final );
    }

final:
    vsi_safe_release_node(temp_node[0]);
    vsi_safe_release_node(temp_node[1]);
    vsi_safe_release_node(temp_node[2]);
    vsi_safe_release_node(temp_node[3]);
    vsi_safe_release_node(temp_node[4]);
    vsi_safe_release_node(temp_node[5]);
    vsi_safe_release_tensor(dummy_tensor[0]);
    vsi_safe_release_tensor(dummy_tensor[1]);
    vsi_safe_release_tensor(dummy_tensor[2]);
    vsi_safe_release_tensor(dummy_tensor[3]);
    vsi_safe_release_tensor(dummy_tensor[4]);

    return node;
}

#define REGISTER_LSTMUNIT_ACTIVATION_STREAM_PROCESSOR_KERNEL( kernel_name )   \
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

REGISTER_LSTMUNIT_ACTIVATION_STREAM_PROCESSOR_KERNEL( lstmunit_activation )
{
    vsi_nn_kernel_node_t node = NULL;
    int32_t  _is_ln= 0;
    int32_t  _is_cifg= 0;
    int32_t  _is_proj= 0;
    int32_t  _is_hybrid= 0;
    int32_t  _is_peephole= 0;
    int32_t  recurrent_activation;
    float    forget_bias;
    vsi_nn_kernel_dtype_e in_dtype = vsi_nn_kernel_map_dtype( inputs[LSTMUNIT_ACT_INPUT_FC_F]->attr.dtype.vx_type );

    _is_ln               = vsi_nn_kernel_param_get_int32( params, "_is_ln" );
    _is_cifg             = vsi_nn_kernel_param_get_int32( params, "_is_cifg" );
    _is_proj             = vsi_nn_kernel_param_get_int32( params, "_is_proj" );
    _is_hybrid           = vsi_nn_kernel_param_get_int32( params, "_is_hybrid" );
    _is_peephole         = vsi_nn_kernel_param_get_int32( params, "_is_peephole" );
    recurrent_activation = vsi_nn_kernel_param_get_int32( params, "recurrent_activation" );
    forget_bias          = vsi_nn_kernel_param_get_float32(params, "forget_bias");

    if ( _is_hybrid || _is_peephole
        || recurrent_activation == VSI_NN_ACT_HARD_SIGMOID
        || in_dtype == U8 )
    {
        return NULL;
    }

    if (_is_ln)
    {
        node = vsi_nn_sp_lstmunit_ln_activation
        (
            graph,
            inputs[LSTMUNIT_ACT_INPUT_FC_I],
            inputs[LSTMUNIT_ACT_INPUT_FC_F],
            inputs[LSTMUNIT_ACT_INPUT_FC_C],
            inputs[LSTMUNIT_ACT_INPUT_FC_O],
            inputs[LSTMUNIT_ACT_CSTATE_IN],
            outputs[LSTMUNIT_ACT_OUTPUT],
            outputs[LSTMUNIT_ACT_CSTATE_OUT],
            outputs[LSTMUNIT_ACT_HSTATE_OUT],
            _is_cifg,
            _is_proj,
            forget_bias
        );
    }
    else
    {
        node = vsi_nn_sp_lstmunit_float_activation
        (
            graph,
            inputs[LSTMUNIT_ACT_INPUT_FC_I],
            inputs[LSTMUNIT_ACT_INPUT_FC_F],
            inputs[LSTMUNIT_ACT_INPUT_FC_C],
            inputs[LSTMUNIT_ACT_INPUT_FC_O],
            inputs[LSTMUNIT_ACT_CSTATE_IN],
            inputs[LSTMUNIT_ACT_HSTATE_FC_I],
            inputs[LSTMUNIT_ACT_HSTATE_FC_F],
            inputs[LSTMUNIT_ACT_HSTATE_FC_C],
            inputs[LSTMUNIT_ACT_HSTATE_FC_O],
            outputs[LSTMUNIT_ACT_OUTPUT],
            outputs[LSTMUNIT_ACT_CSTATE_OUT],
            outputs[LSTMUNIT_ACT_HSTATE_OUT],
            _is_cifg,
            _is_proj,
            forget_bias
        );
    }

    return node;
} /* lstmunit_activation() */

#undef REGISTER_LSTMUNIT_ACTIVATION_STREAM_PROCESSOR_KERNEL

#endif
