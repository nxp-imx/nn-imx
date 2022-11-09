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
__BEGIN_DECLS

vsi_nn_kernel_node_t vsi_nn_sp_add_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input0,
        vsi_nn_tensor_t             * input1,
        vsi_nn_tensor_t             * output,
        char                        * kernel_name
    )
{
    const int32_t spInitInstsNum = 1;
    const int32_t spLoopInstsNum = 2;
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
    float input0_scale = vsi_nn_get_tensor_scale(input0);
    float input1_scale = vsi_nn_get_tensor_scale(input1);
    float output_scale = vsi_nn_get_tensor_scale(output);
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

    /* init inst0: r5 = const2 */
    status  = vsi_nn_sp_move_constant(&sp_insts_param[0], const2, VSI_NN_SP_SR5);
    /* loop inst0: r1 = clamp(in * r3, r7, r6) | r2 = r1 + r2 */
    status |= vsi_nn_sp_mul_clamp(&sp_insts_param[1], VSI_NN_SP_SRIN, VSI_NN_SP_SR3, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_add(&sp_insts_param[1], VSI_NN_SP_SR1, VSI_NN_SP_SR2, VSI_NN_SP_SR2);
    /* loop inst1: r2 = in * r4 | out = r2 + r5 */
    status |= vsi_nn_sp_mul(&sp_insts_param[2], VSI_NN_SP_SRIN, VSI_NN_SP_SR4, VSI_NN_SP_SR2);
    status |= vsi_nn_sp_add(&sp_insts_param[2], VSI_NN_SP_SR2, VSI_NN_SP_SR5, VSI_NN_SP_SROUT);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.ignored_leading_outputs = 3;
    attr.flush_cycle_num = 6;

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

vsi_nn_kernel_node_t vsi_nn_sp_sub_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input0,
        vsi_nn_tensor_t             * input1,
        vsi_nn_tensor_t             * output,
        char                        * kernel_name
    )
{
    const int32_t spInitInstsNum = 1;
    const int32_t spLoopInstsNum = 2;
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
    float input0_scale = vsi_nn_get_tensor_scale(input0);
    float input1_scale = vsi_nn_get_tensor_scale(input1);
    float output_scale = vsi_nn_get_tensor_scale(output);
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

    /* output = in * const0 - (in1 * const1 + const2) */
    /* init inst0: r5 = const2 */
    status  = vsi_nn_sp_move_constant(&sp_insts_param[0], const2, VSI_NN_SP_SR5);
    /* loop inst0: r1 = clamp(in * r3, r7, r6) | r2 = r1 - r2 */
    status |= vsi_nn_sp_mul_clamp(&sp_insts_param[1], VSI_NN_SP_SRIN, VSI_NN_SP_SR3, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_sub(&sp_insts_param[1], VSI_NN_SP_SR1, VSI_NN_SP_SR2, VSI_NN_SP_SR2);
    /* loop inst1: r2 = in * r4 | out = r2 - r5 */
    status |= vsi_nn_sp_mul(&sp_insts_param[2], VSI_NN_SP_SRIN, VSI_NN_SP_SR4, VSI_NN_SP_SR2);
    status |= vsi_nn_sp_sub(&sp_insts_param[2], VSI_NN_SP_SR2, VSI_NN_SP_SR5, VSI_NN_SP_SROUT);
    CHECK_STATUS_FAIL_GOTO(status, final );

    VSI_NN_SP_ATTR_SET_CONST_TO_SR3(attr, scale0);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR4(attr, scale1);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR6(attr, clamp_max);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR7(attr, clamp_min);

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;
    attr.input_setup = VSI_NN_SP_INPUT_SETUP_INTERLEAVE_TWO_INPUT;

    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.ignored_leading_outputs = 3;
    attr.flush_cycle_num = 6;

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

vsi_nn_kernel_node_t vsi_nn_sp_mul_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input0,
        vsi_nn_tensor_t             * input1,
        vsi_nn_tensor_t             * output,
        float                         scale,
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

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[2];
    vsi_nn_spinst_attr_t attr;

    vsi_status status = VSI_FAILURE;
    float input0_scale = vsi_nn_get_tensor_scale(input0);
    float input1_scale = vsi_nn_get_tensor_scale(input1);
    float output_scale = vsi_nn_get_tensor_scale(output);
    float scale0 = scale * input0_scale * input1_scale / output_scale;
    float const1 = -(float)vsi_nn_get_tensor_zero_point(input1);
    float clamp_min = 0;
    float clamp_max = 0;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    vsi_nn_get_tensor_clamp_min_max(input0, &clamp_min, &clamp_max);
    clamp_min = clamp_min * scale0;
    clamp_max = clamp_max * scale0;

    /* loop inst0: v11 = clamp(in * r3, r7, r6) */
    status  = vsi_nn_sp_mul_clamp(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_SR3, VSI_NN_SP_VR11);
    /* loop inst1: out = v11 * r5 | r5 = in + r4 */
    status |= vsi_nn_sp_mul(&sp_insts_param[1], VSI_NN_SP_VR11, VSI_NN_SP_SR5, VSI_NN_SP_SROUT);
    status |= vsi_nn_sp_add(&sp_insts_param[1], VSI_NN_SP_SRIN, VSI_NN_SP_SR4, VSI_NN_SP_SR5);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;
    attr.input_setup = VSI_NN_SP_INPUT_SETUP_INTERLEAVE_TWO_INPUT;

    attr.ignored_leading_outputs = 2;
    attr.flush_cycle_num = 4;
    attr.v11_reset_at_start = VSI_NN_SP_V_RESET_AT_START_RESET;
    attr.ignored_leading_v11_rd = 2;
    attr.ignored_leading_v11_wr = 0;

    attr.num_of_v11_rd_in_flush_cycle = 2;

    VSI_NN_SP_ATTR_SET_CONST_TO_SR3(attr, scale0);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR4(attr, const1);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR6(attr, clamp_max);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR7(attr, clamp_min);

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

vsi_nn_kernel_node_t vsi_nn_sp_rcp_to_v11_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input,
        vsi_nn_tensor_t             * output,
        char                        * kernel_name
    )
{
    const int32_t spInitInstsNum = 0;
    const int32_t spLoopInstsNum = 2;
    const int32_t spInstsNum = spInitInstsNum + spLoopInstsNum;

    const uint32_t input_count = 1;
    const uint32_t output_count = 1;
    vx_tensor inputs_tensor[1] = {NULL};
    vx_tensor outputs_tensor[1] = {NULL};
    vx_node node = NULL;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[2];
    vsi_nn_spinst_attr_t attr;
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth;

    vsi_nn_sp_lut_params sp_lut_params;
    vx_lut_params_s vx_lut_params;
    vsi_status status = VSI_FAILURE;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);
    memset(&sp_lut_params, 0, sizeof(vsi_nn_sp_lut_params));
    memset(&vx_lut_params, 0, sizeof(vx_lut_params_s));

    /* loop inst0: r1 = setup(in) | R6 = R5 * R2 | v11 = r4 + r6 | r3 = r1 */
    status  = vsi_nn_sp_pwl_setup0(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_SR5, VSI_NN_SP_SR2, VSI_NN_SP_SR6);
    status |= vsi_nn_sp_add(&sp_insts_param[0], VSI_NN_SP_SR4, VSI_NN_SP_SR6, VSI_NN_SP_VR11);
    status |= vsi_nn_sp_move(&sp_insts_param[0], VSI_NN_SP_SR1, VSI_NN_SP_SR3);
    /* loop inst1: r5 = pwlMul * pwlMul | r2 = pwlAdd + pwlAdd | r4 = r3 */
    status |= vsi_nn_sp_mul(&sp_insts_param[1], VSI_NN_SP_PWLMUL, VSI_NN_SP_PWLMUL, VSI_NN_SP_SR5);
    status |= vsi_nn_sp_sub(&sp_insts_param[1], VSI_NN_SP_PWLADD, VSI_NN_SP_PWLADD, VSI_NN_SP_SR2);
    status |= vsi_nn_sp_move(&sp_insts_param[1], VSI_NN_SP_SR3, VSI_NN_SP_SR4);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;
    attr.input_setup = VSI_NN_SP_INPUT_SETUP_SINGLE_INPUT;

    attr.flush_cycle_num = 10;
    attr.v11_reset_at_start = VSI_NN_SP_V_RESET_AT_START_RESET;
    attr.ignored_leading_v11_rd = 0;
    attr.ignored_leading_v11_wr = 5;

    attr.num_of_v11_wr_in_flush_cycle = 5;

    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_XYZ;
    attr.split_max_vector_depth = max_vector_depth;

    spinst = vsi_nn_create_spinst(graph);
    CHECK_PTR_FAIL_GOTO( spinst, "Create spInst fail.", final );
    status  = vsi_nn_add_spinst_insts(spinst, sp_insts_param, spInstsNum);
    status |= vsi_nn_set_spinst_attr(spinst, attr);
    CHECK_STATUS_FAIL_GOTO(status, final );

    vx_lut_params.lut_function = VX_NN_ACTIVATION_CUSTOM;
    vx_lut_params.in_lut = vxCreateLUT( graph->ctx->c, VX_TYPE_FLOAT32, VSI_NN_SP_LUT_MAX_SIZE);
    vx_lut_params.out_lut = vxCreateLUT( graph->ctx->c, VX_TYPE_FLOAT32, VSI_NN_SP_LUT_MAX_SIZE);

    sp_lut_params.act_type = VSI_NN_SP_ACT_RCP;
    vsi_nn_sp_lut(vx_lut_params.in_lut, vx_lut_params.out_lut, &sp_lut_params);

    inputs_tensor[0] = input->t;
    outputs_tensor[0] = output->t;
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

vsi_nn_kernel_node_t vsi_nn_sp_mul_times_v11_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input0,
        vsi_nn_tensor_t             * input1,
        vsi_nn_tensor_t             * output,
        float                         scale,
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

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[2];
    vsi_nn_spinst_attr_t attr;
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth;

    vsi_status status = VSI_FAILURE;
    float input0_scale = vsi_nn_get_tensor_scale(input0);
    float output_scale = vsi_nn_get_tensor_scale(output);
    float scale0 = scale * input0_scale / output_scale;
    float clamp_min = 0;
    float clamp_max = 0;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    vsi_nn_get_tensor_clamp_min_max(input0, &clamp_min, &clamp_max);
    clamp_min = clamp_min * scale0;
    clamp_max = clamp_max * scale0;

    /* loop inst0: r1 = clamp(in * r3, r7, r6) | r2 = v11 */
    status  = vsi_nn_sp_mul_clamp(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_SR3, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_move(&sp_insts_param[0], VSI_NN_SP_VR11, VSI_NN_SP_SR2);
    /* loop inst1: out = r1 * r2 */
    status |= vsi_nn_sp_mul(&sp_insts_param[1], VSI_NN_SP_SR1, VSI_NN_SP_SR2, VSI_NN_SP_SROUT);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;
    attr.input_setup = VSI_NN_SP_INPUT_SETUP_SINGLE_INPUT;

    attr.ignored_leading_outputs = 1;
    attr.flush_cycle_num = 3;

    VSI_NN_SP_ATTR_SET_CONST_TO_SR3(attr, scale0);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR6(attr, clamp_max);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR7(attr, clamp_min);

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

/*load input tensor to vector registers*/
vsi_nn_kernel_node_t vsi_nn_sp_preload_node
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
    vx_tensor inputs_tensor[2] = {NULL};
    vx_tensor outputs_tensor[1] = {NULL};
    vx_node node = NULL;
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth /
        graph->ctx->config.sp_exec_count;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[2];
    vsi_nn_spinst_attr_t attr;

    vsi_status status = VSI_FAILURE;
    float input_scale = vsi_nn_get_tensor_scale(input);
    float clamp_min = 0;
    float clamp_max = 0;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    vsi_nn_get_tensor_clamp_min_max(input, &clamp_min, &clamp_max);
    clamp_min = clamp_min * input_scale;
    clamp_max = clamp_max * input_scale;

    /* loop inst0: v11 = clamp(in * r3, r7, r6) */
    status  = vsi_nn_sp_mul_clamp(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_SR3, VSI_NN_SP_VR11);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;
    attr.input_setup = VSI_NN_SP_INPUT_SETUP_SINGLE_INPUT;
    attr.ch0_post_redistribute = VSI_NN_SP_CH_POST_REDISTRIBUTE_VECTOR_GATHER;

    attr.ignored_leading_outputs = 0;
    attr.flush_cycle_num = 0;
    attr.v11_reset_at_start = VSI_NN_SP_V_RESET_AT_START_RESET;
    attr.ignored_leading_v11_rd = 0;
    attr.ignored_leading_v11_wr = 0;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_Z;
    attr.split_max_vector_depth = max_vector_depth;

    VSI_NN_SP_ATTR_SET_CONST_TO_SR3(attr, input_scale);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR6(attr, clamp_max);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR7(attr, clamp_min);

    attr.prog_init_instr_num = spInitInstsNum;
    attr.prog_loop_instr_num = spLoopInstsNum;

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

vsi_nn_kernel_node_t vsi_nn_sp_mul_per_channel_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input0,
        vsi_nn_tensor_t             * input1,
        vsi_nn_tensor_t             * output,
        float                         scale,
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
    int32_t max_vector_depth = graph->ctx->config.sp_vector_depth /
        graph->ctx->config.sp_exec_count;

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[2];
    vsi_nn_spinst_attr_t attr;

    vsi_status status = VSI_FAILURE;
    float input0_scale = vsi_nn_get_tensor_scale(input0);
    float output_scale = vsi_nn_get_tensor_scale(output);
    float scale0 = scale * input0_scale / output_scale;
    float clamp_min = 0;
    float clamp_max = 0;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    vsi_nn_get_tensor_clamp_min_max(input0, &clamp_min, &clamp_max);
    clamp_min = clamp_min * scale0;
    clamp_max = clamp_max * scale0;

    /* loop inst0: v11 = clamp(in * r3, r7, r6) */
    status  = vsi_nn_sp_mul_clamp(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_SR3, VSI_NN_SP_VR12);
    /* loop inst1: out = v11 * v12*/
    status |= vsi_nn_sp_mul(&sp_insts_param[1], VSI_NN_SP_VR11, VSI_NN_SP_VR12, VSI_NN_SP_SROUT);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;
    attr.input_setup = VSI_NN_SP_INPUT_SETUP_SINGLE_INPUT;

    attr.ignored_leading_outputs = 2;
    attr.flush_cycle_num = 5;
    attr.v12_reset_at_start = VSI_NN_SP_V_RESET_AT_START_RESET;
    attr.v11_push_pop_config = VSI_NN_SP_PUSH_POP_EVERY_ROW;
    attr.ignored_leading_v11_rd = 2;
    attr.ignored_leading_v11_wr = 0;
    attr.ignored_leading_v12_rd = 2;
    attr.ignored_leading_v12_wr = 0;

    attr.num_of_v11_rd_in_flush_cycle = 3;
    attr.num_of_v12_rd_in_flush_cycle = 3;

    attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_Z;
    attr.split_max_vector_depth = max_vector_depth;

    VSI_NN_SP_ATTR_SET_CONST_TO_SR3(attr, scale0);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR6(attr, clamp_max);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR7(attr, clamp_min);

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

vsi_nn_kernel_node_t vsi_nn_sp_select_a_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input0,
        vsi_nn_tensor_t             * input1,
        vsi_nn_tensor_t             * output,
        float                         output_scale,
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

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[2];
    vsi_nn_spinst_attr_t attr;

    vsi_status status = VSI_FAILURE;
    float input0_scale = vsi_nn_get_tensor_scale(input0);
    float scale0 = input0_scale / output_scale;
    float clamp_min = 0;
    float clamp_max = 0;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    vsi_nn_get_tensor_clamp_min_max(input0, &clamp_min, &clamp_max);
    clamp_min = clamp_min * scale0;
    clamp_max = clamp_max * scale0;

    /* loop inst0: r4 = clamp(in * r3, r7, r6) | out = r2 ? r4 : r5 */
    status  = vsi_nn_sp_mul_clamp(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_SR3, VSI_NN_SP_SR4);
    status |= vsi_nn_sp_move_sel0(&sp_insts_param[0], VSI_NN_SP_SR2, VSI_NN_SP_SR4, VSI_NN_SP_SROUT);
    /* loop inst1: r2 = abs(in) */
    status |= vsi_nn_sp_abs(&sp_insts_param[1], VSI_NN_SP_SRIN, VSI_NN_SP_SR2);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.ignored_leading_outputs = 2;
    attr.flush_cycle_num = 3;

    VSI_NN_SP_ATTR_SET_CONST_TO_SR3(attr, scale0);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR5_LOW_PRECISION(attr, 0);
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

vsi_nn_kernel_node_t vsi_nn_sp_select_b_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input0,
        vsi_nn_tensor_t             * input1,
        vsi_nn_tensor_t             * output,
        float                         output_scale,
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

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[2];
    vsi_nn_spinst_attr_t attr;

    vsi_status status = VSI_FAILURE;
    float input0_scale = vsi_nn_get_tensor_scale(input0);
    float scale0 = input0_scale / output_scale;
    float clamp_min = 0;
    float clamp_max = 0;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    vsi_nn_get_tensor_clamp_min_max(input0, &clamp_min, &clamp_max);
    clamp_min = clamp_min * scale0;
    clamp_max = clamp_max * scale0;

    /* loop inst0: r5 = clamp(in * r3, r7, r6) | out = r2 ? r4 : r5 */
    status  = vsi_nn_sp_mul_clamp(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_SR3, VSI_NN_SP_SR5);
    status |= vsi_nn_sp_move_sel0(&sp_insts_param[0], VSI_NN_SP_SR2, VSI_NN_SP_SR4, VSI_NN_SP_SROUT);
    /* loop inst1: r2 = abs(in) */
    status |= vsi_nn_sp_abs(&sp_insts_param[1], VSI_NN_SP_SRIN, VSI_NN_SP_SR2);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.ignored_leading_outputs = 2;
    attr.flush_cycle_num = 3;

    VSI_NN_SP_ATTR_SET_CONST_TO_SR3(attr, scale0);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR4(attr, 0);
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

vsi_nn_kernel_node_t vsi_nn_sp_select_add_node
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

    vsi_nn_spinst_t *spinst = NULL;
    vsi_nn_spinst_inst_param sp_insts_param[2];
    vsi_nn_spinst_attr_t attr;

    vsi_status status = VSI_FAILURE;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    /* loop inst0: r5 = clamp(in * r3, r7, r6) | r1 = in */
    status  = vsi_nn_sp_add(&sp_insts_param[0], VSI_NN_SP_SR1, VSI_NN_SP_SR2, VSI_NN_SP_SROUT);
    status |= vsi_nn_sp_move(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_SR1);
    /* loop inst1: r2 = in */
    status |= vsi_nn_sp_move(&sp_insts_param[1], VSI_NN_SP_SRIN, VSI_NN_SP_SR2);
    CHECK_STATUS_FAIL_GOTO(status, final );

    attr.ignored_leading_outputs = 2;
    attr.flush_cycle_num = 3;

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

vsi_bool vsi_nn_sp_nn_alu_support_types
    (
        vsi_nn_tensor_t             * input0,
        vsi_nn_tensor_t             * input1,
        vsi_nn_tensor_t             * output
    )
{
    vsi_nn_kernel_dtype_e in0_dtype;
    vsi_nn_kernel_dtype_e in1_dtype;
    vsi_bool support = TRUE;

    in0_dtype = vsi_nn_kernel_map_dtype( input0->attr.dtype.vx_type );
    in1_dtype = vsi_nn_kernel_map_dtype( input1->attr.dtype.vx_type );

    switch ( in0_dtype )
    {
    case I4:
    case U4:
    case I8:
    case U8:
    case I16:
        support = TRUE;
        break;
    default:
        support = FALSE;
        break;
    }

    switch ( in1_dtype )
    {
    case I4:
    case U4:
    case I8:
    case U8:
    case I16:
        break;
    default:
        support = FALSE;
        break;
    }

    return support;
}

#define REGISTER_ELTWISE_STREAM_PROCESSOR_KERNEL( kernel_name )   \
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

REGISTER_ELTWISE_STREAM_PROCESSOR_KERNEL( add )
{
#define SE4M9_MAX_VALUE     (511.5)
#define SE4M9_MIN_VALUE     (-511.5)

    float input1_scale = vsi_nn_get_tensor_scale(inputs[1]);
    float output_scale = vsi_nn_get_tensor_scale(outputs[0]);
    float scale1 = input1_scale / output_scale;
    float const2 = -scale1 * (float)vsi_nn_get_tensor_zero_point(inputs[1]);
    vsi_nn_kernel_node_t node = NULL;

    if ( vsi_nn_is_broadcast_operaton(inputs, input_num, outputs[0]) ||
        ( vsi_nn_sp_nn_alu_support_types(inputs[0], inputs[1], outputs[0]) &&
            const2 <= SE4M9_MAX_VALUE && const2 >= SE4M9_MIN_VALUE ) )
    {
        return NULL;
    }

    node = vsi_nn_sp_add_node(graph, inputs[0], inputs[1], outputs[0], "tensoradd");
    CHECK_PTR_FAIL_GOTO( node, "Create sp_add node fail.", final );

final:

    return node;
} /* add() */

REGISTER_ELTWISE_STREAM_PROCESSOR_KERNEL( sub )
{
    float input1_scale = vsi_nn_get_tensor_scale(inputs[1]);
    float output_scale = vsi_nn_get_tensor_scale(outputs[0]);
    float scale1 = input1_scale / output_scale;
    float const2 = -scale1 * (float)vsi_nn_get_tensor_zero_point(inputs[1]);
    vsi_nn_kernel_node_t node = NULL;

    if ( vsi_nn_is_broadcast_operaton(inputs, input_num, outputs[0]) ||
        ( vsi_nn_sp_nn_alu_support_types(inputs[0], inputs[1], outputs[0]) &&
            const2 <= SE4M9_MAX_VALUE && const2 >= SE4M9_MIN_VALUE ) )
    {
        return NULL;
    }

    node = vsi_nn_sp_sub_node(graph, inputs[0], inputs[1], outputs[0], "tensorsub");
    CHECK_PTR_FAIL_GOTO( node, "Create sp_sub node fail.", final );

final:

    return node;
} /* sub() */

REGISTER_ELTWISE_STREAM_PROCESSOR_KERNEL( mul )
{
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_kernel_node_t tmp_node = NULL;
    vsi_nn_tensor_attr_t attr;
    float scale = 0;
    int32_t axis[2] = {0 , 1};
    int32_t axis_num = 2;
    vsi_nn_tensor_t *temp_tensor = NULL;
    vsi_nn_tensor_t * dummy_tensor = NULL;
    vsi_enum overflow_policy;
    vsi_bool is_broadcast = vsi_nn_is_broadcast_operaton(inputs, input_num, outputs[0]);
    vsi_bool is_broadcast_axis01 = vsi_nn_is_broadcast_axes_operaton(inputs, input_num, outputs[0], axis, axis_num);
    vsi_ssize_t input0_slice_size = inputs[0]->attr.size[0] * inputs[0]->attr.size[1];
    vsi_ssize_t input1_slice_size = inputs[1]->attr.size[0] * inputs[1]->attr.size[1];
    vsi_bool is_per_channel = is_broadcast_axis01 && (input0_slice_size == 1 || input1_slice_size == 1);
#define _swap_tensor(a, b, tmp)  \
    do { \
        tmp = a; \
        a = b; \
        b = tmp; \
    } while(0)

    scale = vsi_nn_kernel_param_get_float32(params, "scale");
    overflow_policy = vsi_nn_kernel_param_get_int32(params, "overflow_policy");

    if ( is_broadcast ||
         vsi_nn_sp_nn_alu_support_types(inputs[0], inputs[1], outputs[0]) ||
         VX_CONVERT_POLICY_WRAP == overflow_policy
        /*&& !is_per_channel*/ )
    {
        return NULL;
    }

    if (is_per_channel && inputs[0]->attr.size[0] == 1)
    {
        _swap_tensor(inputs[0], inputs[1], temp_tensor);
    }

    if (is_per_channel)
    {
        memset( &attr, 0, sizeof(vsi_nn_tensor_attr_t) );
        memcpy( attr.size, inputs[1]->attr.size, sizeof(attr.size) );
        attr.dim_num = inputs[1]->attr.dim_num;
        attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
        attr.vtl = TRUE;
        dummy_tensor = vsi_nn_create_dummy_tensor( graph, &attr );
        CHECK_PTR_FAIL_GOTO( dummy_tensor, "Create dummy_tensor fail.", final );

        tmp_node = vsi_nn_sp_preload_node(graph, inputs[1], dummy_tensor, "tensormul_0");
        CHECK_PTR_FAIL_GOTO( tmp_node, "Create sp_preload node fail.", final );

        node = vsi_nn_sp_mul_per_channel_node(graph, inputs[0], dummy_tensor, outputs[0],
            scale, "tensormul_1");
        CHECK_PTR_FAIL_GOTO( node, "Create sp_mul_per_channel node fail.", final );
    }
    else
    {
        node = vsi_nn_sp_mul_node(graph, inputs[0], inputs[1], outputs[0],
            scale, "tensormul");
        CHECK_PTR_FAIL_GOTO( node, "Create sp_mul node fail.", final );
    }

final:
#undef _swap_tensor
    vsi_safe_release_node(tmp_node);
    vsi_safe_release_tensor(dummy_tensor);

    return node;
} /* mul() */

REGISTER_ELTWISE_STREAM_PROCESSOR_KERNEL( div )
{
    vsi_nn_kernel_node_t node[2] = {NULL};
    float scale = vsi_nn_kernel_param_get_float32(params, "scale");

    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_t * output_tensor[] = {NULL};

    if (vsi_nn_is_broadcast_operaton(inputs, input_num, outputs[0]))
    {
        return NULL;
    }

    memcpy( &attr, &outputs[0]->attr, sizeof(vsi_nn_tensor_attr_t) );
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    attr.is_const = FALSE;
    attr.vtl = TRUE;
    output_tensor[0] = vsi_nn_create_dummy_tensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( output_tensor[0], "Create tensor fail.", final );

    node[0] = vsi_nn_sp_rcp_to_v11_node(graph, inputs[1], output_tensor[0], "tensordiv_0");
    CHECK_PTR_FAIL_GOTO( node[0], "Create sp_div node fail.", final );
    node[1] = vsi_nn_sp_mul_times_v11_node(graph, inputs[0], output_tensor[0], outputs[0], scale, "tensordiv_1");
    CHECK_PTR_FAIL_GOTO( node[1], "Create sp_div node fail.", final );

final:
    vsi_safe_release_node(node[0]);
    vsi_safe_release_tensor(output_tensor[0]);

    return node[1];
} /* div() */

REGISTER_ELTWISE_STREAM_PROCESSOR_KERNEL( select )
{
    vsi_nn_kernel_node_t node[3] = {NULL};
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_t * output_tensor[2] = {NULL};
    float output_scale = vsi_nn_get_tensor_scale(outputs[0]);

    if (vsi_nn_is_broadcast_operaton(inputs, input_num, outputs[0]))
    {
        return NULL;
    }

    memcpy( &attr, &outputs[0]->attr, sizeof(vsi_nn_tensor_attr_t) );
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    attr.is_const = FALSE;
    attr.vtl = TRUE;
    output_tensor[0] = vsi_nn_CreateTensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( output_tensor[0], "Create tensor fail.", final );
    output_tensor[1] = vsi_nn_CreateTensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( output_tensor[1], "Create tensor fail.", final );

    node[0] = vsi_nn_sp_select_a_node(graph, inputs[1], inputs[0], output_tensor[0], output_scale, "select_0");
    CHECK_PTR_FAIL_GOTO( node[0], "Create select_a node fail.", final );
    node[1] = vsi_nn_sp_select_b_node(graph, inputs[2], inputs[0], output_tensor[1], output_scale, "select_1");
    CHECK_PTR_FAIL_GOTO( node[1], "Create select_b node fail.", final );
    node[2] = vsi_nn_sp_select_add_node(graph, output_tensor[0], output_tensor[1], outputs[0], "select_2");
    CHECK_PTR_FAIL_GOTO( node[2], "Create select_add node fail.", final );

final:
    vsi_safe_release_node(node[0]);
    vsi_safe_release_node(node[1]);
    vsi_safe_release_tensor(output_tensor[0]);
    vsi_safe_release_tensor(output_tensor[1]);

    return node[2];
} /* select() */

__END_DECLS

#undef REGISTER_ELTWISE_STREAM_PROCESSOR_KERNEL

#endif
