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

vsi_nn_kernel_node_t vsi_nn_sp_sum_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input,
        vsi_nn_tensor_t             * dummy_output,
        int32_t                       axis_num,
        char                        * kernel_name
    )
{
    const int32_t spLoopInstsNum = 1;
    const int32_t spInstsNum = spLoopInstsNum;

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
    float clamp_min = 0;
    float clamp_max = 0;

    vsi_nn_get_tensor_clamp_min_max(input, &clamp_min, &clamp_max);

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    vsi_nn_init_spinst_attr(&attr);

    /* loop inst0: acc = clamp(in * r3, r6, r7) */
    status = vsi_nn_sp_mul_clamp(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_SR3, VSI_NN_SP_ACC);
    CHECK_STATUS_FAIL_GOTO(status, final );

    if (axis_num == 1)
    {
        attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_YZMERGE;
        attr.output_collapse_x = VSI_NN_SP_ATTR_OUTPUT_COLLAPSE_ENABLED;

        attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_YZ;
        attr.split_max_vector_depth = max_vector_depth;
    }
    else
    {
        attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;
        attr.output_collapse_x = VSI_NN_SP_ATTR_OUTPUT_COLLAPSE_ENABLED;
        attr.output_collapse_y = VSI_NN_SP_ATTR_OUTPUT_COLLAPSE_ENABLED;

        attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_Z;
        attr.split_max_vector_depth = max_vector_depth;
    }
    attr.input_setup = VSI_NN_SP_INPUT_SETUP_SINGLE_INPUT;

    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.ignored_leading_outputs = 0;
    attr.flush_cycle_num = 0;
    attr.accelerator_input_select = VSI_NN_SP_ACCELERATOR_IN_FROM_ACCEL;
    attr.sum_engine_reset = VSI_NN_SP_SUM_ENGINE_RESET_START_FROM_ZERO;
    attr.sum_engine_control = VSI_NN_SP_ACCUM_2D;
    attr.sum_engine_num_ch_minus_one = VSI_NN_SP_SUM_ENGINE_NUM_CH_ONE_CH;
    attr.sum_engine_2d_accum_storeage = VSI_NN_SP_ACCM_STOREAGE_DIFFERENT;
    attr.v11_reset_at_start = VSI_NN_SP_V_RESET_AT_START_RESET;

    VSI_NN_SP_ATTR_SET_CONST_TO_SR3(attr, 1.0f);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR6(attr, clamp_max);
    VSI_NN_SP_ATTR_SET_CONST_TO_SR7(attr, clamp_min);

    spinst = vsi_nn_create_spinst(graph);
    CHECK_PTR_FAIL_GOTO( spinst, "Create spInst fail.", final );
    status  = vsi_nn_add_spinst_insts(spinst, sp_insts_param, spInstsNum);
    status |= vsi_nn_set_spinst_attr(spinst, attr);
    CHECK_STATUS_FAIL_GOTO(status, final );

    inputs_tensor[0] = input->t;
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

vsi_nn_kernel_node_t vsi_nn_sp_v11_times_scale_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input,
        vsi_nn_tensor_t             * dummy_output,
        int32_t                       axis_num,
        float                         scale,
        char                        * kernel_name
    )
{
    const int32_t spLoopInstsNum = 1;
    const int32_t spInstsNum = spLoopInstsNum;

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

    /* loop inst0: out = v11 * r3 */
    status = vsi_nn_sp_mul(&sp_insts_param[0], VSI_NN_SP_VR11, VSI_NN_SP_SR3, VSI_NN_SP_SROUT);
    CHECK_STATUS_FAIL_GOTO(status, final );

    if (axis_num == 1)
    {
        attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_YZMERGE;

        attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_YZ;
        attr.split_max_vector_depth = max_vector_depth;
    }
    else
    {
        attr.input_tile_mapping = VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE;

        attr.split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_Z;
        attr.split_max_vector_depth = max_vector_depth;
    }
    attr.input_setup = VSI_NN_SP_INPUT_SETUP_V11;

    attr.prog_loop_instr_num = spLoopInstsNum;
    attr.ignored_leading_v11_rd = 0;
    attr.ignored_leading_outputs = 0;
    attr.flush_cycle_num = 0;

    VSI_NN_SP_ATTR_SET_CONST_TO_SR3(attr, scale);

    spinst = vsi_nn_create_spinst(graph);
    CHECK_PTR_FAIL_GOTO( spinst, "Create spInst fail.", final );
    status  = vsi_nn_add_spinst_insts(spinst, sp_insts_param, spInstsNum);
    status |= vsi_nn_set_spinst_attr(spinst, attr);
    CHECK_STATUS_FAIL_GOTO(status, final );

    inputs_tensor[0] = input->t;
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

#define REGISTER_REDUCE_MEAN_STREAM_PROCESSOR_KERNEL( kernel_name )   \
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

REGISTER_REDUCE_MEAN_STREAM_PROCESSOR_KERNEL( reduce_mean )
{
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_t * dummy_tensor[1] = {NULL};
    int32_t axis_num = vsi_nn_kernel_param_get_int32( params, "axis_num" );
    float input_scale = vsi_nn_get_tensor_scale(inputs[0]);
    float output_scale = vsi_nn_get_tensor_scale(outputs[0]);
    float scale = vsi_nn_kernel_param_get_float32( params, "scale" );

    scale = scale * input_scale / output_scale;

    memcpy( &attr, &inputs[0]->attr, sizeof(vsi_nn_tensor_attr_t) );
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    attr.is_const = FALSE;
    attr.vtl = TRUE;
    attr.size[0] = 1;
    attr.size[1] = axis_num > 1 ? 1 : attr.size[1];
    dummy_tensor[0] = vsi_nn_create_dummy_tensor( graph, &attr );
    CHECK_PTR_FAIL_GOTO( dummy_tensor[0], "Create dummy_tensor fail.", final );

    node = vsi_nn_sp_sum_node(graph, inputs[0], dummy_tensor[0], axis_num, "reduce_0");
    CHECK_PTR_FAIL_GOTO( node, "Create sp_sum node fail.", final );
    node = vsi_nn_sp_v11_times_scale_node(graph, dummy_tensor[0], outputs[0], axis_num, scale, "reduce_1");
    CHECK_PTR_FAIL_GOTO( node, "Create v11_times_scale fail.", final );

final:
    vsi_safe_release_tensor(dummy_tensor[0]);

    return node;
} /* reduce_mean() */

#undef REGISTER_REDUCE_MEAN_STREAM_PROCESSOR_KERNEL

#endif
