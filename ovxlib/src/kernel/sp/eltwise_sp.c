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

#define HASH_SUB_ADD_ALU_SUPPORT_KEY(IN0_TYPE, IN1_TYPE) \
    ( IN0_TYPE | (IN1_TYPE << 8) )

#define HASH_ELTWISE_OP_SUPPORT_TYPE( IN0_TYPE, IN1_TYPE ) \
    { HASH_SUB_ADD_ALU_SUPPORT_KEY(IN0_TYPE, IN1_TYPE) }

static const struct {
        uint32_t key;
    } eltwise_alu_kernel_map[] =
{
    HASH_ELTWISE_OP_SUPPORT_TYPE(U8,  U8),
    HASH_ELTWISE_OP_SUPPORT_TYPE(U8,  I8),
    HASH_ELTWISE_OP_SUPPORT_TYPE(I8,  I8),
    HASH_ELTWISE_OP_SUPPORT_TYPE(I8,  U8),
};

static vsi_bool _is_float_tensor(vsi_nn_tensor_t * input)
{
    if ( input->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT16 ||
         input->attr.dtype.vx_type == VSI_NN_TYPE_BFLOAT16 ||
         input->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT32 )
    {
        return TRUE;
    }

    return FALSE;
}

vsi_nn_kernel_node_t vsi_nn_sp_add_node
    (
        vsi_nn_graph_t              * graph,
        vsi_nn_tensor_t             * input0,
        vsi_nn_tensor_t             * input1,
        vsi_nn_tensor_t             * output
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
    float scale0 = input0_scale / output_scale;
    float scale1 = input1_scale / output_scale;
    float const2 = -scale1 * (float)vsi_nn_get_tensor_zero_point(input1);
    float clamp_min = 0;
    float clamp_max = 0;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    memset(&attr, 0, sizeof(vsi_nn_spinst_attr_t));

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
        vsi_nn_tensor_t             * output
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
    float scale0 = input0_scale / output_scale;
    float scale1 = input1_scale / output_scale;
    float const2 = -scale1 * (float)vsi_nn_get_tensor_zero_point(input1);
    float clamp_min = 0;
    float clamp_max = 0;

    memset(sp_insts_param, 0, sizeof(vsi_nn_spinst_inst_param) * spInstsNum);
    memset(&attr, 0, sizeof(vsi_nn_spinst_attr_t));

    vsi_nn_get_tensor_clamp_min_max(input0, &clamp_min, &clamp_max);
    clamp_min = clamp_min * scale0;
    clamp_max = clamp_max * scale0;

    /* output = in * const0 - (in1 * const1 + const2) */
    /* loop inst0: r1 = clamp(in * r3, r7, r6) || r2 = r1 - r2 */
    status  = vsi_nn_sp_mul_clamp(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_SR3, VSI_NN_SP_SR1);
    status |= vsi_nn_sp_sub(&sp_insts_param[0], VSI_NN_SP_SR1, VSI_NN_SP_SR2, VSI_NN_SP_SR2);
    /* loop inst1: r2 = in * r4 || out = r2 - r5 */
    status |= vsi_nn_sp_mul(&sp_insts_param[1], VSI_NN_SP_SRIN, VSI_NN_SP_SR4, VSI_NN_SP_SR2);
    status |= vsi_nn_sp_sub(&sp_insts_param[1], VSI_NN_SP_SR2, VSI_NN_SP_SR5, VSI_NN_SP_SROUT);
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
    attr.ignored_leading_outputs = 2;
    attr.flush_cycle_num = 3;

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
        vsi_enum                      overflow_policy,
        vsi_enum                      rounding_policy
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
    memset(&attr, 0, sizeof(vsi_nn_spinst_attr_t));

    vsi_nn_get_tensor_clamp_min_max(input0, &clamp_min, &clamp_max);
    clamp_min = clamp_min * scale0;
    clamp_max = clamp_max * scale0;

    /* loop inst0: v11 = clamp(in * r3, r7, r6) */
    status  = vsi_nn_sp_mul_clamp(&sp_insts_param[0], VSI_NN_SP_SRIN, VSI_NN_SP_SR3, VSI_NN_SP_VR11);
    /* loop inst1: out = v11 * r5 || r5 = in + r4 */
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

vsi_bool vsi_nn_sp_nn_alu_support_types
    (
        vsi_nn_tensor_t             * input0,
        vsi_nn_tensor_t             * input1,
        vsi_nn_tensor_t             * output
    )
{
    vsi_nn_kernel_dtype_e in0_dtype;
    vsi_nn_kernel_dtype_e in1_dtype;
    uint32_t key = 0;
    int32_t i = 0;

    in0_dtype = vsi_nn_kernel_map_dtype( input0->attr.dtype.vx_type );
    in1_dtype = vsi_nn_kernel_map_dtype( input1->attr.dtype.vx_type );

    key = HASH_SUB_ADD_ALU_SUPPORT_KEY(in0_dtype, in1_dtype);

    for ( i = 0; i < _cnt_of_array(eltwise_alu_kernel_map); i ++ )
    {
        if ( eltwise_alu_kernel_map[i].key == key )
        {
            return TRUE;
        }
    }

    return FALSE;
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
    vsi_nn_kernel_node_t node = NULL;

    if (vsi_nn_is_broadcast_operaton(inputs, input_num, outputs[0]) ||
        vsi_nn_sp_nn_alu_support_types(inputs[0], inputs[1], outputs[0]))
    {
        return NULL;
    }

    node = vsi_nn_sp_add_node(graph, inputs[0], inputs[1], outputs[0]);
    CHECK_PTR_FAIL_GOTO( node, "Create sp_add node fail.", final );

final:

    return node;
} /* add() */

REGISTER_ELTWISE_STREAM_PROCESSOR_KERNEL( sub )
{
    vsi_nn_kernel_node_t node = NULL;

    if (vsi_nn_is_broadcast_operaton(inputs, input_num, outputs[0]) ||
        vsi_nn_sp_nn_alu_support_types(inputs[0], inputs[1], outputs[0]))
    {
        return NULL;
    }

    node = vsi_nn_sp_sub_node(graph, inputs[0], inputs[1], outputs[0]);
    CHECK_PTR_FAIL_GOTO( node, "Create sp_sub node fail.", final );

final:

    return node;
} /* sub() */

REGISTER_ELTWISE_STREAM_PROCESSOR_KERNEL( mul )
{
    vsi_nn_kernel_node_t node = NULL;
    float scale;
    vsi_enum overflow_policy, rounding_policy;

    scale = vsi_nn_kernel_param_get_float32(params, "scale");
    overflow_policy = vsi_nn_kernel_param_get_int32(params, "overflow_policy");
    rounding_policy = vsi_nn_kernel_param_get_int32(params, "rounding_policy");

    if (vsi_nn_is_broadcast_operaton(inputs, input_num, outputs[0]) ||
        vsi_nn_sp_nn_alu_support_types(inputs[0], inputs[1], outputs[0]))
    {
        return NULL;
    }

    node = vsi_nn_sp_mul_node(graph, inputs[0], inputs[1], outputs[0],
        scale, overflow_policy, rounding_policy);
    CHECK_PTR_FAIL_GOTO( node, "Create sp_sub node fail.", final );

final:

    return node;
} /* sub() */
__END_DECLS

#undef REGISTER_ELTWISE_STREAM_PROCESSOR_KERNEL

#endif
