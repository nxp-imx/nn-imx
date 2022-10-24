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
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdint.h>
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_error.h"
#include "utils/vsi_nn_math.h"
#include "kernel/vsi_nn_spinst.h"
#include "kernel/vsi_nn_sp_unit_operation.h"

#if VX_STREAM_PROCESSOR_SUPPORT

static vsi_nn_spinst_t * _create_spinst
    (
    vx_context      context
    )
{
    vsi_nn_spinst_t * spinst = NULL;

    if ( NULL == context  )
    {
        return spinst;
    }

    spinst = (vsi_nn_spinst_t *)malloc( sizeof( vsi_nn_spinst_t ) );

    if ( NULL != spinst )
    {
        memset( spinst, 0, sizeof( vsi_nn_spinst_t ) );
        spinst->sp = vxCreateSPINST(context);
        if ( NULL == spinst->sp )
        {
            VSILOGE( "Create vx spinst fail." );
            free( spinst );
            spinst = NULL;
        }
    }

    return spinst;
}/* _create_spinst() */

vsi_nn_spinst_t * vsi_nn_create_spinst
    (
    vsi_nn_graph_t       * graph
    )
{
    if ( NULL == graph )
    {
        return NULL;
    }

    return _create_spinst(graph->ctx->c);
} /* vsi_nn_create_spinst() */

vsi_nn_spinst_t * vsi_nn_create_spinst_by_context
    (
    vx_context      context
    )
{
    return _create_spinst(context);
} /* vsi_nn_create_spinst_by_context() */

void vsi_nn_release_spinst
    (
    vsi_nn_spinst_t ** spinst
    )
{
    vsi_nn_spinst_t * ptr;
    ptr = *spinst;
    if( NULL != spinst && NULL != *spinst )
    {
        if( NULL != ptr->sp )
        {
            vxReleaseSPINST( &ptr->sp );
        }

        free( ptr );
        *spinst = NULL;
    }
} /* vsi_nn_release_spinst() */

vsi_status vsi_nn_set_spinst_attr
    (
    vsi_nn_spinst_t * spinst,
    const vsi_nn_spinst_attr_t attrs
    )
{
    vsi_status status;
    uint32_t constant_data[5] = {0};

    status = VSI_SUCCESS;
    if( NULL == spinst )
    {
        return VSI_FAILURE;
    }

    status  = vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_INPUT_TILE_MAPPING,
        attrs.input_tile_mapping);
    status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_OUTPUT_COLLAPSE_X,
        attrs.output_collapse_x);
    status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_OUTPUT_COLLAPSE_Y,
        attrs.output_collapse_y);
    status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_OUTPUT_COLLAPSE_Z,
        attrs.output_collapse_z);

    status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_PROG_INIT_INSTR_NUM,
        attrs.prog_init_instr_num);
    status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_PROG_LOOP_INSTR_NUM,
        attrs.prog_loop_instr_num);
    status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_PROG_COMPLETE_INSTR_NUM,
        attrs.prog_complete_instr_num);
    status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_PROG_ROUNDING_MODE,
        attrs.prog_rounding_mode);
    status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_INPUT_SETUP,
        attrs.input_setup);

    status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_IGNORED_LEADING_OUTPUTS,
        attrs.ignored_leading_outputs);
    status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_FLUSH_CYCLE_NUM,
        attrs.flush_cycle_num);
    status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_IGNORED_LEADING_V11_WR,
        attrs.ignored_leading_v11_wr);
    status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_IGNORED_LEADING_V12_WR,
        attrs.ignored_leading_v12_wr);
    status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_IGNORED_LEADING_V11_RD,
        attrs.ignored_leading_v11_rd);
    status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_IGNORED_LEADING_V12_RD,
        attrs.ignored_leading_v12_rd);

    status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_CH0_POST_REDISTRIBUTE,
        attrs.ch0_post_redistribute);
    status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_CH1_POST_REDISTRIBUTE,
        attrs.ch1_post_redistribute);
    status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_V11_RESET_AT_START,
        attrs.v11_reset_at_start);
    status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_V12_RESET_AT_START,
        attrs.v12_reset_at_start);
    status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_V11_PUSH_POP_CONFIG,
        attrs.v11_push_pop_config);
    status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_V12_PUSH_POP_CONFIG,
        attrs.v12_push_pop_config);

    status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_ACCELERATOR_INPUT_SELECT,
        attrs.accelerator_input_select);
    status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_IGNORED_LEADING_ACC_OUT,
        attrs.ignored_leading_acc_out);
    status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_SUM_ENGINE_RESET,
        attrs.sum_engine_reset);
    status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_SUM_ENGINE_CONTROL,
        attrs.sum_engine_control);
    status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_SUM_ENGINE_NUM_CH_MINUS_ONE,
        attrs.sum_engine_num_ch_minus_one);
    status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_SUM_ENGINE_2D_ACCUM_STORAGE,
        attrs.sum_engine_2d_accum_storeage);

    status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_SUM_ENGINE_OP_SELECT,
        attrs.sum_engine_op_select);
    status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_NUM_OF_ELEMENTS_PER_LOOP_PER_INPUT,
        attrs.num_of_elements_per_loop_per_input);

    constant_data[0] = *(uint32_t *)&attrs.init_r3;
    constant_data[1] = *(uint32_t *)&attrs.init_r4;
    constant_data[2] = *(uint32_t *)&attrs.init_r5;
    constant_data[3] = *(uint32_t *)&attrs.init_r6;
    constant_data[4] = *(uint32_t *)&attrs.init_r7;

    if (attrs.load_const_bits.load_const_sr3)
    {
        status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_CONST0,
            constant_data[0]);
    }
    if (attrs.load_const_bits.load_const_sr4)
    {
        status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_CONST1,
            constant_data[1]);
    }
    if (attrs.load_const_bits.load_const_sr5)
    {
        status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_CONST2,
            constant_data[2]);
    }
    if (attrs.load_const_bits.load_const_sr6)
    {
        status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_CONST3,
            constant_data[3]);
    }
    if (attrs.load_const_bits.load_const_sr7)
    {
        status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_CONST4,
            constant_data[4]);
    }

    status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_SPLIT_AXIS,
        attrs.split_axis);
    status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_SPLIT_MAX_SIZE,
        attrs.split_max_vector_depth);
    status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_TILEX_EQUAL_IMGX,
        attrs.split_tilex_equal_imgx);

    status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_NUM_OF_V11_RD_IN_FLUSH_CYCLE,
        attrs.num_of_v11_rd_in_flush_cycle);
    status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_NUM_OF_V12_RD_IN_FLUSH_CYCLE,
        attrs.num_of_v12_rd_in_flush_cycle);
    status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_NUM_OF_V11_WR_IN_FLUSH_CYCLE,
        attrs.num_of_v11_wr_in_flush_cycle);
    status |= vxSetAttributeToSPINST(spinst->sp, VSI_NN_SP_ATTRIBUTE_NUM_OF_V12_WR_IN_FLUSH_CYCLE,
        attrs.num_of_v12_wr_in_flush_cycle);

    return status;
} /* vsi_nn_set_spinst_attr() */

vsi_status vsi_nn_add_units_to_spinst
    (
    vsi_nn_spinst_t * spinst,
    vsi_nn_spinst_unit_param *inst_units,
    uint8_t unit_count
    )
{
    vsi_status status = VSI_SUCCESS;

    if ( NULL == spinst || NULL == inst_units )
    {
        return VSI_FAILURE;
    }

    status = vxAddOneInstToSPINST(spinst->sp, (vx_spinst_unit_param *)inst_units, unit_count);

    return status;
} /* vsi_nn_add_spinst_insts() */

vsi_status vsi_nn_add_spinst_insts
    (
    vsi_nn_spinst_t * spinst,
    vsi_nn_spinst_inst_param *insts,
    int32_t insts_count
    )
{
    vsi_status status = VSI_SUCCESS;
    int32_t i = 0;

    if ( NULL == spinst || NULL == insts )
    {
        return VSI_FAILURE;
    }

    for (i = 0; i < insts_count; i++)
    {
        status |= vsi_nn_add_units_to_spinst(spinst, insts[i].inst_units, insts[i].unit_count);
    }

    return status;
} /* vsi_nn_add_spinst_insts() */

void vsi_nn_init_spinst_attr
    (
    vsi_nn_spinst_attr_t * attrs
    )
{
    memset(attrs, 0, sizeof(vsi_nn_spinst_attr_t));

    /*default per loop to process one input or output pixel*/
    attrs->num_of_elements_per_loop_per_input = 1;
    attrs->accelerator_input_select = VSI_NN_SP_ACCELERATOR_IN_FROM_ACCEL;
    attrs->split_axis = VSI_SP_ATTR_SPLIT_ON_AXIS_XYZ;
} /* vsi_nn_init_spinst_attr() */

#endif