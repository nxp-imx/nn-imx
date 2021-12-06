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

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include "vsi_nn_context.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_types.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_node.h"
#include "vsi_nn_log.h"
#include <float.h>
#include "vsi_nn_error.h"
#include "utils/vsi_nn_dtype_util_prv.h"
#include "vsi_nn_tensor_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "kernel/vsi_nn_spinst.h"
#include "kernel/vsi_nn_sp_unit_operation.h"
#include "utils/vsi_nn_dtype_util.h"

#if VX_STREAM_PROCESSOR_SUPPORT

vsi_status vsi_nn_sp_nop
    (
    vsi_nn_spinst_inst_param *one_inst
    )
{
    vsi_status status = VSI_SUCCESS;

    if ( NULL == one_inst || one_inst->unit_count >= VSI_NN_MAX_SP_UNIT_NUM )
    {
        return VSI_FAILURE;
    }

    one_inst->unit_count ++;

    return status;
} /* vsi_nn_sp_nop() */

vsi_status vsi_nn_sp_add
    (
    vsi_nn_spinst_inst_param *one_inst,
    uint8_t src0,
    uint8_t src1,
    uint8_t dst
    )
{
    vsi_status status = VSI_SUCCESS;
    uint32_t index = one_inst->unit_count;

    if ( NULL == one_inst || one_inst->unit_count >= VSI_NN_MAX_SP_UNIT_NUM )
    {
        return VSI_FAILURE;
    }

    one_inst->inst_units[index].unit_type = VSI_NN_SP_UNIT_FADD;
    one_inst->inst_units[index].unit.op = VSI_NN_SP_FADD_ADD;
    one_inst->inst_units[index].unit.var.src0 = src0;
    one_inst->inst_units[index].unit.var.src1 = src1;
    one_inst->inst_units[index].unit.var.dst = dst;

    one_inst->unit_count ++;

    return status;
} /* vsi_nn_sp_add() */

vsi_status vsi_nn_sp_sub
    (
    vsi_nn_spinst_inst_param *one_inst,
    uint8_t src0,
    uint8_t src1,
    uint8_t dst
    )
{
    vsi_status status = VSI_SUCCESS;
    uint32_t index = one_inst->unit_count;

    if ( NULL == one_inst || one_inst->unit_count >= VSI_NN_MAX_SP_UNIT_NUM )
    {
        return VSI_FAILURE;
    }

    one_inst->inst_units[index].unit_type = VSI_NN_SP_UNIT_FADD;
    one_inst->inst_units[index].unit.op = VSI_NN_SP_FADD_SUB;
    one_inst->inst_units[index].unit.var.src0 = src0;
    one_inst->inst_units[index].unit.var.src1 = src1;
    one_inst->inst_units[index].unit.var.dst = dst;

    one_inst->unit_count ++;

    return status;
} /* vsi_nn_sp_sub() */

vsi_status vsi_nn_sp_fa_nop
    (
    vsi_nn_spinst_inst_param *one_inst,
    uint8_t src0
    )
{
    vsi_status status = VSI_SUCCESS;
    uint32_t index = one_inst->unit_count;

    if ( NULL == one_inst || one_inst->unit_count >= VSI_NN_MAX_SP_UNIT_NUM )
    {
        return VSI_FAILURE;
    }

    one_inst->inst_units[index].unit_type = VSI_NN_SP_UNIT_FADD;
    one_inst->inst_units[index].unit_type = VSI_NN_SP_FADD_IDLE;
    one_inst->inst_units[index].unit.var.src0 = src0;

    one_inst->unit_count ++;

    return status;
} /* vsi_nn_sp_fa_nop() */

vsi_status vsi_nn_sp_mul
    (
    vsi_nn_spinst_inst_param *one_inst,
    uint8_t src0,
    uint8_t src1,
    uint8_t dst
    )
{
    vsi_status status = VSI_SUCCESS;
    uint32_t index = one_inst->unit_count;

    if ( NULL == one_inst || one_inst->unit_count >= VSI_NN_MAX_SP_UNIT_NUM )
    {
        return VSI_FAILURE;
    }

    one_inst->inst_units[index].unit_type = VSI_NN_SP_UNIT_FMUL;
    one_inst->inst_units[index].unit.op = VSI_NN_SP_FMUL_MUL;
    one_inst->inst_units[index].unit.var.src0 = src0;
    one_inst->inst_units[index].unit.var.src1 = src1;
    one_inst->inst_units[index].unit.var.dst = dst;

    one_inst->unit_count ++;

    return status;
} /* vsi_nn_sp_mul() */

vsi_status vsi_nn_sp_mul_clamp
    (
    vsi_nn_spinst_inst_param *one_inst,
    uint8_t src0,
    uint8_t src1,
    uint8_t dst
    )
{
    vsi_status status = VSI_SUCCESS;
    uint32_t index = one_inst->unit_count;

    if ( NULL == one_inst || one_inst->unit_count >= VSI_NN_MAX_SP_UNIT_NUM )
    {
        return VSI_FAILURE;
    }

    one_inst->inst_units[index].unit_type = VSI_NN_SP_UNIT_FMUL;
    one_inst->inst_units[index].unit.op = VSI_NN_SP_FMUL_MUL_CLAMP;
    one_inst->inst_units[index].unit.var.src0 = src0;
    one_inst->inst_units[index].unit.var.src1 = src1;
    one_inst->inst_units[index].unit.var.dst = dst;

    one_inst->unit_count ++;

    return status;
} /* vsi_nn_sp_mul_clamp() */

vsi_status vsi_nn_sp_move
    (
    vsi_nn_spinst_inst_param *one_inst,
    uint8_t src1,
    uint8_t dst
    )
{
    vsi_status status = VSI_SUCCESS;
    uint32_t index = one_inst->unit_count;

    if ( NULL == one_inst || one_inst->unit_count >= VSI_NN_MAX_SP_UNIT_NUM )
    {
        return VSI_FAILURE;
    }

    one_inst->inst_units[index].unit_type = VSI_NN_SP_UNIT_MOVE;
    one_inst->inst_units[index].unit.op = VSI_NN_SP_MOVE_MOVE;
    one_inst->inst_units[index].unit.var.src1 = src1;
    one_inst->inst_units[index].unit.var.dst = dst;

    one_inst->unit_count ++;

    return status;
} /* vsi_nn_sp_mov() */

vsi_status vsi_nn_sp_move_sel0
    (
    vsi_nn_spinst_inst_param *one_inst,
    uint8_t src0,
    uint8_t src1,
    uint8_t dst
    )
{
    vsi_status status = VSI_SUCCESS;
    uint32_t index = one_inst->unit_count;

    if ( NULL == one_inst || one_inst->unit_count >= VSI_NN_MAX_SP_UNIT_NUM )
    {
        return VSI_FAILURE;
    }

    one_inst->inst_units[index].unit_type = VSI_NN_SP_UNIT_MOVE;
    one_inst->inst_units[index].unit.op = VSI_NN_SP_MOVE_SEL0;
    one_inst->inst_units[index].unit.var.src0 = src0;
    one_inst->inst_units[index].unit.var.src1 = src1;
    one_inst->inst_units[index].unit.var.dst = dst;

    one_inst->unit_count ++;

    return status;
} /* vsi_nn_sp_mov_sel0() */

vsi_status vsi_nn_sp_move_sel1
    (
    vsi_nn_spinst_inst_param *one_inst,
    uint8_t src0,
    uint8_t src1,
    uint8_t dst
    )
{
    vsi_status status = VSI_SUCCESS;
    uint32_t index = one_inst->unit_count;

    if ( NULL == one_inst || one_inst->unit_count >= VSI_NN_MAX_SP_UNIT_NUM )
    {
        return VSI_FAILURE;
    }

    one_inst->inst_units[index].unit_type = VSI_NN_SP_UNIT_MOVE;
    one_inst->inst_units[index].unit.op = VSI_NN_SP_MOVE_SEL1;
    one_inst->inst_units[index].unit.var.src0 = src0;
    one_inst->inst_units[index].unit.var.src1 = src1;
    one_inst->inst_units[index].unit.var.dst = dst;

    one_inst->unit_count ++;

    return status;
} /* vsi_nn_sp_mov_sel1() */

vsi_status vsi_nn_sp_move_constant
    (
    vsi_nn_spinst_inst_param *one_inst,
    float constant,
    uint8_t dst
    )
{
    vsi_status status = VSI_SUCCESS;
    uint32_t index = one_inst->unit_count;

    if ( NULL == one_inst || one_inst->unit_count >= VSI_NN_MAX_SP_UNIT_NUM )
    {
        return VSI_FAILURE;
    }

    one_inst->inst_units[index].unit_type = VSI_NN_SP_UNIT_MOVE;
    one_inst->inst_units[index].unit.op = VSI_NN_SP_MOVE_IMMD;
    one_inst->inst_units[index].unit.var.constant = constant;
    one_inst->inst_units[index].unit.var.dst = dst;

    one_inst->unit_count ++;

    return status;
} /* vsi_nn_sp_move_constant() */

vsi_status vsi_nn_sp_abs
    (
    vsi_nn_spinst_inst_param *one_inst,
    uint8_t src1,
    uint8_t dst
    )
{
    vsi_status status = VSI_SUCCESS;
    uint32_t index = one_inst->unit_count;

    if ( NULL == one_inst || one_inst->unit_count >= VSI_NN_MAX_SP_UNIT_NUM )
    {
        return VSI_FAILURE;
    }

    one_inst->inst_units[index].unit_type = VSI_NN_SP_UNIT_MOVE;
    one_inst->inst_units[index].unit.op = VSI_NN_SP_MOVE_ABS;
    one_inst->inst_units[index].unit.var.src1 = src1;
    one_inst->inst_units[index].unit.var.dst = dst;

    one_inst->unit_count ++;

    return status;
} /* vsi_nn_sp_abs() */

vsi_status vsi_nn_sp_pwl_setup0
    (
    vsi_nn_spinst_inst_param *one_inst,
    uint8_t src0,
    uint8_t dst
    )
{
    vsi_status status = VSI_SUCCESS;
    uint32_t index = one_inst->unit_count;

    if ( NULL == one_inst || one_inst->unit_count >= VSI_NN_MAX_SP_UNIT_NUM )
    {
        return VSI_FAILURE;
    }

    VSI_ASSERT( dst == VSI_NN_SP_SR1 );

    one_inst->inst_units[index].unit_type = VSI_NN_SP_UNIT_PWL;
    one_inst->inst_units[index].unit.op = VSI_NN_SP_PWL_SETUP_0;
    one_inst->inst_units[index].unit.var.src0 = src0;
    one_inst->inst_units[index].unit.var.dst = dst;

    one_inst->unit_count ++;

    return status;
} /* vsi_nn_sp_pwl_setup0() */

vsi_status vsi_nn_sp_pwl_sigmoid
    (
    vsi_nn_spinst_inst_param *one_inst,
    uint8_t src0,
    uint8_t dst
    )
{
    vsi_status status = VSI_SUCCESS;
    uint32_t index = one_inst->unit_count;

    if ( NULL == one_inst || one_inst->unit_count >= VSI_NN_MAX_SP_UNIT_NUM )
    {
        return VSI_FAILURE;
    }

    VSI_ASSERT( dst == VSI_NN_SP_SR1 );

    one_inst->inst_units[index].unit_type = VSI_NN_SP_UNIT_PWL;
    one_inst->inst_units[index].unit.op = VSI_NN_SP_PWL_SETUP_1;
    one_inst->inst_units[index].unit.var.src0 = src0;
    one_inst->inst_units[index].unit.var.dst = dst;

    one_inst->unit_count ++;

    return status;
} /* vsi_nn_sp_pwl_sigmoid() */

vsi_status vsi_nn_sp_pwl_tanh
    (
    vsi_nn_spinst_inst_param *one_inst,
    uint8_t src0,
    uint8_t dst
    )
{
    vsi_status status = VSI_SUCCESS;
    uint32_t index = one_inst->unit_count;

    if ( NULL == one_inst || one_inst->unit_count >= VSI_NN_MAX_SP_UNIT_NUM )
    {
        return VSI_FAILURE;
    }

    VSI_ASSERT( dst == VSI_NN_SP_SR1 );

    one_inst->inst_units[index].unit_type = VSI_NN_SP_UNIT_PWL;
    one_inst->inst_units[index].unit.op = VSI_NN_SP_PWL_SETUP_2;
    one_inst->inst_units[index].unit.var.src0 = src0;
    one_inst->inst_units[index].unit.var.dst = dst;

    one_inst->unit_count ++;

    return status;
} /* vsi_nn_sp_pwl_tanh() */

#endif