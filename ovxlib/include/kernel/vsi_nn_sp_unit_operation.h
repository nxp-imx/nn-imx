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

#ifndef _VSI_NN_SP_UNIT_OPERATION_H
#define _VSI_NN_SP_UNIT_OPERATION_H

#if VX_STREAM_PROCESSOR_SUPPORT

#include <stdint.h>
#include "kernel/vsi_nn_spinst.h"

__BEGIN_DECLS
/**
 * stream processor unit
 */
typedef enum
{
    VSI_NN_SP_UNIT_FADD = VX_SP_INST_TYPE_FADD,
    VSI_NN_SP_UNIT_FMUL = VX_SP_INST_TYPE_FMULT,
    VSI_NN_SP_UNIT_MOVE = VX_SP_INST_TYPE_MOVE,
    VSI_NN_SP_UNIT_PWL  = VX_SP_INST_TYPE_PWL,
} vsi_nn_sp_unit_type_e;

/**
 * stream processor registers
 */
typedef enum
{
    VSI_NN_SP_SRIN      =  VX_SP_INST_SPINOUT,
    VSI_NN_SP_SROUT     =  VX_SP_INST_SPINOUT,
    VSI_NN_SP_SR1       =  VX_SP_INST_SR1,
    VSI_NN_SP_SR2       =  VX_SP_INST_SR2,
    VSI_NN_SP_SR3       =  VX_SP_INST_SR3,
    VSI_NN_SP_SR4       =  VX_SP_INST_SR4,
    VSI_NN_SP_SR5       =  VX_SP_INST_SR5,
    VSI_NN_SP_SR6       =  VX_SP_INST_SR6,
    VSI_NN_SP_SR7       =  VX_SP_INST_SR7,
    VSI_NN_SP_SR8       =  VX_SP_INST_SR8,
    VSI_NN_SP_SR9       =  VX_SP_INST_SR9,
    VSI_NN_SP_SR10      =  VX_SP_INST_SR10,
    VSI_NN_SP_VR11      =  VX_SP_INST_VR11,
    VSI_NN_SP_VR12      =  VX_SP_INST_VR12,
    VSI_NN_SP_PWLMUL    =  VX_SP_INST_SETUPOUT,
    VSI_NN_SP_PWLADD    =  VX_SP_INST_SETUPOUT,
    VSI_NN_SP_ACC       =  VX_SP_INST_SETUPOUT,
} vsi_nn_sp_src_dst_e;

/**
 * stream processor pwl operations
 */
typedef enum
{
    VSI_NN_SP_PWL_IDLE      = VX_SP_INST_TYPE_PWL_IDLE,
    VSI_NN_SP_PWL_SETUP_0   = VX_SP_INST_TYPE_PWL_SETUP_0,
    VSI_NN_SP_PWL_SETUP_1   = VX_SP_INST_TYPE_PWL_SETUP_1,
    VSI_NN_SP_PWL_SETUP_2   = VX_SP_INST_TYPE_PWL_SETUP_2,

    VSI_NN_SP_PWL_OP_END    = VX_SP_INST_TYPE_PWL_COUNT
} vsi_nn_sp_pwl_op_e;

/**
 * stream processor fmul operations
 */
typedef enum
{
    VSI_NN_SP_FMUL_IDLE       = VX_SP_INST_TYPE_FMULT_IDLE,
    VSI_NN_SP_FMUL_MUL        = VX_SP_INST_TYPE_FMULT_MUL,
    VSI_NN_SP_FMUL_MUL_CLAMP  = VX_SP_INST_TYPE_FMULT_MUL_CLAMP,

    VSI_NN_SP_FMUL_OP_END     = VX_SP_INST_TYPE_FMULT_COUNT
} vsi_nn_sp_fmul_op_e;

/**
 * stream processor fadd operations
 */
typedef enum
{
    VSI_NN_SP_FADD_IDLE       = VX_SP_INST_TYPE_FADD_IDLE,
    VSI_NN_SP_FADD_ADD        = VX_SP_INST_TYPE_FADD_ADD,
    VSI_NN_SP_FADD_SUB        = VX_SP_INST_TYPE_FADD_SUB,

    VSI_NN_SP_FADD_OP_END     = VX_SP_INST_TYPE_FADD_COUNT
} vsi_nn_sp_fadd_op_e;

/**
 * stream processor move operations
 */
typedef enum
{
    VSI_NN_SP_MOVE_IDLE       = VX_SP_INST_TYPE_MOVE_IDLE,
    VSI_NN_SP_MOVE_MOVE       = VX_SP_INST_TYPE_MOVE_MOVE,
    VSI_NN_SP_MOVE_SEL0       = VX_SP_INST_TYPE_MOVE_SEL0,
    VSI_NN_SP_MOVE_SEL1       = VX_SP_INST_TYPE_MOVE_SEL1,
    VSI_NN_SP_MOVE_IMMD       = VX_SP_INST_TYPE_MOVE_IMMD,
    VSI_NN_SP_MOVE_ABS        = VX_SP_INST_TYPE_MOVE_ABS,

    VSI_NN_SP_MOVE_OP_END     = VX_SP_INST_TYPE_MOVE_COUNT
} vsi_nn_sp_move_op_e;

/**
 * stream processor input setup
 */
typedef enum _vsi_nn_sp_input_setup
{
    VSI_NN_SP_INPUT_SETUP_SINGLE_INPUT         = VX_SP_ATTRIBUTE_INPUT_SETUP_SINGLE_INPUT,
    VSI_NN_SP_INPUT_SETUP_INTERLEAVE_TWO_INPUT = VX_SP_ATTRIBUTE_INPUT_SETUP_INTERLEAVE_TWO_INPUTS,
    VSI_NN_SP_INPUT_SETUP_V11                  = VX_SP_ATTRIBUTE_INPUT_SETUP_V11,
    VSI_NN_SP_INPUT_SETUP_V12                  = VX_SP_ATTRIBUTE_INPUT_SETUP_V12
} vsi_nn_sp_input_setup_e;

/**
 * stream processor channel post redistribute
 */
typedef enum
{
    VSI_NN_SP_CH_POST_REDISTRIBUTE_DIST_DISABLE   = VX_SP_ATTRIBUTE_CH_POST_REDISTRIBUTE_DISABLED,
    VSI_NN_SP_CH_POST_REDISTRIBUTE_SCALAR_GATHER  = VX_SP_ATTRIBUTE_CH_POST_REDISTRIBUTE_SCALAR_GATHER,
    VSI_NN_SP_CH_POST_REDISTRIBUTE_VECTOR_GATHER  = VX_SP_ATTRIBUTE_CH_POST_REDISTRIBUTE_VECTOR_GATHER,
    VSI_NN_SP_CH_POST_REDISTRIBUTE_VECTOR_SCATTER = VX_SP_ATTRIBUTE_CH_POST_REDISTRIBUTE_VECTOR_SCATTER,
} vsi_nn_sp_ch_post_redistribute_e;

/**
 * stream processor push pop config
 */
typedef enum
{
    VSI_NN_SP_PUSH_POP_EVERY_READ   = VX_SP_ATTRIBUTE_V_POP_CONFIG_EVERY_READ,
    VSI_NN_SP_PUSH_POP_EVERY_ROW    = VX_SP_ATTRIBUTE_V_POP_CONFIG_EVERY_ROW
} vsi_nn_sp_pop_config_e;

/**
 * stream processor rounding mode
 */
typedef enum
{
    VSI_NN_SP_ROUNDING_RTNE   = VX_SP_ATTRIBUTE_PROG_ROUNDING_MODE_RTNE,
    VSI_NN_SP_ROUNDING_STICKY = VX_SP_ATTRIBUTE_PROG_ROUNDING_MODE_STICKY,
} vsi_nn_sp_rounding_mode_e;

/**
 * stream processor accumulator input select
 */
typedef enum
{
    VSI_NN_SP_ACCELERATOR_IN_FROM_OUTPUT = VX_SP_ATTRIBUTE_ACCELERATOR_INPUT_SELECT_FROM_OUTPUT,
    VSI_NN_SP_ACCELERATOR_IN_FROM_ACCEL  = VX_SP_ATTRIBUTE_ACCELERATOR_INPUT_SELECT_FROM_ACCLERATOR,
} vsi_nn_sp_acc_in_select_e;

/**
 * stream processor sum engine control
 */
typedef enum
{
    VSI_NN_SP_ACCUM_INTERNAL = VX_SP_ATTRIBUTE_SUM_ENGINE_CONTROL_ACCUM_INTERNAL,
    VSI_NN_SP_ACCUM_1D       = VX_SP_ATTRIBUTE_SUM_ENGINE_CONTROL_ACCUM_1D,
    VSI_NN_SP_ACCUM_2D       = VX_SP_ATTRIBUTE_SUM_ENGINE_CONTROL_ACCUM_2D
} vsi_nn_sp_sum_engine_control_e;

/**
 * stream processor sum engine number of channel
 */
typedef enum
{
    VSI_NN_SP_SUM_ENGINE_NUM_CH_ONE_CH  = VX_SP_ATTRIBUTE_SUM_ENGINE_NUM_CH_MINUS_ONE_ONE_CH,
    VSI_NN_SP_SUM_ENGINE_NUM_CH_TWO_CH  = VX_SP_ATTRIBUTE_SUM_ENGINE_NUM_CH_MINUS_ONE_TWO_CH
} vsi_nn_sp_sum_engine_num_ch_minus1_e;

/**
 * stream processor sum engine 2D acumulation storage
 */
typedef enum
{
    VSI_NN_SP_ACCM_STOREAGE_SAME      = VX_SP_ATTRIBUTE_SUM_ENGINE_2D_ACCUM_STORAGE_SAME,
    VSI_NN_SP_ACCM_STOREAGE_DIFFERENT = VX_SP_ATTRIBUTE_SUM_ENGINE_2D_ACCUM_STORAGE_DIFFERENT
} vsi_nn_sp_sum_engine_2d_accum_storage_e;

/**
 * stream processor sum engine reset
 */
typedef enum
{
    VSI_NN_SP_SUM_ENGINE_RESET_NONE  = VX_SP_ATTRIBUTE_SUM_ENGINE_RESET_NONE,
    VSI_NN_SP_SUM_ENGINE_RESET_START_FROM_ZERO = VX_SP_ATTRIBUTE_SUM_ENGINE_RESET_RESET,
} vsi_nn_sp_sum_engine_reset_e;

/**
 * stream processor vector reset
 */
typedef enum
{
    VSI_NN_SP_V_RESET_AT_START_NONE  = VX_SP_ATTRIBUTE_V_RESET_AT_START_NONE,
    VSI_NN_SP_V_RESET_AT_START_RESET = VX_SP_ATTRIBUTE_V_RESET_AT_START_RESET,
} vsi_nn_sp_v_reset_at_start_e;

/**
 * stream processor input tile mapping
 */
typedef enum
{
    VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_XYMERGE = VX_SP_ATTRIBUTE_INPUT_TILE_MAPPING_XYMERGE,
    VSI_NN_SP_ATTR_INPUT_TILE_MAPPING_YZMERGE = VX_SP_ATTRIBUTE_INPUT_TILE_MAPPING_YZMERGE,
} vsi_nn_sp_attribute_input_tile_mapping_e;

/**
 * stream processor output collapse
 */
typedef enum
{
    VSI_NN_SP_ATTR_OUTPUT_COLLAPSE_DISABLED = VX_SP_ATTRIBUTE_OUTPUT_COLLAPSE_DISABLED,
    VSI_NN_SP_ATTR_OUTPUT_COLLAPSE_ENABLED  = VX_SP_ATTRIBUTE_OUTPUT_COLLAPSE_ENABLED,
} vsi_nn_sp_attribute_output_collapse_e;

/**
 * add a nop operation
 *
 * @param[in] one_inst vsi_nn_spinst_inst_param handle
 *
 * @return VSI_SUCCESS on success, or error core otherwise.
 */
vsi_status vsi_nn_sp_nop
    (
    vsi_nn_spinst_inst_param *one_inst
    );

/**
 * add a add operation, dst = src0 + src1
 *
 * @param[in] one_inst vsi_nn_spinst_inst_param handle
 * @param[in] src0 a unit input
 * @param[in] src1 a unit input
 * @param[in] dst  a unit output
 * @return VSI_SUCCESS on success, or error core otherwise.
 */
vsi_status vsi_nn_sp_add
    (
    vsi_nn_spinst_inst_param *one_inst,
    uint8_t src0,
    uint8_t src1,
    uint8_t dst
    );

/**
 * add a sub operation, dst = src0 - src1
 *
 * @param[in] one_inst vsi_nn_spinst_inst_param handle
 * @param[in] src0 a unit input
 * @param[in] src1 a unit input
 * @param[in] dst  a unit output
 * @return VSI_SUCCESS on success, or error core otherwise.
 */
vsi_status vsi_nn_sp_sub
    (
    vsi_nn_spinst_inst_param *one_inst,
    uint8_t src0,
    uint8_t src1,
    uint8_t dst
    );

/**
 * add src0 to add unit src0 and set fadd unit idle
 *
 * @param[in] one_inst vsi_nn_spinst_inst_param handle
 * @param[in] src0 a unit input
 * @return VSI_SUCCESS on success, or error core otherwise.
 */
vsi_status vsi_nn_sp_fa_nop
    (
    vsi_nn_spinst_inst_param *one_inst,
    uint8_t src0
    );

/**
 * add a mul operation, dst = src0 * src1
 *
 * @param[in] one_inst vsi_nn_spinst_inst_param handle
 * @param[in] src0 a unit input
 * @param[in] src1 a unit input
 * @param[in] dst  a unit output
 * @return VSI_SUCCESS on success, or error core otherwise.
 */
vsi_status vsi_nn_sp_mul
    (
    vsi_nn_spinst_inst_param *one_inst,
    uint8_t src0,
    uint8_t src1,
    uint8_t dst
    );

/**
 * add a mul and clamp operation, dst = clamp(src0 * src1, r7, r6)
 *
 * @param[in] one_inst vsi_nn_spinst_inst_param handle
 * @param[in] src0 a unit input
 * @param[in] src1 a unit input
 * @param[in] dst  a unit output
 * @return VSI_SUCCESS on success, or error core otherwise.
 */
vsi_status vsi_nn_sp_mul_clamp
    (
    vsi_nn_spinst_inst_param *one_inst,
    uint8_t src0,
    uint8_t src1,
    uint8_t dst
    );

/**
 * add a move operation, dst = src1
 *
 * @param[in] one_inst vsi_nn_spinst_inst_param handle
 * @param[in] src1 a unit input
 * @param[in] dst  a unit output
 * @return VSI_SUCCESS on success, or error core otherwise.
 */
vsi_status vsi_nn_sp_move
    (
    vsi_nn_spinst_inst_param *one_inst,
    uint8_t src1,
    uint8_t dst
    );

/**
 * add a select0 operation, dst = src0 > 0 ? src1[0] : src1[1]
 *
 * @param[in] one_inst vsi_nn_spinst_inst_param handle
 * @param[in] src0 a unit input
 * @param[in] src1 a unit input
 * @param[in] dst  a unit output
 * @return VSI_SUCCESS on success, or error core otherwise.
 */
vsi_status vsi_nn_sp_move_sel0
    (
    vsi_nn_spinst_inst_param *one_inst,
    uint8_t src0,
    uint8_t src1,
    uint8_t dst
    );

/**
 * add a select1 operation, dst = src0 > 0 ? src1 : ADD-src0
 *
 * @param[in] one_inst vsi_nn_spinst_inst_param handle
 * @param[in] src0 a unit input
 * @param[in] src1 a unit input
 * @param[in] dst  a unit output
 * @return VSI_SUCCESS on success, or error core otherwise.
 */
vsi_status vsi_nn_sp_move_sel1
    (
    vsi_nn_spinst_inst_param *one_inst,
    uint8_t src0,
    uint8_t src1,
    uint8_t dst
    );

/**
 * add a move operation, dst = constant
 *
 * @param[in] one_inst vsi_nn_spinst_inst_param handle
 * @param[in] constant a unit input
 * @param[in] dst  a unit output
 * @return VSI_SUCCESS on success, or error core otherwise.
 */
vsi_status vsi_nn_sp_move_constant
    (
    vsi_nn_spinst_inst_param *one_inst,
    float constant,
    uint8_t dst
    );

/**
 * add a move operation, dst = abs(src1)
 *
 * @param[in] one_inst vsi_nn_spinst_inst_param handle
 * @param[in] src1 a unit input
 * @param[in] dst  a unit output
 * @return VSI_SUCCESS on success, or error core otherwise.
 */
vsi_status vsi_nn_sp_abs
    (
    vsi_nn_spinst_inst_param *one_inst,
    uint8_t src1,
    uint8_t dst
    );

/**
 * add a pwl operation, {dst,FMInA,FMInB,FAInA,FAInB} = Setup0(MV-src0)
 *
 * @param[in] one_inst vsi_nn_spinst_inst_param handle
 * @param[in] src0 a unit input
 * @param[in] dst  a unit output
 * @return VSI_SUCCESS on success, or error core otherwise.
 */
vsi_status vsi_nn_sp_pwl_setup0
    (
    vsi_nn_spinst_inst_param *one_inst,
    uint8_t src0,
    uint8_t dst
    );

/**
 * add a pwl sigmoid operation, {dst,FMInA,FMInB,FAInA,FAInB} = Setup1(MV-src0)
 *
 * @param[in] one_inst vsi_nn_spinst_inst_param handle
 * @param[in] src0 a unit input
 * @param[in] dst  a unit output
 * @return VSI_SUCCESS on success, or error core otherwise.
 */
vsi_status vsi_nn_sp_pwl_sigmoid
    (
    vsi_nn_spinst_inst_param *one_inst,
    uint8_t src0,
    uint8_t dst
    );

/**
 * add a pwl tanh operation, {dst,FMInA,FMInB,FAInA,FAInB} = Setup2(MV-src0)
 *
 * @param[in] one_inst vsi_nn_spinst_inst_param handle
 * @param[in] src0 a unit input
 * @param[in] dst  a unit output
 * @return VSI_SUCCESS on success, or error core otherwise.
 */
vsi_status vsi_nn_sp_pwl_tanh
    (
    vsi_nn_spinst_inst_param *one_inst,
    uint8_t src0,
    uint8_t dst
    );

__END_DECLS

#endif

#endif
