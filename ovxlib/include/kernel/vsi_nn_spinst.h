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

#ifndef _VSI_NN_SPINST_H
#define _VSI_NN_SPINST_H

#include "vsi_nn_platform.h"
#include "vsi_nn_types.h"

#if VX_STREAM_PROCESSOR_SUPPORT

__BEGIN_DECLS

#include <VX/vx_spinst.h>

/**
 * Maximum stream processor unit number
 *
 */
#define VSI_NN_MAX_SP_UNIT_NUM      (4)

#define VSI_NN_SP_ATTR_SET_CONST_TO_SR3(attr, data) \
    attr.init_r3 = data; \
    attr.load_const_bits.load_const_sr3 = 1

#define VSI_NN_SP_ATTR_SET_CONST_TO_SR4(attr, data) \
    attr.init_r4 = data; \
    attr.load_const_bits.load_const_sr4 = 1

#define VSI_NN_SP_ATTR_SET_CONST_TO_SR5_LOW_PRECISION(attr, data) \
    attr.init_r5 = data; \
    attr.load_const_bits.load_const_sr5 = 1

#define VSI_NN_SP_ATTR_SET_CONST_TO_SR6(attr, data) \
    attr.init_r6 = data; \
    attr.load_const_bits.load_const_sr6 = 1

#define VSI_NN_SP_ATTR_SET_CONST_TO_SR7(attr, data) \
    attr.init_r7 = data; \
    attr.load_const_bits.load_const_sr7 = 1

/**
 * stream processor split axis attribute
 */
typedef enum
{
    VSI_SP_ATTR_SPLIT_ON_AXIS_X     = VX_SP_ATTRIBUTE_SPLIT_ON_AXIS_X,
    VSI_SP_ATTR_SPLIT_ON_AXIS_Y     = VX_SP_ATTRIBUTE_SPLIT_ON_AXIS_Y,
    VSI_SP_ATTR_SPLIT_ON_AXIS_Z     = VX_SP_ATTRIBUTE_SPLIT_ON_AXIS_Z,
    VSI_SP_ATTR_SPLIT_ON_AXIS_XY    = VX_SP_ATTRIBUTE_SPLIT_ON_AXIS_XY,
    VSI_SP_ATTR_SPLIT_ON_AXIS_YZ    = VX_SP_ATTRIBUTE_SPLIT_ON_AXIS_YZ,
    VSI_SP_ATTR_SPLIT_ON_AXIS_XYZ   = VX_SP_ATTRIBUTE_SPLIT_ON_AXIS_XYZ,
}vsi_sp_attr_split_axis_e;

/**
 * spinst attribute
 */
typedef struct _vsi_nn_spinst_attr
{
    uint32_t input_tile_mapping;
    uint32_t output_collapse_x;
    uint32_t output_collapse_y;
    uint32_t output_collapse_z;

    uint32_t prog_init_instr_num;
    uint32_t prog_loop_instr_num;
    uint32_t prog_complete_instr_num;
    uint32_t prog_rounding_mode;
    uint32_t input_setup;

    uint32_t ignored_leading_outputs;
    uint32_t flush_cycle_num;
    uint32_t ignored_leading_v11_wr;
    uint32_t ignored_leading_v12_wr;
    uint32_t ignored_leading_v11_rd;
    uint32_t ignored_leading_v12_rd;

    uint32_t ch0_post_redistribute;
    uint32_t ch1_post_redistribute;
    uint32_t v11_reset_at_start;
    uint32_t v12_reset_at_start;
    uint32_t v11_push_pop_config;
    uint32_t v12_push_pop_config;
    uint32_t accelerator_input_select;
    uint32_t ignored_leading_acc_out;
    uint32_t sum_engine_reset;
    uint32_t sum_engine_control;
    uint32_t sum_engine_num_ch_minus_one;
    uint32_t sum_engine_2d_accum_storeage;
    uint32_t sum_engine_op_select;
    uint32_t num_of_elements_per_loop_per_input;
    uint32_t split_axis;
    uint32_t split_max_vector_depth;
    vsi_bool split_tilex_equal_imgx;
    uint32_t num_of_v11_rd_in_flush_cycle;
    uint32_t num_of_v12_rd_in_flush_cycle;
    uint32_t num_of_v11_wr_in_flush_cycle;
    uint32_t num_of_v12_wr_in_flush_cycle;

    float init_r3;
    float init_r4;
    float init_r5;
    float init_r6;
    float init_r7;

    struct {
        uint32_t load_const_sr3  : 1;
        uint32_t load_const_sr4  : 1;
        uint32_t load_const_sr5  : 1;
        uint32_t load_const_sr6  : 1;
        uint32_t load_const_sr7  : 1;
        uint32_t reserved        : 27;
    }load_const_bits;
} vsi_nn_spinst_attr_t;
/**
 * Spir structure
 */
typedef struct _vsi_nn_spinst
{
    /** OVX SPInst */
    vx_spinst sp;
} vsi_nn_spinst_t;

/**
 * stream processor attribute
 */
typedef enum
{
    VSI_NN_SP_ATTRIBUTE_NONE                        = VX_SP_ATTRIBUTE_NONE,

    VSI_NN_SP_ATTRIBUTE_INPUT_TILE_MAPPING          = VX_SP_ATTRIBUTE_INPUT_TILE_MAPPING,
    VSI_NN_SP_ATTRIBUTE_OUTPUT_COLLAPSE_X           = VX_SP_ATTRIBUTE_OUTPUT_COLLAPSE_X,
    VSI_NN_SP_ATTRIBUTE_OUTPUT_COLLAPSE_Y           = VX_SP_ATTRIBUTE_OUTPUT_COLLAPSE_Y,
    VSI_NN_SP_ATTRIBUTE_OUTPUT_COLLAPSE_Z           = VX_SP_ATTRIBUTE_OUTPUT_COLLAPSE_Z,

    VSI_NN_SP_ATTRIBUTE_PROG_INIT_INSTR_NUM         = VX_SP_ATTRIBUTE_PROG_INIT_INSTR_NUM,
    VSI_NN_SP_ATTRIBUTE_PROG_LOOP_INSTR_NUM         = VX_SP_ATTRIBUTE_PROG_LOOP_INSTR_NUM,
    VSI_NN_SP_ATTRIBUTE_PROG_COMPLETE_INSTR_NUM     = VX_SP_ATTRIBUTE_PROG_COMPLETE_INSTR_NUM,
    VSI_NN_SP_ATTRIBUTE_PROG_ROUNDING_MODE          = VX_SP_ATTRIBUTE_PROG_ROUNDING_MODE,
    VSI_NN_SP_ATTRIBUTE_INPUT_SETUP                 = VX_SP_ATTRIBUTE_INPUT_SETUP,

    VSI_NN_SP_ATTRIBUTE_IGNORED_LEADING_OUTPUTS     = VX_SP_ATTRIBUTE_IGNORED_LEADING_OUTPUTS,
    VSI_NN_SP_ATTRIBUTE_FLUSH_CYCLE_NUM             = VX_SP_ATTRIBUTE_FLUSH_CYCLE_NUM,
    VSI_NN_SP_ATTRIBUTE_IGNORED_LEADING_V11_WR      = VX_SP_ATTRIBUTE_IGNORED_LEADING_V11_WR,
    VSI_NN_SP_ATTRIBUTE_IGNORED_LEADING_V12_WR      = VX_SP_ATTRIBUTE_IGNORED_LEADING_V12_WR,
    VSI_NN_SP_ATTRIBUTE_IGNORED_LEADING_V11_RD      = VX_SP_ATTRIBUTE_IGNORED_LEADING_V11_RD,
    VSI_NN_SP_ATTRIBUTE_IGNORED_LEADING_V12_RD      = VX_SP_ATTRIBUTE_IGNORED_LEADING_V12_RD,

    VSI_NN_SP_ATTRIBUTE_CH0_POST_REDISTRIBUTE       = VX_SP_ATTRIBUTE_CH0_POST_REDISTRIBUTE,
    VSI_NN_SP_ATTRIBUTE_CH1_POST_REDISTRIBUTE       = VX_SP_ATTRIBUTE_CH1_POST_REDISTRIBUTE,
    VSI_NN_SP_ATTRIBUTE_V11_RESET_AT_START          = VX_SP_ATTRIBUTE_V11_RESET_AT_START,
    VSI_NN_SP_ATTRIBUTE_V12_RESET_AT_START          = VX_SP_ATTRIBUTE_V12_RESET_AT_START,
    VSI_NN_SP_ATTRIBUTE_V11_PUSH_POP_CONFIG         = VX_SP_ATTRIBUTE_V11_POP_CONFIG,
    VSI_NN_SP_ATTRIBUTE_V12_PUSH_POP_CONFIG         = VX_SP_ATTRIBUTE_V12_POP_CONFIG,
    VSI_NN_SP_ATTRIBUTE_ACCELERATOR_INPUT_SELECT    = VX_SP_ATTRIBUTE_ACCELERATOR_INPUT_SELECT,
    VSI_NN_SP_ATTRIBUTE_IGNORED_LEADING_ACC_OUT     = VX_SP_ATTRIBUTE_IGNORED_LEADING_ACC_OUT,
    VSI_NN_SP_ATTRIBUTE_SUM_ENGINE_RESET            = VX_SP_ATTRIBUTE_SUM_ENGINE_RESET,
    VSI_NN_SP_ATTRIBUTE_SUM_ENGINE_CONTROL          = VX_SP_ATTRIBUTE_SUM_ENGINE_CONTROL,
    VSI_NN_SP_ATTRIBUTE_SUM_ENGINE_NUM_CH_MINUS_ONE = VX_SP_ATTRIBUTE_SUM_ENGINE_NUM_CH_MINUS_ONE,
    VSI_NN_SP_ATTRIBUTE_SUM_ENGINE_2D_ACCUM_STORAGE = VX_SP_ATTRIBUTE_SUM_ENGINE_2D_ACCUM_STORAGE,
    VSI_NN_SP_ATTRIBUTE_SUM_ENGINE_OP_SELECT        = VX_SP_ATTRIBUTE_SUM_ENGINE_OP_SELECT,
    VSI_NN_SP_ATTRIBUTE_NUM_OF_ELEMENTS_PER_LOOP_PER_INPUT = VX_SP_ATTRIBUTE_NUM_OF_ELEMENTS_PER_LOOP_PER_INPUT,

    VSI_NN_SP_ATTRIBUTE_CONST0                      = VX_SP_ATTRIBUTE_CONST0,
    VSI_NN_SP_ATTRIBUTE_CONST1                      = VX_SP_ATTRIBUTE_CONST1,
    VSI_NN_SP_ATTRIBUTE_CONST2                      = VX_SP_ATTRIBUTE_CONST2,
    VSI_NN_SP_ATTRIBUTE_CONST3                      = VX_SP_ATTRIBUTE_CONST3,
    VSI_NN_SP_ATTRIBUTE_CONST4                      = VX_SP_ATTRIBUTE_CONST4,

    VSI_NN_SP_ATTRIBUTE_SPLIT_AXIS                  = VX_SP_ATTRIBUTE_SPLIT_AXIS,
    VSI_NN_SP_ATTRIBUTE_SPLIT_MAX_SIZE              = VX_SP_ATTRIBUTE_SPLIT_MAX_SIZE,
    VSI_NN_SP_ATTRIBUTE_TILEX_EQUAL_IMGX            = VX_SP_ATTRIBUTE_SPLIT_TILEX_EQUAL_INIMAGEX,

    VSI_NN_SP_ATTRIBUTE_NUM_OF_V11_RD_IN_FLUSH_CYCLE = VX_SP_ATTRIBUTE_NUM_OF_V11_RD_IN_FLUSH_CYCLE,
    VSI_NN_SP_ATTRIBUTE_NUM_OF_V12_RD_IN_FLUSH_CYCLE = VX_SP_ATTRIBUTE_NUM_OF_V12_RD_IN_FLUSH_CYCLE,
    VSI_NN_SP_ATTRIBUTE_NUM_OF_V11_WR_IN_FLUSH_CYCLE = VX_SP_ATTRIBUTE_NUM_OF_V11_WR_IN_FLUSH_CYCLE,
    VSI_NN_SP_ATTRIBUTE_NUM_OF_V12_WR_IN_FLUSH_CYCLE = VX_SP_ATTRIBUTE_NUM_OF_V12_WR_IN_FLUSH_CYCLE,

    VSI_NN_SP_ATTRIBUTE_COUNT                       = VX_SP_ATTRIBUTE_TOTAL_COUNT,
} vsi_nn_sp_attribute_e;

typedef struct _vsi_nn_spinst_unit_param
{
    /* vsi_nn_sp_unit_type_e */
    vsi_enum unit_type;

    struct
    {
        /* vsi_nn_sp_fmul/fadd/move/pwl_op_e */
        vsi_enum op;

        struct
        {
            /* vsi_nn_sp_src_dst_e */
            uint8_t src0;
            uint8_t src1;
            uint8_t dst;
            float constant;
        } var;
    } unit;
} vsi_nn_spinst_unit_param;

typedef struct _vsi_nn_spinst_inst_param
{
    vsi_nn_spinst_unit_param inst_units[VSI_NN_MAX_SP_UNIT_NUM];

    uint8_t unit_count;
} vsi_nn_spinst_inst_param;

/**
 * Create a new spinst
 *
 * @param[in] graph Graph handle
 *
 * @return spinst handle on success, or NULL otherwise.
 */
vsi_nn_spinst_t * vsi_nn_create_spinst
    (
    vsi_nn_graph_t       * graph
    );

/**
 * Create a new spinst
 *
 * @param[in] context Context handle
 *
 * @return spinst handle on success, or NULL otherwise.
 */
vsi_nn_spinst_t * vsi_nn_create_spinst_by_context
    (
    vx_context      context
    );

/**
 * Release spinst
 * Relase current spinst and set the handle to NULL.
 *
 * @param[in] spinst to release
 */
void vsi_nn_release_spinst
    (
    vsi_nn_spinst_t ** spinst
    );

/**
 * Set spinst's vx attribute
 * The value should be type of vsi_nn_spinst_attr_t.
 *
 * @param[in] spinst handle
 * @param[in] attrs New attributes to update
 * @see vsi_nn_spinst_attr_t
 *
 * @return VSI_SUCCESS on success, or error core otherwise.
 */
vsi_status vsi_nn_set_spinst_attr
    (
    vsi_nn_spinst_t * spinst,
    const vsi_nn_spinst_attr_t attrs
    );

/**
 * Set a instruction to spinst
 * add a instrutions to SPINST
 *
 * @param[in] spinst SPINST handle
 * @param[in] inst_units vsi_nn_spinst_inst_unit_param handle
 * @param[in] unit_count the unit count of one instruction
 *
 * @return VSI_SUCCESS on success, or error core otherwise.
 */
vsi_status vsi_nn_add_units_to_spinst
    (
    vsi_nn_spinst_t * spinst,
    vsi_nn_spinst_unit_param *inst_units,
    uint8_t unit_count
    );

/**
 * Set insts_count instruction to spinst
 * add instrutions to SPINST
 *
 * @param[in] spinst SPInst handle
 * @param[in] insts vsi_nn_spinst_inst_param handle
 * @param[in] insts_count the instruction num
 *
 * @return VSI_SUCCESS on success, or error core otherwise.
 */
vsi_status vsi_nn_add_spinst_insts
    (
    vsi_nn_spinst_t * spinst,
    vsi_nn_spinst_inst_param *insts,
    int32_t insts_count
    );

void vsi_nn_init_spinst_attr
    (
    vsi_nn_spinst_attr_t * attrs
    );

__END_DECLS

#endif

#endif
