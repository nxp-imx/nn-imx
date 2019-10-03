/****************************************************************************
*
*    Copyright (c) 2019 Vivante Corporation
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
#include <string.h>
#include <stdlib.h>

#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"
#include "vsi_nn_log.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_prv.h"
#include "utils/vsi_nn_math.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "client/vsi_nn_vxkernel.h"
#include "vsi_nn_internal_node.h"

#define SAFE_FREE( _PTR ) if( _PTR ){ \
    free( _PTR ); _PTR = NULL; }

static vsi_bool setup_op_shapes
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_gru_ovxlib_param* curr_param = &self->nn_param.gru_ovxlib;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* output_tensor = NULL;
    uint32_t num_units =  0;
    uint32_t output_size = 0;
    uint32_t batch_size = 0;

    if( curr_param->time_major )
    {
        batch_size = inputs[GRU_INPUT_INPUT]->attr.size[1];
    }
    else
    {
        batch_size = inputs[GRU_INPUT_INPUT]->attr.size[2];
    }

    num_units = inputs[GRU_INPUT_WEIGHT_I2R]->attr.size[1];
    output_size = num_units;

    /* create h_state input/output if app doesn't provide them */
    if( !inputs[GRU_INPUT_H_STATE] )
    {
        attr.dim_num = 2;
        attr.size[1] = batch_size;
        attr.size[0] = output_size;
        memcpy( &attr.dtype, &outputs[GRU_OUTPUT_OUTPUT]->attr.dtype, sizeof( attr.dtype ) );
        attr.vtl = FALSE;
        attr.is_const = TRUE;

        output_tensor = vsi_nn_new_internal_tensor( self, &attr, 0.0f );
        inputs[GRU_INPUT_H_STATE] = output_tensor->t;
    }

    if( !outputs[GRU_OUTPUT_H_STATE] )
    {
        memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
        attr.dim_num = VSI_NN_DIM_AUTO;
        memcpy( &attr.dtype, &outputs[GRU_OUTPUT_OUTPUT]->attr.dtype, sizeof( attr.dtype ) );
        attr.vtl = TRUE;
        output_tensor = vsi_nn_new_internal_tensor( self, &attr, 0.0f );
        outputs[GRU_OUTPUT_H_STATE] = output_tensor->t;
    }

    /* output */
    if( VSI_NN_DIM_AUTO == outputs[GRU_OUTPUT_OUTPUT]->attr.dim_num )
    {
        outputs[GRU_OUTPUT_OUTPUT]->attr.size[0] = output_size;
        outputs[GRU_OUTPUT_OUTPUT]->attr.size[1] = inputs[GRU_INPUT_INPUT]->attr.size[1];
        outputs[GRU_OUTPUT_OUTPUT]->attr.size[2] = inputs[GRU_INPUT_INPUT]->attr.size[2];
        outputs[GRU_OUTPUT_OUTPUT]->attr.dim_num = 3;
    }

    /* output_state_out */
    if( VSI_NN_DIM_AUTO == outputs[GRU_OUTPUT_H_STATE]->attr.dim_num )
    {
        outputs[GRU_OUTPUT_H_STATE]->attr.size[0] = output_size;
        outputs[GRU_OUTPUT_H_STATE]->attr.size[1] = batch_size;
        outputs[GRU_OUTPUT_H_STATE]->attr.dim_num = 2;
    }

    return TRUE;
}

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    return vsi_nn_compute_internal_node( self );
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    /*TODO: Check tensor shapes. */
    return TRUE;
} /* op_check() */

static vsi_status op_optimize
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_opt_direction_e direction
    )
{
    return vsi_nn_optimize_internal_node( self, direction );
} /* op_optimize() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_gru_ovxlib_param* curr_param = &self->nn_param.gru_ovxlib;
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* output_tensor = NULL;
    vsi_nn_tensor_t** split_output_tensors = NULL;
    vsi_nn_tensor_t** grucell_reshape_output_tensors =NULL;
    vsi_nn_tensor_t* last_step_h_state = NULL;
    vsi_nn_tensor_t* tensor = NULL;
    vsi_nn_tensor_t* input_tensor = NULL;
    vsi_bool use_virtual_tensor = FALSE;
    uint32_t batch_size = 0;
    uint32_t time_step = 0;
    uint32_t i = 0;
    uint32_t ofst = 0;
    uint32_t* slices = NULL;

    vsi_nn_init_internal_node_wksp( self );

    if( curr_param->time_major )
    {
        batch_size = inputs[GRU_INPUT_INPUT]->attr.size[1];
        time_step = inputs[GRU_INPUT_INPUT]->attr.size[2];
    }
    else
    {
        batch_size = inputs[GRU_INPUT_INPUT]->attr.size[2];
        time_step = inputs[GRU_INPUT_INPUT]->attr.size[1];
    }

    setup_op_shapes( self, inputs, outputs);

    /* default to input */
    input_tensor = inputs[GRU_INPUT_INPUT];
    if( !curr_param->time_major )
    {
        uint32_t* permute_in_perm = NULL;

        /* transpose to time_major */
        memcpy( &attr.dtype, &inputs[GRU_INPUT_INPUT]->attr.dtype, sizeof( attr.dtype ) );
        attr.dim_num = VSI_NN_DIM_AUTO;
        attr.vtl = use_virtual_tensor;
        attr.is_const = FALSE;
        output_tensor = vsi_nn_new_internal_tensor( self, &attr, 0.0f );

        curr = vsi_nn_new_internal_node( self, VSI_NN_OP_PERMUTE, 0, 0 );
        permute_in_perm = (uint32_t *)vsi_nn_new_internal_node_param(curr, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
        permute_in_perm[0] = 0;
        permute_in_perm[1] = 2;
        permute_in_perm[2] = 1;

        curr->node->nn_param.permute.perm = permute_in_perm;
        curr->node->nn_param.permute.dim_num = 3;
        curr->inputs[0] = inputs[GRU_INPUT_INPUT];
        curr->outputs[0] = output_tensor->t;
        vsi_nn_setup_internal_node_op(self, curr);

        input_tensor = output_tensor->t;
    }

    /* split input tensor */
    split_output_tensors = (vsi_nn_tensor_t **)malloc(time_step * sizeof(vsi_nn_tensor_t **));
    memset( split_output_tensors, 0x00, time_step * sizeof(vsi_nn_tensor_t **));
    grucell_reshape_output_tensors = (vsi_nn_tensor_t **)malloc(time_step * sizeof(vsi_nn_tensor_t **));
    memset( grucell_reshape_output_tensors, 0x00, time_step * sizeof(vsi_nn_tensor_t **));

    curr = vsi_nn_new_internal_node( self, VSI_NN_OP_SPLIT, 1, time_step );
    slices = (uint32_t *)vsi_nn_new_internal_node_param(curr, time_step * sizeof(uint32_t));
    curr->node->nn_param.split.axis = 2; /* timestep axis */
    curr->node->nn_param.split.slices_num = time_step;
    curr->inputs[0] = input_tensor;

    curr->node->nn_param.split.slices = slices;
    for( i = 0; i < time_step; i++ )
    {
        slices[i] = 1;

        memcpy( &attr.dtype, &input_tensor->attr.dtype, sizeof( attr.dtype ) );
        memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
        attr.dim_num = VSI_NN_DIM_AUTO;
        attr.vtl = use_virtual_tensor;
        attr.is_const = FALSE;
        output_tensor = vsi_nn_new_internal_tensor( self, &attr, 0.0f );
        curr->outputs[i] = output_tensor->t;
        split_output_tensors[i] = output_tensor->t;
    }
    vsi_nn_setup_internal_node_op( self, curr );

    ofst = 0;
    for( i = 0; i < time_step; i++ )
    {
        uint32_t tensor_size = vsi_nn_GetTensorSize( split_output_tensors[i]->attr.size,
            split_output_tensors[i]->attr.dim_num, split_output_tensors[i]->attr.dtype.vx_type );

        if( ofst & 0x3f )
        {
            memcpy( &attr.dtype, &split_output_tensors[i]->attr.dtype, sizeof( attr.dtype ) );
            memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
            attr.dim_num = VSI_NN_DIM_AUTO;
            attr.vtl = use_virtual_tensor;
            attr.is_const = FALSE;
            output_tensor = vsi_nn_new_internal_tensor( self, &attr, 0.0f );

            curr = vsi_nn_new_internal_node( self, VSI_NN_OP_DATACONVERT, 0, 0 );
            curr->inputs[0] = split_output_tensors[i];
            curr->outputs[0] = output_tensor->t;
            vsi_nn_setup_internal_node_op( self, curr );

            split_output_tensors[i] = output_tensor->t;
        }

        ofst += tensor_size;
    }

    last_step_h_state = inputs[GRU_INPUT_H_STATE];
    for( i = 0; i < time_step; i++ )
    {
        vsi_nn_tensor_t* reshape_output = NULL;
        vsi_nn_tensor_t* grucell_out0 = NULL;
        vsi_nn_tensor_t* grucell_out1 = NULL;
        uint32_t *reshape_split_size = NULL;
        uint32_t* reshape_grucell_output_size = NULL;

        /* reshape for split output */
        memcpy( &attr.dtype, &split_output_tensors[i]->attr.dtype, sizeof( attr.dtype ) );
        memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
        attr.dim_num = VSI_NN_DIM_AUTO;
        attr.vtl = use_virtual_tensor;
        attr.is_const = FALSE;
        output_tensor = vsi_nn_new_internal_tensor( self, &attr, 0.0f );
        reshape_output = output_tensor->t;

        curr = vsi_nn_new_internal_node( self, VSI_NN_OP_RESHAPE, 0, 0 );
        reshape_split_size = (uint32_t *)vsi_nn_new_internal_node_param(curr, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
        reshape_split_size[0] = -1;
        reshape_split_size[1] = batch_size;

        curr->node->nn_param.reshape.size = reshape_split_size;
        curr->node->nn_param.reshape.dim_num = 2;
        curr->inputs[0] = split_output_tensors[i];
        curr->outputs[0] = reshape_output;
        vsi_nn_setup_internal_node_op( self, curr );

        /* grucell output */
        memcpy( &attr.dtype, &outputs[GRU_OUTPUT_OUTPUT]->attr.dtype, sizeof( attr.dtype ) );
        memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
        attr.dim_num = VSI_NN_DIM_AUTO;
        attr.vtl = use_virtual_tensor;
        attr.is_const = FALSE;
        output_tensor = vsi_nn_new_internal_tensor( self, &attr, 0.0f );
        grucell_out0 = output_tensor->t;

        if( i != time_step - 1 )
        {
            /* grucell output h_state */
            memcpy( &attr.dtype, &outputs[GRU_OUTPUT_H_STATE]->attr.dtype, sizeof( attr.dtype ) );
            memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
            attr.dim_num = VSI_NN_DIM_AUTO;
            attr.vtl = use_virtual_tensor;
            attr.is_const = FALSE;
            output_tensor = vsi_nn_new_internal_tensor( self, &attr, 0.0f );
            grucell_out1 = output_tensor->t;
        }
        else
        {
            grucell_out1 = outputs[GRU_OUTPUT_H_STATE];
        }

        curr = vsi_nn_new_internal_node( self, VSI_NN_OP_GRUCELL_OVXLIB, 0, 0 );
        curr->node->nn_param.grucell_ovxlib.activation = curr_param->activation;
        memcpy( curr->node->nn_param.grucell_ovxlib.internal_dtype,
            curr_param->internal_dtype, sizeof( curr_param->internal_dtype ) );
        curr->inputs[GRUCELL_INPUT_INPUT] = reshape_output;
        curr->inputs[GRUCELL_INPUT_H_STATE] = last_step_h_state;

        curr->inputs[GRUCELL_INPUT_WEIGHT_I2R] = inputs[GRU_INPUT_WEIGHT_I2R];
        curr->inputs[GRUCELL_INPUT_WEIGHT_I2Z] = inputs[GRU_INPUT_WEIGHT_I2Z];
        curr->inputs[GRUCELL_INPUT_WEIGHT_H2R] = inputs[GRU_INPUT_WEIGHT_H2R];
        curr->inputs[GRUCELL_INPUT_WEIGHT_H2Z] = inputs[GRU_INPUT_WEIGHT_H2Z];

        curr->inputs[GRUCELL_INPUT_BIAS_R] = inputs[GRU_INPUT_BIAS_R];
        curr->inputs[GRUCELL_INPUT_BIAS_Z] = inputs[GRU_INPUT_BIAS_Z];

        curr->inputs[GRUCELL_INPUT_WEIGHT_I2C] = inputs[GRU_INPUT_WEIGHT_I2C];
        curr->inputs[GRUCELL_INPUT_WEIGHT_R2C] = inputs[GRU_INPUT_WEIGHT_R2C];

        curr->inputs[GRUCELL_INPUT_BIAS_C] = inputs[GRU_INPUT_BIAS_C];

        curr->outputs[GRUCELL_OUTPUT_OUTPUT] = grucell_out0;
        curr->outputs[GRUCELL_OUTPUT_H_STATE] = grucell_out1;

        vsi_nn_setup_internal_node_op( self, curr );

        last_step_h_state = grucell_out1;

        /* reshape output to 3-dims */
        memcpy( &attr.dtype, &grucell_out0->attr.dtype, sizeof( attr.dtype ) );
        memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
        attr.dim_num = VSI_NN_DIM_AUTO;
        attr.vtl = use_virtual_tensor;
        attr.is_const = FALSE;
        output_tensor = vsi_nn_new_internal_tensor( self, &attr, 0.0f );
        grucell_reshape_output_tensors[i] = output_tensor->t;

        curr = vsi_nn_new_internal_node( self, VSI_NN_OP_RESHAPE, 0, 0 );
        reshape_grucell_output_size = (uint32_t *)vsi_nn_new_internal_node_param(curr,
            VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
        reshape_grucell_output_size[0] = -1;
        reshape_grucell_output_size[1] = batch_size;
        reshape_grucell_output_size[2] = 1;

        curr->node->nn_param.reshape.size = reshape_grucell_output_size;
        curr->node->nn_param.reshape.dim_num = 3;
        curr->inputs[0] = grucell_out0;
        curr->outputs[0] = grucell_reshape_output_tensors[i];
        vsi_nn_setup_internal_node_op( self, curr );
    }

    tensor = outputs[GRU_OUTPUT_OUTPUT];
    if( !curr_param->time_major )
    {
        memcpy( &attr.dtype, &outputs[GRU_OUTPUT_OUTPUT]->attr.dtype, sizeof( attr.dtype ) );
        memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
        attr.dim_num = VSI_NN_DIM_AUTO;
        attr.vtl = use_virtual_tensor;
        attr.is_const = FALSE;
        output_tensor = vsi_nn_new_internal_tensor( self, &attr, 0.0f );

        tensor = output_tensor->t;
    }

    /* concat grucell output, the gru's output is 3-dims */
    curr = vsi_nn_new_internal_node( self, VSI_NN_OP_CONCAT, time_step, 1 );
    curr->node->nn_param.concat.axis = 2;
    for( i = 0; i < time_step; i++ )
    {
        curr->inputs[i] = grucell_reshape_output_tensors[i];
    }
    curr->outputs[0] = tensor;
    vsi_nn_setup_internal_node_op( self, curr );

    if( !curr_param->time_major )
    {
        uint32_t* permute_in_perm = NULL;

        /* transpose time_major to batch_major*/
        curr = vsi_nn_new_internal_node( self, VSI_NN_OP_PERMUTE, 0, 0 );
        permute_in_perm = (uint32_t *)vsi_nn_new_internal_node_param(curr, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
        permute_in_perm[0] = 0;
        permute_in_perm[1] = 2;
        permute_in_perm[2] = 1;

        curr->node->nn_param.permute.perm = permute_in_perm;
        curr->node->nn_param.permute.dim_num = 3;
        curr->inputs[0] = tensor;
        curr->outputs[0] = outputs[GRU_OUTPUT_OUTPUT];
        vsi_nn_setup_internal_node_op( self, curr );
    }

    SAFE_FREE( split_output_tensors );
    SAFE_FREE( grucell_reshape_output_tensors );

    return TRUE;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    vsi_nn_deinit_internal_node_wksp( self );

    return status;
} /* op_deinit() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ GRU_OVXLIB,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ GRU_INPUT_CNT,
    /* output_num */ GRU_OUTPUT_CNT
    );
#ifdef __cplusplus
}
#endif
