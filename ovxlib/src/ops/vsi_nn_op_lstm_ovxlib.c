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

#define SAFE_FREE( _PTR ) if( _PTR ){                       \
    free( _PTR ); _PTR = NULL; }

static vsi_bool setup_op_shapes
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_lstm_ovxlib_param* curr_param = &self->nn_param.lstm_ovxlib;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* output_tensor = NULL;
    uint32_t num_units =  0;
    uint32_t output_size = 0;
    uint32_t batch_size = 0;

    if( curr_param->time_major )
    {
        batch_size = inputs[LSTM_INPUT_INPUT]->attr.size[1];
    }
    else
    {
        batch_size = inputs[LSTM_INPUT_INPUT]->attr.size[2];
    }

    num_units = inputs[LSTM_INPUT_WEIGHT_I2F]->attr.size[1];
    output_size = num_units;
    if( inputs[LSTM_INPUT_WEIGHT_PROJ] )
    {
        output_size = inputs[LSTM_INPUT_WEIGHT_PROJ]->attr.size[1];
    }

    /* create h_state and c_state input/output if app doesn't provide them */
    if( !inputs[LSTM_INPUT_H_STATE] )
    {
        attr.dim_num = 2;
        attr.size[1] = batch_size;
        attr.size[0] = output_size;
        memcpy( &attr.dtype, &outputs[LSTM_OUTPUT_OUTPUT]->attr.dtype, sizeof( attr.dtype ) );
        attr.vtl = FALSE;
        attr.is_const = TRUE;

        output_tensor = vsi_nn_new_internal_tensor( self, &attr, 0.0f );
        inputs[LSTM_INPUT_H_STATE] = output_tensor->t;
    }

    if( !inputs[LSTM_INPUT_C_STATE] )
    {
        attr.dim_num = 2;
        attr.size[1] = batch_size;
        attr.size[0] = num_units;
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
        attr.vtl = FALSE;
        attr.is_const = TRUE;

        output_tensor = vsi_nn_new_internal_tensor( self, &attr, 0.0f );
        inputs[LSTM_INPUT_C_STATE] = output_tensor->t;
    }

    if( !outputs[LSTM_OUTPUT_H_STATE] )
    {
        memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
        attr.dim_num = VSI_NN_DIM_AUTO;
        memcpy( &attr.dtype, &outputs[LSTM_OUTPUT_OUTPUT]->attr.dtype, sizeof( attr.dtype ) );
        attr.vtl = TRUE;
        output_tensor = vsi_nn_new_internal_tensor( self, &attr, 0.0f );
        outputs[LSTM_OUTPUT_H_STATE] = output_tensor->t;
    }

    if( !outputs[LSTM_OUTPUT_C_STATE] )
    {
        memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
        attr.dim_num = VSI_NN_DIM_AUTO;
        memcpy( &attr.dtype, &inputs[LSTM_INPUT_C_STATE]->attr.dtype, sizeof( attr.dtype ) );
        attr.vtl = TRUE;
        output_tensor = vsi_nn_new_internal_tensor( self, &attr, 0.0f );
        outputs[LSTM_OUTPUT_C_STATE] = output_tensor->t;
    }

    /* output */
    if( VSI_NN_DIM_AUTO == outputs[LSTM_OUTPUT_OUTPUT]->attr.dim_num )
    {
        outputs[LSTM_OUTPUT_OUTPUT]->attr.size[0] = output_size;
        if( curr_param->return_sequences )
        {
            outputs[LSTM_OUTPUT_OUTPUT]->attr.size[1] = inputs[LSTM_INPUT_INPUT]->attr.size[1];
            outputs[LSTM_OUTPUT_OUTPUT]->attr.size[2] = inputs[LSTM_INPUT_INPUT]->attr.size[2];
            outputs[LSTM_OUTPUT_OUTPUT]->attr.dim_num = 3;
        }
        else
        {
            outputs[LSTM_OUTPUT_OUTPUT]->attr.size[1] = batch_size;
            outputs[LSTM_OUTPUT_OUTPUT]->attr.dim_num = 2;
        }
    }

    /* output_state_out */
    if( VSI_NN_DIM_AUTO == outputs[LSTM_OUTPUT_H_STATE]->attr.dim_num )
    {
        outputs[LSTM_OUTPUT_H_STATE]->attr.size[0] = output_size;
        outputs[LSTM_OUTPUT_H_STATE]->attr.size[1] = batch_size;
        outputs[LSTM_OUTPUT_H_STATE]->attr.dim_num = 2;
    }

    /* cell_state_out */
    if(VSI_NN_DIM_AUTO == outputs[LSTM_OUTPUT_C_STATE]->attr.dim_num)
    {
        outputs[LSTM_OUTPUT_C_STATE]->attr.size[0] = num_units;
        outputs[LSTM_OUTPUT_C_STATE]->attr.size[1] = batch_size;
        outputs[LSTM_OUTPUT_C_STATE]->attr.dim_num = 2;
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
    vsi_nn_lstm_ovxlib_param* curr_param = &self->nn_param.lstm_ovxlib;
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* output_tensor = NULL;
    vsi_nn_tensor_t** split_output_tensors = NULL;
    vsi_nn_tensor_t** lstmunit_reshape_output_tensors =NULL;
    vsi_nn_tensor_t* last_step_h_state = NULL;
    vsi_nn_tensor_t* last_step_c_state = NULL;
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
        batch_size = inputs[LSTM_INPUT_INPUT]->attr.size[1];
        time_step = inputs[LSTM_INPUT_INPUT]->attr.size[2];
    }
    else
    {
        batch_size = inputs[LSTM_INPUT_INPUT]->attr.size[2];
        time_step = inputs[LSTM_INPUT_INPUT]->attr.size[1];
    }

    setup_op_shapes( self, inputs, outputs);

    /* default to input */
    input_tensor = inputs[LSTM_INPUT_INPUT];
    if( !curr_param->time_major )
    {
        uint32_t* permute_in_perm = NULL;

        /* transpose to time_major */
        memcpy( &attr.dtype, &inputs[LSTM_INPUT_INPUT]->attr.dtype, sizeof( attr.dtype ) );
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
        curr->inputs[0] = inputs[LSTM_INPUT_INPUT];
        curr->outputs[0] = output_tensor->t;
        vsi_nn_setup_internal_node_op(self, curr);

        input_tensor = output_tensor->t;
    }

    /* split input tensor */
    split_output_tensors = (vsi_nn_tensor_t **)malloc(time_step * \
        sizeof(vsi_nn_tensor_t **));
    memset( split_output_tensors, 0x00, time_step * sizeof(vsi_nn_tensor_t **));
    lstmunit_reshape_output_tensors = (vsi_nn_tensor_t **)malloc(time_step * \
        sizeof(vsi_nn_tensor_t **));
    memset( lstmunit_reshape_output_tensors, 0x00, time_step * sizeof(vsi_nn_tensor_t **));

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

    last_step_h_state = inputs[LSTM_INPUT_H_STATE];
    last_step_c_state = inputs[LSTM_INPUT_C_STATE];
    for( i = 0; i < time_step; i++ )
    {
        vsi_nn_tensor_t* reshape_output = NULL;
        vsi_nn_tensor_t* lstmunit_out0 = NULL;
        vsi_nn_tensor_t* lstmunit_out1 = NULL;
        vsi_nn_tensor_t* lstmunit_out2 = NULL;
        uint32_t *reshape_split_size = NULL;

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

        /* lstmunit output */
        if( (i == time_step - 1) && !curr_param->return_sequences )
        {
            lstmunit_out0 = outputs[LSTM_OUTPUT_OUTPUT];
        }
        else
        {
            memcpy( &attr.dtype, &outputs[LSTM_OUTPUT_OUTPUT]->attr.dtype, sizeof( attr.dtype ) );
            memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
            attr.dim_num = VSI_NN_DIM_AUTO;
            attr.vtl = use_virtual_tensor;
            attr.is_const = FALSE;
            output_tensor = vsi_nn_new_internal_tensor( self, &attr, 0.0f );
            lstmunit_out0 = output_tensor->t;
        }

        if( i != time_step - 1 )
        {
            /* lstmunit output h_state */
            memcpy( &attr.dtype, &outputs[LSTM_OUTPUT_H_STATE]->attr.dtype, sizeof( attr.dtype ) );
            memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
            attr.dim_num = VSI_NN_DIM_AUTO;
            attr.vtl = use_virtual_tensor;
            attr.is_const = FALSE;
            output_tensor = vsi_nn_new_internal_tensor( self, &attr, 0.0f );
            lstmunit_out1 = output_tensor->t;

            /* lstmunit output c_state */
            memcpy( &attr.dtype, &outputs[LSTM_OUTPUT_C_STATE]->attr.dtype, sizeof( attr.dtype ) );
            memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
            attr.dim_num = VSI_NN_DIM_AUTO;
            attr.vtl = use_virtual_tensor;
            attr.is_const = FALSE;
            output_tensor = vsi_nn_new_internal_tensor( self, &attr, 0.0f );
            lstmunit_out2 = output_tensor->t;
        }
        else
        {
            lstmunit_out1 = outputs[LSTM_OUTPUT_H_STATE];
            lstmunit_out2 = outputs[LSTM_OUTPUT_C_STATE];
        }

        curr = vsi_nn_new_internal_node( self, VSI_NN_OP_LSTMUNIT_OVXLIB, 0, 0 );
        curr->node->nn_param.lstmunit_ovxlib.activation = curr_param->activation;
        curr->node->nn_param.lstmunit_ovxlib.cell_clip = curr_param->cell_clip;
        curr->node->nn_param.lstmunit_ovxlib.forget_bias = curr_param->forget_bias;
        curr->node->nn_param.lstmunit_ovxlib.proj_clip = curr_param->proj_clip;
        curr->node->nn_param.lstmunit_ovxlib.recurrent_activation = curr_param->recurrent_activation;
        memcpy( curr->node->nn_param.lstmunit_ovxlib.internal_dtype,
            curr_param->internal_dtype, sizeof( curr_param->internal_dtype ) );
        curr->inputs[LSTMUNIT_INPUT_INPUT] = reshape_output;
        curr->inputs[LSTMUNIT_INPUT_H_STATE] = last_step_h_state;
        curr->inputs[LSTMUNIT_INPUT_C_STATE] = last_step_c_state;
        curr->inputs[LSTMUNIT_INPUT_WEIGHT_I2I] = inputs[LSTM_INPUT_WEIGHT_I2I];
        curr->inputs[LSTMUNIT_INPUT_WEIGHT_I2F] = inputs[LSTM_INPUT_WEIGHT_I2F];
        curr->inputs[LSTMUNIT_INPUT_WEIGHT_I2C] = inputs[LSTM_INPUT_WEIGHT_I2C];
        curr->inputs[LSTMUNIT_INPUT_WEIGHT_I2O] = inputs[LSTM_INPUT_WEIGHT_I2O];

        curr->inputs[LSTMUNIT_INPUT_WEIGHT_R2I] = inputs[LSTM_INPUT_WEIGHT_R2I];
        curr->inputs[LSTMUNIT_INPUT_WEIGHT_R2F] = inputs[LSTM_INPUT_WEIGHT_R2F];
        curr->inputs[LSTMUNIT_INPUT_WEIGHT_R2C] = inputs[LSTM_INPUT_WEIGHT_R2C];
        curr->inputs[LSTMUNIT_INPUT_WEIGHT_R2O] = inputs[LSTM_INPUT_WEIGHT_R2O];

        curr->inputs[LSTMUNIT_INPUT_WEIGHT_C2I] = inputs[LSTM_INPUT_WEIGHT_C2I];
        curr->inputs[LSTMUNIT_INPUT_WEIGHT_C2F] = inputs[LSTM_INPUT_WEIGHT_C2F];
        curr->inputs[LSTMUNIT_INPUT_WEIGHT_C2O] = inputs[LSTM_INPUT_WEIGHT_C2O];

        curr->inputs[LSTMUNIT_INPUT_BIAS_I] = inputs[LSTM_INPUT_BIAS_I];
        curr->inputs[LSTMUNIT_INPUT_BIAS_F] = inputs[LSTM_INPUT_BIAS_F];
        curr->inputs[LSTMUNIT_INPUT_BIAS_C] = inputs[LSTM_INPUT_BIAS_C];
        curr->inputs[LSTMUNIT_INPUT_BIAS_O] = inputs[LSTM_INPUT_BIAS_O];

        curr->inputs[LSTMUNIT_INPUT_WEIGHT_PROJ] = inputs[LSTM_INPUT_WEIGHT_PROJ];
        curr->inputs[LSTMUNIT_INPUT_BIAS_PROJ] = inputs[LSTM_INPUT_BIAS_PROJ];

        curr->inputs[LSTMUNIT_INPUT_LAYERNORM_I] = inputs[LSTM_INPUT_LAYERNORM_I];
        curr->inputs[LSTMUNIT_INPUT_LAYERNORM_F] = inputs[LSTM_INPUT_LAYERNORM_F];
        curr->inputs[LSTMUNIT_INPUT_LAYERNORM_C] = inputs[LSTM_INPUT_LAYERNORM_C];
        curr->inputs[LSTMUNIT_INPUT_LAYERNORM_O] = inputs[LSTM_INPUT_LAYERNORM_O];

        curr->outputs[LSTMUNIT_OUTPUT_OUTPUT] = lstmunit_out0;
        curr->outputs[LSTMUNIT_OUTPUT_H_STATE] = lstmunit_out1;
        curr->outputs[LSTMUNIT_OUTPUT_C_STATE] = lstmunit_out2;

        vsi_nn_setup_internal_node_op( self, curr );

        last_step_h_state = lstmunit_out1;
        last_step_c_state = lstmunit_out2;

        if( curr_param->return_sequences )
        {
            uint32_t* reshape_lstmunit_output_size = NULL;

            /* reshape output to 3-dims */
            memcpy( &attr.dtype, &lstmunit_out0->attr.dtype, sizeof( attr.dtype ) );
            memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
            attr.dim_num = VSI_NN_DIM_AUTO;
            attr.vtl = use_virtual_tensor;
            attr.is_const = FALSE;
            output_tensor = vsi_nn_new_internal_tensor( self, &attr, 0.0f );
            lstmunit_reshape_output_tensors[i] = output_tensor->t;

            curr = vsi_nn_new_internal_node( self, VSI_NN_OP_RESHAPE, 0, 0 );
            reshape_lstmunit_output_size = (uint32_t *)vsi_nn_new_internal_node_param(curr,
                VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
            reshape_lstmunit_output_size[0] = -1;
            reshape_lstmunit_output_size[1] = batch_size;
            reshape_lstmunit_output_size[2] = 1;

            curr->node->nn_param.reshape.size = reshape_lstmunit_output_size;
            curr->node->nn_param.reshape.dim_num = 3;
            curr->inputs[0] = lstmunit_out0;
            curr->outputs[0] = lstmunit_reshape_output_tensors[i];
            vsi_nn_setup_internal_node_op( self, curr );
        }
    }

    if( curr_param->return_sequences )
    {
        tensor = outputs[LSTM_OUTPUT_OUTPUT];
        if( !curr_param->time_major )
        {
            memcpy( &attr.dtype, &outputs[LSTM_OUTPUT_OUTPUT]->attr.dtype, sizeof( attr.dtype ) );
            memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
            attr.dim_num = VSI_NN_DIM_AUTO;
            attr.vtl = use_virtual_tensor;
            attr.is_const = FALSE;
            output_tensor = vsi_nn_new_internal_tensor( self, &attr, 0.0f );

            tensor = output_tensor->t;
        }

        /* concat */
        curr = vsi_nn_new_internal_node( self, VSI_NN_OP_CONCAT, time_step, 1 );
        curr->node->nn_param.concat.axis = 2;
        for( i = 0; i < time_step; i++ )
        {
            curr->inputs[i] = lstmunit_reshape_output_tensors[i];
        }
        curr->outputs[0] = tensor;
        vsi_nn_setup_internal_node_op( self, curr );

        if( !curr_param->time_major )
        {
            uint32_t* permute_in_perm = NULL;

            /* transpose to time_major */
            memcpy( &attr.dtype, &inputs[LSTM_INPUT_INPUT]->attr.dtype, sizeof( attr.dtype ) );
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
            curr->inputs[0] = tensor;
            curr->outputs[0] = outputs[LSTM_OUTPUT_OUTPUT];
            vsi_nn_setup_internal_node_op( self, curr );
        }
    }

    SAFE_FREE( split_output_tensors );
    SAFE_FREE( lstmunit_reshape_output_tensors );

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

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    self->nn_param.lstm_ovxlib.activation = VSI_NN_LSTMUNIT_ACT_TANH;
    self->nn_param.lstm_ovxlib.recurrent_activation = VSI_NN_LSTMUNIT_ACT_SIGMOID;
    self->nn_param.lstm_ovxlib.return_sequences = TRUE;
    self->nn_param.lstm_ovxlib.time_major = TRUE;

    return status;
} /* op_init() */

#ifdef __cpluplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ LSTM_OVXLIB,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ LSTM_INPUT_CNT,
    /* output_num */ LSTM_OUTPUT_CNT
    );
#ifdef __cpluplus
}
#endif
