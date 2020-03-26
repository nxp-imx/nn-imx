/****************************************************************************
*
*    Copyright (c) 2020 Vivante Corporation
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
#include "ops/vsi_nn_op_grucell_ovxlib.h"
#include "vsi_nn_internal_node.h"
#include "vsi_nn_rnn.h"
#include "utils/vsi_nn_tensor_op.h"

static vsi_nn_internal_tensor_t* create_multiply
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input1,
    vsi_nn_tensor_t * input2,
    const vsi_nn_dtype_t* output_dtype,
    vsi_bool use_virtual_tensor
    )
{
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* tensor1 = NULL;
    vsi_nn_internal_node_t* tmp_inode = NULL;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    vsi_nn_internal_node_init_attr(&attr, output_dtype, use_virtual_tensor);
    tensor1 = vsi_nn_new_internal_tensor( self, &attr, 0.0f );

    tmp_inode = vsi_nn_new_internal_node(self, VSI_NN_OP_MULTIPLY, 0, 0 );

    tmp_inode->inputs[0] = input1;
    tmp_inode->inputs[1] = input2;
    tmp_inode->node->nn_param.multiply.scale = 1.0f;
    tmp_inode->node->vx_param.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    tmp_inode->node->vx_param.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    tmp_inode->outputs[0] = tensor1->t;
    vsi_nn_setup_internal_node_op(self, tmp_inode);

    return tensor1;
}

static vsi_bool setup_op_shapes
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* output_tensor = NULL;
    uint32_t output_size = 0;
    uint32_t batch_size = 0;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    batch_size = inputs[GRUCELL_INPUT_INPUT]->attr.size[1];
    output_size = inputs[GRUCELL_INPUT_WEIGHT_I2R]->attr.size[1];

    /* create h_state input/output if app doesn't provide them */
    if( !inputs[GRUCELL_INPUT_H_STATE] )
    {
        attr.dim_num = 2;
        attr.size[1] = batch_size;
        attr.size[0] = output_size;
        memcpy( &attr.dtype, &outputs[GRUCELL_OUTPUT_OUTPUT]->attr.dtype, sizeof( attr.dtype ) );
        attr.vtl = FALSE;
        attr.is_const = TRUE;

        output_tensor = vsi_nn_new_internal_tensor( self, &attr, 0.0f );
        inputs[GRUCELL_INPUT_H_STATE] = output_tensor->t;
    }

    if( !outputs[GRUCELL_OUTPUT_H_STATE] )
    {
        vsi_nn_internal_node_init_attr(&attr,
            &outputs[GRUCELL_OUTPUT_OUTPUT]->attr.dtype, TRUE);
        output_tensor = vsi_nn_new_internal_tensor( self, &attr, 0.0f );
        outputs[GRUCELL_OUTPUT_H_STATE] = output_tensor->t;
    }

    /* setup grucell output tensors' shape */
    /* output */
    if(VSI_NN_DIM_AUTO == outputs[GRUCELL_OUTPUT_OUTPUT]->attr.dim_num)
    {
        /* num_units */
        outputs[GRUCELL_OUTPUT_OUTPUT]->attr.size[0] = inputs[GRUCELL_INPUT_WEIGHT_I2C]->attr.size[1];
        /* batch_size */
        outputs[GRUCELL_OUTPUT_OUTPUT]->attr.size[1] = inputs[GRUCELL_INPUT_INPUT]->attr.size[1];
        outputs[GRUCELL_OUTPUT_OUTPUT]->attr.dim_num = inputs[GRUCELL_INPUT_INPUT]->attr.dim_num;
    }

    /* output_state_out */
    if(VSI_NN_DIM_AUTO == outputs[GRUCELL_OUTPUT_H_STATE]->attr.dim_num)
    {
        outputs[GRUCELL_OUTPUT_H_STATE]->attr.dim_num = outputs[GRUCELL_OUTPUT_OUTPUT]->attr.dim_num;
        memcpy( outputs[GRUCELL_OUTPUT_H_STATE]->attr.size, outputs[GRUCELL_OUTPUT_OUTPUT]->attr.size,
            VSI_NN_MAX_DIM_NUM * sizeof( uint32_t ) );
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
    vsi_nn_grucell_ovxlib_param* p = &self->nn_param.grucell_ovxlib;
    vsi_nn_tensor_attr_t attr;
    vsi_bool is_input_fc_on_tp = FALSE;
    vsi_bool is_hstate_fc_on_tp = FALSE;
    vsi_bool is_input_cand_fc_op_tp = FALSE;
    vsi_bool is_hstate_cand_fc_op_tp = FALSE;
    vsi_nn_internal_tensor_t* input_tensor = NULL;
    vsi_nn_internal_tensor_t* output_tensor = NULL;
    vsi_nn_internal_tensor_t* tmp_tensor = NULL;
    vsi_nn_internal_tensor_t* hstate_input_tensor = NULL;
    vsi_nn_internal_tensor_t* input_gate_fc_outputs[GRUCELL_RZ_GATE_COUNT] = { NULL };
    vsi_nn_internal_tensor_t* hstate_gate_fc_outputs[GRUCELL_RZ_GATE_COUNT] = { NULL };
    vsi_nn_tensor_t* gate_bias_tensors[GRUCELL_RZ_GATE_COUNT] = { NULL };
    vsi_nn_internal_tensor_t* gate_fc_outputs[GRUCELL_RZ_GATE_COUNT] = { NULL };
    vsi_nn_internal_tensor_t* gate_act_outputs[GRUCELL_RZ_GATE_COUNT] = { NULL };
    vsi_nn_internal_tensor_t* rh_mul_outputs = NULL;
    vsi_nn_internal_tensor_t* input_cand_fc_output = NULL;
    vsi_nn_internal_tensor_t* rh_cand_fc_output = NULL;
    vsi_nn_internal_tensor_t* cand_fc_output = NULL;
    vsi_nn_internal_tensor_t* cand_act_output = NULL;
    vsi_nn_internal_node_t* curr = NULL;
    vsi_bool use_virtual_tensor = FALSE;
    uint32_t kernel_h = 1;
    uint32_t kernel_w = 1;
    int32_t i = 0;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    vsi_nn_init_internal_node_wksp( self );

    memset( &p->local, 0x00, sizeof( p->local ) );
    memset( &attr, 0x00, sizeof( attr ) );
    p->local.multi_batch = (inputs[GRUCELL_INPUT_INPUT]->attr.size[1] > 1);
    if (p->local.gate_activation == VSI_NN_ACT_NONE)
    {
        p->local.gate_activation = VSI_NN_ACT_SIGMOID;
    }
    if (p->activation != VSI_NN_ACT_NONE)
    {
        p->local.candidate_activation = p->activation;
    }

    if( inputs[GRUCELL_INPUT_INPUT]->attr.dtype.qnt_type
        != inputs[GRUCELL_INPUT_WEIGHT_I2R]->attr.dtype.qnt_type)
    {
        /* input and input weights have different qtype, only TP can do this operation */
        is_input_fc_on_tp = TRUE;
    }
    else if( inputs[GRUCELL_INPUT_INPUT]->attr.size[0] % 64 != 0 )
    {
        /* NN performs bad if input's shape is not aligned to 64-byte */
        is_input_fc_on_tp = TRUE;
    }

    if( inputs[GRUCELL_INPUT_H_STATE]->attr.dtype.qnt_type
        != inputs[GRUCELL_INPUT_WEIGHT_H2R]->attr.dtype.qnt_type)
    {
        /* recurrent and recurrent weights have different qtype, only TP can do this operation */
        is_hstate_fc_on_tp = TRUE;
    }
    else if( inputs[GRUCELL_INPUT_H_STATE]->attr.size[0] % 64 != 0 )
    {
        /* NN performs bad if inputs' shape is not aligned to 64-byte */
        is_hstate_fc_on_tp = TRUE;
    }

    /* if both input fc and recurrent fc could be executed on NN, offloads one to TP*/
    if( !is_input_fc_on_tp && !is_hstate_fc_on_tp )
    {
        is_input_fc_on_tp = TRUE;
    }
    /* TODO: now, all fc on tp because can't fetch the HW feature */
    is_input_fc_on_tp = TRUE;
    is_hstate_fc_on_tp = TRUE;

    setup_op_shapes(self, inputs, outputs);

    for( i = 0; i < GRUCELL_RZ_GATE_COUNT; i++)
    {
        gate_bias_tensors[i] = inputs[GRUCELL_INPUT_BIAS_R + i];
    }

    /* Input FC */
    if( is_input_fc_on_tp )
    {
        /* tp */
        for( i = 0; i < GRUCELL_RZ_GATE_COUNT; i++)
        {
            input_gate_fc_outputs[i] = vsi_nn_rnn_create_tp_fc(self,
                                                inputs[GRUCELL_INPUT_INPUT],
                                                inputs[GRUCELL_INPUT_WEIGHT_I2R + i],
                                                gate_bias_tensors[i],
                                                &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_I2R + i],
                                                use_virtual_tensor);
        }
    }
    else
    {
        /* reshape and transpose input */
        vsi_nn_rnn_find_best_kernel_size(p->local.multi_batch,
            inputs[GRUCELL_INPUT_INPUT]->attr.size[0], &kernel_h, &kernel_w);
        input_tensor = vsi_nn_rnn_process_input_for_nn_fc(self, inputs[GRUCELL_INPUT_INPUT],
                                                kernel_h, kernel_w, use_virtual_tensor);

        for( i = 0; i < GRUCELL_RZ_GATE_COUNT; i++)
        {
            vsi_nn_internal_tensor_t* tmp = vsi_nn_rnn_create_nn_fc(self,
                                                input_tensor->t,
                                                inputs[GRUCELL_INPUT_WEIGHT_I2R + i],
                                                gate_bias_tensors[i],
                                                kernel_h, kernel_w,
                                                &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_I2R + i],
                                                use_virtual_tensor);
            /* transpose and reshape output */
            input_gate_fc_outputs[i] = vsi_nn_rnn_process_output_for_nn_fc(self,
                tmp->t, kernel_h, kernel_w, use_virtual_tensor);
        }
    }

    /* Hstate FC */
    if( is_hstate_fc_on_tp )
    {
        for( i = 0; i < GRUCELL_RZ_GATE_COUNT; i++)
        {
            hstate_gate_fc_outputs[i] = vsi_nn_rnn_create_tp_fc(self,
                                                inputs[GRUCELL_INPUT_H_STATE],
                                                inputs[GRUCELL_INPUT_WEIGHT_H2R + i],
                                                NULL,
                                                &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_H2R + i],
                                                use_virtual_tensor);
        }
    }
    else
    {
        /* reshape and transpose input */
        vsi_nn_rnn_find_best_kernel_size(p->local.multi_batch,
            inputs[GRUCELL_INPUT_H_STATE]->attr.size[0], &kernel_h, &kernel_w);
        hstate_input_tensor = vsi_nn_rnn_process_input_for_nn_fc(self,
            inputs[GRUCELL_INPUT_H_STATE], kernel_h, kernel_w, use_virtual_tensor);

        for( i = 0; i < GRUCELL_RZ_GATE_COUNT; i++)
        {
            vsi_nn_internal_tensor_t* tmp = vsi_nn_rnn_create_nn_fc(self,
                                                hstate_input_tensor->t,
                                                inputs[GRUCELL_INPUT_WEIGHT_H2R + i],
                                                NULL,
                                                kernel_h, kernel_w,
                                                &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_H2R + i],
                                                use_virtual_tensor);
            /* transpose and reshape output */
            hstate_gate_fc_outputs[i] = vsi_nn_rnn_process_output_for_nn_fc(self,
                tmp->t, kernel_h, kernel_w, use_virtual_tensor);
        }
    }

    /* Gate Input FC add Hstate FC */
    for ( i = 0;  i < GRUCELL_RZ_GATE_COUNT;  i++)
    {
        gate_fc_outputs[i] = vsi_nn_rnn_create_tensor_add(self,
                                 input_gate_fc_outputs[i]->t,
                                 hstate_gate_fc_outputs[i]->t,
                                 &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_I2R + i],
                                 use_virtual_tensor);
    }

    /* Gate activations */
    for ( i = 0; i < GRUCELL_RZ_GATE_COUNT; i++)
    {
        gate_act_outputs[i] = vsi_nn_rnn_create_activation(self,
                                  gate_fc_outputs[i]->t,
                                  p->local.gate_activation,
                                  &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_I2R + i],
                                  use_virtual_tensor);
    }

    /* Candidate FC */
    rh_mul_outputs = create_multiply(self,
                         gate_act_outputs[GRUCELL_GATE_R]->t,
                         inputs[GRUCELL_INPUT_H_STATE],
                         &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_H2R],
                         use_virtual_tensor);

    if( inputs[GRUCELL_INPUT_INPUT]->attr.dtype.qnt_type
        != inputs[GRUCELL_INPUT_WEIGHT_I2C]->attr.dtype.qnt_type)
    {
        /* input and input weights have different qtype, only TP can do this operation */
        is_input_cand_fc_op_tp = TRUE;
    }
    else if( inputs[GRUCELL_INPUT_INPUT]->attr.size[0] % 64 != 0 )
    {
        /* NN performs bad if input's shape is not aligned to 64-byte */
        is_input_cand_fc_op_tp = TRUE;
    }

    if( rh_mul_outputs->t->attr.dtype.qnt_type
        != inputs[GRUCELL_INPUT_WEIGHT_R2C]->attr.dtype.qnt_type)
    {
        /* recurrent and recurrent weights have different qtype, only TP can do this operation */
        is_hstate_cand_fc_op_tp = TRUE;
    }
    else if( rh_mul_outputs->t->attr.size[0] % 64 != 0 )
    {
        /* NN performs bad if inputs' shape is not aligned to 64-byte */
        is_hstate_cand_fc_op_tp = TRUE;
    }
    /* if both input fc and recurrent fc could be executed on NN, offloads one to TP*/
    if( !is_input_cand_fc_op_tp && !is_hstate_cand_fc_op_tp )
    {
        is_input_cand_fc_op_tp = TRUE;
    }
    /* TODO: now, all fc on tp because can't fetch the HW feature */
    is_input_cand_fc_op_tp = TRUE;
    is_hstate_cand_fc_op_tp = TRUE;

    if ( is_input_cand_fc_op_tp )
    {
        input_cand_fc_output = vsi_nn_rnn_create_tp_fc(self,
                                   inputs[GRUCELL_INPUT_INPUT],
                                   inputs[GRUCELL_INPUT_WEIGHT_I2C],
                                   inputs[GRUCELL_INPUT_BIAS_C],
                                   &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_I2C],
                                   use_virtual_tensor);
    }
    else
    {
        vsi_nn_internal_tensor_t* tmp = NULL;
        /* reshape and transpose input */
        vsi_nn_rnn_find_best_kernel_size(p->local.multi_batch,
            inputs[GRUCELL_INPUT_INPUT]->attr.size[0], &kernel_h, &kernel_w);
        input_tensor = vsi_nn_rnn_process_input_for_nn_fc(self, inputs[GRUCELL_INPUT_INPUT],
                                                kernel_h, kernel_w, use_virtual_tensor);
        tmp = vsi_nn_rnn_create_nn_fc(self,
                  input_tensor->t,
                  inputs[GRUCELL_INPUT_WEIGHT_I2C],
                  inputs[GRUCELL_INPUT_BIAS_C],
                  kernel_h, kernel_w,
                  &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_I2C],
                  use_virtual_tensor);
        /* transpose and reshape output */
        input_cand_fc_output = vsi_nn_rnn_process_output_for_nn_fc(self,
            tmp->t, kernel_h, kernel_w, use_virtual_tensor);
    }
    if ( is_hstate_cand_fc_op_tp )
    {
        /* if the tp support in:fp16,weight:u8,bias:fp32 batch>1, remove this. */
        if ((rh_mul_outputs->t->attr.dtype.vx_type) != (inputs[GRUCELL_INPUT_WEIGHT_R2C]->attr.dtype.vx_type)
            && (p->local.multi_batch))
        {
            vsi_nn_tensor_t* wei_r2c_tensor = NULL;

            memcpy(&attr, &(inputs[GRUCELL_INPUT_WEIGHT_R2C]->attr), sizeof(attr));
            attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
            attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;

            wei_r2c_tensor = vsi_nn_ConvertTensorDtype(self->graph, inputs[GRUCELL_INPUT_WEIGHT_R2C], &(attr.dtype));
            rh_cand_fc_output = vsi_nn_rnn_create_tp_fc(self,
                                    rh_mul_outputs->t,
                                    wei_r2c_tensor,
                                    NULL,
                                    &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_R2C],
                                    use_virtual_tensor);
        }
        else
        {
            rh_cand_fc_output = vsi_nn_rnn_create_tp_fc(self,
                                    rh_mul_outputs->t,
                                    inputs[GRUCELL_INPUT_WEIGHT_R2C],
                                    NULL,
                                    &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_R2C],
                                    use_virtual_tensor);
        }
    }
    else
    {
        vsi_nn_internal_tensor_t* tmp = NULL;
        /* reshape and transpose input */
        vsi_nn_rnn_find_best_kernel_size(p->local.multi_batch,
            rh_mul_outputs->t->attr.size[0], &kernel_h, &kernel_w);
        hstate_input_tensor = vsi_nn_rnn_process_input_for_nn_fc(self, rh_mul_outputs->t,
                                                kernel_h, kernel_w, use_virtual_tensor);
        tmp = vsi_nn_rnn_create_nn_fc(self,
                  hstate_input_tensor->t,
                  inputs[GRUCELL_INPUT_WEIGHT_R2C],
                  NULL,
                  kernel_h, kernel_w,
                  &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_R2C],
                  use_virtual_tensor);
        /* transpose and reshape output */
        rh_cand_fc_output = vsi_nn_rnn_process_output_for_nn_fc(self,
            tmp->t, kernel_h, kernel_w, use_virtual_tensor);
    }

    /* Candidate input FC add r*h FC */
    cand_fc_output = vsi_nn_rnn_create_tensor_add(self,
                         input_cand_fc_output->t,
                         rh_cand_fc_output->t,
                         &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_I2C],
                         use_virtual_tensor);

    /* Candidate activation */
    cand_act_output = vsi_nn_rnn_create_activation(self,
                                  cand_fc_output->t,
                                  p->local.candidate_activation,
                                  &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_I2C],
                                  use_virtual_tensor);

    /* GRU cell output */
    memcpy( &attr.dtype, &gate_act_outputs[GRUCELL_GATE_Z]->t->attr.dtype, sizeof( attr.dtype ) );
    memcpy( &attr.size, &gate_act_outputs[GRUCELL_GATE_Z]->t->attr.size, sizeof( attr.size ) );
    attr.dim_num = gate_act_outputs[GRUCELL_GATE_Z]->t->attr.dim_num;
    attr.vtl = use_virtual_tensor;
    attr.is_const = TRUE;
    input_tensor = vsi_nn_new_internal_tensor(self, &attr, 1.0f);

    memset( &attr, 0x00, sizeof(attr) );
    //memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
    attr.dim_num = VSI_NN_DIM_AUTO;
    attr.vtl = use_virtual_tensor;
    attr.is_const = FALSE;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    tmp_tensor = vsi_nn_new_internal_tensor(self, &attr, 0.0f);

    /* create internal tensor sub node (1-zt)*c */
    curr = vsi_nn_new_internal_node( self, VSI_NN_OP_SUBTRACT, 0, 0 );
    curr->inputs[0] = input_tensor->t;
    curr->inputs[1] = gate_act_outputs[GRUCELL_GATE_Z]->t;
    curr->outputs[0] = tmp_tensor->t;

    vsi_nn_setup_internal_node_op(self, curr);

    /* create internal multiply node (1-zt)*c */
    output_tensor = create_multiply(self,
                        tmp_tensor->t,
                        cand_act_output->t,
                        &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_I2C],
                        use_virtual_tensor);

    /* create internal multiply node zt*hstate */
    tmp_tensor = create_multiply(self,
                     gate_act_outputs[GRUCELL_GATE_Z]->t,
                     inputs[GRUCELL_INPUT_H_STATE],
                     &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_I2Z],
                     use_virtual_tensor);

     /* create internal tensor add node (1-zt)*c + zt*hstate */
    curr = vsi_nn_new_internal_node( self, VSI_NN_OP_ADD, 0, 0 );
    curr->inputs[0] = output_tensor->t;
    curr->inputs[1] = tmp_tensor->t;
    curr->outputs[0] = outputs[GRUCELL_OUTPUT_OUTPUT];

    vsi_nn_setup_internal_node_op(self, curr);

    /* copy output to h_state  */
    curr = vsi_nn_new_internal_node( self, VSI_NN_OP_DATACONVERT, 0, 0 );
    curr->inputs[0] = outputs[GRUCELL_OUTPUT_OUTPUT];
    curr->outputs[0] = outputs[GRUCELL_OUTPUT_H_STATE];
    vsi_nn_setup_internal_node_op(self, curr);

    return TRUE;
} /* op_setup() */

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    self->nn_param.grucell_ovxlib.local.candidate_activation = VSI_NN_ACT_TANH;
    self->nn_param.grucell_ovxlib.local.gate_activation = VSI_NN_ACT_SIGMOID;

    return status;
} /* op_init() */

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
    /* op_name    */ GRUCELL_OVXLIB,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ GRUCELL_INPUT_CNT,
    /* output_num */ GRUCELL_OUTPUT_CNT
    );
#ifdef __cplusplus
}
#endif
