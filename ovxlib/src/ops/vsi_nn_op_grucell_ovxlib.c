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
#include "ops/vsi_nn_op_grucell_ovxlib.h"
#include "vsi_nn_internal_node.h"

static vsi_bool find_best_kernel_size
    (
    vsi_nn_node_t * self,
    uint32_t input_size,
    uint32_t* p_kernel_h,
    uint32_t* p_kernel_w
    )
{
    vsi_nn_grucell_ovxlib_param* p = &self->nn_param.grucell_ovxlib;
    uint32_t kernel_h = 1;
    uint32_t kernel_w = 1;

    if( p->local.multi_batch)
    {
        /* batch FC only be converted to 1x1 or 1xN conv */
        /* try 1xN */
        kernel_h = 7;
        while( input_size % kernel_h != 0 )
        {
            kernel_h--;
        }
    }
    else
    {
        /* try NxN */
        if( !p->local.multi_batch )
        {
            #if( !defined( _WIN32 ) )
            /* try NxN conv */
            kernel_h = 8;
            while( input_size % (kernel_h * kernel_h) != 0 )
            {
                kernel_h--;
            }
            #endif
        }

        if( kernel_h > 1 )
        {
            kernel_w = kernel_h;
        }
        else
        {
            /* Only 1x1 found, try 1xN */
            kernel_h = 7;
            while( input_size % kernel_h != 0 )
            {
                kernel_h--;
            }
            kernel_w = 1;
        }

    }

    VSILOGD("Use kernel_h: %d, kernel_w: %d to convert FC", kernel_h, kernel_w);
    if( p_kernel_h )
    {
        *p_kernel_h = kernel_h;
    }

    if( p_kernel_w )
    {
        *p_kernel_w = kernel_w;
    }

    return TRUE;
}

static vsi_nn_internal_tensor_t* create_tp_fc
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vsi_nn_tensor_t * weight,
    vsi_nn_tensor_t * bias,
    const vsi_nn_dtype_t* output_dtype,
    vsi_bool use_virtual_tensor
    )
{
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_t* tensor = NULL;
    vsi_nn_internal_tensor_t* tensor1 = NULL;
    vsi_nn_internal_tensor_t* tensor2 = NULL;
    vsi_nn_internal_node_t* tmp_inode = NULL;

    memset( &attr, 0x00, sizeof(attr));

    tensor = bias;
    if( !bias )
    {
        /* create zero bias for NN/TP */
        tensor1 = vsi_nn_create_zero_bias_tensor(self, &input->attr, &weight->attr);
        tensor = tensor1->t;
    }

    memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
    attr.dim_num = VSI_NN_DIM_AUTO;
    attr.vtl = use_virtual_tensor;
    attr.is_const = FALSE;

    if( output_dtype->qnt_type == VSI_NN_QNT_TYPE_NONE )
    {
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    }
    else
    {
        memcpy( &attr.dtype, output_dtype, sizeof( attr.dtype ) );
    }
    tensor2 = vsi_nn_new_internal_tensor( self, &attr, 0.0f );

    tmp_inode = vsi_nn_new_internal_node(self, VSI_NN_OP_FCL, 0, 0 );
    tmp_inode->node->nn_param.fcl.axis = 0;
    tmp_inode->node->nn_param.fcl.weights = weight->attr.size[1];
    tmp_inode->node->vx_param.overflow_policy = VX_CONVERT_POLICY_WRAP;
    tmp_inode->node->vx_param.rounding_policy = VX_ROUND_POLICY_TO_ZERO;
    tmp_inode->node->vx_param.down_scale_size_rounding = VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_FLOOR;

    tmp_inode->inputs[0] = input;
    tmp_inode->inputs[1] = weight;
    tmp_inode->inputs[2] = tensor;
    tmp_inode->outputs[0] = tensor2->t;
    vsi_nn_setup_internal_node_op(self, tmp_inode);

    return tensor2;
}

static vsi_nn_internal_tensor_t* create_nn_fc
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vsi_nn_tensor_t * weight,
    vsi_nn_tensor_t * bias,
    uint32_t kernel_h,
    uint32_t kernel_w,
    const vsi_nn_dtype_t* output_dtype,
    vsi_bool use_virtual_tensor
    )
{
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_t* tensor = NULL;
    vsi_nn_internal_tensor_t* tensor1 = NULL;
    vsi_nn_internal_tensor_t* tensor2 = NULL;
    vsi_nn_internal_tensor_t* reshaped_weight_tensor = NULL;
    uint32_t reshaped_weight_shape[VSI_NN_MAX_DIM_NUM] = { 0 };
    vsi_nn_internal_node_t* tmp_inode = NULL;

    memset( &attr, 0x00, sizeof(attr));

    tensor = bias;
    if( !bias )
    {
        /* create zero bias for NN/TP */
        tensor1 = vsi_nn_create_zero_bias_tensor(self, &input->attr, &weight->attr);
        tensor = tensor1->t;
    }

    memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
    attr.dim_num = VSI_NN_DIM_AUTO;
    attr.vtl = use_virtual_tensor;
    attr.is_const = FALSE;

    if( output_dtype->qnt_type == VSI_NN_QNT_TYPE_NONE )
    {
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    }
    else
    {
        memcpy( &attr.dtype, output_dtype, sizeof( attr.dtype ) );
    }
    tensor2 = vsi_nn_new_internal_tensor( self, &attr, 0.0f );

    reshaped_weight_shape[3] = weight->attr.size[1];
    reshaped_weight_shape[2] = weight->attr.size[0] / ( kernel_h * kernel_w );
    reshaped_weight_shape[1] = kernel_h;
    reshaped_weight_shape[0] = kernel_w;

    memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
    attr.dim_num = VSI_NN_DIM_AUTO;
    attr.vtl = weight->attr.vtl;
    attr.is_const = FALSE; //weight->attr.is_const;
    memcpy( &attr.dtype, &weight->attr.dtype, sizeof(attr.dtype) );
    memcpy( &attr.size, &reshaped_weight_shape, sizeof(attr.size));
    reshaped_weight_tensor = vsi_nn_new_internal_tensor( self, &attr, 0.0f );

    vsi_nn_ReshapeTensor( self->graph, weight, reshaped_weight_tensor->t, reshaped_weight_shape, 4 );

    tmp_inode = vsi_nn_new_internal_node(self, VSI_NN_OP_CONV2D, 0, 0 );
    tmp_inode->node->nn_param.conv2d.ksize[0] = kernel_w;
    tmp_inode->node->nn_param.conv2d.ksize[1] = kernel_h;
    tmp_inode->node->nn_param.conv2d.stride[0] = 1;
    tmp_inode->node->nn_param.conv2d.stride[1] = 1;
    tmp_inode->node->nn_param.conv2d.pad[0] = 0;
    tmp_inode->node->nn_param.conv2d.pad[1] = 0;
    tmp_inode->node->nn_param.conv2d.pad[2] = 0;
    tmp_inode->node->nn_param.conv2d.pad[3] = 0;
    tmp_inode->node->nn_param.conv2d.group = 1;
    tmp_inode->node->nn_param.conv2d.dilation[0] = 1;
    tmp_inode->node->nn_param.conv2d.dilation[1] = 1;
    tmp_inode->node->nn_param.conv2d.weights = weight->attr.size[1];
    tmp_inode->node->vx_param.overflow_policy = VX_CONVERT_POLICY_WRAP;
    tmp_inode->node->vx_param.rounding_policy = VX_ROUND_POLICY_TO_ZERO;
    tmp_inode->node->vx_param.down_scale_size_rounding = VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_FLOOR;

    tmp_inode->inputs[0] = input;
    tmp_inode->inputs[1] = reshaped_weight_tensor->t;
    tmp_inode->inputs[2] = tensor;
    tmp_inode->outputs[0] = tensor2->t;
    vsi_nn_setup_internal_node_op(self, tmp_inode);

    return tensor2;
}

static vsi_nn_internal_tensor_t* process_input_for_nn_fc
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    uint32_t kernel_h,
    uint32_t kernel_w,
    int32_t use_virtual_tensor
    )
{
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* tensor1 = NULL;
    vsi_nn_internal_tensor_t* tensor2 = NULL;
    uint32_t* reshape_in_size = NULL;
    uint32_t* permute_in_perm = NULL;
    vsi_nn_internal_node_t* tmp_inode = NULL;

    memset( &attr, 0x00, sizeof(attr));

    memcpy( &attr.dtype, &input->attr.dtype, sizeof( attr.dtype ) );
    attr.dim_num = VSI_NN_DIM_AUTO;
    attr.vtl = use_virtual_tensor;
    attr.is_const = FALSE;
    tensor1 = vsi_nn_new_internal_tensor(self, &attr, 0.0f);

    tmp_inode = vsi_nn_new_internal_node(self, VSI_NN_OP_RESHAPE, 0, 0 );
    reshape_in_size = (uint32_t *)vsi_nn_new_internal_node_param(tmp_inode, 4 * sizeof(uint32_t));

    reshape_in_size[3] = input->attr.size[1];
    reshape_in_size[2] = input->attr.size[0] / (kernel_h * kernel_w);
    reshape_in_size[1] = kernel_h;
    reshape_in_size[0] = kernel_w;

    tmp_inode->node->nn_param.reshape.size = reshape_in_size;
    tmp_inode->node->nn_param.reshape.dim_num = 4;
    tmp_inode->inputs[0] = input;
    tmp_inode->outputs[0] = tensor1->t;
    vsi_nn_setup_internal_node_op(self, tmp_inode);

    if( kernel_h != kernel_w )
    {
        tensor2 = vsi_nn_new_internal_tensor(self, &attr, 0.0f);
        tmp_inode = vsi_nn_new_internal_node(self, VSI_NN_OP_PERMUTE, 0, 0 );
        permute_in_perm = (uint32_t *)vsi_nn_new_internal_node_param(tmp_inode, 4 * sizeof(uint32_t));

        permute_in_perm[0] = 3;
        permute_in_perm[1] = 1;
        permute_in_perm[2] = 2;
        permute_in_perm[3] = 0;

        tmp_inode->node->nn_param.permute.perm = permute_in_perm;
        tmp_inode->node->nn_param.permute.dim_num = 4;
        tmp_inode->inputs[0] = tensor1->t;
        tmp_inode->outputs[0] = tensor2->t;
        vsi_nn_setup_internal_node_op(self, tmp_inode);

        tensor1 = tensor2;
    }

    return tensor1;
}

static vsi_nn_internal_tensor_t* process_output_for_nn_fc
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    uint32_t kernel_h,
    uint32_t kernel_w,
    int32_t use_virtual_tensor
    )
{
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* tensor1 = NULL;
    vsi_nn_internal_tensor_t* tensor2 = NULL;
    uint32_t* reshape_in_size = NULL;
    uint32_t* permute_in_perm = NULL;
    vsi_nn_internal_node_t* tmp_inode = NULL;
    vsi_nn_tensor_t* tensor = input;

    memset( &attr, 0x00, sizeof(attr));

    memcpy( &attr.dtype, &input->attr.dtype, sizeof( attr.dtype ) );
    attr.dim_num = VSI_NN_DIM_AUTO;
    attr.vtl = use_virtual_tensor;
    attr.is_const = FALSE;

    if( kernel_h != kernel_w )
    {
        tensor1 = vsi_nn_new_internal_tensor(self, &attr, 0.0f);
        tmp_inode = vsi_nn_new_internal_node(self, VSI_NN_OP_PERMUTE, 0, 0 );
        permute_in_perm = (uint32_t *)vsi_nn_new_internal_node_param(tmp_inode, 4 * sizeof(uint32_t));

        permute_in_perm[0] = 3;
        permute_in_perm[1] = 1;
        permute_in_perm[2] = 2;
        permute_in_perm[3] = 0;

        tmp_inode->node->nn_param.permute.perm = permute_in_perm;
        tmp_inode->node->nn_param.permute.dim_num = 4;
        tmp_inode->inputs[0] = tensor;
        tmp_inode->outputs[0] = tensor1->t;
        vsi_nn_setup_internal_node_op(self, tmp_inode);

        tensor = tensor1->t;
    }

    tensor2 = vsi_nn_new_internal_tensor(self, &attr, 0.0f);
    tmp_inode = vsi_nn_new_internal_node(self, VSI_NN_OP_RESHAPE, 0, 0 );
    reshape_in_size = (uint32_t *)vsi_nn_new_internal_node_param(tmp_inode, 4 * sizeof(uint32_t));

    reshape_in_size[1] = tensor->attr.size[3];
    reshape_in_size[0] = tensor->attr.size[2];

    tmp_inode->node->nn_param.reshape.size = reshape_in_size;
    tmp_inode->node->nn_param.reshape.dim_num = 2;
    tmp_inode->inputs[0] = tensor;
    tmp_inode->outputs[0] = tensor2->t;
    vsi_nn_setup_internal_node_op(self, tmp_inode);

    return tensor2;
}

static vsi_nn_internal_tensor_t* create_tensor_add
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

    memset( &attr, 0x00, sizeof(attr));

    memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
    attr.dim_num = VSI_NN_DIM_AUTO;
    attr.vtl = use_virtual_tensor;
    attr.is_const = FALSE;

    if( output_dtype->qnt_type == VSI_NN_QNT_TYPE_NONE )
    {
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    }
    else
    {
        memcpy( &attr.dtype, output_dtype, sizeof( attr.dtype ) );
    }
    tensor1 = vsi_nn_new_internal_tensor( self, &attr, 0.0f );

    tmp_inode = vsi_nn_new_internal_node(self, VSI_NN_OP_ADD, 0, 0 );

    tmp_inode->inputs[0] = input1;
    tmp_inode->inputs[1] = input2;
    tmp_inode->outputs[0] = tensor1->t;
    vsi_nn_setup_internal_node_op(self, tmp_inode);

    return tensor1;
}

static vsi_nn_op_t get_act_op_type(vsi_nn_grucell_activation_e type)
{
    switch (type)
    {
    case VSI_NN_GRU_ACT_RELU:
        return VSI_NN_OP_RELU;
    case VSI_NN_GRU_ACT_RELU6:
        return VSI_NN_OP_RELU6;
    case VSI_NN_GRU_ACT_TANH:
        return VSI_NN_OP_TANH;
    case VSI_NN_GRU_ACT_SIGMOID:
        return VSI_NN_OP_SIGMOID;
    default:
        VSILOGE("error activation type %d", type);
        break;
    }

    return VSI_NN_OP_TANH;
}

static vsi_nn_internal_tensor_t* create_activation
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vsi_nn_grucell_activation_e act_type,
    const vsi_nn_dtype_t* output_dtype,
    vsi_bool use_virtual_tensor
    )
{
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* tensor1 = NULL;
    vsi_nn_internal_node_t* tmp_inode = NULL;

    memset( &attr, 0x00, sizeof(attr));

    memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
    attr.dim_num = VSI_NN_DIM_AUTO;
    attr.vtl = use_virtual_tensor;
    attr.is_const = FALSE;

    if( output_dtype->qnt_type == VSI_NN_QNT_TYPE_NONE )
    {
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    }
    else
    {
        memcpy( &attr.dtype, output_dtype, sizeof( attr.dtype ) );
    }
    tensor1 = vsi_nn_new_internal_tensor( self, &attr, 0.0f );

    tmp_inode = vsi_nn_new_internal_node(self, get_act_op_type(act_type), 0, 0 );

    tmp_inode->inputs[0] = input;
    tmp_inode->node->nn_param.tanh.scale_a = 1.0f;
    tmp_inode->node->nn_param.tanh.scale_b = 1.0f;
    tmp_inode->outputs[0] = tensor1->t;
    vsi_nn_setup_internal_node_op(self, tmp_inode);

    return tensor1;
}

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

    memset( &attr, 0x00, sizeof(attr));

    memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
    attr.dim_num = VSI_NN_DIM_AUTO;
    attr.vtl = use_virtual_tensor;
    attr.is_const = FALSE;

    if( output_dtype->qnt_type == VSI_NN_QNT_TYPE_NONE )
    {
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    }
    else
    {
        memcpy( &attr.dtype, output_dtype, sizeof( attr.dtype ) );
    }
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
        memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
        attr.dim_num = VSI_NN_DIM_AUTO;
        memcpy( &attr.dtype, &outputs[GRUCELL_OUTPUT_OUTPUT]->attr.dtype, sizeof( attr.dtype ) );
        attr.vtl = TRUE;
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

    vsi_nn_init_internal_node_wksp( self );

    memset( &p->local, 0x00, sizeof( p->local ) );
    memset( &attr, 0x00, sizeof( attr ) );
    p->local.multi_batch = (inputs[GRUCELL_INPUT_INPUT]->attr.size[1]);
    if (p->local.gate_activation == VSI_NN_GRU_ACT_NONE)
    {
        p->local.gate_activation = VSI_NN_GRU_ACT_SIGMOID;
    }
    if (p->activation != VSI_NN_GRU_ACT_NONE)
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
            input_gate_fc_outputs[i] = create_tp_fc(self,
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
        find_best_kernel_size(self, inputs[GRUCELL_INPUT_INPUT]->attr.size[0], &kernel_h, &kernel_w);
        input_tensor = process_input_for_nn_fc(self, inputs[GRUCELL_INPUT_INPUT],
                                                kernel_h, kernel_w, use_virtual_tensor);

        for( i = 0; i < GRUCELL_RZ_GATE_COUNT; i++)
        {
            vsi_nn_internal_tensor_t* tmp = create_nn_fc(self,
                                                input_tensor->t,
                                                inputs[GRUCELL_INPUT_WEIGHT_I2R + i],
                                                gate_bias_tensors[i],
                                                kernel_h, kernel_w,
                                                &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_I2R + i],
                                                use_virtual_tensor);
            /* transpose and reshape output */
            input_gate_fc_outputs[i] = process_output_for_nn_fc(self, tmp->t, kernel_h, kernel_w, use_virtual_tensor);
        }
    }

    /* Hstate FC */
    if( is_hstate_fc_on_tp )
    {
        for( i = 0; i < GRUCELL_RZ_GATE_COUNT; i++)
        {
            hstate_gate_fc_outputs[i] = create_tp_fc(self,
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
        find_best_kernel_size(self, inputs[GRUCELL_INPUT_H_STATE]->attr.size[0], &kernel_h, &kernel_w);
        hstate_input_tensor = process_input_for_nn_fc(self, inputs[GRUCELL_INPUT_H_STATE],
                                                kernel_h, kernel_w, use_virtual_tensor);

        for( i = 0; i < GRUCELL_RZ_GATE_COUNT; i++)
        {
            vsi_nn_internal_tensor_t* tmp = create_nn_fc(self,
                                                hstate_input_tensor->t,
                                                inputs[GRUCELL_INPUT_WEIGHT_H2R + i],
                                                NULL,
                                                kernel_h, kernel_w,
                                                &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_H2R + i],
                                                use_virtual_tensor);
            /* transpose and reshape output */
            hstate_gate_fc_outputs[i] = process_output_for_nn_fc(self, tmp->t, kernel_h, kernel_w, use_virtual_tensor);
        }
    }

    /* Gate Input FC add Hstate FC */
    for ( i = 0;  i < GRUCELL_RZ_GATE_COUNT;  i++)
    {
        gate_fc_outputs[i] = create_tensor_add(self,
                                 input_gate_fc_outputs[i]->t,
                                 hstate_gate_fc_outputs[i]->t,
                                 &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_I2R + i],
                                 use_virtual_tensor);
    }

    /* Gate activations */
    for ( i = 0; i < GRUCELL_RZ_GATE_COUNT; i++)
    {
        gate_act_outputs[i] = create_activation(self,
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

    input_cand_fc_output = create_tp_fc(self,
                               inputs[GRUCELL_INPUT_INPUT],
                               inputs[GRUCELL_INPUT_WEIGHT_I2C],
                               inputs[GRUCELL_INPUT_BIAS_C],
                               &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_I2C],
                               use_virtual_tensor);
    rh_cand_fc_output = create_tp_fc(self,
                            rh_mul_outputs->t,
                            inputs[GRUCELL_INPUT_WEIGHT_R2C],
                            NULL,
                            &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_R2C],
                            use_virtual_tensor);

    /* Candidate input FC add r*h FC */
    cand_fc_output = create_tensor_add(self,
                         input_cand_fc_output->t,
                         rh_cand_fc_output->t,
                         &p->internal_dtype[GRUCELL_QUANTIZE_PARAM_I2C],
                         use_virtual_tensor);

    /* Candidate activation */
    cand_act_output = create_activation(self,
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
    memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
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

    self->nn_param.grucell_ovxlib.local.candidate_activation = VSI_NN_GRU_ACT_TANH;
    self->nn_param.grucell_ovxlib.local.gate_activation = VSI_NN_GRU_ACT_SIGMOID;

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
