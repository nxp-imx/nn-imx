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
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_math.h"
#include "utils/vsi_nn_tensor_op.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "client/vsi_nn_vxkernel.h"

#define CONCAT_FC_OUTPUTS TRUE

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_lstmunit_ovxlib_param* curr_param = &self->nn_param.lstmunit_ovxlib;
    vsi_nn_internal_node_t* curr = NULL;
    vsi_status status = VSI_FAILURE;
    int32_t i = 0;
    uint32_t j = 0;

    for( i = 0; i < LSTMUNIT_OVXLIB_INT_NODE_CNT; i++ )
    {
        curr = curr_param->local.nodes[i];
        if( curr )
        {
            for ( j = 0; j < curr->node->output.num; j++ )
            {
                if( NULL == curr->outputs[j] || NULL != curr->outputs[j]->t )
                    continue;
                vsi_nn_TensorReinit( self->graph, curr->outputs[j] );
            }

            status = vsi_nn_OpCompute( curr->node->op, curr->node, curr->inputs, curr->outputs );
            if( VSI_SUCCESS != status )
            {
                VSILOGE("op_compute fail @%d:%d", i, curr->node->op);
                break;
            }
        }
    }

    return status;
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
    vsi_nn_lstmunit_ovxlib_param* curr_param = &self->nn_param.lstmunit_ovxlib;
    vsi_nn_internal_node_t* curr = NULL;
    vsi_status status = VSI_SUCCESS;
    int32_t i = 0;

    for( i = 0; i < LSTMUNIT_OVXLIB_INT_NODE_CNT; i++ )
    {
        curr = curr_param->local.nodes[i];
        if( curr )
        {
            status = vsi_nn_OpOptimize( curr->node->op, curr->node,
                curr->inputs, curr->outputs, direction );
            if( VSI_SUCCESS != status )
            {
                VSILOGE("op_optimize fail @%d:%d", i, curr->node->op);
                break;
            }
        }
    }

    return status;
} /* op_optimize() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_internal_node_t* next = NULL;
    vsi_nn_tensor_t* zero_bias_tensor = NULL;
    vsi_nn_tensor_t* output_tensor = NULL;
    vsi_nn_tensor_t* tensors[4] = { 0 };
    vsi_nn_lstmunit_ovxlib_param* curr_param = &self->nn_param.lstmunit_ovxlib;
    int32_t weight_tensor_start_index = 0;
    int32_t bias_tensor_start_index = 0;
    int32_t ifco_start_index = 0;
    int32_t i = 0;
    int32_t j = 0;
    int32_t k = 0;
    int32_t shape[] = { 4, -1 };
    uint32_t concat_axis = 0;
    vsi_bool success = FALSE;
    vsi_bool use_virtual_tensor = FALSE;
    uint32_t *slices = NULL;

    memset( &curr_param->local, 0x00, sizeof( curr_param->local ) );
    curr_param->local.use_cifg = NULL == inputs[3];
    curr_param->local.use_layer_norm = NULL != inputs[21];
    curr_param->local.use_projection = NULL != inputs[18];
    curr_param->local.use_projection_bias = FALSE;//NULL != inputs[19];

    memset( &attr, 0x00, sizeof( attr ) );
    ifco_start_index = curr_param->local.use_cifg ? 1 : 0;

    weight_tensor_start_index = 3;
    bias_tensor_start_index = 14;
    /* input fc */
    for( i = ifco_start_index; i < 4; i++ )
    {
        int32_t weight_tensor_index = weight_tensor_start_index + i;
        int32_t bias_tensor_index = bias_tensor_start_index + i;

        memcpy( &attr.size, &inputs[bias_tensor_index]->attr.size, sizeof( attr.size ) );
        attr.dim_num = inputs[bias_tensor_index]->attr.dim_num;
        attr.vtl = FALSE;
        attr.is_const = TRUE;
        attr.dtype.scale = inputs[weight_tensor_index]->attr.dtype.scale;
        attr.dtype.zero_point = 0;
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
        attr.dtype.vx_type = VSI_NN_TYPE_INT32;
        zero_bias_tensor = vsi_nn_CreateTensorWithDefault( self->graph, &attr, 0.0f );
        curr_param->local.tensors[LSTMUNIT_TENSOR_ZERO_BIAS_I2I + i] = zero_bias_tensor;

        memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
        attr.dim_num = VSI_NN_DIM_AUTO;
        attr.vtl = use_virtual_tensor;
        attr.is_const = FALSE;
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
        output_tensor = vsi_nn_CreateTensor( self->graph, &attr );
        curr_param->local.tensors[LSTMUNIT_TENSOR_OUTPUT_I2I + i] = output_tensor;

        /* create internal nodes */
        curr = vsi_nn_internal_create_node( self->graph, VSI_NN_OP_FCL, 0, 0 );
        curr->node->nn_param.fcl.axis = 0;
        curr->node->nn_param.fcl.weights = inputs[weight_tensor_index]->attr.size[1];
        curr->node->vx_param.overflow_policy = VX_CONVERT_POLICY_WRAP;
        curr->node->vx_param.rounding_policy = VX_ROUND_POLICY_TO_ZERO;
        curr->node->vx_param.down_scale_size_rounding = VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_FLOOR;
        curr->inputs[0] = inputs[0]; /* input */
        curr->inputs[1] = inputs[weight_tensor_index]; /* weight */
        curr->inputs[2] = zero_bias_tensor;/* bias */
        curr->outputs[0] = output_tensor;
        vsi_nn_OpSetup( curr->node->op, curr->node, curr->inputs, curr->outputs );

        curr_param->local.nodes[LSTMUNIT_OVXLIB_FC_I2I + i] = curr;
        curr->node->uid = LSTMUNIT_OVXLIB_FC_I2I + i;
    }

    weight_tensor_start_index = 7;
    bias_tensor_start_index = 14;
    /* recurrent fc */
    for( i = ifco_start_index; i < 4; i++ )
    {
        int32_t weight_tensor_index = weight_tensor_start_index + i;
        int32_t bias_tensor_index = bias_tensor_start_index + i;

        memcpy( &attr.size, &inputs[bias_tensor_index]->attr.size, sizeof( attr.size ) );
        attr.dim_num = inputs[bias_tensor_index]->attr.dim_num;
        attr.vtl = FALSE;
        attr.is_const = TRUE;
        attr.dtype.scale = inputs[weight_tensor_index]->attr.dtype.scale;
        attr.dtype.zero_point = 0;
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
        attr.dtype.vx_type = VSI_NN_TYPE_INT32;
        zero_bias_tensor = vsi_nn_CreateTensorWithDefault( self->graph, &attr, 0.0f );
        curr_param->local.tensors[LSTMUNIT_TENSOR_ZERO_BIAS_R2I + i] = zero_bias_tensor;

        memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
        attr.dim_num = VSI_NN_DIM_AUTO;
        attr.vtl = use_virtual_tensor;
        attr.is_const = FALSE;
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
        output_tensor = vsi_nn_CreateTensor( self->graph, &attr );
        curr_param->local.tensors[LSTMUNIT_TENSOR_OUTPUT_R2I + i] = output_tensor;

        /* create internal nodes */
        curr = vsi_nn_internal_create_node( self->graph, VSI_NN_OP_FCL, 0, 0 );
        curr->node->nn_param.fcl.axis = 0;
        curr->node->nn_param.fcl.weights = inputs[weight_tensor_index]->attr.size[1];
        curr->node->vx_param.overflow_policy = VX_CONVERT_POLICY_WRAP;
        curr->node->vx_param.rounding_policy = VX_ROUND_POLICY_TO_ZERO;
        curr->node->vx_param.down_scale_size_rounding = VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_FLOOR;
        curr->inputs[0] = inputs[1]; /* input */
        curr->inputs[1] = inputs[weight_tensor_index]; /* weight */
        curr->inputs[2] = zero_bias_tensor;/* bias */
        curr->outputs[0] = output_tensor;
        vsi_nn_OpSetup( curr->node->op, curr->node, curr->inputs, curr->outputs );

        curr_param->local.nodes[LSTMUNIT_OVXLIB_FC_R2I + i] = curr;
        curr->node->uid = LSTMUNIT_OVXLIB_FC_R2I + i;
    }

    bias_tensor_start_index = 14;
    for( i = ifco_start_index; i < 4; i++ )
    {
        int32_t bias_tensor_index = bias_tensor_start_index + i;

        memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
        attr.dim_num = VSI_NN_DIM_AUTO;
        attr.vtl = use_virtual_tensor;
        attr.is_const = FALSE;
        memcpy( &attr.dtype, &inputs[bias_tensor_index]->attr.dtype, sizeof(attr.dtype) );
        tensors[i] = vsi_nn_CreateTensor( self->graph, &attr );

        success = vsi_nn_ReshapeTensor( self->graph, inputs[bias_tensor_index], tensors[i], (uint32_t *)shape, 2 );
        if( !success )
        {
            VSILOGE("Reshape tensor fail!\n");
        }
    }

    curr_param->local.tensors[LSTMUNIT_TENSOR_CONCATED_BIAS] = \
        vsi_nn_Concat(self->graph, &tensors[ifco_start_index], 4 - ifco_start_index, concat_axis );

    for( i = ifco_start_index; i < 4; i++ )
    {
        vsi_nn_ReleaseTensor( &tensors[i] );
    }

    /* create internal nodes */
    next = vsi_nn_internal_create_node( self->graph, VSI_NN_OP_LSTMUNIT_ACTIVATION, 0, 0 );
    next->node->nn_param.lstmunit_activation.cell_clip = self->nn_param.lstmunit_ovxlib.cell_clip;
    next->node->nn_param.lstmunit_activation.proj_clip = self->nn_param.lstmunit_ovxlib.proj_clip;
    next->node->nn_param.lstmunit_activation.forget_bias = 0.0f;
    next->node->nn_param.lstmunit_activation.is_cifg = curr_param->local.use_cifg;
    next->node->nn_param.lstmunit_activation.is_projection = curr_param->local.use_projection;
    next->node->nn_param.lstmunit_activation.is_layer_norm = curr_param->local.use_layer_norm;
    next->node->nn_param.lstmunit_activation.is_peephole = FALSE;
    next->node->nn_param.lstmunit_activation.is_hybrid = TRUE;
    next->node->vx_param.overflow_policy = VX_CONVERT_POLICY_WRAP;
    next->node->vx_param.rounding_policy = VX_ROUND_POLICY_TO_ZERO;
    next->node->vx_param.down_scale_size_rounding = VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_FLOOR;
    next->inputs[ACT_CSTATE_IN] = inputs[2];
    next->inputs[ACT_DATA_B] = curr_param->local.tensors[LSTMUNIT_TENSOR_CONCATED_BIAS];

    if( curr_param->local.use_layer_norm )
    {
        bias_tensor_start_index = 20;
        for( i = ifco_start_index; i < 4; i++ )
        {
            int32_t bias_tensor_index = bias_tensor_start_index + i;

            memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
            attr.dim_num = VSI_NN_DIM_AUTO;
            attr.vtl = use_virtual_tensor;
            attr.is_const = FALSE;
            memcpy( &attr.dtype, &inputs[bias_tensor_index]->attr.dtype, sizeof(attr.dtype) );
            tensors[i] = vsi_nn_CreateTensor( self->graph, &attr );

            success = vsi_nn_ReshapeTensor( self->graph, inputs[bias_tensor_index], tensors[i], (uint32_t *)shape, 2 );
            if( !success )
            {
                VSILOGE("Reshape tensor fail!\n");
            }
        }

        curr_param->local.tensors[LSTMUNIT_TENSOR_CONCATED_LN_W] = \
            vsi_nn_Concat(self->graph, &tensors[ifco_start_index], 4 - ifco_start_index, concat_axis );

        for( i = ifco_start_index; i < 4; i++ )
        {
            vsi_nn_ReleaseTensor( &tensors[i] );
        }

        next->inputs[ACT_LN_W] = curr_param->local.tensors[LSTMUNIT_TENSOR_CONCATED_LN_W];

        #if( CONCAT_FC_OUTPUTS )
        for( k = 0; k < 2; k++ )
        {
            int inputs_tensor_id[] = { LSTMUNIT_TENSOR_OUTPUT_I2I, LSTMUNIT_TENSOR_OUTPUT_R2I };

            memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
            attr.dim_num = VSI_NN_DIM_AUTO;
            attr.vtl = use_virtual_tensor;
            attr.is_const = FALSE;
            attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
            attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
            output_tensor = vsi_nn_CreateTensor( self->graph, &attr );
            curr_param->local.tensors[LSTMUNIT_TENSOR_INPUT_FC_OUTPUTS + k] = output_tensor;

            /* create internal nodes */
            curr = vsi_nn_internal_create_node( self->graph, VSI_NN_OP_CONCAT, 0, 0 );
            curr->node->nn_param.concat.axis = 1;
            for( j = 0, i = ifco_start_index; i < 4; i++, j++ )
            {
                curr->inputs[j] = curr_param->local.tensors[inputs_tensor_id[k] + i];
            }
            curr->outputs[0] = output_tensor;
            vsi_nn_OpSetup( curr->node->op, curr->node, curr->inputs, curr->outputs );

            curr_param->local.nodes[LSTMUNIT_OVXLIB_INPUT_FC_OUTPUTS_CONCAT + k] = curr;
            curr->node->uid = LSTMUNIT_OVXLIB_INPUT_FC_OUTPUTS_CONCAT + k;
        }

        memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
        attr.dim_num = VSI_NN_DIM_AUTO;
        attr.vtl = use_virtual_tensor;
        attr.is_const = FALSE;
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
        output_tensor = vsi_nn_CreateTensor( self->graph, &attr );
        curr_param->local.tensors[LSTMUNIT_TENSOR_LAYER_NORM_OUTPUT] = output_tensor;

        /* create internal nodes */
        curr = vsi_nn_internal_create_node( self->graph, VSI_NN_OP_TENSOR_ADD_MEAN_STDDEV_NORM, 0, 0 );
        curr->node->nn_param.tensor_add_mean_stddev_norm.eps = 1e-8;
        for( k = 0; k < 2; k++ )
        {
            curr->inputs[k] = curr_param->local.tensors[LSTMUNIT_TENSOR_INPUT_FC_OUTPUTS + k];
        }
        curr->outputs[0] = output_tensor;
        vsi_nn_OpSetup( curr->node->op, curr->node, curr->inputs, curr->outputs );

        curr_param->local.nodes[LSTMUNIT_OVXLIB_LAYER_NORM] = curr;
        curr->node->uid = LSTMUNIT_OVXLIB_LAYER_NORM;

        /* create internal nodes */
        curr = vsi_nn_internal_create_node( self->graph, VSI_NN_OP_SPLIT, 0, 0 );
        curr->node->nn_param.split.axis = 1;
        curr->node->nn_param.split.slices_num = 4 - ifco_start_index;
        slices = (uint32_t *)malloc(curr->node->nn_param.split.slices_num * sizeof( uint32_t ));
        curr->node->nn_param.split.slices = slices;
        for( j = 0, i = ifco_start_index; i < 4; i++, j++ )
        {
            slices[j] = 1;
        }
        curr->inputs[0] = curr_param->local.tensors[LSTMUNIT_TENSOR_LAYER_NORM_OUTPUT];

        for( j = 0, i = ifco_start_index; i < 4; i++, j++ )
        {
            memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
            attr.dim_num = VSI_NN_DIM_AUTO;
            attr.vtl = use_virtual_tensor;
            attr.is_const = FALSE;
            attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
            attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
            output_tensor = vsi_nn_CreateTensor( self->graph, &attr );
            curr_param->local.tensors[LSTMUNIT_TENSOR_LAYER_NORM_OUTPUT_I + i] = output_tensor;

            curr->outputs[j] = output_tensor;
        }
        vsi_nn_OpSetup( curr->node->op, curr->node, curr->inputs, curr->outputs );
        curr_param->local.nodes[LSTMUNIT_OVXLIB_LAYER_NORM_SPLIT] = curr;
        curr->node->uid = LSTMUNIT_OVXLIB_LAYER_NORM_SPLIT;
        if( slices )
        {
            free( slices );
            slices = NULL;
        }
        #else
        /* add/stddev mean */
        for( i = ifco_start_index; i < 4; i++ )
        {
            memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
            attr.dim_num = VSI_NN_DIM_AUTO;
            attr.vtl = use_virtual_tensor;
            attr.is_const = FALSE;
            attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
            attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
            output_tensor = vsi_nn_CreateTensor( self->graph, &attr );
            curr_param->local.tensors[LSTMUNIT_TENSOR_LAYER_NORM_OUTPUT_I + i] = output_tensor;

            /* create internal nodes */
            curr = vsi_nn_internal_create_node( self->graph, VSI_NN_OP_TENSOR_ADD_MEAN_STDDEV_NORM, 0, 0 );
            curr->node->nn_param.tensor_add_mean_stddev_norm.eps = 1e-8;
            curr->inputs[0] = curr_param->local.tensors[LSTMUNIT_TENSOR_OUTPUT_I2I + i];
            curr->inputs[1] = curr_param->local.tensors[LSTMUNIT_TENSOR_OUTPUT_R2I + i];
            curr->outputs[0] = output_tensor;
            vsi_nn_OpSetup( curr->node->op, curr->node, curr->inputs, curr->outputs );

            curr_param->local.nodes[LSTMUNIT_OVXLIB_LAYER_NORM_I + i] = curr;
            curr->node->uid = LSTMUNIT_OVXLIB_LAYER_NORM_I + i;
        }
        #endif

        next->inputs[ACT_INPUT_FC_I] = curr_param->local.tensors[LSTMUNIT_TENSOR_LAYER_NORM_OUTPUT_I];
        next->inputs[ACT_INPUT_FC_F] = curr_param->local.tensors[LSTMUNIT_TENSOR_LAYER_NORM_OUTPUT_F];
        next->inputs[ACT_INPUT_FC_C] = curr_param->local.tensors[LSTMUNIT_TENSOR_LAYER_NORM_OUTPUT_C];
        next->inputs[ACT_INPUT_FC_O] = curr_param->local.tensors[LSTMUNIT_TENSOR_LAYER_NORM_OUTPUT_O];
    }
    else
    {
        next->inputs[ACT_INPUT_FC_I] = curr_param->local.tensors[LSTMUNIT_TENSOR_OUTPUT_I2I];
        next->inputs[ACT_INPUT_FC_F] = curr_param->local.tensors[LSTMUNIT_TENSOR_OUTPUT_I2F];
        next->inputs[ACT_INPUT_FC_C] = curr_param->local.tensors[LSTMUNIT_TENSOR_OUTPUT_I2C];
        next->inputs[ACT_INPUT_FC_O] = curr_param->local.tensors[LSTMUNIT_TENSOR_OUTPUT_I2O];
        next->inputs[ACT_HSTATE_FC_I] = curr_param->local.tensors[LSTMUNIT_TENSOR_OUTPUT_R2I];
        next->inputs[ACT_HSTATE_FC_F] = curr_param->local.tensors[LSTMUNIT_TENSOR_OUTPUT_R2F];
        next->inputs[ACT_HSTATE_FC_C] = curr_param->local.tensors[LSTMUNIT_TENSOR_OUTPUT_R2C];
        next->inputs[ACT_HSTATE_FC_O] = curr_param->local.tensors[LSTMUNIT_TENSOR_OUTPUT_R2O];
    }

    if( curr_param->local.use_projection )
    {
        memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
        attr.dim_num = VSI_NN_DIM_AUTO;
        attr.vtl = use_virtual_tensor;
        attr.is_const = FALSE;
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
        output_tensor = vsi_nn_CreateTensor( self->graph, &attr );
        curr_param->local.tensors[LSTMUNIT_TENSOR_ACTIVATION_OUTPUT] = output_tensor;

        next->outputs[ACT_OUTPUT] = output_tensor;
        next->outputs[ACT_CSTATE_OUT] = outputs[2];
        next->outputs[ACT_HSTATE_OUT] = NULL;
    }
    else
    {
        next->outputs[ACT_OUTPUT] = outputs[0];
        next->outputs[ACT_CSTATE_OUT] = outputs[2];
        next->outputs[ACT_HSTATE_OUT] = outputs[1];
    }
    vsi_nn_OpSetup( next->node->op, next->node, next->inputs, next->outputs );
    curr_param->local.nodes[LSTMUNIT_OVXLIB_ACTIVATIONS] = next;
    next->node->uid = LSTMUNIT_OVXLIB_ACTIVATIONS;

    if( curr_param->local.use_projection )
    {
        const int32_t weight_tensor_index = 18;
        attr.size[0] = inputs[weight_tensor_index]->attr.size[0];
        attr.dim_num = 1;
        attr.vtl = FALSE;
        attr.is_const = TRUE;
        attr.dtype.scale = inputs[weight_tensor_index]->attr.dtype.scale;
        attr.dtype.zero_point = 0;
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
        attr.dtype.vx_type = VSI_NN_TYPE_INT32;
        zero_bias_tensor = vsi_nn_CreateTensorWithDefault( self->graph, &attr, 0.0f );
        curr_param->local.tensors[LSTMUNIT_TENSOR_ZERO_BIAS_PROJECTION] = zero_bias_tensor;

        /* create internal nodes */
        curr = vsi_nn_internal_create_node( self->graph, VSI_NN_OP_FCL, 0, 0 );
        curr->node->nn_param.fcl.axis = 0;
        curr->node->nn_param.fcl.weights = inputs[weight_tensor_index]->attr.size[1];
        curr->node->vx_param.overflow_policy = VX_CONVERT_POLICY_WRAP;
        curr->node->vx_param.rounding_policy = VX_ROUND_POLICY_TO_ZERO;
        curr->node->vx_param.down_scale_size_rounding = VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_FLOOR;

        curr->inputs[0] = curr_param->local.tensors[LSTMUNIT_TENSOR_ACTIVATION_OUTPUT];
        curr->inputs[1] = inputs[weight_tensor_index];
        curr->inputs[2] = curr_param->local.tensors[LSTMUNIT_TENSOR_ZERO_BIAS_PROJECTION];

        if( curr_param->local.use_projection_bias )
        {
            memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
            attr.dim_num = VSI_NN_DIM_AUTO;
            attr.vtl = use_virtual_tensor;
            attr.is_const = FALSE;
            memcpy( &attr.dtype, &outputs[0]->attr.dtype, sizeof(attr.dtype) );
            output_tensor = vsi_nn_CreateTensor( self->graph, &attr );
            curr_param->local.tensors[LSTMUNIT_TENSOR_PROJECTION_FC_OUTPUT] = output_tensor;

            curr->outputs[0] = output_tensor;
        }
        else
        {
            curr->outputs[0] = outputs[0];
        }

        vsi_nn_OpSetup( curr->node->op, curr->node, curr->inputs, curr->outputs );
        curr_param->local.nodes[LSTMUNIT_OVXLIB_FC_PROJ] = curr;
        curr->node->uid = LSTMUNIT_OVXLIB_FC_PROJ;

        if( curr_param->local.use_projection_bias )
        {
            curr = vsi_nn_internal_create_node( self->graph, VSI_NN_OP_ADD, 0, 0 );
            curr->node->nn_param.reshape.size = outputs[0]->attr.size;
            curr->node->nn_param.reshape.dim_num = outputs[0]->attr.dim_num;
            curr->inputs[0] = curr_param->local.tensors[LSTMUNIT_TENSOR_PROJECTION_FC_OUTPUT];
            curr->inputs[1] = inputs[19];
            curr->outputs[0] = outputs[0];
            vsi_nn_OpSetup( curr->node->op, curr->node, curr->inputs, curr->outputs );
            curr_param->local.nodes[LSTMUNIT_OVXLIB_TEST_NODE] = curr;
            curr->node->uid = LSTMUNIT_OVXLIB_TEST_NODE;
        }

        curr = vsi_nn_internal_create_node( self->graph, VSI_NN_OP_RESHAPE, 0, 0 );
        curr->node->nn_param.reshape.size = outputs[0]->attr.size;
        curr->node->nn_param.reshape.dim_num = outputs[0]->attr.dim_num;
        curr->inputs[0] = outputs[0];
        curr->outputs[0] = outputs[1];
        vsi_nn_OpSetup( curr->node->op, curr->node, curr->inputs, curr->outputs );
        curr_param->local.nodes[LSTMUNIT_OVXLIB_TEST_NODE] = curr;
        curr->node->uid = LSTMUNIT_OVXLIB_TEST_NODE;
    }

    #if( 0 )
    /* test node */
    curr = vsi_nn_internal_create_node( self->graph, VSI_NN_OP_RESHAPE, 0, 0 );
    curr->node->nn_param.reshape.size = curr_param->local.tensors[LSTMUNIT_TENSOR_CONCATED_LN_W]->attr.size;
    curr->node->nn_param.reshape.dim_num = curr_param->local.tensors[LSTMUNIT_TENSOR_CONCATED_LN_W]->attr.dim_num;
    curr->inputs[0] = curr_param->local.tensors[LSTMUNIT_TENSOR_CONCATED_LN_W];
    curr->outputs[0] = outputs[0];
    vsi_nn_OpSetup( curr->node->op, curr->node, curr->inputs, curr->outputs );
    curr_param->local.nodes[LSTMUNIT_OVXLIB_TEST_NODE] = curr;
    #endif

    return TRUE;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_nn_lstmunit_ovxlib_param* curr_param = &self->nn_param.lstmunit_ovxlib;
    vsi_nn_internal_node_t* curr_node = NULL;
    vsi_nn_tensor_t* curr_tensor = NULL;
    vsi_status status = VSI_SUCCESS;
    int32_t i = 0;

    for( i = 0; i < LSTMUNIT_OVXLIB_INT_NODE_CNT; i++ )
    {
        curr_node = curr_param->local.nodes[i];
        if( curr_node )
        {
            vsi_nn_internal_release_node( &curr_param->local.nodes[i] );
        }
    }

    for( i = 0; i < LSTMUNIT_TENSOR_CNT; i++ )
    {
        curr_tensor = curr_param->local.tensors[i];
        if( curr_tensor )
        {
            vsi_nn_ReleaseTensor( &curr_param->local.tensors[i] );
        }
    }

    return status;
} /* op_deinit() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ LSTMUNIT_OVXLIB,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ 24,
    /* output_num */ 4
    );
#ifdef __cplusplus
}
#endif
