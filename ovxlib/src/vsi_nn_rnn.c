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
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "vsi_nn_graph.h"
#include "vsi_nn_log.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_util.h"
#include "vsi_nn_rnn_prv.h"
#include "vsi_nn_internal_node.h"

/**********************************************************
* MACROS
**********************************************************/
#define RNN_WKSP(_GRAPH) ( (vsi_nn_rnn_wksp_t *)((_GRAPH)->rnn_wksp) )

/**********************************************************
* LOCAL FUNCTIONS
**********************************************************/
static vsi_status internal_buffer_init
    (
    vsi_nn_rnn_internal_buffer_t* buffer,
    vsi_nn_tensor_t* tensor,
    float default_value
    )
{
    vsi_status  status      = VSI_FAILURE;
    uint32_t    element_num = 0;
    uint32_t    i           = 0;
    uint32_t    stride      = 0;
    uint32_t    data_size   = 0;
    uint8_t*    data        = NULL;

    if( TRUE == tensor->attr.vtl )
    {
        VSILOGE("Internal tensors cannot be dumpped.");
        return status;
    }

    if( NULL == buffer )
    {
        VSILOGE("Internal buffer is NULL.");
        return status;
    }

    memcpy(&buffer->attr, &tensor->attr, sizeof(tensor->attr));
    data_size = vsi_nn_GetTensorSize( buffer->attr.size, buffer->attr.dim_num, buffer->attr.dtype.vx_type );
    element_num = vsi_nn_GetElementNum(tensor);
    stride = vsi_nn_TypeGetBytes( tensor->attr.dtype.vx_type );

    data = (uint8_t *)malloc(data_size);
    if( NULL == buffer )
    {
        VSILOGE("Out of memoery.");
        goto error;
    }

    /* init data with zero */
    for( i = 0; i < element_num; i++ )
    {
        status = vsi_nn_Float32ToDtype(default_value, data + i * stride, &buffer->attr.dtype);
        if( VSI_SUCCESS != status )
        {
            VSILOGE("Convert default value to dtype fail");
            goto error;
        }
    }

    buffer->data = data;
    buffer->data_size = data_size;

error:
    if( VSI_SUCCESS != status )
    {
        vsi_nn_safe_free(data);
    }
    return status;
} /* internal_buffer_init() */

static vsi_status internal_buffer_deinit
    (
    vsi_nn_rnn_internal_buffer_t* buffer
    )
{
    vsi_status status = VSI_FAILURE;

    if( NULL == buffer )
    {
        VSILOGE("Internal buffer is NULL.");
        return status;
    }

    vsi_nn_safe_free( buffer->data );

    return VSI_SUCCESS;
} /* internal_buffer_deinit() */

static vsi_status internal_buffer_copy_to_tensor
    (
    vsi_nn_graph_t* graph,
    vsi_nn_rnn_internal_buffer_t* buffer,
    vsi_nn_tensor_id_t tensorid
    )
{
    vsi_status status = VSI_FAILURE;
    uint32_t request_data_size = 0;
    vsi_nn_tensor_t* tensor = NULL;

    if( NULL == buffer )
    {
        VSILOGE("Internal buffer is NULL.\n");
        return status;
    }

    tensor = vsi_nn_GetTensor( graph, tensorid );
    request_data_size = vsi_nn_GetTensorSize( tensor->attr.size, tensor->attr.dim_num, tensor->attr.dtype.vx_type );
    if( request_data_size != buffer->data_size )
    {
        VSILOGE("Internal buffer size error.\n");
        return status;
    }

    status = vsi_nn_CopyDataToTensor( graph, tensor, buffer->data );

    return status;
} /* internal_buffer_copy_to_tensor() */

static vsi_status internal_buffer_copy_from_tensor
    (
    vsi_nn_graph_t* graph,
    vsi_nn_rnn_internal_buffer_t* buffer,
    vsi_nn_tensor_id_t tensorid
    )
{
    vsi_status status = VSI_FAILURE;
    uint32_t request_data_size = 0;
    uint8_t* data = NULL;
    vsi_nn_tensor_t* tensor = NULL;

    if( NULL == buffer )
    {
        VSILOGE("Internal buffer is NULL.\n");
        return status;
    }

    tensor = vsi_nn_GetTensor( graph, tensorid );
    request_data_size = vsi_nn_GetTensorSize( tensor->attr.size, tensor->attr.dim_num, tensor->attr.dtype.vx_type );
    if( request_data_size != buffer->data_size )
    {
        VSILOGE("Internal buffer size error.\n");
        return status;
    }

    data = vsi_nn_ConvertTensorToData( graph, tensor );
    if( buffer->data && data )
    {
        memcpy( buffer->data, data, request_data_size );
        status = VSI_SUCCESS;
    }

    vsi_nn_safe_free( data );

    return status;
} /* internal_buffer_copy_from_tensor() */

static vsi_status _swap_rnn_tensor_handle
    (
    vsi_nn_graph_t* graph,
    vsi_nn_tensor_id_t output_id,
    vsi_nn_tensor_id_t input_id
    )
{
    vsi_nn_tensor_t* tensor_out = NULL;
    vsi_nn_tensor_t* tensor_in = NULL;

    tensor_out = vsi_nn_GetTensor( graph, output_id );
    tensor_in = vsi_nn_GetTensor( graph, input_id );

    return vsi_nn_SwapTensorHandle( tensor_out, tensor_in );
} /* _swap_rnn_tensor_handle() */

/**********************************************************
* PUBLIC FUNCTIONS
**********************************************************/
vsi_status vsi_nn_rnn_feed_internal_state
    (
    vsi_nn_graph_t* graph
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_rnn_connection_t * cur_conn = NULL;
    uint32_t i = 0;

    /* copy previous data from internal buffer to related input tensors */
    if( NULL != graph->rnn_wksp )
    {
        cur_conn = RNN_WKSP(graph)->external_connection_list;
        while( NULL != cur_conn && VSI_SUCCESS == status )
        {
            if( cur_conn->tensor_swappable )
            {
                status = _swap_rnn_tensor_handle( graph, cur_conn->connection.output,
                            cur_conn->connection.inputs[0] );
                if( VSI_SUCCESS != status )
                {
                    VSILOGE("Swap handle of RNN input/output fail.");
                    break;
                }
            }
            else
            {
                for( i = 0; i < cur_conn->connection_inputs_count; i++ )
                {
                    vsi_nn_tensor_id_t input = cur_conn->connection.inputs[i];

                    status = internal_buffer_copy_to_tensor( graph, &cur_conn->buffer, input );
                    if( VSI_SUCCESS != status )
                    {
                        break;
                    }
                }
            }
            cur_conn = (vsi_nn_rnn_connection_t *)vsi_nn_LinkListNext( (vsi_nn_link_list_t *)cur_conn );
        }
    }

    return status;
} /* vsi_nn_rnn_feed_internal_state() */

vsi_status vsi_nn_rnn_save_internal_state
    (
    vsi_nn_graph_t* graph
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_rnn_connection_t * cur_conn = NULL;

    if( VSI_SUCCESS == status )
    {
        /* copy tensors' data to internal buffer */
        if( NULL != graph->rnn_wksp )
        {
            cur_conn = RNN_WKSP(graph)->external_connection_list;
            while( NULL != cur_conn && VSI_SUCCESS == status )
            {
                if( !cur_conn->tensor_swappable )
                {
                    status = internal_buffer_copy_from_tensor( graph,
                                &cur_conn->buffer, cur_conn->connection.output );
                }
                cur_conn = (vsi_nn_rnn_connection_t *)vsi_nn_LinkListNext( (vsi_nn_link_list_t *)cur_conn );
            }
        }
    }

    return status;
} /* vsi_nn_rnn_save_internal_state() */

vsi_status vsi_nn_rnn_DeinitWksp
    (
    vsi_nn_graph_t* graph
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_rnn_connection_t * cur_conn = NULL;

    if( NULL == graph )
    {
        status = VSI_FAILURE;
        return status;
    }

    if( NULL == graph->rnn_wksp )
    {
        return status;
    }

    while( NULL != RNN_WKSP(graph)->external_connection_list )
    {
        cur_conn = (vsi_nn_rnn_connection_t *)vsi_nn_LinkListPopStart(
            (vsi_nn_link_list_t **)&RNN_WKSP(graph)->external_connection_list );
        internal_buffer_deinit( &cur_conn->buffer );
        vsi_nn_safe_free( cur_conn );
    }

    vsi_nn_safe_free( graph->rnn_wksp );

    return status;
} /* vsi_nn_rnn_DeinitWksp() */

vsi_status vsi_nn_rnn_InitWksp
    (
    vsi_nn_graph_t* graph,
    const vsi_nn_rnn_external_connection_t* connections,
    uint32_t connections_count,
    void* user_data
    )
{
    vsi_status status = VSI_SUCCESS;
    uint32_t i = 0;
    uint32_t j = 0;
    vsi_nn_rnn_connection_t* cur_conn = NULL;

    if( NULL == graph )
    {
        status = VSI_FAILURE;
        return status;
    }

    vsi_nn_rnn_DeinitWksp( graph );

    graph->rnn_wksp = malloc( sizeof( vsi_nn_rnn_wksp_t ) );
    if( NULL == graph->rnn_wksp )
    {
        VSILOGE("Malloc memory for rnn_wksp fail, Out of memory.");
        status = VSI_FAILURE;
        return status;
    }

    memset( graph->rnn_wksp, 0x00, sizeof( vsi_nn_rnn_wksp_t ) );
    RNN_WKSP(graph)->user_data = user_data;
    for( i = 0; i < connections_count; i++ )
    {
        cur_conn = (vsi_nn_rnn_connection_t *)malloc( sizeof( vsi_nn_rnn_connection_t ) );
        if( NULL == cur_conn )
        {
            VSILOGE("Malloc memory for connection fail, Out of memory.");
            status = VSI_FAILURE;
            break;
        }
        memset( cur_conn, 0x00, sizeof( vsi_nn_rnn_connection_t ) );
        memcpy( &cur_conn->connection, &connections[i], sizeof( connections[i] ) );
        internal_buffer_init( &cur_conn->buffer,
            vsi_nn_GetTensor( graph, cur_conn->connection.output ), 0.0f );
        vsi_nn_FillTensorWithValue( graph,
            vsi_nn_GetTensor( graph, cur_conn->connection.output ), 0.0f );

        for( j = 0; j < VSI_NN_MAX_RNN_CONNECTION_INPUTS; j++ )
        {
            if( VSI_NN_TENSOR_ID_NA == cur_conn->connection.inputs[j] )
            {
                break;
            }
        }

        if( j == VSI_NN_MAX_RNN_CONNECTION_INPUTS )
        {
            VSILOGE("The count of inputs is greater than maximum value: %d.", VSI_NN_MAX_RNN_CONNECTION_INPUTS);
            status = VSI_FAILURE;
            vsi_nn_safe_free( cur_conn );
            break;
        }
        else
        {
            cur_conn->connection_inputs_count = j;
        }

        if( cur_conn->connection_inputs_count == 1 )
        {
            vsi_nn_tensor_t* output_tensor = vsi_nn_GetTensor( graph, cur_conn->connection.output );
            vsi_nn_tensor_t* input_tensor = vsi_nn_GetTensor( graph, cur_conn->connection.inputs[0] );

            if( output_tensor && output_tensor->attr.is_created_from_handle
                && input_tensor && input_tensor->attr.is_created_from_handle )
            {
                cur_conn->tensor_swappable = TRUE;
            }
        }

        vsi_nn_LinkListPushEnd(
            (vsi_nn_link_list_t **)&RNN_WKSP(graph)->external_connection_list,
            (vsi_nn_link_list_t *)cur_conn );
    }

    return status;
} /* vsi_nn_rnn_InitWksp() */

vsi_status vsi_nn_rnn_ResetBuffers
    (
    vsi_nn_graph_t* graph
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_rnn_connection_t * cur_conn = NULL;

    if( NULL == graph )
    {
        status = VSI_FAILURE;
        return status;
    }

    if( NULL == graph->rnn_wksp )
    {
        return status;
    }

    if( NULL != graph->rnn_wksp )
    {
        cur_conn = RNN_WKSP(graph)->external_connection_list;
        while( NULL != cur_conn && VSI_SUCCESS == status )
        {
            status = internal_buffer_deinit( &cur_conn->buffer );
            status = internal_buffer_init( &cur_conn->buffer,
                vsi_nn_GetTensor( graph, cur_conn->connection.output ), 0.0f );
            status = vsi_nn_FillTensorWithValue( graph,
                vsi_nn_GetTensor( graph, cur_conn->connection.output ), 0.0f );

            cur_conn = (vsi_nn_rnn_connection_t *)vsi_nn_LinkListNext( (vsi_nn_link_list_t *)cur_conn );
        }
    }

    return status;
} /* vsi_nn_rnn_ResetBuffers() */

vsi_status vsi_nn_rnn_RunGraph
    (
    vsi_nn_graph_t* graph
    )
{
    vsi_status status = VSI_SUCCESS;

    status = vsi_nn_rnn_feed_internal_state( graph );

    if( VSI_SUCCESS == status )
    {
        status = vsi_nn_RunGraph( graph );
    }

    if( VSI_SUCCESS == status )
    {
        status = vsi_nn_rnn_save_internal_state( graph );
    }

    return status;
} /* vsi_nn_rnn_RunGraph() */

/**********************************************************
* HELPER FUNCTIONS
* The helper function to construct rnn variants in ovxlib.
**********************************************************/

vsi_bool vsi_nn_rnn_find_best_kernel_size
    (
    vsi_bool multi_batch,
    uint32_t input_size,
    uint32_t* p_kernel_h,
    uint32_t* p_kernel_w
    )
{
    uint32_t kernel_h = 1;
    uint32_t kernel_w = 1;

    if( multi_batch)
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
        if( !multi_batch )
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

vsi_nn_internal_tensor_t* vsi_nn_rnn_process_input_for_nn_fc
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

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    vsi_nn_internal_node_init_attr(&attr, &input->attr.dtype, use_virtual_tensor);
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

vsi_nn_internal_tensor_t* vsi_nn_rnn_process_output_for_nn_fc
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

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    vsi_nn_internal_node_init_attr(&attr, &input->attr.dtype, use_virtual_tensor);

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

vsi_nn_internal_tensor_t* vsi_nn_rnn_create_tp_fc
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

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    tensor = bias;
    if( !bias )
    {
        /* create zero bias for NN/TP */
        tensor1 = vsi_nn_create_zero_bias_tensor(self, &input->attr, &weight->attr);
        tensor = tensor1->t;
    }
    vsi_nn_internal_node_init_attr(&attr, output_dtype, use_virtual_tensor);
    tensor2 = vsi_nn_new_internal_tensor( self, &attr, 0.0f );

    tmp_inode = vsi_nn_new_internal_node(self, VSI_NN_OP_FCL, 0, 0 );
    tmp_inode->node->nn_param.fcl.axis = 0;
    tmp_inode->node->nn_param.fcl.weights = weight->attr.size[1];
    tmp_inode->node->vx_param.overflow_policy = VX_CONVERT_POLICY_WRAP;
    tmp_inode->node->vx_param.rounding_policy = VX_ROUND_POLICY_TO_ZERO;
    tmp_inode->node->vx_param.down_scale_size_rounding =
        VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_FLOOR;

    tmp_inode->inputs[0] = input;
    tmp_inode->inputs[1] = weight;
    tmp_inode->inputs[2] = tensor;
    tmp_inode->outputs[0] = tensor2->t;
    vsi_nn_setup_internal_node_op(self, tmp_inode);

    return tensor2;
}

vsi_nn_internal_tensor_t* vsi_nn_rnn_create_nn_fc
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

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    tensor = bias;
    if( !bias )
    {
        /* create zero bias for NN/TP */
        tensor1 = vsi_nn_create_zero_bias_tensor(self, &input->attr, &weight->attr);
        tensor = tensor1->t;
    }

    vsi_nn_internal_node_init_attr(&attr, output_dtype, use_virtual_tensor);
    tensor2 = vsi_nn_new_internal_tensor( self, &attr, 0.0f );

    reshaped_weight_shape[3] = weight->attr.size[1];
    reshaped_weight_shape[2] = weight->attr.size[0] / ( kernel_h * kernel_w );
    reshaped_weight_shape[1] = kernel_h;
    reshaped_weight_shape[0] = kernel_w;

    memset( attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
    attr.dim_num = VSI_NN_DIM_AUTO;
    attr.vtl = weight->attr.vtl;
    attr.is_const = FALSE; //weight->attr.is_const;
    memcpy( &attr.dtype, &weight->attr.dtype, sizeof(attr.dtype));
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
    tmp_inode->node->vx_param.down_scale_size_rounding =
        VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_FLOOR;

    tmp_inode->inputs[0] = input;
    tmp_inode->inputs[1] = reshaped_weight_tensor->t;
    tmp_inode->inputs[2] = tensor;
    tmp_inode->outputs[0] = tensor2->t;
    vsi_nn_setup_internal_node_op(self, tmp_inode);

    return tensor2;
}

vsi_nn_internal_tensor_t* vsi_nn_rnn_create_tensor_add
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

    tmp_inode = vsi_nn_new_internal_node(self, VSI_NN_OP_ADD, 0, 0 );

    tmp_inode->inputs[0] = input1;
    tmp_inode->inputs[1] = input2;
    tmp_inode->outputs[0] = tensor1->t;
    vsi_nn_setup_internal_node_op(self, tmp_inode);

    return tensor1;
}

vsi_nn_op_t vsi_nn_rnn_get_act_op_type
    (
    vsi_nn_activation_e type
    )
{
    switch (type)
    {
    case VSI_NN_ACT_RELU:
        return VSI_NN_OP_RELU;
    case VSI_NN_ACT_RELU6:
        return VSI_NN_OP_RELU6;
    case VSI_NN_ACT_TANH:
        return VSI_NN_OP_TANH;
    case VSI_NN_ACT_SIGMOID:
        return VSI_NN_OP_SIGMOID;
    default:
        VSILOGE("error activation type %d", type);
        break;
    }

    return VSI_NN_OP_TANH;
}

vsi_nn_internal_tensor_t* vsi_nn_rnn_create_activation
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vsi_nn_activation_e act_type,
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

    tmp_inode = vsi_nn_new_internal_node(self, vsi_nn_rnn_get_act_op_type(act_type), 0, 0 );

    tmp_inode->inputs[0] = input;
    tmp_inode->node->nn_param.tanh.scale_a = 1.0f;
    tmp_inode->node->nn_param.tanh.scale_b = 1.0f;
    tmp_inode->outputs[0] = tensor1->t;
    vsi_nn_setup_internal_node_op(self, tmp_inode);

    return tensor1;
}

vsi_nn_internal_tensor_t* vsi_nn_rnn_transpose_time_major
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vsi_nn_tensor_t * output,
    vsi_bool use_virtual_tensor
    )
{
    vsi_nn_tensor_attr_t attr;
    uint32_t* permute_in_perm = NULL;
    vsi_nn_internal_tensor_t* output_tensor = NULL;
    vsi_nn_internal_node_t* curr = NULL;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));

    if (output == NULL)
    {
        vsi_nn_internal_node_init_attr(&attr,
            &input->attr.dtype, use_virtual_tensor);
        output_tensor = vsi_nn_new_internal_tensor( self, &attr, 0.0f );
    }

    curr = vsi_nn_new_internal_node( self, VSI_NN_OP_PERMUTE, 0, 0 );
    permute_in_perm = (uint32_t *)vsi_nn_new_internal_node_param(curr,
        VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
    permute_in_perm[0] = 0;
    permute_in_perm[1] = 2;
    permute_in_perm[2] = 1;

    curr->node->nn_param.permute.perm = permute_in_perm;
    curr->node->nn_param.permute.dim_num = 3;
    curr->inputs[0] = input;

    if (output == NULL)
    {
        curr->outputs[0] = output_tensor->t;
    }
    else
    {
        curr->outputs[0] = output;
    }
    vsi_nn_setup_internal_node_op(self, curr);

    return output_tensor;
}

void vsi_nn_rnn_split_input_tensor
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vsi_nn_tensor_t ** output,
    uint32_t time_step,
    vsi_bool use_virtual_tensor
    )
{
    uint32_t* slices = NULL;
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* output_tensor = NULL;
    uint32_t i = 0;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    curr = vsi_nn_new_internal_node( self, VSI_NN_OP_SPLIT, 1, time_step );
    slices = (uint32_t *)vsi_nn_new_internal_node_param(curr, time_step * sizeof(uint32_t));
    curr->node->nn_param.split.axis = 2; /* timestep axis */
    curr->node->nn_param.split.slices_num = time_step;
    curr->inputs[0] = input;

    curr->node->nn_param.split.slices = slices;
    for( i = 0; i < time_step; i++ )
    {
        slices[i] = 1;
        vsi_nn_internal_node_init_attr(&attr, &input->attr.dtype, use_virtual_tensor);
        output_tensor = vsi_nn_new_internal_tensor( self, &attr, 0.0f );
        curr->outputs[i] = output_tensor->t;
        output[i] = output_tensor->t;
    }
    vsi_nn_setup_internal_node_op( self, curr );
}

void vsi_nn_rnn_data_check_aligned
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** input,
    uint32_t time_step,
    vsi_bool use_virtual_tensor
    )
{
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* output_tensor = NULL;
    uint32_t i = 0;
    uint32_t ofst = 0;
    ofst = 0;
    for( i = 0; i < time_step; i++ )
    {
        uint32_t tensor_size = vsi_nn_GetTensorSize( input[i]->attr.size,
            input[i]->attr.dim_num, input[i]->attr.dtype.vx_type );

        if( ofst & 0x3f )
        {
            vsi_nn_internal_node_init_attr(&attr, &input[i]->attr.dtype, use_virtual_tensor);
            output_tensor = vsi_nn_new_internal_tensor( self, &attr, 0.0f );

            curr = vsi_nn_new_internal_node( self, VSI_NN_OP_DATACONVERT, 0, 0 );
            curr->inputs[0] = input[i];
            curr->outputs[0] = output_tensor->t;
            vsi_nn_setup_internal_node_op( self, curr );

            input[i] = output_tensor->t;
        }

        ofst += tensor_size;
    }
}

vsi_nn_internal_tensor_t* vsi_nn_rnn_reshape_split_output
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    uint32_t batch_size,
    vsi_bool use_virtual_tensor
    )
{
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* output_tensor = NULL;
    uint32_t *reshape_split_size = NULL;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    /* reshape for split output */
    vsi_nn_internal_node_init_attr(&attr, &input->attr.dtype, use_virtual_tensor);
    output_tensor = vsi_nn_new_internal_tensor( self, &attr, 0.0f );

    curr = vsi_nn_new_internal_node( self, VSI_NN_OP_RESHAPE, 0, 0 );
    reshape_split_size = (uint32_t *)vsi_nn_new_internal_node_param(curr,
        VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
    reshape_split_size[0] = -1;
    reshape_split_size[1] = batch_size;

    curr->node->nn_param.reshape.size = reshape_split_size;
    curr->node->nn_param.reshape.dim_num = 2;
    curr->inputs[0] = input;
    curr->outputs[0] = output_tensor->t;
    vsi_nn_setup_internal_node_op( self, curr );

    return output_tensor;
}

vsi_nn_internal_tensor_t* vsi_nn_rnn_reshape_cell_output
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    uint32_t batch_size,
    vsi_bool use_virtual_tensor
    )
{
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t* output_tensor = NULL;
    uint32_t* reshape_grucell_output_size = NULL;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));

    /* reshape output to 3-dims */
    vsi_nn_internal_node_init_attr(&attr, &input->attr.dtype, use_virtual_tensor);
    output_tensor = vsi_nn_new_internal_tensor( self, &attr, 0.0f );

    curr = vsi_nn_new_internal_node( self, VSI_NN_OP_RESHAPE, 0, 0 );
    reshape_grucell_output_size = (uint32_t *)vsi_nn_new_internal_node_param(curr,
        VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
    reshape_grucell_output_size[0] = -1;
    reshape_grucell_output_size[1] = batch_size;
    reshape_grucell_output_size[2] = 1;

    curr->node->nn_param.reshape.size = reshape_grucell_output_size;
    curr->node->nn_param.reshape.dim_num = 3;
    curr->inputs[0] = input;
    curr->outputs[0] = output_tensor->t;
    vsi_nn_setup_internal_node_op( self, curr );

    return output_tensor;
}
