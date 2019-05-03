/****************************************************************************
*
*    Copyright (c) 2018 Vivante Corporation
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
#include "vsi_nn_rnn_prv.h"

/**********************************************************
* MACROS
**********************************************************/
#define RNN_SAFE_FREE(_PTR) { if( _PTR ) { free( _PTR ); _PTR = NULL; } }
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
        RNN_SAFE_FREE(data);
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

    RNN_SAFE_FREE( buffer->data );

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

    RNN_SAFE_FREE( data );

    return status;
} /* internal_buffer_copy_from_tensor() */

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
            for( i = 0; i < VSI_NN_MAX_RNN_CONNECTION_INPUTS; i++ )
            {
                vsi_nn_tensor_id_t input = VSI_NN_TENSOR_ID_NA;

                input = cur_conn->connection.inputs[i];
                if( VSI_NN_TENSOR_ID_NA == input )
                {
                    break;
                }

                status = internal_buffer_copy_to_tensor( graph, &cur_conn->buffer, input );
                if( VSI_SUCCESS != status )
                {
                    break;
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
                status = internal_buffer_copy_from_tensor( graph, &cur_conn->buffer, cur_conn->connection.output );
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
        RNN_SAFE_FREE( cur_conn );
    }

    RNN_SAFE_FREE( graph->rnn_wksp );

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

