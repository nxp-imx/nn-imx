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

#include "vsi_nn_log.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_types.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_rnn.h"
#include "vsi_nn_test.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_vdata.h"
#include "utils/vsi_nn_link_list.h"

static vsi_status _set_reference_name
    (
    vsi_nn_graph_t *graph,
    vsi_nn_node_t *node
    )
{
#define _NODE_ID_LEN 64
    vsi_status status;
    vsi_nn_tensor_t *tensor;
    uint32_t i;
    char name[_NODE_ID_LEN];

    if(NULL == node || NULL == graph)
    {
        return VSI_FAILURE;
    }

    status = VSI_SUCCESS;
    for(i = 0; i < node->output.num; i++)
    {
        memset(name, 0, sizeof(char) * _NODE_ID_LEN);
        snprintf(name, sizeof(char) * _NODE_ID_LEN, "uid_%u_out_%u", node->uid, i);
        tensor = vsi_nn_GetTensor(graph, node->output.tensors[i]);
        if(tensor && tensor->t)
        {
            status = vxSetReferenceName((vx_reference)tensor->t, name);
            TEST_CHECK_STATUS(status, final);
        }
    }

final:
    return status;
} /* _set_reference_name() */

static void free_io_buffer
    (
    vsi_nn_tensor_t **buffer
    )
{
    if(buffer)
    {
        free(buffer);
        buffer = NULL;
    }
} /* free_io_buffer() */

static vsi_nn_tensor_t **allocate_io_buffer
    (
    vsi_nn_graph_t * graph
    )
{
    vsi_nn_tensor_t **buffer;

    buffer = (vsi_nn_tensor_t **)malloc(sizeof(vsi_nn_tensor_t *) * graph->max_node_io);
    if(NULL == buffer)
    {
        return NULL;
    }

    return buffer;
} /* allocate_io_buffer() */

static vsi_status update_max_node_io
    (
    vsi_nn_graph_t * graph,
    vsi_nn_node_id_t *node_list
    )
{
    uint32_t i,max_io;
    vsi_status status;
    vsi_nn_node_id_t node_id;
    vsi_nn_node_t   *node;

    status = VSI_SUCCESS;
    max_io = VSI_NN_MAX_IO_NUM; /* default max node io */
    for(i = 0; i < graph->node_num; i++)
    {
        node_id = node_list[i];
        node = vsi_nn_GetNode( graph, node_id );
        if(node->input.num > max_io)
        {
            max_io = node->input.num;
        }
        if(node->output.num > max_io)
        {
            max_io = node->output.num;
        }
    }

    graph->max_node_io = max_io;
    return status;
} /* update_max_node_io() */

static vsi_status optimize_node_backward
    (
    vsi_nn_graph_t * graph,
    vsi_nn_node_id_t *node_list
    )
{
    int32_t i;
    vsi_status status;
    vsi_nn_tensor_t **inputs;
    vsi_nn_tensor_t **outputs;
    vsi_nn_node_id_t node_id;
    vsi_nn_node_t   *node;

    status = VSI_SUCCESS;
    inputs = allocate_io_buffer(graph);
    outputs = allocate_io_buffer(graph);
    if(NULL == inputs || NULL == outputs)
    {
        VSILOGE("allocate io buffer fail");
        status =  VSI_FAILURE;
        goto final;
    }

    for( i = graph->node_num - 1; i >= 0; i-- )
    {
        node_id = node_list[i];
        memset( inputs, 0, graph->max_node_io * sizeof( vsi_nn_tensor_t * ) );
        memset( outputs, 0, graph->max_node_io * sizeof( vsi_nn_tensor_t * ) );

        /* Get inputs, outputs. */
        node = vsi_nn_GetNode( graph, node_id );
        vsi_nn_GetTensors( graph, node->input.tensors,
            node->input.num, inputs );
        vsi_nn_GetTensors( graph, node->output.tensors,
            node->output.num, outputs );

        status = vsi_nn_OpOptimize(node->op, node, inputs, outputs, VSI_NN_OPTIMIZE_BACKWARD);
        if( status != VSI_SUCCESS )
        {
            VSILOGE( "Backward optimize node[%u] %s fail",
                node_id, vsi_nn_OpGetName(node->op));
            break;
        }
    }

final:
    free_io_buffer(inputs);
    free_io_buffer(outputs);
    return status;
} /* optimize_node_backward() */

static vsi_status optimize_node_forward
    (
    vsi_nn_graph_t * graph,
    vsi_nn_node_id_t *node_list
    )
{
    uint32_t i;
    vsi_status status;
    vsi_nn_tensor_t **inputs;
    vsi_nn_tensor_t **outputs;
    vsi_nn_node_id_t node_id;
    vsi_nn_node_t   *node;

    status = VSI_SUCCESS;
    inputs = allocate_io_buffer(graph);
    outputs = allocate_io_buffer(graph);
    if(NULL == inputs || NULL == outputs)
    {
        VSILOGE("allocate io buffer fail");
        status =  VSI_FAILURE;
        goto final;
    }

    for( i = 0; i < graph->node_num; i++ )
    {
        node_id = node_list[i];
        memset( inputs, 0, graph->max_node_io * sizeof( vsi_nn_tensor_t * ) );
        memset( outputs, 0, graph->max_node_io * sizeof( vsi_nn_tensor_t * ) );

        /* Get inputs, outputs. */
        node = vsi_nn_GetNode( graph, node_id );
        vsi_nn_GetTensors( graph, node->input.tensors,
            node->input.num, inputs );
        vsi_nn_GetTensors( graph, node->output.tensors,
            node->output.num, outputs );

        status = vsi_nn_OpOptimize(node->op, node, inputs, outputs, VSI_NN_OPTIMIZE_FORWARD);
        if( status != VSI_SUCCESS )
        {
            VSILOGE( "Forward optimize node[%u] %s fail",
                node_id, vsi_nn_OpGetName(node->op));
            break;
        }
    }

final:
    free_io_buffer(inputs);
    free_io_buffer(outputs);
    return status;
} /* optimize_node_forward() */

static vsi_status compute_node
    (
    vsi_nn_graph_t * graph,
    vsi_nn_node_id_t *node_list
    )
{
    uint32_t i,j;
    vsi_status status;
    vsi_nn_tensor_t **inputs;
    vsi_nn_tensor_t **outputs;
    vsi_nn_node_id_t node_id;
    vsi_nn_node_t   *node;

    status = VSI_SUCCESS;
    inputs = allocate_io_buffer(graph);
    outputs = allocate_io_buffer(graph);
    if(NULL == inputs || NULL == outputs)
    {
        VSILOGE("allocate io buffer fail");
        status =  VSI_FAILURE;
        goto final;
    }

    for( i = 0; i < graph->node_num; i++ )
    {
        node_id = node_list[i];
        memset( inputs, 0, graph->max_node_io * sizeof( vsi_nn_tensor_t * ) );
        memset( outputs, 0, graph->max_node_io * sizeof( vsi_nn_tensor_t * ) );

        /* Get inputs, outputs. */
        node = vsi_nn_GetNode( graph, node_id );
        vsi_nn_GetTensors( graph, node->input.tensors,
            node->input.num, inputs );
        vsi_nn_GetTensors( graph, node->output.tensors,
            node->output.num, outputs );

        /* Create vx output tensor */
        VSILOGD("Create node[%u] %s", node_id, vsi_nn_OpGetName(node->op));
        for ( j = 0; j < node->output.num; j++ )
        {
            if( NULL == outputs[j] || NULL != outputs[j]->t )
                continue;
            vsi_nn_TensorReinit( graph, outputs[j] );
        }

        /* Create vx node */
        status = vsi_nn_OpCompute( node->op, node, inputs, outputs );
        if( VSI_SUCCESS != status )
        {
            VSILOGE( "Create node[%d] %s fail", node_id, vsi_nn_OpGetName(node->op));
            break;
        }
        status = _set_reference_name(graph, node);
        if( VSI_SUCCESS != status )
        {
            VSILOGW("Set reference name fail");
        }
    }

final:
    free_io_buffer(inputs);
    free_io_buffer(outputs);
    return status;
} /* compute_node */

static vsi_status optimize_node
    (
    vsi_nn_graph_t * graph,
    vsi_nn_node_id_t *node_list
    )
{
    vsi_status status;

    status = VSI_FAILURE;
    VSILOGD("Backward optimize neural network");
    status = optimize_node_backward(graph, node_list);
    if(status != VSI_SUCCESS)
    {
        return VSI_FAILURE;
    }

    VSILOGD("Forward optimize neural network");
    status = optimize_node_forward(graph, node_list);
    if(status != VSI_SUCCESS)
    {
        return VSI_FAILURE;
    }

    return status;
} /* optimize_node() */

static vsi_status setup_node
    (
    vsi_nn_graph_t * graph,
    vsi_nn_node_id_t *node_list
    )
{
    uint32_t i;
    vsi_status status;
    vsi_bool ret;
    vsi_nn_tensor_t **inputs;
    vsi_nn_tensor_t **outputs;
    vsi_nn_node_id_t node_id;
    vsi_nn_node_t   *node;

    status = VSI_SUCCESS;
    ret = TRUE;
    inputs = allocate_io_buffer(graph);
    outputs = allocate_io_buffer(graph);
    if(NULL == inputs || NULL == outputs)
    {
        VSILOGE("allocate io buffer fail");
        status =  VSI_FAILURE;
        goto final;
    }

    for( i = 0; i < graph->node_num; i++ )
    {
        node_id = node_list[i];
        memset( inputs, 0, graph->max_node_io * sizeof( vsi_nn_tensor_t * ) );
        memset( outputs, 0, graph->max_node_io * sizeof( vsi_nn_tensor_t * ) );

        /* Get inputs, outputs. */
        node = vsi_nn_GetNode( graph, node_id );
        vsi_nn_GetTensors( graph, node->input.tensors,
            node->input.num, inputs );
        vsi_nn_GetTensors( graph, node->output.tensors,
            node->output.num, outputs );

        VSILOGD("Preprocess node[%u] %s", node_id, vsi_nn_OpGetName(node->op));
        if( vsi_nn_OpCheck( node->op, node, inputs, outputs ) )
        {
            ret = vsi_nn_OpGenerateTensor( node, inputs, outputs );
            if(ret != TRUE)
            {
                VSILOGE( "Setup node[%u] %s fail", node_id, vsi_nn_OpGetName(node->op));
                status = VSI_FAILURE;
                break;
            }
        }
        else
        {
            VSILOGE( "Check node[%u] %s fail", node_id, vsi_nn_OpGetName(node->op));
            status = VSI_FAILURE;
            break;
        }
    }

final:
    free_io_buffer(inputs);
    free_io_buffer(outputs);
    return status;
} /* setup_node() */

vsi_nn_graph_t * vsi_nn_CreateGraph
    (
    vsi_nn_context_t ctx,
    uint32_t        max_tensor_num,
    uint32_t        max_node_num
    )
{
    vsi_nn_graph_t * graph;
    graph = NULL;
    if( NULL == ctx )
    {
        return graph;
    }

    graph = (vsi_nn_graph_t *)malloc( sizeof( vsi_nn_graph_t ) );
    if( NULL != graph )
    {
        memset( graph, 0, sizeof( vsi_nn_graph_t ) );
        graph->g = vxCreateGraph( ctx->c );
        if( NULL != graph->g )
        {
            graph->max_tensor_num = max_tensor_num;
            graph->max_node_num = max_node_num;
            graph->tensor_num = 0;
            graph->node_num = 0;
            graph->ctx = ctx;
            graph->rnn_wksp = NULL;
            if( 0 != max_tensor_num )
            {
                graph->tensors = (vsi_nn_tensor_t **)malloc(
                    max_tensor_num * sizeof( vsi_nn_tensor_t * ) );
                memset( graph->tensors, 0, max_tensor_num * sizeof( vsi_nn_tensor_t * ) );
            }
            if( 0 != max_node_num )
            {
                graph->nodes = (vsi_nn_node_t **)malloc(
                    max_node_num * sizeof( vsi_nn_node_t * ) );
                memset( graph->nodes, 0, max_node_num * sizeof( vsi_nn_node_t * ) );
            }
        }
        else
        {
            VSILOGE( "Create vx graph fail." );
            free( graph );
            graph = NULL;
        }
    }

    return graph;
} /* vsi_nn_CreateGraph() */

void vsi_nn_ReleaseGraph
    (
    vsi_nn_graph_t ** graph
    )
{
    uint32_t i;
    vsi_nn_graph_t * ptr;
    ptr = *graph;
    if( NULL != graph && NULL != * graph )
    {
        if( NULL != ptr->tensors )
        {
            for( i = 0; i < ptr->tensor_num; i++ )
            {
                vsi_nn_ReleaseTensor( &ptr->tensors[i] );
            }
            free( ptr->tensors );
        }
        if( NULL != ptr->nodes )
        {
            for( i = 0; i < ptr->node_num; i++ )
            {
                vsi_nn_ReleaseNode( &ptr->nodes[i] );
            }
            free( ptr->nodes );
        }
        if( NULL != ptr->input.tensors )
        {
            free( ptr->input.tensors );
        }
        if( NULL != ptr->output.tensors )
        {
            free( ptr->output.tensors );
        }
        if( NULL != ptr->rnn_wksp )
        {
            vsi_nn_rnn_DeinitWksp( ptr );
        }
        if( NULL != ptr->g )
        {
            vxReleaseGraph( &ptr->g );
        }
        free( ptr );
        *graph = NULL;
    }
} /* vsi_nn_ReleaseGraph() */

/*
* Create vx tensor and nodes.
* */
vsi_status vsi_nn_SetupGraph
    (
    vsi_nn_graph_t * graph,
    vsi_bool          sort
    )
{
    uint32_t i;
    vsi_status status;
    vsi_nn_node_id_t *sorted_nodes;
    vsi_nn_node_id_t *nodes_list;

    status = VSI_FAILURE;
    sorted_nodes = NULL;
    nodes_list = NULL;
    if( NULL == graph )
    {
        return status;
    }

#define MAX_NODES_IN_SDK 1024
    if(MAX_NODES_IN_SDK < graph->node_num)
    {
        VSILOGE("The number of nodes must be less than 1024\n");
        return status;
    }

    /* Prepare node list */
    nodes_list = (vsi_nn_node_id_t *)malloc(
        graph->node_num * sizeof( vsi_nn_node_id_t ) );
    if( !nodes_list )
    {
        goto final;
    }
    if( TRUE == sort )
    {
        VSILOGD( "Sort graph nodes.");
        sorted_nodes = vsi_nn_SortGraphNode( graph );
        memcpy(nodes_list, sorted_nodes,
            graph->node_num * sizeof( vsi_nn_node_id_t ));
    }
    else
    {
        for ( i = 0; i < graph->node_num; i++ )
        {
            nodes_list[i] = i;
        }
    }

    status = update_max_node_io( graph, nodes_list );
    if(VSI_SUCCESS != status)
    {
        goto final;
    }

    /* Preprocess node and tensor */
    status = setup_node( graph, nodes_list );
    if(VSI_SUCCESS != status)
    {
        goto final;
    }

    /* Optimize graph */
    status = optimize_node( graph, nodes_list );
    if(VSI_SUCCESS != status)
    {
        goto final;
    }

    /* Create vx node and vx virtual tensor */
    status = compute_node( graph, nodes_list );
    if(VSI_SUCCESS != status)
    {
        goto final;
    }

final:
    if( NULL != sorted_nodes )
    {
        free( sorted_nodes );
    }
    if ( NULL != nodes_list )
    {
        free( nodes_list );
    }
    return status;
} /* vsi_nn_SetupGraph() */

/*
* Call vx verify graph.
* */
vsi_status vsi_nn_VerifyGraph
    (
    vsi_nn_graph_t * graph
    )
{
    vsi_status status;
    status = VSI_FAILURE;
    if( NULL != graph->g )
    {
        status = vxVerifyGraph( graph->g );
    }
    return status;
} /* vsi_nn_VerifyGraph() */

vsi_status vsi_nn_RunGraph
    (
    vsi_nn_graph_t * graph
    )
{
    vsi_status status;
    status = VSI_FAILURE;
    if( NULL != graph->g )
    {
        if( vsi_nn_HasRNN( graph ) )
        {
            status = vsi_nn_rnn_feed_internal_state( graph );
        }
        else
        {
            status = VSI_SUCCESS;
        }

        if( VSI_SUCCESS == status )
        {
            status = vxProcessGraph( graph->g );
        }

        if( VSI_SUCCESS == status && vsi_nn_HasRNN( graph ) )
        {
            status = vsi_nn_rnn_save_internal_state( graph );
        }
    }
    return status;
} /* vsi_nn_RunGraph() */

vsi_nn_tensor_id_t vsi_nn_AddTensor
    (
    vsi_nn_graph_t       * graph,
    vsi_nn_tensor_id_t     id,
    vsi_nn_tensor_attr_t * attr,
    uint8_t             * data
    )
{
    vsi_nn_tensor_t * tensor;
    tensor = NULL;
    if( NULL == graph || NULL == attr )
    {
        return VSI_NN_TENSOR_ID_NA;
    }
    if( VSI_NN_TENSOR_ID_AUTO == id )
    {
        id = graph->cur_tid;
        graph->tensor_num = graph->cur_tid;
    }
    if( id < graph->max_tensor_num )
    {
        if( VSI_NN_TYPE_VDATA == attr->dtype.vx_type )
        {
            if( NULL == data )
            {
                id = VSI_NN_TENSOR_ID_NA;
            }
            else
            {
                tensor = vsi_nn_CreateVDataTensor( graph, data, attr );
            }
        }
        else if( NULL != data )
        {
            tensor = vsi_nn_CreateTensorFromData( graph, data, attr );
        }
        else
        {
            tensor = vsi_nn_CreateTensor( graph, attr );
        }
        graph->tensors[id] = tensor;
        if( NULL != tensor )
        {
            graph->cur_tid ++;
        }
    }
    else
    {
        id = VSI_NN_TENSOR_ID_NA;
    }
    return id;
} /* vsi_nn_AddTensor() */

vsi_nn_tensor_id_t vsi_nn_AttachTensorToGraph
    (
    vsi_nn_graph_t       * graph,
    vsi_nn_tensor_id_t     id,
    vsi_nn_tensor_t      * tensor
    )
{
    if( NULL == graph || NULL == tensor )
    {
        return VSI_NN_TENSOR_ID_NA;
    }
    if( VSI_NN_TENSOR_ID_AUTO == id )
    {
        id = graph->cur_tid;
        graph->tensor_num = graph->cur_tid;
    }
    if( id < graph->max_tensor_num )
    {
        graph->cur_tid ++;
        graph->tensors[id] = tensor;
    }
    else
    {
        id = VSI_NN_TENSOR_ID_NA;
    }
    return id;
} /* vsi_nn_AttachTensorToGraph() */

void vsi_nn_DeleteTensor
    (
    vsi_nn_graph_t       * graph,
    vsi_nn_tensor_id_t     id
    )
{
    if( NULL != graph && id < graph->tensor_num )
    {
        vsi_nn_ReleaseTensor( &graph->tensors[id] );
    }
} /* vsi_nn_DeleteTensor() */

vsi_nn_tensor_t * vsi_nn_GetTensor
    (
    vsi_nn_graph_t   * graph,
    vsi_nn_tensor_id_t id
    )
{
    vsi_nn_tensor_t * tensor;
    tensor = NULL;
    if( NULL != graph && id < graph->tensor_num )
    {
        tensor = graph->tensors[id];
    }
    return tensor;
} /* vsi_nn_GetTensor() */

vsi_nn_node_t * vsi_nn_GetNode
    (
    vsi_nn_graph_t   * graph,
    vsi_nn_node_id_t   id
    )
{
    vsi_nn_node_t * node;
    node = NULL;
    if( NULL != graph && id < graph->node_num )
    {
        node = graph->nodes[id];
    }
    return node;
} /* vsi_nn_GetTensor() */

void vsi_nn_GetTensors
    (
    vsi_nn_graph_t     * graph,
    vsi_nn_tensor_id_t * tensors_id,
    uint32_t            num,
    vsi_nn_tensor_t   ** tensors
    )
{
    uint32_t i;

    if( NULL == graph || NULL == graph->tensors
        || NULL == tensors_id || NULL == tensors)
    {
        return;
    }
    memset( &tensors[0], 0, sizeof( vsi_nn_tensor_t * ) * num  );
    if( num > graph->max_node_io )
    {
        VSILOGW( "Tensor num(%d) is greater than the MAX(%d), \
                 set to max num.", num, graph->max_node_io );
        num = graph->max_node_io;
    }
    for( i = 0; i < num; i++ )
    {
        if( VSI_NN_TENSOR_ID_NA == tensors_id[i] )
        {
            continue;
        }
        if( tensors_id[i] >= graph->tensor_num )
        {
            VSILOGE( "Tensor id %d/%d", tensors_id[i], graph->tensor_num );
            continue;
        }
        tensors[i] = graph->tensors[tensors_id[i]];
    }
} /* vsi_nn_GetTensors() */

vsi_nn_node_t * vsi_nn_AddNode
    (
    vsi_nn_graph_t      * graph,
    vsi_nn_op_t           op,
    uint32_t              input_num,
    uint32_t              output_num,
    vsi_nn_node_id_t    * node_id
    )
{
    vsi_nn_node_t * node;
    vsi_nn_node_id_t id;

    if( NULL == graph )
    {
        return NULL;
    }

    node = NULL;
    id = graph->cur_nid;

    if( id < graph->max_node_num )
    {
        node = vsi_nn_NewNode(graph, op, input_num, output_num);
        graph->nodes[id] = node;
        if( NULL != node )
        {
            graph->cur_nid ++;
            graph->node_num = graph->cur_nid;
        }
        else
        {
            id = VSI_NN_NODE_ID_NA;
        }
    }
    if( NULL != node_id )
    {
        *node_id = id;
    }
    return node;
} /* vsi_nn_AddNode() */

/*
 * Deprecated, Use vsi_nn_AddNode instead
 */
vsi_nn_node_t * vsi_nn_AppendNode
    (
    vsi_nn_graph_t      * graph,
    vsi_nn_op_t           op,
    vsi_nn_node_id_t    * node_id
    )
{
    return vsi_nn_AddNode( graph, op, 0, 0, node_id );
} /* vsi_nn_AppendNode() */

void vsi_nn_RemoveNode
    (
    vsi_nn_graph_t      * graph,
    vsi_nn_node_id_t      id
    )
{
    if( NULL != graph && id < graph->node_num )
    {
        vsi_nn_ReleaseNode( &graph->nodes[id] );
    }
} /* vsi_nn_RemoveNode() */

vsi_bool vsi_nn_SetGraphInputs
    (
    vsi_nn_graph_t      * graph,
    vsi_nn_tensor_id_t  * tensors_id,
    uint32_t             tensor_num
    )
{
    vsi_bool ret;
    ret = FALSE;

    if( NULL == graph || tensor_num == 0 )
    {
        return ret;
    }

    graph->input.tensors = (vsi_nn_tensor_id_t *)malloc(
        tensor_num * sizeof( vsi_nn_tensor_id_t ) );

    if( NULL != graph->input.tensors )
    {
        graph->input.num = tensor_num;
        ret = TRUE;
        if( NULL != tensors_id )
        {
            memcpy( graph->input.tensors, tensors_id,
                tensor_num * sizeof( vsi_nn_tensor_id_t ) );
        }
    }

    return ret;
} /* vsi_nn_SetGreaphInputs() */

vsi_bool vsi_nn_SetGraphOutputs
    (
    vsi_nn_graph_t      * graph,
    vsi_nn_tensor_id_t  * tensors_id,
    uint32_t             tensor_num
    )
{
    vsi_bool ret;
    ret = FALSE;

    if( NULL == graph || tensor_num == 0 )
    {
        return ret;
    }

    graph->output.tensors = (vsi_nn_tensor_id_t *)malloc(
        tensor_num * sizeof( vsi_nn_tensor_id_t ) );
    if( NULL != graph->output.tensors )
    {
        graph->output.num = tensor_num;
        ret = TRUE;
        if( NULL != tensors_id )
        {
            memcpy( graph->output.tensors, tensors_id,
                tensor_num * sizeof( vsi_nn_tensor_id_t ) );
        }
    }

    return ret;

} /* vsi_nn_SetGraphOutputs() */

vsi_nn_node_id_t * vsi_nn_SortGraphNode
    (
    vsi_nn_graph_t * graph
    )
{
    uint32_t i,j;
    uint32_t             count;
    vsi_bool             dirty;
    vsi_bool             all_tensor_processed;
    vsi_bool           * tensors;
    vsi_nn_node_id_t   * nodes;
    vsi_nn_node_id_t   * sorted_nodes;
    vsi_nn_node_t      * node;
    vsi_nn_node_id_t     node_id;
    vsi_nn_tensor_id_t   tensor_id;

    if( NULL == graph || NULL == graph->nodes
        || NULL == graph->tensors )
    {
        return NULL;
    }

    tensors      = NULL;
    sorted_nodes = NULL;
    nodes        = NULL;
    node         = NULL;

    /* Init variables. */
    tensors = (vsi_bool *)malloc(
        graph->tensor_num * sizeof( vsi_bool ) );

    if( NULL == tensors )
    {
        goto _SortGraphNodeFinally;
    }

    sorted_nodes = (vsi_nn_node_id_t *)malloc(
        graph->node_num * sizeof( vsi_nn_node_id_t ) );
    nodes = (vsi_nn_node_id_t *)malloc(
        graph->node_num * sizeof( vsi_nn_node_id_t ) );

    if( NULL == sorted_nodes || NULL == nodes)
    {
        goto _SortGraphNodeFinally;
    }

    for( i = 0; i < graph->tensor_num; i++ )
    {
        if( NULL == graph->tensors[i]
        || TRUE == graph->tensors[i]->attr.is_const )
        {
            tensors[i] = TRUE;
        }
        else
        {
            tensors[i] = FALSE;
        }
    }

    for( i = 0; i < graph->input.num; i++ )
    {
        tensor_id = graph->input.tensors[i];
        tensors[tensor_id] = TRUE;
    }

    for( i = 0; i < graph->node_num; i++ )
    {
        nodes[i] = i;
    }
    count = graph->node_num;
    do
    {
        dirty = FALSE;
        for( i = 0; i < count; i ++ )
        {
            node_id = nodes[i];
            node = vsi_nn_GetNode( graph, node_id );
            all_tensor_processed = TRUE;
            for( j = 0; j < node->input.num; j ++ )
            {
                tensor_id = node->input.tensors[j];
                if( VSI_NN_TENSOR_ID_NA == tensor_id )
                {
                    break;
                }
                if( FALSE == tensors[tensor_id] )
                {
                    all_tensor_processed = FALSE;
                    break;
                }
            }
            if( TRUE == all_tensor_processed )
            {
                sorted_nodes[graph->node_num - count] = nodes[i];
                nodes[i] = nodes[count - 1];
                count --;
                i --;
                dirty = TRUE;
                for( j = 0; j < node->output.num; j ++ )
                {
                    tensor_id = node->output.tensors[j];
                    if( VSI_NN_TENSOR_ID_NA == tensor_id )
                    {
                        break;
                    }
                    tensors[tensor_id] = TRUE;
                }
            }
        }
        if( FALSE == dirty )
        {
            break;
        }
    } while( count > 0 );

    if( count != 0 )
    {
        free( sorted_nodes );
        sorted_nodes = NULL;
    }

_SortGraphNodeFinally:

    /* Release memory. */
    free( tensors );
    free( nodes );
    return sorted_nodes;
} /* vsi_nn_SortGraphNode() */

uint32_t vsi_nn_GetNodesByUids
    (
    vsi_nn_graph_t   * graph,
    uint32_t        * node_uids,
    uint32_t          node_uids_size,
    vsi_nn_node_id_t * nodes,
    uint32_t          nodes_num
    )
{
    uint32_t sz;
    uint32_t i;
    uint32_t j;

    sz = 0;
    if( NULL == nodes || 0 >= nodes_num )
    {
        return sz;
    }
    if( NULL != node_uids )
    {
        for( i = 0; i < node_uids_size; i++ )
        {
            for( j = 0; j < graph->node_num; j++ )
            {
                if( node_uids[i] == graph->nodes[j]->uid )
                {
                    nodes[sz] = (vsi_nn_node_id_t)j;
                    sz ++;
                    break;
                }
            }
        }
    }
    else
    {
        for( j = 0; j < graph->node_num; j++ )
        {
            nodes[j] = (vsi_nn_node_id_t)j;
        }
        sz = graph->node_num;
    }
    return sz;
} /* vsi_nn_GetNodesByUids() */

void vsi_nn_DumpGraphNodeOutputs
    (
    vsi_nn_graph_t * graph,
    const char     * path,
    uint32_t       * node_uids,
    uint32_t         node_uids_size,
    vsi_bool         force_fp32,
    vsi_nn_dim_fmt_e data_fmt
    )
{
    vsi_nn_DumpGraphNodeOutputsEx(graph, path, NULL, node_uids, node_uids_size, force_fp32, data_fmt );
} /* vsi_nn_DumpGraphNodeOutputs() */

void vsi_nn_DumpGraphNodeOutputsEx
    (
    vsi_nn_graph_t * graph,
    const char     * path,
    const char     * prefix,
    uint32_t       * node_uids,
    uint32_t         node_uids_size,
    vsi_bool         force_fp32,
    vsi_nn_dim_fmt_e data_fmt
    )
{
#define _MAX_TENSOR_NAME_SZ (1024)
#define _SHAPE_BUF_SZ   (64)
    char shape[_SHAPE_BUF_SZ] = { 0 };
    char filename[_MAX_TENSOR_NAME_SZ] = { 0 };
    char filename_prefix[_MAX_TENSOR_NAME_SZ] = { 0 };
    const char * op_name;
    uint32_t i;
    uint32_t o;
    uint32_t node_num;
    vsi_nn_node_id_t * nodes;
    vsi_nn_node_t    * node;
    vsi_nn_tensor_t  * tensor;

    if(vsi_nn_CheckFilePath(path) == FALSE)
    {
        return ;
    }

    if( NULL == node_uids )
    {
        node_num = graph->node_num;
    }
    else
    {
        if( node_uids_size <= 0 )
        {
            VSILOGE("Error node_uids_size: %d.", node_uids_size);
            return;
        }
        node_num = node_uids_size;
    }
    nodes = (vsi_nn_node_id_t *)malloc( node_num * sizeof( vsi_nn_node_id_t ) );
    if( NULL == nodes )
    {
        VSILOGE("Malloc nodes memory fail.");
        return;
    }
    node_num = vsi_nn_GetNodesByUids( graph, node_uids, node_uids_size,
        nodes, node_num );

    if( NULL != prefix )
    {
        strncpy(filename_prefix, prefix, _MAX_TENSOR_NAME_SZ);
        filename_prefix[_MAX_TENSOR_NAME_SZ - 1] = '\0';

        strncat(filename_prefix, "_", _MAX_TENSOR_NAME_SZ);
        filename_prefix[_MAX_TENSOR_NAME_SZ - 1] = '\0';
    }

    VSILOGD("Dump %u nodes.", node_num);
    for( i = 0; i < node_num; i++ )
    {
        node = vsi_nn_GetNode( graph, (vsi_nn_node_id_t)i );
        for( o = 0; o < node->output.num; o++ )
        {
            tensor = vsi_nn_GetTensor( graph, node->output.tensors[o] );
            if( NULL != tensor )
            {
                if( TRUE == tensor->attr.vtl )
                {
                    VSILOGW("Uid %u node's tensor %d is virtual",
                        node->uid, o);
                    continue;
                }
                // TODO: Support different tensor format
                vsi_nn_ShapeToString( tensor->attr.size, tensor->attr.dim_num,
                    shape, _SHAPE_BUF_SZ, FALSE );
                op_name = vsi_nn_OpGetName( node->op );
                snprintf( filename, _MAX_TENSOR_NAME_SZ,
                    "%s/%s%s_uid_%u_t_%u_s_%s.txt", path, filename_prefix, op_name, node->uid, o, shape);
                if( FALSE == force_fp32 )
                {
                    vsi_nn_SaveTensorToText( graph, tensor, filename, NULL );
                }
                else
                {
                    vsi_nn_SaveTensorToTextByFp32( graph, tensor, filename, NULL );
                }
            }
        }
    }
    free( nodes );
} /* vsi_nn_DumpGraphNodeOutputsEx */

void vsi_nn_PrintGraph
    (
    vsi_nn_graph_t * graph
    )
{
    vsi_nn_tensor_t * tensor;
    vsi_nn_node_t * node;
    uint32_t i;

    if( NULL == graph )
    {
        return;
    }

    VSILOGI( "Graph:" );
    VSILOGI( "***************** Tensors ******************" );
    for( i = 0; i < graph->tensor_num; i ++ )
    {
        tensor = graph->tensors[i];
        if( NULL != tensor )
        {
            vsi_nn_PrintTensor( tensor, (vsi_nn_tensor_id_t)i );
        }
    }
    VSILOGI( "***************** Nodes ******************" );
    for( i = 0; i < graph->node_num; i ++ )
    {
        node = vsi_nn_GetNode( graph, (vsi_nn_node_id_t)i );
        if( NULL != node )
        {
            vsi_nn_PrintNode( node, (vsi_nn_node_id_t)i );
        }
    }
    VSILOGI("******************************************" );
} /* vsi_nn_PrintGraph() */

void vsi_nn_DumpGraphToJson
    (
    vsi_nn_graph_t *graph
    )
{
#define _SHAPE_BUF_SIZE 64
    uint32_t i,j;
    FILE *fp;
    vsi_nn_tensor_rel_t *tensor_ref, *tio;
    vsi_nn_tensor_rel_table_t *table;
    vsi_nn_node_t *node,*in_node;
    vsi_nn_tensor_t *tensor;
    char shape[_SHAPE_BUF_SIZE] = { 0 };

    if(NULL == graph)
    {
        return ;
    }

    fp = fopen("graph.json", "w+");
    if(NULL == fp)
    {
        VSILOGE("Create dump file fail");
        return ;
    }

    tensor_ref = vsi_nn_CreateTensorRelevance(graph);
    if(NULL == tensor_ref)
    {
        VSILOGE("build tensor io fail");
        return ;
    }

    fprintf(fp, "{\n");
    fprintf(fp, "\t\"Layers\":{\n");
    for(i = 0; i < graph->node_num; i++)
    {
        node = vsi_nn_GetNode(graph, i);
        if(node)
        {
            fprintf(fp, "\t\t\"uid_%u\":{\n\t\t\t\"op\": \"%s\",\n",
                node->uid, vsi_nn_OpGetName(node->op));

            /* dump inputs */
            fprintf(fp, "\t\t\t\"inputs\": [ ");
            for(j = 0; j < node->input.num; j++)
            {
                tio = &tensor_ref[node->input.tensors[j]];
                if(tio->input.num > 0)
                {
                    table = tio->input.table;

                    /* tensor only 1 input node */
                    in_node = vsi_nn_GetNode(graph, table[0].node);
                    if(j == node->input.num - 1)
                    {
                        fprintf(fp, "\"@uid_%u:out%u\" ", in_node->uid, table[0].index);
                    }
                    else
                    {
                        fprintf(fp, "\"@uid_%u:out%u\", ", in_node->uid, table[0].index);
                    }
                }

            }

            /* dump input shape */
            fprintf(fp, "],\n\t\t\t\"inut_shape\": [ ");
            for(j = 0; j < node->input.num; j++)
            {
                tensor = vsi_nn_GetTensor(graph, node->input.tensors[j]);
                if(vsi_nn_ShapeToString( tensor->attr.size, tensor->attr.dim_num,
                    shape, _SHAPE_BUF_SIZE, TRUE ) > 0)
                {
                    fprintf(fp, "[%s ]", shape);
                }
                else
                {
                    fprintf(fp, "[ - ]");
                }
                if(j < node->input.num - 1)
                {
                    fprintf(fp, ",");
                }
            }

            /* dump output */
            fprintf(fp, " ],\n\t\t\t\"outputs\": [ ");
            for(j = 0; j < node->output.num; j++)
            {
                if(j == node->output.num - 1)
                {
                    fprintf(fp, "\"out%u\" ", j);
                }
                else
                {
                    fprintf(fp, "\"out%u\", ", j);
                }
            }

            //output shape
            fprintf(fp, "],\n\t\t\t\"output_shape\": [ ");
            for(j = 0; j < node->output.num; j++)
            {
                tensor = vsi_nn_GetTensor(graph, node->output.tensors[j]);
                if(vsi_nn_ShapeToString( tensor->attr.size, tensor->attr.dim_num,
                    shape, _SHAPE_BUF_SIZE, TRUE ) > 0)
                {
                    fprintf(fp, "[%s ]", shape);
                }
                else
                {
                    fprintf(fp, "[ - ]");
                }
                if(j < node->output.num - 1)
                {
                    fprintf(fp, ",");
                }
            }
            fprintf(fp, " ]\n\t\t}");

            if(i != graph->node_num - 1)
            {
                fprintf(fp, ",");
            }
            fprintf(fp, "\n");
        }
    }
    fprintf(fp, "\t}\n}\n");

    vsi_nn_ReleaseTensorRelevance(graph, tensor_ref);
    fclose(fp);
} /* vsi_nn_DumpGraphToJson() */

vsi_status vsi_nn_SetupRNNConnections
    (
    vsi_nn_graph_t* graph,
    const vsi_nn_rnn_external_connection_t* connections,
    uint32_t connections_count
    )
{
    return vsi_nn_rnn_InitWksp( graph, connections, connections_count, NULL );
} /* vsi_nn_SetupRNNConnections() */

vsi_status vsi_nn_ResetRNNBuffers
    (
    vsi_nn_graph_t* graph
    )
{
    return vsi_nn_rnn_ResetBuffers( graph );
} /* vsi_nn_ResetRNNBuffers() */

vsi_bool vsi_nn_HasRNN
    (
    vsi_nn_graph_t* graph
    )
{
    return NULL != graph && NULL != graph->rnn_wksp;
} /* vsi_nn_HasRNN() */

