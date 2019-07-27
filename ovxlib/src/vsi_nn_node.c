/****************************************************************************
*
*    Copyright 2012 - 2019 Vivante Corporation, Santa Clara, California.
*    All Rights Reserved.
*
*    Permission is hereby granted, free of charge, to any person obtaining
*    a copy of this software and associated documentation files (the
*    'Software'), to deal in the Software without restriction, including
*    without limitation the rights to use, copy, modify, merge, publish,
*    distribute, sub license, and/or sell copies of the Software, and to
*    permit persons to whom the Software is furnished to do so, subject
*    to the following conditions:
*
*    The above copyright notice and this permission notice (including the
*    next paragraph) shall be included in all copies or substantial
*    portions of the Software.
*
*    THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
*    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
*    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
*    IN NO EVENT SHALL VIVANTE AND/OR ITS SUPPLIERS BE LIABLE FOR ANY
*    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
*    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
*    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/


#include <stdlib.h>
#include <string.h>

#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_types.h"
#include "utils/vsi_nn_util.h"

vsi_nn_node_t * vsi_nn_NewNode
    (
    vsi_nn_graph_t * graph,
    vsi_nn_op_t      op,
    uint32_t         input_num,
    uint32_t         output_num
    )
{
    vsi_nn_node_t * node;

    node = NULL;
    if(NULL == graph || FALSE == vsi_nn_OpIsValid(op))
    {
        VSILOGE("Create node %s. fail", vsi_nn_OpGetName(op));
        return NULL;
    }

    node = (vsi_nn_node_t *)malloc( sizeof( vsi_nn_node_t ) );
    if( NULL != node )
    {
        memset( node, 0, sizeof( vsi_nn_node_t ) );
        node->graph = graph;
        node->op = op;

        /* init op */
        vsi_nn_OpInit( node->op, node );

        if( 0 == input_num && 0 == output_num )
            {
            vsi_nn_OpGetIoNum( op, node, &input_num, &output_num );
            }

        /* init output struct */
        node->output.num = output_num;
        node->output.tensors = (vsi_nn_tensor_id_t *) malloc(
            output_num * sizeof( vsi_nn_tensor_id_t ) );
        vsi_nn_InitTensorsId( node->output.tensors, output_num );

        /* init input struct */
        node->input.num = input_num;
        node->input.tensors = (vsi_nn_tensor_id_t *) malloc(
            input_num * sizeof( vsi_nn_tensor_id_t ) );
        vsi_nn_InitTensorsId( node->input.tensors, input_num );
    }

    node->uid = VSI_NN_NODE_UID_NA;
    return node;
} /* vsi_nn_NewNode() */

/*
* Deprecated: Use vsi_nn_NewNode() instead
*/
vsi_nn_node_t * vsi_nn_CreateNode
    (
    vsi_nn_graph_t * graph,
    vsi_nn_op_t      op
    )
{
    return vsi_nn_NewNode( graph, op, 0, 0 );
} /* vsi_nn_CreateNode() */

void vsi_nn_ReleaseNode
    (
    vsi_nn_node_t ** node
    )
{
    vsi_nn_node_t * ptr;
    ptr = *node;
    if( NULL != node && NULL != *node )
    {
        vsi_nn_OpDeinit( ptr->op, ptr );
        if( NULL != ptr->input.tensors )
        {
            free( ptr->input.tensors );
        }
        if( NULL != ptr->output.tensors )
        {
            free( ptr->output.tensors );
        }
        free( ptr );
        *node = NULL;
    }
} /* vsi_nn_ReleaseNode() */

void vsi_nn_PrintNode
    (
    vsi_nn_node_t * node,
    vsi_nn_node_id_t id
    )
{
#define _MAX_PRINT_BUF_SZ   (256)
    uint32_t i;
    int count;
    char buf[_MAX_PRINT_BUF_SZ];

    if( NULL == node )
    {
        return;
    }
    count = snprintf( &buf[0], _MAX_PRINT_BUF_SZ, "%s", "[in:" );
    for( i = 0; i < node->input.num; i ++ )
    {
        if( count >= _MAX_PRINT_BUF_SZ )
        {
            break;
        }
        count += snprintf( &buf[count], _MAX_PRINT_BUF_SZ - count,
            " %d,", node->input.tensors[i] );
    }
    count --;
    count += snprintf( &buf[count], _MAX_PRINT_BUF_SZ - count,
        "%s", " ], [out:" );
    for( i = 0; i < node->output.num; i ++ )
    {
        if( count >= _MAX_PRINT_BUF_SZ )
        {
            break;
        }
        count += snprintf( &buf[count], _MAX_PRINT_BUF_SZ - count,
            " %d,", node->output.tensors[i] );
    }
    count --;
    count += snprintf( &buf[count], _MAX_PRINT_BUF_SZ - count,
        "%s", " ]" );
    VSILOGI( "(%16s)node[%u] %s [%08x]", vsi_nn_OpGetName(node->op), id, buf, node->n );
} /* vsi_nn_PrintNode() */

