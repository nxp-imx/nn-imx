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
#include "vsi_nn_internal_node.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_vdata.h"
#include "utils/vsi_nn_map.h"

vsi_nn_internal_node_t* vsi_nn_internal_create_node
    (
    vsi_nn_graph_t* graph,
    vsi_nn_op_t op,
    uint32_t input_num,
    uint32_t output_num
    )
{
    vsi_nn_internal_node_t* node = NULL;
    vsi_nn_node_t* n = NULL;
    vsi_nn_tensor_t** inputs = NULL;
    vsi_nn_tensor_t** outputs = NULL;

    node = (vsi_nn_internal_node_t *)malloc( sizeof(vsi_nn_internal_node_t) );
    if( node )
    {
        memset(node, 0x00, sizeof(vsi_nn_internal_node_t) );

        n = vsi_nn_NewNode( graph, op, input_num, output_num );
        if( n )
        {
            inputs = (vsi_nn_tensor_t **)malloc( n->input.num * sizeof(vsi_nn_tensor_t*));
            if( inputs )
            {
                memset( inputs, 0x00, ( n->input.num * sizeof(vsi_nn_tensor_t*)) );
            }
            outputs = (vsi_nn_tensor_t **)malloc( n->output.num * sizeof(vsi_nn_tensor_t*));
            if( outputs )
            {
                memset( outputs, 0x00, ( n->output.num * sizeof(vsi_nn_tensor_t*)) );
            }
        }
    }

    if( node && n && inputs && outputs )
    {
        node->node = n;
        node->inputs = inputs;
        node->outputs = outputs;

        return node;
    }
    else
    {
        vsi_nn_internal_release_node( &node );
        return NULL;
    }
}

vsi_status vsi_nn_internal_release_node
    (
    vsi_nn_internal_node_t** node
    )
{
    if( node && *node )
    {
        vsi_nn_internal_node_t* ptr = *node;

        if( ptr->inputs && ptr->node->input.num )
        {
            free( ptr->inputs );
            ptr->inputs = NULL;
        }
        if( ptr->outputs && ptr->node->output.num )
        {
            free( ptr->outputs );
            ptr->outputs = NULL;
        }
        if( ptr->node )
        {
            vsi_nn_ReleaseNode( &ptr->node );
        }

        free( ptr );
        *node = NULL;
    }

    return VSI_SUCCESS;
}

vsi_status vsi_nn_internal_create_node_outputs
    (
    vsi_nn_internal_node_t* node,
    vsi_nn_tensor_attr_t* attr

    )
{
    vsi_status status = VSI_SUCCESS;
    uint32_t i = 0;

    for( i = 0; i < node->node->output.num; i++ )
    {
        if( NULL == node->outputs[i] )
        {
            node->outputs[i] = vsi_nn_CreateTensor( node->node->graph, attr );
            if( !node->outputs[i] )
            {
                VSILOGE("Create output tensor fail");
                break;
            }
        }
    }

    if( i != node->node->output.num )
    {
        status = VSI_FAILURE;
    }

    return status;
}
