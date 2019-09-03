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
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_log.h"
#include "utils/vsi_nn_util.h"

uint8_t * vsi_nn_VdataCreate
    (
    vsi_nn_graph_t * graph,
    vsi_nn_node_t  * node,
    uint32_t      * p_stream_size
    )
{
    /* declare variables */
    uint32_t           stream_size;
    uint8_t          * stream;
    vsi_status           status;
    vsi_nn_tensor_t   * input_tensors[3];
    vsi_nn_tensor_t   * output_tensors[2];

    stream = NULL;
    stream_size = 0;

    output_tensors[1] = NULL;
    vsi_nn_GetTensors( graph, node->input.tensors,
        node->input.num, input_tensors );
    vsi_nn_GetTensors( graph, node->output.tensors,
        node->output.num, output_tensors );

    status = vsi_nn_OpSetup( node->op, node,
        input_tensors, output_tensors );
    status = vsi_nn_OpOptimize( node->op, node,
        input_tensors, output_tensors, VSI_NN_OPTIMIZE_FORWARD );
    if( VSI_FAILURE == status )
    {
        goto fail;
    }

    /* get vdata buffer */
    stream = (uint8_t *)vxWeightsBiasesParameterToStream(
        graph->ctx->c,
        input_tensors[1]->wb, &stream_size,
        (vx_bool)FALSE
        );

fail:
    *p_stream_size = stream_size;
    return stream;
} /* vsi_nn_VdataCreate() */

vsi_nn_tensor_t * vsi_nn_CreateVDataTensor
    (
    vsi_nn_graph_t       * graph,
    uint8_t             * stream,
    vsi_nn_tensor_attr_t * attr
    )
{
    vsi_nn_tensor_t * tensor;

    tensor = NULL;
    if( NULL == graph || NULL == graph->ctx
        || NULL == attr || NULL == stream
        || NULL == graph->ctx->c )
    {
        return tensor;
    }
    tensor = (vsi_nn_tensor_t *)malloc( sizeof( vsi_nn_tensor_t ) );
    vsi_nn_UpdateTensorDims( attr );
    if( NULL != tensor )
    {
        memset( tensor, 0, sizeof( vsi_nn_tensor_t ) );
        memcpy( &tensor->attr, attr, sizeof( vsi_nn_tensor_attr_t ) );
        tensor->wb = vxCreateWeightsBiasesParameterFromStream(
            graph->ctx->c, (uint32_t *)stream );
        if( NULL == tensor->wb )
        {
            VSILOGE( "Create vdata fail." );
            free( tensor );
            tensor = NULL;
        }
    }
    return tensor;
} /* vsi_nn_CreateVDataTensor() */

