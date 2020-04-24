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
#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <VX/vx_khr_cnn.h>

#include "vsi_nn_pub.h"

#define _CHECK_TENSOR_ID( id, lbl )      do {\
    if( VSI_NN_TENSOR_ID_NA == id ) {\
        printf("CHECK TENSOR ID %d", __LINE__);\
        goto lbl;\
        }\
    } while(0)

#define _CHECK_PTR( ptr, lbl )      do {\
    if( NULL == ptr ) {\
        printf("CHECK PTR %d", __LINE__);\
        goto lbl;\
    }\
} while(0)

#define _CHECK_STATUS( stat, lbl )  do {\
    if( VX_SUCCESS != stat ) {\
        printf("CHECK STATUS %d", __LINE__);\
        goto lbl;\
    }\
} while(0)


#define _CONST_NODE_NUM     0
#define _TENSOR_NUM         (3)
#define _NODE_NUM           1

int main( int argc, char *argv[] )
{
vx_status             status;
vsi_nn_graph_t      * graph;
vsi_nn_context_t      ctx;
vsi_nn_node_t       * node;
vsi_nn_tensor_attr_t  attr;
vsi_nn_tensor_id_t    input[2];
vsi_nn_tensor_id_t    output[1];
vsi_nn_tensor_t     * tensor;
vx_float32            data1;
vx_float32            data2;
vx_float32          * out = NULL;

status = VSI_FAILURE;

ctx = vsi_nn_CreateContext();
_CHECK_PTR( ctx, final );

graph = vsi_nn_CreateGraph( ctx, _TENSOR_NUM, _NODE_NUM );
_CHECK_PTR( graph, final );

vsi_nn_SetGraphInputs( graph, NULL, 2 );
vsi_nn_SetGraphOutputs( graph, NULL, 1 );

node = vsi_nn_AppendNode( graph, VSI_NN_OP_MULTIPLY, NULL );
node->nn_param.multiply.scale = 0.5f;
node->vx_param.overflow_policy = VX_CONVERT_POLICY_WRAP;
node->vx_param.rounding_policy = VX_ROUND_POLICY_TO_ZERO;
_CHECK_PTR( node, final );

memset( &attr, 0, sizeof( vsi_nn_tensor_attr_t ) );
attr.size[0] = 1;
attr.size[1] = 1;
attr.size[2] = 1;
attr.dim_num = 3;
attr.vtl = FALSE;
attr.is_const = FALSE;
attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;

data1 = (vx_float32)1.5f;
data2 = (vx_float32)2.3f;

input[0] = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO, &attr, (vx_uint8*)&data1 );
input[1] = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO, &attr, (vx_uint8*)&data2 );
output[0] = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO, &attr, NULL );
_CHECK_TENSOR_ID( input[0], final );
_CHECK_TENSOR_ID( input[1], final );
_CHECK_TENSOR_ID( output[0], final );

node->input.tensors[0] = input[0];
node->input.tensors[1] = input[1];
node->output.tensors[0] = output[0];

graph->input.tensors[0] = input[0];
graph->input.tensors[1] = input[1];
graph->output.tensors[0] = output[0];

status = vsi_nn_SetupGraph( graph, TRUE );
_CHECK_STATUS( status, final );
status = vsi_nn_VerifyGraph( graph );
_CHECK_STATUS( status, final );
status = vsi_nn_RunGraph( graph );
_CHECK_STATUS( status, final );

tensor = vsi_nn_GetTensor( graph, output[0] );
_CHECK_PTR( tensor, final );
out = (vx_float32*)vsi_nn_ConvertTensorToData( graph, tensor );
_CHECK_PTR( out, final );

if( (data1 * data2 * 0.5f - *out) < 0.001 )
{
    status = VSI_SUCCESS;
}
else
{
    status = VSI_FAILURE;
}

final:
if(out != NULL)
{
    free(out);
    out = NULL;
}
vsi_nn_ReleaseGraph( &graph );
vsi_nn_ReleaseContext( &ctx );

return status;
}

