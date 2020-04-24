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
/*-------------------------------------------
                   Includes
 -------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>

#include "vsi_nn_pub.h"

#include "vnn_lenet.h"
/*-------------------------------------------
                   Macros
 -------------------------------------------*/

#define NEW_VXNODE(_node, _type) do {\
        _node = vsi_nn_AppendNode( graph, _type, NULL );\
        if( NULL == _node ) {\
            goto error;\
        }\
    } while(0)

#define NEW_VIRTUAL_TENSOR(_id, _attr, _dtype, _fl) do {\
        memset( _attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(vx_uint32));\
        _attr.dim_num = VSI_NN_DIM_AUTO;\
        _attr.vtl = vx_true_e;\
        _attr.dtype.vx_type = _dtype;\
        _attr.dtype.fl = _fl;\
        _id = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO,\
                & _attr, NULL );\
        if( VSI_NN_TENSOR_ID_NA == _id ) {\
            goto error;\
        }\
    } while(0)

// Set const tensor dims out of this macro.
#define NEW_CONST_TENSOR(_id, _attr, _dtype, _fl, _ofst, _size) do {\
        data = load_data( fp, _ofst, _size  );\
        _attr.vtl = vx_false_e;\
        _attr.is_const = vx_true_e;\
        _attr.dtype.vx_type = _dtype;\
        _attr.dtype.fl = _fl;\
        _id = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO,\
                & _attr, data );\
        free( data );\
        if( VSI_NN_TENSOR_ID_NA == _id ) {\
            goto error;\
        }\
    } while(0)

// Set generic tensor dims out of this macro.
#define NEW_NORM_TENSOR(_id, _attr, _dtype, _fl) do {\
        _attr.vtl = vx_false_e;\
        _attr.dtype.vx_type = _dtype;\
        _attr.dtype.fl = _fl;\
        _id = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO,\
                & _attr, NULL );\
        if( VSI_NN_TENSOR_ID_NA == _id ) {\
            goto error;\
        }\
    } while(0)

#define NET_NODE_NUM            (5)
#define NET_NORM_TENSOR_NUM     (2)
#define NET_CONST_TENSOR_NUM    (8)
#define NET_VIRTUAL_TENSOR_NUM  (5)
#define NET_TOTAL_TENSOR_NUM    (NET_NORM_TENSOR_NUM + NET_CONST_TENSOR_NUM + NET_VIRTUAL_TENSOR_NUM)

/*-------------------------------------------
               Local Variables
 -------------------------------------------*/

/*-------------------------------------------
                  Functions
 -------------------------------------------*/

static vx_uint8* load_data
    (
    FILE  * fp,
    size_t  ofst,
    size_t  sz
    )
{
vx_uint8* data;
size_t    ret;
data = NULL;
if( NULL == fp )
    {
    return NULL;
    }

ret = fseek(fp, ofst, SEEK_SET);
if (ret != 0)
    {
    VSILOGE("blob seek failure.");
    return NULL;
    }

data = (vx_uint8*)malloc(sz);
if (data == NULL)
    {
    VSILOGE("buffer malloc failure.");
    return NULL;
    }
ret = fread(data, 1, sz, fp);
VSILOGI("Read %d data.", ret);
return data;
} /* load_data() */

vsi_nn_graph_t * vnn_CreateLenet
    (
    char * data_file_name,
    vsi_nn_context_t in_ctx
    )
{
vx_status               status;
vx_bool                 release_ctx;
vsi_nn_context_t        ctx;
vsi_nn_graph_t *        graph;
vsi_nn_node_t *         node[NET_NODE_NUM];
vsi_nn_tensor_id_t      norm_tensor[NET_NORM_TENSOR_NUM];
vsi_nn_tensor_id_t      const_tensor[NET_CONST_TENSOR_NUM];
vsi_nn_tensor_attr_t    attr;
FILE *                  fp;
vx_uint8 *              data;


ctx = NULL;
graph = NULL;
status = VX_FAILURE;

fp = fopen( data_file_name, "rb" );
if( NULL == fp )
    {
    goto error;
    }

if( NULL == in_ctx )
    {
    ctx = vsi_nn_CreateContext();
    }
else
    {
    ctx = in_ctx;
    }

graph = vsi_nn_CreateGraph( ctx, NET_TOTAL_TENSOR_NUM, NET_NODE_NUM );
if( NULL == graph )
    {
    VSILOGE( "Create graph fail." );
    goto error;
    }
vsi_nn_SetGraphInputs( graph, NULL, 1 );
vsi_nn_SetGraphOutputs( graph, NULL, 1 );

/*-----------------------------------------
  Node definitions
 -----------------------------------------*/

/*-----------------------------------------
  lid       - conv1_1_pool1_2
  var       - node[0]
  name      - convolutionrelupool
  operation - convolutionrelupool
  in_shape  - [[28, 28, 1]]
  out_shape - [[12, 12, 20]]
 -----------------------------------------*/
NEW_VXNODE(node[0], VSI_NN_OP_CONV_RELU_POOL);
node[0]->nn_param.conv2d.ksize[0] = 5;
node[0]->nn_param.conv2d.ksize[1] = 5;
node[0]->nn_param.conv2d.weights = 20;
node[0]->nn_param.conv2d.stride[0] = 1;
node[0]->nn_param.conv2d.stride[1] = 1;
node[0]->nn_param.conv2d.pad[0] = 0;
node[0]->nn_param.conv2d.pad[1] = 0;
node[0]->nn_param.conv2d.pad[2] = 0;
node[0]->nn_param.conv2d.pad[3] = 0;
node[0]->nn_param.conv2d.group = 1;
node[0]->nn_param.conv2d.dilation[0] = 1;
node[0]->nn_param.conv2d.dilation[1] = 1;
node[0]->nn_param.pool.type = VX_CONVOLUTIONAL_NETWORK_POOLING_MAX;
node[0]->nn_param.pool.ksize[0] = 2;
node[0]->nn_param.pool.ksize[1] = 2;
node[0]->nn_param.pool.stride[0] = 2;
node[0]->nn_param.pool.stride[1] = 2;
node[0]->nn_param.pool.pad[0] = 0;
node[0]->nn_param.pool.pad[1] = 0;
node[0]->nn_param.pool.pad[2] = 0;
node[0]->nn_param.pool.pad[3] = 0;
node[0]->vx_param.has_relu = vx_false_e;
node[0]->vx_param.overflow_policy = VX_CONVERT_POLICY_WRAP;
node[0]->vx_param.rounding_policy = VX_ROUND_POLICY_TO_ZERO;
node[0]->vx_param.down_scale_size_rounding = VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_FLOOR;

/*-----------------------------------------
  lid       - conv2_3_pool2_4
  var       - node[1]
  name      - convolutionrelupool
  operation - convolutionrelupool
  in_shape  - [[12, 12, 20]]
  out_shape - [[4, 4, 50]]
 -----------------------------------------*/
NEW_VXNODE(node[1], VSI_NN_OP_CONV_RELU_POOL);
node[1]->nn_param.conv2d.ksize[0] = 5;
node[1]->nn_param.conv2d.ksize[1] = 5;
node[1]->nn_param.conv2d.weights = 50;
node[1]->nn_param.conv2d.stride[0] = 1;
node[1]->nn_param.conv2d.stride[1] = 1;
node[1]->nn_param.conv2d.pad[0] = 0;
node[1]->nn_param.conv2d.pad[1] = 0;
node[1]->nn_param.conv2d.pad[2] = 0;
node[1]->nn_param.conv2d.pad[3] = 0;
node[1]->nn_param.conv2d.group = 1;
node[1]->nn_param.conv2d.dilation[0] = 1;
node[1]->nn_param.conv2d.dilation[1] = 1;
node[1]->nn_param.pool.type = VX_CONVOLUTIONAL_NETWORK_POOLING_MAX;
node[1]->nn_param.pool.ksize[0] = 2;
node[1]->nn_param.pool.ksize[1] = 2;
node[1]->nn_param.pool.stride[0] = 2;
node[1]->nn_param.pool.stride[1] = 2;
node[1]->nn_param.pool.pad[0] = 0;
node[1]->nn_param.pool.pad[1] = 0;
node[1]->nn_param.pool.pad[2] = 0;
node[1]->nn_param.pool.pad[3] = 0;
node[1]->vx_param.has_relu = vx_false_e;
node[1]->vx_param.overflow_policy = VX_CONVERT_POLICY_WRAP;
node[1]->vx_param.rounding_policy = VX_ROUND_POLICY_TO_ZERO;
node[1]->vx_param.down_scale_size_rounding = VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_FLOOR;

/*-----------------------------------------
  lid       - ip1_5_relu1_6
  var       - node[2]
  name      - fullconnectrelu
  operation - fullconnectrelu
  in_shape  - [[4, 4, 50]]
  out_shape - [[500]]
 -----------------------------------------*/
NEW_VXNODE(node[2], VSI_NN_OP_FCL_RELU);
node[2]->nn_param.fcl.weights = 500;
node[2]->vx_param.has_relu = vx_true_e;
node[2]->vx_param.overflow_policy = VX_CONVERT_POLICY_WRAP;
node[2]->vx_param.rounding_policy = VX_ROUND_POLICY_TO_ZERO;
node[2]->vx_param.down_scale_size_rounding = VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_FLOOR;

/*-----------------------------------------
  lid       - trans_ip2_7
  var       - node[3]
  name      - fullconnectrelu
  operation - fullconnectrelu
  in_shape  - [[500]]
  out_shape - [[10]]
 -----------------------------------------*/
NEW_VXNODE(node[3], VSI_NN_OP_FCL_RELU);
node[3]->nn_param.fcl.weights = 10;
node[3]->vx_param.has_relu = vx_false_e;
node[3]->vx_param.overflow_policy = VX_CONVERT_POLICY_WRAP;
node[3]->vx_param.rounding_policy = VX_ROUND_POLICY_TO_ZERO;
node[3]->vx_param.down_scale_size_rounding = VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_FLOOR;

/*-----------------------------------------
  lid       - prob_8
  var       - node[4]
  name      - prob
  operation - softmax
  in_shape  - [[10]]
  out_shape - [[10]]
 -----------------------------------------*/
NEW_VXNODE(node[4], VSI_NN_OP_SOFTMAX);


/*-----------------------------------------
  Tensor initialize
 -----------------------------------------*/
attr.dtype.fmt = VSI_NN_DIM_FMT_NCHW;
attr.is_const = vx_false_e;
attr.size[0] = 28;
attr.size[1] = 28;
attr.size[2] = 1;
attr.dim_num = 3;
NEW_NORM_TENSOR(norm_tensor[0], attr, VSI_NN_TYPE_FLOAT16, 0);

attr.size[0] = 10;
attr.dim_num = 1;
NEW_NORM_TENSOR(norm_tensor[1], attr, VSI_NN_TYPE_FLOAT16, 0);



attr.size[0] = 5;
attr.size[1] = 5;
attr.size[2] = 1;
attr.size[3] = 20;
attr.dim_num = 4;
NEW_CONST_TENSOR(const_tensor[0], attr, VSI_NN_TYPE_FLOAT16, 0, 80, 1000);/* @conv1_1_pool1_2:weight */

attr.size[0] = 20;
attr.dim_num = 1;
NEW_CONST_TENSOR(const_tensor[1], attr, VSI_NN_TYPE_FLOAT32, 0, 0, 80);/* @conv1_1_pool1_2:bias */

attr.size[0] = 5;
attr.size[1] = 5;
attr.size[2] = 20;
attr.size[3] = 50;
attr.dim_num = 4;
NEW_CONST_TENSOR(const_tensor[2], attr, VSI_NN_TYPE_FLOAT16, 0, 1280, 50000);/* @conv2_3_pool2_4:weight */

attr.size[0] = 50;
attr.dim_num = 1;
NEW_CONST_TENSOR(const_tensor[3], attr, VSI_NN_TYPE_FLOAT32, 0, 1080, 200);/* @conv2_3_pool2_4:bias */

attr.size[0] = 800;
attr.size[1] = 500;
attr.dim_num = 2;
NEW_CONST_TENSOR(const_tensor[4], attr, VSI_NN_TYPE_FLOAT16, 0, 53280, 800000);/* @ip1_5_relu1_6:weight */

attr.size[0] = 500;
attr.dim_num = 1;
NEW_CONST_TENSOR(const_tensor[5], attr, VSI_NN_TYPE_FLOAT32, 0, 51280, 2000);/* @ip1_5_relu1_6:bias */

attr.size[0] = 500;
attr.size[1] = 10;
attr.dim_num = 2;
NEW_CONST_TENSOR(const_tensor[6], attr, VSI_NN_TYPE_FLOAT16, 0, 853320, 10000);/* @trans_ip2_7:weight */

attr.size[0] = 10;
attr.dim_num = 1;
NEW_CONST_TENSOR(const_tensor[7], attr, VSI_NN_TYPE_FLOAT32, 0, 853280, 40);/* @trans_ip2_7:bias */



attr.is_const = vx_false_e;
NEW_VIRTUAL_TENSOR(node[0]->output.tensors[0], attr, VSI_NN_TYPE_FLOAT16, 0);

NEW_VIRTUAL_TENSOR(node[1]->output.tensors[0], attr, VSI_NN_TYPE_FLOAT16, 0);

NEW_VIRTUAL_TENSOR(node[2]->output.tensors[0], attr, VSI_NN_TYPE_FLOAT16, 0);

NEW_VIRTUAL_TENSOR(node[3]->output.tensors[0], attr, VSI_NN_TYPE_FLOAT16, 0);



/*-----------------------------------------
  Connection initialize
 -----------------------------------------*/
node[0]->input.tensors[0] = norm_tensor[0];

node[0]->input.tensors[1] = const_tensor[0];
node[0]->input.tensors[2] = const_tensor[1];

node[1]->input.tensors[0] = node[0]->output.tensors[0];
node[1]->input.tensors[1] = const_tensor[2];
node[1]->input.tensors[2] = const_tensor[3];

node[2]->input.tensors[0] = node[1]->output.tensors[0];
node[2]->input.tensors[1] = const_tensor[4];
node[2]->input.tensors[2] = const_tensor[5];

node[3]->input.tensors[0] = node[2]->output.tensors[0];
node[3]->input.tensors[1] = const_tensor[6];
node[3]->input.tensors[2] = const_tensor[7];

node[4]->input.tensors[0] = node[3]->output.tensors[0];

node[4]->output.tensors[0] = norm_tensor[1];


graph->input.tensors[0] = norm_tensor[0];
graph->output.tensors[0] = norm_tensor[1];

status = vsi_nn_SetupGraph( graph, vx_false_e );
if( VX_FAILURE == status )
    {
    goto error;
    }

fclose( fp );

return graph;

error:

if( NULL != fp )
    {
    fclose( fp );
    }

release_ctx = (vx_bool)( NULL == in_ctx );
vnn_ReleaseLenet( graph, release_ctx );

return NULL;
} /* vsi_nn_CreateLenet() */

void vnn_ReleaseLenet
    (
    vsi_nn_graph_t * graph,
    vx_bool release_ctx
    )
{
vsi_nn_context_t ctx;
if( NULL != graph )
    {
    ctx = graph->ctx;
    vsi_nn_ReleaseGraph( &graph );
    if( release_ctx )
        {
        vsi_nn_ReleaseContext( &ctx );
        }
    }
} /* vsi_nn_ReleaseLenet() */

