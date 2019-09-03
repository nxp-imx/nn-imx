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
#include <stdlib.h>
#include <string.h>

#include "vsi_nn_context.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_node_attr_template.h"
#include "vsi_nn_log.h"
#include "utils/vsi_nn_vdata.h"
#include "pycc/vsi_pycc_interface.h"

/* For pycc only */
static vsi_nn_context_t s_context = NULL;

static void _pycc_InitContext
    ( void )
{
    if( NULL != s_context)
    {
        ; // Do nothing
    }
    else
    {
        s_context = vsi_nn_CreateContext();
    }
} /* _pycc_InitContext() */

void vsi_pycc_VdataGeneratorInit
    ( void )
{
    _pycc_InitContext();
} /* vsi_pycc_VdataGeneratorInit() */

void vsi_pycc_VdataGeneratorDeinit
    ( void )
{
    vsi_nn_ReleaseContext( &s_context );
} /* vsi_pycc_VdataGeneratorDeinit() */

static vsi_nn_type_e _parse_vxtype
    (
    uint32_t ofst
    )
{
    vsi_nn_type_e val;
    val = VSI_NN_TYPE_FLOAT32;
    /* Index same with vdata_generator.py */
    switch( ofst )
    {
    case 0: val = VSI_NN_TYPE_FLOAT16;break;
    case 1: val = VSI_NN_TYPE_FLOAT32;break;
    case 2: val = VSI_NN_TYPE_INT8;break;
    case 3: val = VSI_NN_TYPE_INT16;break;
    case 4: val = VSI_NN_TYPE_INT32;break;
    case 5: val = VSI_NN_TYPE_UINT8;break;
    case 6: val = VSI_NN_TYPE_UINT16;break;
    case 7: val = VSI_NN_TYPE_UINT32;break;
    default:
        break;
    }
    return val;
} /* _parse_vxtype() */

static vsi_nn_op_t _parse_op
    (
    uint32_t ofst
    )
{
    vsi_nn_op_t val;
    val = VSI_NN_OP_CONV_RELU;
    /* Index same with vdata_generator.py */
    switch( ofst )
    {
    case 0: val = VSI_NN_OP_CONV_RELU;break;
    case 1: val = VSI_NN_OP_CONV_RELU_POOL;break;
    case 2: val = VSI_NN_OP_FCL_RELU;break;
    default:
        break;
    }
    return val;
} /* _parse_op() */

static vsi_nn_qnt_type_e _parse_qnt_type
    (
    uint32_t ofst
    )
{
    vsi_nn_qnt_type_e val;
    val = 0;
    /* Index same with vdata_generator.py */
    switch( ofst )
    {
    case 1: val = VSI_NN_QNT_TYPE_DFP;break;
    case 2: val = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;break;
    case 3: val = VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC;break;
    default:
        break;
    }
    return val;
} /* _parse_qnt_type() */

static vsi_enum _parse_pool_type
    (
    uint32_t ofst
    )
{
    vsi_enum val;
    val = VX_CONVOLUTIONAL_NETWORK_POOLING_MAX;
    /* Index same with vdata_generator.py */
    switch( ofst )
    {
    case 1: val = VX_CONVOLUTIONAL_NETWORK_POOLING_MAX;break;
    case 2: val = VX_CONVOLUTIONAL_NETWORK_POOLING_AVG;break;
    default:
        break;
    }
    return val;
} /* _parse_pool_type() */

static vsi_nn_round_type_e _parse_round_type
    (
    uint32_t ofst
    )
{
    vsi_nn_round_type_e val;
    val = VSI_NN_ROUND_CEIL;
    /* Index same with vdata_generator.py */
    switch( ofst )
    {
    case 0: val = VSI_NN_ROUND_CEIL;break;
    case 1: val = VSI_NN_ROUND_FLOOR;break;
    default:
        break;
    }
    return val;
} /* _parse_pool_type() */

static void _convert_tensor_attr
    (
    vsi_nn_tensor_attr_t * attr
    )
{
    attr->dtype.vx_type = _parse_vxtype( attr->dtype.vx_type );
    attr->dtype.qnt_type = _parse_qnt_type( attr->dtype.qnt_type );
} /* _convert_tensor_attr() */

uint32_t vsi_pycc_VdataCreate
    (
    vsi_pycc_params * pycc_params,
    uint8_t * buffer
    )
{
    uint32_t           stream_size;
    uint8_t          * stream;
    vsi_nn_graph_t    * graph;
    vsi_nn_tensor_t   * tensor;
    vsi_nn_node_t     * node;
    vsi_nn_context_t    ctx;

    stream_size = 0;
    graph = NULL;
    tensor = NULL;
    stream = NULL;
    node = NULL;

    if( NULL == s_context )
    {
        VSILOGW("No context inited.");
        return 0;
    }

    ctx = s_context;
    graph = vsi_nn_CreateGraph( ctx, 4, 1 );

    if( NULL == graph )
    {
        goto fail;
    }

    /* Convert enums */
    pycc_params->op = _parse_op( pycc_params->op );
    pycc_params->pool_type = _parse_pool_type( pycc_params->pool_type );
    pycc_params->pool_round_type = _parse_round_type( pycc_params->pool_round_type );
    _convert_tensor_attr( &pycc_params->input_attr );
    _convert_tensor_attr( &pycc_params->output_attr );
    _convert_tensor_attr( &pycc_params->weight_attr );
    _convert_tensor_attr( &pycc_params->bias_attr );

    /* Create node. */
    node = vsi_nn_AppendNode( graph, pycc_params->op, NULL );
    vsi_nn_apply_node_attr_template( node );

    switch( pycc_params->op )
    {
    case VSI_NN_OP_CONV_RELU_POOL:
        node->nn_param.pool.ksize[0]   = pycc_params->pool_ksize_h;
        node->nn_param.pool.ksize[1]   = pycc_params->pool_ksize_w;
        node->nn_param.pool.stride[0]  = pycc_params->pool_stride_h;
        node->nn_param.pool.stride[1]  = pycc_params->pool_stride_w;
        node->nn_param.pool.pad[0]     = pycc_params->pool_pad_left;
        node->nn_param.pool.pad[1]     = pycc_params->pool_pad_right;
        node->nn_param.pool.pad[2]     = pycc_params->pool_pad_top;
        node->nn_param.pool.pad[3]     = pycc_params->pool_pad_bottom;
        node->nn_param.pool.type       = pycc_params->pool_type;
        node->nn_param.pool.round_type = pycc_params->pool_round_type;
        /* Do not break */
    case VSI_NN_OP_CONV_RELU:
        node->nn_param.conv2d.ksize[0]    = pycc_params->ksize_h;
        node->nn_param.conv2d.ksize[1]    = pycc_params->ksize_w;
        node->nn_param.conv2d.stride[0]   = pycc_params->stride_h;
        node->nn_param.conv2d.stride[1]   = pycc_params->stride_w;
        node->nn_param.conv2d.pad[0]      = pycc_params->pad_left;
        node->nn_param.conv2d.pad[1]      = pycc_params->pad_right;
        node->nn_param.conv2d.pad[2]      = pycc_params->pad_top;
        node->nn_param.conv2d.pad[3]      = pycc_params->pad_bottom;
        node->nn_param.conv2d.dilation[0] = pycc_params->dilation_h;
        node->nn_param.conv2d.dilation[1] = pycc_params->dilation_w;
        node->nn_param.conv2d.weights     = pycc_params->weights;
        node->nn_param.conv2d.group       = 1;
        break;
    case VSI_NN_OP_FCL_RELU:
        node->nn_param.fcl.weights     = pycc_params->weights;
        break;
    default:
        VSILOGW( "Unsupport op %#x", pycc_params->op );
        return 0;
    }
    node->vx_param.has_relu = (vsi_bool)pycc_params->enable_relu;

    node->input.tensors[0] = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO,
        &pycc_params->input_attr, NULL );
    node->output.tensors[0] = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO,
        &pycc_params->output_attr, NULL );

    node->input.tensors[1] = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO,
        &pycc_params->weight_attr, pycc_params->weight_stream );

    /* Set dim auto to force compute shape. */
    tensor = vsi_nn_GetTensor( graph, node->output.tensors[0] );
    tensor->attr.dim_num = VSI_NN_DIM_AUTO;
    if( pycc_params->bias_stream_size > 0 )
    {
        node->input.tensors[2] = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO,
            &pycc_params->bias_attr, pycc_params->bias_stream );
    }

    stream = vsi_nn_VdataCreate( graph, node, &stream_size );
    if( 0 == stream_size )
    {
        VSILOGW("Create vdata fail.");
        vsi_nn_PrintGraph(graph);
    }

    /* release memory */
    if( NULL != stream)
    {
        memcpy( buffer, stream, stream_size );
        free( stream );
    }

fail:
    if( NULL != graph )
    {
        vsi_nn_ReleaseGraph( &graph );
    }

    return (uint32_t)stream_size;
} /* vsi_pycc_VdataCreate() */

