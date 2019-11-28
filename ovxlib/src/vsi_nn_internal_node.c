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

/**********************************************************
* MACROS
**********************************************************/
#define LINKLIST_APPEND( _HEAD, _ITEM ) do {                \
    vsi_nn_LinkListPushEnd((vsi_nn_link_list_t **)&(_HEAD), \
    (vsi_nn_link_list_t *)(_ITEM) ); } while( 0 )

#define WKSP(_NODE_PTR) ((vsi_nn_internal_node_wksp_t *)    \
    ((_NODE_PTR)->internal_node_wksp))

/**********************************************************
* LOCAL FUNCTIONS
**********************************************************/
static vsi_nn_internal_node_t* vsi_nn_create_internal_node
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
        if(n)
        {
            vsi_nn_ReleaseNode(&n);
            n = NULL;
        }
        if(inputs)
        {
            free(inputs);
            inputs = NULL;
        }
        if(outputs)
        {
            free(outputs);
            outputs = NULL;
        }
        vsi_nn_release_internal_node( &node );
        return NULL;
    }
} /* vsi_nn_create_internal_node() */

static vsi_nn_internal_tensor_t * vsi_nn_create_internal_tensor
    (
    vsi_nn_graph_t       * graph,
    vsi_nn_tensor_attr_t * attr,
    float                  default_value
    )
{
    vsi_nn_internal_tensor_t* tensor = NULL;

    if( !graph || !attr )
    {
        return tensor;
    }

    tensor = (vsi_nn_internal_tensor_t *)malloc( sizeof(vsi_nn_internal_tensor_t) );
    if( tensor )
    {
        memset( tensor, 0x00, sizeof(vsi_nn_internal_tensor_t) );
        if( attr->is_const )
        {
            tensor->t = vsi_nn_CreateTensorWithDefault( graph, attr, default_value );
        }
        else
        {
            tensor->t = vsi_nn_CreateTensor( graph, attr );
        }

        if( !tensor->t )
        {
            vsi_nn_release_internal_tensor( &tensor );
        }
    }

    return tensor;
} /* vsi_nn_create_internal_tensor() */

/**********************************************************
* PUBLIC FUNCTIONS
**********************************************************/
vsi_status vsi_nn_init_internal_node_wksp
    (
    vsi_nn_node_t* node
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_internal_node_wksp_t* wksp = NULL;

    if( node->internal_node_wksp )
    {
        vsi_nn_deinit_internal_node_wksp( node );
    }

    wksp = (vsi_nn_internal_node_wksp_t *)malloc( sizeof( vsi_nn_internal_node_wksp_t ) );
    if( wksp )
    {
        memset( wksp, 0x00, sizeof( vsi_nn_internal_node_wksp_t ) );
        wksp->curr_node_uid = 1;

        node->internal_node_wksp = wksp;

        status = VSI_SUCCESS;
    }

    return status;
} /* vsi_nn_init_internal_node_wksp() */

vsi_status vsi_nn_deinit_internal_node_wksp
    (
    vsi_nn_node_t* node
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_internal_node_t* head = NULL;
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_internal_tensor_t* tensor_head = NULL;
    vsi_nn_internal_tensor_t* tensor_curr = NULL;

    if( node && node->internal_node_wksp )
    {
        head = WKSP(node)->nodes;
        while( NULL != head )
        {
            curr = (vsi_nn_internal_node_t *)vsi_nn_LinkListPopStart(
                (vsi_nn_link_list_t **)&head );
            vsi_nn_release_internal_node( &curr );
        }

        tensor_head = WKSP(node)->tensors;
        while( NULL != tensor_head )
        {
            tensor_curr = (vsi_nn_internal_tensor_t *)vsi_nn_LinkListPopStart(
                (vsi_nn_link_list_t **)&tensor_head );
            vsi_nn_release_internal_tensor( &tensor_curr );
        }

        free( node->internal_node_wksp );
        node->internal_node_wksp = NULL;
    }

    return status;
} /* vsi_nn_deinit_internal_node_wksp() */

vsi_nn_internal_node_t* vsi_nn_new_internal_node
    (
    vsi_nn_node_t* node,
    vsi_nn_op_t op,
    uint32_t input_num,
    uint32_t output_num
    )
{
    vsi_nn_internal_node_t* inode = NULL;

    inode = vsi_nn_create_internal_node( node->graph,
                op, input_num, output_num );
    return inode;
} /* vsi_nn_new_internal_node() */

void* vsi_nn_new_internal_node_param
    (
    vsi_nn_internal_node_t* inode,
    size_t size /* in bytes */
    )
{
    vsi_nn_internal_node_param_t* param = NULL;
    size_t buf_sz = sizeof(vsi_nn_internal_node_param_t) + size;
    void* ptr = NULL;
    if( !inode )
    {
        return ptr;
    }

    param = (vsi_nn_internal_node_param_t *)malloc(buf_sz);
    if( param )
    {
        memset(param, 0x00, buf_sz);
        ptr = (void *)(&param->param[0]);
        LINKLIST_APPEND(inode->param, param);
    }

    return ptr;
} /* vsi_nn_new_internal_node_param() */

vsi_nn_internal_tensor_t* vsi_nn_new_internal_tensor
    (
    vsi_nn_node_t*          node,
    vsi_nn_tensor_attr_t*   attr,
    float                   default_value
    )
{
    vsi_nn_internal_tensor_t* tensor = NULL;

    tensor = vsi_nn_create_internal_tensor( node->graph,
                attr, default_value );
    if( tensor )
    {
        LINKLIST_APPEND( WKSP(node)->tensors, tensor );
    }

    return tensor;
} /* vsi_nn_new_internal_tensor() */

vsi_bool vsi_nn_setup_internal_node_op
    (
    vsi_nn_node_t* node,
    vsi_nn_internal_node_t* inode
    )
{
    vsi_bool retn = TRUE;

    retn = vsi_nn_OpSetup( inode->node->op, inode->node, inode->inputs, inode->outputs );
    if( retn )
    {
        inode->node->uid = WKSP(node)->curr_node_uid;
        LINKLIST_APPEND( WKSP(node)->nodes, inode );
        WKSP(node)->curr_node_uid++;
    }

    return retn;
} /* vsi_nn_setup_internal_node_op() */

vsi_status vsi_nn_compute_internal_node
    (
    vsi_nn_node_t * node
    )
{
    vsi_status status =  VSI_SUCCESS;
    vsi_nn_internal_node_t* curr = NULL;
    uint32_t j = 0;

    curr = WKSP(node)->nodes;
    while( NULL != curr )
    {
        for ( j = 0; j < curr->node->output.num; j++ )
        {
            if( NULL == curr->outputs[j] || NULL != curr->outputs[j]->t )
                continue;
            vsi_nn_TensorReinit( node->graph, curr->outputs[j] );
        }

        VSILOGD("Compute node uid[%u] sub_uid[%u] op[%s]",
            node->uid, curr->node->uid, vsi_nn_OpGetName(curr->node->op));
        status = vsi_nn_OpCompute( curr->node->op, curr->node, curr->inputs, curr->outputs );
        if( VSI_SUCCESS != status )
        {
            VSILOGE("op_compute fail %d", curr->node->op);
            break;
        }

        curr = (vsi_nn_internal_node_t *)vsi_nn_LinkListNext( (vsi_nn_link_list_t *)curr );
    }

    return status;
} /* vsi_nn_compute_internal_node() */

vsi_status vsi_nn_optimize_internal_node
    (
    vsi_nn_node_t * node,
    vsi_nn_opt_direction_e direction
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_internal_node_t* curr = NULL;

    curr = WKSP(node)->nodes;
    while( NULL != curr )
    {
        VSILOGD("Optimize node uid[%u] sub_uid[%u] op[%s]",
            node->uid, curr->node->uid, vsi_nn_OpGetName(curr->node->op));

        status = vsi_nn_OpOptimize( curr->node->op, curr->node,
            curr->inputs, curr->outputs, direction );
        if( VSI_SUCCESS != status )
        {
            VSILOGE("op_optimize fail %d", curr->node->op);
            break;
        }

        curr = (vsi_nn_internal_node_t *)vsi_nn_LinkListNext( (vsi_nn_link_list_t *)curr );
    }

    return status;
} /* vsi_nn_optimize_internal_node() */

vsi_status vsi_nn_deinit_internal_node
    (
    vsi_nn_node_t * node
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_internal_node_t* curr = NULL;

    curr = WKSP(node)->nodes;
    while( NULL != curr )
    {
        VSILOGD("Optimize node uid[%u] sub_uid[%u] op[%s]",
            node->uid, curr->node->uid, vsi_nn_OpGetName(curr->node->op));

        status = vsi_nn_OpDeinit( curr->node->op, curr->node );
        if( VSI_SUCCESS != status )
        {
            VSILOGE("op_optimize fail %d", curr->node->op);
            break;
        }

        curr = (vsi_nn_internal_node_t *)vsi_nn_LinkListNext( (vsi_nn_link_list_t *)curr );
    }

    return status;
} /* vsi_nn_deinit_internal_node() */

vsi_status vsi_nn_release_internal_node
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
        if( ptr->param )
        {
            vsi_nn_LinkListDeinit((vsi_nn_link_list_t *)(ptr->param), NULL);
        }
        if( ptr->node )
        {
            vsi_nn_ReleaseNode( &ptr->node );
        }

        free( ptr );
        *node = NULL;
    }

    return VSI_SUCCESS;
} /* vsi_nn_release_internal_node() */

vsi_status vsi_nn_release_internal_tensor
    (
    vsi_nn_internal_tensor_t** tensor
    )
{
    if( tensor && *tensor )
    {
        vsi_nn_internal_tensor_t* ptr = *tensor;

        if( ptr->t )
        {
            vsi_nn_ReleaseTensor( &ptr->t );
        }
        free( ptr );
        *tensor = NULL;
    }

    return VSI_SUCCESS;
} /* vsi_nn_release_internal_tensor() */

vsi_nn_internal_tensor_t* vsi_nn_create_zero_bias_tensor
    (
    vsi_nn_node_t * node,
    vsi_nn_tensor_attr_t* input_attr,
    vsi_nn_tensor_attr_t* weight_attr
    )
{
    vsi_nn_tensor_attr_t attr;
    float scale = 1.0f;
    int8_t fl = 0;

    memset(&attr, 0x0, sizeof(vsi_nn_tensor_attr_t));

    /* create zero bias for NN/TP */
    attr.size[0] = weight_attr->size[1];
    attr.dim_num = 1;
    attr.vtl = FALSE;
    attr.is_const = TRUE;
    if (input_attr->dtype.qnt_type == VSI_NN_QNT_TYPE_NONE)
    {
        attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    }
    else
    {
        attr.dtype.vx_type = VSI_NN_TYPE_INT32;
    }

    switch(input_attr->dtype.qnt_type)
    {
        case VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC:
            scale = input_attr->dtype.scale;
            break;

        case VSI_NN_QNT_TYPE_DFP:
            fl = input_attr->dtype.fl;
            break;

        case VSI_NN_QNT_TYPE_NONE:
            scale = 1.0f;
            fl = 0;
            break;

        default:
            VSILOGE("Unsupported quantization type: %d", input_attr->dtype.qnt_type);
            break;
    }

    switch(weight_attr->dtype.qnt_type)
    {
        case VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC:
            attr.dtype.scale = weight_attr->dtype.scale * scale;
            attr.dtype.zero_point = 0;
            attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
            break;

        case VSI_NN_QNT_TYPE_DFP:
            attr.dtype.fl = weight_attr->dtype.fl + fl;
            attr.dtype.qnt_type = VSI_NN_QNT_TYPE_DFP;
            break;

        case VSI_NN_QNT_TYPE_NONE:
            break;

        default:
            VSILOGE("Unsupported quantization type: %d", weight_attr->dtype.qnt_type);
            break;
    }

    return vsi_nn_new_internal_tensor(node, &attr, 0.0f);
}

void vsi_nn_internal_node_init_attr
    (
    vsi_nn_tensor_attr_t* attr,
    const vsi_nn_dtype_t* dtype,
    vsi_bool use_virtual_tensor
    )
{
    memset(attr, 0x00, sizeof(vsi_nn_tensor_attr_t));

    //memset(attr->size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
    attr->dim_num = VSI_NN_DIM_AUTO;
    attr->vtl = use_virtual_tensor;
    attr->is_const = FALSE;

    if( dtype->qnt_type == VSI_NN_QNT_TYPE_NONE )
    {
        attr->dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        attr->dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    }
    else
    {
        memcpy(&attr->dtype, dtype, sizeof(vsi_nn_dtype_t));
    }
}
