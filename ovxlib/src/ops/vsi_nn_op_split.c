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
#include <string.h>
#include <stdlib.h>

#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_link_list.h"

static vsi_status copy_view_to_tensor
    (
    vsi_nn_node_t   * self,
    vx_tensor         src_tensor,
    uint32_t         axis,
    vsi_nn_tensor_t * dst_out
    )
{
    vsi_status ret;
    vsi_nn_split_lcl_data * data;

    ret = VSI_SUCCESS;
    /* Malloc ptr */
    data = (vsi_nn_split_lcl_data *)malloc( sizeof(vsi_nn_split_lcl_data) );
    if( NULL == data )
    {
        VSILOGE( "Create split local data fail." );
        return VSI_FAILURE;
    }
    memset( data, 0, sizeof(vsi_nn_split_lcl_data) );
    data->src_tensor = src_tensor;
    data->dst_tensor = dst_out->t;

    /* Store node, ptr */
    vsi_nn_LinkListPushStart(
        (vsi_nn_link_list_t **)&self->nn_param.split.lcl_data,
        (vsi_nn_link_list_t *)data );

    return ret;
} /* copy_view_to_tensor() */

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status;
    vsi_nn_split_lcl_data * iter;

    status = VSI_SUCCESS;
    iter = self->nn_param.split.lcl_data;
    while( NULL != iter )
    {
        iter->cp_node = vxTensorCopyNode(self->graph->g,
            iter->src_tensor, iter->dst_tensor );
        if( NULL == iter->cp_node )
        {
            VSILOGE( "Create TensorCopy fail." );
            status = VSI_FAILURE;
            break;
        }
        iter = (vsi_nn_split_lcl_data *)vsi_nn_LinkListNext( (vsi_nn_link_list_t *)iter );
    }
    return status;
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret;
    uint32_t num,i,j;
    uint32_t slices_num = self->nn_param.split.slices_num;
    uint32_t axis = self->nn_param.split.axis;

    /* compute the output tensor number */
    num = (uint32_t)(self->output.num - 1);
    while( num >= 0 && NULL == outputs[num] )
    {
        num --;
    }
    if( 0 > num )
    {
        return FALSE;
    }
    num++;

    ret = TRUE;
    /* 1. check the input tensor number */
    if(self->input.num != 1)
    {
        VSILOGE("The split layer input num must be 1, here is %u\n", self->input.num);
        return FALSE;
    }

    /* 2. check output tensor number */
    if(slices_num == 0)
    {
        uint32_t remaind = inputs[0]->attr.size[axis] % num;
        if(remaind != 0)
        {
            VSILOGE("Can not average the input tensor %u shape\n", axis);
            return FALSE;
        }
    }
    else if(slices_num != num)
    {
        VSILOGE( "slices num %u != output tensor num %u\n", slices_num, num);
        return FALSE;
    }

    /* 3. check output tensor shape and dimensions */
    for( i = 0; i < num; i ++ )
    {
        /* the virtual tensor shape has not been calculated yet */
        if(outputs[i]->attr.vtl == TRUE
            || outputs[i]->attr.dim_num == VSI_NN_DIM_AUTO)
            continue;

        if( outputs[i]->attr.dim_num != inputs[0]->attr.dim_num )
        {
            VSILOGE( "Split dims num(%d vs %d)",
                outputs[i]->attr.dim_num,
                inputs[0]->attr.dim_num);
            ret = FALSE;
            break;
        }

        for( j = 0; j < outputs[i]->attr.dim_num; j ++ )
        {
            if( axis == j )
            {
                continue;
            }

            if( outputs[i]->attr.size[j] != inputs[0]->attr.size[j] )
            {
                VSILOGE( "Split dims size(%d vs %d)",
                    outputs[i]->attr.size[j],
                    inputs[0]->attr.size[j]);
                ret = FALSE;
                break;
            }
        }

        if( FALSE == ret )
        {
            break;
        }
    }

    return ret;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret;
    uint32_t i,num,average;
    uint32_t start[VSI_NN_MAX_DIM_NUM] = { 0 };
    uint32_t end[VSI_NN_MAX_DIM_NUM] = { 0 };
    uint32_t axis = self->nn_param.split.axis;
    uint32_t *slices = self->nn_param.split.slices;
    uint32_t slices_num = self->nn_param.split.slices_num;

    ret = TRUE;
    average = 1;
    /* compute the output tensor number */
    num = (uint32_t)(self->output.num - 1);
    while( num >= 0 && NULL == outputs[num] )
    {
        num --;
    }
    if( 0 > num )
    {
        return FALSE;
    }
    num++;

    if(slices_num == 0)
    {
        average = inputs[0]->attr.size[axis] / num;
    }

    end[0] = inputs[0]->attr.size[0];
    end[1] = inputs[0]->attr.size[1];
    end[2] = inputs[0]->attr.size[2];
    end[3] = inputs[0]->attr.size[3];
    end[axis] = 0;
    for(i = 0; i < num; i++)
    {
        start[axis] = end[axis];
        if(slices_num == 0)
            end[axis] += average;
        else
            end[axis] += slices[i];

        memcpy(&outputs[i]->attr.dtype, &inputs[0]->attr.dtype, sizeof(vsi_nn_dtype_t));
        outputs[i]->attr.dim_num = inputs[0]->attr.dim_num;
        outputs[i]->attr.size[0] = inputs[0]->attr.size[0];
        outputs[i]->attr.size[1] = inputs[0]->attr.size[1];
        outputs[i]->attr.size[2] = inputs[0]->attr.size[2];
        outputs[i]->attr.size[3] = inputs[0]->attr.size[3];
        outputs[i]->attr.size[axis] = end[axis] - start[axis];
    }

    return ret;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_nn_split_lcl_data * data;
    vsi_nn_split_lcl_data * tmp;
    data = self->nn_param.split.lcl_data;
    while( NULL != data )
    {
        tmp = (vsi_nn_split_lcl_data *)vsi_nn_LinkListPopStart(
            (vsi_nn_link_list_t **)&data );
        vxReleaseNode( &tmp->cp_node );
        vxReleaseTensor( &tmp->src_tensor );
        free( tmp );
    }
    return VSI_SUCCESS;
} /* op_deinit() */

static vsi_status op_optimize
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_opt_direction_e direction
    )
{
    vsi_status status;
    uint32_t i,num;
    vx_tensor in_view_tensor;
    uint32_t start[VSI_NN_MAX_DIM_NUM] = { 0 };
    uint32_t end[VSI_NN_MAX_DIM_NUM] = { 0 };
    uint32_t axis = self->nn_param.split.axis;

    status = VSI_SUCCESS;
    /* Only forward run split's optimize */
    if( direction == VSI_NN_OPTIMIZE_BACKWARD )
    {
        return VSI_SUCCESS;
    }

    VSILOGD("Optimize %s, uid %u", vsi_nn_OpGetName(self->op), self->uid);
    num = (uint32_t)(self->output.num - 1);
    while( num >= 0 && NULL == outputs[num] )
    {
        num --;
    }
    if( 0 > num )
    {
        return VSI_FAILURE;
    }
    num++;

    if( NULL == inputs[0]->t )
    {
        vsi_nn_TensorReinit( self->graph, inputs[0] );
    }

    end[0] = inputs[0]->attr.size[0];
    end[1] = inputs[0]->attr.size[1];
    end[2] = inputs[0]->attr.size[2];
    end[3] = inputs[0]->attr.size[3];
    end[axis] = 0;
    for(i = 0; i < num; i++)
    {
        start[axis] = end[axis];
        end[axis] += outputs[i]->attr.size[axis];

        in_view_tensor = vsi_nn_CreateViewTensor(self->graph, start, end, inputs[0] );
        if( NULL == in_view_tensor )
        {
            VSILOGE( "Create tensor %d from view fail.", i );
            status = VSI_FAILURE;
            break;
        }

        if( NULL != outputs[i]->t )
        {
            VSILOGW( "Split copy %d tensor.", i );
            // Copy old tensor values to the new address.
            status = copy_view_to_tensor( self, in_view_tensor, axis, outputs[i] );
            if( VSI_FAILURE == status )
            {
                break;
            }
        }
        else
        {
            outputs[i]->t = in_view_tensor;
        }
    }

    return status;
}

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ SPLIT,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ 1,
    /* output_num */ 16
    );
#ifdef __cplusplus
}
#endif

