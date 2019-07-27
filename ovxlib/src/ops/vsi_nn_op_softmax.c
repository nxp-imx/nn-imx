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


#include <string.h>
#include <stdlib.h>

#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_log.h"
#include "utils/vsi_nn_util.h"
#include "vsi_nn_internal_node.h"
#include "utils/vsi_nn_math.h"

static vsi_bool _is_same_shape
    (
    vsi_nn_tensor_t * inputs,
    uint32_t *sizes,
    uint32_t dims
    )
{
    uint32_t i = 0;

    if (inputs->attr.dim_num != dims)
        return FALSE;

    for (i = 0; i < dims; i++)
    {
        if (sizes[i] != inputs->attr.size[i])
            return FALSE;
    }

    return TRUE;
}

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    return vsi_nn_compute_internal_node( self );
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    /*TODO: Check tensor shapes. */
    return TRUE;
} /* op_check() */

static vsi_status op_optimize
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_opt_direction_e direction
    )
{
    return vsi_nn_optimize_internal_node( self, direction );
} /* op_optimize() */


static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    vsi_nn_deinit_internal_node_wksp( self );

    return status;
} /* op_deinit() */

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;
    uint32_t graph_version_major = 0;
    uint32_t graph_version_minor = 0;
    uint32_t graph_version_patch = 0;

    vsi_nn_GetGraphVersion( self->graph, &graph_version_major, &graph_version_minor, &graph_version_patch );
    if (!( graph_version_major >= 1 && graph_version_minor >= 1 && graph_version_patch >= 7 ))
    {
        self->nn_param.softmax.axis = -1;
    }
    if (self->nn_param.softmax.beta == 0.f)
    {
        self->nn_param.softmax.beta = 1.f;
    }

    return status;
} /* op_init() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_softmax_param * p;
    uint32_t dim_num;
    uint32_t sizes[VSI_NN_MAX_DIM_NUM];
    vsi_nn_tensor_attr_t attr;
    uint32_t i = 0;
    int32_t axis = -1;
    vsi_nn_internal_tensor_t* output0_tensor = NULL;
    vsi_nn_internal_tensor_t* output1_tensor = NULL;
    vsi_bool use_virtual_tensor = TRUE;

    if( NULL == self )
    {
        return FALSE;
    }

    p = &(self->nn_param.softmax);
    axis = p->axis;

    vsi_nn_init_internal_node_wksp( self );

    if (axis != -1)
    {
        uint32_t outerSize = 1;

        for (i = 0; i < (uint32_t)axis; i++)
        {
            sizes[i] = inputs[0]->attr.size[i];
        }

        for (i = (uint32_t)(axis + 1); i < inputs[0]->attr.dim_num; i++)
        {
            outerSize *= inputs[0]->attr.size[i];
        }

        if (axis == 1)
        {
            sizes[axis] = 1;
            sizes[axis + 1] = inputs[0]->attr.size[axis];
            sizes[axis + 2] = outerSize;

            dim_num = vsi_nn_min((uint32_t)(axis + 3), inputs[0]->attr.dim_num);
        }
        else
        {
            sizes[axis] = inputs[0]->attr.size[axis];
            sizes[axis + 1] = outerSize;

            dim_num = vsi_nn_min((uint32_t)(axis + 2), inputs[0]->attr.dim_num);
        }
    }

    if (axis == -1 || _is_same_shape(inputs[0], sizes, dim_num))
    {
        curr = vsi_nn_new_internal_node( self, VSI_NN_OP_SOFTMAX_INTERNAL, 0, 0 );
        curr->inputs[0] = inputs[0];
        curr->outputs[0] = outputs[0];

        vsi_nn_setup_internal_node_op(self, curr);
    }
    else
    {

        if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
        {
            outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;

            for (i = 0; i < inputs[0]->attr.dim_num; i++)
            {
                outputs[0]->attr.size[i] = inputs[0]->attr.size[i];
            }
        }

        memcpy(&attr, &(inputs[0]->attr), sizeof(vsi_nn_tensor_attr_t));
        memcpy(&attr.size, sizes, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
        attr.dim_num = dim_num;
        attr.vtl = use_virtual_tensor;
        attr.is_const = FALSE;
        output0_tensor = vsi_nn_new_internal_tensor( self, &attr, 0.0f );
        vsi_nn_ReshapeTensor(self->graph, inputs[0], output0_tensor->t, attr.size, attr.dim_num);

        memcpy(&attr, &(outputs[0]->attr), sizeof(vsi_nn_tensor_attr_t));
        memcpy(&attr.size, sizes, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
        attr.dim_num = dim_num;
        attr.vtl = use_virtual_tensor;
        attr.is_const = FALSE;
        output1_tensor = vsi_nn_new_internal_tensor( self, &attr, 0.0f );
        vsi_nn_ReshapeTensor(self->graph, outputs[0], output1_tensor->t, attr.size, attr.dim_num);

        curr = vsi_nn_new_internal_node( self, VSI_NN_OP_SOFTMAX_INTERNAL, 0, 0 );
        curr->inputs[0] = output0_tensor->t;
        curr->outputs[0] = output1_tensor->t;

        vsi_nn_setup_internal_node_op( self, curr );
    }
    curr->node->nn_param.softmax_internal.beta = self->nn_param.softmax.beta;

    return TRUE;
}

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ SOFTMAX,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ 1,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif

