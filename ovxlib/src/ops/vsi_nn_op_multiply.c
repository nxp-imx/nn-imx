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
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_math.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_link_list.h"

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status;
    float scale;
    vsi_enum overflow_policy,rounding_policy;
    vx_scalar scale_s;

    status = VSI_FAILURE;
    scale = self->nn_param.multiply.scale;
    overflow_policy = self->vx_param.overflow_policy;
    rounding_policy = self->vx_param.rounding_policy;

    scale_s = vxCreateScalar(self->graph->ctx->c, VX_TYPE_FLOAT32, &scale);
    if(!scale_s)
    {
        VSILOGE("CreateScalar fail\n");
        return VSI_FAILURE;
    }

    self->n = vxTensorMultiplyNode( self->graph->g,
        inputs[0]->t, inputs[1]->t,
        scale_s,
        overflow_policy,
        rounding_policy,
        outputs[0]->t );
    if( NULL != self->n )
    {
        status = VSI_SUCCESS;
    }
    vxReleaseScalar(&scale_s);

    return status;
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    uint32_t i,dim0,dim1,s0,s1;

    dim0 = inputs[0]->attr.dim_num;
    dim1 = inputs[1]->attr.dim_num;
    for(i = 0; i < vsi_nn_max(dim0,dim1); i++)
    {
        if(i < dim0 && i < dim1)
        {
            s0 = inputs[0]->attr.size[i];
            s1 = inputs[1]->attr.size[i];
            if(s0 > s1 && s1 != 1)
            {
                VSILOGE("Invalid broadcast for inputs[1] size[%u]", s1);
                return FALSE;
            }
            else if(s0 < s1 && s0 != 1)
            {
                VSILOGE("Invalid broadcast for inputs[0] size[%u]", s0);
                return FALSE;
            }
        }
    }

    return TRUE;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    uint32_t i,dim0,dim1, dim;

    if(VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num)
    {
        dim0 = inputs[0]->attr.dim_num;
        dim1 = inputs[1]->attr.dim_num;
        dim = vsi_nn_max(dim0, dim1);

        outputs[0]->attr.dim_num = dim;
        for(i = 0; i < dim; i++)
        {
            uint32_t sz0, sz1;
            sz0 = i < dim0 ? inputs[0]->attr.size[i] : 1;
            sz1 = i < dim1 ? inputs[1]->attr.size[i] : 1;
            outputs[0]->attr.size[i] = vsi_nn_max(sz0, sz1);
        }
    }

    return TRUE;
} /* op_setup() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ MULTIPLY,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 2,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif
