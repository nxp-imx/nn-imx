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
    return TRUE;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    uint32_t sz0,sz1;

    if(VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num)
    {
        sz0 = vsi_nn_GetElementNum(inputs[0]);
        sz1 = vsi_nn_GetElementNum(inputs[1]);
        if(sz0 > sz1)
        {
            outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
            memcpy( outputs[0]->attr.size, inputs[0]->attr.size,
                VSI_NN_MAX_DIM_NUM * sizeof( uint32_t ) );
        }
        else
        {
            outputs[0]->attr.dim_num = inputs[1]->attr.dim_num;
            memcpy( outputs[0]->attr.size, inputs[1]->attr.size,
                VSI_NN_MAX_DIM_NUM * sizeof( uint32_t ) );
        }
    }

    return TRUE;
} /* op_setup() */

#ifdef __cpluplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ MULTIPLY,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 2,
    /* output_num */ 1
    );
#ifdef __cpluplus
}
#endif
