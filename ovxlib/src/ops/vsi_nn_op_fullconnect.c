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

#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "utils/vsi_nn_math.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_log.h"
#include "utils/vsi_nn_dtype_util.h"

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status;
    status = VSI_FAILURE;

    self->n = vxFullyConnectedLayer(
        self->graph->g,
        inputs[0]->t,
        inputs[1]->t,
        inputs[2]->t,
        0,
        0,
        self->vx_param.overflow_policy,
        self->vx_param.rounding_policy,
        self->vx_param.down_scale_size_rounding,
        outputs[0]->t
        );

    if( NULL != self->n )
    {
        status = VSI_SUCCESS;
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
    vsi_bool ret = FALSE;

    /* Check fl and scale*/
    ret = vsi_nn_QuantCheck(inputs[0], inputs[1], inputs[2]);

    return ret;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    uint32_t dim_num;
    uint32_t perm[4] = { 0 };
    uint32_t as_shape[4] = { 0 };

    /* TODO: Driver should handle this,
    * Check transpose
    * */
    if( VSI_NN_DIM_FMT_NHWC == inputs[1]->attr.dtype.fmt &&
        VSI_NN_TYPE_VDATA != inputs[1]->attr.dtype.vx_type )
    {
        /* TODO: This is used to handle the first fcl. */
        if( 1 != inputs[0]->attr.size[0] || 1 != inputs[0]->attr.size[1] )
        {
            dim_num = 4;
            perm[0] = 3;
            perm[1] = 2;
            perm[2] = 0;
            perm[3] = 1;
            as_shape[0] = inputs[0]->attr.size[0];
            as_shape[1] = inputs[0]->attr.size[1];
            as_shape[2] = inputs[0]->attr.size[2];
            as_shape[3] = inputs[1]->attr.size[3];
        }
        else
        {
            dim_num = 2;
            perm[0] = 1;
            perm[1] = 0;
            as_shape[0] = vsi_nn_ShapeProduct( inputs[0]->attr.size,
                inputs[0]->attr.dim_num );
            as_shape[1] = inputs[1]->attr.size[3];
        }
        vsi_nn_TransposeTensor( self->graph, inputs[1], perm, dim_num, as_shape );
        inputs[1]->attr.dtype.fmt = VSI_NN_DIM_FMT_NCHW;
    }

    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        uint32_t input_dim = inputs[0]->attr.dim_num;
        switch (input_dim)
        {
        // add a workaround to handle fc layer's output
        case 1:
        case 3:
            outputs[0]->attr.dim_num = 1;
            outputs[0]->attr.size[0] = self->nn_param.fcl.weights;
            break;
        case 2:
        case 4:
            outputs[0]->attr.dim_num = 2;
            outputs[0]->attr.size[0] = self->nn_param.fcl.weights;
            outputs[0]->attr.size[1] = inputs[0]->attr.size[input_dim-1];
            break;
        default:
            VSILOGE("input dim[%u] error\n", inputs[0]->attr.dim_num);
            return FALSE;
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
    /* op_name    */ FCL,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 3,
    /* output_num */ 1
    );
#ifdef __cpluplus
}
#endif

