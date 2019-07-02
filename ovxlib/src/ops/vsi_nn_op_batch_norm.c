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
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_log.h"

static vsi_status _try_set_high_presision_tensor
    (
    vsi_nn_tensor_t **inputs
    )
{
    vsi_status status;
    vsi_nn_vxtensor_attr_t attr;

    status = VSI_SUCCESS;
    attr = VSI_NN_TENSOR_ATTR_HIGH_PRECISION;

    if(VSI_NN_TYPE_FLOAT32 == inputs[1]->attr.dtype.vx_type)
    {
        status = vsi_nn_SetTensorAttr(inputs[1], attr);
        if(VSI_SUCCESS != status)
        {
            return status;
        }
    }
    if(VSI_NN_TYPE_FLOAT32 == inputs[2]->attr.dtype.vx_type)
    {
        status = vsi_nn_SetTensorAttr(inputs[2], attr);
        if(VSI_SUCCESS != status)
        {
            return status;
        }
    }
    if(VSI_NN_TYPE_FLOAT32 == inputs[3]->attr.dtype.vx_type)
    {
        status = vsi_nn_SetTensorAttr(inputs[3], attr);
        if(VSI_SUCCESS != status)
        {
            return status;
        }
    }
    if(VSI_NN_TYPE_FLOAT32 == inputs[4]->attr.dtype.vx_type)
    {
        status = vsi_nn_SetTensorAttr(inputs[4], attr);
        if(VSI_SUCCESS != status)
        {
            return status;
        }
    }

    return status;
}

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status         status;
    status = VSI_FAILURE;

    status = _try_set_high_presision_tensor(inputs);
    if(VSI_SUCCESS != status)
    {
        VSILOGE("Set tensor attr of high presision fail");
        return status;
    }

    self->n = vxBatchNormalizationLayer(
        self->graph->g,
        self->nn_param.batch_norm.eps,
        inputs[1]->t,
        inputs[2]->t,
        inputs[3]->t,
        inputs[4]->t,
        inputs[0]->t,
        outputs[0]->t
        );
    if( NULL == self->n )
    {
        status = VSI_FAILURE;
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
    //TODO: Check tensor shapes.
    return TRUE;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
        outputs[0]->attr.size[0] = inputs[0]->attr.size[0];
        outputs[0]->attr.size[1] = inputs[0]->attr.size[1];
        outputs[0]->attr.size[2] = inputs[0]->attr.size[2];
        outputs[0]->attr.size[3] = inputs[0]->attr.size[3];
    }
    return TRUE;
} /* op_setup() */


#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ BATCH_NORM,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 5,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif

