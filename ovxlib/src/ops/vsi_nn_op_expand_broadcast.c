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
#include <string.h>
#include <stdlib.h>
#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_log.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_internal_node.h"
#include "utils/vsi_nn_dtype_util_prv.h"
#include "utils/vsi_nn_math.h"

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    return vsi_nn_internal_compute_node( self );;
}

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
    uint32_t i;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t *input_1;
    vsi_nn_internal_node_t* mul_node = NULL;
    vsi_nn_expand_broadcast_param *p = &self->nn_param.expand_broadcast;

    vsi_nn_internal_init_node_wksp(self);

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    attr.dim_num = p->dim_num;
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    attr.is_const = TRUE;
    for(i = 0; i < p->dim_num; i++)
    {
        attr.size[i] = p->shape[i];
    }
    input_1 = vsi_nn_internal_new_tensor( self, &attr, 1.0f );

    mul_node = vsi_nn_internal_new_node(self, VSI_NN_OP_MULTIPLY, 0, 0 );
    mul_node->inputs[0] = inputs[0];
    mul_node->inputs[1] = input_1->t;
    mul_node->node->nn_param.multiply.scale = 1.0f;
    mul_node->node->vx_param.overflow_policy = VX_CONVERT_POLICY_SATURATE;
    mul_node->node->vx_param.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
    mul_node->outputs[0] = outputs[0];
    vsi_nn_internal_setup_node(self, mul_node);

    return TRUE;
}

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    vsi_nn_internal_deinit_node_wksp( self );
    return status;
}

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ EXPAND_BROADCAST,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 1,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif
