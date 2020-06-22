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
#include "utils/vsi_nn_math.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "client/vsi_nn_vxkernel.h"
#include "kernel/vsi_nn_kernel.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"

#define _INPUT_NUM          (2)
#define _OUTPUT_NUM         (1)

#define VSI_NN_L2NORMALIZESCALE_DEFAULT_AXIS 2

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status      = VSI_FAILURE;
    int32_t    axis        = 0;
    int32_t    new_axis    = 0;
    uint32_t   axis_size   = 0;
    uint32_t   rank_in     = 0;
    uint32_t   rank_out    = 0;
    uint32_t   size   = 1;
    uint32_t   i      = 0;
    int32_t shapes[3][VSI_NN_MAX_DIM_NUM] = {{ 1 }};
    vsi_nn_tensor_t* reshape_tensors[3] = { NULL };
    vsi_nn_l2normalizescale_param * p = NULL;
    vsi_bool ret = FALSE;
    vsi_nn_kernel_param_t * param = NULL;

    p = &(self->nn_param.l2normalizescale);
    axis = p->axis;

    param =vsi_nn_kernel_param_create();

    ret = vsi_nn_kernel_optimize_reduce_shape(
            (int32_t *)inputs[0]->attr.size, inputs[0]->attr.dim_num,
            &axis, 1,
            (int32_t *)outputs[0]->attr.size, outputs[0]->attr.dim_num,
            shapes[0], &rank_in, shapes[2], &rank_out,
            &new_axis, &axis_size);
    size = inputs[1]->attr.size[0];
    for (i = 1; i < inputs[1]->attr.dim_num; i ++)
    {
        size *= inputs[1]->attr.size[i];
    }
    shapes[1][0] = (int32_t)size;
    shapes[1][1] = 1;
    shapes[1][2] = 1;
    shapes[1][3] = 1;

    vsi_nn_kernel_param_add_int32( param, "axis",  new_axis );

    if( ret )
    {
        reshape_tensors[0] = vsi_nn_reshape_tensor( self->graph,
                inputs[0], (uint32_t*)shapes[0], rank_in );
        reshape_tensors[1] = vsi_nn_reshape_tensor( self->graph,
                inputs[1], (uint32_t*)shapes[1], 2 );
        reshape_tensors[2] = vsi_nn_reshape_tensor( self->graph,
                outputs[0], (uint32_t*)shapes[0], rank_in );

        self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
                "l2normalizescale",
                &reshape_tensors[0], _INPUT_NUM,
                &reshape_tensors[2], _OUTPUT_NUM, param );

        vsi_nn_ReleaseTensor( &reshape_tensors[0] );
        vsi_nn_ReleaseTensor( &reshape_tensors[1] );
        vsi_nn_ReleaseTensor( &reshape_tensors[2] );
    }
    if( self->n )
    {
        status = VSI_SUCCESS;
    }

    vsi_nn_kernel_param_release( &param );

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

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    uint32_t i;
    for (i = 0; i < _VSI_NN_L2NORMALIZESCALE_LOCAL_TENSOR_NUM; i++)
    {
        if (self->nn_param.l2normalizescale.local.local_tensor[i] != NULL)
        {
            vxReleaseTensor(&(self->nn_param.l2normalizescale.local.local_tensor[i]));
            self->nn_param.l2normalizescale.local.local_tensor[i] = NULL;
        }
    }
    vsi_nn_op_common_deinit(self);

    return VSI_SUCCESS;
} /* op_deinit() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = TRUE;
    if( NULL == self )
    {
        return FALSE;
    }

    if (self->nn_param.l2normalizescale.axis < 0)
    {
        self->nn_param.l2normalizescale.axis += (int32_t)inputs[0]->attr.dim_num;
    }

    if (self->nn_param.l2normalizescale.axis < 0)
    {
        VSILOGD("l2normalizescale Invalid Axis: %d", self->nn_param.l2normalizescale.axis);
        return FALSE;
    }

    ret = vsi_nn_op_common_setup(self, inputs, outputs);

    return ret;
}

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;
    uint32_t  i;

    if (vsi_nn_compareVersion(self->graph, 1, 1, 13) == -1)
    {
        self->nn_param.l2normalizescale.axis = VSI_NN_L2NORMALIZESCALE_DEFAULT_AXIS;
    }
    for (i = 0; i < _VSI_NN_L2NORMALIZESCALE_LOCAL_TENSOR_NUM; i++)
    {
        self->nn_param.l2normalizescale.local.local_tensor[i] = NULL;
    }

    return status;
} /* op_init() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ L2NORMALIZESCALE,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 2,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif

