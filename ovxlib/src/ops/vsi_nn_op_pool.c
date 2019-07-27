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

#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"


static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status;
    vx_nn_pooling_params_t params;
    status = VSI_FAILURE;

    memset( &params, 0, sizeof( params ) );
    params.pool_type = self->nn_param.pool.type;
    params.pool_size_x = self->nn_param.pool.ksize[0];
    params.pool_size_y = self->nn_param.pool.ksize[1];
    params.pool_pad_x_left = self->nn_param.pool.pad[0];
    params.pool_pad_x_right = self->nn_param.pool.pad[1];
    params.pool_pad_y_top = self->nn_param.pool.pad[2];
    params.pool_pad_y_bottom = self->nn_param.pool.pad[3];
    params.rounding = self->vx_param.down_scale_size_rounding;
    if( NULL == outputs[1] )
    {
        self->n = vxPoolingLayer2(
            self->graph->g,
            inputs[0]->t,
            &params,
            sizeof( params ),
            outputs[0]->t
            );
    }
    else
    {
        // TODO:
    }

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
    vsi_bool ret;

    ret = TRUE;
    vsi_nn_compute_padding(
        inputs[0]->attr.size,
        self->nn_param.pool.ksize,
        self->nn_param.pool.stride,
        NULL,
        self->nn_param.pool.pad_type,
        self->nn_param.pool.pad
    );

    /* Pooling */
    outputs[0]->attr.size[0] = vsi_nn_ComputeFilterSize
        (
        inputs[0]->attr.size[0],
        self->nn_param.pool.ksize[0],
        &self->nn_param.pool.pad[0],
        self->nn_param.pool.stride[0],
        0,
        self->nn_param.pool.round_type
        );
    outputs[0]->attr.size[1] = vsi_nn_ComputeFilterSize
        (
        inputs[0]->attr.size[1],
        self->nn_param.pool.ksize[1],
        &self->nn_param.pool.pad[2],
        self->nn_param.pool.stride[1],
        0,
        self->nn_param.pool.round_type
        );

    outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
    outputs[0]->attr.size[2] = inputs[0]->attr.size[2];
    outputs[0]->attr.size[3] = inputs[0]->attr.size[3];
    if( NULL != outputs[1] )
    {
        outputs[1]->attr.dim_num = outputs[0]->attr.dim_num;
        memcpy( outputs[1]->attr.size, outputs[0]->attr.size,
            VSI_NN_MAX_DIM_NUM * sizeof( uint32_t ) );
    }

    return ret;
} /* op_setup() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ POOL,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 1,
    /* output_num */ 2
    );
#ifdef __cplusplus
}
#endif

