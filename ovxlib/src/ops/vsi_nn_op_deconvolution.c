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
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "vsi_nn_log.h"

#define COMPUTE_DECONV_SZ( in, ksize, pad_1, pad_2, stride )\
    (( in - 1 ) * stride + ksize - pad_1 - pad_2)
static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status;
    vx_nn_deconvolution_params_ext2_t param;

    status = VSI_FAILURE;

    // param.a_x = self->nn_param.deconv.dilation;
    // param.a_y = self->nn_param.deconv.dilation;
    param.ext.khr.a_x = 1;
    param.ext.khr.a_y = 1;
    param.ext.khr.padding_x = (size_t)self->nn_param.deconv.pad[0];
    param.ext.khr.padding_y = (size_t)self->nn_param.deconv.pad[2];
    param.ext.khr.overflow_policy = self->vx_param.overflow_policy;
    param.ext.khr.rounding_policy = self->vx_param.rounding_policy;
    param.ext.padding_x_right = (size_t)self->nn_param.deconv.pad[1];
    param.ext.padding_y_bottom = (size_t)self->nn_param.deconv.pad[3];
    param.ext.channel_group = self->nn_param.deconv.group;
    param.stride_x = self->nn_param.deconv.stride[0];
    param.stride_y = self->nn_param.deconv.stride[1];
    //param.border_mode;
    //param.border_const;

    self->n = vxDeconvolutionLayer(
        self->graph->g,
        inputs[0]->t,
        inputs[1]->t,
        (NULL == inputs[2]) ? NULL : inputs[2]->t,
        (vx_nn_deconvolution_params_t *)&param,
        sizeof( vx_nn_deconvolution_params_ext2_t ),
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
    vsi_nn_deconv_param *nn_param;
    uint32_t perm[] = { 3, 2, 0, 1 };

    /* TODO: Driver should handle this,
    * Check transpose
    * */
    if( VSI_NN_DIM_FMT_NHWC == inputs[1]->attr.dtype.fmt )
    {
        vsi_nn_TransposeTensor( self->graph, inputs[1], perm, 4, NULL );
        inputs[1]->attr.dtype.fmt = VSI_NN_DIM_FMT_NCHW;
    }

    nn_param = &self->nn_param.deconv;

    nn_param->group = ( 0 == nn_param->group ) ? 1 : nn_param->group;

    nn_param->ksize[0] = inputs[1]->attr.size[0];
    nn_param->ksize[1] = inputs[1]->attr.size[1];


    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.size[0] = COMPUTE_DECONV_SZ(
            inputs[0]->attr.size[0],
            nn_param->ksize[0],
            nn_param->pad[0],
            nn_param->pad[1],
            nn_param->stride[0]
        );

        outputs[0]->attr.size[1] = COMPUTE_DECONV_SZ(
            inputs[0]->attr.size[1],
            nn_param->ksize[1],
            nn_param->pad[2],
            nn_param->pad[3],
            nn_param->stride[1]
        );

        outputs[0]->attr.size[2] = nn_param->weights;
        outputs[0]->attr.size[3] = inputs[0]->attr.size[3];
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
    }
    return TRUE;
} /* op_setup() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ DECONVOLUTION,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 3,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif

