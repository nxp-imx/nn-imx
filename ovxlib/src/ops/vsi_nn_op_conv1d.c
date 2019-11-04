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
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"

static void _reshape_tensor
    (
    vsi_nn_tensor_t * input,
    vx_tensor * output
    )
{
    vsi_nn_tensor_attr_t attr;
    memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));

    attr.size[0] = 1;
    attr.size[1] = input->attr.size[0];
    attr.size[2] = input->attr.size[1];
    attr.size[3] = input->attr.size[2];
    attr.dim_num = 4;
    *output = vxReshapeTensor( input->t, (int32_t *)attr.size, attr.dim_num );
}

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status;
    vx_nn_convolution_params_ext_t p;
    vsi_nn_conv1d_param *nn_param = &self->nn_param.conv1d;

    memset( &p, 0, sizeof( vx_nn_convolution_params_ext_t ) );

    status = VSI_FAILURE;

    p.khr.padding_y = self->nn_param.conv1d.pad[0];
    if (self->nn_param.conv1d.dilation > 0)
    {
        p.khr.dilation_y = self->nn_param.conv1d.dilation - 1;
    }
    p.khr.overflow_policy = self->vx_param.overflow_policy;
    p.khr.rounding_policy =  self->vx_param.rounding_policy;
    p.khr.down_scale_size_rounding = self->vx_param.down_scale_size_rounding;

    p.padding_y_bottom = self->nn_param.conv1d.pad[1];

    _reshape_tensor(inputs[0], &(nn_param->local.input_tensor));
    _reshape_tensor(inputs[1], &(nn_param->local.weight_tensor));
    _reshape_tensor(outputs[0], &(nn_param->local.output_tensor));

    self->n = vxConvolutionLayer(
        self->graph->g,
        nn_param->local.input_tensor,
        nn_param->local.weight_tensor,
        (NULL == inputs[2]) ? NULL : inputs[2]->t,
        (vx_nn_convolution_params_t *)&p,
        sizeof( vx_nn_convolution_params_ext_t ),
        nn_param->local.output_tensor
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
    vsi_nn_conv1d_param *nn_param = &self->nn_param.conv1d;

    vsi_nn_compute_padding_conv1d(
        inputs[0]->attr.size,
        &self->nn_param.conv1d.ksize,
        &self->nn_param.conv1d.stride,
        &self->nn_param.conv1d.dilation,
        self->nn_param.conv1d.pad_type,
        self->nn_param.conv1d.pad
    );

    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.size[0] = vsi_nn_ComputeFilterSize
            (
            inputs[0]->attr.size[0],
            nn_param->ksize,
            &nn_param->pad[0],
            nn_param->stride,
            nn_param->dilation,
            VSI_NN_ROUND_FLOOR
            );

        outputs[0]->attr.size[1] = nn_param->weights;
        outputs[0]->attr.size[2] = inputs[0]->attr.size[2];
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
    }
    return TRUE;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    if (self->nn_param.conv1d.local.input_tensor != NULL)
    {
        vxReleaseTensor(&self->nn_param.conv1d.local.input_tensor);
        self->nn_param.conv1d.local.input_tensor = NULL;
    }
    if (self->nn_param.conv1d.local.weight_tensor != NULL)
    {
        vxReleaseTensor(&self->nn_param.conv1d.local.weight_tensor);
        self->nn_param.conv1d.local.weight_tensor = NULL;
    }
    if (self->nn_param.conv1d.local.output_tensor != NULL)
    {
        vxReleaseTensor(&self->nn_param.conv1d.local.output_tensor);
        self->nn_param.conv1d.local.output_tensor = NULL;
    }
    vsi_nn_op_common_deinit(self);
    return VSI_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ CONV1D,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 3,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif

