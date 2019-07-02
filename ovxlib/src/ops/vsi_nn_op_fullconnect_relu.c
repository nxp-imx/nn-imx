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
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "utils/vsi_nn_math.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"

static vsi_status _set_fc_relu_parameter
    (
    vsi_nn_node_t * self,
    vx_nn_convolution_relu_pooling_params_t * param
    );

static vsi_status _set_fc_relu_parameter
    (
    vsi_nn_node_t * self,
    vx_nn_convolution_relu_pooling_params_t * param
    )
{
    vx_scalar pad_const;
    int32_t pad_const_val;

    pad_const_val = 0;
    memset( param, 0, sizeof(vx_nn_convolution_relu_pooling_params_t) );
    pad_const = vxCreateScalar(self->graph->ctx->c, VX_TYPE_INT32, &pad_const_val);
    if( !pad_const )
    {
        VSILOGE("Create scalar fail\n");
        return VSI_FAILURE;
    }

    param->pad_x_left    = 0;
    param->pad_x_right   = 0;
    param->pad_y_top     = 0;
    param->pad_y_bottom  = 0;
    param->dilation_x    = 0;
    param->dilation_y    = 0;
    param->accumulator_bits = self->vx_param.accumulator_bits;
    param->overflow_policy = self->vx_param.overflow_policy;
    param->rounding_policy = self->vx_param.rounding_policy;
    param->down_scale_size_rounding = self->vx_param.down_scale_size_rounding;
    param->enable_relu = self->vx_param.has_relu;
    param->pool_type = 0;
    param->pool_size_x = 0;
    param->pool_size_y = 0;
    param->pad_mode = VX_PAD_CONSTANT;
    param->pad_const = pad_const;

    return VSI_SUCCESS;
} /* _set_fc_relu_parameter() */

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status;
    status = VSI_FAILURE;

    self->n = vxFullyConnectedReluLayer(
        self->graph->g,
        inputs[0]->t,
        inputs[1]->wb,
        0,
        0,
        self->vx_param.overflow_policy,
        self->vx_param.rounding_policy,
        self->vx_param.down_scale_size_rounding,
        self->vx_param.has_relu,
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
    vsi_bool ret;
    vx_nn_convolution_relu_pooling_params_t p;
    vx_weights_biases_parameter_optimizations_ext_t opt;
    vx_weights_biases_parameter_optimizations_ext_t * p_opt;

    ret = vsi_nn_OpSetup( VSI_NN_OP_FCL, self, inputs, outputs );

    /* Prepare weight_bias */
    if(inputs[1]->wb == NULL)
    {
        if( _set_fc_relu_parameter( self, &p ) != VSI_SUCCESS )
        {
            VSILOGE("set fc_relu weightbias parameter fail\n");
            return FALSE;
        }

        p_opt = NULL;
        memset( &opt, 0, sizeof( opt ) );
        if( outputs[0]->attr.dtype.qnt_type == VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC
         || inputs[0]->attr.dtype.qnt_type == VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC)
        {
            opt.inputZeroPoint = inputs[0]->attr.dtype.zero_point;
        }
        opt.zrl = -1;
        opt.outputFormat = outputs[0]->attr.dtype.vx_type;
        opt.num_of_input_dims = inputs[0]->attr.dim_num;
        opt.num_of_output_dims = outputs[0]->attr.dim_num;
        p_opt = &opt;

        inputs[1]->wb = vxCreateWeightsBiasesParameterFromTensors3(
            VX_CONVOLUTIONAL_NETWORK_FULLYCONNECTED_LAYER,
            inputs[0]->attr.size,
            outputs[0]->attr.size,
            outputs[0]->attr.size,
            &p,
            sizeof(p),
            (vx_weights_biases_parameter_optimizations_t *)p_opt,
            sizeof(opt),
            inputs[1]->t, inputs[2]->t
            );
        if( p.pad_const )
        {
            vxReleaseScalar( &p.pad_const );
        }
    }


    if( NULL == inputs[1]->wb )
    {
        VSILOGE( "Create weight bias fail." );
        ret = FALSE;
    }

    return ret;
} /* op_setup() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ FCL_RELU,
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

