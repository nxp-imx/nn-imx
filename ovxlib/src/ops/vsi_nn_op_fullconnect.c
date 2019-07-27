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
    uint32_t axis;
    uint32_t i = 0;
    uint32_t num_fc = 1, num_no_fc = 1;
    uint32_t num_of_intput_dims = 0;
    int32_t input_size[VSI_NN_MAX_DIM_NUM] = {0};
    uint32_t dims = 0;
    vx_tensor input = NULL;
    vx_tensor output = NULL;
    vx_tensor weight = NULL;
    vx_tensor bias = NULL;

    status = VSI_FAILURE;

    memcpy(input_size, inputs[0]->attr.size, sizeof(uint32_t) * VSI_NN_MAX_DIM_NUM);
    num_of_intput_dims = inputs[0]->attr.dim_num;
    axis = inputs[0]->attr.dim_num - 2;

    for(i = 0; i <= (uint32_t)axis; ++i)
    {
        num_fc *= input_size[i];
    }
    for(i = axis + 1; i < num_of_intput_dims; ++i)
    {
        num_no_fc *= input_size[i];
    }

    input_size[0] = num_fc;
    input_size[1] = num_no_fc;
    dims= 2;
    input = vxReshapeTensor(inputs[0]->t, input_size, dims);

    weight = inputs[1]->t;

    if( inputs[2] != NULL )
    {
        bias = inputs[2]->t;
    }

    output = outputs[0]->t;

    self->n = vxFullyConnectedLayer(
        self->graph->g,
        input,
        weight,
        bias,
        0,
        0,
        self->vx_param.overflow_policy,
        self->vx_param.rounding_policy,
        self->vx_param.down_scale_size_rounding,
        output
        );
    if( NULL != self->n )
    {
        status = VSI_SUCCESS;
    }

    if (input)  vxReleaseTensor(&input);

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
    uint32_t graph_version_major = 0;
    uint32_t graph_version_minor = 0;
    uint32_t graph_version_patch = 0;

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
        vsi_nn_GetGraphVersion( self->graph, &graph_version_major, &graph_version_minor, &graph_version_patch );
        if ( graph_version_major >= 1 && graph_version_minor >= 1 && graph_version_patch >= 0 )
        {
            switch (input_dim)
            {
            // CAUTION: FC input shape need contain batch size.
            // and graph version no smaller than 5.0.0
            case 2:
            case 3:
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
        else
        {
            switch (input_dim)
            {
            // CAUTION: FC input shape with/without batch size.
            // and graph version smaller than 5.0.0
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
    }
    return TRUE;
} /* op_setup() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ FCL,
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

