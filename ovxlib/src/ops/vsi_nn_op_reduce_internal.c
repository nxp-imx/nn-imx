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
#include "vsi_nn_log.h"
#include "vsi_nn_node.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_math.h"
#include "kernel/vsi_nn_kernel.h"
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"

#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)


static vsi_status _reduce_internal_op_compute
    (
    const char * kernel_name,
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status;
    vsi_nn_tensor_t* reshape_tensors[2] = { NULL };
    int32_t shapes[2][VSI_NN_MAX_DIM_NUM] = { { 0 } };
    uint32_t rank_in = 0;
    uint32_t rank_out = 0;
    int32_t axis = 0;
    int32_t new_axis = 0;
    uint32_t axis_size = 0;
    vsi_bool ret;
    vsi_nn_kernel_param_t * param = NULL;

    if( NULL == self )
    {
        return VSI_FAILURE;
    }
    status = VSI_FAILURE;



    param =vsi_nn_kernel_param_create();

    if (strcmp(kernel_name, "reducemax_internal") == 0)
    {
        vsi_nn_reducemax_internal_param * p = &(self->nn_param.reducemax_internal);
        axis = p->axis[0];
    }
    else if (strcmp(kernel_name, "reducemin_internal") == 0)
    {
        vsi_nn_reducemin_internal_param * p = &(self->nn_param.reducemin_internal);
        axis = p->axis[0];
    }
    else
    {
        return VSI_FAILURE;

    }

    ret = vsi_nn_kernel_optimize_reduce_shape(
            (int32_t *)inputs[0]->attr.size, inputs[0]->attr.dim_num,
            &axis, 1,
            (int32_t *)outputs[0]->attr.size, outputs[0]->attr.dim_num,
            shapes[0], &rank_in, shapes[1], &rank_out,
            &new_axis, &axis_size);

    // Add params
     vsi_nn_kernel_param_add_int32( param, "axis", new_axis );

    if( ret )
    {
        reshape_tensors[0] = vsi_nn_reshape_tensor( self->graph,
                inputs[0], (uint32_t*)shapes[0], rank_in );
        reshape_tensors[1] = vsi_nn_reshape_tensor( self->graph,
                outputs[0], (uint32_t*)shapes[1], rank_out );

        self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
                kernel_name,
                &reshape_tensors[0], 1,
                &reshape_tensors[1], 1, param );

        vsi_nn_ReleaseTensor( &reshape_tensors[0] );
        vsi_nn_ReleaseTensor( &reshape_tensors[1] );
    }
    if( self->n )
    {
        status = VSI_SUCCESS;
    }

    vsi_nn_kernel_param_release( &param );
    return status;
} /* op_compute() */

static vsi_bool _reduce_internal_op_setup
    (
    const char * kernel_name,
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    int32_t axis = 0;

    if (strcmp(kernel_name, "reducemax_internal") == 0)
    {
        vsi_nn_reducemax_internal_param * p = &(self->nn_param.reducemax_internal);

        axis = p->axis[0];
        if (axis < 0)
        {
            axis = axis + inputs[0]->attr.dim_num;
            if (axis < 0)
            {
                VSILOGW("error input axis value %d input dim num is %d",
                 p->axis[0], inputs[0]->attr.dim_num);
                return FALSE;
            }
            p->axis[0] = axis;
        }
    }
    if (strcmp(kernel_name, "reducemin_internal") == 0)
    {
        vsi_nn_reducemin_internal_param * p = &(self->nn_param.reducemin_internal);

        axis = p->axis[0];
        if (axis < 0)
        {
            axis = axis + inputs[0]->attr.dim_num;
            if (axis < 0)
            {
                VSILOGW("error input axis value %d input dim num is %d",
                 p->axis[0], inputs[0]->attr.dim_num);
                return FALSE;
            }
            p->axis[0] = axis;
        }
    }
    else
    {
         return FALSE;
    }

    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        uint32_t i = 0;
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num - 1;

        for (i = 0; i < (uint32_t)axis; i++)
        {
            outputs[0]->attr.size[i] = inputs[0]->attr.size[i];
        }

        for (i = axis; i < outputs[0]->attr.dim_num; i++)
        {
            outputs[0]->attr.size[i] = inputs[0]->attr.size[i + 1];
        }

        if (inputs[0]->attr.dim_num == 1)
        {
            outputs[0]->attr.dim_num = 1;
            outputs[0]->attr.size[0] = 1;
        }
    }

    return TRUE;
} /* op_setup() */



#ifdef __cplusplus
extern "C" {
#endif

#define DEF_REDUCE_INTERNAL_OP(name, kernel_name) \
            static vsi_status op_compute_##kernel_name \
                ( \
                vsi_nn_node_t * self, \
                vsi_nn_tensor_t ** inputs, \
                vsi_nn_tensor_t ** outputs \
                ) \
            { \
                return _reduce_internal_op_compute( ""#kernel_name, self, inputs, outputs ); \
            } \
            static vsi_bool op_setup_##kernel_name \
                ( \
                vsi_nn_node_t * self, \
                vsi_nn_tensor_t ** inputs, \
                vsi_nn_tensor_t ** outputs \
                ) \
            { \
                return _reduce_internal_op_setup( ""#kernel_name, self, inputs, outputs ); \
            } \
DEF_OP_REG  \
    ( \
    /* op_name    */ name, \
    /* init       */ NULL, \
    /* compute    */ op_compute_##kernel_name, \
    /* deinit     */ NULL, \
    /* check      */ NULL, \
    /* setup      */ op_setup_##kernel_name, \
    /* optimize   */ NULL, \
    /* input_num  */ 1, \
    /* output_num */ 1 \
    )


DEF_REDUCE_INTERNAL_OP( REDUCEMAX_INTERNAL, reducemax_internal );
DEF_REDUCE_INTERNAL_OP( REDUCEMIN_INTERNAL, reducemin_internal );

#undef DEF_REDUCE_INTERNAL_OP
#ifdef __cplusplus
}
#endif
