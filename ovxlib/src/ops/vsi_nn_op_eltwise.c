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
#include "vsi_nn_error.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_math.h"
#include "kernel/vsi_nn_kernel.h"
#include "kernel/vsi_nn_kernel_eltwise.h"

static vsi_status _eltwise_op_compute
    (
    const char * kernel_name,
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status;
    vsi_nn_tensor_t* reshape_tensors[3] = { NULL };
    int32_t shapes[3][VSI_NN_MAX_DIM_NUM] = { { 0 } };
    uint32_t new_rank = 0;
    vsi_bool ret = TRUE;
    vx_bool doShapeOptimized = TRUE;
    vsi_nn_kernel_param_t * param = NULL;

    if( NULL == self )
    {
        return VSI_FAILURE;
    }
    status = VSI_FAILURE;

    if ( strcmp(kernel_name, "sub") == 0
      || strcmp(kernel_name, "add") == 0
      || strcmp(kernel_name, "mul") == 0 )
    {
        doShapeOptimized = FALSE;

        reshape_tensors[0] = inputs[0];
        reshape_tensors[1] = inputs[1];
        reshape_tensors[2] = outputs[0];
    }

    // TODO: This optimzie is a hack for gpu path,
    // it should be moved to gpu kernel setup.
    if (doShapeOptimized)
    {
        ret = vsi_nn_kernel_optimize_eltwise_shape(
                (int32_t *)inputs[0]->attr.size, inputs[0]->attr.dim_num,
                (int32_t *)inputs[1]->attr.size, inputs[1]->attr.dim_num,
                (int32_t *)outputs[0]->attr.size, outputs[0]->attr.dim_num,
                shapes[0], shapes[1], shapes[2], &new_rank );
    }

    if( ret )
    {
        // Add params
        param = vsi_nn_kernel_param_create();
        vsi_nn_kernel_param_add_float32( param, "scale", self->nn_param.multiply.scale );
        vsi_nn_kernel_param_add_int32( param, "overflow_policy", self->vx_param.overflow_policy );
        vsi_nn_kernel_param_add_int32( param, "rounding_policy", self->vx_param.rounding_policy );

        if (doShapeOptimized)
        {
            reshape_tensors[0] = vsi_nn_reshape_tensor( self->graph,
                    inputs[0], (uint32_t*)shapes[0], new_rank );
            reshape_tensors[1] = vsi_nn_reshape_tensor( self->graph,
                    inputs[1], (uint32_t*)shapes[1], new_rank );
            reshape_tensors[2] = vsi_nn_reshape_tensor( self->graph,
                    outputs[0], (uint32_t*)shapes[2], new_rank );
        }

        self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
                kernel_name,
                &reshape_tensors[0], 2,
                &reshape_tensors[2], 1, param );

        if (doShapeOptimized)
        {
            vsi_nn_ReleaseTensor( &reshape_tensors[0] );
            vsi_nn_ReleaseTensor( &reshape_tensors[1] );
            vsi_nn_ReleaseTensor( &reshape_tensors[2] );
        }

        vsi_nn_kernel_param_release( &param );
    }
    if( self->n )
    {
        status = VSI_SUCCESS;
    }

    return status;
} /* _eltwise_op_compute() */

vsi_bool vsi_nn_op_eltwise_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    uint32_t i, j, out_rank, in2_rank;
    uint32_t shape[VSI_NN_MAX_DIM_NUM] = { 0 };
    vsi_bool ret = TRUE;

    out_rank = inputs[0]->attr.dim_num;
    for ( i = 1; i < self->input.num; i++)
    {
        in2_rank = inputs[i]->attr.dim_num;
        out_rank = vsi_nn_max( out_rank, in2_rank );
    }

    for(i = 0; i < out_rank; i++)
    {
        uint32_t sz0, sz1;

        sz0 = i < inputs[0]->attr.dim_num ? inputs[0]->attr.size[i] : 1;
        for ( j = 1; j < self->input.num; j++)
        {
            sz1 = i < inputs[j]->attr.dim_num  ? inputs[j]->attr.size[i] : 1;
            sz0 = vsi_nn_max( sz0, sz1 );
            if (sz0 != sz1 && sz0 != 1 && sz1 != 1)
            {
                /* Two dimensions are compatible when:
                1. they are equal, or
                2. one of them is 1*/
                VSILOGE("Input size mismatch.");
                return FALSE;
            }
        }
        shape[i] = sz0;
    }

    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.dim_num = out_rank;
        memcpy( outputs[0]->attr.size, shape, out_rank * sizeof(uint32_t) );
    }
    else
    {
        uint32_t total_size_got;
        uint32_t total_size_expected;
        total_size_expected = vsi_nn_ShapeProduct( shape, out_rank );
        total_size_got = vsi_nn_ShapeProduct( outputs[0]->attr.size,
                outputs[0]->attr.dim_num );
        if( total_size_expected != total_size_got )
        {
            VSILOGW("Output size mismatch, expect %d, but got %d",
                    total_size_expected, total_size_got);
            ret = FALSE;
        }
    }

    return ret;
} /* vsi_nn_op_eltwise_setup() */

#ifdef __cplusplus
extern "C" {
#endif

#define DEF_ELEMENT_WISE_OP(name, kernel_name) \
            static vsi_status op_compute_##kernel_name \
                ( \
                vsi_nn_node_t * self, \
                vsi_nn_tensor_t ** inputs, \
                vsi_nn_tensor_t ** outputs \
                ) \
            { \
                return _eltwise_op_compute( ""#kernel_name, self, inputs, outputs ); \
            } \
DEF_OP_REG(name, NULL, op_compute_##kernel_name, vsi_nn_op_common_deinit, NULL, vsi_nn_op_eltwise_setup, NULL, 2, 1)

DEF_ELEMENT_WISE_OP( MINIMUM, minimum );
DEF_ELEMENT_WISE_OP( MAXIMUM, maximum );
DEF_ELEMENT_WISE_OP( ADD, add );
DEF_ELEMENT_WISE_OP( SUBTRACT, sub );
DEF_ELEMENT_WISE_OP( DIVIDE, div );
DEF_ELEMENT_WISE_OP( MULTIPLY, mul );
DEF_ELEMENT_WISE_OP( POW, pow );


#undef DEF_ELEMENT_WISE_OP

#ifdef __cplusplus
}
#endif
