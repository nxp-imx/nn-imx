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
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "ops/vsi_nn_op_upsample.h"
#include "client/vsi_nn_vxkernel.h"
#include "kernel/vsi_nn_kernel_eltwise.h"

#define _INPUT_NUM          (2)
#define _OUTPUT_NUM         (1)


static vsi_bool vsi_nn_upsample_optimize_shape
    (
    vsi_nn_node_t * self,
    const int32_t* shape_in0, const int32_t* shape_in1,
    const int32_t* shape_out, const size_t rank_in,
    int32_t* out_shape_input0, int32_t* out_shape_input1,
    int32_t* out_shape_output, uint32_t* out_rank_output
    )
{
    vsi_bool   enable_image_2d = FALSE;
    int32_t    hwLitimLen      = 65536;

    if ((2 == self->nn_param.upsample.scale[0])
       && (2 == self->nn_param.upsample.scale[1]))
    {
        if (rank_in < 3)
        {
            enable_image_2d = TRUE;
        }
        else
        {
            enable_image_2d = (vsi_bool)((shape_out[1] * shape_out[2] < hwLitimLen)
                                        && ( (shape_out[1] % 2) == 0 ));
        }
    }

    if( rank_in == 1 )
    {
        out_shape_input0[0]  = shape_in0[0];
        out_shape_input0[1]  = 1;
        out_shape_input0[2]  = 1;
        out_shape_input1[0]  = shape_in1[0];
        out_shape_input1[1]  = 1;
        out_shape_input1[2]  = 1;
        out_shape_output[0]  = shape_out[0];
        out_shape_output[1]  = 1;
        out_shape_output[2]  = 1;
        *out_rank_output     = 2;
    }
    else if(rank_in == 3 && enable_image_2d)
    {
        out_shape_input0[0]  = shape_in0[0];
        out_shape_input0[1]  = shape_in0[1] * shape_in0[2];
        out_shape_input0[2]  = 1;
        out_shape_input1[0]  = shape_in1[0];
        out_shape_input1[1]  = shape_in1[1] * shape_in1[2];
        out_shape_input1[2]  = 1;
        out_shape_output[0]  = shape_out[0];
        out_shape_output[1]  = shape_out[1] * shape_out[2];
        out_shape_output[2]  = 1;
        *out_rank_output     = 2;
    }
    else if(rank_in == 4 && enable_image_2d)
    {
        out_shape_input0[0]  = shape_in0[0];
        out_shape_input0[1]  = shape_in0[1] * shape_in0[2];
        out_shape_input0[2]  = 1;
        out_shape_input0[3]  = shape_in0[3];
        out_shape_input1[0]  = shape_in1[0];
        out_shape_input1[1]  = shape_in1[1] * shape_in1[2];
        out_shape_input1[2]  = 1;
        out_shape_input1[3]  = shape_in1[3];
        out_shape_output[0]  = shape_out[0];
        out_shape_output[1]  = shape_out[1] * shape_out[2];
        out_shape_output[2]  = 1;
        out_shape_output[3]  = shape_out[3];
        if (1 == shape_in0[3])
        {
            *out_rank_output     = 2;
        }
        else
        {
            *out_rank_output     = 4;
        }
    }
    else
    {
        uint32_t i;
        for (i = 0; i < rank_in; i++)
        {
            out_shape_input0[i]  = shape_in0[i];
            out_shape_input1[i]  = shape_in1[i];
            out_shape_output[i]  = shape_out[i];
        }
        *out_rank_output = (uint32_t)rank_in;
    }

    return TRUE;
}

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_tensor_t* reshape_tensors[3] = { NULL };
    int32_t shapes[3][VSI_NN_MAX_DIM_NUM] = {{ 1 }};
    uint32_t new_rank = 0;
    vsi_bool ret;
    vsi_nn_kernel_param_t * param = NULL;
    int32_t scale_x  = (int32_t)self->nn_param.upsample.scale[0];
    int32_t scale_y  = (int32_t)self->nn_param.upsample.scale[1];

    if( NULL == self )
    {
        return VSI_FAILURE;
    }

    param =vsi_nn_kernel_param_create();

    ret = vsi_nn_upsample_optimize_shape(self,
            (int32_t *)inputs[0]->attr.size,  (int32_t *)inputs[1]->attr.size,
            (int32_t *)outputs[0]->attr.size, inputs[0]->attr.dim_num,
            shapes[0], shapes[1], shapes[2], &new_rank );

    vsi_nn_kernel_param_add_int32( param, "scale_x",  scale_x );
    vsi_nn_kernel_param_add_int32( param, "scale_y",  scale_y );

    if( ret )
    {

        reshape_tensors[0] = vsi_nn_reshape_tensor( self->graph,
                inputs[0],  (uint32_t*)shapes[0], new_rank );
        reshape_tensors[1] = vsi_nn_reshape_tensor( self->graph,
                inputs[1],  (uint32_t*)shapes[1], new_rank );
        reshape_tensors[2] = vsi_nn_reshape_tensor( self->graph,
                outputs[0], (uint32_t*)shapes[2], new_rank );
        self->n = (vx_node)vsi_nn_kernel_selector( self->graph, "upsample",
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

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    uint32_t h;
    uint32_t w;

    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        w = self->nn_param.upsample.size[0];
        h = self->nn_param.upsample.size[1];
        if (0 == self->nn_param.upsample.scale[0])
        {
            self->nn_param.upsample.scale[0] = self->nn_param.upsample.size[0] /
                inputs[0]->attr.size[0];
        }
        if (0 == self->nn_param.upsample.scale[1])
        {
            self->nn_param.upsample.scale[1] = self->nn_param.upsample.size[1] /
                inputs[0]->attr.size[1];
        }
        if ( 0 == self->nn_param.upsample.size[0] )
        {
            w = inputs[0]->attr.size[0] * self->nn_param.upsample.scale[0];
        }
        if ( 0 == self->nn_param.upsample.size[1] )
        {
            h = inputs[0]->attr.size[1] * self->nn_param.upsample.scale[1];
        }
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
        outputs[0]->attr.size[0] = w;
        outputs[0]->attr.size[1] = h;
        outputs[0]->attr.size[2] = inputs[0]->attr.size[2];
        outputs[0]->attr.size[3] = inputs[1]->attr.size[3];
    }

    return TRUE;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    uint32_t i;
    for (i = 0; i < _VSI_NN_UPSAMPLE_LOCAL_TENSOR_NUM; i++)
    {
        if (self->nn_param.upsample.local.local_tensor[i] != NULL)
        {
            vxReleaseTensor(&(self->nn_param.upsample.local.local_tensor[i]));
            self->nn_param.upsample.local.local_tensor[i] = NULL;
        }
    }
    vsi_nn_op_common_deinit(self);

    return VSI_SUCCESS;
} /* op_deinit() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ UPSAMPLE,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );
#ifdef __cplusplus
}
#endif

