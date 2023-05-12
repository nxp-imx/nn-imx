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
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"

typedef struct _custom_tiny_yolov4_postprocess_box_local_data_t {
    int32_t placeholder;
} custom_tiny_yolov4_postprocess_box_local_data_t;

/*
 Declare number of input and output.
 */
#define _INPUT_NUM          (2)
#define _OUTPUT_NUM         (1)

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_param_t * param = NULL;
    float bias_0 = self->nn_param.custom_tiny_yolov4_postprocess_box.bias_0;
    float bias_1 = self->nn_param.custom_tiny_yolov4_postprocess_box.bias_1;

    param = vsi_nn_kernel_param_create();

    vsi_nn_kernel_param_add_float32( param, "bias_0", bias_0 );
    vsi_nn_kernel_param_add_float32( param, "bias_1", bias_1 );

    self->n = vsi_nn_kernel_selector( self->graph, "tiny_yolov4_postprocess_box",
        inputs, _INPUT_NUM, outputs, _OUTPUT_NUM, param );

    if ( self->n )
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
    /*TODO: Check tensor shapes. */
    VSI_UNREFERENCED(self);
    VSI_UNREFERENCED(inputs);
    VSI_UNREFERENCED(outputs);
    return TRUE;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    uint32_t rank = inputs[0]->attr.dim_num;
    vsi_bool ret = TRUE;

    VSI_UNREFERENCED(self);

    if ( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.dim_num = rank;
        outputs[0]->attr.size[0] = inputs[0]->attr.size[2];
        outputs[0]->attr.size[1] = inputs[0]->attr.size[0];
        outputs[0]->attr.size[2] = inputs[0]->attr.size[1];
        if (rank > 3)
        {
            memcpy( &outputs[0]->attr.size[3], &inputs[0]->attr.size[3], (rank - 3) * sizeof(vsi_size_t) );
        }
    }

    return ret;
} /* op_setup() */


__BEGIN_DECLS

/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ CUSTOM_TINY_YOLOV4_POSTPROCESS_BOX,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );

__END_DECLS

