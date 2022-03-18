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

#include "vsi_nn_log.h"
#include "kernel/vsi_nn_kernel.h"
#include "ops/vsi_nn_op_deconvolution.h"

#define _INPUT_NUM          (3)
#define _OUTPUT_NUM         (1)
#define _VX_KERNEL_VAR      (vx_cpu_backend_kernel_DECONV2D)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)

extern vsi_nn_op_proc_t vsi_nn_op_DECONVOLUTION;
#define InternalDeconv2DOp vsi_nn_op_DECONVOLUTION

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_param_t * param = NULL;

    param = vsi_nn_kernel_param_create();

    vsi_nn_kernel_param_add_buffer( param, "stride", self->nn_param.deconv.stride, 2 );
    vsi_nn_kernel_param_add_buffer( param, "pad", self->nn_param.deconv.pad, 4 );

    if ( inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_UINT8
            && outputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_UINT8
            && self->nn_param.deconv.group <= 1)
    {
        self->n = (vx_node)vsi_nn_kernel_selector( self->graph, "cpu beckend conv2d",
                inputs, 3, outputs, 1, param );

        if ( self->n )
        {
            status = VSI_SUCCESS;
        }
    }
    else
    {
        status = InternalDeconv2DOp.compute( self, inputs, outputs );
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
    return InternalDeconv2DOp.check( self, inputs, outputs );
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * node,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    return InternalDeconv2DOp.setup( node, inputs, outputs );
} /* op_setup() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ CPU_BACKEND_DECONV2D,
    /* op_init    */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );
#ifdef __cplusplus
}
#endif
