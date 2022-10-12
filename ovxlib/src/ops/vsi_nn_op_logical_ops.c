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
#include "vsi_nn_error.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "utils/vsi_nn_constraint_check.h"

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

    if ( NULL == self )
    {
        return VSI_FAILURE;
    }

    param = vsi_nn_kernel_param_create();
    vsi_nn_kernel_param_add_int32( param, "ops_type", self->nn_param.logical_ops.op );

    self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
            "logical_ops",
            inputs, _INPUT_NUM,
            outputs, _OUTPUT_NUM, param );

    vsi_nn_kernel_param_release( &param );

    if ( self->n )
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
    BEGIN_IO_TYPE_DECL(LOGICAL_OPS, 2, 1)
        IO_TYPE(D_I8, D_I8, D_I8)
        IO_TYPE(D_BOOL8, D_BOOL8, D_BOOL8)
        IO_TYPE(D_BF16, D_BF16, D_BOOL8)
    END_IO_TYPE_DECL(LOGICAL_OPS)
    if(!VALIDATE_OP_IO_TYPES(LOGICAL_OPS, self, inputs, self->input.num, outputs, self->output.num)) {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }

    return TRUE;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    uint32_t i, out_rank, in1_rank, in2_rank;
    vsi_size_t shape[VSI_NN_MAX_DIM_NUM] = { 0 };
    vsi_bool ret = TRUE;

    in1_rank = inputs[0]->attr.dim_num;
    in2_rank = inputs[1]->attr.dim_num;
    out_rank = vsi_nn_max( in1_rank, in2_rank );

    for(i = 0; i < out_rank; i++)
    {
        vsi_size_t sz0, sz1;
        sz0 = i < in1_rank ? inputs[0]->attr.size[i] : 1;
        sz1 = i < in2_rank ? inputs[1]->attr.size[i] : 1;
        shape[i] = vsi_nn_max( sz0, sz1 );
    }

    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.dim_num = out_rank;
        memcpy( outputs[0]->attr.size, shape, out_rank * sizeof(vsi_size_t) );
    }
    else
    {
        vsi_size_t total_size_got;
        vsi_size_t total_size_expected;
        total_size_expected = vsi_nn_ShapeProduct( shape, out_rank );
        total_size_got = vsi_nn_ShapeProduct( outputs[0]->attr.size,
                outputs[0]->attr.dim_num );
        if( total_size_expected != total_size_got )
        {
            VSILOGW("Output size mismatch, expect %"VSI_SIZE_T_SPECIFIER", but got %"VSI_SIZE_T_SPECIFIER"",
                    total_size_expected, total_size_got);
            ret = FALSE;
        }
    }

    return ret;
} /* op_setup() */


#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ LOGICAL_OPS,
    /* init       */ NULL,
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
