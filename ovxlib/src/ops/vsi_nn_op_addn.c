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
#include <stdlib.h>
#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_log.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_internal_node.h"

static int32_t _get_input_num
    (
    vsi_nn_node_t   * self,
    vsi_nn_tensor_t ** inputs
    )
{
    int32_t num;
    num = (int32_t)(self->input.num - 1);
    while( num >= 0 && NULL == inputs[num] )
    {
        num --;
    }
    if( 0 > num )
    {
        return -1;
    }

    num++;
    return num;
}

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    return vsi_nn_compute_internal_node( self );
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    return TRUE;
} /* op_check() */

static vsi_status op_optimize
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_opt_direction_e direction
    )
{
    return vsi_nn_optimize_internal_node( self, direction );
} /* op_optimize() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = TRUE;
    uint32_t i;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_internal_tensor_t* temp_output_tensor;
    uint32_t input_num = 0;

    vsi_nn_init_internal_node_wksp( self );

    input_num = _get_input_num(self, inputs);
    for(i = 0; i < input_num -1; i++)
    {
        /* loop call add for input_num -1 times */

        /* setup input for each add */
        curr = vsi_nn_new_internal_node( self, VSI_NN_OP_ADD, 0, 0 );
        if(i == 0)
        {
            curr->inputs[0] = inputs[i];
        }
        else
        {
            curr->inputs[0] = temp_output_tensor->t;
        }
        curr->inputs[1] = inputs[i+1];

        /* setup output for each add */
        if(i < input_num - 2)
        {
            memset(&attr, 0, sizeof(attr));
            attr.dim_num = VSI_NN_DIM_AUTO;
            attr.vtl = TRUE;
            attr.is_const = FALSE;
            attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;

            temp_output_tensor = vsi_nn_new_internal_tensor( self, &attr, 0.0f );

            curr->outputs[0] = temp_output_tensor->t;
        }
        else
        {
            curr->outputs[0] = outputs[0];
        }

        vsi_nn_setup_internal_node_op( self, curr );
    }
    return ret;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    vsi_nn_deinit_internal_node_wksp( self );

    return status;
} /* op_deinit() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ ADDN,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ 2,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif
