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
#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{

    if (self->nn_param.dataconvert.lcl_data->use_reshape == FALSE
        && inputs[0]->t != NULL && outputs[0]->t != NULL)
    {
        self->n = vxTensorCopyNode(
            self->graph->g,
            inputs[0]->t,
            outputs[0]->t
            );

        if(NULL == self->n)
        {
            VSILOGE( "Create vxTensorCopyNode fail." );
            return VSI_FAILURE;
        }
    }

    return VSI_SUCCESS;
} /* op_compute() */

static vsi_bool _is_same_quant
    (
    vsi_nn_node_t   * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_dtype_t *dtype,*_dtype;

    dtype = &inputs[0]->attr.dtype;
    _dtype = &outputs[0]->attr.dtype;

    if(vsi_nn_DtypeCompare(dtype, _dtype) == FALSE)
    {
        return FALSE;
    }

    return TRUE;
} /* _is_same_quant */


static vsi_status op_optimize
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_opt_direction_e direction
    )
{
    vsi_status     status;

    status = VSI_SUCCESS;

    if ( _is_same_quant(self, inputs, outputs) == FALSE ||
        (inputs[0]->t != NULL && outputs[0]->t != NULL))
    {
        return status;
    }

    VSILOGD("Optimize %s, uid %u", vsi_nn_OpGetName(self->op), self->uid);
    if( direction == VSI_NN_OPTIMIZE_FORWARD )
    {
        if(NULL == inputs[0]->t && NULL != outputs[0]->t)
        {
            inputs[0]->t = vxReshapeTensor(outputs[0]->t,
                (int32_t *)inputs[0]->attr.size, inputs[0]->attr.dim_num);
            if( inputs[0]->t == NULL )
            {
                VSILOGE("Call vxReshapeTensor fail");
                return VSI_FAILURE;
            }
            self->nn_param.dataconvert.lcl_data->use_reshape = TRUE;
        }
    }
    else
    {
        if(NULL == outputs[0]->t && NULL != inputs[0]->t)
        {
            outputs[0]->t = vxReshapeTensor(inputs[0]->t,
                (int32_t *)outputs[0]->attr.size, outputs[0]->attr.dim_num);
            if( outputs[0]->t == NULL )
            {
                VSILOGE("Call vxReshapeTensor fail");
                return VSI_FAILURE;
            }
            self->nn_param.dataconvert.lcl_data->use_reshape = TRUE;
        }
    }

    return status;
} /* op_optimize() */

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    self->nn_param.dataconvert.lcl_data =
    (vsi_nn_dataconvert_lcl_data *)malloc(sizeof(vsi_nn_dataconvert_lcl_data));
    if (NULL == self->nn_param.dataconvert.lcl_data)
    {
        return  VX_ERROR_NO_MEMORY;
    }

    memset( self->nn_param.dataconvert.lcl_data, 0, sizeof(vsi_nn_dataconvert_lcl_data) );

    return status;
} /* op_init() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    if(self->nn_param.dataconvert.lcl_data)
    {

        free(self->nn_param.dataconvert.lcl_data);
        self->nn_param.dataconvert.lcl_data = NULL;
    }
    vsi_nn_op_common_deinit(self);

    return VSI_SUCCESS;
} /* op_deinit() */

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

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ DATACONVERT,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ vsi_nn_op_common_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ 1,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif

