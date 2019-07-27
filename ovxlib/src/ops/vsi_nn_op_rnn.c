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
#include "vsi_nn_log.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"

#ifdef __cplusplus
extern "C" {
#endif

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_tensor_t* act_tensor = NULL;
    vx_nn_rnn_params_t param = {0};

    act_tensor = vsi_nn_VariableToTensor(self,
        (uint8_t*)&self->nn_param.rnn.activation,
        VSI_NN_TYPE_INT32);

    if (!act_tensor)
    {
        VSILOGE("RNN->Create Activation Tensor failed");
        status = VSI_FAILURE;
    }
    else
    {
        param.weights            = REQUIRED_IO(inputs[1]);
        param.recurrent_weights  = REQUIRED_IO(inputs[2]);
        param.bias               = REQUIRED_IO(inputs[3]);
        param.state_in           = REQUIRED_IO(inputs[4]);
        param.activation         = REQUIRED_IO(act_tensor);
        self->n = vxRNNLayer(
                    self->graph->g,
                    REQUIRED_IO(inputs[0]),
                    &param,
                    sizeof(param),
                    /*state output*/REQUIRED_IO(outputs[0]),
                    /*output*/REQUIRED_IO(outputs[1]));

        vsi_nn_ReleaseTensor(&act_tensor);
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
    uint32_t input_idx = 0;
    do {
        vsi_bool break_early = FALSE;

        // input_idx = 0 : inputs[0].shape = shape(batch_size, input_size)
        if (input_idx >= self->input.num) break;
        break_early = (inputs[input_idx]->attr.dim_num != 2);
        if (break_early) break;
        input_idx ++;

        // input_idx = 1 : inputs[1].shape = shape(num_units, input_size)
        if (input_idx >= self->input.num) break;
        break_early = (inputs[input_idx]->attr.dim_num != 2);
        if (break_early) break;
        input_idx ++;

        // input_idx = 2 : inputs[2].shape = shape(num_units, num_units)
        if (input_idx >= self->input.num) break;
        break_early = (inputs[input_idx]->attr.dim_num != 2);
        if (break_early) break;
        input_idx ++;

        // input_idx = 3 : inputs[3].shape = shape(num_units)
        if (input_idx >= self->input.num) break;
        break_early = (inputs[input_idx]->attr.dim_num != 1);
        if (break_early) break;
        input_idx ++;

        // input_idx = 4 : inputs[4].shape = shape(batch_size, num_units)
        if (input_idx >= self->input.num) break;
        break_early = (inputs[input_idx]->attr.dim_num != 2);
        if (break_early) break;
        input_idx ++;

        return TRUE;
    } while(0);

    VSILOGE("RNN check shape faild at Input[%d]", input_idx);
    return FALSE;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    if (VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num) {
        outputs[0]->attr.size[0] = inputs[4]->attr.size[0];
        outputs[0]->attr.size[1] = inputs[4]->attr.size[1];
        outputs[1]->attr.size[0] = inputs[4]->attr.size[0];
        outputs[1]->attr.size[1] = inputs[4]->attr.size[1];

        outputs[0]->attr.dim_num = outputs[1]->attr.dim_num = inputs[4]->attr.dim_num;
    }

    return TRUE;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    if (NULL == self)
    {
        return VSI_FAILURE;
    }

    if (NULL != self->n)
    {
        vxReleaseNode(&self->n);
        self->n = NULL;
    }

    return VSI_SUCCESS;
} /* op_deinit() */

/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ RNN,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 5,
    /* output_num */ 2
    );
#ifdef __cplusplus
}
#endif
