/****************************************************************************
*
*    Copyright (c) 2018 Vivante Corporation
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
#include "vsi_nn_platform.h"
#include "vsi_nn_types.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_tensor_util.h"

vsi_status vsi_nn_depth2space_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status;
    vsi_nn_tensor_t *block_size_tensor = NULL;
    vx_nn_reorg_params_t param;

    status = VSI_FAILURE;
    memset(&param, 0, sizeof(vx_nn_reorg_params_t));

    block_size_tensor = vsi_nn_VariableToTensor(self,
        (uint8_t *)&self->nn_param.depth2space.block_size,
        VSI_NN_TYPE_UINT32);
    if( NULL == block_size_tensor )
    {
        VSILOGE("Create block_size_tensor fail.(depth2space)");
        return VSI_FAILURE;
    }
    self->nn_param.depth2space.local.block_size_tensor = block_size_tensor;
    param.block_size = REQUIRED_IO(block_size_tensor);
    param.type = VX_REORG_DEPTH_TO_SPACE;

    self->n = vxReorgLayer2( self->graph->g,
        inputs[0]->t,
        &param,
        sizeof(vx_nn_reorg_params_t),
        outputs[0]->t);
    if( NULL != self->n )
    {
        status = VSI_SUCCESS;
    }
    return status;
} /* vsi_nn_lrn_compute() */