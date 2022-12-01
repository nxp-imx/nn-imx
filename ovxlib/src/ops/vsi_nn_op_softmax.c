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
#include "vsi_nn_log.h"
#include "utils/vsi_nn_util.h"
#include "vsi_nn_internal_node.h"
#include "utils/vsi_nn_math.h"
#include "utils/vsi_nn_constraint_check.h"
#include "vsi_nn_tensor_util_prv.h"

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    return vsi_nn_internal_compute_node( self );
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = vsi_nn_is_stream_process_supported_types(self->graph, inputs, self->input.num);

    if (!ret)
    {
        /* check inputs outputs data type */
        BEGIN_IO_TYPE_DECL(SOFTMAX, 1, 1)
            /* IO_TYPE(INPUT, OUTPUT) */
            IO_TYPE(D_F32,          D_F32)
            IO_TYPE(D_F32,          D_F16)
            IO_TYPE(D_F16,          D_F16)
            IO_TYPE(D_F16,          D_F32)
            IO_TYPE(D_F16,          D_I16|Q_DFP)
            IO_TYPE(D_F16,          D_I16|Q_ASYM)
            IO_TYPE(D_F16,          D_I16|Q_SYM)
            IO_TYPE(D_F16,          D_I8|Q_DFP)
            IO_TYPE(D_F16,          D_I8|Q_SYM)
            IO_TYPE(D_F16,          D_I8|Q_ASYM)
            IO_TYPE(D_F16,          D_U8|Q_ASYM)
            IO_TYPE(D_BF16,         D_BF16)
            IO_TYPE(D_BF16,         D_F32)
            IO_TYPE(D_BF16,         D_F16)

            IO_TYPE(D_U8|Q_ASYM,    D_U8|Q_ASYM)
            IO_TYPE(D_U8|Q_ASYM,    D_F16)
            IO_TYPE(D_U8|Q_ASYM,    D_F32)

            IO_TYPE(D_I8|Q_ASYM,    D_I8|Q_ASYM)
            IO_TYPE(D_I8|Q_SYM,     D_I8|Q_SYM)
            IO_TYPE(D_I8|Q_ASYM,    D_F16)
            IO_TYPE(D_I8|Q_SYM,     D_F16)
            IO_TYPE(D_I8|Q_ASYM,    D_F32)
            IO_TYPE(D_I8|Q_SYM,     D_F32)
            IO_TYPE(D_I8|Q_DFP,     D_I8|Q_DFP)
            IO_TYPE(D_I8|Q_DFP,     D_F16)
            IO_TYPE(D_I8|Q_DFP,     D_F32)


            IO_TYPE(D_I16|Q_ASYM,   D_I16|Q_ASYM)
            IO_TYPE(D_I16|Q_SYM,    D_I16|Q_SYM)
            IO_TYPE(D_I16|Q_ASYM,   D_F16)
            IO_TYPE(D_I16|Q_SYM,    D_F16)
            IO_TYPE(D_I16|Q_ASYM,   D_F32)
            IO_TYPE(D_I16|Q_SYM,    D_F32)
            IO_TYPE(D_I16|Q_DFP,    D_F32)
            IO_TYPE(D_I16|Q_DFP,    D_F16)
            IO_TYPE(D_I16|Q_DFP,    D_I16|Q_DFP)
        END_IO_TYPE_DECL(SOFTMAX)
        if (!VALIDATE_OP_IO_TYPES(SOFTMAX, self, inputs, self->input.num, outputs, self->output.num)) {
            char* desc = generate_op_io_types_desc(inputs,
                    self->input.num, outputs, self->output.num);
            VSILOGE("Inputs/Outputs data type not support: %s", desc);
            destroy_op_io_types_desc(desc);
            return FALSE;
        }
    }

    return TRUE;
} /* op_check() */

#define VSI_NN_SOFTMAX_DEFAULT_AXIS     (10000)
static vsi_status op_optimize
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_opt_direction_e direction
    )
{
    if (VSI_NN_OPTIMIZE_BACKWARD == direction)
    {
        return VSI_SUCCESS;
    }

    return vsi_nn_internal_optimize_node( self, direction );
} /* op_optimize() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_softmax_param * p = &(self->nn_param.softmax);
    if (p->local.reshaped_input)
    {
        vsi_nn_ReleaseTensor(&(p->local.reshaped_input));
    }
    if (p->local.reshaped_output)
    {
        vsi_nn_ReleaseTensor(&(p->local.reshaped_output));
    }

    vsi_nn_internal_deinit_node_wksp( self );
    return status;
} /* op_deinit() */

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    if (self->nn_param.softmax.beta == 0.f)
    {
        self->nn_param.softmax.beta = 1.f;
    }

    return status;
} /* op_init() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_internal_node_t* curr = NULL;
    if( NULL == self )
    {
        return FALSE;
    }

    if (vsi_nn_compareVersion(self->graph, 1, 1, 7) == -1)
    {
        if (inputs[0]->attr.dim_num < 3)
        {
            self->nn_param.softmax.axis = 0;
        }
        else
        {
            self->nn_param.softmax.axis = 2;
        }
    }

    if (self->nn_param.softmax.axis < 0)
        self->nn_param.softmax.axis += (int32_t)inputs[0]->attr.dim_num;

    if (self->nn_param.softmax.axis < 0)
    {
        VSILOGD("SoftMax Invalid Axis: %d", self->nn_param.softmax.axis);
        return FALSE;
    }

    vsi_nn_internal_init_node_wksp(self);
    curr = vsi_nn_internal_new_node(self, VSI_NN_OP_SOFTMAX_INTERNAL, 0, 0);
    curr->inputs[0] = inputs[0];
    curr->outputs[0] = outputs[0];
    curr->node->nn_param.softmax_internal.beta = self->nn_param.softmax.beta;
    curr->node->nn_param.softmax_internal.axis = self->nn_param.softmax.axis;
    vsi_nn_internal_setup_node(self, curr);

    return TRUE;
}

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ SOFTMAX,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ 1,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif
