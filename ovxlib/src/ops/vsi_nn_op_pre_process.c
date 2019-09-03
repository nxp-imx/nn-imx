/****************************************************************************
*
*    Copyright (c) 2019 Vivante Corporation
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
#include "vsi_nn_log.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_prv.h"
#include "utils/vsi_nn_math.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "client/vsi_nn_vxkernel.h"
#include "vsi_nn_internal_node.h"

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
    /*TODO: Check tensor shapes. */
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
    /* TODO: Add code to comput outputs' shape. */
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_pre_process_param * p;

    p = (vsi_nn_pre_process_param *)&(self->nn_param.pre_process);

    vsi_nn_init_internal_node_wksp( self );

    if (p->type == VSI_NN_PRE_PROCESS_TENSOR)
    {
        curr = vsi_nn_new_internal_node( self, VSI_NN_OP_PRE_PROCESS_TENSOR, 0, 0 );

        curr->node->nn_param.pre_process_tensor.perm = p->perm;
        curr->node->nn_param.pre_process_tensor.dim_num = p->dim_num;

        curr->inputs[0] = inputs[PRE_PROCESS_INPUT0];
        curr->outputs[0] = outputs[PRE_PROCESS_OUTPUT];

        vsi_nn_setup_internal_node_op(self, curr);
    }
    else if (p->type == VSI_NN_PRE_PROCESS_GRAY)
    {
        curr = vsi_nn_new_internal_node( self, VSI_NN_OP_PRE_PROCESS_GRAY, 0, 0 );

        curr->node->nn_param.pre_process_gray.mean = p->norm.mean[0];
        curr->node->nn_param.pre_process_gray.scale = p->norm.scale;
        curr->node->nn_param.pre_process_gray.rect.left = p->rect.left;
        curr->node->nn_param.pre_process_gray.rect.top = p->rect.top;
        curr->node->nn_param.pre_process_gray.rect.width = p->rect.width;
        curr->node->nn_param.pre_process_gray.rect.height = p->rect.height;
        curr->node->nn_param.pre_process_gray.output_attr.size = p->output_attr.size;
        curr->node->nn_param.pre_process_gray.output_attr.dim_num = p->output_attr.dim_num;

        curr->inputs[0] = inputs[PRE_PROCESS_INPUT0];
        curr->outputs[0] = outputs[PRE_PROCESS_OUTPUT];

        vsi_nn_setup_internal_node_op(self, curr);
    }
    else if (p->type == VSI_NN_PRE_PROCESS_RGB)
    {
        curr = vsi_nn_new_internal_node( self, VSI_NN_OP_PRE_PROCESS_RGB, 0, 0 );

        if (p->reverse_channel)
        {
            curr->node->nn_param.pre_process_rgb.r_mean = p->norm.mean[2];
            curr->node->nn_param.pre_process_rgb.g_mean = p->norm.mean[1];
            curr->node->nn_param.pre_process_rgb.b_mean = p->norm.mean[0];
        }
        else
        {
            curr->node->nn_param.pre_process_rgb.r_mean = p->norm.mean[0];
            curr->node->nn_param.pre_process_rgb.g_mean = p->norm.mean[1];
            curr->node->nn_param.pre_process_rgb.b_mean = p->norm.mean[2];
        }

        curr->node->nn_param.pre_process_rgb.rgb_scale = p->norm.scale;
        curr->node->nn_param.pre_process_rgb.reverse_channel = p->reverse_channel;
        curr->node->nn_param.pre_process_rgb.rect.left = p->rect.left;
        curr->node->nn_param.pre_process_rgb.rect.top = p->rect.top;
        curr->node->nn_param.pre_process_rgb.rect.width = p->rect.width;
        curr->node->nn_param.pre_process_rgb.rect.height = p->rect.height;
        curr->node->nn_param.pre_process_rgb.output_attr.size = p->output_attr.size;
        curr->node->nn_param.pre_process_rgb.output_attr.dim_num = p->output_attr.dim_num;
        curr->node->nn_param.pre_process_rgb.perm = p->perm;
        curr->node->nn_param.pre_process_rgb.dim_num = p->dim_num;

        curr->inputs[0] = inputs[PRE_PROCESS_INPUT0];
        curr->outputs[0] = outputs[PRE_PROCESS_OUTPUT];

        vsi_nn_setup_internal_node_op(self, curr);
    }
    else if (p->type == VSI_NN_PRE_PROCESS_YUV420)
    {
        curr = vsi_nn_new_internal_node( self, VSI_NN_OP_PRE_PROCESS_YUV420, 0, 0 );

        if (p->reverse_channel)
        {
            curr->node->nn_param.pre_process_rgb.r_mean = p->norm.mean[2];
            curr->node->nn_param.pre_process_rgb.g_mean = p->norm.mean[1];
            curr->node->nn_param.pre_process_rgb.b_mean = p->norm.mean[0];
        }
        else
        {
            curr->node->nn_param.pre_process_rgb.r_mean = p->norm.mean[0];
            curr->node->nn_param.pre_process_rgb.g_mean = p->norm.mean[1];
            curr->node->nn_param.pre_process_rgb.b_mean = p->norm.mean[2];
        }

        curr->node->nn_param.pre_process_rgb.rgb_scale = p->norm.scale;
        curr->node->nn_param.pre_process_rgb.reverse_channel = p->reverse_channel;
        curr->node->nn_param.pre_process_rgb.rect.left = p->rect.left;
        curr->node->nn_param.pre_process_rgb.rect.top = p->rect.top;
        curr->node->nn_param.pre_process_rgb.rect.width = p->rect.width;
        curr->node->nn_param.pre_process_rgb.rect.height = p->rect.height;
        curr->node->nn_param.pre_process_rgb.output_attr.size = p->output_attr.size;
        curr->node->nn_param.pre_process_rgb.output_attr.dim_num = p->output_attr.dim_num;
        curr->node->nn_param.pre_process_rgb.perm = p->perm;
        curr->node->nn_param.pre_process_rgb.dim_num = p->dim_num;

        curr->inputs[0] = inputs[PRE_PROCESS_INPUT0];
        curr->inputs[1] = inputs[PRE_PROCESS_INPUT1];
        curr->inputs[2] = inputs[PRE_PROCESS_INPUT2];
        curr->outputs[0] = outputs[PRE_PROCESS_OUTPUT];

        vsi_nn_setup_internal_node_op(self, curr);
    }
    else if (p->type == VSI_NN_PRE_PROCESS_BGRA)
    {
        curr = vsi_nn_new_internal_node( self, VSI_NN_OP_PRE_PROCESS_BGRA, 0, 0 );

        if (p->reverse_channel)
        {
            curr->node->nn_param.pre_process_bgra.r_mean = p->norm.mean[2];
            curr->node->nn_param.pre_process_bgra.g_mean = p->norm.mean[1];
            curr->node->nn_param.pre_process_bgra.b_mean = p->norm.mean[0];
        }
        else
        {
            curr->node->nn_param.pre_process_bgra.r_mean = p->norm.mean[0];
            curr->node->nn_param.pre_process_bgra.g_mean = p->norm.mean[1];
            curr->node->nn_param.pre_process_bgra.b_mean = p->norm.mean[2];
        }

        curr->node->nn_param.pre_process_bgra.rgb_scale = p->norm.scale;
        curr->node->nn_param.pre_process_bgra.reverse_channel = p->reverse_channel;
        curr->node->nn_param.pre_process_bgra.rect.left = p->rect.left;
        curr->node->nn_param.pre_process_bgra.rect.top = p->rect.top;
        curr->node->nn_param.pre_process_bgra.rect.width = p->rect.width;
        curr->node->nn_param.pre_process_bgra.rect.height = p->rect.height;
        curr->node->nn_param.pre_process_bgra.output_attr.size = p->output_attr.size;
        curr->node->nn_param.pre_process_bgra.output_attr.dim_num = p->output_attr.dim_num;
        curr->node->nn_param.pre_process_bgra.perm = p->perm;
        curr->node->nn_param.pre_process_bgra.dim_num = p->dim_num;

        curr->inputs[0] = inputs[PRE_PROCESS_INPUT0];
        curr->outputs[0] = outputs[PRE_PROCESS_OUTPUT];

        vsi_nn_setup_internal_node_op(self, curr);
    }
    else
    {
        VSILOGE( "Not support this type!(PRE_PROCESS)\n");
        return FALSE;
    }

    return TRUE;
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
    /* op_name    */ PRE_PROCESS,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ PRE_PROCESS_INPUT_CNT,
    /* output_num */ PRE_PROCESS_OUTPUT_CNT
    );
#ifdef __cplusplus
}
#endif
