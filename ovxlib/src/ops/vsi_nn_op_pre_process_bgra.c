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
#include "vsi_nn_log.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_prv.h"
#include "utils/vsi_nn_math.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "client/vsi_nn_vxkernel.h"

#define _ARG_NUM            (10)
#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)

extern vx_kernel_description_t * vx_kernel_PRE_PROCESS_BGRA_list[];

static void _set_inputs_outputs
    (
    vsi_nn_node_t * self,
    vx_reference * params,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    uint32_t i;
    uint32_t cnt;
    vsi_nn_pre_process_bgra_param * p;

    p = (vsi_nn_pre_process_bgra_param *)&(self->nn_param.pre_process_bgra);

    /* Set inputs */
    cnt = 0;
    for( i = 0; i < _INPUT_NUM; i ++, cnt ++ )
    {
        params[cnt] = (vx_reference)inputs[i]->t;
    }

    /* Set outputs */
    if (p->local.enable_perm)
    {
        uint32_t sizes[VSI_NN_MAX_DIM_NUM] = {1};
        uint32_t num_of_dims = vsi_nn_max(outputs[0]->attr.dim_num, 2);

        memcpy(sizes, outputs[0]->attr.size, sizeof(uint32_t) * VSI_NN_MAX_DIM_NUM);

        sizes[0] = sizes[0] * sizes[1];
        sizes[1] = sizes[2];
        sizes[2] = 1;

        p->local.local_tensor =
            vxReshapeTensor(outputs[0]->t, (int32_t *)sizes, num_of_dims);

        params[1] = (vx_reference)p->local.local_tensor;
    }
    else
    {
        params[1] = (vx_reference)outputs[0]->t;
    }
} /* _set_inputs_outputs() */

static vsi_status _create_params
    (
    vsi_nn_node_t * node,
    vx_reference * params,
    uint32_t num
    )
{
    vsi_status status;
    vx_context ctx;
    vsi_nn_pre_process_bgra_param * p;
    if( 0 == num )
    {
        return VSI_SUCCESS;
    }
    memset( params, 0, sizeof( vx_reference * ) * num );
    p = &(node->nn_param.pre_process_bgra);
    ctx = vxGetContext( (vx_reference)node->graph->g );
    /* Init parameters */
    #define _SET_PARAM( i, type, arg ) do{ \
        params[i] = (vx_reference)vxCreateScalar( ctx, type, &p->arg ); \
        status = vxGetStatus( params[i] ); \
        if( VSI_SUCCESS != status ) { \
            goto set_param_error; \
            } \
        } while(0)
    _SET_PARAM( 0, VX_TYPE_INT32, local.scale_x);
    _SET_PARAM( 1, VX_TYPE_INT32, local.scale_y);
    _SET_PARAM( 2, VX_TYPE_INT32, rect.left);
    _SET_PARAM( 3, VX_TYPE_INT32, rect.top);
    _SET_PARAM( 4, VX_TYPE_FLOAT32, r_mean);
    _SET_PARAM( 5, VX_TYPE_FLOAT32, g_mean);
    _SET_PARAM( 6, VX_TYPE_FLOAT32, b_mean);
    _SET_PARAM( 7, VX_TYPE_FLOAT32, rgb_scale);
    _SET_PARAM( 8, VX_TYPE_INT32, reverse_channel);
    _SET_PARAM( 9, VX_TYPE_INT32, local.enable_perm);
    #undef _SET_PARAM
set_param_error:

    return status;
} /* _create_params */

static void _release_params
    (
    vx_reference * params,
    uint32_t num
    )
{
    uint32_t i;
    vx_scalar scalar;
    for( i = 0; i < num; i ++ )
    {
        scalar = (vx_scalar)params[i];
        vxReleaseScalar( &scalar );
    }
} /* _release_params() */

static vsi_status cpu_op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_reference params[_PARAM_NUM];
    vx_reference * args;

    args = &params[_IO_NUM];

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    /* Set inputs and outputs */
    _set_inputs_outputs( self, params, inputs, outputs );

    /* Init parameters. */
    _create_params( self, args, _ARG_NUM );

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, _PARAM_NUM );

    _release_params( args, _ARG_NUM );

    return status;
}

static vsi_status vx_op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_reference params[_PARAM_NUM];
    vx_reference * args;
    vx_border_t border;

    args = &params[_IO_NUM];

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    /* Set inputs and outputs */
    _set_inputs_outputs( self, params, inputs, outputs );
    /*TODO: Add code if need to change your parameter*/

    /* Init parameters. */
    _create_params( self, args, _ARG_NUM );

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, _PARAM_NUM );

    border.mode = VX_BORDER_REPLICATE;
    border.constant_value.U32 = 0;
    status |= vxSetNodeAttribute(self->n, VX_NODE_BORDER, &border, sizeof(border));

    _release_params( args, _ARG_NUM );

    return status;
}

static vsi_nn_op_compute_t op_compute_list[] =
{
    cpu_op_compute,
    vx_op_compute,
    NULL
};

static vsi_status vx_op_pre_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_kernel_info_t * kernel_info
    )
{
    vsi_nn_type_e inputDataFormat     = inputs[0]->attr.dtype.vx_type;
    vsi_nn_type_e outputDataFormat    = outputs[0]->attr.dtype.vx_type;

    vsi_nn_pre_process_bgra_param * p;
    p = &(self->nn_param.pre_process_bgra);

    if (inputDataFormat == VSI_NN_TYPE_UINT8 && outputDataFormat == VSI_NN_TYPE_UINT8)
    {
        if(p->local.enable_copy)
            kernel_info->kernel_index = 1;
        else if(p->local.enable_perm)
        {
            kernel_info->resource_name[0] = "vsi_nn_kernel_pre_process_bgra_trans";
            kernel_info->kernel_index = 3;
        }
        else
            kernel_info->kernel_index = 2;
    }
    else
    {
        VSILOGE("Not support input or output data format!(preprocess bgra) at [%s : %d]\n", __FILE__, __LINE__);
        return VSI_FAILURE;
    }

    return VSI_SUCCESS;
}

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status;
    vsi_nn_kernel_info_t kernel_info;

    memset(&kernel_info, 0x0, sizeof(vsi_nn_kernel_info_t));
    status = VSI_FAILURE;
    kernel_info.type = vsi_nn_GetVXKernelTypeForShader();
    kernel_info.kernel = vx_kernel_PRE_PROCESS_BGRA_list;
    kernel_info.resource_num = 1;
    kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
    kernel_info.resource_name[0] = "vsi_nn_kernel_pre_process_bgra";

    if( kernel_info.type == VX_KERNEL_TYPE_VX)
    {
        kernel_info.kernel_index = 1;
        kernel_info.init_index = 1;
        vx_op_pre_compute(self, inputs, outputs, &kernel_info);
    }
    else /*kernel_info.type = VX_KERNEL_TYPE_CPU;*/
    {
        kernel_info.type = VX_KERNEL_TYPE_CPU;
        kernel_info.kernel_index = 0;
        kernel_info.init_index = 0;
    }

    self->n = vsi_nn_RegisterClientKernelAndNewNode(
            self->graph, &kernel_info);
    if (kernel_info.resource_name)
    {
        free(kernel_info.resource_name);
    }
    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }
    if (NULL != op_compute_list[kernel_info.init_index])
    {
        status = op_compute_list[kernel_info.init_index](self, inputs, outputs);
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
    /*TODO: Check tensor shapes. */
    return TRUE;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    /* TODO: Add code to comput outputs' shape. */
    vsi_nn_pre_process_bgra_param * p;
    uint32_t axis;
    uint32_t i;
    p = (vsi_nn_pre_process_bgra_param *)&(self->nn_param.pre_process_bgra);

    if (p->rect.width == 0 || p->rect.height == 0)
    {
        VSILOGE("Image size cannot be zero !(PRE_PROCESS_BGRA)\n");
        return FALSE;
    }
    else
    {
        for (i = 0; i < p->output_attr.dim_num; i++)
        {
            if (p->output_attr.size[i] == 0)
            {
                VSILOGE("output size cannot be zero!(PRE_PROCESS_BGRA)\n");
                return FALSE;
            }
        }
    }

    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        if (p->output_attr.dim_num > 0)
        {
            for (i = 0; i < p->output_attr.dim_num; i++)
            {
                if (p->output_attr.size[i] == 0)
                {
                    VSILOGE("output size cannot be zero!(PRE_PROCESS_BGRA)\n");
                    return FALSE;
                }
                else
                {
                    outputs[0]->attr.dim_num = p->output_attr.dim_num;
                    outputs[0]->attr.size[i] = p->output_attr.size[i];
                }
            }
        }
        else
        {
            VSILOGE("output dim num cannot be zero!(PRE_PROCESS_BGRA)\n");
            return FALSE;
        }
    }

    for (i = 0; i < self->nn_param.pre_process_bgra.dim_num; i++)
    {
        axis = self->nn_param.pre_process_bgra.perm[i];
        if (axis != i)
            break;
    }

    if (i == self->nn_param.pre_process_bgra.dim_num)
        self->nn_param.pre_process_bgra.local.enable_perm = FALSE;
    else
        self->nn_param.pre_process_bgra.local.enable_perm = TRUE;

    if (self->nn_param.pre_process_bgra.local.enable_perm == FALSE)
    {
        p->local.scale_x = (p->rect.width << 15) / outputs[0]->attr.size[0];
        p->local.scale_y = (p->rect.height << 15) / outputs[0]->attr.size[1];
    }
    else
    {
        if(outputs[0]->attr.size[2] < 2)
        {
            p->local.scale_x = (p->rect.width << 15) / (outputs[0]->attr.size[0] / 3);
            p->local.scale_y = (p->rect.height << 15) / outputs[0]->attr.size[1];
        }
        else
        {
            p->local.scale_x = (p->rect.width << 15) / outputs[0]->attr.size[0];
            p->local.scale_y = (p->rect.height << 15) / outputs[0]->attr.size[1];
        }
    }

    p->local.enable_copy = ((p->local.scale_x == p->local.scale_y) && (p->local.scale_x == (1 << 15)));

    return TRUE;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    if (self->nn_param.pre_process_bgra.local.local_tensor != NULL)
    {
        vxReleaseTensor(&self->nn_param.pre_process_bgra.local.local_tensor);
        self->nn_param.pre_process_bgra.local.local_tensor = NULL;
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
    /* op_name    */ PRE_PROCESS_BGRA,
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
