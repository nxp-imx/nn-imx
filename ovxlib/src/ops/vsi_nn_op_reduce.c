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
#include <string.h>
#include <stdlib.h>
#include "vsi_nn_platform.h"

#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "client/vsi_nn_vxkernel.h"

#define _ARG_NUM            (6)
#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)

#define USE_OVX_API TRUE

#if (USE_OVX_API == FALSE)
extern vx_kernel_description_t * vx_kernel_REDUCE_list[];

static void _set_inputs_outputs
    (
    vx_reference * params,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vx_uint32 i;
    vx_uint32 cnt;

    /* Set inputs */
    cnt = 0;
    for( i = 0; i < _INPUT_NUM; i ++, cnt ++ )
    {
        params[cnt] = (vx_reference)inputs[i]->t;
    }

    /* Set outputs */
    for( i = 0; i < _OUTPUT_NUM; i ++, cnt ++ )
    {
        params[cnt] = (vx_reference)outputs[i]->t;
    }
} /* _set_inputs_outputs() */

static vx_status _create_params
    (
    vsi_nn_node_t * node,
    vx_reference * params,
    vx_uint32 num
    )
{
    vx_status status;
    vx_context ctx;
    vsi_nn_reduce_param * p;
    if( 0 == num )
    {
        return VX_SUCCESS;
    }
    memset( params, 0, sizeof( vx_reference * ) * num );
    p = &node->nn_param.reduce;
    ctx = vxGetContext( (vx_reference)node->graph->g );
    /* Init parameters */
#define _SET_PARAM( i, type, arg ) do{ \
    params[i] = (vx_reference)vxCreateScalar( ctx, type, &p->arg ); \
    status = vxGetStatus( params[i] ); \
    if( VX_SUCCESS != status ) { \
    goto set_param_error; \
    } \
    } while(0)
    _SET_PARAM( 0, VX_TYPE_INT32, axis_num );
    _SET_PARAM( 1, VX_TYPE_INT32, keep_dim );
    _SET_PARAM( 2, VX_TYPE_INT32, axis[0] );
    _SET_PARAM( 3, VX_TYPE_INT32, axis[1] );
    _SET_PARAM( 4, VX_TYPE_INT32, axis[2] );
    _SET_PARAM( 5, VX_TYPE_INT32, axis[3] );

#undef _SET_PARAM
set_param_error:

    return status;
} /* _create_params */

static void _release_params
    (
    vx_reference * params,
    vx_uint32 num
    )
{
    vx_uint32 i;
    vx_scalar scalar;
    for( i = 0; i < num; i ++ )
    {
        scalar = (vx_scalar)params[i];
        vxReleaseScalar( &scalar );
    }
} /* _release_params() */

static vx_status cpu_op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vx_status status = VX_SUCCESS;
    vx_reference params[_PARAM_NUM];
    vx_reference * args;

    args = &params[_IO_NUM];

    if( NULL == self->n )
    {
        return VX_FAILURE;
    }

    /* Set inputs and outputs */
    _set_inputs_outputs( params, inputs, outputs );

    /* Init parameters. */
    _create_params( self, args, _ARG_NUM );

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, _PARAM_NUM );

    _release_params( args, _ARG_NUM );

    return status;
}

static vsi_nn_op_compute_t op_compute_list[] =
{
    cpu_op_compute,
    NULL
};
#endif

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;
#if (USE_OVX_API == TRUE)
    if (self->nn_param.reduce.type == VSI_NN_REDUCE_MEAN)
    {
        vx_nn_mean_params_t para;
        vsi_nn_tensor_t *axis_tensor = NULL;
        vsi_nn_tensor_attr_t attr;

        memset(&attr, 0, sizeof(attr));
        attr.size[0] = self->nn_param.reduce.axis_num;
        attr.dim_num = 1;
        attr.is_const = TRUE;
        attr.dtype.vx_type = VSI_NN_TYPE_UINT32;
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        axis_tensor = vsi_nn_CreateTensorFromData(
            self->graph,
            (uint8_t *)self->nn_param.reduce.axis,
            &attr);
        if( NULL == axis_tensor )
        {
            VSILOGE("Create axis_tensor fail.(reduce)");
            return VSI_FAILURE;
        }

        self->nn_param.reduce.local.axis_tensor = axis_tensor;
        para.axis = REQUIRED_IO(axis_tensor);
        para.keep_dims = self->nn_param.reduce.keep_dim;

        self->n = vxTensorMeanNode( self->graph->g, inputs[0]->t, &para,
            sizeof(vx_nn_mean_params_t), outputs[0]->t );
    }
    else if (self->nn_param.reduce.type == VSI_NN_REDUCE_SUM)
    {
        self->n = vxTensorReduceSumNode( self->graph->g, inputs[0]->t, outputs[0]->t,
            self->nn_param.reduce.axis, self->nn_param.reduce.axis_num,
            self->nn_param.reduce.keep_dim);
    }
    if( NULL != self->n )
    {
        status = VSI_SUCCESS;
    }
#else
    vsi_nn_kernel_info_t kernel_info = {0};

    kernel_info.resource_num = 1;
    kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
    kernel_info.resource_name[0] = "vsi_nn_kernel_reduce";
    kernel_info.type = VX_KERNEL_TYPE_CPU;
    kernel_info.kernel = vx_kernel_REDUCE_list;
    kernel_info.kernel_index = 0;
    kernel_info.init_index = 0;

    self->n = vsi_nn_RegisterClientKernelAndNewNode(
        self->graph, &kernel_info);
    if (kernel_info.resource_name) free(kernel_info.resource_name);
    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }
    if (NULL != op_compute_list[kernel_info.init_index])
    {
        status = op_compute_list[kernel_info.init_index](self, inputs, outputs);
    }
#endif
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
    return vx_true_e;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    if (self->nn_param.reduce.type != VSI_NN_REDUCE_MEAN &&
        self->nn_param.reduce.type != VSI_NN_REDUCE_SUM)
    {
        VSILOGE("The type of reduce is not supported now.(reduce)");
        return vx_false_e;
    }
    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        int valid_dim_num = inputs[0]->attr.dim_num;
        uint32_t i;
        char dim_map[VSI_NN_MAX_DIM_NUM] = {0};

        for (i = 0; i < self->nn_param.reduce.axis_num; i++)
        {
            int index = self->nn_param.reduce.axis[i];
            if (dim_map[index] == 0) {
                dim_map[index] = 1;
                valid_dim_num --;
            }
        }

        if (self->nn_param.reduce.keep_dim)
        {
            outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
            for (i = 0; i < inputs[0]->attr.dim_num; i++)
            {
                if (dim_map[i] == 0)
                {
                    outputs[0]->attr.size[i] = inputs[0]->attr.size[i];
                }
                else
                {
                    outputs[0]->attr.size[i] = 1;
                }
            }
        }
        else
        {
            int index = 0;
            if (valid_dim_num == 0)
            {
                outputs[0]->attr.dim_num = 1;
                outputs[0]->attr.size[0] = 1;
            }
            else
            {
                outputs[0]->attr.dim_num = valid_dim_num;
                for (i = 0; i < inputs[0]->attr.dim_num; i++)
                {
                    if (dim_map[i] == 0)
                    {
                        outputs[0]->attr.size[index] = inputs[0]->attr.size[i];
                        index++;
                    }
                }
            }
        }
    }
    return vx_true_e;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    if (self->nn_param.reduce.local.axis_tensor != NULL)
    {
        vsi_nn_ReleaseTensor(&(self->nn_param.reduce.local.axis_tensor));
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
    /* op_name    */ REDUCE,
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

