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
#include "vsi_nn_log.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_prv.h"
#include "utils/vsi_nn_math.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "client/vsi_nn_vxkernel.h"

#define _ARG_NUM            (1)
#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (VSI_NN_UNSTACK_MAX_OUTPUTS)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)

extern vx_kernel_description_t * vx_kernel_UNSTACK_list[];

static void _set_inputs_outputs
    (
    vsi_nn_node_t * self,
    vx_reference * params,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    uint32_t i;
    vsi_nn_unstack_lcl_data *data = NULL;
    vx_tensor *tensors = NULL;
    uint32_t num = self->input.num;

    tensors = (vx_tensor *)malloc(sizeof(vx_tensor) * num);
    if(NULL == tensors)
    {
        VSILOGE("unstack op malloc fail.");
        return;
    }

    for(i = 0; i < num; i++)
    {
        tensors[i] = outputs[i]->t;
    }
    data = (vsi_nn_unstack_lcl_data *)malloc(sizeof(vsi_nn_unstack_lcl_data));
    data->array = vxCreateTensorObjectArray(self->graph->ctx->c,
                    num, tensors);

    params[0] = (vx_reference)inputs[0]->t;
    params[1] = (vx_reference)data->array;

    self->nn_param.unstack.lcl_data = data;

    if (tensors) free(tensors);
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
    vsi_nn_unstack_param * p;
    if( 0 == num )
    {
        return VSI_SUCCESS;
    }
    memset( params, 0, sizeof( vx_reference * ) * num );
    p = &(node->nn_param.unstack);
    ctx = vxGetContext( (vx_reference)node->graph->g );
    /* Init parameters */
    #define _SET_PARAM( i, type, arg ) do{ \
        params[i] = (vx_reference)vxCreateScalar( ctx, type, &p->arg ); \
        status = vxGetStatus( params[i] ); \
        if( VSI_SUCCESS != status ) { \
            goto set_param_error; \
            } \
        } while(0)
    _SET_PARAM( 0, VX_TYPE_INT32, axis );
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
    vx_reference * args;
    vx_reference params[_PARAM_NUM];

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

    _release_params( args, _ARG_NUM );

    return status;
}

static vsi_nn_op_compute_t op_compute_list[] =
{
    cpu_op_compute,
    vx_op_compute,
    NULL
};

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
#define NN_MAX_INPUT_SIZE (65536)
    vsi_status status  = VSI_FAILURE;

    uint32_t axis;
    uint32_t i = 0;
    uint32_t tensor_num = self->output.num;
    vsi_nn_unstack_param * p;
    vsi_nn_tensor_t * perm_tensor = NULL;
    vx_tensor input_rs = NULL;
    vsi_nn_tensor_attr_t attr;
    uint32_t sizes[VSI_NN_MAX_DIM_NUM] = {0};
    uint32_t perm[VSI_NN_MAX_DIM_NUM] = {0};
    uint32_t start[VSI_NN_MAX_DIM_NUM] = { 0 };
    uint32_t end[VSI_NN_MAX_DIM_NUM] = { 0 };
    vx_tensor_view view;
    uint32_t block_size = 1;
    uint32_t block_num = 1;
    uint32_t dims = 2;

    p = (vsi_nn_unstack_param *)&(self->nn_param.unstack);
    axis = p->axis;

    for (i = 0; i < axis; i++)
    {
        block_size *= inputs[0]->attr.size[i];
    }

    for (i = axis + 1; i < inputs[0]->attr.dim_num; i++)
    {
        block_num *= inputs[0]->attr.size[i];
    }

    sizes[0] = block_size * tensor_num;
    sizes[1] = block_num;

    if (block_size * tensor_num < NN_MAX_INPUT_SIZE && block_num < NN_MAX_INPUT_SIZE)
    {
        input_rs = vxReshapeTensor(inputs[0]->t, (int32_t*)sizes, dims);

        sizes[0] = block_num;
        sizes[1] = block_size * tensor_num;
        memcpy(&attr, &(inputs[0]->attr), sizeof(attr));
        memcpy(attr.size, sizes,  VSI_NN_MAX_DIM_NUM * sizeof( uint32_t ));
        attr.dim_num = dims;
        attr.vtl = TRUE;
        perm_tensor = vsi_nn_CreateTensor(self->graph, &attr);

        /* Create tensor from view */
        memset( start, 0, sizeof( uint32_t ) * VSI_NN_MAX_DIM_NUM );
        memset( end, 0, sizeof( uint32_t ) * VSI_NN_MAX_DIM_NUM );
        end[0] = block_num;

        perm[0] = 1;
        perm[1] = 0;

        self->n = vxTensorPermuteNode(
            self->graph->g,
            input_rs,
            perm_tensor->t,
            perm,
            dims
            );
        if( NULL != self->n )
        {
            status = VSI_SUCCESS;
        }

        for (i = 0; i < tensor_num; i++)
        {
            vx_tensor in_view_tensor;
            vx_tensor output_rs = NULL;

            sizes[0] = block_size;
            sizes[1] = block_num;
            output_rs = vxReshapeTensor(outputs[i]->t, (int32_t*)sizes, dims);

            start[1] = end[1];
            end[1] += block_size;

            view = vxCreateTensorView( self->graph->ctx->c,
                start, end, dims );
            if( NULL == view )
            {
                VSILOGE( "Create tensor %d view fail.", i );
                status = VSI_FAILURE;
                break;
            }

            in_view_tensor = vxCreateTensorFromView( perm_tensor->t, view );
            vxReleaseTensorView( &view );
            if( NULL == in_view_tensor )
            {
                VSILOGE( "Create tensor %d from view fail.", i );
                status = VSI_FAILURE;
                break;
            }

            self->n = vxTensorPermuteNode(
                self->graph->g,
                in_view_tensor,
                output_rs,
                perm,
                dims
                );
            if( NULL != self->n )
            {
                status = VSI_SUCCESS;
            }

            if (in_view_tensor) vxReleaseTensor(&in_view_tensor);
            if (output_rs) vxReleaseTensor(&output_rs);
            if (self->n) vxReleaseNode(&self->n);
        }
    }
    else
    {
        vsi_nn_kernel_info_t kernel_info = {0};

        status = VSI_FAILURE;
        kernel_info.type = VX_KERNEL_TYPE_CPU;
        kernel_info.kernel = vx_kernel_UNSTACK_list;
        kernel_info.resource_num = 1;
        kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
        kernel_info.resource_name[0] = "vsi_nn_kernel_unstack";

        if( kernel_info.type == VX_KERNEL_TYPE_VX)
        {
            kernel_info.kernel_index = 1;
            kernel_info.init_index = 1;
        }
        else /*kernel_info.type = VX_KERNEL_TYPE_CPU;*/
        {
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
    vsi_nn_node_t * node,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_unstack_param * p;
    uint32_t i, j;

    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        p = (vsi_nn_unstack_param *)&(node->nn_param.unstack);
        if(p->axis == 0)
        {
            for(i = 0; i < inputs[0]->attr.dim_num-1; i++)
            {
                for(j = 0; j < node->output.num; j++)
                {
                    outputs[j]->attr.size[i] = inputs[0]->attr.size[i + 1];
                }
            }
        }
        else if(p->axis == 1)
        {
            for(j = 0; j < node->output.num; j++)
            {
                outputs[j]->attr.size[0] = inputs[0]->attr.size[0];

                for(i = 1; i < inputs[0]->attr.dim_num-1; i++)
                {
                    outputs[j]->attr.size[i] = inputs[0]->attr.size[i + 1];
                }
            }
        }
        else if(p->axis == 2)
        {
            for(j = 0; j < node->output.num; j++)
            {
                outputs[j]->attr.size[0] = inputs[0]->attr.size[0];
                outputs[j]->attr.size[1] = inputs[0]->attr.size[1];

                for(i = 2; i < inputs[0]->attr.dim_num-1; i++)
                {
                    outputs[j]->attr.size[i] = inputs[0]->attr.size[i + 1];
                }
            }
        }
        else if(p->axis == 3)
        {
            for(j = 0; j < node->output.num; j++)
            {
                outputs[j]->attr.size[0] = inputs[0]->attr.size[0];
                outputs[j]->attr.size[1] = inputs[0]->attr.size[1];
                outputs[j]->attr.size[2] = inputs[0]->attr.size[2];
            }
        }
    }
    return TRUE;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_nn_unstack_lcl_data * data;

    if(NULL == self)
    {
        return VSI_FAILURE;
    }

    data = self->nn_param.unstack.lcl_data;
    if(data && data->array)
    {
        vxReleaseObjectArray(&data->array);
        free(data);
        data = NULL;
    }
    vsi_nn_op_common_deinit(self);

    return VSI_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ UNSTACK,
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
