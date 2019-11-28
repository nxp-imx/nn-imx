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
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_link_list.h"
#include "utils/vsi_nn_dtype_util.h"
#include "client/vsi_nn_vxkernel.h"

#define _ARG_NUM            (1)
#define _INPUT_NUM          VSI_NN_STACK_MAX_INPUTS
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (2)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)

extern vx_kernel_description_t * vx_kernel_STACK_list[];


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

static vsi_bool _is_same_quant
    (
    vsi_nn_node_t   * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    uint32_t i,num;
    vsi_nn_dtype_t *dtype,*_dtype;

    dtype = NULL;
    /* check inputs dtype */
    num = _get_input_num(self, inputs);
    for(i = 0; i < num; i++)
    {
        if(NULL == dtype)
        {
            dtype = &inputs[i]->attr.dtype;
            continue;
        }

        _dtype = &inputs[i]->attr.dtype;
        if(vsi_nn_DtypeCompare(dtype, _dtype) == FALSE)
        {
            return FALSE;
        }

        dtype = _dtype;
    }

    /* check outputs dtype */
    _dtype = &outputs[0]->attr.dtype;
    if(vsi_nn_DtypeCompare(dtype, _dtype) == FALSE)
    {
        return FALSE;
    }

    return TRUE;
} /* _is_same_quant */

static vsi_status copy_tensor_to_view
    (
    vsi_nn_node_t   * self,
    vsi_nn_tensor_t * src_in,
    uint32_t         axis,
    vx_tensor         dst_tensor
    )
{
    vsi_status ret;
    vsi_nn_stack_lcl_data * data;

    ret = VSI_SUCCESS;
    /* Malloc ptr */
    data = (vsi_nn_stack_lcl_data *)malloc( sizeof(vsi_nn_stack_lcl_data) );
    if( NULL == data )
    {
        VSILOGE( "Create stack local data fail." );
        return VSI_FAILURE;
    }
    memset( data, 0, sizeof(vsi_nn_stack_lcl_data) );
    data->src_tensor = src_in->t;
    data->dst_tensor = dst_tensor;
    data->src_in     = src_in;
    /* Store node, ptr */
    vsi_nn_LinkListPushStart(
        (vsi_nn_link_list_t **)&self->nn_param.stack.lcl_data,
        (vsi_nn_link_list_t *)data );

    return ret;
} /* copy_tensor_to_view() */

static void _set_inputs_outputs
    (
    vsi_nn_node_t * self,
    vx_reference * params,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    uint32_t i;
    vsi_nn_stack_lcl_data *data = NULL;
    vx_tensor *tensors = NULL;
    uint32_t num = self->input.num;

    tensors = (vx_tensor *)malloc(sizeof(vx_tensor) * num);
    if(NULL == tensors)
    {
        VSILOGE("stack op malloc fail.");
        return;
    }

    for(i = 0; i < num; i++)
    {
        tensors[i] = inputs[i]->t;
    }
    data = (vsi_nn_stack_lcl_data *)malloc(sizeof(vsi_nn_stack_lcl_data));
    if (NULL == data)
    {
        VSILOGE("vsi_nn_stack_lcl_data malloc fail.\n");
        goto final;
    }
    memset(data, 0, sizeof(vsi_nn_stack_lcl_data));
    data->array = vxCreateTensorObjectArray(self->graph->ctx->c,
                    num, tensors);

    params[0] = (vx_reference)data->array;
    params[1] = (vx_reference)outputs[0]->t;

    self->nn_param.stack.lcl_data = data;
final:
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
    vsi_nn_stack_param * p;
    if( 0 == num )
    {
        return VSI_SUCCESS;
    }
    memset( params, 0, sizeof( vx_reference * ) * num );
    p = &(node->nn_param.stack);
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
    vsi_status status  = VSI_SUCCESS;

    uint32_t i = 0;
    uint32_t tensor_num = self->input.num;
    vsi_nn_stack_param * p;
    vsi_nn_tensor_t * perm_tensor = NULL;
    vx_tensor output_rs = NULL;
    vsi_nn_tensor_attr_t attr;
    uint32_t sizes[VSI_NN_MAX_DIM_NUM] = {0};
    uint32_t perm[VSI_NN_MAX_DIM_NUM] = {0};
    uint32_t start[VSI_NN_MAX_DIM_NUM] = { 0 };
    uint32_t end[VSI_NN_MAX_DIM_NUM] = { 0 };
    uint32_t block_size = 1;
    uint32_t block_num = 1;
    uint32_t dims = 2;
    vsi_nn_stack_lcl_data * iter;

    p = (vsi_nn_stack_param *)&(self->nn_param.stack);

    block_size = p->local.block_size;
    block_num  = p->local.block_num;

    sizes[0] = block_size * tensor_num;
    sizes[1] = block_num;

    if(block_num == 1 && _is_same_quant(self, inputs, outputs))
    {
        iter = self->nn_param.stack.lcl_data;
        while( NULL != iter )
        {
            iter->cp_node = vxTensorCopyNode(self->graph->g,
                iter->src_tensor, iter->dst_tensor );
            if( NULL == iter->cp_node )
            {
                VSILOGE( "Create vxTensorCopyNode fail." );
                status = VSI_FAILURE;
                break;
            }
            iter = (vsi_nn_stack_lcl_data *)vsi_nn_LinkListNext( (vsi_nn_link_list_t *)iter );
        }
    }
    else if (TRUE/*block_size * tensor_num < NN_MAX_INPUT_SIZE && block_num < NN_MAX_INPUT_SIZE*/)
    {
        output_rs = vxReshapeTensor(outputs[0]->t, (int32_t*)sizes, dims);

        sizes[0] = block_num;
        sizes[1] = block_size * tensor_num;
        memcpy(&attr, &(outputs[0]->attr), sizeof(attr));
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

        for (i = 0; i < tensor_num; i++)
        {
            vx_tensor in_view_tensor;
            vx_tensor input_rs = NULL;

            sizes[0] = block_size;
            sizes[1] = block_num;
            input_rs = vxReshapeTensor(inputs[i]->t, (int32_t*)sizes, dims);

            start[1] = end[1];
            end[1] += block_size;
            in_view_tensor = vsi_nn_CreateViewTensor(self->graph, start, end, perm_tensor);
            if( NULL == in_view_tensor )
            {
                VSILOGE( "Create tensor %d from view fail.", i );
                status = VSI_FAILURE;
                break;
            }

            self->n = vxTensorPermuteNode(
                self->graph->g,
                input_rs,
                in_view_tensor,
                perm,
                dims
                );
            if( NULL != self->n )
            {
                status = VSI_SUCCESS;
            }

            if (in_view_tensor) vxReleaseTensor(&in_view_tensor);
            if (input_rs) vxReleaseTensor(&input_rs);
            if (self->n) vxReleaseNode(&self->n);
        }

        if (output_rs)
        {
            char name[64];
            memset(name, 0, sizeof(char) * 64);
            snprintf(name, sizeof(char) * 64, "uid_%u_out_%u", self->uid, i);
            if(output_rs)
            {
                status = vxSetReferenceName((vx_reference)output_rs, name);
            }

            self->n = vxTensorPermuteNode(
                self->graph->g,
                perm_tensor->t,
                output_rs,
                perm,
                dims
                );
            if( NULL != self->n )
            {
                status = VSI_SUCCESS;
            }

            if (output_rs) vxReleaseTensor(&output_rs);
            if (self->n) vxReleaseNode(&self->n);
        }
        if (perm_tensor) vsi_nn_ReleaseTensor(&perm_tensor);
    }
    else
    {
        vsi_nn_kernel_info_t kernel_info;
        char *path = NULL;

        memset(&kernel_info, 0x0, sizeof(vsi_nn_kernel_info_t));
        kernel_info.type = VX_KERNEL_TYPE_CPU;
        kernel_info.kernel = vx_kernel_STACK_list;
        kernel_info.resource_num = 1;
        kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
        kernel_info.resource_name[0] = "vsi_nn_kernel_stack";
        path = getenv("USER_VX_SOURCE_PATH");
        if(path)
            vsi_nn_VxResourceSetPath(path);

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
    vsi_nn_stack_param * p;
    uint32_t i, j;
    uint32_t block_size = 1;
    uint32_t block_num = 1;
    uint32_t axis;

    p = (vsi_nn_stack_param *)&(node->nn_param.stack);
    axis = p->axis;

    for (i = 0; i < axis; i++)
    {
        block_size *= inputs[0]->attr.size[i];
    }

    for (i = axis; i < inputs[0]->attr.dim_num; i++)
    {
        block_num *= inputs[0]->attr.size[i];
    }

    p->local.block_size = block_size;
    p->local.block_num = block_num;

    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        for(i = 0, j = 0; i < inputs[0]->attr.dim_num; i++)
        {
            if (i == p->axis)
            {
                outputs[0]->attr.size[j] = node->input.num;
                j++;
            }
            outputs[0]->attr.size[j] = inputs[0]->attr.size[i];
            j++;
        }
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num + 1;
    }
    return TRUE;
} /* op_setup() */


static vsi_status op_optimize
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_opt_direction_e direction
    )
{
    vsi_status     status;
    int32_t        num,i;
    uint32_t       axis;
    vx_tensor      in_view_tensor;
    uint32_t       sizes[2] = {1};
    uint32_t       block_size = 1;
    uint32_t       block_num = 1;
    uint32_t       dims = 2;
    vsi_nn_tensor_t *output = NULL;
    vsi_nn_tensor_t *input = NULL;
    uint32_t       start[VSI_NN_MAX_DIM_NUM] = { 0 };
    uint32_t       end[VSI_NN_MAX_DIM_NUM] = { 0 };

    status = VSI_SUCCESS;
    /* we don't create tensor view if the axis is not the highest dimension */
    if (self->nn_param.stack.local.block_num != 1 ||
        _is_same_quant(self, inputs, outputs) == FALSE)
    {
        return status;
    }
    /* Only backward run stack's optimize */
    if( direction == VSI_NN_OPTIMIZE_FORWARD )
    {
        return status;
    }

    num = _get_input_num(self, inputs);
    if(num < 0)
    {
        return VSI_FAILURE;
    }

    VSILOGD("Optimize %s, uid %u", vsi_nn_OpGetName(self->op), self->uid);
    axis = self->nn_param.stack.axis;

    if( NULL == outputs[0]->t )
    {
        vsi_nn_TensorReinit( self->graph, outputs[0] );
    }

    block_size = self->nn_param.stack.local.block_size;
    block_num  = self->nn_param.stack.local.block_num;
    sizes[0] = block_size;
    sizes[1] = block_num * num;
    output = vsi_nn_reshape_tensor(self->graph, outputs[0], sizes, dims);
    axis = 1;
    /* Create tensor from view */
    memset( start, 0, sizeof( uint32_t ) * VSI_NN_MAX_DIM_NUM );
    memset( end, 0, sizeof( uint32_t ) * VSI_NN_MAX_DIM_NUM );
    end[0] = block_size;
    end[axis] = 0;
    for( i = 0; i < num; i++ )
    {
        start[axis] = end[axis];
        end[axis] += 1;
        in_view_tensor = vsi_nn_CreateViewTensor(self->graph, start, end, output);
        if( NULL == in_view_tensor )
        {
            VSILOGE( "Create tensor %d from view fail.", i );
            status = VSI_FAILURE;
            break;
        }

        if( NULL != inputs[i]->t )
        {
            VSILOGW( "stack copy %d tensor.", i );
            sizes[0] = block_size;
            sizes[1] = block_num;
            input = vsi_nn_reshape_tensor(self->graph, inputs[0], sizes, dims);
            // Copy old tensor values to the new address.
            status = copy_tensor_to_view( self, input, axis, in_view_tensor );
            if( VSI_FAILURE == status )
            {
                break;
            }
        }
        else
        {
            vx_tensor t = NULL;
            t = vxReshapeTensor(in_view_tensor, (int32_t*)inputs[i]->attr.size, inputs[i]->attr.dim_num);
            inputs[i]->t = t;
            self->nn_param.stack.local.local_tensor[i] = in_view_tensor;
        }
    }

    if (output) vsi_nn_ReleaseTensor(&output);
    output = NULL;

    return status;
} /* op_optimize() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_nn_stack_lcl_data * data;
    vsi_nn_stack_lcl_data * tmp;
    uint32_t i = 0;

    if(NULL == self)
    {
        return VSI_FAILURE;
    }

    data = self->nn_param.stack.lcl_data;

    for (i = 0; i < VSI_NN_STACK_MAX_INPUTS; i++)
    {
        if (self->nn_param.stack.local.local_tensor[i] != NULL)
        {
            vxReleaseTensor(&(self->nn_param.stack.local.local_tensor[i]));
            self->nn_param.stack.local.local_tensor[i] = NULL;
        }
    }

    if(self->n)
    {
        if( NULL != self && NULL != self->n )
        {
            if(data && data->array)
            {
                vxReleaseObjectArray(&data->array);
                free(data);
                data = NULL;
            }
            vxReleaseNode( &self->n );
            self->n = NULL;
        }
    }
    else
    {
        while( NULL != data )
        {
            tmp = (vsi_nn_stack_lcl_data *)vsi_nn_LinkListPopStart(
                (vsi_nn_link_list_t **)&data );
            vxReleaseNode( &tmp->cp_node );
            vxReleaseTensor( &tmp->dst_tensor );
            vsi_nn_ReleaseTensor(&tmp->src_in);
            free( tmp );
        }
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
    /* op_name    */ STACK,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );
#ifdef __cplusplus
}
#endif
