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

#define _ARG_NUM            (1)
#define _INPUT_NUM          (2)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)

extern vx_kernel_description_t * vx_kernel_RANDOM_MULTINOMIAL_list[];

static void _set_inputs_outputs
    (
    vx_reference * params,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    uint32_t i;
    uint32_t cnt;

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

static vsi_status _create_params
    (
    vsi_nn_node_t * node,
    vx_reference * params,
    uint32_t num
    )
{
    vsi_status status;
    vx_context ctx;
    vsi_nn_random_multinomial_param * p;
    if( 0 == num )
    {
        return VSI_SUCCESS;
    }
    memset( params, 0, sizeof( vx_reference * ) * num );
    p = &(node->nn_param.random_multinomial);
    ctx = vxGetContext( (vx_reference)node->graph->g );
    /* Init parameters */
    #define _SET_PARAM( i, type, arg ) do{ \
        params[i] = (vx_reference)vxCreateScalar( ctx, type, &p->arg ); \
        status = vxGetStatus( params[i] ); \
        if( VSI_SUCCESS != status ) { \
            goto set_param_error; \
            } \
        } while(0)
    _SET_PARAM( 0, VX_TYPE_INT32, sample_num );
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
    _set_inputs_outputs( params, inputs, outputs );

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
    vsi_nn_tensor_t ** outputs,
    vx_array * arrayList,
    int32_t class_size,
    int32_t class_max_stride
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_reference params[5];
    vx_reference * args;
    vx_border_t border;

    args = &params[3];

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    /* Set inputs and outputs */
    {
        vx_context ctx;
        ctx = vxGetContext( (vx_reference)self->graph->g );

        params[0] = (vx_reference)inputs[0]->t;
        params[1] = (vx_reference)arrayList[0];
        params[2] = (vx_reference)outputs[0]->t;
        params[3] = (vx_reference)vxCreateScalar( ctx, VX_TYPE_INT32, &class_size );
        params[4] = (vx_reference)vxCreateScalar( ctx, VX_TYPE_INT32, &class_max_stride );
    }

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, 5 );

    _release_params( args, 2 );

    border.mode = VX_BORDER_REPLICATE;
    border.constant_value.U32 = 0;
    status |= vxSetNodeAttribute(self->n, VX_NODE_BORDER, &border, sizeof(border));

    return status;
}

static vsi_status vx_op_gen_keys_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_reference params[2];
    vx_border_t border;

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    /* Set inputs and outputs */
    {
        params[0] = (vx_reference)inputs[1]->t;
        params[1] = (vx_reference)outputs[0]->t;
    }
    /*TODO: Add code if need to change your parameter*/

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, 2 );

    border.mode = VX_BORDER_REPLICATE;
    border.constant_value.U32 = 0;
    status |= vxSetNodeAttribute(self->n, VX_NODE_BORDER, &border, sizeof(border));

    return status;
}

static vsi_status vx_op_calculate_cdf_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vx_array * arrayList
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_reference params[2];
    vx_border_t border;

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    /* Set inputs and outputs */
    {
        params[0] = (vx_reference)inputs[0]->t;
        params[1] = (vx_reference)arrayList[0];
    }
    /*TODO: Add code if need to change your parameter*/

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, 2 );

    border.mode = VX_BORDER_REPLICATE;
    border.constant_value.U32 = 0;
    status |= vxSetNodeAttribute(self->n, VX_NODE_BORDER, &border, sizeof(border));

    return status;
}

static vsi_status vx_op_gen_keys_pre_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_kernel_info_t * kernel_info
    )
{
    vsi_nn_type_e inputDataFormat     = inputs[1]->attr.dtype.vx_type;
    if (inputDataFormat == VSI_NN_TYPE_INT32)
    {
        kernel_info->kernel_index = 1;
    }
    else
    {
        VSILOGE("Not support input or output data format!(RANDOM_MULTINOMIAL) at [%s : %d]\n", __FILE__, __LINE__);
        return VSI_FAILURE;
    }
    return VSI_SUCCESS;
}

static vsi_status vx_op_cdf_pre_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_kernel_info_t * kernel_info
    )
{
    vsi_nn_type_e inputDataFormat     = inputs[0]->attr.dtype.vx_type;
    if (inputDataFormat == VSI_NN_TYPE_FLOAT16)
    {
        kernel_info->kernel_index = 2;
    }
    else if (inputDataFormat == VSI_NN_TYPE_FLOAT32)
    {
        kernel_info->kernel_index = 3;
    }
    else
    {
        VSILOGE("Not support input or output data format!(RANDOM_MULTINOMIAL) at [%s : %d]\n", __FILE__, __LINE__);
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
    vx_array arrayCdf = NULL;
    vsi_nn_tensor_t* random_keys = NULL;
    vsi_nn_type_e inputDataFormat     = inputs[0]->attr.dtype.vx_type;

    memset(&kernel_info, 0x0, sizeof(vsi_nn_kernel_info_t));
    status = VSI_FAILURE;
    kernel_info.type = vsi_nn_GetVXKernelTypeForShader();
    kernel_info.kernel = vx_kernel_RANDOM_MULTINOMIAL_list;

    if( kernel_info.type == VX_KERNEL_TYPE_VX
        && (inputDataFormat == VSI_NN_TYPE_FLOAT32 || inputDataFormat == VSI_NN_TYPE_FLOAT16)
        && ((outputs[0]->attr.size[0] * outputs[0]->attr.size[1]) > 128))
    {
        vx_context ctx;
        vsi_nn_tensor_attr_t attr;
        uint32_t class_max_stride = 0;
        uint32_t class_size;

        kernel_info.resource_num = 2;
        kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
        kernel_info.resource_name[0] = "vsi_nn_kernel_header";
        kernel_info.resource_name[1] = "vsi_nn_kernel_random_multinomial";

        ctx = vxGetContext( (vx_reference)self->graph->g );
        memcpy(&attr, &(outputs[0]->attr), sizeof(vsi_nn_tensor_attr_t));
        attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
        attr.vtl = FALSE;
        random_keys = vsi_nn_CreateTensor(self->graph, &attr);

        class_size = inputs[0]->attr.size[0];
        if(inputDataFormat == VSI_NN_TYPE_FLOAT32)
            class_max_stride = ((inputs[0]->attr.size[0] + 3) >> 2) << 2;
        else
            class_max_stride = ((inputs[0]->attr.size[0] + 7) >> 3) << 3;
        arrayCdf = vxCreateArray(ctx, VX_TYPE_FLOAT32, class_max_stride * inputs[0]->attr.size[1] * sizeof(vx_float32));

        // generate random keys
        {
            status = vx_op_gen_keys_pre_compute(self, inputs, &random_keys, &kernel_info);
            if(status != VSI_SUCCESS)
                goto final;

            self->n = vsi_nn_RegisterClientKernelAndNewNode(
                self->graph, &kernel_info);
            //if (kernel_info.resource_name) free(kernel_info.resource_name);
            if( NULL == self->n )
            {
                status = VSI_FAILURE;
                goto final;
            }
            status |= vx_op_gen_keys_compute(self, inputs, &random_keys);
            if(status != VSI_SUCCESS)
                goto final;
        }

        // calculate cdf
        {
            status = vx_op_cdf_pre_compute(self, inputs, outputs, &kernel_info);
            if(status != VSI_SUCCESS)
                goto final;

            self->n = vsi_nn_RegisterClientKernelAndNewNode(
                self->graph, &kernel_info);
            //if (kernel_info.resource_name) free(kernel_info.resource_name);
            if( NULL == self->n )
            {
                status = VSI_FAILURE;
                goto final;
            }
            status |= vx_op_calculate_cdf_compute(self, inputs, &arrayCdf);
            if(status != VSI_SUCCESS)
                goto final;
        }

        // random multinomial
        {
            kernel_info.kernel_index = 4;

            self->n = vsi_nn_RegisterClientKernelAndNewNode(
                self->graph, &kernel_info);
            if (kernel_info.resource_name) free(kernel_info.resource_name);
            if( NULL == self->n )
            {
                status = VSI_FAILURE;
                goto final;
            }
            status |= vx_op_compute(self, &random_keys, outputs, &arrayCdf, class_size, class_max_stride);
            if(status != VSI_SUCCESS)
                goto final;
        }
    }
    else /*kernel_info.type = VX_KERNEL_TYPE_CPU;*/
    {
        kernel_info.resource_num = 1;
        kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
        kernel_info.resource_name[0] = "vsi_nn_kernel_random_multinomial";
        kernel_info.type = VX_KERNEL_TYPE_CPU;
        kernel_info.kernel_index = 0;
        kernel_info.init_index = 0;

        self->n = vsi_nn_RegisterClientKernelAndNewNode(
            self->graph, &kernel_info);
        if (kernel_info.resource_name)
        {
            free(kernel_info.resource_name);
        }
        if( NULL == self->n )
        {
            status = VSI_FAILURE;
            goto final;
        }

        status = cpu_op_compute(self, inputs, outputs);

        goto final;
    }
final:
   if(random_keys)
   {
    vsi_nn_ReleaseTensor(&random_keys);
    random_keys = NULL;
   }
   if(kernel_info.resource_name)
   {
       free(kernel_info.resource_name);
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
    vsi_nn_random_multinomial_param * p;

    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        p = (vsi_nn_random_multinomial_param *)&(self->nn_param.random_multinomial);
        outputs[0]->attr.size[0] = p->sample_num;
        outputs[0]->attr.size[1] = inputs[0]->attr.size[1];
        outputs[0]->attr.dim_num = 2;
    }
    return TRUE;
} /* op_setup() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ RANDOM_MULTINOMIAL,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );
#ifdef __cplusplus
}
#endif
