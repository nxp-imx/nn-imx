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

#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "utils/vsi_nn_math.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "client/vsi_nn_vxkernel.h"
#include "utils/vsi_nn_util.h"

#define _ARG_NUM            (0)
#define _INPUT_NUM          (2)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)

#define TENSOR_ALL 0

extern vx_kernel_description_t * vx_kernel_RELATIONAL_OPS_list[];

static void check_tensor_shape
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vx_reference * params,
    uint32_t index,
    vx_bool rsFlg
    )
{
    vsi_nn_tensor_attr_t attr;

    if (index == 0)
    {
        if( input->attr.dim_num == 1)
        {
            memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
            attr.size[1] = 1;
            attr.dim_num = 2;
            self->nn_param.relational_ops.local.local_tensor[index] =
                vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
            params[index] =  (vx_reference)self->nn_param.relational_ops.local.local_tensor[index];
        }
        else
            params[index] = (vx_reference)input->t;
    }
    else if (index == 1)
    {
        if( input->attr.dim_num == 1)
        {
            memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
            attr.size[1] = 1;
            attr.dim_num = 2;
            self->nn_param.relational_ops.local.local_tensor[index] =
                vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
            params[index] =  (vx_reference)self->nn_param.relational_ops.local.local_tensor[index];
        }
        else
            params[index] = (vx_reference)input->t;
    }
    else if(index == 2)
    {
        if(input->attr.dim_num == 1)
        {
            memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
            attr.size[1] = 1;
            attr.dim_num = 2;
            self->nn_param.relational_ops.local.local_tensor[index] =
                vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
            params[index] =  (vx_reference)self->nn_param.relational_ops.local.local_tensor[index];
        }
        else if(input->attr.dim_num == 4)
        {
            memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
            attr.size[2] *= attr.size[3];
            attr.size[3] = 1;
            attr.dim_num = 3;
            self->nn_param.relational_ops.local.local_tensor[index] =
                vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
            params[index] =  (vx_reference)self->nn_param.relational_ops.local.local_tensor[index];
        }
        else
             params[index] = (vx_reference)input->t;
    }
    else
    {
        VSILOGE("No more local tensor!(relational_ops) at [%s : %d]\n", __FILE__, __LINE__);
    }
}

static vsi_status _create_params
    (
    vsi_nn_node_t * node,
    vx_reference * params,
    uint32_t num
    )
{
    vsi_status status;
    vx_context ctx;
    vsi_nn_relational_ops_param * p;
    if( 0 == num )
    {
        return VSI_SUCCESS;
    }
    memset( params, 0, sizeof( vx_reference * ) * num );
    p = &(node->nn_param.relational_ops);
    ctx = vxGetContext( (vx_reference)node->graph->g );
    /* Init parameters */
#define _SET_PARAM( i, type, arg ) do{ \
    params[i] = (vx_reference)vxCreateScalar( ctx, type, &p->arg ); \
    status = vxGetStatus( params[i] ); \
    if( VSI_SUCCESS != status ) { \
    goto set_param_error; \
    } \
    } while(0)
    _SET_PARAM( 0, VX_TYPE_UINT32, op );
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
    vx_reference params[_PARAM_NUM + 1];
    vx_bool rsFlg = vx_false_e;
    vx_reference * args;
    args = &params[_IO_NUM];

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    /* Set inputs and outputs */
    //_set_inputs_outputs( params, inputs, outputs );
    check_tensor_shape(self, inputs[0], params, 0, rsFlg);
    check_tensor_shape(self, inputs[1], params, 1, rsFlg);
    check_tensor_shape(self, outputs[0], params, 2, rsFlg);
    _create_params( self, args, 1 );
    /* Pass parameters to node. */
     status = vsi_nn_ClientNodePassParameters( self->n, params, _PARAM_NUM + 1 );

    _release_params( args, 1 );

    return status;
}

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
    uint32_t op = self->nn_param.relational_ops.op;

    if(inputDataFormat == VSI_NN_TYPE_FLOAT16 && outputDataFormat == VSI_NN_TYPE_FLOAT16)
    {
        if(op == VSI_NN_RELATIONAL_OPS_GREAT)
            kernel_info->kernel_index = 1;
        else if(op == VSI_NN_RELATIONAL_OPS_GREAT_EQUAL)
            kernel_info->kernel_index = 2;
        else if(op == VSI_NN_RELATIONAL_OPS_LESS)
            kernel_info->kernel_index = 3;
        else if(op == VSI_NN_RELATIONAL_OPS_LESS_EQUAL)
            kernel_info->kernel_index = 4;
        else if(op == VSI_NN_RELATIONAL_OPS_NOT_EQUAL)
            kernel_info->kernel_index = 5;
        else if(op == VSI_NN_RELATIONAL_OPS_EQUAL)
            kernel_info->kernel_index = 6;
        else
        {
            VSILOGE("Not support operation!(relational_ops) at [%s : %d]\n", __FILE__, __LINE__);
            return VSI_FAILURE;
        }
    }
    else if(inputDataFormat == VSI_NN_TYPE_INT16 && outputDataFormat == VSI_NN_TYPE_INT16)
    {
        if(op == VSI_NN_RELATIONAL_OPS_GREAT)
            kernel_info->kernel_index = 7;
        else if(op == VSI_NN_RELATIONAL_OPS_GREAT_EQUAL)
            kernel_info->kernel_index = 8;
        else if(op == VSI_NN_RELATIONAL_OPS_LESS)
            kernel_info->kernel_index = 9;
        else if(op == VSI_NN_RELATIONAL_OPS_LESS_EQUAL)
            kernel_info->kernel_index = 10;
        else if(op == VSI_NN_RELATIONAL_OPS_NOT_EQUAL)
            kernel_info->kernel_index = 11;
        else if(op == VSI_NN_RELATIONAL_OPS_EQUAL)
            kernel_info->kernel_index = 12;
        else
        {
            VSILOGE("Not support operation!(relational_ops) at [%s : %d]\n", __FILE__, __LINE__);
            return VSI_FAILURE;
        }
    }
    else if ((inputDataFormat == VSI_NN_TYPE_INT8 && outputDataFormat == VSI_NN_TYPE_INT8))
    {
        if(op == VSI_NN_RELATIONAL_OPS_GREAT)
            kernel_info->kernel_index = 13;
        else if(op == VSI_NN_RELATIONAL_OPS_GREAT_EQUAL)
            kernel_info->kernel_index = 14;
        else if(op == VSI_NN_RELATIONAL_OPS_LESS)
            kernel_info->kernel_index = 15;
        else if(op == VSI_NN_RELATIONAL_OPS_LESS_EQUAL)
            kernel_info->kernel_index = 16;
        else if(op == VSI_NN_RELATIONAL_OPS_NOT_EQUAL)
            kernel_info->kernel_index = 17;
        else if(op == VSI_NN_RELATIONAL_OPS_EQUAL)
            kernel_info->kernel_index = 18;
        else
        {
            VSILOGE("Not support operation!(relational_ops) at [%s : %d]\n", __FILE__, __LINE__);
            return VSI_FAILURE;
        }
    }
    else if(inputDataFormat == VSI_NN_TYPE_UINT8 && outputDataFormat == VSI_NN_TYPE_UINT8)
    {
        if(op == VSI_NN_RELATIONAL_OPS_GREAT)
            kernel_info->kernel_index = 19;
        else if(op == VSI_NN_RELATIONAL_OPS_GREAT_EQUAL)
            kernel_info->kernel_index = 20;
        else if(op == VSI_NN_RELATIONAL_OPS_LESS)
            kernel_info->kernel_index = 21;
        else if(op == VSI_NN_RELATIONAL_OPS_LESS_EQUAL)
            kernel_info->kernel_index = 22;
        else if(op == VSI_NN_RELATIONAL_OPS_NOT_EQUAL)
            kernel_info->kernel_index = 23;
        else if(op == VSI_NN_RELATIONAL_OPS_EQUAL)
            kernel_info->kernel_index = 24;
        else
        {
            VSILOGE("Not support operation!(relational_ops) at [%s : %d]\n", __FILE__, __LINE__);
            return VSI_FAILURE;
        }
    }
    else
    {
        VSILOGE("Not support input or output data format!(relational_ops) at [%s : %d]\n", __FILE__, __LINE__);
        return VSI_FAILURE;
    }
    return VSI_SUCCESS;
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
    vx_border_t border;
    vx_bool rsFlg = vx_false_e;

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    /* Set inputs and outputs */
    //_set_inputs_outputs( params, inputs, outputs );
    check_tensor_shape(self, inputs[0], params, 0, rsFlg);
    check_tensor_shape(self, inputs[1], params, 1, rsFlg);
    check_tensor_shape(self, outputs[0], params, 2, rsFlg);

    /* Pass parameters to node. */
    status |= vsi_nn_ClientNodePassParameters( self->n, params, _PARAM_NUM);

    border.mode = VX_BORDER_REPLICATE;
    status |= vxSetNodeAttribute(self->n, VX_NODE_BORDER, &border, sizeof(border));

    return status;
}

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_kernel_info_t kernel_info;
    vsi_nn_type_e inputDataFormat     = inputs[0]->attr.dtype.vx_type;
    vsi_nn_type_e outputDataFormat    = outputs[0]->attr.dtype.vx_type;

    memset(&kernel_info, 0x0, sizeof(vsi_nn_kernel_info_t));
    if(inputDataFormat == outputDataFormat && inputDataFormat != VSI_NN_TYPE_FLOAT32 && vsi_nn_IsEVISFeatureAvaiable(self->graph->ctx))
    {
        kernel_info.resource_num = 1;
        kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
        kernel_info.resource_name[0] = "vsi_nn_kernel_relational_ops";
        kernel_info.type = vsi_nn_GetVXKernelTypeForShader();
        kernel_info.kernel = vx_kernel_RELATIONAL_OPS_list;
        kernel_info.init_index = 1;

        if (vsi_nn_is_do_vx_op_pre_init(kernel_info.type))
        {
            vx_op_pre_compute(self, inputs, outputs, &kernel_info);
        }

        self->n = vsi_nn_RegisterClientKernelAndNewNode(
            self->graph, &kernel_info);
        if (kernel_info.resource_name) free(kernel_info.resource_name);
        if( NULL == self->n )
        {
            return VSI_FAILURE;
        }

        status |= vx_op_compute(self, inputs, outputs);
    }
    else
    {
        kernel_info.resource_num = 1;
        kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
        kernel_info.resource_name[0] = "vsi_nn_kernel_relational_ops";
        kernel_info.type = VX_KERNEL_TYPE_CPU;
        kernel_info.kernel = vx_kernel_RELATIONAL_OPS_list;
        kernel_info.init_index = 0;

        self->n = vsi_nn_RegisterClientKernelAndNewNode(
            self->graph, &kernel_info);
        if (kernel_info.resource_name) free(kernel_info.resource_name);
        if( NULL == self->n )
        {
            return VSI_FAILURE;
        }

        status = cpu_op_compute(self, inputs, outputs);
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
    //TODO: Check tensor shapes.
    return TRUE;
} /* op_check() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    uint32_t i;
    for (i = 0; i < _VSI_NN_RELATIONAL_OPS_LOCAL_TENSOR_NUM; i++)
    {
        if (self->nn_param.relational_ops.local.local_tensor[i] != NULL)
        {
            vxReleaseTensor(&(self->nn_param.relational_ops.local.local_tensor[i]));
            self->nn_param.relational_ops.local.local_tensor[i] = NULL;
        }
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
    /* op_name    */ RELATIONAL_OPS,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ vsi_nn_op_common_setup,
    /* optimize   */ NULL,
    /* input_num  */ 2,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif

