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
#define _VX_PARAM_NUM       (_ARG_NUM + _IO_NUM + 2)

extern vx_kernel_description_t * vx_kernel_GATHER_list[];

static vsi_bool _get_Gather_tensor_reshape_size
    (
    vsi_nn_tensor_t ** inputs,
    uint32_t sizes[VSI_NN_MAX_DIM_NUM],
    uint32_t block_size,
    uint32_t idxFlg
    )
{
    uint32_t dims_num = inputs[0]->attr.dim_num;
    uint32_t *input_size = inputs[0]->attr.size;
    uint32_t i = 0;
    uint32_t elementCnt = 1;

    for(i = 0; i < dims_num; ++i)
    {
        elementCnt *= input_size[i];
    }

    if(idxFlg)
    {
        sizes[0] = elementCnt;
        sizes[1] = 1;
    }
    else
    {
        sizes[0] = block_size;
        sizes[1] = elementCnt / block_size;
    }

    return TRUE;
} /* _get_EltOP_tensor_reshape_size */

static void reshape_tensor_shape
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vx_reference * params,
    uint32_t index,
    uint32_t block_size,
    uint32_t idxFlg
    )
{
    uint32_t sizes[VSI_NN_MAX_DIM_NUM] = {1};
    uint32_t dims = 2;
    vsi_bool ret = FALSE;

    ret = _get_Gather_tensor_reshape_size(&input, sizes, block_size, idxFlg);

    if (ret)
        self->nn_param.gather.local.local_tensor[index] =
            vxReshapeTensor(input->t, (int32_t *)sizes, dims);

    params[index] =
        ret ? (vx_reference)self->nn_param.gather.local.local_tensor[index] : (vx_reference)input->t;
}

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
    vsi_nn_gather_param * p;
    if( 0 == num )
    {
        return VSI_SUCCESS;
    }
    memset( params, 0, sizeof( vx_reference * ) * num );
    p = &(node->nn_param.gather);
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
    uint32_t block_size,
    uint32_t block_num,
    uint32_t axis_num
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_reference params[_VX_PARAM_NUM];
    vx_reference * args;
    vx_context ctx;

    args = &params[_IO_NUM];
    ctx = vxGetContext( (vx_reference)self->graph->g );

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    /* Set inputs and outputs */
    reshape_tensor_shape(self, inputs[0], params, 0, block_size, 0);
    reshape_tensor_shape(self, inputs[1], params, 1, 1, 1);
    reshape_tensor_shape(self, outputs[0], params, 2, block_size, 0);

    /*TODO: Add code if need to change your parameter*/

    /* Init parameters. */
    args[0] = (vx_reference)vxCreateScalar( ctx, VX_TYPE_INT32, &block_size );
    args[1] = (vx_reference)vxCreateScalar( ctx, VX_TYPE_INT32, &block_num );
    args[2] = (vx_reference)vxCreateScalar( ctx, VX_TYPE_INT32, &axis_num );

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, _VX_PARAM_NUM );

    _release_params( args, 3 );

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

    if (inputDataFormat == VSI_NN_TYPE_INT8 || inputDataFormat == VSI_NN_TYPE_UINT8)
    {
        kernel_info->kernel_index = 1;
    }
    else if(inputDataFormat == VSI_NN_TYPE_FLOAT16 || inputDataFormat == VSI_NN_TYPE_INT16)
    {
        kernel_info->kernel_index = 2;
    }
    else
    {
        VSILOGE("Not support input or output data format!(gather) at [%s : %d]\n", __FILE__, __LINE__);
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
    uint32_t i = 0;
    uint32_t block_size = 1, block_num = 1, axis_num = 0;
    int32_t axis = self->nn_param.gather.axis;
    uint32_t *input_size = inputs[0]->attr.size;
    uint32_t dims_num = inputs[0]->attr.dim_num;
    uint32_t eCnt0 = 1, eCnt1 = 1, eCnt2 = 1;
    vx_bool_e dataTypeFlg = vx_false_e, sizeFlg = vx_false_e;
    uint32_t max_img_width = 65536;
    status = VSI_FAILURE;

    memset(&kernel_info, 0x0, sizeof(vsi_nn_kernel_info_t));
    kernel_info.type = VX_KERNEL_TYPE_CPU;

    for(i = 0; i < (uint32_t)axis; ++i)
    {
        block_size *= input_size[i];
    }

    for(i = 0; i < (uint32_t)dims_num; ++i)
    {
        eCnt0 *= input_size[i];
    }

    for(i = 0; i < (uint32_t)inputs[1]->attr.dim_num; ++i)
    {
        eCnt1 *= inputs[1]->attr.size[i];
    }

    for(i = 0; i < (uint32_t)outputs[0]->attr.dim_num; ++i)
    {
        eCnt2 *= outputs[0]->attr.size[i];
    }

    if(((eCnt0/block_size) < max_img_width)
        && eCnt1 < max_img_width
        && ((eCnt2/block_size) < max_img_width)
        )
    {
        sizeFlg = vx_true_e;
    }

    if(inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT16
        || inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_INT16
        || inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_INT8
        || inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_UINT8)
        dataTypeFlg = vx_true_e;

    if(sizeFlg && dataTypeFlg)
        kernel_info.type = VX_KERNEL_TYPE_VX;

    kernel_info.kernel = vx_kernel_GATHER_list;
    kernel_info.resource_num = 1;
    kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
    kernel_info.resource_name[0] = "vsi_nn_kernel_gather";

    if( kernel_info.type == VX_KERNEL_TYPE_VX)
    {
        kernel_info.kernel_index = 1;
        kernel_info.init_index = 1;
        status = vx_op_pre_compute(self, inputs, outputs, &kernel_info);
        if (status != VX_SUCCESS)
        {
            goto final;
        }
    }
    else /*kernel_info.type = VX_KERNEL_TYPE_CPU;*/
    {
        kernel_info.type = VX_KERNEL_TYPE_CPU;
        kernel_info.kernel_index = 0;
        kernel_info.init_index = 0;
    }

    self->n = vsi_nn_RegisterClientKernelAndNewNode(
            self->graph, &kernel_info);
    if( NULL == self->n )
    {
        status = VSI_FAILURE;
        goto final;
    }

    if( kernel_info.type == VX_KERNEL_TYPE_VX)
    {
        axis_num = input_size[axis];
        for(i = axis + 1; i < dims_num; ++i)
        {
            block_num *= input_size[i];
        }
        status = vx_op_compute(self, inputs, outputs, block_size, block_num, axis_num);
    }
    else
    {
        status = cpu_op_compute(self, inputs, outputs);
    }
final:
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
    /* TODO: Add code to comput outputs' shape. */
    uint32_t i;
    vsi_nn_gather_param * p;

    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        uint32_t j = 0;
        p = &(self->nn_param.gather);
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num + inputs[1]->attr.dim_num - 1;
        for (i = 0; i < (uint32_t)p->axis; i++)
        {
            outputs[0]->attr.size[j] = inputs[0]->attr.size[i];
            j++;
        }
        for (i = 0; i < inputs[1]->attr.dim_num; i++)
        {
            outputs[0]->attr.size[j] = inputs[1]->attr.size[i];
            j++;
        }
        for (i = (uint32_t)p->axis + 1; i < inputs[0]->attr.dim_num; i++)
        {
            outputs[0]->attr.size[j] = inputs[0]->attr.size[i];
            j++;
        }
    }
    return TRUE;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    uint32_t i;
    for (i = 0; i < _VSI_NN_GATHER_LOCAL_TENSOR_NUM; i++)
    {
        if (self->nn_param.gather.local.local_tensor[i] != NULL)
        {
            vxReleaseTensor(&(self->nn_param.gather.local.local_tensor[i]));
            self->nn_param.gather.local.local_tensor[i] = NULL;
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
    /* op_name    */ GATHER,
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
