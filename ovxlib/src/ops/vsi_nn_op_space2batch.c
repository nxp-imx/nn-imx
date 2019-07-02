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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "vsi_nn_platform.h"

#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_math.h"
#include "client/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"

#define _ARG_NUM            (5)
#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)

#define USE_OVX_API TRUE

#if (USE_OVX_API == FALSE)
extern vx_kernel_description_t * vx_kernel_SPACE2BATCH_list[];

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
    vsi_nn_space2batch_param * p;
    if( 0 == num )
    {
        return VSI_SUCCESS;
    }
    memset( params, 0, sizeof( vx_reference * ) * num );
    p = &(node->nn_param.space2batch);
    ctx = vxGetContext( (vx_reference)node->graph->g );
    /* Init parameters */
#define _SET_PARAM( i, type, arg ) do{ \
    params[i] = (vx_reference)vxCreateScalar( ctx, type, &p->arg ); \
    status = vxGetStatus( params[i] ); \
    if( VSI_SUCCESS != status ) { \
    goto set_param_error; \
    } \
    } while(0)
    _SET_PARAM( 0, VX_TYPE_INT32, block_size );
    _SET_PARAM( 1, VX_TYPE_INT32, pad[0] );
    _SET_PARAM( 2, VX_TYPE_INT32, pad[1] );
    _SET_PARAM( 3, VX_TYPE_INT32, pad[2] );
    _SET_PARAM( 4, VX_TYPE_INT32, pad[3] );
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
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_SUCCESS;

    return status;
}

static vsi_nn_op_compute_t op_compute_list[] =
{
    cpu_op_compute,
    vx_op_compute,
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
    vx_nn_reorg_params_ext_t param;
    vsi_nn_tensor_t *block_size_tensor = NULL;
    vsi_nn_tensor_t *pad_tensor = NULL;
    vsi_nn_tensor_attr_t attr;

    memset(&param, 0, sizeof(vx_nn_reorg_params_t));

    memset(&attr, 0, sizeof(attr));
    attr.size[0] = 2;
    attr.dim_num = 1;
    attr.is_const = TRUE;
    attr.dtype.vx_type = VSI_NN_TYPE_UINT32;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    block_size_tensor = vsi_nn_CreateTensorFromData(
        self->graph,
        (uint8_t *)self->nn_param.space2batch.block_size_2,
        &attr);
    if( NULL == block_size_tensor )
    {
        VSILOGE("Create block_size_tensor fail.(space2batch)");
        return VSI_FAILURE;
    }

    memset(&attr, 0, sizeof(attr));
    attr.size[0] = 4;
    attr.dim_num = 1;
    attr.is_const = TRUE;
    attr.dtype.vx_type = VSI_NN_TYPE_UINT32;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    pad_tensor = vsi_nn_CreateTensorFromData(
        self->graph,
        (uint8_t *)self->nn_param.space2batch.pad,
        &attr);
    if( NULL == pad_tensor )
    {
        VSILOGE("Create pad_tensor fail.(space2batch)");
        return VSI_FAILURE;
    }

    self->nn_param.space2batch.local.block_size_tensor = block_size_tensor;
    self->nn_param.space2batch.local.pad_tensor = pad_tensor;
    param.base.block_size = REQUIRED_IO(block_size_tensor);
    param.pad = OPTIONAL_IO(pad_tensor);
    param.base.type = VX_REORG_SPACE_TO_BATCH_ND;

    self->n = vxReorgLayer2( self->graph->g,
        inputs[0]->t,
        (vx_nn_reorg_params_t *)&param,
        sizeof(vx_nn_reorg_params_ext_t),
        outputs[0]->t);

    if( NULL != self->n )
    {
        status = VSI_SUCCESS;
    }
#else
    vsi_nn_kernel_info_t kernel_info = {0};

    status = VSI_FAILURE;
    kernel_info.resource_num = 1;
    kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
    kernel_info.resource_name[0] = "vsi_nn_kernel_space2batch";
    kernel_info.type = VX_KERNEL_TYPE_CPU;
    kernel_info.kernel = vx_kernel_SPACE2BATCH_list;
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
    //TODO: Check tensor shapes.
    if (inputs[0]->attr.dim_num != 4)
    {
        VSILOGE("The input tensor shape must be 4-D!(space2batch)");
        return FALSE;
    }
    return TRUE;
} /* op_add_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_space2batch_param * p;
    p = (vsi_nn_space2batch_param *)&(self->nn_param.space2batch);
    if (p->block_size != 0)
    {
        p->block_size_2[0] = p->block_size;
        p->block_size_2[1] = p->block_size;
    }

    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.size[3] = inputs[0]->attr.size[3] *
            p->block_size_2[0] * p->block_size_2[1];
        outputs[0]->attr.size[2] = inputs[0]->attr.size[2];
        outputs[0]->attr.size[1] = (p->pad[0] + p->pad[1] + inputs[0]->attr.size[1])
            / p->block_size_2[1];
        outputs[0]->attr.size[0] = (p->pad[2] + p->pad[3] + inputs[0]->attr.size[0])
            / p->block_size_2[0];
        outputs[0]->attr.dim_num = 4;
    }

    return TRUE;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    if (self->nn_param.space2batch.local.block_size_tensor != NULL)
    {
        vsi_nn_ReleaseTensor(&(self->nn_param.space2batch.local.block_size_tensor));
    }
    if (self->nn_param.space2batch.local.pad_tensor != NULL)
    {
        vsi_nn_ReleaseTensor(&(self->nn_param.space2batch.local.pad_tensor));
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
    /* op_name    */ SPACE2BATCH,
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

