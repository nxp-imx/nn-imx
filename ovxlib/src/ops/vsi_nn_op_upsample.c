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
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "ops/vsi_nn_op_upsample.h"
#include "client/vsi_nn_vxkernel.h"

#define _ARG_NUM            (2)
#define _INPUT_NUM          (2)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)

extern vx_kernel_description_t * vx_kernel_UPSAMPLE_list[];

static void check_tensor_shape
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vx_reference * params,
    uint32_t index,
    vsi_bool rsFlg
    )
{
    vsi_nn_tensor_attr_t attr;

    if( input->attr.dim_num == 1 )
    {
        memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
        attr.size[1] = 1;
        attr.size[2] = 1;
        attr.dim_num = 2;
        self->nn_param.pool.local.local_tensor[index] =
            vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
        params[index] =  (vx_reference)self->nn_param.pool.local.local_tensor[index];
    }
    else if(input->attr.dim_num == 3 && rsFlg)
    {
        memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
        attr.size[1] *= attr.size[2];
        attr.size[2] = 1;
        self->nn_param.pool.local.local_tensor[index] =
            vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
        params[index] =  (vx_reference)self->nn_param.pool.local.local_tensor[index];
    }
    else if(input->attr.dim_num == 4 && rsFlg)
    {
        memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
        attr.size[1] *= attr.size[2];
        attr.size[2] = 1;
        attr.size[3] = attr.size[3];
        self->nn_param.pool.local.local_tensor[index] =
            vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
        params[index] =  (vx_reference)self->nn_param.pool.local.local_tensor[index];
    }
    else
        params[index] = (vx_reference)input->t;
}

#if 0
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
#endif

static vsi_status _create_params
    (
    vsi_nn_node_t * node,
    vx_reference * params,
    uint32_t num
    )
{
    vsi_status status;
    vx_context ctx;
    vsi_nn_upsample_param * p;
    if( 0 == num )
    {
        return VSI_SUCCESS;
    }
    memset( params, 0, sizeof( vx_reference * ) * num );
    p = &(node->nn_param.upsample);
    ctx = vxGetContext( (vx_reference)node->graph->g );
    /* Init parameters */
#define _SET_PARAM( i, type, arg ) do{ \
    params[i] = (vx_reference)vxCreateScalar( ctx, type, &p->arg ); \
    status = vxGetStatus( params[i] ); \
    if( VSI_SUCCESS != status ) { \
    goto set_param_error; \
    } \
    } while(0)

    _SET_PARAM( 0, VX_TYPE_INT32, scale[0] );
    _SET_PARAM( 1, VX_TYPE_INT32, scale[1] );
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

static vsi_status vx_op_pre_init
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_kernel_info_t * kernel_info
    )
{
    vsi_nn_type_e dataType          = inputs[0]->attr.dtype.vx_type;
    vsi_nn_type_e axDataType        = inputs[1]->attr.dtype.vx_type;
    vsi_nn_type_e outDataType       = outputs[0]->attr.dtype.vx_type;
    vx_uint32     height            = outputs[0]->attr.size[1];
    vx_uint32     depth             = outputs[0]->attr.size[2];
    vsi_bool      enable_image_2d   = FALSE;
    vx_uint32     hwLitimLen        = 65536;

     enable_image_2d = (vsi_bool)(height * depth < hwLitimLen
         && ((height % 2 == 0) || depth == 1));

    if((dataType == VSI_NN_TYPE_FLOAT16) &&
        (axDataType == VSI_NN_TYPE_UINT8 || axDataType == VSI_NN_TYPE_INT8) &&
        (outDataType == VSI_NN_TYPE_FLOAT16))
    {
        kernel_info->resource_name[0] = "vsi_nn_kernel_upsample";
        kernel_info->kernel_index = 1;
    }
    else if(dataType == VSI_NN_TYPE_INT8 && (axDataType == VSI_NN_TYPE_UINT8
        || axDataType == VSI_NN_TYPE_INT8) && outDataType == VSI_NN_TYPE_INT8)
    {
        kernel_info->resource_name[0] = "vsi_nn_kernel_upsample_i8";
        if (inputs[0]->attr.dtype.fl == outputs[0]->attr.dtype.fl)
        {
            kernel_info->kernel_index = 2;
        }
        else
        {
            kernel_info->resource_name[0] = "vsi_nn_kernel_upsample_opt";
            kernel_info->kernel_index = 14;
        }
    }
    else if(dataType == VSI_NN_TYPE_UINT8 && (axDataType == VSI_NN_TYPE_UINT8
        || axDataType == VSI_NN_TYPE_INT8) && outDataType == VSI_NN_TYPE_UINT8)
    {
        if (enable_image_2d && axDataType == VSI_NN_TYPE_UINT8)
        {
            kernel_info->resource_name[0] = "vsi_nn_kernel_upsample_u8";
            kernel_info->kernel_index = 17;
        }
        else
        {
            kernel_info->resource_name[0] = "vsi_nn_kernel_upsample_i8";
            kernel_info->kernel_index = 3;
        }
    }
    else if(dataType == VSI_NN_TYPE_INT16
        && (axDataType == VSI_NN_TYPE_UINT8 || axDataType == VSI_NN_TYPE_INT8)
        && outDataType == VSI_NN_TYPE_INT16)
    {
        kernel_info->resource_name[0] = "vsi_nn_kernel_upsample";
        kernel_info->kernel_index = 4;
        if(inputs[0]->attr.dtype.fl != outputs[0]->attr.dtype.fl && axDataType == VSI_NN_TYPE_UINT8)
        {
            kernel_info->resource_name[0] = "vsi_nn_kernel_upsample_opt";
            kernel_info->kernel_index = 13;
        }
        else if(axDataType == VSI_NN_TYPE_UINT8)
        {
            kernel_info->resource_name[0] = "vsi_nn_kernel_upsample";
            kernel_info->kernel_index = 1;
        }
    }
    else if(dataType == VSI_NN_TYPE_INT16 && axDataType == VSI_NN_TYPE_INT16
        && outDataType == VSI_NN_TYPE_FLOAT16)
    {
        kernel_info->resource_name[0] = "vsi_nn_kernel_upsample";
        kernel_info->kernel_index = 5;
    }
    else if((dataType == VSI_NN_TYPE_FLOAT16 || dataType == VSI_NN_TYPE_INT16) &&
        (axDataType == VSI_NN_TYPE_UINT8 || axDataType == VSI_NN_TYPE_INT8) &&
        (outDataType == VSI_NN_TYPE_UINT8))
    {
        if (enable_image_2d && axDataType == VSI_NN_TYPE_UINT8 && dataType == VSI_NN_TYPE_FLOAT16)
        {
            kernel_info->resource_name[0] = "vsi_nn_kernel_upsample_u8";
            kernel_info->kernel_index = 18;
        }
        else
        {
            kernel_info->resource_name[0] = "vsi_nn_kernel_upsample_2";
            kernel_info->kernel_index = 6;
        }
    }
    else if(dataType == VSI_NN_TYPE_UINT8
        && (axDataType == VSI_NN_TYPE_UINT8 || axDataType == VSI_NN_TYPE_INT8)
        && outDataType == VSI_NN_TYPE_FLOAT16)
    {
        kernel_info->resource_name[0] = "vsi_nn_kernel_upsample_u8";
        kernel_info->kernel_index = 7;
    }
    else if(dataType == VSI_NN_TYPE_FLOAT16 && axDataType == VSI_NN_TYPE_FLOAT16
        && outDataType == VSI_NN_TYPE_UINT8)
    {
        kernel_info->resource_name[0] = "vsi_nn_kernel_upsample_2";
        kernel_info->kernel_index = 8;
    }
    else if(dataType == VSI_NN_TYPE_FLOAT16
        && (axDataType == VSI_NN_TYPE_UINT8 || axDataType == VSI_NN_TYPE_INT8)
        && outDataType == VSI_NN_TYPE_INT8)
    {
        kernel_info->resource_name[0] = "vsi_nn_kernel_upsample_2";
        kernel_info->kernel_index = 10;
    }
    else if(dataType == VSI_NN_TYPE_FLOAT16
        && (axDataType == VSI_NN_TYPE_UINT8 || axDataType == VSI_NN_TYPE_INT8)
        && outDataType == VSI_NN_TYPE_INT16)
    {
        kernel_info->resource_name[0] = "vsi_nn_kernel_upsample_2";
        kernel_info->kernel_index = 11;
    }
    else if(dataType == VSI_NN_TYPE_INT16 && axDataType == VSI_NN_TYPE_INT16
        && outDataType == VSI_NN_TYPE_INT16)
    {
        kernel_info->resource_name[0] = "vsi_nn_kernel_upsample";
        kernel_info->kernel_index = 12;
    }
    else if(dataType == VSI_NN_TYPE_INT8
        && (axDataType == VSI_NN_TYPE_UINT8 || axDataType == VSI_NN_TYPE_INT8)
        && outDataType == VSI_NN_TYPE_FLOAT16)
    {
        kernel_info->resource_name[0] = "vsi_nn_kernel_upsample_i8";
        kernel_info->kernel_index = 15;
    }
    else if(dataType == VSI_NN_TYPE_INT16
        && (axDataType == VSI_NN_TYPE_UINT8 || axDataType == VSI_NN_TYPE_INT8)
        && outDataType == VSI_NN_TYPE_FLOAT16)
    {
        kernel_info->resource_name[0] = "vsi_nn_kernel_upsample_i8";
        kernel_info->kernel_index = 16;
    }
    else
    {
        VSILOGE("Unsupported data type(unpooling).\n");
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
    vx_reference * args;
    vsi_enum        srcFormat, dstFormat, axsFormat;
    vx_uint32       height          = outputs[0]->attr.size[1];
    vx_uint32       depth           = outputs[0]->attr.size[2];
    vsi_bool        enable_image_2d = FALSE;
    vx_uint32       hwLitimLen       = 65536;

    args = &params[_IO_NUM];

    srcFormat = inputs[0]->attr.dtype.vx_type;
    axsFormat = inputs[1]->attr.dtype.vx_type;
    dstFormat = outputs[0]->attr.dtype.vx_type;

    if ((srcFormat == VX_TYPE_UINT8 && dstFormat == VX_TYPE_UINT8 && axsFormat == VX_TYPE_UINT8)
      ||(srcFormat == VX_TYPE_FLOAT16 && dstFormat == VX_TYPE_UINT8 && axsFormat == VX_TYPE_UINT8))
    {
        if ((height % 2 == 0) || depth == 1)
        {
            enable_image_2d = (vsi_bool)(height * depth < hwLitimLen);
        }
        else
        {
            enable_image_2d = FALSE;
        }
    }
    else
        enable_image_2d = FALSE;

    args = &params[_IO_NUM];

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    check_tensor_shape(self, inputs[0],  params, 0, enable_image_2d);
    check_tensor_shape(self, inputs[1],  params, 1, enable_image_2d);
    check_tensor_shape(self, outputs[0], params, 2, enable_image_2d);
    /* Set inputs and outputs */
    //_set_inputs_outputs( params, inputs, outputs );

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
    NULL,
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
    vsi_status status;
    vsi_nn_kernel_info_t kernel_info = {0};

    status = VSI_FAILURE;
    kernel_info.resource_num = 1;
    kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
    kernel_info.resource_name[0] = "vsi_nn_kernel_upsample";
    kernel_info.type = vsi_nn_GetVXKernelTypeForShader();
    kernel_info.kernel = vx_kernel_UPSAMPLE_list;
    kernel_info.init_index = 1;

    if (vsi_nn_is_do_vx_op_pre_init(kernel_info.type))
    {
        vx_op_pre_init(self, inputs, outputs, &kernel_info);
    }

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

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    uint32_t h;
    uint32_t w;
    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        w = self->nn_param.upsample.size[0];
        h = self->nn_param.upsample.size[1];
        if ( 0 == self->nn_param.upsample.size[0] )
        {
            w = inputs[0]->attr.size[0] * self->nn_param.upsample.scale[0];
        }
        if ( 0 == self->nn_param.upsample.size[1] )
        {
            h = inputs[0]->attr.size[1] * self->nn_param.upsample.scale[1];
        }
        if (0 == self->nn_param.upsample.scale[0])
        {
            self->nn_param.upsample.scale[0] = self->nn_param.upsample.size[0] /
                inputs[0]->attr.size[0];
        }
        if (0 == self->nn_param.upsample.scale[1])
        {
            self->nn_param.upsample.scale[1] = self->nn_param.upsample.size[1] /
                inputs[0]->attr.size[1];
        }
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
        outputs[0]->attr.size[0] = w;
        outputs[0]->attr.size[1] = h;
        outputs[0]->attr.size[2] = inputs[0]->attr.size[2];
        outputs[0]->attr.size[3] = inputs[1]->attr.size[3];
    }
    return TRUE;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    uint32_t i;
    for (i = 0; i < _VSI_NN_UPSAMPLE_LOCAL_TENSOR_NUM; i++)
    {
        if (self->nn_param.upsample.local.local_tensor[i] != NULL)
        {
            vxReleaseTensor(&(self->nn_param.upsample.local.local_tensor[i]));
            self->nn_param.upsample.local.local_tensor[i] = NULL;
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
    /* op_name    */ UPSAMPLE,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 2,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif

