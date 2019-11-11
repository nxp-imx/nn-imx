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

#define _ARG_NUM            (1)
#define _INPUT_NUM          (4)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)

#define _SUM_ARG_NUM            (2)
#define _SUM_INPUT_NUM          (1)
#define _SUM_OUTPUT_NUM         (1)
#define _SUM_IO_NUM             (_SUM_INPUT_NUM + _SUM_OUTPUT_NUM)
#define _SUM_PARAM_NUM          (_SUM_ARG_NUM + _SUM_IO_NUM)

#define _SQR_ARG_NUM            (5)
#define _SQR_INPUT_NUM          (1)
#define _SQR_OUTPUT_NUM         (1)
#define _SQR_IO_NUM             (_SQR_INPUT_NUM + _SQR_OUTPUT_NUM)
#define _SQR_PARAM_NUM          (_SQR_ARG_NUM + _SQR_IO_NUM)

#define _VARI_ARG_NUM            (1)
#define _VARI_INPUT_NUM          (1)
#define _VARI_OUTPUT_NUM         (1)
#define _VARI_IO_NUM             (_VARI_INPUT_NUM + _VARI_OUTPUT_NUM)
#define _VARI_PARAM_NUM          (_VARI_ARG_NUM + _VARI_IO_NUM)

#define ENABLE_CPU 0

extern vx_kernel_description_t * vx_kernel_INSTANCENORM_list[];

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

    if (index == 0 )
    {
        if( input->attr.dim_num == 1 )
        {
            memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
            attr.size[1] = 1;
            attr.dim_num = 2;
            self->nn_param.instancenorm.local.local_tensor[index] =
                vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
            params[index] =  (vx_reference)self->nn_param.instancenorm.local.local_tensor[index];
        }
        else
            params[index] = (vx_reference)input->t;
    }
    else if(index == 1 )
    {
        if(input->attr.dim_num == 1)
        {
            memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
            attr.size[1] = 1;
            attr.dim_num = 2;
            self->nn_param.instancenorm.local.local_tensor[index] =
                vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
            params[index] =  (vx_reference)self->nn_param.instancenorm.local.local_tensor[index];
        }
        else
             params[index] = (vx_reference)input->t;

    }
    else if(index == 2 )
    {
        if(input->attr.dim_num == 1)
        {
            memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
            attr.size[1] = 1;
            attr.dim_num = 2;
            self->nn_param.instancenorm.local.local_tensor[index] =
                vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
            params[index] =  (vx_reference)self->nn_param.instancenorm.local.local_tensor[index];
        }
        else
             params[index] = (vx_reference)input->t;
    }
    else if(index == 3 )
    {
        if(input->attr.dim_num == 1)
        {
            memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
            attr.size[1] = 1;
            attr.dim_num = 2;
            self->nn_param.instancenorm.local.local_tensor[index] =
                vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
            params[index] =  (vx_reference)self->nn_param.instancenorm.local.local_tensor[index];
        }
        else
             params[index] = (vx_reference)input->t;
    }
    else if(index == 4 )
    {
        if(input->attr.dim_num == 1)
        {
            memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
            attr.size[1] = 1;
            attr.dim_num = 2;
            self->nn_param.instancenorm.local.local_tensor[index] =
                vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
            params[index] =  (vx_reference)self->nn_param.instancenorm.local.local_tensor[index];
        }
        else
             params[index] = (vx_reference)input->t;
    }
    else
    {
        VSILOGE("No more local tensor!(INSTANCENORM) at [%s : %d]\n", __FILE__, __LINE__);
    }
}

#if ENABLE_CPU
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
    vsi_nn_instancenormalize_param * p;
    if( 0 == num )
    {
        return VSI_SUCCESS;
    }
    memset( params, 0, sizeof( vx_reference * ) * num );
    p = &(node->nn_param.instancenorm);
    ctx = vxGetContext( (vx_reference)node->graph->g );
    /* Init parameters */
#define _SET_PARAM( i, type, arg ) do{ \
    params[i] = (vx_reference)vxCreateScalar( ctx, type, &p->arg ); \
    status = vxGetStatus( params[i] ); \
    if( VSI_SUCCESS != status ) { \
    goto set_param_error; \
    } \
    } while(0)
    _SET_PARAM( 0, VX_TYPE_FLOAT32, eps );
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

#if ENABLE_CPU
static vsi_status cpu_op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vx_array ** arrayList
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
    args[1] = (vx_reference)*arrayList[0];
    args[2] = (vx_reference)*arrayList[1];

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, _PARAM_NUM );

    _release_params( args, 1 );

    return status;
}
#endif

static vsi_status vx_sum_op_pre_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_kernel_info_t * kernel_info
    )
{
    vsi_nn_type_e inputDataFormat     = inputs[0]->attr.dtype.vx_type;
    if (inputDataFormat == VSI_NN_TYPE_UINT8)
    {
        kernel_info->kernel_index = 4;
    }
    else
    {
        VSILOGE("Not support input or output data format!(INSTANCENORM_SUM) at [%s : %d]\n", __FILE__, __LINE__);
        return VSI_FAILURE;
    }
    return VSI_SUCCESS;
}

static vsi_status vx_sqr_op_pre_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_kernel_info_t * kernel_info
    )
{
    vsi_nn_type_e inputDataFormat     = inputs[0]->attr.dtype.vx_type;
    if (inputDataFormat == VSI_NN_TYPE_UINT8)
    {
        kernel_info->kernel_index = 5;
    }
    else
    {
        VSILOGE("Not support input or output data format!(INSTANCENORM_SQR) at [%s : %d]\n", __FILE__, __LINE__);
        return VSI_FAILURE;
    }
    return VSI_SUCCESS;
}

static vsi_status vx_mean_vari_op_pre_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_kernel_info_t * kernel_info
    )
{
    vsi_nn_type_e inputDataFormat     = inputs[0]->attr.dtype.vx_type;
    if (inputDataFormat == VSI_NN_TYPE_UINT8)
    {
        kernel_info->kernel_index = 6;
    }
    else
    {
        VSILOGE("Not support input or output data format!(INSTANCENORM_SQR) at [%s : %d]\n", __FILE__, __LINE__);
        return VSI_FAILURE;
    }
    return VSI_SUCCESS;
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
    vsi_nn_type_e scaleDataFormat     = inputs[2]->attr.dtype.vx_type;
    if (inputDataFormat == VSI_NN_TYPE_FLOAT16 && outputDataFormat == VSI_NN_TYPE_FLOAT16)
    {
        kernel_info->kernel_index = 1;
    }
    else if (inputDataFormat == VSI_NN_TYPE_UINT8 && outputDataFormat == VSI_NN_TYPE_UINT8
             && scaleDataFormat == VSI_NN_TYPE_FLOAT16)
    {
        kernel_info->kernel_index = 2;
    }
    else if (inputDataFormat == VSI_NN_TYPE_UINT8 && outputDataFormat == VSI_NN_TYPE_FLOAT16
             && scaleDataFormat == VSI_NN_TYPE_FLOAT16)
    {
        kernel_info->kernel_index = 3;
    }
    else if (inputDataFormat == VSI_NN_TYPE_INT8 && outputDataFormat == VSI_NN_TYPE_INT8
             && scaleDataFormat == VSI_NN_TYPE_FLOAT16)
    {
        kernel_info->resource_name[0] = "vsi_nn_kernel_instancenormalize_i8";
        kernel_info->kernel_index = 7;
    }
    else if (inputDataFormat == VSI_NN_TYPE_INT8 && outputDataFormat == VSI_NN_TYPE_FLOAT16
             && scaleDataFormat == VSI_NN_TYPE_FLOAT16)
    {
        kernel_info->resource_name[0] = "vsi_nn_kernel_instancenormalize_i8";
        kernel_info->kernel_index = 8;
    }
    else if (inputDataFormat == VSI_NN_TYPE_INT16 && outputDataFormat == VSI_NN_TYPE_INT16
             && scaleDataFormat == VSI_NN_TYPE_FLOAT16)
    {
        kernel_info->resource_name[0] = "vsi_nn_kernel_instancenormalize_i8";
        kernel_info->kernel_index = 9;
    }
    else if (inputDataFormat == VSI_NN_TYPE_INT16 && outputDataFormat == VSI_NN_TYPE_FLOAT16
             && scaleDataFormat == VSI_NN_TYPE_FLOAT16)
    {
        kernel_info->resource_name[0] = "vsi_nn_kernel_instancenormalize_i8";
        kernel_info->kernel_index = 10;
    }
    else
    {
        VSILOGE("Not support input or output data format!(INSTANCENORM) at [%s : %d]\n", __FILE__, __LINE__);
        return VSI_FAILURE;
    }
    return VSI_SUCCESS;
}

static vsi_status vx_sum_op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vx_array ** arrayList
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_reference params[_SUM_PARAM_NUM];
    vx_border_t border;
    vx_reference * args;
    vx_bool rsFlg = vx_false_e;
    int32_t in_zp;
    vsi_nn_type_e inputDataFormat = inputs[0]->attr.dtype.vx_type;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_t* tmpTensor = NULL;
    vsi_nn_tensor_attr_t input_attr;

    memset(&input_attr, 0, sizeof(vsi_nn_tensor_attr_t));

    args = &params[_SUM_IO_NUM];

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    /* Set inputs and outputs */
    check_tensor_shape(self, inputs[0], params, 0, rsFlg);
    //check_tensor_shape(self, outputs[0], params, 1, rsFlg);
#if 1
    {
        memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
        attr.size[0] = 1;
        attr.size[1] = 1;
        attr.size[2] = 1;
        attr.size[3] = 1;
        attr.dim_num = 3;
        attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
        attr.vtl = FALSE;
        tmpTensor = vsi_nn_CreateTensor(self->graph, &attr);
        check_tensor_shape(self, tmpTensor, params, 1, rsFlg);
    }
#endif
    status  = vsi_nn_vxGetTensorAttr(inputs[0]->t, &input_attr);
    if (status != VX_SUCCESS)
    {
        VSILOGE("vsi_nn_vxGetTensorAttr  failure! at line %d\n", __LINE__);
        return status;
    }
    in_zp = input_attr.dtype.zero_point;
    /* Init parameters. */
    args[0] = (vx_reference)*arrayList[0];
    args[1] = (vx_reference)*arrayList[1];

    /* Pass parameters to node. */
    status |= vsi_nn_ClientNodePassParameters( self->n, params, _SUM_PARAM_NUM );

    //_release_params( args, _ARG_NUM );

    border.mode = VX_BORDER_CONSTANT;
    border.constant_value.U32 = 0;
    border.constant_value.S16 = 0;
    border.constant_value.U8 = 0;
    if(inputDataFormat == VSI_NN_TYPE_UINT8)
    {
        border.constant_value.U32 = in_zp;
        border.constant_value.S16 = in_zp;
        border.constant_value.U8 = in_zp;
    }
    status |= vxSetNodeAttribute(self->n, VX_NODE_BORDER, &border, sizeof(border));

    if(tmpTensor)vsi_nn_ReleaseTensor(&tmpTensor);

    return status;
}

static vsi_status vx_sqr_op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vx_array ** arrayList
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_reference params[_SQR_PARAM_NUM];
    vx_border_t border;
    vx_reference * args;
    vx_bool rsFlg = vx_false_e;
    int32_t in_zp;
    vsi_nn_type_e inputDataFormat = inputs[0]->attr.dtype.vx_type;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_t* tmpTensor = NULL;
    vsi_nn_tensor_attr_t input_attr;

    memset(&input_attr, 0, sizeof(vsi_nn_tensor_attr_t));

    args = &params[_SQR_IO_NUM];

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    /* Set inputs and outputs */
    check_tensor_shape(self, inputs[0], params, 0, rsFlg);
    //check_tensor_shape(self, outputs[1], params, 1, rsFlg);
#if 1
    {
        memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
        attr.size[0] = 1;
        attr.size[1] = 1;
        attr.size[2] = 1;
        attr.size[3] = 1;
        attr.dim_num = 3;
        attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
        attr.vtl = FALSE;
        tmpTensor = vsi_nn_CreateTensor(self->graph, &attr);
        check_tensor_shape(self, tmpTensor, params, 1, rsFlg);
    }
#endif
    _create_params( self, args, _SQR_ARG_NUM );
    /* Init parameters. */
    args[1] = (vx_reference)*arrayList[0];
    args[2] = (vx_reference)*arrayList[1];
    args[3] = (vx_reference)*arrayList[2];
    args[4] = (vx_reference)*arrayList[3];

    /* Pass parameters to node. */
    status  = vsi_nn_ClientNodePassParameters( self->n, params, _SQR_PARAM_NUM );
    status |= vsi_nn_vxGetTensorAttr(inputs[0]->t, &input_attr);
    in_zp = input_attr.dtype.zero_point;
    _release_params( args, 1 );
    border.mode = VX_BORDER_CONSTANT;
    border.constant_value.U32 = 0;
    border.constant_value.S16 = 0;
    border.constant_value.U8 = 0;
    if(inputDataFormat == VSI_NN_TYPE_UINT8)
    {
        border.constant_value.U32 = in_zp;
        border.constant_value.S16 = in_zp;
        border.constant_value.U8 = in_zp;
    }
    status |= vxSetNodeAttribute(self->n, VX_NODE_BORDER, &border, sizeof(border));

    if(tmpTensor)vsi_nn_ReleaseTensor(&tmpTensor);

    return status;
}

static vsi_status vx_mean_vari_op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t * output
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_reference params[_VARI_PARAM_NUM];
    vx_border_t border;
    vx_reference * args;
    vx_bool rsFlg = vx_false_e;
    int32_t in_zp;
    vsi_nn_type_e inputDataFormat = inputs[0]->attr.dtype.vx_type;
    vsi_nn_tensor_attr_t input_attr;

    memset(&input_attr, 0, sizeof(vsi_nn_tensor_attr_t));
    //vsi_nn_tensor_attr_t attr;
    //vsi_nn_tensor_t* tmpTensor = NULL;
    args = &params[_VARI_IO_NUM];

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    /* Set inputs and outputs */
    check_tensor_shape(self, inputs[0], params, 0, rsFlg);
    check_tensor_shape(self, output, params, 1, rsFlg);
#if 0
    {
        memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
        attr.size[0] = 1;
        attr.size[1] = 1;
        attr.size[2] = 1;
        attr.size[3] = 1;
        attr.dim_num = 3;
        attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
        attr.vtl = FALSE;
        tmpTensor = vsi_nn_CreateTensor(self->graph, &attr);
        check_tensor_shape(self, tmpTensor, params, 1, rsFlg);
    }
#endif
    _create_params( self, args, _VARI_ARG_NUM );

    /* Pass parameters to node. */
    status  = vsi_nn_ClientNodePassParameters( self->n, params, _VARI_PARAM_NUM );
    status |= vsi_nn_vxGetTensorAttr(inputs[0]->t, &input_attr);
    in_zp = input_attr.dtype.zero_point;
    _release_params( args, 1 );

    border.mode = VX_BORDER_CONSTANT;
    border.constant_value.U32 = 0;
    border.constant_value.S16 = 0;
    border.constant_value.U8 = 0;
    if(inputDataFormat == VSI_NN_TYPE_UINT8)
    {
        border.constant_value.U32 = in_zp;
        border.constant_value.S16 = in_zp;
        border.constant_value.U8 = in_zp;
    }
    status |= vxSetNodeAttribute(self->n, VX_NODE_BORDER, &border, sizeof(border));

    //if(tmpTensor)vsi_nn_ReleaseTensor(&tmpTensor);

    return status;
}

static vsi_status vx_op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_tensor_t * tmpInput
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_reference params[_PARAM_NUM];
    vx_border_t border;
    vx_reference * args;
    vx_bool rsFlg = vx_false_e;
    int32_t in_zp;
    vsi_nn_type_e inputDataFormat = inputs[0]->attr.dtype.vx_type;
    vsi_nn_tensor_attr_t input_attr;

    memset(&input_attr, 0, sizeof(vsi_nn_tensor_attr_t));
    args = &params[_IO_NUM];

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    /* Set inputs and outputs */
    //_set_inputs_outputs( params, inputs, outputs );
    check_tensor_shape(self, inputs[0], params, 0, rsFlg);
    check_tensor_shape(self, inputs[1], params, 1, rsFlg);
    check_tensor_shape(self, inputs[2], params, 2, rsFlg);
    check_tensor_shape(self, outputs[0], params, 3, rsFlg);
    check_tensor_shape(self, tmpInput, params, 4, rsFlg);
    status = vsi_nn_vxGetTensorAttr(inputs[0]->t, &input_attr);
    in_zp = input_attr.dtype.zero_point;
    /* Init parameters. */
    _create_params( self, args, _ARG_NUM );

    /* Pass parameters to node. */
    status |= vsi_nn_ClientNodePassParameters( self->n, params, _PARAM_NUM );

    _release_params( args, 1 );

    border.mode = VX_BORDER_CONSTANT;
    border.constant_value.U32 = 0;
    border.constant_value.S16 = 0;
    border.constant_value.U8 = 0;
    if(inputDataFormat == VSI_NN_TYPE_UINT8)
    {
        border.constant_value.U32 = in_zp;
        border.constant_value.S16 = in_zp;
        border.constant_value.U8 = in_zp;
    }
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
    vsi_status status;
    vsi_nn_kernel_info_t kernel_info;

    vsi_nn_type_e inputDataFormat     = inputs[0]->attr.dtype.vx_type;
    uint32_t input_size[4] = {1, 1, 1, 1};
    vx_array* array_list[4] = {0};
    vx_array arraySum = NULL;
    vx_array arraySqr = NULL;
    vx_array resultSum = NULL;
    vx_array resultSqr = NULL;
    vsi_nn_tensor_t* tmpMeanVari = NULL;
    uint32_t input_dims = 1;
    uint32_t i = 0;
    vsi_nn_tensor_attr_t input_attr;

    status = VSI_FAILURE;
    memset(&kernel_info, 0x0, sizeof(vsi_nn_kernel_info_t));
    memset(&input_attr, 0, sizeof(vsi_nn_tensor_attr_t));
    status = vsi_nn_vxGetTensorAttr(inputs[0]->t, &input_attr);
    if (status != VX_SUCCESS)
    {
        VSILOGE("vsi_nn_vxGetTensorAttr  failure! at line %d\n", __LINE__);
        return status;
    }
    input_dims  = input_attr.dim_num;
    for (i = 0; i < input_dims; i++)
    {
        input_size[i] = input_attr.size[i];
    }

    {
        vx_context ctx;
        vsi_nn_tensor_attr_t attr;
        vx_int32 iter = (input_size[1] + 7) / 8;

        ctx = vxGetContext( (vx_reference)self->graph->g );
        arraySum = vxCreateArray(ctx, VX_TYPE_INT32, input_size[2] * sizeof(vx_int32));
        arraySqr = vxCreateArray(ctx, VX_TYPE_INT32, input_size[2] * iter * sizeof(vx_int32));
        resultSum = vxCreateArray(ctx, VX_TYPE_FLOAT32, input_size[2] * sizeof(vx_float32));
        resultSqr = vxCreateArray(ctx, VX_TYPE_FLOAT32, input_size[2] * sizeof(vx_float32));
        array_list[0] = &arraySum;
        array_list[1] = &arraySqr;
        array_list[2] = &resultSum;
        array_list[3] = &resultSqr;

        memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
        attr.size[0] = input_size[2] * 4;
        attr.size[1] = 1;
        attr.size[2] = 1;
        attr.size[3] = 1;
        attr.dim_num = 2;
        attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
        attr.vtl = FALSE;
        tmpMeanVari = vsi_nn_CreateTensor(self->graph, &attr);
    }

#if ENABLE_CPU //cpu
    {
        kernel_info.resource_num = 1;
        kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
        kernel_info.resource_name[0] = "vsi_nn_kernel_instancenormalize";
        kernel_info.type = VX_KERNEL_TYPE_CPU;
        kernel_info.kernel = vx_kernel_INSTANCENORM_list;
        kernel_info.init_index = 0;

        self->n = vsi_nn_RegisterClientKernelAndNewNode(
        self->graph, &kernel_info);
        if (kernel_info.resource_name) free(kernel_info.resource_name);
        if( NULL == self->n )
        {
            return VSI_FAILURE;
        }

        status = cpu_op_compute(self, inputs, outputs, array_list);

        if(arraySum)vxReleaseArray(&arraySum);
        if(arraySqr)vxReleaseArray(&arraySqr);
        if(resultSum)vxReleaseArray(&resultSum);
        if(resultSqr)vxReleaseArray(&resultSqr);
        return status;
    }
#endif

    // prepare sum & sqr
    if(inputDataFormat == VSI_NN_TYPE_UINT8 && 0)
    {
        kernel_info.resource_num = 1;
        kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
        kernel_info.resource_name[0] = "vsi_nn_kernel_instancenormalize";
        kernel_info.type = VX_KERNEL_TYPE_VX;
        kernel_info.kernel = vx_kernel_INSTANCENORM_list;
        kernel_info.init_index = 1;
        if (kernel_info.type == VX_KERNEL_TYPE_VX)
        {
            vx_sum_op_pre_compute(self, inputs, outputs, &kernel_info);
        }

        self->n = vsi_nn_RegisterClientKernelAndNewNode(
            self->graph, &kernel_info);
        if (kernel_info.resource_name) free(kernel_info.resource_name);
        if( NULL == self->n )
        {
            status = VSI_FAILURE;
            goto OnError;
        }
        status |= vx_sum_op_compute(self, inputs, outputs, array_list);
    }

    // float sum & sqr
    if(inputDataFormat == VSI_NN_TYPE_UINT8 && 0)
    {
        kernel_info.resource_num = 1;
        kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
        kernel_info.resource_name[0] = "vsi_nn_kernel_instancenormalize";
        kernel_info.type = VX_KERNEL_TYPE_VX;
        kernel_info.kernel = vx_kernel_INSTANCENORM_list;
        kernel_info.init_index = 1;
        if (kernel_info.type == VX_KERNEL_TYPE_VX)
        {
            vx_sqr_op_pre_compute(self, inputs, outputs, &kernel_info);
        }

        self->n = vsi_nn_RegisterClientKernelAndNewNode(
            self->graph, &kernel_info);
        if (kernel_info.resource_name) free(kernel_info.resource_name);
        if( NULL == self->n )
        {
            status = VSI_FAILURE;
            goto OnError;
        }
        status |= vx_sqr_op_compute(self, inputs, outputs, array_list);
    }

    if(inputDataFormat == VSI_NN_TYPE_UINT8)
    {
        kernel_info.resource_num = 1;
        kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
        kernel_info.resource_name[0] = "vsi_nn_kernel_instancenormalize";
        kernel_info.type = VX_KERNEL_TYPE_VX;
        kernel_info.kernel = vx_kernel_INSTANCENORM_list;
        kernel_info.init_index = 1;
        if (kernel_info.type == VX_KERNEL_TYPE_VX)
        {
            vx_mean_vari_op_pre_compute(self, inputs, outputs, &kernel_info);
        }

        self->n = vsi_nn_RegisterClientKernelAndNewNode(
            self->graph, &kernel_info);
        if (kernel_info.resource_name) free(kernel_info.resource_name);
        if( NULL == self->n )
        {
            status = VSI_FAILURE;
            goto OnError;
        }
        status |= vx_mean_vari_op_compute(self, inputs, tmpMeanVari);
    }

    kernel_info.resource_num = 1;
    kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
    kernel_info.resource_name[0] = "vsi_nn_kernel_instancenormalize";
    kernel_info.type = vsi_nn_GetVXKernelTypeForShader();
    kernel_info.kernel = vx_kernel_INSTANCENORM_list;
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
        status = VSI_FAILURE;
        goto OnError;
    }

    status |= vx_op_compute(self, inputs, outputs, tmpMeanVari);

OnError:
    if(arraySum)vxReleaseArray(&arraySum);
    if(arraySqr)vxReleaseArray(&arraySqr);
    if(resultSum)vxReleaseArray(&resultSum);
    if(resultSqr)vxReleaseArray(&resultSqr);
    if(tmpMeanVari)vsi_nn_ReleaseTensor(&tmpMeanVari);

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
    for (i = 0; i < _VSI_NN_INSTANCENORM_LOCAL_TENSOR_NUM; i++)
    {
        if (self->nn_param.instancenorm.local.local_tensor[i] != NULL)
        {
            vxReleaseTensor(&(self->nn_param.instancenorm.local.local_tensor[i]));
            self->nn_param.instancenorm.local.local_tensor[i] = NULL;
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
    /* op_name    */ INSTANCE_NORM,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ vsi_nn_op_common_setup,
    /* optimize   */ NULL,
    /* input_num  */ 3,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif

