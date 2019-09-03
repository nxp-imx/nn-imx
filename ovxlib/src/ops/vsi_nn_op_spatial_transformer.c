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
#include "utils/vsi_nn_dtype_util.h"

#define _ARG_NUM            (2)
#define _INPUT_NUM          (2)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)
#define _VSI_PARAM          (vsi_nn_spatial_transformer_param)

extern vx_kernel_description_t * vx_kernel_SPATIAL_TRANSFORMER_list[];

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
    vsi_status status = VSI_SUCCESS;
    vx_context ctx;
    vsi_nn_spatial_transformer_param * p;
    int  flag;
    vsi_nn_tensor_t * thre_tensor;
    vsi_nn_tensor_attr_t attr;

    vx_uint16 value_buf[6];

    if( 0 == num )
    {
        return VSI_SUCCESS;
    }
    memset( params, 0, sizeof( vx_reference * ) * num );
    p = (vsi_nn_spatial_transformer_param *)node->nn_param.client_param;
    ctx = vxGetContext( (vx_reference)node->graph->g );

    flag = ((p->has_theta_1_1 == 1)
            | ((p->has_theta_1_2 == 1) << 1)
            | ((p->has_theta_1_3 == 1) << 2)
            | ((p->has_theta_2_1 == 1) << 3)
            | ((p->has_theta_2_2 == 1) << 4)
            | ((p->has_theta_2_3 == 1) << 5));

    params[0] = (vx_reference)vxCreateScalar( ctx, VSI_NN_TYPE_INT32, &flag );

    memset( &attr, 0, sizeof( vsi_nn_tensor_attr_t ) );
    attr.size[0] = 6;
    attr.size[1] = 1;
    attr.size[2] = 1;
    attr.size[3] = 1;
    attr.dim_num = 4;
    attr.is_const = TRUE;
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;

    vsi_nn_Float32ToDtype(p->theta_1_1, (uint8_t*)(&value_buf[0]), &attr.dtype);
    vsi_nn_Float32ToDtype(p->theta_1_2, (uint8_t*)(&value_buf[1]), &attr.dtype);
    vsi_nn_Float32ToDtype(p->theta_1_3, (uint8_t*)(&value_buf[2]), &attr.dtype);
    vsi_nn_Float32ToDtype(p->theta_2_1, (uint8_t*)(&value_buf[3]), &attr.dtype);
    vsi_nn_Float32ToDtype(p->theta_2_2, (uint8_t*)(&value_buf[4]), &attr.dtype);
    vsi_nn_Float32ToDtype(p->theta_2_3, (uint8_t*)(&value_buf[5]), &attr.dtype);

    thre_tensor = vsi_nn_CreateTensorFromData( node->graph,(uint8_t *)&value_buf, &attr );

    params[1] = (vx_reference)thre_tensor->t;
#if 0
    /* Init parameters */
    #define _SET_PARAM( i, type, arg ) do{ \
        params[i] = (vx_reference)vxCreateScalar( ctx, type, &p->arg ); \
        status = vxGetStatus( params[i] ); \
        if( VSI_SUCCESS != status ) { \
            goto set_param_error; \
            } \
        } while(0)
    _SET_PARAM( 0, VSI_NN_TYPE_FLOAT32, has_theta_1_3 );
    _SET_PARAM( 1, VSI_NN_TYPE_FLOAT32, has_theta_2_1 );
    _SET_PARAM( 2, VSI_NN_TYPE_FLOAT32, has_theta_1_2 );
    _SET_PARAM( 3, VSI_NN_TYPE_FLOAT32, theta_2_1 );
    _SET_PARAM( 4, VSI_NN_TYPE_FLOAT32, has_output_W );
    _SET_PARAM( 5, VSI_NN_TYPE_INT32, output_W );
    _SET_PARAM( 6, VSI_NN_TYPE_FLOAT32, theta_1_3 );
    _SET_PARAM( 7, VSI_NN_TYPE_FLOAT32, theta_2_2 );
    _SET_PARAM( 8, VSI_NN_TYPE_FLOAT32, theta_1_2 );
    _SET_PARAM( 9, VSI_NN_TYPE_INT32, output_H );
    _SET_PARAM( 10, VSI_NN_TYPE_FLOAT32, has_theta_2_3 );
    _SET_PARAM( 11, VSI_NN_TYPE_FLOAT32, theta_2_3 );
    _SET_PARAM( 12, VSI_NN_TYPE_FLOAT32, has_theta_2_2 );
    _SET_PARAM( 13, VSI_NN_TYPE_FLOAT32, has_output_H );
    _SET_PARAM( 14, VSI_NN_TYPE_FLOAT32, has_theta_1_1 );
    _SET_PARAM( 15, VSI_NN_TYPE_FLOAT32, theta_1_1 );
    #undef _SET_PARAM
#endif
//set_param_error:

    return status;
} /* _create_params */

static void _release_params
    (
    vx_reference * params,
    uint32_t num
    )
{
    uint32_t i;
    vx_tensor tensor;
    for( i = 0; i < num; i ++ )
    {
        tensor = (vx_tensor)params[i];
        vxReleaseTensor( &tensor );
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
    //_set_inputs_outputs( params, inputs, outputs );

    /* Init parameters. */
    _create_params( self, args, _ARG_NUM );

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, _PARAM_NUM );

    _release_params( args, _ARG_NUM );

    return status;
}

int setUPGridData(vx_uint32 output_W_, vx_uint32 output_H_, vx_float32 scale, vx_int32 zeropoint,
                  vsi_nn_dtype_t data_type, vsi_nn_qnt_type_e qnt_type, vx_uint8 fp, vx_uint8 *tensorData)
{
    vx_uint32 x                = 0;
    vx_uint32 y                = 0;
    //float     fval            = 0.0;
    vx_uint32 idx             = 0;
    float *tmp_buf = NULL;
    vx_uint32 i = 0;

    tmp_buf = (float*) malloc(output_W_ * output_H_ * 3 * sizeof(float));

    for (y = 0; y < output_H_; y++)
    {
        for (x = 0; x < output_W_; x++)
        {
            float data0 = y * (float)1.0 / (float)output_H_ * 2 - 1;
            float data1 = x * (float)1.0 / (float)output_W_ * 2 - 1;
            float data2 = 1;

            tmp_buf[idx++] = data0;
            tmp_buf[idx++] = data1;
            tmp_buf[idx++] = data2;

            //vxnneSaveDataExt(data_type, qnt_type, idx++, data0, tensorData, fp, zeropoint, scale);
            //vxnneSaveDataExt(data_type, qnt_type, idx++, data1, tensorData, fp, zeropoint, scale);
            //vxnneSaveDataExt(data_type, qnt_type, idx++, data2, tensorData, fp, zeropoint, scale);
        }
    }

    for(i = 0; i < output_H_ * output_W_ * 3; i++)
    {
        vsi_nn_Float32ToDtype(tmp_buf[i],(uint8_t*)&tensorData[vsi_nn_GetTypeBytes(data_type.vx_type)*i], &data_type);
    }

    if(tmp_buf)
    {
        free(tmp_buf);
        tmp_buf = NULL;
    }

    return 0;
}

#if 0
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
    _set_inputs_outputs( params, inputs, outputs );
    /*TODO: Add code if need to change your parameter*/

    /* Init parameters. */
    _create_params( self, args, _ARG_NUM );

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, _PARAM_NUM );

    _release_params( args, _ARG_NUM );

    return status;
}
#endif

static vsi_status vx_op_compute_setupThre
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_reference params[4];
    //vx_reference * args;
    vsi_nn_spatial_transformer_param * p;
    int  flag;
    vsi_nn_tensor_t * thre_tensor;
    vsi_nn_tensor_attr_t attr;
    vx_context ctx;
    vx_scalar flag_s;
    vx_tensor tmp_t, tmp_t1;

    //float flag_buf[6];
    vx_uint16 value_buf[6];

    memset( params, 0, sizeof( vx_reference * ) * 4 );
    p = (vsi_nn_spatial_transformer_param *)self->nn_param.client_param;
    ctx = vxGetContext( (vx_reference)self->graph->g );

    flag = ((p->has_theta_1_1 == 1)
            | ((p->has_theta_1_2 == 1) << 1)
            | ((p->has_theta_1_3 == 1) << 2)
            | ((p->has_theta_2_1 == 1) << 3)
            | ((p->has_theta_2_2 == 1) << 4)
            | ((p->has_theta_2_3 == 1) << 5));

    memset( &attr, 0, sizeof( vsi_nn_tensor_attr_t ) );
    attr.size[0] = 6;
    attr.size[1] = 1;
    attr.size[2] = 1;
    attr.size[3] = 1;
    attr.dim_num = 4;
    attr.is_const = TRUE;
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;

    vsi_nn_Float32ToDtype(p->theta_1_1, (uint8_t*)(&value_buf[0]), &attr.dtype);
    vsi_nn_Float32ToDtype(p->theta_1_2, (uint8_t*)(&value_buf[1]), &attr.dtype);
    vsi_nn_Float32ToDtype(p->theta_1_3, (uint8_t*)(&value_buf[2]), &attr.dtype);
    vsi_nn_Float32ToDtype(p->theta_2_1, (uint8_t*)(&value_buf[3]), &attr.dtype);
    vsi_nn_Float32ToDtype(p->theta_2_2, (uint8_t*)(&value_buf[4]), &attr.dtype);
    vsi_nn_Float32ToDtype(p->theta_2_3, (uint8_t*)(&value_buf[5]), &attr.dtype);

    thre_tensor = vsi_nn_CreateTensorFromData( self->graph,(uint8_t *)&value_buf, &attr );

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    flag_s = vxCreateScalar( ctx, VSI_NN_TYPE_INT32, &flag );

    params[0] = (vx_reference)thre_tensor->t;

    attr.size[0] = inputs[1]->attr.size[0] * inputs[1]->attr.size[1];
    attr.size[1] = 1;
    attr.size[2] = inputs[1]->attr.size[2];
    attr.size[3] = inputs[1]->attr.size[3];
    attr.dim_num = inputs[1]->attr.dim_num;

    tmp_t = vxReshapeTensor(inputs[1]->t, (vx_int32*)attr.size, attr.dim_num);

    params[1] = (vx_reference)tmp_t;
    params[2] = (vx_reference)flag_s;

    attr.size[0] = outputs[0]->attr.size[0] * outputs[0]->attr.size[1];
    attr.size[1] = 1;
    attr.size[2] = outputs[0]->attr.size[2];
    attr.size[3] = outputs[0]->attr.size[3];
    attr.dim_num = outputs[0]->attr.dim_num;

    tmp_t1 = vxReshapeTensor(outputs[0]->t, (vx_int32*)attr.size,attr.dim_num);

    params[3] = (vx_reference)tmp_t1;

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, 4 );

    //_release_params( args, 4 );

    vxReleaseTensor( &thre_tensor->t );
    vxReleaseTensor( &tmp_t );
    vxReleaseTensor( &tmp_t1 );
    vxReleaseScalar( &flag_s );

    return status;
}


static vsi_status vx_op_compute_gemm
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_reference params[3];
    vx_tensor paraTensor0, paraTensor1, paraTensor2;
    vsi_nn_spatial_transformer_param * p;

    vx_context ctx;
    int     size[4]    = {1};
    vx_tensor_addressing out_addr;
    vsi_nn_tensor_attr_t out_attr;
    uint32_t out_stride[6];
    uint8_t *out_buffer = NULL;

    p = (vsi_nn_spatial_transformer_param *)self->nn_param.client_param;
    ctx = vxGetContext( (vx_reference)self->graph->g );

    out_buffer = (uint8_t *)vsi_nn_ConvertRawTensorToData2(ctx,inputs[1]->t,
                                                            &out_attr,out_stride,&out_addr,VX_WRITE_ONLY);

    setUPGridData(p->output_W, p->output_H, out_attr.dtype.scale, out_attr.dtype.zero_point,
        out_attr.dtype, out_attr.dtype.qnt_type ,out_attr.dtype.fl, out_buffer);

    status = vxCopyTensorPatch(inputs[1]->t,NULL,out_addr,out_buffer,VX_WRITE_ONLY,0);

    memset( params, 0, sizeof( vx_reference * ) * 3 );

    size[0] = inputs[0]->attr.size[0] * inputs[0]->attr.size[1];
    size[1] = 1;
    paraTensor0 = vxReshapeTensor(inputs[0]->t,size,2);

    size[0] = inputs[1]->attr.size[0] * p->output_W;
    size[1] = p->output_H;
    paraTensor1 = vxReshapeTensor(inputs[1]->t,size,2);

    size[0] = inputs[0]->attr.size[1] * p->output_W;
    size[1] = p->output_H;
    paraTensor2 = vxReshapeTensor(inputs[2]->t,size,2);

    if (out_addr)
    {
        vxReleaseTensorAddressing(&out_addr);
    }

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    params[0] = (vx_reference)paraTensor0;
    params[1] = (vx_reference)paraTensor1;
    params[2] = (vx_reference)paraTensor2;
    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, 3 );

    vxReleaseTensor(&paraTensor0);
    vxReleaseTensor(&paraTensor1);
    vxReleaseTensor(&paraTensor2);

    return status;
}


static vsi_status vx_op_compute_interp
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_reference params[3];
    vx_border_t border;

    memset( params, 0, sizeof( vx_reference * ) * 3 );

    params[0] = (vx_reference)inputs[3]->t;
    params[1] = (vx_reference)inputs[2]->t;
    params[2] =(vx_reference)outputs[0]->t;

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, 3 );

    border.mode = VX_BORDER_CONSTANT;
    border.constant_value.S16 = 0;

    status |= vxSetNodeAttribute(self->n, VX_NODE_BORDER,
        &border, sizeof(border));
   // _release_params( args, 3 );

    return status;
}

static vsi_nn_op_compute_t op_compute_list[] =
{
    cpu_op_compute,
    vx_op_compute_setupThre,
    vx_op_compute_gemm,
    vx_op_compute_interp,
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
    char *path = NULL;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_t *tmp_output_tensor[5] = {0};
    vsi_nn_spatial_transformer_param * p;

    p = (vsi_nn_spatial_transformer_param *)self->nn_param.client_param;

    // Tensor for thre_output
    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    attr.size[0] = 3;
    attr.size[1] = 2;
    attr.size[2] = 1;
    attr.size[3] = 1;
    attr.dim_num = 2;
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    attr.vtl = FALSE;
    tmp_output_tensor[0] = vsi_nn_CreateTensor(self->graph, &attr);

    // Tensor for grid
    attr.size[0] = 3;
    attr.size[1] = p->output_H * p->output_W;
    attr.size[2] = 1;
    attr.size[3] = 1;
    attr.dim_num = 2;
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    attr.vtl = FALSE;
    tmp_output_tensor[1] = vsi_nn_CreateTensor(self->graph, &attr);

    // Tensor for grid_out
    attr.size[0] = 2 * p->output_W;
    attr.size[1] = p->output_H ;
    attr.size[2] = 1;
    attr.size[3] = 1;
    attr.dim_num = 2;
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    attr.vtl = FALSE;
    tmp_output_tensor[2] = vsi_nn_CreateTensor(self->graph, &attr);


    status = VSI_FAILURE;
#if 0
    kernel_info.type = VX_KERNEL_TYPE_CPU;
    kernel_info.kernel = vx_kernel_SPATIAL_TRANSFORMER_list;
    kernel_info.resource_num = 1;
    kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
    kernel_info.resource_name[0] = "spatialtransformer";
#else

    kernel_info.type = VX_KERNEL_TYPE_VX;
    kernel_info.kernel = vx_kernel_SPATIAL_TRANSFORMER_list;
    kernel_info.resource_num = 1;
    kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
    kernel_info.resource_name[0] = "vsi_nn_kernel_transform_setupThres";

    //kernel_info.resource_name[0] = "";

#endif
    path = getenv("USER_VX_SOURCE_PATH");
    if(path)
        vsi_nn_VxResourceSetPath(path);

    kernel_info.kernel_index = 1;
    kernel_info.init_index = 1;

    // add setupThre
    self->n = vsi_nn_RegisterClientKernelAndNewNode(
            self->graph, &kernel_info);

    if (NULL != op_compute_list[kernel_info.init_index])
    {
        status = op_compute_list[kernel_info.init_index](self, inputs, tmp_output_tensor);
    }

     if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    // add gemm
    kernel_info.kernel_index = 2;
    kernel_info.init_index = 2;
    kernel_info.resource_name[0] = "vsi_nn_kernel_transform_gemm";
    self->n = vsi_nn_RegisterClientKernelAndNewNode(
            self->graph, &kernel_info);

    if (NULL != op_compute_list[kernel_info.init_index])
    {
        status = op_compute_list[kernel_info.init_index](self, tmp_output_tensor, outputs);
    }

    // add interp
    if(inputs[0]->attr.dim_num == 2 && inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT16
            && outputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT16)
        kernel_info.kernel_index = 3;
    else if(inputs[0]->attr.dim_num == 4 && inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT16
            && outputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT16)
        kernel_info.kernel_index = 4;
    kernel_info.init_index = 3;
    kernel_info.resource_name[0] = "vsi_nn_kernel_transform_interp";
    self->n = vsi_nn_RegisterClientKernelAndNewNode(
            self->graph, &kernel_info);
    if (kernel_info.resource_name)
    {
        free(kernel_info.resource_name);
    }

    tmp_output_tensor[3] = inputs[0];

    if (NULL != op_compute_list[kernel_info.init_index])
    {
        status = op_compute_list[kernel_info.init_index](self, tmp_output_tensor, outputs);
    }

    vsi_nn_ReleaseTensor(&tmp_output_tensor[0]);
    vsi_nn_ReleaseTensor(&tmp_output_tensor[1]);
    vsi_nn_ReleaseTensor(&tmp_output_tensor[2]);

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
    /* TODO: Add code to comput outputs' shape. */
    vsi_nn_spatial_transformer_param * p;
    p = (vsi_nn_spatial_transformer_param *)&node->nn_param.client_param;

    outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
    outputs[0]->attr.size[0] = p->output_W;  // W
    outputs[0]->attr.size[1] = p->output_H;  // H
    outputs[0]->attr.size[2] = inputs[0]->attr.size[2]; // C
    outputs[0]->attr.size[3] = inputs[0]->attr.size[3]; // N
    return TRUE;
} /* op_setup() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ SPATIAL_TRANSFORMER,
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
