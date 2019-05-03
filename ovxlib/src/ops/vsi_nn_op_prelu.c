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

#define _ARG_NUM            (0)
#define _INPUT_NUM          (2)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)

#define USE_OVX_API TRUE

#if (USE_OVX_API == FALSE)
extern vx_kernel_description_t * vx_kernel_PRELU_list[];

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
            attr.size[2] = 1;
            attr.dim_num = 2;
            self->nn_param.prelu.local.local_tensor[index] =
                vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
            params[index] =  (vx_reference)self->nn_param.prelu.local.local_tensor[index];
        }
        else if(input->attr.dim_num == 2 && rsFlg)
        {
            memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
            attr.size[0] *= attr.size[1];
            attr.size[1] = 1;
            self->nn_param.prelu.local.local_tensor[index] =
                vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
            params[index] =  (vx_reference)self->nn_param.prelu.local.local_tensor[index];
        }
        else if(input->attr.dim_num == 3 && rsFlg)
        {
            memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
            attr.size[0] *= attr.size[1];
            attr.size[1] = attr.size[2];
            attr.size[2] = 1;
            self->nn_param.prelu.local.local_tensor[index] =
                vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
            params[index] =  (vx_reference)self->nn_param.prelu.local.local_tensor[index];
        }
        else if(input->attr.dim_num == 4 && rsFlg)
        {
            memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
            attr.size[0] *= attr.size[1];
            attr.size[1] = attr.size[2];
            attr.size[2] = 1;
            attr.size[3] = attr.size[3];
            self->nn_param.prelu.local.local_tensor[index] =
                vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
            params[index] =  (vx_reference)self->nn_param.prelu.local.local_tensor[index];
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
            self->nn_param.prelu.local.local_tensor[index] =
                vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
            params[index] =  (vx_reference)self->nn_param.prelu.local.local_tensor[index];
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
            self->nn_param.prelu.local.local_tensor[index] =
                vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
            params[index] =  (vx_reference)self->nn_param.prelu.local.local_tensor[index];
        }
        else if(input->attr.dim_num == 2 && rsFlg)
        {
            memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
            attr.size[0] *= attr.size[1];
            attr.size[1] = 1;
            self->nn_param.prelu.local.local_tensor[index] =
                vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
            params[index] =  (vx_reference)self->nn_param.prelu.local.local_tensor[index];
        }
        else if(input->attr.dim_num == 3 && rsFlg)
        {
            memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
            attr.size[0] *= attr.size[1];
            attr.size[1] = attr.size[2];
            attr.size[2] = 1;
            self->nn_param.prelu.local.local_tensor[index] =
                vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
            params[index] =  (vx_reference)self->nn_param.prelu.local.local_tensor[index];
        }
        else if(input->attr.dim_num == 4 && rsFlg)
        {
            memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
            attr.size[0] *= attr.size[1];
            attr.size[1] = attr.size[2];
            attr.size[2] = 1;
            attr.size[3] = attr.size[3];
            self->nn_param.prelu.local.local_tensor[index] =
                vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
            params[index] =  (vx_reference)self->nn_param.prelu.local.local_tensor[index];
        }
        else
             params[index] = (vx_reference)input->t;
    }
}

static void _set_inputs_outputs
    (
    vsi_nn_node_t * self,
    vx_reference * params,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    uint32_t i;
    uint32_t cnt;

    /* Set inputs */
    params[0] = (vx_reference)inputs[0]->t;
    if (self->nn_param.prelu.local.local_tensor[0] == NULL)
    {
        params[1] = (vx_reference)inputs[1]->t;
    }
    else
    {
        params[1] = (vx_reference)self->nn_param.prelu.local.local_tensor[0];
    }
    cnt = 2;

    /* Set outputs */
    for( i = 0; i < _OUTPUT_NUM; i ++, cnt ++ )
    {
        params[cnt] = (vx_reference)outputs[i]->t;
    }
} /* _set_inputs_outputs() */

static vsi_status cpu_op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_reference params[_PARAM_NUM];

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    /* Set inputs and outputs */
    _set_inputs_outputs( self, params, inputs, outputs );

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, _PARAM_NUM );

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
    vsi_nn_type_e inputDataformat = inputs[0]->attr.dtype.vx_type;
    vsi_nn_type_e paraDataformat = inputs[1]->attr.dtype.vx_type;
    vsi_nn_type_e outputDataformat = outputs[0]->attr.dtype.vx_type;
    int8_t input_fixPointPos = inputs[0]->attr.dtype.fl;
    int8_t output_fixPointPos = outputs[0]->attr.dtype.fl;
    vx_uint32   width               = inputs[0]->attr.size[0];
    vx_uint32   height              = inputs[0]->attr.size[1];
    vx_uint32   depth               = inputs[0]->attr.size[2];
    vx_bool     enable_image_2d     = vx_false_e;
    vx_uint32   hwLitimLen          = 65536;

    enable_image_2d = (vx_bool)((width * height < hwLitimLen && depth < hwLitimLen)
                        || depth == 1);

    if (paraDataformat != VSI_NN_TYPE_FLOAT16)
    {
        VSILOGE("Not support parameter data format!(PRELU)\n");
        return VSI_FAILURE;
    }

    if ((inputDataformat == VSI_NN_TYPE_FLOAT16) && (outputDataformat == VSI_NN_TYPE_FLOAT16))
    {
        kernel_info->kernel_index = 1;
    }
    else if ((inputDataformat == VSI_NN_TYPE_INT8) && (outputDataformat == VSI_NN_TYPE_INT8))
    {
        if((input_fixPointPos >= output_fixPointPos) && (input_fixPointPos - output_fixPointPos < 32)
            && enable_image_2d)
        {
            kernel_info->resource_name[0] = "vsi_nn_kernel_header";
            kernel_info->resource_name[1] = "vsi_nn_kernel_prelu_i8_i16";

            kernel_info->resource_num     = 2;
            kernel_info->kernel_index     = 10;
        }
        else if((input_fixPointPos < output_fixPointPos) && (output_fixPointPos -  input_fixPointPos < 16)
            && enable_image_2d)
        {
            kernel_info->resource_name[0] = "vsi_nn_kernel_header";
            kernel_info->resource_name[1] = "vsi_nn_kernel_prelu_i8_i16";

            kernel_info->resource_num     = 2;
            kernel_info->kernel_index     = 12;
        }
        else
        {
            kernel_info->kernel_index = 2;
        }
    }
    else if ((inputDataformat == VSI_NN_TYPE_INT8) && (outputDataformat == VSI_NN_TYPE_FLOAT16))
    {
        kernel_info->kernel_index = 3;
    }
    else if ((inputDataformat == VSI_NN_TYPE_UINT8) && (outputDataformat == VSI_NN_TYPE_UINT8))
    {
        if(enable_image_2d)
        {
            kernel_info->resource_name[0] = "vsi_nn_kernel_header";
            kernel_info->resource_name[1] = "vsi_nn_kernel_prelu_u8";

            kernel_info->resource_num     = 2;
            kernel_info->kernel_index     = 14;
        }
        else
        {
            kernel_info->kernel_index = 4;
        }
    }
    else if ((inputDataformat == VSI_NN_TYPE_INT16) && (outputDataformat == VSI_NN_TYPE_INT16))
    {
        if((input_fixPointPos - output_fixPointPos >= 0) && (input_fixPointPos - output_fixPointPos < 32)
            && enable_image_2d)
        {
            kernel_info->resource_name[0] = "vsi_nn_kernel_header";
            kernel_info->resource_name[1] = "vsi_nn_kernel_prelu_i8_i16";

            kernel_info->resource_num     = 2;
            kernel_info->kernel_index = 11;
        }
        else if((input_fixPointPos < output_fixPointPos) && (output_fixPointPos -  input_fixPointPos < 16)
            && enable_image_2d)
        {
            kernel_info->resource_name[0] = "vsi_nn_kernel_header";
            kernel_info->resource_name[1] = "vsi_nn_kernel_prelu_i8_i16";

            kernel_info->resource_num     = 2;
            kernel_info->kernel_index     = 13;
        }
        else
        {
            kernel_info->kernel_index = 5;
        }
    }
    else if ((inputDataformat == VSI_NN_TYPE_FLOAT16) && (outputDataformat == VSI_NN_TYPE_UINT8))
    {
        kernel_info->kernel_index = 6;
    }
    else if ((inputDataformat == VSI_NN_TYPE_FLOAT16) && (outputDataformat == VSI_NN_TYPE_INT16))
    {
        kernel_info->kernel_index = 7;
    }
    else if ((inputDataformat == VSI_NN_TYPE_INT16) && (outputDataformat == VSI_NN_TYPE_FLOAT16))
    {
        kernel_info->kernel_index = 8;
    }
    else if ((inputDataformat == VSI_NN_TYPE_UINT8) && (outputDataformat == VSI_NN_TYPE_FLOAT16))
    {
        if(enable_image_2d)
        {
            kernel_info->resource_name[0] = "vsi_nn_kernel_header";
            kernel_info->resource_name[1] = "vsi_nn_kernel_prelu_u8";

            kernel_info->resource_num     = 2;
            kernel_info->kernel_index     = 15;
        }
        else
        {
            kernel_info->resource_name[0] = "vsi_nn_kernel_header";
            kernel_info->resource_name[1] = "vsi_nn_kernel_prelu_u8";

            kernel_info->resource_num     = 2;
            kernel_info->kernel_index = 9;
        }
    }
    else if ((inputDataformat == VSI_NN_TYPE_FLOAT16) && (outputDataformat == VSI_NN_TYPE_INT8))
    {
        kernel_info->resource_name[0] = "vsi_nn_kernel_header";
        kernel_info->resource_name[1] = "vsi_nn_kernel_prelu_u8";
        kernel_info->kernel_index = 16;
    }
    else
    {
        VSILOGE("Not support input or output data format!(PRELU)\n");
        return VSI_FAILURE;
    }
    return VSI_SUCCESS;
}

static vsi_status check_const_tensor_shape
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs
    )
{
    vsi_status status = VSI_SUCCESS;
#if 0
    self->nn_param.prelu.local.local_tensor[1] = NULL;

    if (inputs[1]->attr.dim_num == 1)
    {
        uint32_t num = inputs[1]->attr.size[0];
        vsi_nn_tensor_attr_t attr;
        attr.size[0] = num;
        attr.size[1] = 1;
        attr.size[2] = 1;
        attr.size[3] = 1;
        attr.dim_num = 4;

        self->nn_param.prelu.local.local_tensor[1] = vxReshapeTensor(inputs[1]->t,
            (int32_t *)(attr.size), attr.dim_num);
    }
#endif
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
    vx_border_t border;
    vx_bool     rsFlg = vx_false_e;
    vsi_enum    inDataType, outDataType;
    int8_t      input_fixPointPos   = 0;
    int8_t      output_fixPointPos  = 0;
    vx_uint32   width               = inputs[0]->attr.size[0];
    vx_uint32   height              = inputs[0]->attr.size[1];
    vx_uint32   depth               = inputs[0]->attr.size[2];
    vx_bool     enable_image_2d     = vx_false_e;
    vx_uint32   hwLitimLen          = 65536;

    enable_image_2d = (vx_bool)(width * height < hwLitimLen && depth < hwLitimLen);

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }
    inDataType  = inputs[0]->attr.dtype.vx_type;
    outDataType = outputs[0]->attr.dtype.vx_type;

    if((inDataType == VX_TYPE_INT8 && outDataType == VX_TYPE_INT8)
    || (inDataType == VX_TYPE_INT16 && outDataType == VX_TYPE_INT16))
    {
        input_fixPointPos  = inputs[0]->attr.dtype.fl;
        output_fixPointPos = outputs[0]->attr.dtype.fl;

        if((((input_fixPointPos  >= output_fixPointPos) && (input_fixPointPos - output_fixPointPos < 32))
         ||((input_fixPointPos  < output_fixPointPos) && (output_fixPointPos - input_fixPointPos < 16)))
         && enable_image_2d)
        {
            rsFlg = vx_true_e;
        }
    }
    else if (inDataType == VX_TYPE_UINT8 && enable_image_2d
        && (outDataType == VX_TYPE_UINT8 || outDataType == VX_TYPE_FLOAT16))
    {
        rsFlg = vx_true_e;
    }

    check_tensor_shape(self,inputs[0],params,0,rsFlg);
    check_tensor_shape(self,inputs[1],params,1,rsFlg);
    check_tensor_shape(self,outputs[0],params,2,rsFlg);
    /* Set inputs and outputs */
    //_set_inputs_outputs( self, params, inputs, outputs );

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, _PARAM_NUM );

    border.mode = VX_BORDER_REPLICATE;
    border.constant_value.U32 = 0;
    status |= vxSetNodeAttribute(self->n, VX_NODE_BORDER, &border, sizeof(border));

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
    vsi_status status;
#if (USE_OVX_API == TRUE)
    self->n = vxPReluLayer(
        self->graph->g,
        inputs[0]->t,
        inputs[1]->t,
        outputs[0]->t
        );
    if( NULL != self->n )
    {
        status = VSI_SUCCESS;
    }
#else
    vsi_nn_kernel_info_t kernel_info = {0};

    status = VSI_FAILURE;
    check_const_tensor_shape(self, inputs);
    kernel_info.resource_num  = 2;
    kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
    kernel_info.resource_name[0] = "vsi_nn_kernel_header";
    kernel_info.resource_name[1] = "vsi_nn_kernel_prelu";
    kernel_info.type = vsi_nn_GetVXKernelTypeForShader();
    kernel_info.kernel = vx_kernel_PRELU_list;
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
    return TRUE;
} /* op_check() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
#if (USE_OVX_API == FALSE)
    uint32_t i;
    for (i = 0; i < _VSI_NN_PRELU_LOCAL_TENSOR_NUM; i++)
    {
        if (self->nn_param.prelu.local.local_tensor[i] != NULL)
        {
            vxReleaseTensor(&(self->nn_param.prelu.local.local_tensor[i]));
            self->nn_param.prelu.local.local_tensor[i] = NULL;
        }
    }
#endif
    vsi_nn_op_common_deinit(self);

    return VSI_SUCCESS;
} /* op_deinit() */

#ifdef __cpluplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ PRELU,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ vsi_nn_op_common_setup,
    /* optimize   */ NULL,
    /* input_num  */ 2,
    /* output_num */ 1
    );
#ifdef __cpluplus
}
#endif

