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

#define _ARG_NUM            (0)
#define _INPUT_NUM          (2)
#define _OUTPUT_NUM         (1)
#define _VX_KERNEL_VAR      (vx_client_kernel_MINIMUM)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)
#define _VSI_PARAM          (vsi_nn_client_minimum_param)

extern vx_kernel_description_t * vx_kernel_MINIMUM_list[];

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

    if( input->attr.dim_num == 1)
    {
        memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
        attr.size[1] = 1;
        attr.dim_num = 2;
        self->nn_param.pow.local.local_tensor[index] =
            vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
        params[index] =  (vx_reference)self->nn_param.pow.local.local_tensor[index];
    }
    else if( input->attr.dim_num == 2 && rsFlg)
    {
        memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
        attr.size[0] *= attr.size[1];
        attr.size[1] = 1;
        self->nn_param.minimum.local.local_tensor[index] =
            vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
        params[index] =  (vx_reference)self->nn_param.minimum.local.local_tensor[index];
    }
    else if( input->attr.dim_num == 3 && rsFlg)
    {
        memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
        attr.size[0] *= attr.size[1];
        attr.size[1] = attr.size[2];
        attr.size[2] = 1;
        self->nn_param.minimum.local.local_tensor[index] =
            vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
        params[index] =  (vx_reference)self->nn_param.minimum.local.local_tensor[index];
    }
    else if( input->attr.dim_num == 4)
    {
        memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
        if (rsFlg)
        {
            attr.size[0] *= attr.size[1];
            attr.size[1] = attr.size[2] * attr.size[3];
            attr.size[2] = 1;
            attr.size[3] = 1;
        }
        else
        {
            attr.size[2] = attr.size[2] * attr.size[3];
            attr.size[3] = 1;
        }

        self->nn_param.minimum.local.local_tensor[index] =
            vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
        params[index] =  (vx_reference)self->nn_param.minimum.local.local_tensor[index];
    }
    else if (rsFlg == FALSE && input->attr.dim_num <= 4)
    {
        params[index] = (vx_reference)input->t;
    }
    else
    {
        VSILOGE("No more local tensor!(minimum) at [%s : %d]\n", __FILE__, __LINE__);
    }
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
    _set_inputs_outputs(params, inputs, outputs );

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
    vsi_nn_type_e src0Format    = inputs[0]->attr.dtype.vx_type;
    vsi_nn_type_e src1Format    = inputs[1]->attr.dtype.vx_type;
    vsi_nn_type_e dstFormat     = outputs[0]->attr.dtype.vx_type;
    uint32_t   dims             = outputs[0]->attr.dim_num;
    uint32_t   width            = outputs[0]->attr.size[0];
    uint32_t   height           = dims > 1 ? outputs[0]->attr.size[1] : 1;
    uint32_t   depth            = dims > 2 ? outputs[0]->attr.size[2] : 1;
    uint32_t   batch            = dims > 3 ? outputs[0]->attr.size[3] : 1;
    vsi_bool   enable_image_2d  = FALSE;
    vsi_bool   enable_brdcst    = FALSE;
    uint32_t   i                = 0;
    uint32_t   hwLitimLen       = 65536;

    for (i = 0; i < outputs[0]->attr.dim_num; i++)
    {
        vx_uint32 size0 = inputs[0]->attr.dim_num > i ? inputs[0]->attr.size[i] : 1;
        vx_uint32 size1 = inputs[1]->attr.dim_num > i ? inputs[1]->attr.size[i] : 1;

        if (size0 != size1)
        {
            enable_brdcst = vx_true_e;
            break;
        }
    }

    if (enable_brdcst == vx_false_e)
    {
        enable_image_2d = (vx_bool)(((width * height < hwLitimLen) && (depth * batch < hwLitimLen))
                                  || depth * batch == 1);
    }


    if (src0Format == src1Format && src0Format == VSI_NN_TYPE_FLOAT16
        && dstFormat == src0Format)
    {
        if (enable_image_2d)
            kernel_info->kernel_index = TENSOR_MIN_F16F16TOF16_2D_KERNEL;
        else
            kernel_info->kernel_index = TENSOR_MIN_F16F16TOF16_KERNEL;
    }
    else if (src0Format == src1Format && src0Format == VSI_NN_TYPE_INT8
          && dstFormat == src0Format)
    {
        if(enable_image_2d)
            kernel_info->kernel_index   = TENSOR_MIN_I8I8TOI8_2D_KERNEL;
        else
            kernel_info->kernel_index   = TENSOR_MIN_I8I8TOI8_KERNEL;
    }
    else if (src0Format == src1Format && src0Format == VSI_NN_TYPE_UINT8
          && dstFormat == src0Format)
    {
        if(enable_image_2d)
            kernel_info->kernel_index   = TENSOR_MIN_U8TOU8_2D_KERNEL;
        else
            kernel_info->kernel_index   = TENSOR_MIN_U8TOU8_KERNEL;
    }
    else if (src0Format == src1Format && src0Format == VSI_NN_TYPE_INT16
          && dstFormat == src0Format)
    {
        if(enable_image_2d)
            kernel_info->kernel_index   = TENSOR_MIN_I16I16TOI16_2D_KERNEL;
        else
            kernel_info->kernel_index   = TENSOR_MIN_I16I16TOI16_KERNEL;
    }
    else if (src0Format == VSI_NN_TYPE_INT8 && src1Format == VSI_NN_TYPE_FLOAT16
          && dstFormat == VSI_NN_TYPE_INT8)
    {
        kernel_info->resource_name[1] = "vsi_nn_kernel_minimum_fp16";
        if(enable_image_2d)
            kernel_info->kernel_index   = TENSOR_MIN_I8F16TOI8_2D_KERNEL;
        else
            kernel_info->kernel_index   = TENSOR_MIN_I8F16TOI8_KERNEL;
    }
    else if (src0Format == VSI_NN_TYPE_INT8 && src1Format == VSI_NN_TYPE_FLOAT16
          && dstFormat == VSI_NN_TYPE_FLOAT16)
    {
        kernel_info->resource_name[1] = "vsi_nn_kernel_minimum_fp16";
        if(enable_image_2d)
            kernel_info->kernel_index   = TENSOR_MIN_I8F16TOF16_2D_KERNEL;
        else
            kernel_info->kernel_index   = TENSOR_MIN_I8F16TOF16_KERNEL;
    }
    else if (src0Format == VSI_NN_TYPE_UINT8 && src1Format == VSI_NN_TYPE_FLOAT16
          && dstFormat == VSI_NN_TYPE_FLOAT16)
    {
        kernel_info->resource_name[1] = "vsi_nn_kernel_minimum_fp16";
        if(enable_image_2d)
            kernel_info->kernel_index   = TENSOR_MIN_U8F16TOF16_2D_KERNEL;
        else
            kernel_info->kernel_index   = TENSOR_MIN_U8F16TOF16_KERNEL;
    }
    else if (src0Format == VSI_NN_TYPE_UINT8 && src1Format == VSI_NN_TYPE_FLOAT16
          && dstFormat == VSI_NN_TYPE_UINT8)
    {
        kernel_info->resource_name[1] = "vsi_nn_kernel_minimum_fp16";
        if(enable_image_2d)
            kernel_info->kernel_index   = TENSOR_MIN_U8F16TOU8_2D_KERNEL;
        else
            kernel_info->kernel_index   = TENSOR_MIN_U8F16TOU8_KERNEL;
    }
    else if (src0Format == VSI_NN_TYPE_INT16 && src1Format == VSI_NN_TYPE_FLOAT16
          && dstFormat == VSI_NN_TYPE_INT16)
    {
        kernel_info->resource_name[1] = "vsi_nn_kernel_minimum_i16";
        if(enable_image_2d)
            kernel_info->kernel_index   = TENSOR_MIN_I16F16TOI16_2D_KERNEL;
        else
            kernel_info->kernel_index   = TENSOR_MIN_I16F16TOI16_KERNEL;
    }
    else if (src0Format == VSI_NN_TYPE_INT16 && src1Format == VSI_NN_TYPE_FLOAT16
          && dstFormat == VSI_NN_TYPE_FLOAT16)
    {
        kernel_info->resource_name[1] = "vsi_nn_kernel_minimum_i16";
        if(enable_image_2d)
            kernel_info->kernel_index   = TENSOR_MIN_I16F16TOF16_2D_KERNEL;
        else
            kernel_info->kernel_index   = TENSOR_MIN_I16F16TOF16_KERNEL;
    }
    else
    {
        VSILOGE("Not support input or output data format!(minimum)\n");
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
    uint32_t    dims             = outputs[0]->attr.dim_num;
    uint32_t    width            = outputs[0]->attr.size[0];
    uint32_t    height           = dims > 1 ? outputs[0]->attr.size[1] : 1;
    uint32_t    depth            = dims > 2 ? outputs[0]->attr.size[2] : 1;
    uint32_t    batch            = dims > 3 ? outputs[0]->attr.size[3] : 1;
    vsi_bool    enable_image_2d  = FALSE;
    vsi_bool    enable_brdcst    = FALSE;
    uint32_t    hwLitimLen       = 65536;
    uint32_t    i                = 0;

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    for (i = 0; i < outputs[0]->attr.dim_num; i++)
    {
        vx_uint32 size0 = inputs[0]->attr.dim_num > i ? inputs[0]->attr.size[i] : 1;
        vx_uint32 size1 = inputs[1]->attr.dim_num > i ? inputs[1]->attr.size[i] : 1;

        if (size0 != size1)
        {
            enable_brdcst = vx_true_e;
            break;
        }
    }

    if (enable_brdcst == vx_false_e)
    {
        enable_image_2d = (vx_bool)((width * height < hwLitimLen)
                                 && (depth * batch < hwLitimLen));
    }

    //_set_inputs_outputs( params, inputs, outputs );
    check_tensor_shape(self, inputs[0], params, 0, enable_image_2d);
    check_tensor_shape(self, inputs[1], params, 1, enable_image_2d);
    check_tensor_shape(self, outputs[0], params, 2, enable_image_2d);
    /*TODO: Add code if need to change your parameter*/

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

#define SWAP_INPUT_TENSOR(a, b) \
do     \
{      \
    vsi_nn_tensor_t * tmp; \
    tmp = (a);     \
    (a) = (b);     \
    (b) = tmp;     \
} while (0)

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status;
    vsi_nn_kernel_info_t kernel_info = {0};
    vsi_nn_type_e input0_Format = inputs[0]->attr.dtype.vx_type;
    vsi_nn_type_e input1_Format = inputs[1]->attr.dtype.vx_type;

    status = VSI_FAILURE;
    kernel_info.resource_num = 2;
    kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
    kernel_info.resource_name[0] = "vsi_nn_kernel_header";
    kernel_info.resource_name[1] = "vsi_nn_kernel_minimum";
    kernel_info.type = vsi_nn_GetVXKernelTypeForShader();
    kernel_info.kernel = vx_kernel_MINIMUM_list;
    kernel_info.init_index = 1;

    if (input0_Format != input1_Format && input0_Format == VSI_NN_TYPE_FLOAT16)
        SWAP_INPUT_TENSOR(inputs[0], inputs[1]);

    if (vsi_nn_is_do_vx_op_pre_init(kernel_info.type))
    {
        vx_op_pre_compute(self, inputs, outputs, &kernel_info);
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

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    uint32_t i;

    for (i = 0; i < _VSI_NN_MINIMUM_LOCAL_TENSOR_NUM; i++)
    {
        if (self->nn_param.minimum.local.local_tensor[i] != NULL)
        {
            vxReleaseTensor(&(self->nn_param.minimum.local.local_tensor[i]));
            self->nn_param.minimum.local.local_tensor[i] = NULL;
        }
    }

    vsi_nn_op_common_deinit(self);

    return VSI_SUCCESS;
} /* op_deinit() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = FALSE;

    ret = vsi_nn_OpSetup( VSI_NN_OP_MULTIPLY, self, inputs, outputs );

    return ret;
} /* op_setup() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ MINIMUM,
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
