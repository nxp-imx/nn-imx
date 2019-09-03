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
#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)
#define IMG_MAX_WIDTH 65536

extern vx_kernel_description_t * vx_kernel_EXP_list[];


/* Greatest Common Divisor*/
static vsi_bool vxoGetDataDivisors(uint32_t input_value, uint32_t *divisors, uint32_t gcd)
{
    uint32_t i                 = 0;

    for (i = vsi_nn_min(input_value, IMG_MAX_WIDTH - 1); i > 0; i --)
    {
        if ((i % gcd == 0) && (input_value % i == 0))
        {
            *divisors = i;

            return TRUE;
        }
    }

    return FALSE;
}

static vsi_bool vxoElementOptimization_GetTensorShape(vsi_nn_tensor_t *input,uint32_t sizes[VSI_NN_MAX_DIM_NUM],
                                                      uint32_t * num_of_dims)
{
    uint32_t element_count     = 0;
    uint32_t i                 = 0;

    element_count = vsi_nn_GetElementNum(input);

    for (i = 0; i < VSI_NN_MAX_DIM_NUM; i++)
    {
        sizes[i] = 1;
    }

    if (element_count < IMG_MAX_WIDTH)
    {
        sizes[0] = element_count;

        *num_of_dims = 2;
    }
    else
    {
        uint32_t divisors = 1;
        for (i = 0; i < 2; i++)
        {
            divisors = 1;
            vxoGetDataDivisors(element_count, &divisors, 1);

            sizes[i] = divisors;
            element_count = element_count / divisors;
        }

        sizes[2] = element_count;
        *num_of_dims = 3;
    }

    return TRUE;
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
    _set_inputs_outputs( params, inputs, outputs );

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
    vsi_nn_type_e input_format = inputs[0]->attr.dtype.vx_type;
    vsi_nn_type_e output_format = outputs[0]->attr.dtype.vx_type;
    vsi_bool useImage2D  =  FALSE;
    uint32_t sizes[VSI_NN_MAX_DIM_NUM] = {1};
    uint32_t num_of_dims = 0;

    vxoElementOptimization_GetTensorShape(inputs[0], sizes, &num_of_dims);
    useImage2D  =  (vsi_bool)(sizes[2] == 1);

    if (useImage2D)
    {
        if (input_format == VSI_NN_TYPE_FLOAT16)
        {
            if (output_format == VSI_NN_TYPE_FLOAT16)
                kernel_info->kernel_index = TENSOR_EXP_F16TOF16_2D_KERNEL;
            else if (output_format == VSI_NN_TYPE_INT16)
                kernel_info->kernel_index = TENSOR_EXP_F16TOI16_2D_KERNEL;
            else if (output_format == VSI_NN_TYPE_INT8)
                kernel_info->kernel_index = TENSOR_EXP_F16TOI8_2D_KERNEL;
            else if (output_format == VSI_NN_TYPE_UINT8)
                kernel_info->kernel_index = TENSOR_EXP_F16TOU8_2D_KERNEL;
            else
            {
                VSILOGE("Not support input or output data format!(EXP)\n");
                return VSI_FAILURE;
            }
        }
        else if (input_format == VSI_NN_TYPE_INT16)
        {
            if (output_format == VSI_NN_TYPE_INT16)
                kernel_info->kernel_index = TENSOR_EXP_I16TOI16_2D_KERNEL;
            else if (output_format == VSI_NN_TYPE_FLOAT16)
                kernel_info->kernel_index = TENSOR_EXP_I16TOF16_2D_KERNEL;
            else
            {
                VSILOGE("Not support input or output data format!(EXP)\n");
                return VSI_FAILURE;
            }
        }
        else if (input_format == VSI_NN_TYPE_INT8)
        {
            if (output_format == VSI_NN_TYPE_INT8)
                kernel_info->kernel_index = TENSOR_EXP_I8TOI8_2D_KERNEL;
            else if (output_format == VSI_NN_TYPE_FLOAT16)
                kernel_info->kernel_index = TENSOR_EXP_I8TOF16_2D_KERNEL;
            else
            {
                VSILOGE("Not support input or output data format!(EXP)\n");
                return VSI_FAILURE;
            }
        }
        else if (input_format == VSI_NN_TYPE_UINT8)
        {
            if (output_format == VSI_NN_TYPE_UINT8)
                kernel_info->kernel_index = TENSOR_EXP_U8TOU8_2D_KERNEL;
            else if (output_format == VSI_NN_TYPE_FLOAT16)
                kernel_info->kernel_index = TENSOR_EXP_U8TOF16_2D_KERNEL;
            else
            {
                VSILOGE("Not support input or output data format!(EXP)\n");
                return VSI_FAILURE;
            }
        }
        else
        {
            VSILOGE("Not support input or output data format!(EXP)\n");
            return VSI_FAILURE;
        }
    }
    else
    {
        if (input_format == VSI_NN_TYPE_FLOAT16)
        {
            if (output_format == VSI_NN_TYPE_FLOAT16)
                kernel_info->kernel_index = TENSOR_EXP_F16TOF16_KERNEL;
            else if (output_format == VSI_NN_TYPE_INT16)
                kernel_info->kernel_index = TENSOR_EXP_F16TOI16_KERNEL;
            else if (output_format == VSI_NN_TYPE_INT8)
                kernel_info->kernel_index = TENSOR_EXP_F16TOI8_KERNEL;
            else if (output_format == VSI_NN_TYPE_UINT8)
                kernel_info->kernel_index = TENSOR_EXP_F16TOU8_KERNEL;
            else
            {
                VSILOGE("Not support input or output data format!(EXP)\n");
                return VSI_FAILURE;
            }
        }
        else if (input_format == VSI_NN_TYPE_INT16)
        {
            if (output_format == VSI_NN_TYPE_INT16)
                kernel_info->kernel_index = TENSOR_EXP_I16TOI16_KERNEL;
            else if (output_format == VSI_NN_TYPE_FLOAT16)
                kernel_info->kernel_index = TENSOR_EXP_I16TOF16_KERNEL;
            else
            {
                VSILOGE("Not support input or output data format!(EXP)\n");
                return VSI_FAILURE;
            }
        }
        else if (input_format == VSI_NN_TYPE_INT8)
        {
            if (output_format == VSI_NN_TYPE_INT8)
                kernel_info->kernel_index = TENSOR_EXP_I8TOI8_KERNEL;
            else if (output_format == VSI_NN_TYPE_FLOAT16)
                kernel_info->kernel_index = TENSOR_EXP_I8TOF16_KERNEL;
            else
            {
                VSILOGE("Not support input or output data format!(EXP)\n");
                return VSI_FAILURE;
            }
        }
        else if (input_format == VSI_NN_TYPE_UINT8)
        {
            if (output_format == VSI_NN_TYPE_UINT8)
                kernel_info->kernel_index = TENSOR_EXP_U8TOU8_KERNEL;
            else if (output_format == VSI_NN_TYPE_FLOAT16)
                kernel_info->kernel_index = TENSOR_EXP_U8TOF16_KERNEL;
            else
            {
                VSILOGE("Not support input or output data format!(EXP)\n");
                return VSI_FAILURE;
            }
        }
        else
        {
            VSILOGE("Not support input or output data format!(EXP)\n");
            return VSI_FAILURE;
        }
    }

    return VSI_SUCCESS;
}

static void reshape_tensor_shape
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vx_reference * params,
    uint32_t index
    )
{
    uint32_t sizes[VSI_NN_MAX_DIM_NUM] = {1};
    uint32_t num_of_dims = vsi_nn_max(input->attr.dim_num, 2);

    vxoElementOptimization_GetTensorShape(input, sizes, &num_of_dims);

    self->nn_param.exp.local.local_tensor[index] =
         vxReshapeTensor(input->t, (int32_t *)sizes, num_of_dims);

    params[index] = (vx_reference)self->nn_param.exp.local.local_tensor[index];
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

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    reshape_tensor_shape(self, inputs[0], params, 0);
    reshape_tensor_shape(self, outputs[0], params, 1);

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
    kernel_info.resource_name[0] = "vsi_nn_kernel_exp";
    kernel_info.type = vsi_nn_GetVXKernelTypeForShader();
    kernel_info.kernel = vx_kernel_EXP_list;
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
    for (i = 0; i < _VSI_NN_EXP_LOCAL_TENSOR_NUM; i++)
    {
        if (self->nn_param.exp.local.local_tensor[i] != NULL)
        {
            vxReleaseTensor(&(self->nn_param.exp.local.local_tensor[i]));
            self->nn_param.exp.local.local_tensor[i] = NULL;
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
    /* op_name    */ EXP,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ vsi_nn_op_common_setup,
    /* optimize   */ NULL,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );
#ifdef __cplusplus
}
#endif
