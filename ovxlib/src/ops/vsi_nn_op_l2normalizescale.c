/****************************************************************************
*
*    Copyright (c) 2020 Vivante Corporation
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
#include "kernel/vsi_nn_kernel.h"
#include "utils/vsi_nn_util.h"

#define VSI_NN_L2NORMALIZESCALE_DEFAULT_AXIS 2

/* Type enum */
typedef enum _l2normalizescale_nn_image_dims_e
{
    IMAGE_2D  = TRUE,
    IMAGE     = FALSE,
}l2normalizescale_nn_activation_type_e;

#define VSI_NN_GEN_L2NORMALIZESCALE_KEY(_axis, _input_type, _output_type, _image_2d) \
    ((_axis << 20) | (_input_type << 12) | (_output_type << 4) | (_image_2d))

#define VSI_NN_GEN_L2NORMALIZESCALE_KERNEL_SOURCE_NAME(_suffix) \
    "vsi_nn_kernel_l2normalizescaleAxis"#_suffix

#define VSI_NN_GEN_L2NORMALIZESCALE_STRUCT_ITEMS(_axis, _input_type, _output_type, _image_2d) \
    VSI_NN_GEN_L2NORMALIZESCALE_KEY(_axis, _input_type, _output_type, _image_2d), \
    VSI_NN_L2NORMSACLE_SQRTSUM_KERNEL_IDX(_axis, _input_type, _image_2d) \
    VSI_NN_L2NORMSACLE_MULSCALE_KERNEL_IDX(_axis, _input_type, _output_type, _image_2d) \
    VSI_NN_GEN_L2NORMALIZESCALE_KERNEL_SOURCE_NAME(_axis)

static struct {
        uint32_t key;
        uint32_t kernel_sum_index;
        uint32_t kernel_mul_index;
        char *resource_name;
    } l2normalizescale_map[] =
    {
        {VSI_NN_GEN_L2NORMALIZESCALE_STRUCT_ITEMS(1, F16, F16, IMAGE_2D)},
        {VSI_NN_GEN_L2NORMALIZESCALE_STRUCT_ITEMS(1, I8, I8, IMAGE_2D)},
        {VSI_NN_GEN_L2NORMALIZESCALE_STRUCT_ITEMS(1, I8, F16, IMAGE_2D)},
        {VSI_NN_GEN_L2NORMALIZESCALE_STRUCT_ITEMS(1, U8, U8, IMAGE_2D)},
        {VSI_NN_GEN_L2NORMALIZESCALE_STRUCT_ITEMS(1, U8, F16, IMAGE_2D)},
        {VSI_NN_GEN_L2NORMALIZESCALE_STRUCT_ITEMS(1, I16, I16, IMAGE_2D)},
        {VSI_NN_GEN_L2NORMALIZESCALE_STRUCT_ITEMS(1, I16, F16, IMAGE_2D)},
        {VSI_NN_GEN_L2NORMALIZESCALE_STRUCT_ITEMS(0, F16, F16, IMAGE_2D)},
        {VSI_NN_GEN_L2NORMALIZESCALE_STRUCT_ITEMS(0, I8, I8, IMAGE_2D)},
        {VSI_NN_GEN_L2NORMALIZESCALE_STRUCT_ITEMS(0, I8, F16, IMAGE_2D)},
        {VSI_NN_GEN_L2NORMALIZESCALE_STRUCT_ITEMS(0, U8, U8, IMAGE_2D)},
        {VSI_NN_GEN_L2NORMALIZESCALE_STRUCT_ITEMS(0, U8, F16, IMAGE_2D)},
        {VSI_NN_GEN_L2NORMALIZESCALE_STRUCT_ITEMS(0, I16, I16, IMAGE_2D)},
        {VSI_NN_GEN_L2NORMALIZESCALE_STRUCT_ITEMS(0, I16, F16, IMAGE_2D)},
    };

static vsi_bool _check_tensor_shape
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = FALSE;
    uint32_t dims = inputs[0]->attr.dim_num;
    int32_t  axis = self->nn_param.l2normalizescale.axis;
    uint32_t input_size[VSI_NN_MAX_DIM_NUM] = {1};
    uint32_t i;

    for (i = 0; i < inputs[0]->attr.dim_num; i++)
    {
        input_size[i] = inputs[0]->attr.size[i];
    }

    for(; i < VSI_NN_MAX_DIM_NUM; i++)
    {
        input_size[i] = 1;
    }
#define VSI_NN_TENSOR_WIDTH_MAX (65536)

    if (axis == 0)
    {
        if (dims < 3 || (input_size[1] * input_size[2] < VSI_NN_TENSOR_WIDTH_MAX))
            ret = TRUE;
    }
    else if (axis == 2)
    {
        if (dims < 3 || (input_size[0] * input_size[1] < VSI_NN_TENSOR_WIDTH_MAX))
            ret = TRUE;
    }
    else if (axis == 1)
    {
        if (dims < 3 || (input_size[0] == 1) || (input_size[2] == 1))
            ret = TRUE;
    }

#undef VSI_NN_TENSOR_WIDTH_MAX

    return ret;
}

#define _ARG_NUM            (1)
#define _INPUT_NUM          (2)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)

extern vx_kernel_description_t * vx_kernel_L2NORMALIZESCALE_list[];

static void _set_inputs_outputs
    (
    vx_reference * params,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    uint16_t input_num,
    uint16_t output_num
    )
{
    uint32_t i;
    uint32_t cnt;

    /* Set inputs */
    cnt = 0;
    for( i = 0; i < input_num; i ++, cnt ++ )
    {
        params[cnt] = (vx_reference)inputs[i]->t;
    }

    /* Set outputs */
    for( i = 0; i < output_num; i ++, cnt ++ )
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
    vsi_nn_l2normalizescale_param * p;
    if( 0 == num )
    {
        return VSI_SUCCESS;
    }
    memset( params, 0, sizeof( vx_reference * ) * num );
    p = &(node->nn_param.l2normalizescale);
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
    vx_reference * args = NULL;

    args = &params[_IO_NUM];

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    /* Set inputs and outputs */
    _set_inputs_outputs( params, inputs, outputs, _INPUT_NUM, _OUTPUT_NUM);

    /* Init parameters. */
    _create_params( self, args, _ARG_NUM );

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, _PARAM_NUM );

    _release_params( args, _ARG_NUM );

    return status;
}

static int32_t reshape_tensor_set_input_output
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vx_bool is_reshape
    )
{
    uint32_t sizes[VSI_NN_MAX_DIM_NUM] = {1};
    uint32_t dims = vsi_nn_max(inputs[0]->attr.dim_num, 2);
    int32_t axis = 0;
    vsi_nn_l2normalizescale_param * p = NULL;
    vsi_bool is_2d_image = FALSE;
    uint32_t input_size[VSI_NN_MAX_DIM_NUM] = {1};
    uint32_t i;

    for (i = 0; i < inputs[0]->attr.dim_num; i++)
    {
        input_size[i] = inputs[0]->attr.size[i];
    }

    for(; i < VSI_NN_MAX_DIM_NUM; i++)
    {
        input_size[i] = 1;
    }

    p = &(self->nn_param.l2normalizescale);
    axis = p->axis;

    is_2d_image = _check_tensor_shape(self, inputs, outputs);

    if (axis == 0)
    {
        sizes[0] = input_size[0];

        if (is_2d_image)
        {
            sizes[1] = input_size[1] * input_size[2];
            sizes[2] = 1;
            sizes[3] = dims > 3 ? input_size[3] : 1;
        }
        else
        {
            sizes[1] = input_size[1];
            sizes[2] = dims > 2 ? input_size[2] : 1;
            sizes[3] = dims > 3 ? input_size[3] : 1;
        }
    }
    else if (axis == 1)
    {
        if (1 == input_size[0])
        {
            sizes[0] = input_size[1];
            sizes[1] = dims > 2 ? input_size[2] : 1;
            sizes[2] = 1;
            sizes[3] = dims > 3 ? input_size[3] : 1;
            axis = 0;
        }
        else
        {
            sizes[0] = input_size[0];
            sizes[1] = input_size[1];
            sizes[2] = dims > 2 ? input_size[2] : 1;
            sizes[3] = dims > 3 ? input_size[3] : 1;
        }
    }
    else if (axis == 2)
    {
        if(1 == input_size[0] && 1 == input_size[1])
        {
            sizes[0] = input_size[2];
            sizes[1] = 1;
            sizes[2] = 1;
            sizes[3] = dims > 3 ? input_size[3] : 1;
            axis = 0;
        }
        else if (is_2d_image)
        {
            sizes[0] = input_size[0] * input_size[1];
            sizes[1] = input_size[2];
            sizes[2] = 1;
            sizes[3] = dims > 3 ? input_size[3] : 1;
            axis = 1;
        }
        else
        {
            sizes[0] = input_size[0];
            sizes[1] = input_size[1];
            sizes[2] = dims > 2 ? input_size[2] : 1;
            sizes[3] = dims > 3 ? input_size[3] : 1;
        }
    }

    if (is_reshape)
    {
        p->axis = axis;
        p->local.local_tensor[0] =
            vxReshapeTensor(inputs[0]->t,  (int32_t *)sizes, dims);
        p->local.local_tensor[2] =
            vxReshapeTensor(outputs[0]->t, (int32_t *)sizes, dims);
    }

    return axis;
}


static void _get_l2normalizescale_hashtable_idx
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_type_e inputFormat = inputs[0]->attr.dtype.vx_type;
    vsi_nn_type_e outputFormat  = outputs[0]->attr.dtype.vx_type;
    vsi_nn_kernel_dtype_e _input_type;
    vsi_nn_kernel_dtype_e _output_type;
    int32_t axis = 0;
    uint32_t key;
    vsi_bool is_2d_image = FALSE;
    uint32_t i = 0;
    vsi_nn_l2normalizescale_param * p = NULL;

    p = &(self->nn_param.l2normalizescale);
    axis = reshape_tensor_set_input_output(self, inputs, outputs, vx_false_e);
    _input_type  = vsi_nn_kernel_map_dtype(inputFormat);
    _output_type = vsi_nn_kernel_map_dtype(outputFormat);
    is_2d_image = _check_tensor_shape(self, inputs, outputs);
    key = VSI_NN_GEN_L2NORMALIZESCALE_KEY(axis, _input_type, _output_type, is_2d_image);

    for (i = 0; i < sizeof(l2normalizescale_map) / sizeof(l2normalizescale_map[0]); i++)
    {
        if (key == l2normalizescale_map[i].key)
        {
            p->local.hash_idx = i;
            p->local.execute_on_sw = FALSE;
            return;
        }
    }

    p->local.execute_on_sw = TRUE;
    VSILOGE("Shader unsupport data format or axis! execute on the SW [l2normalizescale]\n");
}

static vsi_status vx_op_pre_init_sum_r_sqrt
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_kernel_info_t * kernel_info
    )
{

    vsi_nn_l2normalizescale_param * p = NULL;

    p = &(self->nn_param.l2normalizescale);

    kernel_info->kernel_index = (uint8_t)l2normalizescale_map[p->local.hash_idx].kernel_sum_index;
    kernel_info->resource_num = 1;
    kernel_info->resource_name[0] = l2normalizescale_map[p->local.hash_idx].resource_name;

    return VSI_SUCCESS;
}

static vsi_status vx_op_pre_init_mul_scale
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_kernel_info_t * kernel_info
    )
{

    vsi_nn_l2normalizescale_param * p = NULL;

    p = &(self->nn_param.l2normalizescale);

    kernel_info->kernel_index = (uint8_t)l2normalizescale_map[p->local.hash_idx].kernel_mul_index;
    kernel_info->resource_num = 1;
    kernel_info->resource_name[0] = l2normalizescale_map[p->local.hash_idx].resource_name;

    return VSI_SUCCESS;
}


#define _ARG_NUM_SUM_R_SQRT            (1)
#define _INPUT_NUM_SUM_R_SQRT          (1)
#define _OUTPUT_NUM_SUM_R_SQRT         (1)
#define _IO_NUM_SUM_R_SQRT             (_INPUT_NUM_SUM_R_SQRT + _OUTPUT_NUM_SUM_R_SQRT)
#define _PARAM_NUM_SUM_R_SQRT          (_ARG_NUM_SUM_R_SQRT + _IO_NUM_SUM_R_SQRT)

static vsi_status vx_op_compute_sum_r_sqrt
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_reference params[_PARAM_NUM_SUM_R_SQRT];
    vx_border_t border;
    vx_reference * args = NULL;
    vsi_nn_tensor_attr_t attr;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    args = &params[_IO_NUM_SUM_R_SQRT];
    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    params[0] = (vx_reference)self->nn_param.l2normalizescale.local.local_tensor[0];
    params[1] = (vx_reference)outputs[0]->t;

    /* Init parameters. */
    _create_params( self, args, _ARG_NUM_SUM_R_SQRT );

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, _PARAM_NUM_SUM_R_SQRT );

    _release_params( args, _ARG_NUM_SUM_R_SQRT );

    border.mode = VX_BORDER_CONSTANT;
    if (inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_INT8)
        border.constant_value.U8 = 0;
    else if (inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_UINT8)
        border.constant_value.U8 = (uint8_t)inputs[0]->attr.dtype.zero_point;
    else
        border.constant_value.S16 = 0;
    status |= vxSetNodeAttribute(self->n, VX_NODE_BORDER, &border, sizeof(border));

    return status;
}

#define _ARG_NUM_MUL_SCALE            (1)
#define _INPUT_NUM_MUL_SCALE          (3)
#define _OUTPUT_NUM_MUL_SCALE         (1)
#define _IO_NUM_MUL_SCALE             (_INPUT_NUM_MUL_SCALE + _OUTPUT_NUM_MUL_SCALE)
#define _PARAM_NUM_MUL_SCALE          (_ARG_NUM_MUL_SCALE + _IO_NUM_MUL_SCALE)

static vsi_status vx_op_compute_mul_scale
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_reference params[_PARAM_NUM_MUL_SCALE];
    vx_border_t border;
    vx_reference * args = NULL;
    vsi_nn_tensor_attr_t attr;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    args = &params[_IO_NUM_MUL_SCALE];
    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    params[0] = (vx_reference)self->nn_param.l2normalizescale.local.local_tensor[0];
    params[1] = (vx_reference)inputs[1]->t;
    if (self->nn_param.l2normalizescale.local.local_tensor[1] == NULL)
    {
        params[2] = (vx_reference)inputs[2]->t;
    }
    else
    {
        params[2] = (vx_reference)self->nn_param.l2normalizescale.local.local_tensor[1];
    }
    params[3] = (vx_reference)self->nn_param.l2normalizescale.local.local_tensor[2];

    /* Init parameters. */
    _create_params( self, args, _ARG_NUM_MUL_SCALE );

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, _PARAM_NUM_MUL_SCALE );

    _release_params( args, _ARG_NUM_MUL_SCALE );

    border.mode = VX_BORDER_REPLICATE;
    border.constant_value.U32 = 0;
    status |= vxSetNodeAttribute(self->n, VX_NODE_BORDER, &border, sizeof(border));

    return status;
}

static vsi_status check_const_tensor_shape
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs
    )
{
    vsi_status status = VSI_SUCCESS;
    uint32_t   size   = 1;
    uint32_t   i      = 0;
    self->nn_param.l2normalizescale.local.local_tensor[1] = NULL;

    size = inputs[1]->attr.size[0];
    for (i = 1; i < inputs[1]->attr.dim_num; i ++)
    {
        size *= inputs[1]->attr.size[i];
    }

    if (1)
    {
        vsi_nn_tensor_attr_t attr;
        attr.size[0] = size;
        attr.size[1] = 1;
        attr.size[2] = 1;
        attr.size[3] = 1;
        attr.dim_num = 4;

        self->nn_param.l2normalizescale.local.local_tensor[1] = vxReshapeTensor(inputs[1]->t,
            (int32_t *)(attr.size), attr.dim_num);
    }

    return status;
}

static vsi_nn_op_compute_t op_compute_list[] =
{
    cpu_op_compute,
    vx_op_compute_sum_r_sqrt,
    vx_op_compute_mul_scale
};

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_info_t kernel_info;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_t * outputs_sum_r_sqrt[_OUTPUT_NUM_SUM_R_SQRT] = {0};
    vsi_nn_tensor_t * inputs_mul_scale[_INPUT_NUM_MUL_SCALE] = {0};
    vsi_nn_l2normalizescale_param * p = NULL;
    int32_t   axis = 0;
    int32_t   input_size[4] = {1, 1, 1, 1};
    uint32_t  input_dims = 0;

    memset(&kernel_info, 0x0, sizeof(vsi_nn_kernel_info_t));
    p = &(self->nn_param.l2normalizescale);
    _get_l2normalizescale_hashtable_idx(self, inputs, outputs);

    if (p->local.execute_on_sw || !vsi_nn_IsEVISFeatureAvaiable(self->graph->ctx))
    {
        kernel_info.resource_num = 1;
        kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
        kernel_info.resource_name[0] = "vsi_nn_kernel_l2normalizescale";
        kernel_info.type = VX_KERNEL_TYPE_CPU;
        kernel_info.kernel = vx_kernel_L2NORMALIZESCALE_list;
        kernel_info.init_index = 0;

        self->n = vsi_nn_RegisterClientKernelAndNewNode(
        self->graph, &kernel_info);
        if( NULL == self->n )
        {
            status = VSI_FAILURE;
            goto final;
        }
        status = cpu_op_compute(self, inputs, outputs);
        if(VX_SUCCESS != status)
        {
            goto final;
        }
    }
    else
    {
        vsi_nn_tensor_attr_t lc_attr;
        uint32_t i;

        memset(&lc_attr, 0, sizeof(vsi_nn_tensor_attr_t));
        axis = reshape_tensor_set_input_output(self, inputs, outputs, vx_true_e);
        check_const_tensor_shape(self, inputs);
        status  = vsi_nn_vxGetTensorAttr(p->local.local_tensor[0], &lc_attr);
        if(VX_SUCCESS != status)
        {
            goto final;
        }
        input_dims  = lc_attr.dim_num;
        for (i = 0; i < input_dims; i++)
        {
            input_size[i] = lc_attr.size[i];
        }
        input_size[axis] = 1;
        memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
        attr.size[0] = input_size[0];
        attr.size[1] = input_size[1];
        attr.size[2] = input_dims > 2 ? input_size[2] : 1;
        attr.size[3] = input_dims > 3 ? input_size[3] : 1;
        attr.dim_num = input_dims > 1 ? input_dims : 2;
        attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
        attr.vtl = FALSE;
        outputs_sum_r_sqrt[0] = vsi_nn_CreateTensor(self->graph, &attr);

        kernel_info.resource_num = 1;
        kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
        kernel_info.resource_name[0] = "vsi_nn_kernel_l2normalizescale";
        kernel_info.type = vsi_nn_GetVXKernelTypeForShader();
        kernel_info.kernel = vx_kernel_L2NORMALIZESCALE_list;
        kernel_info.init_index = 1;

        if (vsi_nn_is_do_vx_op_pre_init(kernel_info.type))
        {
            vx_op_pre_init_sum_r_sqrt(self, inputs, outputs_sum_r_sqrt, &kernel_info);
        }

        self->n = vsi_nn_RegisterClientKernelAndNewNode(
            self->graph, &kernel_info);
        if( NULL == self->n )
        {
            status = VSI_FAILURE;
            goto final;
        }

        if (NULL != op_compute_list[kernel_info.init_index])
        {
            status = op_compute_list[kernel_info.init_index](self, inputs, outputs_sum_r_sqrt);
        }

        inputs_mul_scale[0] = inputs[0];
        inputs_mul_scale[1] = outputs_sum_r_sqrt[0];
        inputs_mul_scale[2] = inputs[1];

        kernel_info.resource_name[0] = "vsi_nn_kernel_l2normalizescale";
        kernel_info.type = vsi_nn_GetVXKernelTypeForShader();
        kernel_info.kernel = vx_kernel_L2NORMALIZESCALE_list;
        kernel_info.init_index = 2;

        if (vsi_nn_is_do_vx_op_pre_init(kernel_info.type))
        {
            vx_op_pre_init_mul_scale(self, inputs_mul_scale, outputs, &kernel_info);
        }

        self->n = vsi_nn_RegisterClientKernelAndNewNode(
            self->graph, &kernel_info);
        if( NULL == self->n )
        {
            status = VSI_FAILURE;
            goto final;
        }

        if (NULL != op_compute_list[kernel_info.init_index])
        {
            status = op_compute_list[kernel_info.init_index](self, inputs_mul_scale, outputs);
        }
    }

final:
    if (kernel_info.resource_name)
    {
        free(kernel_info.resource_name);
        kernel_info.resource_name = NULL;
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
    for (i = 0; i < _VSI_NN_L2NORMALIZESCALE_LOCAL_TENSOR_NUM; i++)
    {
        if (self->nn_param.l2normalizescale.local.local_tensor[i] != NULL)
        {
            vxReleaseTensor(&(self->nn_param.l2normalizescale.local.local_tensor[i]));
            self->nn_param.l2normalizescale.local.local_tensor[i] = NULL;
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
    vsi_bool ret = TRUE;
    if( NULL == self )
    {
        return FALSE;
    }

    if (self->nn_param.l2normalizescale.axis < 0)
    {
        self->nn_param.l2normalizescale.axis += (int32_t)inputs[0]->attr.dim_num;
    }

    if (self->nn_param.l2normalizescale.axis < 0)
    {
        VSILOGD("l2normalizescale Invalid Axis: %d", self->nn_param.l2normalizescale.axis);
        return FALSE;
    }

    ret = vsi_nn_op_common_setup(self, inputs, outputs);

    return ret;
}

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;
    uint32_t  i;

    if (vsi_nn_compareVersion(self->graph, 1, 1, 13) == -1)
    {
        self->nn_param.l2normalizescale.axis = VSI_NN_L2NORMALIZESCALE_DEFAULT_AXIS;
    }
    for (i = 0; i < _VSI_NN_L2NORMALIZESCALE_LOCAL_TENSOR_NUM; i++)
    {
        self->nn_param.l2normalizescale.local.local_tensor[i] = NULL;
    }

    return status;
} /* op_init() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ L2NORMALIZESCALE,
    /* init       */ op_init,
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

