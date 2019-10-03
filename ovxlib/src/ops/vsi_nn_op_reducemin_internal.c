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
#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)

extern vx_kernel_description_t * vx_kernel_REDUCEMIN_INTERNAL_list[];

/* Type enum */
typedef enum _reducemin_nn_image_dims_e
{
    IMAGE_2D  = TRUE,
    IMAGE     = FALSE,
}reducemin_nn_activation_type_e;

typedef enum _reducemin_nn_type_e
{
    I8 = 0,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F16,
    F32,
    BF16,
}reducemin_nn_type_e;

#define VSI_NN_GEN_REDUCEMIN_KEY(_axis, _input_type, _output_type, _image_2d) \
    ((_axis << 20) | (_input_type << 12) | (_output_type << 4) | (_image_2d))

#define VSI_NN_GEN_REDUCEMIN_KERNEL_SOURCE_NAME(_suffix) \
    "vsi_nn_kernel_internal_reduceminAxis"#_suffix

#define VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(_axis, _input_type, _output_type, _image_2d) \
    VSI_NN_GEN_REDUCEMIN_KEY(_axis, _input_type, _output_type, _image_2d), \
    VSI_NN_REDUCEMIN_SH_KERNEL_IDX(_axis, _input_type, _output_type, _image_2d) \
    VSI_NN_GEN_REDUCEMIN_KERNEL_SOURCE_NAME(_axis)

static struct {
        uint32_t key;
        uint32_t kernel_index;
        char *resource_name;
    } reducemin_map[] =
    {
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(0, F16, F16, IMAGE)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(0, F16, I16, IMAGE)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(0, F16, I8, IMAGE)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(0, F16, U8, IMAGE)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(0, I16, I16, IMAGE)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(0, I8,  I8, IMAGE)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(0, U8,  U8, IMAGE)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(0, I16, F16, IMAGE)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(0, I8,  F16, IMAGE)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(0, U8,  F16, IMAGE)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(0, F16, F16, IMAGE_2D)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(0, F16, I16, IMAGE_2D)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(0, F16, I8,  IMAGE_2D)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(0, F16, U8,  IMAGE_2D)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(0, I16, I16, IMAGE_2D)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(0, I8,  I8,  IMAGE_2D)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(0, U8,  U8,  IMAGE_2D)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(0, I16, F16, IMAGE_2D)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(0, I8,  F16, IMAGE_2D)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(0, U8,  F16, IMAGE_2D)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(1, F16, F16, IMAGE)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(1, F16, I16, IMAGE)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(1, F16, I8, IMAGE)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(1, F16, U8, IMAGE)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(1, I16, I16, IMAGE)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(1, I8,  I8, IMAGE)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(1, U8,  U8, IMAGE)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(1, I16, F16, IMAGE)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(1, I8,  F16, IMAGE)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(1, U8,  F16, IMAGE)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(1, F16, F16, IMAGE_2D)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(1, F16, I16, IMAGE_2D)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(1, F16, I8,  IMAGE_2D)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(1, F16, U8,  IMAGE_2D)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(1, I16, I16, IMAGE_2D)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(1, I8,  I8,  IMAGE_2D)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(1, U8,  U8,  IMAGE_2D)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(1, I16, F16, IMAGE_2D)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(1, I8,  F16, IMAGE_2D)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(1, U8,  F16, IMAGE_2D)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(2, F16, F16, IMAGE)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(2, F16, I16, IMAGE)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(2, F16, I8, IMAGE)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(2, F16, U8, IMAGE)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(2, I16, I16, IMAGE)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(2, I8,  I8, IMAGE)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(2, U8,  U8, IMAGE)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(2, I16, F16, IMAGE)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(2, I8,  F16, IMAGE)},
        {VSI_NN_GEN_REDUCEMIN_STRUCT_ITEMS(2, U8,  F16, IMAGE)},
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
    int32_t  axis = self->nn_param.reducemin_internal.axis[0];
#define VSI_NN_TENSOR_WIDTH_MAX (65536)

    if (axis == 0)
    {
        if (dims < 3 || (inputs[0]->attr.size[1] * inputs[0]->attr.size[2] < VSI_NN_TENSOR_WIDTH_MAX))
            ret = TRUE;
    }
    else if (axis == 2)
    {
        if (dims < 3 || (inputs[0]->attr.size[0] * inputs[0]->attr.size[1] < VSI_NN_TENSOR_WIDTH_MAX))
            ret = TRUE;
    }
    else if (axis == 1)
    {
        if (dims < 3 || (inputs[0]->attr.size[0] == 1))
            ret = TRUE;
    }

#undef VSI_NN_TENSOR_WIDTH_MAX

    return ret;
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
    vsi_nn_reducemin_internal_param * p;
    if( 0 == num )
    {
        return VSI_SUCCESS;
    }
    memset( params, 0, sizeof( vx_reference * ) * num );
    p = &(node->nn_param.reducemin_internal);
    ctx = vxGetContext( (vx_reference)node->graph->g );
    /* Init parameters */
    #define _SET_PARAM( i, type, arg ) do{ \
        params[i] = (vx_reference)vxCreateScalar( ctx, type, &p->arg ); \
        status = vxGetStatus( params[i] ); \
        if( VSI_SUCCESS != status ) { \
            goto set_param_error; \
            } \
        } while(0)
    _SET_PARAM( 0, VX_TYPE_INT32,   axis[0] );
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

static reducemin_nn_type_e get_reducemin_intra_type(vsi_nn_type_e type)
{
    switch (type)
    {
    case VSI_NN_TYPE_INT8:
        return I8;
    case VSI_NN_TYPE_INT16:
        return I16;
    case VSI_NN_TYPE_INT32:
        return I32;
    case VSI_NN_TYPE_INT64:
        return I64;
    case VSI_NN_TYPE_UINT8:
        return U8;
    case VSI_NN_TYPE_UINT16:
        return U16;
    case VSI_NN_TYPE_UINT32:
        return U32;
    case VSI_NN_TYPE_FLOAT16:
        return F16;
    case VSI_NN_TYPE_FLOAT32:
        return F32;
    case VSI_NN_TYPE_BFLOAT16:
        return BF16;
    default:
        VSILOGE("error data type %d", type);
        break;
    }
    return I8;
}


static int32_t reshape_tensor_set_input_output
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vsi_nn_tensor_t * output,
    vx_reference * params
    )
{
    uint32_t sizes[VSI_NN_MAX_DIM_NUM] = {1};
    uint32_t dst_sizes[VSI_NN_MAX_DIM_NUM] = {1};
    uint32_t dims = vsi_nn_max(input->attr.dim_num, 2);
    int32_t axis = 0;
    vsi_nn_reducemin_internal_param * p;
    vsi_bool is_2d_image = FALSE;

    p = &(self->nn_param.reducemin_internal);
    axis = p->axis[0];

    is_2d_image = _check_tensor_shape(self, &input, &output);

    if (axis == 0)
    {
        sizes[0] = input->attr.size[0];

        if (is_2d_image)
        {
            sizes[1] = input->attr.size[1] * input->attr.size[2];
            sizes[2] = 1;
            sizes[3] = dims > 3 ? input->attr.size[3] : 1;

            dst_sizes[0] = 1;
            dst_sizes[1] = input->attr.size[1] * input->attr.size[2];
            dst_sizes[2] = 1;
            dst_sizes[3] = dims > 3 ? input->attr.size[3] : 1;
        }
        else
        {
            sizes[1] = input->attr.size[1];
            sizes[2] = dims > 2 ? input->attr.size[2] : 1;
            sizes[3] = dims > 3 ? input->attr.size[3] : 1;

            dst_sizes[0] = 1;
            dst_sizes[1] = input->attr.size[1];
            dst_sizes[2] = dims > 2 ? input->attr.size[2] : 1;
            dst_sizes[3] = dims > 3 ? input->attr.size[3] : 1;
        }
    }
    else if (axis == 1)
    {
        if (1 == input->attr.size[0])
        {
            sizes[0] = input->attr.size[1];
            sizes[1] = dims > 2 ? input->attr.size[2] : 1;
            sizes[2] = 1;
            sizes[3] = dims > 3 ? input->attr.size[3] : 1;
            dst_sizes[0] = 1;
            dst_sizes[1] = sizes[1];
            dst_sizes[2] = sizes[2];
            dst_sizes[3] = sizes[3];
            axis = 0;
        }
        else
        {
            sizes[0] = input->attr.size[0];
            sizes[1] = input->attr.size[1];
            sizes[2] = dims > 2 ? input->attr.size[2] : 1;
            sizes[3] = dims > 3 ? input->attr.size[3] : 1;
            dst_sizes[0] = input->attr.size[0];
            dst_sizes[1] = 1;
            dst_sizes[2] = dims > 2 ? input->attr.size[2] : 1;
            dst_sizes[3] = dims > 3 ? input->attr.size[3] : 1;
        }
    }
    else if (axis == 2)
    {
        if(1 == input->attr.size[0] && 1 == input->attr.size[1])
        {
            sizes[0] = input->attr.size[2];
            sizes[1] = 1;
            sizes[2] = 1;
            sizes[3] = dims > 3 ? input->attr.size[3] : 1;

            dst_sizes[0] = 1;
            dst_sizes[1] = 1;
            dst_sizes[2] = 1;
            dst_sizes[3] = dims > 3 ? input->attr.size[3] : 1;
            axis = 0;
        }
        else if (is_2d_image)
        {
            sizes[0] = input->attr.size[0] * input->attr.size[1];
            sizes[1] = input->attr.size[2];
            sizes[2] = 1;
            sizes[3] = dims > 3 ? input->attr.size[3] : 1;

            dst_sizes[0] = input->attr.size[0] * input->attr.size[1];
            dst_sizes[1] = 1;
            dst_sizes[2] = 1;
            dst_sizes[3] = dims > 3 ? input->attr.size[3] : 1;
            axis = 1;
        }
        else
        {
            sizes[0] = input->attr.size[0];
            sizes[1] = input->attr.size[1];
            sizes[2] = dims > 2 ? input->attr.size[2] : 1;
            sizes[3] = dims > 3 ? input->attr.size[3] : 1;

            dst_sizes[0] = input->attr.size[0];
            dst_sizes[1] = input->attr.size[1];
            dst_sizes[2] = 1;
            dst_sizes[3] = dims > 3 ? input->attr.size[3] : 1;
        }
    }

    if (params)
    {
        p->axis[0] = axis;
        self->nn_param.reducemin_internal.local.local_tensor[0] =
            vxReshapeTensor(input->t, (int32_t *)sizes, dims);
        self->nn_param.reducemin_internal.local.local_tensor[1] =
            vxReshapeTensor(output->t, (int32_t *)dst_sizes, dims);
        params[0] = (vx_reference)self->nn_param.reducemin_internal.local.local_tensor[0];
        params[1] = (vx_reference)self->nn_param.reducemin_internal.local.local_tensor[1];
    }

    return axis;
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
    vx_reference * args;
    uint32_t dims = vsi_nn_max(inputs[0]->attr.dim_num, 2);
    int32_t axis = 0;
    vsi_nn_reducemin_internal_param * p;
    vx_border_t border;

    p = &(self->nn_param.reducemin_internal);
    axis = p->axis[0];
    if (axis < 0)
    {
        axis = (int32_t)dims + axis;
        if (axis < 0)
        {
            axis = 0;
        }
        p->axis[0] = axis;
    }

    args = &params[_IO_NUM];

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    /* Set inputs and outputs */
    reshape_tensor_set_input_output( self, inputs[0], outputs[0], params);
    /*TODO: Add code if need to change your parameter*/

    /* Init parameters. */
    _create_params( self, args, _ARG_NUM );

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, _PARAM_NUM );

    _release_params( args, _ARG_NUM );
    border.mode = VX_BORDER_REPLICATE;
    status |= vxSetNodeAttribute(self->n, VX_NODE_BORDER, &border, sizeof(border));

    return status;
}

static void _get_reducemin_hashtable_idx
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_type_e inputFormat = inputs[0]->attr.dtype.vx_type;
    vsi_nn_type_e outputFormat  = outputs[0]->attr.dtype.vx_type;
    reducemin_nn_type_e _input_type;
    reducemin_nn_type_e _output_type;
    int32_t axis = 0;
    uint32_t key;
    vsi_bool is_2d_image = FALSE;
    uint32_t i = 0;
    uint32_t dims = vsi_nn_max(inputs[0]->attr.dim_num, 2);
    vsi_nn_reducemin_internal_param * p;

    p = &(self->nn_param.reducemin_internal);
    axis = p->axis[0];
    if (axis < 0)
    {
        axis = (int32_t)dims + axis;
        if (axis < 0)
        {
            axis = 0;
        }
        p->axis[0] = axis;
    }

    axis = reshape_tensor_set_input_output(self, inputs[0], outputs[0], NULL);
    _input_type = get_reducemin_intra_type(inputFormat);
    _output_type = get_reducemin_intra_type(outputFormat);
    if (BF16 == _input_type && BF16 == _output_type)
    {
        _input_type  = F16;
        _output_type = F16;
    }
    is_2d_image = _check_tensor_shape(self, inputs, outputs);
    key = VSI_NN_GEN_REDUCEMIN_KEY(axis, _input_type, _output_type, is_2d_image);

    for (i = 0; i < sizeof(reducemin_map) / sizeof(reducemin_map[0]); i++)
    {
        if (key == reducemin_map[i].key)
        {
            p->local.hash_idx = i;
            p->local.execute_on_sw = FALSE;
            return;
        }
    }

    p->local.execute_on_sw = TRUE;
    VSILOGE("Shader unsupport data format or axis! execute on the SW [reducemin]\n");
}

static vsi_bool vx_op_pre_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_kernel_info_t * kernel_info
    )
{
    vsi_nn_reducemin_internal_param * p;

    p = &(self->nn_param.reducemin_internal);

    kernel_info->kernel_index = reducemin_map[p->local.hash_idx].kernel_index;
    kernel_info->resource_num = 2;
    kernel_info->resource_name[0] = "vsi_nn_kernel_header";
    kernel_info->resource_name[1] = reducemin_map[p->local.hash_idx].resource_name;

    return TRUE;
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
    vsi_nn_reducemin_internal_param * p;

    status = VSI_FAILURE;
    if( NULL == self )
    {
        return VSI_FAILURE;
    }

    p = &(self->nn_param.reducemin_internal);
    _get_reducemin_hashtable_idx(self, inputs, outputs);

   if (p->local.execute_on_sw)
    {
        kernel_info.resource_num = 1;
        kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
        kernel_info.resource_name[0] = "vsi_nn_kernel_internal_reducemin";
        kernel_info.type = VX_KERNEL_TYPE_CPU;
        kernel_info.kernel = vx_kernel_REDUCEMIN_INTERNAL_list;
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
    else
    {
        kernel_info.type = vsi_nn_GetVXKernelTypeForShader();
        kernel_info.kernel = vx_kernel_REDUCEMIN_INTERNAL_list;
        kernel_info.resource_num = 2;
        kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
        kernel_info.init_index = 1;
        kernel_info.resource_name[0] = "vsi_nn_kernel_header";
        kernel_info.resource_name[1] = "vsi_nn_kernel_internal_reduceminAxis0";

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
    return TRUE;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    uint32_t i;

    for (i = 0; i < _VSI_NN_REDUCEMIN_LOCAL_TENSOR_NUM; i++)
    {
        if (self->nn_param.reducemin_internal.local.local_tensor[i] != NULL)
        {
            vxReleaseTensor(&(self->nn_param.reducemin_internal.local.local_tensor[i]));
            self->nn_param.reducemin_internal.local.local_tensor[i] = NULL;
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
    /* op_name    */ REDUCEMIN_INTERNAL,
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
