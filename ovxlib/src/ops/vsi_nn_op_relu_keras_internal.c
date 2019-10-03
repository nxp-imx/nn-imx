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

#define _ARG_NUM            (3)
#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)

extern vx_kernel_description_t * vx_kernel_RELU_KERAS_INTERNAL_list[];

/* Type enum */
typedef enum _keras_relu_nn_image_dims_e
{
    IMAGE_2D = TRUE,
    IMAGE_ARRAY = FALSE,
}keras_relu_nn_activation_type_e;

typedef enum _keras_relu_nn_type_e
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
}keras_relu_nn_type_e;

#define VSI_NN_GEN_KERAS_RELU_KEY(_input_type, _output_type, _image_2d) \
    ((_input_type << 20) | (_output_type << 10) | (_image_2d))

#define VSI_NN_GEN_KERAS_RELU_STRUCT_ITEMS(_input_type, _output_type, _image_2d) \
    VSI_NN_GEN_KERAS_RELU_KEY(_input_type, _output_type, _image_2d), \
    VSI_NN_KERAS_RELU_SH_KERNEL_IDX(_input_type, _output_type, _image_2d) \

static struct {
        uint32_t key;
        uint32_t kernel_index;
    } keras_relu_map[] =
    {
        {VSI_NN_GEN_KERAS_RELU_STRUCT_ITEMS(BF16, BF16, IMAGE_ARRAY)},
        {VSI_NN_GEN_KERAS_RELU_STRUCT_ITEMS(F16,  F16,  IMAGE_ARRAY)},
        {VSI_NN_GEN_KERAS_RELU_STRUCT_ITEMS(F16,  I16,  IMAGE_ARRAY)},
        {VSI_NN_GEN_KERAS_RELU_STRUCT_ITEMS(F16,  I8,   IMAGE_ARRAY)},
        {VSI_NN_GEN_KERAS_RELU_STRUCT_ITEMS(F16,  U8,   IMAGE_ARRAY)},
        {VSI_NN_GEN_KERAS_RELU_STRUCT_ITEMS(I16,  I16,  IMAGE_ARRAY)},
        {VSI_NN_GEN_KERAS_RELU_STRUCT_ITEMS(I16,  F16,  IMAGE_ARRAY)},
        {VSI_NN_GEN_KERAS_RELU_STRUCT_ITEMS(I8,   I8,   IMAGE_ARRAY)},
        {VSI_NN_GEN_KERAS_RELU_STRUCT_ITEMS(I8,   F16,  IMAGE_ARRAY)},
        {VSI_NN_GEN_KERAS_RELU_STRUCT_ITEMS(U8,   U8,   IMAGE_ARRAY)},
        {VSI_NN_GEN_KERAS_RELU_STRUCT_ITEMS(U8,   F16,  IMAGE_ARRAY)},

        {VSI_NN_GEN_KERAS_RELU_STRUCT_ITEMS(BF16, BF16, IMAGE_2D)},
        {VSI_NN_GEN_KERAS_RELU_STRUCT_ITEMS(F16,  F16,  IMAGE_2D)},
        {VSI_NN_GEN_KERAS_RELU_STRUCT_ITEMS(F16,  I16,  IMAGE_2D)},
        {VSI_NN_GEN_KERAS_RELU_STRUCT_ITEMS(F16,  I8,   IMAGE_2D)},
        {VSI_NN_GEN_KERAS_RELU_STRUCT_ITEMS(F16,  U8,   IMAGE_2D)},
        {VSI_NN_GEN_KERAS_RELU_STRUCT_ITEMS(I16,  I16,  IMAGE_2D)},
        {VSI_NN_GEN_KERAS_RELU_STRUCT_ITEMS(I16,  F16,  IMAGE_2D)},
        {VSI_NN_GEN_KERAS_RELU_STRUCT_ITEMS(I8,   I8,   IMAGE_2D)},
        {VSI_NN_GEN_KERAS_RELU_STRUCT_ITEMS(I8,   F16,  IMAGE_2D)},
        {VSI_NN_GEN_KERAS_RELU_STRUCT_ITEMS(U8,   U8,   IMAGE_2D)},
        {VSI_NN_GEN_KERAS_RELU_STRUCT_ITEMS(U8,   F16,  IMAGE_2D)},
    };

/* Greatest Common Divisor*/
static vsi_bool vxoGetDataDivisors(uint32_t input_value, uint32_t *divisors, uint32_t gcd)
{
    uint32_t i                 = 0;
#define VSI_NN_TENSOR_WIDTH_MAX (65536)
    for (i = vsi_nn_min(input_value, VSI_NN_TENSOR_WIDTH_MAX - 1); i > 0; i --)
    {
        if ((i % gcd == 0) && (input_value % i == 0))
        {
            *divisors = i;

            return TRUE;
        }
    }
#undef VSI_NN_TENSOR_WIDTH_MAX
    return FALSE;
}

static vsi_bool vxoElementOptimization_GetTensorShape(vsi_nn_tensor_t *input,uint32_t sizes[VSI_NN_MAX_DIM_NUM],
                                                      uint32_t * num_of_dims)
{
    uint32_t element_count     = 0;
    uint32_t i                 = 0;

    element_count = vsi_nn_GetElementNum(input);

#define VSI_NN_TENSOR_WIDTH_MAX (65536)
    for (i = 0; i < VSI_NN_MAX_DIM_NUM; i++)
    {
        sizes[i] = 1;
    }

    if (element_count < VSI_NN_TENSOR_WIDTH_MAX)
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

#undef VSI_NN_TENSOR_WIDTH_MAX
    return TRUE;
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

    self->nn_param.relu_keras_internal.local.local_tensor[index] =
         vxReshapeTensor(input->t, (int32_t *)sizes, num_of_dims);

    params[index] = (vx_reference)self->nn_param.exp.local.local_tensor[index];
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
    vsi_nn_relu_keras_internal_param * p;
    if( 0 == num )
    {
        return VSI_SUCCESS;
    }
    memset( params, 0, sizeof( vx_reference * ) * num );
    p = &(node->nn_param.relu_keras_internal);
    ctx = vxGetContext( (vx_reference)node->graph->g );
    /* Init parameters */
    #define _SET_PARAM( i, type, arg ) do{ \
        params[i] = (vx_reference)vxCreateScalar( ctx, type, &p->arg ); \
        status = vxGetStatus( params[i] ); \
        if( VSI_SUCCESS != status ) { \
            goto set_param_error; \
            } \
        } while(0)
    _SET_PARAM( 0, VX_TYPE_FLOAT32, alpha );
    _SET_PARAM( 1, VX_TYPE_FLOAT32, max_value );
    _SET_PARAM( 2, VX_TYPE_FLOAT32, threshold );
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


static keras_relu_nn_type_e get_keras_relu_intra_type(vsi_nn_type_e type)
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
    case VSI_NN_TYPE_BFLOAT16:
        return BF16;
    case VSI_NN_TYPE_FLOAT32:
        return F32;
    default:
        VSILOGE("error data type %d", type);
        break;
    }

    return I8;
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
    vx_border_t border;

    args = &params[_IO_NUM];

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    /* Set inputs and outputs */
    reshape_tensor_shape(self, inputs[0], params, 0);
    reshape_tensor_shape(self, outputs[0], params, 1);

    /* Init parameters. */
    _create_params( self, args, _ARG_NUM );

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, _PARAM_NUM );

    _release_params( args, _ARG_NUM );

    border.mode = VX_BORDER_REPLICATE;
    status |= vxSetNodeAttribute(self->n, VX_NODE_BORDER, &border, sizeof(border));

    return status;
}

static vsi_bool _check_tensor_shape
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = FALSE;
    uint32_t sizes[VSI_NN_MAX_DIM_NUM] = {1};
    uint32_t num_of_dims = vsi_nn_max(inputs[0]->attr.dim_num, 2);

    vxoElementOptimization_GetTensorShape(inputs[0], sizes, &num_of_dims);

    if (num_of_dims < 3 || sizes[2] == 1)
        ret = TRUE;

    return ret;
}

static void _get_keras_relu_hashtable_idx
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_type_e inputFormat = inputs[0]->attr.dtype.vx_type;
    vsi_nn_type_e outputFormat  = outputs[0]->attr.dtype.vx_type;
    keras_relu_nn_type_e _input_type;
    keras_relu_nn_type_e _output_type;
    uint32_t key;
    vsi_bool is_2d_image = FALSE;
    uint32_t i = 0;

    vsi_nn_relu_keras_internal_param * p;

    p = &(self->nn_param.relu_keras_internal);

    _input_type = get_keras_relu_intra_type(inputFormat);
    _output_type = get_keras_relu_intra_type(outputFormat);
    is_2d_image = _check_tensor_shape(self, inputs, outputs);

    key = VSI_NN_GEN_KERAS_RELU_KEY(_input_type, _output_type, is_2d_image);

    for (i = 0; i < sizeof(keras_relu_map) / sizeof(keras_relu_map[0]); i++)
    {
        if (key == keras_relu_map[i].key)
        {
            p->local.hash_idx = i;
            p->local.execute_on_sw = FALSE;
            return;
        }
    }

    p->local.execute_on_sw = TRUE;
    VSILOGE("Shader unsupport data format! execute on the SW [keras relu]\n");
}

static vsi_bool vx_op_pre_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_kernel_info_t * kernel_info
    )
{
    vsi_nn_relu_keras_internal_param * p;

    p = &(self->nn_param.relu_keras_internal);

    kernel_info->kernel_index = keras_relu_map[p->local.hash_idx].kernel_index;
    kernel_info->resource_num = 2;
    kernel_info->resource_name[0] = "vsi_nn_kernel_relu_keras_header";
    kernel_info->resource_name[1] = "vsi_nn_kernel_relu_keras_internal";

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
    vsi_nn_relu_keras_internal_param * p;

    status = VSI_FAILURE;
    if( NULL == self )
    {
        return VSI_FAILURE;
    }

    p = &(self->nn_param.relu_keras_internal);

   _get_keras_relu_hashtable_idx(self, inputs, outputs);

   if (p->local.execute_on_sw)
    {
        kernel_info.resource_num = 1;
        kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
        kernel_info.resource_name[0] = "vsi_nn_kernel_keras_relu_internal";
        kernel_info.type = VX_KERNEL_TYPE_CPU;
        kernel_info.kernel = vx_kernel_RELU_KERAS_INTERNAL_list;
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
        kernel_info.kernel = vx_kernel_RELU_KERAS_INTERNAL_list;
        kernel_info.resource_num = 2;
        kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
        kernel_info.init_index = 1;
        kernel_info.resource_name[0] = "vsi_nn_kernel_relu_keras_header";
        kernel_info.resource_name[1] = "vsi_nn_kernel_relu_keras_internal";

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

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    uint32_t i;
    for (i = 0; i < _VSI_NN_RELU_KERAS_INTERNAL_LOCAL_TENSOR_NUM; i++)
    {
        if (self->nn_param.relu_keras_internal.local.local_tensor[i] != NULL)
        {
            vxReleaseTensor(&(self->nn_param.relu_keras_internal.local.local_tensor[i]));
            self->nn_param.relu_keras_internal.local.local_tensor[i] = NULL;
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
    /* op_name    */ RELU_KERAS_INTERNAL,
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
