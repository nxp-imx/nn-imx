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
#include "utils/vsi_nn_util.h"
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

/* Type enum */
typedef enum _minimum_nn_image_dims_e
{
    IMAGE_2D = TRUE,
    IMAGE_3D = FALSE,
}minimum_nn_image_dims_e;

#define VSI_NN_GEN_MINIMUM_KEY(_input0_type, _input1_type, _output_type, _image_2d) \
    ((_input0_type << 24) | (_input1_type << 16) | (_output_type << 8) | (_image_2d))

#define VSI_NN_GEN_MINIMUM_STRUCT_ITEMS(_input0_type, _input1_type, _output_type, _image_2d) \
    VSI_NN_GEN_MINIMUM_KEY(_input0_type, _input1_type, _output_type, _image_2d), \
    VSI_NN_MINIMUM_SH_KERNEL_IDX(_input0_type, _input1_type, _output_type, _image_2d)

static struct {
        uint32_t key;
        uint32_t kernel_index;
        char *resource_name;
    } minimum_map[] =
    {
        {VSI_NN_GEN_MINIMUM_STRUCT_ITEMS(F16, F16, F16, IMAGE_3D)  "vsi_nn_kernel_minimum"},
        {VSI_NN_GEN_MINIMUM_STRUCT_ITEMS(I8,  F16, I8,  IMAGE_3D)  "vsi_nn_kernel_minimum_fp16"},
        {VSI_NN_GEN_MINIMUM_STRUCT_ITEMS(I8,  F16, F16, IMAGE_3D)  "vsi_nn_kernel_minimum_fp16"},
        {VSI_NN_GEN_MINIMUM_STRUCT_ITEMS(U8,  F16, U8,  IMAGE_3D)  "vsi_nn_kernel_minimum_fp16"},
        {VSI_NN_GEN_MINIMUM_STRUCT_ITEMS(U8,  F16, F16, IMAGE_3D)  "vsi_nn_kernel_minimum_fp16"},
        {VSI_NN_GEN_MINIMUM_STRUCT_ITEMS(I8,  I8,  I8,  IMAGE_3D)  "vsi_nn_kernel_minimum"},
        {VSI_NN_GEN_MINIMUM_STRUCT_ITEMS(U8,  U8,  U8,  IMAGE_3D)  "vsi_nn_kernel_minimum"},
        {VSI_NN_GEN_MINIMUM_STRUCT_ITEMS(I16, I16, I16, IMAGE_3D)  "vsi_nn_kernel_minimum"},
        {VSI_NN_GEN_MINIMUM_STRUCT_ITEMS(I16, F16, I16, IMAGE_3D)  "vsi_nn_kernel_minimum_i16"},
        {VSI_NN_GEN_MINIMUM_STRUCT_ITEMS(I16, F16, F16, IMAGE_3D)  "vsi_nn_kernel_minimum_i16"},
        {VSI_NN_GEN_MINIMUM_STRUCT_ITEMS(F16, F16, U8,  IMAGE_3D)  "vsi_nn_kernel_minimum"},
        {VSI_NN_GEN_MINIMUM_STRUCT_ITEMS(F16, F16, I8,  IMAGE_3D)  "vsi_nn_kernel_minimum"},

        {VSI_NN_GEN_MINIMUM_STRUCT_ITEMS(F16, F16, F16, IMAGE_2D)  "vsi_nn_kernel_minimum"},
        {VSI_NN_GEN_MINIMUM_STRUCT_ITEMS(I8,  F16, I8,  IMAGE_2D)  "vsi_nn_kernel_minimum_fp16"},
        {VSI_NN_GEN_MINIMUM_STRUCT_ITEMS(I8,  F16, F16, IMAGE_2D)  "vsi_nn_kernel_minimum_fp16"},
        {VSI_NN_GEN_MINIMUM_STRUCT_ITEMS(U8,  F16, U8,  IMAGE_2D)  "vsi_nn_kernel_minimum_fp16"},
        {VSI_NN_GEN_MINIMUM_STRUCT_ITEMS(U8,  F16, F16, IMAGE_2D)  "vsi_nn_kernel_minimum_fp16"},
        {VSI_NN_GEN_MINIMUM_STRUCT_ITEMS(I8,  I8,  I8,  IMAGE_2D)  "vsi_nn_kernel_minimum"},
        {VSI_NN_GEN_MINIMUM_STRUCT_ITEMS(U8,  U8,  U8,  IMAGE_2D)  "vsi_nn_kernel_minimum"},
        {VSI_NN_GEN_MINIMUM_STRUCT_ITEMS(I16, I16, I16, IMAGE_2D)  "vsi_nn_kernel_minimum"},
        {VSI_NN_GEN_MINIMUM_STRUCT_ITEMS(I16, F16, I16, IMAGE_2D)  "vsi_nn_kernel_minimum_i16"},
        {VSI_NN_GEN_MINIMUM_STRUCT_ITEMS(I16, F16, F16, IMAGE_2D)  "vsi_nn_kernel_minimum_i16"},
        {VSI_NN_GEN_MINIMUM_STRUCT_ITEMS(F16, F16, U8,  IMAGE_2D)  "vsi_nn_kernel_minimum"},
        {VSI_NN_GEN_MINIMUM_STRUCT_ITEMS(F16, F16, I8,  IMAGE_2D)  "vsi_nn_kernel_minimum"},
    };

static void check_tensor_shape
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vx_reference * params,
    uint32_t index,
    vsi_bool rsFlg,
    uint32_t *sizes,
    uint32_t dims
    )
{
    vsi_nn_tensor_attr_t attr;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));

    if (rsFlg)
    {
        self->nn_param.minimum.local->local_tensor[index] =
            vxReshapeTensor(input->t, (int32_t*)(sizes), dims);
        params[index] =  (vx_reference)self->nn_param.minimum.local->local_tensor[index];
    }
    else
    {
        if( input->attr.dim_num == 1)
        {
            memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
            attr.size[1] = 1;
            attr.dim_num = 2;
            self->nn_param.minimum.local->local_tensor[index] =
                vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
            params[index] =  (vx_reference)self->nn_param.minimum.local->local_tensor[index];
        }
        else
        {
            params[index] = (vx_reference)input->t;
            self->nn_param.minimum.local->local_tensor[index] = NULL;
        }
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

static vsi_nn_shader_kernel_type_e get_minimum_intra_type(vsi_nn_type_e type)
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
    case VSI_NN_TYPE_BFLOAT16:
        return F16;
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
#define VSI_NN_TENSOR_WIDTH_MAX (65536)
    vsi_status status = VSI_SUCCESS;
    vx_reference params[_PARAM_NUM];
    vx_border_t border;
    vsi_bool    enable_image_2d  = FALSE;
    vsi_nn_minimum_param * p     = NULL;

    border.mode = VX_BORDER_REPLICATE;
    border.constant_value.U32 = 0;

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    p = &(self->nn_param.minimum);
    enable_image_2d = p->local->enable_image_2d;

    //_set_inputs_outputs( params, inputs, outputs );
    check_tensor_shape(self, inputs[0],  params, 0, enable_image_2d, p->local->sizes0, p->local->dim_num);
    check_tensor_shape(self, inputs[1],  params, 1, enable_image_2d, p->local->sizes1, p->local->dim_num);
    check_tensor_shape(self, outputs[0], params, 2, enable_image_2d, p->local->sizes2, p->local->dim_num);

    /* Pass parameters to node. */
    status  = vsi_nn_ClientNodePassParameters( self->n, params, _PARAM_NUM );
    status |= vxSetNodeAttribute(self->n, VX_NODE_BORDER, &border, sizeof(border));

#undef VSI_NN_TENSOR_WIDTH_MAX
    return status;
}

static void _get_minimum_hashtable_idx
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
#define VSI_NN_TENSOR_WIDTH_MAX (65536)
    vsi_nn_type_e input0Format = inputs[0]->attr.dtype.vx_type;
    vsi_nn_type_e input1Format = inputs[1]->attr.dtype.vx_type;
    vsi_nn_type_e outputFormat = outputs[0]->attr.dtype.vx_type;
    vsi_nn_shader_kernel_type_e _input0_type;
    vsi_nn_shader_kernel_type_e _input1_type;
    vsi_nn_shader_kernel_type_e _output_type;
    vsi_bool    enable_image_2d  = FALSE;
    uint32_t    key = 0;
    uint32_t    i   = 0;
    vsi_nn_minimum_param * p     = NULL;

    p = &(self->nn_param.minimum);

   enable_image_2d = p->local->enable_image_2d;
#undef VSI_NN_TENSOR_WIDTH_MAX

    _input0_type = get_minimum_intra_type(input0Format);
    _input1_type = get_minimum_intra_type(input1Format);
    _output_type = get_minimum_intra_type(outputFormat);

    key = VSI_NN_GEN_MINIMUM_KEY(_input0_type, _input1_type, _output_type, enable_image_2d);

    for (i = 0; i < sizeof(minimum_map) / sizeof(minimum_map[0]); i++)
    {
        if (key == minimum_map[i].key)
        {
            p->local->hash_idx = i;
            p->local->execute_on_sw = FALSE;
            return;
        }
    }

    p->local->execute_on_sw = TRUE;
    VSILOGE("Shader unsupport data format! execute on the SW [minimum]\n");
}

static vsi_bool vx_op_pre_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_kernel_info_t * kernel_info
    )
{
    vsi_nn_minimum_param * p = NULL;

    p = &(self->nn_param.minimum);

    kernel_info->kernel_index = minimum_map[p->local->hash_idx].kernel_index;
    kernel_info->resource_num = 2;
    kernel_info->resource_name[0] = "vsi_nn_kernel_header";
    kernel_info->resource_name[1] = minimum_map[p->local->hash_idx].resource_name;

    return TRUE;
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
    vsi_nn_kernel_info_t kernel_info;
    vsi_nn_type_e input0_Format = inputs[0]->attr.dtype.vx_type;
    vsi_nn_type_e input1_Format = inputs[1]->attr.dtype.vx_type;
    vsi_nn_minimum_param * p    = NULL;

    memset(&kernel_info, 0x0, sizeof(vsi_nn_kernel_info_t));
    status = VSI_FAILURE;
    if( NULL == self )
    {
        return VSI_FAILURE;
    }
    p = &(self->nn_param.minimum);

    if (input0_Format != input1_Format && input0_Format == VSI_NN_TYPE_FLOAT16)
        SWAP_INPUT_TENSOR(inputs[0], inputs[1]);

    p->local->enable_image_2d = vsi_nn_OptimizedEltWiseOPShape(inputs[0], inputs[1], outputs[0],
            p->local->sizes0, p->local->sizes1, p->local->sizes2, &p->local->dim_num);

    _get_minimum_hashtable_idx(self, inputs, outputs);

    if (p->local->execute_on_sw || !vsi_nn_IsEVISFeatureAvaiable(self->graph->ctx))
    {
        kernel_info.resource_num = 1;
        kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
        kernel_info.resource_name[0] = "vsi_nn_kernel_minimum";
        kernel_info.type = VX_KERNEL_TYPE_CPU;
        kernel_info.kernel = vx_kernel_MINIMUM_list;
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
        kernel_info.kernel = vx_kernel_MINIMUM_list;
        kernel_info.resource_num = 2;
        kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
        kernel_info.init_index = 1;
        kernel_info.resource_name[0] = "vsi_nn_kernel_header";
        kernel_info.resource_name[1] = "vsi_nn_kernel_minimum";

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

    if (self->nn_param.minimum.local)
    {
        for (i = 0; i < _VSI_NN_MINIMUM_LOCAL_TENSOR_NUM; i++)
        {
            if (self->nn_param.minimum.local->local_tensor[i] != NULL)
            {
                vxReleaseTensor(&(self->nn_param.minimum.local->local_tensor[i]));
                self->nn_param.minimum.local->local_tensor[i] = NULL;
            }
        }
        free(self->nn_param.minimum.local);
        self->nn_param.minimum.local = NULL;
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

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    self->nn_param.minimum.local   =
    (vsi_nn_minimum_lcl_data *)malloc(sizeof(vsi_nn_minimum_lcl_data));
    if (NULL == self->nn_param.minimum.local)
    {
        return  VX_ERROR_NO_MEMORY;
    }

    return status;
} /* op_init() */


#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ MINIMUM,
    /* init       */ op_init,
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
