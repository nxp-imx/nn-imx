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
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"

#define _ARG_NUM            (0)
#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)

extern vx_kernel_description_t * vx_kernel_LOG_list[];

/* Type enum */
typedef enum _log_nn_image_dims_e
{
    IMAGE_2D = TRUE,
    IMAGE_3D = FALSE,
}log_nn_image_dims_e;


#define VSI_NN_GEN_LOG_KEY(_input_type, _output_type, _image_2d) \
    ((_input_type << 24) | (_output_type << 8) | (_image_2d))

#define VSI_NN_GEN_LOG_STRUCT_ITEMS(_input_type, _output_type, _image_2d) \
    VSI_NN_GEN_LOG_KEY(_input_type, _output_type, _image_2d), \
    VSI_NN_LOG_SH_KERNEL_IDX(_input_type, _output_type, _image_2d)

static struct {
        uint32_t key;
        uint32_t kernel_index;
        char *resource_name;
    } log_map[] =
    {
        {VSI_NN_GEN_LOG_STRUCT_ITEMS(F16,  F16,  IMAGE_3D)  "vsi_nn_kernel_log"},
        {VSI_NN_GEN_LOG_STRUCT_ITEMS(F16,  I8,   IMAGE_3D)  "vsi_nn_kernel_log"},
        {VSI_NN_GEN_LOG_STRUCT_ITEMS(F16,  I16,  IMAGE_3D)  "vsi_nn_kernel_log"},
        {VSI_NN_GEN_LOG_STRUCT_ITEMS(F16,  U8,   IMAGE_3D)  "vsi_nn_kernel_log"},
        {VSI_NN_GEN_LOG_STRUCT_ITEMS(I8,   I8,   IMAGE_3D)  "vsi_nn_kernel_log"},
        {VSI_NN_GEN_LOG_STRUCT_ITEMS(I8,   F16,  IMAGE_3D)  "vsi_nn_kernel_log"},
        {VSI_NN_GEN_LOG_STRUCT_ITEMS(I16,  I16,  IMAGE_3D)  "vsi_nn_kernel_log"},
        {VSI_NN_GEN_LOG_STRUCT_ITEMS(I16,  F16,  IMAGE_3D)  "vsi_nn_kernel_log"},
        {VSI_NN_GEN_LOG_STRUCT_ITEMS(U8,   U8,   IMAGE_3D)  "vsi_nn_kernel_log"},
        {VSI_NN_GEN_LOG_STRUCT_ITEMS(U8,   F16,  IMAGE_3D)  "vsi_nn_kernel_log"},
        {VSI_NN_GEN_LOG_STRUCT_ITEMS(BF16, BF16, IMAGE_3D)  "vsi_nn_kernel_log"},

        {VSI_NN_GEN_LOG_STRUCT_ITEMS(F16,  F16,  IMAGE_2D)  "vsi_nn_kernel_log"},
        {VSI_NN_GEN_LOG_STRUCT_ITEMS(F16,  I8,   IMAGE_2D)  "vsi_nn_kernel_log"},
        {VSI_NN_GEN_LOG_STRUCT_ITEMS(F16,  I16,  IMAGE_2D)  "vsi_nn_kernel_log"},
        {VSI_NN_GEN_LOG_STRUCT_ITEMS(F16,  U8,   IMAGE_2D)  "vsi_nn_kernel_log"},
        {VSI_NN_GEN_LOG_STRUCT_ITEMS(I8,   I8,   IMAGE_2D)  "vsi_nn_kernel_log"},
        {VSI_NN_GEN_LOG_STRUCT_ITEMS(I8,   F16,  IMAGE_2D)  "vsi_nn_kernel_log"},
        {VSI_NN_GEN_LOG_STRUCT_ITEMS(I16,  I16,  IMAGE_2D)  "vsi_nn_kernel_log"},
        {VSI_NN_GEN_LOG_STRUCT_ITEMS(I16,  F16,  IMAGE_2D)  "vsi_nn_kernel_log"},
        {VSI_NN_GEN_LOG_STRUCT_ITEMS(U8,   U8,   IMAGE_2D)  "vsi_nn_kernel_log"},
        {VSI_NN_GEN_LOG_STRUCT_ITEMS(U8,   F16,  IMAGE_2D)  "vsi_nn_kernel_log"},
        {VSI_NN_GEN_LOG_STRUCT_ITEMS(BF16, BF16, IMAGE_2D)  "vsi_nn_kernel_log"},
    };

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

static void reshape_tensor_shape
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vx_reference * params,
    uint32_t index
    )
{
    uint32_t sizes[VSI_NN_MAX_DIM_NUM] = {1};
    uint32_t num_of_dims = 0;

    vsi_nn_OptimizedEltOPShape(input, sizes, &num_of_dims);
    num_of_dims = vsi_nn_max(input->attr.dim_num, 2);
    num_of_dims = vsi_nn_min(input->attr.dim_num, 4);

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

static void _get_log_hashtable_idx
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_type_e inputFormat = inputs[0]->attr.dtype.vx_type;
    vsi_nn_type_e outputFormat = outputs[0]->attr.dtype.vx_type;
    vsi_nn_kernel_dtype_e _input_type;
    vsi_nn_kernel_dtype_e _output_type;
    vsi_bool enable_image_2d  = FALSE;
    uint32_t sizes[VSI_NN_MAX_DIM_NUM] = {1};
    uint32_t num_of_dims = 0;
    uint32_t key = 0;
    uint32_t i   = 0;
    vsi_nn_log_param * p     = NULL;

    p = &(self->nn_param.log);

    vsi_nn_OptimizedEltOPShape(inputs[0], sizes, &num_of_dims);

    enable_image_2d = (vsi_bool)(num_of_dims < 3 || (sizes[2] == 1));

    _input_type  = vsi_nn_kernel_map_dtype(inputFormat);
    _output_type = vsi_nn_kernel_map_dtype(outputFormat);

    key = VSI_NN_GEN_LOG_KEY(_input_type, _output_type, enable_image_2d);

    for (i = 0; i < sizeof(log_map) / sizeof(log_map[0]); i++)
    {
        if (key == log_map[i].key)
        {
            p->local.hash_idx = i;
            p->local.execute_on_sw = FALSE;
            return;
        }
    }

    p->local.execute_on_sw = TRUE;
    VSILOGE("Shader unsupport data format! execute on the SW [log]\n");
}

static vsi_bool vx_op_pre_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_kernel_info_t * kernel_info
    )
{
    vsi_nn_log_param * p = NULL;

    p = &(self->nn_param.log);

    kernel_info->kernel_index = log_map[p->local.hash_idx].kernel_index;
    kernel_info->resource_num = 1;
    kernel_info->resource_name[0] = log_map[p->local.hash_idx].resource_name;

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
    vsi_nn_kernel_info_t kernel_info;
    vsi_nn_log_param * p    = NULL;

    memset(&kernel_info, 0x0, sizeof(vsi_nn_kernel_info_t));
    status = VSI_FAILURE;
    if( NULL == self )
    {
        return VSI_FAILURE;
    }
    p = &(self->nn_param.log);

    _get_log_hashtable_idx(self, inputs, outputs);

    if (p->local.execute_on_sw || !vsi_nn_IsEVISFeatureAvaiable(self->graph->ctx))
    {
        kernel_info.resource_num = 1;
        kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
        kernel_info.resource_name[0] = "vsi_nn_kernel_log";
        kernel_info.type = VX_KERNEL_TYPE_CPU;
        kernel_info.kernel = vx_kernel_LOG_list;
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
        kernel_info.kernel = vx_kernel_LOG_list;
        kernel_info.resource_num = 1;
        kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
        kernel_info.init_index = 1;
        kernel_info.resource_name[0] = "vsi_nn_kernel_log";

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
    for (i = 0; i < _VSI_NN_LOG_LOCAL_TENSOR_NUM; i++)
    {
        if (self->nn_param.log.local.local_tensor[i] != NULL)
        {
            vxReleaseTensor(&(self->nn_param.log.local.local_tensor[i]));
            self->nn_param.log.local.local_tensor[i] = NULL;
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
    /* op_name    */ LOG,
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
