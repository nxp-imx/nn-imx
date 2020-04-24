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
#include "kernel/vsi_nn_kernel.h"
#include "vsi_nn_internal_node.h"

/* Type enum */
typedef enum _clip_nn_image_dims_e
{
    IMAGE_2D  = TRUE,
    IMAGE     = FALSE,
}clip_nn_image_dims_e;

#define VSI_NN_GEN_CLIP_KEY(_input_type, _output_type, _image_2d) \
    ((_input_type << 12) | (_output_type << 4) | (_image_2d))

#define VSI_NN_GEN_CLIP_KERNEL_SOURCE_NAME(_suffix) \
    "vsi_nn_kernel_clip_"#_suffix

#define VSI_NN_GEN_CLIP_STRUCT_ITEMS(_input_type, _output_type, _image_2d) \
    VSI_NN_GEN_CLIP_KEY(_input_type, _output_type, _image_2d), \
    VSI_NN_CLIP_SH_KERNEL_IDX(_input_type, _output_type, _image_2d) \
    VSI_NN_GEN_CLIP_KERNEL_SOURCE_NAME(_input_type)


static struct {
        uint32_t key;
        uint32_t kernel_index;
        char *resource_name;
    } clip_map[] =
    {
        {VSI_NN_GEN_CLIP_STRUCT_ITEMS(F16,  F16,  IMAGE)},
        {VSI_NN_GEN_CLIP_STRUCT_ITEMS(F16,  I16,  IMAGE)},
        {VSI_NN_GEN_CLIP_STRUCT_ITEMS(F16,  I8,   IMAGE)},
        {VSI_NN_GEN_CLIP_STRUCT_ITEMS(F16,  U8,   IMAGE)},
        {VSI_NN_GEN_CLIP_STRUCT_ITEMS(I16,  F16,  IMAGE)},
        {VSI_NN_GEN_CLIP_STRUCT_ITEMS(I8,   F16,  IMAGE)},
        {VSI_NN_GEN_CLIP_STRUCT_ITEMS(U8,   F16,  IMAGE)},
        {VSI_NN_GEN_CLIP_STRUCT_ITEMS(I16,  I16,  IMAGE)},
        {VSI_NN_GEN_CLIP_STRUCT_ITEMS(I8,   I8,   IMAGE)},
        {VSI_NN_GEN_CLIP_STRUCT_ITEMS(U8,   U8,   IMAGE)},
        {VSI_NN_GEN_CLIP_STRUCT_ITEMS(F16,  F16,  IMAGE_2D)},
        {VSI_NN_GEN_CLIP_STRUCT_ITEMS(F16,  I16,  IMAGE_2D)},
        {VSI_NN_GEN_CLIP_STRUCT_ITEMS(F16,  I8,   IMAGE_2D)},
        {VSI_NN_GEN_CLIP_STRUCT_ITEMS(F16,  U8,   IMAGE_2D)},
        {VSI_NN_GEN_CLIP_STRUCT_ITEMS(I16,  F16,  IMAGE_2D)},
        {VSI_NN_GEN_CLIP_STRUCT_ITEMS(I8,   F16,  IMAGE_2D)},
        {VSI_NN_GEN_CLIP_STRUCT_ITEMS(U8,   F16,  IMAGE_2D)},
        {VSI_NN_GEN_CLIP_STRUCT_ITEMS(I16,  I16,  IMAGE_2D)},
        {VSI_NN_GEN_CLIP_STRUCT_ITEMS(I8,   I8,   IMAGE_2D)},
        {VSI_NN_GEN_CLIP_STRUCT_ITEMS(U8,   U8,   IMAGE_2D)},
    };


#define _ARG_NUM            (2)
#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)
#define IMG_MAX_WIDTH       65536

extern vx_kernel_description_t * vx_kernel_CLIP_list[];

static void reshape_tensor_shape
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vx_reference * params,
    uint32_t index,
    uint32_t *sizes,
    uint32_t  dims
    )
{
    self->nn_param.clip.local.local_tensor[index] =
         vxReshapeTensor(input->t, (int32_t *)sizes, dims);

    params[index] = (vx_reference)self->nn_param.clip.local.local_tensor[index];
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
    vsi_nn_clip_param * p;
    if( 0 == num )
    {
        return VSI_SUCCESS;
    }
    memset( params, 0, sizeof( vx_reference * ) * num );
    p = &(node->nn_param.clip);
    ctx = vxGetContext( (vx_reference)node->graph->g );
    /* Init parameters */
    #define _SET_PARAM( i, type, arg ) do{ \
        params[i] = (vx_reference)vxCreateScalar( ctx, type, &p->arg ); \
        status = vxGetStatus( params[i] ); \
        if( VSI_SUCCESS != status ) { \
            goto set_param_error; \
            } \
        } while(0)
    _SET_PARAM( 0, VX_TYPE_FLOAT32, min );
    _SET_PARAM( 1, VX_TYPE_FLOAT32, max );
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

static void _get_clip_hashtable_idx
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
    uint32_t key = 0;
    vsi_bool is_2d_image = FALSE;
    uint32_t i = 0;
    vsi_nn_clip_param * p = NULL;

    p = &(self->nn_param.clip);
    _input_type  = vsi_nn_kernel_map_dtype(inputFormat);
    _output_type = vsi_nn_kernel_map_dtype(outputFormat);
    if (BF16 == _input_type && BF16 == _output_type)
    {
        _input_type  = F16;
        _output_type = F16;
    }
    is_2d_image  = p->local2->enable_image_2d;
    key = VSI_NN_GEN_CLIP_KEY(_input_type, _output_type, is_2d_image);

    for (i = 0; i < sizeof(clip_map) / sizeof(clip_map[0]); i++)
    {
        if (key == clip_map[i].key)
        {
            p->local2->hash_idx = i;
            p->local2->execute_on_sw = FALSE;
            return;
        }
    }

    p->local2->execute_on_sw = TRUE;
    VSILOGE("Shader unsupport data format or axis! execute on the SW [clip]\n");
}

static vsi_status vx_op_pre_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_kernel_info_t * kernel_info
    )
{
    vsi_nn_clip_param * p = NULL;

    p = &(self->nn_param.clip);

    kernel_info->kernel_index = clip_map[p->local2->hash_idx].kernel_index;
    kernel_info->resource_num = 2;
    kernel_info->resource_name[0] = "vsi_nn_kernel_header";
    kernel_info->resource_name[1] = clip_map[p->local2->hash_idx].resource_name;

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
    vx_reference * args = NULL;
    vx_border_t border;
    vsi_nn_clip_param * p = NULL;

    border.mode = VX_BORDER_REPLICATE;
    border.constant_value.U32 = 0;
    args = &params[_IO_NUM];
    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    /* Set inputs and outputs */
    //_set_inputs_outputs( params, inputs, outputs );
    p = &(self->nn_param.clip);
    reshape_tensor_shape(self, inputs[0],  params, 0, p->local2->sizes0, p->local2->dim_num);
    reshape_tensor_shape(self, outputs[0], params, 1, p->local2->sizes1, p->local2->dim_num);

    /* Init parameters. */
    _create_params( self, args, _ARG_NUM );

    /* Pass parameters to node. */
    status  = vsi_nn_ClientNodePassParameters( self->n, params, _PARAM_NUM );

    _release_params( args, _ARG_NUM );

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
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_info_t kernel_info;
    vsi_nn_clip_param * p = NULL;
    float min = self->nn_param.clip.min;
    float max = self->nn_param.clip.max;

    if ( (min == -1.0f && max == 1.0f)
      || (min == 0.0f && max == 6.0f) )
    {
        status = VSI_SUCCESS;
        vsi_nn_internal_compute_node( self );
    }
    else
    {
        memset(&kernel_info, 0x0, sizeof(vsi_nn_kernel_info_t));
        if( NULL == self )
        {
            return VSI_FAILURE;
        }

        p = &(self->nn_param.clip);

        vsi_nn_OptimizedEltOPShape(inputs[0],  p->local2->sizes0, &(p->local2->dim_num));
        vsi_nn_OptimizedEltOPShape(outputs[0], p->local2->sizes1, &(p->local2->dim_num));

        if (p->local2->dim_num < 3)
        {
            p->local2->enable_image_2d = vx_true_e;
        }
        else
        {
            p->local2->enable_image_2d = vx_false_e;
        }

        _get_clip_hashtable_idx(self, inputs, outputs);

        if (p->local2->execute_on_sw || !vsi_nn_IsEVISFeatureAvaiable(self->graph->ctx))
        {
            kernel_info.resource_num = 1;
            kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
            kernel_info.resource_name[0] = "vsi_nn_kernel_clip_sw";
            kernel_info.type = VX_KERNEL_TYPE_CPU;
            kernel_info.kernel = vx_kernel_CLIP_list;
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
            kernel_info.kernel = vx_kernel_CLIP_list;
            kernel_info.resource_num = 2;
            kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
            kernel_info.init_index = 1;
            kernel_info.resource_name[0] = "vsi_nn_kernel_header";
            kernel_info.resource_name[1] = "vsi_nn_kernel_clip";

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
    float min = self->nn_param.clip.min;
    float max = self->nn_param.clip.max;

    for (i = 0; i < _VSI_NN_CLIP_LOCAL_TENSOR_NUM; i++)
    {
        if (self->nn_param.clip.local.local_tensor[i] != NULL)
        {
            vxReleaseTensor(&(self->nn_param.clip.local.local_tensor[i]));
            self->nn_param.clip.local.local_tensor[i] = NULL;
        }
    }

    if (self->nn_param.clip.local2 != NULL)
    {
        free(self->nn_param.clip.local2);
        self->nn_param.clip.local2 = NULL;
    }

    if ( (min == -1.0f && max == 1.0f)
      || (min == 0.0f && max == 6.0f) )
    {
        vsi_nn_internal_deinit_node_wksp( self );
    }

    vsi_nn_op_common_deinit(self);

    return VSI_SUCCESS;
} /* op_deinit() */

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;
    self->nn_param.clip.local2   =
    (vsi_nn_clip_lcl2_data *)malloc(sizeof(vsi_nn_clip_lcl2_data));
    if (NULL == self->nn_param.reduce.local2)
    {
        return  VX_ERROR_NO_MEMORY;
    }
    memset(self->nn_param.clip.local2, 0, sizeof(vsi_nn_clip_lcl2_data));
    return status;
} /* op_init() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = TRUE;
    vsi_nn_internal_node_t* curr = NULL;
    float min = self->nn_param.clip.min;
    float max = self->nn_param.clip.max;

    if ( (min == -1.0f && max == 1.0f)
      || (min == 0.0f && max == 6.0f) )
    {
        vsi_nn_internal_init_node_wksp(self);
        if (min == -1.0f && max == 1.0f)
        {
            curr = vsi_nn_internal_new_node(self, VSI_NN_OP_RELU1, 0, 0);
        }
        else
        {
            curr = vsi_nn_internal_new_node(self, VSI_NN_OP_RELU6, 0, 0);
        }
        curr->inputs[0] = inputs[0];
        curr->outputs[0] = outputs[0];

        vsi_nn_internal_setup_node(self, curr);
    }
    else
    {
        ret = vsi_nn_op_common_setup(self, inputs, outputs);
    }
    return ret;
} /* op_init() */


#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ CLIP,
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
