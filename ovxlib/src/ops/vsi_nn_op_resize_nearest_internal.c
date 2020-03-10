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
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "client/vsi_nn_vxkernel.h"
#include "kernel/vsi_nn_kernel.h"
#include "utils/vsi_nn_util.h"

#define _ARG_NUM            (2)
#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)

extern vx_kernel_description_t * vx_kernel_RESIZE_NEAREST_INTERNAL_list[];


/* Type enum */
typedef enum _resize_nearest_mode_e
{
    LARGE  = 0,
    SMALL  = 1,
}resize_nearest_mode_e;

#define VSI_NN_GEN_NEAREST_INTERNAL_KEY(_input_type, _output_type, _resize_mode) \
    ((_input_type << 12) | (_output_type << 4) | (_resize_mode))

#define VSI_NN_GEN_NEAREST_INTERNAL_KERNEL_SOURCE_NAME( ) \
    "vsi_nn_kernel_resize_nearest_internal"

#define VSI_NN_GEN_RESIZE_NEAREST_STRUCT_ITEMS(_input_type, _output_type, _resize_mode) \
    VSI_NN_GEN_NEAREST_INTERNAL_KEY(_input_type, _output_type, _resize_mode), \
    VSI_NN_NEAREST_INTERNAL_SH_KERNEL_IDX(_input_type, _output_type, _resize_mode) \
    VSI_NN_GEN_NEAREST_INTERNAL_KERNEL_SOURCE_NAME( )

static struct {
        uint32_t key;
        uint32_t kernel_index;
        char *resource_name;
    } resize_nearest_map[] =
    {
        {VSI_NN_GEN_RESIZE_NEAREST_STRUCT_ITEMS(F16, F16, LARGE)},
        {VSI_NN_GEN_RESIZE_NEAREST_STRUCT_ITEMS(I16, I16, LARGE)},
        {VSI_NN_GEN_RESIZE_NEAREST_STRUCT_ITEMS(I8, I8, LARGE)},
        {VSI_NN_GEN_RESIZE_NEAREST_STRUCT_ITEMS(U8, U8, LARGE)},
        {VSI_NN_GEN_RESIZE_NEAREST_STRUCT_ITEMS(F16, F16, SMALL)},
        {VSI_NN_GEN_RESIZE_NEAREST_STRUCT_ITEMS(I16, I16, SMALL)},
        {VSI_NN_GEN_RESIZE_NEAREST_STRUCT_ITEMS(I8, I8, SMALL)},
        {VSI_NN_GEN_RESIZE_NEAREST_STRUCT_ITEMS(U8, U8, SMALL)},
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

static vsi_status _create_params
    (
    vsi_nn_node_t * node,
    vx_reference * params,
    uint32_t num
    )
{
    vsi_status status;
    vx_context ctx;
    vsi_nn_resize_nearest_internal_param * p = NULL;
    if( 0 == num )
    {
        return VSI_SUCCESS;
    }
    memset( params, 0, sizeof( vx_reference * ) * num );
    p = &(node->nn_param.resize_nearest_internal);
    ctx = vxGetContext( (vx_reference)node->graph->g );
    /* Init parameters */
    #define _SET_PARAM( i, type, arg ) do{ \
        params[i] = (vx_reference)vxCreateScalar( ctx, type, &p->arg ); \
        status = vxGetStatus( params[i] ); \
        if( VSI_SUCCESS != status ) { \
            goto set_param_error; \
            } \
        } while(0)
    _SET_PARAM( 0, VX_TYPE_INT32, align_corners );
    _SET_PARAM( 1, VX_TYPE_INT32, half_pixel_centers );
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
    _set_inputs_outputs( params, inputs, outputs );

    /* Init parameters. */
    _create_params( self, args, _ARG_NUM );

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, _PARAM_NUM );

    _release_params( args, _ARG_NUM );

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
    vx_reference * args = NULL;
    vx_border_t border;

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
    border.mode = VX_BORDER_REPLICATE;
    status |= vxSetNodeAttribute(self->n, VX_NODE_BORDER, &border, sizeof(border));
    return status;
}


static vsi_bool _get_resize_nearest_hashtable_idx
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_type_e inputFormat = inputs[0]->attr.dtype.vx_type;
    vsi_nn_type_e outputFormat  = outputs[0]->attr.dtype.vx_type;
    uint32_t inputWidth = inputs[0]->attr.size[0];
    uint32_t outputWidth = outputs[0]->attr.size[0];
    vsi_nn_kernel_dtype_e _input_type;
    vsi_nn_kernel_dtype_e _output_type;
    uint32_t key;
    resize_nearest_mode_e resize_mode = LARGE;
    uint32_t i = 0;
    vsi_nn_resize_nearest_internal_param *p = NULL;
    vx_float32   scale_factor;
    p = &(self->nn_param.resize_nearest_internal);
    if (NULL == p->lcl_data_ptr)
    {
        VSILOGE("lcl_data_ptr is not malloc\n");
        return FALSE;
    }

    if (VSI_NN_TYPE_BFLOAT16 == inputFormat && VSI_NN_TYPE_BFLOAT16 == outputFormat)
    {
        inputFormat  = VSI_NN_TYPE_FLOAT16;
        outputFormat = VSI_NN_TYPE_FLOAT16;
    }

    _input_type  = vsi_nn_kernel_map_dtype(inputFormat);
    _output_type = vsi_nn_kernel_map_dtype(outputFormat);

    if (p->align_corners && outputWidth > 1)
    {
        scale_factor = (vx_float32)(inputWidth - 1) / (vx_float32)(outputWidth - 1);
    }
    else
    {
        scale_factor = (vx_float32)inputWidth / (vx_float32)outputWidth;
    }

    if (scale_factor < 4.0f)
    {
        resize_mode = SMALL;
    }
    else
    {
        resize_mode = LARGE;
    }

    key = VSI_NN_GEN_NEAREST_INTERNAL_KEY(_input_type, _output_type, resize_mode);
    for (i = 0; i < sizeof(resize_nearest_map) / sizeof(resize_nearest_map[0]); i++)
    {
        if (key == resize_nearest_map[i].key)
        {
            p->lcl_data_ptr->hash_idx = i;
            p->lcl_data_ptr->execute_on_sw = FALSE;
            return TRUE;
        }
    }
    p->lcl_data_ptr->execute_on_sw = TRUE;
    VSILOGE("Shader unsupport data format or axis! execute on the SW [reduceprod]\n");
    return TRUE;
}

static vsi_bool vx_op_pre_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_kernel_info_t * kernel_info
    )
{
    vsi_nn_resize_nearest_internal_param *p = NULL;

    p = &(self->nn_param.resize_nearest_internal);

    if (NULL == p->lcl_data_ptr)
    {
        VSILOGE("lcl_data_ptr is not malloc\n");
        return FALSE;
    }
    kernel_info->kernel_index = resize_nearest_map[p->lcl_data_ptr->hash_idx].kernel_index;
    kernel_info->resource_num = 1;
    kernel_info->resource_name[0] = resize_nearest_map[p->lcl_data_ptr->hash_idx].resource_name;

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
    vsi_nn_resize_nearest_internal_param *p = NULL;

    memset(&kernel_info, 0x0, sizeof(vsi_nn_kernel_info_t));
    status = VSI_FAILURE;
    if( NULL == self )
    {
        return VSI_FAILURE;
    }
    p = &(self->nn_param.resize_nearest_internal);
    if (NULL == p->lcl_data_ptr)
    {
        VSILOGE("lcl_data_ptr is not malloc\n");
        return VSI_FAILURE;
    }
    _get_resize_nearest_hashtable_idx(self, inputs, outputs);


   if (p->lcl_data_ptr->execute_on_sw || !vsi_nn_IsEVISFeatureAvaiable(self->graph->ctx))
    {
        kernel_info.resource_num = 1;
        kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
        kernel_info.resource_name[0] = "vsi_nn_kernel_resize_nearest_internal";
        kernel_info.type = VX_KERNEL_TYPE_CPU;
        kernel_info.kernel = vx_kernel_RESIZE_NEAREST_INTERNAL_list;
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
        kernel_info.kernel = vx_kernel_RESIZE_NEAREST_INTERNAL_list;
        kernel_info.resource_num = 1;
        kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
        kernel_info.init_index = 1;
        kernel_info.resource_name[0] = "vsi_nn_kernel_resize_nearest_internal";

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
    float factor = self->nn_param.resize_nearest_internal.factor;

    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.dim_num = inputs[0]->attr.dim_num;
        if (factor != 0)
        {
            outputs[0]->attr.size[0] = (uint32_t)(inputs[0]->attr.size[0] * factor);
            outputs[0]->attr.size[1] = (uint32_t)(inputs[0]->attr.size[1] * factor);
        }
        else
        {
            outputs[0]->attr.size[0] = self->nn_param.resize.size[0];
            outputs[0]->attr.size[1] = self->nn_param.resize.size[1];
        }
        outputs[0]->attr.size[2] = inputs[0]->attr.size[2];
        outputs[0]->attr.size[3] = inputs[0]->attr.size[3];
    }
    return TRUE;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    if (self->nn_param.resize_nearest_internal.lcl_data_ptr)
    {
        free(self->nn_param.resize_nearest_internal.lcl_data_ptr);
        self->nn_param.resize_nearest_internal.lcl_data_ptr = NULL;
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

    self->nn_param.resize_nearest_internal.lcl_data_ptr   =
    (vsi_nn_resize_nearest_in_lcl_data *)malloc(sizeof(vsi_nn_resize_nearest_in_lcl_data));
    if (NULL == self->nn_param.resize_nearest_internal.lcl_data_ptr)
    {
        return  VX_ERROR_NO_MEMORY;
    }

    memset(self->nn_param.resize_nearest_internal.lcl_data_ptr, 0, sizeof(vsi_nn_resize_nearest_in_lcl_data));

    return status;
} /* op_init() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ RESIZE_NEAREST_INTERNAL,
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
