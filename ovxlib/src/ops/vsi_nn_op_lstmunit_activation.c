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
#define _INPUT_NUM          (ACT_INPUTS_COUNT)
#define _OUTPUT_NUM         (ACT_OUTUTS_COUNT)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)

extern vx_kernel_description_t * vx_kernel_LSTMUNIT_ACTIVATION_list[];

uint32_t _set_inputs_outputs
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
    for( i = 0; i < _INPUT_NUM; i ++)
    {
        if (inputs[i])
        {
            params[cnt] = (vx_reference)inputs[i]->t;
            cnt ++;
        }
    }

    /* Set outputs */
    for( i = 0; i < _OUTPUT_NUM; i ++ )
    {
        if (outputs[i])
        {
            params[cnt] = (vx_reference)outputs[i]->t;
            cnt ++;
        }
    }

    return cnt;
} /* _set_inputs_outputs() */


static void _set_sw_inputs_outputs
    (
    vx_reference * params,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    uint32_t i;
    uint32_t cnt;
    vsi_status status;

    /* Set inputs */
    cnt = 0;
    for( i = 0; i < _INPUT_NUM; i ++, cnt ++)
    {
        if (inputs[i])
        {
            /* Support high precision for inputs */
            if( NULL != inputs[i] && VSI_NN_TYPE_FLOAT32 == inputs[i]->attr.dtype.vx_type )
            {
                status = vsi_nn_SetTensorAttr(inputs[i], VSI_NN_TENSOR_ATTR_HIGH_PRECISION);
                if(VSI_SUCCESS != status)
                {
                    VSILOGE("Set tensor attr of inputs[%d] to high presision fail", i);
                }
            }

            params[cnt] = (vx_reference)inputs[i]->t;
        }
        else
            params[cnt] = NULL;
    }

    /* Set outputs */
    for( i = 0; i < _OUTPUT_NUM; i ++, cnt ++ )
    {
        if (outputs[i])
        {
            status = vsi_nn_SetTensorAttr(outputs[i], VSI_NN_TENSOR_ATTR_HIGH_PRECISION);
            if(VSI_SUCCESS != status)
            {
                VSILOGE("Set tensor attr of outputs[%d] to high presision fail", i);
            }
            params[cnt] = (vx_reference)outputs[i]->t;
        }
        else
            params[cnt] = NULL;
    }

} /* _set_inputs_outputs() */

#if 0
static vsi_status _create_params
    (
    vsi_nn_node_t * node,
    vx_reference * params,
    uint32_t num
    )
{
    vsi_status status;
    vx_context ctx;
    vsi_nn_lstmunit_activation_param * p;
    if( 0 == num )
    {
        return VSI_SUCCESS;
    }
    memset( params, 0, sizeof( vx_reference * ) * num );
    p = &(node->nn_param.lstmunit_activation);
    ctx = vxGetContext( (vx_reference)node->graph->g );
    /* Init parameters */
    #define _SET_PARAM( i, type, arg ) do{ \
        params[i] = (vx_reference)vxCreateScalar( ctx, type, &p->arg ); \
        status = vxGetStatus( params[i] ); \
        if( VSI_SUCCESS != status ) { \
            goto set_param_error; \
            } \
        } while(0)
    _SET_PARAM( 0, VX_TYPE_FLOAT32, forget_bias );
    _SET_PARAM( 1, VX_TYPE_FLOAT32, cell_clip );
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
#endif

static vsi_status cpu_op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_reference params[_PARAM_NUM];
    uint32_t param_num = _IO_NUM;
    vsi_nn_tensor_t * lstmunit_param = NULL;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_lstmunit_activation_param * p;

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    p = &(self->nn_param.lstmunit_activation);

    /* Set inputs and outputs */
    _set_sw_inputs_outputs( params, inputs, outputs );

    memset(&attr, 0, sizeof(attr));
    attr.vtl = FALSE;
    attr.dim_num = 2;
    attr.size[0] = sizeof(vsi_nn_lstmunit_activation_param) / sizeof(uint8_t);
    attr.size[1] = 1;
    attr.dtype.vx_type = VSI_NN_TYPE_UINT8;

    lstmunit_param = vsi_nn_CreateTensor(self->graph, &attr);

    vsi_nn_CopyDataToTensor(self->graph, lstmunit_param, (uint8_t*)p);
    /* Init parameters. */
    params[param_num] = (vx_reference)lstmunit_param->t;

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, param_num + 1 );

    return status;
}

static vsi_status vx_op_pre_compute_layer_norm
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_kernel_info_t * kernel_info
    )
{
    vsi_nn_type_e inputFormat;
    vsi_nn_type_e cellFormat;
    vsi_nn_type_e outputFormat  = outputs[0]->attr.dtype.vx_type;

    vsi_nn_lstmunit_activation_param * p;

    p = &(self->nn_param.lstmunit_activation);

    if (p->is_layer_norm)
    {
        inputFormat   = inputs[ACT_INPUT_FC_F]->attr.dtype.vx_type;
        cellFormat    = inputs[ACT_CSTATE_IN]->attr.dtype.vx_type;

        if (inputFormat != VSI_NN_TYPE_FLOAT16)
        {
            VSILOGE("Not support input data format!(lstm unit activation layernorm)\n");
            return VSI_FAILURE;
        }

        if (p->is_cifg)
        {
            if (p->is_projection)
            {
                if (outputFormat == VSI_NN_TYPE_FLOAT16)
                {
                    if (cellFormat == VSI_NN_TYPE_FLOAT32)
                        kernel_info->kernel_index = 1;
                    else
                        kernel_info->kernel_index = 2;
                }

                kernel_info->resource_name[0] = "vsi_nn_kernel_lstmunit_activation_clp";
            }
            else
            {
                if (cellFormat == VSI_NN_TYPE_FLOAT16)
                {
                    if (outputFormat == VSI_NN_TYPE_FLOAT16)
                        kernel_info->kernel_index = 5;
                    else if (outputFormat == VSI_NN_TYPE_INT16)
                        kernel_info->kernel_index = 6;
                    else if (outputFormat == VSI_NN_TYPE_INT8)
                        kernel_info->kernel_index = 7;
                    else if (outputFormat == VSI_NN_TYPE_UINT8)
                        kernel_info->kernel_index = 8;
                }
                else if (cellFormat == VSI_NN_TYPE_FLOAT32)
                {
                    if (outputFormat == VSI_NN_TYPE_FLOAT16)
                        kernel_info->kernel_index = 9;
                    else if (outputFormat == VSI_NN_TYPE_INT16)
                        kernel_info->kernel_index = 10;
                    else if (outputFormat == VSI_NN_TYPE_INT8)
                        kernel_info->kernel_index = 11;
                    else if (outputFormat == VSI_NN_TYPE_UINT8)
                        kernel_info->kernel_index = 12;
                }

                kernel_info->resource_name[0] = "vsi_nn_kernel_lstmunit_activation_cl";
            }
        }
        else
        {
            if (p->is_projection)
            {
                if (outputFormat == VSI_NN_TYPE_FLOAT16)
                {
                    if (cellFormat == VSI_NN_TYPE_FLOAT32)
                        kernel_info->kernel_index = 3;
                    else
                        kernel_info->kernel_index = 4;
                }

                kernel_info->resource_name[0] = "vsi_nn_kernel_lstmunit_activation_lp";
            }
            else
            {
                if (cellFormat == VSI_NN_TYPE_FLOAT16)
                {
                    if (outputFormat == VSI_NN_TYPE_FLOAT16)
                        kernel_info->kernel_index = 13;
                    else if (outputFormat == VSI_NN_TYPE_INT16)
                        kernel_info->kernel_index = 14;
                    else if (outputFormat == VSI_NN_TYPE_INT8)
                        kernel_info->kernel_index = 15;
                    else if (outputFormat == VSI_NN_TYPE_UINT8)
                        kernel_info->kernel_index = 16;
                }
                else if (cellFormat == VSI_NN_TYPE_FLOAT32)
                {
                    if (outputFormat == VSI_NN_TYPE_FLOAT16)
                        kernel_info->kernel_index = 17;
                    else if (outputFormat == VSI_NN_TYPE_INT16)
                        kernel_info->kernel_index = 18;
                    else if (outputFormat == VSI_NN_TYPE_INT8)
                        kernel_info->kernel_index = 19;
                    else if (outputFormat == VSI_NN_TYPE_UINT8)
                        kernel_info->kernel_index = 20;
                }

                kernel_info->resource_name[0] = "vsi_nn_kernel_lstmunit_activation_l";
            }
        }
    }
    else
    {
        goto OnError;
    }

    return VSI_SUCCESS;
OnError:
        VSILOGE("Not support input or output data format!(lstm unit activation)\n");
        return VSI_FAILURE;
}

static vsi_status vx_op_pre_compute_hybrid_fp16
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_kernel_info_t * kernel_info
    )
{
#define LSTMUNIT_HYBRID_FP16_START   (LSTMUNIT_ACT_LN_KERNEL_COUNTS)
    vsi_nn_type_e cellFormat;
    vsi_nn_type_e outputFormat  = outputs[0]->attr.dtype.vx_type;

    vsi_nn_lstmunit_activation_param * p;

    p = &(self->nn_param.lstmunit_activation);

    cellFormat    = inputs[ACT_CSTATE_IN]->attr.dtype.vx_type;

    if (!p->is_cifg)
    {
        if (p->is_projection)
        {
            if (outputFormat == VSI_NN_TYPE_FLOAT16)
            {
                if (cellFormat == VSI_NN_TYPE_FLOAT32)
                    kernel_info->kernel_index = LSTMUNIT_HYBRID_FP16_START;
                else
                    kernel_info->kernel_index = LSTMUNIT_HYBRID_FP16_START + 1;

                kernel_info->resource_name[0] = "vsi_nn_kernel_lstmunit_activation_bp_fp16";
            }
            else
                goto OnError;
        }
        else
        {
            if (cellFormat == VSI_NN_TYPE_FLOAT16)
            {
                if (outputFormat == VSI_NN_TYPE_FLOAT16)
                    kernel_info->kernel_index = LSTMUNIT_HYBRID_FP16_START + 2;
                else if (outputFormat == VSI_NN_TYPE_INT16)
                    kernel_info->kernel_index = LSTMUNIT_HYBRID_FP16_START + 3;
                else if (outputFormat == VSI_NN_TYPE_INT8)
                    kernel_info->kernel_index = LSTMUNIT_HYBRID_FP16_START + 4;
                else if (outputFormat == VSI_NN_TYPE_UINT8)
                    kernel_info->kernel_index = LSTMUNIT_HYBRID_FP16_START + 5;
            }
            else if (cellFormat == VSI_NN_TYPE_FLOAT32)
            {
                if (outputFormat == VSI_NN_TYPE_FLOAT16)
                    kernel_info->kernel_index = LSTMUNIT_HYBRID_FP16_START + 6;
                else if (outputFormat == VSI_NN_TYPE_INT16)
                    kernel_info->kernel_index = LSTMUNIT_HYBRID_FP16_START + 7;
                else if (outputFormat == VSI_NN_TYPE_INT8)
                    kernel_info->kernel_index = LSTMUNIT_HYBRID_FP16_START + 8;
                else if (outputFormat == VSI_NN_TYPE_UINT8)
                    kernel_info->kernel_index = LSTMUNIT_HYBRID_FP16_START + 9;
            }

            kernel_info->resource_name[0] = "vsi_nn_kernel_lstmunit_activation_b_fp16";
        }
    }
    else if (p->is_cifg)
    {
        if (p->is_projection)
        {
            if (outputFormat == VSI_NN_TYPE_FLOAT16)
            {
                if (cellFormat == VSI_NN_TYPE_FLOAT32)
                    kernel_info->kernel_index = LSTMUNIT_HYBRID_FP16_START + 10;
                else
                    kernel_info->kernel_index = LSTMUNIT_HYBRID_FP16_START + 11;

                kernel_info->resource_name[0] = "vsi_nn_kernel_lstmunit_activation_cbp_fp16";
            }
            else
                goto OnError;
        }
        else
        {
            if (cellFormat == VSI_NN_TYPE_FLOAT16)
            {
                if (outputFormat == VSI_NN_TYPE_FLOAT16)
                    kernel_info->kernel_index = LSTMUNIT_HYBRID_FP16_START + 12;
                else if (outputFormat == VSI_NN_TYPE_INT16)
                    kernel_info->kernel_index = LSTMUNIT_HYBRID_FP16_START + 13;
                else if (outputFormat == VSI_NN_TYPE_INT8)
                    kernel_info->kernel_index = LSTMUNIT_HYBRID_FP16_START + 14;
                else if (outputFormat == VSI_NN_TYPE_UINT8)
                    kernel_info->kernel_index = LSTMUNIT_HYBRID_FP16_START + 15;
            }
            else if (cellFormat == VSI_NN_TYPE_FLOAT32)
            {
                if (outputFormat == VSI_NN_TYPE_FLOAT16)
                    kernel_info->kernel_index = LSTMUNIT_HYBRID_FP16_START + 16;
                else if (outputFormat == VSI_NN_TYPE_INT16)
                    kernel_info->kernel_index = LSTMUNIT_HYBRID_FP16_START + 17;
                else if (outputFormat == VSI_NN_TYPE_INT8)
                    kernel_info->kernel_index = LSTMUNIT_HYBRID_FP16_START + 18;
                else if (outputFormat == VSI_NN_TYPE_UINT8)
                    kernel_info->kernel_index = LSTMUNIT_HYBRID_FP16_START + 19;
            }

            kernel_info->resource_name[0] = "vsi_nn_kernel_lstmunit_activation_cb_fp16";
        }
    }
    else
    {
        goto OnError;
    }

    return VSI_SUCCESS;
OnError:
        VSILOGE("Not support input or output data format!(lstm unit activation)\n");
        return VSI_FAILURE;
}

static vsi_status vx_op_pre_compute_hybrid_u8
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_kernel_info_t * kernel_info
    )
{
#define LSTMUNIT_HYBRID_U8_START  (LSTMUNIT_HYBRID_FP16_START + LSTMUNIT_ACT_HYBRID_KERNEL_FP16_COUNTS)
    vsi_nn_type_e cellFormat;
    vsi_nn_type_e outputFormat  = outputs[0]->attr.dtype.vx_type;

    vsi_nn_lstmunit_activation_param * p;

    p = &(self->nn_param.lstmunit_activation);

    cellFormat    = inputs[ACT_CSTATE_IN]->attr.dtype.vx_type;

    if (!p->is_cifg)
    {
        if (p->is_projection)
        {
            if (outputFormat == VSI_NN_TYPE_FLOAT16)
            {
                if (cellFormat == VSI_NN_TYPE_FLOAT32)
                    kernel_info->kernel_index = LSTMUNIT_HYBRID_U8_START;
                else
                    kernel_info->kernel_index = LSTMUNIT_HYBRID_U8_START + 1;

                kernel_info->resource_name[0] = "vsi_nn_kernel_lstmunit_activation_bp_u8";
            }
            else
                goto OnError;
        }
        else
        {
            if (cellFormat == VSI_NN_TYPE_FLOAT16)
            {
                if (outputFormat == VSI_NN_TYPE_FLOAT16)
                    kernel_info->kernel_index = LSTMUNIT_HYBRID_U8_START + 2;
                else if (outputFormat == VSI_NN_TYPE_UINT8)
                    kernel_info->kernel_index = LSTMUNIT_HYBRID_U8_START + 3;
            }
            else if (cellFormat == VSI_NN_TYPE_FLOAT32)
            {
                if (outputFormat == VSI_NN_TYPE_FLOAT16)
                    kernel_info->kernel_index = LSTMUNIT_HYBRID_U8_START + 4;
                else if (outputFormat == VSI_NN_TYPE_UINT8)
                    kernel_info->kernel_index = LSTMUNIT_HYBRID_U8_START + 5;
            }

            kernel_info->resource_name[0] = "vsi_nn_kernel_lstmunit_activation_b_u8";
        }
    }
    else if (p->is_cifg)
    {
        if (p->is_projection)
        {
            if (outputFormat == VSI_NN_TYPE_FLOAT16)
            {
                if (cellFormat == VSI_NN_TYPE_FLOAT32)
                    kernel_info->kernel_index = LSTMUNIT_HYBRID_U8_START + 6;
                else
                    kernel_info->kernel_index = LSTMUNIT_HYBRID_U8_START + 7;

                kernel_info->resource_name[0] = "vsi_nn_kernel_lstmunit_activation_cbp_u8";
            }
            else
                goto OnError;
        }
        else
        {
            if (cellFormat == VSI_NN_TYPE_FLOAT16)
            {
                if (outputFormat == VSI_NN_TYPE_FLOAT16)
                    kernel_info->kernel_index = LSTMUNIT_HYBRID_U8_START + 8;
                else if (outputFormat == VSI_NN_TYPE_UINT8)
                    kernel_info->kernel_index = LSTMUNIT_HYBRID_U8_START + 9;
            }
            else if (cellFormat == VSI_NN_TYPE_FLOAT32)
            {
                if (outputFormat == VSI_NN_TYPE_FLOAT16)
                    kernel_info->kernel_index = LSTMUNIT_HYBRID_U8_START + 10;
                else if (outputFormat == VSI_NN_TYPE_UINT8)
                    kernel_info->kernel_index = LSTMUNIT_HYBRID_U8_START + 11;
            }

            kernel_info->resource_name[0] = "vsi_nn_kernel_lstmunit_activation_cb_u8";
        }
    }
    else
    {
        goto OnError;
    }

    return VSI_SUCCESS;
OnError:
        VSILOGE("Not support input or output data format!(lstm unit activation)\n");
        return VSI_FAILURE;
}


static vsi_status vx_op_pre_compute_hybrid
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_kernel_info_t * kernel_info
    )
{

    vsi_nn_type_e inputFormat;

    inputFormat   = inputs[ACT_INPUT_FC_F]->attr.dtype.vx_type;

    if (inputFormat == VSI_NN_TYPE_FLOAT16)
    {
        return vx_op_pre_compute_hybrid_fp16(self, inputs, outputs, kernel_info);
    }
    else if (inputFormat == VSI_NN_TYPE_UINT8)
    {
        return vx_op_pre_compute_hybrid_u8(self, inputs, outputs, kernel_info);
    }
    else
    {
        goto OnError;
    }

    return VSI_SUCCESS;
OnError:
        VSILOGE("Not support input or output data format!(lstm unit activation)\n");
        return VSI_FAILURE;
}

static vsi_status vx_op_pre_compute_standard_fp16
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_kernel_info_t * kernel_info
    )
{
#define LSTMUNIT_ST_FP16_START   (LSTMUNIT_ACT_LN_KERNEL_COUNTS + LSTMUNIT_ACT_HYBRID_KERNEL_COUNTS)
    vsi_nn_type_e cellFormat;
    vsi_nn_type_e outputFormat  = outputs[0]->attr.dtype.vx_type;

    vsi_nn_lstmunit_activation_param * p;

    p = &(self->nn_param.lstmunit_activation);

    cellFormat    = inputs[ACT_CSTATE_IN]->attr.dtype.vx_type;

    if (!p->is_cifg)
    {
        if (p->is_projection)
        {
            if (outputFormat == VSI_NN_TYPE_FLOAT16)
            {
                if (cellFormat == VSI_NN_TYPE_FLOAT32)
                    kernel_info->kernel_index = LSTMUNIT_ST_FP16_START;
                else
                    kernel_info->kernel_index = LSTMUNIT_ST_FP16_START + 1;

                kernel_info->resource_name[0] = "vsi_nn_kernel_lstmunit_activation_sp_fp16";
            }
            else
                goto OnError;
        }
        else
        {
            if (cellFormat == VSI_NN_TYPE_FLOAT16)
            {
                if (outputFormat == VSI_NN_TYPE_FLOAT16)
                    kernel_info->kernel_index = LSTMUNIT_ST_FP16_START + 2;
                else if (outputFormat == VSI_NN_TYPE_INT16)
                    kernel_info->kernel_index = LSTMUNIT_ST_FP16_START + 3;
                else if (outputFormat == VSI_NN_TYPE_INT8)
                    kernel_info->kernel_index = LSTMUNIT_ST_FP16_START + 4;
                else if (outputFormat == VSI_NN_TYPE_UINT8)
                    kernel_info->kernel_index = LSTMUNIT_ST_FP16_START + 5;
            }
            else if (cellFormat == VSI_NN_TYPE_FLOAT32)
            {
                if (outputFormat == VSI_NN_TYPE_FLOAT16)
                    kernel_info->kernel_index = LSTMUNIT_ST_FP16_START + 6;
                else if (outputFormat == VSI_NN_TYPE_INT16)
                    kernel_info->kernel_index = LSTMUNIT_ST_FP16_START + 7;
                else if (outputFormat == VSI_NN_TYPE_INT8)
                    kernel_info->kernel_index = LSTMUNIT_ST_FP16_START + 8;
                else if (outputFormat == VSI_NN_TYPE_UINT8)
                    kernel_info->kernel_index = LSTMUNIT_ST_FP16_START + 9;
            }

            kernel_info->resource_name[0] = "vsi_nn_kernel_lstmunit_activation_s_fp16";
        }
    }
    else if (p->is_cifg)
    {
        if (p->is_projection)
        {
            if (outputFormat == VSI_NN_TYPE_FLOAT16)
            {
                if (cellFormat == VSI_NN_TYPE_FLOAT32)
                    kernel_info->kernel_index = LSTMUNIT_ST_FP16_START + 10;
                else
                    kernel_info->kernel_index = LSTMUNIT_ST_FP16_START + 11;

                kernel_info->resource_name[0] = "vsi_nn_kernel_lstmunit_activation_csp_fp16";
            }
            else
                goto OnError;
        }
        else
        {
            if (cellFormat == VSI_NN_TYPE_FLOAT16)
            {
                if (outputFormat == VSI_NN_TYPE_FLOAT16)
                    kernel_info->kernel_index = LSTMUNIT_ST_FP16_START + 12;
                else if (outputFormat == VSI_NN_TYPE_INT16)
                    kernel_info->kernel_index = LSTMUNIT_ST_FP16_START + 13;
                else if (outputFormat == VSI_NN_TYPE_INT8)
                    kernel_info->kernel_index = LSTMUNIT_ST_FP16_START + 14;
                else if (outputFormat == VSI_NN_TYPE_UINT8)
                    kernel_info->kernel_index = LSTMUNIT_ST_FP16_START + 15;
            }
            else if (cellFormat == VSI_NN_TYPE_FLOAT32)
            {
                if (outputFormat == VSI_NN_TYPE_FLOAT16)
                    kernel_info->kernel_index = LSTMUNIT_ST_FP16_START + 16;
                else if (outputFormat == VSI_NN_TYPE_INT16)
                    kernel_info->kernel_index = LSTMUNIT_ST_FP16_START + 17;
                else if (outputFormat == VSI_NN_TYPE_INT8)
                    kernel_info->kernel_index = LSTMUNIT_ST_FP16_START + 18;
                else if (outputFormat == VSI_NN_TYPE_UINT8)
                    kernel_info->kernel_index = LSTMUNIT_ST_FP16_START + 19;
            }

            kernel_info->resource_name[0] = "vsi_nn_kernel_lstmunit_activation_cs_fp16";
        }
    }
    else
    {
        goto OnError;
    }

    return VSI_SUCCESS;
OnError:
        VSILOGE("Not support input or output data format!(lstm unit activation)\n");
        return VSI_FAILURE;
}

static vsi_status vx_op_pre_compute_standard_u8
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_kernel_info_t * kernel_info
    )
{
#define LSTMUNIT_ST_U8_START     (LSTMUNIT_ST_FP16_START + LSTMUNIT_ACT_ST_KERNEL_FP16_COUNTS)
    vsi_nn_type_e cellFormat;
    vsi_nn_type_e outputFormat  = outputs[0]->attr.dtype.vx_type;

    vsi_nn_lstmunit_activation_param * p;

    p = &(self->nn_param.lstmunit_activation);

    cellFormat    = inputs[ACT_CSTATE_IN]->attr.dtype.vx_type;

    if (!p->is_cifg)
    {
        if (p->is_projection)
        {
            if (outputFormat == VSI_NN_TYPE_FLOAT16)
            {
                if (cellFormat == VSI_NN_TYPE_FLOAT32)
                    kernel_info->kernel_index = LSTMUNIT_ST_U8_START;
                else
                    kernel_info->kernel_index = LSTMUNIT_ST_U8_START + 1;

                kernel_info->resource_name[0] = "vsi_nn_kernel_lstmunit_activation_sp_u8";
            }
            else
                goto OnError;
        }
        else
        {
            if (cellFormat == VSI_NN_TYPE_FLOAT16)
            {
                if (outputFormat == VSI_NN_TYPE_FLOAT16)
                    kernel_info->kernel_index = LSTMUNIT_ST_U8_START + 2;
                else if (outputFormat == VSI_NN_TYPE_UINT8)
                    kernel_info->kernel_index = LSTMUNIT_ST_U8_START + 3;
            }
            else if (cellFormat == VSI_NN_TYPE_FLOAT32)
            {
                if (outputFormat == VSI_NN_TYPE_FLOAT16)
                    kernel_info->kernel_index = LSTMUNIT_ST_U8_START + 4;
                else if (outputFormat == VSI_NN_TYPE_UINT8)
                    kernel_info->kernel_index = LSTMUNIT_ST_U8_START + 5;
            }

            kernel_info->resource_name[0] = "vsi_nn_kernel_lstmunit_activation_s_u8";
        }
    }
    else if (p->is_cifg)
    {
        if (p->is_projection)
        {
            if (outputFormat == VSI_NN_TYPE_FLOAT16)
            {
                if (cellFormat == VSI_NN_TYPE_FLOAT32)
                    kernel_info->kernel_index = LSTMUNIT_ST_U8_START + 6;
                else
                    kernel_info->kernel_index = LSTMUNIT_ST_U8_START + 7;

                kernel_info->resource_name[0] = "vsi_nn_kernel_lstmunit_activation_csp_u8";
            }
            else
                goto OnError;
        }
        else
        {
            if (cellFormat == VSI_NN_TYPE_FLOAT16)
            {
                if (outputFormat == VSI_NN_TYPE_FLOAT16)
                    kernel_info->kernel_index = LSTMUNIT_ST_U8_START + 8;
                else if (outputFormat == VSI_NN_TYPE_UINT8)
                    kernel_info->kernel_index = LSTMUNIT_ST_U8_START + 9;
            }
            else if (cellFormat == VSI_NN_TYPE_FLOAT32)
            {
                if (outputFormat == VSI_NN_TYPE_FLOAT16)
                    kernel_info->kernel_index = LSTMUNIT_ST_U8_START + 10;
                else if (outputFormat == VSI_NN_TYPE_UINT8)
                    kernel_info->kernel_index = LSTMUNIT_ST_U8_START + 11;
            }

            kernel_info->resource_name[0] = "vsi_nn_kernel_lstmunit_activation_cs_u8";
        }
    }
    else
    {
        goto OnError;
    }

    return VSI_SUCCESS;
OnError:
        VSILOGE("Not support input or output data format!(lstm unit activation)\n");
        return VSI_FAILURE;
}


static vsi_status vx_op_pre_compute_standard
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_kernel_info_t * kernel_info
    )
{
    vsi_nn_type_e inputFormat;

    inputFormat   = inputs[ACT_INPUT_FC_F]->attr.dtype.vx_type;

    if (inputFormat == VSI_NN_TYPE_FLOAT16)
    {
        return vx_op_pre_compute_standard_fp16(self, inputs, outputs, kernel_info);
    }
    else if (inputFormat == VSI_NN_TYPE_UINT8)
    {
        return vx_op_pre_compute_standard_u8(self, inputs, outputs, kernel_info);
    }
    else
    {
        goto OnError;
    }

    return VSI_SUCCESS;
OnError:
        VSILOGE("Not support input or output data format!(lstm unit activation)\n");
        return VSI_FAILURE;
}

static vsi_status vx_op_pre_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_kernel_info_t * kernel_info
    )
{
    vsi_nn_lstmunit_activation_param * p;

    p = &(self->nn_param.lstmunit_activation);

    if (p->is_layer_norm)
    {
        return vx_op_pre_compute_layer_norm(self, inputs, outputs, kernel_info);
    }
    else if (p->is_hybrid)
    {
        return vx_op_pre_compute_hybrid(self, inputs, outputs, kernel_info);
    }
    else if (!p->is_layer_norm && !p->is_hybrid && !p->is_peephole)
    {
        vx_op_pre_compute_standard(self, inputs, outputs, kernel_info);
    }
    else
        goto OnError;

    return VSI_SUCCESS;
OnError:
        VSILOGE("Not support this feature or format!(lstm unit activation)\n");
        return VSI_FAILURE;
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
    vsi_nn_tensor_t * lstmunit_param = NULL;
    vsi_nn_tensor_attr_t attr;
    uint32_t param_num = 0;
    vsi_nn_lstmunit_activation_param * p;

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    p = &(self->nn_param.lstmunit_activation);

    /* Set inputs and outputs */
    param_num = _set_inputs_outputs( params, inputs, outputs );
    /*TODO: Add code if need to change your parameter*/

    memset(&attr, 0, sizeof(attr));
    attr.vtl = FALSE;
    attr.dim_num = 2;
    attr.size[0] = sizeof(vsi_nn_lstmunit_activation_param) / sizeof(uint8_t);
    attr.size[1] = 1;
    attr.dtype.vx_type = VSI_NN_TYPE_UINT8;

    lstmunit_param = vsi_nn_CreateTensor(self->graph, &attr);

    vsi_nn_CopyDataToTensor(self->graph, lstmunit_param, (uint8_t*)p);
    /* Init parameters. */
    //_create_params( self, args, _ARG_NUM );
    params[param_num] = (vx_reference)lstmunit_param->t;

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, param_num + 1 );

    //_release_params( args, _ARG_NUM );

    border.mode = VX_BORDER_REPLICATE;
    status |= vxSetNodeAttribute(self->n, VX_NODE_BORDER, &border, sizeof(border));

    if (lstmunit_param) vsi_nn_ReleaseTensor(&lstmunit_param);

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
    vsi_nn_type_e srcFormat    = inputs[1]->attr.dtype.vx_type;
    vsi_nn_lstmunit_activation_param * p;

    status = VSI_FAILURE;
    if( NULL == self )
    {
        return VSI_FAILURE;
    }

    p = &(self->nn_param.lstmunit_activation);

    if (srcFormat != VSI_NN_TYPE_FLOAT16 && p->is_layer_norm)
    {
        kernel_info.resource_num = 1;
        kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
        kernel_info.resource_name[0] = "vsi_nn_kernel_lstmunit_activation";
        kernel_info.type = VX_KERNEL_TYPE_CPU;
        kernel_info.kernel = vx_kernel_LSTMUNIT_ACTIVATION_list;
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
        kernel_info.kernel = vx_kernel_LSTMUNIT_ACTIVATION_list;
        kernel_info.resource_num = 1;
        kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
        kernel_info.init_index = 1;
        kernel_info.resource_name[0] = "vsi_nn_kernel_lstmunit_activation_clp";

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
    vsi_nn_node_t * node,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    /* TODO: Add code to comput outputs' shape. */
    if( VSI_NN_DIM_AUTO == outputs[ACT_OUTPUT]->attr.dim_num )
    {
        outputs[ACT_OUTPUT]->attr.dim_num = inputs[ACT_INPUT_FC_F]->attr.dim_num;
        outputs[ACT_OUTPUT]->attr.size[0] = inputs[ACT_INPUT_FC_F]->attr.size[0];
        outputs[ACT_OUTPUT]->attr.size[1] = inputs[ACT_INPUT_FC_F]->attr.size[1];
        outputs[ACT_OUTPUT]->attr.size[2] = inputs[ACT_INPUT_FC_F]->attr.size[2];
        outputs[ACT_OUTPUT]->attr.size[3] = inputs[ACT_INPUT_FC_F]->attr.size[3];
    }

    if( VSI_NN_DIM_AUTO == outputs[ACT_CSTATE_OUT]->attr.dim_num )
    {
        outputs[ACT_CSTATE_OUT]->attr.dim_num = inputs[ACT_CSTATE_IN]->attr.dim_num;
        outputs[ACT_CSTATE_OUT]->attr.size[0] = inputs[ACT_CSTATE_IN]->attr.size[0];
        outputs[ACT_CSTATE_OUT]->attr.size[1] = inputs[ACT_CSTATE_IN]->attr.size[1];
        outputs[ACT_CSTATE_OUT]->attr.size[2] = inputs[ACT_CSTATE_IN]->attr.size[2];
        outputs[ACT_CSTATE_OUT]->attr.size[3] = inputs[ACT_CSTATE_IN]->attr.size[3];
    }

    if (outputs[ACT_HSTATE_OUT] && VSI_NN_DIM_AUTO == outputs[ACT_HSTATE_OUT]->attr.dim_num )
    {
        outputs[ACT_HSTATE_OUT]->attr.dim_num = outputs[ACT_OUTPUT]->attr.dim_num;
        outputs[ACT_HSTATE_OUT]->attr.size[0] = outputs[ACT_OUTPUT]->attr.size[0];
        outputs[ACT_HSTATE_OUT]->attr.size[1] = outputs[ACT_OUTPUT]->attr.size[1];
        outputs[ACT_HSTATE_OUT]->attr.size[2] = outputs[ACT_OUTPUT]->attr.size[2];
        outputs[ACT_HSTATE_OUT]->attr.size[3] = outputs[ACT_OUTPUT]->attr.size[3];
    }

    return TRUE;
} /* op_setup() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ LSTMUNIT_ACTIVATION,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );
#ifdef __cplusplus
}
#endif
