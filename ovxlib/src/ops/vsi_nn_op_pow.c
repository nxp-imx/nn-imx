/****************************************************************************
*
*    Copyright (c) 2018 Vivante Corporation
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

#define _ARG_NUM            (0)
#define _INPUT_NUM          (2)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)

#define TENSOR_ALL 0

extern vx_kernel_description_t * vx_kernel_POW_list[];

#define VSI_NN_GEN_POW_KEY(_input0_type, _input1_type, _output_type) \
    ((_input0_type << 24) | (_input1_type << 16) | (_output_type << 8))

#define VSI_NN_GEN_POW_STRUCT_ITEMS(_input0_type, _input1_type, _output_type) \
    VSI_NN_GEN_POW_KEY(_input0_type, _input1_type, _output_type), \
    VSI_NN_POW_SH_KERNEL_IDX(_input0_type, _input1_type, _output_type)

static struct {
        uint32_t key;
        uint32_t kernel_index;
        char *resource_name;
    } pow_map[] =
    {
        {VSI_NN_GEN_POW_STRUCT_ITEMS(F16, F16, F16)  "vsi_nn_kernel_pow"},
        {VSI_NN_GEN_POW_STRUCT_ITEMS(I16, I16, I16)  "vsi_nn_kernel_pow"},
        {VSI_NN_GEN_POW_STRUCT_ITEMS(I8,  I8,  I8)  "vsi_nn_kernel_pow"},
        {VSI_NN_GEN_POW_STRUCT_ITEMS(U8,  U8,  U8)  "vsi_nn_kernel_pow"},
        {VSI_NN_GEN_POW_STRUCT_ITEMS(U8,  F16, F16)  "vsi_nn_kernel_pow"},
        {VSI_NN_GEN_POW_STRUCT_ITEMS(I8,  F16, F16)  "vsi_nn_kernel_pow_i8_i16"},
        {VSI_NN_GEN_POW_STRUCT_ITEMS(I16, F16, F16)  "vsi_nn_kernel_pow_i8_i16"},
        {VSI_NN_GEN_POW_STRUCT_ITEMS(F16, U8,  F16)  "vsi_nn_kernel_pow_i8_i16"},
    };

static vsi_nn_shader_kernel_type_e get_pow_intra_type(vsi_nn_type_e type)
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
    default:
        VSILOGE("error data type %d", type);
        break;
    }

    return I8;
}

static void _get_pow_hashtable_idx
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_type_e input0Format = inputs[0]->attr.dtype.vx_type;
    vsi_nn_type_e input1Format = inputs[1]->attr.dtype.vx_type;
    vsi_nn_type_e outputFormat  = outputs[0]->attr.dtype.vx_type;
    vsi_nn_shader_kernel_type_e _input0_type;
    vsi_nn_shader_kernel_type_e _input1_type;
    vsi_nn_shader_kernel_type_e _output_type;
    uint32_t key;
    uint32_t i = 0;

    vsi_nn_pow_param * p = &(self->nn_param.pow);

    _input0_type = get_pow_intra_type(input0Format);
    _input1_type = get_pow_intra_type(input1Format);
    _output_type = get_pow_intra_type(outputFormat);

    key = VSI_NN_GEN_POW_KEY(_input0_type, _input1_type, _output_type);

    for (i = 0; i < sizeof(pow_map) / sizeof(pow_map[0]); i++)
    {
        if (key == pow_map[i].key)
        {
            p->local.hash_idx = i;
            p->local.execute_on_sw = FALSE;
            return;
        }
    }

    p->local.execute_on_sw = TRUE;
    VSILOGE("Shader unsupport data format! execute on the SW [pow]\n");
}

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

    if (index == 0)
    {
        if( input->attr.dim_num == 1)
        {
            memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
            attr.size[1] = 1;
            attr.dim_num = 2;
            self->nn_param.pow.local.local_tensor[index] =
                vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
            params[index] =  (vx_reference)self->nn_param.pow.local.local_tensor[index];
        }
        else
            params[index] = (vx_reference)input->t;
    }
    else if (index == 1)
    {
        if( input->attr.dim_num == 1)
        {
            memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
            attr.size[1] = 1;
            attr.dim_num = 2;
            self->nn_param.pow.local.local_tensor[index] =
                vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
            params[index] =  (vx_reference)self->nn_param.pow.local.local_tensor[index];
        }
        else
            params[index] = (vx_reference)input->t;
    }
    else if(index == 2)
    {
        if(input->attr.dim_num == 1)
        {
            memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
            attr.size[1] = 1;
            attr.dim_num = 2;
            self->nn_param.pow.local.local_tensor[index] =
                vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
            params[index] =  (vx_reference)self->nn_param.pow.local.local_tensor[index];
        }
        else
             params[index] = (vx_reference)input->t;
    }
    else
    {
        VSILOGE("No more local tensor!(pow) at [%s : %d]\n", __FILE__, __LINE__);
    }
}

static vsi_status cpu_op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_reference params[_PARAM_NUM + 1];
    vx_bool rsFlg = vx_false_e;

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    /* Set inputs and outputs */
    //_set_inputs_outputs( params, inputs, outputs );
    check_tensor_shape(self, inputs[0], params, 0, rsFlg);
    check_tensor_shape(self, inputs[1], params, 1, rsFlg);
    check_tensor_shape(self, outputs[0], params, 2, rsFlg);
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
    vsi_nn_pow_param * p = &(self->nn_param.pow);

    kernel_info->kernel_index = pow_map[p->local.hash_idx].kernel_index;
    kernel_info->resource_num = 1;
    kernel_info->resource_name[0] = pow_map[p->local.hash_idx].resource_name;

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
    vx_bool rsFlg = vx_false_e;

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    /* Set inputs and outputs */
    //_set_inputs_outputs( params, inputs, outputs );
    check_tensor_shape(self, inputs[0], params, 0, rsFlg);
    check_tensor_shape(self, inputs[1], params, 1, rsFlg);
    check_tensor_shape(self, outputs[0], params, 2, rsFlg);

    /* Pass parameters to node. */
    status |= vsi_nn_ClientNodePassParameters( self->n, params, _PARAM_NUM);

    border.mode = VX_BORDER_REPLICATE;
    status |= vxSetNodeAttribute(self->n, VX_NODE_BORDER, &border, sizeof(border));

    return status;
}

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_kernel_info_t kernel_info;
    vsi_nn_pow_param* p = &(self->nn_param.pow);

    memset(&kernel_info, 0x0, sizeof(vsi_nn_kernel_info_t));
    _get_pow_hashtable_idx(self, inputs, outputs);
    if (p->local.execute_on_sw)
    {
        kernel_info.resource_num = 1;
        kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
        kernel_info.resource_name[0] = "vsi_nn_kernel_pow";
        kernel_info.type = VX_KERNEL_TYPE_CPU;
        kernel_info.kernel = vx_kernel_POW_list;
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
        kernel_info.resource_num = 1;
        kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
        kernel_info.resource_name[0] = "vsi_nn_kernel_pow";
        kernel_info.type = vsi_nn_GetVXKernelTypeForShader();
        kernel_info.kernel = vx_kernel_POW_list;
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

        status = vx_op_compute(self, inputs, outputs);
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
    for (i = 0; i < _VSI_NN_POW_LOCAL_TENSOR_NUM; i++)
    {
        if (self->nn_param.pow.local.local_tensor[i] != NULL)
        {
            vxReleaseTensor(&(self->nn_param.pow.local.local_tensor[i]));
            self->nn_param.pow.local.local_tensor[i] = NULL;
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
    /* op_name    */ POW,
    /* init       */ NULL,
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

