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

#define _ARG_NUM            (2)
#define _INPUT_NUM          (4)
#define _OUTPUT_NUM         (1)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)

#define _VARI_ARG_NUM            (2)
#define _VARI_INPUT_NUM          (1)
#define _VARI_OUTPUT_NUM         (1)
#define _VARI_IO_NUM             (_VARI_INPUT_NUM + _VARI_OUTPUT_NUM)
#define _VARI_PARAM_NUM          (_VARI_ARG_NUM + _VARI_IO_NUM)
#define TEMP_DIM_SIZE            (4)

extern vx_kernel_description_t * vx_kernel_INSTANCENORM_list[];

#define VSI_NN_GEN_INSTANCENORM_KEY(_input0_type, _output_type, _reshape_flag) \
    ((_input0_type << 24) | (_output_type << 16) | (_reshape_flag << 8))

#define VSI_NN_GEN_INSTANCENORM_STRUCT_ITEMS(_input0_type, _output_type, _reshape_flag) \
    VSI_NN_GEN_INSTANCENORM_KEY(_input0_type, _output_type, _reshape_flag), \
    VSI_NN_SUMSQR_SH_KERNEL_IDX(_input0_type, _reshape_flag) \
    VSI_NN_INSTANCENORM_SH_KERNEL_IDX(_input0_type, _output_type, _reshape_flag)

static struct {
        uint32_t key;
        uint32_t kernel_index0;
        uint32_t kernel_index1;
        char *resource_name;
    } instancenorm_map[] =
    {
        {VSI_NN_GEN_INSTANCENORM_STRUCT_ITEMS(U8, U8, 1)  "vsi_nn_kernel_instancenormalize_u8"},
        {VSI_NN_GEN_INSTANCENORM_STRUCT_ITEMS(U8, F16, 1)  "vsi_nn_kernel_instancenormalize_u8"},
        {VSI_NN_GEN_INSTANCENORM_STRUCT_ITEMS(U8, U8, 0)  "vsi_nn_kernel_instancenormalize_u8"},
        {VSI_NN_GEN_INSTANCENORM_STRUCT_ITEMS(U8, F16, 0)  "vsi_nn_kernel_instancenormalize_u8"},
        {VSI_NN_GEN_INSTANCENORM_STRUCT_ITEMS(I8, I8, 1)  "vsi_nn_kernel_instancenormalize"},
        {VSI_NN_GEN_INSTANCENORM_STRUCT_ITEMS(I8, F16, 1)  "vsi_nn_kernel_instancenormalize"},
        {VSI_NN_GEN_INSTANCENORM_STRUCT_ITEMS(I8, I8, 0)  "vsi_nn_kernel_instancenormalize"},
        {VSI_NN_GEN_INSTANCENORM_STRUCT_ITEMS(I8, F16, 0)  "vsi_nn_kernel_instancenormalize"},
        {VSI_NN_GEN_INSTANCENORM_STRUCT_ITEMS(I16, I16, 1)  "vsi_nn_kernel_instancenormalize_i16"},
        {VSI_NN_GEN_INSTANCENORM_STRUCT_ITEMS(I16, F16, 1)  "vsi_nn_kernel_instancenormalize_i16"},
        {VSI_NN_GEN_INSTANCENORM_STRUCT_ITEMS(I16, I16, 0)  "vsi_nn_kernel_instancenormalize_i16"},
        {VSI_NN_GEN_INSTANCENORM_STRUCT_ITEMS(I16, F16, 0)  "vsi_nn_kernel_instancenormalize_i16"},
        {VSI_NN_GEN_INSTANCENORM_STRUCT_ITEMS(F16, F16, 1)  "vsi_nn_kernel_instancenormalize_fp16"},
        {VSI_NN_GEN_INSTANCENORM_STRUCT_ITEMS(F16, F16, 0)  "vsi_nn_kernel_instancenormalize_fp16"},
    };

static vsi_nn_shader_kernel_type_e get_instancenorm_intra_type(vsi_nn_type_e type)
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

static void _get_instancenorm_hashtable_idx
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_type_e input0Format = inputs[0]->attr.dtype.vx_type;
    vsi_nn_type_e outputFormat  = outputs[0]->attr.dtype.vx_type;
    vsi_nn_shader_kernel_type_e _input0_type;
    vsi_nn_shader_kernel_type_e _output_type;
    uint32_t key;
    uint32_t i = 0;
    uint32_t rsFlg = 0;

    vsi_nn_instancenormalize_param * p = &(self->nn_param.instancenorm);

    rsFlg = p->lcl2_data->reshapeFlg;
    _input0_type = get_instancenorm_intra_type(input0Format);
    _output_type = get_instancenorm_intra_type(outputFormat);

    key = VSI_NN_GEN_INSTANCENORM_KEY(_input0_type, _output_type, rsFlg);

    for (i = 0; i < sizeof(instancenorm_map) / sizeof(instancenorm_map[0]); i++)
    {
        if (key == instancenorm_map[i].key)
        {
            p->lcl2_data->hash_idx = i;
            p->lcl2_data->execute_on_sw = FALSE;
            return;
        }
    }

    p->lcl2_data->execute_on_sw = TRUE;
    VSILOGE("Shader unsupport data format! execute on the SW [instancenorm]\n");
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

    if (index == 0 )
    {
        if( input->attr.dim_num == 1 )
        {
            memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
            attr.size[1] = 1;
            attr.dim_num = 2;
            self->nn_param.instancenorm.local.local_tensor[index] =
                vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
            params[index] =  (vx_reference)self->nn_param.instancenorm.local.local_tensor[index];
        }
        else if(rsFlg)
        {
            if(self->nn_param.instancenorm.local.local_tensor[index] == NULL)
            {
                memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
                attr.size[1] = attr.size[1] * attr.size[2];
                attr.size[2] = 1;
                attr.dim_num = 2;
                self->nn_param.instancenorm.local.local_tensor[index] =
                    vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
            }
            params[index] =  (vx_reference)self->nn_param.instancenorm.local.local_tensor[index];
        }
        else
        {
            params[index] = (vx_reference)input->t;
        }
    }
    else if(index == 1 )
    {
        if(input->attr.dim_num == 1)
        {
            memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
            attr.size[1] = 1;
            attr.dim_num = 2;
            self->nn_param.instancenorm.local.local_tensor[index] =
                vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
            params[index] =  (vx_reference)self->nn_param.instancenorm.local.local_tensor[index];
        }
        else
        {
             params[index] = (vx_reference)input->t;
        }

    }
    else if(index == 2 )
    {
        if(input->attr.dim_num == 1)
        {
            memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
            attr.size[1] = 1;
            attr.dim_num = 2;
            self->nn_param.instancenorm.local.local_tensor[index] =
                vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
            params[index] =  (vx_reference)self->nn_param.instancenorm.local.local_tensor[index];
        }
        else
        {
             params[index] = (vx_reference)input->t;
        }
    }
    else if(index == 3 )
    {
        if(input->attr.dim_num == 1)
        {
            memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
            attr.size[1] = 1;
            attr.dim_num = 2;
            self->nn_param.instancenorm.local.local_tensor[index] =
                vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
            params[index] =  (vx_reference)self->nn_param.instancenorm.local.local_tensor[index];
        }
        else if(rsFlg)
        {
            if(self->nn_param.instancenorm.local.local_tensor[index] == NULL)
            {
                memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
                attr.size[1] = attr.size[1] * attr.size[2];
                attr.size[2] = 1;
                attr.dim_num = 2;
                self->nn_param.instancenorm.local.local_tensor[index] =
                    vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
            }
            params[index] =  (vx_reference)self->nn_param.instancenorm.local.local_tensor[index];
        }
        else
        {
             params[index] = (vx_reference)input->t;
        }
    }
    else if(index == 4 )
    {
        if(input->attr.dim_num == 1)
        {
            memcpy(&attr, &(input->attr), sizeof(vsi_nn_tensor_attr_t));
            attr.size[1] = 1;
            attr.dim_num = 2;
            self->nn_param.instancenorm.local.local_tensor[index] =
                vxReshapeTensor(input->t, (int32_t*)(attr.size), attr.dim_num);
            params[index] =  (vx_reference)self->nn_param.instancenorm.local.local_tensor[index];
        }
        else
        {
             params[index] = (vx_reference)input->t;
        }
    }
    else
    {
        VSILOGE("No more local tensor!(INSTANCENORM) at [%s : %d]\n", __FILE__, __LINE__);
    }
}

static vsi_status _create_params
    (
    vsi_nn_node_t * node,
    vx_reference * params,
    uint32_t num
    )
{
    vsi_status status;
    vx_context ctx;
    vsi_nn_instancenormalize_param * p;
    if( 0 == num )
    {
        return VSI_SUCCESS;
    }
    memset( params, 0, sizeof( vx_reference * ) * num );
    p = &(node->nn_param.instancenorm);
    ctx = vxGetContext( (vx_reference)node->graph->g );
    /* Init parameters */
#define _SET_PARAM( i, type, arg ) do{ \
    params[i] = (vx_reference)vxCreateScalar( ctx, type, &p->arg ); \
    status = vxGetStatus( params[i] ); \
    if( VSI_SUCCESS != status ) { \
    goto set_param_error; \
    } \
    } while(0)
    _SET_PARAM( 0, VX_TYPE_FLOAT32, eps );
    _SET_PARAM( 1, VX_TYPE_INT32, lcl2_data->reshapeFlg );
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
    vsi_nn_tensor_t ** outputs,
    vsi_nn_tensor_t * tmpInput
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
    check_tensor_shape(self, inputs[0], params, 0, 0);
    check_tensor_shape(self, inputs[1], params, 1, 0);
    check_tensor_shape(self, inputs[2], params, 2, 0);
    check_tensor_shape(self, outputs[0], params, 3, 0);
    check_tensor_shape(self, tmpInput, params, 4, 0);

    /* Init parameters. */
    _create_params( self, args, _ARG_NUM );

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, _PARAM_NUM );

    _release_params( args, _ARG_NUM );

    return status;
}

static vsi_status vx_mean_vari_op_pre_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_kernel_info_t * kernel_info
    )
{
    vsi_nn_instancenormalize_param * p = &(self->nn_param.instancenorm);

    kernel_info->kernel_index = instancenorm_map[p->lcl2_data->hash_idx].kernel_index0;
    kernel_info->resource_num = 2;
    kernel_info->resource_name[0] = "vsi_nn_kernel_header";
    kernel_info->resource_name[1] = instancenorm_map[p->lcl2_data->hash_idx].resource_name;

    return VSI_SUCCESS;
}

static vsi_status vx_op_pre_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_kernel_info_t * kernel_info
    )
{
    vsi_nn_instancenormalize_param * p = &(self->nn_param.instancenorm);

    kernel_info->kernel_index = instancenorm_map[p->lcl2_data->hash_idx].kernel_index1;
    kernel_info->resource_num = 2;
    kernel_info->resource_name[0] = "vsi_nn_kernel_header";
    kernel_info->resource_name[1] = instancenorm_map[p->lcl2_data->hash_idx].resource_name;

    return VSI_SUCCESS;
}

static vsi_status vx_mean_vari_op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t * output
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_reference params[_VARI_PARAM_NUM];
    vx_border_t border;
    vx_reference * args;
    vx_bool rsFlg = vx_false_e;
    int32_t in_zp = inputs[0]->attr.dtype.zero_point;
    vsi_nn_type_e inputDataFormat = inputs[0]->attr.dtype.vx_type;
    //vsi_nn_tensor_attr_t attr;
    //vsi_nn_tensor_t* tmpTensor = NULL;
    args = &params[_VARI_IO_NUM];

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    if(self->nn_param.instancenorm.lcl2_data->reshapeFlg)
    {
        rsFlg = vx_true_e;
    }
    /* Set inputs and outputs */
    check_tensor_shape(self, inputs[0], params, 0, rsFlg);
    check_tensor_shape(self, output, params, 1, rsFlg);
    _create_params( self, args, _VARI_ARG_NUM );

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, _VARI_PARAM_NUM );
    _release_params( args, _VARI_ARG_NUM );

    border.mode = VX_BORDER_CONSTANT;
    border.constant_value.U32 = 0;
    border.constant_value.S16 = 0;
    border.constant_value.U8 = 0;
    if(inputDataFormat == VSI_NN_TYPE_UINT8)
    {
        border.constant_value.U32 = in_zp;
        border.constant_value.S16 = in_zp;
        border.constant_value.U8 = in_zp;
    }
    status |= vxSetNodeAttribute(self->n, VX_NODE_BORDER, &border, sizeof(border));

    //if(tmpTensor)vsi_nn_ReleaseTensor(&tmpTensor);

    return status;
}

static vsi_status vx_op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_tensor_t * tmpInput
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_reference params[_PARAM_NUM];
    vx_border_t border;
    vx_reference * args;
    vx_bool rsFlg = vx_false_e;
    int32_t in_zp = inputs[0]->attr.dtype.zero_point;
    vsi_nn_type_e inputDataFormat = inputs[0]->attr.dtype.vx_type;

    args = &params[_IO_NUM];

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    if(self->nn_param.instancenorm.lcl2_data->reshapeFlg)
    {
        rsFlg = vx_true_e;
    }

    /* Set inputs and outputs */
    check_tensor_shape(self, inputs[0], params, 0, rsFlg);
    check_tensor_shape(self, inputs[1], params, 1, rsFlg);
    check_tensor_shape(self, inputs[2], params, 2, rsFlg);
    check_tensor_shape(self, outputs[0], params, 3, rsFlg);
    check_tensor_shape(self, tmpInput, params, 4, rsFlg);
    /* Init parameters. */
    _create_params( self, args, _ARG_NUM );

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, _PARAM_NUM );

    _release_params( args, _ARG_NUM );

    border.mode = VX_BORDER_CONSTANT;
    border.constant_value.U32 = 0;
    border.constant_value.S16 = 0;
    border.constant_value.U8 = 0;
    if(inputDataFormat == VSI_NN_TYPE_UINT8)
    {
        border.constant_value.U32 = in_zp;
        border.constant_value.S16 = in_zp;
        border.constant_value.U8 = in_zp;
    }
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
    vsi_status status = VX_SUCCESS;
    vsi_nn_kernel_info_t kernel_info;
    vsi_nn_instancenormalize_param* p = &(self->nn_param.instancenorm);
    vsi_nn_type_e inputDataFormat     = inputs[0]->attr.dtype.vx_type;
    uint32_t input_size[TEMP_DIM_SIZE] = {0};
    vsi_nn_tensor_t* tmpMeanVari = NULL;
    uint32_t i = 0;

    memset(&kernel_info, 0x0, sizeof(vsi_nn_kernel_info_t));

    // create temp tensor
    {
        vsi_nn_tensor_attr_t attr;
        memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));

        for(i = 0; i < inputs[0]->attr.dim_num; i++)
        {
            input_size[i] = inputs[0]->attr.size[i];
        }

        if(input_size[2] < 1)
        {
            input_size[2] = 1;
        }

        if((input_size[1] * input_size[2] < 65536)
            && inputs[0]->attr.dim_num > 2)
        {
            self->nn_param.instancenorm.lcl2_data->reshapeFlg = 1;
        }

        attr.size[0] = ((input_size[0] + 255) / 256) * 4;
        if(inputDataFormat == VSI_NN_TYPE_INT16 || VSI_NN_TYPE_FLOAT16)
        {
            attr.size[0] = ((input_size[0] + 127) / 128) * 4;
        }
        attr.size[1] = input_size[2];
        attr.size[2] = 1;
        attr.size[3] = 1;
        attr.dim_num = 2;
        attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
        attr.vtl = FALSE;
        tmpMeanVari = vsi_nn_CreateTensor(self->graph, &attr);
    }

    // check shader kernel
    _get_instancenorm_hashtable_idx(self, inputs, outputs);

    if(p->lcl2_data->execute_on_sw)
    {
        kernel_info.resource_num = 1;
        kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
        kernel_info.resource_name[0] = "vsi_nn_kernel_instancenormalize";
        kernel_info.type = VX_KERNEL_TYPE_CPU;
        kernel_info.kernel = vx_kernel_INSTANCENORM_list;
        kernel_info.init_index = 0;

        self->n = vsi_nn_RegisterClientKernelAndNewNode(
        self->graph, &kernel_info);
        if (kernel_info.resource_name) {free(kernel_info.resource_name);}
        if( NULL == self->n )
        {
            goto OnError;
        }

        status = cpu_op_compute(self, inputs, outputs, tmpMeanVari);
        if(tmpMeanVari){vsi_nn_ReleaseTensor(&tmpMeanVari);}

        return status;
    }
    else
    {
        {
            kernel_info.resource_num = 2;
            kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
            kernel_info.resource_name[0] = "vsi_nn_kernel_header";
            kernel_info.resource_name[1] = "vsi_nn_kernel_instancenormalize";
            kernel_info.type = VX_KERNEL_TYPE_VX;
            kernel_info.kernel = vx_kernel_INSTANCENORM_list;
            kernel_info.init_index = 1;
            if (kernel_info.type == VX_KERNEL_TYPE_VX)
            {
                vx_mean_vari_op_pre_compute(self, inputs, outputs, &kernel_info);
            }

            self->n = vsi_nn_RegisterClientKernelAndNewNode(
                self->graph, &kernel_info);
            if (kernel_info.resource_name) {free(kernel_info.resource_name);}
            if( NULL == self->n )
            {
                status = VSI_FAILURE;
                goto OnError;
            }
            status = vx_mean_vari_op_compute(self, inputs, tmpMeanVari);
            if(status != VX_SUCCESS)
            {
                goto OnError;
            }
        }

        kernel_info.resource_num = 2;
        kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
        kernel_info.resource_name[0] = "vsi_nn_kernel_header";
        kernel_info.resource_name[1] = "vsi_nn_kernel_instancenormalize";
        kernel_info.type = vsi_nn_GetVXKernelTypeForShader();
        kernel_info.kernel = vx_kernel_INSTANCENORM_list;
        kernel_info.init_index = 1;

        if (vsi_nn_is_do_vx_op_pre_init(kernel_info.type))
        {
            vx_op_pre_compute(self, inputs, outputs, &kernel_info);
        }

        self->n = vsi_nn_RegisterClientKernelAndNewNode(
            self->graph, &kernel_info);
        if (kernel_info.resource_name) {free(kernel_info.resource_name);}
        if( NULL == self->n )
        {
            status = VSI_FAILURE;
            goto OnError;
        }

        status = vx_op_compute(self, inputs, outputs, tmpMeanVari);
    }

OnError:
    if(tmpMeanVari){vsi_nn_ReleaseTensor(&tmpMeanVari);}

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

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    self->nn_param.instancenorm.lcl2_data =
    (vsi_nn_instancenorm_lcl_data2 *)malloc(sizeof(vsi_nn_instancenorm_lcl_data2));
    if (NULL == self->nn_param.instancenorm.lcl2_data)
    {
        return  VX_ERROR_NO_MEMORY;
    }

    memset( self->nn_param.instancenorm.lcl2_data, 0, sizeof(vsi_nn_instancenorm_lcl_data2) );

    self->nn_param.instancenorm.lcl2_data->reshapeFlg = 0;
    self->nn_param.instancenorm.lcl2_data->execute_on_sw = 0;
    self->nn_param.instancenorm.lcl2_data->hash_idx = 0;

    return status;
} /* op_init() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    uint32_t i;
    for (i = 0; i < _VSI_NN_INSTANCENORM_LOCAL_TENSOR_NUM; i++)
    {
        if (self->nn_param.instancenorm.local.local_tensor[i] != NULL)
        {
            vxReleaseTensor(&(self->nn_param.instancenorm.local.local_tensor[i]));
            self->nn_param.instancenorm.local.local_tensor[i] = NULL;
        }
    }
    if(self->nn_param.instancenorm.lcl2_data)
    {
        free(self->nn_param.instancenorm.lcl2_data);
        self->nn_param.instancenorm.lcl2_data = NULL;
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
    /* op_name    */ INSTANCE_NORM,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ vsi_nn_op_common_setup,
    /* optimize   */ NULL,
    /* input_num  */ 3,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif

