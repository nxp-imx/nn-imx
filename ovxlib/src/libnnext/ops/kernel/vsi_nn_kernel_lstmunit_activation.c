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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "vsi_nn_platform.h"

#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_math.h"
#include "client/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"

#define _VX_KERNEL_ID           (VX_KERNEL_ENUM_LSTMUNIT_ACTIVATION)
#define _VX_KERNEL_FUNC_KERNEL  (vxnneSWLSTMUnitActivationKernel)

#define ARG_NUM           (1)
#define TENSOR_NUM_INPUT  (LSTMUNIT_ACT_INPUTS_COUNT)
#define TENSOR_NUM_OUTPUT (LSTMUNIT_ACT_OUTUTS_COUNT)
#define TENSOR_NUM (TENSOR_NUM_INPUT+TENSOR_NUM_OUTPUT)

/* c -> cifg, l -> layer norm, p -> projection, h -> peephole, b -> hybrid bias fp32, s -> standard*/

static float vsi_nn_DtypeToFloat32_Ex
    (
    uint8_t   * src,
    uint32_t    index,
    const vsi_nn_dtype_t * src_dtype
    )
{
    float value = 0.0f;
    vsi_status status;

    src = src + index * vsi_nn_TypeGetBytes(src_dtype->vx_type);

    status = vsi_nn_DtypeToFloat32(src, &value, src_dtype);
    if(VSI_SUCCESS != status)
    {
        VSILOGE("Convert data to float32 fail!");
        value = 0.0f;
    }

    return value;
}

static vsi_status vsi_nn_Float32ToDtype_Ext
    (
    float   src,
    uint8_t   * dst,
    uint32_t    index,
    const vsi_nn_dtype_t * dst_dtype
    )
{
    vsi_nn_dtype_t src_dtype;

    dst = dst + index * vsi_nn_TypeGetBytes(dst_dtype->vx_type);

    memset( &src_dtype, 0, sizeof( vsi_nn_dtype_t ) );
    src_dtype.vx_type = VSI_NN_TYPE_FLOAT32;

    return vsi_nn_DtypeConvert( (uint8_t *)&src, &src_dtype, dst, dst_dtype );
} /* vsi_nn_Float32ToDtype_Ext */

float activationFunctor(float a, vsi_nn_activation_e act_)
{
    switch (act_)
    {
      case VSI_NN_ACT_NONE:
        return a;
      case VSI_NN_ACT_RELU:
        return a < 0.f ? 0.f : a;
      case VSI_NN_ACT_RELU6:
        return vsi_nn_max(0.f, vsi_nn_min(a, 6.f));
      case VSI_NN_ACT_TANH:
        return (float)tanh(a);
      case VSI_NN_ACT_SIGMOID:
        return (float)(1.0f / (1.0f + exp(-a)));
      case VSI_NN_ACT_HARD_SIGMOID:
          a = a * 0.2f + 0.5f;
        return vsi_nn_max(0.f, vsi_nn_min(a, 1.f));
      default:
        // TODO(aselle): More informative fatal error!
        exit(1);
    }

    return a;
  }

#define gcoMATH_Exp(X)        (float)(expf((X)))
#define gcoMATH_TangentH(X)   (float)(tanhf((X)))
static vsi_status VX_CALLBACK vxnneSWLSTMUnitActivationKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
    vsi_status status = VX_SUCCESS;
    vsi_nn_tensor_attr_t attr[TENSOR_NUM];
    vx_tensor_addressing user_addr[TENSOR_NUM]  = {NULL};
    vx_uint8    *buffer_ptr[TENSOR_NUM]            = {NULL};
    vx_tensor   tensor[TENSOR_NUM];
    uint32_t    stride_size[TENSOR_NUM][VSI_NN_MAX_DIM_NUM];

    vx_uint32    i                              = 0;
    vx_uint32    b                              = 0;
    vx_uint32    n_batch                        = 0;
    vx_uint32    n_cell                         = 0;
    vx_tensor    lstmunit_param                 = (vx_tensor)paramObj[paramNum - 1];
    vx_context   context                        = vxGetContext((vx_reference)node);
    vsi_nn_tensor_attr_t lstmunit_param_attr;
    vsi_nn_lstmunit_activation_param *p = NULL;

    status = vsi_nn_vxGetTensorAttr(lstmunit_param, &lstmunit_param_attr);
    p = (vsi_nn_lstmunit_activation_param*)vsi_nn_vxCopyTensorToData(context, lstmunit_param, &lstmunit_param_attr);


    for( i = 0; i < TENSOR_NUM_INPUT; i ++ )
    {
        tensor[i] = (vx_tensor)paramObj[i];
        buffer_ptr[i] = vsi_nn_ConvertRawTensorToData2(context, tensor[i],
            &(attr[i]), stride_size[i], &(user_addr[i]), VX_READ_ONLY);
    }

    for( i = TENSOR_NUM_INPUT; i < TENSOR_NUM; i ++ )
    {
        tensor[i] = (vx_tensor)paramObj[i];
        buffer_ptr[i] = vsi_nn_ConvertRawTensorToData2(context, tensor[i],
            &(attr[i]), stride_size[i], &(user_addr[i]), VX_WRITE_ONLY);
    }

    n_cell  = attr[LSTMUNIT_ACT_CSTATE_IN].size[0];
    n_batch = attr[LSTMUNIT_ACT_CSTATE_IN].size[1];

    for (b = 0; b < n_batch; b ++)
    {
        for (i = 0; i < n_cell; i++)
        {
            uint32_t index = i + n_cell * b;
            float    data_i_t = 0;
            float    data_f_t = 0;
            float    data_g_t = 0;
            float    data_o_t = 0;
            float    data_c_t = 0;
            float    data_h_t = 0;

            data_i_t = p->is_cifg ? 0 : vsi_nn_DtypeToFloat32_Ex(buffer_ptr[LSTMUNIT_ACT_INPUT_FC_I],
                index, &attr[LSTMUNIT_ACT_INPUT_FC_I].dtype);

            data_f_t = vsi_nn_DtypeToFloat32_Ex(buffer_ptr[LSTMUNIT_ACT_INPUT_FC_F],
                index, &attr[LSTMUNIT_ACT_INPUT_FC_F].dtype);

            data_g_t = vsi_nn_DtypeToFloat32_Ex(buffer_ptr[LSTMUNIT_ACT_INPUT_FC_C],
                index, &attr[LSTMUNIT_ACT_INPUT_FC_C].dtype);

            data_o_t = vsi_nn_DtypeToFloat32_Ex(buffer_ptr[LSTMUNIT_ACT_INPUT_FC_O],
                index, &attr[LSTMUNIT_ACT_INPUT_FC_O].dtype);

            data_c_t = vsi_nn_DtypeToFloat32_Ex(buffer_ptr[LSTMUNIT_ACT_CSTATE_IN],
                index, &attr[LSTMUNIT_ACT_CSTATE_IN].dtype);

            if (!p->is_layer_norm)
            {
                data_i_t += p->is_cifg ? 0 : vsi_nn_DtypeToFloat32_Ex(buffer_ptr[LSTMUNIT_ACT_HSTATE_FC_I],
                    index, &attr[LSTMUNIT_ACT_HSTATE_FC_I].dtype);

                data_f_t += vsi_nn_DtypeToFloat32_Ex(buffer_ptr[LSTMUNIT_ACT_HSTATE_FC_F],
                    index, &attr[LSTMUNIT_ACT_HSTATE_FC_F].dtype);

                data_g_t += vsi_nn_DtypeToFloat32_Ex(buffer_ptr[LSTMUNIT_ACT_HSTATE_FC_C],
                    index, &attr[LSTMUNIT_ACT_HSTATE_FC_C].dtype);

                data_o_t += vsi_nn_DtypeToFloat32_Ex(buffer_ptr[LSTMUNIT_ACT_HSTATE_FC_O],
                    index, &attr[LSTMUNIT_ACT_HSTATE_FC_O].dtype);
            }

            if (!p->is_cifg)
            {
                if (p->is_layer_norm)
                {
                    data_i_t *= vsi_nn_DtypeToFloat32_Ex(buffer_ptr[LSTMUNIT_ACT_LN_WI],
                        i, &attr[LSTMUNIT_ACT_LN_WI].dtype);
                    data_i_t += vsi_nn_DtypeToFloat32_Ex(buffer_ptr[LSTMUNIT_ACT_DATA_BI],
                        i, &attr[LSTMUNIT_ACT_DATA_BI].dtype);
                }
                else if (p->is_hybrid)
                {
                    data_i_t += vsi_nn_DtypeToFloat32_Ex(buffer_ptr[LSTMUNIT_ACT_DATA_BI],
                        i, &attr[LSTMUNIT_ACT_DATA_BI].dtype);
                }
            }

            if (p->is_layer_norm)
            {
                data_f_t *= vsi_nn_DtypeToFloat32_Ex(buffer_ptr[LSTMUNIT_ACT_LN_WF],
                    i, &attr[LSTMUNIT_ACT_LN_WF].dtype);
                data_f_t += vsi_nn_DtypeToFloat32_Ex(buffer_ptr[LSTMUNIT_ACT_DATA_BF],
                    i, &attr[LSTMUNIT_ACT_DATA_BF].dtype);

                data_g_t *= vsi_nn_DtypeToFloat32_Ex(buffer_ptr[LSTMUNIT_ACT_LN_WC],
                    i, &attr[LSTMUNIT_ACT_LN_WC].dtype);
                data_g_t += vsi_nn_DtypeToFloat32_Ex(buffer_ptr[LSTMUNIT_ACT_DATA_BC],
                    i, &attr[LSTMUNIT_ACT_DATA_BC].dtype);

                data_o_t *= vsi_nn_DtypeToFloat32_Ex(buffer_ptr[LSTMUNIT_ACT_LN_WO],
                    i, &attr[LSTMUNIT_ACT_LN_WO].dtype);
                data_o_t += vsi_nn_DtypeToFloat32_Ex(buffer_ptr[LSTMUNIT_ACT_DATA_BO],
                    i, &attr[LSTMUNIT_ACT_DATA_BO].dtype);
            }
            else if (p->is_hybrid)
            {
                data_f_t += vsi_nn_DtypeToFloat32_Ex(buffer_ptr[LSTMUNIT_ACT_DATA_BF],
                    i, &attr[LSTMUNIT_ACT_DATA_BF].dtype);

                data_g_t += vsi_nn_DtypeToFloat32_Ex(buffer_ptr[LSTMUNIT_ACT_DATA_BC],
                    i, &attr[LSTMUNIT_ACT_DATA_BC].dtype);

                data_o_t += vsi_nn_DtypeToFloat32_Ex(buffer_ptr[LSTMUNIT_ACT_DATA_BO],
                    i, &attr[LSTMUNIT_ACT_DATA_BO].dtype);
            }

            data_f_t += p->forget_bias;

            data_f_t = activationFunctor(data_f_t, p->recurrent_activation);
            if (p->is_cifg)
                data_i_t = 1 - data_f_t;
            else
                data_i_t = activationFunctor(data_i_t, p->recurrent_activation);
            data_g_t = gcoMATH_TangentH(data_g_t);
            data_o_t = activationFunctor(data_o_t, p->recurrent_activation);
            data_c_t = data_f_t * data_c_t + data_i_t * data_g_t;
            data_h_t = data_o_t * gcoMATH_TangentH(data_c_t);

            vsi_nn_Float32ToDtype_Ext(data_c_t, buffer_ptr[LSTMUNIT_ACT_CSTATE_OUT + LSTMUNIT_ACT_INPUTS_COUNT],
                index, &attr[LSTMUNIT_ACT_CSTATE_OUT + LSTMUNIT_ACT_INPUTS_COUNT].dtype);
            vsi_nn_Float32ToDtype_Ext(data_h_t, buffer_ptr[LSTMUNIT_ACT_OUTPUT + LSTMUNIT_ACT_INPUTS_COUNT],
                index, &attr[LSTMUNIT_ACT_OUTPUT + LSTMUNIT_ACT_INPUTS_COUNT].dtype);

            if (!p->is_projection)
            {
                vsi_nn_Float32ToDtype_Ext(data_h_t, buffer_ptr[LSTMUNIT_ACT_HSTATE_OUT + LSTMUNIT_ACT_INPUTS_COUNT],
                    index, &attr[LSTMUNIT_ACT_HSTATE_OUT + LSTMUNIT_ACT_INPUTS_COUNT].dtype);
            }
        }
    }

    //save data
    for( i = TENSOR_NUM_INPUT; i < TENSOR_NUM; i ++ )
    {
        if (buffer_ptr[i])
        {
            status = vsi_nn_copy_tensor_patch(tensor[i], &attr[i], buffer_ptr[i], VX_WRITE_ONLY);
        }

        if (user_addr[i]) vxReleaseTensorAddressing(&(user_addr[i]));
    }

    for( i = 0; i < TENSOR_NUM; i ++ )
    {
        if (buffer_ptr[i]) free(buffer_ptr[i]);

        if (user_addr[i]) vxReleaseTensorAddressing(&(user_addr[i]));
    }

    if (p) free(p);

    return status;
}

static vx_param_description_t sw_params[] =
    {
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL },  /*0  input_fc_i */
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },  /*1  input_fc_f */
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },  /*2  input_fc_c */
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },  /*3  input_fc_o */
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },  /*4  cs_in */
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL },  /*5  hstate_fc_i */
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL },  /*6  hstate_fc_f */
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL },  /*7  hstate_fc_c */
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL },  /*8  hstate_fc_o */
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL },  /*9  biases_i*/
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL },  /*10 biases_f*/
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL },  /*11 biases_c*/
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL },  /*12 biases_o*/
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL },  /*13 ln_w_i*/
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL },  /*14 ln_w_f*/
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL },  /*15 ln_w_c*/
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL },  /*16 ln_w_o*/

    { VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED }, /*11 output*/
    { VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED }, /*12 cs_out*/
    { VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL }, /*13 hs_out*/
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL },  /*14 param*/
    };

typedef enum _lstmunit_cifg_ln_proj_e
{
    CLP_INPUT_FC_F,
    CLP_INPUT_FC_C,
    CLP_INPUT_FC_O,
    CLP_CSTATE_IN,
    CLP_BIASES_F,
    CLP_BIASES_C,
    CLP_BIASES_O,
    CLP_LN_WF,
    CLP_LN_WC,
    CLP_LN_WO,
    CLP_OUTPUT,
    CLP_CSTATE_OUT,
    CLP_PARAM
} lstmunit_cifg_ln_proj_e;

static vx_param_description_t vxLSTMUNIT_CLP_Param[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
};

typedef enum _lstmunit_cifg_ln_e
{
    CL_INPUT_FC_F,
    CL_INPUT_FC_C,
    CL_INPUT_FC_O,
    CL_CSTATE_IN,
    CL_BIASES_F,
    CL_BIASES_C,
    CL_BIASES_O,
    CL_LN_WF,
    CL_LN_WC,
    CL_LN_WO,
    CL_OUTPUT,
    CL_CSTATE_OUT,
    CL_HSTATE_OUT,
    CL_LSTMUNIT_PARAM,
} lstmunit_cifg_ln_e;

static vx_param_description_t vxLSTMUNIT_CL_Param[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
};

typedef enum _lstmunit_ln_proj_e
{
    LP_INPUT_FC_I,
    LP_INPUT_FC_F,
    LP_INPUT_FC_C,
    LP_INPUT_FC_O,
    LP_CSTATE_IN,
    LP_BIASES_I,
    LP_BIASES_F,
    LP_BIASES_C,
    LP_BIASES_O,
    LP_LN_WI,
    LP_LN_WF,
    LP_LN_WC,
    LP_LN_WO,
    LP_OUTPUT,
    LP_CSTATE_OUT,
    LP_PARAM
} lstmunit_ln_proj_e;

static vx_param_description_t vxLSTMUNIT_LP_Param[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
};

typedef enum _lstmunit_ln_e
{
    L_INPUT_FC_I,
    L_INPUT_FC_F,
    L_INPUT_FC_C,
    L_INPUT_FC_O,
    L_CSTATE_IN,
    L_BIASES_I,
    L_BIASES_F,
    L_BIASES_C,
    L_BIASES_O,
    L_LN_WI,
    L_LN_WF,
    L_LN_WC,
    L_LN_WO,
    L_OUTPUT,
    L_CSTATE_OUT,
    L_HSTATE_OUT,
    L_LSTMUNIT_PARAM,
} lstmunit_ln_e;

static vx_param_description_t vxLSTMUNIT_L_Param[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
};

typedef enum _lstmunit_hybrid_proj_e
{
    BP_INPUT_FC_I,
    BP_INPUT_FC_F,
    BP_INPUT_FC_C,
    BP_INPUT_FC_O,
    BP_CSTATE_IN,
    BP_HSTATE_FC_I,
    BP_HSTATE_FC_F,
    BP_HSTATE_FC_C,
    BP_HSTATE_FC_O,
    BP_BIASES_I,
    BP_BIASES_F,
    BP_BIASES_C,
    BP_BIASES_O,
    BP_OUTPUT,
    BP_CSTATE_OUT,
    BP_PARAM
} lstmunit_hybrid_proj_e;

static vx_param_description_t vxLSTMUNIT_BP_Param[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
};

typedef enum _lstmunit_hybrid_e
{
    B_INPUT_FC_I,
    B_INPUT_FC_F,
    B_INPUT_FC_C,
    B_INPUT_FC_O,
    B_CSTATE_IN,
    B_HSTATE_FC_I,
    B_HSTATE_FC_F,
    B_HSTATE_FC_C,
    B_HSTATE_FC_O,
    B_BIASES_I,
    B_BIASES_F,
    B_BIASES_C,
    B_BIASES_O,
    B_OUTPUT,
    B_CSTATE_OUT,
    B_HSTATE_OUT,
    B_LSTMUNIT_PARAM,
} lstmunit_hybrid_e;

static vx_param_description_t vxLSTMUNIT_B_Param[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
};

typedef enum _lstmunit_cifg_hybrid_proj_e
{
    CBP_INPUT_FC_F,
    CBP_INPUT_FC_C,
    CBP_INPUT_FC_O,
    CBP_CSTATE_IN,
    CBP_HSTATE_FC_F,
    CBP_HSTATE_FC_C,
    CBP_HSTATE_FC_O,
    CBP_BIASES_F,
    CBP_BIASES_C,
    CBP_BIASES_O,
    CBP_OUTPUT,
    CBP_CSTATE_OUT,
    CBP_PARAM
} lstmunit_cifg_hybrid_proj_e;

static vx_param_description_t vxLSTMUNIT_CBP_Param[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
};

typedef enum _lstmunit_cifg_hybrid_e
{
    CB_INPUT_FC_F,
    CB_INPUT_FC_C,
    CB_INPUT_FC_O,
    CB_CSTATE_IN,
    CB_HSTATE_FC_F,
    CB_HSTATE_FC_C,
    CB_HSTATE_FC_O,
    CB_BIASES_F,
    CB_BIASES_C,
    CB_BIASES_O,
    CB_OUTPUT,
    CB_CSTATE_OUT,
    CB_HSTATE_OUT,
    CB_LSTMUNIT_PARAM,
} lstmunit_cifg_hybrid_e;

static vx_param_description_t vxLSTMUNIT_CB_Param[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
};

typedef enum _lstmunit_standard_proj_e
{
    SP_INPUT_FC_I,
    SP_INPUT_FC_F,
    SP_INPUT_FC_C,
    SP_INPUT_FC_O,
    SP_CSTATE_IN,
    SP_HSTATE_FC_I,
    SP_HSTATE_FC_F,
    SP_HSTATE_FC_C,
    SP_HSTATE_FC_O,
    SP_OUTPUT,
    SP_CSTATE_OUT,
    SP_PARAM
} lstmunit_standard_proj_e;

static vx_param_description_t vxLSTMUNIT_SP_Param[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
};

typedef enum _lstmunit_standard_e
{
    S_INPUT_FC_I,
    S_INPUT_FC_F,
    S_INPUT_FC_C,
    S_INPUT_FC_O,
    S_CSTATE_IN,
    S_HSTATE_FC_I,
    S_HSTATE_FC_F,
    S_HSTATE_FC_C,
    S_HSTATE_FC_O,
    S_OUTPUT,
    S_CSTATE_OUT,
    S_HSTATE_OUT,
    S_LSTMUNIT_PARAM,
} lstmunit_standard_e;

static vx_param_description_t vxLSTMUNIT_S_Param[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
};

typedef enum _lstmunit_cifg_standard_proj_e
{
    CSP_INPUT_FC_F,
    CSP_INPUT_FC_C,
    CSP_INPUT_FC_O,
    CSP_CSTATE_IN,
    CSP_HSTATE_FC_F,
    CSP_HSTATE_FC_C,
    CSP_HSTATE_FC_O,
    CSP_OUTPUT,
    CSP_CSTATE_OUT,
    CSP_PARAM
} lstmunit_cifg_standard_proj_e;

static vx_param_description_t vxLSTMUNIT_CSP_Param[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
};

typedef enum _lstmunit_cifg_standard_e
{
    CS_INPUT_FC_F,
    CS_INPUT_FC_C,
    CS_INPUT_FC_O,
    CS_CSTATE_IN,
    CS_HSTATE_FC_F,
    CS_HSTATE_FC_C,
    CS_HSTATE_FC_O,
    CS_OUTPUT,
    CS_CSTATE_OUT,
    CS_HSTATE_OUT,
    CS_LSTMUNIT_PARAM,
} lstmunit_cifg_standard_e;

static vx_param_description_t vxLSTMUNIT_CS_Param[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
};

vx_status VX_CALLBACK vxLSTMUnit_Activation_Initializer
    (
    vx_node node,
    const vx_reference *paramObj,
    vx_uint32 paraNum
    )
{
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
    vx_kernel_execution_parameters_t shaderParam = {
        2,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vx_status    status                 = VX_SUCCESS;
    vx_tensor    lstmunit_param         = (vx_tensor)paramObj[paraNum - 1];
    vx_tensor    cell_state_in          = NULL;
    vx_tensor    output                 = NULL;
    vx_uint32    output_size[4]         = {1, 1, 1, 1};
    vx_float32   cell_clip              = 0;
    vx_float32   outputScale            = 1.0f;
    vx_float32   outputZP               = 0;
    vx_int32     dstZP                  = 0;
    vx_float32   dstScale               = 0;
    vx_enum      cellFormat             = VSI_NN_TYPE_FLOAT16;
    vx_enum      dstFormat              = VSI_NN_TYPE_FLOAT16;
    vx_enum      dstQuantType           = 0;
    vx_int8      dstFixPointPos         = 0;
    vx_float32   logE                   = (vx_float32)(log10(exp(1.0f)) / log10(2.0f));
    vx_float32   twoLogE                = 2 * logE;
    vx_uint32    uint_min               = 0xFBFFFFFF;
    vx_uint32    uint_max               = 0x7BFFFFFF;
    vx_float32   float_min              = *(vx_float32 *)&uint_min;
    vx_float32   float_max              = *(vx_float32 *)&uint_max;
    vx_float32   clip_Min_F[4]          = {0};
    vx_float32   clip_Max_F[4]          = {0};
    vx_uint32    i                      = 0;
    vx_int32     input0Array_ZP[4]      = {0};
    vx_int32     input1Array_ZP[4]      = {0};
    vx_float32   input0Array_Scale[4]   = {1.0f};
    vx_float32   input1Array_Scale[4]   = {1.0f};
    vx_context   context                = vxGetContext((vx_reference)node);
    vsi_nn_tensor_attr_t input_attr[9];
    vsi_nn_tensor_attr_t lstmunit_param_attr;
    vsi_nn_lstmunit_activation_param *p = NULL;
    vsi_nn_tensor_attr_t attr[2];
    uint32_t output_dims = 0;

    status = vsi_nn_vxGetTensorAttr(lstmunit_param, &lstmunit_param_attr);
    p = (vsi_nn_lstmunit_activation_param*)vsi_nn_vxCopyTensorToData(context, lstmunit_param, &lstmunit_param_attr);

    cell_clip = p->cell_clip;

    if (p->is_cifg)
    {
        cell_state_in = (vx_tensor)paramObj[CL_CSTATE_IN];
        if (p->is_layer_norm)
            output = (vx_tensor)paramObj[CL_OUTPUT];
        else if (p->is_hybrid)
            output = (vx_tensor)paramObj[CB_OUTPUT];
        else
            output = (vx_tensor)paramObj[CS_OUTPUT];
    }
    else
    {
        cell_state_in = (vx_tensor)paramObj[L_CSTATE_IN];
        if (p->is_layer_norm)
            output = (vx_tensor)paramObj[L_OUTPUT];
        else if (p->is_hybrid)
            output = (vx_tensor)paramObj[B_OUTPUT];
        else
            output = (vx_tensor)paramObj[S_OUTPUT];
    }

    for (i = 0; i < 9; i++)
    {
        vsi_nn_vxGetTensorAttr((vx_tensor)paramObj[i], &input_attr[i]);
    }
    memset(&attr[0], 0, sizeof(vsi_nn_tensor_attr_t));
    memset(&attr[1], 0, sizeof(vsi_nn_tensor_attr_t));
    status  = vsi_nn_vxGetTensorAttr(cell_state_in, &attr[0]);
    status |= vsi_nn_vxGetTensorAttr(output, &attr[1]);
    if (status != VX_SUCCESS)
    {
        VSILOGE("vsi_nn_vxGetTensorAttr failure! at line %d\n", __LINE__);
        goto final;
    }

    cellFormat = attr[0].dtype.vx_type;
    output_dims = attr[1].dim_num;
    dstFormat   = attr[1].dtype.vx_type;
    for (i = 0; i < output_dims; i++)
    {
        output_size[i] = attr[1].size[i];
    }
    dstQuantType = attr[1].dtype.qnt_type;
    dstFixPointPos = attr[1].dtype.fl;
    dstZP = attr[1].dtype.zero_point;
    dstScale = attr[1].dtype.scale;

    outputZP  = (vx_float32)dstZP;

    shaderParam.globalWorkScale[0]  = 4;
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.globalWorkSize[0]   = gcmALIGN((output_size[0] + shaderParam.globalWorkScale[0] - 1)
        / shaderParam.globalWorkScale[0], 4);
    shaderParam.globalWorkSize[1]   = (output_size[1] + shaderParam.globalWorkScale[1] - 1)
        / shaderParam.globalWorkScale[1];


    if (cell_clip > 0)
    {
        float_max = cell_clip;
        float_min = -cell_clip;
    }

    for (i = 0; i < 4; i++)
    {
        clip_Min_F[i] = float_min;
        clip_Max_F[i] = float_max;
    }

    {
        vx_uint32 uniFp16toFp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };
        vx_uint32 uniExtractHalf4_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00020000, 0x00060004, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };
        vx_uint32 uniExtractInteger_2x8[16] = {
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniExtractHalf8_2x8[16] = {
            0x11111111, // TCfg
            0x11110000, // ASelt
            0x06040200, 0x06040200, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00,
            0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
        };
        vx_uint32 uniFp16AddFp16toFp32_4x4[16] = {
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00110000, 0x00330022, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000,
            0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000 // Constant
        };
        vx_uint32 uniU8AddS32_4x4[16] = {
            0x0d0d0d0d, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        };

        if (dstQuantType == VX_QUANT_DYNAMIC_FIXED_POINT)
        {
            if (dstFixPointPos >= 0)
                outputScale *= (vx_float32)(1 << dstFixPointPos);
            else if (dstFixPointPos < 0)
                outputScale *= 1.0f / (vx_float32) (1 << -dstFixPointPos);

            outputZP = 0;
        }
        else if (dstQuantType == VX_QUANT_AFFINE_SCALE)
        {
            outputScale = 1.0f / dstScale;
        }

        if (cellFormat == VSI_NN_TYPE_FLOAT16)
            vxSetNodeUniform(node, "uniExtractHalf4_4x4", 1, uniExtractHalf4_4x4);

        if (dstFormat == VSI_NN_TYPE_FLOAT16)
            vxSetNodeUniform(node, "uniExtract8Data_2x8", 1, uniExtractHalf8_2x8);
        else
            vxSetNodeUniform(node, "uniExtract8Data_2x8", 1, uniExtractInteger_2x8);

        vxSetNodeUniform(node, "uniFp16toFp32_4x4", 1, uniFp16toFp32_4x4);
        vxSetNodeUniform(node, "logE", 1, &logE);
        vxSetNodeUniform(node, "twoLogE", 1, &twoLogE);
        vxSetNodeUniform(node, "outputScale", 1, &outputScale);
        vxSetNodeUniform(node, "outputZP", 1, &outputZP);
        vxSetNodeUniform(node, "forget_bias", 1, &p->forget_bias);
        vxSetNodeUniform(node, "clip_Min_F", 1, clip_Min_F);
        vxSetNodeUniform(node, "clip_Max_F", 1, clip_Max_F);

        if (!p->is_layer_norm && input_attr[S_INPUT_FC_F].dtype.vx_type == VSI_NN_TYPE_FLOAT16)
        {
            vxSetNodeUniform(node, "uniFp16AddFp16toFp32_4x4", 1, uniFp16AddFp16toFp32_4x4);
        }

        if (input_attr[S_INPUT_FC_F].dtype.vx_type == VSI_NN_TYPE_UINT8 &&
            input_attr[S_INPUT_FC_F].dtype.qnt_type == VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC)
        {
            if (p->is_cifg)
            {
                input0Array_ZP[1]    = 0 -  input_attr[CS_INPUT_FC_F].dtype.zero_point;
                input0Array_ZP[2]    = 0 -  input_attr[CS_INPUT_FC_C].dtype.zero_point;
                input0Array_ZP[3]    = 0 -  input_attr[CS_INPUT_FC_O].dtype.zero_point;

                input0Array_Scale[1] = input_attr[CS_INPUT_FC_F].dtype.scale;
                input0Array_Scale[2] = input_attr[CS_INPUT_FC_C].dtype.scale;
                input0Array_Scale[3] = input_attr[CS_INPUT_FC_O].dtype.scale;

                if (!p->is_layer_norm)
                {
                    input1Array_ZP[1]    = 0 -  input_attr[CS_HSTATE_FC_F].dtype.zero_point;
                    input1Array_ZP[2]    = 0 -  input_attr[CS_HSTATE_FC_C].dtype.zero_point;
                    input1Array_ZP[3]    = 0 -  input_attr[CS_HSTATE_FC_O].dtype.zero_point;

                    input1Array_Scale[1] = input_attr[CS_HSTATE_FC_F].dtype.scale;
                    input1Array_Scale[2] = input_attr[CS_HSTATE_FC_C].dtype.scale;
                    input1Array_Scale[3] = input_attr[CS_HSTATE_FC_O].dtype.scale;
                }
            }
            else
            {
                input0Array_ZP[0]    = 0 -  input_attr[S_INPUT_FC_I].dtype.zero_point;
                input0Array_ZP[1]    = 0 -  input_attr[S_INPUT_FC_F].dtype.zero_point;
                input0Array_ZP[2]    = 0 -  input_attr[S_INPUT_FC_C].dtype.zero_point;
                input0Array_ZP[3]    = 0 -  input_attr[S_INPUT_FC_O].dtype.zero_point;

                input0Array_Scale[0] = input_attr[S_INPUT_FC_I].dtype.scale;
                input0Array_Scale[1] = input_attr[S_INPUT_FC_F].dtype.scale;
                input0Array_Scale[2] = input_attr[S_INPUT_FC_C].dtype.scale;
                input0Array_Scale[3] = input_attr[S_INPUT_FC_O].dtype.scale;

                if (!p->is_layer_norm)
                {
                    input1Array_ZP[0]    = 0 -  input_attr[S_HSTATE_FC_I].dtype.zero_point;
                    input1Array_ZP[1]    = 0 -  input_attr[S_HSTATE_FC_F].dtype.zero_point;
                    input1Array_ZP[2]    = 0 -  input_attr[S_HSTATE_FC_C].dtype.zero_point;
                    input1Array_ZP[3]    = 0 -  input_attr[S_HSTATE_FC_O].dtype.zero_point;

                    input1Array_Scale[0] = input_attr[S_HSTATE_FC_I].dtype.scale;
                    input1Array_Scale[1] = input_attr[S_HSTATE_FC_F].dtype.scale;
                    input1Array_Scale[2] = input_attr[S_HSTATE_FC_C].dtype.scale;
                    input1Array_Scale[3] = input_attr[S_HSTATE_FC_O].dtype.scale;
                }
            }

            vxSetNodeUniform(node, "input0Array_ZP", 1, input0Array_ZP);
            vxSetNodeUniform(node, "input0Array_Scale", 1, input0Array_Scale);
            vxSetNodeUniform(node, "input1Array_ZP", 1, input1Array_ZP);
            vxSetNodeUniform(node, "input1Array_Scale", 1, input1Array_Scale);
            vxSetNodeUniform(node, "uniU8AddS32_4x4", 1, uniU8AddS32_4x4);
        }
    }

    status |= vxSetNodeAttribute(node, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));
final:
    if (p) free(p);

    return status;
}

#ifdef __cplusplus
extern "C" {
#endif

#define GEN_SH_KERNEL_NAME(_C_L_P_H, _INPUT_TYPE, _OUTPUT_TYPE, _CELL_TYPE, _ACTIVATION) \
    "com.vivantecorp.extension.vxLSTMUnit_"#_C_L_P_H"_"#_INPUT_TYPE"to"#_OUTPUT_TYPE"_"#_CELL_TYPE#_ACTIVATION

vx_kernel_description_t vxLSTMUnit_SW_Kernel =
{
    _VX_KERNEL_ID,
    VX_KERNEL_NAME_LSTMUNIT_ACTIVATION,
    _VX_KERNEL_FUNC_KERNEL,
    sw_params,
    _cnt_of_array( sw_params ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

#define LSTMUINT_KERENLS(_C_L_P_H, _INPUT_TYPE, _OUTPUT_TYPE, _CELL_TYPE, _ACTIVATION) \
    vx_kernel_description_t vxLSTMUnit_##_C_L_P_H##_##_INPUT_TYPE##to##_OUTPUT_TYPE## \
_##_CELL_TYPE##_ACTIVATION##_Kernel = \
{ \
    _VX_KERNEL_ID, \
    GEN_SH_KERNEL_NAME(_C_L_P_H, _INPUT_TYPE, _OUTPUT_TYPE, _CELL_TYPE, _ACTIVATION), \
    NULL, \
    vxLSTMUNIT_##_C_L_P_H##_Param, \
    _cnt_of_array( vxLSTMUNIT_##_C_L_P_H##_Param ), \
    vsi_nn_KernelValidator, \
    NULL, \
    NULL, \
    vxLSTMUnit_Activation_Initializer, \
    vsi_nn_KernelDeinitializer \
};

    /* layer norm + cifg + projection */
    LSTMUINT_KERENLS(CLP, F16, F16, F32, )
    LSTMUINT_KERENLS(CLP, F16, F16, F16, )
    /* layer norm + projection */
    LSTMUINT_KERENLS(LP,  F16, F16, F32, )
    LSTMUINT_KERENLS(LP,  F16, F16, F16, )
    /* layer norm + cifg */
    LSTMUINT_KERENLS(CL,  F16, F16, F16, )
    LSTMUINT_KERENLS(CL,  F16, I16, F16, )
    LSTMUINT_KERENLS(CL,  F16, I8,  F16, )
    LSTMUINT_KERENLS(CL,  F16, U8,  F16, )
    LSTMUINT_KERENLS(CL,  F16, F16, F32, )
    LSTMUINT_KERENLS(CL,  F16, I16, F32, )
    LSTMUINT_KERENLS(CL,  F16, I8,  F32, )
    LSTMUINT_KERENLS(CL,  F16, U8,  F32, )
    /* layer norm */
    LSTMUINT_KERENLS(L,   F16, F16, F16, )
    LSTMUINT_KERENLS(L,   F16, I16, F16, )
    LSTMUINT_KERENLS(L,   F16, I8,  F16, )
    LSTMUINT_KERENLS(L,   F16, U8,  F16, )
    LSTMUINT_KERENLS(L,   F16, F16, F32, )
    LSTMUINT_KERENLS(L,   F16, I16, F32, )
    LSTMUINT_KERENLS(L,   F16, I8,  F32, )
    LSTMUINT_KERENLS(L,   F16, U8,  F32, )

    /* layer norm + cifg + projection */
    LSTMUINT_KERENLS(CLP, F16, I16, F32, )
    LSTMUINT_KERENLS(CLP, F16, I8,  F32, )
    LSTMUINT_KERENLS(CLP, F16, U8,  F32, )
    LSTMUINT_KERENLS(CLP, F16, I16, F16, )
    LSTMUINT_KERENLS(CLP, F16, I8,  F16, )
    LSTMUINT_KERENLS(CLP, F16, U8,  F16, )
    /* layer norm + projection */
    LSTMUINT_KERENLS(LP,  F16, I16, F32, )
    LSTMUINT_KERENLS(LP,  F16, I8,  F32, )
    LSTMUINT_KERENLS(LP,  F16, U8,  F32, )
    LSTMUINT_KERENLS(LP,  F16, I16, F16, )
    LSTMUINT_KERENLS(LP,  F16, I8,  F16, )
    LSTMUINT_KERENLS(LP,  F16, U8,  F16, )

    /* hybrid + projection */
    LSTMUINT_KERENLS(BP,  F16, F16, F32, )
    LSTMUINT_KERENLS(BP,  F16, F16, F16, )
    /* hybrid */
    LSTMUINT_KERENLS(B,   F16, F16, F16, )
    LSTMUINT_KERENLS(B,   F16, I16, F16, )
    LSTMUINT_KERENLS(B,   F16, I8,  F16, )
    LSTMUINT_KERENLS(B,   F16, U8,  F16, )
    LSTMUINT_KERENLS(B,   F16, F16, F32, )
    LSTMUINT_KERENLS(B,   F16, I16, F32, )
    LSTMUINT_KERENLS(B,   F16, I8,  F32, )
    LSTMUINT_KERENLS(B,   F16, U8,  F32, )
    /* cifg + hybrid + projection */
    LSTMUINT_KERENLS(CBP, F16, F16, F32, )
    LSTMUINT_KERENLS(CBP, F16, F16, F16, )
    /* cifg + hybrid */
    LSTMUINT_KERENLS(CB,  F16, F16, F16, )
    LSTMUINT_KERENLS(CB,  F16, I16, F16, )
    LSTMUINT_KERENLS(CB,  F16, I8,  F16, )
    LSTMUINT_KERENLS(CB,  F16, U8,  F16, )
    LSTMUINT_KERENLS(CB,  F16, F16, F32, )
    LSTMUINT_KERENLS(CB,  F16, I16, F32, )
    LSTMUINT_KERENLS(CB,  F16, I8,  F32, )
    LSTMUINT_KERENLS(CB,  F16, U8,  F32, )
    /* cifg + hybrid + projection */
    LSTMUINT_KERENLS(CBP, F16, I16, F32, )
    LSTMUINT_KERENLS(CBP, F16, I8,  F32, )
    LSTMUINT_KERENLS(CBP, F16, U8,  F32, )
    LSTMUINT_KERENLS(CBP, F16, I16, F16, )
    LSTMUINT_KERENLS(CBP, F16, I8,  F16, )
    LSTMUINT_KERENLS(CBP, F16, U8,  F16, )
    /* hybrid + projection */
    LSTMUINT_KERENLS(BP,  F16, I16, F32, )
    LSTMUINT_KERENLS(BP,  F16, I8,  F32, )
    LSTMUINT_KERENLS(BP,  F16, U8,  F32, )
    LSTMUINT_KERENLS(BP,  F16, I16, F16, )
    LSTMUINT_KERENLS(BP,  F16, I8,  F16, )
    LSTMUINT_KERENLS(BP,  F16, U8,  F16, )
    /* hybrid + projection */
    LSTMUINT_KERENLS(BP,  U8,  F16, F32, )
    LSTMUINT_KERENLS(BP,  U8,  F16, F16, )
    /* hybrid */
    LSTMUINT_KERENLS(B,   U8,  F16, F16, )
    LSTMUINT_KERENLS(B,   U8,  U8,  F16, )
    LSTMUINT_KERENLS(B,   U8,  F16, F32, )
    LSTMUINT_KERENLS(B,   U8,  U8,  F32, )
    /* cifg + hybrid + projection */
    LSTMUINT_KERENLS(CBP, U8,  F16, F32, )
    LSTMUINT_KERENLS(CBP, U8,  F16, F16, )
    /* cifg + hybrid */
    LSTMUINT_KERENLS(CB,  U8,  F16, F16, )
    LSTMUINT_KERENLS(CB,  U8,  U8,  F16, )
    LSTMUINT_KERENLS(CB,  U8,  F16, F32, )
    LSTMUINT_KERENLS(CB,  U8,  U8,  F32, )
    /* cifg + hybrid + projection */
    LSTMUINT_KERENLS(CBP, U8,  I16, F32, )
    LSTMUINT_KERENLS(CBP, U8,  I8,  F32, )
    LSTMUINT_KERENLS(CBP, U8,  U8,  F32, )
    LSTMUINT_KERENLS(CBP, U8,  I16, F16, )
    LSTMUINT_KERENLS(CBP, U8,  I8,  F16, )
    LSTMUINT_KERENLS(CBP, U8,  U8,  F16, )
    /* hybrid + projection */
    LSTMUINT_KERENLS(BP,  U8,  I16, F32, )
    LSTMUINT_KERENLS(BP,  U8,  I8,  F32, )
    LSTMUINT_KERENLS(BP,  U8,  U8,  F32, )
    LSTMUINT_KERENLS(BP,  U8,  I16, F16, )
    LSTMUINT_KERENLS(BP,  U8,  I8,  F16, )
    LSTMUINT_KERENLS(BP,  U8,  U8,  F16, )

    /* standard + projection */
    LSTMUINT_KERENLS(SP,  F16, F16, F32, )
    LSTMUINT_KERENLS(SP,  F16, F16, F16, )
    /* standard */
    LSTMUINT_KERENLS(S,   F16, F16, F16, )
    LSTMUINT_KERENLS(S,   F16, I16, F16, )
    LSTMUINT_KERENLS(S,   F16, I8,  F16, )
    LSTMUINT_KERENLS(S,   F16, U8,  F16, )
    LSTMUINT_KERENLS(S,   F16, F16, F32, )
    LSTMUINT_KERENLS(S,   F16, I16, F32, )
    LSTMUINT_KERENLS(S,   F16, I8,  F32, )
    LSTMUINT_KERENLS(S,   F16, U8,  F32, )
    /* cifg + standard + projection */
    LSTMUINT_KERENLS(CSP, F16, F16, F32, )
    LSTMUINT_KERENLS(CSP, F16, F16, F16, )
    /* cifg + standard */
    LSTMUINT_KERENLS(CS,  F16, F16, F16, )
    LSTMUINT_KERENLS(CS,  F16, I16, F16, )
    LSTMUINT_KERENLS(CS,  F16, I8,  F16, )
    LSTMUINT_KERENLS(CS,  F16, U8,  F16, )
    LSTMUINT_KERENLS(CS,  F16, F16, F32, )
    LSTMUINT_KERENLS(CS,  F16, I16, F32, )
    LSTMUINT_KERENLS(CS,  F16, I8,  F32, )
    LSTMUINT_KERENLS(CS,  F16, U8,  F32, )
    /* cifg + standard + projection */
    LSTMUINT_KERENLS(CSP, F16, I16, F32, )
    LSTMUINT_KERENLS(CSP, F16, I8,  F32, )
    LSTMUINT_KERENLS(CSP, F16, U8,  F32, )
    LSTMUINT_KERENLS(CSP, F16, I16, F16, )
    LSTMUINT_KERENLS(CSP, F16, I8,  F16, )
    LSTMUINT_KERENLS(CSP, F16, U8,  F16, )
    /* standard + projection */
    LSTMUINT_KERENLS(SP,  F16, I16, F32, )
    LSTMUINT_KERENLS(SP,  F16, I8,  F32, )
    LSTMUINT_KERENLS(SP,  F16, U8,  F32, )
    LSTMUINT_KERENLS(SP,  F16, I16, F16, )
    LSTMUINT_KERENLS(SP,  F16, I8,  F16, )
    LSTMUINT_KERENLS(SP,  F16, U8,  F16, )
    /* standard + projection */
    LSTMUINT_KERENLS(SP,  U8,  F16, F32, )
    LSTMUINT_KERENLS(SP,  U8,  F16, F16, )
    /* standard */
    LSTMUINT_KERENLS(S,   U8,  F16, F16, )
    LSTMUINT_KERENLS(S,   U8,  U8,  F16, )
    LSTMUINT_KERENLS(S,   U8,  F16, F32, )
    LSTMUINT_KERENLS(S,   U8,  U8,  F32, )
    /* cifg + standard + projection */
    LSTMUINT_KERENLS(CSP, U8,  F16, F32, )
    LSTMUINT_KERENLS(CSP, U8,  F16, F16, )
    /* cifg + standard */
    LSTMUINT_KERENLS(CS,  U8,  F16, F16, )
    LSTMUINT_KERENLS(CS,  U8,  U8,  F16, )
    LSTMUINT_KERENLS(CS,  U8,  F16, F32, )
    LSTMUINT_KERENLS(CS,  U8,  U8,  F32, )
    /* cifg + standard + projection */
    LSTMUINT_KERENLS(CSP, U8, I16, F32, )
    LSTMUINT_KERENLS(CSP, U8, I8,  F32, )
    LSTMUINT_KERENLS(CSP, U8, U8,  F32, )
    LSTMUINT_KERENLS(CSP, U8, I16, F16, )
    LSTMUINT_KERENLS(CSP, U8, I8,  F16, )
    LSTMUINT_KERENLS(CSP, U8, U8,  F16, )
    /* standard + projection */
    LSTMUINT_KERENLS(SP,  U8, I16, F32, )
    LSTMUINT_KERENLS(SP,  U8, I8,  F32, )
    LSTMUINT_KERENLS(SP,  U8, U8,  F32, )
    LSTMUINT_KERENLS(SP,  U8, I16, F16, )
    LSTMUINT_KERENLS(SP,  U8, I8,  F16, )
    LSTMUINT_KERENLS(SP,  U8, U8,  F16, )

    /* layer norm + cifg + projection */
    LSTMUINT_KERENLS(CLP, F16, F16, F32, _HARD)
    LSTMUINT_KERENLS(CLP, F16, F16, F16, _HARD)
    /* layer norm + projection + hard_sigmoid */
    LSTMUINT_KERENLS(LP,  F16, F16, F32, _HARD)
    LSTMUINT_KERENLS(LP,  F16, F16, F16, _HARD)
    /* layer norm + cifg + hard_sigmoid */
    LSTMUINT_KERENLS(CL,  F16, F16, F16, _HARD)
    LSTMUINT_KERENLS(CL,  F16, I16, F16, _HARD)
    LSTMUINT_KERENLS(CL,  F16, I8,  F16, _HARD)
    LSTMUINT_KERENLS(CL,  F16, U8,  F16, _HARD)
    LSTMUINT_KERENLS(CL,  F16, F16, F32, _HARD)
    LSTMUINT_KERENLS(CL,  F16, I16, F32, _HARD)
    LSTMUINT_KERENLS(CL,  F16, I8,  F32, _HARD)
    LSTMUINT_KERENLS(CL,  F16, U8,  F32, _HARD)
    /* layer norm + hard_sigmoid */
    LSTMUINT_KERENLS(L,   F16, F16, F16, _HARD)
    LSTMUINT_KERENLS(L,   F16, I16, F16, _HARD)
    LSTMUINT_KERENLS(L,   F16, I8,  F16, _HARD)
    LSTMUINT_KERENLS(L,   F16, U8,  F16, _HARD)
    LSTMUINT_KERENLS(L,   F16, F16, F32, _HARD)
    LSTMUINT_KERENLS(L,   F16, I16, F32, _HARD)
    LSTMUINT_KERENLS(L,   F16, I8,  F32, _HARD)
    LSTMUINT_KERENLS(L,   F16, U8,  F32, _HARD)

    /* layer norm + cifg + projection + hard_sigmoid */
    LSTMUINT_KERENLS(CLP, F16, I16, F32, _HARD)
    LSTMUINT_KERENLS(CLP, F16, I8,  F32, _HARD)
    LSTMUINT_KERENLS(CLP, F16, U8,  F32, _HARD)
    LSTMUINT_KERENLS(CLP, F16, I16, F16, _HARD)
    LSTMUINT_KERENLS(CLP, F16, I8,  F16, _HARD)
    LSTMUINT_KERENLS(CLP, F16, U8,  F16, _HARD)
    /* layer norm + projection + hard_sigmoid */
    LSTMUINT_KERENLS(LP,  F16, I16, F32, _HARD)
    LSTMUINT_KERENLS(LP,  F16, I8,  F32, _HARD)
    LSTMUINT_KERENLS(LP,  F16, U8,  F32, _HARD)
    LSTMUINT_KERENLS(LP,  F16, I16, F16, _HARD)
    LSTMUINT_KERENLS(LP,  F16, I8,  F16, _HARD)
    LSTMUINT_KERENLS(LP,  F16, U8,  F16, _HARD)

    /* hybrid + projection + hard_sigmoid */
    LSTMUINT_KERENLS(BP,  F16, F16, F32, _HARD)
    LSTMUINT_KERENLS(BP,  F16, F16, F16, _HARD)
    /* hybrid + hard_sigmoid */
    LSTMUINT_KERENLS(B,   F16, F16, F16, _HARD)
    LSTMUINT_KERENLS(B,   F16, I16, F16, _HARD)
    LSTMUINT_KERENLS(B,   F16, I8,  F16, _HARD)
    LSTMUINT_KERENLS(B,   F16, U8,  F16, _HARD)
    LSTMUINT_KERENLS(B,   F16, F16, F32, _HARD)
    LSTMUINT_KERENLS(B,   F16, I16, F32, _HARD)
    LSTMUINT_KERENLS(B,   F16, I8,  F32, _HARD)
    LSTMUINT_KERENLS(B,   F16, U8,  F32, _HARD)
    /* cifg + hybrid + projection + hard_sigmoid */
    LSTMUINT_KERENLS(CBP, F16, F16, F32, _HARD)
    LSTMUINT_KERENLS(CBP, F16, F16, F16, _HARD)
    /* cifg + hybrid + hard_sigmoid */
    LSTMUINT_KERENLS(CB,  F16, F16, F16, _HARD)
    LSTMUINT_KERENLS(CB,  F16, I16, F16, _HARD)
    LSTMUINT_KERENLS(CB,  F16, I8,  F16, _HARD)
    LSTMUINT_KERENLS(CB,  F16, U8,  F16, _HARD)
    LSTMUINT_KERENLS(CB,  F16, F16, F32, _HARD)
    LSTMUINT_KERENLS(CB,  F16, I16, F32, _HARD)
    LSTMUINT_KERENLS(CB,  F16, I8,  F32, _HARD)
    LSTMUINT_KERENLS(CB,  F16, U8,  F32, _HARD)
    /* cifg + hybrid + projection + hard_sigmoid */
    LSTMUINT_KERENLS(CBP, F16, I16, F32, _HARD)
    LSTMUINT_KERENLS(CBP, F16, I8,  F32, _HARD)
    LSTMUINT_KERENLS(CBP, F16, U8,  F32, _HARD)
    LSTMUINT_KERENLS(CBP, F16, I16, F16, _HARD)
    LSTMUINT_KERENLS(CBP, F16, I8,  F16, _HARD)
    LSTMUINT_KERENLS(CBP, F16, U8,  F16, _HARD)
    /* hybrid + projection + hard_sigmoid */
    LSTMUINT_KERENLS(BP,  F16, I16, F32, _HARD)
    LSTMUINT_KERENLS(BP,  F16, I8,  F32, _HARD)
    LSTMUINT_KERENLS(BP,  F16, U8,  F32, _HARD)
    LSTMUINT_KERENLS(BP,  F16, I16, F16, _HARD)
    LSTMUINT_KERENLS(BP,  F16, I8,  F16, _HARD)
    LSTMUINT_KERENLS(BP,  F16, U8,  F16, _HARD)
    /* hybrid + projection + hard_sigmoid */
    LSTMUINT_KERENLS(BP,  U8,  F16, F32, _HARD)
    LSTMUINT_KERENLS(BP,  U8,  F16, F16, _HARD)
    /* hybrid + hard_sigmoid */
    LSTMUINT_KERENLS(B,   U8,  F16, F16, _HARD)
    LSTMUINT_KERENLS(B,   U8,  U8,  F16, _HARD)
    LSTMUINT_KERENLS(B,   U8,  F16, F32, _HARD)
    LSTMUINT_KERENLS(B,   U8,  U8,  F32, _HARD)
    /* cifg + hybrid + projection + hard_sigmoid */
    LSTMUINT_KERENLS(CBP, U8,  F16, F32, _HARD)
    LSTMUINT_KERENLS(CBP, U8,  F16, F16, _HARD)
    /* cifg + hybrid + hard_sigmoid */
    LSTMUINT_KERENLS(CB,  U8,  F16, F16, _HARD)
    LSTMUINT_KERENLS(CB,  U8,  U8,  F16, _HARD)
    LSTMUINT_KERENLS(CB,  U8,  F16, F32, _HARD)
    LSTMUINT_KERENLS(CB,  U8,  U8,  F32, _HARD)
    /* cifg + hybrid + projection + hard_sigmoid */
    LSTMUINT_KERENLS(CBP, U8,  I16, F32, _HARD)
    LSTMUINT_KERENLS(CBP, U8,  I8,  F32, _HARD)
    LSTMUINT_KERENLS(CBP, U8,  U8,  F32, _HARD)
    LSTMUINT_KERENLS(CBP, U8,  I16, F16, _HARD)
    LSTMUINT_KERENLS(CBP, U8,  I8,  F16, _HARD)
    LSTMUINT_KERENLS(CBP, U8,  U8,  F16, _HARD)
    /* hybrid + projection + hard_sigmoid */
    LSTMUINT_KERENLS(BP,  U8,  I16, F32, _HARD)
    LSTMUINT_KERENLS(BP,  U8,  I8,  F32, _HARD)
    LSTMUINT_KERENLS(BP,  U8,  U8,  F32, _HARD)
    LSTMUINT_KERENLS(BP,  U8,  I16, F16, _HARD)
    LSTMUINT_KERENLS(BP,  U8,  I8,  F16, _HARD)
    LSTMUINT_KERENLS(BP,  U8,  U8,  F16, _HARD)

    /* standard + projection + hard_sigmoid */
    LSTMUINT_KERENLS(SP,  F16, F16, F32, _HARD)
    LSTMUINT_KERENLS(SP,  F16, F16, F16, _HARD)
    /* standard + hard_sigmoid */
    LSTMUINT_KERENLS(S,   F16, F16, F16, _HARD)
    LSTMUINT_KERENLS(S,   F16, I16, F16, _HARD)
    LSTMUINT_KERENLS(S,   F16, I8,  F16, _HARD)
    LSTMUINT_KERENLS(S,   F16, U8,  F16, _HARD)
    LSTMUINT_KERENLS(S,   F16, F16, F32, _HARD)
    LSTMUINT_KERENLS(S,   F16, I16, F32, _HARD)
    LSTMUINT_KERENLS(S,   F16, I8,  F32, _HARD)
    LSTMUINT_KERENLS(S,   F16, U8,  F32, _HARD)
    /* cifg + standard + projection + hard_sigmoid */
    LSTMUINT_KERENLS(CSP, F16, F16, F32, _HARD)
    LSTMUINT_KERENLS(CSP, F16, F16, F16, _HARD)
    /* cifg + standard + hard_sigmoid */
    LSTMUINT_KERENLS(CS,  F16, F16, F16, _HARD)
    LSTMUINT_KERENLS(CS,  F16, I16, F16, _HARD)
    LSTMUINT_KERENLS(CS,  F16, I8,  F16, _HARD)
    LSTMUINT_KERENLS(CS,  F16, U8,  F16, _HARD)
    LSTMUINT_KERENLS(CS,  F16, F16, F32, _HARD)
    LSTMUINT_KERENLS(CS,  F16, I16, F32, _HARD)
    LSTMUINT_KERENLS(CS,  F16, I8,  F32, _HARD)
    LSTMUINT_KERENLS(CS,  F16, U8,  F32, _HARD)
    /* cifg + standard + projection + hard_sigmoid */
    LSTMUINT_KERENLS(CSP, F16, I16, F32, _HARD)
    LSTMUINT_KERENLS(CSP, F16, I8,  F32, _HARD)
    LSTMUINT_KERENLS(CSP, F16, U8,  F32, _HARD)
    LSTMUINT_KERENLS(CSP, F16, I16, F16, _HARD)
    LSTMUINT_KERENLS(CSP, F16, I8,  F16, _HARD)
    LSTMUINT_KERENLS(CSP, F16, U8,  F16, _HARD)
    /* standard + projection + hard_sigmoid */
    LSTMUINT_KERENLS(SP,  F16, I16, F32, _HARD)
    LSTMUINT_KERENLS(SP,  F16, I8,  F32, _HARD)
    LSTMUINT_KERENLS(SP,  F16, U8,  F32, _HARD)
    LSTMUINT_KERENLS(SP,  F16, I16, F16, _HARD)
    LSTMUINT_KERENLS(SP,  F16, I8,  F16, _HARD)
    LSTMUINT_KERENLS(SP,  F16, U8,  F16, _HARD)
    /* standard + projection + hard_sigmoid */
    LSTMUINT_KERENLS(SP,  U8,  F16, F32, _HARD)
    LSTMUINT_KERENLS(SP,  U8,  F16, F16, _HARD)
    /* standard + hard_sigmoid */
    LSTMUINT_KERENLS(S,   U8,  F16, F16, _HARD)
    LSTMUINT_KERENLS(S,   U8,  U8,  F16, _HARD)
    LSTMUINT_KERENLS(S,   U8,  F16, F32, _HARD)
    LSTMUINT_KERENLS(S,   U8,  U8,  F32, _HARD)
    /* cifg + standard + projection + hard_sigmoid */
    LSTMUINT_KERENLS(CSP, U8,  F16, F32, _HARD)
    LSTMUINT_KERENLS(CSP, U8,  F16, F16, _HARD)
    /* cifg + standard + hard_sigmoid */
    LSTMUINT_KERENLS(CS,  U8,  F16, F16, _HARD)
    LSTMUINT_KERENLS(CS,  U8,  U8,  F16, _HARD)
    LSTMUINT_KERENLS(CS,  U8,  F16, F32, _HARD)
    LSTMUINT_KERENLS(CS,  U8,  U8,  F32, _HARD)
    /* cifg + standard + projection + hard_sigmoid */
    LSTMUINT_KERENLS(CSP, U8,  I16, F32, _HARD)
    LSTMUINT_KERENLS(CSP, U8,  I8,  F32, _HARD)
    LSTMUINT_KERENLS(CSP, U8,  U8,  F32, _HARD)
    LSTMUINT_KERENLS(CSP, U8,  I16, F16, _HARD)
    LSTMUINT_KERENLS(CSP, U8,  I8,  F16, _HARD)
    LSTMUINT_KERENLS(CSP, U8,  U8,  F16, _HARD)
    /* standard + projection + hard_sigmoid */
    LSTMUINT_KERENLS(SP,  U8,  I16, F32, _HARD)
    LSTMUINT_KERENLS(SP,  U8,  I8,  F32, _HARD)
    LSTMUINT_KERENLS(SP,  U8,  U8,  F32, _HARD)
    LSTMUINT_KERENLS(SP,  U8,  I16, F16, _HARD)
    LSTMUINT_KERENLS(SP,  U8,  I8,  F16, _HARD)
    LSTMUINT_KERENLS(SP,  U8,  U8,  F16, _HARD)

#define LSTMUINT_KERENL_NAME(_C_L_P_H, _INPUT_TYPE, _OUTPUT_TYPE, _CELL_TYPE, _ACTIVATION) \
    &vxLSTMUnit_##_C_L_P_H##_##_INPUT_TYPE##to##_OUTPUT_TYPE##_##_CELL_TYPE##_ACTIVATION##_Kernel,

vx_kernel_description_t * vx_kernel_LSTMUNIT_ACTIVATION_list[] =
{
    &vxLSTMUnit_SW_Kernel,
    /* layer norm + cifg + projection */
    LSTMUINT_KERENL_NAME(CLP, F16, F16, F32, )
    LSTMUINT_KERENL_NAME(CLP, F16, F16, F16, )
    /* layer norm + projection */
    LSTMUINT_KERENL_NAME(LP,  F16, F16, F32, )
    LSTMUINT_KERENL_NAME(LP,  F16, F16, F16, )
    /* layer norm + cifg */
    LSTMUINT_KERENL_NAME(CL,  F16, F16, F16, )
    LSTMUINT_KERENL_NAME(CL,  F16, I16, F16, )
    LSTMUINT_KERENL_NAME(CL,  F16, I8,  F16, )
    LSTMUINT_KERENL_NAME(CL,  F16, U8,  F16, )
    LSTMUINT_KERENL_NAME(CL,  F16, F16, F32, )
    LSTMUINT_KERENL_NAME(CL,  F16, I16, F32, )
    LSTMUINT_KERENL_NAME(CL,  F16, I8,  F32, )
    LSTMUINT_KERENL_NAME(CL,  F16, U8,  F32, )
    /* layer norm */
    LSTMUINT_KERENL_NAME(L,   F16, F16, F16, )
    LSTMUINT_KERENL_NAME(L,   F16, I16, F16, )
    LSTMUINT_KERENL_NAME(L,   F16, I8,  F16, )
    LSTMUINT_KERENL_NAME(L,   F16, U8,  F16, )
    LSTMUINT_KERENL_NAME(L,   F16, F16, F32, )
    LSTMUINT_KERENL_NAME(L,   F16, I16, F32, )
    LSTMUINT_KERENL_NAME(L,   F16, I8,  F32, )
    LSTMUINT_KERENL_NAME(L,   F16, U8,  F32, )

    /* layer norm + cifg + projection */
    LSTMUINT_KERENL_NAME(CLP, F16, I16, F32, )
    LSTMUINT_KERENL_NAME(CLP, F16, I8,  F32, )
    LSTMUINT_KERENL_NAME(CLP, F16, U8,  F32, )
    LSTMUINT_KERENL_NAME(CLP, F16, I16, F16, )
    LSTMUINT_KERENL_NAME(CLP, F16, I8,  F16, )
    LSTMUINT_KERENL_NAME(CLP, F16, U8,  F16, )
    /* layer norm + projection */
    LSTMUINT_KERENL_NAME(LP,  F16, I16, F32, )
    LSTMUINT_KERENL_NAME(LP,  F16, I8,  F32, )
    LSTMUINT_KERENL_NAME(LP,  F16, U8,  F32, )
    LSTMUINT_KERENL_NAME(LP,  F16, I16, F16, )
    LSTMUINT_KERENL_NAME(LP,  F16, I8,  F16, )
    LSTMUINT_KERENL_NAME(LP,  F16, U8,  F16, )

    /* hybrid + projection */
    LSTMUINT_KERENL_NAME(BP,  F16, F16, F32, )
    LSTMUINT_KERENL_NAME(BP,  F16, F16, F16, )
    /* hybrid */
    LSTMUINT_KERENL_NAME(B,   F16, F16, F16, )
    LSTMUINT_KERENL_NAME(B,   F16, I16, F16, )
    LSTMUINT_KERENL_NAME(B,   F16, I8,  F16, )
    LSTMUINT_KERENL_NAME(B,   F16, U8,  F16, )
    LSTMUINT_KERENL_NAME(B,   F16, F16, F32, )
    LSTMUINT_KERENL_NAME(B,   F16, I16, F32, )
    LSTMUINT_KERENL_NAME(B,   F16, I8,  F32, )
    LSTMUINT_KERENL_NAME(B,   F16, U8,  F32, )
    /* cifg + hybrid + projection */
    LSTMUINT_KERENL_NAME(CBP, F16, F16, F32, )
    LSTMUINT_KERENL_NAME(CBP, F16, F16, F16, )
    /* cifg + hybrid */
    LSTMUINT_KERENL_NAME(CB,  F16, F16, F16, )
    LSTMUINT_KERENL_NAME(CB,  F16, I16, F16, )
    LSTMUINT_KERENL_NAME(CB,  F16, I8,  F16, )
    LSTMUINT_KERENL_NAME(CB,  F16, U8,  F16, )
    LSTMUINT_KERENL_NAME(CB,  F16, F16, F32, )
    LSTMUINT_KERENL_NAME(CB,  F16, I16, F32, )
    LSTMUINT_KERENL_NAME(CB,  F16, I8,  F32, )
    LSTMUINT_KERENL_NAME(CB,  F16, U8,  F32, )
    /* cifg + hybrid + projection */
    LSTMUINT_KERENL_NAME(CBP, F16, I16, F32, )
    LSTMUINT_KERENL_NAME(CBP, F16, I8,  F32, )
    LSTMUINT_KERENL_NAME(CBP, F16, U8,  F32, )
    LSTMUINT_KERENL_NAME(CBP, F16, I16, F16, )
    LSTMUINT_KERENL_NAME(CBP, F16, I8,  F16, )
    LSTMUINT_KERENL_NAME(CBP, F16, U8,  F16, )
    /* hybrid + projection */
    LSTMUINT_KERENL_NAME(BP,  F16, I16, F32, )
    LSTMUINT_KERENL_NAME(BP,  F16, I8,  F32, )
    LSTMUINT_KERENL_NAME(BP,  F16, U8,  F32, )
    LSTMUINT_KERENL_NAME(BP,  F16, I16, F16, )
    LSTMUINT_KERENL_NAME(BP,  F16, I8,  F16, )
    LSTMUINT_KERENL_NAME(BP,  F16, U8,  F16, )
    /* hybrid + projection */
    LSTMUINT_KERENL_NAME(BP,  U8,  F16, F32, )
    LSTMUINT_KERENL_NAME(BP,  U8,  F16, F16, )
    /* hybrid */
    LSTMUINT_KERENL_NAME(B,   U8,  F16, F16, )
    LSTMUINT_KERENL_NAME(B,   U8,  U8,  F16, )
    LSTMUINT_KERENL_NAME(B,   U8,  F16, F32, )
    LSTMUINT_KERENL_NAME(B,   U8,  U8,  F32, )
    /* cifg + hybrid + projection */
    LSTMUINT_KERENL_NAME(CBP, U8,  F16, F32, )
    LSTMUINT_KERENL_NAME(CBP, U8,  F16, F16, )
    /* cifg + hybrid */
    LSTMUINT_KERENL_NAME(CB,  U8,  F16, F16, )
    LSTMUINT_KERENL_NAME(CB,  U8,  U8,  F16, )
    LSTMUINT_KERENL_NAME(CB,  U8,  F16, F32, )
    LSTMUINT_KERENL_NAME(CB,  U8,  U8,  F32, )
    /* cifg + hybrid + projection */
    LSTMUINT_KERENL_NAME(CBP, U8,  I16, F32, )
    LSTMUINT_KERENL_NAME(CBP, U8,  I8,  F32, )
    LSTMUINT_KERENL_NAME(CBP, U8,  U8,  F32, )
    LSTMUINT_KERENL_NAME(CBP, U8,  I16, F16, )
    LSTMUINT_KERENL_NAME(CBP, U8,  I8,  F16, )
    LSTMUINT_KERENL_NAME(CBP, U8,  U8,  F16, )
    /* hybrid + projection */
    LSTMUINT_KERENL_NAME(BP,  U8,  I16, F32, )
    LSTMUINT_KERENL_NAME(BP,  U8,  I8,  F32, )
    LSTMUINT_KERENL_NAME(BP,  U8,  U8,  F32, )
    LSTMUINT_KERENL_NAME(BP,  U8,  I16, F16, )
    LSTMUINT_KERENL_NAME(BP,  U8,  I8,  F16, )
    LSTMUINT_KERENL_NAME(BP,  U8,  U8,  F16, )

    /* standard + projection */
    LSTMUINT_KERENL_NAME(SP,  F16, F16, F32, )
    LSTMUINT_KERENL_NAME(SP,  F16, F16, F16, )
    /* standard */
    LSTMUINT_KERENL_NAME(S,   F16, F16, F16, )
    LSTMUINT_KERENL_NAME(S,   F16, I16, F16, )
    LSTMUINT_KERENL_NAME(S,   F16, I8,  F16, )
    LSTMUINT_KERENL_NAME(S,   F16, U8,  F16, )
    LSTMUINT_KERENL_NAME(S,   F16, F16, F32, )
    LSTMUINT_KERENL_NAME(S,   F16, I16, F32, )
    LSTMUINT_KERENL_NAME(S,   F16, I8,  F32, )
    LSTMUINT_KERENL_NAME(S,   F16, U8,  F32, )
    /* cifg + standard + projection */
    LSTMUINT_KERENL_NAME(CSP, F16, F16, F32, )
    LSTMUINT_KERENL_NAME(CSP, F16, F16, F16, )
    /* cifg + standard */
    LSTMUINT_KERENL_NAME(CS,  F16, F16, F16, )
    LSTMUINT_KERENL_NAME(CS,  F16, I16, F16, )
    LSTMUINT_KERENL_NAME(CS,  F16, I8,  F16, )
    LSTMUINT_KERENL_NAME(CS,  F16, U8,  F16, )
    LSTMUINT_KERENL_NAME(CS,  F16, F16, F32, )
    LSTMUINT_KERENL_NAME(CS,  F16, I16, F32, )
    LSTMUINT_KERENL_NAME(CS,  F16, I8,  F32, )
    LSTMUINT_KERENL_NAME(CS,  F16, U8,  F32, )
    /* cifg + standard + projection */
    LSTMUINT_KERENL_NAME(CSP, F16, I16, F32, )
    LSTMUINT_KERENL_NAME(CSP, F16, I8,  F32, )
    LSTMUINT_KERENL_NAME(CSP, F16, U8,  F32, )
    LSTMUINT_KERENL_NAME(CSP, F16, I16, F16, )
    LSTMUINT_KERENL_NAME(CSP, F16, I8,  F16, )
    LSTMUINT_KERENL_NAME(CSP, F16, U8,  F16, )
    /* standard + projection */
    LSTMUINT_KERENL_NAME(SP,  F16, I16, F32, )
    LSTMUINT_KERENL_NAME(SP,  F16, I8,  F32, )
    LSTMUINT_KERENL_NAME(SP,  F16, U8,  F32, )
    LSTMUINT_KERENL_NAME(SP,  F16, I16, F16, )
    LSTMUINT_KERENL_NAME(SP,  F16, I8,  F16, )
    LSTMUINT_KERENL_NAME(SP,  F16, U8,  F16, )
    /* standard + projection */
    LSTMUINT_KERENL_NAME(SP,  U8,  F16, F32, )
    LSTMUINT_KERENL_NAME(SP,  U8,  F16, F16, )
    /* standard */
    LSTMUINT_KERENL_NAME(S,   U8,  F16, F16, )
    LSTMUINT_KERENL_NAME(S,   U8,  U8,  F16, )
    LSTMUINT_KERENL_NAME(S,   U8,  F16, F32, )
    LSTMUINT_KERENL_NAME(S,   U8,  U8,  F32, )
    /* cifg + standard + projection */
    LSTMUINT_KERENL_NAME(CSP, U8,  F16, F32, )
    LSTMUINT_KERENL_NAME(CSP, U8,  F16, F16, )
    /* cifg + standard */
    LSTMUINT_KERENL_NAME(CS,  U8,  F16, F16, )
    LSTMUINT_KERENL_NAME(CS,  U8,  U8,  F16, )
    LSTMUINT_KERENL_NAME(CS,  U8,  F16, F32, )
    LSTMUINT_KERENL_NAME(CS,  U8,  U8,  F32, )
    /* cifg + standard + projection */
    LSTMUINT_KERENL_NAME(CSP, U8, I16, F32, )
    LSTMUINT_KERENL_NAME(CSP, U8, I8,  F32, )
    LSTMUINT_KERENL_NAME(CSP, U8, U8,  F32, )
    LSTMUINT_KERENL_NAME(CSP, U8, I16, F16, )
    LSTMUINT_KERENL_NAME(CSP, U8, I8,  F16, )
    LSTMUINT_KERENL_NAME(CSP, U8, U8,  F16, )
    /* standard + projection */
    LSTMUINT_KERENL_NAME(SP,  U8, I16, F32, )
    LSTMUINT_KERENL_NAME(SP,  U8, I8,  F32, )
    LSTMUINT_KERENL_NAME(SP,  U8, U8,  F32, )
    LSTMUINT_KERENL_NAME(SP,  U8, I16, F16, )
    LSTMUINT_KERENL_NAME(SP,  U8, I8,  F16, )
    LSTMUINT_KERENL_NAME(SP,  U8, U8,  F16, )

    /* layer norm + cifg + projection */
    LSTMUINT_KERENL_NAME(CLP, F16, F16, F32, _HARD)
    LSTMUINT_KERENL_NAME(CLP, F16, F16, F16, _HARD)
    /* layer norm + projection + hard_sigmoid */
    LSTMUINT_KERENL_NAME(LP,  F16, F16, F32, _HARD)
    LSTMUINT_KERENL_NAME(LP,  F16, F16, F16, _HARD)
    /* layer norm + cifg + hard_sigmoid */
    LSTMUINT_KERENL_NAME(CL,  F16, F16, F16, _HARD)
    LSTMUINT_KERENL_NAME(CL,  F16, I16, F16, _HARD)
    LSTMUINT_KERENL_NAME(CL,  F16, I8,  F16, _HARD)
    LSTMUINT_KERENL_NAME(CL,  F16, U8,  F16, _HARD)
    LSTMUINT_KERENL_NAME(CL,  F16, F16, F32, _HARD)
    LSTMUINT_KERENL_NAME(CL,  F16, I16, F32, _HARD)
    LSTMUINT_KERENL_NAME(CL,  F16, I8,  F32, _HARD)
    LSTMUINT_KERENL_NAME(CL,  F16, U8,  F32, _HARD)
    /* layer norm + hard_sigmoid */
    LSTMUINT_KERENL_NAME(L,   F16, F16, F16, _HARD)
    LSTMUINT_KERENL_NAME(L,   F16, I16, F16, _HARD)
    LSTMUINT_KERENL_NAME(L,   F16, I8,  F16, _HARD)
    LSTMUINT_KERENL_NAME(L,   F16, U8,  F16, _HARD)
    LSTMUINT_KERENL_NAME(L,   F16, F16, F32, _HARD)
    LSTMUINT_KERENL_NAME(L,   F16, I16, F32, _HARD)
    LSTMUINT_KERENL_NAME(L,   F16, I8,  F32, _HARD)
    LSTMUINT_KERENL_NAME(L,   F16, U8,  F32, _HARD)

    /* layer norm + cifg + projection + hard_sigmoid */
    LSTMUINT_KERENL_NAME(CLP, F16, I16, F32, _HARD)
    LSTMUINT_KERENL_NAME(CLP, F16, I8,  F32, _HARD)
    LSTMUINT_KERENL_NAME(CLP, F16, U8,  F32, _HARD)
    LSTMUINT_KERENL_NAME(CLP, F16, I16, F16, _HARD)
    LSTMUINT_KERENL_NAME(CLP, F16, I8,  F16, _HARD)
    LSTMUINT_KERENL_NAME(CLP, F16, U8,  F16, _HARD)
    /* layer norm + projection + hard_sigmoid */
    LSTMUINT_KERENL_NAME(LP,  F16, I16, F32, _HARD)
    LSTMUINT_KERENL_NAME(LP,  F16, I8,  F32, _HARD)
    LSTMUINT_KERENL_NAME(LP,  F16, U8,  F32, _HARD)
    LSTMUINT_KERENL_NAME(LP,  F16, I16, F16, _HARD)
    LSTMUINT_KERENL_NAME(LP,  F16, I8,  F16, _HARD)
    LSTMUINT_KERENL_NAME(LP,  F16, U8,  F16, _HARD)

    /* hybrid + projection + hard_sigmoid */
    LSTMUINT_KERENL_NAME(BP,  F16, F16, F32, _HARD)
    LSTMUINT_KERENL_NAME(BP,  F16, F16, F16, _HARD)
    /* hybrid + hard_sigmoid */
    LSTMUINT_KERENL_NAME(B,   F16, F16, F16, _HARD)
    LSTMUINT_KERENL_NAME(B,   F16, I16, F16, _HARD)
    LSTMUINT_KERENL_NAME(B,   F16, I8,  F16, _HARD)
    LSTMUINT_KERENL_NAME(B,   F16, U8,  F16, _HARD)
    LSTMUINT_KERENL_NAME(B,   F16, F16, F32, _HARD)
    LSTMUINT_KERENL_NAME(B,   F16, I16, F32, _HARD)
    LSTMUINT_KERENL_NAME(B,   F16, I8,  F32, _HARD)
    LSTMUINT_KERENL_NAME(B,   F16, U8,  F32, _HARD)
    /* cifg + hybrid + projection + hard_sigmoid */
    LSTMUINT_KERENL_NAME(CBP, F16, F16, F32, _HARD)
    LSTMUINT_KERENL_NAME(CBP, F16, F16, F16, _HARD)
    /* cifg + hybrid + hard_sigmoid */
    LSTMUINT_KERENL_NAME(CB,  F16, F16, F16, _HARD)
    LSTMUINT_KERENL_NAME(CB,  F16, I16, F16, _HARD)
    LSTMUINT_KERENL_NAME(CB,  F16, I8,  F16, _HARD)
    LSTMUINT_KERENL_NAME(CB,  F16, U8,  F16, _HARD)
    LSTMUINT_KERENL_NAME(CB,  F16, F16, F32, _HARD)
    LSTMUINT_KERENL_NAME(CB,  F16, I16, F32, _HARD)
    LSTMUINT_KERENL_NAME(CB,  F16, I8,  F32, _HARD)
    LSTMUINT_KERENL_NAME(CB,  F16, U8,  F32, _HARD)
    /* cifg + hybrid + projection + hard_sigmoid */
    LSTMUINT_KERENL_NAME(CBP, F16, I16, F32, _HARD)
    LSTMUINT_KERENL_NAME(CBP, F16, I8,  F32, _HARD)
    LSTMUINT_KERENL_NAME(CBP, F16, U8,  F32, _HARD)
    LSTMUINT_KERENL_NAME(CBP, F16, I16, F16, _HARD)
    LSTMUINT_KERENL_NAME(CBP, F16, I8,  F16, _HARD)
    LSTMUINT_KERENL_NAME(CBP, F16, U8,  F16, _HARD)
    /* hybrid + projection + hard_sigmoid */
    LSTMUINT_KERENL_NAME(BP,  F16, I16, F32, _HARD)
    LSTMUINT_KERENL_NAME(BP,  F16, I8,  F32, _HARD)
    LSTMUINT_KERENL_NAME(BP,  F16, U8,  F32, _HARD)
    LSTMUINT_KERENL_NAME(BP,  F16, I16, F16, _HARD)
    LSTMUINT_KERENL_NAME(BP,  F16, I8,  F16, _HARD)
    LSTMUINT_KERENL_NAME(BP,  F16, U8,  F16, _HARD)
    /* hybrid + projection + hard_sigmoid */
    LSTMUINT_KERENL_NAME(BP,  U8,  F16, F32, _HARD)
    LSTMUINT_KERENL_NAME(BP,  U8,  F16, F16, _HARD)
    /* hybrid + hard_sigmoid */
    LSTMUINT_KERENL_NAME(B,   U8,  F16, F16, _HARD)
    LSTMUINT_KERENL_NAME(B,   U8,  U8,  F16, _HARD)
    LSTMUINT_KERENL_NAME(B,   U8,  F16, F32, _HARD)
    LSTMUINT_KERENL_NAME(B,   U8,  U8,  F32, _HARD)
    /* cifg + hybrid + projection + hard_sigmoid */
    LSTMUINT_KERENL_NAME(CBP, U8,  F16, F32, _HARD)
    LSTMUINT_KERENL_NAME(CBP, U8,  F16, F16, _HARD)
    /* cifg + hybrid + hard_sigmoid */
    LSTMUINT_KERENL_NAME(CB,  U8,  F16, F16, _HARD)
    LSTMUINT_KERENL_NAME(CB,  U8,  U8,  F16, _HARD)
    LSTMUINT_KERENL_NAME(CB,  U8,  F16, F32, _HARD)
    LSTMUINT_KERENL_NAME(CB,  U8,  U8,  F32, _HARD)
    /* cifg + hybrid + projection + hard_sigmoid */
    LSTMUINT_KERENL_NAME(CBP, U8,  I16, F32, _HARD)
    LSTMUINT_KERENL_NAME(CBP, U8,  I8,  F32, _HARD)
    LSTMUINT_KERENL_NAME(CBP, U8,  U8,  F32, _HARD)
    LSTMUINT_KERENL_NAME(CBP, U8,  I16, F16, _HARD)
    LSTMUINT_KERENL_NAME(CBP, U8,  I8,  F16, _HARD)
    LSTMUINT_KERENL_NAME(CBP, U8,  U8,  F16, _HARD)
    /* hybrid + projection + hard_sigmoid */
    LSTMUINT_KERENL_NAME(BP,  U8,  I16, F32, _HARD)
    LSTMUINT_KERENL_NAME(BP,  U8,  I8,  F32, _HARD)
    LSTMUINT_KERENL_NAME(BP,  U8,  U8,  F32, _HARD)
    LSTMUINT_KERENL_NAME(BP,  U8,  I16, F16, _HARD)
    LSTMUINT_KERENL_NAME(BP,  U8,  I8,  F16, _HARD)
    LSTMUINT_KERENL_NAME(BP,  U8,  U8,  F16, _HARD)

    /* standard + projection + hard_sigmoid */
    LSTMUINT_KERENL_NAME(SP,  F16, F16, F32, _HARD)
    LSTMUINT_KERENL_NAME(SP,  F16, F16, F16, _HARD)
    /* standard + hard_sigmoid */
    LSTMUINT_KERENL_NAME(S,   F16, F16, F16, _HARD)
    LSTMUINT_KERENL_NAME(S,   F16, I16, F16, _HARD)
    LSTMUINT_KERENL_NAME(S,   F16, I8,  F16, _HARD)
    LSTMUINT_KERENL_NAME(S,   F16, U8,  F16, _HARD)
    LSTMUINT_KERENL_NAME(S,   F16, F16, F32, _HARD)
    LSTMUINT_KERENL_NAME(S,   F16, I16, F32, _HARD)
    LSTMUINT_KERENL_NAME(S,   F16, I8,  F32, _HARD)
    LSTMUINT_KERENL_NAME(S,   F16, U8,  F32, _HARD)
    /* cifg + standard + projection + hard_sigmoid */
    LSTMUINT_KERENL_NAME(CSP, F16, F16, F32, _HARD)
    LSTMUINT_KERENL_NAME(CSP, F16, F16, F16, _HARD)
    /* cifg + standard + hard_sigmoid */
    LSTMUINT_KERENL_NAME(CS,  F16, F16, F16, _HARD)
    LSTMUINT_KERENL_NAME(CS,  F16, I16, F16, _HARD)
    LSTMUINT_KERENL_NAME(CS,  F16, I8,  F16, _HARD)
    LSTMUINT_KERENL_NAME(CS,  F16, U8,  F16, _HARD)
    LSTMUINT_KERENL_NAME(CS,  F16, F16, F32, _HARD)
    LSTMUINT_KERENL_NAME(CS,  F16, I16, F32, _HARD)
    LSTMUINT_KERENL_NAME(CS,  F16, I8,  F32, _HARD)
    LSTMUINT_KERENL_NAME(CS,  F16, U8,  F32, _HARD)
    /* cifg + standard + projection + hard_sigmoid */
    LSTMUINT_KERENL_NAME(CSP, F16, I16, F32, _HARD)
    LSTMUINT_KERENL_NAME(CSP, F16, I8,  F32, _HARD)
    LSTMUINT_KERENL_NAME(CSP, F16, U8,  F32, _HARD)
    LSTMUINT_KERENL_NAME(CSP, F16, I16, F16, _HARD)
    LSTMUINT_KERENL_NAME(CSP, F16, I8,  F16, _HARD)
    LSTMUINT_KERENL_NAME(CSP, F16, U8,  F16, _HARD)
    /* standard + projection + hard_sigmoid */
    LSTMUINT_KERENL_NAME(SP,  F16, I16, F32, _HARD)
    LSTMUINT_KERENL_NAME(SP,  F16, I8,  F32, _HARD)
    LSTMUINT_KERENL_NAME(SP,  F16, U8,  F32, _HARD)
    LSTMUINT_KERENL_NAME(SP,  F16, I16, F16, _HARD)
    LSTMUINT_KERENL_NAME(SP,  F16, I8,  F16, _HARD)
    LSTMUINT_KERENL_NAME(SP,  F16, U8,  F16, _HARD)
    /* standard + projection + hard_sigmoid */
    LSTMUINT_KERENL_NAME(SP,  U8,  F16, F32, _HARD)
    LSTMUINT_KERENL_NAME(SP,  U8,  F16, F16, _HARD)
    /* standard + hard_sigmoid */
    LSTMUINT_KERENL_NAME(S,   U8,  F16, F16, _HARD)
    LSTMUINT_KERENL_NAME(S,   U8,  U8,  F16, _HARD)
    LSTMUINT_KERENL_NAME(S,   U8,  F16, F32, _HARD)
    LSTMUINT_KERENL_NAME(S,   U8,  U8,  F32, _HARD)
    /* cifg + standard + projection + hard_sigmoid */
    LSTMUINT_KERENL_NAME(CSP, U8,  F16, F32, _HARD)
    LSTMUINT_KERENL_NAME(CSP, U8,  F16, F16, _HARD)
    /* cifg + standard + hard_sigmoid */
    LSTMUINT_KERENL_NAME(CS,  U8,  F16, F16, _HARD)
    LSTMUINT_KERENL_NAME(CS,  U8,  U8,  F16, _HARD)
    LSTMUINT_KERENL_NAME(CS,  U8,  F16, F32, _HARD)
    LSTMUINT_KERENL_NAME(CS,  U8,  U8,  F32, _HARD)
    /* cifg + standard + projection + hard_sigmoid */
    LSTMUINT_KERENL_NAME(CSP, U8, I16,  F32, _HARD)
    LSTMUINT_KERENL_NAME(CSP, U8, I8,   F32, _HARD)
    LSTMUINT_KERENL_NAME(CSP, U8, U8,   F32, _HARD)
    LSTMUINT_KERENL_NAME(CSP, U8, I16,  F16, _HARD)
    LSTMUINT_KERENL_NAME(CSP, U8, I8,   F16, _HARD)
    LSTMUINT_KERENL_NAME(CSP, U8, U8,   F16, _HARD)
    /* standard + projection + hard_sigmoid */
    LSTMUINT_KERENL_NAME(SP,  U8, I16,  F32, _HARD)
    LSTMUINT_KERENL_NAME(SP,  U8, I8,   F32, _HARD)
    LSTMUINT_KERENL_NAME(SP,  U8, U8,   F32, _HARD)
    LSTMUINT_KERENL_NAME(SP,  U8, I16,  F16, _HARD)
    LSTMUINT_KERENL_NAME(SP,  U8, I8,   F16, _HARD)
    LSTMUINT_KERENL_NAME(SP,  U8, U8,   F16, _HARD)

    NULL
};
#ifdef __cplusplus
}
#endif
