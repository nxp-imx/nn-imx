/****************************************************************************
*
*    Copyright (c) 2021 Vivante Corporation
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

#include "vsi_nn_types.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_node.h"
#include "vsi_nn_log.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_error.h"
#include "vsi_nn_kernel_prv.h"
#include "kernel/vsi_nn_kernel.h"
#include "kernel/vsi_nn_sp_unit_operation.h"
#include "kernel/vsi_nn_sp_lut.h"

#if (VX_STREAM_PROCESSOR_SUPPORT && VX_ACTIVATION_EXT2_SUPPORT)

static vsi_nn_kernel_node_t _setup
    (
    vsi_nn_graph_t              * graph,
    vsi_nn_tensor_t            ** inputs,
    size_t                        input_num,
    vsi_nn_tensor_t            ** outputs,
    size_t                        output_num,
    const vsi_nn_kernel_param_t * params,
    vsi_nn_kernel_t             * kernel,
    const vx_enum                 act_function
    )
{
    vx_node node = NULL;
    vsi_nn_kernel_dtype_e in_dtype =  vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    float a_v = 0;
    float b_v = 0;

    if (act_function == VX_CONVOLUTIONAL_NETWORK_ACTIVATION_CLIP)
    {
        a_v = vsi_nn_kernel_param_get_float32( params, "min_value" );
        b_v = vsi_nn_kernel_param_get_float32( params, "max_value" );
    }
    else if (act_function != VX_CONVOLUTIONAL_NETWORK_ACTIVATION_ERF)
    {
        a_v = vsi_nn_kernel_param_get_float32( params, "alpha" );
        b_v = vsi_nn_kernel_param_get_float32( params, "beta" );
    }

    switch ( act_function )
    {
    case VX_CONVOLUTIONAL_NETWORK_ACTIVATION_CLIP:
    case VX_CONVOLUTIONAL_NETWORK_ACTIVATION_HSIGMOID:
    case VX_CONVOLUTIONAL_NETWORK_ACTIVATION_NEG:
    case VX_CONVOLUTIONAL_NETWORK_ACTIVATION_SIGN:
        break;
    default:
        if (in_dtype != I8 &&
            in_dtype != U8)
        {
            return NULL;
        }
        break;
    }

    node = vxActivationLayer(
        graph->g,
        inputs[0]->t,
        act_function,
        a_v,
        b_v,
        outputs[0]->t
        );

    return (vsi_nn_kernel_node_t)node;
}

#define REGISTER_ELTWISE_UNARY_BACKEND_SP(KERNEL_NAME, ACT_FUNC) \
    static vsi_nn_kernel_node_t _##KERNEL_NAME##_setup \
        ( \
        vsi_nn_graph_t              * graph, \
        vsi_nn_tensor_t            ** inputs, \
        size_t                        input_num, \
        vsi_nn_tensor_t            ** outputs, \
        size_t                        output_num, \
        const vsi_nn_kernel_param_t * params, \
        vsi_nn_kernel_t             * kernel \
        ) \
    { \
        return _setup(graph, inputs, input_num, outputs, output_num, \
                params, kernel, ACT_FUNC); \
    } \
    REGISTER_BACKEND_STREAM_PROCESSOR( KERNEL_NAME, _##KERNEL_NAME##_setup )

REGISTER_ELTWISE_UNARY_BACKEND_SP( sign, VX_CONVOLUTIONAL_NETWORK_ACTIVATION_SIGN )
REGISTER_ELTWISE_UNARY_BACKEND_SP( hard_sigmoid, VX_CONVOLUTIONAL_NETWORK_ACTIVATION_HSIGMOID )
REGISTER_ELTWISE_UNARY_BACKEND_SP( neg,  VX_CONVOLUTIONAL_NETWORK_ACTIVATION_NEG )
REGISTER_ELTWISE_UNARY_BACKEND_SP( clip, VX_CONVOLUTIONAL_NETWORK_ACTIVATION_CLIP )
REGISTER_ELTWISE_UNARY_BACKEND_SP( exp,  VX_CONVOLUTIONAL_NETWORK_ACTIVATION_EXP )
REGISTER_ELTWISE_UNARY_BACKEND_SP( sin,  VX_CONVOLUTIONAL_NETWORK_ACTIVATION_SIN )
REGISTER_ELTWISE_UNARY_BACKEND_SP( cos,  VX_CONVOLUTIONAL_NETWORK_ACTIVATION_COS )
REGISTER_ELTWISE_UNARY_BACKEND_SP( log,  VX_CONVOLUTIONAL_NETWORK_ACTIVATION_LOG )
REGISTER_ELTWISE_UNARY_BACKEND_SP( mish, VX_CONVOLUTIONAL_NETWORK_ACTIVATION_MISH )
REGISTER_ELTWISE_UNARY_BACKEND_SP( gelu,  VX_CONVOLUTIONAL_NETWORK_ACTIVATION_GELU )
REGISTER_ELTWISE_UNARY_BACKEND_SP( hard_gelu, VX_CONVOLUTIONAL_NETWORK_ACTIVATION_HGELU )
REGISTER_ELTWISE_UNARY_BACKEND_SP( elu,  VX_CONVOLUTIONAL_NETWORK_ACTIVATION_SELU )
REGISTER_ELTWISE_UNARY_BACKEND_SP( selu, VX_CONVOLUTIONAL_NETWORK_ACTIVATION_SELU )
REGISTER_ELTWISE_UNARY_BACKEND_SP( celu, VX_CONVOLUTIONAL_NETWORK_ACTIVATION_CELU )
REGISTER_ELTWISE_UNARY_BACKEND_SP( rcp,  VX_CONVOLUTIONAL_NETWORK_ACTIVATION_RECIPROCAL )
REGISTER_ELTWISE_UNARY_BACKEND_SP( softsign, VX_CONVOLUTIONAL_NETWORK_ACTIVATION_SOFTSIGN )
REGISTER_ELTWISE_UNARY_BACKEND_SP( atan, VX_CONVOLUTIONAL_NETWORK_ACTIVATION_ATAN )
REGISTER_ELTWISE_UNARY_BACKEND_SP( atanh, VX_CONVOLUTIONAL_NETWORK_ACTIVATION_ATANH )
REGISTER_ELTWISE_UNARY_BACKEND_SP( acosh, VX_CONVOLUTIONAL_NETWORK_ACTIVATION_ACOSH )
REGISTER_ELTWISE_UNARY_BACKEND_SP( inverse_sigmoid, VX_CONVOLUTIONAL_NETWORK_ACTIVATION_INVERSE_SIGMOID )
REGISTER_ELTWISE_UNARY_BACKEND_SP( round, VX_CONVOLUTIONAL_NETWORK_ACTIVATION_ROUND )
REGISTER_ELTWISE_UNARY_BACKEND_SP( erf, VX_CONVOLUTIONAL_NETWORK_ACTIVATION_ERF )

#undef REGISTER_ELTWISE_UNARY_BACKEND_SP
#endif
