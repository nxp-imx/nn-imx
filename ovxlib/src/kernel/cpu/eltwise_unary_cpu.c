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

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "vsi_nn_types.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_log.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_error.h"
#include "kernel/vsi_nn_kernel.h"

__BEGIN_DECLS

/** Unary Kernel internal type */
typedef enum
{
    UNARY_SIN,
    UNARY_COS,
    UNARY_EXP,
    UNARY_LOG,
    UNARY_NEG,
    UNARY_HSIGMOID,
    UNARY_MISH,
    UNARY_ROUND,
    UNARY_GELU,
    UNARY_HGELU,
    UNARY_SELU,
    UNARY_CELU,
    UNARY_RCP,
    UNARY_SIGN,
    UNARY_SOFTSIGN,
    UNARY_ATAN,
    UNARY_ATANH,
    UNARY_ACOSH,
    UNARY_INVERSE_SIGMOID,
} unary_type_e;


#define _CPU_ARG_NUM            (3)
#define _CPU_INPUT_NUM          (1)
#define _CPU_OUTPUT_NUM         (1)
#define _CPU_IO_NUM             (_CPU_INPUT_NUM + _CPU_OUTPUT_NUM)
#define _CPU_PARAM_NUM          (_CPU_ARG_NUM + _CPU_IO_NUM)
#define _KERNEL_NAME            CVIVANTE_NAMESPACE("eltwise_unary_sw")

static float exp_eval(float data)
{
    return expf(data);
}

static float sin_eval(float data)
{
    return sinf(data);
}

static float cos_eval(float data)
{
    return cosf(data);
}

static float log_eval(float data)
{
    return logf(data);
}

static float neg_eval(float data)
{
    return data * -1.0f;
}

static float hsigmoid_eval(float data, float alpha, float beta)
{
    data = (float)(alpha * data + beta);
    data = vsi_nn_clamp(data, 0, 1);

    return data;
}

static float soft_plus_eval(float data)
{
    return log_eval(exp_eval(data) + 1);
}

static float mish_eval(float data)
{
    data = (float)(data * tanh(soft_plus_eval(data)));

    return data;
}

static float round_eval(float data)
{
    data = (float)(vsi_rtne(data));

    return data;
}

static float gelu_eval(float data)
{
    data = (float)(0.5f * data * (1 + vsi_nn_erf_impl(data / (float)sqrt(2.0f))));

    return data;
}

#define VSI_SQRT_2_RCP_PI  0.7978845834732056f
static float hgelu_eval(float data)
{
    float cdf = (float)(0.5f * (1.0f + tanh((VSI_SQRT_2_RCP_PI *
        (data + 0.044715f * data * data * data)))));

    return data * cdf;
}

static float selu_eval(float data, float alpha, float gamma)
{
    float y0 = alpha * gamma * expf(data) - alpha * gamma;
    float y1 = gamma * data;
    float y = data <= 0 ? y0 : y1;

    return y;
}

static float celu_eval(float x, float alpha)
{
    float positive = vsi_nn_max(0, x);
    float negative = vsi_nn_min(alpha * (expf(x / alpha) - 1), 0);

    return positive + negative;
}

static float rcp_eval(float x)
{
    return 1 / x;
}

static float sign_eval(float x)
{
    return x > 0 ? 1.0f : x < 0 ? -1.0f : 0;
}

static float atan_eval(float x)
{
    return atanf(x);
}

static float softsign_eval(float x)
{
    return x / (1.0f + vsi_abs(x));
}

static float atanh_eval(float x)
{
    return (log_eval(1 + x) - log_eval(1 - x)) / 2;
}

static float acosh_eval(float x)
{
    return (log_eval(x + (float)sqrt(x * x - 1)));
}

static float inverse_sigmoid_eval(float x, float eps)
{
    float x1,x2;
    x = vsi_nn_clamp(x, 0, 1);
    x1 = vsi_nn_clamp(x, eps, 1);
    x2 = vsi_nn_clamp((1 - x), eps, 1);

    return log_eval(x1 / x2);
}

DEF_KERNEL_EXECUTOR(_eltwise_unary_exec)
    (
    vsi_nn_kernel_node_t node,
    const vsi_nn_kernel_node_param_t * param,
    size_t param_size
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_tensor_t tensors[_CPU_IO_NUM] = { NULL };
    float * buffer[_CPU_IO_NUM] = { NULL };
    size_t out_elements = 0;
    vsi_nn_kernel_tensor_attr_t * attr[_CPU_IO_NUM] = { NULL };
    int32_t i;
    float alpha = 0;
    float beta = 0;
    int32_t unary_type = 0;

    VSI_UNREFERENCED(node);
    VSI_UNREFERENCED(param_size);

    tensors[0]  = (vsi_nn_kernel_tensor_t)param[0];
    tensors[1]  = (vsi_nn_kernel_tensor_t)param[1];

    attr[0] = vsi_nn_kernel_tensor_attr_create( tensors[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( tensors[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[2], &unary_type);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[3], &alpha);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[4], &beta);
    CHECK_STATUS_FAIL_GOTO(status, final );

    buffer[0] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[0], attr[0], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[0], "Create input buffer fail.", final );

    out_elements = vsi_nn_kernel_tensor_attr_get_size( attr[1] );
    buffer[1] = (float *)malloc( out_elements * sizeof(float) );
    CHECK_PTR_FAIL_GOTO( buffer[1], "Create output buffer fail.", final );
    memset( buffer[1], 0, out_elements * sizeof(float) );

    for ( i = 0; i < (int32_t)out_elements; ++i)
    {
        float data = buffer[0][i];

        switch (unary_type)
        {
        case UNARY_SIN:
            data = sin_eval(data);
            break;
        case UNARY_COS:
            data = cos_eval(data);
            break;
        case UNARY_EXP:
            data = exp_eval(data);
            break;
        case UNARY_LOG:
            data = log_eval(data);
            break;
        case UNARY_NEG:
            data = neg_eval(data);
            break;
        case UNARY_HSIGMOID:
            data = hsigmoid_eval(data, alpha, beta);
            break;
        case UNARY_MISH:
            data = mish_eval(data);
            break;
        case UNARY_ROUND:
            data = round_eval(data);
            break;
        case UNARY_GELU:
            data = gelu_eval(data);
            break;
        case UNARY_HGELU:
            data = hgelu_eval(data);
            break;
        case UNARY_SELU:
            data = selu_eval(data, alpha, beta);
            break;
        case UNARY_CELU:
            data = celu_eval(data, alpha);
            break;
        case UNARY_RCP:
            data = rcp_eval(data);
            break;
        case UNARY_SIGN:
            data = sign_eval(data);
            break;
        case UNARY_SOFTSIGN:
            data = softsign_eval(data);
            break;
        case UNARY_ATAN:
            data = atan_eval(data);
            break;
        case UNARY_ATANH:
            data = atanh_eval(data);
            break;
        case UNARY_ACOSH:
            data = acosh_eval(data);
            break;
        case UNARY_INVERSE_SIGMOID:
            data = inverse_sigmoid_eval(data, alpha);
            break;
        default:
            break;
        }
        buffer[1][i] = (float)data;
    }

    status = vsi_nn_kernel_tensor_write_from_float( tensors[1], attr[1],
            buffer[1], out_elements );
    CHECK_STATUS_FAIL_GOTO( status, final );

final:
#define SAFE_FREE_TENSOR_ATTR(_PTR) if( _PTR ) { vsi_nn_kernel_tensor_attr_release( &_PTR ); _PTR = NULL; }
    SAFE_FREE_TENSOR_ATTR(attr[0]);
    SAFE_FREE_TENSOR_ATTR(attr[1]);
#undef SAFE_FREE_TENSOR_ATTR
    for( i = 0; i < _CPU_IO_NUM; i ++ )
    {
        if( buffer[i] )
        {
            free( buffer[i] );
            buffer[i] = NULL;
        }
    }
    return status;
} /* _eltwise_unary_exec() */

static vx_param_description_t kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

#define INPUT_FUNC_TYPE           (2)
#define INPUT_SCALAR_ALPHA        (3)
#define INPUT_SCALAR_BETA         (4)

static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_nn_kernel_t* kernel
    )
{
    VSI_UNREFERENCED(inputs);
    VSI_UNREFERENCED(outputs);
    snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _KERNEL_NAME );
    kernel->info.function    = _eltwise_unary_exec;
    kernel->info.parameters  = kernel_param_def;
    kernel->info.numParams   = _cnt_of_array( kernel_param_def );

    return VSI_SUCCESS;
} /* _query_kernel() */

static vsi_nn_kernel_node_t _setup
    (
    vsi_nn_graph_t              * graph,
    vsi_nn_tensor_t            ** inputs,
    size_t                        input_num,
    vsi_nn_tensor_t            ** outputs,
    size_t                        output_num,
    const vsi_nn_kernel_param_t * params,
    vsi_nn_kernel_t             * kernel,
    const unary_type_e            unary_type
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_kernel_node_param_t backend_params[_CPU_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    float alpha = vsi_nn_kernel_param_get_float32( params, "alpha" );
    float beta = vsi_nn_kernel_param_get_float32( params, "beta" );

    VSI_UNREFERENCED(input_num);
    VSI_UNREFERENCED(output_num);

    status = _query_kernel( inputs, outputs, kernel );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( backend_params, _CPU_PARAM_NUM,
                    inputs, _CPU_INPUT_NUM, outputs, _CPU_OUTPUT_NUM );
            backend_params[INPUT_FUNC_TYPE] = vsi_nn_kernel_scalar_create(
                    graph, I32, &unary_type );
            backend_params[INPUT_SCALAR_ALPHA] = vsi_nn_kernel_scalar_create(
                    graph, F32, &alpha );
            backend_params[INPUT_SCALAR_BETA] = vsi_nn_kernel_scalar_create(
                    graph, F32, &beta );
            /* Pass parameters to node. */
            status = vsi_nn_kernel_node_pass_param( node, backend_params, _CPU_PARAM_NUM );

            vsi_nn_kernel_scalar_release( &backend_params[INPUT_FUNC_TYPE] );
            vsi_nn_kernel_scalar_release( &backend_params[INPUT_SCALAR_ALPHA] );
            vsi_nn_kernel_scalar_release( &backend_params[INPUT_SCALAR_BETA] );
        }
        else
        {
            status = VSI_FAILURE;
        }
    }

    return node;
} /* _setup() */

#define REGISTER_ELTWISE_UNARY_BACKEND_CPU(KERNEL_NAME, UNARY_TYPE) \
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
                params, kernel, UNARY_TYPE); \
    } \
    REGISTER_BACKEND_CPU( KERNEL_NAME, _##KERNEL_NAME##_setup )

__END_DECLS

REGISTER_ELTWISE_UNARY_BACKEND_CPU( sin,          UNARY_SIN )
REGISTER_ELTWISE_UNARY_BACKEND_CPU( cos,          UNARY_COS )
REGISTER_ELTWISE_UNARY_BACKEND_CPU( exp,          UNARY_EXP )
REGISTER_ELTWISE_UNARY_BACKEND_CPU( log,          UNARY_LOG )
REGISTER_ELTWISE_UNARY_BACKEND_CPU( neg,          UNARY_NEG )
REGISTER_ELTWISE_UNARY_BACKEND_CPU( hard_sigmoid, UNARY_HSIGMOID )
REGISTER_ELTWISE_UNARY_BACKEND_CPU( mish,         UNARY_MISH )
REGISTER_ELTWISE_UNARY_BACKEND_CPU( round,        UNARY_ROUND )
REGISTER_ELTWISE_UNARY_BACKEND_CPU( gelu,         UNARY_GELU )
REGISTER_ELTWISE_UNARY_BACKEND_CPU( hard_gelu,    UNARY_HGELU )
REGISTER_ELTWISE_UNARY_BACKEND_CPU( selu,         UNARY_SELU )
REGISTER_ELTWISE_UNARY_BACKEND_CPU( celu,         UNARY_CELU )
REGISTER_ELTWISE_UNARY_BACKEND_CPU( rcp,          UNARY_RCP )
REGISTER_ELTWISE_UNARY_BACKEND_CPU( sign,         UNARY_SIGN )
REGISTER_ELTWISE_UNARY_BACKEND_CPU( softsign,     UNARY_SOFTSIGN )
REGISTER_ELTWISE_UNARY_BACKEND_CPU( atan,         UNARY_ATAN )
REGISTER_ELTWISE_UNARY_BACKEND_CPU( atanh,        UNARY_ATANH )
REGISTER_ELTWISE_UNARY_BACKEND_CPU( acosh,        UNARY_ACOSH )
REGISTER_ELTWISE_UNARY_BACKEND_CPU( inverse_sigmoid, UNARY_INVERSE_SIGMOID )