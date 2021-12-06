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

#include "vsi_nn_types.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_node.h"
#include "vsi_nn_log.h"
#include "vsi_nn_prv.h"
#include <float.h>
#include "utils/vsi_nn_dtype_util_prv.h"
#include "vsi_nn_tensor_util.h"
#include "kernel/vsi_nn_kernel.h"

typedef struct _sort_lut_s
{
    float index;
    float val;
} sort_lut;

typedef struct _vsi_nn_lut_param_s
{
    float alpha;
    float beta;
} vsi_nn_lut_param;

static float exp_eval(float val, vsi_nn_lut_param lut_param)
{
    return expf(val);
}

static float log_eval(float data, vsi_nn_lut_param lut_param)
{
    return logf(data);
}

static float elu_eval(float data, vsi_nn_lut_param lut_param)
{
    float alpha = lut_param.alpha;
    return data >=0 ? data : expf(data) * alpha - alpha;
}

static float neg_eval(float data, vsi_nn_lut_param lut_param)
{
    return data * -1.0f;
}

static float hsigmoid_eval(float data, vsi_nn_lut_param lut_param)
{
    float alpha = lut_param.alpha;
    float beta = lut_param.beta;

    data = (float)(alpha * data + beta);
    data = vsi_nn_clamp(data, 0, 1);

    return data;
}

static float soft_plus_eval(float data, vsi_nn_lut_param lut_param)
{
    return log_eval(exp_eval(data, lut_param) + 1, lut_param);
}

static float mish_eval(float data, vsi_nn_lut_param lut_param)
{
    data = (float)(data * tanh(soft_plus_eval(data, lut_param)));

    return data;
}

static float erf_eval(float x)
{
    float res = 0;
    float tmp = x;
    float factorial = 1; /*n!*/
    float x_pow = x;
    int32_t one = 1;
    int32_t n = 1;

    if (x <= -3)
    {
        return -1;
    }
    else if (x >= 3)
    {
        return 1;
    }

    while (vsi_abs(tmp) > 1e-5)
    {
        res += tmp;

        factorial *= n;
        one *= -1;
        x_pow *= x * x;
        tmp = one / factorial * x_pow / ( 2 * n + 1);

        n ++;
    }
#define VSI_MUL2_RSQRTPI    (1.1283791670955126f)

    res *= VSI_MUL2_RSQRTPI;

    return res;
}

static float gelu_eval(float data, vsi_nn_lut_param lut_param)
{
    data = (float)(0.5f * data * (1 + erf_eval(data / (float)sqrt(2.0f))));

    return data;
}

#define VSI_SQRT_2_RCP_PI  0.7978845834732056f
static float hgelu_eval(float data, vsi_nn_lut_param lut_param)
{
    float cdf = (float)(0.5f * (1.0f + tanh((VSI_SQRT_2_RCP_PI *
        (data + 0.044715f * data * data * data)))));

    return data * cdf;
}

#ifdef VX_USER_LOOKUP_TABLE_SUPPORT
static int32_t _lut_comparator(const void *pa, const void *pb)
{
    sort_lut a = *(sort_lut *)pa;
    sort_lut b = *(sort_lut *)pb;
    float diff = a.index - b.index;
    if ( diff > 0 )
    {
        return 1;
    }
    else if ( diff < 0 )
    {
        return -1;
    }

    return 0;
}

static void _set_unary_table_lookup
    (
    float func(float,
    vsi_nn_lut_param),
    float *index,
    float *value,
    vsi_nn_lut_param lut_param
    )
{
#define VSI_NN_MAX_LUT_SIZE     (1024)
#define FLT16_MAX               (57344)
#define FLT16_MIN               (-57344)
    uint32_t i = 0;
    sort_lut *lut = (sort_lut *)calloc(VSI_NN_MAX_LUT_SIZE, sizeof(sort_lut));

    for ( i = 0; i < VSI_NN_MAX_LUT_SIZE; i++)
    {
        int16_t val = (int16_t)(i << 6);
        lut[i].index = fp16_to_fp32(val);
        lut[i].val = func(lut[i].index, lut_param);
    }

    for (i = 0x0; i < 0x10; i++)
    {
        lut[i].index = 0;
        lut[i].val = func(lut[i].index, lut_param);
    }

    for (i = 0x1F0; i < 0x200; i++)
    {
        lut[i].index = FLT16_MAX;
        lut[i].val = func(lut[i].index, lut_param);
    }

    for (i = 0x3F0; i < 0x400; i++)
    {
        lut[i].index = FLT16_MIN;
        lut[i].val = func(lut[i].index, lut_param);
    }

    qsort(lut, VSI_NN_MAX_LUT_SIZE, sizeof(sort_lut), _lut_comparator);

    for ( i = 0; i < VSI_NN_MAX_LUT_SIZE; i++)
    {
        index[i] = lut[i].index;
        value[i] = lut[i].val;
    }

    vsi_nn_safe_free(lut);

#undef VSI_NN_MAX_LUT_SIZE
#undef FLT16_MIN
#undef FLT16_MAX
}
#endif

static vsi_nn_kernel_node_t _setup
    (
    vsi_nn_graph_t              * graph,
    vsi_nn_tensor_t            ** inputs,
    size_t                        input_num,
    vsi_nn_tensor_t            ** outputs,
    size_t                        output_num,
    const vsi_nn_kernel_param_t * params,
    vsi_nn_kernel_t             * kernel,
    float                      func(float, vsi_nn_lut_param)
    )
{
#ifdef VX_USER_LOOKUP_TABLE_SUPPORT
    vx_lut lut1 = NULL;
    vx_lut lut2 = NULL;
    vx_node node = NULL;
    vsi_nn_lut_param lut_param;
    float alpha = vsi_nn_kernel_param_get_float32( params, "alpha" );
    float beta = vsi_nn_kernel_param_get_float32( params, "beta" );
    float index[1024] = {0};
    float value[1024] = {0};

    if ( inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_INT32   ||
         outputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_INT32  )
    {
        return NULL;
    }

    lut_param.alpha = alpha;
    lut_param.beta = beta;

    _set_unary_table_lookup(func, index, value, lut_param);

    lut1 = vxCreateLUT( graph->ctx->c, VX_TYPE_FLOAT32, 1024);
    lut2 = vxCreateLUT( graph->ctx->c, VX_TYPE_FLOAT32, 1024);
    if( NULL == lut1 || NULL == lut2 )
    {
        VSILOGE("create lut object fail.");
        goto OnError;
    }

    vxCopyLUT(lut1, (void*)&index, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyLUT(lut2, (void*)&value, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);

    node = vxTensorTableLookupLayer( graph->g, inputs[0]->t, lut1, lut2, outputs[0]->t);
    if( NULL == node )
    {
        VSILOGW("Call vxTensorTableLookupLayer fail.");
        goto OnError;
    }

OnError:
    if (lut1)
    {
        vxReleaseLUT(&lut1);
        lut1 = NULL;
    }
    if (lut2)
    {
        vxReleaseLUT(&lut2);
        lut2 = NULL;
    }

    return (vsi_nn_kernel_node_t)node;
#else
    return NULL;
#endif
} /* _setup() */

#define REGISTER_ELTWISE_UNARY_LUT_OPENVX_KERNEL(KERNEL_NAME, UNARY_FUNC) \
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
                params, kernel, UNARY_FUNC); \
    } \
    REGISTER_BACKEND_OPENVX( KERNEL_NAME, _##KERNEL_NAME##_setup )

REGISTER_ELTWISE_UNARY_LUT_OPENVX_KERNEL( mish,         mish_eval )
//REGISTER_ELTWISE_UNARY_LUT_OPENVX_KERNEL( exp,          exp_eval )
REGISTER_ELTWISE_UNARY_LUT_OPENVX_KERNEL( log,          log_eval )
REGISTER_ELTWISE_UNARY_LUT_OPENVX_KERNEL( elu,          elu_eval )
REGISTER_ELTWISE_UNARY_LUT_OPENVX_KERNEL( neg,          neg_eval )
REGISTER_ELTWISE_UNARY_LUT_OPENVX_KERNEL( hard_sigmoid, hsigmoid_eval )
REGISTER_ELTWISE_UNARY_LUT_OPENVX_KERNEL( gelu,         gelu_eval )
REGISTER_ELTWISE_UNARY_LUT_OPENVX_KERNEL( hard_gelu,    hgelu_eval )

#undef REGISTER_ELTWISE_UNARY_LUT_OPENVX_KERNEL

#define REGISTER_ELTWISE_UNARY_OPENVX_KERNEL( kernel_name )   \
    static vsi_nn_kernel_node_t _##kernel_name##setup \
        ( \
        vsi_nn_graph_t              * graph, \
        vsi_nn_tensor_t            ** inputs, \
        size_t                        input_num, \
        vsi_nn_tensor_t            ** outputs, \
        size_t                        output_num,\
        const vsi_nn_kernel_param_t * params, \
        vsi_nn_kernel_t             * kernel \
        ); \
    REGISTER_BACKEND_OPENVX( kernel_name, _##kernel_name##setup ) \
    static vsi_nn_kernel_node_t _##kernel_name##setup \
        ( \
        vsi_nn_graph_t              * graph, \
        vsi_nn_tensor_t            ** inputs, \
        size_t                        input_num, \
        vsi_nn_tensor_t            ** outputs, \
        size_t                        output_num,\
        const vsi_nn_kernel_param_t * params, \
        vsi_nn_kernel_t             * kernel \
        )

REGISTER_ELTWISE_UNARY_OPENVX_KERNEL( abs )
{
    vx_node node = NULL;
    vsi_size_t input_size[VSI_NN_MAX_DIM_NUM] = {0};
    uint32_t dims = 0;
    vx_tensor input = NULL, input0 = NULL;
    vx_tensor output = NULL, output0 = NULL;

    if (inputs[0]->attr.dim_num > 4)
    {
        input_size[0] = vsi_nn_GetElementNum(inputs[0]) /
            inputs[0]->attr.size[inputs[0]->attr.dim_num - 1];
        input_size[1] = inputs[0]->attr.size[inputs[0]->attr.dim_num - 1];
        dims = 2;
#ifdef VSI_40BIT_VA_SUPPORT
        input = vxReshapeTensor(inputs[0]->t, input_size, dims);
        output = vxReshapeTensor(outputs[0]->t, input_size, dims);
#else
        input = vxReshapeTensor(inputs[0]->t, (vx_int32*)input_size, (vx_uint32)dims);
        output = vxReshapeTensor(outputs[0]->t, (vx_int32*)input_size, (vx_uint32)dims);
#endif
        input0 = input;
        output0 = output;
    }
    else
    {
        input0 = inputs[0]->t;
        output0 = outputs[0]->t;
    }

    node = vxLeakyReluLayer(
        graph->g,
        input0,
        -1,
        output0
        );

    if (input)  vxReleaseTensor(&input);
    if (output) vxReleaseTensor(&output);

    return (vsi_nn_kernel_node_t)node;
} /* abs() */

REGISTER_ELTWISE_UNARY_OPENVX_KERNEL( linear )
{
    vx_node node = NULL;
    float a_v = vsi_nn_kernel_param_get_float32( params, "a_v" );
    float b_v = vsi_nn_kernel_param_get_float32( params, "b_v" );

    node = vxActivationLayer(
        graph->g,
        inputs[0]->t,
        VX_CONVOLUTIONAL_NETWORK_ACTIVATION_LINEAR,
        a_v,
        b_v,
        outputs[0]->t
        );

    return (vsi_nn_kernel_node_t)node;
} /* linear() */

REGISTER_ELTWISE_UNARY_OPENVX_KERNEL( sigmoid )
{
    vx_node node = NULL;

    node = vxActivationLayer(
        graph->g,
        inputs[0]->t,
        VX_CONVOLUTIONAL_NETWORK_ACTIVATION_LOGISTIC,
        0,
        0,
        outputs[0]->t
        );

    return (vsi_nn_kernel_node_t)node;
} /* sigmoid() */

REGISTER_ELTWISE_UNARY_OPENVX_KERNEL( tanh )
{
    vx_node node = NULL;
    float scale_a = vsi_nn_kernel_param_get_float32( params, "scale_a" );
    float scale_b = vsi_nn_kernel_param_get_float32( params, "scale_b" );

    node = vxActivationLayer(
        graph->g,
        inputs[0]->t,
        VX_CONVOLUTIONAL_NETWORK_ACTIVATION_HYPERBOLIC_TAN,
        scale_a,
        scale_b,
        outputs[0]->t
        );

    return (vsi_nn_kernel_node_t)node;
} /* tanh() */

#undef REGISTER_ELTWISE_UNARY_OPENVX_KERNEL
