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
#include "vsi_nn_tensor_util_prv.h"
#include "vsi_nn_kernel_prv.h"
#include "kernel/vsi_nn_kernel.h"

#if (VX_STREAM_PROCESSOR_SUPPORT) && (VX_LSTM_ACTIVATION_SUPPORT)

#define REGISTER_LSTMUNIT_ACTIVATION_STREAM_PROCESSOR_KERNEL( kernel_name )   \
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
    REGISTER_BACKEND_STREAM_PROCESSOR( kernel_name, _##kernel_name##setup ) \
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

REGISTER_LSTMUNIT_ACTIVATION_STREAM_PROCESSOR_KERNEL(lstmunit_activation) {
    vsi_nn_kernel_node_t nodes[2] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    int32_t _is_ln = 0;
    int32_t _is_cifg = 0;
    int32_t _is_proj = 0;
    int32_t _is_hybrid = 0;
    int32_t _is_peephole = 0;
    int32_t recurrent_activation;
    float forget_bias;
    vsi_nn_kernel_dtype_e in_dtype = vsi_nn_kernel_map_dtype(
        inputs[LSTMUNIT_ACT_INPUT_FC_F]->attr.dtype.vx_type);
    vx_nn_lstm_activation_params_t lstm_activation_param = {0};
    vx_tensor inputs_tensor[LSTMUNIT_ACT_INPUTS_COUNT] = {NULL};
    vx_tensor outputs_tensor[2] = {NULL};
    size_t i = 0;
    uint32_t input_real_num = 0;
    for (i = 0; i < input_num; i++)
    {
        if (inputs[i] && inputs[i]->t)
        {
            inputs_tensor[input_real_num] = inputs[i]->t;

            input_real_num++;
        }
    }

    outputs_tensor[0] = outputs[0]->t;
    outputs_tensor[1] = outputs[1]->t;
    VSI_UNREFERENCED(kernel);
    VSI_UNREFERENCED(output_num);

    _is_ln = vsi_nn_kernel_param_get_int32(params, "_is_ln");
    _is_cifg = vsi_nn_kernel_param_get_int32(params, "_is_cifg");
    _is_proj = vsi_nn_kernel_param_get_int32(params, "_is_proj");
    _is_hybrid = vsi_nn_kernel_param_get_int32(params, "_is_hybrid");
    _is_peephole = vsi_nn_kernel_param_get_int32(params, "_is_peephole");
    recurrent_activation =
        vsi_nn_kernel_param_get_int32(params, "recurrent_activation");
    forget_bias = vsi_nn_kernel_param_get_float32(params, "forget_bias");

    if (_is_hybrid || _is_peephole || _is_ln ||
        (recurrent_activation == VSI_NN_ACT_HARD_SIGMOID && _is_ln) ||
        in_dtype == U8)
    {
        return NULL;
    }

    lstm_activation_param.is_ln = _is_ln;
    lstm_activation_param.is_cifg = _is_cifg;
    lstm_activation_param.is_proj = _is_proj;
    lstm_activation_param.is_hybrid = _is_hybrid;
    lstm_activation_param.is_peephole = _is_peephole;
    lstm_activation_param.recurrent_activation = recurrent_activation;
    lstm_activation_param.forget_bias = forget_bias;
    nodes[0] = vxLSTMActivationLayer(graph->g,
                                    inputs_tensor,
                                    input_real_num,
                                    &lstm_activation_param,
                                    outputs_tensor,
                                    2);
    CHECK_PTR_FAIL_GOTO( nodes[0], "Create vxLSTMActivationLayer node  fail.", final );

    if (outputs[2] && outputs[2]->t)
    {
        nodes[1] = vxTensorCopyNode(graph->g, outputs[0]->t, outputs[2]->t);
        CHECK_PTR_FAIL_GOTO( nodes[1], "Create vxTensorCopyNode node  fail.", final );

        node = nodes[1];
    }
    else
    {
        node = nodes[0];
    }

final:
    return node;
} /* lstmunit_activation() */

#undef REGISTER_LSTMUNIT_ACTIVATION_STREAM_PROCESSOR_KERNEL

#endif
