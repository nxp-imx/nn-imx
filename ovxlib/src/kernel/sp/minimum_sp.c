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

#if defined(VX_STREAM_PROCESSOR_SUPPORT) && defined(VX_MAX_MIN_IMUM_VX_SUPPORT)

#define REGISTER_MINIMUM_STREAM_PROCESSOR_KERNEL( kernel_name )   \
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

REGISTER_MINIMUM_STREAM_PROCESSOR_KERNEL( minimum )
{
    vsi_nn_kernel_node_t node = NULL;
    vx_tensor inputs_tensor[2] = {NULL};

    if (vsi_nn_is_broadcast_operaton(inputs, input_num, outputs[0]))
    {
        return NULL;
    }

    inputs_tensor[0] = inputs[0]->t;
    inputs_tensor[1] = inputs[1]->t;

    VSI_UNREFERENCED(kernel);
    VSI_UNREFERENCED(params);
    VSI_UNREFERENCED(output_num);

    node = vxMinimumLayer(
        graph->g,
        inputs_tensor,
        (uint32_t)input_num,
        outputs[0]->t
        );
    CHECK_PTR_FAIL_GOTO( node, "Create minimum ops node  fail.", final );
final:
    return (vsi_nn_kernel_node_t)node;
} /* minimum() */

#undef REGISTER_MINIMUM_STREAM_PROCESSOR_KERNEL

#endif
