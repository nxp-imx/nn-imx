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

#if defined(VX_STREAM_PROCESSOR_SUPPORT) && defined(VX_LOGICAL_VX_SUPPORT)
__BEGIN_DECLS

#define REGISTER_LOGICAL_STREAM_PROCESSOR_KERNEL( kernel_name )   \
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

REGISTER_LOGICAL_STREAM_PROCESSOR_KERNEL( logical_ops )
{
    vsi_nn_kernel_node_t node = NULL;
    int32_t ops_type  = vsi_nn_kernel_param_get_int32( params, "ops_type" );
    vx_tensor inputs_tensor[2] = {NULL};
    vx_tensor output_tensor = NULL;

    inputs_tensor[0] = inputs[0]->t;
    inputs_tensor[1] = inputs[1]->t;
    output_tensor = outputs[0]->t;

    VSI_UNREFERENCED(kernel);
    VSI_UNREFERENCED(output_num);

    node = vxLogicalOpsLayer(
        graph->g,
        ops_type,
        inputs_tensor,
        (vx_uint32)input_num,
        output_tensor
        );
    CHECK_PTR_FAIL_GOTO( node, "Create logical ops node  fail.", final );
final:
    return (vsi_nn_kernel_node_t)node;
} /* logical_ops() */

REGISTER_LOGICAL_STREAM_PROCESSOR_KERNEL( logical_not )
{
    vsi_nn_kernel_node_t node = NULL;

    VSI_UNREFERENCED(kernel);
    VSI_UNREFERENCED(output_num);
    VSI_UNREFERENCED(input_num);
    VSI_UNREFERENCED(params);
    node = vxLogicalNotLayer(
        graph->g,
        inputs[0]->t,
        outputs[0]->t
        );
    CHECK_PTR_FAIL_GOTO( node, "Create logical not node  fail.", final );
final:
    return (vsi_nn_kernel_node_t)node;
} /* logical_not() */


__END_DECLS

#undef REGISTER_LOGICAL_STREAM_PROCESSOR_KERNEL

#endif
