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
#include "kernel/vsi_nn_sp_unit_operation.h"
#include "kernel/vsi_nn_sp_lut.h"

#if (VX_STREAM_PROCESSOR_SUPPORT) && (VSI_NN_SUPPORT_LSTM_GRU_SP_IMPL) && (VX_GRU_CELL_VX_SUPPORT)

#define REGISTER_GRUCELL_ACTIVATION_STREAM_PROCESSOR_KERNEL( kernel_name )   \
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

REGISTER_GRUCELL_ACTIVATION_STREAM_PROCESSOR_KERNEL( grucell_activation_z_h )
{
    vsi_nn_kernel_node_t node[2] = {NULL};
    int32_t recurrent_activation = vsi_nn_kernel_param_get_int32( params, "recurrent_activation" );
    int32_t activation = vsi_nn_kernel_param_get_int32( params, "activation" );
    vx_tensor inputs_tensor[GRUCELL_ACT_Z_H_IN_CNT] = {NULL};
    vx_tensor output_tensor[1] = {NULL};
    vx_int32 i = 0;

    if ( recurrent_activation != VSI_NN_ACT_SIGMOID ||
         activation != VSI_NN_ACT_TANH )
    {
        return NULL;
    }
    for(i = 0; i < GRUCELL_ACT_Z_H_IN_CNT ;i++)
    {
        inputs_tensor[i] = inputs[i]->t;
    }

    output_tensor[0] = outputs[0]->t;

    VSI_UNREFERENCED(input_num);
    VSI_UNREFERENCED(output_num);
    VSI_UNREFERENCED(kernel);

    node[0] = vxGruCellActivationZHLayer(graph->g,
        inputs_tensor, (uint32_t)GRUCELL_ACT_Z_H_IN_CNT, recurrent_activation, activation, output_tensor, 1);
    CHECK_PTR_FAIL_GOTO( node[0], "Create vxGruCellActivationZHLayer node fail.", final );

    node[1] = vxTensorCopyNode( graph->g, outputs[0]->t, outputs[1]->t);
    CHECK_PTR_FAIL_GOTO( node[1], "Create vxTensorCopyNode node fail.", final );
final:
    return node[1];

} /* grucell_activation_z_h() */

#undef REGISTER_GRUCELL_ACTIVATION_STREAM_PROCESSOR_KERNEL

#endif
