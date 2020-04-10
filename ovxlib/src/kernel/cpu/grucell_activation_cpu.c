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
#include "vsi_nn_error.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "libnnext/vx_lib_nnext.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
#define _KERNEL_NAME        CVIVANTE_NAMESPACE("cpu.grucell_activation")

/*
 * Kernel params
 */
static vx_param_description_t _grucell_activation_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _GRUCELL_ACTIVATION_PARAM_NUM  _cnt_of_array( _grucell_activation_kernel_param_def )

#define _IO_COUNT               (5)
#define _GATE_ACTIVATION_INDEX  (5)
#define _CAND_ACTIVATION_IDNEX  (6)

/*
 * Kernel function
 */
DEF_KERNEL_EXECUTOR(_compute)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    int32_t i = 0;
    int32_t batch = 0;
    int32_t hidden_units = 0;
    float * buffer[_IO_COUNT] = { NULL };
    vsi_status status = VSI_FAILURE;
    vsi_nn_activation_e gate_activation;
    vsi_nn_activation_e candidate_activation;
    vsi_nn_kernel_tensor_t tensors[_IO_COUNT] = { NULL };
    vsi_nn_kernel_tensor_attr_t* attr[_IO_COUNT] = { NULL };

    tensors[0] = (vsi_nn_kernel_tensor_t)param[0];
    tensors[1] = (vsi_nn_kernel_tensor_t)param[1];
    tensors[2] = (vsi_nn_kernel_tensor_t)param[2];
    tensors[3] = (vsi_nn_kernel_tensor_t)param[3];
    tensors[4] = (vsi_nn_kernel_tensor_t)param[4];

    attr[0] = vsi_nn_kernel_tensor_attr_create( tensors[0] );
    attr[1] = vsi_nn_kernel_tensor_attr_create( tensors[1] );
    attr[2] = vsi_nn_kernel_tensor_attr_create( tensors[2] );
    attr[3] = vsi_nn_kernel_tensor_attr_create( tensors[3] );
    attr[4] = vsi_nn_kernel_tensor_attr_create( tensors[4] );

    /* z{t_} */
    buffer[0] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[0], attr[0], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[0], "Create input buffer fail.", final );
    buffer[1] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[1], attr[1], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[1], "Create input buffer fail.", final );
    buffer[2] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[2], attr[2], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[2], "Create input buffer fail.", final );
    buffer[3] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[3], attr[3], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[3], "Create input buffer fail.", final );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[4], &gate_activation);
    CHECK_STATUS_FAIL_GOTO(status, final);
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[5], &candidate_activation);
    CHECK_STATUS_FAIL_GOTO(status, final);

    batch = attr[0]->shape->data[1];
    hidden_units = attr[0]->shape->data[0];

    for( i = 0; i < batch * hidden_units; i++ )
    {
        float zt = vsi_nn_activation(buffer[0][i], gate_activation);
        float ht_ = vsi_nn_activation(buffer[1][i], candidate_activation);
        float ht_1 = buffer[2][i];
        float ht = zt * (ht_1 - ht_) + ht_;

        buffer[3][i] = ht;
    }

    status = vsi_nn_kernel_tensor_write_from_float( tensors[3], attr[3],
            buffer[3], batch * hidden_units );
    CHECK_STATUS_FAIL_GOTO( status, final );
    status = vsi_nn_kernel_tensor_write_from_float( tensors[4], attr[4],
            buffer[3], batch * hidden_units );
    CHECK_STATUS_FAIL_GOTO( status, final );

final:
    for( i = 0; i < 5; i ++ )
    {
        if( buffer[i] )
        {
            free( buffer[i] );
        }
        vsi_nn_kernel_tensor_attr_release( &attr[i] );
    }
    return status;
} /* _compute() */

/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs
    /* Add extra params */
    )
{
    vsi_status status = VSI_FAILURE;
    snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _KERNEL_NAME );
    kernel->info.function    = _compute;
    kernel->info.parameters  = _grucell_activation_kernel_param_def;
    kernel->info.numParams   = _cnt_of_array( _grucell_activation_kernel_param_def );
    status = VSI_SUCCESS;
    return status;
} /* _query_kernel() */

static vsi_nn_kernel_node_t _setup
    (
    vsi_nn_graph_t              * graph,
    vsi_nn_tensor_t            ** inputs,
    size_t                        input_num,
    vsi_nn_tensor_t            ** outputs,
    size_t                        output_num,
    const vsi_nn_kernel_param_t * params,
    vsi_nn_kernel_t             * kernel
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_node_param_t node_params[_GRUCELL_ACTIVATION_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;
    int32_t gate_activation = vsi_nn_kernel_param_get_int32( params, "gate_activation" );
    int32_t candidate_activation = vsi_nn_kernel_param_get_int32( params, "candidate_activation" );

    status = _query_kernel( kernel, inputs, outputs /* Add extra params */ );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _GRUCELL_ACTIVATION_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            node_params[5] = vsi_nn_kernel_scalar_create(graph, I32, &gate_activation );
            node_params[6] = vsi_nn_kernel_scalar_create(graph, I32, &candidate_activation );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _GRUCELL_ACTIVATION_PARAM_NUM );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( grucell_activation, _setup )
