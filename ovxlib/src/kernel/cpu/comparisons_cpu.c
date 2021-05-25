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
#include "libnnext/vsi_nn_vxkernel.h"

__BEGIN_DECLS


#define _CPU_ARG_NUM            (1)
#define _CPU_INPUT_NUM          (2)
#define _CPU_OUTPUT_NUM         (1)
#define _CPU_IO_NUM             (_CPU_INPUT_NUM + _CPU_OUTPUT_NUM)
#define _CPU_PARAM_NUM          (_CPU_ARG_NUM + _CPU_IO_NUM)
#define _KERNEL_NAME            CVIVANTE_NAMESPACE("comparisons_sw")

typedef enum
{
    COMP_GREAT = VSI_NN_RELATIONAL_OPS_GREAT,
    COMP_GREAT_EQUAL = VSI_NN_RELATIONAL_OPS_GREAT_EQUAL,
    COMP_LESS = VSI_NN_RELATIONAL_OPS_LESS,
    COMP_LESS_EQUAL = VSI_NN_RELATIONAL_OPS_LESS_EQUAL,
    COMP_NOT_EQUAL = VSI_NN_RELATIONAL_OPS_NOT_EQUAL,
    COMP_EQUAL = VSI_NN_RELATIONAL_OPS_EQUAL,
} relational_type_e;


static int32_t _expand_offset
    (
    int32_t index,
    int32_t * shape, size_t rank,
    size_t * strides, int32_t * out_shape
    )
{
    uint32_t i;
    int32_t offset = 0;

    for( i = 0; i < rank && index; i ++ )
    {
        if( shape[i] == out_shape[i] )
        {
            offset += (int32_t)strides[i] * ( index % out_shape[i] );
        }
        index /= out_shape[i];
    }
    return offset;
}

DEF_KERNEL_EXECUTOR(_comparisons_exec)
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
    size_t stride_size[_CPU_INPUT_NUM][VSI_NN_MAX_DIM_NUM] = {{0}};
    int32_t i = 0;
    int32_t operation = 0;

    tensors[0]  = (vsi_nn_kernel_tensor_t)param[0];
    tensors[1]  = (vsi_nn_kernel_tensor_t)param[1];
    tensors[2]  = (vsi_nn_kernel_tensor_t)param[2];

    attr[0] = vsi_nn_kernel_tensor_attr_create( tensors[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( tensors[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );
    attr[2] = vsi_nn_kernel_tensor_attr_create( tensors[2] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", final );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[3], &operation);
    CHECK_STATUS_FAIL_GOTO(status, final );


    vsi_nn_kernel_tensor_attr_get_stride( attr[0], stride_size[0] );
    vsi_nn_kernel_tensor_attr_get_stride( attr[1], stride_size[1] );

    out_elements = vsi_nn_kernel_tensor_attr_get_size( attr[2] );

    buffer[0] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[0], attr[0], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[0], "Create input0 buffer fail.", final );

    buffer[1] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[1], attr[1], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[1], "Create input1 buffer fail.", final );

    buffer[2] = (float *)malloc( out_elements * sizeof(float) );
    CHECK_PTR_FAIL_GOTO( buffer[2], "Create output buffer fail.", final );
    memset( buffer[2], 0, out_elements * sizeof(float) );

    for (i = 0; i < (int32_t)out_elements; i++)
    {
        int32_t in0_offset = 0;
        int32_t in1_offset = 0;
        float val1 = 0.f;
        float val2 = 0.f;
        vsi_bool data = 0;

        in0_offset = _expand_offset( i, attr[0]->shape->data, attr[0]->shape->size,
                stride_size[0], attr[2]->shape->data );
        in1_offset = _expand_offset( i, attr[1]->shape->data, attr[1]->shape->size,
                stride_size[1], attr[2]->shape->data );

        val1 = buffer[0][in0_offset];
        val2 = buffer[1][in1_offset];

        switch (operation)
        {
        case COMP_GREAT:
            data = val1 > val2;
            break;
        case COMP_GREAT_EQUAL:
            data = val1 >= val2;
            break;
        case COMP_LESS:
            data = val1 < val2;
            break;
        case COMP_LESS_EQUAL:
            data = val1 <= val2;
            break;
        case COMP_EQUAL:
            data = val1 == val2;
            break;
        case COMP_NOT_EQUAL:
            data = val1 != val2;
            break;
        default:
            break;
        }
        buffer[2][i] = (float)data;
    }

    status = vsi_nn_kernel_tensor_write_from_float( tensors[1], attr[1],
            buffer[1], out_elements );
    CHECK_STATUS_FAIL_GOTO( status, final );

final:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }
    if (attr[1])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[1] );
        attr[1] = NULL;
    }
    if (attr[2])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[2] );
        attr[2] = NULL;
    }

    for( i = 0; i < _CPU_IO_NUM; i ++ )
    {
        if( buffer[i] )
        {
            free( buffer[i] );
            buffer[i] = NULL;
        }
    }
    return status;
} /* _comparisons_exec() */

static vx_param_description_t kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

#define INPUT_FUNC_OP           (3)

static const vx_kernel_description_t _kernel_info =
{
    KERNEL_ID_PLACEHOLDER,
    _KERNEL_NAME,
    _comparisons_exec,
    kernel_param_def,
    _cnt_of_array( kernel_param_def ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_nn_kernel_t* kernel
    )
{
    memmove( &kernel->info, &_kernel_info, sizeof(vx_kernel_description_t) );
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
    vsi_nn_kernel_t             * kernel
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_kernel_node_param_t backend_params[_CPU_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    int32_t operation = 0;

    operation = vsi_nn_kernel_param_get_int32( params, "operation" );

    status = _query_kernel( inputs, outputs, kernel );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( backend_params, _CPU_PARAM_NUM,
                    inputs, _CPU_INPUT_NUM, outputs, _CPU_OUTPUT_NUM );
            backend_params[INPUT_FUNC_OP] = vsi_nn_kernel_scalar_create(
                    graph, I32, &operation );
            /* Pass parameters to node. */
            status = vsi_nn_kernel_node_pass_param( node, backend_params, _CPU_PARAM_NUM );

            vsi_nn_kernel_scalar_release( &backend_params[INPUT_FUNC_OP] );
        }
        else
        {
            status = VSI_FAILURE;
        }
    }

    return node;
} /* _setup() */


__END_DECLS

REGISTER_BACKEND_CPU( relational_ops, _setup )
