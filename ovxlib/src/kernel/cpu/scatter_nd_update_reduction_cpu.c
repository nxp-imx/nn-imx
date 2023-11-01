/****************************************************************************
*
*    Copyright (c) 2019 Vivante Corporation
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
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_error.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
#define _CPU_ARG_NUM            (4)
#define _CPU_INPUT_NUM          (3)
#define _CPU_OUTPUT_NUM         (1)
#define _CPU_IO_NUM             (_CPU_INPUT_NUM + _CPU_OUTPUT_NUM)
#define _CPU_PARAM_NUM          (_CPU_ARG_NUM + _CPU_IO_NUM)
#define _KERNEL_NAME            CVIVANTE_NAMESPACE("cpu.scatter_nd_update_reduction")

DEF_KERNEL_EXECUTOR(_scatter_nd_update_reduction_exec)
    (
    vsi_nn_kernel_node_t node,
    const vsi_nn_kernel_node_param_t * param,
    size_t param_size
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_tensor_t tensors[_CPU_IO_NUM] = { NULL };
    uint32_t *   para_buffer[1] = { NULL };
    float * buffer[3] = { NULL };
    size_t out_elements = 0;
    vsi_nn_kernel_tensor_attr_t * attr[4] = { NULL };
    int32_t i = 0, j = 0;
    int32_t block_size = 1, indices_num = 1;
    int32_t coord_dim = 1, reduction = 0;

    VSI_UNREFERENCED(node);
    VSI_UNREFERENCED(param_size);

    tensors[0]  = (vsi_nn_kernel_tensor_t)param[0]; // ref
    tensors[1]  = (vsi_nn_kernel_tensor_t)param[1]; // idx    int
    tensors[2]  = (vsi_nn_kernel_tensor_t)param[2]; // update
    tensors[3]  = (vsi_nn_kernel_tensor_t)param[3]; // output

    attr[0] = vsi_nn_kernel_tensor_attr_create( tensors[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( tensors[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );
    attr[2] = vsi_nn_kernel_tensor_attr_create( tensors[2] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", final );
    attr[3] = vsi_nn_kernel_tensor_attr_create( tensors[3] );
    CHECK_PTR_FAIL_GOTO( attr[3], "Create tensor attr buffer fail.", final );

    out_elements = vsi_nn_kernel_tensor_attr_get_size( attr[3] );

    buffer[0] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[0], attr[0], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[0], "Create input0 buffer fail.", final );

    para_buffer[0] = (uint32_t*)vsi_nn_kernel_tensor_create_buffer( tensors[1], attr[1], FALSE );
    CHECK_PTR_FAIL_GOTO( para_buffer[0], "Create input1 buffer fail.", final );

    buffer[1] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[2], attr[2], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[1], "Create input2 buffer fail.", final );

    vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[4], &(block_size));
    vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[5], &(coord_dim));
    vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[6], &(indices_num));
    vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[7], &(reduction));

    buffer[2] = (float *)malloc( out_elements * sizeof(float) );
    CHECK_PTR_FAIL_GOTO( buffer[2], "Create output buffer fail.", final );
    memcpy( buffer[2], buffer[0], out_elements * sizeof(float) );

    if (coord_dim <= VSI_NN_MAX_DIM_NUM)
    {
        vsi_ssize_t stride[VSI_NN_MAX_DIM_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
        vsi_ssize_t new_shape[VSI_NN_MAX_DIM_NUM] = {1, 1, 1, 1, 1, 1, 1, 1};
        vsi_ssize_t merge_dim = (vsi_ssize_t)attr[3]->shape->size - coord_dim + 1;

        for (i = 0; i < merge_dim; ++i)
        {
            new_shape[0] *= attr[3]->shape->data[i];
        }
        stride[0] = new_shape[0] / block_size;

        for (i = 1; i < coord_dim; ++i)
        {
            vsi_ssize_t idx = merge_dim + (vsi_ssize_t)i - 1;

            if (idx >= 0)
            {
                new_shape[i] = attr[3]->shape->data[idx];
                stride[i] = stride[i - 1] * new_shape[i];
            }
        }

        for (i = 0; i < indices_num; i++)
        {
            uint32_t in_index = i * block_size;
            vsi_size_t out_index = 0;
            uint32_t coord[VSI_NN_MAX_DIM_NUM] = {0};
            int32_t byd_flg = 0;
            vsi_ssize_t  mask_idx = 0;

            for (j = 0; j < coord_dim; j++)
            {
                coord[j] = para_buffer[0][i * coord_dim + coord_dim - j - 1];
                if (coord[j] >= (uint32_t)new_shape[j])
                {
                    byd_flg = 1;
                    break;
                }
            }
            if (byd_flg)
            {
                continue;
            }
            mask_idx = coord[0];
            for (j = 0; j < coord_dim - 1; j++)
            {
                mask_idx += coord[j + 1] * stride[j];
            }
            out_index = mask_idx * block_size;

            for (j = 0; j < block_size; j++)
            {
                switch (reduction)
                {
                case VSI_NN_REDUCTION_TYPE_ADD:
                    buffer[2][out_index + j] += buffer[1][in_index + j];
                    break;
                case VSI_NN_REDUCTION_TYPE_MUL:
                    buffer[2][out_index + j] *= buffer[1][in_index + j];
                    break;
                case VSI_NN_REDUCTION_TYPE_MAX:
                    buffer[2][out_index + j] = vsi_nn_max(buffer[1][in_index + j], buffer[2][out_index + j]);
                    break;
                case VSI_NN_REDUCTION_TYPE_MIN:
                    buffer[2][out_index + j] = vsi_nn_min(buffer[1][in_index + j], buffer[2][out_index + j]);
                    break;
                default:
                    break;
                }
            }
        }
    }
    else
    {
        status = VSI_FAILURE;
        CHECK_STATUS_FAIL_GOTO( status, final );
    }

    status = vsi_nn_kernel_tensor_write_from_float( tensors[3], attr[3],
            buffer[2], out_elements );
    CHECK_STATUS_FAIL_GOTO( status, final );

final:
    vsi_nn_safe_free( para_buffer[0] );

    for ( i = 0; i < 3; i ++ )
    {
        vsi_nn_safe_free( buffer[i] );
    }
    for ( i = 0; i < 4; i ++ )
    {
        if (attr[i]) { vsi_nn_kernel_tensor_attr_release( &attr[i] ); }
    }
    return status;
} /* _scatter_nd_update_reduction_exec() */
/*
 * Kernel params
 */
static vx_param_description_t _scatter_nd_update_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};

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
    kernel->info.function    = _scatter_nd_update_reduction_exec;
    kernel->info.parameters  = _scatter_nd_update_kernel_param_def;
    kernel->info.numParams   = _cnt_of_array( _scatter_nd_update_kernel_param_def );

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
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_node_param_t backend_params[_CPU_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;
    int32_t block_size  = vsi_nn_kernel_param_get_int32( params, "block_size" );
    int32_t coord_dim  = vsi_nn_kernel_param_get_int32( params, "coord_dim" );
    int32_t idx_num  = vsi_nn_kernel_param_get_int32( params, "idx_num" );
    int32_t reduction  = vsi_nn_kernel_param_get_int32( params, "reduction" );

    VSI_UNREFERENCED(input_num);
    VSI_UNREFERENCED(output_num);

    status = _query_kernel( inputs, outputs, kernel );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            uint32_t index = 4;
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( backend_params, _CPU_PARAM_NUM,
                    inputs, _CPU_INPUT_NUM, outputs, _CPU_OUTPUT_NUM );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &block_size );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &coord_dim );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &idx_num );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &reduction );
            /* Pass parameters to node. */
            status = vsi_nn_kernel_node_pass_param( node, backend_params, _CPU_PARAM_NUM );
            CHECK_STATUS( status );
            vsi_nn_kernel_scalar_release( &backend_params[4] );
            vsi_nn_kernel_scalar_release( &backend_params[5] );
            vsi_nn_kernel_scalar_release( &backend_params[6] );
            vsi_nn_kernel_scalar_release( &backend_params[7] );
        }
        else
        {
            status = VSI_FAILURE;
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( scatter_nd_update_reduction, _setup )
