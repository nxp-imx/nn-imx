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

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)
#define _KERNEL_NAME        CVIVANTE_NAMESPACE("cpu.pool")

#define VSI_FLOAT32_MIN     (1.175494351e-38F)

 typedef enum
{
    _error = -1,
    _MAX = 0,
    _AVG
} vsi_nn_pool_type_e;

/*
 * Kernel params
 */
static vx_param_description_t _maxpool_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _MAXPOOL_PARAM_NUM  _cnt_of_array( _maxpool_kernel_param_def )


/*
 * Kernel function
 */
DEF_KERNEL_EXECUTOR(_maxpool_exec)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_tensor_t input[_INPUT_NUM] = {NULL};
    vsi_nn_kernel_tensor_t output[_OUTPUT_NUM] = {NULL};
    float * buffer[_INPUT_NUM + _OUTPUT_NUM] = { NULL };
    size_t out_elements = 0;
    vsi_nn_kernel_tensor_attr_t * attr[_INPUT_NUM + _OUTPUT_NUM] = { NULL };
    int32_t pool_pad_x_left = 0, pool_pad_y_top = 0, kernel_dia_x = 0, kernel_dia_y = 0;
    int32_t stride_x = 0, stride_y = 0, dilation_x = 0, dilation_y = 0;
    int32_t i = 0;
    input[0] = (vsi_nn_kernel_tensor_t)param[0];
    output[0] = (vsi_nn_kernel_tensor_t)param[1];

    VSI_UNREFERENCED(node);
    VSI_UNREFERENCED(param_size);
    attr[0] = vsi_nn_kernel_tensor_attr_create( input[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( output[0] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );
    out_elements = vsi_nn_kernel_tensor_attr_get_size( attr[1] );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[2], &stride_x);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[3], &stride_y);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[4], &pool_pad_x_left);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[5], &pool_pad_y_top);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[6], &kernel_dia_x);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[7], &kernel_dia_y);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[8], &dilation_x);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[9], &dilation_y);
    CHECK_STATUS_FAIL_GOTO(status, final );

    buffer[0] = (float*)vsi_nn_kernel_tensor_create_buffer( input[0], attr[0], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[0], "Create input0 buffer fail.", final );

    buffer[1] = (float *)malloc( out_elements * sizeof(float) );
    CHECK_PTR_FAIL_GOTO( buffer[1], "Create output buffer fail.", final );
    memset( buffer[1], 0, out_elements * sizeof(float) );

    {
        int32_t height_o = (int32_t)attr[1]->shape->data[1];
        int32_t width_o  = (int32_t)attr[1]->shape->data[0];
        int32_t height   = (int32_t)attr[0]->shape->data[1];
        int32_t width    = (int32_t)attr[0]->shape->data[0];
        int32_t b = 0, j = 0, ii = 0, jj = 0, batch = 1;
        int32_t output_base = 0;
        int32_t input_base  = 0;

        for (b = 2; b < (int32_t)attr[1]->shape->size; b++)
        {
            batch *= (int32_t)attr[1]->shape->data[b];
        }

        for (b = 0; b < batch; b++)
        {
            output_base = b * height_o * width_o;
            input_base = b * height * width;
            for (j = 0; j < height_o; j++)
            {
                int32_t y = j * stride_y - pool_pad_y_top;
                int32_t start_y = vsi_nn_min(y, height - 1);
                int32_t end_y   = vsi_nn_min(y + kernel_dia_y, height);
                for (i = 0; i < width_o; i++)
                {
                    float maxVal = VSI_FLOAT32_MIN, data = 0;
                    int32_t x = i * stride_x - pool_pad_x_left;
                    int32_t start_x = vsi_nn_min(x, width - 1);
                    int32_t end_x   = vsi_nn_min(x + kernel_dia_x, width);

                    for (jj = start_y; jj < end_y; jj += dilation_y)
                    {
                        for (ii = start_x; ii < end_x && jj >= 0; ii += dilation_x)
                        {
                            int32_t index = ii + jj * width;
                            if (ii < 0)
                                continue;

                            data = buffer[0][index + input_base];
                            maxVal = vsi_nn_max(data, maxVal);
                        }
                    }

                    buffer[1][i + j * width_o + output_base] = maxVal;
                }
            }
        }
    }
    status = vsi_nn_kernel_tensor_write_from_float( output[0], attr[1],
            buffer[1], out_elements );
final:
    for ( i = 0; i < _INPUT_NUM + _OUTPUT_NUM; i ++ )
    {
        vsi_nn_safe_free( buffer[i] );
        if (attr[i])
        {
            vsi_nn_kernel_tensor_attr_release( &attr[i] );
        }
    }
#undef VSI_FLOAT32_MIN

    return status;
} /* _maxpool_exec() */


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
    VSI_UNREFERENCED(inputs);
    VSI_UNREFERENCED(outputs);
    snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _KERNEL_NAME );
    kernel->info.function    = _maxpool_exec;
    kernel->info.parameters  = _maxpool_kernel_param_def;
    kernel->info.numParams   = _cnt_of_array( _maxpool_kernel_param_def );
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
    vsi_nn_kernel_node_param_t node_params[_MAXPOOL_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;
    int32_t pool_type         = vsi_nn_kernel_param_get_int32( params, "pool_type" );
    int32_t pool_size_x       = vsi_nn_kernel_param_get_int32( params, "pool_size_x" );
    int32_t pool_size_y       = vsi_nn_kernel_param_get_int32( params, "pool_size_y" );
    int32_t pool_pad_x_left   = vsi_nn_kernel_param_get_int32( params, "pool_pad_x_left" );
    int32_t pool_pad_y_top    = vsi_nn_kernel_param_get_int32( params, "pool_pad_y_top" );
    int32_t stride_x          = vsi_nn_kernel_param_get_int32( params, "stride_x" );
    int32_t stride_y          = vsi_nn_kernel_param_get_int32( params, "stride_y" );
    int32_t dilation_x        = vsi_nn_kernel_param_get_int32( params, "dilation_x" );
    int32_t dilation_y        = vsi_nn_kernel_param_get_int32( params, "dilation_y" );
    int32_t kernel_dia_x      = pool_size_x * dilation_x;
    int32_t kernel_dia_y      = pool_size_y * dilation_y;

    if (pool_type != _MAX)
    {
        return NULL;
    }

    status = _query_kernel( kernel, inputs, outputs );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            int32_t index = 2;
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _MAXPOOL_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &stride_x );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &stride_y );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &pool_pad_x_left );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &pool_pad_y_top );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &kernel_dia_x );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &kernel_dia_y );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &dilation_x );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &dilation_y );

            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _MAXPOOL_PARAM_NUM );
            VSI_ASSERT( status == VSI_SUCCESS );
            vsi_nn_kernel_scalar_release( &node_params[2] );
            vsi_nn_kernel_scalar_release( &node_params[3] );
            vsi_nn_kernel_scalar_release( &node_params[4] );
            vsi_nn_kernel_scalar_release( &node_params[5] );
            vsi_nn_kernel_scalar_release( &node_params[6] );
            vsi_nn_kernel_scalar_release( &node_params[7] );
            vsi_nn_kernel_scalar_release( &node_params[8] );
            vsi_nn_kernel_scalar_release( &node_params[9] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( pool, _setup )

