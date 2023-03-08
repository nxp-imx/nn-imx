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
#define _KERNEL_NAME        CVIVANTE_NAMESPACE("cpu.l1norm")


/*
 * Kernel params
 */
static vx_param_description_t _l1norm_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _L1NORM_PARAM_NUM  _cnt_of_array( _l1norm_kernel_param_def )


/*
 * Kernel function
 */
DEF_KERNEL_EXECUTOR(_l1norm)
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
    vsi_size_t out_elements = 0;
    vsi_nn_kernel_tensor_attr_t * attr[_INPUT_NUM + _OUTPUT_NUM] = { NULL };
    int32_t axis = 0;
    int32_t i = 0;
    int32_t j = 0;
    int32_t outerSize = 1;
    int32_t innerSize = 1;
    int32_t axisSize  = 1;
    float epsilon = (float)1e-12;
    input[0] = (vsi_nn_kernel_tensor_t)param[0];
    output[0] = (vsi_nn_kernel_tensor_t)param[1];
    attr[0] = vsi_nn_kernel_tensor_attr_create( input[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( output[0] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );
    out_elements = vsi_nn_kernel_tensor_attr_get_size( attr[1] );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[2], &axis);
    CHECK_STATUS_FAIL_GOTO(status, final );

    buffer[0] = (float*)vsi_nn_kernel_tensor_create_buffer( input[0], attr[0], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[0], "Create input0 buffer fail.", final );
    buffer[1] = (float *)malloc( out_elements * sizeof(float) );
    CHECK_PTR_FAIL_GOTO( buffer[1], "Create output buffer fail.", final );
    memset( buffer[1], 0, out_elements * sizeof(float) );

    axisSize = (int32_t)attr[0]->shape->data[axis];

    for (i = 0; i < axis; i++)
    {
        innerSize *= (int32_t)attr[0]->shape->data[i];
    }

    for (i = axis + 1; i < (int32_t)(attr[0]->shape->size); i++)
    {
        outerSize *= (int32_t)attr[0]->shape->data[i];
    }

    {
        float sum = 0;
        float rcp_sum = 0;
        float data = 0;
        float out_data = 0;

        for (i = 0; i < outerSize; i++)
        {
            for (j = 0; j < innerSize; j++)
            {
                int32_t inner = i * axisSize * innerSize;
                int32_t begin = i * axisSize * innerSize + j;
                int32_t end   = begin +  axisSize * innerSize;

                sum = 0;
                for (inner = begin; inner < end; inner += innerSize)
                {
                    data = buffer[0][inner];
                    sum += (float)fabs(data);
                }
                rcp_sum = 1 / (sum + epsilon);
                for (inner = begin; inner < end; inner += innerSize)
                {
                    data = buffer[0][inner];
                    out_data = data * rcp_sum;
                    buffer[1][inner] = out_data;
                }
            }
        }
    }
    status = vsi_nn_kernel_tensor_write_from_float( output[0], attr[1],
        buffer[1], out_elements );
    CHECK_STATUS_FAIL_GOTO(status, final);
final:
    for ( i = 0; i < _INPUT_NUM + _OUTPUT_NUM; i ++ )
    {
        vsi_nn_safe_free( buffer[i] );
        buffer[i] = NULL;
        if (attr[i])
        {
            vsi_nn_kernel_tensor_attr_release( &attr[i] );
        }
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
    kernel->info.function    = _l1norm;
    kernel->info.parameters  = _l1norm_kernel_param_def;
    kernel->info.numParams   = _cnt_of_array( _l1norm_kernel_param_def );
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
    vsi_nn_kernel_node_param_t node_params[_L1NORM_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;

    int32_t axis       = vsi_nn_kernel_param_get_int32(params, "axis");

    status = _query_kernel( kernel, inputs, outputs );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            int32_t index = 2;
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _L1NORM_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &axis );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _L1NORM_PARAM_NUM );
            VSI_ASSERT( status == VSI_SUCCESS );
            vsi_nn_kernel_scalar_release( &node_params[2] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( l1norm, _setup )

