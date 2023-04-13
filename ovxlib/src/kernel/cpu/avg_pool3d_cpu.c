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
#define _KERNEL_NAME        CVIVANTE_NAMESPACE("cpu.avg_pool3d")


/*
 * Kernel params
 */
static vx_param_description_t _avg_pool3d_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _AVG_POOL3D_PARAM_NUM  _cnt_of_array( _avg_pool3d_kernel_param_def )


/*
 * Kernel function
 */
DEF_KERNEL_EXECUTOR(_avg_pool3d_exec)
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
    int32_t ksize_x = 0, ksize_y = 0, ksize_z = 0, stride_x = 0, stride_y = 0, stride_z = 0;
    int32_t pad_left = 0, pad_right = 0, pad_top = 0, pad_bottom = 0, pad_front = 0, pad_end = 0;
    int32_t depth_in = 0, depth_out = 0, count_include_pad = 0;
    int32_t i;

    VSI_UNREFERENCED(node);
    VSI_UNREFERENCED(param_size);

    input[0] = (vsi_nn_kernel_tensor_t)param[0];
    output[0] = (vsi_nn_kernel_tensor_t)param[1];
    attr[0] = vsi_nn_kernel_tensor_attr_create( input[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( output[0] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );
    out_elements = vsi_nn_kernel_tensor_attr_get_size( attr[1] );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[2], &ksize_x);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[3], &ksize_y);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[4], &ksize_z);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[5], &pad_left);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[6], &pad_right);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[7], &pad_top);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[8], &pad_bottom);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[9], &pad_front);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[10], &pad_end);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[11], &stride_x);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[12], &stride_y);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[13], &stride_z);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[14], &depth_in);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[15], &depth_out);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[16], &count_include_pad);

    CHECK_STATUS_FAIL_GOTO(status, final );

    buffer[0] = (float*)vsi_nn_kernel_tensor_create_buffer( input[0], attr[0], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[0], "Create input0 buffer fail.", final );

    buffer[1] = (float *)malloc( out_elements * sizeof(float) );
    CHECK_PTR_FAIL_GOTO( buffer[1], "Create output buffer fail.", final );
    memset( buffer[1], 0, out_elements * sizeof(float) );

    {
        int32_t depth_o  = (int32_t)attr[1]->shape->data[2];
        int32_t height_o = (int32_t)attr[1]->shape->data[1];
        int32_t width_o  = (int32_t)attr[1]->shape->data[0];
        int32_t height   = (int32_t)attr[0]->shape->data[1];
        int32_t width    = (int32_t)attr[0]->shape->data[0];
        int32_t batch    = depth_o / depth_out;
        int32_t b = 0, k = 0, j = 0;
        int32_t output_base = 0;
        int32_t input_base  = 0;

        for (b = 0; b < batch; b++)
        {
            output_base = b * depth_out * height_o * width_o;
            input_base = b * depth_in * height * width;
            for (k = 0; k < depth_o; k++)
            {
                for (j = 0; j < height_o; j++)
                {
                    for (i = 0; i < width_o; i++)
                    {
                        int32_t dstart = k * stride_z - pad_front;
                        int32_t hstart = j * stride_y - pad_top;
                        int32_t wstart = i * stride_x - pad_left;
                        int32_t dend = vsi_nn_min(dstart + ksize_z, depth_in);
                        int32_t hend = vsi_nn_min(hstart + ksize_y, height);
                        int32_t wend = vsi_nn_min(wstart + ksize_x, width);
                        int32_t pool_index = output_base + k * height_o * width_o + j * width_o + i;
                        int32_t d = 0, h = 0, w = 0, count = 0;
                        float sum = 0;

                        dstart = vsi_nn_max(dstart, 0);
                        hstart = vsi_nn_max(hstart, 0);
                        wstart = vsi_nn_max(wstart, 0);

                        for (d = dstart; d < dend; ++ d)
                        {
                            for (h = hstart; h < hend; ++ h)
                            {
                                for (w = wstart; w < wend; ++ w)
                                {
                                    int32_t index = input_base + d * height * width + h * width + w;
                                    float data = buffer[0][index];
                                    sum += data;
                                    count++;
                                }
                            }
                        }
                        if (count_include_pad)
                        {
                            count = ksize_x * ksize_y * ksize_z;
                        }
                        buffer[1][pool_index] = sum / count;
                    }
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

    return status;

} /* _avg_pool3d_exec */


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
    kernel->info.function    = _avg_pool3d_exec;
    kernel->info.parameters  = _avg_pool3d_kernel_param_def;
    kernel->info.numParams   = _cnt_of_array( _avg_pool3d_kernel_param_def );
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
    vsi_nn_kernel_node_param_t node_params[_AVG_POOL3D_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;

    int32_t ksize_x    = vsi_nn_kernel_param_get_int32(params, "ksize_x");
    int32_t ksize_y    = vsi_nn_kernel_param_get_int32(params, "ksize_y");
    int32_t ksize_z    = vsi_nn_kernel_param_get_int32(params, "ksize_z");
    int32_t pad_left   = vsi_nn_kernel_param_get_int32(params, "pad_left");
    int32_t pad_right  = vsi_nn_kernel_param_get_int32(params, "pad_right");
    int32_t pad_top    = vsi_nn_kernel_param_get_int32(params, "pad_top");
    int32_t pad_bottom = vsi_nn_kernel_param_get_int32(params, "pad_bottom");
    int32_t pad_front  = vsi_nn_kernel_param_get_int32(params, "pad_front");
    int32_t pad_end    = vsi_nn_kernel_param_get_int32(params, "pad_end");
    int32_t stride_x   = vsi_nn_kernel_param_get_int32(params, "stride_x");
    int32_t stride_y   = vsi_nn_kernel_param_get_int32(params, "stride_y");
    int32_t stride_z   = vsi_nn_kernel_param_get_int32(params, "stride_z");
    int32_t depth_in   = vsi_nn_kernel_param_get_int32(params, "depth_in");
    int32_t depth_out  = vsi_nn_kernel_param_get_int32(params, "depth_out");
    int32_t count_include_pad   = vsi_nn_kernel_param_get_int32(params, "count_include_pad");

    status = _query_kernel( kernel, inputs, outputs );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            int32_t index = 2;
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _AVG_POOL3D_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &ksize_x );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &ksize_y );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &ksize_z );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &pad_left );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &pad_right );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &pad_top );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &pad_bottom );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &pad_front );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &pad_end );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &stride_x );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &stride_y );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &stride_z );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &depth_in );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &depth_out );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &count_include_pad );

            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _AVG_POOL3D_PARAM_NUM );
            VSI_ASSERT( status == VSI_SUCCESS );
            vsi_nn_kernel_scalar_release( &node_params[2] );
            vsi_nn_kernel_scalar_release( &node_params[3] );
            vsi_nn_kernel_scalar_release( &node_params[4] );
            vsi_nn_kernel_scalar_release( &node_params[5] );
            vsi_nn_kernel_scalar_release( &node_params[6] );
            vsi_nn_kernel_scalar_release( &node_params[7] );
            vsi_nn_kernel_scalar_release( &node_params[8] );
            vsi_nn_kernel_scalar_release( &node_params[9] );
            vsi_nn_kernel_scalar_release( &node_params[10] );
            vsi_nn_kernel_scalar_release( &node_params[11] );
            vsi_nn_kernel_scalar_release( &node_params[12] );
            vsi_nn_kernel_scalar_release( &node_params[13] );
            vsi_nn_kernel_scalar_release( &node_params[14] );
            vsi_nn_kernel_scalar_release( &node_params[15] );
            vsi_nn_kernel_scalar_release( &node_params[16] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( avg_pool3d, _setup )

