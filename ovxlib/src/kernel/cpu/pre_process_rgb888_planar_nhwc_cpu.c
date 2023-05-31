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
#define _CPU_INPUT_NUM      (3)
#define _CPU_OUTPUT_NUM     (1)
#define _CPU_IO_NUM         (_CPU_INPUT_NUM + _CPU_OUTPUT_NUM)
#define _KERNEL_NAME        CVIVANTE_NAMESPACE("cpu.pre_process_rgb888_planar_nhwc")

/*
 * Kernel params
 */
static vx_param_description_t _pre_process_rgb888_planar_nhwc_kernel_param_def[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL},
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
};
#define _PRE_PROCESS_RGB888_PLANAR_NHWC_PARAM_NUM  _cnt_of_array( _pre_process_rgb888_planar_nhwc_kernel_param_def )


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
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_tensor_t tensors[_CPU_IO_NUM] = { NULL };
    float * buffer[_CPU_IO_NUM] = { NULL };
    size_t out_elements = 0;
    vsi_nn_kernel_tensor_attr_t * attr[_CPU_IO_NUM] = { NULL };
    uint32_t i = 0;
    int32_t xRatio = 0, yRatio = 0, xOffset = 0, yOffset = 0;
    float mean[3] = {0}, scale = 1;
    vsi_bool is_rgb888 = FALSE;
    int32_t reverse = 0;
    vsi_size_t dx = 0, dy = 0;
    vsi_size_t src_width = 0;
    vsi_size_t src_height= 0;
    vsi_size_t dst_width = 0;
    vsi_size_t dst_height= 0;

    VSI_UNREFERENCED(node);
    VSI_UNREFERENCED(param_size);

    for (i = 0; i < _CPU_IO_NUM; i++)
    {
        tensors[i] = (vsi_nn_kernel_tensor_t)param[i];
        if (tensors[i])
        {
            attr[i] = vsi_nn_kernel_tensor_attr_create( tensors[i] );
            CHECK_PTR_FAIL_GOTO( attr[i], "Create tensor attr buffer fail.", final );
        }
    }

    is_rgb888 = tensors[1] == NULL;

    out_elements = vsi_nn_kernel_tensor_attr_get_size( attr[3] );

    i = 4;
    status  = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[i++], &xRatio);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[i++], &yRatio);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[i++], &xOffset);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[i++], &yOffset);
    status |= vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[i++], &mean[0]);
    status |= vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[i++], &mean[1]);
    status |= vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[i++], &mean[2]);
    status |= vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[i++], &scale);
    status |= vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[i++], &reverse);
    CHECK_STATUS_FAIL_GOTO(status, final );

    for (i = 0; i < 3; i++)
    {
        if (tensors[i])
        {
            buffer[i] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[i], attr[i], TRUE );
            CHECK_PTR_FAIL_GOTO( buffer[i], "Create input0 buffer fail.", final );
        }
    }

    buffer[3] = (float *)malloc( out_elements * sizeof(float) );
    CHECK_PTR_FAIL_GOTO( buffer[3], "Create output buffer fail.", final );
    memset( buffer[3], 0, out_elements * sizeof(float) );

    src_width = attr[0]->shape->data[0];
    src_height = attr[0]->shape->data[1];
    dst_width = attr[3]->shape->data[1];
    dst_height = attr[3]->shape->data[2];

    for ( dy = 0; dy < dst_height; dy ++)
    {
        for ( dx = 0; dx < dst_width; dx ++)
        {
            int32_t line1[2] = {0}, line2[2] = {0};
            vsi_size_t src_index = 0;
            vsi_size_t dst_index = dx * 3 + dy * dst_width * 3;
            float rgb_value[3] = {0};

            if (xRatio != (1 << 15) && yRatio != (1 << 15))
            {
                int32_t fx = ((int32_t)dx * xRatio + (xRatio >> 1)) - (1 << 14);
                int32_t sx = fx & 0xffff8000; // Floor
                int32_t fy = 0, sy = 0;
                int32_t temp1 = 0;
                int32_t temp2 = 0;

                fx -= sx;
                sx = sx >> 15;

                sx = sx < 0 ? 0 : sx;
                sx = sx > (int32_t)src_width ? (int32_t)src_width - 1: sx;

                fx = (fx +(1 << 4)) >> 5;

                // for y
                fy = ((int32_t)dy * yRatio + (yRatio >> 1)) - (1<< 14);
                sy = fy & 0xffff8000; // Floor
                fy -= sy;
                sy = sy >> 15;

                sy = sy < 0 ? 0 : sy;
                fy = fy < 0 ? 0 : fy;

                fy = (fy + (1<< 4)) >> 5;

                sx += xOffset;
                sy += yOffset;
                src_index = (sx + sy * src_width);

#define DESCALE(x) (((x) + (1<<19)) >> 20)
                if (is_rgb888)
                {
                    int32_t c = 0;
                    for ( c = 0; c < 3; c++ )
                    {
                        vsi_size_t offset = src_width * src_height * c;
                        uint8_t result = 0;

                        line1[0] = (int32_t)buffer[0][src_index + offset];
                        line1[1] = (int32_t)buffer[0][src_index + 1 + offset];
                        line2[0] = (int32_t)buffer[0][src_index + src_width + offset];
                        line2[1] = (int32_t)buffer[0][src_index + src_width + 1 + offset];

                        temp1 = fx * (line1[1] - line1[0]) + (line1[0] << 10);
                        temp2 = fx * (line2[1] - line2[0]) + (line2[0] << 10);
                        temp1 = fy * (temp2 - temp1) + (temp1 << 10);
                        result = (uint8_t)(DESCALE(temp1));
                        rgb_value[c] = (result - mean[c]) * scale;
                    }
                }
                else
                {
                    int32_t c = 0;
                    for ( c = 0; c < 3; c++ )
                    {
                        uint8_t result = 0;
                        line1[0] = (int32_t)buffer[c][src_index];
                        line1[1] = (int32_t)buffer[c][src_index + 1];
                        line2[0] = (int32_t)buffer[c][src_index + src_width];
                        line2[1] = (int32_t)buffer[c][src_index + src_width + 1];

                        temp1 = fx * (line1[1] - line1[0]) + (line1[0] << 10);
                        temp2 = fx * (line2[1] - line2[0]) + (line2[0] << 10);
                        temp1 = fy * (temp2 - temp1) + (temp1 << 10);
                        result = (uint8_t)(DESCALE(temp1));
                        rgb_value[c] = (result - mean[c]) * scale;
                    }
                }
#undef DESCALE
            }
            else
            {
                int32_t ofset = xOffset + yOffset * (int32_t)src_width;
                src_index = dx + dy * src_width + ofset;

                if (is_rgb888)
                {
                    int32_t c = 0;
                    for ( c = 0; c < 3; c++ )
                    {
                        vsi_size_t offset = src_width * src_height * c;

                        rgb_value[c] = (buffer[0][src_index + offset] - mean[c]) * scale;
                    }
                }
                else
                {
                    int32_t c = 0;
                    for ( c = 0; c < 3; c++ )
                    {
                        rgb_value[c] = (buffer[c][src_index] - mean[c]) * scale;
                    }
                }
            }

            if (reverse)
            {
                buffer[3][dst_index + 2] = rgb_value[0];
                buffer[3][dst_index + 1] = rgb_value[1];
                buffer[3][dst_index + 0] = rgb_value[2];
            }
            else
            {
                buffer[3][dst_index + 0] = rgb_value[0];
                buffer[3][dst_index + 1] = rgb_value[1];
                buffer[3][dst_index + 2] = rgb_value[2];
            }
        }
    }

    for (i = 3; i < _CPU_IO_NUM; i++)
    {
        status = vsi_nn_kernel_tensor_write_from_float( tensors[i], attr[i],
                buffer[i], out_elements );
        CHECK_STATUS_FAIL_GOTO( status, final );
    }

final:
    for ( i = 0; i < _CPU_IO_NUM; i ++ )
    {
        if ( buffer[i] )
        {
            free( buffer[i] );
        }
        if (attr[i]) { vsi_nn_kernel_tensor_attr_release( &attr[i] ); }
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
    vsi_status status = VSI_SUCCESS;
    VSI_UNREFERENCED(inputs);
    VSI_UNREFERENCED(outputs);

    snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _KERNEL_NAME );
    kernel->info.function    = _compute;
    kernel->info.parameters  = _pre_process_rgb888_planar_nhwc_kernel_param_def;
    kernel->info.numParams   = _cnt_of_array( _pre_process_rgb888_planar_nhwc_kernel_param_def );

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
    vsi_nn_kernel_node_param_t node_params[_PRE_PROCESS_RGB888_PLANAR_NHWC_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;

    status = _query_kernel( kernel, inputs, outputs /* Add extra params */ );
    if ( VSI_SUCCESS == status)
    {
        uint32_t index = 4;
        int32_t scale_x  = vsi_nn_kernel_param_get_int32( params, "scale_x" );
        int32_t scale_y  = vsi_nn_kernel_param_get_int32( params, "scale_y" );
        int32_t left     = vsi_nn_kernel_param_get_int32( params, "left" );
        int32_t top      = vsi_nn_kernel_param_get_int32( params, "top" );
        int32_t reverse  = vsi_nn_kernel_param_get_int32( params, "reverse" );
        float r_mean = vsi_nn_kernel_param_get_float32( params, "r_mean" );
        float g_mean = vsi_nn_kernel_param_get_float32( params, "g_mean" );
        float b_mean = vsi_nn_kernel_param_get_float32( params, "b_mean" );
        float scale      = vsi_nn_kernel_param_get_float32( params, "scale" );

        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _PRE_PROCESS_RGB888_PLANAR_NHWC_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &scale_x );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &scale_y );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &left );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &top );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &r_mean );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &g_mean );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &b_mean );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &scale );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &reverse );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _PRE_PROCESS_RGB888_PLANAR_NHWC_PARAM_NUM );
            vsi_nn_kernel_scalar_release( &node_params[4] );
            vsi_nn_kernel_scalar_release( &node_params[5] );
            vsi_nn_kernel_scalar_release( &node_params[6] );
            vsi_nn_kernel_scalar_release( &node_params[7] );
            vsi_nn_kernel_scalar_release( &node_params[8] );
            vsi_nn_kernel_scalar_release( &node_params[9] );
            vsi_nn_kernel_scalar_release( &node_params[10] );
            vsi_nn_kernel_scalar_release( &node_params[11] );
            vsi_nn_kernel_scalar_release( &node_params[12] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( pre_process_rgb888_planar_nhwc, _setup )

