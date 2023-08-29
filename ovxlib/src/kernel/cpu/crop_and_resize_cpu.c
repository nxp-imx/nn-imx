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
#define _INPUT_NUM          (3)
#define _OUTPUT_NUM         (1)
#define _KERNEL_NAME        CVIVANTE_NAMESPACE("cpu.crop_and_resize")

#define ROUNDF(x) floorf(x + 0.5f)

/*
 * Kernel params
 */
static vx_param_description_t _crop_and_resize_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _CROP_AND_RESIZE_PARAM_NUM  _cnt_of_array( _crop_and_resize_kernel_param_def )


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
    vsi_nn_kernel_tensor_t tensors[_INPUT_NUM + _OUTPUT_NUM] = {NULL};
    float *buffer[_INPUT_NUM + _OUTPUT_NUM] = {NULL};
    vsi_nn_kernel_tensor_attr_t *attr[_INPUT_NUM + _OUTPUT_NUM] = {NULL};
    size_t out_elements = 0;
    int32_t boxes_num = 0;
    int32_t i  = 0;
    int32_t b  = 0;
    int32_t bb = 0;
    int32_t x  = 0;
    int32_t y  = 0;
    int32_t d  = 0;
    float   extrapolation_value = 0;
    int32_t resize_method = 0;
    int32_t crop_height = 0;
    int32_t crop_width = 0;
    int32_t image_height = 0;
    int32_t image_width = 0;
    int32_t depth = 0;

    VSI_UNREFERENCED(node);
    VSI_UNREFERENCED(param_size);

    tensors[0] = (vsi_nn_kernel_tensor_t)param[0];
    tensors[1] = (vsi_nn_kernel_tensor_t)param[1];
    tensors[2] = (vsi_nn_kernel_tensor_t)param[2];
    tensors[3] = (vsi_nn_kernel_tensor_t)param[3];

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
    buffer[1] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[1], attr[1], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[0], "Create input0 buffer fail.", final );
    buffer[2] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[2], attr[2], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[2], "Create input0 buffer fail.", final );
    buffer[3] = (float *)malloc( out_elements * sizeof(float) );
    CHECK_PTR_FAIL_GOTO( buffer[3], "Create input0 buffer fail.", final );
    memset( buffer[3], 0, out_elements * sizeof(float) );

    status  = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[4],&resize_method);
    status |= vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[5],&extrapolation_value);

    crop_width = (int32_t)attr[3]->shape->data[0];
    crop_height = (int32_t)attr[3]->shape->data[1];
    depth = attr[3]->shape->size > 2 ? (int32_t)attr[3]->shape->data[2] : 1;

    image_height = (int32_t)attr[0]->shape->data[1];
    image_width = (int32_t)attr[0]->shape->data[0];

    {
        boxes_num = (int32_t)attr[1]->shape->data[1];
        for (bb = 0; bb < boxes_num; bb++)
        {
            b = (int32_t)buffer[2][bb];
            for (d = 0; d < depth; d++)
            {
                float y1 = buffer[1][bb * 4];
                float x1 = buffer[1][bb * 4 + 1];
                float y2 = buffer[1][bb * 4 + 2];
                float x2 = buffer[1][bb * 4 + 3];
                float height_scale = 0;
                float width_scale = 0;
                float value = 0;

                height_scale = (crop_height > 1) ? (y2 - y1) * (float)(image_height - 1) / (crop_height -1) : 0;
                width_scale = (crop_width > 1) ? (x2 - x1) * (float)(image_width - 1) / (crop_width -1) : 0;

                for (y = 0; y < crop_height; y++)
                {
                    float in_y = (crop_height > 1) ? y1 * (float)(image_height - 1) + y
                        * height_scale : 0.5f * (y1 + y2) * (float)(image_height - 1);
                    int32_t top_y_index = 0;
                    int32_t bottom_y_index = 0;
                    float y_lerp = 0;

                    if (resize_method == VSI_NN_INTERPOLATION_BILINEAR)
                    {
                        top_y_index = (int32_t)floorf(in_y);
                        bottom_y_index = (int32_t)ceilf(in_y);
                        y_lerp = in_y - top_y_index;
                        for (x = 0; x < crop_width; x++)
                        {
                            float in_x = (crop_width > 1) ? x1 * (image_width - 1) + x * width_scale
                                : 0.5f * (x1 + x2) * (image_width - 1);
                            int32_t left_x_index = 0;
                            int32_t right_x_index = 0;
                            float x_lerp = 0;
                            float top_left, top_right, bottom_left, bottom_right, top, bottom;
                            if (in_x < 0 || in_x > image_width - 1 || in_y < 0 || in_y > image_height -1)
                            {
                                buffer[3][bb * depth * crop_height * crop_width + d * crop_height
                                                     * crop_width + y * crop_width + x] = extrapolation_value;
                                continue;
                            }
                            left_x_index = (int32_t)floorf(in_x);
                            right_x_index = (int32_t)ceilf(in_x);
                            x_lerp = in_x - left_x_index;

                            top_left = buffer[0][b * depth * image_height * image_width + d * image_height *
                                                   image_width + top_y_index * image_width + left_x_index];
                            top_right = buffer[0][b * depth * image_height * image_width + d * image_height *
                                                   image_width + top_y_index * image_width + right_x_index];
                            bottom_left = buffer[0][b * depth * image_height * image_width + d * image_height *
                                                   image_width + bottom_y_index * image_width + left_x_index];
                            bottom_right = buffer[0][b * depth * image_height * image_width + d * image_height *
                                                   image_width + bottom_y_index * image_width + right_x_index];

                            top = top_left + (top_right - top_left) * x_lerp;
                            bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
                            value = top + (bottom - top) * y_lerp;
                            buffer[3][bb * depth * crop_height * crop_width + d * crop_height * crop_width
                                                             + y * crop_width + x] = value;
                        }
                    }
                    else
                    {
                        for (x = 0; x < crop_width; x++)
                        {
                            float in_x = (crop_width > 1) ? x1 * (image_width - 1) + x * width_scale
                                : 0.5f * (x1 + x2) * (image_width - 1);
                            int32_t closest_x_index, closest_y_index;
                            if (in_x < 0 || in_x > image_width - 1 || in_y < 0 || in_y > image_height -1)
                            {
                                buffer[3][bb * depth * crop_height * crop_width +
                                    d * crop_height * crop_width + y * crop_width + x] = extrapolation_value;
                                continue;
                            }
                            closest_x_index = (int32_t)ROUNDF(in_x);
                            closest_y_index = (int32_t)ROUNDF(in_y);
                            value = buffer[0][b * depth * image_height * image_width + d * image_height *
                                            image_width + closest_y_index * image_width + closest_x_index];
                            buffer[3][bb * depth * crop_height * crop_width +
                                d * crop_height * crop_width + y * crop_width + x] = value;
                        }
                    }
                }
            }
        }
    }
    status = vsi_nn_kernel_tensor_write_from_float( tensors[3], attr[3],
            buffer[3], out_elements );
    CHECK_STATUS_FAIL_GOTO( status, final );
final:
    for (i = 0; i < _INPUT_NUM + _OUTPUT_NUM; i++)
    {
        if ( buffer[i] )
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
    VSI_UNREFERENCED(inputs);
    VSI_UNREFERENCED(outputs);
    snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _KERNEL_NAME );
    kernel->info.function    = _compute;
    kernel->info.parameters  = _crop_and_resize_kernel_param_def;
    kernel->info.numParams   = _cnt_of_array( _crop_and_resize_kernel_param_def );
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
    vsi_nn_kernel_node_param_t node_params[_CROP_AND_RESIZE_PARAM_NUM] = {NULL};
    vsi_nn_kernel_node_t node = NULL;

    float extrapolation_value = vsi_nn_kernel_param_get_float32( params, "extrapolation_value" );
    int32_t resize_method = vsi_nn_kernel_param_get_int32( params, "resize_method" );

    status = _query_kernel( kernel, inputs, outputs );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _CROP_AND_RESIZE_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            node_params[4] = vsi_nn_kernel_scalar_create( graph, I32, &resize_method );
            node_params[5] = vsi_nn_kernel_scalar_create( graph, F32, &extrapolation_value );
            /* Pass parameters to node. */
            status  = vsi_nn_kernel_node_pass_param( node, node_params, _CROP_AND_RESIZE_PARAM_NUM );
            CHECK_STATUS( status );
            vsi_nn_kernel_scalar_release( &node_params[4] );
            vsi_nn_kernel_scalar_release( &node_params[5] );
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( crop_and_resize, _setup )

