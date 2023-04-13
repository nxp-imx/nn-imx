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
#define _INPUT_NUM          (2)
#define _OUTPUT_NUM         (1)
#define _KERNEL_NAME        CVIVANTE_NAMESPACE("cpu.bilinear_grid_sample")


/*
 * Kernel params
 */
static vx_param_description_t _bilinear_grid_sample_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _BILINEAR_GRID_SAMPLE_PARAM_NUM  _cnt_of_array( _bilinear_grid_sample_kernel_param_def )

#define SCALAR_ALIGN_CORNERS (3)

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
    vsi_nn_kernel_tensor_t input[_INPUT_NUM] = {NULL};
    vsi_nn_kernel_tensor_t output[_OUTPUT_NUM] = {NULL};
    float* f32_in_buffer[_INPUT_NUM] = {NULL};
    float* f32_out_buffer[_OUTPUT_NUM] = {NULL};
    vsi_nn_kernel_tensor_attr_t* in_attr[_INPUT_NUM];
    vsi_nn_kernel_tensor_attr_t* out_attr[_OUTPUT_NUM];
    vsi_size_t out_stride_size[_OUTPUT_NUM][VSI_NN_MAX_DIM_NUM] = {{0}};
    vsi_size_t out_elements[_OUTPUT_NUM] = {0};
    uint32_t i;
    int32_t align_corners;
    vsi_size_t batch, input0_c, input0_h, input0_w, input0_batch_size;
    vsi_size_t input1_gh, input1_gw, input1_dim0, input1_batch_size;
    vsi_size_t output_batch_size;
    vsi_size_t b = 0, c = 0, w = 0, h = 0;

    VSI_UNREFERENCED(node);
    VSI_UNREFERENCED(param_size);
    /* prepare data */
    for (i = 0; i < _INPUT_NUM; i++)
    {
        input[i] = (vsi_nn_kernel_tensor_t)param[i];
        in_attr[i] = vsi_nn_kernel_tensor_attr_create(input[i]);
        CHECK_PTR_FAIL_GOTO( in_attr[i], "Create tensor attr buffer fail.", final );
        f32_in_buffer[i] = (float*)vsi_nn_kernel_tensor_create_buffer(
            input[i], in_attr[i], TRUE);
        CHECK_PTR_FAIL_GOTO(
            f32_in_buffer[i], "Create input0 buffer fail.", final);
    }
    for (i = 0; i < _OUTPUT_NUM; i++)
    {
        output[i] = (vsi_nn_kernel_tensor_t)param[i + _INPUT_NUM];
        out_attr[i] = vsi_nn_kernel_tensor_attr_create(output[i]);
        CHECK_PTR_FAIL_GOTO( out_attr[i], "Create tensor attr buffer fail.", final );
        vsi_nn_kernel_tensor_attr_get_stride(out_attr[i], out_stride_size[i]);
        out_elements[i] = vsi_nn_kernel_tensor_attr_get_size(out_attr[i]);
        f32_out_buffer[i] = (float*)malloc(out_elements[i] * sizeof(float));
        CHECK_PTR_FAIL_GOTO(
            f32_out_buffer[i], "Create output buffer fail.", final);
        memset(f32_out_buffer[i], 0, out_elements[i] * sizeof(float));
    }
    vsi_nn_kernel_scalar_read_int32(
        (vsi_nn_kernel_scalar_t)param[SCALAR_ALIGN_CORNERS], &(align_corners));

    batch = (vsi_size_t)in_attr[0]->shape->size > 3 ? in_attr[0]->shape->data[3]
                                                    : 1;
    input0_c = in_attr[0]->shape->data[2];
    input0_h = in_attr[0]->shape->data[1];
    input0_w = in_attr[0]->shape->data[0];
    input1_gh = in_attr[1]->shape->data[2];
    input1_gw = in_attr[1]->shape->data[1];
    input1_dim0 = in_attr[1]->shape->data[0];
    input1_batch_size = input1_gh * input1_gw * input1_dim0;
    input0_batch_size = input0_c * input0_h * input0_w;
    output_batch_size = input0_c * input1_gh * input1_gw;
    for (b = 0; b < batch; b++) {
        float* input0_ptr = f32_in_buffer[0] + b * input0_batch_size;
        float* input1_ptr = f32_in_buffer[1] + b * input1_batch_size;
        float* output_ptr = f32_out_buffer[0] + b * output_batch_size;

        for (h = 0; h < input1_gh; h++)
        {
            for (w = 0; w < input1_gw; w++)
            {
                vsi_size_t input1_index = 2 * (h * input1_gw + w);
                float x_f = input1_ptr[input1_index];
                float y_f = input1_ptr[input1_index + 1];
                int x0, y0, x1, y1;
                float wa, wb, wc, wd;

                if (align_corners)
                {
                    x_f = ((x_f + 1.0f) / 2.0f) * (input0_w - 1);
                    y_f = ((y_f + 1.0f) / 2.0f) * (input0_h - 1);
                }
                else
                {
                    x_f = ((x_f + 1.0f) * input0_w - 1.0f) / 2.0f;
                    y_f = ((y_f + 1.0f) * input0_h - 1.0f) / 2.0f;
                }
                x0 = (int)floorf(x_f);
                y0 = (int)floorf(y_f);
                x1 = x0 + 1;
                y1 = y0 + 1;
                wa = (x1 - x_f) * (y1 - y_f);
                wb = (x1 - x_f) * (y_f - y0);
                wc = (x_f - x0) * (y1 - y_f);
                wd = (x_f - x0) * (y_f - y0);

                x0 = x0 >= (int)input0_w ? -1 : x0;
                x1 = x1 >= (int)input0_w ? -1 : x1;
                y0 = y0 >= (int)input0_h ? -1 : y0;
                y1 = y1 >= (int)input0_h ? -1 : y1;
                for (c = 0; c < input0_c; c++)
                {
                    vsi_size_t output_index =
                        c * input1_gh * input1_gw + h * input1_gw + w;
                    vsi_size_t input0_base_index;
                    float* input0_c_ptr = NULL;
                    float v_00, v_01, v_10, v_11;

                    input0_base_index = c * input0_w * input0_h;
                    input0_c_ptr = input0_ptr + input0_base_index;
                    v_00 = x0 < 0
                               ? 0
                               : y0 < 0 ? 0 : input0_c_ptr[y0 * input0_w + x0];
                    v_01 = x1 < 0
                               ? 0
                               : y0 < 0 ? 0 : input0_c_ptr[y0 * input0_w + x1];
                    v_10 = x0 < 0
                               ? 0
                               : y1 < 0 ? 0 : input0_c_ptr[y1 * input0_w + x0];
                    v_11 = x1 < 0
                               ? 0
                               : y1 < 0 ? 0 : input0_c_ptr[y1 * input0_w + x1];
                    output_ptr[output_index] =
                        v_00 * wa + v_10 * wb + v_01 * wc + v_11 * wd;
                }
            }
        }
    }

    /* save data */
    for (i = 0; i < _OUTPUT_NUM; i++)
    {
        status = vsi_nn_kernel_tensor_write_from_float(
            output[i], out_attr[i], f32_out_buffer[i], out_elements[i]);
        CHECK_STATUS_FAIL_GOTO(status, final);
    }
final:
    for (i = 0; i < _INPUT_NUM; i++)
    {
        if (f32_in_buffer[i])
        {
            free(f32_in_buffer[i]);
            f32_in_buffer[i] = NULL;
        }
        vsi_nn_kernel_tensor_attr_release(&in_attr[i]);
    }
    for (i = 0; i < _OUTPUT_NUM; i++)
    {
        if (f32_out_buffer[i])
        {
            free(f32_out_buffer[i]);
            f32_out_buffer[i] = NULL;
        }
        vsi_nn_kernel_tensor_attr_release(&out_attr[i]);
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
    )
{
    vsi_status status = VSI_FAILURE;
    VSI_UNREFERENCED(inputs);
    VSI_UNREFERENCED(outputs);
    snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _KERNEL_NAME );
    kernel->info.function    = _compute;
    kernel->info.parameters  = _bilinear_grid_sample_kernel_param_def;
    kernel->info.numParams   = _cnt_of_array( _bilinear_grid_sample_kernel_param_def );
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
    vsi_nn_kernel_node_param_t node_params[_BILINEAR_GRID_SAMPLE_PARAM_NUM];
    vsi_nn_kernel_node_t node = NULL;

    int32_t align_corners =
        vsi_nn_kernel_param_get_int32(params, "align_corners");

    status = _query_kernel( kernel, inputs, outputs);
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( node_params, _BILINEAR_GRID_SAMPLE_PARAM_NUM,
                    inputs, input_num, outputs, output_num );
            node_params[SCALAR_ALIGN_CORNERS] =
                vsi_nn_kernel_scalar_create(graph, I32, &align_corners);
            /* Pass parameters to node. */
            status = vsi_nn_kernel_node_pass_param(
                node, node_params, _BILINEAR_GRID_SAMPLE_PARAM_NUM);
            VSI_ASSERT(status == VSI_SUCCESS);
            vsi_nn_kernel_scalar_release(&node_params[SCALAR_ALIGN_CORNERS]);
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( bilinear_grid_sample, _setup )

