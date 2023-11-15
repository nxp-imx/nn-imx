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

__BEGIN_DECLS

#define _CPU_ARG_NUM            (13)
#define _CPU_INPUT_NUM          (2)
#define _CPU_OUTPUT_NUM         (1)
#define _CPU_IO_NUM             (_CPU_INPUT_NUM + _CPU_OUTPUT_NUM)
#define _CPU_PARAM_NUM          (_CPU_ARG_NUM + _CPU_IO_NUM)
#define _KERNEL_NAME            CVIVANTE_NAMESPACE("cpu.pre_process_nv12_rggb_sw")

#define DESCALE(x) (((x) + (1<<19)) >> 20)

static vsi_bool _check_nv12_type_from_env()
{
    vsi_bool ret = FALSE;
    char* env_s = vsi_nn_getenv("VSI_NN_ENABLE_OCV_NV12");
    if (env_s)
    {
        ret = TRUE;
    }
    return ret;
}

DEF_KERNEL_EXECUTOR(_pre_process_nv12_rggb_exec)
    (
    vsi_nn_kernel_node_t node,
    const vsi_nn_kernel_node_param_t * param,
    size_t param_size
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_tensor_t tensors[_CPU_IO_NUM] = { NULL };
    float * buffer[_CPU_IO_NUM] = { NULL };
    float * outBuffer = NULL;
    size_t out_elements = 0;
    vsi_nn_kernel_tensor_attr_t * attr[_CPU_IO_NUM] = { NULL };
    uint32_t i = 0;
    int32_t xRatio = 0, yRatio = 0, xOffset = 0, yOffset = 0;
    float rMean = 0, gMean = 0, bMean = 0;
    float r_scale = 0, g_scale = 0, b_scale = 0;
    int32_t order = 0, trans = 0, nv_type = 0;

    VSI_UNREFERENCED(node);
    VSI_UNREFERENCED(param_size);

    tensors[0]  = (vsi_nn_kernel_tensor_t)param[0];
    tensors[1]  = (vsi_nn_kernel_tensor_t)param[1];
    tensors[2]  = (vsi_nn_kernel_tensor_t)param[2];

    attr[0] = vsi_nn_kernel_tensor_attr_create( tensors[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );
    attr[1] = vsi_nn_kernel_tensor_attr_create( tensors[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );
    attr[2] = vsi_nn_kernel_tensor_attr_create( tensors[2] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", final );

    out_elements = vsi_nn_kernel_tensor_attr_get_size( attr[2] );

    i = 3;
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[i++], &xRatio);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[i++], &yRatio);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[i++], &xOffset);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[i++], &yOffset);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[i++], &rMean);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[i++], &gMean);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[i++], &bMean);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[i++], &r_scale);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[i++], &order);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[i++], &trans);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[i++], &nv_type);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[i++], &g_scale);
    CHECK_STATUS_FAIL_GOTO(status, final );
    status = vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[i++], &b_scale);
    CHECK_STATUS_FAIL_GOTO(status, final );

    buffer[0] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[0], attr[0], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[0], "Create input0 buffer fail.", final );

    buffer[1] = (float*)vsi_nn_kernel_tensor_create_buffer( tensors[1], attr[1], TRUE );
    CHECK_PTR_FAIL_GOTO( buffer[1], "Create input1 buffer fail.", final );

    buffer[2] = (float *)malloc( out_elements * sizeof(float) );
    CHECK_PTR_FAIL_GOTO( buffer[2], "Create output buffer fail.", final );
    memset( buffer[2], 0, out_elements * sizeof(float) );

    if (trans)
    {
        outBuffer = (float *)malloc( out_elements * sizeof(float) );
        CHECK_PTR_FAIL_GOTO( outBuffer, "Create output buffer fail.", final );
        memset( outBuffer, 0, out_elements * sizeof(float) );
    }

    {
        int32_t dx, dy, dz;
        int32_t src_width = (int32_t)attr[0]->shape->data[0];
        int32_t dst_width = (int32_t)(trans ? attr[2]->shape->data[1] : attr[2]->shape->data[0]);
        int32_t dst_height = (int32_t)(trans ? attr[2]->shape->data[2] : attr[2]->shape->data[1]);
        int32_t stride = (int32_t)(dst_width * dst_height);
        int32_t rOffset = 0;
        int32_t g0Offset = 1 * stride;
        int32_t g1Offset = 2 * stride;
        int32_t bOffset = 3 * stride;
        float D = 0;
        float E = 0;
        uint32_t R = 0;
        uint32_t G = 0;
        uint32_t B = 0;
        float* src_y_slice = NULL;
        float* src_uv_yScanline = NULL;

        uint32_t roi_width = (xRatio * dst_width) >> 15;
        uint32_t roi_height = (yRatio * dst_height) >> 15;
        uint32_t xrIntFloat_16 = (roi_width << 16) / dst_width + 1;
        uint32_t yrIntFloat_16 = (roi_height << 16) / dst_height + 1;
        uint32_t srcy = 0, srcx = 0;
        vsi_bool ocv_nv12 = _check_nv12_type_from_env();

        if (order)
        {
            rOffset = 3 * stride;
            bOffset = 0;
        }

        if (nv_type == VSI_NN_YUV_TYPE_NV21_BGGR)
        {
            int tmpOffset = rOffset;
            rOffset = bOffset;
            bOffset = tmpOffset;
        }

        for ( dz = 0; dz < 1; dz ++)
        {
            for ( dy = 0; dy < (int32_t)dst_height; dy ++)
            {
                srcy = (((uint32_t)dy * yrIntFloat_16) >> 16) + yOffset;
                src_y_slice = buffer[0] + (srcy) * src_width;
                src_uv_yScanline = buffer[1] + (srcy / 2) * src_width;

                for ( dx = 0; dx < (int32_t)dst_width; dx ++)
                {
                    float finalVal = 0;
                    int32_t output_index = 0;
                    int32_t dstR_idx = 0, dstG0_idx = 0, dstG1_idx = 0, dstB_idx = 0;
                    float tmpY = 0.0f;
                    float tmpU = 0.0f;
                    float tmpV = 0.0f;

                    srcx = (((uint32_t)dx * xrIntFloat_16) >> 16) + xOffset;
                    tmpY = src_y_slice[srcx];
                    if (nv_type == VSI_NN_YUV_TYPE_NV12_RGGB)
                    {
                        tmpU = src_uv_yScanline[(srcx / 2) * 2];
                        tmpV = src_uv_yScanline[(srcx / 2) * 2 + 1];
                    }
                    else
                    {
                        tmpU = src_uv_yScanline[(srcx / 2) * 2 + 1];
                        tmpV = src_uv_yScanline[(srcx / 2) * 2];
                    }

                    D = (tmpU - 128);
                    E = (tmpV - 128);

                    if (ocv_nv12)
                    {
                        B = (uint32_t)vsi_clamp((1.164 * (tmpY - 16) + 2.018 * D + 0.5), 0, 255);
                        G = (uint32_t)vsi_clamp((1.164 * (tmpY - 16) - 0.391 * D - 0.813 * E + 0.5), 0, 255);
                        R = (uint32_t)vsi_clamp((1.164 * (tmpY - 16) + 1.596 * E + 0.5), 0, 255);
                    }
                    else
                    {
                        B = (uint32_t)vsi_clamp((tmpY + (1.7790 * D)), 0, 255);
                        G = (uint32_t)vsi_clamp((tmpY - 0.3455 * D - 0.7169 * E), 0, 255);
                        R = (uint32_t)vsi_clamp((tmpY + 1.4065 * E), 0, 255);
                    }

                    output_index = dx + dy * dst_width;

                    dstR_idx  = output_index + rOffset;
                    dstG0_idx = output_index + g0Offset;
                    dstG1_idx = output_index + g1Offset;
                    dstB_idx  = output_index + bOffset;

                    finalVal = (B - bMean) * b_scale;
                    buffer[2][dstB_idx] = finalVal;

                    finalVal = (G - gMean) * g_scale;
                    buffer[2][dstG0_idx] = finalVal;
                    buffer[2][dstG1_idx] = finalVal;

                    finalVal = (R - rMean) * r_scale;
                    buffer[2][dstR_idx] = finalVal;
                }
            }
        }
    }

    if (trans)
    {
        vsi_size_t shape[] = { 0, 0, 0, 0 };
        vsi_size_t perm[] = {1, 2, 0, 3};
        shape[0] = attr[2]->shape->data[0];
        shape[1] = attr[2]->shape->data[1];
        shape[2] = attr[2]->shape->data[2];
        shape[3] = 1;
        vsi_nn_Transpose((uint8_t*)outBuffer, (uint8_t*)buffer[2],
                        shape, (uint32_t)attr[2]->shape->size, perm, VSI_NN_TYPE_FLOAT32);

        status = vsi_nn_kernel_tensor_write_from_float( tensors[2], attr[2],
            outBuffer, out_elements );
        CHECK_STATUS_FAIL_GOTO( status, final );
    }
    else
    {
        status = vsi_nn_kernel_tensor_write_from_float( tensors[2], attr[2],
                buffer[2], out_elements );
        CHECK_STATUS_FAIL_GOTO( status, final );
    }

final:
    if (outBuffer)
    {
        free (outBuffer);
    }
    for ( i = 0; i < _CPU_IO_NUM; i ++ )
    {
        if ( buffer[i] )
        {
            free ( buffer[i] );
        }
        if (attr[i]) { vsi_nn_kernel_tensor_attr_release( &attr[i] ); }
    }
    return status;
} /* _pre_process_nv12_rggb_exec() */

static vx_param_description_t kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
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
    kernel->info.function    = _pre_process_nv12_rggb_exec;
    kernel->info.parameters  = kernel_param_def;
    kernel->info.numParams   = _cnt_of_array( kernel_param_def );

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

    VSI_UNREFERENCED(input_num);
    VSI_UNREFERENCED(output_num);

    status = _query_kernel( inputs, outputs, kernel );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            uint32_t index   = 3;
            int32_t scale_x  = vsi_nn_kernel_param_get_int32( params, "scale_x" );
            int32_t scale_y  = vsi_nn_kernel_param_get_int32( params, "scale_y" );
            int32_t left     = vsi_nn_kernel_param_get_int32( params, "left" );
            int32_t top      = vsi_nn_kernel_param_get_int32( params, "top" );
            float r_mean     = vsi_nn_kernel_param_get_float32( params, "r_mean" );
            float g_mean     = vsi_nn_kernel_param_get_float32( params, "g_mean" );
            float b_mean     = vsi_nn_kernel_param_get_float32( params, "b_mean" );
            float r_scale    = vsi_nn_kernel_param_get_float32( params, "r_scale" );
            float g_scale    = vsi_nn_kernel_param_get_float32( params, "g_scale" );
            float b_scale    = vsi_nn_kernel_param_get_float32( params, "b_scale" );
            int32_t reverse  = vsi_nn_kernel_param_get_int32( params, "reverse" );
            int32_t trans    = vsi_nn_kernel_param_get_int32( params, "enable_perm" );
            int32_t nv_type  = vsi_nn_kernel_param_get_int32( params, "nv_type" );

            /* Set inputs and outputs */
            vsi_nn_kernel_node_pack_io( backend_params, _CPU_PARAM_NUM,
                    inputs, _CPU_INPUT_NUM, outputs, _CPU_OUTPUT_NUM );

            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &scale_x );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &scale_y );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &left );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &top );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &r_mean );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &g_mean );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &b_mean );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &r_scale );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &reverse );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &trans );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &nv_type );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &g_scale );
            backend_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &b_scale );
            /* Pass parameters to node. */
            status = vsi_nn_kernel_node_pass_param( node, backend_params, _CPU_PARAM_NUM );
            CHECK_STATUS( status );
            vsi_nn_kernel_scalar_release( &backend_params[3] );
            vsi_nn_kernel_scalar_release( &backend_params[4] );
            vsi_nn_kernel_scalar_release( &backend_params[5] );
            vsi_nn_kernel_scalar_release( &backend_params[6] );
            vsi_nn_kernel_scalar_release( &backend_params[7] );
            vsi_nn_kernel_scalar_release( &backend_params[8] );
            vsi_nn_kernel_scalar_release( &backend_params[9] );
            vsi_nn_kernel_scalar_release( &backend_params[10] );
            vsi_nn_kernel_scalar_release( &backend_params[11] );
            vsi_nn_kernel_scalar_release( &backend_params[12] );
            vsi_nn_kernel_scalar_release( &backend_params[13] );
            vsi_nn_kernel_scalar_release( &backend_params[14] );
            vsi_nn_kernel_scalar_release( &backend_params[15] );
        }
        else
        {
            status = VSI_FAILURE;
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_CPU( pre_process_nv12_rggb, _setup )
