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
#include <assert.h>
#include <string.h>
#include <stdlib.h>

#include "vsi_nn_prv.h"
#include "vsi_nn_test.h"
#include "vsi_nn_log.h"
#include "vsi_nn_types.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_error.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_dlfcn.h"
#include "kernel/vsi_nn_kernel.h"

typedef struct {
    void * lib_handle;
    vsi_bool exists;

    void (*conv2d_quant8)(
        const uint32_t* input_shape, const uint32_t* kernel_shape,
        const uint32_t* output_shape,
        float input_scale, float kernel_scale, float output_scale,
        int32_t input_zp, int32_t kernel_zp, int32_t output_zp,
        int32_t stride_h, int32_t stride_w,
        int32_t dilation_h, int32_t dilation_w,
        int32_t pad_h_front, int32_t pad_h_end,
        int32_t pad_w_front, int32_t pad_w_end,
        const uint8_t* input, const uint8_t* kernel,
        const int32_t* bias, uint8_t* output);

    void (*transpose_conv2d_quant8)(
        const uint32_t* input_shape, const uint32_t* kernel_shape,
        const uint32_t* output_shape,
        float input_scale, float kernel_scale, float output_scale,
        int32_t input_zp, int32_t kernel_zp, int32_t output_zp,
        int32_t stride_h, int32_t stride_w,
        int32_t dilation_h, int32_t dilation_w,
        int32_t pad_h_front, int32_t pad_h_end,
        int32_t pad_w_front, int32_t pad_w_end,
        const uint8_t* input, const uint8_t* kernel,
        const int32_t* bias, uint8_t* output);

} npuref_impl_t;

static npuref_impl_t s_npuref;
static void* _load_function( void* handle, const char* name, vsi_bool optional )
{
    void* fn = NULL;
    char * dl_error;
    if( NULL == handle )
    {
        return NULL;
    }
    fn = vsi_nn_dlsym( handle, name );
    dl_error = vsi_nn_dlerror();
    if( NULL == fn && !optional )
    {
        VSILOGW("Load symbol %s fail, reason: \"%s\"", name, dl_error);
    }
    return fn;
 } /* _load_function() */

#define LOAD_FUNCTION(handle, name) _load_function(handle, name, FALSE)
#define LOAD_FUNCTION_OPTIONAL(handle, name) _load_function(handle, name, TRUE)

static npuref_impl_t* npuref_load()
{
    void* libnpuref = NULL;
    char* dl_error;
#if (defined(_MSC_VER) || defined(_WIN32) || defined(__MINGW32))
    const char* libname = "libnpureference.dll";
#else
    const char* libname = "libnpureference.so";
#endif
    libnpuref = vsi_nn_dlopen(libname, RTLD_LAZY | RTLD_LOCAL);
    dl_error = vsi_nn_dlerror();
    if (NULL == libnpuref)
    {
        VSILOGD("Skip npuref lib, reason: \"%s\"", dl_error);
    }
    s_npuref.exists = (NULL != libnpuref);
    s_npuref.conv2d_quant8 = LOAD_FUNCTION( libnpuref, "npuref_conv2d_quant8" );
    s_npuref.transpose_conv2d_quant8 = LOAD_FUNCTION( libnpuref, "npuref_transpose_conv2d_quant8" );
    s_npuref.lib_handle = libnpuref;
    return &s_npuref;
} /* npuref_load() */

static const npuref_impl_t* npuref_impl()
{
    static npuref_impl_t* npuref = NULL;
    if( NULL == npuref ) {
        npuref = npuref_load();
        if( npuref->exists )
        {
            VSILOGD("Load npuref success.");
        }
    }
    return npuref;
} /* npuref_impl() */

vsi_bool npuref_exists()
{
    return npuref_impl()->exists;
} /* npuref_exists() */

/*
 * Documented in npuref_interface.h
 */
void npuref_interface_quant_conv2d(
    const void* input_buffer, const vsi_nn_kernel_tensor_attr_t* input_attr,
    const void* kernel_buffer, const vsi_nn_kernel_tensor_attr_t* kernel_attr,
    const void* bias_buffer,
    const int* pad, const int* strides, const int* dilation,
    const vsi_nn_kernel_tensor_attr_t* output_attr, void* output_buffer)
{
    npuref_impl()->conv2d_quant8(
        (uint32_t *)input_attr->shape->data, (uint32_t *)kernel_attr->shape->data,
        (uint32_t *)output_attr->shape->data,
        input_attr->asymm.scale, kernel_attr->asymm.scale, output_attr->asymm.scale,
        input_attr->asymm.zero_point, kernel_attr->asymm.zero_point,
        output_attr->asymm.zero_point,
        strides[1], strides[0],
        dilation[1], dilation[0],
        pad[2], pad[3], pad[0], pad[1],
        input_buffer, kernel_buffer, bias_buffer, output_buffer);
} /* npuref_interface_quant_conv2d() */

/*
 * Documented in npuref_interface.h
 */
void npuref_interface_quant_deconv2d(
    const void* input_buffer, const vsi_nn_kernel_tensor_attr_t* input_attr,
    const void* kernel_buffer, const vsi_nn_kernel_tensor_attr_t* kernel_attr,
    const void* bias_buffer,
    const int* pad, const int* strides, const int* dilation,
    const vsi_nn_kernel_tensor_attr_t* output_attr, void* output_buffer)
{
    npuref_impl()->transpose_conv2d_quant8(
        (uint32_t *)input_attr->shape->data, (uint32_t *)kernel_attr->shape->data,
        (uint32_t *)output_attr->shape->data,
        input_attr->asymm.scale, kernel_attr->asymm.scale, output_attr->asymm.scale,
        input_attr->asymm.zero_point, kernel_attr->asymm.zero_point,
        output_attr->asymm.zero_point,
        strides[1], strides[0],
        dilation[1], dilation[0],
        pad[2], pad[3], pad[0], pad[1],
        input_buffer, kernel_buffer, bias_buffer, output_buffer);
} /* npuref_interface_quant_deconv2d() */

void npuref_interface_quant_depthwise_conv2d(
    const void* input_buffer,
    const void* kernel_buffer,
    const void* bias_buffer,
    const int32_t* input_shape, uint32_t input_dim,
    const int32_t* kernel_shape, uint32_t kernel_dim,
    const int32_t* output_shape, uint32_t output_dim,
    float input_scale, int32_t input_zero_point,
    float kernel_scale, int32_t kernel_zero_point,
    float output_scale, int32_t output_zero_point,
    int32_t pad_h_front, int32_t pad_h_end,
    int32_t pad_w_front, int32_t pad_w_end,
    int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w,
    void* output_buffer)
{
    const int32_t group = input_shape[2];
    int32_t i;
    uint32_t sinput_shape[4] = { 0 };
    uint32_t skernel_shape[4] = { 0 };
    uint32_t soutput_shape[4] = { 0 };
    uint32_t input_ofset = 0;
    uint32_t output_ofset = 0;
    uint32_t kernel_ofset = 0;
    uint32_t multiplier = 0;
    memcpy( sinput_shape, input_shape, sizeof(uint32_t) * 4 );
    memcpy( skernel_shape, kernel_shape, sizeof(uint32_t) * 4 );
    memcpy( soutput_shape, output_shape, sizeof(uint32_t) * 4 );
    skernel_shape[3] = (int)(kernel_shape[2] / input_shape[2]);
    sinput_shape[2] = 1;
    skernel_shape[2] = 1;
    soutput_shape[2] = skernel_shape[3];
    multiplier = skernel_shape[3];
    VSI_ASSERT( sinput_shape[3] == 1 );
    //VSILOGD("!!! ORG shape - [%d %d %d %d] [%d %d %d %d] [%d %d %d %d]",
    //            input_shape[0], input_shape[1], input_shape[2], input_shape[3],
    //            kernel_shape[0], kernel_shape[1], kernel_shape[2], kernel_shape[3],
    //            output_shape[0], output_shape[1], output_shape[2], output_shape[3]);
    //VSILOGD("PAD %d %d", pad_h_front, pad_h_end);
    //VSILOGD("strde %d %d", stride_h, stride_w);
    for( i = 0; i < group; i ++ )
    {
        //VSILOGD("!!! %d !!!", i);
        //VSILOGD("!!! shape - [%d %d %d %d], [%d %d %d %d], [%d %d %d %d]",
        //        sinput_shape[0], sinput_shape[1], sinput_shape[2], sinput_shape[3],
        //        skernel_shape[0], skernel_shape[1], skernel_shape[2], skernel_shape[3],
        //        soutput_shape[0], soutput_shape[1], soutput_shape[2], soutput_shape[3]
        //        );
        //VSILOGD("!!! ofset - %d %d %d", input_ofset, kernel_ofset, output_ofset);
        npuref_impl()->conv2d_quant8(
            sinput_shape, skernel_shape, soutput_shape,
            input_scale, kernel_scale, output_scale,
            input_zero_point, kernel_zero_point,
            output_zero_point,
            stride_h, stride_w,
            dilation_h, dilation_w,
            pad_h_front, pad_h_end, pad_w_front, pad_w_end,
            ((const uint8_t*)input_buffer + input_ofset),
            ((const uint8_t*)kernel_buffer + kernel_ofset),
            bias_buffer ? ((const int32_t*)bias_buffer + multiplier*i) : NULL,
            ((uint8_t*)output_buffer + output_ofset));
        input_ofset += sinput_shape[0] * sinput_shape[1];
        kernel_ofset += skernel_shape[0] * skernel_shape[1];
        output_ofset += soutput_shape[0] * soutput_shape[1];
    }
} /* npuref_interface_quant_depthwise_conv2d() */

void npuref_init()
{
    memset( &s_npuref, 0, sizeof(s_npuref) );
    npuref_exists();
} /* npuref_init() */

void npuref_shutdown()
{
    if( s_npuref.exists )
    {
        vsi_nn_dlclose( s_npuref.lib_handle );
        memset( &s_npuref, 0, sizeof(s_npuref) );
    }
} /* npuref_shutdown() */

