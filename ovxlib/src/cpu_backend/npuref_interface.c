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
#include <assert.h>
#include <string.h>
#include <stdlib.h>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif
#include "vsi_nn_prv.h"
#include "vsi_nn_test.h"
#include "vsi_nn_log.h"
#include "vsi_nn_types.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"

typedef struct {
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

static void* _load_function( void* handle, const char* name, vsi_bool optional )
{
    void* fn = NULL;
    char * dl_error;
    if( NULL == handle )
    {
        return NULL;
    }
#if defined(_WIN32)
    fn = vsi_nn_dlsym_win32( handle, name );
    dl_error = vsi_nn_dlerror_win32();
#else
    fn = dlsym( handle, name );
    dl_error = dlerror();
#endif
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
    static npuref_impl_t npuref;
    void* libnpuref = NULL;
    char * dl_error;
#if defined(_WIN32)
    libnpuref = vsi_nn_dlopen_win32("libnpureference.dll", RTLD_LAZY | RTLD_LOCAL);
    dl_error = vsi_nn_dlerror_win32();
#else
    libnpuref = dlopen("libnpureference.so", RTLD_LAZY | RTLD_LOCAL);
    dl_error = dlerror();
#endif
    if (NULL == libnpuref)
    {
        VSILOGD("Skip npuref lib, reason: \"%s\"", dl_error);
    }
    npuref.exists = (NULL != libnpuref);
    npuref.conv2d_quant8 = LOAD_FUNCTION( libnpuref, "npuref_conv2d_quant8" );
    npuref.transpose_conv2d_quant8 = LOAD_FUNCTION( libnpuref, "npuref_transpose_conv2d_quant8" );
    return &npuref;
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
    const void* input_buffer, const vsi_nn_tensor_attr_t* input_attr,
    const void* kernel_buffer, const vsi_nn_tensor_attr_t* kernel_attr,
    const void* bias_buffer,
    const int* pad, const int* strides, const int* dilation,
    const vsi_nn_tensor_attr_t* output_attr, void* output_buffer)
{
    npuref_impl()->conv2d_quant8(
        input_attr->size, kernel_attr->size, output_attr->size,
        input_attr->dtype.scale, kernel_attr->dtype.scale, output_attr->dtype.scale,
        input_attr->dtype.zero_point, kernel_attr->dtype.zero_point,
        output_attr->dtype.zero_point,
        strides[1], strides[0],
        dilation[1], dilation[0],
        pad[2], pad[3], pad[0], pad[1],
        input_buffer, kernel_buffer, bias_buffer, output_buffer);
} /* npuref_interface_quant_conv2d() */

/*
 * Documented in npuref_interface.h
 */
void npuref_interface_quant_deconv2d(
    const void* input_buffer, const vsi_nn_tensor_attr_t* input_attr,
    const void* kernel_buffer, const vsi_nn_tensor_attr_t* kernel_attr,
    const void* bias_buffer,
    const int* pad, const int* strides, const int* dilation,
    const vsi_nn_tensor_attr_t* output_attr, void* output_buffer)
{
    npuref_impl()->transpose_conv2d_quant8(
        input_attr->size, kernel_attr->size, output_attr->size,
        input_attr->dtype.scale, kernel_attr->dtype.scale, output_attr->dtype.scale,
        input_attr->dtype.zero_point, kernel_attr->dtype.zero_point,
        output_attr->dtype.zero_point,
        strides[1], strides[0],
        dilation[1], dilation[0],
        pad[2], pad[3], pad[0], pad[1],
        input_buffer, kernel_buffer, bias_buffer, output_buffer);
} /* npuref_interface_quant_conv2d() */
