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
#include <math.h>

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

    void (*im2col_8bit)(
        const void* im,
        const uint8_t zero_point,
        int channels,
        const uint32_t* im_size,
        const uint32_t* kernel_size,
        const int* pad,
        const int* strides,
        const int* dilation,
        void* col);

    void (*gemm_asymmetric_uint8)(
        const uint8_t* col_buffer,
        int32_t col_h,
        int32_t col_w,
        const uint8_t* kernel_buffer,
        int32_t kernel_h,
        int32_t kernel_w,
        const int32_t* bias_buffer,
        int input_zero_point,
        int kernel_zero_point,
        int output_zero_point,
        int multiplier,
        int shift,
        uint8_t* output_buffer );
} cbee_impl_t;

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

static cbee_impl_t* cbee_load()
{
    static cbee_impl_t cbee;
    void* libcbee = NULL;
    char * dl_error;
#if defined(_WIN32)
    libcbee = vsi_nn_dlopen_win32("libcbee.dll", RTLD_LAZY | RTLD_LOCAL);
    dl_error = vsi_nn_dlerror_win32();
#else
    libcbee = dlopen("libcbee.so", RTLD_LAZY | RTLD_LOCAL);
    dl_error = dlerror();
#endif
    if (NULL == libcbee)
    {
        VSILOGD("Skip cbee lib, reason: \"%s\"", dl_error);
    }
    cbee.exists = (NULL != libcbee);
    cbee.im2col_8bit = LOAD_FUNCTION( libcbee, "cbee_im2col_8bit" );
    cbee.gemm_asymmetric_uint8 = LOAD_FUNCTION( libcbee, "cbee_gemm_asymmetric_uint8" );
    return &cbee;
} /* cbee_load() */

static const cbee_impl_t* cbee_impl()
{
    static cbee_impl_t* cbee = NULL;
    if( NULL == cbee ) {
        cbee = cbee_load();
        if( cbee->exists )
        {
            VSILOGD("Load cbee success.");
        }
    }
    return cbee;
} /* cbee_impl() */

vsi_bool cbee_exists()
{
    return cbee_impl()->exists;
} /* cbee_exists() */

static void _compute_quantized_multiplier_and_shift(
        float multiplier, int* quantized_multiplier, int* shift)
{
    int s = 0;
    int64_t q = 0;

    assert(multiplier > 0.f);
    assert(multiplier < 1.f);

    while (multiplier < 0.5f) {
        multiplier *= 2.0f;
        s++;
    }
#define ROUND(x)        ((int)(x + 0.5f))
    q = ROUND(multiplier * (1ll<<31));
#undef ROUND
    assert(q <= (1ll << 31));
    if (q == (1ll << 31)) {
        q /= 2;
        s--;
    }
    assert(s >= 0);
    *quantized_multiplier = (int)q;
    *shift = s;
} /* _compute_quantized_multiplier_and_shift() */

/*
 * Documented in cbee_interface.h
 */
void cbee_interface_quant_conv2d(
    const void* input_buffer, const vsi_nn_tensor_attr_t* input_attr,
    const void* kernel_buffer, const vsi_nn_tensor_attr_t* kernel_attr,
    const void* bias_buffer,
    const int* pad, const int* strides, const int* dilation,
    const vsi_nn_tensor_attr_t* output_attr, void* output_buffer)
{
    uint8_t* col_buffer = NULL;
    float float_multiplier;
    int32_t quantized_multiplier = 0;
    int32_t shift = 0;
    int32_t col_h = 0;
    int32_t col_w = 0;
    col_h = kernel_attr->size[1] * kernel_attr->size[0] * input_attr->size[2];
    col_w = output_attr->size[1] * output_attr->size[0];
    col_buffer = (uint8_t*)malloc( col_h * col_w );
    TEST_CHECK_PTR( col_buffer, final );
    memset( col_buffer, 0, col_h * col_w );
    cbee_impl()->im2col_8bit( input_buffer, input_attr->dtype.zero_point,
            input_attr->size[2], &input_attr->size[0],
            &kernel_attr->size[0], pad, strides, dilation, col_buffer );
    float_multiplier = (input_attr->dtype.scale * kernel_attr->dtype.scale)\
                       / output_attr->dtype.scale;
    _compute_quantized_multiplier_and_shift( float_multiplier, &quantized_multiplier, &shift );
    //VSILOGD("multiplier: %f, %d, %d", float_multiplier, quantized_multiplier, shift);
    cbee_impl()->gemm_asymmetric_uint8( col_buffer, col_h, col_w,
            (const uint8_t *)kernel_buffer, (int32_t)kernel_attr->size[3], col_h,
            (const int32_t *)bias_buffer, input_attr->dtype.zero_point,
            kernel_attr->dtype.zero_point, output_attr->dtype.zero_point,
            quantized_multiplier, shift, (uint8_t *)output_buffer );
final:
    if( col_buffer )
    {
        free( col_buffer );
    }
} /* cbee_interface_quant_conv2d() */

