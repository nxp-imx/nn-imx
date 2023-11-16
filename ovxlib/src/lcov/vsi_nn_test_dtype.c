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
#include <string.h>
#include <stdlib.h>
#include "vsi_nn_test.h"
#include "vsi_nn_pub.h"
#include "lcov/vsi_nn_test_dtype.h"
#define BUF_SIZE (64)

static vsi_status vsi_nn_test_dtype_convert_float_to_dtype(void)
{
    VSILOGI("vsi_nn_test_dtype_convert_float_to_dtype");
    float temp[1] = {1};
    const float *buffer;
    buffer = temp;
    size_t size = 1;
    vsi_nn_kernel_dtype_e dtype = U8;
    void *out_buffer;
    out_buffer = temp;
    vsi_bool result;
    result = vsi_nn_dtype_convert_float_to_dtype(buffer, size, dtype, out_buffer);
    if(result != 1) return VSI_FAILURE;

    dtype = U16;
    result = vsi_nn_dtype_convert_float_to_dtype(buffer, size, dtype, out_buffer);
    if(result != 1) return VSI_FAILURE;

    dtype = U32;
    result = vsi_nn_dtype_convert_float_to_dtype(buffer, size, dtype, out_buffer);
    if(result != 1) return VSI_FAILURE;

    dtype = U32;
    result = vsi_nn_dtype_convert_float_to_dtype(buffer, size, dtype, out_buffer);
    if(result != 1) return VSI_FAILURE;

    dtype = BF16;
    result = vsi_nn_dtype_convert_float_to_dtype(buffer, size, dtype, out_buffer);
    if(result != 1) return VSI_FAILURE;

    VSILOGI("vsi_nn_dtype_convert_dtype_to_float");
    dtype = I16;
    result = vsi_nn_dtype_convert_dtype_to_float(buffer, size, dtype, out_buffer);
    if(result != 1) return VSI_FAILURE;

    dtype = U8;
    result = vsi_nn_dtype_convert_dtype_to_float(buffer, size, dtype, out_buffer);
    if(result != 1) return VSI_FAILURE;

    dtype = U16;
    result = vsi_nn_dtype_convert_dtype_to_float(buffer, size, dtype, out_buffer);
    if(result != 1) return VSI_FAILURE;

    dtype = U32;
    result = vsi_nn_dtype_convert_dtype_to_float(buffer, size, dtype, out_buffer);
    if(result != 1) return VSI_FAILURE;

    dtype = F16;
    result = vsi_nn_dtype_convert_dtype_to_float(buffer, size, dtype, out_buffer);
    if(result != 1) return VSI_FAILURE;

    dtype = BF16;
    result = vsi_nn_dtype_convert_dtype_to_float(buffer, size, dtype, out_buffer);
    if(result != 1) return VSI_FAILURE;

    dtype = I4;
    result = vsi_nn_dtype_convert_dtype_to_float(buffer, size, dtype, out_buffer);
    if(result != 0) return VSI_FAILURE;
    return VSI_SUCCESS;
}

static vsi_status vsi_nn_test_dtype_convert_float_to_quantize_asymm(void)
{
    VSILOGI("vsi_nn_test_dtype_convert_float_to_quantize_asymm");
    float temp[1] = {1};
    const float *buffer;
    buffer = temp;
    size_t size = 1;
    void *out_buffer;
    out_buffer = temp;
    float scale = 1.0;
    int32_t zero_point = 0;
    vsi_bool result;

    vsi_nn_kernel_dtype_e dtype = I4;
    result = vsi_nn_dtype_convert_float_to_quantize_asymm(buffer, size, dtype, scale, zero_point, out_buffer);
    if(result != 1) return VSI_FAILURE;

    dtype = U4;
    result = vsi_nn_dtype_convert_float_to_quantize_asymm(buffer, size, dtype, scale, zero_point, out_buffer);
    if(result != 1) return VSI_FAILURE;

    dtype = I8;
    result = vsi_nn_dtype_convert_float_to_quantize_asymm(buffer, size, dtype, scale, zero_point, out_buffer);
    if(result != 1) return VSI_FAILURE;

    dtype = FP8_E4M3;
    result = vsi_nn_dtype_convert_float_to_quantize_asymm(buffer, size, dtype, scale, zero_point, out_buffer);
    if(result != 1) return VSI_FAILURE;

    dtype = FP8_E5M2;
    result = vsi_nn_dtype_convert_float_to_quantize_asymm(buffer, size, dtype, scale, zero_point, out_buffer);
    if(result != 1) return VSI_FAILURE;

    dtype = I16;
    result = vsi_nn_dtype_convert_float_to_quantize_asymm(buffer, size, dtype, scale, zero_point, out_buffer);
    if(result != 1) return VSI_FAILURE;

    dtype = U16;
    result = vsi_nn_dtype_convert_float_to_quantize_asymm(buffer, size, dtype, scale, zero_point, out_buffer);
    if(result != 1) return VSI_FAILURE;

    dtype = BOOL8;
    result = vsi_nn_dtype_convert_float_to_quantize_asymm(buffer, size, dtype, scale, zero_point, out_buffer);
    if(0 == result){
        return VSI_SUCCESS;
    }
    return VSI_FAILURE;
}

static vsi_status vsi_nn_test_dtype_convert_float_to_quantize_symm(void)
{
    VSILOGI("vsi_nn_test_dtype_convert_float_to_quantize_symm");
    float temp[1] = {1};
    const float *buffer;
    buffer = temp;
    size_t size = 1;
    int64_t buf[2] = {0};
    void *out_buffer;
    out_buffer = buf;
    float scale = 1.0;
    int32_t zero_point = 0;
    vsi_bool result;

    vsi_nn_kernel_dtype_e dtype = I32;
    result = vsi_nn_dtype_convert_float_to_quantize_symm(buffer, size, dtype, scale, zero_point, out_buffer);
    if(result != 1) return VSI_FAILURE;

    dtype = I64;
    result = vsi_nn_dtype_convert_float_to_quantize_symm(buffer, size, dtype, scale, zero_point, out_buffer);
    if(result != 1) return VSI_FAILURE;

    dtype = U4;
    result = vsi_nn_dtype_convert_float_to_quantize_symm(buffer, size, dtype, scale, zero_point, out_buffer);
    if(result != 0) return VSI_FAILURE;

    return VSI_SUCCESS;
}

static vsi_status vsi_nn_test_dtype_convert_quantize_asymm_to_float(void)
{
    VSILOGI("vsi_nn_test_dtype_convert_quantize_asymm_to_float");
    float temp[1] = {1};
    const float *buffer;
    buffer = temp;
    size_t size = 1;
    void *out_buffer;
    out_buffer = temp;
    float scale = 1.0;
    int32_t zero_point = 0;
    vsi_bool result;

    vsi_nn_kernel_dtype_e dtype = FP8_E4M3;
    result = vsi_nn_dtype_convert_quantize_asymm_to_float(buffer, size, dtype, scale, zero_point, out_buffer);
    if(result != 1) return VSI_FAILURE;

    dtype = FP8_E5M2;
    result = vsi_nn_dtype_convert_quantize_asymm_to_float(buffer, size, dtype, scale, zero_point, out_buffer);
    if(result != 1) return VSI_FAILURE;

    dtype = I64;
    result = vsi_nn_dtype_convert_quantize_asymm_to_float(buffer, size, dtype, scale, zero_point, out_buffer);
    if(result != 0) return VSI_FAILURE;

    dtype = U8;
    result = vsi_nn_dtype_convert_quantize_symm_to_float(buffer, size, dtype, scale, zero_point, out_buffer);
    if(result != 0) return VSI_FAILURE;

    return VSI_SUCCESS;
}

static vsi_status vsi_nn_test_dtype_convert_float_to_quantize_symm8_perchannel(void)
{
    float buf_tmp[1] = {1.0};
    const float *buffer = buf_tmp;
    size_t size = 1;
    vsi_size_t shape_tmp[1] = {1};
    const vsi_size_t *shape = shape_tmp;
    size_t rank = 1;
    float scale_tmp[1] = {1.0};
    const float *scale = scale_tmp;
    size_t scale_size = 1;
    int32_t zp_tmp[1] = {0};
    const int32_t *zero_point = zp_tmp;
    size_t zero_point_size = 1;
    int32_t channel_dim = 1;
    int8_t *out_buffer = NULL;
    float *out_buffer1 = NULL;
    vsi_bool result;

    VSILOGI("vsi_nn_test_dtype_convert_float_to_quantize_symm_perchannel");
    vsi_nn_kernel_dtype_e dtype = I8;
    result = vsi_nn_dtype_convert_float_to_quantize_symm_perchannel
    (
        buffer, size, dtype, shape, rank, scale, scale_size,
        zero_point, zero_point_size, channel_dim, out_buffer
    );
    if(result != 0) return VSI_FAILURE;

    VSILOGI("vsi_nn_test_dtype_convert_quantize_symm_perchannel_to_float");
    result = vsi_nn_dtype_convert_quantize_symm_perchannel_to_float
    (
        buffer, size, dtype, shape, rank, scale, scale_size,
        zero_point, zero_point_size, channel_dim, out_buffer1
    );
    if(result != 0) return VSI_FAILURE;

     VSILOGI("vsi_nn_test_dtype_convert_float_to_quantize_symm_perchannel");
    dtype = U8;
    result = vsi_nn_dtype_convert_float_to_quantize_symm_perchannel
    (
        buffer, size, dtype, shape, rank, scale, scale_size,
        zero_point, zero_point_size, channel_dim, out_buffer
    );
    if(result != 0) return VSI_FAILURE;

    VSILOGI("vsi_nn_test_dtype_convert_quantize_symm_perchannel_to_float");
    result = vsi_nn_dtype_convert_quantize_symm_perchannel_to_float
    (
        buffer, size, dtype, shape, rank, scale, scale_size,
        zero_point, zero_point_size, channel_dim, out_buffer1
    );
    if(result != 0) return VSI_FAILURE;
    return VSI_SUCCESS;
}

vsi_status vsi_nn_test_dtype( void )
{
    vsi_status status = VSI_FAILURE;
    status = vsi_nn_test_dtype_convert_float_to_dtype();
    TEST_CHECK_STATUS(status, final);

    status = vsi_nn_test_dtype_convert_float_to_quantize_asymm();
    TEST_CHECK_STATUS(status, final);

    status = vsi_nn_test_dtype_convert_float_to_quantize_symm();
    TEST_CHECK_STATUS(status, final);

    status = vsi_nn_test_dtype_convert_quantize_asymm_to_float();
    TEST_CHECK_STATUS(status, final);

    status = vsi_nn_test_dtype_convert_float_to_quantize_symm8_perchannel();
    TEST_CHECK_STATUS(status, final);

final:
    return status;
}
