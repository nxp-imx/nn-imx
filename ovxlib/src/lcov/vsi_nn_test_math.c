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
#include "lcov/vsi_nn_test_math.h"
#define BUF_SIZE (64)

static vsi_status vsi_nn_test_random_init_for_philox_4x32_10(void)
{
    VSILOGI("vsi_nn_test_random_init_for_philox_4x32_10");
    uint32_t low = 0;
    uint32_t high = 0;
    vsi_nn_random_init_for_philox_4x32_10(low, high);
    return VSI_SUCCESS;
}

static vsi_status vsi_nn_test_random_generate_by_philox_4x32_10(void)
{
    VSILOGI("vsi_nn_test_random_generate_by_philox_4x32_10");
    uint32_t random_buf[BUF_SIZE] = {0};
    uint32_t len = 10;
    vsi_nn_random_generate_by_philox_4x32_10(random_buf, len);
    return VSI_SUCCESS;
}

static vsi_status vsi_nn_test_random_uniform_transform(void)
{
    VSILOGI("vsi_nn_test_random_uniform_transform");
    uint32_t random_buf[2] = {1, 2};
    float uniform_buf[2] = {0};
    uint32_t len = 2;
    vsi_nn_random_uniform_transform(random_buf, uniform_buf, len);
    return VSI_SUCCESS;
}

static vsi_status vsi_nn_test_erf_impl(void)
{
    VSILOGI("vsi_nn_test_erf_impl");
    float x = 1;
    float result;
    result = vsi_nn_erf_impl(x);
    float golden = 0.842701;
    float diff = result - golden;
    if(diff < 1e-5){
        return VSI_SUCCESS;
    }
    return VSI_FAILURE;
}

vsi_status vsi_nn_test_math( void )
{
    vsi_status status = VSI_FAILURE;
    status = vsi_nn_test_random_init_for_philox_4x32_10();
    TEST_CHECK_STATUS(status, final);

    status = vsi_nn_test_random_generate_by_philox_4x32_10();
    TEST_CHECK_STATUS(status, final);

    status = vsi_nn_test_random_uniform_transform();
    TEST_CHECK_STATUS(status, final);

    status = vsi_nn_test_erf_impl();
    TEST_CHECK_STATUS(status, final);

final:
    return status;
}