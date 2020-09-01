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
#include "vsi_nn_pub.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "utils/vsi_nn_dtype_util.h"
#include "kernel/vsi_nn_gpu.h"

static int test_quantize_multiplier()
{
    const double data[] =
    {
        0.123456789, 1.23456789, 12.3456789, 123.3456789, 0.012345678,
        0.987654321, 9.87654321, 98.7654321, 987.654321, 0.0987654321,
        1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 0.0
    };
    const size_t s = _cnt_of_array( data );
    size_t i = 0;
    uint16_t mul0 = 0;
    int8_t shift0 = 0;
    uint16_t mul1 = 0;
    int32_t shift1 = 0;
    int ret = TRUE;

    for( i = 0; i < s; i ++ )
    {
        vsi_nn_GetFP32MultiAndPostShift(data[i], &mul0, &shift0);
        gpu_quantize_multiplier_16bit( data[i], &mul1, &shift1 );
        if( mul0 != mul1 || shift0 != shift1 )
        {
            VSILOGE("%.15lf, mul(%d vs %d), shift(%d vs %d)",data[i], mul0, mul1, shift0, shift1);
            ret = FALSE;
        }
    }
    return ret;
} /* test_quantize_multiplier() */

int main( int argc, char* argv[] )
{
    int ret = test_quantize_multiplier();
    if( ret )
    {
        VSILOGI("Pass");
    }
    return ret;
}
