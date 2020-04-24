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
#include "kernel/vsi_nn_kernel.h"

template<typename T>
static int compare_data(T * g, T * t, size_t size, float atol = 0.0f, float rtol = 0.0f)
{
    uint32_t i;
    for( i = 0; i < size; i ++ )
    {
        float range = fabs(atol + rtol * g[i]);
        if( fabs(t[i] - g[i]) > range )
        {
            VSILOGW("Mismatch %d %f vs %f, fprange:%f", i, g[i], t[i], range);
            return FALSE;
        }
    }
    return TRUE;
} /* compare_data() */

int test_half_to_float()
{
    const float atol = 5.0f * 0.0009765625f;
    const float rtol = 5.0f * 0.0009765625f;
    vsi_float16 t[] = { 0x00a8, 0x0011, 0x0002, 0x80a8, 0x8011,
        0x8002, 0x2fe7, 0x3cf0, 0x4a2c, 0x57b7, 0xafe7, 0xbcf0, 0xca2c, 0xd7b7 };
    float g[] = { 1.0013580e-05f, 1.0132790e-06f, 1.1920929e-07f, -1.0013580e-05f,
                -1.0132790e-06f, -1.1920929e-07f, 1.2347412e-01f, 1.2343750e+00f,
                1.2343750e+01f, 1.2343750e+02f, -1.2347412e-01f, -1.2343750e+00f,
                -1.2343750e+01f, -1.2343750e+02f };
    float data[_cnt_of_array(t)] = {0.0f};
    const size_t size = _cnt_of_array(t);

    vsi_nn_dtype_convert_dtype_to_float(t, size, F16, data);

    return compare_data(g, data, size, atol, rtol);
} /* test_half_to_float() */

int main()
{
    int ret = TRUE;
    int tmp;
    tmp = test_half_to_float();
    if( tmp )
    {
        VSILOGD("Pass");
    }
    else
    {
        VSILOGE("Fail");
    }
    ret &= tmp;
    return ret;
}

