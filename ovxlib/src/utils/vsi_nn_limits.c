/****************************************************************************
*
*    Copyright (c) 2018 Vivante Corporation
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
#include "vsi_nn_types.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"

void vsi_nn_TypeGetRange
    (
    vsi_nn_type_e type,
    double  * max_range,
    double  * min_range
    )
{
    int32_t bits;
    double from, to;

    from = 0.0;
    to = 0.0;
    bits = vsi_nn_GetTypeBytes( type ) * 8;
    if( vsi_nn_TypeIsInteger( type ) )
    {
        if( vsi_nn_TypeIsSigned( type ) )
        {
            from = (double)(-(1L << (bits - 1)));
            to = (double)((1UL << (bits - 1)) - 1);
        }
        else
        {
            from = 0.0;
            to = (double)((1UL << bits) - 1);
        }
    }
    else
    {
        //  TODO: Add float
    }
    if( NULL != max_range )
    {
        *max_range = to;
    }
    if( NULL != min_range )
    {
        *min_range = from;
    }
} /* vsi_nn_TypeGetRange() */

