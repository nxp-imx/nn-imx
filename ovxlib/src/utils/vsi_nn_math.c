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
#include <math.h>
#include <string.h>
#include "vsi_nn_tensor.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_types.h"
#include "utils/vsi_nn_math.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"

static void _compute_stride
    (
    uint32_t * shape,
    uint32_t   dim_num,
    uint32_t * stride
    );

static double _vsi_copysign
    (
    double number,
    double sign
    )
{
    double value = vsi_nn_abs(number);
    return (sign > 0) ? value : (-value);
} /* _vsi_copysign() */

static void _compute_stride
    (
    uint32_t * shape,
    uint32_t   dim_num,
    uint32_t * stride
    )
{
    int i;
    uint32_t s;
    s = 1;
    for( i = dim_num - 1; i >= 0; i -- )
    {
        stride[i] = s;
        s *= shape[i];
    }
} /* _compute_stride() */

float vsi_nn_SimpleRound
    (
    float x
    )
{
    return (float) _vsi_copysign(floorf(fabsf(x) + 0.5f), x);
} /* vsi_nn_SimplieRound() */

void vsi_nn_Transpose
    (
    uint8_t  * dst,
    uint8_t  * data,
    uint32_t * shape,
    uint32_t   dim_num,
    uint32_t * perm,
    vsi_nn_type_e type
    )
{
    uint32_t i;
    uint32_t i_dst;
    uint32_t i_org;
    uint32_t i_t;
    uint32_t size;
    uint32_t unit_bytes;
    uint32_t org_stride[VSI_NN_MAX_DIM_NUM];
    uint32_t dst_stride[VSI_NN_MAX_DIM_NUM];
    uint32_t dst_shape[VSI_NN_MAX_DIM_NUM];

    if( NULL == data || NULL == dst || NULL == shape || NULL == perm
        || 0 == dim_num || dim_num > VSI_NN_MAX_DIM_NUM )
    {
        return;
    }
    if( 1 == dim_num )
    {
        VSILOGW( "Transpose error, incorrect dim %d", dim_num );
        return;
    }
    for( i = 0; i < dim_num; i ++ )
    {
        if( perm[i] >= dim_num )
        {
            VSILOGW( "Incorrect perm %d", perm[i] );
            return;
        }
        dst_shape[i] = shape[perm[i]];
    }
    unit_bytes = vsi_nn_GetTypeBytes( type );
    _compute_stride( shape, dim_num, org_stride );
    _compute_stride( dst_shape, dim_num, dst_stride );
    size = vsi_nn_ShapeProduct( shape, dim_num );
    for( i_dst = 0; i_dst < size; i_dst ++ )
    {
        i_org = 0;
        i_t = i_dst;
        for( i = 0; i < dim_num; i ++ )
        {
            i_org += ( i_t / dst_stride[i] ) * org_stride[perm[i]];
            i_t %= dst_stride[i];
        }
        memcpy( &dst[i_dst * unit_bytes], &data[i_org * unit_bytes], unit_bytes );
        //dst[i_dst] = data[i_org];
    }
} /* vsi_nn_Transpose() */

void vsi_nn_SqueezeShape
    (
    uint32_t * shape,
    uint32_t * dim_num
    )
{
    int i;
    int origin_count;
    int count;
    int start;
    count = *dim_num;
    origin_count = count;
    if( 1 == count )
    {
        return;
    }
    start = 0;
    for( i = 0; i < count; i ++ )
    {
        if( 1 == shape[i] )
        {
            continue;
        }
        else if( i > start )
        {
            memmove( &shape[start], &shape[i], (count - i) * sizeof( uint32_t ) );
            count -= i - start;
            start += i - start;
        }
        else
        {
            start = i + 1;
        }
    }
    *dim_num = count;
    memset( &shape[count], 0, sizeof( uint32_t ) * ( origin_count - count ) );
} /* vsi_nn_SqueezeShape() */

uint32_t vsi_nn_ShapeProduct
    (
    uint32_t * shape,
    uint32_t   dim_num
    )
{
    uint32_t i;
    uint32_t res;
    res = 1;
    for ( i = 0; i < dim_num; i++ )
    {
        res *= shape[i];
    }
    return res;
} /* vsi_nn_ShapeProduct() */

void vsi_nn_InvertShape
    (
    uint32_t * in,
    uint32_t   dim_num,
    uint32_t * out
    )
{
    uint32_t i;
    for ( i = 0; i < dim_num; i++ )
    {
        out[i] = in[dim_num - 1 - i];
    }
} /* vsi_nn_InvertShape() */

void vsi_nn_InvertPermuteShape
    (
    uint32_t * in,
    uint32_t   dim_num,
    uint32_t * out
    )
{
    uint32_t i;
    for ( i = 0; i < dim_num; i++ )
    {
        out[in[i]] = i;
    }
} /* vsi_nn_InvertPermuteShape() */

double vsi_nn_Rint
    (
    double x
    )
{
#define _EPSILON 1e-8

    double decimal;
    double inter;

    decimal = modf((double)x, &inter);

    if( vsi_nn_abs((vsi_nn_abs(decimal) - 0.5f)) < _EPSILON )
    {
        inter += (int32_t)(inter) % 2;
    }
    else
    {
        return vsi_nn_SimpleRound( (float)x );
    }

    return inter;
} /* vsi_nn_Rint() */

