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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <VX/vx_khr_cnn.h>

#include "vsi_nn_pub.h"
#include "vsi_nn_test.h"

static void _set_dtype
    (
    vsi_nn_dtype_t * dtype,
    vx_float32  max,
    vx_float32  min
    )
{
if( vsi_nn_TypeIsInteger( dtype->vx_type ) )
    {
    if( vsi_nn_TypeIsSigned( dtype->vx_type ) )
        {
        dtype->qnt_type = VSI_NN_QNT_TYPE_DFP;
        vsi_nn_QuantDFPCalParam( dtype->vx_type, max, min, &dtype->fl );
        }
    else
        {
        dtype->qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
        vsi_nn_QuantAffineCalParam( dtype->vx_type, max, min,
                &dtype->scale, &dtype->zero_point );
        }
    }
} /* _set_dtype() */

static vx_bool _equal
    (
    vx_float32 a,
    vx_float32 b,
    vsi_nn_dtype_t * dtype
    )
{
vx_bool ret;
vx_float32 error;
ret = vx_false_e;
switch( dtype->vx_type )
    {
    case VSI_NN_TYPE_FLOAT16:
        error = 0.001;
        break;
    case VSI_NN_TYPE_FLOAT32:
        error = 0.00001;
        break;
    case VSI_NN_TYPE_INT8:
    case VSI_NN_TYPE_INT16:
        if( dtype->fl > 0 )
            {
            error = 1.0f / dtype->fl;
            }
        else
            {
            error = vsi_nn_abs( dtype->fl );
            }
        break;
    case VSI_NN_TYPE_UINT8:
        error = dtype->scale;
        break;
    }
ret = ((a - b) <= error);
return ret;
} /* _equal() */

TEST_CASE_INIT( dtype_converter )
{
#define TENSOR_HEIGHT   (400)
#define TENSOR_WIDTH    (100)
#define TENSOR_CHANNEL  (3)
#define DATA_RANGE      (6)
#define TOTAL_SIZE      (TENSOR_HEIGHT * TENSOR_WIDTH * TENSOR_CHANNEL)
vx_float32  test_data[TOTAL_SIZE] = { 0 };
vx_uint32   i, j, k;
vx_float32  rand_a, rand_b, rand_zero_point;
vx_float32  max_data, min_data;
vx_int32    ret;
vx_int32    supported_vx_type[] = {
    VSI_NN_TYPE_FLOAT32,
    VSI_NN_TYPE_FLOAT16,
    VSI_NN_TYPE_INT8,
    VSI_NN_TYPE_UINT8
    };
vx_float32     src, dst, dst_src;
vsi_nn_dtype_t src_dtype;
vsi_nn_dtype_t dst_dtype;
vsi_nn_dtype_t org_dtype;

ret = -1;
max_data = 0 - DATA_RANGE;
min_data = DATA_RANGE;

srand( time( NULL ) );   // should only be called once
for( i = 0; i < TOTAL_SIZE; i++ )
    {
    rand_a = (vx_float32)(rand() % DATA_RANGE);
    rand_b = (vx_float32)(rand() % DATA_RANGE);
    if( 0 == rand_b )
        {
        rand_b = 1;
        }
    rand_zero_point = (vx_float32)(rand() % DATA_RANGE);
    test_data[i] = (rand_a - rand_zero_point) / rand_b;
    min_data = vsi_nn_min( min_data, test_data[i]);
    max_data = vsi_nn_max( max_data, test_data[i]);
    }

memset( &org_dtype, 0, sizeof( org_dtype ) );
org_dtype.vx_type = VSI_NN_TYPE_FLOAT32;
for( i = 0; i < _cnt_of_array( supported_vx_type ); i++ )
    {
    for( j = 0; j < _cnt_of_array( supported_vx_type ); j++ )
        {
        memset( &src_dtype, 0, sizeof( src_dtype ) );
        memset( &dst_dtype, 0, sizeof( dst_dtype ) );
        src_dtype.vx_type = supported_vx_type[i];
        dst_dtype.vx_type = supported_vx_type[j];
        _set_dtype( &src_dtype, max_data, min_data );
        _set_dtype( &dst_dtype, max_data, min_data );
        for( k = 0; k < TOTAL_SIZE; k ++ )
            {
            src = test_data[k];
            dst = 0.0f;
            dst_src = 0.0f;
            /* Convert float32 to src type */
            if( VX_FAILURE == vsi_nn_DtypeConvert( (void *)&src, &org_dtype,
                        (void *)&src, &src_dtype ) )
                {
                goto final;
                }
            /* Convert src type to dst_type */
            if( VX_FAILURE == vsi_nn_DtypeConvert( (void *)&src, &src_dtype,
                        (void *)&dst, &dst_dtype ) )
                {
                goto final;
                }
            /* Convert dst type to src_type */
            if( VX_FAILURE == vsi_nn_DtypeConvert( (void *)&dst, &dst_dtype,
                        (void *)&dst_src, &src_dtype ) )
                {
                goto final;
                }
            /* Convert float32 to compare */
            /* Convert dst_src to float32 */
            if( VX_FAILURE == vsi_nn_DtypeConvert( (void *)&dst_src, &src_dtype,
                        (void *)&dst_src, &org_dtype ) )
                {
                goto final;
                }
            /* Convert src to float32 */
            if( VX_FAILURE == vsi_nn_DtypeConvert( (void *)&src, &src_dtype,
                        (void *)&src, &org_dtype ) )
                {
                goto final;
                }
            if( _equal( src, dst_src, &dst_dtype ) == vx_false_e )
                {
                VSILOGI("Convert error:  %f != %f", src, dst_src);
                VSILOGI("Src: %#x(%f), Dst: %#x, Dst_src: %#x(%f)",src,src,dst,dst_src,dst_src);
                VSILOGI("Origin data: %f", test_data[k]);
                VSILOGI("Src dtype: %#x(%#x)", src_dtype.vx_type, src_dtype.qnt_type);
                VSILOGI("Dst dtype: %#x(%#x)", dst_dtype.vx_type, dst_dtype.qnt_type);
                goto final;
                }
            }
        }
    }

ret = 0;
final:
return ret;
}

int main
    (
    int argc,
    char *argv[]
    )
{
return TEST_RUN_CASE( dtype_converter );
}

