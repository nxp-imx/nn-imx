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
#include "utils/vsi_nn_hashmap.h"
#include "kernel/vsi_nn_kernel.h"

static int compare_data( int * x, int * y, int size )
{
    int i;
    for( i = 0; i < size; i ++ )
    {
        if( x[i] != y[i] )
        {
            return FALSE;
        }
    }
    return TRUE;
}

#define CHECK_RESULT( cond, ... ) \
    do { \
        if( !(cond) ) { VSILOGE(__VA_ARGS__); } \
        ret &= cond; \
    } while(0)

int test_hashmap()
{
    vsi_nn_hashmap_t * map;
    int data[3][5] = {
        {1,2,3,4,5},
        {2,3,4,5,6},
        {3,4,5,6,7},
    };
    int * data_ptr[3];
    int ret = TRUE;
    map = vsi_nn_hashmap_create();
    vsi_nn_hashmap_add( map, "Key2", data[1] );
    vsi_nn_hashmap_add( map, "Key1", data[0] );
    vsi_nn_hashmap_add( map, "Key3", data[2] );
    data_ptr[1] = vsi_nn_hashmap_get( map, "Key2" );
    data_ptr[2] = vsi_nn_hashmap_get( map, "Key3" );
    data_ptr[0] = vsi_nn_hashmap_get( map, "Key1" );
    CHECK_RESULT( compare_data(data_ptr[0], data[0], 5), "data 0 mismatch" );
    CHECK_RESULT( compare_data(data_ptr[1], data[1], 5), "data 1 mismatch" );
    CHECK_RESULT( compare_data(data_ptr[2], data[2], 5), "data 2 mismatch" );
    vsi_nn_hashmap_clear( map );

    vsi_nn_hashmap_add( map, "Key1", data[2] );
    vsi_nn_hashmap_add( map, "Key2", data[0] );
    vsi_nn_hashmap_add( map, "Key3", data[1] );

    data_ptr[0] = vsi_nn_hashmap_get( map, "Key2" );
    data_ptr[1] = vsi_nn_hashmap_get( map, "Key3" );
    data_ptr[2] = vsi_nn_hashmap_get( map, "Key1" );
    CHECK_RESULT( compare_data(data_ptr[0], data[0], 5), "reset data 0 mismatch" );
    CHECK_RESULT( compare_data(data_ptr[1], data[1], 5), "reset data 1 mismatch" );
    CHECK_RESULT( compare_data(data_ptr[2], data[2], 5), "reset data 2 mismatch" );
    vsi_nn_hashmap_remove( map, "Key1" );
    data_ptr[2] = vsi_nn_hashmap_get( map, "Key1" );
    CHECK_RESULT( data_ptr[2] == NULL, "Remove Key1 error" );
    vsi_nn_hashmap_release( &map );
    return ret;
}

int test_kernel_param()
{
    const int32_t i32 = 0xABCDEF;
    const int64_t i64 = 0xFEDCBAABCDEF;
    const float f32 = 123456.789;
    int data[3][5] = {
        {1,2,3,4,5},
        {2,3,4,5,6},
        {3,4,5,6,7},
    };
    const char *strs[3] = {
        "Verisilicon",
        "NeuralNetwoks",
        "test",
    };
    vsi_nn_kernel_param_t * param;
    int * data_ptr[_cnt_of_array(data)];
    const char * strs_ptr[_cnt_of_array(strs)];
    int data_sz = _cnt_of_array(data);
    int ret = TRUE;



    param = vsi_nn_kernel_param_create();
    vsi_nn_kernel_param_add_buffer( param, "Key2", data[1], sizeof(data[1]) );
    vsi_nn_kernel_param_add_buffer( param, "Key1", data[0], sizeof(data[2]) );
    vsi_nn_kernel_param_add_buffer( param, "Key3", data[2], sizeof(data[3]) );
    vsi_nn_kernel_param_add_int32( param, "i32", i32 );
    vsi_nn_kernel_param_add_int64( param, "i64", i64 );
    vsi_nn_kernel_param_add_float32( param, "f32", f32 );
    vsi_nn_kernel_param_add_str( param, "Str1", strs[0] );
    vsi_nn_kernel_param_add_str( param, "Str2", strs[1] );
    vsi_nn_kernel_param_add_str( param, "Str3", strs[2] );

    data_ptr[1] = (int*)vsi_nn_kernel_param_get_buffer( param, "Key2", NULL );
    data_ptr[2] = (int*)vsi_nn_kernel_param_get_buffer( param, "Key3", NULL );
    data_ptr[0] = (int*)vsi_nn_kernel_param_get_buffer( param, "Key1", NULL );
    strs_ptr[2] = vsi_nn_kernel_param_get_str( param, "Str3" );
    strs_ptr[1] = vsi_nn_kernel_param_get_str( param, "Str2" );
    strs_ptr[0] = vsi_nn_kernel_param_get_str( param, "Str1" );

    CHECK_RESULT( compare_data(data_ptr[0], data[0], data_sz), "data 0 mismatch" );
    CHECK_RESULT( compare_data(data_ptr[1], data[1], data_sz), "data 1 mismatch" );
    CHECK_RESULT( compare_data(data_ptr[2], data[2], data_sz), "data 2 mismatch" );
    CHECK_RESULT( 0 == strcmp( strs[0], strs_ptr[0] ), "str 0 mismatch %s - %s", strs[0], strs_ptr[0] );
    CHECK_RESULT( 0 == strcmp( strs[1], strs_ptr[1] ), "str 1 mismatch %s - %s", strs[1], strs_ptr[1] );
    CHECK_RESULT( 0 == strcmp( strs[2], strs_ptr[2] ), "str 2 mismatch %s - %s", strs[2], strs_ptr[2] );
    CHECK_RESULT( i32 == vsi_nn_kernel_param_get_int32( param, "i32" ), "Int32 mismatch" );
    CHECK_RESULT( i64 == vsi_nn_kernel_param_get_int64( param, "i64" ), "Int64 mismatch" );
    CHECK_RESULT( f32 == vsi_nn_kernel_param_get_float32( param, "f32" ), "Float mismatch" );

    vsi_nn_kernel_param_clear( param );

    vsi_nn_kernel_param_add_str( param, "Str1", strs[0] );
    vsi_nn_kernel_param_add_str( param, "Str2", strs[1] );
    vsi_nn_kernel_param_add_str( param, "Str3", strs[2] );
    vsi_nn_kernel_param_add_buffer( param, "Key2", data[1], sizeof(data[1]) );
    vsi_nn_kernel_param_add_buffer( param, "Key1", data[0], sizeof(data[2]) );
    vsi_nn_kernel_param_add_buffer( param, "Key3", data[2], sizeof(data[3]) );
    vsi_nn_kernel_param_add_int32( param, "i32", i32 );
    vsi_nn_kernel_param_add_int64( param, "i64", i64 );
    vsi_nn_kernel_param_add_float32( param, "f32", f32 );

    data_ptr[1] = (int*)vsi_nn_kernel_param_get_buffer( param, "Key2", NULL );
    data_ptr[2] = (int*)vsi_nn_kernel_param_get_buffer( param, "Key3", NULL );
    data_ptr[0] = (int*)vsi_nn_kernel_param_get_buffer( param, "Key1", NULL );
    strs_ptr[2] = vsi_nn_kernel_param_get_str( param, "Str3" );
    strs_ptr[1] = vsi_nn_kernel_param_get_str( param, "Str2" );
    strs_ptr[0] = vsi_nn_kernel_param_get_str( param, "Str1" );

    CHECK_RESULT( compare_data(data_ptr[0], data[0], 5), "reset data 0 mismatch" );
    CHECK_RESULT( compare_data(data_ptr[1], data[1], 5), "reset data 1 mismatch" );
    CHECK_RESULT( compare_data(data_ptr[2], data[2], 5), "reset data 2 mismatch" );
    CHECK_RESULT( 0 == strcmp( strs[0], strs_ptr[0] ), "reset str 0 mismatch" );
    CHECK_RESULT( 0 == strcmp( strs[1], strs_ptr[1] ), "reset str 1 mismatch" );
    CHECK_RESULT( 0 == strcmp( strs[2], strs_ptr[2] ), "reset str 2 mismatch" );
    CHECK_RESULT( i32 == vsi_nn_kernel_param_get_int32( param, "i32" ), "reset Int32 mismatch" );
    CHECK_RESULT( i64 == vsi_nn_kernel_param_get_int64( param, "i64" ), "reset Int64 mismatch" );
    CHECK_RESULT( f32 == vsi_nn_kernel_param_get_float32( param, "f32" ), "reset Float mismatch" );

    vsi_nn_kernel_param_release( &param );
    return ret;
}

int main( int argc, char* argv[] )
{
    int ret = test_hashmap();
    ret &= test_kernel_param();
    if( ret )
    {
        VSILOGI("Pass");
    }
    return ret;
}
