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
#include <stdlib.h>
#include <limits.h>
#include "vsi_nn_log.h"
#include "vsi_nn_error.h"
#include "kernel/vsi_nn_kernel.h"

typedef enum
{
    MEMORY_ACCESSOR_READ_ONLY = 0,
    MEMORY_ACCESSOR_WRITE_ONLY = 1,
} mem_accessor_e;

vsi_status _copy_tensor
    (
    vsi_nn_kernel_tensor_t tensor,
    const vsi_nn_kernel_tensor_attr_t * attr,
    mem_accessor_e accessor,
    void * buffer,
    size_t buffer_size
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_tensor_attr_t * internal_attr = NULL;
    size_t rank;
    size_t start[VSI_NN_MAX_DIM_NUM]  = { 0 };
    size_t end[VSI_NN_MAX_DIM_NUM]    = { 0 };
    size_t stride[VSI_NN_MAX_DIM_NUM] = { 0 };
    size_t type_bytes;
    size_t total_bytes;
    uint32_t i;

    if( !tensor || !buffer || !buffer_size )
    {
        VSILOGE("Invalid parameter");
        return status;
    }
    if( !attr )
    {
        attr = vsi_nn_kernel_tensor_attr_create( tensor );
        CHECK_PTR_FAIL_GOTO( attr, "Create tensor attr fail.", final );
        attr = internal_attr;
    }

    total_bytes = vsi_nn_kernel_tensor_attr_get_bytes( attr );
    if( total_bytes != buffer_size )
    {
        VSILOGE("Read buffer size mismatch %d vs %d", total_bytes, buffer_size);
        goto final;
    }

    vsi_nn_shape_get_stride( attr->shape->data, attr->shape->size, stride );
    type_bytes = vsi_nn_kernel_dtype_get_bytes( attr->dtype );
    rank = attr->shape->size;
    for( i = 0; i < rank; i++ )
    {
        start[i]  = 0;
        end[i]    = attr->shape->data[i];
        stride[i] = stride[i] * type_bytes;
    }
    switch( accessor )
    {
        case MEMORY_ACCESSOR_READ_ONLY:
            status = vxCopyTensorPatch( (vx_tensor)tensor, rank,
                    start, end, stride, buffer, VX_READ_ONLY, 0);
            break;
        case MEMORY_ACCESSOR_WRITE_ONLY:
            status = vxCopyTensorPatch( (vx_tensor)tensor, rank,
                    start, end, stride, buffer, VX_WRITE_ONLY, 0);
            break;
        default:
            VSI_ASSERT( FALSE );
            break;
    }

final:
    if( internal_attr )
    {
        vsi_nn_kernel_tensor_attr_release( &internal_attr );
    }
    return status;
} /* _copy_tensor() */

void * vsi_nn_kernel_tensor_create_buffer
    (
    vsi_nn_kernel_tensor_t tensor,
    const vsi_nn_kernel_tensor_attr_t * attr,
    vsi_bool convert_to_float
    )
{
    vsi_status status = VSI_FAILURE;
    void * buffer = NULL;
    void * out_buffer = NULL;
    size_t bytes;
    size_t float_bytes;
    size_t tensor_size = 0;
    vsi_nn_kernel_tensor_attr_t * internal_attr = NULL;

    if( !tensor )
    {
        return NULL;
    }

    if( !attr )
    {
        internal_attr = vsi_nn_kernel_tensor_attr_create( tensor );
        CHECK_PTR_FAIL_GOTO( internal_attr, "Create tensor attr fail.", final );
        attr = internal_attr;
    }
    bytes = vsi_nn_kernel_tensor_attr_get_bytes( attr );
    out_buffer = malloc( bytes );
    CHECK_PTR_FAIL_GOTO( out_buffer, "Out of memory, create buffer fail.", final );

    status = vsi_nn_kernel_tensor_read( tensor, attr, out_buffer, bytes );
    if( status != VSI_SUCCESS )
    {
        VSILOGE("Read tensor fail with error \"%s\".", vsi_nn_DescribeStatus(status));
        free( out_buffer );
        out_buffer = NULL;
        goto final;
    }

    if( convert_to_float && F32 != attr->dtype )
    {
        buffer = out_buffer;
        tensor_size = vsi_nn_kernel_tensor_attr_get_size( attr );
        float_bytes = tensor_size * sizeof(float);
        out_buffer = malloc( float_bytes );
        if( !out_buffer )
        {
            VSILOGE("Out of memory, create float buffer fail.");
            free( buffer );
            buffer = NULL;
            goto final;
        }
        if( vsi_nn_kernel_tensor_attr_is_quantized( attr ) )
        {
            switch( attr->quant )
            {
                case VSI_NN_KERNEL_QUANT_DFP:
                    vsi_nn_dtype_convert_quantize_dfp_to_float(
                            buffer, tensor_size, attr->dtype,
                            attr->dfp.fl, out_buffer );
                    break;
                case VSI_NN_KERNEL_QUANT_ASYMM:
                    vsi_nn_dtype_convert_quantize_asymm_to_float(
                            buffer, tensor_size, attr->dtype,
                            attr->asymm.scale, attr->asymm.zero_point,
                            out_buffer );
                    break;
                case VSI_NN_KERNEL_QUANT_SYMM_PERCHANNEL:
                    vsi_nn_dtype_convert_quantize_symm_perchannel_to_float(
                            buffer, tensor_size, attr->dtype,
                            attr->shape->data, attr->shape->size,
                            attr->asymm_v.scale->data,
                            attr->asymm_v.scale->size,
                            attr->asymm_v.zero_point->data,
                            attr->asymm_v.zero_point->size,
                            attr->asymm_v.channel_dim,
                            out_buffer );
                    break;
                default:
                    VSILOGE("Donot support quantize type %d", attr->quant);
                    VSI_ASSERT( FALSE );
                    break;
            }
        }
        else
        {
            vsi_nn_dtype_convert_dtype_to_float( buffer, tensor_size,
                    attr->dtype, out_buffer );
        }
        free( buffer );
    }

final:
    if( internal_attr )
    {
        vsi_nn_kernel_tensor_attr_release( &internal_attr );
    }
    return out_buffer;
} /* vsi_nn_kernel_tensor_create_buffer() */

vsi_status vsi_nn_kernel_tensor_read
    (
    vsi_nn_kernel_tensor_t tensor,
    const vsi_nn_kernel_tensor_attr_t * attr,
    void * out_buffer,
    size_t out_buffer_size
    )
{
    return _copy_tensor( tensor, attr, MEMORY_ACCESSOR_READ_ONLY,
            out_buffer, out_buffer_size );
} /* vsi_nn_kernel_tensor_read() */

vsi_status vsi_nn_kernel_tensor_write
    (
    vsi_nn_kernel_tensor_t tensor,
    const vsi_nn_kernel_tensor_attr_t * attr,
    const void * buffer,
    size_t size
    )
{
    // NOTE: openvx api vxCopyTensorPatch access non-const buffer pointer,
    // so here we convert const to non-const ptr.
    return _copy_tensor( tensor, attr, MEMORY_ACCESSOR_WRITE_ONLY,
            (void*)buffer, size );
} /* vsi_nn_kernel_tensor_write() */

vsi_status vsi_nn_kernel_tensor_write_from_float
    (
    vsi_nn_kernel_tensor_t tensor,
    const vsi_nn_kernel_tensor_attr_t * attr,
    const float * float_buffer,
    size_t size
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_tensor_attr_t * internal_attr = NULL;
    size_t bytes;
    const void * buffer = NULL;
    void * internal_buffer = NULL;
    size_t tensor_size = 0;
    if( !attr )
    {
        attr = vsi_nn_kernel_tensor_attr_create( tensor );
        CHECK_PTR_FAIL_GOTO( attr, "Create tensor attr fail.", final );
        attr = internal_attr;
    }
    bytes = vsi_nn_kernel_tensor_attr_get_bytes( attr );
    tensor_size = vsi_nn_kernel_tensor_attr_get_size( attr );
    if( tensor_size != size )
    {
        VSILOGE("Tensor and buffer size mismatch %d vs %d", tensor_size, size);
        goto final;
    }

    if( attr->dtype != F32 )
    {
        internal_buffer = malloc( bytes );
        CHECK_PTR_FAIL_GOTO( internal_buffer, "Create buffer fail.", final );
        if( vsi_nn_kernel_tensor_attr_is_quantized( attr ) )
        {
            switch( attr->quant )
            {
                case VSI_NN_KERNEL_QUANT_DFP:
                    vsi_nn_dtype_convert_float_to_quantize_dfp(
                            float_buffer, size, attr->dtype,
                            attr->dfp.fl, internal_buffer );
                    break;
                case VSI_NN_KERNEL_QUANT_ASYMM:
                    vsi_nn_dtype_convert_float_to_quantize_asymm(
                            float_buffer, size, attr->dtype,
                            attr->asymm.scale, attr->asymm.zero_point,
                            internal_buffer );
                    break;
                case VSI_NN_KERNEL_QUANT_SYMM_PERCHANNEL:
                    vsi_nn_dtype_convert_float_to_quantize_symm_perchannel(
                            float_buffer, size, attr->dtype,
                            attr->shape->data, attr->shape->size,
                            attr->asymm_v.scale->data,
                            attr->asymm_v.scale->size,
                            attr->asymm_v.zero_point->data,
                            attr->asymm_v.zero_point->size,
                            attr->asymm_v.channel_dim,
                            internal_buffer );
                    break;
                default:
                    VSILOGE("Donot support quantize type %d", attr->quant);
                    VSI_ASSERT( FALSE );
                    break;
            }
        }
        else
        {
            vsi_nn_dtype_convert_float_to_dtype( float_buffer, size,
                    attr->dtype, internal_buffer );
        }
        buffer = (const void*)internal_buffer;
    }
    else
    {
        buffer = (const void*)float_buffer;
    }
    status = vsi_nn_kernel_tensor_write( tensor, attr, buffer, bytes );
final:
    if( internal_attr )
    {
        vsi_nn_kernel_tensor_attr_release( &internal_attr );
    }
    if( internal_buffer )
    {
        free( internal_buffer );
    }
    return status;
} /* vsi_nn_kernel_tensor_write_from_float() */

vsi_status vsi_nn_kernel_scalar_get_dtype
    (
    vsi_nn_kernel_scalar_t scalar,
    vsi_nn_kernel_dtype_e * dtype
    )
{
    vsi_status status;
    vx_enum type;
    if( !dtype )
    {
        VSILOGW("Pointer to dtype is NULL");
        return VSI_FAILURE;
    }
    status = vxQueryScalar( (vx_scalar)scalar, VX_SCALAR_TYPE, &type, sizeof(vx_enum) );
    if( status == VSI_SUCCESS )
    {
        *dtype = vsi_nn_kernel_map_dtype( (vsi_nn_type_e)type );
    }
    return status;
} /* vsi_nn_kernel_scalar_get_dtype() */

#define DEF_KERNEL_SCALAR_FUNC( READ_FUNC_NAME, WRITE_FUNC_NAME, DTYPE, DTYPE_ID ) \
    vsi_status READ_FUNC_NAME \
        ( vsi_nn_kernel_scalar_t scalar, DTYPE * ptr  ) \
    { \
        vsi_status status; \
        vsi_nn_kernel_dtype_e dtype; \
        if( !ptr ) \
        { \
            VSILOGE("Pointer to store scalar is null"); \
            return VSI_FAILURE; \
        } \
        status = vsi_nn_kernel_scalar_get_dtype( scalar, &dtype ); \
        if( dtype != DTYPE_ID ) \
        { \
            VSILOGE("Try read scalar type %d as %d", dtype, DTYPE_ID); \
            return VSI_FAILURE; \
        } \
        if( status == VSI_SUCCESS ) \
        { \
            status = vxCopyScalarWithSize( (vx_scalar)scalar, sizeof(DTYPE), \
                    ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST ); \
        } \
        return status; \
    } \
    vsi_status WRITE_FUNC_NAME \
        ( vsi_nn_kernel_scalar_t scalar, DTYPE data  ) \
    { \
        vsi_status status; \
        status = vxCopyScalarWithSize( (vx_scalar)scalar, sizeof(DTYPE), \
                &data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST ); \
        return status; \
    }

DEF_KERNEL_SCALAR_FUNC( vsi_nn_kernel_scalar_read_int8,
                        vsi_nn_kernel_scalar_write_int8,
                        int8_t,   I8 )
DEF_KERNEL_SCALAR_FUNC( vsi_nn_kernel_scalar_read_int32,
                        vsi_nn_kernel_scalar_write_int32,
                        int32_t,  I32 )
DEF_KERNEL_SCALAR_FUNC( vsi_nn_kernel_scalar_read_uint8,
                        vsi_nn_kernel_scalar_write_uint8,
                        uint8_t,  U8 )
DEF_KERNEL_SCALAR_FUNC( vsi_nn_kernel_scalar_read_uint32,
                        vsi_nn_kernel_scalar_write_uint32,
                        uint32_t, U32 )
DEF_KERNEL_SCALAR_FUNC( vsi_nn_kernel_scalar_read_int64,
                        vsi_nn_kernel_scalar_write_int64,
                        int64_t,  I64 )
DEF_KERNEL_SCALAR_FUNC( vsi_nn_kernel_scalar_read_float32,
                        vsi_nn_kernel_scalar_write_float32,
                        float,    F32 )
DEF_KERNEL_SCALAR_FUNC( vsi_nn_kernel_scalar_read_float64,
                        vsi_nn_kernel_scalar_write_float64,
                        double,   F64 )
#undef DEF_KERNEL_SCALAR_FUNC

