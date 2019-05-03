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
#include <string.h>

#include "vsi_nn_tensor.h"
#include "vsi_nn_log.h"
#include "utils/vsi_nn_math.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_limits.h"
#include "quantization/vsi_nn_asymmetric_affine.h"
#include "quantization/vsi_nn_dynamic_fixed_point.h"

uint32_t vsi_nn_TypeGetBitWidth
    (
    const vsi_nn_type_e type
    )
{
    uint32_t bw;
    bw = 8 * vsi_nn_TypeGetBytes( type );
    if( vsi_nn_TypeIsSigned( type ) )
    {
        bw --;
    }
    return bw;
} /* vsi_nn_TypeGetBitWidth() */

int32_t vsi_nn_Fp32ToDFP
    (
    const float in,
    const int8_t    fl,
    const vsi_nn_type_e type
    )
{
    int32_t data;
    double max_range;
    double min_range;

    vsi_nn_TypeGetRange( type, &max_range, &min_range );
    if( fl > 0 )
    {
        data = (int32_t)vsi_nn_Rint( in * (float)( 1 << fl ) );
    }
    else
    {
        data = (int32_t)vsi_nn_Rint( in * ( 1.0f / (float)( 1 << -fl ) ) );
    }
    data = vsi_nn_min( data, (int32_t)max_range );
    data = vsi_nn_max( data, (int32_t)min_range );

    return data;
} /* vsi_nn_Fp32ToDPF() */

float vsi_nn_DFPToFp32
    (
    const int32_t val,
    const int8_t  fl,
    const vsi_nn_type_e type
    )
{
    float result;

    if( fl > 0 )
    {
        result = (float)val * ( 1.0f / ( (float) ( 1 << fl ) ) );
    }
    else
    {
        result = (float)val * ( (float) ( 1 << -fl ) );
    }

    return result;
} /* vsi_nn_DFPToFp32() */

int32_t vsi_nn_Fp32ToAffine
    (
    const float  in,
    const float  scale,
    const uint8_t    zero_point,
    const vsi_nn_type_e type
    )
{
    int32_t data;
    double max_range;
    double min_range;

    vsi_nn_TypeGetRange( type, &max_range, &min_range );

    data = (int32_t)(vsi_nn_Rint( in / scale ) + zero_point );
    data = vsi_nn_max( (int32_t)min_range, vsi_nn_min( (int32_t)max_range , data ) );

    return data;
} /* vsi_nn_Fp32ToAffine() */

float vsi_nn_AffineToFp32
    (
    const int32_t    val,
    const float  scale,
    const uint8_t    zero_point,
    const vsi_nn_type_e type
    )
{
    float data;

    data = ( (float)val - zero_point ) * scale;

    return data;
} /* vsi_nn_AffineToFp32() */

uint16_t vsi_nn_Fp32ToFp16
    (
    float in
    )
{
    uint32_t fp32 = *((uint32_t *) &in);
    uint32_t t1 = (fp32 & 0x80000000u) >> 16;  /* sign bit. */
    uint32_t t2 = (fp32 & 0x7F800000u) >> 13;  /* Exponent bits */
    uint32_t t3 = (fp32 & 0x007FE000u) >> 13;  /* Mantissa bits, no rounding */
    uint32_t fp16 = 0u;

    if( t2 >= 0x023c00u )
    {
        fp16 = t1 | 0x7BFF;     /* Don't round to infinity. */
    }
    else if( t2 <= 0x01c000u )
    {
        fp16 = t1;
    }
    else
    {
        t2 -= 0x01c000u;
        fp16 = t1 | t2 | t3;
    }

    return (uint16_t) fp16;
} /* vsi_nn_Fp32ToFp16() */

float vsi_nn_Fp16ToFp32
    (
    int16_t in
    )
{
    int32_t t1;
    int32_t t2;
    int32_t t3;
    float out;

    t1 = in & 0x7fff;         // Non-sign bits
    t2 = in & 0x8000;         // Sign bit
    t3 = in & 0x7c00;         // Exponent

    t1 <<= 13;                // Align mantissa on MSB
    t2 <<= 16;                // Shift sign bit into position

    t1 += 0x38000000;         // Adjust bias

    t1 = (t3 == 0 ? 0 : t1);  // Denormals-as-zero

    t1 |= t2;                 // Re-insert sign bit

    *((uint32_t*)&out) = t1;

    return out;
} /* vsi_nn_Fp16ToFp32() */

vsi_status vsi_nn_IntegerConvert
    (
    const void *    src,
    vsi_nn_type_e   src_type,
    void *          dest,
    vsi_nn_type_e   dest_type
    )
{
    vsi_status status = VSI_SUCCESS;

    if( vsi_nn_TypeIsInteger( src_type ) && vsi_nn_TypeIsInteger( dest_type ) )
    {
        uint8_t    all_zeros[] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
        uint8_t    all_ones[] = { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff };
        uint32_t   src_sz = vsi_nn_TypeGetBytes( src_type );
        uint32_t   dest_sz = vsi_nn_TypeGetBytes( dest_type );
        uint8_t*   buffer = all_zeros;

        if( vsi_nn_TypeIsSigned( src_type ) && (((int8_t *)src)[src_sz - 1] & 0x80) )
        {
            buffer = all_ones;
        }

        memcpy( buffer, src, src_sz );
        memcpy( dest, buffer, dest_sz );
    }
    else
    {
        VSILOGE( "src_type and dest_type must be Integer, but %#x %#x\n", src_type, dest_type );
        status = VSI_FAILURE;
    }

    return status;
} /* vsi_nn_IntegerConvert() */

vsi_status vsi_nn_DtypeConvert
    (
    uint8_t * src,
    const vsi_nn_dtype_t * src_dtype,
    uint8_t * dst,
    const vsi_nn_dtype_t * dst_dtype
    )
{
    float data;
    int32_t type_bytes;

    /* Convert src type to float32 */
    data = 0.0f;
    switch( src_dtype->vx_type )
    {
    case VSI_NN_TYPE_FLOAT32:
        {
            data = *(float *)src;
        }
        break;
    case VSI_NN_TYPE_FLOAT16:
        {
            data = vsi_nn_Fp16ToFp32( *(int16_t *)src );
        }
        break;
    case VSI_NN_TYPE_INT8:
    case VSI_NN_TYPE_UINT8:
    case VSI_NN_TYPE_INT16:
        {
            int32_t src_value = 0;

            vsi_nn_IntegerConvert(src, src_dtype->vx_type, &src_value, VSI_NN_TYPE_INT32 );
            switch( src_dtype->qnt_type )
            {
            case VSI_NN_QNT_TYPE_DFP:
                data = vsi_nn_DFPToFp32( src_value, src_dtype->fl, src_dtype->vx_type );
                break;

            case VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC:
                data = vsi_nn_AffineToFp32( src_value,
                    src_dtype->scale, src_dtype->zero_point, src_dtype->vx_type );
                break;

            case VSI_NN_QNT_TYPE_NONE:
                data = (float)src_value;
                break;

            default:
                break;
            }
        }
        break;
    default:
        VSILOGE( "Unsupported format %#x\n", src_dtype->vx_type );
        return VSI_FAILURE;
    }

    type_bytes = vsi_nn_TypeGetBytes( dst_dtype->vx_type );
    memset( dst, 0, type_bytes );
    /* Convert float32 to dst type */
    switch( dst_dtype->vx_type )
    {
    case VSI_NN_TYPE_FLOAT32:
        *(float *)dst = data;
        break;
    case VSI_NN_TYPE_FLOAT16:
        *(int16_t *)dst = vsi_nn_Fp32ToFp16( data );
        break;
    case VSI_NN_TYPE_INT8:
    case VSI_NN_TYPE_UINT8:
    case VSI_NN_TYPE_INT16:
        {
            int32_t dst_value = 0;

            switch( dst_dtype->qnt_type )
            {
            case VSI_NN_QNT_TYPE_DFP:
                dst_value = vsi_nn_Fp32ToDFP( data, dst_dtype->fl, dst_dtype->vx_type );
                break;

            case VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC:
                dst_value = vsi_nn_Fp32ToAffine( data,
                    dst_dtype->scale, dst_dtype->zero_point, dst_dtype->vx_type );
                break;

            case VSI_NN_QNT_TYPE_NONE:
                dst_value = (int32_t)data;
                break;

            default:
                break;
            }

            vsi_nn_IntegerConvert( &dst_value, VSI_NN_TYPE_INT32, dst, dst_dtype->vx_type );
        }
        break;

    default:
        VSILOGE("Unsupported dst format %d\n", dst_dtype->vx_type);
        return VSI_FAILURE;
    }

    return VSI_SUCCESS;
} /* vsi_nn_DtypeConvert */

/*
* Deprated: Use vsi_nn_DtypeToFloat32() instead
*/
vsi_status vsi_nn_DtypeToFp32
    (
    void       * src,
    float * dst,
    uint32_t    index,     /* index to src buffer */
    const vsi_nn_dtype_t * src_dtype
    )
{
    uint8_t * ptr;
    ptr = (uint8_t *)src;

    //VSILOGW("Deprecated API, use vsi_nn_DtypeToFloat32 instead.");
    ptr += vsi_nn_TypeGetBytes( src_dtype->vx_type ) * index;

    return vsi_nn_DtypeToFloat32( ptr, dst, src_dtype );
} /* vsi_nn_DtypeToFp32() */

/*
* Deprated: Use vsi_nn_Float32ToDtype() instead
*/
vsi_status vsi_nn_Fp32toDtype
    (
    float   src,
    void       * dst,
    uint32_t    index,     /* index to dst buffer */
    const vsi_nn_dtype_t * dst_dtype
    )
{
    uint8_t * ptr;
    ptr = (uint8_t *)dst;

    //VSILOGW("Deprecated API, use vsi_nn_Float32ToDtype instead.");
    ptr += vsi_nn_TypeGetBytes( dst_dtype->vx_type ) * index;

    return vsi_nn_Float32ToDtype( src, ptr, dst_dtype );
} /* vsi_nn_Fp32toDtype */

vsi_status vsi_nn_DtypeToFloat32
    (
    uint8_t   * src,
    float * dst,
    const vsi_nn_dtype_t * src_dtype
    )
{
    vsi_nn_dtype_t dst_dtype;

    memset( &dst_dtype, 0, sizeof( vsi_nn_dtype_t ) );
    dst_dtype.vx_type = VSI_NN_TYPE_FLOAT32;

    return vsi_nn_DtypeConvert( src, src_dtype, (uint8_t *)dst, &dst_dtype );
} /* vsi_nn_DtypeToFloat32() */

vsi_status vsi_nn_Float32ToDtype
    (
    float   src,
    uint8_t   * dst,
    const vsi_nn_dtype_t * dst_dtype
    )
{
    vsi_nn_dtype_t src_dtype;

    memset( &src_dtype, 0, sizeof( vsi_nn_dtype_t ) );
    src_dtype.vx_type = VSI_NN_TYPE_FLOAT32;

    return vsi_nn_DtypeConvert( (uint8_t *)&src, &src_dtype, dst, dst_dtype );
} /* vsi_nn_Float32ToDtype */

int32_t vsi_nn_DtypeConvertRawData
    (
    uint8_t * src,
    int32_t   src_bytes,
    const vsi_nn_dtype_t * src_dtype,
    uint8_t * dst,
    int32_t   dst_bytes,
    const vsi_nn_dtype_t * dst_dtype
    )
{
    uint8_t * src_iter;
    uint8_t * dst_iter;
    int32_t count;
    int32_t elements;
    int32_t src_type_bytes;
    int32_t dst_type_bytes;
    int32_t target_bytes;
    int32_t i;
    vsi_status status;
    count = 0;
    if( NULL == src || NULL == dst || NULL == src_dtype )
    {
        return count;
    }

    src_type_bytes = vsi_nn_TypeGetBytes( src_dtype->vx_type );
    dst_type_bytes = vsi_nn_TypeGetBytes( dst_dtype->vx_type );
    elements = (int32_t)( src_bytes / src_type_bytes );
    target_bytes = dst_type_bytes * elements;
    if( dst_bytes < target_bytes )
    {
        VSILOGW("Wrong dest buffer size: %d, require: %d", dst_bytes, target_bytes);
        return count;
    }
    src_iter = src;
    dst_iter = dst;
    for( i = 0; i < elements; i ++ )
    {
        status = vsi_nn_DtypeConvert( src_iter, src_dtype, dst_iter, dst_dtype );
        if( VSI_FAILURE == status )
        {
            break;
        }
        src_iter += src_type_bytes;
        dst_iter += dst_type_bytes;
    }
    count = i;
    return count;
} /* vsi_nn_DtypeConvertRawData() */

int32_t vsi_nn_DtypeConvertRawDataToFloat32
    (
    uint8_t   * src,
    int32_t     src_bytes,
    const vsi_nn_dtype_t * src_dtype,
    float * dst,
    int32_t     dst_size
    )
{
    vsi_nn_dtype_t dst_dtype;
    memset( &dst_dtype, 0, sizeof( vsi_nn_dtype_t ) );
    dst_dtype.vx_type = VX_TYPE_FLOAT32;
    return vsi_nn_DtypeConvertRawData(
        src, src_bytes, src_dtype,
        (uint8_t *)dst, dst_size * sizeof( float ), &dst_dtype );
} /*vsi_nn_DtypeConvertRawDataToFloat32()*/

int32_t vsi_nn_DtypeConvertFloat32ToRawData
    (
    float * src,
    int32_t     src_size,
    uint8_t   * dst,
    int32_t     dst_bytes,
    const vsi_nn_dtype_t * dst_dtype
    )
{
    vsi_nn_dtype_t src_dtype;
    memset( &src_dtype, 0, sizeof( vsi_nn_dtype_t ) );
    src_dtype.vx_type = VX_TYPE_FLOAT32;
    return vsi_nn_DtypeConvertRawData(
        (uint8_t *)src, src_size * sizeof( float ), &src_dtype,
        dst, dst_bytes, dst_dtype );
} /*vsi_nn_DtypeConvertFloat32ToRawData()*/

vsi_bool vsi_nn_TypeIsInteger
    (
    const vsi_nn_type_e type
    )
{
    vsi_bool ret;
    ret = FALSE;
    switch( type )
    {
    case VSI_NN_TYPE_INT8:
    case VSI_NN_TYPE_INT16:
    case VSI_NN_TYPE_INT32:
    case VSI_NN_TYPE_INT64:
    case VSI_NN_TYPE_UINT8:
    case VSI_NN_TYPE_UINT16:
    case VSI_NN_TYPE_UINT32:
    case VSI_NN_TYPE_UINT64:
        ret = TRUE;
        break;
    default:
        break;
    }
    return ret;
} /* vsi_nn_TypeIsInteger() */

vsi_bool vsi_nn_TypeIsSigned
    (
    const vsi_nn_type_e type
    )
{
    vsi_bool ret;
    ret = FALSE;
    switch( type )
    {
    case VSI_NN_TYPE_INT8:
    case VSI_NN_TYPE_INT16:
    case VSI_NN_TYPE_INT32:
    case VSI_NN_TYPE_INT64:
    case VSI_NN_TYPE_FLOAT16:
    case VSI_NN_TYPE_FLOAT32:
    case VSI_NN_TYPE_FLOAT64:
        ret = TRUE;
        break;
    default:
        break;
    }
    return ret;
} /* vsi_nn_TypeIsSigned() */

uint32_t vsi_nn_TypeGetBytes
    (
    const vsi_nn_type_e type
    )
{
    switch( type )
    {
    case VSI_NN_TYPE_INT8:
    case VSI_NN_TYPE_UINT8:
        return 1;
    case VSI_NN_TYPE_INT16:
    case VSI_NN_TYPE_UINT16:
    case VSI_NN_TYPE_FLOAT16:
        return 2;
    case VSI_NN_TYPE_INT32:
    case VSI_NN_TYPE_UINT32:
    case VSI_NN_TYPE_FLOAT32:
        return 4;
    case VSI_NN_TYPE_INT64:
    case VSI_NN_TYPE_UINT64:
    case VSI_NN_TYPE_FLOAT64:
        return 8;
    default:
        VSILOGW( "Unkonwn type %d", type );
        return 0;
    }
} /* vsi_nn_TypeGetBytes() */

/*
* Deprecated: use vsi_nn_TypeGetBytes() insteatd.
*/
uint32_t vsi_nn_GetTypeBytes
    (
    const vsi_nn_type_e type
    )
{
    return vsi_nn_TypeGetBytes( type );
} /* vsi_nn_GetTypeBytes() */

vsi_bool vsi_nn_QuantCheck
    (
    vsi_nn_tensor_t *input,
    vsi_nn_tensor_t *weight,
    vsi_nn_tensor_t *bias
    )
{
    vsi_bool ret;
    vsi_nn_type_e dtype;
    vsi_nn_qnt_type_e qnt_type;

    ret = TRUE;
    dtype = input->attr.dtype.vx_type;
    if(VSI_NN_TYPE_VDATA == weight->attr.dtype.vx_type)
    {
        return ret;
    }
    if(vsi_nn_TypeIsInteger(dtype) == FALSE)
    {
        return ret;
    }

    qnt_type = input->attr.dtype.qnt_type;
    switch(qnt_type)
    {
    case VSI_NN_QNT_TYPE_DFP:
        ret = vsi_nn_QuantDFPCheck(input, weight, bias);
        if(ret == FALSE)
        {
            VSILOGE("input_fl[%d] + weight_fl[%d] != bias_fl[%d]",
                input->attr.dtype.fl,
                weight->attr.dtype.fl,
                bias->attr.dtype.fl);
        }
        break;
    case VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC:
        ret = vsi_nn_QuantAffineCheck(input, weight, bias);
        if(ret == FALSE)
        {
            VSILOGE("abs(input_scale[%f] * weight_scale[%f] - bias_scale[%f]) > 1e-5",
                input->attr.dtype.scale,
                weight->attr.dtype.scale,
                bias->attr.dtype.scale);
        }
        break;
    default:
        ret = FALSE;
        break;
    }

    return ret;
} /* vsi_nn_QuantCheck() */

