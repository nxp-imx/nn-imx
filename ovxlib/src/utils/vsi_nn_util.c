/****************************************************************************
*
*    Copyright 2012 - 2019 Vivante Corporation, Santa Clara, California.
*    All Rights Reserved.
*
*    Permission is hereby granted, free of charge, to any person obtaining
*    a copy of this software and associated documentation files (the
*    'Software'), to deal in the Software without restriction, including
*    without limitation the rights to use, copy, modify, merge, publish,
*    distribute, sub license, and/or sell copies of the Software, and to
*    permit persons to whom the Software is furnished to do so, subject
*    to the following conditions:
*
*    The above copyright notice and this permission notice (including the
*    next paragraph) shall be included in all copies or substantial
*    portions of the Software.
*
*    THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
*    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
*    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
*    IN NO EVENT SHALL VIVANTE AND/OR ITS SUPPLIERS BE LIABLE FOR ANY
*    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
*    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
*    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>

#ifdef _WIN32
#include <io.h>
#include <direct.h>
#else
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#endif

#include "vsi_nn_prv.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_types.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_log.h"
#include "utils/vsi_nn_math.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"

#define _GET_MAX(a, b)     ( (a) > (b) ? (a) : (b) )

static uint32_t _compute_stride_rounding
    (
    uint32_t out,
    uint32_t stride,
    vsi_nn_round_type_e rounding
    )
{
    if( VSI_NN_ROUND_CEIL == rounding )
    {
        out = ( out + stride - 1 ) / stride;
    }
    else
    {
        out = out / stride;
    }
    return out;
}

static uint32_t _compute_padding
    (
    uint32_t in_size,
    uint32_t ksize,
    uint32_t stride,
    uint32_t dilation_rate,
    uint32_t out_size
    )
{
    uint32_t effective_ksize;
    int32_t padding;
    effective_ksize = (ksize - 1) * dilation_rate + 1;
    padding = (out_size - 1) * stride + effective_ksize - in_size;
    return _GET_MAX(padding, 0);
} /* _compute_padding() */

uint8_t * vsi_nn_LoadBinaryData
    (
    const char * filename,
    uint32_t  * sz
    )
{
    uint8_t  * data;
    uint32_t   fsize;
    size_t      cnt;
    FILE      * fp;

    fp = fopen( filename, "rb" );
    if( NULL == fp )
    {
        return NULL;
    }
    fseek( fp, 0L, SEEK_END );
    fsize = (uint32_t)ftell( fp );
    fseek( fp, 0L, SEEK_SET );
    data = (uint8_t *)malloc( fsize );
    cnt = 0;
    if( NULL == data )
    {
        VSILOGE( "Malloc %d memory fail.", fsize );
    }
    else
    {
        while( (uint32_t)cnt < fsize )
        {
            cnt += fread( &data[cnt], 1, fsize, fp );
            if( cnt == 0 )
            {
                break;
            }
        }
        VSILOGW( "Read %d bytes from file %s.", (uint32_t)cnt, filename );
    }
    fclose( fp );
    if( NULL != sz )
    {
        *sz = (uint32_t)cnt;
    }
    return data;
} /* vsi_nn_LoadBinaryData() */

uint32_t vsi_nn_GetStrideSize
    (
    vsi_nn_tensor_attr_t * attr,
    uint32_t            * stride
    )
{

    if( NULL == attr || NULL == stride )
    {
        return 0;
    }

    return vsi_nn_GetStrideSizeBySize(attr->size, attr->dim_num, attr->dtype.vx_type, stride);
} /* vsi_nn_GetStrideSize() */

uint32_t vsi_nn_GetStrideSizeBySize
    (
    uint32_t   * size,
    uint32_t     dim_num,
    vsi_nn_type_e type,
    uint32_t   * stride
    )
{
    uint32_t total_bytes;
    uint32_t i;

    if( NULL == size || NULL == stride )
    {
        return 0;
    }

    stride[0] = vsi_nn_GetTypeBytes( type );
    total_bytes = stride[0];
    for( i = 1; i < dim_num; i ++ )
    {
        stride[i] = size[i - 1] * stride[i - 1];
        total_bytes *= size[i];
    }
    total_bytes *= size[0];
    return total_bytes;
} /* vsi_nn_GetStrideSizeBySize() */

uint32_t vsi_nn_GetTotalBytesBySize
    (
    uint32_t   * size,
    uint32_t     dim_num,
    vsi_nn_type_e type
    )
{
    return vsi_nn_ShapeProduct( size, dim_num ) * vsi_nn_GetTypeBytes( type );
} /* vsi_nn_GetTotalBytesBySize() */

float vsi_nn_DataAsFloat32
    (
    uint8_t    * data,
    vsi_nn_type_e type
    )
{
    float val;
    int16_t fp16;

    val = 0xFFFFFFFF;
    switch( type )
    {
    case VSI_NN_TYPE_INT8:
        val = (float)((int8_t*)data)[0];
        break;
    case VSI_NN_TYPE_UINT8:
        val = (float)data[0];
        break;
    case VSI_NN_TYPE_INT16:
        val = (float)( (int16_t *)data )[0];
        break;
    case VSI_NN_TYPE_UINT16:
        val = (float)( (uint16_t *)data )[0];
        break;
    case VSI_NN_TYPE_FLOAT16:
        fp16 = ( (int16_t *)data )[0];
        val = vsi_nn_Fp16ToFp32( fp16 );
        break;
    case VSI_NN_TYPE_INT32:
        val = (float)( (int32_t *)data )[0];
        break;
    case VSI_NN_TYPE_UINT32:
        val = (float)( (uint32_t *)data )[0];
        break;
    case VSI_NN_TYPE_FLOAT32:
        val = ( (float *)data )[0];
        break;
    case VSI_NN_TYPE_INT64:
    case VSI_NN_TYPE_UINT64:
    case VSI_NN_TYPE_FLOAT64:
    default:
        VSILOGW( "Unsupport type %d", type );
        break;
    }
    return val;
} /* vsi_nn_DataAsFloat32() */

void vsi_nn_UpdateTensorDims
    (
    vsi_nn_tensor_attr_t * attr
    )
{
    uint32_t i;
    uint32_t num;
    if( NULL == attr )
    {
        return;
    }

    num = 0;
    for( i = 0; i < attr->dim_num; i ++ )
    {
        if( 0 == attr->size[i] )
        {
            break;
        }
        num ++;
    }

    if( attr->dim_num > VSI_NN_MAX_DIM_NUM )
    {
        VSILOGW( "Error dim number: %d", attr->dim_num );
        attr->dim_num = num;
    }
    else if( attr->dim_num != num )
    {
        VSILOGW( "Dim number and size mismatch: %d vs calculated = %d ", attr->dim_num, num );
        attr->dim_num = VSI_NN_DIM_AUTO;
    }
} /* vsi_nn_UpdateTensorDims() */


uint32_t vsi_nn_ComputeFilterSize
    (
    uint32_t   i_size,
    uint32_t   ksize,
    uint32_t * pad,
    uint32_t   stride,
    uint32_t   dilation,
    vsi_nn_round_type_e rounding
    )
{
    uint32_t out;
    if( 0 == stride )
    {
        VSILOGW( "Error stride value: 0." );
        return 0;
    }
    if (dilation > 1)
    {
        ksize = dilation * (ksize - 1) + 1;
    }
    out = i_size + pad[0] + pad[1] - ksize;
    out = _compute_stride_rounding( out, stride, rounding );
    out ++;
    return out;
} /* vsi_nn_ComputeFilterSize() */

uint32_t vsi_nn_compute_filter_shape
    (
    vsi_nn_pad_e padding_type,
    uint32_t image_size,
    uint32_t ksize,
    uint32_t stride,
    uint32_t dilation_rate
    )
{
    uint32_t effective_ksize;
    effective_ksize = (ksize - 1) * dilation_rate + 1;
    switch (padding_type)
    {
    case VSI_NN_PAD_SAME:
        return (image_size + stride - 1) / stride;
    case VSI_NN_PAD_VALID:
        return (image_size + stride - effective_ksize) / stride;
    default:
        return 0;
    }
} /* vsi_nn_compute_filter_shape() */

void vsi_nn_compute_padding
    (
    uint32_t   * in_shape,
    uint32_t   * ksize,
    uint32_t   * stride,
    uint32_t   * dilation,
    vsi_nn_pad_e pad_type,
    uint32_t   * out_pad
    )
{
    uint32_t out_w, out_h;
    uint32_t pad_w, pad_h;
    uint32_t dilation_w, dilation_h;
    if (NULL == in_shape || NULL == ksize
        || NULL == stride || NULL == out_pad)
    {
        return;
    }
    if (pad_type == VSI_NN_PAD_AUTO)
    {
        return;
    }
    if (NULL == dilation || (dilation[0] == 0 && dilation[1] == 0))
    {
        dilation_w = 1;
        dilation_h = 1;
    }
    else
    {
        dilation_w = dilation[0];
        dilation_h = dilation[1];
    }

    out_w = vsi_nn_compute_filter_shape(pad_type, in_shape[0], ksize[0], stride[0], dilation_w);
    out_h = vsi_nn_compute_filter_shape(pad_type, in_shape[1], ksize[1], stride[1], dilation_h);
    pad_w = _compute_padding(in_shape[0], ksize[0], stride[0], dilation_w, out_w);
    pad_h = _compute_padding(in_shape[1], ksize[1], stride[1], dilation_h, out_h);
    out_pad[0] = pad_w / 2;
    out_pad[1] = pad_w - out_pad[0];
    out_pad[2] = pad_h / 2;
    out_pad[3] = pad_h - out_pad[2];
} /* vsi_nn_compute_padding() */

void vsi_nn_ComputePadWithPadType
    (
    uint32_t   * in_shape,
    uint32_t     in_dim_num,
    uint32_t   * ksize,
    uint32_t   * stride,
    vsi_nn_pad_e pad_type,
    vsi_nn_round_type_e rounding,
    uint32_t   * out_pad
    )
{
    vsi_nn_compute_padding(in_shape, ksize, stride, NULL, pad_type, out_pad);
} /* vsi_nn_ComputePadWithPadType() */

void vsi_nn_compute_padding_conv1d
(
    uint32_t   * in_shape,
    uint32_t   * ksize,
    uint32_t   * stride,
    uint32_t   * dilation,
    vsi_nn_pad_e pad_type,
    uint32_t   * out_pad
)
{
    uint32_t out_h;
    uint32_t pad_h;
    uint32_t dilation_h;
    if (NULL == in_shape || NULL == ksize
        || NULL == stride || NULL == out_pad)
    {
        return;
    }
    if (pad_type == VSI_NN_PAD_AUTO)
    {
        return;
    }
    if (NULL == dilation || dilation[0] == 0)
    {
        dilation_h = 1;
    }
    else
    {
        dilation_h = dilation[0];
    }

    out_h = vsi_nn_compute_filter_shape(pad_type, in_shape[0], ksize[0], stride[0], dilation_h);
    pad_h = _compute_padding(in_shape[0], ksize[0], stride[0], dilation_h, out_h);
    out_pad[0] = pad_h / 2;
    out_pad[1] = pad_h - out_pad[0];
} /* vsi_nn_compute_padding_conv1d() */

void vsi_nn_ComputePadWithPadTypeForConv1D
    (
    uint32_t   * in_shape,
    uint32_t     in_dim_num,
    uint32_t   * ksize,
    uint32_t   * stride,
    vsi_nn_pad_e pad_type,
    vsi_nn_round_type_e rounding,
    uint32_t   * out_pad
    )
{
    vsi_nn_compute_padding_conv1d(in_shape, ksize, stride, NULL, pad_type, out_pad);
} /* vsi_nn_ComputePadWithPadTypeForConv1D() */

void vsi_nn_InitTensorsId
    (
    vsi_nn_tensor_id_t * ids,
    int                  num
    )
{
    num --;
    while( num >=0 )
    {
        ids[num] = VSI_NN_TENSOR_ID_NA;
        num --;
    }
} /* vsi_nn_InitTensorsId() */

void vsi_nn_GetPadForOvx
    (
    uint32_t * in_pad,
    uint32_t * out_pad
    )
{
    if( NULL == in_pad || NULL == out_pad )
    {
        return;
    }

    out_pad[0] = in_pad[0];
    out_pad[1] = in_pad[2];
    if( out_pad[0] != in_pad[1] )
    {
        out_pad[0] = (uint32_t)( 0 - (int32_t)out_pad[0] );
    }
    if( out_pad[1] != in_pad[3] )
    {
        out_pad[1] = (uint32_t)( 0 - (int32_t)out_pad[1] );
    }
} /* vsi_nn_PadForDriver() */

vsi_bool vsi_nn_CreateTensorGroup
    (
    vsi_nn_graph_t  *  graph,
    vsi_nn_tensor_t *  in_tensor,
    uint32_t          axis,
    vsi_nn_tensor_t ** out_tensors,
    uint32_t          group_number
    )
{
    vsi_bool   ret;
    uint32_t sz;
    uint32_t i;
    uint32_t start[VSI_NN_MAX_DIM_NUM];
    uint32_t end[VSI_NN_MAX_DIM_NUM];
    vx_tensor_view       view;
    vsi_nn_tensor_attr_t attr;

    if( NULL == graph || NULL == in_tensor
        || NULL == out_tensors || 0 == group_number
        || 0 == in_tensor->attr.size[axis] )
    {
        VSILOGW( "Create tensor group fail." );
        return FALSE;
    }

    if( 1 == group_number )
    {
        out_tensors[0] = in_tensor;
        return TRUE;
    }
    if( 0 != ( in_tensor->attr.size[axis] % group_number ) )
    {
        VSILOGW( "Create tensor group fail." );
        return FALSE;
    }

    ret = TRUE;
    sz = in_tensor->attr.size[axis] / group_number;

    memcpy( &attr, &in_tensor->attr, sizeof( attr ) );
    attr.size[axis] = sz;
    memset( start, 0, sizeof( uint32_t ) * VSI_NN_MAX_DIM_NUM );
    end[0] = in_tensor->attr.size[0];
    end[1] = in_tensor->attr.size[1];
    end[2] = in_tensor->attr.size[2];
    end[3] = in_tensor->attr.size[3];
    end[axis] = 0;

    for( i = 0; i <  group_number; i ++ )
    {
        start[axis] = end[axis];
        end[axis] += sz;
        view = vxCreateTensorView( graph->ctx->c,
            start, end, in_tensor->attr.dim_num  );
        if( NULL == view )
        {
            VSILOGE( "Create tensor %d view fail.", i );
            ret = FALSE;
            break;
        }
        out_tensors[i] = vsi_nn_CreateTensor( graph, &attr );
        if( NULL == out_tensors[i] )
        {
            VSILOGE( "Create tensor %d fail.", i );
            ret = FALSE;
            break;
        }
        out_tensors[i]->t = vxCreateTensorFromView( in_tensor->t, view );
        if( NULL == out_tensors[i]->t )
        {
            VSILOGE( "Create tensor %d from view fail.", i );
            ret = FALSE;
            break;
        }
    }
    return ret;
} /* vsi_nn_CreateTensorGroup() */

uint32_t vsi_nn_ShapeToString
    (
    uint32_t * shape,
    uint32_t   dim_num,
    char      * buf,
    uint32_t   buf_sz,
    vsi_bool     for_print
    )
{
#define _PRINT_FMT     (0)
#define _NOT_PRINT_FMT (1)
    uint32_t s;
    uint32_t count;
    const char * all_fmt[] = {" %d,", "%d_" };
    const char * fmt;
    if( NULL == shape || NULL == buf
        || dim_num == 0 || buf_sz == 0 )
    {
        return 0;
    }
    if( FALSE == for_print )
    {
        fmt = all_fmt[_NOT_PRINT_FMT];
    }
    else
    {
        fmt = all_fmt[_PRINT_FMT];
    }
    count = 0;
    for( s = 0; s < dim_num; s++ )
    {
        if( count >= buf_sz )
        {
            break;
        }
        count += snprintf( &buf[count], buf_sz - count,
            fmt, shape[s] );
    }
    buf[count - 1] = 0;
    return count;
} /* vsi_nn_ShapeToString() */

int32_t vsi_nn_Access
    (
    const char *path,
    int32_t mode
    )
{
    if(NULL == path)
    {
        return -1;
    }

#ifdef _WIN32
    return _access(path, mode);
#else
    return access(path, mode);
#endif
} /* vsi_nn_Access() */

int32_t vsi_nn_Mkdir
    (
    const char *path,
    int32_t mode
    )
{
    if(NULL == path)
    {
        return -1;
    }

#ifdef _WIN32
    return _mkdir(path);
#else
    return mkdir(path, mode);
#endif
} /* vsi_nn_Mkdir() */

vsi_bool vsi_nn_CheckFilePath
    (
    const char *path
    )
{
    if(NULL == path)
    {
        VSILOGE("Please set file path");
        return FALSE;
    }

    if(vsi_nn_Access(path, 0) == 0)
    {
        return TRUE;
    }

    if(vsi_nn_Mkdir(path, 0775) == 0)
    {
        VSILOGI("Create directory %s", path);
        return TRUE;
    }
    else
    {
        VSILOGE("Create directory %s fail", path);
    }

    return FALSE;
} /* vsi_nn_CheckFilePath() */

void vsi_nn_GetFP32MultiAndPostShift
    (
    vx_float32 mult,
    vx_uint16 *M0,
    vx_int8 *N
    )
{
    vx_uint32 uintMult          = *((vx_uint32*)(&mult));
    vx_uint32 tmpMultiply       = 0;
    vx_int32  exp               = 0;
    vx_uint32 postShiftBit6to5  = 0;
    vx_uint32 postShift         = 0;
    vx_int8   tmpPostShift      = 0;

    tmpMultiply         = (uintMult & 0x7FFFFF) >> 8;
    *M0                 = (vx_uint16)((1U << 15) + tmpMultiply);

    exp                 = (uintMult & 0x7F800000) >> 23; /* postShift is Scale's exp*/
    tmpPostShift        = 15 - ((vx_int8)exp - 127);
    postShift           = tmpPostShift & 0x1F;
    tmpPostShift        = tmpPostShift >> 5;
    postShiftBit6to5    = tmpPostShift & 3;

    *N = (vx_int8)(((postShiftBit6to5 << 5) | (postShift & 0x1F)));
    *N = (((vx_int32)*N << 25) >> 25);
}/* vsi_nn_GetFP32MultiAndPostShift() */

typedef struct
{
    uint8_t* raw_addr;
} aligned_header;

uint8_t * vsi_nn_MallocAlignedBuffer
    (
    uint32_t mem_size,
    uint32_t align_start_size,
    uint32_t align_block_size
    )
{
    uint32_t sz;
    long temp;
    uint8_t* raw_addr;
    uint8_t* p;
    uint8_t* align_addr;
    aligned_header* header;

    sz = sizeof(aligned_header) + mem_size + align_start_size + align_block_size;
    raw_addr = (uint8_t *)malloc( sz * sizeof( uint8_t ) );
    memset(raw_addr, 0, sizeof( uint8_t ) * sz);
    p = raw_addr + sizeof(aligned_header);

    temp = (long)(p) % align_start_size;
    if (temp == 0)
    {
        align_addr = p;
    }
    else
    {
        align_addr = p + align_start_size - temp;
    }
    header = (aligned_header*)(align_addr - sizeof(aligned_header));
    header->raw_addr = raw_addr;
    return align_addr;
}/* vsi_nn_MallocAlignedBuffer() */

void vsi_nn_FreeAlignedBuffer
    (
    uint8_t* handle
    )
{
    aligned_header* header;
    header = (aligned_header*)(handle - sizeof(aligned_header));
    free(header->raw_addr);
}

vsi_bool vsi_nn_IsBufferAligned
    (
    uint8_t * buf,
    uint32_t align_start_size
    )
{
    long temp;

    temp = (long)(buf) % align_start_size;
    if (temp == 0)
    {
        return TRUE;
    }
    return FALSE;
}/* vsi_nn_IsBufferAligned() */

void vsi_nn_FormatToString
    (
    vsi_nn_tensor_t *tensor,
    char *buf,
    uint32_t buf_sz
    )
{
    switch(tensor->attr.dtype.vx_type)
    {
    case VSI_NN_TYPE_INT8:strncpy(buf,  "i8 ",  buf_sz);break;
    case VSI_NN_TYPE_INT16:strncpy(buf, "i16", buf_sz);break;
    case VSI_NN_TYPE_INT32:strncpy(buf, "i32", buf_sz);break;
    case VSI_NN_TYPE_INT64:strncpy(buf, "i64", buf_sz);break;
    case VSI_NN_TYPE_UINT8:strncpy(buf,  "u8 ",  buf_sz);break;
    case VSI_NN_TYPE_UINT16:strncpy(buf, "u16", buf_sz);break;
    case VSI_NN_TYPE_UINT32:strncpy(buf, "u32", buf_sz);break;
    case VSI_NN_TYPE_UINT64:strncpy(buf, "u64", buf_sz);break;
    case VSI_NN_TYPE_FLOAT16:strncpy(buf, "f16", buf_sz);break;
    case VSI_NN_TYPE_FLOAT32:strncpy(buf, "f32", buf_sz);break;
    case VSI_NN_TYPE_FLOAT64:strncpy(buf, "f64", buf_sz);break;
    default:
        break;
    }
} /* vsi_nn_FormatToString() */
