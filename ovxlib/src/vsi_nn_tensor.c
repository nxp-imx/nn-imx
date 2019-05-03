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
#include <stdlib.h>
#include <string.h>

#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_types.h"
#include "utils/vsi_nn_math.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"

static vsi_bool _try_set_const_tensor
    (
    vsi_nn_tensor_t *tensor
    );

static vsi_bool _auto_cal_shape
    (
    uint32_t * input_shape,
    uint32_t   input_dim,
    uint32_t * shape,
    uint32_t * dim_num
    );

static vsi_bool _init_tensor
    (
    vsi_nn_graph_t  * graph,
    vsi_nn_tensor_t * tensor
    );

static vsi_nn_tensor_rel_t *_init_tensor_rel_buffer
    (
    vsi_nn_graph_t *graph,
    uint32_t max_io
    )
{
    uint32_t i,tensor_num;
    vsi_nn_tensor_rel_t *tensor_ref;

    tensor_num = graph->tensor_num;
    tensor_ref = (vsi_nn_tensor_rel_t *)malloc(tensor_num * sizeof(vsi_nn_tensor_rel_t));
    if(NULL == tensor_ref)
    {
        return NULL;
    }
    memset(tensor_ref, 0, sizeof(vsi_nn_tensor_rel_t) * tensor_num);

    for(i = 0; i < tensor_num; i++)
    {
        tensor_ref[i].input.num = 0;
        tensor_ref[i].output.num = 0;
        tensor_ref[i].input.table  = (vsi_nn_tensor_rel_table_t *)malloc(
            max_io * sizeof(vsi_nn_tensor_rel_table_t));
        tensor_ref[i].output.table = (vsi_nn_tensor_rel_table_t *)malloc(
            max_io * sizeof(vsi_nn_tensor_rel_table_t));
        if(NULL == tensor_ref->input.table || NULL == tensor_ref->output.table)
        {
            return NULL;
        }
        memset(tensor_ref[i].input.table,  0, max_io * sizeof(vsi_nn_tensor_rel_table_t));
        memset(tensor_ref[i].output.table, 0, max_io * sizeof(vsi_nn_tensor_rel_table_t));
    }

    return tensor_ref;
} /* _init_tensor_rel_buffer() */

static vsi_bool _try_set_const_tensor
    (
    vsi_nn_tensor_t *tensor
    )
{
    vsi_status status;
    vsi_bool ret;
    vsi_nn_vxtensor_attr_t attr;

    ret = TRUE;
    status = VSI_SUCCESS;
    if( TRUE == tensor->attr.is_const )
    {
        attr = VSI_NN_TENSOR_ATTR_CONST;
        status = vsi_nn_SetTensorAttr(tensor, attr);
    }
    if( VSI_FAILURE == status )
    {
        ret = FALSE;
    }

    return ret;
} /* _set_const_tensor() */

static vsi_bool _auto_cal_shape
    (
    uint32_t * input_shape,
    uint32_t   input_dim,
    uint32_t * shape,
    uint32_t * dim_num
    )
{
    vsi_bool   ret;
    int32_t  neg_idx;
    uint32_t i;
    uint32_t total_size;

    ret = TRUE;
    neg_idx = -1;
    total_size = vsi_nn_ShapeProduct( input_shape, input_dim );
    if (-1 == *dim_num)
    {
        *dim_num = 1;
        shape[0] = total_size;
        return ret;
    }

    for( i = 0; i < *dim_num; i ++ )
    {
        if( -1 != (int32_t)shape[i] )
        {
            if (0 == shape[i])
            {
                if (i >= input_dim)
                {
                    VSILOGE( "Wrong shape '%d' ", (int32_t)shape[i] );
                    ret = FALSE;
                    break;
                }
                shape[i] = input_shape[i];
            }
            total_size /= shape[i];
        }
        else if( -1 == neg_idx )
        {
            neg_idx = i;
        }
        else
        {
            VSILOGE( "Wrong shape '%d' ", (int32_t)shape[i] );
            ret = FALSE;
            break;
        }
    }
    if( FALSE == ret  )
    {
        shape[neg_idx] = -1;
    }
    else if(neg_idx != -1)
    {
        shape[neg_idx] = total_size;
    }
    return ret;
} /* _auto_cal_shape() */

static vsi_bool _init_tensor
    (
    vsi_nn_graph_t  * graph,
    vsi_nn_tensor_t * tensor
    )
{
    vsi_bool ret;
    vx_tensor_create_params_t params;

    ret = TRUE;

    memset( &params, 0, sizeof( vx_tensor_create_params_t ) );
    params.num_of_dims = tensor->attr.dim_num;
    params.sizes = tensor->attr.size;
    params.data_format = (vsi_enum)tensor->attr.dtype.vx_type;
    params.quant_format = (vsi_enum)tensor->attr.dtype.qnt_type;
    switch( tensor->attr.dtype.qnt_type )
    {
    case VSI_NN_QNT_TYPE_DFP:
        params.quant_data.dfp.fixed_point_pos = (uint8_t)tensor->attr.dtype.fl;
        break;
    case VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC:
        params.quant_data.affine.scale = tensor->attr.dtype.scale;
        params.quant_data.affine.zeroPoint = (int32_t)tensor->attr.dtype.zero_point;
        break;
    default:
        break;
    }

    if( NULL != tensor->t )
    {
        vxReleaseTensor( &tensor->t );
    }
    if( NULL != tensor->wb )
    {
        vxReleaseWeightsBiasesParameter( &tensor->wb );
    }
    if( FALSE == tensor->attr.vtl )
    {
        tensor->t = vxCreateTensor2( graph->ctx->c,
            &params, sizeof( vx_tensor_create_params_t ) );
    }
    else
    {
        tensor->t = vxCreateVirtualTensor2( graph->g,
            &params, sizeof( vx_tensor_create_params_t ) );
    }
    if( NULL == tensor->t )
    {
        VSILOGE( "Create vx tensor fail." );
        ret = FALSE;
    }
    ret = _try_set_const_tensor( tensor );

    return ret;
} /* _init_tensor() */

vsi_bool vsi_nn_TensorReinit
    (
    vsi_nn_graph_t  * graph,
    vsi_nn_tensor_t * tensor
    )
{
    vsi_bool ret;
    ret = TRUE;

    if( NULL == graph || NULL == tensor )
    {
        return FALSE;
    }
    if( tensor->attr.dim_num != VSI_NN_DIM_AUTO )
    {
        ret = _init_tensor( graph, tensor );
    }
    return ret;
} /* vsi_nn_TensorReinit() */

vsi_nn_tensor_t * vsi_nn_CreateTensor
    (
    vsi_nn_graph_t       * graph,
    vsi_nn_tensor_attr_t * attr
    )
{
    vsi_nn_tensor_t * tensor;

    tensor = NULL;
    if( NULL == graph || NULL == graph->g || NULL == attr )
    {
        return tensor;
    }

    tensor = (vsi_nn_tensor_t *)malloc( sizeof( vsi_nn_tensor_t ) );
    vsi_nn_UpdateTensorDims( attr );

    if( NULL != tensor )
    {
        memset( tensor, 0, sizeof( vsi_nn_tensor_t ) );
        memcpy( &tensor->attr, attr, sizeof( vsi_nn_tensor_attr_t ) );
        if( attr->dim_num != VSI_NN_DIM_AUTO )
        {
            _init_tensor( graph, tensor );
            if( NULL == tensor->t )
            {
                VSILOGE( "Create vx tensor fail." );
                free( tensor );
                tensor = NULL;
            }
        }
    }
    return tensor;
} /* vsi_nn_CreateTensor() */

void vsi_nn_ReleaseTensor
    (
    vsi_nn_tensor_t ** tensor
    )
{
    vsi_nn_tensor_t * ptr;
    ptr = *tensor;
    if( NULL != tensor && NULL != *tensor )
    {
        if( NULL != ptr->t )
        {
            vxReleaseTensor( &ptr->t );
        }
        if(ptr->wb)
            vxReleaseWeightsBiasesParameter( &ptr->wb );
        free( ptr );
        *tensor = NULL;
    }
} /* vsi_nn_ReleaseTensor() */

vsi_status vsi_nn_SetTensorAttr
    (
    vsi_nn_tensor_t * tensor,
    const vsi_nn_vxtensor_attr_t attrs
    )
{
    vsi_status status;

    status = VSI_SUCCESS;
    if( NULL == tensor )
    {
        return VSI_FAILURE;
    }

    if( VSI_SUCCESS == status && vsi_nn_hasattr( attrs, VSI_NN_TENSOR_ATTR_CONST ) )
    {
        vx_enum data_lifetime;
        if(tensor->attr.is_const == TRUE)
        {
            data_lifetime = VX_TENSOR_LIFE_TIME_STATIC;
        }
        else
        {
            data_lifetime = VX_TENSOR_LIFE_TIME_DYNAMIC;
        }
        status = vxSetTensorAttribute(tensor->t,
                                      VX_TENSOR_LIFETIME,
                                      &data_lifetime,
                                      sizeof(vx_enum));
    }
    if( VSI_SUCCESS == status && vsi_nn_hasattr( attrs, VSI_NN_TENSOR_ATTR_HIGH_PRECISION ) )
    {
        vx_enum precision = VX_TENSOR_PRECISION_HIGH;
        status = vxSetTensorAttribute(tensor->t,
                                      VX_TENSOR_PRECISION,
                                      &precision,
                                      sizeof(vx_enum));
    }

    return status;
}

vsi_status vsi_nn_QueryTensorAttr
    (
    vsi_nn_tensor_t * tensor,
    const vsi_nn_vxtensor_attr_t attrs
    )
{
    vsi_status status;

    status = VSI_SUCCESS;
    if( NULL == tensor )
    {
        return VSI_FAILURE;
    }

    if( VSI_SUCCESS == status && vsi_nn_hasattr( attrs, VSI_NN_TENSOR_ATTR_DIM_NUM ) )
    {
        status = vxQueryTensor( tensor->t, VX_TENSOR_NUM_OF_DIMS,
            &tensor->attr.dim_num, sizeof( tensor->attr.dim_num ) );
    }

    if( VSI_SUCCESS == status && vsi_nn_hasattr( attrs, VSI_NN_TENSOR_ATTR_DTYPE ) )
    {
        status = vxQueryTensor( tensor->t, VX_TENSOR_DATA_TYPE,
            &tensor->attr.dtype.vx_type, sizeof( tensor->attr.dtype.vx_type ) );
    }

    if( VSI_SUCCESS == status && vsi_nn_hasattr( attrs, VSI_NN_TENSOR_ATTR_SIZE ) )
    {
        status = vxQueryTensor( tensor->t, VX_TENSOR_DIMS,
            &tensor->attr.size, sizeof( tensor->attr.size ) );
    }

    if( VSI_SUCCESS == status && vsi_nn_hasattr( attrs, VSI_NN_TENSOR_ATTR_FIXED_POINT_POS ) )
    {
        status = vxQueryTensor( tensor->t, VX_TENSOR_FIXED_POINT_POS,
            &tensor->attr.dtype.fl, sizeof( tensor->attr.dtype.fl ) );
    }

    return status;
} /* vsi_nn_QueryTensorAttr() */

uint32_t vsi_nn_CopyTensorToBuffer
    (
    vsi_nn_graph_t  * graph,
    vsi_nn_tensor_t * tensor,
    uint8_t        * buffer
    )
{
    uint32_t     sz;
    uint32_t     stride_size[VSI_NN_MAX_DIM_NUM];
    vsi_status     status;
    vx_tensor_addressing addr;

    if( NULL == tensor || NULL == graph || NULL == buffer )
    {
        return 0;
    }

    sz = 0;
    status = VSI_FAILURE;
    addr = NULL;

    vsi_nn_GetStrideSize( &tensor->attr, stride_size );
    addr = vxCreateTensorAddressing( graph->ctx->c, tensor->attr.size,
        stride_size, tensor->attr.dim_num );
    if( NULL != addr )
    {
        status = vxCopyTensorPatch( tensor->t, NULL, addr, buffer, VX_READ_ONLY, 0 );
        vxReleaseTensorAddressing( &addr );
        if( VSI_SUCCESS != status )
        {
            VSILOGE( "Copy tensor patch fail." );
            sz = 0;
        }
    }
    else
    {
        sz = 0;
    }
    return sz;

} /* vsi_nn_CopyTensorToData() */

float * vsi_nn_ConvertTensorToFloat32Data
    (
    vsi_nn_graph_t *graph,
    vsi_nn_tensor_t *tensor
    )
{
    vsi_status status;
    uint8_t *tensor_data;
    vsi_nn_size_t sz;
    uint32_t i,stride;
    float *data;

    if(NULL == graph || NULL == tensor)
    {
        return NULL;
    }

    sz = vsi_nn_GetElementNum(tensor);
    stride = vsi_nn_TypeGetBytes(tensor->attr.dtype.vx_type);

    data = NULL;
    data = (float *)malloc(sz * sizeof(float));

    tensor_data = vsi_nn_ConvertTensorToData(graph, tensor);
    for(i=0; i<sz; i++)
    {
        status = vsi_nn_DtypeToFloat32(&tensor_data[stride * i], &data[i], &tensor->attr.dtype);
        if(status != VSI_SUCCESS)
        {
            free(data);
            data = NULL;
            break;
        }
    }

    if(tensor_data)free(tensor_data);
    return data;
} /* vsi_nn_ConvertTensorToFloat32Data() */

uint8_t * vsi_nn_ConvertTensorToData
    (
    vsi_nn_graph_t  * graph,
    vsi_nn_tensor_t * tensor
    )
{
    uint8_t    * data;
    uint32_t     buf_sz;
    uint32_t     stride_size[VSI_NN_MAX_DIM_NUM];
    vsi_status     status;
    vx_tensor_addressing addr;
    if( NULL == tensor || NULL == graph )
    {
        return NULL;
    }

    status = VSI_FAILURE;
    data = NULL;
    addr = NULL;

    buf_sz = vsi_nn_GetStrideSize( &tensor->attr, stride_size );
    // TODO: Fix this to use copy tensor to buffer
    if( buf_sz > 0 )
    {
        data = (uint8_t *)malloc( buf_sz );
    }

    if( NULL != data )
    {
        VSILOGI( "Create %d data.", buf_sz );
        addr = vxCreateTensorAddressing( graph->ctx->c, tensor->attr.size,
            stride_size, tensor->attr.dim_num );
    }
    if( NULL != addr )
    {
        status = vxCopyTensorPatch( tensor->t, NULL, addr, data, VX_READ_ONLY, 0 );
        vxReleaseTensorAddressing( &addr );
        if( VSI_SUCCESS != status )
        {
            VSILOGE( "Copy tensor patch fail %d.", status );
            free( data );
            data = NULL;
        }
    }
    else if( NULL != data )
    {
        VSILOGE( "Copy tensor addr fail." );
        free( data );
        data = NULL;
    }
    return data;

} /* vsi_nn_ConvertTensorToData() */

/*
* Deprecated: Use vsi_nn_ConvertRawTensorToData2() instead
* WARNING: This is a bad API,
*          please add a new API for WRITE_ONLY accessor.
*/
uint8_t * vsi_nn_ConvertRawTensorToData
    (
    vx_context context,
    vx_tensor tensor,
    uint32_t * dim,
    vx_enum  * data_format,
    uint32_t * size,
    uint32_t * stride_size,
    vx_tensor_addressing * addr,
    vx_enum accessor
    )
{
    uint8_t    * data;
    uint32_t     buf_sz;
    vsi_status     status;

    if( NULL == tensor || NULL == context )
    {
        return NULL;
    }

    status = VSI_FAILURE;
    data = NULL;

    status = vxQueryTensor(tensor, VX_TENSOR_NUM_OF_DIMS, dim, sizeof(uint32_t));
    status = vxQueryTensor(tensor, VX_TENSOR_DIMS, size, sizeof(uint32_t) * (*dim));
    status = vxQueryTensor(tensor, VX_TENSOR_DATA_TYPE, data_format, sizeof(vsi_enum));

    buf_sz = vsi_nn_GetStrideSizeBySize(size, *dim, *data_format, stride_size);
    // TODO: Fix this to use copy tensor to buffer
    if( buf_sz > 0 )
    {
        data = (uint8_t *)malloc( buf_sz );
    }

    if( NULL != data )
    {
        VSILOGI( "Create %d data.", buf_sz );
        *addr = vxCreateTensorAddressing(context, size,
            stride_size, *dim );
    }
    if( NULL != *addr )
    {
        if (accessor != VX_READ_ONLY)
        {
            return data;
        }
        status = vxCopyTensorPatch( tensor, NULL, *addr, data, VX_READ_ONLY, 0 );
        vxReleaseTensorAddressing( addr );
        if( VSI_SUCCESS != status )
        {
            VSILOGE( "Copy tensor patch fail %d.", status );
            free( data );
            data = NULL;
        }
    }
    else if( NULL != data )
    {
        VSILOGE( "Copy tensor addr fail." );
        free( data );
        data = NULL;
    }
    return data;

} /* vsi_nn_ConvertRawTensorToData() */

/*
* WARNING: This is a bad API,
*          please add the new APIs for WRITE_ONLY and READ_ONLY.
*          Then deprecate this function.
*/
uint8_t * vsi_nn_ConvertRawTensorToData2
    (
    vx_context context,
    vx_tensor tensor,
    vsi_nn_tensor_attr_t * attr,
    uint32_t * stride_size,
    vx_tensor_addressing * addr,
    vx_enum accessor
    )
{
    uint8_t * data;
    uint32_t buf_sz;
    vsi_status status;

    if( NULL == tensor || NULL == context )
    {
        return NULL;
    }

    status = VSI_FAILURE;
    data = NULL;

    status = vxQueryTensor(tensor, VX_TENSOR_NUM_OF_DIMS,
        &(attr->dim_num), sizeof(uint32_t));
    status = vxQueryTensor(tensor, VX_TENSOR_DIMS,
        attr->size, sizeof(uint32_t) * (attr->dim_num));
    status = vxQueryTensor(tensor, VX_TENSOR_DATA_TYPE,
        &(attr->dtype.vx_type), sizeof(vsi_enum));
    status = vxQueryTensor(tensor, VX_TENSOR_QUANT_FORMAT,
        &(attr->dtype.qnt_type), sizeof(uint32_t));
    switch( attr->dtype.qnt_type )
    {
    case VSI_NN_QNT_TYPE_DFP:
        status = vxQueryTensor(tensor, VX_TENSOR_FIXED_POINT_POS,
            &(attr->dtype.fl), sizeof(int8_t));
        break;
    case VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC:
        status = vxQueryTensor(tensor, VX_TENSOR_ZERO_POINT,
            &(attr->dtype.zero_point), sizeof(int32_t));
        status = vxQueryTensor(tensor, VX_TENSOR_SCALE,
            &(attr->dtype.scale), sizeof(float));
        break;
    default:
        break;
    }


    buf_sz = vsi_nn_GetStrideSize( attr, stride_size );
    // TODO: Fix this to use copy tensor to buffer
    if( buf_sz > 0 )
    {
        data = (uint8_t *)malloc( buf_sz );
    }

    if( NULL != data )
    {
        VSILOGI( "Create %d data.", buf_sz );
        *addr = vxCreateTensorAddressing(context, attr->size,
            stride_size, attr->dim_num );
    }
    if( NULL != *addr )
    {
        if (accessor != VX_READ_ONLY)
        {
            return data;
        }
        status = vxCopyTensorPatch( tensor, NULL, *addr, data, VX_READ_ONLY, 0 );
        vxReleaseTensorAddressing( addr );
        if( VSI_SUCCESS != status )
        {
            VSILOGE( "Copy tensor patch fail %d.", status );
            free( data );
            data = NULL;
        }
    }
    else if( NULL != data )
    {
        VSILOGE( "Copy tensor addr fail." );
        free( data );
        data = NULL;
    }
    return data;

} /* vsi_nn_ConvertRawTensorToData2() */

void vsi_nn_SaveTensorToTextByFp32
    (
    vsi_nn_graph_t   * graph,
    vsi_nn_tensor_t  * tensor,
    const char       * filename,
    char             * seperator
    )
{
#define _TENSOR_TMPBUF_SZ  (512)
    const float   c_flush_th = 0.7f;
    uint8_t    * data;
    uint8_t    * ptr;
    uint8_t      type_bytes;
    uint8_t      buf[_TENSOR_TMPBUF_SZ];
    FILE        * fp;
    float    write_data;
    uint32_t     sz;
    uint32_t     i;
    uint32_t     count;

    if( NULL == graph || NULL == tensor || NULL == filename )
    {
        return;
    }
    if( NULL == seperator )
    {
        seperator = "\n";
    }

    data = vsi_nn_ConvertTensorToData( graph, tensor );
    if( NULL == data )
    {
        VSILOGE( "Convert data fail." );
        return;
    }

    fp = fopen( filename, "w" );
    sz = vsi_nn_GetElementNum( tensor );

    ptr = data;
    type_bytes = vsi_nn_TypeGetBytes( tensor->attr.dtype.vx_type );
    count = 0;
    for( i = 0; i < sz; i ++ )
    {
        vsi_nn_DtypeToFloat32( ptr, &write_data, &tensor->attr.dtype );
        ptr += type_bytes;

        count += snprintf( (char *)&buf[count], _TENSOR_TMPBUF_SZ - count,
            "%f%s", write_data, seperator );
        if( ((float)count / _TENSOR_TMPBUF_SZ) > c_flush_th )
        {
            fwrite( buf, count, 1, fp );
            count = 0;
        }
    }
    fwrite( buf, count, 1, fp );
    fclose( fp );
    free( data );
} /* vsi_nn_SaveTensorToTextByFp32() */

void vsi_nn_SaveTensorToText
    (
    vsi_nn_graph_t   * graph,
    vsi_nn_tensor_t  * tensor,
    const char       * filename,
    char             * seperator
    )
{
    uint8_t * data;
    uint32_t  sz;

    if( NULL == graph || NULL == tensor || NULL == filename )
    {
        return;
    }

    data = vsi_nn_ConvertTensorToData( graph, tensor );
    if( NULL == data )
    {
        VSILOGE( "Convert data fail." );
        return;
    }

    sz = vsi_nn_GetElementNum( tensor );
    vsi_nn_SaveDataToText( filename, data, sz,
        tensor->attr.dtype.vx_type, seperator );
    free( data );
} /* vsi_nn_SaveTensorToText() */

void vsi_nn_SaveDataToText
    (
    const char  * filename,
    uint8_t    * data,
    uint32_t     data_size,
    vsi_nn_type_e type,
    char        * seperator
    )
{
#define _TENSOR_TMPBUF_SZ  (512)
    const float   c_flush_th = 0.7f;
    uint8_t      buf[_TENSOR_TMPBUF_SZ];
    FILE        * fp;
    float    write_data;
    uint32_t     type_bytes;
    uint32_t     i;
    uint32_t     count;

    if(  NULL == filename )
    {
        return;
    }
    if( NULL == seperator )
    {
        seperator = "\n";
    }

    if( NULL == data )
    {
        return;
    }

    fp = fopen( filename, "w" );
    type_bytes = vsi_nn_GetTypeBytes( type );

    count = 0;
    for( i = 0; i < data_size; i ++ )
    {
        write_data = vsi_nn_DataAsFloat32( &data[type_bytes * i],
            type );
        if( type == VSI_NN_TYPE_UINT8 || type == VSI_NN_TYPE_INT8 )
        {
            count += snprintf( (char *)&buf[count], _TENSOR_TMPBUF_SZ - count,
                "%d%s", (int32_t)write_data, seperator );
        }
        else
        {
            count += snprintf( (char *)&buf[count], _TENSOR_TMPBUF_SZ - count,
                "%f%s", write_data, seperator );
        }
        if( ((float) count / _TENSOR_TMPBUF_SZ ) > c_flush_th )
        {
            fwrite( buf, count, 1, fp );
            count = 0;
        }
    }
    fwrite( buf, count, 1, fp );
    fclose( fp );
} /* vsi_nn_SaveDataToText() */

void vsi_nn_SaveTensorToBinary
    (
    vsi_nn_graph_t   * graph,
    vsi_nn_tensor_t  * tensor,
    const char       * filename
    )
{
    uint8_t        * data;
    FILE            * fp;
    uint32_t         sz;
    uint32_t         i;

    if( NULL == graph || NULL == tensor || NULL == filename )
    {
        return;
    }

    data = vsi_nn_ConvertTensorToData( graph, tensor );
    if( NULL == data )
    {
        VSILOGE( "Convert data fail." );
        return;
    }

    fp = fopen( filename, "wb" );
    sz = vsi_nn_GetTypeBytes( tensor->attr.dtype.vx_type );
    for( i = 0; i < tensor->attr.dim_num; i ++ )
    {
        sz *= tensor->attr.size[i];
    }
    fwrite( data, sz, 1, fp );
    fclose( fp );
    free( data );
} /* vsi_nn_SaveTensorToBinary() */

vsi_nn_tensor_t * vsi_nn_CreateTensorFromData
    (
    vsi_nn_graph_t       * graph,
    uint8_t             * data,
    vsi_nn_tensor_attr_t * attr
    )
{
    vsi_status         status;
    vsi_nn_tensor_t * tensor;

    status = VSI_FAILURE;
    tensor = NULL;

    if( NULL == graph || NULL == data || NULL == attr )
    {
        return NULL;
    }

    tensor = vsi_nn_CreateTensor( graph, attr );

    status = vsi_nn_CopyDataToTensor( graph, tensor, data );

    if( VSI_SUCCESS != status )
    {
        VSILOGE("Create tensor from data fail.");
        if( NULL != tensor )
        {
            vsi_nn_ReleaseTensor( &tensor );
        }
    }
    return tensor;
} /* vsi_nn_CreateTensorFromData() */

vsi_status vsi_nn_CopyDataToTensor
    (
    vsi_nn_graph_t       * graph,
    vsi_nn_tensor_t      * tensor,
    uint8_t             * data
    )
{
    vsi_status         status;
    uint32_t         stride_size[VSI_NN_MAX_DIM_NUM];
    vx_tensor_addressing addr;

    status = VSI_FAILURE;
    addr = NULL;

    if( NULL == graph || NULL == data || NULL == tensor )
    {
        return status;
    }

    vsi_nn_GetStrideSize( &tensor->attr, stride_size );
    addr = vxCreateTensorAddressing( graph->ctx->c, tensor->attr.size,
        stride_size, tensor->attr.dim_num );
    if( NULL != addr )
    {
        status = vxCopyTensorPatch( tensor->t, NULL, addr,
            (void *)data, VX_WRITE_ONLY, 0);
        vxReleaseTensorAddressing( &addr );
    }
    return status;
} /* vsi_nn_CopyDataToTensor() */

vsi_status vsi_nn_CopyRawDataToTensor
    (
    vsi_nn_graph_t*         graph,
    uint8_t*                src_data,
    const vsi_nn_dtype_t*   src_dtype,
    vsi_nn_tensor_t*        tensor
    )
{
    vsi_status status           = VSI_FAILURE;
    vsi_nn_size_t src_data_sz   = 0;
    uint8_t* buffer             = NULL;
    uint32_t target_tensor_size = 0; /* in bytes */

    src_data_sz = vsi_nn_GetElementNum(tensor) * vsi_nn_GetTypeBytes(src_dtype->vx_type);
    target_tensor_size = vsi_nn_GetTensorSize( tensor->attr.size, tensor->attr.dim_num, tensor->attr.dtype.vx_type );
    buffer = (uint8_t *)malloc(target_tensor_size);

    vsi_nn_DtypeConvertRawData(src_data, src_data_sz, src_dtype, buffer, target_tensor_size, &tensor->attr.dtype);
    status = vsi_nn_CopyDataToTensor(graph, tensor, buffer);

    if( NULL != buffer )
    {
        free( buffer );
        buffer = NULL;
    }
    return status;
} /* vsi_nn_CopyRawDataToTensor */

vsi_bool vsi_nn_CalcReshapeTensor
    (
    vsi_nn_tensor_t * input,
    vsi_nn_tensor_t * output,
    uint32_t       * shape,
    uint32_t         dim_num
    )
{
    vsi_bool ret;
    uint32_t i;
    uint32_t total_size;
    uint32_t dst_size;

    if( NULL == input || NULL == output
        || NULL == shape || 0 == dim_num )
    {
        VSILOGE( "Wrong reshape parameters." );
        return FALSE;
    }

    ret = _auto_cal_shape( input->attr.size, input->attr.dim_num, shape, &dim_num );
    if( FALSE == ret )
    {
        return ret;
    }

    /* Check total size */
    total_size = vsi_nn_ShapeProduct( input->attr.size, input->attr.dim_num );
    dst_size = vsi_nn_ShapeProduct( shape, dim_num );
    if( total_size != dst_size )
    {
        VSILOGE( "Cannot calculate the reshape tensor %u to %u.",
            total_size, dst_size );
        return FALSE;
    }

    if( TRUE == ret )
    {
        if( VSI_NN_DIM_AUTO == output->attr.dim_num )
        {
            for( i = 0; i < dim_num; i ++ )
            {
                output->attr.size[i] = shape[i];
            }
            output->attr.dim_num = dim_num;
        }
    }

    return ret;
} /* vsi_nn_CalcReshapeTensor() */

vsi_bool vsi_nn_ReshapeTensor
    (
    vsi_nn_graph_t  * graph,
    vsi_nn_tensor_t * input,
    vsi_nn_tensor_t * output,
    uint32_t       * shape,
    uint32_t         dim_num
    )
{
    vsi_bool ret;

    ret = TRUE;
    ret = vsi_nn_CalcReshapeTensor(input, output, shape, dim_num);
    if( FALSE == ret )
    {
        return FALSE;
    }

    if( NULL == input->t )
    {
        ret = vsi_nn_TensorReinit( graph, input );
    }

    if( NULL != output->t )
    {
        VSILOGW( "Free tensor." );
    }

    /* Create reshape tensor */
    output->t = vxReshapeTensor( input->t, (int32_t *)shape, dim_num );
    if( NULL == output->t )
    {
        ret = FALSE;
    }

    if( FALSE == ret )
    {
        VSILOGW( "Reshape tensor error." );
    }

    return ret;
} /* vsi_nn_ReshapeTensor() */

void vsi_nn_TransposeTensor
    (
    vsi_nn_graph_t  * graph,
    vsi_nn_tensor_t * tensor,
    uint32_t       * perm,
    uint32_t         dim_num,
    uint32_t       * as_shape
    )
{
    uint8_t * buf;
    uint8_t * dst;
    uint32_t  buf_sz;
    uint32_t  tensor_sz;
    uint32_t * shape_ptr;
    vsi_status  status;

    if( NULL == tensor || NULL == perm || 0 == dim_num )
    {
        VSILOGE( "Wrong perm dims." );
        return;
    }
    tensor_sz = vsi_nn_GetTensorSize( tensor->attr.size, tensor->attr.dim_num,
        tensor->attr.dtype.vx_type );
    shape_ptr = tensor->attr.size;

    if( NULL != as_shape )
    {
        buf_sz = vsi_nn_GetTensorSize( as_shape, dim_num, tensor->attr.dtype.vx_type );
        if( buf_sz != tensor_sz )
        {
            VSILOGW( "The shape does not match origin tensor's shape." );
            return;
        }
        shape_ptr = as_shape;
    }
    buf = vsi_nn_ConvertTensorToData( graph, tensor );

    if( NULL == buf )
    {
        VSILOGE( "Create tensor buf fail." );
        return;
    }
    dst = (uint8_t *)malloc( tensor_sz * sizeof( uint8_t ) );
    // TODO: Check memory allocate.

    vsi_nn_Transpose( dst, buf, shape_ptr, dim_num, perm, tensor->attr.dtype.vx_type );
    status = vsi_nn_CopyDataToTensor( graph, tensor, dst );
    if( VSI_SUCCESS != status )
    {
        VSILOGE( "Copy transpose data fail with code %#x.", status );
    }

    free( buf );
    free( dst );
} /* vsi_nn_TransposeTensor() */

vsi_nn_size_t vsi_nn_GetElementNum
    (
    vsi_nn_tensor_t * tensor
    )
{
    vsi_nn_size_t num;
    vsi_nn_size_t sz;
    vsi_nn_size_t dsize;

    if( NULL == tensor )
    {
        return 0;
    }

    sz = vsi_nn_GetTensorSize( tensor->attr.size,
        tensor->attr.dim_num, tensor->attr.dtype.vx_type );
    dsize = vsi_nn_GetTypeBytes( tensor->attr.dtype.vx_type );
    num = (vsi_nn_size_t)(sz / dsize);
    return num;
} /* vsi_nn_GetElementNum() */

uint32_t vsi_nn_GetTensorSize
    (
    uint32_t   * shape,
    uint32_t     dim_num,
    vsi_nn_type_e type
    )
{
    uint32_t sz;
    uint32_t i;
    sz = 0;
    if( NULL == shape || 0 == dim_num )
    {
        return sz;
    }
    sz = 1;
    for( i = 0; i < dim_num; i ++ )
    {
        sz *= shape[i];
    }
    sz *= vsi_nn_GetTypeBytes( type );
    return sz;
} /* vsi_nn_GetTensorSize() */

vsi_nn_tensor_t * vsi_nn_VariableToTensor
    (
    vsi_nn_node_t * self,
    uint8_t * data,
    vsi_nn_type_e type
    )
{
    vsi_nn_tensor_t * tensor;
    vsi_nn_tensor_attr_t attr;

    if(NULL == data || NULL == self)
    {
        return NULL;
    }

    memset( &attr, 0, sizeof( attr ) );
    attr.size[0] = 1;
    attr.dim_num = 1;
    attr.is_const = TRUE;
    attr.dtype.vx_type = type;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    tensor = vsi_nn_CreateTensorFromData(
        self->graph,
        data,
        &attr);
    if(NULL == tensor)
    {
        return NULL;
    }

    return tensor;
} /* vsi_nn_VariableToTensor() */

void vsi_nn_PrintTensor
    (
    vsi_nn_tensor_t * tensor,
    vsi_nn_tensor_id_t id
    )
{
#define _SHAPE_BUF_SZ   (64)
#define _EXT_ATTR_BUF_SZ   (64)
    int count;
    char shape[_SHAPE_BUF_SZ] = { 0 };
    char ext_attr[_EXT_ATTR_BUF_SZ] = { 0 };

    vsi_nn_ShapeToString( tensor->attr.size, tensor->attr.dim_num,
        shape, _SHAPE_BUF_SZ, TRUE );
    /* Process quantize parameters */
    switch( tensor->attr.dtype.qnt_type )
    {
    case VSI_NN_QNT_TYPE_DFP:
        count = snprintf( &ext_attr[0], _EXT_ATTR_BUF_SZ,
            " %3d,", tensor->attr.dtype.fl );
        ext_attr[count - 1] = 0;
        break;
    case VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC:
        count = snprintf( &ext_attr[0], _EXT_ATTR_BUF_SZ,
            "%3d, %.6f",
            tensor->attr.dtype.zero_point, tensor->attr.dtype.scale );
        ext_attr[count - 1] = 0;
        break;
    default:
        break;
    }


    VSILOGI( "%u(%#x, %s, const: %d, virtual: %d, fmt: %#x): [%s ]", id,
        tensor->attr.dtype.vx_type, ext_attr,
        tensor->attr.is_const, tensor->attr.vtl, tensor->attr.dtype.fmt, shape );
} /* vsi_nn_PrintTensor() */

vx_tensor vsi_nn_CreateViewTensor
    (
    vsi_nn_graph_t *graph,
    uint32_t *start,
    uint32_t *end,
    vsi_nn_tensor_t *tensor
    )
{
    vx_tensor_view view;
    vx_tensor view_tensor;
    if(NULL == graph
        || NULL == start
        || NULL == end
        || NULL == tensor)
    {
        return NULL;
    }

    view = vxCreateTensorView( graph->ctx->c,
        start, end, tensor->attr.dim_num );
    if( NULL == view )
    {
        VSILOGE("Call vxCreateTensorView fail.");
        return NULL;
    }
    view_tensor = vxCreateTensorFromView( tensor->t, view );
    vxReleaseTensorView( &view );
    if( NULL == view_tensor )
    {
        VSILOGE("Call vxCreateTensorFromView fail.");
        return NULL;
    }

    return view_tensor;
} /* vsi_nn_CreateViewTensor() */

void vsi_nn_Free
    (
    void * data
    )
{
    if(NULL != data)
        free(data);
} /* vsi_nn_Free() */

void vsi_nn_ReleaseTensorRelevance
    (
    vsi_nn_graph_t *graph,
    vsi_nn_tensor_rel_t *tensor_ref
    )
{
    uint32_t i;
    if(NULL == tensor_ref || NULL == graph)
    {
        return ;
    }

    for(i = 0; i < graph->tensor_num; i++)
    {
        if(tensor_ref[i].input.table)
        {
            free(tensor_ref[i].input.table);
            tensor_ref[i].input.table = NULL;
        }
        if(tensor_ref[i].output.table)
        {
            free(tensor_ref[i].output.table);
            tensor_ref[i].output.table = NULL;
        }
    }

    if(tensor_ref)
    {
        free(tensor_ref);
        tensor_ref = NULL;
    }
} /* vsi_nn_ReleaseTensorRelevance() */

vsi_nn_tensor_rel_t *vsi_nn_CreateTensorRelevance
    (
    vsi_nn_graph_t *graph
    )
{
    uint32_t i,j,k;
    uint32_t in_num,out_num;
    uint32_t max_io,tensor_num;
    vsi_nn_tensor_rel_t *tensor_ref;
    vsi_nn_node_t *node;

#define _MAX_TENSOR_IO 32
    max_io = _MAX_TENSOR_IO;
    tensor_num = graph->tensor_num;
    tensor_ref = _init_tensor_rel_buffer(graph, max_io);
    if(NULL == tensor_ref)
    {
        VSILOGE("init tensor_ref buffer fail");
        return NULL;
    }

    for (i = 0; i < tensor_num; i++)
    {
        in_num = 0;
        out_num = 0;

        for(j = 0; j < graph->node_num; j++)
        {
            node = vsi_nn_GetNode( graph, (vsi_nn_node_id_t)j );
            for(k = 0; k < node->output.num; k++)
            {
                if(node->output.tensors[k] == i)
                {
                    if(in_num > max_io)
                    {
                        VSILOGW("tensor ref input num > max_io %u, stop build", max_io);
                        break;
                    }
                    tensor_ref[i].input.table[in_num].node  = j;
                    tensor_ref[i].input.table[in_num].index = k;
                    in_num++;
                }
            }
            for(k = 0; k < node->input.num; k++)
            {
                if(node->input.tensors[k] == i)
                {
                    if(out_num > max_io)
                    {
                        VSILOGW("tensor ref output num > max_io %u, stop build", max_io);
                        break;
                    }
                    tensor_ref[i].output.table[out_num].node  = j;
                    tensor_ref[i].output.table[out_num].index = k;
                    out_num++;
                }
            }
        }
        tensor_ref[i].input.num = in_num;
        tensor_ref[i].output.num = out_num;
    }

    return tensor_ref;
} /* vsi_nn_CreateTensorRelevance() */
