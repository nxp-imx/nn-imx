/****************************************************************************
*
*    Copyright (c) 2005 - 2019 by Vivante Corp.  All rights reserved.
*
*    The material in this file is confidential and contains trade secrets
*    of Vivante Corporation. This is proprietary information owned by
*    Vivante Corporation. No part of this work may be disclosed,
*    reproduced, copied, transmitted, or used in any way for any purpose,
*    without the express written permission of Vivante Corporation.
*
*****************************************************************************/



#include "util.h"

#define F16_EXPONENT_BITS 0x1F
#define F16_EXPONENT_BIAS 15

#define F16_EXPONENT_SHIFT 10
#define F16_MANTISSA_BITS ((1 << F16_EXPONENT_SHIFT) - 1)
#define F16_MANTISSA_SHIFT (23 - F16_EXPONENT_SHIFT)
#define F16_MAX_EXPONENT (F16_EXPONENT_BITS << F16_EXPONENT_SHIFT)

#define F21_EXPONENT_SHIFT 15
#define F21_MANTISSA_BITS ((1 << F21_EXPONENT_SHIFT) - 1)
#define F21_MANTISSA_SHIFT (23 - F21_EXPONENT_SHIFT)
#define F21_MAX_EXPONENT (F16_EXPONENT_BITS << F21_EXPONENT_SHIFT)

double getCurrentSystemTimeMs()
{
    double t = 0;
#if defined(_MSC_VER)
    t = omp_get_wtime() * 1000;
#elif defined(__linux__)
    struct timeval tv;
    gettimeofday(&tv, NULL);
    t = tv.tv_sec * 1000 + tv.tv_usec / 1000;
#endif
    return t;
}

vx_float32 Fp16toFp32(const vx_int16 in)
{
    vx_int32 t1;
    vx_int32 t2;
    vx_int32 t3;
    vx_float32 out;

    t1 = in & 0x7fff;                       // Non-sign bits
    t2 = in & 0x8000;                       // Sign bit
    t3 = in & 0x7c00;                       // Exponent

    t1 <<= 13;                              // Align mantissa on MSB
    t2 <<= 16;                              // Shift sign bit into position

    t1 += 0x38000000;                       // Adjust bias

    t1 = (t3 == 0 ? 0 : t1);                // Denormals-as-zero

    t1 |= t2;                               // Re-insert sign bit

    *((uint32_t*)&out) = t1;

    return out;
}

vx_int16 Fp32toFp16(vx_float32 val)
{
    vx_uint32 f32 = (*(vx_uint32 *) &val);
    vx_int16 f16 = 0;
    /* Decode IEEE 754 little-endian 32-bit floating-point value */
    int sign = (f32 >> 16) & 0x8000;
    /* Map exponent to the range [-127,128] */
    int exponent = ((f32 >> 23) & 0xff) - 127;
    int mantissa = f32 & 0x007fffff;
    if (exponent == 128)
    { /* Infinity or NaN */
        if (mantissa)
        {
            /* Flush NaN to 0. */
            f16 = (vx_int16)sign;
        }
        else
        {
            /* Clamp to HALF_MAX/HALF_MIN. */
            f16 = (vx_int16)(sign | ((F16_EXPONENT_BITS - 1) << F16_EXPONENT_SHIFT) | F16_MANTISSA_BITS);
        }
    }
    else if (exponent > 15)
    { /* Overflow - clamp to HALF_MAX/HALF_MIN. */
        f16 = (vx_int16)(sign | ((F16_EXPONENT_BITS - 1) << F16_EXPONENT_SHIFT) | F16_MANTISSA_BITS);
    }
    else if (exponent > -15)
    { /* Representable value */
        /* RTNE */
        int roundingBit = (mantissa >> (F16_MANTISSA_SHIFT - 1)) & 0x1;
        int stickyBits = mantissa & 0xFFF;
        exponent += F16_EXPONENT_BIAS;
        mantissa >>= F16_MANTISSA_SHIFT;
        if (roundingBit)
        {
            if (stickyBits || (mantissa & 0x1))
            {
                mantissa++;
                if (mantissa > F16_MANTISSA_BITS)
                {
                    exponent++;
                    if (exponent > 30)
                    {
                        /* Clamp to HALF_MAX/HALF_MIN. */
                        exponent--;
                        mantissa--;
                    }
                    else
                    {
                        mantissa &= F16_MANTISSA_BITS;
                    }
                }
            }
        }
        f16 = (vx_int16)(sign | exponent << F16_EXPONENT_SHIFT | mantissa);
    }
    else
    {
        f16 = (vx_int16)sign;
    }
    return f16;
}

int vxcGetTypeSize(vx_enum format)
{
    switch(format)
    {
        case VX_TYPE_INT8:
        case VX_TYPE_UINT8:
            return 1;
        case VX_TYPE_INT16:
        case VX_TYPE_UINT16:
            return 2;
        case VX_TYPE_INT32:
        case VX_TYPE_UINT32:
            return 4;
        case VX_TYPE_INT64:
        case VX_TYPE_UINT64:
            return 8;
        case VX_TYPE_FLOAT32:
            return 4;
        case VX_TYPE_FLOAT64:
            return 8;
        case VX_TYPE_ENUM:
            return 4;
        case VX_TYPE_FLOAT16:
            return 2;
    }
    return 4;
}

/*refine the function with ovx1.2 api, but this fucntion only supports no-view tensor*/
int vxcMemcpy(vx_context context, vx_tensor& tensor, void *hostPtr, vx_accessor_e usage)
{
    vx_uint32       output_size[NN_TENSOR_MAX_DIMENSION];
    vx_size         stride_size[NN_TENSOR_MAX_DIMENSION];
    vx_size         view_s[NN_TENSOR_MAX_DIMENSION];
    vx_size         view_e[NN_TENSOR_MAX_DIMENSION];
    vx_int32        num_of_dims;
    vx_enum         data_format;

    VX_ERR_CHECK( vxQueryTensor(tensor, VX_TENSOR_DIMS, output_size, sizeof(output_size)) );
    VX_ERR_CHECK( vxQueryTensor(tensor, VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)) );
    VX_ERR_CHECK( vxQueryTensor(tensor, VX_TENSOR_DATA_TYPE, &data_format, sizeof(data_format)) );

    memset(view_s, 0, sizeof(view_s));
    memset(view_e, 0, sizeof(view_e));

    for (int i = 0; i < num_of_dims; i++)
        view_e[i] = output_size[i];

    stride_size[0] = vxcGetTypeSize(data_format);
    for (int i = 1; i < num_of_dims; i++)
        stride_size[i] = stride_size[i-1] * output_size[i- 1];

    VX_ERR_CHECK( vxCopyTensorPatch(tensor, num_of_dims, view_s, view_e, stride_size, hostPtr, usage, VX_MEMORY_TYPE_HOST) );

    vx_bool value = vx_true_e;
    if(usage == VX_WRITE_ONLY)
        VX_ERR_CHECK( vxSetTensorAttribute(tensor, VX_TENSOR_VALUE,     &value,     sizeof(vx_bool)) );

    return ANEURALNETWORKS_NO_ERROR;
}

template <typename T>
void convertRank_nhwc2whcn(T *org_data, T* dst_data,
                              vx_uint32 whcn_dims[4])
{
    vx_uint32 dim_w = whcn_dims[0];
    vx_uint32 dim_h = whcn_dims[1];
    vx_uint32 dim_c = whcn_dims[2];
    vx_uint32 dim_n = whcn_dims[3];

    for(vx_uint32 n = 0; n < dim_n; n++)
    {
        vx_uint32 block = dim_w * dim_h * dim_c;
        for(vx_uint32 c = 0; c < dim_c; c++)
        {
            vx_uint32 slice = dim_w * dim_h;
            for(vx_uint32 h = 0; h < dim_h; h++)
            {
                for(vx_uint32 w = 0; w < dim_w; w++)
                    dst_data[w + n * block + c * slice + h * dim_w] = org_data[c + n * block + h * dim_w * dim_c + w * dim_c];
            }
        }
    }

}

vx_status convertDims(vx_uint32 * dims, vx_uint32 *org_dims, vx_uint32 count, bool SNforFC)
{
     /* Convert dims from CWHN(NHWC) => WHCN */
    switch (count)
    {
    case 4:
        dims[0] = org_dims[2]; /* W : */
        dims[1] = org_dims[1]; /* H : */
        dims[2] = org_dims[3]; /* C : */
        dims[3] = org_dims[0]; /* N : */
        break;
    case 3:
        dims[0] = org_dims[2]; /* W : */
        dims[1] = org_dims[1]; /* H : */
        dims[2] = org_dims[0]; /* C : */
        dims[3] = 1;           /* N : */
        break;
    case 2:
        {
            if(SNforFC)
            {
            dims[0] = 1;            /* S : */
            dims[1] = 1;            /* N : */
            dims[2] = org_dims[1];  /* C : */
            dims[3] = org_dims[0];  /* N : */
            }
            else
            {
                dims[0] = org_dims[1];
                dims[1] = org_dims[0];
                dims[2] = 1;
                dims[3] = 1;
            }
        }

        break;
    case 1:
        dims[0] = org_dims[0];
        break;
    default:
        break;
    }

    return VX_SUCCESS;
}

void convertTensorDataFromFp322Fp16(vx_context context,AnnOperand &operand)
{
#if !HIGH_PRECISION_COMPUTE
    if( !( operand.tensorAttribute.valued       == vx_true_e &&
           operand.tensorAttribute.lifeTime     == VX_TENSOR_LIFE_TIME_STATIC &&
           operand.tensorAttribute.precision    == VX_TENSOR_PRECISION_AUTO &&
           operand.tensorAttribute.dataType     == VX_TYPE_FLOAT32
          )
       )
        return;

    vx_uint32 data_length = 1;
    for(vx_uint32 i = 0; i < operand.dimensionCount; i++)
        data_length *= operand.dimensions[i];

    void *orgData = malloc(data_length * vxcGetTypeSize(VX_TYPE_FLOAT32));
    vxcMemcpy(context, operand.tensor, orgData, VX_READ_ONLY);

    void *formattedData = NULL;
    formattedData  = malloc(data_length * vxcGetTypeSize(VX_TYPE_FLOAT16));

    for(uint32_t i = 0; i < data_length; i++)
        *((vx_int16 *)formattedData  + i) = Fp32toFp16(((float *) orgData)[i]);

    std::vector<vx_size> dimensions;
    for(vx_uint32 i  = 0; i < operand.dimensionCount; i++)
        dimensions.push_back(operand.dimensions[i]);
    vx_tensor tensorFp16 = vxCreateTensor(context, operand.dimensionCount, &dimensions[0], VX_TYPE_FLOAT16,0);
    vxcMemcpy(context, tensorFp16, formattedData, VX_WRITE_ONLY);

    vxReleaseTensor(&operand.tensor);
    operand.tensor = tensorFp16;
    operand.tensorAttribute.dataType = VX_TYPE_FLOAT16;

    VX_ERR_CHECK( vxSetTensorAttribute(operand.tensor, VX_TENSOR_RANK,      &operand.tensorAttribute.rank,      sizeof(vx_enum)) );
    VX_ERR_CHECK( vxSetTensorAttribute(operand.tensor, VX_TENSOR_VALUE,     &operand.tensorAttribute.valued,    sizeof(vx_bool)) );
    VX_ERR_CHECK( vxSetTensorAttribute(operand.tensor, VX_TENSOR_LIFETIME,  &operand.tensorAttribute.lifeTime,  sizeof(vx_enum)) );
    VX_ERR_CHECK( vxSetTensorAttribute(operand.tensor, VX_TENSOR_PRECISION, &operand.tensorAttribute.precision, sizeof(vx_enum)) );


    if(orgData)         free(orgData);
    if(formattedData)   free(formattedData);
#endif
}

void convertRankAndFormat(vx_context context,AnnOperand &operand, bool convertSNForFC)
{
    vx_tensor tensor = operand.tensor;
    vx_bool changed = vx_false_e;
    vx_bool valuedFlag = operand.tensorAttribute.valued;
    vx_enum life_time = operand.tensorAttribute.lifeTime;

    /*vxQueryTensor(tensor, VX_TENSOR_VALUE, &valuedFlag, sizeof(vx_bool));
    vxQueryTensor(tensor, VX_TENSOR_LIFETIME, &life_time, sizeof(vx_enum));
    */
    if(valuedFlag == vx_true_e && life_time == VX_TENSOR_LIFE_TIME_STATIC)
    {
        vx_enum         rank, dst_rank;
        vx_enum         precision;
        vx_int32        num_of_dims;
        vx_enum         org_data_format, dst_data_format;
        vx_uint32       tensor_size[NN_TENSOR_MAX_DIMENSION];
        vx_uint32       stride_size[NN_TENSOR_MAX_DIMENSION];
        vx_uint32       convert_tensor_size[NN_TENSOR_MAX_DIMENSION] = {1, 1, 1, 1};

        VX_ERR_CHECK( vxQueryTensor(tensor, VX_TENSOR_PRECISION,    &precision,     sizeof(vx_enum)) );
        VX_ERR_CHECK( vxQueryTensor(tensor, VX_TENSOR_RANK,         &rank,          sizeof(vx_enum)) );
        VX_ERR_CHECK( vxQueryTensor(tensor, VX_TENSOR_DIMS,         tensor_size,    sizeof(tensor_size)) );
        VX_ERR_CHECK( vxQueryTensor(tensor, VX_TENSOR_NUMBER_OF_DIMS,  &num_of_dims,   sizeof(num_of_dims)) );
        VX_ERR_CHECK( vxQueryTensor(tensor, VX_TENSOR_DATA_TYPE,    &org_data_format,   sizeof(org_data_format)) );

        if(rank == VX_TENSOR_RANK_WHCN && precision == VX_TENSOR_PRECISION_HIGH)
            return;

        dst_rank = rank;
        dst_data_format = org_data_format;

        stride_size[0] = 1;
        for (int i = 1; i < num_of_dims; i++)
        {
            stride_size[i] = stride_size[i-1] * tensor_size[i - 1];
        }

        vx_uint32 data_length = 1;
        for(int i = 0; i < num_of_dims; i++)
            data_length *= tensor_size[i];

        void *formattedData = NULL;
        void *orgData = malloc(data_length * vxcGetTypeSize(org_data_format));
        vxcMemcpy(context, tensor, orgData, VX_READ_ONLY);

#if !HIGH_PRECISION_COMPUTE
        if(precision != VX_TENSOR_PRECISION_HIGH && org_data_format == VX_TYPE_FLOAT32)
        {
            dst_data_format = VX_TYPE_FLOAT16;
            formattedData  = malloc(data_length * vxcGetTypeSize(dst_data_format));
            for(uint32_t i = 0; i < data_length; i++)
                *((vx_int16 *)formattedData  + i) = Fp32toFp16(((float *) orgData)[i]);

            changed  = vx_true_e;
        }
        else
#endif
            formattedData = orgData;

        void *rankedData = NULL;
        if(rank == VX_TENSOR_RANK_CWHN || rank == VX_TENSOR_RANK_SN)
        {
            dst_rank = VX_TENSOR_RANK_WHCN;
            convertDims(convert_tensor_size, tensor_size, num_of_dims, convertSNForFC);
            rankedData = malloc(data_length * vxcGetTypeSize(dst_data_format));

            operand.dimensionCount = convertSNForFC ? 4 : num_of_dims;
            for(vx_uint32 i = 0; i < operand.dimensionCount; i++)
                operand.dimensions[i] = convert_tensor_size[i];

            switch (dst_data_format)
            {
                case VX_TYPE_FLOAT16:
                    convertRank_nhwc2whcn<vx_uint16>( (vx_uint16 *)formattedData, (vx_uint16 *)rankedData, convert_tensor_size);
                    break;
                case VX_TYPE_FLOAT32:
                    convertRank_nhwc2whcn<vx_float32>( (vx_float32*)formattedData, (vx_float32*)rankedData, convert_tensor_size);
                    break;
                case VX_TYPE_UINT32:
                case VX_TYPE_INT32:
                    convertRank_nhwc2whcn<vx_uint32>( (vx_uint32 *)formattedData, (vx_uint32 *)rankedData, convert_tensor_size);
                    break;
                case VX_TYPE_UINT8:
                case VX_TYPE_INT8:
                    convertRank_nhwc2whcn<vx_uint8>( (vx_uint8*)formattedData, (vx_uint8*)rankedData, convert_tensor_size);
                    break;
                default:
                    fprintf(stderr, "the data type have not been supported\n");
                    assert(0);
                    break;
            }
            changed  = vx_true_e;
        }
        else
        {
            memcpy(convert_tensor_size, tensor_size, num_of_dims * sizeof(vx_int32));
            rankedData = formattedData;
        }

        if(changed)
        {
            vx_enum quant_format = (dst_data_format == VX_TYPE_UINT8) || (dst_data_format == VX_TYPE_INT32) ? VX_QUANT_AFFINE_SCALE : VX_QUANT_DYNAMIC_FIXED_POINT;
            vx_tensor_create_params_t param = { operand.dimensionCount, convert_tensor_size, dst_data_format, quant_format, {{0}}};
            if(quant_format == VX_QUANT_AFFINE_SCALE)
            {
                param.quant_data.affine.scale     = (operand.scale != 0)? operand.scale: 1.0f;
                param.quant_data.affine.zeroPoint = operand.zeroPoint;
            }
            else
            {
                param.quant_data.dfp.fixed_point_pos = 0;/*TODO: need to be modify the hard core*/
            }

            /*param.quant_data.affine.scale     = operand.scale;
            param.quant_data.affine.zeroPoint = operand.zeroPoint;*/

            vxReleaseTensor(&tensor);
            operand.tensor = vxCreateTensor2(context, &param, sizeof(vx_tensor_create_params_t));

            VX_ERR_CHECK( vxSetTensorAttribute(operand.tensor, VX_TENSOR_RANK,  &dst_rank,  sizeof(vx_enum)) );
            VX_ERR_CHECK( vxSetTensorAttribute(operand.tensor, VX_TENSOR_VALUE,  &valuedFlag,  sizeof(vx_bool)) );
            VX_ERR_CHECK( vxSetTensorAttribute(operand.tensor, VX_TENSOR_LIFETIME,  &life_time,  sizeof(vx_enum)) );
            VX_ERR_CHECK( vxSetTensorAttribute(operand.tensor, VX_TENSOR_PRECISION,  &precision,  sizeof(vx_enum)) );

            operand.tensorAttribute.dataType = dst_data_format;
            operand.tensorAttribute.rank = dst_rank;

            vxcMemcpy(context, operand.tensor, rankedData, VX_WRITE_ONLY);

            if(rankedData != formattedData)
                free(rankedData);
            if(formattedData != orgData)
                free(formattedData);
            if(orgData)
                free(orgData);
        }
    }
    return ;
}

vx_tensor convertScalar2Tensor(vx_context context,AnnOperand &operand)
{
    vx_tensor tensor = NULL;
    if(operand.type > 2)
        return tensor;

    vx_enum dataType = enumConvertorANN2VX(operand.type);
    vx_enum quant_format = ( dataType == VX_TYPE_FLOAT32)? VX_QUANT_DYNAMIC_FIXED_POINT : VX_QUANT_AFFINE_SCALE;
        vx_uint32 size[1] = {1}, stride[1] = { (vx_uint32)vxcGetTypeSize(dataType)};
    vx_tensor_create_params_t param;
    INITIALIZE_STRUCT(param);
    param.sizes = size;
    param.num_of_dims = 1;
    param.data_format = dataType;
    param.quant_format = quant_format;
    if(quant_format == VX_QUANT_AFFINE_SCALE)
    {
        param.quant_data.affine.scale = 1.0f;
        param.quant_data.affine.zeroPoint = 0;
    }

    tensor = vxCreateTensor2(context, &param, sizeof(vx_tensor_create_params_t) );
    if (tensor == NULL)
    {
        printf("vxCreateTensor failure! at line %d\n", __LINE__);
        assert(0);
    }
    operand.tensor = tensor;

    vx_tensor_addressing addr = vxCreateTensorAddressing(context, size, stride, 1);
    VX_ERR_CHECK( vxCopyTensorPatchForNN11(operand.tensor, NULL, addr, &operand.scalar.i32, VX_WRITE_ONLY, 0) );
    vxReleaseTensorAddressing(&addr);

    vx_enum precision = VX_TENSOR_PRECISION_HIGH;
    vx_enum life_time = VX_TENSOR_LIFE_TIME_STATIC;
    VX_ERR_CHECK( vxSetTensorAttribute(operand.tensor, VX_TENSOR_PRECISION,     &precision,     sizeof(vx_enum)) );
    VX_ERR_CHECK( vxSetTensorAttribute(operand.tensor, VX_TENSOR_LIFETIME,     &life_time,     sizeof(vx_enum)) );

    operand.type = ANEURALNETWORKS_TENSOR_INT32;
    operand.tensorAttribute.dataType = VX_TYPE_INT32;
    operand.tensorAttribute.lifeTime = VX_TENSOR_LIFE_TIME_STATIC;
    operand.tensorAttribute.valued = vx_true_e;

    return tensor;
}
