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
#include <stdint.h>
#include <math.h>

#include "vsi_nn_platform.h"

#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_math.h"
#include "client/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"

#define PARA_NUM 7
#define KENEL_PARA_CNT 5
#define TENSOR_CNT 5

static float vsi_nn_DtypeToFloat32_Ex
    (
    uint8_t   * src,
    uint32_t    index,
    const vsi_nn_dtype_t * src_dtype
    )
{
    float value = 0.0f;
    vsi_status status = VSI_SUCCESS;

    src = src + index * vsi_nn_TypeGetBytes(src_dtype->vx_type);

    status = vsi_nn_DtypeToFloat32(src, &value, src_dtype);
    if(VSI_SUCCESS != status)
    {
        VSILOGE("Convert data to float32 fail!");
        value = 0.0f;
    }

    return value;
}

static vsi_status vsi_nn_Float32ToDtype_Ext
    (
    float   src,
    uint8_t   * dst,
    uint32_t    index,
    const vsi_nn_dtype_t * dst_dtype
    )
{
    vsi_nn_dtype_t src_dtype;

    dst = dst + index * vsi_nn_TypeGetBytes(dst_dtype->vx_type);

    memset( &src_dtype, 0, sizeof( vsi_nn_dtype_t ) );
    src_dtype.vx_type = VSI_NN_TYPE_FLOAT32;

    return vsi_nn_DtypeConvert( (uint8_t *)&src, &src_dtype, dst, dst_dtype );
} /* vsi_nn_Float32ToDtype_Ext */

static uint32_t getExpandTensorOffset(uint32_t index, uint32_t num_of_dims, uint32_t * in_dims,
                                       uint32_t *strides, uint32_t * out_dims)
{
    uint32_t offset = 0;
    uint32_t i = 0;

    for(i = 0; i < num_of_dims; i++)
    {
        if(in_dims[i] == out_dims[i])
            offset += strides[i] * (index % out_dims[i]);

        index /= out_dims[i];
    }

    return offset;
}

vsi_status VX_CALLBACK vxInstanceNormKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
    vsi_status status = VX_ERROR_INVALID_PARAMETERS;

    if(paramNum == PARA_NUM)
    {
        vx_context context = NULL;
        // tensor
        vx_tensor imgObj[KENEL_PARA_CNT] = { NULL };
        uint8_t* scale = NULL;
        uint8_t *input = NULL;
        uint8_t *output = NULL;
        uint8_t *bias = NULL;
        uint32_t i = 0;

        vsi_nn_tensor_attr_t in_attr, scale_attr, bias_attr, out_attr;
        uint32_t    stride_size[TENSOR_CNT][VSI_NN_MAX_DIM_NUM];
        vx_tensor_addressing user_addr[TENSOR_CNT]  = {NULL};

        // scalar
        vx_scalar scalar[1] = { NULL };
        float eps = .0f;

        status = VSI_SUCCESS;
        memset(&in_attr, 0, sizeof(vsi_nn_tensor_attr_t));
        memset(&scale_attr, 0, sizeof(vsi_nn_tensor_attr_t));
        memset(&bias_attr, 0, sizeof(vsi_nn_tensor_attr_t));
        memset(&out_attr, 0, sizeof(vsi_nn_tensor_attr_t));

        imgObj[0] = (vx_tensor)paramObj[0];
        imgObj[1] = (vx_tensor)paramObj[1];
        imgObj[2] = (vx_tensor)paramObj[2];
        imgObj[3] = (vx_tensor)paramObj[3];
        scalar[0] = (vx_scalar)paramObj[5];
        context = vxGetContext((vx_reference)node);
        if (context == NULL)
        {
            VSILOGE("vxGetContext failure! at line %d\n", __LINE__);
            goto OnError;
        }

        status = vsi_nn_vxGetTensorAttr(imgObj[0], &in_attr);
        status |= vsi_nn_vxGetTensorAttr(imgObj[1], &bias_attr);
        status |= vsi_nn_vxGetTensorAttr(imgObj[2], &scale_attr);
        status |= vsi_nn_vxGetTensorAttr(imgObj[3], &out_attr);
        if (status != VSI_SUCCESS)
        {
            VSILOGE("vsi_nn_vxGetTensorAttr failure! at line %d\n", __LINE__);
            goto OnError;
        }

        input = vsi_nn_ConvertRawTensorToData2(context, imgObj[0],
            &(in_attr), stride_size[0], &(user_addr[0]), VX_READ_ONLY);
        bias = vsi_nn_ConvertRawTensorToData2(context, imgObj[1],
            &(bias_attr), stride_size[1], &(user_addr[1]), VX_READ_ONLY);
        scale = vsi_nn_ConvertRawTensorToData2(context, imgObj[2],
            &(scale_attr), stride_size[2], &(user_addr[2]), VX_READ_ONLY);

        output = (uint8_t*)malloc(vsi_nn_vxGetTensorElementNum(&out_attr)*vsi_nn_GetTypeBytes(out_attr.dtype.vx_type));

        // scalar
        status = vxCopyScalar(scalar[0], &eps, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
        if (status != VSI_SUCCESS)
        {
            VSILOGE("vxCopyScalar failure! at line %d\n", __LINE__);
            goto OnError;
        }

        {
            uint32_t b = 0, c = 0, h = 0, w = 0;
            uint32_t height = in_attr.size[1];
            uint32_t width = in_attr.size[0];
            uint32_t ch = in_attr.size[2] > 0 ? in_attr.size[2] : 1;
            uint32_t bh = in_attr.size[3] > 0 ? in_attr.size[3] : 1;

            for (b = 0; b < bh; b++)
            {
                for (c = 0; c < ch; c++)
                {
                    vx_uint8   *scale_ptr  = NULL;
                    vx_uint8   *bias_ptr  = NULL;
                    vx_uint8   *input_ptr  = NULL;
                    uint32_t  scaleoffset = 0;
                    uint32_t  biasoffset = 0;

                    uint32_t  inoffset = 0;

                    uint32_t page = c * (height * width) + b * (height * width * ch);
                    float sum = .0f;
                    float sumsq = .0f;
                    float mean = .0f;
                    float vari = .0f;
                    float scaleVal = .0f;
                    float biasVal = .0f;
                    float data = 0;
                    //float scaleVal = vsi_nn_Fp16toFp32(scale[c]);
                    //float biasVal = bias[c];

                    scaleoffset = getExpandTensorOffset(c, scale_attr.dim_num, scale_attr.size, stride_size[2], scale_attr.size);
                    biasoffset = getExpandTensorOffset(c, bias_attr.dim_num, bias_attr.size, stride_size[1], bias_attr.size);
                    scale_ptr = (vx_uint8 *)scale + scaleoffset;
                    bias_ptr = (vx_uint8 *)bias + biasoffset;

                    scaleVal = vsi_nn_DtypeToFloat32_Ex(scale_ptr, 0, &scale_attr.dtype);
                    biasVal = vsi_nn_DtypeToFloat32_Ex(bias_ptr, 0, &bias_attr.dtype);

                    for (h = 0; h < height; h++)
                    {
                        uint32_t len = page + h * width;

                        for (w = 0; w < width; w++)
                        {
                            uint32_t index = len + w;
                            inoffset = getExpandTensorOffset(index, in_attr.dim_num, in_attr.size, stride_size[0], in_attr.size);
                            input_ptr = (vx_uint8 *)input + inoffset;
                            sum += vsi_nn_DtypeToFloat32_Ex(input_ptr, 0, &in_attr.dtype);
                        }
                    }
                    mean = sum / (width * height);
                    for (h = 0; h < height; h++)
                    {
                        uint32_t len = page + h * width;
                        for (w = 0; w < width; w++)
                        {
                            uint32_t index = len + w;
                            inoffset = getExpandTensorOffset(index, in_attr.dim_num, in_attr.size, stride_size[0], in_attr.size);
                            input_ptr = (vx_uint8 *)input + inoffset;
                            data = vsi_nn_DtypeToFloat32_Ex(input_ptr, 0, &in_attr.dtype) - mean;
                            sumsq += data * data;
                        }
                    }
                    vari = sumsq / (width * height);
                    vari = (float)(1.0 / sqrtf(vari + eps));
                    for (h = 0; h < height; h++)
                    {
                        uint32_t len = page + h * width;
                        for (w = 0; w < width; w++)
                        {
                            float normVal = 0;
                            uint32_t index = len + w;
                            inoffset = getExpandTensorOffset(index, in_attr.dim_num, in_attr.size, stride_size[0], in_attr.size);
                            input_ptr = (vx_uint8 *)input + inoffset;
                            data = vsi_nn_DtypeToFloat32_Ex(input_ptr, 0, &in_attr.dtype) - mean;

                            normVal = data * vari * scaleVal + biasVal;
                            vsi_nn_Float32ToDtype_Ext(normVal, output, index, &out_attr.dtype);
                        }
                    }
                }
            }
        }

        //output tensor
        status = vsi_nn_copy_tensor_patch(imgObj[3], &out_attr, output, VX_WRITE_ONLY);
        if (status != VSI_SUCCESS)
        {
            VSILOGE("vsi_nn_copy_tensor_patch failure! at line %d\n", __LINE__);
            goto OnError;
        }
OnError:
        if(input) free(input);
        if(scale) free(scale);
        if(bias) free(bias);
        if(output) free(output);
        for(i = 0; i < TENSOR_CNT; i++)
            if (user_addr[i]) vxReleaseTensorAddressing(&(user_addr[i]));
    }
    return status;
}

vsi_status VX_CALLBACK vsi_nn_InstanceNormMeanVariValidator
    (
    vx_node node,
    const vx_reference parameters[],
    uint32_t num,
    vx_meta_format metas[]
)
{
    return VSI_SUCCESS;
}

vsi_status VX_CALLBACK vxInstanceNormInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    uint32_t paraNum
    )
{
    vsi_status status = VSI_SUCCESS;
    // Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
    vx_kernel_execution_parameters_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vx_tensor     input           = (vx_tensor)paramObj[0];
    vx_tensor     output          = (vx_tensor)paramObj[3];
    vx_tensor     sumSqr          = (vx_tensor)paramObj[4];
    vx_scalar     scalar_flg      = (vx_scalar)paramObj[6];
    vsi_nn_tensor_attr_t in_attr, sqrsum_attr, out_attr;
    vsi_nn_type_e inputDataFormat = VSI_NN_TYPE_FLOAT16;
    vsi_nn_type_e outputDataFormat = VSI_NN_TYPE_FLOAT16;
    vx_float32 scaleIn = 0;
    vx_float32 scaleOut = 0;
    vx_float32 reScaleOut_u8 = 0;
    vx_float32 scale_inOut = 0;
    int32_t output_ZP = 0;
    int32_t input_ZP = 0;
    vx_float32 in_scale_fl = 1, out_scale_fl = 1, inOut_fl_scale = 1;
    vx_float32 dimRatio = 0;
    vx_uint32 group_num = 0;
    vx_int32 height = 0, width = 0, chn = 0;
    int32_t rsFlg = 0;

    memset(&in_attr, 0, sizeof(vsi_nn_tensor_attr_t));
    memset(&sqrsum_attr, 0, sizeof(vsi_nn_tensor_attr_t));
    memset(&out_attr, 0, sizeof(vsi_nn_tensor_attr_t));

    status = vsi_nn_vxGetTensorAttr(input, &in_attr);
    status |= vsi_nn_vxGetTensorAttr(sumSqr, &sqrsum_attr);
    status |= vsi_nn_vxGetTensorAttr(output, &out_attr);
    status |= vxCopyScalar(scalar_flg, &rsFlg, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    if(VSI_SUCCESS != status)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
        return status;
    }

    inputDataFormat = in_attr.dtype.vx_type;
    input_ZP = in_attr.dtype.zero_point;
    scaleIn = in_attr.dtype.scale;
    outputDataFormat = out_attr.dtype.vx_type;
    output_ZP = out_attr.dtype.zero_point;
    scaleOut = out_attr.dtype.scale;

    if(inputDataFormat == VSI_NN_TYPE_INT8
        || inputDataFormat == VSI_NN_TYPE_INT16)
    {
        if (in_attr.dtype.fl > 0)
        {
            in_scale_fl = (1.0f / ((vx_float32) (1 << in_attr.dtype.fl)));
        }
        else
        {
            in_scale_fl = ((vx_float32) (1 << -in_attr.dtype.fl));
        }
    }

    if(outputDataFormat == VSI_NN_TYPE_INT8
        || outputDataFormat == VSI_NN_TYPE_INT16)
    {
        if (out_attr.dtype.fl > 0)
        {
            out_scale_fl = (vx_float32)(1 << out_attr.dtype.fl);
        }
        else
        {
            out_scale_fl = (1.0f / (vx_float32)(1 << -out_attr.dtype.fl));
        }
    }

    if((outputDataFormat == VSI_NN_TYPE_INT8
        || outputDataFormat == VSI_NN_TYPE_INT16)
        && (inputDataFormat == VSI_NN_TYPE_INT8
        || inputDataFormat == VSI_NN_TYPE_INT16))
    {
        inOut_fl_scale = in_scale_fl * out_scale_fl;
    }

    width = in_attr.size[0];
    height = in_attr.size[1];
    chn = sqrsum_attr.size[1];
    if(rsFlg)
    {
        height = in_attr.size[1] / sqrsum_attr.size[1];
    }

    if(outputDataFormat == VSI_NN_TYPE_UINT8)
        reScaleOut_u8 = 1 / scaleOut;
    dimRatio = (vx_float32)(1.0 / (width * height));

    in_attr.size[2] = (in_attr.dim_num <= 2)?1:in_attr.size[2];

    group_num = (width + 255) / 256;

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkOffset[2] = 0;
    shaderParam.globalWorkScale[0]  = 16;
    if(inputDataFormat == VSI_NN_TYPE_INT16 || inputDataFormat == VSI_NN_TYPE_FLOAT16)
    {
        shaderParam.globalWorkScale[0]  = 8;
        group_num = (width + 127) / 128;
    }
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.globalWorkScale[2]  = 1;
    shaderParam.globalWorkSize[0]   = gcmALIGN((width + shaderParam.globalWorkScale[0] - 1)
        / shaderParam.globalWorkScale[0], 4);
#if 1
    shaderParam.globalWorkSize[1]   = (chn + shaderParam.globalWorkScale[1] - 1)
        / shaderParam.globalWorkScale[1];
    shaderParam.globalWorkSize[2]   = (1 + shaderParam.globalWorkScale[2] - 1)
        / shaderParam.globalWorkScale[2];
#elif 0
    shaderParam.globalWorkSize[1]   = in_attr.size[1];
    shaderParam.globalWorkSize[2]   = in_attr.size[2];
#else
    shaderParam.globalWorkScale[1]  = 16;
    shaderParam.globalWorkSize[1]   = (in_attr.size[1] + shaderParam.globalWorkScale[1] - 1)
        / shaderParam.globalWorkScale[1];
    shaderParam.globalWorkSize[2]   = in_attr.size[2];
#endif

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));
    if(VSI_SUCCESS != status)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
    }
    {
        vx_uint32 UniFP16toFP32Lo4_dp4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };
        uint32_t uniConvertInt32toUint8_2x8[16] = {
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        uint32_t uniConvertEndInt16Fp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };
        vx_uint32 uniConvert1stUint8SubZpToFp32_4x4[16] = {
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0xffff0001, 0x00000000, 0xffff0001, 0x00000000, 0xffff0001, 0x00000000, 0xffff0001, 0x00000000 // Constant
        };
        vx_uint32 uniConvert2ndUint8SubZpToFp32_4x4[16] = {
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00050004, 0x00070006, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0xffff0001, 0x00000000, 0xffff0001, 0x00000000, 0xffff0001, 0x00000000, 0xffff0001, 0x00000000 // Constant
        };
        vx_uint32 uniConvert3rdUint8SubZpToFp32_4x4[16] = {
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00090008, 0x000b000a, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0xffff0001, 0x00000000, 0xffff0001, 0x00000000, 0xffff0001, 0x00000000, 0xffff0001, 0x00000000 // Constant
        };
        vx_uint32 uniConvert4thUint8SubZpToFp32_4x4[16] = {
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x000d000c, 0x000f000e, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0xffff0001, 0x00000000, 0xffff0001, 0x00000000, 0xffff0001, 0x00000000, 0xffff0001, 0x00000000 // Constant
        };
        uint32_t uniConvertInt16Fp32Fst_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        };
        uint32_t uniConvertInt16Fp32Secd_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        };
        uint32_t uniConvertInt32toInt16_2x8[16] = {
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        uint32_t uniConvertDirUint8Fp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };
        uint32_t uniConvertEndUint8Fp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };
        uint32_t uniConvertTrdUint8Fp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00090008, 0x000b000a, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };
        uint32_t uniConvertFthUint8Fp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x000d000c, 0x000f000e, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };
        vx_uint32 uniConvertHalfToFp16_2x8[16] = {
            0x11111111, // TCfg
            0x11110000, // ASelt
            0x06040200, 0x06040200, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
        };

        status = vxSetNodeUniform(nodObj, "height", 1, &height);
        status |= vxSetNodeUniform(nodObj, "dimRatio", 1, &dimRatio);
        status |= vxSetNodeUniform(nodObj, "group_num", 1, &group_num);
        status |= vxSetNodeUniform(nodObj, "UniFP16toFP32Lo4_dp4x4", 1, UniFP16toFP32Lo4_dp4x4);
        status |= vxSetNodeUniform(nodObj, "uniConvertHalfToFp16_2x8", 1, uniConvertHalfToFp16_2x8);
        if(inputDataFormat == VSI_NN_TYPE_FLOAT16)
        {
            status |= vxSetNodeUniform(nodObj, "uniConvertEndInt16Fp32_4x4", 1, uniConvertEndInt16Fp32_4x4);
        }

        if(inputDataFormat == VSI_NN_TYPE_UINT8)
        {
            status |= vxSetNodeUniform(nodObj, "uniConvertInt32toUint8_2x8", 1, uniConvertInt32toUint8_2x8);
            status |= vxSetNodeUniform(nodObj, "uniConvert1stUint8SubZpToFp32_4x4", 1, uniConvert1stUint8SubZpToFp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvert2ndUint8SubZpToFp32_4x4", 1, uniConvert2ndUint8SubZpToFp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvert3rdUint8SubZpToFp32_4x4", 1, uniConvert3rdUint8SubZpToFp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvert4thUint8SubZpToFp32_4x4", 1, uniConvert4thUint8SubZpToFp32_4x4);
            status |= vxSetNodeUniform(nodObj, "inputZP", 1, &input_ZP);
            status |= vxSetNodeUniform(nodObj, "input_scale", 1, &scaleIn);
        }
        else if(inputDataFormat == VSI_NN_TYPE_INT8)
        {
            status |= vxSetNodeUniform(nodObj, "uniConvertInt32toUint8_2x8", 1, uniConvertInt32toUint8_2x8);
            status |= vxSetNodeUniform(nodObj, "uniConvertDirInt8Fp32_4x4", 1, uniConvertDirUint8Fp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvertEndInt8Fp32_4x4", 1, uniConvertEndUint8Fp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvertTrdInt8Fp32_4x4", 1, uniConvertTrdUint8Fp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvertFthInt8Fp32_4x4", 1, uniConvertFthUint8Fp32_4x4);

            status |= vxSetNodeUniform(nodObj, "input_fl_scale", 1, &in_scale_fl);
        }
        else if(inputDataFormat == VSI_NN_TYPE_INT16)
        {
            status |= vxSetNodeUniform(nodObj, "uniConvertInt16Fp32Fst_4x4",
                1, uniConvertInt16Fp32Fst_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvertInt16Fp32Secd_4x4",
                1, uniConvertInt16Fp32Secd_4x4);
            status |= vxSetNodeUniform(nodObj, "input_fl_scale", 1, &in_scale_fl);
        }

        if(outputDataFormat == VSI_NN_TYPE_UINT8)
        {
            status |= vxSetNodeUniform(nodObj, "output_ZP", 1, &output_ZP);
            status |= vxSetNodeUniform(nodObj, "outputScale", 1, &reScaleOut_u8);
        }
        else if(outputDataFormat == VSI_NN_TYPE_INT8)
        {
            status |= vxSetNodeUniform(nodObj, "output_fl_scale", 1, &out_scale_fl);
        }
        else if(outputDataFormat == VSI_NN_TYPE_INT16)
        {
            status |= vxSetNodeUniform(nodObj, "uniConvertInt32toInt16_2x8",
                1, uniConvertInt32toInt16_2x8);
            status |= vxSetNodeUniform(nodObj, "output_fl_scale", 1, &out_scale_fl);
        }

        if(outputDataFormat == VSI_NN_TYPE_UINT8 && inputDataFormat == VSI_NN_TYPE_UINT8)
        {
            scale_inOut = reScaleOut_u8 * scaleIn;
            status |= vxSetNodeUniform(nodObj, "scale_inOut", 1, &scale_inOut);
        }
        else if((outputDataFormat == VSI_NN_TYPE_INT8 && inputDataFormat == VSI_NN_TYPE_INT8)
            || (outputDataFormat == VSI_NN_TYPE_INT16 && inputDataFormat == VSI_NN_TYPE_INT16))
        {
            status |= vxSetNodeUniform(nodObj, "inOut_fl_scale", 1, &inOut_fl_scale);
        }
        if(VSI_SUCCESS != status)
        {
            VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
        }
    }
    return status;
}
static vx_param_description_t vxInstanceNormKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED}
};

vsi_status VX_CALLBACK vxInstanceNormMeanVariInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    uint32_t paraNum
    )
{
    vsi_status status = VSI_SUCCESS;
    // Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
    vx_kernel_execution_parameters_t shaderParam = {
        2,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vx_tensor     input           = (vx_tensor)paramObj[0];
    vx_tensor     output          = (vx_tensor)paramObj[1];
    vx_scalar     scalar_flg      = (vx_scalar)paramObj[3];
    vsi_nn_tensor_attr_t in_attr, out_attr;
    vsi_nn_type_e inputDataFormat = VSI_NN_TYPE_FLOAT16;
    vx_float32 scaleIn = 0;
    int32_t input_ZP = 0;
    vx_uint32 iter = 0;
    int32_t sumInZp = 0;
    int32_t tmpZp1 = 0;
    vx_float32 tmpZp2 = 0;
    vx_float32 e2InScale = 0;
    vx_float32 rowSumScale = 0;
    int32_t rsFlg = 0;
    int32_t width = 0;
    int32_t height = 0;
    int32_t chn = 0;
    vx_float32 in_scale_fl = 1, inFlScale_s2 = 1;

    memset(&in_attr, 0, sizeof(vsi_nn_tensor_attr_t));
    memset(&out_attr, 0, sizeof(vsi_nn_tensor_attr_t));

    status = vsi_nn_vxGetTensorAttr(input, &in_attr);
    status |= vsi_nn_vxGetTensorAttr(output, &out_attr);
    status |= vxCopyScalar(scalar_flg, &rsFlg, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    if(VSI_SUCCESS != status)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
        return status;
    }

    inputDataFormat = in_attr.dtype.vx_type;
    input_ZP = in_attr.dtype.zero_point;
    scaleIn = in_attr.dtype.scale;

    if(inputDataFormat == VSI_NN_TYPE_INT8
        || inputDataFormat == VSI_NN_TYPE_INT16)
    {
        if (in_attr.dtype.fl > 0)
        {
            in_scale_fl = (1.0f / ((vx_float32) (1 << in_attr.dtype.fl)));
        }
        else
        {
            in_scale_fl = ((vx_float32) (1 << -in_attr.dtype.fl));
        }
        inFlScale_s2 = in_scale_fl * in_scale_fl;
    }

    width = in_attr.size[0];
    height = in_attr.size[1];
    chn = out_attr.size[1];
    if(rsFlg)
    {
        height = in_attr.size[1] / out_attr.size[1];
    }
    iter = height * 16;
    if(inputDataFormat == VSI_NN_TYPE_UINT8)
    {
        sumInZp = input_ZP * iter * (-1);
        tmpZp1 = (-2) * input_ZP;
        e2InScale = scaleIn * scaleIn;
        tmpZp2 = input_ZP * input_ZP * e2InScale;
        rowSumScale = height * 16 * tmpZp2;
    }

    in_attr.size[2] = (in_attr.dim_num <= 2)?1:in_attr.size[2];

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkScale[0]  = 1;
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.localWorkSize[0]    = 16;
    shaderParam.localWorkSize[1]    = 1;
    //shaderParam.globalWorkSize[0]   = 16;
    if(inputDataFormat == VSI_NN_TYPE_INT8
        || inputDataFormat == VSI_NN_TYPE_UINT8)
        shaderParam.globalWorkSize[0]   = (in_attr.size[0] + 255) / 256 * 16;
    else if(inputDataFormat == VSI_NN_TYPE_INT16
        || inputDataFormat == VSI_NN_TYPE_FLOAT16)
        shaderParam.globalWorkSize[0]   = (in_attr.size[0] + 127) / 128 * 16;
    shaderParam.globalWorkSize[1]   = chn;

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));
    if(VSI_SUCCESS != status)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
    }
    if(inputDataFormat == VSI_NN_TYPE_UINT8)
    {
        vx_uint32 uniSumU8_16x1[16] = {
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x76543210, 0xfedcba98, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
        };
        vx_uint32 uniSqrSum_16x1[16] = {
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x76543210, 0xfedcba98, // ABin
            0x55555555, // BSelt
            0x76543210, 0xfedcba98, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        status |= vxSetNodeUniform(nodObj, "uniSumU8_16x1", 1, uniSumU8_16x1);
        status |= vxSetNodeUniform(nodObj, "uniSqrSum_16x1", 1, uniSqrSum_16x1);
        status |= vxSetNodeUniform(nodObj, "sumInZp", 1, &sumInZp);
        status |= vxSetNodeUniform(nodObj, "tmpZp1", 1, &tmpZp1);
        status |= vxSetNodeUniform(nodObj, "input_scale", 1, &scaleIn);
        status |= vxSetNodeUniform(nodObj, "e2InScale", 1, &e2InScale);
        status |= vxSetNodeUniform(nodObj, "rowSumScale", 1, &rowSumScale);
    }
    else if(inputDataFormat == VSI_NN_TYPE_INT8)
    {
        vx_uint32 uniSumInt8_16x1[16] = {
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x76543210, 0xfedcba98, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
        };
        vx_uint32 uniSqrSumInt8_16x1[16] = {
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x76543210, 0xfedcba98, // ABin
            0x55555555, // BSelt
            0x76543210, 0xfedcba98, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };

        status |= vxSetNodeUniform(nodObj, "uniSumInt8_16x1", 1, uniSumInt8_16x1);
        status |= vxSetNodeUniform(nodObj, "uniSqrSumInt8_16x1", 1, uniSqrSumInt8_16x1);
        status |= vxSetNodeUniform(nodObj, "input_fl_scale", 1, &in_scale_fl);
        status |= vxSetNodeUniform(nodObj, "inFlScale_s2", 1, &inFlScale_s2);
    }
    else if(inputDataFormat == VSI_NN_TYPE_INT16)
    {
        vx_uint32 uniInt16SumSqr_dp8x2[16] = {
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x76543210, 0x76543210, // ABin
            0x5555aaaa, // BSelt
            0x00000000, 0x76543210, // BBin
            0x00000300, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        status |= vxSetNodeUniform(nodObj, "uniInt16SumSqr_dp8x2", 1, uniInt16SumSqr_dp8x2);
        status |= vxSetNodeUniform(nodObj, "input_fl_scale", 1, &in_scale_fl);
        status |= vxSetNodeUniform(nodObj, "inFlScale_s2", 1, &inFlScale_s2);
    }
    else if(inputDataFormat == VSI_NN_TYPE_FLOAT16)
    {
        vx_uint32 uniFp16SumSqr_dp8x2[16] = {
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x76543210, 0x76543210, // ABin
            0x5555aaaa, // BSelt
            0x00000000, 0x76543210, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        status |= vxSetNodeUniform(nodObj, "uniFp16SumSqr_dp8x2", 1, uniFp16SumSqr_dp8x2);
    }
    status |= vxSetNodeUniform(nodObj, "width", 1, &width);
    status |= vxSetNodeUniform(nodObj, "height", 1, &height);

    if(VSI_SUCCESS != status)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
    }
    return status;
}
static vx_param_description_t vxInstanceNormMeanVariKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED}
};
#ifdef __cplusplus
extern "C" {
#endif
vx_kernel_description_t vxInstanceNormKernelInfo =
{
    VX_KERNEL_ENUM_INSTANCENORM,
    VX_KERNEL_NAME_INSTANCENORM,
    NULL,
    vxInstanceNormKernelParam,
    (sizeof(vxInstanceNormKernelParam) / sizeof(vxInstanceNormKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxInstanceNormInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxInstanceNormMeanVariKernelInfo_U8 =
{
    VX_KERNEL_ENUM_INSTANCENORM,
    VX_KERNEL_NAME_INSTANCENORMMEAN_VARI_UINT8,
    NULL,
    vxInstanceNormMeanVariKernelParam,
    (sizeof(vxInstanceNormMeanVariKernelParam) / sizeof(vxInstanceNormMeanVariKernelParam[0])),
    vsi_nn_InstanceNormMeanVariValidator,
    NULL,
    NULL,
    vxInstanceNormMeanVariInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxInstanceNormMeanVariKernelInfo_U8_Rs =
{
    VX_KERNEL_ENUM_INSTANCENORM,
    VX_KERNEL_NAME_INSTANCENORMMEAN_VARI_UINT8_RSHC,
    NULL,
    vxInstanceNormMeanVariKernelParam,
    (sizeof(vxInstanceNormMeanVariKernelParam) / sizeof(vxInstanceNormMeanVariKernelParam[0])),
    vsi_nn_InstanceNormMeanVariValidator,
    NULL,
    NULL,
    vxInstanceNormMeanVariInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxInstanceNormMeanVariKernelInfo_I8 =
{
    VX_KERNEL_ENUM_INSTANCENORM,
    VX_KERNEL_NAME_INSTANCENORMMEAN_VARI_INT8,
    NULL,
    vxInstanceNormMeanVariKernelParam,
    (sizeof(vxInstanceNormMeanVariKernelParam) / sizeof(vxInstanceNormMeanVariKernelParam[0])),
    vsi_nn_InstanceNormMeanVariValidator,
    NULL,
    NULL,
    vxInstanceNormMeanVariInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxInstanceNormMeanVariKernelInfo_I8_Rs =
{
    VX_KERNEL_ENUM_INSTANCENORM,
    VX_KERNEL_NAME_INSTANCENORMMEAN_VARI_INT8_RSHC,
    NULL,
    vxInstanceNormMeanVariKernelParam,
    (sizeof(vxInstanceNormMeanVariKernelParam) / sizeof(vxInstanceNormMeanVariKernelParam[0])),
    vsi_nn_InstanceNormMeanVariValidator,
    NULL,
    NULL,
    vxInstanceNormMeanVariInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxInstanceNormMeanVariKernelInfo_I16 =
{
    VX_KERNEL_ENUM_INSTANCENORM,
    VX_KERNEL_NAME_INSTANCENORMMEAN_VARI_INT16,
    NULL,
    vxInstanceNormMeanVariKernelParam,
    (sizeof(vxInstanceNormMeanVariKernelParam) / sizeof(vxInstanceNormMeanVariKernelParam[0])),
    vsi_nn_InstanceNormMeanVariValidator,
    NULL,
    NULL,
    vxInstanceNormMeanVariInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxInstanceNormMeanVariKernelInfo_I16_Rs =
{
    VX_KERNEL_ENUM_INSTANCENORM,
    VX_KERNEL_NAME_INSTANCENORMMEAN_VARI_INT16_RSHC,
    NULL,
    vxInstanceNormMeanVariKernelParam,
    (sizeof(vxInstanceNormMeanVariKernelParam) / sizeof(vxInstanceNormMeanVariKernelParam[0])),
    vsi_nn_InstanceNormMeanVariValidator,
    NULL,
    NULL,
    vxInstanceNormMeanVariInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxInstanceNormMeanVariKernelInfo_Fp16 =
{
    VX_KERNEL_ENUM_INSTANCENORM,
    VX_KERNEL_NAME_INSTANCENORMMEAN_VARI_FP16,
    NULL,
    vxInstanceNormMeanVariKernelParam,
    (sizeof(vxInstanceNormMeanVariKernelParam) / sizeof(vxInstanceNormMeanVariKernelParam[0])),
    vsi_nn_InstanceNormMeanVariValidator,
    NULL,
    NULL,
    vxInstanceNormMeanVariInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxInstanceNormMeanVariKernelInfo_Fp16_Rs =
{
    VX_KERNEL_ENUM_INSTANCENORM,
    VX_KERNEL_NAME_INSTANCENORMMEAN_VARI_FP16_RSHC,
    NULL,
    vxInstanceNormMeanVariKernelParam,
    (sizeof(vxInstanceNormMeanVariKernelParam) / sizeof(vxInstanceNormMeanVariKernelParam[0])),
    vsi_nn_InstanceNormMeanVariValidator,
    NULL,
    NULL,
    vxInstanceNormMeanVariInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxInstanceNormKernelInfo_U8 =
{
    VX_KERNEL_ENUM_INSTANCENORM,
    VX_KERNEL_NAME_INSTANCENORM_UINT8,
    NULL,
    vxInstanceNormKernelParam,
    (sizeof(vxInstanceNormKernelParam) / sizeof(vxInstanceNormKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxInstanceNormInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxInstanceNormKernelInfoU8_Rs =
{
    VX_KERNEL_ENUM_INSTANCENORM,
    VX_KERNEL_NAME_INSTANCENORM_UINT8_RSHC,
    NULL,
    vxInstanceNormKernelParam,
    (sizeof(vxInstanceNormKernelParam) / sizeof(vxInstanceNormKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxInstanceNormInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxInstanceNormKernelInfoU8_Fp16 =
{
    VX_KERNEL_ENUM_INSTANCENORM,
    VX_KERNEL_NAME_INSTANCENORM_UINT8_FP16,
    NULL,
    vxInstanceNormKernelParam,
    (sizeof(vxInstanceNormKernelParam) / sizeof(vxInstanceNormKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxInstanceNormInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxInstanceNormKernelInfoU8Fp16_Rs =
{
    VX_KERNEL_ENUM_INSTANCENORM,
    VX_KERNEL_NAME_INSTANCENORM_UINT8_FP16_RSHC,
    NULL,
    vxInstanceNormKernelParam,
    (sizeof(vxInstanceNormKernelParam) / sizeof(vxInstanceNormKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxInstanceNormInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxInstanceNormKernelInfo_I8 =
{
    VX_KERNEL_ENUM_INSTANCENORM,
    VX_KERNEL_NAME_INSTANCENORM_INT8,
    NULL,
    vxInstanceNormKernelParam,
    (sizeof(vxInstanceNormKernelParam) / sizeof(vxInstanceNormKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxInstanceNormInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxInstanceNormKernelInfoI8_Rs =
{
    VX_KERNEL_ENUM_INSTANCENORM,
    VX_KERNEL_NAME_INSTANCENORM_INT8_RSHC,
    NULL,
    vxInstanceNormKernelParam,
    (sizeof(vxInstanceNormKernelParam) / sizeof(vxInstanceNormKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxInstanceNormInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxInstanceNormKernelInfoI8_Fp16 =
{
    VX_KERNEL_ENUM_INSTANCENORM,
    VX_KERNEL_NAME_INSTANCENORM_INT8_FP16,
    NULL,
    vxInstanceNormKernelParam,
    (sizeof(vxInstanceNormKernelParam) / sizeof(vxInstanceNormKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxInstanceNormInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxInstanceNormKernelInfoI8Fp16_Rs =
{
    VX_KERNEL_ENUM_INSTANCENORM,
    VX_KERNEL_NAME_INSTANCENORM_INT8_FP16_RSHC,
    NULL,
    vxInstanceNormKernelParam,
    (sizeof(vxInstanceNormKernelParam) / sizeof(vxInstanceNormKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxInstanceNormInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxInstanceNormKernelInfo_I16 =
{
    VX_KERNEL_ENUM_INSTANCENORM,
    VX_KERNEL_NAME_INSTANCENORM_INT16,
    NULL,
    vxInstanceNormKernelParam,
    (sizeof(vxInstanceNormKernelParam) / sizeof(vxInstanceNormKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxInstanceNormInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxInstanceNormKernelInfoI16_Rs =
{
    VX_KERNEL_ENUM_INSTANCENORM,
    VX_KERNEL_NAME_INSTANCENORM_INT16_RSHC,
    NULL,
    vxInstanceNormKernelParam,
    (sizeof(vxInstanceNormKernelParam) / sizeof(vxInstanceNormKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxInstanceNormInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxInstanceNormKernelInfoI16_Fp16 =
{
    VX_KERNEL_ENUM_INSTANCENORM,
    VX_KERNEL_NAME_INSTANCENORM_INT16_FP16,
    NULL,
    vxInstanceNormKernelParam,
    (sizeof(vxInstanceNormKernelParam) / sizeof(vxInstanceNormKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxInstanceNormInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxInstanceNormKernelInfoI16Fp16_Rs =
{
    VX_KERNEL_ENUM_INSTANCENORM,
    VX_KERNEL_NAME_INSTANCENORM_INT16_FP16_RSHC,
    NULL,
    vxInstanceNormKernelParam,
    (sizeof(vxInstanceNormKernelParam) / sizeof(vxInstanceNormKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxInstanceNormInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxInstanceNormKernelInfo_Fp16 =
{
    VX_KERNEL_ENUM_INSTANCENORM,
    VX_KERNEL_NAME_INSTANCENORM_FP16,
    NULL,
    vxInstanceNormKernelParam,
    (sizeof(vxInstanceNormKernelParam) / sizeof(vxInstanceNormKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxInstanceNormInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxInstanceNormKernelInfoFp16_Rs =
{
    VX_KERNEL_ENUM_INSTANCENORM,
    VX_KERNEL_NAME_INSTANCENORM_FP16_RSHC,
    NULL,
    vxInstanceNormKernelParam,
    (sizeof(vxInstanceNormKernelParam) / sizeof(vxInstanceNormKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxInstanceNormInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxInstanceNormKernelInfo_CPU =
{
    VX_KERNEL_ENUM_INSTANCENORM,
    VX_KERNEL_NAME_INSTANCENORM,
    vxInstanceNormKernel,
    vxInstanceNormKernelParam,
    (sizeof(vxInstanceNormKernelParam) / sizeof(vxInstanceNormKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_INSTANCENORM_list[] =
{
    &vxInstanceNormKernelInfo_CPU,
    &vxInstanceNormMeanVariKernelInfo_U8_Rs,
    &vxInstanceNormMeanVariKernelInfo_I8_Rs,
    &vxInstanceNormMeanVariKernelInfo_I16_Rs,
    &vxInstanceNormMeanVariKernelInfo_Fp16_Rs,
    &vxInstanceNormMeanVariKernelInfo_U8,
    &vxInstanceNormMeanVariKernelInfo_I8,
    &vxInstanceNormMeanVariKernelInfo_I16,
    &vxInstanceNormMeanVariKernelInfo_Fp16,
    &vxInstanceNormKernelInfoU8_Rs,
    &vxInstanceNormKernelInfoU8Fp16_Rs,
    &vxInstanceNormKernelInfoI8_Rs,
    &vxInstanceNormKernelInfoI8Fp16_Rs,
    &vxInstanceNormKernelInfoI16_Rs,
    &vxInstanceNormKernelInfoI16Fp16_Rs,
    &vxInstanceNormKernelInfoFp16_Rs,
    &vxInstanceNormKernelInfo_U8,
    &vxInstanceNormKernelInfoU8_Fp16,
    &vxInstanceNormKernelInfo_I8,
    &vxInstanceNormKernelInfoI8_Fp16,
    &vxInstanceNormKernelInfo_I16,
    &vxInstanceNormKernelInfoI16_Fp16,
    &vxInstanceNormKernelInfo_Fp16,
    NULL
};
#ifdef __cplusplus
}
#endif
