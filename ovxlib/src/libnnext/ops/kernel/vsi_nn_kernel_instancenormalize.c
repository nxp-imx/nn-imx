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

#define INPUT_FP16 0
#define INPUT_U8 1
#define INPUT_I8 0
#define INPUT_I16 0
#define OUTPUT_FP16 0
#define OUTPUT_I16 0
#define OUTPUT_U8 1
#define OUTPUT_I8 0


vx_status getDataFactor(vx_uint32 data, vx_uint32 *factor, vx_uint32 minLimit, vx_uint32 maxLimit, vx_uint32 alignData)
{
    vx_uint32 i         = 0;
    vx_uint32 maxFactor = alignData - 1;
    vx_status status    = VX_FAILURE;

    for (i = minLimit; i <= maxLimit; i ++)
    {
        if (data % i == 0)
        {
            if (status == VX_FAILURE && data % i == 0)
            {
                *factor      = i;
                maxFactor    = i;
                status       = VX_SUCCESS;
                continue;
            }
            else if ((i % alignData) < (maxFactor % alignData))
            {
               *factor      = i;
               maxFactor    = i;
               status       = VX_SUCCESS;
            }
        }
    }

    return status;
}

void myInstanceNormFunc
    (
    void* imgIn,
    int16_t* scale,
    float*   bias,
    float    eps,
    void* imgOut,
    uint32_t input_dim,
    uint32_t width,
    uint32_t height,
    uint32_t channel,
    uint32_t batch,
    int32_t inZp,
    int32_t outZp,
    float inScale,
    float outScale,
    int8_t inFl,
    int8_t outFl
    )
{
#if INPUT_FP16
    int16_t* tmpIn = (int16_t*)imgIn;
#elif INPUT_U8
    uint8_t* tmpIn = (uint8_t*)imgIn;
#elif INPUT_I8
    int8_t* tmpIn = (int8_t*)imgIn;
#elif INPUT_I16
    int16_t* tmpIn = (int16_t*)imgIn;
#endif
#if OUTPUT_FP16
    int16_t* tmpOut = (int16_t*)imgOut;
#elif OUTPUT_U8
    uint8_t* tmpOut = (uint8_t*)imgOut;
#elif OUTPUT_I8
    int8_t* tmpOut = (int8_t*)imgOut;
#elif OUTPUT_I16
    int16_t* tmpOut = (int16_t*)imgOut;
#endif
    uint32_t ch = (input_dim <= 2) ? 1 : channel;
    uint32_t bn = (input_dim <= 3) ? 1 : batch;
    uint32_t b = 0, c = 0, h = 0, w = 0;
    VSILOGE("Hello myInstanceNormFunc!\n");
    for (b = 0; b < bn; b++)
    {
        for (c = 0; c < ch; c++)
        {
            uint32_t page = c * (height * width) + b * (height * width * ch);
            float sum = .0f;
            float sumsq = .0f;
            float mean = .0f;
            float vari = .0f;
            float scaleVal = vsi_nn_Fp16toFp32(scale[c]);
            float biasVal = bias[c];

            for (h = 0; h < height; h++)
            {
                uint32_t len = page + h * width;

                for (w = 0; w < width; w++)
                {
                    uint32_t index = len + w;
#if INPUT_FP16
                    sum += vsi_nn_Fp16toFp32(tmpIn[index]);
#elif INPUT_U8
                    sum += vsi_nn_AffineToFp32(tmpIn[index], inScale, inZp, VSI_NN_TYPE_UINT8);
#elif INPUT_I8
                    sum += vsi_nn_DFPToFp32(tmpIn[index], inFl, VSI_NN_TYPE_INT8);
#elif INPUT_I16
                    sum += vsi_nn_DFPToFp32(tmpIn[index], inFl, VSI_NN_TYPE_INT16);
#endif
                }
            }
            mean = sum / (width * height);
            for (h = 0; h < height; h++)
            {
                uint32_t len = page + h * width;
                for (w = 0; w < width; w++)
                {
                    uint32_t index = len + w;
#if INPUT_FP16
                    float data = vsi_nn_Fp16toFp32(tmpIn[index]) - mean;
#elif INPUT_U8
                    float data = vsi_nn_AffineToFp32(tmpIn[index], inScale, inZp, VSI_NN_TYPE_UINT8) - mean;
#elif INPUT_I8
                    float data = vsi_nn_DFPToFp32(tmpIn[index], inFl, VSI_NN_TYPE_INT8) - mean;
#elif INPUT_I16
                    float data = vsi_nn_DFPToFp32(tmpIn[index], inFl, VSI_NN_TYPE_INT16) - mean;
#endif
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
                    float data, normVal;
                    uint32_t index = len + w;
#if INPUT_FP16
                    data = vsi_nn_Fp16toFp32(tmpIn[index]) - mean;
#elif INPUT_U8
                    data = vsi_nn_AffineToFp32(tmpIn[index], inScale, inZp, VSI_NN_TYPE_UINT8) - mean;
#elif INPUT_I8
                    data = vsi_nn_DFPToFp32(tmpIn[index], inFl, VSI_NN_TYPE_INT8) - mean;
#elif INPUT_I16
                    data = vsi_nn_DFPToFp32(tmpIn[index], inFl, VSI_NN_TYPE_INT16) - mean;
#endif

                    normVal = data * vari * scaleVal + biasVal;
#if OUTPUT_FP16
                    tmpOut[index] = vsi_nn_Fp32ToFp16(normVal);
#elif OUTPUT_U8
                    tmpOut[index] = vsi_nn_Fp32ToAffine(normVal, outScale, outZp, VSI_NN_TYPE_UINT8);
#elif OUTPUT_I8
                    tmpOut[index] = vsi_nn_Fp32ToDFP(normVal, outFl, VSI_NN_TYPE_INT8);
#elif OUTPUT_I16
                    tmpOut[index] = vsi_nn_Fp32ToDFP(normVal, outFl, VSI_NN_TYPE_INT16);
#endif
                }
            }
        }
    }
    return;
}
vsi_status VX_CALLBACK vxInstanceNormKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
    vsi_status status = VX_ERROR_INVALID_PARAMETERS;

    if(paramNum == 6)
    {
        vx_context context = NULL;
        // tensor
        vx_tensor imgObj[5] = { NULL };
        int16_t* scale = NULL;
#if INPUT_FP16
        int16_t *input = NULL;
#else
        uint8_t *input = NULL;
#endif
#if OUTPUT_FP16
        int16_t *output = NULL;
#else
        uint8_t *output = NULL;
#endif
        float *bias = NULL;
        uint32_t input_size[DIM_SIZE] = {1, 1, 1, 1}, output_size[DIM_SIZE] = {1, 1, 1, 1};
        uint32_t scale_size[DIM_SIZE] = {1, 1, 1, 1}, bias_size[DIM_SIZE] = {1, 1, 1, 1};
        uint32_t input_stride_size[4]  = {0};
        uint32_t output_stride_size[4] = {0};
        uint32_t scale_stride_size[4]  = {0};
        uint32_t bias_stride_size[4] = {0};
        vx_tensor_addressing input_user_addr = NULL;
        vx_tensor_addressing output_user_addr = NULL;
        vx_tensor_addressing scale_user_addr = NULL;
        vx_tensor_addressing bias_user_addr = NULL;
        vsi_nn_type_e inputFormat = VSI_NN_TYPE_FLOAT16, outputFormat = VSI_NN_TYPE_FLOAT16;
        vsi_nn_type_e scaleFormat = VSI_NN_TYPE_FLOAT16, biasFormat = VSI_NN_TYPE_FLOAT16;
        uint32_t input_dims = 0, output_dims = 0;
        uint32_t scale_dims = 0, bias_dims = 0;
        uint32_t i;
        int32_t in_zp, out_zp;
        float in_scale, out_scale;
        int8_t in_fixpoint = 0, out_fixpoint = 0;
        // scalar
        vx_scalar scalar[1] = { NULL };
        float eps = .0f;
        vsi_nn_tensor_attr_t attr[4];

        memset(&attr[0], 0, sizeof(vsi_nn_tensor_attr_t));
        memset(&attr[1], 0, sizeof(vsi_nn_tensor_attr_t));
        memset(&attr[2], 0, sizeof(vsi_nn_tensor_attr_t));
        memset(&attr[3], 0, sizeof(vsi_nn_tensor_attr_t));

        status  = vsi_nn_vxGetTensorAttr(imgObj[0], &attr[0]);
        status |= vsi_nn_vxGetTensorAttr(imgObj[1], &attr[1]);
        status |= vsi_nn_vxGetTensorAttr(imgObj[2], &attr[2]);
        status |= vsi_nn_vxGetTensorAttr(imgObj[3], &attr[3]);
        if (status != VX_SUCCESS)
        {
            VSILOGE("vsi_nn_vxGetTensorAttr  failure! at line %d\n", __LINE__);
            return status;
        }

        status = VX_SUCCESS;

        imgObj[0] = (vx_tensor)paramObj[0];
        imgObj[1] = (vx_tensor)paramObj[1];
        imgObj[2] = (vx_tensor)paramObj[2];
        imgObj[3] = (vx_tensor)paramObj[3];
        scalar[0] = (vx_scalar)paramObj[5];
        context = vxGetContext((vx_reference)node);
        if (context == NULL)
        {
            VSILOGE("vxGetContext failure! at line %d\n", __LINE__);
            return status;
        }
        //input
        input_dims  = attr[0].dim_num;
        inputFormat = attr[0].dtype.vx_type;
        for (i = 0; i < input_dims; i++)
        {
            input_size[i] = attr[0].size[i];
        }
        in_zp       = attr[0].dtype.zero_point;
        in_scale    = attr[0].dtype.scale;
        in_fixpoint = attr[0].dtype.fl;
        //bias
        bias_dims   = attr[1].dim_num;
        for (i = 0; i < bias_dims; i++)
        {
            bias_size[i] = attr[1].size[i];
        }
        biasFormat  = attr[1].dtype.vx_type;
        //scale
        scale_dims   = attr[2].dim_num;
        for (i = 0; i < scale_dims; i++)
        {
            scale_size[i] = attr[2].size[i];
        }
        scaleFormat  = attr[2].dtype.vx_type;
        //output
        output_dims  = attr[3].dim_num;
        outputFormat = attr[3].dtype.vx_type;
        for (i = 0; i < output_dims; i++)
        {
            output_size[i] = attr[3].size[i];
        }
        out_zp       = attr[0].dtype.zero_point;
        out_scale    = attr[0].dtype.scale;
        out_fixpoint = attr[0].dtype.fl;

        input_size[2] = (input_dims <= 2)?1:input_size[2];
        input_size[3] = (input_dims <= 3)?1:input_size[3];

        input_stride_size[0]  = vsi_nn_GetTypeBytes(inputFormat);
        output_stride_size[0] = vsi_nn_GetTypeBytes(outputFormat);
        for (i=1; i< input_dims; i++)
        {
            input_stride_size[i]  = input_stride_size[i-1] * input_size[i-1];
            output_stride_size[i] = output_stride_size[i-1] * output_size[i-1];
        }

#if INPUT_FP16
        input  = (int16_t*)malloc(input_size[0]*input_size[1]*input_size[2]*sizeof(int16_t));
#else
        input  = (uint8_t*)malloc(input_size[0]*input_size[1]*input_size[2]*sizeof(uint8_t));
#endif
#if OUTPUT_FP16
        output = (int16_t*)malloc(output_size[0]*output_size[1]*output_size[2]*sizeof(int16_t));
#else
        output = (uint8_t*)malloc(output_size[0]*output_size[1]*output_size[2]*sizeof(uint8_t));
#endif
        input_user_addr = vxCreateTensorAddressing(context, input_size, input_stride_size, input_dims);
        vxCopyTensorPatch_11(imgObj[0], NULL, input_user_addr, input, VX_READ_ONLY, 0);
        //scale and bias
        scale_stride_size[0]  = vsi_nn_GetTypeBytes(scaleFormat);
        bias_stride_size[0] = vsi_nn_GetTypeBytes(biasFormat);
        for (i=1; i< scale_dims; i++)
        {
            scale_stride_size[i]  = scale_stride_size[i-1] * scale_size[i-1];
            bias_stride_size[i] = bias_stride_size[i-1] * bias_size[i-1];
        }
        scale  = (int16_t*)malloc(scale_size[0]*sizeof(int16_t));
        bias = (float*)malloc(bias_size[0]*sizeof(float));
        bias_user_addr = vxCreateTensorAddressing(context, bias_size, bias_stride_size, bias_dims);
        vxCopyTensorPatch_11(imgObj[1], NULL, bias_user_addr, bias, VX_READ_ONLY, 0);
        scale_user_addr = vxCreateTensorAddressing(context, scale_size, scale_stride_size, scale_dims);
        vxCopyTensorPatch_11(imgObj[2], NULL, scale_user_addr, scale, VX_READ_ONLY, 0);

        // scalar
        status = vxCopyScalar(scalar[0], &eps, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxCopyScalar failure! at line %d\n", __LINE__);
            goto OnError;
        }
        // Call C Prototype
        myInstanceNormFunc(input, scale, bias, eps, output, input_dims, input_size[0],
            input_size[1], input_size[2], input_size[3], in_zp, out_zp, in_scale, out_scale, in_fixpoint, out_fixpoint);

        //output tensor
        output_user_addr = vxCreateTensorAddressing(context, output_size,
            output_stride_size, output_dims);
        vxCopyTensorPatch_11(imgObj[3], NULL, output_user_addr, output, VX_WRITE_ONLY, 0);
OnError:
        if(input) free(input);
        if(scale) free(scale);
        if(bias) free(bias);
        if(output) free(output);
        if(input_user_addr) vxReleaseTensorAddressing(&input_user_addr);
        if(scale_user_addr) vxReleaseTensorAddressing(&scale_user_addr);
        if(bias_user_addr) vxReleaseTensorAddressing(&bias_user_addr);
        if(output_user_addr) vxReleaseTensorAddressing(&output_user_addr);
    }
    return status;
}

vsi_status VX_CALLBACK vsi_nn_InstanceNormSumValidator
    (
    vx_node node,
    const vx_reference parameters[],
    uint32_t num,
    vx_meta_format metas[]
)
{
    vsi_status status = VSI_SUCCESS;
    vx_uint32 index = 0;
    for(index = 0; index < num; index++)
    {
        if(index == 2 || index == 3)
        {
            vx_enum item_type = 0;
            vx_size capacity = 0;
            vx_array array = (vx_array)parameters[index];
            status = vxQueryArray(array, VX_ARRAY_ITEMTYPE, &item_type, sizeof(vx_enum));
            status |= vxQueryArray(array, VX_ARRAY_CAPACITY, &capacity, sizeof(vx_size));
            status |= vxSetMetaFormatAttribute(metas[index], VX_ARRAY_ITEMTYPE, &item_type, sizeof(item_type));
            status |= vxSetMetaFormatAttribute(metas[index], VX_ARRAY_CAPACITY, &capacity, sizeof(capacity));
            status |= vxSetArrayAttribute(array, VX_ARRAY_NUMITEMS, &capacity, sizeof(capacity));
        }
    }

    return status;
}

vsi_status VX_CALLBACK vsi_nn_InstanceNormSqrValidator
    (
    vx_node node,
    const vx_reference parameters[],
    uint32_t num,
    vx_meta_format metas[]
)
{
    vsi_status status = VSI_SUCCESS;
    vx_uint32 index = 0;
    for(index = 0; index < num; index++)
    {
        if(index == 5 || index == 6)
        {
            vx_enum item_type = 0;
            vx_size capacity = 0;
            vx_array array = (vx_array)parameters[index];
            status = vxQueryArray(array, VX_ARRAY_ITEMTYPE, &item_type, sizeof(vx_enum));
            status |= vxQueryArray(array, VX_ARRAY_CAPACITY, &capacity, sizeof(vx_size));
            status |= vxSetMetaFormatAttribute(metas[index], VX_ARRAY_ITEMTYPE, &item_type, sizeof(item_type));
            status |= vxSetMetaFormatAttribute(metas[index], VX_ARRAY_CAPACITY, &capacity, sizeof(capacity));
            status |= vxSetArrayAttribute(array, VX_ARRAY_NUMITEMS, &capacity, sizeof(capacity));
        }
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
    vsi_status status = VX_SUCCESS;
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
    uint32_t      input_size[DIM_SIZE]   = {1, 1, 1, 1};
    uint32_t      input_dims      = 0;
    vsi_nn_type_e inputDataFormat = VSI_NN_TYPE_FLOAT16;
    vsi_nn_type_e outputDataFormat = VSI_NN_TYPE_FLOAT16;
    vx_float32 scaleIn = 0;
    vx_float32 scaleOut = 0;
    vx_float32 reScaleOut_u8 = 0;
    vx_float32 scale_inOut = 0;
    int32_t output_ZP = 0;
    int32_t input_ZP = 0;
    int8_t input_fl, output_fl;
    vx_float32 in_scale_fl = 1, out_scale_fl = 1, inFlScale_s2 = 1;
    vx_float32 dimRatio = 0;
    vx_uint32 factor = 1;
    vx_uint32 maxWorkGroupSize = 40;

    vx_uint32  i        = 0;
    vsi_nn_tensor_attr_t attr[2];
    memset(&attr[0], 0, sizeof(vsi_nn_tensor_attr_t));
    memset(&attr[1], 0, sizeof(vsi_nn_tensor_attr_t));
    status  = vsi_nn_vxGetTensorAttr(input, &attr[0]);
    status |= vsi_nn_vxGetTensorAttr(output, &attr[1]);
    if (status != VX_SUCCESS)
    {
        VSILOGE("vsi_nn_vxGetTensorAttr  failure! at line %d\n", __LINE__);
        return status;
    }

    input_dims       = attr[0].dim_num;
    for (i = 0; i < input_dims; i++)
    {
        input_size[i] = attr[0].size[i];
    }
    inputDataFormat   = attr[0].dtype.vx_type;
    input_ZP          = attr[0].dtype.zero_point;
    scaleIn           = attr[0].dtype.scale;
    input_fl          = attr[0].dtype.fl;
    outputDataFormat  = attr[1].dtype.vx_type;
    output_ZP         = attr[1].dtype.zero_point;
    scaleOut          = attr[1].dtype.scale;
    output_fl         = attr[1].dtype.fl;

    if(inputDataFormat == VSI_NN_TYPE_INT8
        || inputDataFormat == VSI_NN_TYPE_INT16)
    {
        if (input_fl > 0)
        {
            in_scale_fl = (1.0f / ((vx_float32) (1 << input_fl)));
        }
        else
        {
            in_scale_fl = ((vx_float32) (1 << -input_fl));
        }
        inFlScale_s2 = in_scale_fl * in_scale_fl;
    }

    if(outputDataFormat == VSI_NN_TYPE_INT8
        || outputDataFormat == VSI_NN_TYPE_INT16)
    {
        if (output_fl > 0)
        {
            out_scale_fl = (vx_float32)(1 << output_fl);
        }
        else
        {
            out_scale_fl = (1.0f / (vx_float32)(1 << -output_fl));
        }
    }

    if(outputDataFormat == VSI_NN_TYPE_UINT8)
        reScaleOut_u8 = 1 / scaleOut;
    dimRatio = (vx_float32)(1.0 / (input_size[0] * input_size[1]));

    input_size[2] = (input_dims <= 2)?1:input_size[2];
    input_size[2] = (input_dims == 4)?(input_size[2] * input_size[3]):input_size[2];

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkOffset[2] = 0;
    shaderParam.globalWorkScale[0]  = 16;
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.globalWorkScale[2]  = 1;
    shaderParam.globalWorkSize[0]   = gcmALIGN((input_size[0] + shaderParam.globalWorkScale[0] - 1)
        / shaderParam.globalWorkScale[0], 4);
    shaderParam.globalWorkSize[1]   = (input_size[2] + shaderParam.globalWorkScale[1] - 1)
        / shaderParam.globalWorkScale[1];
    shaderParam.globalWorkSize[2]   = (1 + shaderParam.globalWorkScale[2] - 1)
        / shaderParam.globalWorkScale[2];

    if(inputDataFormat == VSI_NN_TYPE_FLOAT16
      || inputDataFormat == VSI_NN_TYPE_INT8
      || inputDataFormat == VSI_NN_TYPE_INT16
      )
    {
        shaderParam.globalWorkScale[0]  = 1;
        shaderParam.localWorkSize[0]    = 1;
        shaderParam.localWorkSize[1]    = 1;
        if (input_size[2] <= maxWorkGroupSize)
            shaderParam.localWorkSize[2]    = input_size[2];
        else if (getDataFactor(input_size[2], &factor, 2, maxWorkGroupSize, 8) == VX_SUCCESS)
            shaderParam.localWorkSize[2]    = factor;
        else
            shaderParam.localWorkSize[2]    = 1;

        shaderParam.globalWorkSize[0]   = gcmALIGN((1/*input_size[0]*/ + shaderParam.globalWorkScale[0] - 1)
            / shaderParam.globalWorkScale[0], shaderParam.localWorkSize[0]);
        shaderParam.globalWorkSize[1]   = gcmALIGN((1/*input_size[1]*/ + shaderParam.globalWorkScale[1] - 1)
            / shaderParam.globalWorkScale[1], shaderParam.localWorkSize[1]);
        shaderParam.globalWorkSize[2]   = input_size[2];
    }

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));
    if(status < 0)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
    }
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
        vx_uint32 uniInt16SumSqr_dp8x2[16] = {
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x76543210, 0x76543210, // ABin
            0x5555aaaa, // BSelt
            0x00000000, 0x76543210, // BBin
            0x00000300, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniFp16SumSqr_dp8x2[16] = {
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x76543210, 0x76543210, // ABin
            0x5555aaaa, // BSelt
            0x00000000, 0x76543210, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
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

        status |= vxSetNodeUniform(nodObj, "width", 1, &input_size[0]);
        status |= vxSetNodeUniform(nodObj, "height", 1, &input_size[1]);
        status |= vxSetNodeUniform(nodObj, "dimRatio", 1, &dimRatio);
        status |= vxSetNodeUniform(nodObj, "UniFP16toFP32Lo4_dp4x4", 1, UniFP16toFP32Lo4_dp4x4);
        status |= vxSetNodeUniform(nodObj, "uniConvertInt32toUint8_2x8", 1, uniConvertInt32toUint8_2x8);
        if(inputDataFormat == VSI_NN_TYPE_UINT8 || inputDataFormat == VSI_NN_TYPE_FLOAT16)
        {
            status  = vxSetNodeUniform(nodObj, "uniFp16SumSqr_dp8x2", 1, uniFp16SumSqr_dp8x2);
            status |= vxSetNodeUniform(nodObj, "uniConvertEndInt16Fp32_4x4", 1, uniConvertEndInt16Fp32_4x4);
        }

        if(inputDataFormat == VSI_NN_TYPE_UINT8)
        {
            status |= vxSetNodeUniform(nodObj, "uniConvert1stUint8SubZpToFp32_4x4", 1, uniConvert1stUint8SubZpToFp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvert2ndUint8SubZpToFp32_4x4", 1, uniConvert2ndUint8SubZpToFp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvert3rdUint8SubZpToFp32_4x4", 1, uniConvert3rdUint8SubZpToFp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvert4thUint8SubZpToFp32_4x4", 1, uniConvert4thUint8SubZpToFp32_4x4);
            status |= vxSetNodeUniform(nodObj, "inputZP", 1, &input_ZP);
            status |= vxSetNodeUniform(nodObj, "input_scale", 1, &scaleIn);
        }
        else if(inputDataFormat == VSI_NN_TYPE_INT8)
        {
            status |= vxSetNodeUniform(nodObj, "uniConvertDirInt8Fp32_4x4", 1, uniConvertDirUint8Fp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvertEndInt8Fp32_4x4", 1, uniConvertEndUint8Fp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvertTrdInt8Fp32_4x4", 1, uniConvertTrdUint8Fp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvertFthInt8Fp32_4x4", 1, uniConvertFthUint8Fp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniSumInt8_16x1", 1, uniSumInt8_16x1);
            status |= vxSetNodeUniform(nodObj, "uniSqrSumInt8_16x1", 1, uniSqrSumInt8_16x1);
            status |= vxSetNodeUniform(nodObj, "input_fl_scale", 1, &in_scale_fl);
            status |= vxSetNodeUniform(nodObj, "inFlScale_s2", 1, &inFlScale_s2);
        }
        else if(inputDataFormat == VSI_NN_TYPE_INT16)
        {
            status |= vxSetNodeUniform(nodObj, "uniInt16SumSqr_dp8x2", 1, uniInt16SumSqr_dp8x2);
            status |= vxSetNodeUniform(nodObj, "uniConvertInt16Fp32Fst_4x4",
                1, uniConvertInt16Fp32Fst_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvertInt16Fp32Secd_4x4",
                1, uniConvertInt16Fp32Secd_4x4);
            status |= vxSetNodeUniform(nodObj, "input_fl_scale", 1, &in_scale_fl);
            status |= vxSetNodeUniform(nodObj, "inFlScale_s2", 1, &inFlScale_s2);
        }

        if(outputDataFormat == VSI_NN_TYPE_UINT8)
        {
            status |= vxSetNodeUniform(nodObj, "output_ZP", 1, &output_ZP);
            status |= vxSetNodeUniform(nodObj, "outputScale", 1, &reScaleOut_u8);
        }
        else if(outputDataFormat == VSI_NN_TYPE_INT8)
        {
            status |= vxSetNodeUniform(nodObj, "output_fl_Scale", 1, &out_scale_fl);
        }
        else if(outputDataFormat == VSI_NN_TYPE_INT16)
        {
            status |= vxSetNodeUniform(nodObj, "uniConvertInt32toInt16_2x8",
                1, uniConvertInt32toInt16_2x8);
            status |= vxSetNodeUniform(nodObj, "output_fl_Scale", 1, &out_scale_fl);
        }

        if(outputDataFormat == VSI_NN_TYPE_UINT8 && inputDataFormat == VSI_NN_TYPE_UINT8)
        {
            scale_inOut = reScaleOut_u8 * scaleIn;
            status |= vxSetNodeUniform(nodObj, "scale_inOut", 1, &scale_inOut);
        }
        if(status < 0)
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
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED}
};

vsi_status VX_CALLBACK vxInstanceNormSumInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    uint32_t paraNum
    )
{
    vsi_status status = VX_SUCCESS;
    // Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
    vx_kernel_execution_parameters_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vx_tensor     input           = (vx_tensor)paramObj[0];
    uint32_t      input_size[DIM_SIZE]   = {1, 1, 1, 1};
    uint32_t      input_dims      = 0;
    int32_t input_ZP = 0;
    int32_t tmpZp1 = 0;
    int32_t segCnt = 0;
    vx_uint32 factor = 1;
    vx_uint32 maxWorkGroupSize = 8;
    vx_uint32  i        = 0;
    vsi_nn_tensor_attr_t attr;
    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    status  = vsi_nn_vxGetTensorAttr(input, &attr);
    if (status != VX_SUCCESS)
    {
        VSILOGE("vsi_nn_vxGetTensorAttr  failure! at line %d\n", __LINE__);
        return status;
    }

    input_dims       = attr.dim_num;
    for (i = 0; i < input_dims; i++)
    {
        input_size[i] = attr.size[i];
    }
    input_ZP     = attr.dtype.zero_point;

    tmpZp1 = (-2) * input_ZP;
    input_size[2] = (input_dims <= 2)?1:input_size[2];
    input_size[2] = (input_dims == 4)?(input_size[2] * input_size[3]):input_size[2];
    segCnt = (input_size[1] + 7) / 8;

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkOffset[2] = 0;
    shaderParam.globalWorkScale[0]  = 16;
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.globalWorkScale[2]  = 1;
    shaderParam.localWorkSize[0]    = 1;
    shaderParam.localWorkSize[1]    = 1;

    if (input_size[2] <= maxWorkGroupSize)
        shaderParam.localWorkSize[2]    = input_size[2];
    else if (getDataFactor(input_size[2], &factor, 2, maxWorkGroupSize, 8) == VX_SUCCESS)
        shaderParam.localWorkSize[2]    = factor;
    else
        shaderParam.localWorkSize[2]    = 1;

    shaderParam.globalWorkSize[0]   = gcmALIGN((input_size[0] + shaderParam.globalWorkScale[0] - 1)
        / shaderParam.globalWorkScale[0], shaderParam.localWorkSize[0]);
    shaderParam.globalWorkSize[1]   = gcmALIGN((input_size[1] + shaderParam.globalWorkScale[1] - 1)
        / shaderParam.globalWorkScale[1], shaderParam.localWorkSize[1]);
    shaderParam.globalWorkSize[2]   = input_size[2];

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));
    if(status < 0)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
    }
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
        status |= vxSetNodeUniform(nodObj, "tmpZp1", 1, &tmpZp1);
        status |= vxSetNodeUniform(nodObj, "segCnt", 1, &segCnt);
    }
    if(status < 0)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
    }
    return status;
}
static vx_param_description_t vxInstanceNormSumKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED}
};

vsi_status VX_CALLBACK vxInstanceNormSqrInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    uint32_t paraNum
    )
{
    vsi_status status = VX_SUCCESS;
    // Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
    vx_kernel_execution_parameters_t shaderParam = {
        2,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vx_tensor     input           = (vx_tensor)paramObj[0];
    uint32_t      input_size[DIM_SIZE]   = {1, 1, 1, 1};
    uint32_t      input_dims      = 0;
    vsi_nn_type_e inputDataFormat = VSI_NN_TYPE_FLOAT16;
    vx_float32 scaleIn = 0;
    int32_t input_ZP = 0;
    vx_uint32 iter = 0;
    vx_uint32 iter2 = 0;
    int32_t sumInZp = 0;
    int32_t segHeight = 0;
    //int32_t tmpZp1 = 0;
    vx_float32 tmpZp2 = 0;
    vx_float32 sumZpScale = 0;
    vx_float32 e2InScale = 0;
    vx_float32 dimRatio = 0;
    vx_uint32  i        = 0;
    vsi_nn_tensor_attr_t attr;
    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    status  = vsi_nn_vxGetTensorAttr(input, &attr);
    if (status != VX_SUCCESS)
    {
        VSILOGE("vsi_nn_vxGetTensorAttr  failure! at line %d\n", __LINE__);
        return status;
    }

    input_dims       = attr.dim_num;
    inputDataFormat  = attr.dtype.vx_type;
    for (i = 0; i < input_dims; i++)
    {
        input_size[i] = attr.size[i];
    }
    input_ZP     = attr.dtype.zero_point;
    scaleIn      = attr.dtype.scale;

    dimRatio = (vx_float32)(1.0 / (input_size[0] * input_size[1]));
    iter = ((input_size[0] + 15) / 16) * input_size[1] * 16;
    iter2 = ((input_size[0] + 15) / 16) * 8 * 16;
    sumInZp = input_ZP * iter * (-1);
    segHeight = (input_size[1] + 7) / 8;
    //tmpZp1 = (-2) * input_ZP;
    //tmpZp2 = input_ZP * input_ZP * iter;
    e2InScale = scaleIn * scaleIn;
    tmpZp2 = input_ZP * input_ZP * e2InScale;
    sumZpScale = tmpZp2 * iter2;
    input_size[2] = (input_dims <= 2)?1:input_size[2];
    input_size[2] = (input_dims == 4)?(input_size[2] * input_size[3]):input_size[2];

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkScale[0]  = 1;
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.localWorkSize[0]    = 1;
    shaderParam.localWorkSize[1]    = 1;
    shaderParam.globalWorkSize[0]   = gcmALIGN((input_size[2] + shaderParam.globalWorkScale[0] - 1)
        / shaderParam.globalWorkScale[0], shaderParam.localWorkSize[0]);
    shaderParam.globalWorkSize[1]   = gcmALIGN((1/*input_size[2]*/ + shaderParam.globalWorkScale[1] - 1)
        / shaderParam.globalWorkScale[1], shaderParam.localWorkSize[1]);

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));
    if(status < 0)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
    }
    if(inputDataFormat == VSI_NN_TYPE_UINT8)
    {
        status |= vxSetNodeUniform(nodObj, "sumInZp", 1, &sumInZp);
        status |= vxSetNodeUniform(nodObj, "segHeight", 1, &segHeight);
        status |= vxSetNodeUniform(nodObj, "dimRatio", 1, &dimRatio);
        status |= vxSetNodeUniform(nodObj, "input_scale", 1, &scaleIn);
        status |= vxSetNodeUniform(nodObj, "e2InScale", 1, &e2InScale);
        status |= vxSetNodeUniform(nodObj, "sumZpScale", 1, &sumZpScale);
        //status |= vxSetNodeUniform(nodObj, "pageNum", 1, &input_size[2]);
    }
    if(status < 0)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
    }
    return status;
}
static vx_param_description_t vxInstanceNormSqrKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED}
};

vsi_status VX_CALLBACK vxInstanceNormMeanVariInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    uint32_t paraNum
    )
{
    vsi_status status = VX_SUCCESS;
    // Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
    vx_kernel_execution_parameters_t shaderParam = {
        2,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vx_tensor     input           = (vx_tensor)paramObj[0];
    uint32_t      input_size[DIM_SIZE]   = {1, 1, 1, 1};
    uint32_t      input_dims      = 0;
    vsi_nn_type_e inputDataFormat = VSI_NN_TYPE_FLOAT16;
    vx_float32 scaleIn = 0;
    int32_t input_ZP = 0;
    vx_uint32 iter = 0;
    int32_t sumInZp = 0;
    //int32_t segHeight = 0;
    int32_t tmpZp1 = 0;
    vx_float32 tmpZp2 = 0;
    vx_float32 e2InScale = 0;
    vx_float32 dimRatio = 0;
    vx_float32 rowSumScale = 0;
    vx_uint32  i        = 0;
    vsi_nn_tensor_attr_t attr;
    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    status  = vsi_nn_vxGetTensorAttr(input, &attr);
    if (status != VX_SUCCESS)
    {
        VSILOGE("vsi_nn_vxGetTensorAttr  failure! at line %d\n", __LINE__);
        return status;
    }

    input_dims       = attr.dim_num;
    inputDataFormat  = attr.dtype.vx_type;
    for (i = 0; i < input_dims; i++)
    {
        input_size[i] = attr.size[i];
    }
    input_ZP     = attr.dtype.zero_point;
    scaleIn      = attr.dtype.scale;

    dimRatio = (vx_float32)(1.0 / (input_size[0] * input_size[1]));
    //iter = ((input_size[0] + 15) / 16) * input_size[1] * 16;
    iter = input_size[1] * 16;
    sumInZp = input_ZP * iter * (-1);
    //segHeight = (input_size[1] + 7) / 8;
    tmpZp1 = (-2) * input_ZP;
    e2InScale = scaleIn * scaleIn;
    tmpZp2 = input_ZP * input_ZP * e2InScale;
    //rowSumScale = ((input_size[1] + 15) / 16) * 16 * tmpZp2;
    rowSumScale = input_size[1] * 16 * tmpZp2;
    input_size[2] = (input_dims <= 2)?1:input_size[2];
    input_size[2] = (input_dims == 4)?(input_size[2] * input_size[3]):input_size[2];

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkScale[0]  = 16;
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.localWorkSize[0]    = 16;
    shaderParam.localWorkSize[1]    = 1;
    shaderParam.globalWorkSize[0]   = 16;
    shaderParam.globalWorkSize[1]   = input_size[2];

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));
    if(status < 0)
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
        status |= vxSetNodeUniform(nodObj, "width", 1, &input_size[0]);
        status |= vxSetNodeUniform(nodObj, "height", 1, &input_size[1]);
        status |= vxSetNodeUniform(nodObj, "sumInZp", 1, &sumInZp);
        status |= vxSetNodeUniform(nodObj, "tmpZp1", 1, &tmpZp1);
        status |= vxSetNodeUniform(nodObj, "dimRatio", 1, &dimRatio);
        status |= vxSetNodeUniform(nodObj, "input_scale", 1, &scaleIn);
        status |= vxSetNodeUniform(nodObj, "e2InScale", 1, &e2InScale);
        status |= vxSetNodeUniform(nodObj, "rowSumScale", 1, &rowSumScale);
        //status |= vxSetNodeUniform(nodObj, "pageNum", 1, &input_size[2]);
    }
    if(status < 0)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
    }
    return status;
}
static vx_param_description_t vxInstanceNormMeanVariKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
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

vx_kernel_description_t vxInstanceNormSumKernelInfo_U8 =
{
    VX_KERNEL_ENUM_INSTANCENORM,
    VX_KERNEL_NAME_INSTANCENORMSUM_UINT8,
    NULL,
    vxInstanceNormSumKernelParam,
    (sizeof(vxInstanceNormSumKernelParam) / sizeof(vxInstanceNormSumKernelParam[0])),
    vsi_nn_InstanceNormSumValidator,
    NULL,
    NULL,
    vxInstanceNormSumInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxInstanceNormSqrKernelInfo_U8 =
{
    VX_KERNEL_ENUM_INSTANCENORM,
    VX_KERNEL_NAME_INSTANCENORMSQR_UINT8,
    NULL,
    vxInstanceNormSqrKernelParam,
    (sizeof(vxInstanceNormSqrKernelParam) / sizeof(vxInstanceNormSqrKernelParam[0])),
    vsi_nn_InstanceNormSqrValidator,
    NULL,
    NULL,
    vxInstanceNormSqrInitializer,
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

vx_kernel_description_t vxInstanceNormKernelInfo_INT8 =
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

vx_kernel_description_t vxInstanceNormKernelInfoInt8_Fp16 =
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

vx_kernel_description_t vxInstanceNormKernelInfo_INT16 =
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

vx_kernel_description_t vxInstanceNormKernelInfoInt16_Fp16 =
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

vx_kernel_description_t * vx_kernel_INSTANCENORM_list[] =
{
    &vxInstanceNormKernelInfo_CPU,
    &vxInstanceNormKernelInfo,
    &vxInstanceNormKernelInfo_U8,
    &vxInstanceNormKernelInfoU8_Fp16,
    &vxInstanceNormSumKernelInfo_U8,
    &vxInstanceNormSqrKernelInfo_U8,
    &vxInstanceNormMeanVariKernelInfo_U8,
    &vxInstanceNormKernelInfo_INT8,
    &vxInstanceNormKernelInfoInt8_Fp16,
    &vxInstanceNormKernelInfo_INT16,
    &vxInstanceNormKernelInfoInt16_Fp16,
    NULL
};
#ifdef __cplusplus
}
#endif
