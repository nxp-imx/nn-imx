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
#define OUTPUT_FP16 0

void myFloorDivFunc
    (
    void* imgIn,
    void* imgIn1,
    void* imgOut,
    uint32_t input_dim,
    uint32_t width,
    uint32_t height,
    uint32_t channel,
    uint32_t batch,
    float inScale0,
    float inScale1,
    float outScale,
    int inZp0,
    int inZp1,
    int outZp,
    int8_t inFl0,
    int8_t inFl1,
    int8_t outFl,
    vsi_nn_type_e type
    )
{
    uint32_t k;
    uint32_t iter = batch * channel * height * width;

    if(type == VSI_NN_TYPE_FLOAT16)
    {
        uint16_t* tmpIn = (uint16_t*)imgIn;
        uint16_t* tmpIn1 = (uint16_t*)imgIn1;
        uint16_t* tmpOut = (uint16_t*)imgOut;
        float data0, data1, data2 = 0;

        for(k = 0; k < iter; k++)
        {
            data0 = vsi_nn_Fp16toFp32(tmpIn[k]);
            data1 = vsi_nn_Fp16toFp32(tmpIn1[k]);
            data2 = (float)floor(data0/data1);
            tmpOut[k] = vsi_nn_Fp32toFp16(data2);
        }
    }
    else if(type == VSI_NN_TYPE_INT16)
    {
        int16_t* tmpIn = (int16_t*)imgIn;
        int16_t* tmpIn1 = (int16_t*)imgIn1;
        int16_t* tmpOut = (int16_t*)imgOut;
        float data0, data1, data2 = 0;

        for(k = 0; k < iter; k++)
        {
            data0 = vsi_nn_DFPToFp32(tmpIn[k], inFl0, VSI_NN_TYPE_INT16);
            data1 = vsi_nn_DFPToFp32(tmpIn1[k], inFl1, VSI_NN_TYPE_INT16);
            data2 = (float)floor(data0/data1);
            tmpOut[k] = vsi_nn_Fp32ToDFP(data2, outFl, VSI_NN_TYPE_INT16);
        }
    }
    else if(type == VSI_NN_TYPE_INT8)
    {
        int8_t* tmpIn = (int8_t*)imgIn;
        int8_t* tmpIn1 = (int8_t*)imgIn1;
        int8_t* tmpOut = (int8_t*)imgOut;
        float data0, data1, data2 = 0;

        for(k = 0; k < iter; k++)
        {
            data0 = vsi_nn_DFPToFp32(tmpIn[k], inFl0, VSI_NN_TYPE_INT8);
            data1 = vsi_nn_DFPToFp32(tmpIn1[k], inFl1, VSI_NN_TYPE_INT8);
            data2 = (float)floor(data0/data1);
            tmpOut[k] = vsi_nn_Fp32ToDFP(data2, outFl, VSI_NN_TYPE_INT8);
        }
    }
    else if(type == VSI_NN_TYPE_UINT8)
    {
        uint8_t* tmpIn = (uint8_t*)imgIn;
        uint8_t* tmpIn1 = (uint8_t*)imgIn1;
        uint8_t* tmpOut = (uint8_t*)imgOut;
        float data0, data1, data2 = 0;

        for(k = 0; k < iter; k++)
        {
            data0 = vsi_nn_AffineToFp32(tmpIn[k], inScale0, inZp0, VSI_NN_TYPE_UINT8);
            data1 = vsi_nn_AffineToFp32(tmpIn1[k], inScale1, inZp1, VSI_NN_TYPE_UINT8);
            data2 = (float)floor(data0/data1);
            tmpOut[k] = vsi_nn_Fp32ToAffine(data2, outScale, outZp, VSI_NN_TYPE_UINT8);
        }
    }

    return;
}

vsi_status VX_CALLBACK vxFloorDivKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
    vsi_status status = VX_ERROR_INVALID_PARAMETERS;

    if(paramNum == 3)
    {
        vx_context context = NULL;
        // tensor
        vx_tensor imgObj[3] = { NULL };
#if INPUT_FP16
        int16_t *input = NULL;
#else
        uint8_t *input = NULL;
        uint8_t *input1 = NULL;
#endif
#if OUTPUT_FP16
        int16_t *output = NULL;
#else
        uint8_t *output = NULL;
#endif

        uint32_t input_size[DIM_SIZE] = {0}, output_size[DIM_SIZE] = {0};

        uint32_t input_stride_size[4]  = {0};
        uint32_t output_stride_size[4] = {0};

        vx_tensor_addressing input_user_addr = NULL;
        vx_tensor_addressing input1_user_addr = NULL;
        vx_tensor_addressing output_user_addr = NULL;

        vsi_nn_type_e inputFormat = VSI_NN_TYPE_FLOAT16,
            inputFormat1 = VSI_NN_TYPE_FLOAT16, outputFormat = VSI_NN_TYPE_FLOAT16;
        uint32_t input_dims = 0, output_dims = 0, tmpDim = 0;
        uint32_t i;
        vx_int8 infp0 = 0, infp1 = 0, outfp = 0;
        int32_t in_zp, in_zp1, out_zp;
        float in_scale, in_scale1, out_scale;

        status = VX_SUCCESS;

        imgObj[0] = (vx_tensor)paramObj[0];
        imgObj[1] = (vx_tensor)paramObj[1];
        imgObj[2] = (vx_tensor)paramObj[2];  //output
        context = vxGetContext((vx_reference)node);
        if (context == NULL)
        {
            VSILOGE("vxGetContext failure! at line %d\n", __LINE__);
            return status;
        }
        //input
        status = vxQueryTensor(imgObj[0], VX_TENSOR_NUM_OF_DIMS, &input_dims, sizeof(input_dims));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor input_dims failure! at line %d\n", __LINE__);
            return status;
        }
        status = vxQueryTensor(imgObj[0], VX_TENSOR_DATA_TYPE, &inputFormat, sizeof(inputFormat));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor inputFormat failure! at line %d\n", __LINE__);
            return status;
        }
        status = vxQueryTensor(imgObj[0], VX_TENSOR_DIMS, input_size, sizeof(input_size));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor input_size failure! at line %d\n", __LINE__);
            return status;
        }
        status = vxQueryTensor(imgObj[0], VX_TENSOR_ZERO_POINT, &in_zp, sizeof(in_zp));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor input_size failure! at line %d\n", __LINE__);
            return status;
        }
        status = vxQueryTensor(imgObj[0], VX_TENSOR_SCALE, &in_scale, sizeof(in_scale));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor input_size failure! at line %d\n", __LINE__);
            return status;
        }
        status = vxQueryTensor(imgObj[0], VX_TENSOR_FIXED_POINT_POSITION, &infp0, sizeof(infp0));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor input_size failure! at line %d\n", __LINE__);
            return status;
        }

        // input1
        status = vxQueryTensor(imgObj[1], VX_TENSOR_DATA_TYPE, &inputFormat1, sizeof(inputFormat1));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor inputFormat failure! at line %d\n", __LINE__);
            return status;
        }
        status = vxQueryTensor(imgObj[1], VX_TENSOR_ZERO_POINT, &in_zp1, sizeof(in_zp1));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor input_size failure! at line %d\n", __LINE__);
            return status;
        }
        status = vxQueryTensor(imgObj[1], VX_TENSOR_SCALE, &in_scale1, sizeof(in_scale1));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor input_size failure! at line %d\n", __LINE__);
            return status;
        }
        status = vxQueryTensor(imgObj[1], VX_TENSOR_FIXED_POINT_POSITION, &infp1, sizeof(infp1));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor input_size failure! at line %d\n", __LINE__);
            return status;
        }

        //output
        status  = vxQueryTensor(imgObj[2], VX_TENSOR_DATA_TYPE, &outputFormat, sizeof(outputFormat));
        status |= vxQueryTensor(imgObj[2], VX_TENSOR_NUM_OF_DIMS, &output_dims, sizeof(output_dims));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor outputFormat failure! at line %d\n", __LINE__);
            return status;
        }
        status = vxQueryTensor(imgObj[2], VX_TENSOR_DIMS, output_size, sizeof(output_size));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor output_size failure! at line %d\n", __LINE__);
            return status;
        }
        status = vxQueryTensor(imgObj[2], VX_TENSOR_ZERO_POINT, &out_zp, sizeof(out_zp));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor input_size failure! at line %d\n", __LINE__);
            return status;
        }
        status = vxQueryTensor(imgObj[2], VX_TENSOR_SCALE, &out_scale, sizeof(out_scale));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor input_size failure! at line %d\n", __LINE__);
            return status;
        }
        status = vxQueryTensor(imgObj[2], VX_TENSOR_FIXED_POINT_POSITION, &outfp, sizeof(outfp));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor input_size failure! at line %d\n", __LINE__);
            return status;
        }

        input_size[2] = (input_dims <= 2)?1:input_size[2];
        input_size[3] = (input_dims <= 3)?1:input_size[3];

        input_stride_size[0]  = vsi_nn_GetTypeBytes(inputFormat);
        output_stride_size[0] = vsi_nn_GetTypeBytes(outputFormat);
        //length_stride_size[0] = vsi_nn_GetTypeBytes(paraFormat);
        for (i=1; i< input_dims; i++)
        {
            input_stride_size[i]  = input_stride_size[i-1] * input_size[i-1];
        }
        for (i=1; i< output_dims; i++)
        {
            output_stride_size[i] = output_stride_size[i-1] * output_size[i-1];
        }

#if INPUT_FP16
        input  = (int16_t*)malloc(input_size[0]*input_size[1]*input_size[2]*sizeof(int16_t));
#else
        input  = (uint8_t*)malloc(input_size[0]*input_size[1]*input_size[2]*vsi_nn_GetTypeBytes(inputFormat));
        input1  = (uint8_t*)malloc(input_size[0]*input_size[1]*input_size[2]*vsi_nn_GetTypeBytes(inputFormat));
#endif
#if OUTPUT_FP16
        output = (int16_t*)malloc(output_size[0]*output_size[1]*output_size[2]*sizeof(int16_t));
#else
        output = (uint8_t*)malloc(output_size[0]*output_size[1]*output_size[2]*vsi_nn_GetTypeBytes(outputFormat));
#endif

        input_user_addr = vxCreateTensorAddressing(context, input_size, input_stride_size, input_dims);
        vxCopyTensorPatch(imgObj[0], NULL, input_user_addr, input, VX_READ_ONLY, 0);

        input1_user_addr = vxCreateTensorAddressing(context, input_size, input_stride_size, input_dims);
        vxCopyTensorPatch(imgObj[1], NULL, input1_user_addr, input1, VX_READ_ONLY, 0);

        // Call C Prototype
        myFloorDivFunc(input, input1, output, tmpDim, input_size[0],
            input_size[1], input_size[2], input_size[3],
            in_scale, in_scale1, out_scale, in_zp, in_zp1, out_zp, infp0, infp1, outfp, inputFormat);

        //output tensor
        output_user_addr = vxCreateTensorAddressing(context, output_size,
            output_stride_size, output_dims);
        vxCopyTensorPatch(imgObj[2], NULL, output_user_addr, output, VX_WRITE_ONLY, 0);

        if(input) free(input);
        if(input1) free(input1);
        if(output) free(output);

        if(input_user_addr) vxReleaseTensorAddressing(&input_user_addr);
        if(input1_user_addr) vxReleaseTensorAddressing(&input1_user_addr);
        if(output_user_addr) vxReleaseTensorAddressing(&output_user_addr);
    }

    return status;
}

vsi_status VX_CALLBACK vxFloorDivInitializer
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

    vx_tensor     input0          = (vx_tensor)paramObj[0];
    vx_tensor     input1          = (vx_tensor)paramObj[1];
    vx_tensor     output          = (vx_tensor)paramObj[2];

    uint32_t      input_size[DIM_SIZE]   = {0};
    uint32_t      input_dims      = 0;
    uint32_t      output_dims     = 0;
    uint32_t      zAx             = 1;
    vsi_nn_type_e inputDataFormat = VSI_NN_TYPE_FLOAT16;
    vsi_nn_type_e inputDataFormat1 = VSI_NN_TYPE_FLOAT16;
    vsi_nn_type_e outputDataFormat = VSI_NN_TYPE_FLOAT16;
    int8_t      input_fixPointPos0      = 0;
    int8_t      input_fixPointPos1      = 0;
    int8_t      output_fixPointPos      = 0;
    vx_float32  u8InScale0 = 0;
    vx_float32  u8InScale1 = 0;
    vx_float32  u8OutScale = 0;
    vx_float32  inScale0 = 0;
    vx_float32  inScale1 = 0;
    vx_float32  outScale = 0;
    int32_t     inZp0 = 0;
    int32_t     inZp1 = 0;
    int32_t     outZp = 0;
    //vx_uint32 factor = 1;
    //vx_uint32 maxWorkGroupSize = 8;

    status  = vxQueryTensor(input0, VX_TENSOR_DIMS, input_size, sizeof(input_size));
    status |= vxQueryTensor(input0, VX_TENSOR_NUM_OF_DIMS, &input_dims, sizeof(input_dims));
    status |= vxQueryTensor(input0, VX_TENSOR_DATA_TYPE, &inputDataFormat, sizeof(inputDataFormat));
    status |= vxQueryTensor(input0, VX_TENSOR_FIXED_POINT_POS, &input_fixPointPos0, sizeof(input_fixPointPos0));
    status |= vxQueryTensor(input0, VX_TENSOR_ZERO_POINT, &inZp0, sizeof(inZp0));
    status |= vxQueryTensor(input0, VX_TENSOR_SCALE, &u8InScale0, sizeof(u8InScale0));
    status |= vxQueryTensor(input1, VX_TENSOR_DATA_TYPE, &inputDataFormat1, sizeof(inputDataFormat1));
    status |= vxQueryTensor(input1, VX_TENSOR_FIXED_POINT_POS, &input_fixPointPos1, sizeof(input_fixPointPos1));
    status |= vxQueryTensor(input1, VX_TENSOR_ZERO_POINT, &inZp1, sizeof(inZp1));
    status |= vxQueryTensor(input1, VX_TENSOR_SCALE, &u8InScale1, sizeof(u8InScale1));
    status |= vxQueryTensor(output, VX_TENSOR_NUM_OF_DIMS, &output_dims, sizeof(output_dims));
    status |= vxQueryTensor(output, VX_TENSOR_DATA_TYPE, &outputDataFormat, sizeof(outputDataFormat));
    status |= vxQueryTensor(output, VX_TENSOR_FIXED_POINT_POS, &output_fixPointPos, sizeof(output_fixPointPos));
    status |= vxQueryTensor(output, VX_TENSOR_ZERO_POINT, &outZp, sizeof(outZp));
    status |= vxQueryTensor(output, VX_TENSOR_SCALE, &u8OutScale, sizeof(u8OutScale));
    if(VX_SUCCESS != status)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
        return status;
    }

    if(inputDataFormat == VSI_NN_TYPE_INT16 || inputDataFormat == VSI_NN_TYPE_INT8)
    {
        if (input_fixPointPos0 >= 0)
        {
            inScale0 = 1.0f / (vx_float32) (1 << input_fixPointPos0);
        }
        else if (input_fixPointPos0 < 0)
        {
            inScale0 = (vx_float32) (1 << -input_fixPointPos0);
        }
    }
    else if(inputDataFormat == VSI_NN_TYPE_UINT8)
    {
        inScale0 = u8InScale0;
    }

    if(inputDataFormat1 == VSI_NN_TYPE_INT16 || inputDataFormat1 == VSI_NN_TYPE_INT8)
    {
        if (input_fixPointPos1 >= 0)
        {
            inScale1 = 1.0f / (vx_float32) (1 << input_fixPointPos1);
        }
        else if (input_fixPointPos1 < 0)
        {
            inScale1 = (vx_float32) (1 << -input_fixPointPos1);
        }
    }
    else if(inputDataFormat1 == VSI_NN_TYPE_UINT8)
    {
        inScale1 = u8InScale1;
    }

    if(outputDataFormat == VSI_NN_TYPE_INT16 || outputDataFormat == VSI_NN_TYPE_INT8)
    {
        if (output_fixPointPos >= 0)
        {
            outScale = (vx_float32) (1 << output_fixPointPos);
        }
        else if (output_fixPointPos < 0)
        {
            outScale = 1.0f / (vx_float32) (1 << -output_fixPointPos);
        }
    }
    else if(outputDataFormat == VSI_NN_TYPE_UINT8)
    {
        outScale = 1 / u8OutScale;
    }

    if(input_dims == 4)
        zAx = input_size[3] * input_size[2];
    else if(input_dims == 3)
        zAx = input_size[2];

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkOffset[2] = 0;
    shaderParam.globalWorkScale[0]  = 8;
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.globalWorkScale[2]  = 1;
    shaderParam.localWorkSize[0]    = 4;
    shaderParam.localWorkSize[1]    = 2;
    shaderParam.localWorkSize[2]    = 1;
    shaderParam.globalWorkSize[0]   = gcmALIGN((input_size[0] + shaderParam.globalWorkScale[0] - 1)
        / shaderParam.globalWorkScale[0], shaderParam.localWorkSize[0]);
    shaderParam.globalWorkSize[1]   = gcmALIGN((input_size[1] + shaderParam.globalWorkScale[1] - 1)
        / shaderParam.globalWorkScale[1], shaderParam.localWorkSize[1]);
    shaderParam.globalWorkSize[2]   = gcmALIGN((zAx + shaderParam.globalWorkScale[2] - 1)
        / shaderParam.globalWorkScale[2], shaderParam.localWorkSize[2]);

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));
    if(status < 0)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
    }
#if 1
    {
        vx_uint32 uniConvertUint8SubZpToFp32_4x4[16] = {
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002100, // AccumType, ConstantType, and PostShift
            0xbc003c00, 0x00000000, 0xbc003c00, 0x00000000, 0xbc003c00, 0x00000000, 0xbc003c00, 0x00000000 // Constant
        };

        vx_uint32 uniConvertSecUint8SubZpToFp32_4x4[16] = {
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00050004, 0x00070006, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0xbc003c00, 0x00000000, 0xbc003c00, 0x00000000, 0xbc003c00, 0x00000000, 0xbc003c00, 0x00000000 // Constant
        };

        vx_uint32 uniConvertDirInt16Fp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };
        vx_uint32 uniConvertEndInt16Fp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };
        vx_uint32 uniConvertInt32toUint8_2x8[16] = {
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };

        vx_uint32 uniConvertFstFp16Fp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };
        vx_uint32 uniConvertSecFp16Fp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };

        vx_uint32 uniConvertInt8FstFp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        };
        vx_uint32 uniConvertInt8SecFp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        };

        status |= vxSetNodeUniform(nodObj, "uniConvertInt8FstFp32_4x4", 1, uniConvertInt8FstFp32_4x4);
        status |= vxSetNodeUniform(nodObj, "uniConvertInt8SecFp32_4x4", 1, uniConvertInt8SecFp32_4x4);

        status |= vxSetNodeUniform(nodObj, "uniConvertFstFp16Fp32_4x4", 1, uniConvertFstFp16Fp32_4x4);
        status |= vxSetNodeUniform(nodObj, "uniConvertSecFp16Fp32_4x4", 1, uniConvertSecFp16Fp32_4x4);

        status |= vxSetNodeUniform(nodObj, "uniConvertUint8SubZpToFp32_4x4", 1, uniConvertUint8SubZpToFp32_4x4);
        status |= vxSetNodeUniform(nodObj, "uniConvertSecUint8SubZpToFp32_4x4", 1, uniConvertSecUint8SubZpToFp32_4x4);

        status |= vxSetNodeUniform(nodObj, "uniConvertDirInt16Fp32_4x4", 1, uniConvertDirInt16Fp32_4x4);
        status |= vxSetNodeUniform(nodObj, "uniConvertEndInt16Fp32_4x4", 1, uniConvertEndInt16Fp32_4x4);
        status |= vxSetNodeUniform(nodObj, "uniConvertInt32toUint8_2x8", 1, uniConvertInt32toUint8_2x8);
        status |= vxSetNodeUniform(nodObj, "in_scale0", 1, &inScale0);
        status |= vxSetNodeUniform(nodObj, "in_scale1", 1, &inScale1);
        status |= vxSetNodeUniform(nodObj, "out_scale", 1, &outScale);
        status |= vxSetNodeUniform(nodObj, "in_zp0", 1, &inZp0);
        status |= vxSetNodeUniform(nodObj, "in_zp1", 1, &inZp1);
        status |= vxSetNodeUniform(nodObj, "out_zp", 1, &outZp);
        if(status < 0)
        {
            VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
        }
    }
#endif
    return status;
}
static vx_param_description_t vxFloorDivKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED}
};
#ifdef __cplusplus
extern "C" {
#endif
vx_kernel_description_t vxFloorDivKernelInfo_fp16_fp16 =
{
    VX_KERNEL_ENUM_FLOORDIV,
    VX_KERNEL_NAME_FLOORDIV_FP16,
    NULL,
    vxFloorDivKernelParam,
    (sizeof(vxFloorDivKernelParam) / sizeof(vxFloorDivKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxFloorDivInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxFloorDivKernelInfo_int16_int16 =
{
    VX_KERNEL_ENUM_FLOORDIV,
    VX_KERNEL_NAME_FLOORDIV_INT16,
    NULL,
    vxFloorDivKernelParam,
    (sizeof(vxFloorDivKernelParam) / sizeof(vxFloorDivKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxFloorDivInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxFloorDivKernelInfo_int8_int8 =
{
    VX_KERNEL_ENUM_FLOORDIV,
    VX_KERNEL_NAME_FLOORDIV_INT8,
    NULL,
    vxFloorDivKernelParam,
    (sizeof(vxFloorDivKernelParam) / sizeof(vxFloorDivKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxFloorDivInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxFloorDivKernelInfo_uint8_uint8 =
{
    VX_KERNEL_ENUM_FLOORDIV,
    VX_KERNEL_NAME_FLOORDIV_UINT8,
    NULL,
    vxFloorDivKernelParam,
    (sizeof(vxFloorDivKernelParam) / sizeof(vxFloorDivKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxFloorDivInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxFloorDivKernelInfo_CPU =
{
    VX_KERNEL_ENUM_FLOORDIV,
    VX_KERNEL_NAME_FLOORDIV_FP16,
    vxFloorDivKernel,
    vxFloorDivKernelParam,
    (sizeof(vxFloorDivKernelParam) / sizeof(vxFloorDivKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_FLOORDIV_list[] =
{
    &vxFloorDivKernelInfo_CPU,
    &vxFloorDivKernelInfo_fp16_fp16,
    &vxFloorDivKernelInfo_int16_int16,
    &vxFloorDivKernelInfo_int8_int8,
    &vxFloorDivKernelInfo_uint8_uint8,
    NULL
};
#ifdef __cplusplus
}
#endif
