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

void myRelationalOpsFunc
    (
    void* imgIn,
    void* imgIn1,
    void* imgOut,
    uint32_t input_dim,
    uint32_t width,
    uint32_t height,
    uint32_t channel,
    uint32_t batch,
    uint32_t cmp_op,
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
            if(data0 > data1 && cmp_op == 0)
                data2 = 1;
            else if(data0 >= data1 && cmp_op == 1)
                data2 = 1;
            else if(data0 < data1 && cmp_op == 2)
                data2 = 1;
            else if(data0 <= data1 && cmp_op == 3)
                data2 = 1;
            else if(data0 != data1 && cmp_op == 4)
                data2 = 1;
            else if(data0 == data1 && cmp_op == 5)
                data2 = 1;
            tmpOut[k] = vsi_nn_Fp32toFp16(data2);
            data2 = 0;
        }
    }
    else if(type == VSI_NN_TYPE_INT16)
    {
        int16_t* tmpIn = (int16_t*)imgIn;
        int16_t* tmpIn1 = (int16_t*)imgIn1;
        int16_t* tmpOut = (int16_t*)imgOut;
        int16_t data0, data1, data2 = 0;

        for(k = 0; k < iter; k++)
        {
            data0 = tmpIn[k];
            data1 = tmpIn1[k];
            if(data0 > data1 && cmp_op == 0)
                data2 = 1;
            else if(data0 >= data1 && cmp_op == 1)
                data2 = 1;
            else if(data0 < data1 && cmp_op == 2)
                data2 = 1;
            else if(data0 <= data1 && cmp_op == 3)
                data2 = 1;
            else if(data0 != data1 && cmp_op == 4)
                data2 = 1;
            else if(data0 == data1 && cmp_op == 5)
                data2 = 1;
            tmpOut[k] = data2;
            data2 = 0;
        }
    }
    else if(type == VSI_NN_TYPE_INT8)
    {
        int8_t* tmpIn = (int8_t*)imgIn;
        int8_t* tmpIn1 = (int8_t*)imgIn1;
        int8_t* tmpOut = (int8_t*)imgOut;
        int8_t data0, data1, data2 = 0;

        for(k = 0; k < iter; k++)
        {
            data0 = tmpIn[k];
            data1 = tmpIn1[k];
            if(data0 > data1 && cmp_op == 0)
                data2 = 1;
            else if(data0 >= data1 && cmp_op == 1)
                data2 = 1;
            else if(data0 < data1 && cmp_op == 2)
                data2 = 1;
            else if(data0 <= data1 && cmp_op == 3)
                data2 = 1;
            else if(data0 != data1 && cmp_op == 4)
                data2 = 1;
            else if(data0 == data1 && cmp_op == 5)
                data2 = 1;
            tmpOut[k] = data2;
            data2 = 0;
        }
    }
    else if(type == VSI_NN_TYPE_UINT8)
    {
        uint8_t* tmpIn = (uint8_t*)imgIn;
        uint8_t* tmpIn1 = (uint8_t*)imgIn1;
        uint8_t* tmpOut = (uint8_t*)imgOut;
        uint8_t data0, data1, data2 = 0;

        for(k = 0; k < iter; k++)
        {
            data0 = tmpIn[k];
            data1 = tmpIn1[k];
            if(data0 > data1 && cmp_op == 0)
                data2 = 1;
            else if(data0 >= data1 && cmp_op == 1)
                data2 = 1;
            else if(data0 < data1 && cmp_op == 2)
                data2 = 1;
            else if(data0 <= data1 && cmp_op == 3)
                data2 = 1;
            else if(data0 != data1 && cmp_op == 4)
                data2 = 1;
            else if(data0 == data1 && cmp_op == 5)
                data2 = 1;
            tmpOut[k] = data2;
            data2 = 0;
        }
    }

    return;
}

vsi_status VX_CALLBACK vxRelationalOpsKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
    vsi_status status = VX_ERROR_INVALID_PARAMETERS;

    if(paramNum == 4)
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
        vsi_nn_tensor_attr_t in_attr, in_attr1, out_attr;

        vsi_nn_type_e inputFormat = VSI_NN_TYPE_FLOAT16, outputFormat = VSI_NN_TYPE_FLOAT16;
        uint32_t input_dims = 0, output_dims = 0, tmpDim = 0;
        int32_t in_zp = 0, out_zp = 0;
        float in_scale = 1, out_scale = 1;
        uint32_t cmp_op = 0;
        vx_scalar scalar[1] = { NULL };

        status = VX_SUCCESS;

        memset(&in_attr, 0x0, sizeof(vsi_nn_tensor_attr_t));
        memset(&in_attr1, 0x0, sizeof(vsi_nn_tensor_attr_t));
        memset(&out_attr, 0x0, sizeof(vsi_nn_tensor_attr_t));

        imgObj[0] = (vx_tensor)paramObj[0];
        imgObj[1] = (vx_tensor)paramObj[1];  //output
        imgObj[2] = (vx_tensor)paramObj[2];
        scalar[0] = (vx_scalar)paramObj[3];
        context = vxGetContext((vx_reference)node);
        if (context == NULL)
        {
            VSILOGE("vxGetContext failure! at line %d\n", __LINE__);
            goto OnError;
        }
        //input
        status = vxQueryTensor(imgObj[0], VX_TENSOR_NUM_OF_DIMS, &input_dims, sizeof(input_dims));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor input_dims failure! at line %d\n", __LINE__);
            goto OnError;
        }
        status = vxQueryTensor(imgObj[0], VX_TENSOR_DATA_TYPE, &inputFormat, sizeof(inputFormat));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor inputFormat failure! at line %d\n", __LINE__);
            goto OnError;
        }
        status = vxQueryTensor(imgObj[0], VX_TENSOR_DIMS, input_size, sizeof(input_size));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor input_size failure! at line %d\n", __LINE__);
            goto OnError;
        }
        status = vxQueryTensor(imgObj[0], VX_TENSOR_ZERO_POINT, &in_zp, sizeof(in_zp));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor input_size failure! at line %d\n", __LINE__);
            goto OnError;
        }
        status = vxQueryTensor(imgObj[0], VX_TENSOR_SCALE, &in_scale, sizeof(in_scale));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor input_size failure! at line %d\n", __LINE__);
            goto OnError;
        }
        //output
        status  = vxQueryTensor(imgObj[2], VX_TENSOR_DATA_TYPE, &outputFormat, sizeof(outputFormat));
        status |= vxQueryTensor(imgObj[2], VX_TENSOR_NUM_OF_DIMS, &output_dims, sizeof(output_dims));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor outputFormat failure! at line %d\n", __LINE__);
            goto OnError;
        }
        status = vxQueryTensor(imgObj[2], VX_TENSOR_DIMS, output_size, sizeof(output_size));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor output_size failure! at line %d\n", __LINE__);
            goto OnError;
        }
        status = vxQueryTensor(imgObj[2], VX_TENSOR_ZERO_POINT, &out_zp, sizeof(out_zp));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor input_size failure! at line %d\n", __LINE__);
            goto OnError;
        }
        status = vxQueryTensor(imgObj[2], VX_TENSOR_SCALE, &out_scale, sizeof(out_scale));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor input_size failure! at line %d\n", __LINE__);
            goto OnError;
        }

        input_size[2] = (input_dims <= 2)?1:input_size[2];
        input_size[3] = (input_dims <= 3)?1:input_size[3];

#if 0
        for (i=1; i< length_dims; i++)
        {
            length_stride_size[i] = length_stride_size[i-1] * length_stride_size[i-1];
        }
#endif

#if INPUT_FP16
        input  = (int16_t*)malloc(input_size[0]*input_size[1]*input_size[2]*sizeof(int16_t));
#else
        //input  = (uint8_t*)malloc(input_size[0]*input_size[1]*input_size[2]*vsi_nn_GetTypeBytes(inputFormat));
        //input1  = (uint8_t*)malloc(input_size[0]*input_size[1]*input_size[2]*vsi_nn_GetTypeBytes(inputFormat));
#endif
#if OUTPUT_FP16
        output = (int16_t*)malloc(output_size[0]*output_size[1]*output_size[2]*sizeof(int16_t));
#else
        output = (uint8_t*)malloc(output_size[0]*output_size[1]*output_size[2]*vsi_nn_GetTypeBytes(outputFormat));
#endif
        status = vsi_nn_vxGetTensorAttr(imgObj[0], &in_attr);
        status |= vsi_nn_vxGetTensorAttr(imgObj[1], &in_attr1);
        status |= vsi_nn_vxGetTensorAttr(imgObj[2], &out_attr);
        if (status != VX_SUCCESS)
        {
            VSILOGE("vsi_nn_vxGetTensorAttr failure! at line %d\n", __LINE__);
            goto OnError;
        }
        input = vsi_nn_vxCopyTensorToData(context, imgObj[0], &in_attr);
        input1 = vsi_nn_vxCopyTensorToData(context, imgObj[1], &in_attr1);

        // scalar
        status = vxCopyScalar(scalar[0], &cmp_op, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxCopyScalar failure! at line %d\n", __LINE__);
            goto OnError;
        }

        // Call C Prototype
        myRelationalOpsFunc(input, input1, output, tmpDim, input_size[0],
            input_size[1], input_size[2], input_size[3], cmp_op, inputFormat);

        //output tensor
        status = vsi_nn_vxCopyDataToTensor(context, imgObj[2], &out_attr, output);
        if (status != VX_SUCCESS)
        {
            VSILOGE("vsi_nn_vxCopyDataToTensor failure! at line %d\n", __LINE__);
            goto OnError;
        }
OnError:
        if(input) free(input);
        if(input1) free(input1);
        if(output) free(output);
    }

    return status;
}

vsi_status VX_CALLBACK vxRelationalOpsInitializer
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
    vx_tensor     output          = (vx_tensor)paramObj[2];

    uint32_t      input_size[DIM_SIZE]   = {0};
    uint32_t      input_dims      = 0;
    uint32_t      output_dims     = 0;
    uint32_t      zAx             = 1;
    vsi_nn_type_e inputDataFormat = VSI_NN_TYPE_FLOAT16;
    vsi_nn_type_e outputDataFormat = VSI_NN_TYPE_FLOAT16;
    //vx_uint32 factor = 1;
    //vx_uint32 maxWorkGroupSize = 8;

    status  = vxQueryTensor(input0, VX_TENSOR_DIMS, input_size, sizeof(input_size));
    status |= vxQueryTensor(input0, VX_TENSOR_NUM_OF_DIMS, &input_dims, sizeof(input_dims));
    status |= vxQueryTensor(input0, VX_TENSOR_DATA_TYPE, &inputDataFormat, sizeof(inputDataFormat));
    status |= vxQueryTensor(output, VX_TENSOR_NUM_OF_DIMS, &output_dims, sizeof(output_dims));
    status |= vxQueryTensor(output, VX_TENSOR_DATA_TYPE, &outputDataFormat, sizeof(outputDataFormat));
    if(VX_SUCCESS != status)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
        return status;
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

    if(inputDataFormat == VSI_NN_TYPE_FLOAT16)
    {
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
        vx_uint32 uniMulShortMinus1toFp16_2x8[16] = {
            0x22222222, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
        };
        status |= vxSetNodeUniform(nodObj, "uniConvertFstFp16Fp32_4x4", 1, uniConvertFstFp16Fp32_4x4);
        status |= vxSetNodeUniform(nodObj, "uniConvertSecFp16Fp32_4x4", 1, uniConvertSecFp16Fp32_4x4);
        status |= vxSetNodeUniform(nodObj, "uniConvertInt32toUint8_2x8", 1, uniConvertInt32toUint8_2x8);
        status |= vxSetNodeUniform(nodObj, "uniMulShortMinus1toFp16_2x8", 1, uniMulShortMinus1toFp16_2x8);
    }
    if(status < 0)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
    }
    return status;
}
static vx_param_description_t vxRelationalOpsKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED}
};
static vx_param_description_t vxRelationalOpsCpuKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED}
};
#ifdef __cplusplus
extern "C" {
#endif
vx_kernel_description_t vxRelationalKernelInfo_gt_fp16 =
{
    VX_KERNEL_ENUM_RELATIONALOPS,
    VX_KERNEL_NAME_RELATIONAL_GREATER_FP16,
    NULL,
    vxRelationalOpsKernelParam,
    (sizeof(vxRelationalOpsKernelParam) / sizeof(vxRelationalOpsKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxRelationalOpsInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxRelationalKernelInfo_gte_fp16 =
{
    VX_KERNEL_ENUM_RELATIONALOPS,
    VX_KERNEL_NAME_RELATIONAL_GREATEREQUAL_FP16,
    NULL,
    vxRelationalOpsKernelParam,
    (sizeof(vxRelationalOpsKernelParam) / sizeof(vxRelationalOpsKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxRelationalOpsInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxRelationalKernelInfo_ls_fp16 =
{
    VX_KERNEL_ENUM_RELATIONALOPS,
    VX_KERNEL_NAME_RELATIONAL_LESS_FP16,
    NULL,
    vxRelationalOpsKernelParam,
    (sizeof(vxRelationalOpsKernelParam) / sizeof(vxRelationalOpsKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxRelationalOpsInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxRelationalKernelInfo_lse_fp16 =
{
    VX_KERNEL_ENUM_RELATIONALOPS,
    VX_KERNEL_NAME_RELATIONAL_LESSEQUAL_FP16,
    NULL,
    vxRelationalOpsKernelParam,
    (sizeof(vxRelationalOpsKernelParam) / sizeof(vxRelationalOpsKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxRelationalOpsInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxRelationalKernelInfo_ne_fp16 =
{
    VX_KERNEL_ENUM_RELATIONALOPS,
    VX_KERNEL_NAME_RELATIONAL_NOTEQUAL_FP16,
    NULL,
    vxRelationalOpsKernelParam,
    (sizeof(vxRelationalOpsKernelParam) / sizeof(vxRelationalOpsKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxRelationalOpsInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxRelationalKernelInfo_e_fp16 =
{
    VX_KERNEL_ENUM_RELATIONALOPS,
    VX_KERNEL_NAME_RELATIONAL_EQUAL_FP16,
    NULL,
    vxRelationalOpsKernelParam,
    (sizeof(vxRelationalOpsKernelParam) / sizeof(vxRelationalOpsKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxRelationalOpsInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxRelationalKernelInfo_gt_int16 =
{
    VX_KERNEL_ENUM_RELATIONALOPS,
    VX_KERNEL_NAME_RELATIONAL_GREATER_INT16,
    NULL,
    vxRelationalOpsKernelParam,
    (sizeof(vxRelationalOpsKernelParam) / sizeof(vxRelationalOpsKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxRelationalOpsInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxRelationalKernelInfo_gte_int16 =
{
    VX_KERNEL_ENUM_RELATIONALOPS,
    VX_KERNEL_NAME_RELATIONAL_GREATEREQUAL_INT16,
    NULL,
    vxRelationalOpsKernelParam,
    (sizeof(vxRelationalOpsKernelParam) / sizeof(vxRelationalOpsKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxRelationalOpsInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxRelationalKernelInfo_ls_int16 =
{
    VX_KERNEL_ENUM_RELATIONALOPS,
    VX_KERNEL_NAME_RELATIONAL_LESS_INT16,
    NULL,
    vxRelationalOpsKernelParam,
    (sizeof(vxRelationalOpsKernelParam) / sizeof(vxRelationalOpsKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxRelationalOpsInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxRelationalKernelInfo_lse_int16 =
{
    VX_KERNEL_ENUM_RELATIONALOPS,
    VX_KERNEL_NAME_RELATIONAL_LESSEQUAL_INT16,
    NULL,
    vxRelationalOpsKernelParam,
    (sizeof(vxRelationalOpsKernelParam) / sizeof(vxRelationalOpsKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxRelationalOpsInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxRelationalKernelInfo_ne_int16 =
{
    VX_KERNEL_ENUM_RELATIONALOPS,
    VX_KERNEL_NAME_RELATIONAL_NOTEQUAL_INT16,
    NULL,
    vxRelationalOpsKernelParam,
    (sizeof(vxRelationalOpsKernelParam) / sizeof(vxRelationalOpsKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxRelationalOpsInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxRelationalKernelInfo_e_int16 =
{
    VX_KERNEL_ENUM_RELATIONALOPS,
    VX_KERNEL_NAME_RELATIONAL_EQUAL_INT16,
    NULL,
    vxRelationalOpsKernelParam,
    (sizeof(vxRelationalOpsKernelParam) / sizeof(vxRelationalOpsKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxRelationalOpsInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxRelationalKernelInfo_gt_int8 =
{
    VX_KERNEL_ENUM_RELATIONALOPS,
    VX_KERNEL_NAME_RELATIONAL_GREATER_INT8,
    NULL,
    vxRelationalOpsKernelParam,
    (sizeof(vxRelationalOpsKernelParam) / sizeof(vxRelationalOpsKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxRelationalOpsInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxRelationalKernelInfo_gte_int8 =
{
    VX_KERNEL_ENUM_RELATIONALOPS,
    VX_KERNEL_NAME_RELATIONAL_GREATEREQUAL_INT8,
    NULL,
    vxRelationalOpsKernelParam,
    (sizeof(vxRelationalOpsKernelParam) / sizeof(vxRelationalOpsKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxRelationalOpsInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxRelationalKernelInfo_ls_int8 =
{
    VX_KERNEL_ENUM_RELATIONALOPS,
    VX_KERNEL_NAME_RELATIONAL_LESS_INT8,
    NULL,
    vxRelationalOpsKernelParam,
    (sizeof(vxRelationalOpsKernelParam) / sizeof(vxRelationalOpsKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxRelationalOpsInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxRelationalKernelInfo_lse_int8 =
{
    VX_KERNEL_ENUM_RELATIONALOPS,
    VX_KERNEL_NAME_RELATIONAL_LESSEQUAL_INT8,
    NULL,
    vxRelationalOpsKernelParam,
    (sizeof(vxRelationalOpsKernelParam) / sizeof(vxRelationalOpsKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxRelationalOpsInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxRelationalKernelInfo_ne_int8 =
{
    VX_KERNEL_ENUM_RELATIONALOPS,
    VX_KERNEL_NAME_RELATIONAL_NOTEQUAL_INT8,
    NULL,
    vxRelationalOpsKernelParam,
    (sizeof(vxRelationalOpsKernelParam) / sizeof(vxRelationalOpsKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxRelationalOpsInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxRelationalKernelInfo_e_int8 =
{
    VX_KERNEL_ENUM_RELATIONALOPS,
    VX_KERNEL_NAME_RELATIONAL_EQUAL_INT8,
    NULL,
    vxRelationalOpsKernelParam,
    (sizeof(vxRelationalOpsKernelParam) / sizeof(vxRelationalOpsKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxRelationalOpsInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxRelationalKernelInfo_gt_uint8 =
{
    VX_KERNEL_ENUM_RELATIONALOPS,
    VX_KERNEL_NAME_RELATIONAL_GREATER_UINT8,
    NULL,
    vxRelationalOpsKernelParam,
    (sizeof(vxRelationalOpsKernelParam) / sizeof(vxRelationalOpsKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxRelationalOpsInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxRelationalKernelInfo_gte_uint8 =
{
    VX_KERNEL_ENUM_RELATIONALOPS,
    VX_KERNEL_NAME_RELATIONAL_GREATEREQUAL_UINT8,
    NULL,
    vxRelationalOpsKernelParam,
    (sizeof(vxRelationalOpsKernelParam) / sizeof(vxRelationalOpsKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxRelationalOpsInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxRelationalKernelInfo_ls_uint8 =
{
    VX_KERNEL_ENUM_RELATIONALOPS,
    VX_KERNEL_NAME_RELATIONAL_LESS_UINT8,
    NULL,
    vxRelationalOpsKernelParam,
    (sizeof(vxRelationalOpsKernelParam) / sizeof(vxRelationalOpsKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxRelationalOpsInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxRelationalKernelInfo_lse_uint8 =
{
    VX_KERNEL_ENUM_RELATIONALOPS,
    VX_KERNEL_NAME_RELATIONAL_LESSEQUAL_UINT8,
    NULL,
    vxRelationalOpsKernelParam,
    (sizeof(vxRelationalOpsKernelParam) / sizeof(vxRelationalOpsKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxRelationalOpsInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxRelationalKernelInfo_ne_uint8 =
{
    VX_KERNEL_ENUM_RELATIONALOPS,
    VX_KERNEL_NAME_RELATIONAL_NOTEQUAL_UINT8,
    NULL,
    vxRelationalOpsKernelParam,
    (sizeof(vxRelationalOpsKernelParam) / sizeof(vxRelationalOpsKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxRelationalOpsInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxRelationalKernelInfo_e_uint8 =
{
    VX_KERNEL_ENUM_RELATIONALOPS,
    VX_KERNEL_NAME_RELATIONAL_EQUAL_UINT8,
    NULL,
    vxRelationalOpsKernelParam,
    (sizeof(vxRelationalOpsKernelParam) / sizeof(vxRelationalOpsKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxRelationalOpsInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxRelationalKernelInfo_CPU =
{
    VX_KERNEL_ENUM_RELATIONALOPS,
    VX_KERNEL_NAME_RELATIONAL_GREATER_FP16,
    vxRelationalOpsKernel,
    vxRelationalOpsCpuKernelParam,
    (sizeof(vxRelationalOpsCpuKernelParam) / sizeof(vxRelationalOpsCpuKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_RELATIONAL_OPS_list[] =
{
    &vxRelationalKernelInfo_CPU,
    &vxRelationalKernelInfo_gt_fp16,
    &vxRelationalKernelInfo_gte_fp16,
    &vxRelationalKernelInfo_ls_fp16,
    &vxRelationalKernelInfo_lse_fp16,
    &vxRelationalKernelInfo_ne_fp16,
    &vxRelationalKernelInfo_e_fp16,
    &vxRelationalKernelInfo_gt_int16,
    &vxRelationalKernelInfo_gte_int16,
    &vxRelationalKernelInfo_ls_int16,
    &vxRelationalKernelInfo_lse_int16,
    &vxRelationalKernelInfo_ne_int16,
    &vxRelationalKernelInfo_e_int16,
    &vxRelationalKernelInfo_gt_int8,
    &vxRelationalKernelInfo_gte_int8,
    &vxRelationalKernelInfo_ls_int8,
    &vxRelationalKernelInfo_lse_int8,
    &vxRelationalKernelInfo_ne_int8,
    &vxRelationalKernelInfo_e_int8,
    &vxRelationalKernelInfo_gt_uint8,
    &vxRelationalKernelInfo_gte_uint8,
    &vxRelationalKernelInfo_ls_uint8,
    &vxRelationalKernelInfo_lse_uint8,
    &vxRelationalKernelInfo_ne_uint8,
    &vxRelationalKernelInfo_e_uint8,
    NULL
};
#ifdef __cplusplus
}
#endif
