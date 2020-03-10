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

#include "vsi_nn_platform.h"

#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_math.h"
#include "client/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"

vx_status VX_CALLBACK vxgemmInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    vx_uint32 paraNum
    )
{
    vx_status status = VX_SUCCESS;
    // Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
    vx_kernel_execution_parameters_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vx_tensor input     = (vx_tensor)paramObj[0];
    vx_tensor input2    = (vx_tensor)paramObj[1];
    vx_tensor output    = (vx_tensor)paramObj[2];
    vx_scalar transAs   = (vx_scalar)paramObj[3];
    vx_scalar transBs   = (vx_scalar)paramObj[4];
    vx_uint32 output_size[DIM_SIZE]    = {1, 1, 1, 1};
    vx_enum inDataType, inDataType2, outDataType;
    vx_float32 scaleIn1 = 0;
    vx_float32 scaleIn2 = 0;
    vx_float32 scaleOut = 0;
    vx_float32 reScaleOut = 0.f;
    vx_float32 scaleIn2divOut = 0;
    vx_float32 inScaleMul = 0.f;
    vx_float32 inScaledivOut = 0.f;
    int32_t output_ZP = 0;
    int32_t input1_ZP = 0;
    int32_t input2_ZP = 0;
    vx_bool transA = FALSE;
    vx_bool transB = FALSE;
    vsi_nn_tensor_attr_t attr[3];
    uint32_t i, output_dim;

    memset(&attr[0], 0, sizeof(vsi_nn_tensor_attr_t));
    memset(&attr[1], 0, sizeof(vsi_nn_tensor_attr_t));
    memset(&attr[2], 0, sizeof(vsi_nn_tensor_attr_t));
    status  = vsi_nn_vxGetTensorAttr(input, &attr[0]);
    status |= vsi_nn_vxGetTensorAttr(input2, &attr[1]);
    status |= vsi_nn_vxGetTensorAttr(output, &attr[2]);
    if (status != VX_SUCCESS)
    {
        VSILOGE("vsi_nn_vxGetTensorAttr failure! at line %d\n", __LINE__);
        return status;
    }

    output_dim  = attr[2].dim_num;
    for (i = 0; i < output_dim; i++)
    {
        output_size[i] = attr[2].size[i];
    }
    inDataType  = attr[0].dtype.vx_type;
    inDataType2 = attr[1].dtype.vx_type;
    outDataType = attr[2].dtype.vx_type;
    input1_ZP   = attr[0].dtype.zero_point;
    scaleIn1    = attr[0].dtype.scale;
    input2_ZP   = attr[1].dtype.zero_point;
    scaleIn2    = attr[1].dtype.scale;
    output_ZP   = attr[2].dtype.zero_point;
    scaleOut    = attr[2].dtype.scale;

    status |= vxCopyScalar(transAs, &transA, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    status |= vxCopyScalar(transBs, &transB, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkOffset[2] = 0;
    shaderParam.globalWorkScale[0]  = 4;
    shaderParam.globalWorkScale[1]  = 4;
    shaderParam.globalWorkScale[2]  = 1;
    shaderParam.globalWorkSize[0]   = gcmALIGN((output_size[0] + shaderParam.globalWorkScale[0] - 1)
        / shaderParam.globalWorkScale[0], 4);
    shaderParam.globalWorkSize[1]   = (output_size[1] + shaderParam.globalWorkScale[1] - 1)
        / shaderParam.globalWorkScale[1];
    shaderParam.globalWorkSize[2]   = output_size[2];

    if(scaleOut == 0)
        scaleOut = 1;
    {
        vx_uint32 uniU8SubZptoFp16_dp2x8[16] = {
            0x99999999, // TCfg
            0x44444444, // ASelt
            0x03020100, 0x07060504, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
        };
        vx_uint32 uniFp16MulFp16AddtoFp32_dp8x2[16] = {
            0x00005555, // TCfg
            0x00000000, // ASelt
            0x76543210, 0x00000000, // ABin
            0x00005555, // BSelt
            0x76543210, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
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
        vx_uint32 uniConvertUint8SubZpToFp32_4x4[16] = {
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002100, // AccumType, ConstantType, and PostShift
            0xbc003c00, 0x00000000, 0xbc003c00, 0x00000000, 0xbc003c00, 0x00000000, 0xbc003c00, 0x00000000 // Constant
        };

        scaleIn2divOut = scaleIn2 / scaleOut;
        inScaleMul = scaleIn1 * scaleIn2;
        reScaleOut = 1 / scaleOut;
        inScaledivOut = inScaleMul / scaleOut;
        if (transA == FALSE && transB == TRUE &&
           (inDataType == VSI_NN_TYPE_UINT8 && inDataType2 == VSI_NN_TYPE_UINT8))
        {
            /* vsi_nn_kernel_matrixmul_transbp2.vx */
            status |= vxSetNodeUniform(nodObj, "input1_ZP", 1, &input1_ZP);
            status |= vxSetNodeUniform(nodObj, "input2_ZP", 1, &input2_ZP);
            status |= vxSetNodeUniform(nodObj, "inScaleMul", 1, &inScaleMul);
            status |= vxSetNodeUniform(nodObj, "uniU8SubZptoFp16_dp2x8", 1, uniU8SubZptoFp16_dp2x8);
            status |= vxSetNodeUniform(nodObj, "uniFp16MulFp16AddtoFp32_dp8x2", 1, uniFp16MulFp16AddtoFp32_dp8x2);
            status |= vxSetNodeUniform(nodObj, "inScaledivOut", 1, &inScaledivOut);
            status |= vxSetNodeUniform(nodObj, "uniConvertInt32toUint8_2x8", 1, uniConvertInt32toUint8_2x8);
            status |= vxSetNodeUniform(nodObj, "output_ZP", 1, &output_ZP);
        }
        else if (transA == FALSE && transB == TRUE &&
           (inDataType == VSI_NN_TYPE_UINT8 || inDataType2 == VSI_NN_TYPE_UINT8 || outDataType == VSI_NN_TYPE_UINT8))
        {
            /* vsi_nn_kernel_matrixmul_transbp1.vx */
            status |= vxSetNodeUniform(nodObj, "input2_ZP", 1, &input2_ZP);
            status |= vxSetNodeUniform(nodObj, "input2Scale", 1, &scaleIn2);
            status |= vxSetNodeUniform(nodObj, "uniU8SubZptoFp16_dp2x8", 1, uniU8SubZptoFp16_dp2x8);
            status |= vxSetNodeUniform(nodObj, "uniFp16MulFp16AddtoFp32_dp8x2", 1, uniFp16MulFp16AddtoFp32_dp8x2);
            status |= vxSetNodeUniform(nodObj, "scaleIn2divOut", 1, &scaleIn2divOut);
            status |= vxSetNodeUniform(nodObj, "uniConvertInt32toUint8_2x8", 1, uniConvertInt32toUint8_2x8);
            status |= vxSetNodeUniform(nodObj, "output_ZP", 1, &output_ZP);
        }
        else if(transA == FALSE && transB == FALSE &&
            inDataType == VSI_NN_TYPE_FLOAT16 && inDataType2 == VSI_NN_TYPE_FLOAT16 && outDataType == VSI_NN_TYPE_UINT8)
        {
            /* vsi_nn_kernel_matrixmul_fp16.vx */
            status |= vxSetNodeUniform(nodObj, "uniConvertInt32toUint8_2x8", 1, uniConvertInt32toUint8_2x8);
            status |= vxSetNodeUniform(nodObj, "outputScale", 1, &reScaleOut);
            status |= vxSetNodeUniform(nodObj, "output_ZP", 1, &output_ZP);
        }
        else if(transA == FALSE && transB == FALSE &&
            (inDataType == VSI_NN_TYPE_UINT8 || inDataType2 == VSI_NN_TYPE_UINT8 || outDataType == VSI_NN_TYPE_UINT8))
        {
            /* vsi_nn_kernel_matrixmul.vx */
            status |= vxSetNodeUniform(nodObj, "uniConvertUint8SubZpToFp32_4x4", 1, uniConvertUint8SubZpToFp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvertInt32toUint8_2x8", 1, uniConvertInt32toUint8_2x8);
            status |= vxSetNodeUniform(nodObj, "input1Scale", 1, &scaleIn1);
            status |= vxSetNodeUniform(nodObj, "input1_ZP", 1, &input1_ZP);
            status |= vxSetNodeUniform(nodObj, "input2Scale", 1, &scaleIn2);
            status |= vxSetNodeUniform(nodObj, "input2_ZP", 1, &input2_ZP);
            status |= vxSetNodeUniform(nodObj, "outputScale", 1, &reScaleOut);
            status |= vxSetNodeUniform(nodObj, "output_ZP", 1, &output_ZP);
        }
        else if (transA == FALSE && transB == FALSE &&
            inDataType == VSI_NN_TYPE_FLOAT16 && inDataType2 == VSI_NN_TYPE_FLOAT16 && outDataType == VSI_NN_TYPE_FLOAT16)
        {
            /* vsi_nn_kernel_matrixmul_fp16.vx */
            status |= VX_SUCCESS;
        }
        else if (transA == FALSE && transB == TRUE &&
           (inDataType == VSI_NN_TYPE_FLOAT16 || inDataType2 == VSI_NN_TYPE_FLOAT16 || outDataType == VSI_NN_TYPE_FLOAT16))
        {
            /* vsi_nn_kernel_matrixmul_transbfp16.vx */
            status |= vxSetNodeUniform(nodObj, "uniFp16MulFp16AddtoFp32_dp8x2", 1, uniFp16MulFp16AddtoFp32_dp8x2);
        }
        else
        {
            VSILOGE("[%s : %d]Initializer  failure!(MATRIXMUL)\n",__FILE__, __LINE__);
            return VX_FAILURE;
        }
    }

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));

    if(status < 0)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
    }
    return status;
}

static vx_param_description_t vxgemmKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED}
};

#ifdef __cplusplus
extern "C" {
#endif
vx_kernel_description_t vxgemmKernelInfo =
{
    VX_KERNEL_ENUM_GEMM,
    VX_KERNEL_NAME_GEMM,
    NULL,
    vxgemmKernelParam,
    (sizeof(vxgemmKernelParam) / sizeof(vxgemmKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxgemmInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxgemmKernelInfo_u8 =
{
    VX_KERNEL_ENUM_GEMM,
    VX_KERNEL_NAME_GEMM_UINT8,
    NULL,
    vxgemmKernelParam,
    (sizeof(vxgemmKernelParam) / sizeof(vxgemmKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxgemmInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxgemmKernelInfo_Fp16U8_Fp16 =
{
    VX_KERNEL_ENUM_GEMM,
    VX_KERNEL_NAME_GEMM_FP16U8_Fp16,
    NULL,
    vxgemmKernelParam,
    (sizeof(vxgemmKernelParam) / sizeof(vxgemmKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxgemmInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxgemmKernelInfo_Fp16U8_U8 =
{
    VX_KERNEL_ENUM_GEMM,
    VX_KERNEL_NAME_GEMM_FP16U8_U8,
    NULL,
    vxgemmKernelParam,
    (sizeof(vxgemmKernelParam) / sizeof(vxgemmKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxgemmInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxgemmKernelInfo_TransBFp16U8toFp16 =
{
    VX_KERNEL_ENUM_GEMM,
    VX_KERNEL_NAME_GEMM_TRANSB_FP16U8TOFP16,
    NULL,
    vxgemmKernelParam,
    (sizeof(vxgemmKernelParam) / sizeof(vxgemmKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxgemmInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxgemmKernelInfo_TransBFp16U8toU8 =
{
    VX_KERNEL_ENUM_GEMM,
    VX_KERNEL_NAME_GEMM_TRANSB_FP16U8TOU8,
    NULL,
    vxgemmKernelParam,
    (sizeof(vxgemmKernelParam) / sizeof(vxgemmKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxgemmInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxgemmKernelInfo_TransBU8U8toFp16 =
{
    VX_KERNEL_ENUM_GEMM,
    VX_KERNEL_NAME_GEMM_TRANSB_U8U8TOFP16,
    NULL,
    vxgemmKernelParam,
    (sizeof(vxgemmKernelParam) / sizeof(vxgemmKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxgemmInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxgemmKernelInfo_TransBU8U8toU8 =
{
    VX_KERNEL_ENUM_GEMM,
    VX_KERNEL_NAME_GEMM_TRANSB_U8U8TOU8,
    NULL,
    vxgemmKernelParam,
    (sizeof(vxgemmKernelParam) / sizeof(vxgemmKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxgemmInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxgemmKernelInfo_Fp16Fp16_U8 =
{
    VX_KERNEL_ENUM_GEMM,
    VX_KERNEL_NAME_GEMM_FP16FP16_U8,
    NULL,
    vxgemmKernelParam,
    (sizeof(vxgemmKernelParam) / sizeof(vxgemmKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxgemmInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxgemmKernelInfo_TransBFp16Fp16toFp16 =
{
    VX_KERNEL_ENUM_GEMM,
    VX_KERNEL_NAME_GEMM_TRANSB_FP16FP16TOFP16,
    NULL,
    vxgemmKernelParam,
    (sizeof(vxgemmKernelParam) / sizeof(vxgemmKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxgemmInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_MATRIXMUL_list[] =
{
    NULL,
    &vxgemmKernelInfo,
    &vxgemmKernelInfo_u8,
    &vxgemmKernelInfo_Fp16U8_Fp16,
    &vxgemmKernelInfo_Fp16U8_U8,
    &vxgemmKernelInfo_TransBFp16U8toFp16,
    &vxgemmKernelInfo_TransBFp16U8toU8,
    &vxgemmKernelInfo_TransBU8U8toFp16,
    &vxgemmKernelInfo_Fp16Fp16_U8,
    &vxgemmKernelInfo_TransBU8U8toU8,
    &vxgemmKernelInfo_TransBFp16Fp16toFp16,
    NULL
};
#ifdef __cplusplus
}
#endif
