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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "vsi_nn_platform.h"

#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_test.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_math.h"
#include "client/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"

#define TENSOR_NUM_INPUT  (RELU_KERAS_INPUTS_COUNT)
#define TENSOR_NUM_OUTPUT (RELU_KERAS_OUTPUTS_COUNT)
#define TENSOR_NUM (TENSOR_NUM_INPUT+TENSOR_NUM_OUTPUT)

static float vsi_nn_DtypeToFloat32_Ex
    (
    uint8_t   * src,
    uint32_t    index,
    const vsi_nn_dtype_t * src_dtype
    )
{
    float value = 0.0f;
    vsi_status status;

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

vsi_status VX_CALLBACK vxRelu_Keras_InternalKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
    vsi_status status = VX_SUCCESS;
    vsi_nn_tensor_attr_t attr[TENSOR_NUM];
    vx_tensor_addressing user_addr[TENSOR_NUM]  = {NULL};
    vx_uint8    *buffer_ptr[TENSOR_NUM]            = {NULL};
    vx_tensor   tensor[TENSOR_NUM];
    uint32_t    stride_size[TENSOR_NUM][VSI_NN_MAX_DIM_NUM];

    vx_context   context                        = vxGetContext((vx_reference)node);
    vx_uint32    i                              = 0;
    uint32_t     element_count                  = 1;
    float        alpha                          = 0;
    float        max_value                      = 0;
    float        threshold                      = 0;

    for(i = 0; i < TENSOR_NUM; i++)
    {
        memset(&attr[i], 0x0, sizeof(vsi_nn_tensor_attr_t));
    }
    memset(stride_size, 0x0, TENSOR_NUM * VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));
    for( i = 0; i < TENSOR_NUM_INPUT; i ++ )
    {
        tensor[i] = (vx_tensor)paramObj[i];
        buffer_ptr[i] = vsi_nn_ConvertRawTensorToData2(context, tensor[i],
            &(attr[i]), stride_size[i], &(user_addr[i]), VX_READ_ONLY);
    }

    for( i = TENSOR_NUM_INPUT; i < TENSOR_NUM; i ++ )
    {
        tensor[i] = (vx_tensor)paramObj[i];
        buffer_ptr[i] = vsi_nn_ConvertRawTensorToData2(context, tensor[i],
            &(attr[i]), stride_size[i], &(user_addr[i]), VX_WRITE_ONLY);
    }

    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 0], &(alpha), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 1], &(max_value), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 2], &(threshold), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    for (i = 0; i < attr[RELU_KERAS_INPUT].dim_num; i++)
    {
        element_count *= attr[RELU_KERAS_INPUT].size[i];
    }

    for (i = 0; i < element_count; i++)
    {
        float data = vsi_nn_DtypeToFloat32_Ex(buffer_ptr[RELU_KERAS_INPUT], i, &attr[RELU_KERAS_INPUT].dtype);

        data = data >= max_value ? max_value : data;
        data = data < threshold ? alpha * (data - threshold) : data;

        vsi_nn_Float32ToDtype_Ext(data, buffer_ptr[RELU_KERAS_INPUTS_COUNT + RELU_KERAS_OUTPUT],
            i, &attr[RELU_KERAS_INPUTS_COUNT + RELU_KERAS_OUTPUT].dtype);
    }

    //save data
    for( i = TENSOR_NUM_INPUT; i < TENSOR_NUM; i ++ )
    {
        if (buffer_ptr[i])
        {
            status = vsi_nn_vxCopyDataToTensor(context, tensor[i], &attr[i], buffer_ptr[i]);
            TEST_CHECK_STATUS(status, final);
        }
    }

final:
    for( i = 0; i < TENSOR_NUM; i ++ )
    {
        if (buffer_ptr[i]) free(buffer_ptr[i]);

        if (user_addr[i]) vxReleaseTensorAddressing(&(user_addr[i]));
    }

    return status;
}

vx_status VX_CALLBACK vxRelu_keras_internalInitializer
    (
    vx_node node,
    const vx_reference *paramObj,
    vx_uint32 paraNum
    )
{
// Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
    vx_kernel_execution_parameters_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {1, 1, 1},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vx_status status = VX_SUCCESS;

    vx_tensor    input              = (vx_tensor)paramObj[0];
    vx_tensor    output             = (vx_tensor)paramObj[1];
    vx_enum      srcFormat          = VSI_NN_TYPE_FLOAT16;
    vx_enum      dstFormat          = VSI_NN_TYPE_FLOAT16;
    vx_float32   alpha              = 0;
    vx_float32   threshold          = 0;
    vx_int8      srcFixPointPos     = 0;
    vx_int8      dstFixPointPos     = 0;
    vx_enum      srcQntType         = VX_QUANT_NONE;
    vx_enum      dstQntType         = VX_QUANT_NONE;
    vx_float32   input_scale         = 1.0f;
    vx_float32   inputZP            = 0;
    vx_float32   offset             = 0;
    vx_float32   output_scale       = 1.0f;
    vx_float32   outputZP           = 0;
    vx_float32   inputTail          = 0;
    vsi_nn_tensor_attr_t attr[2];

    memset(&attr[0], 0, sizeof(vsi_nn_tensor_attr_t));
    memset(&attr[1], 0, sizeof(vsi_nn_tensor_attr_t));

    status  = vsi_nn_vxGetTensorAttr(input, &attr[0]);
    status |= vsi_nn_vxGetTensorAttr(output, &attr[1]);
    status |= vxCopyScalar((vx_scalar)paramObj[2], &(alpha),VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    status |= vxCopyScalar((vx_scalar)paramObj[4], &(threshold),VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    if(status < 0)
        printf("error-%s,%d\n",__FILE__,__LINE__);

    srcFormat = attr[0].dtype.vx_type;
    dstFormat = attr[1].dtype.vx_type;
    srcFixPointPos = attr[0].dtype.fl;
    dstFixPointPos = attr[1].dtype.fl;
    input_scale = attr[0].dtype.scale;
    inputZP = (vx_float32)attr[0].dtype.zero_point;
    output_scale = 1.0f / attr[0].dtype.scale;
    outputZP = (vx_float32)attr[0].dtype.zero_point;
    srcQntType = attr[0].dtype.qnt_type;
    dstQntType = attr[1].dtype.qnt_type;
    offset = alpha * threshold;

    shaderParam.globalWorkScale[0] = 8;
    shaderParam.globalWorkScale[1] = 1;
    shaderParam.globalWorkScale[2] = 1;

    shaderParam.globalWorkSize[0]   = gcmALIGN((attr[1].size[0] + shaderParam.globalWorkScale[0] - 1)
                                        / shaderParam.globalWorkScale[0], 4);
    shaderParam.globalWorkSize[1]   = (attr[1].size[1] + shaderParam.globalWorkScale[1] - 1)
                                        / shaderParam.globalWorkScale[1];
    shaderParam.globalWorkSize[2]   = attr[1].dim_num > 2 ? attr[1].size[2] : 1;


    if (srcQntType == VX_QUANT_AFFINE_SCALE)
    {
        inputTail = -inputZP * input_scale;
        status |= vxSetNodeUniform(node, "input_scale", 1, &input_scale);
        status |= vxSetNodeUniform(node, "inputTail", 1, &inputTail);
    }
    else if (srcQntType == VX_QUANT_DYNAMIC_FIXED_POINT)
    {
        if (srcFixPointPos >=0 )
            input_scale = 1.0f / (vx_float32) (1 << srcFixPointPos);
        else
            input_scale = (vx_float32) (1 << -srcFixPointPos);

        status |= vxSetNodeUniform(node, "input_scale", 1, &input_scale);
    }

    if (dstQntType == VX_QUANT_AFFINE_SCALE)
    {
        status |= vxSetNodeUniform(node, "output_scale", 1, &output_scale);
        status |= vxSetNodeUniform(node, "outputZP", 1, &outputZP);
    }
    else if (dstQntType == VX_QUANT_DYNAMIC_FIXED_POINT)
    {
        if (srcFixPointPos >=0 )
            output_scale = (vx_float32) (1 << dstFixPointPos);
        else
            output_scale = 1.0f / (vx_float32) (1 << -dstFixPointPos);

        status |= vxSetNodeUniform(node, "output_scale", 1, &output_scale);
    }

    if (srcFormat == VSI_NN_TYPE_FLOAT16)
    {
        vx_uint32 uniConvIntegertoFP32_Lo_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };
        vx_uint32 uniConvIntegertoFP32_Hi_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };

        status |= vxSetNodeUniform(node, "uniConvIntegertoFP32_Lo_4x4", 1, uniConvIntegertoFP32_Lo_4x4);
        status |= vxSetNodeUniform(node, "uniConvIntegertoFP32_Hi_4x4", 1, uniConvIntegertoFP32_Hi_4x4);
    }
    else if (srcFormat == VSI_NN_TYPE_BFLOAT16)
    {
        vx_uint32 uniConvBF16toF32_Part0_2x8[16] = {
            0x11111111, // TCfg
            0x01010101, // ASelt
            0x01050004, 0x03070206, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };
        vx_uint32 uniConvBF16toF32_Part1_2x8[16] = {
            0x11111111, // TCfg
            0x01010101, // ASelt
            0x05050404, 0x07070606, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };

        status |= vxSetNodeUniform(node, "uniConvBF16toF32_Part0_2x8", 1, uniConvBF16toF32_Part0_2x8);
        status |= vxSetNodeUniform(node, "uniConvBF16toF32_Part1_2x8", 1, uniConvBF16toF32_Part1_2x8);
    }
    else
    {
        vx_uint32 uniConvIntegertoFP32_Lo_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        };
        vx_uint32 uniConvIntegertoFP32_Hi_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        };

        status |= vxSetNodeUniform(node, "uniConvIntegertoFP32_Lo_4x4", 1, uniConvIntegertoFP32_Lo_4x4);
        status |= vxSetNodeUniform(node, "uniConvIntegertoFP32_Hi_4x4", 1, uniConvIntegertoFP32_Hi_4x4);
    }

    if (dstFormat == VSI_NN_TYPE_FLOAT16)
    {
        vx_uint32 uniExtractHalf8_2x8[16] = {
            0x11111111, // TCfg
            0x11110000, // ASelt
            0x06040200, 0x06040200, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
        };

        status |= vxSetNodeUniform(node, "uniExtractHalf8_2x8", 1, uniExtractHalf8_2x8);
    }
    else if (dstFormat == VSI_NN_TYPE_BFLOAT16)
    {
        vx_uint32 uniPackedBF16_2x8[16] = {
            0x11111111, // TCfg
            0x11110000, // ASelt
            0x07050301, 0x07050301, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };

        status |= vxSetNodeUniform(node, "uniPackedBF16_2x8", 1, uniPackedBF16_2x8);
    }
    else
    {
        vx_uint32 uniExtractInteger_2x8[16] = {
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };

        status |= vxSetNodeUniform(node, "uniExtractInteger_2x8", 1, uniExtractInteger_2x8);
    }

    status |= vxSetNodeUniform(node, "offset", 1, &offset);
    status |= vxSetNodeAttribute(node, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));

    return status;
}

static vx_param_description_t vxRelu_Keras_InternalKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#ifdef __cplusplus
extern "C" {
#endif
vx_kernel_description_t vxRelu_keras_internal_CPU =
{
    VX_KERNEL_ENUM_RELU_KERAS_INTERNAL,
    "com.vivantecorp.extension.relu_keras_sw",
    vxRelu_Keras_InternalKernel,
    vxRelu_Keras_InternalKernelParam,
    _cnt_of_array( vxRelu_Keras_InternalKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

#define GEN_KERAS_RELU_SH_KERNEL_NAME(SRC_TYPE, DST_TYPE, DIMS) \
    "com.vivantecorp.extension.vxKerasRelu_"#SRC_TYPE"to"#DST_TYPE"_"#DIMS

#define TENSOR_KERAS_RELU_KERNELS(SRC_TYPE, DST_TYPE, DIMS) \
    vx_kernel_description_t vxKerasRelu_##SRC_TYPE##to##DST_TYPE##_##DIMS##_Kernel = \
{ \
    VX_KERNEL_ENUM_RELU_KERAS_INTERNAL, \
    GEN_KERAS_RELU_SH_KERNEL_NAME(SRC_TYPE, DST_TYPE, DIMS), \
    NULL, \
    vxRelu_Keras_InternalKernelParam, \
    (sizeof(vxRelu_Keras_InternalKernelParam) / sizeof(vxRelu_Keras_InternalKernelParam[0])), \
    vsi_nn_KernelValidator, \
    NULL, \
    NULL, \
    vxRelu_keras_internalInitializer, \
    vsi_nn_KernelDeinitializer \
};

TENSOR_KERAS_RELU_KERNELS(BF16, BF16, 3D)
TENSOR_KERAS_RELU_KERNELS(F16,  F16,  3D)
TENSOR_KERAS_RELU_KERNELS(F16,  I16,  3D)
TENSOR_KERAS_RELU_KERNELS(F16,  I8,   3D)
TENSOR_KERAS_RELU_KERNELS(F16,  U8,   3D)
TENSOR_KERAS_RELU_KERNELS(I16,  I16,  3D)
TENSOR_KERAS_RELU_KERNELS(I16,  F16,  3D)
TENSOR_KERAS_RELU_KERNELS(I8,   I8,   3D)
TENSOR_KERAS_RELU_KERNELS(I8,   F16,  3D)
TENSOR_KERAS_RELU_KERNELS(U8,   U8,   3D)
TENSOR_KERAS_RELU_KERNELS(U8,   F16,  3D)

TENSOR_KERAS_RELU_KERNELS(BF16, BF16, 2D)
TENSOR_KERAS_RELU_KERNELS(F16,  F16,  2D)
TENSOR_KERAS_RELU_KERNELS(F16,  I16,  2D)
TENSOR_KERAS_RELU_KERNELS(F16,  I8,   2D)
TENSOR_KERAS_RELU_KERNELS(F16,  U8,   2D)
TENSOR_KERAS_RELU_KERNELS(I16,  I16,  2D)
TENSOR_KERAS_RELU_KERNELS(I16,  F16,  2D)
TENSOR_KERAS_RELU_KERNELS(I8,   I8,   2D)
TENSOR_KERAS_RELU_KERNELS(I8,   F16,  2D)
TENSOR_KERAS_RELU_KERNELS(U8,   U8,   2D)
TENSOR_KERAS_RELU_KERNELS(U8,   F16,  2D)

#define TENSOR_KERAS_RELU_KERENLS_NAME(SRC_TYPE, DST_TYPE, DIMS) \
    &vxKerasRelu_##SRC_TYPE##to##DST_TYPE##_##DIMS##_Kernel,

vx_kernel_description_t * vx_kernel_RELU_KERAS_INTERNAL_list[] =
{
    &vxRelu_keras_internal_CPU,
    TENSOR_KERAS_RELU_KERENLS_NAME(BF16, BF16, 3D)
    TENSOR_KERAS_RELU_KERENLS_NAME(F16,  F16,  3D)
    TENSOR_KERAS_RELU_KERENLS_NAME(F16,  I16,  3D)
    TENSOR_KERAS_RELU_KERENLS_NAME(F16,  I8,   3D)
    TENSOR_KERAS_RELU_KERENLS_NAME(F16,  U8,   3D)
    TENSOR_KERAS_RELU_KERENLS_NAME(I16,  I16,  3D)
    TENSOR_KERAS_RELU_KERENLS_NAME(I16,  F16,  3D)
    TENSOR_KERAS_RELU_KERENLS_NAME(I8,   I8,   3D)
    TENSOR_KERAS_RELU_KERENLS_NAME(I8,   F16,  3D)
    TENSOR_KERAS_RELU_KERENLS_NAME(U8,   U8,   3D)
    TENSOR_KERAS_RELU_KERENLS_NAME(U8,   F16,  3D)

    TENSOR_KERAS_RELU_KERENLS_NAME(BF16, BF16, 2D)
    TENSOR_KERAS_RELU_KERENLS_NAME(F16,  F16,  2D)
    TENSOR_KERAS_RELU_KERENLS_NAME(F16,  I16,  2D)
    TENSOR_KERAS_RELU_KERENLS_NAME(F16,  I8,   2D)
    TENSOR_KERAS_RELU_KERENLS_NAME(F16,  U8,   2D)
    TENSOR_KERAS_RELU_KERENLS_NAME(I16,  I16,  2D)
    TENSOR_KERAS_RELU_KERENLS_NAME(I16,  F16,  2D)
    TENSOR_KERAS_RELU_KERENLS_NAME(I8,   I8,   2D)
    TENSOR_KERAS_RELU_KERENLS_NAME(I8,   F16,  2D)
    TENSOR_KERAS_RELU_KERENLS_NAME(U8,   U8,   2D)
    TENSOR_KERAS_RELU_KERENLS_NAME(U8,   F16,  2D)
    NULL
};
#ifdef __cplusplus
}
#endif
