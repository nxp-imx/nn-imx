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

#define TENSOR_NUM_INPUT  (REDUCEPROD_INPUTS_COUNT)
#define TENSOR_NUM_OUTPUT (REDUCEPROD_OUTPUTS_COUNT)
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

static vsi_status VX_CALLBACK vxReduceProd_internalKernel
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
    vx_uint32  size[4];
    vx_uint32  dims, innerSize, outerSize, axisSize;
    vx_uint32  outer, inner, i, index;
    vx_float32 prodValue, tmpValue;
    int32_t axis;

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
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM], &(axis), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    dims = attr[REDUCEPROD_INPUT].dim_num;
    size[0] = attr[REDUCEPROD_INPUT].size[0];
    size[1] = dims > 1 ? attr[REDUCEPROD_INPUT].size[1] : 1;
    size[2] = dims > 2 ? attr[REDUCEPROD_INPUT].size[2] : 1;
    size[3] = dims > 3 ? attr[REDUCEPROD_INPUT].size[3] : 1;
    axisSize =  attr[REDUCEPROD_INPUT].size[axis];

    switch(axis)
    {
        case 0:
            innerSize = 1;
            outerSize = size[1] * size[2] * size[3];
            break;
        case 1:
            innerSize = size[0];
            outerSize = size[2] * size[3];
            break;
        case 2:
            innerSize = size[0] * size[1];
            outerSize = size[3];
            break;
        case 3:
            innerSize = size[0] * size[1] * size[2];
            outerSize = 1;
            break;
        default:
        printf("Input tensor error dimension[%u]\n", dims);
        return VX_ERROR_INVALID_DIMENSION;
    }

    for (outer = 0; outer < outerSize; ++outer) {
        for (inner = 0; inner < innerSize; ++inner) {
            index = outer * axisSize * innerSize + inner;
            prodValue = vsi_nn_DtypeToFloat32_Ex(buffer_ptr[REDUCEPROD_INPUT],
                                                 index, &attr[REDUCEPROD_INPUT].dtype);
            for (i = 1; i < axisSize; ++i) {
                index    = (outer * axisSize + i) * innerSize + inner;
                tmpValue = vsi_nn_DtypeToFloat32_Ex(buffer_ptr[REDUCEPROD_INPUT],
                                                   index, &attr[REDUCEPROD_INPUT].dtype);
                prodValue = prodValue * tmpValue;
            }

            index     =  outer * innerSize + inner;
            vsi_nn_Float32ToDtype_Ext(prodValue, buffer_ptr[REDUCEPROD_INPUTS_COUNT + REDUCEPROD_OUTPUT],
                index, &attr[REDUCEPROD_INPUTS_COUNT + REDUCEPROD_OUTPUT].dtype);
        }
    }

    for( i = TENSOR_NUM_INPUT; i < TENSOR_NUM; i ++ )
    {
        if (buffer_ptr[i])
        {
            status = vxCopyTensorPatch(
                tensor[i],
                NULL,
                user_addr[i],
                buffer_ptr[i],
                VX_WRITE_ONLY,
                0
                );
        }

        if (user_addr[i]) vxReleaseTensorAddressing(&(user_addr[i]));
    }

    for( i = 0; i < TENSOR_NUM; i ++ )
    {
        if (buffer_ptr[i]) free(buffer_ptr[i]);

        if (user_addr[i]) vxReleaseTensorAddressing(&(user_addr[i]));
    }
    return status;
} /* _VX_KERNEL_FUNC_KERNEL() */


vx_status VX_CALLBACK vxReduceProd_internalInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    vx_uint32 paraNum
    )
{
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)

    vx_kernel_execution_parameters_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {1, 1, 1},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread
    vx_status    status             = VX_SUCCESS;
    vx_tensor    input              = (vx_tensor)paramObj[0];
    vx_tensor    output             = (vx_tensor)paramObj[1];
    vx_uint32    width              = 0;
    vx_uint32    height             = 0;
    vx_uint32    depth              = 0;
    vx_uint32    axis               = 0;
    vx_int8      srcFixPointPos     = 0;
    vx_int8      dstFixPointPos     = 0;
    vx_float32   input_scale        = 0;
    vx_float32   output_scale       = 0;
    vx_int32     inputZP            = 0;
    vx_int32     outputZP           = 0;
    vx_enum      srcFormat          = VSI_NN_TYPE_FLOAT16;
    vx_enum      dstFormat          = VSI_NN_TYPE_FLOAT16;
    vx_enum      srcQntType         = VX_QUANT_NONE;
    vx_enum      dstQntType         = VX_QUANT_NONE;
    vsi_nn_tensor_attr_t attr[2];
    vx_int32     axisSize = 0;

    vxCopyScalar((vx_scalar)paramObj[2], &(axis),VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    memset(&attr[0], 0, sizeof(vsi_nn_tensor_attr_t));
    memset(&attr[1], 0, sizeof(vsi_nn_tensor_attr_t));

    status  = vsi_nn_vxGetTensorAttr(input, &attr[0]);
    status |= vsi_nn_vxGetTensorAttr(output, &attr[1]);

    if(status < 0)
    {
        VSILOGE("error-%s,%d\n",__FILE__,__LINE__);
        return status;
    }

    srcFormat      = attr[0].dtype.vx_type;
    dstFormat      = attr[1].dtype.vx_type;
    srcFixPointPos = attr[0].dtype.fl;
    dstFixPointPos = attr[1].dtype.fl;
    input_scale    = attr[0].dtype.scale;
    output_scale   = attr[1].dtype.scale;
    inputZP        = attr[0].dtype.zero_point;
    outputZP       = attr[1].dtype.zero_point;
    width          = attr[0].size[0];
    height         = attr[0].size[1];
    depth          = attr[0].dim_num > 2 ? attr[0].size[2] : 1;
    axisSize       = attr[0].size[axis];
    srcQntType     = attr[0].dtype.qnt_type;
    dstQntType     = attr[1].dtype.qnt_type;

    shaderParam.workDim             = 2;
    switch (axis)
    {
        case 0:
            shaderParam.globalWorkScale[0]  = 1;
            shaderParam.globalWorkScale[1]  = 1;
            shaderParam.globalWorkSize[0]   = height;
            shaderParam.globalWorkSize[1]   = depth;
        break;
        case 1:
            shaderParam.globalWorkScale[0]  = 8;
            shaderParam.globalWorkScale[1]  = 1;
            shaderParam.globalWorkSize[0]   =\
            gcmALIGN((width + shaderParam.globalWorkScale[0] - 1) / shaderParam.globalWorkScale[0], 4);
            shaderParam.globalWorkSize[1]   = depth;
        break;
        case 2:
            shaderParam.globalWorkScale[0]  = 8;
            shaderParam.globalWorkScale[1]  = 1;
            shaderParam.globalWorkSize[0]   =\
            gcmALIGN((width + shaderParam.globalWorkScale[0] - 1) / shaderParam.globalWorkScale[0], 4);
            shaderParam.globalWorkSize[1]   = height;
        break;
        default:
            VSILOGE("error input axis value %d \n", axis);
            return VX_ERROR_INVALID_PARAMETERS;
        break;
    }

    {
        vx_uint32 uniGetLoData_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        };
        vx_uint32 uniGetHiData_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        };
        vx_uint32 uniGetEndLoData_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };
        vx_uint32 uniGetEndHiData_2x8[16] =  {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };

        vx_uint32 uniConvertInt32toUint8_2x8[16] = {
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };

        vx_uint32 uniConvBF16toF32_Part0_2x8[16] = {
            0x11111111, // TCfg
            0x01010101, // ASelt
            0x01050004, 0x03070206, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };
        vx_uint32 uniConvBF16toF32_Part1_2x8[16] = {
            0x11111111, // TCfg
            0x01010101, // ASelt
            0x05050404, 0x07070606, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };
        vx_uint32 uniExtractOddData_2x8[16] = {
            0x11111111, // TCfg
            0x11110000, // ASelt
            0x07050301, 0x07050301, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };

        vx_uint32 uniGetEndConfig[4] = {0x11111111, 0x11111100, 0x11110000, 0x11000000};

        if (0 == axis)
        {
            vx_int32    inputWidth        = 0;
            vx_int32    inputWidthRemain8 = 0;
            if ((axisSize % 8) == 0 )
            {
                inputWidth        = (axisSize / 8 - 1) * 8;
            }
            else
            {
                inputWidth        = axisSize / 8 * 8;
                inputWidthRemain8 = axisSize % 8;
                if (inputWidthRemain8 >= 4)
                {
                    inputWidthRemain8 = inputWidthRemain8 - 4;
                    uniGetEndHiData_2x8[1] = uniGetEndConfig[inputWidthRemain8];
                }
                else
                {
                    uniGetEndLoData_2x8[1] = uniGetEndConfig[inputWidthRemain8];
                    uniGetEndHiData_2x8[1] = uniGetEndConfig[0];
                }
            }
            status |= vxSetNodeUniform(nodObj, "inputWidth", 1, &inputWidth);
            status |= vxSetNodeUniform(nodObj, "uniGetEndLoData_2x8", 1, uniGetEndLoData_2x8);
            status |= vxSetNodeUniform(nodObj, "uniGetEndHiData_2x8", 1, uniGetEndHiData_2x8);
        }
        else if (1 == axis || 2 == axis)
        {
            status |= vxSetNodeUniform(nodObj, "uniConvertInt32toUint8_2x8", 1, uniConvertInt32toUint8_2x8);
            status |= vxSetNodeUniform(nodObj, "axisSize", 1, &axisSize);
            if (dstFormat == VSI_NN_TYPE_BFLOAT16)
            {
                status |= vxSetNodeUniform(nodObj, "uniExtractOddData_2x8", 1, uniExtractOddData_2x8);
            }
        }

        if (srcFormat == VSI_NN_TYPE_BFLOAT16)
        {
            status |= vxSetNodeUniform(nodObj, "uniConvBF16toF32_Part0_2x8", 1, uniConvBF16toF32_Part0_2x8);
            status |= vxSetNodeUniform(nodObj, "uniConvBF16toF32_Part1_2x8", 1, uniConvBF16toF32_Part1_2x8);
        }
        else
        {
            status |= vxSetNodeUniform(nodObj, "uniGetLoData_4x4", 1, uniGetLoData_4x4);
            status |= vxSetNodeUniform(nodObj, "uniGetHiData_4x4", 1, uniGetHiData_4x4);
        }
    }

    if (srcQntType == VX_QUANT_DYNAMIC_FIXED_POINT)
    {
        if (srcFixPointPos >= 0)
        {
            input_scale = 1.0f / (vx_float32) (1 << srcFixPointPos);
        }
        else if (srcFixPointPos < 0)
        {
            input_scale = (vx_float32) (1 << -srcFixPointPos);
        }
        inputZP     = 0;
        status |= vxSetNodeUniform(nodObj, "inputScale", 1, &input_scale);
    }
    else if (srcQntType == VX_QUANT_AFFINE_SCALE)
    {
        vx_float32 input_offset_asymmetric = 0;
        input_scale    = attr[0].dtype.scale;
        inputZP        = attr[0].dtype.zero_point;
        input_offset_asymmetric = (vx_float32)inputZP;
        status |= vxSetNodeUniform(nodObj, "inputScale", 1, &input_scale);
        status |= vxSetNodeUniform(nodObj, "input_offset_asymmetric", 1, &input_offset_asymmetric);
    }
    else
    {
        input_scale = 1.0;
        inputZP     = 0;
    }

    if (dstQntType == VX_QUANT_DYNAMIC_FIXED_POINT)
    {
        if (dstFixPointPos >= 0)
        {
            output_scale = (vx_float32) (1 << dstFixPointPos);
        }
        else if (dstFixPointPos < 0)
        {
            output_scale = 1.0f / (vx_float32) (1 << -dstFixPointPos);
        }
        status |= vxSetNodeUniform(nodObj, "outputScale", 1, &output_scale);
    }
    else if (dstQntType == VX_QUANT_AFFINE_SCALE)
    {
        vx_float32 output_offset_asymmetric = 0;
        output_scale = 1.0f / (vx_float32)(output_scale);
        output_offset_asymmetric = (vx_float32)outputZP;
        status |= vxSetNodeUniform(nodObj, "outputScale", 1, &output_scale);
        status |= vxSetNodeUniform(nodObj, "output_offset_asymmetric", 1, &output_offset_asymmetric);
    }
    else
    {
        output_scale = 1;
        outputZP     = 0;
    }

    status |= vxSetNodeAttribute(nodObj,
    VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS, &shaderParam, sizeof(vx_kernel_execution_parameters_t));
#undef gcmALIGN
    return status;
}


static vx_param_description_t vxReduceProdKernelParam[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

#ifdef __cplusplus
extern "C" {
#endif

vx_kernel_description_t vxReduceProd_internal_CPU =
{
    VX_KERNEL_ENUM_REDUCEPROD_INTERNAL,
    "com.vivantecorp.extension.vxcReduceProd_sw",
    vxReduceProd_internalKernel,
    vxReduceProdKernelParam,
    (sizeof(vxReduceProdKernelParam) / sizeof(vxReduceProdKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};


#define REDUCEPROD_KERNELS(AXI_INDEX, SRC_TYPE, DST_TYPE) \
    vx_kernel_description_t vxReduceProd_##AXI_INDEX##_##SRC_TYPE##to##DST_TYPE##_Kernel = \
{ \
    VX_KERNEL_ENUM_REDUCEPROD_INTERNAL, \
    VX_KERNEL_NAME_REDUCEPROD_##AXI_INDEX##_##SRC_TYPE##TO##DST_TYPE, \
    NULL, \
    vxReduceProdKernelParam, \
    (sizeof(vxReduceProdKernelParam) / sizeof(vxReduceProdKernelParam[0])), \
    vsi_nn_KernelValidator, \
    NULL, \
    NULL, \
    vxReduceProd_internalInitializer, \
    vsi_nn_KernelDeinitializer \
};

#define REDUCEPROD_KERNELS_2D(AXI_INDEX, SRC_TYPE, DST_TYPE) \
    vx_kernel_description_t vxReduceProd_##AXI_INDEX##_##SRC_TYPE##to##DST_TYPE##_2D_Kernel = \
{ \
    VX_KERNEL_ENUM_REDUCEPROD_INTERNAL, \
    VX_KERNEL_NAME_REDUCEPROD_##AXI_INDEX##_##SRC_TYPE##TO##DST_TYPE##_2D, \
    NULL, \
    vxReduceProdKernelParam, \
    (sizeof(vxReduceProdKernelParam) / sizeof(vxReduceProdKernelParam[0])), \
    vsi_nn_KernelValidator, \
    NULL, \
    NULL, \
    vxReduceProd_internalInitializer, \
    vsi_nn_KernelDeinitializer \
};

REDUCEPROD_KERNELS(AXI0, F16, F16)
REDUCEPROD_KERNELS(AXI0, F16, I16)
REDUCEPROD_KERNELS(AXI0, F16, I8)
REDUCEPROD_KERNELS(AXI0, F16, U8)
REDUCEPROD_KERNELS(AXI0, I16, I16)
REDUCEPROD_KERNELS(AXI0, I8,  I8)
REDUCEPROD_KERNELS(AXI0, U8,  U8)
REDUCEPROD_KERNELS(AXI0, I16, F16)
REDUCEPROD_KERNELS(AXI0, I8,  F16)
REDUCEPROD_KERNELS(AXI0, U8,  F16)
REDUCEPROD_KERNELS(AXI0, BF16,  BF16)

REDUCEPROD_KERNELS(AXI1, F16, F16)
REDUCEPROD_KERNELS(AXI1, F16, I16)
REDUCEPROD_KERNELS(AXI1, F16, I8)
REDUCEPROD_KERNELS(AXI1, F16, U8)
REDUCEPROD_KERNELS(AXI1, I16, I16)
REDUCEPROD_KERNELS(AXI1, I8,  I8)
REDUCEPROD_KERNELS(AXI1, U8,  U8)
REDUCEPROD_KERNELS(AXI1, I16, F16)
REDUCEPROD_KERNELS(AXI1, I8,  F16)
REDUCEPROD_KERNELS(AXI1, U8,  F16)
REDUCEPROD_KERNELS(AXI1, BF16,  BF16)

REDUCEPROD_KERNELS(AXI2, F16, F16)
REDUCEPROD_KERNELS(AXI2, F16, I16)
REDUCEPROD_KERNELS(AXI2, F16, I8)
REDUCEPROD_KERNELS(AXI2, F16, U8)
REDUCEPROD_KERNELS(AXI2, I16, I16)
REDUCEPROD_KERNELS(AXI2, I8,  I8)
REDUCEPROD_KERNELS(AXI2, U8,  U8)
REDUCEPROD_KERNELS(AXI2, I16, F16)
REDUCEPROD_KERNELS(AXI2, I8,  F16)
REDUCEPROD_KERNELS(AXI2, U8,  F16)
REDUCEPROD_KERNELS(AXI2, BF16,  BF16)

REDUCEPROD_KERNELS_2D(AXI0, F16, F16)
REDUCEPROD_KERNELS_2D(AXI0, F16, I16)
REDUCEPROD_KERNELS_2D(AXI0, F16, I8)
REDUCEPROD_KERNELS_2D(AXI0, F16, U8)
REDUCEPROD_KERNELS_2D(AXI0, I16, I16)
REDUCEPROD_KERNELS_2D(AXI0, I8,  I8)
REDUCEPROD_KERNELS_2D(AXI0, U8,  U8)
REDUCEPROD_KERNELS_2D(AXI0, I16, F16)
REDUCEPROD_KERNELS_2D(AXI0, I8,  F16)
REDUCEPROD_KERNELS_2D(AXI0, U8,  F16)
REDUCEPROD_KERNELS_2D(AXI0, BF16,  BF16)

REDUCEPROD_KERNELS_2D(AXI1, F16, F16)
REDUCEPROD_KERNELS_2D(AXI1, F16, I16)
REDUCEPROD_KERNELS_2D(AXI1, F16, I8)
REDUCEPROD_KERNELS_2D(AXI1, F16, U8)
REDUCEPROD_KERNELS_2D(AXI1, I16, I16)
REDUCEPROD_KERNELS_2D(AXI1, I8,  I8)
REDUCEPROD_KERNELS_2D(AXI1, U8,  U8)
REDUCEPROD_KERNELS_2D(AXI1, I16, F16)
REDUCEPROD_KERNELS_2D(AXI1, I8,  F16)
REDUCEPROD_KERNELS_2D(AXI1, U8,  F16)
REDUCEPROD_KERNELS_2D(AXI1, BF16,  BF16)

#define REDUCEPROD_KERENLS_NAME(AXI_INDEX, SRC_TYPE, DST_TYPE, INSTR) \
    &vxReduceProd_##AXI_INDEX##_##SRC_TYPE##to##DST_TYPE##_##INSTR##Kernel,

vx_kernel_description_t* vx_kernel_REDUCEPROD_INTERNAL_list[] =
{
    &vxReduceProd_internal_CPU,
    REDUCEPROD_KERENLS_NAME(AXI0, F16, F16, )
    REDUCEPROD_KERENLS_NAME(AXI0, F16, I16, )
    REDUCEPROD_KERENLS_NAME(AXI0, F16, I8, )
    REDUCEPROD_KERENLS_NAME(AXI0, F16, U8, )
    REDUCEPROD_KERENLS_NAME(AXI0, I16, I16, )
    REDUCEPROD_KERENLS_NAME(AXI0, I8,  I8, )
    REDUCEPROD_KERENLS_NAME(AXI0, U8,  U8, )
    REDUCEPROD_KERENLS_NAME(AXI0, I16, F16, )
    REDUCEPROD_KERENLS_NAME(AXI0, I8,  F16, )
    REDUCEPROD_KERENLS_NAME(AXI0, U8,  F16, )
    REDUCEPROD_KERENLS_NAME(AXI0, BF16,  BF16, )
    REDUCEPROD_KERENLS_NAME(AXI0, F16, F16, 2D_)
    REDUCEPROD_KERENLS_NAME(AXI0, F16, I16, 2D_)
    REDUCEPROD_KERENLS_NAME(AXI0, F16, I8,  2D_)
    REDUCEPROD_KERENLS_NAME(AXI0, F16, U8,  2D_)
    REDUCEPROD_KERENLS_NAME(AXI0, I16, I16, 2D_)
    REDUCEPROD_KERENLS_NAME(AXI0, I8,  I8,  2D_)
    REDUCEPROD_KERENLS_NAME(AXI0, U8,  U8,  2D_)
    REDUCEPROD_KERENLS_NAME(AXI0, I16, F16, 2D_)
    REDUCEPROD_KERENLS_NAME(AXI0, I8,  F16, 2D_)
    REDUCEPROD_KERENLS_NAME(AXI0, U8,  F16, 2D_)
    REDUCEPROD_KERENLS_NAME(AXI0, BF16,  BF16, 2D_)
    REDUCEPROD_KERENLS_NAME(AXI1, F16, F16, )
    REDUCEPROD_KERENLS_NAME(AXI1, F16, I16, )
    REDUCEPROD_KERENLS_NAME(AXI1, F16, I8, )
    REDUCEPROD_KERENLS_NAME(AXI1, F16, U8, )
    REDUCEPROD_KERENLS_NAME(AXI1, I16, I16, )
    REDUCEPROD_KERENLS_NAME(AXI1, I8,  I8, )
    REDUCEPROD_KERENLS_NAME(AXI1, U8,  U8, )
    REDUCEPROD_KERENLS_NAME(AXI1, I16, F16, )
    REDUCEPROD_KERENLS_NAME(AXI1, I8,  F16, )
    REDUCEPROD_KERENLS_NAME(AXI1, U8,  F16, )
    REDUCEPROD_KERENLS_NAME(AXI1, BF16,  BF16, )
    REDUCEPROD_KERENLS_NAME(AXI1, F16, F16, 2D_)
    REDUCEPROD_KERENLS_NAME(AXI1, F16, I16, 2D_)
    REDUCEPROD_KERENLS_NAME(AXI1, F16, I8,  2D_)
    REDUCEPROD_KERENLS_NAME(AXI1, F16, U8,  2D_)
    REDUCEPROD_KERENLS_NAME(AXI1, I16, I16, 2D_)
    REDUCEPROD_KERENLS_NAME(AXI1, I8,  I8,  2D_)
    REDUCEPROD_KERENLS_NAME(AXI1, U8,  U8,  2D_)
    REDUCEPROD_KERENLS_NAME(AXI1, I16, F16, 2D_)
    REDUCEPROD_KERENLS_NAME(AXI1, I8,  F16, 2D_)
    REDUCEPROD_KERENLS_NAME(AXI1, U8,  F16, 2D_)
    REDUCEPROD_KERENLS_NAME(AXI1, BF16,  BF16, 2D_)
    REDUCEPROD_KERENLS_NAME(AXI2, F16, F16, )
    REDUCEPROD_KERENLS_NAME(AXI2, F16, I16, )
    REDUCEPROD_KERENLS_NAME(AXI2, F16, I8, )
    REDUCEPROD_KERENLS_NAME(AXI2, F16, U8, )
    REDUCEPROD_KERENLS_NAME(AXI2, I16, I16, )
    REDUCEPROD_KERENLS_NAME(AXI2, I8,  I8, )
    REDUCEPROD_KERENLS_NAME(AXI2, U8,  U8, )
    REDUCEPROD_KERENLS_NAME(AXI2, I16, F16, )
    REDUCEPROD_KERENLS_NAME(AXI2, I8,  F16, )
    REDUCEPROD_KERENLS_NAME(AXI2, U8,  F16, )
    REDUCEPROD_KERENLS_NAME(AXI2, BF16,  BF16, )
    NULL
};

#ifdef __cplusplus
}
#endif
