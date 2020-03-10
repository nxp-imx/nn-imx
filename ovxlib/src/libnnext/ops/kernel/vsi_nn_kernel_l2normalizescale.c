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
#include <math.h>
#include <stdint.h>

#include "vsi_nn_platform.h"

#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_math.h"
#include "client/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"

/*************************************L2NormalizeScale_CPU**************************************/
vsi_status VX_CALLBACK vxL2NormalizeScaleValidator
    (
    vx_node node,
    const vx_reference parameters[],
    uint32_t num,
    vx_meta_format metas[]
)
{
    vsi_status status = VX_SUCCESS;
    vx_parameter param = NULL;
    uint32_t index = 0;

    for(index = 0; index < num; index++)
    {
        // Validator
        if(index == 0) //tensor
        {
            vx_tensor input_tensor;
            param = vxGetParameterByIndex(node, index);
            if(param != NULL)
            {
                vsi_nn_tensor_attr_t attr;

                status |= vxQueryParameter(param, VX_PARAMETER_REF, &input_tensor,
                    sizeof(input_tensor));
                if (status != VX_SUCCESS)
                {
                    VSILOGE("vxQueryParameter failure! at line %d\n", __LINE__);
                }

                memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
                status  = vsi_nn_vxGetTensorAttr(input_tensor, &attr);
                if (status != VX_SUCCESS)
                {
                    VSILOGE("vsi_nn_vxGetTensorAttr  failure! at line %d\n", __LINE__);
                }
                status |= vxReleaseTensor(&input_tensor);

                status |= vxReleaseParameter(&param);
            }
            else
            {
                status |= VX_ERROR_INVALID_VALUE;
            }

        }
        else if(index == 1) //tensor
        {
            vx_tensor input_tensor;
            param = vxGetParameterByIndex(node, index);
            if(param != NULL)
            {
                vsi_enum     data_format;
                vsi_nn_tensor_attr_t attr;

                status |= vxQueryParameter(param, VX_PARAMETER_REF, &input_tensor,
                    sizeof(input_tensor));
                if (status != VX_SUCCESS)
                {
                    VSILOGE("vxQueryParameter failure! at line %d\n", __LINE__);
                }
                memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
                status  = vsi_nn_vxGetTensorAttr(input_tensor, &attr);
                if (status != VX_SUCCESS)
                {
                    VSILOGE("vsi_nn_vxGetTensorAttr  failure! at line %d\n", __LINE__);
                }
                data_format = attr.dtype.vx_type;
                if (data_format != VSI_NN_TYPE_FLOAT16)
                    status |= VX_ERROR_INVALID_TYPE;

                status |= vxReleaseTensor(&input_tensor);

                status |= vxReleaseParameter(&param);
            }
            else
            {
                status |= VX_ERROR_INVALID_VALUE;
            }

        }
        else if(index == 2) //tensor
        {
        }
        else if(index == 3) //scalar
        {
            param = vxGetParameterByIndex(node, index);
            if(param != NULL)
            {
                vx_scalar scalar = NULL;
                status |= vxQueryParameter(param, VX_PARAMETER_REF, &scalar, sizeof(scalar));
                if(status == VX_SUCCESS)
                {
                    // VX_SCALAR_TYPE
                    vsi_enum type = 0;
                    status |= vxQueryScalar(scalar, VX_SCALAR_TYPE, &type, sizeof(type));
                    if (type != VSI_NN_TYPE_INT32)
                        status = VX_ERROR_INVALID_TYPE;

                    status |= vxReleaseScalar(&scalar);
                }
                else
                {
                    status |= VX_ERROR_INVALID_VALUE;
                }
                status |= vxReleaseParameter(&param);
            }
            else
            {
                status |= VX_ERROR_INVALID_VALUE;
            }
        }
        else
        {
            VSILOGE("Validator  failure! at line %d,invalid index = %d\n", __LINE__,index);
        }

        if(status < 0)
        {
            VSILOGE("Validator  failure! at line %d,index = %d, status = %d\n",
                __LINE__,index,status);
        }
    }
    return status;
}

#define TENSOR_NUM_INPUT  (L2NORMSACLE_INPUTS_COUNT)
#define TENSOR_NUM_OUTPUT (L2NORMSACLE_OUTPUTS_COUNT)
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

static vsi_status VX_CALLBACK vxL2NormalizeScaleKernel
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
    vx_tensor   tensor[TENSOR_NUM] = {NULL};
    uint32_t    stride_size[TENSOR_NUM][VSI_NN_MAX_DIM_NUM];
    vx_context   context                        = vxGetContext((vx_reference)node);
    vx_uint32  size[4];
    vx_uint32  dims, innerSize, outerSize, axisSize;
    vx_uint32  outer, inner, i, index;
    vx_float32 l2Value, tmpValue;
    int32_t axis;
    vx_float32 rsqrt = 0.0f, scaleValue = 0.0f;
    vx_float32 epsilon = (float)10e-12;

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

    if(paramNum != 4)
    {
        VSILOGE("Error ParamNum input %d \n", paramNum);
        status = VX_ERROR_INVALID_PARAMETERS;
        goto final;
    }

    dims = attr[L2NORMSACLE_INPUT].dim_num;
    size[0] = attr[L2NORMSACLE_INPUT].size[0];
    size[1] = dims > 1 ? attr[L2NORMSACLE_INPUT].size[1] : 1;
    size[2] = dims > 2 ? attr[L2NORMSACLE_INPUT].size[2] : 1;
    size[3] = dims > 3 ? attr[L2NORMSACLE_INPUT].size[3] : 1;
    axisSize =  attr[L2NORMSACLE_INPUT].size[axis];

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
        VSILOGE("Input tensor error dimension[%u]\n", dims);
        status = VX_ERROR_INVALID_DIMENSION;
        goto final;
    }

    for (outer = 0; outer < outerSize; ++outer) {
        for (inner = 0; inner < innerSize; ++inner) {
            vx_float32 sum = 0.0f;
            for (i = 0; i < axisSize; ++i) {
                index    = (outer * axisSize + i) * innerSize + inner;
                tmpValue = vsi_nn_DtypeToFloat32_Ex(buffer_ptr[L2NORMSACLE_INPUT],
                                                   index, &attr[L2NORMSACLE_INPUT].dtype);
                sum += tmpValue * tmpValue;
            }
            rsqrt = 1.0f / sqrtf(gcoMATH_MAX(sum, epsilon));
            for (i = 0; i < axisSize; ++i) {
                index    = (outer * axisSize + i) * innerSize + inner;
                tmpValue = vsi_nn_DtypeToFloat32_Ex(buffer_ptr[L2NORMSACLE_INPUT],
                                                   index, &attr[L2NORMSACLE_INPUT].dtype);
                scaleValue = vsi_nn_DtypeToFloat32_Ex(buffer_ptr[L2NORMSACLE_INPUT1],
                                                   i, &attr[L2NORMSACLE_INPUT1].dtype);
                l2Value = tmpValue * rsqrt * scaleValue;
                vsi_nn_Float32ToDtype_Ext(l2Value, buffer_ptr[L2NORMSACLE_INPUTS_COUNT + L2NORMSACLE_OUTPUT],
                            index, &attr[L2NORMSACLE_INPUTS_COUNT + L2NORMSACLE_OUTPUT].dtype);
            }
        }
    }

    for( i = TENSOR_NUM_INPUT; i < TENSOR_NUM; i ++ )
    {
        if (buffer_ptr[i])
        {
            status = vsi_nn_copy_tensor_patch(tensor[i], &attr[i], buffer_ptr[i], VX_WRITE_ONLY);
        }

        if (user_addr[i]) vxReleaseTensorAddressing(&(user_addr[i]));
    }
final:
    for( i = 0; i < TENSOR_NUM; i ++ )
    {
        if (buffer_ptr[i]) free(buffer_ptr[i]);

        if (user_addr[i]) vxReleaseTensorAddressing(&(user_addr[i]));
    }
    return status;
} /* _VX_KERNEL_FUNC_KERNEL() */

static vx_param_description_t vxL2NormalizeScaleKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED}
};

/*************************************L2NormalizeScale_VX**********************************/
vsi_status VX_CALLBACK vxL2NormScale_SumRsqrtInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    uint32_t paraNum
    )
{
    // Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
    vx_kernel_execution_parameters_t shaderParam = {
        2,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread
    uint32_t UniFp16MulLo_dp4x4[16] = {
        0x01010101, // TCfg
        0x00000000, // ASelt
        0x00010000, 0x00030002, // ABin
        0x01010101, // BSelt
        0x00010000, 0x00030002, // BBin
        0x00000400, // AccumType, ConstantType, and PostShift
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
    };
    uint32_t UniFp16MulHi_dp4x4[16] = {
        0x01010101, // TCfg
        0x00000000, // ASelt
        0x00050004, 0x00070006, // ABin
        0x01010101, // BSelt
        0x00050004, 0x00070006, // BBin
        0x00000400, // AccumType, ConstantType, and PostShift
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
    };
    uint32_t uniIntegerSquareLo_4x4[16] = {
        0x01010101, // TCfg
        0x00000000, // ASelt
        0x00010000, 0x00030002, // ABin
        0x00000000, // BSelt
        0x00010000, 0x00030002, // BBin
        0x00000400, // AccumType, ConstantType, and PostShift
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
    };
    uint32_t uniIntegerSquareHi_4x4[16] = {
        0x01010101, // TCfg
        0x00000000, // ASelt
        0x00050004, 0x00070006, // ABin
        0x00000000, // BSelt
        0x00050004, 0x00070006, // BBin
        0x00000400, // AccumType, ConstantType, and PostShift
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
    };
    vx_uint32 uniDataSquareAddU32Lo_4x4[16] = {
        0x0d0d0d0d, // TCfg
        0x04040404, // ASelt
        0x00110000, 0x00330022, // ABin
        0x00000000, // BSelt
        0x00010000, 0x00030002, // BBin
        0x00005400, // AccumType, ConstantType, and PostShift
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
    };
    vx_uint32 uniDataSquareAddU32Hi_4x4[16] = {
        0x0d0d0d0d, // TCfg
        0x04040404, // ASelt
        0x00150004, 0x00370026, // ABin
        0x00000000, // BSelt
        0x00050004, 0x00070006, // BBin
        0x00005400, // AccumType, ConstantType, and PostShift
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
    };
    vx_uint32 uniUInt8SquareLo_4x4[16] = {
        0x69696969, // TCfg
        0x40404040, // ASelt
        0x01110000, 0x03330222, // ABin
        0x54545454, // BSelt
        0x00010000, 0x00030002, // BBin
        0x00000400, // AccumType, ConstantType, and PostShift
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
    };
    vx_uint32 uniUInt8SquareHi_4x4[16] = {
        0x69696969, // TCfg
        0x40404040, // ASelt
        0x05550444, 0x07770666, // ABin
        0x54545454, // BSelt
        0x00050004, 0x00070006, // BBin
        0x00000400, // AccumType, ConstantType, and PostShift
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
    };
    vx_uint32 uniSumSqrt_16x1[16] = {
        0x55555555, // TCfg
        0x55550000, // ASelt
        0x76543210, 0x76543210, // ABin
        0x55550000, // BSelt
        0x76543210, 0x76543210, // BBin
        0x00000400, // AccumType, ConstantType, and PostShift
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
    };
    vx_uint32 uniSumAll_16x1[16] = {
        0x55555555, // TCfg
        0x55550000, // ASelt
        0x76543210, 0x76543210, // ABin
        0xaaaaaaaa, // BSelt
        0x00000000, 0x00000000, // BBin
        0x00000400, // AccumType, ConstantType, and PostShift
        0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
    };

    vsi_status status = VX_SUCCESS;

    vx_tensor input         = (vx_tensor)paramObj[0];
    int32_t   input_size[4] = {1, 1, 1, 1};
    vsi_enum  dataFormat;
    int8_t    fixPointPos   = 0;
    int32_t   inputZP       = 0;
    float     inputScale    = 1.0f;
    float     r_inputScale  = 1.0f;
    int32_t   axis          = 1;
    vsi_nn_tensor_attr_t attr;
    uint32_t i;
    uint32_t input_dims;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    status  = vsi_nn_vxGetTensorAttr(input, &attr);
    if (status != VX_SUCCESS)
    {
        VSILOGE("vsi_nn_vxGetTensorAttr  failure! at line %d\n", __LINE__);
        return status;
    }
    input_dims  = attr.dim_num;
    dataFormat = attr.dtype.vx_type;
    for (i = 0; i < input_dims; i++)
    {
        input_size[i] = attr.size[i];
    }
    fixPointPos = attr.dtype.fl;
    inputZP     = attr.dtype.zero_point;
    inputScale  = attr.dtype.scale;

    vxCopyScalar((vx_scalar)paramObj[2], &(axis), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    if(dataFormat == VSI_NN_TYPE_INT8 || dataFormat == VSI_NN_TYPE_INT16)
    {
        if (fixPointPos >= 0)
            inputScale = 1.0f / (float) (1 << fixPointPos);
        else
            inputScale = (float) (1 << -fixPointPos);
    }

    r_inputScale = 1.0f / inputScale;

    if (1 == axis)
    {
        shaderParam.globalWorkOffset[0] = 0;
        shaderParam.globalWorkOffset[1] = 0;
        shaderParam.globalWorkScale[0]  = 8;
        shaderParam.globalWorkScale[1]  = 1;
        shaderParam.globalWorkScale[2]  = 1;
        shaderParam.globalWorkSize[0]   = gcmALIGN((input_size[0] + shaderParam.globalWorkScale[0] - 1)
            / shaderParam.globalWorkScale[0], 4);
        shaderParam.globalWorkSize[1] = 1;
    }
    else if (0 == axis)
    {
        shaderParam.globalWorkOffset[0] = 0;
        shaderParam.globalWorkOffset[1] = 0;
        shaderParam.globalWorkScale[0]  = 16;
        shaderParam.globalWorkScale[1]  = 1;
        shaderParam.localWorkSize[0]    = 16;
        shaderParam.localWorkSize[1]    = 1;
        shaderParam.globalWorkSize[0]   = 16;
        shaderParam.globalWorkSize[1]   = input_size[1];
    }
    else
    {
        VSILOGE("Input tensor error dimension[%u]\n", axis);
        return VX_ERROR_INVALID_DIMENSION;
    }

    if (1 == axis)
    {
        vxSetNodeUniform(nodObj, "L2NorS_depth", 1, &input_size[1]);
        if(dataFormat == VSI_NN_TYPE_FLOAT16)
        {
            vxSetNodeUniform(nodObj, "UniFp16MulLo_dp4x4", 1, UniFp16MulLo_dp4x4);
            vxSetNodeUniform(nodObj, "UniFp16MulHi_dp4x4", 1, UniFp16MulHi_dp4x4);
        }
        else if(dataFormat == VSI_NN_TYPE_INT8)
        {
            vxSetNodeUniform(nodObj, "r_inputScale", 1, &r_inputScale);
            vxSetNodeUniform(nodObj, "uniDataSquareAddU32Lo_4x4", 1, uniDataSquareAddU32Lo_4x4);
            vxSetNodeUniform(nodObj, "uniDataSquareAddU32Hi_4x4", 1, uniDataSquareAddU32Hi_4x4);
        }
        else if(dataFormat == VSI_NN_TYPE_INT16)
        {
            vxSetNodeUniform(nodObj, "r_inputScale", 1, &r_inputScale);
            vxSetNodeUniform(nodObj, "uniIntegerSquareLo_4x4", 1, uniIntegerSquareLo_4x4);
            vxSetNodeUniform(nodObj, "uniIntegerSquareHi_4x4", 1, uniIntegerSquareHi_4x4);
        }
        else if(dataFormat == VSI_NN_TYPE_UINT8)
        {
            vxSetNodeUniform(nodObj, "r_inputScale", 1, &r_inputScale);
            vxSetNodeUniform(nodObj, "inputZP", 1, &inputZP);
            vxSetNodeUniform(nodObj, "uniUInt8SquareLo_4x4", 1, uniUInt8SquareLo_4x4);
            vxSetNodeUniform(nodObj, "uniUInt8SquareHi_4x4", 1, uniUInt8SquareHi_4x4);
        }
    }
    else if (0 == axis)
    {
        int32_t inputWidth, inputWidthCount, inputWidthRemain256;
        inputWidth          = input_size[0];
        inputWidthRemain256 = input_size[0] % 256;
        inputWidthCount     = input_size[0] / 256;
        vxSetNodeUniform(nodObj, "inputWidth", 1, &inputWidth);
        vxSetNodeUniform(nodObj, "inputWidthRemain256", 1, &inputWidthRemain256);
        vxSetNodeUniform(nodObj, "inputWidthCount", 1, &inputWidthCount);
        vxSetNodeUniform(nodObj, "uniSumSqrt_16x1", 1, uniSumSqrt_16x1);
        if (dataFormat == VSI_NN_TYPE_INT16 || dataFormat == VSI_NN_TYPE_INT8)
        {
            vxSetNodeUniform(nodObj, "r_inputScale", 1, &r_inputScale);
        }
        else if(dataFormat == VSI_NN_TYPE_UINT8)
        {
            float zP2x = 2 * (float)inputZP;
            float zpSqrt16x =  16 * (float)inputZP * (float)inputZP;
            vxSetNodeUniform(nodObj, "r_inputScale", 1, &r_inputScale);
            vxSetNodeUniform(nodObj, "zP2x", 1, &zP2x);
            vxSetNodeUniform(nodObj, "zpSqrt16x", 1, &zpSqrt16x);
            vxSetNodeUniform(nodObj, "uniSumAll_16x1", 1, uniSumAll_16x1);
            vxSetNodeUniform(nodObj, "inputZP", 1, &inputZP);
        }
    }

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));
    return VX_SUCCESS;
}

static vx_param_description_t vxL2NormScale_SumRsqrtKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED}
};

vsi_status VX_CALLBACK vxL2NormScale_MulScaleInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    uint32_t paraNum
    )
{
    // Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
    vx_kernel_execution_parameters_t shaderParam = {
        2,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vsi_status status = VX_SUCCESS;

    vsi_enum    inputFormat;
    vsi_enum    outputFormat;
    vx_tensor   input           = (vx_tensor)paramObj[0];
    vx_tensor   output          = (vx_tensor)paramObj[3];
    int32_t     input_size[DIM_SIZE]   = {1, 1, 1, 1};
    int8_t      srcFixPointPos  = 0;
    int32_t     inputZP         = 0;
    int8_t      dstFixPointPos  = 0;
    int32_t     outputZP        = 0;
    float       inputScale      = 1.0f;
    float       outputScale     = 1.0f;
    int32_t     axis            = 1;
    vsi_nn_tensor_attr_t attr[2];
    uint32_t i;
    uint32_t input_dims;

    memset(&attr[0], 0, sizeof(vsi_nn_tensor_attr_t));
    memset(&attr[1], 0, sizeof(vsi_nn_tensor_attr_t));
    status  = vsi_nn_vxGetTensorAttr(input, &attr[0]);
    status |= vsi_nn_vxGetTensorAttr(output, &attr[1]);
    if (status != VX_SUCCESS)
    {
        VSILOGE("vsi_nn_vxGetTensorAttr failure! at line %d\n", __LINE__);
        return status;
    }

    input_dims  = attr[0].dim_num;
    inputFormat = attr[0].dtype.vx_type;
    for (i = 0; i < input_dims; i++)
    {
        input_size[i] = attr[0].size[i];
    }
    srcFixPointPos = attr[0].dtype.fl;
    inputZP        = attr[0].dtype.zero_point;
    inputScale     = attr[0].dtype.scale;
    outputFormat   = attr[1].dtype.vx_type;
    outputZP       = attr[1].dtype.zero_point;
    outputScale    = attr[1].dtype.scale;
    dstFixPointPos = attr[1].dtype.fl;

    vxCopyScalar((vx_scalar)paramObj[4], &(axis), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    if(inputFormat == VSI_NN_TYPE_INT8 || inputFormat == VSI_NN_TYPE_INT16)
    {
        if (srcFixPointPos >= 0)
            inputScale = 1.0f / (float) (1 << srcFixPointPos);
        else
            inputScale = (float) (1 << -srcFixPointPos);

        inputZP = 0;
    }
    else if(inputFormat == VSI_NN_TYPE_FLOAT16)
    {
        inputScale     = 1.0f;
        inputZP        = 0;
    }

    if(outputFormat == VSI_NN_TYPE_INT8 || outputFormat == VSI_NN_TYPE_INT16)
    {
        if (dstFixPointPos >= 0)
            outputScale = (float) (1 << dstFixPointPos);
        else
            outputScale = 1.0f / (float) (1 << -dstFixPointPos);

        outputZP = 0;
    }
    else if(outputFormat == VSI_NN_TYPE_FLOAT16)
    {
        outputScale    = 1.0f;
        outputZP       = 0;
    }

    if (1 == axis)
    {
        shaderParam.globalWorkOffset[0] = 0;
        shaderParam.globalWorkOffset[1] = 0;
        shaderParam.globalWorkScale[0]  = 8;
        shaderParam.globalWorkScale[1]  = 1;
        shaderParam.globalWorkSize[0]   = gcmALIGN((input_size[0] + shaderParam.globalWorkScale[0] - 1)
            / shaderParam.globalWorkScale[0], 4);
        shaderParam.globalWorkSize[1]   = 1;
    }
    else if (0 == axis)
    {
        shaderParam.globalWorkOffset[0] = 0;
        shaderParam.globalWorkOffset[1] = 0;
        shaderParam.globalWorkScale[0]  = 8;
        shaderParam.globalWorkScale[1]  = 1;
        shaderParam.localWorkSize[0]    = 16;
        shaderParam.localWorkSize[1]    = 1;
        shaderParam.globalWorkSize[0]   = 16;
        shaderParam.globalWorkSize[1]   = input_size[1];
    }
    else
    {
        VSILOGE("Input tensor error dimension[%u]\n", axis);
        return VX_ERROR_INVALID_DIMENSION;
    }

    {
        vx_float32 IntergerScale = inputScale;
        vx_float32 output_ZP      = (vx_float32)outputZP;
        vx_uint32 uniExtact8Bin_2x8[16] = {
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniDataSubZPtoFp32Part0_4x4[16] = {
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000 // Constant
        };
        vx_uint32 uniDataSubZPtoFp32Part1_4x4[16] = {
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00050004, 0x00070006, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000 // Constant
        };
        vx_uint32 uniExtractHalf8_2x8[16] = {
            0x11111111, // TCfg
            0x11110000, // ASelt
            0x06040200, 0x06040200, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
        };
        vx_uint32 uniFp16toFp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };
        vx_uint32 uniFp16toFp32Hi_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };


        if (outputFormat == VSI_NN_TYPE_UINT8)
            IntergerScale = IntergerScale / outputScale;
        else
            IntergerScale = IntergerScale * outputScale;

        vxSetNodeUniform(nodObj, "IntergerScale", 1, &IntergerScale);
        vxSetNodeUniform(nodObj, "inputZP", 1, &inputZP);
        vxSetNodeUniform(nodObj, "output_ZP", 1, &output_ZP);
        vxSetNodeUniform(nodObj, "L2NorS_depth", 1, &input_size[1]);
        vxSetNodeUniform(nodObj, "uniDataSubZPtoFp32Part0_4x4", 1, uniDataSubZPtoFp32Part0_4x4);
        vxSetNodeUniform(nodObj, "uniDataSubZPtoFp32Part1_4x4", 1, uniDataSubZPtoFp32Part1_4x4);
        vxSetNodeUniform(nodObj, "uniFp16toFp32_4x4", 1, uniFp16toFp32_4x4);
        vxSetNodeUniform(nodObj, "uniFp16toFp32Hi_4x4", 1, uniFp16toFp32Hi_4x4);

        if(outputFormat == VSI_NN_TYPE_FLOAT16)
            vxSetNodeUniform(nodObj, "uniExtact8Bin_2x8", 1, uniExtractHalf8_2x8);
        else
            vxSetNodeUniform(nodObj, "uniExtact8Bin_2x8", 1, uniExtact8Bin_2x8);

        if (0 == axis)
        {
            int32_t inputWidth, inputWidthCount, inputWidthRemain128;
            inputWidth          = input_size[0];
            inputWidthRemain128 = input_size[0] % 128;
            inputWidthCount     = input_size[0] / 128;
            vxSetNodeUniform(nodObj, "inputWidth", 1, &inputWidth);
            vxSetNodeUniform(nodObj, "inputWidthRemain128", 1, &inputWidthRemain128);
            vxSetNodeUniform(nodObj, "inputWidthCount", 1, &inputWidthCount);
        }

    }

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));
    return VX_SUCCESS;
}

static vx_param_description_t vxL2NormScale_MulScaleKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED}
};

#ifdef __cplusplus
extern "C" {
#endif
vx_kernel_description_t vxL2NormalizeScaleKernelInfo_CPU =
{
    VX_KERNEL_ENUM_L2NORMALIZESCALE,
    VX_KERNEL_NAME_L2NORMALIZESCALE,
    vxL2NormalizeScaleKernel,
    vxL2NormalizeScaleKernelParam,
    (sizeof(vxL2NormalizeScaleKernelParam) / sizeof(vxL2NormalizeScaleKernelParam[0])),
    vxL2NormalizeScaleValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

#define L2NORMSACLE_SQRTSUM_KERNELS_2D(AXI_INDEX, SRC_TYPE) \
    vx_kernel_description_t vxL2NormScale_SumRsqrt_##AXI_INDEX##_##SRC_TYPE##_2D_Kernel = \
{ \
    VX_KERNEL_ENUM_L2NORMSCALE_SUMRSQRT, \
    VX_KERNEL_NAME_L2NORMSCALE_SUMRSQRT_##AXI_INDEX##_##SRC_TYPE##_2D, \
    NULL, \
    vxL2NormScale_SumRsqrtKernelParam, \
    (sizeof(vxL2NormScale_SumRsqrtKernelParam) / sizeof(vxL2NormScale_SumRsqrtKernelParam[0])), \
    vsi_nn_KernelValidator, \
    NULL, \
    NULL, \
    vxL2NormScale_SumRsqrtInitializer, \
    vsi_nn_KernelDeinitializer \
};

L2NORMSACLE_SQRTSUM_KERNELS_2D(AXI1, F16)
L2NORMSACLE_SQRTSUM_KERNELS_2D(AXI1, I8)
L2NORMSACLE_SQRTSUM_KERNELS_2D(AXI1, U8)
L2NORMSACLE_SQRTSUM_KERNELS_2D(AXI1, I16)
L2NORMSACLE_SQRTSUM_KERNELS_2D(AXI0, F16)
L2NORMSACLE_SQRTSUM_KERNELS_2D(AXI0, I8)
L2NORMSACLE_SQRTSUM_KERNELS_2D(AXI0, U8)
L2NORMSACLE_SQRTSUM_KERNELS_2D(AXI0, I16)


#define L2NORMSACLE_MULSCALE_KERNELS_2D(AXI_INDEX, SRC_TYPE, DST_TYPE) \
    vx_kernel_description_t vxL2NormScale_MulScale_##AXI_INDEX##_##SRC_TYPE##to##DST_TYPE##_2D_Kernel = \
{ \
    VX_KERNEL_ENUM_L2NORMSCALE_MULSCALE, \
    VX_KERNEL_NAME_L2NORMSCALE_##AXI_INDEX##_##SRC_TYPE##TO##DST_TYPE##_2D, \
    NULL, \
    vxL2NormScale_MulScaleKernelParam, \
    (sizeof(vxL2NormScale_MulScaleKernelParam) / sizeof(vxL2NormScale_MulScaleKernelParam[0])), \
    vsi_nn_KernelValidator, \
    NULL, \
    NULL, \
    vxL2NormScale_MulScaleInitializer, \
    vsi_nn_KernelDeinitializer \
};

L2NORMSACLE_MULSCALE_KERNELS_2D(AXI1, F16, F16)
L2NORMSACLE_MULSCALE_KERNELS_2D(AXI1, I8, I8)
L2NORMSACLE_MULSCALE_KERNELS_2D(AXI1, I8, F16)
L2NORMSACLE_MULSCALE_KERNELS_2D(AXI1, U8, U8)
L2NORMSACLE_MULSCALE_KERNELS_2D(AXI1, U8, F16)
L2NORMSACLE_MULSCALE_KERNELS_2D(AXI1, I16, I16)
L2NORMSACLE_MULSCALE_KERNELS_2D(AXI1, I16, F16)
L2NORMSACLE_MULSCALE_KERNELS_2D(AXI0, F16, F16)
L2NORMSACLE_MULSCALE_KERNELS_2D(AXI0, I8, I8)
L2NORMSACLE_MULSCALE_KERNELS_2D(AXI0, I8, F16)
L2NORMSACLE_MULSCALE_KERNELS_2D(AXI0, U8, U8)
L2NORMSACLE_MULSCALE_KERNELS_2D(AXI0, U8, F16)
L2NORMSACLE_MULSCALE_KERNELS_2D(AXI0, I16, I16)
L2NORMSACLE_MULSCALE_KERNELS_2D(AXI0, I16, F16)


#define L2NORMSACLE_SQRTSUM_KERENLS_NAME(AXI_INDEX, SRC_TYPE, INSTR) \
    &vxL2NormScale_SumRsqrt_##AXI_INDEX##_##SRC_TYPE##_##INSTR##Kernel,

#define L2NORMSACLE_MULSCALE_KERENLS_NAME(AXI_INDEX, SRC_TYPE, DST_TYPE, INSTR) \
    &vxL2NormScale_MulScale_##AXI_INDEX##_##SRC_TYPE##to##DST_TYPE##_##INSTR##Kernel,


vx_kernel_description_t * vx_kernel_L2NORMALIZESCALE_list[] =
{
    &vxL2NormalizeScaleKernelInfo_CPU,
    L2NORMSACLE_SQRTSUM_KERENLS_NAME(AXI1, F16, 2D_)
    L2NORMSACLE_SQRTSUM_KERENLS_NAME(AXI1, I8, 2D_)
    L2NORMSACLE_SQRTSUM_KERENLS_NAME(AXI1, U8, 2D_)
    L2NORMSACLE_SQRTSUM_KERENLS_NAME(AXI1, I16, 2D_)
    L2NORMSACLE_MULSCALE_KERENLS_NAME(AXI1, F16, F16, 2D_)
    L2NORMSACLE_MULSCALE_KERENLS_NAME(AXI1, I8, I8, 2D_)
    L2NORMSACLE_MULSCALE_KERENLS_NAME(AXI1, I8, F16, 2D_)
    L2NORMSACLE_MULSCALE_KERENLS_NAME(AXI1, U8, U8, 2D_)
    L2NORMSACLE_MULSCALE_KERENLS_NAME(AXI1, U8, F16, 2D_)
    L2NORMSACLE_MULSCALE_KERENLS_NAME(AXI1, I16, I16, 2D_)
    L2NORMSACLE_MULSCALE_KERENLS_NAME(AXI1, I16, F16, 2D_)
    L2NORMSACLE_SQRTSUM_KERENLS_NAME(AXI0, F16, 2D_)
    L2NORMSACLE_SQRTSUM_KERENLS_NAME(AXI0, I8, 2D_)
    L2NORMSACLE_SQRTSUM_KERENLS_NAME(AXI0, U8, 2D_)
    L2NORMSACLE_SQRTSUM_KERENLS_NAME(AXI0, I16, 2D_)
    L2NORMSACLE_MULSCALE_KERENLS_NAME(AXI0, F16, F16, 2D_)
    L2NORMSACLE_MULSCALE_KERENLS_NAME(AXI0, I8, I8, 2D_)
    L2NORMSACLE_MULSCALE_KERENLS_NAME(AXI0, I8, F16, 2D_)
    L2NORMSACLE_MULSCALE_KERENLS_NAME(AXI0, U8, U8, 2D_)
    L2NORMSACLE_MULSCALE_KERENLS_NAME(AXI0, U8, F16, 2D_)
    L2NORMSACLE_MULSCALE_KERENLS_NAME(AXI0, I16, I16, 2D_)
    L2NORMSACLE_MULSCALE_KERENLS_NAME(AXI0, I16, F16, 2D_)
    NULL
};
#ifdef __cplusplus
}
#endif
