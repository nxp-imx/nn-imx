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
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_math.h"
#include "vsi_nn_log.h"
#include "client/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"
#include "utils/vsi_nn_dtype_util.h"

#define TENSOR_NUM_INPUT  (PRELLU_INPUTS_COUNT)
#define TENSOR_NUM_OUTPUT (PRELLU_OUTPUTS_COUNT)
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

static vsi_status VX_CALLBACK vxParametricReluKernel
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
    vx_uint32   size[4] = {1, 1, 1, 1};
    vx_uint32   dims = 0, innerSize = 0, outerSize = 0, axisSize = 0;
    vx_uint32   outer = 0, inner = 0, i = 0, index = 0, j = 0;
    vx_float32  inValue = 0.0f, result = 0.0f;
    int32_t    axis = 0;
    vx_float32 aV = 0.0f;

    for (i = 0; i < TENSOR_NUM; i++)
    {
        memset(&attr[i], 0, sizeof(vsi_nn_tensor_attr_t));
        for (j = 0; j < VSI_NN_MAX_DIM_NUM; j++)
        {
            stride_size[i][j] = 1;
        }
    }

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

    dims = attr[PRELLU_INPUT].dim_num;
    size[0] = attr[PRELLU_INPUT].size[0];
    size[1] = dims > 1 ? attr[PRELLU_INPUT].size[1] : 1;
    size[2] = dims > 2 ? attr[PRELLU_INPUT].size[2] : 1;
    size[3] = dims > 3 ? attr[PRELLU_INPUT].size[3] : 1;
    axisSize =  attr[PRELLU_INPUT].size[axis];

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
            for (i = 0; i < axisSize; ++i) {
                index    = (outer * axisSize + i) * innerSize + inner;
                inValue  = vsi_nn_DtypeToFloat32_Ex(buffer_ptr[PRELLU_INPUT],
                                                   index, &attr[PRELLU_INPUT].dtype);
                aV       = vsi_nn_DtypeToFloat32_Ex(buffer_ptr[PRELLU_INPUT1],
                                                   i, &attr[PRELLU_INPUT1].dtype);
                result   = (inValue < 0) ? inValue * aV : inValue;
                vsi_nn_Float32ToDtype_Ext(result, buffer_ptr[PRELLU_INPUTS_COUNT + PRELLU_OUTPUT],
                            index, &attr[PRELLU_INPUTS_COUNT + PRELLU_OUTPUT].dtype);
            }
        }
    }

    for( i = TENSOR_NUM_INPUT; i < TENSOR_NUM; i ++ )
    {
        if (buffer_ptr[i])
        {
            status = vsi_nn_copy_tensor_patch(tensor[i], &attr[i], buffer_ptr[i], VX_WRITE_ONLY);
        }
    }
final:
    for( i = 0; i < TENSOR_NUM; i ++ )
    {
        if (buffer_ptr[i]) free(buffer_ptr[i]);

        if (user_addr[i]) vxReleaseTensorAddressing(&(user_addr[i]));
    }
    return status;
} /* _VX_KERNEL_FUNC_KERNEL() */


vsi_status VX_CALLBACK vxParametricReluInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    uint32_t paraNum
    )
{
    // Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
#define gcmMIN(x, y)            (((x) <= (y)) ?  (x) :  (y))
#define IMG_MAX_WIDTH 65536
#define MAX_MULTIPLIER_NUM      (65535)
#define MAX_POST_SHIFT_BITS     (31)
    vx_kernel_execution_parameters_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vsi_status    status                = VX_SUCCESS;
    vx_tensor     input                 = (vx_tensor)paramObj[0];
    vx_tensor     output                = (vx_tensor)paramObj[2];
    vx_uint32     dims                  = 0;
    vx_uint32     width                 = 0;
    vx_uint32     height                = 0;
    vx_uint32     depth                 = 0;
    vx_uint32     thread_width          = 0;
    vx_uint32     thread_height         = 0;
    vx_enum       srcFormat             = VSI_NN_TYPE_FLOAT16;
    vx_enum       dstFormat             = VSI_NN_TYPE_FLOAT16;
    vx_float32    input_scale           = 1.0f;
    vx_float32    output_scale          = 1.0f;
    vx_int32      inputZP               = 0;
    vx_int32      outputZP              = 0;
    vx_int8       srcFixPointPos        = 0;
    vx_int8       dstFixPointPos        = 0;
    vx_enum       srcQuantType          = 0;
    vx_enum       dstQuantType          = 0;
    vx_bool       enable_image_2d       = vx_false_e;
    vx_uint32     hwLitimLen            = IMG_MAX_WIDTH;
    vx_uint16     M0                    = 0;
    vx_int8       postShift             = 0;
    int32_t       axis                  = 0;
    vsi_nn_tensor_attr_t attr[2];

    memset(&attr[0], 0, sizeof(vsi_nn_tensor_attr_t));
    memset(&attr[1], 0, sizeof(vsi_nn_tensor_attr_t));
    status  = vsi_nn_vxGetTensorAttr(input,  &attr[0]);
    status |= vsi_nn_vxGetTensorAttr(output, &attr[1]);
    if (status != VX_SUCCESS)
    {
        VSILOGE("vsi_nn_vxGetTensorAttr failure! at line %d\n", __LINE__);
        return status;
    }

    vxCopyScalar((vx_scalar)paramObj[3], &(axis), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    dims            = attr[1].dim_num;
    width           = attr[1].size[0];
    height          = dims > 1 ? attr[1].size[1] : 1;
    depth           = dims > 2 ? attr[1].size[2] : 1;
    srcFormat       = attr[0].dtype.vx_type;
    dstFormat       = attr[1].dtype.vx_type;
    input_scale     = attr[0].dtype.scale;
    output_scale    = attr[1].dtype.scale;
    inputZP         = attr[0].dtype.zero_point;
    outputZP        = attr[1].dtype.zero_point;
    srcFixPointPos  = attr[0].dtype.fl;
    dstFixPointPos  = attr[1].dtype.fl;
    srcQuantType    = attr[0].dtype.qnt_type;
    dstQuantType    = attr[1].dtype.qnt_type;

    if (0 == axis)
    {
        if ((height * depth < hwLitimLen) && width < hwLitimLen)
        {
            enable_image_2d = vx_true_e;
        }
    }
    else if(1 == axis)
    {
        if (1 == depth)
        {
            enable_image_2d = vx_true_e;
        }
    }

    if(srcQuantType == VSI_NN_QNT_TYPE_DFP)
    {
        if (srcFixPointPos >= 0)
        {
            input_scale = 1.0f / (vx_float32) (1 << srcFixPointPos);
        }
        else if (srcFixPointPos < 0)
        {
            input_scale = (vx_float32) (1 << -srcFixPointPos);
        }
    }
    else if (VSI_NN_QNT_TYPE_NONE == srcQuantType)
    {
        srcFixPointPos = 0;
        input_scale    = 1.0;
    }

    if(dstQuantType == VSI_NN_QNT_TYPE_DFP)
    {
        if (dstFixPointPos >= 0)
        {
            output_scale = (vx_float32) (1 << dstFixPointPos);
        }
        else if (dstFixPointPos < 0)
        {
            output_scale = 1.0f / (vx_float32) (1 << -dstFixPointPos);
        }
    }
    else if (VSI_NN_QNT_TYPE_NONE == dstQuantType)
    {
        dstFixPointPos = 0;
        output_scale   = 1.0;
    }

    vsi_nn_GetFP32MultiAndPostShift(input_scale / output_scale, &M0, &postShift);

    shaderParam.globalWorkScale[0]  = 8;
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.globalWorkScale[2]  = 1;

    if (srcFormat == VSI_NN_TYPE_INT8 && (dstFormat == VSI_NN_TYPE_INT8 || dstFormat == VSI_NN_TYPE_FLOAT16))
    {
        vx_uint32 uniF16MulF16_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x11111111, // BSelt
            0x03020100, 0x07060504, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniPreluI8toF16Lo_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };
        vx_uint32 uniPreluI8toF16Hi_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x0b0a0908, 0x0f0e0d0c, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };

        if (dstFormat == VSI_NN_TYPE_FLOAT16)
        {
            dstFixPointPos = 0;
        }

        if (srcFixPointPos >= dstFixPointPos)
        {
            vx_uint8  postshift      = gcmMIN(srcFixPointPos - dstFixPointPos, MAX_POST_SHIFT_BITS);

            uniPreluI8toF16Lo_2x8[7]    = uniPreluI8toF16Lo_2x8[7] | (postshift & 0x1F);
            uniPreluI8toF16Hi_2x8[7]    = uniPreluI8toF16Hi_2x8[7] | (postshift & 0x1F);
        }
        else
        {
            vx_uint32 idx = 0;
            vx_int32 multiplier = gcmMIN(1 << (dstFixPointPos - srcFixPointPos), MAX_MULTIPLIER_NUM);
            for (idx = 8; idx < 16; idx ++)
            {
                uniPreluI8toF16Hi_2x8[idx] = uniPreluI8toF16Lo_2x8[idx] = (vx_uint32)(multiplier << 16) | (multiplier & 0xffff);
            }
        }
        status |= vxSetNodeUniform(nodObj, "uniPreluI8toF16Lo_2x8", 1, uniPreluI8toF16Lo_2x8);
        status |= vxSetNodeUniform(nodObj, "uniPreluI8toF16Hi_2x8", 1, uniPreluI8toF16Hi_2x8);
        status |= vxSetNodeUniform(nodObj, "uniF16MulF16_2x8", 1, uniF16MulF16_2x8);
        if (dstFormat == VSI_NN_TYPE_INT8)
        {
            shaderParam.globalWorkScale[0]  = 16;
        }
    }
    else if (srcFormat == VSI_NN_TYPE_INT16 && (dstFormat == VSI_NN_TYPE_INT16 || dstFormat == VSI_NN_TYPE_FLOAT16))
    {
        vx_uint32 uniF16MulF16_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x11111111, // BSelt
            0x03020100, 0x07060504, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniPreluI16toF16_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };

        if (dstFormat == VSI_NN_TYPE_FLOAT16)
        {
            dstFixPointPos = 0;
        }

        if (srcFixPointPos >= dstFixPointPos)
        {
            vx_uint8  postshift      = gcmMIN(srcFixPointPos - dstFixPointPos, MAX_POST_SHIFT_BITS);

            uniPreluI16toF16_2x8[7]    = uniPreluI16toF16_2x8[7] | (postshift & 0x1F);
        }
        else
        {
            vx_uint32 idx = 0;
            vx_int32 multiplier = gcmMIN(1 << (dstFixPointPos - srcFixPointPos), MAX_MULTIPLIER_NUM);
            for (idx = 8; idx < 16; idx ++)
            {
                uniPreluI16toF16_2x8[idx] = (vx_uint32)(multiplier << 16) | (multiplier & 0xffff);
            }
        }
        status |= vxSetNodeUniform(nodObj, "uniPreluI16toF16_2x8", 1, uniPreluI16toF16_2x8);
        status |= vxSetNodeUniform(nodObj, "uniF16MulF16_2x8", 1, uniF16MulF16_2x8);
    }
    else if (srcFormat == VSI_NN_TYPE_UINT8 && dstFormat == VSI_NN_TYPE_UINT8)
    {
        vx_uint32 idx = 0;
        vx_uint32 uniU8SubZP_MulM_PStoF16Lo_2x8[16] = {
            0x99999999, // TCfg
            0x44444444, // ASelt
            0x03020100, 0x07060504, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
        };
        vx_uint32 uniU8SubZP_MulM_PStoF16Hi_2x8[16] = {
            0x99999999, // TCfg
            0x44444444, // ASelt
            0x0b0a0908, 0x0f0e0d0c, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
        };
        vx_uint32 uniF16MulF16_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x11111111, // BSelt
            0x03020100, 0x07060504, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniS16AddZP_2x8[16] = {
            0x55555555, // TCfg
            0x44444444, // ASelt
            0x03020100, 0x07060504, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
        };
        uniU8SubZP_MulM_PStoF16Lo_2x8[7] |= postShift;
        uniU8SubZP_MulM_PStoF16Hi_2x8[7] |= postShift;

        for (idx = 8; idx < 16; idx ++)
        {
            uniU8SubZP_MulM_PStoF16Hi_2x8[idx] = uniU8SubZP_MulM_PStoF16Lo_2x8[idx] = (vx_uint32)(M0 << 16) | M0;
        }

        status |= vxSetNodeUniform(nodObj, "uniU8SubZP_MulM_PStoF16Lo_2x8", 1, uniU8SubZP_MulM_PStoF16Lo_2x8);
        status |= vxSetNodeUniform(nodObj, "uniU8SubZP_MulM_PStoF16Hi_2x8", 1, uniU8SubZP_MulM_PStoF16Hi_2x8);
        status |= vxSetNodeUniform(nodObj, "uniF16MulF16_2x8", 1, uniF16MulF16_2x8);
        status |= vxSetNodeUniform(nodObj, "uniS16AddZP_2x8", 1, uniS16AddZP_2x8);
        status |= vxSetNodeUniform(nodObj, "outputZP", 1, &outputZP);
        status |= vxSetNodeUniform(nodObj, "inputZP", 1, &inputZP);

        shaderParam.globalWorkScale[0]  = 16;
    }
    else if (srcFormat == VSI_NN_TYPE_UINT8 && dstFormat == VSI_NN_TYPE_FLOAT16)
    {
        vx_uint32 idx = 0;
        vx_uint32 uniU8SubZP_MulM_PStoF16Lo_2x8[16] = {
            0x99999999, // TCfg
            0x44444444, // ASelt
            0x03020100, 0x07060504, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
        };
        vx_uint32 uniU8SubZP_MulM_PStoF16Hi_2x8[16] = {
            0x99999999, // TCfg
            0x44444444, // ASelt
            0x0b0a0908, 0x0f0e0d0c, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
        };
        vx_uint32 uniF16MulF16_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x11111111, // BSelt
            0x03020100, 0x07060504, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        uniU8SubZP_MulM_PStoF16Lo_2x8[7] |= postShift;
        uniU8SubZP_MulM_PStoF16Hi_2x8[7] |= postShift;

        for (idx = 8; idx < 16; idx ++)
        {
            uniU8SubZP_MulM_PStoF16Hi_2x8[idx] = uniU8SubZP_MulM_PStoF16Lo_2x8[idx] = (vx_uint32)(M0 << 16) | M0;
        }
        status |= vxSetNodeUniform(nodObj, "uniU8SubZP_MulM_PStoF16Lo_2x8", 1, uniU8SubZP_MulM_PStoF16Lo_2x8);
        status |= vxSetNodeUniform(nodObj, "uniU8SubZP_MulM_PStoF16Hi_2x8", 1, uniU8SubZP_MulM_PStoF16Hi_2x8);
        status |= vxSetNodeUniform(nodObj, "uniF16MulF16_2x8", 1, uniF16MulF16_2x8);
        status |= vxSetNodeUniform(nodObj, "inputZP", 1, &inputZP);
        shaderParam.globalWorkScale[0]  = 16;
    }
    else if (srcFormat == VSI_NN_TYPE_FLOAT16)
    {
        vx_uint32 UniFP16Mul_dp2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x11111111, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
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

        if (0 == axis)
        {
            vx_uint32 mul_config[2] = {0x03020100, 0x07060504};
            UniFP16Mul_dp2x8[5] = mul_config[0];
            UniFP16Mul_dp2x8[6] = mul_config[1];
        }

        if (dstFormat == VSI_NN_TYPE_FLOAT16)
        {
            status |= vxSetNodeUniform(nodObj, "UniFP16Mul_dp2x8", 1, UniFP16Mul_dp2x8);
        }
        else if (dstFormat == VSI_NN_TYPE_INT8)
        {
            status |= vxSetNodeUniform(nodObj, "uniConvertDirInt16Fp32_4x4", 1, uniConvertDirInt16Fp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvertEndInt16Fp32_4x4", 1, uniConvertEndInt16Fp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvertInt32toUint8_2x8", 1, uniConvertInt32toUint8_2x8);
            status |= vxSetNodeUniform(nodObj, "outputScale", 1, &output_scale);
        }
        else if (dstFormat == VSI_NN_TYPE_UINT8)
        {
            output_scale = 1.0f / output_scale;
            status |= vxSetNodeUniform(nodObj, "uniConvertDirInt16Fp32_4x4", 1, uniConvertDirInt16Fp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvertEndInt16Fp32_4x4", 1, uniConvertEndInt16Fp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvertInt32toUint8_2x8", 1, uniConvertInt32toUint8_2x8);
            status |= vxSetNodeUniform(nodObj, "outputScale", 1, &output_scale);
            status |= vxSetNodeUniform(nodObj, "outputZP", 1, &outputZP);
        }
        else if (dstFormat == VSI_NN_TYPE_INT16)
        {
            status |= vxSetNodeUniform(nodObj, "uniConvertDirInt16Fp32_4x4", 1, uniConvertDirInt16Fp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvertEndInt16Fp32_4x4", 1, uniConvertEndInt16Fp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvertInt32toUint8_2x8", 1, uniConvertInt32toUint8_2x8);
            status |= vxSetNodeUniform(nodObj, "outputScale", 1, &output_scale);
        }
    }
    else if (srcFormat == VSI_NN_TYPE_BFLOAT16 && dstFormat == VSI_NN_TYPE_BFLOAT16)
    {
        vx_uint32 uniConvBF16toF32_Part0_2x8[16] = {
            0x11111111, // TCfg
            0x01010101, // ASelt
            0x01010000, 0x03030202, // ABin
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
        vx_uint32 uniConvF16toF32_Part0_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };
        vx_uint32 uniConvF16toF32_Part1_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };
        vx_uint32 uniPackedBF16_2x8[16] = {
            0x11111111, // TCfg
            0x11110000, // ASelt
            0x07050301, 0x07050301, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };

        status |= vxSetNodeUniform(nodObj, "uniConvBF16toF32_Part0_2x8", 1, uniConvBF16toF32_Part0_2x8);
        status |= vxSetNodeUniform(nodObj, "uniConvBF16toF32_Part1_2x8", 1, uniConvBF16toF32_Part1_2x8);
        status |= vxSetNodeUniform(nodObj, "uniConvF16toF32_Part0_4x4", 1, uniConvF16toF32_Part0_4x4);
        if (0 == axis)
        {
            status |= vxSetNodeUniform(nodObj, "uniConvF16toF32_Part1_4x4", 1, uniConvF16toF32_Part1_4x4);
        }
        status |= vxSetNodeUniform(nodObj, "uniPackedBF16_2x8", 1, uniPackedBF16_2x8);
    }

    if (enable_image_2d)
    {
        if (0 == axis || (1 == axis && 1 == depth))
        {
            thread_width  =  width;
            thread_height =  height * depth;
        }
        else
        {
            thread_width  =  width * height;
            thread_height =  depth;
        }
        shaderParam.workDim             = 2;
        shaderParam.globalWorkSize[0]   = gcmALIGN((thread_width + shaderParam.globalWorkScale[0] - 1)
            / shaderParam.globalWorkScale[0], 4);
        shaderParam.globalWorkSize[1]   = (thread_height + shaderParam.globalWorkScale[1] - 1)
            / shaderParam.globalWorkScale[1];
    }
    else
    {
        shaderParam.workDim             = 3;
        shaderParam.globalWorkSize[0]   = gcmALIGN((width + shaderParam.globalWorkScale[0] - 1)
            / shaderParam.globalWorkScale[0], 4);
        shaderParam.globalWorkSize[1]   = (height + shaderParam.globalWorkScale[1] - 1)
            / shaderParam.globalWorkScale[1];
        shaderParam.globalWorkSize[2]   = depth;
    }

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));


#undef gcmALIGN
#undef IMG_MAX_WIDTH
#undef gcmMIN
#undef MAX_MULTIPLIER_NUM
#undef MAX_POST_SHIFT_BITS
    return status;
}

static vx_param_description_t vxParametricReluKernelParam[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED}
};

#ifdef __cplusplus
extern "C" {
#endif

vx_kernel_description_t vxParametricReluKernelInfo_CPU =
{
    VX_KERNEL_ENUM_PARAMETRICRELU,
    "com.vivantecorp.extension.vxcParametricRelu_sw",
    vxParametricReluKernel,
    vxParametricReluKernelParam,
    (sizeof(vxParametricReluKernelParam) / sizeof(vxParametricReluKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};


#define PRELU_KERNELS(AXI_INDEX, SRC_TYPE, ALPHA_TYPE, DST_TYPE) \
    vx_kernel_description_t vxParametricRelu_##AXI_INDEX##_##SRC_TYPE##ALPHA_TYPE##to##DST_TYPE##_Kernel = \
{ \
    VX_KERNEL_ENUM_PARAMETRICRELU, \
    VX_KERNEL_NAME_PARAMETRICRELU_##AXI_INDEX##_##SRC_TYPE##ALPHA_TYPE##TO##DST_TYPE, \
    NULL, \
    vxParametricReluKernelParam, \
    (sizeof(vxParametricReluKernelParam) / sizeof(vxParametricReluKernelParam[0])), \
    vsi_nn_KernelValidator, \
    NULL, \
    NULL, \
    vxParametricReluInitializer, \
    vsi_nn_KernelDeinitializer \
};

#define PRELU_KERNELS_2D(AXI_INDEX, SRC_TYPE, ALPHA_TYPE, DST_TYPE) \
    vx_kernel_description_t vxParametricRelu_##AXI_INDEX##_##SRC_TYPE##ALPHA_TYPE##to##DST_TYPE##_2D_Kernel = \
{ \
    VX_KERNEL_ENUM_PARAMETRICRELU, \
    VX_KERNEL_NAME_PARAMETRICRELU_##AXI_INDEX##_##SRC_TYPE##ALPHA_TYPE##TO##DST_TYPE##_2D, \
    NULL, \
    vxParametricReluKernelParam, \
    (sizeof(vxParametricReluKernelParam) / sizeof(vxParametricReluKernelParam[0])), \
    vsi_nn_KernelValidator, \
    NULL, \
    NULL, \
    vxParametricReluInitializer, \
    vsi_nn_KernelDeinitializer \
};

PRELU_KERNELS(AXI0, BF16, F16, BF16)
PRELU_KERNELS(AXI0, BF16, BF16, BF16)
PRELU_KERNELS(AXI0, F16, F16, F16)
PRELU_KERNELS(AXI0, F16, F16, I16)
PRELU_KERNELS(AXI0, F16, F16, I8)
PRELU_KERNELS(AXI0, F16, F16, U8)
PRELU_KERNELS(AXI0, I16, F16, I16)
PRELU_KERNELS(AXI0, I8,  F16, I8)
PRELU_KERNELS(AXI0, U8,  F16, U8)
PRELU_KERNELS(AXI0, I16, F16, F16)
PRELU_KERNELS(AXI0, I8,  F16, F16)
PRELU_KERNELS(AXI0, U8,  F16, F16)

PRELU_KERNELS(AXI1, BF16, F16, BF16)
PRELU_KERNELS(AXI1, BF16, BF16, BF16)
PRELU_KERNELS(AXI1, F16, F16, F16)
PRELU_KERNELS(AXI1, F16, F16, I16)
PRELU_KERNELS(AXI1, F16, F16, I8)
PRELU_KERNELS(AXI1, F16, F16, U8)
PRELU_KERNELS(AXI1, I16, F16, I16)
PRELU_KERNELS(AXI1, I8,  F16, I8)
PRELU_KERNELS(AXI1, U8,  F16, U8)
PRELU_KERNELS(AXI1, I16, F16, F16)
PRELU_KERNELS(AXI1, I8,  F16, F16)
PRELU_KERNELS(AXI1, U8,  F16, F16)

PRELU_KERNELS_2D(AXI0, BF16, F16, BF16)
PRELU_KERNELS_2D(AXI0, BF16, BF16, BF16)
PRELU_KERNELS_2D(AXI0, F16, F16, F16)
PRELU_KERNELS_2D(AXI0, F16, F16, I16)
PRELU_KERNELS_2D(AXI0, F16, F16, I8)
PRELU_KERNELS_2D(AXI0, F16, F16, U8)
PRELU_KERNELS_2D(AXI0, I16, F16, I16)
PRELU_KERNELS_2D(AXI0, I8,  F16, I8)
PRELU_KERNELS_2D(AXI0, U8,  F16, U8)
PRELU_KERNELS_2D(AXI0, I16, F16, F16)
PRELU_KERNELS_2D(AXI0, I8,  F16, F16)
PRELU_KERNELS_2D(AXI0, U8,  F16, F16)

PRELU_KERNELS_2D(AXI1, BF16, F16, BF16)
PRELU_KERNELS_2D(AXI1, BF16, BF16, BF16)
PRELU_KERNELS_2D(AXI1, F16, F16, F16)
PRELU_KERNELS_2D(AXI1, F16, F16, I16)
PRELU_KERNELS_2D(AXI1, F16, F16, I8)
PRELU_KERNELS_2D(AXI1, F16, F16, U8)
PRELU_KERNELS_2D(AXI1, I16, F16, I16)
PRELU_KERNELS_2D(AXI1, I8,  F16, I8)
PRELU_KERNELS_2D(AXI1, U8,  F16, U8)
PRELU_KERNELS_2D(AXI1, I16, F16, F16)
PRELU_KERNELS_2D(AXI1, I8,  F16, F16)
PRELU_KERNELS_2D(AXI1, U8,  F16, F16)

#define PRELU_KERENLS_NAME(AXI_INDEX, SRC_TYPE, ALPHA_TYPE, DST_TYPE, INSTR) \
    &vxParametricRelu_##AXI_INDEX##_##SRC_TYPE##ALPHA_TYPE##to##DST_TYPE##_##INSTR##Kernel,


vx_kernel_description_t * vx_kernel_PRELU_list[] =
{
    &vxParametricReluKernelInfo_CPU,
    PRELU_KERENLS_NAME(AXI0, BF16, F16, BF16, )
    PRELU_KERENLS_NAME(AXI0, BF16, BF16, BF16, )
    PRELU_KERENLS_NAME(AXI0, F16,  F16, F16, )
    PRELU_KERENLS_NAME(AXI0, F16,  F16, I16, )
    PRELU_KERENLS_NAME(AXI0, F16,  F16, I8, )
    PRELU_KERENLS_NAME(AXI0, F16,  F16, U8, )
    PRELU_KERENLS_NAME(AXI0, I16,  F16, I16, )
    PRELU_KERENLS_NAME(AXI0, I8,   F16, I8, )
    PRELU_KERENLS_NAME(AXI0, U8,   F16, U8, )
    PRELU_KERENLS_NAME(AXI0, I16,  F16, F16, )
    PRELU_KERENLS_NAME(AXI0, I8,   F16, F16, )
    PRELU_KERENLS_NAME(AXI0, U8,   F16, F16, )
    PRELU_KERENLS_NAME(AXI0, BF16, F16, BF16, 2D_)
    PRELU_KERENLS_NAME(AXI0, BF16, BF16, BF16, 2D_)
    PRELU_KERENLS_NAME(AXI0, F16,  F16, F16,  2D_)
    PRELU_KERENLS_NAME(AXI0, F16,  F16, I16,  2D_)
    PRELU_KERENLS_NAME(AXI0, F16,  F16, I8,   2D_)
    PRELU_KERENLS_NAME(AXI0, F16,  F16, U8,   2D_)
    PRELU_KERENLS_NAME(AXI0, I16,  F16, I16,  2D_)
    PRELU_KERENLS_NAME(AXI0, I8,   F16, I8,   2D_)
    PRELU_KERENLS_NAME(AXI0, U8,   F16, U8,   2D_)
    PRELU_KERENLS_NAME(AXI0, I16,  F16, F16,  2D_)
    PRELU_KERENLS_NAME(AXI0, I8,   F16, F16,  2D_)
    PRELU_KERENLS_NAME(AXI0, U8,   F16, F16,  2D_)
    PRELU_KERENLS_NAME(AXI1, BF16, F16, BF16, )
    PRELU_KERENLS_NAME(AXI1, BF16, BF16, BF16, )
    PRELU_KERENLS_NAME(AXI1, F16,  F16, F16, )
    PRELU_KERENLS_NAME(AXI1, F16,  F16, I16, )
    PRELU_KERENLS_NAME(AXI1, F16,  F16, I8, )
    PRELU_KERENLS_NAME(AXI1, F16,  F16, U8, )
    PRELU_KERENLS_NAME(AXI1, I16,  F16, I16, )
    PRELU_KERENLS_NAME(AXI1, I8,   F16, I8, )
    PRELU_KERENLS_NAME(AXI1, U8,   F16, U8, )
    PRELU_KERENLS_NAME(AXI1, I16,  F16, F16, )
    PRELU_KERENLS_NAME(AXI1, I8,   F16, F16, )
    PRELU_KERENLS_NAME(AXI1, U8,   F16, F16, )
    PRELU_KERENLS_NAME(AXI1, BF16, F16, BF16, 2D_)
    PRELU_KERENLS_NAME(AXI1, BF16, BF16, BF16, 2D_)
    PRELU_KERENLS_NAME(AXI1, F16,  F16, F16,  2D_)
    PRELU_KERENLS_NAME(AXI1, F16,  F16, I16,  2D_)
    PRELU_KERENLS_NAME(AXI1, F16,  F16, I8,   2D_)
    PRELU_KERENLS_NAME(AXI1, F16,  F16, U8,   2D_)
    PRELU_KERENLS_NAME(AXI1, I16,  F16, I16,  2D_)
    PRELU_KERENLS_NAME(AXI1, I8,   F16, I8,   2D_)
    PRELU_KERENLS_NAME(AXI1, U8,   F16, U8,   2D_)
    PRELU_KERENLS_NAME(AXI1, I16,  F16, F16,  2D_)
    PRELU_KERENLS_NAME(AXI1, I8,   F16, F16,  2D_)
    PRELU_KERENLS_NAME(AXI1, U8,   F16, F16,  2D_)
    NULL
};
#ifdef __cplusplus
}
#endif
