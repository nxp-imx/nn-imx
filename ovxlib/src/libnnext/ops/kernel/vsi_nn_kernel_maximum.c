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

#include "vsi_nn_platform.h"

#include "vsi_nn_prv.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_math.h"
#include "vsi_nn_log.h"
#include "client/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"

#define _VX_KERNEL_ID           (VX_KERNEL_ENUM_MAXIMUM)
#define TENSOR_NUM_INPUT  (MAXIMUM_INPUTS_COUNT)
#define TENSOR_NUM_OUTPUT (MAXIMUM_OUTPUTS_COUNT)
#define TENSOR_NUM (TENSOR_NUM_INPUT+TENSOR_NUM_OUTPUT)

#define MAX_MULTIPLIER_NUM      (65535)
#define MAX_POST_SHIFT_BITS     (31)

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

static uint32_t getExpandTensorOffset(uint32_t index, uint32_t num_of_dims, uint32_t * in_dims,
                                       uint32_t *strides, uint32_t * out_dims)
{
    uint32_t offset = 0;
    uint32_t i;

    for(i = 0; i < num_of_dims; i++)
    {
        if(in_dims[i] == out_dims[i])
            offset += strides[i] * (index % out_dims[i]);

        index /= out_dims[i];
    }

    return offset;
}

vsi_status VX_CALLBACK vxMaximumKernel
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
    vx_uint32    i                              = 0;
    uint32_t     elementCount                   = 1;

    for (i = 0; i < TENSOR_NUM; i++)
    {
        memset(&attr[i], 0, sizeof(vsi_nn_tensor_attr_t));
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

    for (i = 0; i < attr[2].dim_num; i++)
    {
        elementCount *= attr[2].size[i];
    }

    for (i = 0; i < elementCount; i++)
    {
        uint32_t  in0offset = 0;
        uint32_t  in1offset = 0;
        vx_uint8   *in0_ptr  = NULL;
        vx_uint8   *in1_ptr  = NULL;
        vx_float32 in0Data   = 0;
        vx_float32 in1Data   = 0;
        vx_float32 outData   = 0;

        in0offset = getExpandTensorOffset(i, attr[0].dim_num, attr[0].size, stride_size[0], attr[2].size);
        in1offset = getExpandTensorOffset(i, attr[1].dim_num, attr[1].size, stride_size[1], attr[2].size);

        in0_ptr = (vx_uint8 *)buffer_ptr[0] + in0offset;
        in1_ptr = (vx_uint8 *)buffer_ptr[1] + in1offset;

        in0Data = vsi_nn_DtypeToFloat32_Ex(in0_ptr, 0, &attr[0].dtype);
        in1Data = vsi_nn_DtypeToFloat32_Ex(in1_ptr, 0, &attr[1].dtype);

        outData = vsi_nn_max(in0Data, in1Data);

        vsi_nn_Float32ToDtype_Ext(outData, buffer_ptr[2], i, &attr[2].dtype);
    }

    //save data
    for( i = TENSOR_NUM_INPUT; i < TENSOR_NUM; i ++ )
    {
        if (buffer_ptr[i])
        {
            status = vsi_nn_copy_tensor_patch(tensor[i], &attr[i], buffer_ptr[i], VX_WRITE_ONLY);
        }

        if (user_addr[i]) vxReleaseTensorAddressing(&(user_addr[i]));
    }

    for( i = 0; i < TENSOR_NUM; i ++ )
    {
        if (buffer_ptr[i]) free(buffer_ptr[i]);

        if (user_addr[i]) vxReleaseTensorAddressing(&(user_addr[i]));
    }

    return status;
}

vsi_status VX_CALLBACK vxMaximumInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    uint32_t paraNum
    )
{
// Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
#define gcmMIN(x, y)            (((x) <= (y)) ?  (x) :  (y))
#define gcmMAX(x, y)            (((x) >= (y)) ?  (x) :  (y))
#define MAX_MULTIPLIER_NUM      (65535)
#define MAX_POST_SHIFT_BITS     (31)
    vx_kernel_execution_parameters_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vx_status    status             = VX_SUCCESS;
    vx_tensor    input0             = (vx_tensor)paramObj[0];
    vx_tensor    input1             = (vx_tensor)paramObj[1];
    vx_tensor    output             = (vx_tensor)paramObj[2];
    vx_enum      src0Format         = VSI_NN_TYPE_FLOAT16;
    vx_enum      src1Format         = VSI_NN_TYPE_FLOAT16;
    vx_enum      dstFormat          = VSI_NN_TYPE_FLOAT16;
    vx_enum      src0QuantType      = 0;
    vx_int8      src0FixPointPos    = 0;
    vx_int32     src0ZP             = 0;
    vx_float32   src0Scale          = 1.0f;
    vx_enum      src1QuantType      = 0;
    vx_int8      src1FixPointPos    = 0;
    vx_int32     src1ZP             = 0;
    vx_float32   src1Scale          = 1.0f;
    vx_enum      dstQuantType       = 0;
    vx_int8      dstFixPointPos     = 0;
    vx_int32     dstZP              = 0;
    vx_float32   dstScale           = 1.0f;
    vx_bool      isDymFixPoint      = vx_false_e;
    vsi_nn_tensor_attr_t attr[3];

    memset(&attr[0], 0, sizeof(vsi_nn_tensor_attr_t));
    memset(&attr[1], 0, sizeof(vsi_nn_tensor_attr_t));
    memset(&attr[2], 0, sizeof(vsi_nn_tensor_attr_t));

    status  = vsi_nn_vxGetTensorAttr(input0, &attr[0]);
    status |= vsi_nn_vxGetTensorAttr(input1, &attr[1]);
    status |= vsi_nn_vxGetTensorAttr(output, &attr[2]);

    if(status < 0)
        VSILOGE("error-%s,%d\n",__FILE__,__LINE__);

    src0Format = attr[0].dtype.vx_type;
    src0QuantType = attr[0].dtype.qnt_type;
    src0FixPointPos = attr[0].dtype.fl;
    if (src0QuantType == VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC)
    {
        src0ZP = attr[0].dtype.zero_point;
        src0Scale = attr[0].dtype.scale;
    }

    src1Format = attr[1].dtype.vx_type;
    src1QuantType = attr[1].dtype.qnt_type;
    src1FixPointPos = attr[1].dtype.fl;
    if (src1QuantType == VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC)
    {
        src1ZP = attr[1].dtype.zero_point;
        src1Scale = attr[1].dtype.scale;
    }

    dstFormat = attr[2].dtype.vx_type;
    dstQuantType = attr[2].dtype.qnt_type;
    dstFixPointPos = attr[2].dtype.fl;
    if (src1QuantType == VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC)
    {
        dstZP = attr[2].dtype.zero_point;
        dstScale = attr[2].dtype.scale;
    }

    isDymFixPoint = (vx_bool)(src0QuantType == src1QuantType && src0QuantType == VSI_NN_QNT_TYPE_DFP
                           && dstQuantType == VSI_NN_QNT_TYPE_DFP);

    if (dstFormat == VSI_NN_TYPE_FLOAT16 || dstFormat == VSI_NN_TYPE_INT16
        || dstFormat == VSI_NN_TYPE_BFLOAT16)
    {
        shaderParam.globalWorkScale[0]  = 8;
        shaderParam.globalWorkScale[1]  = 1;
        shaderParam.globalWorkScale[2]  = 1;
    }
    else
    {
        shaderParam.globalWorkScale[0]  = 16;
        shaderParam.globalWorkScale[1]  = 1;
        shaderParam.globalWorkScale[2]  = 1;
    }

    shaderParam.workDim = attr[2].dim_num < 3 ? 2 : 3;
    shaderParam.globalWorkSize[0]   = gcmALIGN((attr[2].size[0] + shaderParam.globalWorkScale[0] - 1)
                                        / shaderParam.globalWorkScale[0], 4);
    shaderParam.globalWorkSize[1]   = (attr[2].size[1] + shaderParam.globalWorkScale[1] - 1)
                                        / shaderParam.globalWorkScale[1];
    shaderParam.globalWorkSize[2]   = attr[2].size[2];

    if ((isDymFixPoint && src0Format == VSI_NN_TYPE_INT8
        && src1Format == VSI_NN_TYPE_INT8 && dstFormat == VSI_NN_TYPE_INT8)
    || (src0Format == VSI_NN_TYPE_INT8 && src1Format == VSI_NN_TYPE_FLOAT16 && dstFormat == VSI_NN_TYPE_INT8))
    {
        vx_uint32 uniConvertI8toI8_0_part0_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };
        vx_uint32 uniConvertI8toI8_0_part1_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x0b0a0908, 0x0f0e0d0c, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };
        vx_uint32 uniConvertI8toI8_1_part0_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };
        vx_uint32 uniConvertI8toI8_1_part1_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x0b0a0908, 0x0f0e0d0c, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };

        if (src0FixPointPos > dstFixPointPos)
        {
            vx_uint8  postshift = gcmMIN(src0FixPointPos - dstFixPointPos, MAX_POST_SHIFT_BITS);

            uniConvertI8toI8_0_part0_2x8[7] |= (postshift & 0x1F);
            uniConvertI8toI8_0_part1_2x8[7] |= (postshift & 0x1F);
        }
        else
        {
            vx_uint32 multiplier = gcmMIN(1 << (dstFixPointPos - src0FixPointPos), MAX_MULTIPLIER_NUM);
            vx_uint32 i          = 0;

            for (i = 0; i < 8; i++)
            {
                uniConvertI8toI8_0_part0_2x8[i + 8] = multiplier;
                uniConvertI8toI8_0_part1_2x8[i + 8] = multiplier;
            }
        }

        status |= vxSetNodeUniform(nodObj, "uniConvertI8toI8_0_part0_2x8", 1, uniConvertI8toI8_0_part0_2x8);
        status |= vxSetNodeUniform(nodObj, "uniConvertI8toI8_0_part1_2x8", 1, uniConvertI8toI8_0_part1_2x8);

        if(src1Format == VSI_NN_TYPE_FLOAT16)
        {
            vx_uint32 uinConvertFp16ToInt8_2x8[16] = {
                0x11111111, // TCfg
                0x00000000, // ASelt
                0x03020100, 0x07060504, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001,
                0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
            };

            if (0 > dstFixPointPos)
            {
                vx_uint8  postshift      = gcmMIN(0 - dstFixPointPos, MAX_POST_SHIFT_BITS);

                uinConvertFp16ToInt8_2x8[7] |= (postshift & 0x1F);
            }
            else
            {
                vx_uint32 multiplier = gcmMIN(1 << (dstFixPointPos - 0), MAX_MULTIPLIER_NUM);
                vx_uint32 i          = 0;

                for (i = 0; i < 8; i++)
                {
                    uinConvertFp16ToInt8_2x8[i + 8] = multiplier;
                }
            }

            status |= vxSetNodeUniform(nodObj, "uinConvertFp16ToInt8_2x8", 1, uinConvertFp16ToInt8_2x8);
        }
        else
        {
            if (src1FixPointPos > dstFixPointPos)
            {
                vx_uint8  postshift      = gcmMIN(src1FixPointPos - dstFixPointPos, MAX_POST_SHIFT_BITS);

                uniConvertI8toI8_1_part0_2x8[7] |= (postshift & 0x1F);
                uniConvertI8toI8_1_part1_2x8[7] |= (postshift & 0x1F);
            }
            else
            {
                vx_uint32 multiplier = gcmMIN(1 << (dstFixPointPos - src1FixPointPos), MAX_MULTIPLIER_NUM);
                vx_uint32 i          = 0;

                for (i = 0; i < 8; i++)
                {
                    uniConvertI8toI8_1_part0_2x8[i + 8] = multiplier;
                    uniConvertI8toI8_1_part1_2x8[i + 8] = multiplier;
                }
            }

            status |= vxSetNodeUniform(nodObj, "uniConvertI8toI8_1_part0_2x8", 1, uniConvertI8toI8_1_part0_2x8);
            status |= vxSetNodeUniform(nodObj, "uniConvertI8toI8_1_part1_2x8", 1, uniConvertI8toI8_1_part1_2x8);
        }
    }
    else if (isDymFixPoint && src0Format == VSI_NN_TYPE_INT16
        && src1Format == VSI_NN_TYPE_INT16 && dstFormat == VSI_NN_TYPE_INT16)
    {
        vx_uint32 uniConvertI16toI16_0_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };
        vx_uint32 uniConvertI16toI16_1_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };

        if (src0FixPointPos > dstFixPointPos)
        {
            vx_uint8  postshift      = gcmMIN(src0FixPointPos - dstFixPointPos, MAX_POST_SHIFT_BITS);

            uniConvertI16toI16_0_2x8[7] |= (postshift & 0x1F);
        }
        else
        {
            vx_uint32 multiplier = gcmMIN(1 << (dstFixPointPos - src0FixPointPos), MAX_MULTIPLIER_NUM);
            vx_uint32 i          = 0;

            for (i = 0; i < 8; i++)
            {
                uniConvertI16toI16_0_2x8[i + 8] = multiplier;
            }
        }

        status |= vxSetNodeUniform(nodObj, "uniConvertI16toI16_0_2x8", 1, uniConvertI16toI16_0_2x8);

        if (src1FixPointPos > dstFixPointPos)
        {
            vx_uint8  postshift      = gcmMIN(src1FixPointPos - dstFixPointPos, MAX_POST_SHIFT_BITS);

            uniConvertI16toI16_1_2x8[7] |= (postshift & 0x1F);
        }
        else
        {
            vx_uint32 multiplier = gcmMIN(1 << (dstFixPointPos - src1FixPointPos), MAX_MULTIPLIER_NUM);
            vx_uint32 i          = 0;

            for (i = 0; i < 8; i++)
            {
                uniConvertI16toI16_1_2x8[i + 8] = multiplier;
            }
        }

        status |= vxSetNodeUniform(nodObj, "uniConvertI16toI16_1_2x8", 1, uniConvertI16toI16_1_2x8);
    }
    else if ((src0Format == VSI_NN_TYPE_UINT8 && src1Format == VSI_NN_TYPE_UINT8 && dstFormat == VSI_NN_TYPE_UINT8)
         || (src0Format == VSI_NN_TYPE_FLOAT16 && src1Format == VSI_NN_TYPE_FLOAT16 && dstFormat == VSI_NN_TYPE_UINT8)
         || (src0Format == VSI_NN_TYPE_UINT8 && src1Format == VSI_NN_TYPE_FLOAT16 && dstFormat == VSI_NN_TYPE_UINT8))
    {
        vx_uint16   M0                      = 0;
        vx_int8     postShift0              = 0;
        vx_uint16   M1                      = 0;
        vx_int8     postShift1              = 0;
        vx_uint32  multAndoutZP0[2]         = {0};
        vx_uint32  multAndoutZP1[2]         = {0};

        vx_uint32 uniU8MulAndPostShift_Lo_2x8[16] = {
            0xdddddddd, // TCfg
            0x44444444, // ASelt
            0x13121110, 0x17161514, // ABin
            0x11111111, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniU8MulAndPostShift_Hi_2x8[16] = {
            0xdddddddd, // TCfg
            0x44444444, // ASelt
            0x1b1a1918, 0x1f1e1d1c, // ABin
            0x11111111, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };

        vsi_nn_GetFP32MultiAndPostShift(src0Scale / dstScale, &M0, &postShift0);
        vsi_nn_GetFP32MultiAndPostShift(src1Scale / dstScale, &M1, &postShift1);

        multAndoutZP0[0] = (vx_uint32)(M0);
        multAndoutZP0[1] = (vx_uint32)((dstZP << postShift0) - src0ZP * M0);
        multAndoutZP1[0] = (vx_uint32)(M1);
        multAndoutZP1[1] = (vx_uint32)((dstZP << postShift1) - src1ZP * M1);

        uniU8MulAndPostShift_Lo_2x8[7] = 0x00002600 | (postShift0 & 0x1F);
        uniU8MulAndPostShift_Hi_2x8[7] = 0x00002600 | (postShift0 & 0x1F);

        if (src0Format == VSI_NN_TYPE_UINT8)
        {
            status |= vxSetNodeUniform(nodObj, "uniU8MulAndPostShift0_Lo_2x8",  1, uniU8MulAndPostShift_Lo_2x8);
            status |= vxSetNodeUniform(nodObj, "uniU8MulAndPostShift0_Hi_2x8",  1, uniU8MulAndPostShift_Hi_2x8);
            status |= vxSetNodeUniform(nodObj, "multAndoutZP0",  1, multAndoutZP0);
        }

        status |= vxSetNodeUniform(nodObj, "multAndoutZP1",  1, multAndoutZP1);

        if(src1Format == VSI_NN_TYPE_FLOAT16)
        {
            vx_uint32 uniConvertFp16toU8_2x8[16] = {
                0xdddddddd, // TCfg
                0x44444444, // ASelt
                0x13121110, 0x17161514, // ABin
                0x11111111, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002600, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            };

            uniConvertFp16toU8_2x8[7] = 0x00002600 | (postShift1 & 0x1F);
            status |= vxSetNodeUniform(nodObj, "uniConvertFp16toU8_2x8",  1, uniConvertFp16toU8_2x8);
        }
        else
        {
            uniU8MulAndPostShift_Lo_2x8[7] = 0x00002600 | (postShift1 & 0x1F);
            uniU8MulAndPostShift_Hi_2x8[7] = 0x00002600 | (postShift1 & 0x1F);
            status |= vxSetNodeUniform(nodObj, "uniU8MulAndPostShift1_Lo_2x8",  1, uniU8MulAndPostShift_Lo_2x8);
            status |= vxSetNodeUniform(nodObj, "uniU8MulAndPostShift1_Hi_2x8",  1, uniU8MulAndPostShift_Hi_2x8);
        }
    }
    else if(src0Format == VSI_NN_TYPE_INT8 && src1Format == VSI_NN_TYPE_FLOAT16 && dstFormat == VSI_NN_TYPE_FLOAT16)
    {
        vx_uint32 uniConvertInt8toFp16_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };

        if (src0FixPointPos > 0)
        {
            vx_uint8  postshift = gcmMIN(src0FixPointPos, MAX_POST_SHIFT_BITS);

            uniConvertInt8toFp16_2x8[7] |= (postshift & 0x1F);
        }
        else
        {
            vx_uint32 multiplier = gcmMIN(1 << (0 - src0FixPointPos), MAX_MULTIPLIER_NUM);
            vx_uint32 i          = 0;

            for (i = 0; i < 8; i++)
            {
                uniConvertInt8toFp16_2x8[i + 8] = multiplier;
            }
        }

        status |= vxSetNodeUniform(nodObj, "uniConvertInt8toFp16_2x8", 1, uniConvertInt8toFp16_2x8);
    }
    else if(src0Format == VSI_NN_TYPE_UINT8 && src1Format == VSI_NN_TYPE_FLOAT16 && dstFormat == VSI_NN_TYPE_FLOAT16)
    {
        vx_uint16  M0                   = 0;
        vx_int8    postShift            = 0;
        vx_uint32    multAndoutZP0[2]   = {0};
        vx_uint32 uniU8MulAndPostShift_0_Lo_2x8[16] = {
            0xdddddddd, // TCfg
            0x44444444, // ASelt
            0x13121110, 0x17161514, // ABin
            0x11111111, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };

        vsi_nn_GetFP32MultiAndPostShift(src0Scale / dstScale, &M0, &postShift);
        multAndoutZP0[0] = (vx_uint32)(M0);
        multAndoutZP0[1] = (vx_uint32)((dstZP << postShift) - src0ZP * M0);

        uniU8MulAndPostShift_0_Lo_2x8[7] |= (postShift & 0x1F);
        status |= vxSetNodeUniform(nodObj, "uniU8MulAndPostShift_0_Lo_2x8", 1, uniU8MulAndPostShift_0_Lo_2x8);
        status |= vxSetNodeUniform(nodObj, "multAndoutZP0", 1, multAndoutZP0);
    }
    else if(src0Format == VSI_NN_TYPE_INT16 && src1Format == VSI_NN_TYPE_FLOAT16 && dstFormat == VSI_NN_TYPE_INT16)
    {
        vx_uint32 uniConvertI16toI16_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };
        vx_uint32 uinConvertFp16ToInt16_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };

        if (src0FixPointPos > dstFixPointPos)
        {
            vx_uint8  postshift = gcmMIN(src0FixPointPos - dstFixPointPos, MAX_POST_SHIFT_BITS);

            uniConvertI16toI16_2x8[7] |= (postshift & 0x1F);
        }
        else
        {
            vx_uint32 multiplier = gcmMIN(1 << (dstFixPointPos - src0FixPointPos), MAX_MULTIPLIER_NUM);
            vx_uint32 i          = 0;

            for (i = 0; i < 8; i++)
            {
                uniConvertI16toI16_2x8[i + 8] = multiplier;
            }
        }

        if (0 > dstFixPointPos)
        {
            vx_uint8  postshift      = gcmMIN(0 - dstFixPointPos, MAX_POST_SHIFT_BITS);
            uinConvertFp16ToInt16_2x8[7] |= (postshift & 0x1F);
        }
        else
        {
            vx_uint32 multiplier = gcmMIN(1 << (dstFixPointPos - 0), MAX_MULTIPLIER_NUM);
            vx_uint32 i          = 0;

            for (i = 0; i < 8; i++)
            {
                uinConvertFp16ToInt16_2x8[i + 8] = multiplier;
            }
        }

        status |= vxSetNodeUniform(nodObj, "uniConvertI16toI16_2x8", 1, uniConvertI16toI16_2x8);
        status |= vxSetNodeUniform(nodObj, "uinConvertFp16ToInt16_2x8", 1, uinConvertFp16ToInt16_2x8);
    }
    else if(src0Format == VSI_NN_TYPE_INT16 && src1Format == VSI_NN_TYPE_FLOAT16 && dstFormat == VSI_NN_TYPE_FLOAT16)
    {
        vx_uint32 uniConvertInt16toFp16_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };

        if (src0FixPointPos > 0)
        {
            vx_uint8  postshift = gcmMIN(src0FixPointPos, MAX_POST_SHIFT_BITS);
            uniConvertInt16toFp16_2x8[7] |= (postshift & 0x1F);
        }
        else
        {
            vx_uint32 multiplier = gcmMIN(1 << (0 - src0FixPointPos), MAX_MULTIPLIER_NUM);
            vx_uint32 i          = 0;

            for (i = 0; i < 8; i++)
            {
                uniConvertInt16toFp16_2x8[i + 8] = multiplier;
            }
        }

        status |= vxSetNodeUniform(nodObj, "uniConvertInt16toFp16_2x8", 1, uniConvertInt16toFp16_2x8);
    }
    else if (src0Format == VSI_NN_TYPE_FLOAT16 && src1Format == VSI_NN_TYPE_FLOAT16 && dstFormat == VSI_NN_TYPE_INT8)
    {
        vx_uint32 uinConvertFp16ToInt8_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };

        if (0 > dstFixPointPos)
        {
            vx_uint8  postshift      = gcmMIN(0 - dstFixPointPos, MAX_POST_SHIFT_BITS);
            uinConvertFp16ToInt8_2x8[7] |= (postshift & 0x1F);
        }
        else
        {
            vx_uint32 multiplier = gcmMIN(1 << (dstFixPointPos - 0), MAX_MULTIPLIER_NUM);
            vx_uint32 i          = 0;

            for (i = 0; i < 8; i++)
            {
                uinConvertFp16ToInt8_2x8[i + 8] = multiplier;
            }
        }

        status |= vxSetNodeUniform(nodObj, "uinConvertFp16ToInt8_2x8", 1, uinConvertFp16ToInt8_2x8);
    }

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));

    return status;
}

static vx_param_description_t vxMaximumKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED}
};

#ifdef __cplusplus
extern "C" {
#endif

vx_kernel_description_t vxMaximumKernelInfo_CPU =
{
    VX_KERNEL_ENUM_MAXIMUM,
    "com.vivantecorp.extension.maximum_sw",
    vxMaximumKernel,
    vxMaximumKernelParam,
    (sizeof(vxMaximumKernelParam) / sizeof(vxMaximumKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

#define GEN_MAXIMUM_SH_KERNEL_NAME(SRC0_TYPE, SRC1_TYPE, DST_TYPE) \
    "com.vivantecorp.extension.vxcTensorMaximum_"#SRC0_TYPE#SRC1_TYPE"to"#DST_TYPE

#define GEN_MAXIMUM_SH_KERNEL_NAME_2D(SRC0_TYPE, SRC1_TYPE, DST_TYPE) \
    "com.vivantecorp.extension.vxcTensorMaximum_"#SRC0_TYPE#SRC1_TYPE"to"#DST_TYPE"_2D"

#define TENSOR_MAX_KERNELS(SRC0_TYPE, SRC1_TYPE, DST_TYPE) \
    vx_kernel_description_t vxTensorMaximum_##SRC0_TYPE##SRC1_TYPE##to##DST_TYPE##_Kernel = \
{ \
    _VX_KERNEL_ID, \
    GEN_MAXIMUM_SH_KERNEL_NAME(SRC0_TYPE, SRC1_TYPE, DST_TYPE), \
    NULL, \
    vxMaximumKernelParam, \
    (sizeof(vxMaximumKernelParam) / sizeof(vxMaximumKernelParam[0])), \
    vsi_nn_KernelValidator, \
    NULL, \
    NULL, \
    vxMaximumInitializer, \
    vsi_nn_KernelDeinitializer \
};

#define TENSOR_MAX_KERNELS_2D(SRC0_TYPE, SRC1_TYPE, DST_TYPE) \
    vx_kernel_description_t vxTensorMaximum_##SRC0_TYPE##SRC1_TYPE##to##DST_TYPE##_2D_Kernel = \
{ \
    _VX_KERNEL_ID, \
    GEN_MAXIMUM_SH_KERNEL_NAME_2D(SRC0_TYPE, SRC1_TYPE, DST_TYPE), \
    NULL, \
    vxMaximumKernelParam, \
    (sizeof(vxMaximumKernelParam) / sizeof(vxMaximumKernelParam[0])), \
    vsi_nn_KernelValidator, \
    NULL, \
    NULL, \
    vxMaximumInitializer, \
    vsi_nn_KernelDeinitializer \
};


TENSOR_MAX_KERNELS(F16, F16, F16)
TENSOR_MAX_KERNELS(I8,  F16, I8)
TENSOR_MAX_KERNELS(I8,  F16, F16)
TENSOR_MAX_KERNELS(U8,  F16, U8)
TENSOR_MAX_KERNELS(U8,  F16, F16)
TENSOR_MAX_KERNELS(I8,  I8, I8)
TENSOR_MAX_KERNELS(U8,  U8, U8)
TENSOR_MAX_KERNELS(I16, I16, I16)
TENSOR_MAX_KERNELS(I16, F16, I16)
TENSOR_MAX_KERNELS(I16, F16, F16)
TENSOR_MAX_KERNELS(F16,  F16, U8)
TENSOR_MAX_KERNELS(F16,  F16, I8)

TENSOR_MAX_KERNELS_2D(F16, F16, F16)
TENSOR_MAX_KERNELS_2D(I8,  F16, I8)
TENSOR_MAX_KERNELS_2D(I8,  F16, F16)
TENSOR_MAX_KERNELS_2D(U8,  F16, U8)
TENSOR_MAX_KERNELS_2D(U8,  F16, F16)
TENSOR_MAX_KERNELS_2D(I8,  I8, I8)
TENSOR_MAX_KERNELS_2D(U8,  U8, U8)
TENSOR_MAX_KERNELS_2D(I16, I16, I16)
TENSOR_MAX_KERNELS_2D(I16, F16, I16)
TENSOR_MAX_KERNELS_2D(I16, F16, F16)
TENSOR_MAX_KERNELS_2D(F16,  F16, U8)
TENSOR_MAX_KERNELS_2D(F16,  F16, I8)

#define TENSOR_MAX_KERENLS_NAME(SRC0_TYPE, SRC1_TYPE, DST_TYPE, INSTR) \
    &vxTensorMaximum_##SRC0_TYPE##SRC1_TYPE##to##DST_TYPE##_##INSTR##Kernel,


vx_kernel_description_t * vx_kernel_MAXIMUM_list[] =
{
    &vxMaximumKernelInfo_CPU,
    TENSOR_MAX_KERENLS_NAME(F16, F16, F16, )
    TENSOR_MAX_KERENLS_NAME(I8,  F16, I8, )
    TENSOR_MAX_KERENLS_NAME(I8,  F16, F16, )
    TENSOR_MAX_KERENLS_NAME(U8,  F16, U8, )
    TENSOR_MAX_KERENLS_NAME(U8,  F16, F16, )
    TENSOR_MAX_KERENLS_NAME(I8,  I8, I8, )
    TENSOR_MAX_KERENLS_NAME(U8,  U8, U8, )
    TENSOR_MAX_KERENLS_NAME(I16, I16, I16, )
    TENSOR_MAX_KERENLS_NAME(I16, F16, I16, )
    TENSOR_MAX_KERENLS_NAME(I16, F16, F16, )
    TENSOR_MAX_KERENLS_NAME(F16, F16, U8, )
    TENSOR_MAX_KERENLS_NAME(F16, F16, I8, )

    TENSOR_MAX_KERENLS_NAME(F16, F16, F16, 2D_)
    TENSOR_MAX_KERENLS_NAME(I8,  F16, I8, 2D_)
    TENSOR_MAX_KERENLS_NAME(I8,  F16, F16, 2D_)
    TENSOR_MAX_KERENLS_NAME(U8,  F16, U8, 2D_)
    TENSOR_MAX_KERENLS_NAME(U8,  F16, F16, 2D_)
    TENSOR_MAX_KERENLS_NAME(I8,  I8, I8, 2D_)
    TENSOR_MAX_KERENLS_NAME(U8,  U8, U8, 2D_)
    TENSOR_MAX_KERENLS_NAME(I16, I16, I16, 2D_)
    TENSOR_MAX_KERENLS_NAME(I16, F16, I16, 2D_)
    TENSOR_MAX_KERENLS_NAME(I16, F16, F16, 2D_)
    TENSOR_MAX_KERENLS_NAME(F16, F16, U8, 2D_)
    TENSOR_MAX_KERENLS_NAME(F16, F16, I8, 2D_)
    NULL
};
#ifdef __cplusplus
}
#endif
