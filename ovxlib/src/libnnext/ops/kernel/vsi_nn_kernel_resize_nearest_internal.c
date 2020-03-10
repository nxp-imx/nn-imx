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
#include "vsi_nn_test.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_math.h"
#include "client/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"

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


static vsi_status VX_CALLBACK vxResize_nearest_internalKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
    /* TODO: */
#define ARG_NUM            (1)
#define TENSOR_NUM_INPUT (1)
#define TENSOR_NUM_OUTPUT (1)
#define TENSOR_NUM (TENSOR_NUM_INPUT+TENSOR_NUM_OUTPUT)
#define gcmMIN(x, y)            (((x) <= (y)) ?  (x) :  (y))
    vsi_status status = VX_SUCCESS;
    vx_context context = NULL;
    vsi_nn_tensor_attr_t attr[TENSOR_NUM];
    uint32_t stride_size[TENSOR_NUM][VSI_NN_MAX_DIM_NUM];
    vx_tensor_addressing user_addr[TENSOR_NUM] = {NULL};
    uint8_t *buffer_ptr[TENSOR_NUM] = {NULL};
    vx_tensor tensor[TENSOR_NUM];
    int32_t align_corners, half_pixel_centers;
    vx_float32 width_scale;
    vx_float32 height_scale;
    vx_uint32 input_width, output_width, input_height, output_height;
    vx_uint32 b = 0, d = 0, w = 0, h = 0;
    vx_uint32 output_depth, input_depth;
    vx_uint32 output_batch;
    vx_uint32  output_dims, input_dims;
    vx_uint32 input_width_orig;
    vx_uint32 output_width_orig;
    uint32_t i;

    //prepare data
    context = vxGetContext((vx_reference)node);

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

    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM], &(align_corners), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 1], &(half_pixel_centers), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    input_width  = attr[0].size[0];
    input_height = attr[0].size[1];
    output_width  = attr[1].size[0];
    output_height = attr[1].size[1];
    output_dims  = attr[1].dim_num;
    output_depth = output_dims > 2 ? attr[1].size[2] : 1;
    output_batch = output_dims > 3 ? attr[1].size[3] : 1;
    input_dims  = attr[0].dim_num;
    input_depth = input_dims > 2 ? attr[0].size[2] : 1;
    input_width_orig  = input_width;
    output_width_orig = output_width;

    if (align_corners && output_width > 1)
    {
        width_scale = ((vx_float32)(input_width - 1) * 1.0f) / (vx_float32)(output_width - 1);
    }
    else
    {
        width_scale = ((vx_float32)input_width * 1.0f) / (vx_float32)output_width;
    }

    if (align_corners && output_height > 1)
    {
        height_scale = ((vx_float32)(input_height - 1) * 1.0f) / (vx_float32)(output_height - 1);
    }
    else
    {
        height_scale = ((vx_float32)input_height * 1.0f) / (vx_float32)output_height;
    }

    for (b = 0; b < output_batch; b ++)
    {
        for (d = 0; d < output_depth; d ++)
        {
            vx_int32 input_base = b * input_depth * input_width_orig * input_height + d * input_width_orig * input_height;
            vx_int32 output_base = b * output_depth * output_width_orig * output_height + d * output_width_orig * output_height;

            for (h = 0; h < output_height; h ++)
            {
                vx_float32 input_h;
                vx_uint32  in_y;

                if (half_pixel_centers)
                {
                    input_h = ((vx_float32)h + 0.5f) * height_scale;
                }
                else
                {
                    input_h = h * height_scale;
                }
                if (align_corners)
                {
                    in_y = gcmMIN((vx_uint32)simple_round(input_h), input_height - 1);
                }
                else
                {
                    in_y = gcmMIN((vx_uint32)floorf(input_h), input_height - 1);
                }

                for (w = 0; w < output_width; w ++)
                {
                    vx_float32  input_w;
                    vx_uint32   in_x;
                    vx_int32    in_index;
                    vx_int32    out_index;
                    vx_float32  data;

                    if (half_pixel_centers)
                    {
                        input_w = ((vx_float32)w + 0.5f) * width_scale;
                    }
                    else
                    {
                        input_w = w * width_scale;
                    }
                    if (align_corners)
                    {
                        in_x = gcmMIN((vx_uint32)simple_round(input_w), input_width - 1);
                    }
                    else
                    {
                        in_x = gcmMIN((vx_uint32)floorf(input_w), input_width - 1);
                    }
                    in_index    = in_x + in_y * input_width_orig + input_base;
                    out_index   = w + h * output_width_orig + output_base;
                    data = vsi_nn_DtypeToFloat32_Ex(buffer_ptr[0], in_index, &attr[0].dtype);
                    vsi_nn_Float32ToDtype_Ext(data, buffer_ptr[1], out_index, &attr[1].dtype);
                }
            }
        }
    }

    //save data
    for( i = TENSOR_NUM_INPUT; i < TENSOR_NUM; i ++ )
    {
        status = vsi_nn_copy_tensor_patch(tensor[i], &attr[i], buffer_ptr[i], VX_WRITE_ONLY);
        if (user_addr[i]) vxReleaseTensorAddressing(&(user_addr[i]));
    }
    for( i = 0; i < TENSOR_NUM; i ++ )
    {
        if (buffer_ptr[i]) free(buffer_ptr[i]);
    }

#undef ARG_NUM
#undef TENSOR_NUM_INPUT
#undef TENSOR_NUM_OUTPUT
#undef TENSOR_NUM
#undef gcmMIN
    return status;
} /* _VX_KERNEL_FUNC_KERNEL() */

static vx_param_description_t vxResize_nearest_internalKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

static void getFP32M0AndN(vx_float32 mult, vx_uint16 *M0, vx_int8 *N)
{
    vx_uint32 uintMult          = *((vx_uint32*)(&mult));
    vx_uint32 tmpMultiply       = 0;
    vx_int32  exp               = 0;
    vx_uint32 postShiftBit6to5  = 0;
    vx_uint32 postShift         = 0;
    vx_int8   tmpPostShift      = 0;

    tmpMultiply         = (uintMult & 0x7FFFFF) >> 8;
    *M0                 = (vx_uint16)((1U << 15) + tmpMultiply);

    exp                 = (uintMult & 0x7F800000) >> 23; /* postShift is Scale's exp*/
    tmpPostShift        = 15 - ((vx_int8)exp - 127);
    postShift           = tmpPostShift & 0x1F;
    tmpPostShift        = tmpPostShift >> 5;
    postShiftBit6to5    = tmpPostShift & 3;

    *N = (vx_int8)(((postShiftBit6to5 << 5) | (postShift & 0x1F)));
    *N = (((vx_int32)*N << 25) >> 25);
}

vx_status VX_CALLBACK vxResize_nearest_internalInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    vx_uint32 paraNum
    )
{
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
#define gcmMIN(x, y)            (((x) <= (y)) ?  (x) :  (y))
#define MAX_MULTIPLIER_NUM      (65535)
#define MAX_POST_SHIFT_BITS     (31)
    vx_kernel_execution_parameters_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {1, 1, 1},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread
    vx_status    status             = VX_SUCCESS;
    vx_tensor    input              = (vx_tensor)paramObj[0];
    vx_tensor    output             = (vx_tensor)paramObj[1];
    vx_uint32    depth              = 0;
    vx_int8      srcFixPointPos     = 0;
    vx_int8      dstFixPointPos     = 0;
    vx_float32   input_scale        = 1.0;
    vx_int32     inputZP            = 0;
    vx_float32   output_scale       = 1.0;
    vx_int32     outputZP           = 0;
    vx_enum      srcFormat          = VSI_NN_TYPE_FLOAT16;
    vx_enum      dstFormat          = VSI_NN_TYPE_FLOAT16;
    vsi_nn_tensor_attr_t attr[2];
    int32_t align_corners, half_pixel_centers;
    vx_float32   scale_factor[2];
    vx_uint32    in_width;
    vx_uint32    in_height;
    vx_uint32    out_width;
    vx_uint32    out_height;
    vx_float32   half_pixel_value = 0.0f;
    vx_float32   round_value      = 0.0f;

    vxCopyScalar((vx_scalar)paramObj[2], &(align_corners), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[3], &(half_pixel_centers), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    memset(&attr[0], 0, sizeof(vsi_nn_tensor_attr_t));
    memset(&attr[1], 0, sizeof(vsi_nn_tensor_attr_t));
    status  = vsi_nn_vxGetTensorAttr(input, &attr[0]);
    status |= vsi_nn_vxGetTensorAttr(output, &attr[1]);
    if(status < 0)
    {
        VSILOGE("error-%s,%d\n",__FILE__,__LINE__);
        goto final;
    }

    in_width          = attr[0].size[0];
    in_height         = attr[0].size[1];
    depth             = attr[0].size[2];
    out_width         = attr[1].size[0];
    out_height        = attr[1].size[1];
    srcFormat         = attr[0].dtype.vx_type;
    dstFormat         = attr[1].dtype.vx_type;
    srcFixPointPos    = attr[0].dtype.fl;
    dstFixPointPos    = attr[1].dtype.fl;

    if (srcFormat == VSI_NN_TYPE_BFLOAT16 && dstFormat == VSI_NN_TYPE_BFLOAT16)
    {
        srcFormat = VSI_NN_TYPE_FLOAT16;
        dstFormat = VSI_NN_TYPE_FLOAT16;
    }

    if (align_corners && out_width > 1)
    {
        scale_factor[0] = ((vx_float32)(in_width - 1) * 1.0f) / (vx_float32)(out_width - 1);
    }
    else
    {
        scale_factor[0] = ((vx_float32)in_width * 1.0f) / (vx_float32)out_width;
    }

    if (align_corners && out_height > 1)
    {
        scale_factor[1] = ((vx_float32)(in_height - 1) * 1.0f) / (vx_float32)(out_height - 1);
    }
    else
    {
        scale_factor[1] = ((vx_float32)in_height * 1.0f) / (vx_float32)out_height;
    }

    if (align_corners)
    {
        round_value = 0.5f;
    }
    else
    {
        round_value = 0.0f;
    }

    if (half_pixel_centers)
    {
        half_pixel_value = 0.5f;
    }
    else
    {
        half_pixel_value = 0.0f;
    }

    if (srcFormat == VSI_NN_TYPE_UINT8)
    {
        input_scale    = attr[0].dtype.scale;
        inputZP        = attr[0].dtype.zero_point;
    }
    else if (srcFormat == VSI_NN_TYPE_INT8 || srcFormat == VSI_NN_TYPE_INT16)
    {
        srcFixPointPos = attr[0].dtype.fl;
    }
    else
    {
        input_scale = 1.0f;
        inputZP     = 0;
    }

    if (dstFormat == VSI_NN_TYPE_UINT8)
    {
        output_scale   = attr[1].dtype.scale;
        outputZP       = attr[1].dtype.zero_point;
    }
    else if (dstFormat == VSI_NN_TYPE_INT8 || dstFormat == VSI_NN_TYPE_INT16)
    {
        dstFixPointPos = attr[1].dtype.fl;
    }
    else
    {
        output_scale = 1.0f;
        outputZP     = 0;
    }

    if (srcFormat == VSI_NN_TYPE_FLOAT16 && dstFormat == VSI_NN_TYPE_FLOAT16)
    {
        vx_uint32 uniGetExtractData_2x8[16] = {
            0x00009999, // TCfg
            0x00000000, // ASelt
            0x06040200, 0x00000000, // ABin
            0x0000aaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000300, // AccumType, ConstantType, and PostShift
            0x00100010, 0x00100010, 0x00100010, 0x00100010, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };

        if (scale_factor[0] < 4.0f)
        {
            status  = vxSetNodeUniform(nodObj, "uniGetExtractData_2x8", 1, uniGetExtractData_2x8);
            if (status != VX_SUCCESS) goto final;
        }

        shaderParam.globalWorkScale[0] = 4;
        shaderParam.globalWorkScale[1] = 1;
        shaderParam.globalWorkScale[2] = 1;
        status  = vxSetNodeUniform(nodObj, "scale_xy", 1, scale_factor);
        if (status != VX_SUCCESS) goto final;
    }
    else if ( srcFormat == dstFormat && (srcFormat == VSI_NN_TYPE_INT8 || srcFormat == VSI_NN_TYPE_INT16))
    {
        vx_uint32 uniGetExtractData_2x8[16] = {
            0x00009999, // TCfg
            0x00000000, // ASelt
            0x06040200, 0x00000000, // ABin
            0x0000aaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000300, // AccumType, ConstantType, and PostShift
            0x00080008, 0x00080008, 0x00080008, 0x00080008, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniConvertI8toI8_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };

        if (srcFormat == VSI_NN_TYPE_INT16)
        {
            uniGetExtractData_2x8[8]  = 0x00100010;
            uniGetExtractData_2x8[9]  = 0x00100010;
            uniGetExtractData_2x8[10] = 0x00100010;
            uniGetExtractData_2x8[11] = 0x00100010;
            uniGetExtractData_2x8[12] = 0x00100010;
            uniGetExtractData_2x8[13] = 0x00100010;
            uniGetExtractData_2x8[14] = 0x00100010;
            uniGetExtractData_2x8[15] = 0x00100010;
        }

        if (srcFixPointPos > dstFixPointPos)
        {
            vx_uint8  postshift      = gcmMIN(srcFixPointPos - dstFixPointPos, MAX_POST_SHIFT_BITS);

            uniConvertI8toI8_2x8[7] |= (postshift & 0x1F);
        }
        else
        {
            vx_uint32 multiplier = gcmMIN(1 << (dstFixPointPos - srcFixPointPos), MAX_MULTIPLIER_NUM);
            vx_uint32 i          = 0;

            for (i = 0; i < 8; i++)
            {
                uniConvertI8toI8_2x8[i + 8] = multiplier;
            }
        }

        if (scale_factor[0] < 4.0f)
        {
            status  = vxSetNodeUniform(nodObj, "uniGetExtractData_2x8", 1, uniGetExtractData_2x8);
            if (status != VX_SUCCESS) goto final;
        }

        shaderParam.globalWorkScale[0] = 4;
        shaderParam.globalWorkScale[1] = 1;
        shaderParam.globalWorkScale[2] = 1;

        status  = vxSetNodeUniform(nodObj, "scale_xy", 1, scale_factor);
        status |= vxSetNodeUniform(nodObj, "uniConvertI8toI8_2x8", 1, uniConvertI8toI8_2x8);
        if (status != VX_SUCCESS) goto final;
    }
    else if (srcFormat == VSI_NN_TYPE_UINT8 && dstFormat == VSI_NN_TYPE_UINT8)
    {
        vx_uint16  M0                   = 0;
        vx_int8    postShift            = 0;
        vx_uint32    multAndoutZP[2]    = {0};
        vx_uint32 uniMultiplyAndPostShift_2x8[16] = {
            0xdddddddd, // TCfg
            0x44444444, // ASelt
            0x13121110, 0x17161514, // ABin
            0x11111111, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniGetExtractData_2x8[16] = {
            0x00009999, // TCfg
            0x00000000, // ASelt
            0x06040200, 0x00000000, // ABin
            0x0000aaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000300, // AccumType, ConstantType, and PostShift
            0x00080008, 0x00080008, 0x00080008, 0x00080008, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };

        getFP32M0AndN(input_scale / output_scale, &M0, &postShift);

        multAndoutZP[0] = (vx_uint32)(M0);
        multAndoutZP[1] = (vx_uint32)((outputZP << postShift) - inputZP * M0);

        uniMultiplyAndPostShift_2x8[7] |= (postShift & 0x1F);

        if (scale_factor[0] < 4.0f)
        {
            status  = vxSetNodeUniform(nodObj, "uniGetExtractData_2x8", 1, uniGetExtractData_2x8);
            if (status != VX_SUCCESS) goto final;
        }

        shaderParam.globalWorkScale[0] = 4;
        shaderParam.globalWorkScale[1] = 1;
        shaderParam.globalWorkScale[2] = 1;


        status  = vxSetNodeUniform(nodObj, "scale_xy", 1, scale_factor);
        status |= vxSetNodeUniform(nodObj, "multAndoutZP", 1, multAndoutZP);
        status |= vxSetNodeUniform(nodObj, "uniMultiplyAndPostShift_2x8", 1, uniMultiplyAndPostShift_2x8);
        if (status != VX_SUCCESS) goto final;
    }
    status  = vxSetNodeUniform(nodObj, "half_pixel_value", 1, &half_pixel_value);
    status |= vxSetNodeUniform(nodObj, "round_value", 1, &round_value);
    if (status != VX_SUCCESS) goto final;

    shaderParam.globalWorkSize[0]   = gcmALIGN((out_width  + shaderParam.globalWorkScale[0] - 1) / shaderParam.globalWorkScale[0], 4);
    shaderParam.globalWorkSize[1]   = (out_height + shaderParam.globalWorkScale[1] - 1) / shaderParam.globalWorkScale[1];
    shaderParam.globalWorkSize[2]   = depth;

    status |= vxSetNodeAttribute(nodObj,
    VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS, &shaderParam, sizeof(vx_kernel_execution_parameters_t));
#undef gcmALIGN
#undef gcmMIN
#undef MAX_MULTIPLIER_NUM
#undef MAX_POST_SHIFT_BITS
final:
    return status;
}


#ifdef __cplusplus
extern "C" {
#endif
vx_kernel_description_t vxResize_nearest_internal_CPU =
{
    VX_KERNEL_ENUM_RESIZE_NEAREST_INTERNAL,
    "com.vivantecorp.extension.vxcResize_nearest_sw",
    vxResize_nearest_internalKernel,
    vxResize_nearest_internalKernelParam,
    (sizeof(vxResize_nearest_internalKernelParam) / sizeof(vxResize_nearest_internalKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

#define RESIZE_NEAREST_KERNELS( SRC_TYPE, DST_TYPE) \
vx_kernel_description_t vxResize_nearest_##SRC_TYPE##to##DST_TYPE##_Kernel = \
{ \
    VX_KERNEL_ENUM_RESIZE_NEAREST_INTERNAL, \
    VX_KERNEL_NAME_NEAREST_INTERNAL_##SRC_TYPE##TO##DST_TYPE, \
    NULL, \
    vxResize_nearest_internalKernelParam, \
    (sizeof(vxResize_nearest_internalKernelParam) / sizeof(vxResize_nearest_internalKernelParam[0])), \
    vsi_nn_KernelValidator, \
    NULL, \
    NULL, \
    vxResize_nearest_internalInitializer, \
    vsi_nn_KernelDeinitializer \
};

#define RESIZE_NEAREST_KERNELS_OP( SRC_TYPE, DST_TYPE) \
vx_kernel_description_t vxResize_nearest_##SRC_TYPE##to##DST_TYPE##_OP_Kernel = \
{ \
    VX_KERNEL_ENUM_RESIZE_NEAREST_INTERNAL, \
    VX_KERNEL_NAME_NEAREST_INTERNAL_##SRC_TYPE##TO##DST_TYPE##_OP, \
    NULL, \
    vxResize_nearest_internalKernelParam, \
    (sizeof(vxResize_nearest_internalKernelParam) / sizeof(vxResize_nearest_internalKernelParam[0])), \
    vsi_nn_KernelValidator, \
    NULL, \
    NULL, \
    vxResize_nearest_internalInitializer, \
    vsi_nn_KernelDeinitializer \
};

RESIZE_NEAREST_KERNELS(F16, F16)
RESIZE_NEAREST_KERNELS(I16, I16)
RESIZE_NEAREST_KERNELS(I8, I8)
RESIZE_NEAREST_KERNELS(U8, U8)
RESIZE_NEAREST_KERNELS_OP(F16, F16)
RESIZE_NEAREST_KERNELS_OP(I16, I16)
RESIZE_NEAREST_KERNELS_OP(I8, I8)
RESIZE_NEAREST_KERNELS_OP(U8, U8)

#define RESIZE_NEAREST_INTERNAL_KERNELS_NAME(SRC_TYPE, DST_TYPE, INSTR) \
    &vxResize_nearest_##SRC_TYPE##to##DST_TYPE##_##INSTR##Kernel,

vx_kernel_description_t * vx_kernel_RESIZE_NEAREST_INTERNAL_list[] =
{
    &vxResize_nearest_internal_CPU,
    RESIZE_NEAREST_INTERNAL_KERNELS_NAME(F16, F16, )
    RESIZE_NEAREST_INTERNAL_KERNELS_NAME(I16, I16, )
    RESIZE_NEAREST_INTERNAL_KERNELS_NAME(I8, I8, )
    RESIZE_NEAREST_INTERNAL_KERNELS_NAME(U8, U8, )
    RESIZE_NEAREST_INTERNAL_KERNELS_NAME(F16, F16, OP_)
    RESIZE_NEAREST_INTERNAL_KERNELS_NAME(I16, I16, OP_)
    RESIZE_NEAREST_INTERNAL_KERNELS_NAME(I8, I8, OP_)
    RESIZE_NEAREST_INTERNAL_KERNELS_NAME(U8, U8, OP_)
    NULL
};
#ifdef __cplusplus
}
#endif
