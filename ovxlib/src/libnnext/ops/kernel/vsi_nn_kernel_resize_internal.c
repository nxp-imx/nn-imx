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

static vsi_status VX_CALLBACK vxResize_internalKernel
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
    vx_float32 data00 = .0f, data01 = .0f, data10 = .0f, data11 = .0f, interpolation = .0f;
    vx_uint32 input_width_orig;
    vx_uint32 output_width_orig;
    vx_uint32 index;
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
                vx_float32 input_h = h * height_scale;
                vx_uint32 h0;
                vx_uint32 h1;

                if (half_pixel_centers)
                {
                    input_h = ((vx_float32)h + 0.5f) * height_scale - 0.5f;
                }
                else
                {
                    input_h = h * height_scale;
                }
                h0 = (vx_int32)input_h;
                h1 = gcmMIN(h0 + 1, input_height - 1);
                for (w = 0; w < output_width; w ++)
                {
                    vx_float32 input_w;
                    vx_int32 w0;
                    vx_int32 w1;
                    if (half_pixel_centers)
                    {
                        input_w = ((vx_float32)w + 0.5f) * width_scale - 0.5f;
                    }
                    else
                    {
                        input_w = w * width_scale;
                    }
                    w0 = (vx_int32)input_w;
                    w1 = gcmMIN(w0 + 1, (vx_int32)(input_width - 1));
                    index = input_base + h0 * input_width_orig + w0;
                    data00 = vsi_nn_DtypeToFloat32_Ex(buffer_ptr[0], index,
                                                      &attr[0].dtype);

                    index = input_base + h0 * input_width_orig + w1;
                    data01 = vsi_nn_DtypeToFloat32_Ex(buffer_ptr[0], index,
                                                      &attr[0].dtype);
                    index = input_base + h1 * input_width_orig + w0;
                    data10 = vsi_nn_DtypeToFloat32_Ex(buffer_ptr[0], index,
                                                      &attr[0].dtype);
                    index = input_base + h1 * input_width_orig + w1;
                    data11 = vsi_nn_DtypeToFloat32_Ex(buffer_ptr[0], index,
                                                      &attr[0].dtype);

                    interpolation = data00 * (1 - (input_h - h0)) * (1 - (input_w - w0)) +
                                    data10 * (input_h - h0) * (1 - (input_w - w0)) +
                                    data01 * (1 - (input_h - h0)) * (input_w - w0) +
                                    data11 * (input_h - h0) * (input_w - w0);
                    index = output_base + h * output_width_orig + w;
                    vsi_nn_Float32ToDtype_Ext(interpolation, buffer_ptr[1],
                        index, &attr[1].dtype);
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

static vx_param_description_t vxResize_internalKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

vx_status VX_CALLBACK vxResize_internalInitializer
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
    vx_uint32    depth              = 0;
    vx_int8      srcFixPointPos     = 0;
    vx_int8      dstFixPointPos     = 0;
    vx_float32   input_scale        = 1.0;
    vx_int32     inputZP            = 0;
    vx_float32   output_scale       = 1.0;
    vx_int32     outputZP           = 0;
    vx_enum      srcQntType         = VX_QUANT_NONE;
    vx_enum      dstQntType         = VX_QUANT_NONE;
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

    vxCopyScalar((vx_scalar)paramObj[2], &(align_corners), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[3], &(half_pixel_centers), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    memset(&attr[0], 0, sizeof(vsi_nn_tensor_attr_t));
    memset(&attr[1], 0, sizeof(vsi_nn_tensor_attr_t));
    status  = vsi_nn_vxGetTensorAttr(input, &attr[0]);
    status |= vsi_nn_vxGetTensorAttr(output, &attr[1]);
    if(status < 0)
    {
        VSILOGE("error-%s,%d\n",__FILE__,__LINE__);
        return status;
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
    srcQntType        = attr[0].dtype.qnt_type;
    dstQntType        = attr[1].dtype.qnt_type;

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

    if (half_pixel_centers)
    {
        half_pixel_value = 0.5f;
    }
    else
    {
        half_pixel_value = 0.0f;
    }

    if (srcFormat == VSI_NN_TYPE_UINT8 && srcQntType == VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC)
    {
        input_scale    = attr[0].dtype.scale;
        inputZP        = attr[0].dtype.zero_point;
    }
    else if (srcQntType == VSI_NN_QNT_TYPE_DFP)
    {
        if (srcFixPointPos >= 0)
        {
            input_scale = 1.0f / (vx_float32) (1 << srcFixPointPos);
        }
        else if (srcFixPointPos < 0)
        {
            input_scale = (vx_float32)(1 << -srcFixPointPos);
        }
        inputZP = 0;
    }
    else
    {
        input_scale = 1.0f;
        inputZP     = 0;
    }

    if (dstFormat == VSI_NN_TYPE_UINT8 && dstQntType == VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC)
    {
        output_scale   = attr[1].dtype.scale;
        outputZP       = attr[1].dtype.zero_point;
    }
    else if (dstQntType == VSI_NN_QNT_TYPE_DFP)
    {
        if (dstFixPointPos >= 0)
        {
            output_scale = (vx_float32) (1 << dstFixPointPos);
        }
        else if (dstFixPointPos < 0)
        {
            output_scale = 1.0f / (vx_float32) (1 << -dstFixPointPos);
        }
        outputZP = 0;
    }
    else
    {
        output_scale = 1.0;
        outputZP     = 0;
    }

    shaderParam.globalWorkScale[0] = 4;
    shaderParam.globalWorkScale[1] = 1;
    shaderParam.globalWorkScale[2] = 1;

    if (srcQntType == VSI_NN_QNT_TYPE_DFP)
    {
        vx_float32 dfpScale = input_scale * output_scale;
        vx_uint32 uniConvertDFP2FP32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000300, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        };
        vx_uint32 uniExtact8Bit_2x8[16] = {
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };

        if (srcFormat == VSI_NN_TYPE_INT8 && dstFormat == VSI_NN_TYPE_INT8 && out_width > in_width)
        {
            vx_uint32 uniConvertI32toI16_2x8[16] = {
                0x33333333, // TCfg
                0x11110000, // ASelt
                0x03020100, 0x03020100, // ABin
                0x00000000, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            };
            vx_uint32 uniGetMaskShift_2x8[16] = {
                0x99999999, // TCfg
                0x00000000, // ASelt
                0x03020100, 0x07060504, // ABin
                0x55555555, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            };
            vx_uint32 uniConvertDFP2FP32_part1_4x4[16] = {
                0x09090909, // TCfg
                0x00000000, // ASelt
                0x00150004, 0x00370026, // ABin
                0x0a0a0a0a, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000300, // AccumType, ConstantType, and PostShift
                0x00010001, 0x00000000, 0x00010001, 0x00000000, 0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
            };

            status  = vxSetNodeUniform(nodObj, "uniConvertI32toI16_2x8", 1, uniConvertI32toI16_2x8);
            status |= vxSetNodeUniform(nodObj, "uniGetMaskShift_2x8", 1, uniGetMaskShift_2x8);
            status |= vxSetNodeUniform(nodObj, "uniConvertDFP2FP32_part1_4x4", 1, uniConvertDFP2FP32_part1_4x4);
            status |= vxSetNodeUniform(nodObj, "depth", 1, &depth);
            if (status != VX_SUCCESS) goto final;

            shaderParam.globalWorkScale[2] = depth;
        }
        else if (srcFormat == VSI_NN_TYPE_INT16 && dstFormat == VSI_NN_TYPE_INT16 && out_width > in_width)
        {
            vx_uint32 uniConvertI32toI16_2x8[16] = {
                0x33333333, // TCfg
                0x11110000, // ASelt
                0x03020100, 0x03020100, // ABin
                0x00000000, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            };
            vx_uint32 uniGetMaskShift_2x8[16] = {
                0x99999999, // TCfg
                0x00000000, // ASelt
                0x03020100, 0x07060504, // ABin
                0x55555555, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            };
            vx_uint32 uniConvertDFP2FP32_part1_4x4[16] = {
                0x09090909, // TCfg
                0x00000000, // ASelt
                0x00150004, 0x00370026, // ABin
                0x0a0a0a0a, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000300, // AccumType, ConstantType, and PostShift
                0x00010001, 0x00000000, 0x00010001, 0x00000000, 0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
            };

            status  = vxSetNodeUniform(nodObj, "uniConvertI32toI16_2x8", 1, uniConvertI32toI16_2x8);
            status |= vxSetNodeUniform(nodObj, "uniGetMaskShift_2x8", 1, uniGetMaskShift_2x8);
            status |= vxSetNodeUniform(nodObj, "uniConvertDFP2FP32_part1_4x4", 1, uniConvertDFP2FP32_part1_4x4);
            status |= vxSetNodeUniform(nodObj, "depth", 1, &depth);
            if (status != VX_SUCCESS) goto final;

            shaderParam.globalWorkScale[2] = depth;
        }

        status  = vxSetNodeUniform(nodObj, "uniConvertDFP2FP32_4x4", 1, uniConvertDFP2FP32_4x4);
        status |= vxSetNodeUniform(nodObj, "uniExtact8Bit_2x8", 1, uniExtact8Bit_2x8);
        status |= vxSetNodeUniform(nodObj, "scale_xy", 1, scale_factor);
        status |= vxSetNodeUniform(nodObj, "dfpScale", 1, &dfpScale);
        if (status != VX_SUCCESS) goto final;
    }
    else if (srcFormat == VSI_NN_TYPE_UINT8 && (dstFormat == VSI_NN_TYPE_UINT8 || dstFormat == VSI_NN_TYPE_FLOAT16))
    {
        vx_float32   uint8Scale             = input_scale / output_scale;
        vx_float32   uint8ZP_out            = (vx_float32)outputZP;
        vx_uint32 uniExtact8Bit_2x8[16] = {
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };
        vx_uint32 uniU8SubZPtoFp32_4x4[16] = {
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00000000, 0x00010001, 0x00000000, 0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
        };

        if (dstQntType == VSI_NN_TYPE_FLOAT16)
        {
            status = vxSetNodeUniform(nodObj, "uint8Scale", 1, &input_scale);
            if (status != VX_SUCCESS) goto final;
        }
        else
        {
            if (out_width > in_width)
            {
                vx_uint32 uniConvertI32toI16_2x8[16] = {
                    0x33333333, // TCfg
                    0x11110000, // ASelt
                    0x03020100, 0x03020100, // ABin
                    0x00000000, // BSelt
                    0x00000000, 0x00000000, // BBin
                    0x00002400, // AccumType, ConstantType, and PostShift
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
                };
                vx_uint32 uniGetMaskShift_2x8[16] = {
                    0x99999999, // TCfg
                    0x00000000, // ASelt
                    0x03020100, 0x07060504, // ABin
                    0x55555555, // BSelt
                    0x00000000, 0x00000000, // BBin
                    0x00000400, // AccumType, ConstantType, and PostShift
                    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
                };
                vx_uint32 uniU8SubZPtoFp32_part1_4x4[16] = {
                    0x09090909, // TCfg
                    0x00000000, // ASelt
                    0x00150004, 0x00370026, // ABin
                    0x0a0a0a0a, // BSelt
                    0x00000000, 0x00000000, // BBin
                    0x00000400, // AccumType, ConstantType, and PostShift
                    0x00010001, 0x00000000, 0x00010001, 0x00000000, 0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
                };

                status  = vxSetNodeUniform(nodObj, "uniConvertI32toI16_2x8", 1, uniConvertI32toI16_2x8);
                status |= vxSetNodeUniform(nodObj, "uniGetMaskShift_2x8", 1, uniGetMaskShift_2x8);
                status |= vxSetNodeUniform(nodObj, "uniU8SubZPtoFp32_part1_4x4", 1, uniU8SubZPtoFp32_part1_4x4);
                status |= vxSetNodeUniform(nodObj, "depth", 1, &depth);
                if (status != VX_SUCCESS) goto final;

                shaderParam.globalWorkScale[2] = depth;
            }

            status  = vxSetNodeUniform(nodObj, "uniExtact8Bit_2x8", 1, uniExtact8Bit_2x8);
            status |= vxSetNodeUniform(nodObj, "uint8Scale", 1, &uint8Scale);
            status |= vxSetNodeUniform(nodObj, "output_ZP", 1, &uint8ZP_out);
            if (status != VX_SUCCESS) goto final;
        }

        status  = vxSetNodeUniform(nodObj, "uniU8SubZPtoFp32_4x4", 1, uniU8SubZPtoFp32_4x4);
        status |= vxSetNodeUniform(nodObj, "scale_xy", 1, scale_factor);
        if (status != VX_SUCCESS) goto final;

        status = vxSetNodeUniform(nodObj, "input_ZP", 1, &inputZP);
        if (status != VX_SUCCESS) goto final;
    }
    else if (srcFormat == VSI_NN_TYPE_FLOAT16 && (dstFormat == VSI_NN_TYPE_UINT8 || dstFormat == VSI_NN_TYPE_FLOAT16))
    {
        vx_float32   uint8Scale             = 1.0f / output_scale;
        vx_float32   uint8ZP_out            = (vx_float32)outputZP;
        vx_uint32 uniExtact8Bit_2x8[16] = {
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
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
        vx_uint32 uniRightSubLeft_4x4[16] = {
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00110000, 0x00330022, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000 // Constant
        };
        vx_uint32 uniExtactHalf8_2x8[16] = {
            0x11111111, // TCfg
            0x11110000, // ASelt
            0x06040200, 0x06040200, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
        };

        if (srcFormat == VSI_NN_TYPE_FLOAT16 && dstFormat == VSI_NN_TYPE_FLOAT16 && out_width > in_width)
        {
            vx_uint32 uniConvertI32toI16_2x8[16] = {
                0x33333333, // TCfg
                0x11110000, // ASelt
                0x03020100, 0x03020100, // ABin
                0x00000000, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            };
            vx_uint32 uniGetMaskShift_2x8[16] = {
                0x99999999, // TCfg
                0x00000000, // ASelt
                0x03020100, 0x07060504, // ABin
                0x55555555, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            };
            vx_uint32 uniFp16toFp32_part1_4x4[16] = {
                0x09090909, // TCfg
                0x00000000, // ASelt
                0x00150004, 0x00370026, // ABin
                0x0a0a0a0a, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000, 0x3c003c00, 0x00000000 // Constant
            };

            status  = vxSetNodeUniform(nodObj, "uniConvertI32toI16_2x8", 1, uniConvertI32toI16_2x8);
            status |= vxSetNodeUniform(nodObj, "uniGetMaskShift_2x8", 1, uniGetMaskShift_2x8);
            status |= vxSetNodeUniform(nodObj, "uniFp16toFp32_part1_4x4", 1, uniFp16toFp32_part1_4x4);
            status |= vxSetNodeUniform(nodObj, "uniExtactHalf8_2x8", 1, uniExtactHalf8_2x8);
            status |= vxSetNodeUniform(nodObj, "depth", 1, &depth);
            if (status != VX_SUCCESS) goto final;

            shaderParam.globalWorkScale[2] = depth;
        }
        else if (dstFormat == VSI_NN_TYPE_FLOAT16)
        {
            status = vxSetNodeUniform(nodObj, "uniExtactHalf8_2x8", 1, uniExtactHalf8_2x8);
            if (status != VX_SUCCESS) goto final;
        }
        else
        {
            status  = vxSetNodeUniform(nodObj, "uniExtact8Bit_2x8", 1, uniExtact8Bit_2x8);
            status |= vxSetNodeUniform(nodObj, "uint8Scale", 1, &uint8Scale);
            status |= vxSetNodeUniform(nodObj, "output_ZP", 1, &uint8ZP_out);
            if (status != VX_SUCCESS) goto final;
        }

        status  = vxSetNodeUniform(nodObj, "uniFp16toFp32_4x4", 1, uniFp16toFp32_4x4);
        status |= vxSetNodeUniform(nodObj, "uniRightSubLeft_4x4", 1, uniRightSubLeft_4x4);
        status |= vxSetNodeUniform(nodObj, "scale_xy", 1, scale_factor);
        if (status != VX_SUCCESS) goto final;
    }
    else if (srcFormat == VSI_NN_TYPE_BFLOAT16 && dstFormat == VSI_NN_TYPE_BFLOAT16)
    {
        if (out_width > in_width)
        {
            vx_uint32 uniConvertI32toI16_2x8[16] = {
                0x33333333, // TCfg
                0x11110000, // ASelt
                0x03020100, 0x03020100, // ABin
                0x00000000, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            };
            vx_uint32 uniGetMaskShift_2x8[16] = {
                0x99999999, // TCfg
                0x00000000, // ASelt
                0x03020100, 0x07060504, // ABin
                0x55555555, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
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
            status  = vxSetNodeUniform(nodObj, "scale_xy", 1, scale_factor);
            status |= vxSetNodeUniform(nodObj, "uniConvertI32toI16_2x8", 1, uniConvertI32toI16_2x8);
            status |= vxSetNodeUniform(nodObj, "uniGetMaskShift_2x8", 1, uniGetMaskShift_2x8);
            status |= vxSetNodeUniform(nodObj, "uniConvBF16toF32_Part0_2x8",  1, uniConvBF16toF32_Part0_2x8);
            status |= vxSetNodeUniform(nodObj, "uniConvBF16toF32_Part1_2x8", 1, uniConvBF16toF32_Part1_2x8);
            status |= vxSetNodeUniform(nodObj, "depth", 1, &depth);
            if (status != VX_SUCCESS) goto final;

            shaderParam.globalWorkScale[2] = depth;
        }
        else
        {
            vx_uint32 uniConvBF16toF32_odd_2x8[16] = {
                0x11111111, // TCfg
                0x01010101, // ASelt
                0x02050004, 0x06070406, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
            };
            vx_uint32 uniConvBF16toF32_even_2x8[16] = {
                0x11111111, // TCfg
                0x01010101, // ASelt
                0x03050104, 0x07070506, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
            };

            status  = vxSetNodeUniform(nodObj, "scale_xy", 1, scale_factor);
            status |= vxSetNodeUniform(nodObj, "uniConvBF16toF32_odd_2x8",  1, uniConvBF16toF32_odd_2x8);
            status |= vxSetNodeUniform(nodObj, "uniConvBF16toF32_even_2x8", 1, uniConvBF16toF32_even_2x8);
            if (status != VX_SUCCESS) goto final;
        }
    }
    else
    {
        VSILOGE("input or output's format is not support");
        goto final;
    }
    status  = vxSetNodeUniform(nodObj, "half_pixel_value", 1, &half_pixel_value);

    shaderParam.globalWorkSize[0]   = gcmALIGN((out_width  + shaderParam.globalWorkScale[0] - 1) / shaderParam.globalWorkScale[0], 4);
    shaderParam.globalWorkSize[1]   = (out_height + shaderParam.globalWorkScale[1] - 1) / shaderParam.globalWorkScale[1];
    shaderParam.globalWorkSize[2]   = depth / shaderParam.globalWorkScale[2];

    status |= vxSetNodeAttribute(nodObj,
    VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS, &shaderParam, sizeof(vx_kernel_execution_parameters_t));

#undef gcmALIGN
final:
    return status;
}


#ifdef __cplusplus
extern "C" {
#endif
vx_kernel_description_t vxResize_internal_CPU =
{
    VX_KERNEL_ENUM_RESIZE_INTERNAL,
    "com.vivantecorp.extension.vxcResize_sw",
    vxResize_internalKernel,
    vxResize_internalKernelParam,
    (sizeof(vxResize_internalKernelParam) / sizeof(vxResize_internalKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

#define RESIZE_INTERNAL_KERNELS( SRC_TYPE, DST_TYPE) \
vx_kernel_description_t vxResize_internal_##SRC_TYPE##to##DST_TYPE##_Kernel = \
{ \
    VX_KERNEL_ENUM_RESIZE_INTERNAL, \
    VX_KERNEL_NAME_RESIZE_INTERNAL_##SRC_TYPE##TO##DST_TYPE, \
    NULL, \
    vxResize_internalKernelParam, \
    (sizeof(vxResize_internalKernelParam) / sizeof(vxResize_internalKernelParam[0])), \
    vsi_nn_KernelValidator, \
    NULL, \
    NULL, \
    vxResize_internalInitializer, \
    vsi_nn_KernelDeinitializer \
};

#define RESIZE_INTERNAL_KERNELS_UP( SRC_TYPE, DST_TYPE) \
vx_kernel_description_t vxResize_internal_##SRC_TYPE##to##DST_TYPE##_UP_Kernel = \
{ \
    VX_KERNEL_ENUM_RESIZE_INTERNAL, \
    VX_KERNEL_NAME_RESIZE_INTERNAL_##SRC_TYPE##TO##DST_TYPE##_UP, \
    NULL, \
    vxResize_internalKernelParam, \
    (sizeof(vxResize_internalKernelParam) / sizeof(vxResize_internalKernelParam[0])), \
    vsi_nn_KernelValidator, \
    NULL, \
    NULL, \
    vxResize_internalInitializer, \
    vsi_nn_KernelDeinitializer \
};

RESIZE_INTERNAL_KERNELS(I8, I8)
RESIZE_INTERNAL_KERNELS(I16, I16)
RESIZE_INTERNAL_KERNELS(U8, F16)
RESIZE_INTERNAL_KERNELS(U8, U8)
RESIZE_INTERNAL_KERNELS(F16, F16)
RESIZE_INTERNAL_KERNELS(F16, U8)
RESIZE_INTERNAL_KERNELS(BF16, BF16)
RESIZE_INTERNAL_KERNELS_UP(I8, I8)
RESIZE_INTERNAL_KERNELS_UP(I16, I16)
RESIZE_INTERNAL_KERNELS_UP(U8, U8)
RESIZE_INTERNAL_KERNELS_UP(F16, F16)
RESIZE_INTERNAL_KERNELS_UP(BF16, BF16)

#define RESIZE_INTERNAL_KERNELS_NAME(SRC_TYPE, DST_TYPE, INSTR) \
    &vxResize_internal_##SRC_TYPE##to##DST_TYPE##_##INSTR##Kernel,

vx_kernel_description_t * vx_kernel_RESIZE_INTERNAL_list[] =
{
    &vxResize_internal_CPU,
    RESIZE_INTERNAL_KERNELS_NAME(I8, I8, )
    RESIZE_INTERNAL_KERNELS_NAME(I16, I16, )
    RESIZE_INTERNAL_KERNELS_NAME(U8, F16, )
    RESIZE_INTERNAL_KERNELS_NAME(U8, U8, )
    RESIZE_INTERNAL_KERNELS_NAME(F16, F16, )
    RESIZE_INTERNAL_KERNELS_NAME(F16, U8, )
    RESIZE_INTERNAL_KERNELS_NAME(BF16, BF16, )
    RESIZE_INTERNAL_KERNELS_NAME(I8, I8, UP_)
    RESIZE_INTERNAL_KERNELS_NAME(I16, I16, UP_)
    RESIZE_INTERNAL_KERNELS_NAME(U8, U8, UP_)
    RESIZE_INTERNAL_KERNELS_NAME(F16, F16, UP_)
    RESIZE_INTERNAL_KERNELS_NAME(BF16, BF16, UP_)
    NULL
};
#ifdef __cplusplus
}
#endif
