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

#define _VX_KERNEL_NAME         (VX_KERNEL_NAME_PRE_PROCESS_YUV2RBG_U8)

static vx_uint32 vxcGetTypeSize(vx_enum format)
{
    switch(format)
    {
    case VX_TYPE_INT8:
    case VX_TYPE_UINT8:
        return 1;
    case VX_TYPE_INT16:
    case VX_TYPE_UINT16:
        return 2;
    case VX_TYPE_INT32:
    case VX_TYPE_UINT32:
        return 4;
    case VX_TYPE_INT64:
    case VX_TYPE_UINT64:
        return 8;
    case VX_TYPE_FLOAT32:
        return 4;
    case VX_TYPE_FLOAT64:
        return 8;
    case VX_TYPE_ENUM:
        return 4;
    case VX_TYPE_FLOAT16:
        return 2;
    }

    return 4;
}

vx_status tensorRead(vx_context context, vx_tensor ts, void *buf)
{
    vx_uint32  ts_size[4];
    vx_uint32 input_stride_size[4];
    vx_uint32 output_num,num_of_dim;
    vx_tensor_addressing input_user_addr = NULL;
    vx_status status = VX_FAILURE;
    vx_uint32 i = 0;
    void *dataBuf = (vx_uint16 *)buf;
    vx_uint32 dataFormat;

    status = vxQueryTensor(ts, VX_TENSOR_NUM_OF_DIMS, &num_of_dim, sizeof(num_of_dim));
    status |= vxQueryTensor(ts, VX_TENSOR_DIMS, ts_size, sizeof(ts_size));
    status |= vxQueryTensor(ts, VX_TENSOR_DATA_TYPE, &dataFormat, sizeof(dataFormat));

    input_stride_size[0] = vxcGetTypeSize(dataFormat);
    output_num = ts_size[0];
    for (i=1; i< num_of_dim; i++)
    {
        input_stride_size[i] = input_stride_size[i - 1] * ts_size[i - 1];
        output_num *= ts_size[i];
    }

    if(dataBuf == NULL)
    {
        printf("TensorRead fail! input empty \n");
        return VX_FAILURE;
    }

    input_user_addr = vxCreateTensorAddressing(
        context,
        ts_size,
        input_stride_size,
        num_of_dim
        );
    status = vxCopyTensorPatch(
        ts,
        NULL,
        input_user_addr,
        dataBuf,
        VX_READ_ONLY,
        0
        );
    vxReleaseTensorAddressing(&input_user_addr);
    if(status < 0)
    {
        free(dataBuf);
        dataBuf = NULL;
        printf("TensorRead fail! status = %d\n",status);
        return status;
    }

    return VX_SUCCESS;
}

#define clipMIN(x, y)            (((x) <= (y)) ?  (x) :  (y))
#define clipMAX(x, y)            (((x) >= (y)) ?  (x) :  (y))
#define CLAMP(x)                 (clipMIN(clipMAX((x), (0)), (255)))

#define DESCALE(x) (((x) + (1<<19)) >> 20)

static vsi_status VX_CALLBACK vxYuv2rbgKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
#define ARG_NUM            (10)
#define TENSOR_NUM_INPUT (3)
#define TENSOR_NUM_OUTPUT (1)
#define TENSOR_NUM (TENSOR_NUM_INPUT+TENSOR_NUM_OUTPUT)

    vsi_status status = VSI_FAILURE;
    vx_context context = NULL;
    vx_tensor input[TENSOR_NUM_INPUT] = {0};
    vx_tensor output[TENSOR_NUM_OUTPUT] = {0};
    uint8_t *u8_in_buffer[TENSOR_NUM_INPUT] = {0};
    uint8_t *u8_out_buffer[TENSOR_NUM_OUTPUT] = {0};
    vsi_nn_tensor_attr_t in_attr[TENSOR_NUM_INPUT] = {0};
    vsi_nn_tensor_attr_t out_attr[TENSOR_NUM_OUTPUT] = {0};
    uint32_t in_elements[TENSOR_NUM_INPUT] = {0};
    uint32_t out_elements[TENSOR_NUM_OUTPUT]= {0};
    vx_uint32  dst_size[4]   = {0, 0, 0, 0};
    vx_uint32  src_size[4]   = {0, 0, 0, 0};
    vx_float32 outputScale      = 1.0;
    vx_int32   output_ZP        = 0;
    vx_tensor_addressing output_user_addr = NULL;
    uint32_t output_stride_size[4] = {0};
    int32_t output_dims = 0;
    vsi_nn_type_e outputFormat;

    int32_t xRatio, yRatio, xOffset, yOffset;
    float rMean, gMean, bMean, var;
    int32_t order/*, trans*/;

    int32_t i;

    /* prepare data */
    context = vxGetContext((vx_reference)node);

    for(i = 0; i < TENSOR_NUM_INPUT; i ++)
    {
        input[i] = (vx_tensor)paramObj[i];
        status = vsi_nn_vxGetTensorAttr(input[i], &in_attr[i]);
        TEST_CHECK_STATUS(status, final);
        in_elements[i] = vsi_nn_vxGetTensorElementNum(&in_attr[i]);
        u8_in_buffer[i] = (uint8_t *)malloc(in_elements[i] * sizeof(uint8_t));
        status = tensorRead(context, input[i], u8_in_buffer[i]);
        TEST_CHECK_STATUS(status, final);
        if(i == 0)
            status |= vxQueryTensor(input[i], VX_TENSOR_DIMS, src_size, sizeof(src_size));
    }
    for(i = 0; i < TENSOR_NUM_OUTPUT; i ++)
    {
        output[i] = (vx_tensor)paramObj[i + TENSOR_NUM_INPUT];
        status = vsi_nn_vxGetTensorAttr(output[i], &out_attr[i]);
        TEST_CHECK_STATUS(status, final);
        out_elements[i] = vsi_nn_vxGetTensorElementNum(&out_attr[i]);
        u8_out_buffer[i]= (uint8_t *)malloc(out_elements[i] * sizeof(uint8_t));
        memset(u8_out_buffer[i], 0, out_elements[i] * sizeof(uint8_t));

        status |= vxQueryTensor(output[i], VX_TENSOR_NUMBER_OF_DIMS, &output_dims, sizeof(output_dims));
        status |= vxQueryTensor(output[i], VX_TENSOR_DIMS, dst_size, sizeof(dst_size));
        status |= vxQueryTensor(output[i], VX_TENSOR_SCALE, &outputScale, sizeof(outputScale));
        status |= vxQueryTensor(output[i], VX_TENSOR_ZERO_POINT, &output_ZP, sizeof(output_ZP));
        status |= vxQueryTensor(output[i], VX_TENSOR_DATA_TYPE, &outputFormat, sizeof(outputFormat));
    }

    output_stride_size[0] = vsi_nn_GetTypeBytes(outputFormat);
    for (i=1; i< output_dims; i++)
    {
        output_stride_size[i] = output_stride_size[i-1] * dst_size[i-1];
    }
    status |= vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM], &(xRatio),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    status |= vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 1], &(yRatio),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    status |= vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 2], &(xOffset),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    status |= vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 3], &(yOffset),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    status |= vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 4], &(rMean),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    status |= vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 5], &(gMean),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    status |= vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 6], &(bMean),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    status |= vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 7], &(var),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    status |= vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 8], &(order),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    /*status |= vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 9], &(trans),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);*/
    TEST_CHECK_STATUS(status, final);
    /* TODO: Add CPU kernel implement */
    /* example code : copy data form input tensor to output tensor*/
    {
        vx_uint8 rline1[2], rline2[2];
        vx_uint8 gline1[2], gline2[2];
        vx_uint8 bline1[2], bline2[2];
        vx_int32 dx, dy, dz;
        vx_int32 src_width = src_size[0];
        vx_int32 src_height = src_size[1];
        vx_int32 subWidth = src_width >> 1;
        vx_int32 subHeight = src_height >> 1;
        vx_int32 dst_width = dst_size[0];
        vx_int32 dst_height = dst_size[1];
        vx_int32 stride = dst_width * dst_height;
        vx_int32 rOffset = 0;
        vx_int32 gOffset = 1 * stride;
        vx_int32 bOffset = 2 * stride;
        vx_int32 subIdx = 0;
        vx_int32 C, D, E;
        vx_uint8 R, G, B;
        vx_uint8 *ySrc = u8_in_buffer[0];
        vx_uint8 *uSrc = u8_in_buffer[1];
        vx_uint8 *vSrc = u8_in_buffer[2];
        vx_uint8 *dst = u8_out_buffer[0];

        if(order)
        {
            rOffset = 2 * stride;
            bOffset = 0;
        }

        for ( dz = 0; dz < 1; dz ++)
        {
            for ( dy = 0; dy < (vx_int32)dst_size[1]; dy ++)
            {
                for ( dx = 0; dx < (vx_int32)dst_size[0]; dx ++)
                {
                    //flag = 0
                    // for x
                    vx_int32 fx = (dx * xRatio + (xRatio >> 1)) - (1 << 14);
                    vx_int32 sx = fx & 0xffff8000; // Floor
                    vx_int32 fy, sy;
                    vx_int32 source_index, dstR_idx, dstG_idx, dstB_idx, output_index;
                    vx_float32 final;

                    fx -= sx;
                    sx = sx >> 15;

                    sx = sx < 0 ? 0 : sx;
                    sx = sx > src_width ? src_width - 1: sx;

                    fx = (fx +(1 << 4)) >> 5;

                    // for y
                    fy = (dy * yRatio + (yRatio >> 1)) - (1<< 14);
                    sy = fy & 0xffff8000; // Floor
                    fy -= sy;
                    sy = sy >> 15;

                    sy = sy < 0 ? 0 : sy;
                    fy = fy < 0 ? 0 : fy;

                    fy = (fy + (1<< 4)) >> 5;

                    sx += xOffset;
                    sy += yOffset;
                    source_index = (sx + sy * src_width + dz * src_width * src_height + 0);
                    output_index = dx + dy * dst_width;

                    dstR_idx = output_index + rOffset;
                    dstG_idx = output_index + gOffset;
                    dstB_idx = output_index + bOffset;

                    subIdx = ((sx >> 1) + (sy >> 1) * subWidth + dz * subWidth * subHeight + 0);

                    if(dx <= dst_width - 1 && dy <= dst_height - 1 && (xRatio != (1 << 15) || yRatio != (1 << 15)))
                    {
                        //vx_int32 offset = xOffset + yOffset * src_width;
                        //vx_uint8 result;
                        vx_int32 temp1;
                        vx_int32 temp2;

                        C = ySrc[source_index] - 16;
                        D = uSrc[subIdx] - 128;
                        E = vSrc[subIdx] - 128;

                        rline1[0]            = CLAMP((298 * C + 409 * E + 128) >> 8);
                        gline1[0]            = CLAMP((298 * C - 100* D - 208 * E + 128) >> 8);
                        bline1[0]            = CLAMP((298 * C + 516 * D + 128) >> 8);

                        // right
                        C = ySrc[source_index + 1] - 16;
                        subIdx = (((sx + 1) >> 1) + (sy >> 1) * subWidth + dz * subWidth * subHeight);
                        //subIdx = (((sx + 0) >> 1) + (sy >> 1) * subWidth + dz * subWidth * subHeight);
                        D = uSrc[subIdx] - 128;
                        E = vSrc[subIdx] - 128;

                        rline1[1]            = CLAMP((298 * C + 409 * E + 128) >> 8);
                        gline1[1]            = CLAMP((298 * C - 100* D - 208 * E + 128) >> 8);
                        bline1[1]            = CLAMP((298 * C + 516 * D + 128) >> 8);

                        // below
                        C = ySrc[source_index + src_width] - 16;
                        subIdx = (((sx + 0) >> 1) + ((sy + 1) >> 1) * subWidth + dz * subWidth * subHeight);
                        D = uSrc[subIdx] - 128;
                        E = vSrc[subIdx] - 128;

                        rline2[0]            = CLAMP((298 * C + 409 * E + 128) >> 8);
                        gline2[0]            = CLAMP((298 * C - 100* D - 208 * E + 128) >> 8);
                        bline2[0]            = CLAMP((298 * C + 516 * D + 128) >> 8);

                        // below right
                        C = ySrc[source_index + src_width + 1] - 16;
                        subIdx = (((sx + 1) >> 1) + ((sy + 1) >> 1) * subWidth + dz * subWidth * subHeight);
                        //subIdx = (((sx + 0) >> 1) + ((sy + 1) >> 1) * subWidth + dz * subWidth * subHeight);
                        D = uSrc[subIdx] - 128;
                        E = vSrc[subIdx] - 128;

                        rline2[1]            = CLAMP((298 * C + 409 * E + 128) >> 8);
                        gline2[1]            = CLAMP((298 * C - 100* D - 208 * E + 128) >> 8);
                        bline2[1]            = CLAMP((298 * C + 516 * D + 128) >> 8);

                        //B
                        temp1 = fx * (bline1[1] - bline1[0]) + (bline1[0] << 10);
                        temp2 = fx * (bline2[1] - bline2[0]) + (bline2[0] << 10);
                        temp1 = fy * (temp2 - temp1) + (temp1 << 10);
                        B = (vx_uint8)(DESCALE(temp1));
                        final = (B - bMean) * var;
                        dst[dstB_idx] = vsi_nn_Fp32ToAffine(final, outputScale, output_ZP, VSI_NN_TYPE_UINT8);

                        //G
                        temp1 = fx * (gline1[1] - gline1[0]) + (gline1[0] << 10);
                        temp2 = fx * (gline2[1] - gline2[0]) + (gline2[0] << 10);
                        temp1 = fy * (temp2 - temp1) + (temp1 << 10);

                        G = (vx_uint8)(DESCALE(temp1));
                        final = (G - gMean) * var;
                        dst[dstG_idx] = vsi_nn_Fp32ToAffine(final, outputScale, output_ZP, VSI_NN_TYPE_UINT8);

                        // R
                        temp1 = fx * (rline1[1] - rline1[0]) + (rline1[0] << 10);
                        temp2 = fx * (rline2[1] - rline2[0]) + (rline2[0] << 10);
                        temp1 = fy * (temp2 - temp1) + (temp1 << 10);
                        R = (vx_uint8)(DESCALE(temp1));
                        final = (R - rMean) * var;
                        dst[dstR_idx] = vsi_nn_Fp32ToAffine(final, outputScale, output_ZP, VSI_NN_TYPE_UINT8);
                    }
                    else
                    {
                        // do conversion
                        {
                            C = ySrc[source_index] - 16;
                            D = uSrc[subIdx] - 128;
                            E = vSrc[subIdx] - 128;

                            R            = CLAMP((298 * C + 409 * E + 128) >> 8);
                            G            = CLAMP((298 * C - 100* D - 208 * E + 128) >> 8);
                            B            = CLAMP((298 * C + 516 * D + 128) >> 8);
                        }

                        final = (B - bMean) * var;
                        dst[dstB_idx] = vsi_nn_Fp32ToAffine(final, outputScale, output_ZP, VSI_NN_TYPE_UINT8);
                        final = (G - gMean) * var;
                        dst[dstG_idx] = vsi_nn_Fp32ToAffine(final, outputScale, output_ZP, VSI_NN_TYPE_UINT8);
                        final = (R - rMean) * var;
                        dst[dstR_idx] = vsi_nn_Fp32ToAffine(final, outputScale, output_ZP, VSI_NN_TYPE_UINT8);
                    }
                }
            }
        }
    }

    /* save data */
    output_user_addr = vxCreateTensorAddressing(context, dst_size,
        output_stride_size, output_dims);
    vxCopyTensorPatch(output[0], NULL, output_user_addr, u8_out_buffer[0], VX_WRITE_ONLY, 0);

final:
    for (i = 0; i < TENSOR_NUM_INPUT; i++)
    {
        if (u8_in_buffer[i]) free(u8_in_buffer[i]);
    }
    for(i = 0; i < TENSOR_NUM_OUTPUT; i++)
    {
        if (u8_out_buffer[i]) free(u8_out_buffer[i]);
    }
    if(output_user_addr) vxReleaseTensorAddressing(&output_user_addr);
    return status;
} /* _VX_KERNEL_FUNC_KERNEL() */

static vx_param_description_t vxYuv2rbgKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

vx_status VX_CALLBACK vxYuv2rbgInitializer
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

    vx_tensor  output           = (vx_tensor)paramObj[3];
    vx_scalar reorder_s         = (vx_scalar)paramObj[12];

    vx_enum  outFormat;
    vx_uint32 width             = 0;
    vx_uint32 height            = 0;
    vx_float32 outputScale      = 1.0;
    vx_int32   output_ZP        = 0;
    vx_int32 reorder = 0;
    vx_int32 order1 = 2;

    vx_uint32  dst_size[4]   = {0, 0, 0, 0};

    status = vxQueryTensor(output, VX_TENSOR_DIMS, dst_size, sizeof(dst_size));
    status |= vxQueryTensor(output, VX_TENSOR_DATA_TYPE, &outFormat, sizeof(outFormat));
    status |= vxQueryTensor(output, VX_TENSOR_SCALE, &outputScale, sizeof(outputScale));
    status |= vxQueryTensor(output, VX_TENSOR_ZERO_POINT, &output_ZP, sizeof(output_ZP));
    status |= vxCopyScalar(reorder_s, (void*)&reorder, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    if(reorder != 0)
    {
        reorder = 2;
        order1 = 0;
    }

    if(outFormat == VX_TYPE_UINT8)
    {
        outputScale = 1.0f / outputScale;
    }

    width = dst_size[0];
    height = dst_size[1];

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkOffset[2] = 0;
    if(outFormat == VX_TYPE_UINT8)
        shaderParam.globalWorkScale[0]  = 16;
    else if(outFormat == VX_TYPE_INT16 || outFormat == VX_TYPE_FLOAT16)
        shaderParam.globalWorkScale[0]  = 8;
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.globalWorkScale[2]  = 1;
    shaderParam.localWorkSize[0]    = 16;
    shaderParam.localWorkSize[1]    = 1;
    shaderParam.localWorkSize[2]    = 1;
    shaderParam.globalWorkSize[0]   = gcmALIGN((width + shaderParam.globalWorkScale[0] - 1)
                                        / shaderParam.globalWorkScale[0], shaderParam.localWorkSize[0]);
    shaderParam.globalWorkSize[1]   = gcmALIGN((height + shaderParam.globalWorkScale[1] - 1)
                                        / shaderParam.globalWorkScale[1], shaderParam.localWorkSize[1]);
    shaderParam.globalWorkSize[2]   = 1;

    if(outFormat == VX_TYPE_UINT8)
    {
        vx_uint32 uniPackBG0_2x8[16] = {
            0x11011011, // TCfg
            0x10010010, // ASelt
            0x01000000, 0x02020001, // ABin
            0x22022022, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000000, 0x00000001, 0x00000001, 0x00000000, 0x00000001, 0x00000001 // Constant
        };

        vx_uint32 uniPackTmpAndR_2x8[16] = {
            0x11111111, // TCfg
            0x00100100, // ASelt
            0x03000100, 0x07060104, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };

        vx_uint32 uniPackRB0_2x8[16] = {
            0x11011011, // TCfg
            0x10010010, // ASelt
            0x03000302, 0x05040004, // ABin
            0x22022022, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000000, 0x00000001, 0x00000001, 0x00000000, 0x00000001, 0x00000001 // Constant
        };
        vx_uint32 uniPackTmp0AndG_2x8[16] = {
            0x11111111, // TCfg
            0x00100100, // ASelt
            0x03030100, 0x07060404, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };

        vx_uint32 uniPackGR1_2x8[16] = {
            0x11011011, // TCfg
            0x10010010, // ASelt
            0x06000505, 0x07070006, // ABin
            0x22022022, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000000, 0x00000001, 0x00000001, 0x00000000, 0x00000001, 0x00000001 // Constant
        };
        vx_uint32 uniPackTmp1AndB_2x8[16] = {
            0x11111111, // TCfg
            0x00100100, // ASelt
            0x03060100, 0x07060704, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };
        vx_uint32 uniPackBG1_2x8[16] = {
            0x11011011, // TCfg
            0x10010010, // ASelt
            0x09000808, 0x0a0a0009, // ABin
            0x22022022, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000000, 0x00000001, 0x00000001, 0x00000000, 0x00000001, 0x00000001 // Constant
        };
        vx_uint32 uniPackTmp1AndR_2x8[16] = {
            0x11111111, // TCfg
            0x00100100, // ASelt
            0x03080100, 0x07060904, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };

        vx_uint32 uniPackRB2_2x8[16] = {
            0x11011011, // TCfg
            0x10010010, // ASelt
            0x0b000b0a, 0x0d0c000c, // ABin
            0x22022022, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000000, 0x00000001, 0x00000001, 0x00000000, 0x00000001, 0x00000001 // Constant
        };
        vx_uint32 uniPackTmp2AndG_2x8[16] = {
            0x11111111, // TCfg
            0x00100100, // ASelt
            0x030b0100, 0x07060c04, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };
        vx_uint32 uniPackGR2_2x8[16] = {
            0x11011011, // TCfg
            0x10010010, // ASelt
            0x0e000d0d, 0x0f0f000e, // ABin
            0x22022022, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000000, 0x00000001, 0x00000001, 0x00000000, 0x00000001, 0x00000001 // Constant
        };
        vx_uint32 uniPackTmp2AndB_2x8[16] = {
            0x11111111, // TCfg
            0x00100100, // ASelt
            0x030e0100, 0x07060f04, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };

        vx_uint32 uniCalculateTmpR1st_4x4[16] = {
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00130012, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x0199012a, 0x00000000, 0x0199012a, 0x00000000, 0x0199012a, 0x00000000, 0x0199012a, 0x00000000 // Constant
        };
        vx_uint32 uniCalculateTmpR2nd_4x4[16] = {
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00250024, 0x00370036, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x0199012a, 0x00000000, 0x0199012a, 0x00000000, 0x0199012a, 0x00000000, 0x0199012a, 0x00000000 // Constant
        };
        vx_uint32 uniCalculateTmpR3rd_4x4[16] = {
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00490048, 0x005b005a, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x0199012a, 0x00000000, 0x0199012a, 0x00000000, 0x0199012a, 0x00000000, 0x0199012a, 0x00000000 // Constant
        };
        vx_uint32 uniCalculateTmpR4th_4x4[16] = {
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x006d006c, 0x007f007e, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x0199012a, 0x00000000, 0x0199012a, 0x00000000, 0x0199012a, 0x00000000, 0x0199012a, 0x00000000 // Constant
        };
        vx_uint32 uniCalculateR1st_4x4[16] = {
            0x0f0f0f0f, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002608, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };

        vx_uint32 uniCalculateTmpG1st_4x4[16] = {
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00130012, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00d0012a, 0x00000000, 0x00d0012a, 0x00000000, 0x00d0012a, 0x00000000, 0x00d0012a, 0x00000000 // Constant
        };
        vx_uint32 uniCalculateTmpG2nd_4x4[16] = {
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00250024, 0x00370036, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00d0012a, 0x00000000, 0x00d0012a, 0x00000000, 0x00d0012a, 0x00000000, 0x00d0012a, 0x00000000 // Constant
        };
        vx_uint32 uniCalculateTmpG3rd_4x4[16] = {
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00490048, 0x005b005a, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00d0012a, 0x00000000, 0x00d0012a, 0x00000000, 0x00d0012a, 0x00000000, 0x00d0012a, 0x00000000 // Constant
        };
        vx_uint32 uniCalculateTmpG4th_4x4[16] = {
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x006d006c, 0x007f007e, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00d0012a, 0x00000000, 0x00d0012a, 0x00000000, 0x00d0012a, 0x00000000, 0x00d0012a, 0x00000000 // Constant
        };

        vx_uint32 uniCalculateTmpGbyU_2x8[16] = {
            0x66666666, // TCfg
            0x44444444, // ASelt
            0x03020100, 0x07060504, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00010064, 0x00010064, 0x00010064, 0x00010064, 0x00010064, 0x00010064, 0x00010064, 0x00010064 // Constant
        };
        vx_uint32 uniCalculateG1st_4x4[16] = {
            0x07070707, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00130012, // ABin
            0x08080808, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002608, // AccumType, ConstantType, and PostShift
            0x00010000, 0x00000000, 0x00010000, 0x00000000, 0x00010000, 0x00000000, 0x00010000, 0x00000000 // Constant
        };
        vx_uint32 uniCalculateG2nd_4x4[16] = {
            0x07130707, // TCfg
            0x04100404, // ASelt
            0x00210020, 0x00330302, // ABin
            0x08200808, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002608, // AccumType, ConstantType, and PostShift
            0x00010000, 0x00000000, 0x00010000, 0x00000000, 0x00000000, 0x00000001, 0x00010000, 0x00000000 // Constant
        };
        vx_uint32 uniCalculateG3rd_4x4[16] = {
            0x07130707, // TCfg
            0x04100404, // ASelt
            0x00410040, 0x00530502, // ABin
            0x08200808, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002608, // AccumType, ConstantType, and PostShift
            0x00010000, 0x00000000, 0x00010000, 0x00000000, 0x00000000, 0x00000001, 0x00010000, 0x00000000 // Constant
        };
        vx_uint32 uniCalculateG4th_4x4[16] = {
            0x07070707, // TCfg
            0x04040404, // ASelt
            0x00610060, 0x00730072, // ABin
            0x08080808, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002608, // AccumType, ConstantType, and PostShift
            0x00010000, 0x00000000, 0x00010000, 0x00000000, 0x00010000, 0x00000000, 0x00010000, 0x00000000 // Constant
        };

        vx_uint32 uniCalculateTmpB1st_4x4[16] = {
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00130012, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x0204012a, 0x00000000, 0x0204012a, 0x00000000, 0x0204012a, 0x00000000, 0x0204012a, 0x00000000 // Constant
        };
        vx_uint32 uniCalculateTmpB2nd_4x4[16] = {
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00250024, 0x00370036, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x0204012a, 0x00000000, 0x0204012a, 0x00000000, 0x0204012a, 0x00000000, 0x0204012a, 0x00000000 // Constant
        };
        vx_uint32 uniCalculateTmpB3rd_4x4[16] = {
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00490048, 0x005b005a, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x0204012a, 0x00000000, 0x0204012a, 0x00000000, 0x0204012a, 0x00000000, 0x0204012a, 0x00000000 // Constant
        };
        vx_uint32 uniCalculateTmpB4th_4x4[16] = {
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x006d006c, 0x007f007e, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x0204012a, 0x00000000, 0x0204012a, 0x00000000, 0x0204012a, 0x00000000, 0x0204012a, 0x00000000 // Constant
        };

        vx_uint32 uniQuantU8toU8LoB_2x8[16] = {
            0x99999999, // TCfg
            0x44444444, // ASelt
            0x03020100, 0x07060504, // ABin
            0x99999999, // BSelt
            0x06060606, 0x06060606, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000 // Constant
        };
        vx_uint32 uniQuantU8toU8HiB_2x8[16] = {
            0x99999999, // TCfg
            0x44444444, // ASelt
            0x0b0a0908, 0x0f0e0d0c, // ABin
            0x99999999, // BSelt
            0x06060606, 0x06060606, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000 // Constant
        };
        vx_uint32 uniQuantU8toU8LoG_2x8[16] = {
            0x99999999, // TCfg
            0x44444444, // ASelt
            0x23222120, 0x27262524, // ABin
            0x99999999, // BSelt
            0x06060606, 0x06060606, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000 // Constant
        };
        vx_uint32 uniQuantU8toU8HiG_2x8[16] = {
            0x99999999, // TCfg
            0x44444444, // ASelt
            0x2b2a2928, 0x2f2e2d2c, // ABin
            0x99999999, // BSelt
            0x06060606, 0x06060606, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000 // Constant
        };
        vx_uint32 uniQuantU8toU8LoR_2x8[16] = {
            0x99999999, // TCfg
            0x44444444, // ASelt
            0x43424140, 0x47464544, // ABin
            0x99999999, // BSelt
            0x06060606, 0x06060606, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000 // Constant
        };
        vx_uint32 uniQuantU8toU8HiR_2x8[16] = {
            0x99999999, // TCfg
            0x44444444, // ASelt
            0x4b4a4948, 0x4f4e4d4c, // ABin
            0x99999999, // BSelt
            0x06060606, 0x06060606, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000, 0x3c000000 // Constant
        };

        // R
        status |= vxSetNodeUniform(nodObj, "uniCalculateTmpR1st_4x4", 1, uniCalculateTmpR1st_4x4);
        status |= vxSetNodeUniform(nodObj, "uniCalculateTmpR2nd_4x4", 1, uniCalculateTmpR2nd_4x4);
        status |= vxSetNodeUniform(nodObj, "uniCalculateTmpR3rd_4x4", 1, uniCalculateTmpR3rd_4x4);
        status |= vxSetNodeUniform(nodObj, "uniCalculateTmpR4th_4x4", 1, uniCalculateTmpR4th_4x4);
        status |= vxSetNodeUniform(nodObj, "uniCalculateR1st_4x4", 1, uniCalculateR1st_4x4);

        //G
        status |= vxSetNodeUniform(nodObj, "uniCalculateTmpG1st_4x4", 1, uniCalculateTmpG1st_4x4);
        status |= vxSetNodeUniform(nodObj, "uniCalculateTmpG2nd_4x4", 1, uniCalculateTmpG2nd_4x4);
        status |= vxSetNodeUniform(nodObj, "uniCalculateTmpG3rd_4x4", 1, uniCalculateTmpG3rd_4x4);
        status |= vxSetNodeUniform(nodObj, "uniCalculateTmpG4th_4x4", 1, uniCalculateTmpG4th_4x4);
        status |= vxSetNodeUniform(nodObj, "uniCalculateTmpGbyU_2x8", 1, uniCalculateTmpGbyU_2x8);
        status |= vxSetNodeUniform(nodObj, "uniCalculateG1st_4x4", 1, uniCalculateG1st_4x4);
        status |= vxSetNodeUniform(nodObj, "uniCalculateG2nd_4x4", 1, uniCalculateG2nd_4x4);
        status |= vxSetNodeUniform(nodObj, "uniCalculateG3rd_4x4", 1, uniCalculateG3rd_4x4);
        status |= vxSetNodeUniform(nodObj, "uniCalculateG4th_4x4", 1, uniCalculateG4th_4x4);

        //B
        status |= vxSetNodeUniform(nodObj, "uniCalculateTmpB1st_4x4", 1, uniCalculateTmpB1st_4x4);
        status |= vxSetNodeUniform(nodObj, "uniCalculateTmpB2nd_4x4", 1, uniCalculateTmpB2nd_4x4);
        status |= vxSetNodeUniform(nodObj, "uniCalculateTmpB3rd_4x4", 1, uniCalculateTmpB3rd_4x4);
        status |= vxSetNodeUniform(nodObj, "uniCalculateTmpB4th_4x4", 1, uniCalculateTmpB4th_4x4);
        status |= vxSetNodeUniform(nodObj, "uniCalculateB1st_4x4", 1, uniCalculateR1st_4x4);

        status |= vxSetNodeUniform(nodObj, "uniPackBG0_2x8", 1, uniPackBG0_2x8);
        status |= vxSetNodeUniform(nodObj, "uniPackTmpAndR_2x8", 1, uniPackTmpAndR_2x8);
        status |= vxSetNodeUniform(nodObj, "uniPackRB0_2x8", 1, uniPackRB0_2x8);
        status |= vxSetNodeUniform(nodObj, "uniPackTmp0AndG_2x8", 1, uniPackTmp0AndG_2x8);
        status |= vxSetNodeUniform(nodObj, "uniPackGR1_2x8", 1, uniPackGR1_2x8);
        status |= vxSetNodeUniform(nodObj, "uniPackTmp1AndB_2x8", 1, uniPackTmp1AndB_2x8);
        status |= vxSetNodeUniform(nodObj, "uniPackBG1_2x8", 1, uniPackBG1_2x8);
        status |= vxSetNodeUniform(nodObj, "uniPackTmp1AndR_2x8", 1, uniPackTmp1AndR_2x8);
        status |= vxSetNodeUniform(nodObj, "uniPackRB2_2x8", 1, uniPackRB2_2x8);
        status |= vxSetNodeUniform(nodObj, "uniPackTmp2AndG_2x8", 1, uniPackTmp2AndG_2x8);
        status |= vxSetNodeUniform(nodObj, "uniPackGR2_2x8", 1, uniPackGR2_2x8);
        status |= vxSetNodeUniform(nodObj, "uniPackTmp2AndB_2x8", 1, uniPackTmp2AndB_2x8);

        status |= vxSetNodeUniform(nodObj, "uniQuantU8toU8LoB_2x8", 1, uniQuantU8toU8LoB_2x8);
        status |= vxSetNodeUniform(nodObj, "uniQuantU8toU8HiB_2x8", 1, uniQuantU8toU8HiB_2x8);
        status |= vxSetNodeUniform(nodObj, "uniQuantU8toU8LoG_2x8", 1, uniQuantU8toU8LoG_2x8);
        status |= vxSetNodeUniform(nodObj, "uniQuantU8toU8HiG_2x8", 1, uniQuantU8toU8HiG_2x8);
        status |= vxSetNodeUniform(nodObj, "uniQuantU8toU8LoR_2x8", 1, uniQuantU8toU8LoR_2x8);
        status |= vxSetNodeUniform(nodObj, "uniQuantU8toU8HiR_2x8", 1, uniQuantU8toU8HiR_2x8);

        status |= vxSetNodeUniform(nodObj, "zp", 1, &output_ZP);
        status |= vxSetNodeUniform(nodObj, "outputScale", 1, &outputScale);
    }

    status |= vxSetNodeUniform(nodObj, "rOrder", 1, &reorder);
    status |= vxSetNodeUniform(nodObj, "bOrder", 1, &order1);

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
                                    &shaderParam, sizeof(vx_kernel_execution_parameters_t));

    if(status < 0)
        printf("error-%s,%d\n",__FILE__,__LINE__);

    return status;
}

vx_status VX_CALLBACK vxYuvScaleRbgInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    vx_uint32 paraNum
    )
{
    vx_status status = VX_SUCCESS;
    /*TODO: Add initial code for VX program*/
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
    vx_kernel_execution_parameters_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vx_tensor  output           = (vx_tensor)paramObj[3];
    vx_scalar reorder_s         = (vx_scalar)paramObj[12];
    vx_enum  outFormat;
    vx_uint32 width             = 0;
    vx_uint32 height            = 0;
    vx_float32 outputScale      = 1.0;
    vx_int32   output_ZP        = 0;
    vx_int32 reorder = 0;
    vx_int32 order1 = 2;

    vx_uint32  dst_size[4]   = {0, 0, 0, 0};

    status |= vxQueryTensor(output, VX_TENSOR_DATA_TYPE, &outFormat, sizeof(outFormat));
    status |= vxQueryTensor(output, VX_TENSOR_DIMS, dst_size, sizeof(dst_size));
    status |= vxQueryTensor(output, VX_TENSOR_SCALE, &outputScale, sizeof(outputScale));
    status |= vxQueryTensor(output, VX_TENSOR_ZERO_POINT, &output_ZP, sizeof(output_ZP));
    status |= vxCopyScalar(reorder_s, (void*)&reorder, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    if(reorder != 0)
    {
        reorder = 2;
        order1 = 0;
    }

    if(outFormat == VX_TYPE_UINT8)
    {
        outputScale = 1.0f / outputScale;
    }

    width = dst_size[0];
    height = dst_size[1];

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkOffset[2] = 0;
    //shaderParam.globalWorkScale[0]  = 1;
    shaderParam.globalWorkScale[0]  = 4; // opt
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.globalWorkScale[2]  = 1;
    shaderParam.localWorkSize[0]    = 8;
    shaderParam.localWorkSize[1]    = 1;
    shaderParam.localWorkSize[2]    = 1;
    shaderParam.globalWorkSize[0]   = gcmALIGN((width + shaderParam.globalWorkScale[0] - 1) /
                                            shaderParam.globalWorkScale[0], shaderParam.localWorkSize[0]);
    shaderParam.globalWorkSize[1]   = gcmALIGN((height + shaderParam.globalWorkScale[1] - 1) /
                                            shaderParam.globalWorkScale[1], shaderParam.localWorkSize[1]);
    shaderParam.globalWorkSize[2]   = 1;

    if(outFormat == VX_TYPE_UINT8)
    {
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

        vx_uint32 uniCalculateTmpRWise_4x4[16] = {
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00110000, 0x00330022, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x0199012a, 0x00000000, 0x0199012a, 0x00000000, 0x0199012a, 0x00000000, 0x0199012a, 0x00000000 // Constant
        };
        vx_uint32 uniCalculateTmpRWise2nd_4x4[16] = {
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00550044, 0x00770066, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x0199012a, 0x00000000, 0x0199012a, 0x00000000, 0x0199012a, 0x00000000, 0x0199012a, 0x00000000 // Constant
        };
        vx_uint32 uniCalculateTmpRWise3rd_4x4[16] = {
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00990088, 0x00bb00aa, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x0199012a, 0x00000000, 0x0199012a, 0x00000000, 0x0199012a, 0x00000000, 0x0199012a, 0x00000000 // Constant
        };
        vx_uint32 uniCalculateTmpRWise4th_4x4[16] = {
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00dd00cc, 0x00ff00ee, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x0199012a, 0x00000000, 0x0199012a, 0x00000000, 0x0199012a, 0x00000000, 0x0199012a, 0x00000000 // Constant
        };
        vx_uint32 uniCalculateR1st_4x4[16] = {
            0x0f0f0f0f, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002608, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };

        vx_uint32 uniCalculateTmpGWise_4x4[16] = {
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00110000, 0x00330022, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00d0012a, 0x00000000, 0x00d0012a, 0x00000000, 0x00d0012a, 0x00000000, 0x00d0012a, 0x00000000 // Constant
        };
        vx_uint32 uniCalculateTmpGWise2nd_4x4[16] = {
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00550044, 0x00770066, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00d0012a, 0x00000000, 0x00d0012a, 0x00000000, 0x00d0012a, 0x00000000, 0x00d0012a, 0x00000000 // Constant
        };
        vx_uint32 uniCalculateTmpGWise3rd_4x4[16] = {
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00990088, 0x00bb00aa, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00d0012a, 0x00000000, 0x00d0012a, 0x00000000, 0x00d0012a, 0x00000000, 0x00d0012a, 0x00000000 // Constant
        };
        vx_uint32 uniCalculateTmpGWise4th_4x4[16] = {
            0x09090909, // TCfg
            0x04040404, // ASelt
            0x00dd00cc, 0x00ff00ee, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00d0012a, 0x00000000, 0x00d0012a, 0x00000000, 0x00d0012a, 0x00000000, 0x00d0012a, 0x00000000 // Constant
        };

        vx_uint32 uniCalculateTmpGbyU_2x8[16] = {
            0x66666666, // TCfg
            0x44444444, // ASelt
            0x03020100, 0x07060504, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00010064, 0x00010064, 0x00010064, 0x00010064, 0x00010064, 0x00010064, 0x00010064, 0x00010064 // Constant
        };
        vx_uint32 uniCalculateTmpGbyU2nd_2x8[16] = {
            0x66666666, // TCfg
            0x44444444, // ASelt
            0x0b0a0908, 0x0f0e0d0c, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00010064, 0x00010064, 0x00010064, 0x00010064, 0x00010064, 0x00010064, 0x00010064, 0x00010064 // Constant
        };
        vx_uint32 uniCalculateGWise_4x4[16] = {
            0x07070707, // TCfg
            0x04040404, // ASelt
            0x00110000, 0x00330022, // ABin
            0x08080808, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002608, // AccumType, ConstantType, and PostShift
            0x00010000, 0x00000000, 0x00010000, 0x00000000, 0x00010000, 0x00000000, 0x00010000, 0x00000000 // Constant
        };
        vx_uint32 uniCalculateGWise2nd_4x4[16] = {
            0x07070707, // TCfg
            0x04040404, // ASelt
            0x00510040, 0x00730062, // ABin
            0x08080808, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002608, // AccumType, ConstantType, and PostShift
            0x00010000, 0x00000000, 0x00010000, 0x00000000, 0x00010000, 0x00000000, 0x00010000, 0x00000000 // Constant
        };

        vx_uint32 uniCalculateTmpBWise_4x4[16] = {
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00110000, 0x00330022, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x0204012a, 0x00000000, 0x0204012a, 0x00000000, 0x0204012a, 0x00000000, 0x0204012a, 0x00000000 // Constant
        };
        vx_uint32 uniCalculateTmpBWise2nd_4x4[16] = {
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00550044, 0x00770066, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x0204012a, 0x00000000, 0x0204012a, 0x00000000, 0x0204012a, 0x00000000, 0x0204012a, 0x00000000 // Constant
        };
        vx_uint32 uniCalculateTmpBWise3rd_4x4[16] = {
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00990088, 0x00bb00aa, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x0204012a, 0x00000000, 0x0204012a, 0x00000000, 0x0204012a, 0x00000000, 0x0204012a, 0x00000000 // Constant
        };
        vx_uint32 uniCalculateTmpBWise4th_4x4[16] = {
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00dd00cc, 0x00ff00ee, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x0204012a, 0x00000000, 0x0204012a, 0x00000000, 0x0204012a, 0x00000000, 0x0204012a, 0x00000000 // Constant
        };

        vx_uint32 uniDescaleU8_4x4[16] = {
            0x0f0f0f0f, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002614, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };

        vx_uint32 uniBilinearTmp1st_4x4[16] = {
            0x09090909, // TCfg
            0x00000000, // ASelt
            0x00450001, 0x00cd0089, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00000000, 0x00010001, 0x00000000, 0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
        };
        vx_uint32 uniBilinearTmp2nd_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00040000, 0x000c0008, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000 // Constant
        };
        vx_uint32 uniBilinearTmp3rd_4x4[16] = {
            0x69696969, // TCfg
            0x00000000, // ASelt
            0x45670123, 0xcdef89ab, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
        };
        vx_uint32 uniBilinearTmp4th_4x4[16] = {
            0x09090909, // TCfg
            0x00000000, // ASelt
            0x00460002, 0x00ce008a, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x04000400, 0x00000000, 0x04000400, 0x00000000, 0x04000400, 0x00000000, 0x04000400, 0x00000000 // Constant
        };

        status |= vxSetNodeUniform(nodObj, "uniCalculateR1st_4x4", 1, uniCalculateR1st_4x4);
        status |= vxSetNodeUniform(nodObj, "uniCalculateTmpGbyU_2x8", 1, uniCalculateTmpGbyU_2x8);
        status |= vxSetNodeUniform(nodObj, "uniCalculateTmpGbyU2nd_2x8", 1, uniCalculateTmpGbyU2nd_2x8);

        status |= vxSetNodeUniform(nodObj, "uniCalculateB1st_4x4", 1, uniCalculateR1st_4x4);

        status |= vxSetNodeUniform(nodObj, "uniDescaleU8_4x4", 1, uniDescaleU8_4x4);

        status |= vxSetNodeUniform(nodObj, "uniCalculateTmpRWise_4x4", 1, uniCalculateTmpRWise_4x4);
        status |= vxSetNodeUniform(nodObj, "uniCalculateTmpRWise2nd_4x4", 1, uniCalculateTmpRWise2nd_4x4);
        status |= vxSetNodeUniform(nodObj, "uniCalculateTmpRWise3rd_4x4", 1, uniCalculateTmpRWise3rd_4x4);
        status |= vxSetNodeUniform(nodObj, "uniCalculateTmpRWise4th_4x4", 1, uniCalculateTmpRWise4th_4x4);
        status |= vxSetNodeUniform(nodObj, "uniCalculateTmpGWise_4x4", 1, uniCalculateTmpGWise_4x4);
        status |= vxSetNodeUniform(nodObj, "uniCalculateTmpGWise2nd_4x4", 1, uniCalculateTmpGWise2nd_4x4);
        status |= vxSetNodeUniform(nodObj, "uniCalculateTmpGWise3rd_4x4", 1, uniCalculateTmpGWise3rd_4x4);
        status |= vxSetNodeUniform(nodObj, "uniCalculateTmpGWise4th_4x4", 1, uniCalculateTmpGWise4th_4x4);
        status |= vxSetNodeUniform(nodObj, "uniCalculateGWise_4x4", 1, uniCalculateGWise_4x4);
        status |= vxSetNodeUniform(nodObj, "uniCalculateGWise2nd_4x4", 1, uniCalculateGWise2nd_4x4);
        status |= vxSetNodeUniform(nodObj, "uniCalculateTmpBWise_4x4", 1, uniCalculateTmpBWise_4x4);
        status |= vxSetNodeUniform(nodObj, "uniCalculateTmpBWise2nd_4x4", 1, uniCalculateTmpBWise2nd_4x4);
        status |= vxSetNodeUniform(nodObj, "uniCalculateTmpBWise3rd_4x4", 1, uniCalculateTmpBWise3rd_4x4);
        status |= vxSetNodeUniform(nodObj, "uniCalculateTmpBWise4th_4x4", 1, uniCalculateTmpBWise4th_4x4);

        status |= vxSetNodeUniform(nodObj, "uniBilinearTmp1st_4x4", 1, uniBilinearTmp1st_4x4);
        status |= vxSetNodeUniform(nodObj, "uniBilinearTmp2nd_4x4", 1, uniBilinearTmp2nd_4x4);
        status |= vxSetNodeUniform(nodObj, "uniBilinearTmp3rd_4x4", 1, uniBilinearTmp3rd_4x4);
        status |= vxSetNodeUniform(nodObj, "uniBilinearTmp4th_4x4", 1, uniBilinearTmp4th_4x4);

        status |= vxSetNodeUniform(nodObj, "zp", 1, &output_ZP);
        status |= vxSetNodeUniform(nodObj, "outputScale", 1, &outputScale);

        status |= vxSetNodeUniform(nodObj, "uniConvertInt32toUint8_2x8", 1, uniConvertInt32toUint8_2x8);
        status |= vxSetNodeUniform(nodObj, "rOrder", 1, &reorder);
        status |= vxSetNodeUniform(nodObj, "bOrder", 1, &order1);

        if(status < 0)
            printf("error-%s,%d\n",__FILE__,__LINE__);

        status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
                                        &shaderParam, sizeof(vx_kernel_execution_parameters_t));
    }

    return status;
}


#ifdef __cplusplus
extern "C" {
#endif
vx_kernel_description_t vxYuv2rbg_CPU =
{
    VX_KERNEL_ENUM_PRE_PROCESS_YUV420,
    _VX_KERNEL_NAME,
    vxYuv2rbgKernel,
    vxYuv2rbgKernelParam,
    _cnt_of_array( vxYuv2rbgKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxYuv2rbg_u8 =
{
    VX_KERNEL_ENUM_PRE_PROCESS_YUV420,
    VX_KERNEL_NAME_PRE_PROCESS_YUV2RBG_U8,
    NULL,
    vxYuv2rbgKernelParam,
    _cnt_of_array( vxYuv2rbgKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxYuv2rbgInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxYuv2rbg_trans_u8 =
{
    VX_KERNEL_ENUM_PRE_PROCESS_YUV420,
    VX_KERNEL_NAME_PRE_PROCESS_YUV2RBG_TRANS_U8,
    NULL,
    vxYuv2rbgKernelParam,
    _cnt_of_array( vxYuv2rbgKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxYuv2rbgInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxYuv2rbg_resize_norm_u8 =
{
    VX_KERNEL_ENUM_PRE_PROCESS_YUV420,
    VX_KERNEL_NAME_PRE_PROCESS_YUV2RBG_RESIZE_NORM_U8,
    NULL,
    vxYuv2rbgKernelParam,
    _cnt_of_array( vxYuv2rbgKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxYuvScaleRbgInitializer,
    vsi_nn_KernelDeinitializer
};


vx_kernel_description_t * vx_kernel_YUV2RBG_list[] =
{
    &vxYuv2rbg_CPU,
    &vxYuv2rbg_u8,
    &vxYuv2rbg_trans_u8,
    &vxYuv2rbg_resize_norm_u8,
    NULL
};
#ifdef __cplusplus
}
#endif
