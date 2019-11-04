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

#define _VX_KERNEL_NAME         (VX_KERNEL_NAME_PRE_PROCESS_BGRA)
#define _VX_KERNEL_FUNC_KERNEL  (vxPre_process_bgraKernel)

#define clipMIN(x, y)            (((x) <= (y)) ?  (x) :  (y))
#define clipMAX(x, y)            (((x) >= (y)) ?  (x) :  (y))
#define CLAMP(x)                 (clipMIN(clipMAX((x), (0)), (255)))

#define DESCALE(x) (((x) + (1<<19)) >> 20)

static vsi_status VX_CALLBACK vxPre_process_bgraKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
#define ARG_NUM            (11)
#define TENSOR_NUM_INPUT (1)
#define TENSOR_NUM_OUTPUT (1)
#define TENSOR_NUM (TENSOR_NUM_INPUT+TENSOR_NUM_OUTPUT)

    vsi_status status = VSI_FAILURE;
    vx_context context = NULL;
    vx_tensor input[TENSOR_NUM_INPUT] = {0};
    vx_tensor output[TENSOR_NUM_OUTPUT] = {0};
    uint8_t *u8_in_buffer[TENSOR_NUM_INPUT] = {0};
    uint8_t *u8_out_buffer[TENSOR_NUM_OUTPUT] = {0};
    vsi_nn_tensor_attr_t in_attr[TENSOR_NUM_INPUT];
    vsi_nn_tensor_attr_t out_attr[TENSOR_NUM_OUTPUT];
    uint32_t out_elements[TENSOR_NUM_OUTPUT]= {0};
    vx_uint32  dst_size[4]   = {0, 0, 0, 0};
    vx_uint32  src_size[4]   = {0, 0, 0, 0};
    vx_float32 outputScale      = 1.0;
    vx_int32   output_ZP        = 0;
    vsi_nn_type_e outputFormat = VSI_NN_TYPE_FLOAT16;

    int32_t xRatio = 0, yRatio = 0, xOffset = 0, yOffset = 0;
    float rMean = 0, gMean = 0, bMean = 0, var = 0;
    int32_t order = 0, trans = 0;

    int32_t i = 0;
    for(i = 0; i < TENSOR_NUM_INPUT; i++)
    {
        memset(&in_attr[i], 0x0, sizeof(vsi_nn_tensor_attr_t));
    }
    for(i = 0; i < TENSOR_NUM_OUTPUT; i++)
    {
        memset(&out_attr[i], 0x0, sizeof(vsi_nn_tensor_attr_t));
    }
    /* prepare data */
    context = vxGetContext((vx_reference)node);

    for(i = 0; i < TENSOR_NUM_INPUT; i ++)
    {
        input[i] = (vx_tensor)paramObj[i];
        status = vsi_nn_vxGetTensorAttr(input[i], &in_attr[i]);
        TEST_CHECK_STATUS(status, final);
        u8_in_buffer[i] = vsi_nn_vxCopyTensorToData(context, input[i], &in_attr[i]);
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

        status |= vxQueryTensor(output[i], VX_TENSOR_DIMS, dst_size, sizeof(dst_size));
        status |= vxQueryTensor(output[i], VX_TENSOR_SCALE, &outputScale, sizeof(outputScale));
        status |= vxQueryTensor(output[i], VX_TENSOR_ZERO_POINT, &output_ZP, sizeof(output_ZP));
        status |= vxQueryTensor(output[i], VX_TENSOR_DATA_TYPE, &outputFormat, sizeof(outputFormat));
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
    status |= vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 9], &(trans),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    /* TODO: Add CPU kernel implement */
    /* example code : copy data form input tensor to output tensor*/
    {
        vx_uint8 rline1[2], rline2[2];
        vx_uint8 gline1[2], gline2[2];
        vx_uint8 bline1[2], bline2[2];
        vx_int32 dx = 0, dy = 0, dz = 0;
        vx_int32 src_width = src_size[0];
        vx_int32 src_height = src_size[1];
        vx_int32 dst_width = dst_size[0];
        vx_int32 dst_height = dst_size[1];
        vx_int32 stride = dst_width * dst_height;
        vx_int32 rOffset = 0;
        vx_int32 gOffset = 1 * stride;
        vx_int32 bOffset = 2 * stride;
        vx_uint8 R, G, B;
        vx_uint8 *bSrc = u8_in_buffer[0];
        vx_uint8 *gSrc = u8_in_buffer[0] + 1;
        vx_uint8 *rSrc = u8_in_buffer[0] + 2;
        vx_uint8 *dst = u8_out_buffer[0];
        vx_int32 elementSize = 4;

        if(order)
        {
            rOffset = 2 * stride;
            bOffset = 0;
        }

        if(trans)
        {
            vx_int32 dstElementSize = 3;

            gOffset = 1;
            rOffset = 2;

            if(order)
            {
                rOffset = 2;
                bOffset = 0;
            }

            dst_width = dst_size[0] / 3;
            for ( dz = 0; dz < 1; dz ++)
            {
                for ( dy = 0; dy < (vx_int32)dst_size[1]; dy ++)
                {
                    for ( dx = 0; dx < dst_width; dx ++)
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
                        source_index = (sx * 4 + sy * src_width + dz * src_width * src_height + 0);
                        output_index = dx * dstElementSize + dy * dst_size[0];

                        dstR_idx = output_index + rOffset;
                        dstG_idx = output_index + gOffset;
                        dstB_idx = output_index + bOffset;

                        if(dx <= dst_width - 1 && dy <= dst_height - 1 && (xRatio != (1 << 15) || yRatio != (1 << 15)))
                        {
                            //vx_int32 offset = xOffset + yOffset * src_width;
                            //vx_uint8 result;
                            vx_int32 temp1;
                            vx_int32 temp2;

                            bline1[0] = bSrc[source_index ];
                            bline1[1] = bSrc[source_index + 1 * elementSize];
                            bline2[0] = bSrc[source_index + src_width];
                            bline2[1] = bSrc[source_index + src_width + 1 * elementSize];

                            gline1[0] = gSrc[source_index ];
                            gline1[1] = gSrc[source_index + 1 * elementSize];
                            gline2[0] = gSrc[source_index + src_width];
                            gline2[1] = gSrc[source_index + src_width + 1 * elementSize];

                            rline1[0] = rSrc[source_index ];
                            rline1[1] = rSrc[source_index + 1 * elementSize];
                            rline2[0] = rSrc[source_index + src_width];
                            rline2[1] = rSrc[source_index + src_width + 1 * elementSize];

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
                            B = bSrc[source_index ];
                            G = gSrc[source_index ];
                            R = rSrc[source_index ];

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
        else
        {
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
                        source_index = (sx * 4 + sy * src_width + dz * src_width * src_height + 0);
                        output_index = dx + dy * dst_width;

                        dstR_idx = output_index + rOffset;
                        dstG_idx = output_index + gOffset;
                        dstB_idx = output_index + bOffset;

                        if(dx <= dst_width - 1 && dy <= dst_height - 1 && (xRatio != (1 << 15) || yRatio != (1 << 15)))
                        {
                            //vx_int32 offset = xOffset + yOffset * src_width;
                            //vx_uint8 result;
                            vx_int32 temp1;
                            vx_int32 temp2;

                            bline1[0] = bSrc[source_index ];
                            bline1[1] = bSrc[source_index + 1 * elementSize];
                            bline2[0] = bSrc[source_index + src_width];
                            bline2[1] = bSrc[source_index + src_width + 1 * elementSize];

                            gline1[0] = gSrc[source_index ];
                            gline1[1] = gSrc[source_index + 1 * elementSize];
                            gline2[0] = gSrc[source_index + src_width];
                            gline2[1] = gSrc[source_index + src_width + 1 * elementSize];

                            rline1[0] = rSrc[source_index ];
                            rline1[1] = rSrc[source_index + 1 * elementSize];
                            rline2[0] = rSrc[source_index + src_width];
                            rline2[1] = rSrc[source_index + src_width + 1 * elementSize];

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
                            B = bSrc[source_index ];
                            G = gSrc[source_index ];
                            R = rSrc[source_index ];

                            final = (B - bMean) * var;
                            dst[dstB_idx] = vsi_nn_Fp32ToAffine(final, outputScale, output_ZP, outputFormat);
                            final = (G - gMean) * var;
                            dst[dstG_idx] = vsi_nn_Fp32ToAffine(final, outputScale, output_ZP, outputFormat);
                            final = (R - rMean) * var;
                            dst[dstR_idx] = vsi_nn_Fp32ToAffine(final, outputScale, output_ZP, outputFormat);
                        }
                    }
                }
            }
        }
    }

    /* save data */
    status = vsi_nn_vxCopyDataToTensor(context, output[0], &out_attr[0], u8_out_buffer[0]);
    TEST_CHECK_STATUS(status, final);

final:
    for (i = 0; i < TENSOR_NUM_INPUT; i++)
    {
        if (u8_in_buffer[i]) free(u8_in_buffer[i]);
    }
    for(i = 0; i < TENSOR_NUM_OUTPUT; i++)
    {
        if (u8_out_buffer[i]) free(u8_out_buffer[i]);
    }
    return status;
} /* _VX_KERNEL_FUNC_KERNEL() */

static vx_param_description_t vxPre_process_bgraKernelParam[] =
{
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

vx_status VX_CALLBACK vxPre_process_bgra_copyInitializer
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
    vx_tensor  output           = (vx_tensor)paramObj[1];
    vx_scalar reorder_s         = (vx_scalar)paramObj[10];
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
    shaderParam.globalWorkScale[0]  = 4; // opt
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.globalWorkScale[2]  = 1;
    shaderParam.localWorkSize[0]    = 4;
    shaderParam.localWorkSize[1]    = 1;
    shaderParam.localWorkSize[2]    = 1;
    shaderParam.globalWorkSize[0]   = gcmALIGN((width + shaderParam.globalWorkScale[0] - 1) /
                                            shaderParam.globalWorkScale[0], shaderParam.localWorkSize[0]);
    shaderParam.globalWorkSize[1]   = gcmALIGN((height + shaderParam.globalWorkScale[1] - 1) /
                                            shaderParam.globalWorkScale[1], shaderParam.localWorkSize[1]);
    shaderParam.globalWorkSize[2]   = 1;

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
        vx_uint32 uniExtractBfromBgra_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00040000, 0x000c0008, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        };
        vx_uint32 uniExtractGfromBgra_4x4[16] = {
            0x01010401, // TCfg
            0x00000000, // ASelt
            0x00500001, 0x000d0009, // ABin
            0x02020802, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00010000, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        };
        vx_uint32 uniExtractRfromBgra_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00060002, 0x000e000a, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        };

        status |= vxSetNodeUniform(nodObj, "uniConvertInt32toUint8_2x8", 1, uniConvertInt32toUint8_2x8);
        status |= vxSetNodeUniform(nodObj, "uniExtractBfromBgra_4x4", 1, uniExtractBfromBgra_4x4);
        status |= vxSetNodeUniform(nodObj, "uniExtractGfromBgra_4x4", 1, uniExtractGfromBgra_4x4);
        status |= vxSetNodeUniform(nodObj, "uniExtractRfromBgra_4x4", 1, uniExtractRfromBgra_4x4);

        status |= vxSetNodeUniform(nodObj, "zp", 1, &output_ZP);
        status |= vxSetNodeUniform(nodObj, "outputScale", 1, &outputScale);

        status |= vxSetNodeUniform(nodObj, "rOrder", 1, &reorder);
        status |= vxSetNodeUniform(nodObj, "bOrder", 1, &order1);

        if(status < 0)
            printf("error-%s,%d\n",__FILE__,__LINE__);

        status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
                                        &shaderParam, sizeof(vx_kernel_execution_parameters_t));
    }

    return status;
}

vx_status VX_CALLBACK vxPre_process_bgraInitializer
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

    vx_tensor  output           = (vx_tensor)paramObj[1];
    vx_scalar reorder_s         = (vx_scalar)paramObj[10];
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
    shaderParam.globalWorkScale[0]  = 4; // opt
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.globalWorkScale[2]  = 1;
    shaderParam.localWorkSize[0]    = 4;
    shaderParam.localWorkSize[1]    = 1;
    shaderParam.localWorkSize[2]    = 1;
    shaderParam.globalWorkSize[0]   = gcmALIGN((width + shaderParam.globalWorkScale[0] - 1) /
                                            shaderParam.globalWorkScale[0], shaderParam.localWorkSize[0]);
    shaderParam.globalWorkSize[1]   = gcmALIGN((height + shaderParam.globalWorkScale[1] - 1) /
                                            shaderParam.globalWorkScale[1], shaderParam.localWorkSize[1]);
    shaderParam.globalWorkSize[2]   = 1;
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

        vx_uint32 uniBilinearTmp1BgraShort_4x4[16] = {
            0x19191919, // TCfg
            0x00000000, // ASelt
            0x01150004, 0x03370226, // ABin
            0x25252525, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400 // Constant
        };
        vx_uint32 uniBilinearTmp2BgraShort_4x4[16] = {
            0x19191919, // TCfg
            0x00000000, // ASelt
            0x099d088c, 0x0bbf0aae, // ABin
            0x25252525, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400 // Constant
        };
        vx_uint32 uniBilinearTmp3BgraShort_4x4[16] = {
            0x19191919, // TCfg
            0x00000000, // ASelt
            0x01150004, 0x03370226, // ABin
            0x25252525, // BSelt
            0x00110011, 0x00110011, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400 // Constant
        };
        vx_uint32 uniBilinearTmp4BgraShort_4x4[16] = {
            0x19191919, // TCfg
            0x00000000, // ASelt
            0x099d088c, 0x0bbf0aae, // ABin
            0x25252525, // BSelt
            0x00110011, 0x00110011, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400 // Constant
        };
        vx_uint32 uniBilinearTmp5BgraShort_4x4[16] = {
            0x19191919, // TCfg
            0x00000000, // ASelt
            0x01150004, 0x03370226, // ABin
            0x25252525, // BSelt
            0x00220022, 0x00220022, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400 // Constant
        };
        vx_uint32 uniBilinearTmp6BgraShort_4x4[16] = {
            0x19191919, // TCfg
            0x00000000, // ASelt
            0x099d088c, 0x0bbf0aae, // ABin
            0x25252525, // BSelt
            0x00220022, 0x00220022, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400 // Constant
        };
        vx_uint32 uniBilinearTmp7BgraShort_4x4[16] = {
            0x19191919, // TCfg
            0x00000000, // ASelt
            0x01150004, 0x03370226, // ABin
            0x25252525, // BSelt
            0x00330033, 0x00330033, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400 // Constant
        };
        vx_uint32 uniBilinearTmp8BgraShort_4x4[16] = {
            0x19191919, // TCfg
            0x00000000, // ASelt
            0x099d088c, 0x0bbf0aae, // ABin
            0x25252525, // BSelt
            0x00330033, 0x00330033, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400 // Constant
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

        vx_uint32 uniExtractInt32BgraToU8_2x8[16] = {
            0x33333333, // TCfg
            0x10101010, // ASelt
            0x01010000, 0x03030202, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };

        vx_uint32 uniExchangeBgra_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x09080100, 0x0b0a0302, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };
        vx_uint32 uniExchangeBgra2_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x0d0c0504, 0x0f0e0706, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };

        status |= vxSetNodeUniform(nodObj, "uniConvertInt32toUint8_2x8", 1, uniConvertInt32toUint8_2x8);
        status |= vxSetNodeUniform(nodObj, "uniBilinearTmp1BgraShort_4x4", 1, uniBilinearTmp1BgraShort_4x4);
        status |= vxSetNodeUniform(nodObj, "uniBilinearTmp2BgraShort_4x4", 1, uniBilinearTmp2BgraShort_4x4);
        status |= vxSetNodeUniform(nodObj, "uniBilinearTmp3BgraShort_4x4", 1, uniBilinearTmp3BgraShort_4x4);
        status |= vxSetNodeUniform(nodObj, "uniBilinearTmp4BgraShort_4x4", 1, uniBilinearTmp4BgraShort_4x4);
        status |= vxSetNodeUniform(nodObj, "uniBilinearTmp5BgraShort_4x4", 1, uniBilinearTmp5BgraShort_4x4);
        status |= vxSetNodeUniform(nodObj, "uniBilinearTmp6BgraShort_4x4", 1, uniBilinearTmp6BgraShort_4x4);
        status |= vxSetNodeUniform(nodObj, "uniBilinearTmp7BgraShort_4x4", 1, uniBilinearTmp7BgraShort_4x4);
        status |= vxSetNodeUniform(nodObj, "uniBilinearTmp8BgraShort_4x4", 1, uniBilinearTmp8BgraShort_4x4);

        status |= vxSetNodeUniform(nodObj, "uniDescaleU8_4x4", 1, uniDescaleU8_4x4);
        status |= vxSetNodeUniform(nodObj, "uniExtractInt32BgraToU8_2x8", 1, uniExtractInt32BgraToU8_2x8);
        status |= vxSetNodeUniform(nodObj, "uniExchangeBgra_2x8", 1, uniExchangeBgra_2x8);
        status |= vxSetNodeUniform(nodObj, "uniExchangeBgra2_2x8", 1, uniExchangeBgra2_2x8);

        status |= vxSetNodeUniform(nodObj, "zp", 1, &output_ZP);
        status |= vxSetNodeUniform(nodObj, "outputScale", 1, &outputScale);

        status |= vxSetNodeUniform(nodObj, "rOrder", 1, &reorder);
        status |= vxSetNodeUniform(nodObj, "bOrder", 1, &order1);

        if(status < 0)
            printf("error-%s,%d\n",__FILE__,__LINE__);

        status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
                                        &shaderParam, sizeof(vx_kernel_execution_parameters_t));
    }

    return status;
}

vx_status VX_CALLBACK vxPre_process_bgra_transInitializer
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

    vx_tensor  output           = (vx_tensor)paramObj[1];
    vx_enum  outFormat;
    vx_uint32 width             = 0;
    vx_uint32 height            = 0;
    vx_float32 outputScale      = 1.0;
    vx_int32   output_ZP        = 0;

    vx_uint32  dst_size[4]   = {0, 0, 0, 0};

    status |= vxQueryTensor(output, VX_TENSOR_DATA_TYPE, &outFormat, sizeof(outFormat));
    status |= vxQueryTensor(output, VX_TENSOR_DIMS, dst_size, sizeof(dst_size));
    status |= vxQueryTensor(output, VX_TENSOR_SCALE, &outputScale, sizeof(outputScale));
    status |= vxQueryTensor(output, VX_TENSOR_ZERO_POINT, &output_ZP, sizeof(output_ZP));

    if(outFormat == VX_TYPE_UINT8)
    {
        outputScale = 1.0f / outputScale;
    }

    width = dst_size[0] / 3;
    height = dst_size[1];

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkOffset[2] = 0;
    shaderParam.globalWorkScale[0]  = 4; // opt
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.globalWorkScale[2]  = 1;
    shaderParam.localWorkSize[0]    = 4;
    shaderParam.localWorkSize[1]    = 1;
    shaderParam.localWorkSize[2]    = 1;
    shaderParam.globalWorkSize[0]   = gcmALIGN((width + shaderParam.globalWorkScale[0] - 1) /
                                            shaderParam.globalWorkScale[0], shaderParam.localWorkSize[0]);
    shaderParam.globalWorkSize[1]   = gcmALIGN((height + shaderParam.globalWorkScale[1] - 1) /
                                            shaderParam.globalWorkScale[1], shaderParam.localWorkSize[1]);
    shaderParam.globalWorkSize[2]   = 1;
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

        vx_uint32 uniBilinearTmp1BgraShort_4x4[16] = {
            0x19191919, // TCfg
            0x00000000, // ASelt
            0x01150004, 0x03370226, // ABin
            0x25252525, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400 // Constant
        };
        vx_uint32 uniBilinearTmp2BgraShort_4x4[16] = {
            0x19191919, // TCfg
            0x00000000, // ASelt
            0x099d088c, 0x0bbf0aae, // ABin
            0x25252525, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400 // Constant
        };
        vx_uint32 uniBilinearTmp3BgraShort_4x4[16] = {
            0x19191919, // TCfg
            0x00000000, // ASelt
            0x01150004, 0x03370226, // ABin
            0x25252525, // BSelt
            0x00110011, 0x00110011, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400 // Constant
        };
        vx_uint32 uniBilinearTmp4BgraShort_4x4[16] = {
            0x19191919, // TCfg
            0x00000000, // ASelt
            0x099d088c, 0x0bbf0aae, // ABin
            0x25252525, // BSelt
            0x00110011, 0x00110011, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400 // Constant
        };
        vx_uint32 uniBilinearTmp5BgraShort_4x4[16] = {
            0x19191919, // TCfg
            0x00000000, // ASelt
            0x01150004, 0x03370226, // ABin
            0x25252525, // BSelt
            0x00220022, 0x00220022, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400 // Constant
        };
        vx_uint32 uniBilinearTmp6BgraShort_4x4[16] = {
            0x19191919, // TCfg
            0x00000000, // ASelt
            0x099d088c, 0x0bbf0aae, // ABin
            0x25252525, // BSelt
            0x00220022, 0x00220022, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400 // Constant
        };
        vx_uint32 uniBilinearTmp7BgraShort_4x4[16] = {
            0x19191919, // TCfg
            0x00000000, // ASelt
            0x01150004, 0x03370226, // ABin
            0x25252525, // BSelt
            0x00330033, 0x00330033, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400 // Constant
        };
        vx_uint32 uniBilinearTmp8BgraShort_4x4[16] = {
            0x19191919, // TCfg
            0x00000000, // ASelt
            0x099d088c, 0x0bbf0aae, // ABin
            0x25252525, // BSelt
            0x00330033, 0x00330033, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400, 0x00000000, 0x00000400 // Constant
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

        vx_uint32 uniExtractInt32BgraToU8Bgr_2x8[16] = {
            0x00333333, // TCfg
            0x00111000, // ASelt
            0x00020100, 0x00000201, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002600, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };

        status |= vxSetNodeUniform(nodObj, "uniConvertInt32toUint8_2x8", 1, uniConvertInt32toUint8_2x8);
        status |= vxSetNodeUniform(nodObj, "uniExtractInt32BgraToU8Bgr_2x8", 1, uniExtractInt32BgraToU8Bgr_2x8);
        status |= vxSetNodeUniform(nodObj, "uniBilinearTmp1BgraShort_4x4", 1, uniBilinearTmp1BgraShort_4x4);
        status |= vxSetNodeUniform(nodObj, "uniBilinearTmp2BgraShort_4x4", 1, uniBilinearTmp2BgraShort_4x4);
        status |= vxSetNodeUniform(nodObj, "uniBilinearTmp3BgraShort_4x4", 1, uniBilinearTmp3BgraShort_4x4);
        status |= vxSetNodeUniform(nodObj, "uniBilinearTmp4BgraShort_4x4", 1, uniBilinearTmp4BgraShort_4x4);
        status |= vxSetNodeUniform(nodObj, "uniBilinearTmp5BgraShort_4x4", 1, uniBilinearTmp5BgraShort_4x4);
        status |= vxSetNodeUniform(nodObj, "uniBilinearTmp6BgraShort_4x4", 1, uniBilinearTmp6BgraShort_4x4);
        status |= vxSetNodeUniform(nodObj, "uniBilinearTmp7BgraShort_4x4", 1, uniBilinearTmp7BgraShort_4x4);
        status |= vxSetNodeUniform(nodObj, "uniBilinearTmp8BgraShort_4x4", 1, uniBilinearTmp8BgraShort_4x4);
        status |= vxSetNodeUniform(nodObj, "uniDescaleU8_4x4", 1, uniDescaleU8_4x4);

        status |= vxSetNodeUniform(nodObj, "zp", 1, &output_ZP);
        status |= vxSetNodeUniform(nodObj, "outputScale", 1, &outputScale);

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
vx_kernel_description_t vxPre_process_bgra_CPU =
{
    VX_KERNEL_ENUM_PRE_PROCESS_BGRA,
    _VX_KERNEL_NAME,
    _VX_KERNEL_FUNC_KERNEL,
    vxPre_process_bgraKernelParam,
    _cnt_of_array( vxPre_process_bgraKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxPre_process_bgra_scale =
{
    VX_KERNEL_ENUM_PRE_PROCESS_BGRA,
    VX_KERNEL_NAME_PRE_PROCESS_BGRA,
    NULL,
    vxPre_process_bgraKernelParam,
    _cnt_of_array( vxPre_process_bgraKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxPre_process_bgraInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxPre_process_bgra_copy =
{
    VX_KERNEL_ENUM_PRE_PROCESS_BGRA,
    VX_KERNEL_NAME_PRE_PROCESS_BGRA_COPY,
    NULL,
    vxPre_process_bgraKernelParam,
    _cnt_of_array( vxPre_process_bgraKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxPre_process_bgra_copyInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxPre_process_bgra_scale_trans =
{
    VX_KERNEL_ENUM_PRE_PROCESS_BGRA,
    VX_KERNEL_NAME_PRE_PROCESS_BGRA_TRANS,
    NULL,
    vxPre_process_bgraKernelParam,
    _cnt_of_array( vxPre_process_bgraKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxPre_process_bgra_transInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_PRE_PROCESS_BGRA_list[] =
{
    &vxPre_process_bgra_CPU,
    &vxPre_process_bgra_copy,
    &vxPre_process_bgra_scale,
    &vxPre_process_bgra_scale_trans,
    NULL
};
#ifdef __cplusplus
}
#endif
