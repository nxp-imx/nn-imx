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
#include <math.h>

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

#define _VX_KERNEL_VAR          (vx_kernel_ROI_ALIGN)
#define _VX_KERNEL_ID           (VX_KERNEL_ENUM_ROI_ALIGN)
#define _VX_KERNEL_NAME         (VX_KERNEL_NAME_ROI_ALIGN)
#define _VX_KERNEL_FUNC_KERNEL  (vxRoi_alignKernel)

#undef MAX
#define MAX(a,b)    ((a) > (b) ? (a) : (b))
#undef MIN
#define MIN(a,b)    ((a) < (b) ? (a) : (b))

static vsi_status VX_CALLBACK vxRoi_alignKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
#define ARG_NUM            (6)
#define TENSOR_NUM_INPUT (3)
#define TENSOR_NUM_OUTPUT (1)
#define TENSOR_NUM (TENSOR_NUM_INPUT+TENSOR_NUM_OUTPUT)

    vsi_status status = VSI_FAILURE;
    vx_context context = NULL;
    vx_tensor input[TENSOR_NUM_INPUT] = {0};
    vx_tensor output[TENSOR_NUM_OUTPUT] = {0};
    float *f32_in_buffer[TENSOR_NUM_INPUT] = {0};
    int32_t* int32_in_buffer[TENSOR_NUM_INPUT] = {0};
    float *f32_out_buffer[TENSOR_NUM_OUTPUT] = {0};
    vsi_nn_tensor_attr_t in_attr[TENSOR_NUM_INPUT];
    vsi_nn_tensor_attr_t out_attr[TENSOR_NUM_OUTPUT];
    uint32_t in_elements[TENSOR_NUM_INPUT] = {0};
    uint32_t out_elements[TENSOR_NUM_OUTPUT]= {0};

    int32_t output_height;
    int32_t output_width;
    float height_ratio;
    float width_ratio;
    int32_t height_sample_num;
    int32_t width_sample_num;

    int32_t i;
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
        in_elements[i] = vsi_nn_vxGetTensorElementNum(&in_attr[i]);
        if (i == 2)
        {
            int32_in_buffer[i] = (int32_t *)vsi_nn_vxCopyTensorToData(context,
                input[i], &in_attr[i]);
        }
        else
        {
            f32_in_buffer[i] = (float *)malloc(in_elements[i] * sizeof(float));
            status = vsi_nn_vxConvertTensorToFloat32Data(
                context, input[i], &in_attr[i], f32_in_buffer[i],
                in_elements[i] * sizeof(float));
            TEST_CHECK_STATUS(status, final);
        }
    }
    for(i = 0; i < TENSOR_NUM_OUTPUT; i ++)
    {
        output[i] = (vx_tensor)paramObj[i + TENSOR_NUM_INPUT];
        status = vsi_nn_vxGetTensorAttr(output[i], &out_attr[i]);
        TEST_CHECK_STATUS(status, final);
        out_elements[i] = vsi_nn_vxGetTensorElementNum(&out_attr[i]);
        f32_out_buffer[i]= (float *)malloc(out_elements[i] * sizeof(float));
        memset(f32_out_buffer[i], 0, out_elements[i] * sizeof(float));
    }
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM], &(output_height),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 1], &(output_width),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 2], &(height_ratio),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 3], &(width_ratio),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 4], &(height_sample_num),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 5], &(width_sample_num),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    /* TODO: Add CPU kernel implement */
    {
        uint32_t n, i, j, k;
        uint32_t numRois = in_attr[1].size[1];
        float heightScale = 1.0f / height_ratio;
        float widthScale = 1.0f / width_ratio;
        for(n = 0; n < numRois; n++)
        {
            uint32_t batchId = int32_in_buffer[2][n];
            float wRoiStart = f32_in_buffer[1][n * 4 + 0] * widthScale;
            float hRoiStart = f32_in_buffer[1][n * 4 + 1] * heightScale;
            float wRoiEnd = f32_in_buffer[1][n * 4 + 2] * widthScale;
            float hRoiEnd = f32_in_buffer[1][n * 4 + 3] * heightScale;

            float roiWidth = MAX((wRoiEnd - wRoiStart), 1.0f);
            float roiHeight = MAX((hRoiEnd - hRoiStart), 1.0f);
            float wStepSize = roiWidth / out_attr[0].size[0];
            float hStepSize = roiHeight / out_attr[0].size[1];

            uint32_t wSamplingRatio = width_sample_num > 0
                ? width_sample_num : (uint32_t)ceil(wStepSize);
            uint32_t hSamplingRatio = height_sample_num > 0
                ? height_sample_num : (uint32_t)ceil(hStepSize);
            int32_t numSamplingPoints = wSamplingRatio * hSamplingRatio;
            float wBinSize = wStepSize / (float)(wSamplingRatio);
            float hBinSize = hStepSize / (float)(hSamplingRatio);

            for (i = 0; i < out_attr[0].size[1]; i++)
            {
                for (j = 0; j < out_attr[0].size[0]; j++)
                {
                    float wStart = wStepSize * j + wRoiStart;
                    //float wEnd = wStepSize * (j + 1) + wRoiStart;
                    float hStart = hStepSize * i + hRoiStart;
                    //float hEnd = hStepSize * (i + 1) + hRoiStart;

                    uint32_t xInd, yInd;
                    for (k = 0; k < in_attr[0].size[2]; k++)
                    {
                        for (yInd = 0; yInd < hSamplingRatio; yInd++)
                        {
                            for (xInd = 0; xInd < wSamplingRatio; xInd++)
                            {
                                float y = hStart + hBinSize / 2 + hBinSize * yInd;
                                float x = wStart + wBinSize / 2 + wBinSize * xInd;
                                uint32_t x1 = (uint32_t)floor(x);
                                uint32_t y1 = (uint32_t)floor(y);
                                uint32_t x2 = x1 + 1, y2 = y1 + 1;
                                float dx1 = x - (float)(x1);
                                float dy1 = y - (float)(y1);

                                if (x1 >= in_attr[0].size[0] - 1) {
                                    x1 = x2 = in_attr[0].size[0] - 1;
                                    dx1 = 0;
                                }
                                if (y1 >= in_attr[0].size[1] - 1) {
                                    y1 = y2 = in_attr[0].size[1] - 1;
                                    dy1 = 0;
                                }

                                {
                                    float dx2 = 1.0f - dx1, dy2 = 1.0f - dy1;
                                    float ws[] = {dx2 * dy2, dx1 * dy2,
                                        dx2 * dy1, dx1 * dy1};
                                    uint32_t offsets[] = {y1 * in_attr[0].size[0] + x1,
                                        y1 * in_attr[0].size[0] + x2,
                                        y2 * in_attr[0].size[0] + x1,
                                        y2 * in_attr[0].size[0] + x2};

                                    float interpolation = 0;
                                    uint32_t in_coords[] = {0, 0, k, batchId};
                                    uint32_t in_index = vsi_nn_GetOffsetByCoords(
                                        &in_attr[0], in_coords);
                                    uint32_t out_coords[] = {j, i, k, n};
                                    uint32_t out_index = vsi_nn_GetOffsetByCoords(
                                        &out_attr[0], out_coords);
                                    uint32_t c;
                                    for (c = 0; c < 4; c++)
                                    {
                                        interpolation += ws[c]
                                        * f32_in_buffer[0][in_index + offsets[c]];
                                    }
                                    interpolation /= (float)(numSamplingPoints);
                                    f32_out_buffer[0][out_index] = interpolation;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /* save data */
    for(i = 0; i < TENSOR_NUM_OUTPUT; i++)
    {
        status = vsi_nn_vxConvertFloat32DataToTensor(
            context, output[i], &out_attr[i], f32_out_buffer[i],
            out_elements[i] * sizeof(float));
        TEST_CHECK_STATUS(status, final);
    }

final:
    for (i = 0; i < TENSOR_NUM_INPUT; i++)
    {
        if (f32_in_buffer[i]) free(f32_in_buffer[i]);
        if (int32_in_buffer[i]) free(int32_in_buffer[i]);
    }
    for(i = 0; i < TENSOR_NUM_OUTPUT; i++)
    {
        if (f32_out_buffer[i]) free(f32_out_buffer[i]);
    }
    return status;
} /* _VX_KERNEL_FUNC_KERNEL() */

static vx_param_description_t vxRoi_alignKernelParam[] =
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
};

vx_status VX_CALLBACK vxRoi_alignInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    vx_uint32 paraNum
    )
{
    vx_status status = VX_SUCCESS;
    /*TODO: Add initial code for VX program*/

    return status;
}


#ifdef __cplusplus
extern "C" {
#endif
vx_kernel_description_t vxRoi_align_CPU =
{
    _VX_KERNEL_ID,
    _VX_KERNEL_NAME,
    _VX_KERNEL_FUNC_KERNEL,
    vxRoi_alignKernelParam,
    _cnt_of_array( vxRoi_alignKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxRoi_align_VX =
{
    _VX_KERNEL_ID,
    _VX_KERNEL_NAME,
    NULL,
    vxRoi_alignKernelParam,
    _cnt_of_array( vxRoi_alignKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxRoi_alignInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_ROI_ALIGN_list[] =
{
    &vxRoi_align_CPU,
    &vxRoi_align_VX,
    NULL
};
#ifdef __cplusplus
}
#endif
