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
#include <float.h>
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

#define _VX_KERNEL_VAR          (vx_kernel_DETECTION_POSTPROCESS)
#define _VX_KERNEL_ID           (VX_KERNEL_ENUM_DETECTION_POSTPROCESS)
#define _VX_KERNEL_NAME         (VX_KERNEL_NAME_DETECTION_POSTPROCESS)
#define _VX_KERNEL_FUNC_KERNEL  (vxDetection_postprocessKernel)

#undef MAX
#define MAX(a,b)    ((a) > (b) ? (a) : (b))
#undef MIN
#define MIN(a,b)    ((a) < (b) ? (a) : (b))

// swap_element is implemented in vsi_nn_kernel_box_with_nms_limit.c
void swap_element
    (
    uint32_t* list,
    uint32_t first,
    uint32_t second
    );

// max_element is implemented in vsi_nn_kernel_box_with_nms_limit.c
uint32_t max_element
    (
    float* data,
    uint32_t* index_list,
    uint32_t len
    );

// getIoUAxisAligned is implemented in vsi_nn_kernel_box_with_nms_limit.c
float getIoUAxisAligned
    (
    const float* roi1,
    const float* roi2
    );

// sort_element_by_score is implemented in vsi_nn_kernel_box_with_nms_limit.c
void sort_element_by_score
    (
    float* data,
    uint32_t* index_list,
    uint32_t len
    );

float max_element_value
    (
    float* data,
    uint32_t len
    )
{
    uint32_t i;
    float max_val = data[0];
    for(i = 1; i < len; i++)
    {
        float val = data[i];
        if (max_val < val)
        {
            max_val = val;
        }
    }
    return max_val;
}

void iota
    (
    int32_t * data,
    uint32_t len,
    int32_t value
    )
{
    uint32_t i;
    for(i = 0; i < len; i++)
    {
        data [i] = value;
        value++;
    }
}

static vsi_status VX_CALLBACK vxDetection_postprocessKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
#define ARG_NUM            (11)
#define TENSOR_NUM_INPUT (3)
#define TENSOR_NUM_OUTPUT (4)
#define TENSOR_NUM (TENSOR_NUM_INPUT+TENSOR_NUM_OUTPUT)

    vsi_status status = VSI_FAILURE;
    vx_context context = NULL;
    vx_tensor input[TENSOR_NUM_INPUT] = {0};
    vx_tensor output[TENSOR_NUM_OUTPUT] = {0};
    float *f32_in_buffer[TENSOR_NUM_INPUT] = {0};
    float *f32_out_buffer[TENSOR_NUM_OUTPUT] = {0};
    int32_t* int32_out_buffer[TENSOR_NUM_OUTPUT] = {0};
    vsi_nn_tensor_attr_t in_attr[TENSOR_NUM_INPUT];
    vsi_nn_tensor_attr_t out_attr[TENSOR_NUM_OUTPUT];
    uint32_t in_elements[TENSOR_NUM_INPUT] = {0};
    uint32_t out_elements[TENSOR_NUM_OUTPUT]= {0};

    float scaleY;
    float scaleX;
    float scaleH;
    float scaleW;
    int32_t useRegularNms;
    int32_t maxNumDetections;
    int32_t maxClassesPerDetection;
    int32_t maxNumDetectionsPerClass;
    float scoreThreshold;
    float iouThreshold;
    int32_t isBGInLabel;

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
        f32_in_buffer[i] = (float *)malloc(in_elements[i] * sizeof(float));
        status = vsi_nn_vxConvertTensorToFloat32Data(
            context, input[i], &in_attr[i], f32_in_buffer[i],
            in_elements[i] * sizeof(float));
        TEST_CHECK_STATUS(status, final);
    }
    for(i = 0; i < TENSOR_NUM_OUTPUT; i ++)
    {
        output[i] = (vx_tensor)paramObj[i + TENSOR_NUM_INPUT];
        status = vsi_nn_vxGetTensorAttr(output[i], &out_attr[i]);
        TEST_CHECK_STATUS(status, final);
        out_elements[i] = vsi_nn_vxGetTensorElementNum(&out_attr[i]);
        if (i < 2)
        {
            f32_out_buffer[i] = (float *)malloc(out_elements[i] * sizeof(float));
            memset(f32_out_buffer[i], 0, out_elements[i] * sizeof(float));
        }
        else
        {
            int32_out_buffer[i] = (int32_t *)malloc(out_elements[i] * sizeof(int32_t));
            memset(int32_out_buffer[i], 0, out_elements[i] * sizeof(int32_t));
        }
    }
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM], &(scaleY),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 1], &(scaleX),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 2], &(scaleH),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 3], &(scaleW),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 4], &(useRegularNms),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 5], &(maxNumDetections),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 6], &(maxClassesPerDetection),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 7], &(maxNumDetectionsPerClass),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 8], &(scoreThreshold),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 9], &(iouThreshold),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM + 10], &(isBGInLabel),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    /* TODO: Add CPU kernel implement */
    {
        const uint32_t kRoiDim = 4;
        uint32_t numBatches = in_attr[0].size[2];
        uint32_t numAnchors = in_attr[0].size[1];
        uint32_t numClasses = in_attr[0].size[0];
        uint32_t lengthBoxEncoding = in_attr[1].size[0];
        uint32_t numOutDetection = out_attr[0].size[0];
        float* roiBuffer = (float*)malloc(numAnchors * kRoiDim * sizeof(float));
        float* scoreBuffer = (float*)malloc(numAnchors * sizeof(float));
        uint32_t* select = (uint32_t*)malloc(numAnchors * numClasses * sizeof(uint32_t));
        uint32_t a, b, c, n, i, j;
        uint32_t scores_index = 0;
        uint32_t scores_out_index = 0;
        uint32_t roi_out_index = 0;
        uint32_t class_out_index = 0;
        float* maxScores = (float*)malloc(numAnchors * sizeof(float));
        uint32_t* scoreInds = (uint32_t*)malloc((numClasses - 1) * sizeof(uint32_t));

        for (n = 0; n < numBatches; n++)
        {
            for (a = 0; a < numAnchors; a++)
            {
                float yCtr = f32_in_buffer[2][a * kRoiDim] + f32_in_buffer[2][a * kRoiDim + 2]
                    * f32_in_buffer[1][a * lengthBoxEncoding] / scaleY;
                float xCtr = f32_in_buffer[2][a * kRoiDim + 1] + f32_in_buffer[2][a * kRoiDim + 3]
                    * f32_in_buffer[1][a * lengthBoxEncoding + 1] / scaleX;
                float hHalf = f32_in_buffer[2][a * kRoiDim + 2] *
                    (float)exp(f32_in_buffer[1][a * lengthBoxEncoding + 2] / scaleH) * 0.5f;
                float wHalf = f32_in_buffer[2][a * kRoiDim + 3] *
                    (float)exp(f32_in_buffer[1][a * lengthBoxEncoding + 3] / scaleW) * 0.5f;
                roiBuffer[a * kRoiDim] = yCtr - hHalf;
                roiBuffer[a * kRoiDim + 1] = xCtr - wHalf;
                roiBuffer[a * kRoiDim + 2] = yCtr + hHalf;
                roiBuffer[a * kRoiDim + 3] = xCtr + wHalf;
            }
            if (useRegularNms)
            {
                uint32_t select_size = 0;
                uint32_t select_start = 0;
                uint32_t select_len = 0;
                uint32_t numDetections = 0;
                for (c = 1; c < numClasses; c++)
                {
                    select_start = select_size;
                    for (b = 0; b < numAnchors; b++)
                    {
                        const uint32_t index = b * numClasses + c;
                        float score = f32_in_buffer[0][scores_index + index];
                        if (score > scoreThreshold) {
                            select[select_size] = index;
                            select_size++;
                        }
                    }
                    select_len = select_size - select_start;

                    if (maxNumDetectionsPerClass < 0)
                    {
                        maxNumDetectionsPerClass = select_len;
                    }
                    numDetections = 0;
                    for (j = 0; (j < select_len && numDetections < (uint32_t)maxNumDetectionsPerClass); j++)
                    {
                        // find max score and swap to the front.
                        int32_t max_index = max_element(&(f32_in_buffer[0][scores_index]),
                            &(select[select_start]), select_len);
                        swap_element(&(select[select_start]), max_index, j);

                        // Calculate IoU of the rest, swap to the end (disgard) if needed.
                        for (i = j + 1; i < select_len; i++)
                        {
                            int32_t roiBase0 = (select[select_start + i] / numClasses) * kRoiDim;
                            int32_t roiBase1 = (select[select_start + j] / numClasses) * kRoiDim;
                            float iou = getIoUAxisAligned(&(roiBuffer[roiBase0]),
                                &(roiBuffer[roiBase1]));

                            if (iou >= iouThreshold)
                            {
                                swap_element(&(select[select_start]), i, select_len - 1);
                                i--;
                                select_len--;
                            }
                        }
                        numDetections++;
                    }
                    select_size = select_start + numDetections;
                }

                //select_size = select_start + select_len;
                select_len = select_size;
                select_start = 0;

                // Take top maxNumDetections.
                sort_element_by_score(&(f32_in_buffer[0][scores_index]),
                    &(select[select_start]), select_len);

                for (i = 0; i < select_len; i++)
                {
                    uint32_t ind = select[i];
                    f32_out_buffer[0][scores_out_index + i] =
                        f32_in_buffer[0][scores_index + ind];
                    memcpy(&(f32_out_buffer[1][roi_out_index + i * kRoiDim]),
                        &roiBuffer[(ind / numClasses) * kRoiDim], kRoiDim * sizeof(float));
                    int32_out_buffer[2][class_out_index + i] = (ind % numClasses)
                        - (isBGInLabel ? 0 : 1);
                }
                int32_out_buffer[3][n] = select_len;
            }
            else
            {
                uint32_t numOutClasses = MIN(numClasses - 1, (uint32_t)maxClassesPerDetection);
                uint32_t select_size = 0;
                uint32_t select_start = 0;
                uint32_t select_len = 0;
                uint32_t numDetections = 0;
                for (a = 0; a < numAnchors; a++)
                {
                    // exclude background class: 0
                    maxScores[a] = max_element_value(&(f32_in_buffer[0]
                        [scores_index + a * numClasses + 1]), numClasses - 1);
                    if (maxScores[a] > scoreThreshold)
                    {
                            select[select_size] = a;
                            select_size++;
                    }
                }
                select_len = select_size - select_start;

                if (maxNumDetections < 0)
                {
                    maxNumDetections = select_len;
                }
                for (j = 0; (j < select_len && numDetections < (uint32_t)maxNumDetectionsPerClass); j++)
                {
                    // find max score and swap to the front.
                    int32_t max_index = max_element(maxScores,
                        &(select[select_start]), select_len);
                    swap_element(&(select[select_start]), max_index, j);

                    // Calculate IoU of the rest, swap to the end (disgard) if needed.
                    for (i = j + 1; i < select_len; i++)
                    {
                        int32_t roiBase0 = select[i] * kRoiDim;
                        int32_t roiBase1 = select[j] * kRoiDim;
                        float iou = getIoUAxisAligned(&(roiBuffer[roiBase0]),
                            &(roiBuffer[roiBase1]));
                        if (iou >= iouThreshold)
                        {
                            swap_element(&(select[select_start]), i, select_len - 1);
                            i--;
                            select_len--;
                        }
                        numDetections++;
                    }
                }
                select_size = select_start + select_len;

                for (i = 0; i < select_len; i++)
                {
                    iota((int32_t*)scoreInds, numClasses - 1, 1);
                    sort_element_by_score(&(f32_in_buffer[0][scores_index + i * numClasses]),
                        scoreInds, numClasses - 1);
                    for (c = 0; c < numOutClasses; c++)
                    {
                        f32_out_buffer[0][scores_out_index + i * numOutClasses + c] =
                            f32_in_buffer[0][scores_index + i * numClasses + scoreInds[c]];
                        memcpy(&(f32_out_buffer[1][roi_out_index + (i * numOutClasses + c)
                            * kRoiDim]), &roiBuffer[i * kRoiDim], kRoiDim * sizeof(float));
                        int32_out_buffer[2][class_out_index + + i * numOutClasses + c]
                            = scoreInds[c] - (isBGInLabel ? 0 : 1);
                    }
                }
            }
            scores_index += numAnchors * numClasses;
            scores_out_index += numOutDetection;
            roi_out_index += numOutDetection * kRoiDim;
            class_out_index += numOutDetection;
        }

        if (roiBuffer) free(roiBuffer);
        if (scoreBuffer) free(scoreBuffer);
        if (select) free(select);
        if (maxScores) free(maxScores);
        if (scoreInds) free(scoreInds);
    }

    /* save data */
    for(i = 0; i < TENSOR_NUM_OUTPUT; i++)
    {
        if (i < 2)
        {
            status = vsi_nn_vxConvertFloat32DataToTensor(
                context, output[i], &out_attr[i], f32_out_buffer[i],
                out_elements[i] * sizeof(float));
            TEST_CHECK_STATUS(status, final);
        }
        else
        {
            vsi_nn_vxCopyDataToTensor(context, output[i], &out_attr[i],
                (uint8_t *)int32_out_buffer[i]);
        }
    }

final:
    for (i = 0; i < TENSOR_NUM_INPUT; i++)
    {
        if (f32_in_buffer[i]) free(f32_in_buffer[i]);
    }
    for(i = 0; i < TENSOR_NUM_OUTPUT; i++)
    {
        if (f32_out_buffer[i]) free(f32_out_buffer[i]);
        if (int32_out_buffer[i]) free(int32_out_buffer[i]);
    }
    return status;
} /* _VX_KERNEL_FUNC_KERNEL() */

static vx_param_description_t vxDetection_postprocessKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
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
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

vx_status VX_CALLBACK vxDetection_postprocessInitializer
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
vx_kernel_description_t vxDetection_postprocess_CPU =
{
    _VX_KERNEL_ID,
    _VX_KERNEL_NAME,
    _VX_KERNEL_FUNC_KERNEL,
    vxDetection_postprocessKernelParam,
    _cnt_of_array( vxDetection_postprocessKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxDetection_postprocess_VX =
{
    _VX_KERNEL_ID,
    _VX_KERNEL_NAME,
    NULL,
    vxDetection_postprocessKernelParam,
    _cnt_of_array( vxDetection_postprocessKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxDetection_postprocessInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_DETECTION_POSTPROCESS_list[] =
{
    &vxDetection_postprocess_CPU,
    &vxDetection_postprocess_VX,
    NULL
};
#ifdef __cplusplus
}
#endif
