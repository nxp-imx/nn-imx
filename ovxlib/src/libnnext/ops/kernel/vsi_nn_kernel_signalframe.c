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
#include <math.h>

#include "vsi_nn_platform.h"

#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_math.h"
#include "client/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"

#define INPUT_FP16 0
#define OUTPUT_FP16 0

vx_status getFactor(vx_uint32 data, vx_uint32 *factor, vx_uint32 minLimit, vx_uint32 maxLimit, vx_uint32 alignData)
{
    vx_uint32 i         = 0;
    vx_uint32 maxFactor = alignData - 1;
    vx_status status    = VX_FAILURE;

    for (i = minLimit; i <= maxLimit; i ++)
    {
        if (data % i == 0)
        {
            if (status == VX_FAILURE && data % i == 0)
            {
                *factor      = i;
                maxFactor    = i;
                status       = VX_SUCCESS;
                continue;
            }
            else if ((i % alignData) < (maxFactor % alignData))
            {
               *factor      = i;
               maxFactor    = i;
               status       = VX_SUCCESS;
            }
        }
    }

    return status;
}

void mySignalFrameFunc
    (
    void* imgIn,
    void* imgOut,
    uint32_t input_dim,
    uint32_t width,
    uint32_t height,
    uint32_t channel,
    uint32_t batch,
    uint32_t frame_len, // window size
    uint32_t step,
    uint32_t pad_end,
    uint32_t pad_val,
    uint32_t axis,
    uint32_t *dstW,
    uint32_t *dstH,
    uint32_t *dstC,
    uint32_t *dstB
    )
{
    uint8_t* tmpIn = (uint8_t*)imgIn;
    uint8_t* tmpOut = (uint8_t*)imgOut;

    uint32_t i,j,k;
    uint32_t size = 0;
    uint32_t iter = 0;

    if(input_dim == 1)
    {
        if(axis != 0)
        {
            printf("error.\n");
            return;
        }
        *dstW = frame_len;
        //*dstH = (len - frame_len) / step  + 1;
        *dstH = pad_end ? ((width + step - 1 ) / step) : ((width - frame_len) / step  + 1);
        *dstC = 1;
        *dstB = 1;

        size = (*dstW) * sizeof(int16_t);
        iter = pad_end ? width : (width - frame_len + 1);
        if(pad_end)
        {
            int16_t* output = (int16_t*)tmpOut;
            int16_t* input = (int16_t*)tmpIn;
            uint32_t m = 0;
            for(i = 0, j = 0; i < iter; i += step)
            {
                for(m = i; m < frame_len + i; m++)
                {
                    if(m >= width)
                    {
                        output[j] = 0;
                    }
                    else
                    {
                        output[j] = input[m];
                    }
                    j++;
                }
            }
        }
        else
        {
            for(i = 0, j = 0; i < iter; i += step, j++)
            {
                memcpy(tmpOut + j * size, tmpIn + i * sizeof(int16_t), size);
            }
        }
    }
    else if(input_dim == 2)
    {
        if(axis == 0)
        {
            uint8_t* src = tmpIn;
            uint8_t* dst = tmpOut;

            *dstH = frame_len;
            *dstW = width;
            *dstC = pad_end ? ((height + step - 1) / step) : ((height - frame_len) / step  + 1);

            *dstB = 1;

            size = width * frame_len * sizeof(int16_t);
            iter = pad_end ? (height) : (height - frame_len + 1);
            if(pad_end)
            {
                uint32_t m = 0;
                size = width * sizeof(int16_t);
                for(i = 0, j = 0; i < iter; i += step)
                {
                    for(m = i; m < frame_len + i; m++)
                    {
                        if(m >= height)
                        {
                            memset(dst + j * size, 0, size);
                        }
                        else
                        {
                            memcpy(dst + j * size, src + m * width * sizeof(int16_t), size);
                        }
                        j++;
                    }
                }
            }
            else
            {
                for(i = 0, j = 0; i < iter; i += step, j++)
                {
                    memcpy(dst + j * size, src + i * width * sizeof(int16_t), size);
                }
            }
        }
        else if(axis == 1)
        {
            *dstW = frame_len;

            //*dstH = (len - frame_len) / step  + 1;
            *dstH = pad_end ? ((width + step - 1 ) / step) : ((width - frame_len) / step  + 1);

            *dstC = height;
            *dstB = 1;

            size = (*dstW) * sizeof(int16_t);
            iter = pad_end ? width : (width - frame_len + 1);
            if(pad_end)
            {
                for(k = 0; k < height; k++)
                {
                    uint8_t* src = tmpIn + k * width * sizeof(int16_t);
                    uint8_t* dst = tmpOut + k * (*dstW) * (*dstH) * sizeof(int16_t);

                    int16_t* output = (int16_t*)dst;
                    int16_t* input = (int16_t*)src;
                    uint32_t m = 0;
                    for(i = 0, j = 0; i < iter; i += step)
                    {
                        for(m = i; m < frame_len + i; m++)
                        {
                            if(m >= width)
                            {
                                output[j] = 0;
                            }
                            else
                            {
                                output[j] = input[m];
                            }
                            j++;
                        }
                    }
                }
            }
            else
            {
                for(k = 0; k < height; k++)
                {
                    uint8_t* src = tmpIn + k * width * sizeof(int16_t);
                    uint8_t* dst = tmpOut + k * (*dstW) * (*dstH) * sizeof(int16_t);

                    for(i = 0, j = 0; i < iter; i += step, j++)
                    {
                        memcpy(dst + j * size, src + i * sizeof(int16_t), size);
                    }
                }
            }
        }
    }
    else if(input_dim == 3)
    {
        if(axis == 0)
        {
            uint8_t* src = tmpIn;
            uint8_t* dst = tmpOut;
            size = width * height * frame_len * sizeof(int16_t);

            *dstW = width;
            *dstH = height;
            *dstC = frame_len;
            *dstB = pad_end ? ((channel + step - 1) / step) :((channel - frame_len) / step  + 1);
            iter = pad_end ? channel : (channel - frame_len + 1);
            if(pad_end)
            {
                uint32_t m = 0;
                size = width * height * sizeof(int16_t);
                for(i = 0, j = 0; i < iter; i += step)
                {
                    for(m = i; m < frame_len + i; m++)
                    {
                        if(m >= channel)
                        {
                            memset(dst + j * size, 0 , size);
                        }
                        else
                        {
                            memcpy(dst + j * size, src + m * width * height * sizeof(int16_t), size);
                        }
                        j++;
                    }
                }
            }
            else
            {
                for(i = 0, j = 0; i < iter; i += step, j++)
                {
                    memcpy(dst + j * size, src + i * width * height * sizeof(int16_t), size);
                }
            }
        }
        else if(axis == 1)
        {
            *dstH = frame_len;
            *dstW = width;
            *dstC = pad_end ? ((height + step - 1) / step) : ((height - frame_len) / step  + 1);
            *dstB = channel;

            size = width * frame_len * sizeof(int16_t);
            iter = pad_end ? (height) : (height - frame_len + 1);
            if(pad_end)
            {
                uint32_t m = 0;
                size = width * sizeof(int16_t);
                for(k = 0; k < channel; k++)
                {
                    uint8_t* src = tmpIn + k * width * height* sizeof(int16_t);
                    uint8_t* dst = tmpOut + k * (*dstC) * (*dstW) * (*dstH) * sizeof(int16_t);

                    for(i = 0, j = 0; i < iter; i += step)
                    {
                        for(m = i; m < frame_len + i; m++)
                        {
                            if(m >= height)
                            {
                                memset(dst + j * size, 0, size);
                            }
                            else
                            {
                                memcpy(dst + j * size, src + m * width * sizeof(int16_t), size);
                            }
                            j++;
                        }
                    }
                }
            }
            else
            {
                for(k = 0; k < channel; k++)
                {
                    uint8_t* src = tmpIn + k * width * height* sizeof(int16_t);
                    uint8_t* dst = tmpOut + k * (*dstC) * (*dstW) * (*dstH) * sizeof(int16_t);

                    for(i = 0, j = 0; i < iter; i += step, j++)
                    {
                        memcpy(dst + j * size, src + i * width * sizeof(int16_t), size);
                    }
                }
            }
        }
        else if(axis == 2)
        {
            //*dstH = (len - frame_len) / step  + 1;
            *dstH = pad_end ? ((width + step - 1 ) / step) : ((width - frame_len) / step  + 1);
            *dstW = frame_len;
            *dstC = height;
            *dstB = channel;

            size = (*dstW) * sizeof(int16_t);
            iter = pad_end ? width : (width - frame_len + 1);

            if(pad_end)
            {
                for(k = 0; k < channel * height; k++)
                {
                    uint8_t* src = tmpIn + k * width * sizeof(int16_t);
                    uint8_t* dst = tmpOut + k * (*dstW) * (*dstH) * sizeof(int16_t);

                    int16_t* output = (int16_t*)dst;
                    int16_t* input = (int16_t*)src;
                    uint32_t m = 0;
                    for(i = 0, j = 0; i < iter; i += step)
                    {
                        for(m = i; m < frame_len + i; m++)
                        {
                            if(m >= width)
                            {
                                output[j] = 0;
                            }
                            else
                            {
                                output[j] = input[m];
                            }
                            j++;
                        }
                    }
                }
            }
            else
            {
                for(k = 0; k < channel * height; k++)
                {
                    uint8_t* src = tmpIn + k * width * sizeof(int16_t);
                    uint8_t* dst = tmpOut + k * (*dstW) * (*dstH) * sizeof(int16_t);
                    for(i = 0, j = 0; i < iter; i += step, j++)
                    {
                        memcpy(dst + j * size, src + i * sizeof(int16_t), size);
                    }
                }
            }
        }
    }

    return;
}

vsi_status VX_CALLBACK vxSignalFrameKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
    vsi_status status = VX_ERROR_INVALID_PARAMETERS;

    if(paramNum == 7)
    {
        vx_context context = NULL;
        // tensor
        vx_tensor imgObj[7] = { NULL };
#if INPUT_FP16
        int16_t *input = NULL;
#else
        uint8_t *input = NULL;
#endif
#if OUTPUT_FP16
        int16_t *output = NULL;
#else
        uint8_t *output = NULL;
#endif
#if 0
        uint32_t *length = NULL;
        uint32_t *step = NULL;
        uint32_t *padFlg = NULL;
        uint32_t *pad = NULL;
        uint32_t *axis = NULL;
#endif
        uint32_t input_size[DIM_SIZE] = {0}, output_size[DIM_SIZE] = {0}, dst_size[DIM_SIZE] = {0};
#if 0
        uint32_t length_size[DIM_SIZE] = {0}, step_size[DIM_SIZE] = {0};
        uint32_t padFlg_size[DIM_SIZE] = {0}, pad_size[DIM_SIZE] = {0};
        uint32_t axis_size[DIM_SIZE] = {0}, dst_size[DIM_SIZE] = {0};
#endif
        uint32_t input_stride_size[4]  = {0};
        uint32_t output_stride_size[4] = {0};
#if 0
        uint32_t length_stride_size[4] = {0};
        uint32_t step_stride_size[4] = {0};
        uint32_t padFlg_stride_size[4] = {0};
        uint32_t pad_stride_size[4] = {0};
        uint32_t axis_stride_size[4] = {0};
#endif
        vx_tensor_addressing input_user_addr = NULL;
        vx_tensor_addressing output_user_addr = NULL;
#if 0
        vx_tensor_addressing window_user_addr = NULL;
        vx_tensor_addressing step_user_addr = NULL;
        vx_tensor_addressing padEnd_user_addr = NULL;
        vx_tensor_addressing padVal_user_addr = NULL;
        vx_tensor_addressing axis_user_addr = NULL;
#endif

        vsi_nn_type_e inputFormat = VSI_NN_TYPE_FLOAT16, outputFormat = VSI_NN_TYPE_FLOAT16;
        uint32_t input_dims = 0, output_dims = 0, tmpDim = 0;
        //uint32_t length_dims = 0, step_dims = 0, padFlg_dims = 0, pad_dims = 0, axis_dims = 0;
        uint32_t i;
        int32_t in_zp, out_zp;
        float in_scale, out_scale;
        vx_scalar scalar[5] = { NULL };
        uint32_t frame_length = 0, step = 0, pad_end = 0, pad = 0, axis = 0, axis0 = 0;

        imgObj[0] = (vx_tensor)paramObj[0];
        imgObj[1] = (vx_tensor)paramObj[1];  //output
        scalar[0] = (vx_scalar)paramObj[2];
        scalar[1] = (vx_scalar)paramObj[3];
        scalar[2] = (vx_scalar)paramObj[4];
        scalar[3] = (vx_scalar)paramObj[5];
        scalar[4] = (vx_scalar)paramObj[6];
        context = vxGetContext((vx_reference)node);
        if (context == NULL)
        {
            VSILOGE("vxGetContext failure! at line %d\n", __LINE__);
            goto OnError;
        }
        //input
        status = vxQueryTensor(imgObj[0], VX_TENSOR_NUM_OF_DIMS, &input_dims, sizeof(input_dims));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor input_dims failure! at line %d\n", __LINE__);
            goto OnError;
        }
        status = vxQueryTensor(imgObj[0], VX_TENSOR_DATA_TYPE, &inputFormat, sizeof(inputFormat));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor inputFormat failure! at line %d\n", __LINE__);
            goto OnError;
        }
        status = vxQueryTensor(imgObj[0], VX_TENSOR_DIMS, input_size, sizeof(input_size));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor input_size failure! at line %d\n", __LINE__);
            goto OnError;
        }
        status = vxQueryTensor(imgObj[0], VX_TENSOR_ZERO_POINT, &in_zp, sizeof(in_zp));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor input_size failure! at line %d\n", __LINE__);
            goto OnError;
        }
        status = vxQueryTensor(imgObj[0], VX_TENSOR_SCALE, &in_scale, sizeof(in_scale));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor input_size failure! at line %d\n", __LINE__);
            goto OnError;
        }
        //output
        status  = vxQueryTensor(imgObj[1], VX_TENSOR_DATA_TYPE, &outputFormat, sizeof(outputFormat));
        status |= vxQueryTensor(imgObj[1], VX_TENSOR_NUM_OF_DIMS, &output_dims, sizeof(output_dims));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor outputFormat failure! at line %d\n", __LINE__);
            goto OnError;
        }
        status = vxQueryTensor(imgObj[1], VX_TENSOR_DIMS, output_size, sizeof(output_size));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor output_size failure! at line %d\n", __LINE__);
            goto OnError;
        }
        status = vxQueryTensor(imgObj[1], VX_TENSOR_ZERO_POINT, &out_zp, sizeof(out_zp));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor input_size failure! at line %d\n", __LINE__);
            goto OnError;
        }
        status = vxQueryTensor(imgObj[1], VX_TENSOR_SCALE, &out_scale, sizeof(out_scale));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor input_size failure! at line %d\n", __LINE__);
            goto OnError;
        }

#if 0
        status |= vxQueryTensor(imgObj[2], VX_TENSOR_DATA_TYPE, &paraFormat, sizeof(paraFormat));
        status |= vxQueryTensor(imgObj[2], VX_TENSOR_NUM_OF_DIMS, &length_dims, sizeof(length_dims));
        status |= vxQueryTensor(imgObj[2], VX_TENSOR_DIMS, length_size, sizeof(length_size));
        status |= vxQueryTensor(imgObj[3], VX_TENSOR_NUM_OF_DIMS, &step_dims, sizeof(step_dims));
        status |= vxQueryTensor(imgObj[3], VX_TENSOR_DIMS, step_size, sizeof(step_size));
        status |= vxQueryTensor(imgObj[4], VX_TENSOR_NUM_OF_DIMS, &padFlg_dims, sizeof(padFlg_dims));
        status |= vxQueryTensor(imgObj[4], VX_TENSOR_DIMS, padFlg_size, sizeof(padFlg_size));
        status |= vxQueryTensor(imgObj[5], VX_TENSOR_NUM_OF_DIMS, &pad_dims, sizeof(pad_dims));
        status |= vxQueryTensor(imgObj[5], VX_TENSOR_DIMS, pad_size, sizeof(pad_size));
        status |= vxQueryTensor(imgObj[6], VX_TENSOR_NUM_OF_DIMS, &axis_dims, sizeof(axis_dims));
        status |= vxQueryTensor(imgObj[6], VX_TENSOR_DIMS, axis_size, sizeof(axis_size));
#endif

        input_size[2] = (input_dims <= 2)?1:input_size[2];
        input_size[3] = (input_dims <= 3)?1:input_size[3];

        input_stride_size[0]  = vsi_nn_GetTypeBytes(inputFormat);
        output_stride_size[0] = vsi_nn_GetTypeBytes(outputFormat);
        //length_stride_size[0] = vsi_nn_GetTypeBytes(paraFormat);
        for (i=1; i< input_dims; i++)
        {
            input_stride_size[i]  = input_stride_size[i-1] * input_size[i-1];
        }
        for (i=1; i< output_dims; i++)
        {
            output_stride_size[i] = output_stride_size[i-1] * output_size[i-1];
        }
#if 0
        for (i=1; i< length_dims; i++)
        {
            length_stride_size[i] = length_stride_size[i-1] * length_stride_size[i-1];
        }
#endif

#if INPUT_FP16
        input  = (int16_t*)malloc(input_size[0]*input_size[1]*input_size[2]*sizeof(int16_t));
#else
        input  = (uint8_t*)malloc(input_size[0]*input_size[1]*input_size[2]*vsi_nn_GetTypeBytes(inputFormat));
#endif
#if OUTPUT_FP16
        output = (int16_t*)malloc(output_size[0]*output_size[1]*output_size[2]*sizeof(int16_t));
#else
        output = (uint8_t*)malloc(output_size[0]*output_size[1]*output_size[2]*vsi_nn_GetTypeBytes(outputFormat));
#endif

#if 0
        length = (uint32_t*)malloc(1*sizeof(uint32_t));
        step = (uint32_t*)malloc(1*sizeof(uint32_t));
        padFlg = (uint32_t*)malloc(1*sizeof(uint32_t));
        pad = (uint32_t*)malloc(1*sizeof(uint32_t));
        axis = (uint32_t*)malloc(1*sizeof(uint32_t));
#endif

        input_user_addr = vxCreateTensorAddressing(context, input_size, input_stride_size, input_dims);
        vxCopyTensorPatch(imgObj[0], NULL, input_user_addr, input, VX_READ_ONLY, 0);
#if 0
        window_user_addr = vxCreateTensorAddressing(context, length_size, length_stride_size, length_dims);
        vxCopyTensorPatch(imgObj[2], NULL, window_user_addr, length, VX_READ_ONLY, 0);
        step_user_addr = vxCreateTensorAddressing(context, length_size, length_stride_size, length_dims);
        vxCopyTensorPatch(imgObj[3], NULL, step_user_addr, step, VX_READ_ONLY, 0);
        padEnd_user_addr = vxCreateTensorAddressing(context, length_size, length_stride_size, length_dims);
        vxCopyTensorPatch(imgObj[4], NULL, padEnd_user_addr, padFlg, VX_READ_ONLY, 0);
        padVal_user_addr = vxCreateTensorAddressing(context, length_size, length_stride_size, length_dims);
        vxCopyTensorPatch(imgObj[5], NULL, padVal_user_addr, pad, VX_READ_ONLY, 0);
        axis_user_addr = vxCreateTensorAddressing(context, length_size, length_stride_size, length_dims);
        vxCopyTensorPatch(imgObj[6], NULL, axis_user_addr, axis, VX_READ_ONLY, 0);
#endif
        // scalar
        status = vxCopyScalar(scalar[0], &frame_length, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
        status |= vxCopyScalar(scalar[1], &step, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
        status |= vxCopyScalar(scalar[2], &pad_end, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
        status |= vxCopyScalar(scalar[3], &pad, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
        status |= vxCopyScalar(scalar[4], &axis, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxCopyScalar failure! at line %d\n", __LINE__);
            goto OnError;
        }

        // Call C Prototype
        if(output_dims == 2)
            tmpDim = 1;
        else
            tmpDim = input_dims;
        {
            axis0 = input_dims - axis - 1;
        }
        mySignalFrameFunc(input, output, tmpDim, input_size[0],
            input_size[1], input_size[2], input_size[3],
            frame_length, step, pad_end, pad, axis0,
            &dst_size[0], &dst_size[1], &dst_size[2], &dst_size[3]);

        //output tensor
        output_user_addr = vxCreateTensorAddressing(context, output_size,
            output_stride_size, output_dims);
        vxCopyTensorPatch(imgObj[1], NULL, output_user_addr, output, VX_WRITE_ONLY, 0);

OnError:
        if(input) free(input);
        if(output) free(output);
#if 0
        if(length) free(length);
        if(step) free(step);
        if(padFlg) free(padFlg);
        if(pad) free(pad);
        if(axis) free(axis);
#endif
        if(input_user_addr) vxReleaseTensorAddressing(&input_user_addr);
        if(output_user_addr) vxReleaseTensorAddressing(&output_user_addr);
#if 0
        if(window_user_addr) vxReleaseTensorAddressing(&window_user_addr);
        if(step_user_addr) vxReleaseTensorAddressing(&step_user_addr);
        if(padEnd_user_addr) vxReleaseTensorAddressing(&padEnd_user_addr);
        if(padVal_user_addr) vxReleaseTensorAddressing(&padVal_user_addr);
        if(axis_user_addr) vxReleaseTensorAddressing(&axis_user_addr);
#endif
    }

    return status;
}

vsi_status VX_CALLBACK vxSignalFrameInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    uint32_t paraNum
    )
{
    vsi_status status = VX_SUCCESS;
    // Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
    vx_kernel_execution_parameters_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread
    vx_scalar     scalar[5];
    vx_tensor     input           = (vx_tensor)paramObj[0];
    vx_tensor     output          = (vx_tensor)paramObj[1];

    uint32_t      input_size[DIM_SIZE]   = {0};
    uint32_t      input_dims      = 0;
    uint32_t      output_dims      = 0;
    vsi_nn_type_e inputDataFormat = VSI_NN_TYPE_FLOAT16;
    vsi_nn_type_e outputDataFormat = VSI_NN_TYPE_FLOAT16;
    //vx_uint32 factor = 1;
    //vx_uint32 maxWorkGroupSize = 8;
    uint32_t frame_length, step, pad_end, pad, axis, axis0;
    uint32_t output_channel = 0;

    scalar[0]            = (vx_scalar)paramObj[2];
    scalar[1]            = (vx_scalar)paramObj[3];
    scalar[2]            = (vx_scalar)paramObj[4];
    scalar[3]            = (vx_scalar)paramObj[5];
    scalar[4]            = (vx_scalar)paramObj[6];

    status  = vxQueryTensor(input, VX_TENSOR_DIMS, input_size, sizeof(input_size));
    status |= vxQueryTensor(input, VX_TENSOR_NUM_OF_DIMS, &input_dims, sizeof(input_dims));
    status |= vxQueryTensor(input, VX_TENSOR_DATA_TYPE, &inputDataFormat, sizeof(inputDataFormat));
    status |= vxQueryTensor(output, VX_TENSOR_NUM_OF_DIMS, &output_dims, sizeof(output_dims));
    status |= vxQueryTensor(output, VX_TENSOR_DATA_TYPE, &outputDataFormat, sizeof(outputDataFormat));
    if(VX_SUCCESS != status)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
        return status;
    }

    status = vxCopyScalar(scalar[0], &frame_length, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    status |= vxCopyScalar(scalar[1], &step, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    status |= vxCopyScalar(scalar[2], &pad_end, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    status |= vxCopyScalar(scalar[3], &pad, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    status |= vxCopyScalar(scalar[4], &axis0, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    if (status != VX_SUCCESS)
    {
        VSILOGE("vxCopyScalar failure! at line %d\n", __LINE__);
        return status;
    }

    {
        if(input_dims == 2 && output_dims == 2)
        {
            axis = input_dims - axis0 - 2;
        }
        else
        {
            axis = input_dims - axis0 - 1;
        }
    }

    input_size[2] = (input_dims <= 2)?1:input_size[2];
    //input_size[2] = (input_dims == 4)?(input_size[2] * input_size[3]):input_size[2];

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkOffset[2] = 0;
    if((output_dims == 2)
        || (input_dims == 2 && output_dims == 3 && axis == 1)
        || (input_dims == 3 && axis == 2))
    {
        shaderParam.globalWorkScale[0]  = 1;
        shaderParam.globalWorkScale[1]  = 1;
        shaderParam.globalWorkScale[2]  = 1;
        shaderParam.localWorkSize[0]    = 1;
        shaderParam.localWorkSize[1]    = 1;
#if 0
        if (input_size[1] <= maxWorkGroupSize)
            shaderParam.localWorkSize[1]    = input_size[1];
        else if (getFactor(input_size[1], &factor, 2, maxWorkGroupSize, 8) == VX_SUCCESS)
            shaderParam.localWorkSize[1]    = factor;
        else
            shaderParam.localWorkSize[1]    = 1;
#endif

        shaderParam.localWorkSize[2]    = 1;
        shaderParam.globalWorkSize[0]   = gcmALIGN((1 + shaderParam.globalWorkScale[0] - 1)
            / shaderParam.globalWorkScale[0], shaderParam.localWorkSize[0]);

        shaderParam.globalWorkSize[1]   = gcmALIGN((input_size[1] + shaderParam.globalWorkScale[1] - 1)
            / shaderParam.globalWorkScale[1], shaderParam.localWorkSize[1]);
        //shaderParam.globalWorkSize[1]   = input_size[1];
        shaderParam.globalWorkSize[2]   = gcmALIGN((input_size[2] + shaderParam.globalWorkScale[2] - 1)
            / shaderParam.globalWorkScale[2], shaderParam.localWorkSize[2]);
    }
    else if((input_dims == 2 && output_dims == 3 && axis == 0)
        || (input_dims == 3 && axis == 1))
    {
        int height = (pad_end == 0) ? (input_size[1] - frame_length + 1) : (input_size[1]);
        shaderParam.globalWorkScale[0]  = 8;
        shaderParam.globalWorkScale[1]  = step;
        shaderParam.globalWorkScale[2]  = 1;
        shaderParam.localWorkSize[0]    = 1;
        shaderParam.localWorkSize[1]    = 1;
        shaderParam.localWorkSize[2]    = 1;
        shaderParam.globalWorkSize[0]   = gcmALIGN((input_size[0] + shaderParam.globalWorkScale[0] - 1)
            / shaderParam.globalWorkScale[0], shaderParam.localWorkSize[0]);
        shaderParam.globalWorkSize[1]   = gcmALIGN((height + shaderParam.globalWorkScale[1] - 1)
            / shaderParam.globalWorkScale[1], shaderParam.localWorkSize[1]);
        shaderParam.globalWorkSize[2]   = gcmALIGN((input_size[2] + shaderParam.globalWorkScale[2] - 1)
            / shaderParam.globalWorkScale[2], shaderParam.localWorkSize[2]);

        output_channel = (pad_end == 0) ? ((input_size[1] - frame_length) / step  + 1) : ((input_size[1] + step - 1) / step);
    }
    else if(input_dims == 3 && axis == 0)
    {
        int channel = (pad_end == 0) ? (input_size[2] - frame_length + 1) : (input_size[2]);
        shaderParam.globalWorkScale[0]  = 8;
        shaderParam.globalWorkScale[1]  = 1;
        shaderParam.globalWorkScale[2]  = step;
        shaderParam.localWorkSize[0]    = 1;
        shaderParam.localWorkSize[1]    = 1;
        shaderParam.localWorkSize[2]    = 1;
        shaderParam.globalWorkSize[0]   = gcmALIGN((input_size[0] + shaderParam.globalWorkScale[0] - 1)
            / shaderParam.globalWorkScale[0], shaderParam.localWorkSize[0]);
        shaderParam.globalWorkSize[1]   = gcmALIGN((input_size[1] + shaderParam.globalWorkScale[1] - 1)
            / shaderParam.globalWorkScale[1], shaderParam.localWorkSize[1]);
        shaderParam.globalWorkSize[2]   = gcmALIGN((channel + shaderParam.globalWorkScale[2] - 1)
            / shaderParam.globalWorkScale[2], shaderParam.localWorkSize[2]);
    }

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));
    if(status < 0)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
    }
    {
        status |= vxSetNodeUniform(nodObj, "input_width", 1, &input_size[0]);
        status |= vxSetNodeUniform(nodObj, "input_height", 1, &input_size[1]);
        status |= vxSetNodeUniform(nodObj, "input_channel", 1, &input_size[2]);
        status |= vxSetNodeUniform(nodObj, "output_channel", 1, &output_channel);
        if(status < 0)
        {
            VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
        }
    }
    return status;
}
static vx_param_description_t vxSignalFrameKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED}
};
#ifdef __cplusplus
extern "C" {
#endif
vx_kernel_description_t vxSignalFrameKernelInfo =
{
    VX_KERNEL_ENUM_SIGNALFRAME,
    VX_KERNEL_NAME_SIGNALFRAME_WIDTH,
    NULL,
    vxSignalFrameKernelParam,
    (sizeof(vxSignalFrameKernelParam) / sizeof(vxSignalFrameKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxSignalFrameInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxSignalFrameKernelInfo_height =
{
    VX_KERNEL_ENUM_SIGNALFRAME,
    VX_KERNEL_NAME_SIGNALFRAME_HEIGHT,
    NULL,
    vxSignalFrameKernelParam,
    (sizeof(vxSignalFrameKernelParam) / sizeof(vxSignalFrameKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxSignalFrameInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxSignalFrameKernelInfo_channel =
{
    VX_KERNEL_ENUM_SIGNALFRAME,
    VX_KERNEL_NAME_SIGNALFRAME_CHANNEL,
    NULL,
    vxSignalFrameKernelParam,
    (sizeof(vxSignalFrameKernelParam) / sizeof(vxSignalFrameKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxSignalFrameInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxSignalFrameKernelInfo_8bit =
{
    VX_KERNEL_ENUM_SIGNALFRAME,
    VX_KERNEL_NAME_SIGNALFRAME_WIDTH_8BITS,
    NULL,
    vxSignalFrameKernelParam,
    (sizeof(vxSignalFrameKernelParam) / sizeof(vxSignalFrameKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxSignalFrameInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxSignalFrameKernelInfo_height_8bit =
{
    VX_KERNEL_ENUM_SIGNALFRAME,
    VX_KERNEL_NAME_SIGNALFRAME_HEIGHT_8BITS,
    NULL,
    vxSignalFrameKernelParam,
    (sizeof(vxSignalFrameKernelParam) / sizeof(vxSignalFrameKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxSignalFrameInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxSignalFrameKernelInfo_channel_8bit =
{
    VX_KERNEL_ENUM_SIGNALFRAME,
    VX_KERNEL_NAME_SIGNALFRAME_CHANNEL_8BITS,
    NULL,
    vxSignalFrameKernelParam,
    (sizeof(vxSignalFrameKernelParam) / sizeof(vxSignalFrameKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxSignalFrameInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxSignalFrameKernelInfo_CPU =
{
    VX_KERNEL_ENUM_SIGNALFRAME,
    VX_KERNEL_NAME_SIGNALFRAME_WIDTH,
    vxSignalFrameKernel,
    vxSignalFrameKernelParam,
    (sizeof(vxSignalFrameKernelParam) / sizeof(vxSignalFrameKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_SIGNALFRAME_list[] =
{
    &vxSignalFrameKernelInfo_CPU,
    &vxSignalFrameKernelInfo,
    &vxSignalFrameKernelInfo_height,
    &vxSignalFrameKernelInfo_channel,
    &vxSignalFrameKernelInfo_8bit,
    &vxSignalFrameKernelInfo_height_8bit,
    &vxSignalFrameKernelInfo_channel_8bit,
    NULL
};
#ifdef __cplusplus
}
#endif
