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
#include <float.h>

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

#define _VX_KERNEL_ID           (VX_KERNEL_ENUM_RANDOM_MULTINOMIAL)
#define _VX_KERNEL_NAME         (VX_KERNEL_NAME_RANDOM_MULTINOMIAL)
#define _VX_KERNEL_FUNC_KERNEL  (vxRandom_multinomialKernel)

#undef MAX
#define MAX(a,b)    ((a) > (b) ? (a) : (b))
#undef MIN
#define MIN(a,b)    ((a) < (b) ? (a) : (b))

static int upper_bound(float* a, int n, float x) {
    int l = 0;
    int h = n;
    while (l < h) {
        int mid = (l + h) / 2;
        if (x >= a[mid]) {
            l = mid + 1;
        } else {
            h = mid;
        }
    }
    return l;
}

static vsi_status VX_CALLBACK vxRandom_multinomialKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
#define ARG_NUM            (1)
#define TENSOR_NUM_INPUT (2)
#define TENSOR_NUM_OUTPUT (1)
#define TENSOR_NUM (TENSOR_NUM_INPUT+TENSOR_NUM_OUTPUT)

    vsi_status status = VSI_FAILURE;
    vx_context context = NULL;
    vx_tensor input[TENSOR_NUM_INPUT] = {0};
    vx_tensor output[TENSOR_NUM_OUTPUT] = {0};
    float *f32_in_buffer[TENSOR_NUM_INPUT] = {0};
    int32_t* int32_in_buffer[TENSOR_NUM_INPUT] = {0};
    int32_t *int32_out_buffer[TENSOR_NUM_OUTPUT] = {0};
    vsi_nn_tensor_attr_t in_attr[TENSOR_NUM_INPUT];
    vsi_nn_tensor_attr_t out_attr[TENSOR_NUM_OUTPUT];
    uint32_t in_elements[TENSOR_NUM_INPUT] = {0};
    uint32_t out_elements[TENSOR_NUM_OUTPUT]= {0};

    int32_t sample_num;

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
        if (i == 1)
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
        int32_out_buffer[i]= (int32_t *)malloc(out_elements[i] * sizeof(int32_t));
        memset(int32_out_buffer[i], 0, out_elements[i] * sizeof(int32_t));
    }
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM], &(sample_num),
        VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    /* TODO: Add CPU kernel implement */
    {
        uint32_t n, c;
        uint32_t batch = in_attr[0].size[1];
        uint32_t class_size = in_attr[0].size[0];
        float *cdf = (float *)malloc(class_size * sizeof(float));
        uint32_t *random_integer = (uint32_t *)malloc(out_elements[0] * sizeof(uint32_t));
        float *random_float = (float *)malloc(out_elements[0] * sizeof(float));
        vsi_nn_random_init_for_philox_4x32_10((uint32_t)(int32_in_buffer[1][0]),
            (uint32_t)(int32_in_buffer[1][1]));
        vsi_nn_random_generate_by_philox_4x32_10(random_integer, out_elements[0]);
        vsi_nn_random_uniform_transform(random_integer,
            random_float, out_elements[0]);
        for(n = 0; n < batch; n++)
        {
            float batch_max = -FLT_MAX;
            float total = 0;
            for(c = 0; c < class_size; c++)
            {
                uint32_t index = n * class_size + c;
                batch_max = MAX(batch_max, f32_in_buffer[0][index]);
            }
            for(c = 0; c < class_size; c++)
            {
                uint32_t index = n * class_size + c;
                total += (float)(exp(f32_in_buffer[0][index] - batch_max));
                cdf[c] = total;
            }

            for(c = 0; c < (uint32_t)sample_num; c++)
            {
                uint32_t index = n * sample_num + c;
                float target = random_float[index] * total;
                uint32_t out_class = upper_bound(cdf, class_size, target);
                int32_out_buffer[0][index] = out_class;
            }
        }

        if (cdf) free(cdf);
        if (random_integer) free(random_integer);
        if (random_float) free(random_float);
    }

    /* save data */
    for(i = 0; i < TENSOR_NUM_OUTPUT; i++)
    {
        vsi_nn_vxCopyDataToTensor(context, output[i], &out_attr[i],
            (uint8_t *)(int32_out_buffer[i]));
    }

final:
    for (i = 0; i < TENSOR_NUM_INPUT; i++)
    {
        if (f32_in_buffer[i]) free(f32_in_buffer[i]);
        if (int32_in_buffer[i]) free(int32_in_buffer[i]);
    }
    for(i = 0; i < TENSOR_NUM_OUTPUT; i++)
    {
        if (int32_out_buffer[i]) free(int32_out_buffer[i]);
    }
    return status;
} /* _VX_KERNEL_FUNC_KERNEL() */

static vx_param_description_t vxRandom_multinomialKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

static vx_param_description_t vxRandom_generateKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
};

static vx_param_description_t vxRandom_sumKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED},
};

static vx_param_description_t vxRandom_multinomial2KernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

vx_status VX_CALLBACK vxRandom_generateInitializer
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

    vx_tensor     output          = (vx_tensor)paramObj[1];
    uint32_t      output_size[DIM_SIZE]  = {0};
    uint32_t      stride = 0;
    uint32_t      iter = 8;
    uint32_t      w = 0;
    float         rand_max = (float)(pow(2.0,32));
    float         re_rand_max = 1 / rand_max;
    vx_uint32 i = 0;
    vsi_nn_tensor_attr_t attr;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    status = vsi_nn_vxGetTensorAttr(output, &attr);
    if (status != VX_SUCCESS)
    {
        VSILOGE("vsi_nn_vxGetTensorAttr  failure! at line %d\n", __LINE__);
        return status;
    }
    for (i = 0; i < attr.dim_num; i++)
    {
        output_size[i] = attr.size[i];
    }

    if(output_size[0] <= 4)
    {
        iter = 1;
        w = 1;
    }
    else if(output_size[0] <= 32)
    {
        iter = (output_size[0] + 3) / 4;
        w = 1;
    }
    else
    {
        w = (output_size[0] + 31) / 32;
    }
    stride = iter * 4;

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkOffset[2] = 0;
    shaderParam.globalWorkScale[0]  = 1;
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.globalWorkScale[2]  = 1;
    shaderParam.globalWorkSize[0]   = gcmALIGN((w + shaderParam.globalWorkScale[0] - 1)
        / shaderParam.globalWorkScale[0], 4);
    shaderParam.globalWorkSize[1]   = output_size[1];
    shaderParam.globalWorkSize[2]   = 1;

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));
    if(status < 0)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
    }

    status |= vxSetNodeUniform(nodObj, "stride", 1, &stride);
    status |= vxSetNodeUniform(nodObj, "iter", 1, &iter);
    status |= vxSetNodeUniform(nodObj, "re_rand_max", 1, &re_rand_max);
    if(status < 0)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
    }

    return status;
}

vx_status VX_CALLBACK vxRandom_sumInitializer
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

    vx_tensor     input           = (vx_tensor)paramObj[0];
    vsi_nn_type_e inputDataFormat = VSI_NN_TYPE_FLOAT16;
    uint32_t      input_size[DIM_SIZE]   = {1, 1, 1, 1};
    uint32_t      class_size = 0, batch = 0;
    uint32_t      class_max_stride = 0;
    uint32_t      class_max_iter = 0;
    vx_uint32 i = 0;
    vsi_nn_tensor_attr_t attr;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    status = vsi_nn_vxGetTensorAttr(input, &attr);
    if (status != VX_SUCCESS)
    {
        VSILOGE("vsi_nn_vxGetTensorAttr  failure! at line %d\n", __LINE__);
        return status;
    }
    for (i = 0; i < attr.dim_num; i++)
    {
        input_size[i] = attr.size[i];
    }
    inputDataFormat = attr.dtype.vx_type;

    class_size = input_size[0];
    batch = input_size[1];
    if(inputDataFormat == VSI_NN_TYPE_FLOAT32)
    {
        class_max_iter = (class_size + 3) >> 2;
        class_max_stride = class_max_iter << 2;
    }
    else
    {
        class_max_iter = (class_size + 7) >> 3;
        class_max_stride = class_max_iter << 3;
    }

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkOffset[2] = 0;
    shaderParam.globalWorkScale[0]  = 1;
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.globalWorkScale[2]  = 1;
    shaderParam.globalWorkSize[0]   =  1;
    shaderParam.globalWorkSize[1]   = batch;
    shaderParam.globalWorkSize[2]   = 1;
    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));
    if(status < 0)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
    }

    {
        vx_uint32 uniHorzSubMaxFp16_2x8[16] = {
            0x99999999, // TCfg
            0x44444444, // ASelt
            0x03020100, 0x07060504, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
        };
        vx_uint32 uniConvertFstFp16Fp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };
        vx_uint32 uniConvertSecFp16Fp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };

        status |= vxSetNodeUniform(nodObj, "class_max_iter", 1, &class_max_iter);
        status |= vxSetNodeUniform(nodObj, "class_max_stride", 1, &class_max_stride);
        status |= vxSetNodeUniform(nodObj, "uniConvertFstFp16Fp32_4x4", 1, uniConvertFstFp16Fp32_4x4);
        status |= vxSetNodeUniform(nodObj, "uniConvertSecFp16Fp32_4x4", 1, uniConvertSecFp16Fp32_4x4);
        status |= vxSetNodeUniform(nodObj, "uniHorzSubMaxFp16_2x8", 1, uniHorzSubMaxFp16_2x8);
        if(status < 0)
        {
            VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
        }
    }

    return status;
}

vx_status VX_CALLBACK vxRandom_multinomialInitializer
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

    vx_tensor     input           = (vx_tensor)paramObj[0];
    uint32_t      input_size[DIM_SIZE]   = {1, 1, 1, 1};
    uint32_t      sample_num = 0;
    uint32_t      batch = 0;
    vx_uint32 i = 0;
    vsi_nn_tensor_attr_t attr;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    status = vsi_nn_vxGetTensorAttr(input, &attr);
    if (status != VX_SUCCESS)
    {
        VSILOGE("vsi_nn_vxGetTensorAttr  failure! at line %d\n", __LINE__);
        return status;
    }
    for (i = 0; i < attr.dim_num; i++)
    {
        input_size[i] = attr.size[i];
    }

    sample_num = input_size[0];
    batch = input_size[1];

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkOffset[2] = 0;
    shaderParam.globalWorkScale[0]  = 4;
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.globalWorkScale[2]  = 1;
    shaderParam.globalWorkSize[0]   = gcmALIGN((sample_num + shaderParam.globalWorkScale[0] - 1)
        / shaderParam.globalWorkScale[0], 4);
    shaderParam.globalWorkSize[1]   = batch;
    shaderParam.globalWorkSize[2]   = 1;
    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));
    if(status < 0)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
    }

    return status;
}


#ifdef __cplusplus
extern "C" {
#endif
vx_kernel_description_t vxRandom_multinomial_CPU =
{
    _VX_KERNEL_ID,
    _VX_KERNEL_NAME,
    _VX_KERNEL_FUNC_KERNEL,
    vxRandom_multinomialKernelParam,
    _cnt_of_array( vxRandom_multinomialKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxRandom_generate_VX =
{
    VX_KERNEL_ENUM_RANDOM_MULTINOMIAL,
    VX_KERNEL_NAME_RANDOM_GENERATE,
    NULL,
    vxRandom_generateKernelParam,
    _cnt_of_array( vxRandom_generateKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxRandom_generateInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxRandom_sum_fp16 =
{
    VX_KERNEL_ENUM_RANDOM_MULTINOMIAL,
    VX_KERNEL_NAME_RANDOM_SUM_FP16,
    NULL,
    vxRandom_sumKernelParam,
    _cnt_of_array( vxRandom_sumKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxRandom_sumInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxRandom_sum_fp32 =
{
    VX_KERNEL_ENUM_RANDOM_MULTINOMIAL,
    VX_KERNEL_NAME_RANDOM_SUM_FP32,
    NULL,
    vxRandom_sumKernelParam,
    _cnt_of_array( vxRandom_sumKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxRandom_sumInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxRandom_multinomial =
{
    VX_KERNEL_ENUM_RANDOM_MULTINOMIAL,
    VX_KERNEL_NAME_RANDOM_MULTINOMIAL,
    NULL,
    vxRandom_multinomial2KernelParam,
    _cnt_of_array( vxRandom_multinomial2KernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxRandom_multinomialInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_RANDOM_MULTINOMIAL_list[] =
{
    &vxRandom_multinomial_CPU,
    &vxRandom_generate_VX,
    &vxRandom_sum_fp16,
    &vxRandom_sum_fp32,
    &vxRandom_multinomial,
    NULL
};
#ifdef __cplusplus
}
#endif
