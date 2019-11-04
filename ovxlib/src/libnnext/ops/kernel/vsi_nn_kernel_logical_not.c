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

#define _VX_KERNEL_ID           (VX_KERNEL_ENUM_LOGICAL_NOT)
#define _VX_KERNEL_NAME         (VX_KERNEL_NAME_LOGICAL_NOT_INT8)
#define _VX_KERNEL_FUNC_KERNEL  (vxLogical_notKernel)

void myLogicalNotFunc
    (
    void* imgIn,
    void* imgOut,
    uint32_t width,
    uint32_t height,
    uint32_t channel,
    uint32_t batch,
    vsi_nn_type_e type
    )
{
    uint32_t k;
    uint32_t iter = batch * channel * height * width;

    if(type == VSI_NN_TYPE_INT16 || type == VSI_NN_TYPE_FLOAT16)
    {
        int16_t* tmpIn = (int16_t*)imgIn;
        int16_t* tmpOut = (int16_t*)imgOut;

        for(k = 0; k < iter; k++)
        {
            tmpOut[k] = !tmpIn[k];
        }
    }
    else if(type == VSI_NN_TYPE_INT8)
    {
        int8_t* tmpIn = (int8_t*)imgIn;
        int8_t* tmpOut = (int8_t*)imgOut;

        for(k = 0; k < iter; k++)
        {
            tmpOut[k] = !tmpIn[k];
        }
    }
    else if(type == VSI_NN_TYPE_UINT8)
    {
        uint8_t* tmpIn = (uint8_t*)imgIn;
        uint8_t* tmpOut = (uint8_t*)imgOut;

        for(k = 0; k < iter; k++)
        {
            tmpOut[k] = !tmpIn[k];
        }
    }

    return;
}

static vsi_status VX_CALLBACK vxLogical_notKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
    vsi_status status = VX_ERROR_INVALID_PARAMETERS;

    if(paramNum == 2)
    {
        vx_context context = NULL;
        // tensor
        vx_tensor imgObj[2] = { NULL };
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

        uint32_t input_size[DIM_SIZE] = {0}, output_size[DIM_SIZE] = {0};
        vsi_nn_tensor_attr_t in_attr, out_attr;

        vsi_nn_type_e inputFormat = VSI_NN_TYPE_FLOAT16, outputFormat = VSI_NN_TYPE_FLOAT16;
        uint32_t input_dims = 0, output_dims = 0;

        status = VX_SUCCESS;

        memset(&in_attr, 0x0, sizeof(vsi_nn_tensor_attr_t));
        memset(&out_attr, 0x0, sizeof(vsi_nn_tensor_attr_t));

        imgObj[0] = (vx_tensor)paramObj[0];
        imgObj[1] = (vx_tensor)paramObj[1];  //output
        context = vxGetContext((vx_reference)node);
        if (context == NULL)
        {
            VSILOGE("vxGetContext failure! at line %d\n", __LINE__);
            goto final;
        }
        //input
        status = vxQueryTensor(imgObj[0], VX_TENSOR_NUM_OF_DIMS, &input_dims, sizeof(input_dims));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor input_dims failure! at line %d\n", __LINE__);
            goto final;
        }
        status = vxQueryTensor(imgObj[0], VX_TENSOR_DATA_TYPE, &inputFormat, sizeof(inputFormat));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor inputFormat failure! at line %d\n", __LINE__);
            goto final;
        }
        status = vxQueryTensor(imgObj[0], VX_TENSOR_DIMS, input_size, sizeof(input_size));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor input_size failure! at line %d\n", __LINE__);
            goto final;
        }

        //output
        status  = vxQueryTensor(imgObj[1], VX_TENSOR_DATA_TYPE, &outputFormat, sizeof(outputFormat));
        status |= vxQueryTensor(imgObj[1], VX_TENSOR_NUM_OF_DIMS, &output_dims, sizeof(output_dims));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor outputFormat failure! at line %d\n", __LINE__);
            goto final;
        }
        status = vxQueryTensor(imgObj[1], VX_TENSOR_DIMS, output_size, sizeof(output_size));
        if (status != VX_SUCCESS)
        {
            VSILOGE("vxQueryTensor output_size failure! at line %d\n", __LINE__);
            goto final;
        }

        input_size[2] = (input_dims <= 2)?1:input_size[2];
        input_size[3] = (input_dims <= 3)?1:input_size[3];

#if INPUT_FP16
        input  = (int16_t*)malloc(input_size[0]*input_size[1]*input_size[2]*sizeof(int16_t));
#else
        //input  = (uint8_t*)malloc(input_size[0]*input_size[1]*input_size[2]*vsi_nn_GetTypeBytes(inputFormat));
#endif
#if OUTPUT_FP16
        output = (int16_t*)malloc(output_size[0]*output_size[1]*output_size[2]*sizeof(int16_t));
#else
        output = (uint8_t*)malloc(output_size[0]*output_size[1]*output_size[2]*vsi_nn_GetTypeBytes(outputFormat));
#endif
        status = vsi_nn_vxGetTensorAttr(imgObj[0], &in_attr);
        status |= vsi_nn_vxGetTensorAttr(imgObj[1], &out_attr);
        if (status != VX_SUCCESS)
        {
            VSILOGE("vsi_nn_vxGetTensorAttr failure! at line %d\n", __LINE__);
            goto final;
        }
        input = vsi_nn_vxCopyTensorToData(context, imgObj[0], &in_attr);

        // Call C Prototype
        myLogicalNotFunc(input, output, input_size[0],
            input_size[1], input_size[2], input_size[3], inputFormat);

        //output tensor
        status = vsi_nn_vxCopyDataToTensor(context, imgObj[1], &out_attr, output);
        if (status != VX_SUCCESS)
        {
            VSILOGE("vsi_nn_vxCopyDataToTensor failure! at line %d\n", __LINE__);
            goto final;
        }

final:
        if(input) free(input);
        if(output) free(output);
    }

    return status;
} /* _VX_KERNEL_FUNC_KERNEL() */

static vx_param_description_t vxLogical_notKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
};

vx_status VX_CALLBACK vxLogical_notInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    vx_uint32 paraNum
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

    vx_tensor     input           = (vx_tensor)paramObj[0];

    uint32_t      input_size[DIM_SIZE]   = {0};
    uint32_t      input_dims      = 0;
    uint32_t      zAx             = 1;
    vsi_nn_type_e inputDataFormat = VSI_NN_TYPE_FLOAT16;

    status  = vxQueryTensor(input, VX_TENSOR_DIMS, input_size, sizeof(input_size));
    status |= vxQueryTensor(input, VX_TENSOR_NUM_OF_DIMS, &input_dims, sizeof(input_dims));
    status |= vxQueryTensor(input, VX_TENSOR_DATA_TYPE, &inputDataFormat, sizeof(inputDataFormat));
    if(VX_SUCCESS != status)
    {
        VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
        return status;
    }
    if(input_dims == 4)
        zAx = input_size[3] * input_size[2];
    else if(input_dims == 3)
        zAx = input_size[2];

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkOffset[2] = 0;
    shaderParam.globalWorkScale[0]  = 8;
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.globalWorkScale[2]  = 1;
    shaderParam.localWorkSize[0]    = 4;
    shaderParam.localWorkSize[1]    = 2;
    shaderParam.localWorkSize[2]    = 1;
    shaderParam.globalWorkSize[0]   = gcmALIGN((input_size[0] + shaderParam.globalWorkScale[0] - 1)
        / shaderParam.globalWorkScale[0], shaderParam.localWorkSize[0]);
    shaderParam.globalWorkSize[1]   = gcmALIGN((input_size[1] + shaderParam.globalWorkScale[1] - 1)
        / shaderParam.globalWorkScale[1], shaderParam.localWorkSize[1]);
    shaderParam.globalWorkSize[2]   = gcmALIGN((zAx + shaderParam.globalWorkScale[2] - 1)
        / shaderParam.globalWorkScale[2], shaderParam.localWorkSize[2]);

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
vx_kernel_description_t vxLogical_not_CPU =
{
    _VX_KERNEL_ID,
    _VX_KERNEL_NAME,
    _VX_KERNEL_FUNC_KERNEL,
    vxLogical_notKernelParam,
    _cnt_of_array( vxLogical_notKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxLogical_not_i8 =
{
    _VX_KERNEL_ID,
    VX_KERNEL_NAME_LOGICAL_NOT_INT8,
    NULL,
    vxLogical_notKernelParam,
    _cnt_of_array( vxLogical_notKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxLogical_notInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxLogical_not_i16 =
{
    _VX_KERNEL_ID,
    VX_KERNEL_NAME_LOGICAL_NOT_INT16,
    NULL,
    vxLogical_notKernelParam,
    _cnt_of_array( vxLogical_notKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxLogical_notInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxLogical_not_u8 =
{
    _VX_KERNEL_ID,
    VX_KERNEL_NAME_LOGICAL_NOT_UINT8,
    NULL,
    vxLogical_notKernelParam,
    _cnt_of_array( vxLogical_notKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxLogical_notInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_LOGICAL_NOT_list[] =
{
    &vxLogical_not_CPU,
    &vxLogical_not_i8,
    &vxLogical_not_i16,
    &vxLogical_not_u8,
    NULL
};
#ifdef __cplusplus
}
#endif
