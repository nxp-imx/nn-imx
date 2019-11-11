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
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_math.h"
#include "client/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"

#define _VX_KERNEL_ID           (VX_KERNEL_ENUM_SELECT)
#define _VX_KERNEL_FUNC_KERNEL  (vxSelectKernel)

void mySelectFunc
    (
    void* imgCond,
    void* imgIn,
    void* imgIn1,
    void* imgOut,
    uint32_t input_dim,
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
        int16_t* tmpIn1 = (int16_t*)imgIn1;
        int16_t* tmpOut = (int16_t*)imgOut;
        int16_t data0, data1;
        uint8_t* tmpCond = (uint8_t*)imgCond;
        uint8_t cond = 0;

        for(k = 0; k < iter; k++)
        {
            cond = tmpCond[k];
            data0 = tmpIn[k];
            data1 = tmpIn1[k];
            tmpOut[k] = cond ? data0 : data1;
        }
    }
    else if(type == VSI_NN_TYPE_INT8)
    {
        int8_t* tmpIn = (int8_t*)imgIn;
        int8_t* tmpIn1 = (int8_t*)imgIn1;
        int8_t* tmpOut = (int8_t*)imgOut;
        int8_t data0, data1;
        uint8_t* tmpCond = (uint8_t*)imgCond;
        uint8_t cond = 0;

        for(k = 0; k < iter; k++)
        {
            cond = tmpCond[k];
            data0 = tmpIn[k];
            data1 = tmpIn1[k];
            tmpOut[k] = cond ? data0 : data1;
        }
    }
    else if(type == VSI_NN_TYPE_UINT8)
    {
        uint8_t* tmpIn = (uint8_t*)imgIn;
        uint8_t* tmpIn1 = (uint8_t*)imgIn1;
        uint8_t* tmpOut = (uint8_t*)imgOut;
        uint8_t data0, data1;
        uint8_t* tmpCond = (uint8_t*)imgCond;
        uint8_t cond = 0;

        for(k = 0; k < iter; k++)
        {
            cond = tmpCond[k];
            data0 = tmpIn[k];
            data1 = tmpIn1[k];
            tmpOut[k] = cond ? data0 : data1;
        }
    }

    return;
}

static vsi_status VX_CALLBACK vxSelectKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
    vsi_status status = VX_ERROR_INVALID_PARAMETERS;

    if(paramNum == 4)
    {
        vx_context context = NULL;
        // tensor
        vx_tensor imgObj[4] = { NULL };
#if INPUT_FP16
        int16_t *input = NULL;
#else
        uint8_t *condition = NULL;
        uint8_t *input = NULL;
        uint8_t *input1 = NULL;
#endif
#if OUTPUT_FP16
        int16_t *output = NULL;
#else
        uint8_t *output = NULL;
#endif

        uint32_t input_size[DIM_SIZE] = {1, 1, 1, 1}, output_size[DIM_SIZE] = {1, 1, 1, 1};
        vsi_nn_tensor_attr_t in_attr, in_attr1, con_attr, out_attr;

        vsi_nn_type_e inputFormat = VSI_NN_TYPE_FLOAT16, outputFormat = VSI_NN_TYPE_FLOAT16;
        vsi_nn_type_e condFormat = VSI_NN_TYPE_INT8;
        uint32_t input_dims = 0, output_dims = 0, tmpDim = 0, i = 0;

        memset(&in_attr, 0x0, sizeof(vsi_nn_tensor_attr_t));
        memset(&in_attr1, 0x0, sizeof(vsi_nn_tensor_attr_t));
        memset(&con_attr, 0x0, sizeof(vsi_nn_tensor_attr_t));
        memset(&out_attr, 0x0, sizeof(vsi_nn_tensor_attr_t));

        status  = vsi_nn_vxGetTensorAttr(imgObj[0], &con_attr);
        status |= vsi_nn_vxGetTensorAttr(imgObj[1], &in_attr);
        status |= vsi_nn_vxGetTensorAttr(imgObj[2], &in_attr1);
        status |= vsi_nn_vxGetTensorAttr(imgObj[3], &out_attr);
        if (status != VX_SUCCESS)
        {
            VSILOGE("vsi_nn_vxGetTensorAttr failure! at line %d\n", __LINE__);
            goto OnError;
        }

        imgObj[0] = (vx_tensor)paramObj[0];
        imgObj[1] = (vx_tensor)paramObj[1];  //output
        imgObj[2] = (vx_tensor)paramObj[2];
        imgObj[3] = (vx_tensor)paramObj[3];
        context = vxGetContext((vx_reference)node);
        if (context == NULL)
        {
            VSILOGE("vxGetContext failure! at line %d\n", __LINE__);
            goto OnError;
        }

        //condition
        condFormat = con_attr.dtype.vx_type;
        if(condFormat != VSI_NN_TYPE_BOOL8)
        {
            VSILOGE("Condition format failure! at line %d\n", __LINE__);
            goto OnError;
        }

        //input
        input_dims  = in_attr.dim_num;
        inputFormat = in_attr.dtype.vx_type;
        for (i = 0; i < input_dims; i++)
        {
            input_size[i] = in_attr.size[i];
        }
        //output
        outputFormat = out_attr.dtype.vx_type;
        output_dims  = out_attr.dim_num;
        for (i = 0; i < output_dims; i++)
        {
            output_size[i] = out_attr.size[i];
        }

        input_size[2] = (input_dims <= 2)?1:input_size[2];
        input_size[3] = (input_dims <= 3)?1:input_size[3];


#if INPUT_FP16
        input  = (int16_t*)malloc(input_size[0]*input_size[1]*input_size[2]*sizeof(int16_t));
#else
        //condition = (uint8_t*)malloc(input_size[0]*input_size[1]*input_size[2]*vsi_nn_GetTypeBytes(condFormat));
        //input  = (uint8_t*)malloc(input_size[0]*input_size[1]*input_size[2]*vsi_nn_GetTypeBytes(inputFormat));
        //input1  = (uint8_t*)malloc(input_size[0]*input_size[1]*input_size[2]*vsi_nn_GetTypeBytes(inputFormat));
#endif
#if OUTPUT_FP16
        output = (int16_t*)malloc(output_size[0]*output_size[1]*output_size[2]*sizeof(int16_t));
#else
        output = (uint8_t*)malloc(output_size[0]*output_size[1]*output_size[2]*vsi_nn_GetTypeBytes(outputFormat));
#endif

        condition = vsi_nn_vxCopyTensorToData(context, imgObj[0], &in_attr);
        input = vsi_nn_vxCopyTensorToData(context, imgObj[1], &in_attr);
        input1 = vsi_nn_vxCopyTensorToData(context, imgObj[2], &in_attr1);

        // Call C Prototype
        mySelectFunc(condition, input, input1, output, tmpDim, input_size[0],
            input_size[1], input_size[2], input_size[3], inputFormat);

        //output tensor
        status = vsi_nn_vxCopyDataToTensor(context, imgObj[3], &out_attr, output);
        if (status != VX_SUCCESS)
        {
            VSILOGE("vsi_nn_vxCopyDataToTensor failure! at line %d\n", __LINE__);
            goto OnError;
        }
OnError:
        if(condition) free(condition);
        if(input) free(input);
        if(input1) free(input1);
        if(output) free(output);
    }

    return status;
} /* _VX_KERNEL_FUNC_KERNEL() */

static vx_param_description_t vxSelectKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
};

vx_status VX_CALLBACK vxSelectInitializer
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

    vx_tensor     input0          = (vx_tensor)paramObj[1];
    vx_tensor     output          = (vx_tensor)paramObj[3];

    uint32_t      input_size[DIM_SIZE]   = {1, 1, 1, 1};
    uint32_t      input_dims      = 0;
    uint32_t      zAx             = 1;
    vsi_nn_type_e inputDataFormat = VSI_NN_TYPE_FLOAT16;
    //vx_uint32 factor = 1;
    //vx_uint32 maxWorkGroupSize = 8;
    vx_uint32  i        = 0;
    vsi_nn_tensor_attr_t attr[2];
    memset(&attr[0], 0, sizeof(vsi_nn_tensor_attr_t));
    memset(&attr[1], 0, sizeof(vsi_nn_tensor_attr_t));
    status  = vsi_nn_vxGetTensorAttr(input0, &attr[0]);
    status |= vsi_nn_vxGetTensorAttr(output, &attr[1]);
    if (status != VX_SUCCESS)
    {
        VSILOGE("vsi_nn_vxGetTensorAttr  failure! at line %d\n", __LINE__);
        return status;
    }

    input_dims       = attr[0].dim_num;
    for (i = 0; i < input_dims; i++)
    {
        input_size[i] = attr[0].size[i];
    }
    inputDataFormat   = attr[0].dtype.vx_type;
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
    if(inputDataFormat == VSI_NN_TYPE_INT8 || inputDataFormat == VSI_NN_TYPE_UINT8)
    {
        shaderParam.globalWorkScale[0]  = 16;
        shaderParam.globalWorkSize[0]   = gcmALIGN((input_size[0] + shaderParam.globalWorkScale[0] - 1)
        / shaderParam.globalWorkScale[0], shaderParam.localWorkSize[0]);
    }

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
vx_kernel_description_t vxSelect_CPU =
{
    _VX_KERNEL_ID,
    VX_KERNEL_NAME_SELECT_UINT8,
    _VX_KERNEL_FUNC_KERNEL,
    vxSelectKernelParam,
    _cnt_of_array( vxSelectKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxSelect_Uint8 =
{
    _VX_KERNEL_ID,
    VX_KERNEL_NAME_SELECT_UINT8,
    NULL,
    vxSelectKernelParam,
    _cnt_of_array( vxSelectKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxSelectInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxSelect_Bool_Int8 =
{
    _VX_KERNEL_ID,
    VX_KERNEL_NAME_SELECT_BOOL_INT8,
    NULL,
    vxSelectKernelParam,
    _cnt_of_array( vxSelectKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxSelectInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxSelect_Bool_Int16 =
{
    _VX_KERNEL_ID,
    VX_KERNEL_NAME_SELECT_BOOL_INT16,
    NULL,
    vxSelectKernelParam,
    _cnt_of_array( vxSelectKernelParam ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxSelectInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_SELECT_list[] =
{
    &vxSelect_CPU,
    &vxSelect_Uint8,
    &vxSelect_Bool_Int8,
    &vxSelect_Bool_Int16,
    NULL
};
#ifdef __cplusplus
}
#endif
