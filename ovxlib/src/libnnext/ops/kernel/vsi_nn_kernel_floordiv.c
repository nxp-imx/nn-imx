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

static uint32_t getExpandTensorOffset(uint32_t index, uint32_t num_of_dims, uint32_t * in_dims,
                                       uint32_t *strides, uint32_t * out_dims)
{
    uint32_t offset = 0;
    uint32_t i;

    for(i = 0; i < num_of_dims; i++)
    {
        if(in_dims[i] == out_dims[i])
            offset += strides[i] * (index % out_dims[i]);

        index /= out_dims[i];
    }

    return offset;
}


vsi_status VX_CALLBACK vxFloorDivKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
    vsi_status status = VX_ERROR_INVALID_PARAMETERS;

    if(paramNum == 3)
    {
        vx_context context = NULL;
        // tensor
        vx_tensor imgObj[3] = { NULL };
        uint8_t *input = NULL;
        uint8_t *input1 = NULL;
        uint8_t *output = NULL;

        vsi_nn_tensor_attr_t in_attr, in_attr1, out_attr;
        uint32_t    stride_size[3][VSI_NN_MAX_DIM_NUM];
        vx_tensor_addressing user_addr[3]  = {NULL};

        int32_t elementCount = 0;
        int32_t i = 0;

        status = VX_SUCCESS;

        memset(&in_attr, 0x0, sizeof(vsi_nn_tensor_attr_t));
        memset(&in_attr1, 0x0, sizeof(vsi_nn_tensor_attr_t));
        memset(&out_attr, 0x0, sizeof(vsi_nn_tensor_attr_t));

        imgObj[0] = (vx_tensor)paramObj[0];
        imgObj[1] = (vx_tensor)paramObj[1];
        imgObj[2] = (vx_tensor)paramObj[2];  //output
        context = vxGetContext((vx_reference)node);
        if (context == NULL)
        {
            VSILOGE("vxGetContext failure! at line %d\n", __LINE__);
            goto final;
        }

        status = vsi_nn_vxGetTensorAttr(imgObj[0], &in_attr);
        status |= vsi_nn_vxGetTensorAttr(imgObj[1], &in_attr1);
        status |= vsi_nn_vxGetTensorAttr(imgObj[2], &out_attr);
        if (status != VX_SUCCESS)
        {
            VSILOGE("vsi_nn_vxGetTensorAttr failure! at line %d\n", __LINE__);
            goto final;
        }

        output = (uint8_t*)malloc(vsi_nn_vxGetTensorElementNum(&out_attr)*vsi_nn_GetTypeBytes(out_attr.dtype.vx_type));

        input = vsi_nn_ConvertRawTensorToData2(context, imgObj[0],
            &(in_attr), stride_size[0], &(user_addr[0]), VX_READ_ONLY);
        input1 = vsi_nn_ConvertRawTensorToData2(context, imgObj[1],
            &(in_attr1), stride_size[1], &(user_addr[1]), VX_READ_ONLY);

        elementCount = vsi_nn_vxGetTensorElementNum(&in_attr);

        for (i = 0; i < elementCount; i++)
        {
            uint32_t  in0offset = 0;
            uint32_t  in1offset = 0;
            vx_uint8   *in0_ptr  = NULL;
            vx_uint8   *in1_ptr  = NULL;
            vx_float32 in0Data   = 0;
            vx_float32 in1Data   = 0;
            vx_float32 outData   = 0;

            in0offset = getExpandTensorOffset(i, in_attr.dim_num, in_attr.size, stride_size[0], out_attr.size);
            in1offset = getExpandTensorOffset(i, in_attr1.dim_num, in_attr.size, stride_size[1], out_attr.size);

            in0_ptr = (vx_uint8 *)input + in0offset;
            in1_ptr = (vx_uint8 *)input1 + in1offset;

            in0Data = vsi_nn_DtypeToFloat32_Ex(in0_ptr, 0, &in_attr.dtype);
            in1Data = vsi_nn_DtypeToFloat32_Ex(in1_ptr, 0, &in_attr1.dtype);

            outData = (float)floor(in0Data/in1Data);

            vsi_nn_Float32ToDtype_Ext(outData, output, i, &out_attr.dtype);
        }

        //output tensor
        status = vsi_nn_vxCopyDataToTensor(context, imgObj[2], &out_attr, output);
        if (status != VX_SUCCESS)
        {
            VSILOGE("vsi_nn_vxCopyDataToTensor failure! at line %d\n", __LINE__);
            goto final;
        }
final:
        if(input) free(input);
        if(input1) free(input1);
        if(output) free(output);
        for(i = 0; i < 3; i++)
            if (user_addr[i]) vxReleaseTensorAddressing(&(user_addr[i]));

    }

    return status;
}

vsi_status VX_CALLBACK vxFloorDivInitializer
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

    vx_tensor     input0          = (vx_tensor)paramObj[0];
    vx_tensor     input1          = (vx_tensor)paramObj[1];
    vx_tensor     output          = (vx_tensor)paramObj[2];

    uint32_t      input_size[DIM_SIZE]   = {1, 1, 1, 1};
    uint32_t      input_dims      = 0;
    uint32_t      zAx             = 1;
    vsi_nn_type_e inputDataFormat = VSI_NN_TYPE_FLOAT16;
    vsi_nn_type_e inputDataFormat1 = VSI_NN_TYPE_FLOAT16;
    vsi_nn_type_e outputDataFormat = VSI_NN_TYPE_FLOAT16;
    int8_t      input_fixPointPos0      = 0;
    int8_t      input_fixPointPos1      = 0;
    int8_t      output_fixPointPos      = 0;
    vx_float32  u8InScale0 = 0;
    vx_float32  u8InScale1 = 0;
    vx_float32  u8OutScale = 0;
    vx_float32  inScale0 = 0;
    vx_float32  inScale1 = 0;
    vx_float32  outScale = 0;
    int32_t     inZp0 = 0;
    int32_t     inZp1 = 0;
    int32_t     outZp = 0;
    vx_uint32   i     = 0;
    vsi_nn_tensor_attr_t attr[3];

    memset(&attr[0], 0, sizeof(vsi_nn_tensor_attr_t));
    memset(&attr[1], 0, sizeof(vsi_nn_tensor_attr_t));
    memset(&attr[2], 0, sizeof(vsi_nn_tensor_attr_t));

    status  = vsi_nn_vxGetTensorAttr(input0, &attr[0]);
    status |= vsi_nn_vxGetTensorAttr(input1, &attr[1]);
    status |= vsi_nn_vxGetTensorAttr(output, &attr[2]);
    if (status != VX_SUCCESS)
    {
        VSILOGE("vsi_nn_vxGetTensorAttr  failure! at line %d\n", __LINE__);
        return status;
    }

    //vx_uint32 factor = 1;
    //vx_uint32 maxWorkGroupSize = 8;
    input_dims = attr[0].dim_num;
    for (i = 0; i < input_dims; i++)
    {
        input_size[i] = attr[0].size[i];
    }
    inputDataFormat     = attr[0].dtype.vx_type;
    input_fixPointPos0  = attr[0].dtype.fl;
    inZp0               = attr[0].dtype.zero_point;
    u8InScale0          = attr[0].dtype.scale;
    inputDataFormat1    = attr[1].dtype.vx_type;
    input_fixPointPos1  = attr[1].dtype.fl;
    inZp1               = attr[1].dtype.zero_point;
    u8InScale1          = attr[1].dtype.scale;
    outputDataFormat    = attr[2].dtype.vx_type;
    output_fixPointPos  = attr[2].dtype.fl;
    outZp               = attr[2].dtype.zero_point;
    u8OutScale          = attr[2].dtype.scale;

    if(inputDataFormat == VSI_NN_TYPE_INT16 || inputDataFormat == VSI_NN_TYPE_INT8)
    {
        if (input_fixPointPos0 >= 0)
        {
            inScale0 = 1.0f / (vx_float32) (1 << input_fixPointPos0);
        }
        else if (input_fixPointPos0 < 0)
        {
            inScale0 = (vx_float32) (1 << -input_fixPointPos0);
        }
    }
    else if(inputDataFormat == VSI_NN_TYPE_UINT8)
    {
        inScale0 = u8InScale0;
    }

    if(inputDataFormat1 == VSI_NN_TYPE_INT16 || inputDataFormat1 == VSI_NN_TYPE_INT8)
    {
        if (input_fixPointPos1 >= 0)
        {
            inScale1 = 1.0f / (vx_float32) (1 << input_fixPointPos1);
        }
        else if (input_fixPointPos1 < 0)
        {
            inScale1 = (vx_float32) (1 << -input_fixPointPos1);
        }
    }
    else if(inputDataFormat1 == VSI_NN_TYPE_UINT8)
    {
        inScale1 = u8InScale1;
    }

    if(outputDataFormat == VSI_NN_TYPE_INT16 || outputDataFormat == VSI_NN_TYPE_INT8)
    {
        if (output_fixPointPos >= 0)
        {
            outScale = (vx_float32) (1 << output_fixPointPos);
        }
        else if (output_fixPointPos < 0)
        {
            outScale = 1.0f / (vx_float32) (1 << -output_fixPointPos);
        }
    }
    else if(outputDataFormat == VSI_NN_TYPE_UINT8)
    {
        outScale = 1 / u8OutScale;
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
#if 1
    {
        vx_uint32 uniConvertUint8SubZpToFp32_4x4[16] = {
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002100, // AccumType, ConstantType, and PostShift
            0xbc003c00, 0x00000000, 0xbc003c00, 0x00000000, 0xbc003c00, 0x00000000, 0xbc003c00, 0x00000000 // Constant
        };

        vx_uint32 uniConvertSecUint8SubZpToFp32_4x4[16] = {
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00050004, 0x00070006, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0xbc003c00, 0x00000000, 0xbc003c00, 0x00000000, 0xbc003c00, 0x00000000, 0xbc003c00, 0x00000000 // Constant
        };

        vx_uint32 uniConvertDirInt16Fp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };
        vx_uint32 uniConvertEndInt16Fp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };
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

        vx_uint32 uniConvertInt8FstFp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        };
        vx_uint32 uniConvertInt8SecFp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        };

        status |= vxSetNodeUniform(nodObj, "uniConvertInt8FstFp32_4x4", 1, uniConvertInt8FstFp32_4x4);
        status |= vxSetNodeUniform(nodObj, "uniConvertInt8SecFp32_4x4", 1, uniConvertInt8SecFp32_4x4);

        status |= vxSetNodeUniform(nodObj, "uniConvertFstFp16Fp32_4x4", 1, uniConvertFstFp16Fp32_4x4);
        status |= vxSetNodeUniform(nodObj, "uniConvertSecFp16Fp32_4x4", 1, uniConvertSecFp16Fp32_4x4);

        status |= vxSetNodeUniform(nodObj, "uniConvertUint8SubZpToFp32_4x4", 1, uniConvertUint8SubZpToFp32_4x4);
        status |= vxSetNodeUniform(nodObj, "uniConvertSecUint8SubZpToFp32_4x4", 1, uniConvertSecUint8SubZpToFp32_4x4);

        status |= vxSetNodeUniform(nodObj, "uniConvertDirInt16Fp32_4x4", 1, uniConvertDirInt16Fp32_4x4);
        status |= vxSetNodeUniform(nodObj, "uniConvertEndInt16Fp32_4x4", 1, uniConvertEndInt16Fp32_4x4);
        status |= vxSetNodeUniform(nodObj, "uniConvertInt32toUint8_2x8", 1, uniConvertInt32toUint8_2x8);
        status |= vxSetNodeUniform(nodObj, "in_scale0", 1, &inScale0);
        status |= vxSetNodeUniform(nodObj, "in_scale1", 1, &inScale1);
        status |= vxSetNodeUniform(nodObj, "out_scale", 1, &outScale);
        status |= vxSetNodeUniform(nodObj, "in_zp0", 1, &inZp0);
        status |= vxSetNodeUniform(nodObj, "in_zp1", 1, &inZp1);
        status |= vxSetNodeUniform(nodObj, "out_zp", 1, &outZp);
        if(status < 0)
        {
            VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
        }
    }
#endif
    return status;
}
static vx_param_description_t vxFloorDivKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED}
};
#ifdef __cplusplus
extern "C" {
#endif
vx_kernel_description_t vxFloorDivKernelInfo_fp16_fp16 =
{
    VX_KERNEL_ENUM_FLOORDIV,
    VX_KERNEL_NAME_FLOORDIV_FP16,
    NULL,
    vxFloorDivKernelParam,
    (sizeof(vxFloorDivKernelParam) / sizeof(vxFloorDivKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxFloorDivInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxFloorDivKernelInfo_int16_int16 =
{
    VX_KERNEL_ENUM_FLOORDIV,
    VX_KERNEL_NAME_FLOORDIV_INT16,
    NULL,
    vxFloorDivKernelParam,
    (sizeof(vxFloorDivKernelParam) / sizeof(vxFloorDivKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxFloorDivInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxFloorDivKernelInfo_int8_int8 =
{
    VX_KERNEL_ENUM_FLOORDIV,
    VX_KERNEL_NAME_FLOORDIV_INT8,
    NULL,
    vxFloorDivKernelParam,
    (sizeof(vxFloorDivKernelParam) / sizeof(vxFloorDivKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxFloorDivInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxFloorDivKernelInfo_uint8_uint8 =
{
    VX_KERNEL_ENUM_FLOORDIV,
    VX_KERNEL_NAME_FLOORDIV_UINT8,
    NULL,
    vxFloorDivKernelParam,
    (sizeof(vxFloorDivKernelParam) / sizeof(vxFloorDivKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxFloorDivInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxFloorDivKernelInfo_CPU =
{
    VX_KERNEL_ENUM_FLOORDIV,
    VX_KERNEL_NAME_FLOORDIV_FP16,
    vxFloorDivKernel,
    vxFloorDivKernelParam,
    (sizeof(vxFloorDivKernelParam) / sizeof(vxFloorDivKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_FLOORDIV_list[] =
{
    &vxFloorDivKernelInfo_CPU,
    &vxFloorDivKernelInfo_fp16_fp16,
    &vxFloorDivKernelInfo_int16_int16,
    &vxFloorDivKernelInfo_int8_int8,
    &vxFloorDivKernelInfo_uint8_uint8,
    NULL
};
#ifdef __cplusplus
}
#endif
