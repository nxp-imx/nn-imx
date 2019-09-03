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

#define TENSOR_NUM_INPUT  (POWER_INPUTS_COUNT)
#define TENSOR_NUM_OUTPUT (POWER_OUTPUTS_COUNT)
#define TENSOR_NUM (TENSOR_NUM_INPUT+TENSOR_NUM_OUTPUT)

static vx_uint32 getExpandTensorOffset(vx_uint32 index, vsi_nn_tensor_attr_t input_attr,
                                       uint32_t *input_stride_sz, vx_uint32 * out_dims)
{
    vx_uint32 offset = 0;
    vx_uint32 i;

    for(i = 0; i < input_attr.dim_num; i++)
    {
        if(input_attr.size[i] == out_dims[i])
            offset += input_stride_sz[i] * (index % out_dims[i]);

        index /= out_dims[i];
    }

    return offset;
}


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


vsi_status VX_CALLBACK vxPowKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
    vsi_status status = VX_SUCCESS;
    vsi_nn_tensor_attr_t attr[TENSOR_NUM];
    vx_tensor_addressing user_addr[TENSOR_NUM]  = {NULL};
    vx_uint8    *buffer_ptr[TENSOR_NUM]            = {NULL};
    vx_tensor   tensor[TENSOR_NUM];
    uint32_t    stride_size[TENSOR_NUM][VSI_NN_MAX_DIM_NUM];

    vx_context   context                        = vxGetContext((vx_reference)node);
    vx_uint32    i                              = 0;
    vx_uint32    elementCount                   = 1;

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

    for (i = 0; i < attr[POWER_INPUTS_COUNT + POWER_OUTPUT].dim_num; i++)
    {
        elementCount *= attr[POWER_INPUTS_COUNT + POWER_OUTPUT].size[i];
    }

    for (i = 0; i < elementCount; i++)
    {
        vx_uint32 in0offset, in1offset;
        vx_float32 value0 = 0;
        vx_float32 value1 = 0;
        vx_float32 dst = 0;

        vx_uint8 *input_data_ptr0 = NULL;
        vx_uint8 *input_data_ptr1 = NULL;

        in0offset = getExpandTensorOffset(i, attr[POWER_INPUT0],
            stride_size[POWER_INPUT0], attr[POWER_INPUTS_COUNT + POWER_OUTPUT].size);
        in1offset = getExpandTensorOffset(i, attr[POWER_INPUT1],
            stride_size[POWER_INPUT1], attr[POWER_INPUTS_COUNT + POWER_OUTPUT].size);

        input_data_ptr0 = buffer_ptr[POWER_INPUT0] + in0offset;
        input_data_ptr1 = buffer_ptr[POWER_INPUT1] + in1offset;

        value0 = vsi_nn_DtypeToFloat32_Ex(input_data_ptr0, 0, &attr[POWER_INPUT0].dtype);
        value1 = vsi_nn_DtypeToFloat32_Ex(input_data_ptr1, 0, &attr[POWER_INPUT1].dtype);
        dst = (vx_float32)pow(value0, value1);

        vsi_nn_Float32ToDtype_Ext(dst, buffer_ptr[POWER_INPUTS_COUNT + POWER_OUTPUT],
                    i, &attr[POWER_INPUTS_COUNT + POWER_OUTPUT].dtype);
    }

    //save data
    for( i = TENSOR_NUM_INPUT; i < TENSOR_NUM; i ++ )
    {
        if (buffer_ptr[i])
        {
            status = vxCopyTensorPatch(
                tensor[i],
                NULL,
                user_addr[i],
                buffer_ptr[i],
                VX_WRITE_ONLY,
                0
                );
        }

        if (user_addr[i]) vxReleaseTensorAddressing(&(user_addr[i]));
    }

    for( i = 0; i < TENSOR_NUM; i ++ )
    {
        if (buffer_ptr[i]) free(buffer_ptr[i]);

        if (user_addr[i]) vxReleaseTensorAddressing(&(user_addr[i]));
    }

    return status;
}

vsi_status VX_CALLBACK vxPowInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    uint32_t paraNum
    )
{
    vsi_status status = VX_SUCCESS;
    // Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
#define gcmMIN(x, y)            (((x) <= (y)) ?  (x) :  (y))
#define gcmMAX(x, y)            (((x) >= (y)) ?  (x) :  (y))
#define MAX_MULTIPLIER_NUM      (65535)
#define MAX_POST_SHIFT_BITS     (31)
    vx_kernel_execution_parameters_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vx_tensor     input0          = (vx_tensor)paramObj[0];
    vx_tensor     input1          = (vx_tensor)paramObj[1];
    vx_tensor     output          = (vx_tensor)paramObj[2];

    uint32_t      input_size[DIM_SIZE]   = {0};
    uint32_t      input_dims      = 0;
    uint32_t      output_dims     = 0;
    uint32_t      zAx             = 1;
    vsi_nn_type_e inputDataFormat = VSI_NN_TYPE_FLOAT16;
    vsi_nn_type_e inputDataFormat1 = VSI_NN_TYPE_FLOAT16;
    vsi_nn_type_e outputDataFormat = VSI_NN_TYPE_FLOAT16;
    vx_float32  scaleIn                 = 0;
    int32_t     input_ZP                = 0;
    vx_int8      src0FixPointPos    = 0;
    //vx_uint32 factor = 1;
    //vx_uint32 maxWorkGroupSize = 8;

    status  = vxQueryTensor(input0, VX_TENSOR_DIMS, input_size, sizeof(input_size));
    status |= vxQueryTensor(input0, VX_TENSOR_NUM_OF_DIMS, &input_dims, sizeof(input_dims));
    status |= vxQueryTensor(input0, VX_TENSOR_DATA_TYPE, &inputDataFormat, sizeof(inputDataFormat));
    status |= vxQueryTensor(input0, VX_TENSOR_ZERO_POINT, &input_ZP, sizeof(input_ZP));
    status |= vxQueryTensor(input0, VX_TENSOR_SCALE, &scaleIn, sizeof(scaleIn));
    status |= vxQueryTensor(input0, VX_TENSOR_FIXED_POINT_POSITION, &src0FixPointPos, sizeof(src0FixPointPos));
    status |= vxQueryTensor(input1, VX_TENSOR_DATA_TYPE, &inputDataFormat1, sizeof(inputDataFormat1));
    status |= vxQueryTensor(output, VX_TENSOR_NUM_OF_DIMS, &output_dims, sizeof(output_dims));
    status |= vxQueryTensor(output, VX_TENSOR_DATA_TYPE, &outputDataFormat, sizeof(outputDataFormat));
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

    {
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

        vx_uint32 uniConvertDirUint8Fp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };
        vx_uint32 uniConvertEndUint8Fp32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
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

        if(inputDataFormat == VSI_NN_TYPE_UINT8 && inputDataFormat1 == VSI_NN_TYPE_FLOAT16
            && outputDataFormat == VSI_NN_TYPE_FLOAT16)
        {
            vx_uint32 uniConvertUint8SubZpToFp32_4x4[16] = {
                0x05050505, // TCfg
                0x04040404, // ASelt
                0x00010000, 0x00030002, // ABin
                0x0a0a0a0a, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002100, // AccumType, ConstantType, and PostShift
                0xbc003c00, 0x00000000, 0xbc003c00, 0x00000000,
                    0xbc003c00, 0x00000000, 0xbc003c00, 0x00000000 // Constant
            };

            vx_uint32 uniConvertSecUint8SubZpToFp32_4x4[16] = {
                0x05050505, // TCfg
                0x04040404, // ASelt
                0x00050004, 0x00070006, // ABin
                0x0a0a0a0a, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0xbc003c00, 0x00000000, 0xbc003c00, 0x00000000,
                    0xbc003c00, 0x00000000, 0xbc003c00, 0x00000000 // Constant
            };
            status |= vxSetNodeUniform(nodObj, "uniConvertUint8SubZpToFp32_4x4", 1,
                            uniConvertUint8SubZpToFp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvertSecUint8SubZpToFp32_4x4", 1,
                            uniConvertSecUint8SubZpToFp32_4x4);
            status |= vxSetNodeUniform(nodObj, "input_ZP0", 1, &input_ZP);
            status |= vxSetNodeUniform(nodObj, "inputScale0", 1, &scaleIn);
        }
        else if(inputDataFormat == VSI_NN_TYPE_INT8 && inputDataFormat1 == VSI_NN_TYPE_FLOAT16
            && outputDataFormat == VSI_NN_TYPE_FLOAT16)
        {
            if (src0FixPointPos > 0)
            {
                vx_uint8  postshift = gcmMIN(src0FixPointPos - 0, MAX_POST_SHIFT_BITS);

                uniConvertInt8FstFp32_4x4[7] |= (postshift & 0x1F);
                uniConvertInt8SecFp32_4x4[7] |= (postshift & 0x1F);
            }
            else
            {
                vx_uint32 multiplier = gcmMIN(1 << (0 - src0FixPointPos), MAX_MULTIPLIER_NUM);
                vx_uint32 i          = 0;

                for (i = 0; i < 8; i++)
                {
                    uniConvertInt8FstFp32_4x4[i + 8] = multiplier;
                    uniConvertInt8SecFp32_4x4[i + 8] = multiplier;
                }
            }

            status |= vxSetNodeUniform(nodObj, "uniConvertInt8FstFp32_4x4", 1, uniConvertInt8FstFp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvertInt8SecFp32_4x4", 1, uniConvertInt8SecFp32_4x4);
        }
        else if(inputDataFormat == VSI_NN_TYPE_INT16 && inputDataFormat1 == VSI_NN_TYPE_FLOAT16
            && outputDataFormat == VSI_NN_TYPE_FLOAT16)
        {
            if (src0FixPointPos > 0)
            {
                vx_uint8  postshift = gcmMIN(src0FixPointPos - 0, MAX_POST_SHIFT_BITS);

                uniConvertDirInt16Fp32_4x4[7] |= (postshift & 0x1F);
                uniConvertEndInt16Fp32_4x4[7] |= (postshift & 0x1F);
            }
            else
            {
                vx_uint32 multiplier = gcmMIN(1 << (0 - src0FixPointPos), MAX_MULTIPLIER_NUM);
                vx_uint32 i          = 0;

                for (i = 0; i < 8; i++)
                {
                    uniConvertDirInt16Fp32_4x4[i + 8] = multiplier;
                    uniConvertEndInt16Fp32_4x4[i + 8] = multiplier;
                }
            }

            status |= vxSetNodeUniform(nodObj, "uniConvertDirInt16Fp32_4x4", 1, uniConvertDirInt16Fp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvertEndInt16Fp32_4x4", 1, uniConvertEndInt16Fp32_4x4);
        }
        else if(outputDataFormat == VSI_NN_TYPE_UINT8 && inputDataFormat == VSI_NN_TYPE_UINT8)
        {
            status |= vxSetNodeUniform(nodObj, "uniConvertDirUint8Fp32_4x4", 1, uniConvertDirUint8Fp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvertEndUint8Fp32_4x4", 1, uniConvertEndUint8Fp32_4x4);
        }
        else if(outputDataFormat == VSI_NN_TYPE_INT8 && inputDataFormat == VSI_NN_TYPE_INT8)
        {
            status |= vxSetNodeUniform(nodObj, "uniConvertInt8FstFp32_4x4", 1, uniConvertInt8FstFp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvertInt8SecFp32_4x4", 1, uniConvertInt8SecFp32_4x4);
        }
        else if(outputDataFormat == VSI_NN_TYPE_INT16 && inputDataFormat == VSI_NN_TYPE_INT16)
        {
            status |= vxSetNodeUniform(nodObj, "uniConvertDirInt16Fp32_4x4", 1, uniConvertDirInt16Fp32_4x4);
            status |= vxSetNodeUniform(nodObj, "uniConvertEndInt16Fp32_4x4", 1, uniConvertEndInt16Fp32_4x4);
        }

        status |= vxSetNodeUniform(nodObj, "uniConvertFstFp16Fp32_4x4", 1, uniConvertFstFp16Fp32_4x4);
        status |= vxSetNodeUniform(nodObj, "uniConvertSecFp16Fp32_4x4", 1, uniConvertSecFp16Fp32_4x4);
        status |= vxSetNodeUniform(nodObj, "uniConvertInt32toUint8_2x8", 1, uniConvertInt32toUint8_2x8);
        if(status < 0)
        {
            VSILOGE("[%s : %d]Initializer  failure! \n",__FILE__, __LINE__);
        }
    }
    return status;
}
static vx_param_description_t vxPowKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED}
};
#ifdef __cplusplus
extern "C" {
#endif
vx_kernel_description_t vxPowKernelInfo_fp16_fp16 =
{
    VX_KERNEL_ENUM_POW,
    VX_KERNEL_NAME_POW_FP16,
    NULL,
    vxPowKernelParam,
    (sizeof(vxPowKernelParam) / sizeof(vxPowKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxPowInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxPowKernelInfo_int16_int16 =
{
    VX_KERNEL_ENUM_POW,
    VX_KERNEL_NAME_POW_INT16,
    NULL,
    vxPowKernelParam,
    (sizeof(vxPowKernelParam) / sizeof(vxPowKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxPowInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxPowKernelInfo_int8_int8 =
{
    VX_KERNEL_ENUM_POW,
    VX_KERNEL_NAME_POW_INT8,
    NULL,
    vxPowKernelParam,
    (sizeof(vxPowKernelParam) / sizeof(vxPowKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxPowInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxPowKernelInfo_uint8_uint8 =
{
    VX_KERNEL_ENUM_POW,
    VX_KERNEL_NAME_POW_UINT8,
    NULL,
    vxPowKernelParam,
    (sizeof(vxPowKernelParam) / sizeof(vxPowKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxPowInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxPowKernelInfoU8_Fp16Fp16 =
{
    VX_KERNEL_ENUM_POW,
    VX_KERNEL_NAME_POW_UINT8_FP16FP16,
    NULL,
    vxPowKernelParam,
    (sizeof(vxPowKernelParam) / sizeof(vxPowKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxPowInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxPowKernelInfoI8_Fp16Fp16 =
{
    VX_KERNEL_ENUM_POW,
    VX_KERNEL_NAME_POW_INT8_FP16FP16,
    NULL,
    vxPowKernelParam,
    (sizeof(vxPowKernelParam) / sizeof(vxPowKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxPowInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxPowKernelInfoI16_Fp16Fp16 =
{
    VX_KERNEL_ENUM_POW,
    VX_KERNEL_NAME_POW_INT16_FP16FP16,
    NULL,
    vxPowKernelParam,
    (sizeof(vxPowKernelParam) / sizeof(vxPowKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxPowInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t vxPowKernelInfo_CPU =
{
    VX_KERNEL_ENUM_POW,
    VX_KERNEL_NAME_POW_FP16,
    vxPowKernel,
    vxPowKernelParam,
    (sizeof(vxPowKernelParam) / sizeof(vxPowKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_POW_list[] =
{
    &vxPowKernelInfo_CPU,
    &vxPowKernelInfo_fp16_fp16,
    &vxPowKernelInfo_int16_int16,
    &vxPowKernelInfo_int8_int8,
    &vxPowKernelInfo_uint8_uint8,
    &vxPowKernelInfoU8_Fp16Fp16,
    &vxPowKernelInfoI8_Fp16Fp16,
    &vxPowKernelInfoI16_Fp16Fp16,
    NULL
};
#ifdef __cplusplus
}
#endif
