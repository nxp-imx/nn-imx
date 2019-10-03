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



#define TENSOR_NUM_INPUT  (REDUCEANY_INPUTS_COUNT)
#define TENSOR_NUM_OUTPUT (REDUCEANY_OUTPUTS_COUNT)
#define TENSOR_NUM (TENSOR_NUM_INPUT+TENSOR_NUM_OUTPUT)

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

static vsi_status VX_CALLBACK vxReduceany_internalKernel
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
    vx_uint32  size[4];
    vx_uint32  dims, innerSize, outerSize, axisSize;
    vx_uint32  outer, inner, i, index;
    int8_t     any_result, tmpValue;
    int32_t axis;

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
    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM], &(axis), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    dims = attr[REDUCEANY_INPUT].dim_num;
    size[0] = attr[REDUCEANY_INPUT].size[0];
    size[1] = dims > 1 ? attr[REDUCEANY_INPUT].size[1] : 1;
    size[2] = dims > 2 ? attr[REDUCEANY_INPUT].size[2] : 1;
    size[3] = dims > 3 ? attr[REDUCEANY_INPUT].size[3] : 1;
    axisSize =  attr[REDUCEANY_INPUT].size[axis];

    switch(axis)
    {
        case 0:
            innerSize = 1;
            outerSize = size[1] * size[2] * size[3];
            break;
        case 1:
            innerSize = size[0];
            outerSize = size[2] * size[3];
            break;
        case 2:
            innerSize = size[0] * size[1];
            outerSize = size[3];
            break;
        case 3:
            innerSize = size[0] * size[1] * size[2];
            outerSize = 1;
            break;
        default:
        printf("Input tensor error dimension[%u]\n", dims);
        return VX_ERROR_INVALID_DIMENSION;
    }

    for (outer = 0; outer < outerSize; ++outer) {
        for (inner = 0; inner < innerSize; ++inner) {
            index = outer * axisSize * innerSize + inner;
            tmpValue = (int8_t)vsi_nn_DtypeToFloat32_Ex(buffer_ptr[REDUCEANY_INPUT],
            index, &attr[REDUCEANY_INPUT].dtype);
            any_result = tmpValue ?  1 : 0;
            for (i = 1; i < axisSize; ++i) {
                index    = (outer * axisSize + i) * innerSize + inner;
                tmpValue = (int8_t)vsi_nn_DtypeToFloat32_Ex(buffer_ptr[REDUCEANY_INPUT],
                index, &attr[REDUCEANY_INPUT].dtype);
                tmpValue = tmpValue ? 1 : 0;
                any_result = any_result | tmpValue;
            }

            index     =  outer * innerSize + inner;
            vsi_nn_Float32ToDtype_Ext(any_result, buffer_ptr[REDUCEANY_INPUTS_COUNT + REDUCEANY_OUTPUT],
                index, &attr[REDUCEANY_INPUTS_COUNT + REDUCEANY_OUTPUT].dtype);
        }
    }
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
} /* _VX_KERNEL_FUNC_KERNEL() */


vx_status VX_CALLBACK vxReduceany_internalInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    vx_uint32 paraNum
    )
{
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)

    vx_kernel_execution_parameters_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {1, 1, 1},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread
    vx_status    status             = VX_SUCCESS;
    vx_tensor    input              = (vx_tensor)paramObj[0];
    vx_uint32    width              = 0;
    vx_uint32    height             = 0;
    vx_uint32    depth              = 0;
    vx_uint32    axis               = 0;
    vsi_nn_tensor_attr_t attr[2];
    vx_int32     axisSize = 0;

    vxCopyScalar((vx_scalar)paramObj[2], &(axis),VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    memset(&attr[0], 0, sizeof(vsi_nn_tensor_attr_t));

    status  = vsi_nn_vxGetTensorAttr(input, &attr[0]);

    if(status < 0)
    {
        VSILOGE("error-%s,%d\n",__FILE__,__LINE__);
        return status;
    }

    width          = attr[0].size[0];
    height         = attr[0].size[1];
    depth          = attr[0].dim_num > 2 ? attr[0].size[2] : 1;
    axisSize       = attr[0].size[axis];

    shaderParam.workDim             = 2;
    switch (axis)
    {
        case 0:
            shaderParam.globalWorkScale[0]  = 1;
            shaderParam.globalWorkScale[1]  = 1;
            shaderParam.globalWorkSize[0]   = height;
            shaderParam.globalWorkSize[1]   = depth;
        break;
        case 1:
            shaderParam.globalWorkScale[0]  = 16;
            shaderParam.globalWorkScale[1]  = 1;
            shaderParam.globalWorkSize[0]   = \
            gcmALIGN((width + shaderParam.globalWorkScale[0] - 1) / shaderParam.globalWorkScale[0], 4);
            shaderParam.globalWorkSize[1]   = depth;
        break;
        case 2:
            shaderParam.globalWorkScale[0]  = 16;
            shaderParam.globalWorkScale[1]  = 1;
            shaderParam.globalWorkSize[0]   = \
            gcmALIGN((width + shaderParam.globalWorkScale[0] - 1) / shaderParam.globalWorkScale[0], 4);
            shaderParam.globalWorkSize[1]   = height;
        break;
        default:
            printf("error input axis value %d \n", axis);
            return VX_ERROR_INVALID_PARAMETERS;
        break;
    }

    {
        vx_uint32 uniS8AddAll_16x1[16] = {
            0xffffffff, // TCfg
            0x00000000, // ASelt
            0x76543210, 0xfedcba98, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00004400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        };

        if (0 == axis)
        {
            status |= vxSetNodeUniform(nodObj, "uniS8AddAll_16x1", 1, uniS8AddAll_16x1);
        }
    }

    status |= vxSetNodeUniform(nodObj, "axisSize", 1, &axisSize);
    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,\
    &shaderParam, sizeof(vx_kernel_execution_parameters_t));

#undef gcmALIGN
    return status;
}

static vx_param_description_t vxReduceanyKernelParam[] =
{
    {VX_INPUT,  VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};


#ifdef __cplusplus
extern "C" {
#endif

vx_kernel_description_t vxReduceany_internal_CPU =
{
    VX_KERNEL_ENUM_REDUCEANY_INTERNAL,
    "com.vivantecorp.extension.vxcReduceany_sw",
    vxReduceany_internalKernel,
    vxReduceanyKernelParam,
    (sizeof(vxReduceanyKernelParam) / sizeof(vxReduceanyKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};


#define REDUCEANY_KERNELS(AXI_INDEX, SRC_TYPE, DST_TYPE) \
    vx_kernel_description_t vxReduceany_internal_##AXI_INDEX##_##SRC_TYPE##to##DST_TYPE##_Kernel = \
{ \
    VX_KERNEL_ENUM_REDUCEANY_INTERNAL, \
    VX_KERNEL_NAME_REDUCEANY_##AXI_INDEX##_##SRC_TYPE##TO##DST_TYPE, \
    NULL, \
    vxReduceanyKernelParam, \
    (sizeof(vxReduceanyKernelParam) / sizeof(vxReduceanyKernelParam[0])), \
    vsi_nn_KernelValidator, \
    NULL, \
    NULL, \
    vxReduceany_internalInitializer, \
    vsi_nn_KernelDeinitializer \
};

#define REDUCEANY_KERNELS_2D(AXI_INDEX, SRC_TYPE, DST_TYPE) \
    vx_kernel_description_t vxReduceany_internal_##AXI_INDEX##_##SRC_TYPE##to##DST_TYPE##_2D_Kernel = \
{ \
    VX_KERNEL_ENUM_REDUCEANY_INTERNAL, \
    VX_KERNEL_NAME_REDUCEANY_##AXI_INDEX##_##SRC_TYPE##TO##DST_TYPE##_2D, \
    NULL, \
    vxReduceanyKernelParam, \
    (sizeof(vxReduceanyKernelParam) / sizeof(vxReduceanyKernelParam[0])), \
    vsi_nn_KernelValidator, \
    NULL, \
    NULL, \
    vxReduceany_internalInitializer, \
    vsi_nn_KernelDeinitializer \
};

REDUCEANY_KERNELS(AXI0, I8,  I8)
REDUCEANY_KERNELS(AXI1, I8,  I8)
REDUCEANY_KERNELS(AXI2, I8,  I8)
REDUCEANY_KERNELS_2D(AXI0, I8,  I8)
REDUCEANY_KERNELS_2D(AXI1, I8,  I8)

#define REDUCEANY_KERENLS_NAME(AXI_INDEX, SRC_TYPE, DST_TYPE, INSTR) \
    &vxReduceany_internal_##AXI_INDEX##_##SRC_TYPE##to##DST_TYPE##_##INSTR##Kernel,

vx_kernel_description_t* vx_kernel_REDUCEANY_INTERNAL_list[] =
{
    &vxReduceany_internal_CPU,
    REDUCEANY_KERENLS_NAME(AXI0, I8,  I8, )
    REDUCEANY_KERENLS_NAME(AXI0, I8,  I8,  2D_)
    REDUCEANY_KERENLS_NAME(AXI1, I8,  I8, )
    REDUCEANY_KERENLS_NAME(AXI1, I8,  I8,  2D_)
    REDUCEANY_KERENLS_NAME(AXI2, I8,  I8, )
    NULL
};

#ifdef __cplusplus
}
#endif
