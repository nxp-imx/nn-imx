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
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_dtype_util.h"
#include "utils/vsi_nn_math.h"
#include "vsi_nn_log.h"
#include "client/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"

#define TENSOR_NUM_INPUT  (ARGMIN_INPUTS_COUNT)
#define TENSOR_NUM_OUTPUT (ARGMIN_OUTPUTS_COUNT)
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

vsi_status VX_CALLBACK vxArgMinKernel
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
    int32_t      axis                           = 0;
    vx_uint32    inner                          = 0;
    vx_uint32    outer                          = 0;
    vx_uint32    innerSize                      = 1;
    vx_uint32    outerSize                      = 1;
    vx_uint32    axisSize                       = 1;


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

    axisSize = attr[ARGMIN_INPUT].size[axis];

    for (i = 0; i < (uint32_t)axis; i++)
    {
        innerSize *= attr[ARGMIN_INPUT].size[i];
    }

    for (i = axis + 1; i < attr[ARGMIN_INPUT].dim_num; i++)
    {
        outerSize *= attr[ARGMIN_INPUT].size[i];
    }

   for (outer = 0; outer < outerSize; outer++)
    {
        for (inner = 0; inner < innerSize; inner++)
        {
            vx_uint32 index = outer * axisSize * innerSize + inner;
            vx_float32 maxminVal = 0;
            vx_float32 val = 0;
            vx_int32 maxminIndex = 0;

            maxminVal = vsi_nn_DtypeToFloat32_Ex(buffer_ptr[ARGMIN_INPUT], index, &attr[ARGMIN_INPUT].dtype);

            for (i = 1; i < axisSize; i++)
            {
                index = (outer * axisSize + i) * innerSize + inner;

                val = vsi_nn_DtypeToFloat32_Ex(buffer_ptr[ARGMIN_INPUT], index, &attr[ARGMIN_INPUT].dtype);

                if (val < maxminVal)
                {
                    maxminVal = val;
                    maxminIndex = i;
                }
            }

            index = outer * innerSize + inner;
            vsi_nn_Float32ToDtype_Ext((float)maxminIndex, buffer_ptr[ARGMIN_INPUTS_COUNT + ARGMIN_OUTPUT],
                index, &attr[ARGMIN_INPUTS_COUNT + ARGMIN_OUTPUT].dtype);
        }
    }

    //save data
    for( i = TENSOR_NUM_INPUT; i < TENSOR_NUM; i ++ )
    {
        if (buffer_ptr[i])
        {
            status = vsi_nn_copy_tensor_patch(tensor[i], &attr[i], buffer_ptr[i], VX_WRITE_ONLY);
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

vsi_status VX_CALLBACK vxArgMinInitializer
    (
    vx_node node,
    const vx_reference *paramObj,
    uint32_t paraNum
    )
{
// Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
    vx_kernel_execution_parameters_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {1, 1, 1},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vx_status status = VX_SUCCESS;

    vx_tensor    input              = (vx_tensor)paramObj[0];
    vx_tensor    output             = (vx_tensor)paramObj[1];
    vx_enum      srcFormat          = VSI_NN_TYPE_FLOAT16;
    vx_enum      dstFormat          = VSI_NN_TYPE_FLOAT16;
    int32_t      axis               = 0;
    vx_uint32    argLenSub1         = 0;
    vx_uint32    packedArgIdx[4]    = {0};
    vsi_nn_tensor_attr_t attr[2];

    memset(&attr[0], 0, sizeof(vsi_nn_tensor_attr_t));
    memset(&attr[1], 0, sizeof(vsi_nn_tensor_attr_t));

    status  = vsi_nn_vxGetTensorAttr(input, &attr[0]);
    status |= vsi_nn_vxGetTensorAttr(output, &attr[1]);

    srcFormat = attr[0].dtype.vx_type;
    dstFormat = attr[1].dtype.vx_type;

    srcFormat = srcFormat == VSI_NN_TYPE_BFLOAT16 ? VSI_NN_TYPE_FLOAT16 : srcFormat;
    dstFormat = dstFormat == VSI_NN_TYPE_BFLOAT16 ? VSI_NN_TYPE_FLOAT16 : dstFormat;

    vxCopyScalar((vx_scalar)paramObj[TENSOR_NUM], &(axis), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    if (axis == 2 && attr[0].size[2] == 1)
    {
        argLenSub1 = attr[0].size[1] - 1;
    }
    else
    {
        if (axis == 2)
            argLenSub1 = attr[0].size[2] - 1;
        else if (axis == 1)
            argLenSub1 = attr[0].size[1] - 1;
    }

    if (axis == 0)
    {
        shaderParam.globalWorkScale[0]  = 1;
        shaderParam.globalWorkScale[1]  = 1;
        shaderParam.globalWorkScale[2]  = 1;

        if (srcFormat == VSI_NN_TYPE_FLOAT16)
        {
            packedArgIdx[0] = 0x00000000;
            packedArgIdx[1] = 0x00000001;
            packedArgIdx[2] = 0x00000002;
            packedArgIdx[3] = 0x00000003;
        }
        else if (dstFormat == VSI_NN_TYPE_INT8 || dstFormat == VSI_NN_TYPE_UINT8)
        {
            packedArgIdx[0] = 0x03020100;
            packedArgIdx[1] = 0x07060504;
            packedArgIdx[2] = 0x0b0a0908;
            packedArgIdx[3] = 0x0f0e0d0c;
        }
        else
        {
            packedArgIdx[0] = 0x00010000;
            packedArgIdx[1] = 0x00030002;
            packedArgIdx[2] = 0x00050004;
            packedArgIdx[3] = 0x00070006;
        }
    }
    else
    {
        shaderParam.globalWorkScale[0]  = 8;
        shaderParam.globalWorkScale[1]  = 1;
        shaderParam.globalWorkScale[2]  = 1;
        packedArgIdx[0] = packedArgIdx[1] = (argLenSub1 << 16) | (argLenSub1 & 0xFFFF);
        packedArgIdx[2] = packedArgIdx[3] = (argLenSub1 << 16) | (argLenSub1 & 0xFFFF);

        if (srcFormat == VSI_NN_TYPE_INT8 ||
            srcFormat == VSI_NN_TYPE_UINT8)
        {
            if ( dstFormat == VSI_NN_TYPE_INT8 ||
                 dstFormat == VSI_NN_TYPE_UINT8)
            {
                vx_uint32 pack = ((argLenSub1 & 0xFF) << 24) | ((argLenSub1 & 0xFF) << 16)
                                 | ((argLenSub1 & 0xFF) << 8) | (argLenSub1 & 0xFF);
                packedArgIdx[0] = packedArgIdx[1] = pack;
                packedArgIdx[2] = packedArgIdx[3] = pack;
            }
        }
    }

    shaderParam.globalWorkSize[0]   = gcmALIGN((attr[1].size[0] + shaderParam.globalWorkScale[0] - 1)
                                        / shaderParam.globalWorkScale[0], 4);
    shaderParam.globalWorkSize[1]   = (attr[1].size[1] + shaderParam.globalWorkScale[1] - 1)
                                        / shaderParam.globalWorkScale[1];
    shaderParam.globalWorkSize[2]   = attr[1].dim_num > 2 ? attr[1].size[2] : 1;

    if (axis == 0)
    {
        vx_uint32 uniPackedIdxAddSat_2x8[16] = {
            0x55555555, // TCfg
            0x44444444, // ASelt
            0x33221100, 0x77665544, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0xffff0001, 0xffff0001, 0xffff0001, 0xffff0001,
            0xffff0001, 0xffff0001, 0xffff0001, 0xffff0001 // Constant
        };
        vx_uint32 uniSrcT2DstT_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x0000ffff, 0x0000ffff, 0x0000ffff, 0x0000ffff,
            0x0000ffff, 0x0000ffff, 0x0000ffff, 0x0000ffff // Constant
        };
        vx_uint32 uniConvertHalf2Float32_4x4[16] = {
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        };

        if (srcFormat == VSI_NN_TYPE_FLOAT16)
            vxSetNodeUniform(node, "uniConvertHalf2Float32_4x4", 1, uniConvertHalf2Float32_4x4);
        else
        {
            vxSetNodeUniform(node, "uniPackedIdxAddSat_2x8", 1, uniPackedIdxAddSat_2x8);
            vxSetNodeUniform(node, "uniSrcT2DstT_2x8", 1, uniSrcT2DstT_2x8);
        }
        vxSetNodeUniform(node, "inputWidth", 1, &attr[0].size[0]);
    }
    else
    {
        vx_uint32 uniExtractData_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        };

        status |= vxSetNodeUniform(node, "uniExtractData_2x8", 1, uniExtractData_2x8);

        status |= vxSetNodeUniform(node, "argLenSub1", 1, &argLenSub1);
    }

    status |= vxSetNodeUniform(node, "packedArgIdx", 1, packedArgIdx);
    status |= vxSetNodeAttribute(node, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));

    return status;
}

static vx_param_description_t vxArgMinKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED}
};
#ifdef __cplusplus
extern "C" {
#endif

vx_kernel_description_t vxArgMinKernelInfo_CPU =
{
    VX_KERNEL_ENUM_ARGMIN,
    "com.vivantecorp.extension.argmin_sw",
    vxArgMinKernel,
    vxArgMinKernelParam,
    (sizeof(vxArgMinKernelParam) / sizeof(vxArgMinKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

#define GEN_ARGMIN_SH_KERNEL_NAME(AXIS, SRC_TYPE, DST_TYPE) \
    "com.vivantecorp.extension.vxArgmin_Axis"#AXIS"_"#SRC_TYPE"to"#DST_TYPE

#define GEN_ARGMIN_SH_KERNEL_NAME_2D(AXIS, SRC_TYPE, DST_TYPE) \
    "com.vivantecorp.extension.vxArgmin_Axis"#AXIS"_"#SRC_TYPE"to"#DST_TYPE"_2D"


#define TENSOR_ARGMIN_KERNELS(AXIS, SRC_TYPE, DST_TYPE) \
    vx_kernel_description_t vxArgmin_Axis##AXIS##_##SRC_TYPE##to##DST_TYPE##_Kernel = \
{ \
    VX_KERNEL_ENUM_ARGMIN, \
    GEN_ARGMIN_SH_KERNEL_NAME(AXIS, SRC_TYPE, DST_TYPE), \
    NULL, \
    vxArgMinKernelParam, \
    (sizeof(vxArgMinKernelParam) / sizeof(vxArgMinKernelParam[0])), \
    vsi_nn_KernelValidator, \
    NULL, \
    NULL, \
    vxArgMinInitializer, \
    vsi_nn_KernelDeinitializer \
};

#define TENSOR_ARGMIN_KERNELS_2D(AXIS, SRC_TYPE, DST_TYPE) \
    vx_kernel_description_t vxArgmin_Axis##AXIS##_##SRC_TYPE##to##DST_TYPE##_2D_Kernel = \
{ \
    VX_KERNEL_ENUM_ARGMIN, \
    GEN_ARGMIN_SH_KERNEL_NAME_2D(AXIS, SRC_TYPE, DST_TYPE), \
    NULL, \
    vxArgMinKernelParam, \
    (sizeof(vxArgMinKernelParam) / sizeof(vxArgMinKernelParam[0])), \
    vsi_nn_KernelValidator, \
    NULL, \
    NULL, \
    vxArgMinInitializer, \
    vsi_nn_KernelDeinitializer \
};

/* axis 0 */
TENSOR_ARGMIN_KERNELS(0, I8,  U8)
TENSOR_ARGMIN_KERNELS(0, I8,  I16)
TENSOR_ARGMIN_KERNELS(0, U8,  U8)
TENSOR_ARGMIN_KERNELS(0, U8,  I16)
TENSOR_ARGMIN_KERNELS(0, I16, U8)
TENSOR_ARGMIN_KERNELS(0, I16, I16)
TENSOR_ARGMIN_KERNELS(0, F16, U8)
TENSOR_ARGMIN_KERNELS(0, F16, I16)

TENSOR_ARGMIN_KERNELS_2D(0, I8,  U8)
TENSOR_ARGMIN_KERNELS_2D(0, I8,  I16)
TENSOR_ARGMIN_KERNELS_2D(0, U8,  U8)
TENSOR_ARGMIN_KERNELS_2D(0, U8,  I16)
TENSOR_ARGMIN_KERNELS_2D(0, I16, U8)
TENSOR_ARGMIN_KERNELS_2D(0, I16, I16)
TENSOR_ARGMIN_KERNELS_2D(0, F16, U8)
TENSOR_ARGMIN_KERNELS_2D(0, F16, I16)

/* axis 1 */
TENSOR_ARGMIN_KERNELS(1, I8,  U8)
TENSOR_ARGMIN_KERNELS(1, I8,  I16)
TENSOR_ARGMIN_KERNELS(1, U8,  U8)
TENSOR_ARGMIN_KERNELS(1, U8,  I16)
TENSOR_ARGMIN_KERNELS(1, I16, U8)
TENSOR_ARGMIN_KERNELS(1, I16, I16)
TENSOR_ARGMIN_KERNELS(1, F16, U8)
TENSOR_ARGMIN_KERNELS(1, F16, I16)

/* axis 2 */
TENSOR_ARGMIN_KERNELS(2, I8,  U8)
TENSOR_ARGMIN_KERNELS(2, I8,  I16)
TENSOR_ARGMIN_KERNELS(2, U8,  U8)
TENSOR_ARGMIN_KERNELS(2, U8,  I16)
TENSOR_ARGMIN_KERNELS(2, I16, U8)
TENSOR_ARGMIN_KERNELS(2, I16, I16)
TENSOR_ARGMIN_KERNELS(2, F16, U8)
TENSOR_ARGMIN_KERNELS(2, F16, I16)

TENSOR_ARGMIN_KERNELS_2D(2, I8,  U8)
TENSOR_ARGMIN_KERNELS_2D(2, I8,  I16)
TENSOR_ARGMIN_KERNELS_2D(2, U8,  U8)
TENSOR_ARGMIN_KERNELS_2D(2, U8,  I16)
TENSOR_ARGMIN_KERNELS_2D(2, I16, U8)
TENSOR_ARGMIN_KERNELS_2D(2, I16, I16)
TENSOR_ARGMIN_KERNELS_2D(2, F16, U8)
TENSOR_ARGMIN_KERNELS_2D(2, F16, I16)

#define TENSOR_ARGMIN_KERENLS_NAME(AXIS, SRC_TYPE, DST_TYPE) \
    &vxArgmin_Axis##AXIS##_##SRC_TYPE##to##DST_TYPE##_Kernel,

#define TENSOR_ARGMIN_KERENLS_NAME_2D(AXIS, SRC_TYPE, DST_TYPE) \
    &vxArgmin_Axis##AXIS##_##SRC_TYPE##to##DST_TYPE##_2D_Kernel,

vx_kernel_description_t * vx_kernel_ARGMIN_list[] =
{
    &vxArgMinKernelInfo_CPU,

    /* axis 0 */
    TENSOR_ARGMIN_KERENLS_NAME(0, I8,  U8)
    TENSOR_ARGMIN_KERENLS_NAME(0, I8,  I16)
    TENSOR_ARGMIN_KERENLS_NAME(0, U8,  U8)
    TENSOR_ARGMIN_KERENLS_NAME(0, U8,  I16)
    TENSOR_ARGMIN_KERENLS_NAME(0, I16, U8)
    TENSOR_ARGMIN_KERENLS_NAME(0, I16, I16)
    TENSOR_ARGMIN_KERENLS_NAME(0, F16, U8)
    TENSOR_ARGMIN_KERENLS_NAME(0, F16, I16)

    TENSOR_ARGMIN_KERENLS_NAME_2D(0, I8,  U8)
    TENSOR_ARGMIN_KERENLS_NAME_2D(0, I8,  I16)
    TENSOR_ARGMIN_KERENLS_NAME_2D(0, U8,  U8)
    TENSOR_ARGMIN_KERENLS_NAME_2D(0, U8,  I16)
    TENSOR_ARGMIN_KERENLS_NAME_2D(0, I16, U8)
    TENSOR_ARGMIN_KERENLS_NAME_2D(0, I16, I16)
    TENSOR_ARGMIN_KERENLS_NAME_2D(0, F16, U8)
    TENSOR_ARGMIN_KERENLS_NAME_2D(0, F16, I16)

    /* axis 1 */
    TENSOR_ARGMIN_KERENLS_NAME(1, I8,  U8)
    TENSOR_ARGMIN_KERENLS_NAME(1, I8,  I16)
    TENSOR_ARGMIN_KERENLS_NAME(1, U8,  U8)
    TENSOR_ARGMIN_KERENLS_NAME(1, U8,  I16)
    TENSOR_ARGMIN_KERENLS_NAME(1, I16, U8)
    TENSOR_ARGMIN_KERENLS_NAME(1, I16, I16)
    TENSOR_ARGMIN_KERENLS_NAME(1, F16, U8)
    TENSOR_ARGMIN_KERENLS_NAME(1, F16, I16)

    /* axis 2 */
    TENSOR_ARGMIN_KERENLS_NAME(2, I8,  U8)
    TENSOR_ARGMIN_KERENLS_NAME(2, I8,  I16)
    TENSOR_ARGMIN_KERENLS_NAME(2, U8,  U8)
    TENSOR_ARGMIN_KERENLS_NAME(2, U8,  I16)
    TENSOR_ARGMIN_KERENLS_NAME(2, I16, U8)
    TENSOR_ARGMIN_KERENLS_NAME(2, I16, I16)
    TENSOR_ARGMIN_KERENLS_NAME(2, F16, U8)
    TENSOR_ARGMIN_KERENLS_NAME(2, F16, I16)

    TENSOR_ARGMIN_KERENLS_NAME_2D(2, I8,  U8)
    TENSOR_ARGMIN_KERENLS_NAME_2D(2, I8,  I16)
    TENSOR_ARGMIN_KERENLS_NAME_2D(2, U8,  U8)
    TENSOR_ARGMIN_KERENLS_NAME_2D(2, U8,  I16)
    TENSOR_ARGMIN_KERENLS_NAME_2D(2, I16, U8)
    TENSOR_ARGMIN_KERENLS_NAME_2D(2, I16, I16)
    TENSOR_ARGMIN_KERENLS_NAME_2D(2, F16, U8)
    TENSOR_ARGMIN_KERENLS_NAME_2D(2, F16, I16)

    NULL
};
#ifdef __cplusplus
}
#endif
