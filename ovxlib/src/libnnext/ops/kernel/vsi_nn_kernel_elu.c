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
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_math.h"
#include "vsi_nn_log.h"
#include "client/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"

vx_status VX_CALLBACK vxTensorEluInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    vx_uint32 paraNum
    )
{
    // Alignment with a power of two value.
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
    vx_kernel_execution_parameters_t shaderParam = {
        2,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread
    vx_uint32 uniunPackedLoData_4x4[16] = {
        0x01010101, // TCfg
        0x00000000, // ASelt
        0x00010000, 0x00030002, // ABin
        0x02020202, // BSelt
        0x00000000, 0x00000000, // BBin
        0x00000100, // AccumType, ConstantType, and PostShift
        0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
        0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
    };
    vx_uint32 uniunPackedHiData_4x4[16] = {
        0x01010101, // TCfg
        0x00000000, // ASelt
        0x00050004, 0x00070006, // ABin
        0x02020202, // BSelt
        0x00000000, 0x00000000, // BBin
        0x00000100, // AccumType, ConstantType, and PostShift
        0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
        0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
    };
    vx_uint32 uniExtractHalf8_2x8[16] = {
        0x11111111, // TCfg
        0x11110000, // ASelt
        0x06040200, 0x06040200, // ABin
        0x22222222, // BSelt
        0x00000000, 0x00000000, // BBin
        0x00000100, // AccumType, ConstantType, and PostShift
        0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00,
        0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
    };
    vx_status status = VX_SUCCESS;

    vx_tensor input = (vx_tensor)paramObj[0];
    vx_uint32 input_size[DIM_SIZE];
    vx_float32 scaleLogE = (vx_float32)(log10(exp(1.0f)) / log10(2.0f));

    status = vxQueryTensor(input, VX_TENSOR_DIMS, input_size, sizeof(input_size));

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkScale[0]  = 8;
    shaderParam.globalWorkScale[1]  = 2;
    shaderParam.globalWorkSize[0]   = gcmALIGN((input_size[0] + shaderParam.globalWorkScale[0] - 1)
        / shaderParam.globalWorkScale[0], 4);
    shaderParam.globalWorkSize[1]   = (input_size[1] + shaderParam.globalWorkScale[1] - 1)
        / shaderParam.globalWorkScale[1];

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));

    vxSetNodeUniform(nodObj, "uniunPackedLoData_4x4", 1, uniunPackedLoData_4x4);
    vxSetNodeUniform(nodObj, "uniunPackedHiData_4x4", 1, uniunPackedHiData_4x4);
    vxSetNodeUniform(nodObj, "uniExtractHalf8_2x8_elu", 1, uniExtractHalf8_2x8);
    vxSetNodeUniform(nodObj, "scaleLogE_elu", 1, &scaleLogE);

    return VX_SUCCESS;
}

static vx_param_description_t vxTensorEluKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED}
};


#ifdef __cpluplus
extern "C" {
#endif

vx_kernel_description_t vxTensorEluKernelInfo =
{
    VX_KERNEL_ENUM_TENSORELU,
    VX_KERNEL_NAME_TENSORELU_FP16_2D,
#ifdef USE_TENSORELU_VXC
    NULL,
#else
    NULL,
#endif
    vxTensorEluKernelParam,
    (sizeof(vxTensorEluKernelParam) / sizeof(vxTensorEluKernelParam[0])),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vxTensorEluInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_ELU_list[] =
{
    NULL,
    &vxTensorEluKernelInfo,
    NULL
};
#ifdef __cpluplus
}
#endif
