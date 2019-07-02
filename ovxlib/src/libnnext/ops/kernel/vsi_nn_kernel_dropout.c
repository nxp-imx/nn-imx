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

#include "vsi_nn_platform.h"

#include "vsi_nn_prv.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_math.h"
#include "vsi_nn_log.h"
#include "client/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"

vsi_status VX_CALLBACK vxParametricDropoutValidator
    (
    vx_node node,
    const vx_reference parameters[],
    uint32_t num,
    vx_meta_format metas[]
    )
{
    vsi_status status = VX_SUCCESS;
    return status;
}

vsi_status VX_CALLBACK vxParametricDropoutInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    uint32_t paraNum
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

    vsi_status status = VX_SUCCESS;
    uint32_t dims = 0;
    uint32_t size[DIM_SIZE] = {0};

    vx_tensor tensor = (vx_tensor)paramObj[0];
    status = vxQueryTensor(tensor, VX_TENSOR_NUM_OF_DIMS, &dims, sizeof(uint32_t));
    status = vxQueryTensor(tensor, VX_TENSOR_DIMS, size, sizeof(uint32_t) * dims);

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkScale[0]  = 8;
    shaderParam.globalWorkScale[1]  = 1;
    shaderParam.globalWorkSize[0]   = gcmALIGN((size[0] + shaderParam.globalWorkScale[0] - 1) /
        shaderParam.globalWorkScale[0], 4);
    shaderParam.globalWorkSize[1]   = (size[1] + shaderParam.globalWorkScale[1] - 1) /
        shaderParam.globalWorkScale[1];

    {
        // uniforms
        uint32_t fp16MulFp16ToFp16_2x8[16] = {
            0x11111111, // TCfg
            0x00000000, // ASelt
            0x03020100, 0x07060504, // ABin
            0x11111111, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000 // Constant
        };
        status |= vxSetNodeUniform(nodObj, "fp16MulFp16ToFp16_2x8", 1, fp16MulFp16ToFp16_2x8);
    }

    status |= vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS,
        &shaderParam, sizeof(vx_kernel_execution_parameters_t));

    return VX_SUCCESS;
}

vsi_status VX_CALLBACK vxParametricDropoutDeinitializer
    (
    vx_node nodObj,
    const vx_reference *paraObj,
    uint32_t paraNum
    )
{
    return VX_SUCCESS;
}

static vx_param_description_t vxParametricDropoutKernelParam[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};

#ifdef __cplusplus
extern "C" {
#endif
vx_kernel_description_t vxParametricDropoutKernelInfo =
{
    KERNEL_ENUM_DROPOUT,
    VX_KERNEL_NAME_DROPOUT,
    NULL,
    vxParametricDropoutKernelParam,
    (sizeof(vxParametricDropoutKernelParam) / sizeof(vxParametricDropoutKernelParam[0])),
    vxParametricDropoutValidator,
    NULL,
    NULL,
    vxParametricDropoutInitializer,
    vxParametricDropoutDeinitializer
};

vx_kernel_description_t * vx_kernel_DROPOUT_list[] =
{
    NULL,
    &vxParametricDropoutKernelInfo,
    NULL
};
#ifdef __cplusplus
}
#endif
