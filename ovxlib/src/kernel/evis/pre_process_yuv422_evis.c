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


#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "vsi_nn_types.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_log.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_error.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "kernel/vsi_nn_kernel_eltwise.h"

__BEGIN_DECLS

#define VX_KERNEL_NAME_PRE_PROCESS_YUV422_SCALE_U8TOF16    CVIVANTE_NAMESPACE("evis.pre_process_yuv422_scale_U8toF16")
#define VX_KERNEL_NAME_PRE_PROCESS_YUV422_SCALE_U8TOI16    CVIVANTE_NAMESPACE("evis.pre_process_yuv422_scale_U8toI16")
#define VX_KERNEL_NAME_PRE_PROCESS_YUV422_SCALE_U8TOU8     CVIVANTE_NAMESPACE("evis.pre_process_yuv422_scale_U8toU8")
#define VX_KERNEL_NAME_PRE_PROCESS_YUV422_SCALE_U8TOI8     CVIVANTE_NAMESPACE("evis.pre_process_yuv422_scale_U8toI8")
#define VX_KERNEL_NAME_PRE_PROCESS_YUV422_COPY_U8TOU8      CVIVANTE_NAMESPACE("evis.pre_process_yuv422_copy_U8toU8")
#define VX_KERNEL_NAME_PRE_PROCESS_YUV422_COPY_U8TOI8      CVIVANTE_NAMESPACE("evis.pre_process_yuv422_copy_U8toI8")
#define VX_KERNEL_NAME_PRE_PROCESS_YUV422_COPY_U8TOI16     CVIVANTE_NAMESPACE("evis.pre_process_yuv422_copy_U8toI16")
#define VX_KERNEL_NAME_PRE_PROCESS_YUV422_COPY_U8TOF16     CVIVANTE_NAMESPACE("evis.pre_process_yuv422_copy_U8toF16")

#define KERNEL_SOURCE_1    "pre_process_yuv422_copy",
#define KERNEL_SOURCE_2    "pre_process_yuv422_scale",

typedef enum
{
    COPY = 0,
    SCALE,
    TRANS
} vsi_nn_kernel_convert_type_e;


// Add kernel hashtable here
#define HASH_PRE_PROCESS_YUV422_KEY(_input0_type, _output_type, _convert_type) \
    ((_input0_type << 24) | (_output_type << 16) | (_convert_type << 8))

#define TENSOR_PRE_PROCESS_YUV422_KERNELS(IN0_TYPE, OUT_TYPE, CONVERT_TYPE, SOURCE) \
    { HASH_PRE_PROCESS_YUV422_KEY(IN0_TYPE, OUT_TYPE, CONVERT_TYPE), \
        VX_KERNEL_NAME_PRE_PROCESS_YUV422_##CONVERT_TYPE##_##IN0_TYPE##TO##OUT_TYPE, \
        SOURCE },

static const struct {
        uint32_t key;
        char* function_name;
        const char* source_name;
    } pre_process_yuv422_map[] =
{
    TENSOR_PRE_PROCESS_YUV422_KERNELS(U8, U8,  COPY,         KERNEL_SOURCE_1)
    TENSOR_PRE_PROCESS_YUV422_KERNELS(U8, I8,  COPY,         KERNEL_SOURCE_1)
    TENSOR_PRE_PROCESS_YUV422_KERNELS(U8, I16, COPY,         KERNEL_SOURCE_1)
    TENSOR_PRE_PROCESS_YUV422_KERNELS(U8, F16, COPY,         KERNEL_SOURCE_1)
    TENSOR_PRE_PROCESS_YUV422_KERNELS(U8, U8,  SCALE,        KERNEL_SOURCE_2)
    TENSOR_PRE_PROCESS_YUV422_KERNELS(U8, I8,  SCALE,        KERNEL_SOURCE_2)
    TENSOR_PRE_PROCESS_YUV422_KERNELS(U8, F16, SCALE,        KERNEL_SOURCE_2)
    TENSOR_PRE_PROCESS_YUV422_KERNELS(U8, I16, SCALE,        KERNEL_SOURCE_2)
};

static vx_param_description_t vxPreProcessyuv422Kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
};
#define _EVIS_PRE_PROCESS_YUV422_PARAM_NUM    _cnt_of_array(vxPreProcessyuv422Kernel_param_def)

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_pre_process_yuv422_copy_initializer)
    (
    vsi_nn_kernel_node_t node,
    const vsi_nn_kernel_node_param_t * param,
    size_t param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    float       output_zp      = 0;
    float       output_scale   = 1;
    int32_t     reorder     = 0;
    int32_t     order1      = 2;
    uint32_t    width       = 0;
    uint32_t    height      = 0;
    float       bMean = 0.0f, gMean= 0.0f, rMean = 0.0f, var = 0.0f;
    float       outputScaleVar = 0.0f;
    float       bMeanScaleVarZp = 0.0f,  gMeanScaleVarZp = 0.0f,  rMeanScaleVarZp = 0.0f;

    vsi_nn_kernel_tensor_attr_t * attr[1] = { NULL };
    vsi_size_array_t * out_shape = NULL;

    VSI_UNREFERENCED(param_size);

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );

    status = vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[6], &rMean);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[7], &gMean);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[8], &bMean);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[9], &var);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[10], &reorder);
    CHECK_STATUS_FAIL_GOTO(status, OnError );

    out_shape  = attr[0]->shape;
    output_scale = 1.0f / attr[0]->scale;
    output_zp = (float)attr[0]->zero_point;
    width      = (uint32_t)(out_shape->data[0]);
    height     = (uint32_t)(out_shape->data[1]);

    if (reorder != 0)
    {
        reorder = 2;
        order1 = 0;
    }

    outputScaleVar = output_scale * var;
    bMeanScaleVarZp = output_zp - bMean * outputScaleVar;
    gMeanScaleVarZp = output_zp - gMean * outputScaleVar;
    rMeanScaleVarZp = output_zp - rMean * outputScaleVar;

    shaderParam.global_scale[0]  = 4;
    shaderParam.global_scale[1]  = 1;
    shaderParam.global_scale[2]  = 1;
    shaderParam.global_size[0]   = gpu_align_p2((width + shaderParam.global_scale[0] - 1)
        / shaderParam.global_scale[0], 4);
    shaderParam.global_size[1]   = gpu_align_p2((height + shaderParam.global_scale[1] - 1)
        / shaderParam.global_scale[1], 1);
    shaderParam.global_size[2]   = 1;

    status = vsi_nn_kernel_gpu_config( node, &shaderParam );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

    {
        gpu_dp_inst_t uniConvertInt32toUint8_2x8 = {{
                0x00003333, // TCfg
                0x00000000, // ASelt
                0x03020100, 0x00000000, // ABin
                0x00000000, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };

        gpu_dp_inst_t uniConvertYUV422toB_4x4 = {{
                0x05050505, // TCfg
                0x00000000, // ASelt
                0x00120010, 0x00560054, // ABin
                0x0a0a0a0a, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x40083ca7, 0x00000000, 0x40083ca7, 0x00000000,
                0x40083ca7, 0x00000000, 0x40083ca7, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniConvertYUV422toG_4x4 = {{
                0x29292929, // TCfg
                0x00000000, // ASelt
                0x03120310, 0x07560754, // ABin
                0x2a2a2a2a, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x36453ca7, 0x00003a81, 0x36453ca7, 0x00003a81,
                0x36453ca7, 0x00003a81, 0x36453ca7, 0x00003a81 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniConvertYUV422toR_4x4 = {{
                0x05050505, // TCfg
                0x00000000, // ASelt
                0x00320030, 0x00760074, // ABin
                0x0a0a0a0a, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x3e623ca7, 0x00000000, 0x3e623ca7, 0x00000000,
                0x3e623ca7, 0x00000000, 0x3e623ca7, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniExtractYUVtoShortSub_2x8 = {{
                0x99999999, // TCfg
                0x44444444, // ASelt
                0x03120110, 0x07160514, // ABin
                0xaaaaaaaa, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000700, // AccumType, ConstantType, and PostShift
                0x00010001, 0x00010001, 0x00010001, 0x00010001,
                0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniConvertHalftoFp16_2x8 = {{
            0x11111111, // TCfg
            0x11110000, // ASelt
            0x06040200, 0x06040200, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16 };

        status = vsi_nn_kernel_gpu_add_param(node, "uniConvertYUV422toB_4x4", &uniConvertYUV422toB_4x4);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertYUV422toG_4x4", &uniConvertYUV422toG_4x4);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertYUV422toR_4x4", &uniConvertYUV422toR_4x4);
        status |= vsi_nn_kernel_gpu_add_param(node, "rOrder", &reorder);
        status |= vsi_nn_kernel_gpu_add_param(node, "bOrder", &order1);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniExtractYUVtoShortSub_2x8", &uniExtractYUVtoShortSub_2x8);
        status |= vsi_nn_kernel_gpu_add_param(node, "outputScaleVar", &outputScaleVar);
        status |= vsi_nn_kernel_gpu_add_param(node, "bMeanScaleVarZp", &bMeanScaleVarZp);
        status |= vsi_nn_kernel_gpu_add_param(node, "gMeanScaleVarZp", &gMeanScaleVarZp);
        status |= vsi_nn_kernel_gpu_add_param(node, "rMeanScaleVarZp", &rMeanScaleVarZp);
        switch( attr[0]->dtype )
        {
        case U8:
        case I8:
        case I16:
            {
                status = vsi_nn_kernel_gpu_add_param(node, "uniExtract8Data_2x8", &uniConvertInt32toUint8_2x8);
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case F16:
            {
                status = vsi_nn_kernel_gpu_add_param(node, "uniExtract8Data_2x8", &uniConvertHalftoFp16_2x8);
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        default:
            break;
        }
        CHECK_STATUS_FAIL_GOTO(status, OnError );
    }

OnError:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }
    return status;
} /* _pre_process_yuv422_copy_initializer() */


DEF_KERNEL_INITIALIZER(_pre_process_yuv422_initializer)
    (
    vsi_nn_kernel_node_t node,
    const vsi_nn_kernel_node_param_t * param,
    size_t param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    float       output_zp     = 0;
    float       output_scale  = 1;
    int32_t     reorder       = 0;
    int32_t     order1        = 2;
    uint32_t    width         = 0;
    uint32_t    height        = 0;
    uint32_t    roi_width     = 0;
    uint32_t    roi_height    = 0;
    uint32_t    xrIntFloat_16 = 0;
    uint32_t    yrIntFloat_16 = 0;
    int32_t     xRatio        = 0;
    int32_t     yRatio        = 0;
    float       bMean = 0.0f, gMean= 0.0f, rMean = 0.0f, var = 0.0f;
    float       outputScaleVar = 0.0f;
    float       bMeanScaleVarZp = 0.0f,  gMeanScaleVarZp = 0.0f,  rMeanScaleVarZp = 0.0f;

    vsi_nn_kernel_tensor_attr_t * attr[1] = { NULL };
    vsi_size_array_t * out_shape = NULL;

    VSI_UNREFERENCED(param_size);

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[2], &xRatio);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[3], &yRatio);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[6], &rMean);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[7], &gMean);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[8], &bMean);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_float32((vsi_nn_kernel_scalar_t)param[9], &var);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[10], &reorder);
    CHECK_STATUS_FAIL_GOTO(status, OnError );

    out_shape  = attr[0]->shape;
    output_scale = 1.0f / attr[0]->scale;
    output_zp = (float)attr[0]->zero_point;
    width      = (uint32_t)(out_shape->data[0]);
    height     = (uint32_t)(out_shape->data[1]);

    if (reorder != 0)
    {
        reorder = 2;
        order1 = 0;
    }

    roi_width = (xRatio * width) >> 15;
    roi_height = (yRatio * height) >> 15;
    xrIntFloat_16 = (uint32_t)((roi_width << 16) / width + 1);
    yrIntFloat_16 = (uint32_t)((roi_height << 16) / height + 1);

    outputScaleVar = output_scale * var;
    bMeanScaleVarZp = output_zp - bMean * outputScaleVar;
    gMeanScaleVarZp = output_zp - gMean * outputScaleVar;
    rMeanScaleVarZp = output_zp - rMean * outputScaleVar;

    shaderParam.global_scale[0]  = 4;
    shaderParam.global_scale[1]  = 1;
    shaderParam.global_scale[2]  = 1;
    shaderParam.global_size[0]   = gpu_align_p2((width + shaderParam.global_scale[0] - 1)
        / shaderParam.global_scale[0], 4);
    shaderParam.global_size[1]   = gpu_align_p2((height + shaderParam.global_scale[1] - 1)
        / shaderParam.global_scale[1], 1);
    shaderParam.global_size[2]   = 1;

    status = vsi_nn_kernel_gpu_config( node, &shaderParam );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

    {
        gpu_dp_inst_t uniConvertInt32toUint8_2x8 = {{
                0x33333333, // TCfg
                0x11110000, // ASelt
                0x03020100, 0x03020100, // ABin
                0x00000000, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };

        gpu_dp_inst_t uniConvertYUV422toB_4x4 = {{
                0x05050505, // TCfg
                0x04040404, // ASelt
                0x00110000, 0x00330022, // ABin
                0x0a0a0a0a, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x40083ca7, 0x00000000, 0x40083ca7, 0x00000000,
                0x40083ca7, 0x00000000, 0x40083ca7, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniConvertYUV422toG_4x4 = {{
                0x29292929, // TCfg
                0x14141414, // ASelt
                0x05110400, 0x07330622, // ABin
                0x2a2a2a2a, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x36453ca7, 0x00003a81, 0x36453ca7, 0x00003a81,
                0x36453ca7, 0x00003a81, 0x36453ca7, 0x00003a81 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniConvertYUV422toR_4x4 = { {
                0x05050505, // TCfg
                0x04040404, // ASelt
                0x00510040, 0x00730062, // ABin
                0x0a0a0a0a, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000100, // AccumType, ConstantType, and PostShift
                0x3e623ca7, 0x00000000, 0x3e623ca7, 0x00000000,
                0x3e623ca7, 0x00000000, 0x3e623ca7, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniExtractUVtoCharSub128_2x8 = {{
                0x99999999, // TCfg
                0x44444444, // ASelt
                0x03020100, 0x07060504, // ABin
                0xaaaaaaaa, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00010001, 0x00010001, 0x00010001, 0x00010001,
                0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniExtractYtoShortSub16_4x4 = {{
                0x09090909, // TCfg
                0x04040404, // ASelt
                0x00010000, 0x00030002, // ABin
                0x0a0a0a0a, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0x00010001, 0x00000000, 0x00010001, 0x00000000,
                0x00010001, 0x00000000, 0x00010001, 0x00000000 // Constant
        },  GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniConvertHalftoFp16_2x8 = {{
                0x11111111, // TCfg
                0x11110000, // ASelt
                0x06040200, 0x06040200, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001,
                0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16 };

        status = vsi_nn_kernel_gpu_add_param(node, "uniConvertYUV422toB_4x4", &uniConvertYUV422toB_4x4);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertYUV422toG_4x4", &uniConvertYUV422toG_4x4);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniConvertYUV422toR_4x4", &uniConvertYUV422toR_4x4);
        status |= vsi_nn_kernel_gpu_add_param(node, "xrIntFloat_16", &xrIntFloat_16);
        status |= vsi_nn_kernel_gpu_add_param(node, "yrIntFloat_16", &yrIntFloat_16);
        status |= vsi_nn_kernel_gpu_add_param(node, "outputScaleVar", &outputScaleVar);
        status |= vsi_nn_kernel_gpu_add_param(node, "bMeanScaleVarZp", &bMeanScaleVarZp);
        status |= vsi_nn_kernel_gpu_add_param(node, "gMeanScaleVarZp", &gMeanScaleVarZp);
        status |= vsi_nn_kernel_gpu_add_param(node, "rMeanScaleVarZp", &rMeanScaleVarZp);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniExtractUVtoCharSub128_2x8", &uniExtractUVtoCharSub128_2x8);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniExtractYtoShortSub16_4x4", &uniExtractYtoShortSub16_4x4);
        status |= vsi_nn_kernel_gpu_add_param(node, "rOrder", &reorder);
        status |= vsi_nn_kernel_gpu_add_param(node, "bOrder", &order1);
        CHECK_STATUS_FAIL_GOTO(status, OnError );

        switch( attr[0]->dtype )
        {
        case U8:
        case I8:
        case I16:
            {
                status = vsi_nn_kernel_gpu_add_param(node, "uniExtract8Data_2x8", &uniConvertInt32toUint8_2x8);
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case F16:
            {
                status = vsi_nn_kernel_gpu_add_param(node, "uniExtract8Data_2x8", &uniConvertHalftoFp16_2x8);
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        default:
            break;
        }
    }

OnError:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }
    return status;
} /* _pre_process_yuv422_initializer() */


/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_nn_kernel_t* kernel,
    const vsi_nn_kernel_param_t * params,
    int32_t scale_x
    )
{
    vsi_nn_kernel_dtype_e input0_dtype = U8;
    vsi_nn_kernel_dtype_e output_dtype = U8;
    vsi_nn_kernel_convert_type_e convert_type = SCALE;
    vsi_status status = VSI_FAILURE;
    uint32_t key = 0;
    size_t i = 0;
    vsi_bool enable_copy  = vsi_nn_kernel_param_get_int32( params, "enable_copy" );

    VSI_UNREFERENCED(scale_x);

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if (enable_copy)
    {
        convert_type = COPY;
    }
    else
    {
        convert_type = SCALE;
    }

    key = HASH_PRE_PROCESS_YUV422_KEY( input0_dtype, output_dtype, convert_type );

    for ( i = 0; i < _cnt_of_array(pre_process_yuv422_map); i ++ )
    {
        if ( pre_process_yuv422_map[i].key == key )
        {
            break;
        }
    }
    if ( i < _cnt_of_array(pre_process_yuv422_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  pre_process_yuv422_map[i].function_name );
        kernel->info.parameters = vxPreProcessyuv422Kernel_param_def;
        kernel->info.numParams = _cnt_of_array( vxPreProcessyuv422Kernel_param_def );

        if (convert_type == COPY)
        {
            kernel->info.initialize = _pre_process_yuv422_copy_initializer;
        }
        else
        {
            kernel->info.initialize = _pre_process_yuv422_initializer;
        }
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                pre_process_yuv422_map[i].source_name );
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                pre_process_yuv422_map[i].source_name );
        status = VSI_SUCCESS;
    }
    return status;
} /* _query_kernel() */


static vsi_nn_kernel_node_t _setup
    (
    vsi_nn_graph_t              * graph,
    vsi_nn_tensor_t            ** inputs,
    size_t                        input_num,
    vsi_nn_tensor_t            ** outputs,
    size_t                        output_num,
    const vsi_nn_kernel_param_t * params,
    vsi_nn_kernel_t             * kernel
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_node_param_t tmp_params[_EVIS_PRE_PROCESS_YUV422_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_tensor_t* reshape_tensors[1] = {NULL};
    int32_t trans = 0;
    int32_t scale_x  = vsi_nn_kernel_param_get_int32( params, "scale_x" );

    VSI_UNREFERENCED(input_num);
    VSI_UNREFERENCED(output_num);

    if ( !vsi_nn_kernel_gpu_check_shape( outputs[0]->attr.size,
                outputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    status = _query_kernel( inputs, outputs, kernel, params, scale_x );
    if ( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if ( node )
        {
            uint32_t index = 2;
            int32_t scale_y  = vsi_nn_kernel_param_get_int32( params, "scale_y" );
            int32_t left     = vsi_nn_kernel_param_get_int32( params, "left" );
            int32_t top      = vsi_nn_kernel_param_get_int32( params, "top" );
            float r_mean     = vsi_nn_kernel_param_get_float32( params, "r_mean" );
            float g_mean     = vsi_nn_kernel_param_get_float32( params, "g_mean" );
            float b_mean     = vsi_nn_kernel_param_get_float32( params, "b_mean" );
            float rgb_scale  = vsi_nn_kernel_param_get_float32( params, "rgb_scale" );
            int32_t reverse  = vsi_nn_kernel_param_get_int32( params, "reverse" );
            int32_t yuv422_type = vsi_nn_kernel_param_get_int32( params, "yuv422_type" );

            /* Pass parameters to node. */
            vsi_nn_kernel_node_pack_io( tmp_params, _EVIS_PRE_PROCESS_YUV422_PARAM_NUM,
                    inputs, 1, outputs, 1 );

            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &scale_x );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &scale_y );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &left );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &top );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &r_mean );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &g_mean );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &b_mean );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &rgb_scale );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &reverse );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &trans );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &yuv422_type );
            status = vsi_nn_kernel_node_pass_param( node, tmp_params, _EVIS_PRE_PROCESS_YUV422_PARAM_NUM );
            CHECK_STATUS(status);
            vsi_nn_kernel_scalar_release( &tmp_params[2] );
            vsi_nn_kernel_scalar_release( &tmp_params[3] );
            vsi_nn_kernel_scalar_release( &tmp_params[4] );
            vsi_nn_kernel_scalar_release( &tmp_params[5] );
            vsi_nn_kernel_scalar_release( &tmp_params[6] );
            vsi_nn_kernel_scalar_release( &tmp_params[7] );
            vsi_nn_kernel_scalar_release( &tmp_params[8] );
            vsi_nn_kernel_scalar_release( &tmp_params[9] );
            vsi_nn_kernel_scalar_release( &tmp_params[10] );
            vsi_nn_kernel_scalar_release( &tmp_params[11] );
            vsi_nn_kernel_scalar_release( &tmp_params[12] );
        }
    }
    vsi_safe_release_tensor(reshape_tensors[0]);

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( pre_process_yuv422, _setup )

