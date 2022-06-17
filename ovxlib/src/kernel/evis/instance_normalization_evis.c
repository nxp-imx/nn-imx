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
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */

typedef enum
{
    INTERNAL_KERNEL_SUMS,
    INTERNAL_KERNEL_NORM,
} _internal_kernel_e;

#define KERNEL_SOURCE_0    "instance_normalization_0"
#define KERNEL_SOURCE_1    "instance_normalization_1"
#define KERNEL_SOURCE_2    "instance_normalization_2"
#define KERNEL_SOURCE_3    "instance_normalization_3"

#define HASH_INSTANCENORM_SUMS_SH_KERNEL_NAME(SRC0_TYPE) \
    CVIVANTE_NAMESPACE("evis.instance_norm_sums_"#SRC0_TYPE)

#define HASH_INSTANCENORM_SUMS_SH_KERNEL_2D_NAME(SRC0_TYPE) \
    CVIVANTE_NAMESPACE("evis.instance_norm_sums_"#SRC0_TYPE"_2D")

#define HASH_INSTANCENORM_SCALE_SH_KERNEL_NAME(SRC0_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.instance_norm_"#SRC0_TYPE"_F32to"#DST_TYPE)

#define HASH_INSTANCENORM_SCALE_SH_KERNEL_2D_NAME(SRC0_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.instance_norm_"#SRC0_TYPE"_F32to"#DST_TYPE"_2D")

// Add kernel hashtable here
#define HASH_INSTANCENORM_SUMS_KEY(_input0_type, _output_type, _reshape_flag) \
    ((_input0_type << 24) | (_output_type << 16) | (_reshape_flag << 8))

#define TENSOR_INSTANCENORM_SUMS_KERNELS_3D(IN0_TYPE, OUT_TYPE, SOURCE) \
    { HASH_INSTANCENORM_SUMS_KEY(IN0_TYPE, OUT_TYPE, 0), \
        HASH_INSTANCENORM_SUMS_SH_KERNEL_NAME(IN0_TYPE), \
        SOURCE },

#define TENSOR_INSTANCENORM_SUMS_KERNELS_2D(IN0_TYPE, OUT_TYPE, SOURCE) \
    { HASH_INSTANCENORM_SUMS_KEY(IN0_TYPE, OUT_TYPE, 1), \
        HASH_INSTANCENORM_SUMS_SH_KERNEL_2D_NAME(IN0_TYPE), \
        SOURCE },

// normalization
#define HASH_INSTANCENORM_KEY(_input0_type, _input1_type, _output_type, _reshape_flag) \
    ((_input0_type << 24) | (_input1_type << 16) | (_output_type << 8) | (_reshape_flag << 4))

#define TENSOR_INSTANCENORM_SCALE_KERNELS_3D(IN0_TYPE, OUT_TYPE, SOURCE) \
    { HASH_INSTANCENORM_KEY(IN0_TYPE, F32, OUT_TYPE, 0), \
        HASH_INSTANCENORM_SCALE_SH_KERNEL_NAME(IN0_TYPE, OUT_TYPE), \
        SOURCE },

#define TENSOR_INSTANCENORM_SCALE_KERNELS_2D(IN0_TYPE, OUT_TYPE, SOURCE) \
    { HASH_INSTANCENORM_KEY(IN0_TYPE, F32, OUT_TYPE, 1), \
        HASH_INSTANCENORM_SCALE_SH_KERNEL_2D_NAME(IN0_TYPE, OUT_TYPE), \
        SOURCE },

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _instancenorm_sums_kernel_map[] =
{
    // Register kernel here
    TENSOR_INSTANCENORM_SUMS_KERNELS_3D( I8,   F32, KERNEL_SOURCE_0 )
    TENSOR_INSTANCENORM_SUMS_KERNELS_2D( I8,   F32, KERNEL_SOURCE_0 )
    TENSOR_INSTANCENORM_SUMS_KERNELS_3D( U8,   F32, KERNEL_SOURCE_0 )
    TENSOR_INSTANCENORM_SUMS_KERNELS_2D( U8,   F32, KERNEL_SOURCE_0 )
    TENSOR_INSTANCENORM_SUMS_KERNELS_3D( I16,  F32, KERNEL_SOURCE_2 )
    TENSOR_INSTANCENORM_SUMS_KERNELS_2D( I16,  F32, KERNEL_SOURCE_2 )
    TENSOR_INSTANCENORM_SUMS_KERNELS_3D( F16,  F32, KERNEL_SOURCE_2 )
    TENSOR_INSTANCENORM_SUMS_KERNELS_2D( F16,  F32, KERNEL_SOURCE_2 )
    TENSOR_INSTANCENORM_SUMS_KERNELS_3D( BF16, F32, KERNEL_SOURCE_3 )
    TENSOR_INSTANCENORM_SUMS_KERNELS_2D( BF16, F32, KERNEL_SOURCE_3 )
};

static const _kernel_map_type _instancenorm_kernel_map[] =
{
    // Register kernel here

    TENSOR_INSTANCENORM_SCALE_KERNELS_3D( U8,  U8,  KERNEL_SOURCE_0 )
    TENSOR_INSTANCENORM_SCALE_KERNELS_2D( U8,  U8,  KERNEL_SOURCE_0 )
    TENSOR_INSTANCENORM_SCALE_KERNELS_3D( I8,  I8,  KERNEL_SOURCE_0 )
    TENSOR_INSTANCENORM_SCALE_KERNELS_2D( I8,  I8,  KERNEL_SOURCE_0 )

    TENSOR_INSTANCENORM_SCALE_KERNELS_3D( U8,  F16, KERNEL_SOURCE_1 )
    TENSOR_INSTANCENORM_SCALE_KERNELS_2D( U8,  F16, KERNEL_SOURCE_1 )
    TENSOR_INSTANCENORM_SCALE_KERNELS_3D( I8,  F16, KERNEL_SOURCE_1 )
    TENSOR_INSTANCENORM_SCALE_KERNELS_2D( I8,  F16, KERNEL_SOURCE_1 )

    TENSOR_INSTANCENORM_SCALE_KERNELS_3D( I16, I16, KERNEL_SOURCE_2 )
    TENSOR_INSTANCENORM_SCALE_KERNELS_2D( I16, I16, KERNEL_SOURCE_2 )
    TENSOR_INSTANCENORM_SCALE_KERNELS_3D( F16, F16, KERNEL_SOURCE_2 )
    TENSOR_INSTANCENORM_SCALE_KERNELS_2D( F16, F16, KERNEL_SOURCE_2 )

    TENSOR_INSTANCENORM_SCALE_KERNELS_3D( I16, F16, KERNEL_SOURCE_2 )
    TENSOR_INSTANCENORM_SCALE_KERNELS_2D( I16, F16, KERNEL_SOURCE_2 )
    TENSOR_INSTANCENORM_SCALE_KERNELS_3D( F16, I16, KERNEL_SOURCE_2 )
    TENSOR_INSTANCENORM_SCALE_KERNELS_2D( F16, I16, KERNEL_SOURCE_2 )
    TENSOR_INSTANCENORM_SCALE_KERNELS_3D( F16, I8,  KERNEL_SOURCE_2 )
    TENSOR_INSTANCENORM_SCALE_KERNELS_2D( F16, I8,  KERNEL_SOURCE_2 )
    TENSOR_INSTANCENORM_SCALE_KERNELS_3D( F16, U8,  KERNEL_SOURCE_2 )
    TENSOR_INSTANCENORM_SCALE_KERNELS_2D( F16, U8,  KERNEL_SOURCE_2 )

    TENSOR_INSTANCENORM_SCALE_KERNELS_3D( BF16, BF16, KERNEL_SOURCE_3 )
    TENSOR_INSTANCENORM_SCALE_KERNELS_2D( BF16, BF16, KERNEL_SOURCE_3 )
};

/*
 * Kernel params
 */
static vx_param_description_t _instancenorm_sums_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _INSTANCENORM_SUMS_PARAM_NUM  _cnt_of_array( _instancenorm_sums_kernel_param_def )

static vx_param_description_t _instancenorm_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _INSTANCENORM_PARAM_NUM  _cnt_of_array( _instancenorm_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_instancenorm_sums_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vsi_nn_kernel_tensor_attr_t* attr[2] = {NULL, NULL};
    vsi_size_array_t * input_shape = NULL;
    int32_t rs_flag = 0;
    int32_t width = 0;
    int32_t height = 0;
    int32_t chn = 0;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", OnError );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[3], &rs_flag);
    CHECK_STATUS_FAIL_GOTO(status, OnError );

    input_shape  = attr[0]->shape;

    width = (int32_t)(input_shape->data[0]);
    height = (int32_t)(input_shape->data[1]);
    chn = (int32_t)(attr[1]->shape->data[1]);
    if (rs_flag)
    {
        height = height / chn;
    }

    shaderParam.global_scale[0]  = 1;
    shaderParam.global_scale[1]  = 1;
    shaderParam.global_scale[2]  = 1;
    shaderParam.local_size[0]  = 16;
    shaderParam.local_size[1]  = 1;
    shaderParam.local_size[2]  = 1;

    if (attr[0]->dtype == I8 || attr[0]->dtype == U8)
    {
        shaderParam.global_size[0]   = (width + 255) / 256 * 16;
    }
    else if (attr[0]->dtype == I16 || attr[0]->dtype == F16 || attr[0]->dtype == BF16)
    {
        shaderParam.global_size[0]   = (width + 127) / 128 * 16;
    }
    shaderParam.global_size[1]   = chn;
    shaderParam.global_size[2]   = 1;

    status = vsi_nn_kernel_gpu_config( node, &shaderParam );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

    if (attr[0]->dtype == U8 || attr[0]->dtype == I8)
    {
        gpu_dp_inst_t uniSumX_16x1 = {{
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x76543210, 0xfedcba98, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniSumX2_16x1 = {{
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x76543210, 0xfedcba98, // ABin
            0x55555555, // BSelt
            0x76543210, 0xfedcba98, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        status  = vsi_nn_kernel_gpu_add_param(node, "uniSumX_16x1", &uniSumX_16x1);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniSumX2_16x1", &uniSumX2_16x1);
        CHECK_STATUS_FAIL_GOTO(status, OnError );
    }
    else if (attr[0]->dtype == I16 || attr[0]->dtype == F16)
    {
        gpu_dp_inst_t uniSum_X_X2_8x2 = {{
            0x55555555, // TCfg
            0x00000000, // ASelt
            0x76543210, 0x76543210, // ABin
            0x0000aaaa, // BSelt
            0x00000000, 0x76543210, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x3c003c00, 0x3c003c00, 0x3c003c00, 0x3c003c00,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        status  = vsi_nn_kernel_gpu_add_param(node, "uniSum_X_X2_8x2", &uniSum_X_X2_8x2);
        CHECK_STATUS_FAIL_GOTO(status, OnError );
    }
    else if (attr[0]->dtype == BF16)
    {
        gpu_dp_inst_t uniConvBF16toF32_Part0_2x8 = {{
                0x11111111, // TCfg
                0x01010101, // ASelt
                0x01050004, 0x03070206, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001,
                0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniConvBF16toF32_Part1_2x8 = {{
                0x11111111, // TCfg
                0x01010101, // ASelt
                0x05050404, 0x07070606, // ABin
                0x22222222, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000600, // AccumType, ConstantType, and PostShift
                0x00000001, 0x00000001, 0x00000001, 0x00000001,
                0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16};
        status = vsi_nn_kernel_gpu_add_param(node, "uniConvBF16toF32_Part0_2x8", &uniConvBF16toF32_Part0_2x8);
        status |= vsi_nn_kernel_gpu_add_param(node, "uniConvBF16toF32_Part1_2x8", &uniConvBF16toF32_Part1_2x8);
        CHECK_STATUS_FAIL_GOTO(status, OnError );
    }

    status = vsi_nn_kernel_gpu_add_param(node, "width", &width);
    status |= vsi_nn_kernel_gpu_add_param(node, "height", &height);
    CHECK_STATUS_FAIL_GOTO(status, OnError );

OnError:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }
    if (attr[1])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[1] );
        attr[1] = NULL;
    }

    return status;
}

DEF_KERNEL_INITIALIZER(_instancenorm_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t shaderParam = {
        3,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in thread
        {0, 0, 0}}; // globalWorkSize: image size in thread

    vsi_nn_kernel_tensor_attr_t* attr[4] = {NULL, NULL};
    vsi_size_array_t * input_shape = NULL;
    float output_scale = 1;
    float output_zp = 0;
    float inv_multiplier = 0;
    vx_uint32 group_num = 0;
    vx_int32 height = 0, width = 0, chn = 0;
    int32_t rs_flag = 0;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[2] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", OnError );
    attr[2] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[3] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", OnError );
    attr[3] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[4] );
    CHECK_PTR_FAIL_GOTO( attr[3], "Create tensor attr buffer fail.", OnError );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[6], &rs_flag);
    CHECK_STATUS_FAIL_GOTO(status, OnError );

    input_shape  = attr[0]->shape;
    output_scale = 1.0f / attr[3]->scale;
    output_zp = (float)attr[3]->zero_point;

    width = (int32_t)(input_shape->data[0]);
    height = (int32_t)(input_shape->data[1]);
    chn = (int32_t)(attr[2]->shape->data[1]);
    if (rs_flag)
    {
        height = height / chn;
    }

    inv_multiplier = (float)(1.0 / (width * height));

    group_num = (width + 255) / 256;

    shaderParam.global_scale[0]  = 16;
    if (attr[0]->dtype == I16 || attr[0]->dtype == F16 || attr[0]->dtype == BF16)
    {
        shaderParam.global_scale[0]  = 8;
        group_num = (width + 127) / 128;
    }

    shaderParam.global_scale[1]  = 1;
    shaderParam.global_scale[2]  = 1;
    shaderParam.global_size[0]   = gpu_align_p2((width + shaderParam.global_scale[0] - 1)
                                        / shaderParam.global_scale[0], 4);
    shaderParam.global_size[1]   = (chn + shaderParam.global_scale[1] - 1)
        / shaderParam.global_scale[1];
    shaderParam.global_size[2]   = 1;

    status = vsi_nn_kernel_gpu_config( node, &shaderParam );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

    {
        gpu_dp_inst_t uniDataToFP32_0_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000300, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniDataToFP32_1_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00050004, 0x00070006, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000300, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniDataToFP32_2_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00090008, 0x000b000a, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000300, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniDataToFP32_3_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x000d000c, 0x000f000e, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000300, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000000, 0x00000001, 0x00000000,
            0x00000001, 0x00000000, 0x00000001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniExtractHalf8_2x8 = {{
            0x11111111, // TCfg
            0x11110000, // ASelt
            0x06040200, 0x06040200, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00,
            0x00003c00, 0x00003c00, 0x00003c00, 0x00003c00 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniExtractInteger_2x8 = {{
            0x33333333, // TCfg
            0x11110000, // ASelt
            0x03020100, 0x03020100, // ABin
            0x00000000, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniConvBF16toF32_Part0_2x8 = {{
            0x11111111, // TCfg
            0x01010101, // ASelt
            0x01050004, 0x03070206, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniConvBF16toF32_Part1_2x8 = {{
            0x11111111, // TCfg
            0x01010101, // ASelt
            0x05050404, 0x07070606, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16};
        gpu_dp_inst_t uniExtractOddData_2x8 = {{
            0x11111111, // TCfg
            0x11110000, // ASelt
            0x07050301, 0x07050301, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16};

        uint32_t pack_key      = 0;
#define _PACK_SELECT_KEY( IN0_TYPE, OUT_TYPE )    \
        (IN0_TYPE | (OUT_TYPE << 16))

        pack_key = _PACK_SELECT_KEY( attr[0]->dtype, attr[3]->dtype );

        status  = vsi_nn_kernel_gpu_add_param(node, "height", &height);
        status |= vsi_nn_kernel_gpu_add_param(node, "inv_multiplier", &inv_multiplier);
        status |= vsi_nn_kernel_gpu_add_param(node, "group_num", &group_num);
        CHECK_STATUS_FAIL_GOTO(status, OnError );

        switch( pack_key )
        {
            case _PACK_SELECT_KEY( U8, F16 ):
            case _PACK_SELECT_KEY( I8, F16 ):
            case _PACK_SELECT_KEY( U8, U8 ):
            case _PACK_SELECT_KEY( I8, I8 ):
                {
                    if (attr[3]->dtype == F16)
                    {
                        status = vsi_nn_kernel_gpu_add_param(node, "uniExtract8Data_2x8",
                            &uniExtractHalf8_2x8);
                    }
                    else
                    {
                        status = vsi_nn_kernel_gpu_add_param(node, "uniExtract8Data_2x8",
                            &uniExtractInteger_2x8);
                        status |= vsi_nn_kernel_gpu_add_param(node, "output_scale", &output_scale);
                        status |= vsi_nn_kernel_gpu_add_param(node, "output_zp", &output_zp);
                    }
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniDataToFP32_0_4x4",
                        &uniDataToFP32_0_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniDataToFP32_1_4x4",
                        &uniDataToFP32_1_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniDataToFP32_2_4x4",
                        &uniDataToFP32_2_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniDataToFP32_3_4x4",
                        &uniDataToFP32_3_4x4);
                    CHECK_STATUS_FAIL_GOTO(status, OnError );
                }
                break;
            case _PACK_SELECT_KEY( I16, F16 ):
            case _PACK_SELECT_KEY( F16, F16 ):
            case _PACK_SELECT_KEY( I16, I16 ):
            case _PACK_SELECT_KEY( F16, I16 ):
            case _PACK_SELECT_KEY( F16, U8 ):
            case _PACK_SELECT_KEY( F16, I8 ):
                {
                    if (attr[3]->dtype == F16)
                    {
                        status = vsi_nn_kernel_gpu_add_param(node, "uniExtract8Data_2x8",
                            &uniExtractHalf8_2x8);
                    }
                    else
                    {
                        status = vsi_nn_kernel_gpu_add_param(node, "uniExtract8Data_2x8",
                            &uniExtractInteger_2x8);
                    }
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniDataToFP32_0_4x4",
                        &uniDataToFP32_0_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "uniDataToFP32_1_4x4",
                        &uniDataToFP32_1_4x4);
                    status |= vsi_nn_kernel_gpu_add_param(node, "output_scale", &output_scale);
                    status |= vsi_nn_kernel_gpu_add_param(node, "output_zp", &output_zp);
                    CHECK_STATUS_FAIL_GOTO(status, OnError );
                }
                break;
            case _PACK_SELECT_KEY( BF16, BF16 ):
                {
                    status  = vsi_nn_kernel_gpu_add_param( node,
                                "uniConvBF16toF32_Part0_2x8", &uniConvBF16toF32_Part0_2x8 );
                    status |= vsi_nn_kernel_gpu_add_param( node,
                                "uniConvBF16toF32_Part1_2x8", &uniConvBF16toF32_Part1_2x8 );
                    status |= vsi_nn_kernel_gpu_add_param( node,
                                "uniExtractOddData_2x8", &uniExtractOddData_2x8 );
                    CHECK_STATUS_FAIL_GOTO(status, OnError );
                }
                break;
            default:
                VSI_ASSERT( FALSE );
                return VSI_FAILURE;
        }
#undef _PACK_SELECT_KEY
    }

OnError:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }
    if (attr[1])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[1] );
        attr[1] = NULL;
    }
    if (attr[2])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[2] );
        attr[2] = NULL;
    }
    if (attr[3])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[3] );
        attr[3] = NULL;
    }

    return status;
}

/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    const uint32_t hashkey,
    _internal_kernel_e kernel_id
    /* Add extra params */
    )
{
    vsi_status status = VSI_FAILURE;
    vx_kernel_initialize_f  initializer = NULL;
    vx_param_description_t * param_def = NULL;
    const _kernel_map_type* kernel_map;
    size_t kernel_map_size = 0;
    size_t param_size = 0;
    uint32_t i = 0;

    switch ( kernel_id )
    {
        case INTERNAL_KERNEL_SUMS:
            initializer = _instancenorm_sums_initializer;
            kernel_map = _instancenorm_sums_kernel_map;
            kernel_map_size = _cnt_of_array( _instancenorm_sums_kernel_map );
            param_def = _instancenorm_sums_kernel_param_def;
            param_size = _INSTANCENORM_SUMS_PARAM_NUM;
            break;
        case INTERNAL_KERNEL_NORM:
            initializer = _instancenorm_initializer;
            kernel_map = _instancenorm_kernel_map;
            kernel_map_size = _cnt_of_array( _instancenorm_kernel_map );
            param_def = _instancenorm_kernel_param_def;
            param_size = _INSTANCENORM_PARAM_NUM;
            break;
        default:
            VSI_ASSERT( FALSE );
            return VSI_FAILURE;
    }

    for ( i = 0; i < kernel_map_size; i ++ )
    {
        if ( kernel_map[i].key == hashkey )
        {
            break;
        }
    }
    if ( i < kernel_map_size )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  kernel_map[i].function_name );
        kernel->info.parameters  = param_def;
        kernel->info.numParams   = (uint32_t)param_size;
        kernel->info.initialize  = initializer;
        // Register code source
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                kernel_map[i].source_name );
        // Register binary source
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                kernel_map[i].source_name );
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
#define INTERNAL_KERNEL_SIZE    (1)
#define MEAN_VARI_INDEX  (0)
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_node_param_t sums_node_params[_INSTANCENORM_SUMS_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_param_t node_params[_INSTANCENORM_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_t tmp_node = NULL;
    vsi_nn_kernel_node_t node = NULL;
    vsi_nn_kernel_dtype_e in0_dtype = U8;
    vsi_nn_kernel_dtype_e in1_dtype = F16;
    vsi_nn_kernel_dtype_e out_dtype = U8;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_kernel_t * ikernels[INTERNAL_KERNEL_SIZE] = { NULL };
    vsi_nn_tensor_t * tensors[INTERNAL_KERNEL_SIZE] = { NULL };
    vsi_nn_kernel_tensor_t rs_input = NULL, rs_output = NULL, rs_gamma = NULL, rs_beta = NULL;
    vsi_size_t  shape[VSI_NN_MAX_DIM_NUM] = {0};
    uint32_t hashkeys[INTERNAL_KERNEL_SIZE] = { 0 };
    uint32_t hashkey = 0;
    int32_t i = 0;
    int32_t axis[VSI_NN_MAX_DIM_NUM] = {0, 1};
    int32_t axis_num  = 2;
    int32_t new_axis[VSI_NN_MAX_DIM_NUM] = {0};
    vsi_size_t new_shape[VSI_NN_MAX_DIM_NUM] = { 1 };
    uint32_t axis_size = 0;
    uint32_t rank = outputs[0]->attr.dim_num;
    vsi_nn_tensor_t *reshape_tensor[2] = {NULL};
    float input_scale = vsi_nn_get_tensor_scale(inputs[0]);
    float eps  = vsi_nn_kernel_param_get_float32( params, "eps" ) /
                (input_scale * input_scale);
    int32_t reshape_flg  = 0;
    vsi_bool ret = FALSE;

    ret = vsi_nn_kernel_optimize_tensor_shape(
        inputs[0]->attr.size, inputs[0]->attr.dim_num,
        axis, axis_num, new_shape, &rank, new_axis, &axis_size);
    if ( ret == FALSE || axis_size > 2)
    {
        return NULL;
    }
    if (axis_size == 1)
    {
        for (i = rank; i > 1; i--)
        {
            new_shape[i] = new_shape[i - 1];
        }
        new_shape[1] = 1;
        rank ++;
    }

    reshape_tensor[0] = vsi_nn_reshape_tensor( graph,
            inputs[0], new_shape, rank );
    reshape_tensor[1] = vsi_nn_reshape_tensor( graph,
            outputs[0], new_shape, rank );

    reshape_flg = rank > 2 && new_shape[1] * new_shape[2] < GPU_TENSOR_MAX_WIDTH;
    // Check if gpu can support the size
    if ( !vsi_nn_kernel_gpu_check_shape(
        reshape_tensor[1]->attr.size, reshape_tensor[1]->attr.dim_num ) ||
        rank > 4 )
    {
        return NULL;
    }

    for( i = 0; i < INTERNAL_KERNEL_SIZE; i ++ )
    {
        ikernels[i] = vsi_nn_kernel_create( VSI_NN_KERNEL_TYPE_EVIS );
        // Assign unique_id
        ikernels[i]->unique_id = kernel->unique_id;
    }

    in0_dtype = vsi_nn_kernel_map_dtype( reshape_tensor[0]->attr.dtype.vx_type );
    in1_dtype = vsi_nn_kernel_map_dtype( inputs[2]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( reshape_tensor[1]->attr.dtype.vx_type );
    in1_dtype = in1_dtype == F16 ? F32 : in1_dtype;

    hashkeys[MEAN_VARI_INDEX]= HASH_INSTANCENORM_SUMS_KEY( in0_dtype, F32, reshape_flg );
    hashkey = HASH_INSTANCENORM_KEY( in0_dtype, in1_dtype, out_dtype, reshape_flg );

    status = _query_kernel( ikernels[MEAN_VARI_INDEX], hashkeys[MEAN_VARI_INDEX], INTERNAL_KERNEL_SUMS );
    if ( VSI_SUCCESS != status )
    {
        goto final;
    }
    status = _query_kernel( kernel, hashkey, INTERNAL_KERNEL_NORM );
    if ( VSI_SUCCESS != status )
    {
        goto final;
    }

    if (reshape_flg)
    {
        shape[0] = new_shape[0];
        shape[1] = new_shape[1] * new_shape[2];
        shape[2] = 1;
        shape[3] = reshape_tensor[0]->attr.dim_num > 3 ? new_shape[3] : 1;
        rs_input = vsi_nn_kernel_tensor_reshape( reshape_tensor[0]->t, shape, 4 );
        rs_output = vsi_nn_kernel_tensor_reshape( reshape_tensor[1]->t, shape, 4 );
    }
    else if (new_shape[0] < new_shape[1])
    {
        shape[0] = new_shape[1];
        shape[1] = new_shape[0];
        shape[2] = new_shape[2];
        shape[3] = inputs[0]->attr.dim_num > 3 ? new_shape[3] : 1;
        rs_input = vsi_nn_kernel_tensor_reshape( reshape_tensor[0]->t, shape, 4 );
        rs_output = vsi_nn_kernel_tensor_reshape( reshape_tensor[1]->t, shape, 4 );
    }
    else
    {
        shape[0] = new_shape[0];
        rs_input = vsi_nn_kernel_tensor_reshape( reshape_tensor[0]->t, new_shape, rank );
        rs_output = vsi_nn_kernel_tensor_reshape( reshape_tensor[1]->t, new_shape, rank );
    }

    memset( &attr, 0, sizeof(vsi_nn_tensor_attr_t) );
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    attr.is_const = FALSE;
    attr.vtl = TRUE;
    attr.size[0] = ((shape[0] + 255) / 256) * 4;
    if ( inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_INT16
        || inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT16
        || inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_BFLOAT16)
    {
        attr.size[0] = ((shape[0] + 127) / 128) * 4;
    }
    attr.size[1] = inputs[0]->attr.dim_num > 2 ? inputs[0]->attr.size[2] : 1;
    attr.size[2] = 1;
    attr.size[3] = inputs[0]->attr.dim_num > 3 ? inputs[0]->attr.size[3] : 1;
    attr.dim_num = 4;
    tensors[MEAN_VARI_INDEX] = vsi_nn_CreateTensor( graph, &attr );

    shape[0] = 1;
    shape[1] = rank > 2 ? new_shape[2] : 1;
    rs_beta = vsi_nn_kernel_tensor_reshape( inputs[1]->t, shape, 2 );
    rs_gamma = vsi_nn_kernel_tensor_reshape( inputs[2]->t, shape, 2 );

    // Mean Vari
    {
        tmp_node = vsi_nn_kernel_create_node( graph, ikernels[MEAN_VARI_INDEX] );
        if (tmp_node)
        {
            uint32_t index = 0;

            sums_node_params[index++] = rs_input;
            vsi_nn_kernel_node_pack_io( &sums_node_params[index],
                            _INSTANCENORM_SUMS_PARAM_NUM, NULL, 0, tensors, 1 );
            index = 2;
            sums_node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &eps );
            sums_node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &reshape_flg );

            status  = vsi_nn_kernel_node_pass_param( tmp_node, sums_node_params,
                        _INSTANCENORM_SUMS_PARAM_NUM );
            CHECK_STATUS(status);
            vsi_nn_kernel_scalar_release( &sums_node_params[2] );
            vsi_nn_kernel_scalar_release( &sums_node_params[3] );
            {
                // Set default border mode.
                vx_border_t border;
                border.mode = VX_BORDER_CONSTANT;
                border.constant_value.U16 = 0;
                status = vxSetNodeAttribute( (vx_node)tmp_node, VX_NODE_BORDER, &border, sizeof(border) );
                CHECK_STATUS(status);
            }
        }
    }

    // Nomalization
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if (node)
        {
            uint32_t index = 0;
            node_params[index++] = rs_input;
            node_params[index++] = rs_beta;
            node_params[index++] = rs_gamma;
            node_params[index++] = (vsi_nn_kernel_node_param_t)tensors[MEAN_VARI_INDEX]->t;
            node_params[index++] = rs_output;
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, F32, &eps );
            node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &reshape_flg );

            status  = vsi_nn_kernel_node_pass_param( node, node_params,
                        _INSTANCENORM_PARAM_NUM );
            CHECK_STATUS(status);
            vsi_nn_kernel_scalar_release( &node_params[5] );
            vsi_nn_kernel_scalar_release( &node_params[6] );
        }
    }

    /* Pass parameters to node. */
final:
    vsi_safe_release_tensor(reshape_tensor[0]);
    vsi_safe_release_tensor(reshape_tensor[1]);
    if (rs_beta)
    {
        vsi_nn_kernel_tensor_release( &rs_beta );
    }
    if (rs_gamma)
    {
        vsi_nn_kernel_tensor_release( &rs_gamma );
    }
    if (rs_input)
    {
        vsi_nn_kernel_tensor_release( &rs_input );
    }
    if (rs_output)
    {
        vsi_nn_kernel_tensor_release( &rs_output );
    }
    for ( i = 0; i < INTERNAL_KERNEL_SIZE; i ++ )
    {
        if ( ikernels[i] )
        {
            vsi_nn_kernel_release( &ikernels[i] );
        }
        vsi_safe_release_tensor(tensors[i]);
    }
    if (tmp_node) {vsi_nn_kernel_node_release( &tmp_node );}
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( instance_norm, _setup )
