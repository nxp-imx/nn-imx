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
#include "libnnext/vx_lib_nnext.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
#define KERNEL_SOURCE_1    "matrixmul"
#define KERNEL_SOURCE_2    "matrixmul_f16"
#define KERNEL_SOURCE_3    "matrixmul_transB_f16"
#define KERNEL_SOURCE_4    "matrixmul_transB_f16_mix"
#define KERNEL_SOURCE_5    "matrixmul_transB_u8_mix"
#define KERNEL_SOURCE_6    "matrixmul_u8"
#define KERNEL_SOURCE_7    "matrixmul_transA"

#define HASH_MATRIX_MUL_KEY(_input0_type, _input1_type, _output_type, _trans_a, _trans_b) \
    ((_input0_type << 24) | (_input1_type << 16) | (_output_type << 8) | (_trans_a << 4) | (_trans_b))

#define HASH_MATRIX_MUL_SH_KERNEL_NAME(SRC0_TYPE, SRC1_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.gemm_"#SRC0_TYPE#SRC1_TYPE"to"#DST_TYPE)

#define HASH_MATRIX_MUL_TRANSB_SH_KERNEL_NAME(SRC0_TYPE, SRC1_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.gemm_transb_"#SRC0_TYPE#SRC1_TYPE"to"#DST_TYPE)

#define HASH_MATRIX_MUL_TRANSA_SH_KERNEL_NAME(SRC0_TYPE, SRC1_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.gemm_transa_"#SRC0_TYPE#SRC1_TYPE"to"#DST_TYPE)

#define TENSOR_MATRIX_MUL_KERNELS(IN0_TYPE, IN1_TYPE, OUT_TYPE, SOURCE) \
    { HASH_MATRIX_MUL_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, 0, 0), \
        HASH_MATRIX_MUL_SH_KERNEL_NAME(IN0_TYPE, IN1_TYPE, OUT_TYPE), \
        SOURCE },

#define TENSOR_MATRIX_MUL_TRANSB_KERNELS(IN0_TYPE, IN1_TYPE, OUT_TYPE, SOURCE) \
    { HASH_MATRIX_MUL_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, 0, 1), \
        HASH_MATRIX_MUL_TRANSB_SH_KERNEL_NAME(IN0_TYPE, IN1_TYPE, OUT_TYPE), \
        SOURCE },

#define TENSOR_MATRIX_MUL_TRANSA_KERNELS(IN0_TYPE, IN1_TYPE, OUT_TYPE, SOURCE) \
    { HASH_MATRIX_MUL_KEY(IN0_TYPE, IN1_TYPE, OUT_TYPE, 1, 0), \
        HASH_MATRIX_MUL_TRANSA_SH_KERNEL_NAME(IN0_TYPE, IN1_TYPE, OUT_TYPE), \
        SOURCE },


static const struct {
        uint32_t key;
        char* function_name;
        const char* source_name;
    } matrix_mul_map[] =
{
    TENSOR_MATRIX_MUL_KERNELS(U8,  U8,  U8,       KERNEL_SOURCE_1)
    TENSOR_MATRIX_MUL_KERNELS(F16, U8,  F16,      KERNEL_SOURCE_1)
    TENSOR_MATRIX_MUL_KERNELS(F16, U8,  U8,       KERNEL_SOURCE_1)
    TENSOR_MATRIX_MUL_KERNELS(U8,  F16, U8,       KERNEL_SOURCE_6)
    TENSOR_MATRIX_MUL_KERNELS(I8,  F16, I8,       KERNEL_SOURCE_6)
    TENSOR_MATRIX_MUL_KERNELS(I16, F16, I16,      KERNEL_SOURCE_6)
    TENSOR_MATRIX_MUL_KERNELS(U8,  U8,  F16,      KERNEL_SOURCE_6)
    TENSOR_MATRIX_MUL_KERNELS(I8,  I8,  F16,      KERNEL_SOURCE_6)
    TENSOR_MATRIX_MUL_KERNELS(I16, I16, F16,      KERNEL_SOURCE_6)
    TENSOR_MATRIX_MUL_KERNELS(U8,  F16, F16,      KERNEL_SOURCE_6)
    TENSOR_MATRIX_MUL_KERNELS(I8,  F16, F16,      KERNEL_SOURCE_6)
    TENSOR_MATRIX_MUL_KERNELS(I16, F16, F16,      KERNEL_SOURCE_6)
    TENSOR_MATRIX_MUL_KERNELS(F16, F16, F16,      KERNEL_SOURCE_2)
    TENSOR_MATRIX_MUL_KERNELS(F16, F16, U8,       KERNEL_SOURCE_2)
    TENSOR_MATRIX_MUL_KERNELS(F32, F32, F32,      KERNEL_SOURCE_2)
    TENSOR_MATRIX_MUL_TRANSB_KERNELS(F16, F16, F16,    KERNEL_SOURCE_3)
    TENSOR_MATRIX_MUL_TRANSB_KERNELS(F16, U8,  F16,    KERNEL_SOURCE_4)
    TENSOR_MATRIX_MUL_TRANSB_KERNELS(F16, U8,  U8,     KERNEL_SOURCE_4)
    TENSOR_MATRIX_MUL_TRANSB_KERNELS(U8,  U8,  F16,    KERNEL_SOURCE_5)
    TENSOR_MATRIX_MUL_TRANSB_KERNELS(U8,  U8,  U8,     KERNEL_SOURCE_5)
    TENSOR_MATRIX_MUL_TRANSA_KERNELS(U8,  U8,  U8,     KERNEL_SOURCE_7)
    TENSOR_MATRIX_MUL_TRANSA_KERNELS(I8,  I8,  I8,     KERNEL_SOURCE_7)
    TENSOR_MATRIX_MUL_TRANSA_KERNELS(I16, I16, I16,    KERNEL_SOURCE_7)
    TENSOR_MATRIX_MUL_TRANSA_KERNELS(U8,  F16, U8,     KERNEL_SOURCE_7)
    TENSOR_MATRIX_MUL_TRANSA_KERNELS(I8,  F16, I8,     KERNEL_SOURCE_7)
    TENSOR_MATRIX_MUL_TRANSA_KERNELS(I16, F16, I16,    KERNEL_SOURCE_7)
    TENSOR_MATRIX_MUL_TRANSA_KERNELS(F16, F16, F16,    KERNEL_SOURCE_7)
};

/*
 * Kernel params
 */
static vx_param_description_t _matrix_mul_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};
#define _MATRIX_MUL_PARAM_NUM  _cnt_of_array( _matrix_mul_kernel_param_def )

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_matrix_mul_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t gpu_param = {
        3,
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        };

    vsi_nn_kernel_tensor_attr_t * attr[3] = { NULL };
    int32_t       transA = 0;
    int32_t       transB = 0;
    int32_t       width  = 0;
    int32_t       height = 0;
    int32_t       chn    = 0;

    int32_t     src0ZP     = 0;
    float       src0Scale  = 0;
    int32_t     src1ZP     = 0;
    float       src1Scale  = 0;
    int32_t     dstZP      = 0;
    float       dstScale   = 0;

    uint32_t pack_key = 0;
    int32_t  ac2zero = 0;
    int32_t  bc2zero = 0;

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", OnError );
    attr[2] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[2] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", OnError );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[3], &transA);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[4], &transB);
    CHECK_STATUS_FAIL_GOTO(status, OnError );

    src0ZP     = attr[0]->asymm.zero_point;
    src0Scale  = attr[0]->asymm.scale;
    src1ZP     = attr[1]->asymm.zero_point;
    src1Scale  = attr[1]->asymm.scale;
    dstZP      = attr[2]->asymm.zero_point;
    dstScale   = attr[2]->asymm.scale;

    if( attr[0]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        if (attr[0]->dfp.fl > 0)
        {
            src0Scale = (1.0f / ((float) (1 << attr[0]->dfp.fl)));
        }
        else
        {
            src0Scale = ((float) (1 << -attr[0]->dfp.fl));
        }
        src0ZP = 0;
    }
    else if(attr[0]->quant == VSI_NN_KERNEL_QUANT_NONE)
    {
        src0Scale = 1;
    }

    if( attr[1]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        if (attr[1]->dfp.fl > 0)
        {
            src1Scale = (1.0f / ((float) (1 << attr[1]->dfp.fl)));
        }
        else
        {
            src1Scale = ((float) (1 << -attr[1]->dfp.fl));
        }
    }
    else if(attr[1]->quant == VSI_NN_KERNEL_QUANT_NONE)
    {
        src1Scale = 1;
    }

    if( attr[2]->quant == VSI_NN_KERNEL_QUANT_DFP )
    {
        if (attr[2]->dfp.fl > 0)
        {
            dstScale = (float)(1 << attr[2]->dfp.fl);
        }
        else
        {
            dstScale = (1.0f / (float)(1 << -attr[2]->dfp.fl));
        }
        dstScale = 1.0f / dstScale;
        dstZP = 0;
    }
    else if( attr[2]->quant == VSI_NN_KERNEL_QUANT_NONE )
    {
        dstScale = 1;
    }

    if((attr[0]->shape->size > attr[1]->shape->size) ||
        (attr[0]->shape->data[2] > attr[1]->shape->data[2]
    && attr[0]->shape->size > 2 && attr[1]->shape->size > 2))
    {
        bc2zero = 1;
    }
    else if((attr[1]->shape->size > attr[0]->shape->size) ||
        (attr[1]->shape->data[2] > attr[0]->shape->data[2]
    && attr[0]->shape->size > 2 && attr[1]->shape->size > 2))
    {
        ac2zero = 1;
    }

    width = attr[2]->shape->data[0];
    height = attr[2]->shape->data[1];
    chn = attr[2]->shape->size > 2 ? attr[2]->shape->data[2] : 1;

    gpu_param.global_scale[0]  = 4;
    gpu_param.global_scale[1]  = 4;
    gpu_param.global_scale[2]  = 1;

    gpu_param.global_size[0]   = gpu_align_p2((width + gpu_param.global_scale[0] - 1)
                                        / gpu_param.global_scale[0], 4);
    gpu_param.global_size[1]   = height;
    gpu_param.global_size[2]   = chn;

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

#define _PACK_SELECT_KEY( IN0_TYPE, IN1_TYPE, OUT_TYPE, TRANSA, TRANSB)    \
        ((IN0_TYPE << 24) | (IN1_TYPE << 16) | (OUT_TYPE << 8) | (TRANSA << 4) | (TRANSB))

    pack_key = _PACK_SELECT_KEY( attr[0]->dtype, attr[1]->dtype, attr[2]->dtype, transA, transB);
    {
        gpu_dp_inst_t uniU8SubZptoFp16_dp2x8 = {{
            0x99999999, // TCfg
            0x44444444, // ASelt
            0x03020100, 0x07060504, // ABin
            0xaaaaaaaa, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001 // Constant
        }, GPU_DP_TYPE_16 };
        gpu_dp_inst_t uniFp16MulFp16AddtoFp32_dp8x2 = {{
            0x00005555, // TCfg
            0x00000000, // ASelt
            0x76543210, 0x00000000, // ABin
            0x00005555, // BSelt
            0x76543210, 0x00000000, // BBin
            0x00000400, // AccumType, ConstantType, and PostShift
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };
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
        gpu_dp_inst_t uniConvertUint8SubZpToFp32_4x4 = {{
            0x05050505, // TCfg
            0x04040404, // ASelt
            0x00010000, 0x00030002, // ABin
            0x0a0a0a0a, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00002100, // AccumType, ConstantType, and PostShift
            0xbc003c00, 0x00000000, 0xbc003c00, 0x00000000, 0xbc003c00, 0x00000000, 0xbc003c00, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };

        float scaleIn1divOut = src1Scale / dstScale;
        float inScaleMul = src0Scale * src1Scale;
        float reScaleOut = 1 / dstScale;
        float inScaledivOut = inScaleMul / dstScale;

        switch( pack_key )
        {
        case _PACK_SELECT_KEY( U8, U8, F16, 0, 1 ):
        case _PACK_SELECT_KEY( U8, U8, U8, 0, 1 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniU8SubZptoFp16_dp2x8", &uniU8SubZptoFp16_dp2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniFp16MulFp16AddtoFp32_dp8x2", &uniFp16MulFp16AddtoFp32_dp8x2 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertInt32toUint8_2x8", &uniConvertInt32toUint8_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node, "input1_ZP", &src0ZP );
                status |= vsi_nn_kernel_gpu_add_param( node, "input2_ZP", &src1ZP );
                status |= vsi_nn_kernel_gpu_add_param( node, "inScaleMul", &inScaleMul );
                status |= vsi_nn_kernel_gpu_add_param( node, "inScaledivOut", &inScaledivOut );
                status |= vsi_nn_kernel_gpu_add_param( node, "output_ZP", &dstZP );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( F16, U8, F16, 0, 1 ):
        case _PACK_SELECT_KEY( F16, U8, U8, 0, 1 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniU8SubZptoFp16_dp2x8", &uniU8SubZptoFp16_dp2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniFp16MulFp16AddtoFp32_dp8x2", &uniFp16MulFp16AddtoFp32_dp8x2 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertInt32toUint8_2x8", &uniConvertInt32toUint8_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node, "input2Scale", &src1Scale );
                status |= vsi_nn_kernel_gpu_add_param( node, "input2_ZP", &src1ZP );
                status |= vsi_nn_kernel_gpu_add_param( node, "scaleIn2divOut", &scaleIn1divOut );
                status |= vsi_nn_kernel_gpu_add_param( node, "output_ZP", &dstZP );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( F16, F16, F16, 0, 1 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniFp16MulFp16AddtoFp32_dp8x2", &uniFp16MulFp16AddtoFp32_dp8x2 );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( F16, F16, U8, 0, 0 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertInt32toUint8_2x8", &uniConvertInt32toUint8_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node, "outputScale", &reScaleOut );
                status |= vsi_nn_kernel_gpu_add_param( node, "output_ZP", &dstZP );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( U8,  U8,  U8,  0, 0 ):
        case _PACK_SELECT_KEY( F16, U8,  F16, 0, 0 ):
        case _PACK_SELECT_KEY( F16, U8,  U8,  0, 0 ):
        case _PACK_SELECT_KEY( U8,  U8,  U8,  1, 0 ):
        case _PACK_SELECT_KEY( I8,  I8,  I8,  1, 0 ):
        case _PACK_SELECT_KEY( I16, I16, I16, 1, 0 ):
        case _PACK_SELECT_KEY( F16, F16, F16, 1, 0 ):
        case _PACK_SELECT_KEY( U8,  F16,  U8, 1, 0 ):
        case _PACK_SELECT_KEY( I8,  F16,  I8, 1, 0 ):
        case _PACK_SELECT_KEY( I16, F16, I16, 1, 0 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertUint8SubZpToFp32_4x4", &uniConvertUint8SubZpToFp32_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertInt32toUint8_2x8", &uniConvertInt32toUint8_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node, "input1Scale", &src0Scale );
                status |= vsi_nn_kernel_gpu_add_param( node, "input1_ZP", &src0ZP );
                status |= vsi_nn_kernel_gpu_add_param( node, "input2Scale", &src1Scale );
                status |= vsi_nn_kernel_gpu_add_param( node, "input2_ZP", &src1ZP );
                status |= vsi_nn_kernel_gpu_add_param( node, "outputScale", &reScaleOut );
                status |= vsi_nn_kernel_gpu_add_param( node, "output_ZP", &dstZP );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( U8,  F16, U8,  0, 0 ):
        case _PACK_SELECT_KEY( I8,  F16, I8,  0, 0 ):
        case _PACK_SELECT_KEY( I16, F16, I16, 0, 0 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertUint8SubZpToFp32_4x4", &uniConvertUint8SubZpToFp32_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertInt32toUint8_2x8", &uniConvertInt32toUint8_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node, "input1Scale", &src0Scale );
                status |= vsi_nn_kernel_gpu_add_param( node, "input1_ZP", &src0ZP );
                status |= vsi_nn_kernel_gpu_add_param( node, "outputScale", &reScaleOut );
                status |= vsi_nn_kernel_gpu_add_param( node, "output_ZP", &dstZP );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( U8,  U8, F16,  0, 0 ):
        case _PACK_SELECT_KEY( I8,  I8, F16,  0, 0 ):
        case _PACK_SELECT_KEY( I16, I16, F16, 0, 0 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertUint8SubZpToFp32_4x4", &uniConvertUint8SubZpToFp32_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertInt32toUint8_2x8", &uniConvertInt32toUint8_2x8 );
                status |= vsi_nn_kernel_gpu_add_param( node, "input1Scale", &src0Scale );
                status |= vsi_nn_kernel_gpu_add_param( node, "input1_ZP", &src0ZP );
                status |= vsi_nn_kernel_gpu_add_param( node, "input2Scale", &src1Scale );
                status |= vsi_nn_kernel_gpu_add_param( node, "input2_ZP", &src1ZP );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( U8,  F16, F16,  0, 0 ):
        case _PACK_SELECT_KEY( I8,  F16, F16,  0, 0 ):
        case _PACK_SELECT_KEY( I16, F16, F16,  0, 0 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertUint8SubZpToFp32_4x4", &uniConvertUint8SubZpToFp32_4x4 );
                status |= vsi_nn_kernel_gpu_add_param( node, "input1Scale", &src0Scale );
                status |= vsi_nn_kernel_gpu_add_param( node, "input1_ZP", &src0ZP );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        case _PACK_SELECT_KEY( F32, F32, F32, 0, 0 ):
            {
                status = vsi_nn_kernel_gpu_add_param( node,
                        "uniConvertInt32toUint8_2x8", &uniConvertInt32toUint8_2x8 );
                CHECK_STATUS_FAIL_GOTO(status, OnError );
            }
            break;
        default:
            break;
        }
        status = vsi_nn_kernel_gpu_add_param( node, "ac2zero", &ac2zero );
        status |= vsi_nn_kernel_gpu_add_param( node, "bc2zero", &bc2zero );
        CHECK_STATUS_FAIL_GOTO(status, OnError );
    }
#undef _PACK_SELECT_KEY

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
    return status;
} /* _matrix_mul_initializer() */

/*
 * Query kernel
 */
static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_nn_kernel_t* kernel,
    int32_t transa,
    int32_t transb
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_dtype_e input0_dtype = U8;
    vsi_nn_kernel_dtype_e input1_dtype = U8;
    vsi_nn_kernel_dtype_e output_dtype = U8;
    uint32_t key = 0;
    int i = 0;

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    input1_dtype = vsi_nn_kernel_map_dtype( inputs[1]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = HASH_MATRIX_MUL_KEY( input0_dtype, input1_dtype, output_dtype, transa, transb );

    for( i = 0; i < _cnt_of_array(matrix_mul_map); i ++ )
    {
        if( matrix_mul_map[i].key == key )
        {
            break;
        }
    }
    if( i < _cnt_of_array(matrix_mul_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  matrix_mul_map[i].function_name );
        kernel->info.parameters = _matrix_mul_kernel_param_def;
        kernel->info.numParams = _cnt_of_array( _matrix_mul_kernel_param_def );
        kernel->info.initialize = _matrix_mul_initializer;

        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                matrix_mul_map[i].source_name );
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                matrix_mul_map[i].source_name );
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
    vsi_nn_kernel_node_param_t tmp_params[_MATRIX_MUL_PARAM_NUM] = { NULL };
    vsi_nn_kernel_node_t node = NULL;
    int32_t transposeA  = vsi_nn_kernel_param_get_int32( params, "transposeA" );
    int32_t transposeB  = vsi_nn_kernel_param_get_int32( params, "transposeB" );
    int32_t adjointA  = vsi_nn_kernel_param_get_int32( params, "adjointA" );
    int32_t adjointB  = vsi_nn_kernel_param_get_int32( params, "adjointB" );
    uint32_t M = inputs[0]->attr.size[1];
    uint32_t K = inputs[0]->attr.size[0];
    uint32_t N = inputs[1]->attr.size[0];

    if((inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT32
        && inputs[1]->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT32
        && outputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT32)
        &&(M % 4 != 0 || K % 4 != 0 || N %4 != 0))
    {
        return NULL;
    }

    if( !vsi_nn_kernel_gpu_check_shape( (int32_t*)outputs[0]->attr.size,
                outputs[0]->attr.dim_num ) )
    {
        return NULL;
    }

    if(transposeA)
    {
        K = inputs[0]->attr.size[1];
        M = inputs[0]->attr.size[0];
    }
    else if(transposeB)
    {
        N = inputs[1]->attr.size[1];
    }

    status = _query_kernel( inputs, outputs, kernel, transposeA, transposeB );
    if( VSI_SUCCESS == status)
    {
        node = vsi_nn_kernel_create_node( graph, kernel );
        if( node )
        {
            uint32_t index = 3;
            /* Pass parameters to node. */
            vsi_nn_kernel_node_pack_io( tmp_params, _MATRIX_MUL_PARAM_NUM,
                    inputs, 2, outputs, 1 );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &transposeA );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &transposeB );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &adjointA );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &adjointB );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &M );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &K );
            tmp_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &N );
            status = vsi_nn_kernel_node_pass_param( node, tmp_params, _MATRIX_MUL_PARAM_NUM );
            CHECK_STATUS(status);
            vsi_nn_kernel_scalar_release( &tmp_params[3] );
            vsi_nn_kernel_scalar_release( &tmp_params[4] );
            vsi_nn_kernel_tensor_release( &tmp_params[5] );
            vsi_nn_kernel_tensor_release( &tmp_params[6] );
            vsi_nn_kernel_tensor_release( &tmp_params[7] );
            vsi_nn_kernel_tensor_release( &tmp_params[8] );
            vsi_nn_kernel_tensor_release( &tmp_params[9] );
            {
                // Set default border mode.
                vx_border_t border;
                border.mode = VX_BORDER_CONSTANT;
                border.constant_value.U8 = 0;
                border.constant_value.S16 = 0;
                border.constant_value.U16 = 0;
                border.constant_value.S32 = 0;
                border.constant_value.U32 = 0;
                if(inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_UINT8)
                {
                    border.constant_value.U8 = (vx_uint8)inputs[0]->attr.dtype.zero_point;
                }
                status = vxSetNodeAttribute( (vx_node)node, VX_NODE_BORDER, &border, sizeof(border) );
                CHECK_STATUS(status);
            }
        }
    }
    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( matrixmul, _setup )

