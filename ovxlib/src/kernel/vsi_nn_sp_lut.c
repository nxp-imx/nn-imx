/****************************************************************************
*
*    Copyright (c) 2021 Vivante Corporation
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
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include "vsi_nn_context.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_types.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_node.h"
#include "vsi_nn_log.h"
#include <float.h>
#include "vsi_nn_error.h"
#include "utils/vsi_nn_dtype_util_prv.h"
#include "vsi_nn_tensor_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "kernel/vsi_nn_spinst.h"
#include "kernel/vsi_nn_sp_lut.h"
#include "utils/vsi_nn_dtype_util.h"

#if VX_STREAM_PROCESSOR_SUPPORT

vx_status vsi_nn_sp_get_max_min_by_dtype
    (
    vsi_nn_kernel_dtype_e dtype,
    float* max,
    float* min
    )
{
    float maximum = 0;
    float minimum = 0;

    switch (dtype)
    {
    case I8:
        {
        maximum = (float)((0x1 << 7) - 1);
        minimum = maximum - (float)((0x1 << 8) - 1);
        break;
        }
    case U8:
        {
        maximum = (float)((0x1 << 8) - 1);
        minimum = (float)0;
        break;
        }
    case I16:
        {
        maximum = (float)((0x1 << 15) - 1);
        minimum = maximum - (float)((0x1 << 16) - 1);
        break;
        }
    case U16:
        {
        maximum = (float)((0x1 << 16) - 1);
        minimum = (float)0;
        break;
        }
    case I32:
        {
        maximum = (float)((0x1LL << 31) - 1);
        minimum = maximum - (float)((0x1LL << 32) - 1);
        break;
        }
    case U32:
        {
        maximum = (float)((0x1LL << 32) - 1);
        minimum = (float)0;
        break;
        }
    case F32:
        {
        int32_t data = 0x7F7FFFFF;
        maximum = *(float*)(&data);
        data = 0xFF7FFFFF;
        minimum = *(float*)(&data);
        break;
        }
    case F16:
        {
        uint16_t max_val = 0x7BFF;
        uint16_t min_val = 0xFBFF;
        vsi_nn_dtype_convert_dtype_to_float(&max_val, 1, dtype, &maximum);
        vsi_nn_dtype_convert_dtype_to_float(&min_val, 1, dtype, &minimum);
        break;
        }
    case BF16:
        {
        uint16_t max_val = 0x7BFF;
        uint16_t min_val = 0xFF7F;
        vsi_nn_dtype_convert_dtype_to_float(&max_val, 1, dtype, &maximum);
        vsi_nn_dtype_convert_dtype_to_float(&min_val, 1, dtype, &minimum);
        break;
        }
    default:
        {
        break;
        }
    }

    if (max)
        *max = maximum;
    if (min)
        *min = minimum;

    return VX_SUCCESS;
}

static uint32_t _get_baseF24(uint32_t base, uint32_t expBits, vsi_bool aluPwlSignSupport)
{
    uint32_t baseF24 = 0;
    uint32_t baseU17 = 0;
    uint32_t signBits = (aluPwlSignSupport ? 1 : 0);
    uint32_t baseU9;

    if(aluPwlSignSupport)
    {
        baseU9 = base & 0x1FF;
        if(expBits == 8)
            baseF24 = base << 14;
        else if(baseU9 >= (0x1 << 8))
        {
            baseU17 = ((0xFF << 10 | ((base & 0x1FF) << signBits)) << expBits) & 0x1FFFF;
            baseF24 = (base & 0x200) << 14 | baseU17 << 5;
        }
        else
        {
            baseU17 = (((base & 0x1FF) << signBits ) << expBits) & 0x1FFFF;;
            baseF24 = (base & 0x200) << 14 | 0x1 << 22 | (baseU17  << 5);
        }
    }
    else
    {
        baseU9 = base & 0x1FF;

        if(expBits == 8)
        {
            baseF24 = base << 13;
        }
        else if(base >= (0x1 << 9))
        {
            baseU17 = ((0x7F << 10 | base )<< expBits) & 0x1FFFF;
            baseF24 = baseU17 << 5;
        }
        else
        {
            baseU17 = (base << expBits) & 0x1FFFF;
            baseF24 = (baseU17 << 5) | 0x1 << 22;
        }
    }
    return baseF24;
}

static float _SE8M15toF32(uint32_t val)
{
    float f32;
    if (((val >> 15 ) & 0xFF) == 0xFF)
        *(uint32_t *)&f32 = 0x7F7FFFFF | ((val >> 23) & 0x1) << 31;
    else
        *(uint32_t *)&f32 = (val << 8);
    return f32;
}

static int32_t _comparator(const void *pa, const void *pb)
{
    vsi_nn_sp_lut_t a = *(vsi_nn_sp_lut_t *)pa;
    vsi_nn_sp_lut_t b = *(vsi_nn_sp_lut_t *)pb;
    float diff = a.index - b.index;

    if ( diff > 0)
    {
        return 1;
    }
    else if ( diff < 0)
    {
        return -1;
    }

    return 0;
}

vsi_status vsi_nn_sp_fill_linear_exp_lut
    (
    vsi_nn_sp_lut_t *lut,
    uint32_t expBits,
    float a,
    float b,
    float *index,
    float *value
    )
{
    uint16_t base = 0;
    uint32_t baseF24 = 0;
    uint32_t i = 0;
    float  baseF32 = 0;
    vsi_bool aluPwlSignSupport = TRUE;
    float max = 0, min = 0;
    vsi_nn_kernel_dtype_e dtype = F32;

    vsi_nn_sp_get_max_min_by_dtype(dtype, &max, &min);

    for (base = 0; base < 0x400; base++)
    {
        baseF24 = 0;
        baseF32 = 0;

        baseF24 = _get_baseF24(base, expBits, aluPwlSignSupport);

        baseF32 = _SE8M15toF32(baseF24);

        baseF32 = vsi_nn_min(vsi_nn_max(baseF32, min), max);

        lut[base].index = baseF32;

        if (((baseF24 >> 15) & 0xFF) == 0xFF)
        {
            if ((baseF24 >> 23) == 0x1)
            {
                lut[base].val = 0;
            }
            else
            {
                baseF24 = 0x7F7FFF;
                baseF32 = _SE8M15toF32(baseF24);

                lut[base].val = baseF32;
            }
        }
        else
        {
            lut[base].val = expf(baseF32 * a + b);
        }
    }

    qsort(lut, VSI_NN_SP_LUT_MAX_SIZE, sizeof(lut[0]), _comparator);

    for ( i = 0; i < VSI_NN_SP_LUT_MAX_SIZE; i++)
    {
        index[i] = lut[i].index;
        value[i] = lut[i].val;
    }

    return VSI_SUCCESS;
}

vsi_status vsi_nn_sp_fill_linear_rsqrt_lut
    (
    vsi_nn_sp_lut_t *lut,
    uint32_t expBits,
    float a,
    float b,
    float *index,
    float *value
    )
{
    uint16_t   base = 0;
    uint32_t   baseF24 = 0;
    uint32_t i = 0;
    float  baseF32 = 0;
    vsi_bool aluPwlSignSupport = TRUE;
    float max = 0, min = 0;
    vsi_nn_kernel_dtype_e dtype = F32;

    vsi_nn_sp_get_max_min_by_dtype(dtype, &max, &min);

    for (base = 0; base < 0x400; base++)
    {
        baseF24 = 0;
        baseF32 = 0;

        baseF24 = _get_baseF24(base, expBits, aluPwlSignSupport);

        baseF32 = _SE8M15toF32(baseF24);

        baseF32 = vsi_nn_min(vsi_nn_max(baseF32, min), max);

        lut[base].index = baseF32;

        if (((baseF24 >> 15) & 0xFF) == 0xFF)
        {
            if ((baseF24 >> 23) == 0x1)
            {
                baseF24 = 0x7F7FFF;
                baseF32 = _SE8M15toF32(baseF24);

                lut[base].val = baseF32;
            }
            else
            {
                lut[base].val = 0;
            }
        }
        else
        {
            lut[base].val = 1.0f / sqrtf(a * baseF32 + b);
        }
    }

    qsort(lut, VSI_NN_SP_LUT_MAX_SIZE, sizeof(lut[0]), _comparator);

    for ( i = 0; i < VSI_NN_SP_LUT_MAX_SIZE; i++)
    {
        index[i] = lut[i].index;
        value[i] = lut[i].val;
    }

    return VSI_SUCCESS;
}

vsi_status vsi_nn_sp_fill_linear_sigmoid_lut
    (
    vsi_nn_sp_lut_t *lut,
    uint32_t expBits,
    float a,
    float b,
    float *index,
    float *value
    )
{
    uint16_t   base = 0;
    uint32_t   baseF24 = 0;
    uint32_t i = 0;
    float  baseF32 = 0;
    vsi_bool aluPwlSignSupport = TRUE;
    float max = 0, min = 0;
    vsi_nn_kernel_dtype_e dtype = F32;

    vsi_nn_sp_get_max_min_by_dtype(dtype, &max, &min);

    for (base = 0; base < 0x400; base++)
    {
        baseF24 = 0;
        baseF32 = 0;

        baseF24 = _get_baseF24(base, expBits, aluPwlSignSupport);

        baseF32 = _SE8M15toF32(baseF24);

        baseF32 = vsi_nn_min(vsi_nn_max(baseF32, min), max);

        lut[base].index = baseF32;

        if (((baseF24 >> 15) & 0xFF) == 0xFF)
        {
            if ((baseF24 >> 23) == 0x1)
            {
                lut[base].val = 0;
            }
            else
            {
                lut[base].val = 1;
            }
        }
        else
        {
            lut[base].val = 1.0f / (1 + expf(a * baseF32 + b));
        }
    }

    qsort(lut, VSI_NN_SP_LUT_MAX_SIZE, sizeof(lut[0]), _comparator);

    for ( i = 0; i < VSI_NN_SP_LUT_MAX_SIZE; i++)
    {
        index[i] = lut[i].index;
        value[i] = lut[i].val;
    }

    return VSI_SUCCESS;
}

vsi_status vsi_nn_sp_fill_rcp_lut
    (
    vsi_nn_sp_lut_t *lut,
    uint32_t expBits,
    float *index,
    float *value
    )
{
    uint16_t   base = 0;
    uint32_t   baseF24 = 0;
    uint32_t i = 0;
    float  baseF32 = 0;
    vsi_bool aluPwlSignSupport = TRUE;
    float max = 0, min = 0;
    vsi_nn_kernel_dtype_e dtype = F32;

    vsi_nn_sp_get_max_min_by_dtype(dtype, &max, &min);

    for (base = 0; base < 0x400; base++)
    {
        baseF24 = 0;
        baseF32 = 0;

        baseF24 = _get_baseF24(base, expBits, aluPwlSignSupport);

        baseF32 = _SE8M15toF32(baseF24);

        baseF32 = vsi_nn_min(vsi_nn_max(baseF32, min), max);

        lut[base].index = baseF32;

        lut[base].val = 1.0f / baseF32;
    }

    qsort(lut, VSI_NN_SP_LUT_MAX_SIZE, sizeof(lut[0]), _comparator);

    for ( i = 0; i < VSI_NN_SP_LUT_MAX_SIZE; i++)
    {
        index[i] = lut[i].index;
        value[i] = lut[i].val;
    }

    return VSI_SUCCESS;
}

vsi_status vsi_nn_sp_lut
    (
    vx_lut index_lut,
    vx_lut output_lut,
    vsi_nn_sp_lut_params *param
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_sp_lut_t *lut = NULL;
    float index[1024] = {0};
    float value[1024] = {0};

    if (index_lut == NULL || output_lut == NULL || param == NULL)
    {
        return VSI_FAILURE;
    }

    lut = (vsi_nn_sp_lut_t *)calloc(VSI_NN_SP_LUT_MAX_SIZE, sizeof(vsi_nn_sp_lut_t));
    CHECK_PTR_FAIL_GOTO( lut, "Create LUT buffer fail.", final );

    switch (param->act_type)
    {
        case VSI_NN_SP_ACT_LINEAR_EXP:
        {
            uint32_t expBits = 4;
            vsi_nn_sp_fill_linear_exp_lut(lut, expBits,
                param->params[0], param->params[1], index, value);
        }
        break;
        case VSI_NN_SP_ACT_LINEAR_RSQRT:
        {
            uint32_t expBits = 4;
            vsi_nn_sp_fill_linear_rsqrt_lut(lut, expBits,
                param->params[0], param->params[1], index, value);
        }
        break;
        case VSI_NN_SP_ACT_LINEAR_SIGMOID:
        {
            uint32_t expBits = 4;
            vsi_nn_sp_fill_linear_sigmoid_lut(lut, expBits,
                param->params[0], param->params[1], index, value);
        }
        break;
        case VSI_NN_SP_ACT_RCP:
        {
            uint32_t expBits = 4;
            vsi_nn_sp_fill_rcp_lut(lut, expBits, index, value);
        }
        break;
    default:
        break;
    }

    status  = vxCopyLUT(index_lut, (void*)&index, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    status |= vxCopyLUT(output_lut, (void*)&value, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
final:
    vsi_nn_safe_free(lut);

    return status;
}

#endif