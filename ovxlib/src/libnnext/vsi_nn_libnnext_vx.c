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

/* WARNING! AUTO-GENERATED, DO NOT MODIFY MANUALLY */

#include "libnnext/vsi_nn_libnnext_vx.h"

#ifndef _cnt_of_array
#define _cnt_of_array( arr )            (sizeof( arr )/sizeof( arr[0] ))
#endif

static const char vsi_nn_kernel_addn_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
__kernel void vxcAddn(\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_array_t  output)\n\
{\n\
\n\
}\n\
"; /* end of vsi_nn_kernel_addn_vx*/

static const char vsi_nn_kernel_argmax_axis0_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform int4 packedArgIdx;\n\
_viv_uniform int inputWidth;\n\
_viv_uniform VXC_512Bits uniPackedIdxAddSat_2x8;\n\
_viv_uniform VXC_512Bits uniSrcT2DstT_2x8;\n\
\n\
#define TENSOR_ARGMAX_AXIS0_8BITS(src_type_name, dst_type_name, src_type, \\\n\
                                cond_type0, cond_type1, dst_type, cond_type) \\\n\
__kernel void vxArgmax_Axis0_##src_type_name##to##dst_type_name( \\\n\
__read_only  image2d_array_t  input, \\\n\
__write_only image2d_array_t  output, \\\n\
        int  axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(0, get_global_id(0), get_global_id(1), 0); \\\n\
    src_type maxValue, maxVec, value; \\\n\
    dst_type packIdx, currIdx; \\\n\
 \\\n\
    VXC_ReadImage2DArray(maxVec, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, packIdx, packedArgIdx, 16); \\\n\
    _viv_asm(COPY, currIdx, packedArgIdx, 16); \\\n\
    coord.x += 8; \\\n\
    for (; coord.x < inputWidth; ) \\\n\
    { \\\n\
        VXC_ReadImage2DArray(value, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        coord.x += 8; \\\n\
 \\\n\
        currIdx = currIdx + 8; \\\n\
        dst_type condition; \\\n\
        cond_type0 src_condition0 = value > maxVec; \\\n\
        cond_type1 src_condition; \\\n\
        _viv_asm(COPY, src_condition, src_condition0, 8); \\\n\
        cond_type condition_tmp; \\\n\
        VXC_DP2x8(condition_tmp, src_condition, src_condition, \\\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1), uniSrcT2DstT_2x8); \\\n\
        _viv_asm(COPY, condition, condition_tmp, 16); \\\n\
        packIdx = condition ? currIdx : packIdx; \\\n\
        maxVec = max(maxVec, value); \\\n\
    } \\\n\
 \\\n\
    VXC_HorzMax3_Integer(maxValue, maxVec, VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_HorzMax3_Integer(maxValue, maxValue.s035, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0)); \\\n\
 \\\n\
    maxValue.s01234567 = maxValue.s00000000; \\\n\
 \\\n\
    cond_type1 _maxVal; \\\n\
    VXC_Clamp(_maxVal, maxVec, maxValue, maxValue, VXC_MODIFIER_CLAMP(0, 7, 0, 1)); \\\n\
    _maxVal += 1; \\\n\
 \\\n\
    VXC_DP2x8(packIdx, packIdx, _maxVal, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1), uniPackedIdxAddSat_2x8); \\\n\
 \\\n\
    VXC_HorzMin3_Integer(packIdx, packIdx, VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_HorzMin3_Integer(packIdx, packIdx.s035, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0)); \\\n\
 \\\n\
    VXC_WriteImage(output, coord.yz, packIdx, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
TENSOR_ARGMAX_AXIS0_8BITS(I8,  I16, vxc_char8,  vxc_char8,  vxc_uchar8,  vxc_short8, vxc_ushort8)\n\
TENSOR_ARGMAX_AXIS0_8BITS(I8,  U8,  vxc_char8,  vxc_char8,  vxc_uchar8,  vxc_uchar8, vxc_uchar8)\n\
TENSOR_ARGMAX_AXIS0_8BITS(U8,  I16, vxc_uchar8, vxc_char8,  vxc_uchar8,  vxc_short8, vxc_ushort8)\n\
TENSOR_ARGMAX_AXIS0_8BITS(U8,  U8,  vxc_uchar8, vxc_char8,  vxc_uchar8,  vxc_uchar8, vxc_uchar8)\n\
TENSOR_ARGMAX_AXIS0_8BITS(I16, I16, vxc_short8, vxc_short8, vxc_ushort8, vxc_short8, vxc_ushort8)\n\
TENSOR_ARGMAX_AXIS0_8BITS(I16, U8,  vxc_short8, vxc_short8, vxc_ushort8, vxc_uchar8, vxc_uchar8)\n\
\n\
#define TENSOR_ARGMAX_AXIS0_8BITS_2D(src_type_name, dst_type_name, src_type, \\\n\
                                    cond_type0, cond_type1, dst_type, cond_type) \\\n\
__kernel void vxArgmax_Axis0_##src_type_name##to##dst_type_name##_2D( \\\n\
__read_only  image2d_array_t  input, \\\n\
__write_only image2d_array_t  output, \\\n\
        int  axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(0, get_global_id(0), 0, 0); \\\n\
    src_type maxValue, maxVec, value; \\\n\
    dst_type packIdx, currIdx; \\\n\
 \\\n\
    VXC_ReadImage(maxVec, input, coord.xy, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, packIdx, packedArgIdx, 16); \\\n\
    _viv_asm(COPY, currIdx, packedArgIdx, 16); \\\n\
    coord.x += 8; \\\n\
    for (; coord.x < inputWidth; ) \\\n\
    { \\\n\
        VXC_ReadImage(value, input, coord.xy, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        coord.x += 8; \\\n\
 \\\n\
        currIdx = currIdx + 8; \\\n\
        dst_type condition; \\\n\
        cond_type0 src_condition0 = value > maxVec; \\\n\
        cond_type1 src_condition; \\\n\
        _viv_asm(COPY, src_condition, src_condition0, 8); \\\n\
        cond_type condition_tmp; \\\n\
        VXC_DP2x8(condition_tmp, src_condition, src_condition, \\\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1), uniSrcT2DstT_2x8); \\\n\
        _viv_asm(COPY, condition, condition_tmp, 16); \\\n\
        packIdx = condition ? currIdx : packIdx; \\\n\
        maxVec = max(maxVec, value); \\\n\
    } \\\n\
 \\\n\
    VXC_HorzMax3_Integer(maxValue, maxVec, VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_HorzMax3_Integer(maxValue, maxValue.s035, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0)); \\\n\
 \\\n\
    maxValue.s01234567 = maxValue.s00000000; \\\n\
 \\\n\
    cond_type1 _maxVal; \\\n\
    VXC_Clamp(_maxVal, maxVec, maxValue, maxValue, VXC_MODIFIER_CLAMP(0, 7, 0, 1)); \\\n\
    _maxVal += 1; \\\n\
 \\\n\
    VXC_DP2x8(packIdx, packIdx, _maxVal, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1), uniPackedIdxAddSat_2x8); \\\n\
 \\\n\
    VXC_HorzMin3_Integer(packIdx, packIdx, VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_HorzMin3_Integer(packIdx, packIdx.s035, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0)); \\\n\
 \\\n\
    VXC_WriteImage(output, coord.yz, packIdx, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
\n\
TENSOR_ARGMAX_AXIS0_8BITS_2D(I8,  I16, vxc_char8,  vxc_char8,  vxc_uchar8,  vxc_short8, vxc_ushort8)\n\
TENSOR_ARGMAX_AXIS0_8BITS_2D(I8,  U8,  vxc_char8,  vxc_char8,  vxc_uchar8,  vxc_uchar8, vxc_uchar8)\n\
TENSOR_ARGMAX_AXIS0_8BITS_2D(U8,  I16, vxc_uchar8, vxc_char8,  vxc_uchar8,  vxc_short8, vxc_ushort8)\n\
TENSOR_ARGMAX_AXIS0_8BITS_2D(U8,  U8,  vxc_uchar8, vxc_char8,  vxc_uchar8,  vxc_uchar8, vxc_uchar8)\n\
TENSOR_ARGMAX_AXIS0_8BITS_2D(I16, I16, vxc_short8, vxc_short8, vxc_ushort8, vxc_short8, vxc_ushort8)\n\
TENSOR_ARGMAX_AXIS0_8BITS_2D(I16, U8,  vxc_short8, vxc_short8, vxc_ushort8, vxc_uchar8, vxc_uchar8)\n\
\n\
_viv_uniform VXC_512Bits uniConvertHalf2Float32_4x4;\n\
\n\
#define TENSOR_ARGMAX_AXIS0_F16_2D(dst_type_name, dst_type) \\\n\
__kernel void vxArgmax_Axis0_F16to##dst_type_name##_2D( \\\n\
__read_only  image2d_array_t  input, \\\n\
__write_only image2d_array_t  output, \\\n\
        int  axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(0, get_global_id(0), 0, 0); \\\n\
    vxc_short8 vec0, vec1; \\\n\
    vxc_half8 src; \\\n\
    uint4 packIdx, currIdx; \\\n\
    float4 maxValue, value; \\\n\
 \\\n\
    VXC_ReadImage(vec0, input, coord.xy, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src, vec0, 16); \\\n\
    _viv_asm(COPY, packIdx, packedArgIdx, 16); \\\n\
    _viv_asm(COPY, currIdx, packedArgIdx, 16); \\\n\
    coord.x += 4; \\\n\
    VXC_DP4x4(maxValue, src, src, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertHalf2Float32_4x4); \\\n\
    for (; coord.x < inputWidth; ) \\\n\
    { \\\n\
        VXC_ReadImage(vec1, input, coord.xy, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, src, vec1, 16); \\\n\
        coord.x += 4; \\\n\
 \\\n\
       currIdx = currIdx + 4; \\\n\
        VXC_DP4x4(value, src, src, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertHalf2Float32_4x4); \\\n\
 \\\n\
        int4 condition; \\\n\
        condition = value > maxValue; \\\n\
 \\\n\
        uint4 iCondition; \\\n\
        _viv_asm(COPY, iCondition, condition, 16); \\\n\
        packIdx = iCondition ? currIdx : packIdx; \\\n\
        maxValue = value > maxValue ? value : maxValue; \\\n\
    } \\\n\
 \\\n\
    float4 maxVec; \\\n\
    float2 maxVal2 = maxValue.xy > maxValue.zw ? maxValue.xy : maxValue.zw; \\\n\
    maxVec.x = maxVal2.x > maxVal2.y ? maxVal2.x : maxVal2.y; \\\n\
    int4 condition; \\\n\
    condition = maxVec.xxxx == maxValue; \\\n\
    uint4 iCondition; \\\n\
    _viv_asm(COPY, iCondition, condition, 16); \\\n\
    iCondition += 1; \\\n\
 \\\n\
    packIdx = mad_sat(iCondition, 0xFFFFFFFF, packIdx); \\\n\
 \\\n\
    uint2 val2 = packIdx.xy < packIdx.zw ? packIdx.xy : packIdx.zw; \\\n\
    val2.x = val2.x < val2.y ? val2.x : val2.y; \\\n\
 \\\n\
    dst_type dst; \\\n\
    _viv_asm(COPY, dst, val2, 4); \\\n\
    VXC_WriteImage(output, coord.yz, dst, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
TENSOR_ARGMAX_AXIS0_F16_2D(I16, vxc_ushort8)\n\
TENSOR_ARGMAX_AXIS0_F16_2D(U8,  vxc_uchar8)\n\
\n\
\n\
#define TENSOR_ARGMAX_AXIS0_F16(dst_type_name, dst_type) \\\n\
__kernel void vxArgmax_Axis0_F16to##dst_type_name( \\\n\
__read_only  image2d_array_t  input, \\\n\
__write_only image2d_array_t  output, \\\n\
        int  axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(0, get_global_id(0), get_global_id(1), 0); \\\n\
    vxc_short8 vec0, vec1; \\\n\
    vxc_half8 src; \\\n\
    uint4 packIdx, currIdx; \\\n\
    float4 maxValue, value; \\\n\
 \\\n\
    VXC_ReadImage2DArray(vec0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src, vec0, 16); \\\n\
    _viv_asm(COPY, packIdx, packedArgIdx, 16); \\\n\
    _viv_asm(COPY, currIdx, packedArgIdx, 16); \\\n\
    coord.x += 4; \\\n\
    VXC_DP4x4(maxValue, src, src, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertHalf2Float32_4x4); \\\n\
    for (; coord.x < inputWidth; ) \\\n\
    { \\\n\
        VXC_ReadImage2DArray(vec1, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, src, vec1, 16); \\\n\
        coord.x += 4; \\\n\
 \\\n\
       currIdx = currIdx + 4; \\\n\
        VXC_DP4x4(value, src, src, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertHalf2Float32_4x4); \\\n\
 \\\n\
        int4 condition; \\\n\
        condition = value > maxValue; \\\n\
 \\\n\
        uint4 iCondition; \\\n\
        _viv_asm(COPY, iCondition, condition, 16); \\\n\
        packIdx = iCondition ? currIdx : packIdx; \\\n\
        maxValue = value > maxValue ? value : maxValue; \\\n\
    } \\\n\
 \\\n\
    float4 maxVec; \\\n\
    float2 maxVal2 = maxValue.xy > maxValue.zw ? maxValue.xy : maxValue.zw; \\\n\
    maxVec.x = maxVal2.x > maxVal2.y ? maxVal2.x : maxVal2.y; \\\n\
    int4 condition; \\\n\
    condition = maxVec.xxxx == maxValue; \\\n\
    uint4 iCondition; \\\n\
    _viv_asm(COPY, iCondition, condition, 16); \\\n\
    iCondition += 1; \\\n\
 \\\n\
    packIdx = mad_sat(iCondition, 0xFFFFFFFF, packIdx); \\\n\
 \\\n\
    uint2 val2 = packIdx.xy < packIdx.zw ? packIdx.xy : packIdx.zw; \\\n\
    val2.x = val2.x < val2.y ? val2.x : val2.y; \\\n\
 \\\n\
    dst_type dst; \\\n\
    _viv_asm(COPY, dst, val2, 4); \\\n\
    VXC_WriteImage(output, coord.yz, dst, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
TENSOR_ARGMAX_AXIS0_F16(I16, vxc_ushort8)\n\
TENSOR_ARGMAX_AXIS0_F16(U8,  vxc_uchar8)\n\
"; /* end of vsi_nn_kernel_argmax_axis0_vx*/

static const char vsi_nn_kernel_argmax_axis1_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform int4 packedArgIdx;\n\
_viv_uniform int argLenSub1;\n\
_viv_uniform VXC_512Bits uniExtractData_2x8;\n\
\n\
#define TENSOR_ARGMAX_AXIS1_16BITS(src_type_name, dst_type_name, src_type,\\\n\
                                    copy_type, axis_type, dst_type, inst_type) \\\n\
    __kernel void vxArgmax_Axis1_##src_type_name##to##dst_type_name( \\\n\
__read_only  image2d_array_t  input, \\\n\
__write_only image2d_array_t  output, \\\n\
        int  axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(get_global_id(0), argLenSub1, get_global_id(1), 0); \\\n\
    copy_type vec; \\\n\
    src_type src; \\\n\
    src_type maxVal; \\\n\
    copy_type max; \\\n\
    VXC_ReadImage2DArray(max, input, coord.xyzw, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, maxVal, max, 16); \\\n\
    axis_type axis; \\\n\
    axis_type packIdx; \\\n\
 \\\n\
    _viv_asm(COPY, axis, packedArgIdx, 16); \\\n\
    _viv_asm(COPY, packIdx, packedArgIdx, 16); \\\n\
 \\\n\
    coord.y --; \\\n\
    do \\\n\
    { \\\n\
       VXC_ReadImage2DArray(vec, input, coord.xyzw, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
       _viv_asm(COPY, src, vec, 16); \\\n\
       coord.y --; \\\n\
       packIdx --; \\\n\
       VXC_VertMax3_##inst_type(maxVal, maxVal, maxVal, src, VXC_MODIFIER_BIN(0, 7, 0)); \\\n\
       _viv_asm(COPY, max, maxVal, 16); \\\n\
       axis = (max == vec) ? packIdx : axis; \\\n\
    } while (coord.y >= 0); \\\n\
 \\\n\
    dst_type dst_axis; \\\n\
    VXC_DP2x8(dst_axis, axis, axis, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniExtractData_2x8); \\\n\
    VXC_WriteImage(output, coord.xz, dst_axis, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
TENSOR_ARGMAX_AXIS1_16BITS(F16, I16, vxc_half8,  vxc_short8, vxc_short8, vxc_short8, Half)\n\
TENSOR_ARGMAX_AXIS1_16BITS(F16, U8,  vxc_half8,  vxc_short8, vxc_short8, vxc_uchar8, Half)\n\
TENSOR_ARGMAX_AXIS1_16BITS(I16, I16, vxc_short8, vxc_short8, vxc_short8, vxc_short8, Integer)\n\
TENSOR_ARGMAX_AXIS1_16BITS(I16, U8,  vxc_short8, vxc_short8, vxc_short8, vxc_uchar8, Integer)\n\
\n\
\n\
#define TENSOR_ARGMAX_AXIS1_8BITS(src_type_name, dst_type_name, src_type, dst_type) \\\n\
__kernel void vxArgmax_Axis1_##src_type_name##to##dst_type_name( \\\n\
__read_only  image2d_array_t  input, \\\n\
__write_only image2d_array_t  output, \\\n\
        int  axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(get_global_id(0), argLenSub1, get_global_id(1), 0); \\\n\
    src_type src; \\\n\
    src_type maxVal; \\\n\
    VXC_ReadImage2DArray(maxVal, input, coord.xyzw, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    dst_type axis; \\\n\
    dst_type packIdx; \\\n\
 \\\n\
    _viv_asm(COPY, axis, packedArgIdx, 16); \\\n\
    _viv_asm(COPY, packIdx, packedArgIdx, 16); \\\n\
 \\\n\
    coord.y --; \\\n\
    do \\\n\
    { \\\n\
       VXC_ReadImage2DArray(src, input, coord.xyzw, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
       coord.y --; \\\n\
       packIdx --; \\\n\
       maxVal = max(maxVal, src); \\\n\
       dst_type condition; \\\n\
       VXC_Clamp(condition, src, maxVal, maxVal, VXC_MODIFIER_CLAMP(0, 7, 0, 1)); \\\n\
       axis = condition ? packIdx : axis; \\\n\
    } while (coord.y >= 0); \\\n\
 \\\n\
    VXC_WriteImage(output, coord.xz, axis, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
TENSOR_ARGMAX_AXIS1_8BITS(I8,  I16, vxc_char16,  vxc_short8)\n\
TENSOR_ARGMAX_AXIS1_8BITS(I8,  U8,  vxc_char16,  vxc_uchar16)\n\
TENSOR_ARGMAX_AXIS1_8BITS(U8,  I16, vxc_uchar16, vxc_short8)\n\
TENSOR_ARGMAX_AXIS1_8BITS(U8,  U8,  vxc_uchar16, vxc_uchar16)\n\
"; /* end of vsi_nn_kernel_argmax_axis1_vx*/

static const char vsi_nn_kernel_argmax_axis2_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform int4 packedArgIdx;\n\
_viv_uniform int argLenSub1;\n\
_viv_uniform VXC_512Bits uniExtractData_2x8;\n\
\n\
#define TENSOR_ARGMAX_AXIS2_16BITS(src_type_name, dst_type_name,\\\n\
                src_type, copy_type, axis_type, dst_type, inst_type) \\\n\
    __kernel void vxArgmax_Axis2_##src_type_name##to##dst_type_name( \\\n\
__read_only  image2d_array_t  input, \\\n\
__write_only image2d_array_t  output, \\\n\
        int  axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), argLenSub1, 0); \\\n\
    copy_type vec; \\\n\
    src_type src; \\\n\
    src_type maxVal; \\\n\
    copy_type max; \\\n\
    VXC_ReadImage2DArray(max, input, coord.xyzw, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, maxVal, max, 16); \\\n\
    axis_type axis; \\\n\
    axis_type packIdx; \\\n\
 \\\n\
    _viv_asm(COPY, axis, packedArgIdx, 16); \\\n\
    _viv_asm(COPY, packIdx, packedArgIdx, 16); \\\n\
 \\\n\
    coord.z --; \\\n\
    do \\\n\
    { \\\n\
       VXC_ReadImage2DArray(vec, input, coord.xyzw, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
       _viv_asm(COPY, src, vec, 16); \\\n\
       coord.z --; \\\n\
       packIdx --; \\\n\
       VXC_VertMax3_##inst_type(maxVal, maxVal, maxVal, src, VXC_MODIFIER_BIN(0, 7, 0)); \\\n\
       _viv_asm(COPY, max, maxVal, 16); \\\n\
       axis = (max == vec) ? packIdx : axis; \\\n\
    } while (coord.z >= 0); \\\n\
 \\\n\
    dst_type dst_axis; \\\n\
    VXC_DP2x8(dst_axis, axis, axis, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniExtractData_2x8); \\\n\
    VXC_WriteImage(output, coord.xy, dst_axis, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
TENSOR_ARGMAX_AXIS2_16BITS(F16, I16, vxc_half8,  vxc_short8, vxc_short8, vxc_short8, Half)\n\
TENSOR_ARGMAX_AXIS2_16BITS(F16, U8,  vxc_half8,  vxc_short8, vxc_short8, vxc_uchar8, Half)\n\
TENSOR_ARGMAX_AXIS2_16BITS(I16, I16, vxc_short8, vxc_short8, vxc_short8, vxc_short8, Integer)\n\
TENSOR_ARGMAX_AXIS2_16BITS(I16, U8,  vxc_short8, vxc_short8, vxc_short8, vxc_uchar8, Integer)\n\
\n\
#define TENSOR_ARGMAX_AXIS2_16BITS_2D(src_type_name, dst_type_name, src_type, \\\n\
                                    copy_type, axis_type, dst_type, inst_type) \\\n\
    __kernel void vxArgmax_Axis2_##src_type_name##to##dst_type_name##_2D( \\\n\
__read_only  image2d_array_t  input, \\\n\
__write_only image2d_array_t  output, \\\n\
        int  axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(get_global_id(0), argLenSub1, 0, 0); \\\n\
    copy_type vec; \\\n\
    src_type src; \\\n\
    src_type maxVal; \\\n\
    copy_type max; \\\n\
    VXC_ReadImage(max, input, coord.xy, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, maxVal, max, 16); \\\n\
    axis_type axis; \\\n\
    axis_type packIdx; \\\n\
 \\\n\
    _viv_asm(COPY, axis, packedArgIdx, 16); \\\n\
    _viv_asm(COPY, packIdx, packedArgIdx, 16); \\\n\
 \\\n\
    coord.y --; \\\n\
    do \\\n\
    { \\\n\
       VXC_ReadImage(vec, input, coord.xy, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
       _viv_asm(COPY, src, vec, 16); \\\n\
       coord.y --; \\\n\
       packIdx --; \\\n\
       VXC_VertMax3_##inst_type(maxVal, maxVal, maxVal, src, VXC_MODIFIER_BIN(0, 7, 0)); \\\n\
       _viv_asm(COPY, max, maxVal, 16); \\\n\
       axis = (max == vec) ? packIdx : axis; \\\n\
    } while (coord.y >= 0); \\\n\
 \\\n\
    dst_type dst_axis; \\\n\
    VXC_DP2x8(dst_axis, axis, axis, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniExtractData_2x8); \\\n\
    VXC_WriteImage(output, coord.xz, dst_axis, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
TENSOR_ARGMAX_AXIS2_16BITS_2D(F16, I16, vxc_half8,  vxc_short8, vxc_short8, vxc_short8, Half)\n\
TENSOR_ARGMAX_AXIS2_16BITS_2D(F16, U8,  vxc_half8,  vxc_short8, vxc_short8, vxc_uchar8, Half)\n\
TENSOR_ARGMAX_AXIS2_16BITS_2D(I16, I16, vxc_short8, vxc_short8, vxc_short8, vxc_short8, Integer)\n\
TENSOR_ARGMAX_AXIS2_16BITS_2D(I16, U8,  vxc_short8, vxc_short8, vxc_short8, vxc_uchar8, Integer)\n\
\n\
\n\
#define TENSOR_ARGMAX_AXIS2_8BITS(src_type_name, dst_type_name, src_type, dst_type) \\\n\
    __kernel void vxArgmax_Axis2_##src_type_name##to##dst_type_name( \\\n\
__read_only  image2d_array_t  input, \\\n\
__write_only image2d_array_t  output, \\\n\
        int  axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), argLenSub1, 0); \\\n\
    src_type src; \\\n\
    src_type maxVal; \\\n\
    VXC_ReadImage2DArray(maxVal, input, coord.xyzw, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    dst_type axis; \\\n\
    dst_type packIdx; \\\n\
 \\\n\
    _viv_asm(COPY, axis, packedArgIdx, 16); \\\n\
    _viv_asm(COPY, packIdx, packedArgIdx, 16); \\\n\
 \\\n\
    coord.z --; \\\n\
    do \\\n\
    { \\\n\
       VXC_ReadImage2DArray(src, input, coord.xyzw, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
       coord.z --; \\\n\
       packIdx --; \\\n\
       maxVal = max(maxVal, src); \\\n\
       dst_type condition; \\\n\
       VXC_Clamp(condition, src, maxVal, maxVal, VXC_MODIFIER_CLAMP(0, 7, 0, 1)); \\\n\
       axis = condition ? packIdx : axis; \\\n\
    } while (coord.z >= 0); \\\n\
 \\\n\
    VXC_WriteImage(output, coord.xy, axis, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
TENSOR_ARGMAX_AXIS2_8BITS(I8,  I16, vxc_char8,  vxc_short8)\n\
TENSOR_ARGMAX_AXIS2_8BITS(I8,  U8,  vxc_char8,  vxc_uchar8)\n\
TENSOR_ARGMAX_AXIS2_8BITS(U8,  I16, vxc_uchar8, vxc_short8)\n\
TENSOR_ARGMAX_AXIS2_8BITS(U8,  U8,  vxc_uchar8, vxc_uchar8)\n\
\n\
#define TENSOR_ARGMAX_AXIS2_8BITS_2D(src_type_name, dst_type_name, src_type, dst_type) \\\n\
    __kernel void vxArgmax_Axis2_##src_type_name##to##dst_type_name##_2D( \\\n\
__read_only  image2d_array_t  input, \\\n\
__write_only image2d_array_t  output, \\\n\
        int  axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(get_global_id(0), argLenSub1, 0, 0); \\\n\
    src_type src; \\\n\
    src_type maxVal; \\\n\
    VXC_ReadImage(maxVal, input, coord.xy, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    dst_type axis; \\\n\
    dst_type packIdx; \\\n\
 \\\n\
    _viv_asm(COPY, axis, packedArgIdx, 16); \\\n\
    _viv_asm(COPY, packIdx, packedArgIdx, 16); \\\n\
 \\\n\
    coord.y --; \\\n\
    do \\\n\
    { \\\n\
       VXC_ReadImage(src, input, coord.xy, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
       coord.y --; \\\n\
       packIdx --; \\\n\
       maxVal = max(maxVal, src); \\\n\
       dst_type condition; \\\n\
       VXC_Clamp(condition, src, maxVal, maxVal, VXC_MODIFIER_CLAMP(0, 7, 0, 1)); \\\n\
       axis = condition ? packIdx : axis; \\\n\
    } while (coord.y >= 0); \\\n\
 \\\n\
    VXC_WriteImage(output, coord.xz, axis, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
TENSOR_ARGMAX_AXIS2_8BITS_2D(I8,  I16, vxc_char8,  vxc_short8)\n\
TENSOR_ARGMAX_AXIS2_8BITS_2D(I8,  U8,  vxc_char8,  vxc_uchar8)\n\
TENSOR_ARGMAX_AXIS2_8BITS_2D(U8,  I16, vxc_uchar8, vxc_short8)\n\
TENSOR_ARGMAX_AXIS2_8BITS_2D(U8,  U8,  vxc_uchar8, vxc_uchar8)\n\
"; /* end of vsi_nn_kernel_argmax_axis2_vx*/

static const char vsi_nn_kernel_argmin_axis0_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform int4 packedArgIdx;\n\
_viv_uniform int inputWidth;\n\
_viv_uniform VXC_512Bits uniPackedIdxAddSat_2x8;\n\
_viv_uniform VXC_512Bits uniSrcT2DstT_2x8;\n\
\n\
#define TENSOR_ARGMIN_AXIS0_8BITS(src_type_name, dst_type_name, src_type, \\\n\
                                cond_type0, cond_type1, dst_type, cond_type) \\\n\
__kernel void vxArgmin_Axis0_##src_type_name##to##dst_type_name( \\\n\
__read_only  image2d_array_t  input, \\\n\
__write_only image2d_array_t  output, \\\n\
        int  axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(0, get_global_id(0), get_global_id(1), 0); \\\n\
    src_type minValue, minVec, value; \\\n\
    dst_type packIdx, currIdx; \\\n\
 \\\n\
    VXC_ReadImage2DArray(minVec, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, packIdx, packedArgIdx, 16); \\\n\
    _viv_asm(COPY, currIdx, packedArgIdx, 16); \\\n\
    coord.x += 8; \\\n\
    for (; coord.x < inputWidth; ) \\\n\
    { \\\n\
        VXC_ReadImage2DArray(value, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        coord.x += 8; \\\n\
 \\\n\
        currIdx = currIdx + 8; \\\n\
        dst_type condition; \\\n\
        cond_type0 src_condition0 = value < minVec; \\\n\
        cond_type1 src_condition; \\\n\
        _viv_asm(COPY, src_condition, src_condition0, 8); \\\n\
        cond_type condition_tmp; \\\n\
        VXC_DP2x8(condition_tmp, src_condition, src_condition, \\\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1), uniSrcT2DstT_2x8); \\\n\
        _viv_asm(COPY, condition, condition_tmp, 16); \\\n\
        packIdx = condition ? currIdx : packIdx; \\\n\
        minVec = min(minVec, value); \\\n\
    } \\\n\
 \\\n\
    VXC_HorzMin3_Integer(minValue, minVec, VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_HorzMin3_Integer(minValue, minValue.s035, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0)); \\\n\
 \\\n\
    minValue.s01234567 = minValue.s00000000; \\\n\
 \\\n\
    cond_type1 _minVal; \\\n\
    VXC_Clamp(_minVal, minVec, minValue, minValue, VXC_MODIFIER_CLAMP(0, 7, 0, 1)); \\\n\
    _minVal += 1; \\\n\
 \\\n\
    VXC_DP2x8(packIdx, packIdx, _minVal, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1), uniPackedIdxAddSat_2x8); \\\n\
 \\\n\
    VXC_HorzMin3_Integer(packIdx, packIdx, VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_HorzMin3_Integer(packIdx, packIdx.s035, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0)); \\\n\
 \\\n\
    VXC_WriteImage(output, coord.yz, packIdx, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
TENSOR_ARGMIN_AXIS0_8BITS(I8,  I16, vxc_char8,  vxc_char8,  vxc_uchar8,  vxc_short8, vxc_ushort8)\n\
TENSOR_ARGMIN_AXIS0_8BITS(I8,  U8,  vxc_char8,  vxc_char8,  vxc_uchar8,  vxc_uchar8, vxc_uchar8)\n\
TENSOR_ARGMIN_AXIS0_8BITS(U8,  I16, vxc_uchar8, vxc_char8,  vxc_uchar8,  vxc_short8, vxc_ushort8)\n\
TENSOR_ARGMIN_AXIS0_8BITS(U8,  U8,  vxc_uchar8, vxc_char8,  vxc_uchar8,  vxc_uchar8, vxc_uchar8)\n\
TENSOR_ARGMIN_AXIS0_8BITS(I16, I16, vxc_short8, vxc_short8, vxc_ushort8, vxc_short8, vxc_ushort8)\n\
TENSOR_ARGMIN_AXIS0_8BITS(I16, U8,  vxc_short8, vxc_short8, vxc_ushort8, vxc_uchar8, vxc_uchar8)\n\
\n\
#define TENSOR_ARGMIN_AXIS0_8BITS_2D(src_type_name, dst_type_name, src_type, \\\n\
                                    cond_type0, cond_type1, dst_type, cond_type) \\\n\
__kernel void vxArgmin_Axis0_##src_type_name##to##dst_type_name##_2D( \\\n\
__read_only  image2d_array_t  input, \\\n\
__write_only image2d_array_t  output, \\\n\
        int  axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(0, get_global_id(0), 0, 0); \\\n\
    src_type minValue, minVec, value; \\\n\
    dst_type packIdx, currIdx; \\\n\
 \\\n\
    VXC_ReadImage(minVec, input, coord.xy, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, packIdx, packedArgIdx, 16); \\\n\
    _viv_asm(COPY, currIdx, packedArgIdx, 16); \\\n\
    coord.x += 8; \\\n\
    for (; coord.x < inputWidth; ) \\\n\
    { \\\n\
        VXC_ReadImage(value, input, coord.xy, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        coord.x += 8; \\\n\
 \\\n\
        currIdx = currIdx + 8; \\\n\
        dst_type condition; \\\n\
        cond_type0 src_condition0 = value < minVec; \\\n\
        cond_type1 src_condition; \\\n\
        _viv_asm(COPY, src_condition, src_condition0, 8); \\\n\
        cond_type condition_tmp; \\\n\
        VXC_DP2x8(condition_tmp, src_condition, src_condition, \\\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1), uniSrcT2DstT_2x8); \\\n\
        _viv_asm(COPY, condition, condition_tmp, 16); \\\n\
        packIdx = condition ? currIdx : packIdx; \\\n\
        minVec = min(minVec, value); \\\n\
    } \\\n\
 \\\n\
    VXC_HorzMin3_Integer(minValue, minVec, VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_HorzMin3_Integer(minValue, minValue.s035, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0)); \\\n\
 \\\n\
    minValue.s01234567 = minValue.s00000000; \\\n\
 \\\n\
    cond_type1 _minVal; \\\n\
    VXC_Clamp(_minVal, minVec, minValue, minValue, VXC_MODIFIER_CLAMP(0, 7, 0, 1)); \\\n\
    _minVal += 1; \\\n\
 \\\n\
    VXC_DP2x8(packIdx, packIdx, _minVal, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1), uniPackedIdxAddSat_2x8); \\\n\
 \\\n\
    VXC_HorzMin3_Integer(packIdx, packIdx, VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_HorzMin3_Integer(packIdx, packIdx.s035, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0)); \\\n\
 \\\n\
    VXC_WriteImage(output, coord.yz, packIdx, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
\n\
TENSOR_ARGMIN_AXIS0_8BITS_2D(I8,  I16, vxc_char8,  vxc_char8,  vxc_uchar8,  vxc_short8, vxc_ushort8)\n\
TENSOR_ARGMIN_AXIS0_8BITS_2D(I8,  U8,  vxc_char8,  vxc_char8,  vxc_uchar8,  vxc_uchar8, vxc_uchar8)\n\
TENSOR_ARGMIN_AXIS0_8BITS_2D(U8,  I16, vxc_uchar8, vxc_char8,  vxc_uchar8,  vxc_short8, vxc_ushort8)\n\
TENSOR_ARGMIN_AXIS0_8BITS_2D(U8,  U8,  vxc_uchar8, vxc_char8,  vxc_uchar8,  vxc_uchar8, vxc_uchar8)\n\
TENSOR_ARGMIN_AXIS0_8BITS_2D(I16, I16, vxc_short8, vxc_short8, vxc_ushort8, vxc_short8, vxc_ushort8)\n\
TENSOR_ARGMIN_AXIS0_8BITS_2D(I16, U8,  vxc_short8, vxc_short8, vxc_ushort8, vxc_uchar8, vxc_uchar8)\n\
\n\
_viv_uniform VXC_512Bits uniConvertHalf2Float32_4x4;\n\
\n\
#define TENSOR_ARGMIN_AXIS0_F16_2D(dst_type_name, dst_type) \\\n\
__kernel void vxArgmin_Axis0_F16to##dst_type_name##_2D( \\\n\
__read_only  image2d_array_t  input, \\\n\
__write_only image2d_array_t  output, \\\n\
        int  axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(0, get_global_id(0), 0, 0); \\\n\
    vxc_short8 vec0, vec1; \\\n\
    vxc_half8 src; \\\n\
    uint4 packIdx, currIdx; \\\n\
    float4 minValue, value; \\\n\
 \\\n\
    VXC_ReadImage(vec0, input, coord.xy, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src, vec0, 16); \\\n\
    _viv_asm(COPY, packIdx, packedArgIdx, 16); \\\n\
    _viv_asm(COPY, currIdx, packedArgIdx, 16); \\\n\
    coord.x += 4; \\\n\
    VXC_DP4x4(minValue, src, src, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertHalf2Float32_4x4); \\\n\
    for (; coord.x < inputWidth; ) \\\n\
    { \\\n\
        VXC_ReadImage(vec1, input, coord.xy, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, src, vec1, 16); \\\n\
        coord.x += 4; \\\n\
 \\\n\
       currIdx = currIdx + 4; \\\n\
        VXC_DP4x4(value, src, src, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertHalf2Float32_4x4); \\\n\
 \\\n\
        int4 condition; \\\n\
        condition = value < minValue; \\\n\
 \\\n\
        uint4 iCondition; \\\n\
        _viv_asm(COPY, iCondition, condition, 16); \\\n\
        packIdx = iCondition ? currIdx : packIdx; \\\n\
        minValue = value < minValue ? value : minValue; \\\n\
    } \\\n\
 \\\n\
    float4 minVec; \\\n\
    float2 minVal2 = minValue.xy < minValue.zw ? minValue.xy : minValue.zw; \\\n\
    minVec.x = minVal2.x < minVal2.y ? minVal2.x : minVal2.y; \\\n\
    int4 condition; \\\n\
    condition = minVec.xxxx == minValue; \\\n\
    uint4 iCondition; \\\n\
    _viv_asm(COPY, iCondition, condition, 16); \\\n\
    iCondition += 1; \\\n\
 \\\n\
    packIdx = mad_sat(iCondition, 0xFFFFFFFF, packIdx); \\\n\
 \\\n\
    uint2 val2 = packIdx.xy < packIdx.zw ? packIdx.xy : packIdx.zw; \\\n\
    val2.x = val2.x < val2.y ? val2.x : val2.y; \\\n\
 \\\n\
    dst_type dst; \\\n\
    _viv_asm(COPY, dst, val2, 4); \\\n\
    VXC_WriteImage(output, coord.yz, dst, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
TENSOR_ARGMIN_AXIS0_F16_2D(I16, vxc_ushort8)\n\
TENSOR_ARGMIN_AXIS0_F16_2D(U8,  vxc_uchar8)\n\
\n\
\n\
#define TENSOR_ARGMIN_AXIS0_F16(dst_type_name, dst_type) \\\n\
__kernel void vxArgmin_Axis0_F16to##dst_type_name( \\\n\
__read_only  image2d_array_t  input, \\\n\
__write_only image2d_array_t  output, \\\n\
        int  axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(0, get_global_id(0), get_global_id(1), 0); \\\n\
    vxc_short8 vec0, vec1; \\\n\
    vxc_half8 src; \\\n\
    uint4 packIdx, currIdx; \\\n\
    float4 minValue, value; \\\n\
 \\\n\
    VXC_ReadImage2DArray(vec0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src, vec0, 16); \\\n\
    _viv_asm(COPY, packIdx, packedArgIdx, 16); \\\n\
    _viv_asm(COPY, currIdx, packedArgIdx, 16); \\\n\
    coord.x += 4; \\\n\
    VXC_DP4x4(minValue, src, src, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertHalf2Float32_4x4); \\\n\
    for (; coord.x < inputWidth; ) \\\n\
    { \\\n\
        VXC_ReadImage2DArray(vec1, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, src, vec1, 16); \\\n\
        coord.x += 4; \\\n\
 \\\n\
       currIdx = currIdx + 4; \\\n\
        VXC_DP4x4(value, src, src, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertHalf2Float32_4x4); \\\n\
 \\\n\
        int4 condition; \\\n\
        condition = value < minValue; \\\n\
 \\\n\
        uint4 iCondition; \\\n\
        _viv_asm(COPY, iCondition, condition, 16); \\\n\
        packIdx = iCondition ? currIdx : packIdx; \\\n\
        minValue = value < minValue ? value : minValue; \\\n\
    } \\\n\
 \\\n\
    float4 minVec; \\\n\
    float2 minVal2 = minValue.xy < minValue.zw ? minValue.xy : minValue.zw; \\\n\
    minVec.x = minVal2.x < minVal2.y ? minVal2.x : minVal2.y; \\\n\
    int4 condition; \\\n\
    condition = minVec.xxxx == minValue; \\\n\
    uint4 iCondition; \\\n\
    _viv_asm(COPY, iCondition, condition, 16); \\\n\
    iCondition += 1; \\\n\
 \\\n\
    packIdx = mad_sat(iCondition, 0xFFFFFFFF, packIdx); \\\n\
 \\\n\
    uint2 val2 = packIdx.xy < packIdx.zw ? packIdx.xy : packIdx.zw; \\\n\
    val2.x = val2.x < val2.y ? val2.x : val2.y; \\\n\
 \\\n\
    dst_type dst; \\\n\
    _viv_asm(COPY, dst, val2, 4); \\\n\
    VXC_WriteImage(output, coord.yz, dst, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
TENSOR_ARGMIN_AXIS0_F16(I16, vxc_ushort8)\n\
TENSOR_ARGMIN_AXIS0_F16(U8,  vxc_uchar8)\n\
"; /* end of vsi_nn_kernel_argmin_axis0_vx*/

static const char vsi_nn_kernel_argmin_axis1_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform int4 packedArgIdx;\n\
_viv_uniform int argLenSub1;\n\
_viv_uniform VXC_512Bits uniExtractData_2x8;\n\
\n\
#define TENSOR_ARGMIN_AXIS1_16BITS(src_type_name, dst_type_name, src_type,\\\n\
                                    copy_type, axis_type, dst_type, inst_type) \\\n\
    __kernel void vxArgmin_Axis1_##src_type_name##to##dst_type_name( \\\n\
__read_only  image2d_array_t  input, \\\n\
__write_only image2d_array_t  output, \\\n\
        int  axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(get_global_id(0), argLenSub1, get_global_id(1), 0); \\\n\
    copy_type vec; \\\n\
    src_type src; \\\n\
    src_type minVal; \\\n\
    copy_type min; \\\n\
    VXC_ReadImage2DArray(min, input, coord.xyzw, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, minVal, min, 16); \\\n\
    axis_type axis; \\\n\
    axis_type packIdx; \\\n\
 \\\n\
    _viv_asm(COPY, axis, packedArgIdx, 16); \\\n\
    _viv_asm(COPY, packIdx, packedArgIdx, 16); \\\n\
 \\\n\
    coord.y --; \\\n\
    do \\\n\
    { \\\n\
       VXC_ReadImage2DArray(vec, input, coord.xyzw, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
       _viv_asm(COPY, src, vec, 16); \\\n\
       coord.y --; \\\n\
       packIdx --; \\\n\
       VXC_VertMin3_##inst_type(minVal, minVal, minVal, src, VXC_MODIFIER_BIN(0, 7, 0)); \\\n\
       _viv_asm(COPY, min, minVal, 16); \\\n\
       axis = (min == vec) ? packIdx : axis; \\\n\
    } while (coord.y >= 0); \\\n\
 \\\n\
    dst_type dst_axis; \\\n\
    VXC_DP2x8(dst_axis, axis, axis, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniExtractData_2x8); \\\n\
    VXC_WriteImage(output, coord.xz, dst_axis, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
TENSOR_ARGMIN_AXIS1_16BITS(F16, I16, vxc_half8,  vxc_short8, vxc_short8, vxc_short8, Half)\n\
TENSOR_ARGMIN_AXIS1_16BITS(F16, U8,  vxc_half8,  vxc_short8, vxc_short8, vxc_uchar8, Half)\n\
TENSOR_ARGMIN_AXIS1_16BITS(I16, I16, vxc_short8, vxc_short8, vxc_short8, vxc_short8, Integer)\n\
TENSOR_ARGMIN_AXIS1_16BITS(I16, U8,  vxc_short8, vxc_short8, vxc_short8, vxc_uchar8, Integer)\n\
\n\
\n\
#define TENSOR_ARGMIN_AXIS1_8BITS(src_type_name, dst_type_name, src_type, dst_type) \\\n\
__kernel void vxArgmin_Axis1_##src_type_name##to##dst_type_name( \\\n\
__read_only  image2d_array_t  input, \\\n\
__write_only image2d_array_t  output, \\\n\
        int  axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(get_global_id(0), argLenSub1, get_global_id(1), 0); \\\n\
    src_type src; \\\n\
    src_type minVal; \\\n\
    VXC_ReadImage2DArray(minVal, input, coord.xyzw, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    dst_type axis; \\\n\
    dst_type packIdx; \\\n\
 \\\n\
    _viv_asm(COPY, axis, packedArgIdx, 16); \\\n\
    _viv_asm(COPY, packIdx, packedArgIdx, 16); \\\n\
 \\\n\
    coord.y --; \\\n\
    do \\\n\
    { \\\n\
       VXC_ReadImage2DArray(src, input, coord.xyzw, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
       coord.y --; \\\n\
       packIdx --; \\\n\
       minVal = min(minVal, src); \\\n\
       dst_type condition; \\\n\
       VXC_Clamp(condition, src, minVal, minVal, VXC_MODIFIER_CLAMP(0, 7, 0, 1)); \\\n\
       axis = condition ? packIdx : axis; \\\n\
    } while (coord.y >= 0); \\\n\
 \\\n\
    VXC_WriteImage(output, coord.xz, axis, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
TENSOR_ARGMIN_AXIS1_8BITS(I8,  I16, vxc_char16,  vxc_short8)\n\
TENSOR_ARGMIN_AXIS1_8BITS(I8,  U8,  vxc_char16,  vxc_uchar16)\n\
TENSOR_ARGMIN_AXIS1_8BITS(U8,  I16, vxc_uchar16, vxc_short8)\n\
TENSOR_ARGMIN_AXIS1_8BITS(U8,  U8,  vxc_uchar16, vxc_uchar16)\n\
"; /* end of vsi_nn_kernel_argmin_axis1_vx*/

static const char vsi_nn_kernel_argmin_axis2_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform int4 packedArgIdx;\n\
_viv_uniform int argLenSub1;\n\
_viv_uniform VXC_512Bits uniExtractData_2x8;\n\
\n\
#define TENSOR_ARGMIN_AXIS2_16BITS(src_type_name, dst_type_name,\\\n\
                src_type, copy_type, axis_type, dst_type, inst_type) \\\n\
    __kernel void vxArgmin_Axis2_##src_type_name##to##dst_type_name( \\\n\
__read_only  image2d_array_t  input, \\\n\
__write_only image2d_array_t  output, \\\n\
        int  axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), argLenSub1, 0); \\\n\
    copy_type vec; \\\n\
    src_type src; \\\n\
    src_type minVal; \\\n\
    copy_type min; \\\n\
    VXC_ReadImage2DArray(min, input, coord.xyzw, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, minVal, min, 16); \\\n\
    axis_type axis; \\\n\
    axis_type packIdx; \\\n\
 \\\n\
    _viv_asm(COPY, axis, packedArgIdx, 16); \\\n\
    _viv_asm(COPY, packIdx, packedArgIdx, 16); \\\n\
 \\\n\
    coord.z --; \\\n\
    do \\\n\
    { \\\n\
       VXC_ReadImage2DArray(vec, input, coord.xyzw, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
       _viv_asm(COPY, src, vec, 16); \\\n\
       coord.z --; \\\n\
       packIdx --; \\\n\
       VXC_VertMin3_##inst_type(minVal, minVal, minVal, src, VXC_MODIFIER_BIN(0, 7, 0)); \\\n\
       _viv_asm(COPY, min, minVal, 16); \\\n\
       axis = (min == vec) ? packIdx : axis; \\\n\
    } while (coord.z >= 0); \\\n\
 \\\n\
    dst_type dst_axis; \\\n\
    VXC_DP2x8(dst_axis, axis, axis, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniExtractData_2x8); \\\n\
    VXC_WriteImage(output, coord.xy, dst_axis, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
TENSOR_ARGMIN_AXIS2_16BITS(F16, I16, vxc_half8,  vxc_short8, vxc_short8, vxc_short8, Half)\n\
TENSOR_ARGMIN_AXIS2_16BITS(F16, U8,  vxc_half8,  vxc_short8, vxc_short8, vxc_uchar8, Half)\n\
TENSOR_ARGMIN_AXIS2_16BITS(I16, I16, vxc_short8, vxc_short8, vxc_short8, vxc_short8, Integer)\n\
TENSOR_ARGMIN_AXIS2_16BITS(I16, U8,  vxc_short8, vxc_short8, vxc_short8, vxc_uchar8, Integer)\n\
\n\
#define TENSOR_ARGMIN_AXIS2_16BITS_2D(src_type_name, dst_type_name, src_type, \\\n\
                                    copy_type, axis_type, dst_type, inst_type) \\\n\
    __kernel void vxArgmin_Axis2_##src_type_name##to##dst_type_name##_2D( \\\n\
__read_only  image2d_array_t  input, \\\n\
__write_only image2d_array_t  output, \\\n\
        int  axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(get_global_id(0), argLenSub1, 0, 0); \\\n\
    copy_type vec; \\\n\
    src_type src; \\\n\
    src_type minVal; \\\n\
    copy_type min; \\\n\
    VXC_ReadImage(min, input, coord.xy, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, minVal, min, 16); \\\n\
    axis_type axis; \\\n\
    axis_type packIdx; \\\n\
 \\\n\
    _viv_asm(COPY, axis, packedArgIdx, 16); \\\n\
    _viv_asm(COPY, packIdx, packedArgIdx, 16); \\\n\
 \\\n\
    coord.y --; \\\n\
    do \\\n\
    { \\\n\
       VXC_ReadImage(vec, input, coord.xy, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
       _viv_asm(COPY, src, vec, 16); \\\n\
       coord.y --; \\\n\
       packIdx --; \\\n\
       VXC_VertMin3_##inst_type(minVal, minVal, minVal, src, VXC_MODIFIER_BIN(0, 7, 0)); \\\n\
       _viv_asm(COPY, min, minVal, 16); \\\n\
       axis = (min == vec) ? packIdx : axis; \\\n\
    } while (coord.y >= 0); \\\n\
 \\\n\
    dst_type dst_axis; \\\n\
    VXC_DP2x8(dst_axis, axis, axis, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniExtractData_2x8); \\\n\
    VXC_WriteImage(output, coord.xz, dst_axis, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
TENSOR_ARGMIN_AXIS2_16BITS_2D(F16, I16, vxc_half8,  vxc_short8, vxc_short8, vxc_short8, Half)\n\
TENSOR_ARGMIN_AXIS2_16BITS_2D(F16, U8,  vxc_half8,  vxc_short8, vxc_short8, vxc_uchar8, Half)\n\
TENSOR_ARGMIN_AXIS2_16BITS_2D(I16, I16, vxc_short8, vxc_short8, vxc_short8, vxc_short8, Integer)\n\
TENSOR_ARGMIN_AXIS2_16BITS_2D(I16, U8,  vxc_short8, vxc_short8, vxc_short8, vxc_uchar8, Integer)\n\
\n\
\n\
#define TENSOR_ARGMIN_AXIS2_8BITS(src_type_name, dst_type_name, src_type, dst_type) \\\n\
    __kernel void vxArgmin_Axis2_##src_type_name##to##dst_type_name( \\\n\
__read_only  image2d_array_t  input, \\\n\
__write_only image2d_array_t  output, \\\n\
        int  axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), argLenSub1, 0); \\\n\
    src_type src; \\\n\
    src_type minVal; \\\n\
    VXC_ReadImage2DArray(minVal, input, coord.xyzw, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    dst_type axis; \\\n\
    dst_type packIdx; \\\n\
 \\\n\
    _viv_asm(COPY, axis, packedArgIdx, 16); \\\n\
    _viv_asm(COPY, packIdx, packedArgIdx, 16); \\\n\
 \\\n\
    coord.z --; \\\n\
    do \\\n\
    { \\\n\
       VXC_ReadImage2DArray(src, input, coord.xyzw, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
       coord.z --; \\\n\
       packIdx --; \\\n\
       minVal = min(minVal, src); \\\n\
       dst_type condition; \\\n\
       VXC_Clamp(condition, src, minVal, minVal, VXC_MODIFIER_CLAMP(0, 7, 0, 1)); \\\n\
       axis = condition ? packIdx : axis; \\\n\
    } while (coord.z >= 0); \\\n\
 \\\n\
    VXC_WriteImage(output, coord.xy, axis, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
TENSOR_ARGMIN_AXIS2_8BITS(I8,  I16, vxc_char8,  vxc_short8)\n\
TENSOR_ARGMIN_AXIS2_8BITS(I8,  U8,  vxc_char8,  vxc_uchar8)\n\
TENSOR_ARGMIN_AXIS2_8BITS(U8,  I16, vxc_uchar8, vxc_short8)\n\
TENSOR_ARGMIN_AXIS2_8BITS(U8,  U8,  vxc_uchar8, vxc_uchar8)\n\
\n\
#define TENSOR_ARGMIN_AXIS2_8BITS_2D(src_type_name, dst_type_name, src_type, dst_type) \\\n\
    __kernel void vxArgmin_Axis2_##src_type_name##to##dst_type_name##_2D( \\\n\
__read_only  image2d_array_t  input, \\\n\
__write_only image2d_array_t  output, \\\n\
        int  axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(get_global_id(0), argLenSub1, 0, 0); \\\n\
    src_type src; \\\n\
    src_type minVal; \\\n\
    VXC_ReadImage(minVal, input, coord.xy, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    dst_type axis; \\\n\
    dst_type packIdx; \\\n\
 \\\n\
    _viv_asm(COPY, axis, packedArgIdx, 16); \\\n\
    _viv_asm(COPY, packIdx, packedArgIdx, 16); \\\n\
 \\\n\
    coord.y --; \\\n\
    do \\\n\
    { \\\n\
       VXC_ReadImage(src, input, coord.xy, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
       coord.y --; \\\n\
       packIdx --; \\\n\
       minVal = min(minVal, src); \\\n\
       dst_type condition; \\\n\
       VXC_Clamp(condition, src, minVal, minVal, VXC_MODIFIER_CLAMP(0, 7, 0, 1)); \\\n\
       axis = condition ? packIdx : axis; \\\n\
    } while (coord.y >= 0); \\\n\
 \\\n\
    VXC_WriteImage(output, coord.xz, axis, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
TENSOR_ARGMIN_AXIS2_8BITS_2D(I8,  I16, vxc_char8,  vxc_short8)\n\
TENSOR_ARGMIN_AXIS2_8BITS_2D(I8,  U8,  vxc_char8,  vxc_uchar8)\n\
TENSOR_ARGMIN_AXIS2_8BITS_2D(U8,  I16, vxc_uchar8, vxc_short8)\n\
TENSOR_ARGMIN_AXIS2_8BITS_2D(U8,  U8,  vxc_uchar8, vxc_uchar8)\n\
"; /* end of vsi_nn_kernel_argmin_axis2_vx*/

static const char vsi_nn_kernel_axis_aligned_bbox_transform_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
__kernel void vxcAxis_aligned_bbox_transform(\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_array_t  output)\n\
{\n\
\n\
}\n\
"; /* end of vsi_nn_kernel_axis_aligned_bbox_transform_vx*/

static const char vsi_nn_kernel_box_with_nms_limit_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
__kernel void vxcBox_with_nms_limit(\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_array_t  output)\n\
{\n\
\n\
}\n\
"; /* end of vsi_nn_kernel_box_with_nms_limit_vx*/

static const char vsi_nn_kernel_clip_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform int4 packedMinData_FP16;\n\
_viv_uniform int4 packedMaxData_FP16;\n\
\n\
__kernel void vxcTensorClip_Fp16(\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_array_t  output,\n\
                        float minData,\n\
                        float maxData)\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_short8 vec0, dst;\n\
    vxc_half8  src0, src1, minHf, maxHf;\n\
    VXC_ReadImage2DArray(vec0, input, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, src0, vec0, 16);\n\
\n\
    _viv_asm(COPY, minHf, packedMinData_FP16, 16);\n\
    _viv_asm(COPY, maxHf, packedMaxData_FP16, 16);\n\
    VXC_Clamp_Half(src1, src0, minHf, maxHf, VXC_MODIFIER_CLAMP(0, 7, 0, 0));\n\
    _viv_asm(COPY, dst, src1, 16);\n\
\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform VXC_512Bits uniConvertIntegerLo_2x8;\n\
_viv_uniform VXC_512Bits uniConvertIntegerHi_2x8;\n\
\n\
_viv_uniform int4 packedMinDataInt16;\n\
_viv_uniform int4 packedMaxDataInt16;\n\
_viv_uniform int4 packedMinData;\n\
_viv_uniform int4 packedMaxData;\n\
\n\
__kernel void vxcTensorClip_Int16(\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_array_t  output,\n\
                        float minData,\n\
                        float maxData)\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_short8 src0, min, max;\n\
    VXC_ReadImage2DArray(src0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_DP2x8(src0, src0, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertIntegerLo_2x8);\n\
\n\
    _viv_asm(COPY, min, packedMinDataInt16, 16);\n\
    _viv_asm(COPY, max, packedMaxDataInt16, 16);\n\
    VXC_Clamp(src0, src0, min, max, VXC_MODIFIER_CLAMP(0, 7, 0, 0));\n\
    VXC_WriteImage2DArray(output, coord, src0, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcTensorClip_Int8(\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_array_t  output,\n\
                        float minData,\n\
                        float maxData)\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_char16 src0, min, max;\n\
\n\
    VXC_ReadImage2DArray(src0, input, coord, 0, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_DP2x8(src0, src0, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertIntegerLo_2x8);\n\
    VXC_DP2x8(src0, src0, src0, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniConvertIntegerHi_2x8);\n\
    _viv_asm(COPY, min, packedMinData, 16);\n\
    _viv_asm(COPY, max, packedMaxData, 16);\n\
    VXC_Clamp(src0, src0, min, max, VXC_MODIFIER_CLAMP(0, 15, 0, 0));\n\
\n\
    VXC_WriteImage2DArray(output, coord, src0, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform VXC_512Bits uniU8MulAndPostShift_Lo_2x8;\n\
_viv_uniform VXC_512Bits uniU8MulAndPostShift_Hi_2x8;\n\
_viv_uniform int2 multAndoutZP;//[0:15] multiplier, [31:63] output zp\n\
\n\
__kernel void vxcTensorClip_Uint8(\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_array_t  output,\n\
                        float minData,\n\
                        float maxData)\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_uchar16 vec0, min, max, dst;\n\
    VXC_ReadImage2DArray(vec0, input,  coord,\\\n\
         VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    vxc_ushort8 multiplier;\n\
    _viv_asm(COPY, multiplier, multAndoutZP, 16);\n\
    VXC_DP2x8(dst, vec0, multiplier,\\\n\
         VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniU8MulAndPostShift_Lo_2x8);\n\
    VXC_DP2x8(dst, vec0, multiplier,\\\n\
         VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniU8MulAndPostShift_Hi_2x8);\n\
    _viv_asm(COPY, min, packedMinData, 16);\n\
    _viv_asm(COPY, max, packedMaxData, 16);\n\
    VXC_Clamp(dst, dst, min, max, VXC_MODIFIER_CLAMP(0, 15, 0, 0));\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 15, 0,VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_clip_vx*/

static const char vsi_nn_kernel_crop_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
//-----------------------------------------------tensor crop-------------------------------\n\
__kernel void vxcTensorCrop_Int16(\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_array_t  output,\n\
        int offset0,\n\
        int offset1,\n\
        int offset2)\n\
{\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    vxc_ushort8 src0, src1, src2, src3;\n\
\n\
    VXC_ReadImage2DArray(src0, input,  coord_in, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input,  coord_in, VXC_5BITOFFSET_XY(0, 1),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src2, input,  coord_in, VXC_5BITOFFSET_XY(0, 2),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src3, input,  coord_in, VXC_5BITOFFSET_XY(0, 3),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    int4 coord_out = (int4)(get_global_id(0) - offset0, get_global_id(1)\\\n\
        - offset1, get_global_id(2) - offset2, 0);\n\
\n\
    VXC_WriteImage2DArray(output, coord_out, src0, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
    coord_out.y ++;\n\
    VXC_WriteImage2DArray(output, coord_out, src1, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
    coord_out.y ++;\n\
    VXC_WriteImage2DArray(output, coord_out, src2, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
    coord_out.y ++;\n\
    VXC_WriteImage2DArray(output, coord_out, src3, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcTensorCrop_Int8(\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_array_t  output,\n\
        int offset0,\n\
        int offset1,\n\
        int offset2)\n\
{\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_uchar16 src0, src1, src2, src3;\n\
\n\
    VXC_ReadImage2DArray(src0, input,  coord_in, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input,  coord_in, VXC_5BITOFFSET_XY(0, 1),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src2, input,  coord_in, VXC_5BITOFFSET_XY(0, 2),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src3, input,  coord_in, VXC_5BITOFFSET_XY(0, 3),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    int4 coord_out = (int4)(get_global_id(0) - offset0, get_global_id(1) - offset1,\\\n\
        get_global_id(2) - offset2, 0);\n\
\n\
    VXC_WriteImage2DArray(output, coord_out, src0, VXC_MODIFIER(0, 15, 0,VXC_RM_TowardZero, 0));\n\
    coord_out.y ++;\n\
    VXC_WriteImage2DArray(output, coord_out, src1, VXC_MODIFIER(0, 15, 0,VXC_RM_TowardZero, 0));\n\
    coord_out.y ++;\n\
    VXC_WriteImage2DArray(output, coord_out, src2, VXC_MODIFIER(0, 15, 0,VXC_RM_TowardZero, 0));\n\
    coord_out.y ++;\n\
    VXC_WriteImage2DArray(output, coord_out, src3, VXC_MODIFIER(0, 15, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform VXC_512Bits uniConvertInt16toFp16_2x8;\n\
\n\
__kernel void vxcTensorCrop_Int16_Fp16(\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_array_t  output,\n\
        int offset0,\n\
        int offset1,\n\
        int offset2)\n\
{\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    vxc_short8 src0, src1, src2, src3;\n\
\n\
    VXC_ReadImage2DArray(src0, input,  coord_in, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input,  coord_in, VXC_5BITOFFSET_XY(0, 1),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src2, input,  coord_in, VXC_5BITOFFSET_XY(0, 2),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src3, input,  coord_in, VXC_5BITOFFSET_XY(0, 3),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    int4 coord_out = (int4)(get_global_id(0) - offset0, get_global_id(1)\\\n\
        - offset1, get_global_id(2) - offset2, 0);\n\
\n\
    vxc_half8 dst0, dst1, dst2, dst3;\n\
    VXC_DP2x8(dst0, src0, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1),\\\n\
        uniConvertInt16toFp16_2x8);\n\
    VXC_DP2x8(dst1, src1, src1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1),\\\n\
        uniConvertInt16toFp16_2x8);\n\
    VXC_DP2x8(dst2, src2, src2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1),\\\n\
        uniConvertInt16toFp16_2x8);\n\
    VXC_DP2x8(dst3, src3, src3, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1),\\\n\
        uniConvertInt16toFp16_2x8);\n\
\n\
    vxc_short8 out0, out1, out2, out3;\n\
    _viv_asm(COPY, out0, dst0, 16);\n\
    _viv_asm(COPY, out1, dst1, 16);\n\
    _viv_asm(COPY, out2, dst2, 16);\n\
    _viv_asm(COPY, out3, dst3, 16);\n\
\n\
    VXC_WriteImage2DArray(output, coord_out, out0, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
    coord_out.y ++;\n\
    VXC_WriteImage2DArray(output, coord_out, out1, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
    coord_out.y ++;\n\
    VXC_WriteImage2DArray(output, coord_out, out2, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
    coord_out.y ++;\n\
    VXC_WriteImage2DArray(output, coord_out, out3, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_crop_vx*/

static const char vsi_nn_kernel_detection_postprocess_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
__kernel void vxcDetection_postprocess(\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_array_t  output)\n\
{\n\
\n\
}\n\
"; /* end of vsi_nn_kernel_detection_postprocess_vx*/

static const char vsi_nn_kernel_dropout_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
_viv_uniform VXC_512Bits fp16MulFp16ToFp16_2x8;\n\
__kernel void dropoutVXC\n\
    (\n\
    image2d_array_t input,\n\
    image2d_array_t output,\n\
    float scale\n\
    )\n\
{\n\
    vxc_short8 din, dout;\n\
    vxc_half8 dinHalf, doutHalf;\n\
    half scaleFp16;\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    VXC_ReadImage(din, input, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(CONV, scaleFp16, scale);\n\
    _viv_asm(COPY, dinHalf, din, 16);\n\
    VXC_DP2x8(doutHalf, dinHalf, scaleFp16, VXC_MODIFIER(0, 7, 0,\\\n\
        VXC_RM_TowardZero, 0), fp16MulFp16ToFp16_2x8);\n\
    _viv_asm(COPY, dout, doutHalf, 16);\n\
    VXC_WriteImage(output, coord.xy, dout, VXC_MODIFIER(0, 7, 0,\\\n\
        VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_dropout_vx*/

static const char vsi_nn_kernel_elu_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform float logE;\n\
\n\
float4 elu(float4 val)\n\
{\n\
    float4 x = val * logE;\n\
    x = exp2(x) - 1;\n\
\n\
    return val < 0 ? x : val;\n\
}\n\
\n\
_viv_uniform float inputScale;\n\
_viv_uniform float inputTail;\n\
_viv_uniform float outputScale;\n\
_viv_uniform float outputZP;\n\
_viv_uniform VXC_512Bits uniExtract8Data_2x8;\n\
_viv_uniform VXC_512Bits uniDatatoFp32Part0_4x4;\n\
_viv_uniform VXC_512Bits uniDatatoFp32Part1_4x4;\n\
\n\
#define TENSOR_ELU(src_type_name, dst_type_name, src_type, src_copy_type, \\\n\
                    convert_type, dst_type, dst_copy_type) \\\n\
    __kernel void vxTensorElu_##src_type_name##to##dst_type_name( \\\n\
    __read_only  image2d_array_t  input, \\\n\
    __write_only image2d_array_t  output \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0); \\\n\
    src_type      src0; \\\n\
    src_copy_type src1; \\\n\
    VXC_ReadImage2DArray(src0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src1, src0, 16); \\\n\
 \\\n\
    float4 vecA; \\\n\
    float4 vecB; \\\n\
    VXC_DP4x4(vecA, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniDatatoFp32Part0_4x4); \\\n\
    VXC_DP4x4(vecB, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniDatatoFp32Part1_4x4); \\\n\
    vecA = vecA * inputScale + inputTail; \\\n\
    vecB = vecB * inputScale + inputTail; \\\n\
    vecA = elu(vecA); \\\n\
    vecB = elu(vecB); \\\n\
    vecA = vecA * outputScale + outputZP; \\\n\
    vecB = vecB * outputScale + outputZP; \\\n\
 \\\n\
    convert_type dst0, dst1; \\\n\
    _viv_asm(CONV_RTE, dst0, vecA); \\\n\
    _viv_asm(CONV_RTE, dst1, vecB); \\\n\
    dst_type dst2; \\\n\
    VXC_DP2x8(dst2, dst0, dst1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1), uniExtract8Data_2x8); \\\n\
    dst_copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst2, 16); \\\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
//ELU\n\
TENSOR_ELU(F16, F16, vxc_short8, vxc_half8,  half4, vxc_half8,  vxc_short8)\n\
TENSOR_ELU(F16, I8,  vxc_short8, vxc_half8,  int4,  vxc_char8,  vxc_char8)\n\
TENSOR_ELU(F16, U8,  vxc_short8, vxc_half8,  int4,  vxc_uchar8, vxc_uchar8)\n\
TENSOR_ELU(F16, I16, vxc_short8, vxc_half8,  int4,  vxc_short8, vxc_short8)\n\
TENSOR_ELU(I8,  I8,  vxc_char8,  vxc_char8,  int4,  vxc_char8,  vxc_char8)\n\
TENSOR_ELU(I8,  F16, vxc_char8,  vxc_char8,  half4, vxc_half8,  vxc_short8)\n\
TENSOR_ELU(U8,  U8,  vxc_uchar8, vxc_uchar8, int4,  vxc_uchar8, vxc_uchar8)\n\
TENSOR_ELU(U8,  F16, vxc_uchar8, vxc_uchar8, half4, vxc_half8,  vxc_short8)\n\
TENSOR_ELU(I16, I16, vxc_short8, vxc_short8, int4,  vxc_short8, vxc_short8)\n\
TENSOR_ELU(I16, F16, vxc_short8, vxc_short8, half4, vxc_half8,  vxc_short8)\n\
\n\
\n\
#define TENSOR_ELU_2D(src_type_name, dst_type_name, src_type, src_copy_type, \\\n\
                        convert_type, dst_type, dst_copy_type) \\\n\
    __kernel void vxTensorElu_##src_type_name##to##dst_type_name##_2D( \\\n\
    __read_only  image2d_array_t  input, \\\n\
    __write_only image2d_array_t  output \\\n\
    ) \\\n\
{ \\\n\
    int2 coord = (int2)(get_global_id(0), get_global_id(1)); \\\n\
    src_type      src0; \\\n\
    src_copy_type src1; \\\n\
    VXC_ReadImage(src0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src1, src0, 16); \\\n\
 \\\n\
    float4 vecA; \\\n\
    float4 vecB; \\\n\
    VXC_DP4x4(vecA, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniDatatoFp32Part0_4x4); \\\n\
    VXC_DP4x4(vecB, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniDatatoFp32Part1_4x4); \\\n\
    vecA = vecA * inputScale + inputTail; \\\n\
    vecB = vecB * inputScale + inputTail; \\\n\
    vecA = elu(vecA); \\\n\
    vecB = elu(vecB); \\\n\
    vecA = vecA * outputScale + outputZP; \\\n\
    vecB = vecB * outputScale + outputZP; \\\n\
 \\\n\
    convert_type dst0, dst1; \\\n\
    _viv_asm(CONV_RTE, dst0, vecA); \\\n\
    _viv_asm(CONV_RTE, dst1, vecB); \\\n\
    dst_type dst2; \\\n\
    VXC_DP2x8(dst2, dst0, dst1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1), uniExtract8Data_2x8); \\\n\
    dst_copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst2, 16); \\\n\
    VXC_WriteImage(output, coord, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
//ELU\n\
TENSOR_ELU_2D(F16, F16, vxc_short8, vxc_half8, half4, vxc_half8,  vxc_short8)\n\
TENSOR_ELU_2D(F16, I8,  vxc_short8, vxc_half8, int4,  vxc_char8,  vxc_char8)\n\
TENSOR_ELU_2D(F16, U8,  vxc_short8, vxc_half8, int4,  vxc_uchar8, vxc_uchar8)\n\
TENSOR_ELU_2D(F16, I16, vxc_short8, vxc_half8, int4,  vxc_short8, vxc_short8)\n\
TENSOR_ELU_2D(I8,  I8,  vxc_char8,  vxc_char8,  int4,  vxc_char8,  vxc_char8)\n\
TENSOR_ELU_2D(I8,  F16, vxc_char8,  vxc_char8,  half4, vxc_half8,  vxc_short8)\n\
TENSOR_ELU_2D(U8,  U8,  vxc_uchar8, vxc_uchar8, int4,  vxc_uchar8, vxc_uchar8)\n\
TENSOR_ELU_2D(U8,  F16, vxc_uchar8, vxc_uchar8, half4, vxc_half8,  vxc_short8)\n\
TENSOR_ELU_2D(I16, I16, vxc_short8, vxc_short8, int4,  vxc_short8, vxc_short8)\n\
TENSOR_ELU_2D(I16, F16, vxc_short8, vxc_short8, half4, vxc_half8,  vxc_short8)\n\
"; /* end of vsi_nn_kernel_elu_vx*/

static const char vsi_nn_kernel_exp_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform float logE;\n\
float4 Exp_(float4 x)\n\
{\n\
    x *= logE;\n\
    x = exp2(x);\n\
    return x;\n\
}\n\
_viv_uniform float inputScale;\n\
_viv_uniform float inputTail;\n\
_viv_uniform float outputScale;\n\
_viv_uniform float outputZP;\n\
_viv_uniform VXC_512Bits uniExtract8Data_2x8;\n\
_viv_uniform VXC_512Bits uniDatatoFp32Part0_4x4;\n\
_viv_uniform VXC_512Bits uniDatatoFp32Part1_4x4;\n\
\n\
#define TENSOR_TRANSCENDENTAL(funcName, src_type_name, dst_type_name, src_type, \\\n\
                src_copy_type, convert_type, dst_type, dst_copy_type) \\\n\
    __kernel void vxTensor##funcName##_##src_type_name##to##dst_type_name( \\\n\
    __read_only  image2d_array_t  input, \\\n\
    __write_only image2d_array_t  output \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0); \\\n\
    src_type      src0; \\\n\
    src_copy_type src1; \\\n\
    VXC_ReadImage2DArray(src0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src1, src0, 16); \\\n\
 \\\n\
    float4 vecA; \\\n\
    float4 vecB; \\\n\
    VXC_DP4x4(vecA, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniDatatoFp32Part0_4x4); \\\n\
    VXC_DP4x4(vecB, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniDatatoFp32Part1_4x4); \\\n\
    vecA = vecA * inputScale + inputTail; \\\n\
    vecB = vecB * inputScale + inputTail; \\\n\
    vecA = funcName##_(vecA); \\\n\
    vecB = funcName##_(vecB); \\\n\
    vecA = vecA * outputScale + outputZP; \\\n\
    vecB = vecB * outputScale + outputZP; \\\n\
 \\\n\
    convert_type dst0, dst1; \\\n\
    _viv_asm(CONV_RTE, dst0, vecA); \\\n\
    _viv_asm(CONV_RTE, dst1, vecB); \\\n\
    dst_type dst2; \\\n\
    VXC_DP2x8(dst2, dst0, dst1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1), uniExtract8Data_2x8); \\\n\
    dst_copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst2, 16); \\\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
//EXP\n\
TENSOR_TRANSCENDENTAL(Exp, F16, F16, vxc_short8, vxc_half8,  half4, vxc_half8,  vxc_short8)\n\
TENSOR_TRANSCENDENTAL(Exp, F16, I8,  vxc_short8, vxc_half8,  int4,  vxc_char8,  vxc_char8)\n\
TENSOR_TRANSCENDENTAL(Exp, F16, U8,  vxc_short8, vxc_half8,  int4,  vxc_uchar8, vxc_uchar8)\n\
TENSOR_TRANSCENDENTAL(Exp, F16, I16, vxc_short8, vxc_half8,  int4,  vxc_short8, vxc_short8)\n\
TENSOR_TRANSCENDENTAL(Exp, I8,  I8,  vxc_char8,  vxc_char8,  int4,  vxc_char8,  vxc_char8)\n\
TENSOR_TRANSCENDENTAL(Exp, I8,  F16, vxc_char8,  vxc_char8,  half4, vxc_half8,  vxc_short8)\n\
TENSOR_TRANSCENDENTAL(Exp, U8,  U8,  vxc_uchar8, vxc_uchar8, int4,  vxc_uchar8, vxc_uchar8)\n\
TENSOR_TRANSCENDENTAL(Exp, U8,  F16, vxc_uchar8, vxc_uchar8, half4, vxc_half8,  vxc_short8)\n\
TENSOR_TRANSCENDENTAL(Exp, I16, I16, vxc_short8, vxc_short8, int4,  vxc_short8, vxc_short8)\n\
TENSOR_TRANSCENDENTAL(Exp, I16, F16, vxc_short8, vxc_short8, half4, vxc_half8,  vxc_short8)\n\
\n\
#define TENSOR_TRANSCENDENTAL_2D(funcName, src_type_name, dst_type_name, src_type, \\\n\
        src_copy_type, convert_type, dst_type, dst_copy_type) \\\n\
    __kernel void vxTensor##funcName##_##src_type_name##to##dst_type_name##_2D( \\\n\
    __read_only  image2d_array_t  input, \\\n\
    __write_only image2d_array_t  output \\\n\
    ) \\\n\
{ \\\n\
    int2 coord = (int2)(get_global_id(0), get_global_id(1)); \\\n\
    src_type      src0; \\\n\
    src_copy_type src1; \\\n\
    VXC_ReadImage(src0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src1, src0, 16); \\\n\
 \\\n\
    float4 vecA; \\\n\
    float4 vecB; \\\n\
    VXC_DP4x4(vecA, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniDatatoFp32Part0_4x4); \\\n\
    VXC_DP4x4(vecB, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniDatatoFp32Part1_4x4); \\\n\
    vecA = vecA * inputScale + inputTail; \\\n\
    vecB = vecB * inputScale + inputTail; \\\n\
    vecA = funcName##_(vecA); \\\n\
    vecB = funcName##_(vecB); \\\n\
    vecA = vecA * outputScale + outputZP; \\\n\
    vecB = vecB * outputScale + outputZP; \\\n\
 \\\n\
    convert_type dst0, dst1; \\\n\
    _viv_asm(CONV_RTE, dst0, vecA); \\\n\
    _viv_asm(CONV_RTE, dst1, vecB); \\\n\
    dst_type dst2; \\\n\
    VXC_DP2x8(dst2, dst0, dst1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1), uniExtract8Data_2x8); \\\n\
    dst_copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst2, 16); \\\n\
    VXC_WriteImage(output, coord, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
//EXP\n\
TENSOR_TRANSCENDENTAL_2D(Exp, F16, F16, vxc_short8, vxc_half8, half4, vxc_half8,  vxc_short8)\n\
TENSOR_TRANSCENDENTAL_2D(Exp, F16, I8,  vxc_short8, vxc_half8, int4,  vxc_char8,  vxc_char8)\n\
TENSOR_TRANSCENDENTAL_2D(Exp, F16, U8,  vxc_short8, vxc_half8, int4,  vxc_uchar8, vxc_uchar8)\n\
TENSOR_TRANSCENDENTAL_2D(Exp, F16, I16, vxc_short8, vxc_half8, int4,  vxc_short8, vxc_short8)\n\
TENSOR_TRANSCENDENTAL_2D(Exp, I8,  I8,  vxc_char8,  vxc_char8,  int4,  vxc_char8,  vxc_char8)\n\
TENSOR_TRANSCENDENTAL_2D(Exp, I8,  F16, vxc_char8,  vxc_char8,  half4, vxc_half8,  vxc_short8)\n\
TENSOR_TRANSCENDENTAL_2D(Exp, U8,  U8,  vxc_uchar8, vxc_uchar8, int4,  vxc_uchar8, vxc_uchar8)\n\
TENSOR_TRANSCENDENTAL_2D(Exp, U8,  F16, vxc_uchar8, vxc_uchar8, half4, vxc_half8,  vxc_short8)\n\
TENSOR_TRANSCENDENTAL_2D(Exp, I16, I16, vxc_short8, vxc_short8, int4,  vxc_short8, vxc_short8)\n\
TENSOR_TRANSCENDENTAL_2D(Exp, I16, F16, vxc_short8, vxc_short8, half4, vxc_half8,  vxc_short8)\n\
"; /* end of vsi_nn_kernel_exp_vx*/

static const char vsi_nn_kernel_extra_ending_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
__kernel void vxcExtra_ending_i16(\n\
    __read_only image2d_array_t   input0,\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_array_t  output)\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_short8 data;\n\
    VXC_ReadImage2DArray(data, input, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_WriteImage2DArray(output, coord, data, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcExtra_ending_i8(\n\
    __read_only image2d_array_t   input0,\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_array_t  output)\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_char8 data;\n\
    VXC_ReadImage2DArray(data, input, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_WriteImage2DArray(output, coord, data, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcExtra_ending_u8(\n\
    __read_only image2d_array_t   input0,\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_array_t  output)\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_uchar8 data;\n\
    VXC_ReadImage2DArray(data, input, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_WriteImage2DArray(output, coord, data, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_extra_ending_vx*/

static const char vsi_nn_kernel_floordiv_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform VXC_512Bits uniConvertDirInt16Fp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertEndInt16Fp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertInt32toUint8_2x8;\n\
\n\
_viv_uniform VXC_512Bits uniConvertUint8SubZpToFp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertSecUint8SubZpToFp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertFstFp16Fp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertSecFp16Fp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertInt8FstFp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertInt8SecFp32_4x4;\n\
\n\
_viv_uniform float in_scale0;\n\
_viv_uniform float in_scale1;\n\
_viv_uniform float out_scale;\n\
_viv_uniform int in_zp0;\n\
_viv_uniform int in_zp1;\n\
_viv_uniform int out_zp;\n\
\n\
__kernel __attribute__((reqd_work_group_size(8, 1, 1))) void vxcTensorFloorDiv_Fp16(\n\
    image2d_array_t input0,\n\
    image2d_array_t input1,\n\
    image2d_array_t output)\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_short8 src0, src1;\n\
    vxc_short8 dst;\n\
    vxc_half8 data0, data1;\n\
    VXC_ReadImage2DArray(src0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, data0, src0, 16);\n\
    VXC_ReadImage2DArray(src1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, data1, src1, 16);\n\
\n\
    float4 x0, x1;\n\
    float4 y0, y1;\n\
    float4 tmpDst0, tmpDst1;\n\
    VXC_DP4x4(x0, data0, data0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertFstFp16Fp32_4x4);\n\
    VXC_DP4x4(x1, data0, data0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertSecFp16Fp32_4x4);\n\
    VXC_DP4x4(y0, data1, data1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertFstFp16Fp32_4x4);\n\
    VXC_DP4x4(y1, data1, data1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertSecFp16Fp32_4x4);\n\
    tmpDst0 = floor(x0/y0);\n\
    tmpDst1 = floor(x1/y1);\n\
\n\
    half4 tmpVal0, tmpVal1;\n\
    _viv_asm(CONV, tmpVal0, tmpDst0);\n\
    _viv_asm(CONV, tmpVal1, tmpDst1);\n\
    VXC_DP2x8(dst, tmpVal0, tmpVal1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvertInt32toUint8_2x8);\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel __attribute__((reqd_work_group_size(8, 1, 1))) void vxcTensorFloorDiv_Int16(\n\
    image2d_array_t input0,\n\
    image2d_array_t input1,\n\
    image2d_array_t output)\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_short8 src0, src1;\n\
    vxc_short8 dst;\n\
    VXC_ReadImage2DArray(src0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    float4 x0, x1;\n\
    float4 y0, y1;\n\
    float4 tmpDst0, tmpDst1;\n\
    float4 data0, data1;\n\
    VXC_DP4x4(x0, src0, src0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertDirInt16Fp32_4x4);\n\
    VXC_DP4x4(x1, src0, src0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertEndInt16Fp32_4x4);\n\
    VXC_DP4x4(y0, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertDirInt16Fp32_4x4);\n\
    VXC_DP4x4(y1, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertEndInt16Fp32_4x4);\n\
\n\
    tmpDst0 = x0 * in_scale0 / (y0 * in_scale1);\n\
    data0 = floor(tmpDst0) * out_scale;\n\
    tmpDst1 = x1 * in_scale0 / (y1 * in_scale1);\n\
    data1 = floor(tmpDst1) * out_scale;\n\
\n\
    vxc_int4 tmpVal0, tmpVal1;\n\
    tmpVal0 = convert_int4_rte(data0);\n\
    tmpVal1 = convert_int4_rte(data1);\n\
    VXC_DP2x8(dst, tmpVal0, tmpVal1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1), uniConvertInt32toUint8_2x8);\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel __attribute__((reqd_work_group_size(8, 1, 1))) void vxcTensorFloorDiv_Uint8(\n\
    image2d_array_t input0,\n\
    image2d_array_t input1,\n\
    image2d_array_t output)\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_uchar8 src0, src1;\n\
    vxc_uchar8 dst;\n\
    VXC_ReadImage2DArray(src0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    float4 x0, x1;\n\
    float4 y0, y1;\n\
    float4 tmpDst0, tmpDst1;\n\
    float4 data0, data1;\n\
\n\
    short zp0 = in_zp0;\n\
    short zp1 = in_zp1;\n\
    VXC_DP4x4(x0, src0, zp0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertUint8SubZpToFp32_4x4);\n\
    VXC_DP4x4(x1, src0, zp0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertSecUint8SubZpToFp32_4x4);\n\
    VXC_DP4x4(y0, src1, zp1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertUint8SubZpToFp32_4x4);\n\
    VXC_DP4x4(y1, src1, zp1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertSecUint8SubZpToFp32_4x4);\n\
\n\
    tmpDst0 = x0 * in_scale0 / (y0 * in_scale1);\n\
    data0 = floor(tmpDst0) * out_scale;\n\
    tmpDst1 = x1 * in_scale0 / (y1 * in_scale1);\n\
    data1 = floor(tmpDst1) * out_scale;\n\
\n\
    vxc_int4 tmpVal0, tmpVal1;\n\
    tmpVal0 = convert_int4_rte(data0 + out_zp);\n\
    tmpVal1 = convert_int4_rte(data1 + out_zp);\n\
\n\
    VXC_DP2x8(dst, tmpVal0, tmpVal1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1), uniConvertInt32toUint8_2x8);\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel __attribute__((reqd_work_group_size(8, 1, 1))) void vxcTensorFloorDiv_Int8(\n\
    image2d_array_t input0,\n\
    image2d_array_t input1,\n\
    image2d_array_t output)\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_char8 src0, src1;\n\
    vxc_char8 dst;\n\
    VXC_ReadImage2DArray(src0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    float4 x0, x1;\n\
    float4 y0, y1;\n\
    float4 tmpDst0, tmpDst1;\n\
    float4 data0, data1;\n\
    VXC_DP4x4(x0, src0, src0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertInt8FstFp32_4x4);\n\
    VXC_DP4x4(x1, src0, src0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertInt8SecFp32_4x4);\n\
    VXC_DP4x4(y0, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertInt8FstFp32_4x4);\n\
    VXC_DP4x4(y1, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertInt8SecFp32_4x4);\n\
\n\
    tmpDst0 = x0 * in_scale0 / (y0 * in_scale1);\n\
    data0 = floor(tmpDst0) * out_scale;\n\
    tmpDst1 = x1 * in_scale0 / (y1 * in_scale1);\n\
    data1 = floor(tmpDst1) * out_scale;\n\
\n\
    vxc_int4 tmpVal0, tmpVal1;\n\
    tmpVal0 = convert_int4_rte(data0);\n\
    tmpVal1 = convert_int4_rte(data1);\n\
    VXC_DP2x8(dst, tmpVal0, tmpVal1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1), uniConvertInt32toUint8_2x8);\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_floordiv_vx*/

static const char vsi_nn_kernel_fullconnect2_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform int loopNum;\n\
_viv_uniform VXC_512Bits uniMulAcc_16x1;\n\
__kernel void vsi_nn_kernel_fullconnect2(\n\
     __read_only image2d_array_t   input,\n\
     __read_only image2d_array_t   weight,\n\
     __read_only image2d_array_t   bias,\n\
     __write_only image2d_array_t  output)\n\
{\n\
    int4 coord_in = (int4)(16, get_global_id(0), get_global_id(1), 0);\n\
    int2 coord_out = (int2)(get_global_id(0), get_global_id(1));\n\
\n\
    vxc_short8 v0, v1, v2, v3, v4, v5, v6, v7;\n\
    vxc_half8 i0, i1, i2, i3;\n\
    vxc_half8 w0, w1, w2, w3;\n\
    float4 sum = 0;\n\
    float dst = 0;\n\
    dst = read_imagef(bias, coord_in.ywww).x;\n\
    do\n\
    {\n\
        VXC_ReadImage(v0, input,  coord_in.xz, VXC_5BITOFFSET_XY(-16, 0),\\\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        _viv_asm(COPY, i0, v0, 16);\n\
        VXC_ReadImage(v1, weight, coord_in.xy, VXC_5BITOFFSET_XY(-16, 0),\\\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        _viv_asm(COPY, w0, v1, 16);\n\
        VXC_ReadImage(v2, input,  coord_in.xz, VXC_5BITOFFSET_XY(-8, 0),\\\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        _viv_asm(COPY, i1, v2, 16);\n\
        VXC_ReadImage(v3, weight, coord_in.xy, VXC_5BITOFFSET_XY(-8, 0),\\\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        _viv_asm(COPY, w1, v3, 16);\n\
        VXC_ReadImage(v4, input,  coord_in.xz, VXC_5BITOFFSET_XY(0, 0),\\\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        _viv_asm(COPY, i2, v4, 16);\n\
        VXC_ReadImage(v5, weight, coord_in.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        _viv_asm(COPY, w2, v5, 16);\n\
        VXC_ReadImage(v6, input,  coord_in.xz, VXC_5BITOFFSET_XY(8, 0),\\\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        _viv_asm(COPY, i3, v6, 16);\n\
        VXC_ReadImage(v7, weight, coord_in.xy, VXC_5BITOFFSET_XY(8, 0),\\\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        _viv_asm(COPY, w3, v7, 16);\n\
\n\
        coord_in.x += 32;\n\
\n\
        VXC_DP16x1(sum, i0, w0, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0), uniMulAcc_16x1);\n\
        VXC_DP16x1(sum, i1, w1, VXC_MODIFIER(1, 1, 0, VXC_RM_TowardZero, 0), uniMulAcc_16x1);\n\
        VXC_DP16x1(sum, i2, w2, VXC_MODIFIER(2, 2, 0, VXC_RM_TowardZero, 0), uniMulAcc_16x1);\n\
        VXC_DP16x1(sum, i3, w3, VXC_MODIFIER(3, 3, 0, VXC_RM_TowardZero, 0), uniMulAcc_16x1);\n\
\n\
        float4 tmp = {1, 1, 1, 1};\n\
        dst = dst + dot(sum, tmp);\n\
\n\
    } while (coord_in.x < loopNum);\n\
\n\
    vxc_half v;\n\
    _viv_asm(CONV, v, dst);\n\
    _viv_asm(COPY, v0, v, 16);\n\
    VXC_WriteImage(output, coord_out.xy, v0, VXC_MODIFIER(0, 0, 0,VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_fullconnect2_vx*/

static const char vsi_nn_kernel_gather_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform int indices_num;\n\
\n\
__kernel void vxcGather_i8(\n\
    __read_only image2d_t   input0,\n\
    __read_only image2d_array_t   input1,\n\
    __write_only image2d_t  output,\n\
    int block_size,\n\
    int block_num,\n\
    int axis_num\n\
    )\n\
{\n\
    int gidx = get_global_id(0);  // block_size\n\
    int gidy = get_global_id(1);  // indices_num\n\
    int gidz = get_global_id(2);  // block_num\n\
\n\
    int4 coord_in = (int4)(gidy, 0, gidx, 0);\n\
    int4 indice = read_imagei(input1, coord_in.xyyy);\n\
    coord_in.w = gidz * axis_num + indice.x;\n\
\n\
    vxc_char16 src;\n\
    VXC_ReadImage(src, input0, coord_in.zw, 0, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    int2 coord = (int2)(gidx, gidz * indices_num + gidy);\n\
    VXC_WriteImage(output, coord, src, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcGather_i16(\n\
    __read_only image2d_t   input0,\n\
    __read_only image2d_array_t   input1,\n\
    __write_only image2d_t  output,\n\
    int block_size,\n\
    int block_num,\n\
    int axis_num\n\
    )\n\
{\n\
    int gidx = get_global_id(0);  // block_size\n\
    int gidy = get_global_id(1);  // indices_num\n\
    int gidz = get_global_id(2);  // block_num\n\
\n\
    int4 coord_in = (int4)(gidy, 0, gidx, 0);\n\
\n\
\n\
    int4 indice = read_imagei(input1, coord_in.xyyy);\n\
    coord_in.w = gidz * axis_num + indice.x;\n\
\n\
    vxc_short8 src;\n\
    VXC_ReadImage(src, input0, coord_in.zw, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    int2 coord = (int2)(gidx, gidz * indices_num + gidy);\n\
    VXC_WriteImage(output, coord, src, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_gather_vx*/

static const char vsi_nn_kernel_generate_proposals_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
__kernel void vxcGenerate_proposals(\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_array_t  output)\n\
{\n\
\n\
}\n\
"; /* end of vsi_nn_kernel_generate_proposals_vx*/

static const char vsi_nn_kernel_header_vx[] = "/*\n\
 ============================================================================\n\
 Name        : libNNExt.vx\n\
 Author      : VSI\n\
 Version     :\n\
 Copyright   : Your copyright notice\n\
 Description :\n\
 ============================================================================\n\
 */\n\
#include \"cl_viv_vx_ext.h\"\n\
\n\
#if (VX_VERSION==1)\n\
#define VXC_DP2x8_b_(dst, src0, src1, src2, info, uniform)\\\n\
do\\\n\
{\\\n\
    _viv_asm(COPY, dst, src0, 16); \\\n\
} while (0)\n\
\n\
#define VXC_VertMin3_Integer(dst, src0, src1, src2, info)\\\n\
do\\\n\
{\\\n\
    dst = min(src0, src1);\\\n\
    dst = min(src2, dst);\\\n\
} while (0)\n\
\n\
#define VXC_VertMin3_Half(dst, src0, src1, src2, info)\\\n\
do\\\n\
{\\\n\
    vxc_short8 val0, val1, val2, minVal, maxVal;\\\n\
    _viv_asm(COPY, val0, src0, 16);\\\n\
    _viv_asm(COPY, val1, src1, 16);\\\n\
    _viv_asm(COPY, val2, src2, 16);\\\n\
    maxVal = max(val0, val1);\\\n\
    maxVal = max(val2, maxVal);\\\n\
    minVal = min(val0, val1);\\\n\
    minVal = min(val2, minVal);\\\n\
    maxVal = maxVal >= 0 ? minVal : maxVal;\\\n\
    _viv_asm(COPY, dst, maxVal, 16); \\\n\
} while (0)\n\
\n\
#define VXC_VertMax3_Integer(dst, src0, src1, src2, info)\\\n\
do\\\n\
{\\\n\
    int startBin     = (info & VXC_START_BIN_BITMASK) >> 12;\\\n\
    int endBin         = (info & VXC_END_BIN_BITMASK) >> 8;\\\n\
    int sourceBin     = (info & VXC_SOURCE_BIN_BITMASK) >> 4;\\\n\
    int mod1 = VXC_MODIFIER_CLAMP(startBin, endBin, sourceBin, 0);\\\n\
    typeof (dst) tmp;\\\n\
    tmp = max(src0, src1);\\\n\
    tmp = max(src2, tmp);\\\n\
    VXC_Clamp(dst, tmp, tmp, tmp, mod1);\\\n\
} while (0)\n\
\n\
#define VXC_VertMax3_Half(dst, src0, src1, src2, info)\\\n\
 do\\\n\
 {\\\n\
     vxc_short8 val0, val1, val2, minVal, maxVal;\\\n\
     _viv_asm(COPY, val0, src0, 16);\\\n\
     _viv_asm(COPY, val1, src1, 16);\\\n\
     _viv_asm(COPY, val2, src2, 16);\\\n\
     maxVal = max(val0, val1);\\\n\
     maxVal = max(val2, maxVal);\\\n\
     minVal = min(val0, val1);\\\n\
     minVal = min(val2, minVal);\\\n\
     maxVal = maxVal >= 0 ? maxVal : minVal;\\\n\
     _viv_asm(COPY, dst, maxVal, 16); \\\n\
 } while (0)\n\
\n\
#define VXC_HorzMax3_Integer(dst, src0, info)\\\n\
do\\\n\
{\\\n\
    int startBin     = (info & VXC_START_BIN_BITMASK) >> 12;\\\n\
    int endBin         = (info & VXC_END_BIN_BITMASK) >> 8;\\\n\
    int sourceBin     = (info & VXC_SOURCE_BIN_BITMASK) >> 4;\\\n\
    int clamp         = (info & VXC_CLAMP_BITMASK) >> 22;\\\n\
    int mod1 = VXC_MODIFIER_FILTER(startBin, endBin, sourceBin, VXC_FM_Max, clamp);\\\n\
    VXC_OP4(filter, dst, src0, src0, src0, mod1);\\\n\
} while (0)\n\
\n\
#define VXC_HorzMax3_Half(dst, src0, info)\\\n\
do\\\n\
{\\\n\
    int startBin     = (info & VXC_START_BIN_BITMASK) >> 12;\\\n\
    int endBin         = (info & VXC_END_BIN_BITMASK) >> 8;\\\n\
    int sourceBin     = (info & VXC_SOURCE_BIN_BITMASK) >> 4;\\\n\
    int clamp         = (info & VXC_CLAMP_BITMASK) >> 22;\\\n\
    int mod1 = VXC_MODIFIER_FILTER(startBin, endBin, sourceBin, VXC_FM_Max, clamp);\\\n\
    int mod2 = VXC_MODIFIER_FILTER(startBin, endBin, sourceBin, VXC_FM_Min, clamp);\\\n\
    vxc_short8 val0, minVal, maxVal;\\\n\
    _viv_asm(COPY, val0, src0, 16);\\\n\
    VXC_OP4(filter, maxVal, val0, val0, val0, mod1);\\\n\
    VXC_OP4(filter, minVal, val0, val0, val0, mod2);\\\n\
    maxVal = maxVal >= 0 ? maxVal : minVal;\\\n\
    _viv_asm(COPY, dst, maxVal, 16);\\\n\
} while (0)\n\
\n\
#define VXC_HorzMin3_Integer(dst, src0, info)\\\n\
do\\\n\
{\\\n\
    int startBin     = (info & VXC_START_BIN_BITMASK) >> 12;\\\n\
    int endBin         = (info & VXC_END_BIN_BITMASK) >> 8;\\\n\
    int sourceBin     = (info & VXC_SOURCE_BIN_BITMASK) >> 4;\\\n\
    int clamp         = (info & VXC_CLAMP_BITMASK) >> 22;\\\n\
    int mod1 = VXC_MODIFIER_FILTER(startBin, endBin, sourceBin, VXC_FM_Min, clamp);\\\n\
    VXC_OP4(filter, dst, src0, src0, src0, mod1);\\\n\
} while (0)\n\
\n\
#define VXC_HorzMin3_Half(dst, src0, info)\\\n\
do\\\n\
{\\\n\
    int startBin     = (info & VXC_START_BIN_BITMASK) >> 12;\\\n\
    int endBin         = (info & VXC_END_BIN_BITMASK) >> 8;\\\n\
    int sourceBin     = (info & VXC_SOURCE_BIN_BITMASK) >> 4;\\\n\
    int clamp         = (info & VXC_CLAMP_BITMASK) >> 22;\\\n\
    int mod1 = VXC_MODIFIER_FILTER(startBin, endBin, sourceBin, VXC_FM_Max, clamp);\\\n\
    int mod2 = VXC_MODIFIER_FILTER(startBin, endBin, sourceBin, VXC_FM_Min, clamp);\\\n\
    int mod3 = VXC_MODIFIER_FILTER(startBin, endBin, sourceBin, VXC_FM_Median, clamp);\\\n\
    vxc_short8 val0, minVal, maxVal, midVal;\\\n\
    _viv_asm(COPY, val0, src0, 16);\\\n\
    VXC_OP4(filter, maxVal, val0, val0, val0, mod1);\\\n\
    VXC_OP4(filter, minVal, val0, val0, val0, mod2);\\\n\
    VXC_OP4(filter, midVal, val0, val0, val0, mod3);\\\n\
    minVal = midVal  < 0 ? midVal : minVal;\\\n\
    minVal = maxVal  < 0 ? maxVal : minVal;\\\n\
    _viv_asm(COPY, dst, minVal, 16);\\\n\
} while (0)\n\
\n\
#define VXC_Clamp_Half(dst, src0, src1, src2, info)\\\n\
do\\\n\
{\\\n\
    VXC_VertMax3_Half(dst, src0, src0, src1, info);\\\n\
    VXC_VertMin3_Half(dst, dst, dst, src2, info);\\\n\
} while (0)\n\
\n\
#else\n\
#define VXC_DP2x8_b_(dst, src0, src1, src2, info, uniform)\\\n\
do\\\n\
{\\\n\
    VXC_DP2x8_b(dst, src0, src1, src2, info, uniform); \\\n\
} while (0)\n\
\n\
#define VXC_VertMin3_Integer(dst, src0, src1, src2, info)\\\n\
 do\\\n\
 {\\\n\
    VXC_VertMin3(dst, src0, src1, src2, info);\\\n\
 } while (0)\n\
\n\
#define VXC_VertMin3_Half(dst, src0, src1, src2, info)\\\n\
 do\\\n\
 {\\\n\
    VXC_VertMin3(dst, src0, src1, src2, info);\\\n\
 } while (0)\n\
\n\
#define VXC_VertMax3_Integer(dst, src0, src1, src2, info)\\\n\
do\\\n\
{\\\n\
    VXC_VertMax3(dst, src0, src1, src2, info);\\\n\
} while (0)\n\
\n\
#define VXC_VertMax3_Half(dst, src0, src1, src2, info)\\\n\
do\\\n\
{\\\n\
    VXC_VertMax3(dst, src0, src1, src2, info);\\\n\
} while (0)\n\
\n\
#define VXC_HorzMax3_Integer(dst, src0, info)\\\n\
do\\\n\
{\\\n\
    VXC_HorzMax3(dst, src0, info);\\\n\
} while (0)\n\
\n\
#define VXC_HorzMax3_Half(dst, src0, info)\\\n\
do\\\n\
{\\\n\
    VXC_HorzMax3(dst, src0, info);\\\n\
} while (0)\n\
\n\
#define VXC_HorzMin3_Integer(dst, src0, info)\\\n\
do\\\n\
{\\\n\
    VXC_HorzMin3(dst, src0, info);\\\n\
} while (0)\n\
\n\
#define VXC_HorzMin3_Half(dst, src0, info)\\\n\
do\\\n\
{\\\n\
    VXC_HorzMin3(dst, src0, info);\\\n\
} while (0)\n\
\n\
#define VXC_Clamp_Half(dst, src0, src1, src2, info)\\\n\
do\\\n\
{\\\n\
    VXC_Clamp(dst, src0, src1, src2, info);\\\n\
} while (0)\n\
#endif\n\
"; /* end of vsi_nn_kernel_header_vx*/

static const char vsi_nn_kernel_heatmap_max_keypoint_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
__kernel void vxcHeatmap_max_keypoint(\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_array_t  output)\n\
{\n\
\n\
}\n\
"; /* end of vsi_nn_kernel_heatmap_max_keypoint_vx*/

static const char vsi_nn_kernel_imageprocess_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform VXC_512Bits uniVecShift10;\n\
_viv_uniform VXC_512Bits uniAddRShift;\n\
_viv_uniform VXC_512Bits uniGetTempVal;\n\
_viv_uniform VXC_512Bits uniExtractBytes;\n\
_viv_uniform VXC_512Bits uniUnpackToR;\n\
_viv_uniform VXC_512Bits uniUnpackToG;\n\
_viv_uniform VXC_512Bits uniUnpackToB;\n\
_viv_uniform VXC_512Bits uniDataMulAlpha_4x4;\n\
_viv_uniform VXC_512Bits uniDataSubMean_4x4;\n\
\n\
_viv_uniform VXC_512Bits uniConvertIntergetoF32_4x4;\n\
_viv_uniform float outputScale;\n\
_viv_uniform VXC_512Bits uniExtactInteger_2x8;\n\
\n\
#define DESCALE(x) (((x) + (1<<19)) >> 20)\n\
__kernel void ScaletoTensor_Int8\n\
    (\n\
    __read_only image2d_t        input,\n\
    __write_only image2d_array_t output,\n\
        global int               *xRatio,\n\
        global int               *yRatio,\n\
        global int               *xOffset,\n\
        global int               *yOffset,\n\
               float             rMean,\n\
               float             gMean,\n\
               float             bMean,\n\
               float             f32Var\n\
    )\n\
{\n\
    int2 ratioXY = (int2)(*xRatio, *yRatio);\n\
\n\
    int4 xPos        = get_global_id(0);\n\
    int yPos        = get_global_id(1);\n\
\n\
    int2 ratioSufXY = (ratioXY >> 1) - (1 << 14);\n\
    xPos += (int4)(0, 1, 2, 3);\n\
\n\
    //x\n\
    int4 fx0 = xPos * ratioXY.x + ratioSufXY.x;\n\
    int4 sx = fx0 & 0xffff8000;\n\
    fx0 -= sx;\n\
    sx = sx >> 15;\n\
\n\
    vxc_short4 fx;\n\
    VXC_DP4x4(fx, fx0, 1 << 4, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniAddRShift);\n\
    //y\n\
    int fy = yPos * ratioXY.y + ratioSufXY.y;\n\
    int sy = fy & 0xffff8000; // Floor\n\
\n\
    fy -= sy;\n\
    sy = sy >> 15;\n\
\n\
    fy = (fy + (1<< 4)) >> 5;\n\
\n\
    //R\n\
    vxc_uchar16 line0RGB1, line0RGB2;\n\
    vxc_uchar16 line1RGB3, line1RGB4;\n\
    int4 coord;\n\
    sx = sx * 3 + *xOffset;\n\
    coord.xyz    = sx.xyz;\n\
    coord.w        = sy + *yOffset;\n\
    int2 coord1 = (int2)(sx.w, coord.w);\n\
    VXC_ReadImage(line0RGB1, input, coord.xw, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line0RGB1, input, coord.yw, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(6, 11, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line0RGB2, input, coord.zw, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line0RGB2, input, coord1, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(6, 11, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_ReadImage(line1RGB3, input, coord.xw, VXC_5BITOFFSET_XY(0, 1),\\\n\
        VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line1RGB3, input, coord.yw, VXC_5BITOFFSET_XY(0, 1),\\\n\
        VXC_MODIFIER(6, 11, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line1RGB4, input, coord.zw, VXC_5BITOFFSET_XY(0, 1),\\\n\
        VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line1RGB4, input, coord1, VXC_5BITOFFSET_XY(0, 1),\\\n\
        VXC_MODIFIER(6, 11, 0, VXC_RM_TowardZero, 0));\n\
\n\
    float4      bgrMean = (float4)(bMean, gMean, rMean, 0);\n\
\n\
    bgrMean *= f32Var;\n\
\n\
    int4 test01, temp1;\n\
    int4 test02, temp2;\n\
    int4 tt;\n\
    vxc_uchar4 val;\n\
    int4 coord_out = (int4)(xPos.x, yPos, 2, 0);\n\
\n\
    vxc_uchar8 line1, line2;\n\
\n\
    //R\n\
    VXC_DP2x8(line1, line0RGB1, line0RGB2,\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniUnpackToR);\n\
    VXC_DP2x8(line2, line1RGB3, line1RGB4,\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniUnpackToR);\n\
\n\
    VXC_DP4x4(test01, line1, line1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10);\n\
    VXC_DP4x4(temp1, line1, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal);\n\
    temp1 = temp1 + test01;\n\
\n\
    VXC_DP4x4(test02, line2, line2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10);\n\
    VXC_DP4x4(temp2, line2, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal);\n\
    temp2 = temp2 + test02;\n\
    temp2 = fy * (temp2 - temp1) + (temp1 << 10);\n\
\n\
    vxc_float4    tmp_dst;\n\
    vxc_uchar4 u8_dst;\n\
    VXC_DP4x4(u8_dst, temp2, 1 << 19, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), uniExtractBytes);\n\
    VXC_DP4x4(tmp_dst, u8_dst, u8_dst,\\\n\
        VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), uniConvertIntergetoF32_4x4);\n\
\n\
    //convert U8 to dfp8\n\
    int4 dst0;\n\
    vxc_char4 dst;\n\
    tmp_dst = tmp_dst * f32Var - bgrMean.z;\n\
    tmp_dst *= outputScale;\n\
    dst0 = convert_int4_rte(tmp_dst);\n\
    VXC_DP2x8(dst, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniExtactInteger_2x8);\n\
\n\
    VXC_WriteImage2DArray(output, coord_out, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
\n\
    //G\n\
    VXC_DP2x8(line1, line0RGB1, line0RGB2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniUnpackToG);\n\
    VXC_DP2x8(line2, line1RGB3, line1RGB4, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniUnpackToG);\n\
\n\
    coord_out.z = 1;\n\
    VXC_DP4x4(test01, line1, line1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10);\n\
    VXC_DP4x4(temp1, line1, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal);\n\
    temp1 = temp1 + test01;\n\
\n\
    VXC_DP4x4(test02, line2, line2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10);\n\
    VXC_DP4x4(temp2, line2, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal);\n\
    temp2 = temp2 + test02;\n\
    temp2 = fy * (temp2 - temp1) + (temp1 << 10);\n\
\n\
    VXC_DP4x4(u8_dst, temp2, 1 << 19, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), uniExtractBytes);\n\
    VXC_DP4x4(tmp_dst, u8_dst, u8_dst,\\\n\
        VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), uniConvertIntergetoF32_4x4);\n\
\n\
    tmp_dst = tmp_dst * f32Var - bgrMean.y;\n\
    tmp_dst *= outputScale;\n\
    dst0 = convert_int4_rte(tmp_dst);\n\
    VXC_DP2x8(dst, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniExtactInteger_2x8);\n\
    VXC_WriteImage2DArray(output, coord_out, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
\n\
    //B\n\
    VXC_DP2x8(line1, line0RGB1, line0RGB2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniUnpackToB);\n\
    VXC_DP2x8(line2, line1RGB3, line1RGB4, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniUnpackToB);\n\
\n\
    coord_out.z = 0;\n\
    VXC_DP4x4(test01, line1, line1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10);\n\
    VXC_DP4x4(temp1, line1, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal);\n\
    temp1 = temp1 + test01;\n\
\n\
    VXC_DP4x4(test02, line2, line2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10);\n\
    VXC_DP4x4(temp2, line2, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal);\n\
    temp2 = temp2 + test02;\n\
    temp2 = fy * (temp2 - temp1) + (temp1 << 10);\n\
\n\
    VXC_DP4x4(u8_dst, temp2, 1 << 19,\\\n\
        VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), uniExtractBytes);\n\
    VXC_DP4x4(tmp_dst, u8_dst, u8_dst,\\\n\
        VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), uniConvertIntergetoF32_4x4);\n\
\n\
    tmp_dst = tmp_dst * f32Var - bgrMean.x;\n\
    tmp_dst *= outputScale;\n\
    dst0 = convert_int4_rte(tmp_dst);\n\
    VXC_DP2x8(dst, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniExtactInteger_2x8);\n\
    VXC_WriteImage2DArray(output, coord_out, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void ScaletoTensor_Fp16\n\
    (\n\
    __read_only image2d_t        input,\n\
    __write_only image2d_array_t output,\n\
        global int               *xRatio,\n\
        global int               *yRatio,\n\
        global int               *xOffset,\n\
        global int               *yOffset,\n\
                float            rMean,\n\
                float            gMean,\n\
                float            bMean,\n\
                float            f32Var\n\
    )\n\
{\n\
    int2 ratioXY = (int2)(*xRatio, *yRatio);\n\
\n\
    int4 xPos       = get_global_id(0);\n\
    int yPos        = get_global_id(1);\n\
\n\
    int2 ratioSufXY = (ratioXY >> 1) - (1 << 14);\n\
    xPos += (int4)(0, 1, 2, 3);\n\
\n\
    //x\n\
    int4 fx0 = xPos * ratioXY.x + ratioSufXY.x;\n\
    int4 sx = fx0 & 0xffff8000;\n\
    fx0 -= sx;\n\
    sx = sx >> 15;\n\
\n\
    vxc_short4 fx;\n\
    VXC_DP4x4(fx, fx0, 1 << 4, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniAddRShift);\n\
    //y\n\
    int fy = yPos * ratioXY.y + ratioSufXY.y;\n\
    int sy = fy & 0xffff8000; // Floor\n\
\n\
    fy -= sy;\n\
    sy = sy >> 15;\n\
\n\
    fy = (fy + (1<< 4)) >> 5;\n\
\n\
    //R\n\
    vxc_uchar16 line0RGB1, line0RGB2;\n\
    vxc_uchar16 line1RGB3, line1RGB4;\n\
    int4 coord;\n\
    sx = sx * 3 + *xOffset;\n\
    coord.xyz    = sx.xyz;\n\
    coord.w        = sy + *yOffset;\n\
    int2 coord1 = (int2)(sx.w, coord.w);\n\
    VXC_ReadImage(line0RGB1, input, coord.xw,\\\n\
        VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line0RGB1, input, coord.yw,\\\n\
        VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(6, 11, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line0RGB2, input, coord.zw,\\\n\
        VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line0RGB2, input, coord1,\\\n\
        VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(6, 11, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_ReadImage(line1RGB3, input, coord.xw,\\\n\
        VXC_5BITOFFSET_XY(0, 1), VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line1RGB3, input, coord.yw,\\\n\
        VXC_5BITOFFSET_XY(0, 1), VXC_MODIFIER(6, 11, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line1RGB4, input, coord.zw,\\\n\
        VXC_5BITOFFSET_XY(0, 1), VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line1RGB4, input, coord1,\\\n\
        VXC_5BITOFFSET_XY(0, 1), VXC_MODIFIER(6, 11, 0, VXC_RM_TowardZero, 0));\n\
\n\
    float4      bgrMean = (float4)(bMean, gMean, rMean, 0);\n\
\n\
    int4 test01, temp1;\n\
    int4 test02, temp2;\n\
    int4 tt;\n\
    vxc_uchar4 val;\n\
    int4 coord_out = (int4)(xPos.x, yPos, 2, 0);\n\
\n\
    vxc_uchar8 line1, line2;\n\
\n\
    //R\n\
    VXC_DP2x8(line1, line0RGB1, line0RGB2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniUnpackToR);\n\
    VXC_DP2x8(line2, line1RGB3, line1RGB4, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniUnpackToR);\n\
\n\
    VXC_DP4x4(test01, line1, line1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10);\n\
    VXC_DP4x4(temp1, line1, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal);\n\
    temp1 = temp1 + test01;\n\
\n\
    VXC_DP4x4(test02, line2, line2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10);\n\
    VXC_DP4x4(temp2, line2, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal);\n\
    temp2 = temp2 + test02;\n\
    temp2 = fy * (temp2 - temp1) + (temp1 << 10);\n\
\n\
    VXC_DP4x4(val, temp2, 1 << 19, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), uniExtractBytes);\n\
\n\
    //convert U8 to FP16\n\
    half4 f16mean;\n\
    half f16alpha;\n\
    vxc_half4    dst;\n\
    vxc_short4 tmp_dst;\n\
    _viv_asm(CONV, f16mean, bgrMean);\n\
    _viv_asm(CONV, f16alpha, f32Var);\n\
    VXC_DP4x4(dst, val, f16mean.z, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniDataSubMean_4x4);\n\
    VXC_DP4x4(dst, dst, f16alpha, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniDataMulAlpha_4x4);\n\
    _viv_asm(COPY, tmp_dst, dst, 8);\n\
    VXC_WriteImage2DArray(output, coord_out, tmp_dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
\n\
    //G\n\
    VXC_DP2x8(line1, line0RGB1, line0RGB2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniUnpackToG);\n\
    VXC_DP2x8(line2, line1RGB3, line1RGB4, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniUnpackToG);\n\
\n\
    coord_out.z = 1;\n\
    VXC_DP4x4(test01, line1, line1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10);\n\
    VXC_DP4x4(temp1, line1, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal);\n\
    temp1 = temp1 + test01;\n\
\n\
    VXC_DP4x4(test02, line2, line2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10);\n\
    VXC_DP4x4(temp2, line2, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal);\n\
    temp2 = temp2 + test02;\n\
    temp2 = fy * (temp2 - temp1) + (temp1 << 10);\n\
\n\
    VXC_DP4x4(val, temp2, 1 << 19, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), uniExtractBytes);\n\
\n\
    VXC_DP4x4(dst, val, f16mean.y, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniDataSubMean_4x4);\n\
    VXC_DP4x4(dst, dst, f16alpha, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniDataMulAlpha_4x4);\n\
    _viv_asm(COPY, tmp_dst, dst, 8);\n\
    VXC_WriteImage2DArray(output, coord_out, tmp_dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
\n\
    //B\n\
    VXC_DP2x8(line1, line0RGB1, line0RGB2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniUnpackToB);\n\
    VXC_DP2x8(line2, line1RGB3, line1RGB4, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniUnpackToB);\n\
\n\
    coord_out.z = 0;\n\
    VXC_DP4x4(test01, line1, line1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10);\n\
    VXC_DP4x4(temp1, line1, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal);\n\
    temp1 = temp1 + test01;\n\
\n\
    VXC_DP4x4(test02, line2, line2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10);\n\
    VXC_DP4x4(temp2, line2, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal);\n\
    temp2 = temp2 + test02;\n\
    temp2 = fy * (temp2 - temp1) + (temp1 << 10);\n\
\n\
    VXC_DP4x4(val, temp2, 1 << 19, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), uniExtractBytes);\n\
\n\
    VXC_DP4x4(dst, val, f16mean.x, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniDataSubMean_4x4);\n\
    VXC_DP4x4(dst, dst, f16alpha, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniDataMulAlpha_4x4);\n\
    _viv_asm(COPY, tmp_dst, dst, 8);\n\
    VXC_WriteImage2DArray(output, coord_out, tmp_dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
\n\
}\n\
\n\
"; /* end of vsi_nn_kernel_imageprocess_vx*/

static const char vsi_nn_kernel_imageprocess_2_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform VXC_512Bits uniVecShift10;\n\
_viv_uniform VXC_512Bits uniAddRShift;\n\
_viv_uniform VXC_512Bits uniGetTempVal;\n\
_viv_uniform VXC_512Bits uniExtractBytes;\n\
_viv_uniform VXC_512Bits uniUnpackToR;\n\
_viv_uniform VXC_512Bits uniUnpackToG;\n\
_viv_uniform VXC_512Bits uniUnpackToB;\n\
_viv_uniform VXC_512Bits uniDataMulAlpha_4x4;\n\
_viv_uniform VXC_512Bits uniDataSubMean_4x4;\n\
\n\
_viv_uniform VXC_512Bits uniConvertIntergetoF32_4x4;\n\
_viv_uniform float outputScale;\n\
_viv_uniform VXC_512Bits uniExtactInteger_2x8;\n\
\n\
#define DESCALE(x) (((x) + (1<<19)) >> 20)\n\
__kernel void ScaletoTensor_Int16\n\
    (\n\
    __read_only image2d_t        input,\n\
    __write_only image2d_array_t output,\n\
        global int               *xRatio,\n\
        global int               *yRatio,\n\
        global int               *xOffset,\n\
        global int               *yOffset,\n\
               float             rMean,\n\
               float             gMean,\n\
               float             bMean,\n\
               float             f32Var\n\
    )\n\
{\n\
    int2 ratioXY = (int2)(*xRatio, *yRatio);\n\
\n\
    int4 xPos        = get_global_id(0);\n\
    int yPos        = get_global_id(1);\n\
\n\
    int2 ratioSufXY = (ratioXY >> 1) - (1 << 14);\n\
    xPos += (int4)(0, 1, 2, 3);\n\
\n\
    //x\n\
    int4 fx0 = xPos * ratioXY.x + ratioSufXY.x;\n\
    int4 sx = fx0 & 0xffff8000;\n\
    fx0 -= sx;\n\
    sx = sx >> 15;\n\
\n\
    vxc_short4 fx;\n\
    VXC_DP4x4(fx, fx0, 1 << 4, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniAddRShift);\n\
    //y\n\
    int fy = yPos * ratioXY.y + ratioSufXY.y;\n\
    int sy = fy & 0xffff8000; // Floor\n\
\n\
    fy -= sy;\n\
    sy = sy >> 15;\n\
\n\
    fy = (fy + (1<< 4)) >> 5;\n\
\n\
    //R\n\
    vxc_uchar16 line0RGB1, line0RGB2;\n\
    vxc_uchar16 line1RGB3, line1RGB4;\n\
    int4 coord;\n\
    sx = sx * 3 + *xOffset;\n\
    coord.xyz    = sx.xyz;\n\
    coord.w        = sy + *yOffset;\n\
    int2 coord1 = (int2)(sx.w, coord.w);\n\
    VXC_ReadImage(line0RGB1, input, coord.xw,\\\n\
        VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line0RGB1, input, coord.yw,\\\n\
        VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(6, 11, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line0RGB2, input, coord.zw,\\\n\
        VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line0RGB2, input, coord1,\\\n\
        VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(6, 11, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_ReadImage(line1RGB3, input, coord.xw,\\\n\
        VXC_5BITOFFSET_XY(0, 1), VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line1RGB3, input, coord.yw,\\\n\
        VXC_5BITOFFSET_XY(0, 1), VXC_MODIFIER(6, 11, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line1RGB4, input, coord.zw,\\\n\
        VXC_5BITOFFSET_XY(0, 1), VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line1RGB4, input, coord1,\\\n\
        VXC_5BITOFFSET_XY(0, 1), VXC_MODIFIER(6, 11, 0, VXC_RM_TowardZero, 0));\n\
\n\
    float4      bgrMean = (float4)(bMean, gMean, rMean, 0);\n\
\n\
    bgrMean *= f32Var;\n\
\n\
    int4 test01, temp1;\n\
    int4 test02, temp2;\n\
    int4 tt;\n\
    vxc_uchar4 val;\n\
    int4 coord_out = (int4)(xPos.x, yPos, 2, 0);\n\
\n\
    vxc_uchar8 line1, line2;\n\
\n\
    //R\n\
    VXC_DP2x8(line1, line0RGB1, line0RGB2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniUnpackToR);\n\
    VXC_DP2x8(line2, line1RGB3, line1RGB4, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniUnpackToR);\n\
\n\
    VXC_DP4x4(test01, line1, line1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10);\n\
    VXC_DP4x4(temp1, line1, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal);\n\
    temp1 = temp1 + test01;\n\
\n\
    VXC_DP4x4(test02, line2, line2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10);\n\
    VXC_DP4x4(temp2, line2, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal);\n\
    temp2 = temp2 + test02;\n\
    temp2 = fy * (temp2 - temp1) + (temp1 << 10);\n\
\n\
    vxc_float4    tmp_dst;\n\
    vxc_uchar4 u8_dst;\n\
    VXC_DP4x4(u8_dst, temp2, 1 << 19, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), uniExtractBytes);\n\
    VXC_DP4x4(tmp_dst, u8_dst, u8_dst,\\\n\
        VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), uniConvertIntergetoF32_4x4);\n\
\n\
    //convert U8 to dfp8\n\
    int4 dst0;\n\
    vxc_short4 dst;\n\
    tmp_dst = tmp_dst * f32Var - bgrMean.z;\n\
    tmp_dst *= outputScale;\n\
    dst0 = convert_int4_rte(tmp_dst);\n\
    VXC_DP2x8(dst, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniExtactInteger_2x8);\n\
\n\
    VXC_WriteImage2DArray(output, coord_out, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
\n\
    //G\n\
    VXC_DP2x8(line1, line0RGB1, line0RGB2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniUnpackToG);\n\
    VXC_DP2x8(line2, line1RGB3, line1RGB4, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniUnpackToG);\n\
\n\
    coord_out.z = 1;\n\
    VXC_DP4x4(test01, line1, line1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10);\n\
    VXC_DP4x4(temp1, line1, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal);\n\
    temp1 = temp1 + test01;\n\
\n\
    VXC_DP4x4(test02, line2, line2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10);\n\
    VXC_DP4x4(temp2, line2, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal);\n\
    temp2 = temp2 + test02;\n\
    temp2 = fy * (temp2 - temp1) + (temp1 << 10);\n\
\n\
    VXC_DP4x4(u8_dst, temp2, 1 << 19, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), uniExtractBytes);\n\
    VXC_DP4x4(tmp_dst, u8_dst, u8_dst,\\\n\
        VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), uniConvertIntergetoF32_4x4);\n\
\n\
    tmp_dst = tmp_dst * f32Var - bgrMean.y;\n\
    tmp_dst *= outputScale;\n\
    dst0 = convert_int4_rte(tmp_dst);\n\
    VXC_DP2x8(dst, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniExtactInteger_2x8);\n\
    VXC_WriteImage2DArray(output, coord_out, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
\n\
    //B\n\
    VXC_DP2x8(line1, line0RGB1, line0RGB2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniUnpackToB);\n\
    VXC_DP2x8(line2, line1RGB3, line1RGB4, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniUnpackToB);\n\
\n\
    coord_out.z = 0;\n\
    VXC_DP4x4(test01, line1, line1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10);\n\
    VXC_DP4x4(temp1, line1, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal);\n\
    temp1 = temp1 + test01;\n\
\n\
    VXC_DP4x4(test02, line2, line2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10);\n\
    VXC_DP4x4(temp2, line2, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal);\n\
    temp2 = temp2 + test02;\n\
    temp2 = fy * (temp2 - temp1) + (temp1 << 10);\n\
\n\
    VXC_DP4x4(u8_dst, temp2, 1 << 19, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), uniExtractBytes);\n\
    VXC_DP4x4(tmp_dst, u8_dst, u8_dst,\\\n\
        VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), uniConvertIntergetoF32_4x4);\n\
\n\
    tmp_dst = tmp_dst * f32Var - bgrMean.x;\n\
    tmp_dst *= outputScale;\n\
    dst0 = convert_int4_rte(tmp_dst);\n\
    VXC_DP2x8(dst, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniExtactInteger_2x8);\n\
    VXC_WriteImage2DArray(output, coord_out, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform float outputZP;\n\
__kernel void ScaletoTensor_UInt8\n\
    (\n\
    __read_only image2d_t        input,\n\
    __write_only image2d_array_t output,\n\
        global int               *xRatio,\n\
        global int               *yRatio,\n\
        global int               *xOffset,\n\
        global int               *yOffset,\n\
               float             rMean,\n\
               float             gMean,\n\
               float             bMean,\n\
               float             f32Var\n\
    )\n\
{\n\
    int2 ratioXY = (int2)(*xRatio, *yRatio);\n\
\n\
    int4 xPos        = get_global_id(0);\n\
    int yPos        = get_global_id(1);\n\
\n\
    int2 ratioSufXY = (ratioXY >> 1) - (1 << 14);\n\
    xPos += (int4)(0, 1, 2, 3);\n\
\n\
    //x\n\
    int4 fx0 = xPos * ratioXY.x + ratioSufXY.x;\n\
    int4 sx = fx0 & 0xffff8000;\n\
    fx0 -= sx;\n\
    sx = sx >> 15;\n\
\n\
    vxc_short4 fx;\n\
    VXC_DP4x4(fx, fx0, 1 << 4, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniAddRShift);\n\
    //y\n\
    int fy = yPos * ratioXY.y + ratioSufXY.y;\n\
    int sy = fy & 0xffff8000; // Floor\n\
\n\
    fy -= sy;\n\
    sy = sy >> 15;\n\
\n\
    fy = (fy + (1<< 4)) >> 5;\n\
\n\
    //R\n\
    vxc_uchar16 line0RGB1, line0RGB2;\n\
    vxc_uchar16 line1RGB3, line1RGB4;\n\
    int4 coord;\n\
    sx = sx * 3 + *xOffset;\n\
    coord.xyz    = sx.xyz;\n\
    coord.w        = sy + *yOffset;\n\
    int2 coord1 = (int2)(sx.w, coord.w);\n\
    VXC_ReadImage(line0RGB1, input, coord.xw,\\\n\
        VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line0RGB1, input, coord.yw,\\\n\
        VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(6, 11, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line0RGB2, input, coord.zw,\\\n\
        VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line0RGB2, input, coord1,\\\n\
        VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(6, 11, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_ReadImage(line1RGB3, input, coord.xw,\\\n\
        VXC_5BITOFFSET_XY(0, 1), VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line1RGB3, input, coord.yw,\\\n\
        VXC_5BITOFFSET_XY(0, 1), VXC_MODIFIER(6, 11, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line1RGB4, input, coord.zw,\\\n\
        VXC_5BITOFFSET_XY(0, 1), VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line1RGB4, input, coord1,\\\n\
        VXC_5BITOFFSET_XY(0, 1), VXC_MODIFIER(6, 11, 0, VXC_RM_TowardZero, 0));\n\
\n\
    float4      bgrMean = (float4)(bMean, gMean, rMean, 0);\n\
\n\
    bgrMean *= f32Var;\n\
\n\
    int4 test01, temp1;\n\
    int4 test02, temp2;\n\
    int4 tt;\n\
    vxc_uchar4 val;\n\
    int4 coord_out = (int4)(xPos.x, yPos, 2, 0);\n\
\n\
    vxc_uchar8 line1, line2;\n\
\n\
    //R\n\
    VXC_DP2x8(line1, line0RGB1, line0RGB2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniUnpackToR);\n\
    VXC_DP2x8(line2, line1RGB3, line1RGB4, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniUnpackToR);\n\
\n\
    VXC_DP4x4(test01, line1, line1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10);\n\
    VXC_DP4x4(temp1, line1, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal);\n\
    temp1 = temp1 + test01;\n\
\n\
    VXC_DP4x4(test02, line2, line2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10);\n\
    VXC_DP4x4(temp2, line2, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal);\n\
    temp2 = temp2 + test02;\n\
    temp2 = fy * (temp2 - temp1) + (temp1 << 10);\n\
\n\
    vxc_float4    tmp_dst;\n\
    vxc_uchar4 u8_dst;\n\
    VXC_DP4x4(u8_dst, temp2, 1 << 19, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), uniExtractBytes);\n\
    VXC_DP4x4(tmp_dst, u8_dst, u8_dst,\\\n\
        VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), uniConvertIntergetoF32_4x4);\n\
\n\
    //convert U8 to dfp8\n\
    int4 dst0;\n\
    vxc_uchar4 dst;\n\
    tmp_dst = tmp_dst * f32Var - bgrMean.z;\n\
    tmp_dst = tmp_dst * outputScale + outputZP;\n\
    dst0 = convert_int4_rte(tmp_dst);\n\
    VXC_DP2x8(dst, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniExtactInteger_2x8);\n\
\n\
    VXC_WriteImage2DArray(output, coord_out, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
\n\
    //G\n\
    VXC_DP2x8(line1, line0RGB1, line0RGB2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniUnpackToG);\n\
    VXC_DP2x8(line2, line1RGB3, line1RGB4, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniUnpackToG);\n\
\n\
    coord_out.z = 1;\n\
    VXC_DP4x4(test01, line1, line1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10);\n\
    VXC_DP4x4(temp1, line1, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal);\n\
    temp1 = temp1 + test01;\n\
\n\
    VXC_DP4x4(test02, line2, line2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10);\n\
    VXC_DP4x4(temp2, line2, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal);\n\
    temp2 = temp2 + test02;\n\
    temp2 = fy * (temp2 - temp1) + (temp1 << 10);\n\
\n\
    VXC_DP4x4(u8_dst, temp2, 1 << 19, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), uniExtractBytes);\n\
    VXC_DP4x4(tmp_dst, u8_dst, u8_dst,\\\n\
        VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), uniConvertIntergetoF32_4x4);\n\
\n\
    tmp_dst = tmp_dst * f32Var - bgrMean.y;\n\
    tmp_dst = tmp_dst * outputScale + outputZP;\n\
    dst0 = convert_int4_rte(tmp_dst);\n\
    VXC_DP2x8(dst, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniExtactInteger_2x8);\n\
    VXC_WriteImage2DArray(output, coord_out, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
\n\
    //B\n\
    VXC_DP2x8(line1, line0RGB1, line0RGB2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniUnpackToB);\n\
    VXC_DP2x8(line2, line1RGB3, line1RGB4, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniUnpackToB);\n\
\n\
    coord_out.z = 0;\n\
    VXC_DP4x4(test01, line1, line1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10);\n\
    VXC_DP4x4(temp1, line1, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal);\n\
    temp1 = temp1 + test01;\n\
\n\
    VXC_DP4x4(test02, line2, line2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10);\n\
    VXC_DP4x4(temp2, line2, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal);\n\
    temp2 = temp2 + test02;\n\
    temp2 = fy * (temp2 - temp1) + (temp1 << 10);\n\
\n\
    VXC_DP4x4(u8_dst, temp2, 1 << 19, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), uniExtractBytes);\n\
    VXC_DP4x4(tmp_dst, u8_dst, u8_dst,\\\n\
        VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), uniConvertIntergetoF32_4x4);\n\
\n\
    tmp_dst = tmp_dst * f32Var - bgrMean.x;\n\
    tmp_dst = tmp_dst * outputScale + outputZP;\n\
    dst0 = convert_int4_rte(tmp_dst);\n\
    VXC_DP2x8(dst, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniExtactInteger_2x8);\n\
    VXC_WriteImage2DArray(output, coord_out, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_imageprocess_2_vx*/

static const char vsi_nn_kernel_imageprocess_3_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform VXC_512Bits uniExtractR_2x8;\n\
_viv_uniform VXC_512Bits uniExtractG_2x8;\n\
_viv_uniform VXC_512Bits uniExtractB_2x8;\n\
_viv_uniform float outputScale;\n\
__kernel void ScaletoTensor_Fp16_copy\n\
    (\n\
    __read_only image2d_t        input,\n\
    __write_only image2d_array_t output,\n\
         global int              *xRatio,\n\
         global int              *yRatio,\n\
         global int              *xOffset,\n\
         global int              *yOffset,\n\
                float            rMean,\n\
                float            gMean,\n\
                float            bMean,\n\
                float            f32Var\n\
    )\n\
{\n\
    int2 coord      = (int2)(get_global_id(0) * 3, get_global_id(1));\n\
\n\
    coord.xy += (int2) (*xOffset, *yOffset);\n\
    vxc_uchar16 src0, src1;\n\
    vxc_half8   dst;\n\
\n\
    VXC_ReadImage(src0, input, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(src1, input, coord.xy, VXC_5BITOFFSET_XY(15, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    float4      paramData = (float4)(rMean * f32Var, gMean * f32Var, bMean * f32Var, f32Var);\n\
    //convert U8 to FP16\n\
    half4 paramData_f16;\n\
    vxc_short8 tmp_dst;\n\
    _viv_asm(CONV, paramData_f16, paramData);\n\
\n\
    int4 coord_out = (int4)(get_global_id(0), get_global_id(1), 2, 0);\n\
    //R\n\
    VXC_DP2x8(dst, src0, paramData_f16, VXC_MODIFIER(0, 4, 0, VXC_RM_TowardZero, 0), uniExtractR_2x8);\n\
    VXC_DP2x8(dst, src1, paramData_f16, VXC_MODIFIER(5, 7, 0, VXC_RM_TowardZero, 0), uniExtractR_2x8);\n\
    _viv_asm(COPY, tmp_dst, dst, 16);\n\
    VXC_WriteImage2DArray(output, coord_out, tmp_dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
\n\
    //G\n\
    coord_out.z = 1;\n\
    VXC_DP2x8(dst, src0, paramData_f16, VXC_MODIFIER(0, 4, 0, VXC_RM_TowardZero, 0), uniExtractG_2x8);\n\
    VXC_DP2x8(dst, src1, paramData_f16, VXC_MODIFIER(5, 7, 0, VXC_RM_TowardZero, 0), uniExtractG_2x8);\n\
    _viv_asm(COPY, tmp_dst, dst, 16);\n\
    VXC_WriteImage2DArray(output, coord_out, tmp_dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    //B\n\
    coord_out.z = 0;\n\
    VXC_DP2x8(dst, src0, paramData_f16, VXC_MODIFIER(0, 4, 0, VXC_RM_TowardZero, 0), uniExtractB_2x8);\n\
    VXC_DP2x8(dst, src1, paramData_f16, VXC_MODIFIER(5, 7, 0, VXC_RM_TowardZero, 0), uniExtractB_2x8);\n\
    _viv_asm(COPY, tmp_dst, dst, 16);\n\
    VXC_WriteImage2DArray(output, coord_out, tmp_dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void ScaletoTensor_Int8_copy\n\
    (\n\
    __read_only image2d_t        input,\n\
    __write_only image2d_array_t output,\n\
         global int              *xRatio,\n\
         global int              *yRatio,\n\
         global int              *xOffset,\n\
         global int              *yOffset,\n\
                float            rMean,\n\
                float            gMean,\n\
                float            bMean,\n\
                float            f32Var\n\
    )\n\
{\n\
    int2 coord      = (int2)(get_global_id(0) * 3, get_global_id(1));\n\
\n\
    coord.xy += (int2) (*xOffset, *yOffset);\n\
    vxc_uchar16 src0, src1;\n\
    vxc_char16   dst;\n\
\n\
    VXC_ReadImage(src0, input, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(src1, input, coord.xy, VXC_5BITOFFSET_XY(15, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    f32Var *= outputScale;\n\
    float4      paramData = (float4)(rMean * f32Var, gMean * f32Var, bMean * f32Var, f32Var);\n\
    //convert U8 to FP16\n\
    half4 paramData_f16;\n\
    _viv_asm(CONV, paramData_f16, paramData);\n\
\n\
    int4 coord_out = (int4)(get_global_id(0), get_global_id(1), 2, 0);\n\
    //R\n\
    VXC_DP2x8(dst, src0, paramData_f16, VXC_MODIFIER(0, 4, 0, VXC_RM_ToNearestEven, 1), uniExtractR_2x8);\n\
    VXC_DP2x8(dst, src1, paramData_f16, VXC_MODIFIER(5, 9, 0, VXC_RM_ToNearestEven, 1), uniExtractR_2x8);\n\
    VXC_WriteImage2DArray(output, coord_out, dst, VXC_MODIFIER(0, 9, 0, VXC_RM_TowardZero, 0));\n\
\n\
\n\
    //G\n\
    coord_out.z = 1;\n\
    VXC_DP2x8(dst, src0, paramData_f16, VXC_MODIFIER(0, 4, 0, VXC_RM_ToNearestEven, 1), uniExtractG_2x8);\n\
    VXC_DP2x8(dst, src1, paramData_f16, VXC_MODIFIER(5, 9, 0, VXC_RM_ToNearestEven, 1), uniExtractG_2x8);\n\
    VXC_WriteImage2DArray(output, coord_out, dst, VXC_MODIFIER(0, 9, 0, VXC_RM_TowardZero, 0));\n\
\n\
    //B\n\
    coord_out.z = 0;\n\
    VXC_DP2x8(dst, src0, paramData_f16, VXC_MODIFIER(0, 4, 0, VXC_RM_ToNearestEven, 1), uniExtractB_2x8);\n\
    VXC_DP2x8(dst, src1, paramData_f16, VXC_MODIFIER(5, 9, 0, VXC_RM_ToNearestEven, 1), uniExtractB_2x8);\n\
    VXC_WriteImage2DArray(output, coord_out, dst, VXC_MODIFIER(0, 9, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void ScaletoTensor_Int16_copy\n\
    (\n\
    __read_only image2d_t        input,\n\
    __write_only image2d_array_t output,\n\
         global int              *xRatio,\n\
         global int              *yRatio,\n\
         global int              *xOffset,\n\
         global int              *yOffset,\n\
                float            rMean,\n\
                float            gMean,\n\
                float            bMean,\n\
                float            f32Var\n\
    )\n\
{\n\
    int2 coord      = (int2)(get_global_id(0) * 3, get_global_id(1));\n\
\n\
    coord.xy += (int2) (*xOffset, *yOffset);\n\
    vxc_uchar16 src0, src1;\n\
    vxc_short8   dst;\n\
\n\
    VXC_ReadImage(src0, input, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(src1, input, coord.xy, VXC_5BITOFFSET_XY(15, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    f32Var *= outputScale;\n\
    float4      paramData = (float4)(rMean * f32Var, gMean * f32Var, bMean * f32Var, f32Var);\n\
    //convert U8 to FP16\n\
    half4 paramData_f16;\n\
    _viv_asm(CONV, paramData_f16, paramData);\n\
\n\
    int4 coord_out = (int4)(get_global_id(0), get_global_id(1), 2, 0);\n\
    //R\n\
    VXC_DP2x8(dst, src0, paramData_f16, VXC_MODIFIER(0, 4, 0, VXC_RM_ToNearestEven, 1), uniExtractR_2x8);\n\
    VXC_DP2x8(dst, src1, paramData_f16, VXC_MODIFIER(5, 7, 0, VXC_RM_ToNearestEven, 1), uniExtractR_2x8);\n\
    VXC_WriteImage2DArray(output, coord_out, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
\n\
    //G\n\
    coord_out.z = 1;\n\
    VXC_DP2x8(dst, src0, paramData_f16, VXC_MODIFIER(0, 4, 0, VXC_RM_ToNearestEven, 1), uniExtractG_2x8);\n\
    VXC_DP2x8(dst, src1, paramData_f16, VXC_MODIFIER(5, 7, 0, VXC_RM_ToNearestEven, 1), uniExtractG_2x8);\n\
    VXC_WriteImage2DArray(output, coord_out, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    //B\n\
    coord_out.z = 0;\n\
    VXC_DP2x8(dst, src0, paramData_f16, VXC_MODIFIER(0, 4, 0, VXC_RM_ToNearestEven, 1), uniExtractB_2x8);\n\
    VXC_DP2x8(dst, src1, paramData_f16, VXC_MODIFIER(5, 7, 0, VXC_RM_ToNearestEven, 1), uniExtractB_2x8);\n\
    VXC_WriteImage2DArray(output, coord_out, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform float outputZP;\n\
__kernel void ScaletoTensor_UInt8_copy\n\
    (\n\
    __read_only image2d_t        input,\n\
    __write_only image2d_array_t output,\n\
         global int              *xRatio,\n\
         global int              *yRatio,\n\
         global int              *xOffset,\n\
         global int              *yOffset,\n\
                float            rMean,\n\
                float            gMean,\n\
                float            bMean,\n\
                float            f32Var\n\
    )\n\
{\n\
    int2 coord      = (int2)(get_global_id(0) * 3, get_global_id(1));\n\
\n\
    coord.xy += (int2) (*xOffset, *yOffset);\n\
    vxc_uchar16 src0, src1;\n\
    vxc_uchar16   dst;\n\
\n\
    VXC_ReadImage(src0, input, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(src1, input, coord.xy, VXC_5BITOFFSET_XY(15, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    f32Var *= outputScale;\n\
    float4      paramData = (float4)(rMean * f32Var - outputZP,\\\n\
        gMean * f32Var - outputZP, bMean * f32Var - outputZP, f32Var);\n\
    //convert U8 to FP16\n\
    half4 paramData_f16;\n\
    _viv_asm(CONV, paramData_f16, paramData);\n\
\n\
    int4 coord_out = (int4)(get_global_id(0), get_global_id(1), 2, 0);\n\
    //R\n\
    VXC_DP2x8(dst, src0, paramData_f16, VXC_MODIFIER(0, 4, 0, VXC_RM_ToNearestEven, 1), uniExtractR_2x8);\n\
    VXC_DP2x8(dst, src1, paramData_f16, VXC_MODIFIER(5, 9, 0, VXC_RM_ToNearestEven, 1), uniExtractR_2x8);\n\
    VXC_WriteImage2DArray(output, coord_out, dst, VXC_MODIFIER(0, 9, 0, VXC_RM_TowardZero, 0));\n\
\n\
\n\
    //G\n\
    coord_out.z = 1;\n\
    VXC_DP2x8(dst, src0, paramData_f16, VXC_MODIFIER(0, 4, 0, VXC_RM_ToNearestEven, 1), uniExtractG_2x8);\n\
    VXC_DP2x8(dst, src1, paramData_f16, VXC_MODIFIER(5, 9, 0, VXC_RM_ToNearestEven, 1), uniExtractG_2x8);\n\
    VXC_WriteImage2DArray(output, coord_out, dst, VXC_MODIFIER(0, 9, 0, VXC_RM_TowardZero, 0));\n\
\n\
    //B\n\
    coord_out.z = 0;\n\
    VXC_DP2x8(dst, src0, paramData_f16, VXC_MODIFIER(0, 4, 0, VXC_RM_ToNearestEven, 1), uniExtractB_2x8);\n\
    VXC_DP2x8(dst, src1, paramData_f16, VXC_MODIFIER(5, 9, 0, VXC_RM_ToNearestEven, 1), uniExtractB_2x8);\n\
    VXC_WriteImage2DArray(output, coord_out, dst, VXC_MODIFIER(0, 9, 0, VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_imageprocess_3_vx*/

static const char vsi_nn_kernel_imageprocess_4_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform VXC_512Bits uniVecShift10;\n\
_viv_uniform VXC_512Bits uniAddRShift;\n\
_viv_uniform VXC_512Bits uniGetTempVal;\n\
_viv_uniform VXC_512Bits uniExtractBytes;\n\
\n\
_viv_uniform VXC_512Bits uniConvertIntergetoF32_4x4;\n\
_viv_uniform float outputScale;\n\
_viv_uniform VXC_512Bits uniExtactInteger_2x8;\n\
\n\
#define DESCALE(x) (((x) + (1<<19)) >> 20)\n\
__kernel void GrayScaletoTensor_Int8\n\
    (\n\
    __read_only image2d_t        input,\n\
    __write_only image2d_array_t output,\n\
        global int               *xRatio,\n\
        global int               *yRatio,\n\
        global int               *xOffset,\n\
        global int               *yOffset,\n\
               float             mean,\n\
               float             f32Var\n\
    )\n\
{\n\
    int2 ratioXY = (int2)(*xRatio, *yRatio);\n\
\n\
    int4 xPos       = get_global_id(0);\n\
    int yPos        = get_global_id(1);\n\
\n\
    int2 ratioSufXY = (ratioXY >> 1) - (1 << 14);\n\
    xPos += (int4)(0, 1, 2, 3);\n\
\n\
    //x\n\
    int4 fx0 = xPos * ratioXY.x + ratioSufXY.x;\n\
    int4 sx = fx0 & 0xffff8000;\n\
    fx0 -= sx;\n\
    sx = sx >> 15;\n\
\n\
    vxc_short4 fx;\n\
    VXC_DP4x4(fx, fx0, 1 << 4, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniAddRShift);\n\
    //y\n\
    int fy = yPos * ratioXY.y + ratioSufXY.y;\n\
    int sy = fy & 0xffff8000; // Floor\n\
\n\
    fy -= sy;\n\
    sy = sy >> 15;\n\
\n\
    fy = (fy + (1<< 4)) >> 5;\n\
\n\
    //R\n\
    vxc_uchar16 line0Y;\n\
    vxc_uchar16 line1Y;\n\
    int4 coord;\n\
    sx = sx + *xOffset;\n\
    coord.xyz    = sx.xyz;\n\
    coord.w        = sy + *yOffset;\n\
    int2 coord1 = (int2)(sx.w, coord.w);\n\
    VXC_ReadImage(line0Y, input, coord.xw, VXC_5BITOFFSET_XY(0, 0),\n\
        VXC_MODIFIER(0, 1, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line0Y, input, coord.yw, VXC_5BITOFFSET_XY(0, 0),\n\
        VXC_MODIFIER(2, 3, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line0Y, input, coord.zw, VXC_5BITOFFSET_XY(0, 0),\n\
        VXC_MODIFIER(4, 5, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line0Y, input, coord1, VXC_5BITOFFSET_XY(0, 0),\n\
        VXC_MODIFIER(6, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_ReadImage(line1Y, input, coord.xw, VXC_5BITOFFSET_XY(0, 1),\n\
        VXC_MODIFIER(0, 1, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line1Y, input, coord.yw, VXC_5BITOFFSET_XY(0, 1),\n\
        VXC_MODIFIER(2, 3, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line1Y, input, coord.zw, VXC_5BITOFFSET_XY(0, 1),\n\
        VXC_MODIFIER(4, 5, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line1Y, input, coord1, VXC_5BITOFFSET_XY(0, 1),\n\
        VXC_MODIFIER(6, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    float grayMean = mean * f32Var;\n\
\n\
    int4 test01, temp1;\n\
    int4 test02, temp2;\n\
    int4 tt;\n\
    vxc_uchar4 val;\n\
    int2 coord_out = (int2)(xPos.x, yPos);\n\
\n\
    vxc_uchar8 line1, line2;\n\
\n\
    VXC_DP4x4(test01, line0Y, line0Y, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10);\n\
    VXC_DP4x4(temp1, line0Y, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal);\n\
    temp1 = temp1 + test01;\n\
\n\
    VXC_DP4x4(test02, line1Y, line1Y, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10);\n\
    VXC_DP4x4(temp2, line1Y, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal);\n\
    temp2 = temp2 + test02;\n\
    temp2 = fy * (temp2 - temp1) + (temp1 << 10);\n\
\n\
    vxc_float4 tmp_dst;\n\
    vxc_uchar4 u8_dst;\n\
    VXC_DP4x4(u8_dst, temp2, 1 << 19, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), uniExtractBytes);\n\
    VXC_DP4x4(tmp_dst, u8_dst, u8_dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero,\n\
        1), uniConvertIntergetoF32_4x4);\n\
\n\
    //convert U8 to dfp8\n\
    int4 dst0;\n\
    vxc_char4 dst;\n\
    tmp_dst = tmp_dst * f32Var - grayMean;\n\
    tmp_dst *= outputScale;\n\
    dst0 = convert_int4_rte(tmp_dst);\n\
    VXC_DP2x8(dst, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniExtactInteger_2x8);\n\
\n\
    VXC_WriteImage(output, coord_out, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform VXC_512Bits uniDataMulAlpha_4x4;\n\
_viv_uniform VXC_512Bits uniDataSubMean_4x4;\n\
__kernel void GrayScaletoTensor_Fp16\n\
    (\n\
    __read_only image2d_t        input,\n\
    __write_only image2d_array_t output,\n\
         global int              *xRatio,\n\
         global int              *yRatio,\n\
         global int              *xOffset,\n\
         global int              *yOffset,\n\
                float            mean,\n\
                float            f32Var\n\
    )\n\
{\n\
    int2 ratioXY = (int2)(*xRatio, *yRatio);\n\
\n\
    int4 xPos       = get_global_id(0);\n\
    int yPos        = get_global_id(1);\n\
\n\
    int2 ratioSufXY = (ratioXY >> 1) - (1 << 14);\n\
    xPos += (int4)(0, 1, 2, 3);\n\
\n\
    //x\n\
    int4 fx0 = xPos * ratioXY.x + ratioSufXY.x;\n\
    int4 sx = fx0 & 0xffff8000;\n\
    fx0 -= sx;\n\
    sx = sx >> 15;\n\
\n\
    vxc_short4 fx;\n\
    VXC_DP4x4(fx, fx0, 1 << 4, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniAddRShift);\n\
    //y\n\
    int fy = yPos * ratioXY.y + ratioSufXY.y;\n\
    int sy = fy & 0xffff8000; // Floor\n\
\n\
    fy -= sy;\n\
    sy = sy >> 15;\n\
\n\
    fy = (fy + (1<< 4)) >> 5;\n\
\n\
    //R\n\
    vxc_uchar16 line0Y;\n\
    vxc_uchar16 line1Y;\n\
    int4 coord;\n\
    sx = sx + *xOffset;\n\
    coord.xyz    = sx.xyz;\n\
    coord.w        = sy + *yOffset;\n\
    int2 coord1 = (int2)(sx.w, coord.w);\n\
    VXC_ReadImage(line0Y, input, coord.xw, VXC_5BITOFFSET_XY(0, 0),\n\
        VXC_MODIFIER(0, 1, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line0Y, input, coord.yw, VXC_5BITOFFSET_XY(0, 0),\n\
        VXC_MODIFIER(2, 3, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line0Y, input, coord.zw, VXC_5BITOFFSET_XY(0, 0),\n\
        VXC_MODIFIER(4, 5, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line0Y, input, coord1, VXC_5BITOFFSET_XY(0, 0),\n\
        VXC_MODIFIER(6, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_ReadImage(line1Y, input, coord.xw, VXC_5BITOFFSET_XY(0, 1),\n\
        VXC_MODIFIER(0, 1, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line1Y, input, coord.yw, VXC_5BITOFFSET_XY(0, 1),\n\
        VXC_MODIFIER(2, 3, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line1Y, input, coord.zw, VXC_5BITOFFSET_XY(0, 1),\n\
        VXC_MODIFIER(4, 5, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line1Y, input, coord1, VXC_5BITOFFSET_XY(0, 1),\n\
        VXC_MODIFIER(6, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    float grayMean = mean;\n\
\n\
    int4 test01, temp1;\n\
    int4 test02, temp2;\n\
    int4 tt;\n\
    vxc_uchar4 val;\n\
    int2 coord_out = (int2)(xPos.x, yPos);\n\
\n\
    VXC_DP4x4(test01, line0Y, line0Y, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10);\n\
    VXC_DP4x4(temp1, line0Y, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal);\n\
    temp1 = temp1 + test01;\n\
\n\
    VXC_DP4x4(test02, line1Y, line1Y, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10);\n\
    VXC_DP4x4(temp2, line1Y, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal);\n\
    temp2 = temp2 + test02;\n\
    temp2 = fy * (temp2 - temp1) + (temp1 << 10);\n\
\n\
    VXC_DP4x4(val, temp2, 1 << 19, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), uniExtractBytes);\n\
\n\
    //convert U8 to FP16\n\
    half f16mean;\n\
    half f16alpha;\n\
    vxc_half4    dst;\n\
    vxc_short4 tmp_dst;\n\
    _viv_asm(CONV, f16mean, grayMean);\n\
    _viv_asm(CONV, f16alpha, f32Var);\n\
    VXC_DP4x4(dst, val, f16mean, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniDataSubMean_4x4);\n\
    VXC_DP4x4(dst, dst, f16alpha, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniDataMulAlpha_4x4);\n\
    _viv_asm(COPY, tmp_dst, dst, 8);\n\
    VXC_WriteImage(output, coord_out, tmp_dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_imageprocess_4_vx*/

static const char vsi_nn_kernel_imageprocess_5_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform VXC_512Bits uniVecShift10;\n\
_viv_uniform VXC_512Bits uniAddRShift;\n\
_viv_uniform VXC_512Bits uniGetTempVal;\n\
_viv_uniform VXC_512Bits uniExtractBytes;\n\
\n\
_viv_uniform VXC_512Bits uniConvertIntergetoF32_4x4;\n\
_viv_uniform float outputScale;\n\
_viv_uniform VXC_512Bits uniExtactInteger_2x8;\n\
\n\
_viv_uniform VXC_512Bits uniDataMeanStddevLo_2x8;\n\
_viv_uniform VXC_512Bits uniDataMeanStddevHi_2x8;\n\
__kernel void GrayScaletoTensor_Fp16_copy\n\
    (\n\
    __read_only image2d_t        input,\n\
    __write_only image2d_array_t output,\n\
         global int              *xRatio,\n\
         global int              *yRatio,\n\
         global int              *xOffset,\n\
         global int              *yOffset,\n\
                float            mean,\n\
                float            f32Var\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), get_global_id(1));\n\
\n\
    coord.xy += (int2) (*xOffset, *yOffset);\n\
    vxc_uchar16 src0;\n\
    vxc_half8   dst0, dst1;\n\
\n\
    VXC_ReadImage(src0, input, coord.xy, VXC_5BITOFFSET_XY(0, 0),\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    coord.x = coord.z + 8;\n\
    float4 paramData = (float4)(mean * f32Var, mean * f32Var, mean * f32Var, f32Var);\n\
    //convert U8 to FP16\n\
    half4 paramData_f16;\n\
    vxc_short8 tmp_dst;\n\
    _viv_asm(CONV, paramData_f16, paramData);\n\
\n\
    VXC_DP2x8(dst0, src0, paramData_f16,\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniDataMeanStddevLo_2x8);\n\
    VXC_DP2x8(dst1, src0, paramData_f16,\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniDataMeanStddevHi_2x8);\n\
    _viv_asm(COPY, tmp_dst, dst0, 16);\n\
    VXC_WriteImage(output, coord.zw, tmp_dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, tmp_dst, dst1, 16);\n\
    VXC_WriteImage(output, coord.xw, tmp_dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void GrayScaletoTensor_Int8_copy\n\
    (\n\
    __read_only image2d_t        input,\n\
    __write_only image2d_array_t output,\n\
         global int              *xRatio,\n\
         global int              *yRatio,\n\
         global int              *xOffset,\n\
         global int              *yOffset,\n\
                float            mean,\n\
                float            f32Var\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), get_global_id(1));\n\
\n\
    coord.xy += (int2) (*xOffset, *yOffset);\n\
    vxc_uchar16 src0;\n\
    vxc_char16   dst;\n\
\n\
    VXC_ReadImage(src0, input, coord.xy, VXC_5BITOFFSET_XY(0, 0),\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    f32Var *= outputScale;\n\
    float4 paramData = (float4)(mean * f32Var, mean * f32Var, mean * f32Var, f32Var);\n\
    //convert U8 to FP16\n\
    half4 paramData_f16;\n\
    _viv_asm(CONV, paramData_f16, paramData);\n\
\n\
    VXC_DP2x8(dst, src0, paramData_f16,\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniDataMeanStddevLo_2x8);\n\
    VXC_DP2x8(dst, src0, paramData_f16,\n\
        VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniDataMeanStddevHi_2x8);\n\
    VXC_WriteImage(output, coord.zw, dst,\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
}\n\
\n\
__kernel void GrayScaletoTensor_Int16\n\
    (\n\
    __read_only image2d_t        input,\n\
    __write_only image2d_array_t output,\n\
        global int               *xRatio,\n\
        global int               *yRatio,\n\
        global int               *xOffset,\n\
        global int               *yOffset,\n\
               float             mean,\n\
               float             f32Var\n\
    )\n\
{\n\
    int2 ratioXY = (int2)(*xRatio, *yRatio);\n\
\n\
    int4 xPos        = get_global_id(0);\n\
    int yPos        = get_global_id(1);\n\
\n\
    int2 ratioSufXY = (ratioXY >> 1) - (1 << 14);\n\
    xPos += (int4)(0, 1, 2, 3);\n\
\n\
    //x\n\
    int4 fx0 = xPos * ratioXY.x + ratioSufXY.x;\n\
    int4 sx = fx0 & 0xffff8000;\n\
    fx0 -= sx;\n\
    sx = sx >> 15;\n\
\n\
    vxc_short4 fx;\n\
    VXC_DP4x4(fx, fx0, 1 << 4, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniAddRShift);\n\
    //y\n\
    int fy = yPos * ratioXY.y + ratioSufXY.y;\n\
    int sy = fy & 0xffff8000; // Floor\n\
\n\
    fy -= sy;\n\
    sy = sy >> 15;\n\
\n\
    fy = (fy + (1<< 4)) >> 5;\n\
\n\
    vxc_uchar16 line0Y;\n\
    vxc_uchar16 line1Y;\n\
    int4 coord;\n\
    sx = sx + *xOffset;\n\
    coord.xyz    = sx.xyz;\n\
    coord.w        = sy + *yOffset;\n\
    int2 coord1 = (int2)(sx.w, coord.w);\n\
    VXC_ReadImage(line0Y, input, coord.xw,\n\
        VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 1, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line0Y, input, coord.yw,\n\
        VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(2, 3, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line0Y, input, coord.zw,\n\
        VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(4, 5, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line0Y, input, coord1, VXC_5BITOFFSET_XY(0, 0),\n\
        VXC_MODIFIER(6, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_ReadImage(line1Y, input, coord.xw, VXC_5BITOFFSET_XY(0, 1),\n\
        VXC_MODIFIER(0, 1, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line1Y, input, coord.yw, VXC_5BITOFFSET_XY(0, 1),\n\
        VXC_MODIFIER(2, 3, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line1Y, input, coord.zw, VXC_5BITOFFSET_XY(0, 1),\n\
        VXC_MODIFIER(4, 5, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line1Y, input, coord1, VXC_5BITOFFSET_XY(0, 1),\n\
        VXC_MODIFIER(6, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    float grayMean = mean * f32Var;\n\
\n\
    int4 test01, temp1;\n\
    int4 test02, temp2;\n\
    int4 tt;\n\
    vxc_uchar4 val;\n\
    int2 coord_out = (int2)(xPos.x, yPos);\n\
\n\
    vxc_uchar8 line1, line2;\n\
\n\
    VXC_DP4x4(test01, line0Y, line0Y, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10);\n\
    VXC_DP4x4(temp1, line0Y, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal);\n\
    temp1 = temp1 + test01;\n\
\n\
    VXC_DP4x4(test02, line1Y, line1Y, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10);\n\
    VXC_DP4x4(temp2, line1Y, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal);\n\
    temp2 = temp2 + test02;\n\
    temp2 = fy * (temp2 - temp1) + (temp1 << 10);\n\
\n\
    vxc_float4    tmp_dst;\n\
    vxc_uchar4 u8_dst;\n\
    VXC_DP4x4(u8_dst, temp2, 1 << 19, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), uniExtractBytes);\n\
    VXC_DP4x4(tmp_dst, u8_dst, u8_dst,\n\
        VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), uniConvertIntergetoF32_4x4);\n\
\n\
    //convert U8 to dfp8\n\
    int4 dst0;\n\
    vxc_short4 dst;\n\
    tmp_dst = tmp_dst * f32Var - grayMean;\n\
    tmp_dst *= outputScale;\n\
    dst0 = convert_int4_rte(tmp_dst);\n\
    VXC_DP2x8(dst, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniExtactInteger_2x8);\n\
\n\
    VXC_WriteImage(output, coord_out, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void GrayScaletoTensor_Int16_copy\n\
    (\n\
    __read_only image2d_t        input,\n\
    __write_only image2d_array_t output,\n\
         global int              *xRatio,\n\
         global int              *yRatio,\n\
         global int              *xOffset,\n\
         global int              *yOffset,\n\
                float            mean,\n\
                float            f32Var\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), get_global_id(1));\n\
\n\
    coord.xy += (int2) (*xOffset, *yOffset);\n\
    vxc_uchar16 src0;\n\
    vxc_short8  dst0, dst1;\n\
\n\
    VXC_ReadImage(src0, input, coord.xy, VXC_5BITOFFSET_XY(0, 0),\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    coord.x = coord.z + 8;\n\
\n\
    f32Var *= outputScale;\n\
    float4      paramData = (float4)(mean * f32Var, mean * f32Var, mean * f32Var, f32Var);\n\
    //convert U8 to FP16\n\
    half4 paramData_f16;\n\
    _viv_asm(CONV, paramData_f16, paramData);\n\
\n\
\n\
    VXC_DP2x8(dst0, src0, paramData_f16,\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniDataMeanStddevLo_2x8);\n\
    VXC_DP2x8(dst1, src0, paramData_f16,\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniDataMeanStddevHi_2x8);\n\
    VXC_WriteImage(output, coord.zw, dst0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_WriteImage(output, coord.xw, dst1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform float outputZP;\n\
__kernel void GrayScaletoTensor_UInt8_copy\n\
    (\n\
    __read_only image2d_t        input,\n\
    __write_only image2d_array_t output,\n\
         global int              *xRatio,\n\
         global int              *yRatio,\n\
         global int              *xOffset,\n\
         global int              *yOffset,\n\
                float            mean,\n\
                float            f32Var\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), get_global_id(1));\n\
\n\
    coord.xy += (int2) (*xOffset, *yOffset);\n\
    vxc_uchar16 src0;\n\
    vxc_uchar16 dst;\n\
\n\
    VXC_ReadImage(src0, input, coord.xy, VXC_5BITOFFSET_XY(0, 0),\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    f32Var *= outputScale;\n\
    float4  paramData = (float4)(mean * f32Var - outputZP, mean * f32Var - outputZP,\n\
        mean * f32Var - outputZP, f32Var);\n\
    //convert U8 to FP16\n\
    half4 paramData_f16;\n\
    _viv_asm(CONV, paramData_f16, paramData);\n\
\n\
    VXC_DP2x8(dst, src0, paramData_f16,\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniDataMeanStddevLo_2x8);\n\
    VXC_DP2x8(dst, src0, paramData_f16,\n\
        VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniDataMeanStddevHi_2x8);\n\
    VXC_WriteImage(output, coord.zw, dst, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void GrayScaletoTensor_UInt8\n\
    (\n\
    __read_only image2d_t        input,\n\
    __write_only image2d_array_t output,\n\
        global int               *xRatio,\n\
        global int               *yRatio,\n\
        global int               *xOffset,\n\
        global int               *yOffset,\n\
               float             mean,\n\
               float             f32Var\n\
    )\n\
{\n\
    int2 ratioXY = (int2)(*xRatio, *yRatio);\n\
\n\
    int4 xPos        = get_global_id(0);\n\
    int yPos        = get_global_id(1);\n\
\n\
    int2 ratioSufXY = (ratioXY >> 1) - (1 << 14);\n\
    xPos += (int4)(0, 1, 2, 3);\n\
\n\
    //x\n\
    int4 fx0 = xPos * ratioXY.x + ratioSufXY.x;\n\
    int4 sx = fx0 & 0xffff8000;\n\
    fx0 -= sx;\n\
    sx = sx >> 15;\n\
\n\
    vxc_short4 fx;\n\
    VXC_DP4x4(fx, fx0, 1 << 4, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniAddRShift);\n\
    //y\n\
    int fy = yPos * ratioXY.y + ratioSufXY.y;\n\
    int sy = fy & 0xffff8000; // Floor\n\
\n\
    fy -= sy;\n\
    sy = sy >> 15;\n\
\n\
    fy = (fy + (1<< 4)) >> 5;\n\
\n\
    //R\n\
    vxc_uchar16 line0Y;\n\
    vxc_uchar16 line1Y;\n\
    int4 coord;\n\
    sx = sx + *xOffset;\n\
    coord.xyz    = sx.xyz;\n\
    coord.w        = sy + *yOffset;\n\
    int2 coord1 = (int2)(sx.w, coord.w);\n\
    VXC_ReadImage(line0Y, input, coord.xw, VXC_5BITOFFSET_XY(0, 0),\n\
        VXC_MODIFIER(0, 1, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line0Y, input, coord.yw, VXC_5BITOFFSET_XY(0, 0),\n\
        VXC_MODIFIER(2, 3, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line0Y, input, coord.zw, VXC_5BITOFFSET_XY(0, 0),\n\
        VXC_MODIFIER(4, 5, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line0Y, input, coord1, VXC_5BITOFFSET_XY(0, 0),\n\
        VXC_MODIFIER(6, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_ReadImage(line1Y, input, coord.xw, VXC_5BITOFFSET_XY(0, 1),\n\
        VXC_MODIFIER(0, 1, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line1Y, input, coord.yw, VXC_5BITOFFSET_XY(0, 1),\n\
        VXC_MODIFIER(2, 3, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line1Y, input, coord.zw, VXC_5BITOFFSET_XY(0, 1),\n\
        VXC_MODIFIER(4, 5, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line1Y, input, coord1, VXC_5BITOFFSET_XY(0, 1),\n\
        VXC_MODIFIER(6, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    float grayMean = mean * f32Var;\n\
\n\
    int4 test01, temp1;\n\
    int4 test02, temp2;\n\
    int4 tt;\n\
    vxc_uchar4 val;\n\
    int2 coord_out = (int2)(xPos.x, yPos);\n\
\n\
    VXC_DP4x4(test01, line0Y, line0Y, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10);\n\
    VXC_DP4x4(temp1, line0Y, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal);\n\
    temp1 = temp1 + test01;\n\
\n\
    VXC_DP4x4(test02, line1Y, line1Y, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10);\n\
    VXC_DP4x4(temp2, line1Y, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal);\n\
    temp2 = temp2 + test02;\n\
    temp2 = fy * (temp2 - temp1) + (temp1 << 10);\n\
\n\
    vxc_float4 tmp_dst;\n\
    vxc_uchar4 u8_dst;\n\
    VXC_DP4x4(u8_dst, temp2, 1 << 19, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), uniExtractBytes);\n\
    VXC_DP4x4(tmp_dst, u8_dst, u8_dst,\n\
        VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), uniConvertIntergetoF32_4x4);\n\
\n\
    //convert U8 to dfp8\n\
    int4 dst0;\n\
    vxc_uchar4 dst;\n\
    tmp_dst = tmp_dst * f32Var - grayMean;\n\
    tmp_dst = tmp_dst * outputScale + outputZP;\n\
    dst0 = convert_int4_rte(tmp_dst);\n\
    VXC_DP2x8(dst, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniExtactInteger_2x8);\n\
\n\
    VXC_WriteImage(output, coord_out, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_imageprocess_5_vx*/

static const char vsi_nn_kernel_instancenormalize_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
/********************************instancenorm float16*********************************/\n\
_viv_uniform int width;\n\
_viv_uniform int height;\n\
_viv_uniform float dimRatio;\n\
_viv_uniform VXC_512Bits uniFp16SumSqr_dp8x2;\n\
_viv_uniform VXC_512Bits UniFP16toFP32Lo4_dp4x4;\n\
_viv_uniform VXC_512Bits uniExtractHalf4_dp4x4;\n\
_viv_uniform VXC_512Bits uniConvertInt32toUint8_2x8;\n\
_viv_uniform VXC_512Bits uniConvertEndInt16Fp32_4x4;\n\
\n\
__kernel __attribute__((reqd_work_group_size(16, 1, 1))) void vxcInstanceNorm(\n\
    image2d_array_t input,\n\
    image2d_array_t bias,\n\
    image2d_array_t scale,\n\
    image2d_array_t output,\n\
    image2d_array_t meanVari,\n\
              float eps)\n\
{\n\
    int4 coord = (int4)(0, 0, get_global_id(2), 0);\n\
    vxc_short8 src0, src1, src2;\n\
    float sum = 0, sqr = 0, scale_vari, bias_val;\n\
    vxc_half8 in_h, scale_h;\n\
    vxc_float4 bias_f, scale_f;\n\
\n\
    for(coord.y = 0; coord.y < height; coord.y++)\n\
    {\n\
        for(coord.x = 0; coord.x < width; coord.x += 8)\n\
        {\n\
            VXC_ReadImage2DArray(src0, input, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
            _viv_asm(COPY, in_h, src0, 16);\n\
            vxc_float4 sumsqr;\n\
            VXC_DP8x2(sumsqr, in_h, in_h, VXC_MODIFIER(0, 1, 0, VXC_RM_TowardZero, 0),\\\n\
                uniFp16SumSqr_dp8x2);\n\
            sum += sumsqr.x;\n\
            sqr += sumsqr.y;\n\
        }\n\
    }\n\
    coord.w = 0;\n\
    VXC_ReadImage(src1, scale, coord.zw, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
    bias_f = read_imagef(bias, coord.zwww);\n\
    _viv_asm(COPY, scale_h, src1, 16);\n\
    VXC_DP4x4(scale_f, scale_h, scale_h, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        UniFP16toFP32Lo4_dp4x4);\n\
    bias_val = bias_f.s0;\n\
\n\
    float mean, vari;\n\
    mean = sum * dimRatio;\n\
    vari = sqr*dimRatio - mean*mean;\n\
    vari += eps;\n\
    vari = rsqrt(vari);\n\
    scale_vari = scale_f.s0 * vari;\n\
\n\
    for(coord.y = 0; coord.y < height; coord.y++)\n\
    {\n\
        for(coord.x = 0; coord.x < width; coord.x += 8)\n\
        {\n\
            VXC_ReadImage2DArray(src0, input, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
            _viv_asm(COPY, in_h, src0, 16);\n\
\n\
            vxc_float4 in_f0, in_f1;\n\
            VXC_DP4x4(in_f0, in_h, in_h, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
                UniFP16toFP32Lo4_dp4x4);\n\
            VXC_DP4x4(in_f1, in_h, in_h, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
                uniConvertEndInt16Fp32_4x4);\n\
\n\
            vxc_float4 sub, norm;\n\
            half4 norm_h0, norm_h1;\n\
\n\
            sub = in_f0 - mean;\n\
            norm = scale_vari * sub + bias_val;\n\
            _viv_asm(CONV, norm_h0, norm);\n\
\n\
            sub = in_f1 - mean;\n\
            norm = scale_vari * sub + bias_val;\n\
            _viv_asm(CONV, norm_h1, norm);\n\
\n\
            VXC_DP2x8(src2, norm_h0, norm_h1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\n\
                uniConvertInt32toUint8_2x8);\n\
            VXC_WriteImage2DArray(output, coord, src2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        }\n\
    }\n\
}\n\
\n\
_viv_uniform VXC_512Bits uniConvert1stUint8SubZpToFp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvert2ndUint8SubZpToFp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvert3rdUint8SubZpToFp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvert4thUint8SubZpToFp32_4x4;\n\
_viv_uniform VXC_512Bits uniSumU8_16x1;\n\
_viv_uniform VXC_512Bits uniSqrSum_16x1;\n\
_viv_uniform float input_scale;\n\
_viv_uniform int inputZP;\n\
_viv_uniform float scale_scale;\n\
_viv_uniform int scaleZP;\n\
_viv_uniform float outputScale;\n\
_viv_uniform int output_ZP;\n\
_viv_uniform int iter;\n\
_viv_uniform int sumInZp;\n\
_viv_uniform int tmpZp1;\n\
_viv_uniform int tmpZp2;\n\
_viv_uniform float e2InScale;\n\
_viv_uniform float rowSumScale;\n\
_viv_uniform int segCnt;\n\
_viv_uniform int segHeight;\n\
_viv_uniform float sumZpScale;\n\
_viv_uniform float scale_inOut;\n\
\n\
__kernel __attribute__((reqd_work_group_size(16, 1, 1))) void vxcInstanceNorm_U8(\n\
    image2d_array_t input,\n\
    image2d_array_t bias,\n\
    image2d_array_t scale,\n\
    image2d_array_t output,\n\
    image2d_array_t meanVari,\n\
              float eps)\n\
{\n\
    int gidz = get_global_id(1);\n\
    int4 coord = (int4)(get_global_id(0), 0, gidz, 0);\n\
    int4 coord_para = (int4)(gidz, 0, 0, 0);\n\
    vxc_uchar16 src0, src2;\n\
    vxc_short8 src1;\n\
    vxc_half8 scale_h;\n\
    float scale_vari, bias_val;\n\
    vxc_float4 bias_f, scale_f, mean_vari;\n\
\n\
    VXC_ReadImage(src1, scale, coord_para.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, scale_h, src1, 16);\n\
    VXC_DP4x4(scale_f, scale_h, scale_h, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), UniFP16toFP32Lo4_dp4x4);\n\
\n\
    bias_f = read_imagef(bias, coord_para);\n\
    //bias_val = bias_f.s0;\n\
    coord_para.x = gidz << 2;\n\
    mean_vari = read_imagef(meanVari, coord_para);\n\
\n\
    scale_vari = scale_f.s0 * mean_vari.s1;\n\
    short zp = inputZP;\n\
    vxc_int4 tmpVal0, tmpVal1;\n\
    vxc_float4  tmpData0, tmpData1, tmpData2, tmpData3;\n\
    float alpha = scale_inOut * scale_vari;\n\
    bias_val = (bias_f.s0 - scale_vari * mean_vari.s0) * outputScale + output_ZP;\n\
\n\
    for(coord.y = 0; coord.y < height; coord.y++)\n\
    {\n\
    VXC_ReadImage2DArray(src0, input, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_DP4x4(tmpData0, src0, zp, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvert1stUint8SubZpToFp32_4x4);\n\
    VXC_DP4x4(tmpData1, src0, zp, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvert2ndUint8SubZpToFp32_4x4);\n\
    VXC_DP4x4(tmpData2, src0, zp, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvert3rdUint8SubZpToFp32_4x4);\n\
    VXC_DP4x4(tmpData3, src0, zp, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvert4thUint8SubZpToFp32_4x4);\n\
    vxc_float4 norm;\n\
    norm = tmpData0 * alpha + bias_val;\n\
    tmpVal0 = convert_int4_rte(norm);\n\
    norm = tmpData1 * alpha + bias_val;\n\
    tmpVal1 = convert_int4_rte(norm);\n\
    VXC_DP2x8(src2, tmpVal0, tmpVal1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1),\\\n\
        uniConvertInt32toUint8_2x8);\n\
    norm = tmpData2 * alpha + bias_val;\n\
    tmpVal0 = convert_int4_rte(norm);\n\
    norm = tmpData3 * alpha + bias_val;\n\
    tmpVal1 = convert_int4_rte(norm);\n\
    VXC_DP2x8(src2, tmpVal0, tmpVal1, VXC_MODIFIER(8, 15, 0, VXC_RM_TowardZero, 1),\\\n\
        uniConvertInt32toUint8_2x8);\n\
    VXC_WriteImage2DArray(output, coord, src2, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    }\n\
}\n\
\n\
__kernel __attribute__((reqd_work_group_size(16, 1, 1))) void vxcInstanceNormU8_fp16(\n\
    image2d_array_t input,\n\
    image2d_array_t bias,\n\
    image2d_array_t scale,\n\
    image2d_array_t output,\n\
    image2d_array_t meanVari,\n\
              float eps)\n\
{\n\
    int gidz = get_global_id(1);\n\
    int4 coord = (int4)(get_global_id(0), 0, gidz, 0);\n\
    int4 coord_para = (int4)(gidz, 0, 0, 0);\n\
    vxc_uchar16 src0;\n\
    vxc_short8 src1;\n\
    vxc_half8 scale_h;\n\
    float scale_vari, bias_val;\n\
    vxc_float4 bias_f, scale_f, mean_vari;\n\
\n\
    VXC_ReadImage(src1, scale, coord_para.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, scale_h, src1, 16);\n\
    VXC_DP4x4(scale_f, scale_h, scale_h, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        UniFP16toFP32Lo4_dp4x4);\n\
\n\
    bias_f = read_imagef(bias, coord_para);\n\
    //bias_val = bias_f.s0;\n\
\n\
    coord_para.x = gidz << 2;\n\
    mean_vari = read_imagef(meanVari, coord_para);\n\
\n\
    scale_vari = scale_f.s0 * mean_vari.s1;\n\
    short zp = inputZP;\n\
    vxc_float4  tmpData0, tmpData1, tmpData2, tmpData3;\n\
    vxc_short8 outval;\n\
    half4 tmpVal0, tmpVal1;\n\
    float alpha = input_scale * scale_vari;\n\
    bias_val = (bias_f.s0 - scale_vari * mean_vari.s0);\n\
\n\
    for(coord.y = 0; coord.y < height; coord.y++)\n\
    {\n\
    coord_para = coord;\n\
    VXC_ReadImage2DArray(src0, input, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_DP4x4(tmpData0, src0, zp, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvert1stUint8SubZpToFp32_4x4);\n\
    VXC_DP4x4(tmpData1, src0, zp, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvert2ndUint8SubZpToFp32_4x4);\n\
    VXC_DP4x4(tmpData2, src0, zp, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvert3rdUint8SubZpToFp32_4x4);\n\
    VXC_DP4x4(tmpData3, src0, zp, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvert4thUint8SubZpToFp32_4x4);\n\
    vxc_float4 norm;\n\
    norm = alpha * tmpData0 + bias_val;\n\
    _viv_asm(CONV, tmpVal0, norm);\n\
    norm = alpha * tmpData1 + bias_val;\n\
    _viv_asm(CONV, tmpVal1, norm);\n\
    VXC_DP2x8(outval, tmpVal0, tmpVal1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertInt32toUint8_2x8);\n\
    VXC_WriteImage2DArray(output, coord_para, outval, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    coord_para.x += 8;\n\
    norm = alpha * tmpData2 + bias_val;\n\
    _viv_asm(CONV, tmpVal0, norm);\n\
    norm = alpha * tmpData3 + bias_val;\n\
    _viv_asm(CONV, tmpVal1, norm);\n\
    VXC_DP2x8(outval, tmpVal0, tmpVal1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertInt32toUint8_2x8);\n\
    VXC_WriteImage2DArray(output, coord_para, outval, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    }\n\
}\n\
\n\
__kernel __attribute__((reqd_work_group_size(1, 1, 1))) void vxcInstanceNormSum_u8(\n\
    image2d_array_t input,\n\
    image2d_array_t output,\n\
       vx_array_int arraySum,\n\
       vx_array_int arraySqr)\n\
{\n\
    int gidy = get_global_id(1);\n\
    int gidz = get_global_id(2);\n\
    int lidx = get_local_id(0);\n\
    int lidy = get_local_id(1);\n\
    int4 coord = (int4)(get_global_id(0), gidy, gidz, 0);\n\
    vxc_uchar16 src0;\n\
    int tmpSum = 0, tmpSqr = 0;\n\
    int tmpSqr1;\n\
    __local int lcl_sum[1];\n\
    __local int lcl_sqr[1];\n\
    if(lidx == 0 && lidy == 0)\n\
    {\n\
        lcl_sum[0] = 0;\n\
        lcl_sqr[0] = 0;\n\
    }\n\
    barrier(CLK_LOCAL_MEM_FENCE);\n\
\n\
    VXC_ReadImage2DArray(src0, input, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_DP16x1(tmpSum, src0, src0, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0), uniSumU8_16x1);\n\
    VXC_DP16x1(tmpSqr1, src0, src0, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0), uniSqrSum_16x1);\n\
    tmpSqr = (tmpSqr1 + tmpZp1 * tmpSum);\n\
\n\
    int offset = gidz * segCnt + (gidy >> 3);\n\
    __global int* gSum = (__global int*)arraySum.item;\n\
    __global int* gSqr = (__global int*)arraySqr.item;\n\
\n\
    //atom_add(gSum + gidz, tmpSum);\n\
    //atom_add(gSqr, tmpSqr);\n\
    atom_add(lcl_sum, tmpSum);\n\
    atom_add(lcl_sqr, tmpSqr);\n\
    barrier(CLK_LOCAL_MEM_FENCE);\n\
    if(lidx == 0 && lidy == 0)\n\
    {\n\
        gSum[offset] = lcl_sum[0];\n\
        gSqr[offset] = lcl_sqr[0];\n\
    }\n\
}\n\
\n\
__kernel __attribute__((reqd_work_group_size(8, 1, 1))) void vxcInstanceNormSqr_u8(\n\
    image2d_array_t input,\n\
    image2d_array_t output,\n\
              float eps,\n\
       vx_array_int arraySum,\n\
       vx_array_int arraySqr,\n\
     vx_array_float outMean,\n\
     vx_array_float outSqr)\n\
{\n\
    int gidz = get_global_id(0);\n\
    float sum = 0;\n\
\n\
    __global int* gSum = (__global int*)arraySum.item;\n\
    __global int* gSqr = (__global int*)arraySqr.item;\n\
\n\
    __global float* pMean = (__global float*)outMean.item;\n\
    __global float* pSqr = (__global float*)outSqr.item;\n\
\n\
    int offset = gidz * segHeight;\n\
    int tmpSum = gSum[gidz];\n\
    sum = (tmpSum + sumInZp) * input_scale;\n\
    float mean = sum * dimRatio;\n\
    pMean[gidz] = mean;\n\
\n\
    float sqr = 0;\n\
    //sqr = (tmpSqr + tmpZp2) * e2InScale;\n\
    for(int i = 0; i < segHeight; i++)\n\
    {\n\
        int tmpSqr = gSqr[offset + i];\n\
        sqr += (tmpSqr * e2InScale + sumZpScale);\n\
    }\n\
    //pSqr[gidz] = sqr;\n\
    float vari = sqr * dimRatio - mean*mean;\n\
    vari += eps;\n\
    vari = rsqrt(vari);\n\
    pSqr[gidz] = vari;\n\
}\n\
\n\
__kernel __attribute__((reqd_work_group_size(16, 1, 1))) void vxcInstanceNormMeanVari_u8(\n\
    image2d_array_t input,\n\
    image2d_array_t output,\n\
              float eps)\n\
{\n\
    int lidx = get_local_id(0);\n\
    int gidz = get_global_id(1);\n\
    int4 coord = (int4)(0, 0, gidz, 0);\n\
    vxc_uchar16 src0;\n\
    float sum = 0, sqr = 0;\n\
    int tmpSum = 0, tmpSqr = 0;\n\
    int colCnt = 0;\n\
    int tmpSum1, tmpSqr1;\n\
\n\
    __local float lcl_sum[16];\n\
    __local float lcl_sqr[16];\n\
\n\
    for(coord.x = (lidx << 4); coord.x < width; coord.x += 256)\n\
    {\n\
        tmpSqr = 0;\n\
        for(coord.y = 0; coord.y < height; coord.y++)\n\
        {\n\
            VXC_ReadImage2DArray(src0, input, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
            VXC_DP16x1(tmpSum1, src0, src0, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0), uniSumU8_16x1);\n\
            tmpSum += (tmpSum1);\n\
            VXC_DP16x1(tmpSqr1, src0, src0, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0), uniSqrSum_16x1);\n\
            tmpSqr += (tmpSqr1 + tmpZp1 * tmpSum1);\n\
        }\n\
        sqr += (tmpSqr * e2InScale + rowSumScale);\n\
        colCnt++;\n\
    }\n\
    sum = (tmpSum + sumInZp * colCnt) * input_scale;\n\
    //sqr = (tmpSqr + tmpZp2) * e2InScale;\n\
\n\
    lcl_sum[lidx] = sum;\n\
    lcl_sqr[lidx] = sqr;\n\
    barrier(CLK_LOCAL_MEM_FENCE);\n\
\n\
    int4 coord_out = (int4)(gidz << 2, 0, 0, 0);\n\
    if(lidx == 0)\n\
    {\n\
        sum = 0; sqr = 0;\n\
        for(int i = 0; i < 16; i++)\n\
        {\n\
            sum += lcl_sum[i];\n\
            sqr += lcl_sqr[i];\n\
        }\n\
        float mean = sum * dimRatio;\n\
        float vari = sqr * dimRatio - mean*mean;\n\
        vari += eps;\n\
        vari = rsqrt(vari);\n\
        float4 data = (float4)(0, 0, 0, 0);\n\
        data.x = mean;\n\
        data.y = vari;\n\
        write_imagef(output, coord_out, data);\n\
    }\n\
}"; /* end of vsi_nn_kernel_instancenormalize_vx*/

static const char vsi_nn_kernel_instancenormalize_i8_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
/**************************instancenorm int8***************************/\n\
_viv_uniform int width;\n\
_viv_uniform int height;\n\
_viv_uniform float dimRatio;\n\
_viv_uniform VXC_512Bits uniInt16SumSqr_dp8x2;\n\
_viv_uniform VXC_512Bits UniFP16toFP32Lo4_dp4x4;\n\
_viv_uniform VXC_512Bits uniConvertInt32toUint8_2x8;\n\
\n\
_viv_uniform VXC_512Bits uniConvertDirInt8Fp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertEndInt8Fp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertTrdInt8Fp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertFthInt8Fp32_4x4;\n\
_viv_uniform VXC_512Bits uniSumInt8_16x1;\n\
_viv_uniform VXC_512Bits uniSqrSumInt8_16x1;\n\
_viv_uniform float input_fl_scale;\n\
_viv_uniform float output_fl_Scale;\n\
_viv_uniform float inFlScale_s2;\n\
\n\
__kernel __attribute__((reqd_work_group_size(16, 1, 1))) void vxcInstanceNorm_int8(\n\
    image2d_array_t input,\n\
    image2d_array_t bias,\n\
    image2d_array_t scale,\n\
    image2d_array_t output,\n\
    image2d_array_t meanVari,\n\
              float eps)\n\
{\n\
    int4 coord = (int4)(0, 0, get_global_id(2), 0);\n\
    vxc_char16 src0, src2;\n\
    vxc_short8 src1;\n\
    vxc_half8 scale_h;\n\
    float sum = 0, sqr = 0, scale_vari, bias_val;\n\
    vxc_float4 bias_f, scale_f;\n\
    int tmpSum = 0, tmpSqr = 0;\n\
    int tmpSum1;\n\
    int tmpSqr1;\n\
    vxc_float4  tmpData0, tmpData1, tmpData2, tmpData3;\n\
\n\
    for(coord.y = 0; coord.y < height; coord.y++)\n\
    {\n\
        tmpSqr = 0;\n\
        for(coord.x = 0; coord.x < width; coord.x += 16)\n\
        {\n\
            VXC_ReadImage2DArray(src0, input, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
            VXC_DP16x1(tmpSum1, src0, src0,\\\n\
                VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0), uniSumInt8_16x1);\n\
            tmpSum += (tmpSum1);\n\
            VXC_DP16x1(tmpSqr1, src0, src0,\\\n\
                VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0), uniSqrSumInt8_16x1);\n\
            tmpSqr += (tmpSqr1);\n\
        }\n\
        sqr += (tmpSqr * inFlScale_s2);\n\
    }\n\
    sum = tmpSum * input_fl_scale;\n\
\n\
    coord.w = 0;\n\
    VXC_ReadImage(src1, scale, coord.zw, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, scale_h, src1, 16);\n\
    VXC_DP4x4(scale_f, scale_h, scale_h, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        UniFP16toFP32Lo4_dp4x4);\n\
\n\
    bias_f = read_imagef(bias, coord.zwww);\n\
    bias_val = bias_f.s0 * output_fl_Scale; // bias_f.s0\n\
\n\
    float mean, vari;\n\
    mean = sum * dimRatio;\n\
    vari = sqr*dimRatio - mean*mean;\n\
    vari += eps;\n\
    vari = rsqrt(vari);\n\
    scale_vari = scale_f.s0 * vari * output_fl_Scale;\n\
    //scale_f.s0 * vari, output_fl_Scale for i8&i16\n\
\n\
    vxc_int4 tmpVal0, tmpVal1;\n\
\n\
    for(coord.y = 0; coord.y < height; coord.y++)\n\
    {\n\
        for(coord.x = 0; coord.x < width; coord.x += 16)\n\
        {\n\
            VXC_ReadImage2DArray(src0, input, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
            VXC_DP4x4(tmpData0, src0, src0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
                uniConvertDirInt8Fp32_4x4);\n\
            VXC_DP4x4(tmpData1, src0, src0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
                uniConvertEndInt8Fp32_4x4);\n\
            VXC_DP4x4(tmpData2, src0, src0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
                uniConvertTrdInt8Fp32_4x4);\n\
            VXC_DP4x4(tmpData3, src0, src0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
                uniConvertFthInt8Fp32_4x4);\n\
            tmpData0 *= input_fl_scale;\n\
            tmpData1 *= input_fl_scale;\n\
            tmpData2 *= input_fl_scale;\n\
            tmpData3 *= input_fl_scale;\n\
\n\
            vxc_float4 norm;\n\
            tmpData0 -= mean;\n\
            norm = scale_vari * tmpData0 + bias_val;\n\
            tmpVal0 = convert_int4_rte(norm);\n\
\n\
            tmpData1 -= mean;\n\
            norm = scale_vari * tmpData1 + bias_val;\n\
            tmpVal1 = convert_int4_rte(norm);\n\
            VXC_DP2x8(src2, tmpVal0, tmpVal1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\n\
                uniConvertInt32toUint8_2x8);\n\
\n\
            tmpData2 -= mean;\n\
            norm = scale_vari * tmpData2 + bias_val;\n\
            tmpVal0 = convert_int4_rte(norm);\n\
\n\
            tmpData3 -= mean;\n\
            norm = scale_vari * tmpData3 + bias_val;\n\
            tmpVal1 = convert_int4_rte(norm);\n\
            VXC_DP2x8(src2, tmpVal0, tmpVal1, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1),\n\
                uniConvertInt32toUint8_2x8);\n\
            VXC_WriteImage2DArray(output, coord, src2, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
        }\n\
    }\n\
}\n\
\n\
__kernel __attribute__((reqd_work_group_size(16, 1, 1))) void vxcInstanceNormInt8_fp16(\n\
    image2d_array_t input,\n\
    image2d_array_t bias,\n\
    image2d_array_t scale,\n\
    image2d_array_t output,\n\
    image2d_array_t meanVari,\n\
              float eps)\n\
{\n\
    int4 coord = (int4)(0, 0, get_global_id(2), 0);\n\
    vxc_char16 src0;\n\
    vxc_short8 src1;\n\
    vxc_half8 scale_h;\n\
    float sum = 0, sqr = 0, scale_vari, bias_val;\n\
    vxc_float4 bias_f, scale_f;\n\
    int tmpSum = 0, tmpSqr = 0;\n\
    int tmpSum1;\n\
    int tmpSqr1;\n\
    vxc_float4  tmpData0, tmpData1, tmpData2, tmpData3;\n\
\n\
    for(coord.y = 0; coord.y < height; coord.y++)\n\
    {\n\
        tmpSqr = 0;\n\
        for(coord.x = 0; coord.x < width; coord.x += 16)\n\
        {\n\
            VXC_ReadImage2DArray(src0, input, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
            VXC_DP16x1(tmpSum1, src0, src0, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0), uniSumInt8_16x1);\n\
            tmpSum += (tmpSum1);\n\
            VXC_DP16x1(tmpSqr1, src0, src0, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0), uniSqrSumInt8_16x1);\n\
            tmpSqr += (tmpSqr1);\n\
        }\n\
        sqr += (tmpSqr * inFlScale_s2);\n\
    }\n\
    sum = tmpSum * input_fl_scale;\n\
\n\
    coord.w = 0;\n\
    VXC_ReadImage(src1, scale, coord.zw, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, scale_h, src1, 16);\n\
    VXC_DP4x4(scale_f, scale_h, scale_h, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        UniFP16toFP32Lo4_dp4x4);\n\
\n\
    bias_f = read_imagef(bias, coord.zwww);\n\
    bias_val = bias_f.s0;\n\
\n\
    float mean, vari;\n\
    mean = sum * dimRatio;\n\
    vari = sqr*dimRatio - mean*mean;\n\
    vari += eps;\n\
    vari = rsqrt(vari);\n\
    scale_vari = scale_f.s0 * vari;\n\
    vxc_short8 outval;\n\
    half4 tmpVal0, tmpVal1;\n\
\n\
    for(coord.y = 0; coord.y < height; coord.y++)\n\
    {\n\
        for(coord.x = 0; coord.x < width; coord.x += 16)\n\
        {\n\
            int4 coord_out = coord;\n\
            VXC_ReadImage2DArray(src0, input, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
            VXC_DP4x4(tmpData0, src0, src0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
                uniConvertDirInt8Fp32_4x4);\n\
            VXC_DP4x4(tmpData1, src0, src0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
                uniConvertEndInt8Fp32_4x4);\n\
            VXC_DP4x4(tmpData2, src0, src0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
                uniConvertTrdInt8Fp32_4x4);\n\
            VXC_DP4x4(tmpData3, src0, src0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
                uniConvertFthInt8Fp32_4x4);\n\
            tmpData0 *= input_fl_scale;\n\
            tmpData1 *= input_fl_scale;\n\
            tmpData2 *= input_fl_scale;\n\
            tmpData3 *= input_fl_scale;\n\
\n\
            vxc_float4 norm;\n\
            tmpData0 -= mean;\n\
            norm = scale_vari * tmpData0 + bias_val;\n\
            _viv_asm(CONV, tmpVal0, norm);\n\
\n\
            tmpData1 -= mean;\n\
            norm = scale_vari * tmpData1 + bias_val;\n\
            _viv_asm(CONV, tmpVal1, norm);\n\
            VXC_DP2x8(outval, tmpVal0, tmpVal1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\n\
                uniConvertInt32toUint8_2x8);\n\
            VXC_WriteImage2DArray(output, coord_out, outval,\\\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
            coord_out.x += 8;\n\
\n\
            tmpData2 -= mean;\n\
            norm = scale_vari * tmpData2 + bias_val;\n\
            _viv_asm(CONV, tmpVal0, norm);\n\
\n\
            tmpData3 -= mean;\n\
            norm = scale_vari * tmpData3 + bias_val;\n\
            _viv_asm(CONV, tmpVal1, norm);\n\
            VXC_DP2x8(outval, tmpVal0, tmpVal1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\n\
                uniConvertInt32toUint8_2x8);\n\
            VXC_WriteImage2DArray(output, coord_out, outval, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        }\n\
    }\n\
}\n\
\n\
_viv_uniform VXC_512Bits uniConvertInt16Fp32Fst_4x4;\n\
_viv_uniform VXC_512Bits uniConvertInt16Fp32Secd_4x4;\n\
_viv_uniform VXC_512Bits uniConvertInt32toInt16_2x8;\n\
\n\
__kernel __attribute__((reqd_work_group_size(16, 1, 1))) void vxcInstanceNorm_int16(\n\
    image2d_array_t input,\n\
    image2d_array_t bias,\n\
    image2d_array_t scale,\n\
    image2d_array_t output,\n\
    image2d_array_t meanVari,\n\
              float eps)\n\
{\n\
    int4 coord = (int4)(0, 0, get_global_id(2), 0);\n\
    vxc_short8 src0, src1, src2;\n\
    float sum = 0, sqr = 0, scale_vari, bias_val;\n\
    vxc_half8 scale_h;\n\
    vxc_float4 bias_f, scale_f;\n\
    vxc_int4 sumsqr;\n\
    int tmpSum = 0;\n\
    int tmpSqr = 0;\n\
\n\
    for(coord.y = 0; coord.y < height; coord.y++)\n\
    {\n\
        tmpSqr = 0;\n\
        for(coord.x = 0; coord.x < width; coord.x += 8)\n\
        {\n\
            VXC_ReadImage2DArray(src0, input, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
            VXC_DP8x2(sumsqr, src0, src0, VXC_MODIFIER(0, 1, 0, VXC_RM_TowardZero, 0),\\\n\
                uniInt16SumSqr_dp8x2);\n\
            tmpSum += sumsqr.x;\n\
            tmpSqr += sumsqr.y;\n\
        }\n\
        sqr += (tmpSqr * inFlScale_s2);\n\
    }\n\
    sum = tmpSum * input_fl_scale;\n\
\n\
    coord.w = 0;\n\
    VXC_ReadImage(src1, scale, coord.zw, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
    bias_f = read_imagef(bias, coord.zwww);\n\
    _viv_asm(COPY, scale_h, src1, 16);\n\
    VXC_DP4x4(scale_f, scale_h, scale_h, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        UniFP16toFP32Lo4_dp4x4);\n\
    bias_val = bias_f.s0 * output_fl_Scale;\n\
\n\
    float mean, vari;\n\
    mean = sum * dimRatio;\n\
    vari = sqr*dimRatio - mean*mean;\n\
    vari += eps;\n\
    vari = rsqrt(vari);\n\
    scale_vari = scale_f.s0 * vari * output_fl_Scale;\n\
\n\
    vxc_float4 tmpVal0, tmpVal1;\n\
    vxc_int4 tmpData0, tmpData1;\n\
\n\
    for(coord.y = 0; coord.y < height; coord.y++)\n\
    {\n\
        for(coord.x = 0; coord.x < width; coord.x += 8)\n\
        {\n\
            VXC_ReadImage2DArray(src0, input, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
            VXC_DP4x4(tmpVal0, src0, src0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
                uniConvertInt16Fp32Fst_4x4);\n\
            VXC_DP4x4(tmpVal1, src0, src0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
                uniConvertInt16Fp32Secd_4x4);\n\
            tmpVal0 *= input_fl_scale;\n\
            tmpVal1 *= input_fl_scale;\n\
\n\
            vxc_float4 sub, norm;\n\
\n\
            sub = tmpVal0 - mean;\n\
            norm = scale_vari * sub + bias_val;\n\
            tmpData0 = convert_int4_rte(norm);\n\
\n\
            sub = tmpVal1 - mean;\n\
            norm = scale_vari * sub + bias_val;\n\
            tmpData1 = convert_int4_rte(norm);\n\
\n\
            VXC_DP2x8(src2, tmpData0, tmpData1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\n\
                uniConvertInt32toInt16_2x8);\n\
            VXC_WriteImage2DArray(output, coord, src2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        }\n\
    }\n\
}\n\
\n\
__kernel __attribute__((reqd_work_group_size(16, 1, 1))) void vxcInstanceNormInt16_fp16(\n\
    image2d_array_t input,\n\
    image2d_array_t bias,\n\
    image2d_array_t scale,\n\
    image2d_array_t output,\n\
    image2d_array_t meanVari,\n\
              float eps)\n\
{\n\
    int4 coord = (int4)(0, 0, get_global_id(2), 0);\n\
    vxc_short8 src0, src1, src2;\n\
    float sum = 0, sqr = 0, scale_vari, bias_val;\n\
    vxc_half8 in_h, scale_h;\n\
    vxc_float4 bias_f, scale_f;\n\
    vxc_int4 sumsqr;\n\
    int tmpSum = 0;\n\
    int tmpSqr = 0;\n\
\n\
    for(coord.y = 0; coord.y < height; coord.y++)\n\
    {\n\
        tmpSqr = 0;\n\
        for(coord.x = 0; coord.x < width; coord.x += 8)\n\
        {\n\
            VXC_ReadImage2DArray(src0, input, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
            VXC_DP8x2(sumsqr, src0, src0, VXC_MODIFIER(0, 1, 0, VXC_RM_TowardZero, 0),\\\n\
                uniInt16SumSqr_dp8x2);\n\
            tmpSum += sumsqr.x;\n\
            tmpSqr += sumsqr.y;\n\
        }\n\
        sqr += (tmpSqr * inFlScale_s2);\n\
    }\n\
    sum = tmpSum * input_fl_scale;\n\
\n\
    coord.w = 0;\n\
    VXC_ReadImage(src1, scale, coord.zw, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
    bias_f = read_imagef(bias, coord.zwww);\n\
    _viv_asm(COPY, scale_h, src1, 16);\n\
    VXC_DP4x4(scale_f, scale_h, scale_h, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        UniFP16toFP32Lo4_dp4x4);\n\
    bias_val = bias_f.s0;\n\
\n\
    float mean, vari;\n\
    mean = sum * dimRatio;\n\
    vari = sqr*dimRatio - mean*mean;\n\
    vari += eps;\n\
    vari = rsqrt(vari);\n\
    scale_vari = scale_f.s0 * vari;\n\
\n\
    vxc_float4 tmpVal0, tmpVal1;\n\
\n\
    for(coord.y = 0; coord.y < height; coord.y++)\n\
    {\n\
        for(coord.x = 0; coord.x < width; coord.x += 8)\n\
        {\n\
            VXC_ReadImage2DArray(src0, input, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
            VXC_DP4x4(tmpVal0, src0, src0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
                uniConvertInt16Fp32Fst_4x4);\n\
            VXC_DP4x4(tmpVal1, src0, src0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
                uniConvertInt16Fp32Secd_4x4);\n\
            tmpVal0 *= input_fl_scale;\n\
            tmpVal1 *= input_fl_scale;\n\
\n\
            vxc_float4 sub, norm;\n\
            half4 norm_h0, norm_h1;\n\
\n\
            sub = tmpVal0 - mean;\n\
            norm = scale_vari * sub + bias_val;\n\
            _viv_asm(CONV, norm_h0, norm);\n\
\n\
            sub = tmpVal1 - mean;\n\
            norm = scale_vari * sub + bias_val;\n\
            _viv_asm(CONV, norm_h1, norm);\n\
\n\
            VXC_DP2x8(src2, norm_h0, norm_h1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
                uniConvertInt32toUint8_2x8);\n\
            VXC_WriteImage2DArray(output, coord, src2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        }\n\
    }\n\
}\n\
\n\
\n\
"; /* end of vsi_nn_kernel_instancenormalize_i8_vx*/

static const char vsi_nn_kernel_internal_reduceProdAxis0_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
_viv_uniform float       outputScale;\n\
_viv_uniform float       output_offset_asymmetric;\n\
_viv_uniform float       inputScale;\n\
_viv_uniform float       input_offset_asymmetric;\n\
_viv_uniform VXC_512Bits uniGetLoData_4x4;\n\
_viv_uniform VXC_512Bits uniGetHiData_4x4;\n\
_viv_uniform VXC_512Bits uniConvBF16toF32_Part0_2x8;\n\
_viv_uniform VXC_512Bits uniConvBF16toF32_Part1_2x8;\n\
\n\
_viv_uniform int         inputWidth;\n\
_viv_uniform VXC_512Bits uniGetEndLoData_2x8;\n\
_viv_uniform VXC_512Bits uniGetEndHiData_2x8;\n\
\n\
#define REDUCEPROD_PROCESS_AXIS0(read_fun, IN_SCALE, IN_OFFSET) \\\n\
    while(coord.x < inputWidth) \\\n\
    { \\\n\
        read_fun(val0, input,  coord, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, img_val, val0, 16); \\\n\
        VXC_DP4x4(tmpProdLo, img_val, img_val, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetLoData_4x4); \\\n\
        VXC_DP4x4(tmpProdHi, img_val, img_val, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetHiData_4x4); \\\n\
        tmpProdLo = (tmpProdLo - IN_OFFSET) * IN_SCALE; \\\n\
        tmpProdHi = (tmpProdHi - IN_OFFSET) * IN_SCALE; \\\n\
        tmpProd = tmpProdLo * tmpProdHi; \\\n\
        prodValue = prodValue * tmpProd; \\\n\
        coord.x += 8; \\\n\
    } \\\n\
    vxc_ushort8 tmpProdInt0, tmpProdInt1; \\\n\
    vxc_ushort8 tmpOnesInt = {0, 16256, 0, 16256, 0, 16256, 0, 16256}; \\\n\
    read_fun(val0, input,  coord, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, img_val, val0, 16); \\\n\
    VXC_DP4x4(tmpProdLo, img_val, img_val, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetLoData_4x4); \\\n\
    VXC_DP4x4(tmpProdHi, img_val, img_val, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetHiData_4x4); \\\n\
    tmpProdLo = (tmpProdLo - IN_OFFSET) * IN_SCALE; \\\n\
    tmpProdHi = (tmpProdHi - IN_OFFSET) * IN_SCALE; \\\n\
    _viv_asm(COPY, tmpProdInt0, tmpProdLo, 16); \\\n\
    _viv_asm(COPY, tmpProdInt1, tmpProdHi, 16); \\\n\
    VXC_DP2x8(tmpProdInt0, tmpProdInt0, tmpOnesInt,\\\n\
    VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniGetEndLoData_2x8); \\\n\
    VXC_DP2x8(tmpProdInt1, tmpProdInt1, tmpOnesInt,\\\n\
    VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniGetEndHiData_2x8); \\\n\
    _viv_asm(COPY, tmpProdLo, tmpProdInt0, 16); \\\n\
    _viv_asm(COPY, tmpProdHi, tmpProdInt1, 16); \\\n\
    tmpProd = tmpProdLo * tmpProdHi; \\\n\
    prodValue = prodValue * tmpProd; \\\n\
    tmpProd.xy = prodValue.xy * prodValue.zw; \\\n\
    prodValue.x = tmpProd.x * tmpProd.y;\n\
\n\
#define REDUCEPROD_PROCESS_AXIS0_SAVE(dst_type, save_type, conv_mode, OUT_SCALE, OUT_OFFSET, write_fun) \\\n\
    dst_type vec1; \\\n\
    save_type dst; \\\n\
    prodValue = prodValue * OUT_SCALE + OUT_OFFSET; \\\n\
    _viv_asm(conv_mode, vec1, prodValue); \\\n\
    _viv_asm(COPY, dst, vec1, 16); \\\n\
    coord.x = 0; \\\n\
    write_fun(output, coord, dst, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
\n\
#define REDUCEPROD_AXIS0(src_name, dst_name, src_type, copy_type, dst_type,\\\n\
              save_type, conv_mode, OUT_SCALE, OUT_OFFSET, IN_SCALE, IN_OFFSET) \\\n\
__kernel void vxcReduceProdAxis0_##src_name##to##dst_name \\\n\
    ( \\\n\
    __read_only  image2d_array_t input, \\\n\
    __write_only image2d_array_t output, \\\n\
    int   axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(0, get_global_id(0), get_global_id(1), 0); \\\n\
    vxc_float4 prodValue = {1.0f, 1.0f, 1.0f, 1.0f}; \\\n\
    vxc_float4 tmpProdLo, tmpProdHi, tmpProd; \\\n\
    src_type img_val; \\\n\
    copy_type val0; \\\n\
    REDUCEPROD_PROCESS_AXIS0(VXC_ReadImage2DArray, IN_SCALE, IN_OFFSET); \\\n\
    REDUCEPROD_PROCESS_AXIS0_SAVE(dst_type, save_type, conv_mode,\\\n\
    OUT_SCALE, OUT_OFFSET, VXC_WriteImage2DArray); \\\n\
}\n\
\n\
REDUCEPROD_AXIS0(F16, F16, vxc_half8, vxc_short8, half4,  vxc_short8, CONV, 1, 0, 1, 0)\n\
REDUCEPROD_AXIS0(F16, I16, vxc_half8, vxc_short8, short4, vxc_short8, CONV_SAT_RTE, outputScale, 0, 1, 0)\n\
REDUCEPROD_AXIS0(F16, I8,  vxc_half8, vxc_short8, char4,  vxc_char8,  CONV_SAT_RTE, outputScale, 0, 1, 0)\n\
REDUCEPROD_AXIS0(F16, U8,  vxc_half8, vxc_short8, uchar4, vxc_uchar8,\\\n\
                    CONV_SAT_RTE, outputScale, output_offset_asymmetric, 1, 0)\n\
REDUCEPROD_AXIS0(I16, F16, vxc_short8, vxc_short8, half4,  vxc_short8, CONV, 1, 0, inputScale, 0)\n\
REDUCEPROD_AXIS0(I8, F16, vxc_char16, vxc_char16, half4,  vxc_short8, CONV, 1, 0, inputScale, 0)\n\
REDUCEPROD_AXIS0(U8, F16, vxc_uchar16, vxc_uchar16, half4,  vxc_short8,\\\n\
CONV, 1, 0, inputScale, input_offset_asymmetric)\n\
\n\
REDUCEPROD_AXIS0(I16, I16, vxc_short8, vxc_short8, short4,\\\n\
vxc_short8, CONV_SAT_RTE, outputScale, 0, inputScale, 0)\n\
REDUCEPROD_AXIS0(I8, I8,  vxc_char16, vxc_char16,  char4,\\\n\
vxc_char8, CONV_SAT_RTE, outputScale, 0, inputScale, 0)\n\
REDUCEPROD_AXIS0(U8, U8,  vxc_uchar16, vxc_uchar16, uchar4, vxc_uchar8, CONV_SAT_RTE,\\\n\
outputScale, output_offset_asymmetric, inputScale, input_offset_asymmetric)\n\
\n\
#define REDUCEPROD_AXIS0_2D(src_name, dst_name, src_type, copy_type,\\\n\
                           dst_type, save_type, conv_mode, OUT_SCALE, OUT_OFFSET, IN_SCALE, IN_OFFSET) \\\n\
__kernel void vxcReduceProdAxis0_##src_name##to##dst_name##_2D \\\n\
    ( \\\n\
    __read_only  image2d_array_t input, \\\n\
    __write_only image2d_array_t output, \\\n\
    int   axisVal \\\n\
    ) \\\n\
{ \\\n\
    int2 coord = (int2)(0, get_global_id(0)); \\\n\
    vxc_float4 prodValue = {1.0f, 1.0f, 1.0f, 1.0f}; \\\n\
    vxc_float4 tmpProdLo, tmpProdHi, tmpProd; \\\n\
    src_type img_val; \\\n\
    copy_type val0; \\\n\
    REDUCEPROD_PROCESS_AXIS0(VXC_ReadImage, IN_SCALE, IN_OFFSET); \\\n\
    REDUCEPROD_PROCESS_AXIS0_SAVE(dst_type, save_type, conv_mode,\\\n\
    OUT_SCALE, OUT_OFFSET, VXC_WriteImage); \\\n\
}\n\
\n\
REDUCEPROD_AXIS0_2D(F16, F16, vxc_half8, vxc_short8, half4,  vxc_short8, CONV, 1, 0, 1, 0)\n\
REDUCEPROD_AXIS0_2D(F16, I16, vxc_half8, vxc_short8, short4, vxc_short8, CONV_SAT_RTE, outputScale, 0, 1, 0)\n\
REDUCEPROD_AXIS0_2D(F16, I8,  vxc_half8, vxc_short8, char4,  vxc_char8,  CONV_SAT_RTE, outputScale, 0, 1, 0)\n\
REDUCEPROD_AXIS0_2D(F16, U8,  vxc_half8, vxc_short8, uchar4, vxc_uchar8,\\\n\
                    CONV_SAT_RTE, outputScale, output_offset_asymmetric, 1, 0)\n\
REDUCEPROD_AXIS0_2D(I16, F16, vxc_short8, vxc_short8, half4,  vxc_short8, CONV, 1, 0, inputScale, 0)\n\
REDUCEPROD_AXIS0_2D(I8, F16, vxc_char16, vxc_char16, half4,  vxc_short8, CONV, 1, 0, inputScale, 0)\n\
REDUCEPROD_AXIS0_2D(U8, F16, vxc_uchar16, vxc_uchar16, half4, \\\n\
vxc_short8, CONV, 1, 0, inputScale, input_offset_asymmetric)\n\
REDUCEPROD_AXIS0_2D(I16, I16, vxc_short8, vxc_short8, short4,\\\n\
vxc_short8, CONV_SAT_RTE, outputScale, 0, inputScale, 0)\n\
REDUCEPROD_AXIS0_2D(I8, I8,  vxc_char16, vxc_char16,  char4,\\\n\
vxc_char8, CONV_SAT_RTE, outputScale, 0, inputScale, 0)\n\
REDUCEPROD_AXIS0_2D(U8, U8,  vxc_uchar16, vxc_uchar16, uchar4, vxc_uchar8, CONV_SAT_RTE,\\\n\
outputScale, output_offset_asymmetric, inputScale, input_offset_asymmetric)\n\
\n\
\n\
#define REDUCEPROD_PROCESS_AXIS0_BF16(read_fun) \\\n\
    while(coord.x < inputWidth) \\\n\
    { \\\n\
        read_fun(img_val, input,  coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        VXC_DP2x8(val0, img_val, zero, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvBF16toF32_Part0_2x8); \\\n\
        _viv_asm(COPY, tmpProdLo, val0, 16); \\\n\
        VXC_DP2x8(val0, img_val, zero, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvBF16toF32_Part1_2x8); \\\n\
        _viv_asm(COPY, tmpProdHi, val0, 16); \\\n\
        tmpProd = tmpProdLo * tmpProdHi; \\\n\
        prodValue = prodValue * tmpProd; \\\n\
        coord.x += 8; \\\n\
    } \\\n\
    vxc_ushort8 tmpProdInt0, tmpProdInt1; \\\n\
    vxc_ushort8 tmpOnesInt = {0, 16256, 0, 16256, 0, 16256, 0, 16256}; \\\n\
    read_fun(img_val, input,  coord, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_DP2x8(val0, img_val, zero, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part0_2x8); \\\n\
    _viv_asm(COPY, tmpProdLo, val0, 16); \\\n\
    VXC_DP2x8(val0, img_val, zero, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part1_2x8); \\\n\
    _viv_asm(COPY, tmpProdHi, val0, 16); \\\n\
    _viv_asm(COPY, tmpProdInt0, tmpProdLo, 16); \\\n\
    _viv_asm(COPY, tmpProdInt1, tmpProdHi, 16); \\\n\
    VXC_DP2x8(tmpProdInt0, tmpProdInt0, tmpOnesInt,\\\n\
    VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniGetEndLoData_2x8); \\\n\
    VXC_DP2x8(tmpProdInt1, tmpProdInt1, tmpOnesInt,\\\n\
    VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniGetEndHiData_2x8); \\\n\
    _viv_asm(COPY, tmpProdLo, tmpProdInt0, 16); \\\n\
    _viv_asm(COPY, tmpProdHi, tmpProdInt1, 16); \\\n\
    tmpProd = tmpProdLo * tmpProdHi; \\\n\
    prodValue = prodValue * tmpProd; \\\n\
    tmpProd.xy = prodValue.xy * prodValue.zw; \\\n\
    prodValue.x = tmpProd.x * tmpProd.y;\n\
\n\
#define REDUCEPROD_PROCESS_AXIS0_BF16_SAVE(write_fun) \\\n\
    vxc_ushort8 dst; \\\n\
    _viv_asm(COPY, dst, prodValue, 16); \\\n\
    dst.s0 = dst.s1; \\\n\
    coord.x = 0; \\\n\
    write_fun(output, coord, dst, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
\n\
__kernel void vxcReduceProdAxis0_BF16toBF16\n\
    (\n\
    __read_only  image2d_array_t input,\n\
    __write_only image2d_array_t output,\n\
    int   axisVal\n\
    )\n\
{\n\
    int4 coord = (int4)(0, get_global_id(0), get_global_id(1), 0);\n\
    vxc_float4 prodValue = {1.0f, 1.0f, 1.0f, 1.0f};\n\
    vxc_float4 tmpProdLo, tmpProdHi, tmpProd;\n\
    vxc_short8 img_val;\n\
    vxc_short8 val0;\n\
    vxc_short8 zero = (vxc_short8)(0, 0, 0, 0, 0, 0, 0, 0);\n\
    REDUCEPROD_PROCESS_AXIS0_BF16(VXC_ReadImage2DArray);\n\
    REDUCEPROD_PROCESS_AXIS0_BF16_SAVE(VXC_WriteImage2DArray);\n\
}\n\
\n\
__kernel void vxcReduceProdAxis0_BF16toBF16_2D\n\
    (\n\
    __read_only  image2d_array_t input,\n\
    __write_only image2d_array_t output,\n\
    int   axisVal\n\
    )\n\
{\n\
    int2 coord = (int2)(0, get_global_id(0));\n\
    vxc_float4 prodValue = {1.0f, 1.0f, 1.0f, 1.0f};\n\
    vxc_float4 tmpProdLo, tmpProdHi, tmpProd;\n\
    vxc_short8 img_val;\n\
    vxc_short8 val0;\n\
    vxc_short8 zero = (vxc_short8)(0, 0, 0, 0, 0, 0, 0, 0);\n\
    REDUCEPROD_PROCESS_AXIS0_BF16(VXC_ReadImage);\n\
    REDUCEPROD_PROCESS_AXIS0_BF16_SAVE(VXC_WriteImage);\n\
}\n\
\n\
"; /* end of vsi_nn_kernel_internal_reduceProdAxis0_vx*/

static const char vsi_nn_kernel_internal_reduceProdAxis1_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform float       outputScale;\n\
_viv_uniform float       output_offset_asymmetric;\n\
_viv_uniform float       inputScale;\n\
_viv_uniform float       input_offset_asymmetric;\n\
_viv_uniform VXC_512Bits uniGetLoData_4x4;\n\
_viv_uniform VXC_512Bits uniGetHiData_4x4;\n\
_viv_uniform VXC_512Bits uniConvBF16toF32_Part0_2x8;\n\
_viv_uniform VXC_512Bits uniConvBF16toF32_Part1_2x8;\n\
\n\
_viv_uniform int         axisSize;\n\
_viv_uniform VXC_512Bits uniConvertInt32toUint8_2x8;\n\
_viv_uniform VXC_512Bits uniExtractOddData_2x8;\n\
\n\
#define REDUCEPROD_PROCESS_AXIS1(read_fun, IN_SCALE, IN_OFFSET) \\\n\
    do \\\n\
    { \\\n\
        read_fun(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, vec0, in0, 16); \\\n\
        VXC_DP4x4(tmpProdLo, vec0, vec0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetLoData_4x4); \\\n\
        VXC_DP4x4(tmpProdHi, vec0, vec0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetHiData_4x4); \\\n\
        tmpProdLo = (tmpProdLo - IN_OFFSET) * IN_SCALE; \\\n\
        tmpProdHi = (tmpProdHi - IN_OFFSET) * IN_SCALE; \\\n\
        prodValueLo = prodValueLo * tmpProdLo; \\\n\
        prodValueHi = prodValueHi * tmpProdHi; \\\n\
        coord.y++; \\\n\
    } \\\n\
    while(coord.y < axisSize);\n\
\n\
#define REDUCEPROD_PROCESS_AXIS1_SAVE(dst_type, save_type, conv_mode, OUT_SCALE, OUT_OFFSET, write_fun) \\\n\
    coord.y = 0; \\\n\
    dst_type dst0, dst1; \\\n\
    save_type vect; \\\n\
    prodValueLo = prodValueLo * OUT_SCALE + OUT_OFFSET; \\\n\
    _viv_asm(conv_mode, dst0, prodValueLo); \\\n\
    prodValueHi = prodValueHi * OUT_SCALE + OUT_OFFSET; \\\n\
    _viv_asm(conv_mode, dst1, prodValueHi); \\\n\
    VXC_DP2x8(vect, dst0, dst1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvertInt32toUint8_2x8); \\\n\
    write_fun(output, coord, vect, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
\n\
#define REDUCEPROD_AXIS1(src_name, dst_name, src_type, copy_type, dst_type, save_type,\\\n\
conv_mode, OUT_SCALE, OUT_OFFSET, IN_SCALE, IN_OFFSET) \\\n\
__kernel void vxcReduceProdAxis1_##src_name##to##dst_name \\\n\
    ( \\\n\
    __read_only  image2d_array_t input, \\\n\
    __write_only image2d_array_t output, \\\n\
    int   axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(get_global_id(0), 0, get_global_id(1), 0); \\\n\
    vxc_float4 prodValueLo = {1.0f, 1.0f, 1.0f, 1.0f}; \\\n\
    vxc_float4 prodValueHi = {1.0f, 1.0f, 1.0f, 1.0f};\\\n\
    vxc_float4 tmpProdLo, tmpProdHi; \\\n\
    src_type vec0; \\\n\
    copy_type in0; \\\n\
    REDUCEPROD_PROCESS_AXIS1(VXC_ReadImage2DArray, IN_SCALE, IN_OFFSET) \\\n\
    REDUCEPROD_PROCESS_AXIS1_SAVE(dst_type, save_type, conv_mode,\\\n\
    OUT_SCALE, OUT_OFFSET, VXC_WriteImage2DArray); \\\n\
}\n\
\n\
\n\
\n\
REDUCEPROD_AXIS1(F16, F16, vxc_half8, vxc_short8, half4,  vxc_short8, CONV, 1, 0, 1, 0)\n\
REDUCEPROD_AXIS1(F16, I16, vxc_half8, vxc_short8, short4, vxc_short8, CONV_SAT_RTE, outputScale, 0, 1, 0)\n\
REDUCEPROD_AXIS1(F16, I8,  vxc_half8, vxc_short8, char4,  vxc_char8,  CONV_SAT_RTE, outputScale, 0, 1, 0)\n\
REDUCEPROD_AXIS1(F16, U8,  vxc_half8, vxc_short8, uchar4, vxc_uchar8,\\\n\
CONV_SAT_RTE, outputScale, output_offset_asymmetric, 1, 0)\n\
REDUCEPROD_AXIS1(I16, F16, vxc_short8, vxc_short8, half4,  vxc_short8, CONV, 1, 0, inputScale, 0)\n\
REDUCEPROD_AXIS1(I8, F16, vxc_char16, vxc_char16, half4,  vxc_short8, CONV, 1, 0, inputScale, 0)\n\
REDUCEPROD_AXIS1(U8, F16, vxc_uchar16, vxc_uchar16, half4, \\\n\
vxc_short8, CONV, 1, 0, inputScale, input_offset_asymmetric)\n\
REDUCEPROD_AXIS1(I16, I16, vxc_short8, vxc_short8, short4,\\\n\
vxc_short8, CONV_SAT_RTE, outputScale, 0, inputScale, 0)\n\
REDUCEPROD_AXIS1(I8, I8,  vxc_char16, vxc_char16,  char4,\\\n\
vxc_char8, CONV_SAT_RTE, outputScale, 0, inputScale, 0)\n\
REDUCEPROD_AXIS1(U8, U8,  vxc_uchar16, vxc_uchar16, uchar4, vxc_uchar8, CONV_SAT_RTE,\\\n\
outputScale, output_offset_asymmetric, inputScale, input_offset_asymmetric)\n\
\n\
\n\
#define REDUCEPROD_AXIS1_2D(src_name, dst_name, src_type, copy_type, dst_type, save_type,\\\n\
conv_mode, OUT_SCALE, OUT_OFFSET, IN_SCALE, IN_OFFSET) \\\n\
__kernel void vxcReduceProdAxis1_##src_name##to##dst_name##_2D \\\n\
    ( \\\n\
    __read_only  image2d_array_t input, \\\n\
    __write_only image2d_array_t output, \\\n\
    int   axisVal \\\n\
    ) \\\n\
{ \\\n\
    int2 coord = (int2)(get_global_id(0), 0); \\\n\
    vxc_float4 prodValueLo = {1.0f, 1.0f, 1.0f, 1.0f}; \\\n\
    vxc_float4 prodValueHi = {1.0f, 1.0f, 1.0f, 1.0f};\\\n\
    vxc_float4 tmpProdLo, tmpProdHi; \\\n\
    src_type vec0; \\\n\
    copy_type in0; \\\n\
    REDUCEPROD_PROCESS_AXIS1(VXC_ReadImage, IN_SCALE, IN_OFFSET) \\\n\
    REDUCEPROD_PROCESS_AXIS1_SAVE(dst_type, save_type, conv_mode, OUT_SCALE, OUT_OFFSET, VXC_WriteImage); \\\n\
}\n\
\n\
REDUCEPROD_AXIS1_2D(F16, F16, vxc_half8, vxc_short8, half4,  vxc_short8, CONV, 1, 0, 1, 0)\n\
REDUCEPROD_AXIS1_2D(F16, I16, vxc_half8, vxc_short8, short4, vxc_short8, CONV_SAT_RTE, outputScale, 0, 1, 0)\n\
REDUCEPROD_AXIS1_2D(F16, I8,  vxc_half8, vxc_short8, char4,  vxc_char8,  CONV_SAT_RTE, outputScale, 0, 1, 0)\n\
REDUCEPROD_AXIS1_2D(F16, U8,  vxc_half8, vxc_short8, uchar4, vxc_uchar8,\\\n\
CONV_SAT_RTE, outputScale, output_offset_asymmetric, 1, 0)\n\
REDUCEPROD_AXIS1_2D(I16, F16, vxc_short8, vxc_short8, half4,  vxc_short8, CONV, 1, 0, inputScale, 0)\n\
REDUCEPROD_AXIS1_2D(I8, F16, vxc_char16, vxc_char16, half4,  vxc_short8, CONV, 1, 0, inputScale, 0)\n\
REDUCEPROD_AXIS1_2D(U8, F16, vxc_uchar16, vxc_uchar16, half4,  vxc_short8,\\\n\
CONV, 1, 0, inputScale, input_offset_asymmetric)\n\
REDUCEPROD_AXIS1_2D(I16, I16, vxc_short8, vxc_short8, short4,\\\n\
vxc_short8, CONV_SAT_RTE, outputScale, 0, inputScale, 0)\n\
REDUCEPROD_AXIS1_2D(I8, I8,  vxc_char16, vxc_char16,  char4,\\\n\
vxc_char8, CONV_SAT_RTE, outputScale, 0, inputScale, 0)\n\
REDUCEPROD_AXIS1_2D(U8, U8,  vxc_uchar16, vxc_uchar16, uchar4, vxc_uchar8, CONV_SAT_RTE,\\\n\
outputScale, output_offset_asymmetric, inputScale, input_offset_asymmetric)\n\
\n\
#define REDUCEPROD_PROCESS_AXIS1_BF16(read_fun) \\\n\
    do \\\n\
    { \\\n\
        read_fun(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        VXC_DP2x8(vec0, in0, zero, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part0_2x8); \\\n\
        _viv_asm(COPY, tmpProdLo, vec0, 16); \\\n\
        VXC_DP2x8(vec0, in0, zero, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part1_2x8); \\\n\
        _viv_asm(COPY, tmpProdHi, vec0, 16); \\\n\
        prodValueLo = prodValueLo * tmpProdLo; \\\n\
        prodValueHi = prodValueHi * tmpProdHi; \\\n\
        coord.y++; \\\n\
    } \\\n\
    while(coord.y < axisSize);\n\
\n\
\n\
#define REDUCEPROD_PROCESS_AXIS1_SAVE_BF16(write_fun) \\\n\
    coord.y = 0; \\\n\
    vxc_ushort8 dst0, dst1; \\\n\
    vxc_ushort8 vect; \\\n\
    _viv_asm(COPY, dst0, prodValueLo, 16); \\\n\
    _viv_asm(COPY, dst1, prodValueHi, 16); \\\n\
    VXC_DP2x8(vect, dst0, dst1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniExtractOddData_2x8); \\\n\
    write_fun(output, coord, vect, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
__kernel void vxcReduceProdAxis1_BF16toBF16\n\
    (\n\
    __read_only  image2d_array_t input,\n\
    __write_only image2d_array_t output,\n\
    int   axisVal\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), 0, get_global_id(1), 0);\n\
    vxc_float4 prodValueLo = {1.0f, 1.0f, 1.0f, 1.0f};\n\
    vxc_float4 prodValueHi = {1.0f, 1.0f, 1.0f, 1.0f};\n\
    vxc_float4 tmpProdLo, tmpProdHi;\n\
    vxc_short8 vec0;\n\
    vxc_short8 in0;\n\
    vxc_short8 zero = (vxc_short8)(0, 0, 0, 0, 0, 0, 0, 0);\n\
    REDUCEPROD_PROCESS_AXIS1_BF16(VXC_ReadImage2DArray)\n\
    REDUCEPROD_PROCESS_AXIS1_SAVE_BF16(VXC_WriteImage2DArray);\n\
}\n\
\n\
__kernel void vxcReduceProdAxis1_BF16toBF16_2D\n\
    (\n\
    __read_only  image2d_array_t input,\n\
    __write_only image2d_array_t output,\n\
    int   axisVal\n\
    )\n\
{\n\
    int2 coord = (int2)(get_global_id(0), 0);\n\
    vxc_float4 prodValueLo = {1.0f, 1.0f, 1.0f, 1.0f};\n\
    vxc_float4 prodValueHi = {1.0f, 1.0f, 1.0f, 1.0f};\n\
    vxc_float4 tmpProdLo, tmpProdHi;\n\
    vxc_short8 vec0;\n\
    vxc_short8 in0;\n\
    vxc_short8 zero = (vxc_short8)(0, 0, 0, 0, 0, 0, 0, 0);\n\
    REDUCEPROD_PROCESS_AXIS1_BF16(VXC_ReadImage)\n\
    REDUCEPROD_PROCESS_AXIS1_SAVE_BF16(VXC_WriteImage);\n\
}\n\
"; /* end of vsi_nn_kernel_internal_reduceProdAxis1_vx*/

static const char vsi_nn_kernel_internal_reduceProdAxis2_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform float       outputScale;\n\
_viv_uniform float       output_offset_asymmetric;\n\
_viv_uniform float       inputScale;\n\
_viv_uniform float       input_offset_asymmetric;\n\
_viv_uniform VXC_512Bits uniGetLoData_4x4;\n\
_viv_uniform VXC_512Bits uniGetHiData_4x4;\n\
_viv_uniform VXC_512Bits uniConvBF16toF32_Part0_2x8;\n\
_viv_uniform VXC_512Bits uniConvBF16toF32_Part1_2x8;\n\
\n\
_viv_uniform int         axisSize;\n\
_viv_uniform VXC_512Bits uniConvertInt32toUint8_2x8;\n\
_viv_uniform VXC_512Bits uniExtractOddData_2x8;\n\
\n\
#define REDUCEPROD_PROCESS_AXIS2(read_fun, IN_SCALE, IN_OFFSET) \\\n\
    do \\\n\
    { \\\n\
        read_fun(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, vec0, in0, 16); \\\n\
        VXC_DP4x4(tmpProdLo, vec0, vec0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetLoData_4x4); \\\n\
        VXC_DP4x4(tmpProdHi, vec0, vec0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetHiData_4x4); \\\n\
        tmpProdLo = (tmpProdLo - IN_OFFSET) * IN_SCALE; \\\n\
        tmpProdHi = (tmpProdHi - IN_OFFSET) * IN_SCALE; \\\n\
        prodValueLo = prodValueLo * tmpProdLo; \\\n\
        prodValueHi = prodValueHi * tmpProdHi; \\\n\
        coord.z++; \\\n\
    } \\\n\
    while(coord.z < axisSize);\n\
\n\
\n\
#define REDUCEPROD_PROCESS_AXIS2_SAVE(dst_type, save_type, conv_mode, OUT_SCALE, OUT_OFFSET, write_fun) \\\n\
    coord.z = 0; \\\n\
    dst_type dst0, dst1; \\\n\
    save_type vect; \\\n\
    prodValueLo = prodValueLo * OUT_SCALE + OUT_OFFSET; \\\n\
    _viv_asm(conv_mode, dst0, prodValueLo); \\\n\
    prodValueHi = prodValueHi * OUT_SCALE + OUT_OFFSET; \\\n\
    _viv_asm(conv_mode, dst1, prodValueHi); \\\n\
    VXC_DP2x8(vect, dst0, dst1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvertInt32toUint8_2x8); \\\n\
    write_fun(output, coord, vect, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
\n\
#define REDUCEPROD_AXIS2(src_name, dst_name, src_type, copy_type, dst_type, save_type,\\\n\
conv_mode, OUT_SCALE, OUT_OFFSET, IN_SCALE, IN_OFFSET) \\\n\
__kernel void vxcReduceProdAxis2_##src_name##to##dst_name \\\n\
    ( \\\n\
    __read_only  image2d_array_t input, \\\n\
    __write_only image2d_array_t output, \\\n\
    int   axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), 0, 0); \\\n\
    vxc_float4 prodValueLo = {1.0f, 1.0f, 1.0f, 1.0f}; \\\n\
    vxc_float4 prodValueHi = {1.0f, 1.0f, 1.0f, 1.0f};\\\n\
    vxc_float4 tmpProdLo, tmpProdHi; \\\n\
    src_type vec0; \\\n\
    copy_type in0; \\\n\
    REDUCEPROD_PROCESS_AXIS2(VXC_ReadImage2DArray, IN_SCALE, IN_OFFSET) \\\n\
    REDUCEPROD_PROCESS_AXIS2_SAVE(dst_type, save_type, conv_mode,\\\n\
    OUT_SCALE, OUT_OFFSET,  VXC_WriteImage2DArray); \\\n\
}\n\
\n\
\n\
\n\
REDUCEPROD_AXIS2(F16, F16, vxc_half8, vxc_short8, half4,  vxc_short8, CONV, 1, 0, 1, 0)\n\
REDUCEPROD_AXIS2(F16, I16, vxc_half8, vxc_short8, short4, vxc_short8, CONV_SAT_RTE, outputScale, 0, 1, 0)\n\
REDUCEPROD_AXIS2(F16, I8,  vxc_half8, vxc_short8, char4,  vxc_char8,  CONV_SAT_RTE, outputScale, 0, 1, 0)\n\
REDUCEPROD_AXIS2(F16, U8,  vxc_half8, vxc_short8, uchar4, vxc_uchar8,\\\n\
CONV_SAT_RTE, outputScale, output_offset_asymmetric, 1, 0)\n\
REDUCEPROD_AXIS2(I16, F16, vxc_short8, vxc_short8, half4,  vxc_short8, CONV, 1, 0, inputScale, 0)\n\
REDUCEPROD_AXIS2(I8, F16, vxc_char16, vxc_char16, half4,  vxc_short8, CONV, 1, 0, inputScale, 0)\n\
REDUCEPROD_AXIS2(U8, F16, vxc_uchar16, vxc_uchar16, half4,  vxc_short8,\\\n\
CONV, 1, 0, inputScale, input_offset_asymmetric)\n\
\n\
REDUCEPROD_AXIS2(I16, I16, vxc_short8, vxc_short8, short4,\\\n\
vxc_short8, CONV_SAT_RTE, outputScale, 0, inputScale, 0)\n\
REDUCEPROD_AXIS2(I8, I8,  vxc_char16, vxc_char16,  char4,\\\n\
vxc_char8, CONV_SAT_RTE, outputScale, 0, inputScale, 0)\n\
REDUCEPROD_AXIS2(U8, U8,  vxc_uchar16, vxc_uchar16, uchar4, vxc_uchar8, CONV_SAT_RTE,\\\n\
outputScale, output_offset_asymmetric, inputScale, input_offset_asymmetric)\n\
\n\
#define REDUCEPROD_PROCESS_AXIS2_BF16(read_fun) \\\n\
    do \\\n\
    { \\\n\
        read_fun(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        VXC_DP2x8(vec0, in0, zero, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part0_2x8); \\\n\
        _viv_asm(COPY, tmpProdLo, vec0, 16); \\\n\
        VXC_DP2x8(vec0, in0, zero, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part1_2x8); \\\n\
        _viv_asm(COPY, tmpProdHi, vec0, 16); \\\n\
        prodValueLo = prodValueLo * tmpProdLo; \\\n\
        prodValueHi = prodValueHi * tmpProdHi; \\\n\
        coord.z++; \\\n\
    } \\\n\
    while(coord.z < axisSize);\n\
\n\
\n\
#define REDUCEPROD_PROCESS_AXIS2_SAVE_BF16(write_fun) \\\n\
    coord.z = 0; \\\n\
    vxc_ushort8 dst0, dst1; \\\n\
    vxc_ushort8 vect; \\\n\
    _viv_asm(COPY, dst0, prodValueLo, 16); \\\n\
    _viv_asm(COPY, dst1, prodValueHi, 16); \\\n\
    VXC_DP2x8(vect, dst0, dst1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniExtractOddData_2x8); \\\n\
    write_fun(output, coord, vect, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
__kernel void vxcReduceProdAxis2_BF16toBF16\n\
    (\n\
    __read_only  image2d_array_t input,\n\
    __write_only image2d_array_t output,\n\
    int   axisVal\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), 0, 0);\n\
    vxc_float4 prodValueLo = {1.0f, 1.0f, 1.0f, 1.0f};\n\
    vxc_float4 prodValueHi = {1.0f, 1.0f, 1.0f, 1.0f};\n\
    vxc_float4 tmpProdLo, tmpProdHi;\n\
    vxc_short8 vec0;\n\
    vxc_short8 in0;\n\
    vxc_short8 zero = (vxc_short8)(0, 0, 0, 0, 0, 0, 0, 0);\n\
    REDUCEPROD_PROCESS_AXIS2_BF16(VXC_ReadImage2DArray)\n\
    REDUCEPROD_PROCESS_AXIS2_SAVE_BF16(VXC_WriteImage2DArray);\n\
}\n\
"; /* end of vsi_nn_kernel_internal_reduceProdAxis2_vx*/

static const char vsi_nn_kernel_internal_reduceallAxis0_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
_viv_uniform int         axisSize;\n\
\n\
_viv_uniform VXC_512Bits  uniS8AddAll_16x1;\n\
\n\
#define REDUCEALL_AXIS0_PROCESS(read_fun, write_fun) \\\n\
    vxc_char16 ones  = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}; \\\n\
    vxc_char16 zeros = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; \\\n\
    int4 sum_val = 0; \\\n\
    result = ones; \\\n\
    do \\\n\
    { \\\n\
        read_fun(val0, input,  coord, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0)); \\\n\
        val = val0 != zeros ? ones : zeros; \\\n\
        VXC_DP16x1(sum_val, val, val, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0), uniS8AddAll_16x1); \\\n\
        if (sum_val.x != 16) \\\n\
        { \\\n\
            result = zeros; \\\n\
            break; \\\n\
        } \\\n\
        coord.x += 16; \\\n\
    } \\\n\
    while(coord.x < axisSize); \\\n\
    coord.x = 0; \\\n\
    write_fun(output, coord, result, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
\n\
\n\
__kernel void vxcReduceallAxis0_I8toI8\n\
    (\n\
    __read_only  image2d_array_t input,\n\
    __write_only image2d_array_t output,\n\
    int   axisVal\n\
    )\n\
{\n\
    int4 coord = (int4)(0, get_global_id(0), get_global_id(1), 0);\n\
    vxc_char16 val0;\n\
    vxc_char16 val, result;\n\
    REDUCEALL_AXIS0_PROCESS(VXC_ReadImage2DArray, VXC_WriteImage2DArray)\n\
}\n\
\n\
__kernel void vxcReduceallAxis0_I8toI8_2D\n\
    (\n\
    __read_only  image2d_array_t input,\n\
    __write_only image2d_array_t output,\n\
    int   axisVal\n\
    )\n\
{\n\
    int2 coord = (int2)(0, get_global_id(0));\n\
    vxc_char16 val0;\n\
    vxc_char16 val, result;\n\
    REDUCEALL_AXIS0_PROCESS(VXC_ReadImage, VXC_WriteImage)\n\
}\n\
\n\
\n\
"; /* end of vsi_nn_kernel_internal_reduceallAxis0_vx*/

static const char vsi_nn_kernel_internal_reduceallAxis1_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform int         axisSize;\n\
\n\
#define REDUCEALL_AXIS1_PROCESS(read_fun, write_fun) \\\n\
    vxc_char16 ones  = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}; \\\n\
    vxc_char16 zeros = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; \\\n\
    read_fun(val0, input,  coord, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0)); \\\n\
    result = val0 != zeros ? ones : zeros; \\\n\
    coord.y++; \\\n\
    while(coord.y < axisSize) \\\n\
    { \\\n\
        read_fun(val0, input,  coord, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0)); \\\n\
        val = val0 != zeros ? ones : zeros; \\\n\
        result = result & val; \\\n\
        coord.y++; \\\n\
    } \\\n\
    coord.y = 0; \\\n\
    write_fun(output, coord, result, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
\n\
__kernel void vxcReduceallAxis1_I8toI8\n\
    (\n\
    __read_only  image2d_array_t input,\n\
    __write_only image2d_array_t output,\n\
    int   axisVal\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), 0, get_global_id(1), 0);\n\
    vxc_char16 val0;\n\
    vxc_char16 val, result;\n\
    REDUCEALL_AXIS1_PROCESS(VXC_ReadImage2DArray, VXC_WriteImage2DArray)\n\
}\n\
\n\
__kernel void vxcReduceallAxis1_I8toI8_2D\n\
    (\n\
    __read_only  image2d_array_t input,\n\
    __write_only image2d_array_t output,\n\
    int   axisVal\n\
    )\n\
{\n\
    int2 coord = (int2)(get_global_id(0), 0);\n\
    vxc_char16 val0;\n\
    vxc_char16 val, result;\n\
    REDUCEALL_AXIS1_PROCESS(VXC_ReadImage, VXC_WriteImage)\n\
}\n\
\n\
"; /* end of vsi_nn_kernel_internal_reduceallAxis1_vx*/

static const char vsi_nn_kernel_internal_reduceallAxis2_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform int         axisSize;\n\
\n\
#define REDUCEALL_AXIS2_PROCESS(read_fun, write_fun) \\\n\
    vxc_char16 ones  = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}; \\\n\
    vxc_char16 zeros = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; \\\n\
    read_fun(val0, input,  coord, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0)); \\\n\
    result = val0 != zeros ? ones : zeros; \\\n\
    coord.z++; \\\n\
    while(coord.z < axisSize) \\\n\
    { \\\n\
        read_fun(val0, input,  coord, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0)); \\\n\
        val = val0 != zeros ? ones : zeros; \\\n\
        result = result & val; \\\n\
        coord.z++; \\\n\
    } \\\n\
    coord.z = 0; \\\n\
    write_fun(output, coord, result, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
\n\
__kernel void vxcReduceallAxis2_I8toI8\n\
    (\n\
    __read_only  image2d_array_t input,\n\
    __write_only image2d_array_t output,\n\
    int   axisVal\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), 0, 0);\n\
    vxc_char16 val0;\n\
    vxc_char16 val, result;\n\
    REDUCEALL_AXIS2_PROCESS(VXC_ReadImage2DArray, VXC_WriteImage2DArray)\n\
}\n\
\n\
\n\
"; /* end of vsi_nn_kernel_internal_reduceallAxis2_vx*/

static const char vsi_nn_kernel_internal_reduceanyAxis0_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
_viv_uniform int         axisSize;\n\
\n\
_viv_uniform VXC_512Bits  uniS8AddAll_16x1;\n\
\n\
#define REDUCEANY_AXIS0_PROCESS(read_fun, write_fun) \\\n\
    vxc_char16 ones  = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}; \\\n\
    vxc_char16 zeros = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; \\\n\
    int4 sum_val = 0; \\\n\
    result = zeros; \\\n\
    do \\\n\
    { \\\n\
        read_fun(val0, input,  coord, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0)); \\\n\
        val = val0 != zeros ? ones : zeros; \\\n\
        VXC_DP16x1(sum_val, val, val, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0), uniS8AddAll_16x1); \\\n\
        if (sum_val.x != 0) \\\n\
        { \\\n\
            result = ones; \\\n\
            break; \\\n\
        } \\\n\
        coord.x += 16; \\\n\
    } \\\n\
    while(coord.x < axisSize); \\\n\
    coord.x = 0; \\\n\
    write_fun(output, coord, result, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
\n\
\n\
__kernel void vxcReduceanyAxis0_I8toI8\n\
    (\n\
    __read_only  image2d_array_t input,\n\
    __write_only image2d_array_t output,\n\
    int   axisVal\n\
    )\n\
{\n\
    int4 coord = (int4)(0, get_global_id(0), get_global_id(1), 0);\n\
    vxc_char16 val0;\n\
    vxc_char16 val, result;\n\
    REDUCEANY_AXIS0_PROCESS(VXC_ReadImage2DArray, VXC_WriteImage2DArray)\n\
}\n\
\n\
__kernel void vxcReduceanyAxis0_I8toI8_2D\n\
    (\n\
    __read_only  image2d_array_t input,\n\
    __write_only image2d_array_t output,\n\
    int   axisVal\n\
    )\n\
{\n\
    int2 coord = (int2)(0, get_global_id(0));\n\
    vxc_char16 val0;\n\
    vxc_char16 val, result;\n\
    REDUCEANY_AXIS0_PROCESS(VXC_ReadImage, VXC_WriteImage)\n\
}\n\
\n\
\n\
"; /* end of vsi_nn_kernel_internal_reduceanyAxis0_vx*/

static const char vsi_nn_kernel_internal_reduceanyAxis1_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform int         axisSize;\n\
\n\
#define REDUCEANY_AXIS1_PROCESS(read_fun, write_fun) \\\n\
    vxc_char16 ones  = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}; \\\n\
    vxc_char16 zeros = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; \\\n\
    read_fun(val0, input,  coord, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0)); \\\n\
    result = val0 != zeros ? ones : zeros; \\\n\
    coord.y++; \\\n\
    while(coord.y < axisSize) \\\n\
    { \\\n\
        read_fun(val0, input,  coord, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0)); \\\n\
        val = val0 != zeros ? ones : zeros; \\\n\
        result = result | val; \\\n\
        coord.y++; \\\n\
    } \\\n\
    coord.y = 0; \\\n\
    write_fun(output, coord, result, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
\n\
__kernel void vxcReduceanyAxis1_I8toI8\n\
    (\n\
    __read_only  image2d_array_t input,\n\
    __write_only image2d_array_t output,\n\
    int   axisVal\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), 0, get_global_id(1), 0);\n\
    vxc_char16 val0;\n\
    vxc_char16 val, result;\n\
    REDUCEANY_AXIS1_PROCESS(VXC_ReadImage2DArray, VXC_WriteImage2DArray)\n\
}\n\
\n\
__kernel void vxcReduceanyAxis1_I8toI8_2D\n\
    (\n\
    __read_only  image2d_array_t input,\n\
    __write_only image2d_array_t output,\n\
    int   axisVal\n\
    )\n\
{\n\
    int2 coord = (int2)(get_global_id(0), 0);\n\
    vxc_char16 val0;\n\
    vxc_char16 val, result;\n\
    REDUCEANY_AXIS1_PROCESS(VXC_ReadImage, VXC_WriteImage)\n\
}\n\
\n\
"; /* end of vsi_nn_kernel_internal_reduceanyAxis1_vx*/

static const char vsi_nn_kernel_internal_reduceanyAxis2_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform int         axisSize;\n\
\n\
#define REDUCEANY_AXIS2_PROCESS(read_fun, write_fun) \\\n\
    vxc_char16 ones  = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}; \\\n\
    vxc_char16 zeros = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; \\\n\
    read_fun(val0, input,  coord, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0)); \\\n\
    result = val0 != zeros ? ones : zeros; \\\n\
    coord.z++; \\\n\
    while(coord.z < axisSize) \\\n\
    { \\\n\
        read_fun(val0, input,  coord, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0)); \\\n\
        val = val0 != zeros ? ones : zeros; \\\n\
        result = result | val; \\\n\
        coord.z++; \\\n\
    } \\\n\
    coord.z = 0; \\\n\
    write_fun(output, coord, result, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
\n\
__kernel void vxcReduceanyAxis2_I8toI8\n\
    (\n\
    __read_only  image2d_array_t input,\n\
    __write_only image2d_array_t output,\n\
    int   axisVal\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), 0, 0);\n\
    vxc_char16 val0;\n\
    vxc_char16 val, result;\n\
    REDUCEANY_AXIS2_PROCESS(VXC_ReadImage2DArray, VXC_WriteImage2DArray)\n\
}\n\
\n\
\n\
"; /* end of vsi_nn_kernel_internal_reduceanyAxis2_vx*/

static const char vsi_nn_kernel_internal_reducemaxAxis0_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
_viv_uniform int         axisSize;\n\
_viv_uniform float       outputScale;\n\
_viv_uniform float       output_offset_asymmetric;\n\
_viv_uniform float       inputScale;\n\
_viv_uniform float       input_offset_asymmetric;\n\
_viv_uniform VXC_512Bits uniGetLoData_4x4;\n\
\n\
_viv_uniform VXC_512Bits uniPackMaxData_2x8;\n\
\n\
#define REDUCEMAX_PROCESS_AXIS0(read_fun, vert_max_fun, horz_max_fun) \\\n\
    read_fun(val0, input,  coord, VXC_5BITOFFSET_XY(-16, 0), VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, val, val0, 16); \\\n\
    coord.x += 8; \\\n\
    do \\\n\
    { \\\n\
        read_fun(val0, input,  coord, VXC_5BITOFFSET_XY(-16, 0), VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, img_val0, val0, 16); \\\n\
        read_fun(val1, input,  coord, VXC_5BITOFFSET_XY(-8, 0), VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, img_val1, val1, 16); \\\n\
        read_fun(val2, input,  coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, img_val2, val2, 16); \\\n\
        read_fun(val3, input,  coord, VXC_5BITOFFSET_XY(8, 0), VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, img_val3, val3, 16); \\\n\
        coord.x += 32; \\\n\
        vert_max_fun(val, img_val0, img_val1, val, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        vert_max_fun(val, img_val2, img_val3, val, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        horz_max_fun(val, val, VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0)); \\\n\
        VXC_DP2x8(val, val, val, VXC_MODIFIER(0, 2, 0, VXC_RM_TowardZero, 0), uniPackMaxData_2x8); \\\n\
        horz_max_fun(val, val, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0)); \\\n\
    } \\\n\
    while(coord.x < (axisSize + 16));\n\
\n\
#define REDUCEMAX_PROCESS_AXIS0_SAVE_SAME(save_type, write_fun) \\\n\
    save_type dst; \\\n\
    _viv_asm(COPY, dst, val, 16); \\\n\
    coord.x = 0; \\\n\
    write_fun(output, coord, dst, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
\n\
#define REDUCEMAX_PROCESS_AXIS0_SAVE(dst_type, save_type, conv_mode,\\\n\
                      OUT_SCALE, OUT_OFFSET, IN_SCALE, IN_OFFSET, write_fun) \\\n\
    vxc_float4 prob; \\\n\
    dst_type vec1; \\\n\
    save_type dst; \\\n\
    VXC_DP4x4(prob, val, val, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0), uniGetLoData_4x4); \\\n\
    prob = ((prob - IN_OFFSET) * IN_SCALE) * OUT_SCALE + OUT_OFFSET; \\\n\
    _viv_asm(conv_mode, vec1, prob); \\\n\
    _viv_asm(COPY, dst, vec1, 16); \\\n\
    coord.x = 0; \\\n\
    write_fun(output, coord, dst, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
\n\
#define REDUCEMAX_AXIS0_SAME(src_name, dst_name, src_type, copy_type, save_type, vert_max_fun, horz_max_fun) \\\n\
__kernel void vxcReducemaxAxis0_##src_name##to##dst_name \\\n\
    ( \\\n\
    __read_only  image2d_array_t input, \\\n\
    __write_only image2d_array_t output, \\\n\
    int   axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(16, get_global_id(0), get_global_id(1), 0); \\\n\
    src_type img_val0, img_val1, img_val2, img_val3; \\\n\
    copy_type val0, val1, val2, val3; \\\n\
    src_type val; \\\n\
    REDUCEMAX_PROCESS_AXIS0(VXC_ReadImage2DArray, vert_max_fun, horz_max_fun); \\\n\
    REDUCEMAX_PROCESS_AXIS0_SAVE_SAME(save_type, VXC_WriteImage2DArray); \\\n\
}\n\
\n\
\n\
#define REDUCEMAX_AXIS0(src_name, dst_name, src_type, copy_type, dst_type,\\\n\
              save_type, conv_mode, OUT_SCALE, OUT_OFFSET, IN_SCALE, IN_OFFSET, vert_max_fun, horz_max_fun) \\\n\
__kernel void vxcReducemaxAxis0_##src_name##to##dst_name \\\n\
    ( \\\n\
    __read_only  image2d_array_t input, \\\n\
    __write_only image2d_array_t output, \\\n\
    int   axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(16, get_global_id(0), get_global_id(1), 0); \\\n\
    src_type img_val0, img_val1, img_val2, img_val3; \\\n\
    copy_type val0, val1, val2, val3; \\\n\
    src_type val; \\\n\
    REDUCEMAX_PROCESS_AXIS0(VXC_ReadImage2DArray, vert_max_fun, horz_max_fun); \\\n\
    REDUCEMAX_PROCESS_AXIS0_SAVE(dst_type, save_type, conv_mode,\\\n\
    OUT_SCALE, OUT_OFFSET, IN_SCALE, IN_OFFSET, VXC_WriteImage2DArray); \\\n\
}\n\
\n\
REDUCEMAX_AXIS0_SAME(F16, F16, vxc_half8, vxc_short8,  vxc_short8, VXC_VertMax3_Half, VXC_HorzMax3_Half)\n\
\n\
REDUCEMAX_AXIS0(F16, I16, vxc_half8, vxc_short8, short4, vxc_short8,\\\n\
CONV_SAT_RTE, outputScale, 0, 1, 0, VXC_VertMax3_Half, VXC_HorzMax3_Half)\n\
REDUCEMAX_AXIS0(F16, I8,  vxc_half8, vxc_short8, char4,  vxc_char8, \\\n\
CONV_SAT_RTE, outputScale, 0, 1, 0, VXC_VertMax3_Half, VXC_HorzMax3_Half)\n\
REDUCEMAX_AXIS0(F16, U8,  vxc_half8, vxc_short8, uchar4, vxc_uchar8,\\\n\
                    CONV_SAT_RTE, outputScale, output_offset_asymmetric,\\\n\
                    1, 0, VXC_VertMax3_Half, VXC_HorzMax3_Half)\n\
REDUCEMAX_AXIS0(I16, F16, vxc_short8, vxc_short8, half4,  vxc_short8,\\\n\
CONV, 1, 0, inputScale, 0, VXC_VertMax3_Integer, VXC_HorzMax3_Integer)\n\
REDUCEMAX_AXIS0(I8, F16, vxc_char16, vxc_char16, half4,  vxc_short8,\\\n\
CONV, 1, 0, inputScale, 0, VXC_VertMax3_Integer, VXC_HorzMax3_Integer)\n\
REDUCEMAX_AXIS0(U8, F16, vxc_uchar16, vxc_uchar16, half4,  vxc_short8,\n\
CONV, 1, 0, inputScale, input_offset_asymmetric, VXC_VertMax3_Integer, VXC_HorzMax3_Integer)\n\
\n\
REDUCEMAX_AXIS0(I16, I16, vxc_short8, vxc_short8, short4,\n\
vxc_short8, CONV_SAT_RTE, outputScale, 0, inputScale, 0, VXC_VertMax3_Integer, VXC_HorzMax3_Integer)\n\
REDUCEMAX_AXIS0(I8, I8,  vxc_char16, vxc_char16,  char4, vxc_char8,\n\
CONV_SAT_RTE, outputScale, 0, inputScale, 0, VXC_VertMax3_Integer, VXC_HorzMax3_Integer)\n\
REDUCEMAX_AXIS0(U8, U8,  vxc_uchar16, vxc_uchar16, uchar4, vxc_uchar8, CONV_SAT_RTE,\\\n\
outputScale, output_offset_asymmetric, inputScale,\\\n\
input_offset_asymmetric, VXC_VertMax3_Integer, VXC_HorzMax3_Integer)\n\
\n\
#define REDUCEMAX_AXIS0_SAME_2D(src_name, dst_name, src_type, copy_type, save_type, vert_max_fun, horz_max_fun) \\\n\
__kernel void vxcReducemaxAxis0_##src_name##to##dst_name##_2D \\\n\
    ( \\\n\
    __read_only  image2d_array_t input, \\\n\
    __write_only image2d_array_t output, \\\n\
    int   axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(16, get_global_id(0), get_global_id(1), 0); \\\n\
    src_type img_val0, img_val1, img_val2, img_val3; \\\n\
    copy_type val0, val1, val2, val3; \\\n\
    src_type val; \\\n\
    REDUCEMAX_PROCESS_AXIS0(VXC_ReadImage, vert_max_fun, horz_max_fun); \\\n\
    REDUCEMAX_PROCESS_AXIS0_SAVE_SAME(save_type, VXC_WriteImage); \\\n\
}\n\
\n\
#define REDUCEMAX_AXIS0_2D(src_name, dst_name, src_type, copy_type,\\\n\
                           dst_type, save_type, conv_mode,\\\n\
                           OUT_SCALE, OUT_OFFSET, IN_SCALE, IN_OFFSET, vert_max_fun, horz_max_fun) \\\n\
__kernel void vxcReducemaxAxis0_##src_name##to##dst_name##_2D \\\n\
    ( \\\n\
    __read_only  image2d_array_t input, \\\n\
    __write_only image2d_array_t output, \\\n\
    int   axisVal \\\n\
    ) \\\n\
{ \\\n\
    int2 coord = (int2)(16, get_global_id(0)); \\\n\
    src_type img_val0, img_val1, img_val2, img_val3; \\\n\
    copy_type val0, val1, val2, val3; \\\n\
    src_type val; \\\n\
    REDUCEMAX_PROCESS_AXIS0(VXC_ReadImage, vert_max_fun, horz_max_fun); \\\n\
    REDUCEMAX_PROCESS_AXIS0_SAVE(dst_type, save_type, conv_mode,\\\n\
    OUT_SCALE, OUT_OFFSET, IN_SCALE, IN_OFFSET, VXC_WriteImage); \\\n\
}\n\
\n\
REDUCEMAX_AXIS0_SAME_2D(F16, F16, vxc_half8, vxc_short8,  vxc_short8, VXC_VertMax3_Half, VXC_HorzMax3_Half)\n\
\n\
REDUCEMAX_AXIS0_2D(F16, I16, vxc_half8, vxc_short8, short4, vxc_short8,\\\n\
CONV_SAT_RTE, outputScale, 0, 1, 0, VXC_VertMax3_Half, VXC_HorzMax3_Half)\n\
REDUCEMAX_AXIS0_2D(F16, I8,  vxc_half8, vxc_short8, char4,  vxc_char8,\\\n\
CONV_SAT_RTE, outputScale, 0, 1, 0, VXC_VertMax3_Half, VXC_HorzMax3_Half)\n\
REDUCEMAX_AXIS0_2D(F16, U8,  vxc_half8, vxc_short8, uchar4, vxc_uchar8,\\\n\
                    CONV_SAT_RTE, outputScale, output_offset_asymmetric,\\\n\
                    1, 0, VXC_VertMax3_Half, VXC_HorzMax3_Half)\n\
REDUCEMAX_AXIS0_2D(I16, F16, vxc_short8, vxc_short8, half4,  vxc_short8,\\\n\
CONV, 1, 0, inputScale, 0, VXC_VertMax3_Integer, VXC_HorzMax3_Integer)\n\
REDUCEMAX_AXIS0_2D(I8, F16, vxc_char16, vxc_char16, half4,  vxc_short8,\\\n\
CONV, 1, 0, inputScale, 0, VXC_VertMax3_Integer, VXC_HorzMax3_Integer)\n\
REDUCEMAX_AXIS0_2D(U8, F16, vxc_uchar16, vxc_uchar16, half4,\n\
                   vxc_short8, CONV, 1, 0, inputScale,\\\n\
                   input_offset_asymmetric, VXC_VertMax3_Integer, VXC_HorzMax3_Integer)\n\
REDUCEMAX_AXIS0_2D(I16, I16, vxc_short8, vxc_short8, short4, vxc_short8,\n\
                   CONV_SAT_RTE, outputScale, 0, inputScale, 0, VXC_VertMax3_Integer, VXC_HorzMax3_Integer)\n\
REDUCEMAX_AXIS0_2D(I8, I8,  vxc_char16, vxc_char16,  char4, vxc_char8,\n\
                   CONV_SAT_RTE, outputScale, 0, inputScale, 0, VXC_VertMax3_Integer, VXC_HorzMax3_Integer)\n\
REDUCEMAX_AXIS0_2D(U8, U8,  vxc_uchar16, vxc_uchar16, uchar4, vxc_uchar8, CONV_SAT_RTE,\\\n\
outputScale, output_offset_asymmetric, inputScale,\\\n\
input_offset_asymmetric, VXC_VertMax3_Integer, VXC_HorzMax3_Integer)\n\
\n\
\n\
"; /* end of vsi_nn_kernel_internal_reducemaxAxis0_vx*/

static const char vsi_nn_kernel_internal_reducemaxAxis1_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform int         axisSize;\n\
_viv_uniform float       outputScale;\n\
_viv_uniform float       output_offset_asymmetric;\n\
_viv_uniform float       inputScale;\n\
_viv_uniform float       input_offset_asymmetric;\n\
_viv_uniform VXC_512Bits uniGetLoData_4x4;\n\
\n\
_viv_uniform VXC_512Bits uniConvertInt32toUint8_2x8;\n\
_viv_uniform VXC_512Bits uniGetHiData_4x4;\n\
\n\
#define REDUCEMAX_PROCESS_AXIS1(read_fun, vert_max_fun) \\\n\
    read_fun(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, max, in0, 16); \\\n\
    coord.y++; \\\n\
    do \\\n\
    { \\\n\
        read_fun(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, vec0, in0, 16); \\\n\
        vert_max_fun(max, max, max, vec0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        coord.y++; \\\n\
    } \\\n\
    while(coord.y < axisSize);\n\
\n\
#define REDUCEMAX_PROCESS_AXIS1_SAVE_SAME(save_type, write_fun) \\\n\
    coord.y = 0; \\\n\
    save_type vect; \\\n\
    _viv_asm(COPY, vect, max, 16); \\\n\
    write_fun(output, coord, vect, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
#define REDUCEMAX_PROCESS_AXIS1_SAVE(dst_type, save_type, conv_mode,\\\n\
OUT_SCALE, OUT_OFFSET, IN_SCALE, IN_OFFSET, write_fun) \\\n\
    coord.y = 0; \\\n\
    dst_type dst0, dst1; \\\n\
    save_type vect; \\\n\
    VXC_DP4x4(data0, max, max, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetLoData_4x4); \\\n\
    data0 = ((data0 - IN_OFFSET) * IN_SCALE) * OUT_SCALE + OUT_OFFSET; \\\n\
    _viv_asm(conv_mode, dst0, data0); \\\n\
    VXC_DP4x4(data0, max, max, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetHiData_4x4); \\\n\
    data0 = ((data0 - IN_OFFSET) * IN_SCALE) * OUT_SCALE + OUT_OFFSET; \\\n\
    _viv_asm(conv_mode, dst1, data0); \\\n\
    VXC_DP2x8(vect, dst0, dst1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvertInt32toUint8_2x8); \\\n\
    write_fun(output, coord, vect, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
#define REDUCEMAX_AXIS1_SAME(src_name, dst_name, src_type, copy_type, save_type, vert_max_fun) \\\n\
__kernel void vxcReducemaxAxis1_##src_name##to##dst_name \\\n\
    ( \\\n\
    __read_only  image2d_array_t input, \\\n\
    __write_only image2d_array_t output, \\\n\
    int   axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(get_global_id(0), 0, get_global_id(1), 0); \\\n\
    src_type vec0, max; \\\n\
    copy_type in0; \\\n\
    vxc_float4 data0; \\\n\
    REDUCEMAX_PROCESS_AXIS1(VXC_ReadImage2DArray, vert_max_fun) \\\n\
    REDUCEMAX_PROCESS_AXIS1_SAVE_SAME(save_type, VXC_WriteImage2DArray); \\\n\
}\n\
\n\
#define REDUCEMAX_AXIS1(src_name, dst_name, src_type, copy_type, dst_type, save_type,\\\n\
conv_mode, OUT_SCALE, OUT_OFFSET, IN_SCALE, IN_OFFSET, vert_max_fun) \\\n\
__kernel void vxcReducemaxAxis1_##src_name##to##dst_name \\\n\
    ( \\\n\
    __read_only  image2d_array_t input, \\\n\
    __write_only image2d_array_t output, \\\n\
    int   axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(get_global_id(0), 0, get_global_id(1), 0); \\\n\
    src_type vec0, max; \\\n\
    copy_type in0; \\\n\
    vxc_float4 data0; \\\n\
    REDUCEMAX_PROCESS_AXIS1(VXC_ReadImage2DArray, vert_max_fun) \\\n\
    REDUCEMAX_PROCESS_AXIS1_SAVE(dst_type, save_type, conv_mode,\\\n\
    OUT_SCALE, OUT_OFFSET, IN_SCALE, IN_OFFSET, VXC_WriteImage2DArray); \\\n\
}\n\
\n\
\n\
\n\
REDUCEMAX_AXIS1_SAME(F16, F16, vxc_half8, vxc_short8,  vxc_short8, VXC_VertMax3_Half)\n\
\n\
\n\
REDUCEMAX_AXIS1(F16, I16, vxc_half8, vxc_short8, short4, vxc_short8,\\\n\
CONV_SAT_RTE, outputScale, 0, 1, 0, VXC_VertMax3_Half)\n\
REDUCEMAX_AXIS1(F16, I8,  vxc_half8, vxc_short8, char4,  vxc_char8, \\\n\
CONV_SAT_RTE, outputScale, 0, 1, 0, VXC_VertMax3_Half)\n\
REDUCEMAX_AXIS1(F16, U8,  vxc_half8, vxc_short8, uchar4, vxc_uchar8,\n\
CONV_SAT_RTE, outputScale, output_offset_asymmetric, 1, 0, VXC_VertMax3_Half)\n\
REDUCEMAX_AXIS1(I16, F16, vxc_short8, vxc_short8, half4,  vxc_short8,\\\n\
CONV, 1, 0, inputScale, 0, VXC_VertMax3_Integer)\n\
REDUCEMAX_AXIS1(I8, F16, vxc_char16, vxc_char16, half4,  vxc_short8,\\\n\
CONV, 1, 0, inputScale, 0, VXC_VertMax3_Integer)\n\
REDUCEMAX_AXIS1(U8, F16, vxc_uchar16, vxc_uchar16, half4,  vxc_short8,\n\
CONV, 1, 0, inputScale, input_offset_asymmetric, VXC_VertMax3_Integer)\n\
REDUCEMAX_AXIS1(I16, I16, vxc_short8, vxc_short8, short4, vxc_short8,\n\
CONV_SAT_RTE, outputScale, 0, inputScale, 0, VXC_VertMax3_Integer)\n\
REDUCEMAX_AXIS1(I8, I8,  vxc_char16, vxc_char16,  char4, vxc_char8,\n\
CONV_SAT_RTE, outputScale, 0, inputScale, 0, VXC_VertMax3_Integer)\n\
REDUCEMAX_AXIS1(U8, U8,  vxc_uchar16, vxc_uchar16, uchar4, vxc_uchar8, CONV_SAT_RTE,\\\n\
outputScale, output_offset_asymmetric, inputScale, input_offset_asymmetric, VXC_VertMax3_Integer)\n\
\n\
\n\
#define REDUCEMAX_AXIS1_SAME_2D(src_name, dst_name, src_type, copy_type, save_type, vert_max_fun) \\\n\
__kernel void vxcReducemaxAxis1_##src_name##to##dst_name##_2D \\\n\
    ( \\\n\
    __read_only  image2d_array_t input, \\\n\
    __write_only image2d_array_t output, \\\n\
    int   axisVal \\\n\
    ) \\\n\
{ \\\n\
    int2 coord = (int2)(get_global_id(0), 0); \\\n\
    src_type vec0, max; \\\n\
    copy_type in0; \\\n\
    vxc_float4 data0; \\\n\
    REDUCEMAX_PROCESS_AXIS1(VXC_ReadImage, vert_max_fun) \\\n\
    REDUCEMAX_PROCESS_AXIS1_SAVE_SAME(save_type, VXC_WriteImage); \\\n\
}\n\
\n\
#define REDUCEMAX_AXIS1_2D(src_name, dst_name, src_type, copy_type, dst_type,\\\n\
save_type, conv_mode, OUT_SCALE, OUT_OFFSET, IN_SCALE, IN_OFFSET, vert_max_fun) \\\n\
__kernel void vxcReducemaxAxis1_##src_name##to##dst_name##_2D \\\n\
    ( \\\n\
    __read_only  image2d_array_t input, \\\n\
    __write_only image2d_array_t output, \\\n\
    int   axisVal \\\n\
    ) \\\n\
{ \\\n\
    int2 coord = (int2)(get_global_id(0), 0); \\\n\
    src_type vec0, max; \\\n\
    copy_type in0; \\\n\
    vxc_float4 data0; \\\n\
    REDUCEMAX_PROCESS_AXIS1(VXC_ReadImage, vert_max_fun) \\\n\
    REDUCEMAX_PROCESS_AXIS1_SAVE(dst_type, save_type, conv_mode,\\\n\
    OUT_SCALE, OUT_OFFSET, IN_SCALE, IN_OFFSET, VXC_WriteImage); \\\n\
}\n\
\n\
REDUCEMAX_AXIS1_SAME_2D(F16, F16, vxc_half8, vxc_short8,  vxc_short8, VXC_VertMax3_Half)\n\
\n\
REDUCEMAX_AXIS1_2D(F16, I16, vxc_half8, vxc_short8, short4, vxc_short8,\\\n\
CONV_SAT_RTE, outputScale, 0, 1, 0, VXC_VertMax3_Half)\n\
REDUCEMAX_AXIS1_2D(F16, I8,  vxc_half8, vxc_short8, char4,  vxc_char8, \\\n\
CONV_SAT_RTE, outputScale, 0, 1, 0, VXC_VertMax3_Half)\n\
REDUCEMAX_AXIS1_2D(F16, U8,  vxc_half8, vxc_short8, uchar4, vxc_uchar8,\n\
CONV_SAT_RTE, outputScale, output_offset_asymmetric, 1, 0, VXC_VertMax3_Half)\n\
REDUCEMAX_AXIS1_2D(I16, F16, vxc_short8, vxc_short8, half4,  vxc_short8,\\\n\
CONV, 1, 0, inputScale, 0, VXC_VertMax3_Integer)\n\
REDUCEMAX_AXIS1_2D(I8, F16, vxc_char16, vxc_char16, half4,  vxc_short8,\\\n\
CONV, 1, 0, inputScale, 0, VXC_VertMax3_Integer)\n\
REDUCEMAX_AXIS1_2D(U8, F16, vxc_uchar16, vxc_uchar16, half4,\n\
vxc_short8, CONV, 1, 0, inputScale, input_offset_asymmetric, VXC_VertMax3_Integer)\n\
REDUCEMAX_AXIS1_2D(I16, I16, vxc_short8, vxc_short8,\\\n\
short4, vxc_short8, CONV_SAT_RTE, outputScale, 0, inputScale, 0, VXC_VertMax3_Integer)\n\
REDUCEMAX_AXIS1_2D(I8, I8,  vxc_char16, vxc_char16,\\\n\
char4, vxc_char8, CONV_SAT_RTE, outputScale, 0, inputScale, 0, VXC_VertMax3_Integer)\n\
REDUCEMAX_AXIS1_2D(U8, U8,  vxc_uchar16, vxc_uchar16, uchar4, vxc_uchar8, CONV_SAT_RTE,\\\n\
outputScale, output_offset_asymmetric, inputScale, input_offset_asymmetric, VXC_VertMax3_Integer)\n\
"; /* end of vsi_nn_kernel_internal_reducemaxAxis1_vx*/

static const char vsi_nn_kernel_internal_reducemaxAxis2_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform int         axisSize;\n\
_viv_uniform float       outputScale;\n\
_viv_uniform float       output_offset_asymmetric;\n\
_viv_uniform float       inputScale;\n\
_viv_uniform float       input_offset_asymmetric;\n\
_viv_uniform VXC_512Bits uniGetLoData_4x4;\n\
\n\
_viv_uniform VXC_512Bits uniConvertInt32toUint8_2x8;\n\
_viv_uniform VXC_512Bits uniGetHiData_4x4;\n\
\n\
#define REDUCEMAX_PROCESS_AXIS2(read_fun, vert_max_fun) \\\n\
    read_fun(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, max, in0, 16); \\\n\
    coord.z++; \\\n\
    do \\\n\
    { \\\n\
        read_fun(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, vec0, in0, 16); \\\n\
        vert_max_fun(max, max, max, vec0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        coord.z++; \\\n\
    } \\\n\
    while(coord.z < axisSize);\n\
\n\
#define REDUCEMAX_PROCESS_AXIS2_SAVE_SAME(save_type, write_fun) \\\n\
    coord.z = 0; \\\n\
    save_type vect; \\\n\
    _viv_asm(COPY, vect, max, 16); \\\n\
    write_fun(output, coord, vect, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
#define REDUCEMAX_PROCESS_AXIS2_SAVE(dst_type, save_type, conv_mode,\\\n\
OUT_SCALE, OUT_OFFSET, IN_SCALE, IN_OFFSET, write_fun) \\\n\
    coord.z = 0; \\\n\
    dst_type dst0, dst1; \\\n\
    save_type vect; \\\n\
    VXC_DP4x4(data0, max, max, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetLoData_4x4); \\\n\
    data0 = ((data0 - IN_OFFSET) * IN_SCALE) * OUT_SCALE + OUT_OFFSET; \\\n\
    _viv_asm(conv_mode, dst0, data0); \\\n\
    VXC_DP4x4(data0, max, max, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetHiData_4x4); \\\n\
    data0 = ((data0 - IN_OFFSET) * IN_SCALE) * OUT_SCALE + OUT_OFFSET; \\\n\
    _viv_asm(conv_mode, dst1, data0); \\\n\
    VXC_DP2x8(vect, dst0, dst1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvertInt32toUint8_2x8); \\\n\
    write_fun(output, coord, vect, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
#define REDUCEMAX_AXIS2_SAME(src_name, dst_name, src_type, copy_type, save_type, vert_max_fun) \\\n\
__kernel void vxcReducemaxAxis2_##src_name##to##dst_name \\\n\
    ( \\\n\
    __read_only  image2d_array_t input, \\\n\
    __write_only image2d_array_t output, \\\n\
    int   axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), 0, 0); \\\n\
    src_type vec0, max; \\\n\
    copy_type in0; \\\n\
    vxc_float4 data0; \\\n\
    REDUCEMAX_PROCESS_AXIS2(VXC_ReadImage2DArray, vert_max_fun) \\\n\
    REDUCEMAX_PROCESS_AXIS2_SAVE_SAME(save_type, VXC_WriteImage2DArray); \\\n\
}\n\
\n\
#define REDUCEMAX_AXIS2(src_name, dst_name, src_type, copy_type, dst_type,\\\n\
save_type, conv_mode, OUT_SCALE, OUT_OFFSET, IN_SCALE, IN_OFFSET, vert_max_fun) \\\n\
__kernel void vxcReducemaxAxis2_##src_name##to##dst_name \\\n\
    ( \\\n\
    __read_only  image2d_array_t input, \\\n\
    __write_only image2d_array_t output, \\\n\
    int   axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), 0, 0); \\\n\
    src_type vec0, max; \\\n\
    copy_type in0; \\\n\
    vxc_float4 data0; \\\n\
    REDUCEMAX_PROCESS_AXIS2(VXC_ReadImage2DArray, vert_max_fun) \\\n\
    REDUCEMAX_PROCESS_AXIS2_SAVE(dst_type, save_type, conv_mode, OUT_SCALE,\\\n\
    OUT_OFFSET, IN_SCALE, IN_OFFSET, VXC_WriteImage2DArray); \\\n\
}\n\
\n\
\n\
\n\
REDUCEMAX_AXIS2_SAME(F16, F16, vxc_half8, vxc_short8,  vxc_short8, VXC_VertMax3_Half)\n\
\n\
\n\
REDUCEMAX_AXIS2(F16, I16, vxc_half8, vxc_short8, short4, vxc_short8,\\\n\
CONV_SAT_RTE, outputScale, 0, 1, 0, VXC_VertMax3_Half)\n\
REDUCEMAX_AXIS2(F16, I8,  vxc_half8, vxc_short8, char4,  vxc_char8, \\\n\
CONV_SAT_RTE, outputScale, 0, 1, 0, VXC_VertMax3_Half)\n\
REDUCEMAX_AXIS2(F16, U8,  vxc_half8, vxc_short8, uchar4, vxc_uchar8,\\\n\
CONV_SAT_RTE, outputScale, output_offset_asymmetric, 1, 0, VXC_VertMax3_Half)\n\
REDUCEMAX_AXIS2(I16, F16, vxc_short8, vxc_short8, half4,  vxc_short8,\\\n\
CONV, 1, 0, inputScale, 0, VXC_VertMax3_Integer)\n\
REDUCEMAX_AXIS2(I8, F16, vxc_char16, vxc_char16, half4,  vxc_short8,\\\n\
CONV, 1, 0, inputScale, 0, VXC_VertMax3_Integer)\n\
REDUCEMAX_AXIS2(U8, F16, vxc_uchar16, vxc_uchar16, half4,  vxc_short8,\\\n\
CONV, 1, 0, inputScale, input_offset_asymmetric, VXC_VertMax3_Integer)\n\
\n\
REDUCEMAX_AXIS2(I16, I16, vxc_short8, vxc_short8, short4, vxc_short8,\\\n\
CONV_SAT_RTE, outputScale, 0, inputScale, 0, VXC_VertMax3_Integer)\n\
REDUCEMAX_AXIS2(I8, I8,  vxc_char16, vxc_char16,  char4, vxc_char8,\\\n\
CONV_SAT_RTE, outputScale, 0, inputScale, 0, VXC_VertMax3_Integer)\n\
REDUCEMAX_AXIS2(U8, U8,  vxc_uchar16, vxc_uchar16, uchar4, vxc_uchar8, CONV_SAT_RTE,\\\n\
outputScale, output_offset_asymmetric, inputScale, input_offset_asymmetric, VXC_VertMax3_Integer)\n\
\n\
"; /* end of vsi_nn_kernel_internal_reducemaxAxis2_vx*/

static const char vsi_nn_kernel_internal_reduceminAxis0_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
_viv_uniform int         axisSize;\n\
_viv_uniform float       outputScale;\n\
_viv_uniform float       output_offset_asymmetric;\n\
_viv_uniform float       inputScale;\n\
_viv_uniform float       input_offset_asymmetric;\n\
_viv_uniform VXC_512Bits uniGetLoData_4x4;\n\
\n\
_viv_uniform VXC_512Bits uniPackMaxData_2x8;\n\
\n\
#define REDUCEMIN_PROCESS_AXIS0(read_fun, vert_min_fun, horz_min_fun) \\\n\
    read_fun(val0, input,  coord, VXC_5BITOFFSET_XY(-16, 0), VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, val, val0, 16); \\\n\
    coord.x += 8; \\\n\
    do \\\n\
    { \\\n\
        read_fun(val0, input,  coord, VXC_5BITOFFSET_XY(-16, 0), VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, img_val0, val0, 16); \\\n\
        read_fun(val1, input,  coord, VXC_5BITOFFSET_XY(-8, 0), VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, img_val1, val1, 16); \\\n\
        read_fun(val2, input,  coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, img_val2, val2, 16); \\\n\
        read_fun(val3, input,  coord, VXC_5BITOFFSET_XY(8, 0), VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, img_val3, val3, 16); \\\n\
        coord.x += 32; \\\n\
        vert_min_fun(val, img_val0, img_val1, val, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        vert_min_fun(val, img_val2, img_val3, val, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        horz_min_fun(val, val, VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0)); \\\n\
        VXC_DP2x8(val, val, val, VXC_MODIFIER(0, 2, 0, VXC_RM_TowardZero, 0), uniPackMaxData_2x8); \\\n\
        horz_min_fun(val, val, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0)); \\\n\
    } \\\n\
    while(coord.x < (axisSize + 16));\n\
\n\
#define REDUCEMIN_PROCESS_AXIS0_SAVE_SAME(save_type, write_fun) \\\n\
    save_type dst; \\\n\
    _viv_asm(COPY, dst, val, 16); \\\n\
    coord.x = 0; \\\n\
    write_fun(output, coord, dst, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
\n\
#define REDUCEMIN_PROCESS_AXIS0_SAVE(dst_type, save_type, conv_mode,\\\n\
OUT_SCALE, OUT_OFFSET, IN_SCALE, IN_OFFSET, write_fun) \\\n\
    vxc_float4 prob; \\\n\
    dst_type vec1; \\\n\
    save_type dst; \\\n\
    VXC_DP4x4(prob, val, val, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0), uniGetLoData_4x4); \\\n\
    prob = ((prob - IN_OFFSET) * IN_SCALE) * OUT_SCALE + OUT_OFFSET; \\\n\
    _viv_asm(conv_mode, vec1, prob); \\\n\
    _viv_asm(COPY, dst, vec1, 16); \\\n\
    coord.x = 0; \\\n\
    write_fun(output, coord, dst, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
\n\
#define REDUCEMIN_AXIS0_SAME(src_name, dst_name, src_type, copy_type, save_type, vert_min_fun, horz_min_fun) \\\n\
__kernel void vxcReduceminAxis0_##src_name##to##dst_name \\\n\
    ( \\\n\
    __read_only  image2d_array_t input, \\\n\
    __write_only image2d_array_t output, \\\n\
    int   axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(16, get_global_id(0), get_global_id(1), 0); \\\n\
    src_type img_val0, img_val1, img_val2, img_val3; \\\n\
    copy_type val0, val1, val2, val3; \\\n\
    src_type val; \\\n\
    REDUCEMIN_PROCESS_AXIS0(VXC_ReadImage2DArray, vert_min_fun, horz_min_fun); \\\n\
    REDUCEMIN_PROCESS_AXIS0_SAVE_SAME(save_type, VXC_WriteImage2DArray); \\\n\
}\n\
\n\
\n\
#define REDUCEMIN_AXIS0(src_name, dst_name, src_type, copy_type, dst_type,\\\n\
              save_type, conv_mode, OUT_SCALE, OUT_OFFSET,\\\n\
              IN_SCALE, IN_OFFSET, vert_min_fun, horz_min_fun) \\\n\
__kernel void vxcReduceminAxis0_##src_name##to##dst_name \\\n\
    ( \\\n\
    __read_only  image2d_array_t input, \\\n\
    __write_only image2d_array_t output, \\\n\
    int   axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(16, get_global_id(0), get_global_id(1), 0); \\\n\
    src_type img_val0, img_val1, img_val2, img_val3; \\\n\
    copy_type val0, val1, val2, val3; \\\n\
    src_type val; \\\n\
    REDUCEMIN_PROCESS_AXIS0(VXC_ReadImage2DArray, vert_min_fun, horz_min_fun); \\\n\
    REDUCEMIN_PROCESS_AXIS0_SAVE(dst_type, save_type, conv_mode,\\\n\
    OUT_SCALE, OUT_OFFSET, IN_SCALE, IN_OFFSET, VXC_WriteImage2DArray); \\\n\
}\n\
\n\
REDUCEMIN_AXIS0_SAME(F16, F16, vxc_half8, vxc_short8,  vxc_short8, VXC_VertMin3_Half, VXC_HorzMin3_Half)\n\
\n\
REDUCEMIN_AXIS0(F16, I16, vxc_half8, vxc_short8, short4, vxc_short8, CONV_SAT_RTE, outputScale,\\\n\
0, 1, 0, VXC_VertMin3_Half, VXC_HorzMin3_Half)\n\
REDUCEMIN_AXIS0(F16, I8,  vxc_half8, vxc_short8, char4,  vxc_char8,  CONV_SAT_RTE, outputScale,\\\n\
0, 1, 0, VXC_VertMin3_Half, VXC_HorzMin3_Half)\n\
REDUCEMIN_AXIS0(F16, U8,  vxc_half8, vxc_short8, uchar4, vxc_uchar8,\\\n\
 CONV_SAT_RTE, outputScale, output_offset_asymmetric, 1, 0, VXC_VertMin3_Half, VXC_HorzMin3_Half)\n\
REDUCEMIN_AXIS0(I16, F16, vxc_short8, vxc_short8, half4,  vxc_short8, CONV, 1, 0,\\\n\
inputScale, 0, VXC_VertMin3_Integer, VXC_HorzMin3_Integer)\n\
REDUCEMIN_AXIS0(I8, F16, vxc_char16, vxc_char16, half4,  vxc_short8, CONV, 1, 0,\\\n\
inputScale, 0, VXC_VertMin3_Integer, VXC_HorzMin3_Integer)\n\
REDUCEMIN_AXIS0(U8, F16, vxc_uchar16, vxc_uchar16, half4,  vxc_short8, CONV,\\\n\
1, 0, inputScale, input_offset_asymmetric, VXC_VertMin3_Integer, VXC_HorzMin3_Integer)\n\
\n\
REDUCEMIN_AXIS0(I16, I16, vxc_short8, vxc_short8, short4, vxc_short8,\\\n\
CONV_SAT_RTE, outputScale, 0, inputScale, 0, VXC_VertMin3_Integer, VXC_HorzMin3_Integer)\n\
REDUCEMIN_AXIS0(I8, I8,  vxc_char16, vxc_char16,  char4, vxc_char8,\\\n\
CONV_SAT_RTE, outputScale, 0, inputScale, 0, VXC_VertMin3_Integer, VXC_HorzMin3_Integer)\n\
REDUCEMIN_AXIS0(U8, U8,  vxc_uchar16, vxc_uchar16, uchar4, vxc_uchar8, CONV_SAT_RTE,\\\n\
outputScale, output_offset_asymmetric, inputScale, input_offset_asymmetric,\\\n\
VXC_VertMin3_Integer, VXC_HorzMin3_Integer)\n\
\n\
#define REDUCEMIN_AXIS0_SAME_2D(src_name, dst_name, src_type, copy_type,\\\n\
save_type, vert_min_fun, horz_min_fun) \\\n\
__kernel void vxcReduceminAxis0_##src_name##to##dst_name##_2D \\\n\
    ( \\\n\
    __read_only  image2d_array_t input, \\\n\
    __write_only image2d_array_t output, \\\n\
    int   axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(16, get_global_id(0), get_global_id(1), 0); \\\n\
    src_type img_val0, img_val1, img_val2, img_val3; \\\n\
    copy_type val0, val1, val2, val3; \\\n\
    src_type val; \\\n\
    REDUCEMIN_PROCESS_AXIS0(VXC_ReadImage, vert_min_fun, horz_min_fun); \\\n\
    REDUCEMIN_PROCESS_AXIS0_SAVE_SAME(save_type, VXC_WriteImage); \\\n\
}\n\
\n\
#define REDUCEMIN_AXIS0_2D(src_name, dst_name, src_type, copy_type,\\\n\
                           dst_type, save_type, conv_mode, OUT_SCALE,\\\n\
                           OUT_OFFSET, IN_SCALE, IN_OFFSET, vert_min_fun, horz_min_fun) \\\n\
__kernel void vxcReduceminAxis0_##src_name##to##dst_name##_2D \\\n\
    ( \\\n\
    __read_only  image2d_array_t input, \\\n\
    __write_only image2d_array_t output, \\\n\
    int   axisVal \\\n\
    ) \\\n\
{ \\\n\
    int2 coord = (int2)(16, get_global_id(0)); \\\n\
    src_type img_val0, img_val1, img_val2, img_val3; \\\n\
    copy_type val0, val1, val2, val3; \\\n\
    src_type val; \\\n\
    REDUCEMIN_PROCESS_AXIS0(VXC_ReadImage, vert_min_fun, horz_min_fun); \\\n\
    REDUCEMIN_PROCESS_AXIS0_SAVE(dst_type, save_type, conv_mode,\\\n\
    OUT_SCALE, OUT_OFFSET, IN_SCALE, IN_OFFSET, VXC_WriteImage); \\\n\
}\n\
\n\
REDUCEMIN_AXIS0_SAME_2D(F16, F16, vxc_half8, vxc_short8,  vxc_short8, VXC_VertMin3_Half, VXC_HorzMin3_Half)\n\
\n\
REDUCEMIN_AXIS0_2D(F16, I16, vxc_half8, vxc_short8, short4, vxc_short8,\\\n\
CONV_SAT_RTE, outputScale, 0, 1, 0, VXC_VertMin3_Half, VXC_HorzMin3_Half)\n\
REDUCEMIN_AXIS0_2D(F16, I8,  vxc_half8, vxc_short8, char4,  vxc_char8,\\\n\
CONV_SAT_RTE, outputScale, 0, 1, 0, VXC_VertMin3_Half, VXC_HorzMin3_Half)\n\
REDUCEMIN_AXIS0_2D(F16, U8,  vxc_half8, vxc_short8, uchar4, vxc_uchar8,\\\n\
CONV_SAT_RTE, outputScale, output_offset_asymmetric, 1, 0, VXC_VertMin3_Half, VXC_HorzMin3_Half)\n\
\n\
REDUCEMIN_AXIS0_2D(I16, F16, vxc_short8, vxc_short8, half4,  vxc_short8,\\\n\
CONV, 1, 0, inputScale, 0, VXC_VertMin3_Integer, VXC_HorzMin3_Integer)\n\
REDUCEMIN_AXIS0_2D(I8, F16, vxc_char16, vxc_char16, half4,  vxc_short8,\\\n\
CONV, 1, 0, inputScale, 0, VXC_VertMin3_Integer, VXC_HorzMin3_Integer)\n\
REDUCEMIN_AXIS0_2D(U8, F16, vxc_uchar16, vxc_uchar16, half4,  vxc_short8,\\\n\
CONV, 1, 0, inputScale, input_offset_asymmetric, VXC_VertMin3_Integer, VXC_HorzMin3_Integer)\n\
REDUCEMIN_AXIS0_2D(I16, I16, vxc_short8, vxc_short8, short4, vxc_short8,\\\n\
CONV_SAT_RTE, outputScale, 0, inputScale, 0, VXC_VertMin3_Integer, VXC_HorzMin3_Integer)\n\
REDUCEMIN_AXIS0_2D(I8, I8,  vxc_char16, vxc_char16,  char4, vxc_char8,\\\n\
CONV_SAT_RTE, outputScale, 0, inputScale, 0, VXC_VertMin3_Integer, VXC_HorzMin3_Integer)\n\
REDUCEMIN_AXIS0_2D(U8, U8,  vxc_uchar16, vxc_uchar16, uchar4, vxc_uchar8, CONV_SAT_RTE,\\\n\
outputScale, output_offset_asymmetric, inputScale, input_offset_asymmetric,\\\n\
VXC_VertMin3_Integer, VXC_HorzMin3_Integer)\n\
\n\
\n\
"; /* end of vsi_nn_kernel_internal_reduceminAxis0_vx*/

static const char vsi_nn_kernel_internal_reduceminAxis1_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform int         axisSize;\n\
_viv_uniform float       outputScale;\n\
_viv_uniform float       output_offset_asymmetric;\n\
_viv_uniform float       inputScale;\n\
_viv_uniform float       input_offset_asymmetric;\n\
_viv_uniform VXC_512Bits uniGetLoData_4x4;\n\
\n\
_viv_uniform VXC_512Bits uniConvertInt32toUint8_2x8;\n\
_viv_uniform VXC_512Bits uniGetHiData_4x4;\n\
\n\
#define REDUCEMIN_PROCESS_AXIS1(read_fun, vert_min_fun) \\\n\
    read_fun(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, min, in0, 16); \\\n\
    coord.y++; \\\n\
    do \\\n\
    { \\\n\
        read_fun(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, vec0, in0, 16); \\\n\
        vert_min_fun(min, min, min, vec0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        coord.y++; \\\n\
    } \\\n\
    while(coord.y < axisSize);\n\
\n\
#define REDUCEMIN_PROCESS_AXIS1_SAVE_SAME(save_type, write_fun) \\\n\
    coord.y = 0; \\\n\
    save_type vect; \\\n\
    _viv_asm(COPY, vect, min, 16); \\\n\
    write_fun(output, coord, vect, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
#define REDUCEMIN_PROCESS_AXIS1_SAVE(dst_type, save_type, conv_mode,\\\n\
OUT_SCALE, OUT_OFFSET, IN_SCALE, IN_OFFSET, write_fun) \\\n\
    coord.y = 0; \\\n\
    dst_type dst0, dst1; \\\n\
    save_type vect; \\\n\
    VXC_DP4x4(data0, min, min, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetLoData_4x4); \\\n\
    data0 = ((data0 - IN_OFFSET) * IN_SCALE) * OUT_SCALE + OUT_OFFSET; \\\n\
    _viv_asm(conv_mode, dst0, data0); \\\n\
    VXC_DP4x4(data0, min, min, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetHiData_4x4); \\\n\
    data0 = ((data0 - IN_OFFSET) * IN_SCALE) * OUT_SCALE + OUT_OFFSET; \\\n\
    _viv_asm(conv_mode, dst1, data0); \\\n\
    VXC_DP2x8(vect, dst0, dst1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvertInt32toUint8_2x8); \\\n\
    write_fun(output, coord, vect, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
#define REDUCEMIN_AXIS1_SAME(src_name, dst_name, src_type, copy_type, save_type, vert_min_fun) \\\n\
__kernel void vxcReduceminAxis1_##src_name##to##dst_name \\\n\
    ( \\\n\
    __read_only  image2d_array_t input, \\\n\
    __write_only image2d_array_t output, \\\n\
    int   axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(get_global_id(0), 0, get_global_id(1), 0); \\\n\
    src_type vec0, min; \\\n\
    copy_type in0; \\\n\
    vxc_float4 data0; \\\n\
    REDUCEMIN_PROCESS_AXIS1(VXC_ReadImage2DArray, vert_min_fun) \\\n\
    REDUCEMIN_PROCESS_AXIS1_SAVE_SAME(save_type, VXC_WriteImage2DArray); \\\n\
}\n\
\n\
#define REDUCEMIN_AXIS1(src_name, dst_name, src_type, copy_type, dst_type, save_type,\\\n\
conv_mode, OUT_SCALE, OUT_OFFSET, IN_SCALE, IN_OFFSET, vert_min_fun) \\\n\
__kernel void vxcReduceminAxis1_##src_name##to##dst_name \\\n\
    ( \\\n\
    __read_only  image2d_array_t input, \\\n\
    __write_only image2d_array_t output, \\\n\
    int   axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(get_global_id(0), 0, get_global_id(1), 0); \\\n\
    src_type vec0, min; \\\n\
    copy_type in0; \\\n\
    vxc_float4 data0; \\\n\
    REDUCEMIN_PROCESS_AXIS1(VXC_ReadImage2DArray, vert_min_fun) \\\n\
    REDUCEMIN_PROCESS_AXIS1_SAVE(dst_type, save_type, conv_mode,\\\n\
    OUT_SCALE, OUT_OFFSET, IN_SCALE, IN_OFFSET, VXC_WriteImage2DArray); \\\n\
}\n\
\n\
\n\
\n\
REDUCEMIN_AXIS1_SAME(F16, F16, vxc_half8, vxc_short8,  vxc_short8, VXC_VertMin3_Half)\n\
\n\
\n\
REDUCEMIN_AXIS1(F16, I16, vxc_half8, vxc_short8, short4, vxc_short8,\\\n\
CONV_SAT_RTE, outputScale, 0, 1, 0, VXC_VertMin3_Half)\n\
REDUCEMIN_AXIS1(F16, I8,  vxc_half8, vxc_short8, char4,  vxc_char8,\\\n\
CONV_SAT_RTE, outputScale, 0, 1, 0, VXC_VertMin3_Half)\n\
REDUCEMIN_AXIS1(F16, U8,  vxc_half8, vxc_short8, uchar4, vxc_uchar8,\\\n\
CONV_SAT_RTE, outputScale, output_offset_asymmetric, 1, 0, VXC_VertMin3_Half)\n\
REDUCEMIN_AXIS1(I16, F16, vxc_short8, vxc_short8, half4,  vxc_short8,\\\n\
CONV, 1, 0, inputScale, 0, VXC_VertMin3_Integer)\n\
REDUCEMIN_AXIS1(I8, F16, vxc_char16, vxc_char16, half4,  vxc_short8,\\\n\
CONV, 1, 0, inputScale, 0, VXC_VertMin3_Integer)\n\
REDUCEMIN_AXIS1(U8, F16, vxc_uchar16, vxc_uchar16, half4,  vxc_short8,\\\n\
CONV, 1, 0, inputScale, input_offset_asymmetric, VXC_VertMin3_Integer)\n\
REDUCEMIN_AXIS1(I16, I16, vxc_short8, vxc_short8, short4,\\\n\
vxc_short8, CONV_SAT_RTE, outputScale, 0, inputScale, 0, VXC_VertMin3_Integer)\n\
REDUCEMIN_AXIS1(I8, I8,  vxc_char16, vxc_char16,  char4,\\\n\
vxc_char8, CONV_SAT_RTE, outputScale, 0, inputScale, 0, VXC_VertMin3_Integer)\n\
REDUCEMIN_AXIS1(U8, U8,  vxc_uchar16, vxc_uchar16, uchar4, vxc_uchar8, CONV_SAT_RTE,\\\n\
outputScale, output_offset_asymmetric, inputScale, input_offset_asymmetric, VXC_VertMin3_Integer)\n\
\n\
\n\
#define REDUCEMIN_AXIS1_SAME_2D(src_name, dst_name, src_type, copy_type, save_type, vert_min_fun) \\\n\
__kernel void vxcReduceminAxis1_##src_name##to##dst_name##_2D \\\n\
    ( \\\n\
    __read_only  image2d_array_t input, \\\n\
    __write_only image2d_array_t output, \\\n\
    int   axisVal \\\n\
    ) \\\n\
{ \\\n\
    int2 coord = (int2)(get_global_id(0), 0); \\\n\
    src_type vec0, min; \\\n\
    copy_type in0; \\\n\
    vxc_float4 data0; \\\n\
    REDUCEMIN_PROCESS_AXIS1(VXC_ReadImage, vert_min_fun) \\\n\
    REDUCEMIN_PROCESS_AXIS1_SAVE_SAME(save_type, VXC_WriteImage); \\\n\
}\n\
\n\
#define REDUCEMIN_AXIS1_2D(src_name, dst_name, src_type, copy_type, dst_type, save_type,\\\n\
conv_mode, OUT_SCALE, OUT_OFFSET, IN_SCALE, IN_OFFSET, vert_min_fun) \\\n\
__kernel void vxcReduceminAxis1_##src_name##to##dst_name##_2D \\\n\
    ( \\\n\
    __read_only  image2d_array_t input, \\\n\
    __write_only image2d_array_t output, \\\n\
    int   axisVal \\\n\
    ) \\\n\
{ \\\n\
    int2 coord = (int2)(get_global_id(0), 0); \\\n\
    src_type vec0, min; \\\n\
    copy_type in0; \\\n\
    vxc_float4 data0; \\\n\
    REDUCEMIN_PROCESS_AXIS1(VXC_ReadImage, vert_min_fun) \\\n\
    REDUCEMIN_PROCESS_AXIS1_SAVE(dst_type, save_type, conv_mode,\\\n\
    OUT_SCALE, OUT_OFFSET, IN_SCALE, IN_OFFSET, VXC_WriteImage); \\\n\
}\n\
\n\
REDUCEMIN_AXIS1_SAME_2D(F16, F16, vxc_half8, vxc_short8,  vxc_short8, VXC_VertMin3_Half)\n\
\n\
REDUCEMIN_AXIS1_2D(F16, I16, vxc_half8, vxc_short8, short4, vxc_short8,\\\n\
CONV_SAT_RTE, outputScale, 0, 1, 0, VXC_VertMin3_Half)\n\
REDUCEMIN_AXIS1_2D(F16, I8,  vxc_half8, vxc_short8, char4,  vxc_char8,\\\n\
CONV_SAT_RTE, outputScale, 0, 1, 0, VXC_VertMin3_Half)\n\
REDUCEMIN_AXIS1_2D(F16, U8,  vxc_half8, vxc_short8, uchar4, vxc_uchar8,\\\n\
CONV_SAT_RTE, outputScale, output_offset_asymmetric, 1, 0, VXC_VertMin3_Half)\n\
REDUCEMIN_AXIS1_2D(I16, F16, vxc_short8, vxc_short8, half4,  vxc_short8,\\\n\
CONV, 1, 0, inputScale, 0, VXC_VertMin3_Integer)\n\
REDUCEMIN_AXIS1_2D(I8, F16, vxc_char16, vxc_char16, half4,  vxc_short8,\\\n\
CONV, 1, 0, inputScale, 0, VXC_VertMin3_Integer)\n\
REDUCEMIN_AXIS1_2D(U8, F16, vxc_uchar16, vxc_uchar16, half4,  vxc_short8,\\\n\
CONV, 1, 0, inputScale, input_offset_asymmetric, VXC_VertMin3_Integer)\n\
REDUCEMIN_AXIS1_2D(I16, I16, vxc_short8, vxc_short8, short4, vxc_short8,\\\n\
CONV_SAT_RTE, outputScale, 0, inputScale, 0, VXC_VertMin3_Integer)\n\
REDUCEMIN_AXIS1_2D(I8, I8,  vxc_char16, vxc_char16,  char4, vxc_char8,\\\n\
CONV_SAT_RTE, outputScale, 0, inputScale, 0, VXC_VertMin3_Integer)\n\
REDUCEMIN_AXIS1_2D(U8, U8,  vxc_uchar16, vxc_uchar16, uchar4, vxc_uchar8, CONV_SAT_RTE,\\\n\
outputScale, output_offset_asymmetric, inputScale, input_offset_asymmetric, VXC_VertMin3_Integer)\n\
"; /* end of vsi_nn_kernel_internal_reduceminAxis1_vx*/

static const char vsi_nn_kernel_internal_reduceminAxis2_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform int         axisSize;\n\
_viv_uniform float       outputScale;\n\
_viv_uniform float       output_offset_asymmetric;\n\
_viv_uniform float       inputScale;\n\
_viv_uniform float       input_offset_asymmetric;\n\
_viv_uniform VXC_512Bits uniGetLoData_4x4;\n\
\n\
_viv_uniform VXC_512Bits uniConvertInt32toUint8_2x8;\n\
_viv_uniform VXC_512Bits uniGetHiData_4x4;\n\
\n\
#define REDUCEMIN_PROCESS_AXIS2(read_fun, vert_min_fun) \\\n\
    read_fun(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, min, in0, 16); \\\n\
    coord.z++; \\\n\
    do \\\n\
    { \\\n\
        read_fun(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, vec0, in0, 16); \\\n\
        vert_min_fun(min, min, min, vec0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        coord.z++; \\\n\
    } \\\n\
    while(coord.z < axisSize);\n\
\n\
#define REDUCEMIN_PROCESS_AXIS2_SAVE_SAME(save_type, write_fun) \\\n\
    coord.z = 0; \\\n\
    save_type vect; \\\n\
    _viv_asm(COPY, vect, min, 16); \\\n\
    write_fun(output, coord, vect, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
#define REDUCEMIN_PROCESS_AXIS2_SAVE(dst_type, save_type, conv_mode,\\\n\
OUT_SCALE, OUT_OFFSET, IN_SCALE, IN_OFFSET, write_fun) \\\n\
    coord.z = 0; \\\n\
    dst_type dst0, dst1; \\\n\
    save_type vect; \\\n\
    VXC_DP4x4(data0, min, min, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetLoData_4x4); \\\n\
    data0 = ((data0 - IN_OFFSET) * IN_SCALE) * OUT_SCALE + OUT_OFFSET; \\\n\
    _viv_asm(conv_mode, dst0, data0); \\\n\
    VXC_DP4x4(data0, min, min, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetHiData_4x4); \\\n\
    data0 = ((data0 - IN_OFFSET) * IN_SCALE) * OUT_SCALE + OUT_OFFSET; \\\n\
    _viv_asm(conv_mode, dst1, data0); \\\n\
    VXC_DP2x8(vect, dst0, dst1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvertInt32toUint8_2x8); \\\n\
    write_fun(output, coord, vect, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
#define REDUCEMIN_AXIS2_SAME(src_name, dst_name, src_type, copy_type, save_type, vert_min_fun) \\\n\
__kernel void vxcReduceminAxis2_##src_name##to##dst_name \\\n\
    ( \\\n\
    __read_only  image2d_array_t input, \\\n\
    __write_only image2d_array_t output, \\\n\
    int   axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), 0, 0); \\\n\
    src_type vec0, min; \\\n\
    copy_type in0; \\\n\
    vxc_float4 data0; \\\n\
    REDUCEMIN_PROCESS_AXIS2(VXC_ReadImage2DArray, vert_min_fun) \\\n\
    REDUCEMIN_PROCESS_AXIS2_SAVE_SAME(save_type, VXC_WriteImage2DArray); \\\n\
}\n\
\n\
#define REDUCEMIN_AXIS2(src_name, dst_name, src_type, copy_type, dst_type,\\\n\
save_type, conv_mode, OUT_SCALE, OUT_OFFSET, IN_SCALE, IN_OFFSET, vert_min_fun) \\\n\
__kernel void vxcReduceminAxis2_##src_name##to##dst_name \\\n\
    ( \\\n\
    __read_only  image2d_array_t input, \\\n\
    __write_only image2d_array_t output, \\\n\
    int   axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), 0, 0); \\\n\
    src_type vec0, min; \\\n\
    copy_type in0; \\\n\
    vxc_float4 data0; \\\n\
    REDUCEMIN_PROCESS_AXIS2(VXC_ReadImage2DArray, vert_min_fun) \\\n\
    REDUCEMIN_PROCESS_AXIS2_SAVE(dst_type, save_type, conv_mode,\\\n\
    OUT_SCALE, OUT_OFFSET, IN_SCALE, IN_OFFSET, VXC_WriteImage2DArray); \\\n\
}\n\
\n\
\n\
\n\
REDUCEMIN_AXIS2_SAME(F16, F16, vxc_half8, vxc_short8,  vxc_short8, VXC_VertMin3_Half)\n\
\n\
\n\
REDUCEMIN_AXIS2(F16, I16, vxc_half8, vxc_short8, short4, vxc_short8,\\\n\
CONV_SAT_RTE, outputScale, 0, 1, 0, VXC_VertMin3_Half)\n\
REDUCEMIN_AXIS2(F16, I8,  vxc_half8, vxc_short8, char4,  vxc_char8,\\\n\
CONV_SAT_RTE, outputScale, 0, 1, 0, VXC_VertMin3_Half)\n\
REDUCEMIN_AXIS2(F16, U8,  vxc_half8, vxc_short8, uchar4, vxc_uchar8,\\\n\
CONV_SAT_RTE, outputScale, output_offset_asymmetric, 1, 0, VXC_VertMin3_Half)\n\
REDUCEMIN_AXIS2(I16, F16, vxc_short8, vxc_short8, half4,  vxc_short8,\\\n\
CONV, 1, 0, inputScale, 0, VXC_VertMin3_Integer)\n\
REDUCEMIN_AXIS2(I8, F16, vxc_char16, vxc_char16, half4,  vxc_short8,\\\n\
CONV, 1, 0, inputScale, 0, VXC_VertMin3_Integer)\n\
REDUCEMIN_AXIS2(U8, F16, vxc_uchar16, vxc_uchar16, half4,  vxc_short8,\\\n\
CONV, 1, 0, inputScale, input_offset_asymmetric, VXC_VertMin3_Integer)\n\
\n\
REDUCEMIN_AXIS2(I16, I16, vxc_short8, vxc_short8, short4, vxc_short8,\\\n\
CONV_SAT_RTE, outputScale, 0, inputScale, 0, VXC_VertMin3_Integer)\n\
REDUCEMIN_AXIS2(I8, I8,  vxc_char16, vxc_char16,  char4, vxc_char8,\\\n\
CONV_SAT_RTE, outputScale, 0, inputScale, 0, VXC_VertMin3_Integer)\n\
REDUCEMIN_AXIS2(U8, U8,  vxc_uchar16, vxc_uchar16, uchar4, vxc_uchar8, CONV_SAT_RTE,\\\n\
outputScale, output_offset_asymmetric, inputScale, input_offset_asymmetric, VXC_VertMin3_Integer)\n\
\n\
"; /* end of vsi_nn_kernel_internal_reduceminAxis2_vx*/

static const char vsi_nn_kernel_l2normalizescale_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
/********************************************L2NormalizeScale*****************************************/\n\
_viv_uniform int L2NorS_depth;\n\
_viv_uniform VXC_512Bits UniFp16MulLo_dp4x4;\n\
_viv_uniform VXC_512Bits UniFp16MulHi_dp4x4;\n\
__kernel void vxcL2NormScale_SumRsqrt\n\
    (\n\
    image2d_array_t input,\n\
    image2d_array_t output,\n\
    int dim\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0);\n\
    vxc_short8 img1_s16, img2_s16;\n\
    vxc_float4 squr, sum_lo = 0, sum_hi = 0;\n\
    vxc_half8 img1_fp16, img2_fp16;\n\
    half4 val1_h, val2_h;\n\
    for(int i = 0; i < L2NorS_depth; i += 2)\n\
    {\n\
        VXC_ReadImage(img1_s16, input, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage(img2_s16, input, coord.xy, VXC_5BITOFFSET_XY(0, 1),\\\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        coord.y += 2;\n\
        _viv_asm(COPY, img1_fp16, img1_s16, 16);\n\
        _viv_asm(COPY, img2_fp16, img2_s16, 16);\n\
        _viv_asm(COPY, img1_fp16, img1_s16, 16);\n\
        _viv_asm(COPY, img2_fp16, img2_s16, 16);\n\
\n\
        VXC_DP4x4(squr, img1_fp16, img1_fp16, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1),\\\n\
            UniFp16MulLo_dp4x4);\n\
        sum_lo += squr;\n\
        VXC_DP4x4(squr, img2_fp16, img2_fp16, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1),\\\n\
            UniFp16MulLo_dp4x4);\n\
        sum_lo += squr;\n\
        VXC_DP4x4(squr, img1_fp16, img1_fp16, VXC_MODIFIER(0, 3, 4, VXC_RM_TowardZero, 1),\\\n\
            UniFp16MulHi_dp4x4);\n\
        sum_hi += squr;\n\
        VXC_DP4x4(squr, img2_fp16, img2_fp16, VXC_MODIFIER(0, 3, 4, VXC_RM_TowardZero, 1),\\\n\
            UniFp16MulHi_dp4x4);\n\
        sum_hi += squr;\n\
    }\n\
    sum_lo = rsqrt(sum_lo);\n\
    sum_hi = rsqrt(sum_hi);\n\
    write_imagef(output, coord.zwww, sum_lo);\n\
    coord.z += 4;\n\
    write_imagef(output, coord.zwww, sum_hi);\n\
}\n\
//int8 version\n\
_viv_uniform float r_inputScale;\n\
_viv_uniform VXC_512Bits uniIntegerSquareLo_4x4;\n\
_viv_uniform VXC_512Bits uniIntegerSquareHi_4x4;\n\
_viv_uniform VXC_512Bits uniDataSquareAddU32Lo_4x4;\n\
_viv_uniform VXC_512Bits uniDataSquareAddU32Hi_4x4;\n\
__kernel void vxcL2NormScale_SumRsqrt_int8\n\
    (\n\
    image2d_array_t input,\n\
    image2d_array_t output,\n\
    int dim\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0);\n\
    vxc_char8 src0, src1;\n\
    vxc_uint4 dst0 = 0, dst1 = 0;\n\
    for(int i = 0; i < L2NorS_depth; i += 2)\n\
    {\n\
        VXC_ReadImage(src0, input, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage(src1, input, coord.xy, VXC_5BITOFFSET_XY(0, 1),\\\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        coord.y += 2;\n\
        VXC_DP4x4(dst0, src0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
            uniDataSquareAddU32Lo_4x4);\n\
        VXC_DP4x4(dst1, src0, dst1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
            uniDataSquareAddU32Hi_4x4);\n\
        VXC_DP4x4(dst0, src1, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
            uniDataSquareAddU32Lo_4x4);\n\
        VXC_DP4x4(dst1, src1, dst1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
            uniDataSquareAddU32Hi_4x4);\n\
    }\n\
    vxc_float4 sum_lo, sum_hi;\n\
    sum_lo = convert_float4(dst0);\n\
    sum_hi = convert_float4(dst1);\n\
    sum_lo = rsqrt(sum_lo) * r_inputScale;\n\
    sum_hi = rsqrt(sum_hi) * r_inputScale;\n\
    write_imagef(output, coord.zwww, sum_lo);\n\
    coord.z += 4;\n\
    write_imagef(output, coord.zwww, sum_hi);\n\
}\n\
__kernel void vxcL2NormScale_SumRsqrt_int16\n\
    (\n\
    image2d_array_t input,\n\
    image2d_array_t output,\n\
    int dim\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0);\n\
    vxc_short8 src0, src1;\n\
    vxc_float4 squr, sum_lo = 0, sum_hi = 0;\n\
    for(int i = 0; i < L2NorS_depth; i += 2)\n\
    {\n\
        VXC_ReadImage(src0, input, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage(src1, input, coord.xy, VXC_5BITOFFSET_XY(0, 1),\\\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        coord.y += 2;\n\
        VXC_DP4x4(squr, src0, src0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
            uniIntegerSquareLo_4x4);\n\
        sum_lo = squr + sum_lo;\n\
        VXC_DP4x4(squr, src0, src0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
            uniIntegerSquareHi_4x4);\n\
        sum_hi = squr + sum_hi;\n\
        VXC_DP4x4(squr, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
            uniIntegerSquareLo_4x4);\n\
        sum_lo = squr + sum_lo;\n\
        VXC_DP4x4(squr, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
            uniIntegerSquareHi_4x4);\n\
        sum_hi = squr + sum_hi;\n\
    }\n\
    sum_lo = rsqrt(sum_lo) * r_inputScale;\n\
    sum_hi = rsqrt(sum_hi) * r_inputScale;\n\
    write_imagef(output, coord.zwww, sum_lo);\n\
    coord.z += 4;\n\
    write_imagef(output, coord.zwww, sum_hi);\n\
}\n\
_viv_uniform VXC_512Bits uniUInt8SquareLo_4x4;\n\
_viv_uniform VXC_512Bits uniUInt8SquareHi_4x4;\n\
_viv_uniform int inputZP;\n\
__kernel void vxcL2NormScale_SumRsqrt_uint8\n\
    (\n\
    image2d_array_t input,\n\
    image2d_array_t output,\n\
    int dim\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0);\n\
    vxc_uchar8 src0, src1;\n\
    vxc_float4 squr, sum_lo = 0, sum_hi = 0;\n\
    for(int i = 0; i < L2NorS_depth; i += 2)\n\
    {\n\
        vxc_uchar8 zero;\n\
        VXC_ReadImage(src0, input, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage(src1, input, coord.xy, VXC_5BITOFFSET_XY(0, 1),\\\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        coord.y += 2;\n\
        _viv_asm(COPY, zero, inputZP, 4);\n\
        VXC_DP4x4(squr, src0, zero, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniUInt8SquareLo_4x4);\n\
        sum_lo = squr + sum_lo;\n\
        VXC_DP4x4(squr, src0, zero, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniUInt8SquareHi_4x4);\n\
        sum_hi = squr + sum_hi;\n\
        VXC_DP4x4(squr, src1, zero, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniUInt8SquareLo_4x4);\n\
        sum_lo = squr + sum_lo;\n\
        VXC_DP4x4(squr, src1, zero, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniUInt8SquareHi_4x4);\n\
        sum_hi = squr + sum_hi;\n\
    }\n\
    sum_lo = rsqrt(sum_lo) * r_inputScale;\n\
    sum_hi = rsqrt(sum_hi) * r_inputScale;\n\
    write_imagef(output, coord.zwww, sum_lo);\n\
    coord.z += 4;\n\
    write_imagef(output, coord.zwww, sum_hi);\n\
}\n\
\n\
/****************************L2NormalizeMulScale**********************************/\n\
\n\
_viv_uniform VXC_512Bits uniDataSubZPtoFp32Part0_4x4;\n\
_viv_uniform VXC_512Bits uniDataSubZPtoFp32Part1_4x4;\n\
_viv_uniform VXC_512Bits uniExtact8Bin_2x8;\n\
_viv_uniform VXC_512Bits uniFp16toFp32_4x4;\n\
_viv_uniform float IntergerScale;\n\
_viv_uniform float output_ZP;\n\
#define L2NORMSCALE_MIXED_MODE(name0, name1, input_type, incopy_type,\\\n\
    output_type, convert_type, copy_type) \\\n\
    __kernel void vxcL2NormScale_##name0##to##name1\\\n\
    (\\\n\
    __read_only  image2d_array_t input1,\\\n\
    __read_only  image2d_array_t input2,\\\n\
    __read_only  image2d_array_t scale,\\\n\
    __write_only image2d_array_t output,\\\n\
    int dim\\\n\
    )\\\n\
{\\\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0);\\\n\
    input_type  vect0, vect1;\\\n\
    incopy_type src0, src1;\\\n\
    vxc_float4 rsqrt0, rsqrt1;\\\n\
    rsqrt0 = read_imagef(input2, coord.zwww);\\\n\
    coord.z += 4;\\\n\
    rsqrt1 = read_imagef(input2, coord.zwww);\\\n\
    rsqrt0 *= IntergerScale;\\\n\
    rsqrt1 *= IntergerScale;\\\n\
    for(int i = 0; i < L2NorS_depth; i += 2)\\\n\
   {\\\n\
        vxc_float4 vec0, vec1;\\\n\
        input_type input_ZP ;\\\n\
        convert_type dst0, dst1;\\\n\
        VXC_ReadImage(vect0, input1, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\\\n\
        _viv_asm(COPY, src0, vect0, 16); \\\n\
        VXC_ReadImage(vect1, input1, coord.xy, VXC_5BITOFFSET_XY(0, 1),\\\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\\\n\
        _viv_asm(COPY, src1, vect1, 16); \\\n\
        vxc_short8 scale_s16;\\\n\
        vxc_half8  scale_f16;\\\n\
        vxc_float4 scale_f32;\\\n\
        VXC_ReadImage(scale_s16, scale, coord.yw, VXC_5BITOFFSET_XY(0, 0),\\\n\
            VXC_MODIFIER(0, 1, 0, VXC_RM_TowardZero, 0));\\\n\
        _viv_asm(COPY, scale_f16, scale_s16, 16); \\\n\
        _viv_asm(COPY, input_ZP, inputZP, 4); \\\n\
        VXC_DP4x4(vec0, src0, input_ZP, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0),\\\n\
            uniDataSubZPtoFp32Part0_4x4);\\\n\
        VXC_DP4x4(vec1, src0, input_ZP, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0),\\\n\
            uniDataSubZPtoFp32Part1_4x4);\\\n\
        VXC_DP4x4(scale_f32, scale_f16, scale_f16, VXC_MODIFIER(0, 1, 0, VXC_RM_TowardInf, 0),\\\n\
            uniFp16toFp32_4x4);\\\n\
        vec0 = vec0 * rsqrt0 + output_ZP;\\\n\
        vec1 = vec1 * rsqrt1 + output_ZP;\\\n\
        vec0 *= scale_f32.xxxx;\\\n\
        vec1 *= scale_f32.xxxx;\\\n\
        _viv_asm(CONV_RTE, dst0, vec0);\\\n\
        _viv_asm(CONV_RTE, dst1, vec1);\\\n\
        output_type dst2;\\\n\
        VXC_DP2x8(dst2, dst0, dst1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniExtact8Bin_2x8);\\\n\
        copy_type dst;\\\n\
        _viv_asm(COPY, dst, dst2, 16); \\\n\
        VXC_WriteImage(output, coord.xy, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\\\n\
        VXC_DP4x4(vec0, src1, input_ZP, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0),\\\n\
            uniDataSubZPtoFp32Part0_4x4);\\\n\
        VXC_DP4x4(vec1, src1, input_ZP, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0),\\\n\
            uniDataSubZPtoFp32Part1_4x4);\\\n\
        vec0 = vec0 * rsqrt0 + output_ZP;\\\n\
        vec1 = vec1 * rsqrt1 + output_ZP;\\\n\
        vec0 *= scale_f32.yyyy;\\\n\
        vec1 *= scale_f32.yyyy;\\\n\
        _viv_asm(CONV_RTE, dst0, vec0);\\\n\
        _viv_asm(CONV_RTE, dst1, vec1);\\\n\
        VXC_DP2x8(dst2, dst0, dst1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniExtact8Bin_2x8);\\\n\
        coord.y++;\\\n\
        _viv_asm(COPY, dst, dst2, 16); \\\n\
        VXC_WriteImage(output, coord.xy, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\\\n\
        coord.y++;\\\n\
    }\\\n\
}\n\
//                     name0, name1, input_type,  incopy_type, output_type, convert_type, copy_type\n\
L2NORMSCALE_MIXED_MODE(Fp16,  Fp16,  vxc_short8,  vxc_half8,   vxc_half8,   half4,        vxc_short8)\n\
L2NORMSCALE_MIXED_MODE(Int8,  Int8,  vxc_char16,  vxc_char16,  vxc_char16,  int4,         vxc_char16)\n\
L2NORMSCALE_MIXED_MODE(Int8,  Fp16,  vxc_char16,  vxc_char16,  vxc_half8,   half4,        vxc_short8)\n\
L2NORMSCALE_MIXED_MODE(UInt8, Fp16,  vxc_uchar16, vxc_uchar16, vxc_half8,   half4,        vxc_short8)\n\
L2NORMSCALE_MIXED_MODE(UInt8, UInt8, vxc_uchar16, vxc_uchar16, vxc_uchar16, int4,         vxc_uchar16)\n\
L2NORMSCALE_MIXED_MODE(Int16, Int16, vxc_short8,  vxc_short8,  vxc_short8,  int4,         vxc_short8)\n\
L2NORMSCALE_MIXED_MODE(Int16, Fp16,  vxc_short8,  vxc_short8,  vxc_half8,   half4,        vxc_short8)\n\
//L2NORMSCALE_MIXED_MODE(Fp16,  Int8,  vxc_short8,  vxc_half8,   vxc_char16,  int4,         vxc_char16)\n\
//L2NORMSCALE_MIXED_MODE(Fp16,  UInt8, vxc_short8,  vxc_half8,   vxc_uchar16, int4,         vxc_uchar16)\n\
//L2NORMSCALE_MIXED_MODE(Fp16,  Int16, vxc_short8,  vxc_half8,   vxc_short8,  int4,         vxc_short8)\n\
"; /* end of vsi_nn_kernel_l2normalizescale_vx*/

static const char vsi_nn_kernel_layernormalize_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
/**************************layernorm float16***********************************/\n\
_viv_uniform int width;\n\
_viv_uniform float dimRatio;\n\
_viv_uniform VXC_512Bits uniFp16SumSqr_dp8x2;\n\
_viv_uniform VXC_512Bits UniFP16toFP32Lo4_dp4x4;\n\
_viv_uniform VXC_512Bits uniExtractHalf4_dp4x4;\n\
\n\
__kernel void vxcLayerNorm(\n\
    image2d_array_t input,\n\
    image2d_array_t bias,\n\
    image2d_array_t scale,\n\
    image2d_array_t output,\n\
              float eps)\n\
{\n\
    int4 coord = (int4)(0, get_global_id(1), 0, 0);\n\
    vxc_short8 src0, src1;\n\
    vxc_float sum = 0, sqr = 0;\n\
    VXC_ReadImage(src0, input, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    for(coord.x = 8; coord.x < (width+8); coord.x += 8)\n\
    {\n\
        vxc_half8  val0_h;\n\
        _viv_asm(COPY, val0_h, src0, 16);\n\
        VXC_ReadImage(src0, input, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        vxc_float4 sumsqr;\n\
        VXC_DP8x2(sumsqr, val0_h, val0_h, VXC_MODIFIER(0, 1, 0, VXC_RM_TowardZero, 0),\\\n\
            uniFp16SumSqr_dp8x2);\n\
        sum += sumsqr.x;\n\
        sqr += sumsqr.y;\n\
    }\n\
    vxc_float mean;\n\
    mean = sum * dimRatio;\n\
    vxc_float vari;\n\
    vari = sqr*dimRatio - mean*mean;\n\
    vari += eps;\n\
    vari = rsqrt(vari);\n\
    vxc_float4 bias_f;\n\
    for(coord.x = 0; coord.x < width; coord.x += 4)\n\
    {\n\
        VXC_ReadImage(src0, input, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage(src1, scale, coord.xw, VXC_5BITOFFSET_XY(0, 0),\\\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        bias_f = read_imagef(bias, coord.xwww);\n\
        vxc_half8 in_h, scale_h;\n\
        _viv_asm(COPY, in_h, src0, 16);\n\
        _viv_asm(COPY, scale_h, src1, 16);\n\
        vxc_float4 in_f, scale_f;\n\
        VXC_DP4x4(in_f, in_h, in_h, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
            UniFP16toFP32Lo4_dp4x4);\n\
        VXC_DP4x4(scale_f, scale_h, scale_h, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
            UniFP16toFP32Lo4_dp4x4);\n\
        vxc_float4 sub, norm;\n\
        sub = in_f - mean;\n\
        norm = scale_f * vari * sub + bias_f;\n\
        half4 norm_h;\n\
        _viv_asm(CONV, norm_h, norm);\n\
        vxc_half8 dst;\n\
        VXC_DP4x4(dst, norm_h, norm_h, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
            uniExtractHalf4_dp4x4);\n\
        vxc_short8 dstval;\n\
        _viv_asm(COPY, dstval, dst, 16);\n\
        VXC_WriteImage(output, coord.xy, dstval, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    }\n\
}\n\
/*****************************layernorm uint8 to uint8****************************/\n\
_viv_uniform VXC_512Bits uniConvert1stUint8SubZpToFp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvert2ndUint8SubZpToFp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvert3rdUint8SubZpToFp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvert4thUint8SubZpToFp32_4x4;\n\
_viv_uniform VXC_512Bits uniSumU8_16x1;\n\
_viv_uniform VXC_512Bits uniSqrSum_16x1;\n\
_viv_uniform float input_scale;\n\
_viv_uniform int inputZP;\n\
_viv_uniform float outputScale;\n\
_viv_uniform int output_ZP;\n\
_viv_uniform int sumInZp;\n\
_viv_uniform int tmpZp1;\n\
_viv_uniform int tmpZp2;\n\
_viv_uniform float e2InScale;\n\
_viv_uniform VXC_512Bits uniConvertSecFp16Fp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertInt32toUint8_2x8;\n\
\n\
__kernel void vxcLayerNorm_u8(\n\
    image2d_array_t input,\n\
    image2d_array_t bias,\n\
    image2d_array_t scale,\n\
    image2d_array_t output,\n\
              float eps)\n\
{\n\
    int4 coord = (int4)(0, get_global_id(1), 0, 0);\n\
    vxc_uchar16 src0, src2;\n\
    vxc_short8 src1;\n\
    vxc_half8 scale_h;\n\
    float sum = 0, sqr = 0;\n\
    vxc_float4 bias_f0, bias_f1, scale_f0, scale_f1;\n\
    int tmpSum = 0, tmpSqr = 0;\n\
    vxc_int4 tmpSum1;\n\
    vxc_int4 tmpSqr1;\n\
    short zp = inputZP;\n\
\n\
    for(coord.x = 0; coord.x < width; coord.x += 16)\n\
    {\n\
        VXC_ReadImage(src0, input, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
            VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
        VXC_DP16x1(tmpSum1, src0, src0, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0), uniSumU8_16x1);\n\
        tmpSum += (tmpSum1.x);\n\
        VXC_DP16x1(tmpSqr1, src0, src0, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0), uniSqrSum_16x1);\n\
        tmpSqr += (tmpSqr1.x + tmpZp1 * tmpSum1.x);\n\
    }\n\
    sum = (tmpSum + sumInZp) * input_scale;\n\
    sqr = (tmpSqr + tmpZp2) * e2InScale;\n\
\n\
    float mean, vari;\n\
    mean = sum * dimRatio;\n\
    vari = sqr*dimRatio - mean*mean;\n\
    vari += eps;\n\
    vari = rsqrt(vari);\n\
    vxc_int4 tmpVal0, tmpVal1;\n\
    vxc_float4  tmpData0, tmpData1, tmpData2, tmpData3;\n\
    int4 coord_bias = (int4)(0, 0, 0, 0);\n\
\n\
    for(coord.x = 0; coord.x < width; coord.x += 16)\n\
    {\n\
        coord_bias.x = coord.x;\n\
        VXC_ReadImage(src0, input, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
            VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage(src1, scale, coord.xw, VXC_5BITOFFSET_XY(0, 0),\\\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        _viv_asm(COPY, scale_h, src1, 16);\n\
        VXC_DP4x4(scale_f0, scale_h, scale_h, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
            UniFP16toFP32Lo4_dp4x4);\n\
        VXC_DP4x4(scale_f1, scale_h, scale_h, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
            uniConvertSecFp16Fp32_4x4);\n\
        bias_f0 = read_imagef(bias, coord_bias);\n\
        coord_bias.x += 4;\n\
        bias_f1 = read_imagef(bias, coord_bias);\n\
        coord_bias.x += 4;\n\
\n\
        VXC_ReadImage(src1, scale, coord.xw, VXC_5BITOFFSET_XY(8, 0),\\\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        _viv_asm(COPY, scale_h, src1, 16);\n\
        VXC_DP4x4(tmpData0, src0, zp, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
            uniConvert1stUint8SubZpToFp32_4x4);\n\
        VXC_DP4x4(tmpData1, src0, zp, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
            uniConvert2ndUint8SubZpToFp32_4x4);\n\
        VXC_DP4x4(tmpData2, src0, zp, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
            uniConvert3rdUint8SubZpToFp32_4x4);\n\
        VXC_DP4x4(tmpData3, src0, zp, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
            uniConvert4thUint8SubZpToFp32_4x4);\n\
        tmpData0 *= input_scale;\n\
        tmpData1 *= input_scale;\n\
        tmpData2 *= input_scale;\n\
        tmpData3 *= input_scale;\n\
\n\
        vxc_float4 norm;\n\
        tmpData0 -= mean;\n\
        norm = scale_f0 * vari * tmpData0 + bias_f0;\n\
        bias_f0 = read_imagef(bias, coord_bias);\n\
        VXC_DP4x4(scale_f0, scale_h, scale_h, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
            UniFP16toFP32Lo4_dp4x4);\n\
        coord_bias.x += 4;\n\
        tmpVal0 = convert_int4_rte(norm * outputScale + output_ZP);\n\
\n\
        tmpData1 -= mean;\n\
        norm = scale_f1 * vari * tmpData1 + bias_f1;\n\
        bias_f1 = read_imagef(bias, coord_bias);\n\
        VXC_DP4x4(scale_f1, scale_h, scale_h, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
            uniConvertSecFp16Fp32_4x4);\n\
        tmpVal1 = convert_int4_rte(norm * outputScale + output_ZP);\n\
        VXC_DP2x8(src2, tmpVal0, tmpVal1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1),\\\n\
            uniConvertInt32toUint8_2x8);\n\
\n\
        tmpData2 -= mean;\n\
        norm = scale_f0 * vari * tmpData2 + bias_f0;\n\
        tmpVal0 = convert_int4_rte(norm * outputScale + output_ZP);\n\
\n\
        tmpData3 -= mean;\n\
        norm = scale_f1 * vari * tmpData3 + bias_f1;\n\
        tmpVal1 = convert_int4_rte(norm * outputScale + output_ZP);\n\
        VXC_DP2x8(src2, tmpVal0, tmpVal1, VXC_MODIFIER(8, 15, 0, VXC_RM_TowardZero, 1),\\\n\
            uniConvertInt32toUint8_2x8);\n\
        VXC_WriteImage(output, coord.xy, src2, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    }\n\
}\n\
/***************************layernorm float16 to uint8**************************/\n\
_viv_uniform float outputZP;\n\
__kernel void vxcLayerNormFP16toU8(\n\
    image2d_array_t input,\n\
    image2d_array_t bias,\n\
    image2d_array_t scale,\n\
    image2d_array_t output,\n\
              float eps)\n\
{\n\
    int4 coord = (int4)(0, get_global_id(1), 0, 0);\n\
    vxc_short8 src0, src1;\n\
    vxc_float sum = 0, sqr = 0;\n\
    VXC_ReadImage(src0, input, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    for(coord.x = 8; coord.x < (width+8); coord.x += 8)\n\
    {\n\
        vxc_half8  val0_h;\n\
        _viv_asm(COPY, val0_h, src0, 16);\n\
        VXC_ReadImage(src0, input, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        vxc_float4 sumsqr;\n\
        VXC_DP8x2(sumsqr, val0_h, val0_h, VXC_MODIFIER(0, 1, 0, VXC_RM_TowardZero, 0),\\\n\
            uniFp16SumSqr_dp8x2);\n\
        sum += sumsqr.x;\n\
        sqr += sumsqr.y;\n\
    }\n\
    vxc_float mean;\n\
    mean = sum * dimRatio;\n\
    vxc_float vari;\n\
    vari = sqr*dimRatio - mean*mean;\n\
    vari += eps;\n\
    vari = rsqrt(vari);\n\
    vxc_float4 bias_f;\n\
    for(coord.x = 0; coord.x < width; coord.x += 4)\n\
    {\n\
        VXC_ReadImage(src0, input, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage(src1, scale, coord.xw, VXC_5BITOFFSET_XY(0, 0),\\\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        bias_f = read_imagef(bias, coord.xwww);\n\
        vxc_half8 in_h, scale_h;\n\
        _viv_asm(COPY, in_h, src0, 16);\n\
        _viv_asm(COPY, scale_h, src1, 16);\n\
        vxc_float4 in_f, scale_f;\n\
        VXC_DP4x4(in_f, in_h, in_h, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
            UniFP16toFP32Lo4_dp4x4);\n\
        VXC_DP4x4(scale_f, scale_h, scale_h, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
            UniFP16toFP32Lo4_dp4x4);\n\
        vxc_float4 sub, norm;\n\
        sub = in_f - mean;\n\
        norm = scale_f * vari * sub + bias_f;\n\
        norm = norm * outputScale + outputZP;\n\
        int4 output_int4;\n\
        output_int4 = convert_int4_rte(norm);\n\
        vxc_uchar8 dst;\n\
        VXC_DP2x8(dst, output_int4, output_int4, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1),\n\
            uniConvertInt32toUint8_2x8);\n\
        VXC_WriteImage(output, coord.xy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    }\n\
}"; /* end of vsi_nn_kernel_layernormalize_vx*/

static const char vsi_nn_kernel_log_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform float rlogE;\n\
float4 log_(float4 x)\n\
{\n\
    x = log2(x);\n\
    return x * rlogE;\n\
}\n\
_viv_uniform float inputScale;\n\
_viv_uniform float inputTail;\n\
_viv_uniform float outputScale;\n\
_viv_uniform float outputZP;\n\
_viv_uniform VXC_512Bits uniExtract8Data_2x8;\n\
_viv_uniform VXC_512Bits uniDatatoFp32Part0_4x4;\n\
_viv_uniform VXC_512Bits uniDatatoFp32Part1_4x4;\n\
\n\
#define TENSOR_TRANSCENDENTAL(funcName, src_type_name, dst_type_name, src_type, \\\n\
                src_copy_type, convert_type, dst_type, dst_copy_type) \\\n\
    __kernel void vxTensor##funcName##_##src_type_name##to##dst_type_name( \\\n\
    __read_only  image2d_array_t  input, \\\n\
    __write_only image2d_array_t  output \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0); \\\n\
    src_type      src0; \\\n\
    src_copy_type src1; \\\n\
    VXC_ReadImage2DArray(src0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src1, src0, 16); \\\n\
 \\\n\
    float4 vecA; \\\n\
    float4 vecB; \\\n\
    VXC_DP4x4(vecA, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniDatatoFp32Part0_4x4); \\\n\
    VXC_DP4x4(vecB, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniDatatoFp32Part1_4x4); \\\n\
    vecA = vecA * inputScale + inputTail; \\\n\
    vecB = vecB * inputScale + inputTail; \\\n\
    vecA = funcName##_(vecA); \\\n\
    vecB = funcName##_(vecB); \\\n\
    vecA = vecA * outputScale + outputZP; \\\n\
    vecB = vecB * outputScale + outputZP; \\\n\
 \\\n\
    convert_type dst0, dst1; \\\n\
    _viv_asm(CONV_RTE, dst0, vecA); \\\n\
    _viv_asm(CONV_RTE, dst1, vecB); \\\n\
    dst_type dst2; \\\n\
    VXC_DP2x8(dst2, dst0, dst1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1), uniExtract8Data_2x8); \\\n\
    dst_copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst2, 16); \\\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
//LOG\n\
TENSOR_TRANSCENDENTAL(Log, F16, F16, vxc_short8, vxc_half8,  half4, vxc_half8,  vxc_short8)\n\
TENSOR_TRANSCENDENTAL(Log, F16, I8,  vxc_short8, vxc_half8,  int4,  vxc_char8,  vxc_char8)\n\
TENSOR_TRANSCENDENTAL(Log, F16, U8,  vxc_short8, vxc_half8,  int4,  vxc_uchar8, vxc_uchar8)\n\
TENSOR_TRANSCENDENTAL(Log, F16, I16, vxc_short8, vxc_half8,  int4,  vxc_short8, vxc_short8)\n\
TENSOR_TRANSCENDENTAL(Log, I8,  I8,  vxc_char8,  vxc_char8,  int4,  vxc_char8,  vxc_char8)\n\
TENSOR_TRANSCENDENTAL(Log, I8,  F16, vxc_char8,  vxc_char8,  half4, vxc_half8,  vxc_short8)\n\
TENSOR_TRANSCENDENTAL(Log, U8,  U8,  vxc_uchar8, vxc_uchar8, int4,  vxc_uchar8, vxc_uchar8)\n\
TENSOR_TRANSCENDENTAL(Log, U8,  F16, vxc_uchar8, vxc_uchar8, half4, vxc_half8,  vxc_short8)\n\
TENSOR_TRANSCENDENTAL(Log, I16, I16, vxc_short8, vxc_short8, int4,  vxc_short8, vxc_short8)\n\
TENSOR_TRANSCENDENTAL(Log, I16, F16, vxc_short8, vxc_short8, half4, vxc_half8,  vxc_short8)\n\
\n\
#define TENSOR_TRANSCENDENTAL_2D(funcName, src_type_name, dst_type_name, src_type, \\\n\
        src_copy_type, convert_type, dst_type, dst_copy_type) \\\n\
    __kernel void vxTensor##funcName##_##src_type_name##to##dst_type_name##_2D( \\\n\
    __read_only  image2d_array_t  input, \\\n\
    __write_only image2d_array_t  output \\\n\
    ) \\\n\
{ \\\n\
    int2 coord = (int2)(get_global_id(0), get_global_id(1)); \\\n\
    src_type      src0; \\\n\
    src_copy_type src1; \\\n\
    VXC_ReadImage(src0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src1, src0, 16); \\\n\
 \\\n\
    float4 vecA; \\\n\
    float4 vecB; \\\n\
    VXC_DP4x4(vecA, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniDatatoFp32Part0_4x4); \\\n\
    VXC_DP4x4(vecB, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniDatatoFp32Part1_4x4); \\\n\
    vecA = vecA * inputScale + inputTail; \\\n\
    vecB = vecB * inputScale + inputTail; \\\n\
    vecA = funcName##_(vecA); \\\n\
    vecB = funcName##_(vecB); \\\n\
    vecA = vecA * outputScale + outputZP; \\\n\
    vecB = vecB * outputScale + outputZP; \\\n\
 \\\n\
    convert_type dst0, dst1; \\\n\
    _viv_asm(CONV_RTE, dst0, vecA); \\\n\
    _viv_asm(CONV_RTE, dst1, vecB); \\\n\
    dst_type dst2; \\\n\
    VXC_DP2x8(dst2, dst0, dst1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1), uniExtract8Data_2x8); \\\n\
    dst_copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst2, 16); \\\n\
    VXC_WriteImage(output, coord, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
//Log\n\
TENSOR_TRANSCENDENTAL_2D(Log, F16, F16, vxc_short8, vxc_half8, half4, vxc_half8,  vxc_short8)\n\
TENSOR_TRANSCENDENTAL_2D(Log, F16, I8,  vxc_short8, vxc_half8, int4,  vxc_char8,  vxc_char8)\n\
TENSOR_TRANSCENDENTAL_2D(Log, F16, U8,  vxc_short8, vxc_half8, int4,  vxc_uchar8, vxc_uchar8)\n\
TENSOR_TRANSCENDENTAL_2D(Log, F16, I16, vxc_short8, vxc_half8, int4,  vxc_short8, vxc_short8)\n\
TENSOR_TRANSCENDENTAL_2D(Log, I8,  I8,  vxc_char8,  vxc_char8,  int4,  vxc_char8,  vxc_char8)\n\
TENSOR_TRANSCENDENTAL_2D(Log, I8,  F16, vxc_char8,  vxc_char8,  half4, vxc_half8,  vxc_short8)\n\
TENSOR_TRANSCENDENTAL_2D(Log, U8,  U8,  vxc_uchar8, vxc_uchar8, int4,  vxc_uchar8, vxc_uchar8)\n\
TENSOR_TRANSCENDENTAL_2D(Log, U8,  F16, vxc_uchar8, vxc_uchar8, half4, vxc_half8,  vxc_short8)\n\
TENSOR_TRANSCENDENTAL_2D(Log, I16, I16, vxc_short8, vxc_short8, int4,  vxc_short8, vxc_short8)\n\
TENSOR_TRANSCENDENTAL_2D(Log, I16, F16, vxc_short8, vxc_short8, half4, vxc_half8,  vxc_short8)\n\
"; /* end of vsi_nn_kernel_log_vx*/

static const char vsi_nn_kernel_log_softmaxAxis0_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
_viv_uniform float       rlogE;\n\
_viv_uniform int         axisSize;\n\
_viv_uniform float       betaValue;\n\
_viv_uniform float       scaleLogE;\n\
_viv_uniform float       outputScale;\n\
_viv_uniform float       output_offset_asymmetric;\n\
_viv_uniform VXC_512Bits uniConvertInt32toUint8_2x8;\n\
\n\
_viv_uniform int         inputWidth;\n\
_viv_uniform int         inputWidthRemain4;\n\
_viv_uniform VXC_512Bits uniGetSubData0to3_4x4;\n\
_viv_uniform VXC_512Bits uniGetSubData4to7_4x4;\n\
_viv_uniform VXC_512Bits uniPackMaxData_2x8;\n\
\n\
#define LOGSOFTMAX_PROCESS_AXIS0(read_fun, vert_max_fun, horz_max_fun) \\\n\
    read_fun(val0, input,  coord, VXC_5BITOFFSET_XY(-16, 0), VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, val, val0, 16); \\\n\
    coord.x += 8; \\\n\
    do \\\n\
    { \\\n\
        read_fun(val0, input,  coord, VXC_5BITOFFSET_XY(-16, 0), VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, img_val0, val0, 16); \\\n\
        read_fun(val1, input,  coord, VXC_5BITOFFSET_XY(-8, 0), VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, img_val1, val1, 16); \\\n\
        read_fun(val2, input,  coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, img_val2, val2, 16); \\\n\
        read_fun(val3, input,  coord, VXC_5BITOFFSET_XY(8, 0), VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, img_val3, val3, 16); \\\n\
        coord.x += 32; \\\n\
        vert_max_fun(val, img_val0, img_val1, val, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        vert_max_fun(val, img_val2, img_val3, val, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        horz_max_fun(val, val, VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0)); \\\n\
        VXC_DP2x8(val, val, val, VXC_MODIFIER(0, 2, 0, VXC_RM_TowardZero, 0), uniPackMaxData_2x8); \\\n\
        horz_max_fun(val, val, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0)); \\\n\
    } \\\n\
    while(coord.x < (axisSize + 16)); \\\n\
    vxc_float4 prob; \\\n\
    float fProbSum = 0; \\\n\
    const float4 one4 = (float4)(1.0, 1.0, 1.0, 1.0); \\\n\
    int idx = 0; \\\n\
    for (coord.x = 0; coord.x < inputWidth; idx ++) \\\n\
    { \\\n\
        read_fun(val0, input,  coord, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, img_val0, val0, 16); \\\n\
        VXC_DP4x4(prob, img_val0, val, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetSubData0to3_4x4); \\\n\
        prob *= scaleLogE; \\\n\
        prob = exp2(prob); \\\n\
        fProbSum += dot(prob, one4); \\\n\
        coord.x += 4; \\\n\
    } \\\n\
    read_fun(val0, input,  coord, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, img_val0, val0, 16); \\\n\
    VXC_DP4x4(prob, img_val0, val, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetSubData0to3_4x4); \\\n\
    prob *= scaleLogE; \\\n\
    if(inputWidthRemain4 == 1) \\\n\
    { \\\n\
        prob.x = exp2(prob.x); \\\n\
        prob.yzw = 0; \\\n\
        fProbSum += dot(prob, one4); \\\n\
    } \\\n\
    else if(inputWidthRemain4 == 2) \\\n\
    { \\\n\
        prob.x = exp2(prob.x); \\\n\
        prob.y = exp2(prob.y); \\\n\
        prob.zw = 0; \\\n\
        fProbSum += dot(prob, one4); \\\n\
    } \\\n\
    else if(inputWidthRemain4 == 3) \\\n\
    { \\\n\
        prob.x = exp2(prob.x); \\\n\
        prob.y = exp2(prob.y); \\\n\
        prob.z = exp2(prob.z); \\\n\
        prob.w = 0; \\\n\
        fProbSum += dot(prob, one4); \\\n\
    } \\\n\
    vxc_float4 probSum_log; \\\n\
    probSum_log.x = log2(fProbSum) * rlogE;\n\
\n\
#define LOGSOFTMAX_PROCESS_AXIS0_SAVE(dst_type, save_type, conv_mode, OUT_SCALE, OUT_OFFSET, read_fun, write_fun) \\\n\
    for (coord.x = 0; coord.x < axisSize; ) \\\n\
    { \\\n\
        dst_type vec0, vec1; \\\n\
        save_type dst; \\\n\
        read_fun(val0, input,  coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, img_val0, val0, 16); \\\n\
        VXC_DP4x4(prob, img_val0, val, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetSubData0to3_4x4); \\\n\
        prob = prob * betaValue - probSum_log.xxxx; \\\n\
        prob = prob * OUT_SCALE + OUT_OFFSET; \\\n\
        _viv_asm(conv_mode, vec0, prob); \\\n\
        VXC_DP4x4(prob, img_val0, val, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetSubData4to7_4x4); \\\n\
        prob = prob * betaValue - probSum_log.xxxx; \\\n\
        prob = prob * OUT_SCALE + OUT_OFFSET; \\\n\
        _viv_asm(conv_mode, vec1, prob); \\\n\
        VXC_DP2x8(dst, vec0, vec1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvertInt32toUint8_2x8); \\\n\
        write_fun(output, coord, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        coord.x += 8; \\\n\
    }\n\
\n\
#define LOGSOFTMAX_AXIS0(src_name, dst_name, src_type, copy_type, dst_type,\\\n\
                        save_type, conv_mode, OUT_SCALE, OUT_OFFSET, vert_max_fun, horz_max_fun) \\\n\
__kernel void vxcLogSoftmaxAxis0_##src_name##to##dst_name \\\n\
    ( \\\n\
    __read_only  image2d_array_t input, \\\n\
    __write_only image2d_array_t output, \\\n\
    float input_Scale, \\\n\
    int   axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(16, get_global_id(0), get_global_id(1), 0); \\\n\
    src_type img_val0, img_val1, img_val2, img_val3; \\\n\
    copy_type val0, val1, val2, val3; \\\n\
    src_type val; \\\n\
    LOGSOFTMAX_PROCESS_AXIS0(VXC_ReadImage2DArray, vert_max_fun, horz_max_fun) \\\n\
    LOGSOFTMAX_PROCESS_AXIS0_SAVE(dst_type, save_type, conv_mode,\\\n\
    OUT_SCALE, OUT_OFFSET, VXC_ReadImage2DArray, VXC_WriteImage2DArray); \\\n\
}\n\
\n\
LOGSOFTMAX_AXIS0(F16, F16, vxc_half8, vxc_short8, half4,  vxc_short8,\\\n\
CONV, 1, 0, VXC_VertMax3_Half, VXC_HorzMax3_Half)\n\
LOGSOFTMAX_AXIS0(F16, I16, vxc_half8, vxc_short8, short4, vxc_short8,\\\n\
CONV_SAT_RTE, outputScale, 0, VXC_VertMax3_Half, VXC_HorzMax3_Half)\n\
LOGSOFTMAX_AXIS0(F16, I8,  vxc_half8, vxc_short8, char4,  vxc_char8,\\\n\
CONV_SAT_RTE, outputScale, 0, VXC_VertMax3_Half, VXC_HorzMax3_Half)\n\
LOGSOFTMAX_AXIS0(F16, U8,  vxc_half8, vxc_short8, uchar4, vxc_uchar8,\\\n\
                 CONV_SAT_RTE, outputScale, output_offset_asymmetric, VXC_VertMax3_Half, VXC_HorzMax3_Half)\n\
LOGSOFTMAX_AXIS0(I16, I16, vxc_short8, vxc_short8, short4, vxc_short8,\\\n\
CONV_SAT_RTE, outputScale, 0, VXC_VertMax3_Integer, VXC_HorzMax3_Integer)\n\
LOGSOFTMAX_AXIS0(I16, F16, vxc_short8, vxc_short8, half4,  vxc_short8,\\\n\
CONV, 1, 0, VXC_VertMax3_Integer, VXC_HorzMax3_Integer)\n\
LOGSOFTMAX_AXIS0(I8, I8,  vxc_char16, vxc_char16, char4,  vxc_char8,\\\n\
CONV_SAT_RTE, outputScale, 0, VXC_VertMax3_Integer, VXC_HorzMax3_Integer)\n\
LOGSOFTMAX_AXIS0(I8, F16, vxc_char16, vxc_char16, half4,  vxc_short8,\\\n\
CONV, 1, 0, VXC_VertMax3_Integer, VXC_HorzMax3_Integer)\n\
LOGSOFTMAX_AXIS0(U8, U8,  vxc_uchar16, vxc_uchar16, uchar4, vxc_uchar8,\\\n\
CONV_SAT_RTE, outputScale, output_offset_asymmetric, VXC_VertMax3_Integer, VXC_HorzMax3_Integer)\n\
LOGSOFTMAX_AXIS0(U8, F16, vxc_uchar16, vxc_uchar16, half4,  vxc_short8,\\\n\
CONV, 1, 0, VXC_VertMax3_Integer, VXC_HorzMax3_Integer)\n\
\n\
#define LOGSOFTMAX_AXIS0_2D(src_name, dst_name, src_type, copy_type,\\\n\
                           dst_type, save_type, conv_mode, OUT_SCALE, OUT_OFFSET, vert_max_fun, horz_max_fun) \\\n\
__kernel void vxcLogSoftmaxAxis0_##src_name##to##dst_name##_2D \\\n\
    ( \\\n\
    __read_only  image2d_array_t input, \\\n\
    __write_only image2d_array_t output, \\\n\
    float input_Scale, \\\n\
    int   axisVal \\\n\
    ) \\\n\
{ \\\n\
    int2 coord = (int2)(16, get_global_id(0)); \\\n\
    src_type img_val0, img_val1, img_val2, img_val3; \\\n\
    copy_type val0, val1, val2, val3; \\\n\
    src_type val; \\\n\
    LOGSOFTMAX_PROCESS_AXIS0(VXC_ReadImage, vert_max_fun, horz_max_fun) \\\n\
    LOGSOFTMAX_PROCESS_AXIS0_SAVE(dst_type, save_type, conv_mode,\\\n\
    OUT_SCALE, OUT_OFFSET, VXC_ReadImage, VXC_WriteImage); \\\n\
}\n\
\n\
LOGSOFTMAX_AXIS0_2D(F16, F16, vxc_half8, vxc_short8, half4,  vxc_short8,\\\n\
CONV, 1, 0, VXC_VertMax3_Half, VXC_HorzMax3_Half)\n\
LOGSOFTMAX_AXIS0_2D(F16, I16, vxc_half8, vxc_short8, short4, vxc_short8,\\\n\
CONV_SAT_RTE, outputScale, 0, VXC_VertMax3_Half, VXC_HorzMax3_Half)\n\
LOGSOFTMAX_AXIS0_2D(F16, I8,  vxc_half8, vxc_short8, char4,  vxc_char8, \\\n\
CONV_SAT_RTE, outputScale, 0, VXC_VertMax3_Half, VXC_HorzMax3_Half)\n\
LOGSOFTMAX_AXIS0_2D(F16, U8,  vxc_half8, vxc_short8, uchar4, vxc_uchar8,\\\n\
CONV_SAT_RTE, outputScale, output_offset_asymmetric, VXC_VertMax3_Half, VXC_HorzMax3_Half)\n\
LOGSOFTMAX_AXIS0_2D(I16, I16, vxc_short8, vxc_short8, short4, vxc_short8,\\\n\
CONV_SAT_RTE, outputScale, 0, VXC_VertMax3_Integer, VXC_HorzMax3_Integer)\n\
LOGSOFTMAX_AXIS0_2D(I16, F16, vxc_short8, vxc_short8, half4,  vxc_short8,\\\n\
CONV, 1, 0, VXC_VertMax3_Integer, VXC_HorzMax3_Integer)\n\
LOGSOFTMAX_AXIS0_2D(I8, I8,  vxc_char16, vxc_char16, char4,  vxc_char8,\\\n\
CONV_SAT_RTE, outputScale, 0, VXC_VertMax3_Integer, VXC_HorzMax3_Integer)\n\
LOGSOFTMAX_AXIS0_2D(I8, F16, vxc_char16, vxc_char16, half4,  vxc_short8,\\\n\
CONV, 1, 0, VXC_VertMax3_Integer, VXC_HorzMax3_Integer)\n\
LOGSOFTMAX_AXIS0_2D(U8, U8,  vxc_uchar16, vxc_uchar16, uchar4, vxc_uchar8,\\\n\
CONV_SAT_RTE, outputScale, output_offset_asymmetric, VXC_VertMax3_Integer, VXC_HorzMax3_Integer)\n\
LOGSOFTMAX_AXIS0_2D(U8, F16, vxc_uchar16, vxc_uchar16, half4,  vxc_short8,\\\n\
CONV, 1, 0, VXC_VertMax3_Integer, VXC_HorzMax3_Integer)\n\
\n\
#define LOGSOFTMAX_PROCESS_AXIS0_TOF32_SAVE(read_fun) \\\n\
    for (coord.x = 0; coord.x < axisSize; ) \\\n\
    { \\\n\
        read_fun(val0, input,  coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, img_val0, val0, 16); \\\n\
        VXC_DP4x4(prob, img_val0, val, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetSubData0to3_4x4); \\\n\
        prob = prob * betaValue - probSum_log.xxxx; \\\n\
        write_imagef(output, coord, prob); \\\n\
        coord.x += 4; \\\n\
        VXC_DP4x4(prob, img_val0, val, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetSubData4to7_4x4); \\\n\
        prob = prob * betaValue - probSum_log.xxxx; \\\n\
        write_imagef(output, coord, prob); \\\n\
        coord.x += 4; \\\n\
    }\n\
\n\
#define LOGSOFTMAX_AXIS0_TOF32(src_name, src_type, copy_type, vert_max_fun, horz_max_fun) \\\n\
__kernel void vxcLogSoftmaxAxis0_##src_name##toF32 \\\n\
    ( \\\n\
    __read_only  image2d_array_t input, \\\n\
    __write_only image2d_array_t output, \\\n\
    float input_Scale, \\\n\
    int   axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(16, get_global_id(0), get_global_id(1), 0); \\\n\
    src_type img_val0, img_val1, img_val2, img_val3; \\\n\
    copy_type val0, val1, val2, val3; \\\n\
    src_type val; \\\n\
    LOGSOFTMAX_PROCESS_AXIS0(VXC_ReadImage2DArray, vert_max_fun, horz_max_fun) \\\n\
    LOGSOFTMAX_PROCESS_AXIS0_TOF32_SAVE(VXC_ReadImage2DArray) \\\n\
}\n\
\n\
LOGSOFTMAX_AXIS0_TOF32(F16, vxc_half8,   vxc_short8, VXC_VertMax3_Half, VXC_HorzMax3_Half)\n\
LOGSOFTMAX_AXIS0_TOF32(I16, vxc_short8,  vxc_short8, VXC_VertMax3_Integer, VXC_HorzMax3_Integer)\n\
LOGSOFTMAX_AXIS0_TOF32(I8,  vxc_char16,  vxc_char16, VXC_VertMax3_Integer, VXC_HorzMax3_Integer)\n\
LOGSOFTMAX_AXIS0_TOF32(U8,  vxc_uchar16, vxc_uchar16, VXC_VertMax3_Integer, VXC_HorzMax3_Integer)\n\
\n\
#define LOGSOFTMAX_AXIS0_TOF32_2D(src_name, src_type, copy_type, vert_max_fun, horz_max_fun) \\\n\
__kernel void vxcLogSoftmaxAxis0_##src_name##toF32_2D \\\n\
    ( \\\n\
    __read_only  image2d_array_t input, \\\n\
    __write_only image2d_t       output, \\\n\
    float input_Scale, \\\n\
    int   axisVal \\\n\
    ) \\\n\
{ \\\n\
    int2 coord = (int2)(16, get_global_id(0)); \\\n\
    src_type img_val0, img_val1, img_val2, img_val3; \\\n\
    copy_type val0, val1, val2, val3; \\\n\
    src_type val; \\\n\
    LOGSOFTMAX_PROCESS_AXIS0(VXC_ReadImage, vert_max_fun, horz_max_fun) \\\n\
    LOGSOFTMAX_PROCESS_AXIS0_TOF32_SAVE(VXC_ReadImage) \\\n\
}\n\
\n\
LOGSOFTMAX_AXIS0_TOF32_2D(F16, vxc_half8,   vxc_short8, VXC_VertMax3_Half, VXC_HorzMax3_Half)\n\
LOGSOFTMAX_AXIS0_TOF32_2D(I16, vxc_short8,  vxc_short8, VXC_VertMax3_Integer, VXC_HorzMax3_Integer)\n\
LOGSOFTMAX_AXIS0_TOF32_2D(I8,  vxc_char16,  vxc_char16, VXC_VertMax3_Integer, VXC_HorzMax3_Integer)\n\
LOGSOFTMAX_AXIS0_TOF32_2D(U8,  vxc_uchar16, vxc_uchar16, VXC_VertMax3_Integer, VXC_HorzMax3_Integer)\n\
\n\
"; /* end of vsi_nn_kernel_log_softmaxAxis0_vx*/

static const char vsi_nn_kernel_log_softmaxAxis0_BF16_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
_viv_uniform float       rlogE;\n\
_viv_uniform int         axisSize;\n\
_viv_uniform float       betaValue;\n\
_viv_uniform float       scaleLogE;\n\
_viv_uniform VXC_512Bits uniConvBF16toF32_Part0_2x8;\n\
\n\
_viv_uniform int         inputWidth;\n\
_viv_uniform int         inputWidthRemain4;\n\
_viv_uniform VXC_512Bits uniPackMaxData_2x8;\n\
_viv_uniform VXC_512Bits uniExtractHalf4_4x4;\n\
\n\
\n\
#define LOGSOFTMAX_PROCESS_AXIS0_BF16(read_fun) \\\n\
    vxc_half8 img_val0, img_val1, img_val2, img_val3; \\\n\
    vxc_short8 val0, val1, val2, val3; \\\n\
    vxc_half8 val; \\\n\
    read_fun(val0, input,  coord, VXC_5BITOFFSET_XY(-16, 0), VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, val, val0, 16); \\\n\
    coord.x += 8; \\\n\
    do \\\n\
    { \\\n\
        read_fun(val0, input,  coord, VXC_5BITOFFSET_XY(-16, 0), VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, img_val0, val0, 16); \\\n\
        read_fun(val1, input,  coord, VXC_5BITOFFSET_XY(-8, 0), VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, img_val1, val1, 16); \\\n\
        read_fun(val2, input,  coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, img_val2, val2, 16); \\\n\
        read_fun(val3, input,  coord, VXC_5BITOFFSET_XY(8, 0), VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, img_val3, val3, 16); \\\n\
        coord.x += 32; \\\n\
        VXC_VertMax3_Half(val, img_val0, img_val1, val, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        VXC_VertMax3_Half(val, img_val2, img_val3, val, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        VXC_HorzMax3_Half(val, val, VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0)); \\\n\
        VXC_DP2x8(val, val, val, VXC_MODIFIER(0, 2, 0, VXC_RM_TowardZero, 0), uniPackMaxData_2x8); \\\n\
        VXC_HorzMax3_Half(val, val, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0)); \\\n\
    } \\\n\
    while(coord.x < (axisSize + 16)); \\\n\
    vxc_short8 zero = (vxc_short8)(0, 0, 0, 0, 0, 0, 0, 0); \\\n\
    vxc_ushort8   bf_val_tmp; \\\n\
    vxc_float4 vecA; \\\n\
    _viv_asm(COPY, bf_val_tmp, val, 16); \\\n\
    VXC_DP2x8(bf_val_tmp, bf_val_tmp, zero,\\\n\
    VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part0_2x8); \\\n\
    _viv_asm(COPY, vecA, bf_val_tmp, 16); \\\n\
    vxc_float4 prob; \\\n\
    float fProbSum = 0; \\\n\
    const float4 one4 = (float4)(1.0, 1.0, 1.0, 1.0); \\\n\
    float max_value = vecA.x * scaleLogE; \\\n\
    float max_value_orig = vecA.x; \\\n\
    for (coord.x = 0; coord.x < inputWidth; ) \\\n\
    { \\\n\
        read_fun(val0, input,  coord, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
        VXC_DP2x8(bf_val_tmp, val0, zero,\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part0_2x8); \\\n\
        _viv_asm(COPY, prob, bf_val_tmp, 16); \\\n\
        prob = prob * scaleLogE - max_value; \\\n\
        prob = exp2(prob); \\\n\
        fProbSum += dot(prob, one4); \\\n\
        coord.x += 4; \\\n\
    } \\\n\
    read_fun(val0, input,  coord, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_DP2x8(bf_val_tmp, val0, zero,\\\n\
    VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part0_2x8); \\\n\
    _viv_asm(COPY, prob, bf_val_tmp, 16); \\\n\
    prob = prob * scaleLogE - max_value; \\\n\
    if(inputWidthRemain4 == 1) \\\n\
    { \\\n\
        prob.x = exp2(prob.x); \\\n\
        prob.yzw = 0; \\\n\
        fProbSum += dot(prob, one4); \\\n\
    } \\\n\
    else if(inputWidthRemain4 == 2) \\\n\
    { \\\n\
        prob.x = exp2(prob.x); \\\n\
        prob.y = exp2(prob.y); \\\n\
        prob.zw = 0; \\\n\
        fProbSum += dot(prob, one4); \\\n\
    } \\\n\
    else if(inputWidthRemain4 == 3) \\\n\
    { \\\n\
        prob.x = exp2(prob.x); \\\n\
        prob.y = exp2(prob.y); \\\n\
        prob.z = exp2(prob.z); \\\n\
        prob.w = 0; \\\n\
        fProbSum += dot(prob, one4); \\\n\
    } \\\n\
    vxc_float4 probSum_log; \\\n\
    probSum_log.x = log2(fProbSum) * rlogE;\n\
\n\
#define LOGSOFTMAX_PROCESS_AXIS0_BF16TOBF16_SAVE(read_fun, write_fun) \\\n\
    for (coord.x = 0; coord.x < axisSize; ) \\\n\
    { \\\n\
        read_fun(val0, input,  coord, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
        VXC_DP2x8(bf_val_tmp, val0, zero,\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part0_2x8); \\\n\
        _viv_asm(COPY, prob, bf_val_tmp, 16); \\\n\
        prob = prob - max_value_orig; \\\n\
        prob = prob * betaValue - probSum_log.xxxx; \\\n\
        vxc_ushort8 tmp, dst; \\\n\
        _viv_asm(COPY, tmp, prob, 16); \\\n\
        dst.s0123 = tmp.s1357; \\\n\
        write_fun(output, coord, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
        coord.x += 4; \\\n\
    }\n\
\n\
#define LOGSOFTMAX_PROCESS_AXIS0_BF16TOF16_SAVE(read_fun, write_fun) \\\n\
    for (coord.x = 0; coord.x < axisSize; ) \\\n\
    { \\\n\
        read_fun(val0, input,  coord, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
        VXC_DP2x8(bf_val_tmp, val0, zero,\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part0_2x8); \\\n\
        _viv_asm(COPY, prob, bf_val_tmp, 16); \\\n\
        prob = prob - max_value_orig; \\\n\
        prob = prob * betaValue - probSum_log.xxxx; \\\n\
        half4 vec; \\\n\
        vxc_half4 tmp; \\\n\
        vxc_short4 dst; \\\n\
        _viv_asm(CONV, vec, prob); \\\n\
        VXC_DP4x4(tmp, vec, vec, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtractHalf4_4x4); \\\n\
        _viv_asm(COPY, dst, tmp, 8); \\\n\
        write_fun(output, coord, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
        coord.x += 4; \\\n\
    }\n\
\n\
#define LOGSOFTMAX_PROCESS_AXIS0_BF16TOF32_SAVE(read_fun) \\\n\
    for (coord.x = 0; coord.x < axisSize; ) \\\n\
    { \\\n\
        read_fun(val0, input,  coord, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
        VXC_DP2x8(bf_val_tmp, val0, zero,\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part0_2x8); \\\n\
        _viv_asm(COPY, prob, bf_val_tmp, 16); \\\n\
        prob = prob - max_value_orig; \\\n\
        prob = prob * betaValue - probSum_log.xxxx; \\\n\
        write_imagef(output, coord, prob); \\\n\
        coord.x += 4; \\\n\
    }\n\
\n\
__kernel void vxcLogSoftmaxAxis0_BF16toBF16(\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_array_t  output,\n\
    float input_Scale,\n\
    int   axisVal )\n\
{\n\
    int4 coord = (int4)(16, get_global_id(0), get_global_id(1), 0);\n\
    LOGSOFTMAX_PROCESS_AXIS0_BF16(VXC_ReadImage2DArray)\n\
    LOGSOFTMAX_PROCESS_AXIS0_BF16TOBF16_SAVE(VXC_ReadImage2DArray, VXC_WriteImage2DArray)\n\
}\n\
__kernel void vxcLogSoftmaxAxis0_BF16toF16(\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_array_t  output,\n\
    float input_Scale,\n\
    int   axisVal )\n\
{\n\
    int4 coord = (int4)(16, get_global_id(0), get_global_id(1), 0);\n\
    LOGSOFTMAX_PROCESS_AXIS0_BF16(VXC_ReadImage2DArray)\n\
    LOGSOFTMAX_PROCESS_AXIS0_BF16TOF16_SAVE(VXC_ReadImage2DArray, VXC_WriteImage2DArray)\n\
}\n\
__kernel void vxcLogSoftmaxAxis0_BF16toF32(\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_array_t  output,\n\
    float input_Scale,\n\
    int   axisVal )\n\
{\n\
    int4 coord = (int4)(16, get_global_id(0), get_global_id(1), 0);\n\
    LOGSOFTMAX_PROCESS_AXIS0_BF16(VXC_ReadImage2DArray)\n\
    LOGSOFTMAX_PROCESS_AXIS0_BF16TOF32_SAVE(VXC_ReadImage2DArray)\n\
}\n\
__kernel void vxcLogSoftmaxAxis0_BF16toBF16_2D(\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_array_t  output,\n\
    float input_Scale,\n\
    int   axisVal )\n\
{\n\
    int2 coord = (int2)(16, get_global_id(0));\n\
    LOGSOFTMAX_PROCESS_AXIS0_BF16(VXC_ReadImage)\n\
    LOGSOFTMAX_PROCESS_AXIS0_BF16TOBF16_SAVE(VXC_ReadImage, VXC_WriteImage)\n\
}\n\
__kernel void vxcLogSoftmaxAxis0_BF16toF16_2D(\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_array_t  output,\n\
    float input_Scale,\n\
    int   axisVal )\n\
{\n\
    int2 coord = (int2)(16, get_global_id(0));\n\
    LOGSOFTMAX_PROCESS_AXIS0_BF16(VXC_ReadImage)\n\
    LOGSOFTMAX_PROCESS_AXIS0_BF16TOF16_SAVE(VXC_ReadImage, VXC_WriteImage)\n\
}\n\
__kernel void vxcLogSoftmaxAxis0_BF16toF32_2D(\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_t        output,\n\
    float input_Scale,\n\
    int   axisVal )\n\
{\n\
    int2 coord = (int2)(16, get_global_id(0));\n\
    LOGSOFTMAX_PROCESS_AXIS0_BF16(VXC_ReadImage)\n\
    LOGSOFTMAX_PROCESS_AXIS0_BF16TOF32_SAVE(VXC_ReadImage)\n\
}\n\
\n\
"; /* end of vsi_nn_kernel_log_softmaxAxis0_BF16_vx*/

static const char vsi_nn_kernel_log_softmaxAxis1_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform float       rlogE;\n\
_viv_uniform int         axisSize;\n\
_viv_uniform float       betaValue;\n\
_viv_uniform float       scaleLogE;\n\
_viv_uniform float       outputScale;\n\
_viv_uniform float       output_offset_asymmetric;\n\
_viv_uniform VXC_512Bits uniConvertInt32toUint8_2x8;\n\
\n\
_viv_uniform VXC_512Bits uniGetSubLoData_4x4;\n\
_viv_uniform VXC_512Bits uniGetSubHiData_4x4;\n\
\n\
#define LOGSOFTMAX_PROCESS_AXIS1(read_fun, vert_max_fun) \\\n\
    read_fun(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, max, in0, 16); \\\n\
    coord.y++; \\\n\
    do \\\n\
    { \\\n\
        read_fun(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, vec0, in0, 16); \\\n\
        vert_max_fun(max, max, max, vec0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        coord.y++; \\\n\
    } \\\n\
    while(coord.y < axisSize); \\\n\
    coord.y = 0; \\\n\
    sum0 = 0; \\\n\
    sum1 = 0; \\\n\
    do \\\n\
    { \\\n\
        read_fun(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, vec0, in0, 16); \\\n\
        VXC_DP4x4(data0, vec0, max, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetSubLoData_4x4); \\\n\
        data0 *= scaleLogE; \\\n\
        data0 = exp2(data0); \\\n\
        sum0 += data0; \\\n\
        VXC_DP4x4(data0, vec0, max, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetSubHiData_4x4); \\\n\
        data0 *= scaleLogE; \\\n\
        data0 = exp2(data0); \\\n\
        sum1 += data0; \\\n\
        coord.y++; \\\n\
    } \\\n\
    while(coord.y < axisSize); \\\n\
    sum0 = log2(sum0) * rlogE; \\\n\
    sum1 = log2(sum1) * rlogE;\n\
\n\
#define LOGSOFTMAX_PROCESS_AXIS1_SAVE(dst_type, save_type, conv_mode,\\\n\
                 OUT_SCALE, OUT_OFFSET, read_fun, write_fun) \\\n\
    coord.y = 0; \\\n\
    dst_type dst0, dst1; \\\n\
    save_type vect; \\\n\
    do \\\n\
    { \\\n\
        read_fun(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, vec0, in0, 16); \\\n\
        VXC_DP4x4(data0, vec0, max, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetSubLoData_4x4); \\\n\
        data0 = data0 * betaValue - sum0; \\\n\
        data0 = data0 * OUT_SCALE + OUT_OFFSET; \\\n\
        _viv_asm(conv_mode, dst0, data0); \\\n\
        VXC_DP4x4(data0, vec0, max, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetSubHiData_4x4); \\\n\
        data0 = data0 * betaValue - sum1; \\\n\
        data0 = data0 * OUT_SCALE + OUT_OFFSET; \\\n\
        _viv_asm(conv_mode, dst1, data0); \\\n\
        VXC_DP2x8(vect, dst0, dst1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvertInt32toUint8_2x8); \\\n\
        write_fun(output, coord, vect, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        coord.y++; \\\n\
    } \\\n\
    while(coord.y < axisSize);\n\
\n\
#define LOGSOFTMAX_AXIS1(src_name, dst_name, src_type, copy_type, dst_type,\\\n\
save_type, conv_mode, OUT_SCALE, OUT_OFFSET, vert_max_fun) \\\n\
__kernel void vxcLogSoftmaxAxis1_##src_name##to##dst_name \\\n\
    ( \\\n\
    __read_only  image2d_array_t input, \\\n\
    __write_only image2d_array_t output, \\\n\
    float input_Scale, \\\n\
    int   axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(get_global_id(0), 0, get_global_id(1), 0); \\\n\
    src_type vec0, max; \\\n\
    copy_type in0; \\\n\
    vxc_float4 data0; \\\n\
    vxc_float4 sum0, sum1; \\\n\
    LOGSOFTMAX_PROCESS_AXIS1(VXC_ReadImage2DArray, vert_max_fun) \\\n\
    LOGSOFTMAX_PROCESS_AXIS1_SAVE(dst_type, save_type, conv_mode,\\\n\
    OUT_SCALE, OUT_OFFSET, VXC_ReadImage2DArray, VXC_WriteImage2DArray); \\\n\
}\n\
\n\
\n\
\n\
LOGSOFTMAX_AXIS1(F16, F16, vxc_half8, vxc_short8, half4,  vxc_short8,\\\n\
CONV, 1, 0, VXC_VertMax3_Half)\n\
LOGSOFTMAX_AXIS1(F16, I16, vxc_half8, vxc_short8, short4, vxc_short8,\\\n\
CONV_SAT_RTE, outputScale, 0, VXC_VertMax3_Half)\n\
LOGSOFTMAX_AXIS1(F16, I8,  vxc_half8, vxc_short8, char4,  vxc_char8,\\\n\
CONV_SAT_RTE, outputScale, 0, VXC_VertMax3_Half)\n\
LOGSOFTMAX_AXIS1(F16, U8,  vxc_half8, vxc_short8, uchar4,\\\n\
vxc_uchar8, CONV_SAT_RTE, outputScale, output_offset_asymmetric, VXC_VertMax3_Half)\n\
\n\
LOGSOFTMAX_AXIS1(I16, I16, vxc_short8, vxc_short8, short4, vxc_short8,\\\n\
CONV_SAT_RTE, outputScale, 0, VXC_VertMax3_Integer)\n\
LOGSOFTMAX_AXIS1(I16, F16, vxc_short8, vxc_short8, half4,  vxc_short8,\\\n\
CONV, 1, 0, VXC_VertMax3_Integer)\n\
\n\
LOGSOFTMAX_AXIS1(I8, I8,  vxc_char16, vxc_char16, char4,  vxc_char8,\\\n\
CONV_SAT_RTE, outputScale, 0, VXC_VertMax3_Integer)\n\
LOGSOFTMAX_AXIS1(I8, F16, vxc_char16, vxc_char16, half4,  vxc_short8,\\\n\
CONV, 1, 0, VXC_VertMax3_Integer)\n\
\n\
LOGSOFTMAX_AXIS1(U8, U8,  vxc_uchar16, vxc_uchar16, uchar4,\\\n\
vxc_uchar8, CONV_SAT_RTE, outputScale, output_offset_asymmetric, VXC_VertMax3_Integer)\n\
LOGSOFTMAX_AXIS1(U8, F16, vxc_uchar16, vxc_uchar16, half4,  vxc_short8, CONV, 1, 0, VXC_VertMax3_Integer)\n\
\n\
#define LOGSOFTMAX_AXIS1_2D(src_name, dst_name, src_type,\\\n\
copy_type, dst_type, save_type, conv_mode, OUT_SCALE, OUT_OFFSET, vert_max_fun) \\\n\
__kernel void vxcLogSoftmaxAxis1_##src_name##to##dst_name##_2D \\\n\
    ( \\\n\
    __read_only  image2d_array_t input, \\\n\
    __write_only image2d_array_t output, \\\n\
    float input_Scale, \\\n\
    int   axisVal \\\n\
    ) \\\n\
{ \\\n\
    int2 coord = (int2)(get_global_id(0), 0); \\\n\
    src_type vec0, max; \\\n\
    copy_type in0; \\\n\
    vxc_float4 data0; \\\n\
    vxc_float4 sum0, sum1; \\\n\
    LOGSOFTMAX_PROCESS_AXIS1(VXC_ReadImage, vert_max_fun) \\\n\
    LOGSOFTMAX_PROCESS_AXIS1_SAVE(dst_type, save_type, conv_mode,\\\n\
    OUT_SCALE, OUT_OFFSET, VXC_ReadImage, VXC_WriteImage); \\\n\
}\n\
\n\
LOGSOFTMAX_AXIS1_2D(F16, F16, vxc_half8, vxc_short8, half4,  vxc_short8,\\\n\
CONV, 1, 0, VXC_VertMax3_Half)\n\
LOGSOFTMAX_AXIS1_2D(F16, I16, vxc_half8, vxc_short8, short4, vxc_short8,\\\n\
CONV_SAT_RTE, outputScale, 0, VXC_VertMax3_Half)\n\
LOGSOFTMAX_AXIS1_2D(F16, I8,  vxc_half8, vxc_short8, char4,  vxc_char8,\\\n\
CONV_SAT_RTE, outputScale, 0, VXC_VertMax3_Half)\n\
LOGSOFTMAX_AXIS1_2D(F16, U8,  vxc_half8, vxc_short8, uchar4,\\\n\
vxc_uchar8, CONV_SAT_RTE, outputScale, output_offset_asymmetric, VXC_VertMax3_Half)\n\
\n\
LOGSOFTMAX_AXIS1_2D(I16, I16, vxc_short8, vxc_short8, short4, vxc_short8,\\\n\
CONV_SAT_RTE, outputScale, 0, VXC_VertMax3_Integer)\n\
LOGSOFTMAX_AXIS1_2D(I16, F16, vxc_short8, vxc_short8, half4,  vxc_short8,\\\n\
CONV, 1, 0, VXC_VertMax3_Integer)\n\
\n\
LOGSOFTMAX_AXIS1_2D(I8, I8,  vxc_char16, vxc_char16, char4,  vxc_char8, \\\n\
CONV_SAT_RTE, outputScale, 0, VXC_VertMax3_Integer)\n\
LOGSOFTMAX_AXIS1_2D(I8, F16, vxc_char16, vxc_char16, half4,  vxc_short8, CONV, 1, 0, VXC_VertMax3_Integer)\n\
\n\
LOGSOFTMAX_AXIS1_2D(U8, U8,  vxc_uchar16, vxc_uchar16, uchar4,\\\n\
vxc_uchar8, CONV_SAT_RTE, outputScale, output_offset_asymmetric, VXC_VertMax3_Integer)\n\
LOGSOFTMAX_AXIS1_2D(U8, F16, vxc_uchar16, vxc_uchar16, half4,  vxc_short8, CONV, 1, 0, VXC_VertMax3_Integer)\n\
\n\
\n\
#define LOGSOFTMAX_AXIS1_TOF32(src_name, src_type, copy_type, vert_max_fun) \\\n\
__kernel void vxcLogSoftmaxAxis1_##src_name##toF32 \\\n\
    ( \\\n\
    __read_only  image2d_array_t input, \\\n\
    __write_only image2d_array_t output, \\\n\
    float input_Scale, \\\n\
    int   axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(get_global_id(0), 0, get_global_id(1), 0); \\\n\
    src_type vec0, max; \\\n\
    copy_type in0; \\\n\
    vxc_float4 data0; \\\n\
    vxc_float4 sum0, sum1; \\\n\
    LOGSOFTMAX_PROCESS_AXIS1(VXC_ReadImage2DArray, vert_max_fun) \\\n\
    coord.y = 0; \\\n\
    do \\\n\
    { \\\n\
        VXC_ReadImage2DArray(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, vec0, in0, 16); \\\n\
        VXC_DP4x4(data0, vec0, max, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetSubLoData_4x4); \\\n\
        data0 = data0 * betaValue - sum0; \\\n\
        write_imagef(output, coord, data0); \\\n\
        coord.x += 4; \\\n\
        VXC_DP4x4(data0, vec0, max, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetSubHiData_4x4); \\\n\
        data0 = data0 * betaValue - sum1; \\\n\
        write_imagef(output, coord, data0); \\\n\
        coord.x -= 4; \\\n\
        coord.y++; \\\n\
    } \\\n\
    while(coord.y < axisSize); \\\n\
}\n\
\n\
LOGSOFTMAX_AXIS1_TOF32(F16, vxc_half8,   vxc_short8, VXC_VertMax3_Half)\n\
LOGSOFTMAX_AXIS1_TOF32(I16, vxc_short8,  vxc_short8, VXC_VertMax3_Integer)\n\
LOGSOFTMAX_AXIS1_TOF32(I8,  vxc_char16,  vxc_char16, VXC_VertMax3_Integer)\n\
LOGSOFTMAX_AXIS1_TOF32(U8,  vxc_uchar16, vxc_uchar16, VXC_VertMax3_Integer)\n\
\n\
#define LOGSOFTMAX_AXIS1_TOF32_2D(src_name, src_type, copy_type, vert_max_fun) \\\n\
__kernel void vxcLogSoftmaxAxis1_##src_name##toF32_2D \\\n\
    ( \\\n\
    __read_only  image2d_array_t input, \\\n\
    __write_only image2d_t       output, \\\n\
    float input_Scale, \\\n\
    int   axisVal \\\n\
    ) \\\n\
{ \\\n\
    int2 coord = (int2)(get_global_id(0), 0); \\\n\
    src_type vec0, max; \\\n\
    copy_type in0; \\\n\
    vxc_float4 data0; \\\n\
    vxc_float4 sum0, sum1; \\\n\
    LOGSOFTMAX_PROCESS_AXIS1(VXC_ReadImage, vert_max_fun) \\\n\
    coord.y = 0; \\\n\
    do \\\n\
    { \\\n\
        VXC_ReadImage(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, vec0, in0, 16); \\\n\
        VXC_DP4x4(data0, vec0, max, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetSubLoData_4x4); \\\n\
        data0 = data0 * betaValue - sum0; \\\n\
        write_imagef(output, coord, data0); \\\n\
        coord.x += 4; \\\n\
        VXC_DP4x4(data0, vec0, max, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetSubHiData_4x4); \\\n\
        data0 = data0 * betaValue - sum1; \\\n\
        write_imagef(output, coord, data0); \\\n\
        coord.x -= 4; \\\n\
        coord.y++; \\\n\
    } \\\n\
    while(coord.y < axisSize); \\\n\
}\n\
\n\
LOGSOFTMAX_AXIS1_TOF32_2D(F16, vxc_half8,   vxc_short8, VXC_VertMax3_Half)\n\
LOGSOFTMAX_AXIS1_TOF32_2D(I16, vxc_short8,  vxc_short8, VXC_VertMax3_Integer)\n\
LOGSOFTMAX_AXIS1_TOF32_2D(I8,  vxc_char16,  vxc_char16, VXC_VertMax3_Integer)\n\
LOGSOFTMAX_AXIS1_TOF32_2D(U8,  vxc_uchar16, vxc_uchar16, VXC_VertMax3_Integer)\n\
"; /* end of vsi_nn_kernel_log_softmaxAxis1_vx*/

static const char vsi_nn_kernel_log_softmaxAxis1_BF16_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform float       rlogE;\n\
_viv_uniform int         axisSize;\n\
_viv_uniform float       betaValue;\n\
_viv_uniform float       scaleLogE;\n\
_viv_uniform VXC_512Bits uniConvBF16toF32_Part0_2x8;\n\
\n\
_viv_uniform VXC_512Bits uniExtractHalf8_2x8;\n\
_viv_uniform VXC_512Bits uniConvBF16toF32_Part1_2x8;\n\
_viv_uniform VXC_512Bits uniExtractOddData_2x8;\n\
\n\
#define LOGSOFTMAX_PROCESS_AXIS1_BF16(read_fun) \\\n\
    read_fun(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, max, in0, 16); \\\n\
    coord.y++; \\\n\
    do \\\n\
    { \\\n\
        read_fun(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, vec0, in0, 16); \\\n\
        VXC_VertMax3_Half(max, max, max, vec0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        coord.y++; \\\n\
    } \\\n\
    while(coord.y < axisSize); \\\n\
    _viv_asm(COPY, tmp0, max, 16); \\\n\
    VXC_DP2x8(tmp1, tmp0, zero, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part0_2x8); \\\n\
    _viv_asm(COPY, max_lo, tmp1, 16); \\\n\
    VXC_DP2x8(tmp1, tmp0, zero, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part1_2x8); \\\n\
    _viv_asm(COPY, max_hi, tmp1, 16); \\\n\
    coord.y = 0; \\\n\
    sum0 = 0; \\\n\
    sum1 = 0; \\\n\
    do \\\n\
    { \\\n\
        read_fun(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        VXC_DP2x8(tmp1, in0, zero, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part0_2x8); \\\n\
        _viv_asm(COPY, data0, tmp1, 16); \\\n\
        data0 = data0 - max_lo; \\\n\
        data0 *= scaleLogE; \\\n\
        sum0  += exp2(data0); \\\n\
        VXC_DP2x8(tmp1, in0, zero, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part1_2x8); \\\n\
        _viv_asm(COPY, data0, tmp1, 16); \\\n\
        data0 = data0 - max_hi; \\\n\
        data0 *= scaleLogE; \\\n\
        sum1  += exp2(data0); \\\n\
        coord.y++; \\\n\
    } \\\n\
    while (coord.y < axisSize); \\\n\
    sum0 = log2(sum0) * rlogE; \\\n\
    sum1 = log2(sum1) * rlogE;\n\
\n\
__kernel void vxcLogSoftmaxAxis1_BF16toBF16(\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_array_t  output,\n\
    float input_Scale,\n\
    int   axisVal )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), 0, get_global_id(1), 0);\n\
    vxc_short8 in0;\n\
    vxc_half8 vec0, max;\n\
    vxc_float4 data0;\n\
    vxc_float4 sum0, sum1;\n\
    vxc_float4 max_lo, max_hi;\n\
    vxc_ushort8   tmp0, tmp1;\n\
    vxc_short8 zero = (vxc_short8)(0, 0, 0, 0, 0, 0, 0, 0);\n\
\n\
    LOGSOFTMAX_PROCESS_AXIS1_BF16(VXC_ReadImage2DArray)\n\
\n\
    coord.y = 0;\n\
    vxc_ushort8 dst0, dst1, dst;\n\
    do\n\
    {\n\
        VXC_ReadImage2DArray(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_DP2x8(tmp1, in0, zero, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part0_2x8);\n\
        _viv_asm(COPY, data0, tmp1, 16);\n\
        data0 = data0 - max_lo;\n\
        data0 = data0 * betaValue - sum0;\n\
        _viv_asm(COPY, dst0, data0, 16);\n\
        VXC_DP2x8(tmp1, in0, zero, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part1_2x8);\n\
        _viv_asm(COPY, data0, tmp1, 16);\n\
        data0 = data0 - max_hi;\n\
        data0 = data0 * betaValue - sum1;\n\
        _viv_asm(COPY, dst1, data0, 16);\n\
        VXC_DP2x8(dst, dst0, dst1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniExtractOddData_2x8);\n\
        VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        coord.y++;\n\
    }\n\
    while(coord.y < axisSize);\n\
}\n\
\n\
__kernel void vxcLogSoftmaxAxis1_BF16toF16(\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_array_t  output,\n\
    float input_Scale,\n\
    int   axisVal )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), 0, get_global_id(1), 0);\n\
    vxc_short8 in0;\n\
    vxc_half8 vec0, max;\n\
    vxc_float4 data0;\n\
    vxc_float4 sum0, sum1;\n\
    vxc_float4 max_lo, max_hi;\n\
    vxc_ushort8   tmp0, tmp1;\n\
    vxc_short8 zero = (vxc_short8)(0, 0, 0, 0, 0, 0, 0, 0);\n\
\n\
    LOGSOFTMAX_PROCESS_AXIS1_BF16(VXC_ReadImage2DArray)\n\
\n\
    coord.y = 0;\n\
    half4 dst0, dst1;\n\
    do\n\
    {\n\
        VXC_ReadImage2DArray(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_DP2x8(tmp1, in0, zero, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part0_2x8);\n\
        _viv_asm(COPY, data0, tmp1, 16);\n\
        data0 = data0 - max_lo;\n\
        data0 = data0 * betaValue - sum0;\n\
        _viv_asm(CONV, dst0, data0);\n\
        VXC_DP2x8(tmp1, in0, zero, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part1_2x8);\n\
        _viv_asm(COPY, data0, tmp1, 16);\n\
        data0 = data0 - max_hi;\n\
        data0 = data0 * betaValue - sum1;\n\
        _viv_asm(CONV, dst1, data0);\n\
        VXC_DP2x8(vec0, dst0, dst1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniExtractHalf8_2x8);\n\
        vxc_short8 vect;\n\
        _viv_asm(COPY, vect, vec0, 16);\n\
        VXC_WriteImage2DArray(output, coord, vect, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        coord.y++;\n\
    }\n\
    while(coord.y < axisSize);\n\
}\n\
\n\
__kernel void vxcLogSoftmaxAxis1_BF16toF32(\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_array_t  output,\n\
    float input_Scale,\n\
    int   axisVal )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), 0, get_global_id(1), 0);\n\
    vxc_short8 in0;\n\
    vxc_half8 vec0, max;\n\
    vxc_float4 data0;\n\
    vxc_float4 sum0, sum1;\n\
    vxc_float4 max_lo, max_hi;\n\
    vxc_ushort8   tmp0, tmp1;\n\
    vxc_short8 zero = (vxc_short8)(0, 0, 0, 0, 0, 0, 0, 0);\n\
\n\
    LOGSOFTMAX_PROCESS_AXIS1_BF16(VXC_ReadImage2DArray)\n\
\n\
    coord.y = 0;\n\
    do\n\
    {\n\
        VXC_ReadImage2DArray(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_DP2x8(tmp1, in0, zero, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part0_2x8);\n\
        _viv_asm(COPY, data0, tmp1, 16);\n\
        data0 = data0 - max_lo;\n\
        data0 = data0 * betaValue - sum0;\n\
        write_imagef(output, coord, data0);\n\
        coord.x += 4;\n\
        VXC_DP2x8(tmp1, in0, zero, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part1_2x8);\n\
        _viv_asm(COPY, data0, tmp1, 16);\n\
        data0 = data0 - max_hi;\n\
        data0 = data0 * betaValue - sum1;\n\
        write_imagef(output, coord, data0);\n\
        coord.x -= 4;\n\
        coord.y++;\n\
    }\n\
    while (coord.y < axisSize);\n\
}\n\
\n\
__kernel void vxcLogSoftmaxAxis1_BF16toBF16_2D(\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_array_t  output,\n\
    float input_Scale,\n\
    int   axisVal )\n\
{\n\
    int2 coord = (int2)(get_global_id(0), 0);\n\
    vxc_short8 in0;\n\
    vxc_half8 vec0, max;\n\
    vxc_float4 data0;\n\
    vxc_float4 sum0, sum1;\n\
    vxc_float4 max_lo, max_hi;\n\
    vxc_ushort8   tmp0, tmp1;\n\
    vxc_short8 zero = (vxc_short8)(0, 0, 0, 0, 0, 0, 0, 0);\n\
\n\
    LOGSOFTMAX_PROCESS_AXIS1_BF16(VXC_ReadImage)\n\
\n\
    coord.y = 0;\n\
    vxc_ushort8 dst0, dst1, dst;\n\
    do\n\
    {\n\
        VXC_ReadImage(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_DP2x8(tmp1, in0, zero, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part0_2x8);\n\
        _viv_asm(COPY, data0, tmp1, 16);\n\
        data0 = data0 - max_lo;\n\
        data0 = data0 * betaValue - sum0;\n\
        _viv_asm(COPY, dst0, data0, 16);\n\
        VXC_DP2x8(tmp1, in0, zero, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part1_2x8);\n\
        _viv_asm(COPY, data0, tmp1, 16);\n\
        data0 = data0 - max_hi;\n\
        data0 = data0 * betaValue - sum1;\n\
        _viv_asm(COPY, dst1, data0, 16);\n\
        VXC_DP2x8(dst, dst0, dst1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniExtractOddData_2x8);\n\
        VXC_WriteImage(output, coord, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        coord.y++;\n\
    }\n\
    while(coord.y < axisSize);\n\
}\n\
\n\
__kernel void vxcLogSoftmaxAxis1_BF16toF16_2D(\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_array_t  output,\n\
    float input_Scale,\n\
    int   axisVal )\n\
{\n\
    int2 coord = (int2)(get_global_id(0), 0);\n\
    vxc_short8 in0;\n\
    vxc_half8 vec0, max;\n\
    vxc_float4 data0;\n\
    vxc_float4 sum0, sum1;\n\
    vxc_float4 max_lo, max_hi;\n\
    vxc_ushort8   tmp0, tmp1;\n\
    vxc_short8 zero = (vxc_short8)(0, 0, 0, 0, 0, 0, 0, 0);\n\
\n\
    LOGSOFTMAX_PROCESS_AXIS1_BF16(VXC_ReadImage)\n\
\n\
    coord.y = 0;\n\
    half4 dst0, dst1;\n\
    do\n\
    {\n\
        VXC_ReadImage(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_DP2x8(tmp1, in0, zero, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part0_2x8);\n\
        _viv_asm(COPY, data0, tmp1, 16);\n\
        data0 = data0 - max_lo;\n\
        data0 = data0 * betaValue - sum0;\n\
        _viv_asm(CONV, dst0, data0);\n\
        VXC_DP2x8(tmp1, in0, zero, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part1_2x8);\n\
        _viv_asm(COPY, data0, tmp1, 16);\n\
        data0 = data0 - max_hi;\n\
        data0 = data0 * betaValue - sum1;\n\
        _viv_asm(CONV, dst1, data0);\n\
        VXC_DP2x8(vec0, dst0, dst1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniExtractHalf8_2x8);\n\
        vxc_short8 vect;\n\
        _viv_asm(COPY, vect, vec0, 16);\n\
        VXC_WriteImage(output, coord, vect, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        coord.y++;\n\
    }\n\
    while(coord.y < axisSize);\n\
}\n\
\n\
__kernel void vxcLogSoftmaxAxis1_BF16toF32_2D(\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_t        output,\n\
    float input_Scale,\n\
    int   axisVal )\n\
{\n\
    int2 coord = (int2)(get_global_id(0), 0);\n\
    vxc_short8 in0;\n\
    vxc_half8 vec0, max;\n\
    vxc_float4 data0;\n\
    vxc_float4 sum0, sum1;\n\
    vxc_float4 max_lo, max_hi;\n\
    vxc_ushort8   tmp0, tmp1;\n\
    vxc_short8 zero = (vxc_short8)(0, 0, 0, 0, 0, 0, 0, 0);\n\
\n\
    LOGSOFTMAX_PROCESS_AXIS1_BF16(VXC_ReadImage)\n\
\n\
    coord.y = 0;\n\
    do\n\
    {\n\
        VXC_ReadImage(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_DP2x8(tmp1, in0, zero, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part0_2x8);\n\
        _viv_asm(COPY, data0, tmp1, 16);\n\
        data0 = data0 - max_lo;\n\
        data0 = data0 * betaValue - sum0;\n\
        write_imagef(output, coord, data0);\n\
        coord.x += 4;\n\
        VXC_DP2x8(tmp1, in0, zero, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part1_2x8);\n\
        _viv_asm(COPY, data0, tmp1, 16);\n\
        data0 = data0 - max_hi;\n\
        data0 = data0 * betaValue - sum1;\n\
        write_imagef(output, coord, data0);\n\
        coord.x -= 4;\n\
        coord.y++;\n\
    }\n\
    while (coord.y < axisSize);\n\
}\n\
"; /* end of vsi_nn_kernel_log_softmaxAxis1_BF16_vx*/

static const char vsi_nn_kernel_log_softmaxAxis2_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform float       rlogE;\n\
_viv_uniform int         axisSize;\n\
_viv_uniform float       betaValue;\n\
_viv_uniform float       scaleLogE;\n\
_viv_uniform float       outputScale;\n\
_viv_uniform float       output_offset_asymmetric;\n\
_viv_uniform VXC_512Bits uniConvertInt32toUint8_2x8;\n\
_viv_uniform VXC_512Bits uniConvBF16toF32_Part0_2x8;\n\
\n\
_viv_uniform VXC_512Bits uniGetSubLoData_4x4;\n\
_viv_uniform VXC_512Bits uniGetSubHiData_4x4;\n\
_viv_uniform VXC_512Bits uniExtractHalf8_2x8;\n\
_viv_uniform VXC_512Bits uniConvBF16toF32_Part1_2x8;\n\
_viv_uniform VXC_512Bits uniExtractOddData_2x8;\n\
\n\
#define LOGSOFTMAX_PROCESS_AXIS2(read_fun, vert_max_fun) \\\n\
    read_fun(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, max, in0, 16); \\\n\
    coord.z++; \\\n\
    do \\\n\
    { \\\n\
        read_fun(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, vec0, in0, 16); \\\n\
        vert_max_fun(max, max, max, vec0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        coord.z++; \\\n\
    } \\\n\
    while(coord.z < axisSize); \\\n\
    coord.z = 0; \\\n\
    sum0 = 0; \\\n\
    sum1 = 0; \\\n\
    do \\\n\
    { \\\n\
        read_fun(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, vec0, in0, 16); \\\n\
        VXC_DP4x4(data0, vec0, max, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetSubLoData_4x4); \\\n\
        data0 *= scaleLogE; \\\n\
        data0 = exp2(data0); \\\n\
        sum0 += data0; \\\n\
        VXC_DP4x4(data0, vec0, max, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetSubHiData_4x4); \\\n\
        data0 *= scaleLogE; \\\n\
        data0 = exp2(data0); \\\n\
        sum1 += data0; \\\n\
        coord.z++; \\\n\
    } \\\n\
    while(coord.z < axisSize); \\\n\
    sum0 = log2(sum0) * rlogE; \\\n\
    sum1 = log2(sum1) * rlogE;\n\
\n\
#define LOGSOFTMAX_PROCESS_AXIS2_SAVE(dst_type, save_type,\\\n\
conv_mode, OUT_SCALE, OUT_OFFSET, read_fun, write_fun) \\\n\
    coord.z = 0; \\\n\
    dst_type dst0, dst1; \\\n\
    save_type vect; \\\n\
    do \\\n\
    { \\\n\
        read_fun(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, vec0, in0, 16); \\\n\
        VXC_DP4x4(data0, vec0, max, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetSubLoData_4x4); \\\n\
        data0 = data0 * betaValue - sum0; \\\n\
        data0 = data0 * OUT_SCALE + OUT_OFFSET; \\\n\
        _viv_asm(conv_mode, dst0, data0); \\\n\
        VXC_DP4x4(data0, vec0, max, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetSubHiData_4x4); \\\n\
        data0 = data0 * betaValue - sum1; \\\n\
        data0 = data0 * OUT_SCALE + OUT_OFFSET; \\\n\
        _viv_asm(conv_mode, dst1, data0); \\\n\
        VXC_DP2x8(vect, dst0, dst1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvertInt32toUint8_2x8); \\\n\
        write_fun(output, coord, vect, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        coord.z++; \\\n\
    } \\\n\
    while(coord.z < axisSize);\n\
\n\
#define LOGSOFTMAX_AXIS2(src_name, dst_name, src_type, copy_type,\\\n\
dst_type, save_type, conv_mode, OUT_SCALE, OUT_OFFSET, vert_max_fun) \\\n\
__kernel void vxcLogSoftmaxAxis2_##src_name##to##dst_name \\\n\
    ( \\\n\
    __read_only  image2d_array_t input, \\\n\
    __write_only image2d_array_t output, \\\n\
    float input_Scale, \\\n\
    int   axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), 0, 0); \\\n\
    src_type vec0, max; \\\n\
    copy_type in0; \\\n\
    vxc_float4 data0; \\\n\
    vxc_float4 sum0, sum1; \\\n\
    LOGSOFTMAX_PROCESS_AXIS2(VXC_ReadImage2DArray, vert_max_fun) \\\n\
    LOGSOFTMAX_PROCESS_AXIS2_SAVE(dst_type, save_type, conv_mode,\\\n\
    OUT_SCALE, OUT_OFFSET, VXC_ReadImage2DArray, VXC_WriteImage2DArray); \\\n\
}\n\
\n\
LOGSOFTMAX_AXIS2(F16, F16, vxc_half8, vxc_short8, half4,  vxc_short8,\\\n\
CONV, 1, 0, VXC_VertMax3_Half)\n\
LOGSOFTMAX_AXIS2(F16, I16, vxc_half8, vxc_short8, short4, vxc_short8,\\\n\
CONV_SAT_RTE, outputScale, 0, VXC_VertMax3_Half)\n\
LOGSOFTMAX_AXIS2(F16, I8,  vxc_half8, vxc_short8, char4,  vxc_char8,\\\n\
CONV_SAT_RTE, outputScale, 0, VXC_VertMax3_Half)\n\
LOGSOFTMAX_AXIS2(F16, U8,  vxc_half8, vxc_short8, uchar4, vxc_uchar8,\\\n\
CONV_SAT_RTE, outputScale, output_offset_asymmetric, VXC_VertMax3_Half)\n\
\n\
LOGSOFTMAX_AXIS2(I16, I16, vxc_short8, vxc_short8, short4, vxc_short8,\\\n\
CONV_SAT_RTE, outputScale, 0, VXC_VertMax3_Integer)\n\
LOGSOFTMAX_AXIS2(I16, F16, vxc_short8, vxc_short8, half4,  vxc_short8,\\\n\
CONV, 1, 0, VXC_VertMax3_Integer)\n\
\n\
LOGSOFTMAX_AXIS2(I8, I8,  vxc_char16, vxc_char16, char4,  vxc_char8,\\\n\
CONV_SAT_RTE, outputScale, 0, VXC_VertMax3_Integer)\n\
LOGSOFTMAX_AXIS2(I8, F16, vxc_char16, vxc_char16, half4,  vxc_short8,\\\n\
CONV, 1, 0, VXC_VertMax3_Integer)\n\
\n\
LOGSOFTMAX_AXIS2(U8, U8,  vxc_uchar16, vxc_uchar16, uchar4, vxc_uchar8,\\\n\
CONV_SAT_RTE, outputScale, output_offset_asymmetric, VXC_VertMax3_Integer)\n\
LOGSOFTMAX_AXIS2(U8, F16, vxc_uchar16, vxc_uchar16, half4,  vxc_short8,\\\n\
CONV, 1, 0, VXC_VertMax3_Integer)\n\
\n\
\n\
#define LOGSOFTMAX_AXIS2_TOF32(src_name, src_type, copy_type, vert_max_fun) \\\n\
__kernel void vxcLogSoftmaxAxis2_##src_name##toF32 \\\n\
    ( \\\n\
    __read_only  image2d_array_t input, \\\n\
    __write_only image2d_array_t output, \\\n\
    float input_Scale, \\\n\
    int   axisVal \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), 0, 0); \\\n\
    src_type vec0, max; \\\n\
    copy_type in0; \\\n\
    vxc_float4 data0; \\\n\
    vxc_float4 sum0, sum1; \\\n\
    LOGSOFTMAX_PROCESS_AXIS2(VXC_ReadImage2DArray, vert_max_fun) \\\n\
    coord.z = 0; \\\n\
    do \\\n\
    { \\\n\
        VXC_ReadImage2DArray(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, vec0, in0, 16); \\\n\
        VXC_DP4x4(data0, vec0, max, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetSubLoData_4x4); \\\n\
        data0 = data0 * betaValue - sum0; \\\n\
        write_imagef(output, coord, data0); \\\n\
        coord.x += 4; \\\n\
        VXC_DP4x4(data0, vec0, max, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetSubHiData_4x4); \\\n\
        data0 = data0 * betaValue - sum1; \\\n\
        write_imagef(output, coord, data0); \\\n\
        coord.x -= 4; \\\n\
        coord.z++; \\\n\
    } \\\n\
    while(coord.z < axisSize); \\\n\
}\n\
\n\
LOGSOFTMAX_AXIS2_TOF32(F16, vxc_half8,   vxc_short8, VXC_VertMax3_Half)\n\
LOGSOFTMAX_AXIS2_TOF32(I16, vxc_short8,  vxc_short8, VXC_VertMax3_Integer)\n\
LOGSOFTMAX_AXIS2_TOF32(I8,  vxc_char16,  vxc_char16, VXC_VertMax3_Integer)\n\
LOGSOFTMAX_AXIS2_TOF32(U8,  vxc_uchar16, vxc_uchar16, VXC_VertMax3_Integer)\n\
\n\
#define LOGSOFTMAX_PROCESS_AXIS2_BF16(read_fun) \\\n\
    read_fun(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, max, in0, 16); \\\n\
    coord.z++; \\\n\
    do \\\n\
    { \\\n\
        read_fun(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        _viv_asm(COPY, vec0, in0, 16); \\\n\
        VXC_VertMax3_Half(max, max, max, vec0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        coord.z++; \\\n\
    } \\\n\
    while(coord.z < axisSize); \\\n\
    _viv_asm(COPY, tmp0, max, 16); \\\n\
    VXC_DP2x8(tmp1, tmp0, zero, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part0_2x8); \\\n\
    _viv_asm(COPY, max_lo, tmp1, 16); \\\n\
    VXC_DP2x8(tmp1, tmp0, zero, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part1_2x8); \\\n\
    _viv_asm(COPY, max_hi, tmp1, 16); \\\n\
    coord.z = 0; \\\n\
    sum0 = 0; \\\n\
    sum1 = 0; \\\n\
    do \\\n\
    { \\\n\
        read_fun(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
        VXC_DP2x8(tmp1, in0, zero, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part0_2x8); \\\n\
        _viv_asm(COPY, data0, tmp1, 16); \\\n\
        data0  = data0 - max_lo; \\\n\
        data0 *= scaleLogE; \\\n\
        sum0  += exp2(data0); \\\n\
        VXC_DP2x8(tmp1, in0, zero, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part1_2x8); \\\n\
        _viv_asm(COPY, data0, tmp1, 16); \\\n\
        data0  = data0 - max_hi; \\\n\
        data0 *= scaleLogE; \\\n\
        sum1  += exp2(data0); \\\n\
        coord.z++; \\\n\
    } \\\n\
    while (coord.z < axisSize); \\\n\
    sum0 = log2(sum0) * rlogE; \\\n\
    sum1 = log2(sum1) * rlogE;\n\
\n\
__kernel void vxcLogSoftmaxAxis2_BF16toBF16(\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_array_t  output,\n\
    float input_Scale,\n\
    int   axisVal )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), 0, 0);\n\
    vxc_short8 in0;\n\
    vxc_half8 vec0, max;\n\
    vxc_float4 data0;\n\
    vxc_float4 sum0, sum1;\n\
    vxc_float4 max_lo, max_hi;\n\
    vxc_ushort8   tmp0, tmp1;\n\
    vxc_short8 zero = (vxc_short8)(0, 0, 0, 0, 0, 0, 0, 0);\n\
\n\
    LOGSOFTMAX_PROCESS_AXIS2_BF16(VXC_ReadImage2DArray)\n\
\n\
    coord.z = 0;\n\
    vxc_ushort8 dst0, dst1, dst;\n\
    do\n\
    {\n\
        VXC_ReadImage2DArray(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_DP2x8(tmp1, in0, zero, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part0_2x8);\n\
        _viv_asm(COPY, data0, tmp1, 16);\n\
        data0 = data0 - max_lo;\n\
        data0 = data0 * betaValue - sum0;\n\
        _viv_asm(COPY, dst0, data0, 16);\n\
        VXC_DP2x8(tmp1, in0, zero, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part1_2x8);\n\
        _viv_asm(COPY, data0, tmp1, 16);\n\
        data0 = data0 - max_hi;\n\
        data0 = data0 * betaValue - sum1;\n\
        _viv_asm(COPY, dst1, data0, 16);\n\
        VXC_DP2x8(dst, dst0, dst1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniExtractOddData_2x8);\n\
        VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        coord.z++;\n\
    }\n\
    while(coord.z < axisSize);\n\
}\n\
\n\
__kernel void vxcLogSoftmaxAxis2_BF16toF16(\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_array_t  output,\n\
    float input_Scale,\n\
    int   axisVal )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), 0, 0);\n\
    vxc_short8 in0;\n\
    vxc_half8 vec0, max;\n\
    vxc_float4 data0;\n\
    vxc_float4 sum0, sum1;\n\
    vxc_float4 max_lo, max_hi;\n\
    vxc_ushort8   tmp0, tmp1;\n\
    vxc_short8 zero = (vxc_short8)(0, 0, 0, 0, 0, 0, 0, 0);\n\
\n\
    LOGSOFTMAX_PROCESS_AXIS2_BF16(VXC_ReadImage2DArray)\n\
\n\
    coord.z = 0;\n\
    half4 dst0, dst1;\n\
    do\n\
    {\n\
        VXC_ReadImage2DArray(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_DP2x8(tmp1, in0, zero, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part0_2x8);\n\
        _viv_asm(COPY, data0, tmp1, 16);\n\
        data0 = data0 - max_lo;\n\
        data0 = data0 * betaValue - sum0;\n\
        _viv_asm(CONV, dst0, data0);\n\
        VXC_DP2x8(tmp1, in0, zero, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part1_2x8);\n\
        _viv_asm(COPY, data0, tmp1, 16);\n\
        data0 = data0 - max_hi;\n\
        data0 = data0 * betaValue - sum1;\n\
        _viv_asm(CONV, dst1, data0);\n\
        VXC_DP2x8(vec0, dst0, dst1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniExtractHalf8_2x8);\n\
        vxc_short8 vect;\n\
        _viv_asm(COPY, vect, vec0, 16);\n\
        VXC_WriteImage2DArray(output, coord, vect, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        coord.z++;\n\
    }\n\
    while(coord.z < axisSize);\n\
}\n\
\n\
__kernel void vxcLogSoftmaxAxis2_BF16toF32(\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_array_t  output,\n\
    float input_Scale,\n\
    int   axisVal )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), 0, 0);\n\
    vxc_short8 in0;\n\
    vxc_half8 vec0, max;\n\
    vxc_float4 data0;\n\
    vxc_float4 sum0, sum1;\n\
    vxc_float4 max_lo, max_hi;\n\
    vxc_ushort8   tmp0, tmp1;\n\
    vxc_short8 zero = (vxc_short8)(0, 0, 0, 0, 0, 0, 0, 0);\n\
\n\
    LOGSOFTMAX_PROCESS_AXIS2_BF16(VXC_ReadImage2DArray)\n\
\n\
    coord.z = 0;\n\
    do\n\
    {\n\
        VXC_ReadImage2DArray(in0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_DP2x8(tmp1, in0, zero, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part0_2x8);\n\
        _viv_asm(COPY, data0, tmp1, 16);\n\
        data0 = data0 - max_lo;\n\
        data0 = data0 * betaValue - sum0;\n\
        write_imagef(output, coord, data0);\n\
        coord.x += 4;\n\
        VXC_DP2x8(tmp1, in0, zero, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part1_2x8);\n\
        _viv_asm(COPY, data0, tmp1, 16);\n\
        data0 = data0 - max_hi;\n\
        data0 = data0 * betaValue - sum1;\n\
        write_imagef(output, coord, data0);\n\
        coord.x -= 4;\n\
        coord.z++;\n\
    }\n\
    while (coord.z < axisSize);\n\
}\n\
"; /* end of vsi_nn_kernel_log_softmaxAxis2_vx*/

static const char vsi_nn_kernel_logical_not_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
__kernel void vxcLogical_not_i8(\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_array_t  output)\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_char8 src0;\n\
    vxc_char8 dst;\n\
    VXC_ReadImage2DArray(src0, input, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    dst = !src0;\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcLogical_not_u8(\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_array_t  output)\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_uchar8 src0;\n\
    vxc_char8 dst;\n\
    VXC_ReadImage2DArray(src0, input, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    dst = !src0;\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcLogical_not_i16(\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_array_t  output)\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_short8 src0;\n\
    vxc_short8 dst;\n\
    VXC_ReadImage2DArray(src0, input, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    dst = !src0;\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_logical_not_vx*/

static const char vsi_nn_kernel_logical_ops_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
#if 0\n\
__kernel void vxcTensorLogical_or_int8(\n\
    __read_only image2d_array_t   input0,\n\
    __read_only image2d_array_t   input1,\n\
    __write_only image2d_array_t  output)\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_char8 src0, src1;\n\
    vxc_char8 dst;\n\
    VXC_ReadImage2DArray(src0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    dst = src0 || src1;\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
#endif\n\
\n\
#if 1\n\
#define TENSORLOGICAL(name0, name1, input_type, copy_type, output_type, out_copy_type, lgc_op) \\\n\
    __kernel void vxcTensorLogical_##name0##_##name1( \\\n\
    __read_only  image2d_array_t in0, \\\n\
    __read_only  image2d_array_t in1, \\\n\
    __write_only image2d_array_t output) \\\n\
{\\\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\\\n\
    input_type vA;\\\n\
    copy_type  src0;\\\n\
    input_type vB;\\\n\
    copy_type  src1;\\\n\
    VXC_ReadImage2DArray(vA,in0,coord,VXC_5BITOFFSET_XY(0,0),VXC_MODIFIER(0,7,0,VXC_RM_TowardZero,0));\\\n\
    _viv_asm(COPY, src0, vA, 16); \\\n\
    VXC_ReadImage2DArray(vB,in1,coord,VXC_5BITOFFSET_XY(0,0),VXC_MODIFIER(0,7,0,VXC_RM_TowardZero,0));\\\n\
    _viv_asm(COPY, src1, vB, 16); \\\n\
    output_type dst; \\\n\
    dst = (src0)lgc_op(src1); \\\n\
    dst *= (-1); \\\n\
    out_copy_type data; \\\n\
    _viv_asm(COPY, data, dst, 16); \\\n\
    VXC_WriteImage2DArray(output, coord, data, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
\n\
_viv_uniform VXC_512Bits uniMulShortMinus1toFp16_2x8;\n\
\n\
#define TENSORLOGICAL_FP(name0, name1, input_type, copy_type, output_type, out_copy_type, lgc_op) \\\n\
    __kernel void vxcTensorLogical_##name0##_##name1( \\\n\
    __read_only  image2d_array_t in0, \\\n\
    __read_only  image2d_array_t in1, \\\n\
    __write_only image2d_array_t output) \\\n\
{\\\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\\\n\
    input_type vA;\\\n\
    copy_type  src0;\\\n\
    input_type vB;\\\n\
    copy_type  src1;\\\n\
    VXC_ReadImage2DArray(vA,in0,coord,VXC_5BITOFFSET_XY(0,0),VXC_MODIFIER(0,7,0,VXC_RM_TowardZero,0));\\\n\
    _viv_asm(COPY, src0, vA, 16); \\\n\
    VXC_ReadImage2DArray(vB,in1,coord,VXC_5BITOFFSET_XY(0,0),VXC_MODIFIER(0,7,0,VXC_RM_TowardZero,0));\\\n\
    _viv_asm(COPY, src1, vB, 16); \\\n\
    output_type dst; \\\n\
    dst = (src0)lgc_op(src1); \\\n\
    vxc_half8 tmpOut; \\\n\
    VXC_DP2x8(tmpOut,dst,dst,VXC_MODIFIER(0,7,0,VXC_RM_TowardZero, 0),uniMulShortMinus1toFp16_2x8); \\\n\
    out_copy_type data; \\\n\
    _viv_asm(COPY, data, tmpOut, 16); \\\n\
    VXC_WriteImage2DArray(output, coord, data, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
//             name0, name1, input_type, copy_type, output_type, out_copy_type, lgc_op\n\
TENSORLOGICAL(or,    int8,   vxc_char8,   vxc_char8,   vxc_char8,   vxc_char8,   ||)\n\
TENSORLOGICAL(or,    uint8,  vxc_uchar8,  vxc_uchar8,  vxc_char8,   vxc_uchar8,  ||)\n\
TENSORLOGICAL(or,    int16,  vxc_short8,  vxc_short8,  vxc_short8,  vxc_short8,  ||)\n\
TENSORLOGICAL_FP(or,  fp16,  vxc_short8,  vxc_short8,  vxc_short8,  vxc_short8,  ||)\n\
TENSORLOGICAL(and,    int8,   vxc_char8,   vxc_char8,   vxc_char8,   vxc_char8,   &&)\n\
TENSORLOGICAL(and,    uint8,  vxc_uchar8,  vxc_uchar8,  vxc_char8,   vxc_uchar8,  &&)\n\
TENSORLOGICAL(and,    int16,  vxc_short8,  vxc_short8,  vxc_short8,  vxc_short8,  &&)\n\
TENSORLOGICAL_FP(and,  fp16,  vxc_short8,  vxc_short8,  vxc_short8,  vxc_short8,  &&)\n\
//TENSORLOGICAL(or,    fp16,   vxc_short8,  vxc_half8,   vxc_short8,  vxc_short8,  ||)\n\
#endif\n\
"; /* end of vsi_nn_kernel_logical_ops_vx*/

static const char vsi_nn_kernel_lstmunit_activation_BP_F16_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform float logE;\n\
_viv_uniform float twoLogE;\n\
_viv_uniform float forget_bias;\n\
float4 sigmoid(float4 x)\n\
{\n\
    x *= -logE;\n\
    x = 1 + exp2(x);\n\
    return 1 / x;\n\
}\n\
float4 hard_sigmoid(float4 x)\n\
{\n\
    x = 0.2 * x + 0.5;\n\
    x = clamp(x, 0, 1);\n\
    return x;\n\
}\n\
float4 tangentH(float4 x)\n\
{\n\
    x *= -twoLogE;\n\
    x = 1 + exp2(x);\n\
    x = 1 / x;\n\
    return 2 * x - 1;\n\
}\n\
_viv_uniform float outputScale;\n\
_viv_uniform float outputZP;\n\
_viv_uniform VXC_512Bits uniExtract8Data_2x8;\n\
_viv_uniform VXC_512Bits uniFp16toFp32_4x4;\n\
_viv_uniform float4 clip_Min_F;\n\
_viv_uniform float4 clip_Max_F;\n\
_viv_uniform VXC_512Bits uniExtractHalf4_4x4;\n\
_viv_uniform VXC_512Bits uniFp16AddFp16toFp32_4x4;\n\
\n\
#define LSTMUNIT_BP_FP16_FP32(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_BP_F16to##out_type_name##_F32##act_name( \\\n\
    __read_only  image2d_array_t  input_i_conv, \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_array_t  hstate_i_conv, \\\n\
    __read_only  image2d_array_t  hstate_f_conv, \\\n\
    __read_only  image2d_array_t  hstate_c_conv, \\\n\
    __read_only  image2d_array_t  hstate_o_conv, \\\n\
    __read_only  image2d_t        bias_i, \\\n\
    __read_only  image2d_t        bias_f, \\\n\
    __read_only  image2d_t        bias_c, \\\n\
    __read_only  image2d_t        bias_o, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_short8 vect0, vect1, vect2, vect3; \\\n\
    vxc_half8  src0, src1, src2, src3; \\\n\
    vxc_short8 vect10, vect11, vect12, vect13; \\\n\
    vxc_half8  src10, src11, src12, src13; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    float4 b0, b1, b2, b3; \\\n\
    VXC_ReadImage(vect0, input_i_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src0, vect0, 16); \\\n\
    VXC_ReadImage(vect10, hstate_i_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src10, vect10, 16); \\\n\
    VXC_ReadImage(vect1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src1, vect1, 16); \\\n\
    VXC_ReadImage(vect11, hstate_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src11, vect11, 16); \\\n\
    VXC_ReadImage(vect2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src2, vect2, 16); \\\n\
    VXC_ReadImage(vect12, hstate_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src12, vect12, 16); \\\n\
    VXC_ReadImage(vect3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src3, vect3, 16); \\\n\
    VXC_ReadImage(vect13, hstate_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src13, vect13, 16); \\\n\
    data_c_t = read_imagef(cell_state_in, coord_in.zy); \\\n\
    b0 = read_imagef(bias_i, coord_in.xw); \\\n\
    b1 = read_imagef(bias_f, coord_in.xw); \\\n\
    b2 = read_imagef(bias_c, coord_in.xw); \\\n\
    b3 = read_imagef(bias_o, coord_in.xw); \\\n\
 \\\n\
    VXC_DP4x4(data_i_t, src0, src10, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_f_t, src1, src11, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_g_t, src2, src12, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_o_t, src3, src13, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    data_i_t = data_i_t + b0; \\\n\
    data_f_t = data_f_t + b1; \\\n\
    data_g_t = data_g_t + b2; \\\n\
    data_o_t = data_o_t + b3; \\\n\
 \\\n\
    convert_type dst0; \\\n\
    data_i_t = act_func(data_i_t); \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    write_imagef(cell_state_out, coord_in.zy, data_c_t); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_BP_FP16_FP32(F16, ,half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_BP_FP16_FP32(I8,  ,int4,  vxc_char4,  vxc_char4,  sigmoid)\n\
LSTMUNIT_BP_FP16_FP32(U8,  ,int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_BP_FP16_FP32(I16, ,int4,  vxc_short4, vxc_short4, sigmoid)\n\
LSTMUNIT_BP_FP16_FP32(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
LSTMUNIT_BP_FP16_FP32(I8,  _HARD, int4,  vxc_char4,  vxc_char4,  hard_sigmoid)\n\
LSTMUNIT_BP_FP16_FP32(U8,  _HARD, int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_BP_FP16_FP32(I16, _HARD, int4,  vxc_short4, vxc_short4, hard_sigmoid)\n\
\n\
#define LSTMUNIT_BP_FP16_FP16(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_BP_F16to##out_type_name##_F16##act_name( \\\n\
    __read_only  image2d_array_t  input_i_conv, \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_array_t  hstate_i_conv, \\\n\
    __read_only  image2d_array_t  hstate_f_conv, \\\n\
    __read_only  image2d_array_t  hstate_c_conv, \\\n\
    __read_only  image2d_array_t  hstate_o_conv, \\\n\
    __read_only  image2d_t        bias_i, \\\n\
    __read_only  image2d_t        bias_f, \\\n\
    __read_only  image2d_t        bias_c, \\\n\
    __read_only  image2d_t        bias_o, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_short8 vect0, vect1, vect2, vect3, vect4; \\\n\
    vxc_half8  src0, src1, src2, src3, src4; \\\n\
    vxc_short8 vect10, vect11, vect12, vect13; \\\n\
    vxc_half8  src10, src11, src12, src13; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    float4 b0, b1, b2, b3; \\\n\
    VXC_ReadImage(vect0, input_i_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src0, vect0, 16); \\\n\
    VXC_ReadImage(vect10, hstate_i_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src10, vect10, 16); \\\n\
    VXC_ReadImage(vect1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src1, vect1, 16); \\\n\
    VXC_ReadImage(vect11, hstate_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src11, vect11, 16); \\\n\
    VXC_ReadImage(vect2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src2, vect2, 16); \\\n\
    VXC_ReadImage(vect12, hstate_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src12, vect12, 16); \\\n\
    VXC_ReadImage(vect3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src3, vect3, 16); \\\n\
    VXC_ReadImage(vect13, hstate_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src13, vect13, 16); \\\n\
    VXC_ReadImage(vect4, cell_state_in, coord_in.zy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src4, vect4, 16); \\\n\
    b0 = read_imagef(bias_i, coord_in.xw); \\\n\
    b1 = read_imagef(bias_f, coord_in.xw); \\\n\
    b2 = read_imagef(bias_c, coord_in.xw); \\\n\
    b3 = read_imagef(bias_o, coord_in.xw); \\\n\
 \\\n\
    VXC_DP4x4(data_i_t, src0, src10, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_f_t, src1, src11, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_g_t, src2, src12, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_c_t, src4, src4, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_o_t, src3, src13, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    data_i_t = data_i_t + b0; \\\n\
    data_f_t = data_f_t + b1; \\\n\
    data_g_t = data_g_t + b2; \\\n\
    data_o_t = data_o_t + b3; \\\n\
 \\\n\
    convert_type dst0; \\\n\
    half4 dst_cell; \\\n\
    data_i_t = act_func(data_i_t); \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    _viv_asm(CONV, dst_cell, data_c_t); \\\n\
    VXC_DP4x4(src0, dst_cell, dst_cell, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtractHalf4_4x4); \\\n\
    _viv_asm(COPY, vect0, src0, 8); \\\n\
    VXC_WriteImage(cell_state_out, coord_in.zy, vect0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_BP_FP16_FP16(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_BP_FP16_FP16(I8,  , int4,  vxc_char4,  vxc_char4,  sigmoid)\n\
LSTMUNIT_BP_FP16_FP16(U8,  , int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_BP_FP16_FP16(I16, , int4,  vxc_short4, vxc_short4, sigmoid)\n\
LSTMUNIT_BP_FP16_FP16(F16, _HARD,half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
LSTMUNIT_BP_FP16_FP16(I8,  _HARD,int4,  vxc_char4,  vxc_char4,  hard_sigmoid)\n\
LSTMUNIT_BP_FP16_FP16(U8,  _HARD,int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_BP_FP16_FP16(I16, _HARD,int4,  vxc_short4, vxc_short4, hard_sigmoid)\n\
"; /* end of vsi_nn_kernel_lstmunit_activation_BP_F16_vx*/

static const char vsi_nn_kernel_lstmunit_activation_BP_U8_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform float logE;\n\
_viv_uniform float twoLogE;\n\
_viv_uniform float forget_bias;\n\
float4 sigmoid(float4 x)\n\
{\n\
    x *= -logE;\n\
    x = 1 + exp2(x);\n\
    return 1 / x;\n\
}\n\
float4 hard_sigmoid(float4 x)\n\
{\n\
    x = 0.2 * x + 0.5;\n\
    x = clamp(x, 0, 1);\n\
    return x;\n\
}\n\
float4 tangentH(float4 x)\n\
{\n\
    x *= -twoLogE;\n\
    x = 1 + exp2(x);\n\
    x = 1 / x;\n\
    return 2 * x - 1;\n\
}\n\
_viv_uniform float outputScale;\n\
_viv_uniform float outputZP;\n\
_viv_uniform VXC_512Bits uniExtract8Data_2x8;\n\
_viv_uniform VXC_512Bits uniFp16toFp32_4x4;\n\
_viv_uniform float4 clip_Min_F;\n\
_viv_uniform float4 clip_Max_F;\n\
_viv_uniform VXC_512Bits uniExtractHalf4_4x4;\n\
_viv_uniform VXC_512Bits uniU8AddS32_4x4;\n\
_viv_uniform int4 input0Array_ZP;\n\
_viv_uniform int4 input1Array_ZP;\n\
_viv_uniform float4 input0Array_Scale;\n\
_viv_uniform float4 input1Array_Scale;\n\
\n\
#define LSTMUNIT_BP_U8_FP32(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_BP_U8to##out_type_name##_F32##act_name( \\\n\
    __read_only  image2d_array_t  input_i_conv, \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_array_t  hstate_i_conv, \\\n\
    __read_only  image2d_array_t  hstate_f_conv, \\\n\
    __read_only  image2d_array_t  hstate_c_conv, \\\n\
    __read_only  image2d_array_t  hstate_o_conv, \\\n\
    __read_only  image2d_t        bias_i, \\\n\
    __read_only  image2d_t        bias_f, \\\n\
    __read_only  image2d_t        bias_c, \\\n\
    __read_only  image2d_t        bias_o, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_uchar4 src0,  src1,  src2,  src3; \\\n\
    vxc_uchar4 src10, src11, src12, src13; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    float4 b0, b1, b2, b3; \\\n\
    float4 vecA, vecB; \\\n\
    VXC_ReadImage(src0, input_i_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src10, hstate_i_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src11, hstate_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src12, hstate_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src13, hstate_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    data_c_t = read_imagef(cell_state_in, coord_in.zy); \\\n\
    b0 = read_imagef(bias_i, coord_in.xw); \\\n\
    b1 = read_imagef(bias_f, coord_in.xw); \\\n\
    b2 = read_imagef(bias_c, coord_in.xw); \\\n\
    b3 = read_imagef(bias_o, coord_in.xw); \\\n\
 \\\n\
    VXC_DP4x4(vecA, src0, input0Array_ZP.xxxx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src10, input1Array_ZP.xxxx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_i_t = vecA * input0Array_Scale.xxxx + vecB * input1Array_Scale.xxxx + b0; \\\n\
    VXC_DP4x4(vecA, src1, input0Array_ZP.yyyy, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src11, input1Array_ZP.yyyy, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_f_t = vecA * input0Array_Scale.yyyy + vecB * input1Array_Scale.yyyy + b1; \\\n\
    VXC_DP4x4(vecA, src2, input0Array_ZP.zzzz, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src12, input1Array_ZP.zzzz, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_g_t = vecA * input0Array_Scale.zzzz + vecB * input1Array_Scale.zzzz + b2; \\\n\
    VXC_DP4x4(vecA, src3, input0Array_ZP.wwww, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src13, input1Array_ZP.wwww, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_o_t = vecA * input0Array_Scale.wwww + vecB * input1Array_Scale.wwww + b3; \\\n\
 \\\n\
    convert_type dst0; \\\n\
    data_i_t = act_func(data_i_t); \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    write_imagef(cell_state_out, coord_in.zy, data_c_t); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_BP_U8_FP32(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_BP_U8_FP32(I8,  , int4,  vxc_char4,  vxc_char4,  sigmoid)\n\
LSTMUNIT_BP_U8_FP32(U8,  , int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_BP_U8_FP32(I16, , int4,  vxc_short4, vxc_short4, sigmoid)\n\
LSTMUNIT_BP_U8_FP32(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
LSTMUNIT_BP_U8_FP32(I8,  _HARD, int4,  vxc_char4,  vxc_char4,  hard_sigmoid)\n\
LSTMUNIT_BP_U8_FP32(U8,  _HARD, int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_BP_U8_FP32(I16, _HARD, int4,  vxc_short4, vxc_short4, hard_sigmoid)\n\
\n\
#define LSTMUNIT_BP_U8_FP16(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_BP_U8to##out_type_name##_F16##act_name( \\\n\
    __read_only  image2d_array_t  input_i_conv, \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_array_t  hstate_i_conv, \\\n\
    __read_only  image2d_array_t  hstate_f_conv, \\\n\
    __read_only  image2d_array_t  hstate_c_conv, \\\n\
    __read_only  image2d_array_t  hstate_o_conv, \\\n\
    __read_only  image2d_t        bias_i, \\\n\
    __read_only  image2d_t        bias_f, \\\n\
    __read_only  image2d_t        bias_c, \\\n\
    __read_only  image2d_t        bias_o, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_short8 vect0; \\\n\
    vxc_half8  src4; \\\n\
    vxc_uchar4 src0,  src1,  src2,  src3; \\\n\
    vxc_uchar4 src10, src11, src12, src13; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    float4 b0, b1, b2, b3; \\\n\
    float4 vecA, vecB; \\\n\
    VXC_ReadImage(src0, input_i_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src10, hstate_i_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src11, hstate_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src12, hstate_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src13, hstate_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(vect0, cell_state_in, coord_in.zy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src4, vect0, 16); \\\n\
    b0 = read_imagef(bias_i, coord_in.xw); \\\n\
    b1 = read_imagef(bias_f, coord_in.xw); \\\n\
    b2 = read_imagef(bias_c, coord_in.xw); \\\n\
    b3 = read_imagef(bias_o, coord_in.xw); \\\n\
 \\\n\
    VXC_DP4x4(data_c_t, src4, src4, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(vecA, src0, input0Array_ZP.xxxx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src10, input1Array_ZP.xxxx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_i_t = vecA * input0Array_Scale.xxxx + vecB * input1Array_Scale.xxxx + b0; \\\n\
    VXC_DP4x4(vecA, src1, input0Array_ZP.yyyy, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src11, input1Array_ZP.yyyy, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_f_t = vecA * input0Array_Scale.yyyy + vecB * input1Array_Scale.yyyy + b1; \\\n\
    VXC_DP4x4(vecA, src2, input0Array_ZP.zzzz, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src12, input1Array_ZP.zzzz, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_g_t = vecA * input0Array_Scale.zzzz + vecB * input1Array_Scale.zzzz + b2; \\\n\
    VXC_DP4x4(vecA, src3, input0Array_ZP.wwww, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src13, input1Array_ZP.wwww, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_o_t = vecA * input0Array_Scale.wwww + vecB * input1Array_Scale.wwww + b3; \\\n\
 \\\n\
    convert_type dst0; \\\n\
    half4 dst_cell; \\\n\
    data_i_t = act_func(data_i_t); \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    _viv_asm(CONV, dst_cell, data_c_t); \\\n\
    VXC_DP4x4(src4, dst_cell, dst_cell, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtractHalf4_4x4); \\\n\
    _viv_asm(COPY, vect0, src4, 8); \\\n\
    VXC_WriteImage(cell_state_out, coord_in.zy, vect0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_BP_U8_FP16(F16, ,half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_BP_U8_FP16(I8,  ,int4,  vxc_char4,  vxc_char4,  sigmoid)\n\
LSTMUNIT_BP_U8_FP16(U8,  ,int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_BP_U8_FP16(I16, ,int4,  vxc_short4, vxc_short4, sigmoid)\n\
LSTMUNIT_BP_U8_FP16(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
LSTMUNIT_BP_U8_FP16(I8,  _HARD, int4,  vxc_char4,  vxc_char4,  hard_sigmoid)\n\
LSTMUNIT_BP_U8_FP16(U8,  _HARD, int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_BP_U8_FP16(I16, _HARD, int4,  vxc_short4, vxc_short4, hard_sigmoid)\n\
"; /* end of vsi_nn_kernel_lstmunit_activation_BP_U8_vx*/

static const char vsi_nn_kernel_lstmunit_activation_B_F16_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform float logE;\n\
_viv_uniform float twoLogE;\n\
_viv_uniform float forget_bias;\n\
float4 sigmoid(float4 x)\n\
{\n\
    x *= -logE;\n\
    x = 1 + exp2(x);\n\
    return 1 / x;\n\
}\n\
float4 hard_sigmoid(float4 x)\n\
{\n\
    x = 0.2 * x + 0.5;\n\
    x = clamp(x, 0, 1);\n\
    return x;\n\
}\n\
float4 tangentH(float4 x)\n\
{\n\
    x *= -twoLogE;\n\
    x = 1 + exp2(x);\n\
    x = 1 / x;\n\
    return 2 * x - 1;\n\
}\n\
_viv_uniform float outputScale;\n\
_viv_uniform float outputZP;\n\
_viv_uniform VXC_512Bits uniExtract8Data_2x8;\n\
_viv_uniform VXC_512Bits uniFp16toFp32_4x4;\n\
_viv_uniform float4 clip_Min_F;\n\
_viv_uniform float4 clip_Max_F;\n\
_viv_uniform VXC_512Bits uniExtractHalf4_4x4;\n\
_viv_uniform VXC_512Bits uniFp16AddFp16toFp32_4x4;\n\
\n\
#define LSTMUNIT_B_FP16_FP32(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_B_F16to##out_type_name##_F32##act_name( \\\n\
    __read_only  image2d_array_t  input_i_conv, \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_array_t  hstate_i_conv, \\\n\
    __read_only  image2d_array_t  hstate_f_conv, \\\n\
    __read_only  image2d_array_t  hstate_c_conv, \\\n\
    __read_only  image2d_array_t  hstate_o_conv, \\\n\
    __read_only  image2d_t        bias_i, \\\n\
    __read_only  image2d_t        bias_f, \\\n\
    __read_only  image2d_t        bias_c, \\\n\
    __read_only  image2d_t        bias_o, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __write_only image2d_t        h_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_short8 vect0, vect1, vect2, vect3; \\\n\
    vxc_half8  src0, src1, src2, src3; \\\n\
    vxc_short8 vect10, vect11, vect12, vect13; \\\n\
    vxc_half8  src10, src11, src12, src13; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    float4 b0, b1, b2, b3; \\\n\
    VXC_ReadImage(vect0, input_i_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src0, vect0, 16); \\\n\
    VXC_ReadImage(vect10, hstate_i_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src10, vect10, 16); \\\n\
    VXC_ReadImage(vect1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src1, vect1, 16); \\\n\
    VXC_ReadImage(vect11, hstate_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src11, vect11, 16); \\\n\
    VXC_ReadImage(vect2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src2, vect2, 16); \\\n\
    VXC_ReadImage(vect12, hstate_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src12, vect12, 16); \\\n\
    VXC_ReadImage(vect3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src3, vect3, 16); \\\n\
    VXC_ReadImage(vect13, hstate_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src13, vect13, 16); \\\n\
    data_c_t = read_imagef(cell_state_in, coord_in.zy); \\\n\
    b0 = read_imagef(bias_i, coord_in.xw); \\\n\
    b1 = read_imagef(bias_f, coord_in.xw); \\\n\
    b2 = read_imagef(bias_c, coord_in.xw); \\\n\
    b3 = read_imagef(bias_o, coord_in.xw); \\\n\
 \\\n\
    VXC_DP4x4(data_i_t, src0, src10, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_f_t, src1, src11, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_g_t, src2, src12, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_o_t, src3, src13, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    data_i_t = data_i_t + b0; \\\n\
    data_f_t = data_f_t + b1; \\\n\
    data_g_t = data_g_t + b2; \\\n\
    data_o_t = data_o_t + b3; \\\n\
 \\\n\
    convert_type dst0; \\\n\
    data_i_t = act_func(data_i_t); \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    write_imagef(cell_state_out, coord_in.zy, data_c_t); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_WriteImage(h_state_out, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_B_FP16_FP32(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_B_FP16_FP32(I8,  , int4,  vxc_char4,  vxc_char4,  sigmoid)\n\
LSTMUNIT_B_FP16_FP32(U8,  , int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_B_FP16_FP32(I16, , int4,  vxc_short4, vxc_short4, sigmoid)\n\
LSTMUNIT_B_FP16_FP32(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
LSTMUNIT_B_FP16_FP32(I8,  _HARD, int4,  vxc_char4,  vxc_char4,  hard_sigmoid)\n\
LSTMUNIT_B_FP16_FP32(U8,  _HARD, int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_B_FP16_FP32(I16, _HARD, int4,  vxc_short4, vxc_short4, hard_sigmoid)\n\
\n\
#define LSTMUNIT_B_FP16_FP16(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_B_F16to##out_type_name##_F16##act_name( \\\n\
    __read_only  image2d_array_t  input_i_conv, \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_array_t  hstate_i_conv, \\\n\
    __read_only  image2d_array_t  hstate_f_conv, \\\n\
    __read_only  image2d_array_t  hstate_c_conv, \\\n\
    __read_only  image2d_array_t  hstate_o_conv, \\\n\
    __read_only  image2d_t        bias_i, \\\n\
    __read_only  image2d_t        bias_f, \\\n\
    __read_only  image2d_t        bias_c, \\\n\
    __read_only  image2d_t        bias_o, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __write_only image2d_t        h_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_short8 vect0, vect1, vect2, vect3, vect4; \\\n\
    vxc_half8  src0, src1, src2, src3, src4; \\\n\
    vxc_short8 vect10, vect11, vect12, vect13; \\\n\
    vxc_half8  src10, src11, src12, src13; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    float4 b0, b1, b2, b3; \\\n\
    VXC_ReadImage(vect0, input_i_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src0, vect0, 16); \\\n\
    VXC_ReadImage(vect10, hstate_i_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src10, vect10, 16); \\\n\
    VXC_ReadImage(vect1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src1, vect1, 16); \\\n\
    VXC_ReadImage(vect11, hstate_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src11, vect11, 16); \\\n\
    VXC_ReadImage(vect2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src2, vect2, 16); \\\n\
    VXC_ReadImage(vect12, hstate_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src12, vect12, 16); \\\n\
    VXC_ReadImage(vect3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src3, vect3, 16); \\\n\
    VXC_ReadImage(vect13, hstate_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src13, vect13, 16); \\\n\
    VXC_ReadImage(vect4, cell_state_in, coord_in.zy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src4, vect4, 16); \\\n\
    b0 = read_imagef(bias_i, coord_in.xw); \\\n\
    b1 = read_imagef(bias_f, coord_in.xw); \\\n\
    b2 = read_imagef(bias_c, coord_in.xw); \\\n\
    b3 = read_imagef(bias_o, coord_in.xw); \\\n\
 \\\n\
    VXC_DP4x4(data_i_t, src0, src10, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_f_t, src1, src11, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_g_t, src2, src12, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_c_t, src4, src4, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_o_t, src3, src13, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    data_i_t = data_i_t + b0; \\\n\
    data_f_t = data_f_t + b1; \\\n\
    data_g_t = data_g_t + b2; \\\n\
    data_o_t = data_o_t + b3; \\\n\
 \\\n\
    convert_type dst0; \\\n\
    half4 dst_cell; \\\n\
    data_i_t = act_func(data_i_t); \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    _viv_asm(CONV, dst_cell, data_c_t); \\\n\
    VXC_DP4x4(src0, dst_cell, dst_cell, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtractHalf4_4x4); \\\n\
    _viv_asm(COPY, vect0, src0, 8); \\\n\
    VXC_WriteImage(cell_state_out, coord_in.zy, vect0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_WriteImage(h_state_out, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_B_FP16_FP16(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_B_FP16_FP16(I8,  , int4,  vxc_char4,  vxc_char4,  sigmoid)\n\
LSTMUNIT_B_FP16_FP16(U8,  , int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_B_FP16_FP16(I16, , int4,  vxc_short4, vxc_short4, sigmoid)\n\
LSTMUNIT_B_FP16_FP16(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
LSTMUNIT_B_FP16_FP16(I8,  _HARD, int4,  vxc_char4,  vxc_char4,  hard_sigmoid)\n\
LSTMUNIT_B_FP16_FP16(U8,  _HARD, int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_B_FP16_FP16(I16, _HARD, int4,  vxc_short4, vxc_short4, hard_sigmoid)\n\
"; /* end of vsi_nn_kernel_lstmunit_activation_B_F16_vx*/

static const char vsi_nn_kernel_lstmunit_activation_B_U8_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform float logE;\n\
_viv_uniform float twoLogE;\n\
_viv_uniform float forget_bias;\n\
float4 sigmoid(float4 x)\n\
{\n\
    x *= -logE;\n\
    x = 1 + exp2(x);\n\
    return 1 / x;\n\
}\n\
float4 hard_sigmoid(float4 x)\n\
{\n\
    x = 0.2 * x + 0.5;\n\
    x = clamp(x, 0, 1);\n\
    return x;\n\
}\n\
float4 tangentH(float4 x)\n\
{\n\
    x *= -twoLogE;\n\
    x = 1 + exp2(x);\n\
    x = 1 / x;\n\
    return 2 * x - 1;\n\
}\n\
_viv_uniform float outputScale;\n\
_viv_uniform float outputZP;\n\
_viv_uniform VXC_512Bits uniExtract8Data_2x8;\n\
_viv_uniform VXC_512Bits uniFp16toFp32_4x4;\n\
_viv_uniform float4 clip_Min_F;\n\
_viv_uniform float4 clip_Max_F;\n\
_viv_uniform VXC_512Bits uniExtractHalf4_4x4;\n\
_viv_uniform VXC_512Bits uniU8AddS32_4x4;\n\
_viv_uniform int4 input0Array_ZP;\n\
_viv_uniform int4 input1Array_ZP;\n\
_viv_uniform float4 input0Array_Scale;\n\
_viv_uniform float4 input1Array_Scale;\n\
\n\
#define LSTMUNIT_B_U8_FP32(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_B_U8to##out_type_name##_F32##act_name( \\\n\
    __read_only  image2d_array_t  input_i_conv, \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_array_t  hstate_i_conv, \\\n\
    __read_only  image2d_array_t  hstate_f_conv, \\\n\
    __read_only  image2d_array_t  hstate_c_conv, \\\n\
    __read_only  image2d_array_t  hstate_o_conv, \\\n\
    __read_only  image2d_t        bias_i, \\\n\
    __read_only  image2d_t        bias_f, \\\n\
    __read_only  image2d_t        bias_c, \\\n\
    __read_only  image2d_t        bias_o, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __write_only image2d_t        h_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_uchar4 src0,  src1,  src2,  src3; \\\n\
    vxc_uchar4 src10, src11, src12, src13; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    float4 b0, b1, b2, b3; \\\n\
    float4 vecA, vecB; \\\n\
    VXC_ReadImage(src0, input_i_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src10, hstate_i_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src11, hstate_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src12, hstate_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src13, hstate_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    data_c_t = read_imagef(cell_state_in, coord_in.zy); \\\n\
    b0 = read_imagef(bias_i, coord_in.xw); \\\n\
    b1 = read_imagef(bias_f, coord_in.xw); \\\n\
    b2 = read_imagef(bias_c, coord_in.xw); \\\n\
    b3 = read_imagef(bias_o, coord_in.xw); \\\n\
 \\\n\
    VXC_DP4x4(vecA, src0, input0Array_ZP.xxxx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src10, input1Array_ZP.xxxx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_i_t = vecA * input0Array_Scale.xxxx + vecB * input1Array_Scale.xxxx + b0; \\\n\
    VXC_DP4x4(vecA, src1, input0Array_ZP.yyyy, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src11, input1Array_ZP.yyyy, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_f_t = vecA * input0Array_Scale.yyyy + vecB * input1Array_Scale.yyyy + b1; \\\n\
    VXC_DP4x4(vecA, src2, input0Array_ZP.zzzz, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src12, input1Array_ZP.zzzz, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_g_t = vecA * input0Array_Scale.zzzz + vecB * input1Array_Scale.zzzz + b2; \\\n\
    VXC_DP4x4(vecA, src3, input0Array_ZP.wwww, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src13, input1Array_ZP.wwww, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_o_t = vecA * input0Array_Scale.wwww + vecB * input1Array_Scale.wwww + b3; \\\n\
 \\\n\
    convert_type dst0; \\\n\
    data_i_t = act_func(data_i_t); \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    write_imagef(cell_state_out, coord_in.zy, data_c_t); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_WriteImage(h_state_out, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
\n\
LSTMUNIT_B_U8_FP32(U8,  , int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_B_U8_FP32(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_B_U8_FP32(U8,  _HARD, int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_B_U8_FP32(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
\n\
#define LSTMUNIT_B_U8_FP16(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_B_U8to##out_type_name##_F16##act_name( \\\n\
    __read_only  image2d_array_t  input_i_conv, \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_array_t  hstate_i_conv, \\\n\
    __read_only  image2d_array_t  hstate_f_conv, \\\n\
    __read_only  image2d_array_t  hstate_c_conv, \\\n\
    __read_only  image2d_array_t  hstate_o_conv, \\\n\
    __read_only  image2d_t        bias_i, \\\n\
    __read_only  image2d_t        bias_f, \\\n\
    __read_only  image2d_t        bias_c, \\\n\
    __read_only  image2d_t        bias_o, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __write_only image2d_t        h_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_short8 vect0; \\\n\
    vxc_half8  src4; \\\n\
    vxc_uchar4 src0,  src1,  src2,  src3; \\\n\
    vxc_uchar4 src10, src11, src12, src13; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    float4 b0, b1, b2, b3; \\\n\
    float4 vecA, vecB; \\\n\
    VXC_ReadImage(src0, input_i_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src10, hstate_i_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src11, hstate_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src12, hstate_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src13, hstate_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(vect0, cell_state_in, coord_in.zy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src4, vect0, 16); \\\n\
    b0 = read_imagef(bias_i, coord_in.xw); \\\n\
    b1 = read_imagef(bias_f, coord_in.xw); \\\n\
    b2 = read_imagef(bias_c, coord_in.xw); \\\n\
    b3 = read_imagef(bias_o, coord_in.xw); \\\n\
 \\\n\
    VXC_DP4x4(data_c_t, src4, src4, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(vecA, src0, input0Array_ZP.xxxx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src10, input1Array_ZP.xxxx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_i_t = vecA * input0Array_Scale.xxxx + vecB * input1Array_Scale.xxxx + b0; \\\n\
    VXC_DP4x4(vecA, src1, input0Array_ZP.yyyy, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src11, input1Array_ZP.yyyy, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_f_t = vecA * input0Array_Scale.yyyy + vecB * input1Array_Scale.yyyy + b1; \\\n\
    VXC_DP4x4(vecA, src2, input0Array_ZP.zzzz, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src12, input1Array_ZP.zzzz, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_g_t = vecA * input0Array_Scale.zzzz + vecB * input1Array_Scale.zzzz + b2; \\\n\
    VXC_DP4x4(vecA, src3, input0Array_ZP.wwww, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src13, input1Array_ZP.wwww, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_o_t = vecA * input0Array_Scale.wwww + vecB * input1Array_Scale.wwww + b3; \\\n\
 \\\n\
    convert_type dst0; \\\n\
    half4 dst_cell; \\\n\
    data_i_t = act_func(data_i_t); \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    _viv_asm(CONV, dst_cell, data_c_t); \\\n\
    VXC_DP4x4(src4, dst_cell, dst_cell, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtractHalf4_4x4); \\\n\
    _viv_asm(COPY, vect0, src4, 8); \\\n\
    VXC_WriteImage(cell_state_out, coord_in.zy, vect0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_WriteImage(h_state_out, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_B_U8_FP16(U8,  , int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_B_U8_FP16(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_B_U8_FP16(U8,  _HARD, int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_B_U8_FP16(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
"; /* end of vsi_nn_kernel_lstmunit_activation_B_U8_vx*/

static const char vsi_nn_kernel_lstmunit_activation_CBP_F16_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform float logE;\n\
_viv_uniform float twoLogE;\n\
_viv_uniform float forget_bias;\n\
float4 sigmoid(float4 x)\n\
{\n\
    x *= -logE;\n\
    x = 1 + exp2(x);\n\
    return 1 / x;\n\
}\n\
float4 hard_sigmoid(float4 x)\n\
{\n\
    x = 0.2 * x + 0.5;\n\
    x = clamp(x, 0, 1);\n\
    return x;\n\
}\n\
float4 tangentH(float4 x)\n\
{\n\
    x *= -twoLogE;\n\
    x = 1 + exp2(x);\n\
    x = 1 / x;\n\
    return 2 * x - 1;\n\
}\n\
_viv_uniform float outputScale;\n\
_viv_uniform float outputZP;\n\
_viv_uniform VXC_512Bits uniExtract8Data_2x8;\n\
_viv_uniform VXC_512Bits uniFp16toFp32_4x4;\n\
_viv_uniform float4 clip_Min_F;\n\
_viv_uniform float4 clip_Max_F;\n\
_viv_uniform VXC_512Bits uniExtractHalf4_4x4;\n\
_viv_uniform VXC_512Bits uniFp16AddFp16toFp32_4x4;\n\
\n\
#define LSTMUNIT_CBP_FP16_FP32(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_CBP_F16to##out_type_name##_F32##act_name( \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_array_t  hstate_f_conv, \\\n\
    __read_only  image2d_array_t  hstate_c_conv, \\\n\
    __read_only  image2d_array_t  hstate_o_conv, \\\n\
    __read_only  image2d_t        bias_f, \\\n\
    __read_only  image2d_t        bias_c, \\\n\
    __read_only  image2d_t        bias_o, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_short8 vect0, vect1, vect2, vect3; \\\n\
    vxc_half8  src0, src1, src2, src3; \\\n\
    vxc_short8 vect10, vect11, vect12, vect13; \\\n\
    vxc_half8  src10, src11, src12, src13; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    float4 b0, b1, b2, b3; \\\n\
    VXC_ReadImage(vect1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src1, vect1, 16); \\\n\
    VXC_ReadImage(vect11, hstate_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src11, vect11, 16); \\\n\
    VXC_ReadImage(vect2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src2, vect2, 16); \\\n\
    VXC_ReadImage(vect12, hstate_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src12, vect12, 16); \\\n\
    VXC_ReadImage(vect3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src3, vect3, 16); \\\n\
    VXC_ReadImage(vect13, hstate_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src13, vect13, 16); \\\n\
    data_c_t = read_imagef(cell_state_in, coord_in.zy); \\\n\
    b1 = read_imagef(bias_f, coord_in.xw); \\\n\
    b2 = read_imagef(bias_c, coord_in.xw); \\\n\
    b3 = read_imagef(bias_o, coord_in.xw); \\\n\
 \\\n\
    VXC_DP4x4(data_f_t, src1, src11, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_g_t, src2, src12, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_o_t, src3, src13, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    data_f_t = data_f_t + b1; \\\n\
    data_g_t = data_g_t + b2; \\\n\
    data_o_t = data_o_t + b3; \\\n\
 \\\n\
    convert_type dst0; \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = 1.0 - data_f_t; \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    write_imagef(cell_state_out, coord_in.zy, data_c_t); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_CBP_FP16_FP32(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_CBP_FP16_FP32(I8,  , int4,  vxc_char4,  vxc_char4,  sigmoid)\n\
LSTMUNIT_CBP_FP16_FP32(U8,  , int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_CBP_FP16_FP32(I16, , int4,  vxc_short4, vxc_short4, sigmoid)\n\
LSTMUNIT_CBP_FP16_FP32(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
LSTMUNIT_CBP_FP16_FP32(I8,  _HARD, int4,  vxc_char4,  vxc_char4,  hard_sigmoid)\n\
LSTMUNIT_CBP_FP16_FP32(U8,  _HARD, int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_CBP_FP16_FP32(I16, _HARD, int4,  vxc_short4, vxc_short4, hard_sigmoid)\n\
\n\
#define LSTMUNIT_CBP_FP16_FP16(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_CBP_F16to##out_type_name##_F16##act_name( \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_array_t  hstate_f_conv, \\\n\
    __read_only  image2d_array_t  hstate_c_conv, \\\n\
    __read_only  image2d_array_t  hstate_o_conv, \\\n\
    __read_only  image2d_t        bias_f, \\\n\
    __read_only  image2d_t        bias_c, \\\n\
    __read_only  image2d_t        bias_o, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_short8 vect0, vect1, vect2, vect3, vect4; \\\n\
    vxc_half8  src0, src1, src2, src3, src4; \\\n\
    vxc_short8 vect10, vect11, vect12, vect13; \\\n\
    vxc_half8  src10, src11, src12, src13; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    float4 b0, b1, b2, b3; \\\n\
    VXC_ReadImage(vect1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src1, vect1, 16); \\\n\
    VXC_ReadImage(vect11, hstate_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src11, vect11, 16); \\\n\
    VXC_ReadImage(vect2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src2, vect2, 16); \\\n\
    VXC_ReadImage(vect12, hstate_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src12, vect12, 16); \\\n\
    VXC_ReadImage(vect3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src3, vect3, 16); \\\n\
    VXC_ReadImage(vect13, hstate_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src13, vect13, 16); \\\n\
    VXC_ReadImage(vect4, cell_state_in, coord_in.zy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src4, vect4, 16); \\\n\
    b1 = read_imagef(bias_f, coord_in.xw); \\\n\
    b2 = read_imagef(bias_c, coord_in.xw); \\\n\
    b3 = read_imagef(bias_o, coord_in.xw); \\\n\
 \\\n\
    VXC_DP4x4(data_f_t, src1, src11, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_g_t, src2, src12, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_c_t, src4, src4, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_o_t, src3, src13, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    data_f_t = data_f_t + b1; \\\n\
    data_g_t = data_g_t + b2; \\\n\
    data_o_t = data_o_t + b3; \\\n\
 \\\n\
    convert_type dst0; \\\n\
    half4 dst_cell; \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = 1.0 - data_f_t; \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    _viv_asm(CONV, dst_cell, data_c_t); \\\n\
    VXC_DP4x4(src0, dst_cell, dst_cell, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtractHalf4_4x4); \\\n\
    _viv_asm(COPY, vect0, src0, 8); \\\n\
    VXC_WriteImage(cell_state_out, coord_in.zy, vect0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_CBP_FP16_FP16(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_CBP_FP16_FP16(I8,  , int4,  vxc_char4,  vxc_char4,  sigmoid)\n\
LSTMUNIT_CBP_FP16_FP16(U8,  , int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_CBP_FP16_FP16(I16, , int4,  vxc_short4, vxc_short4, sigmoid)\n\
LSTMUNIT_CBP_FP16_FP16(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
LSTMUNIT_CBP_FP16_FP16(I8,  _HARD, int4,  vxc_char4,  vxc_char4,  hard_sigmoid)\n\
LSTMUNIT_CBP_FP16_FP16(U8,  _HARD, int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_CBP_FP16_FP16(I16, _HARD, int4,  vxc_short4, vxc_short4, hard_sigmoid)\n\
"; /* end of vsi_nn_kernel_lstmunit_activation_CBP_F16_vx*/

static const char vsi_nn_kernel_lstmunit_activation_CBP_U8_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform float logE;\n\
_viv_uniform float twoLogE;\n\
_viv_uniform float forget_bias;\n\
float4 sigmoid(float4 x)\n\
{\n\
    x *= -logE;\n\
    x = 1 + exp2(x);\n\
    return 1 / x;\n\
}\n\
float4 hard_sigmoid(float4 x)\n\
{\n\
    x = 0.2 * x + 0.5;\n\
    x = clamp(x, 0, 1);\n\
    return x;\n\
}\n\
float4 tangentH(float4 x)\n\
{\n\
    x *= -twoLogE;\n\
    x = 1 + exp2(x);\n\
    x = 1 / x;\n\
    return 2 * x - 1;\n\
}\n\
_viv_uniform float outputScale;\n\
_viv_uniform float outputZP;\n\
_viv_uniform VXC_512Bits uniExtract8Data_2x8;\n\
_viv_uniform VXC_512Bits uniFp16toFp32_4x4;\n\
_viv_uniform float4 clip_Min_F;\n\
_viv_uniform float4 clip_Max_F;\n\
_viv_uniform VXC_512Bits uniExtractHalf4_4x4;\n\
_viv_uniform VXC_512Bits uniU8AddS32_4x4;\n\
_viv_uniform int4 input0Array_ZP;\n\
_viv_uniform int4 input1Array_ZP;\n\
_viv_uniform float4 input0Array_Scale;\n\
_viv_uniform float4 input1Array_Scale;\n\
\n\
#define LSTMUNIT_CBP_U8_FP32(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_CBP_U8to##out_type_name##_F32##act_name( \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_array_t  hstate_f_conv, \\\n\
    __read_only  image2d_array_t  hstate_c_conv, \\\n\
    __read_only  image2d_array_t  hstate_o_conv, \\\n\
    __read_only  image2d_t        bias_f, \\\n\
    __read_only  image2d_t        bias_c, \\\n\
    __read_only  image2d_t        bias_o, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_uchar4 src0,  src1,  src2,  src3; \\\n\
    vxc_uchar4 src10, src11, src12, src13; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    float4 vecA, vecB; \\\n\
    float4 b0, b1, b2, b3; \\\n\
    VXC_ReadImage(src1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src11, hstate_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src12, hstate_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src13, hstate_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    data_c_t = read_imagef(cell_state_in, coord_in.zy); \\\n\
    b1 = read_imagef(bias_f, coord_in.xw); \\\n\
    b2 = read_imagef(bias_c, coord_in.xw); \\\n\
    b3 = read_imagef(bias_o, coord_in.xw); \\\n\
    VXC_DP4x4(vecA, src1, input0Array_ZP.yyyy, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src11, input1Array_ZP.yyyy, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_f_t = vecA * input0Array_Scale.yyyy + vecB * input1Array_Scale.yyyy + b1; \\\n\
    VXC_DP4x4(vecA, src2, input0Array_ZP.zzzz, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src12, input1Array_ZP.zzzz, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_g_t = vecA * input0Array_Scale.zzzz + vecB * input1Array_Scale.zzzz + b2; \\\n\
    VXC_DP4x4(vecA, src3, input0Array_ZP.wwww, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src13, input1Array_ZP.wwww, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_o_t = vecA * input0Array_Scale.wwww + vecB * input1Array_Scale.wwww + b3; \\\n\
 \\\n\
    convert_type dst0; \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = 1.0 - data_f_t; \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    write_imagef(cell_state_out, coord_in.zy, data_c_t); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_CBP_U8_FP32(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_CBP_U8_FP32(I8,  , int4,  vxc_char4,  vxc_char4,  sigmoid)\n\
LSTMUNIT_CBP_U8_FP32(U8,  , int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_CBP_U8_FP32(I16, , int4,  vxc_short4, vxc_short4, sigmoid)\n\
LSTMUNIT_CBP_U8_FP32(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
LSTMUNIT_CBP_U8_FP32(I8,  _HARD, int4,  vxc_char4,  vxc_char4,  hard_sigmoid)\n\
LSTMUNIT_CBP_U8_FP32(U8,  _HARD, int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_CBP_U8_FP32(I16, _HARD, int4,  vxc_short4, vxc_short4, hard_sigmoid)\n\
\n\
#define LSTMUNIT_CBP_U8_FP16(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_CBP_U8to##out_type_name##_F16##act_name( \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_array_t  hstate_f_conv, \\\n\
    __read_only  image2d_array_t  hstate_c_conv, \\\n\
    __read_only  image2d_array_t  hstate_o_conv, \\\n\
    __read_only  image2d_t        bias_f, \\\n\
    __read_only  image2d_t        bias_c, \\\n\
    __read_only  image2d_t        bias_o, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_short8 vect0; \\\n\
    vxc_half8  src4; \\\n\
    vxc_uchar4 src0,  src1,  src2,  src3; \\\n\
    vxc_uchar4 src10, src11, src12, src13; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    float4 b0, b1, b2, b3; \\\n\
    float4 vecA, vecB; \\\n\
    VXC_ReadImage(src1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src11, hstate_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src12, hstate_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src13, hstate_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(vect0, cell_state_in, coord_in.zy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src4, vect0, 16); \\\n\
    b1 = read_imagef(bias_f, coord_in.xw); \\\n\
    b2 = read_imagef(bias_c, coord_in.xw); \\\n\
    b3 = read_imagef(bias_o, coord_in.xw); \\\n\
 \\\n\
    VXC_DP4x4(data_c_t, src4, src4, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(vecA, src1, input0Array_ZP.yyyy, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src11, input1Array_ZP.yyyy, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_f_t = vecA * input0Array_Scale.yyyy + vecB * input1Array_Scale.yyyy + b1; \\\n\
    VXC_DP4x4(vecA, src2, input0Array_ZP.zzzz, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src12, input1Array_ZP.zzzz, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_g_t = vecA * input0Array_Scale.zzzz + vecB * input1Array_Scale.zzzz + b2; \\\n\
    VXC_DP4x4(vecA, src3, input0Array_ZP.wwww, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src13, input1Array_ZP.wwww, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_o_t = vecA * input0Array_Scale.wwww + vecB * input1Array_Scale.wwww + b3; \\\n\
 \\\n\
    convert_type dst0; \\\n\
    half4 dst_cell; \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = 1.0 - data_f_t; \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    _viv_asm(CONV, dst_cell, data_c_t); \\\n\
    VXC_DP4x4(src4, dst_cell, dst_cell, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtractHalf4_4x4); \\\n\
    _viv_asm(COPY, vect0, src4, 8); \\\n\
    VXC_WriteImage(cell_state_out, coord_in.zy, vect0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_CBP_U8_FP16(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_CBP_U8_FP16(I8,  , int4,  vxc_char4,  vxc_char4,  sigmoid)\n\
LSTMUNIT_CBP_U8_FP16(U8,  , int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_CBP_U8_FP16(I16, , int4,  vxc_short4, vxc_short4, sigmoid)\n\
LSTMUNIT_CBP_U8_FP16(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
LSTMUNIT_CBP_U8_FP16(I8,  _HARD, int4,  vxc_char4,  vxc_char4,  hard_sigmoid)\n\
LSTMUNIT_CBP_U8_FP16(U8,  _HARD, int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_CBP_U8_FP16(I16, _HARD, int4,  vxc_short4, vxc_short4, hard_sigmoid)\n\
"; /* end of vsi_nn_kernel_lstmunit_activation_CBP_U8_vx*/

static const char vsi_nn_kernel_lstmunit_activation_CB_F16_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform float logE;\n\
_viv_uniform float twoLogE;\n\
_viv_uniform float forget_bias;\n\
float4 sigmoid(float4 x)\n\
{\n\
    x *= -logE;\n\
    x = 1 + exp2(x);\n\
    return 1 / x;\n\
}\n\
float4 hard_sigmoid(float4 x)\n\
{\n\
    x = 0.2 * x + 0.5;\n\
    x = clamp(x, 0, 1);\n\
    return x;\n\
}\n\
float4 tangentH(float4 x)\n\
{\n\
    x *= -twoLogE;\n\
    x = 1 + exp2(x);\n\
    x = 1 / x;\n\
    return 2 * x - 1;\n\
}\n\
_viv_uniform float outputScale;\n\
_viv_uniform float outputZP;\n\
_viv_uniform VXC_512Bits uniExtract8Data_2x8;\n\
_viv_uniform VXC_512Bits uniFp16toFp32_4x4;\n\
_viv_uniform float4 clip_Min_F;\n\
_viv_uniform float4 clip_Max_F;\n\
_viv_uniform VXC_512Bits uniExtractHalf4_4x4;\n\
_viv_uniform VXC_512Bits uniFp16AddFp16toFp32_4x4;\n\
\n\
#define LSTMUNIT_CB_FP16_FP32(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_CB_F16to##out_type_name##_F32##act_name( \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_array_t  hstate_f_conv, \\\n\
    __read_only  image2d_array_t  hstate_c_conv, \\\n\
    __read_only  image2d_array_t  hstate_o_conv, \\\n\
    __read_only  image2d_t        bias_f, \\\n\
    __read_only  image2d_t        bias_c, \\\n\
    __read_only  image2d_t        bias_o, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __write_only image2d_t        h_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_short8 vect0, vect1, vect2, vect3; \\\n\
    vxc_half8  src0, src1, src2, src3; \\\n\
    vxc_short8 vect10, vect11, vect12, vect13; \\\n\
    vxc_half8  src10, src11, src12, src13; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    float4 b0, b1, b2, b3; \\\n\
    VXC_ReadImage(vect1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src1, vect1, 16); \\\n\
    VXC_ReadImage(vect11, hstate_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src11, vect11, 16); \\\n\
    VXC_ReadImage(vect2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src2, vect2, 16); \\\n\
    VXC_ReadImage(vect12, hstate_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src12, vect12, 16); \\\n\
    VXC_ReadImage(vect3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src3, vect3, 16); \\\n\
    VXC_ReadImage(vect13, hstate_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src13, vect13, 16); \\\n\
    data_c_t = read_imagef(cell_state_in, coord_in.zy); \\\n\
    b1 = read_imagef(bias_f, coord_in.xw); \\\n\
    b2 = read_imagef(bias_c, coord_in.xw); \\\n\
    b3 = read_imagef(bias_o, coord_in.xw); \\\n\
 \\\n\
    VXC_DP4x4(data_f_t, src1, src11, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_g_t, src2, src12, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_o_t, src3, src13, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    data_f_t = data_f_t + b1; \\\n\
    data_g_t = data_g_t + b2; \\\n\
    data_o_t = data_o_t + b3; \\\n\
 \\\n\
    convert_type dst0; \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = 1.0 - data_f_t; \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    write_imagef(cell_state_out, coord_in.zy, data_c_t); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_WriteImage(h_state_out, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_CB_FP16_FP32(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_CB_FP16_FP32(I8,  , int4,  vxc_char4,  vxc_char4,  sigmoid)\n\
LSTMUNIT_CB_FP16_FP32(U8,  , int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_CB_FP16_FP32(I16, , int4,  vxc_short4, vxc_short4, sigmoid)\n\
LSTMUNIT_CB_FP16_FP32(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
LSTMUNIT_CB_FP16_FP32(I8,  _HARD, int4,  vxc_char4,  vxc_char4,  hard_sigmoid)\n\
LSTMUNIT_CB_FP16_FP32(U8,  _HARD, int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_CB_FP16_FP32(I16, _HARD, int4,  vxc_short4, vxc_short4, hard_sigmoid)\n\
\n\
#define LSTMUNIT_CB_FP16_FP16(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_CB_F16to##out_type_name##_F16##act_name( \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_array_t  hstate_f_conv, \\\n\
    __read_only  image2d_array_t  hstate_c_conv, \\\n\
    __read_only  image2d_array_t  hstate_o_conv, \\\n\
    __read_only  image2d_t        bias_f, \\\n\
    __read_only  image2d_t        bias_c, \\\n\
    __read_only  image2d_t        bias_o, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __write_only image2d_t        h_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_short8 vect0, vect1, vect2, vect3, vect4; \\\n\
    vxc_half8  src0, src1, src2, src3, src4; \\\n\
    vxc_short8 vect10, vect11, vect12, vect13; \\\n\
    vxc_half8  src10, src11, src12, src13; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    float4 b0, b1, b2, b3; \\\n\
    VXC_ReadImage(vect1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src1, vect1, 16); \\\n\
    VXC_ReadImage(vect11, hstate_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src11, vect11, 16); \\\n\
    VXC_ReadImage(vect2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src2, vect2, 16); \\\n\
    VXC_ReadImage(vect12, hstate_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src12, vect12, 16); \\\n\
    VXC_ReadImage(vect3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src3, vect3, 16); \\\n\
    VXC_ReadImage(vect13, hstate_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src13, vect13, 16); \\\n\
    VXC_ReadImage(vect4, cell_state_in, coord_in.zy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src4, vect4, 16); \\\n\
    b1 = read_imagef(bias_f, coord_in.xw); \\\n\
    b2 = read_imagef(bias_c, coord_in.xw); \\\n\
    b3 = read_imagef(bias_o, coord_in.xw); \\\n\
 \\\n\
    VXC_DP4x4(data_f_t, src1, src11, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_g_t, src2, src12, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_c_t, src4, src4, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_o_t, src3, src13, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    data_f_t = data_f_t + b1; \\\n\
    data_g_t = data_g_t + b2; \\\n\
    data_o_t = data_o_t + b3; \\\n\
 \\\n\
    convert_type dst0; \\\n\
    half4 dst_cell; \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = 1.0 - data_f_t; \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    _viv_asm(CONV, dst_cell, data_c_t); \\\n\
    VXC_DP4x4(src0, dst_cell, dst_cell, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtractHalf4_4x4); \\\n\
    _viv_asm(COPY, vect0, src0, 8); \\\n\
    VXC_WriteImage(cell_state_out, coord_in.zy, vect0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_WriteImage(h_state_out, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_CB_FP16_FP16(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_CB_FP16_FP16(I8,  , int4,  vxc_char4,  vxc_char4,  sigmoid)\n\
LSTMUNIT_CB_FP16_FP16(U8,  , int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_CB_FP16_FP16(I16, , int4,  vxc_short4, vxc_short4, sigmoid)\n\
LSTMUNIT_CB_FP16_FP16(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
LSTMUNIT_CB_FP16_FP16(I8,  _HARD, int4,  vxc_char4,  vxc_char4,  hard_sigmoid)\n\
LSTMUNIT_CB_FP16_FP16(U8,  _HARD, int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_CB_FP16_FP16(I16, _HARD, int4,  vxc_short4, vxc_short4, hard_sigmoid)\n\
"; /* end of vsi_nn_kernel_lstmunit_activation_CB_F16_vx*/

static const char vsi_nn_kernel_lstmunit_activation_CB_U8_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform float logE;\n\
_viv_uniform float twoLogE;\n\
_viv_uniform float forget_bias;\n\
float4 sigmoid(float4 x)\n\
{\n\
    x *= -logE;\n\
    x = 1 + exp2(x);\n\
    return 1 / x;\n\
}\n\
float4 hard_sigmoid(float4 x)\n\
{\n\
    x = 0.2 * x + 0.5;\n\
    x = clamp(x, 0, 1);\n\
    return x;\n\
}\n\
float4 tangentH(float4 x)\n\
{\n\
    x *= -twoLogE;\n\
    x = 1 + exp2(x);\n\
    x = 1 / x;\n\
    return 2 * x - 1;\n\
}\n\
_viv_uniform float outputScale;\n\
_viv_uniform float outputZP;\n\
_viv_uniform VXC_512Bits uniExtract8Data_2x8;\n\
_viv_uniform VXC_512Bits uniFp16toFp32_4x4;\n\
_viv_uniform float4 clip_Min_F;\n\
_viv_uniform float4 clip_Max_F;\n\
_viv_uniform VXC_512Bits uniExtractHalf4_4x4;\n\
_viv_uniform VXC_512Bits uniU8AddS32_4x4;\n\
_viv_uniform int4 input0Array_ZP;\n\
_viv_uniform int4 input1Array_ZP;\n\
_viv_uniform float4 input0Array_Scale;\n\
_viv_uniform float4 input1Array_Scale;\n\
\n\
#define LSTMUNIT_CB_U8_FP32(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_CB_U8to##out_type_name##_F32##act_name( \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_array_t  hstate_f_conv, \\\n\
    __read_only  image2d_array_t  hstate_c_conv, \\\n\
    __read_only  image2d_array_t  hstate_o_conv, \\\n\
    __read_only  image2d_t        bias_f, \\\n\
    __read_only  image2d_t        bias_c, \\\n\
    __read_only  image2d_t        bias_o, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __write_only image2d_t        h_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_uchar4 src0,  src1,  src2,  src3; \\\n\
    vxc_uchar4 src10, src11, src12, src13; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    float4 b0, b1, b2, b3; \\\n\
    float4 vecA, vecB; \\\n\
    VXC_ReadImage(src1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src11, hstate_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src12, hstate_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src13, hstate_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    data_c_t = read_imagef(cell_state_in, coord_in.zy); \\\n\
    b1 = read_imagef(bias_f, coord_in.xw); \\\n\
    b2 = read_imagef(bias_c, coord_in.xw); \\\n\
    b3 = read_imagef(bias_o, coord_in.xw); \\\n\
 \\\n\
    VXC_DP4x4(vecA, src1, input0Array_ZP.yyyy, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src11, input1Array_ZP.yyyy, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_f_t = vecA * input0Array_Scale.yyyy + vecB * input1Array_Scale.yyyy + b1; \\\n\
    VXC_DP4x4(vecA, src2, input0Array_ZP.zzzz, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src12, input1Array_ZP.zzzz, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_g_t = vecA * input0Array_Scale.zzzz + vecB * input1Array_Scale.zzzz + b2; \\\n\
    VXC_DP4x4(vecA, src3, input0Array_ZP.wwww, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src13, input1Array_ZP.wwww, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_o_t = vecA * input0Array_Scale.wwww + vecB * input1Array_Scale.wwww + b3; \\\n\
 \\\n\
    convert_type dst0; \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = 1.0 - data_f_t; \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    write_imagef(cell_state_out, coord_in.zy, data_c_t); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_WriteImage(h_state_out, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_CB_U8_FP32(U8,  , int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_CB_U8_FP32(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_CB_U8_FP32(U8,  _HARD, int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_CB_U8_FP32(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
\n\
#define LSTMUNIT_CB_U8_FP16(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_CB_U8to##out_type_name##_F16##act_name( \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_array_t  hstate_f_conv, \\\n\
    __read_only  image2d_array_t  hstate_c_conv, \\\n\
    __read_only  image2d_array_t  hstate_o_conv, \\\n\
    __read_only  image2d_t        bias_f, \\\n\
    __read_only  image2d_t        bias_c, \\\n\
    __read_only  image2d_t        bias_o, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __write_only image2d_t        h_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_short8 vect0; \\\n\
    vxc_half8  src4; \\\n\
    vxc_uchar4 src0,  src1,  src2,  src3; \\\n\
    vxc_uchar4 src10, src11, src12, src13; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    float4 b0, b1, b2, b3; \\\n\
    float4 vecA, vecB; \\\n\
    VXC_ReadImage(src1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src11, hstate_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src12, hstate_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src13, hstate_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(vect0, cell_state_in, coord_in.zy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src4, vect0, 16); \\\n\
    b1 = read_imagef(bias_f, coord_in.xw); \\\n\
    b2 = read_imagef(bias_c, coord_in.xw); \\\n\
    b3 = read_imagef(bias_o, coord_in.xw); \\\n\
 \\\n\
    VXC_DP4x4(data_c_t, src4, src4, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(vecA, src1, input0Array_ZP.yyyy, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src11, input1Array_ZP.yyyy, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_f_t = vecA * input0Array_Scale.yyyy + vecB * input1Array_Scale.yyyy + b1; \\\n\
    VXC_DP4x4(vecA, src2, input0Array_ZP.zzzz, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src12, input1Array_ZP.zzzz, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_g_t = vecA * input0Array_Scale.zzzz + vecB * input1Array_Scale.zzzz + b2; \\\n\
    VXC_DP4x4(vecA, src3, input0Array_ZP.wwww, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src13, input1Array_ZP.wwww, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_o_t = vecA * input0Array_Scale.wwww + vecB * input1Array_Scale.wwww + b3; \\\n\
 \\\n\
    convert_type dst0; \\\n\
    half4 dst_cell; \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = 1.0 - data_f_t; \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    _viv_asm(CONV, dst_cell, data_c_t); \\\n\
    VXC_DP4x4(src4, dst_cell, dst_cell, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtractHalf4_4x4); \\\n\
    _viv_asm(COPY, vect0, src4, 8); \\\n\
    VXC_WriteImage(cell_state_out, coord_in.zy, vect0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_WriteImage(h_state_out, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_CB_U8_FP16(U8,  , int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_CB_U8_FP16(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_CB_U8_FP16(U8,  _HARD, int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_CB_U8_FP16(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
"; /* end of vsi_nn_kernel_lstmunit_activation_CB_U8_vx*/

static const char vsi_nn_kernel_lstmunit_activation_CLP_F16_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform float logE;\n\
_viv_uniform float twoLogE;\n\
_viv_uniform float forget_bias;\n\
float4 sigmoid(float4 x)\n\
{\n\
    x *= -logE;\n\
    x = 1 + exp2(x);\n\
    return 1 / x;\n\
}\n\
float4 hard_sigmoid(float4 x)\n\
{\n\
    x = 0.2 * x + 0.5;\n\
    x = clamp(x, 0, 1);\n\
    return x;\n\
}\n\
float4 tangentH(float4 x)\n\
{\n\
    x *= -twoLogE;\n\
    x = 1 + exp2(x);\n\
    x = 1 / x;\n\
    return 2 * x - 1;\n\
}\n\
_viv_uniform float outputScale;\n\
_viv_uniform float outputZP;\n\
_viv_uniform VXC_512Bits uniExtract8Data_2x8;\n\
_viv_uniform VXC_512Bits uniFp16toFp32_4x4;\n\
_viv_uniform float4 clip_Min_F;\n\
_viv_uniform float4 clip_Max_F;\n\
\n\
#define LSTMUNIT_CLP_FP32(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_CLP_F16to##out_type_name##_F32##act_name( \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_t        bias_f, \\\n\
    __read_only  image2d_t        bias_c, \\\n\
    __read_only  image2d_t        bias_o, \\\n\
    __read_only  image2d_t        layer_norm_wf, \\\n\
    __read_only  image2d_t        layer_norm_wc, \\\n\
    __read_only  image2d_t        layer_norm_wo, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_short8 vect0, vect1, vect2, vect3; \\\n\
    vxc_half8  src0, src1, src2, src3; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    float4 w0, w1, w2, b0, b1, b2; \\\n\
    VXC_ReadImage(vect1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src1, vect1, 16); \\\n\
    VXC_ReadImage(vect2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src2, vect2, 16); \\\n\
    VXC_ReadImage(vect3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src3, vect3, 16); \\\n\
    w0 = read_imagef(layer_norm_wf, coord_in.xw); \\\n\
    w1 = read_imagef(layer_norm_wc, coord_in.xw); \\\n\
    w2 = read_imagef(layer_norm_wo, coord_in.xw); \\\n\
    data_c_t = read_imagef(cell_state_in, coord_in.xy); \\\n\
    b0 = read_imagef(bias_f, coord_in.xw); \\\n\
    b1 = read_imagef(bias_c, coord_in.xw); \\\n\
    b2 = read_imagef(bias_o, coord_in.xw); \\\n\
 \\\n\
    VXC_DP4x4(data_f_t, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_g_t, src2, src2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_o_t, src3, src3, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    data_f_t = data_f_t * w0 + b0; \\\n\
    data_g_t = data_g_t * w1 + b1; \\\n\
    data_o_t = data_o_t * w2 + b2; \\\n\
 \\\n\
    convert_type dst0; \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = 1.0 - data_f_t; \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    write_imagef(cell_state_out, coord_in.zy, data_c_t); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_CLP_FP32(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_CLP_FP32(I8,  , int4,  vxc_char4,  vxc_char4,  sigmoid)\n\
LSTMUNIT_CLP_FP32(U8,  , int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_CLP_FP32(I16, , int4,  vxc_short4, vxc_short4, sigmoid)\n\
LSTMUNIT_CLP_FP32(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
LSTMUNIT_CLP_FP32(I8,  _HARD, int4,  vxc_char4,  vxc_char4,  hard_sigmoid)\n\
LSTMUNIT_CLP_FP32(U8,  _HARD, int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_CLP_FP32(I16, _HARD, int4,  vxc_short4, vxc_short4, hard_sigmoid)\n\
\n\
_viv_uniform VXC_512Bits uniExtractHalf4_4x4;\n\
#define LSTMUNIT_CLP_FP16(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_CLP_F16to##out_type_name##_F16##act_name( \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_t        bias_f, \\\n\
    __read_only  image2d_t        bias_c, \\\n\
    __read_only  image2d_t        bias_o, \\\n\
    __read_only  image2d_t        layer_norm_wf, \\\n\
    __read_only  image2d_t        layer_norm_wc, \\\n\
    __read_only  image2d_t        layer_norm_wo, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __read_only  image2d_t        lstmuint_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_short8 vect0, vect1, vect2, vect3, vect4; \\\n\
    vxc_half8  src0, src1, src2, src3, src4; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    float4 w0, w1, w2, b0, b1, b2; \\\n\
    VXC_ReadImage(vect1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src1, vect1, 16); \\\n\
    VXC_ReadImage(vect2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src2, vect2, 16); \\\n\
    VXC_ReadImage(vect3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src3, vect3, 16); \\\n\
    VXC_ReadImage(vect4, cell_state_in, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src4, vect4, 16); \\\n\
    w0 = read_imagef(layer_norm_wf, coord_in.xw); \\\n\
    w1 = read_imagef(layer_norm_wc, coord_in.xw); \\\n\
    w2 = read_imagef(layer_norm_wo, coord_in.xw); \\\n\
    b0 = read_imagef(bias_f, coord_in.xw); \\\n\
    b1 = read_imagef(bias_c, coord_in.xw); \\\n\
    b2 = read_imagef(bias_o, coord_in.xw); \\\n\
 \\\n\
    VXC_DP4x4(data_f_t, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_g_t, src2, src2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_o_t, src3, src3, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_c_t, src4, src4, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    data_f_t = data_f_t * w0 + b0; \\\n\
    data_g_t = data_g_t * w1 + b1; \\\n\
    data_o_t = data_o_t * w2 + b2; \\\n\
 \\\n\
    convert_type dst0; \\\n\
    half4 cell_data; \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = 1.0 - data_f_t; \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    _viv_asm(CONV, cell_data, data_c_t); \\\n\
    VXC_DP4x4(src0, cell_data, cell_data, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtractHalf4_4x4); \\\n\
    _viv_asm(COPY, vect0, src0, 8); \\\n\
    VXC_WriteImage(cell_state_out, coord_in.zy, vect0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_CLP_FP16(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_CLP_FP16(I8,  , int4,  vxc_char4,  vxc_char4,  sigmoid)\n\
LSTMUNIT_CLP_FP16(U8,  , int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_CLP_FP16(I16, , int4,  vxc_short4, vxc_short4, sigmoid)\n\
LSTMUNIT_CLP_FP16(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
LSTMUNIT_CLP_FP16(I8,  _HARD, int4,  vxc_char4,  vxc_char4,  hard_sigmoid)\n\
LSTMUNIT_CLP_FP16(U8,  _HARD, int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_CLP_FP16(I16, _HARD, int4,  vxc_short4, vxc_short4, hard_sigmoid)\n\
"; /* end of vsi_nn_kernel_lstmunit_activation_CLP_F16_vx*/

static const char vsi_nn_kernel_lstmunit_activation_CL_F16_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform float logE;\n\
_viv_uniform float twoLogE;\n\
_viv_uniform float forget_bias;\n\
float4 sigmoid(float4 x)\n\
{\n\
    x *= -logE;\n\
    x = 1 + exp2(x);\n\
    return 1 / x;\n\
}\n\
float4 hard_sigmoid(float4 x)\n\
{\n\
    x = 0.2 * x + 0.5;\n\
    x = clamp(x, 0, 1);\n\
    return x;\n\
}\n\
float4 tangentH(float4 x)\n\
{\n\
    x *= -twoLogE;\n\
    x = 1 + exp2(x);\n\
    x = 1 / x;\n\
    return 2 * x - 1;\n\
}\n\
_viv_uniform float outputScale;\n\
_viv_uniform float outputZP;\n\
_viv_uniform VXC_512Bits uniExtract8Data_2x8;\n\
_viv_uniform VXC_512Bits uniFp16toFp32_4x4;\n\
_viv_uniform float4 clip_Min_F;\n\
_viv_uniform float4 clip_Max_F;\n\
_viv_uniform VXC_512Bits uniExtractHalf4_4x4;\n\
\n\
#define LSTMUNIT_CL_FP32(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_CL_F16to##out_type_name##_F32##act_name( \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_t        bias_f, \\\n\
    __read_only  image2d_t        bias_c, \\\n\
    __read_only  image2d_t        bias_o, \\\n\
    __read_only  image2d_t        layer_norm_wf, \\\n\
    __read_only  image2d_t        layer_norm_wc, \\\n\
    __read_only  image2d_t        layer_norm_wo, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __write_only image2d_t        h_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_short8 vect0, vect1, vect2, vect3; \\\n\
    vxc_half8  src0, src1, src2, src3; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    float4 w0, w1, w2, b0, b1, b2; \\\n\
    VXC_ReadImage(vect1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src1, vect1, 16); \\\n\
    VXC_ReadImage(vect2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src2, vect2, 16); \\\n\
    VXC_ReadImage(vect3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src3, vect3, 16); \\\n\
    w0 = read_imagef(layer_norm_wf, coord_in.xw); \\\n\
    w1 = read_imagef(layer_norm_wc, coord_in.xw); \\\n\
    w2 = read_imagef(layer_norm_wo, coord_in.xw); \\\n\
    data_c_t = read_imagef(cell_state_in, coord_in.xy); \\\n\
    b0 = read_imagef(bias_f, coord_in.xw); \\\n\
    b1 = read_imagef(bias_c, coord_in.xw); \\\n\
    b2 = read_imagef(bias_o, coord_in.xw); \\\n\
 \\\n\
    VXC_DP4x4(data_f_t, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_g_t, src2, src2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_o_t, src3, src3, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    data_f_t = data_f_t * w0 + b0; \\\n\
    data_g_t = data_g_t * w1 + b1; \\\n\
    data_o_t = data_o_t * w2 + b2; \\\n\
 \\\n\
    convert_type dst0; \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = 1.0 - data_f_t; \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    write_imagef(cell_state_out, coord_in.zy, data_c_t); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_WriteImage(h_state_out, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_CL_FP32(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_CL_FP32(I8,  , int4,  vxc_char4,  vxc_char4,  sigmoid)\n\
LSTMUNIT_CL_FP32(U8,  , int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_CL_FP32(I16, , int4,  vxc_short4, vxc_short4, sigmoid)\n\
LSTMUNIT_CL_FP32(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
LSTMUNIT_CL_FP32(I8,  _HARD, int4,  vxc_char4,  vxc_char4,  hard_sigmoid)\n\
LSTMUNIT_CL_FP32(U8,  _HARD, int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_CL_FP32(I16, _HARD, int4,  vxc_short4, vxc_short4, hard_sigmoid)\n\
\n\
#define LSTMUNIT_CL_FP16(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_CL_F16to##out_type_name##_F16##act_name( \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_t        bias_f, \\\n\
    __read_only  image2d_t        bias_c, \\\n\
    __read_only  image2d_t        bias_o, \\\n\
    __read_only  image2d_t        layer_norm_wf, \\\n\
    __read_only  image2d_t        layer_norm_wc, \\\n\
    __read_only  image2d_t        layer_norm_wo, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __write_only image2d_t        h_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_short8 vect0, vect1, vect2, vect3, vect4; \\\n\
    vxc_half8  src0, src1, src2, src3, src4; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    float4 w0, w1, w2, b0, b1, b2; \\\n\
    VXC_ReadImage(vect1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src1, vect1, 16); \\\n\
    VXC_ReadImage(vect2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src2, vect2, 16); \\\n\
    VXC_ReadImage(vect3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src3, vect3, 16); \\\n\
    VXC_ReadImage(vect4, cell_state_in, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src4, vect4, 16); \\\n\
    w0 = read_imagef(layer_norm_wf, coord_in.xw); \\\n\
    w1 = read_imagef(layer_norm_wc, coord_in.xw); \\\n\
    w2 = read_imagef(layer_norm_wo, coord_in.xw); \\\n\
    b0 = read_imagef(bias_f, coord_in.xw); \\\n\
    b1 = read_imagef(bias_c, coord_in.xw); \\\n\
    b2 = read_imagef(bias_o, coord_in.xw); \\\n\
 \\\n\
    VXC_DP4x4(data_f_t, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_g_t, src2, src2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_o_t, src3, src3, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_c_t, src4, src4, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    data_f_t = data_f_t * w0 + b0; \\\n\
    data_g_t = data_g_t * w1 + b1; \\\n\
    data_o_t = data_o_t * w2 + b2; \\\n\
 \\\n\
    convert_type dst0; \\\n\
    half4 cell_data; \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = 1.0 - data_f_t; \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    _viv_asm(CONV, cell_data, data_c_t); \\\n\
    VXC_DP4x4(src0, cell_data, cell_data, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtractHalf4_4x4); \\\n\
    _viv_asm(COPY, vect0, src0, 8); \\\n\
    VXC_WriteImage(cell_state_out, coord_in.zy, vect0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_WriteImage(h_state_out, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_CL_FP16(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_CL_FP16(I8,  , int4,  vxc_char4,  vxc_char4,  sigmoid)\n\
LSTMUNIT_CL_FP16(U8,  , int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_CL_FP16(I16, , int4,  vxc_short4, vxc_short4, sigmoid)\n\
LSTMUNIT_CL_FP16(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
LSTMUNIT_CL_FP16(I8,  _HARD, int4,  vxc_char4,  vxc_char4,  hard_sigmoid)\n\
LSTMUNIT_CL_FP16(U8,  _HARD, int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_CL_FP16(I16, _HARD, int4,  vxc_short4, vxc_short4, hard_sigmoid)\n\
"; /* end of vsi_nn_kernel_lstmunit_activation_CL_F16_vx*/

static const char vsi_nn_kernel_lstmunit_activation_CSP_F16_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform float logE;\n\
_viv_uniform float twoLogE;\n\
_viv_uniform float forget_bias;\n\
float4 sigmoid(float4 x)\n\
{\n\
    x *= -logE;\n\
    x = 1 + exp2(x);\n\
    return 1 / x;\n\
}\n\
float4 hard_sigmoid(float4 x)\n\
{\n\
    x = 0.2 * x + 0.5;\n\
    x = clamp(x, 0, 1);\n\
    return x;\n\
}\n\
float4 tangentH(float4 x)\n\
{\n\
    x *= -twoLogE;\n\
    x = 1 + exp2(x);\n\
    x = 1 / x;\n\
    return 2 * x - 1;\n\
}\n\
_viv_uniform float outputScale;\n\
_viv_uniform float outputZP;\n\
_viv_uniform VXC_512Bits uniExtract8Data_2x8;\n\
_viv_uniform VXC_512Bits uniFp16toFp32_4x4;\n\
_viv_uniform float4 clip_Min_F;\n\
_viv_uniform float4 clip_Max_F;\n\
_viv_uniform VXC_512Bits uniExtractHalf4_4x4;\n\
_viv_uniform VXC_512Bits uniFp16AddFp16toFp32_4x4;\n\
\n\
#define LSTMUNIT_CSP_FP16_FP32(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_CSP_F16to##out_type_name##_F32##act_name( \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_array_t  hstate_f_conv, \\\n\
    __read_only  image2d_array_t  hstate_c_conv, \\\n\
    __read_only  image2d_array_t  hstate_o_conv, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_short8 vect0, vect1, vect2, vect3; \\\n\
    vxc_half8  src0, src1, src2, src3; \\\n\
    vxc_short8 vect10, vect11, vect12, vect13; \\\n\
    vxc_half8  src10, src11, src12, src13; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    VXC_ReadImage(vect1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src1, vect1, 16); \\\n\
    VXC_ReadImage(vect11, hstate_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src11, vect11, 16); \\\n\
    VXC_ReadImage(vect2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src2, vect2, 16); \\\n\
    VXC_ReadImage(vect12, hstate_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src12, vect12, 16); \\\n\
    VXC_ReadImage(vect3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src3, vect3, 16); \\\n\
    VXC_ReadImage(vect13, hstate_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src13, vect13, 16); \\\n\
    data_c_t = read_imagef(cell_state_in, coord_in.zy); \\\n\
 \\\n\
    VXC_DP4x4(data_f_t, src1, src11, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_g_t, src2, src12, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_o_t, src3, src13, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
 \\\n\
    convert_type dst0; \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = 1.0 - data_f_t; \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    write_imagef(cell_state_out, coord_in.zy, data_c_t); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_CSP_FP16_FP32(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_CSP_FP16_FP32(I8,  , int4,  vxc_char4,  vxc_char4,  sigmoid)\n\
LSTMUNIT_CSP_FP16_FP32(U8,  , int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_CSP_FP16_FP32(I16, , int4,  vxc_short4, vxc_short4, sigmoid)\n\
LSTMUNIT_CSP_FP16_FP32(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
LSTMUNIT_CSP_FP16_FP32(I8,  _HARD, int4,  vxc_char4,  vxc_char4,  hard_sigmoid)\n\
LSTMUNIT_CSP_FP16_FP32(U8,  _HARD, int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_CSP_FP16_FP32(I16, _HARD, int4,  vxc_short4, vxc_short4, hard_sigmoid)\n\
\n\
#define LSTMUNIT_CSP_FP16_FP16(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_CSP_F16to##out_type_name##_F16##act_name( \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_array_t  hstate_f_conv, \\\n\
    __read_only  image2d_array_t  hstate_c_conv, \\\n\
    __read_only  image2d_array_t  hstate_o_conv, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_short8 vect0, vect1, vect2, vect3, vect4; \\\n\
    vxc_half8  src0, src1, src2, src3, src4; \\\n\
    vxc_short8 vect10, vect11, vect12, vect13; \\\n\
    vxc_half8  src10, src11, src12, src13; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    VXC_ReadImage(vect1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src1, vect1, 16); \\\n\
    VXC_ReadImage(vect11, hstate_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src11, vect11, 16); \\\n\
    VXC_ReadImage(vect2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src2, vect2, 16); \\\n\
    VXC_ReadImage(vect12, hstate_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src12, vect12, 16); \\\n\
    VXC_ReadImage(vect3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src3, vect3, 16); \\\n\
    VXC_ReadImage(vect13, hstate_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src13, vect13, 16); \\\n\
    VXC_ReadImage(vect4, cell_state_in, coord_in.zy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src4, vect4, 16); \\\n\
 \\\n\
    VXC_DP4x4(data_f_t, src1, src11, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_g_t, src2, src12, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_c_t, src4, src4, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_o_t, src3, src13, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
 \\\n\
    convert_type dst0; \\\n\
    half4 dst_cell; \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = 1.0 - data_f_t; \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    _viv_asm(CONV, dst_cell, data_c_t); \\\n\
    VXC_DP4x4(src0, dst_cell, dst_cell, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtractHalf4_4x4); \\\n\
    _viv_asm(COPY, vect0, src0, 8); \\\n\
    VXC_WriteImage(cell_state_out, coord_in.zy, vect0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_CSP_FP16_FP16(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_CSP_FP16_FP16(I8,  , int4,  vxc_char4,  vxc_char4,  sigmoid)\n\
LSTMUNIT_CSP_FP16_FP16(U8,  , int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_CSP_FP16_FP16(I16, , int4,  vxc_short4, vxc_short4, sigmoid)\n\
LSTMUNIT_CSP_FP16_FP16(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
LSTMUNIT_CSP_FP16_FP16(I8,  _HARD, int4,  vxc_char4,  vxc_char4,  hard_sigmoid)\n\
LSTMUNIT_CSP_FP16_FP16(U8,  _HARD, int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_CSP_FP16_FP16(I16, _HARD, int4,  vxc_short4, vxc_short4, hard_sigmoid)\n\
"; /* end of vsi_nn_kernel_lstmunit_activation_CSP_F16_vx*/

static const char vsi_nn_kernel_lstmunit_activation_CSP_U8_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform float logE;\n\
_viv_uniform float twoLogE;\n\
_viv_uniform float forget_bias;\n\
float4 sigmoid(float4 x)\n\
{\n\
    x *= -logE;\n\
    x = 1 + exp2(x);\n\
    return 1 / x;\n\
}\n\
float4 hard_sigmoid(float4 x)\n\
{\n\
    x = 0.2 * x + 0.5;\n\
    x = clamp(x, 0, 1);\n\
    return x;\n\
}\n\
float4 tangentH(float4 x)\n\
{\n\
    x *= -twoLogE;\n\
    x = 1 + exp2(x);\n\
    x = 1 / x;\n\
    return 2 * x - 1;\n\
}\n\
_viv_uniform float outputScale;\n\
_viv_uniform float outputZP;\n\
_viv_uniform VXC_512Bits uniExtract8Data_2x8;\n\
_viv_uniform VXC_512Bits uniFp16toFp32_4x4;\n\
_viv_uniform float4 clip_Min_F;\n\
_viv_uniform float4 clip_Max_F;\n\
_viv_uniform VXC_512Bits uniExtractHalf4_4x4;\n\
_viv_uniform VXC_512Bits uniU8AddS32_4x4;\n\
_viv_uniform int4 input0Array_ZP;\n\
_viv_uniform int4 input1Array_ZP;\n\
_viv_uniform float4 input0Array_Scale;\n\
_viv_uniform float4 input1Array_Scale;\n\
\n\
#define LSTMUNIT_CSP_U8_FP32(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_CSP_U8to##out_type_name##_F32##act_name( \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_array_t  hstate_f_conv, \\\n\
    __read_only  image2d_array_t  hstate_c_conv, \\\n\
    __read_only  image2d_array_t  hstate_o_conv, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_uchar4 src0,  src1,  src2,  src3; \\\n\
    vxc_uchar4 src10, src11, src12, src13; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    float4 vecA, vecB; \\\n\
    VXC_ReadImage(src1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src11, hstate_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src12, hstate_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src13, hstate_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    data_c_t = read_imagef(cell_state_in, coord_in.zy); \\\n\
    VXC_DP4x4(vecA, src1, input0Array_ZP.yyyy, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src11, input1Array_ZP.yyyy, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_f_t = vecA * input0Array_Scale.yyyy + vecB * input1Array_Scale.yyyy; \\\n\
    VXC_DP4x4(vecA, src2, input0Array_ZP.zzzz, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src12, input1Array_ZP.zzzz, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_g_t = vecA * input0Array_Scale.zzzz + vecB * input1Array_Scale.zzzz; \\\n\
    VXC_DP4x4(vecA, src3, input0Array_ZP.wwww, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src13, input1Array_ZP.wwww, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_o_t = vecA * input0Array_Scale.wwww + vecB * input1Array_Scale.wwww; \\\n\
 \\\n\
    convert_type dst0; \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = 1.0 - data_f_t; \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    write_imagef(cell_state_out, coord_in.zy, data_c_t); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_CSP_U8_FP32(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_CSP_U8_FP32(I8,  , int4,  vxc_char4,  vxc_char4,  sigmoid)\n\
LSTMUNIT_CSP_U8_FP32(U8,  , int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_CSP_U8_FP32(I16, , int4,  vxc_short4, vxc_short4, sigmoid)\n\
LSTMUNIT_CSP_U8_FP32(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
LSTMUNIT_CSP_U8_FP32(I8,  _HARD, int4,  vxc_char4,  vxc_char4,  hard_sigmoid)\n\
LSTMUNIT_CSP_U8_FP32(U8,  _HARD, int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_CSP_U8_FP32(I16, _HARD, int4,  vxc_short4, vxc_short4, hard_sigmoid)\n\
\n\
#define LSTMUNIT_CSP_U8_FP16(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_CSP_U8to##out_type_name##_F16##act_name( \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_array_t  hstate_f_conv, \\\n\
    __read_only  image2d_array_t  hstate_c_conv, \\\n\
    __read_only  image2d_array_t  hstate_o_conv, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_short8 vect0; \\\n\
    vxc_half8  src4; \\\n\
    vxc_uchar4 src0,  src1,  src2,  src3; \\\n\
    vxc_uchar4 src10, src11, src12, src13; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    float4 vecA, vecB; \\\n\
    VXC_ReadImage(src1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src11, hstate_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src12, hstate_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src13, hstate_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(vect0, cell_state_in, coord_in.zy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src4, vect0, 16); \\\n\
 \\\n\
    VXC_DP4x4(data_c_t, src4, src4, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(vecA, src1, input0Array_ZP.yyyy, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src11, input1Array_ZP.yyyy, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_f_t = vecA * input0Array_Scale.yyyy + vecB * input1Array_Scale.yyyy; \\\n\
    VXC_DP4x4(vecA, src2, input0Array_ZP.zzzz, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src12, input1Array_ZP.zzzz, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_g_t = vecA * input0Array_Scale.zzzz + vecB * input1Array_Scale.zzzz; \\\n\
    VXC_DP4x4(vecA, src3, input0Array_ZP.wwww, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src13, input1Array_ZP.wwww, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_o_t = vecA * input0Array_Scale.wwww + vecB * input1Array_Scale.wwww; \\\n\
 \\\n\
    convert_type dst0; \\\n\
    half4 dst_cell; \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = 1.0 - data_f_t; \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    _viv_asm(CONV, dst_cell, data_c_t); \\\n\
    VXC_DP4x4(src4, dst_cell, dst_cell, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtractHalf4_4x4); \\\n\
    _viv_asm(COPY, vect0, src4, 8); \\\n\
    VXC_WriteImage(cell_state_out, coord_in.zy, vect0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_CSP_U8_FP16(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_CSP_U8_FP16(I8,  , int4,  vxc_char4,  vxc_char4,  sigmoid)\n\
LSTMUNIT_CSP_U8_FP16(U8,  , int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_CSP_U8_FP16(I16, , int4,  vxc_short4, vxc_short4, sigmoid)\n\
LSTMUNIT_CSP_U8_FP16(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
LSTMUNIT_CSP_U8_FP16(I8,  _HARD, int4,  vxc_char4,  vxc_char4,  hard_sigmoid)\n\
LSTMUNIT_CSP_U8_FP16(U8,  _HARD, int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_CSP_U8_FP16(I16, _HARD, int4,  vxc_short4, vxc_short4, hard_sigmoid)\n\
"; /* end of vsi_nn_kernel_lstmunit_activation_CSP_U8_vx*/

static const char vsi_nn_kernel_lstmunit_activation_CS_F16_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform float logE;\n\
_viv_uniform float twoLogE;\n\
_viv_uniform float forget_bias;\n\
float4 sigmoid(float4 x)\n\
{\n\
    x *= -logE;\n\
    x = 1 + exp2(x);\n\
    return 1 / x;\n\
}\n\
float4 hard_sigmoid(float4 x)\n\
{\n\
    x = 0.2 * x + 0.5;\n\
    x = clamp(x, 0, 1);\n\
    return x;\n\
}\n\
float4 tangentH(float4 x)\n\
{\n\
    x *= -twoLogE;\n\
    x = 1 + exp2(x);\n\
    x = 1 / x;\n\
    return 2 * x - 1;\n\
}\n\
_viv_uniform float outputScale;\n\
_viv_uniform float outputZP;\n\
_viv_uniform VXC_512Bits uniExtract8Data_2x8;\n\
_viv_uniform VXC_512Bits uniFp16toFp32_4x4;\n\
_viv_uniform float4 clip_Min_F;\n\
_viv_uniform float4 clip_Max_F;\n\
_viv_uniform VXC_512Bits uniExtractHalf4_4x4;\n\
_viv_uniform VXC_512Bits uniFp16AddFp16toFp32_4x4;\n\
\n\
#define LSTMUNIT_CS_FP16_FP32(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_CS_F16to##out_type_name##_F32##act_name( \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_array_t  hstate_f_conv, \\\n\
    __read_only  image2d_array_t  hstate_c_conv, \\\n\
    __read_only  image2d_array_t  hstate_o_conv, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __write_only image2d_t        h_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_short8 vect0, vect1, vect2, vect3; \\\n\
    vxc_half8  src0, src1, src2, src3; \\\n\
    vxc_short8 vect10, vect11, vect12, vect13; \\\n\
    vxc_half8  src10, src11, src12, src13; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    VXC_ReadImage(vect1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src1, vect1, 16); \\\n\
    VXC_ReadImage(vect11, hstate_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src11, vect11, 16); \\\n\
    VXC_ReadImage(vect2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src2, vect2, 16); \\\n\
    VXC_ReadImage(vect12, hstate_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src12, vect12, 16); \\\n\
    VXC_ReadImage(vect3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src3, vect3, 16); \\\n\
    VXC_ReadImage(vect13, hstate_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src13, vect13, 16); \\\n\
    data_c_t = read_imagef(cell_state_in, coord_in.zy); \\\n\
 \\\n\
    VXC_DP4x4(data_f_t, src1, src11, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_g_t, src2, src12, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_o_t, src3, src13, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
 \\\n\
    convert_type dst0; \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = 1.0 - data_f_t; \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    write_imagef(cell_state_out, coord_in.zy, data_c_t); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_WriteImage(h_state_out, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_CS_FP16_FP32(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_CS_FP16_FP32(I8,  , int4,  vxc_char4,  vxc_char4,  sigmoid)\n\
LSTMUNIT_CS_FP16_FP32(U8,  , int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_CS_FP16_FP32(I16, , int4,  vxc_short4, vxc_short4, sigmoid)\n\
LSTMUNIT_CS_FP16_FP32(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
LSTMUNIT_CS_FP16_FP32(I8,  _HARD, int4,  vxc_char4,  vxc_char4,  hard_sigmoid)\n\
LSTMUNIT_CS_FP16_FP32(U8,  _HARD, int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_CS_FP16_FP32(I16, _HARD, int4,  vxc_short4, vxc_short4, hard_sigmoid)\n\
\n\
#define LSTMUNIT_CS_FP16_FP16(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_CS_F16to##out_type_name##_F16##act_name( \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_array_t  hstate_f_conv, \\\n\
    __read_only  image2d_array_t  hstate_c_conv, \\\n\
    __read_only  image2d_array_t  hstate_o_conv, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __write_only image2d_t        h_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_short8 vect0, vect1, vect2, vect3, vect4; \\\n\
    vxc_half8  src0, src1, src2, src3, src4; \\\n\
    vxc_short8 vect10, vect11, vect12, vect13; \\\n\
    vxc_half8  src10, src11, src12, src13; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    VXC_ReadImage(vect1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src1, vect1, 16); \\\n\
    VXC_ReadImage(vect11, hstate_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src11, vect11, 16); \\\n\
    VXC_ReadImage(vect2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src2, vect2, 16); \\\n\
    VXC_ReadImage(vect12, hstate_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src12, vect12, 16); \\\n\
    VXC_ReadImage(vect3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src3, vect3, 16); \\\n\
    VXC_ReadImage(vect13, hstate_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src13, vect13, 16); \\\n\
    VXC_ReadImage(vect4, cell_state_in, coord_in.zy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src4, vect4, 16); \\\n\
 \\\n\
    VXC_DP4x4(data_f_t, src1, src11, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_g_t, src2, src12, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_c_t, src4, src4, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_o_t, src3, src13, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
 \\\n\
    convert_type dst0; \\\n\
    half4 dst_cell; \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = 1.0 - data_f_t; \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    _viv_asm(CONV, dst_cell, data_c_t); \\\n\
    VXC_DP4x4(src0, dst_cell, dst_cell, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtractHalf4_4x4); \\\n\
    _viv_asm(COPY, vect0, src0, 8); \\\n\
    VXC_WriteImage(cell_state_out, coord_in.zy, vect0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_WriteImage(h_state_out, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_CS_FP16_FP16(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_CS_FP16_FP16(I8,  , int4,  vxc_char4,  vxc_char4,  sigmoid)\n\
LSTMUNIT_CS_FP16_FP16(U8,  , int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_CS_FP16_FP16(I16, , int4,  vxc_short4, vxc_short4, sigmoid)\n\
LSTMUNIT_CS_FP16_FP16(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
LSTMUNIT_CS_FP16_FP16(I8,  _HARD, int4,  vxc_char4,  vxc_char4,  hard_sigmoid)\n\
LSTMUNIT_CS_FP16_FP16(U8,  _HARD, int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_CS_FP16_FP16(I16, _HARD, int4,  vxc_short4, vxc_short4, hard_sigmoid)\n\
"; /* end of vsi_nn_kernel_lstmunit_activation_CS_F16_vx*/

static const char vsi_nn_kernel_lstmunit_activation_CS_U8_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform float logE;\n\
_viv_uniform float twoLogE;\n\
_viv_uniform float forget_bias;\n\
float4 sigmoid(float4 x)\n\
{\n\
    x *= -logE;\n\
    x = 1 + exp2(x);\n\
    return 1 / x;\n\
}\n\
float4 hard_sigmoid(float4 x)\n\
{\n\
    x = 0.2 * x + 0.5;\n\
    x = clamp(x, 0, 1);\n\
    return x;\n\
}\n\
float4 tangentH(float4 x)\n\
{\n\
    x *= -twoLogE;\n\
    x = 1 + exp2(x);\n\
    x = 1 / x;\n\
    return 2 * x - 1;\n\
}\n\
_viv_uniform float outputScale;\n\
_viv_uniform float outputZP;\n\
_viv_uniform VXC_512Bits uniExtract8Data_2x8;\n\
_viv_uniform VXC_512Bits uniFp16toFp32_4x4;\n\
_viv_uniform float4 clip_Min_F;\n\
_viv_uniform float4 clip_Max_F;\n\
_viv_uniform VXC_512Bits uniExtractHalf4_4x4;\n\
_viv_uniform VXC_512Bits uniU8AddS32_4x4;\n\
_viv_uniform int4 input0Array_ZP;\n\
_viv_uniform int4 input1Array_ZP;\n\
_viv_uniform float4 input0Array_Scale;\n\
_viv_uniform float4 input1Array_Scale;\n\
\n\
#define LSTMUNIT_CS_U8_FP32(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_CS_U8to##out_type_name##_F32##act_name( \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_array_t  hstate_f_conv, \\\n\
    __read_only  image2d_array_t  hstate_c_conv, \\\n\
    __read_only  image2d_array_t  hstate_o_conv, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __write_only image2d_t        h_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_uchar4 src0,  src1,  src2,  src3; \\\n\
    vxc_uchar4 src10, src11, src12, src13; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    float4 vecA, vecB; \\\n\
    VXC_ReadImage(src1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src11, hstate_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src12, hstate_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src13, hstate_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    data_c_t = read_imagef(cell_state_in, coord_in.zy); \\\n\
 \\\n\
    VXC_DP4x4(vecA, src1, input0Array_ZP.yyyy, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src11, input1Array_ZP.yyyy, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_f_t = vecA * input0Array_Scale.yyyy + vecB * input1Array_Scale.yyyy; \\\n\
    VXC_DP4x4(vecA, src2, input0Array_ZP.zzzz, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src12, input1Array_ZP.zzzz, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_g_t = vecA * input0Array_Scale.zzzz + vecB * input1Array_Scale.zzzz; \\\n\
    VXC_DP4x4(vecA, src3, input0Array_ZP.wwww, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src13, input1Array_ZP.wwww, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_o_t = vecA * input0Array_Scale.wwww + vecB * input1Array_Scale.wwww; \\\n\
 \\\n\
    convert_type dst0; \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = 1.0 - data_f_t; \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    write_imagef(cell_state_out, coord_in.zy, data_c_t); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_WriteImage(h_state_out, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_CS_U8_FP32(U8,  , int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_CS_U8_FP32(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_CS_U8_FP32(U8,  _HARD, int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_CS_U8_FP32(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
\n\
#define LSTMUNIT_CS_U8_FP16(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_CS_U8to##out_type_name##_F16##act_name( \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_array_t  hstate_f_conv, \\\n\
    __read_only  image2d_array_t  hstate_c_conv, \\\n\
    __read_only  image2d_array_t  hstate_o_conv, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __write_only image2d_t        h_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_short8 vect0; \\\n\
    vxc_half8  src4; \\\n\
    vxc_uchar4 src0,  src1,  src2,  src3; \\\n\
    vxc_uchar4 src10, src11, src12, src13; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    float4 vecA, vecB; \\\n\
    VXC_ReadImage(src1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src11, hstate_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src12, hstate_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src13, hstate_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(vect0, cell_state_in, coord_in.zy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src4, vect0, 16); \\\n\
 \\\n\
    VXC_DP4x4(data_c_t, src4, src4, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(vecA, src1, input0Array_ZP.yyyy, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src11, input1Array_ZP.yyyy, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_f_t = vecA * input0Array_Scale.yyyy + vecB * input1Array_Scale.yyyy; \\\n\
    VXC_DP4x4(vecA, src2, input0Array_ZP.zzzz, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src12, input1Array_ZP.zzzz, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_g_t = vecA * input0Array_Scale.zzzz + vecB * input1Array_Scale.zzzz; \\\n\
    VXC_DP4x4(vecA, src3, input0Array_ZP.wwww, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src13, input1Array_ZP.wwww, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_o_t = vecA * input0Array_Scale.wwww + vecB * input1Array_Scale.wwww; \\\n\
 \\\n\
    convert_type dst0; \\\n\
    half4 dst_cell; \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = 1.0 - data_f_t; \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    _viv_asm(CONV, dst_cell, data_c_t); \\\n\
    VXC_DP4x4(src4, dst_cell, dst_cell, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtractHalf4_4x4); \\\n\
    _viv_asm(COPY, vect0, src4, 8); \\\n\
    VXC_WriteImage(cell_state_out, coord_in.zy, vect0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_WriteImage(h_state_out, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_CS_U8_FP16(U8,  , int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_CS_U8_FP16(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_CS_U8_FP16(U8,  _HARD, int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_CS_U8_FP16(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
"; /* end of vsi_nn_kernel_lstmunit_activation_CS_U8_vx*/

static const char vsi_nn_kernel_lstmunit_activation_LP_F16_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform float logE;\n\
_viv_uniform float twoLogE;\n\
_viv_uniform float forget_bias;\n\
float4 sigmoid(float4 x)\n\
{\n\
    x *= -logE;\n\
    x = 1 + exp2(x);\n\
    return 1 / x;\n\
}\n\
float4 hard_sigmoid(float4 x)\n\
{\n\
    x = 0.2 * x + 0.5;\n\
    x = clamp(x, 0, 1);\n\
    return x;\n\
}\n\
float4 tangentH(float4 x)\n\
{\n\
    x *= -twoLogE;\n\
    x = 1 + exp2(x);\n\
    x = 1 / x;\n\
    return 2 * x - 1;\n\
}\n\
_viv_uniform float outputScale;\n\
_viv_uniform float outputZP;\n\
_viv_uniform VXC_512Bits uniExtract8Data_2x8;\n\
_viv_uniform VXC_512Bits uniFp16toFp32_4x4;\n\
_viv_uniform float4 clip_Min_F;\n\
_viv_uniform float4 clip_Max_F;\n\
_viv_uniform VXC_512Bits uniExtractHalf4_4x4;\n\
\n\
#define LSTMUNIT_LP_FP32(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_LP_F16to##out_type_name##_F32##act_name( \\\n\
    __read_only  image2d_array_t  input_i_conv, \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_t        bias_i, \\\n\
    __read_only  image2d_t        bias_f, \\\n\
    __read_only  image2d_t        bias_c, \\\n\
    __read_only  image2d_t        bias_o, \\\n\
    __read_only  image2d_t        layer_norm_wi, \\\n\
    __read_only  image2d_t        layer_norm_wf, \\\n\
    __read_only  image2d_t        layer_norm_wc, \\\n\
    __read_only  image2d_t        layer_norm_wo, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_short8 vect0, vect1, vect2, vect3; \\\n\
    vxc_half8  src0, src1, src2, src3; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    float4 w0, w1, w2, w3, b0, b1, b2, b3; \\\n\
    VXC_ReadImage(vect0, input_i_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src0, vect0, 16); \\\n\
    VXC_ReadImage(vect1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src1, vect1, 16); \\\n\
    VXC_ReadImage(vect2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src2, vect2, 16); \\\n\
    VXC_ReadImage(vect3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src3, vect3, 16); \\\n\
    w0 = read_imagef(layer_norm_wi, coord_in.xw); \\\n\
    w1 = read_imagef(layer_norm_wf, coord_in.xw); \\\n\
    w2 = read_imagef(layer_norm_wc, coord_in.xw); \\\n\
    w3 = read_imagef(layer_norm_wo, coord_in.xw); \\\n\
    data_c_t = read_imagef(cell_state_in, coord_in.xy); \\\n\
    b0 = read_imagef(bias_i, coord_in.xw); \\\n\
    b1 = read_imagef(bias_f, coord_in.xw); \\\n\
    b2 = read_imagef(bias_c, coord_in.xw); \\\n\
    b3 = read_imagef(bias_o, coord_in.xw); \\\n\
 \\\n\
    VXC_DP4x4(data_i_t, src0, src0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_f_t, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_g_t, src2, src2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_o_t, src3, src3, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    data_i_t = data_i_t * w0 + b0; \\\n\
    data_f_t = data_f_t * w1 + b1; \\\n\
    data_g_t = data_g_t * w2 + b2; \\\n\
    data_o_t = data_o_t * w3 + b3; \\\n\
 \\\n\
    convert_type dst0; \\\n\
    data_i_t = act_func(data_i_t); \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    write_imagef(cell_state_out, coord_in.zy, data_c_t); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_LP_FP32(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_LP_FP32(I8,  , int4,  vxc_char4,  vxc_char4,  sigmoid)\n\
LSTMUNIT_LP_FP32(U8,  , int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_LP_FP32(I16, , int4,  vxc_short4, vxc_short4, sigmoid)\n\
LSTMUNIT_LP_FP32(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
LSTMUNIT_LP_FP32(I8,  _HARD, int4,  vxc_char4,  vxc_char4,  hard_sigmoid)\n\
LSTMUNIT_LP_FP32(U8,  _HARD, int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_LP_FP32(I16, _HARD, int4,  vxc_short4, vxc_short4, hard_sigmoid)\n\
\n\
#define LSTMUNIT_LP_FP16(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_LP_F16to##out_type_name##_F16##act_name( \\\n\
    __read_only  image2d_array_t  input_i_conv, \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_t        bias_i, \\\n\
    __read_only  image2d_t        bias_f, \\\n\
    __read_only  image2d_t        bias_c, \\\n\
    __read_only  image2d_t        bias_o, \\\n\
    __read_only  image2d_t        layer_norm_wi, \\\n\
    __read_only  image2d_t        layer_norm_wf, \\\n\
    __read_only  image2d_t        layer_norm_wc, \\\n\
    __read_only  image2d_t        layer_norm_wo, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_short8 vect0, vect1, vect2, vect3, vect4; \\\n\
    vxc_half8  src0, src1, src2, src3, src4; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    float4 w0, w1, w2, w3, b0, b1, b2, b3; \\\n\
    VXC_ReadImage(vect0, input_i_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src0, vect0, 16); \\\n\
    VXC_ReadImage(vect1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src1, vect1, 16); \\\n\
    VXC_ReadImage(vect2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src2, vect2, 16); \\\n\
    VXC_ReadImage(vect3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src3, vect3, 16); \\\n\
    VXC_ReadImage(vect4, cell_state_in, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src4, vect4, 16); \\\n\
    w0 = read_imagef(layer_norm_wi, coord_in.xw); \\\n\
    w1 = read_imagef(layer_norm_wf, coord_in.xw); \\\n\
    w2 = read_imagef(layer_norm_wc, coord_in.xw); \\\n\
    w3 = read_imagef(layer_norm_wo, coord_in.xw); \\\n\
    b0 = read_imagef(bias_i, coord_in.xw); \\\n\
    b1 = read_imagef(bias_f, coord_in.xw); \\\n\
    b2 = read_imagef(bias_c, coord_in.xw); \\\n\
    b3 = read_imagef(bias_o, coord_in.xw); \\\n\
 \\\n\
    VXC_DP4x4(data_i_t, src0, src0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_f_t, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_g_t, src2, src2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_o_t, src3, src3, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_c_t, src4, src4, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    data_i_t = data_i_t * w0 + b0; \\\n\
    data_f_t = data_f_t * w1 + b1; \\\n\
    data_g_t = data_g_t * w2 + b2; \\\n\
    data_o_t = data_o_t * w3 + b3; \\\n\
 \\\n\
    convert_type dst0; \\\n\
    half4 cell_data; \\\n\
    data_i_t = act_func(data_i_t); \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    _viv_asm(CONV, cell_data, data_c_t); \\\n\
    VXC_DP4x4(src0, cell_data, cell_data, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtractHalf4_4x4); \\\n\
    _viv_asm(COPY, vect0, src0, 8); \\\n\
    VXC_WriteImage(cell_state_out, coord_in.zy, vect0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_LP_FP16(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_LP_FP16(I8,  , int4,  vxc_char4,  vxc_char4,  sigmoid)\n\
LSTMUNIT_LP_FP16(U8,  , int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_LP_FP16(I16, , int4,  vxc_short4, vxc_short4, sigmoid)\n\
LSTMUNIT_LP_FP16(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
LSTMUNIT_LP_FP16(I8,  _HARD, int4,  vxc_char4,  vxc_char4,  hard_sigmoid)\n\
LSTMUNIT_LP_FP16(U8,  _HARD, int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_LP_FP16(I16, _HARD, int4,  vxc_short4, vxc_short4, hard_sigmoid)\n\
"; /* end of vsi_nn_kernel_lstmunit_activation_LP_F16_vx*/

static const char vsi_nn_kernel_lstmunit_activation_L_F16_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform float logE;\n\
_viv_uniform float twoLogE;\n\
_viv_uniform float forget_bias;\n\
float4 sigmoid(float4 x)\n\
{\n\
    x *= -logE;\n\
    x = 1 + exp2(x);\n\
    return 1 / x;\n\
}\n\
float4 hard_sigmoid(float4 x)\n\
{\n\
    x = 0.2 * x + 0.5;\n\
    x = clamp(x, 0, 1);\n\
    return x;\n\
}\n\
float4 tangentH(float4 x)\n\
{\n\
    x *= -twoLogE;\n\
    x = 1 + exp2(x);\n\
    x = 1 / x;\n\
    return 2 * x - 1;\n\
}\n\
_viv_uniform float outputScale;\n\
_viv_uniform float outputZP;\n\
_viv_uniform VXC_512Bits uniExtract8Data_2x8;\n\
_viv_uniform VXC_512Bits uniFp16toFp32_4x4;\n\
_viv_uniform float4 clip_Min_F;\n\
_viv_uniform float4 clip_Max_F;\n\
_viv_uniform VXC_512Bits uniExtractHalf4_4x4;\n\
\n\
#define LSTMUNIT_L_FP32(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_L_F16to##out_type_name##_F32##act_name( \\\n\
    __read_only  image2d_array_t  input_i_conv, \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_t        bias_i, \\\n\
    __read_only  image2d_t        bias_f, \\\n\
    __read_only  image2d_t        bias_c, \\\n\
    __read_only  image2d_t        bias_o, \\\n\
    __read_only  image2d_t        layer_norm_wi, \\\n\
    __read_only  image2d_t        layer_norm_wf, \\\n\
    __read_only  image2d_t        layer_norm_wc, \\\n\
    __read_only  image2d_t        layer_norm_wo, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __write_only image2d_t        h_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_short8 vect0, vect1, vect2, vect3; \\\n\
    vxc_half8  src0, src1, src2, src3; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    float4 w0, w1, w2, w3, b0, b1, b2, b3; \\\n\
    VXC_ReadImage(vect0, input_i_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src0, vect0, 16); \\\n\
    VXC_ReadImage(vect1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src1, vect1, 16); \\\n\
    VXC_ReadImage(vect2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src2, vect2, 16); \\\n\
    VXC_ReadImage(vect3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src3, vect3, 16); \\\n\
    w0 = read_imagef(layer_norm_wi, coord_in.xw); \\\n\
    w1 = read_imagef(layer_norm_wf, coord_in.xw); \\\n\
    w2 = read_imagef(layer_norm_wc, coord_in.xw); \\\n\
    w3 = read_imagef(layer_norm_wo, coord_in.xw); \\\n\
    data_c_t = read_imagef(cell_state_in, coord_in.xy); \\\n\
    b0 = read_imagef(bias_i, coord_in.xw); \\\n\
    b1 = read_imagef(bias_f, coord_in.xw); \\\n\
    b2 = read_imagef(bias_c, coord_in.xw); \\\n\
    b3 = read_imagef(bias_o, coord_in.xw); \\\n\
 \\\n\
    VXC_DP4x4(data_i_t, src0, src0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_f_t, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_g_t, src2, src2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_o_t, src3, src3, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    data_i_t = data_i_t * w0 + b0; \\\n\
    data_f_t = data_f_t * w1 + b1; \\\n\
    data_g_t = data_g_t * w2 + b2; \\\n\
    data_o_t = data_o_t * w3 + b3; \\\n\
 \\\n\
    convert_type dst0; \\\n\
    data_i_t = act_func(data_i_t); \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    write_imagef(cell_state_out, coord_in.zy, data_c_t); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_WriteImage(h_state_out, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_L_FP32(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_L_FP32(I8,  , int4,  vxc_char4,  vxc_char4,  sigmoid)\n\
LSTMUNIT_L_FP32(U8,  , int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_L_FP32(I16, , int4,  vxc_short4, vxc_short4, sigmoid)\n\
LSTMUNIT_L_FP32(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
LSTMUNIT_L_FP32(I8,  _HARD, int4,  vxc_char4,  vxc_char4,  hard_sigmoid)\n\
LSTMUNIT_L_FP32(U8,  _HARD, int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_L_FP32(I16, _HARD, int4,  vxc_short4, vxc_short4, hard_sigmoid)\n\
\n\
#define LSTMUNIT_L_FP16(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_L_F16to##out_type_name##_F16##act_name( \\\n\
    __read_only  image2d_array_t  input_i_conv, \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_t        bias_i, \\\n\
    __read_only  image2d_t        bias_f, \\\n\
    __read_only  image2d_t        bias_c, \\\n\
    __read_only  image2d_t        bias_o, \\\n\
    __read_only  image2d_t        layer_norm_wi, \\\n\
    __read_only  image2d_t        layer_norm_wf, \\\n\
    __read_only  image2d_t        layer_norm_wc, \\\n\
    __read_only  image2d_t        layer_norm_wo, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __write_only image2d_t        h_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_short8 vect0, vect1, vect2, vect3, vect4; \\\n\
    vxc_half8  src0, src1, src2, src3, src4; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    float4 w0, w1, w2, w3, b0, b1, b2, b3; \\\n\
    VXC_ReadImage(vect0, input_i_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src0, vect0, 16); \\\n\
    VXC_ReadImage(vect1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src1, vect1, 16); \\\n\
    VXC_ReadImage(vect2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src2, vect2, 16); \\\n\
    VXC_ReadImage(vect3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src3, vect3, 16); \\\n\
    VXC_ReadImage(vect4, cell_state_in, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src4, vect4, 16); \\\n\
    w0 = read_imagef(layer_norm_wi, coord_in.xw); \\\n\
    w1 = read_imagef(layer_norm_wf, coord_in.xw); \\\n\
    w2 = read_imagef(layer_norm_wc, coord_in.xw); \\\n\
    w3 = read_imagef(layer_norm_wo, coord_in.xw); \\\n\
    b0 = read_imagef(bias_i, coord_in.xw); \\\n\
    b1 = read_imagef(bias_f, coord_in.xw); \\\n\
    b2 = read_imagef(bias_c, coord_in.xw); \\\n\
    b3 = read_imagef(bias_o, coord_in.xw); \\\n\
 \\\n\
    VXC_DP4x4(data_i_t, src0, src0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_f_t, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_g_t, src2, src2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_o_t, src3, src3, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_c_t, src4, src4, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    data_i_t = data_i_t * w0 + b0; \\\n\
    data_f_t = data_f_t * w1 + b1; \\\n\
    data_g_t = data_g_t * w2 + b2; \\\n\
    data_o_t = data_o_t * w3 + b3; \\\n\
 \\\n\
    convert_type dst0; \\\n\
    half4 cell_data; \\\n\
    data_i_t = act_func(data_i_t); \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    _viv_asm(CONV, cell_data, data_c_t); \\\n\
    VXC_DP4x4(src0, cell_data, cell_data, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtractHalf4_4x4); \\\n\
    _viv_asm(COPY, vect0, src0, 8); \\\n\
    VXC_WriteImage(cell_state_out, coord_in.zy, vect0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_WriteImage(h_state_out, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_L_FP16(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_L_FP16(I8,  , int4,  vxc_char4,  vxc_char4,  sigmoid)\n\
LSTMUNIT_L_FP16(U8,  , int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_L_FP16(I16, , int4,  vxc_short4, vxc_short4, sigmoid)\n\
LSTMUNIT_L_FP16(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
LSTMUNIT_L_FP16(I8,  _HARD, int4,  vxc_char4,  vxc_char4,  hard_sigmoid)\n\
LSTMUNIT_L_FP16(U8,  _HARD, int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_L_FP16(I16, _HARD, int4,  vxc_short4, vxc_short4, hard_sigmoid)\n\
\n\
"; /* end of vsi_nn_kernel_lstmunit_activation_L_F16_vx*/

static const char vsi_nn_kernel_lstmunit_activation_SP_F16_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform float logE;\n\
_viv_uniform float twoLogE;\n\
_viv_uniform float forget_bias;\n\
float4 sigmoid(float4 x)\n\
{\n\
    x *= -logE;\n\
    x = 1 + exp2(x);\n\
    return 1 / x;\n\
}\n\
float4 hard_sigmoid(float4 x)\n\
{\n\
    x = 0.2 * x + 0.5;\n\
    x = clamp(x, 0, 1);\n\
    return x;\n\
}\n\
float4 tangentH(float4 x)\n\
{\n\
    x *= -twoLogE;\n\
    x = 1 + exp2(x);\n\
    x = 1 / x;\n\
    return 2 * x - 1;\n\
}\n\
_viv_uniform float outputScale;\n\
_viv_uniform float outputZP;\n\
_viv_uniform VXC_512Bits uniExtract8Data_2x8;\n\
_viv_uniform VXC_512Bits uniFp16toFp32_4x4;\n\
_viv_uniform float4 clip_Min_F;\n\
_viv_uniform float4 clip_Max_F;\n\
_viv_uniform VXC_512Bits uniExtractHalf4_4x4;\n\
_viv_uniform VXC_512Bits uniFp16AddFp16toFp32_4x4;\n\
\n\
#define LSTMUNIT_SP_FP16_FP32(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_SP_F16to##out_type_name##_F32##act_name( \\\n\
    __read_only  image2d_array_t  input_i_conv, \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_array_t  hstate_i_conv, \\\n\
    __read_only  image2d_array_t  hstate_f_conv, \\\n\
    __read_only  image2d_array_t  hstate_c_conv, \\\n\
    __read_only  image2d_array_t  hstate_o_conv, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_short8 vect0, vect1, vect2, vect3; \\\n\
    vxc_half8  src0, src1, src2, src3; \\\n\
    vxc_short8 vect10, vect11, vect12, vect13; \\\n\
    vxc_half8  src10, src11, src12, src13; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    VXC_ReadImage(vect0, input_i_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src0, vect0, 16); \\\n\
    VXC_ReadImage(vect10, hstate_i_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src10, vect10, 16); \\\n\
    VXC_ReadImage(vect1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src1, vect1, 16); \\\n\
    VXC_ReadImage(vect11, hstate_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src11, vect11, 16); \\\n\
    VXC_ReadImage(vect2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src2, vect2, 16); \\\n\
    VXC_ReadImage(vect12, hstate_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src12, vect12, 16); \\\n\
    VXC_ReadImage(vect3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src3, vect3, 16); \\\n\
    VXC_ReadImage(vect13, hstate_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src13, vect13, 16); \\\n\
    data_c_t = read_imagef(cell_state_in, coord_in.zy); \\\n\
 \\\n\
    VXC_DP4x4(data_i_t, src0, src10, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_f_t, src1, src11, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_g_t, src2, src12, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_o_t, src3, src13, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
 \\\n\
    convert_type dst0; \\\n\
    data_i_t = act_func(data_i_t); \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    write_imagef(cell_state_out, coord_in.zy, data_c_t); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_SP_FP16_FP32(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_SP_FP16_FP32(I8,  , int4,  vxc_char4,  vxc_char4,  sigmoid)\n\
LSTMUNIT_SP_FP16_FP32(U8,  , int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_SP_FP16_FP32(I16, , int4,  vxc_short4, vxc_short4, sigmoid)\n\
LSTMUNIT_SP_FP16_FP32(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
LSTMUNIT_SP_FP16_FP32(I8,  _HARD, int4,  vxc_char4,  vxc_char4,  hard_sigmoid)\n\
LSTMUNIT_SP_FP16_FP32(U8,  _HARD, int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_SP_FP16_FP32(I16, _HARD, int4,  vxc_short4, vxc_short4, hard_sigmoid)\n\
\n\
#define LSTMUNIT_SP_FP16_FP16(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_SP_F16to##out_type_name##_F16##act_name( \\\n\
    __read_only  image2d_array_t  input_i_conv, \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_array_t  hstate_i_conv, \\\n\
    __read_only  image2d_array_t  hstate_f_conv, \\\n\
    __read_only  image2d_array_t  hstate_c_conv, \\\n\
    __read_only  image2d_array_t  hstate_o_conv, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_short8 vect0, vect1, vect2, vect3, vect4; \\\n\
    vxc_half8  src0, src1, src2, src3, src4; \\\n\
    vxc_short8 vect10, vect11, vect12, vect13; \\\n\
    vxc_half8  src10, src11, src12, src13; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    VXC_ReadImage(vect0, input_i_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src0, vect0, 16); \\\n\
    VXC_ReadImage(vect10, hstate_i_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src10, vect10, 16); \\\n\
    VXC_ReadImage(vect1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src1, vect1, 16); \\\n\
    VXC_ReadImage(vect11, hstate_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src11, vect11, 16); \\\n\
    VXC_ReadImage(vect2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src2, vect2, 16); \\\n\
    VXC_ReadImage(vect12, hstate_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src12, vect12, 16); \\\n\
    VXC_ReadImage(vect3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src3, vect3, 16); \\\n\
    VXC_ReadImage(vect13, hstate_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src13, vect13, 16); \\\n\
    VXC_ReadImage(vect4, cell_state_in, coord_in.zy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src4, vect4, 16); \\\n\
 \\\n\
    VXC_DP4x4(data_i_t, src0, src10, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_f_t, src1, src11, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_g_t, src2, src12, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_c_t, src4, src4, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_o_t, src3, src13, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
 \\\n\
    convert_type dst0; \\\n\
    half4 dst_cell; \\\n\
    data_i_t = act_func(data_i_t); \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    _viv_asm(CONV, dst_cell, data_c_t); \\\n\
    VXC_DP4x4(src0, dst_cell, dst_cell, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtractHalf4_4x4); \\\n\
    _viv_asm(COPY, vect0, src0, 8); \\\n\
    VXC_WriteImage(cell_state_out, coord_in.zy, vect0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_SP_FP16_FP16(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_SP_FP16_FP16(I8,  , int4,  vxc_char4,  vxc_char4,  sigmoid)\n\
LSTMUNIT_SP_FP16_FP16(U8,  , int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_SP_FP16_FP16(I16, , int4,  vxc_short4, vxc_short4, sigmoid)\n\
LSTMUNIT_SP_FP16_FP16(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
LSTMUNIT_SP_FP16_FP16(I8,  _HARD, int4,  vxc_char4,  vxc_char4,  hard_sigmoid)\n\
LSTMUNIT_SP_FP16_FP16(U8,  _HARD, int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_SP_FP16_FP16(I16, _HARD, int4,  vxc_short4, vxc_short4, hard_sigmoid)\n\
"; /* end of vsi_nn_kernel_lstmunit_activation_SP_F16_vx*/

static const char vsi_nn_kernel_lstmunit_activation_SP_U8_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform float logE;\n\
_viv_uniform float twoLogE;\n\
_viv_uniform float forget_bias;\n\
float4 sigmoid(float4 x)\n\
{\n\
    x *= -logE;\n\
    x = 1 + exp2(x);\n\
    return 1 / x;\n\
}\n\
float4 hard_sigmoid(float4 x)\n\
{\n\
    x = 0.2 * x + 0.5;\n\
    x = clamp(x, 0, 1);\n\
    return x;\n\
}\n\
float4 tangentH(float4 x)\n\
{\n\
    x *= -twoLogE;\n\
    x = 1 + exp2(x);\n\
    x = 1 / x;\n\
    return 2 * x - 1;\n\
}\n\
_viv_uniform float outputScale;\n\
_viv_uniform float outputZP;\n\
_viv_uniform VXC_512Bits uniExtract8Data_2x8;\n\
_viv_uniform VXC_512Bits uniFp16toFp32_4x4;\n\
_viv_uniform float4 clip_Min_F;\n\
_viv_uniform float4 clip_Max_F;\n\
_viv_uniform VXC_512Bits uniExtractHalf4_4x4;\n\
_viv_uniform VXC_512Bits uniU8AddS32_4x4;\n\
_viv_uniform int4 input0Array_ZP;\n\
_viv_uniform int4 input1Array_ZP;\n\
_viv_uniform float4 input0Array_Scale;\n\
_viv_uniform float4 input1Array_Scale;\n\
\n\
#define LSTMUNIT_SP_U8_FP32(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_SP_U8to##out_type_name##_F32##act_name( \\\n\
    __read_only  image2d_array_t  input_i_conv, \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_array_t  hstate_i_conv, \\\n\
    __read_only  image2d_array_t  hstate_f_conv, \\\n\
    __read_only  image2d_array_t  hstate_c_conv, \\\n\
    __read_only  image2d_array_t  hstate_o_conv, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_uchar4 src0,  src1,  src2,  src3; \\\n\
    vxc_uchar4 src10, src11, src12, src13; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    float4 vecA, vecB; \\\n\
    VXC_ReadImage(src0, input_i_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src10, hstate_i_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src11, hstate_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src12, hstate_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src13, hstate_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    data_c_t = read_imagef(cell_state_in, coord_in.zy); \\\n\
 \\\n\
    VXC_DP4x4(vecA, src0, input0Array_ZP.xxxx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src10, input1Array_ZP.xxxx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_i_t = vecA * input0Array_Scale.xxxx + vecB * input1Array_Scale.xxxx; \\\n\
    VXC_DP4x4(vecA, src1, input0Array_ZP.yyyy, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src11, input1Array_ZP.yyyy, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_f_t = vecA * input0Array_Scale.yyyy + vecB * input1Array_Scale.yyyy; \\\n\
    VXC_DP4x4(vecA, src2, input0Array_ZP.zzzz, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src12, input1Array_ZP.zzzz, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_g_t = vecA * input0Array_Scale.zzzz + vecB * input1Array_Scale.zzzz; \\\n\
    VXC_DP4x4(vecA, src3, input0Array_ZP.wwww, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src13, input1Array_ZP.wwww, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_o_t = vecA * input0Array_Scale.wwww + vecB * input1Array_Scale.wwww; \\\n\
 \\\n\
    convert_type dst0; \\\n\
    data_i_t = act_func(data_i_t); \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    write_imagef(cell_state_out, coord_in.zy, data_c_t); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_SP_U8_FP32(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_SP_U8_FP32(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
\n\
#define LSTMUNIT_SP_U8_FP16(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_SP_U8to##out_type_name##_F16##act_name( \\\n\
    __read_only  image2d_array_t  input_i_conv, \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_array_t  hstate_i_conv, \\\n\
    __read_only  image2d_array_t  hstate_f_conv, \\\n\
    __read_only  image2d_array_t  hstate_c_conv, \\\n\
    __read_only  image2d_array_t  hstate_o_conv, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_short8 vect0; \\\n\
    vxc_half8  src4; \\\n\
    vxc_uchar4 src0,  src1,  src2,  src3; \\\n\
    vxc_uchar4 src10, src11, src12, src13; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    float4 vecA, vecB; \\\n\
    VXC_ReadImage(src0, input_i_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src10, hstate_i_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src11, hstate_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src12, hstate_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src13, hstate_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(vect0, cell_state_in, coord_in.zy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src4, vect0, 16); \\\n\
 \\\n\
    VXC_DP4x4(data_c_t, src4, src4, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(vecA, src0, input0Array_ZP.xxxx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src10, input1Array_ZP.xxxx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_i_t = vecA * input0Array_Scale.xxxx + vecB * input1Array_Scale.xxxx; \\\n\
    VXC_DP4x4(vecA, src1, input0Array_ZP.yyyy, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src11, input1Array_ZP.yyyy, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_f_t = vecA * input0Array_Scale.yyyy + vecB * input1Array_Scale.yyyy; \\\n\
    VXC_DP4x4(vecA, src2, input0Array_ZP.zzzz, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src12, input1Array_ZP.zzzz, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_g_t = vecA * input0Array_Scale.zzzz + vecB * input1Array_Scale.zzzz; \\\n\
    VXC_DP4x4(vecA, src3, input0Array_ZP.wwww, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src13, input1Array_ZP.wwww, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_o_t = vecA * input0Array_Scale.wwww + vecB * input1Array_Scale.wwww; \\\n\
 \\\n\
    convert_type dst0; \\\n\
    half4 dst_cell; \\\n\
    data_i_t = act_func(data_i_t); \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    _viv_asm(CONV, dst_cell, data_c_t); \\\n\
    VXC_DP4x4(src4, dst_cell, dst_cell, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtractHalf4_4x4); \\\n\
    _viv_asm(COPY, vect0, src4, 8); \\\n\
    VXC_WriteImage(cell_state_out, coord_in.zy, vect0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_SP_U8_FP16(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_SP_U8_FP16(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
"; /* end of vsi_nn_kernel_lstmunit_activation_SP_U8_vx*/

static const char vsi_nn_kernel_lstmunit_activation_S_F16_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform float logE;\n\
_viv_uniform float twoLogE;\n\
_viv_uniform float forget_bias;\n\
float4 sigmoid(float4 x)\n\
{\n\
    x *= -logE;\n\
    x = 1 + exp2(x);\n\
    return 1 / x;\n\
}\n\
float4 hard_sigmoid(float4 x)\n\
{\n\
    x = 0.2 * x + 0.5;\n\
    x = clamp(x, 0, 1);\n\
    return x;\n\
}\n\
float4 tangentH(float4 x)\n\
{\n\
    x *= -twoLogE;\n\
    x = 1 + exp2(x);\n\
    x = 1 / x;\n\
    return 2 * x - 1;\n\
}\n\
_viv_uniform float outputScale;\n\
_viv_uniform float outputZP;\n\
_viv_uniform VXC_512Bits uniExtract8Data_2x8;\n\
_viv_uniform VXC_512Bits uniFp16toFp32_4x4;\n\
_viv_uniform float4 clip_Min_F;\n\
_viv_uniform float4 clip_Max_F;\n\
_viv_uniform VXC_512Bits uniExtractHalf4_4x4;\n\
_viv_uniform VXC_512Bits uniFp16AddFp16toFp32_4x4;\n\
\n\
#define LSTMUNIT_S_FP16_FP32(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_S_F16to##out_type_name##_F32##act_name( \\\n\
    __read_only  image2d_array_t  input_i_conv, \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_array_t  hstate_i_conv, \\\n\
    __read_only  image2d_array_t  hstate_f_conv, \\\n\
    __read_only  image2d_array_t  hstate_c_conv, \\\n\
    __read_only  image2d_array_t  hstate_o_conv, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __write_only image2d_t        h_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_short8 vect0, vect1, vect2, vect3; \\\n\
    vxc_half8  src0, src1, src2, src3; \\\n\
    vxc_short8 vect10, vect11, vect12, vect13; \\\n\
    vxc_half8  src10, src11, src12, src13; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    VXC_ReadImage(vect0, input_i_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src0, vect0, 16); \\\n\
    VXC_ReadImage(vect10, hstate_i_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src10, vect10, 16); \\\n\
    VXC_ReadImage(vect1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src1, vect1, 16); \\\n\
    VXC_ReadImage(vect11, hstate_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src11, vect11, 16); \\\n\
    VXC_ReadImage(vect2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src2, vect2, 16); \\\n\
    VXC_ReadImage(vect12, hstate_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src12, vect12, 16); \\\n\
    VXC_ReadImage(vect3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src3, vect3, 16); \\\n\
    VXC_ReadImage(vect13, hstate_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src13, vect13, 16); \\\n\
    data_c_t = read_imagef(cell_state_in, coord_in.zy); \\\n\
 \\\n\
    VXC_DP4x4(data_i_t, src0, src10, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_f_t, src1, src11, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_g_t, src2, src12, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_o_t, src3, src13, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
 \\\n\
    convert_type dst0; \\\n\
    data_i_t = act_func(data_i_t); \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    write_imagef(cell_state_out, coord_in.zy, data_c_t); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_WriteImage(h_state_out, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_S_FP16_FP32(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_S_FP16_FP32(I8,  , int4,  vxc_char4,  vxc_char4,  sigmoid)\n\
LSTMUNIT_S_FP16_FP32(U8,  , int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_S_FP16_FP32(I16, , int4,  vxc_short4, vxc_short4, sigmoid)\n\
LSTMUNIT_S_FP16_FP32(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
LSTMUNIT_S_FP16_FP32(I8,  _HARD, int4,  vxc_char4,  vxc_char4,  hard_sigmoid)\n\
LSTMUNIT_S_FP16_FP32(U8,  _HARD, int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_S_FP16_FP32(I16, _HARD, int4,  vxc_short4, vxc_short4, hard_sigmoid)\n\
\n\
#define LSTMUNIT_S_FP16_FP16(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_S_F16to##out_type_name##_F16##act_name( \\\n\
    __read_only  image2d_array_t  input_i_conv, \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_array_t  hstate_i_conv, \\\n\
    __read_only  image2d_array_t  hstate_f_conv, \\\n\
    __read_only  image2d_array_t  hstate_c_conv, \\\n\
    __read_only  image2d_array_t  hstate_o_conv, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __write_only image2d_t        h_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_short8 vect0, vect1, vect2, vect3, vect4; \\\n\
    vxc_half8  src0, src1, src2, src3, src4; \\\n\
    vxc_short8 vect10, vect11, vect12, vect13; \\\n\
    vxc_half8  src10, src11, src12, src13; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    VXC_ReadImage(vect0, input_i_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src0, vect0, 16); \\\n\
    VXC_ReadImage(vect10, hstate_i_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src10, vect10, 16); \\\n\
    VXC_ReadImage(vect1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src1, vect1, 16); \\\n\
    VXC_ReadImage(vect11, hstate_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src11, vect11, 16); \\\n\
    VXC_ReadImage(vect2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src2, vect2, 16); \\\n\
    VXC_ReadImage(vect12, hstate_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src12, vect12, 16); \\\n\
    VXC_ReadImage(vect3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src3, vect3, 16); \\\n\
    VXC_ReadImage(vect13, hstate_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src13, vect13, 16); \\\n\
    VXC_ReadImage(vect4, cell_state_in, coord_in.zy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src4, vect4, 16); \\\n\
 \\\n\
    VXC_DP4x4(data_i_t, src0, src10, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_f_t, src1, src11, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_g_t, src2, src12, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_c_t, src4, src4, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(data_o_t, src3, src13, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16AddFp16toFp32_4x4); \\\n\
 \\\n\
    convert_type dst0; \\\n\
    half4 dst_cell; \\\n\
    data_i_t = act_func(data_i_t); \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    _viv_asm(CONV, dst_cell, data_c_t); \\\n\
    VXC_DP4x4(src0, dst_cell, dst_cell, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtractHalf4_4x4); \\\n\
    _viv_asm(COPY, vect0, src0, 8); \\\n\
    VXC_WriteImage(cell_state_out, coord_in.zy, vect0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_WriteImage(h_state_out, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_S_FP16_FP16(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_S_FP16_FP16(I8,  , int4,  vxc_char4,  vxc_char4,  sigmoid)\n\
LSTMUNIT_S_FP16_FP16(U8,  , int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_S_FP16_FP16(I16, , int4,  vxc_short4, vxc_short4, sigmoid)\n\
LSTMUNIT_S_FP16_FP16(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
LSTMUNIT_S_FP16_FP16(I8,  _HARD, int4,  vxc_char4,  vxc_char4,  hard_sigmoid)\n\
LSTMUNIT_S_FP16_FP16(U8,  _HARD, int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_S_FP16_FP16(I16, _HARD, int4,  vxc_short4, vxc_short4, hard_sigmoid)\n\
"; /* end of vsi_nn_kernel_lstmunit_activation_S_F16_vx*/

static const char vsi_nn_kernel_lstmunit_activation_S_U8_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform float logE;\n\
_viv_uniform float twoLogE;\n\
_viv_uniform float forget_bias;\n\
float4 sigmoid(float4 x)\n\
{\n\
    x *= -logE;\n\
    x = 1 + exp2(x);\n\
    return 1 / x;\n\
}\n\
float4 hard_sigmoid(float4 x)\n\
{\n\
    x = 0.2 * x + 0.5;\n\
    x = clamp(x, 0, 1);\n\
    return x;\n\
}\n\
float4 tangentH(float4 x)\n\
{\n\
    x *= -twoLogE;\n\
    x = 1 + exp2(x);\n\
    x = 1 / x;\n\
    return 2 * x - 1;\n\
}\n\
_viv_uniform float outputScale;\n\
_viv_uniform float outputZP;\n\
_viv_uniform VXC_512Bits uniExtract8Data_2x8;\n\
_viv_uniform VXC_512Bits uniFp16toFp32_4x4;\n\
_viv_uniform float4 clip_Min_F;\n\
_viv_uniform float4 clip_Max_F;\n\
_viv_uniform VXC_512Bits uniExtractHalf4_4x4;\n\
_viv_uniform VXC_512Bits uniU8AddS32_4x4;\n\
_viv_uniform int4 input0Array_ZP;\n\
_viv_uniform int4 input1Array_ZP;\n\
_viv_uniform float4 input0Array_Scale;\n\
_viv_uniform float4 input1Array_Scale;\n\
\n\
#define LSTMUNIT_S_U8_FP32(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_S_U8to##out_type_name##_F32##act_name( \\\n\
    __read_only  image2d_array_t  input_i_conv, \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_array_t  hstate_i_conv, \\\n\
    __read_only  image2d_array_t  hstate_f_conv, \\\n\
    __read_only  image2d_array_t  hstate_c_conv, \\\n\
    __read_only  image2d_array_t  hstate_o_conv, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __write_only image2d_t        h_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_uchar4 src0,  src1,  src2,  src3; \\\n\
    vxc_uchar4 src10, src11, src12, src13; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    float4 vecA, vecB; \\\n\
    VXC_ReadImage(src0, input_i_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src10, hstate_i_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src11, hstate_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src12, hstate_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src13, hstate_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    data_c_t = read_imagef(cell_state_in, coord_in.zy); \\\n\
 \\\n\
    VXC_DP4x4(vecA, src0, input0Array_ZP.xxxx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src10, input1Array_ZP.xxxx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_i_t = vecA * input0Array_Scale.xxxx + vecB * input1Array_Scale.xxxx; \\\n\
    VXC_DP4x4(vecA, src1, input0Array_ZP.yyyy, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src11, input1Array_ZP.yyyy, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_f_t = vecA * input0Array_Scale.yyyy + vecB * input1Array_Scale.yyyy; \\\n\
    VXC_DP4x4(vecA, src2, input0Array_ZP.zzzz, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src12, input1Array_ZP.zzzz, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_g_t = vecA * input0Array_Scale.zzzz + vecB * input1Array_Scale.zzzz; \\\n\
    VXC_DP4x4(vecA, src3, input0Array_ZP.wwww, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src13, input1Array_ZP.wwww, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_o_t = vecA * input0Array_Scale.wwww + vecB * input1Array_Scale.wwww; \\\n\
 \\\n\
    convert_type dst0; \\\n\
    data_i_t = act_func(data_i_t); \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    write_imagef(cell_state_out, coord_in.zy, data_c_t); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_WriteImage(h_state_out, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_S_U8_FP32(U8,  , int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_S_U8_FP32(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_S_U8_FP32(U8,  _HARD, int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_S_U8_FP32(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
\n\
#define LSTMUNIT_S_U8_FP16(out_type_name, act_name, convert_type, dst_type, copy_type, act_func) \\\n\
__kernel void vxLSTMUnit_S_U8to##out_type_name##_F16##act_name( \\\n\
    __read_only  image2d_array_t  input_i_conv, \\\n\
    __read_only  image2d_array_t  input_f_conv, \\\n\
    __read_only  image2d_array_t  input_c_conv, \\\n\
    __read_only  image2d_array_t  input_o_conv, \\\n\
    __read_only  image2d_t        cell_state_in, \\\n\
    __read_only  image2d_array_t  hstate_i_conv, \\\n\
    __read_only  image2d_array_t  hstate_f_conv, \\\n\
    __read_only  image2d_array_t  hstate_c_conv, \\\n\
    __read_only  image2d_array_t  hstate_o_conv, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __write_only image2d_t        cell_state_out, \\\n\
    __write_only image2d_t        h_state_out, \\\n\
    __read_only  image2d_t        lstmunit_param \\\n\
    ) \\\n\
{ \\\n\
    int4 coord_in = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), 0); \\\n\
    vxc_short8 vect0; \\\n\
    vxc_half8  src4; \\\n\
    vxc_uchar4 src0,  src1,  src2,  src3; \\\n\
    vxc_uchar4 src10, src11, src12, src13; \\\n\
    float4 data_i_t, data_f_t, data_g_t, data_o_t, data_c_t; \\\n\
    float4 vecA, vecB; \\\n\
    VXC_ReadImage(src0, input_i_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src10, hstate_i_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src1, input_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src11, hstate_f_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src2, input_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src12, hstate_c_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src3, input_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src13, hstate_o_conv, coord_in.xy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(vect0, cell_state_in, coord_in.zy, 0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src4, vect0, 16); \\\n\
 \\\n\
    VXC_DP4x4(data_c_t, src4, src4, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniFp16toFp32_4x4); \\\n\
    VXC_DP4x4(vecA, src0, input0Array_ZP.xxxx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src10, input1Array_ZP.xxxx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_i_t = vecA * input0Array_Scale.xxxx + vecB * input1Array_Scale.xxxx; \\\n\
    VXC_DP4x4(vecA, src1, input0Array_ZP.yyyy, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src11, input1Array_ZP.yyyy, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_f_t = vecA * input0Array_Scale.yyyy + vecB * input1Array_Scale.yyyy; \\\n\
    VXC_DP4x4(vecA, src2, input0Array_ZP.zzzz, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src12, input1Array_ZP.zzzz, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_g_t = vecA * input0Array_Scale.zzzz + vecB * input1Array_Scale.zzzz; \\\n\
    VXC_DP4x4(vecA, src3, input0Array_ZP.wwww, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    VXC_DP4x4(vecB, src13, input1Array_ZP.wwww, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardInf, 0), uniU8AddS32_4x4);\\\n\
    data_o_t = vecA * input0Array_Scale.wwww + vecB * input1Array_Scale.wwww; \\\n\
 \\\n\
    convert_type dst0; \\\n\
    half4 dst_cell; \\\n\
    data_i_t = act_func(data_i_t); \\\n\
    data_f_t = act_func(data_f_t + forget_bias); \\\n\
    data_g_t = tangentH(data_g_t); \\\n\
    data_i_t = data_i_t * data_g_t; \\\n\
    data_c_t = data_c_t * data_f_t + data_i_t; \\\n\
    data_o_t = act_func(data_o_t); \\\n\
    data_c_t = data_c_t > clip_Max_F ? clip_Max_F : data_c_t; \\\n\
    data_c_t = data_c_t < clip_Min_F ? clip_Min_F : data_c_t; \\\n\
    _viv_asm(CONV, dst_cell, data_c_t); \\\n\
    VXC_DP4x4(src4, dst_cell, dst_cell, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtractHalf4_4x4); \\\n\
    _viv_asm(COPY, vect0, src4, 8); \\\n\
    VXC_WriteImage(cell_state_out, coord_in.zy, vect0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    data_c_t = tangentH(data_c_t); \\\n\
    data_o_t = data_o_t * data_c_t * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, data_o_t); \\\n\
    dst_type dst1; \\\n\
    VXC_DP2x8(dst1, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtract8Data_2x8); \\\n\
    copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_WriteImage(h_state_out, coord_in.zy, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
LSTMUNIT_S_U8_FP16(U8,  , int4,  vxc_uchar4, vxc_uchar4, sigmoid)\n\
LSTMUNIT_S_U8_FP16(F16, , half4, vxc_half4,  vxc_short4, sigmoid)\n\
LSTMUNIT_S_U8_FP16(U8,  _HARD, int4,  vxc_uchar4, vxc_uchar4, hard_sigmoid)\n\
LSTMUNIT_S_U8_FP16(F16, _HARD, half4, vxc_half4,  vxc_short4, hard_sigmoid)\n\
"; /* end of vsi_nn_kernel_lstmunit_activation_S_U8_vx*/

static const char vsi_nn_kernel_matrixmul_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform int input1_ZP;\n\
_viv_uniform int input2_ZP;\n\
_viv_uniform int output_ZP;\n\
_viv_uniform float input1Scale;\n\
_viv_uniform float input2Scale;\n\
_viv_uniform float outputScale;\n\
_viv_uniform VXC_512Bits uniConvertUint8SubZpToFp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertInt32toUint8_2x8;\n\
\n\
__kernel void gemm_block4x4_u8(image2d_array_t inputA,\n\
                        image2d_array_t inputB,\n\
                        image2d_array_t output,\n\
                                    int transposeA,\n\
                                    int transposeB,\n\
                                    int adjointA,\n\
                                    int adjointB,\n\
                        uint M, uint K, uint N)\n\
{\n\
    uint gidy = get_global_id(1);\n\
\n\
    vxc_uchar16 srcA, srcB, outC;\n\
\n\
    int4 coord_a = (int4)(0, gidy, get_global_id(2), 0);\n\
    int4 coord_b = (int4)(get_global_id(0), 0, get_global_id(2), 0);\n\
\n\
    vxc_float4 sum0 = (vxc_float4)(0);\n\
    vxc_float4 sum1 = (vxc_float4)(0);\n\
    vxc_float4 sum2 = (vxc_float4)(0);\n\
    vxc_float4 sum3 = (vxc_float4)(0);\n\
\n\
    short zp1 = input1_ZP;\n\
    short zp2 = input2_ZP;\n\
\n\
    for(int i = 0; i < K; i+=4)\n\
    {\n\
        vxc_float4 tempA0, tempA1, tempA2, tempA3;\n\
        vxc_float4 tempB0, tempB1, tempB2, tempB3;\n\
\n\
        coord_a.x = i;\n\
        coord_b.y = i;\n\
        VXC_ReadImage2DArray(srcA, inputA, coord_a, VXC_5BITOFFSET_XY(0, 0),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcB, inputB, coord_b, VXC_5BITOFFSET_XY(0, 0),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        VXC_DP4x4(tempA0, srcA, zp1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
            uniConvertUint8SubZpToFp32_4x4);\n\
        tempA0 *= input1Scale;\n\
        VXC_DP4x4(tempB0, srcB, zp2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
            uniConvertUint8SubZpToFp32_4x4);\n\
        tempB0 *= input2Scale;\n\
\n\
        VXC_ReadImage2DArray(srcA, inputA, coord_a, VXC_5BITOFFSET_XY(0, 1),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcB, inputB, coord_b, VXC_5BITOFFSET_XY(0, 1),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        VXC_DP4x4(tempA1, srcA, zp1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
            uniConvertUint8SubZpToFp32_4x4);\n\
        tempA1 *= input1Scale;\n\
        VXC_DP4x4(tempB1, srcB, zp2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
            uniConvertUint8SubZpToFp32_4x4);\n\
        tempB1 *= input2Scale;\n\
\n\
        VXC_ReadImage2DArray(srcA, inputA, coord_a, VXC_5BITOFFSET_XY(0, 2),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcB, inputB, coord_b, VXC_5BITOFFSET_XY(0, 2),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        VXC_DP4x4(tempA2, srcA, zp1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
            uniConvertUint8SubZpToFp32_4x4);\n\
        tempA2 *= input1Scale;\n\
        VXC_DP4x4(tempB2, srcB, zp2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
            uniConvertUint8SubZpToFp32_4x4);\n\
        tempB2 *= input2Scale;\n\
\n\
        VXC_ReadImage2DArray(srcA, inputA, coord_a, VXC_5BITOFFSET_XY(0, 3),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcB, inputB, coord_b, VXC_5BITOFFSET_XY(0, 3),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        VXC_DP4x4(tempA3, srcA, zp1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
            uniConvertUint8SubZpToFp32_4x4);\n\
        tempA3 *= input1Scale;\n\
        VXC_DP4x4(tempB3, srcB, zp2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
            uniConvertUint8SubZpToFp32_4x4);\n\
        tempB3 *= input2Scale;\n\
\n\
        sum0 += (tempA0.x * tempB0 + tempA0.y * tempB1 + tempA0.z * tempB2 + tempA0.w * tempB3);\n\
        sum1 += (tempA1.x * tempB0 + tempA1.y * tempB1 + tempA1.z * tempB2 + tempA1.w * tempB3);\n\
        sum2 += (tempA2.x * tempB0 + tempA2.y * tempB1 + tempA2.z * tempB2 + tempA2.w * tempB3);\n\
        sum3 += (tempA3.x * tempB0 + tempA3.y * tempB1 + tempA3.z * tempB2 + tempA3.w * tempB3);\n\
    }\n\
    vxc_int4 tmpOut0, tmpOut1;\n\
    coord_b.y = gidy;\n\
    tmpOut0 = convert_int4_rte(sum0 * outputScale + output_ZP);\n\
    tmpOut1 = convert_int4_rte(sum1 * outputScale + output_ZP);\n\
    VXC_DP2x8(outC, tmpOut0, tmpOut1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1),\\\n\
        uniConvertInt32toUint8_2x8);\n\
    VXC_WriteImage2DArray(output, coord_b, outC.s0123, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    coord_b.y++;\n\
    VXC_WriteImage2DArray(output, coord_b, outC.s4567, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
\n\
    coord_b.y++;\n\
    tmpOut0 = convert_int4_rte(sum2 * outputScale + output_ZP);\n\
    tmpOut1 = convert_int4_rte(sum3 * outputScale + output_ZP);\n\
    VXC_DP2x8(outC, tmpOut0, tmpOut1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1),\\\n\
        uniConvertInt32toUint8_2x8);\n\
    VXC_WriteImage2DArray(output, coord_b, outC.s0123, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    coord_b.y++;\n\
    VXC_WriteImage2DArray(output, coord_b, outC.s4567, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void gemm_block4x4_fp16_u8_fp16(image2d_array_t inputA,\n\
                        image2d_array_t inputB,\n\
                        image2d_array_t output,\n\
                                    int transposeA,\n\
                                    int transposeB,\n\
                                    int adjointA,\n\
                                    int adjointB,\n\
                        uint M, uint K, uint N)\n\
{\n\
    uint gidy = get_global_id(1);\n\
\n\
    half4 valA, valC;\n\
    vxc_short8 srcA, outC, tmpA;\n\
    vxc_uchar16 srcB;\n\
\n\
    int4 coord_a = (int4)(0, gidy, get_global_id(2), 0);\n\
    int4 coord_b = (int4)(get_global_id(0), 0, get_global_id(2), 0);\n\
\n\
    vxc_float4 sum0 = (vxc_float4)(0);\n\
    vxc_float4 sum1 = (vxc_float4)(0);\n\
    vxc_float4 sum2 = (vxc_float4)(0);\n\
    vxc_float4 sum3 = (vxc_float4)(0);\n\
\n\
    short zp2 = input2_ZP;\n\
\n\
    for(int i = 0; i < K; i+=4)\n\
    {\n\
        vxc_float4 tempA0, tempA1, tempA2, tempA3;\n\
        vxc_float4 tempB0, tempB1, tempB2, tempB3;\n\
\n\
        coord_a.x = i;\n\
        coord_b.y = i;\n\
        VXC_ReadImage2DArray(srcA, inputA, coord_a, VXC_5BITOFFSET_XY(0, 0),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcB, inputB, coord_b, VXC_5BITOFFSET_XY(0, 0),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        tmpA = srcA.s04152637;\n\
        _viv_asm(COPY, valA, tmpA, 16);\n\
        _viv_asm(CONV, tempA0, valA);\n\
        VXC_DP4x4(tempB0, srcB, zp2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
            uniConvertUint8SubZpToFp32_4x4);\n\
        tempB0 *= input2Scale;\n\
\n\
        VXC_ReadImage2DArray(srcA, inputA, coord_a, VXC_5BITOFFSET_XY(0, 1),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcB, inputB, coord_b, VXC_5BITOFFSET_XY(0, 1),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        tmpA = srcA.s04152637;\n\
        _viv_asm(COPY, valA, tmpA, 16);\n\
        _viv_asm(CONV, tempA1, valA);\n\
        VXC_DP4x4(tempB1, srcB, zp2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
            uniConvertUint8SubZpToFp32_4x4);\n\
        tempB1 *= input2Scale;\n\
\n\
        VXC_ReadImage2DArray(srcA, inputA, coord_a, VXC_5BITOFFSET_XY(0, 2),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcB, inputB, coord_b, VXC_5BITOFFSET_XY(0, 2),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        tmpA = srcA.s04152637;\n\
        _viv_asm(COPY, valA, tmpA, 16);\n\
        _viv_asm(CONV, tempA2, valA);\n\
        VXC_DP4x4(tempB2, srcB, zp2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
            uniConvertUint8SubZpToFp32_4x4);\n\
        tempB2 *= input2Scale;\n\
\n\
        VXC_ReadImage2DArray(srcA, inputA, coord_a, VXC_5BITOFFSET_XY(0, 3),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcB, inputB, coord_b, VXC_5BITOFFSET_XY(0, 3),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        tmpA = srcA.s04152637;\n\
        _viv_asm(COPY, valA, tmpA, 16);\n\
        _viv_asm(CONV, tempA3, valA);\n\
        VXC_DP4x4(tempB3, srcB, zp2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
            uniConvertUint8SubZpToFp32_4x4);\n\
        tempB3 *= input2Scale;\n\
\n\
        sum0 += (tempA0.x * tempB0 + tempA0.y * tempB1 + tempA0.z * tempB2 + tempA0.w * tempB3);\n\
        sum1 += (tempA1.x * tempB0 + tempA1.y * tempB1 + tempA1.z * tempB2 + tempA1.w * tempB3);\n\
        sum2 += (tempA2.x * tempB0 + tempA2.y * tempB1 + tempA2.z * tempB2 + tempA2.w * tempB3);\n\
        sum3 += (tempA3.x * tempB0 + tempA3.y * tempB1 + tempA3.z * tempB2 + tempA3.w * tempB3);\n\
    }\n\
\n\
    coord_b.y = gidy;\n\
    _viv_asm(CONV, valC, sum0);\n\
    _viv_asm(COPY, outC, valC, 16);\n\
    VXC_WriteImage2DArray(output, coord_b, outC.s0246, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
\n\
    coord_b.y++;\n\
    _viv_asm(CONV, valC, sum1);\n\
    _viv_asm(COPY, outC, valC, 16);\n\
    VXC_WriteImage2DArray(output, coord_b, outC.s0246, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
\n\
    coord_b.y++;\n\
    _viv_asm(CONV, valC, sum2);\n\
    _viv_asm(COPY, outC, valC, 16);\n\
    VXC_WriteImage2DArray(output, coord_b, outC.s0246, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
\n\
    coord_b.y++;\n\
    _viv_asm(CONV, valC, sum3);\n\
    _viv_asm(COPY, outC, valC, 16);\n\
    VXC_WriteImage2DArray(output, coord_b, outC.s0246, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void gemm_block4x4_fp16_u8_u8(image2d_array_t inputA,\n\
                        image2d_array_t inputB,\n\
                        image2d_array_t output,\n\
                                    int transposeA,\n\
                                    int transposeB,\n\
                                    int adjointA,\n\
                                    int adjointB,\n\
                        uint M, uint K, uint N)\n\
{\n\
    uint gidy = get_global_id(1);\n\
\n\
    vxc_short8 srcA, tmpA;\n\
    half4 valA;\n\
    vxc_uchar16 srcB, outC;\n\
\n\
    int4 coord_a = (int4)(0, gidy, get_global_id(2), 0);\n\
    int4 coord_b = (int4)(get_global_id(0), 0, get_global_id(2), 0);\n\
\n\
    vxc_float4 sum0 = (vxc_float4)(0);\n\
    vxc_float4 sum1 = (vxc_float4)(0);\n\
    vxc_float4 sum2 = (vxc_float4)(0);\n\
    vxc_float4 sum3 = (vxc_float4)(0);\n\
\n\
    short zp2 = input2_ZP;\n\
\n\
    for(int i = 0; i < K; i+=4)\n\
    {\n\
        vxc_float4 tempA0, tempA1, tempA2, tempA3;\n\
        vxc_float4 tempB0, tempB1, tempB2, tempB3;\n\
\n\
        coord_a.x = i;\n\
        coord_b.y = i;\n\
        VXC_ReadImage2DArray(srcA, inputA, coord_a, VXC_5BITOFFSET_XY(0, 0),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcB, inputB, coord_b, VXC_5BITOFFSET_XY(0, 0),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        tmpA = srcA.s04152637;\n\
        _viv_asm(COPY, valA, tmpA, 16);\n\
        _viv_asm(CONV, tempA0, valA);\n\
        VXC_DP4x4(tempB0, srcB, zp2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
            uniConvertUint8SubZpToFp32_4x4);\n\
        tempB0 *= input2Scale;\n\
\n\
        VXC_ReadImage2DArray(srcA, inputA, coord_a, VXC_5BITOFFSET_XY(0, 1),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcB, inputB, coord_b, VXC_5BITOFFSET_XY(0, 1),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        tmpA = srcA.s04152637;\n\
        _viv_asm(COPY, valA, tmpA, 16);\n\
        _viv_asm(CONV, tempA1, valA);\n\
        VXC_DP4x4(tempB1, srcB, zp2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
            uniConvertUint8SubZpToFp32_4x4);\n\
        tempB1 *= input2Scale;\n\
\n\
        VXC_ReadImage2DArray(srcA, inputA, coord_a, VXC_5BITOFFSET_XY(0, 2),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcB, inputB, coord_b, VXC_5BITOFFSET_XY(0, 2),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        tmpA = srcA.s04152637;\n\
        _viv_asm(COPY, valA, tmpA, 16);\n\
        _viv_asm(CONV, tempA2, valA);\n\
        VXC_DP4x4(tempB2, srcB, zp2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
            uniConvertUint8SubZpToFp32_4x4);\n\
        tempB2 *= input2Scale;\n\
\n\
        VXC_ReadImage2DArray(srcA, inputA, coord_a, VXC_5BITOFFSET_XY(0, 3),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcB, inputB, coord_b, VXC_5BITOFFSET_XY(0, 3),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        tmpA = srcA.s04152637;\n\
        _viv_asm(COPY, valA, tmpA, 16);\n\
        _viv_asm(CONV, tempA3, valA);\n\
        VXC_DP4x4(tempB3, srcB, zp2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
            uniConvertUint8SubZpToFp32_4x4);\n\
        tempB3 *= input2Scale;\n\
\n\
        sum0 += (tempA0.x * tempB0 + tempA0.y * tempB1 + tempA0.z * tempB2 + tempA0.w * tempB3);\n\
        sum1 += (tempA1.x * tempB0 + tempA1.y * tempB1 + tempA1.z * tempB2 + tempA1.w * tempB3);\n\
        sum2 += (tempA2.x * tempB0 + tempA2.y * tempB1 + tempA2.z * tempB2 + tempA2.w * tempB3);\n\
        sum3 += (tempA3.x * tempB0 + tempA3.y * tempB1 + tempA3.z * tempB2 + tempA3.w * tempB3);\n\
    }\n\
    vxc_int4 tmpOut0, tmpOut1;\n\
    coord_b.y = gidy;\n\
    tmpOut0 = convert_int4_rte(sum0 * outputScale + output_ZP);\n\
    tmpOut1 = convert_int4_rte(sum1 * outputScale + output_ZP);\n\
    VXC_DP2x8(outC, tmpOut0, tmpOut1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1),\\\n\
        uniConvertInt32toUint8_2x8);\n\
    VXC_WriteImage2DArray(output, coord_b, outC.s0123, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    coord_b.y++;\n\
    VXC_WriteImage2DArray(output, coord_b, outC.s4567, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
\n\
    coord_b.y++;\n\
    tmpOut0 = convert_int4_rte(sum2 * outputScale + output_ZP);\n\
    tmpOut1 = convert_int4_rte(sum3 * outputScale + output_ZP);\n\
    VXC_DP2x8(outC, tmpOut0, tmpOut1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1),\\\n\
        uniConvertInt32toUint8_2x8);\n\
    VXC_WriteImage2DArray(output, coord_b, outC.s0123, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    coord_b.y++;\n\
    VXC_WriteImage2DArray(output, coord_b, outC.s4567, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
}"; /* end of vsi_nn_kernel_matrixmul_vx*/

static const char vsi_nn_kernel_matrixmul_fp16_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform int output_ZP;\n\
_viv_uniform float outputScale;\n\
_viv_uniform VXC_512Bits uniConvertInt32toUint8_2x8;\n\
\n\
__kernel void gemm_block4x4_fp16(image2d_array_t inputA,\n\
                        image2d_array_t inputB,\n\
                        image2d_array_t output,\n\
                                    int transposeA,\n\
                                    int transposeB,\n\
                                    int adjointA,\n\
                                    int adjointB,\n\
                        uint M, uint K, uint N)\n\
{\n\
    uint gidy = get_global_id(1);\n\
\n\
    half4 valA, valB, valC;\n\
    vxc_short8 srcA, srcB, outC, tmpA, tmpB;\n\
\n\
    int4 coord_a = (int4)(0, gidy, get_global_id(2), 0);\n\
    int4 coord_b = (int4)(get_global_id(0), 0, get_global_id(2), 0);\n\
\n\
    vxc_float4 sum0 = (vxc_float4)(0);\n\
    vxc_float4 sum1 = (vxc_float4)(0);\n\
    vxc_float4 sum2 = (vxc_float4)(0);\n\
    vxc_float4 sum3 = (vxc_float4)(0);\n\
\n\
    for(int i = 0; i < K; i+=4)\n\
    {\n\
        vxc_float4 tempA0, tempA1, tempA2, tempA3;\n\
        vxc_float4 tempB0, tempB1, tempB2, tempB3;\n\
\n\
        coord_a.x = i;\n\
        coord_b.y = i;\n\
        VXC_ReadImage2DArray(srcA, inputA, coord_a, VXC_5BITOFFSET_XY(0, 0),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcB, inputB, coord_b, VXC_5BITOFFSET_XY(0, 0),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        tmpA = srcA.s04152637;\n\
        tmpB = srcB.s04152637;\n\
        _viv_asm(COPY, valA, tmpA, 16);\n\
        _viv_asm(COPY, valB, tmpB, 16);\n\
        _viv_asm(CONV, tempA0, valA);\n\
        _viv_asm(CONV, tempB0, valB);\n\
\n\
        VXC_ReadImage2DArray(srcA, inputA, coord_a, VXC_5BITOFFSET_XY(0, 1),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcB, inputB, coord_b, VXC_5BITOFFSET_XY(0, 1),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        tmpA = srcA.s04152637;\n\
        tmpB = srcB.s04152637;\n\
        _viv_asm(COPY, valA, tmpA, 16);\n\
        _viv_asm(COPY, valB, tmpB, 16);\n\
        _viv_asm(CONV, tempA1, valA);\n\
        _viv_asm(CONV, tempB1, valB);\n\
        VXC_ReadImage2DArray(srcA, inputA, coord_a, VXC_5BITOFFSET_XY(0, 2),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcB, inputB, coord_b, VXC_5BITOFFSET_XY(0, 2),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        tmpA = srcA.s04152637;\n\
        tmpB = srcB.s04152637;\n\
        _viv_asm(COPY, valA, tmpA, 16);\n\
        _viv_asm(COPY, valB, tmpB, 16);\n\
        _viv_asm(CONV, tempA2, valA);\n\
        _viv_asm(CONV, tempB2, valB);\n\
        VXC_ReadImage2DArray(srcA, inputA, coord_a, VXC_5BITOFFSET_XY(0, 3),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcB, inputB, coord_b, VXC_5BITOFFSET_XY(0, 3),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        tmpA = srcA.s04152637;\n\
        tmpB = srcB.s04152637;\n\
        _viv_asm(COPY, valA, tmpA, 16);\n\
        _viv_asm(COPY, valB, tmpB, 16);\n\
        _viv_asm(CONV, tempA3, valA);\n\
        _viv_asm(CONV, tempB3, valB);\n\
\n\
        sum0 += (tempA0.x * tempB0 + tempA0.y * tempB1 + tempA0.z * tempB2 + tempA0.w * tempB3);\n\
        sum1 += (tempA1.x * tempB0 + tempA1.y * tempB1 + tempA1.z * tempB2 + tempA1.w * tempB3);\n\
        sum2 += (tempA2.x * tempB0 + tempA2.y * tempB1 + tempA2.z * tempB2 + tempA2.w * tempB3);\n\
        sum3 += (tempA3.x * tempB0 + tempA3.y * tempB1 + tempA3.z * tempB2 + tempA3.w * tempB3);\n\
    }\n\
    coord_b.y = gidy;\n\
    _viv_asm(CONV, valC, sum0);\n\
    _viv_asm(COPY, outC, valC, 16);\n\
    VXC_WriteImage2DArray(output, coord_b, outC.s0246, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
\n\
    coord_b.y++;\n\
    _viv_asm(CONV, valC, sum1);\n\
    _viv_asm(COPY, outC, valC, 16);\n\
    VXC_WriteImage2DArray(output, coord_b, outC.s0246, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
\n\
    coord_b.y++;\n\
    _viv_asm(CONV, valC, sum2);\n\
    _viv_asm(COPY, outC, valC, 16);\n\
    VXC_WriteImage2DArray(output, coord_b, outC.s0246, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
\n\
    coord_b.y++;\n\
    _viv_asm(CONV, valC, sum3);\n\
    _viv_asm(COPY, outC, valC, 16);\n\
    VXC_WriteImage2DArray(output, coord_b, outC.s0246, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void gemm_block4x4_fp16_fp16_u8(image2d_array_t inputA,\n\
                        image2d_array_t inputB,\n\
                        image2d_array_t output,\n\
                                    int transposeA,\n\
                                    int transposeB,\n\
                                    int adjointA,\n\
                                    int adjointB,\n\
                        uint M, uint K, uint N)\n\
{\n\
    uint gidy = get_global_id(1);\n\
\n\
    half4 valA, valB, valC;\n\
    vxc_short8 srcA, srcB, tmpA, tmpB;\n\
\n\
    int4 coord_a = (int4)(0, gidy, get_global_id(2), 0);\n\
    int4 coord_b = (int4)(get_global_id(0), 0, get_global_id(2), 0);\n\
\n\
    vxc_float4 sum0 = (vxc_float4)(0);\n\
    vxc_float4 sum1 = (vxc_float4)(0);\n\
    vxc_float4 sum2 = (vxc_float4)(0);\n\
    vxc_float4 sum3 = (vxc_float4)(0);\n\
\n\
    for(int i = 0; i < K; i+=4)\n\
    {\n\
        vxc_float4 tempA0, tempA1, tempA2, tempA3;\n\
        vxc_float4 tempB0, tempB1, tempB2, tempB3;\n\
\n\
        coord_a.x = i;\n\
        coord_b.y = i;\n\
        VXC_ReadImage2DArray(srcA, inputA, coord_a, VXC_5BITOFFSET_XY(0, 0),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcB, inputB, coord_b, VXC_5BITOFFSET_XY(0, 0),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        tmpA = srcA.s04152637;\n\
        tmpB = srcB.s04152637;\n\
        _viv_asm(COPY, valA, tmpA, 16);\n\
        _viv_asm(COPY, valB, tmpB, 16);\n\
        _viv_asm(CONV, tempA0, valA);\n\
        _viv_asm(CONV, tempB0, valB);\n\
\n\
        VXC_ReadImage2DArray(srcA, inputA, coord_a, VXC_5BITOFFSET_XY(0, 1),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcB, inputB, coord_b, VXC_5BITOFFSET_XY(0, 1),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        tmpA = srcA.s04152637;\n\
        tmpB = srcB.s04152637;\n\
        _viv_asm(COPY, valA, tmpA, 16);\n\
        _viv_asm(COPY, valB, tmpB, 16);\n\
        _viv_asm(CONV, tempA1, valA);\n\
        _viv_asm(CONV, tempB1, valB);\n\
        VXC_ReadImage2DArray(srcA, inputA, coord_a, VXC_5BITOFFSET_XY(0, 2),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcB, inputB, coord_b, VXC_5BITOFFSET_XY(0, 2),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        tmpA = srcA.s04152637;\n\
        tmpB = srcB.s04152637;\n\
        _viv_asm(COPY, valA, tmpA, 16);\n\
        _viv_asm(COPY, valB, tmpB, 16);\n\
        _viv_asm(CONV, tempA2, valA);\n\
        _viv_asm(CONV, tempB2, valB);\n\
        VXC_ReadImage2DArray(srcA, inputA, coord_a, VXC_5BITOFFSET_XY(0, 3),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcB, inputB, coord_b, VXC_5BITOFFSET_XY(0, 3),\n\
            VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
        tmpA = srcA.s04152637;\n\
        tmpB = srcB.s04152637;\n\
        _viv_asm(COPY, valA, tmpA, 16);\n\
        _viv_asm(COPY, valB, tmpB, 16);\n\
        _viv_asm(CONV, tempA3, valA);\n\
        _viv_asm(CONV, tempB3, valB);\n\
\n\
        sum0 += (tempA0.x * tempB0 + tempA0.y * tempB1 + tempA0.z * tempB2 + tempA0.w * tempB3);\n\
        sum1 += (tempA1.x * tempB0 + tempA1.y * tempB1 + tempA1.z * tempB2 + tempA1.w * tempB3);\n\
        sum2 += (tempA2.x * tempB0 + tempA2.y * tempB1 + tempA2.z * tempB2 + tempA2.w * tempB3);\n\
        sum3 += (tempA3.x * tempB0 + tempA3.y * tempB1 + tempA3.z * tempB2 + tempA3.w * tempB3);\n\
    }\n\
    vxc_int4 tmpOut0, tmpOut1;\n\
    vxc_uchar16 outC;\n\
    coord_b.y = gidy;\n\
    tmpOut0 = convert_int4_rte(sum0 * outputScale + output_ZP);\n\
    tmpOut1 = convert_int4_rte(sum1 * outputScale + output_ZP);\n\
    VXC_DP2x8(outC, tmpOut0, tmpOut1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1),\\\n\
        uniConvertInt32toUint8_2x8);\n\
    VXC_WriteImage2DArray(output, coord_b, outC.s0123, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    coord_b.y++;\n\
    VXC_WriteImage2DArray(output, coord_b, outC.s4567, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
\n\
    coord_b.y++;\n\
    tmpOut0 = convert_int4_rte(sum2 * outputScale + output_ZP);\n\
    tmpOut1 = convert_int4_rte(sum3 * outputScale + output_ZP);\n\
    VXC_DP2x8(outC, tmpOut0, tmpOut1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1),\\\n\
        uniConvertInt32toUint8_2x8);\n\
    VXC_WriteImage2DArray(output, coord_b, outC.s0123, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    coord_b.y++;\n\
    VXC_WriteImage2DArray(output, coord_b, outC.s4567, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
}"; /* end of vsi_nn_kernel_matrixmul_fp16_vx*/

static const char vsi_nn_kernel_matrixmul_transbp1_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
/********************gemm transposeB fp16 uint8 to fp16*************************/\n\
_viv_uniform int input2_ZP;\n\
_viv_uniform float input2Scale;\n\
_viv_uniform VXC_512Bits uniU8SubZptoFp16_dp2x8;\n\
_viv_uniform VXC_512Bits uniFp16MulFp16AddtoFp32_dp8x2;\n\
\n\
__kernel void gemmTransBFp16U8toFp16(image2d_array_t inputA,\n\
                        image2d_array_t inputB,\n\
                        image2d_array_t output,\n\
                                    int transposeA,\n\
                                    int transposeB,\n\
                                    int adjointA,\n\
                                    int adjointB,\n\
                        uint M, uint K, uint N)\n\
{\n\
    int4 coord_out = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    int4 coord_a = (int4)(0, coord_out.y, coord_out.z, 0);\n\
    int4 coord_b = (int4)(0, coord_out.x, coord_out.z, 0);\n\
\n\
    vxc_float4 sum0 = (vxc_float4)(0);\n\
    vxc_float4 sum1 = (vxc_float4)(0);\n\
    vxc_float4 sum2 = (vxc_float4)(0);\n\
    vxc_float4 sum3 = (vxc_float4)(0);\n\
    short zp2 = input2_ZP;\n\
    for(int i = 0; i < K; i+=8)\n\
    {\n\
        coord_a.x = i;\n\
        coord_b.x = i;\n\
        vxc_short8 srcA0,srcA1,srcA2,srcA3;\n\
        vxc_uchar8 srcB0,srcB1,srcB2,srcB3;\n\
        VXC_ReadImage2DArray(srcB0, inputB, coord_b, VXC_5BITOFFSET_XY(0, 0),\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcB1, inputB, coord_b, VXC_5BITOFFSET_XY(0, 1),\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcB2, inputB, coord_b, VXC_5BITOFFSET_XY(0, 2),\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcB3, inputB, coord_b, VXC_5BITOFFSET_XY(0, 3),\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcA0, inputA, coord_a, VXC_5BITOFFSET_XY(0, 0),\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcA1, inputA, coord_a, VXC_5BITOFFSET_XY(0, 1),\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcA2, inputA, coord_a, VXC_5BITOFFSET_XY(0, 2),\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcA3, inputA, coord_a, VXC_5BITOFFSET_XY(0, 3),\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
        vxc_half8 halfB0,halfB1,halfB2,halfB3;\n\
        VXC_DP2x8(halfB0, srcB0, zp2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
            uniU8SubZptoFp16_dp2x8);\n\
        VXC_DP2x8(halfB1, srcB1, zp2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
            uniU8SubZptoFp16_dp2x8);\n\
        VXC_DP2x8(halfB2, srcB2, zp2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
            uniU8SubZptoFp16_dp2x8);\n\
        VXC_DP2x8(halfB3, srcB3, zp2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
            uniU8SubZptoFp16_dp2x8);\n\
        vxc_half8 halfA0,halfA1,halfA2,halfA3;\n\
        _viv_asm(COPY, halfA0, srcA0, 16);\n\
        _viv_asm(COPY, halfA1, srcA1, 16);\n\
        _viv_asm(COPY, halfA2, srcA2, 16);\n\
        _viv_asm(COPY, halfA3, srcA3, 16);\n\
        vxc_float4 fpVal;\n\
        VXC_DP8x2(fpVal, halfA0, halfB0, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        VXC_DP8x2(fpVal, halfA0, halfB1, VXC_MODIFIER(1, 1, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        VXC_DP8x2(fpVal, halfA0, halfB2, VXC_MODIFIER(2, 2, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        VXC_DP8x2(fpVal, halfA0, halfB3, VXC_MODIFIER(3, 3, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        sum0 += fpVal;\n\
        VXC_DP8x2(fpVal, halfA1, halfB0, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        VXC_DP8x2(fpVal, halfA1, halfB1, VXC_MODIFIER(1, 1, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        VXC_DP8x2(fpVal, halfA1, halfB2, VXC_MODIFIER(2, 2, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        VXC_DP8x2(fpVal, halfA1, halfB3, VXC_MODIFIER(3, 3, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        sum1 += fpVal;\n\
        VXC_DP8x2(fpVal, halfA2, halfB0, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        VXC_DP8x2(fpVal, halfA2, halfB1, VXC_MODIFIER(1, 1, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        VXC_DP8x2(fpVal, halfA2, halfB2, VXC_MODIFIER(2, 2, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        VXC_DP8x2(fpVal, halfA2, halfB3, VXC_MODIFIER(3, 3, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        sum2 += fpVal;\n\
        VXC_DP8x2(fpVal, halfA3, halfB0, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        VXC_DP8x2(fpVal, halfA3, halfB1, VXC_MODIFIER(1, 1, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        VXC_DP8x2(fpVal, halfA3, halfB2, VXC_MODIFIER(2, 2, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        VXC_DP8x2(fpVal, halfA3, halfB3, VXC_MODIFIER(3, 3, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        sum3 += fpVal;\n\
    }\n\
    half4 halfDst;\n\
    vxc_short8 valDst;\n\
    sum0 *= input2Scale;\n\
    _viv_asm(CONV, halfDst, sum0);\n\
    _viv_asm(COPY, valDst, halfDst, 16);\n\
    VXC_WriteImage2DArray(output, coord_out, valDst.s0246, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    coord_out.y++;\n\
    sum1 *= input2Scale;\n\
    _viv_asm(CONV, halfDst, sum1);\n\
    _viv_asm(COPY, valDst, halfDst, 16);\n\
    VXC_WriteImage2DArray(output, coord_out, valDst.s0246, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    coord_out.y++;\n\
    sum2 *= input2Scale;\n\
    _viv_asm(CONV, halfDst, sum2);\n\
    _viv_asm(COPY, valDst, halfDst, 16);\n\
    VXC_WriteImage2DArray(output, coord_out, valDst.s0246, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    coord_out.y++;\n\
    sum3 *= input2Scale;\n\
    _viv_asm(CONV, halfDst, sum3);\n\
    _viv_asm(COPY, valDst, halfDst, 16);\n\
    VXC_WriteImage2DArray(output, coord_out, valDst.s0246, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
}\n\
/***********************gemm transposeB fp16 uint8 to uint8***********************************/\n\
_viv_uniform float scaleIn2divOut;\n\
_viv_uniform VXC_512Bits uniConvertInt32toUint8_2x8;\n\
_viv_uniform int output_ZP;\n\
\n\
__kernel void gemmTransBFp16U8toU8(image2d_array_t inputA,\n\
                        image2d_array_t inputB,\n\
                        image2d_array_t output,\n\
                                    int transposeA,\n\
                                    int transposeB,\n\
                                    int adjointA,\n\
                                    int adjointB,\n\
                        uint M, uint K, uint N)\n\
{\n\
    int4 coord_out = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    int4 coord_a = (int4)(0, coord_out.y, coord_out.z, 0);\n\
    int4 coord_b = (int4)(0, coord_out.x, coord_out.z, 0);\n\
\n\
    vxc_float4 sum0 = (vxc_float4)(0);\n\
    vxc_float4 sum1 = (vxc_float4)(0);\n\
    vxc_float4 sum2 = (vxc_float4)(0);\n\
    vxc_float4 sum3 = (vxc_float4)(0);\n\
    short zp2 = input2_ZP;\n\
    for(int i = 0; i < K; i+=8)\n\
    {\n\
        coord_a.x = i;\n\
        coord_b.x = i;\n\
        vxc_short8 srcA0,srcA1,srcA2,srcA3;\n\
        vxc_uchar8 srcB0,srcB1,srcB2,srcB3;\n\
        VXC_ReadImage2DArray(srcB0, inputB, coord_b, VXC_5BITOFFSET_XY(0, 0),\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcB1, inputB, coord_b, VXC_5BITOFFSET_XY(0, 1),\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcB2, inputB, coord_b, VXC_5BITOFFSET_XY(0, 2),\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcB3, inputB, coord_b, VXC_5BITOFFSET_XY(0, 3),\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcA0, inputA, coord_a, VXC_5BITOFFSET_XY(0, 0),\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcA1, inputA, coord_a, VXC_5BITOFFSET_XY(0, 1),\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcA2, inputA, coord_a, VXC_5BITOFFSET_XY(0, 2),\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcA3, inputA, coord_a, VXC_5BITOFFSET_XY(0, 3),\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
        vxc_half8 halfB0,halfB1,halfB2,halfB3;\n\
        VXC_DP2x8(halfB0, srcB0, zp2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
            uniU8SubZptoFp16_dp2x8);\n\
        VXC_DP2x8(halfB1, srcB1, zp2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
            uniU8SubZptoFp16_dp2x8);\n\
        VXC_DP2x8(halfB2, srcB2, zp2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
            uniU8SubZptoFp16_dp2x8);\n\
        VXC_DP2x8(halfB3, srcB3, zp2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
            uniU8SubZptoFp16_dp2x8);\n\
        vxc_half8 halfA0,halfA1,halfA2,halfA3;\n\
        _viv_asm(COPY, halfA0, srcA0, 16);\n\
        _viv_asm(COPY, halfA1, srcA1, 16);\n\
        _viv_asm(COPY, halfA2, srcA2, 16);\n\
        _viv_asm(COPY, halfA3, srcA3, 16);\n\
        vxc_float4 fpVal;\n\
        VXC_DP8x2(fpVal, halfA0, halfB0, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        VXC_DP8x2(fpVal, halfA0, halfB1, VXC_MODIFIER(1, 1, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        VXC_DP8x2(fpVal, halfA0, halfB2, VXC_MODIFIER(2, 2, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        VXC_DP8x2(fpVal, halfA0, halfB3, VXC_MODIFIER(3, 3, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        sum0 += fpVal;\n\
        VXC_DP8x2(fpVal, halfA1, halfB0, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        VXC_DP8x2(fpVal, halfA1, halfB1, VXC_MODIFIER(1, 1, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        VXC_DP8x2(fpVal, halfA1, halfB2, VXC_MODIFIER(2, 2, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        VXC_DP8x2(fpVal, halfA1, halfB3, VXC_MODIFIER(3, 3, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        sum1 += fpVal;\n\
        VXC_DP8x2(fpVal, halfA2, halfB0, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        VXC_DP8x2(fpVal, halfA2, halfB1, VXC_MODIFIER(1, 1, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        VXC_DP8x2(fpVal, halfA2, halfB2, VXC_MODIFIER(2, 2, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        VXC_DP8x2(fpVal, halfA2, halfB3, VXC_MODIFIER(3, 3, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        sum2 += fpVal;\n\
        VXC_DP8x2(fpVal, halfA3, halfB0, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        VXC_DP8x2(fpVal, halfA3, halfB1, VXC_MODIFIER(1, 1, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        VXC_DP8x2(fpVal, halfA3, halfB2, VXC_MODIFIER(2, 2, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        VXC_DP8x2(fpVal, halfA3, halfB3, VXC_MODIFIER(3, 3, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        sum3 += fpVal;\n\
    }\n\
    vxc_int4 tmpOut0, tmpOut1;\n\
    vxc_uchar8 valDst;\n\
    tmpOut0 = convert_int4_rte(sum0 * scaleIn2divOut + output_ZP);\n\
    tmpOut1 = convert_int4_rte(sum1 * scaleIn2divOut + output_ZP);\n\
    VXC_DP2x8(valDst, tmpOut0, tmpOut1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\n\
        uniConvertInt32toUint8_2x8);\n\
    VXC_WriteImage2DArray(output, coord_out, valDst.s0123, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    coord_out.y++;\n\
    VXC_WriteImage2DArray(output, coord_out, valDst.s4567, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    coord_out.y++;\n\
    tmpOut0 = convert_int4_rte(sum2 * scaleIn2divOut + output_ZP);\n\
    tmpOut1 = convert_int4_rte(sum3 * scaleIn2divOut + output_ZP);\n\
    VXC_DP2x8(valDst, tmpOut0, tmpOut1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\n\
        uniConvertInt32toUint8_2x8);\n\
    VXC_WriteImage2DArray(output, coord_out, valDst.s0123, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    coord_out.y++;\n\
    VXC_WriteImage2DArray(output, coord_out, valDst.s4567, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
}"; /* end of vsi_nn_kernel_matrixmul_transbp1_vx*/

static const char vsi_nn_kernel_matrixmul_transbp2_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
/********************gemm transposeB uint8 uint8 to fp16*************************/\n\
_viv_uniform int input1_ZP;\n\
_viv_uniform int input2_ZP;\n\
_viv_uniform float inScaleMul;\n\
_viv_uniform VXC_512Bits uniU8SubZptoFp16_dp2x8;\n\
_viv_uniform VXC_512Bits uniFp16MulFp16AddtoFp32_dp8x2;\n\
\n\
__kernel void gemmTransBU8U8toFp16(image2d_array_t inputA,\n\
                        image2d_array_t inputB,\n\
                        image2d_array_t output,\n\
                                    int transposeA,\n\
                                    int transposeB,\n\
                                    int adjointA,\n\
                                    int adjointB,\n\
                        uint M, uint K, uint N)\n\
{\n\
    int4 coord_out = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    int4 coord_a = (int4)(0, coord_out.y, coord_out.z, 0);\n\
    int4 coord_b = (int4)(0, coord_out.x, coord_out.z, 0);\n\
\n\
    vxc_float4 sum0 = (vxc_float4)(0);\n\
    vxc_float4 sum1 = (vxc_float4)(0);\n\
    vxc_float4 sum2 = (vxc_float4)(0);\n\
    vxc_float4 sum3 = (vxc_float4)(0);\n\
    short zp1 = input1_ZP;\n\
    short zp2 = input2_ZP;\n\
    for(int i = 0; i < K; i+=8)\n\
    {\n\
        coord_a.x = i;\n\
        coord_b.x = i;\n\
        vxc_uchar8 srcA0,srcA1,srcA2,srcA3;\n\
        vxc_uchar8 srcB0,srcB1,srcB2,srcB3;\n\
        VXC_ReadImage2DArray(srcA0, inputA, coord_a, VXC_5BITOFFSET_XY(0, 0),\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcA1, inputA, coord_a, VXC_5BITOFFSET_XY(0, 1),\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcA2, inputA, coord_a, VXC_5BITOFFSET_XY(0, 2),\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcA3, inputA, coord_a, VXC_5BITOFFSET_XY(0, 3),\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcB0, inputB, coord_b, VXC_5BITOFFSET_XY(0, 0),\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcB1, inputB, coord_b, VXC_5BITOFFSET_XY(0, 1),\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcB2, inputB, coord_b, VXC_5BITOFFSET_XY(0, 2),\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage2DArray(srcB3, inputB, coord_b, VXC_5BITOFFSET_XY(0, 3),\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
        vxc_half8 halfA0,halfA1,halfA2,halfA3;\n\
        VXC_DP2x8(halfA0, srcA0, zp1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\n\
            uniU8SubZptoFp16_dp2x8);\n\
        VXC_DP2x8(halfA1, srcA1, zp1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\n\
            uniU8SubZptoFp16_dp2x8);\n\
        VXC_DP2x8(halfA2, srcA2, zp1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\n\
            uniU8SubZptoFp16_dp2x8);\n\
        VXC_DP2x8(halfA3, srcA3, zp1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\n\
            uniU8SubZptoFp16_dp2x8);\n\
        vxc_half8 halfB0,halfB1,halfB2,halfB3;\n\
        VXC_DP2x8(halfB0, srcB0, zp2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\n\
            uniU8SubZptoFp16_dp2x8);\n\
        VXC_DP2x8(halfB1, srcB1, zp2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\n\
            uniU8SubZptoFp16_dp2x8);\n\
        VXC_DP2x8(halfB2, srcB2, zp2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\n\
            uniU8SubZptoFp16_dp2x8);\n\
        VXC_DP2x8(halfB3, srcB3, zp2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\n\
            uniU8SubZptoFp16_dp2x8);\n\
        vxc_float4 fpVal;\n\
        VXC_DP8x2(fpVal, halfA0, halfB0, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        VXC_DP8x2(fpVal, halfA0, halfB1, VXC_MODIFIER(1, 1, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        VXC_DP8x2(fpVal, halfA0, halfB2, VXC_MODIFIER(2, 2, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        VXC_DP8x2(fpVal, halfA0, halfB3, VXC_MODIFIER(3, 3, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        sum0 += fpVal;\n\
        VXC_DP8x2(fpVal, halfA1, halfB0, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        VXC_DP8x2(fpVal, halfA1, halfB1, VXC_MODIFIER(1, 1, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        VXC_DP8x2(fpVal, halfA1, halfB2, VXC_MODIFIER(2, 2, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        VXC_DP8x2(fpVal, halfA1, halfB3, VXC_MODIFIER(3, 3, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        sum1 += fpVal;\n\
        VXC_DP8x2(fpVal, halfA2, halfB0, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        VXC_DP8x2(fpVal, halfA2, halfB1, VXC_MODIFIER(1, 1, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        VXC_DP8x2(fpVal, halfA2, halfB2, VXC_MODIFIER(2, 2, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        VXC_DP8x2(fpVal, halfA2, halfB3, VXC_MODIFIER(3, 3, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        sum2 += fpVal;\n\
        VXC_DP8x2(fpVal, halfA3, halfB0, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        VXC_DP8x2(fpVal, halfA3, halfB1, VXC_MODIFIER(1, 1, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        VXC_DP8x2(fpVal, halfA3, halfB2, VXC_MODIFIER(2, 2, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        VXC_DP8x2(fpVal, halfA3, halfB3, VXC_MODIFIER(3, 3, 0, VXC_RM_TowardZero, 0),\n\
            uniFp16MulFp16AddtoFp32_dp8x2);\n\
        sum3 += fpVal;\n\
    }\n\
    half4 halfDst;\n\
    vxc_short8 valDst;\n\
    sum0 *= inScaleMul;\n\
    _viv_asm(CONV, halfDst, sum0);\n\
    _viv_asm(COPY, valDst, halfDst, 16);\n\
    VXC_WriteImage2DArray(output, coord_out, valDst.s0246, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    coord_out.y++;\n\
    sum1 *= inScaleMul;\n\
    _viv_asm(CONV, halfDst, sum1);\n\
    _viv_asm(COPY, valDst, halfDst, 16);\n\
    VXC_WriteImage2DArray(output, coord_out, valDst.s0246, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    coord_out.y++;\n\
    sum2 *= inScaleMul;\n\
    _viv_asm(CONV, halfDst, sum2);\n\
    _viv_asm(COPY, valDst, halfDst, 16);\n\
    VXC_WriteImage2DArray(output, coord_out, valDst.s0246, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    coord_out.y++;\n\
    sum3 *= inScaleMul;\n\
    _viv_asm(CONV, halfDst, sum3);\n\
    _viv_asm(COPY, valDst, halfDst, 16);\n\
    VXC_WriteImage2DArray(output, coord_out, valDst.s0246, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
}"; /* end of vsi_nn_kernel_matrixmul_transbp2_vx*/

static const char vsi_nn_kernel_maximum_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
__kernel void vxcTensorMaximum_F16F16toF16\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_short8 vec0, vec1, dst;\n\
    vxc_half8  src0, src1;\n\
    VXC_ReadImage2DArray(vec0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, src0, vec0, 16);\n\
    VXC_ReadImage2DArray(vec1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, src1, vec1, 16);\n\
\n\
    VXC_VertMax3_Half(src0, src0, src1, src1, VXC_MODIFIER_CLAMP(0, 7, 0, 0));\n\
    _viv_asm(COPY, dst, src0, 16);\n\
\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcTensorMaximum_F16F16toF16_2D\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(1), get_global_id(1));\n\
\n\
    vxc_short8 vec0, vec1, dst;\n\
    vxc_half8  src0, src1;\n\
    VXC_ReadImage(vec0, input0, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, src0, vec0, 16);\n\
    VXC_ReadImage(vec1, input1, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, src1, vec1, 16);\n\
\n\
    coord.z ++;\n\
\n\
    VXC_VertMax3_Half(src0, src0, src1, src1, VXC_MODIFIER_CLAMP(0, 7, 0, 0));\n\
    _viv_asm(COPY, dst, src0, 16);\n\
\n\
    VXC_WriteImage(output, coord.xy, dst, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform VXC_512Bits uniConvertI8toI8_0_part0_2x8;\n\
_viv_uniform VXC_512Bits uniConvertI8toI8_0_part1_2x8;\n\
_viv_uniform VXC_512Bits uniConvertI8toI8_1_part0_2x8;\n\
_viv_uniform VXC_512Bits uniConvertI8toI8_1_part1_2x8;\n\
__kernel void vxcTensorMaximum_I8I8toI8\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_char16 src0, src1, dst;\n\
    VXC_ReadImage2DArray(src0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_DP2x8(src0, src0, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI8toI8_0_part0_2x8);\n\
    VXC_DP2x8(src0, src0, src0, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniConvertI8toI8_0_part1_2x8);\n\
    VXC_DP2x8(src1, src1, src1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI8toI8_1_part0_2x8);\n\
    VXC_DP2x8(src1, src1, src1, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniConvertI8toI8_1_part1_2x8);\n\
    dst = max(src0, src1);\n\
\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 15, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcTensorMaximum_I8I8toI8_2D\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(1), get_global_id(1));\n\
\n\
    vxc_char16 src0, src1, dst;\n\
    VXC_ReadImage(src0, input0, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(src1, input1, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    coord.z ++;\n\
\n\
    VXC_DP2x8(src0, src0, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI8toI8_0_part0_2x8);\n\
    VXC_DP2x8(src0, src0, src0, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniConvertI8toI8_0_part1_2x8);\n\
    VXC_DP2x8(src1, src1, src1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI8toI8_1_part0_2x8);\n\
    VXC_DP2x8(src1, src1, src1, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniConvertI8toI8_1_part1_2x8);\n\
    dst = max(src0, src1);\n\
\n\
    VXC_WriteImage(output, coord.xy, dst, VXC_MODIFIER(0, 15, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform VXC_512Bits uniU8MulAndPostShift0_Lo_2x8;\n\
_viv_uniform VXC_512Bits uniU8MulAndPostShift0_Hi_2x8;\n\
_viv_uniform VXC_512Bits uniU8MulAndPostShift1_Lo_2x8;\n\
_viv_uniform VXC_512Bits uniU8MulAndPostShift1_Hi_2x8;\n\
_viv_uniform int2 multAndoutZP0;//[0:15] multiplier, [31:63] output zp\n\
_viv_uniform int2 multAndoutZP1;//[0:15] multiplier, [31:63] output zp\n\
__kernel void vxcTensorMaximum_U8U8toU8\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_uchar16 src0, src1, dst;\n\
    VXC_ReadImage2DArray(src0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    vxc_ushort8 mp0, mp1;\n\
    _viv_asm(COPY, mp0, multAndoutZP0, 16);\n\
    _viv_asm(COPY, mp1, multAndoutZP1, 16);\n\
    VXC_DP2x8(src0, src0, mp0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniU8MulAndPostShift0_Lo_2x8);\n\
    VXC_DP2x8(src0, src0, mp0, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniU8MulAndPostShift0_Hi_2x8);\n\
    VXC_DP2x8(src1, src1, mp1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniU8MulAndPostShift1_Lo_2x8);\n\
    VXC_DP2x8(src1, src1, mp1, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniU8MulAndPostShift1_Hi_2x8);\n\
    dst = max(src0, src1);\n\
\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 15, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcTensorMaximum_U8U8toU8_2D\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));\n\
\n\
    vxc_uchar16 src0, src1, dst;\n\
    VXC_ReadImage(src0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(src1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    vxc_ushort8 mp0, mp1;\n\
    _viv_asm(COPY, mp0, multAndoutZP0, 16);\n\
    _viv_asm(COPY, mp1, multAndoutZP1, 16);\n\
    VXC_DP2x8(src0, src0, mp0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniU8MulAndPostShift0_Lo_2x8);\n\
    VXC_DP2x8(src0, src0, mp0, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniU8MulAndPostShift0_Hi_2x8);\n\
    VXC_DP2x8(src1, src1, mp1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniU8MulAndPostShift1_Lo_2x8);\n\
    VXC_DP2x8(src1, src1, mp1, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniU8MulAndPostShift1_Hi_2x8);\n\
    dst = max(src0, src1);\n\
\n\
    VXC_WriteImage(output, coord, dst, VXC_MODIFIER(0, 15, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform VXC_512Bits uniConvertI16toI16_0_2x8;\n\
_viv_uniform VXC_512Bits uniConvertI16toI16_1_2x8;\n\
__kernel void vxcTensorMaximum_I16I16toI16\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_short8 src0, src1, dst;\n\
    VXC_ReadImage2DArray(src0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_DP2x8(src0, src0, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI16toI16_0_2x8);\n\
    VXC_DP2x8(src1, src1, src1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI16toI16_1_2x8);\n\
    dst = max(src0, src1);\n\
\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcTensorMaximum_I16I16toI16_2D\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(1), get_global_id(1));\n\
\n\
    vxc_short8 src0, src1, dst;\n\
    VXC_ReadImage(src0, input0, coord.xy, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(src1, input1, coord.xy, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    coord.z ++;\n\
\n\
    VXC_DP2x8(src0, src0, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI16toI16_0_2x8);\n\
    VXC_DP2x8(src1, src1, src1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI16toI16_1_2x8);\n\
    dst = max(src0, src1);\n\
\n\
    VXC_WriteImage(output, coord.xy, dst, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_maximum_vx*/

static const char vsi_nn_kernel_maximum_fp16_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform VXC_512Bits uniConvertI8toI8_0_part0_2x8;\n\
_viv_uniform VXC_512Bits uniConvertI8toI8_0_part1_2x8;\n\
_viv_uniform VXC_512Bits uinConvertFp16ToInt8_2x8;\n\
_viv_uniform VXC_512Bits uniConvertInt8toFp16_2x8;\n\
\n\
__kernel void vxcTensorMaximum_I8F16toI8\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_char16 src0, src2, dst;\n\
    vxc_short8 src1, src3, src4, src5;\n\
    vxc_half8 data0, data1, data2, data3;\n\
    vxc_char16 tmp0, tmp1;\n\
    VXC_ReadImage2DArray(src0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src4, input1, coord, VXC_5BITOFFSET_XY(8, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    _viv_asm(COPY, data0, src1, 16);\n\
    _viv_asm(COPY, data1, src4, 16);\n\
\n\
    VXC_DP2x8(src0, src0, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI8toI8_0_part0_2x8);\n\
    VXC_DP2x8(src0, src0, src0, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniConvertI8toI8_0_part1_2x8);\n\
    VXC_DP2x8(tmp0, data0, data0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uinConvertFp16ToInt8_2x8);\n\
    VXC_DP2x8(tmp0, data1, data1, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uinConvertFp16ToInt8_2x8);\n\
    dst = max(src0, tmp0);\n\
\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 15, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcTensorMaximum_I8F16toI8_2D\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(1), get_global_id(1));\n\
\n\
    vxc_char16 src0, src2, dst;\n\
    vxc_short8 src1, src3, src4, src5;\n\
    vxc_half8 data0, data1, data2, data3;\n\
    vxc_char16 tmp0;\n\
\n\
    VXC_ReadImage(src0, input0, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(src1, input1, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(src4, input1, coord.xy, VXC_5BITOFFSET_XY(8, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    _viv_asm(COPY, data0, src1, 16);\n\
    _viv_asm(COPY, data1, src4, 16);\n\
\n\
    VXC_DP2x8(src0, src0, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI8toI8_0_part0_2x8);\n\
    VXC_DP2x8(src0, src0, src0, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniConvertI8toI8_0_part1_2x8);\n\
    VXC_DP2x8(tmp0, data0, data0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uinConvertFp16ToInt8_2x8);\n\
    VXC_DP2x8(tmp0, data1, data1, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uinConvertFp16ToInt8_2x8);\n\
    dst = max(src0, tmp0);\n\
\n\
    VXC_WriteImage(output, coord.xy, dst, VXC_MODIFIER(0, 15, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcTensorMaximum_I8F16toF16\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_char8 vec0, vec2;\n\
    vxc_short8 vec1, vec3, dst;\n\
    vxc_half8  src0, src1, src2, src3;\n\
\n\
    VXC_ReadImage2DArray(vec0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(vec1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, src1, vec1, 16);\n\
\n\
    VXC_DP2x8(src0, vec0, vec0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertInt8toFp16_2x8);\n\
\n\
    VXC_VertMax3_Half(src0, src0, src1, src1, VXC_MODIFIER_CLAMP(0, 7, 0, 0));\n\
    _viv_asm(COPY, dst, src0, 16);\n\
\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcTensorMaximum_I8F16toF16_2D\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(1), get_global_id(1));\n\
\n\
    vxc_char8 vec0, vec2;\n\
    vxc_short8 vec1, vec3, dst;\n\
    vxc_half8  src0, src1, src2, src3;\n\
    VXC_ReadImage(vec0, input0, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(vec1, input1, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, src1, vec1, 16);\n\
\n\
    VXC_DP2x8(src0, vec0, vec0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertInt8toFp16_2x8);\n\
\n\
    VXC_VertMax3_Half(src0, src0, src1, src1, VXC_MODIFIER_CLAMP(0, 7, 0, 0));\n\
    _viv_asm(COPY, dst, src0, 16);\n\
\n\
    VXC_WriteImage(output, coord.xy, dst, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform int2 multAndoutZP0;//[0:15] multiplier, [31:63] output zp\n\
_viv_uniform VXC_512Bits uniU8MulAndPostShift_0_Lo_2x8;\n\
\n\
__kernel void vxcTensorMaximum_U8F16toF16\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_uchar8 vec0, vec2;\n\
    vxc_short8 vec1, vec3, dst;\n\
    vxc_half8  src0, src1, src2, src3;\n\
\n\
    VXC_ReadImage2DArray(vec0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(vec1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, src1, vec1, 16);\n\
\n\
    vxc_ushort8 ms0;\n\
    _viv_asm(COPY, ms0, multAndoutZP0, 16);\n\
    VXC_DP2x8(src0, vec0, ms0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1),\\\n\
                uniU8MulAndPostShift_0_Lo_2x8);\n\
\n\
    VXC_VertMax3_Half(src0, src0, src1, src1, VXC_MODIFIER_CLAMP(0, 7, 0, 0));\n\
    _viv_asm(COPY, dst, src0, 16);\n\
\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcTensorMaximum_U8F16toF16_2D\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(1), get_global_id(1));\n\
\n\
    vxc_uchar8 vec0, vec2;\n\
    vxc_short8 vec1, vec3, dst;\n\
    vxc_half8  src0, src1, src2, src3;\n\
\n\
    VXC_ReadImage(vec0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(vec1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, src1, vec1, 16);\n\
\n\
    vxc_ushort8 ms0;\n\
    _viv_asm(COPY, ms0, multAndoutZP0, 16);\n\
    VXC_DP2x8(src0, vec0, ms0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1),\\\n\
                uniU8MulAndPostShift_0_Lo_2x8);\n\
\n\
    VXC_VertMax3_Half(src0, src0, src1, src1, VXC_MODIFIER_CLAMP(0, 7, 0, 0));\n\
    _viv_asm(COPY, dst, src0, 16);\n\
\n\
    VXC_WriteImage(output, coord, dst, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform VXC_512Bits uniU8MulAndPostShift0_Lo_2x8;\n\
_viv_uniform VXC_512Bits uniU8MulAndPostShift0_Hi_2x8;\n\
_viv_uniform VXC_512Bits uniConvertFp16toU8_2x8;\n\
_viv_uniform int2 multAndoutZP1;//[0:15] multiplier, [31:63] output zp\n\
__kernel void vxcTensorMaximum_U8F16toU8\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_uchar16 src0, dst0, dst1;\n\
    vxc_ushort8 src1, src2;\n\
    vxc_half8 data1, data2;\n\
    VXC_ReadImage2DArray(src0, input0, coord, 0, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input1, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src2, input1, coord, VXC_5BITOFFSET_XY(8, 0), \\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, data1, src1, 16);\n\
    _viv_asm(COPY, data2, src2, 16);\n\
\n\
    vxc_ushort8 mp0, mp1;\n\
    _viv_asm(COPY, mp0, multAndoutZP0, 16);\n\
    _viv_asm(COPY, mp1, multAndoutZP1, 16);\n\
    VXC_DP2x8(dst0, src0, mp0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniU8MulAndPostShift0_Lo_2x8);\n\
    VXC_DP2x8(dst0, src0, mp0, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniU8MulAndPostShift0_Hi_2x8);\n\
    VXC_DP2x8(dst1, data1, mp1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniConvertFp16toU8_2x8);\n\
    VXC_DP2x8(dst1, data2, mp1, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniConvertFp16toU8_2x8);\n\
    dst0 = max(dst0, dst1);\n\
\n\
    VXC_WriteImage2DArray(output, coord, dst0, VXC_MODIFIER(0, 15, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcTensorMaximum_U8F16toU8_2D\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int2 coord = (int2)(get_global_id(0), get_global_id(1));\n\
\n\
    vxc_uchar16 src0, dst0, dst1;\n\
    vxc_ushort8 src1, src2;\n\
    vxc_half8 data1, data2;\n\
    VXC_ReadImage(src0, input0, coord, 0, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(src1, input1, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(src2, input1, coord, VXC_5BITOFFSET_XY(8, 0), \\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, data1, src1, 16);\n\
    _viv_asm(COPY, data2, src2, 16);\n\
\n\
    vxc_ushort8 mp0, mp1;\n\
    _viv_asm(COPY, mp0, multAndoutZP0, 16);\n\
    _viv_asm(COPY, mp1, multAndoutZP1, 16);\n\
    VXC_DP2x8(dst0, src0, mp0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniU8MulAndPostShift0_Lo_2x8);\n\
    VXC_DP2x8(dst0, src0, mp0, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniU8MulAndPostShift0_Hi_2x8);\n\
    VXC_DP2x8(dst1, data1, mp1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniConvertFp16toU8_2x8);\n\
    VXC_DP2x8(dst1, data2, mp1, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniConvertFp16toU8_2x8);\n\
    dst0 = max(dst0, dst1);\n\
\n\
    VXC_WriteImage(output, coord, dst0, VXC_MODIFIER(0, 15, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcTensorMaximum_F16F16toU8\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_ushort8 src0, src1;\n\
    vxc_half8 data0, data1;\n\
    vxc_uchar16 dst0, dst1;\n\
    VXC_ReadImage2DArray(src0, input0, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input1, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, data0, src0, 16);\n\
    _viv_asm(COPY, data1, src1, 16);\n\
\n\
    vxc_ushort8 mp1;\n\
    _viv_asm(COPY, mp1, multAndoutZP1, 16);\n\
    VXC_DP2x8(dst0, data0, mp1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniConvertFp16toU8_2x8);\n\
    VXC_DP2x8(dst1, data1, mp1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniConvertFp16toU8_2x8);\n\
    dst0 = max(dst0, dst1);\n\
\n\
    VXC_WriteImage2DArray(output, coord, dst0, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcTensorMaximum_F16F16toU8_2D\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int2 coord = (int2)(get_global_id(0), get_global_id(1));\n\
\n\
    vxc_ushort8 src0, src1;\n\
    vxc_half8 data0, data1;\n\
    vxc_uchar16 dst0, dst1;\n\
    VXC_ReadImage(src0, input0, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(src1, input1, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, data0, src0, 16);\n\
    _viv_asm(COPY, data1, src1, 16);\n\
\n\
    vxc_ushort8 mp1;\n\
    _viv_asm(COPY, mp1, multAndoutZP1, 16);\n\
    VXC_DP2x8(dst0, data0, mp1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniConvertFp16toU8_2x8);\n\
    VXC_DP2x8(dst1, data1, mp1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniConvertFp16toU8_2x8);\n\
    dst0 = max(dst0, dst1);\n\
\n\
    VXC_WriteImage(output, coord, dst0, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_maximum_fp16_vx*/

static const char vsi_nn_kernel_maximum_i16_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform VXC_512Bits uniConvertI16toI16_2x8;\n\
_viv_uniform VXC_512Bits uinConvertFp16ToInt16_2x8;\n\
_viv_uniform VXC_512Bits uniConvertInt16toFp16_2x8;\n\
\n\
__kernel void vxcTensorMaximum_I16F16toI16\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_short8 src0, src1, tmp0, dst;\n\
    vxc_half8 data0;\n\
    VXC_ReadImage2DArray(src0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    _viv_asm(COPY, data0, src1, 16);\n\
\n\
    VXC_DP2x8(src0, src0, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI16toI16_2x8);\n\
    VXC_DP2x8(tmp0, data0, data0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uinConvertFp16ToInt16_2x8);\n\
    dst = max(src0, tmp0);\n\
\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcTensorMaximum_I16F16toI16_2D\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(1), get_global_id(1));\n\
\n\
    vxc_short8 src0, src1, tmp0, dst;\n\
    vxc_half8 data0;\n\
\n\
    VXC_ReadImage(src0, input0, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(src1, input1, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    _viv_asm(COPY, data0, src1, 16);\n\
\n\
    VXC_DP2x8(src0, src0, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI16toI16_2x8);\n\
    VXC_DP2x8(tmp0, data0, data0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uinConvertFp16ToInt16_2x8);\n\
    dst = max(src0, tmp0);\n\
\n\
    VXC_WriteImage(output, coord.xy, dst, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcTensorMaximum_I16F16toF16\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_short8 vec0, vec1, dst;\n\
    vxc_half8  src0, src1;\n\
\n\
    VXC_ReadImage2DArray(vec0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(vec1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, src1, vec1, 16);\n\
\n\
    VXC_DP2x8(src0, vec0, vec0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertInt16toFp16_2x8);\n\
\n\
    VXC_VertMax3_Half(src0, src0, src1, src1, VXC_MODIFIER_CLAMP(0, 7, 0, 0));\n\
    _viv_asm(COPY, dst, src0, 16);\n\
\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcTensorMaximum_I16F16toF16_2D\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(1), get_global_id(1));\n\
\n\
    vxc_short8 vec0, vec1, dst;\n\
    vxc_half8  src0, src1;\n\
    VXC_ReadImage(vec0, input0, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(vec1, input1, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, src1, vec1, 16);\n\
\n\
    VXC_DP2x8(src0, vec0, vec0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertInt16toFp16_2x8);\n\
\n\
    VXC_VertMax3_Half(src0, src0, src1, src1, VXC_MODIFIER_CLAMP(0, 7, 0, 0));\n\
    _viv_asm(COPY, dst, src0, 16);\n\
\n\
    VXC_WriteImage(output, coord.xy, dst, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_maximum_i16_vx*/

static const char vsi_nn_kernel_minimum_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
__kernel void vxcTensorMinimum_F16F16toF16\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_short8 vec0, vec1, dst;\n\
    vxc_half8  src0, src1;\n\
    VXC_ReadImage2DArray(vec0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, src0, vec0, 16);\n\
    VXC_ReadImage2DArray(vec1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, src1, vec1, 16);\n\
\n\
    VXC_VertMin3_Half(src0, src0, src1, src1, VXC_MODIFIER_CLAMP(0, 7, 0, 0));\n\
    _viv_asm(COPY, dst, src0, 16);\n\
\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcTensorMinimum_F16F16toF16_2D\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(1), get_global_id(1));\n\
\n\
    vxc_short8 vec0, vec1, dst;\n\
    vxc_half8  src0, src1;\n\
    VXC_ReadImage(vec0, input0, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, src0, vec0, 16);\n\
    VXC_ReadImage(vec1, input1, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, src1, vec1, 16);\n\
\n\
    coord.z ++;\n\
\n\
    VXC_VertMin3_Half(src0, src0, src1, src1, VXC_MODIFIER_CLAMP(0, 7, 0, 0));\n\
    _viv_asm(COPY, dst, src0, 16);\n\
\n\
    VXC_WriteImage(output, coord.xy, dst, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform VXC_512Bits uniConvertI8toI8_0_part0_2x8;\n\
_viv_uniform VXC_512Bits uniConvertI8toI8_0_part1_2x8;\n\
_viv_uniform VXC_512Bits uniConvertI8toI8_1_part0_2x8;\n\
_viv_uniform VXC_512Bits uniConvertI8toI8_1_part1_2x8;\n\
__kernel void vxcTensorMinimum_I8I8toI8\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_char16 src0, src1, dst;\n\
    VXC_ReadImage2DArray(src0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_DP2x8(src0, src0, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI8toI8_0_part0_2x8);\n\
    VXC_DP2x8(src0, src0, src0, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniConvertI8toI8_0_part1_2x8);\n\
    VXC_DP2x8(src1, src1, src1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI8toI8_1_part0_2x8);\n\
    VXC_DP2x8(src1, src1, src1, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniConvertI8toI8_1_part1_2x8);\n\
    dst = min(src0, src1);\n\
\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 15, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcTensorMinimum_I8I8toI8_2D\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(1), get_global_id(1));\n\
\n\
    vxc_char16 src0, src1, dst;\n\
    VXC_ReadImage(src0, input0, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(src1, input1, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    coord.z ++;\n\
\n\
    VXC_DP2x8(src0, src0, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI8toI8_0_part0_2x8);\n\
    VXC_DP2x8(src0, src0, src0, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniConvertI8toI8_0_part1_2x8);\n\
    VXC_DP2x8(src1, src1, src1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI8toI8_1_part0_2x8);\n\
    VXC_DP2x8(src1, src1, src1, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniConvertI8toI8_1_part1_2x8);\n\
    dst = min(src0, src1);\n\
\n\
    VXC_WriteImage(output, coord.xy, dst, VXC_MODIFIER(0, 15, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform VXC_512Bits uniU8MulAndPostShift0_Lo_2x8;\n\
_viv_uniform VXC_512Bits uniU8MulAndPostShift0_Hi_2x8;\n\
_viv_uniform VXC_512Bits uniU8MulAndPostShift1_Lo_2x8;\n\
_viv_uniform VXC_512Bits uniU8MulAndPostShift1_Hi_2x8;\n\
_viv_uniform int2 multAndoutZP0;//[0:15] multiplier, [31:63] output zp\n\
_viv_uniform int2 multAndoutZP1;//[0:15] multiplier, [31:63] output zp\n\
__kernel void vxcTensorMinimum_U8U8toU8\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_uchar16 src0, src1, dst;\n\
    VXC_ReadImage2DArray(src0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    vxc_ushort8 mp0, mp1;\n\
    _viv_asm(COPY, mp0, multAndoutZP0, 16);\n\
    _viv_asm(COPY, mp1, multAndoutZP1, 16);\n\
    VXC_DP2x8(src0, src0, mp0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniU8MulAndPostShift0_Lo_2x8);\n\
    VXC_DP2x8(src0, src0, mp0, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniU8MulAndPostShift0_Hi_2x8);\n\
    VXC_DP2x8(src1, src1, mp1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniU8MulAndPostShift1_Lo_2x8);\n\
    VXC_DP2x8(src1, src1, mp1, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniU8MulAndPostShift1_Hi_2x8);\n\
    dst = min(src0, src1);\n\
\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 15, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcTensorMinimum_U8U8toU8_2D\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));\n\
\n\
    vxc_uchar16 src0, src1, dst;\n\
    VXC_ReadImage(src0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(src1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    vxc_ushort8 mp0, mp1;\n\
    _viv_asm(COPY, mp0, multAndoutZP0, 16);\n\
    _viv_asm(COPY, mp1, multAndoutZP1, 16);\n\
    VXC_DP2x8(src0, src0, mp0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniU8MulAndPostShift0_Lo_2x8);\n\
    VXC_DP2x8(src0, src0, mp0, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniU8MulAndPostShift0_Hi_2x8);\n\
    VXC_DP2x8(src1, src1, mp1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniU8MulAndPostShift1_Lo_2x8);\n\
    VXC_DP2x8(src1, src1, mp1, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniU8MulAndPostShift1_Hi_2x8);\n\
    dst = min(src0, src1);\n\
\n\
    VXC_WriteImage(output, coord, dst, VXC_MODIFIER(0, 15, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform VXC_512Bits uniConvertI16toI16_0_2x8;\n\
_viv_uniform VXC_512Bits uniConvertI16toI16_1_2x8;\n\
__kernel void vxcTensorMinimum_I16I16toI16\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_short8 src0, src1, dst;\n\
    VXC_ReadImage2DArray(src0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_DP2x8(src0, src0, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI16toI16_0_2x8);\n\
    VXC_DP2x8(src1, src1, src1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI16toI16_1_2x8);\n\
    dst = min(src0, src1);\n\
\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcTensorMinimum_I16I16toI16_2D\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(1), get_global_id(1));\n\
\n\
    vxc_short8 src0, src1, dst;\n\
    VXC_ReadImage(src0, input0, coord.xy, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(src1, input1, coord.xy, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    coord.z ++;\n\
\n\
    VXC_DP2x8(src0, src0, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI16toI16_0_2x8);\n\
    VXC_DP2x8(src1, src1, src1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI16toI16_1_2x8);\n\
    dst = min(src0, src1);\n\
\n\
    VXC_WriteImage(output, coord.xy, dst, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_minimum_vx*/

static const char vsi_nn_kernel_minimum_fp16_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform VXC_512Bits uniConvertI8toI8_0_part0_2x8;\n\
_viv_uniform VXC_512Bits uniConvertI8toI8_0_part1_2x8;\n\
_viv_uniform VXC_512Bits uinConvertFp16ToInt8_2x8;\n\
_viv_uniform VXC_512Bits uniConvertInt8toFp16_2x8;\n\
\n\
__kernel void vxcTensorMinimum_I8F16toI8\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_char16 src0, src2, dst;\n\
    vxc_short8 src1, src3, src4, src5;\n\
    vxc_half8 data0, data1, data2, data3;\n\
    vxc_char16 tmp0, tmp1;\n\
    VXC_ReadImage2DArray(src0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src4, input1, coord, VXC_5BITOFFSET_XY(8, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    _viv_asm(COPY, data0, src1, 16);\n\
    _viv_asm(COPY, data1, src4, 16);\n\
\n\
    VXC_DP2x8(src0, src0, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI8toI8_0_part0_2x8);\n\
    VXC_DP2x8(src0, src0, src0, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniConvertI8toI8_0_part1_2x8);\n\
    VXC_DP2x8(tmp0, data0, data0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uinConvertFp16ToInt8_2x8);\n\
    VXC_DP2x8(tmp0, data1, data1, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uinConvertFp16ToInt8_2x8);\n\
    dst = min(src0, tmp0);\n\
\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 15, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcTensorMinimum_I8F16toI8_2D\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(1), get_global_id(1));\n\
\n\
    vxc_char16 src0, src2, dst;\n\
    vxc_short8 src1, src3, src4, src5;\n\
    vxc_half8 data0, data1, data2, data3;\n\
    vxc_char16 tmp0;\n\
\n\
    VXC_ReadImage(src0, input0, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(src1, input1, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(src4, input1, coord.xy, VXC_5BITOFFSET_XY(8, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    _viv_asm(COPY, data0, src1, 16);\n\
    _viv_asm(COPY, data1, src4, 16);\n\
\n\
    VXC_DP2x8(src0, src0, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI8toI8_0_part0_2x8);\n\
    VXC_DP2x8(src0, src0, src0, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniConvertI8toI8_0_part1_2x8);\n\
    VXC_DP2x8(tmp0, data0, data0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uinConvertFp16ToInt8_2x8);\n\
    VXC_DP2x8(tmp0, data1, data1, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uinConvertFp16ToInt8_2x8);\n\
    dst = min(src0, tmp0);\n\
\n\
    VXC_WriteImage(output, coord.xy, dst, VXC_MODIFIER(0, 15, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcTensorMinimum_I8F16toF16\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_char8 vec0, vec2;\n\
    vxc_short8 vec1, vec3, dst;\n\
    vxc_half8  src0, src1, src2, src3;\n\
\n\
    VXC_ReadImage2DArray(vec0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(vec1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, src1, vec1, 16);\n\
\n\
    VXC_DP2x8(src0, vec0, vec0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertInt8toFp16_2x8);\n\
\n\
    VXC_VertMin3_Half(src0, src0, src1, src1, VXC_MODIFIER_CLAMP(0, 7, 0, 0));\n\
    _viv_asm(COPY, dst, src0, 16);\n\
\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcTensorMinimum_I8F16toF16_2D\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(1), get_global_id(1));\n\
\n\
    vxc_char8 vec0, vec2;\n\
    vxc_short8 vec1, vec3, dst;\n\
    vxc_half8  src0, src1, src2, src3;\n\
    VXC_ReadImage(vec0, input0, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(vec1, input1, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, src1, vec1, 16);\n\
\n\
    VXC_DP2x8(src0, vec0, vec0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertInt8toFp16_2x8);\n\
\n\
    VXC_VertMin3_Half(src0, src0, src1, src1, VXC_MODIFIER_CLAMP(0, 7, 0, 0));\n\
    _viv_asm(COPY, dst, src0, 16);\n\
\n\
    VXC_WriteImage(output, coord.xy, dst, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform int2 multAndoutZP0;//[0:15] multiplier, [31:63] output zp\n\
_viv_uniform VXC_512Bits uniU8MulAndPostShift_0_Lo_2x8;\n\
\n\
__kernel void vxcTensorMinimum_U8F16toF16\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_uchar8 vec0, vec2;\n\
    vxc_short8 vec1, vec3, dst;\n\
    vxc_half8  src0, src1, src2, src3;\n\
\n\
    VXC_ReadImage2DArray(vec0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(vec1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, src1, vec1, 16);\n\
\n\
    vxc_ushort8 ms0;\n\
    _viv_asm(COPY, ms0, multAndoutZP0, 16);\n\
    VXC_DP2x8(src0, vec0, ms0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1),\\\n\
                uniU8MulAndPostShift_0_Lo_2x8);\n\
\n\
    VXC_VertMin3_Half(src0, src0, src1, src1, VXC_MODIFIER_CLAMP(0, 7, 0, 0));\n\
    _viv_asm(COPY, dst, src0, 16);\n\
\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcTensorMinimum_U8F16toF16_2D\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(1), get_global_id(1));\n\
\n\
    vxc_uchar8 vec0, vec2;\n\
    vxc_short8 vec1, vec3, dst;\n\
    vxc_half8  src0, src1, src2, src3;\n\
\n\
    VXC_ReadImage(vec0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(vec1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, src1, vec1, 16);\n\
\n\
    vxc_ushort8 ms0;\n\
    _viv_asm(COPY, ms0, multAndoutZP0, 16);\n\
    VXC_DP2x8(src0, vec0, ms0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1),\\\n\
                uniU8MulAndPostShift_0_Lo_2x8);\n\
\n\
    VXC_VertMin3_Half(src0, src0, src1, src1, VXC_MODIFIER_CLAMP(0, 7, 0, 0));\n\
    _viv_asm(COPY, dst, src0, 16);\n\
\n\
    VXC_WriteImage(output, coord, dst, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform VXC_512Bits uniU8MulAndPostShift0_Lo_2x8;\n\
_viv_uniform VXC_512Bits uniU8MulAndPostShift0_Hi_2x8;\n\
_viv_uniform VXC_512Bits uniConvertFp16toU8_2x8;\n\
_viv_uniform int2 multAndoutZP1;//[0:15] multiplier, [31:63] output zp\n\
__kernel void vxcTensorMinimum_U8F16toU8\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_uchar16 src0, dst0, dst1;\n\
    vxc_ushort8 src1, src2;\n\
    vxc_half8 data1, data2;\n\
    VXC_ReadImage2DArray(src0, input0, coord, 0, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input1, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src2, input1, coord, VXC_5BITOFFSET_XY(8, 0), \\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, data1, src1, 16);\n\
    _viv_asm(COPY, data2, src2, 16);\n\
\n\
    vxc_ushort8 mp0, mp1;\n\
    _viv_asm(COPY, mp0, multAndoutZP0, 16);\n\
    _viv_asm(COPY, mp1, multAndoutZP1, 16);\n\
    VXC_DP2x8(dst0, src0, mp0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniU8MulAndPostShift0_Lo_2x8);\n\
    VXC_DP2x8(dst0, src0, mp0, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniU8MulAndPostShift0_Hi_2x8);\n\
    VXC_DP2x8(dst1, data1, mp1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniConvertFp16toU8_2x8);\n\
    VXC_DP2x8(dst1, data2, mp1, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniConvertFp16toU8_2x8);\n\
    dst0 = min(dst0, dst1);\n\
\n\
    VXC_WriteImage2DArray(output, coord, dst0, VXC_MODIFIER(0, 15, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcTensorMinimum_U8F16toU8_2D\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int2 coord = (int2)(get_global_id(0), get_global_id(1));\n\
\n\
    vxc_uchar16 src0, dst0, dst1;\n\
    vxc_ushort8 src1, src2;\n\
    vxc_half8 data1, data2;\n\
    VXC_ReadImage(src0, input0, coord, 0, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(src1, input1, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(src2, input1, coord, VXC_5BITOFFSET_XY(8, 0), \\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, data1, src1, 16);\n\
    _viv_asm(COPY, data2, src2, 16);\n\
\n\
    vxc_ushort8 mp0, mp1;\n\
    _viv_asm(COPY, mp0, multAndoutZP0, 16);\n\
    _viv_asm(COPY, mp1, multAndoutZP1, 16);\n\
    VXC_DP2x8(dst0, src0, mp0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniU8MulAndPostShift0_Lo_2x8);\n\
    VXC_DP2x8(dst0, src0, mp0, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniU8MulAndPostShift0_Hi_2x8);\n\
    VXC_DP2x8(dst1, data1, mp1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniConvertFp16toU8_2x8);\n\
    VXC_DP2x8(dst1, data2, mp1, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniConvertFp16toU8_2x8);\n\
    dst0 = min(dst0, dst1);\n\
\n\
    VXC_WriteImage(output, coord, dst0, VXC_MODIFIER(0, 15, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcTensorMinimum_F16F16toU8\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_ushort8 src0, src1;\n\
    vxc_half8 data0, data1;\n\
    vxc_uchar16 dst0, dst1;\n\
    VXC_ReadImage2DArray(src0, input0, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input1, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, data0, src0, 16);\n\
    _viv_asm(COPY, data1, src1, 16);\n\
\n\
    vxc_ushort8 mp1;\n\
    _viv_asm(COPY, mp1, multAndoutZP1, 16);\n\
    VXC_DP2x8(dst0, data0, mp1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniConvertFp16toU8_2x8);\n\
    VXC_DP2x8(dst1, data1, mp1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniConvertFp16toU8_2x8);\n\
    dst0 = min(dst0, dst1);\n\
\n\
    VXC_WriteImage2DArray(output, coord, dst0, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcTensorMinimum_F16F16toU8_2D\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int2 coord = (int2)(get_global_id(0), get_global_id(1));\n\
\n\
    vxc_ushort8 src0, src1;\n\
    vxc_half8 data0, data1;\n\
    vxc_uchar16 dst0, dst1;\n\
    VXC_ReadImage(src0, input0, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(src1, input1, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, data0, src0, 16);\n\
    _viv_asm(COPY, data1, src1, 16);\n\
\n\
    vxc_ushort8 mp1;\n\
    _viv_asm(COPY, mp1, multAndoutZP1, 16);\n\
    VXC_DP2x8(dst0, data0, mp1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniConvertFp16toU8_2x8);\n\
    VXC_DP2x8(dst1, data1, mp1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniConvertFp16toU8_2x8);\n\
    dst0 = min(dst0, dst1);\n\
\n\
    VXC_WriteImage(output, coord, dst0, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_minimum_fp16_vx*/

static const char vsi_nn_kernel_minimum_i16_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform VXC_512Bits uniConvertI16toI16_2x8;\n\
_viv_uniform VXC_512Bits uinConvertFp16ToInt16_2x8;\n\
_viv_uniform VXC_512Bits uniConvertInt16toFp16_2x8;\n\
\n\
__kernel void vxcTensorMinimum_I16F16toI16\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_short8 src0, src1, tmp0, dst;\n\
    vxc_half8 data0;\n\
    VXC_ReadImage2DArray(src0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    _viv_asm(COPY, data0, src1, 16);\n\
\n\
    VXC_DP2x8(src0, src0, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI16toI16_2x8);\n\
    VXC_DP2x8(tmp0, data0, data0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uinConvertFp16ToInt16_2x8);\n\
    dst = min(src0, tmp0);\n\
\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcTensorMinimum_I16F16toI16_2D\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(1), get_global_id(1));\n\
\n\
    vxc_short8 src0, src1, tmp0, dst;\n\
    vxc_half8 data0;\n\
\n\
    VXC_ReadImage(src0, input0, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(src1, input1, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    _viv_asm(COPY, data0, src1, 16);\n\
\n\
    VXC_DP2x8(src0, src0, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI16toI16_2x8);\n\
    VXC_DP2x8(tmp0, data0, data0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uinConvertFp16ToInt16_2x8);\n\
    dst = min(src0, tmp0);\n\
\n\
    VXC_WriteImage(output, coord.xy, dst, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcTensorMinimum_I16F16toF16\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_short8 vec0, vec1, dst;\n\
    vxc_half8  src0, src1;\n\
\n\
    VXC_ReadImage2DArray(vec0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(vec1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, src1, vec1, 16);\n\
\n\
    VXC_DP2x8(src0, vec0, vec0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertInt16toFp16_2x8);\n\
\n\
    VXC_VertMin3_Half(src0, src0, src1, src1, VXC_MODIFIER_CLAMP(0, 7, 0, 0));\n\
    _viv_asm(COPY, dst, src0, 16);\n\
\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcTensorMinimum_I16F16toF16_2D\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(1), get_global_id(1));\n\
\n\
    vxc_short8 vec0, vec1, dst;\n\
    vxc_half8  src0, src1;\n\
    VXC_ReadImage(vec0, input0, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(vec1, input1, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, src1, vec1, 16);\n\
\n\
    VXC_DP2x8(src0, vec0, vec0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertInt16toFp16_2x8);\n\
\n\
    VXC_VertMin3_Half(src0, src0, src1, src1, VXC_MODIFIER_CLAMP(0, 7, 0, 0));\n\
    _viv_asm(COPY, dst, src0, 16);\n\
\n\
    VXC_WriteImage(output, coord.xy, dst, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_minimum_i16_vx*/

static const char vsi_nn_kernel_neg_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform VXC_512Bits UniF16Neg_2x8;\n\
\n\
#define TENSOR_NEG_F16(dst_name, dst_type, copy_type) \\\n\
    __kernel void vxTensorNeg_F16to##dst_name \\\n\
    ( \\\n\
    __read_only  image2d_array_t    input, \\\n\
    __write_only image2d_array_t    output \\\n\
    ) \\\n\
{ \\\n\
    vxc_short8 src0; \\\n\
    vxc_half8  line0; \\\n\
    dst_type   temp; \\\n\
    copy_type  dst; \\\n\
 \\\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0); \\\n\
 \\\n\
    VXC_ReadImage2DArray(src0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, line0, src0, 16); \\\n\
    VXC_DP2x8(temp, line0, line0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), UniF16Neg_2x8); \\\n\
    _viv_asm(COPY, dst, temp, 16); \\\n\
 \\\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
TENSOR_NEG_F16(F16, vxc_half8, vxc_short8)\n\
TENSOR_NEG_F16(I8,  vxc_char8, vxc_char8)\n\
TENSOR_NEG_F16(I16, vxc_short8, vxc_short8)\n\
\n\
#define TENSOR_NEG_F16_2D(dst_name, dst_type, copy_type) \\\n\
    __kernel void vxTensorNeg_F16to##dst_name##_2D \\\n\
    ( \\\n\
    __read_only  image2d_array_t    input, \\\n\
    __write_only image2d_array_t    output \\\n\
    ) \\\n\
{ \\\n\
    vxc_short8 src0; \\\n\
    vxc_half8  line0; \\\n\
    dst_type   temp; \\\n\
    copy_type  dst; \\\n\
 \\\n\
    int2 coord = (int2)(get_global_id(0), get_global_id(1)); \\\n\
 \\\n\
    VXC_ReadImage(src0, input, coord.xy, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, line0, src0, 16); \\\n\
 \\\n\
    VXC_DP2x8(temp, line0, line0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), UniF16Neg_2x8); \\\n\
    _viv_asm(COPY, dst, temp, 16); \\\n\
 \\\n\
    VXC_WriteImage(output, coord.xy, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
TENSOR_NEG_F16_2D(F16, vxc_half8, vxc_short8)\n\
TENSOR_NEG_F16_2D(I8,  vxc_char8, vxc_char8)\n\
TENSOR_NEG_F16_2D(I16, vxc_short8, vxc_short8)\n\
\n\
_viv_uniform VXC_512Bits UniNegI8Lo_2x8;\n\
_viv_uniform VXC_512Bits UniNegI8Hi_2x8;\n\
\n\
__kernel void vxTensorNeg_I8toI8\n\
    (\n\
    __read_only  image2d_array_t    input,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    vxc_char16  src0, dst;\n\
\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    VXC_ReadImage2DArray(src0, input, coord, 0, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_DP2x8(dst, src0, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), UniNegI8Lo_2x8);\n\
    VXC_DP2x8(dst, src0, src0, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), UniNegI8Hi_2x8);\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxTensorNeg_I8toI8_2D\n\
    (\n\
    __read_only  image2d_array_t    input,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    vxc_char16  src0, dst;\n\
\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(1), get_global_id(1));\n\
\n\
    VXC_ReadImage(src0, input, coord.xy, 0, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_DP2x8(dst, src0, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), UniNegI8Lo_2x8);\n\
    VXC_DP2x8(dst, src0, src0, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), UniNegI8Hi_2x8);\n\
    VXC_WriteImage(output, coord.xy, dst, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxTensorNeg_I8toF16\n\
    (\n\
    __read_only  image2d_array_t    input,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    vxc_char16  src0;\n\
    vxc_short8  vec0, vec1;\n\
    vxc_half8   dst0, dst1;\n\
\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    VXC_ReadImage2DArray(src0, input, coord, 0, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_DP2x8(dst0, src0, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), UniNegI8Lo_2x8);\n\
    VXC_DP2x8(dst1, src0, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), UniNegI8Hi_2x8);\n\
    _viv_asm(COPY, vec0, dst0, 16);\n\
    _viv_asm(COPY, vec1, dst1, 16);\n\
\n\
    VXC_WriteImage2DArray(output, coord, vec0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    coord.x += 8;\n\
    VXC_WriteImage2DArray(output, coord, vec1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxTensorNeg_I8toF16_2D\n\
    (\n\
    __read_only  image2d_array_t    input,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    vxc_char16  src0;\n\
    vxc_short8  vec0, vec1;\n\
    vxc_half8   dst0, dst1;\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), get_global_id(0));\n\
\n\
    VXC_ReadImage(src0, input, coord.xy, 0, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    coord.z += 8;\n\
    VXC_DP2x8(dst0, src0, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), UniNegI8Lo_2x8);\n\
    VXC_DP2x8(dst1, src0, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), UniNegI8Hi_2x8);\n\
    _viv_asm(COPY, vec0, dst0, 16);\n\
    _viv_asm(COPY, vec1, dst1, 16);\n\
\n\
    VXC_WriteImage(output, coord.xy, vec0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_WriteImage(output, coord.zy, vec1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform VXC_512Bits UniNegI16_2x8;\n\
\n\
__kernel void vxTensorNeg_I16toI16\n\
    (\n\
    __read_only  image2d_array_t    input,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    vxc_short8  src0, dst;\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    VXC_ReadImage2DArray(src0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_DP2x8(dst, src0, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), UniNegI16_2x8);\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxTensorNeg_I16toI16_2D\n\
    (\n\
    __read_only  image2d_array_t    input,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    vxc_short8  src0, dst;\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(1), get_global_id(1));\n\
\n\
    VXC_ReadImage(src0, input, coord.xy, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_DP2x8(dst, src0, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), UniNegI16_2x8);\n\
    VXC_WriteImage(output, coord.xy, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxTensorNeg_I16toF16\n\
    (\n\
    __read_only  image2d_array_t    input,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    vxc_short8  src0;\n\
    vxc_short8  vec0;\n\
    vxc_half8   dst0;\n\
\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    VXC_ReadImage2DArray(src0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_DP2x8(dst0, src0, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), UniNegI16_2x8);\n\
    _viv_asm(COPY, vec0, dst0, 16);\n\
\n\
    VXC_WriteImage2DArray(output, coord, vec0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxTensorNeg_I16toF16_2D\n\
    (\n\
    __read_only  image2d_array_t    input,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    vxc_short8  src0;\n\
    vxc_short8  vec0;\n\
    vxc_half8   dst0;\n\
\n\
    int2 coord = (int2)(get_global_id(0), get_global_id(1));\n\
\n\
    VXC_ReadImage(src0, input, coord.xy, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_DP2x8(dst0, src0, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), UniNegI16_2x8);\n\
    _viv_asm(COPY, vec0, dst0, 16);\n\
    VXC_WriteImage(output, coord.xy, vec0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform int inputZP;\n\
_viv_uniform int2 multAndoutZP;//[0:15] multiplier, [31:63] output zp\n\
_viv_uniform VXC_512Bits uniDataMulAndPostShift_2x8;\n\
_viv_uniform VXC_512Bits uniU8toI16Lo_2x8;\n\
_viv_uniform VXC_512Bits uniU8toI16Hi_2x8;\n\
\n\
__kernel void vxTensorNeg_U8toU8\n\
    (\n\
    __read_only  image2d_array_t    input,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    vxc_uchar16 src, dst;\n\
    vxc_short8  src0, src1;\n\
    vxc_ushort8 srcA, srcB;\n\
\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    VXC_ReadImage2DArray(src, input, coord, 0, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    vxc_uchar16 zp;\n\
\n\
    _viv_asm(COPY, zp, inputZP, 16);\n\
    VXC_DP2x8(src0, src, zp, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniU8toI16Lo_2x8);\n\
    VXC_DP2x8(src1, src, zp, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniU8toI16Hi_2x8);\n\
\n\
    _viv_asm(NEG, srcA, src0);\n\
    _viv_asm(NEG, srcB, src1);\n\
\n\
    vxc_ushort8 multiplier;\n\
    _viv_asm(COPY, multiplier, multAndoutZP, 16);\n\
    VXC_DP2x8(dst, srcA, multiplier, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniDataMulAndPostShift_2x8);\n\
    VXC_DP2x8(dst, srcB, multiplier, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniDataMulAndPostShift_2x8);\n\
\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxTensorNeg_U8toU8_2D\n\
    (\n\
    __read_only  image2d_array_t    input,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    vxc_uchar16 src, dst;\n\
    vxc_short8  src0, src1;\n\
    vxc_ushort8 srcA, srcB;\n\
\n\
    int2 coord = (int2)(get_global_id(0), get_global_id(1));\n\
\n\
    VXC_ReadImage(src, input, coord, 0, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    vxc_uchar16 zp;\n\
\n\
    _viv_asm(COPY, zp, inputZP, 16);\n\
    VXC_DP2x8(src0, src, zp, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniU8toI16Lo_2x8);\n\
    VXC_DP2x8(src1, src, zp, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniU8toI16Hi_2x8);\n\
\n\
    _viv_asm(NEG, srcA, src0);\n\
    _viv_asm(NEG, srcB, src1);\n\
\n\
    vxc_ushort8 multiplier;\n\
    _viv_asm(COPY, multiplier, multAndoutZP, 16);\n\
    VXC_DP2x8(dst, srcA, multiplier, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniDataMulAndPostShift_2x8);\n\
    VXC_DP2x8(dst, srcB, multiplier, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniDataMulAndPostShift_2x8);\n\
\n\
    VXC_WriteImage(output, coord, dst, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
\n\
__kernel void vxTensorNeg_U8toF16\n\
    (\n\
    __read_only  image2d_array_t    input,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    vxc_uchar16 src, dst;\n\
    vxc_short8  src0, src1;\n\
    vxc_ushort8 srcA, srcB;\n\
    vxc_short8  vec0, vec1;\n\
    vxc_half8   dst0, dst1;\n\
\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    VXC_ReadImage2DArray(src, input, coord, 0, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    vxc_uchar16 zp;\n\
\n\
    _viv_asm(COPY, zp, inputZP, 16);\n\
    VXC_DP2x8(src0, src, zp, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniU8toI16Lo_2x8);\n\
    VXC_DP2x8(src1, src, zp, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniU8toI16Hi_2x8);\n\
\n\
    _viv_asm(NEG, srcA, src0);\n\
    _viv_asm(NEG, srcB, src1);\n\
\n\
    vxc_ushort8 multiplier;\n\
    _viv_asm(COPY, multiplier, multAndoutZP, 16);\n\
    VXC_DP2x8(dst0, srcA, multiplier, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniDataMulAndPostShift_2x8);\n\
    VXC_DP2x8(dst1, srcB, multiplier, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniDataMulAndPostShift_2x8);\n\
    _viv_asm(COPY, vec0, dst0, 16);\n\
    _viv_asm(COPY, vec1, dst1, 16);\n\
\n\
    VXC_WriteImage2DArray(output, coord, vec0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    coord.x += 8;\n\
    VXC_WriteImage2DArray(output, coord, vec1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxTensorNeg_U8toF16_2D\n\
    (\n\
    __read_only  image2d_array_t    input,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    vxc_uchar16 src, dst;\n\
    vxc_short8  src0, src1;\n\
    vxc_ushort8 srcA, srcB;\n\
    vxc_short8  vec0, vec1;\n\
    vxc_half8   dst0, dst1;\n\
\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), get_global_id(0));\n\
\n\
    VXC_ReadImage(src, input, coord.xy, 0, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    coord.z += 8;\n\
\n\
    vxc_uchar16 zp;\n\
\n\
    _viv_asm(COPY, zp, inputZP, 16);\n\
    VXC_DP2x8(src0, src, zp, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniU8toI16Lo_2x8);\n\
    VXC_DP2x8(src1, src, zp, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniU8toI16Hi_2x8);\n\
\n\
    _viv_asm(NEG, srcA, src0);\n\
    _viv_asm(NEG, srcB, src1);\n\
\n\
    vxc_ushort8 multiplier;\n\
    _viv_asm(COPY, multiplier, multAndoutZP, 16);\n\
    VXC_DP2x8(dst0, srcA, multiplier, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniDataMulAndPostShift_2x8);\n\
    VXC_DP2x8(dst1, srcB, multiplier, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniDataMulAndPostShift_2x8);\n\
    _viv_asm(COPY, vec0, dst0, 16);\n\
    _viv_asm(COPY, vec1, dst1, 16);\n\
\n\
    VXC_WriteImage(output, coord.xy, vec0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_WriteImage(output, coord.zy, vec1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxTensorNeg_F16toU8\n\
    (\n\
    __read_only  image2d_t          input,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    vxc_short8 src0;\n\
    vxc_half8  line0, line1;\n\
    vxc_uchar8 dst;\n\
\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    VXC_ReadImage2DArray(src0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, line0, src0, 16);\n\
    VXC_DP2x8(line1, line0, line0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), UniF16Neg_2x8);\n\
\n\
    vxc_ushort8 multiplier;\n\
    _viv_asm(COPY, multiplier, multAndoutZP, 16);\n\
    VXC_DP2x8(dst, line1, multiplier, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniDataMulAndPostShift_2x8);\n\
\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxTensorNeg_F16toU8_2D\n\
    (\n\
    __read_only  image2d_t          input,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    vxc_short8 src0;\n\
    vxc_half8  line0, line1;\n\
    vxc_uchar8 dst;\n\
\n\
    int2 coord = (int2)(get_global_id(0), get_global_id(1));\n\
\n\
    VXC_ReadImage(src0, input, coord.xy, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, line0, src0, 16);\n\
    VXC_DP2x8(line1, line0, line0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), UniF16Neg_2x8);\n\
\n\
    vxc_ushort8 multiplier;\n\
    _viv_asm(COPY, multiplier, multAndoutZP, 16);\n\
    VXC_DP2x8(dst, line1, multiplier, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniDataMulAndPostShift_2x8);\n\
\n\
    VXC_WriteImage(output, coord.xy, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}"; /* end of vsi_nn_kernel_neg_vx*/

static const char vsi_nn_kernel_poolwithargmax_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
//-------------------max pooling with argmax---------------\n\
_viv_uniform VXC_512Bits poolingEncode;\n\
\n\
__kernel void poolingWithArgmax\n\
    (\n\
    image2d_array_t tensorIn,\n\
    image2d_array_t tensorOut,\n\
    image2d_array_t axis,\n\
    int type,\n\
    int sizeX,\n\
    int sizeY,\n\
    int paddingX,\n\
    int paddingY\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    vxc_short8 din0, din1, maxData, src0, src1;\n\
    vxc_half8 din0Fp16, din1Fp16;\n\
    vxc_half8 maxDataVer, maxDataVer1;\n\
    int4 bitExtractCoeff;\n\
    vxc_short8 din0EqualTmp, din1EqualTmp;\n\
    vxc_uchar8 din0Equal, din1Equal;\n\
    vxc_uchar4 axisEncode;\n\
    vxc_uchar4 axisOut;\n\
\n\
    VXC_ReadImage2DArray(src0, tensorIn, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, tensorIn, coord, VXC_5BITOFFSET_XY(0, 1),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    int4 coordOut = (int4)(coord.x >> 1, coord.y >> 1, coord.z, 0);\n\
    _viv_asm(COPY, din0Fp16, src0, 16);\n\
    _viv_asm(COPY, din1Fp16, src1, 16);\n\
    VXC_VertMax3_Half(maxDataVer, din0Fp16, din1Fp16, din1Fp16, VXC_MODIFIER_BIN(0, 7, 0));\n\
    _viv_asm(COPY, din0, maxDataVer, 16);\n\
    din1 = din0.s10325476;\n\
    _viv_asm(COPY, maxDataVer1, din1, 16);\n\
    VXC_VertMax3_Half(maxDataVer, maxDataVer1, maxDataVer, maxDataVer, VXC_MODIFIER_BIN(0, 7, 0));\n\
    _viv_asm(COPY, din0, maxDataVer, 16);\n\
    din1 = din0.s02460246;//output\n\
    //get axis\n\
    _viv_asm(COPY, maxData, maxDataVer, 16);\n\
    vxc_short8 one = (vxc_short8)(1, 1, 1, 1, 1, 1, 1, 1);\n\
    vxc_short8 zero = (vxc_short8)(0, 0, 0, 0, 0, 0, 0, 0);\n\
    din0EqualTmp = src0 == maxData ? one : zero;\n\
    din1EqualTmp = src1 == maxData ? one : zero;\n\
    VXC_DP4x4(axisEncode, din0EqualTmp, din1EqualTmp, VXC_MODIFIER_BIN(0, 3, 0), poolingEncode);\n\
    axisOut = clz(axisEncode);//output\n\
    //write data out\n\
    VXC_WriteImage2DArray(tensorOut, coordOut, din1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    VXC_WriteImage2DArray(axis, coordOut, axisOut, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
//-------------------max pooling with argmax uint8---------------\n\
\n\
_viv_uniform int input_ZP;\n\
_viv_uniform int output_ZP;\n\
_viv_uniform float inputScale;\n\
_viv_uniform float outputScale;\n\
_viv_uniform VXC_512Bits uniConvertUint8ToFp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertSubZpUint8Fp32_4x4;\n\
_viv_uniform VXC_512Bits uniPackHalf2Short_2x8;\n\
_viv_uniform VXC_512Bits uniExtractHalf2Short_2x8;\n\
_viv_uniform VXC_512Bits uniPackHalf8_2x8;\n\
_viv_uniform VXC_512Bits uniU8EvenBinSubZP_MulM_2x8;\n\
_viv_uniform VXC_512Bits uniEncodeUint8_4x8;\n\
_viv_uniform VXC_512Bits uniS16AddOutZP_2x8;\n\
_viv_uniform vxc_uint4 packed_outputZP;\n\
\n\
__kernel void poolingWithArgmaxUint8_s2k2p0\n\
    (\n\
    image2d_array_t tensorIn,\n\
    image2d_array_t tensorOut,\n\
    image2d_array_t axis,\n\
    int type,\n\
    int sizeX,\n\
    int sizeY,\n\
    int paddingX,\n\
    int paddingY\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    vxc_uchar16 din0, din1;\n\
    vxc_uchar16 maxDataVer, maxDataVer1;\n\
    int4 bitExtractCoeff;\n\
    vxc_uchar16 din0EqualTmp, din1EqualTmp;\n\
    vxc_uchar8 axisEncode;\n\
    vxc_uchar8 axisOut;\n\
\n\
    VXC_ReadImage2DArray(din0, tensorIn, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(din1, tensorIn, coord, VXC_5BITOFFSET_XY(0, 1),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    int4 coordOut = (int4)(coord.x >> 1, coord.y >> 1, coord.z, 0);\n\
    VXC_VertMax3_Integer(maxDataVer, din0, din1, din1, VXC_MODIFIER_BIN(0, 15, 0));\n\
    maxDataVer1 = maxDataVer.s1032547698badcfe;\n\
    VXC_VertMax3_Integer(maxDataVer, maxDataVer1, maxDataVer, maxDataVer, VXC_MODIFIER_BIN(0, 15, 0));\n\
\n\
    vxc_short8 tmp;\n\
    uchar zp = input_ZP;\n\
    VXC_DP2x8(tmp, maxDataVer, zp, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniU8EvenBinSubZP_MulM_2x8);\n\
    vxc_uchar16 packed_outZP;\n\
    _viv_asm(COPY, packed_outZP, packed_outputZP, 16);\n\
    VXC_DP2x8(maxDataVer1, tmp, packed_outZP, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniS16AddOutZP_2x8);\n\
    VXC_WriteImage2DArray(tensorOut, coordOut, maxDataVer1,\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    //get axis\n\
    VXC_Clamp(din0EqualTmp, din0, maxDataVer, maxDataVer, VXC_MODIFIER_CLAMP(0, 15, 0, 1));\n\
    VXC_Clamp(din1EqualTmp, din1, maxDataVer, maxDataVer, VXC_MODIFIER_CLAMP(0, 15, 0, 1));\n\
    din0EqualTmp &= (vxc_uchar16)(1);\n\
    din1EqualTmp &= (vxc_uchar16)(1);\n\
\n\
    VXC_DP4x8(axisEncode, din0EqualTmp, din1EqualTmp, VXC_MODIFIER_BIN(0, 7, 0), uniEncodeUint8_4x8);\n\
    axisOut = clz(axisEncode);//output\n\
    //write data out\n\
    VXC_WriteImage2DArray(axis, coordOut, axisOut, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform VXC_512Bits uniConvertEvenU8ToFp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertEvenU8SubZpToFp32_4x4;\n\
\n\
__kernel void poolingWithArgmaxUint8_fp16_s2k2p0\n\
    (\n\
    image2d_array_t tensorIn,\n\
    image2d_array_t tensorOut,\n\
    image2d_array_t axis,\n\
    int type,\n\
    int sizeX,\n\
    int sizeY,\n\
    int paddingX,\n\
    int paddingY\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    vxc_uchar16 din0, din1;\n\
    vxc_uchar16 maxDataVer, maxDataVer1;\n\
    int4 bitExtractCoeff;\n\
    vxc_uchar16 din0EqualTmp, din1EqualTmp;\n\
    vxc_uchar8 axisEncode;\n\
    vxc_uchar8 axisOut;\n\
    vxc_short8 result;\n\
\n\
    VXC_ReadImage2DArray(din0, tensorIn, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(din1, tensorIn, coord, VXC_5BITOFFSET_XY(0, 1),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    int4 coordOut = (int4)(coord.x >> 1, coord.y >> 1, coord.z, 0);\n\
    VXC_VertMax3_Integer(maxDataVer, din0, din1, din1, VXC_MODIFIER_BIN(0, 15, 0));\n\
    maxDataVer1 = maxDataVer.s1032547698badcfe;\n\
    VXC_VertMax3_Integer(maxDataVer, maxDataVer1, maxDataVer,\\\n\
        maxDataVer, VXC_MODIFIER_BIN(0, 15, 0));\n\
    //maxDataVer1 = maxDataVer.s02468ace02468ace;//output\n\
\n\
    vxc_float4 tmpVal0, tmpVal1, tmpVal2, tmpVal3;\n\
    half4 tmpOut0, tmpOut1;\n\
    vxc_half8 tmpPack;\n\
    vxc_short4 tmpOut2, tmpOut3;\n\
    uchar zp = input_ZP;\n\
    VXC_DP4x4(tmpVal0, maxDataVer, zp, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertEvenU8ToFp32_4x4);\n\
    VXC_DP4x4(tmpVal2, maxDataVer, zp, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertEvenU8SubZpToFp32_4x4);\n\
    tmpVal1 = tmpVal0 * inputScale;\n\
    _viv_asm(CONV, tmpOut0, tmpVal1);\n\
    //_viv_asm(COPY, tmpOut2, tmpOut0, 8);\n\
    tmpVal3 = tmpVal2 * inputScale;\n\
    _viv_asm(CONV, tmpOut1, tmpVal3);\n\
    //_viv_asm(COPY, tmpOut3, tmpOut1, 8);\n\
    //VXC_DP2x8(result, tmpOut2, tmpOut3, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
        uniPackHalf2Short_2x8);\n\
    //VXC_DP2x8(result, tmpOut2, tmpOut3, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
        uniExtractHalf2Short_2x8);\n\
    VXC_DP2x8(tmpPack, tmpOut0, tmpOut1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
        uniPackHalf8_2x8);\n\
    _viv_asm(COPY, result, tmpPack, 16);\n\
    VXC_WriteImage2DArray(tensorOut, coordOut, result,\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    //get axis\n\
    VXC_Clamp(din0EqualTmp, din0, maxDataVer, maxDataVer, VXC_MODIFIER_CLAMP(0, 15, 0, 1));\n\
    VXC_Clamp(din1EqualTmp, din1, maxDataVer, maxDataVer, VXC_MODIFIER_CLAMP(0, 15, 0, 1));\n\
    din0EqualTmp &= (vxc_uchar16)(1);\n\
    din1EqualTmp &= (vxc_uchar16)(1);\n\
    VXC_DP4x8(axisEncode, din0EqualTmp, din1EqualTmp, VXC_MODIFIER_BIN(0, 7, 0), uniEncodeUint8_4x8);\n\
    axisOut = clz(axisEncode);//output\n\
\n\
    //write data out\n\
    VXC_WriteImage2DArray(axis, coordOut, axisOut, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void poolingWithArgmaxUint8_fp16_fp16_s2k2p0\n\
    (\n\
    image2d_array_t tensorIn,\n\
    image2d_array_t tensorOut,\n\
    image2d_array_t axis,\n\
    int type,\n\
    int sizeX,\n\
    int sizeY,\n\
    int paddingX,\n\
    int paddingY\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    vxc_uchar16 din0, din1;\n\
    vxc_uchar16 maxDataVer, maxDataVer1;\n\
    int4 bitExtractCoeff;\n\
    vxc_uchar16 din0EqualTmp, din1EqualTmp;\n\
    vxc_uchar8 axisEncode;\n\
    vxc_uchar8 axisOut;\n\
    vxc_short8 result, axisResult;\n\
\n\
    VXC_ReadImage2DArray(din0, tensorIn, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(din1, tensorIn, coord, VXC_5BITOFFSET_XY(0, 1),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    int4 coordOut = (int4)(coord.x >> 1, coord.y >> 1, coord.z, 0);\n\
    VXC_VertMax3_Integer(maxDataVer, din0, din1, din1, VXC_MODIFIER_BIN(0, 15, 0));\n\
    maxDataVer1 = maxDataVer.s1032547698badcfe;\n\
    VXC_VertMax3_Integer(maxDataVer, maxDataVer1, maxDataVer, maxDataVer, VXC_MODIFIER_BIN(0, 15, 0));\n\
    maxDataVer1 = maxDataVer.s02468ace02468ace;//output\n\
\n\
    vxc_float4 tmpVal0, tmpVal1, tmpVal2, tmpVal3;\n\
    half4 tmpOut0, tmpOut1;\n\
    vxc_half8 tmpPack;\n\
    vxc_short4 tmpOut2, tmpOut3;\n\
    uchar zp = input_ZP;\n\
    VXC_DP4x4(tmpVal0, maxDataVer1, zp, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertUint8ToFp32_4x4);\n\
    VXC_DP4x4(tmpVal2, maxDataVer1, zp, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertSubZpUint8Fp32_4x4);\n\
    tmpVal1 = tmpVal0 * inputScale;\n\
    _viv_asm(CONV, tmpOut0, tmpVal1);\n\
    //_viv_asm(COPY, tmpOut2, tmpOut0, 8);\n\
    tmpVal3 = tmpVal2 * inputScale;\n\
    _viv_asm(CONV, tmpOut1, tmpVal3);\n\
    //_viv_asm(COPY, tmpOut3, tmpOut1, 8);\n\
    //VXC_DP2x8(result, tmpOut2, tmpOut3, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
        uniPackHalf2Short_2x8);\n\
    //VXC_DP2x8(result, tmpOut2, tmpOut3, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
        uniExtractHalf2Short_2x8);\n\
    VXC_DP2x8(tmpPack, tmpOut0, tmpOut1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
        uniPackHalf8_2x8);\n\
    _viv_asm(COPY, result, tmpPack, 16);\n\
\n\
    VXC_Clamp(din0EqualTmp, din0, maxDataVer, maxDataVer, VXC_MODIFIER_CLAMP(0, 15, 0, 1));\n\
    VXC_Clamp(din1EqualTmp, din1, maxDataVer, maxDataVer, VXC_MODIFIER_CLAMP(0, 15, 0, 1));\n\
    din0EqualTmp &= (vxc_uchar16)(1);\n\
    din1EqualTmp &= (vxc_uchar16)(1);\n\
    VXC_DP4x8(axisEncode, din0EqualTmp, din1EqualTmp, VXC_MODIFIER_BIN(0, 7, 0), uniEncodeUint8_4x8);\n\
    axisOut = clz(axisEncode);//output\n\
    _viv_asm(CONV, axisResult, axisOut);\n\
\n\
    //write data out\n\
    VXC_WriteImage2DArray(tensorOut, coordOut, result, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_WriteImage2DArray(axis, coordOut, axisResult, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
"; /* end of vsi_nn_kernel_poolwithargmax_vx*/

static const char vsi_nn_kernel_poolwithargmax_i16_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform VXC_512Bits poolingEncode2;\n\
_viv_uniform VXC_512Bits uniConvertDirInt16Fp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertEndInt16Fp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertInt32toInt16_2x8;\n\
_viv_uniform float scaleSF;\n\
_viv_uniform float input_fl_scale_i16;\n\
_viv_uniform VXC_512Bits uniPackHalf8_2x8_2;\n\
\n\
__kernel void poolingWithArgmaxInt16_s2k2p0\n\
    (\n\
    image2d_array_t tensorIn,\n\
    image2d_array_t tensorOut,\n\
    image2d_array_t axis,\n\
    int type,\n\
    int sizeX,\n\
    int sizeY,\n\
    int paddingX,\n\
    int paddingY\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    vxc_short8 din0, din1;\n\
    vxc_short8 din0Fp16, din1Fp16;\n\
    vxc_short8 maxDataVer, maxDataVer1;\n\
    int4 bitExtractCoeff;\n\
    vxc_short8 din0EqualTmp, din1EqualTmp;\n\
    vxc_uchar8 din0Equal, din1Equal;\n\
    vxc_uchar4 axisEncode;\n\
    vxc_uchar4 axisOut;\n\
\n\
    VXC_ReadImage2DArray(din0, tensorIn, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(din1, tensorIn, coord, VXC_5BITOFFSET_XY(0, 1),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    int4 coordOut = (int4)(coord.x >> 1, coord.y >> 1, coord.z, 0);\n\
    _viv_asm(COPY, din0Fp16, din0, 16);\n\
    _viv_asm(COPY, din1Fp16, din1, 16);\n\
    VXC_VertMax3_Integer(maxDataVer, din0Fp16, din1Fp16, din1Fp16, VXC_MODIFIER_BIN(0, 7, 0));\n\
    _viv_asm(COPY, din0, maxDataVer, 16);\n\
    din1 = din0.s10325476;\n\
    _viv_asm(COPY, maxDataVer1, din1, 16);\n\
    VXC_VertMax3_Integer(maxDataVer, maxDataVer1, maxDataVer, maxDataVer, VXC_MODIFIER_BIN(0, 7, 0));\n\
    _viv_asm(COPY, din0, maxDataVer, 16);\n\
    din1 = din0.s02460246;//output\n\
    VXC_WriteImage2DArray(tensorOut, coordOut, din1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
\n\
    //get axis\n\
    VXC_Clamp(din0EqualTmp, din0Fp16, maxDataVer, maxDataVer, VXC_MODIFIER_CLAMP(0, 7, 0, 1));\n\
    VXC_Clamp(din1EqualTmp, din1Fp16, maxDataVer, maxDataVer, VXC_MODIFIER_CLAMP(0, 7, 0, 1));\n\
    bitExtractCoeff = (int4)(0x30201000, 0x70605040, 0x01010101, 0x01010101);\n\
    VXC_BitExtract(din0Equal, din0EqualTmp, din0EqualTmp, bitExtractCoeff, VXC_MODIFIER_BIN(0, 7, 0));\n\
    VXC_BitExtract(din1Equal, din1EqualTmp, din1EqualTmp, bitExtractCoeff, VXC_MODIFIER_BIN(0, 7, 0));\n\
    VXC_DP4x4(axisEncode, din0Equal, din1Equal, VXC_MODIFIER_BIN(0, 3, 0), poolingEncode2);\n\
    axisOut = clz(axisEncode);//output\n\
\n\
    //write data out\n\
    VXC_WriteImage2DArray(axis, coordOut, axisOut, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void poolingWithArgmaxInt16_int16_s2k2p0\n\
    (\n\
    image2d_array_t tensorIn,\n\
    image2d_array_t tensorOut,\n\
    image2d_array_t axis,\n\
    int type,\n\
    int sizeX,\n\
    int sizeY,\n\
    int paddingX,\n\
    int paddingY\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    vxc_short8 din0, din1;\n\
    vxc_short8 din0Fp16, din1Fp16;\n\
    vxc_short8 maxDataVer, maxDataVer1;\n\
    int4 bitExtractCoeff;\n\
    vxc_short8 din0EqualTmp, din1EqualTmp;\n\
    vxc_uchar8 din0Equal, din1Equal;\n\
    vxc_uchar4 axisEncode;\n\
    vxc_uchar4 axisOut;\n\
\n\
    VXC_ReadImage2DArray(din0, tensorIn, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(din1, tensorIn, coord, VXC_5BITOFFSET_XY(0, 1),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    int4 coordOut = (int4)(coord.x >> 1, coord.y >> 1, coord.z, 0);\n\
    _viv_asm(COPY, din0Fp16, din0, 16);\n\
    _viv_asm(COPY, din1Fp16, din1, 16);\n\
    VXC_VertMax3_Integer(maxDataVer, din0Fp16, din1Fp16, din1Fp16, VXC_MODIFIER_BIN(0, 7, 0));\n\
    _viv_asm(COPY, din0, maxDataVer, 16);\n\
    din1 = din0.s10325476;\n\
    _viv_asm(COPY, maxDataVer1, din1, 16);\n\
    VXC_VertMax3_Integer(maxDataVer, maxDataVer1, maxDataVer, maxDataVer, VXC_MODIFIER_BIN(0, 7, 0));\n\
    _viv_asm(COPY, din0, maxDataVer, 16);\n\
    din1 = din0.s02460246;//output\n\
\n\
    // convert to fp32, and then convert it back\n\
    vxc_float4 tmpVal0, tmpVal1, tmpVal2, tmpVal3;\n\
    vxc_int4 tmpOut0, tmpOut1;\n\
\n\
    //convert to fp32 and then convert back\n\
    VXC_DP4x4(tmpVal0, din1, din1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertDirInt16Fp32_4x4);\n\
    VXC_DP4x4(tmpVal2, din1, din1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertEndInt16Fp32_4x4);\n\
    tmpVal1 = tmpVal0 * scaleSF;\n\
    tmpOut0 = convert_int4_rte(tmpVal1);\n\
    tmpVal3 = tmpVal2 * scaleSF;\n\
    tmpOut1 = convert_int4_rte(tmpVal3);\n\
    VXC_DP2x8(din1, tmpOut0, tmpOut1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniConvertInt32toInt16_2x8);\n\
\n\
    //get axis\n\
    VXC_Clamp(din0EqualTmp, din0Fp16, maxDataVer, maxDataVer, VXC_MODIFIER_CLAMP(0, 7, 0, 1));\n\
    VXC_Clamp(din1EqualTmp, din1Fp16, maxDataVer, maxDataVer, VXC_MODIFIER_CLAMP(0, 7, 0, 1));\n\
    bitExtractCoeff = (int4)(0x30201000, 0x70605040, 0x01010101, 0x01010101);\n\
    VXC_BitExtract(din0Equal, din0EqualTmp, din0EqualTmp, bitExtractCoeff, VXC_MODIFIER_BIN(0, 7, 0));\n\
    VXC_BitExtract(din1Equal, din1EqualTmp, din1EqualTmp, bitExtractCoeff, VXC_MODIFIER_BIN(0, 7, 0));\n\
    VXC_DP4x4(axisEncode, din0Equal, din1Equal, VXC_MODIFIER_BIN(0, 3, 0), poolingEncode2);\n\
    axisOut = clz(axisEncode);//output\n\
\n\
    //write data out\n\
    VXC_WriteImage2DArray(tensorOut, coordOut, din1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    VXC_WriteImage2DArray(axis, coordOut, axisOut, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void poolingWithArgmaxInt16_fp16_s2k2p0\n\
    (\n\
    image2d_array_t tensorIn,\n\
    image2d_array_t tensorOut,\n\
    image2d_array_t axis,\n\
    int type,\n\
    int sizeX,\n\
    int sizeY,\n\
    int paddingX,\n\
    int paddingY\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    vxc_short8 din0, din1;\n\
    vxc_short8 din0Fp16, din1Fp16;\n\
    vxc_short8 maxDataVer, maxDataVer1;\n\
    int4 bitExtractCoeff;\n\
    vxc_short8 din0EqualTmp, din1EqualTmp;\n\
    vxc_uchar8 din0Equal, din1Equal;\n\
    vxc_uchar4 axisEncode;\n\
    vxc_uchar4 axisOut;\n\
\n\
    VXC_ReadImage2DArray(din0, tensorIn, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(din1, tensorIn, coord, VXC_5BITOFFSET_XY(0, 1),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    int4 coordOut = (int4)(coord.x >> 1, coord.y >> 1, coord.z, 0);\n\
    _viv_asm(COPY, din0Fp16, din0, 16);\n\
    _viv_asm(COPY, din1Fp16, din1, 16);\n\
    VXC_VertMax3_Integer(maxDataVer, din0Fp16, din1Fp16, din1Fp16, VXC_MODIFIER_BIN(0, 7, 0));\n\
    _viv_asm(COPY, din0, maxDataVer, 16);\n\
    din1 = din0.s10325476;\n\
    _viv_asm(COPY, maxDataVer1, din1, 16);\n\
    VXC_VertMax3_Integer(maxDataVer, maxDataVer1, maxDataVer, maxDataVer, VXC_MODIFIER_BIN(0, 7, 0));\n\
    _viv_asm(COPY, din0, maxDataVer, 16);\n\
    din1 = din0.s02460246;//output\n\
\n\
    // convert to fp32, and then convert it back\n\
    vxc_float4 tmpVal0, tmpVal1, tmpVal2, tmpVal3;\n\
    half4 tmpOut0, tmpOut1;\n\
    vxc_half8 tmpPack;\n\
\n\
    //convert to fp32 and then convert back\n\
    VXC_DP4x4(tmpVal0, din1, din1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertDirInt16Fp32_4x4);\n\
    VXC_DP4x4(tmpVal2, din1, din1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertEndInt16Fp32_4x4);\n\
    tmpVal1 = tmpVal0 * input_fl_scale_i16;\n\
    _viv_asm(CONV, tmpOut0, tmpVal1);\n\
    tmpVal3 = tmpVal2 * input_fl_scale_i16;\n\
    _viv_asm(CONV, tmpOut1, tmpVal3);\n\
    VXC_DP2x8(tmpPack, tmpOut0, tmpOut1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniPackHalf8_2x8_2);\n\
    _viv_asm(COPY, din1, tmpPack, 16);\n\
\n\
    //get axis\n\
    VXC_Clamp(din0EqualTmp, din0Fp16, maxDataVer, maxDataVer, VXC_MODIFIER_CLAMP(0, 7, 0, 1));\n\
    VXC_Clamp(din1EqualTmp, din1Fp16, maxDataVer, maxDataVer, VXC_MODIFIER_CLAMP(0, 7, 0, 1));\n\
    bitExtractCoeff = (int4)(0x30201000, 0x70605040, 0x01010101, 0x01010101);\n\
    VXC_BitExtract(din0Equal, din0EqualTmp, din0EqualTmp, bitExtractCoeff, VXC_MODIFIER_BIN(0, 7, 0));\n\
    VXC_BitExtract(din1Equal, din1EqualTmp, din1EqualTmp, bitExtractCoeff, VXC_MODIFIER_BIN(0, 7, 0));\n\
    VXC_DP4x4(axisEncode, din0Equal, din1Equal, VXC_MODIFIER_BIN(0, 3, 0), poolingEncode2);\n\
    axisOut = clz(axisEncode);//output\n\
\n\
    //write data out\n\
    VXC_WriteImage2DArray(tensorOut, coordOut, din1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    VXC_WriteImage2DArray(axis, coordOut, axisOut, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void poolingWithArgmaxInt16_axI16_s2k2p0\n\
    (\n\
    image2d_array_t tensorIn,\n\
    image2d_array_t tensorOut,\n\
    image2d_array_t axis,\n\
    int type,\n\
    int sizeX,\n\
    int sizeY,\n\
    int paddingX,\n\
    int paddingY\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    vxc_short8 din0, din1;\n\
    vxc_short8 din0Fp16, din1Fp16;\n\
    vxc_short8 maxDataVer, maxDataVer1;\n\
    int4 bitExtractCoeff;\n\
    vxc_short8 din0EqualTmp, din1EqualTmp;\n\
    vxc_uchar8 din0Equal, din1Equal;\n\
    vxc_uchar4 axisEncode;\n\
    vxc_uchar4 axisOut;\n\
    vxc_short4 axisVal;\n\
\n\
    VXC_ReadImage2DArray(din0, tensorIn, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(din1, tensorIn, coord, VXC_5BITOFFSET_XY(0, 1),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    int4 coordOut = (int4)(coord.x >> 1, coord.y >> 1, coord.z, 0);\n\
    _viv_asm(COPY, din0Fp16, din0, 16);\n\
    _viv_asm(COPY, din1Fp16, din1, 16);\n\
    VXC_VertMax3_Integer(maxDataVer, din0Fp16, din1Fp16, din1Fp16, VXC_MODIFIER_BIN(0, 7, 0));\n\
    _viv_asm(COPY, din0, maxDataVer, 16);\n\
    din1 = din0.s10325476;\n\
    _viv_asm(COPY, maxDataVer1, din1, 16);\n\
    VXC_VertMax3_Integer(maxDataVer, maxDataVer1, maxDataVer, maxDataVer, VXC_MODIFIER_BIN(0, 7, 0));\n\
    _viv_asm(COPY, din0, maxDataVer, 16);\n\
    din1 = din0.s02460246;//output\n\
\n\
    // convert to fp32, and then convert it back\n\
    vxc_float4 tmpVal0, tmpVal1, tmpVal2, tmpVal3;\n\
    vxc_int4 tmpOut0, tmpOut1;\n\
\n\
    //convert to fp32 and then convert back\n\
    VXC_DP4x4(tmpVal0, din1, din1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertDirInt16Fp32_4x4);\n\
    VXC_DP4x4(tmpVal2, din1, din1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertEndInt16Fp32_4x4);\n\
    tmpVal1 = tmpVal0 * scaleSF;\n\
    tmpOut0 = convert_int4_rte(tmpVal1);\n\
    tmpVal3 = tmpVal2 * scaleSF;\n\
    tmpOut1 = convert_int4_rte(tmpVal3);\n\
    VXC_DP2x8(din1, tmpOut0, tmpOut1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertInt32toInt16_2x8);\n\
\n\
    //get axis\n\
    VXC_Clamp(din0EqualTmp, din0Fp16, maxDataVer, maxDataVer, VXC_MODIFIER_CLAMP(0, 7, 0, 1));\n\
    VXC_Clamp(din1EqualTmp, din1Fp16, maxDataVer, maxDataVer, VXC_MODIFIER_CLAMP(0, 7, 0, 1));\n\
    bitExtractCoeff = (int4)(0x30201000, 0x70605040, 0x01010101, 0x01010101);\n\
    VXC_BitExtract(din0Equal, din0EqualTmp, din0EqualTmp, bitExtractCoeff, VXC_MODIFIER_BIN(0, 7, 0));\n\
    VXC_BitExtract(din1Equal, din1EqualTmp, din1EqualTmp, bitExtractCoeff, VXC_MODIFIER_BIN(0, 7, 0));\n\
    VXC_DP4x4(axisEncode, din0Equal, din1Equal, VXC_MODIFIER_BIN(0, 3, 0), poolingEncode2);\n\
    axisOut = clz(axisEncode);//output\n\
\n\
    axisVal = convert_short4(axisOut);\n\
    //write data out\n\
    VXC_WriteImage2DArray(tensorOut, coordOut, din1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    VXC_WriteImage2DArray(axis, coordOut, axisVal, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
}"; /* end of vsi_nn_kernel_poolwithargmax_i16_vx*/

static const char vsi_nn_kernel_poolwithargmax_i8_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform VXC_512Bits uniConvertInt8FstFp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertInt8SecFp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertInt32toInt8_2x8;\n\
_viv_uniform VXC_512Bits poolingEncodeInt8_0;\n\
_viv_uniform VXC_512Bits poolingEncodeInt8_1;\n\
_viv_uniform float scaleSF_i8;\n\
_viv_uniform float inputfl_scale_i8;\n\
\n\
//-------------------max pooling with argmax int8---------------\n\
__kernel void poolingWithArgmaxInt8\n\
    (\n\
    image2d_array_t tensorIn,\n\
    image2d_array_t tensorOut,\n\
    image2d_array_t axis,\n\
    int type,\n\
    int sizeX,\n\
    int sizeY,\n\
    int paddingX,\n\
    int paddingY\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    vxc_char16 din0, din1;\n\
    vxc_char16 maxDataVer, maxDataVer1;\n\
    int4 bitExtractCoeff;\n\
    vxc_char16 din0EqualTmp, din1EqualTmp;\n\
    vxc_uchar8 axisEncode;\n\
    vxc_uchar8 axisOut;\n\
\n\
    VXC_ReadImage2DArray(din0, tensorIn, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(din1, tensorIn, coord, VXC_5BITOFFSET_XY(0, 1),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    int4 coordOut = (int4)(coord.x >> 1, coord.y >> 1, coord.z, 0);\n\
    VXC_VertMax3_Integer(maxDataVer, din0, din1, din1, VXC_MODIFIER_BIN(0, 15, 0));\n\
    maxDataVer1 = maxDataVer.s1032547698badcfe;\n\
    VXC_VertMax3_Integer(maxDataVer, maxDataVer1, maxDataVer, maxDataVer, VXC_MODIFIER_BIN(0, 15, 0));\n\
    maxDataVer1 = maxDataVer.s02468ace02468ace;//output\n\
    VXC_WriteImage2DArray(tensorOut, coordOut, maxDataVer1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    //get axis\n\
    VXC_Clamp(din0EqualTmp, din0, maxDataVer, maxDataVer, VXC_MODIFIER_CLAMP(0, 15, 0, 1));\n\
    VXC_Clamp(din1EqualTmp, din1, maxDataVer, maxDataVer, VXC_MODIFIER_CLAMP(0, 15, 0, 1));\n\
    din0EqualTmp &= (vxc_char16)(1);\n\
    din1EqualTmp &= (vxc_char16)(1);\n\
    VXC_DP4x4(axisEncode, din0EqualTmp, din1EqualTmp, VXC_MODIFIER_BIN(0, 3, 0), poolingEncodeInt8_0);\n\
    VXC_DP4x4(axisEncode, din0EqualTmp, din1EqualTmp, VXC_MODIFIER_BIN(4, 7, 0), poolingEncodeInt8_1);\n\
    axisOut = clz(axisEncode);//output\n\
    //write data out\n\
    VXC_WriteImage2DArray(axis, coordOut, axisOut, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void poolingWithArgmaxInt8_Int8\n\
    (\n\
        image2d_array_t tensorIn,\n\
        image2d_array_t tensorOut,\n\
        image2d_array_t axis,\n\
        int type,\n\
        int sizeX,\n\
        int sizeY,\n\
        int paddingX,\n\
        int paddingY\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    vxc_char16 din0, din1;\n\
    vxc_char16 maxDataVer, maxDataVer1;\n\
    vxc_char16 din0EqualTmp, din1EqualTmp;\n\
    vxc_uchar8 axisEncode;\n\
    vxc_uchar8 axisOut;\n\
\n\
    VXC_ReadImage2DArray(din0, tensorIn, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(din1, tensorIn, coord, VXC_5BITOFFSET_XY(0, 1),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    int4 coordOut = (int4)(coord.x >> 1, coord.y >> 1, coord.z, 0);\n\
    VXC_VertMax3_Integer(maxDataVer, din0, din1, din1, VXC_MODIFIER_BIN(0, 15, 0));\n\
    maxDataVer1 = maxDataVer.s1032547698badcfe;\n\
    VXC_VertMax3_Integer(maxDataVer, maxDataVer1, maxDataVer, maxDataVer, VXC_MODIFIER_BIN(0, 15, 0));\n\
    maxDataVer1 = maxDataVer.s02468ace02468ace;//output\n\
\n\
    vxc_float4 tmpVal0, tmpVal1;\n\
    vxc_int4 tmpData0, tmpData1;\n\
    VXC_DP4x4(tmpVal0, maxDataVer1, maxDataVer1, VXC_MODIFIER_BIN(0, 3, 0), uniConvertInt8FstFp32_4x4);\n\
    VXC_DP4x4(tmpVal1, maxDataVer1, maxDataVer1, VXC_MODIFIER_BIN(0, 3, 0), uniConvertInt8SecFp32_4x4);\n\
    tmpVal0 *= scaleSF_i8;\n\
    tmpVal1 *= scaleSF_i8;\n\
    tmpData0 = convert_int4_rte(tmpVal0);\n\
    tmpData1 = convert_int4_rte(tmpVal1);\n\
    VXC_DP2x8(maxDataVer1, tmpData0, tmpData1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\n\
        uniConvertInt32toInt8_2x8);\n\
\n\
    //get axis\n\
    VXC_Clamp(din0EqualTmp, din0, maxDataVer, maxDataVer, VXC_MODIFIER_CLAMP(0, 15, 0, 1));\n\
    VXC_Clamp(din1EqualTmp, din1, maxDataVer, maxDataVer, VXC_MODIFIER_CLAMP(0, 15, 0, 1));\n\
    din0EqualTmp &= (vxc_char16)(1);\n\
    din1EqualTmp &= (vxc_char16)(1);\n\
    VXC_DP4x4(axisEncode, din0EqualTmp, din1EqualTmp, VXC_MODIFIER_BIN(0, 3, 0), poolingEncodeInt8_0);\n\
    VXC_DP4x4(axisEncode, din0EqualTmp, din1EqualTmp, VXC_MODIFIER_BIN(4, 7, 0), poolingEncodeInt8_1);\n\
    axisOut = clz(axisEncode);//output\n\
    //write data out\n\
    VXC_WriteImage2DArray(tensorOut, coordOut, maxDataVer1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_WriteImage2DArray(axis, coordOut, axisOut, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform VXC_512Bits uniPoolConvertInt8toFp32_4x4;\n\
_viv_uniform VXC_512Bits uniPoolConvertInt8toFp32_4x4_2;\n\
_viv_uniform VXC_512Bits uniPoolQuantInt8_2x8;\n\
\n\
__kernel void poolingWithArgmaxInt8_fp16_s2k2p0\n\
    (\n\
        image2d_array_t tensorIn,\n\
        image2d_array_t tensorOut,\n\
        image2d_array_t axis,\n\
        int type,\n\
        int sizeX,\n\
        int sizeY,\n\
        int paddingX,\n\
        int paddingY\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    vxc_char16 din0, din1;\n\
    vxc_char16 maxDataVer, maxDataVer1;\n\
    vxc_char16 din0EqualTmp, din1EqualTmp;\n\
    vxc_uchar8 axisEncode;\n\
    vxc_uchar8 axisOut;\n\
\n\
    VXC_ReadImage2DArray(din0, tensorIn, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(din1, tensorIn, coord, VXC_5BITOFFSET_XY(0, 1),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    int4 coordOut = (int4)(coord.x >> 1, coord.y >> 1, coord.z, 0);\n\
    VXC_VertMax3_Integer(maxDataVer, din0, din1, din1, VXC_MODIFIER_BIN(0, 15, 0));\n\
    maxDataVer1 = maxDataVer.s1032547698badcfe;\n\
    VXC_VertMax3_Integer(maxDataVer, maxDataVer1, maxDataVer, maxDataVer, VXC_MODIFIER_BIN(0, 15, 0));\n\
    //maxDataVer1 = maxDataVer.s02468ace02468ace;//output\n\
\n\
    half tmpScale;\n\
    _viv_asm(CONV, tmpScale, inputfl_scale_i8);\n\
    float4 tmpVal0, tmpVal1;\n\
    half4 tmpData0, tmpData1;\n\
    vxc_short8 result;\n\
    vxc_half8 tmpOut;\n\
    VXC_DP2x8(tmpOut, maxDataVer, tmpScale, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
        uniPoolQuantInt8_2x8);\n\
    _viv_asm(COPY, result, tmpOut, 16);\n\
    VXC_WriteImage2DArray(tensorOut, coordOut, result, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    //get axis\n\
    VXC_Clamp(din0EqualTmp, din0, maxDataVer, maxDataVer, VXC_MODIFIER_CLAMP(0, 15, 0, 1));\n\
    VXC_Clamp(din1EqualTmp, din1, maxDataVer, maxDataVer, VXC_MODIFIER_CLAMP(0, 15, 0, 1));\n\
    din0EqualTmp &= (vxc_char16)(1);\n\
    din1EqualTmp &= (vxc_char16)(1);\n\
    VXC_DP4x4(axisEncode, din0EqualTmp, din1EqualTmp, VXC_MODIFIER_BIN(0, 3, 0), poolingEncodeInt8_0);\n\
    VXC_DP4x4(axisEncode, din0EqualTmp, din1EqualTmp, VXC_MODIFIER_BIN(4, 7, 0), poolingEncodeInt8_1);\n\
    axisOut = clz(axisEncode);//output\n\
    //write data out\n\
    VXC_WriteImage2DArray(axis, coordOut, axisOut, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}"; /* end of vsi_nn_kernel_poolwithargmax_i8_vx*/

static const char vsi_nn_kernel_poolwithargmax_opt_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform VXC_512Bits poolingEncodeInt8_0_opt;\n\
_viv_uniform VXC_512Bits poolingEncodeInt8_1_opt;\n\
_viv_uniform VXC_512Bits poolingEncode_opt;\n\
_viv_uniform VXC_512Bits uniQuantInOutInt16Even_4x4;\n\
_viv_uniform VXC_512Bits uniQuantInOutInt8Even_2x8;\n\
\n\
__kernel void poolingWithArgmaxInt8_Int8_opt\n\
    (\n\
        image2d_array_t tensorIn,\n\
        image2d_array_t tensorOut,\n\
        image2d_array_t axis,\n\
        int type,\n\
        int sizeX,\n\
        int sizeY,\n\
        int paddingX,\n\
        int paddingY\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    vxc_char16 din0, din1;\n\
    vxc_char16 maxDataVer, maxDataVer1;\n\
    vxc_char16 din0EqualTmp, din1EqualTmp;\n\
    vxc_uchar8 axisEncode;\n\
    vxc_uchar8 axisOut;\n\
\n\
    VXC_ReadImage2DArray(din0, tensorIn, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(din1, tensorIn, coord, VXC_5BITOFFSET_XY(0, 1),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    int4 coordOut = (int4)(coord.x >> 1, coord.y >> 1, coord.z, 0);\n\
    VXC_VertMax3_Integer(maxDataVer, din0, din1, din1, VXC_MODIFIER_BIN(0, 15, 0));\n\
    maxDataVer1 = maxDataVer.s1032547698badcfe;\n\
    VXC_VertMax3_Integer(maxDataVer, maxDataVer1, maxDataVer, maxDataVer, VXC_MODIFIER_BIN(0, 15, 0));\n\
    //maxDataVer1 = maxDataVer.s02468ace02468ace;//output\n\
    VXC_DP2x8(maxDataVer1, maxDataVer, maxDataVer, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\n\
        uniQuantInOutInt8Even_2x8);\n\
    VXC_WriteImage2DArray(tensorOut, coordOut, maxDataVer1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    //get axis\n\
    VXC_Clamp(din0EqualTmp, din0, maxDataVer, maxDataVer, VXC_MODIFIER_CLAMP(0, 15, 0, 1));\n\
    VXC_Clamp(din1EqualTmp, din1, maxDataVer, maxDataVer, VXC_MODIFIER_CLAMP(0, 15, 0, 1));\n\
    din0EqualTmp &= (vxc_char16)(1);\n\
    din1EqualTmp &= (vxc_char16)(1);\n\
    VXC_DP4x4(axisEncode, din0EqualTmp, din1EqualTmp, VXC_MODIFIER_BIN(0, 3, 0), poolingEncodeInt8_0_opt);\n\
    VXC_DP4x4(axisEncode, din0EqualTmp, din1EqualTmp, VXC_MODIFIER_BIN(4, 7, 0), poolingEncodeInt8_1_opt);\n\
    axisOut = clz(axisEncode);//output\n\
    //write data out\n\
    VXC_WriteImage2DArray(axis, coordOut, axisOut, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void poolingWithArgmaxInt16_s2k2p0_opt\n\
    (\n\
    image2d_array_t tensorIn,\n\
    image2d_array_t tensorOut,\n\
    image2d_array_t axis,\n\
    int type,\n\
    int sizeX,\n\
    int sizeY,\n\
    int paddingX,\n\
    int paddingY\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    vxc_short8 din0, din1;\n\
    vxc_short8 din0Fp16, din1Fp16;\n\
    vxc_short8 maxDataVer, maxDataVer1;\n\
    int4 bitExtractCoeff;\n\
    vxc_short8 din0EqualTmp, din1EqualTmp;\n\
    vxc_uchar8 din0Equal, din1Equal;\n\
    vxc_uchar4 axisEncode;\n\
    vxc_uchar4 axisOut;\n\
\n\
    VXC_ReadImage2DArray(din0, tensorIn, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(din1, tensorIn, coord, VXC_5BITOFFSET_XY(0, 1),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    int4 coordOut = (int4)(coord.x >> 1, coord.y >> 1, coord.z, 0);\n\
    _viv_asm(COPY, din0Fp16, din0, 16);\n\
    _viv_asm(COPY, din1Fp16, din1, 16);\n\
    VXC_VertMax3_Integer(maxDataVer, din0Fp16, din1Fp16, din1Fp16, VXC_MODIFIER_BIN(0, 7, 0));\n\
    _viv_asm(COPY, din0, maxDataVer, 16);\n\
    din1 = din0.s10325476;\n\
    _viv_asm(COPY, maxDataVer1, din1, 16);\n\
    VXC_VertMax3_Integer(maxDataVer, maxDataVer1, maxDataVer, maxDataVer, VXC_MODIFIER_BIN(0, 7, 0));\n\
    _viv_asm(COPY, din0, maxDataVer, 16);\n\
    //din1 = din0.s02460246;//output\n\
    VXC_DP4x4(din1, din0, din0, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniQuantInOutInt16Even_4x4);\n\
    VXC_WriteImage2DArray(tensorOut, coordOut, din1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
\n\
    //get axis\n\
    VXC_Clamp(din0EqualTmp, din0Fp16, maxDataVer, maxDataVer, VXC_MODIFIER_CLAMP(0, 7, 0, 1));\n\
    VXC_Clamp(din1EqualTmp, din1Fp16, maxDataVer, maxDataVer, VXC_MODIFIER_CLAMP(0, 7, 0, 1));\n\
    bitExtractCoeff = (int4)(0x30201000, 0x70605040, 0x01010101, 0x01010101);\n\
    VXC_BitExtract(din0Equal, din0EqualTmp, din0EqualTmp, bitExtractCoeff, VXC_MODIFIER_BIN(0, 7, 0));\n\
    VXC_BitExtract(din1Equal, din1EqualTmp, din1EqualTmp, bitExtractCoeff, VXC_MODIFIER_BIN(0, 7, 0));\n\
    VXC_DP4x4(axisEncode, din0Equal, din1Equal, VXC_MODIFIER_BIN(0, 3, 0), poolingEncode_opt);\n\
    axisOut = clz(axisEncode);//output\n\
\n\
    //write data out\n\
    VXC_WriteImage2DArray(axis, coordOut, axisOut, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
"; /* end of vsi_nn_kernel_poolwithargmax_opt_vx*/

static const char vsi_nn_kernel_poolwithargmax_u8_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform int input_ZP;\n\
_viv_uniform VXC_512Bits uniU8EvenBinSubZP_MulM_2x8;\n\
_viv_uniform VXC_512Bits uniEncodeUint8_4x8;\n\
_viv_uniform VXC_512Bits uniS16AddOutZP_2x8;\n\
_viv_uniform vxc_uint4 packed_outputZP;\n\
\n\
__kernel void poolingWithArgmaxU8_s2k2p0_2D\n\
    (\n\
    image2d_array_t tensorIn,\n\
    image2d_array_t tensorOut,\n\
    image2d_array_t axis,\n\
    int type,\n\
    int sizeX,\n\
    int sizeY,\n\
    int paddingX,\n\
    int paddingY\n\
    )\n\
{\n\
    int2 coord = (int2)(get_global_id(0), get_global_id(1));\n\
    vxc_uchar16 din0, din1;\n\
    vxc_uchar16 maxDataVer, maxDataVer1;\n\
    vxc_uchar16 din0EqualTmp, din1EqualTmp;\n\
    vxc_uchar8 axisEncode;\n\
    vxc_uchar8 axisOut;\n\
\n\
    VXC_ReadImage(din0, tensorIn, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(din1, tensorIn, coord, VXC_5BITOFFSET_XY(0, 1),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    int2 coordOut = coord >> 1;\n\
    maxDataVer  = max(din0, din1);\n\
    maxDataVer1 = maxDataVer.s1032547698badcfe;\n\
    maxDataVer  = max(maxDataVer1, maxDataVer);\n\
\n\
    vxc_short8 tmp;\n\
    uchar zp = input_ZP;\n\
    VXC_DP2x8(tmp, maxDataVer, zp, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniU8EvenBinSubZP_MulM_2x8);\n\
    vxc_uchar16 packed_outZP;\n\
    _viv_asm(COPY, packed_outZP, packed_outputZP, 16);\n\
    VXC_DP2x8(maxDataVer1, tmp, packed_outZP, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniS16AddOutZP_2x8);\n\
    VXC_WriteImage(tensorOut, coordOut, maxDataVer1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    //get axis\n\
    VXC_Clamp(din0EqualTmp, din0, maxDataVer, maxDataVer, VXC_MODIFIER_CLAMP(0, 15, 0, 1));\n\
    VXC_Clamp(din1EqualTmp, din1, maxDataVer, maxDataVer, VXC_MODIFIER_CLAMP(0, 15, 0, 1));\n\
    din0EqualTmp &= (vxc_uchar16)(1);\n\
    din1EqualTmp &= (vxc_uchar16)(1);\n\
\n\
    VXC_DP4x8(axisEncode, din0EqualTmp, din1EqualTmp, VXC_MODIFIER_BIN(0, 7, 0), uniEncodeUint8_4x8);\n\
    axisOut = clz(axisEncode);//output\n\
    //write data out\n\
    VXC_WriteImage(axis, coordOut, axisOut, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
\n\
"; /* end of vsi_nn_kernel_poolwithargmax_u8_vx*/

static const char vsi_nn_kernel_pow_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform VXC_512Bits uniConvertDirInt16Fp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertEndInt16Fp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertInt32toUint8_2x8;\n\
\n\
_viv_uniform VXC_512Bits uniConvertDirUint8Fp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertEndUint8Fp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertFstFp16Fp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertSecFp16Fp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertInt8FstFp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertInt8SecFp32_4x4;\n\
\n\
_viv_uniform VXC_512Bits uniConvertUint8SubZpToFp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertSecUint8SubZpToFp32_4x4;\n\
\n\
_viv_uniform int input_ZP0;\n\
_viv_uniform float inputScale0;\n\
\n\
__kernel __attribute__((reqd_work_group_size(8, 1, 1))) void vxcTensorPow_Fp16(\n\
    image2d_array_t input0,\n\
    image2d_array_t input1,\n\
    image2d_array_t output)\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_short8 src0, src1;\n\
    vxc_short8 dst;\n\
    vxc_half8 data0, data1;\n\
    VXC_ReadImage2DArray(src0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, data0, src0, 16);\n\
    VXC_ReadImage2DArray(src1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, data1, src1, 16);\n\
    float4 x0, x1;\n\
    float4 y0, y1;\n\
    float4 tmpDst0, tmpDst1;\n\
    VXC_DP4x4(x0, data0, data0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertFstFp16Fp32_4x4);\n\
    VXC_DP4x4(x1, data0, data0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertSecFp16Fp32_4x4);\n\
    VXC_DP4x4(y0, data1, data1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertFstFp16Fp32_4x4);\n\
    VXC_DP4x4(y1, data1, data1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertSecFp16Fp32_4x4);\n\
\n\
    tmpDst0 = pow(x0, y0);\n\
    tmpDst1 = pow(x1, y1);\n\
\n\
    half4 tmpVal0, tmpVal1;\n\
    _viv_asm(CONV, tmpVal0, tmpDst0);\n\
    _viv_asm(CONV, tmpVal1, tmpDst1);\n\
    VXC_DP2x8(dst, tmpVal0, tmpVal1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvertInt32toUint8_2x8);\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel __attribute__((reqd_work_group_size(8, 1, 1))) void vxcTensorPow_Int16(\n\
    image2d_array_t input0,\n\
    image2d_array_t input1,\n\
    image2d_array_t output)\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_short8 src0, src1;\n\
    vxc_short8 dst;\n\
    VXC_ReadImage2DArray(src0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    float4 x0, x1;\n\
    float4 y0, y1;\n\
    float4 tmpDst0, tmpDst1;\n\
    VXC_DP4x4(x0, src0, src0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertDirInt16Fp32_4x4);\n\
    VXC_DP4x4(x1, src0, src0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertEndInt16Fp32_4x4);\n\
    VXC_DP4x4(y0, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertDirInt16Fp32_4x4);\n\
    VXC_DP4x4(y1, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertEndInt16Fp32_4x4);\n\
\n\
    tmpDst0 = pow(x0, y0);\n\
    tmpDst1 = pow(x1, y1);\n\
    vxc_int4 tmpVal0, tmpVal1;\n\
    tmpVal0 = convert_int4_rte(tmpDst0);\n\
    tmpVal1 = convert_int4_rte(tmpDst1);\n\
    VXC_DP2x8(dst, tmpVal0, tmpVal1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1), uniConvertInt32toUint8_2x8);\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel __attribute__((reqd_work_group_size(8, 1, 1))) void vxcTensorPow_Uint8(\n\
    image2d_array_t input0,\n\
    image2d_array_t input1,\n\
    image2d_array_t output)\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_uchar8 src0, src1;\n\
    vxc_uchar8 dst;\n\
    VXC_ReadImage2DArray(src0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    float4 x0, x1;\n\
    float4 y0, y1;\n\
    float4 tmpDst0, tmpDst1;\n\
    VXC_DP4x4(x0, src0, src0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertDirUint8Fp32_4x4);\n\
    VXC_DP4x4(x1, src0, src0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertEndUint8Fp32_4x4);\n\
    VXC_DP4x4(y0, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertDirUint8Fp32_4x4);\n\
    VXC_DP4x4(y1, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertEndUint8Fp32_4x4);\n\
\n\
    tmpDst0 = pow(x0, y0);\n\
    tmpDst1 = pow(x1, y1);\n\
    vxc_int4 tmpVal0, tmpVal1;\n\
    tmpVal0 = convert_int4_rte(tmpDst0);\n\
    tmpVal1 = convert_int4_rte(tmpDst1);\n\
    VXC_DP2x8(dst, tmpVal0, tmpVal1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1), uniConvertInt32toUint8_2x8);\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel __attribute__((reqd_work_group_size(8, 1, 1))) void vxcTensorPow_Int8(\n\
    image2d_array_t input0,\n\
    image2d_array_t input1,\n\
    image2d_array_t output)\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_char8 src0, src1;\n\
    vxc_char8 dst;\n\
    VXC_ReadImage2DArray(src0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    float4 x0, x1;\n\
    float4 y0, y1;\n\
    float4 tmpDst0, tmpDst1;\n\
    VXC_DP4x4(x0, src0, src0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertInt8FstFp32_4x4);\n\
    VXC_DP4x4(x1, src0, src0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertInt8SecFp32_4x4);\n\
    VXC_DP4x4(y0, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertInt8FstFp32_4x4);\n\
    VXC_DP4x4(y1, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertInt8SecFp32_4x4);\n\
\n\
    tmpDst0 = pow(x0, y0);\n\
    tmpDst1 = pow(x1, y1);\n\
    vxc_int4 tmpVal0, tmpVal1;\n\
    tmpVal0 = convert_int4_rte(tmpDst0);\n\
    tmpVal1 = convert_int4_rte(tmpDst1);\n\
    VXC_DP2x8(dst, tmpVal0, tmpVal1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1), uniConvertInt32toUint8_2x8);\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel __attribute__((reqd_work_group_size(8, 1, 1))) void vxcTensorPow_U8Fp16_Fp16(\n\
    image2d_array_t input0,\n\
    image2d_array_t input1,\n\
    image2d_array_t output)\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_uchar8 src0;\n\
    vxc_short8 src1;\n\
    vxc_short8 dst;\n\
    vxc_half8 data1;\n\
    VXC_ReadImage2DArray(src0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, data1, src1, 16);\n\
    float4 x0, x1;\n\
    float4 y0, y1;\n\
    float4 tmpDst0, tmpDst1;\n\
    short zp = input_ZP0;\n\
    VXC_DP4x4(x0, src0, zp, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertUint8SubZpToFp32_4x4);\n\
    VXC_DP4x4(x1, src0, zp, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertSecUint8SubZpToFp32_4x4);\n\
    VXC_DP4x4(y0, data1, data1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertFstFp16Fp32_4x4);\n\
    VXC_DP4x4(y1, data1, data1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertSecFp16Fp32_4x4);\n\
\n\
    x0 *= inputScale0;\n\
    x1 *= inputScale0;\n\
\n\
    tmpDst0 = pow(x0, y0);\n\
    tmpDst1 = pow(x1, y1);\n\
\n\
    half4 tmpVal0, tmpVal1;\n\
    _viv_asm(CONV, tmpVal0, tmpDst0);\n\
    _viv_asm(CONV, tmpVal1, tmpDst1);\n\
    VXC_DP2x8(dst, tmpVal0, tmpVal1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvertInt32toUint8_2x8);\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_pow_vx*/

static const char vsi_nn_kernel_pow_i8_i16_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform VXC_512Bits uniConvertDirInt16Fp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertEndInt16Fp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertInt32toUint8_2x8;\n\
\n\
_viv_uniform VXC_512Bits uniConvertFstFp16Fp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertSecFp16Fp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertInt8FstFp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertInt8SecFp32_4x4;\n\
\n\
\n\
__kernel __attribute__((reqd_work_group_size(8, 1, 1))) void vxcTensorPow_I8Fp16_Fp16(\n\
    image2d_array_t input0,\n\
    image2d_array_t input1,\n\
    image2d_array_t output)\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_uchar8 src0;\n\
    vxc_short8 src1;\n\
    vxc_short8 dst;\n\
    vxc_half8 data1;\n\
    VXC_ReadImage2DArray(src0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, data1, src1, 16);\n\
    float4 x0, x1;\n\
    float4 y0, y1;\n\
    float4 tmpDst0, tmpDst1;\n\
\n\
    VXC_DP4x4(x0, src0, src0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertInt8FstFp32_4x4);\n\
    VXC_DP4x4(x1, src0, src0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertInt8SecFp32_4x4);\n\
    VXC_DP4x4(y0, data1, data1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertFstFp16Fp32_4x4);\n\
    VXC_DP4x4(y1, data1, data1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertSecFp16Fp32_4x4);\n\
\n\
    tmpDst0 = pow(x0, y0);\n\
    tmpDst1 = pow(x1, y1);\n\
\n\
    half4 tmpVal0, tmpVal1;\n\
    _viv_asm(CONV, tmpVal0, tmpDst0);\n\
    _viv_asm(CONV, tmpVal1, tmpDst1);\n\
    VXC_DP2x8(dst, tmpVal0, tmpVal1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvertInt32toUint8_2x8);\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel __attribute__((reqd_work_group_size(8, 1, 1))) void vxcTensorPow_I16Fp16_Fp16(\n\
    image2d_array_t input0,\n\
    image2d_array_t input1,\n\
    image2d_array_t output)\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_uchar8 src0;\n\
    vxc_short8 src1;\n\
    vxc_short8 dst;\n\
    vxc_half8 data1;\n\
    VXC_ReadImage2DArray(src0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, data1, src1, 16);\n\
    float4 x0, x1;\n\
    float4 y0, y1;\n\
    float4 tmpDst0, tmpDst1;\n\
\n\
    VXC_DP4x4(x0, src0, src0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertDirInt16Fp32_4x4);\n\
    VXC_DP4x4(x1, src0, src0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertEndInt16Fp32_4x4);\n\
    VXC_DP4x4(y0, data1, data1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertFstFp16Fp32_4x4);\n\
    VXC_DP4x4(y1, data1, data1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertSecFp16Fp32_4x4);\n\
\n\
    tmpDst0 = pow(x0, y0);\n\
    tmpDst1 = pow(x1, y1);\n\
\n\
    half4 tmpVal0, tmpVal1;\n\
    _viv_asm(CONV, tmpVal0, tmpDst0);\n\
    _viv_asm(CONV, tmpVal1, tmpDst1);\n\
    VXC_DP2x8(dst, tmpVal0, tmpVal1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvertInt32toUint8_2x8);\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
"; /* end of vsi_nn_kernel_pow_i8_i16_vx*/

static const char vsi_nn_kernel_pre_process_bgra_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform VXC_512Bits uniBilinearTmp1Bgra_4x4;\n\
_viv_uniform VXC_512Bits uniBilinearTmp2Bgra_4x4;\n\
_viv_uniform VXC_512Bits uniDescaleU8_4x4;\n\
_viv_uniform VXC_512Bits uniExtractInt32BgraToU8_2x8;\n\
_viv_uniform VXC_512Bits uniExchangeBgra_2x8;\n\
_viv_uniform VXC_512Bits uniExchangeBgra2_2x8;\n\
\n\
_viv_uniform VXC_512Bits uniBilinearTmp1BgraShort_4x4;\n\
_viv_uniform VXC_512Bits uniBilinearTmp2BgraShort_4x4;\n\
_viv_uniform VXC_512Bits uniBilinearTmp3BgraShort_4x4;\n\
_viv_uniform VXC_512Bits uniBilinearTmp4BgraShort_4x4;\n\
_viv_uniform VXC_512Bits uniBilinearTmp5BgraShort_4x4;\n\
_viv_uniform VXC_512Bits uniBilinearTmp6BgraShort_4x4;\n\
_viv_uniform VXC_512Bits uniBilinearTmp7BgraShort_4x4;\n\
_viv_uniform VXC_512Bits uniBilinearTmp8BgraShort_4x4;\n\
\n\
_viv_uniform VXC_512Bits uniExtractBfromBgra_4x4;\n\
_viv_uniform VXC_512Bits uniExtractGfromBgra_4x4;\n\
_viv_uniform VXC_512Bits uniExtractRfromBgra_4x4;\n\
_viv_uniform VXC_512Bits uniConvertInt32toUint8_2x8;\n\
\n\
_viv_uniform int bOrder;\n\
_viv_uniform int rOrder;\n\
_viv_uniform int zp;\n\
_viv_uniform float outputScale;\n\
\n\
__kernel void vxcPre_process_bgra(\n\
    __read_only image2d_array_t input, __write_only image2d_array_t    output,\n\
    global int *xRatio, global int * yRatio, global int * xOffset, global int * yOffset,\n\
    float rMean, float gMean, float bMean, float var, int reverse_channel, int trans)\n\
{\n\
    int4 gidx = get_global_id(0);\n\
    int gidy = get_global_id(1);\n\
    gidx += (int4)(0, 1, 2, 3);\n\
\n\
    int4 fx = (gidx * (*xRatio) + ((*xRatio) >> 1)) - (1 << 14);\n\
    int4 sx = fx & 0xffff8000; // Floor\n\
    int fy, sy;\n\
    fx -= sx;\n\
    sx = sx >> 15;\n\
    fx = (fx +(1 << 4)) >> 5;\n\
\n\
    // for y\n\
    fy = (gidy * (*yRatio) + ((*yRatio) >> 1)) - (1<< 14);\n\
    sy = fy & 0xffff8000; // Floor\n\
    fy -= sy;\n\
    sy = sy >> 15;\n\
\n\
    sy = sy < 0 ? 0 : sy;\n\
    fy = fy < 0 ? 0 : fy;\n\
\n\
    fy = (fy + (1<< 4)) >> 5;\n\
    sx = (sx + (*xOffset)) * 4 ;\n\
    sy += (*yOffset);\n\
    int4 srcPos = (int4)(sx.x, sy, sy + 1, sx.y);\n\
    vxc_uchar16 lineBGRA0, lineBGRA1, lineBGRA2, lineBGRA3;\n\
    vxc_uchar16 dataB, dataG, dataR;\n\
\n\
    VXC_ReadImage(lineBGRA0, input, srcPos.xy, VXC_5BITOFFSET_XY(0, 0),\n\
                     VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(lineBGRA0, input, srcPos.xz, VXC_5BITOFFSET_XY(0, 0),\n\
                     VXC_MODIFIER(8, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_ReadImage(lineBGRA1, input, srcPos.wy, VXC_5BITOFFSET_XY(0, 0),\n\
                     VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(lineBGRA1, input, srcPos.wz, VXC_5BITOFFSET_XY(0, 0),\n\
                     VXC_MODIFIER(8, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    srcPos.x = sx.z;\n\
    srcPos.w = sx.w;\n\
\n\
    VXC_ReadImage(lineBGRA2, input, srcPos.xy, VXC_5BITOFFSET_XY(0, 0),\n\
                     VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(lineBGRA2, input, srcPos.xz, VXC_5BITOFFSET_XY(0, 0),\n\
                     VXC_MODIFIER(8, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_ReadImage(lineBGRA3, input, srcPos.wy, VXC_5BITOFFSET_XY(0, 0),\n\
                     VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(lineBGRA3, input, srcPos.wz, VXC_5BITOFFSET_XY(0, 0),\n\
                     VXC_MODIFIER(8, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    int4 tmp1, tmp2, result1, result2;\n\
    float4 tmpDst;\n\
    float4 mean = (float4)(bMean, gMean, rMean, 0);\n\
    //tmpFx = (int4)(fx.x, fx.x, fx.x, fx.x);\n\
    int tmpV = 1 << 19;\n\
    vxc_short8 tmpFx;\n\
    VXC_DP2x8(tmpFx, fx, fx, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertInt32toUint8_2x8);\n\
    //tmpFx = fx.xxxx;\n\
    VXC_DP4x4(tmp1, lineBGRA0, tmpFx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniBilinearTmp1BgraShort_4x4);\n\
    VXC_DP4x4(tmp2, lineBGRA0, tmpFx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniBilinearTmp2BgraShort_4x4);\n\
    tmp1 = fy * (tmp2 - tmp1) + (tmp1 << 10);\n\
    VXC_DP4x4(tmpDst, tmp1, tmpV, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniDescaleU8_4x4);\n\
    tmpDst = (tmpDst - mean) * var;\n\
    result1 = convert_int4_rte(tmpDst * outputScale + zp);\n\
\n\
    //tmpFx = fx.yyyy;\n\
    VXC_DP4x4(tmp1, lineBGRA1, tmpFx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniBilinearTmp3BgraShort_4x4);\n\
    VXC_DP4x4(tmp2, lineBGRA1, tmpFx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniBilinearTmp4BgraShort_4x4);\n\
    tmp1 = fy * (tmp2 - tmp1) + (tmp1 << 10);\n\
    VXC_DP4x4(tmpDst, tmp1, tmpV, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniDescaleU8_4x4);\n\
    tmpDst = (tmpDst - mean) * var;\n\
    result2 = convert_int4_rte(tmpDst * outputScale + zp);\n\
\n\
    vxc_uchar16 dst, data;\n\
    VXC_DP2x8(dst, result1, result2, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniExtractInt32BgraToU8_2x8);\n\
\n\
    //tmpFx = fx.zzzz;\n\
    VXC_DP4x4(tmp1, lineBGRA2, tmpFx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniBilinearTmp5BgraShort_4x4);\n\
    VXC_DP4x4(tmp2, lineBGRA2, tmpFx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniBilinearTmp6BgraShort_4x4);\n\
    tmp1 = fy * (tmp2 - tmp1) + (tmp1 << 10);\n\
    VXC_DP4x4(tmpDst, tmp1, tmpV, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniDescaleU8_4x4);\n\
    tmpDst = (tmpDst - mean) * var;\n\
    result1 = convert_int4_rte(tmpDst * outputScale + zp);\n\
\n\
    //tmpFx = fx.wwww;\n\
    VXC_DP4x4(tmp1, lineBGRA3, tmpFx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniBilinearTmp7BgraShort_4x4);\n\
    VXC_DP4x4(tmp2, lineBGRA3, tmpFx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniBilinearTmp8BgraShort_4x4);\n\
    tmp1 = fy * (tmp2 - tmp1) + (tmp1 << 10);\n\
    VXC_DP4x4(tmpDst, tmp1, tmpV, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniDescaleU8_4x4);\n\
    tmpDst = (tmpDst - mean) * var;\n\
    result2 = convert_int4_rte(tmpDst * outputScale + zp);\n\
\n\
    VXC_DP2x8(dst, result1, result2, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniExtractInt32BgraToU8_2x8);\n\
\n\
    VXC_DP2x8(data, dst, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniExchangeBgra_2x8);\n\
    VXC_DP2x8(data, dst, dst, VXC_MODIFIER(8, 15, 0, VXC_RM_TowardZero, 0), uniExchangeBgra2_2x8);\n\
\n\
    int4 dstPos = (int4)(get_global_id(0), gidy, 0, 0);\n\
    dstPos.z = bOrder;\n\
    VXC_WriteImage2DArray(output, dstPos, data, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    dstPos.z = 1;\n\
    VXC_WriteImage2DArray(output, dstPos, data, VXC_MODIFIER(4, 7, 0, VXC_RM_TowardZero, 0));\n\
    dstPos.z = rOrder;\n\
    VXC_WriteImage2DArray(output, dstPos, data, VXC_MODIFIER(8, 11, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcPre_process_bgra_copy(\n\
    __read_only image2d_array_t input, __write_only image2d_array_t    output,\n\
    global int *xRatio, global int * yRatio, global int * xOffset, global int * yOffset,\n\
    float rMean, float gMean, float bMean, float var, int reverse_channel, int trans)\n\
{\n\
    int2 pos = (int2)((get_global_id(0) + (*xOffset)) << 2, get_global_id(1) + (*yOffset));\n\
\n\
    vxc_uchar16 lineBGRA0;\n\
    float4 tmpB, tmpG, tmpR;\n\
    float4 tmpDst;\n\
    int4 result1, result2;\n\
    vxc_uchar16 dst;\n\
\n\
    VXC_ReadImage(lineBGRA0, input, pos, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_DP4x4(tmpB, lineBGRA0, lineBGRA0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtractBfromBgra_4x4);\n\
    VXC_DP4x4(tmpG, lineBGRA0, lineBGRA0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtractGfromBgra_4x4);\n\
    VXC_DP4x4(tmpR, lineBGRA0, lineBGRA0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniExtractRfromBgra_4x4);\n\
\n\
    tmpDst = (tmpB - bMean) * var;\n\
    result1 = convert_int4_rte(tmpDst * outputScale + zp);\n\
\n\
    tmpDst = (tmpG - gMean) * var;\n\
    result2 = convert_int4_rte(tmpDst * outputScale + zp);\n\
    VXC_DP2x8(dst, result1, result2, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertInt32toUint8_2x8);\n\
\n\
    int4 dstPos = (int4)(get_global_id(0), get_global_id(1), 0, 0);\n\
    dstPos.z = bOrder;\n\
    VXC_WriteImage2DArray(output, dstPos, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    dstPos.z = 1;\n\
    VXC_WriteImage2DArray(output, dstPos, dst, VXC_MODIFIER(4, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    tmpDst = (tmpR - rMean) * var;\n\
    result1 = convert_int4_rte(tmpDst * outputScale + zp);\n\
    VXC_DP2x8(dst, result1, result1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertInt32toUint8_2x8);\n\
\n\
    dstPos.z = rOrder;\n\
    VXC_WriteImage2DArray(output, dstPos, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_pre_process_bgra_vx*/

static const char vsi_nn_kernel_pre_process_bgra_trans_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform VXC_512Bits uniDescaleU8_4x4;\n\
\n\
_viv_uniform VXC_512Bits uniBilinearTmp1BgraShort_4x4;\n\
_viv_uniform VXC_512Bits uniBilinearTmp2BgraShort_4x4;\n\
_viv_uniform VXC_512Bits uniBilinearTmp3BgraShort_4x4;\n\
_viv_uniform VXC_512Bits uniBilinearTmp4BgraShort_4x4;\n\
_viv_uniform VXC_512Bits uniBilinearTmp5BgraShort_4x4;\n\
_viv_uniform VXC_512Bits uniBilinearTmp6BgraShort_4x4;\n\
_viv_uniform VXC_512Bits uniBilinearTmp7BgraShort_4x4;\n\
_viv_uniform VXC_512Bits uniBilinearTmp8BgraShort_4x4;\n\
\n\
_viv_uniform VXC_512Bits uniConvertInt32toUint8_2x8;\n\
_viv_uniform VXC_512Bits uniExtractInt32BgraToU8Bgr_2x8;\n\
\n\
_viv_uniform int zp;\n\
_viv_uniform float outputScale;\n\
\n\
__kernel void vxcPre_process_bgra_trans(\n\
    __read_only image2d_array_t input, __write_only image2d_array_t    output,\n\
    global int *xRatio, global int * yRatio, global int * xOffset, global int * yOffset,\n\
    float rMean, float gMean, float bMean, float var, int reverse_channel, int trans)\n\
{\n\
    int4 gidx = get_global_id(0);\n\
    int gidy = get_global_id(1);\n\
    gidx += (int4)(0, 1, 2, 3);\n\
\n\
    int4 fx = (gidx * (*xRatio) + ((*xRatio) >> 1)) - (1 << 14);\n\
    int4 sx = fx & 0xffff8000; // Floor\n\
    int fy, sy;\n\
    fx -= sx;\n\
    sx = sx >> 15;\n\
    fx = (fx +(1 << 4)) >> 5;\n\
\n\
    // for y\n\
    fy = (gidy * (*yRatio) + ((*yRatio) >> 1)) - (1<< 14);\n\
    sy = fy & 0xffff8000; // Floor\n\
    fy -= sy;\n\
    sy = sy >> 15;\n\
\n\
    sy = sy < 0 ? 0 : sy;\n\
    fy = fy < 0 ? 0 : fy;\n\
\n\
    fy = (fy + (1<< 4)) >> 5;\n\
    sx = (sx + (*xOffset)) * 4 ;\n\
    sy += (*yOffset);\n\
    int4 srcPos = (int4)(sx.x, sy, sy + 1, sx.y);\n\
    vxc_uchar16 lineBGRA0, lineBGRA1, lineBGRA2, lineBGRA3;\n\
    vxc_uchar16 dataB, dataG, dataR;\n\
\n\
    VXC_ReadImage(lineBGRA0, input, srcPos.xy, VXC_5BITOFFSET_XY(0, 0),\n\
                     VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(lineBGRA0, input, srcPos.xz, VXC_5BITOFFSET_XY(0, 0),\n\
                     VXC_MODIFIER(8, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_ReadImage(lineBGRA1, input, srcPos.wy, VXC_5BITOFFSET_XY(0, 0),\n\
                     VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(lineBGRA1, input, srcPos.wz, VXC_5BITOFFSET_XY(0, 0),\n\
                     VXC_MODIFIER(8, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    srcPos.x = sx.z;\n\
    srcPos.w = sx.w;\n\
\n\
    VXC_ReadImage(lineBGRA2, input, srcPos.xy, VXC_5BITOFFSET_XY(0, 0),\n\
                     VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(lineBGRA2, input, srcPos.xz, VXC_5BITOFFSET_XY(0, 0),\n\
                     VXC_MODIFIER(8, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_ReadImage(lineBGRA3, input, srcPos.wy, VXC_5BITOFFSET_XY(0, 0),\n\
                     VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(lineBGRA3, input, srcPos.wz, VXC_5BITOFFSET_XY(0, 0),\n\
                     VXC_MODIFIER(8, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    int4 tmp1, tmp2, result1, result2;\n\
    float4 tmpDst;\n\
    float4 mean = (float4)(bMean, gMean, rMean, 0);\n\
    //tmpFx = (int4)(fx.x, fx.x, fx.x, fx.x);\n\
    int tmpV = 1 << 19;\n\
    vxc_short8 tmpFx;\n\
    VXC_DP2x8(tmpFx, fx, fx, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\n\
                 uniConvertInt32toUint8_2x8);\n\
    //tmpFx = fx.xxxx;\n\
    VXC_DP4x4(tmp1, lineBGRA0, tmpFx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\n\
                 uniBilinearTmp1BgraShort_4x4);\n\
    VXC_DP4x4(tmp2, lineBGRA0, tmpFx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\n\
                 uniBilinearTmp2BgraShort_4x4);\n\
    tmp1 = fy * (tmp2 - tmp1) + (tmp1 << 10);\n\
    VXC_DP4x4(tmpDst, tmp1, tmpV, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniDescaleU8_4x4);\n\
    tmpDst = (tmpDst - mean) * var;\n\
    result1 = convert_int4_rte(tmpDst * outputScale + zp);\n\
\n\
    //tmpFx = fx.yyyy;\n\
    VXC_DP4x4(tmp1, lineBGRA1, tmpFx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniBilinearTmp3BgraShort_4x4);\n\
    VXC_DP4x4(tmp2, lineBGRA1, tmpFx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniBilinearTmp4BgraShort_4x4);\n\
    tmp1 = fy * (tmp2 - tmp1) + (tmp1 << 10);\n\
    VXC_DP4x4(tmpDst, tmp1, tmpV, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniDescaleU8_4x4);\n\
    tmpDst = (tmpDst - mean) * var;\n\
    result2 = convert_int4_rte(tmpDst * outputScale + zp);\n\
\n\
    vxc_uchar16 dst;\n\
    VXC_DP2x8(dst, result1, result2, VXC_MODIFIER(0, 5, 0, VXC_RM_ToNearestEven, 1),\n\
                 uniExtractInt32BgraToU8Bgr_2x8);\n\
\n\
    //tmpFx = fx.zzzz;\n\
    VXC_DP4x4(tmp1, lineBGRA2, tmpFx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniBilinearTmp5BgraShort_4x4);\n\
    VXC_DP4x4(tmp2, lineBGRA2, tmpFx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniBilinearTmp6BgraShort_4x4);\n\
    tmp1 = fy * (tmp2 - tmp1) + (tmp1 << 10);\n\
    VXC_DP4x4(tmpDst, tmp1, tmpV, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniDescaleU8_4x4);\n\
    tmpDst = (tmpDst - mean) * var;\n\
    result1 = convert_int4_rte(tmpDst * outputScale + zp);\n\
\n\
    //tmpFx = fx.wwww;\n\
    VXC_DP4x4(tmp1, lineBGRA3, tmpFx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniBilinearTmp7BgraShort_4x4);\n\
    VXC_DP4x4(tmp2, lineBGRA3, tmpFx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniBilinearTmp8BgraShort_4x4);\n\
    tmp1 = fy * (tmp2 - tmp1) + (tmp1 << 10);\n\
    VXC_DP4x4(tmpDst, tmp1, tmpV, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniDescaleU8_4x4);\n\
    tmpDst = (tmpDst - mean) * var;\n\
    result2 = convert_int4_rte(tmpDst * outputScale + zp);\n\
\n\
    VXC_DP2x8(dst, result1, result2, VXC_MODIFIER(6, 11, 0, VXC_RM_ToNearestEven, 1),\n\
                 uniExtractInt32BgraToU8Bgr_2x8);\n\
\n\
    int4 dstPos = (int4)(get_global_id(0) * 3, gidy, 0, 0);\n\
    VXC_WriteImage2DArray(output, dstPos, dst, VXC_MODIFIER(0, 11, 0, VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_pre_process_bgra_trans_vx*/

static const char vsi_nn_kernel_pre_process_gray_vx[] = "/*\n\
 ============================================================================\n\
 Name        : GrayScale.vx\n\
 Author      : Sam\n\
 Version     :\n\
 Copyright   : Your copyright notice\n\
 Description :\n\
 ============================================================================\n\
 */\n\
#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform VXC_512Bits uniVecShift10;\n\
_viv_uniform VXC_512Bits uniAddRShift;\n\
_viv_uniform VXC_512Bits uniGetTempVal;\n\
_viv_uniform VXC_512Bits uniExtractBytes;\n\
\n\
_viv_uniform VXC_512Bits uniConvertIntergetoF32_4x4;\n\
_viv_uniform float outputScale;\n\
_viv_uniform VXC_512Bits uniExtactInteger_2x8;\n\
\n\
__kernel void vxGrayScaletoTensor_I8\n\
    (\n\
    __read_only image2d_array_t  input,\n\
    __write_only image2d_array_t output,\n\
        global int               *xRatio,\n\
        global int               *yRatio,\n\
        global int               *xOffset,\n\
        global int               *yOffset,\n\
               float             mean,\n\
               float             f32Var\n\
    )\n\
{\n\
    int2 ratioXY = (int2)(*xRatio, *yRatio);\n\
\n\
    int4 xPos       = get_global_id(0);\n\
    int yPos        = get_global_id(1);\n\
\n\
    int2 ratioSufXY = (ratioXY >> 1) - (1 << 14);\n\
    xPos += (int4)(0, 1, 2, 3);\n\
\n\
    //x\n\
    int4 fx0 = xPos * ratioXY.x + ratioSufXY.x;\n\
    int4 sx = fx0 & 0xffff8000;\n\
    fx0 -= sx;\n\
    sx = sx >> 15;\n\
\n\
    vxc_short4 fx;\n\
    VXC_DP4x4(fx, fx0, 1 << 4, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniAddRShift);\n\
    //y\n\
    int fy = yPos * ratioXY.y + ratioSufXY.y;\n\
    int sy = fy & 0xffff8000; // Floor\n\
\n\
    fy -= sy;\n\
    sy = sy >> 15;\n\
\n\
    fy = (fy + (1<< 4)) >> 5;\n\
\n\
    //R\n\
    vxc_uchar16 line0Y;\n\
    vxc_uchar16 line1Y;\n\
    int4 coord;\n\
    sx = sx + *xOffset;\n\
    coord.xyz    = sx.xyz;\n\
    coord.w        = sy + *yOffset;\n\
    int2 coord1 = (int2)(sx.w, coord.w);\n\
    VXC_ReadImage(line0Y, input, coord.xw, 0, VXC_MODIFIER(0, 1, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line0Y, input, coord.yw, 0, VXC_MODIFIER(2, 3, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line0Y, input, coord.zw, 0, VXC_MODIFIER(4, 5, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line0Y, input, coord1, 0, VXC_MODIFIER(6, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_ReadImage(line1Y, input, coord.xw, VXC_5BITOFFSET_XY(0, 1),\n\
        VXC_MODIFIER(0, 1, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line1Y, input, coord.yw, VXC_5BITOFFSET_XY(0, 1),\n\
        VXC_MODIFIER(2, 3, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line1Y, input, coord.zw, VXC_5BITOFFSET_XY(0, 1),\n\
        VXC_MODIFIER(4, 5, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line1Y, input, coord1, VXC_5BITOFFSET_XY(0, 1),\n\
        VXC_MODIFIER(6, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    float grayMean = mean * f32Var;\n\
\n\
    int4 test01, temp1;\n\
    int4 test02, temp2;\n\
    int4 tt;\n\
    vxc_uchar4 val;\n\
    int2 coord_out = (int2)(xPos.x, yPos);\n\
\n\
    vxc_uchar8 line1, line2;\n\
\n\
    VXC_DP4x4(test01, line0Y, line0Y, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\n\
        uniVecShift10);\n\
    VXC_DP4x4(temp1, line0Y, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\n\
        uniGetTempVal);\n\
    temp1 = temp1 + test01;\n\
\n\
    VXC_DP4x4(test02, line1Y, line1Y, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\n\
        uniVecShift10);\n\
    VXC_DP4x4(temp2, line1Y, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\n\
        uniGetTempVal);\n\
    temp2 = temp2 + test02;\n\
    temp2 = fy * (temp2 - temp1) + (temp1 << 10);\n\
\n\
    vxc_float4 tmp_dst;\n\
    vxc_uchar4 u8_dst;\n\
    VXC_DP4x4(u8_dst, temp2, 1 << 19, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1),\n\
        uniExtractBytes);\n\
    VXC_DP4x4(tmp_dst, u8_dst, u8_dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1),\n\
        uniConvertIntergetoF32_4x4);\n\
\n\
    //convert U8 to dfp8\n\
    int4 dst0;\n\
    vxc_char4 dst;\n\
    tmp_dst = tmp_dst * f32Var - grayMean;\n\
    tmp_dst *= outputScale;\n\
    dst0 = convert_int4_rte(tmp_dst);\n\
    VXC_DP2x8(dst, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1),\n\
        uniExtactInteger_2x8);\n\
\n\
    VXC_WriteImage(output, coord_out, dst,\n\
        VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform VXC_512Bits uniDataMulAlpha_4x4;\n\
_viv_uniform VXC_512Bits uniDataSubMean_4x4;\n\
__kernel void vxGrayScaletoTensor_F16\n\
    (\n\
    __read_only image2d_array_t  input,\n\
    __write_only image2d_array_t output,\n\
        global int               *xRatio,\n\
        global int               *yRatio,\n\
        global int               *xOffset,\n\
        global int               *yOffset,\n\
               float             mean,\n\
               float             f32Var\n\
    )\n\
{\n\
    int2 ratioXY = (int2)(*xRatio, *yRatio);\n\
\n\
    int4 xPos       = get_global_id(0);\n\
    int yPos        = get_global_id(1);\n\
\n\
    int2 ratioSufXY = (ratioXY >> 1) - (1 << 14);\n\
    xPos += (int4)(0, 1, 2, 3);\n\
\n\
    //x\n\
    int4 fx0 = xPos * ratioXY.x + ratioSufXY.x;\n\
    int4 sx = fx0 & 0xffff8000;\n\
    fx0 -= sx;\n\
    sx = sx >> 15;\n\
\n\
    vxc_short4 fx;\n\
    VXC_DP4x4(fx, fx0, 1 << 4, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniAddRShift);\n\
    //y\n\
    int fy = yPos * ratioXY.y + ratioSufXY.y;\n\
    int sy = fy & 0xffff8000; // Floor\n\
\n\
    fy -= sy;\n\
    sy = sy >> 15;\n\
\n\
    fy = (fy + (1<< 4)) >> 5;\n\
\n\
    //R\n\
    vxc_uchar16 line0Y;\n\
    vxc_uchar16 line1Y;\n\
    int4 coord;\n\
    sx = sx + *xOffset;\n\
    coord.xyz    = sx.xyz;\n\
    coord.w        = sy + *yOffset;\n\
    int2 coord1 = (int2)(sx.w, coord.w);\n\
    VXC_ReadImage(line0Y, input, coord.xw, 0, VXC_MODIFIER(0, 1, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line0Y, input, coord.yw, 0, VXC_MODIFIER(2, 3, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line0Y, input, coord.zw, 0, VXC_MODIFIER(4, 5, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line0Y, input, coord1, 0, VXC_MODIFIER(6, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_ReadImage(line1Y, input, coord.xw, VXC_5BITOFFSET_XY(0, 1),\n\
        VXC_MODIFIER(0, 1, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line1Y, input, coord.yw, VXC_5BITOFFSET_XY(0, 1),\n\
        VXC_MODIFIER(2, 3, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line1Y, input, coord.zw, VXC_5BITOFFSET_XY(0, 1),\n\
        VXC_MODIFIER(4, 5, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line1Y, input, coord1, VXC_5BITOFFSET_XY(0, 1),\n\
        VXC_MODIFIER(6, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    float grayMean = mean;\n\
\n\
    int4 test01, temp1;\n\
    int4 test02, temp2;\n\
    int4 tt;\n\
    vxc_uchar4 val;\n\
    int2 coord_out = (int2)(xPos.x, yPos);\n\
\n\
    VXC_DP4x4(test01, line0Y, line0Y, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\n\
        uniVecShift10);\n\
    VXC_DP4x4(temp1, line0Y, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\n\
        uniGetTempVal);\n\
    temp1 = temp1 + test01;\n\
\n\
    VXC_DP4x4(test02, line1Y, line1Y, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\n\
        uniVecShift10);\n\
    VXC_DP4x4(temp2, line1Y, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\n\
        uniGetTempVal);\n\
    temp2 = temp2 + test02;\n\
    temp2 = fy * (temp2 - temp1) + (temp1 << 10);\n\
\n\
    VXC_DP4x4(val, temp2, 1 << 19, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1),\n\
        uniExtractBytes);\n\
\n\
    //convert U8 to FP16\n\
    half f16mean;\n\
    half f16alpha;\n\
    vxc_half4    dst;\n\
    vxc_short4 tmp_dst;\n\
    _viv_asm(CONV, f16mean, grayMean);\n\
    _viv_asm(CONV, f16alpha, f32Var);\n\
    VXC_DP4x4(dst, val, f16mean, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\n\
        uniDataSubMean_4x4);\n\
    VXC_DP4x4(dst, dst, f16alpha, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\n\
        uniDataMulAlpha_4x4);\n\
    _viv_asm(COPY, tmp_dst, dst, 8);\n\
    VXC_WriteImage(output, coord_out, tmp_dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxGrayScaletoTensor_I16\n\
    (\n\
    __read_only image2d_array_t  input,\n\
    __write_only image2d_array_t output,\n\
        global int               *xRatio,\n\
        global int               *yRatio,\n\
        global int               *xOffset,\n\
        global int               *yOffset,\n\
               float             mean,\n\
               float             f32Var\n\
    )\n\
{\n\
    int2 ratioXY = (int2)(*xRatio, *yRatio);\n\
\n\
    int4 xPos        = get_global_id(0);\n\
    int yPos        = get_global_id(1);\n\
\n\
    int2 ratioSufXY = (ratioXY >> 1) - (1 << 14);\n\
    xPos += (int4)(0, 1, 2, 3);\n\
\n\
    //x\n\
    int4 fx0 = xPos * ratioXY.x + ratioSufXY.x;\n\
    int4 sx = fx0 & 0xffff8000;\n\
    fx0 -= sx;\n\
    sx = sx >> 15;\n\
\n\
    vxc_short4 fx;\n\
    VXC_DP4x4(fx, fx0, 1 << 4, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\n\
        uniAddRShift);\n\
    //y\n\
    int fy = yPos * ratioXY.y + ratioSufXY.y;\n\
    int sy = fy & 0xffff8000; // Floor\n\
\n\
    fy -= sy;\n\
    sy = sy >> 15;\n\
\n\
    fy = (fy + (1<< 4)) >> 5;\n\
\n\
    vxc_uchar16 line0Y;\n\
    vxc_uchar16 line1Y;\n\
    int4 coord;\n\
    sx = sx + *xOffset;\n\
    coord.xyz    = sx.xyz;\n\
    coord.w        = sy + *yOffset;\n\
    int2 coord1 = (int2)(sx.w, coord.w);\n\
    VXC_ReadImage(line0Y, input, coord.xw, 0, VXC_MODIFIER(0, 1, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line0Y, input, coord.yw, 0, VXC_MODIFIER(2, 3, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line0Y, input, coord.zw, 0, VXC_MODIFIER(4, 5, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line0Y, input, coord1, 0, VXC_MODIFIER(6, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_ReadImage(line1Y, input, coord.xw, VXC_5BITOFFSET_XY(0, 1),\n\
        VXC_MODIFIER(0, 1, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line1Y, input, coord.yw, VXC_5BITOFFSET_XY(0, 1),\n\
        VXC_MODIFIER(2, 3, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line1Y, input, coord.zw, VXC_5BITOFFSET_XY(0, 1),\n\
        VXC_MODIFIER(4, 5, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line1Y, input, coord1, VXC_5BITOFFSET_XY(0, 1),\n\
        VXC_MODIFIER(6, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    float grayMean = mean * f32Var;\n\
\n\
    int4 test01, temp1;\n\
    int4 test02, temp2;\n\
    int4 tt;\n\
    vxc_uchar4 val;\n\
    int2 coord_out = (int2)(xPos.x, yPos);\n\
\n\
    vxc_uchar8 line1, line2;\n\
\n\
    VXC_DP4x4(test01, line0Y, line0Y, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\n\
        uniVecShift10);\n\
    VXC_DP4x4(temp1, line0Y, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal);\n\
    temp1 = temp1 + test01;\n\
\n\
    VXC_DP4x4(test02, line1Y, line1Y, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\n\
        uniVecShift10);\n\
    VXC_DP4x4(temp2, line1Y, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal);\n\
    temp2 = temp2 + test02;\n\
    temp2 = fy * (temp2 - temp1) + (temp1 << 10);\n\
\n\
    vxc_float4    tmp_dst;\n\
    vxc_uchar4 u8_dst;\n\
    VXC_DP4x4(u8_dst, temp2, 1 << 19, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1),\n\
        uniExtractBytes);\n\
    VXC_DP4x4(tmp_dst, u8_dst, u8_dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1),\n\
        uniConvertIntergetoF32_4x4);\n\
\n\
    //convert U8 to dfp8\n\
    int4 dst0;\n\
    vxc_short4 dst;\n\
    tmp_dst = tmp_dst * f32Var - grayMean;\n\
    tmp_dst *= outputScale;\n\
    dst0 = convert_int4_rte(tmp_dst);\n\
    VXC_DP2x8(dst, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1),\n\
        uniExtactInteger_2x8);\n\
\n\
    VXC_WriteImage(output, coord_out, dst,\n\
        VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform float outputZP;\n\
__kernel void vxGrayScaletoTensor_U8\n\
    (\n\
    __read_only image2d_array_t  input,\n\
    __write_only image2d_array_t output,\n\
        global int               *xRatio,\n\
        global int               *yRatio,\n\
        global int               *xOffset,\n\
        global int               *yOffset,\n\
               float             mean,\n\
               float             f32Var\n\
    )\n\
{\n\
    int2 ratioXY = (int2)(*xRatio, *yRatio);\n\
\n\
    int4 xPos        = get_global_id(0);\n\
    int yPos        = get_global_id(1);\n\
\n\
    int2 ratioSufXY = (ratioXY >> 1) - (1 << 14);\n\
    xPos += (int4)(0, 1, 2, 3);\n\
\n\
    //x\n\
    int4 fx0 = xPos * ratioXY.x + ratioSufXY.x;\n\
    int4 sx = fx0 & 0xffff8000;\n\
    fx0 -= sx;\n\
    sx = sx >> 15;\n\
\n\
    vxc_short4 fx;\n\
    VXC_DP4x4(fx, fx0, 1 << 4, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniAddRShift);\n\
    //y\n\
    int fy = yPos * ratioXY.y + ratioSufXY.y;\n\
    int sy = fy & 0xffff8000; // Floor\n\
\n\
    fy -= sy;\n\
    sy = sy >> 15;\n\
\n\
    fy = (fy + (1<< 4)) >> 5;\n\
\n\
    //R\n\
    vxc_uchar16 line0Y;\n\
    vxc_uchar16 line1Y;\n\
    int4 coord;\n\
    sx = sx + *xOffset;\n\
    coord.xyz    = sx.xyz;\n\
    coord.w        = sy + *yOffset;\n\
    int2 coord1 = (int2)(sx.w, coord.w);\n\
    VXC_ReadImage(line0Y, input, coord.xw, 0, VXC_MODIFIER(0, 1, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line0Y, input, coord.yw, 0, VXC_MODIFIER(2, 3, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line0Y, input, coord.zw, 0, VXC_MODIFIER(4, 5, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line0Y, input, coord1, 0, VXC_MODIFIER(6, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_ReadImage(line1Y, input, coord.xw, VXC_5BITOFFSET_XY(0, 1),\n\
        VXC_MODIFIER(0, 1, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line1Y, input, coord.yw, VXC_5BITOFFSET_XY(0, 1),\n\
        VXC_MODIFIER(2, 3, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line1Y, input, coord.zw, VXC_5BITOFFSET_XY(0, 1),\n\
        VXC_MODIFIER(4, 5, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(line1Y, input, coord1, VXC_5BITOFFSET_XY(0, 1),\n\
        VXC_MODIFIER(6, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    float grayMean = mean * f32Var;\n\
\n\
    int4 test01, temp1;\n\
    int4 test02, temp2;\n\
    int4 tt;\n\
    vxc_uchar4 val;\n\
    int2 coord_out = (int2)(xPos.x, yPos);\n\
\n\
    VXC_DP4x4(test01, line0Y, line0Y, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\n\
        uniVecShift10);\n\
    VXC_DP4x4(temp1, line0Y, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\n\
        uniGetTempVal);\n\
    temp1 = temp1 + test01;\n\
\n\
    VXC_DP4x4(test02, line1Y, line1Y, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\n\
        uniVecShift10);\n\
    VXC_DP4x4(temp2, line1Y, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\n\
        uniGetTempVal);\n\
    temp2 = temp2 + test02;\n\
    temp2 = fy * (temp2 - temp1) + (temp1 << 10);\n\
\n\
    vxc_float4 tmp_dst;\n\
    vxc_uchar4 u8_dst;\n\
    VXC_DP4x4(u8_dst, temp2, 1 << 19, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1),\n\
        uniExtractBytes);\n\
    VXC_DP4x4(tmp_dst, u8_dst, u8_dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1),\n\
        uniConvertIntergetoF32_4x4);\n\
\n\
    //convert U8 to dfp8\n\
    int4 dst0;\n\
    vxc_uchar4 dst;\n\
    tmp_dst = tmp_dst * f32Var - grayMean;\n\
    tmp_dst = tmp_dst * outputScale + outputZP;\n\
    dst0 = convert_int4_rte(tmp_dst);\n\
    VXC_DP2x8(dst, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1),\n\
        uniExtactInteger_2x8);\n\
\n\
    VXC_WriteImage(output, coord_out, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_pre_process_gray_vx*/

static const char vsi_nn_kernel_pre_process_gray_copy_vx[] = "/*\n\
 ============================================================================\n\
 Name        : GrayScale.vx\n\
 Author      : Sam\n\
 Version     :\n\
 Copyright   : Your copyright notice\n\
 Description :\n\
 ============================================================================\n\
 */\n\
#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform float outputScale;\n\
\n\
_viv_uniform VXC_512Bits uniDataMeanStddevLo_2x8;\n\
_viv_uniform VXC_512Bits uniDataMeanStddevHi_2x8;\n\
__kernel void vxGrayScaletoTensor_F16_copy\n\
    (\n\
    __read_only image2d_array_t  input,\n\
    __write_only image2d_array_t output,\n\
        global int               *xRatio,\n\
        global int               *yRatio,\n\
        global int               *xOffset,\n\
        global int               *yOffset,\n\
               float             mean,\n\
               float             f32Var\n\
    )\n\
{\n\
    int4 coord  = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), get_global_id(1));\n\
\n\
    coord.xy += (int2) (*xOffset, *yOffset);\n\
    vxc_uchar16 src0;\n\
    vxc_half8   dst0, dst1;\n\
\n\
    VXC_ReadImage(src0, input, coord.xy, 0, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    coord.x = coord.z + 8;\n\
    float4      paramData = (float4)(mean * f32Var, mean * f32Var, mean * f32Var, f32Var);\n\
    //convert U8 to FP16\n\
    half4 paramData_f16;\n\
    vxc_short8 tmp_dst;\n\
    _viv_asm(CONV, paramData_f16, paramData);\n\
\n\
    VXC_DP2x8(dst0, src0, paramData_f16, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\n\
        uniDataMeanStddevLo_2x8);\n\
    VXC_DP2x8(dst1, src0, paramData_f16, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\n\
        uniDataMeanStddevHi_2x8);\n\
    _viv_asm(COPY, tmp_dst, dst0, 16);\n\
    VXC_WriteImage(output, coord.zw, tmp_dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, tmp_dst, dst1, 16);\n\
    VXC_WriteImage(output, coord.xw, tmp_dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxGrayScaletoTensor_I8_copy\n\
    (\n\
    __read_only image2d_array_t  input,\n\
    __write_only image2d_array_t output,\n\
        global int               *xRatio,\n\
        global int               *yRatio,\n\
        global int               *xOffset,\n\
        global int               *yOffset,\n\
               float             mean,\n\
               float             f32Var\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), get_global_id(1));\n\
\n\
    coord.xy += (int2) (*xOffset, *yOffset);\n\
    vxc_uchar16 src0;\n\
    vxc_char16   dst;\n\
\n\
    VXC_ReadImage(src0, input, coord.xy, 0, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    f32Var *= outputScale;\n\
    float4      paramData = (float4)(mean * f32Var, mean * f32Var, mean * f32Var, f32Var);\n\
    //convert U8 to FP16\n\
    half4 paramData_f16;\n\
    _viv_asm(CONV, paramData_f16, paramData);\n\
\n\
    VXC_DP2x8(dst, src0, paramData_f16, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\n\
        uniDataMeanStddevLo_2x8);\n\
    VXC_DP2x8(dst, src0, paramData_f16, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1),\n\
        uniDataMeanStddevHi_2x8);\n\
    VXC_WriteImage(output, coord.zw, dst, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
}\n\
\n\
__kernel void vxGrayScaletoTensor_I16_copy\n\
    (\n\
    __read_only image2d_array_t  input,\n\
    __write_only image2d_array_t output,\n\
        global int               *xRatio,\n\
        global int               *yRatio,\n\
        global int               *xOffset,\n\
        global int               *yOffset,\n\
               float             mean,\n\
               float             f32Var\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), get_global_id(1));\n\
\n\
    coord.xy += (int2) (*xOffset, *yOffset);\n\
    vxc_uchar16 src0;\n\
    vxc_short8  dst0, dst1;\n\
\n\
    VXC_ReadImage(src0, input, coord.xy, 0, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    coord.x = coord.z + 8;\n\
\n\
    f32Var *= outputScale;\n\
    float4 paramData = (float4)(mean * f32Var, mean * f32Var, mean * f32Var, f32Var);\n\
    //convert U8 to FP16\n\
    half4 paramData_f16;\n\
    _viv_asm(CONV, paramData_f16, paramData);\n\
\n\
\n\
    VXC_DP2x8(dst0, src0, paramData_f16, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\n\
        uniDataMeanStddevLo_2x8);\n\
    VXC_DP2x8(dst1, src0, paramData_f16, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\n\
        uniDataMeanStddevHi_2x8);\n\
    VXC_WriteImage(output, coord.zw, dst0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_WriteImage(output, coord.xw, dst1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform float outputZP;\n\
__kernel void vxGrayScaletoTensor_U8_copy\n\
    (\n\
    __read_only image2d_array_t  input,\n\
    __write_only image2d_array_t output,\n\
        global int               *xRatio,\n\
        global int               *yRatio,\n\
        global int               *xOffset,\n\
        global int               *yOffset,\n\
               float             mean,\n\
               float             f32Var\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), get_global_id(1));\n\
\n\
    coord.xy += (int2) (*xOffset, *yOffset);\n\
    vxc_uchar16 src0;\n\
    vxc_uchar16 dst;\n\
\n\
    VXC_ReadImage(src0, input, coord.xy, 0, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    f32Var *= outputScale;\n\
    float4  paramData = (float4)(mean * f32Var - outputZP, mean * f32Var - outputZP,\n\
        mean * f32Var - outputZP, f32Var);\n\
    //convert U8 to FP16\n\
    half4 paramData_f16;\n\
    _viv_asm(CONV, paramData_f16, paramData);\n\
\n\
    VXC_DP2x8(dst, src0, paramData_f16, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\n\
        uniDataMeanStddevLo_2x8);\n\
    VXC_DP2x8(dst, src0, paramData_f16, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1),\n\
        uniDataMeanStddevHi_2x8);\n\
    VXC_WriteImage(output, coord.zw, dst, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_pre_process_gray_copy_vx*/

static const char vsi_nn_kernel_pre_process_rgb_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform VXC_512Bits uniVecShift10;\n\
_viv_uniform VXC_512Bits uniAddRShift;\n\
_viv_uniform VXC_512Bits uniGetTempVal;\n\
_viv_uniform VXC_512Bits uniExtractBytes;\n\
_viv_uniform VXC_512Bits uniUnpackToR;\n\
_viv_uniform VXC_512Bits uniUnpackToG;\n\
_viv_uniform VXC_512Bits uniUnpackToB;\n\
\n\
_viv_uniform VXC_512Bits uniConvertIntergetoF32_4x4;\n\
_viv_uniform float outputScale;\n\
_viv_uniform VXC_512Bits uniExtract8Data_2x8;\n\
_viv_uniform float outputZP;\n\
_viv_uniform int r_order;\n\
_viv_uniform int b_order;\n\
\n\
#define DESCALE(x) (((x) + (1<<19)) >> 20)\n\
\n\
#define IMAGE_PRE_PROCESS(dst_name, conv_type, dst_type, copy_type) \\\n\
__kernel void vxRGBScaletoTensor_##dst_name \\\n\
    ( \\\n\
__read_only image2d_array_t  input, \\\n\
__write_only image2d_array_t output, \\\n\
        global int           *xRatio, \\\n\
        global int           *yRatio, \\\n\
        global int           *xOffset, \\\n\
        global int           *yOffset, \\\n\
               float         rMean, \\\n\
               float         gMean, \\\n\
               float         bMean, \\\n\
               float         f32Var, \\\n\
               int           reverse_channel, \\\n\
               int           trans \\\n\
    ) \\\n\
{ \\\n\
    int2 ratioXY = (int2)(*xRatio, *yRatio); \\\n\
    int4 xPos       = get_global_id(0); \\\n\
    int yPos        = get_global_id(1); \\\n\
    int2 ratioSufXY = (ratioXY >> 1) - (1 << 14); \\\n\
    xPos += (int4)(0, 1, 2, 3); \\\n\
 \\\n\
    /*x*/ \\\n\
    int4 fx0 = xPos * ratioXY.x + ratioSufXY.x; \\\n\
    int4 sx = fx0 & 0xffff8000; \\\n\
    fx0 -= sx; \\\n\
    sx = sx >> 15; \\\n\
 \\\n\
    vxc_short4 fx; \\\n\
    VXC_DP4x4(fx, fx0, 1 << 4, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniAddRShift); \\\n\
    /*y*/ \\\n\
    int fy = yPos * ratioXY.y + ratioSufXY.y; \\\n\
    int sy = fy & 0xffff8000; \\\n\
 \\\n\
    fy -= sy; \\\n\
    sy = sy >> 15; \\\n\
 \\\n\
    fy = (fy + (1<< 4)) >> 5; \\\n\
 \\\n\
    vxc_uchar16 line0RGB1, line0RGB2; \\\n\
    vxc_uchar16 line1RGB3, line1RGB4; \\\n\
    int4 coord; \\\n\
    sx = (sx + (*xOffset)) * 3; \\\n\
    coord.xyz    = sx.xyz; \\\n\
    coord.w        = sy + *yOffset; \\\n\
    int2 coord1 = (int2)(sx.w, coord.w); \\\n\
    VXC_ReadImage(line0RGB1, input, coord.xw, 0, VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(line0RGB1, input, coord.yw, 0, VXC_MODIFIER(6, 11, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(line0RGB2, input, coord.zw, 0, VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(line0RGB2, input, coord1, 0, VXC_MODIFIER(6, 11, 0, VXC_RM_TowardZero, 0)); \\\n\
 \\\n\
    VXC_ReadImage(line1RGB3, input, coord.xw, VXC_5BITOFFSET_XY(0, 1), \\\n\
        VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(line1RGB3, input, coord.yw, VXC_5BITOFFSET_XY(0, 1), \\\n\
        VXC_MODIFIER(6, 11, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(line1RGB4, input, coord.zw, VXC_5BITOFFSET_XY(0, 1), \\\n\
        VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(line1RGB4, input, coord1, VXC_5BITOFFSET_XY(0, 1), \\\n\
        VXC_MODIFIER(6, 11, 0, VXC_RM_TowardZero, 0)); \\\n\
 \\\n\
    float4 bgrMean = (float4)(bMean, gMean, rMean, 0); \\\n\
 \\\n\
    bgrMean *= f32Var; \\\n\
 \\\n\
    int4 test01, temp1; \\\n\
    int4 test02, temp2; \\\n\
    int4 tt; \\\n\
    vxc_uchar4 val; \\\n\
    int4 coord_out = (int4)(xPos.x, yPos, r_order, 0); \\\n\
 \\\n\
    vxc_uchar8 line1, line2; \\\n\
 \\\n\
    /*R*/ \\\n\
    VXC_DP2x8(line1, line0RGB1, line0RGB2, \\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniUnpackToR); \\\n\
    VXC_DP2x8(line2, line1RGB3, line1RGB4, \\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniUnpackToR); \\\n\
 \\\n\
    VXC_DP4x4(test01, line1, line1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10); \\\n\
    VXC_DP4x4(temp1, line1, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal); \\\n\
    temp1 = temp1 + test01; \\\n\
 \\\n\
    VXC_DP4x4(test02, line2, line2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10); \\\n\
    VXC_DP4x4(temp2, line2, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal); \\\n\
    temp2 = temp2 + test02; \\\n\
    temp2 = fy * (temp2 - temp1) + (temp1 << 10); \\\n\
 \\\n\
    vxc_float4 tmp_dst; \\\n\
    vxc_uchar4 u8_dst; \\\n\
    VXC_DP4x4(u8_dst, temp2, 1 << 19, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), uniExtractBytes); \\\n\
    VXC_DP4x4(tmp_dst, u8_dst, u8_dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), \\\n\
        uniConvertIntergetoF32_4x4); \\\n\
 \\\n\
    /*convert U8 to dst*/ \\\n\
    dst_type dst; \\\n\
    tmp_dst = tmp_dst * f32Var - bgrMean.zzzz; \\\n\
    tmp_dst = tmp_dst * outputScale + outputZP; \\\n\
    conv_type dst0; \\\n\
    _viv_asm(CONV_RTE, dst0, tmp_dst); \\\n\
    VXC_DP2x8(dst, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniExtract8Data_2x8); \\\n\
    copy_type result; \\\n\
    _viv_asm(COPY, result, dst, 16); \\\n\
    VXC_WriteImage2DArray(output, coord_out, result, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
 \\\n\
    /*G*/ \\\n\
    VXC_DP2x8(line1, line0RGB1, line0RGB2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniUnpackToG); \\\n\
    VXC_DP2x8(line2, line1RGB3, line1RGB4, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniUnpackToG); \\\n\
 \\\n\
    coord_out.z = 1; \\\n\
    VXC_DP4x4(test01, line1, line1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10); \\\n\
    VXC_DP4x4(temp1, line1, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal); \\\n\
    temp1 = temp1 + test01; \\\n\
 \\\n\
    VXC_DP4x4(test02, line2, line2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10); \\\n\
    VXC_DP4x4(temp2, line2, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal); \\\n\
    temp2 = temp2 + test02; \\\n\
    temp2 = fy * (temp2 - temp1) + (temp1 << 10); \\\n\
 \\\n\
    VXC_DP4x4(u8_dst, temp2, 1 << 19, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), uniExtractBytes); \\\n\
    VXC_DP4x4(tmp_dst, u8_dst, u8_dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), \\\n\
            uniConvertIntergetoF32_4x4); \\\n\
 \\\n\
    tmp_dst = tmp_dst * f32Var - bgrMean.y; \\\n\
    tmp_dst = tmp_dst * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, tmp_dst); \\\n\
    VXC_DP2x8(dst, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniExtract8Data_2x8); \\\n\
    _viv_asm(COPY, result, dst, 16); \\\n\
    VXC_WriteImage2DArray(output, coord_out, result, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
 \\\n\
    /*B*/ \\\n\
    VXC_DP2x8(line1, line0RGB1, line0RGB2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniUnpackToB); \\\n\
    VXC_DP2x8(line2, line1RGB3, line1RGB4, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniUnpackToB); \\\n\
 \\\n\
    coord_out.z = b_order; \\\n\
    VXC_DP4x4(test01, line1, line1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10); \\\n\
    VXC_DP4x4(temp1, line1, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal); \\\n\
    temp1 = temp1 + test01; \\\n\
 \\\n\
    VXC_DP4x4(test02, line2, line2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10); \\\n\
    VXC_DP4x4(temp2, line2, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal); \\\n\
    temp2 = temp2 + test02; \\\n\
    temp2 = fy * (temp2 - temp1) + (temp1 << 10); \\\n\
 \\\n\
    VXC_DP4x4(u8_dst, temp2, 1 << 19, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), uniExtractBytes); \\\n\
    VXC_DP4x4(tmp_dst, u8_dst, u8_dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), \\\n\
        uniConvertIntergetoF32_4x4); \\\n\
 \\\n\
    tmp_dst = tmp_dst * f32Var - bgrMean.x; \\\n\
    tmp_dst = tmp_dst * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, tmp_dst); \\\n\
    VXC_DP2x8(dst, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniExtract8Data_2x8); \\\n\
    _viv_asm(COPY, result, dst, 16); \\\n\
    VXC_WriteImage2DArray(output, coord_out, result, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
IMAGE_PRE_PROCESS(U8,  uint4, vxc_uchar16, vxc_uchar16)\n\
IMAGE_PRE_PROCESS(I8,  int4,  vxc_char16,  vxc_char16)\n\
IMAGE_PRE_PROCESS(I16, int4,  vxc_short8,  vxc_short8)\n\
IMAGE_PRE_PROCESS(F16, half4, vxc_half8,   vxc_short8)\n\
"; /* end of vsi_nn_kernel_pre_process_rgb_vx*/

static const char vsi_nn_kernel_pre_process_rgb_copy_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform float outputScale;\n\
_viv_uniform float outputZP;\n\
_viv_uniform int r_order;\n\
_viv_uniform int b_order;\n\
_viv_uniform VXC_512Bits uniExtractR_2x8;\n\
_viv_uniform VXC_512Bits uniExtractG_2x8;\n\
_viv_uniform VXC_512Bits uniExtractB_2x8;\n\
\n\
#define IMAGE_PRE_PROCESS_COPY_16BITS(dst_name, dst_type, copy_type) \\\n\
__kernel void vxRGBScaletoTensor_##dst_name##_copy \\\n\
    ( \\\n\
    __read_only image2d_array_t  input, \\\n\
    __write_only image2d_array_t output, \\\n\
         global int              *xRatio, \\\n\
         global int              *yRatio, \\\n\
         global int              *xOffset, \\\n\
         global int              *yOffset, \\\n\
                float            rMean, \\\n\
                float            gMean, \\\n\
                float            bMean, \\\n\
                float            f32Var, \\\n\
                int              reverse_channel, \\\n\
                int              trans \\\n\
    ) \\\n\
{ \\\n\
    int2 coord      = (int2)(get_global_id(0) * 3, get_global_id(1)); \\\n\
 \\\n\
    coord.xy += (int2) (*xOffset, *yOffset); \\\n\
    vxc_uchar16 src0, src1; \\\n\
    dst_type   dst0; \\\n\
    copy_type   dst; \\\n\
 \\\n\
    VXC_ReadImage(src0, input, coord.xy, 0, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src1, input, coord.xy, VXC_5BITOFFSET_XY(15, 0), \\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0)); \\\n\
 \\\n\
    f32Var *= outputScale; \\\n\
    float4      paramData = (float4)(rMean * f32Var, gMean * f32Var, bMean * f32Var, f32Var); \\\n\
    half4 paramData_f16; \\\n\
    _viv_asm(CONV, paramData_f16, paramData); \\\n\
 \\\n\
    int4 coord_out = (int4)(get_global_id(0), get_global_id(1), r_order, 0); \\\n\
 \\\n\
    VXC_DP2x8(dst0, src0, paramData_f16, VXC_MODIFIER(0, 4, 0, VXC_RM_ToNearestEven, 1), uniExtractR_2x8); \\\n\
    VXC_DP2x8(dst0, src1, paramData_f16, VXC_MODIFIER(5, 7, 0, VXC_RM_ToNearestEven, 1), uniExtractR_2x8); \\\n\
    _viv_asm(COPY, dst, dst0, 16); \\\n\
    VXC_WriteImage2DArray(output, coord_out, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
 \\\n\
    coord_out.z = 1; \\\n\
    VXC_DP2x8(dst0, src0, paramData_f16, VXC_MODIFIER(0, 4, 0, VXC_RM_ToNearestEven, 1), uniExtractG_2x8); \\\n\
    VXC_DP2x8(dst0, src1, paramData_f16, VXC_MODIFIER(5, 7, 0, VXC_RM_ToNearestEven, 1), uniExtractG_2x8); \\\n\
    _viv_asm(COPY, dst, dst0, 16); \\\n\
    VXC_WriteImage2DArray(output, coord_out, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
 \\\n\
    coord_out.z = b_order; \\\n\
    VXC_DP2x8(dst0, src0, paramData_f16, VXC_MODIFIER(0, 4, 0, VXC_RM_ToNearestEven, 1), uniExtractB_2x8); \\\n\
    VXC_DP2x8(dst0, src1, paramData_f16, VXC_MODIFIER(5, 7, 0, VXC_RM_ToNearestEven, 1), uniExtractB_2x8); \\\n\
    _viv_asm(COPY, dst, dst0, 16); \\\n\
    VXC_WriteImage2DArray(output, coord_out, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
IMAGE_PRE_PROCESS_COPY_16BITS(I16, vxc_short8,  vxc_short8)\n\
IMAGE_PRE_PROCESS_COPY_16BITS(F16, vxc_half8,   vxc_short8)\n\
\n\
#define IMAGE_PRE_PROCESS_COPY_8BITS(dst_name, dst_type) \\\n\
__kernel void vxRGBScaletoTensor_##dst_name##_copy \\\n\
    ( \\\n\
    __read_only image2d_array_t  input, \\\n\
    __write_only image2d_array_t output, \\\n\
         global int              *xRatio, \\\n\
         global int              *yRatio, \\\n\
         global int              *xOffset, \\\n\
         global int              *yOffset, \\\n\
                float            rMean, \\\n\
                float            gMean, \\\n\
                float            bMean, \\\n\
                float            f32Var, \\\n\
                int              reverse_channel, \\\n\
                int              trans \\\n\
    ) \\\n\
{ \\\n\
    int2 coord      = (int2)(get_global_id(0) * 3, get_global_id(1)); \\\n\
    coord.xy += (int2) (*xOffset, *yOffset); \\\n\
    vxc_uchar16 src0, src1; \\\n\
    dst_type dst; \\\n\
 \\\n\
    VXC_ReadImage(src0, input, coord.xy, 0, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(src1, input, coord.xy, VXC_5BITOFFSET_XY(15, 0), \\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0)); \\\n\
 \\\n\
    f32Var *= outputScale; \\\n\
    float4  paramData = (float4)(rMean * f32Var - outputZP, gMean * f32Var - outputZP, \\\n\
        bMean * f32Var - outputZP, f32Var); \\\n\
 \\\n\
    half4 paramData_f16; \\\n\
    _viv_asm(CONV, paramData_f16, paramData); \\\n\
 \\\n\
    int4 coord_out = (int4)(get_global_id(0), get_global_id(1), r_order, 0); \\\n\
 \\\n\
    VXC_DP2x8(dst, src0, paramData_f16, VXC_MODIFIER(0, 4, 0, VXC_RM_ToNearestEven, 1), uniExtractR_2x8); \\\n\
    VXC_DP2x8(dst, src1, paramData_f16, VXC_MODIFIER(5, 9, 0, VXC_RM_ToNearestEven, 1), uniExtractR_2x8); \\\n\
    VXC_WriteImage2DArray(output, coord_out, dst, VXC_MODIFIER(0, 9, 0, VXC_RM_TowardZero, 0)); \\\n\
 \\\n\
 \\\n\
    coord_out.z = 1; \\\n\
    VXC_DP2x8(dst, src0, paramData_f16, VXC_MODIFIER(0, 4, 0, VXC_RM_ToNearestEven, 1), uniExtractG_2x8); \\\n\
    VXC_DP2x8(dst, src1, paramData_f16, VXC_MODIFIER(5, 9, 0, VXC_RM_ToNearestEven, 1), uniExtractG_2x8); \\\n\
    VXC_WriteImage2DArray(output, coord_out, dst, VXC_MODIFIER(0, 9, 0, VXC_RM_TowardZero, 0)); \\\n\
 \\\n\
    coord_out.z = b_order; \\\n\
    VXC_DP2x8(dst, src0, paramData_f16, VXC_MODIFIER(0, 4, 0, VXC_RM_ToNearestEven, 1), uniExtractB_2x8); \\\n\
    VXC_DP2x8(dst, src1, paramData_f16, VXC_MODIFIER(5, 9, 0, VXC_RM_ToNearestEven, 1), uniExtractB_2x8); \\\n\
    VXC_WriteImage2DArray(output, coord_out, dst, VXC_MODIFIER(0, 9, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
IMAGE_PRE_PROCESS_COPY_8BITS(U8, vxc_uchar16)\n\
IMAGE_PRE_PROCESS_COPY_8BITS(I8, vxc_char16)\n\
"; /* end of vsi_nn_kernel_pre_process_rgb_copy_vx*/

static const char vsi_nn_kernel_pre_process_rgb_copy_trans_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform float outputScale;\n\
_viv_uniform float outputZP;\n\
_viv_uniform VXC_512Bits uniNormilizationLo_2x8;\n\
_viv_uniform VXC_512Bits uniNormilizationHi_2x8;\n\
#define IMAGE_PRE_PROCESS_COPY_16BITS_NHWC(dst_name, dst_type, copy_type) \\\n\
__kernel void vxRGBScaletoTensor_##dst_name##_copy_NHWC \\\n\
    ( \\\n\
    __read_only image2d_array_t  input, \\\n\
    __write_only image2d_array_t output, \\\n\
         global int              *xRatio, \\\n\
         global int              *yRatio, \\\n\
         global int              *xOffset, \\\n\
         global int              *yOffset, \\\n\
                float            rMean, \\\n\
                float            gMean, \\\n\
                float            bMean, \\\n\
                float            f32Var, \\\n\
                int           reverse_channel, \\\n\
                int           trans \\\n\
    ) \\\n\
{ \\\n\
    int2 coord      = (int2)(get_global_id(0), get_global_id(1)); \\\n\
 \\\n\
    coord.xy += (int2) (*xOffset, *yOffset); \\\n\
    vxc_uchar16 src0, src1; \\\n\
    dst_type   dst0, dst1; \\\n\
    copy_type   dst; \\\n\
 \\\n\
    VXC_ReadImage(src0, input, coord.xy, 0, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0)); \\\n\
 \\\n\
    f32Var *= outputScale; \\\n\
    float4      paramData = (float4)(rMean * f32Var, gMean * f32Var, bMean * f32Var, f32Var); \\\n\
    half4 paramData_f16; \\\n\
    _viv_asm(CONV, paramData_f16, paramData); \\\n\
 \\\n\
    int4 coord_out = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), get_global_id(0)); \\\n\
    coord_out.z = coord_out.x + 8; \\\n\
 \\\n\
    VXC_DP2x8(dst0, src0, paramData_f16, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), \\\n\
        uniNormilizationLo_2x8); \\\n\
    VXC_DP2x8(dst1, src0, paramData_f16, VXC_MODIFIER(0, 6, 0, VXC_RM_ToNearestEven, 1), \\\n\
        uniNormilizationHi_2x8); \\\n\
    _viv_asm(COPY, dst, dst0, 16); \\\n\
    VXC_WriteImage(output, coord_out.xy, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, dst, dst1, 16); \\\n\
    VXC_WriteImage(output, coord_out.zy, dst, VXC_MODIFIER(0, 6, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
IMAGE_PRE_PROCESS_COPY_16BITS_NHWC(I16, vxc_short8,  vxc_short8)\n\
IMAGE_PRE_PROCESS_COPY_16BITS_NHWC(F16, vxc_half8,   vxc_short8)\n\
\n\
#define IMAGE_PRE_PROCESS_COPY_8BITS_NHWC(dst_name, dst_type) \\\n\
__kernel void vxRGBScaletoTensor_##dst_name##_copy_NHWC \\\n\
    ( \\\n\
    __read_only image2d_array_t  input, \\\n\
    __write_only image2d_array_t output, \\\n\
         global int              *xRatio, \\\n\
         global int              *yRatio, \\\n\
         global int              *xOffset, \\\n\
         global int              *yOffset, \\\n\
                float            rMean, \\\n\
                float            gMean, \\\n\
                float            bMean, \\\n\
                float            f32Var, \\\n\
                int              reverse_channel, \\\n\
                int              trans \\\n\
    ) \\\n\
{ \\\n\
    int2 coord      = (int2)(get_global_id(0), get_global_id(1)); \\\n\
    coord.xy += (int2) (*xOffset, *yOffset); \\\n\
    vxc_uchar16 src0, src1; \\\n\
    dst_type dst; \\\n\
 \\\n\
    VXC_ReadImage(src0, input, coord.xy, 0, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0)); \\\n\
 \\\n\
    f32Var *= outputScale; \\\n\
    float4  paramData = (float4)(rMean * f32Var - outputZP, gMean * f32Var - outputZP, \\\n\
        bMean * f32Var - outputZP, f32Var); \\\n\
 \\\n\
    half4 paramData_f16; \\\n\
    _viv_asm(CONV, paramData_f16, paramData); \\\n\
 \\\n\
    int2 coord_out = (int2)(get_global_id(0), get_global_id(1)); \\\n\
 \\\n\
    VXC_DP2x8(dst, src0, paramData_f16, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), \\\n\
        uniNormilizationLo_2x8); \\\n\
    VXC_DP2x8(dst, src0, paramData_f16, VXC_MODIFIER(8, 14, 0, VXC_RM_ToNearestEven, 1), \\\n\
        uniNormilizationHi_2x8); \\\n\
    VXC_WriteImage(output, coord_out, dst, VXC_MODIFIER(0, 14, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
IMAGE_PRE_PROCESS_COPY_8BITS_NHWC(U8, vxc_uchar16)\n\
IMAGE_PRE_PROCESS_COPY_8BITS_NHWC(I8, vxc_char16)\n\
"; /* end of vsi_nn_kernel_pre_process_rgb_copy_trans_vx*/

static const char vsi_nn_kernel_pre_process_rgb_trans_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform VXC_512Bits uniVecShift10;\n\
_viv_uniform VXC_512Bits uniAddRShift;\n\
_viv_uniform VXC_512Bits uniGetTempVal;\n\
_viv_uniform VXC_512Bits uniExtractBytes;\n\
_viv_uniform VXC_512Bits uniUnpackToR;\n\
_viv_uniform VXC_512Bits uniUnpackToG;\n\
_viv_uniform VXC_512Bits uniUnpackToB;\n\
\n\
_viv_uniform VXC_512Bits uniConvertIntergetoF32_4x4;\n\
_viv_uniform float outputScale;\n\
_viv_uniform VXC_512Bits uniExtract8Data_2x8;\n\
_viv_uniform float outputZP;\n\
\n\
_viv_uniform VXC_512Bits uniRePackRGBLo_2x8;\n\
_viv_uniform VXC_512Bits uniRePackRGBHi_2x8;\n\
#define IMAGE_PRE_PROCESS_NHWC(dst_name, conv_type, dst_type, copy_type) \\\n\
    __kernel void vxRGBScaletoTensor_##dst_name##_NHWC \\\n\
    ( \\\n\
__read_only image2d_array_t  input, \\\n\
__write_only image2d_array_t output, \\\n\
        global int           *xRatio, \\\n\
        global int           *yRatio, \\\n\
        global int           *xOffset, \\\n\
        global int           *yOffset, \\\n\
               float         rMean, \\\n\
               float         gMean, \\\n\
               float         bMean, \\\n\
               float         f32Var, \\\n\
               int           reverse_channel, \\\n\
               int           trans \\\n\
    ) \\\n\
{ \\\n\
    int2 ratioXY = (int2)(*xRatio, *yRatio); \\\n\
    int4 xPos       = get_global_id(0); \\\n\
    int yPos        = get_global_id(1); \\\n\
    int2 ratioSufXY = (ratioXY >> 1) - (1 << 14); \\\n\
    xPos += (int4)(0, 1, 2, 3); \\\n\
 \\\n\
    /*x*/ \\\n\
    int4 fx0 = xPos * ratioXY.x + ratioSufXY.x; \\\n\
    int4 sx = fx0 & 0xffff8000; \\\n\
    fx0 -= sx; \\\n\
    sx = sx >> 15; \\\n\
 \\\n\
    vxc_short4 fx; \\\n\
    VXC_DP4x4(fx, fx0, 1 << 4, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniAddRShift); \\\n\
    /*y*/ \\\n\
    int fy = yPos * ratioXY.y + ratioSufXY.y; \\\n\
    int sy = fy & 0xffff8000; \\\n\
 \\\n\
    fy -= sy; \\\n\
    sy = sy >> 15; \\\n\
 \\\n\
    fy = (fy + (1<< 4)) >> 5; \\\n\
 \\\n\
    vxc_uchar16 line0RGB1, line0RGB2; \\\n\
    vxc_uchar16 line1RGB3, line1RGB4; \\\n\
    int4 coord; \\\n\
    sx = sx * 3 + *xOffset; \\\n\
    coord.xyz    = sx.xyz; \\\n\
    coord.w        = sy + *yOffset; \\\n\
    int2 coord1 = (int2)(sx.w, coord.w); \\\n\
    VXC_ReadImage(line0RGB1, input, coord.xw, 0, VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(line0RGB1, input, coord.yw, 0, VXC_MODIFIER(6, 11, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(line0RGB2, input, coord.zw, 0, VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(line0RGB2, input, coord1, 0, VXC_MODIFIER(6, 11, 0, VXC_RM_TowardZero, 0)); \\\n\
 \\\n\
    VXC_ReadImage(line1RGB3, input, coord.xw, VXC_5BITOFFSET_XY(0, 1), \\\n\
        VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(line1RGB3, input, coord.yw, VXC_5BITOFFSET_XY(0, 1), \\\n\
        VXC_MODIFIER(6, 11, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(line1RGB4, input, coord.zw, VXC_5BITOFFSET_XY(0, 1), \\\n\
        VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_ReadImage(line1RGB4, input, coord1, VXC_5BITOFFSET_XY(0, 1), \\\n\
        VXC_MODIFIER(6, 11, 0, VXC_RM_TowardZero, 0)); \\\n\
 \\\n\
    float4 bgrMean = (float4)(bMean, gMean, rMean, 0); \\\n\
 \\\n\
    bgrMean *= f32Var; \\\n\
 \\\n\
    int4 test01, temp1; \\\n\
    int4 test02, temp2; \\\n\
    int4 tt; \\\n\
    vxc_uchar4 val; \\\n\
    int4 coord_out = (int4)(xPos.x * 3, yPos, xPos.x * 3 + 6, 0); \\\n\
 \\\n\
    vxc_uchar8 line1, line2; \\\n\
 \\\n\
    /*R*/ \\\n\
    VXC_DP2x8(line1, line0RGB1, line0RGB2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniUnpackToR); \\\n\
    VXC_DP2x8(line2, line1RGB3, line1RGB4, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniUnpackToR); \\\n\
 \\\n\
    VXC_DP4x4(test01, line1, line1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10); \\\n\
    VXC_DP4x4(temp1, line1, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal); \\\n\
    temp1 = temp1 + test01; \\\n\
 \\\n\
    VXC_DP4x4(test02, line2, line2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10); \\\n\
    VXC_DP4x4(temp2, line2, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal); \\\n\
    temp2 = temp2 + test02; \\\n\
    temp2 = fy * (temp2 - temp1) + (temp1 << 10); \\\n\
 \\\n\
    vxc_float4 tmp_dst; \\\n\
    vxc_uchar4 u8_dst; \\\n\
    VXC_DP4x4(u8_dst, temp2, 1 << 19, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), uniExtractBytes); \\\n\
    VXC_DP4x4(tmp_dst, u8_dst, u8_dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), \\\n\
        uniConvertIntergetoF32_4x4); \\\n\
 \\\n\
    /*convert U8 to dst*/ \\\n\
    dst_type dstRG, dstB, dst; \\\n\
    tmp_dst = tmp_dst * f32Var - bgrMean.zzzz; \\\n\
    tmp_dst = tmp_dst * outputScale + outputZP; \\\n\
    conv_type dst0; \\\n\
    _viv_asm(CONV_RTE, dst0, tmp_dst); \\\n\
    VXC_DP2x8(dstRG, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniExtract8Data_2x8); \\\n\
 \\\n\
    /*G*/ \\\n\
    VXC_DP2x8(line1, line0RGB1, line0RGB2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniUnpackToG); \\\n\
    VXC_DP2x8(line2, line1RGB3, line1RGB4, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniUnpackToG); \\\n\
 \\\n\
    VXC_DP4x4(test01, line1, line1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10); \\\n\
    VXC_DP4x4(temp1, line1, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal); \\\n\
    temp1 = temp1 + test01; \\\n\
 \\\n\
    VXC_DP4x4(test02, line2, line2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10); \\\n\
    VXC_DP4x4(temp2, line2, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal); \\\n\
    temp2 = temp2 + test02; \\\n\
    temp2 = fy * (temp2 - temp1) + (temp1 << 10); \\\n\
 \\\n\
    VXC_DP4x4(u8_dst, temp2, 1 << 19, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), uniExtractBytes); \\\n\
    VXC_DP4x4(tmp_dst, u8_dst, u8_dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), \\\n\
        uniConvertIntergetoF32_4x4); \\\n\
 \\\n\
    tmp_dst = tmp_dst * f32Var - bgrMean.y; \\\n\
    tmp_dst = tmp_dst * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, tmp_dst); \\\n\
    VXC_DP2x8(dstRG, dst0, dst0, VXC_MODIFIER(4, 7, 0, VXC_RM_ToNearestEven, 1), uniExtract8Data_2x8); \\\n\
 \\\n\
    /*B*/ \\\n\
    VXC_DP2x8(line1, line0RGB1, line0RGB2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniUnpackToB); \\\n\
    VXC_DP2x8(line2, line1RGB3, line1RGB4, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniUnpackToB); \\\n\
 \\\n\
    VXC_DP4x4(test01, line1, line1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10); \\\n\
    VXC_DP4x4(temp1, line1, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal); \\\n\
    temp1 = temp1 + test01; \\\n\
 \\\n\
    VXC_DP4x4(test02, line2, line2, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniVecShift10); \\\n\
    VXC_DP4x4(temp2, line2, fx, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGetTempVal); \\\n\
    temp2 = temp2 + test02; \\\n\
    temp2 = fy * (temp2 - temp1) + (temp1 << 10); \\\n\
 \\\n\
    VXC_DP4x4(u8_dst, temp2, 1 << 19, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), uniExtractBytes); \\\n\
    VXC_DP4x4(tmp_dst, u8_dst, u8_dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 1), \\\n\
        uniConvertIntergetoF32_4x4); \\\n\
 \\\n\
    tmp_dst = tmp_dst * f32Var - bgrMean.x; \\\n\
    tmp_dst = tmp_dst * outputScale + outputZP; \\\n\
    _viv_asm(CONV_RTE, dst0, tmp_dst); \\\n\
    VXC_DP2x8(dstB, dst0, dst0, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniExtract8Data_2x8); \\\n\
    VXC_DP2x8(dst, dstRG, dstB, VXC_MODIFIER(0, 5, 0, VXC_RM_ToNearestEven, 1), uniRePackRGBLo_2x8); \\\n\
    copy_type result; \\\n\
    _viv_asm(COPY, result, dst, 16); \\\n\
    VXC_WriteImage(output, coord_out.xy, result, VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0)); \\\n\
    VXC_DP2x8(dst, dstRG, dstB, VXC_MODIFIER(0, 5, 0, VXC_RM_ToNearestEven, 1), uniRePackRGBHi_2x8); \\\n\
    _viv_asm(COPY, result, dst, 16); \\\n\
    VXC_WriteImage(output, coord_out.zy, result, VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
IMAGE_PRE_PROCESS_NHWC(U8,  uint4, vxc_uchar16, vxc_uchar16)\n\
IMAGE_PRE_PROCESS_NHWC(I8,  int4,  vxc_char16,  vxc_char16)\n\
IMAGE_PRE_PROCESS_NHWC(I16, int4,  vxc_short8,  vxc_short8)\n\
IMAGE_PRE_PROCESS_NHWC(F16, half4, vxc_half8,   vxc_short8)\n\
"; /* end of vsi_nn_kernel_pre_process_rgb_trans_vx*/

static const char vsi_nn_kernel_pre_process_yuv420_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform VXC_512Bits uniCalculateTmpR1st_4x4;\n\
_viv_uniform VXC_512Bits uniCalculateTmpR2nd_4x4;\n\
_viv_uniform VXC_512Bits uniCalculateTmpR3rd_4x4;\n\
_viv_uniform VXC_512Bits uniCalculateTmpR4th_4x4;\n\
_viv_uniform VXC_512Bits uniCalculateR1st_4x4;\n\
\n\
_viv_uniform VXC_512Bits uniCalculateTmpG1st_4x4;\n\
_viv_uniform VXC_512Bits uniCalculateTmpG2nd_4x4;\n\
_viv_uniform VXC_512Bits uniCalculateTmpG3rd_4x4;\n\
_viv_uniform VXC_512Bits uniCalculateTmpG4th_4x4;\n\
_viv_uniform VXC_512Bits uniCalculateTmpGbyU_2x8;\n\
\n\
_viv_uniform VXC_512Bits uniCalculateG1st_4x4;\n\
_viv_uniform VXC_512Bits uniCalculateG2nd_4x4;\n\
_viv_uniform VXC_512Bits uniCalculateG3rd_4x4;\n\
_viv_uniform VXC_512Bits uniCalculateG4th_4x4;\n\
\n\
_viv_uniform VXC_512Bits uniCalculateTmpB1st_4x4;\n\
_viv_uniform VXC_512Bits uniCalculateTmpB2nd_4x4;\n\
_viv_uniform VXC_512Bits uniCalculateTmpB3rd_4x4;\n\
_viv_uniform VXC_512Bits uniCalculateTmpB4th_4x4;\n\
_viv_uniform VXC_512Bits uniCalculateB1st_4x4;\n\
\n\
_viv_uniform VXC_512Bits uniPackBG0_2x8;\n\
_viv_uniform VXC_512Bits uniPackTmpAndR_2x8;\n\
_viv_uniform VXC_512Bits uniPackRB0_2x8;\n\
_viv_uniform VXC_512Bits uniPackTmp0AndG_2x8;\n\
_viv_uniform VXC_512Bits uniPackGR1_2x8;\n\
_viv_uniform VXC_512Bits uniPackTmp1AndB_2x8;\n\
_viv_uniform VXC_512Bits uniPackBG1_2x8;\n\
_viv_uniform VXC_512Bits uniPackTmp1AndR_2x8;\n\
_viv_uniform VXC_512Bits uniPackRB2_2x8;\n\
_viv_uniform VXC_512Bits uniPackTmp2AndG_2x8;\n\
_viv_uniform VXC_512Bits uniPackGR2_2x8;\n\
_viv_uniform VXC_512Bits uniPackTmp2AndB_2x8;\n\
\n\
_viv_uniform VXC_512Bits uniQuantU8toU8LoB_2x8;\n\
_viv_uniform VXC_512Bits uniQuantU8toU8HiB_2x8;\n\
_viv_uniform VXC_512Bits uniQuantU8toU8LoG_2x8;\n\
_viv_uniform VXC_512Bits uniQuantU8toU8HiG_2x8;\n\
_viv_uniform VXC_512Bits uniQuantU8toU8LoR_2x8;\n\
_viv_uniform VXC_512Bits uniQuantU8toU8HiR_2x8;\n\
\n\
_viv_uniform int bOrder;\n\
_viv_uniform int rOrder;\n\
_viv_uniform int zp;\n\
_viv_uniform float outputScale;\n\
\n\
__kernel void vxcYuv2rbg_u8(\n\
    __read_only image2d_array_t      y_img,\n\
    __read_only image2d_array_t      u_img,\n\
    __read_only image2d_array_t      v_img,\n\
    __write_only image2d_array_t    output,\n\
        global int *                xRatio,\n\
        global int *                yRatio,\n\
        global int *               xOffset,\n\
        global int *               yOffset,\n\
               float                 rMean,\n\
               float                 gMean,\n\
               float                 bMean,\n\
               float                   var,\n\
               int         reverse_channel,\n\
               int                   trans\n\
    )\n\
{\n\
    int4 pos = (int4)(get_global_id(0) + (*xOffset), get_global_id(1) + (*yOffset), 0, 0);\n\
    int4 pos1 = (int4)((get_global_id(0) + (*xOffset)) >> 1, (get_global_id(1) + (*yOffset)) >> 1, 0, 0);\n\
    vxc_uchar16 Y;\n\
    vxc_uchar8 U, V;\n\
    vxc_int4 C0, C1, C2, C3;\n\
    vxc_uchar16 R, G, B;\n\
    vxc_uchar16 dst0, dst1, dst2;\n\
\n\
    VXC_ReadImage(Y, y_img, pos.xy, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(U, u_img, pos1.xy, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(V, v_img, pos1.xy, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    //C = Y - 16;\n\
    //D = U - 128;\n\
    //E = V - 128;\n\
    // calculate R\n\
    // ((298 * C + 409 * E + 128) >> 8) -->  [(298Y + 409V - 56992) >> 8]\n\
    int tmpV = -56992;\n\
    VXC_DP4x4(C0, Y, V, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniCalculateTmpR1st_4x4);\n\
    VXC_DP4x4(C1, Y, V, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniCalculateTmpR2nd_4x4);\n\
    VXC_DP4x4(C2, Y, V, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniCalculateTmpR3rd_4x4);\n\
    VXC_DP4x4(C3, Y, V, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniCalculateTmpR4th_4x4);\n\
\n\
    VXC_DP4x4(R, C0, tmpV, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniCalculateR1st_4x4);\n\
    VXC_DP4x4(R, C1, tmpV, VXC_MODIFIER(4, 7, 0, VXC_RM_ToNearestEven, 1), uniCalculateR1st_4x4);\n\
    VXC_DP4x4(R, C2, tmpV, VXC_MODIFIER(8, 11, 0, VXC_RM_ToNearestEven, 1), uniCalculateR1st_4x4);\n\
    VXC_DP4x4(R, C3, tmpV, VXC_MODIFIER(12, 15, 0, VXC_RM_ToNearestEven, 1), uniCalculateR1st_4x4);\n\
\n\
    // calculate G\n\
    // ((298 * C - 100* D - 208 * E + 128) >> 8) --> [(298Y - 100U - 208V + 34784) >> 8]\n\
    // 298Y - 208V\n\
    VXC_DP4x4(C0, Y, V, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniCalculateTmpG1st_4x4);\n\
    VXC_DP4x4(C1, Y, V, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniCalculateTmpG2nd_4x4);\n\
    VXC_DP4x4(C2, Y, V, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniCalculateTmpG3rd_4x4);\n\
    VXC_DP4x4(C3, Y, V, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniCalculateTmpG4th_4x4);\n\
    // 34784 - 100U\n\
    ushort tmpG = 34784;\n\
    vxc_ushort8 tmpDstG;\n\
    VXC_DP2x8(tmpDstG, U, tmpG, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniCalculateTmpGbyU_2x8);\n\
    VXC_DP4x4(G, C0, tmpDstG, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniCalculateG1st_4x4);\n\
    VXC_DP4x4(G, C1, tmpDstG, VXC_MODIFIER(4, 7, 0, VXC_RM_ToNearestEven, 1), uniCalculateG2nd_4x4);\n\
    VXC_DP4x4(G, C2, tmpDstG, VXC_MODIFIER(8, 11, 0, VXC_RM_ToNearestEven, 1), uniCalculateG3rd_4x4);\n\
    VXC_DP4x4(G, C3, tmpDstG, VXC_MODIFIER(12, 15, 0, VXC_RM_ToNearestEven, 1), uniCalculateG4th_4x4);\n\
\n\
    // calculate B\n\
    // ((298 * C + 516 * D + 128) >> 8) ==> [(298Y + 516U - 70688) >> 8]\n\
    VXC_DP4x4(C0, Y, U, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniCalculateTmpB1st_4x4);\n\
    VXC_DP4x4(C1, Y, U, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniCalculateTmpB2nd_4x4);\n\
    VXC_DP4x4(C2, Y, U, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniCalculateTmpB3rd_4x4);\n\
    VXC_DP4x4(C3, Y, U, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniCalculateTmpB4th_4x4);\n\
    tmpV = -70688;\n\
    VXC_DP4x4(B, C0, tmpV, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniCalculateB1st_4x4);\n\
    VXC_DP4x4(B, C1, tmpV, VXC_MODIFIER(4, 7, 0, VXC_RM_ToNearestEven, 1), uniCalculateB1st_4x4);\n\
    VXC_DP4x4(B, C2, tmpV, VXC_MODIFIER(8, 11, 0, VXC_RM_ToNearestEven, 1), uniCalculateB1st_4x4);\n\
    VXC_DP4x4(B, C3, tmpV, VXC_MODIFIER(12, 15, 0, VXC_RM_ToNearestEven, 1), uniCalculateB1st_4x4);\n\
\n\
    var *= outputScale;\n\
    float4  paramData = (float4)(bMean * var - zp, gMean * var - zp,\\\n\
        rMean * var - zp, var);\n\
    half4 paramData_f16;\n\
    _viv_asm(CONV, paramData_f16, paramData);\n\
\n\
    VXC_DP2x8(dst0, B, paramData_f16, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniQuantU8toU8LoB_2x8);\n\
    VXC_DP2x8(dst0, B, paramData_f16, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniQuantU8toU8HiB_2x8);\n\
\n\
    VXC_DP2x8(dst1, G, paramData_f16, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniQuantU8toU8LoG_2x8);\n\
    VXC_DP2x8(dst1, G, paramData_f16, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniQuantU8toU8HiG_2x8);\n\
\n\
    VXC_DP2x8(dst2, R, paramData_f16, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniQuantU8toU8LoR_2x8);\n\
    VXC_DP2x8(dst2, R, paramData_f16, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniQuantU8toU8HiR_2x8);\n\
\n\
    pos = (int4)(get_global_id(0), get_global_id(1), 0, 0);\n\
    pos.z = bOrder;\n\
    VXC_WriteImage2DArray(output, pos, dst0, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    pos.z = 1;\n\
    VXC_WriteImage2DArray(output, pos, dst1, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    pos.z = rOrder;\n\
    VXC_WriteImage2DArray(output, pos, dst2, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
// store bgrbgrbgr\n\
__kernel void vxcYuv2rbg_trans_u8(\n\
    __read_only image2d_array_t      y_img,\n\
    __read_only image2d_array_t      u_img,\n\
    __read_only image2d_array_t      v_img,\n\
    __write_only image2d_array_t    output,\n\
        global int *                xRatio,\n\
        global int *                yRatio,\n\
        global int *               xOffset,\n\
        global int *               yOffset,\n\
               float                 rMean,\n\
               float                 gMean,\n\
               float                 bMean,\n\
               float                   var,\n\
               int         reverse_channel,\n\
               int                   trans\n\
    )\n\
{\n\
    int4 pos = (int4)(get_global_id(0) + (*xOffset), get_global_id(1) + (*yOffset), 0, 0);\n\
    int4 pos1 = (int4)((get_global_id(0) + (*xOffset)) >> 1, (get_global_id(1) + (*yOffset)) >> 1, 0, 0);\n\
    vxc_uchar16 Y;\n\
    vxc_uchar8 U, V;\n\
    vxc_int4 C0, C1, C2, C3;\n\
    vxc_uchar16 R, G, B;\n\
    vxc_uchar16 dst;\n\
\n\
    VXC_ReadImage(Y, y_img, pos.xy, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(U, u_img, pos1.xy, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(V, v_img, pos1.xy, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    var *= outputScale;\n\
    float4  paramData = (float4)(bMean * var - zp, gMean * var - zp,\\\n\
        rMean * var - zp, var);\n\
    half4 paramData_f16;\n\
    _viv_asm(CONV, paramData_f16, paramData);\n\
\n\
    //C = Y - 16;\n\
    //D = U - 128;\n\
    //E = V - 128;\n\
    // calculate R\n\
    // ((298 * C + 409 * E + 128) >> 8) -->  [(298Y + 409V - 56992) >> 8]\n\
    int tmpV = -56992;\n\
    VXC_DP4x4(C0, Y, V, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniCalculateTmpR1st_4x4);\n\
    VXC_DP4x4(C1, Y, V, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniCalculateTmpR2nd_4x4);\n\
    VXC_DP4x4(C2, Y, V, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniCalculateTmpR3rd_4x4);\n\
    VXC_DP4x4(C3, Y, V, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniCalculateTmpR4th_4x4);\n\
\n\
    VXC_DP4x4(dst, C0, tmpV, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniCalculateR1st_4x4);\n\
    VXC_DP4x4(dst, C1, tmpV, VXC_MODIFIER(4, 7, 0, VXC_RM_ToNearestEven, 1), uniCalculateR1st_4x4);\n\
    VXC_DP4x4(dst, C2, tmpV, VXC_MODIFIER(8, 11, 0, VXC_RM_ToNearestEven, 1), uniCalculateR1st_4x4);\n\
    VXC_DP4x4(dst, C3, tmpV, VXC_MODIFIER(12, 15, 0, VXC_RM_ToNearestEven, 1), uniCalculateR1st_4x4);\n\
\n\
    VXC_DP2x8(R, dst, paramData_f16, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniQuantU8toU8LoR_2x8);\n\
    VXC_DP2x8(R, dst, paramData_f16, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniQuantU8toU8HiR_2x8);\n\
\n\
    // calculate G\n\
    // ((298 * C - 100* D - 208 * E + 128) >> 8) --> [(298Y - 100U - 208V + 34784) >> 8]\n\
    // 298Y - 208V\n\
    VXC_DP4x4(C0, Y, V, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniCalculateTmpG1st_4x4);\n\
    VXC_DP4x4(C1, Y, V, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniCalculateTmpG2nd_4x4);\n\
    VXC_DP4x4(C2, Y, V, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniCalculateTmpG3rd_4x4);\n\
    VXC_DP4x4(C3, Y, V, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniCalculateTmpG4th_4x4);\n\
    // 34784 - 100U\n\
    ushort tmpG = 34784;\n\
    vxc_ushort8 tmpDstG;\n\
    VXC_DP2x8(tmpDstG, U, tmpG, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniCalculateTmpGbyU_2x8);\n\
    VXC_DP4x4(dst, C0, tmpDstG, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniCalculateG1st_4x4);\n\
    VXC_DP4x4(dst, C1, tmpDstG, VXC_MODIFIER(4, 7, 0, VXC_RM_ToNearestEven, 1), uniCalculateG2nd_4x4);\n\
    VXC_DP4x4(dst, C2, tmpDstG, VXC_MODIFIER(8, 11, 0, VXC_RM_ToNearestEven, 1), uniCalculateG3rd_4x4);\n\
    VXC_DP4x4(dst, C3, tmpDstG, VXC_MODIFIER(12, 15, 0, VXC_RM_ToNearestEven, 1), uniCalculateG4th_4x4);\n\
\n\
    VXC_DP2x8(G, dst, paramData_f16, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniQuantU8toU8LoG_2x8);\n\
    VXC_DP2x8(G, dst, paramData_f16, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniQuantU8toU8HiG_2x8);\n\
\n\
    // calculate B\n\
    // ((298 * C + 516 * D + 128) >> 8) ==> [(298Y + 516U - 70688) >> 8]\n\
    VXC_DP4x4(C0, Y, U, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniCalculateTmpB1st_4x4);\n\
    VXC_DP4x4(C1, Y, U, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniCalculateTmpB2nd_4x4);\n\
    VXC_DP4x4(C2, Y, U, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniCalculateTmpB3rd_4x4);\n\
    VXC_DP4x4(C3, Y, U, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniCalculateTmpB4th_4x4);\n\
    tmpV = -70688;\n\
    VXC_DP4x4(dst, C0, tmpV, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniCalculateB1st_4x4);\n\
    VXC_DP4x4(dst, C1, tmpV, VXC_MODIFIER(4, 7, 0, VXC_RM_ToNearestEven, 1), uniCalculateB1st_4x4);\n\
    VXC_DP4x4(dst, C2, tmpV, VXC_MODIFIER(8, 11, 0, VXC_RM_ToNearestEven, 1), uniCalculateB1st_4x4);\n\
    VXC_DP4x4(dst, C3, tmpV, VXC_MODIFIER(12, 15, 0, VXC_RM_ToNearestEven, 1), uniCalculateB1st_4x4);\n\
\n\
    VXC_DP2x8(B, dst, paramData_f16, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniQuantU8toU8LoB_2x8);\n\
    VXC_DP2x8(B, dst, paramData_f16, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniQuantU8toU8HiB_2x8);\n\
\n\
    // reorder to bgr\n\
    vxc_uchar8 tmpdst0, tmpdst1;\n\
    vxc_uchar16 dst0, dst1, dst2;\n\
\n\
    // BGR BGR BG\n\
    VXC_DP2x8(tmpdst0, B, G, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniPackBG0_2x8);\n\
    VXC_DP2x8(dst0, tmpdst0, R, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniPackTmpAndR_2x8);\n\
\n\
    // RBG RBG RB\n\
    VXC_DP2x8(tmpdst0, R, B, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniPackRB0_2x8);\n\
    VXC_DP2x8(dst0, tmpdst0, G, VXC_MODIFIER(8, 15, 0, VXC_RM_TowardZero, 0), uniPackTmp0AndG_2x8);\n\
\n\
    pos = (int4)(get_global_id(0) * 3, get_global_id(1), 0, 0);\n\
\n\
    VXC_WriteImage2DArray(output, pos, dst0, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    pos.x += 16;\n\
\n\
    // GRB GRB GR\n\
    VXC_DP2x8(tmpdst0, G, R, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniPackGR1_2x8);\n\
    VXC_DP2x8(dst1, tmpdst0, B, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniPackTmp1AndB_2x8);\n\
\n\
    // BGR BGR BG\n\
    VXC_DP2x8(tmpdst0, B, G, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniPackBG1_2x8);\n\
    VXC_DP2x8(dst1, tmpdst0, R, VXC_MODIFIER(8, 15, 0, VXC_RM_TowardZero, 0), uniPackTmp1AndR_2x8);\n\
\n\
    VXC_WriteImage2DArray(output, pos, dst1, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    pos.x += 16;\n\
\n\
    // RBG RBG RB\n\
    VXC_DP2x8(tmpdst0, R, B, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniPackRB2_2x8);\n\
    VXC_DP2x8(dst2, tmpdst0, G, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniPackTmp2AndG_2x8);\n\
\n\
    // GRB GRB GR\n\
    VXC_DP2x8(tmpdst1, G, R, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniPackGR2_2x8);\n\
    VXC_DP2x8(dst2, tmpdst1, B, VXC_MODIFIER(8, 15, 0, VXC_RM_TowardZero, 0), uniPackTmp2AndB_2x8);\n\
\n\
    VXC_WriteImage2DArray(output, pos, dst2, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_pre_process_yuv420_vx*/

static const char vsi_nn_kernel_pre_process_yuv420_scale_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform VXC_512Bits uniCalculateR1st_4x4;\n\
_viv_uniform VXC_512Bits uniCalculateTmpGbyU_2x8;\n\
_viv_uniform VXC_512Bits uniCalculateTmpGbyU2nd_2x8;\n\
\n\
_viv_uniform VXC_512Bits uniCalculateB1st_4x4;\n\
_viv_uniform VXC_512Bits uniConvertInt32toUint8_2x8;\n\
_viv_uniform VXC_512Bits uniDescaleU8_4x4;\n\
\n\
_viv_uniform VXC_512Bits uniCalculateTmpRWise_4x4;\n\
_viv_uniform VXC_512Bits uniCalculateTmpRWise2nd_4x4;\n\
_viv_uniform VXC_512Bits uniCalculateTmpRWise3rd_4x4;\n\
_viv_uniform VXC_512Bits uniCalculateTmpRWise4th_4x4;\n\
_viv_uniform VXC_512Bits uniCalculateTmpGWise_4x4;\n\
_viv_uniform VXC_512Bits uniCalculateTmpGWise2nd_4x4;\n\
_viv_uniform VXC_512Bits uniCalculateTmpGWise3rd_4x4;\n\
_viv_uniform VXC_512Bits uniCalculateTmpGWise4th_4x4;\n\
_viv_uniform VXC_512Bits uniCalculateGWise_4x4;\n\
_viv_uniform VXC_512Bits uniCalculateGWise2nd_4x4;\n\
\n\
_viv_uniform VXC_512Bits uniCalculateTmpBWise_4x4;\n\
_viv_uniform VXC_512Bits uniCalculateTmpBWise2nd_4x4;\n\
_viv_uniform VXC_512Bits uniCalculateTmpBWise3rd_4x4;\n\
_viv_uniform VXC_512Bits uniCalculateTmpBWise4th_4x4;\n\
\n\
_viv_uniform VXC_512Bits uniBilinearTmp1st_4x4;\n\
_viv_uniform VXC_512Bits uniBilinearTmp2nd_4x4;\n\
_viv_uniform VXC_512Bits uniBilinearTmp3rd_4x4;\n\
_viv_uniform VXC_512Bits uniBilinearTmp4th_4x4;\n\
\n\
_viv_uniform int bOrder;\n\
_viv_uniform int rOrder;\n\
_viv_uniform int zp;\n\
_viv_uniform float outputScale;\n\
\n\
__kernel void vxcYuv2rbg_resize_norm_u8(\n\
    __read_only image2d_array_t y_img, __read_only image2d_array_t u_img,\n\
    __read_only image2d_array_t v_img, __write_only image2d_array_t    output,\n\
    global int *xRatio, global int * yRatio, global int * xOffset, global int * yOffset,\n\
    float rMean, float gMean, float bMean, float var, int reverse_channel, int trans)\n\
{\n\
    int4 gidx = get_global_id(0);\n\
    int gidy = get_global_id(1);\n\
    gidx += (int4)(0, 1, 2, 3);\n\
\n\
    int4 fx = (gidx * (*xRatio) + ((*xRatio) >> 1)) - (1 << 14);\n\
    int4 sx = fx & 0xffff8000; // Floor\n\
    int fy, sy;\n\
    fx -= sx;\n\
    sx = sx >> 15;\n\
    fx = (fx +(1 << 4)) >> 5;\n\
\n\
    // for y\n\
    fy = (gidy * (*yRatio) + ((*yRatio) >> 1)) - (1<< 14);\n\
    sy = fy & 0xffff8000; // Floor\n\
    fy -= sy;\n\
    sy = sy >> 15;\n\
\n\
    sy = sy < 0 ? 0 : sy;\n\
    fy = fy < 0 ? 0 : fy;\n\
\n\
    fy = (fy + (1<< 4)) >> 5;\n\
    sx += (*xOffset);\n\
    sy += (*yOffset);\n\
    int4 srcPos = (int4)(sx.x, sy, get_global_id(2), 0);\n\
    int4 srcPos1 = (int4)(sx.x >> 1, sy >> 1, get_global_id(2), 0);\n\
    int4 srcPos2 = (int4)(sx.x >> 1, (sy + 1) >> 1, get_global_id(2), 0);\n\
\n\
    vxc_uchar16 Y, U, V;\n\
    vxc_int4 C0, C1, C2, C3;\n\
    vxc_uchar16 R, G, B;\n\
\n\
    VXC_ReadImage(Y, y_img, srcPos, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 1, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(Y, y_img, srcPos, VXC_5BITOFFSET_XY(0, 1), VXC_MODIFIER(2, 3, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(U, u_img, srcPos1, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(V, v_img, srcPos1, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
    srcPos1.x = (sx.x + 1) >> 1;\n\
    VXC_ReadImage(U, u_img, srcPos1, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(1, 1, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(V, v_img, srcPos1, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(1, 1, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_ReadImage(U, u_img, srcPos2, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(2, 2, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(V, v_img, srcPos2, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(2, 2, 0, VXC_RM_TowardZero, 0));\n\
    srcPos2.x = (sx.x + 1) >> 1;\n\
    VXC_ReadImage(U, u_img, srcPos2, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(3, 3, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(V, v_img, srcPos2, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(3, 3, 0, VXC_RM_TowardZero, 0));\n\
\n\
    srcPos.x = sx.y;\n\
    srcPos1.x = sx.y >> 1;\n\
    srcPos2.x = sx.y >> 1;\n\
    VXC_ReadImage(Y, y_img, srcPos, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(4, 5, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(Y, y_img, srcPos, VXC_5BITOFFSET_XY(0, 1), VXC_MODIFIER(6, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(U, u_img, srcPos1, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(4, 4, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(V, v_img, srcPos1, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(4, 4, 0, VXC_RM_TowardZero, 0));\n\
    srcPos1.x = (sx.y + 1) >> 1;\n\
    VXC_ReadImage(U, u_img, srcPos1, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(5, 5, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(V, v_img, srcPos1, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(5, 5, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_ReadImage(U, u_img, srcPos2, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(6, 6, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(V, v_img, srcPos2, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(6, 6, 0, VXC_RM_TowardZero, 0));\n\
    srcPos2.x = (sx.y + 1) >> 1;\n\
    VXC_ReadImage(U, u_img, srcPos2, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(7, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(V, v_img, srcPos2, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(7, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    srcPos.x = sx.z;\n\
    srcPos1.x = sx.z >> 1;\n\
    srcPos2.x = sx.z >> 1;\n\
    VXC_ReadImage(Y, y_img, srcPos, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(8, 9, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(Y, y_img, srcPos, VXC_5BITOFFSET_XY(0, 1), VXC_MODIFIER(10, 11, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(U, u_img, srcPos1, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(8, 8, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(V, v_img, srcPos1, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(8, 8, 0, VXC_RM_TowardZero, 0));\n\
    srcPos1.x = (sx.z + 1) >> 1;\n\
    VXC_ReadImage(U, u_img, srcPos1, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(9, 9, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(V, v_img, srcPos1, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(9, 9, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_ReadImage(U, u_img, srcPos2, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(10, 10, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(V, v_img, srcPos2, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(10, 10, 0, VXC_RM_TowardZero, 0));\n\
    srcPos2.x = (sx.z + 1) >> 1;\n\
    VXC_ReadImage(U, u_img, srcPos2, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(11, 11, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(V, v_img, srcPos2, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(11, 11, 0, VXC_RM_TowardZero, 0));\n\
\n\
    srcPos.x = sx.w;\n\
    srcPos1.x = sx.w >> 1;\n\
    srcPos2.x = sx.w >> 1;\n\
    VXC_ReadImage(Y, y_img, srcPos, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(12, 13, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(Y, y_img, srcPos, VXC_5BITOFFSET_XY(0, 1), VXC_MODIFIER(14, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(U, u_img, srcPos1, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(12, 12, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(V, v_img, srcPos1, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(12, 12, 0, VXC_RM_TowardZero, 0));\n\
    srcPos1.x = (sx.w + 1) >> 1;\n\
    VXC_ReadImage(U, u_img, srcPos1, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(13, 13, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(V, v_img, srcPos1, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(13, 13, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_ReadImage(U, u_img, srcPos2, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(14, 14, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(V, v_img, srcPos2, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(14, 14, 0, VXC_RM_TowardZero, 0));\n\
    srcPos2.x = (sx.w + 1) >> 1;\n\
    VXC_ReadImage(U, u_img, srcPos2, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(15, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(V, v_img, srcPos2, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(15, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    //C = Y - 16; D = U - 128; E = V - 128;\n\
    // ((298 * C + 409 * E + 128) >> 8) -->  [(298Y + 409V - 56992) >> 8]\n\
    int tmpV = -56992;\n\
    VXC_DP4x4(C0, Y, V, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniCalculateTmpRWise_4x4);\n\
    VXC_DP4x4(C1, Y, V, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniCalculateTmpRWise2nd_4x4);\n\
    VXC_DP4x4(C2, Y, V, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniCalculateTmpRWise3rd_4x4);\n\
    VXC_DP4x4(C3, Y, V, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniCalculateTmpRWise4th_4x4);\n\
    VXC_DP4x4(R, C0, tmpV, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniCalculateR1st_4x4);\n\
    VXC_DP4x4(R, C1, tmpV, VXC_MODIFIER(4, 7, 0, VXC_RM_ToNearestEven, 1), uniCalculateR1st_4x4);\n\
    VXC_DP4x4(R, C2, tmpV, VXC_MODIFIER(8, 11, 0, VXC_RM_ToNearestEven, 1), uniCalculateR1st_4x4);\n\
    VXC_DP4x4(R, C3, tmpV, VXC_MODIFIER(12, 15, 0, VXC_RM_ToNearestEven, 1), uniCalculateR1st_4x4);\n\
\n\
    // ((298 * C - 100* D - 208 * E + 128) >> 8) --> [(298Y - 100U - 208V + 34784) >> 8]\n\
    // 298Y - 208V\n\
    VXC_DP4x4(C0, Y, V, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniCalculateTmpGWise_4x4);\n\
    VXC_DP4x4(C1, Y, V, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniCalculateTmpGWise2nd_4x4);\n\
    VXC_DP4x4(C2, Y, V, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniCalculateTmpGWise3rd_4x4);\n\
    VXC_DP4x4(C3, Y, V, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniCalculateTmpGWise4th_4x4);\n\
    // 34784 - 100U\n\
    ushort tmpG = 34784;\n\
    vxc_ushort8 tmpDstG, tmpDstG1;\n\
    VXC_DP2x8(tmpDstG, U, tmpG, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniCalculateTmpGbyU_2x8);\n\
    VXC_DP2x8(tmpDstG1, U, tmpG, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniCalculateTmpGbyU2nd_2x8);\n\
    VXC_DP4x4(G, C0, tmpDstG, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniCalculateGWise_4x4);\n\
    VXC_DP4x4(G, C1, tmpDstG, VXC_MODIFIER(4, 7, 0, VXC_RM_ToNearestEven, 1), uniCalculateGWise2nd_4x4);\n\
    VXC_DP4x4(G, C2, tmpDstG1, VXC_MODIFIER(8, 11, 0, VXC_RM_ToNearestEven, 1), uniCalculateGWise_4x4);\n\
    VXC_DP4x4(G, C3, tmpDstG1, VXC_MODIFIER(12, 15, 0, VXC_RM_ToNearestEven, 1), uniCalculateGWise2nd_4x4);\n\
\n\
    // ((298 * C + 516 * D + 128) >> 8) ==> [(298Y + 516U - 70688) >> 8]\n\
    VXC_DP4x4(C0, Y, U, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniCalculateTmpBWise_4x4);\n\
    VXC_DP4x4(C1, Y, U, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniCalculateTmpBWise2nd_4x4);\n\
    VXC_DP4x4(C2, Y, U, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniCalculateTmpBWise3rd_4x4);\n\
    VXC_DP4x4(C3, Y, U, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniCalculateTmpBWise4th_4x4);\n\
    tmpV = -70688;\n\
    VXC_DP4x4(B, C0, tmpV, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniCalculateB1st_4x4);\n\
    VXC_DP4x4(B, C1, tmpV, VXC_MODIFIER(4, 7, 0, VXC_RM_ToNearestEven, 1), uniCalculateB1st_4x4);\n\
    VXC_DP4x4(B, C2, tmpV, VXC_MODIFIER(8, 11, 0, VXC_RM_ToNearestEven, 1), uniCalculateB1st_4x4);\n\
    VXC_DP4x4(B, C3, tmpV, VXC_MODIFIER(12, 15, 0, VXC_RM_ToNearestEven, 1), uniCalculateB1st_4x4);\n\
\n\
    int4 result, temp1, temp2;\n\
    int4 tmpData0, tmpData1;\n\
\n\
    VXC_DP4x4(tmpData0, B, B, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniBilinearTmp1st_4x4);\n\
    VXC_DP4x4(tmpData1, B, B, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniBilinearTmp2nd_4x4);\n\
    temp1 = fx * tmpData0 + tmpData1;\n\
    // temp2 - temp1\n\
    VXC_DP4x4(tmpData0, B, B, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniBilinearTmp3rd_4x4);\n\
    VXC_DP4x4(tmpData1, B, B, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniBilinearTmp4th_4x4);\n\
    temp2 = fx * tmpData0 + tmpData1;\n\
    result = fy * temp2 + (temp1 << 10);\n\
\n\
    tmpV = 1 << 19;\n\
    vxc_uchar8 dst;\n\
    float4 tmpDst;\n\
    int4 dstPos = (int4)(get_global_id(0), gidy, 0, 0);\n\
    VXC_DP4x4(tmpDst, result, tmpV, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniDescaleU8_4x4);\n\
    tmpDst = (tmpDst - bMean) * var;\n\
    dstPos.z = bOrder;\n\
    result = convert_int4_rte(tmpDst * outputScale + zp);\n\
    VXC_DP2x8(dst, result, result, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertInt32toUint8_2x8);\n\
    VXC_WriteImage2DArray(output, dstPos, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_DP4x4(tmpData0, G, G, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniBilinearTmp1st_4x4);\n\
    VXC_DP4x4(tmpData1, G, G, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniBilinearTmp2nd_4x4);\n\
    temp1 = fx * tmpData0 + tmpData1;\n\
    VXC_DP4x4(tmpData0, G, G, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniBilinearTmp3rd_4x4);\n\
    VXC_DP4x4(tmpData1, G, G, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniBilinearTmp4th_4x4);\n\
    temp2 = fx * tmpData0 + tmpData1;\n\
    result = fy * temp2 + (temp1 << 10);\n\
    VXC_DP4x4(tmpDst, result, tmpV, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniDescaleU8_4x4);\n\
    tmpDst = (tmpDst - gMean) * var;\n\
    dstPos.z = 1;\n\
    result = convert_int4_rte(tmpDst * outputScale + zp);\n\
    VXC_DP2x8(dst, result, result, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertInt32toUint8_2x8);\n\
    VXC_WriteImage2DArray(output, dstPos, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_DP4x4(tmpData0, R, R, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniBilinearTmp1st_4x4);\n\
    VXC_DP4x4(tmpData1, R, R, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniBilinearTmp2nd_4x4);\n\
    temp1 = fx * tmpData0 + tmpData1;\n\
    VXC_DP4x4(tmpData0, R, R, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniBilinearTmp3rd_4x4);\n\
    VXC_DP4x4(tmpData1, R, R, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniBilinearTmp4th_4x4);\n\
    temp2 = fx * tmpData0 + tmpData1;\n\
    result = fy * temp2 + (temp1 << 10);\n\
    VXC_DP4x4(tmpDst, result, tmpV, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniDescaleU8_4x4);\n\
    tmpDst = (tmpDst - rMean) * var;\n\
    dstPos.z = rOrder;\n\
    result = convert_int4_rte(tmpDst * outputScale + zp);\n\
    VXC_DP2x8(dst, result, result, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertInt32toUint8_2x8);\n\
    VXC_WriteImage2DArray(output, dstPos, dst, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
}"; /* end of vsi_nn_kernel_pre_process_yuv420_scale_vx*/

static const char vsi_nn_kernel_prelu_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform VXC_512Bits UniFP16Mul_dp2x8;\n\
_viv_uniform VXC_512Bits uniConvertDirInt16Fp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertEndInt16Fp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertInt32toUint8_2x8;\n\
_viv_uniform VXC_512Bits uniConvertUint8SubZpToFp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertSecUint8SubZpToFp32_4x4;\n\
_viv_uniform float inScaleInt16;\n\
_viv_uniform float outScaleInt16;\n\
\n\
_viv_uniform int input_ZP;\n\
_viv_uniform float inputScale;\n\
_viv_uniform float outputScale;\n\
_viv_uniform int output_ZP;\n\
\n\
__kernel void vxcParametricRelu\n\
    (\n\
    image2d_array_t input,\n\
    image2d_array_t para,\n\
    image2d_array_t output\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    int4 coord_para = (int4)(coord.z, 0, 0, 0);\n\
\n\
    vxc_short8 img1_s16, para_s16, val_s16;\n\
    vxc_half8 img_fp16, para_fp16, val_fp16;\n\
\n\
    VXC_ReadImage2DArray(img1_s16, input, coord, VXC_5BITOFFSET_XY(0,0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(para_s16, para, coord_para.xy, VXC_5BITOFFSET_XY(0,0),\\\n\
        VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
\n\
    _viv_asm(COPY, para_fp16, para_s16, 16);\n\
    _viv_asm(COPY, img_fp16, img1_s16, 16);\n\
    VXC_DP2x8(val_fp16, img_fp16, para_fp16, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
        UniFP16Mul_dp2x8);\n\
    vxc_short8 mulData;\n\
    _viv_asm(COPY, mulData, val_fp16, 16);\n\
    val_s16 = img1_s16 > 0 ? img1_s16 : mulData;\n\
    VXC_WriteImage2DArray(output, coord, val_s16, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform VXC_512Bits uniConvertInt8FstFp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertInt8SecFp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertInt8TrdFp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertInt8ForFp32_4x4;\n\
_viv_uniform VXC_512Bits UniS8xFp16_dp2x8;\n\
_viv_uniform float in_scale_prelu;\n\
_viv_uniform float out_scale_prelu;\n\
_viv_uniform float scale_inOut;\n\
_viv_uniform float scale_inOut_u8;\n\
\n\
__kernel void vxcParametricRelu_int8\n\
    (\n\
    image2d_array_t input,\n\
    image2d_array_t para,\n\
    image2d_array_t output\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    int4 coord_para = (int4)(coord.z, 0, 0, 0);\n\
\n\
    vxc_char16 img_c16;\n\
    vxc_short8 para_s16;\n\
    half paraHlf;\n\
    float paraFp;\n\
    vxc_char16 outval;\n\
    vxc_float4 imgData0, imgData1, imgData2, imgData3;\n\
    vxc_float4 tmpOut0, tmpOut1, tmpOut2, tmpOut3;\n\
    vxc_int4 tmpVal0, tmpVal1;\n\
\n\
    VXC_ReadImage2DArray(img_c16, input, coord, VXC_5BITOFFSET_XY(0,0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(para_s16, para, coord_para.xy, VXC_5BITOFFSET_XY(0,0),\\\n\
        VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_DP4x4(imgData0, img_c16, img_c16, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertInt8FstFp32_4x4);\n\
    VXC_DP4x4(imgData1, img_c16, img_c16, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertInt8SecFp32_4x4);\n\
    VXC_DP4x4(imgData2, img_c16, img_c16, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertInt8TrdFp32_4x4);\n\
    VXC_DP4x4(imgData3, img_c16, img_c16, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertInt8ForFp32_4x4);\n\
\n\
    _viv_asm(COPY, paraHlf, para_s16, 4);\n\
    _viv_asm(CONV, paraFp, paraHlf);\n\
    imgData0 *= scale_inOut;\n\
    imgData1 *= scale_inOut;\n\
    imgData2 *= scale_inOut;\n\
    imgData3 *= scale_inOut;\n\
\n\
    vxc_float4 maxData0 = imgData0 > 0 ? imgData0 : 0.0;\n\
    vxc_float4 maxData1 = imgData1 > 0 ? imgData1 : 0.0;\n\
    vxc_float4 minData0 = imgData0 < 0 ? imgData0 : 0.0;\n\
    vxc_float4 minData1 = imgData1 < 0 ? imgData1 : 0.0;\n\
    tmpOut0 = maxData0 + paraFp * minData0;\n\
    tmpOut1 = maxData1 + paraFp * minData1;\n\
    tmpVal0 = convert_int4_rte(tmpOut0);\n\
    tmpVal1 = convert_int4_rte(tmpOut1);\n\
    VXC_DP2x8(outval, tmpVal0, tmpVal1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1),\\\n\
        uniConvertInt32toUint8_2x8);\n\
\n\
    maxData0 = imgData2 > 0 ? imgData2 : 0.0;\n\
    maxData1 = imgData3 > 0 ? imgData3 : 0.0;\n\
    minData0 = imgData2 < 0 ? imgData2 : 0.0;\n\
    minData1 = imgData3 < 0 ? imgData3 : 0.0;\n\
    tmpOut0 = maxData0 + paraFp * minData0;\n\
    tmpOut1 = maxData1 + paraFp * minData1;\n\
    tmpVal0 = convert_int4_rte(tmpOut0);\n\
    tmpVal1 = convert_int4_rte(tmpOut1);\n\
    VXC_DP2x8(outval, tmpVal0, tmpVal1, VXC_MODIFIER(8, 15, 0, VXC_RM_TowardZero, 1),\\\n\
        uniConvertInt32toUint8_2x8);\n\
\n\
    VXC_WriteImage2DArray(output, coord, outval, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcParametricRelu_int8_fp16\n\
    (\n\
    image2d_array_t input,\n\
    image2d_array_t para,\n\
    image2d_array_t output\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    int4 coord_para = (int4)(coord.z, 0, 0, 0);\n\
\n\
    vxc_char8 img1_s8;\n\
    vxc_short8 para_s16;\n\
    vxc_half8 img_fp16, para_fp16, val_fp16;\n\
    half inscale_fp16;\n\
\n\
    VXC_ReadImage2DArray(img1_s8, input, coord, VXC_5BITOFFSET_XY(0,0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(para_s16, para, coord_para.xy, VXC_5BITOFFSET_XY(0,0),\\\n\
        VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
\n\
    _viv_asm(CONV, inscale_fp16, in_scale_prelu);\n\
    _viv_asm(COPY, para_fp16, para_s16, 16);\n\
\n\
    VXC_DP2x8(img_fp16, img1_s8, inscale_fp16, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1),\\\n\
        UniS8xFp16_dp2x8);\n\
    VXC_DP2x8(val_fp16, img_fp16, para_fp16, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
        UniFP16Mul_dp2x8);\n\
    VXC_Clamp_Half(img_fp16, img_fp16, val_fp16, img_fp16, VXC_MODIFIER_CLAMP(0, 7, 0, 0));\n\
    _viv_asm(COPY, para_s16, img_fp16, 16);\n\
    VXC_WriteImage2DArray(output, coord, para_s16, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcParametricReluInt16_Int16\n\
    (\n\
    image2d_array_t input,\n\
    image2d_array_t para,\n\
    image2d_array_t output\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    int4 coord_para = (int4)(coord.z, 0, 0, 0);\n\
\n\
    vxc_short8 img_s16, para_s16;\n\
    half paraHlf;\n\
    float paraFp;\n\
    vxc_short8 outval;\n\
    vxc_float4 imgData0, imgData1;\n\
    vxc_float4 tmpOut0, tmpOut1;\n\
    vxc_int4 tmpVal0, tmpVal1;\n\
\n\
    VXC_ReadImage2DArray(img_s16, input, coord, VXC_5BITOFFSET_XY(0,0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(para_s16, para, coord_para.xy, VXC_5BITOFFSET_XY(0,0),\\\n\
        VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_DP4x4(imgData0, img_s16, img_s16, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertDirInt16Fp32_4x4);\n\
    VXC_DP4x4(imgData1, img_s16, img_s16, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertEndInt16Fp32_4x4);\n\
\n\
    _viv_asm(COPY, paraHlf, para_s16, 2);\n\
    _viv_asm(CONV, paraFp, paraHlf);\n\
    imgData0 *= scale_inOut;\n\
    imgData1 *= scale_inOut;\n\
\n\
    vxc_float4 maxData0 = imgData0 > 0 ? imgData0 : 0.0;\n\
    vxc_float4 maxData1 = imgData1 > 0 ? imgData1 : 0.0;\n\
    vxc_float4 minData0 = imgData0 < 0 ? imgData0 : 0.0;\n\
    vxc_float4 minData1 = imgData1 < 0 ? imgData1 : 0.0;\n\
    tmpOut0 = maxData0 + paraFp * minData0;\n\
    tmpOut1 = maxData1 + paraFp * minData1;\n\
\n\
    tmpVal0 = convert_int4_rte(tmpOut0);\n\
    tmpVal1 = convert_int4_rte(tmpOut1);\n\
\n\
    VXC_DP2x8(outval, tmpVal0, tmpVal1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1),\\\n\
        uniConvertInt32toUint8_2x8);\n\
    VXC_WriteImage2DArray(output, coord, outval, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcParametricReluUint8_Uint8\n\
    (\n\
    image2d_array_t input,\n\
    image2d_array_t para,\n\
    image2d_array_t output\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    int4 coord_para = (int4)(coord.z, 0, 0, 0);\n\
\n\
    vxc_uchar16 img_s16;\n\
    vxc_short8 para_s16;\n\
    half paraHlf;\n\
    float paraFp;\n\
    vxc_uchar8 outval;\n\
    vxc_float4 imgData0, imgData1;\n\
    vxc_float4 tmpOut0, tmpOut1;\n\
    vxc_int4 tmpVal0, tmpVal1;\n\
\n\
    VXC_ReadImage2DArray(img_s16, input, coord, VXC_5BITOFFSET_XY(0,0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(para_s16, para, coord_para.xy, VXC_5BITOFFSET_XY(0,0),\\\n\
        VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
\n\
    short zp = input_ZP;\n\
\n\
    VXC_DP4x4(imgData0, img_s16, zp, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertUint8SubZpToFp32_4x4);\n\
    VXC_DP4x4(imgData1, img_s16, zp, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertSecUint8SubZpToFp32_4x4);\n\
\n\
    _viv_asm(COPY, paraHlf, para_s16, 2);\n\
    _viv_asm(CONV, paraFp, paraHlf);\n\
    imgData0 *= scale_inOut_u8;\n\
    imgData1 *= scale_inOut_u8;\n\
\n\
    vxc_float4 maxData0 = imgData0 > 0 ? imgData0 : 0.0;\n\
    vxc_float4 maxData1 = imgData1 > 0 ? imgData1 : 0.0;\n\
    vxc_float4 minData0 = imgData0 < 0 ? imgData0 : 0.0;\n\
    vxc_float4 minData1 = imgData1 < 0 ? imgData1 : 0.0;\n\
    tmpOut0 = maxData0 + paraFp * minData0;\n\
    tmpOut1 = maxData1 + paraFp * minData1;\n\
\n\
    tmpVal0 = convert_int4_rte(tmpOut0 + output_ZP);\n\
    tmpVal1 = convert_int4_rte(tmpOut1 + output_ZP);\n\
\n\
    VXC_DP2x8(outval, tmpVal0, tmpVal1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1),\\\n\
        uniConvertInt32toUint8_2x8);\n\
    VXC_WriteImage2DArray(output, coord, outval, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcParametricReluFp16_Uint8\n\
    (\n\
    image2d_array_t input,\n\
    image2d_array_t para,\n\
    image2d_array_t output\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    int4 coord_para = (int4)(coord.z, 0, 0, 0);\n\
\n\
    vxc_short8 img_s16, para_s16;\n\
    vxc_half8 img_fp16;\n\
    half paraHlf;\n\
    float paraFp;\n\
    vxc_uchar8 outval;\n\
    vxc_float4 imgData0, imgData1;\n\
    vxc_float4 tmpOut0, tmpOut1;\n\
    vxc_int4 tmpVal0, tmpVal1;\n\
\n\
    VXC_ReadImage2DArray(img_s16, input, coord, VXC_5BITOFFSET_XY(0,0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(para_s16, para, coord_para.xy, VXC_5BITOFFSET_XY(0,0),\\\n\
        VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
\n\
    _viv_asm(COPY, img_fp16, img_s16, 16);\n\
    VXC_DP4x4(imgData0, img_fp16, img_fp16, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertDirInt16Fp32_4x4);\n\
    VXC_DP4x4(imgData1, img_fp16, img_fp16, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertEndInt16Fp32_4x4);\n\
\n\
    _viv_asm(COPY, paraHlf, para_s16, 2);\n\
    _viv_asm(CONV, paraFp, paraHlf);\n\
\n\
    vxc_float4 maxData0 = imgData0 > 0 ? imgData0 : 0.0;\n\
    vxc_float4 maxData1 = imgData1 > 0 ? imgData1 : 0.0;\n\
    vxc_float4 minData0 = imgData0 < 0 ? imgData0 : 0.0;\n\
    vxc_float4 minData1 = imgData1 < 0 ? imgData1 : 0.0;\n\
    tmpOut0 = maxData0 + paraFp * minData0;\n\
    tmpOut1 = maxData1 + paraFp * minData1;\n\
\n\
    tmpVal0 = convert_int4_rte(tmpOut0 * outputScale + output_ZP);\n\
    tmpVal1 = convert_int4_rte(tmpOut1 * outputScale + output_ZP);\n\
    VXC_DP2x8(outval, tmpVal0, tmpVal1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1),\\\n\
        uniConvertInt32toUint8_2x8);\n\
    VXC_WriteImage2DArray(output, coord, outval, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcParametricReluFp16_Int16\n\
    (\n\
    image2d_array_t input,\n\
    image2d_array_t para,\n\
    image2d_array_t output\n\
    )\n\
{\n\
    int gidx = get_global_id(0);\n\
    int gidy = get_global_id(1);\n\
    int gidz = get_global_id(2);\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    int4 coord_para = (int4)(coord.z, 0, 0, 0);\n\
\n\
    vxc_short8 img_s16, para_s16;\n\
    vxc_half8 img_fp16;\n\
    half paraHlf;\n\
    float paraFp;\n\
    vxc_float4 p4;\n\
    vxc_short8 outval;\n\
    vxc_float4 imgData0, imgData1;\n\
    vxc_float4 tmpOut0, tmpOut1;\n\
    vxc_int4 tmpVal0, tmpVal1;\n\
\n\
    VXC_ReadImage2DArray(img_s16, input, coord, VXC_5BITOFFSET_XY(0,0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(para_s16, para, coord_para.xy, VXC_5BITOFFSET_XY(0,0),\\\n\
        VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
\n\
    _viv_asm(COPY, img_fp16, img_s16, 16);\n\
    VXC_DP4x4(imgData0, img_fp16, img_fp16, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertDirInt16Fp32_4x4);\n\
    VXC_DP4x4(imgData1, img_fp16, img_fp16, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertEndInt16Fp32_4x4);\n\
\n\
    _viv_asm(COPY, paraHlf, para_s16, 2);\n\
    _viv_asm(CONV, paraFp, paraHlf);\n\
\n\
    vxc_float4 maxData0 = imgData0 > 0 ? imgData0 : 0.0;\n\
    vxc_float4 maxData1 = imgData1 > 0 ? imgData1 : 0.0;\n\
    vxc_float4 minData0 = imgData0 < 0 ? imgData0 : 0.0;\n\
    vxc_float4 minData1 = imgData1 < 0 ? imgData1 : 0.0;\n\
    tmpOut0 = maxData0 + paraFp * minData0;\n\
    tmpOut1 = maxData1 + paraFp * minData1;\n\
    tmpOut0 *= outScaleInt16;\n\
    tmpOut1 *= outScaleInt16;\n\
\n\
    tmpVal0 = convert_int4_rte(tmpOut0);\n\
    tmpVal1 = convert_int4_rte(tmpOut1);\n\
\n\
    VXC_DP2x8(outval, tmpVal0, tmpVal1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1),\\\n\
        uniConvertInt32toUint8_2x8);\n\
    VXC_WriteImage2DArray(output, coord, outval, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcParametricReluInt16_Fp16\n\
    (\n\
    image2d_array_t input,\n\
    image2d_array_t para,\n\
    image2d_array_t output\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    int4 coord_para = (int4)(coord.z, 0, 0, 0);\n\
\n\
    vxc_short8 img_s16, para_s16;\n\
    half paraHlf;\n\
    float paraFp;\n\
    vxc_short8 outval;\n\
    vxc_float4 imgData0, imgData1;\n\
    vxc_float4 tmpOut0, tmpOut1;\n\
    half4 tmpVal0, tmpVal1;\n\
\n\
    VXC_ReadImage2DArray(img_s16, input, coord, VXC_5BITOFFSET_XY(0,0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(para_s16, para, coord_para.xy, VXC_5BITOFFSET_XY(0,0),\\\n\
        VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_DP4x4(imgData0, img_s16, img_s16, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertDirInt16Fp32_4x4);\n\
    VXC_DP4x4(imgData1, img_s16, img_s16, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertEndInt16Fp32_4x4);\n\
\n\
    _viv_asm(COPY, paraHlf, para_s16, 2);\n\
    _viv_asm(CONV, paraFp, paraHlf);\n\
    imgData0 *= inScaleInt16;\n\
    imgData1 *= inScaleInt16;\n\
\n\
    vxc_float4 maxData0 = imgData0 > 0 ? imgData0 : 0.0;\n\
    vxc_float4 maxData1 = imgData1 > 0 ? imgData1 : 0.0;\n\
    vxc_float4 minData0 = imgData0 < 0 ? imgData0 : 0.0;\n\
    vxc_float4 minData1 = imgData1 < 0 ? imgData1 : 0.0;\n\
    tmpOut0 = maxData0 + paraFp * minData0;\n\
    tmpOut1 = maxData1 + paraFp * minData1;\n\
\n\
    _viv_asm(CONV, tmpVal0, tmpOut0);\n\
    _viv_asm(CONV, tmpVal1, tmpOut1);\n\
    VXC_DP2x8(outval, tmpVal0, tmpVal1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertInt32toUint8_2x8);\n\
    VXC_WriteImage2DArray(output, coord, outval, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_prelu_vx*/

static const char vsi_nn_kernel_prelu_i8_i16_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform VXC_512Bits uniPreluInt8Lo_2x8b;\n\
_viv_uniform VXC_512Bits uniPreluInt8Hi_2x8b;\n\
_viv_uniform VXC_512Bits uniPreluInt16_2x8b;\n\
_viv_uniform VXC_512Bits uniPreluInt8_2x8;\n\
_viv_uniform VXC_512Bits uniPreluInt16_4x4;\n\
_viv_uniform VXC_512Bits uniMergeMultiplier_2x8;\n\
_viv_uniform int multiplier;\n\
#if (VX_VERSION==2)\n\
__kernel void vxcParametricRelu_int8_opt\n\
(\n\
    image2d_array_t input,\n\
    image2d_array_t para,\n\
    image2d_array_t output\n\
)\n\
{\n\
    int2 coord = (int2)(get_global_id(0), get_global_id(1));\n\
\n\
    vxc_char16 in, dst;\n\
    vxc_char32 src;\n\
    vxc_short8 para_s16;\n\
    vxc_half8 paraHlf;\n\
    VXC_ReadImage(in, input, coord.xy, VXC_5BITOFFSET_XY(0,0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(para_s16, para, coord.yy, VXC_5BITOFFSET_XY(0,0),\\\n\
        VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, paraHlf, para_s16, 4);\n\
    src.hi = max(in, 0);\n\
    src.lo = min(in, 0);\n\
\n\
    VXC_DP2x8_b_(dst, src.hi, src.lo, paraHlf, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniPreluInt8Lo_2x8b);\n\
    VXC_DP2x8_b_(dst, src.hi, src.lo, paraHlf, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniPreluInt8Hi_2x8b);\n\
    VXC_WriteImage(output, coord, dst, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcParametricRelu_int8_opt1\n\
(\n\
    __read_only  image2d_array_t input,\n\
    __read_only  image2d_array_t para,\n\
    __write_only image2d_array_t output\n\
)\n\
{\n\
    int2 coord = (int2)(get_global_id(0), get_global_id(1));\n\
\n\
    vxc_char16 in, dst;\n\
    vxc_char32 src;\n\
    vxc_short8 para_s16;\n\
    vxc_half8 paraHlf;\n\
    VXC_ReadImage(in, input, coord.xy, VXC_5BITOFFSET_XY(0,0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(para_s16, para, coord.yy, VXC_5BITOFFSET_XY(0,0),\\\n\
        VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, paraHlf, para_s16, 4);\n\
    src.hi = max(in, 0);\n\
    src.lo = min(in, 0);\n\
\n\
    unsigned short src2;\n\
    _viv_asm(COPY, src2, multiplier, 4);\n\
    VXC_DP2x8(paraHlf, paraHlf, src2, VXC_MODIFIER(0, 1, 0, VXC_RM_TowardZero, 0),\\\n\
        uniMergeMultiplier_2x8);\n\
    VXC_DP2x8_b_(dst, src.hi, src.lo, paraHlf, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniPreluInt8Lo_2x8b);\n\
    VXC_DP2x8_b_(dst, src.hi, src.lo, paraHlf, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniPreluInt8Hi_2x8b);\n\
    VXC_WriteImage(output, coord, dst, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcParametricReluInt16_Int16_opt\n\
    (\n\
    image2d_array_t input,\n\
    image2d_array_t para,\n\
    image2d_array_t output\n\
    )\n\
{\n\
    int2 coord = (int2)(get_global_id(0), get_global_id(1));\n\
\n\
    vxc_short8 in, dst;\n\
    vxc_short16 src;\n\
    vxc_short8 para_s16;\n\
    vxc_half8 paraHlf;\n\
    VXC_ReadImage(in, input, coord.xy, VXC_5BITOFFSET_XY(0,0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(para_s16, para, coord.yy, VXC_5BITOFFSET_XY(0,0),\\\n\
        VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, paraHlf, para_s16, 4);\n\
    src.hi = max(in, 0);\n\
    src.lo = min(in, 0);\n\
    VXC_DP2x8_b_(dst, src.hi, src.lo, paraHlf, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniPreluInt16_2x8b);\n\
    VXC_WriteImage(output, coord, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcParametricReluInt16_Int16_opt1\n\
    (\n\
    image2d_array_t input,\n\
    image2d_array_t para,\n\
    image2d_array_t output\n\
    )\n\
{\n\
    int2 coord = (int2)(get_global_id(0), get_global_id(1));\n\
\n\
    vxc_short8 in, dst;\n\
    vxc_short16 src;\n\
    vxc_short8 para_s16;\n\
    vxc_half8 paraHlf;\n\
    VXC_ReadImage(in, input, coord.xy, VXC_5BITOFFSET_XY(0,0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(para_s16, para, coord.yy, VXC_5BITOFFSET_XY(0,0),\\\n\
        VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, paraHlf, para_s16, 4);\n\
    src.hi = max(in, 0);\n\
    src.lo = min(in, 0);\n\
\n\
    unsigned short src2;\n\
    _viv_asm(COPY, src2, multiplier, 4);\n\
    VXC_DP2x8(paraHlf, paraHlf, src2, VXC_MODIFIER(0, 1, 0, VXC_RM_TowardZero, 0),\\\n\
        uniMergeMultiplier_2x8);\n\
    VXC_DP2x8_b_(dst, src.hi, src.lo, paraHlf, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniPreluInt16_2x8b);\n\
    VXC_WriteImage(output, coord, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
#else\n\
\n\
//__kernel void vxcParametricRelu_int8_evis1\n\
__kernel void vxcParametricRelu_int8_opt\n\
(\n\
    image2d_array_t input,\n\
    image2d_array_t para,\n\
    image2d_array_t output\n\
)\n\
{\n\
    int2 coord = (int2)(get_global_id(0), get_global_id(1));\n\
    vxc_char16 in, dst;\n\
    vxc_char16 src0, src1, src;\n\
    vxc_short8 para_s16;\n\
    vxc_half8 paraHlf;\n\
    VXC_ReadImage(in, input, coord.xy, VXC_5BITOFFSET_XY(0,0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(para_s16, para, coord.yy, VXC_5BITOFFSET_XY(0,0),\\\n\
        VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, paraHlf, para_s16, 4);\n\
    src0 = max(in, 0);\n\
    src1 = min(in, 0);\n\
    _viv_asm(COPY, src, src0, 16);\n\
    src.s89abcdef = src1.s01234567;\n\
    VXC_DP2x8(dst, src, paraHlf, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniPreluInt8_2x8);\n\
    _viv_asm(COPY, src, src1, 16);\n\
    src.s01234567 = src0.s89abcdef;\n\
    VXC_DP2x8(dst, src, paraHlf, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniPreluInt8_2x8);\n\
    VXC_WriteImage(output, coord, dst, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
//__kernel void vxcParametricReluInt16_Int16_evis1\n\
__kernel void vxcParametricReluInt16_Int16_opt\n\
    (\n\
    image2d_array_t input,\n\
    image2d_array_t para,\n\
    image2d_array_t output\n\
    )\n\
{\n\
    int2 coord = (int2)(get_global_id(0), get_global_id(1));\n\
    vxc_short8 in, dst;\n\
    vxc_short8 src0, src1, src;\n\
    vxc_short8 para_s16;\n\
    vxc_half8 paraHlf;\n\
    VXC_ReadImage(in, input, coord.xy, VXC_5BITOFFSET_XY(0,0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(para_s16, para, coord.yy, VXC_5BITOFFSET_XY(0,0),\\\n\
        VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, paraHlf, para_s16, 4);\n\
    src0 = max(in, 0);\n\
    src1 = min(in, 0);\n\
    _viv_asm(COPY, src, src0, 16);\n\
    src.s4567 = src1.s0123;\n\
    VXC_DP4x4(dst, src, paraHlf, VXC_MODIFIER(0, 3, 0, VXC_RM_ToNearestEven, 1), uniPreluInt16_4x4);\n\
    _viv_asm(COPY, src, src1, 16);\n\
    src.s0123 = src0.s4567;\n\
    VXC_DP4x4(dst, src, paraHlf, VXC_MODIFIER(4, 7, 0, VXC_RM_ToNearestEven, 1), uniPreluInt16_4x4);\n\
    VXC_WriteImage(output, coord, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
#endif\n\
"; /* end of vsi_nn_kernel_prelu_i8_i16_vx*/

static const char vsi_nn_kernel_prelu_u8_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform VXC_512Bits uniU8SubZP_MulM_PStoF16Lo_2x8;\n\
_viv_uniform VXC_512Bits uniU8SubZP_MulM_PStoF16Hi_2x8;\n\
_viv_uniform VXC_512Bits uniConvertInt32toUint8_2x8;\n\
_viv_uniform VXC_512Bits uniConvertUint8SubZpToFp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertSecUint8SubZpToFp32_4x4;\n\
_viv_uniform int input_ZP;\n\
_viv_uniform float inputScale;\n\
_viv_uniform VXC_512Bits uniF16MulF16_2x8;\n\
_viv_uniform int inputZP;\n\
_viv_uniform int outputZP;\n\
_viv_uniform VXC_512Bits uniS16AddZP_2x8;\n\
\n\
__kernel void vxcParametricReluUint8_Fp16\n\
    (\n\
    image2d_array_t input,\n\
    image2d_array_t para,\n\
    image2d_array_t output\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    int4 coord_para = (int4)(coord.z, 0, 0, 0);\n\
\n\
    vxc_uchar16 img_s16;\n\
    vxc_short8 para_s16;\n\
    half paraHlf;\n\
    float paraFp;\n\
    vxc_short8 outval;\n\
    vxc_float4 imgData0, imgData1;\n\
    float4 tmpOut0, tmpOut1;\n\
    half4 tmpVal0, tmpVal1;\n\
\n\
    VXC_ReadImage2DArray(img_s16, input, coord, VXC_5BITOFFSET_XY(0,0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(para_s16, para, coord_para.xy, VXC_5BITOFFSET_XY(0,0),\\\n\
        VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
\n\
    short zp = input_ZP;\n\
\n\
    VXC_DP4x4(imgData0, img_s16, zp, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertUint8SubZpToFp32_4x4);\n\
    VXC_DP4x4(imgData1, img_s16, zp, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertSecUint8SubZpToFp32_4x4);\n\
\n\
    _viv_asm(COPY, paraHlf, para_s16, 4);\n\
    _viv_asm(CONV, paraFp, paraHlf);\n\
    imgData0 *= inputScale;\n\
    imgData1 *= inputScale;\n\
\n\
    vxc_float4 maxData0 = imgData0 > 0 ? imgData0 : 0.0;\n\
    vxc_float4 maxData1 = imgData1 > 0 ? imgData1 : 0.0;\n\
    vxc_float4 minData0 = imgData0 < 0 ? imgData0 : 0.0;\n\
    vxc_float4 minData1 = imgData1 < 0 ? imgData1 : 0.0;\n\
    tmpOut0 = maxData0 + paraFp * minData0;\n\
    tmpOut1 = maxData1 + paraFp * minData1;\n\
\n\
    _viv_asm(CONV, tmpVal0, tmpOut0);\n\
    _viv_asm(CONV, tmpVal1, tmpOut1);\n\
    VXC_DP2x8(outval, tmpVal0, tmpVal1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertInt32toUint8_2x8);\n\
    VXC_WriteImage2DArray(output, coord, outval, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcParametricRelu_uint8_2d\n\
    (\n\
    __read_only image2d_t           input,\n\
    __read_only image2d_t           param,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    vxc_uchar16 src0, dst;\n\
    vxc_short8  vec0, vec1, vec2;\n\
    vxc_half8   param_h, src2, src3;\n\
    vxc_half16  src;\n\
    vxc_short8  const1 = (vxc_short8)(0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00);\n\
\n\
    int2 coord = (int2)(get_global_id(0), get_global_id(1));\n\
    VXC_ReadImage(src0, input, coord, 0, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(vec0, param, coord.yy, 0, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
\n\
    vxc_uchar16 input_ZP;\n\
    _viv_asm(COPY, input_ZP, inputZP, 4);\n\
    VXC_DP2x8(src2, src0, input_ZP, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
        uniU8SubZP_MulM_PStoF16Lo_2x8);\n\
    VXC_DP2x8(src3, src0, input_ZP, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
        uniU8SubZP_MulM_PStoF16Hi_2x8);\n\
\n\
    vec0 = vec0.s00000000;\n\
    _viv_asm(COPY, vec1, src2, 16);\n\
    vec2 = vec1 >= 0 ? const1 : vec0;\n\
    _viv_asm(COPY, param_h, vec2, 16);\n\
    VXC_DP2x8(vec2, src2, param_h, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniF16MulF16_2x8);\n\
    _viv_asm(COPY, src0, outputZP, 16);\n\
    VXC_DP2x8(dst, vec2, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniS16AddZP_2x8);\n\
\n\
    _viv_asm(COPY, vec1, src3, 16);\n\
    vec2 = vec1 >= 0 ? const1 : vec0;\n\
    _viv_asm(COPY, param_h, vec2, 16);\n\
    VXC_DP2x8(vec2, src3, param_h, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniF16MulF16_2x8);\n\
    VXC_DP2x8(dst, vec2, src0, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniS16AddZP_2x8);\n\
\n\
    VXC_WriteImage(output, coord, dst, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
\n\
__kernel void vxcParametricRelu_uint8tofp16_2d\n\
    (\n\
    __read_only image2d_t           input,\n\
    __read_only image2d_t           param,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    vxc_uchar16 src0, dst;\n\
    vxc_short8  vec0, vec1, vec2;\n\
    vxc_half8   param_h, src2, src3;\n\
    vxc_half16  src;\n\
    vxc_short8  const1 = (vxc_short8)(0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00);\n\
\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(0), get_global_id(1));\n\
    VXC_ReadImage(src0, input, coord.xy, 0, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(vec0, param, coord.yy, 0, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
\n\
    coord.z += 8;\n\
\n\
    vxc_uchar16 input_ZP;\n\
    _viv_asm(COPY, input_ZP, inputZP, 4);\n\
    VXC_DP2x8(src2, src0, input_ZP, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
        uniU8SubZP_MulM_PStoF16Lo_2x8);\n\
    VXC_DP2x8(src3, src0, input_ZP, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
        uniU8SubZP_MulM_PStoF16Hi_2x8);\n\
\n\
    vec0 = vec0.s00000000;\n\
    _viv_asm(COPY, vec1, src2, 16);\n\
    vec2 = vec1 >= 0 ? const1 : vec0;\n\
    _viv_asm(COPY, param_h, vec2, 16);\n\
    VXC_DP2x8(src2, src2, param_h, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniF16MulF16_2x8);\n\
    _viv_asm(COPY, vec2, src2, 16);\n\
    VXC_WriteImage(output, coord.xy, vec2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    _viv_asm(COPY, vec1, src3, 16);\n\
    vec2 = vec1 >= 0 ? const1 : vec0;\n\
    _viv_asm(COPY, param_h, vec2, 16);\n\
    VXC_DP2x8(src3, src3, param_h, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniF16MulF16_2x8);\n\
    _viv_asm(COPY, vec2, src3, 16);\n\
\n\
    VXC_WriteImage(output, coord.zy, vec2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform int outputFl_i8;\n\
_viv_uniform VXC_512Bits uniConvertDirFp16Fp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertEndFp16Fp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertInt32toInt8_2x8;\n\
__kernel void vxcParametricReluFp16_Int8\n\
    (\n\
    image2d_array_t input,\n\
    image2d_array_t para,\n\
    image2d_array_t output\n\
    )\n\
{\n\
    int gidx = get_global_id(0);\n\
    int gidy = get_global_id(1);\n\
    int gidz = get_global_id(2);\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    int4 coord_para = (int4)(coord.z, 0, 0, 0);\n\
\n\
    vxc_short8 img_s16, para_s16;\n\
    vxc_half8 img_fp16;\n\
    half paraHlf;\n\
    float paraFp;\n\
    vxc_float4 p4;\n\
    vxc_char16 outval;\n\
    vxc_float4 imgData0, imgData1;\n\
    vxc_float4 tmpOut0, tmpOut1;\n\
    vxc_int4 tmpVal0, tmpVal1;\n\
\n\
    VXC_ReadImage2DArray(img_s16, input, coord, VXC_5BITOFFSET_XY(0,0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(para_s16, para, coord_para.xy, VXC_5BITOFFSET_XY(0,0),\\\n\
        VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
\n\
    _viv_asm(COPY, img_fp16, img_s16, 16);\n\
    VXC_DP4x4(imgData0, img_fp16, img_fp16, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertDirFp16Fp32_4x4);\n\
    VXC_DP4x4(imgData1, img_fp16, img_fp16, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertEndFp16Fp32_4x4);\n\
\n\
    _viv_asm(COPY, paraHlf, para_s16, 2);\n\
    _viv_asm(CONV, paraFp, paraHlf);\n\
\n\
    vxc_float4 maxData0 = imgData0 > 0 ? imgData0 : 0.0;\n\
    vxc_float4 maxData1 = imgData1 > 0 ? imgData1 : 0.0;\n\
    vxc_float4 minData0 = imgData0 < 0 ? imgData0 : 0.0;\n\
    vxc_float4 minData1 = imgData1 < 0 ? imgData1 : 0.0;\n\
    tmpOut0 = maxData0 + paraFp * minData0;\n\
    tmpOut1 = maxData1 + paraFp * minData1;\n\
    tmpOut0 *= outputFl_i8;\n\
    tmpOut1 *= outputFl_i8;\n\
\n\
    tmpVal0 = convert_int4_rte(tmpOut0);\n\
    tmpVal1 = convert_int4_rte(tmpOut1);\n\
\n\
    VXC_DP2x8(outval, tmpVal0, tmpVal1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1),\\\n\
        uniConvertInt32toInt8_2x8);\n\
    VXC_WriteImage2DArray(output, coord, outval, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_prelu_u8_vx*/

static const char vsi_nn_kernel_random_multinomial_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform int iter;\n\
_viv_uniform int stride;\n\
_viv_uniform int class_max_iter;\n\
_viv_uniform int class_max_stride;\n\
_viv_uniform float re_rand_max;\n\
\n\
_viv_uniform VXC_512Bits uniHorzSubMaxFp16_2x8;\n\
_viv_uniform VXC_512Bits uniConvertFstFp16Fp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertSecFp16Fp32_4x4;\n\
\n\
uint4 _philox4x32bumpkey(uint4 key)\n\
{\n\
    uint4 mask = (uint4)((uint)0x9E3779B9, (uint)0xBB67AE85, 0, 0);\n\
    //key.x += ((uint)0x9E3779B9);\n\
    //key.y += ((uint)0xBB67AE85);\n\
    key += mask;\n\
    return key;\n\
}\n\
\n\
uint mulhilo32(uint a, uint b, uint* hip)\n\
{\n\
    uint product = (uint)(a * b);\n\
    *hip = mul_hi(a, b);\n\
    return product;\n\
}\n\
\n\
uint4 _philox4x32round(uint4 ctr, uint4 key)\n\
{\n\
    uint hi0;\n\
    uint hi1;\n\
    uint PHILOX_M4x32_0 = ((uint)0xD2511F53);\n\
    uint PHILOX_M4x32_1 = ((uint)0xCD9E8D57);\n\
    uint lo0 = mulhilo32(PHILOX_M4x32_0, ctr.x, &hi0);\n\
    uint lo1 = mulhilo32(PHILOX_M4x32_1, ctr.z, &hi1);\n\
    uint4 out = (uint4)(hi1^ctr.y^key.x, lo1, hi0^ctr.w^key.y, lo0);\n\
    return out;\n\
}\n\
\n\
uint4 philox4x32_R_10(uint4 ctr, uint4 key)\n\
{\n\
    uint i;\n\
    ctr = _philox4x32round(ctr, key);\n\
    for (i = 1; i < 10; i++)\n\
    {\n\
        key = _philox4x32bumpkey(key);\n\
        ctr = _philox4x32round(ctr, key);\n\
    }\n\
    return ctr;\n\
}\n\
\n\
__kernel void vxcRandom_generate(\n\
    __read_only image2d_array_t   seeds,\n\
    __write_only image2d_array_t  output)\n\
{\n\
    int gidx = get_global_id(0);\n\
    int gidy = get_global_id(1);\n\
    int4 coord = (int4)(gidx << 1, gidy, 0, 0);\n\
    uint4 key = read_imageui(seeds, coord);\n\
\n\
    uint4 ctr = (uint4)(0);\n\
    float4 result = 0;\n\
\n\
    coord.x = gidx * stride;\n\
    for(int i = 0; i < iter; i++)\n\
    {\n\
        ctr = philox4x32_R_10(ctr, key);\n\
        result = convert_float4(ctr) * re_rand_max;\n\
        write_imagef(output, coord, result);\n\
        coord.x += 4;\n\
    }\n\
}\n\
\n\
// N times of 8\n\
// x dim = 1\n\
__kernel void vxcRandom_sum_fp16(\n\
    __read_only image2d_array_t   input,\n\
    vx_array_float  output)\n\
{\n\
    int gidx = get_global_id(0);\n\
    int gidy = get_global_id(1);\n\
    int4 coord = (int4)(gidx, gidy, 0, 0);\n\
\n\
    vxc_half8 maxData, data;\n\
    vxc_short8 src0 = (vxc_short8)(0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00);\n\
    uint4 ctr = (uint4)(0);\n\
    float4 dst0, dst1;\n\
    float4 one = (float4)(1, 1, 1, 1);\n\
    _viv_asm(COPY, maxData, src0, 16);\n\
    float tmp = 0;\n\
    int offset = gidy * class_max_stride;\n\
    __global float* cdfPtr = (__global float*)output.item + offset;\n\
\n\
    for(int i = 0; i < class_max_iter; i++)\n\
    {\n\
        VXC_ReadImage(src0, input, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        coord.x += 8;\n\
        _viv_asm(COPY, data, src0, 16);\n\
        //VXC_DP2x8(sum, data, data, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\n\
        //        uniHorzSumFp16_8x2);\n\
        VXC_HorzMax3(data, data, VXC_MODIFIER(0, 5, 0,VXC_RM_TowardZero, 0));\n\
        VXC_HorzMax3(data, data, VXC_MODIFIER(0, 3, 0,VXC_RM_TowardZero, 0));\n\
        VXC_HorzMax3(data, data, VXC_MODIFIER(0, 1, 0,VXC_RM_TowardZero, 0));\n\
        VXC_HorzMax3(data, data, VXC_MODIFIER(0, 0, 0,VXC_RM_TowardZero, 0));\n\
        VXC_VertMax3(maxData, maxData, maxData, data, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
    }\n\
    coord.x = 0;\n\
    for(int i = 0; i < class_max_iter; i++)\n\
    {\n\
        float4 val0, val1;\n\
        VXC_ReadImage(src0, input, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        coord.x += 8;\n\
        _viv_asm(COPY, data, src0, 16);\n\
        VXC_DP2x8(data, data, maxData, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\n\
                uniHorzSubMaxFp16_2x8);\n\
        VXC_DP4x4(val0, data, data, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertFstFp16Fp32_4x4);\n\
        VXC_DP4x4(val1, data, data, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertSecFp16Fp32_4x4);\n\
        val0 = exp(val0);\n\
        val1 = exp(val1);\n\
        tmp = dot(val0, one);\n\
        dst0 = (float4)(val0.x, (val0.x + val0.y), (val0.x + val0.y + val0.z), tmp);\n\
        dst1 = (float4)((tmp + val1.x), (tmp + val1.x + val1.y),\n\
                 (tmp + val1.x + val1.y + val1.z), tmp + dot(val1, one));\n\
        vstore4(dst0, 0, cdfPtr);\n\
        vstore4(dst1, 1, cdfPtr);\n\
        cdfPtr += 8;\n\
    }\n\
}\n\
\n\
uint upper_bound(float* a, int n, float x) {\n\
    uint l = 0;\n\
    uint h = n;\n\
    while (l < h) {\n\
        int mid = (l + h) / 2;\n\
        if (x >= a[mid]) {\n\
            l = mid + 1;\n\
        } else {\n\
            h = mid;\n\
        }\n\
    }\n\
    return l;\n\
}\n\
\n\
// one thread calculate 4\n\
__kernel void vxcRandom_multinomial(\n\
    __read_only image2d_array_t   randoms,\n\
                 vx_array_float   cdfs,\n\
   __write_only image2d_array_t   output,\n\
                 int class_size, int max_stride)\n\
{\n\
    int gidx = get_global_id(0);\n\
    int gidy = get_global_id(1);\n\
    int4 coord = (int4)(gidx, gidy, 0, 0);\n\
\n\
    int offset = gidy * max_stride;\n\
    __global float* cdfPtr = (__global float*)cdfs.item + offset;\n\
\n\
    float4 ran = read_imagef(randoms, coord);\n\
    float total = cdfPtr[class_size - 1];\n\
    float4 target = ran * total;\n\
\n\
    uint4 out_class = (uint4)(0);\n\
    out_class.x = upper_bound(cdfPtr, class_size, target.x);\n\
    out_class.y = upper_bound(cdfPtr, class_size, target.y);\n\
    out_class.z = upper_bound(cdfPtr, class_size, target.z);\n\
    out_class.w = upper_bound(cdfPtr, class_size, target.w);\n\
\n\
    write_imageui(output, coord, out_class);\n\
}\n\
\n\
__kernel void vxcRandom_sum_fp32(\n\
    __read_only image2d_array_t   input,\n\
    vx_array_float  output)\n\
{\n\
    int gidx = get_global_id(0);\n\
    int gidy = get_global_id(1);\n\
    int4 coord = (int4)(gidx, gidy, 0, 0);\n\
\n\
    vxc_float4 src0, data;\n\
    float maxData0 = FLT_MIN, maxData1 = FLT_MIN;\n\
    uint4 ctr = (uint4)(0);\n\
    float4 dst;\n\
    float4 one = (float4)(1, 1, 1, 1);\n\
\n\
    float tmp = 0;\n\
    int offset = gidy * class_max_stride;\n\
    __global float* cdfPtr = (__global float*)output.item + offset;\n\
\n\
    for(int i = 0; i < class_max_iter; i++)\n\
    {\n\
        src0 = read_imagef(input, coord);\n\
        coord.x += 4;\n\
        maxData0 = max(src0.x, src0.y);\n\
        maxData1 = max(src0.z, src0.w);\n\
        maxData0 = max(maxData0, maxData1);\n\
    }\n\
    coord.x = 0;\n\
    float4 maxData = (float4)(maxData0, maxData0, maxData0, maxData0);\n\
    for(int i = 0; i < class_max_iter; i++)\n\
    {\n\
        float4 val;\n\
        src0 = read_imagef(input, coord);\n\
        coord.x += 4;\n\
        data = src0 - maxData;\n\
        val = exp(data);\n\
        tmp = dot(val, one);\n\
        dst = (float4)(val.x, (val.x + val.y), (val.x + val.y + val.z), tmp);\n\
        vstore4(dst, 0, cdfPtr);\n\
        cdfPtr += 4;\n\
    }\n\
}"; /* end of vsi_nn_kernel_random_multinomial_vx*/

static const char vsi_nn_kernel_relational_ops_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform VXC_512Bits uniConvertFstFp16Fp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertSecFp16Fp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertInt32toUint8_2x8;\n\
\n\
#if 0\n\
__kernel __attribute__((reqd_work_group_size(8, 1, 1))) void vxcTensorRelation_Gt_Int16(\n\
    image2d_array_t input0,\n\
    image2d_array_t input1,\n\
    image2d_array_t output)\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_short8 src0, src1;\n\
    vxc_short8 tmpDst;\n\
    vxc_short8 dst;\n\
    VXC_ReadImage2DArray(src0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    //dst = isgreater(src0, src1);\n\
    dst = src0 > src1;\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
#endif\n\
\n\
#define TENSORRELATION(name0, name1, input_type, copy_type, output_type, out_copy_type, cmp_op) \\\n\
    __kernel void vxcTensorRelation_##name0##_##name1( \\\n\
    __read_only  image2d_array_t in0, \\\n\
    __read_only  image2d_array_t in1, \\\n\
    __write_only image2d_array_t output) \\\n\
{\\\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\\\n\
    input_type vA;\\\n\
    copy_type  src0;\\\n\
    input_type vB;\\\n\
    copy_type  src1;\\\n\
    VXC_ReadImage2DArray(vA,in0,coord,VXC_5BITOFFSET_XY(0,0),VXC_MODIFIER(0,7,0,VXC_RM_TowardZero,0));\\\n\
    _viv_asm(COPY, src0, vA, 16); \\\n\
    VXC_ReadImage2DArray(vB,in1,coord,VXC_5BITOFFSET_XY(0,0),VXC_MODIFIER(0,7,0,VXC_RM_TowardZero,0));\\\n\
    _viv_asm(COPY, src1, vB, 16); \\\n\
    output_type dst; \\\n\
    dst = (src0)cmp_op(src1); \\\n\
    out_copy_type data; \\\n\
    _viv_asm(COPY, data, dst, 16); \\\n\
    VXC_WriteImage2DArray(output, coord, data, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
//             name0, name1, input_type, copy_type, output_type, out_copy_type, cmp_op\n\
TENSORRELATION(Gt,    Int8,  vxc_char8,  vxc_char8,  vxc_char8,  vxc_char8,  >)\n\
TENSORRELATION(Gte,   Int8,  vxc_char8, vxc_char8,  vxc_char8,  vxc_char8,  >=)\n\
TENSORRELATION(Ls,    Int8,  vxc_char8, vxc_char8,  vxc_char8,  vxc_char8,  <)\n\
TENSORRELATION(Lse,   Int8,  vxc_char8, vxc_char8,  vxc_char8,  vxc_char8,  <=)\n\
TENSORRELATION(Ne,    Int8,  vxc_char8, vxc_char8,  vxc_char8,  vxc_char8,  !=)\n\
TENSORRELATION(E,     Int8,  vxc_char8, vxc_char8,  vxc_char8,  vxc_char8,  ==)\n\
\n\
TENSORRELATION(Gt,    Uint8,  vxc_uchar8,  vxc_uchar8,  vxc_char8,  vxc_uchar8,  >)\n\
TENSORRELATION(Gte,   Uint8,  vxc_uchar8, vxc_uchar8,  vxc_char8,  vxc_uchar8,  >=)\n\
TENSORRELATION(Ls,    Uint8,  vxc_uchar8, vxc_uchar8,  vxc_char8,  vxc_uchar8,  <)\n\
TENSORRELATION(Lse,   Uint8,  vxc_uchar8, vxc_uchar8,  vxc_char8,  vxc_uchar8,  <=)\n\
TENSORRELATION(Ne,    Uint8,  vxc_uchar8, vxc_uchar8,  vxc_char8,  vxc_uchar8,  !=)\n\
TENSORRELATION(E,     Uint8,  vxc_uchar8, vxc_uchar8,  vxc_char8,  vxc_uchar8,  ==)\n\
\n\
TENSORRELATION(Gt,    Int16,  vxc_short8, vxc_short8,  vxc_short8,  vxc_short8,  >)\n\
TENSORRELATION(Gte,   Int16,  vxc_short8, vxc_short8,  vxc_short8,  vxc_short8,  >=)\n\
TENSORRELATION(Ls,    Int16,  vxc_short8, vxc_short8,  vxc_short8,  vxc_short8,  <)\n\
TENSORRELATION(Lse,   Int16,  vxc_short8, vxc_short8,  vxc_short8,  vxc_short8,  <=)\n\
TENSORRELATION(Ne,    Int16,  vxc_short8, vxc_short8,  vxc_short8,  vxc_short8,  !=)\n\
TENSORRELATION(E,     Int16,  vxc_short8, vxc_short8,  vxc_short8,  vxc_short8,  ==)\n\
\n\
_viv_uniform VXC_512Bits uniMulShortMinus1toFp16_2x8;\n\
\n\
#define TENSORCMPHALF(name0, name1, cmp_op) \\\n\
    __kernel void vxcTensorRelation_##name0##_##name1( \\\n\
    __read_only  image2d_array_t in0, \\\n\
    __read_only  image2d_array_t in1, \\\n\
    __write_only image2d_array_t output) \\\n\
{\\\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\\\n\
    vxc_short8 vA;\\\n\
    vxc_half8  src0;\\\n\
    vxc_short8 vB;\\\n\
    vxc_half8  src1;\\\n\
    VXC_ReadImage2DArray(vA,in0,coord,VXC_5BITOFFSET_XY(0,0),VXC_MODIFIER(0,7,0,VXC_RM_TowardZero,0));\\\n\
    _viv_asm(COPY, src0, vA, 16); \\\n\
    VXC_ReadImage2DArray(vB,in1,coord,VXC_5BITOFFSET_XY(0,0),VXC_MODIFIER(0,7,0,VXC_RM_TowardZero,0));\\\n\
    _viv_asm(COPY, src1, vB, 16); \\\n\
    vxc_float4 x0, x1; \\\n\
    vxc_float4 y0, y1; \\\n\
    VXC_DP4x4(x0, src0, src0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertFstFp16Fp32_4x4); \\\n\
    VXC_DP4x4(x1, src0, src0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertSecFp16Fp32_4x4); \\\n\
    VXC_DP4x4(y0, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertFstFp16Fp32_4x4); \\\n\
    VXC_DP4x4(y1, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertSecFp16Fp32_4x4); \\\n\
    vxc_int4 dst0, dst1; \\\n\
    dst0 = (x0)cmp_op(y0); \\\n\
    dst1 = (x1)cmp_op(y1); \\\n\
    vxc_short8 dst; \\\n\
    VXC_DP2x8(dst,dst0,dst1,VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),uniConvertInt32toUint8_2x8);\\\n\
    vxc_half8 tmpOut; \\\n\
    VXC_DP2x8(tmpOut,dst,dst,VXC_MODIFIER(0,7,0,VXC_RM_TowardZero, 0),uniMulShortMinus1toFp16_2x8); \\\n\
    _viv_asm(COPY, dst, tmpOut, 16); \\\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
\n\
//          name0, name1, input_type, copy_type, output_type, out_copy_type, cmp_op\n\
TENSORCMPHALF(Gt,    Fp16,  >)\n\
TENSORCMPHALF(Gte,   Fp16,  >=)\n\
TENSORCMPHALF(Ls,    Fp16,  <)\n\
TENSORCMPHALF(Lse,   Fp16,  <=)\n\
TENSORCMPHALF(Ne,    Fp16,  !=)\n\
TENSORCMPHALF(E,     Fp16,  ==)"; /* end of vsi_nn_kernel_relational_ops_vx*/

static const char vsi_nn_kernel_relu_keras_header_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform VXC_512Bits uniConvFP16toFP32_Lo_4x4;\n\
_viv_uniform VXC_512Bits uniConvFP16toFP32_Hi_4x4;\n\
_viv_uniform VXC_512Bits uniExtractHalf8_2x8;\n\
_viv_uniform VXC_512Bits uniExtractInteger_2x8;\n\
_viv_uniform VXC_512Bits uniConvBF16toF32_Part0_2x8;\n\
_viv_uniform VXC_512Bits uniConvBF16toF32_Part1_2x8;\n\
_viv_uniform VXC_512Bits uniPackedBF16_2x8;\n\
_viv_uniform VXC_512Bits uniConvIntegertoFP32_Lo_4x4;\n\
_viv_uniform VXC_512Bits uniConvIntegertoFP32_Hi_4x4;\n\
_viv_uniform float offset;\n\
_viv_uniform float input_scale;\n\
_viv_uniform float inputTail;\n\
_viv_uniform float output_scale;\n\
_viv_uniform float outputZP;\n\
\n\
float4 I8toF32_Lo(vxc_char8 src)\n\
{\n\
    float4 dst;\n\
\n\
    VXC_DP4x4(dst, src, src, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvIntegertoFP32_Lo_4x4);\n\
    dst *= input_scale;\n\
    return dst;\n\
}\n\
\n\
float4 I8toF32_Hi(vxc_char8 src)\n\
{\n\
    float4 dst;\n\
\n\
    VXC_DP4x4(dst, src, src, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvIntegertoFP32_Hi_4x4);\n\
    dst *= input_scale;\n\
    return dst;\n\
}\n\
\n\
float4 U8toF32_Lo(vxc_uchar8 src)\n\
{\n\
    float4 dst;\n\
\n\
    VXC_DP4x4(dst, src, src, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvIntegertoFP32_Lo_4x4);\n\
    dst = dst * input_scale + inputTail;\n\
    return dst;\n\
}\n\
\n\
float4 U8toF32_Hi(vxc_uchar8 src)\n\
{\n\
    float4 dst;\n\
\n\
    VXC_DP4x4(dst, src, src, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvIntegertoFP32_Hi_4x4);\n\
    dst = dst * input_scale + inputTail;\n\
    return dst;\n\
}\n\
\n\
float4 I16toF32_Lo(vxc_short8 src)\n\
{\n\
    float4 dst;\n\
\n\
    VXC_DP4x4(dst, src, src, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvIntegertoFP32_Lo_4x4);\n\
    dst *= input_scale;\n\
    return dst;\n\
}\n\
\n\
float4 I16toF32_Hi(vxc_short8 src)\n\
{\n\
    float4 dst;\n\
\n\
    VXC_DP4x4(dst, src, src, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvIntegertoFP32_Hi_4x4);\n\
    dst *= input_scale;\n\
    return dst;\n\
}\n\
\n\
float4 F16toF32_Lo(vxc_ushort8 src)\n\
{\n\
    vxc_half8 srcHalf;\n\
    float4 dst;\n\
\n\
    _viv_asm(COPY, srcHalf, src, 16);\n\
    VXC_DP4x4(dst, srcHalf, srcHalf, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvFP16toFP32_Lo_4x4);\n\
    return dst;\n\
}\n\
\n\
float4 F16toF32_Hi(vxc_ushort8 src)\n\
{\n\
    vxc_half8 srcHalf;\n\
    float4 dst;\n\
\n\
    _viv_asm(COPY, srcHalf, src, 16);\n\
    VXC_DP4x4(dst, srcHalf, srcHalf, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvFP16toFP32_Hi_4x4);\n\
    return dst;\n\
}\n\
\n\
float4 BF16toF32_Lo(vxc_ushort8 src)\n\
{\n\
    vxc_ushort8 srcA;\n\
    float4 dst;\n\
\n\
    vxc_short8 zero = (vxc_short8)(0, 0, 0, 0, 0, 0, 0, 0);\n\
    VXC_DP2x8(srcA, src, zero, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part0_2x8);\n\
    _viv_asm(COPY, dst, srcA, 16);\n\
\n\
    return dst;\n\
}\n\
\n\
float4 BF16toF32_Hi(vxc_ushort8 src)\n\
{\n\
    vxc_ushort8 srcA;\n\
    float4 dst;\n\
\n\
    vxc_short8 zero = (vxc_short8)(0, 0, 0, 0, 0, 0, 0, 0);\n\
    VXC_DP2x8(srcA, src, zero, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniConvBF16toF32_Part1_2x8);\n\
    _viv_asm(COPY, dst, srcA, 16);\n\
\n\
    return dst;\n\
}\n\
\n\
vxc_ushort8 F32toF16(float4 src0, float4 src1)\n\
{\n\
    half4 srcHalf0, srcHalf1;\n\
    vxc_half8 dst0;\n\
    vxc_ushort8 dst;\n\
\n\
    _viv_asm(CONV, srcHalf0, src0);\n\
    _viv_asm(CONV, srcHalf1, src1);\n\
\n\
    VXC_DP2x8(dst0, srcHalf0, srcHalf1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniExtractHalf8_2x8);\n\
    _viv_asm(COPY, dst, dst0, 16);\n\
    return dst;\n\
}\n\
\n\
vxc_ushort8 F32toBF16(float4 src0, float4 src1)\n\
{\n\
    vxc_ushort8 srcA, srcB;\n\
    vxc_ushort8 dst;\n\
\n\
    _viv_asm(COPY, srcA, src0, 16);\n\
    _viv_asm(COPY, srcB, src1, 16);\n\
    VXC_DP2x8(dst, srcA, srcB, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniPackedBF16_2x8);\n\
    return dst;\n\
}\n\
\n\
vxc_char8 F32toI8(float4 src0, float4 src1)\n\
{\n\
    int4 srcInt0, srcInt1;\n\
    vxc_char8 dst;\n\
\n\
    src0 *= output_scale;\n\
    src1 *= output_scale;\n\
    _viv_asm(CONV_RTE, srcInt0, src0);\n\
    _viv_asm(CONV_RTE, srcInt1, src1);\n\
\n\
    VXC_DP2x8(dst, srcInt0, srcInt1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniExtractInteger_2x8);\n\
    return dst;\n\
}\n\
\n\
vxc_short8 F32toI16(float4 src0, float4 src1)\n\
{\n\
    int4 srcInt0, srcInt1;\n\
    vxc_short8 dst;\n\
\n\
    src0 *= output_scale;\n\
    src1 *= output_scale;\n\
    _viv_asm(CONV_RTE, srcInt0, src0);\n\
    _viv_asm(CONV_RTE, srcInt1, src1);\n\
\n\
    VXC_DP2x8(dst, srcInt0, srcInt1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniExtractInteger_2x8);\n\
    return dst;\n\
}\n\
\n\
vxc_uchar8 F32toU8(float4 src0, float4 src1)\n\
{\n\
    int4 srcInt0, srcInt1;\n\
    vxc_uchar8 dst;\n\
\n\
    src0 = src0 * output_scale + outputZP;\n\
    src1 = src1 * output_scale + outputZP;\n\
    _viv_asm(CONV_RTE, srcInt0, src0);\n\
    _viv_asm(CONV_RTE, srcInt1, src1);\n\
\n\
    VXC_DP2x8(dst, srcInt0, srcInt1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniExtractInteger_2x8);\n\
    return dst;\n\
}\n\
"; /* end of vsi_nn_kernel_relu_keras_header_vx*/

static const char vsi_nn_kernel_relu_keras_internal_vx[] = "\n\
#define TENSOR_KERAS_RELU(src_type_name, dst_type_name, tensor_dims, image_type, \\\n\
    convert2FP32_Func, convert2DstType_Func, src_type, dst_type) \\\n\
__kernel void vxKerasRelu_##src_type_name##to##dst_type_name##tensor_dims( \\\n\
__read_only  image2d_array_t  input, \\\n\
__write_only image2d_array_t  output, \\\n\
                    float     alpha, \\\n\
                    float     max_value, \\\n\
                    float     threshold \\\n\
                    ) \\\n\
{ \\\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0); \\\n\
    src_type src; \\\n\
 \\\n\
    VXC_Read##image_type(src, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    float4 dataA = convert2FP32_Func##_Lo(src); \\\n\
    float4 dataB = convert2FP32_Func##_Hi(src); \\\n\
    float4 dataA0 = dataA < threshold ? threshold : dataA; \\\n\
    dataA0 = dataA0 > max_value ? max_value : dataA0; \\\n\
    float4 dataB0 = dataB < threshold ? threshold : dataB; \\\n\
    dataB0 = dataB0 > max_value ? max_value : dataB0; \\\n\
 \\\n\
    float4 dataA1 = dataA * alpha - offset; \\\n\
    float4 dataB1 = dataB * alpha - offset; \\\n\
    float4 dst0 = dataA  < threshold ? dataA1 : dataA0; \\\n\
    float4 dst1 = dataB  < threshold ? dataB1 : dataB0; \\\n\
 \\\n\
    dst_type result = convert2DstType_Func(dst0, dst1); \\\n\
 \\\n\
    VXC_Write##image_type(output, coord, result, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
TENSOR_KERAS_RELU(F16,  F16,  _3D, Image2DArray, F16toF32,  F32toF16,  vxc_ushort8, vxc_ushort8)\n\
TENSOR_KERAS_RELU(F16,  I16,  _3D, Image2DArray, F16toF32,  F32toI16,  vxc_ushort8, vxc_short8)\n\
TENSOR_KERAS_RELU(F16,  I8,   _3D, Image2DArray, F16toF32,  F32toI8,   vxc_ushort8, vxc_char8)\n\
TENSOR_KERAS_RELU(F16,  U8,   _3D, Image2DArray, F16toF32,  F32toU8,   vxc_ushort8, vxc_uchar8)\n\
TENSOR_KERAS_RELU(BF16, BF16, _3D, Image2DArray, BF16toF32, F32toBF16, vxc_ushort8, vxc_ushort8)\n\
\n\
TENSOR_KERAS_RELU(I16,  I16,  _3D, Image2DArray, I16toF32,  F32toI16,  vxc_short8,  vxc_short8)\n\
TENSOR_KERAS_RELU(I16,  F16,  _3D, Image2DArray, I16toF32,  F32toF16,  vxc_short8,  vxc_ushort8)\n\
TENSOR_KERAS_RELU(I8,   I8,   _3D, Image2DArray, I8toF32,   F32toI8,   vxc_char8,   vxc_char8)\n\
TENSOR_KERAS_RELU(I8,   F16,  _3D, Image2DArray, I8toF32,   F32toF16,  vxc_char8,   vxc_ushort8)\n\
TENSOR_KERAS_RELU(U8,   U8,   _3D, Image2DArray, U8toF32,   F32toU8,   vxc_uchar8,  vxc_uchar8)\n\
TENSOR_KERAS_RELU(U8,   F16,  _3D, Image2DArray, U8toF32,   F32toF16,  vxc_uchar8,  vxc_ushort8)\n\
\n\
TENSOR_KERAS_RELU(F16,  F16,  _2D, Image,        F16toF32,  F32toF16,  vxc_ushort8, vxc_ushort8)\n\
TENSOR_KERAS_RELU(F16,  I16,  _2D, Image,        F16toF32,  F32toI16,  vxc_ushort8, vxc_short8)\n\
TENSOR_KERAS_RELU(F16,  I8,   _2D, Image,        F16toF32,  F32toI8,   vxc_ushort8, vxc_char8)\n\
TENSOR_KERAS_RELU(F16,  U8,   _2D, Image,        F16toF32,  F32toU8,   vxc_ushort8, vxc_uchar8)\n\
TENSOR_KERAS_RELU(BF16, BF16, _2D, Image,        BF16toF32, F32toBF16, vxc_ushort8, vxc_ushort8)\n\
TENSOR_KERAS_RELU(I16,  I16,  _2D, Image,        I16toF32,  F32toI16,  vxc_short8,  vxc_short8)\n\
TENSOR_KERAS_RELU(I16,  F16,  _2D, Image,        I16toF32,  F32toF16,  vxc_short8,  vxc_ushort8)\n\
TENSOR_KERAS_RELU(I8,   I8,   _2D, Image,        I8toF32,   F32toI8,   vxc_char8,   vxc_char8)\n\
TENSOR_KERAS_RELU(I8,   F16,  _2D, Image,        I8toF32,   F32toF16,  vxc_char8,   vxc_ushort8)\n\
TENSOR_KERAS_RELU(U8,   U8,   _2D, Image,        U8toF32,   F32toU8,   vxc_uchar8,  vxc_uchar8)\n\
TENSOR_KERAS_RELU(U8,   F16,  _2D, Image,        U8toF32,   F32toF16,  vxc_uchar8,  vxc_ushort8)\n\
"; /* end of vsi_nn_kernel_relu_keras_internal_vx*/

static const char vsi_nn_kernel_resize_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
//--------------------------resize-------------------------\n\
_viv_uniform VXC_512Bits uniPackEvenData_2x8;\n\
__kernel void resize_16bits_downsample_quarter\n\
    (\n\
    __read_only image2d_array_t input,\n\
    __write_only image2d_array_t output\n\
    )\n\
{\n\
    int2 coord = (int2)(get_global_id(0), get_global_id(1));\n\
    vxc_short8 src0, src1;\n\
    VXC_ReadImage(src0, input, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(src1, input, coord.xy, VXC_5BITOFFSET_XY(8, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    coord = coord >> 1;\n\
    VXC_DP2x8(src0, src0, src1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardInf, 0), uniPackEvenData_2x8);\n\
    VXC_WriteImage(output, coord, src0, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void resize_8bits_downsample_quarter\n\
    (\n\
    __read_only image2d_array_t input,\n\
    __write_only image2d_array_t output\n\
    )\n\
{\n\
    int2 coord = (int2)(get_global_id(0), get_global_id(1));\n\
    vxc_char16 src0;\n\
    vxc_char8 dst;\n\
    VXC_ReadImage(src0, input, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    coord = coord >> 1;\n\
    dst  = src0.s02468ace;\n\
    VXC_WriteImage(output, coord, dst, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_resize_vx*/

static const char vsi_nn_kernel_reverse_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
/********************************************tensor reverse*****************************************/\n\
_viv_uniform int cur_axis_sz_sub1;\n\
__kernel void tensorReverse_axis0_fp16(\n\
    __read_only     image2d_array_t input,\n\
    __write_only    image2d_array_t output)\n\
{\n\
    int2 coord = (int2)(get_global_id(0), get_global_id(1));\n\
    vxc_short8 vec0;\n\
    VXC_ReadImage(vec0, input, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    coord.y = cur_axis_sz_sub1 - coord.y;\n\
\n\
    VXC_WriteImage(output, coord.xy, vec0, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
"; /* end of vsi_nn_kernel_reverse_vx*/

static const char vsi_nn_kernel_roi_align_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
__kernel void vxcRoi_align(\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_array_t  output)\n\
{\n\
\n\
}\n\
"; /* end of vsi_nn_kernel_roi_align_vx*/

static const char vsi_nn_kernel_scale_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
//--------------------------scale-------------------------\n\
_viv_uniform VXC_512Bits uniExtractHalf8_2x8;\n\
_viv_uniform VXC_512Bits uniFp16MulFp16ToFp32_Lo_4x4;\n\
_viv_uniform VXC_512Bits uniFp16MulFp16ToFp32_Hi_4x4;\n\
__kernel void scale_fp16\n\
    (\n\
    __read_only     image2d_array_t input,\n\
    __read_only     image2d_array_t weights,\n\
    __read_only     image2d_array_t biases,\n\
    __write_only    image2d_array_t output\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), 0, 0);\n\
    vxc_short8 vec0, vec1;\n\
    vxc_half8  src0;\n\
    vxc_half8  w0;\n\
    vxc_float4 b0, b1;\n\
    vxc_float4 dst0, dst1;\n\
    VXC_ReadImage(vec0, input, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, src0, vec0, 16);\n\
    VXC_ReadImage(vec1, weights, coord.xw, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, w0, vec1, 16);\n\
\n\
    coord.z = coord.x + 4;\n\
\n\
    b0 = read_imagef(biases, coord.xwww);\n\
    b1 = read_imagef(biases, coord.zwww);\n\
\n\
    VXC_DP4x4(dst0, src0, w0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniFp16MulFp16ToFp32_Lo_4x4);\n\
    VXC_DP4x4(dst1, src0, w0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniFp16MulFp16ToFp32_Hi_4x4);\n\
    dst0 += b0;\n\
    dst1 += b1;\n\
\n\
    half4 t0, t1;\n\
\n\
    _viv_asm(CONV, t0, dst0);\n\
    _viv_asm(CONV, t1, dst1);\n\
\n\
    VXC_DP2x8(w0, t0, t1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniExtractHalf8_2x8);\n\
    _viv_asm(COPY, vec0, w0, 16);\n\
\n\
    VXC_WriteImage(output, coord.xy, vec0, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_scale_vx*/

static const char vsi_nn_kernel_select_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
__kernel void vxcTensorSelect_int8(\n\
    __read_only image2d_array_t   condition,\n\
    __read_only image2d_array_t   input0,\n\
    __read_only image2d_array_t   input1,\n\
    __write_only image2d_array_t  output)\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_char16 src0, src1, dst;\n\
    vxc_char16 value;\n\
    VXC_ReadImage2DArray(src0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(value, condition, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    //dst = select(src1, src0, value); // result = c ? b : a\n\
    //dst = select(src0, src1, value); // result = c ? b : a\n\
    dst = (value != 0 ? src0 : src1);\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcTensorSelect_uint8(\n\
    __read_only image2d_array_t   condition,\n\
    __read_only image2d_array_t   input0,\n\
    __read_only image2d_array_t   input1,\n\
    __write_only image2d_array_t  output)\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_uchar16 src0, src1, dst;\n\
    vxc_uchar16 value;\n\
    VXC_ReadImage2DArray(src0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(value, condition, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    //dst = select(src1, src0, value); // result = c ? b : a\n\
    dst = (value != 0 ? src0 : src1);\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcTensorSelect_int16(\n\
    __read_only image2d_array_t   condition,\n\
    __read_only image2d_array_t   input0,\n\
    __read_only image2d_array_t   input1,\n\
    __write_only image2d_array_t  output)\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_short8 src0, src1, dst;\n\
    vxc_short8 value;\n\
    VXC_ReadImage2DArray(src0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(value, condition, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    //dst = select(src1, src0, value); // result = c ? b : a\n\
    dst = (value != 0 ? src0 : src1);\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_select_vx*/

static const char vsi_nn_kernel_shufflechannel_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
/******************shuffle channel float16/int16********************/\n\
_viv_uniform int group_column;\n\
_viv_uniform float rgroup_column;\n\
\n\
__kernel void shuffleChannelVXC(\n\
    image2d_array_t input,\n\
    image2d_array_t output,\n\
    int group_number)\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    vxc_short8 src0, src1, src2, src3;\n\
    VXC_ReadImage2DArray(src0, input, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input, coord, VXC_5BITOFFSET_XY(0, 1),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src2, input, coord, VXC_5BITOFFSET_XY(0, 2),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src3, input, coord, VXC_5BITOFFSET_XY(0, 3),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    int coordz = coord.z;\n\
    int index_col = coordz * rgroup_column;\n\
    int index_row = coordz - index_col * group_column;\n\
    coord.z = index_row * group_number + index_col;\n\
    VXC_WriteImage2DArray(output, coord, src0, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
    coord.y ++;\n\
    VXC_WriteImage2DArray(output, coord, src1, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
    coord.y ++;\n\
    VXC_WriteImage2DArray(output, coord, src2, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
    coord.y ++;\n\
    VXC_WriteImage2DArray(output, coord, src3, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
/*****************shuffle channel int8/uint8****************************/\n\
\n\
__kernel void shuffleChannel8BitsVXC(\n\
    image2d_array_t input,\n\
    image2d_array_t output,\n\
    int group_number)\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    vxc_char16 src0, src1, src2, src3;\n\
    VXC_ReadImage2DArray(src0, input, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input, coord, VXC_5BITOFFSET_XY(0, 1),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src2, input, coord, VXC_5BITOFFSET_XY(0, 2),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src3, input, coord, VXC_5BITOFFSET_XY(0, 3),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    int coordz = coord.z;\n\
    int index_col = coordz * rgroup_column;\n\
    int index_row = coordz - index_col * group_column;\n\
    coord.z = index_row * group_number + index_col;\n\
    VXC_WriteImage2DArray(output, coord, src0, VXC_MODIFIER(0, 15, 0,VXC_RM_TowardZero, 0));\n\
    coord.y ++;\n\
    VXC_WriteImage2DArray(output, coord, src1, VXC_MODIFIER(0, 15, 0,VXC_RM_TowardZero, 0));\n\
    coord.y ++;\n\
    VXC_WriteImage2DArray(output, coord, src2, VXC_MODIFIER(0, 15, 0,VXC_RM_TowardZero, 0));\n\
    coord.y ++;\n\
    VXC_WriteImage2DArray(output, coord, src3, VXC_MODIFIER(0, 15, 0,VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_shufflechannel_vx*/

static const char vsi_nn_kernel_signalframe_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform int input_width;\n\
_viv_uniform int input_height;\n\
_viv_uniform int input_channel;\n\
_viv_uniform int output_channel;\n\
\n\
\n\
__kernel __attribute__((reqd_work_group_size(8, 1, 1))) void vxcSignalFrame_width(\n\
    image2d_array_t input,\n\
    image2d_array_t output,\n\
                int frame_length,\n\
                int step,\n\
                int pad_end,\n\
                int pad,\n\
                int axis)\n\
{\n\
    int gidx = get_global_id(0);\n\
    int gidy = get_global_id(1);\n\
    int gidz = get_global_id(2);\n\
    int outChn = gidz * input_height + gidy;\n\
    int4 coord = (int4)(0, gidy, gidz, 0);\n\
    int4 coord_out = (int4)(0, 0, outChn, 0);\n\
\n\
    int endcoord = (pad_end == 0) ? (input_width - frame_length + 1) : (input_width);\n\
    int iter = frame_length / 8;\n\
    int res = frame_length % 8;\n\
    vxc_short8 src0;\n\
\n\
    for(int i = 0; i < endcoord; i += step)\n\
    {\n\
        coord.x = i;\n\
        for(int j = 0; j < iter; j++)\n\
        {\n\
            coord_out.x = j << 3;\n\
            coord.x = i + (j << 3);\n\
            VXC_ReadImage2DArray(src0, input, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
            VXC_WriteImage2DArray(output, coord_out, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        }\n\
        coord.x = i + (iter << 3);\n\
        coord_out.x = (iter << 3);\n\
        for(int j = 0; j < res; j++)\n\
        {\n\
            VXC_ReadImage2DArray(src0, input, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
            VXC_WriteImage2DArray(output, coord_out, src0, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
            coord_out.x++;\n\
            coord.x++;\n\
        }\n\
\n\
        coord_out.y++;\n\
    }\n\
}\n\
\n\
__kernel __attribute__((reqd_work_group_size(8, 1, 1))) void vxcSignalFrame_height(\n\
    image2d_array_t input,\n\
    image2d_array_t output,\n\
                int frame_length,\n\
                int step,\n\
                int pad_end,\n\
                int pad,\n\
                int axis)\n\
{\n\
    int gidx = get_global_id(0);\n\
    int gidy = get_global_id(1);\n\
    int gidz = get_global_id(2);\n\
    int outChn = gidz * output_channel + (gidy / step);\n\
    int4 coord = (int4)(gidx, gidy, gidz, 0);\n\
    int4 coord_out = (int4)(gidx, 0, outChn, 0);\n\
    vxc_short8 src0;\n\
\n\
    for(int i = 0; i < frame_length; i++)\n\
    {\n\
        coord.y = gidy + i;\n\
        coord_out.y = i;\n\
        VXC_ReadImage2DArray(src0, input, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_WriteImage2DArray(output, coord_out, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    }\n\
}\n\
\n\
__kernel __attribute__((reqd_work_group_size(8, 1, 1))) void vxcSignalFrame_channel(\n\
    image2d_array_t input,\n\
    image2d_array_t output,\n\
                int frame_length,\n\
                int step,\n\
                int pad_end,\n\
                int pad,\n\
                int axis)\n\
{\n\
    int gidx = get_global_id(0);\n\
    int gidy = get_global_id(1);\n\
    int gidz = get_global_id(2);\n\
    int outChn = (gidz / step) * frame_length;\n\
    int4 coord = (int4)(gidx, gidy, gidz, 0);\n\
    int4 coord_out = (int4)(gidx, gidy, outChn, 0);\n\
    vxc_short8 src0;\n\
\n\
    for(int i = 0; i < frame_length; i++)\n\
    {\n\
        coord.z = gidz + i;\n\
        coord_out.z = outChn + i;\n\
        if(coord.z < input_channel)\n\
        {\n\
            VXC_ReadImage2DArray(src0, input, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        }\n\
        else\n\
        {\n\
            src0 = (vxc_short8)(0);\n\
        }\n\
        VXC_WriteImage2DArray(output, coord_out, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    }\n\
}\n\
\n\
__kernel __attribute__((reqd_work_group_size(8, 1, 1))) void vxcSignalFrame_width_8bit(\n\
    image2d_array_t input,\n\
    image2d_array_t output,\n\
                int frame_length,\n\
                int step,\n\
                int pad_end,\n\
                int pad,\n\
                int axis)\n\
{\n\
    int gidx = get_global_id(0);\n\
    int gidy = get_global_id(1);\n\
    int gidz = get_global_id(2);\n\
    int outChn = gidz * input_height + gidy;\n\
    int4 coord = (int4)(0, gidy, gidz, 0);\n\
    int4 coord_out = (int4)(0, 0, outChn, 0);\n\
\n\
    int endcoord = (pad_end == 0) ? (input_width - frame_length + 1) : (input_width);\n\
    int iter = frame_length / 8;\n\
    int res = frame_length % 8;\n\
    vxc_char8 src0;\n\
\n\
    for(int i = 0; i < endcoord; i += step)\n\
    {\n\
        coord.x = i;\n\
        for(int j = 0; j < iter; j++)\n\
        {\n\
            coord_out.x = j << 3;\n\
            coord.x = i + (j << 3);\n\
            VXC_ReadImage2DArray(src0, input, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
            VXC_WriteImage2DArray(output, coord_out, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        }\n\
        coord.x = i + (iter << 3);\n\
        coord_out.x = (iter << 3);\n\
        for(int j = 0; j < res; j++)\n\
        {\n\
            VXC_ReadImage2DArray(src0, input, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
            VXC_WriteImage2DArray(output, coord_out, src0, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
            coord_out.x++;\n\
            coord.x++;\n\
        }\n\
\n\
        coord_out.y++;\n\
    }\n\
}\n\
\n\
__kernel __attribute__((reqd_work_group_size(8, 1, 1))) void vxcSignalFrame_height_8bit(\n\
    image2d_array_t input,\n\
    image2d_array_t output,\n\
                int frame_length,\n\
                int step,\n\
                int pad_end,\n\
                int pad,\n\
                int axis)\n\
{\n\
    int gidx = get_global_id(0);\n\
    int gidy = get_global_id(1);\n\
    int gidz = get_global_id(2);\n\
    int outChn = gidz * output_channel + (gidy / step);\n\
    int4 coord = (int4)(gidx, gidy, gidz, 0);\n\
    int4 coord_out = (int4)(gidx, 0, outChn, 0);\n\
    vxc_char8 src0;\n\
\n\
    for(int i = 0; i < frame_length; i++)\n\
    {\n\
        coord.y = gidy + i;\n\
        coord_out.y = i;\n\
        VXC_ReadImage2DArray(src0, input, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_WriteImage2DArray(output, coord_out, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    }\n\
}\n\
\n\
__kernel __attribute__((reqd_work_group_size(8, 1, 1))) void vxcSignalFrame_channel_8bit(\n\
    image2d_array_t input,\n\
    image2d_array_t output,\n\
                int frame_length,\n\
                int step,\n\
                int pad_end,\n\
                int pad,\n\
                int axis)\n\
{\n\
    int gidx = get_global_id(0);\n\
    int gidy = get_global_id(1);\n\
    int gidz = get_global_id(2);\n\
    int outChn = (gidz / step) * frame_length;\n\
    int4 coord = (int4)(gidx, gidy, gidz, 0);\n\
    int4 coord_out = (int4)(gidx, gidy, outChn, 0);\n\
    vxc_char8 src0;\n\
\n\
    for(int i = 0; i < frame_length; i++)\n\
    {\n\
        coord.z = gidz + i;\n\
        coord_out.z = outChn + i;\n\
        if(coord.z < input_channel)\n\
        {\n\
            VXC_ReadImage2DArray(src0, input, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        }\n\
        else\n\
        {\n\
            src0 = (vxc_char8)(0);\n\
        }\n\
        VXC_WriteImage2DArray(output, coord_out, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    }\n\
}\n\
\n\
#if 0\n\
__kernel __attribute__((reqd_work_group_size(16, 1, 1))) void vxcSignalFrame_tensor(\n\
    image2d_array_t input,\n\
    image2d_array_t output,\n\
    image2d_array_t frame_length,\n\
    image2d_array_t steps,\n\
    image2d_array_t pad_end,\n\
    image2d_array_t pad,\n\
    image2d_array_t axis)\n\
{\n\
    int gidx = get_global_id(0);\n\
    int gidy = get_global_id(1);\n\
    int gidz = get_global_id(2);\n\
    int outChn = gidz * input_height + gidy;\n\
    int4 coord = (int4)(0, gidy, gidz, 0);\n\
    int4 coord_out = (int4)(0, 0, outChn, 0);\n\
    int4 coord_para = (int4)(0, 0, 0, 0);\n\
\n\
    int4 size = read_imagei(frame_length, coord_para);\n\
    int4 step = read_imagei(steps, coord_para);\n\
    int4 pe = read_imagei(pad_end, coord_para);\n\
    int4 pd = read_imagei(pad, coord_para);\n\
    int len = input_width + (pe.x ? pd : 0);\n\
    int endcoord = len - size.x + 1;\n\
    int iter = size.x / 8;\n\
    int res = size.x % 8;\n\
    vxc_short8 src0;\n\
\n\
    for(int i = 0; i < endcoord; i += step.x)\n\
    {\n\
        coord.x = i;\n\
        for(int j = 0; j < iter; j++)\n\
        {\n\
            coord_out.x = j << 3;\n\
            coord.x += (j << 3);\n\
            VXC_ReadImage2DArray(src0, input, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
            VXC_WriteImage2DArray(output, coord_out, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        }\n\
        coord.x = i + (iter << 3);\n\
        coord_out.x = (iter << 3);\n\
        for(int j = 0; j < res; j++)\n\
        {\n\
            VXC_ReadImage2DArray(src0, input, coord, VXC_5BITOFFSET_XY(0, 0),\n\
                VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
            VXC_WriteImage2DArray(output, coord_out, src0, VXC_MODIFIER(0, 0, 0, VXC_RM_TowardZero, 0));\n\
            coord_out.x++;\n\
            coord.x++;\n\
        }\n\
\n\
        coord_out.y++;\n\
    }\n\
}\n\
#endif\n\
"; /* end of vsi_nn_kernel_signalframe_vx*/

static const char vsi_nn_kernel_sin_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
float4 Sin_(float4 x)\n\
{\n\
    return sin(x);\n\
}\n\
_viv_uniform float inputScale;\n\
_viv_uniform float inputTail;\n\
_viv_uniform float outputScale;\n\
_viv_uniform float outputZP;\n\
_viv_uniform VXC_512Bits uniExtract8Data_2x8;\n\
_viv_uniform VXC_512Bits uniDatatoFp32Part0_4x4;\n\
_viv_uniform VXC_512Bits uniDatatoFp32Part1_4x4;\n\
\n\
#define TENSOR_TRANSCENDENTAL(funcName, src_type_name, dst_type_name, src_type, \\\n\
                src_copy_type, convert_type, dst_type, dst_copy_type) \\\n\
    __kernel void vxTensor##funcName##_##src_type_name##to##dst_type_name( \\\n\
    __read_only  image2d_array_t  input, \\\n\
    __write_only image2d_array_t  output \\\n\
    ) \\\n\
{ \\\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0); \\\n\
    src_type      src0; \\\n\
    src_copy_type src1; \\\n\
    VXC_ReadImage2DArray(src0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src1, src0, 16); \\\n\
 \\\n\
    float4 vecA; \\\n\
    float4 vecB; \\\n\
    VXC_DP4x4(vecA, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniDatatoFp32Part0_4x4); \\\n\
    VXC_DP4x4(vecB, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniDatatoFp32Part1_4x4); \\\n\
    vecA = vecA * inputScale + inputTail; \\\n\
    vecB = vecB * inputScale + inputTail; \\\n\
    vecA = funcName##_(vecA); \\\n\
    vecB = funcName##_(vecB); \\\n\
    vecA = vecA * outputScale + outputZP; \\\n\
    vecB = vecB * outputScale + outputZP; \\\n\
 \\\n\
    convert_type dst0, dst1; \\\n\
    _viv_asm(CONV_RTE, dst0, vecA); \\\n\
    _viv_asm(CONV_RTE, dst1, vecB); \\\n\
    dst_type dst2; \\\n\
    VXC_DP2x8(dst2, dst0, dst1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1), uniExtract8Data_2x8); \\\n\
    dst_copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst2, 16); \\\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
//SIN\n\
TENSOR_TRANSCENDENTAL(Sin, F16, F16, vxc_short8, vxc_half8,  half4, vxc_half8,  vxc_short8)\n\
TENSOR_TRANSCENDENTAL(Sin, F16, I8,  vxc_short8, vxc_half8,  int4,  vxc_char8,  vxc_char8)\n\
TENSOR_TRANSCENDENTAL(Sin, F16, U8,  vxc_short8, vxc_half8,  int4,  vxc_uchar8, vxc_uchar8)\n\
TENSOR_TRANSCENDENTAL(Sin, F16, I16, vxc_short8, vxc_half8,  int4,  vxc_short8, vxc_short8)\n\
TENSOR_TRANSCENDENTAL(Sin, I8,  I8,  vxc_char8,  vxc_char8,  int4,  vxc_char8,  vxc_char8)\n\
TENSOR_TRANSCENDENTAL(Sin, I8,  F16, vxc_char8,  vxc_char8,  half4, vxc_half8,  vxc_short8)\n\
TENSOR_TRANSCENDENTAL(Sin, U8,  U8,  vxc_uchar8, vxc_uchar8, int4,  vxc_uchar8, vxc_uchar8)\n\
TENSOR_TRANSCENDENTAL(Sin, U8,  F16, vxc_uchar8, vxc_uchar8, half4, vxc_half8,  vxc_short8)\n\
TENSOR_TRANSCENDENTAL(Sin, I16, I16, vxc_short8, vxc_short8, int4,  vxc_short8, vxc_short8)\n\
TENSOR_TRANSCENDENTAL(Sin, I16, F16, vxc_short8, vxc_short8, half4, vxc_half8,  vxc_short8)\n\
\n\
#define TENSOR_TRANSCENDENTAL_2D(funcName, src_type_name, dst_type_name, src_type, \\\n\
        src_copy_type, convert_type, dst_type, dst_copy_type) \\\n\
    __kernel void vxTensor##funcName##_##src_type_name##to##dst_type_name##_2D( \\\n\
    __read_only  image2d_array_t  input, \\\n\
    __write_only image2d_array_t  output \\\n\
    ) \\\n\
{ \\\n\
    int2 coord = (int2)(get_global_id(0), get_global_id(1)); \\\n\
    src_type      src0; \\\n\
    src_copy_type src1; \\\n\
    VXC_ReadImage(src0, input, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    _viv_asm(COPY, src1, src0, 16); \\\n\
 \\\n\
    float4 vecA; \\\n\
    float4 vecB; \\\n\
    VXC_DP4x4(vecA, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniDatatoFp32Part0_4x4); \\\n\
    VXC_DP4x4(vecB, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniDatatoFp32Part1_4x4); \\\n\
    vecA = vecA * inputScale + inputTail; \\\n\
    vecB = vecB * inputScale + inputTail; \\\n\
    vecA = funcName##_(vecA); \\\n\
    vecB = funcName##_(vecB); \\\n\
    vecA = vecA * outputScale + outputZP; \\\n\
    vecB = vecB * outputScale + outputZP; \\\n\
 \\\n\
    convert_type dst0, dst1; \\\n\
    _viv_asm(CONV_RTE, dst0, vecA); \\\n\
    _viv_asm(CONV_RTE, dst1, vecB); \\\n\
    dst_type dst2; \\\n\
    VXC_DP2x8(dst2, dst0, dst1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1), uniExtract8Data_2x8); \\\n\
    dst_copy_type dst; \\\n\
    _viv_asm(COPY, dst, dst2, 16); \\\n\
    VXC_WriteImage(output, coord, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
//SIN\n\
TENSOR_TRANSCENDENTAL_2D(Sin, F16, F16, vxc_short8, vxc_half8, half4, vxc_half8,  vxc_short8)\n\
TENSOR_TRANSCENDENTAL_2D(Sin, F16, I8,  vxc_short8, vxc_half8, int4,  vxc_char8,  vxc_char8)\n\
TENSOR_TRANSCENDENTAL_2D(Sin, F16, U8,  vxc_short8, vxc_half8, int4,  vxc_uchar8, vxc_uchar8)\n\
TENSOR_TRANSCENDENTAL_2D(Sin, F16, I16, vxc_short8, vxc_half8, int4,  vxc_short8, vxc_short8)\n\
TENSOR_TRANSCENDENTAL_2D(Sin, I8,  I8,  vxc_char8,  vxc_char8,  int4,  vxc_char8,  vxc_char8)\n\
TENSOR_TRANSCENDENTAL_2D(Sin, I8,  F16, vxc_char8,  vxc_char8,  half4, vxc_half8,  vxc_short8)\n\
TENSOR_TRANSCENDENTAL_2D(Sin, U8,  U8,  vxc_uchar8, vxc_uchar8, int4,  vxc_uchar8, vxc_uchar8)\n\
TENSOR_TRANSCENDENTAL_2D(Sin, U8,  F16, vxc_uchar8, vxc_uchar8, half4, vxc_half8,  vxc_short8)\n\
TENSOR_TRANSCENDENTAL_2D(Sin, I16, I16, vxc_short8, vxc_short8, int4,  vxc_short8, vxc_short8)\n\
TENSOR_TRANSCENDENTAL_2D(Sin, I16, F16, vxc_short8, vxc_short8, half4, vxc_half8,  vxc_short8)\n\
"; /* end of vsi_nn_kernel_sin_vx*/

static const char vsi_nn_kernel_space2depth_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform VXC_512Bits uniExtractEvenFp16Stride2_4x4;\n\
_viv_uniform VXC_512Bits uniExtractOddFp16Stride2_4x4;\n\
_viv_uniform int input_depth;\n\
\n\
__kernel void vxcReorg2_fp16_fp16_sx2_sy1\n\
    (\n\
    image2d_array_t input,\n\
    image2d_array_t output,\n\
    int stridex,\n\
    int stridey\n\
    )\n\
{\n\
    int gidx = get_global_id(0);\n\
    int gidy = get_global_id(1);\n\
    int gidz = get_global_id(2);\n\
\n\
    int4 coord = (int4)(gidx, gidy, gidz, 0);\n\
    int4 coord_out = (int4)(gidx >> 1, gidy, 0, 0);\n\
    int out_d0, out_d1;\n\
    vxc_short8 imageData;\n\
    vxc_short8 imgVal0, imgVal1;\n\
    //int tmpw = gidz / input_depth; \\n\\\n\
    //int tmpz = gidz % input_depth; \\n\\\n\
\n\
    VXC_ReadImage2DArray(imageData, input, coord, VXC_5BITOFFSET_XY(0, 0),\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_DP4x4(imgVal0, imageData, imageData, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\n\
        uniExtractEvenFp16Stride2_4x4);\n\
    VXC_DP4x4(imgVal1, imageData, imageData, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\n\
        uniExtractOddFp16Stride2_4x4);\n\
\n\
    out_d0 = gidz * 2 * 1;\n\
    out_d1 = out_d0 + 1;\n\
\n\
    coord_out.z = out_d0;\n\
    VXC_WriteImage2DArray(output, coord_out, imgVal0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    coord_out.z = out_d1;\n\
    VXC_WriteImage2DArray(output, coord_out, imgVal1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_space2depth_vx*/

static const char vsi_nn_kernel_stack_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
__kernel void vxcStack(\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_array_t  output)\n\
{\n\
\n\
}\n\
"; /* end of vsi_nn_kernel_stack_vx*/

static const char vsi_nn_kernel_tensor_add_mean_stddev_norm_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
/**************************Tensor add mean stddev norm float16*********************/\n\
_viv_uniform int width;\n\
_viv_uniform float dimRatio;\n\
_viv_uniform float rsEps;\n\
_viv_uniform VXC_512Bits uniAddFp16_2x8;\n\
_viv_uniform VXC_512Bits uniFp16SumSqr_dp8x2;\n\
_viv_uniform VXC_512Bits uniAddFp16toFp32Lo_4x4;\n\
_viv_uniform VXC_512Bits uniAddFp16toFp32Hi_4x4;\n\
_viv_uniform VXC_512Bits uniConvertInt32toUint8_2x8;\n\
\n\
// one group(16 threads) calculates one row\n\
__kernel __attribute__((reqd_work_group_size(16, 1, 1))) void vxcTensorAddMeanStdNorm_fp16(\n\
    image2d_array_t input,\n\
    image2d_array_t input1,\n\
    image2d_array_t output,\n\
    float eps)\n\
{\n\
    int lidx = get_local_id(0);\n\
    int gidx = get_global_id(0);\n\
    int2 coord = (int2)(gidx, get_global_id(1));\n\
    vxc_short8 src0, src1, src2;\n\
    float pSum = 0, pSqr = 0;\n\
    float sum = 0, sqr = 0;\n\
    vxc_half8 in_h, in_h1, in_h2;\n\
\n\
    __local float lcl_sum[16];\n\
    __local float lcl_sqr[16];\n\
\n\
    for(; coord.x < width; coord.x += 128)\n\
    {\n\
        VXC_ReadImage(src0, input, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage(src1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        _viv_asm(COPY, in_h, src0, 16);\n\
        _viv_asm(COPY, in_h1, src1, 16);\n\
        VXC_DP2x8(in_h2, in_h, in_h1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
                uniAddFp16_2x8);\n\
        vxc_float4 sumsqr;\n\
        VXC_DP8x2(sumsqr, in_h2, in_h2, VXC_MODIFIER(0, 1, 0, VXC_RM_TowardZero, 0),\\\n\
                uniFp16SumSqr_dp8x2);\n\
        pSum += sumsqr.x;\n\
        pSqr += sumsqr.y;\n\
    }\n\
\n\
    lcl_sum[lidx] = pSum;\n\
    lcl_sqr[lidx] = pSqr;\n\
    barrier(CLK_LOCAL_MEM_FENCE);\n\
\n\
    float4 *pLocalPtr = (float4 *)&lcl_sum[0];\n\
    float4 one = (float4)(1, 1, 1, 1);\n\
    float4 data0;\n\
    data0 = pLocalPtr[0] + pLocalPtr[1] + pLocalPtr[2] + pLocalPtr[3];\n\
    sum = dot(data0, one);\n\
    pLocalPtr = (float4 *)&lcl_sqr[0];\n\
    data0 = pLocalPtr[0] + pLocalPtr[1] + pLocalPtr[2] + pLocalPtr[3];\n\
    sqr = dot(data0, one);\n\
\n\
    vxc_float mean;\n\
    mean = sum * dimRatio;\n\
    vxc_float vari, stddev_inv, rMeanStd;\n\
    vari = sqr*dimRatio - mean*mean;\n\
    stddev_inv = (vari==0 ? rsEps : rsqrt(vari));\n\
    rMeanStd = (-mean) * stddev_inv;\n\
\n\
    for(coord.x = gidx; coord.x < width; coord.x += 128)\n\
    {\n\
        VXC_ReadImage(src0, input, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage(src1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        _viv_asm(COPY, in_h, src0, 16);\n\
        _viv_asm(COPY, in_h1, src1, 16);\n\
\n\
        vxc_float4 in_f0, in_f1;\n\
        VXC_DP4x4(in_f0, in_h, in_h1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
               uniAddFp16toFp32Lo_4x4);\n\
        VXC_DP4x4(in_f1, in_h, in_h1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
                uniAddFp16toFp32Hi_4x4);\n\
\n\
        vxc_float4 norm0, norm1;\n\
        half4 norm_h0, norm_h1;\n\
\n\
        norm0 = in_f0 * stddev_inv + rMeanStd;\n\
        norm1 = in_f1 * stddev_inv + rMeanStd;\n\
        _viv_asm(CONV, norm_h0, norm0);\n\
        _viv_asm(CONV, norm_h1, norm1);\n\
\n\
        VXC_DP2x8(src2, norm_h0, norm_h1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\n\
                uniConvertInt32toUint8_2x8);\n\
        VXC_WriteImage(output, coord, src2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    }\n\
}\n\
\n\
_viv_uniform int2 multAndoutZP0;//[0:15] multiplier, [31:63] output zp\n\
_viv_uniform int2 multAndoutZP1;//[0:15] multiplier, [31:63] output zp\n\
_viv_uniform VXC_512Bits uniU8MulAndPostShift_0_Lo_2x8;\n\
_viv_uniform VXC_512Bits uniU8MulAndPostShift_1_Lo_2x8;\n\
\n\
__kernel __attribute__((reqd_work_group_size(16, 1, 1))) void vxcTensorAddMeanStdNorm_u8_fp16(\n\
    image2d_array_t input,\n\
    image2d_array_t input1,\n\
    image2d_array_t output,\n\
    float eps)\n\
{\n\
    int lidx = get_local_id(0);\n\
    int gidx = get_global_id(0);\n\
    int2 coord = (int2)(gidx, get_global_id(1));\n\
    vxc_uchar8 src0, src1;\n\
    vxc_short8 src2;\n\
    float pSum = 0, pSqr = 0;\n\
    float sum = 0, sqr = 0;\n\
    vxc_half8 in_h, in_h1, in_h2;\n\
\n\
    __local float lcl_sum[16];\n\
    __local float lcl_sqr[16];\n\
\n\
    vxc_ushort8 ms0, ms1;\n\
    _viv_asm(COPY, ms0, multAndoutZP0, 16);\n\
    _viv_asm(COPY, ms1, multAndoutZP1, 16);\n\
\n\
    for(; coord.x < width; coord.x += 128)\n\
    {\n\
        VXC_ReadImage(src0, input, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage(src1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_DP2x8(in_h, src0, ms0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1),\\\n\
                uniU8MulAndPostShift_0_Lo_2x8);\n\
        VXC_DP2x8(in_h1, src1, ms1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1),\\\n\
                uniU8MulAndPostShift_1_Lo_2x8);\n\
        VXC_DP2x8(in_h2, in_h, in_h1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
                uniAddFp16_2x8);\n\
        vxc_float4 sumsqr;\n\
        VXC_DP8x2(sumsqr, in_h2, in_h2, VXC_MODIFIER(0, 1, 0, VXC_RM_TowardZero, 0),\\\n\
                uniFp16SumSqr_dp8x2);\n\
        pSum += sumsqr.x;\n\
        pSqr += sumsqr.y;\n\
    }\n\
\n\
    lcl_sum[lidx] = pSum;\n\
    lcl_sqr[lidx] = pSqr;\n\
    barrier(CLK_LOCAL_MEM_FENCE);\n\
\n\
    float4 *pLocalPtr = (float4 *)&lcl_sum[0];\n\
    float4 one = (float4)(1, 1, 1, 1);\n\
    float4 data0;\n\
    data0 = pLocalPtr[0] + pLocalPtr[1] + pLocalPtr[2] + pLocalPtr[3];\n\
    sum = dot(data0, one);\n\
    pLocalPtr = (float4 *)&lcl_sqr[0];\n\
    data0 = pLocalPtr[0] + pLocalPtr[1] + pLocalPtr[2] + pLocalPtr[3];\n\
    sqr = dot(data0, one);\n\
\n\
    vxc_float mean;\n\
    mean = sum * dimRatio;\n\
    vxc_float vari, stddev_inv, rMeanStd;\n\
    vari = sqr*dimRatio - mean*mean;\n\
    stddev_inv = (vari==0 ? rsEps : rsqrt(vari));\n\
    rMeanStd = (-mean) * stddev_inv;\n\
\n\
    for(coord.x = gidx; coord.x < width; coord.x += 128)\n\
    {\n\
        VXC_ReadImage(src0, input, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage(src1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_DP2x8(in_h, src0, ms0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1),\\\n\
                uniU8MulAndPostShift_0_Lo_2x8);\n\
        VXC_DP2x8(in_h1, src1, ms1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 1),\\\n\
                uniU8MulAndPostShift_1_Lo_2x8);\n\
\n\
        vxc_float4 in_f0, in_f1;\n\
        VXC_DP4x4(in_f0, in_h, in_h1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
               uniAddFp16toFp32Lo_4x4);\n\
        VXC_DP4x4(in_f1, in_h, in_h1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
                uniAddFp16toFp32Hi_4x4);\n\
\n\
        vxc_float4 norm0, norm1;\n\
        half4 norm_h0, norm_h1;\n\
\n\
        norm0 = in_f0 * stddev_inv + rMeanStd;\n\
        norm1 = in_f1 * stddev_inv + rMeanStd;\n\
        _viv_asm(CONV, norm_h0, norm0);\n\
        _viv_asm(CONV, norm_h1, norm1);\n\
\n\
        VXC_DP2x8(src2, norm_h0, norm_h1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\n\
                uniConvertInt32toUint8_2x8);\n\
        VXC_WriteImage(output, coord, src2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    }\n\
}\n\
\n\
_viv_uniform VXC_512Bits uniConvertInt16ScaleToFp32Fst_4x4;\n\
_viv_uniform VXC_512Bits uniConvertInt16ScaleToFp32Sec_4x4;\n\
_viv_uniform VXC_512Bits uniConvertInt16Fp32Fst_4x4;\n\
_viv_uniform VXC_512Bits uniConvertInt16Fp32Secd_4x4;\n\
_viv_uniform float inScale_i16;\n\
_viv_uniform float inScale1_i16;\n\
\n\
// one group(16 threads) calculates one row\n\
__kernel __attribute__((reqd_work_group_size(16, 1, 1))) void vxcTensorAddMeanStdNorm_i16_fp16(\n\
    image2d_array_t input,\n\
    image2d_array_t input1,\n\
    image2d_array_t output,\n\
    float eps)\n\
{\n\
    int lidx = get_local_id(0);\n\
    int gidx = get_global_id(0);\n\
    int2 coord = (int2)(gidx, get_global_id(1));\n\
    vxc_short8 src0, src1, src2;\n\
    float pSum = 0, pSqr = 0;\n\
    float sum = 0, sqr = 0;\n\
\n\
    __local float lcl_sum[16];\n\
    __local float lcl_sqr[16];\n\
\n\
    half scale_h, scale_h1;\n\
    _viv_asm(CONV, scale_h, inScale_i16);\n\
    _viv_asm(CONV, scale_h1, inScale1_i16);\n\
    float4 tmpVal0, tmpVal1, tmpVal2, tmpVal3;\n\
    float4 one = (float4)(1, 1, 1, 1);\n\
\n\
    for(; coord.x < width; coord.x += 128)\n\
    {\n\
        VXC_ReadImage(src0, input, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage(src1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_DP4x4(tmpVal0, src0, scale_h, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
                uniConvertInt16ScaleToFp32Fst_4x4);\n\
        VXC_DP4x4(tmpVal1, src0, scale_h, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
                uniConvertInt16ScaleToFp32Sec_4x4);\n\
        VXC_DP4x4(tmpVal2, src1, scale_h1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
                uniConvertInt16ScaleToFp32Fst_4x4);\n\
        VXC_DP4x4(tmpVal3, src1, scale_h1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
                uniConvertInt16ScaleToFp32Sec_4x4);\n\
        tmpVal0 += tmpVal2;\n\
        tmpVal1 += tmpVal3;\n\
\n\
        vxc_float4 sumsqr;\n\
        sumsqr = tmpVal0 * tmpVal0 + tmpVal1 * tmpVal1; // sqr\n\
        tmpVal2 = tmpVal0 + tmpVal1; // pre sum\n\
\n\
        pSum += dot(tmpVal2, one);\n\
        pSqr += dot(sumsqr, one);\n\
    }\n\
\n\
    lcl_sum[lidx] = pSum;\n\
    lcl_sqr[lidx] = pSqr;\n\
    barrier(CLK_LOCAL_MEM_FENCE);\n\
\n\
    float4 data0;\n\
    float4 *pLocalPtr = (float4 *)&lcl_sum[0];\n\
    data0 = pLocalPtr[0] + pLocalPtr[1] + pLocalPtr[2] + pLocalPtr[3];\n\
    sum = dot(data0, one);\n\
    pLocalPtr = (float4 *)&lcl_sqr[0];\n\
    data0 = pLocalPtr[0] + pLocalPtr[1] + pLocalPtr[2] + pLocalPtr[3];\n\
    sqr = dot(data0, one);\n\
\n\
    vxc_float mean;\n\
    mean = sum * dimRatio;\n\
    vxc_float vari, stddev_inv, rMeanStd;\n\
    vari = sqr*dimRatio - mean*mean;\n\
    stddev_inv = (vari==0 ? rsEps : rsqrt(vari));\n\
    rMeanStd = (-mean) * stddev_inv;\n\
\n\
    for(coord.x = gidx; coord.x < width; coord.x += 128)\n\
    {\n\
        VXC_ReadImage(src0, input, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_ReadImage(src1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_DP4x4(tmpVal0, src0, scale_h, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
                uniConvertInt16ScaleToFp32Fst_4x4);\n\
        VXC_DP4x4(tmpVal1, src0, scale_h, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
                uniConvertInt16ScaleToFp32Sec_4x4);\n\
        VXC_DP4x4(tmpVal2, src1, scale_h1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
                uniConvertInt16ScaleToFp32Fst_4x4);\n\
        VXC_DP4x4(tmpVal3, src1, scale_h1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
                uniConvertInt16ScaleToFp32Sec_4x4);\n\
        tmpVal0 += tmpVal2;\n\
        tmpVal1 += tmpVal3;\n\
\n\
        vxc_float4 norm0, norm1;\n\
        half4 norm_h0, norm_h1;\n\
\n\
        norm0 = tmpVal0 * stddev_inv + rMeanStd;\n\
        norm1 = tmpVal1 * stddev_inv + rMeanStd;\n\
        _viv_asm(CONV, norm_h0, norm0);\n\
        _viv_asm(CONV, norm_h1, norm1);\n\
\n\
        VXC_DP2x8(src2, norm_h0, norm_h1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\n\
                uniConvertInt32toUint8_2x8);\n\
        VXC_WriteImage(output, coord, src2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    }\n\
}"; /* end of vsi_nn_kernel_tensor_add_mean_stddev_norm_vx*/

static const char vsi_nn_kernel_tensorstackconcat_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
/*******************tensorstackconcat 16BITs********************/\n\
__kernel void vxcTensorStackConcat(\n\
    image2d_array_t input,\n\
    image2d_array_t index,\n\
    image2d_array_t output)\n\
{\n\
    int2 coord = (int2)(get_global_id(0), 0);\n\
    vxc_short8 src0, src1;\n\
    VXC_ReadImage(src0, input, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(src1, input, coord.xy, VXC_5BITOFFSET_XY(8, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    coord.y = read_imagei(index, coord.yyyy).x;\n\
    VXC_WriteImage(output, coord.xy, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    coord.x += 8;\n\
    VXC_WriteImage(output, coord.xy, src1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
/**************tensorstackconcat 8BITs***************************/\n\
__kernel void vxcTensorStackConcat8Bits(\n\
    image2d_array_t input,\n\
    image2d_array_t index,\n\
    image2d_array_t output)\n\
{\n\
    int idx = get_global_id(0);\n\
    int2 coord = (int2)(idx, 0);\n\
    vxc_char16 src0, src1;\n\
    VXC_ReadImage(src0, input, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    coord.x += 16;\n\
    VXC_ReadImage(src1, input, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    coord.x = idx;\n\
    coord.y = read_imagei(index, coord.yyyy).x;\n\
    VXC_WriteImage(output, coord.xy, src0, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    coord.x += 16;\n\
    VXC_WriteImage(output, coord.xy, src1, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
}"; /* end of vsi_nn_kernel_tensorstackconcat_vx*/

static const char vsi_nn_kernel_tile_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform int depth;\n\
_viv_uniform int width;\n\
_viv_uniform int last_idx;\n\
\n\
#define TENSORTILE(name0, name1, res_cnt, data_type) \\\n\
__kernel void vxcTile_##name0##_##name1( \\\n\
    __read_only image2d_array_t   input, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __read_only image2d_array_t   multiples) \\\n\
{ \\\n\
    int gidx = get_global_id(0); \\\n\
    int gidy = get_global_id(1); \\\n\
    int gidz = get_global_id(2); \\\n\
    int4 coord = (int4)(gidx, gidy, gidz, 0); \\\n\
    int4 multi = read_imagei(multiples, coord.wwww); \\\n\
    data_type data; \\\n\
    VXC_ReadImage2DArray(data, input, coord, VXC_5BITOFFSET_XY(0, 0), \\\n\
                VXC_MODIFIER(0, res_cnt, 0, VXC_RM_TowardZero, 0)); \\\n\
    int4 coord_out = (int4)(gidx, gidy, gidz, 0); \\\n\
    int offset = depth * multi.z; \\\n\
    for(int b = 0; b < multi.w; b++) \\\n\
    { \\\n\
        int batch = offset * b; \\\n\
        for(int c = 0; c < multi.z; c++) \\\n\
        { \\\n\
            coord_out.z = gidz * multi.z + c + batch; \\\n\
            for(int h = 0; h < multi.y; h++) \\\n\
            { \\\n\
                coord_out.y = gidy * multi.y + h; \\\n\
                for(int w = 0; w < multi.x; w++) \\\n\
                { \\\n\
                    coord_out.x = gidx + w * width; \\\n\
                    VXC_WriteImage2DArray(output, coord_out, data, \\\n\
                             VXC_MODIFIER(0, res_cnt, 0, VXC_RM_TowardZero, 0)); \\\n\
                } \\\n\
            } \\\n\
        } \\\n\
    } \\\n\
}\n\
\n\
TENSORTILE(u8,   blk8, 7,  vxc_uchar8)\n\
TENSORTILE(u8,   blk4, 3,  vxc_uchar8)\n\
TENSORTILE(i16,  blk8, 7,  vxc_short8)\n\
TENSORTILE(i16,  blk4, 3,  vxc_short8)\n\
\n\
#define TENSORTILE_RES(name0, name1, res_cnt, data_type) \\\n\
__kernel void vxcTile_##name0##_##name1( \\\n\
    __read_only image2d_array_t   input, \\\n\
    __write_only image2d_array_t  output, \\\n\
    __read_only image2d_array_t   multiples) \\\n\
{ \\\n\
    int gidx = get_global_id(0); \\\n\
    int gidy = get_global_id(1); \\\n\
    int gidz = get_global_id(2); \\\n\
    int4 coord = (int4)(gidx, gidy, gidz, 0); \\\n\
    int4 multi = read_imagei(multiples, coord.wwww); \\\n\
    data_type data; \\\n\
    VXC_ReadImage2DArray(data, input, coord, VXC_5BITOFFSET_XY(0, 0), \\\n\
                VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
    int4 coord_out = (int4)(gidx, gidy, gidz, 0); \\\n\
    int offset = depth * multi.z; \\\n\
    for(int b = 0; b < multi.w; b++) \\\n\
    { \\\n\
        int batch = offset * b; \\\n\
        for(int c = 0; c < multi.z; c++) \\\n\
        { \\\n\
            coord_out.z = gidz * multi.z + c + batch; \\\n\
            for(int h = 0; h < multi.y; h++) \\\n\
            { \\\n\
                coord_out.y = gidy * multi.y + h; \\\n\
                for(int w = 0; w < multi.x; w++) \\\n\
                { \\\n\
                    coord_out.x = gidx + w * width; \\\n\
                    if(gidx != last_idx) \\\n\
                    { \\\n\
                        VXC_WriteImage2DArray(output, coord_out, data, \\\n\
                             VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
                    } \\\n\
                    else \\\n\
                    { \\\n\
                        VXC_WriteImage2DArray(output, coord_out, data, \\\n\
                             VXC_MODIFIER(0, res_cnt, 0, VXC_RM_TowardZero, 0)); \\\n\
                    } \\\n\
                } \\\n\
            } \\\n\
        } \\\n\
    } \\\n\
}\n\
\n\
TENSORTILE_RES(u8,  res1, 0,  vxc_uchar8)\n\
TENSORTILE_RES(u8,  res2, 1,  vxc_uchar8)\n\
TENSORTILE_RES(u8,  res3, 2,  vxc_uchar8)\n\
TENSORTILE_RES(u8,  res5, 4,  vxc_uchar8)\n\
TENSORTILE_RES(u8,  res6, 5,  vxc_uchar8)\n\
TENSORTILE_RES(u8,  res7, 6,  vxc_uchar8)\n\
\n\
TENSORTILE_RES(i16,  res1, 0,  vxc_short8)\n\
TENSORTILE_RES(i16,  res2, 1,  vxc_short8)\n\
TENSORTILE_RES(i16,  res3, 2,  vxc_short8)\n\
TENSORTILE_RES(i16,  res5, 4,  vxc_short8)\n\
TENSORTILE_RES(i16,  res6, 5,  vxc_short8)\n\
TENSORTILE_RES(i16,  res7, 6,  vxc_short8)"; /* end of vsi_nn_kernel_tile_vx*/

static const char vsi_nn_kernel_topk_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
__kernel void vxcTopk(\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_array_t  output)\n\
{\n\
\n\
}\n\
"; /* end of vsi_nn_kernel_topk_vx*/

static const char vsi_nn_kernel_transform_gemm_vx[] = "/*\n\
 ============================================================================\n\
 Name        : gemm.vx\n\
 Author      : Sam\n\
 Version     :\n\
 Copyright   : Your copyright notice\n\
 Description :\n\
 ============================================================================\n\
 */\n\
#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform VXC_512Bits uniGemm3x3_4x4;\n\
__kernel void vxcTransform_Gemm_F16toF16\n\
    (\n\
    __read_only     image2d_array_t thetaTensor,\n\
    __read_only     image2d_array_t gridTensor,\n\
    __write_only    image2d_array_t coordinates\n\
    )\n\
{\n\
    int4 coord    = (int4)(0, get_global_id(0), get_global_id(1), 0);\n\
\n\
    vxc_short8 vec0, vec1, vec2;\n\
    vxc_half8  src0, src1, src2, dst;\n\
\n\
    VXC_ReadImage(vec0,thetaTensor,coord.xx,VXC_5BITOFFSET_XY(0,0),VXC_MODIFIER(0,5, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, src0, vec0, 16);\n\
    VXC_ReadImage(vec1,gridTensor,coord.yz,VXC_5BITOFFSET_XY(0,0),VXC_MODIFIER(0,5,0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, src1, vec1, 16);\n\
    VXC_ReadImage(vec2,gridTensor,coord.yz,VXC_5BITOFFSET_XY(6,0),VXC_MODIFIER(0,5,0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, src2, vec2, 16);\n\
\n\
    coord.y = (int)((short)coord.y / (short)3) * 2;\n\
\n\
    VXC_DP4x4(dst, src1, src0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniGemm3x3_4x4);\n\
    VXC_DP4x4(dst, src2, src0, VXC_MODIFIER(4, 7, 0, VXC_RM_TowardZero, 0), uniGemm3x3_4x4);\n\
\n\
    _viv_asm(COPY, vec0, dst, 16);\n\
    VXC_WriteImage(coordinates, coord.yz, vec0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_transform_gemm_vx*/

static const char vsi_nn_kernel_transform_interp_vx[] = "/*\n\
 ============================================================================\n\
 Name        : minimum.vx\n\
 Author      : Sam\n\
 Version     :\n\
 Copyright   : Your copyright notice\n\
 Description :\n\
 ============================================================================\n\
 */\n\
#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform VXC_512Bits uniGetDXY_4x4;\n\
_viv_uniform VXC_512Bits uniConvertF16toF32_4x4;\n\
_viv_uniform int2 packedWH2;\n\
_viv_uniform int  packedWH;\n\
__kernel void vxcTransform_InterP_F16toF16_2D\n\
    (\n\
    __read_only     image2d_array_t input0,\n\
    __read_only     image2d_array_t input1,\n\
    __write_only    image2d_array_t output\n\
    )\n\
{\n\
    int2 coord  =  (int2)(get_global_id(0), get_global_id(1));\n\
\n\
    vxc_short8 vec0;\n\
    vxc_half8  pxy;\n\
    vxc_float4 dxy4;\n\
    vxc_int4   pos4;\n\
    short dst = 0;\n\
\n\
    VXC_ReadImage(vec0, input1, coord.xy, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 1, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, pxy, vec0, 4);\n\
\n\
    coord.x >>= 1;\n\
    vxc_short2 packedWH_16B;\n\
    _viv_asm(COPY, packedWH_16B, packedWH, 4);\n\
    VXC_DP4x4(dxy4, pxy, packedWH_16B, VXC_MODIFIER(0, 1, 0, VXC_RM_TowardZero, 0), uniGetDXY_4x4);\n\
    dxy4.zw = floor(dxy4.xy);\n\
    pos4.xy = convert_int2(dxy4.zw);\n\
    pos4.zw = convert_int2(ceil(dxy4.xy));\n\
\n\
    vxc_short8 vec1;\n\
    vxc_half8  src0, src1;\n\
    VXC_ReadImage(vec0, input0, pos4.xy, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 1, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, src0, vec0, 8);\n\
    VXC_ReadImage(vec1, input0, pos4.xw, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 1, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, src1, vec1, 8);\n\
\n\
    float2 xyLerp        = dxy4.xy - dxy4.zw;\n\
    float2 oneSub_xyLerp = 1.0f - xyLerp;\n\
    float4 coef          = (float4)(oneSub_xyLerp.x * oneSub_xyLerp.y, xyLerp.x * oneSub_xyLerp.y,\n\
                                    oneSub_xyLerp.x * xyLerp.y, xyLerp.x * xyLerp.y);\n\
    float4  data;\n\
\n\
    VXC_DP4x4(data, src0, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertF16toF32_4x4);\n\
\n\
    data.x = dot(data, coef);\n\
\n\
    half tmp;\n\
    _viv_asm(CONV, tmp, data);\n\
    _viv_asm(COPY, dst, tmp, 4);\n\
\n\
    VXC_WriteImage(output, coord, dst, VXC_MODIFIER(0, 0, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform int depth;\n\
__kernel void vxcTransform_InterP_F16toF16\n\
    (\n\
    __read_only     image2d_array_t input0,\n\
    __read_only     image2d_array_t input1,\n\
    __write_only    image2d_array_t output\n\
    )\n\
{\n\
    int4 coord  =  (int4)(get_global_id(0), get_global_id(1), 0, 0);\n\
\n\
    vxc_short8 vec0;\n\
    vxc_half8  pxy;\n\
    vxc_float4 dxy4;\n\
    vxc_int4   pos4;\n\
    short dst = 0;\n\
\n\
    VXC_ReadImage(vec0, input1, coord.xy, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 1, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, pxy, vec0, 4);\n\
\n\
    coord.x >>= 1;\n\
    vxc_short2 packedWH_16B;\n\
    _viv_asm(COPY, packedWH_16B, packedWH, 4);\n\
    VXC_DP4x4(dxy4, pxy, packedWH_16B, VXC_MODIFIER(0, 1, 0, VXC_RM_TowardZero, 0), uniGetDXY_4x4);\n\
    dxy4.zw = floor(dxy4.xy);\n\
    pos4.xy = convert_int2(dxy4.zw);\n\
    pos4.zw = convert_int2(ceil(dxy4.xy));\n\
\n\
\n\
    float2 xyLerp        = dxy4.xy - dxy4.zw;\n\
    float2 oneSub_xyLerp = 1.0f - xyLerp;\n\
    float4 coef          = (float4)(oneSub_xyLerp.x * oneSub_xyLerp.y, xyLerp.x * oneSub_xyLerp.y,\n\
                                    oneSub_xyLerp.x * xyLerp.y, xyLerp.x * xyLerp.y);\n\
\n\
    int4 coord_ = (int4)(pos4.x, pos4.y, 0, 0);\n\
    do\n\
    {\n\
        vxc_short8 vec1;\n\
        vxc_half8  src0, src1;\n\
        VXC_ReadImage2DArray(vec0,input0,coord_,VXC_5BITOFFSET_XY(0,0),VXC_MODIFIER(0,1,0, VXC_RM_TowardZero, 0));\n\
        _viv_asm(COPY, src0, vec0, 8);\n\
        VXC_ReadImage2DArray(vec1,input0,coord_,VXC_5BITOFFSET_XY(0,1),VXC_MODIFIER(0,1,0, VXC_RM_TowardZero, 0));\n\
        _viv_asm(COPY, src1, vec1, 8);\n\
\n\
        coord_.z ++;\n\
        float4  data;\n\
        VXC_DP4x4(data, src0, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertF16toF32_4x4);\n\
\n\
        data.x = dot(data, coef);\n\
\n\
        half tmp;\n\
        _viv_asm(CONV, tmp, data);\n\
        _viv_asm(COPY, dst, tmp, 4);\n\
\n\
\n\
        VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 0, 0,VXC_RM_TowardZero, 0));\n\
        coord.z ++;\n\
\n\
    } while (coord.z < depth);\n\
}\n\
\n\
"; /* end of vsi_nn_kernel_transform_interp_vx*/

static const char vsi_nn_kernel_transform_setupThres_vx[] = "/*\n\
 ============================================================================\n\
 Name        : gemm.vx\n\
 Author      : Sam\n\
 Version     :\n\
 Copyright   : Your copyright notice\n\
 Description :\n\
 ============================================================================\n\
 */\n\
#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform int4 extract_packed;\n\
__kernel void vxcTransform_setupThres_F16toF16\n\
    (\n\
    __read_only     image2d_array_t initTensor,\n\
    __read_only     image2d_array_t inputFC,\n\
     global int*     thresFlag,\n\
    __write_only    image2d_array_t thres\n\
    )\n\
{\n\
    int2 coord    = (int2)(0, 0);\n\
\n\
    vxc_ushort8 src0, src1, dst;\n\
\n\
    int flag = *thresFlag;\n\
    VXC_ReadImage(src0, initTensor, coord, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(src1, inputFC, coord, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_BitExtract(dst, src0, src1, extract_packed, VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_WriteImage(thres, coord, dst, VXC_MODIFIER(0, 5, 0, VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_transform_setupThres_vx*/

static const char vsi_nn_kernel_unstack_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
__kernel void vxcUnstack(\n\
    __read_only image2d_array_t   input,\n\
    __write_only image2d_array_t  output)\n\
{\n\
\n\
}\n\
"; /* end of vsi_nn_kernel_unstack_vx*/

static const char vsi_nn_kernel_upsample_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
//--------------------------unpooling-------------------------\n\
_viv_uniform VXC_512Bits uniConvertDirInt16Fp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertEndInt16Fp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertInt32toInt16_2x8;\n\
\n\
_viv_uniform float inScaleInt16;\n\
_viv_uniform float scaleSF;\n\
_viv_uniform VXC_512Bits ucharMulShort_8x8;\n\
\n\
__kernel void unpooling\n\
    (\n\
        image2d_array_t dataIn,\n\
        image2d_array_t axis,\n\
        image2d_array_t dataOut,\n\
        unsigned int sizeX,\n\
        unsigned int sizeY\n\
    )\n\
{\n\
    vxc_short4 din;\n\
    vxc_uchar4 axisIn;\n\
    vxc_short8 dinExp;\n\
    vxc_uchar8 axisInExp;\n\
    vxc_uchar8 constAxis;\n\
    vxc_uchar8 axisData;\n\
    vxc_short8 dout;\n\
\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    VXC_ReadImage2DArray(din, dataIn, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(axisIn, axis, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    int4 coordOut = (int4)(coord.x << 1, coord.y << 1, coord.z, 0);\n\
    dinExp = din.s00112233;\n\
    axisInExp = axisIn.s00112233;\n\
\n\
    constAxis = (vxc_uchar8)(0, 1, 0, 1, 0, 1, 0, 1);\n\
    VXC_Clamp(axisData, axisInExp, constAxis, constAxis, VXC_MODIFIER_CLAMP(0, 7, 0, 1));\n\
    axisData &= (vxc_uchar8)(1);\n\
    VXC_DP2x8(dout, axisData, dinExp, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), ucharMulShort_8x8);\n\
    //dout = (axisData == constAxis) ? dinExp : constZeros;\n\
    VXC_WriteImage2DArray(dataOut, coordOut, dout, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    constAxis = (vxc_uchar8)(2, 3, 2, 3, 2, 3, 2, 3);\n\
    VXC_Clamp(axisData, axisInExp, constAxis, constAxis, VXC_MODIFIER_CLAMP(0, 7, 0, 1));\n\
    axisData &= (vxc_uchar8)(1);\n\
    VXC_DP2x8(dout, axisData, dinExp, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), ucharMulShort_8x8);\n\
    coordOut.y += 1;\n\
    VXC_WriteImage2DArray(dataOut, coordOut, dout, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void unpoolingInt16_Int16\n\
    (\n\
        image2d_array_t dataIn,\n\
        image2d_array_t axis,\n\
        image2d_array_t dataOut,\n\
        unsigned int sizeX,\n\
        unsigned int sizeY\n\
    )\n\
{\n\
    vxc_short4 din;\n\
    vxc_uchar4 axisIn;\n\
    vxc_short8 dinExp;\n\
    vxc_uchar8 axisInExp;\n\
    vxc_uchar8 constAxis;\n\
    vxc_uchar8 axisData;\n\
    vxc_short8 dout;\n\
\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    VXC_ReadImage2DArray(din, dataIn, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(axisIn, axis, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    int4 coordOut = (int4)(coord.x << 1, coord.y << 1, coord.z, 0);\n\
    dinExp = din.s00112233;\n\
    axisInExp = axisIn.s00112233;\n\
    constAxis = (vxc_uchar8)(0, 1, 0, 1, 0, 1, 0, 1);\n\
    VXC_Clamp(axisData, axisInExp, constAxis, constAxis, VXC_MODIFIER_CLAMP(0, 7, 0, 1));\n\
    axisData &= (vxc_uchar8)(1);\n\
    VXC_DP2x8(dout, axisData, dinExp, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), ucharMulShort_8x8);\n\
    //dout = (axisData == constAxis) ? dinExp : constZeros;\n\
    vxc_float4 tmpVal0, tmpVal1, tmpVal2, tmpVal3;\n\
    vxc_int4 tmpOut0, tmpOut1;\n\
    VXC_DP4x4(tmpVal0, dout, dout, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertDirInt16Fp32_4x4);\n\
    VXC_DP4x4(tmpVal2, dout, dout, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertEndInt16Fp32_4x4);\n\
    tmpVal1 = tmpVal0 * scaleSF;\n\
    tmpOut0 = convert_int4_rte(tmpVal1);\n\
    tmpVal3 = tmpVal2 * scaleSF;\n\
    tmpOut1 = convert_int4_rte(tmpVal3);\n\
    VXC_DP2x8(dout, tmpOut0, tmpOut1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniConvertInt32toInt16_2x8);\n\
    VXC_WriteImage2DArray(dataOut, coordOut, dout, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    constAxis = (vxc_uchar8)(2, 3, 2, 3, 2, 3, 2, 3);\n\
    VXC_Clamp(axisData, axisInExp, constAxis, constAxis, VXC_MODIFIER_CLAMP(0, 7, 0, 1));\n\
    axisData &= (vxc_uchar8)(1);\n\
    VXC_DP2x8(dout, axisData, dinExp, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), ucharMulShort_8x8);\n\
\n\
    VXC_DP4x4(tmpVal0, dout, dout, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertDirInt16Fp32_4x4);\n\
    VXC_DP4x4(tmpVal2, dout, dout, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertEndInt16Fp32_4x4);\n\
    tmpVal1 = tmpVal0 * scaleSF;\n\
    tmpOut0 = convert_int4_rte(tmpVal1);\n\
    tmpVal3 = tmpVal2 * scaleSF;\n\
    tmpOut1 = convert_int4_rte(tmpVal3);\n\
    VXC_DP2x8(dout, tmpOut0, tmpOut1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniConvertInt32toInt16_2x8);\n\
    coordOut.y += 1;\n\
    VXC_WriteImage2DArray(dataOut, coordOut, dout, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void unpoolingInt16_Int16_axI16\n\
    (\n\
        image2d_array_t dataIn,\n\
        image2d_array_t axis,\n\
        image2d_array_t dataOut,\n\
        unsigned int sizeX,\n\
        unsigned int sizeY\n\
    )\n\
{\n\
    vxc_short4 din;\n\
    vxc_short4 axisIn;\n\
    vxc_short8 dinExp;\n\
    vxc_short8 axisInExp;\n\
    vxc_short8 constAxis;\n\
    vxc_short8 axisData;\n\
    vxc_short8 dout;\n\
\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    VXC_ReadImage2DArray(din, dataIn, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(axisIn, axis, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    int4 coordOut = (int4)(coord.x << 1, coord.y << 1, coord.z, 0);\n\
    dinExp = din.s00112233;\n\
    axisInExp = axisIn.s00112233;\n\
    constAxis = (vxc_short8)(0, 1, 0, 1, 0, 1, 0, 1);\n\
    VXC_Clamp(axisData, axisInExp, constAxis, constAxis, VXC_MODIFIER_CLAMP(0, 7, 0, 1));\n\
    axisData &= (vxc_short8)(1);\n\
    VXC_DP2x8(dout, axisData, dinExp, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), ucharMulShort_8x8);\n\
    //dout = (axisData == constAxis) ? dinExp : constZeros;\n\
    vxc_float4 tmpVal0, tmpVal1, tmpVal2, tmpVal3;\n\
    vxc_int4 tmpOut0, tmpOut1;\n\
    VXC_DP4x4(tmpVal0, dout, dout, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertDirInt16Fp32_4x4);\n\
    VXC_DP4x4(tmpVal2, dout, dout, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertEndInt16Fp32_4x4);\n\
    tmpVal1 = tmpVal0 * scaleSF;\n\
    tmpOut0 = convert_int4_rte(tmpVal1);\n\
    tmpVal3 = tmpVal2 * scaleSF;\n\
    tmpOut1 = convert_int4_rte(tmpVal3);\n\
    VXC_DP2x8(dout, tmpOut0, tmpOut1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniConvertInt32toInt16_2x8);\n\
    VXC_WriteImage2DArray(dataOut, coordOut, dout, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    constAxis = (vxc_short8)(2, 3, 2, 3, 2, 3, 2, 3);\n\
    VXC_Clamp(axisData, axisInExp, constAxis, constAxis, VXC_MODIFIER_CLAMP(0, 7, 0, 1));\n\
    axisData &= (vxc_short8)(1);\n\
    VXC_DP2x8(dout, axisData, dinExp, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), ucharMulShort_8x8);\n\
    VXC_DP4x4(tmpVal0, dout, dout, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertDirInt16Fp32_4x4);\n\
    VXC_DP4x4(tmpVal2, dout, dout, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertEndInt16Fp32_4x4);\n\
    tmpVal1 = tmpVal0 * scaleSF;\n\
    tmpOut0 = convert_int4_rte(tmpVal1);\n\
    tmpVal3 = tmpVal2 * scaleSF;\n\
    tmpOut1 = convert_int4_rte(tmpVal3);\n\
    VXC_DP2x8(dout, tmpOut0, tmpOut1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniConvertInt32toInt16_2x8);\n\
    coordOut.y += 1;\n\
    VXC_WriteImage2DArray(dataOut, coordOut, dout, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void unpoolingInt16_Fp16_axI16\n\
    (\n\
        image2d_array_t dataIn,\n\
        image2d_array_t axis,\n\
        image2d_array_t dataOut,\n\
        unsigned int sizeX,\n\
        unsigned int sizeY\n\
    )\n\
{\n\
    vxc_short4 din;\n\
    vxc_short4 axisIn;\n\
    vxc_short8 dinExp;\n\
    vxc_short8 axisInExp;\n\
    vxc_short8 constAxis;\n\
    vxc_short8 axisData;\n\
    vxc_short8 dout;\n\
\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    VXC_ReadImage2DArray(din, dataIn, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(axisIn, axis, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    int4 coordOut = (int4)(coord.x << 1, coord.y << 1, coord.z, 0);\n\
    dinExp = din.s00112233;\n\
    axisInExp = axisIn.s00112233;\n\
\n\
    constAxis = (vxc_short8)(0, 1, 0, 1, 0, 1, 0, 1);\n\
    VXC_Clamp(axisData, axisInExp, constAxis, constAxis, VXC_MODIFIER_CLAMP(0, 7, 0, 1));\n\
    axisData &= (vxc_short8)(1);\n\
    VXC_DP2x8(dout, axisData, dinExp, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
        ucharMulShort_8x8);\n\
    //dout = (axisData == constAxis) ? dinExp : constZeros;\n\
    vxc_float4 tmpVal0, tmpVal1, tmpVal2, tmpVal3;\n\
    half4 tmpOut0, tmpOut1;\n\
    VXC_DP4x4(tmpVal0, dout, dout, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertDirInt16Fp32_4x4);\n\
    VXC_DP4x4(tmpVal2, dout, dout, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertEndInt16Fp32_4x4);\n\
    tmpVal1 = tmpVal0 * inScaleInt16;\n\
    _viv_asm(CONV, tmpOut0, tmpVal1);\n\
    tmpVal3 = tmpVal2 * inScaleInt16;\n\
    _viv_asm(CONV, tmpOut1, tmpVal3);\n\
    VXC_DP2x8(dout, tmpOut0, tmpOut1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertInt32toInt16_2x8);\n\
    VXC_WriteImage2DArray(dataOut, coordOut, dout, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    constAxis = (vxc_short8)(2, 3, 2, 3, 2, 3, 2, 3);\n\
    VXC_Clamp(axisData, axisInExp, constAxis, constAxis, VXC_MODIFIER_CLAMP(0, 7, 0, 1));\n\
    axisData &= (vxc_short8)(1);\n\
    VXC_DP2x8(dout, axisData, dinExp, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), ucharMulShort_8x8);\n\
    VXC_DP4x4(tmpVal0, dout, dout, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertDirInt16Fp32_4x4);\n\
    VXC_DP4x4(tmpVal2, dout, dout, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertEndInt16Fp32_4x4);\n\
    tmpVal1 = tmpVal0 * inScaleInt16;\n\
    _viv_asm(CONV, tmpOut0, tmpVal1);\n\
    tmpVal3 = tmpVal2 * inScaleInt16;\n\
    _viv_asm(CONV, tmpOut1, tmpVal3);\n\
    VXC_DP2x8(dout, tmpOut0, tmpOut1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertInt32toInt16_2x8);\n\
    coordOut.y += 1;\n\
    VXC_WriteImage2DArray(dataOut, coordOut, dout, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_upsample_vx*/

static const char vsi_nn_kernel_upsample_2_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform VXC_512Bits ucharMulShort_8x8_2;\n\
_viv_uniform VXC_512Bits shortMulShort_8x8;\n\
_viv_uniform VXC_512Bits uniConvertFstFp16Fp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertSecFp16Fp32_4x4;\n\
_viv_uniform int upOutput_ZP;\n\
_viv_uniform float upOutput_Scale;\n\
_viv_uniform float reUpOutScale_u8;\n\
_viv_uniform float up_outFlScale_i8;\n\
_viv_uniform float up_outFlScale_i16;\n\
_viv_uniform VXC_512Bits uniConvertInt32toUint8_2x8;\n\
\n\
_viv_uniform VXC_512Bits uniF16MulMultipiler_PostShft_2x8;\n\
_viv_uniform VXC_512Bits uniS16AddOutZP_2x8;\n\
_viv_uniform vxc_uint4 packed_outputZP;\n\
__kernel void unpoolingFp16_Uint8\n\
    (\n\
        image2d_array_t dataIn,\n\
        image2d_array_t axis,\n\
        image2d_array_t dataOut,\n\
        unsigned int sizeX,\n\
        unsigned int sizeY\n\
    )\n\
{\n\
    vxc_short8 din0;\n\
    vxc_uchar16 din;\n\
    vxc_uchar8 axisIn;\n\
    vxc_half8 src;\n\
\n\
    vxc_uchar16 dinExpand;\n\
    vxc_uchar16 axisInExpand;\n\
    vxc_uchar16 constAxis;\n\
    vxc_uchar16 axisData;\n\
    vxc_uchar16 axisData1;\n\
    vxc_uchar16 dout;\n\
\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    VXC_ReadImage2DArray(din0, dataIn, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(axisIn, axis, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    int4 coordOut = (int4)(coord.x << 1, coord.y << 1, coord.z, 0);\n\
\n\
    vxc_short8 tmp;\n\
    uchar zp = 0;\n\
    _viv_asm(COPY, src, din0, 16);\n\
    VXC_DP2x8(tmp, src, src, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniF16MulMultipiler_PostShft_2x8);\n\
    vxc_uchar16 packed_outZP;\n\
    _viv_asm(COPY, packed_outZP, packed_outputZP, 16);\n\
    VXC_DP2x8(din, tmp, packed_outZP, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniS16AddOutZP_2x8);\n\
\n\
\n\
    constAxis      = (vxc_uchar16)(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);\n\
    dinExpand    = din.s0011223344556677;\n\
    axisInExpand = axisIn.s0011223344556677;\n\
    VXC_Clamp(axisData, axisInExpand, constAxis, constAxis, VXC_MODIFIER_CLAMP(0, 15, 0, 1));\n\
    axisData &= (vxc_uchar16)(1);\n\
    _viv_asm(COPY, axisData1, axisData, 16);\n\
    dout = axisData1 * dinExpand;\n\
    VXC_WriteImage2DArray(dataOut, coordOut, dout, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    constAxis = (vxc_uchar16)(2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3);\n\
    VXC_Clamp(axisData, axisInExpand, constAxis, constAxis, VXC_MODIFIER_CLAMP(0, 15, 0, 1));\n\
    axisData &= (vxc_uchar16)(1);\n\
    _viv_asm(COPY, axisData1, axisData, 16);\n\
    dout = axisData1 * dinExpand;  //output\n\
    coordOut.y += 1;\n\
    VXC_WriteImage2DArray(dataOut, coordOut, dout, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void unpoolingFp16Fp16_Uint8\n\
    (\n\
        image2d_array_t dataIn,\n\
        image2d_array_t axis,\n\
        image2d_array_t dataOut,\n\
        unsigned int sizeX,\n\
        unsigned int sizeY\n\
    )\n\
{\n\
    vxc_short4 din;\n\
    vxc_short4 axisIn;\n\
    vxc_short8 dinExp, axisInExp, constAxis,axisData,tmpout;\n\
    vxc_half8 dout;\n\
    vxc_float4 tmpVal1, tmpVal2, convZp;\n\
    vxc_int4 tmpData1, tmpData2, tmpData3;\n\
    vxc_uchar8 result;\n\
\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    VXC_ReadImage2DArray(din, dataIn, coord, VXC_5BITOFFSET_XY(0, 0),\n\
        VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(axisIn, axis, coord, VXC_5BITOFFSET_XY(0, 0),\n\
        VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    int4 coordOut = (int4)(coord.x << 1, coord.y << 1, coord.z, 0);\n\
    dinExp = din.s00112233;\n\
    axisInExp = axisIn.s00112233;\n\
\n\
    constAxis = (vxc_short8)(0, 1, 0, 1, 0, 1, 0, 1);\n\
    VXC_Clamp(axisData, axisInExp, constAxis, constAxis, VXC_MODIFIER_CLAMP(0, 7, 0, 1));\n\
    axisData &= (vxc_short8)(1);\n\
    VXC_DP2x8(tmpout, axisData, dinExp, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), shortMulShort_8x8);\n\
    _viv_asm(COPY, dout, tmpout, 16);\n\
    VXC_DP4x4(tmpVal1, dout, dout, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertFstFp16Fp32_4x4);\n\
    VXC_DP4x4(tmpVal2, dout, dout, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertSecFp16Fp32_4x4);\n\
    tmpVal1 /= upOutput_Scale;\n\
    tmpVal2 /= upOutput_Scale;\n\
    tmpData3 = isnotequal(tmpVal1, 0);\n\
    tmpData3 *= (-upOutput_ZP);\n\
    convZp = convert_float4_rtp(tmpData3);\n\
    tmpVal1 += convZp;\n\
    tmpData3 = isnotequal(tmpVal2, 0);\n\
    tmpData3 *= (-upOutput_ZP);\n\
    convZp = convert_float4_rtp(tmpData3);\n\
    tmpVal2 += convZp;\n\
    tmpData1 = convert_int4_rte(tmpVal1);\n\
    tmpData2 = convert_int4_rte(tmpVal2);\n\
    VXC_DP2x8(result, tmpData1, tmpData2, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\n\
        uniConvertInt32toUint8_2x8);\n\
    VXC_WriteImage2DArray(dataOut, coordOut, result, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    constAxis = (vxc_short8)(2, 3, 2, 3, 2, 3, 2, 3);\n\
    VXC_Clamp(axisData, axisInExp, constAxis, constAxis, VXC_MODIFIER_CLAMP(0, 7, 0, 1));\n\
    axisData &= (vxc_short8)(1);\n\
    VXC_DP2x8(tmpout, axisData, dinExp, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), shortMulShort_8x8);\n\
    _viv_asm(COPY, dout, tmpout, 16);\n\
    VXC_DP4x4(tmpVal1, dout, dout, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertFstFp16Fp32_4x4);\n\
    VXC_DP4x4(tmpVal2, dout, dout, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertSecFp16Fp32_4x4);\n\
    tmpVal1 /= upOutput_Scale;\n\
    tmpVal2 /= upOutput_Scale;\n\
    tmpData3 = isnotequal(tmpVal1, 0);\n\
    tmpData3 *= (-upOutput_ZP);\n\
    convZp = convert_float4_rtp(tmpData3);\n\
    tmpVal1 += convZp;\n\
    tmpData3 = isnotequal(tmpVal2, 0);\n\
    tmpData3 *= (-upOutput_ZP);\n\
    convZp = convert_float4_rtp(tmpData3);\n\
    tmpVal2 += convZp;\n\
    tmpData1 = convert_int4_rte(tmpVal1);\n\
    tmpData2 = convert_int4_rte(tmpVal2);\n\
    VXC_DP2x8(result, tmpData1, tmpData2, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\n\
        uniConvertInt32toUint8_2x8);\n\
    coordOut.y += 1;\n\
    VXC_WriteImage2DArray(dataOut, coordOut, tmpout, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform VXC_512Bits uniConvertFp16toInt8_2x8;\n\
\n\
__kernel void unpoolingFp16_Int8\n\
    (\n\
        image2d_array_t dataIn,\n\
        image2d_array_t axis,\n\
        image2d_array_t dataOut,\n\
        unsigned int sizeX,\n\
        unsigned int sizeY\n\
    )\n\
{\n\
    vxc_short4 din;\n\
    vxc_uchar4 axisIn;\n\
    vxc_short8 dinExp, tmpOut;\n\
    vxc_uchar8 axisInExp;\n\
    vxc_uchar8 constAxis;\n\
    vxc_uchar8 axisData;\n\
    vxc_half8 dout;\n\
    float4 tmpVal1, tmpVal2;\n\
    int4 tmpData1, tmpData2;\n\
    vxc_char8 result;\n\
\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    VXC_ReadImage2DArray(din, dataIn, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(axisIn, axis, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    int4 coordOut = (int4)(coord.x << 1, coord.y << 1, coord.z, 0);\n\
    dinExp = din.s00112233;\n\
    axisInExp = axisIn.s00112233;\n\
    constAxis = (vxc_uchar8)(0, 1, 0, 1, 0, 1, 0, 1);\n\
    VXC_Clamp(axisData, axisInExp, constAxis, constAxis, VXC_MODIFIER_CLAMP(0, 7, 0, 1));\n\
    axisData &= (vxc_uchar8)(1);\n\
    VXC_DP2x8(tmpOut, axisData, dinExp, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), ucharMulShort_8x8_2);\n\
    _viv_asm(COPY, dout, tmpOut, 16);\n\
\n\
    half tmpScale;\n\
    _viv_asm(CONV, tmpScale, up_outFlScale_i8);\n\
    VXC_DP2x8(result, dout, tmpScale, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniConvertFp16toInt8_2x8);\n\
    VXC_WriteImage2DArray(dataOut, coordOut, result, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    constAxis = (vxc_uchar8)(2, 3, 2, 3, 2, 3, 2, 3);\n\
    VXC_Clamp(axisData, axisInExp, constAxis, constAxis, VXC_MODIFIER_CLAMP(0, 7, 0, 1));\n\
    axisData &= (vxc_uchar8)(1);\n\
    VXC_DP2x8(tmpOut, axisData, dinExp, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), ucharMulShort_8x8_2);\n\
    coordOut.y += 1;\n\
    _viv_asm(COPY, dout, tmpOut, 16);\n\
    VXC_DP2x8(result, dout, tmpScale, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniConvertFp16toInt8_2x8);\n\
    VXC_WriteImage2DArray(dataOut, coordOut, result, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void unpoolingFp16_Int16\n\
    (\n\
        image2d_array_t dataIn,\n\
        image2d_array_t axis,\n\
        image2d_array_t dataOut,\n\
        unsigned int sizeX,\n\
        unsigned int sizeY\n\
    )\n\
{\n\
    vxc_short4 din;\n\
    vxc_uchar4 axisIn;\n\
    vxc_short8 dinExp, tmpOut;\n\
    vxc_uchar8 axisInExp;\n\
    vxc_uchar8 constAxis;\n\
    vxc_uchar8 axisData;\n\
    half8 dout;\n\
    float4 tmpVal1, tmpVal2;\n\
    int4 tmpData1, tmpData2;\n\
    vxc_short8 result;\n\
\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    VXC_ReadImage2DArray(din, dataIn, coord, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(axisIn, axis, coord, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    int4 coordOut = (int4)(coord.x << 1, coord.y << 1, coord.z, 0);\n\
    dinExp = din.s00112233;\n\
    axisInExp = axisIn.s00112233;\n\
    constAxis = (vxc_uchar8)(0, 1, 0, 1, 0, 1, 0, 1);\n\
    VXC_Clamp(axisData, axisInExp, constAxis, constAxis, VXC_MODIFIER_CLAMP(0, 7, 0, 1));\n\
    axisData &= (vxc_uchar8)(1);\n\
    VXC_DP2x8(tmpOut, axisData, dinExp, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), ucharMulShort_8x8_2);\n\
\n\
    _viv_asm(COPY, dout, tmpOut, 16);\n\
    VXC_DP4x4(tmpVal1, dout, dout, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertFstFp16Fp32_4x4);\n\
    VXC_DP4x4(tmpVal2, dout, dout, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertSecFp16Fp32_4x4);\n\
    tmpVal1 *= up_outFlScale_i16;\n\
    tmpVal2 *= up_outFlScale_i16;\n\
    tmpData1 = convert_int4_rte(tmpVal1);\n\
    tmpData2 = convert_int4_rte(tmpVal2);\n\
    VXC_DP2x8(result, tmpData1, tmpData2, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\n\
        uniConvertInt32toUint8_2x8);\n\
    VXC_WriteImage2DArray(dataOut, coordOut, result, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    constAxis = (vxc_uchar8)(2, 3, 2, 3, 2, 3, 2, 3);\n\
    VXC_Clamp(axisData, axisInExp, constAxis, constAxis, VXC_MODIFIER_CLAMP(0, 7, 0, 1));\n\
    axisData &= (vxc_uchar8)(1);\n\
    VXC_DP2x8(tmpOut, axisData, dinExp, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), ucharMulShort_8x8_2);\n\
    coordOut.y += 1;\n\
    _viv_asm(COPY, dout, tmpOut, 16);\n\
    VXC_DP4x4(tmpVal1, dout, dout, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertFstFp16Fp32_4x4);\n\
    VXC_DP4x4(tmpVal2, dout, dout, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertSecFp16Fp32_4x4);\n\
    tmpVal1 *= up_outFlScale_i16;\n\
    tmpVal2 *= up_outFlScale_i16;\n\
    tmpData1 = convert_int4_rte(tmpVal1);\n\
    tmpData2 = convert_int4_rte(tmpVal2);\n\
    VXC_DP2x8(result, tmpData1, tmpData2, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\n\
        uniConvertInt32toUint8_2x8);\n\
    VXC_WriteImage2DArray(dataOut, coordOut, result, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_upsample_2_vx*/

static const char vsi_nn_kernel_upsample_i8_vx[] = "\n\
#include \"cl_viv_vx_ext.h\"\n\
\n\
//--------------------------unpooling int8-------------------------\n\
__kernel void unpoolingInt8\n\
    (\n\
        image2d_array_t dataIn,\n\
        image2d_array_t axis,\n\
        image2d_array_t dataOut,\n\
        unsigned int sizeX,\n\
        unsigned int sizeY\n\
    )\n\
{\n\
    vxc_char8 din;\n\
    vxc_uchar8 axisIn;\n\
    vxc_char16 dinExpand;\n\
    vxc_uchar16 axisInExpand;\n\
    vxc_uchar16 constAxis;\n\
    vxc_uchar16 axisData;\n\
    vxc_char16 axisData1;\n\
    vxc_char16 dout;\n\
\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    VXC_ReadImage2DArray(din, dataIn, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(axisIn, axis, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    int4 coordOut = (int4)(coord.x << 1, coord.y << 1, coord.z, 0);\n\
    dinExpand = din.s0011223344556677;\n\
    axisInExpand = axisIn.s0011223344556677;\n\
\n\
    constAxis = (vxc_uchar16)(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);\n\
    VXC_Clamp(axisData, axisInExpand, constAxis, constAxis, VXC_MODIFIER_CLAMP(0, 15, 0, 1));\n\
    axisData &= (vxc_uchar16)(1);\n\
    _viv_asm(COPY, axisData1, axisData, 16);\n\
    dout = axisData1 * dinExpand;\n\
    VXC_WriteImage2DArray(dataOut, coordOut, dout, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    constAxis = (vxc_uchar16)(2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3);\n\
    VXC_Clamp(axisData, axisInExpand, constAxis, constAxis, VXC_MODIFIER_CLAMP(0, 15, 0, 1));\n\
    axisData &= (vxc_uchar16)(1);\n\
    _viv_asm(COPY, axisData1, axisData, 16);\n\
    dout = axisData1 * dinExpand;\n\
    coordOut.y += 1;\n\
    VXC_WriteImage2DArray(dataOut, coordOut, dout, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
//--------------------------unpooling uint8-------------------------\n\
_viv_uniform VXC_512Bits uniConvertDirUint8Fp32_4x4_2;\n\
_viv_uniform VXC_512Bits uniConvertEndUint8Fp32_4x4_2;\n\
_viv_uniform VXC_512Bits uniConvertTrdUint8Fp32_4x4_2;\n\
_viv_uniform VXC_512Bits uniConvertFthUint8Fp32_4x4_2;\n\
_viv_uniform VXC_512Bits uniConvertInt32toUint8_2x8_2;\n\
\n\
__kernel void unpoolingUint8_Uint8\n\
    (\n\
        image2d_array_t dataIn,\n\
        image2d_array_t axis,\n\
        image2d_array_t dataOut,\n\
        unsigned int sizeX,\n\
        unsigned int sizeY\n\
    )\n\
{\n\
    vxc_uchar8  din;\n\
    vxc_uchar8  axisIn;\n\
    vxc_uchar16 dinExpand;\n\
    vxc_uchar16 axisInExpand;\n\
    vxc_uchar16 constAxis;\n\
    vxc_uchar16 axisData;\n\
    vxc_uchar16 axisData1;\n\
    vxc_uchar16 dout;\n\
\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    VXC_ReadImage2DArray(din, dataIn, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(axisIn, axis, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    int4 coordOut = (int4)(coord.x << 1, coord.y << 1, coord.z, 0);\n\
\n\
    constAxis      = (vxc_uchar16)(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);\n\
    dinExpand    = din.s0011223344556677;\n\
    axisInExpand = axisIn.s0011223344556677;\n\
    VXC_Clamp(axisData, axisInExpand, constAxis, constAxis, VXC_MODIFIER_CLAMP(0, 15, 0, 1));\n\
    axisData &= (vxc_uchar16)(1);\n\
    _viv_asm(COPY, axisData1, axisData, 16);\n\
    dout = axisData1 * dinExpand;   //output\n\
    VXC_WriteImage2DArray(dataOut, coordOut, dout, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    constAxis = (vxc_uchar16)(2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3);\n\
    VXC_Clamp(axisData, axisInExpand, constAxis, constAxis, VXC_MODIFIER_CLAMP(0, 15, 0, 1));\n\
    axisData &= (vxc_uchar16)(1);\n\
    _viv_asm(COPY, axisData1, axisData, 16);\n\
    dout = axisData1 * dinExpand;  //output\n\
    coordOut.y += 1;\n\
    VXC_WriteImage2DArray(dataOut, coordOut, dout, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform float inputFl_i8;\n\
_viv_uniform float upInFl_i16;\n\
\n\
__kernel void unpoolingInt8_Fp16\n\
    (\n\
        image2d_array_t dataIn,\n\
        image2d_array_t axis,\n\
        image2d_array_t dataOut,\n\
        unsigned int sizeX,\n\
        unsigned int sizeY\n\
    )\n\
{\n\
    vxc_char8 din;\n\
    vxc_uchar8 axisIn;\n\
    vxc_char16 dinExpand;\n\
    vxc_uchar16 axisInExpand;\n\
    vxc_uchar16 constAxis;\n\
    vxc_uchar16 axisData;\n\
    vxc_char16 axisData1;\n\
    vxc_char16 dout;\n\
\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    VXC_ReadImage2DArray(din, dataIn, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(axisIn, axis, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    int4 coordOut = (int4)(coord.x << 1, coord.y << 1, coord.z, 0);\n\
    int4 coordOut1 = coordOut;\n\
    coordOut1.x += 8;\n\
    dinExpand = din.s0011223344556677;\n\
    axisInExpand = axisIn.s0011223344556677;\n\
    constAxis = (vxc_uchar16)(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);\n\
    VXC_Clamp(axisData, axisInExpand, constAxis, constAxis, VXC_MODIFIER_CLAMP(0, 15, 0, 1));\n\
    axisData &= (vxc_uchar16)(1);\n\
    _viv_asm(COPY, axisData1, axisData, 16);\n\
    dout = axisData1 * dinExpand;   //output\n\
\n\
    vxc_float4 tmpVal0, tmpVal1, tmpVal2, tmpVal3;\n\
    half4 tmpOut0, tmpOut1;\n\
    vxc_short8 rout0, rout1;\n\
\n\
    VXC_DP4x4(tmpVal0, dout, dout, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertDirUint8Fp32_4x4_2);\n\
    VXC_DP4x4(tmpVal1, dout, dout, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertEndUint8Fp32_4x4_2);\n\
    VXC_DP4x4(tmpVal2, dout, dout, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertTrdUint8Fp32_4x4_2);\n\
    VXC_DP4x4(tmpVal3, dout, dout, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertFthUint8Fp32_4x4_2);\n\
    tmpVal0 *= inputFl_i8;\n\
    tmpVal1 *= inputFl_i8;\n\
    tmpVal2 *= inputFl_i8;\n\
    tmpVal3 *= inputFl_i8;\n\
    _viv_asm(CONV, tmpOut0, tmpVal0);\n\
    _viv_asm(CONV, tmpOut1, tmpVal1);\n\
    VXC_DP2x8(rout0, tmpOut0, tmpOut1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertInt32toUint8_2x8_2);\n\
    _viv_asm(CONV, tmpOut0, tmpVal2);\n\
    _viv_asm(CONV, tmpOut1, tmpVal3);\n\
    VXC_DP2x8(rout1, tmpOut0, tmpOut1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertInt32toUint8_2x8_2);\n\
    VXC_WriteImage2DArray(dataOut, coordOut, rout0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_WriteImage2DArray(dataOut, coordOut1, rout1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    constAxis = (vxc_uchar16)(2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3);\n\
    VXC_Clamp(axisData, axisInExpand, constAxis, constAxis, VXC_MODIFIER_CLAMP(0, 15, 0, 1));\n\
    axisData &= (vxc_uchar16)(1);\n\
    _viv_asm(COPY, axisData1, axisData, 16);\n\
    dout = axisData1 * dinExpand;  //output\n\
\n\
    VXC_DP4x4(tmpVal0, dout, dout, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertDirUint8Fp32_4x4_2);\n\
    VXC_DP4x4(tmpVal1, dout, dout, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertEndUint8Fp32_4x4_2);\n\
    VXC_DP4x4(tmpVal2, dout, dout, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertTrdUint8Fp32_4x4_2);\n\
    VXC_DP4x4(tmpVal3, dout, dout, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertFthUint8Fp32_4x4_2);\n\
    tmpVal0 *= inputFl_i8;\n\
    tmpVal1 *= inputFl_i8;\n\
    tmpVal2 *= inputFl_i8;\n\
    tmpVal3 *= inputFl_i8;\n\
    _viv_asm(CONV, tmpOut0, tmpVal0);\n\
    _viv_asm(CONV, tmpOut1, tmpVal1);\n\
    VXC_DP2x8(rout0, tmpOut0, tmpOut1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertInt32toUint8_2x8_2);\n\
    _viv_asm(CONV, tmpOut0, tmpVal2);\n\
    _viv_asm(CONV, tmpOut1, tmpVal3);\n\
    VXC_DP2x8(rout1, tmpOut0, tmpOut1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertInt32toUint8_2x8_2);\n\
\n\
    coordOut.y += 1;\n\
    coordOut1.y += 1;\n\
    VXC_WriteImage2DArray(dataOut, coordOut, rout0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_WriteImage2DArray(dataOut, coordOut1, rout1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform VXC_512Bits ucharMulShort_8x8_3;\n\
_viv_uniform VXC_512Bits uniConvertFstInt16Fp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertSecInt16Fp32_4x4;\n\
\n\
__kernel void unpoolingInt16_Fp16  //fp16->int16\n\
    (\n\
        image2d_array_t dataIn,\n\
        image2d_array_t axis,\n\
        image2d_array_t dataOut,\n\
        unsigned int sizeX,\n\
        unsigned int sizeY\n\
    )\n\
{\n\
    vxc_short4 din;\n\
    vxc_uchar4 axisIn;\n\
    vxc_short8 dinExp, tmpOut;\n\
    vxc_uchar8 axisInExp;\n\
    vxc_uchar8 constAxis;\n\
    vxc_uchar8 axisData;\n\
    half4 tmpOut0, tmpOut1;\n\
    float4 tmpVal1, tmpVal2;\n\
    vxc_short8 result;\n\
\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    VXC_ReadImage2DArray(din, dataIn, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(axisIn, axis, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    int4 coordOut = (int4)(coord.x << 1, coord.y << 1, coord.z, 0);\n\
    dinExp = din.s00112233;\n\
    axisInExp = axisIn.s00112233;\n\
    constAxis = (vxc_uchar8)(0, 1, 0, 1, 0, 1, 0, 1);\n\
    VXC_Clamp(axisData, axisInExp, constAxis, constAxis, VXC_MODIFIER_CLAMP(0, 7, 0, 1));\n\
    axisData &= (vxc_uchar8)(1);\n\
    VXC_DP2x8(tmpOut, axisData, dinExp, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
        ucharMulShort_8x8_3);\n\
\n\
    VXC_DP4x4(tmpVal1, tmpOut, tmpOut, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertFstInt16Fp32_4x4);\n\
    VXC_DP4x4(tmpVal2, tmpOut, tmpOut, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertSecInt16Fp32_4x4);\n\
    tmpVal1 *= upInFl_i16;\n\
    tmpVal2 *= upInFl_i16;\n\
    _viv_asm(CONV, tmpOut0, tmpVal1);\n\
    _viv_asm(CONV, tmpOut1, tmpVal2);\n\
    VXC_DP2x8(result, tmpOut0, tmpOut1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertInt32toUint8_2x8_2);\n\
    VXC_WriteImage2DArray(dataOut, coordOut, result, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    constAxis = (vxc_uchar8)(2, 3, 2, 3, 2, 3, 2, 3);\n\
    VXC_Clamp(axisData, axisInExp, constAxis, constAxis, VXC_MODIFIER_CLAMP(0, 7, 0, 1));\n\
    axisData &= (vxc_uchar8)(1);\n\
    VXC_DP2x8(tmpOut, axisData, dinExp, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
        ucharMulShort_8x8_3);\n\
    coordOut.y += 1;\n\
    VXC_DP4x4(tmpVal1, tmpOut, tmpOut, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertFstInt16Fp32_4x4);\n\
    VXC_DP4x4(tmpVal2, tmpOut, tmpOut, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertSecInt16Fp32_4x4);\n\
    tmpVal1 *= upInFl_i16;\n\
    tmpVal2 *= upInFl_i16;\n\
    _viv_asm(CONV, tmpOut0, tmpVal1);\n\
    _viv_asm(CONV, tmpOut1, tmpVal2);\n\
    VXC_DP2x8(result, tmpOut0, tmpOut1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertInt32toUint8_2x8_2);\n\
    VXC_WriteImage2DArray(dataOut, coordOut, result, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}"; /* end of vsi_nn_kernel_upsample_i8_vx*/

static const char vsi_nn_kernel_upsample_opt_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
_viv_uniform VXC_512Bits uniQuantInOutInt16_2x8;\n\
_viv_uniform VXC_512Bits ucharMulShort_8x8_opt;\n\
\n\
_viv_uniform VXC_512Bits uniQuantInOutInt8_2x8;\n\
_viv_uniform VXC_512Bits uniQuantInOutInt8Hi_2x8;\n\
\n\
__kernel void unpoolingInt8_Int8_opt\n\
    (\n\
        image2d_array_t dataIn,\n\
        image2d_array_t axis,\n\
        image2d_array_t dataOut,\n\
        unsigned int sizeX,\n\
        unsigned int sizeY\n\
    )\n\
{\n\
    vxc_char8 din;\n\
    vxc_uchar8 axisIn;\n\
    vxc_char16 dinExpand;\n\
    vxc_uchar16 axisInExpand;\n\
    vxc_uchar16 constAxis;\n\
    vxc_uchar16 axisData;\n\
    vxc_char16 axisData1;\n\
    vxc_char16 dout;\n\
    vxc_char16 result;\n\
\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    VXC_ReadImage2DArray(din, dataIn, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(axisIn, axis, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    int4 coordOut = (int4)(coord.x << 1, coord.y << 1, coord.z, 0);\n\
    dinExpand = din.s0011223344556677;\n\
    axisInExpand = axisIn.s0011223344556677;\n\
\n\
    constAxis = (vxc_uchar16)(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);\n\
    VXC_Clamp(axisData, axisInExpand, constAxis, constAxis, VXC_MODIFIER_CLAMP(0, 15, 0, 1));\n\
    axisData &= (vxc_uchar16)(1);\n\
    _viv_asm(COPY, axisData1, axisData, 16);\n\
    dout = axisData1 * dinExpand;\n\
\n\
    VXC_DP2x8(result, dout, dout, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniQuantInOutInt8_2x8);\n\
    VXC_DP2x8(result, dout, dout, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniQuantInOutInt8Hi_2x8);\n\
    VXC_WriteImage2DArray(dataOut, coordOut, result, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    constAxis = (vxc_uchar16)(2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3);\n\
    VXC_Clamp(axisData, axisInExpand, constAxis, constAxis, VXC_MODIFIER_CLAMP(0, 15, 0, 1));\n\
    axisData &= (vxc_uchar16)(1);\n\
    _viv_asm(COPY, axisData1, axisData, 16);\n\
    dout = axisData1 * dinExpand;\n\
    coordOut.y += 1;\n\
\n\
    VXC_DP2x8(result, dout, dout, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniQuantInOutInt8_2x8);\n\
    VXC_DP2x8(result, dout, dout, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniQuantInOutInt8Hi_2x8);\n\
    VXC_WriteImage2DArray(dataOut, coordOut, result, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
//--------------------------unpooling int16-------------------------\n\
__kernel void unpoolingInt16_Int16_opt\n\
    (\n\
        image2d_array_t dataIn,\n\
        image2d_array_t axis,\n\
        image2d_array_t dataOut,\n\
        unsigned int sizeX,\n\
        unsigned int sizeY\n\
    )\n\
{\n\
    vxc_short4 din;\n\
    vxc_uchar4 axisIn;\n\
    vxc_short8 dinExp;\n\
    vxc_uchar8 axisInExp;\n\
    vxc_uchar8 constAxis;\n\
    vxc_uchar8 axisData;\n\
    vxc_short8 dout;\n\
\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    VXC_ReadImage2DArray(din, dataIn, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(axisIn, axis, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0));\n\
    int4 coordOut = (int4)(coord.x << 1, coord.y << 1, coord.z, 0);\n\
    dinExp = din.s00112233;\n\
    axisInExp = axisIn.s00112233;\n\
    constAxis = (vxc_uchar8)(0, 1, 0, 1, 0, 1, 0, 1);\n\
    VXC_Clamp(axisData, axisInExp, constAxis, constAxis, VXC_MODIFIER_CLAMP(0, 7, 0, 1));\n\
    axisData &= (vxc_uchar8)(1);\n\
    VXC_DP2x8(dout, axisData, dinExp, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), ucharMulShort_8x8_opt);\n\
    //dout = (axisData == constAxis) ? dinExp : constZeros;\n\
\n\
    VXC_DP2x8(dout, dout, dout, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniQuantInOutInt16_2x8);\n\
    VXC_WriteImage2DArray(dataOut, coordOut, dout, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    constAxis = (vxc_uchar8)(2, 3, 2, 3, 2, 3, 2, 3);\n\
    VXC_Clamp(axisData, axisInExp, constAxis, constAxis, VXC_MODIFIER_CLAMP(0, 7, 0, 1));\n\
    axisData &= (vxc_uchar8)(1);\n\
    VXC_DP2x8(dout, axisData, dinExp, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), ucharMulShort_8x8_opt);\n\
\n\
    VXC_DP2x8(dout, dout, dout, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniQuantInOutInt16_2x8);\n\
    coordOut.y += 1;\n\
    VXC_WriteImage2DArray(dataOut, coordOut, dout, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}"; /* end of vsi_nn_kernel_upsample_opt_vx*/

static const char vsi_nn_kernel_upsample_u8_vx[] = "\n\
#include \"cl_viv_vx_ext.h\"\n\
\n\
//--------------------------unpooling uint8-------------------------\n\
__kernel void unpoolingUint8_Uint8_2D\n\
    (\n\
        image2d_array_t dataIn,\n\
        image2d_array_t axis,\n\
        image2d_array_t dataOut,\n\
        unsigned int sizeX,\n\
        unsigned int sizeY\n\
    )\n\
{\n\
    vxc_uchar8  din;\n\
    vxc_uchar8  axisIn;\n\
    vxc_uchar16 dinExpand;\n\
    vxc_uchar16 axisInExpand;\n\
    vxc_uchar16 constAxis;\n\
    vxc_uchar16 axisData;\n\
    vxc_uchar16 axisData1;\n\
    vxc_uchar16 dout;\n\
\n\
    int2 coord = (int2)(get_global_id(0), get_global_id(1));\n\
    VXC_ReadImage(din, dataIn, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(axisIn, axis, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    int2 coordOut = coord << 1;\n\
\n\
    constAxis      = (vxc_uchar16)(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);\n\
    dinExpand    = din.s0011223344556677;\n\
    axisInExpand = axisIn.s0011223344556677;\n\
    VXC_Clamp(axisData, axisInExpand, constAxis, constAxis, VXC_MODIFIER_CLAMP(0, 15, 0, 1));\n\
    axisData &= (vxc_uchar16)(1);\n\
    _viv_asm(COPY, axisData1, axisData, 16);\n\
    dout = axisData1 * dinExpand;   //output\n\
    VXC_WriteImage(dataOut, coordOut, dout, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    constAxis = (vxc_uchar16)(2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3);\n\
    VXC_Clamp(axisData, axisInExpand, constAxis, constAxis, VXC_MODIFIER_CLAMP(0, 15, 0, 1));\n\
    axisData &= (vxc_uchar16)(1);\n\
    _viv_asm(COPY, axisData1, axisData, 16);\n\
    dout = axisData1 * dinExpand;  //output\n\
    coordOut.y += 1;\n\
    VXC_WriteImage(dataOut, coordOut, dout, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform VXC_512Bits uniF16MulMultipiler_PostShft_2x8;\n\
_viv_uniform VXC_512Bits uniS16AddOutZP_2x8;\n\
_viv_uniform vxc_uint4 packed_outputZP;\n\
__kernel void unpoolingFp16_Uint8_2D\n\
    (\n\
        image2d_array_t dataIn,\n\
        image2d_array_t axis,\n\
        image2d_array_t dataOut,\n\
        unsigned int sizeX,\n\
        unsigned int sizeY\n\
    )\n\
{\n\
    vxc_short8 din0;\n\
    vxc_uchar8 din;\n\
    vxc_uchar8 axisIn;\n\
    vxc_half8 src;\n\
\n\
    vxc_uchar16 dinExpand;\n\
    vxc_uchar16 axisInExpand;\n\
    vxc_uchar16 constAxis;\n\
    vxc_uchar16 axisData;\n\
    vxc_uchar16 axisData1;\n\
    vxc_uchar16 dout;\n\
\n\
    int2 coord = (int2)(get_global_id(0), get_global_id(1));\n\
    VXC_ReadImage(din, dataIn, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(axisIn, axis, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    int2 coordOut = coord << 1;\n\
\n\
    vxc_short8 tmp;\n\
    uchar zp = 0;\n\
    _viv_asm(COPY, src, din, 16);\n\
    VXC_DP2x8(tmp, src, src, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniF16MulMultipiler_PostShft_2x8);\n\
    vxc_uchar16 packed_outZP;\n\
    _viv_asm(COPY, packed_outZP, packed_outputZP, 16);\n\
    VXC_DP2x8(din, tmp, packed_outZP, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniS16AddOutZP_2x8);\n\
\n\
\n\
    constAxis      = (vxc_uchar16)(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);\n\
    dinExpand    = din.s0011223344556677;\n\
    axisInExpand = axisIn.s0011223344556677;\n\
    VXC_Clamp(axisData, axisInExpand, constAxis, constAxis, VXC_MODIFIER_CLAMP(0, 15, 0, 1));\n\
    axisData &= (vxc_uchar16)(1);\n\
    _viv_asm(COPY, axisData1, axisData, 16);\n\
    dout = axisData1 * dinExpand;\n\
    VXC_WriteImage(dataOut, coordOut, dout, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    constAxis = (vxc_uchar16)(2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3);\n\
    VXC_Clamp(axisData, axisInExpand, constAxis, constAxis, VXC_MODIFIER_CLAMP(0, 15, 0, 1));\n\
    axisData &= (vxc_uchar16)(1);\n\
    _viv_asm(COPY, axisData1, axisData, 16);\n\
    dout = axisData1 * dinExpand;  //output\n\
    coordOut.y += 1;\n\
    VXC_WriteImage(dataOut, coordOut, dout, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform VXC_512Bits uniMulMinusZpUint8_4x4;\n\
_viv_uniform VXC_512Bits uniMulMinusZp2Uint8_4x4;\n\
_viv_uniform VXC_512Bits uniMulMinusZp3Uint8_4x4;\n\
_viv_uniform VXC_512Bits uniMulMinusZp4Uint8_4x4;\n\
\n\
_viv_uniform VXC_512Bits uniConvertInt32toInt16_2x8;\n\
_viv_uniform VXC_512Bits uniConvertDirUint8Fp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertEndUint8Fp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertTrdUint8Fp32_4x4;\n\
_viv_uniform VXC_512Bits uniConvertFthUint8Fp32_4x4;\n\
\n\
_viv_uniform float scaleU8Fp16;\n\
_viv_uniform int zpU8Fp16;\n\
\n\
__kernel void unpoolingUint8_Fp16\n\
    (\n\
        image2d_array_t dataIn,\n\
        image2d_array_t axis,\n\
        image2d_array_t dataOut,\n\
        unsigned int sizeX,\n\
        unsigned int sizeY\n\
    )\n\
{\n\
    vxc_uchar8 din;\n\
    vxc_uchar8 axisIn;\n\
    vxc_uchar16 dinExpand;\n\
    vxc_uchar16 axisInExpand;\n\
    vxc_uchar16 constAxis;\n\
    vxc_uchar16 axisData;\n\
    vxc_uchar16 axisData1;\n\
    vxc_uchar16 dout;\n\
\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    VXC_ReadImage2DArray(din, dataIn, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(axisIn, axis, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    int4 coordOut = (int4)(coord.x << 1, coord.y << 1, coord.z, 0);\n\
    int4 coordOut1 = coordOut;\n\
    coordOut1.x += 8;\n\
    dinExpand = din.s0011223344556677;\n\
    axisInExpand = axisIn.s0011223344556677;\n\
    constAxis = (vxc_uchar16)(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);\n\
    VXC_Clamp(axisData, axisInExpand, constAxis, constAxis, VXC_MODIFIER_CLAMP(0, 15, 0, 1));\n\
    axisData &= (vxc_uchar16)(1);\n\
    _viv_asm(COPY, axisData1, axisData, 16);\n\
    dout = axisData1 * dinExpand;   //output\n\
\n\
    vxc_float4 tmpVal0, tmpVal1, tmpVal2, tmpVal3, convZp;\n\
    half4 tmpOut0, tmpOut1;\n\
    vxc_short8 rout0, rout1;\n\
    vxc_int4 tmpV0, tmpV1, tmpV2, tmpV3;\n\
    vxc_float4 tmpData0, tmpData1, tmpData2, tmpData3;\n\
    short tmpZp = (short)(-zpU8Fp16);\n\
    VXC_DP4x4(tmpVal0, dout, dout, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertDirUint8Fp32_4x4);\n\
    VXC_DP4x4(tmpVal1, dout, dout, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertEndUint8Fp32_4x4);\n\
    VXC_DP4x4(tmpVal2, dout, dout, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertTrdUint8Fp32_4x4);\n\
    VXC_DP4x4(tmpVal3, dout, dout, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertFthUint8Fp32_4x4);\n\
    VXC_DP4x4(tmpV0, axisData1, tmpZp, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniMulMinusZpUint8_4x4);\n\
    VXC_DP4x4(tmpV1, axisData1, tmpZp, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniMulMinusZp2Uint8_4x4);\n\
    VXC_DP4x4(tmpV2, axisData1, tmpZp, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniMulMinusZp3Uint8_4x4);\n\
    VXC_DP4x4(tmpV3, axisData1, tmpZp, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniMulMinusZp4Uint8_4x4);\n\
    _viv_asm(CONV, tmpData0, tmpV0);\n\
    _viv_asm(CONV, tmpData1, tmpV1);\n\
    _viv_asm(CONV, tmpData2, tmpV2);\n\
    _viv_asm(CONV, tmpData3, tmpV3);\n\
    tmpVal0 = (tmpVal0 + tmpData0) * scaleU8Fp16;\n\
    tmpVal1 = (tmpVal1 + tmpData1) * scaleU8Fp16;\n\
    tmpVal2 = (tmpVal2 + tmpData2) * scaleU8Fp16;\n\
    tmpVal3 = (tmpVal3 + tmpData3) * scaleU8Fp16;\n\
    _viv_asm(CONV, tmpOut0, tmpVal0);\n\
    _viv_asm(CONV, tmpOut1, tmpVal1);\n\
    VXC_DP2x8(rout0, tmpOut0, tmpOut1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertInt32toInt16_2x8);\n\
    _viv_asm(CONV, tmpOut0, tmpVal2);\n\
    _viv_asm(CONV, tmpOut1, tmpVal3);\n\
    VXC_DP2x8(rout1, tmpOut0, tmpOut1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertInt32toInt16_2x8);\n\
    VXC_WriteImage2DArray(dataOut, coordOut, rout0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_WriteImage2DArray(dataOut, coordOut1, rout1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    constAxis = (vxc_uchar16)(2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3);\n\
    VXC_Clamp(axisData, axisInExpand, constAxis, constAxis, VXC_MODIFIER_CLAMP(0, 15, 0, 1));\n\
    axisData &= (vxc_uchar16)(1);\n\
    _viv_asm(COPY, axisData1, axisData, 16);\n\
    dout = axisData1 * dinExpand;  //output\n\
\n\
    VXC_DP4x4(tmpVal0, dout, dout, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertDirUint8Fp32_4x4);\n\
    VXC_DP4x4(tmpVal1, dout, dout, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertEndUint8Fp32_4x4);\n\
    VXC_DP4x4(tmpVal2, dout, dout, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertTrdUint8Fp32_4x4);\n\
    VXC_DP4x4(tmpVal3, dout, dout, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertFthUint8Fp32_4x4);\n\
    VXC_DP4x4(tmpV0, axisData1, tmpZp, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniMulMinusZpUint8_4x4);\n\
    VXC_DP4x4(tmpV1, axisData1, tmpZp, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniMulMinusZp2Uint8_4x4);\n\
    VXC_DP4x4(tmpV2, axisData1, tmpZp, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniMulMinusZp3Uint8_4x4);\n\
    VXC_DP4x4(tmpV3, axisData1, tmpZp, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniMulMinusZp4Uint8_4x4);\n\
    _viv_asm(CONV, tmpData0, tmpV0);\n\
    _viv_asm(CONV, tmpData1, tmpV1);\n\
    _viv_asm(CONV, tmpData2, tmpV2);\n\
    _viv_asm(CONV, tmpData3, tmpV3);\n\
    tmpVal0 = (tmpVal0 + tmpData0) * scaleU8Fp16;\n\
    tmpVal1 = (tmpVal1 + tmpData1) * scaleU8Fp16;\n\
    tmpVal2 = (tmpVal2 + tmpData2) * scaleU8Fp16;\n\
    tmpVal3 = (tmpVal3 + tmpData3) * scaleU8Fp16;\n\
    _viv_asm(CONV, tmpOut0, tmpVal0);\n\
    _viv_asm(CONV, tmpOut1, tmpVal1);\n\
    VXC_DP2x8(rout0, tmpOut0, tmpOut1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertInt32toInt16_2x8);\n\
    _viv_asm(CONV, tmpOut0, tmpVal2);\n\
    _viv_asm(CONV, tmpOut1, tmpVal3);\n\
    VXC_DP2x8(rout1, tmpOut0, tmpOut1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
        uniConvertInt32toInt16_2x8);\n\
\n\
    coordOut.y += 1;\n\
    coordOut1.y += 1;\n\
    VXC_WriteImage2DArray(dataOut, coordOut, rout0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_WriteImage2DArray(dataOut, coordOut1, rout1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_upsample_u8_vx*/


const vsi_nn_vx_resource_item_type vx_resource_items[] =
{
    {"vsi_nn_kernel_addn_vx", vsi_nn_kernel_addn_vx},
    {"vsi_nn_kernel_argmax_axis0_vx", vsi_nn_kernel_argmax_axis0_vx},
    {"vsi_nn_kernel_argmax_axis1_vx", vsi_nn_kernel_argmax_axis1_vx},
    {"vsi_nn_kernel_argmax_axis2_vx", vsi_nn_kernel_argmax_axis2_vx},
    {"vsi_nn_kernel_argmin_axis0_vx", vsi_nn_kernel_argmin_axis0_vx},
    {"vsi_nn_kernel_argmin_axis1_vx", vsi_nn_kernel_argmin_axis1_vx},
    {"vsi_nn_kernel_argmin_axis2_vx", vsi_nn_kernel_argmin_axis2_vx},
    {"vsi_nn_kernel_axis_aligned_bbox_transform_vx", vsi_nn_kernel_axis_aligned_bbox_transform_vx},
    {"vsi_nn_kernel_box_with_nms_limit_vx", vsi_nn_kernel_box_with_nms_limit_vx},
    {"vsi_nn_kernel_clip_vx", vsi_nn_kernel_clip_vx},
    {"vsi_nn_kernel_crop_vx", vsi_nn_kernel_crop_vx},
    {"vsi_nn_kernel_detection_postprocess_vx", vsi_nn_kernel_detection_postprocess_vx},
    {"vsi_nn_kernel_dropout_vx", vsi_nn_kernel_dropout_vx},
    {"vsi_nn_kernel_elu_vx", vsi_nn_kernel_elu_vx},
    {"vsi_nn_kernel_exp_vx", vsi_nn_kernel_exp_vx},
    {"vsi_nn_kernel_extra_ending_vx", vsi_nn_kernel_extra_ending_vx},
    {"vsi_nn_kernel_floordiv_vx", vsi_nn_kernel_floordiv_vx},
    {"vsi_nn_kernel_fullconnect2_vx", vsi_nn_kernel_fullconnect2_vx},
    {"vsi_nn_kernel_gather_vx", vsi_nn_kernel_gather_vx},
    {"vsi_nn_kernel_generate_proposals_vx", vsi_nn_kernel_generate_proposals_vx},
    {"vsi_nn_kernel_header_vx", vsi_nn_kernel_header_vx},
    {"vsi_nn_kernel_heatmap_max_keypoint_vx", vsi_nn_kernel_heatmap_max_keypoint_vx},
    {"vsi_nn_kernel_imageprocess_vx", vsi_nn_kernel_imageprocess_vx},
    {"vsi_nn_kernel_imageprocess_2_vx", vsi_nn_kernel_imageprocess_2_vx},
    {"vsi_nn_kernel_imageprocess_3_vx", vsi_nn_kernel_imageprocess_3_vx},
    {"vsi_nn_kernel_imageprocess_4_vx", vsi_nn_kernel_imageprocess_4_vx},
    {"vsi_nn_kernel_imageprocess_5_vx", vsi_nn_kernel_imageprocess_5_vx},
    {"vsi_nn_kernel_instancenormalize_vx", vsi_nn_kernel_instancenormalize_vx},
    {"vsi_nn_kernel_instancenormalize_i8_vx", vsi_nn_kernel_instancenormalize_i8_vx},
    {"vsi_nn_kernel_internal_reduceProdAxis0_vx", vsi_nn_kernel_internal_reduceProdAxis0_vx},
    {"vsi_nn_kernel_internal_reduceProdAxis1_vx", vsi_nn_kernel_internal_reduceProdAxis1_vx},
    {"vsi_nn_kernel_internal_reduceProdAxis2_vx", vsi_nn_kernel_internal_reduceProdAxis2_vx},
    {"vsi_nn_kernel_internal_reduceallAxis0_vx", vsi_nn_kernel_internal_reduceallAxis0_vx},
    {"vsi_nn_kernel_internal_reduceallAxis1_vx", vsi_nn_kernel_internal_reduceallAxis1_vx},
    {"vsi_nn_kernel_internal_reduceallAxis2_vx", vsi_nn_kernel_internal_reduceallAxis2_vx},
    {"vsi_nn_kernel_internal_reduceanyAxis0_vx", vsi_nn_kernel_internal_reduceanyAxis0_vx},
    {"vsi_nn_kernel_internal_reduceanyAxis1_vx", vsi_nn_kernel_internal_reduceanyAxis1_vx},
    {"vsi_nn_kernel_internal_reduceanyAxis2_vx", vsi_nn_kernel_internal_reduceanyAxis2_vx},
    {"vsi_nn_kernel_internal_reducemaxAxis0_vx", vsi_nn_kernel_internal_reducemaxAxis0_vx},
    {"vsi_nn_kernel_internal_reducemaxAxis1_vx", vsi_nn_kernel_internal_reducemaxAxis1_vx},
    {"vsi_nn_kernel_internal_reducemaxAxis2_vx", vsi_nn_kernel_internal_reducemaxAxis2_vx},
    {"vsi_nn_kernel_internal_reduceminAxis0_vx", vsi_nn_kernel_internal_reduceminAxis0_vx},
    {"vsi_nn_kernel_internal_reduceminAxis1_vx", vsi_nn_kernel_internal_reduceminAxis1_vx},
    {"vsi_nn_kernel_internal_reduceminAxis2_vx", vsi_nn_kernel_internal_reduceminAxis2_vx},
    {"vsi_nn_kernel_l2normalizescale_vx", vsi_nn_kernel_l2normalizescale_vx},
    {"vsi_nn_kernel_layernormalize_vx", vsi_nn_kernel_layernormalize_vx},
    {"vsi_nn_kernel_log_vx", vsi_nn_kernel_log_vx},
    {"vsi_nn_kernel_log_softmaxAxis0_vx", vsi_nn_kernel_log_softmaxAxis0_vx},
    {"vsi_nn_kernel_log_softmaxAxis0_BF16_vx", vsi_nn_kernel_log_softmaxAxis0_BF16_vx},
    {"vsi_nn_kernel_log_softmaxAxis1_vx", vsi_nn_kernel_log_softmaxAxis1_vx},
    {"vsi_nn_kernel_log_softmaxAxis1_BF16_vx", vsi_nn_kernel_log_softmaxAxis1_BF16_vx},
    {"vsi_nn_kernel_log_softmaxAxis2_vx", vsi_nn_kernel_log_softmaxAxis2_vx},
    {"vsi_nn_kernel_logical_not_vx", vsi_nn_kernel_logical_not_vx},
    {"vsi_nn_kernel_logical_ops_vx", vsi_nn_kernel_logical_ops_vx},
    {"vsi_nn_kernel_lstmunit_activation_BP_F16_vx", vsi_nn_kernel_lstmunit_activation_BP_F16_vx},
    {"vsi_nn_kernel_lstmunit_activation_BP_U8_vx", vsi_nn_kernel_lstmunit_activation_BP_U8_vx},
    {"vsi_nn_kernel_lstmunit_activation_B_F16_vx", vsi_nn_kernel_lstmunit_activation_B_F16_vx},
    {"vsi_nn_kernel_lstmunit_activation_B_U8_vx", vsi_nn_kernel_lstmunit_activation_B_U8_vx},
    {"vsi_nn_kernel_lstmunit_activation_CBP_F16_vx", vsi_nn_kernel_lstmunit_activation_CBP_F16_vx},
    {"vsi_nn_kernel_lstmunit_activation_CBP_U8_vx", vsi_nn_kernel_lstmunit_activation_CBP_U8_vx},
    {"vsi_nn_kernel_lstmunit_activation_CB_F16_vx", vsi_nn_kernel_lstmunit_activation_CB_F16_vx},
    {"vsi_nn_kernel_lstmunit_activation_CB_U8_vx", vsi_nn_kernel_lstmunit_activation_CB_U8_vx},
    {"vsi_nn_kernel_lstmunit_activation_CLP_F16_vx", vsi_nn_kernel_lstmunit_activation_CLP_F16_vx},
    {"vsi_nn_kernel_lstmunit_activation_CL_F16_vx", vsi_nn_kernel_lstmunit_activation_CL_F16_vx},
    {"vsi_nn_kernel_lstmunit_activation_CSP_F16_vx", vsi_nn_kernel_lstmunit_activation_CSP_F16_vx},
    {"vsi_nn_kernel_lstmunit_activation_CSP_U8_vx", vsi_nn_kernel_lstmunit_activation_CSP_U8_vx},
    {"vsi_nn_kernel_lstmunit_activation_CS_F16_vx", vsi_nn_kernel_lstmunit_activation_CS_F16_vx},
    {"vsi_nn_kernel_lstmunit_activation_CS_U8_vx", vsi_nn_kernel_lstmunit_activation_CS_U8_vx},
    {"vsi_nn_kernel_lstmunit_activation_LP_F16_vx", vsi_nn_kernel_lstmunit_activation_LP_F16_vx},
    {"vsi_nn_kernel_lstmunit_activation_L_F16_vx", vsi_nn_kernel_lstmunit_activation_L_F16_vx},
    {"vsi_nn_kernel_lstmunit_activation_SP_F16_vx", vsi_nn_kernel_lstmunit_activation_SP_F16_vx},
    {"vsi_nn_kernel_lstmunit_activation_SP_U8_vx", vsi_nn_kernel_lstmunit_activation_SP_U8_vx},
    {"vsi_nn_kernel_lstmunit_activation_S_F16_vx", vsi_nn_kernel_lstmunit_activation_S_F16_vx},
    {"vsi_nn_kernel_lstmunit_activation_S_U8_vx", vsi_nn_kernel_lstmunit_activation_S_U8_vx},
    {"vsi_nn_kernel_matrixmul_vx", vsi_nn_kernel_matrixmul_vx},
    {"vsi_nn_kernel_matrixmul_fp16_vx", vsi_nn_kernel_matrixmul_fp16_vx},
    {"vsi_nn_kernel_matrixmul_transbp1_vx", vsi_nn_kernel_matrixmul_transbp1_vx},
    {"vsi_nn_kernel_matrixmul_transbp2_vx", vsi_nn_kernel_matrixmul_transbp2_vx},
    {"vsi_nn_kernel_maximum_vx", vsi_nn_kernel_maximum_vx},
    {"vsi_nn_kernel_maximum_fp16_vx", vsi_nn_kernel_maximum_fp16_vx},
    {"vsi_nn_kernel_maximum_i16_vx", vsi_nn_kernel_maximum_i16_vx},
    {"vsi_nn_kernel_minimum_vx", vsi_nn_kernel_minimum_vx},
    {"vsi_nn_kernel_minimum_fp16_vx", vsi_nn_kernel_minimum_fp16_vx},
    {"vsi_nn_kernel_minimum_i16_vx", vsi_nn_kernel_minimum_i16_vx},
    {"vsi_nn_kernel_neg_vx", vsi_nn_kernel_neg_vx},
    {"vsi_nn_kernel_poolwithargmax_vx", vsi_nn_kernel_poolwithargmax_vx},
    {"vsi_nn_kernel_poolwithargmax_i16_vx", vsi_nn_kernel_poolwithargmax_i16_vx},
    {"vsi_nn_kernel_poolwithargmax_i8_vx", vsi_nn_kernel_poolwithargmax_i8_vx},
    {"vsi_nn_kernel_poolwithargmax_opt_vx", vsi_nn_kernel_poolwithargmax_opt_vx},
    {"vsi_nn_kernel_poolwithargmax_u8_vx", vsi_nn_kernel_poolwithargmax_u8_vx},
    {"vsi_nn_kernel_pow_vx", vsi_nn_kernel_pow_vx},
    {"vsi_nn_kernel_pow_i8_i16_vx", vsi_nn_kernel_pow_i8_i16_vx},
    {"vsi_nn_kernel_pre_process_bgra_vx", vsi_nn_kernel_pre_process_bgra_vx},
    {"vsi_nn_kernel_pre_process_bgra_trans_vx", vsi_nn_kernel_pre_process_bgra_trans_vx},
    {"vsi_nn_kernel_pre_process_gray_vx", vsi_nn_kernel_pre_process_gray_vx},
    {"vsi_nn_kernel_pre_process_gray_copy_vx", vsi_nn_kernel_pre_process_gray_copy_vx},
    {"vsi_nn_kernel_pre_process_rgb_vx", vsi_nn_kernel_pre_process_rgb_vx},
    {"vsi_nn_kernel_pre_process_rgb_copy_vx", vsi_nn_kernel_pre_process_rgb_copy_vx},
    {"vsi_nn_kernel_pre_process_rgb_copy_trans_vx", vsi_nn_kernel_pre_process_rgb_copy_trans_vx},
    {"vsi_nn_kernel_pre_process_rgb_trans_vx", vsi_nn_kernel_pre_process_rgb_trans_vx},
    {"vsi_nn_kernel_pre_process_yuv420_vx", vsi_nn_kernel_pre_process_yuv420_vx},
    {"vsi_nn_kernel_pre_process_yuv420_scale_vx", vsi_nn_kernel_pre_process_yuv420_scale_vx},
    {"vsi_nn_kernel_prelu_vx", vsi_nn_kernel_prelu_vx},
    {"vsi_nn_kernel_prelu_i8_i16_vx", vsi_nn_kernel_prelu_i8_i16_vx},
    {"vsi_nn_kernel_prelu_u8_vx", vsi_nn_kernel_prelu_u8_vx},
    {"vsi_nn_kernel_random_multinomial_vx", vsi_nn_kernel_random_multinomial_vx},
    {"vsi_nn_kernel_relational_ops_vx", vsi_nn_kernel_relational_ops_vx},
    {"vsi_nn_kernel_relu_keras_header_vx", vsi_nn_kernel_relu_keras_header_vx},
    {"vsi_nn_kernel_relu_keras_internal_vx", vsi_nn_kernel_relu_keras_internal_vx},
    {"vsi_nn_kernel_resize_vx", vsi_nn_kernel_resize_vx},
    {"vsi_nn_kernel_reverse_vx", vsi_nn_kernel_reverse_vx},
    {"vsi_nn_kernel_roi_align_vx", vsi_nn_kernel_roi_align_vx},
    {"vsi_nn_kernel_scale_vx", vsi_nn_kernel_scale_vx},
    {"vsi_nn_kernel_select_vx", vsi_nn_kernel_select_vx},
    {"vsi_nn_kernel_shufflechannel_vx", vsi_nn_kernel_shufflechannel_vx},
    {"vsi_nn_kernel_signalframe_vx", vsi_nn_kernel_signalframe_vx},
    {"vsi_nn_kernel_sin_vx", vsi_nn_kernel_sin_vx},
    {"vsi_nn_kernel_space2depth_vx", vsi_nn_kernel_space2depth_vx},
    {"vsi_nn_kernel_stack_vx", vsi_nn_kernel_stack_vx},
    {"vsi_nn_kernel_tensor_add_mean_stddev_norm_vx", vsi_nn_kernel_tensor_add_mean_stddev_norm_vx},
    {"vsi_nn_kernel_tensorstackconcat_vx", vsi_nn_kernel_tensorstackconcat_vx},
    {"vsi_nn_kernel_tile_vx", vsi_nn_kernel_tile_vx},
    {"vsi_nn_kernel_topk_vx", vsi_nn_kernel_topk_vx},
    {"vsi_nn_kernel_transform_gemm_vx", vsi_nn_kernel_transform_gemm_vx},
    {"vsi_nn_kernel_transform_interp_vx", vsi_nn_kernel_transform_interp_vx},
    {"vsi_nn_kernel_transform_setupThres_vx", vsi_nn_kernel_transform_setupThres_vx},
    {"vsi_nn_kernel_unstack_vx", vsi_nn_kernel_unstack_vx},
    {"vsi_nn_kernel_upsample_vx", vsi_nn_kernel_upsample_vx},
    {"vsi_nn_kernel_upsample_2_vx", vsi_nn_kernel_upsample_2_vx},
    {"vsi_nn_kernel_upsample_i8_vx", vsi_nn_kernel_upsample_i8_vx},
    {"vsi_nn_kernel_upsample_opt_vx", vsi_nn_kernel_upsample_opt_vx},
    {"vsi_nn_kernel_upsample_u8_vx", vsi_nn_kernel_upsample_u8_vx},
};

const int vx_resource_items_cnt = _cnt_of_array(vx_resource_items);
