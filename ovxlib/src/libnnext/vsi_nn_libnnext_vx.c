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

static const char vsi_nn_kernel_argmax_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
//----------------------------------argmax----------------------------\n\
_viv_uniform int depth;\n\
_viv_uniform VXC_512Bits intToShort8;\n\
_viv_uniform VXC_512Bits uniExtractHalfMax_2x8;\n\
\n\
__kernel void argMaxVXC\n\
    (\n\
    image2d_array_t input,\n\
    image2d_array_t output\n\
    )\n\
{\n\
    vxc_short8 z;\n\
    vxc_short8 zz = (vxc_short8)(-1, -1, -1, -1, -1, -1, -1, -1);\n\
    vxc_short8 din0, axis;\n\
    vxc_half8 dinHalf0, maxHalf;\n\
    vxc_short8 max;\n\
    float4 fltMax = {FLT_MIN, FLT_MIN, FLT_MIN, FLT_MIN};\n\
    half4 hfMax;\n\
    _viv_asm(CONV, hfMax, fltMax);\n\
    vxc_half8 maxTmp;\n\
    VXC_DP2x8(maxTmp, hfMax, hfMax, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniExtractHalfMax_2x8);\n\
\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), depth-1, 0);\n\
    int4 coordOut = (int4)(get_global_id(0), get_global_id(1), 0, 0);\n\
    VXC_DP2x8(z, depth-1, depth, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), intToShort8);\n\
\n\
    while(coord.z >= 0)\n\
    {\n\
        VXC_ReadImage2DArray(din0, input, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        _viv_asm(COPY, dinHalf0, din0, 16);\n\
        VXC_VertMax3_Half(maxHalf, maxTmp, dinHalf0, dinHalf0, VXC_MODIFIER_BIN(0, 7, 0));\n\
        maxTmp = maxHalf;\n\
        _viv_asm(COPY, max, maxTmp, 16);\n\
        axis = (din0 == max) ? z : axis;\n\
        z += zz;\n\
        coord.z -= 1;\n\
    }\n\
    VXC_WriteImage2DArray(output, coordOut, axis, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
//----------------------------------argmax int8----------------------------\n\
\n\
__kernel void argMaxVXCInt8\n\
    (\n\
    image2d_array_t input,\n\
    image2d_array_t output\n\
    )\n\
{\n\
    vxc_char16 z;\n\
    vxc_char16 zz = (vxc_char16)(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);\n\
    vxc_char16 din0, axis;\n\
    vxc_char16 max;\n\
    vxc_char16 maxTmp = {-128, -128, -128, -128, -128, -128, -128, -128,\n\
                         -128, -128, -128, -128, -128, -128, -128, -128};\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), depth-1, 0);\n\
    int4 coordOut = (int4)(get_global_id(0), get_global_id(1), 0, 0);\n\
    VXC_DP2x8(z, depth-1, depth, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), intToShort8);\n\
    VXC_DP2x8(z, depth-1, depth, VXC_MODIFIER(8, 15, 0, VXC_RM_TowardZero, 0), intToShort8);\n\
\n\
    while(coord.z >= 0)\n\
    {\n\
        VXC_ReadImage2DArray(din0, input, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
            VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
        VXC_VertMax3_Integer(max, maxTmp, din0, din0, VXC_MODIFIER_BIN(0, 15, 0));\n\
        maxTmp = max;\n\
        axis = (din0 == max) ? z : axis;\n\
        z += zz;\n\
        coord.z -= 1;\n\
    }\n\
    VXC_WriteImage2DArray(output, coordOut, axis, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
//----------------------------------argmax int16----------------------------\n\
_viv_uniform int depth2;\n\
_viv_uniform VXC_512Bits intToShort8_2;\n\
\n\
__kernel void argMaxVXCInt16\n\
    (\n\
    image2d_array_t input,\n\
    image2d_array_t output\n\
    )\n\
{\n\
    vxc_short8 z;\n\
    vxc_short8 zz = (vxc_short8)(-1, -1, -1, -1, -1, -1, -1, -1);\n\
    vxc_short8 din0, axis;\n\
    vxc_short8 max;\n\
    vxc_short8 maxTmp = {-32768, -32768, -32768, -32768, -32768, -32768, -32768, -32768};\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), depth2-1, 0);\n\
    int4 coordOut = (int4)(get_global_id(0), get_global_id(1), 0, 0);\n\
    VXC_DP2x8(z, depth2-1, depth2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), intToShort8_2);\n\
\n\
    while(coord.z >= 0)\n\
    {\n\
        VXC_ReadImage2DArray(din0, input, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_VertMax3_Integer(max, maxTmp, din0, din0, VXC_MODIFIER_BIN(0, 7, 0));\n\
        maxTmp = max;\n\
        axis = (din0 == max) ? z : axis;\n\
        z += zz;\n\
        coord.z -= 1;\n\
    }\n\
    VXC_WriteImage2DArray(output, coordOut, axis, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
//----------------------------------argmax uint8----------------------------\n\
__kernel void argMaxVXCUint8\n\
    (\n\
    image2d_array_t input,\n\
    image2d_array_t output\n\
    )\n\
{\n\
    vxc_char16 z, axis;\n\
    vxc_char16 zz = (vxc_char16)(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);\n\
    vxc_uchar16 din0;\n\
    vxc_uchar16 max;\n\
    vxc_uchar16 maxTmp = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), depth2-1, 0);\n\
    int4 coordOut = (int4)(get_global_id(0), get_global_id(1), 0, 0);\n\
    VXC_DP2x8(z, depth2-1, depth2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), intToShort8_2);\n\
    VXC_DP2x8(z, depth2-1, depth2, VXC_MODIFIER(8, 15, 0, VXC_RM_TowardZero, 0), intToShort8_2);\n\
\n\
    while(coord.z >= 0)\n\
    {\n\
        VXC_ReadImage2DArray(din0, input, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
            VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
        VXC_VertMax3_Integer(max, maxTmp, din0, din0, VXC_MODIFIER_BIN(0, 15, 0));\n\
        maxTmp = max;\n\
        axis = (din0 == max) ? z : axis;\n\
        z += zz;\n\
        coord.z -= 1;\n\
    }\n\
    VXC_WriteImage2DArray(output, coordOut, axis, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void argMaxVXCUint8_Int16\n\
    (\n\
    image2d_array_t input,\n\
    image2d_array_t output\n\
    )\n\
{\n\
    vxc_short8 z, axis;\n\
    vxc_short8 zz = (vxc_short8)(-1, -1, -1, -1, -1, -1, -1, -1);\n\
    vxc_uchar8 din0, max;\n\
    vxc_uchar8 maxTmp = {0, 0, 0, 0, 0, 0, 0, 0};\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), depth2-1, 0);\n\
    int4 coordOut = (int4)(get_global_id(0), get_global_id(1), 0, 0);\n\
    VXC_DP2x8(z, depth2-1, depth2, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), intToShort8_2);\n\
\n\
    while(coord.z >= 0)\n\
    {\n\
        VXC_ReadImage2DArray(din0, input, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_VertMax3_Integer(max, maxTmp, din0, din0, VXC_MODIFIER_BIN(0, 7, 0));\n\
        maxTmp = max;\n\
        axis = (din0 == max) ? z : axis;\n\
        z += zz;\n\
        coord.z -= 1;\n\
    }\n\
    VXC_WriteImage2DArray(output, coordOut, axis, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform int depthsub1;\n\
_viv_uniform vxc_uint4 packedDepth;\n\
_viv_uniform VXC_512Bits uniPacekedU8toI16Lo_2x8;\n\
_viv_uniform VXC_512Bits uniPacekedU8toI16Hi_2x8;\n\
__kernel void argMax_U8_I16_WxHx256\n\
    (\n\
    image2d_array_t input,\n\
    image2d_array_t output\n\
    )\n\
{\n\
    vxc_uchar16 packedZ, axis = 0;\n\
    vxc_uchar16 src;\n\
    vxc_uchar16 max = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), depthsub1, 0);\n\
    _viv_asm(COPY, packedZ, packedDepth, 16);\n\
\n\
    do\n\
    {\n\
        VXC_ReadImage2DArray(src, input, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
            VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
        packedZ --;\n\
        coord.z --;\n\
        VXC_VertMax3_Integer(max, max, src, src, VXC_MODIFIER_BIN(0, 15, 0));\n\
        axis = (src == max) ? packedZ : axis;\n\
    } while(coord.z >= 0);\n\
\n\
    vxc_short8 dst0, dst1;\n\
    VXC_DP2x8(dst0, axis, axis, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniPacekedU8toI16Lo_2x8);\n\
    VXC_DP2x8(dst1, axis, axis, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniPacekedU8toI16Hi_2x8);\n\
    coord.z = 0;\n\
    VXC_WriteImage2DArray(output, coord, dst0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    coord.x += 8;\n\
    VXC_WriteImage2DArray(output, coord, dst1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void argMax_I8_I16_WxHx256\n\
    (\n\
    image2d_array_t input,\n\
    image2d_array_t output\n\
    )\n\
{\n\
    vxc_uchar16 packedZ, axis = 0, bits;\n\
    vxc_char16 src;\n\
    vxc_char16 maxData = {-128, -128, -128, -128, -128, -128, -128, -128, -128, -128,\\\n\
        -128, -128, -128, -128, -128, -128};\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), depthsub1, 0);\n\
    _viv_asm(COPY, packedZ, packedDepth, 16);\n\
\n\
    do\n\
    {\n\
        VXC_ReadImage2DArray(src, input, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
            VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
        packedZ --;\n\
        coord.z --;\n\
        maxData = max(maxData, src);\n\
        VXC_Clamp(bits, src, maxData, maxData, VXC_MODIFIER_CLAMP(0, 15, 0, 1));\n\
\n\
        axis = bits ? packedZ : axis;\n\
    } while(coord.z >= 0);\n\
\n\
    vxc_short8 dst0, dst1;\n\
    VXC_DP2x8(dst0, axis, axis, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniPacekedU8toI16Lo_2x8);\n\
    VXC_DP2x8(dst1, axis, axis, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), uniPacekedU8toI16Hi_2x8);\n\
    coord.z = 0;\n\
    VXC_WriteImage2DArray(output, coord, dst0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    coord.x += 8;\n\
    VXC_WriteImage2DArray(output, coord, dst1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void argMaxVXCInt8_Int16\n\
    (\n\
    image2d_array_t input,\n\
    image2d_array_t output\n\
    )\n\
{\n\
    vxc_short8 z, axis;\n\
    vxc_short8 zz = (vxc_short8)(-1, -1, -1, -1, -1, -1, -1, -1);\n\
    vxc_char8 din0, max;\n\
    vxc_char8 maxTmp = {-128, -128, -128, -128, -128, -128, -128, -128};\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), depth-1, 0);\n\
    int4 coordOut = (int4)(get_global_id(0), get_global_id(1), 0, 0);\n\
    VXC_DP2x8(z, depth-1, depth, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0), intToShort8);\n\
\n\
    while(coord.z >= 0)\n\
    {\n\
        VXC_ReadImage2DArray(din0, input, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
            VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
        VXC_VertMax3_Integer(max, maxTmp, din0, din0, VXC_MODIFIER_BIN(0, 7, 0));\n\
        maxTmp = max;\n\
        axis = (din0 == max) ? z : axis;\n\
        z += zz;\n\
        coord.z -= 1;\n\
    }\n\
    VXC_WriteImage2DArray(output, coordOut, axis, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
"; /* end of vsi_nn_kernel_argmax_vx*/

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
    int4 coord_out = (int4)(get_global_id(0) - offset0, get_global_id(1)\\\n\
        - offset1, get_global_id(2) - offset2, 0);\n\
\n\
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
    int4 coord_out = (int4)(get_global_id(0) - offset0, get_global_id(1) - offset1,\\\n\
        get_global_id(2) - offset2, 0);\n\
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
\n\
    VXC_WriteImage2DArray(output, coord_out, src0, VXC_MODIFIER(0, 15, 0,VXC_RM_TowardZero, 0));\n\
    coord_out.y ++;\n\
    VXC_WriteImage2DArray(output, coord_out, src1, VXC_MODIFIER(0, 15, 0,VXC_RM_TowardZero, 0));\n\
    coord_out.y ++;\n\
    VXC_WriteImage2DArray(output, coord_out, src2, VXC_MODIFIER(0, 15, 0,VXC_RM_TowardZero, 0));\n\
    coord_out.y ++;\n\
    VXC_WriteImage2DArray(output, coord_out, src3, VXC_MODIFIER(0, 15, 0,VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_crop_vx*/

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

static const char vsi_nn_kernel_eltwisemax_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
//-----------------------eltwise max--------------------------------\n\
__kernel void eltwiseMax\n\
    (\n\
    __read_only image2d_array_t input1,\n\
    __read_only image2d_array_t input2,\n\
    __write_only image2d_array_t output\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    vxc_short8 din0, din1;\n\
    vxc_half8 din0Fp16, din1Fp16, maxData;\n\
\n\
    VXC_ReadImage2DArray(din0, input1, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(din1, input2, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, din0Fp16, din0, 16);\n\
    _viv_asm(COPY, din1Fp16, din1, 16);\n\
    VXC_VertMax3_Half(maxData, din0Fp16, din1Fp16, din1Fp16, VXC_MODIFIER_BIN(0, 7, 0));\n\
    _viv_asm(COPY, din0, maxData, 16);\n\
    VXC_WriteImage2DArray(output, coord, din0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform VXC_512Bits uniDFP8toDFP8Lo_2x8_0;\n\
_viv_uniform VXC_512Bits uniDFP8toDFP8Hi_2x8_0;\n\
_viv_uniform VXC_512Bits uniDFP8toDFP8Lo_2x8_1;\n\
_viv_uniform VXC_512Bits uniDFP8toDFP8Hi_2x8_1;\n\
__kernel void eltwiseMax_int8\n\
    (\n\
    __read_only image2d_array_t input1,\n\
    __read_only image2d_array_t input2,\n\
    __write_only image2d_array_t output\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    vxc_char16 src0, src1;\n\
\n\
    VXC_ReadImage2DArray(src0, input1, coord, 0, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input2, coord, 0, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_DP2x8(src0, src0, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniDFP8toDFP8Lo_2x8_0);\n\
    VXC_DP2x8(src0, src0, src0, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniDFP8toDFP8Hi_2x8_0);\n\
    VXC_DP2x8(src1, src1, src1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniDFP8toDFP8Lo_2x8_1);\n\
    VXC_DP2x8(src1, src1, src1, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniDFP8toDFP8Hi_2x8_1);\n\
    src0 = max(src0, src1);\n\
\n\
    VXC_WriteImage2DArray(output, coord, src0, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
}\n\
__kernel void eltwiseMax_int8_nofl\n\
    (\n\
    __read_only image2d_array_t input1,\n\
    __read_only image2d_array_t input2,\n\
    __write_only image2d_array_t output\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    vxc_char16 src0, src1;\n\
\n\
    VXC_ReadImage2DArray(src0, input1, coord, 0, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input2, coord, 0, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    src0 = max(src0, src1);\n\
    VXC_WriteImage2DArray(output, coord, src0, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform VXC_512Bits uniU8MulAndPostShift0_Lo_2x8;\n\
_viv_uniform VXC_512Bits uniU8MulAndPostShift0_Hi_2x8;\n\
_viv_uniform VXC_512Bits uniU8MulAndPostShift1_Lo_2x8;\n\
_viv_uniform VXC_512Bits uniU8MulAndPostShift1_Hi_2x8;\n\
_viv_uniform int2 multAndoutZP0;//[0:15] multiplier, [31:63] output zp\n\
_viv_uniform int2 multAndoutZP1;//[0:15] multiplier, [31:63] output zp\n\
__kernel void eltwiseMax_uint8\n\
    (\n\
    __read_only image2d_array_t input1,\n\
    __read_only image2d_array_t input2,\n\
    __write_only image2d_array_t output\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    vxc_uchar16 src0, src1, src2, src3, dst0, dst1;\n\
    VXC_ReadImage2DArray(src0, input1, coord, 0, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input2, coord, 0, VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    vxc_ushort8 mp0, mp1;\n\
    _viv_asm(COPY, mp0, multAndoutZP0, 16);\n\
    _viv_asm(COPY, mp1, multAndoutZP1, 16);\n\
    VXC_DP2x8(dst0, src0, mp0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniU8MulAndPostShift0_Lo_2x8);\n\
    VXC_DP2x8(dst0, src0, mp0, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniU8MulAndPostShift0_Hi_2x8);\n\
    VXC_DP2x8(dst1, src1, mp1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniU8MulAndPostShift1_Lo_2x8);\n\
    VXC_DP2x8(dst1, src1, mp1, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1),\\\n\
        uniU8MulAndPostShift1_Hi_2x8);\n\
    dst0 = max(dst0, dst1);\n\
    VXC_WriteImage2DArray(output, coord, dst0, VXC_MODIFIER(0, 15, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform VXC_512Bits uniDFP16toDFP16_2x8_0;\n\
_viv_uniform VXC_512Bits uniDFP16toDFP16_2x8_1;\n\
__kernel void eltwiseMax_int16\n\
    (\n\
    __read_only image2d_array_t input1,\n\
    __read_only image2d_array_t input2,\n\
    __write_only image2d_array_t output\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
    vxc_short8 src0, src1;\n\
\n\
    VXC_ReadImage2DArray(src0, input1, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input2, coord, 0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_DP2x8(src0, src0, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniDFP16toDFP16_2x8_0);\n\
    VXC_DP2x8(src1, src1, src1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniDFP16toDFP16_2x8_1);\n\
    src0 = max(src0, src1);\n\
\n\
    VXC_WriteImage2DArray(output, coord, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_eltwisemax_vx*/

static const char vsi_nn_kernel_elu_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
/***************************tensor elu********************************/\n\
_viv_uniform VXC_512Bits uniunPackedLoData_4x4;\n\
_viv_uniform VXC_512Bits uniunPackedHiData_4x4;\n\
_viv_uniform VXC_512Bits uniExtractHalf8_2x8_elu;\n\
_viv_uniform float scaleLogE;\n\
__kernel void tensorElu_fp16_2D\n\
    (\n\
    __read_only     image2d_array_t input,\n\
    __write_only    image2d_array_t output\n\
    )\n\
{\n\
    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(1), 0);\n\
\n\
    vxc_ushort8 vect0, vect1;\n\
    vxc_half8 vec0, vec1;\n\
    vxc_float4 vectA, vectB, vectC;\n\
    VXC_ReadImage(vect0, input,  coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, vec0, vect0, 16);\n\
    VXC_ReadImage(vect1, input,  coord.xy, VXC_5BITOFFSET_XY(0, 1),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, vec1, vect1, 16);\n\
    coord.z ++;\n\
\n\
    VXC_DP4x4(vectA, vec0, vec0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniunPackedLoData_4x4);\n\
    VXC_DP4x4(vectB, vec0, vec0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniunPackedHiData_4x4);\n\
    vectC = vectA * scaleLogE;\n\
    vectC.x = exp2(vectC.x);\n\
    vectC.y = exp2(vectC.y);\n\
    vectC.z = exp2(vectC.z);\n\
    vectC.w = exp2(vectC.w);\n\
    vectC -= 1;\n\
    vectA = vectA > 0 ? vectA : vectC;\n\
    half4 dst0, dst1;\n\
    _viv_asm(CONV, dst0, vectA);\n\
    vectC = vectB * scaleLogE;\n\
    vectC.x = exp2(vectC.x);\n\
    vectC.y = exp2(vectC.y);\n\
    vectC.z = exp2(vectC.z);\n\
    vectC.w = exp2(vectC.w);\n\
    vectC -= 1;\n\
    vectB = vectB > 0 ? vectB : vectC;\n\
    _viv_asm(CONV, dst1, vectB);\n\
    VXC_DP2x8(vec0, dst0, dst1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
        uniExtractHalf8_2x8_elu);\n\
    _viv_asm(COPY, vect0, vec0, 16);\n\
    VXC_WriteImage(output, coord.xy, vect0, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
\n\
    VXC_DP4x4(vectA, vec1, vec1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniunPackedLoData_4x4);\n\
    VXC_DP4x4(vectB, vec1, vec1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0),\\\n\
        uniunPackedHiData_4x4);\n\
    vectC = vectA * scaleLogE;\n\
    vectC.x = exp2(vectC.x);\n\
    vectC.y = exp2(vectC.y);\n\
    vectC.z = exp2(vectC.z);\n\
    vectC.w = exp2(vectC.w);\n\
    vectC -= 1;\n\
    vectA = vectA > 0 ? vectA : vectC;\n\
\n\
    _viv_asm(CONV, dst0, vectA);\n\
    vectC = vectB * scaleLogE;\n\
    vectC.x = exp2(vectC.x);\n\
    vectC.y = exp2(vectC.y);\n\
    vectC.z = exp2(vectC.z);\n\
    vectC.w = exp2(vectC.w);\n\
    vectC -= 1;\n\
    vectB = vectB > 0 ? vectB : vectC;\n\
    _viv_asm(CONV, dst1, vectB);\n\
    VXC_DP2x8(vec0, dst0, dst1, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0),\\\n\
        uniExtractHalf8_2x8_elu);\n\
    _viv_asm(COPY, vect0, vec0, 16);\n\
    VXC_WriteImage(output, coord.xz, vect0, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_elu_vx*/

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
#define VXC_Clamp_Half(dst, src0, src1, src2, info)\\\n\
do\\\n\
{\\\n\
    VXC_Clamp(dst, src0, src1, src2, info);\\\n\
} while (0)\n\
#endif\n\
"; /* end of vsi_nn_kernel_header_vx*/

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
    out_copy_type data; \\\n\
    _viv_asm(COPY, data, dst, 16); \\\n\
    VXC_WriteImage2DArray(output, coord, data, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
//             name0, name1, input_type, copy_type, output_type, out_copy_type, lgc_op\n\
TENSORLOGICAL(or,    int8,   vxc_char8,   vxc_char8,   vxc_char8,   vxc_char8,   ||)\n\
TENSORLOGICAL(or,    uint8,  vxc_uchar8,  vxc_uchar8,  vxc_char8,   vxc_uchar8,  ||)\n\
TENSORLOGICAL(or,    int16,  vxc_short8,  vxc_short8,  vxc_short8,  vxc_short8,  ||)\n\
//TENSORLOGICAL(or,    fp16,   vxc_short8,  vxc_half8,   vxc_short8,  vxc_short8,  ||)\n\
#endif\n\
"; /* end of vsi_nn_kernel_logical_ops_vx*/

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
"; /* end of vsi_nn_kernel_matrixmul_fp16_vx*/

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

static const char vsi_nn_kernel_minimum_vx[] = "#include \"cl_viv_vx_ext.h\"\n\
\n\
__kernel void vxcTensorMinimum_F16toF16\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_short8 vec0, vec1, vec2, vec3, dst;\n\
    vxc_half8  src0, src1, src2, src3;\n\
    VXC_ReadImage2DArray(vec0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, src0, vec0, 16);\n\
    VXC_ReadImage2DArray(vec1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, src1, vec1, 16);\n\
    VXC_ReadImage2DArray(vec2, input0, coord, VXC_5BITOFFSET_XY(0, 1),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, src2, vec2, 16);\n\
    VXC_ReadImage2DArray(vec3, input1, coord, VXC_5BITOFFSET_XY(0, 1),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, src3, vec3, 16);\n\
\n\
    VXC_VertMin3_Half(src0, src0, src1, src1, VXC_MODIFIER_CLAMP(0, 7, 0, 0));\n\
    _viv_asm(COPY, dst, src0, 16);\n\
\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
    coord.y ++;\n\
\n\
    VXC_VertMin3_Half(src2, src2, src3, src3, VXC_MODIFIER_CLAMP(0, 7, 0, 0));\n\
    _viv_asm(COPY, dst, src2, 16);\n\
\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcTensorMinimum_F16toF16_2D\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(1), get_global_id(1));\n\
\n\
    vxc_short8 vec0, vec1, vec2, vec3, dst;\n\
    vxc_half8  src0, src1, src2, src3;\n\
    VXC_ReadImage(vec0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, src0, vec0, 16);\n\
    VXC_ReadImage(vec1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, src1, vec1, 16);\n\
    VXC_ReadImage(vec2, input0, coord, VXC_5BITOFFSET_XY(0, 1),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, src2, vec2, 16);\n\
    VXC_ReadImage(vec3, input1, coord, VXC_5BITOFFSET_XY(0, 1),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    _viv_asm(COPY, src3, vec3, 16);\n\
\n\
    coord.z ++;\n\
\n\
    VXC_VertMin3_Half(src0, src0, src1, src1, VXC_MODIFIER_CLAMP(0, 7, 0, 0));\n\
    _viv_asm(COPY, dst, src0, 16);\n\
\n\
    VXC_WriteImage(output, coord, dst, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
\n\
    VXC_VertMin3_Half(src2, src2, src3, src3, VXC_MODIFIER_CLAMP(0, 7, 0, 0));\n\
    _viv_asm(COPY, dst, src2, 16);\n\
\n\
    VXC_WriteImage(output, coord.xz, dst, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform VXC_512Bits uniConvertI8toI8_0_part0_2x8;\n\
_viv_uniform VXC_512Bits uniConvertI8toI8_0_part1_2x8;\n\
_viv_uniform VXC_512Bits uniConvertI8toI8_1_part0_2x8;\n\
_viv_uniform VXC_512Bits uniConvertI8toI8_1_part1_2x8;\n\
__kernel void vxcTensorMinimum_I8toI8\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_char16 src0, src1, src2, src3, dst;\n\
    VXC_ReadImage2DArray(src0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src2, input0, coord, VXC_5BITOFFSET_XY(0, 1),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src3, input1, coord, VXC_5BITOFFSET_XY(0, 1),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_DP2x8(src0, src0, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI8toI8_0_part0_2x8);\n\
    VXC_DP2x8(src0, src0, src0, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniConvertI8toI8_0_part1_2x8);\n\
    VXC_DP2x8(src1, src1, src1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI8toI8_1_part0_2x8);\n\
    VXC_DP2x8(src1, src1, src1, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniConvertI8toI8_1_part1_2x8);\n\
    dst = min(src0, src1);\n\
\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 15, 0,VXC_RM_TowardZero, 0));\n\
    coord.y ++;\n\
\n\
    VXC_DP2x8(src2, src2, src2, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI8toI8_0_part0_2x8);\n\
    VXC_DP2x8(src2, src2, src2, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniConvertI8toI8_0_part1_2x8);\n\
    VXC_DP2x8(src3, src3, src3, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI8toI8_1_part0_2x8);\n\
    VXC_DP2x8(src3, src3, src3, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniConvertI8toI8_1_part1_2x8);\n\
    dst = min(src2, src3);\n\
\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 15, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcTensorMinimum_I8toI8_2D\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(1), get_global_id(1));\n\
\n\
    vxc_char16 src0, src1, src2, src3, dst;\n\
    VXC_ReadImage(src0, input0, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(src1, input1, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(src2, input0, coord.xy, VXC_5BITOFFSET_XY(0, 1),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(src3, input1, coord.xy, VXC_5BITOFFSET_XY(0, 1),\\\n\
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
\n\
    VXC_DP2x8(src2, src2, src2, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI8toI8_0_part0_2x8);\n\
    VXC_DP2x8(src2, src2, src2, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniConvertI8toI8_0_part1_2x8);\n\
    VXC_DP2x8(src3, src3, src3, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI8toI8_1_part0_2x8);\n\
    VXC_DP2x8(src3, src3, src3, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniConvertI8toI8_1_part1_2x8);\n\
    dst = min(src2, src3);\n\
\n\
    VXC_WriteImage(output, coord.xz, dst, VXC_MODIFIER(0, 15, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform VXC_512Bits uniU8MulAndPostShift_0_Hi_2x8;\n\
_viv_uniform VXC_512Bits uniU8MulAndPostShift_0_Lo_2x8;\n\
_viv_uniform VXC_512Bits uniU8MulAndPostShift_1_Hi_2x8;\n\
_viv_uniform VXC_512Bits uniU8MulAndPostShift_1_Lo_2x8;\n\
_viv_uniform int2 multAndoutZP0;//[0:15] multiplier, [31:63] output zp\n\
_viv_uniform int2 multAndoutZP1;//[0:15] multiplier, [31:63] output zp\n\
__kernel void vxcTensorMinimum_U8toU8\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_uchar16 src0, src1, src2, src3, dst;\n\
    VXC_ReadImage2DArray(src0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src2, input0, coord, VXC_5BITOFFSET_XY(0, 1),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src3, input1, coord, VXC_5BITOFFSET_XY(0, 1),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    vxc_ushort8 ms0, ms1;\n\
    _viv_asm(COPY, ms0, multAndoutZP0, 16);\n\
    _viv_asm(COPY, ms1, multAndoutZP1, 16);\n\
    VXC_DP2x8(src0, src0, ms0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniU8MulAndPostShift_0_Lo_2x8);\n\
    VXC_DP2x8(src0, src0, ms0, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniU8MulAndPostShift_0_Hi_2x8);\n\
    VXC_DP2x8(src1, src1, ms1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniU8MulAndPostShift_1_Lo_2x8);\n\
    VXC_DP2x8(src1, src1, ms1, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniU8MulAndPostShift_1_Hi_2x8);\n\
    dst = min(src0, src1);\n\
\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 15, 0,VXC_RM_TowardZero, 0));\n\
    coord.y ++;\n\
\n\
    VXC_DP2x8(src2, src2, ms0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniU8MulAndPostShift_0_Lo_2x8);\n\
    VXC_DP2x8(src2, src2, ms0, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniU8MulAndPostShift_0_Hi_2x8);\n\
    VXC_DP2x8(src3, src3, ms1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniU8MulAndPostShift_1_Lo_2x8);\n\
    VXC_DP2x8(src3, src3, ms1, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniU8MulAndPostShift_1_Hi_2x8);\n\
    dst = min(src2, src3);\n\
\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 15, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcTensorMinimum_U8toU8_2D\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(1), get_global_id(1));\n\
\n\
    vxc_uchar16 src0, src1, src2, src3, dst;\n\
    VXC_ReadImage(src0, input0, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(src1, input1, coord.xy, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(src2, input0, coord.xy, VXC_5BITOFFSET_XY(0, 1),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(src3, input1, coord.xy, VXC_5BITOFFSET_XY(0, 1),\\\n\
        VXC_MODIFIER(0, 15, 0, VXC_RM_TowardZero, 0));\n\
\n\
    coord.z ++;\n\
\n\
    vxc_ushort8 ms0, ms1;\n\
    _viv_asm(COPY, ms0, multAndoutZP0, 16);\n\
    _viv_asm(COPY, ms1, multAndoutZP1, 16);\n\
    VXC_DP2x8(src0, src0, ms0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniU8MulAndPostShift_0_Lo_2x8);\n\
    VXC_DP2x8(src0, src0, ms0, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniU8MulAndPostShift_0_Hi_2x8);\n\
    VXC_DP2x8(src1, src1, ms1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniU8MulAndPostShift_1_Lo_2x8);\n\
    VXC_DP2x8(src1, src1, ms1, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniU8MulAndPostShift_1_Hi_2x8);\n\
    dst = min(src0, src1);\n\
\n\
    VXC_WriteImage(output, coord.xy, dst, VXC_MODIFIER(0, 15, 0,VXC_RM_TowardZero, 0));\n\
\n\
    VXC_DP2x8(src2, src2, ms0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniU8MulAndPostShift_0_Lo_2x8);\n\
    VXC_DP2x8(src2, src2, ms0, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniU8MulAndPostShift_0_Hi_2x8);\n\
    VXC_DP2x8(src3, src3, ms1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniU8MulAndPostShift_1_Lo_2x8);\n\
    VXC_DP2x8(src3, src3, ms1, VXC_MODIFIER(8, 15, 0, VXC_RM_ToNearestEven, 1), uniU8MulAndPostShift_1_Hi_2x8);\n\
    dst = min(src2, src3);\n\
\n\
    VXC_WriteImage(output, coord.xz, dst, VXC_MODIFIER(0, 15, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
_viv_uniform VXC_512Bits uniConvertI16toI16_0_2x8;\n\
_viv_uniform VXC_512Bits uniConvertI16toI16_1_2x8;\n\
__kernel void vxcTensorMinimum_I16toI16\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);\n\
\n\
    vxc_short8 src0, src1, src2, src3, dst;\n\
    VXC_ReadImage2DArray(src0, input0, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src1, input1, coord, VXC_5BITOFFSET_XY(0, 0),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src2, input0, coord, VXC_5BITOFFSET_XY(0, 1),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage2DArray(src3, input1, coord, VXC_5BITOFFSET_XY(0, 1),\\\n\
        VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    VXC_DP2x8(src0, src0, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI16toI16_0_2x8);\n\
    VXC_DP2x8(src1, src1, src1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI16toI16_1_2x8);\n\
    dst = min(src0, src1);\n\
\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
    coord.y ++;\n\
\n\
    VXC_DP2x8(src2, src2, src2, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI16toI16_0_2x8);\n\
    VXC_DP2x8(src3, src3, src3, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI16toI16_1_2x8);\n\
    dst = min(src2, src3);\n\
\n\
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
\n\
__kernel void vxcTensorMinimum_I16toI16_2D\n\
    (\n\
    __read_only  image2d_array_t    input0,\n\
    __read_only  image2d_array_t    input1,\n\
    __write_only image2d_array_t    output\n\
    )\n\
{\n\
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(1), get_global_id(1));\n\
\n\
    vxc_short8 src0, src1, src2, src3, dst;\n\
    VXC_ReadImage(src0, input0, coord.xy, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(src1, input1, coord.xy, VXC_5BITOFFSET_XY(0, 0), VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(src2, input0, coord.xy, VXC_5BITOFFSET_XY(0, 1), VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
    VXC_ReadImage(src3, input1, coord.xy, VXC_5BITOFFSET_XY(0, 1), VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0));\n\
\n\
    coord.z ++;\n\
\n\
    VXC_DP2x8(src0, src0, src0, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI16toI16_0_2x8);\n\
    VXC_DP2x8(src1, src1, src1, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI16toI16_1_2x8);\n\
    dst = min(src0, src1);\n\
\n\
    VXC_WriteImage(output, coord.xy, dst, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
\n\
    VXC_DP2x8(src2, src2, src2, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI16toI16_0_2x8);\n\
    VXC_DP2x8(src3, src3, src3, VXC_MODIFIER(0, 7, 0, VXC_RM_ToNearestEven, 1), uniConvertI16toI16_1_2x8);\n\
    dst = min(src2, src3);\n\
\n\
    VXC_WriteImage(output, coord.xz, dst, VXC_MODIFIER(0, 7, 0,VXC_RM_TowardZero, 0));\n\
}\n\
"; /* end of vsi_nn_kernel_minimum_vx*/

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
    vxc_short8 din0, din1;\n\
    vxc_half8 din0Fp16, din1Fp16;\n\
    vxc_half8 maxDataVer, maxDataVer1;\n\
    int4 bitExtractCoeff;\n\
    vxc_half8 din0EqualTmp, din1EqualTmp;\n\
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
    VXC_VertMax3_Half(maxDataVer, din0Fp16, din1Fp16, din1Fp16, VXC_MODIFIER_BIN(0, 7, 0));\n\
    _viv_asm(COPY, din0, maxDataVer, 16);\n\
    din1 = din0.s10325476;\n\
    _viv_asm(COPY, maxDataVer1, din1, 16);\n\
    VXC_VertMax3_Half(maxDataVer, maxDataVer1, maxDataVer, maxDataVer, VXC_MODIFIER_BIN(0, 7, 0));\n\
    _viv_asm(COPY, din0, maxDataVer, 16);\n\
    din1 = din0.s02460246;//output\n\
    //get axis\n\
    VXC_Clamp_Half(din0EqualTmp, din0Fp16, maxDataVer, maxDataVer, VXC_MODIFIER_CLAMP(0, 7, 0, 1));\n\
    VXC_Clamp_Half(din1EqualTmp, din1Fp16, maxDataVer, maxDataVer, VXC_MODIFIER_CLAMP(0, 7, 0, 1));\n\
    bitExtractCoeff = (int4)(0x30201000, 0x70605040, 0x01010101, 0x01010101);\n\
    VXC_BitExtract(din0Equal, din0EqualTmp, din0EqualTmp, bitExtractCoeff, VXC_MODIFIER_BIN(0, 7, 0));\n\
    VXC_BitExtract(din1Equal, din1EqualTmp, din1EqualTmp, bitExtractCoeff, VXC_MODIFIER_BIN(0, 7, 0));\n\
    VXC_DP4x4(axisEncode, din0Equal, din1Equal, VXC_MODIFIER_BIN(0, 3, 0), poolingEncode);\n\
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
    VXC_DP4x4(x0, src0, src0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertFstFp16Fp32_4x4);\n\
    VXC_DP4x4(x1, src0, src0, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertSecFp16Fp32_4x4);\n\
    VXC_DP4x4(y0, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertFstFp16Fp32_4x4);\n\
    VXC_DP4x4(y1, src1, src1, VXC_MODIFIER(0, 3, 0, VXC_RM_TowardZero, 0), uniConvertSecFp16Fp32_4x4);\n\
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
"; /* end of vsi_nn_kernel_pow_vx*/

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
#if 1\n\
TENSORRELATION(Gt,    Uint8,  vxc_uchar8,  vxc_uchar8,  vxc_char8,  vxc_uchar8,  >)\n\
TENSORRELATION(Gte,   Uint8,  vxc_uchar8, vxc_uchar8,  vxc_char8,  vxc_uchar8,  >=)\n\
TENSORRELATION(Ls,    Uint8,  vxc_uchar8, vxc_uchar8,  vxc_char8,  vxc_uchar8,  <)\n\
TENSORRELATION(Lse,   Uint8,  vxc_uchar8, vxc_uchar8,  vxc_char8,  vxc_uchar8,  <=)\n\
TENSORRELATION(Ne,    Uint8,  vxc_uchar8, vxc_uchar8,  vxc_char8,  vxc_uchar8,  !=)\n\
#endif\n\
TENSORRELATION(Gt,    Int16,  vxc_short8, vxc_short8,  vxc_short8,  vxc_short8,  >)\n\
TENSORRELATION(Gte,   Int16,  vxc_short8, vxc_short8,  vxc_short8,  vxc_short8,  >=)\n\
TENSORRELATION(Ls,    Int16,  vxc_short8, vxc_short8,  vxc_short8,  vxc_short8,  <)\n\
TENSORRELATION(Lse,   Int16,  vxc_short8, vxc_short8,  vxc_short8,  vxc_short8,  <=)\n\
TENSORRELATION(Ne,    Int16,  vxc_short8, vxc_short8,  vxc_short8,  vxc_short8,  !=)\n\
\n\
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
    VXC_WriteImage2DArray(output, coord, dst, VXC_MODIFIER(0, 7, 0, VXC_RM_TowardZero, 0)); \\\n\
}\n\
\n\
//          name0, name1, input_type, copy_type, output_type, out_copy_type, cmp_op\n\
TENSORCMPHALF(Gt,    Fp16,  >)\n\
TENSORCMPHALF(Gte,   Fp16,  >=)\n\
TENSORCMPHALF(Ls,    Fp16,  <)\n\
TENSORCMPHALF(Lse,   Fp16,  <=)\n\
TENSORCMPHALF(Ne,    Fp16,  !=)"; /* end of vsi_nn_kernel_relational_ops_vx*/

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
    {"vsi_nn_kernel_argmax_vx", vsi_nn_kernel_argmax_vx},
    {"vsi_nn_kernel_crop_vx", vsi_nn_kernel_crop_vx},
    {"vsi_nn_kernel_dropout_vx", vsi_nn_kernel_dropout_vx},
    {"vsi_nn_kernel_eltwisemax_vx", vsi_nn_kernel_eltwisemax_vx},
    {"vsi_nn_kernel_elu_vx", vsi_nn_kernel_elu_vx},
    {"vsi_nn_kernel_floordiv_vx", vsi_nn_kernel_floordiv_vx},
    {"vsi_nn_kernel_fullconnect2_vx", vsi_nn_kernel_fullconnect2_vx},
    {"vsi_nn_kernel_header_vx", vsi_nn_kernel_header_vx},
    {"vsi_nn_kernel_imageprocess_vx", vsi_nn_kernel_imageprocess_vx},
    {"vsi_nn_kernel_imageprocess_2_vx", vsi_nn_kernel_imageprocess_2_vx},
    {"vsi_nn_kernel_imageprocess_3_vx", vsi_nn_kernel_imageprocess_3_vx},
    {"vsi_nn_kernel_instancenormalize_vx", vsi_nn_kernel_instancenormalize_vx},
    {"vsi_nn_kernel_instancenormalize_i8_vx", vsi_nn_kernel_instancenormalize_i8_vx},
    {"vsi_nn_kernel_l2normalizescale_vx", vsi_nn_kernel_l2normalizescale_vx},
    {"vsi_nn_kernel_layernormalize_vx", vsi_nn_kernel_layernormalize_vx},
    {"vsi_nn_kernel_logical_ops_vx", vsi_nn_kernel_logical_ops_vx},
    {"vsi_nn_kernel_matrixmul_vx", vsi_nn_kernel_matrixmul_vx},
    {"vsi_nn_kernel_matrixmul_fp16_vx", vsi_nn_kernel_matrixmul_fp16_vx},
    {"vsi_nn_kernel_matrixmul_transbp1_vx", vsi_nn_kernel_matrixmul_transbp1_vx},
    {"vsi_nn_kernel_matrixmul_transbp2_vx", vsi_nn_kernel_matrixmul_transbp2_vx},
    {"vsi_nn_kernel_minimum_vx", vsi_nn_kernel_minimum_vx},
    {"vsi_nn_kernel_poolwithargmax_vx", vsi_nn_kernel_poolwithargmax_vx},
    {"vsi_nn_kernel_poolwithargmax_i16_vx", vsi_nn_kernel_poolwithargmax_i16_vx},
    {"vsi_nn_kernel_poolwithargmax_i8_vx", vsi_nn_kernel_poolwithargmax_i8_vx},
    {"vsi_nn_kernel_poolwithargmax_opt_vx", vsi_nn_kernel_poolwithargmax_opt_vx},
    {"vsi_nn_kernel_poolwithargmax_u8_vx", vsi_nn_kernel_poolwithargmax_u8_vx},
    {"vsi_nn_kernel_pow_vx", vsi_nn_kernel_pow_vx},
    {"vsi_nn_kernel_prelu_vx", vsi_nn_kernel_prelu_vx},
    {"vsi_nn_kernel_prelu_i8_i16_vx", vsi_nn_kernel_prelu_i8_i16_vx},
    {"vsi_nn_kernel_prelu_u8_vx", vsi_nn_kernel_prelu_u8_vx},
    {"vsi_nn_kernel_relational_ops_vx", vsi_nn_kernel_relational_ops_vx},
    {"vsi_nn_kernel_resize_vx", vsi_nn_kernel_resize_vx},
    {"vsi_nn_kernel_reverse_vx", vsi_nn_kernel_reverse_vx},
    {"vsi_nn_kernel_scale_vx", vsi_nn_kernel_scale_vx},
    {"vsi_nn_kernel_select_vx", vsi_nn_kernel_select_vx},
    {"vsi_nn_kernel_shufflechannel_vx", vsi_nn_kernel_shufflechannel_vx},
    {"vsi_nn_kernel_signalframe_vx", vsi_nn_kernel_signalframe_vx},
    {"vsi_nn_kernel_space2depth_vx", vsi_nn_kernel_space2depth_vx},
    {"vsi_nn_kernel_tensorstackconcat_vx", vsi_nn_kernel_tensorstackconcat_vx},
    {"vsi_nn_kernel_transform_gemm_vx", vsi_nn_kernel_transform_gemm_vx},
    {"vsi_nn_kernel_transform_interp_vx", vsi_nn_kernel_transform_interp_vx},
    {"vsi_nn_kernel_transform_setupThres_vx", vsi_nn_kernel_transform_setupThres_vx},
    {"vsi_nn_kernel_upsample_vx", vsi_nn_kernel_upsample_vx},
    {"vsi_nn_kernel_upsample_2_vx", vsi_nn_kernel_upsample_2_vx},
    {"vsi_nn_kernel_upsample_i8_vx", vsi_nn_kernel_upsample_i8_vx},
    {"vsi_nn_kernel_upsample_opt_vx", vsi_nn_kernel_upsample_opt_vx},
    {"vsi_nn_kernel_upsample_u8_vx", vsi_nn_kernel_upsample_u8_vx},
};

const int vx_resource_items_cnt = _cnt_of_array(vx_resource_items);
