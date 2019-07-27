/****************************************************************************
*
*    Copyright 2012 - 2019 Vivante Corporation, Santa Clara, California.
*    All Rights Reserved.
*
*    Permission is hereby granted, free of charge, to any person obtaining
*    a copy of this software and associated documentation files (the
*    'Software'), to deal in the Software without restriction, including
*    without limitation the rights to use, copy, modify, merge, publish,
*    distribute, sub license, and/or sell copies of the Software, and to
*    permit persons to whom the Software is furnished to do so, subject
*    to the following conditions:
*
*    The above copyright notice and this permission notice (including the
*    next paragraph) shall be included in all copies or substantial
*    portions of the Software.
*
*    THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
*    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
*    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
*    IN NO EVENT SHALL VIVANTE AND/OR ITS SUPPLIERS BE LIABLE FOR ANY
*    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
*    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
*    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/


#ifndef _VSI_NN_OP_PRE_PROCESS_RGB_H
#define _VSI_NN_OP_PRE_PROCESS_RGB_H

#include "vsi_nn_types.h"


enum
{
    PRE_PROCESS_RGB_INPUT = 0,

    PRE_PROCESS_RGB_INPUT_CNT,

    PRE_PROCESS_RGB_OUTPUT = 0,

    PRE_PROCESS_RGB_OUTPUT_CNT
};


enum {
    PRE_PROCESS_RGB_CPU_KERNEL,

    PRE_PROCESS_RGB_F16_KERNEL,
    PRE_PROCESS_RGB_I16_KERNEL,
    PRE_PROCESS_RGB_I8_KERNEL,
    PRE_PROCESS_RGB_U8_KERNEL,

    PRE_PROCESS_RGB_F16_COPY_KERNEL,
    PRE_PROCESS_RGB_I16_COPY_KERNEL,
    PRE_PROCESS_RGB_I8_COPY_KERNEL,
    PRE_PROCESS_RGB_U8_COPY_KERNEL,

    PRE_PROCESS_RGB_F16_NHWC_KERNEL,
    PRE_PROCESS_RGB_I16_NHWC_KERNEL,
    PRE_PROCESS_RGB_I8_NHWC_KERNEL,
    PRE_PROCESS_RGB_U8_NHWC_KERNEL,

    PRE_PROCESS_RGB_F16_COPY_NHWC_KERNEL,
    PRE_PROCESS_RGB_I16_COPY_NHWC_KERNEL,
    PRE_PROCESS_RGB_I8_COPY_NHWC_KERNEL,
    PRE_PROCESS_RGB_U8_COPY_NHWC_KERNEL,

    PRE_PROCESS_RGB_KERNEL_COUNTS,
};

typedef struct _vsi_nn_pre_process_rgb_lcl_data
{
    int32_t scale_x;
    int32_t scale_y;
    vsi_bool enable_copy;
    vsi_bool enable_perm;
    vx_tensor  local_tensor;
} vsi_nn_pre_process_rgb_lcl_data;

typedef struct _vsi_nn_pre_process_rgb_param
{
    struct
    {
        uint32_t left;
        uint32_t top;
        uint32_t width;
        uint32_t height;
    } rect;

    struct
    {
        uint32_t   *size;
        uint32_t   dim_num;
    } output_attr;

    uint32_t * perm;
    uint32_t   dim_num;

    float r_mean;
    float g_mean;
    float b_mean;
    float rgb_scale;

    vsi_bool reverse_channel;

    /* pre process rgb layer local data structure */
    vsi_nn_pre_process_rgb_lcl_data local;
} vsi_nn_pre_process_rgb_param;

#endif

