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


#ifndef _VSI_NN_OP_NEG_H
#define _VSI_NN_OP_NEG_H

#include "vsi_nn_types.h"

enum {
    TENSOR_NEG_CPU_KERNEL,

    TENSOR_NEG_F16TOF16_KERNEL,
    TENSOR_NEG_F16TOI16_KERNEL,
    TENSOR_NEG_F16TOI8_KERNEL,
    TENSOR_NEG_F16TOU8_KERNEL,
    TENSOR_NEG_I16TOI16_KERNEL,
    TENSOR_NEG_I16TOF16_KERNEL,
    TENSOR_NEG_I8TOI8_KERNEL,
    TENSOR_NEG_I8TOF16_KERNEL,
    TENSOR_NEG_U8TOU8_KERNEL,
    TENSOR_NEG_U8TOF16_KERNEL,

    TENSOR_NEG_F16TOF16_2D_KERNEL,
    TENSOR_NEG_F16TOI16_2D_KERNEL,
    TENSOR_NEG_F16TOI8_2D_KERNEL,
    TENSOR_NEG_F16TOU8_2D_KERNEL,
    TENSOR_NEG_I16TOI16_2D_KERNEL,
    TENSOR_NEG_I16TOF16_2D_KERNEL,
    TENSOR_NEG_I8TOI8_2D_KERNEL,
    TENSOR_NEG_I8TOF16_2D_KERNEL,
    TENSOR_NEG_U8TOU8_2D_KERNEL,
    TENSOR_NEG_U8TOF16_2D_KERNEL,

    TENSOR_NEG_KERNEL_COUNTS,
};

#define _VSI_NN_ELU_LOCAL_TENSOR_NUM 2

typedef struct _vsi_nn_neg_param
{
    vx_tensor   local_tensor[_VSI_NN_ELU_LOCAL_TENSOR_NUM];
} vsi_nn_neg_param;

#endif

