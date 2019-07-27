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


#ifndef _VSI_NN_OP_PAD_H
#define _VSI_NN_OP_PAD_H

#include "vsi_nn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    VSI_NN_PAD_MODE_CONSTANT,
    VSI_NN_PAD_MODE_REPLICATE,
    VSI_NN_PAD_MODE_SYMMETRIC,
    VSI_NN_PAD_MODE_REFLECT,
}vsi_nn_pad_mode_e;

typedef struct _vsi_nn_pad_param
{
    uint32_t         * front_size;
    uint32_t         * back_size;
    uint8_t            dim_num;
    int32_t            const_val;
    vsi_nn_pad_mode_e  mode;
} vsi_nn_pad_param;

#ifdef __cplusplus
}
#endif

#endif
