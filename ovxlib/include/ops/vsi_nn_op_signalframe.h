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


#ifndef _VSI_NN_OP_CLIENT_SIGNALFRAME_H
#define _VSI_NN_OP_CLIENT_SIGNALFRAME_H

#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

#define _VSI_NN_SIGNALFRAME_LOCAL_TENSOR_NUM 7

typedef struct _vsi_nn_signalframe_lcl_data
{
    //vsi_nn_tensor_t *signal_tensor;
    //vsi_nn_tensor_t *frame_tensor;
    vsi_nn_tensor_t *window_length_tensor;
    vsi_nn_tensor_t *step_tensor;
    vsi_nn_tensor_t *pad_end_tensor;
    vsi_nn_tensor_t *pad_tensor;
    vsi_nn_tensor_t *axis_tensor;
    vx_tensor   local_tensor[_VSI_NN_SIGNALFRAME_LOCAL_TENSOR_NUM];
} vsi_nn_signalframe_lcl_data;

typedef struct _vsi_nn_signalframe_param
{
    /* local data must be the first. */
    vsi_nn_signalframe_lcl_data local;

    uint32_t window_length;
    uint32_t step;
    uint32_t pad_end;
    uint32_t pad;
    uint32_t axis;
} vsi_nn_signalframe_param;

#ifdef __cplusplus
}
#endif

#endif
