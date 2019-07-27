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


#ifndef __VSI_NN_TENSOR_OP_H__
#define __VSI_NN_TENSOR_OP_H__

#include "vsi_nn_graph.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_types.h"

vsi_nn_tensor_t* vsi_nn_Concat
    (
    vsi_nn_graph_t* graph,
    vsi_nn_tensor_t** tensors,
    uint32_t tensor_num,
    uint32_t axis
    );

vsi_nn_tensor_t* vsi_nn_ConvertTensorDtype
    (
    vsi_nn_graph_t* graph,
    vsi_nn_tensor_t* tensor,
    const vsi_nn_dtype_t* dst_dtype
    );

#endif
