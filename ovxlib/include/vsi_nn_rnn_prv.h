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


#ifndef _VSI_NN_RNN_PRV_H_
#define _VSI_NN_RNN_PRV_H_
#include "vsi_nn_types.h"
#include "vsi_nn_tensor.h"
#include "utils/vsi_nn_link_list.h"
#include "vsi_nn_rnn.h"

#if defined(__cplusplus)
extern "C"{
#endif

typedef struct
{
    vsi_nn_tensor_attr_t    attr;
    uint8_t*                data;
    vsi_nn_size_t           data_size; /* in bytes */
} vsi_nn_rnn_internal_buffer_t;

typedef struct
{
    vsi_nn_link_list_t link_list;
    vsi_nn_rnn_external_connection_t connection;
    vsi_nn_rnn_internal_buffer_t buffer;
} vsi_nn_rnn_connection_t;

typedef struct
{
    vsi_nn_rnn_connection_t* external_connection_list;
    void* user_data;
} vsi_nn_rnn_wksp_t;

#if defined(__cplusplus)
}
#endif

#endif
