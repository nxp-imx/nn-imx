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


#ifndef _VSI_NN_OP_LSTMUNIT_H
#define _VSI_NN_OP_LSTMUNIT_H

#include "vsi_nn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum _vsi_nn_lstmunit_activation_e
{
    VSI_NN_LSTMUNIT_ACT_NONE    = 0,
    VSI_NN_LSTMUNIT_ACT_RELU    = 1,
    VSI_NN_LSTMUNIT_ACT_RELU6   = 3,
    VSI_NN_LSTMUNIT_ACT_TANH    = 4,
    VSI_NN_LSTMUNIT_ACT_SIGMOID = 6,

    VSI_NN_LSTMUNIT_ACT_HARD_SIGMOID = 31 /* temporary use 31*/
} vsi_nn_lstmunit_activation_e;

typedef struct _vsi_nn_lstmunit_lcl_data_t
{
    vsi_nn_tensor_t *activation_tensor;
    vsi_nn_tensor_t *cell_clip_tensor;
    vsi_nn_tensor_t *proj_clip_tensor;
    vsi_nn_tensor_t *scratch_tensor;

    vsi_nn_tensor_attr_t scratch_attr;
    vsi_nn_tensor_t *forget_bias_tensor;
} vsi_nn_lstmunit_lcl_data_t;

typedef struct _vsi_nn_lstmunit_param_t
{
    vsi_nn_lstmunit_lcl_data_t local;

    float cell_clip;
    float proj_clip;
    vsi_nn_lstmunit_activation_e activation;
    float forget_bias;
} vsi_nn_lstmunit_param;

#ifdef __cplusplus
}
#endif

#endif
