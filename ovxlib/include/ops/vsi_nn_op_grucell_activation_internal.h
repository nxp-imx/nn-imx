/****************************************************************************
*
*    Copyright (c) 2020 Vivante Corporation
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

#ifndef _VSI_NN_OP_GRUCELL_ACTIVATION_INTERNAL_H
#define _VSI_NN_OP_GRUCELL_ACTIVATION_INTERNAL_H

#include "vsi_nn_types.h"

enum {
    GRUCELL_ACTIVATION_INPUT_ZT_    = 0,
    GRUCELL_ACTIVATION_INPUT_HT__   = 1,
    GRUCELL_ACTIVATION_INPUT_HT_1   = 2,

    GRUCELL_ACTIVATION_INPUT_COUNT,

    GRUCELL_ACTIVATION_OUTPUT_OUTPUT    = 0,
    GRUCELL_ACTIVATION_OUTPUT_H_STATE   = 1,
    GRUCELL_ACTIVATION_OUTPUT_COUNT
};

typedef struct _vsi_nn_grucell_activation_internal_local {
    uint32_t placeholder;
} vsi_nn_grucell_activation_internal_local;

typedef struct _vsi_nn_grucell_activation_internal_param
{
    vsi_nn_grucell_activation_internal_local* local;
    vsi_nn_activation_e gate_activation;
    vsi_nn_activation_e candidate_activation;
} vsi_nn_grucell_activation_internal_param;

#endif

