/****************************************************************************
*
*    Copyright (c) 2021 Vivante Corporation
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

#ifndef _VSI_NN_SP_LUT_H
#define _VSI_NN_SP_LUT_H

#if VX_STREAM_PROCESSOR_SUPPORT

#include <stdint.h>
#include "kernel/vsi_nn_spinst.h"

__BEGIN_DECLS

#define VSI_NN_SP_LUT_MAX_SIZE  (1024)

typedef struct _vsi_nn_sp_lut_
{
    float index;
    float val;
} vsi_nn_sp_lut_t;

typedef int32_t vsi_nn_sp_activation_e; enum
{
    VSI_NN_SP_ACT_NONE             = 0,
    VSI_NN_SP_ACT_LINEAR_EXP       = 1,
    VSI_NN_SP_ACT_LINEAR_RSQRT     = 2,
    VSI_NN_SP_ACT_LINEAR_SIGMOID   = 3,
    VSI_NN_SP_ACT_RCP              = 4,
};

typedef struct  _vsi_nn_sp_lut_params
{
    vsi_nn_sp_activation_e act_type;
    float params[16];
} vsi_nn_sp_lut_params;

vsi_status vsi_nn_sp_lut
    (
    vx_lut index_lut,
    vx_lut output_lut,
    vsi_nn_sp_lut_params *param
    );

__END_DECLS

#endif

#endif
