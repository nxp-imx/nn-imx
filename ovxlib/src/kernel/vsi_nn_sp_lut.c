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

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include "vsi_nn_context.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_types.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_node.h"
#include "vsi_nn_log.h"
#include <float.h>
#include "vsi_nn_error.h"
#include "utils/vsi_nn_dtype_util_prv.h"
#include "vsi_nn_tensor_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "kernel/vsi_nn_spinst.h"
#include "kernel/vsi_nn_sp_lut.h"
#include "kernel/vsi_nn_kernel_lut.h"
#include "utils/vsi_nn_dtype_util.h"

#if VX_STREAM_PROCESSOR_SUPPORT

vsi_status vsi_nn_sp_lut
    (
    vx_lut index_lut,
    vx_lut output_lut,
    vsi_nn_sp_lut_params *param
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_kernel_lut_params lut_param;

    if (index_lut == NULL || output_lut == NULL || param == NULL)
    {
        return VSI_FAILURE;
    }

    memset(&lut_param, 0, sizeof(lut_param));

    switch (param->act_type)
    {
        case VSI_NN_SP_ACT_LINEAR_EXP:
        {
            lut_param.act_type = VSI_NN_KERNEL_LUT_LINEAR_EXP;
            lut_param.params[0] = param->params[0];
            lut_param.params[1] = param->params[1];
        }
        break;
        case VSI_NN_SP_ACT_LINEAR_RSQRT:
        {
            lut_param.act_type = VSI_NN_KERNEL_LUT_LINEAR_RSQRT;
            lut_param.pwl_sign_remove_support = param->pwl_sign_remove_support;
            lut_param.params[0] = param->params[0];
            lut_param.params[1] = param->params[1];
            lut_param.params[2] = param->params[2];
        }
        break;
        case VSI_NN_SP_ACT_LINEAR_SIGMOID:
        {
            lut_param.act_type = VSI_NN_KERNEL_LUT_LINEAR_SIGMOID;
            lut_param.params[0] = param->params[0];
            lut_param.params[1] = param->params[1];
        }
        break;
        case VSI_NN_SP_ACT_RCP:
        {
            lut_param.act_type = VSI_NN_KERNEL_LUT_RCP;
        }
        break;
    default:
        break;
    }

    status = vsi_nn_kernel_lut(index_lut, output_lut, &lut_param);

    return status;
}

#endif