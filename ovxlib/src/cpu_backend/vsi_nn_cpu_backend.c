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
#include <stdlib.h>
#include <string.h>
#include "vsi_nn_types.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_client_op.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "cpu_backend/vsi_nn_cpu_backend.h"
#include "cpu_backend/npuref_interface.h"


#define DEF_OP( name )    extern vsi_nn_op_proc_t vsi_nn_op_CPU_BACKEND_##name;
DEF_OP( CONV2D )
DEF_OP( DECONV2D )
#undef DEF_OP

static vsi_nn_op_proc_t * s_client_ops[] =
    {
    #define DEF_OP( name )     &vsi_nn_op_CPU_BACKEND_##name,
    DEF_OP( CONV2D )
    DEF_OP( DECONV2D )
    #undef DEF_OP
    };

static vsi_nn_op_t s_client_ops_id[] =
    {
    #define DEF_OP( name )     VSI_NN_OP_##name,
    DEF_OP( CONV2D )
    DEF_OP( DECONVOLUTION ) // Use DECONVOLUTION because we need to replace it.
    #undef DEF_OP
    };

vsi_bool vsi_nn_RegisterCpuBackendPos
    (
    vsi_nn_graph_t * graph
    )
{
    uint32_t i;
    vsi_bool ret;
    ret = TRUE;
    for( i = 0; i < _cnt_of_array( s_client_ops ); i++ )
    {
        ret = vsi_nn_OpRegisterClient( s_client_ops_id[i],
                s_client_ops[i] );
        if( FALSE == ret )
        {
            break;
        }
    }
    return ret;
} /* vsi_nn_RegisterCpuBackendPos */

vsi_bool vsi_nn_UnregisterCpuBackendPos
    (
    vsi_nn_graph_t * graph
    )
{
    uint32_t i;
    vsi_bool ret;

    ret = TRUE;
    for( i = 0; i < _cnt_of_array( s_client_ops ); i++ )
    {
        vsi_nn_OpRemoveClient( s_client_ops_id[i]);
    }
    return ret;
} /* vsi_nn_UnregisterCpuBackendPos() */

vsi_bool vsi_nn_CpuBackendEnabled()
{
    vsi_bool ret = FALSE;
    char* str = getenv("VSI_NN_ENABLE_CPU_BACKEND");
    if( str )
    {
        ret = (vsi_bool)atoi( str );
    }
    if( ret )
    {
        ret = npuref_exists();
    }
    return ret;
} /* vsi_nn_CpuBackendEnabled() */

