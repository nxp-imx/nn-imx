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

#include "vsi_nn_daemon.h"
#include "vsi_nn_log.h"
#include "kernel/vsi_nn_kernel.h"
#include "cpu_backend/vsi_nn_cpu_backend.h"
#include "cpu_backend/npuref_interface.h"

_INITIALIZER( daemon_start )
{
    //VSILOGD("OVXLIB init ... ");
    vsi_nn_kernel_backend_init();

    if( vsi_nn_CpuBackendEnabled() )
    {
        //npuref_init();
    }
} /* _daemon_start() */

_DEINITIALIZER( daemon_shutdown )
{
    //VSILOGD("OVXLIB shutdown ... ");
    vsi_nn_kernel_backend_deinit();
    npuref_shutdown();
} /* vsi_nn_daemen_shutdown() */

