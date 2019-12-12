/****************************************************************************
*
*    Copyright (c) 2019 Vivante Corporation
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
#include "vsi_nn_prv.h"
#include "vsi_nn_types.h"
#include "vsi_nn_log.h"
#include "vsi_nn_error.h"
#include "vsi_nn_ops.h"
#include "kernel/vsi_nn_kernel.h"
#include "utils/vsi_nn_hashmap.h"


static vsi_bool s_backends_inited = FALSE;
static vsi_nn_hashmap_t* s_backends = NULL;
static vsi_nn_kernel_unique_id_t s_global_id = 0;

void vsi_nn_kernel_backend_register
    (
    const char* kernel_name,
    vsi_nn_kernel_type_e kernel_type,
    vsi_nn_kernel_setup_func_t setup_func
    )
{
    vsi_nn_kernel_backend_t* backend;
    if( !s_backends )
    {
        s_backends = vsi_nn_hashmap_create();
    }
    if( vsi_nn_hashmap_has( s_backends, kernel_name ) )
    {
        backend = vsi_nn_hashmap_get( s_backends, kernel_name );
    }
    else
    {
        backend = (vsi_nn_kernel_backend_t*)malloc( sizeof(vsi_nn_kernel_backend_t) );
        if( !backend )
        {
            VSILOGE("Out of memory, register backend fail.");
            VSI_ASSERT( FALSE );
        }
        memset( backend, 0, sizeof(vsi_nn_kernel_backend_t) );
        vsi_nn_hashmap_add( s_backends, kernel_name, backend );
        backend->unique_id = s_global_id ++;
    }
    if( backend->setup[kernel_type] )
    {
        VSILOGE("Kernel %s backend %d has been registered!", kernel_name, kernel_type);
        VSI_ASSERT( FALSE );
    }
    backend->setup[kernel_type] = setup_func;
} /* vsi_nn_register_backend() */

const vsi_nn_kernel_backend_t* vsi_nn_kernel_backend_get( const char* key )
{
    const vsi_nn_kernel_backend_t* backend = NULL;
    backend = (const vsi_nn_kernel_backend_t*)vsi_nn_hashmap_get( s_backends, key );
    return backend;
} /* vsi_nn_backend_get() */

vsi_status vsi_nn_kernel_backend_init( void )
{
    vsi_status status = VSI_SUCCESS;
    // TODO: Multi-thread support
    if( s_backends_inited )
    {
        return status;
    }
    s_backends_inited = TRUE;
#if defined(__linux__)
#if 0
    extern vsi_nn_kernel_section_meta_t* __start_kernel_meta_section;
    extern vsi_nn_kernel_section_meta_t* __stop_kernel_meta_section;
    vsi_nn_kernel_section_meta_t** iter = &__start_kernel_meta_section;
    for( ; iter < &__stop_kernel_meta_section; iter ++  )
    {
        vsi_nn_kernel_backend_register((*iter)->name,
                (*iter)->kernel_type, (*iter)->func );
    }
#endif
#if 0
    REGISTER_KERNEL_BACKEND_MANUALLY( MINIMUM, CL, cl_minimum_setup );
    REGISTER_KERNEL_BACKEND_MANUALLY( MINIMUM, EVIS, evis_minimum_setup );
    REGISTER_KERNEL_BACKEND_MANUALLY( MINIMUM, CPU, cpu_minimum_setup );
    //REGISTER_KERNEL_BACKEND_MANUALLY( MINIMUM, VX, vx_minimum_setup );
#endif
#endif
    return status;
}

void vsi_nn_kernel_backend_deinit()
{
    vsi_nn_hashmap_release( &s_backends );
} /* vsi_nn_kernel_backend_deinit() */

