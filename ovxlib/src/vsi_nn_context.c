/****************************************************************************
*
*    Copyright (c) 2018 Vivante Corporation
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
#include "vsi_nn_context.h"

vsi_nn_context_t vsi_nn_CreateContext
    ( void )
{
    vsi_nn_context_t context = NULL;
    vx_context c = NULL;

    context = (vsi_nn_context_t)malloc(sizeof(struct _vsi_nn_context_t));
    if(NULL == context)
    {
        return NULL;
    }
    c = vxCreateContext();
    if(NULL == c)
    {
        free(context);
        return NULL;
    }

    memset(context, 0, sizeof(struct _vsi_nn_context_t));
    context->c = c;
    return context;
} /* vsi_nn_CreateContext() */

void vsi_nn_ReleaseContext
    ( vsi_nn_context_t * ctx )
{
    if( NULL != ctx && NULL != *ctx )
    {
        vsi_nn_context_t context = *ctx;
        if(context->c)
        {
            vxReleaseContext( &context->c);
        }
        free(context);
        *ctx = NULL;
    }
} /* vsi_nn_ReleaseContext() */

