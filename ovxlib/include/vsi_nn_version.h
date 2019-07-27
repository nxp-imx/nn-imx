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


#ifndef _VSI_NN_VERSION_H_
#define _VSI_NN_VERSION_H_

#if defined(__cplusplus)
extern "C"{
#endif

#define VSI_NN_VERSION_MAJOR 1
#define VSI_NN_VERSION_MINOR 1
#define VSI_NN_VERSION_PATCH 7
#define VSI_NN_VERSION \
    (VSI_NN_VERSION_MAJOR * 10000 + VSI_NN_VERSION_MINOR * 100 + VSI_NN_VERSION_PATCH)

#define _version_assert _compiler_assert

OVXLIB_API const char *vsi_nn_GetVersion(void);

#if defined(__cplusplus)
}
#endif
#endif
