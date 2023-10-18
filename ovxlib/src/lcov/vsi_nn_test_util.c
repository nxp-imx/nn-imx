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
#include <string.h>
#include "vsi_nn_test.h"
#include "lcov/vsi_nn_test_util.h"

static vsi_status vsi_nn_test_strncat( void )
{
#define BUF_SIZE (64)
    char dst[BUF_SIZE] = "test ";
    char golden[BUF_SIZE] = "test strncat";

    vsi_nn_strncat(dst, "strncat", BUF_SIZE - 1);
    if( 0 == strncmp(dst, golden, sizeof(golden)) )
    {
        return VSI_SUCCESS;
    }

    return VSI_FAILURE;
}

vsi_status vsi_nn_test_util( void )
{
    vsi_status status = VSI_FAILURE;

    status = vsi_nn_test_strncat();
    TEST_CHECK_STATUS(status, final);

final:
    return status;
}