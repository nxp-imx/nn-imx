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
#include <cassert>

#ifdef __linux__
#include <unistd.h>
#include <sys/mman.h>
#endif

#include "vsi_nn_pub.h"
#include "file_map_memory.hpp"
#include "error.hpp"

namespace nnrt
{
int Memory::readFromFd(size_t size, int protect, int fd, size_t offset)
{
    if (fd < 0)
    {
        VSILOGW("Invalid file descriptor.");
        return NNA_ERROR_CODE(UNEXPECTED_NULL);
    }
    if (size <= 0) {
        VSILOGW("Invalid size.");
        return NNA_ERROR_CODE(BAD_DATA);
    }
#ifdef __linux__
    // No need to dup fd??
    //dup_fd_ = dup(fd);
    //if (dup_fd_ == -1) {
    //    VSILOGW("Failed to dup the fd\n");
    //    return NNA_ERROR_CODE(UNEXPECTED_NULL);
    //}

    data_ptr_ = mmap(nullptr, size, protect, MAP_SHARED, fd, offset);
    if (data_ptr_ == MAP_FAILED) {
        VSILOGW("Can't mmap the file descriptor.\n");
        return NNA_ERROR_CODE(UNMAPPABLE);
    }
    data_length_ = size;
#else
    assert(false);
    return NNA_ERROR_CODE(UNEXPECTED_NULL);
#endif
    return NNA_ERROR_CODE(NO_ERROR);
}
}
