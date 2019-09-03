#include <assert.h>

#ifdef __linux__
#include <unistd.h>
#include <sys/mman.h>
#endif

#include "vsi_nn_pub.h"
#include "file_map_memory.h"
#include "error.h"

namespace ovxlib
{
int Memory::readFromFd(size_t size, int protect, int fd, size_t offset)
{
    if (fd < 0)
    {
        VSILOGW("Invalid file descriptor.");
        return AERROR_CODE(UNEXPECTED_NULL);
    }
    if (size <= 0) {
        VSILOGW("Invalid size.");
        return AERROR_CODE(BAD_DATA);
    }
#ifdef __linux__
    // No need to dup fd??
    //dup_fd_ = dup(fd);
    //if (dup_fd_ == -1) {
    //    VSILOGW("Failed to dup the fd\n");
    //    return AERROR_CODE(UNEXPECTED_NULL);
    //}

    data_ptr_ = mmap(nullptr, size, protect, MAP_SHARED, fd, offset);
    if (data_ptr_ == MAP_FAILED) {
        VSILOGW("Can't mmap the file descriptor.\n");
        return AERROR_CODE(UNMAPPABLE);
    }
    data_length_ = size;
#else
    assert(false);
    return AERROR_CODE(UNEXPECTED_NULL);
#endif
    return AERROR_CODE(NO_ERROR);
}
}
