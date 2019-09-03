#ifndef __OVXLIB_MEMORY_H__
#define __OVXLIB_MEMORY_H__

#include <stdint.h>
#ifdef __linux__
#include <unistd.h>
#include <sys/mman.h>
#endif

namespace ovxlib
{
class Memory
{
    public:
        Memory(){};
        virtual ~Memory() {
#ifdef __linux__
            //if (dup_fd_ >= 0)
            //{
            //    close(dup_fd_);
            //}
            if (nullptr != data_ptr_)
            {
                munmap(data_ptr_, data_length_);
            }
#endif
        }

        virtual int readFromFd(size_t size,
                int protect, int fd, size_t offset);

        virtual void* data(size_t offset) const {
            if (nullptr == data_ptr_ || offset >= data_length_)
            {
                return nullptr;
            }
            uint8_t* ptr = static_cast<uint8_t*>(data_ptr_);
            return static_cast<void*>(&ptr[offset]);
        }

        virtual size_t length() const {return data_length_;}

    private:
        int dup_fd_{-1};
        void * data_ptr_{nullptr};
        size_t data_length_{0};
};
}

#endif
