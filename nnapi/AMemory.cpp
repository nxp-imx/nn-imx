/****************************************************************************
*
*    Copyright (c) 2005 - 2019 by Vivante Corp.  All rights reserved.
*
*    The material in this file is confidential and contains trade secrets
*    of Vivante Corporation. This is proprietary information owned by
*    Vivante Corporation. No part of this work may be disclosed,
*    reproduced, copied, transmitted, or used in any way for any purpose,
*    without the express written permission of Vivante Corporation.
*
*****************************************************************************/


#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <memory>
#include <vector>
#include <iostream>
#include <string>
#include <stdio.h>
#include <algorithm>
#include <VX/vx.h>
#include <VX/vxu.h>
#include <VX/vx_api.h>
#include <VX/vx_khr_cnn.h>

#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "ATypeDef.h"
#include "AMemory.h"


AMemory::~AMemory()
{
#ifdef __linux__
    close(m_dupFd);
    m_dataLen = 0;
    m_dataPtr = NULL;
#endif
}

/*read data from fd to memory*/
int AMemory::readFromFd(size_t size, int prot, int fd, size_t offset)
{
    if (fd < 0) {
        fprintf(stderr, "ANeuralNetworksMemory_createFromFd invalid fd %d\n" , fd);
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    if (size <= 0) {
        fprintf(stderr, "Invalid size\n");
        return ANEURALNETWORKS_BAD_DATA;
    }
#ifdef __linux__
    m_dupFd = dup(fd);
    if (m_dupFd == -1) {
        fprintf(stderr, "Failed to dup the fd\n");
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    m_dataLen = size;
    m_dataPtr = mmap(nullptr, size, prot, MAP_SHARED, m_dupFd, offset);
    if (m_dataPtr == MAP_FAILED) {
        fprintf(stderr, "Can't mmap the file descriptor.\n");
        return ANEURALNETWORKS_UNMAPPABLE;
    }
#endif
    return ANEURALNETWORKS_NO_ERROR;
}
