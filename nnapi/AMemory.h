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


#ifndef _NNVX_MEMORY_H_
#define _NNVX_MEMORY_H_
#include <VX/vx.h>
#include <VX/vxu.h>
#include <VX/vx_api.h>
#include <VX/vx_khr_cnn.h>

#include "NeuralNetworks.h"

class AMemory
{
public:
    AMemory():m_dupFd(0), m_dataPtr(NULL), m_dataLen(0){};
    ~AMemory();
    int readFromFd(size_t size, int prot, int fd, size_t offset);
    const void* getMemoryData(size_t offset) const {return (char *)m_dataPtr + offset;}

private:
    int m_dupFd;
    void *m_dataPtr;
    size_t m_dataLen;
};


#endif //_NNVX_MEMORY_H_

