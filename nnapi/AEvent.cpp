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


//
// Created by kg on 18-3-8.
//

#include "NeuralNetworks.h"
#include "AEvent.h"
#include <thread>

//AEvent::~AEvent()


void waitForGraph(AExecution *exe)
{
    VX_ERR_CHECK( vxWaitGraph(exe->getVXgraph()) );
    double finishTime = getCurrentSystemTimeMs();
    double startTime = exe->getStartTime();

    fprintf(stderr, "it takes %lfms to process the graph\n", finishTime - startTime);
    exe->clearRunningStatus();
    if(exe->getExecptFlag())
    {
#ifdef NN_DEBUG
        printf("excpetion for free execution\n");
#endif
        exe->copyDev2Host();
        delete exe;
    }
}

int AEvent::AEvent_wait()
{
#ifndef DUMP_NEURALNETWORK
    if( m_exe->getExecptFlag() )
    {
#if NN_DEBUG
        fprintf(stderr, "except free execution\n");
#endif
        //TODO: ANEURALNETWORKS_ERROR_DELETED
        std::thread t(waitForGraph, m_exe);
        t.detach();
        return  ANEURALNETWORKS_NO_ERROR;
    }

    // waiting to finishing the compute,
    // and then copy data to host

    VX_ERR_CHECK( vxWaitGraph(m_exe->getVXgraph()) );
    double finishTime = getCurrentSystemTimeMs();
    double startTime = m_exe->getStartTime();

    fprintf(stderr, "it takes %lfms to process the graph\n", finishTime - startTime);
#endif
    m_exe->clearRunningStatus();
    m_exe->copyDev2Host();

    return  ANEURALNETWORKS_NO_ERROR;
}
