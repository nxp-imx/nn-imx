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


#include <memory>
#include <vector>
#include <iostream>
#include <string>
#include <stdio.h>
#include <algorithm>
#include <assert.h>
#include <VX/vx.h>
#include <VX/vxu.h>
#include <VX/vx_api.h>
#include <VX/vx_khr_cnn.h>
#include <thread>

#include "AExecution.h"
#include "ACompilation.h"

using namespace std;

 AExecution::AExecution(ACompilation* compilation):
         m_graph(compilation->getGraph()),
         m_AModel(compilation->getAModel()),
        m_runningFlag(false),m_exceptFlag(false)
{
    startTimePoint = 0.0;
#ifdef NN_DEBUG
    cout << "new Execution" <<endl;
#endif
}

AExecution::~AExecution()
{
    //m_graph = NULL;
//    m_runningFlag = false;
//    m_exceptFlag = false;
}

int AExecution::setInput(uint32_t index, const ANeuralNetworksOperandType* type,
                               const void* buffer, size_t length)
{
    if(!m_runningFlag)
    {
        //m_AModel->setOperandValue(m_AModel->getInputOperandIndex(index), buffer, length);
        uint32_t size = (uint32_t)m_inputIndexs.size();
        if (size <= index)
        {
            size = index + 1;
            m_inputIndexs.resize(size);
            m_inputExternBufs.resize(size);
        }

        m_inputIndexs[index].oprandIndex = m_AModel->getInputOperandIndex(index);
        m_inputIndexs[index].bufLen = (int32_t)length;
        m_inputExternBufs[index] = buffer;
#ifdef NN_DEBUG
        cout << "setInput operand index: " << m_inputIndexs[index].oprandIndex << " length: " << length << endl;
#endif
    }
    else
    {
        warning_log("Operation is invalid, since ANeuralNetworksExecution that has started computing could not be modified");
        return ANEURALNETWORKS_INCOMPLETE;
    }
    return ANEURALNETWORKS_NO_ERROR;
}

int AExecution::setInputFromMemory(uint32_t index, const ANeuralNetworksOperandType* type,
                                         const AMemory* memory, size_t offset, size_t length)
{
    return setInput(index, type, memory->getMemoryData(offset), length);
}


int AExecution::setOutput(uint32_t index, const ANeuralNetworksOperandType* type, void* buffer,
                            size_t length)
{
    if(!m_runningFlag)
    {

        if(buffer != NULL && length != 0)
        {
            uint32_t size = (uint32_t)m_outputInedx.size();
            if(size <= index)
            {
                size = index + 1;
                m_outputInedx.resize(size);
                m_outputExternBufs.resize(size);
            }
            m_outputInedx[index].oprandIndex = m_AModel->getOutputOperandIndex(index);
            m_outputInedx[index].bufLen = (int32_t)length;
            m_outputExternBufs[index] = buffer;
#ifdef NN_DEBUG
            cout << "setOutpus operand index: " << m_outputInedx[index].oprandIndex << " length: " << length << endl;
#endif
        }
    }
    else{
        warning_log("Operation is invalid, since ANeuralNetworksExecution that has started computing could not be modified");
        return ANEURALNETWORKS_INCOMPLETE;
    }

    return ANEURALNETWORKS_NO_ERROR;
}


int AExecution::setOutputFromMemory(uint32_t index, const ANeuralNetworksOperandType* type,
                                          const AMemory* memory, size_t offset, size_t length)
{
    return setOutput(index, type, (void *)memory->getMemoryData(offset), length);
}

bool AExecution::checkRuningStatus()
{
    return m_runningFlag;
}

void AExecution::clearRunningStatus()
{
    m_runningFlag = false;
}

void AExecution::setExceptFlag()
{
    m_exceptFlag = true;
}

void AExecution::clearExceptStatus()
{
    m_runningFlag = false;
}

int AExecution::startCompute()
{
    NN_ERR_CHECK( copyHost2Dev() );

    startTimePoint = getCurrentSystemTimeMs();

#ifdef DUMP_NEURALNETWORK
    m_AModel->writeNeuralNetworkToFile();
#else
#ifdef NN_DEBUG
    cout << "nn api start run graph..." << endl;
#endif

#if !NN_DEBUG_GRAPH
    VX_ERR_CHECK( vxScheduleGraph(m_graph) ); //
#else
    VX_ERR_CHECK( vxProcessGraph( m_graph) );
#endif
#endif
    m_runningFlag = true;

    return ANEURALNETWORKS_NO_ERROR;
}


int AExecution::copyHost2Dev()
{
#ifdef NN_DEBUG
    cout << "input copyHost2Dev size: " << m_inputExternBufs.size() << endl;
#endif
    for(uint32_t i = 0; i < m_inputExternBufs.size(); i++)
    {
        NN_ERR_CHECK( m_AModel->setOperandValue(m_inputIndexs[i].oprandIndex, m_inputExternBufs[i], m_inputIndexs[i].bufLen, VX_TENSOR_LIFE_TIME_DYNAMIC) );
    }
    return ANEURALNETWORKS_NO_ERROR;
}

int AExecution::copyDev2Host()
{
    //std::vector<AnnOperand> * oprands = m_AModel->getAllOperand();

    for(uint32_t i = 0; i < m_outputInedx.size(); i++)
    {
        NN_ERR_CHECK( m_AModel->getOperandValue(m_outputInedx[i].oprandIndex, m_outputExternBufs[i], m_outputInedx[i].bufLen) );
    }
    return ANEURALNETWORKS_NO_ERROR;
}

