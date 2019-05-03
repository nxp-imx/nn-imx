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



#ifndef _A_EXECUTION_H_
#define _A_EXECUTION_H_
#include <VX/vx.h>
#include <VX/vxu.h>
#include <VX/vx_api.h>
#include <VX/vx_khr_cnn.h>
#include <memory>
#include "AMemory.h"
#include "NeuralNetworks.h"
#include "ACompilation.h"

class AExecution
{
public:
    explicit AExecution(ACompilation* compilation);
    ~AExecution();

    int setInput(uint32_t index, const ANeuralNetworksOperandType* type, const void* buffer, size_t length);
    int setInputFromMemory(uint32_t index, const ANeuralNetworksOperandType* type, const AMemory* memory, size_t offset, size_t length);

    int setOutput(uint32_t index, const ANeuralNetworksOperandType* type, void* buffer, size_t length);
    int setOutputFromMemory(uint32_t index, const ANeuralNetworksOperandType* type, const AMemory* memory, size_t offset, size_t length);

    int startCompute();

    vx_graph  getVXgraph() { return  m_graph;}

    bool checkRuningStatus();
    void clearRunningStatus();

    void setExceptFlag();
    bool getExecptFlag() {return m_exceptFlag; };
    void clearExceptStatus();

    int copyHost2Dev(); // copy data from cpu to tenser
    int copyDev2Host(); // copy data from tenser to cpu

private:
    vx_graph     m_graph;
    AModel *m_AModel;
    bool m_runningFlag;
    bool m_exceptFlag;

    struct bufInfo{
        int32_t bufLen;
        int32_t oprandIndex;
    };

    // map input|output extern buffer to oprands.
    std::vector<bufInfo> m_inputIndexs;
    std::vector<bufInfo> m_outputInedx;

    std::vector<const void *> m_inputExternBufs;
    std::vector<void*> m_outputExternBufs;
};


#endif //_A_EXECUTION_H_



