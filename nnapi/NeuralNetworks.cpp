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
#include <algorithm>
#include <VX/vx.h>
#include <VX/vxu.h>
#include <VX/vx_api.h>
#include <VX/vx_khr_cnn.h>
#include "AModel.h"
#include "AEvent.h"
#include "ACompilation.h"
#include "NeuralNetworks.h"
#include "AExecution.h"

using namespace std;


int ANeuralNetworksMemory_createFromFd(size_t size, int prot, int fd, size_t offset,
                                       ANeuralNetworksMemory** memory)
{
    *memory = NULL;
    AMemory *vxMemory = new AMemory();
    int n = vxMemory->readFromFd(size, prot, fd, offset);
    if (n != ANEURALNETWORKS_NO_ERROR)
    {
        error_log("fail to read from fd");
    }
    *memory = reinterpret_cast<ANeuralNetworksMemory*>(vxMemory);

    return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksMemory_free(ANeuralNetworksMemory* memory)
{
    AMemory* m = reinterpret_cast<AMemory*>(memory);
    delete m;
}

int ANeuralNetworksModel_create(ANeuralNetworksModel** model)
{
    if (!model) {
        cout << "ANeuralNetworksModel_create passed a nullptr"<<endl;
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    AModel *AnnModel = new AModel();
    if (AnnModel == NULL) {
        *model = NULL;
        return ANEURALNETWORKS_OUT_OF_MEMORY;
    }
    *model = reinterpret_cast<ANeuralNetworksModel*>(AnnModel);

    return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksModel_free(ANeuralNetworksModel* model)
{
    AModel* m = reinterpret_cast<AModel*>(model);
    delete m;
}

int ANeuralNetworksModel_finish(ANeuralNetworksModel* model)
{
    if (!model) {
        cout << "ANeuralNetworksModel_create passed a nullptr"<<endl;
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    AModel* m = reinterpret_cast<AModel*>(model);
    if(m->checkModelStatus())
    {
        fprintf(stderr, "ANeuralNetworksModel_finish has been called more than once");
        return ANEURALNETWORKS_BAD_STATE;
    }

    m->finish();

    return ANEURALNETWORKS_NO_ERROR;
}


int ANeuralNetworksModel_addOperand(ANeuralNetworksModel* model,
                                    const ANeuralNetworksOperandType* type)
{
    if (!model || !type) {
        cout << "ANeuralNetworksModel_addOperand passed a nullptr";
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    AModel* m = reinterpret_cast<AModel*>(model);
    if(m->checkModelStatus())
    {
        fprintf(stderr, "%s can not be modified after finished\n", __FUNCTION__);
        return ANEURALNETWORKS_BAD_DATA;
    }

    return m->addOperand(*type);
}

int ANeuralNetworksModel_setOperandValue(ANeuralNetworksModel* model, int32_t index,
                                         const void* buffer, size_t length)
{
    if (!model || !buffer) {
        cout << "ANeuralNetworksModel_setOperandValue passed a nullptr" <<endl;
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    AModel* m = reinterpret_cast<AModel*>(model);

    return m->setOperandValue(index, buffer, length);
}
int ANeuralNetworksModel_setOperandValueFromMemory(ANeuralNetworksModel* model, int32_t index,
                                                   const ANeuralNetworksMemory* memory,
                                                   size_t offset, size_t length)
{
    if (!model || !memory) {
        cout << "ANeuralNetworksModel_setOperandValue passed a nullptr" << endl;
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    const AMemory* mem = reinterpret_cast<const AMemory*>(memory);
    AModel* m = reinterpret_cast<AModel*>(model);

    return m->setOperandValueFromMemory(index, mem, offset, length);
}

int ANeuralNetworksModel_addOperation(ANeuralNetworksModel* model,
                                      ANeuralNetworksOperationType type, uint32_t inputCount,
                                      const uint32_t* inputs, uint32_t outputCount,
                                      const uint32_t* outputs)
{
    if (!model || !inputs || !outputs) {
        cout << "ANeuralNetworksModel_addOperation passed a nullptr" << endl;
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    AModel* m = reinterpret_cast<AModel*>(model);
    if(m->checkModelStatus())
    {
        fprintf(stderr, "%s can not be modified after finished\n", __FUNCTION__);
        return ANEURALNETWORKS_BAD_DATA;
    }
    return m->addOperation(type, inputCount, inputs, outputCount, outputs);
}

int ANeuralNetworksModel_identifyInputsAndOutputs(ANeuralNetworksModel* model, uint32_t inputCount,
                                                  const uint32_t* inputs, uint32_t outputCount,
                                                  const uint32_t* outputs)
{
    if (!model || !inputs || !outputs) {
        cout << ("ANeuralNetworksModel_identifyInputsAndOutputs passed a nullptr") << endl;
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    AModel* m = reinterpret_cast<AModel*>(model);

    if(m->checkModelStatus())
    {
        fprintf(stderr, "%s can not be modified after finished\n", __FUNCTION__);
        return ANEURALNETWORKS_BAD_DATA;
    }
    return m->identifyInputsAndOutputs(inputCount, inputs, outputCount, outputs);
}


int ANeuralNetworksCompilation_create(ANeuralNetworksModel* model,
                                      ANeuralNetworksCompilation** compilation)
{
    if (!model || !compilation) {
        cout << "ANeuralNetworksCompilation_create passed a nullptr" << endl;
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    AModel* m = reinterpret_cast<AModel*>(model);
    ACompilation* c =  new ACompilation(m);
    if (c == NULL)
    {
        cout << "faile to new ACompilation object" << endl;
        return ANEURALNETWORKS_OUT_OF_MEMORY;
    }

    *compilation = reinterpret_cast<ANeuralNetworksCompilation*>(c);
    return ANEURALNETWORKS_NO_ERROR;
}


void ANeuralNetworksCompilation_free(ANeuralNetworksCompilation* compilation)
{
    ACompilation* c = reinterpret_cast<ACompilation*>(compilation);
    delete c;

    compilation = NULL;
}

int ANeuralNetworksCompilation_setPreference(ANeuralNetworksCompilation* compilation,
                                             int32_t preference)
{
    return  (reinterpret_cast<ACompilation *>(compilation))->setPreference(preference);
}

int ANeuralNetworksCompilation_finish(ANeuralNetworksCompilation* compilation)
{
    return  (reinterpret_cast<ACompilation *>(compilation))->run();
}

int ANeuralNetworksExecution_create(ANeuralNetworksCompilation* compilation,
                                    ANeuralNetworksExecution** execution)
{

    if (!compilation || !execution) {
        cout << "ANeuralNetworksExecution_create passed a nullptr" << endl;
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    ACompilation* c = reinterpret_cast<ACompilation*>(compilation);
    AExecution* r = new AExecution(c);
    if(r == NULL)
    {
        error_log("fail to new AExecution");
    }

    *execution = reinterpret_cast<ANeuralNetworksExecution*>(r);

    return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksExecution_free(ANeuralNetworksExecution* execution)
{
    if (execution == NULL)
    {
        error_log("ANeuralNetworksExecution_free passed a nullptr");
        return;
    }

    AExecution * exe =reinterpret_cast<AExecution*>(execution);
    if( exe->checkRuningStatus())
    {
        exe->setExceptFlag();
    }
    else
    {
        delete exe;
        execution = NULL;
    }
}

int ANeuralNetworksExecution_setInput(ANeuralNetworksExecution* execution, int32_t index,
                                      const ANeuralNetworksOperandType* type, const void* buffer,
                                      size_t length)
{
    if (!execution) {
        cout << "ANeuralNetworksExecution_setInput passed a nullptr" << endl;
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    AExecution* r = reinterpret_cast<AExecution*>(execution);
    if(r->checkRuningStatus() )
    {
        error_log("ANeuralNetworksExecution is running, could not be modified\n");
        return ANEURALNETWORKS_INCOMPLETE;
    }

    return r->setInput(index, type, buffer, length);
}

int ANeuralNetworksExecution_setInputFromMemory(ANeuralNetworksExecution* execution, int32_t index,
                                                const ANeuralNetworksOperandType* type,
                                                const ANeuralNetworksMemory* memory, size_t offset,
                                                size_t length)
 {
    if (!execution || !memory) {
        cout << "ANeuralNetworksExecution_setInputFromMemory passed a nullptr" << endl;
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    const AMemory* m = reinterpret_cast<const AMemory*>(memory);
    AExecution* r = reinterpret_cast<AExecution*>(execution);
    if(r->checkRuningStatus() )
    {
     error_log("ANeuralNetworksExecution is running, could not be modified\n");
     return ANEURALNETWORKS_INCOMPLETE;
    }

    return r->setInputFromMemory(index, type, m, offset, length);
}


int ANeuralNetworksExecution_setOutput(ANeuralNetworksExecution* execution, int32_t index,
                                       const ANeuralNetworksOperandType* type, void* buffer,
                                       size_t length)
{
    if (!execution || !buffer) {
        cout << "ANeuralNetworksExecution_setOutput passed a nullptr" << endl;
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
    AExecution* r = reinterpret_cast<AExecution*>(execution);
    if(r->checkRuningStatus() )
    {
        error_log("ANeuralNetworksExecution is running, could not be modified\n");
        return ANEURALNETWORKS_INCOMPLETE;
    }
    return r->setOutput(index, type, buffer, length);
}

int ANeuralNetworksExecution_setOutputFromMemory(ANeuralNetworksExecution* execution, int32_t index,
                                                 const ANeuralNetworksOperandType* type,
                                                 const ANeuralNetworksMemory* memory, size_t offset,
                                                 size_t length)
 {
    if (!execution || !memory) {
        cout << "ANeuralNetworksExecution_setOutputFromMemory passed a nullptr" <<endl;
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    AExecution* r = reinterpret_cast<AExecution*>(execution);
    const AMemory* m = reinterpret_cast<const AMemory*>(memory);
    if(r->checkRuningStatus() )
    {
     error_log("ANeuralNetworksExecution is running, could not be modified\n");
     return ANEURALNETWORKS_INCOMPLETE;
    }

    return r->setOutputFromMemory(index, type, m, offset, length);

}

int ANeuralNetworksExecution_startCompute(ANeuralNetworksExecution* execution,
                                          ANeuralNetworksEvent** event)
{

    if (!execution || !event) {
        cout << "ANeuralNetworksExecution_startCompute passed a nullptr" <<endl;
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    AExecution* r = reinterpret_cast<AExecution*>(execution);
    if(r->checkRuningStatus() )
    {
        error_log("ANeuralNetworksExecution is running, could not be restart\n");
        return ANEURALNETWORKS_INCOMPLETE;
    }

    int n = r->startCompute();

    AEvent *ae = new AEvent(r);
    *event = reinterpret_cast<ANeuralNetworksEvent*>(ae);

    return n;
}


int ANeuralNetworksEvent_wait(ANeuralNetworksEvent* event)
{
    if (event == NULL)
    {
        std::cout << "ANeuralNetworksEvent_wait passed a nullptr" <<std::endl;
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }
#ifdef NN_DEBUG
    printf(" waiting for event\n");
#endif
    return  (reinterpret_cast<AEvent*> (event))->AEvent_wait();
}


void ANeuralNetworksEvent_free(ANeuralNetworksEvent* event)
{
    if(event == NULL){
        std::cout << "ANeuralNetworksEvent_wait passed a nullptr" <<std::endl;
        return ;
    }

#ifdef NN_DEBUG
    printf(" delete event\n");
#endif
    delete reinterpret_cast<AEvent *>(event);
    event = NULL;
}

// it is always relaxed in our driver.
int ANeuralNetworksModel_relaxComputationFloat32toFloat16(ANeuralNetworksModel* model, bool allow)
{
    return ANEURALNETWORKS_NO_ERROR;
}
