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


#ifndef _A_MODEL_H_
#define _A_MODEL_H_
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
#include "NeuralNetworks.h"
#include "ATypeDef.h"
#include "AMemory.h"
#ifdef DUMP_NEURALNETWORK
#include "json.h"
#endif


typedef struct AnnOperand {
    /** The data type, e.g ANEURALNETWORKS_INT8. */
    int32_t type;
    /** The number of dimensions. It should be 0 for scalars. */
    uint32_t dimensionCount;
    /** The dimensions of the tensor. It should be nullptr for scalars. */
    uint32_t dimensions[NN_TENSOR_MAX_DIMENSION];
    /*the lenght of tensor, 0 by defaut*/
    bool isEmpty;
    /** These two fields are only used for quantized tensors.
     * They should be zero for scalars and non-fixed point tensors.
     * The dequantized value of each entry is (value - zeroPoint) * scale.
     */
    float scale;
    int32_t zeroPoint;
    //int32_t lifeTime;// 0:static 1:dynamic

    /*operand data*/
    vx_tensor tensor;
    struct {
        vx_enum lifeTime;
        vx_enum precision;
        vx_enum dataType;
        vx_enum rank;
        vx_bool valued;
    }tensorAttribute;

    union
    {
        int32_t   i32;
        uint32_t  ui32;
        float    fp32;
    }scalar;

    AnnOperand(): type(0), dimensionCount(0), isEmpty(true), scale(1.0f), zeroPoint(0), tensor(NULL){
        scalar.i32 = 0;
        tensorAttribute.lifeTime = VX_TENSOR_LIFE_TIME_DYNAMIC;
        tensorAttribute.rank = VX_TENSOR_RANK_CWHN;
        tensorAttribute.precision = VX_TENSOR_PRECISION_AUTO;
        tensorAttribute.valued = vx_false_e;
    }
} AnnOperand;

typedef struct AnnOperation {
    ANeuralNetworksOperationType type;
    std::vector<uint32_t> OpInputs;
    std::vector<uint32_t> OpOutputs;
} AnnOperation;

enum{
    NO_VALUE = 10  // indicate that there is no data in tensor
};

class AModel
{
public:
    AModel();
    ~AModel();
    int addOperand(const ANeuralNetworksOperandType& type);
    int createTensor(uint32_t index);
    int setOperandValue(uint32_t index, const void* buffer, size_t length, vx_enum lifeTime = VX_TENSOR_LIFE_TIME_STATIC);
    int setOperandValueFromMemory(uint32_t index, const AMemory *mem, size_t offset, size_t length);
    const int getOperandValue(uint32_t index, void* buffer, size_t length);
    int identifyInputsAndOutputs(uint32_t inputCount, const uint32_t* inputs,
                                       uint32_t outputCount, const uint32_t* outputs);
    int addOperation(ANeuralNetworksOperationType type, uint32_t inputCount,
                                const uint32_t* inputs, uint32_t outputCount,
                                const uint32_t* outputs);
    const std::vector<AnnOperation>* getAllOperation();
    void finish();
    bool checkModelStatus(){ return mCompleteModel;}
    const AnnOperation* getOneOperation(int32_t type);
    const AnnOperand* getOneOperand(uint32_t index);
    std::vector<AnnOperand>* getAllOperand();
    const vx_context getVXContext() { return  m_context;}
    uint32_t getInputOperandIndex(uint32_t i) const { return mInputIndexes[i]; }
    const AnnOperand& getInputOperand(uint32_t i) const {
        return mOperands[getInputOperandIndex(i)];
    }
    uint32_t getOutputOperandIndex(uint32_t i) const { return mOutputIndexes[i]; }
    const AnnOperand& getOutputOperand(uint32_t i) const {
        return mOperands[getOutputOperandIndex(i)];
    }
    size_t getInputOperandCount() const { return mInputIndexes.size(); }
    size_t getOutputOperandCount() const { return mOutputIndexes.size(); }

    int vxcMemcpy(vx_tensor& tensor, void *hostPtr, size_t length, vx_accessor_e usage);
#ifdef DUMP_NEURALNETWORK
    int writeNeuralNetworkToFile();
#endif

private:
    std::vector<AnnOperand> mOperands;
    std::vector<AnnOperation> mOperations;
    const static  uint32_t MAX_NUMBER_OF_OPERANDS = 0xFFFFFFFE;
    const static uint32_t MAX_NUMBER_OF_OPERATIONS = 0xFFFFFFFE;
    bool mCompleteModel;

    uint32_t operandCount() const {
        return static_cast<uint32_t>(mOperands.size());
        }
    uint32_t operationCount() const {
        return static_cast<uint32_t>(mOperations.size());
    }
    uint32_t inputCount() const { return static_cast<uint32_t>(mInputIndexes.size()); }
    uint32_t outputCount() const { return static_cast<uint32_t>(mOutputIndexes.size()); }

    int validateOperandList(uint32_t count, const uint32_t* list, uint32_t operandCount,
                            const char* tag);
    vx_uint32 vxcGetTypeSize(vx_enum format);
    void setFromIntList(std::vector<uint32_t>* vec, uint32_t count, const uint32_t* data);

#ifdef DUMP_NEURALNETWORK
    int getSystemTimeMs();
    int dumpOperand(const ANeuralNetworksOperandType& operand);
    int dumpOperandValue(uint32_t index, const void* buffer, size_t length);
    int dumpOpration(const AnnOperation opration, uint32_t inputCount, uint32_t outputCount);
#endif

private:
    vx_context   m_context;
    std::vector<uint32_t> mInputIndexes;
    std::vector<uint32_t> mOutputIndexes;

#ifdef DUMP_NEURALNETWORK
    Json::Value m_jOperandRoot;
    Json::Value m_jOprationRoot;

    std::vector<uint8_t> m_modelData;
#endif

};


#endif //_A_MODEL_H_
