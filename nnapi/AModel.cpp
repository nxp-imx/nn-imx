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


#include <fstream>
#include "AModel.h"

#ifdef __LINUX__
#include <sys/time.h>
#endif

using namespace std;

extern int enumConvertorANN2VX(int type);

AModel::AModel()
{
    mCompleteModel = false;
    m_context = vxCreateContext();
    if (m_context == NULL)
    {
        cout << "fail to create ovx context" << endl;
        assert(0);
    }
}

AModel::~AModel()
{
    for (uint32_t i = 0; i < operandCount(); i++)
    {
        if (NULL != mOperands[i].tensor)
        {
            vxReleaseTensor(&mOperands[i].tensor);
            mOperands[i].tensor =  NULL;
        }
    }
    if (m_context != NULL)
    {
        vxReleaseContext(&m_context);
    }
}

#ifdef DUMP_NEURALNETWORK

std::string data2str(const unsigned int * buffer, int len)
{
    char *strBuf = new char[8 * len + 1];
    memset(strBuf, 0, 8 * len + 1);
    for (int i = 0; i < len; ++i) {
        sprintf(strBuf + 8 * i, "%08x", buffer[i]);
    }
    strBuf[8*len] = '\0';
    std::string str(strBuf);
    delete [] strBuf;
    return str;
}
#ifdef __LINUX__
int AModel::getSystemTimeMs()
{
    int t = 0;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    t = tv.tv_sec * 1000 + tv.tv_usec / 1000;
    return t;
}
#endif
int AModel::writeNeuralNetworkToFile()
{
    char operandJsonName[100]="dump_operand.json";
    char oprationJsonName[100]="dump_opration.json";
//    int time = getSystemTimeMs();

    //sprintf(operandJsonName, "dump_operand_%d.json", time);
    printf("dump operand data to: %s\n", operandJsonName);
    //sprintf(oprationJsonName, "dump_opration_%d.json", time);
    printf("dump opration data to: %s\n", oprationJsonName);

    m_jOperandRoot["name"] = "operand";
    m_jOprationRoot["name"] = "operation";

    Json::Value modelIn, modelOut;
    for(uint32_t i = 0; i < inputCount(); i++)
    {
        modelIn[i] = mInputIndexes[i];
    }

    for(uint32_t i = 0; i < outputCount(); i++)
    {
        modelOut[i] = mOutputIndexes[i];
    }

    m_jOperandRoot["modelIn"] = modelIn;
    m_jOperandRoot["modelOut"] = modelOut;

    Json::Value tfliteModel;
    tfliteModel["operand"] = m_jOperandRoot;
    tfliteModel["operation"] = m_jOprationRoot;

    ofstream ofs;
    ofs.open("tflite.vivlite");
    ofs << tfliteModel.toStyledString();
    ofs.close();

    FILE *fid = fopen("tflite.vivdata", "wb");
    fwrite(&m_modelData[0], 1, m_modelData.size(), fid);
    fclose(fid);

    return 0;
}
int AModel::dumpOperand(const ANeuralNetworksOperandType& operand)
{
    Json::Value item;
    Json::Value dims;

    item["type"] = static_cast<int32_t>(operand.type);
    item["dimensionCount"] = operand.dimensionCount;
    for (uint32_t i = 0; i < operand.dimensionCount; i++)
        dims[i] = operand.dimensions[i];
    item["dims"] = dims;
    item["scale"] = operand.scale;
    item["zeroPoint"] = operand.zeroPoint;

    m_jOperandRoot["container"].append(item);

    return 0;
}
int AModel::dumpOperandValue(uint32_t index, const void* buffer, size_t length)
{
#ifdef NN_DEBUG
    printf("dumpOperandValue index: %d, length: %d\n", index, (int)length);
#endif
    AnnOperand& operand = mOperands[index];
    Json::Value item = m_jOperandRoot["container"][index];
    uint32_t offset = m_modelData.size() + 4 - (m_modelData.size() & 3);
    item["offset"] = offset;
    item["len"] = length;

    m_modelData.resize(offset + length);
    memcpy(&m_modelData[offset],buffer, length);

    m_jOperandRoot["container"][index] = item;

    return 0;
}
int AModel::dumpOpration(const AnnOperation opration, uint32_t inputCount, uint32_t outputCount)
{
    Json::Value item;
    Json::Value inputs;
    Json::Value outputs;

    item["type"] = static_cast<uint32_t>(opration.type);

    for (uint32_t i = 0; i < inputCount; i++)
        inputs[i] = opration.OpInputs[i];
    for (uint32_t i = 0; i < outputCount; i++)
        outputs[i] = opration.OpOutputs[i];

    item["inputs"] = inputs;
    item["outputs"] = outputs;

    m_jOprationRoot["container"].append(item);
    return 0;
}
#endif

vx_uint32 AModel::vxcGetTypeSize(vx_enum format)
{
    switch(format)
    {
        case VX_TYPE_INT8:
        case VX_TYPE_UINT8:
            return 1;
        case VX_TYPE_INT16:
        case VX_TYPE_UINT16:
            return 2;
        case VX_TYPE_INT32:
        case VX_TYPE_UINT32:
            return 4;
        case VX_TYPE_INT64:
        case VX_TYPE_UINT64:
            return 8;
        case VX_TYPE_FLOAT32:
            return 4;
        case VX_TYPE_FLOAT64:
            return 8;
        case VX_TYPE_ENUM:
            return 4;
        case VX_TYPE_FLOAT16:
            return 2;
    }
    return 4;
}

void AModel::setFromIntList(std::vector<uint32_t>* vec, uint32_t count, const uint32_t* data)
{
    vec->resize(count);
    for (uint32_t i = 0; i < count; i++)
    {
        (*vec)[i] = data[i];
    }
}

const AnnOperand* AModel::getOneOperand(uint32_t index)
{
    if (index > mOperands.size())
    {
        cout << "index out of range, operands size: "  << mOperands.size() << endl;
        assert(0);
        return NULL;
    }
    return &mOperands[index];
}

std::vector<AnnOperand>* AModel::getAllOperand()
{
    return &mOperands;
}

int AModel::vxcMemcpy(vx_tensor& tensor, void *hostPtr, size_t length, vx_accessor_e usage)
{
    vx_uint32       output_size[NN_TENSOR_MAX_DIMENSION];
    vx_uint32       stride_size[NN_TENSOR_MAX_DIMENSION];
    vx_tensor_addressing      tensor_addressing = NULL;
    vx_int32        num_of_dims;
    vx_enum         data_format;

    VX_ERR_CHECK( vxQueryTensor(tensor, VX_TENSOR_DIMS, output_size, sizeof(output_size)) );
    VX_ERR_CHECK( vxQueryTensor(tensor, VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)) );
    VX_ERR_CHECK( vxQueryTensor(tensor, VX_TENSOR_DATA_TYPE, &data_format, sizeof(data_format)) );

    stride_size[0] = vxcGetTypeSize(data_format);
    for (int i = 1; i < num_of_dims; i++)
    {
        stride_size[i] = stride_size[i-1] * output_size[i - 1];
    }

#ifdef NN_DEBUG
    cout << "output dim: " << num_of_dims << " data format: " << data_format << endl;
    cout << "outout size: " << output_size[0] << output_size[1] << output_size[2] << output_size[3] << endl;
#endif

    tensor_addressing    = vxCreateTensorAddressing(m_context, output_size, stride_size, num_of_dims);
    vx_bool value = vx_true_e;
    if (tensor_addressing == NULL)
    {
        cout << "fail to create tensor address" << endl;
        goto error;
    }

    VX_ERR_CHECK( vxCopyTensorPatchForNN11(tensor, NULL, tensor_addressing, hostPtr, usage, 0) );
    VX_ERR_CHECK( vxReleaseTensorAddressing(&tensor_addressing) );

    VX_ERR_CHECK( vxSetTensorAttribute(tensor, VX_TENSOR_VALUE,     &value,     sizeof(vx_bool)) );

    return ANEURALNETWORKS_NO_ERROR;

error:
    if (tensor)
        vxReleaseTensor(&tensor);

    if (tensor_addressing)
        vxReleaseTensorAddressing(&tensor_addressing);

    assert(0);
    return ANEURALNETWORKS_BAD_DATA;
}

int AModel::addOperand(const ANeuralNetworksOperandType& type)
{
    size_t idx = mOperands.size();
    mOperands.resize(idx + 1);

    AnnOperand& operand = mOperands[idx];
    operand.type = static_cast<int32_t>(type.type);
    operand.scale = type.scale;
    operand.zeroPoint = type.zeroPoint;
    operand.dimensionCount = type.dimensionCount;

    for(uint32_t i = 0; i < type.dimensionCount; i++)
    {
        operand.dimensions[i] = type.dimensions[i];
    }

    operand.isEmpty = isEmpty(operand.dimensionCount, operand.dimensions);

#ifdef NN_DEBUG
    fprintf(stderr, "add operand idx: %d, type: %d\n", (int)idx, (int)operand.type);
#endif

#ifdef DUMP_NEURALNETWORK
    dumpOperand(type);
#endif

    return ANEURALNETWORKS_NO_ERROR;
}

int AModel::setOperandValue(uint32_t index, const void* buffer, size_t length, vx_enum lifeTime)
{
    CHECK_LESS(index, operandCount());
#ifdef NN_DEBUG
    printf("setOperandValue index:%d, length: %d\n", index, (int)length);
#endif

    AnnOperand& operand = mOperands[index];
    if(isTensor(operand.type))
    {
        if(buffer != NULL && 0 != length && !operand.isEmpty)
        {
            if (operand.tensor == NULL )
            {
                NN_ERR_CHECK( createTensor(index) );
            }

            NN_ERR_CHECK( vxcMemcpy(operand.tensor, (void *)buffer, length, VX_WRITE_ONLY) );

            VX_ERR_CHECK( vxSetTensorAttribute(operand.tensor, VX_TENSOR_LIFETIME,  &lifeTime,  sizeof(vx_enum)) );
            operand.tensorAttribute.lifeTime = lifeTime;
            operand.tensorAttribute.valued = vx_true_e;
        }

    }
    else
    {
        CHECK_EQUAL(4, length);
        operand.scalar.i32 = ((int32_t*)buffer)[0];
    }

#ifdef DUMP_NEURALNETWORK
    dumpOperandValue(index, buffer, length);
#endif

    return ANEURALNETWORKS_NO_ERROR;
}

int AModel::setOperandValueFromMemory(uint32_t index, const AMemory *mem, size_t offset, size_t length)
{
    CHECK_LESS(index, operandCount());
#ifdef NN_DEBUG
    AnnOperand& operand = mOperands[index];
    printf("setOperandValueFromMemory index:%d, offset: %d, length: %d, operand type: %d, w/h/c/n: %d/%d/%d/%d\n",
           index, (int)offset, (int)length, operand.type, operand.dimensions[0], operand.dimensions[1], operand.dimensions[2], operand.dimensions[3]);
#endif
    const void* dataPtr = (void*)(mem->getMemoryData(offset) );
    setOperandValue(index, dataPtr, length);

#ifdef DUMP_NEURALNETWORK
    dumpOperandValue(index, dataPtr, length);
#endif
    return ANEURALNETWORKS_NO_ERROR;
}

int AModel::createTensor(uint32_t index)
{
    CHECK_LESS(index, operandCount());
    AnnOperand& operand = mOperands[index];
#ifdef NN_DEBUG
    fprintf(stderr,"VXcreateTensor index:%d, %d dims, dim:(", index, operand.dimensionCount);
    for(size_t i = 0; i < operand.dimensionCount; i++)
        fprintf(stderr," %d ", operand.dimensions[i]);
    fprintf(stderr,")\n");
#endif

    if(operand.isEmpty)
        return ANEURALNETWORKS_NO_ERROR;

    vx_tensor_create_params_t param;
    INITIALIZE_STRUCT(param);

    param.num_of_dims = operand.dimensionCount;
    param.sizes = operand.dimensions;
    param.data_format = enumConvertorANN2VX(operand.type);

    if(ANEURALNETWORKS_TENSOR_QUANT8_ASYMM == operand.type || (ANEURALNETWORKS_TENSOR_INT32 == operand.type))
    {
        param.quant_format = ((operand.scale != 0.f) || (operand.zeroPoint != 0)) ? VX_QUANT_AFFINE_SCALE: VX_QUANT_NONE;
        param.quant_data.affine.scale     = operand.scale ==  0.f? 1.0f : operand.scale;
        param.quant_data.affine.zeroPoint = operand.zeroPoint;
    }

    operand.tensor = vxCreateTensor2(m_context, &param, sizeof(vx_tensor_create_params_t));

    if (operand.tensor == NULL)
    {
        cout << "fail to creat tensor" << __LINE__ << endl;
        assert(0);
    }

    vx_enum precison = VX_TENSOR_PRECISION_AUTO;
    vx_enum rank = (2 == operand.dimensionCount? VX_TENSOR_RANK_SN:VX_TENSOR_RANK_CWHN);
    VX_ERR_CHECK( vxSetTensorAttribute(operand.tensor, VX_TENSOR_RANK,      &rank,      sizeof(vx_enum)) );
    VX_ERR_CHECK( vxSetTensorAttribute(operand.tensor, VX_TENSOR_PRECISION, &precison,  sizeof(vx_enum)) );

    operand.tensorAttribute.rank = rank;
    operand.tensorAttribute.precision = precison;
    operand.tensorAttribute.dataType = enumConvertorANN2VX(operand.type);

    return ANEURALNETWORKS_NO_ERROR;
}

//read tensor data from GPU
const int AModel::getOperandValue(uint32_t index, void* buffer, size_t length)
{
    CHECK_LESS(index, operandCount());
#ifdef NN_DEBUG
    cout << "getOperandValue index: " << index << " length: " << length << endl;
#endif
    if (index >= operandCount()) {
        cout << "ANeuralNetworksModel_setOperandValue setting operand " << index << " of "
             << operandCount() << endl;
        assert(0);
    }
    AnnOperand& operand = mOperands[index];

    if (isTensor(operand.type))//tensor
    {
        NN_ERR_CHECK( vxcMemcpy(operand.tensor, buffer, length, VX_READ_ONLY) );
    }
    else
    {
         ((int32_t*)buffer)[0] = operand.scalar.i32;
    }

    return ANEURALNETWORKS_NO_ERROR;
}


const std::vector<AnnOperation>* AModel::getAllOperation()
{
    return &mOperations;
}

const AnnOperation* AModel::getOneOperation(int32_t type)
{
    for (uint32_t i = 0; i < operationCount(); i++)
    {
        AnnOperation& entry = mOperations[i];
        if (entry.type == type)
        {
            return &mOperations[i];
        }
    }
    return NULL;
}

int AModel::addOperation(ANeuralNetworksOperationType type, uint32_t inputCount,
                               const uint32_t* inputs, uint32_t outputCount,
                               const uint32_t* outputs)
{
    uint32_t operationIndex = operationCount();
    mOperations.resize(operationIndex + 1);
    AnnOperation& entry = mOperations[operationIndex];
    entry.type = static_cast<uint32_t>(type);

    for (uint32_t i = 0; i < inputCount; i++)
    {
        CHECK_LESS(inputs[i], operandCount());
    }
    for (uint32_t i = 0; i < outputCount; i++)
    {
        CHECK_LESS(outputs[i], operandCount());
    }

    setFromIntList(&entry.OpInputs, inputCount, inputs);
    setFromIntList(&entry.OpOutputs, outputCount, outputs);

#ifdef DUMP_NEURALNETWORK
    dumpOpration(mOperations[operationIndex], inputCount, outputCount);
#endif

    return ANEURALNETWORKS_NO_ERROR;
}


int AModel::validateOperandList(uint32_t count, const uint32_t* list, uint32_t operandCount,
                        const char* tag)
{
    for (uint32_t i = 0; i < count; i++)
    {
        CHECK_LESS(list[i], operandCount);
    }
    return ANEURALNETWORKS_NO_ERROR;
}

int AModel::identifyInputsAndOutputs(uint32_t inputCount, const uint32_t* inputs,
                                             uint32_t outputCount, const uint32_t* outputs)
{
    NN_ERR_CHECK( validateOperandList(inputCount, inputs, operandCount(),
                                "ANeuralNetworksModel_identifyInputsAndOutputs inputs") );

    NN_ERR_CHECK( validateOperandList(outputCount, outputs, operandCount(),
                            "ANeuralNetworksModel_identifyInputsAndOutputs outputs") );

    // Makes a copy of the index list, validates the arguments, and changes
    // the lifetime info of the corresponding operand.
    auto setArguments = [&](std::vector<uint32_t>* indexVector, uint32_t indexCount,
                            const uint32_t* indexList) -> bool {
        indexVector->resize(indexCount);
        for (uint32_t i = 0; i < indexCount; i++)
        {
            const uint32_t operandIndex = indexList[i];
            if (operandIndex >= mOperands.size()) {
                cout << "ANeuralNetworksModel_identifyInputsAndOutputs Can't set input or output "
                              "to be "
                           << operandIndex << " as this exceeds the number of operands "
                           << mOperands.size() << endl;
                return false;
            }
            (*indexVector)[i] = operandIndex;

            if(mOperands[operandIndex].tensor == NULL)
            {
                createTensor(operandIndex);
                mOperands[operandIndex].tensorAttribute.lifeTime = VX_TENSOR_LIFE_TIME_DYNAMIC;
            }
        }
        return true;
    };

    if (!setArguments(&mInputIndexes, inputCount, inputs) ||
        !setArguments(&mOutputIndexes, outputCount, outputs))
    {
        return ANEURALNETWORKS_BAD_DATA;
    }

    return ANEURALNETWORKS_NO_ERROR;
}

void AModel::finish()
{
    if(mCompleteModel)
    {
        error_log("ANeuralNetworksModel_finish has been called more than once\n");
    }
    mCompleteModel = true;
}
