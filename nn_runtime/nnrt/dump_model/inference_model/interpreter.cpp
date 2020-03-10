/****************************************************************************
*
*    Copyright (c) 2019 Vivante Corporation
*
*    Permission is hereby granted, free of charge, to any person obtaining a
*    copy of this software and associated documentation files (the "Software"),
*    to deal in the Software without restriction, including without limitation
*    the rights to use, copy, modify, merge, publish, distribute, sublicense,
*    and/or sell copies of the Software, and to permit persons to whom the
*    Software is furnished to do so, subject to the following conditions:
*
*    The above copyright notice and this permission notice shall be included in
*    all copies or substantial portions of the Software.
*
*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
*    DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/

#include "interpreter.h"
#include "error.hpp"
#include "nnrt/event.hpp"
#ifdef NNAPI_INTERPRETER
#include "nnrt/model_transform/nnapi_interpreter.hpp"
#else
#include "armnn/backend/vsi_npu/arm_nn_interpreter.hpp"
#endif
#include <fstream>

struct EventShell {
    EventShell()
        : event_(std::make_shared<nnrt::Event>())
    {}
    int wait() { return event_->wait(); };
    nnrt::EventPtr event() { return event_; }
private:
    nnrt::EventPtr event_;
};

int Interpreter::interpreterRT()
{
    Json::Reader reader;
    std::ifstream op_ifs(rtModelPath);
    if (!reader.parse(op_ifs, jModel))
    {
        std::cout << " read oprand error\n";
        assert(0);
        return -1;
    }

    if (-1 == readData(modelDataPath))
        return -1;

    getModelIO(jModel["modelIn"], jModel["modelOut"]);
    //mallocOutputBuffer();
    mModel = new nnrt::Model();
    addOprandFromJsonFile(jModel["operand"]["container"]);
    addOperationFromJsonFile(jModel["operation"]["container"]);
    setModelIO();

    return 0;
}

int Interpreter::run() {
    mCompilator = new nnrt::Compilation(mModel);
#ifdef NNAPI_INTERPRETER
    mCompilator->setInterpreter(new nnrt::NnApiInterpreter());
#else
    mCompilator->setInterpreter(new armnn::Armnn_Interpreter());
#endif
    mCompilator->run();

    mExecution = new nnrt::Execution(mCompilator);
    setInputValue();
    setOutputBuffer();

    mExecution->compute();
    return 0;
}

int Interpreter::readData(const std::string dataFile)
{
    FILE *fid = fopen(dataFile.c_str(), "rb");
    if (fid == NULL)
        return -1;

    fseek(fid, 0L, SEEK_END);
    size_t length = ftell(fid);
    fseek(fid, 0, SEEK_SET);

    mData.resize(length);
    fread(&mData[0], 1, length, fid);
    fclose(fid);

    return 0;
}

static void getIndexFromJson(Json::Value &jindex, std::vector<uint32_t> &vindex) {
    for (auto &idx : jindex)
        vindex.push_back(idx.asUInt());
}

static void updateOperand(nnrt::op::OperandPtr operand, Json::Value &op) {
    operand->type = static_cast<nnrt::OperandType>(op["type"].asInt());
    operand->quant.scalar.scale = op["scale"].asFloat();
    operand->quant.scalar.zeroPoint = op["zeroPoint"].asInt();
    int nDims = op["dimensionCount"].asInt();
    if (nDims > 0) {
        operand->dimensions.resize(nDims);
        for (int i = 0; i < nDims; i++) {
            operand->dimensions[i] = op["dims"][i].asUInt();
        }
    }
}

void Interpreter::getModelIO(Json::Value &modelIn, Json::Value &modelOut) {
    getIndexFromJson(modelIn, mInputs);
    getIndexFromJson(modelOut, mOutputs);
}

void Interpreter::setModelIO(){
    mModel->identifyInputsAndOutputs(
        mInputs.data(), mInputs.size(), mOutputs.data(), mOutputs.size());
}

void Interpreter::addOprandFromJsonFile(Json::Value &operands){

    auto checkIndexIsInOrOUt = [](uint32_t index, std::vector<uint32_t> &vIndex)->bool
    {
        for (auto inputIndex : vIndex)
            if (index == inputIndex)
                return true;
        return false;
    };

    for (int i = 0; i < operands.size(); i++) {
        auto &op = operands[i];

        // get oprand type info
        uint32_t out_index;
        int err = NNA_ERROR_CODE(NO_ERROR);
        nnrt::op::OperandPtr operand = mModel->addOperand(nullptr, &out_index);
        if (!operand){
            err = NNA_ERROR_CODE(OUT_OF_MEMORY);
        }
        updateOperand(operand, op);

        // for input operand, set its value during execution
        if (checkIndexIsInOrOUt(i, mInputs))
            continue;

        // set oprand value.
        if (op["len"].asUInt() > 0){
            size_t offset = op["offset"].asUInt();
            size_t len = op["len"].asUInt();
            void *buffer = mData.data() + offset;
            if (NNA_ERROR_CODE(NO_ERROR) != mModel->setOperandValue(out_index, buffer, len))
                assert(0);
        }
    }//for (uint32_t i = 0; i < oprandContainer.size(); i++)
}

void Interpreter::addOperationFromJsonFile(Json::Value &operations){
    for (auto &operation : operations)
    {
        nnrt::OperationType type =
            static_cast<nnrt::OperationType>(operation["type"].asUInt());
        nnrt::DataLayout layout =
            static_cast<nnrt::DataLayout>(operation["layout"].asUInt());
        std::vector<uint32_t> inputs;
        std::vector<uint32_t> outputs;

        if (operation["inputs"].size())
            getIndexFromJson(operation["inputs"], inputs);
        if (operation["outputs"].size())
            getIndexFromJson(operation["outputs"], outputs);

        nnrt::op::OperationPtr op = mModel->addOperation(type,
            inputs.data(), inputs.size(), outputs.data(), outputs.size());
        if (!op){
            std::cout << "fail to add operation\n";
            assert(0);
        }
        op->setDataLayout(layout);
    }
}

void Interpreter::setInputValue() {
    Json::Value root;
    Json::Reader reader;

    auto & modelInputs = mModel->inputIndexes();
    Json::Value &jOperands = jModel["operand"]["container"];
    for (size_t i = 0; i < modelInputs.size(); i++) {
        auto &jInput = jOperands[ modelInputs[i]];
        uint32_t offset = jInput["offset"].asUInt();
        uint32_t length = jInput["len"].asUInt();
        void * buffer = mData.data() + offset;

        //nnrt::op::OperandPtr operand_ptr = std::make_shared<nnrt::op::Operand>();
        //updateOperand(operand_ptr, jInput);
        mExecution->setInput(i, nullptr, buffer, length);
    }
}

void Interpreter::setOutputBuffer()
{
    auto &modelOutputs = mModel->outputIndexes();
    mOutputData.resize(modelOutputs.size());
    for (size_t i = 0; i < modelOutputs.size(); i++) {
        size_t size = mModel->operand(modelOutputs[i])->bytes();
        mOutputData[i].resize(size);
        mExecution->setOutput(i, nullptr, mOutputData[i].data(), size);
    }
}
