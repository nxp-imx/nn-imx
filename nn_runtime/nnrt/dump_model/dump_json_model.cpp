/****************************************************************************
*
*    Copyright (c) 2020 Vivante Corporation
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
#include <fstream>
#include "dump_json_model.hpp"
#include "model.hpp"

namespace nnrt {
    static size_t operandTypeSize(OperandType type) {
        switch (type) {
        case OperandType::BOOL:
        case OperandType::INT8:
        case OperandType::UINT8:
            return 1;
        case OperandType::INT16:
        case OperandType::UINT16:
        case OperandType::FLOAT16:
            return 2;
        case OperandType::INT32:
        case OperandType::UINT32:
        case OperandType::FLOAT32:
            return 4;
        case OperandType::FLOAT64:
            return 8;
        default:
            NNRT_LOGW_PRINT("the operand is not scalar, cannot get size info");
            break;
        }
        return 0;
    }
    void Dump::dump2json(){
        dumpModelData();
        std::ofstream ofs;
        ofs.open("model.json");
        ofs << rtModel_.toStyledString();
        ofs.close();
    }

    void Dump::getInputsData(const Execution *exec) {
        if (exec) {
            auto inputs = exec->inputs();
            for (size_t i = 0; i < inputs.size(); i++) {
                auto &input = inputs[i];
                size_t length = 0;
                int8_t *buffer = Model::getBuffer<int8_t>(input->weak_mem_ref.lock());
                if (buffer)
                    length = input->weak_mem_ref.lock()->len_;

                dumpOperandValue(
                    rtModel_["operand"]["container"][model_->inputIndexes()[i]],
                    (void *)buffer, length);
            }
        }
    }

    void Dump::convertRTmodel() {
        auto rtOperands = model_->operands();
        auto rtOperations = model_->operations();
        auto rtInputIndexes = model_->inputIndexes();
        auto rtOutputIndexes = model_->outputIndexes();;

        Json::Value jOperands, jOperations;

        for (const auto& operand : rtOperands) {
            dumpOperand(jOperands, operand.second);
            dumpOperandValue(operand.first, jOperands, operand.second);
        }

        for (const auto& operation : rtOperations) {
            dumpOpration(jOperations, operation.second);
        }

        /*dump the input and output of model*/
        Json::Value modelIn, modelOut;
        for (uint32_t i = 0; i < rtInputIndexes.size(); i++) {
            modelIn[i] = rtInputIndexes[i];
        }

        for (uint32_t i = 0; i < rtOutputIndexes.size(); i++) {
            modelOut[i] = rtOutputIndexes[i];
        }

        rtModel_["operand"] = jOperands;
        rtModel_["operation"] = jOperations;
        rtModel_["modelIn"] = modelIn;
        rtModel_["modelOut"] = modelOut;
    }

    void Dump::dumpModelData() {
        FILE *fid = fopen("model.data", "wb");
        if(fid){
            fwrite(&modelData_[0], 1, modelData_.size(), fid);
            fclose(fid);
        }else{
            NNRT_LOGE_PRINT("Fail to create the file");
        }
    }

    int Dump::dumpOperandValue(Json::Value &jOperand,
        void *buffer, size_t length) {
        size_t offset = 0;
        if (length > 0) {
            offset = (modelData_.size() + 3) &(~3);
            modelData_.resize(offset + length);
            memcpy(&modelData_[offset], buffer, length);
        }
        jOperand["offset"] = offset;
        jOperand["len"] = length;
        return 0;
    }

    int Dump::dumpOperandValue(uint32_t index,
                               Json::Value& jOperands,
                               const op::OperandPtr& operand) {
        auto& jOperand = jOperands["container"][index];

        size_t length = 0;
        int8_t* buffer = nullptr;
        if (!operand->isNull()) {
            if (!operand->isTensor()) {
                buffer = (int8_t*)&operand->scalar;
                length = operandTypeSize(operand->type);
            } else {
                // Not need to dump value for model input and output operands.
                // Model input value is dumped in getInputsData.
                const auto modelInputIds = model_->inputIndexes();
                const auto modelOutputIds = model_->outputIndexes();
                if (modelInputIds.end() ==
                        std::find(modelInputIds.begin(), modelInputIds.end(), index) &&
                    modelOutputIds.end() ==
                        std::find(modelOutputIds.begin(), modelOutputIds.end(), index)) {
                    buffer = model_->getBuffer<int8_t>(operand->weak_mem_ref.lock());
                    if (buffer) length = operand->weak_mem_ref.lock()->len_;
                }
            }
        }
        dumpOperandValue(jOperand, buffer, length);
        return 0;
    }

    int Dump::dumpOperand(Json::Value &jOperand,
        const op::OperandPtr& operand) {
        Json::Value item;
        Json::Value dims;

        // operand type info
        item["type"] = static_cast<int32_t>(operand->type);
        item["dimensionCount"] = operand->ndim();
        for (uint32_t i = 0; i < operand->dimensions.size(); i++)
            dims[i] = operand->dimensions[i];
        item["dims"] = dims;
        item["scale"] = operand->quant.scalar.scale;
        item["zeroPoint"] = operand->quant.scalar.zeroPoint;

        jOperand["container"].append(item);
        return 0;
    }

    int Dump::dumpOpration(Json::Value &jOperation,
        const op::OperationPtr& opration) {
        Json::Value item;
        Json::Value inputs;
        Json::Value outputs;

        item["type"] = static_cast<uint32_t>(opration->type());

        for (uint32_t i = 0; i < opration->inputs().size(); i++)
            inputs[i] = opration->inputs()[i];
        for (uint32_t i = 0; i < opration->outputs().size(); i++)
            outputs[i] = opration->outputs()[i];

        item["inputs"] = inputs;
        item["outputs"] = outputs;
        item["layout"] = static_cast<int>(opration->getDataLayout());

        jOperation["container"].append(item);
        return 0;
    }
}