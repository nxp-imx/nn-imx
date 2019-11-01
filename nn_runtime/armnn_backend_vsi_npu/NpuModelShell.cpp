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

#include "NpuModelShell.hpp"
#include "NpuTensorHandler.hpp"
#include "arm_nn_interpreter.hpp"

#include <algorithm>
#include <set>
#include <type_traits>
#include <utility>
#include <boost/log/trivial.hpp>

namespace adaption {
namespace utils {

FinalModelPtr MergeModels(const ModelStack& modelStack) {
    armnn::NpuTensorHandlerPtrList inputTensors;
    armnn::NpuTensorHandlerPtrList outputTensors;
    nnrt::ModelPtr model = std::make_shared<nnrt::Model>();
    // Caculate Set<InputTensor>
    // Caculate Set<OutputTensor>
    // Real input for finalModel is Set<Input> - (Set<Input> & Set<Output>)
    // Real output for finalModel is Set<Output> - (Set<Input> & Set<Output>)
    std::set<armnn::NpuTensorHandler*> allInputTensor;
    std::set<armnn::NpuTensorHandler*> allOutputTensor;
    std::set<armnn::NpuTensorHandler*> allTensor;

    for (auto& singleModel : modelStack) {
        for (auto& input : singleModel.second.first) {
            allInputTensor.insert(input);
            input->SetOperandIdInValid();
        }
        for (auto& output : singleModel.second.second) {
            allOutputTensor.insert(output);
            output->SetOperandIdInValid();
        }
    }

    // allTensor.resize(allInputTensor.size() + allOutputTensor.size());
    std::set_union(allInputTensor.begin(),
                   allInputTensor.end(),
                   allOutputTensor.begin(),
                   allOutputTensor.end(),
                   std::inserter(allTensor, allTensor.end()));

    std::set<armnn::NpuTensorHandler*> intersection;
    // intersection.resize(allTensor.size());
    std::set_intersection(allInputTensor.begin(),
                          allInputTensor.end(),
                          allOutputTensor.begin(),
                          allOutputTensor.end(),
                          std::inserter(intersection, intersection.end()));

    std::set<armnn::NpuTensorHandler*> modelInputTensors;
    // modelInputTensors.resize(allInputTensor.size());
    std::set_difference(allInputTensor.begin(),
                        allInputTensor.end(),
                        intersection.begin(),
                        intersection.end(),
                        std::inserter(modelInputTensors, modelInputTensors.end()));

    std::set<armnn::NpuTensorHandler*> modelOutputTensors;
    // modelOutputTensors.resize(allOutputTensor.size());
    std::set_difference(allOutputTensor.begin(),
                        allOutputTensor.end(),
                        intersection.begin(),
                        intersection.end(),
                        std::inserter(modelOutputTensors, modelOutputTensors.end()));

    std::map<armnn::NpuTensorHandler*, uint32_t> addedOperands;
    for (auto& singleModel : modelStack) {
        auto& curInputs = singleModel.second.first;
        const armnn::NpuTensorHandlerPtrList& curOutputs = singleModel.second.second;
        std::map<uint32_t, uint32_t> operandIdMap;  // previous id -> new id in final model
        // Add const tensors
        auto& operands = singleModel.first->operands();

        // SingleMode should include only single compute node
        assert(singleModel.first->operations().size() == 1);
        std::vector<uint32_t> inputOperandids = singleModel.first->operation(0)->inputs();
        std::vector<uint32_t> outputOperandids = singleModel.first->operation(0)->outputs();
        uint32_t inputOperandPos = 0;
        uint32_t outputOperandPos = 0;
        for (auto& operand : operands) {
            bool isInputOperand =
                (inputOperandids.end() !=
                 std::find(inputOperandids.begin(), inputOperandids.end(), operand.first));
            // TODO: implicit assumption here: input tensor must be located at the head of the input
            // operand list
            isInputOperand &= inputOperandPos < curInputs.size();

            bool isOuputOperand =
                (outputOperandids.end() !=
                 std::find(outputOperandids.begin(), outputOperandids.end(), operand.first));

            bool isInnerParameter = !isInputOperand && !isOuputOperand;

            if (isInnerParameter) {
                uint32_t newOperandId;
                (void)model->addOperand(operand.second, &newOperandId);
                operandIdMap.insert(std::make_pair(operand.first, newOperandId));
                continue;
            }

            if (isInputOperand) {
                if (curInputs[inputOperandPos]->IsOperandIdValid()) {
                    // This operand already setup by other node as its output
                    // Just refresh our operand id in the FinalModel
                    operandIdMap.insert(
                        std::make_pair(operand.first, curInputs[inputOperandPos]->GetOperandId()));
                } else {
                    // Add it to final model
                    uint32_t newOperandId;
                    (void)model->addOperand(operand.second, &newOperandId);
                    operand.second->clearNull();
                    addedOperands.insert(std::make_pair(curInputs[inputOperandPos], newOperandId));
                    curInputs[inputOperandPos]->SetOperandId(newOperandId);
                    operandIdMap.insert(std::make_pair(operand.first, newOperandId));
                }
                ++inputOperandPos;
                continue;
            }
            if (isOuputOperand) {
                if (curOutputs[outputOperandPos]->IsOperandIdValid()) {
                    // This operand already setup by other node as its input
                    // Just refresh our operand id in the FinalModel
                    operandIdMap.insert(std::make_pair(
                        operand.first, curOutputs[outputOperandPos]->GetOperandId()));
                } else {
                    // Add it to final model
                    uint32_t newOperandId;
                    (void)model->addOperand(operand.second, &newOperandId);
                    operand.second->clearNull();
                    addedOperands.insert(
                        std::make_pair(curOutputs[outputOperandPos], newOperandId));
                    curOutputs[outputOperandPos]->SetOperandId(newOperandId);
                    operandIdMap.insert(std::make_pair(operand.first, newOperandId));
                }

                ++outputOperandPos;
                continue;
            }
        }

        auto& opMap = singleModel.first->operations();
        assert(opMap.size() == 1);
        for (auto& opPair : opMap) {
            nnrt::op::OperationPtr op = opPair.second;
            auto& opInputIds = op->inputs();
            auto& opOutputIds = op->outputs();
            std::vector<uint32_t> finalInputIds;
            std::vector<uint32_t> finalOutputIds;
            for (auto& originId : opInputIds) {
                finalInputIds.push_back(operandIdMap[originId]);
            }
            for (auto& originId : opOutputIds) {
                finalOutputIds.push_back(operandIdMap[originId]);
            }

            // create same op in the finalModel
            (void)model->addOperation(opPair.second->type(),
                                      finalInputIds.data(),
                                      finalInputIds.size(),
                                      finalOutputIds.data(),
                                      finalOutputIds.size());
        }
    }

    std::vector<uint32_t> inputIds;
    for (auto& in : modelInputTensors) {
        inputTensors.push_back(in);
        inputIds.push_back(addedOperands.find(in)->second);
    }
    std::vector<uint32_t> outputIds;
    for (auto& out : modelOutputTensors) {
        outputTensors.push_back(out);
        outputIds.push_back(addedOperands.find(out)->second);
    }

    model->identifyInputsAndOutputs(
        inputIds.data(), inputIds.size(), outputIds.data(), outputIds.size());

    model->finish();

    return std::make_unique<adaption::FinalModel>(
        std::make_pair(model, std::make_pair(inputTensors, outputTensors)));
}
}
}  // End nnrt namespace

namespace armnn {
ModelShell::ModelShell(adaption::FinalModelPtr&& finalModel)
    : m_NativeModel(std::forward<adaption::FinalModelPtr&&>(finalModel)) {
    m_Compiler = std::make_unique<nnrt::Compilation>(m_NativeModel->first.get());
    m_Compiler->setInterpreter(new armnn::Armnn_Interpreter());
    m_ExecutionPtr = std::make_unique<nnrt::Execution>(m_Compiler.get());
}

ModelShell::~ModelShell() {
    m_Compiler.reset();
    m_ExecutionPtr.reset();
}

void ModelShell::Execute() {
    auto idx = 0;
    for (auto& input : m_NativeModel->second.first) {
        m_ExecutionPtr->setInput(idx, nullptr, input->data(), input->memSize());
        ++idx;
    }

    idx = 0;
    for (auto& output : m_NativeModel->second.second) {
        m_ExecutionPtr->setOutput(idx, nullptr, output->data(), output->memSize());
        ++idx;
    }

    auto event = std::make_shared<nnrt::Event>();

    // Disable nnrt do fp32tofp16, for armnn support fp16-turbo-mode
    m_NativeModel->first->relax(false);

    auto errCode = m_ExecutionPtr->startCompute(event);

    if (0 != errCode){
        assert(false);
        VSILOGE("Start Compute return error =%d", errCode);
    }else {
        event->wait();
    }
}
}
