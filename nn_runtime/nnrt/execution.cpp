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
#include <thread>
#include <cstring>
#include <cassert>

#include "logging.hpp"
#include "execution.hpp"
#include "compilation.hpp"
#include "prepared_model.hpp"
#include "event.hpp"
#include "error.hpp"
#include "execution_task.hpp"
#include "file_map_memory.hpp"
#ifdef _DUMP_JSON_MODEL_
#include "dump_model/dump_json_model.hpp"
#endif

namespace {
    static const std::string tag = "Execution";
    /* Configure driver mem aligned size,
    * driver requests address and tensor size are aligend to 64 bytes. */
    const uint32_t ADDRESS_ALIGN_BYTES = 64;
    const uint32_t MEMORY_BLOCK_ALIGN_BYTES = 64;
}
namespace nnrt
{
Execution::Execution(Compilation* compilation)
    : compilation_(compilation)
{
    Model* model = compilation->getModel();
    for (auto& index : model->inputIndexes()) {
        op::OperandPtr operand = model->operand(index);
        ExecutionIOPtr io = std::make_shared<ExecutionIO>(operand);
        inputs_.push_back(io);
    }
    for (auto& index : model->outputIndexes()) {
        op::OperandPtr operand = model->operand(index);
        ExecutionIOPtr io = std::make_shared<ExecutionIO>(operand);
        outputs_.push_back(io);
    }
}

Execution::~Execution(){
    inputs_.clear();
    outputs_.clear();
}

int Execution::startCompute(EventPtr event)
{
    if (!compilation_) {
        return NNA_ERROR_CODE(UNEXPECTED_NULL);
    }

    std::unique_lock<std::mutex> lk(mutex_);
    if (isRunning()) {
        NNRT_LOGW_PRINT("Execution is already running.");
        return NNA_ERROR_CODE(OP_FAILED);
    }

    /**********************************************/
    // dump model and input data
#ifdef _DUMP_JSON_MODEL_
    Dump dump(compilation_->getModel());
    dump.getInputsData(this);
    dump.dump2json();
#endif
    /**********************************************/

    int err = NNA_ERROR_CODE(NO_ERROR);
    PreparedModelPtr prepared_model = compilation_->prepareModel(&err, inputs_);
    if (err == NNA_ERROR_CODE(NO_ERROR)) {
        event_ = event;
        event_->notify(Event::IN_PROCESS);
        TaskPtr task = std::make_shared<GraphTask>(
                prepared_model, event, inputs_, outputs_);
        get_global_task_queue().enqueue(task);
    }
    return err;
}

int Execution::compute()
{
    if (!compilation_) {
        return NNA_ERROR_CODE(UNEXPECTED_NULL);
    }
    std::unique_lock<std::mutex> lk(mutex_);
    if (isRunning()) {
        NNRT_LOGW_PRINT("Execution is already running.");
        return NNA_ERROR_CODE(INCOMPLETE);
    }

    /**********************************************/
    // dump model and input data
#ifdef _DUMP_JSON_MODEL_
    Dump dump(compilation_->getModel());
    dump.getInputsData(this);
    dump.dump2json();
#endif
    /**********************************************/

    int err = NNA_ERROR_CODE(NO_ERROR);
    PreparedModelPtr prepared_model = compilation_->prepareModel(&err, inputs_);
    if (err == NNA_ERROR_CODE(NO_ERROR)) {
        TaskPtr task = std::make_shared<GraphTask>(prepared_model, nullptr, inputs_, outputs_);
        if (!task) {
            NNRT_LOGE(tag) << "Fatal error: OOM";
            assert(false);
            err = NNA_ERROR_CODE(OUT_OF_MEMORY);
        } else {
            err = task->execute();
        }
    }
    return err;
}

int Execution::setInput(uint32_t index, const op::OperandPtr operand_type,
        const void* buffer, size_t length)
{
    std::unique_lock<std::mutex> lk(mutex_);
    if (isRunning()) {
        NNRT_LOGW_PRINT("Fail to modify the execution in running.");
        return NNA_ERROR_CODE(INCOMPLETE);
    }
    if (index >= inputs_.size()) {
        NNRT_LOGW_PRINT("Invalid input index(%u), max input size is %u.",
                index, inputs_.size());
        return NNA_ERROR_CODE(BAD_DATA);
    }
    Model* model = compilation_->getModel();
    uint32_t operand_index = model->inputIndex(index);
    if (operand_index < 0 ) {
        NNRT_LOGE(tag)<< "Fatal Error : invalid operand index < 0";
        return NNA_ERROR_CODE(BAD_DATA);
    }

    if (!buffer || 0 == length) {
        NNRT_LOGD_PRINT("Set idx=%d as novalue", index);

        model->operand(operand_index)->setNull();
        inputs_[index]->setNoValue();
    } else {
        model->operand(operand_index)->clearNull();
        if (operand_type) {
            model->updateOperand(operand_index, operand_type);
        }

        inputs_[index]->state = ExecutionIO::BUFFER;
        inputs_[index]->weak_mem_ref = model->add_memory_reference(buffer, length, true);
    }
    return NNA_ERROR_CODE(NO_ERROR);
}

int Execution::setInputFromMemory(uint32_t index, const op::OperandPtr operand_type,
        const Memory* memory, size_t offset, size_t length)
{
    if (!memory) {
        NNRT_LOGW_PRINT("Pass nullptr to memory.");
        return NNA_ERROR_CODE(UNEXPECTED_NULL);
    }
    std::unique_lock<std::mutex> lk(mutex_);
    if (isRunning()) {
        NNRT_LOGW_PRINT("Fail to modify the execution in running.");
        return NNA_ERROR_CODE(INCOMPLETE);
    }
    if (index >= inputs_.size()) {
        NNRT_LOGW_PRINT("Invalid input index(%u), max input size is %u.",
                index, inputs_.size());
        return NNA_ERROR_CODE(BAD_DATA);
    }

    Model* model = compilation_->getModel();
    uint32_t operand_index = model->inputIndex(index);
    if (model->operand(operand_index)) {
        model->operand(operand_index)->clearNull();
    } else {
        NNRT_LOGE(tag) << "Fatal error: found not found operand at " << operand_index;
        assert(false);
    }
    if (operand_type) {
        model->updateOperand(operand_index, operand_type);
    }

    inputs_[index]->state = ExecutionIO::BUFFER;
    inputs_[index]->weak_mem_ref = model->add_memory_reference(memory, offset, length);
    return NNA_ERROR_CODE(NO_ERROR);
}

int Execution::setOutput(uint32_t index, const op::OperandPtr operand_type,
        void* buffer, size_t length)
{
    if (!buffer) {
        NNRT_LOGW_PRINT("Pass nullptr to buffer.");
        return NNA_ERROR_CODE(UNEXPECTED_NULL);
    }
    std::unique_lock<std::mutex> lk(mutex_);
    if (isRunning()) {
        NNRT_LOGW_PRINT("Fail to modify the execution in running.");
        return NNA_ERROR_CODE(INCOMPLETE);
    }
    if (index >= outputs_.size()) {
        NNRT_LOGW_PRINT("Invalid output index(%u), max output size is %u.",
                index, outputs_.size());
        return NNA_ERROR_CODE(BAD_DATA);
    }
    if (!buffer) {
        outputs_[index]->setNoValue();
    } else {
        outputs_[index]->state = ExecutionIO::BUFFER;
        Model* model = compilation_->getModel();
        uint32_t operand_index = model->outputIndex(index);
        if (model->operand(operand_index)) {
            model->operand(operand_index)->clearNull();
        } else {
            NNRT_LOGE(tag) << "Fatal error: found not found operand at " << operand_index;
            assert(false);
        }
        if (operand_type) {
            model->updateOperand(operand_index, operand_type);
        }
        // !!! NEVER allocate inner buffer for output buffer
        outputs_[index]->weak_mem_ref = model->add_memory_reference(buffer, length, true);
    }
    return NNA_ERROR_CODE(NO_ERROR);
}

int Execution::setOutputFromMemory(uint32_t index, const op::OperandPtr operand_type,
        const Memory* memory, size_t offset, size_t length)
{
    if (!memory) {
        NNRT_LOGW_PRINT("Pass nullptr to memory.");
        return NNA_ERROR_CODE(UNEXPECTED_NULL);
    }
    std::unique_lock<std::mutex> lk(mutex_);
    if (isRunning()) {
        NNRT_LOGW_PRINT("Fail to modify the execution in running.");
        return NNA_ERROR_CODE(INCOMPLETE);
    }
    if (index >= outputs_.size()) {
        NNRT_LOGW_PRINT("Invalid output index(%u), max output size is %u.",
                index, outputs_.size());
        return NNA_ERROR_CODE(BAD_DATA);
    }

    Model* model = compilation_->getModel();
    uint32_t operand_index = model->outputIndex(index);
    if (model->operand(operand_index)) {
    model->operand(operand_index)->clearNull();
    }
    else {
        NNRT_LOGE(tag) << "Fatal error: can not found operand at " << operand_index ;
        assert(false);
    }
    if (operand_type) {
        model->updateOperand(operand_index, operand_type);
    }
    outputs_[index]->state = ExecutionIO::BUFFER;
    outputs_[index]->weak_mem_ref = model->add_memory_reference(memory, offset, length);
    return NNA_ERROR_CODE(NO_ERROR);
}

int Execution::getOutputOperandRank(uint32_t index, uint32_t* rank)
{
    if (!rank) {
        return NNA_ERROR_CODE(OUTPUT_INSUFFICIENT_SIZE);
    }
    Model* model = compilation_->getModel();
    uint32_t operand_index = model->outputIndex(index);
    if (operand_index == op::NNRT_INVALID_OPERAND_INDEX) {
        return NNA_ERROR_CODE(BAD_DATA);
    } else {
        *rank = model->operand(operand_index)->dimensions.size();
    }
    return NNA_ERROR_CODE(NO_ERROR);
}

int Execution::getOutputOperandDimensions(uint32_t index, uint32_t* dimensions)
{
    if (!dimensions) {
        return NNA_ERROR_CODE(OUTPUT_INSUFFICIENT_SIZE);
    }
    Model* model = compilation_->getModel();
    uint32_t operand_index = model->outputIndex(index);
    if (operand_index == op::NNRT_INVALID_OPERAND_INDEX) {
        return NNA_ERROR_CODE(BAD_DATA);
    } else {
        op::OperandPtr operand = model->operand(operand_index);
        std::memcpy(dimensions, operand->dimensions.data(),
                operand->dimensions.size() * sizeof(int32_t));
    }
    return NNA_ERROR_CODE(NO_ERROR);
}

}
