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
#include "execution.hpp"
#include <thread>
#include <cstring>
#include <cassert>

#include "nnrt/logging.hpp"
#include "nnrt/shared_context.hpp"
#include "nnrt/compilation.hpp"
#include "nnrt/prepared_model.hpp"
#include "nnrt/event.hpp"
#include "nnrt/error.hpp"
#include "nnrt/execution_io.hpp"
#include "nnrt/execution_task.hpp"
#include "nnrt/file_map_memory.hpp"
#ifdef _DUMP_JSON_MODEL_
#include "nnrt/dump_model/dump_json_model.hpp"
#endif

namespace {
    static const std::string tag = "Execution";
}
namespace nnrt
{
/**
 * Driver only support one context in a thread.
 */
SharedContextPtr global_ovx_context;

struct ContextDeleter {
    void operator()(vsi_nn_context_t ctx) {
        NNRT_LOGD_PRINT("Release context");
        vsi_nn_ReleaseContext(&ctx);
    }
};

struct Execution::Private {
    /**
     * Private interface to check if current is running.
     * @note This API is NOT thread safe.
     */
    inline bool isRunning() {
        return (event_ && event_->state() == Event::IN_PROCESS);
    }

    std::vector<ExecutionIOPtr> inputs_;
    std::vector<ExecutionIOPtr> outputs_;
    SharedContextPtr ovx_context_;
    Compilation* compilation_{};
    std::mutex mutex_;
    EventPtr event_;
};

const std::vector<ExecutionIOPtr> &Execution::inputs() const { return d->inputs_; }

Execution::Execution(Compilation* compilation)
    : d(new Private)
{
    d->compilation_ = compilation;
    Model* model = compilation->getModel();
    for (auto& index : model->inputIndexes()) {
        op::OperandPtr operand = model->operand(index);
        ExecutionIOPtr io = std::make_shared<ExecutionIO>(operand);
        d->inputs_.push_back(io);
    }
    for (auto& index : model->outputIndexes()) {
        op::OperandPtr operand = model->operand(index);
        ExecutionIOPtr io = std::make_shared<ExecutionIO>(operand);
        d->outputs_.push_back(io);
    }
    if (!global_ovx_context) {
        global_ovx_context.reset(vsi_nn_CreateContext(), ContextDeleter());
    }
    d->ovx_context_ = global_ovx_context;
}

Execution::~Execution(){
    for (auto i : d->inputs_) {
        d->compilation_->getModel()->remove_memory_reference(i->weak_mem_ref.lock());
    }
    d->inputs_.clear();
    for (auto o : d->outputs_) {
        d->compilation_->getModel()->remove_memory_reference(o->weak_mem_ref.lock());
    }
    d->outputs_.clear();
}

// Async Compute API
int Execution::startCompute(EventPtr event)
{
    if (!d->compilation_) {
        return NNA_ERROR_CODE(UNEXPECTED_NULL);
    }

    std::unique_lock<std::mutex> lk(d->mutex_);
    if (d->isRunning()) {
        NNRT_LOGW_PRINT("Execution is already running.");
        return NNA_ERROR_CODE(OP_FAILED);
    }

    /**********************************************/
    // dump model and input data
#ifdef _DUMP_JSON_MODEL_
    auto dump_json_enabled = 0;
    ::nnrt::OS::getEnv("DUMP_JSON_MODEL", dump_json_enabled);
    if (dump_json_enabled) {
        Dump dump(d->compilation_->getModel());
        dump.getInputsData(this);
        dump.dump2json();
    }
#endif
    /**********************************************/

    int err = NNA_ERROR_CODE(NO_ERROR);
    PreparedModelPtr prepared_model = d->compilation_->prepareModel(&err, d->inputs_, d->ovx_context_);
    if (err == NNA_ERROR_CODE(NO_ERROR)) {
        d->event_ = event;
        d->event_->notify(Event::IN_PROCESS);
        TaskPtr task = std::make_shared<GraphTask>(
                prepared_model, event, d->inputs_, d->outputs_);
        get_global_task_queue().enqueue(task);
    }
    return err;
}

int Execution::compute()
{
    if (!d->compilation_) {
        return NNA_ERROR_CODE(UNEXPECTED_NULL);
    }
    std::unique_lock<std::mutex> lk(d->mutex_);
    if (d->isRunning()) {
        NNRT_LOGW_PRINT("Execution is already running.");
        return NNA_ERROR_CODE(INCOMPLETE);
    }

    /**********************************************/
    // dump model and input data
#ifdef _DUMP_JSON_MODEL_
    auto dump_json_enabled = 0;
    ::nnrt::OS::getEnv("DUMP_JSON_MODEL", dump_json_enabled);
    if (dump_json_enabled) {
        Dump dump(d->compilation_->getModel());
        dump.getInputsData(this);
        dump.dump2json();
    }
#endif
    /**********************************************/

    int err = NNA_ERROR_CODE(NO_ERROR);
    PreparedModelPtr prepared_model = d->compilation_->prepareModel(&err, d->inputs_, d->ovx_context_);
    if (err == NNA_ERROR_CODE(NO_ERROR)) {
        TaskPtr task = std::make_shared<GraphTask>(prepared_model, nullptr, d->inputs_, d->outputs_);
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
    std::unique_lock<std::mutex> lk(d->mutex_);
    if (d->isRunning()) {
        NNRT_LOGW_PRINT("Fail to modify the execution in running.");
        return NNA_ERROR_CODE(INCOMPLETE);
    }
    if (index >= d->inputs_.size()) {
        NNRT_LOGW_PRINT("Invalid input index(%u), max input size is %u.",
                index, d->inputs_.size());
        return NNA_ERROR_CODE(BAD_DATA);
    }
    Model* model = d->compilation_->getModel();
    uint32_t operand_index = model->inputIndex(index);
    if (operand_index < 0 ) {
        NNRT_LOGE(tag)<< "Fatal Error : invalid operand index < 0";
        return NNA_ERROR_CODE(BAD_DATA);
    }

    if (!buffer || 0 == length) {
        NNRT_LOGD_PRINT("Set idx=%d as novalue", index);

        model->operand(operand_index)->setNull();
        d->inputs_[index]->setNoValue();
    } else {
        model->operand(operand_index)->clearNull();
        if (operand_type) {
            model->updateOperand(operand_index, operand_type);
        }

        d->inputs_[index]->state = ExecutionIO::BUFFER;
        d->inputs_[index]->weak_mem_ref = model->add_memory_reference(buffer, length, true);
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
    std::unique_lock<std::mutex> lk(d->mutex_);
    if (d->isRunning()) {
        NNRT_LOGW_PRINT("Fail to modify the execution in running.");
        return NNA_ERROR_CODE(INCOMPLETE);
    }
    if (index >= d->inputs_.size()) {
        NNRT_LOGW_PRINT("Invalid input index(%u), max input size is %u.",
                index, d->inputs_.size());
        return NNA_ERROR_CODE(BAD_DATA);
    }

    Model* model = d->compilation_->getModel();
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

    d->inputs_[index]->state = ExecutionIO::BUFFER;
    d->inputs_[index]->weak_mem_ref = model->add_memory_reference(memory, offset, length);
    return NNA_ERROR_CODE(NO_ERROR);
}

int Execution::setOutput(uint32_t index, const op::OperandPtr operand_type,
        void* buffer, size_t length)
{
    if (!buffer) {
        NNRT_LOGW_PRINT("Pass nullptr to buffer.");
        return NNA_ERROR_CODE(UNEXPECTED_NULL);
    }
    std::unique_lock<std::mutex> lk(d->mutex_);
    if (d->isRunning()) {
        NNRT_LOGW_PRINT("Fail to modify the execution in running.");
        return NNA_ERROR_CODE(INCOMPLETE);
    }
    if (index >= d->outputs_.size()) {
        NNRT_LOGW_PRINT("Invalid output index(%u), max output size is %u.",
                index, d->outputs_.size());
        return NNA_ERROR_CODE(BAD_DATA);
    }
    if (!buffer) {
        d->outputs_[index]->setNoValue();
    } else {
        d->outputs_[index]->state = ExecutionIO::BUFFER;
        Model* model = d->compilation_->getModel();
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
        d->outputs_[index]->weak_mem_ref = model->add_memory_reference(buffer, length, true);
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
    std::unique_lock<std::mutex> lk(d->mutex_);
    if (d->isRunning()) {
        NNRT_LOGW_PRINT("Fail to modify the execution in running.");
        return NNA_ERROR_CODE(INCOMPLETE);
    }
    if (index >= d->outputs_.size()) {
        NNRT_LOGW_PRINT("Invalid output index(%u), max output size is %u.",
                index, d->outputs_.size());
        return NNA_ERROR_CODE(BAD_DATA);
    }

    Model* model = d->compilation_->getModel();
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
    d->outputs_[index]->state = ExecutionIO::BUFFER;
    d->outputs_[index]->weak_mem_ref = model->add_memory_reference(memory, offset, length);
    return NNA_ERROR_CODE(NO_ERROR);
}

int Execution::getOutputOperandRank(uint32_t index, uint32_t* rank)
{
    if (!rank) {
        return NNA_ERROR_CODE(OUTPUT_INSUFFICIENT_SIZE);
    }
    Model* model = d->compilation_->getModel();
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
    Model* model = d->compilation_->getModel();
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
