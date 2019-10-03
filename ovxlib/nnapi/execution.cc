#include <thread>
#include <assert.h>
#include "vsi_nn_pub.h"
#include "execution.h"
#include "compilation.h"
#include "prepared_model.h"
#include "event.h"
#include "error.h"

#include "memory_pool.h"

namespace ovxlib
{
struct ExecutionIO {
    enum {
        UNSPECIFIED,
        BUFFER,
        POINTER,
        HAS_NO_VALUE,   //!< this is for optional inputs
    } state = UNSPECIFIED;

    ExecutionIO(const Operand* operand = nullptr) {
        if (!operand) {
            assert(false);
            return;
        }
        mem_ref = operand->mem_ref;
        dimensions = operand->dimensions;
    }

    void setNoValue() {
        state = HAS_NO_VALUE;
    }

    mem_pool::shared_ref mem_ref;

    std::vector<uint32_t> dimensions;
};

static void asyncCompute(Execution* execution,
    Event* event)
{
    int status = AERROR_CODE(OP_FAILED);
    PreparedModelPtr prepared_model = execution->getPreparedModel();
    if (prepared_model) {
        if (execution->fillInput(prepared_model) == AERROR_CODE(NO_ERROR)) {
            status = prepared_model->execute();
            if (status == AERROR_CODE(NO_ERROR)) {
                status = execution->fillOutput(prepared_model);
            }
        }
        execution->getCompilation()->detachPreparedModel(prepared_model);
    }
    execution->complete(status, true);
}

Execution::Execution(Compilation* compilation)
    : compilation_(compilation)
    , running_(false)
    , ask_for_quit_(false)
{
    Model* model = compilation->getModel();
    for (auto& index : model->inputIndexes()) {
        Operand* operand = model->operand(index);
        ExecutionIO* io = new ExecutionIO(operand);
        inputs_.push_back(io);
    }
    for (auto& index : model->outputIndexes()) {
        Operand* operand = model->operand(index);
        ExecutionIO * io = new ExecutionIO(operand);
        outputs_.push_back(io);
    }
}

Execution::~Execution(){
    for (size_t i = 0; i < inputs_.size(); ++ i) {
        delete inputs_[i];
    }
    for (size_t i = 0; i < outputs_.size(); ++ i) {
        delete outputs_[i];
    }
    //delete prepared_model_;
    //if (thread_local_context.use_count() == 1) thread_local_context.reset();
}

int Execution::fillInput(PreparedModelPtr prepared_model)
{
    int status = AERROR_CODE(NO_ERROR);
    for (size_t i = 0; i < inputs_.size(); ++ i) {
        auto port = inputs_[i];

        if (ExecutionIO::HAS_NO_VALUE == port->state) {
            VSILOGD("skip No_Value ExecutionIO[%d]", i);
            continue;
        }

        if (ExecutionIO::UNSPECIFIED == port->state) {
            return AERROR_CODE(OP_FAILED);
        }

        if (port->mem_ref) {
            status = prepared_model->setInput(i, inputs_[i]->mem_ref->address_, inputs_[i]->mem_ref->len_);
        }

        if (AERROR_CODE(NO_ERROR) != status) {
            VSILOGE("Fill input error %d", status);
            break;
        }
    }
    return status;
}

int Execution::fillOutput(PreparedModelPtr prepared_model)
{
    int status = AERROR_CODE(NO_ERROR);
    for (size_t i = 0; i < outputs_.size(); ++ i) {
        auto port = outputs_[i];
        if (ExecutionIO::UNSPECIFIED == port->state) {
            VSILOGW("Output %u is unspecified.");
            return AERROR_CODE(OP_FAILED);
        } else if (ExecutionIO::HAS_NO_VALUE == port->state) {
            continue;
        }

        if (port->mem_ref) {
            // TODO: output memory should not be a read-only buffer in memory-pool
            status = prepared_model->getOutput(i, const_cast<void*>(port->mem_ref->address_), port->mem_ref->len_);
        }
        else {
            VSILOGE("Error at getOutput");
            return AERROR_CODE(OP_FAILED);
        }

        if (AERROR_CODE(NO_ERROR) != status) {
            VSILOGD("Fill output error %d", status);
            break;
        }
    }
    return status;
}

void Execution::quit()
{
    ask_for_quit_ = true;
    notify(Event::CANCELED);
}

int Execution::startCompute(Event* event)
{
    if (!compilation_) {
        return AERROR_CODE(UNEXPECTED_NULL);
    }

    std::unique_lock<std::mutex> lk(mutex_);
    if (isRunning()) {
        VSILOGW("Execution is already running.");
        return AERROR_CODE(INCOMPLETE);
    }
    if (event_) {
        VSILOGE("Event is not clean.");
        return AERROR_CODE(OP_FAILED);
    }
    int err = compilation_->prepareModel();
    if (err == AERROR_CODE(NO_ERROR)) {
        running_ = true;
        event_ = event;
        event_->notify(Event::IN_PROCESS);
        std::thread(asyncCompute, this, event).detach();
        //asyncCompute( this, event);
    }
    return err;
}

int Execution::compute()
{
    if (!compilation_) {
        return AERROR_CODE(UNEXPECTED_NULL);
    }
    if (isRunning()) {
        VSILOGW("Execution is already running.");
        return AERROR_CODE(INCOMPLETE);
    }
    int err = compilation_->prepareModel();
    if (err == AERROR_CODE(NO_ERROR)) {
        running_ = true;
        int status = AERROR_CODE(OP_FAILED);
        PreparedModelPtr prepared_model = getPreparedModel();
        if (prepared_model) {
            if (fillInput(prepared_model) == AERROR_CODE(NO_ERROR)) {
                status = prepared_model->execute();
                if (status == AERROR_CODE(NO_ERROR)) {
                    status = fillOutput(prepared_model);
                }
            }
            getCompilation()->detachPreparedModel(prepared_model);
        }
        complete(status, false);
    }
    // TODO:
    running_ = false;
    return AERROR_CODE(NO_ERROR);
}

int Execution::setInput(uint32_t index, const Operand* operand_type,
        const void* buffer, size_t length)
{
    std::unique_lock<std::mutex> lk(mutex_);
    if (isRunning()) {
        VSILOGW("Fail to modify the execution in running.");
        return AERROR_CODE(INCOMPLETE);
    }
    if (index >= inputs_.size()) {
        VSILOGW("Invalid input index(%u), max input size is %u.",
                index, inputs_.size());
        return AERROR_CODE(BAD_DATA);
    }
    Model* model = compilation_->getModel();
    uint32_t operand_index = model->inputIndex(index);
    assert(operand_index >= 0);
    if (!buffer || 0 == length) {
        VSILOGD("Set idx=%d as novalue", index);

        model->operand(operand_index)->setNull();
        inputs_[index]->setNoValue();
    } else {
        model->operand(operand_index)->clearNull();
        if (operand_type) {
            model->updateOperand(operand_index, operand_type);
        }
        inputs_[index]->state = ExecutionIO::BUFFER;
        inputs_[index]->mem_ref = mem_pool::global_memory_pool().add_reference(buffer, length);
    }
    return AERROR_CODE(NO_ERROR);
}

int Execution::setInputFromMemory(uint32_t index, const Operand* operand_type,
        const Memory* memory, size_t offset, size_t length)
{
    if (!memory) {
        VSILOGW("Pass nullptr to memory.");
        return AERROR_CODE(UNEXPECTED_NULL);
    }
    std::unique_lock<std::mutex> lk(mutex_);
    if (isRunning()) {
        VSILOGW("Fail to modify the execution in running.");
        return AERROR_CODE(INCOMPLETE);
    }
    if (index >= inputs_.size()) {
        VSILOGW("Invalid input index(%u), max input size is %u.",
                index, inputs_.size());
        return AERROR_CODE(BAD_DATA);
    }

    inputs_[index]->state = ExecutionIO::BUFFER;
    inputs_[index]->mem_ref = mem_pool::global_memory_pool().add_reference(memory, offset, length);
    return AERROR_CODE(NO_ERROR);
}

int Execution::setOutput(uint32_t index, const Operand* operand_type,
        void* buffer, size_t length)
{
    if (!buffer) {
        VSILOGW("Pass nullptr to buffer.");
        return AERROR_CODE(UNEXPECTED_NULL);
    }
    std::unique_lock<std::mutex> lk(mutex_);
    if (isRunning()) {
        VSILOGW("Fail to modify the execution in running.");
        return AERROR_CODE(INCOMPLETE);
    }
    if (index >= outputs_.size()) {
        VSILOGW("Invalid output index(%u), max output size is %u.",
                index, outputs_.size());
        return AERROR_CODE(BAD_DATA);
    }
    if (!buffer) {
        outputs_[index]->setNoValue();
    } else {
        outputs_[index]->state = ExecutionIO::BUFFER;
        Model* model = compilation_->getModel();
        uint32_t operand_index = model->outputIndex(index);
        model->operand(operand_index)->clearNull();
        if (operand_type) {
            model->updateOperand(operand_index, operand_type);
        }
        // !!! NEVER allocate inner buffer for output buffer
        outputs_[index]->mem_ref = mem_pool::global_memory_pool().add_reference(buffer, length, true);
    }
    return AERROR_CODE(NO_ERROR);
}

int Execution::setOutputFromMemory(uint32_t index, const Operand* operand_type,
        const Memory* memory, size_t offset, size_t length)
{
    if (!memory) {
        VSILOGW("Pass nullptr to memory.");
        return AERROR_CODE(UNEXPECTED_NULL);
    }
    std::unique_lock<std::mutex> lk(mutex_);
    if (isRunning()) {
        VSILOGW("Fail to modify the execution in running.");
        return AERROR_CODE(INCOMPLETE);
    }
    if (index >= outputs_.size()) {
        VSILOGW("Invalid output index(%u), max output size is %u.",
                index, outputs_.size());
        return AERROR_CODE(BAD_DATA);
    }

    outputs_[index]->state = ExecutionIO::BUFFER;
    outputs_[index]->mem_ref = mem_pool::global_memory_pool().add_reference(memory, offset, length);
    return AERROR_CODE(NO_ERROR);
}

PreparedModelPtr Execution::getPreparedModel()
{
    if (!compilation_) {
        return nullptr;
    }
    return compilation_->attachPreparedModel();
}

void Execution::complete(int status, bool notify_event)
{
    int code = Event::ERROR_OCCURED;
    if (AERROR_CODE(NO_ERROR) == status) {
        code = Event::COMPLETED;
    }
    if (ask_for_quit_) {
        delete this;
    } else {
        running_ = false;
        if (notify_event) {
            notify(code);
        }
    }
}

void Execution::notify(int code) {
    std::unique_lock<std::mutex> lk(mutex_);
    Event* event = event_;
    event_ = nullptr;
    event->notify(code);
}

}
