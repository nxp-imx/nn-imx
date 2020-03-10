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
#include "error.hpp"
#include "execution.hpp"
#include "execution_task.hpp"

namespace nnrt
{

TaskQueue& get_global_task_queue() {
    static TaskQueue taskQueue;
    return taskQueue;
}

static void work_process()
{
    for(;;) {
        TaskPtr task = get_global_task_queue().dequeue();
        if (task) {
            task->execute();
        } else {
            break;
        }
    }
}

int GraphTask::executeNormal()
{
    int status = fillInput();
    if (status == NNA_ERROR_CODE(NO_ERROR)) {
        status = prepared_model_->execute();
        if (status == NNA_ERROR_CODE(NO_ERROR)) {
            status = fillOutput();
        }
    }
    return status;
}

int GraphTask::executeWithNotify()
{
    event_->lock();
    int status = fillInput();
    event_->unlock();
    if (status == NNA_ERROR_CODE(NO_ERROR)) {
        status = prepared_model_->execute();
        if (status == NNA_ERROR_CODE(NO_ERROR)) {
            event_->lock();
            status = fillOutput();
            event_->unlock();
        }
    }
    int code = Event::ERROR_OCCURED;
    if (NNA_ERROR_CODE(NO_ERROR) == status) {
        code = Event::COMPLETED;
    }
    event_->notify(code);
    return status;
}

int GraphTask::fillInput()
{
    int status = NNA_ERROR_CODE(NO_ERROR);
    if (event_ && event_->is_canceled()) {
        return NNA_ERROR_CODE(NO_ERROR);
    }
    for (size_t i = 0; i < inputs_.size(); ++ i) {
        auto port = inputs_[i];

        if (ExecutionIO::HAS_NO_VALUE == port->state) {
            NNRT_LOGD_PRINT("skip No_Value ExecutionIO[%d]", i);
            continue;
        }

        if (ExecutionIO::UNSPECIFIED == port->state) {
            status = NNA_ERROR_CODE(OP_FAILED);
            break;
        }

        auto ref_ptr = inputs_[i]->weak_mem_ref.lock();
        if (ref_ptr) {
            status = prepared_model_->setInput(i,
                    ref_ptr->address_, ref_ptr->len_);
        }

        if (NNA_ERROR_CODE(NO_ERROR) != status) {
            NNRT_LOGE_PRINT("Fill input error %d", status);
            break;
        }
    }
    return status;
}

int GraphTask::fillOutput()
{
    int status = NNA_ERROR_CODE(NO_ERROR);
    if (event_ && event_->is_canceled()) {
        return NNA_ERROR_CODE(NO_ERROR);
    }
    for (size_t i = 0; i < outputs_.size(); ++ i) {
        auto port = outputs_[i];
        if (ExecutionIO::UNSPECIFIED == port->state) {
            NNRT_LOGW_PRINT("Output %u is unspecified.");
            status = NNA_ERROR_CODE(OP_FAILED);
            break;
        } else if (ExecutionIO::HAS_NO_VALUE == port->state) {
            continue;
        }

        auto ref_ptr = outputs_[i]->weak_mem_ref.lock();
        if (ref_ptr) {
            // TODO: output memory should not be a read-only buffer in memory-pool
            status = prepared_model_->getOutput(i,
                    const_cast<void*>(ref_ptr->address_), ref_ptr->len_);
        }
        else {
            NNRT_LOGE_PRINT("Error at getOutput");
            status = NNA_ERROR_CODE(OP_FAILED);
            break;
        }

        if (NNA_ERROR_CODE(NO_ERROR) != status) {
            NNRT_LOGD_PRINT("Fill output error %d", status);
            break;
        }
    }
    return status;
}

void TaskQueue::wakeWorkThread()
{
    if (!work_thread_awaked_) {
        work_thread_awaked_ = true;
        std::thread(work_process).detach();
    }
}

int TaskQueue::enqueue(TaskPtr task)
{
    queue_mutex_.lock();
    queue_.push(task);
    wakeWorkThread();
    queue_mutex_.unlock();
    return NNA_ERROR_CODE(NO_ERROR);
}

TaskPtr TaskQueue::dequeue()
{
    std::unique_lock<std::mutex> lk(queue_mutex_);
    TaskPtr task = nullptr;
    if (queue_.size() == 0) {
        // If there is no task to run, exit work thread
        exitWorkThread();
    } else {
        task = queue_.front();
        queue_.pop();
    }
    return task;
}

}
