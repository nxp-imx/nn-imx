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
#include <cassert>
#include "vsi_nn_pub.h"
#include "model.hpp"
#include "compilation.hpp"
#include "error.hpp"
#include "prepared_model.hpp"

#include "model_transform/nnapi_interpreter.hpp"

namespace nnrt
{
/**
 * Thread local context.
 * Driver only support one context in a thread.
 */
thread_local SharedContextPtr thread_local_context;

struct ContextDeleter {
    void operator()(vsi_nn_context_t ctx) {
        NNRT_LOGD_PRINT("Release context");
        vsi_nn_ReleaseContext(&ctx);
    }
};

Compilation::Compilation(Model * model)
    : model_(model)
    , interpreter_(new NnApiInterpreter())
    , prepared_model_cache_size_(1)
{
    if (!thread_local_context) {
        thread_local_context.reset(vsi_nn_CreateContext(), ContextDeleter());
    }
    context_ = thread_local_context;
}

Compilation::~Compilation()
{
    if (interpreter_) {
        delete interpreter_;
    }
    prepared_models_.clear();
    context_.reset();
    if (thread_local_context.use_count() == 1) {
        thread_local_context.reset();
    }
}

int Compilation::run()
{
    int err = NNA_ERROR_CODE(NO_ERROR);
    return err;
}

void Compilation::cachePreparedModel(PreparedModelPtr& prepared_model)
{
    if (prepared_models_.size() + 1 > prepared_model_cache_size_) {
        std::vector<std::string> keys_to_remove;
        for (auto it = prepared_models_.begin(); it != prepared_models_.end(); it ++) {
            if (it->second.use_count() == 1) {
                keys_to_remove.push_back(it->first);
            }
        }
        for (auto& key : keys_to_remove) {
            prepared_models_.erase(key);
        }
    }
    prepared_models_[prepared_model->signature()] = prepared_model;
}

PreparedModelPtr Compilation::prepareModel(int* err_ptr,
                                std::vector<ExecutionIOPtr> &inputs)
{
    int err = NNA_ERROR_CODE(NO_ERROR);
    std::unique_lock<std::mutex> lk(cache_mutex_);
    std::string model_signature = model_->generateSignature();
    auto it = prepared_models_.find(model_signature);
    PreparedModelPtr prepared_model = nullptr;
    if (it == prepared_models_.end()) {
        // We do not support compile variance shapes for same input,
        // assert if we meet that case.
        assert(prepared_models_.size() == 0);
        prepared_model = std::make_shared<PreparedModel>(model_,
                context_, inputs, getInterpreter());
        err = prepared_model->prepare();
        if (err == NNA_ERROR_CODE(NO_ERROR)) {
            cachePreparedModel(prepared_model);
        } else {
            prepared_model.reset();
        }
    } else {
        prepared_model = it->second;
    }
    if (err_ptr) {
        *err_ptr = err;
    }
    return prepared_model;
}

}
