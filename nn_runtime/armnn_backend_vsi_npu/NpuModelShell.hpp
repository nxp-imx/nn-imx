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

#pragma once

#include "nnrt/compilation.hpp"
#include "nnrt/event.hpp"
#include "nnrt/execution.hpp"
#include "nnrt/model.hpp"

#include <memory>
#include <unordered_map>

namespace armnn {
class NpuTensorHandler;
using NpuTensorHandlerPtrList = std::vector<NpuTensorHandler*>;
}

namespace adaption {

using InOutTensorHandles = std::pair<armnn::NpuTensorHandlerPtrList, armnn::NpuTensorHandlerPtrList>;
// ModelStack MUST have a unique key value due to branch in the model
using ModelStack = std::unordered_map<nnrt::ModelPtr, InOutTensorHandles>;

using FinalModel = std::pair<nnrt::ModelPtr, InOutTensorHandles>;
using FinalModelPtr = std::unique_ptr<FinalModel>;

namespace utils {
FinalModelPtr MergeModels(const ModelStack&);
}
}

namespace armnn {
class ModelShell {
   public:
    explicit ModelShell(adaption::FinalModelPtr&&);
    ~ModelShell();

    void Execute();

    adaption::FinalModel* GetFinalModelPtr() {return m_NativeModel.get();}

   private:
    adaption::FinalModelPtr m_NativeModel;
    nnrt::CompilerUniquePtr m_Compiler;  //!< model compiler
    nnrt::ExecUniquePtr m_ExecutionPtr;
};

using ModelShellPtr = std::shared_ptr<ModelShell>;
}
