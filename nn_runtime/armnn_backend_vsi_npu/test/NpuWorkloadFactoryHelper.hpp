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

#include <backendsCommon/test/WorkloadFactoryHelper.hpp>

#include "NpuBackend.hpp"
#include "NpuWorkloadFactory.hpp"
#include "NpuTensorHandleFactory.hpp"

#include <boost/core/ignore_unused.hpp>

using namespace boost;

namespace
{

template<>
struct WorkloadFactoryHelper<armnn::NpuWorkloadFactory>
{
    static armnn::IBackendInternal::IMemoryManagerSharedPtr GetMemoryManager()
    {
        armnn::NPUBackend backend;
        return backend.CreateMemoryManager();
    }

    static armnn::NpuWorkloadFactory GetFactory(
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
    {
        ignore_unused(memoryManager);
        return armnn::NpuWorkloadFactory();
    }

    static armnn::NpuTensorHandleFactory GetTensorHandleFactory(
            const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager = nullptr)
    {
        return armnn::NpuTensorHandleFactory(std::static_pointer_cast<armnn::NpuMemoryManager>(memoryManager));
    }
};

using NpuWorkloadFactoryHelper = WorkloadFactoryHelper<armnn::NpuWorkloadFactory>;

} // anonymous namespace
