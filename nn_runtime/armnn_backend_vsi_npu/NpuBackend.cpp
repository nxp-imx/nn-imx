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

#include "NpuBackend.hpp"
#include "NpuBackendId.hpp"
#include "NpuWorkloadFactory.hpp"
#include "NpuLayerSupport.hpp"
#include "NpuTensorAllocator.hpp"
#include "NpuBackendContext.hpp"

#include <armnn/BackendRegistry.hpp>

#include <armnn/backends/IBackendContext.hpp>
#include <armnn/backends/IMemoryManager.hpp>

#include <Optimizer.hpp>

#include <armnn/utility/IgnoreUnused.hpp>

namespace armnn
{

const BackendId& NPUBackend::GetIdStatic()
{
    static const BackendId s_Id{NpuBackendId()};
    return s_Id;
}

IBackendInternal::IWorkloadFactoryPtr NPUBackend::CreateWorkloadFactory(
    const IBackendInternal::IMemoryManagerSharedPtr& memoryManager) const
{
    IgnoreUnused(memoryManager);
    return std::make_unique<NpuWorkloadFactory>();
}

IBackendInternal::IBackendContextPtr NPUBackend::CreateBackendContext(const IRuntime::CreationOptions& options) const
{
    return IBackendContextPtr{ new NpuBackendContext{options} };
}

IBackendInternal::IMemoryManagerUniquePtr NPUBackend::CreateMemoryManager() const
{
    return std::make_unique<NPUTensorAllocator>();
}

IBackendInternal::Optimizations NPUBackend::GetOptimizations() const
{
    return Optimizations{};
}

IBackendInternal::ILayerSupportSharedPtr NPUBackend::GetLayerSupport() const
{
    static ILayerSupportSharedPtr layerSupport{new NpuLayerSupport};
    return layerSupport;
}

} // namespace armnn
