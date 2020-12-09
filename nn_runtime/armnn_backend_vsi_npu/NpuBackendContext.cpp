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

#include "NpuBackendContext.hpp"

#include <iostream>

#include <boost/core/ignore_unused.hpp>

using namespace boost;

namespace armnn
{

NpuBackendContext::NpuBackendContext(const IRuntime::CreationOptions& options)
    : IBackendContext(options)
    // , m_ClContextControlWrapper(
    //     std::make_unique<ClContextControlWrapper>(options.m_GpuAccTunedParameters.get(),
    //                                               options.m_EnableGpuProfiling))
{
}

bool NpuBackendContext::BeforeLoadNetwork(NetworkId)
{
    return true;
}

bool NpuBackendContext::AfterLoadNetwork(NetworkId networkId)
{
    // {
    //     std::lock_guard<std::mutex> lockGuard(m_Mutex);
    //     m_NetworkIds.insert(networkId);
    // }
    ignore_unused(networkId);
    return true;
}

bool NpuBackendContext::BeforeUnloadNetwork(NetworkId networkId)
{
    // return m_ClContextControlWrapper->Sync();
    ignore_unused(networkId);
    return true;
}

bool NpuBackendContext::AfterUnloadNetwork(NetworkId networkId)
{
    // bool clearCache = false;
    // {
    //     std::lock_guard<std::mutex> lockGuard(m_Mutex);
    //     m_NetworkIds.erase(networkId);
    //     clearCache = m_NetworkIds.empty();
    // }

    // if (clearCache)
    // {
    //     m_ClContextControlWrapper->ClearClCache();
    // }
    ignore_unused(networkId);
    return true;
}

NpuBackendContext::~NpuBackendContext()
{
}

} // namespace armnn
