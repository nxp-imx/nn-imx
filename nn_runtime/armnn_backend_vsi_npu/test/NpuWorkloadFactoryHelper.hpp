//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/test/WorkloadFactoryHelper.hpp>

#include "NpuBackend.hpp"
#include "NpuWorkloadFactory.hpp"

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
        return armnn::NpuWorkloadFactory();
    }
};

using NpuWorkloadFactoryHelper = WorkloadFactoryHelper<armnn::NpuWorkloadFactory>;

} // anonymous namespace
