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

#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <boost/log/trivial.hpp>
#include "TNpuWorkloads.hpp"

namespace armnn {

template <typename armnn::DataType... DataTypes>
struct Base<GatherQueueDescriptor, DataTypes...> {
    using type = BaseWorkload<GatherQueueDescriptor>;
};

template <typename armnn::DataType... DataTypes>
class NpuGatherWorkload : public TNpuWorkload<GatherQueueDescriptor, DataTypes...> {
   public:
    explicit NpuGatherWorkload(const GatherQueueDescriptor& descriptor, const WorkloadInfo& info)
        : TNpuWorkload<GatherQueueDescriptor, DataTypes...>(descriptor, info) {
        std::vector<uint32_t> inputIds = this->AddOperandWithTensorHandle(descriptor.m_Inputs);

        std::vector<uint32_t> outputIds = this->AddOperandWithTensorHandle(descriptor.m_Outputs);

        this->AddOperation(nnrt::OperationType::GATHER,
                           inputIds.size(),
                           inputIds.data(),
                           outputIds.size(),
                           outputIds.data());
    }
};
using NpuGatherWorkloadFp32 = NpuGatherWorkload<armnn::DataType::Float32>;
using NpuGatherWorkloadFp16 = NpuGatherWorkload<armnn::DataType::Float16>;
using NpuGatherWorkloadU8 = NpuGatherWorkload<armnn::DataType::QAsymmU8>;
}  // namespace armnn
