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
class NpuConcatWorkload : public TNpuWorkload<ConcatQueueDescriptor, DataTypes...> {
   public:
    explicit NpuConcatWorkload(const ConcatQueueDescriptor& descriptor,
                                  const WorkloadInfo& info)
        : TNpuWorkload<ConcatQueueDescriptor, DataTypes...>(descriptor, info) {
        std::vector<uint32_t> inputIds = this->AddOperandWithTensorHandle(
                descriptor.m_Inputs);
        inputIds.push_back(this->AddOperandAndSetValue(descriptor.m_Parameters.GetConcatAxis()));
        std::vector<uint32_t> outputIds = this->AddOperandWithTensorHandle(
                descriptor.m_Outputs);

        this->AddOperation(nnrt::OperationType::CONCATENATION,
                inputIds.size(), inputIds.data(), outputIds.size(), outputIds.data());
    }

};
using NpuConcatFloat32Workload = NpuConcatWorkload<armnn::DataType::Float32>;
using NpuConcatFloat16Workload = NpuConcatWorkload<armnn::DataType::Float16>;
using NpuConcatUint8Workload = NpuConcatWorkload<armnn::DataType::QuantisedAsymm8>;
}  // namespace armnn
