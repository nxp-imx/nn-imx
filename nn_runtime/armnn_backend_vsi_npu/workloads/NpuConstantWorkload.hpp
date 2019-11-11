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
class NpuConstantWorkload : public TNpuWorkload<ConstantQueueDescriptor, DataTypes...> {
   public:
    using base_type = TNpuWorkload<ConstantQueueDescriptor, DataTypes...>;
    explicit NpuConstantWorkload(const ConstantQueueDescriptor& descriptor,
                                 const WorkloadInfo& info)
        : TNpuWorkload<ConstantQueueDescriptor, DataTypes...>(descriptor, info),
          m_LayerOutput(std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_LayerOutput))) {
        descriptor.m_Outputs[0]->Import(m_LayerOutput->GetTensor<void>(), MemorySource::Malloc);
    }

    virtual void Execute() const {}

   private:
    std::unique_ptr<ScopedCpuTensorHandle> m_LayerOutput;
};
using NpuConstantFloat32Workload = NpuConstantWorkload<armnn::DataType::Float32>;
using NpuConstantFloat16Workload = NpuConstantWorkload<armnn::DataType::Float16>;
using NpuConstantUint8Workload = NpuConstantWorkload<armnn::DataType::QuantisedAsymm8>;
}  // namespace armnn
