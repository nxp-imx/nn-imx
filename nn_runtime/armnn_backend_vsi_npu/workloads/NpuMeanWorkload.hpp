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
#include <iostream>
#include "TNpuWorkloads.hpp"

namespace armnn {
template <typename armnn::DataType... DataTypes>
class NpuMeanWorkload : public TNpuWorkload<MeanQueueDescriptor, DataTypes...> {
   public:
    explicit NpuMeanWorkload(const MeanQueueDescriptor& descriptor,
                                  const WorkloadInfo& info)
        : TNpuWorkload<MeanQueueDescriptor, DataTypes...>(descriptor, info),
          m_Axes(descriptor.m_Parameters.m_Axis),
          m_KeepDims((int32_t)descriptor.m_Parameters.m_KeepDims) {
        std::vector<uint32_t> inputIds = this->AddOperandWithTensorHandle(
                descriptor.m_Inputs);

        std::vector<uint32_t> axes_shape = {(uint32_t)m_Axes.size()};
        inputIds.push_back(this->AddOperandAndSetValue(
                    axes_shape, DataType::Signed32, m_Axes.data()));
        inputIds.push_back(this->AddOperandAndSetValue(m_KeepDims));

        std::vector<uint32_t> outputIds = this->AddOperandWithTensorHandle(
                descriptor.m_Outputs);

        this->AddOperation(nnrt::OperationType::MEAN,
                inputIds.size(), inputIds.data(), outputIds.size(), outputIds.data());
    }

   private:
    std::vector<uint32_t> m_Axes;
    int32_t m_KeepDims;

};
using NpuMeanFloat32Workload = NpuMeanWorkload<armnn::DataType::Float32>;
using NpuMeanFloat16Workload = NpuMeanWorkload<armnn::DataType::Float16>;
using NpuMeanUint8Workload = NpuMeanWorkload<armnn::DataType::QAsymmU8>;
}  // namespace armnn
