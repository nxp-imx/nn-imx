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
class NpuPadWorkload : public TNpuWorkload<PadQueueDescriptor, DataTypes...> {
   public:
    explicit NpuPadWorkload(const PadQueueDescriptor& descriptor,
                                  const WorkloadInfo& info)
        : TNpuWorkload<PadQueueDescriptor, DataTypes...>(descriptor, info),
          m_PadList(descriptor.m_Parameters.m_PadList),
          m_PadValue(descriptor.m_Parameters.m_PadValue) {
        std::vector<uint32_t> inputIds = this->AddOperandWithTensorHandle(
                descriptor.m_Inputs);

        std::vector<uint32_t> padDimShape = {(uint32_t)m_PadList.size(), 2};
        m_PadDims.reserve(m_PadList.size() * 2);
        for (uint32_t i = 0; i < m_PadList.size(); i ++) {
            m_PadDims[i * 2] = m_PadList[i].first;
            m_PadDims[i * 2 + 1] = m_PadList[i].second;
        }
        inputIds.push_back(this->AddOperandAndSetValue(
                    padDimShape, DataType::Signed32, m_PadDims.data()));

        inputIds.push_back(this->AddOperandAndSetValue(m_PadValue));

        std::vector<uint32_t> outputIds = this->AddOperandWithTensorHandle(
                descriptor.m_Outputs);

        this->AddOperation(nnrt::OperationType::PAD,
                inputIds.size(), inputIds.data(), outputIds.size(), outputIds.data());
    }

   private:
    std::vector<std::pair<uint32_t, uint32_t>> m_PadList;
    float m_PadValue;
    std::vector<int32_t> m_PadDims;
};
using NpuPadFloat32Workload = NpuPadWorkload<armnn::DataType::Float32>;
using NpuPadFloat16Workload = NpuPadWorkload<armnn::DataType::Float16>;
using NpuPadUint8Workload = NpuPadWorkload<armnn::DataType::QAsymmU8>;
}  // namespace armnn
