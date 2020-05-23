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
class NpuSpaceToDepthWorkload : public TNpuWorkload<SpaceToDepthQueueDescriptor, DataTypes...> {
   public:
    explicit NpuSpaceToDepthWorkload(const SpaceToDepthQueueDescriptor& descriptor,
                                     const WorkloadInfo& info)
        : TNpuWorkload<SpaceToDepthQueueDescriptor, DataTypes...>(descriptor, info),
          m_BlockSize(descriptor.m_Parameters.m_BlockSize),
          m_DataLayout(descriptor.m_Parameters.m_DataLayout) {
        std::vector<uint32_t> inputIds = this->AddOperandWithTensorHandle(descriptor.m_Inputs);
        inputIds.push_back(this->AddOperandAndSetValue(m_BlockSize));
        // Add layout operand
        int32_t layoutCode = m_DataLayout == armnn::DataLayout::NCHW
                                 ? int32_t(nnrt::DataLayout::NCHW)
                                 : int32_t(nnrt::DataLayout::NHWC);
        inputIds.push_back(this->AddOperandAndSetValue(layoutCode));

        std::vector<uint32_t> outputIds = this->AddOperandWithTensorHandle(descriptor.m_Outputs);

        this->AddOperation(nnrt::OperationType::SPACE_TO_DEPTH,
                           inputIds.size(),
                           inputIds.data(),
                           outputIds.size(),
                           outputIds.data());
    }

   private:
    uint32_t m_BlockSize;
    DataLayout m_DataLayout;
};
using NpuSpaceToDepthFloat32Workload = NpuSpaceToDepthWorkload<armnn::DataType::Float32>;
using NpuSpaceToDepthFloat16Workload = NpuSpaceToDepthWorkload<armnn::DataType::Float16>;
using NpuSpaceToDepthUint8Workload = NpuSpaceToDepthWorkload<armnn::DataType::QAsymmU8>;
}  // namespace armnn
