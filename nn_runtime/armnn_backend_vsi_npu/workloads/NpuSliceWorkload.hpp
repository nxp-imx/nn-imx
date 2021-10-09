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

#include <backendsCommon/TensorHandle.hpp>
#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include "TNpuWorkloads.hpp"

namespace armnn {
template <typename armnn::DataType... DataTypes>
class NpuSliceWorkload : public TNpuWorkload<SliceQueueDescriptor, DataTypes...> {
   public:
    using base_type = TNpuWorkload<SliceQueueDescriptor, DataTypes...>;
    explicit NpuSliceWorkload(const SliceQueueDescriptor& descriptor, const WorkloadInfo& info)
        : TNpuWorkload<SliceQueueDescriptor, DataTypes...>(descriptor, info),
          m_Begin(descriptor.m_Parameters.m_Begin),
          m_Size(descriptor.m_Parameters.m_Size) {
        // Add inputs operand
        std::vector<uint32_t> inOperandIds;
        NpuTensorHandler* inputTensorHandle =
            dynamic_cast<NpuTensorHandler*>(descriptor.m_Inputs[0]);
        if (inputTensorHandle) {
            uint32_t inputOperandId = this->AddOperandAndSetValue(
                inputTensorHandle->GetTensorInfo(), inputTensorHandle->GetShape(), nullptr);
            inOperandIds.push_back(inputOperandId);
        }

        std::vector<uint32_t> beginShape = {(uint32_t)m_Begin.size()};
        inOperandIds.push_back(this->AddOperandAndSetValue(
                    beginShape, DataType::Signed32, m_Begin.data()));

        std::vector<uint32_t> sizeShape = {(uint32_t)m_Size.size()};
        inOperandIds.push_back(this->AddOperandAndSetValue(
                    sizeShape, DataType::Signed32, m_Size.data()));

        std::vector<uint32_t> outOperandIds;
        NpuTensorHandler* outputTensorHandle =
            dynamic_cast<NpuTensorHandler*>(descriptor.m_Outputs[0]);
        if (outputTensorHandle) {
            uint32_t outputTensorId = this->AddOperandAndSetValue(
                outputTensorHandle->GetTensorInfo(), outputTensorHandle->GetShape(), nullptr);
            outOperandIds.push_back(outputTensorId);
        }

        this->AddOperation(nnrt::OperationType::SLICE,
                           inOperandIds.size(),
                           inOperandIds.data(),
                           outOperandIds.size(),
                           outOperandIds.data());
    }

   private:
    // Beginning indices of the slice in each dimension.
    std::vector<unsigned int> m_Begin;

    // Size of the slice in each dimension.
    std::vector<unsigned int> m_Size;
};
using NpuSliceFloat32Workload = NpuSliceWorkload<armnn::DataType::Float32>;
using NpuSliceFloat16Workload = NpuSliceWorkload<armnn::DataType::Float16>;
using NpuSliceUint8Workload = NpuSliceWorkload<armnn::DataType::QAsymmU8>;
}  // namespace armnn
