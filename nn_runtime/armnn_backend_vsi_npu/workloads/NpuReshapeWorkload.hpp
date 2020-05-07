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
class NpuReshapeWorkload : public TNpuWorkload<ReshapeQueueDescriptor, DataTypes...> {
   public:
    using base_type = TNpuWorkload<ReshapeQueueDescriptor, DataTypes...>;
    explicit NpuReshapeWorkload(const ReshapeQueueDescriptor& descriptor, const WorkloadInfo& info)
        : TNpuWorkload<ReshapeQueueDescriptor, DataTypes...>(descriptor, info),
          m_TargetShape(descriptor.m_Parameters.m_TargetShape) {
        // Add input operand
        // Only 1 input
        std::vector<uint32_t> inOperandIds;
        NpuTensorHandler* inputTensorHandle =
            dynamic_cast<NpuTensorHandler*>(descriptor.m_Inputs[0]);
        uint32_t inputOperandId = this->AddOperandAndSetValue(
            inputTensorHandle->GetTensorInfo(), inputTensorHandle->GetShape(), nullptr);
        inOperandIds.push_back(inputOperandId);

        // Add target shape operand
        uint32_t numDims = m_TargetShape.GetNumDimensions();
        for (uint32_t i = 0; i < numDims; i++) {
            m_TargetData.push_back(m_TargetShape[i]);
        }
        TensorShape targetShape({numDims});
        TensorInfo targetInfo(targetShape, DataType::Signed32);
        inOperandIds.push_back(
            this->AddOperandAndSetValue(targetInfo, targetShape, m_TargetData.data()));

        // Add output operand
        std::vector<uint32_t> outOperandIds;
        NpuTensorHandler* outputTensorHandle =
            dynamic_cast<NpuTensorHandler*>(descriptor.m_Outputs[0]);
        uint32_t outputTensorId = this->AddOperandAndSetValue(
            outputTensorHandle->GetTensorInfo(), outputTensorHandle->GetShape(), nullptr);
        outOperandIds.push_back(outputTensorId);

        this->AddOperation(nnrt::OperationType::RESHAPE,
                           inOperandIds.size(),
                           inOperandIds.data(),
                           outOperandIds.size(),
                           outOperandIds.data());
    }

   private:
    TensorShape m_TargetShape;
    std::vector<uint32_t> m_TargetData;
};
using NpuReshapeFloat32Workload = NpuReshapeWorkload<armnn::DataType::Float32>;
using NpuReshapeFloat16Workload = NpuReshapeWorkload<armnn::DataType::Float16>;
using NpuReshapeUint8Workload = NpuReshapeWorkload<armnn::DataType::QAsymmU8>;
}  // namespace armnn
