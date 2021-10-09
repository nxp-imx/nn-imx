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
#include <iostream>
#include "TNpuWorkloads.hpp"

namespace armnn {
template <typename ParentDescriptor, nnrt::OperationType operationType,
         typename armnn::DataType... DataTypes>
class NpuElementwiseWorkload : public TNpuWorkload<ParentDescriptor, DataTypes...> {
   public:
    using base_type = TNpuWorkload<ParentDescriptor, DataTypes...>;
    explicit NpuElementwiseWorkload(const ParentDescriptor& descriptor, const WorkloadInfo& info)
        : TNpuWorkload<ParentDescriptor, DataTypes...>(descriptor, info) {
        // Add inputs operand
        assert(2 == descriptor.m_Inputs.size());
        std::vector<uint32_t> inOperandIds = this->AddOperandWithTensorHandle(
                descriptor.m_Inputs);

        std::vector<uint32_t> outOperandIds;
        NpuTensorHandler* outputTensorHandle =
            dynamic_cast<NpuTensorHandler*>(descriptor.m_Outputs[0]);
        if (outputTensorHandle) {
            uint32_t outputTensorId = this->AddOperandAndSetValue(
                outputTensorHandle->GetTensorInfo(), outputTensorHandle->GetShape(), nullptr);
            outOperandIds.push_back(outputTensorId);
        }

        this->AddOperation(operationType,
                           inOperandIds.size(),
                           inOperandIds.data(),
                           outOperandIds.size(),
                           outOperandIds.data());
    }
};
using NpuAdditionFloat32Workload = NpuElementwiseWorkload<AdditionQueueDescriptor,
      nnrt::OperationType::ADD, armnn::DataType::Float32>;
using NpuAdditionFloat16Workload = NpuElementwiseWorkload<AdditionQueueDescriptor,
      nnrt::OperationType::ADD, armnn::DataType::Float16>;
using NpuAdditionUint8Workload = NpuElementwiseWorkload<AdditionQueueDescriptor,
      nnrt::OperationType::ADD, armnn::DataType::QAsymmU8>;
using NpuAdditionInt8Workload = NpuElementwiseWorkload<AdditionQueueDescriptor,
      nnrt::OperationType::ADD, armnn::DataType::QAsymmS8>;

using NpuMinimumFloat32Workload = NpuElementwiseWorkload<MinimumQueueDescriptor,
      nnrt::OperationType::MINIMUM, armnn::DataType::Float32>;
using NpuMinimumFloat16Workload = NpuElementwiseWorkload<MinimumQueueDescriptor,
      nnrt::OperationType::MINIMUM, armnn::DataType::Float16>;
using NpuMinimumUint8Workload = NpuElementwiseWorkload<MinimumQueueDescriptor,
      nnrt::OperationType::MINIMUM, armnn::DataType::QAsymmU8>;

using NpuMaximumFloat32Workload = NpuElementwiseWorkload<MaximumQueueDescriptor,
      nnrt::OperationType::MAXIMUM, armnn::DataType::Float32>;
using NpuMaximumFloat16Workload = NpuElementwiseWorkload<MaximumQueueDescriptor,
      nnrt::OperationType::MAXIMUM, armnn::DataType::Float16>;
using NpuMaximumUint8Workload = NpuElementwiseWorkload<MaximumQueueDescriptor,
      nnrt::OperationType::MAXIMUM, armnn::DataType::QAsymmU8>;

using NpuSubtractionFloat32Workload = NpuElementwiseWorkload<SubtractionQueueDescriptor,
      nnrt::OperationType::SUB, armnn::DataType::Float32>;
using NpuSubtractionFloat16Workload = NpuElementwiseWorkload<SubtractionQueueDescriptor,
      nnrt::OperationType::SUB, armnn::DataType::Float16>;
using NpuSubtractionUint8Workload = NpuElementwiseWorkload<SubtractionQueueDescriptor,
      nnrt::OperationType::SUB, armnn::DataType::QAsymmU8>;

using NpuDivisionFloat32Workload = NpuElementwiseWorkload<DivisionQueueDescriptor,
      nnrt::OperationType::DIV, armnn::DataType::Float32>;
using NpuDivisionFloat16Workload = NpuElementwiseWorkload<DivisionQueueDescriptor,
      nnrt::OperationType::DIV, armnn::DataType::Float16>;
using NpuDivisionUint8Workload = NpuElementwiseWorkload<DivisionQueueDescriptor,
      nnrt::OperationType::DIV, armnn::DataType::QAsymmU8>;

using NpuMultiplicationFloat32Workload = NpuElementwiseWorkload<MultiplicationQueueDescriptor,
      nnrt::OperationType::MUL, armnn::DataType::Float32>;
using NpuMultiplicationFloat16Workload = NpuElementwiseWorkload<MultiplicationQueueDescriptor,
      nnrt::OperationType::MUL, armnn::DataType::Float16>;
using NpuMultiplicationUint8Workload = NpuElementwiseWorkload<MultiplicationQueueDescriptor,
      nnrt::OperationType::MUL, armnn::DataType::QAsymmU8>;
}  // namespace armnn
