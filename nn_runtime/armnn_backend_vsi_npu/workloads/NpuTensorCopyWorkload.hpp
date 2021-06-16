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
template <typename ParentDescriptor, nnrt::OperationType operationType,
         typename armnn::DataType... DataTypes>
class NpuTensorCopyWorkload : public TNpuWorkload<ParentDescriptor, DataTypes...> {
   public:
    using base_type = TNpuWorkload<ParentDescriptor, DataTypes...>;
    explicit NpuTensorCopyWorkload(const ParentDescriptor& descriptor, const WorkloadInfo& info)
        : TNpuWorkload<ParentDescriptor, DataTypes...>(descriptor, info) {
        // Add inputs operand
        std::vector<uint32_t> inOperandIds = this->AddOperandWithTensorHandle(
                descriptor.m_Inputs);

        std::vector<uint32_t> outOperandIds = this->AddOperandWithTensorHandle(
                descriptor.m_Outputs);
        this->AddOperation(operationType,
                           inOperandIds.size(),
                           inOperandIds.data(),
                           outOperandIds.size(),
                           outOperandIds.data());
    }
};

using NpuDequantizeUint8Workload = NpuTensorCopyWorkload<
        DequantizeQueueDescriptor, nnrt::OperationType::DATA_CONVERT,
        armnn::DataType::QAsymmU8, armnn::DataType::Float32>;

using NpuFp32ToFp16Workload = NpuTensorCopyWorkload<
        ConvertFp32ToFp16QueueDescriptor, nnrt::OperationType::DATA_CONVERT,
        armnn::DataType::Float32, armnn::DataType::Float16>;

using NpuFp16ToFp32Workload = NpuTensorCopyWorkload<
        ConvertFp16ToFp32QueueDescriptor, nnrt::OperationType::DATA_CONVERT,
        armnn::DataType::Float16, armnn::DataType::Float32>;

using NpuMemCopyFloat32Workload = NpuTensorCopyWorkload<
        MemCopyQueueDescriptor, nnrt::OperationType::DATA_CONVERT,
        armnn::DataType::Float32>;

using NpuMemCopyFloat16Workload = NpuTensorCopyWorkload<
        MemCopyQueueDescriptor, nnrt::OperationType::DATA_CONVERT,
        armnn::DataType::Float16>;

using NpuMemCopyUint8Workload = NpuTensorCopyWorkload<
        MemCopyQueueDescriptor, nnrt::OperationType::DATA_CONVERT,
        armnn::DataType::QAsymmU8>;
}  // namespace armnn
