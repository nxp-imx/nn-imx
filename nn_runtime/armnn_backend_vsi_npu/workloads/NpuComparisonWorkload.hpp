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
class NpuComparisonWorkload : public TNpuWorkload<ComparisonQueueDescriptor, DataTypes...> {
   public:
    using base_type = TNpuWorkload<ComparisonQueueDescriptor, DataTypes...>;
    explicit NpuComparisonWorkload(const ComparisonQueueDescriptor& descriptor, const WorkloadInfo& info)
        : TNpuWorkload<ComparisonQueueDescriptor, DataTypes...>(descriptor, info) {
        // Add inputs operand
        assert(2 == descriptor.m_Inputs.size());
        std::vector<uint32_t> inOperandIds = this->AddOperandWithTensorHandle(descriptor.m_Inputs);

        std::vector<uint32_t> outOperandIds;
        NpuTensorHandler* outputTensorHandle =
            dynamic_cast<NpuTensorHandler*>(descriptor.m_Outputs[0]);
        uint32_t outputTensorId = this->AddOperandAndSetValue(
            outputTensorHandle->GetTensorInfo(), outputTensorHandle->GetShape(), nullptr);
        outOperandIds.push_back(outputTensorId);

        switch (descriptor.m_Parameters.m_Operation) {
            case ComparisonOperation::Equal: {
                this->AddOperation(nnrt::OperationType::EQUAL,
                                   inOperandIds.size(),
                                   inOperandIds.data(),
                                   outOperandIds.size(),
                                   outOperandIds.data());
                break;
            }
            case ComparisonOperation::Greater: {
                this->AddOperation(nnrt::OperationType::GREATER,
                                   inOperandIds.size(),
                                   inOperandIds.data(),
                                   outOperandIds.size(),
                                   outOperandIds.data());
                break;
            }
            case ComparisonOperation::GreaterOrEqual: {
                this->AddOperation(nnrt::OperationType::GREATER_EQUAL,
                                   inOperandIds.size(),
                                   inOperandIds.data(),
                                   outOperandIds.size(),
                                   outOperandIds.data());
                break;
            }
            case ComparisonOperation::NotEqual: {
                this->AddOperation(nnrt::OperationType::NOT_EQUAL,
                                   inOperandIds.size(),
                                   inOperandIds.data(),
                                   outOperandIds.size(),
                                   outOperandIds.data());
                break;
            }
            case ComparisonOperation::Less: {
                this->AddOperation(nnrt::OperationType::LESS,
                                   inOperandIds.size(),
                                   inOperandIds.data(),
                                   outOperandIds.size(),
                                   outOperandIds.data());
                break;
            }
            case ComparisonOperation::LessOrEqual: {
                this->AddOperation(nnrt::OperationType::LESS_EQUAL,
                                   inOperandIds.size(),
                                   inOperandIds.data(),
                                   outOperandIds.size(),
                                   outOperandIds.data());
                break;
            }
            default:
                std::cout << "Not support ComparisonOperation.";
                assert(false);
        }
    }
};
using NpuComparisonFloat32Workload =
    NpuComparisonWorkload<armnn::DataType::Float32, armnn::DataType::Boolean>;
using NpuComparisonFloat16Workload =
    NpuComparisonWorkload<armnn::DataType::Float16, armnn::DataType::Boolean>;
using NpuComparisonUint8Workload =
    NpuComparisonWorkload<armnn::DataType::QAsymmU8, armnn::DataType::Boolean>;
}  // namespace armnn