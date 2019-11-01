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
class NpuPermuteWorkload : public TNpuWorkload<PermuteQueueDescriptor, DataTypes...> {
   public:
    explicit NpuPermuteWorkload(const PermuteQueueDescriptor& descriptor,
                                  const WorkloadInfo& info)
        : TNpuWorkload<PermuteQueueDescriptor, DataTypes...>(descriptor, info),
          m_Dims(descriptor.m_Parameters.m_DimMappings) {
        std::vector<uint32_t> inputIds = this->AddOperandWithTensorHandle(
                descriptor.m_Inputs);

        std::vector<uint32_t> dims_shape = {(uint32_t)m_Dims.GetSize()};
        std::vector<int32_t> dims(m_Dims.GetSize());
        // Armnn permute dims is mapped from output to input
        for (uint32_t i = 0; i < m_Dims.GetSize(); i ++) {
            for (uint32_t j = 0; j < m_Dims.GetSize(); j ++) {
                if (i == m_Dims[j]) {
                    dims[i] = j;
                    break;
                }
            }
        }
        inputIds.push_back(this->AddOperandAndSetValue(
                    dims_shape, DataType::Signed32, dims.data()));

        std::vector<uint32_t> outputIds = this->AddOperandWithTensorHandle(
                descriptor.m_Outputs);

        this->AddOperation(nnrt::OperationType::PERMUTE,
                inputIds.size(), inputIds.data(), outputIds.size(), outputIds.data());
    }

   private:
    PermutationVector m_Dims;

};
using NpuPermuteFloat32Workload = NpuPermuteWorkload<armnn::DataType::Float32>;
using NpuPermuteFloat16Workload = NpuPermuteWorkload<armnn::DataType::Float16>;
using NpuPermuteUint8Workload = NpuPermuteWorkload<armnn::DataType::QuantisedAsymm8>;
}  // namespace armnn
