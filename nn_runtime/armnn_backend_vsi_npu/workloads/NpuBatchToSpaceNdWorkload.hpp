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
class NpuBatchToSpaceNdWorkload : public TNpuWorkload<BatchToSpaceNdQueueDescriptor, DataTypes...> {
   public:
    using base_type = TNpuWorkload<BatchToSpaceNdQueueDescriptor, DataTypes...>;
    explicit NpuBatchToSpaceNdWorkload(const BatchToSpaceNdQueueDescriptor& descriptor,
                                       const WorkloadInfo& info)
        : TNpuWorkload<BatchToSpaceNdQueueDescriptor, DataTypes...>(descriptor, info),
          m_BlockShape(descriptor.m_Parameters.m_BlockShape),
          m_Crops(descriptor.m_Parameters.m_Crops),
          m_DataLayout(descriptor.m_Parameters.m_DataLayout) {
        // Add input operand
        // Only 1 input
        std::vector<uint32_t> inOperandIds;
        NpuTensorHandler* inputTensorHandle =
            dynamic_cast<NpuTensorHandler*>(descriptor.m_Inputs[0]);
        uint32_t inputOperandId = this->AddOperandAndSetValue(
            inputTensorHandle->GetTensorInfo(), inputTensorHandle->GetShape(), nullptr);
        inOperandIds.push_back(inputOperandId);

        // Add block shape operand
        TensorShape blockShape({(uint32_t)m_BlockShape.size()});
        TensorInfo blockInfo(blockShape, DataType::Signed32);
        inOperandIds.push_back(
            this->AddOperandAndSetValue(blockInfo, blockShape, m_BlockShape.data()));

        // Add layout operand
        int32_t layoutCode = m_DataLayout == armnn::DataLayout::NCHW
                                 ? int32_t(nnrt::DataLayout::NCHW)
                                 : int32_t(nnrt::DataLayout::NHWC);
        inOperandIds.push_back(this->AddOperandAndSetValue(layoutCode));

        // Add output operand
        std::vector<uint32_t> outOperandIds;
        NpuTensorHandler* outputTensorHandle =
            dynamic_cast<NpuTensorHandler*>(descriptor.m_Outputs[0]);
        uint32_t outputTensorId = this->AddOperandAndSetValue(
            outputTensorHandle->GetTensorInfo(), outputTensorHandle->GetShape(), nullptr);
        outOperandIds.push_back(outputTensorId);

        this->AddOperation(nnrt::OperationType::BATCH_TO_SPACE_ND,
                           inOperandIds.size(),
                           inOperandIds.data(),
                           outOperandIds.size(),
                           outOperandIds.data());
    }

   private:
    // Block shape value.
    std::vector<unsigned int> m_BlockShape;
    // The values to crop from the input dimension.
    std::vector<std::pair<unsigned int, unsigned int>> m_Crops;
    // The data layout to be used (NCHW, NHWC).
    DataLayout m_DataLayout;
};
using NpuBatchToSpaceNdDFloat32Workload = NpuBatchToSpaceNdWorkload<armnn::DataType::Float32>;
using NpuBatchToSpaceNdDFloat16Workload = NpuBatchToSpaceNdWorkload<armnn::DataType::Float16>;
using NpuBatchToSpaceNdNDUint8Workload =
    NpuBatchToSpaceNdWorkload<armnn::DataType::QuantisedAsymm8>;
}  // namespace armnn