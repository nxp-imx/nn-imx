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
class NpuSpaceToBatchNdWorkload : public TNpuWorkload<SpaceToBatchNdQueueDescriptor, DataTypes...> {
   public:
    using base_type = TNpuWorkload<SpaceToBatchNdQueueDescriptor, DataTypes...>;
    explicit NpuSpaceToBatchNdWorkload(const SpaceToBatchNdQueueDescriptor& descriptor,
                                       const WorkloadInfo& info)
        : TNpuWorkload<SpaceToBatchNdQueueDescriptor, DataTypes...>(descriptor, info),
          m_BlockShape(descriptor.m_Parameters.m_BlockShape),
          m_PadList(descriptor.m_Parameters.m_PadList),
          m_DataLayout(descriptor.m_Parameters.m_DataLayout) {
        // Add input operand
        // Only 1 input
        std::vector<uint32_t> inOperandIds;
        NpuTensorHandler* inputTensorHandle =
            dynamic_cast<NpuTensorHandler*>(descriptor.m_Inputs[0]);
        if (inputTensorHandle) {
            uint32_t inputOperandId = this->AddOperandAndSetValue(
                inputTensorHandle->GetTensorInfo(), inputTensorHandle->GetShape(), nullptr);
            inOperandIds.push_back(inputOperandId);
        }

        // Add block shape operand
        TensorShape blockShape({(uint32_t)m_BlockShape.size()});
        TensorInfo blockInfo(blockShape, DataType::Signed32);
        inOperandIds.push_back(
            this->AddOperandAndSetValue(blockInfo, blockShape, m_BlockShape.data()));

        // Add pad list operand
        TensorShape padListShape({2, 2});
        TensorInfo padListInfo(padListShape, DataType::Signed32);

        m_Pad.push_back(m_PadList[0].first);
        m_Pad.push_back(m_PadList[0].second);

        m_Pad.push_back(m_PadList[1].first);
        m_Pad.push_back(m_PadList[1].second);

        inOperandIds.push_back(
            this->AddOperandAndSetValue(padListInfo, padListShape, m_Pad.data()));

        // Add layout operand
        int32_t layoutCode = m_DataLayout == armnn::DataLayout::NCHW
                                 ? int32_t(nnrt::DataLayout::NCHW)
                                 : int32_t(nnrt::DataLayout::NHWC);
        inOperandIds.push_back(this->AddOperandAndSetValue(layoutCode));

        // Add output operand
        std::vector<uint32_t> outOperandIds;
        NpuTensorHandler* outputTensorHandle =
            dynamic_cast<NpuTensorHandler*>(descriptor.m_Outputs[0]);
        if (outputTensorHandle) {
            uint32_t outputTensorId = this->AddOperandAndSetValue(
                outputTensorHandle->GetTensorInfo(), outputTensorHandle->GetShape(), nullptr);
            outOperandIds.push_back(outputTensorId);
        }

        this->AddOperation(nnrt::OperationType::SPACE_TO_BATCH_ND,
                           inOperandIds.size(),
                           inOperandIds.data(),
                           outOperandIds.size(),
                           outOperandIds.data());
    }

   private:
    // Block shape value.
    std::vector<unsigned int> m_BlockShape;
    // Specifies the padding values for the input dimension:
    /// heightPad{top, bottom} widthPad{left, right}.
    std::vector<std::pair<unsigned int, unsigned int>> m_PadList;
    std::vector<unsigned int> m_Pad;
    // The data layout to be used (NCHW, NHWC).
    DataLayout m_DataLayout;
};
using NpuSpaceToBatchNDFloat32Workload = NpuSpaceToBatchNdWorkload<armnn::DataType::Float32>;
using NpuSpaceToBatchNDFloat16Workload = NpuSpaceToBatchNdWorkload<armnn::DataType::Float16>;
using NpuSpaceToBatchNDUint8Workload = NpuSpaceToBatchNdWorkload<armnn::DataType::QAsymmU8>;
}  // namespace armnn