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
class NpuTransposeConvolution2dWorkload : public TNpuWorkload<TransposeConvolution2dQueueDescriptor, DataTypes...> {
   public:
    explicit NpuTransposeConvolution2dWorkload(const TransposeConvolution2dQueueDescriptor& descriptor,
                                  const WorkloadInfo& info)
        : TNpuWorkload<TransposeConvolution2dQueueDescriptor, DataTypes...>(descriptor, info),
          m_Weight(std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Weight))),
          m_Bias(descriptor.m_Parameters.m_BiasEnabled
                     ? std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Bias))
                     : nullptr),
          m_StrideX(descriptor.m_Parameters.m_StrideX),
          m_StrideY(descriptor.m_Parameters.m_StrideY),
          m_PadLeft(descriptor.m_Parameters.m_PadLeft),
          m_PadRight(descriptor.m_Parameters.m_PadRight),
          m_PadTop(descriptor.m_Parameters.m_PadTop),
          m_PadBottom(descriptor.m_Parameters.m_PadBottom),
          m_DataLayout(descriptor.m_Parameters.m_DataLayout) {

        std::vector<uint32_t> inputIds = this->AddOperandWithTensorHandle(
                descriptor.m_Inputs);
        inputIds.resize(10);

        // Add filter operand
        inputIds[1] = this->AddOperandAndSetValue(
            m_Weight->GetTensorInfo(),
            m_Weight->GetShape(),
            m_Weight->GetTensor<void>());

        if (m_Bias) {
            inputIds[2] = this->AddOperandAndSetValue(
                m_Bias->GetTensorInfo(),
                m_Bias->GetShape(),
                m_Bias->GetTensor<void>());
        } else {
            // Set null data
            std::vector<uint32_t> biasShape = {1};
            inputIds[2] = this->AddOperandAndSetValue(biasShape, DataType::Signed32, nullptr);
        }

        inputIds[3] = this->AddOperandAndSetValue(m_PadLeft);
        inputIds[4] = this->AddOperandAndSetValue(m_PadRight);
        inputIds[5] = this->AddOperandAndSetValue(m_PadTop);
        inputIds[6] = this->AddOperandAndSetValue(m_PadBottom);
        inputIds[7] = this->AddOperandAndSetValue(m_StrideX);
        inputIds[8] = this->AddOperandAndSetValue(m_StrideY);
        int32_t layoutCode = m_DataLayout == armnn::DataLayout::NCHW
                                 ? int32_t(nnrt::DataLayout::NCHW)
                                 : int32_t(nnrt::DataLayout::NHWC);
        inputIds[9] = this->AddOperandAndSetValue(layoutCode);

        std::vector<uint32_t> outputIds = this->AddOperandWithTensorHandle(
                descriptor.m_Outputs);

        this->AddOperation(nnrt::OperationType::DECONV_2D,
                inputIds.size(), inputIds.data(), outputIds.size(), outputIds.data());
    }

   private:
    std::unique_ptr<ScopedCpuTensorHandle> m_Weight;
    std::unique_ptr<ScopedCpuTensorHandle> m_Bias;
    uint32_t m_StrideX;
    uint32_t m_StrideY;
    uint32_t m_PadLeft;
    uint32_t m_PadRight;
    uint32_t m_PadTop;
    uint32_t m_PadBottom;
    armnn::DataLayout m_DataLayout;
};
using NpuTransposeConvolution2dFloat32Workload = NpuTransposeConvolution2dWorkload<armnn::DataType::Float32>;
using NpuTransposeConvolution2dFloat16Workload = NpuTransposeConvolution2dWorkload<armnn::DataType::Float16>;
using NpuTransposeConvolution2dUint8Workload = NpuTransposeConvolution2dWorkload<armnn::DataType::QAsymmU8>;
}  // namespace armnn
