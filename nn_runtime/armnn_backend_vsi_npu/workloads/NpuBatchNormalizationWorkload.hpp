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
class NpuBatchNormalizationWorkload
    : public TNpuWorkload<BatchNormalizationQueueDescriptor, DataTypes...> {
   public:
    using base_type = TNpuWorkload<BatchNormalizationQueueDescriptor, DataTypes...>;
    explicit NpuBatchNormalizationWorkload(const BatchNormalizationQueueDescriptor& descriptor,
                                           const WorkloadInfo& info)
        : TNpuWorkload<BatchNormalizationQueueDescriptor, DataTypes...>(descriptor, info),
          m_Mean(std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Mean))),
          m_Variance(std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Variance))),
          m_Beta(std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Beta))),
          m_Gamma(std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Gamma))),
          m_DataLayout(descriptor.m_Parameters.m_DataLayout) {
        std::vector<uint32_t> inOperandIds;

        // Only 1 input tensor
        NpuTensorHandler* inputTensorHandle =
            dynamic_cast<NpuTensorHandler*>(descriptor.m_Inputs[0]);
        uint32_t inputOperandId = this->AddOperandAndSetValue(
            inputTensorHandle->GetTensorInfo(), inputTensorHandle->GetShape(), nullptr);
        inOperandIds.push_back(inputOperandId);

        // order is important
        auto shape = static_cast<const TensorShape>(m_Mean->GetShape());
        inOperandIds.push_back(
            this->AddOperandAndSetValue(m_Mean->GetTensorInfo(), shape, m_Mean->Map(true)));
        m_Mean->Unmap();

        inOperandIds.push_back(this->AddOperandAndSetValue(
            m_Variance->GetTensorInfo(), m_Variance->GetShape(), m_Variance->Map(true)));
        m_Variance->Unmap();

        inOperandIds.push_back(this->AddOperandAndSetValue(
            m_Gamma->GetTensorInfo(), m_Gamma->GetShape(), m_Gamma->Map(true)));
        m_Gamma->Unmap();

        inOperandIds.push_back(this->AddOperandAndSetValue(
            m_Beta->GetTensorInfo(), m_Beta->GetShape(), m_Beta->Map(true)));
        m_Beta->Unmap();

        inOperandIds.push_back(
            this->AddOperandAndSetValue(descriptor.m_Parameters.m_Eps + 0.0001f));

        // Add layout operand
        int32_t layoutCode = m_DataLayout == armnn::DataLayout::NCHW
                                 ? int32_t(nnrt::DataLayout::NCHW)
                                 : int32_t(nnrt::DataLayout::NHWC);
        inOperandIds.push_back(this->AddOperandAndSetValue(layoutCode));

        std::vector<uint32_t> outOperandIds;
        NpuTensorHandler* outputTensorHandle =
            dynamic_cast<NpuTensorHandler*>(descriptor.m_Outputs[0]);
        uint32_t outputTensorId = this->AddOperandAndSetValue(
            outputTensorHandle->GetTensorInfo(), outputTensorHandle->GetShape(), nullptr);
        outOperandIds.push_back(outputTensorId);

        this->AddOperation(nnrt::OperationType::BATCH_NORM,
                           inOperandIds.size(),
                           inOperandIds.data(),
                           outOperandIds.size(),
                           outOperandIds.data());
    }

   private:
    std::unique_ptr<ScopedCpuTensorHandle> m_Mean;
    std::unique_ptr<ScopedCpuTensorHandle> m_Variance;
    std::unique_ptr<ScopedCpuTensorHandle> m_Beta;
    std::unique_ptr<ScopedCpuTensorHandle> m_Gamma;
    DataLayout m_DataLayout;
};
using NpuBatchNormalizationFloat32Workload =
    NpuBatchNormalizationWorkload<armnn::DataType::Float32>;
using NpuBatchNormalizationFloat16Workload =
    NpuBatchNormalizationWorkload<armnn::DataType::Float16>;
using NpuBatchNormalizationUint8Workload =
    NpuBatchNormalizationWorkload<armnn::DataType::QAsymmU8>;
}  // namespace armnn
