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
class NpuNormalizationWorkload : public TNpuWorkload<NormalizationQueueDescriptor, DataTypes...> {
   public:
    using base_type = TNpuWorkload<NormalizationQueueDescriptor, DataTypes...>;
    explicit NpuNormalizationWorkload(const NormalizationQueueDescriptor& descriptor,
                                      const WorkloadInfo& info)
        : TNpuWorkload<NormalizationQueueDescriptor, DataTypes...>(descriptor, info),
          m_NormChannelType(descriptor.m_Parameters.m_NormChannelType),
          m_NormMethodType(descriptor.m_Parameters.m_NormMethodType),
          m_NormSize(descriptor.m_Parameters.m_NormSize),
          m_Alpha(descriptor.m_Parameters.m_Alpha),
          m_Beta(descriptor.m_Parameters.m_Beta),
          m_K(descriptor.m_Parameters.m_K),
          m_DataLayout(descriptor.m_Parameters.m_DataLayout) {
        std::vector<uint32_t> inOperandIds;

        // Only 1 input tensor
        NpuTensorHandler* inputTensorHandle =
            dynamic_cast<NpuTensorHandler*>(descriptor.m_Inputs[0]);
        uint32_t inputOperandId = this->AddOperandAndSetValue(
            inputTensorHandle->GetTensorInfo(), inputTensorHandle->GetShape(), nullptr);
        inOperandIds.push_back(inputOperandId);

        // order is important
        // Add norm size(radius) operand
        // Strong Assumption on rounding Mode
        inOperandIds.push_back(this->AddOperandAndSetValue(m_NormSize / 2u));

        // Add kappa(bias) operand
        inOperandIds.push_back(this->AddOperandAndSetValue(m_K));

        // Add alpha operand
        inOperandIds.push_back(this->AddOperandAndSetValue(m_Alpha));

        // Add beta oparand
        inOperandIds.push_back(this->AddOperandAndSetValue(m_Beta));

        // Add channel type operand
        inOperandIds.push_back(this->AddOperandAndSetValue(uint32_t(m_NormChannelType)));

        // Add method type operand
        inOperandIds.push_back(this->AddOperandAndSetValue(uint32_t(m_NormMethodType)));

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

        this->AddOperation(nnrt::OperationType::LOCAL_RESPONSE_NORMALIZATION,
                           inOperandIds.size(),
                           inOperandIds.data(),
                           outOperandIds.size(),
                           outOperandIds.data());
    }

   private:
    /// Normalization channel algorithm to use (Across, Within).
    NormalizationAlgorithmChannel m_NormChannelType;
    /// Normalization method algorithm to use (LocalBrightness, LocalContrast).
    NormalizationAlgorithmMethod m_NormMethodType;
    /// Depth radius value.
    uint32_t m_NormSize;
    /// Alpha value for the normalization equation.
    float m_Alpha;
    /// Beta value for the normalization equation.
    float m_Beta;
    /// Kappa value used for the across channel normalization equation.
    float m_K;
    /// The data layout to be used (NCHW, NHWC).
    DataLayout m_DataLayout;
};
using NpuNormalizationFloat32Workload = NpuNormalizationWorkload<armnn::DataType::Float32>;
using NpuNormalizationFloat16Workload = NpuNormalizationWorkload<armnn::DataType::Float16>;
using NpuNormalizationUint8Workload = NpuNormalizationWorkload<armnn::DataType::QAsymmU8>;
}  // namespace armnn