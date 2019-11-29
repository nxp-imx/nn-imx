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

#include <FloatingPointConverter.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <boost/log/trivial.hpp>
#include "Permute.hpp"
#include "TNpuWorkloads.hpp"

#include "FakeBiasSelector.hpp"

namespace armnn {

template <typename armnn::DataType... DataTypes>
class NpuDepthWiseConvolution2dWorkload
    : public TNpuWorkload<DepthwiseConvolution2dQueueDescriptor, DataTypes...> {
   public:
    using FakeBias = FakeBiasTypeSelector<DataTypes...>;

    static_assert(std::is_same<typename FakeBias::type, void>::value == false,
                  "FakeBias data type In DepthWiseConv not added");

    explicit NpuDepthWiseConvolution2dWorkload(
        const DepthwiseConvolution2dQueueDescriptor& descriptor, const WorkloadInfo& info)
        : TNpuWorkload<DepthwiseConvolution2dQueueDescriptor, DataTypes...>(descriptor, info),
          m_Weight(std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Weight))),
          m_Bias(descriptor.m_Parameters.m_BiasEnabled
                     ? std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Bias))
                     : nullptr),
          m_PadLeft(descriptor.m_Parameters.m_PadLeft),
          m_PadRight(descriptor.m_Parameters.m_PadRight),
          m_PadTop(descriptor.m_Parameters.m_PadTop),
          m_PadBottom(descriptor.m_Parameters.m_PadBottom),
          m_StrideX(descriptor.m_Parameters.m_StrideX),
          m_StrideY(descriptor.m_Parameters.m_StrideY),
          m_DataLayout(descriptor.m_Parameters.m_DataLayout) {
        // Add inputs operand
        std::vector<uint32_t> inOperandIds;

        // order is important
        // ONLY 1 input
        assert(1 == descriptor.m_Inputs.size());
        NpuTensorHandler* inputTensorHandle =
            dynamic_cast<NpuTensorHandler*>(descriptor.m_Inputs[0]);
        uint32_t inputOperandId = this->AddOperandAndSetValue(
            inputTensorHandle->GetTensorInfo(), inputTensorHandle->GetShape(), nullptr);
        inOperandIds.push_back(inputOperandId);

        TensorShape weightShape = m_Weight->GetShape();
        const TensorInfo& weightInfo = m_Weight->GetTensorInfo();

        // 1. Driver needs the [1, N*C, H, W] formate of weight and chooses the filter
        //    as the order of batch0-channel0, batch0-channel1, batch1-channel0
        //    batch1-channel1 ... So we need to permute weight data from NCHW to CNHW and reshape
        //    weight shape to [1, N*C, H, W]
        // 2. Weight is always NCHW in armnn, we need to permute weight data from NCHW to NHWC,
        //    because all constant operand will be permuted from NHWC to NCHW
        if (m_DataLayout == armnn::DataLayout::NHWC) {
            uint32_t dataTypeSize = 4;
            if (weightInfo.GetDataType() == DataType::QuantisedAsymm8) {
                dataTypeSize = 1;
            } else if (weightInfo.GetDataType() == DataType::Float16) {
                dataTypeSize = 2;
            }
            // swap N and C
            std::swap(weightShape[0], weightShape[1]);
            // NCHW->CNHW
            const armnn::PermutationVector NCHWToCNHW = {1, 0, 2, 3};
            boost::scoped_array<uint8_t> temp;
            temp.reset(new uint8_t[weightInfo.GetNumBytes()]);
            armnnUtils::Permute(
                weightShape, NCHWToCNHW, m_Weight->GetTensor<void>(), temp.get(), dataTypeSize);

            // convert shape from [C, N, H, W] to [1, C*N, H, W]
            weightShape[1] = weightShape[0] * weightShape[1];
            weightShape[0] = 1;

            // permute for [1, C*N, H, W] to [1, H, W, C*N]
            std::swap(weightShape[1], weightShape[2]);
            std::swap(weightShape[2], weightShape[3]);
            const armnn::PermutationVector NCHWToNHWC = {0, 3, 1, 2};

            m_KernelData.reset(new uint8_t[weightInfo.GetNumBytes()]);
            armnnUtils::Permute(
                weightShape, NCHWToNHWC, temp.get(), m_KernelData.get(), dataTypeSize);

            inOperandIds.push_back(
                this->AddOperandAndSetValue(weightInfo, weightShape, m_KernelData.get()));
        } else {
            // swap N and C
            std::swap(weightShape[0], weightShape[1]);

            // NCHW->CNHW
            const armnn::PermutationVector NCHWToCNHW = {1, 0, 2, 3};

            uint32_t dataTypeSize = 4;
            if (weightInfo.GetDataType() == DataType::QuantisedAsymm8) {
                dataTypeSize = 1;
            } else if (weightInfo.GetDataType() == DataType::Float16) {
                dataTypeSize = 2;
            }
            m_KernelData.reset(new uint8_t[weightInfo.GetNumBytes()]);
            armnnUtils::Permute(weightShape,
                                NCHWToCNHW,
                                m_Weight->GetTensor<void>(),
                                m_KernelData.get(),
                                dataTypeSize);

            // convert shape to [1, N*C, H, W]
            weightShape[1] = weightShape[0] * weightShape[1];
            weightShape[0] = 1;

            inOperandIds.push_back(
                this->AddOperandAndSetValue(weightInfo, weightShape, m_KernelData.get()));
        }

        // Add bias operand
        if (m_Bias) {
            TensorInfo biasInfo = m_Bias->GetTensorInfo();
            const TensorShape biasShape = m_Bias->GetShape();
            if (biasInfo.GetDataType() == DataType::Float16) {
                biasInfo.SetDataType(DataType::Float32);
                m_Fp32BiasData.reset(new float[biasInfo.GetNumElements()]);
                armnnUtils::FloatingPointConverter::ConvertFloat16To32(
                    m_Bias->GetTensor<Half>(), biasInfo.GetNumElements(), m_Fp32BiasData.get());
                inOperandIds.push_back(
                    this->AddOperandAndSetValue(biasInfo, biasShape, m_Fp32BiasData.get()));
            } else {
                inOperandIds.push_back(
                    this->AddOperandAndSetValue(biasInfo, biasShape, m_Bias->GetTensor<void>()));
            }
        } else {
            TensorShape biasShape(1);
            TensorInfo biasInfo(biasShape, FakeBias::value);
            if (m_DataLayout == armnn::DataLayout::NCHW) {
                biasShape[0] = weightShape[1];  // Channels
            } else {
                biasShape[0] = weightShape[3];  // Channels
            }

            m_FakeBiasData.resize(biasShape[0]);
            biasInfo.SetShape(biasShape);

            if (FakeBias::value == DataType::Signed32) {
                auto biasScale = inputTensorHandle->GetTensorInfo().GetQuantizationScale() *
                                 weightInfo.GetQuantizationScale();
                int32_t biasZp = 0;

                biasInfo.SetQuantizationOffset(biasZp);
                biasInfo.SetQuantizationScale(biasScale);
            }
            memset(m_FakeBiasData.data(),
                   0,
                   m_FakeBiasData.size() * sizeof(decltype(m_FakeBiasData[0])));

            inOperandIds.push_back(
                this->AddOperandAndSetValue(biasInfo, biasShape, m_FakeBiasData.data()));
        }

        // Add padding left operand
        inOperandIds.push_back(this->AddOperandAndSetValue(m_PadLeft));

        // Add padding right operand
        inOperandIds.push_back(this->AddOperandAndSetValue(m_PadRight));

        // Add padding top operand
        inOperandIds.push_back(this->AddOperandAndSetValue(m_PadTop));

        // Add padding bottom operand
        inOperandIds.push_back(this->AddOperandAndSetValue(m_PadBottom));

        // Add stride width operand
        inOperandIds.push_back(this->AddOperandAndSetValue(m_StrideX));

        // Add stride height operand
        inOperandIds.push_back(this->AddOperandAndSetValue(m_StrideY));

        // Add depthwise multiplier
        int32_t inputChannels;
        int32_t outputChannels;
        int32_t depthMultiplier;
        auto inputPtr = dynamic_cast<NpuTensorHandler*>(descriptor.m_Inputs[0]);
        auto outputPtr = dynamic_cast<NpuTensorHandler*>(descriptor.m_Outputs[0]);
        if (m_DataLayout == armnn::DataLayout::NHWC) {
            inputChannels = inputPtr->GetShape()[3];
            outputChannels = outputPtr->GetShape()[3];
        } else {
            inputChannels = inputPtr->GetShape()[1];
            outputChannels = outputPtr->GetShape()[1];
        }
        depthMultiplier = outputChannels / inputChannels;
        inOperandIds.push_back(this->AddOperandAndSetValue(depthMultiplier));

        // Add fusecode operand
        int32_t fuseCode = 0;
        inOperandIds.push_back(this->AddOperandAndSetValue(fuseCode));

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

        this->AddOperation(nnrt::OperationType::DEPTHWISE_CONV_2D,
                           inOperandIds.size(),
                           inOperandIds.data(),
                           outOperandIds.size(),
                           outOperandIds.data());
    }

   private:
    std::unique_ptr<ScopedCpuTensorHandle> m_Weight;
    std::unique_ptr<ScopedCpuTensorHandle> m_Bias;
    uint32_t m_PadLeft;
    uint32_t m_PadRight;
    uint32_t m_PadTop;
    uint32_t m_PadBottom;
    uint32_t m_StrideX;
    uint32_t m_StrideY;
    armnn::DataLayout m_DataLayout;
    mutable boost::scoped_array<uint8_t> m_KernelData;

    std::vector<typename FakeBias::type> m_FakeBiasData;  //!< workaround: bias required by shader
    mutable boost::scoped_array<float> m_Fp32BiasData;
};
using NpuDepthWiseConvolution2dFloat32Workload =
    NpuDepthWiseConvolution2dWorkload<armnn::DataType::Float32>;
using NpuDepthWiseConvolution2dFloat16Workload =
    NpuDepthWiseConvolution2dWorkload<armnn::DataType::Float16>;
using NpuDepthWiseConvolution2dUint8Workload =
    NpuDepthWiseConvolution2dWorkload<armnn::DataType::QuantisedAsymm8>;
}  // namespace armnn
