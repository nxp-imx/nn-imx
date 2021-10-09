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

#include <armnnUtils/FloatingPointConverter.hpp>
#include <backendsCommon/TensorHandle.hpp>
#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

#include "FakeBiasSelector.hpp"
#include "TNpuWorkloads.hpp"

namespace armnn {

template <typename armnn::DataType... DataTypes>
class NpuConvolution2dWorkload : public TNpuWorkload<Convolution2dQueueDescriptor, DataTypes...> {
   public:
    using FakeBias = FakeBiasTypeSelector<DataTypes...>;

    static_assert(std::is_same<typename FakeBias::type, void>::value == false,
                  "FakeBias data type In DepthWiseConv not added");

    using base_type = TNpuWorkload<Convolution2dQueueDescriptor, DataTypes...>;
    explicit NpuConvolution2dWorkload(const Convolution2dQueueDescriptor& descriptor,
                                      const WorkloadInfo& info)
        : TNpuWorkload<Convolution2dQueueDescriptor, DataTypes...>(descriptor, info),
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
        uint32_t inputIds[11];
        // Add input operand
        auto inputPtr = dynamic_cast<NpuTensorHandler*>(descriptor.m_Inputs[0]);
        if (inputPtr) {
            inputIds[0] = this->AddOperandAndSetValue(
                inputPtr->GetTensorInfo(), inputPtr->GetShape(), nullptr);
            // inputPtr->SetOperandId(inputIds[0]);
        }

        // Add filter operand
        const TensorShape& weightShape = m_Weight->GetShape();
        TensorInfo weightInfo = m_Weight->GetTensorInfo();

        if (weightInfo.HasPerAxisQuantization()) {
            // convolution weight out channel in armnn is according to data layout,
            // depthwiseconvolution weight([1, H, W, O]) out channel in armnn is const 3 if layout
            // in ArmNN is [N, C, H, W], weight will be [O, I, H, W], there is no convert operation
            // in nnrt, dim=3 if layout in ArmNN is [N, H, W, C], weight will be [O, H, W, I], there
            // are some convert operation in nnrt(permVal=[0, 3, 1, 2]), dim=0
            unsigned int kWeightQuantizationDim4OpenVX = 0;
            if (m_DataLayout == armnn::DataLayout::NCHW) {
                kWeightQuantizationDim4OpenVX = 3;
            }
            weightInfo.SetQuantizationDim(
                armnn::Optional<unsigned int>(kWeightQuantizationDim4OpenVX));
        }
        inputIds[1] =
            this->AddOperandAndSetValue(weightInfo, weightShape, m_Weight->GetTensor<void>());

        // Add bias operand
        // assert(m_Bias != nullptr);
        if (m_Bias) {
            TensorInfo biasInfo = m_Bias->GetTensorInfo();
            const TensorShape biasShape = m_Bias->GetShape();
            if (biasInfo.GetDataType() == DataType::Float16) {
                biasInfo.SetDataType(DataType::Float32);
                m_Fp32BiasData.resize(biasInfo.GetNumElements());
                armnnUtils::FloatingPointConverter::ConvertFloat16To32(
                    m_Bias->GetTensor<Half>(), biasInfo.GetNumElements(), m_Fp32BiasData.data());
                inputIds[2] =
                    this->AddOperandAndSetValue(biasInfo, biasShape, m_Fp32BiasData.data());
            } else {
                inputIds[2] =
                    this->AddOperandAndSetValue(biasInfo, biasShape, m_Bias->GetTensor<void>());
            }
        } else {
            TensorShape biasShape(1);
            TensorInfo biasInfo(biasShape, FakeBias::value);
            biasShape[0] = weightShape[0];  // Channels
            m_FakeBiasData.resize(biasShape[0]);
            biasInfo.SetShape(biasShape);

            if (FakeBias::value == DataType::Signed32) {
                auto biasScale = inputPtr->GetTensorInfo().GetQuantizationScale() *
                                 weightInfo.GetQuantizationScale();
                int32_t biasZp = 0;

                biasInfo.SetQuantizationOffset(biasZp);
                biasInfo.SetQuantizationScale(biasScale);
            }
            memset(m_FakeBiasData.data(), 0, m_FakeBiasData.size());

            inputIds[2] = this->AddOperandAndSetValue(biasInfo, biasShape, m_FakeBiasData.data());
        }

        // Add padding left operand
        inputIds[3] = this->AddOperandAndSetValue(m_PadLeft);

        // Add padding right operand
        inputIds[4] = this->AddOperandAndSetValue(m_PadRight);

        // Add padding top operand
        inputIds[5] = this->AddOperandAndSetValue(m_PadTop);

        // Add padding bottom operand
        inputIds[6] = this->AddOperandAndSetValue(m_PadBottom);

        // Add stride width operand
        inputIds[7] = this->AddOperandAndSetValue(m_StrideX);

        // Add stride height operand
        inputIds[8] = this->AddOperandAndSetValue(m_StrideY);

        // Add fusecode operand
        int32_t fuseCode = 0;
        inputIds[9] = this->AddOperandAndSetValue(fuseCode);

        // Add layout operand
        int32_t layoutCode = m_DataLayout == armnn::DataLayout::NCHW
                                 ? int32_t(nnrt::DataLayout::NCHW)
                                 : int32_t(nnrt::DataLayout::NHWC);
        inputIds[10] = this->AddOperandAndSetValue(layoutCode);

        // Add output operand
        int outputSize = descriptor.m_Outputs.size();
        uint32_t outputIds[outputSize];

        for (int i = 0; i < outputSize; i++) {
            auto outputPtr = dynamic_cast<NpuTensorHandler*>(descriptor.m_Outputs[i]);
            if (outputPtr) {
                outputIds[i] = this->AddOperandAndSetValue(
                    outputPtr->GetTensorInfo(), outputPtr->GetShape(), nullptr);
                // outputPtr->SetOperandId(outputIds[i]);
            }
        }
        this->AddOperation(nnrt::OperationType::CONV_2D, 11, inputIds, outputSize, outputIds);
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

    std::vector<typename FakeBias::type> m_FakeBiasData;  //!< workaround: bias required by shader
    mutable std::vector<float> m_Fp32BiasData;
};
using NpuConvolution2dFloat32Workload = NpuConvolution2dWorkload<armnn::DataType::Float32>;
using NpuConvolution2dFloat16Workload = NpuConvolution2dWorkload<armnn::DataType::Float16>;
using NpuConvolution2dUint8Workload = NpuConvolution2dWorkload<armnn::DataType::QAsymmU8>;
using NpuConvolution2dInt8Workload = NpuConvolution2dWorkload<armnn::DataType::QAsymmS8>;
}  // namespace armnn