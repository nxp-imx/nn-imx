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
#include <armnnUtils/FloatingPointConverter.hpp>
#include "TNpuWorkloads.hpp"

#include "FakeBiasSelector.hpp"

namespace {
template <typename T>
inline void TransposeWeight(T* src, T* dst, const armnn::TensorShape shape) {
    for (uint32_t row = 0; row < shape[0]; row++) {
        for (uint32_t colum = 0; colum < shape[1]; colum++) {
            *(dst + shape[0] * colum + row) =
                *(src + shape[1] * row + colum);
        }
    }
}
}

namespace armnn {
template <typename armnn::DataType... DataTypes>
class NpuFullyConnectedFloatWorkload
    : public TNpuWorkload<FullyConnectedQueueDescriptor, DataTypes...> {
   public:
    using FakeBias = FakeBiasTypeSelector<DataTypes...>;

    static_assert(std::is_same<typename FakeBias::type, void>::value == false,
                  "FakeBias data type In FullyConnected not added");
    using base_type = TNpuWorkload<FullyConnectedQueueDescriptor, DataTypes...>;
    explicit NpuFullyConnectedFloatWorkload(const FullyConnectedQueueDescriptor& descriptor,
                                            const WorkloadInfo& info)
        : TNpuWorkload<FullyConnectedQueueDescriptor, DataTypes...>(descriptor, info),
          m_Weight(std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Weight))),
          m_Bias(descriptor.m_Parameters.m_BiasEnabled
                     ? std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Bias))
                     : nullptr) {
        auto inputPtr = dynamic_cast<NpuTensorHandler*>(descriptor.m_Inputs[0]);

        uint32_t inputOperandId =
            this->AddOperandAndSetValue(inputPtr->GetTensorInfo(), inputPtr->GetShape(), nullptr);

        // Add weight operand
        TensorShape weightShape = m_Weight->GetShape();
        const TensorInfo& weightInfo = m_Weight->GetTensorInfo();
        unsigned int weightOperandId;
        if (descriptor.m_Parameters.m_TransposeWeightMatrix) {
            weightOperandId =
                this->AddOperandAndSetValue(weightInfo, weightShape, m_Weight->GetTensor<void>());
        } else {
            // Transpose weight
            m_TransposedWeight.reset(new uint8_t[weightInfo.GetNumBytes()]);
            if (weightInfo.GetDataType() == DataType::QAsymmU8) {
                TransposeWeight<uint8_t>((uint8_t*)m_Weight->GetTensor<void>(),
                                         (uint8_t*)m_TransposedWeight.get(),
                                         weightShape);
            } else if (weightInfo.GetDataType() == DataType::Float32) {
                TransposeWeight<float>((float*)m_Weight->GetTensor<void>(),
                                       (float*)m_TransposedWeight.get(),
                                       weightShape);
            } else if (weightInfo.GetDataType() == DataType::Float16) {
                TransposeWeight<uint16_t>((uint16_t*)m_Weight->GetTensor<void>(),
                                          (uint16_t*)m_TransposedWeight.get(),
                                          weightShape);
            }
            std::swap(weightShape[0], weightShape[1]);
            weightOperandId =
                this->AddOperandAndSetValue(weightInfo, weightShape, m_TransposedWeight.get());
        }

        // Add bias operand
        // assert(m_Bias != nullptr);
        unsigned int biasOperandId;
        if (m_Bias) {
            TensorInfo biasInfo = m_Bias->GetTensorInfo();
            const TensorShape biasShape = m_Bias->GetShape();
            if (biasInfo.GetDataType() == DataType::Float16) {
                biasInfo.SetDataType(DataType::Float32);
                m_Fp32BiasData.reset(new float[biasInfo.GetNumElements()]);
                armnnUtils::FloatingPointConverter::ConvertFloat16To32(
                    m_Bias->GetTensor<Half>(), biasInfo.GetNumElements(), m_Fp32BiasData.get());
                biasOperandId =
                    this->AddOperandAndSetValue(biasInfo, biasShape, m_Fp32BiasData.get());
            } else {
                biasOperandId =
                    this->AddOperandAndSetValue(biasInfo, biasShape, m_Bias->GetTensor<void>());
            }
        } else {
            TensorShape biasShape(1);
            TensorInfo biasInfo(biasShape, FakeBias::value);
            biasShape[0] = weightShape[0];  // output colum
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
            biasOperandId = this->AddOperandAndSetValue(biasInfo, biasShape, m_FakeBiasData.data());
        }

        // Add fuse operand
        int32_t noneValue = 0;
        unsigned int fuseOperandId = this->AddOperandAndSetValue(noneValue);

        // Add fc operation to model
        int outputSize = descriptor.m_Outputs.size();
        uint32_t addInputIndexes[4];
        uint32_t addOutputIndexes[outputSize];

        addInputIndexes[0] = inputOperandId;
        addInputIndexes[1] = weightOperandId;
        addInputIndexes[2] = biasOperandId;
        addInputIndexes[3] = fuseOperandId;

        for (int i = 0; i < outputSize; i++) {
            auto outputPtr = dynamic_cast<NpuTensorHandler*>(descriptor.m_Outputs[i]);

            addOutputIndexes[i] = this->AddOperandAndSetValue(
                outputPtr->GetTensorInfo(), outputPtr->GetShape(), nullptr);
        }

        this->AddOperation(
            nnrt::OperationType::FULLY_CONNECTED, 4, addInputIndexes, outputSize, addOutputIndexes);
    }

   private:
    std::unique_ptr<ScopedCpuTensorHandle> m_Weight;
    std::unique_ptr<ScopedCpuTensorHandle> m_Bias;
    mutable boost::scoped_array<uint8_t> m_TransposedWeight;
    mutable boost::scoped_array<float> m_Fp32BiasData;
    std::vector<typename FakeBias::type> m_FakeBiasData;  //!< workaround: bias required by shader
};
using NpuFullyConnectedFloat32Workload = NpuFullyConnectedFloatWorkload<armnn::DataType::Float32>;
using NpuFullyConnectedFloat16Workload = NpuFullyConnectedFloatWorkload<armnn::DataType::Float16>;
using NpuFullyConnectedUint8Workload =
    NpuFullyConnectedFloatWorkload<armnn::DataType::QAsymmU8>;
}  // namespace armnn