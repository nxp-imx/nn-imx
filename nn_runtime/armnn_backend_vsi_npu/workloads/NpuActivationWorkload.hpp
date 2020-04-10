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
class NpuActivationWorkload : public TNpuWorkload<ActivationQueueDescriptor, DataTypes...> {
   public:
    using base_type = TNpuWorkload<ActivationQueueDescriptor, DataTypes...>;
    explicit NpuActivationWorkload(const ActivationQueueDescriptor& descriptor,
                                   const WorkloadInfo& info)
        : TNpuWorkload<ActivationQueueDescriptor, DataTypes...>(descriptor, info),
          m_Function(descriptor.m_Parameters.m_Function),
          m_A(descriptor.m_Parameters.m_A),
          m_B(descriptor.m_Parameters.m_B) {
        // Add input operand
        // Only 1 input
        std::vector<uint32_t> inOperandIds;
        NpuTensorHandler* inputTensorHandle =
            dynamic_cast<NpuTensorHandler*>(descriptor.m_Inputs[0]);
        uint32_t inputOperandId = this->AddOperandAndSetValue(
            inputTensorHandle->GetTensorInfo(), inputTensorHandle->GetShape(), nullptr);
        inOperandIds.push_back(inputOperandId);

        // Add alpha operand
        if (ActivationFunction::TanH == m_Function || ActivationFunction::LeakyReLu == m_Function) {
            inOperandIds.push_back(this->AddOperandAndSetValue(m_A));
        }
        if (ActivationFunction::TanH == m_Function) {
            // Add beta operand
            inOperandIds.push_back(this->AddOperandAndSetValue(m_B));
        }
        if (ActivationFunction::Linear == m_Function) {
            std::vector<uint32_t> shape({1});
            inOperandIds.push_back(this->AddOperandAndSetValue(shape, armnn::DataType::Float32, &m_A));
            inOperandIds.push_back(this->AddOperandAndSetValue(shape, armnn::DataType::Float32, &m_B));
        }

        // Add output operand
        std::vector<uint32_t> outOperandIds;
        NpuTensorHandler* outputTensorHandle =
            dynamic_cast<NpuTensorHandler*>(descriptor.m_Outputs[0]);
        uint32_t outputTensorId = this->AddOperandAndSetValue(
            outputTensorHandle->GetTensorInfo(), outputTensorHandle->GetShape(), nullptr);
        outOperandIds.push_back(outputTensorId);

        auto inSize = inOperandIds.size();
        auto inPtr = inOperandIds.data();
        auto outSize = outOperandIds.size();
        auto outPtr = outOperandIds.data();
        switch (m_Function) {
            case ActivationFunction::ReLu:
                this->AddOperation(nnrt::OperationType::RELU, inSize, inPtr, outSize, outPtr);
                break;
            case ActivationFunction::BoundedReLu:
                if (m_A == 1) {
                    this->AddOperation(
                        nnrt::OperationType::RELU1, inSize, inPtr, outSize, outPtr);
                } else if (m_A == 6) {
                    this->AddOperation(
                        nnrt::OperationType::RELU6, inSize, inPtr, outSize, outPtr);

                } else {
                    BOOST_LOG_TRIVIAL(error) << "Unsupported BoundedReLU.";
                    return;
                }
                break;
            case ActivationFunction::Sigmoid:
                this->AddOperation(nnrt::OperationType::SIGMOID, inSize, inPtr, outSize, outPtr);
                break;
            case ActivationFunction::LeakyReLu:
                this->AddOperation(
                    nnrt::OperationType::LEAKY_RELU, inSize, inPtr, outSize, outPtr);
                break;
            case ActivationFunction::SoftReLu:
                this->AddOperation(
                    nnrt::OperationType::SOFT_RELU, inSize, inPtr, outSize, outPtr);
                break;
            case ActivationFunction::Abs:
                this->AddOperation(nnrt::OperationType::ABS, inSize, inPtr, outSize, outPtr);
                break;
            case ActivationFunction::Sqrt:
                this->AddOperation(nnrt::OperationType::SQRT, inSize, inPtr, outSize, outPtr);
                break;
            case ActivationFunction::Square:
                this->AddOperation(nnrt::OperationType::SQUARE, inSize, inPtr, outSize, outPtr);
                break;
            case ActivationFunction::TanH:
                this->AddOperation(nnrt::OperationType::TANH, inSize, inPtr, outSize, outPtr);
                break;
            case ActivationFunction::Linear:
                {
                   this->AddOperation(nnrt::OperationType::LINEAR, inSize, inPtr, outSize, outPtr);
                }
                break;
            default:
                BOOST_LOG_TRIVIAL(error) << "Unsupported Activation Function.";
                return;
        }
    }

   private:
    // (Sigmoid, TanH, Linear, ReLu, BoundedReLu, SoftReLu, LeakyReLu, Abs, Sqrt, Square).
    ActivationFunction m_Function;
    // Alpha upper bound value used by the activation functions. (BoundedReLu, Linear, TanH).
    float m_A;
    // Beta lower bound value used by the activation functions. (BoundedReLu, Linear, TanH).
    float m_B;
};
using NpuActivationFloat32Workload = NpuActivationWorkload<armnn::DataType::Float32>;
using NpuActivationFloat16Workload = NpuActivationWorkload<armnn::DataType::Float16>;
using NpuActivationUint8Workload = NpuActivationWorkload<armnn::DataType::QuantisedAsymm8>;
}  // namespace armnn
