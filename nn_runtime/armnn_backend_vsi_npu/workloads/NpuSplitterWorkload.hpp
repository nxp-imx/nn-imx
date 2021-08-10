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
#include <iostream>
#include "TNpuWorkloads.hpp"

namespace armnn {
template <typename armnn::DataType... DataTypes>
class NpuSplitterWorkload : public TNpuWorkload<SplitterQueueDescriptor, DataTypes...> {
   public:
    explicit NpuSplitterWorkload(const SplitterQueueDescriptor& descriptor,
                                 const WorkloadInfo& info)
        : TNpuWorkload<SplitterQueueDescriptor, DataTypes...>(descriptor, info) {
        std::vector<uint32_t> inputIds = this->AddOperandWithTensorHandle(descriptor.m_Inputs);

        // Compute split axis according to input shape and output shape
        NpuTensorHandler* inputTensorHandle =
            dynamic_cast<NpuTensorHandler*>(descriptor.m_Inputs[0]);
        NpuTensorHandler* outputTensorHandle0 =
            dynamic_cast<NpuTensorHandler*>(descriptor.m_Outputs[0]);
        m_Axis = 0;
        if (inputTensorHandle && outputTensorHandle0) {
            auto inputShape = inputTensorHandle->GetShape();
            auto output0Shape = outputTensorHandle0->GetShape();
            if (inputShape.GetNumDimensions() != output0Shape.GetNumDimensions()) {
                ARMNN_LOG(error) << "Mismatching input and output dimensions ("
                                 << inputShape.GetNumDimensions()
                                 << " != " << output0Shape.GetNumDimensions() << ").\n";
                assert(false);
            }
            if (descriptor.m_Outputs.size() > 1) {
                for (size_t i = 0; i < inputShape.GetNumDimensions(); ++i) {
                    if (inputShape[i] != output0Shape[i]) {
                        m_Axis = i;
                        break;
                    }
                }
            }
        }

        // Compute the slices according to outputs shape
        m_SliceNum = descriptor.m_Outputs.size();
        for (auto& output : descriptor.m_Outputs) {
            NpuTensorHandler* outputTensorHandle = dynamic_cast<NpuTensorHandler*>(output);
            if (outputTensorHandle) {
                auto outputShape = outputTensorHandle->GetShape();
                m_Slices.push_back(outputShape[m_Axis]);
            }
        }

        inputIds.push_back(this->AddOperandAndSetValue(m_Axis));
        inputIds.push_back(this->AddOperandAndSetValue(m_SliceNum));
        std::vector<uint32_t> sliceShape = {static_cast<uint32_t>(m_Slices.size())};
        inputIds.push_back(
            this->AddOperandAndSetValue(sliceShape, DataType::Signed32, m_Slices.data()));

        std::vector<uint32_t> outputIds = this->AddOperandWithTensorHandle(descriptor.m_Outputs);
        this->AddOperation(nnrt::OperationType::SPLIT,
                           inputIds.size(),
                           inputIds.data(),
                           outputIds.size(),
                           outputIds.data());
    }

   private:
    int32_t m_Axis;
    int32_t m_SliceNum;
    std::vector<int32_t> m_Slices;
};
using NpuSplitterFloat32Workload = NpuSplitterWorkload<armnn::DataType::Float32>;
using NpuSplitterFloat16Workload = NpuSplitterWorkload<armnn::DataType::Float16>;
using NpuSplitterUint8Workload = NpuSplitterWorkload<armnn::DataType::QAsymmU8>;
}  // namespace armnn
