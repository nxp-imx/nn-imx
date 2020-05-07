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
class NpuPooling2dWorkload : public TNpuWorkload<Pooling2dQueueDescriptor, DataTypes...> {
   public:
    using base_type = TNpuWorkload<Pooling2dQueueDescriptor, DataTypes...>;
    explicit NpuPooling2dWorkload(const Pooling2dQueueDescriptor& descriptor,
                                  const WorkloadInfo& info)
        : TNpuWorkload<Pooling2dQueueDescriptor, DataTypes...>(descriptor, info),
          m_PoolType(descriptor.m_Parameters.m_PoolType),
          m_PadLeft(descriptor.m_Parameters.m_PadLeft),
          m_PadRight(descriptor.m_Parameters.m_PadRight),
          m_PadTop(descriptor.m_Parameters.m_PadTop),
          m_PadBottom(descriptor.m_Parameters.m_PadBottom),
          m_PoolWidth(descriptor.m_Parameters.m_PoolWidth),
          m_PoolHeight(descriptor.m_Parameters.m_PoolHeight),
          m_StrideX(descriptor.m_Parameters.m_StrideX),
          m_StrideY(descriptor.m_Parameters.m_StrideY),
          m_OutputShapeRounding(descriptor.m_Parameters.m_OutputShapeRounding),
          m_PaddingMethod(descriptor.m_Parameters.m_PaddingMethod),
          m_DataLayout(descriptor.m_Parameters.m_DataLayout) {

        constexpr uint32_t num_inputs = 12;
        uint32_t inputIds[num_inputs];
        // Add input operand
        auto inputPtr = dynamic_cast<NpuTensorHandler*>(descriptor.m_Inputs[0]);
        inputIds[0] = this->AddOperandAndSetValue(inputPtr->GetTensorInfo(), inputPtr->GetShape(), nullptr);
        // inputPtr->SetOperandId(inputIds[0]);

        // Add padding left operand
        inputIds[1] = this->AddOperandAndSetValue(m_PadLeft);

        // Add padding right operand
        inputIds[2] = this->AddOperandAndSetValue(m_PadRight);

        // Add padding top operand
        inputIds[3] = this->AddOperandAndSetValue(m_PadTop);

        // Add padding bottom operand
        inputIds[4] = this->AddOperandAndSetValue(m_PadBottom);

        // Add stride width operand
        inputIds[5] = this->AddOperandAndSetValue(m_StrideX);

        // Add stride height operand
        inputIds[6] = this->AddOperandAndSetValue(m_StrideY);

        // Add filter width operand
        inputIds[7] = this->AddOperandAndSetValue(m_PoolWidth);

        // Add filter height operand
        inputIds[8] = this->AddOperandAndSetValue(m_PoolHeight);

        // Add fuse operand
        int32_t fuseCode = 0;
        inputIds[9] = this->AddOperandAndSetValue(fuseCode);

        // Add layout operand
        int32_t layoutCode = m_DataLayout == armnn::DataLayout::NCHW
                                 ? int32_t(nnrt::DataLayout::NCHW)
                                 : int32_t(nnrt::DataLayout::NHWC);
        inputIds[10] = this->AddOperandAndSetValue(layoutCode);

        inputIds[11] = this->AddOperandAndSetValue(static_cast<int32_t>(m_OutputShapeRounding));

        // Add output operand
        int outputSize = descriptor.m_Outputs.size();
        uint32_t outputIds[outputSize];

        for (int i = 0; i < outputSize; i++) {
            auto outputPtr = dynamic_cast<NpuTensorHandler*>(descriptor.m_Outputs[i]);
            outputIds[i] = this->AddOperandAndSetValue(outputPtr->GetTensorInfo(), outputPtr->GetShape(), nullptr);
        }

        this->AddOperation(
            PoolingTypeConvert(m_PoolType), num_inputs, inputIds, outputSize, outputIds);
    }

   private:
    /// The pooling algorithm to use (Max. Average, L2).
    armnn::PoolingAlgorithm m_PoolType;
    /// Padding left value in the width dimension.
    uint32_t m_PadLeft;
    /// Padding right value in the width dimension.
    uint32_t m_PadRight;
    /// Padding top value in the height dimension.
    uint32_t m_PadTop;
    /// Padding bottom value in the height dimension.
    uint32_t m_PadBottom;
    /// Pooling width value.
    uint32_t m_PoolWidth;
    /// Pooling height value.
    uint32_t m_PoolHeight;
    /// Stride value when proceeding through input for the width dimension.
    uint32_t m_StrideX;
    /// Stride value when proceeding through input for the height dimension.
    uint32_t m_StrideY;
    /// The rounding method for the output shape. (Floor, Ceiling).
    armnn::OutputShapeRounding m_OutputShapeRounding;
    /// The padding method to be used. (Exclude, IgnoreValue).
    armnn::PaddingMethod m_PaddingMethod;
    /// The data layout to be used (NCHW, NHWC).
    armnn::DataLayout m_DataLayout;

    nnrt::OperationType PoolingTypeConvert(armnn::PoolingAlgorithm type) {
        switch (type) {
            case armnn::PoolingAlgorithm::Max:
                return nnrt::OperationType::MAX_POOL_2D;
                break;
            case armnn::PoolingAlgorithm::Average:
                return nnrt::OperationType::AVERAGE_POOL_2D;
                break;
            case armnn::PoolingAlgorithm::L2:
                return nnrt::OperationType::L2_POOL_2D;
            default:
                BOOST_LOG_TRIVIAL(error) << "Pooling type not support.";
                break;
        }

        return nnrt::OperationType::NONE;
    }
};
using NpuPooling2dFloat32Workload = NpuPooling2dWorkload<armnn::DataType::Float32>;
using NpuPooling2dFloat16Workload = NpuPooling2dWorkload<armnn::DataType::Float16>;
using NpuPooling2dUint8Workload = NpuPooling2dWorkload<armnn::DataType::QAsymmU8>;
}  // namespace armnn
