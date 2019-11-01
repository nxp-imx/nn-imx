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
class NpuStridedSliceWorkload : public TNpuWorkload<StridedSliceQueueDescriptor, DataTypes...> {
   public:
    explicit NpuStridedSliceWorkload(const StridedSliceQueueDescriptor& descriptor,
                                  const WorkloadInfo& info)
        : TNpuWorkload<StridedSliceQueueDescriptor, DataTypes...>(descriptor, info),
        m_Begin(descriptor.m_Parameters.m_Begin),
        m_End(descriptor.m_Parameters.m_End),
        m_Stride(descriptor.m_Parameters.m_Stride),
        m_BeginMask(descriptor.m_Parameters.m_BeginMask),
        m_EndMask(descriptor.m_Parameters.m_EndMask),
        m_ShrinkAxisMask(descriptor.m_Parameters.m_ShrinkAxisMask),
        m_DataLayout(descriptor.m_Parameters.m_DataLayout) {
        std::vector<uint32_t> inputIds = this->AddOperandWithTensorHandle(
                descriptor.m_Inputs);

        std::vector<uint32_t> begin_shape = {(uint32_t)m_Begin.size()};
        std::vector<uint32_t> end_shape = {(uint32_t)m_End.size()};
        std::vector<uint32_t> stride_shape = {(uint32_t)m_Stride.size()};
        inputIds.push_back(this->AddOperandAndSetValue(
                    begin_shape, DataType::Signed32, m_Begin.data()));
        inputIds.push_back(this->AddOperandAndSetValue(
                    end_shape, DataType::Signed32, m_End.data()));
        inputIds.push_back(this->AddOperandAndSetValue(
                    stride_shape, DataType::Signed32, m_Stride.data()));
        inputIds.push_back(this->AddOperandAndSetValue(m_BeginMask));
        inputIds.push_back(this->AddOperandAndSetValue(m_EndMask));
        inputIds.push_back(this->AddOperandAndSetValue(m_ShrinkAxisMask));
        // Add layout operand
        int32_t layoutCode = m_DataLayout == armnn::DataLayout::NCHW
                                 ? int32_t(nnrt::DataLayout::NCHW)
                                 : int32_t(nnrt::DataLayout::NHWC);
        inputIds.push_back(this->AddOperandAndSetValue(layoutCode));

        std::vector<uint32_t> outputIds = this->AddOperandWithTensorHandle(
                descriptor.m_Outputs);

        this->AddOperation(nnrt::OperationType::STRIDED_SLICE,
                inputIds.size(), inputIds.data(), outputIds.size(), outputIds.data());
    }

   private:
    std::vector<int32_t> m_Begin;
    std::vector<int32_t> m_End;
    std::vector<int32_t> m_Stride;
    int32_t m_BeginMask;
    int32_t m_EndMask;
    int32_t m_ShrinkAxisMask;
    DataLayout m_DataLayout;

};
using NpuStridedSliceFloat32Workload = NpuStridedSliceWorkload<armnn::DataType::Float32>;
using NpuStridedSliceFloat16Workload = NpuStridedSliceWorkload<armnn::DataType::Float16>;
using NpuStridedSliceUint8Workload = NpuStridedSliceWorkload<armnn::DataType::QuantisedAsymm8>;
}  // namespace armnn
