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
class NpuLstmWorkload : public TNpuWorkload<LstmQueueDescriptor, DataTypes...> {
   public:
    explicit NpuLstmWorkload(const LstmQueueDescriptor& descriptor,
                                  const WorkloadInfo& info)
        : TNpuWorkload<LstmQueueDescriptor, DataTypes...>(descriptor, info)
        , m_ActivationFunc((ActivationFunction)descriptor.m_Parameters.m_ActivationFunc)
        , m_ClippingThresCell(descriptor.m_Parameters.m_ClippingThresCell)
        , m_ClippingThresProj(descriptor.m_Parameters.m_ClippingThresProj) {
        bool usePeephole = descriptor.m_Parameters.m_PeepholeEnabled;
        bool useCifg = descriptor.m_Parameters.m_CifgEnabled;
        bool useProj = descriptor.m_Parameters.m_ProjectionEnabled;

        std::vector<uint32_t> inputIds = this->AddOperandWithTensorHandle(descriptor.m_Inputs);
        inputIds.resize(23);
        std::vector<const ConstCpuTensorHandle*> weightTensors;
        weightTensors.push_back(!useCifg ? descriptor.m_InputToInputWeights : nullptr);
        weightTensors.push_back(descriptor.m_InputToForgetWeights);
        weightTensors.push_back(descriptor.m_InputToCellWeights);
        weightTensors.push_back(descriptor.m_InputToOutputWeights);
        weightTensors.push_back(!useCifg ? descriptor.m_RecurrentToInputWeights : nullptr);
        weightTensors.push_back(descriptor.m_RecurrentToForgetWeights);
        weightTensors.push_back(descriptor.m_RecurrentToCellWeights);
        weightTensors.push_back(descriptor.m_RecurrentToOutputWeights);
        weightTensors.push_back(
               !useCifg && usePeephole ? descriptor.m_CellToInputWeights : nullptr);
        weightTensors.push_back(usePeephole ? descriptor.m_CellToForgetWeights : nullptr);
        weightTensors.push_back(usePeephole ? descriptor.m_CellToOutputWeights : nullptr);
        weightTensors.push_back(!useCifg ? descriptor.m_InputGateBias : nullptr);
        weightTensors.push_back(descriptor.m_ForgetGateBias);
        weightTensors.push_back(descriptor.m_CellBias);
        weightTensors.push_back(descriptor.m_OutputGateBias);
        weightTensors.push_back(useProj ? descriptor.m_ProjectionWeights : nullptr);
        weightTensors.push_back(useProj ? descriptor.m_ProjectionBias : nullptr);
        std::vector<uint32_t> weightIds = this->AddOperandWithTensorHandle(weightTensors);
        for (uint32_t i = 0; i < weightIds.size(); i ++) {
            inputIds[i + 3] = weightIds[i];
        }

        // Armnn uses the same activation enum value as Android NNAPI
        //int32_t activation = (int32_t)nnrt::FusedType::TANH;
        //switch (m_ActivationFunc) {
        //    case ActivationFunction::Sigmoid:
        //        activation = (int32_t)nnrt::FusedType::SIGMOID;
        //        break;
        //    case ActivationFunction::TanH:
        //        activation = (int32_t)nnrt::FusedType::TANH;
        //        break;
        //    default:
        //        // Not support
        //        assert(false);
        //        break;
        //}
        inputIds[20] = this->AddOperandAndSetValue((int32_t)m_ActivationFunc);
        inputIds[21] = this->AddOperandAndSetValue(m_ClippingThresCell);
        inputIds[22] = this->AddOperandAndSetValue(m_ClippingThresProj);

        std::vector<uint32_t> outputIds = this->AddOperandWithTensorHandle(
                descriptor.m_Outputs);
        this->AddOperation(nnrt::OperationType::LSTM_UNIT,
                inputIds.size(), inputIds.data(), outputIds.size(), outputIds.data());
    }

   private:
    ActivationFunction m_ActivationFunc;
    float m_ClippingThresCell;
    float m_ClippingThresProj;
};
using NpuLstmFloat32Workload = NpuLstmWorkload<armnn::DataType::Float32>;
using NpuLstmFloat16Workload = NpuLstmWorkload<armnn::DataType::Float16>;
using NpuLstmUint8Workload = NpuLstmWorkload<armnn::DataType::QuantisedAsymm8>;
}  // namespace armnn
