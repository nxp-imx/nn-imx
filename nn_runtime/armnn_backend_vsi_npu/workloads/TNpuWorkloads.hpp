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

#include "NpuBackend.hpp"
#include "NpuTensorHandler.hpp"

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadInfo.hpp>

#include "NpuModelShell.hpp"

#include <iostream>
#include <type_traits>

namespace armnn {
namespace vnn_helper {
template <typename CPPDtype>
struct ScalarTypeSelector {
    static constexpr nnrt::OperandType type = nnrt::OperandType::NONE;
};

template <>
struct ScalarTypeSelector<float> {
    static constexpr nnrt::OperandType type = nnrt::OperandType::FLOAT32;
};

template <>
struct ScalarTypeSelector<int> {
    static constexpr nnrt::OperandType type = nnrt::OperandType::INT32;
};

template <>
struct ScalarTypeSelector<bool> {
    static constexpr nnrt::OperandType type = nnrt::OperandType::BOOL;
};

template <>
struct ScalarTypeSelector<unsigned int> {
    static constexpr nnrt::OperandType type = nnrt::OperandType::UINT32;
};

template <bool c, typename true_type, typename false_type>
struct If_ {
    using type = true_type;
};

template <typename true_type, typename false_type>
struct If_<false, true_type, false_type> {
    using type = false_type;
};

template <bool c, typename t, typename f>
using If_t = typename If_<c, t, f>::type;

template <bool c, armnn::DataType v0, armnn::DataType v1>
struct If_v {
    static constexpr armnn::DataType value = v0;
};

template <armnn::DataType v0, armnn::DataType v1>
struct If_v<false, v0, v1> {
    static constexpr armnn::DataType value = v1;
};

template <uint32_t idx, uint32_t size, armnn::DataType FirstDType, armnn::DataType... DataTypes>
struct At_impl_ {
    static constexpr armnn::DataType value =
        If_v<idx <= sizeof...(DataTypes),
             At_impl_<idx - 1, sizeof...(DataTypes), DataTypes...>::value,
             (armnn::DataType)(-1)>::value;
};

/*
 specialize for access last item in the original DataTypes...
*/
template <armnn::DataType FirstDType, armnn::DataType... DataTypes>
struct At_impl_<0, 1, FirstDType, DataTypes...> {
    static constexpr armnn::DataType value = FirstDType;
};

template <uint32_t idx, armnn::DataType FirstDType, armnn::DataType... DataTypes>
struct At_impl_<idx, 1, FirstDType, DataTypes...> {
    // This specilize make compiler not complain
    static constexpr armnn::DataType value = (armnn::DataType)(-1);
};

template <uint32_t size, armnn::DataType FirstDType, armnn::DataType... DataTypes>
struct At_impl_<0, size, FirstDType, DataTypes...> {
    static constexpr armnn::DataType value = FirstDType;
};

template <uint32_t idx, armnn::DataType... DataTypes>
struct At_ {
    static constexpr armnn::DataType value = At_impl_<idx, sizeof...(DataTypes), DataTypes...>::value;
};
}

template <typename QueueDescriptor, armnn::DataType... DataTypes>
using NpuBaseWorkload = vnn_helper::If_t<sizeof...(DataTypes) == 2,
                                         MultiTypedWorkload<QueueDescriptor,
                                                            vnn_helper::At_<0, DataTypes...>::value,
                                                            vnn_helper::At_<1, DataTypes...>::value>,
                                         TypedWorkload<QueueDescriptor, DataTypes...>>;

template <typename QueueDescriptor, armnn::DataType... DataTypes>
struct Base{
    using type = NpuBaseWorkload<QueueDescriptor, DataTypes...>;
};

template <typename QueueDescriptor, armnn::DataType... DataTypes>
using Base_t = typename Base<QueueDescriptor, DataTypes...>::type;

template <typename QueueDescriptor, armnn::DataType... DataTypes>
class TNpuWorkload : public Base_t<QueueDescriptor, DataTypes...>{
   public:
    explicit TNpuWorkload(const QueueDescriptor& descriptor, const WorkloadInfo& info)
        : Base_t<QueueDescriptor, DataTypes...>(descriptor, info) {
        for (size_t i = 0; i < descriptor.m_Inputs.size(); i++) {
            NpuTensorHandler* inputTensorHandle =
                dynamic_cast<NpuTensorHandler*>(descriptor.m_Inputs[i]);
            if (inputTensorHandle) {
                m_InputsHandler.push_back(inputTensorHandle);
            }
        }
        for (size_t i = 0; i < descriptor.m_Outputs.size(); i++) {
            NpuTensorHandler* outputTensorHandle =
                dynamic_cast<NpuTensorHandler*>(descriptor.m_Outputs[i]);
            if (outputTensorHandle) {
                m_OutputsHandler.push_back(outputTensorHandle);
            }
        }
        m_InputTensorInfos = info.m_InputTensorInfos;
        m_OutputTensorInfos = info.m_OutputTensorInfos;

        // Create Model for current Workload, put input/output information to local model
        m_LocalModel = std::make_shared<nnrt::Model>();
        m_Executed = false;
    }

    void Execute() const override {

        // Our workload don't need executed repeatly, we just need construct final model
        // which deployed to our backend. But, if armnn also need support dynamic graph
        // we need re-eval every workload each inference and construct final model accordingly
        if (m_Executed) return;
        m_Executed = true;

        adaption::InOutTensorHandles inoutTensorHandles;
        inoutTensorHandles.first.reserve(m_InputsHandler.size());
        for (NpuTensorHandler* inputHandle : m_InputsHandler) {
            inoutTensorHandles.first.push_back(inputHandle);
        }
        inoutTensorHandles.second.reserve(m_OutputsHandler.size());
        for (NpuTensorHandler* outputHandle : m_OutputsHandler) {
            inoutTensorHandles.second.push_back(outputHandle);
        }

        auto local_model = std::make_pair(m_LocalModel, inoutTensorHandles);
        // ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuAcc, "TNpuWorkload_Model_Spreading");
        for (NpuTensorHandler* outputHandle : m_OutputsHandler) {
            adaption::ModelStack& curModelStack = outputHandle->editModelStack();
            curModelStack.insert(local_model);

            for (NpuTensorHandler* inputHandle : m_InputsHandler) {
                std::copy(inputHandle->ModelStack().begin(), inputHandle->ModelStack().end(),
                std::inserter(curModelStack, curModelStack.end()));
            }
        }
    }

   protected:
    nnrt::OperandType convertToOperandType(DataType dtype) {
        nnrt::OperandType type = nnrt::OperandType::NONE;
        switch (dtype) {
            case DataType::Float32:
                type = nnrt::OperandType::TENSOR_FLOAT32;
                break;
            case DataType::Float16:
                type = nnrt::OperandType::TENSOR_FLOAT16;
                break;
            case DataType::QAsymmU8:
                type = nnrt::OperandType::TENSOR_QUANT8_ASYMM;
                break;
            case DataType::QAsymmS8:
                type = nnrt::OperandType::TENSOR_QUANT8_ASYMM_SIGNED;
                break;
            case DataType::QSymmS8:
                type = nnrt::OperandType::TENSOR_QUANT8_SYMM;
                break;
            case DataType::Signed32:
                type = nnrt::OperandType::TENSOR_INT32;
                break;
            case DataType::QSymmS16:
                type = nnrt::OperandType::TENSOR_INT16;
                break;
            case DataType::Boolean:
                type = nnrt::OperandType::TENSOR_BOOL8;
            default:
                break;
        }
        return type;
    }

    uint32_t AddOperandAndSetValue(const std::vector<uint32_t>& shape,
                                   DataType dtype,
                                   const void* valueAddr) {
        uint32_t operandId{0};
        nnrt::op::OperandPtr operand = m_LocalModel->addOperand(nullptr, &operandId);
        operand->type = convertToOperandType(dtype);
        operand->dimensions.assign(shape.begin(), shape.end());
        m_LocalModel->setOperandValue(operandId, valueAddr, operand->bytes());
        return operandId;
    }

    uint32_t AddOperandAndSetValue(const TensorInfo& info,
                                   const TensorShape& shape,
                                   const void* valueAddr,
                                   bool floatToInt = false) {
        std::vector<uint32_t> dims(shape.GetNumDimensions());
        for (unsigned int i = 0; i < shape.GetNumDimensions(); i++) {
            dims[i] = shape[i];
        }

        uint32_t operandId{0};
        nnrt::op::OperandPtr operand = m_LocalModel->addOperand(nullptr, &operandId);
        {
            auto dataType = info.GetDataType();
            operand->type = convertToOperandType(dataType);
            if (floatToInt) {
                operand->type = convertToOperandType(DataType::Signed32);
            }
            assert(operand->type != nnrt::OperandType::NONE);

            operand->dimensions = dims;
            if (info.HasPerAxisQuantization()) {
                operand->quant.vec.channelDim = info.GetQuantizationDim().value();
                operand->quant.vec.scale = info.GetQuantizationScales();
                std::vector<int32_t> zeroPoint(info.GetQuantizationScales().size());
                std::fill(zeroPoint.begin(), zeroPoint.end(), info.GetQuantizationOffset());
                operand->quant.vec.zeroPoint = std::move(zeroPoint);
            } else if (info.IsQuantized()) {
                operand->quant.scalar.scale = info.GetQuantizationScale();
                operand->quant.scalar.zeroPoint = info.GetQuantizationOffset();
            }
        }
        m_LocalModel->setOperandValue(operandId, valueAddr, info.GetNumBytes());

        return operandId;
    }

    template <typename DType>
    uint32_t AddOperandAndSetValue(DType value) {
        constexpr nnrt::OperandType data_type = vnn_helper::ScalarTypeSelector<DType>::type;

        static_assert(data_type != nnrt::OperandType::NONE, "Add your datatype support");

        uint32_t operandId{0};
        nnrt::op::OperandPtr operand = m_LocalModel->addOperand(nullptr, &operandId);
        operand->type = data_type;
        m_LocalModel->setOperandValue(operandId, &value, sizeof(DType));

        return operandId;
    }

    std::vector<uint32_t> AddOperandWithTensorHandle(const std::vector<ITensorHandle*>& tensors) {
        std::vector<uint32_t> operandIds;
        for (uint32_t i = 0; i < tensors.size(); i++) {
            const NpuTensorHandler* tensorHandle = dynamic_cast<const NpuTensorHandler*>(tensors[i]);
            uint32_t operandId = AddOperandWithTensorHandle(tensorHandle);
            operandIds.push_back(operandId);
        }
        return operandIds;
    }

    std::vector<uint32_t> AddOperandWithTensorHandle(
        const std::vector<const ConstCpuTensorHandle*>& tensors) {
        std::vector<uint32_t> operandIds;
        for (uint32_t i = 0; i < tensors.size(); i++) {
            const ScopedCpuTensorHandle* tensorHandle =
                dynamic_cast<const ScopedCpuTensorHandle*>(tensors[i]);
            uint32_t operandId = AddOperandWithTensorHandle(tensorHandle);
            operandIds.push_back(operandId);
        }
        return operandIds;
    }

    uint32_t AddOperandWithTensorHandle(const ScopedCpuTensorHandle* tensor) {
        uint32_t operandId;
        if (tensor != nullptr) {
            operandId = this->AddOperandAndSetValue(
                tensor->GetTensorInfo(), tensor->GetShape(), tensor->GetTensor<void>());
        } else {
            operandId = AddNullOperand();
        }
        return operandId;
    }

    uint32_t AddOperandWithTensorHandle(const NpuTensorHandler* tensor) {
        uint32_t operandId;
        if (tensor != nullptr) {
            operandId = this->AddOperandAndSetValue(tensor->GetTensorInfo(), tensor->GetShape(), nullptr);
        } else {
            operandId = AddNullOperand();
        }
        return operandId;
    }

    uint32_t AddNullOperand(DataType dtype = DataType::Float32) {
        std::vector<uint32_t> scratchBufferShape = {0};
        uint32_t operandId = this->AddOperandAndSetValue(scratchBufferShape, dtype, nullptr);
        return operandId;
    }

    uint32_t AddOperation(const nnrt::OperationType& opType,
                          uint32_t numIn,
                          const uint32_t* inOperands,
                          uint32_t numOut,
                          const uint32_t* outOperands) {
        uint32_t index{0};
        nnrt::op::OperationPtr op =
            m_LocalModel->addOperation(opType, inOperands, numIn, outOperands, numOut, &index);
        if (!op) {
            std::cout << "Out of memory.";
        }

        // op->setOperandLayout(); TODO: setup layout

        return index;
    }

    std::vector<NpuTensorHandler*> m_InputsHandler;
    std::vector<NpuTensorHandler*> m_OutputsHandler;

   private:
    std::vector<TensorInfo> m_InputTensorInfos;
    std::vector<TensorInfo> m_OutputTensorInfos;

    nnrt::ModelPtr m_LocalModel;
    mutable bool m_Executed;
};

}  // namespace  armnn
