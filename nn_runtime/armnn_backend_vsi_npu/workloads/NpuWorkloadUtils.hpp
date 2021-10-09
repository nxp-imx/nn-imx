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

#include <backendsCommon/TensorHandle.hpp>

#include <armnn/Tensor.hpp>
#include <armnn/Types.hpp>
#include <Half.hpp>

namespace armnn
{

////////////////////////////////////////////
/// float32 helpers
////////////////////////////////////////////

inline const TensorInfo& GetTensorInfo(const ITensorHandle* tensorHandle)
{
    // We know that reference workloads use CpuTensorHandles only, so this cast is legitimate.
    const ConstCpuTensorHandle* cpuTensorHandle =
        PolymorphicDowncast<const ConstCpuTensorHandle*>(tensorHandle);
    return cpuTensorHandle->GetTensorInfo();
}

template <typename DataType>
inline const DataType* GetConstCpuData(const ITensorHandle* tensorHandle)
{
    // We know that reference workloads use (Const)CpuTensorHandles only, so this cast is legitimate.
    const ConstCpuTensorHandle* cpuTensorHandle =
        PolymorphicDowncast<const ConstCpuTensorHandle*>(tensorHandle);
    return cpuTensorHandle->GetConstTensor<DataType>();
}

template <typename DataType>
inline DataType* GetCpuData(const ITensorHandle* tensorHandle)
{
    // We know that reference workloads use CpuTensorHandles only, so this cast is legitimate.
    const CpuTensorHandle* cpuTensorHandle = PolymorphicDowncast<const CpuTensorHandle*>(tensorHandle);
    return cpuTensorHandle->GetTensor<DataType>();
};

template <typename DataType, typename PayloadType>
const DataType* GetInputTensorData(unsigned int idx, const PayloadType& data)
{
    const ITensorHandle* tensorHandle = data.m_Inputs[idx];
    return GetConstCpuData<DataType>(tensorHandle);
}

template <typename DataType, typename PayloadType>
DataType* GetOutputTensorData(unsigned int idx, const PayloadType& data)
{
    const ITensorHandle* tensorHandle = data.m_Outputs[idx];
    return GetCpuData<DataType>(tensorHandle);
}

template <typename PayloadType>
const float* GetInputTensorDataFloat(unsigned int idx, const PayloadType& data)
{
    return GetInputTensorData<float>(idx, data);
}

template <typename PayloadType>
float* GetOutputTensorDataFloat(unsigned int idx, const PayloadType& data)
{
    return GetOutputTensorData<float>(idx, data);
}

template <typename PayloadType>
const Half* GetInputTensorDataHalf(unsigned int idx, const PayloadType& data)
{
    return GetInputTensorData<Half>(idx, data);
}

template <typename PayloadType>
Half* GetOutputTensorDataHalf(unsigned int idx, const PayloadType& data)
{
    return GetOutputTensorData<Half>(idx, data);
}

////////////////////////////////////////////
/// u8 helpers
////////////////////////////////////////////

inline const uint8_t* GetConstCpuU8Data(const ITensorHandle* tensorHandle)
{
    // We know that reference workloads use (Const)CpuTensorHandles only, so this cast is legitimate.
    const ConstCpuTensorHandle* cpuTensorHandle =
        PolymorphicDowncast<const ConstCpuTensorHandle*>(tensorHandle);
    return cpuTensorHandle->GetConstTensor<uint8_t>();
};

inline uint8_t* GetCpuU8Data(const ITensorHandle* tensorHandle)
{
    // We know that reference workloads use CpuTensorHandles only, so this cast is legitimate.
    const CpuTensorHandle* cpuTensorHandle = PolymorphicDowncast<const CpuTensorHandle*>(tensorHandle);
    return cpuTensorHandle->GetTensor<uint8_t>();
};

template <typename PayloadType>
const uint8_t* GetInputTensorDataU8(unsigned int idx, const PayloadType& data)
{
    const ITensorHandle* tensorHandle = data.m_Inputs[idx];
    return GetConstCpuU8Data(tensorHandle);
}

template <typename PayloadType>
uint8_t* GetOutputTensorDataU8(unsigned int idx, const PayloadType& data)
{
    const ITensorHandle* tensorHandle = data.m_Outputs[idx];
    return GetCpuU8Data(tensorHandle);
}

template<typename T>
std::vector<float> Dequantize(const T* quant, const TensorInfo& info)
{
    std::vector<float> ret(info.GetNumElements());
    for (size_t i = 0; i < info.GetNumElements(); i++)
    {
        ret[i] = armnn::Dequantize(quant[i], info.GetQuantizationScale(), info.GetQuantizationOffset());
    }
    return ret;
}

template<typename T>
inline void Dequantize(const T* inputData, float* outputData, const TensorInfo& info)
{
    for (unsigned int i = 0; i < info.GetNumElements(); i++)
    {
        outputData[i] = Dequantize<T>(inputData[i], info.GetQuantizationScale(), info.GetQuantizationOffset());
    }
}

inline void Quantize(uint8_t* quant, const float* dequant, const TensorInfo& info)
{
    for (size_t i = 0; i < info.GetNumElements(); i++)
    {
        quant[i] = armnn::Quantize<uint8_t>(dequant[i], info.GetQuantizationScale(), info.GetQuantizationOffset());
    }
}

} //namespace armnn
