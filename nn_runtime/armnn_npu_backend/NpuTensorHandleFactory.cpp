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


#include "NpuTensorHandleFactory.hpp"
#include "NpuTensorHandler.hpp"

#include "Layer.hpp"

#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/NumericCast.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

namespace armnn
{

NpuMemoryManager::NpuMemoryManager()
{}

NpuMemoryManager::~NpuMemoryManager()
{}

NpuMemoryManager::Pool* NpuMemoryManager::Manage(unsigned int numBytes)
{
    if (!m_FreePools.empty())
    {
        Pool* res = m_FreePools.back();
        m_FreePools.pop_back();
        res->Reserve(numBytes);
        return res;
    }
    else
    {
        m_Pools.push_front(Pool(numBytes));
        return &m_Pools.front();
    }
}

void NpuMemoryManager::Allocate(NpuMemoryManager::Pool* pool)
{
    ARMNN_ASSERT(pool);
    m_FreePools.push_back(pool);
}

void* NpuMemoryManager::GetPointer(NpuMemoryManager::Pool* pool)
{
    return pool->GetPointer();
}

void NpuMemoryManager::Acquire()
{
    for (Pool &pool: m_Pools)
    {
         pool.Acquire();
    }
}

void NpuMemoryManager::Release()
{
    for (Pool &pool: m_Pools)
    {
         pool.Release();
    }
}

NpuMemoryManager::Pool::Pool(unsigned int numBytes)
    : m_Size(numBytes),
      m_Pointer(nullptr)
{}

NpuMemoryManager::Pool::~Pool()
{
    if (m_Pointer)
    {
        Release();
    }
}

void* NpuMemoryManager::Pool::GetPointer()
{
    ARMNN_ASSERT_MSG(m_Pointer, "NpuMemoryManager::Pool::GetPointer() called when memory not acquired");
    return m_Pointer;
}

void NpuMemoryManager::Pool::Reserve(unsigned int numBytes)
{
    ARMNN_ASSERT_MSG(!m_Pointer, "NpuMemoryManager::Pool::Reserve() cannot be called after memory acquired");
    m_Size = std::max(m_Size, numBytes);
}

void NpuMemoryManager::Pool::Acquire()
{
    ARMNN_ASSERT_MSG(!m_Pointer, "NpuMemoryManager::Pool::Acquire() called when memory already acquired");
    m_Pointer = ::operator new(size_t(m_Size));
}

void NpuMemoryManager::Pool::Release()
{
    ARMNN_ASSERT_MSG(m_Pointer, "NpuMemoryManager::Pool::Release() called when memory not acquired");
    ::operator delete(m_Pointer);
    m_Pointer = nullptr;
}

using FactoryId = ITensorHandleFactory::FactoryId;

std::unique_ptr<ITensorHandle> NpuTensorHandleFactory::CreateSubTensorHandle(ITensorHandle& parent,
                                                                              const TensorShape& subTensorShape,
                                                                              const unsigned int* subTensorOrigin)
                                                                              const
{
    IgnoreUnused(parent, subTensorShape, subTensorOrigin);
    return nullptr;
}

std::unique_ptr<ITensorHandle> NpuTensorHandleFactory::CreateTensorHandle(const TensorInfo& tensorInfo) const
{
    return std::make_unique<NpuTensorHandler>(tensorInfo);
}

std::unique_ptr<ITensorHandle> NpuTensorHandleFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                           DataLayout dataLayout) const
{
    IgnoreUnused(dataLayout);
    return std::make_unique<NpuTensorHandler>(tensorInfo);
}

std::unique_ptr<ITensorHandle> NpuTensorHandleFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                           const bool IsMemoryManaged) const
{
    IgnoreUnused(IsMemoryManaged);
    return std::make_unique<NpuTensorHandler>(tensorInfo);
}

std::unique_ptr<ITensorHandle> NpuTensorHandleFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                           DataLayout dataLayout,
                                                                           const bool IsMemoryManaged) const
{
    IgnoreUnused(dataLayout);
    IgnoreUnused(IsMemoryManaged);
    return std::make_unique<NpuTensorHandler>(tensorInfo);
}

const FactoryId& NpuTensorHandleFactory::GetIdStatic()
{
    static const FactoryId s_Id(NpuTensorHandleFactoryId());
    return s_Id;
}

const FactoryId& NpuTensorHandleFactory::GetId() const
{
    return GetIdStatic();
}

bool NpuTensorHandleFactory::SupportsSubTensors() const
{
    return false;
}

MemorySourceFlags NpuTensorHandleFactory::GetExportFlags() const
{
    return m_ExportFlags;
}

MemorySourceFlags NpuTensorHandleFactory::GetImportFlags() const
{
    return m_ImportFlags;
}

} // namespace armnn