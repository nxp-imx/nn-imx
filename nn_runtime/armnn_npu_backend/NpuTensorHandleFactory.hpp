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

#include <armnn/backends/IMemoryManager.hpp>
#include <armnn/backends/ITensorHandleFactory.hpp>
#include <forward_list>

namespace armnn {

constexpr const char* NpuTensorHandleFactoryId() {
    return "Arm/Npu/TensorHandleFactory";
}

class NpuMemoryManager : public IMemoryManager {
    public:
    NpuMemoryManager();
    virtual ~NpuMemoryManager();

    class Pool;

    Pool* Manage(unsigned int numBytes);

    void Allocate(Pool* pool);

    void* GetPointer(Pool* pool);

    void Acquire() override;
    void Release() override;

    class Pool {
    public:
        Pool(unsigned int numBytes);
        ~Pool();

        void Acquire();
        void Release();

        void* GetPointer();

        void Reserve(unsigned int numBytes);

    private:
        unsigned int m_Size;
        void* m_Pointer;
    };

    private:
    NpuMemoryManager(const NpuMemoryManager&) = delete;             // Noncopyable
    NpuMemoryManager& operator=(const NpuMemoryManager&) = delete;  // Noncopyable

    std::forward_list<Pool> m_Pools;
    std::vector<Pool*> m_FreePools;
};

class NpuTensorHandleFactory : public ITensorHandleFactory {
    public:
    NpuTensorHandleFactory(std::shared_ptr<NpuMemoryManager> mgr)
        : m_MemoryManager(mgr),
            m_ImportFlags(static_cast<MemorySourceFlags>(MemorySource::Malloc)),
            m_ExportFlags(static_cast<MemorySourceFlags>(MemorySource::Malloc)) {}

    std::unique_ptr<ITensorHandle> CreateSubTensorHandle(
        ITensorHandle& parent, TensorShape const& subTensorShape,
        unsigned int const* subTensorOrigin) const override;

    std::unique_ptr<ITensorHandle> CreateTensorHandle(
        const TensorInfo& tensorInfo) const override;

    std::unique_ptr<ITensorHandle> CreateTensorHandle(
        const TensorInfo& tensorInfo, DataLayout dataLayout) const override;

    std::unique_ptr<ITensorHandle> CreateTensorHandle(
        const TensorInfo& tensorInfo, const bool IsMemoryManaged) const override;

    std::unique_ptr<ITensorHandle> CreateTensorHandle(
        const TensorInfo& tensorInfo, DataLayout dataLayout,
        const bool IsMemoryManaged) const override;

    static const FactoryId& GetIdStatic();

    const FactoryId& GetId() const override;

    bool SupportsSubTensors() const override;

    MemorySourceFlags GetExportFlags() const override;

    MemorySourceFlags GetImportFlags() const override;

    private:
    mutable std::shared_ptr<NpuMemoryManager> m_MemoryManager;
    MemorySourceFlags m_ImportFlags;
    MemorySourceFlags m_ExportFlags;
};

}  // namespace armnn
