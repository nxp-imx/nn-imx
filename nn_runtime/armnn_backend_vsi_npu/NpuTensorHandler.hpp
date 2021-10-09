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

#include <functional>
#include <armnn/Tensor.hpp>
#include <backendsCommon/ITensorHandle.hpp>

// TODO: include proper one
#include "NpuModelShell.hpp"

namespace armnn {
using func = std::function<void(void*, size_t)>;
class NpuTensorHandler : public ITensorHandle {
   public:
    NpuTensorHandler(const TensorInfo& info)
        : m_OperandId(0xFFFFFFFF),
          m_TensorInfo(info),
          m_ExternalMem(nullptr){}

    ~NpuTensorHandler() {}

    virtual void Manage() override {
        // We don't need this right now
    }

    virtual void Allocate() override;
    virtual ITensorHandle* GetParent() const override {
        // Not support yet
        // TODO: learn more about concat
        return nullptr;
    }

    virtual void* Map(bool blocking = true);

    virtual const void* Map(bool blocking = true) const override;

    virtual void Unmap() const override;

    virtual void Unmap();

    virtual TensorShape GetStrides() const override;

    virtual TensorShape GetShape() const override;

    virtual void CopyOutTo(void* memory) const override;

    virtual void CopyInFrom(const void* memory) override;

    void* GetMemArea();

    void SetOperandId(unsigned int index);

    unsigned int GetOperandId() const { return m_OperandId; }

    // None-Inheriented Interface for Backend internal purpose
    bool HasMemory() const { return m_Memory.size() ? true : false; }

    const adaption::ModelStack& ModelStack() { return m_ModelStack; }

    adaption::ModelStack& editModelStack() { return m_ModelStack; }

    void* data() {
        if (m_ExternalMem) {
            return m_ExternalMem;
        } else {
            assert(m_Memory.size());
            return m_Memory.data();
        }
    }

    unsigned int memSize() { return m_TensorInfo.GetNumBytes(); }

    const TensorInfo& GetTensorInfo() const { return m_TensorInfo; }

    bool IsOperandIdValid() const { return m_OperandId != m_InValidOperandId; }

    void SetOperandIdInValid() { m_OperandId = m_InValidOperandId; }

    virtual bool Import(void* memory, MemorySource source) override;

    MemorySourceFlags GetImportFlags() const override;

    func callback = nullptr;

   private:
   /**
    * @brief Get the Memory Ready
    *
    */
    void getMemoryReady() const;

    /**
     * @brief shared ModelShell resource between multipy-output workload
     *  with this, multipy-output won't create ModelShell repeatly
     *
     * @return armnn::ModelShellPtr
     */
    armnn::ModelShellPtr shareModelShell() {
        return m_ModelShell;
    }

   private:
    uint32_t m_OperandId;
    TensorInfo m_TensorInfo;
    mutable std::vector<uint8_t> m_Memory;
    void* m_ExternalMem;
    mutable bool dirty_flag = false;

    mutable adaption::ModelStack m_ModelStack;  //!< TensorHandler doesn't create this, we pass through
                                                // the local model hosted by workload to following workload
                                                // This help us make decesion to create final executable model

    mutable armnn::ModelShellPtr m_ModelShell;
    static constexpr uint32_t m_InValidOperandId{0xFFFFFFFF};
};
}
