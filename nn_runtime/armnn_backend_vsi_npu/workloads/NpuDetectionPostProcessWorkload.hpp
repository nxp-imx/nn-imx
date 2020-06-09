/****************************************************************************
*
*    Copyright (c) 2020 Vivante Corporation
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

static void ConvertIntToFloat(void* ptr, size_t len) {
    if (!ptr) return;
    std::vector<float> tmp;
    for (size_t i = 0; i < len / sizeof(int32_t); i++) {
        tmp.push_back(static_cast<float>(*((int32_t*)ptr + i)));
    }
    std::memcpy(ptr, tmp.data(), len);
}

namespace armnn {
template <typename armnn::DataType... DataTypes>
class NpuDetectionPostProcessWorkload : public TNpuWorkload<DetectionPostProcessQueueDescriptor, DataTypes...> {
   public:
    using base_type = TNpuWorkload<DetectionPostProcessQueueDescriptor, DataTypes...>;
    explicit NpuDetectionPostProcessWorkload(const DetectionPostProcessQueueDescriptor& descriptor,
                                     const WorkloadInfo& info)
        : TNpuWorkload<DetectionPostProcessQueueDescriptor, DataTypes...>(descriptor, info),
          m_Anchors(std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Anchors))),
          m_MaxDetections(descriptor.m_Parameters.m_MaxDetections),
          m_MaxClassesPerDetection(descriptor.m_Parameters.m_MaxClassesPerDetection),
          m_DetectionsPerClass(descriptor.m_Parameters.m_DetectionsPerClass),
          m_NmsScoreThreshold(descriptor.m_Parameters.m_NmsScoreThreshold),
          m_NmsIouThreshold(descriptor.m_Parameters.m_NmsIouThreshold),
          m_NumClasses(descriptor.m_Parameters.m_NumClasses),
          m_UseRegularNms(descriptor.m_Parameters.m_UseRegularNms),
          m_ScaleX(descriptor.m_Parameters.m_ScaleX),
          m_ScaleY(descriptor.m_Parameters.m_ScaleY),
          m_ScaleW(descriptor.m_Parameters.m_ScaleW),
          m_ScaleH(descriptor.m_Parameters.m_ScaleH) {
        std::vector<uint32_t> inOperandIds;
        // Add input operand
        auto scoresPtr = dynamic_cast<NpuTensorHandler*>(descriptor.m_Inputs[1]);
        inOperandIds.push_back(this->AddOperandAndSetValue(
            scoresPtr->GetTensorInfo(), scoresPtr->GetShape(), nullptr));
        auto boxEncodingPtr = dynamic_cast<NpuTensorHandler*>(descriptor.m_Inputs[0]);
        inOperandIds.push_back(this->AddOperandAndSetValue(
            boxEncodingPtr->GetTensorInfo(), boxEncodingPtr->GetShape(), nullptr));

        // swap input npu tensor hanlder
        auto tmp = this->m_InputsHandler[0];
        this->m_InputsHandler[0] = this->m_InputsHandler[1];
        this->m_InputsHandler[1] = tmp;

        inOperandIds.push_back(this->AddOperandWithTensorHandle(m_Anchors.get()));

        inOperandIds.push_back(this->AddOperandAndSetValue(m_ScaleY));
        inOperandIds.push_back(this->AddOperandAndSetValue(m_ScaleX));
        inOperandIds.push_back(this->AddOperandAndSetValue(m_ScaleH));
        inOperandIds.push_back(this->AddOperandAndSetValue(m_ScaleW));
        inOperandIds.push_back(this->AddOperandAndSetValue(m_UseRegularNms));
        inOperandIds.push_back(this->AddOperandAndSetValue(m_MaxDetections));
        inOperandIds.push_back(this->AddOperandAndSetValue(m_MaxClassesPerDetection));
        inOperandIds.push_back(this->AddOperandAndSetValue(m_DetectionsPerClass));
        inOperandIds.push_back(this->AddOperandAndSetValue(m_NmsScoreThreshold));
        inOperandIds.push_back(this->AddOperandAndSetValue(m_NmsIouThreshold));
        inOperandIds.push_back(this->AddOperandAndSetValue(m_NumClasses));

        // Add output operand
        std::vector<uint32_t> outOperandIds;
        auto detectionScoresPtr = dynamic_cast<NpuTensorHandler*>(descriptor.m_Outputs[2]);
        outOperandIds.push_back(this->AddOperandAndSetValue(
            detectionScoresPtr->GetTensorInfo(), detectionScoresPtr->GetShape(), nullptr));

        auto detectionBoxesPtr = dynamic_cast<NpuTensorHandler*>(descriptor.m_Outputs[0]);
        outOperandIds.push_back(this->AddOperandAndSetValue(
            detectionBoxesPtr->GetTensorInfo(), detectionBoxesPtr->GetShape(), nullptr));

        auto detectionClassesPtr = dynamic_cast<NpuTensorHandler*>(descriptor.m_Outputs[1]);
        // convert float to int to compatible with nnapi spec
        outOperandIds.push_back(this->AddOperandAndSetValue(detectionClassesPtr->GetTensorInfo(),
                                                            detectionClassesPtr->GetShape(),
                                                            nullptr,
                                                            true /*convert float to int32*/));
        detectionClassesPtr->callback = &ConvertIntToFloat;

        auto detectionNumsPtr = dynamic_cast<NpuTensorHandler*>(descriptor.m_Outputs[3]);
        outOperandIds.push_back(this->AddOperandAndSetValue(detectionNumsPtr->GetTensorInfo(),
                                                            detectionNumsPtr->GetShape(),
                                                            nullptr,
                                                            true /*convert float to int32*/));
        detectionNumsPtr->callback = &ConvertIntToFloat;

        // swap output npu handler
        tmp = this->m_OutputsHandler[0];
        this->m_OutputsHandler[0] = this->m_OutputsHandler[2];
        this->m_OutputsHandler[2] = tmp;

        tmp = this->m_OutputsHandler[1];
        this->m_OutputsHandler[1] = this->m_OutputsHandler[2];
        this->m_OutputsHandler[2] = tmp;

        this->AddOperation(nnrt::OperationType::DETECTION_POSTPROCESSING,
                           inOperandIds.size(),
                           inOperandIds.data(),
                           outOperandIds.size(),
                           outOperandIds.data());
    }

   private:
    std::unique_ptr<ScopedCpuTensorHandle> m_Anchors;
    /// Maximum numbers of detections.
    unsigned int m_MaxDetections;
    /// Maximum numbers of classes per detection, used in Fast NMS.
    unsigned int m_MaxClassesPerDetection;
    /// Detections per classes, used in Regular NMS.
    unsigned int m_DetectionsPerClass;
    /// NMS score threshold.
    float m_NmsScoreThreshold;
    /// Intersection over union threshold.
    float m_NmsIouThreshold;
    /// Number of classes.
    unsigned int m_NumClasses;
    /// Use Regular NMS.
    bool m_UseRegularNms;
    /// Center size encoding scale x.
    float m_ScaleX;
    /// Center size encoding scale y.
    float m_ScaleY;
    /// Center size encoding scale weight.
    float m_ScaleW;
    /// Center size encoding scale height.
    float m_ScaleH;
};
using NpuDetectionPostProcessFloat32Workload =
    NpuDetectionPostProcessWorkload<armnn::DataType::Float32>;
// using NpuDetectionPostProcessFloat16Workload =
//     NpuDetectionPostProcessWorkload<armnn::DataType::Float16, armnn::DataType::Float32>;
// using NpuDetectionPostProcessUint8Workload =
//     NpuDetectionPostProcessWorkload<armnn::DataType::QAsymmU8, armnn::DataType::Float32>;
}  // namespace armnn
