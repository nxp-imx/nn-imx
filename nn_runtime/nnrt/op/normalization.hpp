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
#ifndef _OP_NORMALIZATION_H_
#define _OP_NORMALIZATION_H_

#include "op/operation.hpp"

namespace nnrt {
namespace op {
struct BatchNormalization : Operation {
    BatchNormalization() : Operation(OperationType::BATCH_NORM) {}
    virtual void handleLayoutInferenceOnInputs(
        nnrt::Model& model,
        std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& out_permute_vectors)
        override;
    float eps;
};

struct L2NormOperation : Operation {
    L2NormOperation() : Operation(OperationType::L2_NORM) {}
    virtual void handleLayoutInferenceOnInputs(
        nnrt::Model& model,
        std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& out_permute_vectors)
        override;
    int32_t axis;
};

struct LocalResponseNormOperation : Operation {
    LocalResponseNormOperation() : Operation(OperationType::LOCAL_RESPONSE_NORM) {}
    virtual void handleLayoutInferenceOnInputs(
        Model& model,
        std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>&
            out_permute_vectors) override;
    int32_t radius;
    float bias;
    float scale;     // alpha
    float exponent;  // beta
    int32_t axis;
    /// Normalization channel algorithm to use (Across, Within).
    NormalizationAlgorithmChannel channelType{NormalizationAlgorithmChannel::Across};
    /// Normalization method algorithm to use (LocalBrightness, LocalContrast).
    NormalizationAlgorithmMethod methodType{NormalizationAlgorithmMethod::LocalBrightness};
};

struct InstanceNormOperation : Operation {
    InstanceNormOperation() : Operation(OperationType::INSTANCE_NORM) {}
    virtual void handleLayoutInferenceOnInputs(
        Model& model,
        std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>&
            out_permute_vectors) override;
    float gamma;
    float beta;
    float eps;
    std::vector<int32_t> axes;
};
}
}

#endif
