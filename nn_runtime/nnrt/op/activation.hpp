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
#ifndef _OP_ACTIVIATION_H_
#define _OP_ACTIVIATION_H_

#include "op/operation.hpp"

namespace nnrt {
namespace op {

struct TanhOperation : Operation {
    TanhOperation() : Operation(OperationType::TANH) {}
    virtual void handleLayoutInferenceOnInputs(
        nnrt::Model& model,
        std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& out_permute_vectors)
        override;
    float scaleA{1.0f};
    float scaleB{1.0f};
};

struct Relu1Operation : Operation {
    Relu1Operation() : Operation(OperationType::RELU1) {}
    virtual void handleLayoutInferenceOnInputs(
        nnrt::Model& model,
        std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& out_permute_vectors)
        override;
};

struct Relu6Operation : Operation {
    Relu6Operation() : Operation(OperationType::RELU6) {}
    virtual void handleLayoutInferenceOnInputs(
        nnrt::Model& model,
        std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& out_permute_vectors)
        override;
};

struct LeakyReluOperation : Operation {
    LeakyReluOperation() : Operation(OperationType::LEAKY_RELU) {}
    virtual void handleLayoutInferenceOnInputs(
        nnrt::Model& model,
        std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& out_permute_vectors)
        override;
    float ratio{0.1f};
};

struct ReluOperation : Operation {
    ReluOperation() : Operation(OperationType::RELU) {}
    virtual void handleLayoutInferenceOnInputs(
        nnrt::Model& model,
        std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& out_permute_vectors)
        override;
};

struct SigmoidOperation : Operation {
    SigmoidOperation() : Operation(OperationType::SIGMOID) {}
    virtual void handleLayoutInferenceOnInputs(
        nnrt::Model& model,
        std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& out_permute_vectors)
        override;
};

struct SoftReluOperation : Operation {
    SoftReluOperation() : Operation(OperationType::SOFT_RELU) {}
    virtual void handleLayoutInferenceOnInputs(
        nnrt::Model& model,
        std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& out_permute_vectors)
        override;
};

struct AbsOperation : Operation {
    AbsOperation() : Operation(OperationType::ABS) {}
    virtual void handleLayoutInferenceOnInputs(
        nnrt::Model& model,
        std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& out_permute_vectors)
        override;
};

struct SqrtOperation : Operation {
    SqrtOperation() : Operation(OperationType::SQRT) {}
    virtual void handleLayoutInferenceOnInputs(
        nnrt::Model& model,
        std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& out_permute_vectors)
        override;
};

struct SquareOperation : Operation {
    SquareOperation() : Operation(OperationType::SQUARE) {}
    virtual void handleLayoutInferenceOnInputs(
        nnrt::Model& model,
        std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& out_permute_vectors)
        override;
};

}
}

#endif
