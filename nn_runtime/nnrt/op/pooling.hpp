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
#ifndef _OP_POOLING_H_
#define _OP_POOLING_H_

#include "operation.hpp"

namespace nnrt {
namespace op {
struct AveragePool2DOperation : Operation {
    AveragePool2DOperation() : Operation(OperationType::AVERAGE_POOL_2D) {
        strides.resize(2);
        ksize.resize(2);
        pad.resize(4);
    }
    virtual void handleLayoutInferenceOnInputs(
        nnrt::Model& model,
        std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& out_permute_vectors)
        override;
    std::vector<int32_t> strides;
    std::vector<int32_t> ksize;
    std::vector<int32_t> pad;
    PadType padType{PadType::AUTO};
    Rounding roundType{Rounding::FLOOR};
    PoolMode poolMode{PoolMode::VALID};
};

struct MaxPool2DOperation : Operation {
    MaxPool2DOperation() : Operation(OperationType::MAX_POOL_2D) {
        strides.resize(2);
        ksize.resize(2);
        pad.resize(4);
    }
    virtual void handleLayoutInferenceOnInputs(
        nnrt::Model& model,
        std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& out_permute_vectors)
        override;
    std::vector<int32_t> strides;
    std::vector<int32_t> ksize;
    std::vector<int32_t> pad;
    PadType padType{PadType::AUTO};
    Rounding roundType{Rounding::FLOOR};
};

struct L2Pool2DOperation : Operation {
    L2Pool2DOperation() : Operation(OperationType::L2_POOL_2D) {
        strides.resize(2);
        ksize.resize(2);
        pad.resize(4);
    }
    virtual void handleLayoutInferenceOnInputs(
        nnrt::Model& model,
        std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& out_permute_vectors)
        override;

    std::vector<int32_t> strides;
    std::vector<int32_t> ksize;
    std::vector<int32_t> pad;
    PadType padType{PadType::AUTO};
    Rounding roundType{Rounding::FLOOR};
};

struct Unpool2DOperation : Operation {
    Unpool2DOperation() : Operation(OperationType::UNPOOL_2D) {}
    int output_height;
    int output_width;
};
}
}

#endif
