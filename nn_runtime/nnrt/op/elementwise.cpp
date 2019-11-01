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
#include "op/elementwise.hpp"

#include "model.hpp"

namespace nnrt {
namespace op {

void AddOperation::handleLayoutInferenceOnInputs(
    Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    assert(inputs().size() == 2);
    // Note: the inputs must have the same dimension
    if (!input_permute_cache_.cached_permutes_.empty()) {
        auto iter = input_permute_cache_.cached_permutes_.begin();
        nnrt::layout_inference::IPermuteVectorPtr requiredPermuteVector = iter->second;

        for (decltype(iter) next = iter++; next != input_permute_cache_.cached_permutes_.end();
             ++next) {
            // permute to required permute
            auto finalPermuteVec = next->second->reverse()->add(requiredPermuteVector);
            auto permuteOp = nnrt::op::utils::asOp(finalPermuteVec);
            if (permuteOp) {
                insertPermute(model, permuteOp, finalPermuteVec->asStdVec(), true, next->first);
            }
        }
        next_permute_vectors.insert(std::make_pair(outputs()[0], requiredPermuteVector));
        // handle const operand
        std::vector<uint32_t> constOprIds;
        for (auto in : inputs()) {
            if (model.operand(in)->isConst()) {
                constOprIds.push_back(in);
            }
        }
        if (!constOprIds.empty()) {
            permuteConstOperands(model, constOprIds, requiredPermuteVector);
        }

    } else {
        auto outOperandPtr = model.operand(outputs()[0]);
        next_permute_vectors.insert(
            std::make_pair(outputs()[0], layout_inference::make_shared(outOperandPtr->ndim())));
    }
}

void MulOperation::handleLayoutInferenceOnInputs(
    Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    assert(inputs().size() == 2);
    // Note: the inputs must have the same dimension
    if (!input_permute_cache_.cached_permutes_.empty()) {
        auto iter = input_permute_cache_.cached_permutes_.begin();
        nnrt::layout_inference::IPermuteVectorPtr requiredPermuteVector = iter->second;

        for (decltype(iter) next = iter++; next != input_permute_cache_.cached_permutes_.end();
             ++next) {
            // permute to required permute
            auto finalPermuteVec = next->second->reverse()->add(requiredPermuteVector);
            auto permuteOp = nnrt::op::utils::asOp(finalPermuteVec);
            if (permuteOp) {
                insertPermute(model, permuteOp, finalPermuteVec->asStdVec(), true, next->first);
            }
        }
        next_permute_vectors.insert(std::make_pair(outputs()[0], requiredPermuteVector));
        // handle const operand
        std::vector<uint32_t> constOprIds;
        for (auto in : inputs()) {
            if (model.operand(in)->isConst()) {
                constOprIds.push_back(in);
            }
        }
        if (!constOprIds.empty()) {
            permuteConstOperands(model, constOprIds, requiredPermuteVector);
        }

    } else {
        auto outOperandPtr = model.operand(outputs()[0]);
        next_permute_vectors.insert(
            std::make_pair(outputs()[0], layout_inference::make_shared(outOperandPtr->ndim())));
    }
}

void SubOperation::handleLayoutInferenceOnInputs(
    Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    assert(inputs().size() == 2);
    // Note: the inputs must have the same dimension
    if (!input_permute_cache_.cached_permutes_.empty()) {
        auto iter = input_permute_cache_.cached_permutes_.begin();
        nnrt::layout_inference::IPermuteVectorPtr requiredPermuteVector = iter->second;

        for (decltype(iter) next = iter++; next != input_permute_cache_.cached_permutes_.end();
             ++next) {
            // permute to required permute
            auto finalPermuteVec = next->second->reverse()->add(requiredPermuteVector);
            auto permuteOp = nnrt::op::utils::asOp(finalPermuteVec);
            if (permuteOp) {
                insertPermute(model, permuteOp, finalPermuteVec->asStdVec(), true, next->first);
            }
        }
        next_permute_vectors.insert(std::make_pair(outputs()[0], requiredPermuteVector));
        // handle const operand
        std::vector<uint32_t> constOprIds;
        for (auto in : inputs()) {
            if (model.operand(in)->isConst()) {
                constOprIds.push_back(in);
            }
        }
        if (!constOprIds.empty()) {
            permuteConstOperands(model, constOprIds, requiredPermuteVector);
        }

    } else {
        auto outOperandPtr = model.operand(outputs()[0]);
        next_permute_vectors.insert(
            std::make_pair(outputs()[0], layout_inference::make_shared(outOperandPtr->ndim())));
    }
}

void DivOperation::handleLayoutInferenceOnInputs(
    Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    assert(inputs().size() == 2);
    // Note: the inputs must have the same dimension
    if (!input_permute_cache_.cached_permutes_.empty()) {
        auto iter = input_permute_cache_.cached_permutes_.begin();
        nnrt::layout_inference::IPermuteVectorPtr requiredPermuteVector = iter->second;

        for (decltype(iter) next = iter++; next != input_permute_cache_.cached_permutes_.end();
             ++next) {
            // permute to required permute
            auto finalPermuteVec = next->second->reverse()->add(requiredPermuteVector);
            auto permuteOp = nnrt::op::utils::asOp(finalPermuteVec);
            if (permuteOp) {
                insertPermute(model, permuteOp, finalPermuteVec->asStdVec(), true, next->first);
            }
        }
        next_permute_vectors.insert(std::make_pair(outputs()[0], requiredPermuteVector));
        // handle const operand
        std::vector<uint32_t> constOprIds;
        for (auto in : inputs()) {
            if (model.operand(in)->isConst()) {
                constOprIds.push_back(in);
            }
        }
        if (!constOprIds.empty()) {
            permuteConstOperands(model, constOprIds, requiredPermuteVector);
        }

    } else {
        auto outOperandPtr = model.operand(outputs()[0]);
        next_permute_vectors.insert(
            std::make_pair(outputs()[0], layout_inference::make_shared(outOperandPtr->ndim())));
    }
}

void MinimumOperation::handleLayoutInferenceOnInputs(
    Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    assert(inputs().size() == 2);
    // Note: the inputs must have the same dimension
    if (!input_permute_cache_.cached_permutes_.empty()) {
        auto iter = input_permute_cache_.cached_permutes_.begin();
        nnrt::layout_inference::IPermuteVectorPtr requiredPermuteVector = iter->second;

        for (decltype(iter) next = iter++; next != input_permute_cache_.cached_permutes_.end();
             ++next) {
            // permute to required permute
            auto finalPermuteVec = next->second->reverse()->add(requiredPermuteVector);
            auto permuteOp = nnrt::op::utils::asOp(finalPermuteVec);
            if (permuteOp) {
                insertPermute(model, permuteOp, finalPermuteVec->asStdVec(), true, next->first);
            }
        }
        next_permute_vectors.insert(std::make_pair(outputs()[0], requiredPermuteVector));
        // handle const operand
        std::vector<uint32_t> constOprIds;
        for (auto in : inputs()) {
            if (model.operand(in)->isConst()) {
                constOprIds.push_back(in);
            }
        }
        if (!constOprIds.empty()) {
            permuteConstOperands(model, constOprIds, requiredPermuteVector);
        }

    } else {
        auto outOperandPtr = model.operand(outputs()[0]);
        next_permute_vectors.insert(
            std::make_pair(outputs()[0], layout_inference::make_shared(outOperandPtr->ndim())));
    }
}

void MaximumOperation::handleLayoutInferenceOnInputs(
    Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    assert(inputs().size() == 2);
    // Note: the inputs must have the same dimension
    if (!input_permute_cache_.cached_permutes_.empty()) {
        auto iter = input_permute_cache_.cached_permutes_.begin();
        nnrt::layout_inference::IPermuteVectorPtr requiredPermuteVector = iter->second;

        for (decltype(iter) next = iter++; next != input_permute_cache_.cached_permutes_.end();
             ++next) {
            // permute to required permute
            auto finalPermuteVec = next->second->reverse()->add(requiredPermuteVector);
            auto permuteOp = nnrt::op::utils::asOp(finalPermuteVec);
            if (permuteOp) {
                insertPermute(model, permuteOp, finalPermuteVec->asStdVec(), true, next->first);
            }
        }
        next_permute_vectors.insert(std::make_pair(outputs()[0], requiredPermuteVector));
        // handle const operand
        std::vector<uint32_t> constOprIds;
        for (auto in : inputs()) {
            if (model.operand(in)->isConst()) {
                constOprIds.push_back(in);
            }
        }
        if (!constOprIds.empty()) {
            permuteConstOperands(model, constOprIds, requiredPermuteVector);
        }

    } else {
        auto outOperandPtr = model.operand(outputs()[0]);
        next_permute_vectors.insert(
            std::make_pair(outputs()[0], layout_inference::make_shared(outOperandPtr->ndim())));
    }
}
}
}