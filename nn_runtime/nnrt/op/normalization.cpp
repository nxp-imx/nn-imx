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
#include "op/normalization.hpp"

#include "model.hpp"

namespace nnrt {
namespace op {

void BatchNormalization::handleLayoutInferenceOnInputs(
    nnrt::Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    assert(input_permute_cache_.cached_permutes_.size() == 1);
    OperandPtr inputOperand = model.operand(inputs()[0]);
    OperandPtr outputOperand = model.operand(outputs()[0]);

    nnrt::layout_inference::IPermuteVectorPtr permuteVector =
        input_permute_cache_.cached_permutes_[inputs()[0]];

    if (inputOperand->ndim() != 4) {
        auto reversePermVec = permuteVector->reverse();

        auto permuteOp = nnrt::op::utils::asOp(reversePermVec);
        if (permuteOp) {
            insertPermute(model, permuteOp, reversePermVec->asStdVec(), true, inputs()[0]);
        }
        next_permute_vectors.insert(
            std::make_pair(outputs()[0], nnrt::layout_inference::make_shared(outputOperand->ndim())));
        return;
    }

    // {0, 1, 2, 3}
    auto requiredPermute = nnrt::layout_inference::make_shared(inputOperand->ndim());
    if (DataLayout::NHWC == getDataLayout()) {
        requiredPermute = std::make_shared<nnrt::layout_inference::PermuteVector<4>>(
            std::initializer_list<uint32_t>({0, 3, 1, 2}));
    }

    auto finalPermute = permuteVector->reverse()->add(requiredPermute);
    auto permuteOp = nnrt::op::utils::asOp(finalPermute);

    if (permuteOp) {
        insertPermute(model, permuteOp, finalPermute->asStdVec(), true, inputs()[0]);
    }

    next_permute_vectors.insert(std::make_pair(outputs()[0], requiredPermute));
}

void L2NormOperation::handleLayoutInferenceOnInputs(
    nnrt::Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    assert(input_permute_cache_.cached_permutes_.size() == 1);
    OperandPtr inputOperand = model.operand(inputs()[0]);
    OperandPtr outputOperand = model.operand(outputs()[0]);

    nnrt::layout_inference::IPermuteVectorPtr permuteVector =
        input_permute_cache_.cached_permutes_[inputs()[0]];

    if (inputOperand->ndim() != 4) {
        auto reversePermVec = permuteVector->reverse();

        auto permuteOp = nnrt::op::utils::asOp(reversePermVec);
        if (permuteOp) {
            insertPermute(model, permuteOp, reversePermVec->asStdVec(), true, inputs()[0]);
        }
        next_permute_vectors.insert(
            std::make_pair(outputs()[0], nnrt::layout_inference::make_shared(outputOperand->ndim())));
        // convert axis to positive number
        if (axis < 0) {
            axis = permuteVector->rank() + axis;
        }
        // Convert axis to org platform format
        axis = nnrt::op::utils::axisMapTo(permuteVector, axis);
        return;
    }

    // {0, 1, 2, 3}
    auto requiredPermute = layout_inference::make_shared(inputOperand->ndim());
    if (DataLayout::NHWC == getDataLayout()) {
        requiredPermute = std::make_shared<layout_inference::PermuteVector<4>>(
            std::initializer_list<uint32_t>({0, 3, 1, 2}));
    }

    auto finalPermute = permuteVector->reverse()->add(requiredPermute);
    auto permuteOp = nnrt::op::utils::asOp(finalPermute);

    if (permuteOp) {
        insertPermute(model, permuteOp, finalPermute->asStdVec(), true, inputs()[0]);
    }

    // convert axis to positive number
    if (axis < 0) {
        axis = permuteVector->rank() + axis;
    }
    // Convert axis to org platform format
    axis = nnrt::op::utils::axisMapTo(finalPermute, axis);

    next_permute_vectors.insert(std::make_pair(outputs()[0], requiredPermute));
}

void LocalResponseNormOperation::handleLayoutInferenceOnInputs(
    Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    assert(input_permute_cache_.cached_permutes_.size() == 1);
    OperandPtr inputOperand = model.operand(inputs()[0]);
    OperandPtr outputOperand = model.operand(outputs()[0]);

    nnrt::layout_inference::IPermuteVectorPtr permuteVector =
        input_permute_cache_.cached_permutes_[inputs()[0]];

    if (inputOperand->ndim() != 4) {
        Operation::handleLayoutInferenceOnInputs(model, next_permute_vectors);
        auto reversePermVec = permuteVector->reverse();
        return;
    }

    // {0, 1, 2, 3}
    auto requiredPermute = layout_inference::make_shared(inputOperand->ndim());
    if (DataLayout::NHWC == getDataLayout()) {
        requiredPermute = std::make_shared<layout_inference::PermuteVector<4>>(
            std::initializer_list<uint32_t>({0, 3, 1, 2}));
    }

    auto finalPermute = permuteVector->reverse()->add(requiredPermute);
    auto permuteOp = nnrt::op::utils::asOp(finalPermute);

    if (permuteOp) {
        insertPermute(model, permuteOp, finalPermute->asStdVec(), true, inputs()[0]);
    }

    // convert axis to positive number
    if (axis < 0) {
        axis = permuteVector->rank() + axis;
    }
    // Convert axis to org platform format
    axis = nnrt::op::utils::axisMapTo(finalPermute, axis);

    next_permute_vectors.insert(std::make_pair(outputs()[0], requiredPermute));
}

/** Copy from LRN
 */
void InstanceNormOperation::handleLayoutInferenceOnInputs(
    Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    assert(input_permute_cache_.cached_permutes_.size() == 1);
    OperandPtr inputOperand = model.operand(inputs()[0]);
    OperandPtr outputOperand = model.operand(outputs()[0]);

    nnrt::layout_inference::IPermuteVectorPtr permuteVector =
        input_permute_cache_.cached_permutes_[inputs()[0]];

    if (inputOperand->ndim() != 4) {
        Operation::handleLayoutInferenceOnInputs(model, next_permute_vectors);
        auto reversePermVec = permuteVector->reverse();
        return;
    }

    // {0, 1, 2, 3}
    auto requiredPermute = layout_inference::make_shared(inputOperand->ndim());
    if (DataLayout::NHWC == getDataLayout()) {
        requiredPermute = std::make_shared<layout_inference::PermuteVector<4>>(
            std::initializer_list<uint32_t>({0, 3, 1, 2}));
    }

    auto finalPermute = permuteVector->reverse()->add(requiredPermute);
    auto permuteOp = nnrt::op::utils::asOp(finalPermute);

    if (permuteOp) {
        insertPermute(model, permuteOp, finalPermute->asStdVec(), true, inputs()[0]);
    }

    next_permute_vectors.insert(std::make_pair(outputs()[0], requiredPermute));
}

}  // namespace op
}  // namespace nnrt
