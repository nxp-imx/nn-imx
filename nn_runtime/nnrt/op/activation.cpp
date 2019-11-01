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
#include "op/activation.hpp"

#include "model.hpp"

namespace nnrt {
namespace op {
void TanhOperation::handleLayoutInferenceOnInputs(
    nnrt::Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    (void)model; // unused variable
    assert(input_permute_cache_.cached_permutes_.size() == 1);
    nnrt::layout_inference::IPermuteVectorPtr permuteVector =
        input_permute_cache_.cached_permutes_[inputs()[0]];
    next_permute_vectors.insert(std::make_pair(outputs()[0], permuteVector));
}

void Relu1Operation::handleLayoutInferenceOnInputs(
    nnrt::Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    (void)model; // unused variable
    assert(input_permute_cache_.cached_permutes_.size() == 1);
    nnrt::layout_inference::IPermuteVectorPtr permuteVector =
        input_permute_cache_.cached_permutes_[inputs()[0]];
    next_permute_vectors.insert(std::make_pair(outputs()[0], permuteVector));
}

void Relu6Operation::handleLayoutInferenceOnInputs(
    nnrt::Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    (void)model; // unused variable
    assert(input_permute_cache_.cached_permutes_.size() == 1);
    nnrt::layout_inference::IPermuteVectorPtr permuteVector =
        input_permute_cache_.cached_permutes_[inputs()[0]];
    next_permute_vectors.insert(std::make_pair(outputs()[0], permuteVector));
}

void LeakyReluOperation::handleLayoutInferenceOnInputs(
    nnrt::Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    (void)model; // unused variable
    assert(input_permute_cache_.cached_permutes_.size() == 1);
    nnrt::layout_inference::IPermuteVectorPtr permuteVector =
        input_permute_cache_.cached_permutes_[inputs()[0]];
    next_permute_vectors.insert(std::make_pair(outputs()[0], permuteVector));
}

void ReluOperation::handleLayoutInferenceOnInputs(
    nnrt::Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    (void)model; // unused variable
    assert(input_permute_cache_.cached_permutes_.size() == 1);
    nnrt::layout_inference::IPermuteVectorPtr permuteVector =
        input_permute_cache_.cached_permutes_[inputs()[0]];
    next_permute_vectors.insert(std::make_pair(outputs()[0], permuteVector));
}

void SigmoidOperation::handleLayoutInferenceOnInputs(
    nnrt::Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    (void)model; // unused variable
    assert(input_permute_cache_.cached_permutes_.size() == 1);
    nnrt::layout_inference::IPermuteVectorPtr permuteVector =
        input_permute_cache_.cached_permutes_[inputs()[0]];
    next_permute_vectors.insert(std::make_pair(outputs()[0], permuteVector));
}

void SoftReluOperation::handleLayoutInferenceOnInputs(
    nnrt::Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    (void)model; // unused variable
    assert(input_permute_cache_.cached_permutes_.size() == 1);
    nnrt::layout_inference::IPermuteVectorPtr permuteVector =
        input_permute_cache_.cached_permutes_[inputs()[0]];
    next_permute_vectors.insert(std::make_pair(outputs()[0], permuteVector));
}

void AbsOperation::handleLayoutInferenceOnInputs(
    nnrt::Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    (void)model; // unused variable
    assert(input_permute_cache_.cached_permutes_.size() == 1);
    nnrt::layout_inference::IPermuteVectorPtr permuteVector =
        input_permute_cache_.cached_permutes_[inputs()[0]];
    next_permute_vectors.insert(std::make_pair(outputs()[0], permuteVector));
}

void SqrtOperation::handleLayoutInferenceOnInputs(
    nnrt::Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    (void)model; // unused variable
    assert(input_permute_cache_.cached_permutes_.size() == 1);
    nnrt::layout_inference::IPermuteVectorPtr permuteVector =
        input_permute_cache_.cached_permutes_[inputs()[0]];
    next_permute_vectors.insert(std::make_pair(outputs()[0], permuteVector));
}

void SquareOperation::handleLayoutInferenceOnInputs(
    nnrt::Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    (void)model; // unused variable
    assert(input_permute_cache_.cached_permutes_.size() == 1);
    nnrt::layout_inference::IPermuteVectorPtr permuteVector =
        input_permute_cache_.cached_permutes_[inputs()[0]];
    next_permute_vectors.insert(std::make_pair(outputs()[0], permuteVector));
}
}
}
