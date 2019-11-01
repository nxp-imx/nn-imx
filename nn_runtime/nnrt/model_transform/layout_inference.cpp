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
#include <cassert>
#include <deque>
#include <vector>

#include "error.hpp"
#include "model.hpp"
#include "model_transform/transformations.hpp"
#include "op/public.hpp"

namespace nnrt {
namespace {
using nnrt::layout_inference::IPermuteVectorPtr;
using nnrt::op::OperationPtr;

static void layoutInference(Model* model) {
    std::deque<std::pair<uint32_t, IPermuteVectorPtr>> operandPermuteQueue;

    for (auto inputOperandId : model->inputIndexes()) {
        auto OperandPtr = model->operand(inputOperandId);
        operandPermuteQueue.push_back(
            std::make_pair(inputOperandId, layout_inference::make_shared(OperandPtr->ndim())));
    }

    while (!operandPermuteQueue.empty()) {
        auto operandPermute = operandPermuteQueue.front();
        operandPermuteQueue.pop_front();
        auto consumerOpIds = model->getConsumers(model->operand(operandPermute.first));

        for (auto consumerOpId : consumerOpIds) {
            OperationPtr consumerOpPtr = model->operation(consumerOpId);
            // only when current operand is the last input operand of consumer_op,
            // the consumerOp can do layout inference
            std::unordered_map<uint32_t, IPermuteVectorPtr> currentPermuteVectors;
            currentPermuteVectors.insert(operandPermute);
            auto nextPermute = consumerOpPtr->layoutInference(*model, currentPermuteVectors);

            std::for_each(
                nextPermute.begin(),
                nextPermute.end(),
                [&operandPermuteQueue](const std::pair<uint32_t, IPermuteVectorPtr>& operandPermute) {
                    operandPermuteQueue.push_back(operandPermute);
                });
        }
    }
}
}

int LayoutInference::run(Model* model, bool* modified) {
    if (nullptr == model) {
        return NNA_ERROR_CODE(NO_ERROR);
    }

    *modified = true;

    layoutInference(model);

    return NNA_ERROR_CODE(NO_ERROR);
}
}