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
#include <vector>
#include <set>

#include "model.hpp"
#include "types.hpp"
#include "logging.hpp"
#include "error.hpp"

#include "op/operand.hpp"
#include "op/operation.hpp"

namespace nnrt {
namespace op {
inline const char* get_operation_string(OperationType type) {
    switch (type) {
        case OperationType::TRANSPOSE:
            return "permute";
        case OperationType::CONV_2D:
            return "conv2d";
        case OperationType::DEPTHWISE_CONV_2D:
            return "depthwise_conv2d";
        case OperationType::RESHAPE:
            return "reshape";
        case OperationType::MAX_POOL_2D:
            return "maxpool";
        case OperationType::AVERAGE_POOL_2D:
            return "avgpool";
        case OperationType::ADD:
            return "add";
        case OperationType::CONCATENATION:
            return "concat";
        case OperationType::FULLY_CONNECTED:
            return "fullyconnected";
        case OperationType::RELU:
            return "relu";
        case OperationType::SOFTMAX:
            return "softmax";
        case OperationType::SQUEEZE:
            return "squeeze";
        case OperationType::MUL:
            return "mul";
        case OperationType::DIV:
            return "div";
        case OperationType::SUB:
            return "sub";
        case OperationType::SPLIT:
            return "split";
        case OperationType::CONV_1D:
            return "conv1d";
        case OperationType::L2_POOL_2D:
            return "l2pool";
        case OperationType::REDUCE_MEAN:
            return "reduce_mean";
        case OperationType::REDUCE_ALL:
            return "reduce_all";
        case OperationType::REDUCE_ANY:
            return "reduce_any";
        case OperationType::REDUCE_MAX:
            return "reduce_max";
        case OperationType::REDUCE_MIN:
            return "reduce_min";
        case OperationType::REDUCE_PROD:
            return "reduce_prod";
        case OperationType::REDUCE_SUM:
            return "reduce_sum";
        case OperationType::PAD:
            return "pad";
        case OperationType::RELU1:
            return "relu1";
        case OperationType::RELU6:
            return "relu6";
        case OperationType::TANH:
            return "tanh";
        case OperationType::LEAKY_RELU:
            return "leakyrelu";
        case OperationType::PRELU:
            return "prelu";
        case OperationType::SIGMOID:
            return "sigmoid";
        case OperationType::RESIZE_BILINEAR:
            return "resize_bilinear";
        case OperationType::RESIZE_NEAREST:
            return "resize_nearest";
        case OperationType::UNPOOL_2D:
            return "unpool";
        case OperationType::L2_NORM:
            return "l2norm";
        case OperationType::LOCAL_RESPONSE_NORM:
            return "local_response_norm";
        case OperationType::BATCH_NORM:
            return "batch_norm";
        case OperationType::DATA_CONVERT:
            return "data_convert";
        case OperationType::REVERSE:
            return "reverse";
        case OperationType::SPACE_TO_DEPTH:
            return "space_to_depth";
        case OperationType::DEPTH_TO_SPACE:
            return "depth_to_space";
        case OperationType::SPACE_TO_BATCH_ND:
            return "space_to_batch";
        case OperationType::BATCH_TO_SPACE_ND:
            return "batch_to_space";
        case OperationType::FLOOR:
            return "floor";
        case OperationType::RNN:
            return "rnn";
        case OperationType::SVDF:
            return "svdf";
        case OperationType::HASHTABLE_LOOKUP:
            return "hashstable_lookup";
        case OperationType::EMBEDDING_LOOKUP:
            return "embedding_lookup";
        case OperationType::LSTM_UNIT:
            return "lstm_unit";
        case OperationType::LSTM_LAYER:
            return "lstm_layer";
        case OperationType::DEQUANTIZE:
            return "dequantize";
        case OperationType::QUANTIZE:
            return "quantize";
        case OperationType::STRIDED_SLICE:
            return "strided_slice";
        case OperationType::NOOP:
            return "noop";
        case OperationType::SQRT:
            return "sqrt";
        case OperationType::RSQRT:
            return "rsqrt";
        case OperationType::MATRIX_MUL:
            return "matmul";
        case OperationType::ABS:
            return "abs";
        case OperationType::POW:
            return "pow";
        case OperationType::MINIMUM:
            return "minimum";
        case OperationType::MAXIMUM:
            return "maximum";
        case OperationType::IMAGE_PROCESS:
            return "image_process";
        case OperationType::DECONV_2D:
            return "deconv2d";
        case OperationType::SOFT_RELU:
            return "soft_relu";
        default:
            return nullptr;
    }
    return nullptr;
}

Operation::Operation(OperationType type) : input_permute_cache_(*this), type_(type) {}

void Operation::setInputs(const uint32_t* inputs, uint32_t input_size) {
    inputs_.clear();
    if (nullptr == inputs || input_size == 0) {
        return;
    }
    inputs_.insert(inputs_.begin(), inputs, inputs + input_size);
}

void Operation::setOutputs(const uint32_t* outputs, uint32_t output_size) {
    outputs_.clear();
    if (nullptr == outputs || output_size == 0) {
        return;
    }
    outputs_.insert(outputs_.begin(), outputs, outputs + output_size);
}

void Operation::setInputs(const std::vector<uint32_t>& inputs) {
    inputs_ = inputs;
}

void Operation::setOutputs(const std::vector<uint32_t>& outputs) {
    outputs_ = outputs;
}

uint32_t Operation::input(uint32_t index) {
    if (index < inputs_.size()) {
        return inputs_[index];
    }
    return NNRT_INVALID_OPERAND_INDEX;
}

uint32_t Operation::output(uint32_t index) {
    if (index < outputs_.size()) {
        return outputs_[index];
    }
    return NNRT_INVALID_OPERAND_INDEX;
}

bool Operation::replaceOutputs(uint32_t org_index, uint32_t new_index) {
    int pos = find_position(outputs_, org_index);
    if (pos < 0) {
        return false;
    }
    outputs_[pos] = new_index;
    return true;
}

bool Operation::replaceInputs(uint32_t org_index, uint32_t new_index) {
    int pos = find_position(inputs_, org_index);
    if (pos < 0) {
        return false;
    }
    inputs_[pos] = new_index;
    return true;
}

int Operation::find_position(std::vector<uint32_t> operands_indexes, uint32_t index) {
    int pos = 0;
    for (; pos < (int)operands_indexes.size(); ++pos) {
        if (index == operands_indexes[pos]) {
            break;
        }
    }
    if (pos == (int)operands_indexes.size()) {
        pos = -1;
    }
    return pos;
}

void Operation::setVxParam(OverflowPolicy overflow_policy,
                           RoundingPolicy rounding_policy,
                           Rounding down_scale_size_rounding,
                           uint32_t accumulator_bits) {
    vx_param_.overflowPolicy = overflow_policy;
    vx_param_.roundingPolicy = rounding_policy;
    vx_param_.downScaleSizeRounding = down_scale_size_rounding;
    vx_param_.accumulatorBits = accumulator_bits;
}

void Operation::echo(uint32_t index) {
    char buf[256] = {0};
    size_t sz = 0;
    sz += snprintf(&buf[sz], 256 - sz, "%-4u ", index);
    const char* op_str = get_operation_string(type_);
    if (!op_str) {
        sz += snprintf(&buf[sz], 256 - sz, "%-30d", (int32_t)type_);
    } else {
        sz += snprintf(&buf[sz], 256 - sz, "%-30s", op_str);
    }
    char subbuf[128] = {'\0'};
    size_t subsz = 0;
    for (uint32_t i = 0; i < inputs_.size(); ++i) {
        subsz += snprintf(&subbuf[subsz], 128 - subsz, "% d,", inputs_[i]);
    }
    sz += snprintf(&buf[sz], 256 - sz, " i[%-20s]", subbuf);
    subsz = 0;
    for (uint32_t i = 0; i < outputs_.size(); ++i) {
        subsz += snprintf(&subbuf[subsz], 128 - subsz, "% d,", outputs_[i]);
    }
    sz += snprintf(&buf[sz], 256 - sz, " o[%-20s]", subbuf);
    NNRT_LOGD_PRINT("%s", buf);
}

bool Operation::InputTensorPermuteVectorCache::isAllInputTensorSetupWithPermute(Model& model) const {
    bool isReady = true;

    for (auto operand_id : op_.inputs_) {
        auto operandPtr = model.operand(operand_id);
        if (operandPtr && operandPtr->isTensor()
                && !operandPtr->isConst() && !operandPtr->isNull()) {
            isReady &= (cached_permutes_.end() != cached_permutes_.find(operand_id));
        }
    }

    return isReady;
}

std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr> Operation::layoutInference(
    Model& model,
    const std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& operand_permute_vectors) {
    bool isReadyForInference = input_permute_cache_.add(model, operand_permute_vectors);

    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr> next_permute_vectors;
    if (isReadyForInference) {
        handleLayoutInferenceOnInputs(model, next_permute_vectors);
        handleLayoutInferenceOnOutputs(model, next_permute_vectors);
    }

    // next_permute_vectors can be empty at:
    // 1). current operation require multiply input, but not all of input permute information known
    // until now;
    // 2). current operation output is the model's output
    return next_permute_vectors;
}

void Operation::handleLayoutInferenceOnInputs(
    Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    // reverse any applied permute on input data, suppose we can handle this operation by default
    for (auto inId : inputs_) {
        if (input_permute_cache_.cached_permutes_.find(inId) == input_permute_cache_.cached_permutes_.end())
            continue;

        nnrt::layout_inference::IPermuteVectorPtr permuteVector = input_permute_cache_.cached_permutes_[inId];
        CHECK_NULL_PTR(permuteVector);

        auto permuteOp = nnrt::op::utils::asOp(permuteVector->reverse());
        if (permuteOp) {
            insertPermute(model, permuteOp, permuteVector->reverse()->asStdVec(), true, inId);
        }
    }

    // because none-permute vector really applied to input data, we just set default permute vector
    // to output data
    for (auto outId : outputs_) {
        auto outOperandPtr = model.operand(outId);
        nnrt::layout_inference::IPermuteVectorPtr permuteVector =
            nnrt::layout_inference::make_shared(outOperandPtr->ndim());
        CHECK_NULL_PTR(permuteVector);
        next_permute_vectors.insert(std::make_pair(outId, permuteVector));
    }
}

void Operation::handleLayoutInferenceOnOutputs(
    Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    // Find model output operand
    std::vector<uint32_t> modelOutputs;
    for (auto permuteVector : next_permute_vectors) {
        if (model.isOutput(permuteVector.first)) {
            modelOutputs.push_back(permuteVector.first);
        }
    }

    // For any output data, any applied permute vector need reverted
    std::for_each(
        next_permute_vectors.begin(),
        next_permute_vectors.end(),
        [&model, modelOutputs, this](
            const std::pair<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& permuteVec) {
            if (std::find(modelOutputs.begin(), modelOutputs.end(), permuteVec.first) != modelOutputs.end()) {
                auto permuteOp = nnrt::op::utils::asOp(permuteVec.second->reverse());
                if (permuteOp != nullptr) {
                    insertPermute(model, permuteOp, permuteVec.second->asStdVec(), false, permuteVec.first);
                }
            }
        });

    // remove permute vector for output tensor - no need spread them except other operation still consume this output
    for (auto modelOutput : modelOutputs) {
        auto modelPermuteVector = next_permute_vectors.find(modelOutput);
        auto consumers = model.getConsumers(model.operand(modelOutput));
        if (consumers.empty()) {
            next_permute_vectors.erase(modelPermuteVector);
        } else {
            // When the operand is identifyed as the output of model,
            // and it still is the input of next operation, we need to reset the permuteVector
            // to prevent the redundant permute is inserted.
            modelPermuteVector->second->reinitialize();
        }
    }
}

std::vector<uint32_t> Operation::dimensionTrans(std::vector<uint32_t>& orgDims,
                                                const std::vector<uint32_t> perm) {
    std::vector<uint32_t> dstDims;
    for (uint32_t i = 0; i < orgDims.size(); i++) {
        dstDims.push_back(orgDims[perm[i]]);
    }
    return dstDims;
}

void Operation::insertPermute(Model& model,
                              std::shared_ptr<PermuteOperation>& permuteOp,
                              const std::vector<uint32_t>& appliedPermuteVec,
                              bool beforeInOrAfterOut,
                              uint32_t operandId) {
    int newOutOperandId = -1;
    uint32_t permInputs[1];
    uint32_t permOutputs[1];
    auto inputOperandPtr = model.operand(operandId);
    OperandPtr newOutOperandPtr = model.cloneOperand(inputOperandPtr, &newOutOperandId);

    if (beforeInOrAfterOut) {  // Insert Permute before data flow into current op
        newOutOperandPtr->dimensions = dimensionTrans(newOutOperandPtr->dimensions, appliedPermuteVec);
        permInputs[0] = operandId;
        permOutputs[0] = (uint32_t)newOutOperandId;
        replaceInputs(operandId, newOutOperandId);
    } else {  // Insert Permute after data flow out from current op
        newOutOperandPtr->dimensions = dimensionTrans(inputOperandPtr->dimensions, appliedPermuteVec);
        permInputs[0] = {(uint32_t)newOutOperandId};
        permOutputs[0] = {operandId};
        replaceOutputs(operandId, newOutOperandId);
    }
    uint32_t permId = 0;
    permuteOp->setInputs(permInputs, 1);
    permuteOp->setOutputs(permOutputs, 1);
    model.addOperation(permuteOp, &permId);
}

void Operation::permuteConstOperands(Model& model,
                                     std::vector<uint32_t>& constOperandIds,
                                     nnrt::layout_inference::IPermuteVectorPtr permVec) {
    auto constOperandsPtr = model.getOperands(constOperandIds);
    for (auto operandPtr : constOperandsPtr) {
        if (!operandPtr->isTensor()) continue;
        nnrt::layout_inference::IPermuteVectorPtr finalVect = permVec;
        if (operandPtr->getPermVector()) {
            finalVect = operandPtr->getPermVector()->reverse()->add(permVec);
        }
        if (finalVect->isAligned()) {
            operandPtr->setPerm(finalVect->asStdVec());
        } else {
            operandPtr->setPermVector(finalVect);
            auto permVal = finalVect->asStdVec();
            if (operandPtr->ndim() == permVal.size()) {
                operandPtr->setPerm(permVal);
                operandPtr->dimensions = dimensionTrans(operandPtr->dimensions, permVal);
            } else {
                NNRT_LOGE_PRINT("Can not convert const operand, ndim != permVal.size()");
                assert(false);
            }
        }
    }
}

void SoftmaxOperation::handleLayoutInferenceOnInputs(
    Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    (void)model;
    assert(input_permute_cache_.cached_permutes_.size() == 1);
    nnrt::layout_inference::IPermuteVectorPtr permuteVector =
        input_permute_cache_.cached_permutes_[inputs()[0]];

    CHECK_NULL_PTR(permuteVector);

    // convert axis to positive number
    if (axis < 0) {
        axis = permuteVector->rank() + axis;
    }
    // Convert axis to org platform format
    axis = nnrt::op::utils::axisMapTo(permuteVector, axis);
    next_permute_vectors.insert(std::make_pair(outputs()[0], permuteVector));
}

void SplitOperation::handleLayoutInferenceOnInputs(
    Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    (void)model;
    assert(input_permute_cache_.cached_permutes_.size() == 1);
    nnrt::layout_inference::IPermuteVectorPtr permuteVector =
        input_permute_cache_.cached_permutes_[inputs()[0]];
    CHECK_NULL_PTR(permuteVector);
    // convert axis to positive number
    if (axis < 0) {
        axis = permuteVector->rank() + axis;
    }
    // Convert axis to org platform format
    axis = nnrt::op::utils::axisMapTo(permuteVector, axis);

    for (auto output : outputs()) {
        next_permute_vectors.insert(std::make_pair(output, permuteVector));
    }
}


void ArgmaxOperation::handleLayoutInferenceOnInputs(
    Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    (void)model;
    assert(input_permute_cache_.cached_permutes_.size() == 1);
    nnrt::layout_inference::IPermuteVectorPtr permuteVector =
        input_permute_cache_.cached_permutes_[inputs()[0]];
    CHECK_NULL_PTR(permuteVector);
    if (axis < 0) {
        axis = permuteVector->rank() + axis;
    }
    // Convert axis to org platform format
    axis = nnrt::op::utils::axisMapTo(permuteVector, axis);
    next_permute_vectors.insert(std::make_pair(outputs()[0], permuteVector));
}

void ArgminOperation::handleLayoutInferenceOnInputs(
    Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    (void)model;
    assert(input_permute_cache_.cached_permutes_.size() == 1);
    nnrt::layout_inference::IPermuteVectorPtr permuteVector =
        input_permute_cache_.cached_permutes_[inputs()[0]];
    CHECK_NULL_PTR(permuteVector);
    if (axis < 0) {
        axis = permuteVector->rank() + axis;
    }
    // Convert axis to org platform format
    axis = nnrt::op::utils::axisMapTo(permuteVector, axis);
    next_permute_vectors.insert(std::make_pair(outputs()[0], permuteVector));
}

void ChannelShuffleOperation::handleLayoutInferenceOnInputs(
    Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    (void)model;
    assert(input_permute_cache_.cached_permutes_.size() == 1);
    nnrt::layout_inference::IPermuteVectorPtr permuteVector =
        input_permute_cache_.cached_permutes_[inputs()[0]];
    CHECK_NULL_PTR(permuteVector);
    if (axis < 0) {
        axis = permuteVector->rank() + axis;
    }
    // Convert axis to org platform format
    axis = nnrt::op::utils::axisMapTo(permuteVector, axis);
    next_permute_vectors.insert(std::make_pair(outputs()[0], permuteVector));
}

void ConcatOperation::handleLayoutInferenceOnInputs(
    Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    (void)model;

    // Concat input tensor could be _1 or _2 parameter
    if (!input_permute_cache_.cached_permutes_.empty()) {
        auto iter = input_permute_cache_.cached_permutes_.begin();
        auto requiredPermuteVector = iter->second;

        if (axis < 0 && requiredPermuteVector) {
            axis += requiredPermuteVector->rank();
        }
        axis = nnrt::op::utils::axisMapTo(requiredPermuteVector, axis);

        // align another input operand if it's not a constant
        for (auto next = ++iter; next != input_permute_cache_.cached_permutes_.end(); ++next) {
            // permute to required permute
            auto finalPermuteVec = next->second->reverse()->add(requiredPermuteVector);
            auto permuteOp = nnrt::op::utils::asOp(finalPermuteVec);
            if (permuteOp) {
                insertPermute(model, permuteOp, finalPermuteVec->asStdVec(), true, next->first);
            }
        }

        // handle const operand IF ANY
        std::vector<uint32_t> constOprIds;
        for (auto in : inputs()) {
            if (model.operand(in)->isConst()) {
                constOprIds.push_back(in);
            }
        }
        if (!constOprIds.empty()) {
            permuteConstOperands(model, constOprIds, requiredPermuteVector);
        }

        next_permute_vectors.insert(std::make_pair(outputs()[0], requiredPermuteVector));
    }
    else {
        NNRT_LOGE("Concat_LayoutInference") << "fatal error in layoutInference for concat";
    }
}

void SpaceToBatchNDOperation::handleLayoutInferenceOnInputs(
    Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    assert(input_permute_cache_.cached_permutes_.size() == 1);
    OperandPtr inputOperand = model.operand(inputs()[0]);
    OperandPtr outputOperand = model.operand(outputs()[0]);

    nnrt::layout_inference::IPermuteVectorPtr permuteVector =
        input_permute_cache_.cached_permutes_[inputs()[0]];

    CHECK_NULL_PTR(permuteVector);

    if (inputOperand->ndim() != 4) {
        Operation::handleLayoutInferenceOnInputs(model, next_permute_vectors);
        auto reversePermVec = permuteVector->reverse();
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

void BatchToSpaceNDOperation::handleLayoutInferenceOnInputs(
    Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    assert(input_permute_cache_.cached_permutes_.size() == 1);
    OperandPtr inputOperand = model.operand(inputs()[0]);
    OperandPtr outputOperand = model.operand(outputs()[0]);

    nnrt::layout_inference::IPermuteVectorPtr permuteVector =
        input_permute_cache_.cached_permutes_[inputs()[0]];

    CHECK_NULL_PTR(permuteVector);
    if (inputOperand->ndim() != 4) {
        Operation::handleLayoutInferenceOnInputs(model, next_permute_vectors);
        auto reversePermVec = permuteVector->reverse();
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

void SpaceToDepthOperation::handleLayoutInferenceOnInputs(
    Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    assert(input_permute_cache_.cached_permutes_.size() == 1);
    OperandPtr inputOperand = model.operand(inputs()[0]);
    OperandPtr outputOperand = model.operand(outputs()[0]);

    nnrt::layout_inference::IPermuteVectorPtr permuteVector =
        input_permute_cache_.cached_permutes_[inputs()[0]];
    CHECK_NULL_PTR(permuteVector);

    if (inputOperand->ndim() != 4) {
        Operation::handleLayoutInferenceOnInputs(model, next_permute_vectors);
        auto reversePermVec = permuteVector->reverse();
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

void ROIAlignOperation::handleLayoutInferenceOnInputs(
    Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    //assert(input_permute_cache_.cached_permutes_.size() == 1);
    OperandPtr inputOperand = model.operand(inputs()[0]);
    OperandPtr outputOperand = model.operand(outputs()[0]);

    nnrt::layout_inference::IPermuteVectorPtr permuteVector =
        input_permute_cache_.cached_permutes_[inputs()[0]];
    CHECK_NULL_PTR(permuteVector);

    if (inputOperand->ndim() != 4) {
        Operation::handleLayoutInferenceOnInputs(model, next_permute_vectors);
        auto reversePermVec = permuteVector->reverse();
        return;
    }

    // {0, 1, 2, 3}
    auto requiredPermute = nnrt::layout_inference::make_shared(inputOperand->ndim());
    if (DataLayout::NCHW == getDataLayout()) {
        requiredPermute = std::make_shared<nnrt::layout_inference::PermuteVector<4>>(
            std::initializer_list<uint32_t>({0, 2, 3, 1}));
    }

    auto finalPermute = permuteVector->reverse()->add(requiredPermute);
    auto permuteOp = nnrt::op::utils::asOp(finalPermute);

    if (permuteOp) {
        insertPermute(model, permuteOp, finalPermute->asStdVec(), true, inputs()[0]);
    }
    next_permute_vectors.insert(std::make_pair(outputs()[0], requiredPermute));
}

void HeatmapMaxKeypointOperation::handleLayoutInferenceOnInputs(
    Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    //assert(input_permute_cache_.cached_permutes_.size() == 1);
    OperandPtr inputOperand = model.operand(inputs()[0]);
    OperandPtr outputOperand = model.operand(outputs()[0]);

    nnrt::layout_inference::IPermuteVectorPtr permuteVector =
        input_permute_cache_.cached_permutes_[inputs()[0]];
    CHECK_NULL_PTR(permuteVector);

    if (inputOperand->ndim() != 4) {
        Operation::handleLayoutInferenceOnInputs(model, next_permute_vectors);
        auto reversePermVec = permuteVector->reverse();
        return;
    }

    // {0, 1, 2, 3}
    auto requiredPermute = nnrt::layout_inference::make_shared(inputOperand->ndim());
    if (DataLayout::NCHW == getDataLayout()) {
        requiredPermute = std::make_shared<nnrt::layout_inference::PermuteVector<4>>(
            std::initializer_list<uint32_t>({0, 2, 3, 1}));
    }

    auto finalPermute = permuteVector->reverse()->add(requiredPermute);
    auto permuteOp = nnrt::op::utils::asOp(finalPermute);

    if (permuteOp) {
        insertPermute(model, permuteOp, finalPermute->asStdVec(), true, inputs()[0]);
    }
    requiredPermute = std::make_shared<nnrt::layout_inference::PermuteVector<4>>(
        std::initializer_list<uint32_t>({0, 1, 2, 3}));
    next_permute_vectors.insert(std::make_pair(outputs()[0], requiredPermute));
}

void GenerateProposalsOperation::handleLayoutInferenceOnInputs(
    Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    //assert(input_permute_cache_.cached_permutes_.size() == 1);
    OperandPtr inputOperand = model.operand(inputs()[0]);
    OperandPtr outputOperand = model.operand(outputs()[0]);

    nnrt::layout_inference::IPermuteVectorPtr permuteVector =
        input_permute_cache_.cached_permutes_[inputs()[0]];
    CHECK_NULL_PTR(permuteVector);

    if (inputOperand->ndim() != 4) {
        Operation::handleLayoutInferenceOnInputs(model, next_permute_vectors);
        auto reversePermVec = permuteVector->reverse();
        return;
    }

    // {0, 1, 2, 3}
    auto requiredPermute = nnrt::layout_inference::make_shared(inputOperand->ndim());
    if (DataLayout::NCHW == getDataLayout()) {
        requiredPermute = std::make_shared<nnrt::layout_inference::PermuteVector<4>>(
            std::initializer_list<uint32_t>({0, 2, 3, 1}));
    }

    auto finalPermute = permuteVector->reverse()->add(requiredPermute);
    auto permuteOp = nnrt::op::utils::asOp(finalPermute);

    if (permuteOp) {
        insertPermute(model, permuteOp, finalPermute->asStdVec(), true, inputs()[0]);
        {
            OperandPtr inputOperand = model.operand(inputs()[1]);
            nnrt::layout_inference::IPermuteVectorPtr permuteVector =
                input_permute_cache_.cached_permutes_[inputs()[1]];
            CHECK_NULL_PTR(permuteVector);
            auto finalPermute = permuteVector->reverse()->add(requiredPermute);
            auto permuteOp = nnrt::op::utils::asOp(finalPermute);
            insertPermute(model, permuteOp, finalPermute->asStdVec(), true, inputs()[1]);
        }
    }

    requiredPermute = std::make_shared<nnrt::layout_inference::PermuteVector<4>>(
        std::initializer_list<uint32_t>({0, 1, 2, 3}));
    next_permute_vectors.insert(std::make_pair(outputs()[0], requiredPermute));
}

void DepthToSpaceOperation::handleLayoutInferenceOnInputs(
    Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    assert(input_permute_cache_.cached_permutes_.size() == 1);
    OperandPtr inputOperand = model.operand(inputs()[0]);
    OperandPtr outputOperand = model.operand(outputs()[0]);

    nnrt::layout_inference::IPermuteVectorPtr permuteVector =
        input_permute_cache_.cached_permutes_[inputs()[0]];
    CHECK_NULL_PTR(permuteVector);

    if (inputOperand->ndim() != 4) {
        Operation::handleLayoutInferenceOnInputs(model, next_permute_vectors);
        auto reversePermVec = permuteVector->reverse();
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

void StridedSliceOperation::handleLayoutInferenceOnInputs(
    Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    assert(input_permute_cache_.cached_permutes_.size() == 1);
    OperandPtr inputOperand = model.operand(inputs()[0]);
    OperandPtr outputOperand = model.operand(outputs()[0]);

    nnrt::layout_inference::IPermuteVectorPtr permuteVector =
        input_permute_cache_.cached_permutes_[inputs()[0]];
    CHECK_NULL_PTR(permuteVector);

    if (inputOperand->ndim() != 4) {
        Operation::handleLayoutInferenceOnInputs(model, next_permute_vectors);
        auto reversePermVec = permuteVector->reverse();
        return;
    }

    // {0, 1, 2, 3}
    auto requiredPermute = nnrt::layout_inference::make_shared(inputOperand->ndim());
    if (DataLayout::NHWC == getDataLayout()) {
        requiredPermute = std::make_shared<nnrt::layout_inference::PermuteVector<4>>(
            std::initializer_list<uint32_t>({0, 3, 1, 2}));
        starts = nnrt::op::utils::permuteArray(starts, requiredPermute);
        ends = nnrt::op::utils::permuteArray(ends, requiredPermute);
        strides = nnrt::op::utils::permuteArray(strides, requiredPermute);
        // TODO: Convert mask
    }

    auto finalPermute = permuteVector->reverse()->add(requiredPermute);
    auto permuteOp = nnrt::op::utils::asOp(finalPermute);

    if (permuteOp) {
        insertPermute(model, permuteOp, finalPermute->asStdVec(), true, inputs()[0]);
    }

    next_permute_vectors.insert(std::make_pair(outputs()[0], requiredPermute));
}

void ResizeBilinearOperation::handleLayoutInferenceOnInputs(
    Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    assert(input_permute_cache_.cached_permutes_.size() == 1);
    OperandPtr inputOperand = model.operand(inputs()[0]);
    OperandPtr outputOperand = model.operand(outputs()[0]);

    nnrt::layout_inference::IPermuteVectorPtr permuteVector =
        input_permute_cache_.cached_permutes_[inputs()[0]];
    CHECK_NULL_PTR(permuteVector);

    if (inputOperand->ndim() != 4) {
        Operation::handleLayoutInferenceOnInputs(model, next_permute_vectors);
        auto reversePermVec = permuteVector->reverse();
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

void ResizeNearestNeighborOperation::handleLayoutInferenceOnInputs(
    Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    assert(input_permute_cache_.cached_permutes_.size() == 1);
    OperandPtr inputOperand = model.operand(inputs()[0]);
    OperandPtr outputOperand = model.operand(outputs()[0]);

    nnrt::layout_inference::IPermuteVectorPtr permuteVector =
        input_permute_cache_.cached_permutes_[inputs()[0]];
    CHECK_NULL_PTR(permuteVector);

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

void PadOperation::handleLayoutInferenceOnInputs(
    Model& model,
    std::unordered_map<uint32_t, nnrt::layout_inference::IPermuteVectorPtr>& next_permute_vectors) {
    assert(input_permute_cache_.cached_permutes_.size() == 1);
    OperandPtr inputOperand = model.operand(inputs()[0]);
    OperandPtr outputOperand = model.operand(outputs()[0]);

    nnrt::layout_inference::IPermuteVectorPtr permuteVector =
        input_permute_cache_.cached_permutes_[inputs()[0]];
    CHECK_NULL_PTR(permuteVector);

    if (inputOperand->ndim() != 4) {
        Operation::handleLayoutInferenceOnInputs(model, next_permute_vectors);
        auto reversePermVec = permuteVector->reverse();
        return;
    }

    // {0, 1, 2, 3}
    auto requiredPermute = nnrt::layout_inference::make_shared(inputOperand->ndim());
    if (DataLayout::NHWC == getDataLayout()) {
        requiredPermute = std::make_shared<nnrt::layout_inference::PermuteVector<4>>(
            std::initializer_list<uint32_t>({0, 3, 1, 2}));
        padFront = nnrt::op::utils::permuteArray(padFront, requiredPermute);
        padBack = nnrt::op::utils::permuteArray(padBack, requiredPermute);
    }

    auto finalPermute = permuteVector->reverse()->add(requiredPermute);
    auto permuteOp = nnrt::op::utils::asOp(finalPermute);

    if (permuteOp) {
        insertPermute(model, permuteOp, finalPermute->asStdVec(), true, inputs()[0]);
    }

    next_permute_vectors.insert(std::make_pair(outputs()[0], requiredPermute));
}

} // End of op namespace

}  // End of nnrt namespace
