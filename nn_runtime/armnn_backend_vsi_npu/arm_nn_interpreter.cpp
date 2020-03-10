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
#include <algorithm>

#include "nnrt/error.hpp"
#include "nnrt/model.hpp"
#include "nnrt/types.hpp"
#include "nnrt/logging.hpp"

#include "arm_nn_interpreter.hpp"

using namespace nnrt;
using namespace nnrt::op;

namespace armnn {
#define NNAPI_CHECK_IO_NUM(op, in_num, out_num)                         \
    do {                                                                \
        if ((in_num > 0 && op->inputs().size() != (size_t)in_num) ||    \
            (out_num > 0 && op->outputs().size() != (size_t)out_num)) { \
            NNRT_LOGE_PRINT("Operation IO number mismatch. %d(%d), %d(%d)",     \
                    op->inputs().size(),                                \
                    in_num,                                             \
                    op->outputs().size(),                               \
                    out_num);                                           \
            return nullptr;                                             \
        }                                                               \
    } while (0)

#define NNAPI_CHECK_PTR(pad) \
    do {                     \
        if (!pad) {          \
            return nullptr;  \
        }                    \
    } while (0)

static void convert2DPadding(int32_t* padding, size_t size, int32_t* front, int32_t* back) {
    if (!padding || !front || !back) {
        return;
    }
    for (size_t i = 0; i < size; i += 2) {
        front[i / 2] = padding[i];
        back[i / 2] = padding[i + 1];
    }
}

Armnn_Interpreter::Armnn_Interpreter() {
#define REGISTER_OP(NAME)                                                   \
    do {                                                                    \
        op_container_[OperationType::NAME] = &Armnn_Interpreter::map_##NAME; \
    } while (0)

    REGISTER_OP(ADD);
    REGISTER_OP(CONV_2D);
    REGISTER_OP(DEPTHWISE_CONV_2D);
    REGISTER_OP(RELU);
    REGISTER_OP(RESHAPE);
    REGISTER_OP(FULLY_CONNECTED);
    REGISTER_OP(TRANSPOSE);
    REGISTER_OP(SOFTMAX);
    REGISTER_OP(CONCATENATION);
    REGISTER_OP(AVERAGE_POOL_2D);
    REGISTER_OP(SQUEEZE);
    REGISTER_OP(MAX_POOL_2D);
    REGISTER_OP(PAD);
    REGISTER_OP(MUL);
    REGISTER_OP(MEAN);
    REGISTER_OP(RELU1);
    REGISTER_OP(RELU6);
    REGISTER_OP(ABS);
    REGISTER_OP(SIGMOID);
    REGISTER_OP(TANH);
    REGISTER_OP(LEAKY_RELU);
    REGISTER_OP(SOFT_RELU);
    REGISTER_OP(SQRT);
    REGISTER_OP(SQUARE);
    REGISTER_OP(FLOOR);
    REGISTER_OP(DIV);
    REGISTER_OP(SUB);
    REGISTER_OP(DEQUANTIZE);
    REGISTER_OP(SPACE_TO_DEPTH);
    REGISTER_OP(DEPTH_TO_SPACE);
    REGISTER_OP(SPACE_TO_BATCH_ND);
    REGISTER_OP(BATCH_TO_SPACE_ND);
    REGISTER_OP(L2_NORMALIZATION);
    REGISTER_OP(RESIZE_BILINEAR);
    REGISTER_OP(LOCAL_RESPONSE_NORMALIZATION);
    REGISTER_OP(EMBEDDING_LOOKUP);
    REGISTER_OP(RNN);
    REGISTER_OP(HASHTABLE_LOOKUP);
    REGISTER_OP(LSTM);
    REGISTER_OP(SVDF);
    REGISTER_OP(LSH_PROJECTION);
    REGISTER_OP(L2_POOL_2D);
    REGISTER_OP(STRIDED_SLICE);
    REGISTER_OP(BATCH_NORM);
    REGISTER_OP(MAXIMUM);
    REGISTER_OP(MINIMUM);
    REGISTER_OP(RSQRT);
    REGISTER_OP(PRELU);
    REGISTER_OP(DECONV_2D);
    REGISTER_OP(DATA_CONVERT);
    REGISTER_OP(GREATER);
    REGISTER_OP(EQUAL);
    REGISTER_OP(SPLIT);

/*customer Op*/
// REGISTER_OP(VSI_RESIZE_NEAREST);
#undef REGISTER_OP
}

Armnn_Interpreter::~Armnn_Interpreter() {}

int Armnn_Interpreter::run(Model* model, bool* modified) {
    *modified = false;
    const std::map<uint32_t, OperationPtr>& operations = model->operations();
    for (auto it = operations.begin(); it != operations.end(); ++it) {
        OperationPtr op = it->second;
        if (op_container_.find(op->type()) == op_container_.end()) {
            NNRT_LOGW_PRINT("Not support operation %d", op->type());
            return NNA_ERROR_CODE(BAD_DATA);
        }
    }

    for (auto it = operations.begin(); it != operations.end(); ++it) {
        uint32_t idx = it->first;
        OperationPtr op = it->second;
        NNRT_LOGI_PRINT("Convert node %u(%d)", idx, op->type());
        OperationPtr new_operation = (this->*op_container_[op->type()])(model, op, idx);
        if (!new_operation) {
            NNRT_LOGW_PRINT("Build operation: %d, index: %d fail", op->type(), idx);
            return NNA_ERROR_CODE(OUT_OF_MEMORY);
        }
        replaceOperation(model, idx, new_operation);
    }

    NNRT_LOGD_PRINT("Convert operation completed.");
    // Unique vector
    for (uint32_t index : operands_to_remove_) {
        // NNRT_LOGD_PRINT("Remove %d", index);
        if (model->isInput(index) || model->isOutput(index)) {
            NNRT_LOGW_PRINT(
                "Try remove operand(%u) from model input or output, \
some operations may not support dynamic configure.",
                index);
        } else {
            model->removeOperand(index);
        }
    }

    return NNA_ERROR_CODE(NO_ERROR);
}

void Armnn_Interpreter::replaceOperation(Model* model,
                                        uint32_t op_index,
                                        OperationPtr new_operation) {
    OperationPtr org_operation = model->operation(op_index);
    if (new_operation->inputNum() == 0) {
        new_operation->setInputs(org_operation->inputs());
    }
    if (new_operation->outputNum() == 0) {
        new_operation->setOutputs(org_operation->outputs());
    }
    new_operation->setFusedType(org_operation->fusedType());
    model->operations()[op_index] = new_operation;
}

typedef enum {
   FUSED_NONE = 0,
   FUSED_RELU = 1,
   FUSED_RELU1 = 2,
   FUSED_RELU6 = 3,
} FuseCode;

FusedType Armnn_Interpreter::mapFusedType(int fused_code) {
    FusedType type = FusedType::NONE;
    switch (fused_code) {
        case FUSED_RELU:
            type = FusedType::RELU;
            break;
        case FUSED_RELU1:
            type = FusedType::RELU1;
            break;
        case FUSED_RELU6:
            type = FusedType::RELU6;
            break;
        default:
            break;
    }
    return type;
}

typedef enum {
    PADDING_SAME = 1,
    PADDING_VALID = 2,
} PaddingCode;

PadType Armnn_Interpreter::mapPadType(int code) {
    PadType type = PadType::AUTO;
    switch (code) {
        case PADDING_SAME:
            type = PadType::SAME;
            break;
        case PADDING_VALID:
            type = PadType::VALID;
            break;
        default:
            NNRT_LOGE_PRINT("Invalid padding type(%d)", type);
            assert(false);
            break;
    }
    return type;
}

LshProjectionType Armnn_Interpreter::mapLshProjectionType(int value) {
    LshProjectionType type = LshProjectionType::SPARSE;
    switch (value) {
        case 1:
            type = LshProjectionType::SPARSE;
            break;
        case 2:
            type = LshProjectionType::DENSE;
            break;
        default:
            NNRT_LOGW_PRINT("Unknow lsh projection type: %d", value);
            break;
    }
    return type;
}

FusedType Armnn_Interpreter::mapLstmActivationType(int value) {
    FusedType type = FusedType::NONE;
    switch (value) {
        case 0:
            type = FusedType::NONE;
            break;
        case 1:
            type = FusedType::RELU;
            break;
        case 3:
            type = FusedType::RELU6;
            break;
        case 4:
            type = FusedType::TANH;
            break;
        case 6:
            type = FusedType::SIGMOID;
            break;
        default:
            NNRT_LOGW_PRINT("Unknown lstm activation: %d.", value);
            break;
    }
    return type;
}

std::vector<uint32_t> Armnn_Interpreter::reorderOperands(std::vector<uint32_t>& operands,
                                                        std::vector<int> order) {
    std::vector<uint32_t> new_operands(operands.size());
    new_operands = operands;
    for (uint32_t i = 0; i < order.size(); ++i) {
        if (order[i] >= (int)order.size()) {
            NNRT_LOGW_PRINT("Got incorrect index %d, max size is %lu", order[i], order.size());
            assert(false);
        }
        new_operands[i] = operands[order[i]];
    }
    return new_operands;
}

std::vector<int32_t> Armnn_Interpreter::convertAxes(int32_t* axes_buffer,
                                                   size_t length,
                                                   size_t dim_num) {
    std::vector<int32_t> axes;
    axes.insert(axes.begin(), axes_buffer, axes_buffer + length);
    return convertAxes(axes, dim_num);
}

std::vector<int32_t> Armnn_Interpreter::convertAxes(std::vector<int32_t>& axes, size_t dim_num) {
    std::vector<int32_t> new_axes(axes.size());
    size_t max_size = axes.size() - 1;
    for (size_t i = 0; i < axes.size(); i++) {
        new_axes[i] = convertAxis(axes[max_size - i], dim_num);
    }
    return new_axes;
}

void Armnn_Interpreter::fillIntArray(Model* model,
                                    OperationPtr operation,
                                    std::vector<int32_t>& array,
                                    int32_t op_index,
                                    bool reverse,
                                    bool is_axis) {
    OperandPtr operand = model->operand(operation->input(op_index));
    int32_t* buffer = model->getBuffer<int32_t>(operand->weak_mem_ref.lock());
    size_t length = operand->size();
    array.clear();
    if (!reverse) {
        array.insert(array.begin(), buffer, buffer + length);
    } else if (is_axis) {
        array = convertPermute(buffer, length);
    } else {
        array = reverseArray<int32_t>(buffer, length);
    }
}

int32_t Armnn_Interpreter::reverseMask(int32_t mask, size_t dim_num) {
    auto get_bit_in_mask = [](int mask, int index) -> int { return (((int)0x1) << index) & mask; };
    int32_t new_mask = 0;
    for (int i = (int)dim_num - 1; i >= 0; --i) {
        new_mask |= (get_bit_in_mask(mask, i) >> i) << ((dim_num - 1) - i);
    }
    return new_mask;
}

void Armnn_Interpreter::truncateOperationIOs(Model* model,
                                            OperationPtr operation,
                                            int32_t input_num,
                                            int32_t output_num) {
    // Size - 1 = axis
    input_num = computeAxis(input_num, operation->inputs().size() + 1);
    output_num = computeAxis(output_num, operation->outputs().size() + 1);
    for (int i = input_num; i < (int)operation->inputs().size(); ++i) {
        operands_to_remove_.emplace(operation->input(i));
    }
    for (int i = output_num; i < (int)operation->outputs().size(); ++i) {
        operands_to_remove_.emplace(operation->output(i));
    }
    operation->inputs().resize(input_num);
    operation->outputs().resize(output_num);
}

#define DECLARE_SAMPLE_OP(NAME, INPUT_NUM, OUTPUT_NUM, OPERATION_TYPE)    \
    OperationPtr Armnn_Interpreter::map_##NAME(                            \
        Model* model, OperationPtr operation, uint32_t operation_index) { \
        NNAPI_CHECK_IO_NUM(operation, INPUT_NUM, OUTPUT_NUM);             \
        return std::make_shared<OPERATION_TYPE>();                        \
    }

OperationPtr Armnn_Interpreter::map_ADD(Model* model,
                                       OperationPtr operation,
                                       uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 2, 1);
    truncateOperationIOs(model, operation, 2, 1);
    return std::make_shared<AddOperation>();
}

OperationPtr Armnn_Interpreter::map_SUB(Model* model,
                                       OperationPtr operation,
                                       uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 2, 1);
    truncateOperationIOs(model, operation, 2, 1);
    return std::make_shared<SubOperation>();
}

OperationPtr Armnn_Interpreter::map_DIV(Model* model,
                                       OperationPtr operation,
                                       uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 2, 1);
    std::shared_ptr<DivOperation> div = std::make_shared<DivOperation>();
    div->setVxParam(OverflowPolicy::SATURATE, RoundingPolicy::RTNE, Rounding::RTNE);
    truncateOperationIOs(model, operation, 2, 1);
    return std::dynamic_pointer_cast<Operation>(div);
}

OperationPtr Armnn_Interpreter::map_CONCATENATION(Model* model,
                                                 OperationPtr operation,
                                                 uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, -1, 1);
    std::shared_ptr<ConcatOperation> concat = std::make_shared<ConcatOperation>();
    NNAPI_CHECK_PTR(concat);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    concat->axis = inputs.back()->scalar.int32;
    truncateOperationIOs(model, operation, -2, 1);
    return std::dynamic_pointer_cast<Operation>(concat);
}

OperationPtr Armnn_Interpreter::map_CONV_2D(Model* model,
                                           OperationPtr operation,
                                           uint32_t operation_index) {
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    std::shared_ptr<Conv2DOperation> conv2d = std::make_shared<Conv2DOperation>();
    NNAPI_CHECK_PTR(conv2d);
    if (inputs.size() == 7) {
        conv2d->padType = mapPadType(inputs[3]->scalar.int32);
        conv2d->strides[0] = inputs[4]->scalar.int32;
        conv2d->strides[1] = inputs[5]->scalar.int32;
        resetFusedType(model, operation, 6);
    } else {
        conv2d->pad[0] = inputs[3]->scalar.int32;
        conv2d->pad[1] = inputs[4]->scalar.int32;
        conv2d->pad[2] = inputs[5]->scalar.int32;
        conv2d->pad[3] = inputs[6]->scalar.int32;
        conv2d->strides[0] = inputs[7]->scalar.int32;
        conv2d->strides[1] = inputs[8]->scalar.int32;
        resetFusedType(model, operation, 9);
        // set operation datalayout
        conv2d->setDataLayout(DataLayout(inputs[10]->scalar.int32));
    }
    /* set default dilation value */
    conv2d->dilations[0] = 1;
    conv2d->dilations[1] = 1;
    conv2d->setVxParam(OverflowPolicy::WRAP, RoundingPolicy::TO_ZERO, Rounding::FLOOR);
    truncateOperationIOs(model, operation, 3, 1);
    return std::dynamic_pointer_cast<Operation>(conv2d);
}

OperationPtr Armnn_Interpreter::map_DECONV_2D(Model* model,
                                             OperationPtr operation,
                                             uint32_t operation_index) {
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    std::shared_ptr<Deconv2DOperation> deconv2d = std::make_shared<Deconv2DOperation>();
    NNAPI_CHECK_PTR(deconv2d);
    assert(inputs.size() == 10);
    deconv2d->pad[0] = inputs[3]->scalar.int32;
    deconv2d->pad[1] = inputs[4]->scalar.int32;
    deconv2d->pad[2] = inputs[5]->scalar.int32;
    deconv2d->pad[3] = inputs[6]->scalar.int32;
    deconv2d->strides[0] = inputs[7]->scalar.int32;
    deconv2d->strides[1] = inputs[8]->scalar.int32;
    deconv2d->setDataLayout(DataLayout(inputs[9]->scalar.int32));
    deconv2d->setVxParam(OverflowPolicy::WRAP, RoundingPolicy::TO_ZERO, Rounding::FLOOR);
    truncateOperationIOs(model, operation, 3, 1);
    return std::dynamic_pointer_cast<Operation>(deconv2d);
}

OperationPtr Armnn_Interpreter::map_DEPTHWISE_CONV_2D(Model* model,
                                                     OperationPtr operation,
                                                     uint32_t operation_index) {
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    std::shared_ptr<DepthwiseConv2DOperation> conv2d = std::make_shared<DepthwiseConv2DOperation>();
    NNAPI_CHECK_PTR(conv2d);
    if (inputs.size() == 12) {
        conv2d->pad[0] = inputs[3]->scalar.int32;
        conv2d->pad[1] = inputs[4]->scalar.int32;
        conv2d->pad[2] = inputs[5]->scalar.int32;
        conv2d->pad[3] = inputs[6]->scalar.int32;
        conv2d->strides[0] = inputs[7]->scalar.int32;
        conv2d->strides[1] = inputs[8]->scalar.int32;
        conv2d->multiplier = inputs[9]->scalar.int32;
        resetFusedType(model, operation, 10);
        conv2d->setDataLayout(DataLayout(inputs[11]->scalar.int32));
    } else {
        conv2d->padType = mapPadType(inputs[3]->scalar.int32);
        conv2d->strides[0] = inputs[4]->scalar.int32;
        conv2d->strides[1] = inputs[5]->scalar.int32;
        conv2d->multiplier = inputs[6]->scalar.int32;
        resetFusedType(model, operation, 7);
    }
    conv2d->setVxParam(OverflowPolicy::WRAP, RoundingPolicy::TO_ZERO, Rounding::FLOOR);
    truncateOperationIOs(model, operation, 3, 1);
    return std::dynamic_pointer_cast<Operation>(conv2d);
}

OperationPtr Armnn_Interpreter::map_RELU(Model* model,
                                        OperationPtr operation,
                                        uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 1, 1);
    return std::make_shared<ReluOperation>();
}

OperationPtr Armnn_Interpreter::map_FULLY_CONNECTED(Model* model,
                                                   OperationPtr operation,
                                                   uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 4, 1);
    std::shared_ptr<FullyConnectedOperation> fc = std::make_shared<FullyConnectedOperation>();
    NNAPI_CHECK_PTR(fc);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    uint32_t weights = inputs[1]->dimensions[1];
    uint32_t batch_size = int(inputs[0]->size() / weights);
    uint32_t tmp = int(inputs[0]->dimensions[0] / batch_size);
    inputs[0]->dimensions[0] = batch_size;
    inputs[0]->dimensions[1] *= tmp;
    resetFusedType(model, operation, 3);
    truncateOperationIOs(model, operation, 3, 1);
    fc->setVxParam(OverflowPolicy::WRAP, RoundingPolicy::TO_ZERO, Rounding::FLOOR);
    return std::dynamic_pointer_cast<Operation>(fc);
}

OperationPtr Armnn_Interpreter::map_RESHAPE(Model* model,
                                           OperationPtr operation,
                                           uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 2, 1);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    std::shared_ptr<ReshapeOperation> reshape = std::make_shared<ReshapeOperation>();
    NNAPI_CHECK_PTR(reshape);
    if (!inputs[1]->isConst()) {
        std::vector<OperandPtr> outputs = model->getOperands(operation->outputs());
        assert(outputs[0]->ndim() > 0);
        reshape->shape =
            std::vector<int32_t>(outputs[0]->dimensions.begin(), outputs[0]->dimensions.end());
    } else {
        fillIntArray(model, operation, reshape->shape, 1, false, false);
    }
    truncateOperationIOs(model, operation, 1, 1);
    return std::dynamic_pointer_cast<Operation>(reshape);
}

OperationPtr Armnn_Interpreter::map_SOFTMAX(Model* model,
                                           OperationPtr operation,
                                           uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 3, 1);
    std::shared_ptr<SoftmaxOperation> softmax = std::make_shared<SoftmaxOperation>();
    NNAPI_CHECK_PTR(softmax);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    softmax->beta = inputs[1]->scalar.float32;
    softmax->axis = inputs[2]->scalar.int32;
    truncateOperationIOs(model, operation, 1, 1);
    return std::dynamic_pointer_cast<Operation>(softmax);
}

OperationPtr Armnn_Interpreter::map_TRANSPOSE(Model* model,
                                             OperationPtr operation,
                                             uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 2, 1);
    std::shared_ptr<PermuteOperation> permute = std::make_shared<PermuteOperation>();
    NNAPI_CHECK_PTR(permute);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    fillIntArray(model, operation, permute->perm, 1, false, false);
    truncateOperationIOs(model, operation, 1, 1);
    return std::dynamic_pointer_cast<Operation>(permute);
}

OperationPtr Armnn_Interpreter::map_AVERAGE_POOL_2D(Model* model,
                                                   OperationPtr operation,
                                                   uint32_t operation_index) {
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    std::shared_ptr<AveragePool2DOperation> pool = std::make_shared<AveragePool2DOperation>();
    NNAPI_CHECK_PTR(pool);
    if (inputs.size() == 12) {
        pool->pad[0] = inputs[1]->scalar.int32;
        pool->pad[1] = inputs[2]->scalar.int32;
        pool->pad[2] = inputs[3]->scalar.int32;
        pool->pad[3] = inputs[4]->scalar.int32;
        pool->strides[0] = inputs[5]->scalar.int32;
        pool->strides[1] = inputs[6]->scalar.int32;
        pool->ksize[0] = inputs[7]->scalar.int32;
        pool->ksize[1] = inputs[8]->scalar.int32;
        resetFusedType(model, operation, 9);
        pool->setDataLayout(DataLayout(inputs[10]->scalar.int32));
        pool->roundType = Rounding(inputs[11]->scalar.int32);
    } else {
        assert(false);
    }
    pool->poolMode = PoolMode::VALID;
    pool->setVxParam(OverflowPolicy::WRAP, RoundingPolicy::TO_ZERO, pool->roundType);
    truncateOperationIOs(model, operation, 1, 1);
    return pool;
}

OperationPtr Armnn_Interpreter::map_MAX_POOL_2D(Model* model,
                                               OperationPtr operation,
                                               uint32_t operation_index) {
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    std::shared_ptr<MaxPool2DOperation> pool = std::make_shared<MaxPool2DOperation>();
    NNAPI_CHECK_PTR(pool);
    if (inputs.size() == 12) {
        pool->pad[0] = inputs[1]->scalar.int32;
        pool->pad[1] = inputs[2]->scalar.int32;
        pool->pad[2] = inputs[3]->scalar.int32;
        pool->pad[3] = inputs[4]->scalar.int32;
        pool->strides[0] = inputs[5]->scalar.int32;
        pool->strides[1] = inputs[6]->scalar.int32;
        pool->ksize[0] = inputs[7]->scalar.int32;
        pool->ksize[1] = inputs[8]->scalar.int32;
        resetFusedType(model, operation, 9);
        pool->setDataLayout(DataLayout(inputs[10]->scalar.int32));
        pool->roundType = Rounding(inputs[11]->scalar.int32);
    } else {
        assert(false);
    }
    pool->setVxParam(OverflowPolicy::WRAP, RoundingPolicy::TO_ZERO, pool->roundType);
    truncateOperationIOs(model, operation, 1, 1);
    return pool;
}

OperationPtr Armnn_Interpreter::map_SQUEEZE(Model* model,
                                           OperationPtr operation,
                                           uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 2, 1);
    std::shared_ptr<SqueezeOperation> squeeze = std::make_shared<SqueezeOperation>();
    NNAPI_CHECK_PTR(squeeze);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    if (inputs[1]->isConst()) {
        squeeze->axes.clear();
        int32_t* buffer = model->getBuffer<int32_t>(inputs[1]->weak_mem_ref.lock());
        squeeze->axes = convertAxes(buffer, inputs[1]->size(), inputs[0]->ndim());
        // TODO: remove buffer
    }
    truncateOperationIOs(model, operation, 1, 1);
    return squeeze;
}

OperationPtr Armnn_Interpreter::map_PAD(Model* model,
                                       OperationPtr operation,
                                       uint32_t operation_index) {
    std::shared_ptr<PadOperation> pad = std::make_shared<PadOperation>();
    NNAPI_CHECK_PTR(pad);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    int32_t* padding = model->getBuffer<int32_t>(inputs[1]->weak_mem_ref.lock());
    pad->padFront.resize(inputs[1]->dimensions[0]);
    pad->padBack.resize(inputs[1]->dimensions[0]);
    convert2DPadding(padding, inputs[1]->size(), pad->padFront.data(), pad->padBack.data());
    pad->padFront = pad->padFront;
    pad->padBack = pad->padBack;
    pad->padValue = (inputs.size() == 3 ? inputs[2]->scalar.float32 : 0.0f);
    pad->padMode = PadMode::CONSTANT;
    truncateOperationIOs(model, operation, 1, 1);
    return pad;
}

OperationPtr Armnn_Interpreter::map_MUL(Model* model,
                                       OperationPtr operation,
                                       uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 2, 1);
    std::shared_ptr<MulOperation> mul = std::make_shared<MulOperation>();
    NNAPI_CHECK_PTR(mul);
    mul->setVxParam(OverflowPolicy::SATURATE, RoundingPolicy::RTNE);
    truncateOperationIOs(model, operation, 2, 1);
    return mul;
}

OperationPtr Armnn_Interpreter::map_MEAN(Model* model,
                                        OperationPtr operation,
                                        uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 3, 1);
    std::shared_ptr<ReduceMeanOperation> mean = std::make_shared<ReduceMeanOperation>();
    NNAPI_CHECK_PTR(mean);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    if (inputs[1]->isConst()) {
        mean->axes.clear();
        int32_t* buffer = model->getBuffer<int32_t>(inputs[1]->weak_mem_ref.lock());
        mean->axes.assign(buffer, buffer + inputs[1]->size());
        // TODO: Remove Buffer
    }
    mean->keepDim = static_cast<bool>(inputs[2]->scalar.int32);
    truncateOperationIOs(model, operation, 1, 1);
    return mean;
}

OperationPtr Armnn_Interpreter::map_SPACE_TO_DEPTH(Model* model,
                                                  OperationPtr operation,
                                                  uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 3, 1);
    std::shared_ptr<SpaceToDepthOperation> sp_to_dp = std::make_shared<SpaceToDepthOperation>();
    NNAPI_CHECK_PTR(sp_to_dp);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    sp_to_dp->blockSize[0] = inputs[1]->scalar.int32;
    sp_to_dp->blockSize[1] = inputs[1]->scalar.int32;
    sp_to_dp->setDataLayout(DataLayout(inputs[2]->scalar.int32));
    truncateOperationIOs(model, operation, 1, 1);
    return sp_to_dp;
}

OperationPtr Armnn_Interpreter::map_DEPTH_TO_SPACE(Model* model,
                                                  OperationPtr operation,
                                                  uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 2, 1);
    std::shared_ptr<DepthToSpaceOperation> dp_to_sp = std::make_shared<DepthToSpaceOperation>();
    NNAPI_CHECK_PTR(dp_to_sp);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    dp_to_sp->blockSize[0] = inputs[1]->scalar.int32;
    dp_to_sp->blockSize[1] = inputs[1]->scalar.int32;
    truncateOperationIOs(model, operation, 1, 1);
    return dp_to_sp;
}

OperationPtr Armnn_Interpreter::map_SPACE_TO_BATCH_ND(Model* model,
                                                     OperationPtr operation,
                                                     uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 4, 1);
    std::shared_ptr<SpaceToBatchNDOperation> sp_to_bp = std::make_shared<SpaceToBatchNDOperation>();
    NNAPI_CHECK_PTR(sp_to_bp);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    if (inputs[1]->isConst() && inputs[2]->isConst()) {
        int32_t* buffer = model->getBuffer<int32_t>(inputs[1]->weak_mem_ref.lock());
        sp_to_bp->blockSize.assign(buffer, buffer + inputs[1]->size());
        buffer = model->getBuffer<int32_t>(inputs[2]->weak_mem_ref.lock());
        sp_to_bp->padFront.resize(inputs[0]->ndim() - 2);
        sp_to_bp->padBack.resize(inputs[0]->ndim() - 2);
        convert2DPadding(
            buffer, inputs[2]->size(), sp_to_bp->padFront.data(), sp_to_bp->padBack.data());
        sp_to_bp->setDataLayout(DataLayout(inputs[3]->scalar.int32));
    } else {
        NNRT_LOGW_PRINT("Not support dynamic SPACE_TO_BATCH_ND.");
        assert(false);
    }
    truncateOperationIOs(model, operation, 1, 1);
    return sp_to_bp;
}

OperationPtr Armnn_Interpreter::map_BATCH_TO_SPACE_ND(Model* model,
                                                     OperationPtr operation,
                                                     uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 3, 1);
    std::shared_ptr<BatchToSpaceNDOperation> bp_to_sp = std::make_shared<BatchToSpaceNDOperation>();
    NNAPI_CHECK_PTR(bp_to_sp);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    if (inputs[1]->isConst()) {
        int32_t* buffer = model->getBuffer<int32_t>(inputs[1]->weak_mem_ref.lock());
        bp_to_sp->blockSize.assign(buffer, buffer + inputs[1]->size());
    }
    bp_to_sp->cropStart.resize(inputs[0]->ndim() - 2);
    bp_to_sp->cropEnd.resize(inputs[0]->ndim() - 2);
    bp_to_sp->setDataLayout(DataLayout(inputs[2]->scalar.int32));

    truncateOperationIOs(model, operation, 1, 1);
    return bp_to_sp;
}

OperationPtr Armnn_Interpreter::map_RESIZE_BILINEAR(Model* model,
                                                   OperationPtr operation,
                                                   uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 4, 1);
    std::shared_ptr<ResizeBilinearOperation> resize = std::make_shared<ResizeBilinearOperation>();
    NNAPI_CHECK_PTR(resize);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    resize->outputHeight = inputs[1]->scalar.int32;
    resize->outputWidth = inputs[2]->scalar.int32;
    resize->setDataLayout(DataLayout(inputs[3]->scalar.int32));
    truncateOperationIOs(model, operation, 1, 1);
    return resize;
}

OperationPtr Armnn_Interpreter::map_LOCAL_RESPONSE_NORMALIZATION(Model* model,
                                                                OperationPtr operation,
                                                                uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 8, 1);
    std::shared_ptr<LocalResponseNormOperation> lrn =
        std::make_shared<LocalResponseNormOperation>();
    NNAPI_CHECK_PTR(lrn);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    lrn->radius = inputs[1]->scalar.int32;
    lrn->bias = inputs[2]->scalar.float32;
    lrn->scale = inputs[3]->scalar.float32;
    lrn->exponent = inputs[4]->scalar.float32;
    lrn->channelType = NormalizationAlgorithmChannel(inputs[5]->scalar.uint32);
    lrn->methodType = NormalizationAlgorithmMethod(inputs[6]->scalar.uint32);
    lrn->setDataLayout(DataLayout(inputs[7]->scalar.uint32));
    // Set default axis = channel
    if (DataLayout::NCHW == lrn->getDataLayout()) {
        lrn->axis = 1;
    } else {
        lrn->axis = -1;
    }
    truncateOperationIOs(model, operation, 1, 1);
    return lrn;
}

OperationPtr Armnn_Interpreter::map_RNN(Model* model,
                                       OperationPtr operation,
                                       uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 6, 2);
    std::shared_ptr<RnnOperation> rnn = std::shared_ptr<RnnOperation>();
    auto inputs = model->getOperands(operation->inputs());

    // RNN's activation is NeuralNetwork::FuseType
    rnn->activation = FusedType(inputs[5]->scalar.int32);
    truncateOperationIOs(model, operation, 5, 2);
    return rnn;
}

OperationPtr Armnn_Interpreter::map_LSTM(Model* model,
                                        OperationPtr operation,
                                        uint32_t operation_index) {
    std::shared_ptr<LstmUnitOperation> new_op = std::make_shared<LstmUnitOperation>();
    NNAPI_CHECK_PTR(new_op);

    operation->setInputs(reorderOperands(
        operation->inputs(),
        {0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 2, 20, 21, 22}));
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    auto input_num = inputs.size();

    new_op->activation = mapLstmActivationType(inputs[20]->scalar.int32);
    new_op->cellClip = inputs[21]->scalar.float32;
    new_op->projClip = inputs[22]->scalar.float32;
    input_num -= 3;
    truncateOperationIOs(model, operation, 20, 4);

    while (input_num < LstmUnitOperation::INPUT_COUNT) {
        operation->inputs().emplace_back(-1);
        NNRT_LOGD_PRINT("Append Inputs at [%d]", input_num);
        ++input_num;
    }

    NNAPI_CHECK_IO_NUM(operation, LstmUnitOperation::INPUT_COUNT, LstmUnitOperation::OUTPUT_COUNT);

    return new_op;
}

OperationPtr Armnn_Interpreter::map_SVDF(Model* model,
                                        OperationPtr operation,
                                        uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 7, 2);
    std::shared_ptr<SvdfOperation> new_op = std::make_shared<SvdfOperation>();
    NNAPI_CHECK_PTR(new_op);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    new_op->rank = inputs[5]->scalar.int32;
    resetFusedType(model, operation, 6);
    truncateOperationIOs(model, operation, 5, 2);
    return new_op;
}

OperationPtr Armnn_Interpreter::map_LSH_PROJECTION(Model* model,
                                                  OperationPtr operation,
                                                  uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 4, 1);
    std::shared_ptr<LshProjectionOperation> new_op = std::make_shared<LshProjectionOperation>();
    NNAPI_CHECK_PTR(new_op);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    new_op->type = mapLshProjectionType(inputs[3]->scalar.int32);
    truncateOperationIOs(model, operation, 3, 1);
    return new_op;
}

OperationPtr Armnn_Interpreter::map_L2_POOL_2D(Model* model,
                                              OperationPtr operation,
                                              uint32_t operation_index) {
    std::shared_ptr<L2Pool2DOperation> pool = std::make_shared<L2Pool2DOperation>();
    NNAPI_CHECK_PTR(pool);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    if (inputs.size() == 12) {
        pool->pad[0] = inputs[1]->scalar.int32;
        pool->pad[1] = inputs[2]->scalar.int32;
        pool->pad[2] = inputs[3]->scalar.int32;
        pool->pad[3] = inputs[4]->scalar.int32;
        pool->strides[0] = inputs[5]->scalar.int32;
        pool->strides[1] = inputs[6]->scalar.int32;
        pool->ksize[0] = inputs[7]->scalar.int32;
        pool->ksize[1] = inputs[8]->scalar.int32;
        resetFusedType(model, operation, 9);
        pool->setDataLayout(DataLayout(inputs[10]->scalar.int32));
        pool->roundType = Rounding(inputs[11]->scalar.int32);
    } else {
        NNRT_LOGE_PRINT("Number of input parameter not valid");
        assert(false);
    }
    pool->setVxParam(OverflowPolicy::WRAP, RoundingPolicy::TO_ZERO, pool->roundType);
    truncateOperationIOs(model, operation, 1, 1);
    return pool;
}

OperationPtr Armnn_Interpreter::map_STRIDED_SLICE(Model* model,
                                                 OperationPtr operation,
                                                 uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 8, 1);
    std::shared_ptr<StridedSliceOperation> new_op = std::make_shared<StridedSliceOperation>();
    NNAPI_CHECK_PTR(new_op);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    int32_t* starts = model->getBuffer<int32_t>(inputs[1]->weak_mem_ref.lock());
    int32_t* ends = model->getBuffer<int32_t>(inputs[2]->weak_mem_ref.lock());
    int32_t* strides = model->getBuffer<int32_t>(inputs[3]->weak_mem_ref.lock());
    new_op->starts.assign(starts, starts + inputs[1]->size());
    new_op->ends.assign(ends, ends + inputs[2]->size());
    new_op->strides.assign(strides, strides + inputs[3]->size());
    new_op->beginMask = inputs[4]->scalar.int32;
    new_op->endMask = inputs[5]->scalar.int32;
    new_op->shrinkAxisMask = inputs[6]->scalar.int32;
    new_op->setDataLayout(DataLayout(inputs[7]->scalar.int32));
    truncateOperationIOs(model, operation, 1, 1);
    return new_op;
}

OperationPtr Armnn_Interpreter::map_BATCH_NORM(Model* model,
                                              OperationPtr operation,
                                              uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 7, 1);

    std::shared_ptr<BatchNormalization> new_op = std::make_shared<BatchNormalization>();
    NNAPI_CHECK_PTR(new_op);

    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    new_op->eps = inputs[5]->scalar.float32;
    new_op->setDataLayout(DataLayout(inputs[6]->scalar.int32));
    truncateOperationIOs(model, operation, 6, 1);

    return new_op;
}

OperationPtr Armnn_Interpreter::map_L2_NORMALIZATION(Model* model,
                                                    OperationPtr operation,
                                                    uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 2, 1);

    std::shared_ptr<L2NormOperation> l2_norm = std::make_shared<L2NormOperation>();
    NNAPI_CHECK_PTR(l2_norm);

    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    l2_norm->setDataLayout(DataLayout(inputs[1]->scalar.int32));
    // Set default axis = channel
    if (DataLayout::NCHW == l2_norm->getDataLayout()) {
        l2_norm->axis = 1;
    } else {
        l2_norm->axis = -1;
    }
    truncateOperationIOs(model, operation, 1, 1);

    return l2_norm;
}

OperationPtr Armnn_Interpreter::map_TANH(Model* model,
                                        OperationPtr operation,
                                        uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 3, 1);

    std::shared_ptr<TanhOperation> tanh = std::make_shared<TanhOperation>();
    NNAPI_CHECK_PTR(tanh);

    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    tanh->scaleA = inputs[1]->scalar.float32;
    tanh->scaleB = inputs[2]->scalar.float32;
    truncateOperationIOs(model, operation, 1, 1);

    return tanh;
}

OperationPtr Armnn_Interpreter::map_LEAKY_RELU(Model* model,
                                              OperationPtr operation,
                                              uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 2, 1);

    std::shared_ptr<LeakyReluOperation> leaky_relu = std::make_shared<LeakyReluOperation>();
    NNAPI_CHECK_PTR(leaky_relu);

    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    leaky_relu->ratio = inputs[1]->scalar.float32;
    truncateOperationIOs(model, operation, 1, 1);

    return leaky_relu;
}

OperationPtr Armnn_Interpreter::map_SPLIT(Model* model,
                                          OperationPtr operation,
                                          uint32_t operation_index) {
    std::shared_ptr<SplitOperation> op = std::make_shared<SplitOperation>();
    NNAPI_CHECK_PTR(op);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    op->axis = inputs[1]->scalar.int32;
    op->split_number = inputs[2]->scalar.int32;
    int32_t* slices = model->getBuffer<int32_t>(inputs[3]->weak_mem_ref.lock());
    op->slices.resize(op->split_number);
    memcpy(op->slices.data(), slices, sizeof(int32_t) * op->split_number);
    truncateOperationIOs(model, operation, 1, operation->outputs().size());
    return op;
}

DECLARE_SAMPLE_OP(RELU1, 1, 1, Relu1Operation)
DECLARE_SAMPLE_OP(RELU6, 1, 1, Relu6Operation)
DECLARE_SAMPLE_OP(ABS, 1, 1, AbsOperation)
DECLARE_SAMPLE_OP(SOFT_RELU, 1, 1, SoftReluOperation)
DECLARE_SAMPLE_OP(SQRT, 1, 1, SqrtOperation)
DECLARE_SAMPLE_OP(SQUARE, 1, 1, SquareOperation)
DECLARE_SAMPLE_OP(SIGMOID, 1, 1, SigmoidOperation)
DECLARE_SAMPLE_OP(FLOOR, 1, 1, FloorOperation)
DECLARE_SAMPLE_OP(DEQUANTIZE, 1, 1, DequantizeOperation)
DECLARE_SAMPLE_OP(EMBEDDING_LOOKUP, 2, 1, EmbeddingLookupOperation)
DECLARE_SAMPLE_OP(HASHTABLE_LOOKUP, 3, 2, HashtableLookupOperation)
DECLARE_SAMPLE_OP(MAXIMUM, 2, 1, MaximumOperation)
DECLARE_SAMPLE_OP(MINIMUM, 2, 1, MinimumOperation)
DECLARE_SAMPLE_OP(RSQRT, 1, 1, RSqrtOperation)
DECLARE_SAMPLE_OP(PRELU, 2, 1, PReluOperation)
DECLARE_SAMPLE_OP(DATA_CONVERT, 1, 1, DataConvertOperation)
DECLARE_SAMPLE_OP(GREATER, 2, 1, GreaterOperation)
DECLARE_SAMPLE_OP(EQUAL, 2, 1, EqualOperation)
}
