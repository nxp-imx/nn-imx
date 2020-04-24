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
#include <algorithm>
#include <cassert>

#include "nnrt/error.hpp"
#include "nnrt/logging.hpp"
#include "nnrt/model.hpp"
#include "nnrt/types.hpp"
#include "nnrt/utils.hpp"

#include "nnrt/api_requirement/nnapi_requirement.hpp"
#include "nnrt/model_transform/nnapi_interpreter.hpp"

namespace nnrt {

#define NNAPI_CHECK_IO_NUM(op, in_num, out_num)                             \
    do {                                                                    \
        if ((in_num > 0 && op->inputs().size() != (size_t)in_num) ||        \
            (out_num > 0 && op->outputs().size() != (size_t)out_num)) {     \
            NNRT_LOGW_PRINT("Operation IO number mismatch. %d(%d), %d(%d)", \
                            op->inputs().size(),                            \
                            in_num,                                         \
                            op->outputs().size(),                           \
                            out_num);                                       \
            return nullptr;                                                 \
        }                                                                   \
    } while (0)

#define NNAPI_CHECK_PTR(ptr) \
    do {                     \
        if (!ptr) {          \
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

NnApiInterpreter::NnApiInterpreter() {
#define REGISTER_OP(NAME)                                                   \
    do {                                                                    \
        op_container_[OperationType::NAME] = &NnApiInterpreter::map_##NAME; \
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
    REGISTER_OP(SIGMOID);
    REGISTER_OP(TANH);
    REGISTER_OP(FLOOR);
    REGISTER_OP(DIV);
    REGISTER_OP(SUB);
    REGISTER_OP(QUANTIZE);
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
    REGISTER_OP(UNIDIRECTIONAL_SEQUENCE_RNN);
    REGISTER_OP(BIDIRECTIONAL_SEQUENCE_RNN);
    REGISTER_OP(UNIDIRECTIONAL_SEQUENCE_LSTM);
    REGISTER_OP(BIDIRECTIONAL_SEQUENCE_LSTM);
    REGISTER_OP(HASHTABLE_LOOKUP);
    REGISTER_OP(LSTM);
    REGISTER_OP(SVDF);
    REGISTER_OP(LSH_PROJECTION);
    REGISTER_OP(L2_POOL_2D);
    REGISTER_OP(STRIDED_SLICE);
    REGISTER_OP(RESIZE_NEAREST);
    REGISTER_OP(ABS);
    REGISTER_OP(ARGMAX);
    REGISTER_OP(ARGMIN);
    REGISTER_OP(EQUAL);
    REGISTER_OP(EXP);
    REGISTER_OP(EXPAND_DIMS);
    REGISTER_OP(GATHER);
    REGISTER_OP(CHANNEL_SHUFFLE);
    REGISTER_OP(GREATER);
    REGISTER_OP(GREATER_EQUAL);
    REGISTER_OP(GROUPED_CONV_2D);
    REGISTER_OP(INSTANCE_NORMALIZATION);
    REGISTER_OP(LESS);
    REGISTER_OP(LESS_EQUAL);
    REGISTER_OP(LOG);
    REGISTER_OP(LOGICAL_AND);
    REGISTER_OP(LOGICAL_OR);
    REGISTER_OP(LOGICAL_NOT);
    REGISTER_OP(MAXIMUM);
    REGISTER_OP(MINIMUM);
    REGISTER_OP(NEG);
    REGISTER_OP(NOT_EQUAL);
    REGISTER_OP(POW);
    REGISTER_OP(PRELU);
    REGISTER_OP(ROI_ALIGN);
    REGISTER_OP(ROI_POOLING);
    REGISTER_OP(SQRT);
    REGISTER_OP(RSQRT);
    REGISTER_OP(SELECT);
    // REGISTER_OP(SLICE);
    REGISTER_OP(SPLIT);
    REGISTER_OP(DECONV_2D);
    REGISTER_OP(SIN);
    REGISTER_OP(REDUCE_ALL);
    REGISTER_OP(REDUCE_ANY);
    REGISTER_OP(REDUCE_MAX);
    REGISTER_OP(REDUCE_MIN);
    REGISTER_OP(REDUCE_SUM);
    REGISTER_OP(REDUCE_PROD);
    REGISTER_OP(AXIS_ALIGNED_BBOX_TRANSFORM);
    REGISTER_OP(GENERATE_PROPOSALS);
    REGISTER_OP(RANDOM_MULTINOMIAL);
    REGISTER_OP(HEATMAP_MAX_KEYPOINT);
    REGISTER_OP(BOX_WITH_NMS_LIMIT);
    REGISTER_OP(LOG_SOFTMAX);
    REGISTER_OP(TOPK);
    REGISTER_OP(DETECTION_POSTPROCESSING);
    REGISTER_OP(TILE);
    REGISTER_OP(PAD_V2);
    REGISTER_OP(DATA_CONVERT);
    REGISTER_OP(CAST);

/*customer Op*/
#undef REGISTER_OP
}

NnApiInterpreter::~NnApiInterpreter() {}

int NnApiInterpreter::run(Model* model, bool* modified) {
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
        NNRT_LOGD_PRINT("Convert node %u(%d)", idx, op->type());
        OperationPtr new_operation = (this->*op_container_[op->type()])(model, op, idx);
        if (!new_operation) {
            NNRT_LOGW_PRINT("Build operation: %d, index: %d fail", op->type(), idx);
            return NNA_ERROR_CODE(BAD_DATA);
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

void NnApiInterpreter::replaceOperation(Model* model,
                                        uint32_t op_index,
                                        OperationPtr new_operation) {
    OperationPtr org_operation = model->operation(op_index);
    new_operation->setInputs(org_operation->inputs());
    new_operation->setOutputs(org_operation->outputs());
    new_operation->setFusedType(org_operation->fusedType());
    model->operations()[op_index] = new_operation;
}

FusedType NnApiInterpreter::mapFusedType(int fused_code) {
    FusedType type = FusedType::NONE;
    switch (fused_code) {
        case ANEURALNETWORKS_FUSED_RELU:
            type = FusedType::RELU;
            break;
        case ANEURALNETWORKS_FUSED_RELU1:
            type = FusedType::RELU1;
            break;
        case ANEURALNETWORKS_FUSED_RELU6:
            type = FusedType::RELU6;
            break;
        default:
            break;
    }
    return type;
}

PadType NnApiInterpreter::mapPadType(int code) {
    PadType type = PadType::AUTO;
    switch (code) {
        case ANEURALNETWORKS_PADDING_SAME:
            type = PadType::SAME;
            break;
        case ANEURALNETWORKS_PADDING_VALID:
            type = PadType::VALID;
            break;
        default:
            NNRT_LOGE_PRINT("Invalid padding type(%d)", type);
            assert(false);
            break;
    }
    return type;
}

LshProjectionType NnApiInterpreter::mapLshProjectionType(int value) {
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

FusedType NnApiInterpreter::mapLstmActivationType(int value) {
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

std::vector<int32_t> NnApiInterpreter::convertAxes(int32_t* axes_buffer,
                                                   size_t length,
                                                   size_t dim_num) {
    std::vector<int32_t> axes;
    axes.insert(axes.begin(), axes_buffer, axes_buffer + length);
    return convertAxes(axes, dim_num);
}

std::vector<int32_t> NnApiInterpreter::convertAxes(std::vector<int32_t>& axes, size_t dim_num) {
    std::vector<int32_t> new_axes(axes.size());
    size_t max_size = axes.size() - 1;
    for (size_t i = 0; i < axes.size(); i++) {
        new_axes[i] = convertAxis(axes[max_size - i], dim_num);
    }
    return new_axes;
}

void NnApiInterpreter::fillIntArray(Model* model,
                                    OperationPtr operation,
                                    std::vector<int32_t>& array,
                                    int32_t op_index,
                                    bool is_axis) {
    OperandPtr operand = model->operand(operation->input(op_index));
    int32_t* buffer = model->getBuffer<int32_t>(operand->weak_mem_ref.lock());
    size_t length = operand->size();
    array.clear();
    if (!is_axis) {
        array.insert(array.begin(), buffer, buffer + length);
    } else {
        array = convertAxes(buffer, length, length);
    }
}

// remove scalar operand in [offset, offset+cnt)
void NnApiInterpreter::removeScalarOperand(OperationPtr& op, size_t ofst, size_t cnt) {
    auto start = computeAxis(ofst, op->inputs().size() + 1);  // Map negative to positive
    auto& inputs = op->inputs();
    auto begin = inputs.begin();
    auto end = inputs.begin();
    std::advance(begin, start);
    std::advance(end, start + cnt);
    for (auto i = begin; i != inputs.end() && i != end; ++i) {
        operands_to_remove_.emplace(*i);
    }
    inputs.erase(begin, end);
}

void NnApiInterpreter::truncateOperationIOs(Model* model,
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

OperationPtr NnApiInterpreter::map_ADD(Model* model,
                                       OperationPtr operation,
                                       uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 3, 1);
    resetFusedType(model, operation, 2);
    truncateOperationIOs(model, operation, 2, 1);
    return std::make_shared<AddOperation>();
}

OperationPtr NnApiInterpreter::map_SUB(Model* model,
                                       OperationPtr operation,
                                       uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 3, 1);
    resetFusedType(model, operation, 2);
    truncateOperationIOs(model, operation, 2, 1);
    return std::make_shared<SubOperation>();
}

OperationPtr NnApiInterpreter::map_DIV(Model* model,
                                       OperationPtr operation,
                                       uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 3, 1);
    std::shared_ptr<DivOperation> div = std::make_shared<DivOperation>();
    if (div) {
        div->setVxParam(OverflowPolicy::SATURATE, RoundingPolicy::RTNE, Rounding::RTNE);
        resetFusedType(model, operation, 2);
        truncateOperationIOs(model, operation, 2, 1);
    } else {
        NNRT_LOGE("nnapi_interpreter") << "OOM";
    }

    return div;
}

OperationPtr NnApiInterpreter::map_CONCATENATION(Model* model,
                                                 OperationPtr operation,
                                                 uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, -1, 1);
    std::shared_ptr<ConcatOperation> concat = std::make_shared<ConcatOperation>();
    NNAPI_CHECK_PTR(concat);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    concat->axis = inputs.back()->scalar.int32;
    truncateOperationIOs(model, operation, -2, 1);
    return concat;
}

OperationPtr NnApiInterpreter::map_CONV_2D(Model* model,
                                           OperationPtr operation,
                                           uint32_t operation_index) {
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    std::shared_ptr<Conv2DOperation> conv2d = std::make_shared<Conv2DOperation>();

    std::vector<OperandType> argTypes;
    std::transform(
        inputs.begin(), inputs.end(), std::back_inserter(argTypes), [](const OperandPtr& operand) {
            return operand->type;
        });

    auto argList = api::requirement::nnapi::match("Convolution2D", argTypes);
    if (argList) {
        if (-1 == argList->ArgPos("explicit_pad_left")) {
            // implicit_pad
            auto padTypeIdx = argList->ArgPos("implicit_pad_type");
            conv2d->padType = mapPadType(inputs[padTypeIdx]->scalar.int32);
        } else {
            conv2d->pad[0] = inputs[argList->ArgPos("explicit_pad_left")]->scalar.int32;
            conv2d->pad[1] = inputs[argList->ArgPos("explicit_pad_right")]->scalar.int32;
            conv2d->pad[2] = inputs[argList->ArgPos("explicit_pad_top")]->scalar.int32;
            conv2d->pad[3] = inputs[argList->ArgPos("explicit_pad_bottom")]->scalar.int32;
        }
        // dilation is required in Lowlevel requirement
        conv2d->dilations[0] = 1;
        conv2d->dilations[1] = 1;
        if (-1 != argList->ArgPos("dilation_w") && -1 != argList->ArgPos("dilation_h")) {
            conv2d->dilations[0] = inputs[argList->ArgPos("dilation_w")]->scalar.int32;
            conv2d->dilations[1] = inputs[argList->ArgPos("dilation_h")]->scalar.int32;
        }
        conv2d->setDataLayout(DataLayout::NHWC);
        if (-1 != argList->ArgPos("data_layout")) {
            conv2d->setDataLayout(
                getDataLayout(inputs[argList->ArgPos("data_layout")]->scalar.boolean));
        }

        resetFusedType(model, operation, argList->ArgPos("fuse_code"));
        conv2d->strides[0] = inputs[argList->ArgPos("stride_w")]->scalar.int32;
        conv2d->strides[1] = inputs[argList->ArgPos("stride_h")]->scalar.int32;
    } else {
        NNRT_LOGE_PRINT("convolution 2d argument list not support");
    }

    /* set default dilation value */
    conv2d->setVxParam(OverflowPolicy::SATURATE, RoundingPolicy::TO_ZERO, Rounding::FLOOR);
    truncateOperationIOs(model, operation, 3, 1);
    return conv2d;
}

OperationPtr NnApiInterpreter::map_GROUPED_CONV_2D(Model* model,
                                                   OperationPtr operation,
                                                   uint32_t operation_index) {
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    std::shared_ptr<GroupedConv2DOperation> conv2d = std::make_shared<GroupedConv2DOperation>();
    NNAPI_CHECK_PTR(conv2d);
    conv2d->dilations[0] = 1;
    conv2d->dilations[1] = 1;
    std::vector<OperandType> argTypes;
    std::transform(
        inputs.begin(), inputs.end(), std::back_inserter(argTypes), [](const OperandPtr& operand) {
            return operand->type;
        });

    auto argList = api::requirement::nnapi::match("GroupedConv2DOperation", argTypes);
    if (argList) {
        if (-1 == argList->ArgPos("pad_left")) {
            // implicit_pad
            auto padTypeIdx = argList->ArgPos("implicit_pad");
            conv2d->padType = mapPadType(inputs[padTypeIdx]->scalar.int32);
        } else {
            conv2d->pad[0] = inputs[argList->ArgPos("pad_left")]->scalar.int32;
            conv2d->pad[1] = inputs[argList->ArgPos("pad_right")]->scalar.int32;
            conv2d->pad[2] = inputs[argList->ArgPos("pad_top")]->scalar.int32;
            conv2d->pad[3] = inputs[argList->ArgPos("pad_bottom")]->scalar.int32;
        }
        conv2d->strides[0] = inputs[argList->ArgPos("stride_w")]->scalar.int32;
        conv2d->strides[1] = inputs[argList->ArgPos("stride_h")]->scalar.int32;
        conv2d->groups = inputs[argList->ArgPos("groups_num")]->scalar.int32;
        resetFusedType(model, operation, argList->ArgPos("fuse_code"));
        conv2d->setDataLayout(
            getDataLayout(inputs[argList->ArgPos("data_layout")]->scalar.boolean));
        // Transpose weight for nchw cases
        if (DataLayout::NCHW == conv2d->getDataLayout()) {
            std::vector<uint32_t> permVal = {0, 3, 1, 2};
            auto kernelIdx = argList->ArgPos("kernel");
            if (inputs[kernelIdx]->isConst()) {
                // Set permute flag. Do transepose in ovxlib delegate
                inputs[kernelIdx]->setPerm(permVal);
                inputs[kernelIdx]->dimensions =
                    conv2d->dimensionTrans(inputs[kernelIdx]->dimensions, permVal);
            } else {
                // Insert permute layer for weight as input
                if (!operand_utils::InsertPermuteBeforeOperand(
                    model, operation, operation->inputs()[1], permVal)) {
                        NNRT_LOGE_PRINT("GroupedConv2d: insert permute failed.");
                        assert(false);
                }
            }
        }
    } else {
        NNRT_LOGE_PRINT("GroupedConv2D argument list not support");
    }

    /* set default dilation value */
    conv2d->setVxParam(OverflowPolicy::SATURATE, RoundingPolicy::TO_ZERO, Rounding::FLOOR);
    truncateOperationIOs(model, operation, 3, 1);
    return conv2d;
}

OperationPtr NnApiInterpreter::map_DEPTHWISE_CONV_2D(Model* model,
                                                     OperationPtr operation,
                                                     uint32_t operation_index) {
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    std::shared_ptr<DepthwiseConv2DOperation> conv2d = std::make_shared<DepthwiseConv2DOperation>();

    NNAPI_CHECK_PTR(conv2d);
    auto argList = matchArgList(inputs, "DepthwiseConvolution2D");
    if (argList) {
        if (-1 != argList->ArgPos("explicit_pad_left")) {
            conv2d->pad[0] = inputs[argList->ArgPos("explicit_pad_left")]->scalar.int32;
            conv2d->pad[1] = inputs[argList->ArgPos("explicit_pad_right")]->scalar.int32;
            conv2d->pad[2] = inputs[argList->ArgPos("explicit_pad_top")]->scalar.int32;
            conv2d->pad[3] = inputs[argList->ArgPos("explicit_pad_bottom")]->scalar.int32;
        } else if (-1 != argList->ArgPos("implicit_pad_type")) {
            conv2d->padType =
                mapPadType(inputs[argList->ArgPos("implicit_pad_type")]->scalar.int32);
        } else {
            assert(0);
            NNRT_LOGE("NNAPI_interpreter") << "Argument padding method not found";
        }

        conv2d->strides[0] = inputs[argList->ArgPos("stride_w")]->scalar.int32;
        conv2d->strides[1] = inputs[argList->ArgPos("stride_h")]->scalar.int32;
        conv2d->multiplier = inputs[argList->ArgPos("multiplier")]->scalar.int32;

        conv2d->setDataLayout(DataLayout::NHWC);  // default layout
        if (-1 != argList->ArgPos("data_layout")) {
            conv2d->setDataLayout(
                getDataLayout(inputs[argList->ArgPos("data_layout")]->scalar.boolean));
        }

        conv2d->dilations[0] = 1;
        conv2d->dilations[1] = 1;
        if (-1 != argList->ArgPos("dilation_w")) {
            conv2d->dilations[0] = inputs[argList->ArgPos("dilation_w")]->scalar.int32;
            NNRT_LOGD("NNAPI_interpreter") << "dilation_w = " << conv2d->dilations[0];
        }
        if (-1 != argList->ArgPos("dilation_h")) {
            conv2d->dilations[1] = inputs[argList->ArgPos("dilation_h")]->scalar.int32;
            NNRT_LOGD("NNAPI_interpreter") << "dilation_h = " << conv2d->dilations[1];
        }

        resetFusedType(model, operation, argList->ArgPos("fuse_code"));
        conv2d->setVxParam(OverflowPolicy::SATURATE, RoundingPolicy::TO_ZERO, Rounding::FLOOR);
        truncateOperationIOs(model, operation, 3, 1);
    } else {
        assert(0);
        NNRT_LOGE("NNAPI_interpreter") << "Argument match failed";
    }

    return conv2d;
}

OperationPtr NnApiInterpreter::map_RELU(Model* model,
                                        OperationPtr operation,
                                        uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 1, 1);
    return std::make_shared<ReluOperation>();
}

OperationPtr NnApiInterpreter::map_FULLY_CONNECTED(Model* model,
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
    return fc;
}

OperationPtr NnApiInterpreter::map_RESHAPE(Model* model,
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
        fillIntArray(model, operation, reshape->shape, 1, false);
    }
    truncateOperationIOs(model, operation, 1, 1);
    return reshape;
}

OperationPtr NnApiInterpreter::map_EXPAND_DIMS(Model* model,
                                               OperationPtr operation,
                                               uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 2, 1);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    std::shared_ptr<ReshapeOperation> reshape = std::make_shared<ReshapeOperation>();

    auto argList = matchArgList(inputs, "ExpandDims");
    if (argList) {
        int32_t dimIndex = inputs[argList->ArgPos("dimIndex")]->scalar.int32;
        int32_t inputRank = inputs[argList->ArgPos("input")]->dimensions.size();

        // dimIndex in [-(n+1), n+1] where n is the input rank :
        // CAUTION : reference cpu code not algined with API spec
        if (-1 * (inputRank + 1) <= dimIndex && dimIndex <= inputRank) {
            std::vector<OperandPtr> outputs = model->getOperands(operation->outputs());
            assert(outputs[0]->ndim() > 0);
            reshape->shape =
                std::vector<int32_t>(inputs[0]->dimensions.begin(), inputs[0]->dimensions.end());

            if (dimIndex < 0) {
                dimIndex += (inputRank + 1);
            }

            reshape->shape.insert(reshape->shape.begin() + dimIndex, 1);
            // Fill output shape
            outputs[0]->dimensions =
                std::vector<uint32_t>(reshape->shape.begin(), reshape->shape.end());
        } else {
            NNRT_LOGD_PRINT(" %d -> %d, %d", dimIndex, -1 * (inputRank + 1), inputRank);
            assert(false);
        }

        truncateOperationIOs(model, operation, 1, 1);
    } else {
        NNRT_LOGE("NnApiInterpreter") << "Fatal error argmuent mismatch";
        assert(false);
    }

    return reshape;
}

OperationPtr NnApiInterpreter::map_SOFTMAX(Model* model,
                                           OperationPtr operation,
                                           uint32_t operation_index) {

    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    auto argList = matchArgList(inputs, "SoftmaxOperation");
    auto softmax = std::make_shared<SoftmaxOperation>();
    NNAPI_CHECK_PTR(softmax);
    if (argList) {
        // Set beta param
        switch (inputs[argList->ArgPos("beta")]->type) {
            case OperandType::FLOAT16: {
                half_float::half beta;
                memcpy(&beta,
                       &inputs[argList->ArgPos("beta")]->scalar.float16,
                       sizeof(half_float::half));
                softmax->beta = beta;
                break;
            }
            case OperandType::FLOAT32: {
                softmax->beta = inputs[argList->ArgPos("beta")]->scalar.float32;
                break;
            }
            default:
                assert(false);
                NNRT_LOGE_PRINT("Softmax doesn't support given datatype");
        }
        // Set axis param
        if (-1 != argList->ArgPos("axis")) {
            softmax->axis = inputs[argList->ArgPos("axis")]->scalar.int32;
        } else {
            softmax->axis = -1;
        }
    } else {
        NNRT_LOGE_PRINT("Softmax argument list not support");
        assert(false);
    }
    truncateOperationIOs(model, operation, 1, 1);
    return softmax;
}

OperationPtr NnApiInterpreter::map_TRANSPOSE(Model* model,
                                             OperationPtr operation,
                                             uint32_t operation_index) {
    // NNAPI_CHECK_IO_NUM(operation, 2, 1);
    if (operation->inputs().size() == 2) {
        std::shared_ptr<PermuteOperation> permute = std::make_shared<PermuteOperation>();
        NNAPI_CHECK_PTR(permute);
        std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
        fillIntArray(model, operation, permute->perm, 1, false);
        // For perm is empty and input rank = 2, need to set perm = {1, 0}
        if (permute->perm.empty()) {
            auto input0 = model->operand(operation->input(0));
            if (2 == input0->ndim()) {
                permute->perm = {1, 0};
            } else {
                NNRT_LOGE("The perm of tranpose is null");
                assert(false);
            }
        }
        truncateOperationIOs(model, operation, 1, 1);
        return permute;
    } else {
        return operation;
    }
}

OperationPtr NnApiInterpreter::map_AVERAGE_POOL_2D(Model* model,
                                                   OperationPtr operation,
                                                   uint32_t operation_index) {
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    std::shared_ptr<AveragePool2DOperation> pool = std::make_shared<AveragePool2DOperation>();
    NNAPI_CHECK_PTR(pool);

    pool->setDataLayout(DataLayout::NHWC);

    if (inputs.size() == 10 || inputs.size() == 11)  // V1.2 add a optional data layout
    {
        pool->pad[0] = inputs[1]->scalar.int32;
        pool->pad[1] = inputs[2]->scalar.int32;
        pool->pad[2] = inputs[3]->scalar.int32;
        pool->pad[3] = inputs[4]->scalar.int32;
        pool->strides[0] = inputs[5]->scalar.int32;
        pool->strides[1] = inputs[6]->scalar.int32;
        pool->ksize[0] = inputs[7]->scalar.int32;
        pool->ksize[1] = inputs[8]->scalar.int32;
        resetFusedType(model, operation, 9);
    } else {
        pool->padType = mapPadType(inputs[1]->scalar.int32);
        pool->strides[0] = inputs[2]->scalar.int32;
        pool->strides[1] = inputs[3]->scalar.int32;
        pool->ksize[0] = inputs[4]->scalar.int32;
        pool->ksize[1] = inputs[5]->scalar.int32;
        resetFusedType(model, operation, 6);
    }

    if (inputs.size() == 11 || inputs.size() == 8) {
        pool->setDataLayout(getDataLayout(inputs.back()->scalar.boolean));
    }

    pool->poolMode = PoolMode::VALID;
    pool->setVxParam(OverflowPolicy::WRAP, RoundingPolicy::TO_ZERO, Rounding::FLOOR);
    truncateOperationIOs(model, operation, 1, 1);
    return pool;
}

OperationPtr NnApiInterpreter::map_MAX_POOL_2D(Model* model,
                                               OperationPtr operation,
                                               uint32_t operation_index) {
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    std::shared_ptr<MaxPool2DOperation> pool = std::make_shared<MaxPool2DOperation>();
    NNAPI_CHECK_PTR(pool);
    pool->setDataLayout(DataLayout::NHWC);
    if (inputs.size() == 10) {
        pool->pad[0] = inputs[1]->scalar.int32;
        pool->pad[1] = inputs[2]->scalar.int32;
        pool->pad[2] = inputs[3]->scalar.int32;
        pool->pad[3] = inputs[4]->scalar.int32;
        pool->strides[0] = inputs[5]->scalar.int32;
        pool->strides[1] = inputs[6]->scalar.int32;
        pool->ksize[0] = inputs[7]->scalar.int32;
        pool->ksize[1] = inputs[8]->scalar.int32;
        resetFusedType(model, operation, 9);
    } else {
        pool->padType = mapPadType(inputs[1]->scalar.int32);
        pool->strides[0] = inputs[2]->scalar.int32;
        pool->strides[1] = inputs[3]->scalar.int32;
        pool->ksize[0] = inputs[4]->scalar.int32;
        pool->ksize[1] = inputs[5]->scalar.int32;
        resetFusedType(model, operation, 6);
    }
    pool->setVxParam(OverflowPolicy::WRAP, RoundingPolicy::TO_ZERO, Rounding::FLOOR);
    truncateOperationIOs(model, operation, 1, 1);
    return pool;
}

OperationPtr NnApiInterpreter::map_SQUEEZE(Model* model,
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

OperationPtr NnApiInterpreter::map_PAD(Model* model,
                                       OperationPtr operation,
                                       uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 2, 1);
    std::shared_ptr<PadOperation> pad = std::make_shared<PadOperation>();
    NNAPI_CHECK_PTR(pad);
    pad->setDataLayout(DataLayout::NHWC);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    int32_t* padding = model->getBuffer<int32_t>(inputs[1]->weak_mem_ref.lock());
    pad->padFront.resize(inputs[1]->dimensions[0]);
    pad->padBack.resize(inputs[1]->dimensions[0]);
    convert2DPadding(padding, inputs[1]->size(), pad->padFront.data(), pad->padBack.data());
    pad->padFront = pad->padFront;
    pad->padBack = pad->padBack;
    pad->padValue = 0.0f;
    pad->padMode = PadMode::CONSTANT;
    truncateOperationIOs(model, operation, 1, 1);
    return pad;
}

OperationPtr NnApiInterpreter::map_PAD_V2(Model* model,
                                       OperationPtr operation,
                                       uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 3, 1);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    auto argList = matchArgList(inputs, "PadV2Operation");
    OperationPtr padV2;
    if (argList) {
        auto paddingIds = argList->ArgPos("padding");
        int32_t* padding = model->getBuffer<int32_t>(inputs[paddingIds]->weak_mem_ref.lock());
        auto inputRank = inputs[paddingIds]->dimensions[0];
        switch (inputs[argList->ArgPos("pad_value")]->type) {
            case OperandType::FLOAT16: {
                // TODO: Add float16 pad value support
                break;
            }
            case OperandType::FLOAT32: {
                // TODO: Add float32 pad value support
                break;
            }
            case OperandType::INT32: {
                padV2 = std::make_shared<PadV2Operation<int32_t>>();
                auto op = std::dynamic_pointer_cast<PadV2Operation<int32_t>>(padV2);
                op->setDataLayout(DataLayout::NHWC);
                op->padValue = inputs[argList->ArgPos("pad_value")]->scalar.int32;
                op->padFront.resize(inputRank);
                op->padBack.resize(inputRank);
                convert2DPadding(padding, inputs[paddingIds]->size(), op->padFront.data(), op->padBack.data());
                op->padMode = PadMode::CONSTANT;
                break;
            }
            default:
                NNRT_LOGE_PRINT("PadV2 doesn't support given datatype");
                assert(false);
        }
    } else {
        NNRT_LOGE_PRINT("PadV2 argument list not support");
        assert(false);
    }
    truncateOperationIOs(model, operation, 1, 1);
    return padV2;
}

OperationPtr NnApiInterpreter::map_MUL(Model* model,
                                       OperationPtr operation,
                                       uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 3, 1);
    std::shared_ptr<MulOperation> mul = std::make_shared<MulOperation>();
    NNAPI_CHECK_PTR(mul);
    mul->setVxParam(OverflowPolicy::SATURATE, RoundingPolicy::RTNE);
    resetFusedType(model, operation, 2);
    truncateOperationIOs(model, operation, 2, 1);
    return mul;
}

OperationPtr NnApiInterpreter::map_MEAN(Model* model,
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

OperationPtr NnApiInterpreter::map_SPACE_TO_DEPTH(Model* model,
                                                  OperationPtr operation,
                                                  uint32_t operation_index) {
    std::shared_ptr<SpaceToDepthOperation> space2depth = std::make_shared<SpaceToDepthOperation>();
    NNAPI_CHECK_PTR(space2depth);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());

    std::vector<OperandType> argTypes;
    std::transform(
        inputs.begin(), inputs.end(), std::back_inserter(argTypes), [](const OperandPtr& operand) {
            return operand->type;
        });

    auto argList = api::requirement::nnapi::match("Space2DepthOperation", argTypes);
    if (argList) {
        auto inputOperand = inputs[argList->ArgPos("input")];
        auto outputOperand = model->getOperands(operation->outputs())[0];
        // No dynamic shape branch
        if (!nnrt::operand_utils::IsDynamicShape(inputOperand) &&
            !nnrt::operand_utils::IsDynamicShape(outputOperand)) {
            space2depth->blockSize[0] = inputs[argList->ArgPos("block_size")]->scalar.int32;
            space2depth->blockSize[1] = inputs[argList->ArgPos("block_size")]->scalar.int32;
            if (-1 != argList->ArgPos("data_layout")) {
                space2depth->setDataLayout(
                    getDataLayout(inputs[argList->ArgPos("data_layout")]->scalar.boolean));
            } else {
                // Default data layout is NHWC
                space2depth->setDataLayout(DataLayout::NHWC);
            }
        } else {
            // TODO: support dynamic input tensor shape
            NNRT_LOGE_PRINT("Dynamic shape not support");
            assert(false);
        }

    } else {
        NNRT_LOGE_PRINT("Space to depth argument list not support.");
    }
    truncateOperationIOs(model, operation, 1, 1);
    return space2depth;
}

OperationPtr NnApiInterpreter::map_DEPTH_TO_SPACE(Model* model,
                                                  OperationPtr operation,
                                                  uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 2, 1);
    std::shared_ptr<DepthToSpaceOperation> dp_to_sp = std::make_shared<DepthToSpaceOperation>();
    NNAPI_CHECK_PTR(dp_to_sp);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    dp_to_sp->blockSize = inputs[1]->scalar.int32;
    dp_to_sp->setDataLayout(DataLayout::NHWC);
    truncateOperationIOs(model, operation, 1, 1);
    return dp_to_sp;
}

OperationPtr NnApiInterpreter::map_SPACE_TO_BATCH_ND(Model* model,
                                                     OperationPtr operation,
                                                     uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 3, 1);
    std::shared_ptr<SpaceToBatchNDOperation> sp_to_bp = std::make_shared<SpaceToBatchNDOperation>();
    NNAPI_CHECK_PTR(sp_to_bp);
    sp_to_bp->setDataLayout(DataLayout::NHWC);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    if (inputs[1]->isConst() && inputs[2]->isConst()) {
        int32_t* buffer = model->getBuffer<int32_t>(inputs[1]->weak_mem_ref.lock());
        sp_to_bp->blockSize.assign(buffer, buffer + inputs[1]->size());
        buffer = model->getBuffer<int32_t>(inputs[2]->weak_mem_ref.lock());
        sp_to_bp->padFront.resize(inputs[0]->ndim() - 2);
        sp_to_bp->padBack.resize(inputs[0]->ndim() - 2);
        convert2DPadding(
            buffer, inputs[2]->size(), sp_to_bp->padFront.data(), sp_to_bp->padBack.data());
    } else {
        NNRT_LOGW_PRINT("Not support dynamic SPACE_TO_BATCH_ND.");
        assert(false);
    }
    truncateOperationIOs(model, operation, 1, 1);
    return sp_to_bp;
}

OperationPtr NnApiInterpreter::map_BATCH_TO_SPACE_ND(Model* model,
                                                     OperationPtr operation,
                                                     uint32_t operation_index) {
    std::shared_ptr<BatchToSpaceNDOperation> bp_to_sp = std::make_shared<BatchToSpaceNDOperation>();
    NNAPI_CHECK_PTR(bp_to_sp);
    bp_to_sp->setDataLayout(DataLayout::NHWC);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    if (inputs[1]->isConst()) {
        int32_t* buffer = model->getBuffer<int32_t>(inputs[1]->weak_mem_ref.lock());
        bp_to_sp->blockSize.assign(buffer, buffer + inputs[1]->size());
    }

    if (inputs.size() == 3) {
        bp_to_sp->setDataLayout(getDataLayout(inputs[2]->scalar.boolean));
    }

    bp_to_sp->cropStart.resize(inputs[0]->ndim() - 2);
    bp_to_sp->cropEnd.resize(inputs[0]->ndim() - 2);

    truncateOperationIOs(model, operation, 1, 1);
    return bp_to_sp;
}

OperationPtr NnApiInterpreter::map_RESIZE_BILINEAR(Model* model,
                                                   OperationPtr operation,
                                                   uint32_t operation_index) {
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    std::shared_ptr<ResizeBilinearOperation> resize = std::make_shared<ResizeBilinearOperation>();
    NNAPI_CHECK_PTR(resize);

    std::vector<OperandType> argTypes;
    std::transform(
        inputs.begin(), inputs.end(), std::back_inserter(argTypes), [](const OperandPtr& operand) {
            return operand->type;
        });

    auto argList = api::requirement::nnapi::match("ResizeBilinearOperation", argTypes);
    if (argList) {
        // default layout
        resize->setDataLayout(DataLayout::NHWC);
        // layout were set
        if (-1 != argList->ArgPos("data_layout")) {
            resize->setDataLayout(
                getDataLayout(inputs[argList->ArgPos("data_layout")]->scalar.boolean));
        }
        auto inputOperand = inputs[argList->ArgPos("input")];
        auto outputOperand = model->getOperands(operation->outputs())[0];
        // No dynamic shape branch
        if (!nnrt::operand_utils::IsDynamicShape(inputOperand) &&
            !nnrt::operand_utils::IsDynamicShape(outputOperand)) {
            if (-1 != argList->ArgPos("output_height")) {
                // give output height and width
                // resize->outputHeight = inputs[argList->ArgPos("output_height")]->scalar.int32;
                // resize->outputWidth = inputs[argList->ArgPos("output_width")]->scalar.int32;
                // Note: The order of height and width is't compatible with the spec,
                // next code should be replaced by above code, once the problem is fixed.
                resize->outputWidth = inputs[argList->ArgPos("output_height")]->scalar.int32;
                resize->outputHeight = inputs[argList->ArgPos("output_width")]->scalar.int32;

            } else {
                // give scale
                uint32_t orgHeight = 0;
                uint32_t orgWidth = 0;
                if (DataLayout::NCHW == resize->getDataLayout()) {
                    orgHeight = inputOperand->dimensions[2];
                    orgWidth = inputOperand->dimensions[3];
                } else if (DataLayout::NHWC == resize->getDataLayout()) {
                    orgHeight = inputOperand->dimensions[1];
                    orgWidth = inputOperand->dimensions[2];
                }
                if (inputs[argList->ArgPos("height_scale")]->type == OperandType::FLOAT32) {
                    // scale is float32
                    float heightScale = inputs[argList->ArgPos("height_scale")]->scalar.float32;
                    float widthScale = inputs[argList->ArgPos("width_scale")]->scalar.float32;
                    // resize->outputHeight = uint32_t(orgHeight * heightScale);
                    // resize->outputWidth = uint32_t(orgWidth * widthScale);
                    resize->outputHeight = uint32_t(orgHeight * widthScale);
                    resize->outputWidth = uint32_t(orgWidth * heightScale);
                } else {
                    // scale is float16
                    // TODO: support float16
                    NNRT_LOGE_PRINT("Float16 scale not support");
                    assert(false);
                }
            }
        } else {
            // TODO: support dynamic input tensor shape
            NNRT_LOGE_PRINT("Dynamic shape not support");
            assert(false);
        }
    } else {
        NNRT_LOGE_PRINT("Resize bilinear argument list not support");
    }
    truncateOperationIOs(model, operation, 1, 1);
    return resize;
}

OperationPtr NnApiInterpreter::map_RESIZE_NEAREST(Model* model,
                                                  OperationPtr operation,
                                                  uint32_t operation_index) {
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    std::shared_ptr<ResizeNearestNeighborOperation> resize =
        std::make_shared<ResizeNearestNeighborOperation>();
    NNAPI_CHECK_PTR(resize);

    std::vector<OperandType> argTypes;
    std::transform(
        inputs.begin(), inputs.end(), std::back_inserter(argTypes), [](const OperandPtr& operand) {
            return operand->type;
        });

    auto argList = api::requirement::nnapi::match("ResizeNearestNeighborOperation", argTypes);
    if (argList) {
        resize->setDataLayout(
            getDataLayout(inputs[argList->ArgPos("data_layout")]->scalar.boolean));
        auto inputOperand = inputs[argList->ArgPos("input")];
        auto outputOperand = model->getOperands(operation->outputs())[0];
        // No dynamic shape branch
        if (!nnrt::operand_utils::IsDynamicShape(inputOperand) &&
            !nnrt::operand_utils::IsDynamicShape(outputOperand)) {
            if (-1 != argList->ArgPos("output_height")) {
                // give output height and width
                // resize->outputHeight = inputs[argList->ArgPos("output_height")]->scalar.int32;
                // resize->outputWidth = inputs[argList->ArgPos("output_width")]->scalar.int32;
                // Note: The order of height and width is't compatible with the spec,
                // next code should be replaced by above code, once the problem is fixed.
                resize->outputWidth = inputs[argList->ArgPos("output_height")]->scalar.int32;
                resize->outputHeight = inputs[argList->ArgPos("output_width")]->scalar.int32;

            } else {
                // give scale
                uint32_t orgHeight = 0;
                uint32_t orgWidth = 0;
                if (DataLayout::NCHW == resize->getDataLayout()) {
                    orgHeight = inputOperand->dimensions[2];
                    orgWidth = inputOperand->dimensions[3];
                } else if (DataLayout::NHWC == resize->getDataLayout()) {
                    orgHeight = inputOperand->dimensions[1];
                    orgWidth = inputOperand->dimensions[2];
                }
                if (inputs[argList->ArgPos("height_scale")]->type == OperandType::FLOAT32) {
                    // scale is float32
                    float heightScale = inputs[argList->ArgPos("height_scale")]->scalar.float32;
                    float widthScale = inputs[argList->ArgPos("width_scale")]->scalar.float32;
                    // resize->outputHeight = uint32_t(orgHeight * heightScale);
                    // resize->outputWidth = uint32_t(orgWidth * widthScale);
                    resize->outputHeight = uint32_t(orgHeight * widthScale);
                    resize->outputWidth = uint32_t(orgWidth * heightScale);

                } else {
                    // scale is float16
                    // TODO: support float16
                    NNRT_LOGE_PRINT("Float16 scale not support");
                    assert(false);
                }
            }
        } else {
            // TODO: support dynamic input tensor shape
            NNRT_LOGE_PRINT("Dynamic shape not support");
            assert(false);
        }
    } else {
        NNRT_LOGE_PRINT("Resize nearest neighbor argument list not support");
    }
    truncateOperationIOs(model, operation, 1, 1);
    return resize;
}

OperationPtr NnApiInterpreter::map_LOCAL_RESPONSE_NORMALIZATION(Model* model,
                                                                OperationPtr operation,
                                                                uint32_t operation_index) {
    std::shared_ptr<LocalResponseNormOperation> lrn =
        std::make_shared<LocalResponseNormOperation>();
    NNAPI_CHECK_PTR(lrn);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());

    std::vector<OperandType> argTypes;
    std::transform(
        inputs.begin(), inputs.end(), std::back_inserter(argTypes), [](const OperandPtr& operand) {
            return operand->type;
        });

    auto argList = api::requirement::nnapi::match("LocalResponseNormOperation", argTypes);
    if (argList) {
        auto inputOperand = inputs[argList->ArgPos("input")];
        auto outputOperand = model->getOperands(operation->outputs())[0];
        // No dynamic shape branch
        if (!nnrt::operand_utils::IsDynamicShape(inputOperand) &&
            !nnrt::operand_utils::IsDynamicShape(outputOperand)) {
            lrn->radius = inputs[argList->ArgPos("radius")]->scalar.int32;
            lrn->bias = inputs[argList->ArgPos("bias")]->scalar.float32;
            lrn->scale = inputs[argList->ArgPos("alpha")]->scalar.float32;
            lrn->exponent = inputs[argList->ArgPos("beta")]->scalar.float32;
            if (-1 != argList->ArgPos("axis")) {
                lrn->axis = inputs[argList->ArgPos("axis")]->scalar.int32;
            } else {
                // default axis = -1
                lrn->axis = -1;
            }
        } else {
            // TODO: support dynamic input tensor shape
            NNRT_LOGE_PRINT("Dynamic shape not support");
            assert(false);
        }

    } else {
        NNRT_LOGE_PRINT("Local response normalization argument list not support");
    }
    truncateOperationIOs(model, operation, 1, 1);
    return lrn;
}

OperationPtr NnApiInterpreter::map_L2_NORMALIZATION(Model* model,
                                                    OperationPtr operation,
                                                    uint32_t operation_index) {
    std::shared_ptr<L2NormOperation> l2norm = std::make_shared<L2NormOperation>();
    NNAPI_CHECK_PTR(l2norm);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());

    std::vector<OperandType> argTypes;
    std::transform(
        inputs.begin(), inputs.end(), std::back_inserter(argTypes), [](const OperandPtr& operand) {
            return operand->type;
        });

    auto argList = api::requirement::nnapi::match("L2NormOperation", argTypes);
    if (argList) {
        auto inputOperand = inputs[argList->ArgPos("input")];
        auto outputOperand = model->getOperands(operation->outputs())[0];
        // No dynamic shape branch
        if (!nnrt::operand_utils::IsDynamicShape(inputOperand) &&
            !nnrt::operand_utils::IsDynamicShape(outputOperand)) {
            if (-1 != argList->ArgPos("axis")) {
                l2norm->axis = inputs[argList->ArgPos("axis")]->scalar.int32;
            } else {
                // default axis = -1
                l2norm->axis = -1;
            }
        } else {
            // TODO: support dynamic input tensor shape
            NNRT_LOGE_PRINT("Dynamic shape not support");
            assert(false);
        }

    } else {
        NNRT_LOGE_PRINT("L2 response argument list not support");
    }
    truncateOperationIOs(model, operation, 1, 1);
    return l2norm;
}

OperationPtr NnApiInterpreter::map_RNN(Model* model,
                                       OperationPtr operation,
                                       uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 6, 2);
    std::shared_ptr<RnnOperation> rnn = std::make_shared<RnnOperation>();
    NNAPI_CHECK_PTR(rnn);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());

    // RNN's activation is NeuralNetwork::FuseType
    rnn->activation = mapFusedType(inputs[5]->scalar.int32);
    truncateOperationIOs(model, operation, 5, 2);
    return rnn;
}

OperationPtr NnApiInterpreter::map_UNIDIRECTIONAL_SEQUENCE_RNN(Model* model,
                                                               OperationPtr operation,
                                                               uint32_t operation_index) {
    std::shared_ptr<UnidirectionalSequenceRnnOperation> op =
        std::make_shared<UnidirectionalSequenceRnnOperation>();
    NNAPI_CHECK_PTR(op);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    auto argList = matchArgList(inputs, "UnidirectionalSequenceRnnOperation");
    if (argList) {
        op->activation = mapFusedType(inputs[argList->ArgPos("activation")]->scalar.int32);
        op->timeMajor = inputs[argList->ArgPos("timeMajor")]->scalar.boolean;
        // Insert Fp16ToFp32 data convert layer before bias if the type of bias is float16
        if (OperandType::TENSOR_FLOAT16 == inputs[argList->ArgPos("bias")]->type) {
            if (!operand_utils::InsertFp16ToFp32LayerBeforeOperand(
                    model, operation, inputs[argList->ArgPos("bias")])) {
                NNRT_LOGE_PRINT("Insert Fp16ToFp32 Layer failed.");
                assert(false);
            }
        }
    } else {
        NNRT_LOGE_PRINT("UnidirectionalSequenceRnn argument list not support");
        assert(false);
    }
    truncateOperationIOs(model, operation, 5, 1);
    return op;
}

OperationPtr NnApiInterpreter::map_BIDIRECTIONAL_SEQUENCE_RNN(Model* model,
                                                              OperationPtr operation,
                                                              uint32_t operation_index) {
    std::shared_ptr<BidirectionalSequenceRnnOperation> op =
        std::make_shared<BidirectionalSequenceRnnOperation>();
    NNAPI_CHECK_PTR(op);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    auto argList = matchArgList(inputs, "BidirectionalSequenceRnnOperation");
    if (argList) {
        op->activation = mapFusedType(inputs[argList->ArgPos("activation")]->scalar.int32);
        op->timeMajor = inputs[argList->ArgPos("timeMajor")]->scalar.boolean;
        op->mergeOutputs = inputs[argList->ArgPos("mergeOutputs")]->scalar.boolean;
        if (OperandType::TENSOR_FLOAT16 == inputs[argList->ArgPos("fw_input_bias")]->type) {
            if (!(operand_utils::InsertFp16ToFp32LayerBeforeOperand(
                    model, operation, inputs[argList->ArgPos("fw_input_bias")]) &&
                operand_utils::InsertFp16ToFp32LayerBeforeOperand(
                    model, operation, inputs[argList->ArgPos("bw_input_bias")]))) {
                NNRT_LOGE_PRINT("Insert Fp16ToFp32 Layer failed.");
                assert(false);
            }
        }
    } else {
        NNRT_LOGE_PRINT("BidirectionalSequenceRnn argument list not support");
        assert(false);
    }
    truncateOperationIOs(model, operation, 12, 2);
    return op;
}

OperationPtr NnApiInterpreter::map_LSTM(Model* model,
                                        OperationPtr operation,
                                        uint32_t operation_index) {
    std::shared_ptr<LstmUnitOperation> new_op = std::make_shared<LstmUnitOperation>();
    NNAPI_CHECK_PTR(new_op);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    auto argList = matchArgList(inputs, "LstmUnit");

    if (argList) {
        if (inputs[0]->type == nnrt::OperandType::TENSOR_FLOAT32) {
            NNRT_LOGI_PRINT("LSMT float32, proj_clip at %d", argList->ArgPos("proj_clip"));
            new_op->activation =
                mapLstmActivationType(inputs[argList->ArgPos("activation")]->scalar.int32);
            new_op->cellClip = inputs[argList->ArgPos("cell_clip")]->scalar.float32;
            new_op->projClip = inputs[argList->ArgPos("proj_clip")]->scalar.float32;
        } else {
            assert(false);
            NNRT_LOGE_PRINT("TODO NNRT not support float16 datatype yet");
            // new_op->activation =
            // mapLstmActivationType(inputs[argList->ArgPos("activation")]->scalar.int32);
            // new_op->cellClip = inputs[argList->ArgPos("cell_clip")]->scalar.float16;
            // new_op->projClip = inputs[argList->ArgPos("proj_clip")]->scalar.float16;
            new_op.reset();
        }
        removeScalarOperand(operation, argList->ArgPos("activation"), 3);
    }

    if (argList->ArgPos("weight_norm_input") != -1) {
        // TODO {Xiang} port LSTM to new ovxlib API
        NNRT_LOGE_PRINT("Don't support layer normal yet: remove related input");
        truncateOperationIOs(model, operation, 20, 4);
    }

    auto input_num = model->getOperands(operation->inputs()).size();
    // TODO {Xiang} port LSTM to new ovxlib API
    while (input_num < LstmUnitOperation::INPUT_COUNT) {
        operation->inputs().emplace_back(-1);
        NNRT_LOGD_PRINT("Append Inputs at [%d]", input_num);
        ++input_num;
    }

    NNAPI_CHECK_IO_NUM(operation, LstmUnitOperation::INPUT_COUNT, LstmUnitOperation::OUTPUT_COUNT);

    return new_op;
}

OperationPtr NnApiInterpreter::map_UNIDIRECTIONAL_SEQUENCE_LSTM(Model* model,
                                                                OperationPtr operation,
                                                                uint32_t operation_index) {
    std::shared_ptr<UnidirectionalSequenceLstmOperation> op =
        std::make_shared<UnidirectionalSequenceLstmOperation>();
    NNAPI_CHECK_PTR(op);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    auto argList = matchArgList(inputs, "UnidirectionalSequenceLstmOperation");
    if (argList) {
        op->activation = mapLstmActivationType(inputs[argList->ArgPos("activation")]->scalar.int32);
        op->timeMajor = inputs[argList->ArgPos("timeMajor")]->scalar.boolean;
        op->cell_clip = inputs[argList->ArgPos("cell_clip")]->scalar.float32;
        op->proj_clip = inputs[argList->ArgPos("proj_clip")]->scalar.float32;
        if (OperandType::TENSOR_FLOAT16 == inputs[argList->ArgPos("bias_i")]->type) {
            if (!(operand_utils::InsertFp16ToFp32LayerBeforeOperand(
                      model, operation, inputs[argList->ArgPos("bias_i")]) &&
                  operand_utils::InsertFp16ToFp32LayerBeforeOperand(
                      model, operation, inputs[argList->ArgPos("bias_f")]) &&
                  operand_utils::InsertFp16ToFp32LayerBeforeOperand(
                      model, operation, inputs[argList->ArgPos("bias_c")]) &&
                  operand_utils::InsertFp16ToFp32LayerBeforeOperand(
                      model, operation, inputs[argList->ArgPos("bias_o")]))) {
                NNRT_LOGE_PRINT("Insert Fp16ToFp32 Layer failed.");
                assert(false);
            }
        }
        removeScalarOperand(operation, argList->ArgPos("activation"), 4);
    } else {
        NNRT_LOGE_PRINT("UnidirectionalSequenceLstm argument list not support");
        assert(false);
    }
    truncateOperationIOs(model, operation, 29, 1);
    return op;
}

OperationPtr NnApiInterpreter::map_BIDIRECTIONAL_SEQUENCE_LSTM(Model* model,
                                                               OperationPtr operation,
                                                               uint32_t operation_index) {
    std::shared_ptr<BidirectionalSequenceLstmOperation> op =
        std::make_shared<BidirectionalSequenceLstmOperation>();
    NNAPI_CHECK_PTR(op);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    auto argList = matchArgList(inputs, "BidirectionalSequenceLstmOperation");
    if (argList) {
        op->activation = mapLstmActivationType(inputs[argList->ArgPos("activation")]->scalar.int32);
        op->timeMajor = inputs[argList->ArgPos("time_major")]->scalar.boolean;
        op->mergeOutputs = inputs[argList->ArgPos("merge_outputs")]->scalar.boolean;
        op->cell_clip = inputs[argList->ArgPos("cell_clip")]->scalar.float32;
        op->proj_clip = inputs[argList->ArgPos("proj_clip")]->scalar.float32;
        if (OperandType::TENSOR_FLOAT16 == inputs[argList->ArgPos("fw_input_bias_i")]->type) {
            if (!(operand_utils::InsertFp16ToFp32LayerBeforeOperand(
                      model, operation, inputs[argList->ArgPos("fw_input_bias_i")]) &&
                  operand_utils::InsertFp16ToFp32LayerBeforeOperand(
                      model, operation, inputs[argList->ArgPos("fw_input_bias_f")]) &&
                  operand_utils::InsertFp16ToFp32LayerBeforeOperand(
                      model, operation, inputs[argList->ArgPos("fw_input_bias_c")]) &&
                  operand_utils::InsertFp16ToFp32LayerBeforeOperand(
                      model, operation, inputs[argList->ArgPos("fw_input_bias_o")]) &&
                  operand_utils::InsertFp16ToFp32LayerBeforeOperand(
                      model, operation, inputs[argList->ArgPos("bw_input_bias_i")]) &&
                  operand_utils::InsertFp16ToFp32LayerBeforeOperand(
                      model, operation, inputs[argList->ArgPos("bw_input_bias_f")]) &&
                  operand_utils::InsertFp16ToFp32LayerBeforeOperand(
                      model, operation, inputs[argList->ArgPos("bw_input_bias_c")]) &&
                  operand_utils::InsertFp16ToFp32LayerBeforeOperand(
                      model, operation, inputs[argList->ArgPos("bw_input_bias_o")]))) {
                NNRT_LOGE_PRINT("Insert Fp16ToFp32 layer failed.");
                assert(false);
            }
            // Insert data converter layer for optional float16 bias
            if (-1 != argList->ArgPos("fw_input_bias_proj")) {
                if (!operand_utils::InsertFp16ToFp32LayerBeforeOperand(
                        model, operation, inputs[argList->ArgPos("fw_input_bias_proj")])) {
                    NNRT_LOGE_PRINT("Insert Fp16ToFp32 layer failed.");
                    assert(false);
                }
            }
            if (-1 != argList->ArgPos("bw_input_bias_proj")) {
                if (!operand_utils::InsertFp16ToFp32LayerBeforeOperand(
                        model, operation, inputs[argList->ArgPos("bw_input_bias_proj")])) {
                    NNRT_LOGE_PRINT("Insert Fp16ToFp32 layer failed.");
                    assert(false);
                }
            }
        }
        removeScalarOperand(operation, argList->ArgPos("activation"), 5);
    } else {
        NNRT_LOGE_PRINT("BidirectionalSequenceLstm argument list not support");
        assert(false);
    }
    // truncateOperationIOs(model, operation, 56, 2);
    return op;
}

OperationPtr NnApiInterpreter::map_SVDF(Model* model,
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

OperationPtr NnApiInterpreter::map_LSH_PROJECTION(Model* model,
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

OperationPtr NnApiInterpreter::map_L2_POOL_2D(Model* model,
                                              OperationPtr operation,
                                              uint32_t operation_index) {
    std::shared_ptr<L2Pool2DOperation> pool = std::make_shared<L2Pool2DOperation>();
    NNAPI_CHECK_PTR(pool);
    pool->setDataLayout(DataLayout::NHWC);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    if (inputs.size() == 10 || inputs.size() == 11 /*API LEVEL 29*/) {
        pool->pad[0] = inputs[1]->scalar.int32;
        pool->pad[1] = inputs[2]->scalar.int32;
        pool->pad[2] = inputs[3]->scalar.int32;
        pool->pad[3] = inputs[4]->scalar.int32;
        pool->strides[0] = inputs[5]->scalar.int32;
        pool->strides[1] = inputs[6]->scalar.int32;
        pool->ksize[0] = inputs[7]->scalar.int32;
        pool->ksize[1] = inputs[8]->scalar.int32;
        resetFusedType(model, operation, 9);

        // TODO: if io_in_nchw is true, no more permute required
        if (inputs.size() == 11) {
            pool->setDataLayout(getDataLayout(inputs[10]->scalar.boolean));
        }
    } else if (inputs.size() == 7 || inputs.size() == 8 /*API LEVEL 29*/) {
        pool->padType = mapPadType(inputs[1]->scalar.int32);
        pool->strides[0] = inputs[2]->scalar.int32;
        pool->strides[1] = inputs[3]->scalar.int32;
        pool->ksize[0] = inputs[4]->scalar.int32;
        pool->ksize[1] = inputs[5]->scalar.int32;
        resetFusedType(model, operation, 6);

        // TODO: if io_in_nchw is true, no more permute required
        if (inputs.size() == 8 && inputs[7]) {
            pool->setDataLayout(DataLayout::NCHW);
        }
    } else {
        NNRT_LOGE_PRINT("Number of input parameter not valid");
        assert(false);
    }
    pool->setVxParam(OverflowPolicy::WRAP, RoundingPolicy::TO_ZERO, Rounding::FLOOR);
    truncateOperationIOs(model, operation, 1, 1);
    return pool;
}

OperationPtr NnApiInterpreter::map_STRIDED_SLICE(Model* model,
                                                 OperationPtr operation,
                                                 uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 7, 1);
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
    truncateOperationIOs(model, operation, 1, 1);
    return new_op;
}

OperationPtr NnApiInterpreter::map_DECONV_2D(Model* model,
                                             OperationPtr operation,
                                             uint32_t operation_index) {
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    std::shared_ptr<Deconv2DOperation> deconv2d;
    auto argList = matchArgList(inputs, "TransposeConv2DOperation");
    if (argList) {
        if (argList->ArgPos("output_shape") != -1) {
            // TODO: We donot support this case,
            // It needs shape inference to convert padType to pad
            // op->padType = mapPadtype(inputs[argList->ArgPos("impilicit_pad")]->scalar.int32);
            NNRT_LOGE_PRINT("Transpose conv2d doesn't support implicit padding.");
            return nullptr;
        }
        deconv2d = std::make_shared<Deconv2DOperation>();
        NNAPI_CHECK_PTR(deconv2d);
        deconv2d->pad[0] = inputs[argList->ArgPos("pad_left")]->scalar.int32;
        deconv2d->pad[1] = inputs[argList->ArgPos("pad_top")]->scalar.int32;
        deconv2d->pad[2] = inputs[argList->ArgPos("pad_right")]->scalar.int32;
        deconv2d->pad[3] = inputs[argList->ArgPos("pad_bottom")]->scalar.int32;
        deconv2d->strides[0] = inputs[argList->ArgPos("stride_w")]->scalar.int32;
        deconv2d->strides[1] = inputs[argList->ArgPos("stride_h")]->scalar.int32;
        resetFusedType(model, operation, argList->ArgPos("fuse_code"));
        deconv2d->setDataLayout(getDataLayout(inputs[argList->ArgPos("layout")]->scalar.boolean));

        // Transpose weight for nchw cases
        if (DataLayout::NCHW == deconv2d->getDataLayout()) {
            std::vector<uint32_t> permVal = {0, 3, 1, 2};
            auto kernelIdx = argList->ArgPos("kernel");
            if (inputs[kernelIdx]->isConst()) {
                // Set permute flag. Do transepose in ovxlib delegate
                inputs[kernelIdx]->setPerm(permVal);
                inputs[kernelIdx]->dimensions =
                    deconv2d->dimensionTrans(inputs[kernelIdx]->dimensions, permVal);
            } else {
                // Insert permute layer for weight as input
                if (!operand_utils::InsertPermuteBeforeOperand(
                    model, operation, operation->inputs()[1], permVal)) {
                        NNRT_LOGE_PRINT("Deconv2D: insert permute failed.");
                        assert(false);
                }
            }
        }
    } else {
        NNRT_LOGE_PRINT("Transpose conv2d argument list not support");
        assert(false);
    }

    /* set default dilation value */
    deconv2d->setVxParam(OverflowPolicy::SATURATE, RoundingPolicy::TO_ZERO, Rounding::FLOOR);
    truncateOperationIOs(model, operation, 3, 1);
    return deconv2d;
}

OperationPtr NnApiInterpreter::map_TOPK(Model* model,
                                        OperationPtr operation,
                                        uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 2, 2);
    std::shared_ptr<TopkOperation> op = std::make_shared<TopkOperation>();
    NNAPI_CHECK_PTR(op);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    auto argList = matchArgList(inputs, "TopkV2Operation");
    if (argList) {
        op->k = inputs[argList->ArgPos("k")]->scalar.int32;
    } else {
        NNRT_LOGE_PRINT("Topk argument list not support");
        assert(false);
    }
    truncateOperationIOs(model, operation, 1, 2);
    return op;
}

OperationPtr NnApiInterpreter::map_ARGMAX(Model* model,
                                          OperationPtr operation,
                                          uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 2, 1);
    std::shared_ptr<ArgmaxOperation> op = std::make_shared<ArgmaxOperation>();
    NNAPI_CHECK_PTR(op);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    op->axis = inputs[1]->scalar.int32;
    truncateOperationIOs(model, operation, 1, 1);
    return op;
}

OperationPtr NnApiInterpreter::map_ARGMIN(Model* model,
                                          OperationPtr operation,
                                          uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 2, 1);
    std::shared_ptr<ArgminOperation> op = std::make_shared<ArgminOperation>();
    NNAPI_CHECK_PTR(op);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    op->axis = inputs[1]->scalar.int32;
    truncateOperationIOs(model, operation, 1, 1);
    return op;
}

OperationPtr NnApiInterpreter::map_GATHER(Model* model,
                                          OperationPtr operation,
                                          uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 3, 1);
    std::shared_ptr<GatherOperation> op = std::make_shared<GatherOperation>();
    NNAPI_CHECK_PTR(op);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    op->axis = inputs[1]->scalar.int32;
    uint32_t axis_index = operation->inputs()[1];
    operation->inputs()[1] = operation->inputs()[2];
    operation->inputs()[2] = axis_index;
    truncateOperationIOs(model, operation, 2, 1);
    return op;
}

OperationPtr NnApiInterpreter::map_CHANNEL_SHUFFLE(Model* model,
                                                   OperationPtr operation,
                                                   uint32_t operation_index) {
    std::shared_ptr<ChannelShuffleOperation> channel_shuffle =
        std::make_shared<ChannelShuffleOperation>();
    NNAPI_CHECK_PTR(channel_shuffle);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());

    std::vector<OperandType> argTypes;
    std::transform(
        inputs.begin(), inputs.end(), std::back_inserter(argTypes), [](const OperandPtr& operand) {
            return operand->type;
        });

    auto argList = api::requirement::nnapi::match("ChannelShuffleOperation", argTypes);
    if (argList) {
        auto inputOperand = inputs[argList->ArgPos("input")];
        auto outputOperand = model->getOperands(operation->outputs())[0];
        // No dynamic shape branch
        if (!nnrt::operand_utils::IsDynamicShape(inputOperand) &&
            !nnrt::operand_utils::IsDynamicShape(outputOperand)) {
            channel_shuffle->groups = inputs[argList->ArgPos("groups")]->scalar.int32;
            channel_shuffle->axis = inputs[argList->ArgPos("axis")]->scalar.int32;
        } else {
            // TODO: support dynamic input tensor shape
            NNRT_LOGE_PRINT("Dynamic shape not support");
            assert(false);
        }

    } else {
        NNRT_LOGE_PRINT("Channel shuffle argument list not support");
    }
    truncateOperationIOs(model, operation, 1, 1);
    return channel_shuffle;
}

OperationPtr NnApiInterpreter::map_SPLIT(Model* model,
                                         OperationPtr operation,
                                         uint32_t operation_index) {
    std::shared_ptr<SplitOperation> split = std::make_shared<SplitOperation>();
    NNAPI_CHECK_PTR(split);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());

    std::vector<OperandType> argTypes;
    std::transform(
        inputs.begin(), inputs.end(), std::back_inserter(argTypes), [](const OperandPtr& operand) {
            return operand->type;
        });

    auto argList = api::requirement::nnapi::match("SplitOperation", argTypes);
    if (argList) {
        auto inputOperand = inputs[argList->ArgPos("input")];
        auto outputOperand = model->getOperands(operation->outputs())[0];
        // No dynamic shape branch
        if (!nnrt::operand_utils::IsDynamicShape(inputOperand) &&
            !nnrt::operand_utils::IsDynamicShape(outputOperand)) {
            split->axis = inputs[argList->ArgPos("axis")]->scalar.int32;
            if (split->axis < 0) {
                split->axis += inputOperand->dimensions.size();
            }
            split->split_number = inputs[argList->ArgPos("split_number")]->scalar.int32;
            split->slices.assign(split->split_number,
                                 inputOperand->dimensions[split->axis] / split->split_number);

        } else {
            // TODO: support dynamic input tensor shape
            NNRT_LOGE_PRINT("Dynamic shape not support");
            assert(false);
        }

    } else {
        NNRT_LOGE_PRINT("Split argument list not support");
    }
    truncateOperationIOs(model, operation, 1, operation->outputs().size());
    return split;
}

OperationPtr NnApiInterpreter::map_INSTANCE_NORMALIZATION(Model* model,
                                                          OperationPtr operation,
                                                          uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 5, 1);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    std::vector<OperandType> argTypes;
    std::transform(
        inputs.begin(), inputs.end(), std::back_inserter(argTypes), [](const OperandPtr& operand) {
            return operand->type;
        });
    auto argList = api::requirement::nnapi::match("InstanceNormOperation", argTypes);
    OperationPtr op;
    if (argList) {
        auto inputOperand = inputs[argList->ArgPos("input")];
        auto outputOperand = model->getOperands(operation->outputs())[0];
        // No dynamic shape branch
        if (!nnrt::operand_utils::IsDynamicShape(inputOperand) &&
            !nnrt::operand_utils::IsDynamicShape(outputOperand)) {
            switch (inputs[argList->ArgPos("gamma")]->type) {
                case OperandType::FLOAT32: {
                    auto instanceNorm = std::make_shared<InstanceNormOperation<float>>();
                    instanceNorm->setDataLayout(
                        getDataLayout(inputs[argList->ArgPos("data_layout")]->scalar.boolean));
                    float gamma = inputs[argList->ArgPos("gamma")]->scalar.float32;
                    float beta = inputs[argList->ArgPos("beta")]->scalar.float32;
                    instanceNorm->eps = inputs[argList->ArgPos("epsilon")]->scalar.float32;

                    // Get input tensor channel num (Nnapi default data layout: NHWC)
                    uint32_t channelNum;
                    if (DataLayout::NHWC == instanceNorm->getDataLayout())
                        channelNum = inputs[0]->dimensions[3];
                    else
                        channelNum = inputs[0]->dimensions[1];

                    // Broadcast
                    for (uint32_t i = 0; i < channelNum; i++) {
                        instanceNorm->gamma.push_back(gamma);
                        instanceNorm->beta.push_back(beta);
                    }

                    // Convert scaler gamma to constant tensor
                    uint32_t gammaIds = 0;
                    OperandPtr gammaOperand = model->addOperand(nullptr, &gammaIds);
                    gammaOperand->type = OperandType::TENSOR_FLOAT32;
                    gammaOperand->dimensions = {channelNum};
                    model->setOperandValue(
                        gammaIds,
                        instanceNorm->gamma.data(),
                        instanceNorm->gamma.size() * sizeof(decltype(instanceNorm->gamma[0])));

                    // Convert scaler beta to constant tensor
                    uint32_t betaIds = 0;
                    OperandPtr betaOperand = model->addOperand(nullptr, &betaIds);
                    betaOperand->type = OperandType::TENSOR_FLOAT32;
                    betaOperand->dimensions = {channelNum};
                    model->setOperandValue(
                        betaIds,
                        instanceNorm->beta.data(),
                        instanceNorm->beta.size() * sizeof(decltype(instanceNorm->beta[0])));

                    // Add gamma and beta operand index into instance norm operation
                    auto inputsIds = operation->inputs();
                    auto it = inputsIds.begin();
                    // Note: the order is important
                    // {bias, scalar}
                    std::vector<uint32_t> insertIds = {betaIds, gammaIds};
                    inputsIds.insert(it + 1, insertIds.begin(), insertIds.end());
                    operation->setInputs(inputsIds);
                    op = instanceNorm;
                    break;
                }
                case OperandType::FLOAT16: {
                    auto instanceNorm = std::make_shared<InstanceNormOperation<half_float::half>>();
                    instanceNorm->setDataLayout(
                        getDataLayout(inputs[argList->ArgPos("data_layout")]->scalar.boolean));
                    half_float::half gamma;
                    half_float::half beta;
                    half_float::half eps;
                    memcpy(&gamma,
                           &inputs[argList->ArgPos("gamma")]->scalar.float16,
                           sizeof(half_float::half));
                    memcpy(&beta,
                           &inputs[argList->ArgPos("beta")]->scalar.float16,
                           sizeof(half_float::half));
                    memcpy(&eps,
                           &inputs[argList->ArgPos("epsilon")]->scalar.float16,
                           sizeof(half_float::half));
                    instanceNorm->eps = eps;

                    uint32_t channelNum;
                    if (DataLayout::NHWC == instanceNorm->getDataLayout())
                        channelNum = inputs[0]->dimensions[3];
                    else
                        channelNum = inputs[0]->dimensions[1];
                    // Broadcast
                    for (uint32_t i = 0; i < channelNum; i++) {
                        instanceNorm->gamma.push_back(gamma);
                        instanceNorm->beta.push_back(beta);
                    }

                    // Convert scaler gamma to constant tensor
                    uint32_t gammaIds = 0;
                    OperandPtr gammaOperand = model->addOperand(nullptr, &gammaIds);
                    gammaOperand->type = OperandType::TENSOR_FLOAT16;
                    gammaOperand->dimensions = {channelNum};
                    model->setOperandValue(
                        gammaIds,
                        instanceNorm->gamma.data(),
                        instanceNorm->gamma.size() * sizeof(decltype(instanceNorm->gamma[0])));

                    // Convert scaler beta to constant tensor
                    uint32_t betaIds = 0;
                    OperandPtr betaOperand = model->addOperand(nullptr, &betaIds);
                    betaOperand->type = OperandType::TENSOR_FLOAT16;
                    betaOperand->dimensions = {channelNum};
                    model->setOperandValue(
                        betaIds,
                        instanceNorm->beta.data(),
                        instanceNorm->beta.size() * sizeof(decltype(instanceNorm->beta[0])));

                    // Add gamma and beta operand index into instance norm operation
                    auto inputsIds = operation->inputs();
                    auto it = inputsIds.begin();
                    // Note: the order is important
                    // {bias, scalar}
                    std::vector<uint32_t> insertIds = {betaIds, gammaIds};
                    inputsIds.insert(it + 1, insertIds.begin(), insertIds.end());
                    operation->setInputs(inputsIds);
                    op = instanceNorm;
                    break;
                }
                default: {
                    NNRT_LOGE_PRINT("InstanceNorm doesn't support given datatype");
                    assert(false);
                }
            }
        } else {
            // TODO: support dynamic input tensor shape
            NNRT_LOGE_PRINT("Dynamic shape not support");
            assert(false);
        }
    } else {
        NNRT_LOGE_PRINT("Instance normalization argument list not support");
    }
    truncateOperationIOs(model, operation, 3, 1);
    return op;
}

OperationPtr NnApiInterpreter::map_GENERATE_PROPOSALS(Model* model,
                                                      OperationPtr operation,
                                                      uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 11, 3);
    std::shared_ptr<GenerateProposalsOperation> op = std::make_shared<GenerateProposalsOperation>();
    NNAPI_CHECK_PTR(op);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    auto argList = matchArgList(inputs, "GenerateProposalsOperation");
    if (argList) {
        switch (inputs[argList->ArgPos("ratio_h")]->type) {
            case OperandType::FLOAT32: {
                op->ratio_h = inputs[argList->ArgPos("ratio_h")]->scalar.float32;
                op->ratio_w = inputs[argList->ArgPos("ratio_w")]->scalar.float32;
                op->pre_nms_topn = inputs[argList->ArgPos("pre_nms_topn")]->scalar.int32;
                op->post_nms_topn = inputs[argList->ArgPos("post_nms_topn")]->scalar.int32;
                op->iou_threshold = inputs[argList->ArgPos("iou_threshold")]->scalar.float32;
                op->min_size = inputs[argList->ArgPos("min_size")]->scalar.float32;
                op->setDataLayout(getDataLayout(inputs[argList->ArgPos("layout")]->scalar.boolean));
                break;
            }
            case OperandType::FLOAT16: {
                half_float::half ratioH;
                memcpy(&ratioH,
                       &inputs[argList->ArgPos("ratio_h")]->scalar.float16,
                       sizeof(half_float::half));
                op->ratio_h = ratioH;

                half_float::half ratioW;
                memcpy(&ratioW,
                       &inputs[argList->ArgPos("ratio_w")]->scalar.float16,
                       sizeof(half_float::half));
                op->ratio_w = ratioW;

                half_float::half iouThreshold;
                memcpy(&iouThreshold,
                       &inputs[argList->ArgPos("iou_threshold")]->scalar.float16,
                       sizeof(half_float::half));
                op->iou_threshold = iouThreshold;

                half_float::half minSize;
                memcpy(&minSize,
                       &inputs[argList->ArgPos("min_size")]->scalar.float16,
                       sizeof(half_float::half));
                op->min_size = minSize;
                break;
            }
            default:
                NNRT_LOGE_PRINT("GenerateProposals does't support given datatype");
                assert(false);
        }
    } else {
        NNRT_LOGE_PRINT("GenerateProposals argument list not support");
        assert(false);
    }
    truncateOperationIOs(model, operation, 4, 3);
    return op;
}

OperationPtr NnApiInterpreter::map_DETECTION_POSTPROCESSING(Model* model,
                                                            OperationPtr operation,
                                                            uint32_t operation_index) {
    std::shared_ptr<DetectionPostprocessingOperation> op =
        std::make_shared<DetectionPostprocessingOperation>();
    NNAPI_CHECK_PTR(op);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    auto argList = matchArgList(inputs, "DetectionPostprocessingOperation");
    if (argList) {
        op->dy = inputs[argList->ArgPos("dy")]->scalar.float32;
        op->dx = inputs[argList->ArgPos("dx")]->scalar.float32;
        op->dh = inputs[argList->ArgPos("dh")]->scalar.float32;
        op->dw = inputs[argList->ArgPos("dw")]->scalar.float32;
        op->nms_type = inputs[argList->ArgPos("nms_type")]->scalar.boolean;
        op->max_num_detections = inputs[argList->ArgPos("max_num_detections")]->scalar.int32;
        op->maximum_class_per_detection =
            inputs[argList->ArgPos("maximum_class_per_detection")]->scalar.int32;
        op->maximum_detection_per_class =
            inputs[argList->ArgPos("maximum_detection_per_class")]->scalar.int32;
        op->score_threshold = inputs[argList->ArgPos("score_threshold")]->scalar.float32;
        op->iou_threshold = inputs[argList->ArgPos("iou_threshold")]->scalar.float32;
        op->is_bg_in_label = inputs[argList->ArgPos("is_bg_in_label")]->scalar.boolean;
    } else {
        NNRT_LOGE_PRINT("DetectionPostprocessing argument list not support");
        assert(false);
    }
    truncateOperationIOs(model, operation, 3, 4);
    return op;
}

OperationPtr NnApiInterpreter::map_RANDOM_MULTINOMIAL(Model* model,
                                                      OperationPtr operation,
                                                      uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 3, 1);
    std::shared_ptr<RandomMultinomialOperation> op = std::make_shared<RandomMultinomialOperation>();
    NNAPI_CHECK_PTR(op);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    auto argList = matchArgList(inputs, "RandomMultinomialOperation");
    if (argList) {
        op->sample_num = inputs[argList->ArgPos("sample_num")]->scalar.int32;
        int32_t input_index = operation->input(argList->ArgPos("input"));
        int32_t seed_index = operation->input(argList->ArgPos("seeds"));
        operation->inputs().clear();
        operation->inputs().emplace_back(input_index);
        operation->inputs().emplace_back(seed_index);
    } else {
        NNRT_LOGE_PRINT("RandomMultinomial argument list not support");
        assert(false);
    }
    return op;
}

OperationPtr NnApiInterpreter::map_TILE(Model* model,
                                        OperationPtr operation,
                                        uint32_t operation_index) {
    std::shared_ptr<TileOperation> op = std::make_shared<TileOperation>();
    NNAPI_CHECK_PTR(op);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    auto argList = matchArgList(inputs, "TileOperation");
    if (argList) {
        auto multiples_tensor = inputs[argList->ArgPos("multiples")];
        int32_t* multiples = model->getBuffer<int32_t>(multiples_tensor->weak_mem_ref.lock());
        op->multiples.assign(multiples, multiples + multiples_tensor->size());
    } else {
        NNRT_LOGE_PRINT("Tile argument list not support");
        assert(false);
    }
    truncateOperationIOs(model, operation, 1, 1);
    return op;
}

OperationPtr NnApiInterpreter::map_ROI_POOLING(Model* model,
                                               OperationPtr operation,
                                               uint32_t operation_index) {
    std::shared_ptr<ROIPoolingOperation> op = std::make_shared<ROIPoolingOperation>();
    NNAPI_CHECK_PTR(op);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    auto argList = matchArgList(inputs, "ROIPoolingOperation");
    if (argList) {
        switch (inputs[argList->ArgPos("height_ratio")]->type) {
            case OperandType::FLOAT16: {
                half_float::half heightRatio;
                memcpy(&heightRatio,
                       &inputs[argList->ArgPos("height_ratio")]->scalar.float16,
                       sizeof(half_float::half));
                op->height_ratio = heightRatio;

                half_float::half widthRatio;
                memcpy(&widthRatio,
                       &inputs[argList->ArgPos("width_ratio")]->scalar.float16,
                       sizeof(half_float::half));
                op->width_ratio = widthRatio;
                break;
            }
            case OperandType::FLOAT32: {
                op->height_ratio = inputs[argList->ArgPos("height_ratio")]->scalar.float32;
                op->width_ratio = inputs[argList->ArgPos("width_ratio")]->scalar.float32;
                break;
            }
            default: {
                NNRT_LOGE_PRINT("ROIPooling does't support given data type.");
                assert(false);
            }
        }
        op->height = inputs[argList->ArgPos("height")]->scalar.int32;
        op->width = inputs[argList->ArgPos("width")]->scalar.int32;
        op->setDataLayout(getDataLayout(inputs[argList->ArgPos("layout")]->scalar.boolean));
    } else {
        NNRT_LOGE_PRINT("ROIPooling argument list not support");
        assert(false);
    }
    truncateOperationIOs(model, operation, 3, 1);
    return op;
}

OperationPtr NnApiInterpreter::map_ROI_ALIGN(Model* model,
                                             OperationPtr operation,
                                             uint32_t operation_index) {
    std::shared_ptr<ROIAlignOperation> op = std::make_shared<ROIAlignOperation>();
    NNAPI_CHECK_PTR(op);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    auto argList = matchArgList(inputs, "ROIAlignOperation");
    if (argList) {
        switch (inputs[argList->ArgPos("height_ratio")]->type) {
            case OperandType::FLOAT16: {
                half_float::half heightRatio;
                memcpy(&heightRatio,
                       &inputs[argList->ArgPos("height_ratio")]->scalar.float16,
                       sizeof(half_float::half));
                op->height_ratio = heightRatio;

                half_float::half widthRatio;
                memcpy(&widthRatio,
                       &inputs[argList->ArgPos("width_ratio")]->scalar.float16,
                       sizeof(half_float::half));
                op->width_ratio = widthRatio;
                break;
            }
            case OperandType::FLOAT32: {
                op->height_ratio = inputs[argList->ArgPos("height_ratio")]->scalar.float32;
                op->width_ratio = inputs[argList->ArgPos("width_ratio")]->scalar.float32;
                break;
            }
            default: {
                NNRT_LOGE_PRINT("RoiAlign does't support the given data type.");
                assert(false);
            }
        }
        op->height = inputs[argList->ArgPos("height")]->scalar.int32;
        op->width = inputs[argList->ArgPos("width")]->scalar.int32;
        op->sampling_points_height =
            inputs[argList->ArgPos("sampling_points_height")]->scalar.int32;
        op->sampling_points_width = inputs[argList->ArgPos("sampling_points_width")]->scalar.int32;
        op->setDataLayout(getDataLayout(inputs[argList->ArgPos("layout")]->scalar.boolean));
    } else {
        NNRT_LOGE_PRINT("ROIAlign argument list not support");
        assert(false);
    }
    truncateOperationIOs(model, operation, 3, 1);
    return op;
}

OperationPtr NnApiInterpreter::map_HEATMAP_MAX_KEYPOINT(Model* model,
                                                        OperationPtr operation,
                                                        uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 3, 2);
    std::shared_ptr<HeatmapMaxKeypointOperation> op =
        std::make_shared<HeatmapMaxKeypointOperation>();
    NNAPI_CHECK_PTR(op);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    auto argList = matchArgList(inputs, "HeatmapMaxKeypointOperation");
    if (argList) {
        op->setDataLayout(getDataLayout(inputs[argList->ArgPos("layout")]->scalar.boolean));
    } else {
        NNRT_LOGE_PRINT("HeatmapMaxPoint argument list not support");
        assert(false);
    }
    truncateOperationIOs(model, operation, 2, 2);
    return op;
}

OperationPtr NnApiInterpreter::map_BOX_WITH_NMS_LIMIT(Model* model,
                                                      OperationPtr operation,
                                                      uint32_t operation_index) {
    std::shared_ptr<BoxWithNmsLimitOperation> op = std::make_shared<BoxWithNmsLimitOperation>();
    NNAPI_CHECK_PTR(op);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    auto argList = matchArgList(inputs, "BoxWithNmsLimitOperation");
    auto choose_nms_kernel_method = [](int32_t value) -> NmsKernelMethod {
        switch (value) {
            case 0:
                return NmsKernelMethod::Hard;
            case 1:
                return NmsKernelMethod::Linear;
            case 2:
                return NmsKernelMethod::Gaussian;
            default:
                NNRT_LOGE_PRINT("Unsupport nms kernel method %d", value);
                break;
        }
        return NmsKernelMethod::Hard;
    };
    if (argList) {
        op->score_threshold = inputs[argList->ArgPos("score_threshold")]->scalar.float32;
        op->max_boxes = inputs[argList->ArgPos("max_boxes")]->scalar.int32;
        op->nms_kernel_method =
            choose_nms_kernel_method(inputs[argList->ArgPos("nms_kernel_method")]->scalar.int32);
        op->iou_threshold = inputs[argList->ArgPos("iou_threshold")]->scalar.float32;
        op->nms_sigma = inputs[argList->ArgPos("nms_sigma")]->scalar.float32;
        op->nms_score_threshold = inputs[argList->ArgPos("nms_score_threshold")]->scalar.float32;
    } else {
        NNRT_LOGE_PRINT("BoxWithNmsLimit argument list not support");
        assert(false);
    }
    truncateOperationIOs(model, operation, 3, 4);
    return op;
}

OperationPtr NnApiInterpreter::map_LOG_SOFTMAX(Model* model,
                                               OperationPtr operation,
                                               uint32_t operation_index) {
    NNAPI_CHECK_IO_NUM(operation, 3, 1);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    auto argList = matchArgList(inputs, "LogSoftmaxOperation");
    auto op_ptr = std::make_shared<LogSoftmaxOperation>();
    if (argList) {
        switch (inputs[argList->ArgPos("beta")]->type) {
            case OperandType::FLOAT16: {
                half_float::half beta;
                memcpy(&beta,
                       &inputs[argList->ArgPos("beta")]->scalar.float16,
                       sizeof(half_float::half));
                op_ptr->beta = beta;
                op_ptr->axis = inputs[argList->ArgPos("axis")]->scalar.int32;
                break;
            }
            case OperandType::FLOAT32: {
                op_ptr->beta = inputs[argList->ArgPos("beta")]->scalar.float32;
                op_ptr->axis = inputs[argList->ArgPos("axis")]->scalar.int32;
                break;
            }
            default:
                assert(false);
                NNRT_LOGE_PRINT("LogSoftmax doesn't support given datatype");
        }
    } else {
        NNRT_LOGE_PRINT("LogSoftmax argument list not support");
        assert(false);
    }
    truncateOperationIOs(model, operation, 1, 1);
    return op_ptr;
}

#define DECLARE_SAMPLE_OP(NAME, INPUT_NUM, OUTPUT_NUM, OPERATION_TYPE)    \
    OperationPtr NnApiInterpreter::map_##NAME(                            \
        Model* model, OperationPtr operation, uint32_t operation_index) { \
        NNAPI_CHECK_IO_NUM(operation, INPUT_NUM, OUTPUT_NUM);             \
        OperationPtr op = std::make_shared<OPERATION_TYPE>();             \
        if (op)                                                           \
            op->setDataLayout(DataLayout::NHWC);                          \
        else                                                              \
            NNRT_LOGE_PRINT("Invalid operaton pointer");                  \
        return op;                                                        \
    }

DECLARE_SAMPLE_OP(RELU1, 1, 1, Relu1Operation)
DECLARE_SAMPLE_OP(RELU6, 1, 1, Relu6Operation)
DECLARE_SAMPLE_OP(TANH, 1, 1, TanhOperation)
DECLARE_SAMPLE_OP(SIGMOID, 1, 1, SigmoidOperation)
DECLARE_SAMPLE_OP(FLOOR, 1, 1, FloorOperation)
DECLARE_SAMPLE_OP(ABS, 1, 1, AbsOperation)
DECLARE_SAMPLE_OP(POW, 2, 1, PowOperation)
DECLARE_SAMPLE_OP(SQRT, 1, 1, SqrtOperation)
DECLARE_SAMPLE_OP(RSQRT, 1, 1, RSqrtOperation)
DECLARE_SAMPLE_OP(NEG, 1, 1, NegOperation)
DECLARE_SAMPLE_OP(EXP, 1, 1, ExpOperation)
DECLARE_SAMPLE_OP(MAXIMUM, 2, 1, MaximumOperation)
DECLARE_SAMPLE_OP(MINIMUM, 2, 1, MinimumOperation)
DECLARE_SAMPLE_OP(QUANTIZE, 1, 1, QuantizeOperation)
DECLARE_SAMPLE_OP(DEQUANTIZE, 1, 1, DequantizeOperation)
DECLARE_SAMPLE_OP(EMBEDDING_LOOKUP, 2, 1, EmbeddingLookupOperation)
DECLARE_SAMPLE_OP(HASHTABLE_LOOKUP, 3, 2, HashtableLookupOperation)
DECLARE_SAMPLE_OP(EQUAL, 2, 1, EqualOperation)
DECLARE_SAMPLE_OP(NOT_EQUAL, 2, 1, NotEqualOperation)
DECLARE_SAMPLE_OP(LESS, 2, 1, LessOperation)
DECLARE_SAMPLE_OP(LESS_EQUAL, 2, 1, LessEqualOperation)
DECLARE_SAMPLE_OP(GREATER, 2, 1, GreaterOperation)
DECLARE_SAMPLE_OP(GREATER_EQUAL, 2, 1, GreaterEqualOperation)
DECLARE_SAMPLE_OP(LOG, 1, 1, LogOperation)
DECLARE_SAMPLE_OP(LOGICAL_AND, 2, 1, LogicalAndOperation)
DECLARE_SAMPLE_OP(LOGICAL_OR, 2, 1, LogicalOrOperation)
DECLARE_SAMPLE_OP(LOGICAL_NOT, 1, 1, LogicalNotOperation)
DECLARE_SAMPLE_OP(PRELU, 2, 1, PReluOperation)
DECLARE_SAMPLE_OP(SELECT, 3, 1, SelectOperation)
DECLARE_SAMPLE_OP(SIN, 1, 1, SinOperation)
DECLARE_SAMPLE_OP(AXIS_ALIGNED_BBOX_TRANSFORM, 4, 1, AxisAlignedBBoxTransformOperation)
DECLARE_SAMPLE_OP(DATA_CONVERT, 1, 1, DataConvertOperation)
DECLARE_SAMPLE_OP(CAST, 1, 1, CastOperation)

#undef DECLARE_SAMPLE_OP

#define DECLARE_REDUCTION_OP(NAME, OPERATION_TYPE)                                              \
    OperationPtr NnApiInterpreter::map_##NAME(                                                  \
        Model* model, OperationPtr operation, uint32_t operation_index) {                       \
        auto op = std::make_shared<OPERATION_TYPE##Operation>();                                \
        NNAPI_CHECK_PTR(op);                                                                    \
        std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());               \
        auto argList = matchArgList(inputs, "" #OPERATION_TYPE);                                \
        if (argList) {                                                                          \
            if (inputs[argList->ArgPos("axes")]->isConst()) {                                   \
                op->axes.clear();                                                               \
                int32_t* buffer = model->getBuffer<int32_t>(inputs[1]->weak_mem_ref.lock());    \
                std::set<int32_t> axes;                                                         \
                for (size_t i = 0; i < inputs[1]->size(); ++i) {                                \
                    if (buffer[i] < 0) {                                                        \
                        axes.insert(buffer[i] + inputs[0]->ndim());                             \
                    } else {                                                                    \
                        axes.insert(buffer[i]);                                                 \
                    }                                                                           \
                }                                                                               \
                std::for_each(axes.begin(), axes.end(), [&op](const int32_t& axis) {            \
                    op->axes.push_back(axis);                                                   \
                });                                                                             \
            }                                                                                   \
            op->keepDim = static_cast<bool>(inputs[argList->ArgPos("keep_dim")]->scalar.int32); \
            truncateOperationIOs(model, operation, 1, 1);                                       \
        } else {                                                                                \
            NNRT_LOGE_PRINT("Number of input parameter is not valid");                          \
        }                                                                                       \
        return op;                                                                              \
    }

DECLARE_REDUCTION_OP(REDUCE_ALL, ReduceAll)
DECLARE_REDUCTION_OP(REDUCE_ANY, ReduceAny)
DECLARE_REDUCTION_OP(REDUCE_MAX, ReduceMax)
DECLARE_REDUCTION_OP(REDUCE_MIN, ReduceMin)
DECLARE_REDUCTION_OP(REDUCE_SUM, ReduceSum)
DECLARE_REDUCTION_OP(REDUCE_PROD, ReduceProd)
#undef DECLARE_REDUCTION_OP
}