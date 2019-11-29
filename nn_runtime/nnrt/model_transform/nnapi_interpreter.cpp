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

#include "logging.hpp"
#include "types.hpp"
#include "model.hpp"
#include "error.hpp"
#include "utils.hpp"

#include "model_transform/nnapi_interpreter.hpp"
#include "api_requirement/nnapi_requirement.hpp"

// TODO: Xiang remove duplicate code
/**
 * Fused activation function types.
 *
 *
 * Available since API level 27.
 */
typedef enum {
    /** NO fused activation function. */
    ANEURALNETWORKS_FUSED_NONE = 0,
    /** Fused ReLU activation function. */
    ANEURALNETWORKS_FUSED_RELU = 1,
    /** Fused ReLU1 activation function. */
    ANEURALNETWORKS_FUSED_RELU1 = 2,
    /** Fused ReLU6 activation function. */
    ANEURALNETWORKS_FUSED_RELU6 = 3,
} FuseCode;


/**
 * Implicit padding algorithms.
 *
 *
 * Available since API level 27.
 */
typedef enum {
    /**
     * SAME padding.
     * Padding on both ends are the "same":
     *     padding_to_beginning =  total_padding / 2
     *     padding_to_end       = (total_padding + 1)/2.
     * i.e., for even number of padding, padding to both ends are exactly
     * the same; for odd number of padding, padding to the ending is bigger
     * than the padding to the beginning by 1.
     *
     * total_padding is a function of input, stride and filter size.
     * It could be computed as follows:
     *    out_size = (input + stride - 1) / stride;
     *    needed_input = (out_size - 1) * stride + filter_size
     *    total_padding = max(0, needed_input - input_size)
     *  The computation is the same for the horizontal and vertical directions.
     */
    ANEURALNETWORKS_PADDING_SAME = 1,

    /**
     * VALID padding.
     * No padding. When the input size is not evenly divisible by
     * the filter size, the input at the end that could not fill
     * the whole filter tile will simply be ignored.
     */
    ANEURALNETWORKS_PADDING_VALID = 2,
} PaddingCode;


namespace nnrt
{

#define NNAPI_CHECK_IO_NUM(op, in_num, out_num)         \
    do {                                                \
        if ((in_num > 0 && op->inputs().size() != (size_t)in_num)       \
         || (out_num > 0 && op->outputs().size() != (size_t)out_num)) {           \
            NNRT_LOGW_PRINT("Operation IO number mismatch. %d(%d), %d(%d)",     \
                    op->inputs().size(), in_num,        \
                    op->outputs().size(), out_num);     \
            return nullptr;                             \
        }                                               \
    } while(0)

#define NNAPI_CHECK_PTR(ptr)                            \
    do {                                                \
        if (!ptr) {                                     \
            return nullptr;                             \
        }                                               \
    } while(0)

static void convert2DPadding(int32_t* padding,
        size_t size, int32_t* front, int32_t* back)
{
    if (!padding || !front || !back) {
        return;
    }
    for (size_t i = 0; i < size; i += 2) {
        front[i / 2] = padding[i];
        back[i / 2] = padding[i + 1];
    }
}

NnApiInterpreter::NnApiInterpreter()
{
#define REGISTER_OP(NAME)   do {                            \
    op_container_[OperationType::NAME] = &NnApiInterpreter::map_##NAME;  \
    } while(0)

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
    //REGISTER_OP(EXPAND_DIMS);
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
    //REGISTER_OP(SLICE);
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

    /*customer Op*/
#undef REGISTER_OP

}

NnApiInterpreter::~NnApiInterpreter()
{

}

int NnApiInterpreter::run(Model* model, bool* modified)
{
    *modified = false;
    const std::map<uint32_t, OperationPtr>& operations = model->operations();
    for (auto it = operations.begin(); it != operations.end(); ++ it)
    {
        OperationPtr op = it->second;
        if (op_container_.find(op->type()) == op_container_.end())
        {
            NNRT_LOGW_PRINT("Not support operation %d", op->type());
            return NNA_ERROR_CODE(BAD_DATA);
        }
    }

    for (auto it = operations.begin(); it != operations.end(); ++ it)
    {
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
        //NNRT_LOGD_PRINT("Remove %d", index);
        if (model->isInput(index) || model->isOutput(index)) {
            NNRT_LOGW_PRINT("Try remove operand(%u) from model input or output, \
some operations may not support dynamic configure.", index);
        } else {
            model->removeOperand(index);
        }
    }

    return NNA_ERROR_CODE(NO_ERROR);
}

void NnApiInterpreter::replaceOperation(Model* model, uint32_t op_index,
        OperationPtr new_operation)
{
    OperationPtr org_operation = model->operation(op_index);
    new_operation->setInputs(org_operation->inputs());
    new_operation->setOutputs(org_operation->outputs());
    new_operation->setFusedType(org_operation->fusedType());
    model->operations()[op_index] = new_operation;
}

FusedType NnApiInterpreter::mapFusedType(int fused_code)
{
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

PadType NnApiInterpreter::mapPadType(int code)
{
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

LshProjectionType NnApiInterpreter::mapLshProjectionType(int value)
{
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

FusedType NnApiInterpreter::mapLstmActivationType(int value)
{
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

std::vector<int32_t> NnApiInterpreter::convertAxes(
        int32_t* axes_buffer, size_t length, size_t dim_num) {
    std::vector<int32_t> axes;
    axes.insert(axes.begin(), axes_buffer, axes_buffer + length);
    return convertAxes(axes, dim_num);
}

std::vector<int32_t> NnApiInterpreter::convertAxes(
        std::vector<int32_t> & axes, size_t dim_num) {
    std::vector<int32_t> new_axes(axes.size());
    size_t max_size = axes.size() - 1;
    for (size_t i = 0; i < axes.size(); i ++)
    {
        new_axes[i] = convertAxis(axes[max_size - i], dim_num);
    }
    return new_axes;
}

void NnApiInterpreter::fillIntArray(Model* model, OperationPtr operation,
        std::vector<int32_t>& array, int32_t op_index, bool is_axis)
{
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

void NnApiInterpreter::truncateOperationIOs(Model* model, OperationPtr operation,
        int32_t input_num, int32_t output_num) {
    // Size - 1 = axis
    input_num = computeAxis(input_num, operation->inputs().size() + 1);
    output_num = computeAxis(output_num, operation->outputs().size() + 1);
    for (int i = input_num; i < (int)operation->inputs().size(); ++ i) {
        operands_to_remove_.emplace(operation->input(i));
    }
    for (int i = output_num; i < (int)operation->outputs().size(); ++ i) {
        operands_to_remove_.emplace(operation->output(i));
    }
    operation->inputs().resize(input_num);
    operation->outputs().resize(output_num);
}



OperationPtr NnApiInterpreter::map_ADD(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 3, 1);
    resetFusedType(model, operation, 2);
    truncateOperationIOs(model, operation, 2, 1);
    return std::make_shared<AddOperation>();
}

OperationPtr NnApiInterpreter::map_SUB(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 3, 1);
    resetFusedType(model, operation, 2);
    truncateOperationIOs(model, operation, 2, 1);
    return std::make_shared<SubOperation>();
}

OperationPtr NnApiInterpreter::map_DIV(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
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
        OperationPtr operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, -1, 1);
    std::shared_ptr<ConcatOperation> concat = std::make_shared<ConcatOperation>();
    NNAPI_CHECK_PTR(concat);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    concat->axis = inputs.back()->scalar.int32;
    truncateOperationIOs(model, operation, -2, 1);
    return concat;
}

OperationPtr NnApiInterpreter::map_CONV_2D(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    std::shared_ptr<Conv2DOperation> conv2d = std::make_shared<Conv2DOperation>();

    std::vector<OperandType> argTypes;
    std::transform(inputs.begin(), inputs.end(), std::back_inserter(argTypes), [](const OperandPtr& operand){
        return operand->type;
    });

    auto argList = api::requirement::nnapi::match("Convolution2D", argTypes);
    if (argList) {
        if (-1 == argList->ArgPos("explicit_pad_left") ) {
            // implicit_pad
            auto padTypeIdx = argList->ArgPos("implicit_pad_type");
            conv2d->padType = mapPadType(inputs[padTypeIdx]->scalar.int32);
        } else {
            conv2d->pad[0] = inputs[argList->ArgPos("explicit_pad_left")    ]->scalar.int32;
            conv2d->pad[1] = inputs[argList->ArgPos("explicit_pad_right")   ]->scalar.int32;
            conv2d->pad[2] = inputs[argList->ArgPos("explicit_pad_top")     ]->scalar.int32;
            conv2d->pad[3] = inputs[argList->ArgPos("explicit_pad_bottom")  ]->scalar.int32;
        }
        // dilation is required in Lowlevel requirement
        conv2d->dilations[0] = 1;
        conv2d->dilations[1] = 1;
        if ( -1 != argList->ArgPos("dilation_w") && -1 != argList->ArgPos("dilation_h")) {
            conv2d->dilations[0] = inputs[argList->ArgPos("dilation_w")]->scalar.int32;
            conv2d->dilations[1] = inputs[argList->ArgPos("dilation_h")]->scalar.int32;
        }
        conv2d->setDataLayout(DataLayout::NHWC);
        if ( -1 != argList->ArgPos("data_layout")) {
            conv2d->setDataLayout(getDataLayout(inputs[argList->ArgPos("data_layout")]->scalar.boolean));
        }

        resetFusedType(model, operation, argList->ArgPos("fuse_code"));
        conv2d->strides[0] = inputs[argList->ArgPos("stride_w")]->scalar.int32;
        conv2d->strides[1] = inputs[argList->ArgPos("stride_h")]->scalar.int32;
    } else {
        NNRT_LOGE_PRINT("convolution 2d argument list not support");
    }

    /* set default dilation value */
    conv2d->setVxParam(OverflowPolicy::WRAP, RoundingPolicy::TO_ZERO, Rounding::FLOOR);
    truncateOperationIOs(model, operation, 3, 1);
    return conv2d;
}

OperationPtr NnApiInterpreter::map_GROUPED_CONV_2D(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    std::shared_ptr<GroupedConv2DOperation> conv2d = std::make_shared<GroupedConv2DOperation>();
    NNAPI_CHECK_PTR(conv2d);
    conv2d->setDataLayout(DataLayout::NHWC);
    conv2d->dilations[0] = 1;
    conv2d->dilations[1] = 1;
    if (inputs.size() == 9) {
        conv2d->padType = mapPadType(inputs[3]->scalar.int32);
        conv2d->strides[0] = inputs[4]->scalar.int32;
        conv2d->strides[1] = inputs[5]->scalar.int32;
        conv2d->groups = inputs[6]->scalar.int32;
        resetFusedType(model, operation, 7);
        conv2d->setDataLayout(getDataLayout(inputs[8]->scalar.boolean));
        //conv2d->dilations[0] = inputs[8]->scalar.int32;
        //conv2d->dilations[1] = inputs[9]->scalar.int32;
    } else {
        conv2d->pad[0] = inputs[3]->scalar.int32;
        conv2d->pad[1] = inputs[4]->scalar.int32;
        conv2d->pad[2] = inputs[5]->scalar.int32;
        conv2d->pad[3] = inputs[6]->scalar.int32;
        conv2d->strides[0] = inputs[7]->scalar.int32;
        conv2d->strides[1] = inputs[8]->scalar.int32;
        conv2d->groups = inputs[9]->scalar.int32;
        resetFusedType(model, operation, 10);
        conv2d->setDataLayout(getDataLayout(inputs[11]->scalar.boolean));
        //conv2d->dilations[0] = inputs[11]->scalar.int32;
        //conv2d->dilations[1] = inputs[12]->scalar.int32;
    }
    /* set default dilation value */
    conv2d->setVxParam(OverflowPolicy::WRAP, RoundingPolicy::TO_ZERO, Rounding::FLOOR);
    truncateOperationIOs(model, operation, 3, 1);
    return conv2d;
}

OperationPtr NnApiInterpreter::map_DEPTHWISE_CONV_2D(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
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
        } else if (-1 != argList->ArgPos("implicit_pad_type")){
            conv2d->padType = mapPadType(inputs[argList->ArgPos("implicit_pad_type")]->scalar.int32);
        } else {
            assert(0);
            NNRT_LOGE("NNAPI_interpreter") << "Argument padding method not found";
        }

        conv2d->strides[0] = inputs[argList->ArgPos("stride_w")]->scalar.int32;
        conv2d->strides[1] = inputs[argList->ArgPos("stride_h")]->scalar.int32;
        conv2d->multiplier = inputs[argList->ArgPos("multiplier")]->scalar.int32;

        conv2d->setDataLayout(DataLayout::NHWC);  // default layout
        if (-1 != argList->ArgPos("data_layout")) {
            conv2d->setDataLayout(getDataLayout(inputs[argList->ArgPos("data_layout")]->scalar.boolean));
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
        conv2d->setVxParam(OverflowPolicy::WRAP, RoundingPolicy::TO_ZERO, Rounding::FLOOR);
        truncateOperationIOs(model, operation, 3, 1);
    } else {
        assert(0);
        NNRT_LOGE("NNAPI_interpreter") << "Argument match failed";
    }

    return conv2d;
}

OperationPtr NnApiInterpreter::map_RELU(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 1, 1);
    return std::make_shared<ReluOperation>();
}

OperationPtr NnApiInterpreter::map_FULLY_CONNECTED(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
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
        OperationPtr operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 2, 1);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    std::shared_ptr<ReshapeOperation> reshape = std::make_shared<ReshapeOperation>();
    NNAPI_CHECK_PTR(reshape);
    if (!inputs[1]->isConst()) {
        std::vector<OperandPtr> outputs = model->getOperands(operation->outputs());
        assert(outputs[0]->ndim() > 0);
        reshape->shape = std::vector<int32_t>(outputs[0]->dimensions.begin(),
                outputs[0]->dimensions.end());
    } else {
        fillIntArray(model, operation, reshape->shape, 1, false);
    }
    truncateOperationIOs(model, operation, 1, 1);
    return reshape;
}

OperationPtr NnApiInterpreter::map_SOFTMAX(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    std::shared_ptr<SoftmaxOperation> softmax = std::make_shared<SoftmaxOperation>();
    NNAPI_CHECK_PTR(softmax);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    softmax->beta = inputs[1]->scalar.float32;
    softmax->axis = inputs.size() == 3 ? inputs[2]->scalar.int32 : -1;
    truncateOperationIOs(model, operation, 1, 1);
    return softmax;
}

OperationPtr NnApiInterpreter::map_TRANSPOSE(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 2, 1);
    std::shared_ptr<PermuteOperation> permute = std::make_shared<PermuteOperation>();
    NNAPI_CHECK_PTR(permute);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    fillIntArray(model, operation, permute->perm, 1, false);
    truncateOperationIOs(model, operation, 1, 1);
    return permute;
}

OperationPtr NnApiInterpreter::map_AVERAGE_POOL_2D(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    std::shared_ptr<AveragePool2DOperation> pool = std::make_shared<AveragePool2DOperation>();
    NNAPI_CHECK_PTR(pool);

    pool->setDataLayout(DataLayout::NHWC);

    if (inputs.size() == 10 || inputs.size() == 11) // V1.2 add a optional data layout
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
    }
    else
    {
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
        OperationPtr operation, uint32_t operation_index)
{
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    std::shared_ptr<MaxPool2DOperation> pool = std::make_shared<MaxPool2DOperation>();
    NNAPI_CHECK_PTR(pool);
    pool->setDataLayout(DataLayout::NHWC);
    if (inputs.size() == 10)
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
    }
    else
    {
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
        OperationPtr operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 2, 1);
    std::shared_ptr<SqueezeOperation> squeeze = std::make_shared<SqueezeOperation>();
    NNAPI_CHECK_PTR(squeeze);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    if (inputs[1]->isConst()) {
        squeeze->axes.clear();
        int32_t* buffer = model->getBuffer<int32_t>(inputs[1]->weak_mem_ref.lock());
        squeeze->axes = convertAxes(buffer, inputs[1]->size(), inputs[0]->ndim());
        //TODO: remove buffer
    }
    truncateOperationIOs(model, operation, 1, 1);
    return squeeze;
}

OperationPtr NnApiInterpreter::map_PAD(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
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

OperationPtr NnApiInterpreter::map_MUL(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 3, 1);
    std::shared_ptr<MulOperation> mul = std::make_shared<MulOperation>();
    NNAPI_CHECK_PTR(mul);
    mul->setVxParam(OverflowPolicy::SATURATE, RoundingPolicy::RTNE);
    resetFusedType(model, operation, 2);
    truncateOperationIOs(model, operation, 2, 1);
    return mul;
}

OperationPtr NnApiInterpreter::map_MEAN(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 3, 1);
    std::shared_ptr<ReduceMeanOperation> mean = std::make_shared<ReduceMeanOperation>();
    NNAPI_CHECK_PTR(mean);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    if (inputs[1]->isConst()) {
        mean->axes.clear();
        int32_t* buffer = model->getBuffer<int32_t>(inputs[1]->weak_mem_ref.lock());
        mean->axes.assign(buffer, buffer + inputs[1]->size());
        //TODO: Remove Buffer
    }
    mean->keepDim = static_cast<bool>(inputs[2]->scalar.int32);
    truncateOperationIOs(model, operation, 1, 1);
    return mean;
}

OperationPtr NnApiInterpreter::map_SPACE_TO_DEPTH(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 2, 1);
    std::shared_ptr<SpaceToDepthOperation> sp_to_dp = std::make_shared<SpaceToDepthOperation>();
    NNAPI_CHECK_PTR(sp_to_dp);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    sp_to_dp->blockSize[0] = inputs[1]->scalar.int32;
    sp_to_dp->blockSize[1] = inputs[1]->scalar.int32;
    sp_to_dp->setDataLayout(DataLayout::NHWC);
    truncateOperationIOs(model, operation, 1, 1);
    return sp_to_dp;
}

OperationPtr NnApiInterpreter::map_DEPTH_TO_SPACE(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 2, 1);
    std::shared_ptr<DepthToSpaceOperation> dp_to_sp = std::make_shared<DepthToSpaceOperation>();
    NNAPI_CHECK_PTR(dp_to_sp);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    dp_to_sp->blockSize[0] = inputs[1]->scalar.int32;
    dp_to_sp->blockSize[1] = inputs[1]->scalar.int32;
    dp_to_sp->setDataLayout(DataLayout::NHWC);
    truncateOperationIOs(model, operation, 1, 1);
    return dp_to_sp;
}

OperationPtr NnApiInterpreter::map_SPACE_TO_BATCH_ND(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
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
        convert2DPadding(buffer, inputs[2]->size(),
                sp_to_bp->padFront.data(), sp_to_bp->padBack.data());
    } else {
        NNRT_LOGW_PRINT("Not support dynamic SPACE_TO_BATCH_ND.");
        assert(false);
    }
    truncateOperationIOs(model, operation, 1, 1);
    return sp_to_bp;
}

OperationPtr NnApiInterpreter::map_BATCH_TO_SPACE_ND(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
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
            resize->setDataLayout(getDataLayout(inputs[argList->ArgPos("data_layout")]->scalar.boolean));
        }
        auto inputOperand = inputs[argList->ArgPos("input")];
        auto outputOperand = model->getOperands(operation->outputs())[0];
        // No dynamic shape branch
        if (!nnrt::operand_utils::IsDynamicShape(inputOperand) &&
            !nnrt::operand_utils::IsDynamicShape(outputOperand)) {
            if (-1 != argList->ArgPos("output_height")) {
                // give output height and width
                resize->outputHeight = inputs[argList->ArgPos("output_height")]->scalar.int32;
                resize->outputWidth = inputs[argList->ArgPos("output_width")]->scalar.int32;

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
                    resize->outputHeight = uint32_t(orgHeight * heightScale);
                    resize->outputWidth = uint32_t(orgWidth * widthScale);
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
        OperationPtr operation, uint32_t operation_index)
{
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    std::shared_ptr<ResizeNearestNeighborOperation> resize = std::make_shared<ResizeNearestNeighborOperation>();
    NNAPI_CHECK_PTR(resize);

    std::vector<OperandType> argTypes;
    std::transform(
        inputs.begin(), inputs.end(), std::back_inserter(argTypes), [](const OperandPtr& operand) {
            return operand->type;
        });

    auto argList = api::requirement::nnapi::match("ResizeNearestNeighborOperation", argTypes);
    if (argList) {
        resize->setDataLayout(getDataLayout(inputs[argList->ArgPos("data_layout")]->scalar.boolean));
        auto inputOperand = inputs[argList->ArgPos("input")];
        auto outputOperand = model->getOperands(operation->outputs())[0];
        // No dynamic shape branch
        if (!nnrt::operand_utils::IsDynamicShape(inputOperand) &&
            !nnrt::operand_utils::IsDynamicShape(outputOperand)) {
            if (-1 != argList->ArgPos("output_height")) {
                // give output height and width
                resize->outputHeight = inputs[argList->ArgPos("output_height")]->scalar.int32;
                resize->outputWidth = inputs[argList->ArgPos("output_width")]->scalar.int32;

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
                    resize->outputHeight = uint32_t(orgHeight * heightScale);
                    resize->outputWidth = uint32_t(orgWidth * widthScale);
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
        OperationPtr operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 6, 2);
    std::shared_ptr<RnnOperation> rnn = std::make_shared<RnnOperation>();
    NNAPI_CHECK_PTR(rnn);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());

    // RNN's activation is NeuralNetwork::FuseType
    rnn->activation = inputs[5]->scalar.int32;
    truncateOperationIOs(model, operation, 5, 2);
    return rnn;
}

OperationPtr NnApiInterpreter::map_LSTM(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    std::shared_ptr<LstmUnitOperation> new_op = std::make_shared<LstmUnitOperation>();
    NNAPI_CHECK_PTR(new_op);

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

OperationPtr NnApiInterpreter::map_SVDF(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
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
        OperationPtr operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 4, 1);
    std::shared_ptr<LshProjectionOperation> new_op = std::make_shared<LshProjectionOperation>();
    NNAPI_CHECK_PTR(new_op);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    new_op->type = mapLshProjectionType(inputs[3]->scalar.int32);
    truncateOperationIOs(model, operation, 3, 1);
    return new_op;
}

OperationPtr NnApiInterpreter::map_L2_POOL_2D(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    std::shared_ptr<L2Pool2DOperation> pool = std::make_shared<L2Pool2DOperation>();
    NNAPI_CHECK_PTR(pool);
    pool->setDataLayout(DataLayout::NHWC);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    if (inputs.size() == 10 || inputs.size() == 11/*API LEVEL 29*/)
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

        //TODO: if io_in_nchw is true, no more permute required
        if (inputs.size() == 11) {
            pool->setDataLayout(getDataLayout(inputs[10]->scalar.boolean));
        }
    }
    else if(inputs.size() == 7 || inputs.size() == 8/*API LEVEL 29*/)
    {
        pool->padType = mapPadType(inputs[1]->scalar.int32);
        pool->strides[0] = inputs[2]->scalar.int32;
        pool->strides[1] = inputs[3]->scalar.int32;
        pool->ksize[0] = inputs[4]->scalar.int32;
        pool->ksize[1] = inputs[5]->scalar.int32;
        resetFusedType(model, operation, 6);

        //TODO: if io_in_nchw is true, no more permute required
        if (inputs.size() == 8 && inputs[7]) {
            pool->setDataLayout(DataLayout::NCHW);
        }
    }
    else{
        NNRT_LOGE_PRINT("Number of input parameter not valid");
        assert(false);
    }
    pool->setVxParam(OverflowPolicy::WRAP, RoundingPolicy::TO_ZERO, Rounding::FLOOR);
    truncateOperationIOs(model, operation, 1, 1);
    return pool;
}

OperationPtr NnApiInterpreter::map_STRIDED_SLICE(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
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
        OperationPtr operation, uint32_t operation_index)
{
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    std::shared_ptr<Deconv2DOperation> deconv2d;
    auto argList = matchArgList(inputs, "TransposeConv2DOperation");
    if (argList) {
        if (argList->ArgPos("output_shape") != -1) {
            // TODO: We donot support this case,
            // It needs shape inference to convert padType to pad
            //op->padType = mapPadtype(inputs[argList->ArgPos("impilicit_pad")]->scalar.int32);
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
        deconv2d->setDataLayout(getDataLayout(
                    inputs[argList->ArgPos("layout")]->scalar.boolean));
    } else {
        NNRT_LOGE_PRINT("Transpose conv2d argument list not support");
        assert(false);
    }

    /* set default dilation value */
    deconv2d->setVxParam(OverflowPolicy::WRAP, RoundingPolicy::TO_ZERO, Rounding::FLOOR);
    truncateOperationIOs(model, operation, 3, 1);
    return deconv2d;
}

OperationPtr NnApiInterpreter::map_TOPK(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
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
        OperationPtr operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 2, 1);
    std::shared_ptr<ArgmaxOperation> op = std::make_shared<ArgmaxOperation>();
    NNAPI_CHECK_PTR(op);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    op->axis = inputs[1]->scalar.int32;
    truncateOperationIOs(model, operation, 1, 1);
    return op;
}

OperationPtr NnApiInterpreter::map_ARGMIN(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 2, 1);
    std::shared_ptr<ArgminOperation> op = std::make_shared<ArgminOperation>();
    NNAPI_CHECK_PTR(op);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    op->axis = inputs[1]->scalar.int32;
    truncateOperationIOs(model, operation, 1, 1);
    return op;
}

OperationPtr NnApiInterpreter::map_GATHER(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
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
        OperationPtr operation, uint32_t operation_index)
{
    std::shared_ptr<ChannelShuffleOperation> channel_shuffle = std::make_shared<ChannelShuffleOperation>();
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
        OperationPtr operation, uint32_t operation_index)
{
    std::shared_ptr<SplitOperation> op = std::make_shared<SplitOperation>();
    NNAPI_CHECK_PTR(op);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    op->axis = inputs[1]->scalar.int32;
    op->split_number = inputs[2]->scalar.int32;
    truncateOperationIOs(model, operation, 1, operation->outputs().size());
    return op;
}

OperationPtr NnApiInterpreter::map_INSTANCE_NORMALIZATION(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 5, 1);
    std::shared_ptr<InstanceNormOperation> op = std::make_shared<InstanceNormOperation>();
    NNAPI_CHECK_PTR(op);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    op->gamma = inputs[1]->scalar.int32;
    op->beta = inputs[2]->scalar.float32;
    op->eps = inputs[3]->scalar.float32;
    op->setDataLayout(getDataLayout(inputs[4]->scalar.boolean));
    op->axes.push_back(-1);
    truncateOperationIOs(model, operation, 1, 1);
    return op;
}

OperationPtr NnApiInterpreter::map_GENERATE_PROPOSALS(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 11, 3);
    std::shared_ptr<GenerateProposalsOperation> op = std::make_shared<GenerateProposalsOperation>();
    NNAPI_CHECK_PTR(op);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    auto argList = matchArgList(inputs, "GenerateProposalsOperation");
    if (argList) {
        op->ratio_h = inputs[argList->ArgPos("ratio_h")]->scalar.float32;
        op->ratio_w = inputs[argList->ArgPos("ratio_w")]->scalar.float32;
        op->pre_nms_topn = inputs[argList->ArgPos("pre_nms_topn")]->scalar.int32;
        op->post_nms_topn = inputs[argList->ArgPos("post_nms_topn")]->scalar.int32;
        op->iou_threshold = inputs[argList->ArgPos("iou_threshold")]->scalar.float32;
        op->min_size = inputs[argList->ArgPos("min_size")]->scalar.float32;
        op->setDataLayout(getDataLayout(inputs[argList->ArgPos("layout")]->scalar.boolean));
    } else {
        NNRT_LOGE_PRINT("GenerateProposals argument list not support");
        assert(false);
    }
    truncateOperationIOs(model, operation, 4, 3);
    return op;
}

OperationPtr NnApiInterpreter::map_DETECTION_POSTPROCESSING(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
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
        op->maximum_class_per_detection = inputs[argList->ArgPos("maximum_class_per_detection")]->scalar.int32;
        op->maximum_detection_per_class = inputs[argList->ArgPos("maximum_detection_per_class")]->scalar.int32;
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
        OperationPtr operation, uint32_t operation_index)
{
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
        OperationPtr operation, uint32_t operation_index)
{
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
        OperationPtr operation, uint32_t operation_index)
{
    std::shared_ptr<ROIPoolingOperation> op = std::make_shared<ROIPoolingOperation>();
    NNAPI_CHECK_PTR(op);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    auto argList = matchArgList(inputs, "ROIPoolingOperation");
    if (argList) {
        op->height = inputs[argList->ArgPos("height")]->scalar.int32;
        op->width = inputs[argList->ArgPos("width")]->scalar.int32;
        op->height_ratio = inputs[argList->ArgPos("height_ratio")]->scalar.float32;
        op->width_ratio = inputs[argList->ArgPos("width_ratio")]->scalar.float32;
        op->setDataLayout(getDataLayout(inputs[argList->ArgPos("layout")]->scalar.boolean));
    } else {
        NNRT_LOGE_PRINT("ROIPooling argument list not support");
        assert(false);
    }
    truncateOperationIOs(model, operation, 3, 1);
    return op;
}

OperationPtr NnApiInterpreter::map_ROI_ALIGN(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    std::shared_ptr<ROIAlignOperation> op = std::make_shared<ROIAlignOperation>();
    NNAPI_CHECK_PTR(op);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    auto argList = matchArgList(inputs, "ROIAlignOperation");
    if (argList) {
        op->height = inputs[argList->ArgPos("height")]->scalar.int32;
        op->width = inputs[argList->ArgPos("width")]->scalar.int32;
        op->height_ratio = inputs[argList->ArgPos("height_ratio")]->scalar.float32;
        op->width_ratio = inputs[argList->ArgPos("width_ratio")]->scalar.float32;
        op->sampling_points_height = inputs[argList->ArgPos("sampling_points_height")]->scalar.int32;
        op->sampling_points_width = inputs[argList->ArgPos("sampling_points_width")]->scalar.int32;
        op->setDataLayout(getDataLayout(inputs[argList->ArgPos("layout")]->scalar.boolean));
        NNRT_LOGE_PRINT("ROIAlign layout: %d", getDataLayout(inputs[argList->ArgPos("layout")]->scalar.boolean));
    } else {
        NNRT_LOGE_PRINT("ROIAlign argument list not support");
        assert(false);
    }
    truncateOperationIOs(model, operation, 3, 1);
    return op;
}

OperationPtr NnApiInterpreter::map_HEATMAP_MAX_KEYPOINT(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 3, 2);
    std::shared_ptr<HeatmapMaxKeypointOperation> op = std::make_shared<HeatmapMaxKeypointOperation>();
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
        OperationPtr operation, uint32_t operation_index)
{
    std::shared_ptr<BoxWithNmsLimitOperation> op = std::make_shared<BoxWithNmsLimitOperation>();
    NNAPI_CHECK_PTR(op);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    auto argList = matchArgList(inputs, "BoxWithNmsLimitOperation");
    auto choose_nms_kernel_method = [](int32_t value) -> NmsKernelMethod {
        switch (value) {
            case 1:
                return NmsKernelMethod::Hard;
            case 2:
                return NmsKernelMethod::Linear;
            case 3:
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
        op->nms_kernel_method = choose_nms_kernel_method(
                inputs[argList->ArgPos("nms_kernel_method")]->scalar.int32);
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
        OperationPtr operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 3, 1);
    std::shared_ptr<LogSoftmaxOperation> op = std::make_shared<LogSoftmaxOperation>();
    NNAPI_CHECK_PTR(op);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    auto argList = matchArgList(inputs, "LogSoftmaxOperation");
    if (argList) {
        op->beta = inputs[argList->ArgPos("beta")]->scalar.float32;
        op->axis = inputs[argList->ArgPos("axis")]->scalar.int32;
    } else {
        NNRT_LOGE_PRINT("LogSoftmax argument list not support");
        assert(false);
    }
    truncateOperationIOs(model, operation, 1, 1);
    return op;
}


#define DECLARE_SAMPLE_OP(NAME, INPUT_NUM, OUTPUT_NUM, OPERATION_TYPE)  \
    OperationPtr NnApiInterpreter::map_##NAME(Model* model,             \
        OperationPtr operation, uint32_t operation_index)               \
    {                                                                   \
        NNAPI_CHECK_IO_NUM(operation, INPUT_NUM, OUTPUT_NUM);           \
        OperationPtr op = std::make_shared<OPERATION_TYPE>();           \
        if(op)                                                          \
            op->setDataLayout(DataLayout::NHWC);                        \
        else                                                            \
            NNRT_LOGE_PRINT("Invalid operaton pointer");                        \
        return op;                                                      \
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
DECLARE_SAMPLE_OP(PRELU, 2, 1, PReluOperation)
DECLARE_SAMPLE_OP(SELECT, 3, 1, SelectOperation)
DECLARE_SAMPLE_OP(SIN, 1, 1, SinOperation)
DECLARE_SAMPLE_OP(AXIS_ALIGNED_BBOX_TRANSFORM, 4, 1, AxisAlignedBBoxTransformOperation)

#undef DECLARE_SAMPLE_OP


#define DECLARE_REDUCTION_OP(NAME, OPERATION_TYPE)   \
    OperationPtr NnApiInterpreter::map_##NAME(Model* model, \
            OperationPtr operation, uint32_t operation_index) {   \
        auto op = std::make_shared<OPERATION_TYPE##Operation>();    \
        NNAPI_CHECK_PTR(op);  \
        std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());   \
        auto argList = matchArgList(inputs, ""#OPERATION_TYPE);   \
        if (argList) {  \
            if (inputs[argList->ArgPos("axes")]->isConst()) { \
                op->axes.clear(); \
                int32_t* buffer = model->getBuffer<int32_t>(inputs[1]->weak_mem_ref.lock());    \
                op->axes.assign(buffer, buffer + inputs[1]->size());  \
            }   \
            op->keepDim = static_cast<bool>(inputs[argList->ArgPos("keep_dim")]->scalar.int32); \
            truncateOperationIOs(model, operation, 1, 1);   \
        } else {    \
            NNRT_LOGE_PRINT("Number of input parameter is not valid"); \
        }   \
        return op;    \
    }

DECLARE_REDUCTION_OP(REDUCE_ALL, ReduceAll)
DECLARE_REDUCTION_OP(REDUCE_ANY, ReduceAny)
DECLARE_REDUCTION_OP(REDUCE_MAX, ReduceMax)
DECLARE_REDUCTION_OP(REDUCE_MIN, ReduceMin)
DECLARE_REDUCTION_OP(REDUCE_SUM, ReduceSum)
DECLARE_REDUCTION_OP(REDUCE_PROD, ReduceProd)
#undef DECLARE_REDUCTION_OP

}
