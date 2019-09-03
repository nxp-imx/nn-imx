#include <vector>
#include "vsi_nn_pub.h"
#include "operation.h"
#include "operand.h"
#include "types.h"

namespace ovxlib
{
inline const char* get_operation_string(OperationType type)
{
    switch(type)
    {
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
        case OperationType::MEAN:
            return "mean";
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
        default:
            return nullptr;
    }
    return nullptr;
}

Operation::Operation(OperationType type)
    : type_(type)
{ }

void Operation::setInputs(const uint32_t* inputs, uint32_t input_size)
{
    inputs_.clear();
    if (nullptr == inputs || input_size == 0) {
        return;
    }
    inputs_.insert(inputs_.begin(),
            inputs, inputs + input_size );
}

void Operation::setOutputs(const uint32_t* outputs, uint32_t output_size)
{
    outputs_.clear();
    if (nullptr == outputs || output_size == 0) {
        return;
    }
    outputs_.insert(outputs_.begin(),
            outputs, outputs + output_size );
}

void Operation::setInputs(const std::vector<uint32_t>& inputs)
{
    inputs_ = inputs;
}

void Operation::setOutputs(const std::vector<uint32_t>& outputs)
{
    outputs_ = outputs;
}

uint32_t Operation::input(uint32_t index) {
    if (index < inputs_.size()) {
        return inputs_[index];
    }
    return OVXLIB_INVALID_OPERAND_INDEX;
}

uint32_t Operation::output(uint32_t index) {
    if (index < outputs_.size()) {
        return outputs_[index];
    }
    return OVXLIB_INVALID_OPERAND_INDEX;
}

bool Operation::replaceOutputs(uint32_t org_index, uint32_t new_index)
{
    int pos = find_position(outputs_, org_index);
    if (pos < 0)
    {
        return false;
    }
    outputs_[pos] = new_index;
    return true;
}

bool Operation::replaceInputs(uint32_t org_index, uint32_t new_index)
{
    int pos = find_position(inputs_, org_index);
    if (pos < 0)
    {
        return false;
    }
    inputs_[pos] = new_index;
    return true;
}

int Operation::find_position(std::vector<uint32_t> operands_indexes, uint32_t index)
{
    int pos = 0;
    for (;pos < (int)operands_indexes.size(); ++ pos)
    {
        if (index == operands_indexes[pos])
        {
            break;
        }
    }
    if (pos == (int)operands_indexes.size())
    {
        pos = -1;
    }
    return pos;
}

void Operation::setVxParam(OverflowPolicy overflow_policy,
    RoundingPolicy rounding_policy, Rounding down_scale_size_rounding,
    uint32_t accumulator_bits)
{
    vx_param_.overflowPolicy = overflow_policy;
    vx_param_.roundingPolicy = rounding_policy;
    vx_param_.downScaleSizeRounding = down_scale_size_rounding;
    vx_param_.accumulatorBits = accumulator_bits;
}

void Operation::echo(uint32_t index) {
    char buf[256] = {0};
    size_t sz = 0;
    sz += snprintf(&buf[sz], 256 - sz, "%-4u ", index);
    const char * op_str =  get_operation_string(type_);
    if (!op_str)
    {
        sz += snprintf(&buf[sz], 256 - sz, "%-30d", (int32_t)type_);
    }
    else
    {
        sz += snprintf(&buf[sz], 256 - sz, "%-30s", op_str);
    }
    char subbuf[128];
    size_t subsz = 0;
    for (uint32_t i = 0; i < inputs_.size(); ++i)
    {
        subsz += snprintf(&subbuf[subsz], 128 - subsz, "% d,", inputs_[i]);
    }
    sz += snprintf(&buf[sz], 256 - sz, " i[%-20s]", subbuf);
    subsz = 0;
    for (uint32_t i = 0; i < outputs_.size(); ++i)
    {
        subsz += snprintf(&subbuf[subsz], 128 - subsz, "% d,", outputs_[i]);
    }
    sz += snprintf(&buf[sz], 256 - sz, " o[%-20s]", subbuf);
    VSILOGD("%s", buf);
}
}
