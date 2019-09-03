#include <assert.h>
#include <algorithm>
#include "vsi_nn_pub.h"
#include "types.h"
#include "model.h"
#include "graph_transformations/transformations.h"
#include "NeuralNetworks.h"
#include "error.h"

namespace ovxlib
{
#define NNAPI_CHECK_IO_NUM(op, in_num, out_num)         \
    do {                                                \
        if ((in_num > 0 && op->inputs().size() != (size_t)in_num)       \
         || (out_num > 0 && op->outputs().size() != (size_t)out_num)) {           \
            VSILOGW("Operation IO number mismatch. %d(%d), %d(%d)",     \
                    op->inputs().size(), in_num,        \
                    op->outputs().size(), out_num);     \
            return nullptr;                             \
        }                                               \
    } while(0)

#define NNAPI_CHECK_PTR(pad)                            \
    do {                                                \
        if (!pad) {                                     \
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

    /*customer Op*/
    //REGISTER_OP(VSI_RESIZE_NEAREST);
#undef REGISTER_OP

}

NnApiInterpreter::~NnApiInterpreter()
{

}

int NnApiInterpreter::run(Model* model, bool* modified)
{
    *modified = false;
    const std::map<uint32_t, Operation*>& operations = model->operations();
    for (auto it = operations.begin(); it != operations.end(); ++ it)
    {
        Operation* op = it->second;
        if (op_container_.find(op->type()) == op_container_.end())
        {
            VSILOGW("Not support operation %d", op->type());
            return AERROR_CODE(BAD_DATA);
        }
    }

    for (auto it = operations.begin(); it != operations.end(); ++ it)
    {
        uint32_t idx = it->first;
        Operation* op = it->second;
        VSILOGD("Convert node %u(%d)", idx, op->type());
        Operation* new_operation = (this->*op_container_[op->type()])(model, op, idx);
        if (!new_operation) {
            VSILOGW("Build operation: %d, index: %d fail", op->type(), idx);
            return AERROR_CODE(OUT_OF_MEMORY);
        }
        replaceOperation(model, idx, new_operation);
    }

    VSILOGD("Convert operation completed.");
    // Unique vector
    for (uint32_t index : operands_to_remove_) {
        //VSILOGD("Remove %d", index);
        if (model->isInput(index) || model->isOutput(index)) {
            VSILOGW("Try remove operand(%u) from model input or output, \
some operations may not support dynamic configure.", index);
        } else {
            model->removeOperand(index);
        }
    }

    return AERROR_CODE(NO_ERROR);
}

void NnApiInterpreter::replaceOperation(Model* model, uint32_t op_index,
        Operation* new_operation)
{
    Operation* org_operation = model->operation(op_index);
    new_operation->setInputs(org_operation->inputs());
    new_operation->setOutputs(org_operation->outputs());
    new_operation->setFusedType(org_operation->fusedType());
    model->operations()[op_index] = new_operation;
    delete org_operation;
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
            VSILOGE("Invalid padding type(%d)", type);
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
            VSILOGW("Unknow lsh projection type: %d", value);
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
            VSILOGW("Unknown lstm activation: %d.", value);
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

void NnApiInterpreter::fillIntArray(Model* model, Operation* operation,
        std::vector<int32_t>& array, int32_t op_index, bool reverse, bool is_axis)
{
    Operand* operand = model->operand(operation->input(op_index));
    int32_t* buffer = model->getBuffer<int32_t>(operand->mem_ref);
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

int32_t NnApiInterpreter::reverseMask(int32_t mask, size_t dim_num)
{
    auto get_bit_in_mask = [](int mask, int index) -> int {
        return (((int)0x1) << index) & mask;
    };
    int32_t new_mask = 0;
    for (int i = (int)dim_num - 1; i >= 0; -- i) {
        new_mask |= (get_bit_in_mask(mask, i) >> i) << ((dim_num - 1) - i);
    }
    return new_mask;
}

void NnApiInterpreter::truncateOperationIOs(Model* model, Operation* operation,
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



#define DECLARE_SAMPLE_OP(NAME, INPUT_NUM, OUTPUT_NUM, OPERATION_TYPE)  \
    Operation* NnApiInterpreter::map_##NAME(Model* model,                \
        Operation* operation, uint32_t operation_index)                 \
    {                                                                   \
        NNAPI_CHECK_IO_NUM(operation, INPUT_NUM, OUTPUT_NUM);            \
        return new OPERATION_TYPE();                                    \
    }

Operation* NnApiInterpreter::map_ADD(Model* model,
        Operation* operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 3, 1);
    resetFusedType(model, operation, 2);
    truncateOperationIOs(model, operation, 2, 1);
    return new AddOperation();
}

Operation* NnApiInterpreter::map_SUB(Model* model,
        Operation* operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 3, 1);
    resetFusedType(model, operation, 2);
    truncateOperationIOs(model, operation, 2, 1);
    return new SubOperation();
}

Operation* NnApiInterpreter::map_DIV(Model* model,
        Operation* operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 3, 1);
    DivOperation* div = new DivOperation();
    div->setVxParam(OverflowPolicy::SATURATE, RoundingPolicy::RTNE, Rounding::RTNE);
    resetFusedType(model, operation, 2);
    truncateOperationIOs(model, operation, 2, 1);
    return div;
}

Operation* NnApiInterpreter::map_CONCATENATION(Model* model,
        Operation* operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, -1, 1);
    ConcatOperation* concat = new ConcatOperation();
    NNAPI_CHECK_PTR(concat);
    std::vector<Operand*> inputs = model->getOperands(operation->inputs());
    concat->axis = inputs.back()->scalar.int32;
    truncateOperationIOs(model, operation, -2, 1);
    return concat;
}

Operation* NnApiInterpreter::map_CONV_2D(Model* model,
        Operation* operation, uint32_t operation_index)
{
    std::vector<Operand*> inputs = model->getOperands(operation->inputs());
    Conv2DOperation* conv2d = new Conv2DOperation();
    NNAPI_CHECK_PTR(conv2d);
    if (inputs.size() == 7)
    {
        conv2d->padType = mapPadType(inputs[3]->scalar.int32);
        conv2d->strides[0] = inputs[4]->scalar.int32;
        conv2d->strides[1] = inputs[5]->scalar.int32;
        resetFusedType(model, operation, 6);
    }
    else
    {
        conv2d->pad[0] = inputs[3]->scalar.int32;
        conv2d->pad[1] = inputs[4]->scalar.int32;
        conv2d->pad[2] = inputs[5]->scalar.int32;
        conv2d->pad[3] = inputs[6]->scalar.int32;
        conv2d->strides[0] = inputs[7]->scalar.int32;
        conv2d->strides[1] = inputs[8]->scalar.int32;
        resetFusedType(model, operation, 9);
    }
    /* set default dilation value */
    conv2d->dilations[0] = 1;
    conv2d->dilations[1] = 1;
    conv2d->setVxParam(OverflowPolicy::WRAP, RoundingPolicy::TO_ZERO, Rounding::FLOOR);
    truncateOperationIOs(model, operation, 3, 1);
    return conv2d;
}

Operation* NnApiInterpreter::map_DEPTHWISE_CONV_2D(Model* model,
        Operation* operation, uint32_t operation_index)
{
    std::vector<Operand*> inputs = model->getOperands(operation->inputs());
    DepthwiseConv2DOperation* conv2d = new DepthwiseConv2DOperation();
    NNAPI_CHECK_PTR(conv2d);
    if (inputs.size() == 11)
    {
        conv2d->pad[0] = inputs[3]->scalar.int32;
        conv2d->pad[1] = inputs[4]->scalar.int32;
        conv2d->pad[2] = inputs[5]->scalar.int32;
        conv2d->pad[3] = inputs[6]->scalar.int32;
        conv2d->strides[0] = inputs[7]->scalar.int32;
        conv2d->strides[1] = inputs[8]->scalar.int32;
        conv2d->multiplier = inputs[9]->scalar.int32;
        resetFusedType(model, operation, 10);
    }
    else
    {
        conv2d->padType = mapPadType(inputs[3]->scalar.int32);
        conv2d->strides[0] = inputs[4]->scalar.int32;
        conv2d->strides[1] = inputs[5]->scalar.int32;
        conv2d->multiplier = inputs[6]->scalar.int32;
        resetFusedType(model, operation, 7);
    }
    conv2d->setVxParam(OverflowPolicy::WRAP, RoundingPolicy::TO_ZERO, Rounding::FLOOR);
    truncateOperationIOs(model, operation, 3, 1);
    return conv2d;
}

Operation* NnApiInterpreter::map_RELU(Model* model,
        Operation* operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 1, 1);
    return new ReluOperation();
}

Operation* NnApiInterpreter::map_FULLY_CONNECTED(Model* model,
        Operation* operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 4, 1);
    FullyConnectedOperation* fc = new FullyConnectedOperation();
    NNAPI_CHECK_PTR(fc);
    std::vector<Operand*> inputs = model->getOperands(operation->inputs());
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

Operation* NnApiInterpreter::map_RESHAPE(Model* model,
        Operation* operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 2, 1);
    std::vector<Operand*> inputs = model->getOperands(operation->inputs());
    ReshapeOperation* reshape = new ReshapeOperation();
    NNAPI_CHECK_PTR(reshape);
    if (!inputs[1]->isConst()) {
        std::vector<Operand*> outputs = model->getOperands(operation->outputs());
        assert(outputs[0]->ndim() > 0);
        reshape->shape = std::vector<int32_t>(outputs[0]->dimensions.begin(),
                outputs[0]->dimensions.end());
    } else {
        fillIntArray(model, operation, reshape->shape, 1, false, false);
    }
    truncateOperationIOs(model, operation, 1, 1);
    return reshape;
}

Operation* NnApiInterpreter::map_SOFTMAX(Model* model,
        Operation* operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 2, 1);
    SoftmaxOperation* softmax = new SoftmaxOperation();
    NNAPI_CHECK_PTR(softmax);
    std::vector<Operand*> inputs = model->getOperands(operation->inputs());
    softmax->beta = inputs[1]->scalar.float32;
    truncateOperationIOs(model, operation, 1, 1);
    return softmax;
}

Operation* NnApiInterpreter::map_TRANSPOSE(Model* model,
        Operation* operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 2, 1);
    PermuteOperation* permute = new PermuteOperation();
    NNAPI_CHECK_PTR(permute);
    std::vector<Operand*> inputs = model->getOperands(operation->inputs());
    fillIntArray(model, operation, permute->perm, 1, false, false);
    truncateOperationIOs(model, operation, 1, 1);
    return permute;
}

Operation* NnApiInterpreter::map_AVERAGE_POOL_2D(Model* model,
        Operation* operation, uint32_t operation_index)
{
    std::vector<Operand*> inputs = model->getOperands(operation->inputs());
    AveragePool2DOperation* pool = new AveragePool2DOperation();
    NNAPI_CHECK_PTR(pool);
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
    pool->poolMode = PoolMode::VALID;
    pool->setVxParam(OverflowPolicy::WRAP, RoundingPolicy::TO_ZERO, Rounding::FLOOR);
    truncateOperationIOs(model, operation, 1, 1);
    return pool;
}

Operation* NnApiInterpreter::map_MAX_POOL_2D(Model* model,
        Operation* operation, uint32_t operation_index)
{
    std::vector<Operand*> inputs = model->getOperands(operation->inputs());
    MaxPool2DOperation* pool = new MaxPool2DOperation();
    NNAPI_CHECK_PTR(pool);
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


Operation* NnApiInterpreter::map_SQUEEZE(Model* model,
        Operation* operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 2, 1);
    SqueezeOperation* squeeze = new SqueezeOperation();
    NNAPI_CHECK_PTR(squeeze);
    std::vector<Operand*> inputs = model->getOperands(operation->inputs());
    if (inputs[1]->isConst()) {
        squeeze->axes.clear();
        int32_t* buffer = model->getBuffer<int32_t>(inputs[1]->mem_ref);
        squeeze->axes = convertAxes(buffer, inputs[1]->size(), inputs[0]->ndim());
        //TODO: remove buffer
    }
    truncateOperationIOs(model, operation, 1, 1);
    return squeeze;
}

Operation* NnApiInterpreter::map_PAD(Model* model,
        Operation* operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 2, 1);
    PadOperation* pad = new PadOperation();
    NNAPI_CHECK_PTR(pad);
    std::vector<Operand*> inputs = model->getOperands(operation->inputs());
    int32_t* padding = model->getBuffer<int32_t>(inputs[1]->mem_ref);
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

Operation* NnApiInterpreter::map_MUL(Model* model,
        Operation* operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 3, 1);
    MulOperation* mul = new MulOperation();
    NNAPI_CHECK_PTR(mul);
    mul->setVxParam(OverflowPolicy::SATURATE, RoundingPolicy::RTNE);
    resetFusedType(model, operation, 2);
    truncateOperationIOs(model, operation, 2, 1);
    return mul;
}

Operation* NnApiInterpreter::map_MEAN(Model* model,
        Operation* operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 3, 1);
    MeanOperation* mean = new MeanOperation();
    NNAPI_CHECK_PTR(mean);
    std::vector<Operand*> inputs = model->getOperands(operation->inputs());
    if (inputs[1]->isConst()) {
        mean->axes.clear();
        int32_t* buffer = model->getBuffer<int32_t>(inputs[1]->mem_ref);
        mean->axes.assign(buffer, buffer + inputs[1]->size());
        //TODO: Remove Buffer
    }
    mean->keepDim = static_cast<bool>(inputs[2]->scalar.int32);
    truncateOperationIOs(model, operation, 1, 1);
    return mean;
}

Operation* NnApiInterpreter::map_SPACE_TO_DEPTH(Model* model,
        Operation* operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 2, 1);
    SpaceToDepthOperation* sp_to_dp = new SpaceToDepthOperation();
    NNAPI_CHECK_PTR(sp_to_dp);
    std::vector<Operand*> inputs = model->getOperands(operation->inputs());
    sp_to_dp->blockSize[0] = inputs[1]->scalar.int32;
    sp_to_dp->blockSize[1] = inputs[1]->scalar.int32;
    truncateOperationIOs(model, operation, 1, 1);
    return sp_to_dp;
}

Operation* NnApiInterpreter::map_DEPTH_TO_SPACE(Model* model,
        Operation* operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 2, 1);
    DepthToSpaceOperation* dp_to_sp = new DepthToSpaceOperation();
    NNAPI_CHECK_PTR(dp_to_sp);
    std::vector<Operand*> inputs = model->getOperands(operation->inputs());
    dp_to_sp->blockSize[0] = inputs[1]->scalar.int32;
    dp_to_sp->blockSize[1] = inputs[1]->scalar.int32;
    truncateOperationIOs(model, operation, 1, 1);
    return dp_to_sp;
}

Operation* NnApiInterpreter::map_SPACE_TO_BATCH_ND(Model* model,
        Operation* operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 3, 1);
    SpaceToBatchNDOperation* sp_to_bp = new SpaceToBatchNDOperation();
    NNAPI_CHECK_PTR(sp_to_bp);
    std::vector<Operand*> inputs = model->getOperands(operation->inputs());
    if (inputs[1]->isConst() && inputs[2]->isConst()) {
        int32_t* buffer = model->getBuffer<int32_t>(inputs[1]->mem_ref);
        sp_to_bp->blockSize.assign(buffer, buffer + inputs[1]->size());
        buffer = model->getBuffer<int32_t>(inputs[2]->mem_ref);
        sp_to_bp->padFront.resize(inputs[0]->ndim() - 2);
        sp_to_bp->padBack.resize(inputs[0]->ndim() - 2);
        convert2DPadding(buffer, inputs[2]->size(),
                sp_to_bp->padFront.data(), sp_to_bp->padBack.data());
    } else {
        VSILOGW("Not support dynamic SPACE_TO_BATCH_ND.");
        assert(false);
    }
    truncateOperationIOs(model, operation, 1, 1);
    return sp_to_bp;
}

Operation* NnApiInterpreter::map_BATCH_TO_SPACE_ND(Model* model,
        Operation* operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 2, 1);
    BatchToSpaceNDOperation* bp_to_sp = new BatchToSpaceNDOperation();
    NNAPI_CHECK_PTR(bp_to_sp);
    std::vector<Operand*> inputs = model->getOperands(operation->inputs());
    if (inputs[1]->isConst()) {
        int32_t* buffer = model->getBuffer<int32_t>(inputs[1]->mem_ref);
        bp_to_sp->blockSize.assign(buffer, buffer + inputs[1]->size());
    }
    bp_to_sp->cropStart.resize(inputs[0]->ndim() - 2);
    bp_to_sp->cropEnd.resize(inputs[0]->ndim() - 2);

    truncateOperationIOs(model, operation, 1, 1);
    return bp_to_sp;
}

Operation* NnApiInterpreter::map_RESIZE_BILINEAR(Model* model,
        Operation* operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 3, 1);
    ResizeBilinearOperation* resize = new ResizeBilinearOperation();
    NNAPI_CHECK_PTR(resize);
    std::vector<Operand*> inputs = model->getOperands(operation->inputs());
    resize->outputHeight = inputs[1]->scalar.int32;
    resize->outputWidth = inputs[2]->scalar.int32;
    truncateOperationIOs(model, operation, 1, 1);
    return resize;
}

Operation* NnApiInterpreter::map_LOCAL_RESPONSE_NORMALIZATION(Model* model,
        Operation* operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 5, 1);
    LocalResponseNormOperation* lrn = new LocalResponseNormOperation();
    NNAPI_CHECK_PTR(lrn);
    std::vector<Operand*> inputs = model->getOperands(operation->inputs());
    lrn->radius = inputs[1]->scalar.int32;
    lrn->bias = inputs[2]->scalar.float32;
    lrn->scale = inputs[3]->scalar.float32;
    lrn->exponent = inputs[4]->scalar.float32;
    truncateOperationIOs(model, operation, 1, 1);
    return lrn;
}

Operation* NnApiInterpreter::map_RNN(Model* model,
        Operation* operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 6, 2);
    auto rnn = new RnnOperation();
    auto inputs = model->getOperands(operation->inputs());

    // RNN's activation is NeuralNetwork::FuseType
    rnn->activation = inputs[5]->scalar.int32;
    truncateOperationIOs(model, operation, 5, 2);
    return rnn;
}

Operation* NnApiInterpreter::map_LSTM(Model* model,
        Operation* operation, uint32_t operation_index)
{
    LstmUnitOperation* new_op = new LstmUnitOperation();
    NNAPI_CHECK_PTR(new_op);

    std::vector<Operand*> inputs = model->getOperands(operation->inputs());
    auto input_num = inputs.size();

    new_op->activation = mapLstmActivationType(inputs[20]->scalar.int32);
    new_op->cellClip = inputs[21]->scalar.float32;
    new_op->projClip = inputs[22]->scalar.float32;
    input_num -= 3;
    truncateOperationIOs(model, operation, 20, 4);

    while (input_num < LstmUnitOperation::INPUT_COUNT) {
        operation->inputs().emplace_back(-1);
        VSILOGD("Append Inputs at [%d]", input_num);
        ++input_num;
    }

    NNAPI_CHECK_IO_NUM(operation, LstmUnitOperation::INPUT_COUNT, LstmUnitOperation::OUTPUT_COUNT);

    return new_op;
}

Operation* NnApiInterpreter::map_SVDF(Model* model,
        Operation* operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 7, 2);
    SvdfOperation* new_op = new SvdfOperation();
    NNAPI_CHECK_PTR(new_op);
    std::vector<Operand*> inputs = model->getOperands(operation->inputs());
    new_op->rank = inputs[5]->scalar.int32;
    resetFusedType(model, operation, 6);
    truncateOperationIOs(model, operation, 5, 2);
    return new_op;
}

Operation* NnApiInterpreter::map_LSH_PROJECTION(Model* model,
        Operation* operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 4, 1);
    LshProjectionOperation* new_op = new LshProjectionOperation();
    NNAPI_CHECK_PTR(new_op);
    std::vector<Operand*> inputs = model->getOperands(operation->inputs());
    new_op->type = mapLshProjectionType(inputs[3]->scalar.int32);
    truncateOperationIOs(model, operation, 3, 1);
    return new_op;
}

Operation* NnApiInterpreter::map_L2_POOL_2D(Model* model,
        Operation* operation, uint32_t operation_index)
{
    L2Pool2DOperation* pool = new L2Pool2DOperation();
    NNAPI_CHECK_PTR(pool);
    std::vector<Operand*> inputs = model->getOperands(operation->inputs());
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
        if (inputs.size() == 11 && inputs[10]) {
            pool->setOperandLayout(OperandLayout::NCHW);
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
            pool->setOperandLayout(OperandLayout::NCHW);
        }
    }
    else{
        VSILOGE("Number of input parameter not valid");
        assert(false);
    }
    pool->setVxParam(OverflowPolicy::WRAP, RoundingPolicy::TO_ZERO, Rounding::FLOOR);
    truncateOperationIOs(model, operation, 1, 1);
    return pool;
}

Operation* NnApiInterpreter::map_STRIDED_SLICE(Model* model,
        Operation* operation, uint32_t operation_index)
{
    NNAPI_CHECK_IO_NUM(operation, 7, 1);
    StridedSliceOperation* new_op = new StridedSliceOperation();
    NNAPI_CHECK_PTR(new_op);
    std::vector<Operand*> inputs = model->getOperands(operation->inputs());
    int32_t* starts = model->getBuffer<int32_t>(inputs[1]->mem_ref);
    int32_t* ends = model->getBuffer<int32_t>(inputs[2]->mem_ref);
    int32_t* strides = model->getBuffer<int32_t>(inputs[3]->mem_ref);
    new_op->starts.assign(starts, starts + inputs[1]->size());
    new_op->ends.assign(ends, ends + inputs[2]->size());
    new_op->strides.assign(strides, strides + inputs[3]->size());
    new_op->beginMask = inputs[4]->scalar.int32;
    new_op->endMask = inputs[5]->scalar.int32;
    new_op->shrinkAxisMask = inputs[6]->scalar.int32;
    truncateOperationIOs(model, operation, 1, 1);
    return new_op;
}

DECLARE_SAMPLE_OP(RELU1, 1, 1, Relu1Operation)
DECLARE_SAMPLE_OP(RELU6, 1, 1, Relu6Operation)
DECLARE_SAMPLE_OP(TANH, 1, 1, TanhOperation)
DECLARE_SAMPLE_OP(SIGMOID, 1, 1, SigmoidOperation)
DECLARE_SAMPLE_OP(FLOOR, 1, 1, FloorOperation)
DECLARE_SAMPLE_OP(DEQUANTIZE, 1, 1, DequantizeOperation)
DECLARE_SAMPLE_OP(L2_NORMALIZATION, 1, 1, L2NormOperation)
DECLARE_SAMPLE_OP(EMBEDDING_LOOKUP, 2, 1, EmbeddingLookupOperation)
DECLARE_SAMPLE_OP(HASHTABLE_LOOKUP, 3, 2, HashtableLookupOperation)

}
