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
#include "model.hpp"
#include "error.hpp"
#include "ovxlib_delegate.hpp"
#include "vsi_nn_pub.h"
#include "op/public.hpp"

namespace nnrt
{
using namespace op;
namespace {
void removeOperationWithNullInput(nnrt::Model* model)
{
    auto operands = model->operands();

    std::vector<uint32_t> operation_reduce_list;
    for (auto op_itor = model->operations().begin();
            op_itor != model->operations().end();
            ++ op_itor) {
        auto operation = op_itor->second;

        bool all_input_empty(true);
        for (size_t i = 0 ; all_input_empty && i < operation->inputNum(); ++i){
            all_input_empty = (operands[operation->input(i)]->isNull());
        }

        for (size_t i = 0; all_input_empty && i < operation->outputNum(); ++i){
            NNRT_LOGD_PRINT("Mark operand(%d) as NUll",  operation->output(i));
            operands[operation->output(i)]->setNull();
        }
        if (all_input_empty){
            NNRT_LOGD_PRINT("Operation[%d] planned to remove", op_itor->first);
            operation_reduce_list.push_back(op_itor->first);
        }
    }

    for (auto op_idx = operation_reduce_list.begin();
            op_idx != operation_reduce_list.end();
            ++ op_idx) {
        model->operations().erase(model->operations().find(*op_idx));
    }
}

vsi_nn_activation_e mapLstmUnitActivation(const FusedType& ftype)
{
    vsi_nn_activation_e rValue = VSI_NN_ACT_NONE;

    switch (ftype){
    case FusedType::RELU1:
        NNRT_LOGE_PRINT("RELU1 Not supported, use RELU in case crash");
        rValue = VSI_NN_ACT_RELU;
        break;
    case FusedType::RELU:
        rValue = VSI_NN_ACT_RELU;
        break;
    case FusedType::RELU6:
        rValue = VSI_NN_ACT_RELU6;
        break;
    case FusedType::TANH:
        rValue = VSI_NN_ACT_TANH;
        break;
    case FusedType::SIGMOID:
        rValue = VSI_NN_ACT_SIGMOID;
        break;
    default:
        NNRT_LOGE_PRINT("Not supported activation type %d for LSTM_Unit", ftype);
        assert(false);
        }

    return rValue;
}
}

OvxlibDelegate::OvxlibDelegate()
{
#define REGISTER_OP(NAME)   do {                            \
    op_container_[OperationType::NAME] = &OvxlibDelegate::addNode_##NAME;  \
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
    REGISTER_OP(DATA_CONVERT);
    REGISTER_OP(RELU1);
    REGISTER_OP(RELU6);
    REGISTER_OP(SIGMOID);
    REGISTER_OP(TANH);
    REGISTER_OP(LEAKY_RELU);
    REGISTER_OP(SOFT_RELU);
    REGISTER_OP(SQRT);
    REGISTER_OP(SQUARE);
    REGISTER_OP(DIV);
    REGISTER_OP(SUB);
    REGISTER_OP(QUANTIZE);
    REGISTER_OP(DEQUANTIZE);
    REGISTER_OP(SPACE_TO_DEPTH);
    REGISTER_OP(DEPTH_TO_SPACE);
    REGISTER_OP(L2_NORM);
    REGISTER_OP(RESIZE_BILINEAR);
    REGISTER_OP(LOCAL_RESPONSE_NORM);
    REGISTER_OP(STRIDED_SLICE);
    REGISTER_OP(SPACE_TO_BATCH_ND);
    REGISTER_OP(BATCH_TO_SPACE_ND);
    REGISTER_OP(EMBEDDING_LOOKUP);
    REGISTER_OP(RNN);
    REGISTER_OP(HASHTABLE_LOOKUP);
    REGISTER_OP(SVDF);
    REGISTER_OP(LSH_PROJECTION);
    REGISTER_OP(L2_POOL_2D);
    REGISTER_OP(LSTM);
    REGISTER_OP(FLOOR);
    REGISTER_OP(BATCH_NORM);
    REGISTER_OP(MAXIMUM);
    REGISTER_OP(MINIMUM);
    REGISTER_OP(RSQRT);
    REGISTER_OP(PRELU);
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
    REGISTER_OP(INSTANCE_NORM);
    REGISTER_OP(LESS);
    REGISTER_OP(LESS_EQUAL);
    REGISTER_OP(LOGICAL_AND);
    REGISTER_OP(LOGICAL_OR);
    REGISTER_OP(NEG);
    REGISTER_OP(NOT_EQUAL);
    REGISTER_OP(POW);
    REGISTER_OP(LOG);
    REGISTER_OP(ROI_ALIGN);
    REGISTER_OP(ROI_POOLING);
    REGISTER_OP(SELECT);
    REGISTER_OP(SLICE);
    REGISTER_OP(SPLIT);
    REGISTER_OP(DECONV_2D);
    REGISTER_OP(SIN);
    REGISTER_OP(REDUCE_ALL);
    REGISTER_OP(REDUCE_ANY);
    REGISTER_OP(REDUCE_MAX);
    REGISTER_OP(REDUCE_MIN);
    REGISTER_OP(REDUCE_PROD);
    REGISTER_OP(REDUCE_SUM);
    REGISTER_OP(AXIS_ALIGNED_BBOX_TRANSFORM);
    REGISTER_OP(GENERATE_PROPOSALS);
    REGISTER_OP(RANDOM_MULTINOMIAL);
    REGISTER_OP(HEATMAP_MAX_KEYPOINT);
    REGISTER_OP(BOX_WITH_NMS_LIMIT);
    REGISTER_OP(LOG_SOFTMAX);
    REGISTER_OP(TOPK);
    REGISTER_OP(DETECTION_POSTPROCESSING);
    REGISTER_OP(TILE);
#undef REGISTER_OP
}

OvxlibDelegate::~OvxlibDelegate()
{

}

int OvxlibDelegate::process(nnrt::Model* model, vsi_nn_context_t ctx)
{
    if (!model) {
        NNRT_LOGW_PRINT("Model is null.");
        return NNA_ERROR_CODE(UNEXPECTED_NULL);
    }
    int err = NNA_ERROR_CODE(NO_ERROR);
    if (nullptr == ctx)
    {
        ctx = vsi_nn_CreateContext();
        if (!ctx) {
            NNRT_LOGW_PRINT("Create context fail.");
            return NNA_ERROR_CODE(OUT_OF_MEMORY);
        }
    }
    graph_ = vsi_nn_CreateGraph(ctx, 0, 0);
    if (nullptr == graph_)
    {
        return NNA_ERROR_CODE(OUT_OF_MEMORY);
    }

    auto operands = model->operands();

    // remove nodes which all input are null and reset its sucessor node's input
    removeOperationWithNullInput(model);

    for(auto it = operands.begin(); it != operands.end(); ++ it)
    {
        int idx = it->first;
        OperandPtr operand = it->second;
        if(!operand->isTensor() || operand->isNull())
        {
            NNRT_LOGD_PRINT("Skip Operand[%d]", idx);
            // If current operand is the graphic input, we should remove it from input
            //auto& model_inputs_idx = model->inputIndexes();
            //auto is_input = std::find(model_inputs_idx.begin(), model_inputs_idx.end(), idx);
            //if (is_input != model_inputs_idx.end()) {
            //    // erase input from model if user set it 'no-value'
            //    model_inputs_idx.erase(is_input);
            //}

            continue;
        }
        //NNRT_LOGD_PRINT("Add tensor (%u)", idx);

        if (operand->isConst())
        {
            void* data = model->getBuffer<void>(operand->weak_mem_ref.lock());
            if (operand->type == OperandType::TENSOR_FLOAT16 &&
                    (operand->bytes() * 2) == operand->weak_mem_ref.lock()->len_ ) {
                void* fp16_ptr = malloc(operand->bytes());
                if (nullptr == fp16_ptr) {
                    NNRT_LOGE_PRINT("Out of memory.");
                    return NNA_ERROR_CODE(OUT_OF_MEMORY);
                }
                vsi_nn_dtype_t fmt16;
                vsi_nn_dtype_t fmt32;
                fmt16.vx_type = VSI_NN_TYPE_FLOAT16;
                fmt32.vx_type = VSI_NN_TYPE_FLOAT32;
                vsi_nn_DtypeConvertRawData((uint8_t*)data, operand->weak_mem_ref.lock()->len_,
                        &fmt32, (uint8_t*)fp16_ptr, operand->bytes(), &fmt16);
                err = addTensor(graph_, operand,
                        TensorLifeTime::CONST, idx, fp16_ptr);
                free(fp16_ptr);
            } else {
                err = addTensor(graph_, operand,
                        TensorLifeTime::CONST, idx, data);
            }
        }
        else if (model->isInput(idx) || model->isOutput(idx))
        {
            err = addTensor(graph_, operand,
                    TensorLifeTime::NORMAL, idx);
        }
        else
        {
            err = addTensor(graph_, operand,
                    TensorLifeTime::VIRTUAL, idx);
        }
        if (err != NNA_ERROR_CODE(NO_ERROR))
        {
            return err;
        }
    }
    for (auto it = tensor_map_.begin(); it != tensor_map_.end(); ++it)
    {
        OperandPtr operand = model->operand(it->first);
        if (!operand) {
            NNRT_LOGW_PRINT("Not operand found: %u, %u", it->first, it->second);
        }
        if (operand->isConst() && operand->perm().size() > 0)
        {
            vsi_nn_tensor_t* tensor = vsi_nn_GetTensor(graph_, it->second);
            // TODO: Fixme transpose tensor to use driver whcn sequence.
            //std::vector<uint32_t> perm = convertPermute(operand->perm());
            std::vector<uint32_t> t2c_perm = operand->perm();
            std::vector<uint32_t> as_shape;
            as_shape.resize(operand->dimensions.size());

            std::vector<uint32_t> c2t_perm(t2c_perm.size());
            for (size_t i = 0; i < t2c_perm.size(); ++i) {
                c2t_perm[i] = t2c_perm[t2c_perm[i]];
            }
            for (size_t i = 0; i < c2t_perm.size(); ++i) {
                as_shape[i] = operand->dimensions[c2t_perm[i]];
            }
            vsi_nn_TransposeTensor(
                graph_, tensor, t2c_perm.data(), t2c_perm.size(), as_shape.data());
        }
    }

    vsi_nn_SetGraphInputs(graph_, nullptr, model->inputIndexes().size());
    vsi_nn_SetGraphOutputs(graph_, nullptr, model->outputIndexes().size());
    for (size_t i = 0; i < model->inputIndexes().size(); ++ i)
    {
        graph_->input.tensors[i] = getMappedTensor(model->inputIndex(i));
    }
    for (size_t i = 0; i < model->outputIndexes().size(); ++ i)
    {
        graph_->output.tensors[i] = getMappedTensor(model->outputIndex(i));
    }
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
        NNRT_LOGD_PRINT("Add node %u(%d)", idx, op->type());
        int err = (this->*op_container_[op->type()])(model, op, idx);
        if (NNA_ERROR_CODE(NO_ERROR) != err)
        {
            NNRT_LOGW_PRINT("Build operation: %d, index: %d ", op->type(), idx);
            return err;
        }
    }
    vsi_nn_PrintGraph(graph_);
    if (VSI_FAILURE == vsi_nn_SetupGraph(graph_, true))
    {
        NNRT_LOGW_PRINT("Setup graph failure.");
        return NNA_ERROR_CODE(BAD_DATA);
    }
    NNRT_LOGD_PRINT("Verify graph ...");
    vsi_status status = vsi_nn_VerifyGraph(graph_);

    if (status != VSI_SUCCESS)
    {
        NNRT_LOGW_PRINT("Verify graph error: %d", status);
        return NNA_ERROR_CODE(BAD_DATA);
    }
    NNRT_LOGD_PRINT("Compile graph completed.");

    return err;
}

vsi_nn_graph_t* OvxlibDelegate::throwGraph()
{
    vsi_nn_graph_t* graph = graph_;
    graph_ = nullptr;
    return graph;
}

vsi_nn_pad_e OvxlibDelegate::getPaddingType(PadType type)
{
    switch (type) {
        case PadType::SAME:
            return VSI_NN_PAD_SAME;
        case PadType::VALID:
            return VSI_NN_PAD_VALID;
        case PadType::AUTO:
            return VSI_NN_PAD_AUTO;
        default:
            NNRT_LOGE_PRINT("Invalid padding type(%d)", type);
            assert(false);
            return VSI_NN_PAD_AUTO;
    }
    return VSI_NN_PAD_AUTO;
}

vsi_nn_round_type_e OvxlibDelegate::getRoundType(Rounding type)
{
    switch (type) {
        case Rounding::CEILING:
            return VSI_NN_ROUND_CEIL;
        case Rounding::FLOOR:
            return VSI_NN_ROUND_FLOOR;
        default:
            NNRT_LOGE_PRINT("Invalid padding type(%d)", type);
            assert(false);
            return VSI_NN_ROUND_FLOOR;
    }
    return VSI_NN_ROUND_FLOOR;
}

vsi_nn_op_t OvxlibDelegate::getActivation(FusedType fused_code)
{
    vsi_nn_op_t op = VSI_NN_OP_NA;
    switch (fused_code){
        case FusedType::RELU:
            op = VSI_NN_OP_RELU;
            break;
        case FusedType::RELU1:
            op = VSI_NN_OP_RELU1;
            break;
        case FusedType::RELU6:
            op = VSI_NN_OP_RELU6;
            break;
        default:
            break;
    }
    return op;
}

vsi_nn_lsh_projection_type_e OvxlibDelegate::mapLshProjectionType(LshProjectionType type)
{
    switch (type) {
        case LshProjectionType::SPARSE:
            return VSI_NN_LSH_PROJECTION_SPARSE;
        case LshProjectionType::DENSE:
            return VSI_NN_LSH_PROJECTION_DENSE;
        default:
            break;
    }
    return VSI_NN_LSH_PROJECTION_SPARSE;
}

vsi_nn_pad_mode_e OvxlibDelegate::mapPadMode(PadMode mode)
{
    switch (mode) {
        case PadMode::CONSTANT:
            return VSI_NN_PAD_MODE_CONSTANT;
        default:
            break;
    }
    return VSI_NN_PAD_MODE_CONSTANT;
}

vsi_nn_type_e OvxlibDelegate::mapTensorType(OperandType code)
{
    vsi_nn_type_e dtype;
    dtype = VSI_NN_TYPE_FLOAT32;
    switch(code)
    {
        case OperandType::TENSOR_FLOAT32:
            dtype = VSI_NN_TYPE_FLOAT32;
            break;
        case OperandType::TENSOR_FLOAT16:
            dtype = VSI_NN_TYPE_FLOAT16;
            break;
        case OperandType::TENSOR_INT32:
        case OperandType::TENSOR_QUANT32_SYMM:
            dtype = VSI_NN_TYPE_INT32;
            break;
        case OperandType::TENSOR_INT16:
            dtype = VSI_NN_TYPE_INT16;
            break;
        case OperandType::TENSOR_QUANT8_ASYMM:
            dtype = VSI_NN_TYPE_UINT8;
            break;
        case OperandType::TENSOR_QUANT8_SYMM:
            dtype = VSI_NN_TYPE_INT8;
            break;
        case OperandType::TENSOR_BOOL8:
            dtype = VSI_NN_TYPE_BOOL8;
            break;
        case OperandType::TENSOR_QUANT16_SYMM:
            NNRT_LOGE_PRINT("Ovxlib doesn't support quant16_symm");
            dtype = VSI_NN_TYPE_INT16;
            break;
        case OperandType::TENSOR_QUANT16_ASYMM:
            NNRT_LOGE_PRINT("Ovxlib doesn't support quant16_asymm");
            dtype = VSI_NN_TYPE_UINT16;
            break;
        case OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL:
            NNRT_LOGE_PRINT("Ovxlib doesn't support quant8_symm_perchannel");
            dtype = VSI_NN_TYPE_INT8;
            break;
        default:
            NNRT_LOGE_PRINT("Unsupport data type %d", code);
            break;
    }
    return dtype;
}

int OvxlibDelegate::addNode(vsi_nn_op_t op,
        std::vector<uint32_t> & inputs,
        std::vector<uint32_t> & outputs, FusedType fused_code,
        std::vector<vsi_nn_node_t*>* output_nodes, uint32_t uid)
{
    int err = NNA_ERROR_CODE(NO_ERROR);
    if (nullptr == graph_)
    {
        return NNA_ERROR_CODE(BAD_DATA);
    }

    auto add_node = [output_nodes, &err](vsi_nn_graph_t* graph, vsi_nn_op_t op,
            uint32_t input_num, uint32_t output_num, uint32_t uid) -> vsi_nn_node_t* {

        vsi_nn_node_t* node = vsi_nn_AddNode(graph, op, input_num, output_num, nullptr);
        if (nullptr == node)
        {
            err = NNA_ERROR_CODE(BAD_DATA);
        }
        else
        {
            node->uid = uid;
        }
        if(nullptr != output_nodes)
        {
            output_nodes->push_back(node);
        }
        return node;
    };

    auto copy_tensor_attr = [](vsi_nn_tensor_attr_t* dst_attr,
            vsi_nn_tensor_attr_t* src_attr) {
        memset(dst_attr, 0, sizeof(vsi_nn_tensor_attr));
        dst_attr->dim_num = src_attr->dim_num;
        memcpy(&dst_attr->dtype, &src_attr->dtype, sizeof(vsi_nn_dtype_t));
        memcpy(dst_attr->size, src_attr->size, dst_attr->dim_num* sizeof(uint32_t));
    };

    auto set_tensors = [](vsi_nn_node_t* node,
            std::vector<vsi_nn_tensor_id_t> & input_tensors,
            std::vector<vsi_nn_tensor_id_t> & output_tensors) {
        if (!node)
        {
            return;
        }
        for (size_t i = 0; i < input_tensors.size(); ++ i)
        {
            node->input.tensors[i] = input_tensors[i];
        }
        for (size_t i = 0; i < output_tensors.size(); ++ i)
        {
            node->output.tensors[i] = output_tensors[i];
        }
    };

    std::vector<vsi_nn_tensor_id_t> input_tensors = getMappedTensors(inputs);
    std::vector<vsi_nn_tensor_id_t> output_tensors = getMappedTensors(outputs);

    if (hasFusedCode(fused_code))
    {
        assert(output_tensors.size() == 1);
        vsi_nn_tensor_t* tensor = vsi_nn_GetTensor(graph_, output_tensors[0]);
        if (nullptr == tensor)
        {
            NNRT_LOGE_PRINT("Tensor(%d) is missing.", output_tensors[0]);
            assert(false);
        }
        vsi_nn_tensor_attr_t attr;
        copy_tensor_attr(&attr, &tensor->attr);
        attr.vtl = (vsi_bool)true;
        std::vector<vsi_nn_tensor_id_t> node_out_tensors(1);
        node_out_tensors[0] = vsi_nn_AddTensor(graph_, VSI_NN_TENSOR_ID_AUTO, &attr, nullptr);

        vsi_nn_node_t* node = add_node(graph_, op, input_tensors.size(), 1, uid);
        set_tensors(node, input_tensors, node_out_tensors);

        vsi_nn_op_t activation_op = getActivation(fused_code);
        //uint32_t new_uid = activation_uid(uid);
        uint32_t new_uid = newNodeUid();
        node = add_node(graph_, activation_op, 1, 1, new_uid);
        set_tensors(node, node_out_tensors, output_tensors);
    }
    else
    {
        vsi_nn_node_t* node = add_node(
            graph_, op, input_tensors.size(), output_tensors.size(), uid);
        set_tensors(node, input_tensors, output_tensors);
    }

    return err;
}

std::vector<vsi_nn_tensor_id_t> OvxlibDelegate::getMappedTensors(
        std::vector<uint32_t> & operand_indexes)
{
    std::vector<vsi_nn_tensor_id_t> tensors;
    for (size_t i = 0; i < operand_indexes.size(); ++ i)
    {
        tensors.push_back(getMappedTensor(operand_indexes[i]));
    }
    return tensors;
}

vsi_nn_tensor_id_t OvxlibDelegate::getMappedTensor(uint32_t operand_index)
{
    if (tensor_map_.find(operand_index) == tensor_map_.end())
    {
        return VSI_NN_TENSOR_ID_NA;
    }
    return tensor_map_[operand_index];
}


void OvxlibDelegate::packTensorAttr
    (
    vsi_nn_tensor_attr_t* attr,
    vsi_nn_type_e dtype,
    std::vector<uint32_t> & nchw_shape,
    bool is_quantized,
    float scale,
    int32_t zero_point,
    TensorLifeTime type
    )
{
    if (nullptr == attr)
    {
        return;
    }
    memset(attr, 0, sizeof(vsi_nn_tensor_attr_t));

    /* reverse shape for driver rank: nchw --> whcn */
    std::vector<uint32_t> whcn_shape = reverseArray(nchw_shape);
    /* Pack shape */
    attr->dim_num = whcn_shape.size();
    for (size_t idx = 0; idx < attr->dim_num; idx ++)
    {
        attr->size[idx] = whcn_shape[idx];
        //attr->size[idx] = nchw_shape[idx];
    }

    /* Pack data type */
    attr->dtype.fmt = VSI_NN_DIM_FMT_NCHW;
    attr->dtype.vx_type = dtype;
    if (is_quantized)
    {
        assert(zero_point >= 0);
        attr->dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
        attr->dtype.zero_point = (uint32_t)zero_point;
        attr->dtype.scale = scale;
    }

    /* Pack tensor type */
    switch(type)
    {
        case TensorLifeTime::VIRTUAL:
            // For debug
            //attr->vtl = false;
            attr->vtl = true;
            break;
        case TensorLifeTime::CONST:
            attr->is_const = true;
            break;
        default:
            break;
    }
}

void OvxlibDelegate::packTensorAttr
    (
    vsi_nn_tensor_attr_t* attr,
    OperandPtr operand,
    TensorLifeTime type
    )
{
    vsi_nn_type_e dtype = mapTensorType(operand->type);
    bool is_quantized = operand->isQuantized();
    packTensorAttr(attr, dtype,
            operand->dimensions, is_quantized,
            operand->quant.scalar.scale, operand->quant.scalar.zeroPoint,
            type);
}

void OvxlibDelegate::mapTensorId(uint32_t operand_id,
        vsi_nn_tensor_id_t tensor_id)
{
    if (tensor_map_.find(operand_id) != tensor_map_.end())
    {
        NNRT_LOGW_PRINT("Operand id(%u) has been registered.", operand_id);
        assert(false);
    }
    tensor_map_[operand_id] = tensor_id;
}

int OvxlibDelegate::addTensor(vsi_nn_graph_t* graph,
    OperandPtr operand, TensorLifeTime type, size_t idx, const void* data)
{
    int err = NNA_ERROR_CODE(BAD_DATA);
    if (nullptr == graph)
    {
        return err;
    }
    std::vector<uint32_t> shape;
    if (operand->isTensor())
    {
        if (type != TensorLifeTime::VIRTUAL)
        {
            shape = operand->dimensions;
        }
        /* Pass 0 dim to tensor so that
         * ovxlib will compute shape automatically. */
        vsi_nn_type_e dtype = mapTensorType(operand->type);
        bool is_quantized = operand->isQuantized();
        err = addTensor(graph, dtype, shape, is_quantized,
                operand->quant.scalar.scale, operand->quant.scalar.zeroPoint, type, idx, data);
    }
    return err;
}

int OvxlibDelegate::addTensor
    (
    vsi_nn_graph_t* graph,
    vsi_nn_type_e dtype,
    std::vector<uint32_t> & shape,
    bool is_quantized,
    float scale, int32_t zero_point,
    TensorLifeTime type, size_t idx,
    const void* data
    )
{
    if (nullptr == graph)
    {
        return NNA_ERROR_CODE(BAD_DATA);
    }
    vsi_nn_tensor_attr_t attr;
    packTensorAttr(&attr, dtype, shape, is_quantized,
            scale, zero_point, type);
    return addTensor(graph, &attr, idx, data);
}

int OvxlibDelegate::addTensor
    (
    vsi_nn_graph_t* graph,
    vsi_nn_tensor_attr_t* attr, size_t idx,
    const void* data
    )
{
    int err = NNA_ERROR_CODE(BAD_DATA);
    if (nullptr == graph || nullptr == attr)
    {
        return err;
    }
    vsi_nn_tensor_id_t tid;
    void* nonconst_ptr = const_cast<void*>(data);
    tid = vsi_nn_AddTensor(graph, VSI_NN_TENSOR_ID_AUTO, attr, (uint8_t*)nonconst_ptr);
    // TODO: OVXLib need to support map loop
    //tid = vsi_nn_AddTensor(graph, (vsi_nn_tensor_id_t)idx, attr, (uint8_t*)data);
    if (VSI_NN_TENSOR_ID_NA == tid)
    {
        err = NNA_ERROR_CODE(BAD_DATA);
        NNRT_LOGW_PRINT("Add operand(%u) tensor fail.", idx);
        assert(false);
    }
    else
    {
        mapTensorId(idx, tid);
        err = NNA_ERROR_CODE(NO_ERROR);
    }
    return err;
}

std::vector<uint32_t> OvxlibDelegate::reorderOperands(
        std::vector<uint32_t>& operands, std::vector<int> order)
{
    std::vector<uint32_t> new_operands(operands.size());
    new_operands = operands;
    for (uint32_t i = 0; i < order.size(); ++ i) {
        if (order[i] >= (int)order.size()) {
            NNRT_LOGW_PRINT("Got incorrect index %d, max size is %lu", order[i], order.size());
            assert(false);
        }
        new_operands[i] = operands[order[i]];
    }
    return new_operands;
}

void OvxlibDelegate::fillVxParam(vsi_nn_vx_param_t* c_vx_param, VxParam& vx_param)
{
    switch (vx_param.overflowPolicy) {
        case OverflowPolicy::WRAP:
            c_vx_param->overflow_policy = VX_CONVERT_POLICY_WRAP;
            break;
        case OverflowPolicy::SATURATE:
            c_vx_param->overflow_policy = VX_CONVERT_POLICY_SATURATE;
            break;
        default:
            break;
    }
    switch (vx_param.roundingPolicy) {
        case RoundingPolicy::TO_ZERO:
            c_vx_param->rounding_policy = VX_ROUND_POLICY_TO_ZERO;
            break;
        case RoundingPolicy::RTNE:
            c_vx_param->rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
            break;
        default:
            break;
    }
    switch (vx_param.downScaleSizeRounding) {
        case Rounding::FLOOR:
            c_vx_param->down_scale_size_rounding = \
                    VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_FLOOR;
            break;
        case Rounding::CEILING:
            c_vx_param->down_scale_size_rounding = \
                    VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_CEILING;
            break;
        default:
            break;
    }
}

#define DECLARE_SAMPLE_OP(NAME, OVXLIB_OP)              \
    int OvxlibDelegate::addNode_##NAME(Model* model,    \
        OperationPtr operation, uint32_t operation_index) \
    {                                                   \
        (void)model;                                    \
        int err = NNA_ERROR_CODE(NO_ERROR);                \
        err = addNode(VSI_NN_OP_##OVXLIB_OP, operation, nullptr, operation_index); \
        return err;                                     \
    }

int OvxlibDelegate::addNode_ADD(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    err = addNode(VSI_NN_OP_ADD, operation, nullptr, operation_index);
    return err;
}

int OvxlibDelegate::addNode_CONCATENATION(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    ConcatOperation* concat =  reinterpret_cast<ConcatOperation*>(operation.get());
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_CONCAT, concat->inputs(), concat->outputs(),
            concat->fusedType(), &nodes, operation_index);
    std::vector<OperandPtr> outputs = model->getOperands(operation->outputs());
    int32_t dim = static_cast<int32_t>(outputs[0]->dimensions.size());
    nodes[0]->nn_param.concat.axis = static_cast<uint32_t>(convertAxis(concat->axis, dim));
    return err;
}

int OvxlibDelegate::addNode_CONV_2D(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    Conv2DOperation* conv2d = reinterpret_cast<Conv2DOperation*>(operation.get());
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    OperandPtr weight = inputs[1];

    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_CONV2D, conv2d->inputs(), conv2d->outputs(),
            conv2d->fusedType(), &nodes, operation_index);
    nodes[0]->nn_param.conv2d.pad_type = getPaddingType(conv2d->padType);
    nodes[0]->nn_param.conv2d.stride[0] = conv2d->strides[0];
    nodes[0]->nn_param.conv2d.stride[1] = conv2d->strides[1];
    nodes[0]->nn_param.conv2d.pad[0] = conv2d->pad[0];
    nodes[0]->nn_param.conv2d.pad[1] = conv2d->pad[1];
    nodes[0]->nn_param.conv2d.pad[2] = conv2d->pad[2];
    nodes[0]->nn_param.conv2d.pad[3] = conv2d->pad[3];
    nodes[0]->nn_param.conv2d.dilation[0] = conv2d->dilations[0];
    nodes[0]->nn_param.conv2d.dilation[1] = conv2d->dilations[1];
    nodes[0]->nn_param.conv2d.ksize[0] = weight->dimensions[3];
    nodes[0]->nn_param.conv2d.ksize[1] = weight->dimensions[2];
    nodes[0]->nn_param.conv2d.weights = weight->dimensions[0];
    fillVxParam(&nodes[0]->vx_param, conv2d->vxParam());
    return err;
}

int OvxlibDelegate::addNode_GROUPED_CONV_2D(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    GroupedConv2DOperation* conv2d = reinterpret_cast<GroupedConv2DOperation*>(operation.get());
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    OperandPtr weight = inputs[1];

    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_GROUPED_CONV2D, conv2d->inputs(), conv2d->outputs(),
            conv2d->fusedType(), &nodes, operation_index);
    nodes[0]->nn_param.conv2d.pad_type = getPaddingType(conv2d->padType);
    nodes[0]->nn_param.conv2d.stride[0] = conv2d->strides[0];
    nodes[0]->nn_param.conv2d.stride[1] = conv2d->strides[1];
    nodes[0]->nn_param.conv2d.pad[0] = conv2d->pad[0];
    nodes[0]->nn_param.conv2d.pad[1] = conv2d->pad[1];
    nodes[0]->nn_param.conv2d.pad[2] = conv2d->pad[2];
    nodes[0]->nn_param.conv2d.pad[3] = conv2d->pad[3];
    nodes[0]->nn_param.conv2d.dilation[0] = conv2d->dilations[0];
    nodes[0]->nn_param.conv2d.dilation[1] = conv2d->dilations[1];
    nodes[0]->nn_param.conv2d.ksize[0] = weight->dimensions[3];
    nodes[0]->nn_param.conv2d.ksize[1] = weight->dimensions[2];
    nodes[0]->nn_param.conv2d.weights = weight->dimensions[0];
    nodes[0]->nn_param.conv2d.group = conv2d->groups;
    fillVxParam(&nodes[0]->vx_param, conv2d->vxParam());
    return err;
}

int OvxlibDelegate::addNode_DEPTHWISE_CONV_2D(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    DepthwiseConv2DOperation* conv2d =  reinterpret_cast<DepthwiseConv2DOperation*>(operation.get());
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    OperandPtr weight = inputs[1];

    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_CONV2D, operation, &nodes, operation_index);
    nodes[0]->nn_param.conv2d.pad_type = getPaddingType(conv2d->padType);
    nodes[0]->nn_param.conv2d.stride[0] = conv2d->strides[0];
    nodes[0]->nn_param.conv2d.stride[1] = conv2d->strides[1];
    nodes[0]->nn_param.conv2d.pad[0] = conv2d->pad[0];
    nodes[0]->nn_param.conv2d.pad[1] = conv2d->pad[1];
    nodes[0]->nn_param.conv2d.pad[2] = conv2d->pad[2];
    nodes[0]->nn_param.conv2d.pad[3] = conv2d->pad[3];
    nodes[0]->nn_param.conv2d.multiplier = conv2d->multiplier;
    nodes[0]->nn_param.conv2d.dilation[0] = conv2d->dilations[0];
    nodes[0]->nn_param.conv2d.dilation[1] = conv2d->dilations[1];
    nodes[0]->nn_param.conv2d.ksize[0] = weight->dimensions[3];
    nodes[0]->nn_param.conv2d.ksize[1] = weight->dimensions[2];
    nodes[0]->nn_param.conv2d.weights = weight->dimensions[0] * weight->dimensions[1];
    fillVxParam(&nodes[0]->vx_param, conv2d->vxParam());
    return err;
}

int OvxlibDelegate::addNode_RELU(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    err = addNode(VSI_NN_OP_RELU, operation, nullptr, operation_index);
    return err;
}

int OvxlibDelegate::addNode_FULLY_CONNECTED(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    FullyConnectedOperation* fc =  reinterpret_cast<FullyConnectedOperation*>(operation.get());

    OperandPtr weight = inputs[1];

    uint32_t axis = 0;
    if (inputs[0]->dimensions.size() == 4)
    {
        axis = 2;
    }

    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_FCL, operation, &nodes, operation_index);
    nodes[0]->nn_param.fcl.axis = axis;
    nodes[0]->nn_param.fcl.weights = weight->dimensions[0];
    fillVxParam(&nodes[0]->vx_param, fc->vxParam());
    return err;
}

int OvxlibDelegate::addNode_RESHAPE(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    ReshapeOperation* reshape =  reinterpret_cast<ReshapeOperation*>(operation.get());
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_RESHAPE, operation, &nodes, operation_index);
    int32_t *shape = addParamPool(reshape->shape, true);
    nodes[0]->nn_param.reshape.size = reinterpret_cast<uint32_t*>(shape);
    nodes[0]->nn_param.reshape.dim_num = reshape->shape.size();
    return err;
}

int OvxlibDelegate::addNode_SOFTMAX(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    SoftmaxOperation* softmax =  reinterpret_cast<SoftmaxOperation*>(operation.get());
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_SOFTMAX, softmax->inputs(), softmax->outputs(),
            softmax->fusedType(), &nodes, operation_index);
    nodes[0]->nn_param.softmax.beta = softmax->beta;
    std::vector<OperandPtr> outputs = model->getOperands(operation->outputs());
    int32_t dim = static_cast<int32_t>(outputs[0]->dimensions.size());
    nodes[0]->nn_param.softmax.axis = static_cast<uint32_t>(convertAxis(softmax->axis, dim));
    return err;
}

int OvxlibDelegate::addNode_LOG_SOFTMAX(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    LogSoftmaxOperation* op =  reinterpret_cast<LogSoftmaxOperation*>(operation.get());
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_LOG_SOFTMAX, op->inputs(), op->outputs(),
            op->fusedType(), &nodes, operation_index);
    nodes[0]->nn_param.log_softmax.betaValue = op->beta;
    std::vector<OperandPtr> outputs = model->getOperands(operation->outputs());
    int32_t dim = static_cast<int32_t>(outputs[0]->dimensions.size());
    nodes[0]->nn_param.log_softmax.axis = static_cast<uint32_t>(convertAxis(op->axis, dim));
    return err;
}

int OvxlibDelegate::addNode_TRANSPOSE(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    PermuteOperation* permute =  reinterpret_cast<PermuteOperation*>(operation.get());
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_PERMUTE, permute->inputs(), permute->outputs(),
            permute->fusedType(), &nodes, operation_index);
    std::vector<int32_t> perm = convertPermute(permute->perm);
    int32_t *perm_buf = addParamPool(perm, false);
    nodes[0]->nn_param.permute.perm = reinterpret_cast<uint32_t*>(perm_buf);
    nodes[0]->nn_param.permute.dim_num = permute->perm.size();
    return err;
}

int OvxlibDelegate::addNode_AVERAGE_POOL_2D(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    AveragePool2DOperation* pool =  reinterpret_cast<AveragePool2DOperation*>(operation.get());
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_POOL, operation, &nodes, operation_index);
    nodes[0]->nn_param.pool.pad[0] = pool->pad[0];
    nodes[0]->nn_param.pool.pad[1] = pool->pad[1];
    nodes[0]->nn_param.pool.pad[2] = pool->pad[2];
    nodes[0]->nn_param.pool.pad[3] = pool->pad[3];
    nodes[0]->nn_param.pool.stride[0] = pool->strides[0];
    nodes[0]->nn_param.pool.stride[1] = pool->strides[1];
    nodes[0]->nn_param.pool.ksize[0] = pool->ksize[0];
    nodes[0]->nn_param.pool.ksize[1] = pool->ksize[1];
    nodes[0]->nn_param.pool.pad_type = getPaddingType(pool->padType);
    if (pool->poolMode == PoolMode::VALID) {
        nodes[0]->nn_param.pool.type = VX_CONVOLUTIONAL_NETWORK_POOLING_AVG_ANDROID;
    } else {
        nodes[0]->nn_param.pool.type = VX_CONVOLUTIONAL_NETWORK_POOLING_AVG;
    }
    nodes[0]->nn_param.pool.round_type = getRoundType(pool->roundType);
    fillVxParam(&nodes[0]->vx_param, pool->vxParam());
    return err;
}

int OvxlibDelegate::addNode_MAX_POOL_2D(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    MaxPool2DOperation* pool =  reinterpret_cast<MaxPool2DOperation*>(operation.get());
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_POOL, operation, &nodes, operation_index);
    nodes[0]->nn_param.pool.pad[0] = pool->pad[0];
    nodes[0]->nn_param.pool.pad[1] = pool->pad[1];
    nodes[0]->nn_param.pool.pad[2] = pool->pad[2];
    nodes[0]->nn_param.pool.pad[3] = pool->pad[3];
    nodes[0]->nn_param.pool.stride[0] = pool->strides[0];
    nodes[0]->nn_param.pool.stride[1] = pool->strides[1];
    nodes[0]->nn_param.pool.ksize[0] = pool->ksize[0];
    nodes[0]->nn_param.pool.ksize[1] = pool->ksize[1];
    nodes[0]->nn_param.pool.pad_type = getPaddingType(pool->padType);
    nodes[0]->nn_param.pool.type = VX_CONVOLUTIONAL_NETWORK_POOLING_MAX;
    nodes[0]->nn_param.pool.round_type = getRoundType(pool->roundType);
    fillVxParam(&nodes[0]->vx_param, pool->vxParam());
    return err;
}


int OvxlibDelegate::addNode_SQUEEZE(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    //SqueezeOperation* squeeze =  reinterpret_cast<SqueezeOperation*>(operation.get());
    std::vector<OperandPtr> outputs = model->getOperands(operation->outputs());
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_RESHAPE, operation, &nodes, operation_index);
    // TODO: add squeeze node
    nodes[0]->nn_param.reshape.size = (uint32_t*)outputs[0]->dimensions.data();
    nodes[0]->nn_param.reshape.dim_num = outputs[0]->dimensions.size();
    return err;
}

int OvxlibDelegate::addNode_DATA_CONVERT(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    err = addNode(VSI_NN_OP_DATACONVERT, operation, nullptr, operation_index);
    return err;
}

int OvxlibDelegate::addNode_PAD(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    PadOperation* pad =  reinterpret_cast<PadOperation*>(operation.get());
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_PAD, operation, &nodes, operation_index);
    int32_t *padFront_buf = addParamPool(pad->padFront, true);
    int32_t *padBack_buf = addParamPool(pad->padBack, true);
    nodes[0]->nn_param.pad.front_size = reinterpret_cast<uint32_t*>(padFront_buf);
    nodes[0]->nn_param.pad.back_size = reinterpret_cast<uint32_t*>(padBack_buf);
    nodes[0]->nn_param.pad.dim_num = static_cast<uint8_t>(pad->padFront.size());
    nodes[0]->nn_param.pad.const_val = static_cast<int32_t>(pad->padValue);
    nodes[0]->nn_param.pad.mode = mapPadMode(pad->padMode);
    return err;
}

int OvxlibDelegate::addNode_MUL(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    MulOperation* mul =  reinterpret_cast<MulOperation*>(operation.get());
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_MULTIPLY, operation, &nodes, operation_index);
    nodes[0]->nn_param.multiply.scale = 1.0f;
    fillVxParam(&nodes[0]->vx_param, mul->vxParam());
    return err;
}

int OvxlibDelegate::addNode_DIV(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    DivOperation* div =  reinterpret_cast<DivOperation*>(operation.get());
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_DIVIDE, operation, &nodes, operation_index);
    nodes[0]->nn_param.divide.scale = 1.0f;
    fillVxParam(&nodes[0]->vx_param, div->vxParam());
    return err;
}

int OvxlibDelegate::addNode_MEAN(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    ReduceMeanOperation* mean =  reinterpret_cast<ReduceMeanOperation*>(operation.get());
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_REDUCE, operation, &nodes, operation_index);
    std::vector<OperandPtr> inputs = model->getOperands(mean->inputs());
    std::vector<int32_t> axes = mean->axes;
    if (axes.size() == 0) {
        for (uint32_t i = 0; i < inputs[0]->ndim(); i ++ ) {
            axes.push_back(i);
        }
    }
    std::vector<int32_t> convert_axes = convertAxes(axes, inputs[0]->ndim());
    int32_t *axes_ptr = addParamPool(convert_axes, true);
    nodes[0]->nn_param.reduce.type = VSI_NN_REDUCE_MEAN;
    nodes[0]->nn_param.reduce.axis = reinterpret_cast<uint32_t*>(axes_ptr);
    nodes[0]->nn_param.reduce.axis_num = axes.size();
    nodes[0]->nn_param.reduce.keep_dim = (uint32_t)mean->keepDim;
    return err;
}

int OvxlibDelegate::addNode_TANH(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    TanhOperation* tanh =  reinterpret_cast<TanhOperation*>(operation.get());
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_TANH, operation, &nodes, operation_index);
    nodes[0]->nn_param.tanh.scale_a = tanh->scaleA;
    nodes[0]->nn_param.tanh.scale_b = tanh->scaleB;
    return err;
}

int OvxlibDelegate::addNode_LEAKY_RELU(Model* model,
                                       OperationPtr operation,
                                       uint32_t operation_index) {
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    LeakyReluOperation* leaky_relu = reinterpret_cast<LeakyReluOperation*>(operation.get());
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_LEAKY_RELU, operation, &nodes, operation_index);
    nodes[0]->nn_param.activation.leaky_ratio = leaky_relu->ratio;
    return err;
}

int OvxlibDelegate::addNode_SPACE_TO_DEPTH(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    SpaceToDepthOperation* sp_to_dp =  reinterpret_cast<SpaceToDepthOperation*>(operation.get());
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_SPACE2DEPTH, operation, &nodes, operation_index);
    nodes[0]->nn_param.space2depth.block_size[0] = sp_to_dp->blockSize[0];
    nodes[0]->nn_param.space2depth.block_size[1] = sp_to_dp->blockSize[1];
    return err;
}

int OvxlibDelegate::addNode_DEPTH_TO_SPACE(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    DepthToSpaceOperation* dp_to_sp =  reinterpret_cast<DepthToSpaceOperation*>(operation.get());
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_DEPTH2SPACE, operation, &nodes, operation_index);
    nodes[0]->nn_param.space2depth.block_size[0] = dp_to_sp->blockSize[0];
    nodes[0]->nn_param.space2depth.block_size[1] = dp_to_sp->blockSize[1];
    return err;
}

int OvxlibDelegate::addNode_BATCH_TO_SPACE_ND(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    BatchToSpaceNDOperation* bp_to_sp =  reinterpret_cast<BatchToSpaceNDOperation*>(operation.get());
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_BATCH2SPACE, operation, &nodes, operation_index);
    int32_t *blockSize_buf = addParamPool(bp_to_sp->blockSize, true);
    nodes[0]->nn_param.batch2space.block_size = reinterpret_cast<uint32_t*>(blockSize_buf);
    nodes[0]->nn_param.batch2space.block_size_num = bp_to_sp->blockSize.size();
    nodes[0]->nn_param.batch2space.crop[0] = bp_to_sp->cropStart[0];
    nodes[0]->nn_param.batch2space.crop[1] = bp_to_sp->cropEnd[0];
    nodes[0]->nn_param.batch2space.crop[2] = bp_to_sp->cropStart[1];
    nodes[0]->nn_param.batch2space.crop[3] = bp_to_sp->cropEnd[1];
    return err;
}

int OvxlibDelegate::addNode_SPACE_TO_BATCH_ND(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    SpaceToBatchNDOperation* sp_to_batch =  reinterpret_cast<SpaceToBatchNDOperation*>(operation.get());
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_SPACE2BATCH, operation, &nodes, operation_index);
    int32_t *blockSize_buf = addParamPool(sp_to_batch->blockSize, true);
    nodes[0]->nn_param.space2batch.block_size = reinterpret_cast<uint32_t*>(blockSize_buf);
    nodes[0]->nn_param.space2batch.block_size_num = sp_to_batch->blockSize.size();
    nodes[0]->nn_param.space2batch.pad[0] = sp_to_batch->padFront[1];
    nodes[0]->nn_param.space2batch.pad[1] = sp_to_batch->padBack[1];
    nodes[0]->nn_param.space2batch.pad[2] = sp_to_batch->padFront[0];
    nodes[0]->nn_param.space2batch.pad[3] = sp_to_batch->padBack[0];
    return err;
}

int OvxlibDelegate::addNode_RESIZE_BILINEAR(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    ResizeBilinearOperation* resize =  reinterpret_cast<ResizeBilinearOperation*>(operation.get());
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_RESIZE, operation, &nodes, operation_index);
    nodes[0]->nn_param.resize.type = VSI_NN_INTERPOLATION_BILINEAR;
    nodes[0]->nn_param.resize.size[0] = resize->outputWidth;
    nodes[0]->nn_param.resize.size[1] = resize->outputHeight;
    // TODO: Suppor factor
    return err;
}

int OvxlibDelegate::addNode_RESIZE_NEAREST(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    ResizeNearestNeighborOperation* resize =  reinterpret_cast<ResizeNearestNeighborOperation*>(operation.get());
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_RESIZE, operation, &nodes, operation_index);
    nodes[0]->nn_param.resize.type = VSI_NN_INTERPOLATION_NEAREST_NEIGHBOR;
    nodes[0]->nn_param.resize.size[0] = resize->outputWidth;
    nodes[0]->nn_param.resize.size[1] = resize->outputHeight;
    // TODO: Suppor factor
    return err;
}

int OvxlibDelegate::addNode_LOCAL_RESPONSE_NORM(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    LocalResponseNormOperation* lrn =  reinterpret_cast<LocalResponseNormOperation*>(operation.get());
    std::vector<vsi_nn_node_t*> nodes;
    // TF uses LRN2
    err = addNode(VSI_NN_OP_LRN2, operation, &nodes, operation_index);
    if(lrn->channelType == NormalizationAlgorithmChannel::Across) {
        nodes[0]->nn_param.lrn.type = VX_CONVOLUTIONAL_NETWORK_NORM_ACROSS_MAPS;
    } else {
        nodes[0]->nn_param.lrn.type = VX_CONVOLUTIONAL_NETWORK_NORM_SAME_MAP;
    }
    // TODO: may be need NormalizationAlgorithmMethod
    nodes[0]->nn_param.lrn.size = lrn->radius * 2 + 1;
    nodes[0]->nn_param.lrn.alpha = lrn->scale;
    nodes[0]->nn_param.lrn.beta = lrn->exponent;
    nodes[0]->nn_param.lrn.bias = lrn->bias;

    std::vector<OperandPtr> outputs = model->getOperands(operation->outputs());
    int32_t dim = static_cast<int32_t>(outputs[0]->dimensions.size());
    nodes[0]->nn_param.lrn.axis = static_cast<uint32_t>(convertAxis(lrn->axis, dim));
    return err;
}

int OvxlibDelegate::addNode_L2_NORM(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    L2NormOperation* l2norm =  reinterpret_cast<L2NormOperation*>(operation.get());
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_L2_NORMALIZE, operation, &nodes, operation_index);

    std::vector<OperandPtr> outputs = model->getOperands(operation->outputs());
    int32_t dim = static_cast<int32_t>(outputs[0]->dimensions.size());
    nodes[0]->nn_param.l2_normalize.axis = static_cast<uint32_t>(convertAxis(l2norm->axis, dim));
    return err;
}

int OvxlibDelegate::addNode_STRIDED_SLICE(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    StridedSliceOperation* stride_slice =  reinterpret_cast<StridedSliceOperation*>(operation.get());
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_STRIDED_SLICE, operation, &nodes, operation_index);
    int32_t *starts  = addParamPool(stride_slice->starts, true);
    int32_t *ends    = addParamPool(stride_slice->ends, true);
    int32_t *strides = addParamPool(stride_slice->strides, true);
    size_t dim_num = stride_slice->starts.size();
    nodes[0]->nn_param.strided_slice.begin_dims = starts;
    nodes[0]->nn_param.strided_slice.begin_dims_num = stride_slice->starts.size();
    nodes[0]->nn_param.strided_slice.end_dims = ends;
    nodes[0]->nn_param.strided_slice.end_dims_num = stride_slice->ends.size();
    nodes[0]->nn_param.strided_slice.stride_dims = strides;
    nodes[0]->nn_param.strided_slice.stride_dims_num = stride_slice->strides.size();
    nodes[0]->nn_param.strided_slice.begin_mask = reverseMask(stride_slice->beginMask, dim_num);
    nodes[0]->nn_param.strided_slice.end_mask = reverseMask(stride_slice->endMask, dim_num);
    nodes[0]->nn_param.strided_slice.shrink_axis_mask =
        reverseMask(stride_slice->shrinkAxisMask, dim_num);
    return err;
}

int OvxlibDelegate::addNode_SVDF(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    SvdfOperation* svdf =  reinterpret_cast<SvdfOperation*>(operation.get());
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());

    std::vector<uint32_t> new_inputs = reorderOperands(svdf->inputs(), {0, 4, 1, 2, 3});
    std::vector<uint32_t> new_outputs = reorderOperands(svdf->outputs(), {1, 0});

    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_SVDF, new_inputs, new_outputs,
            svdf->fusedType(), &nodes, operation_index);
    nodes[0]->nn_param.svdf.rank = svdf->rank;
    // This trait in NNAPI 1.1 spec is incorrect.
    nodes[0]->nn_param.svdf.num_units = int(inputs[1]->dimensions[0] / svdf->rank);
    return err;
}

int OvxlibDelegate::addNode_LSH_PROJECTION(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    LshProjectionOperation* lsh =  reinterpret_cast<LshProjectionOperation*>(operation.get());

    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_LSH_PROJECTION, operation, &nodes, operation_index);
    nodes[0]->nn_param.lsh_projection.type = mapLshProjectionType(lsh->type);

    return err;
}

int OvxlibDelegate::addNode_LSTM(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    LstmUnitOperation* lstm_unit =  reinterpret_cast<LstmUnitOperation*>(operation.get());
    std::vector<vsi_nn_node_t*> nodes;

    auto mapped_inputs = reorderOperands(lstm_unit->inputs(),
                        //0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 11 12  13, 14, 15, 16, 17, 18, 19
                        {0, 18, 19, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 });

    auto mapped_outputs = reorderOperands(lstm_unit->outputs(), {3, 1, 2, 0});

    err = addNode(VSI_NN_OP_LSTMUNIT, mapped_inputs, mapped_outputs,
        lstm_unit->fusedType(), &nodes, operation_index);

    nodes[0]->nn_param.lstmunit.cell_clip = lstm_unit->cellClip;
    nodes[0]->nn_param.lstmunit.proj_clip = lstm_unit->projClip;
    nodes[0]->nn_param.lstmunit.forget_bias = lstm_unit->forgetBias;
    nodes[0]->nn_param.lstmunit.activation = mapLstmUnitActivation(lstm_unit->activation);

    return err;
}

int OvxlibDelegate::addNode_L2_POOL_2D(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    L2Pool2DOperation* pool =  reinterpret_cast<L2Pool2DOperation*>(operation.get());

    std::vector<vsi_nn_node_t*> nodes;
    addNode(VSI_NN_OP_POOL, operation, &nodes, operation_index);

    nodes[0]->nn_param.pool.type = VX_CONVOLUTIONAL_NETWORK_POOLING_L2;
    nodes[0]->nn_param.pool.pad[0] = pool->pad[0];
    nodes[0]->nn_param.pool.pad[1] = pool->pad[1];
    nodes[0]->nn_param.pool.pad[2] = pool->pad[2];
    nodes[0]->nn_param.pool.pad[3] = pool->pad[3];
    nodes[0]->nn_param.pool.stride[0] = pool->strides[0];
    nodes[0]->nn_param.pool.stride[1] = pool->strides[1];
    nodes[0]->nn_param.pool.ksize[0] = pool->ksize[0];
    nodes[0]->nn_param.pool.ksize[1] = pool->ksize[1];
    nodes[0]->nn_param.pool.pad_type = getPaddingType(pool->padType);
    nodes[0]->nn_param.pool.round_type = getRoundType(pool->roundType);

    fillVxParam(&nodes[0]->vx_param, pool->vxParam());

    return err;
}

int OvxlibDelegate::addNode_RNN(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    RnnOperation* rnn =  reinterpret_cast<RnnOperation*>(operation.get());

    std::vector<vsi_nn_node_t*> nodes;
    addNode(VSI_NN_OP_RNN, operation, &nodes, operation_index);
    nodes[0]->nn_param.rnn.activation = rnn->activation;

    return err;
}

int OvxlibDelegate::addNode_BATCH_NORM(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);

    BatchNormalization* bm =  reinterpret_cast<BatchNormalization*>(operation.get());

    std::vector<vsi_nn_node_t*> nodes;
    addNode(VSI_NN_OP_BATCH_NORM, operation, &nodes, operation_index);
    nodes[0]->nn_param.batch_norm.eps = bm->eps;
    return err;
}

int OvxlibDelegate::addNode_DECONV_2D(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    Deconv2DOperation* transposedConv = reinterpret_cast<Deconv2DOperation*>(operation.get());
    OperandPtr kernel = model->getOperands(operation->inputs())[1];

    std::vector<vsi_nn_node_t*> nodes;
    addNode(VSI_NN_OP_DECONVOLUTION, operation, &nodes, operation_index);
    nodes[0]->nn_param.deconv.ksize[0] = kernel->dimensions[3];
    nodes[0]->nn_param.deconv.ksize[1] = kernel->dimensions[2];

    nodes[0]->nn_param.deconv.stride[0] = transposedConv->strides[0];
    nodes[0]->nn_param.deconv.stride[1] = transposedConv->strides[1];
    nodes[0]->nn_param.deconv.pad[0] = transposedConv->pad[0];
    nodes[0]->nn_param.deconv.pad[1] = transposedConv->pad[1];
    nodes[0]->nn_param.deconv.pad[2] = transposedConv->pad[2];
    nodes[0]->nn_param.deconv.pad[3] = transposedConv->pad[3];

    nodes[0]->nn_param.deconv.pad_type = getPaddingType(transposedConv->padType);
    nodes[0]->nn_param.deconv.weights = kernel->dimensions[0];
    nodes[0]->nn_param.deconv.group = 1;    // We don't have group yet

    return err;
}

int OvxlibDelegate::addNode_TOPK(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    TopkOperation* op = reinterpret_cast<TopkOperation*>(operation.get());
    std::vector<vsi_nn_node_t*> nodes;
    addNode(VSI_NN_OP_TOPK, operation, &nodes, operation_index);
    nodes[0]->nn_param.topk.k = op->k;
    return err;
}

int OvxlibDelegate::addNode_ARGMAX(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    ArgmaxOperation* op = reinterpret_cast<ArgmaxOperation*>(operation.get());
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    int32_t dim = static_cast<int32_t>(inputs[0]->ndim());

    std::vector<vsi_nn_node_t*> nodes;
    addNode(VSI_NN_OP_ARGMAX, operation, &nodes, operation_index);
    nodes[0]->nn_param.argmax.axis = static_cast<uint32_t>(convertAxis(op->axis, dim));
    return err;
}

int OvxlibDelegate::addNode_ARGMIN(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    ArgminOperation* op = reinterpret_cast<ArgminOperation*>(operation.get());
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    int32_t dim = static_cast<int32_t>(inputs[0]->ndim());

    std::vector<vsi_nn_node_t*> nodes;
    addNode(VSI_NN_OP_ARGMIN, operation, &nodes, operation_index);
    nodes[0]->nn_param.argmin.axis = static_cast<uint32_t>(convertAxis(op->axis, dim));
    return err;
}

int OvxlibDelegate::addNode_GATHER(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    GatherOperation* op = reinterpret_cast<GatherOperation*>(operation.get());
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    int32_t dim = static_cast<int32_t>(inputs[0]->ndim());

    std::vector<vsi_nn_node_t*> nodes;
    addNode(VSI_NN_OP_GATHER, operation, &nodes, operation_index);
    nodes[0]->nn_param.gather.axis = static_cast<uint32_t>(convertAxis(op->axis, dim));
    return err;
}

int OvxlibDelegate::addNode_CHANNEL_SHUFFLE(Model* model,
                                            OperationPtr operation,
                                            uint32_t operation_index) {
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    ChannelShuffleOperation* op = reinterpret_cast<ChannelShuffleOperation*>(operation.get());
    std::vector<vsi_nn_node_t*> nodes;
    addNode(VSI_NN_OP_SHUFFLECHANNEL, operation, &nodes, operation_index);

    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    int32_t dim = static_cast<int32_t>(inputs[0]->ndim());
    nodes[0]->nn_param.shufflechannel.axis = static_cast<uint32_t>(convertAxis(op->axis, dim));
    nodes[0]->nn_param.shufflechannel.group_number = op->groups;
    return err;
}

int OvxlibDelegate::addNode_SPLIT(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    SplitOperation* op = reinterpret_cast<SplitOperation*>(operation.get());
    std::vector<vsi_nn_node_t*> nodes;
    addNode(VSI_NN_OP_SPLIT, operation, &nodes, operation_index);
    // No need to set slice_number and slice parameters. Ovxlib will comute output shape related
    // to the number of output operands.
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    nodes[0]->nn_param.split.axis = static_cast<uint32_t>(convertAxis(op->axis,
                inputs[0]->ndim()));
    return err;
}

int OvxlibDelegate::addNode_INSTANCE_NORM(Model* model,
                                          OperationPtr operation,
                                          uint32_t operation_index) {
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    InstanceNormOperation* op = reinterpret_cast<InstanceNormOperation*>(operation.get());
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_INSTANCE_NORM, operation, &nodes, operation_index);
    std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
    nodes[0]->nn_param.instancenorm.eps = op->eps;
    return err;
}

int OvxlibDelegate::addNode_SLICE(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    SliceOperation* op =  reinterpret_cast<SliceOperation*>(operation.get());
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_SLICE, operation, &nodes, operation_index);
    int32_t *starts  = addParamPool(op->starts, true);
    int32_t *sizes    = addParamPool(op->sizes, true);
    nodes[0]->nn_param.slice.dims = op->starts.size();
    nodes[0]->nn_param.slice.start = reinterpret_cast<uint32_t*>(starts);
    nodes[0]->nn_param.slice.length = reinterpret_cast<uint32_t*>(sizes);
    return err;
}

int OvxlibDelegate::addNode_GENERATE_PROPOSALS(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    GenerateProposalsOperation* op =  reinterpret_cast<GenerateProposalsOperation*>(operation.get());
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_GENERATE_PROPOSALS, operation, &nodes, operation_index);
    nodes[0]->nn_param.generate_proposals.height_stride = op->ratio_h;
    nodes[0]->nn_param.generate_proposals.width_stride = op->ratio_w;
    nodes[0]->nn_param.generate_proposals.pre_nms_top_n = op->pre_nms_topn;
    nodes[0]->nn_param.generate_proposals.post_nms_top_n = op->post_nms_topn;
    nodes[0]->nn_param.generate_proposals.iou_threshold = op->iou_threshold;
    nodes[0]->nn_param.generate_proposals.min_size = op->min_size;
    return err;
}

int OvxlibDelegate::addNode_RANDOM_MULTINOMIAL(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    RandomMultinomialOperation* op =  reinterpret_cast<RandomMultinomialOperation*>(operation.get());
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_RANDOM_MULTINOMIAL, operation, &nodes, operation_index);
    nodes[0]->nn_param.random_multinomial.sample_num = op->sample_num;
    return err;
}

int OvxlibDelegate::addNode_DETECTION_POSTPROCESSING(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    DetectionPostprocessingOperation* op =
        reinterpret_cast<DetectionPostprocessingOperation*>(operation.get());
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_DETECTION_POSTPROCESS, operation, &nodes, operation_index);
    nodes[0]->nn_param.detection_postprocess.dy = op->dy;
    nodes[0]->nn_param.detection_postprocess.dx = op->dx;
    nodes[0]->nn_param.detection_postprocess.dh = op->dh;
    nodes[0]->nn_param.detection_postprocess.dw = op->dw;
    nodes[0]->nn_param.detection_postprocess.nms_type = op->nms_type;
    nodes[0]->nn_param.detection_postprocess.max_num_detections = op->max_num_detections;
    nodes[0]->nn_param.detection_postprocess.maximum_class_per_detection = op->maximum_class_per_detection;
    nodes[0]->nn_param.detection_postprocess.maximum_detection_per_class = op->maximum_detection_per_class;
    nodes[0]->nn_param.detection_postprocess.score_threshold = op->score_threshold;
    nodes[0]->nn_param.detection_postprocess.iou_threshold = op->iou_threshold;
    nodes[0]->nn_param.detection_postprocess.is_bg_in_label = op->is_bg_in_label;
    return err;
}

int OvxlibDelegate::addNode_TILE(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    TileOperation* op =  reinterpret_cast<TileOperation*>(operation.get());
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_TILE, operation, &nodes, operation_index);
    int32_t *multiples = addParamPool(op->multiples, true);
    nodes[0]->nn_param.tile.multiples = multiples;
    nodes[0]->nn_param.tile.multiples_num = op->multiples.size();
    return err;
}

int OvxlibDelegate::addNode_ROI_POOLING(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    ROIPoolingOperation* op = reinterpret_cast<ROIPoolingOperation*>(operation.get());
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_ROI_POOL, operation, &nodes, operation_index);
    nodes[0]->nn_param.roi_pool.type = VX_NN_POOLING_MAX;
    nodes[0]->nn_param.roi_pool.size[0] = op->width;
    nodes[0]->nn_param.roi_pool.size[1] = op->height;
    nodes[0]->nn_param.roi_pool.scale = op->height_ratio;
    return err;
}

int OvxlibDelegate::addNode_ROI_ALIGN(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    ROIAlignOperation* op = reinterpret_cast<ROIAlignOperation*>(operation.get());
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_ROI_ALIGN, operation, &nodes, operation_index);
    nodes[0]->nn_param.roi_align.output_height = op->width;
    nodes[0]->nn_param.roi_align.output_width = op->height;
    nodes[0]->nn_param.roi_align.height_ratio = op->height_ratio;
    nodes[0]->nn_param.roi_align.width_ratio = op->width_ratio;
    nodes[0]->nn_param.roi_align.height_sample_num = op->sampling_points_height;
    nodes[0]->nn_param.roi_align.width_sample_num = op->sampling_points_width;
    return err;
}

int OvxlibDelegate::addNode_BOX_WITH_NMS_LIMIT(Model* model,
        OperationPtr operation, uint32_t operation_index)
{
    (void)model;
    int err = NNA_ERROR_CODE(NO_ERROR);
    BoxWithNmsLimitOperation* op = reinterpret_cast<BoxWithNmsLimitOperation*>(operation.get());
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_BOX_WITH_NMS_LIMIT, operation, &nodes, operation_index);
    nodes[0]->nn_param.box_with_nms_limit.score_threshold = op->score_threshold;
    nodes[0]->nn_param.box_with_nms_limit.max_num_bbox = op->max_boxes;
    nodes[0]->nn_param.box_with_nms_limit.nms_kernel_method = static_cast<int32_t>(
            op->nms_kernel_method);
    nodes[0]->nn_param.box_with_nms_limit.iou_threshold = op->iou_threshold;
    nodes[0]->nn_param.box_with_nms_limit.sigma = op->nms_sigma;
    nodes[0]->nn_param.box_with_nms_limit.nms_score_threshold = op->nms_score_threshold;
    return err;
}

DECLARE_SAMPLE_OP(RELU1, RELU1)
DECLARE_SAMPLE_OP(RELU6, RELU6)
DECLARE_SAMPLE_OP(SIGMOID, SIGMOID)
DECLARE_SAMPLE_OP(SOFT_RELU, SOFTRELU)
DECLARE_SAMPLE_OP(SQRT, SQRT)
DECLARE_SAMPLE_OP(SQUARE, SQUARE)
DECLARE_SAMPLE_OP(FLOOR, FLOOR)
DECLARE_SAMPLE_OP(SUB, SUBTRACT)
DECLARE_SAMPLE_OP(DEQUANTIZE, DATACONVERT)
DECLARE_SAMPLE_OP(QUANTIZE, DATACONVERT)
DECLARE_SAMPLE_OP(HASHTABLE_LOOKUP, HASHTABLE_LOOKUP)
DECLARE_SAMPLE_OP(EMBEDDING_LOOKUP, EMBEDDING_LOOKUP)
DECLARE_SAMPLE_OP(MINIMUM, MINIMUM)
DECLARE_SAMPLE_OP(MAXIMUM, MAXIMUM)
DECLARE_SAMPLE_OP(RSQRT, RSQRT)
DECLARE_SAMPLE_OP(PRELU, PRELU)
DECLARE_SAMPLE_OP(ABS, ABS)
DECLARE_SAMPLE_OP(EXP, EXP)
DECLARE_SAMPLE_OP(NEG, NEG)
DECLARE_SAMPLE_OP(POW, POW)
DECLARE_SAMPLE_OP(LOG, LOG)
DECLARE_SAMPLE_OP(SELECT, SELECT)
DECLARE_SAMPLE_OP(SIN, SIN)
DECLARE_SAMPLE_OP(AXIS_ALIGNED_BBOX_TRANSFORM, AXIS_ALIGNED_BBOX_TRANSFORM)
DECLARE_SAMPLE_OP(HEATMAP_MAX_KEYPOINT, HEATMAP_MAX_KEYPOINT)
#undef DECLARE_SAMPLE_OP

#define DECLARE_RELATIONAL_OP(NAME, RELATIONAL_OP)              \
    int OvxlibDelegate::addNode_##NAME(Model* model,    \
        OperationPtr operation, uint32_t operation_index) \
    {                                                   \
        (void)model;                                    \
        std::vector<vsi_nn_node_t*> nodes;              \
        addNode(VSI_NN_OP_RELATIONAL_OPS, operation, &nodes, operation_index); \
        nodes[0]->nn_param.relational_ops.op = VSI_NN_RELATIONAL_OPS_##RELATIONAL_OP; \
        return NNA_ERROR_CODE(NO_ERROR);                     \
    }

DECLARE_RELATIONAL_OP(EQUAL, EQUAL)
DECLARE_RELATIONAL_OP(NOT_EQUAL, NOT_EQUAL)
DECLARE_RELATIONAL_OP(LESS, LESS)
DECLARE_RELATIONAL_OP(LESS_EQUAL, LESS_EQUAL)
DECLARE_RELATIONAL_OP(GREATER, GREAT)
DECLARE_RELATIONAL_OP(GREATER_EQUAL, GREAT_EQUAL)

#undef DECLARE_RELATIONAL_OP

#define DECLARE_LOGICAL_OP(NAME, LOGICAL_OP)              \
    int OvxlibDelegate::addNode_##NAME(Model* model,    \
        OperationPtr operation, uint32_t operation_index) \
    {                                                   \
        (void)model;                                    \
        std::vector<vsi_nn_node_t*> nodes;              \
        addNode(VSI_NN_OP_LOGICAL_OPS, operation, &nodes, operation_index); \
        nodes[0]->nn_param.relational_ops.op = VSI_NN_##LOGICAL_OP; \
        return NNA_ERROR_CODE(NO_ERROR);                     \
    }
DECLARE_LOGICAL_OP(LOGICAL_AND, LOGICAL_AND)
DECLARE_LOGICAL_OP(LOGICAL_OR, LOGICAL_OR)

#undef DECLARE_RELATIONAL_OP

#define DECLARE_REDUCTION_OP(NAME, REDUCTION_OP, OP_TYPE)    \
    int OvxlibDelegate::addNode_##NAME(Model* model,  \
            OperationPtr operation, uint32_t operation_index) { \
        (void)model;    \
        int err = NNA_ERROR_CODE(NO_ERROR); \
        OP_TYPE* mean =  reinterpret_cast<OP_TYPE*>(operation.get());   \
        std::vector<vsi_nn_node_t*> nodes;  \
        err = addNode(VSI_NN_OP_REDUCE, operation, &nodes, operation_index);    \
        std::vector<OperandPtr> inputs = model->getOperands(mean->inputs());    \
        std::vector<int32_t> axes = mean->axes; \
        if (axes.size() == 0) { \
            for (uint32_t i = 0; i < inputs[0]->ndim(); i ++ ) {    \
                axes.push_back(i);  \
            }   \
        }   \
        std::vector<int32_t> convert_axes = convertAxes(axes, inputs[0]->ndim());   \
        int32_t *axes_ptr = addParamPool(convert_axes, true);   \
        nodes[0]->nn_param.reduce.type = VSI_NN_##REDUCTION_OP;    \
        nodes[0]->nn_param.reduce.axis = reinterpret_cast<uint32_t*>(axes_ptr); \
        nodes[0]->nn_param.reduce.axis_num = axes.size();   \
        nodes[0]->nn_param.reduce.keep_dim = (uint32_t)mean->keepDim;   \
        return err; \
    }
DECLARE_REDUCTION_OP(REDUCE_ALL, REDUCE_ALL, ReduceAllOperation)
DECLARE_REDUCTION_OP(REDUCE_ANY, REDUCE_ANY, ReduceAnyOperation)
DECLARE_REDUCTION_OP(REDUCE_MAX, REDUCE_MAX, ReduceMaxOperation)
DECLARE_REDUCTION_OP(REDUCE_MIN, REDUCE_MIN, ReduceMinOperation)
DECLARE_REDUCTION_OP(REDUCE_PROD, REDUCE_PROD, ReduceProdOperation)
DECLARE_REDUCTION_OP(REDUCE_SUM, REDUCE_SUM, ReduceSumOperation)
#undef DECLARE_REDUCTION_OP

}
