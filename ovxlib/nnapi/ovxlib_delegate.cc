#include <assert.h>
#include <algorithm>
#include "model.h"
#include "error.h"
#include "ovxlib_delegate.h"
#include "vsi_nn_pub.h"

namespace ovxlib
{
namespace {
void removeOperationWithNullInput(Model* model)
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
            VSILOGD("Mark operand(%d) as NUll",  operation->output(i));
            operands[operation->output(i)]->setNull();
        }
        if (all_input_empty){
            VSILOGD("Operation[%d] planned to remove", op_itor->first);
            operation_reduce_list.push_back(op_itor->first);
        }
    }

    for (auto op_idx = operation_reduce_list.begin();
            op_idx != operation_reduce_list.end();
            ++ op_idx) {
        model->operations().erase(model->operations().find(*op_idx));
    }
}

vsi_nn_lstmunit_activation_e mapLstmUnitActivation(const FusedType& ftype)
{
    vsi_nn_lstmunit_activation_e rValue = VSI_NN_LSTMUNIT_ACT_NONE;

    switch (ftype){
    case FusedType::RELU1:
        VSILOGE("RELU1 Not supported, use RELU in case crash");
    case FusedType::RELU:
        rValue = VSI_NN_LSTMUNIT_ACT_RELU;
        break;
    case FusedType::RELU6:
        rValue = VSI_NN_LSTMUNIT_ACT_RELU6;
        break;
    case FusedType::TANH:
        rValue = VSI_NN_LSTMUNIT_ACT_TANH;
        break;
    case FusedType::SIGMOID:
        rValue = VSI_NN_LSTMUNIT_ACT_SIGMOID;
        break;
    default:
        VSILOGE("Not supported activation type for LSTM_Unit");
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
    REGISTER_OP(DIV);
    REGISTER_OP(SUB);
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
#undef REGISTER_OP
}

OvxlibDelegate::~OvxlibDelegate()
{

}

int OvxlibDelegate::process(Model* model, vsi_nn_context_t ctx)
{
    if (!model) {
        VSILOGW("Model is null.");
        return AERROR_CODE(UNEXPECTED_NULL);
    }
    int err = AERROR_CODE(NO_ERROR);
    if (nullptr == ctx)
    {
        ctx = vsi_nn_CreateContext();
        if (!ctx) {
            VSILOGW("Create context fail.");
            return AERROR_CODE(OUT_OF_MEMORY);
        }
    }
    graph_ = vsi_nn_CreateGraph(ctx, 0, 0);
    if (nullptr == graph_)
    {
        return AERROR_CODE(OUT_OF_MEMORY);
    }

    auto operands = model->operands();

    // remove nodes which all input are null and reset its sucessor node's input
    removeOperationWithNullInput(model);

    for(auto it = operands.begin(); it != operands.end(); ++ it)
    {
        int idx = it->first;
        Operand* operand = it->second;
        if(!operand->isTensor() || operand->isNull())
        {
            VSILOGD("Skip Operand[%d]", idx);
            // If current operand is the graphic input, we should remove it from input
            //auto& model_inputs_idx = model->inputIndexes();
            //auto is_input = std::find(model_inputs_idx.begin(), model_inputs_idx.end(), idx);
            //if (is_input != model_inputs_idx.end()) {
            //    // erase input from model if user set it 'no-value'
            //    model_inputs_idx.erase(is_input);
            //}

            continue;
        }
        //VSILOGD("Add tensor (%u)", idx);

        if (operand->isConst())
        {
            void* data = model->getBuffer<void>(operand->mem_ref);
            if (operand->type == OperandType::TENSOR_FLOAT16 &&
                    (operand->bytes() * 2) == operand->mem_ref->len_ ) {
                void* fp16_ptr = malloc(operand->bytes());
                if (nullptr == fp16_ptr) {
                    VSILOGE("Out of memory.");
                    return AERROR_CODE(OUT_OF_MEMORY);
                }
                vsi_nn_dtype_t fmt16;
                vsi_nn_dtype_t fmt32;
                fmt16.vx_type = VSI_NN_TYPE_FLOAT16;
                fmt32.vx_type = VSI_NN_TYPE_FLOAT32;
                vsi_nn_DtypeConvertRawData((uint8_t*)data, operand->mem_ref->len_,
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
        if (err != AERROR_CODE(NO_ERROR))
        {
            return err;
        }
    }
    for (auto it = tensor_map_.begin(); it != tensor_map_.end(); ++it)
    {
        Operand* operand = model->operand(it->first);
        if (!operand) {
            VSILOGW("Not operand found: %u, %u", it->first, it->second);
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
        VSILOGD("Add node %u(%d)", idx, op->type());
        int err = (this->*op_container_[op->type()])(model, op, idx);
        if (AERROR_CODE(NO_ERROR) != err)
        {
            VSILOGW("Build operation: %d, index: %d ", op->type(), idx);
            return err;
        }
    }
    vsi_nn_PrintGraph(graph_);
    if (VSI_FAILURE == vsi_nn_SetupGraph(graph_, true))
    {
        VSILOGW("Setup graph failure.");
        return AERROR_CODE(BAD_DATA);
    }
    VSILOGD("Verify graph ...");
    vsi_status status = vsi_nn_VerifyGraph(graph_);

    if (status != VSI_SUCCESS)
    {
        VSILOGW("Verify graph error: %d", status);
        return AERROR_CODE(BAD_DATA);
    }
    VSILOGD("Compile graph completed.");

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
            VSILOGE("Invalid padding type(%d)", type);
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
            VSILOGE("Invalid padding type(%d)", type);
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
        default:
            break;
    }
    return dtype;
}

int OvxlibDelegate::addNode(vsi_nn_op_t op,
        std::vector<uint32_t> & inputs,
        std::vector<uint32_t> & outputs, FusedType fused_code,
        std::vector<vsi_nn_node_t*>* output_nodes, uint32_t uid)
{
    int err = AERROR_CODE(NO_ERROR);
    if (nullptr == graph_)
    {
        return AERROR_CODE(BAD_DATA);
    }

    auto add_node = [output_nodes, &err](vsi_nn_graph_t* graph, vsi_nn_op_t op,
            uint32_t input_num, uint32_t output_num, uint32_t uid) -> vsi_nn_node_t* {
        vsi_nn_node_t* node = vsi_nn_AddNode(graph, op, input_num, output_num, nullptr);
        if (nullptr == node)
        {
            err = AERROR_CODE(BAD_DATA);
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
            VSILOGE("Tensor(%d) is missing.", output_tensors[0]);
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
        default:
            break;
    }
}

void OvxlibDelegate::packTensorAttr
    (
    vsi_nn_tensor_attr_t* attr,
    Operand* operand,
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
        VSILOGW("Operand id(%u) has been registered.", operand_id);
        assert(false);
    }
    tensor_map_[operand_id] = tensor_id;
}

int OvxlibDelegate::addTensor(vsi_nn_graph_t* graph,
    Operand* operand, TensorLifeTime type, size_t idx, const void* data)
{
    int err = AERROR_CODE(BAD_DATA);
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
        return AERROR_CODE(BAD_DATA);
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
    int err = AERROR_CODE(BAD_DATA);
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
        err = AERROR_CODE(BAD_DATA);
        VSILOGW("Add operand(%u) tensor fail.", idx);
        assert(false);
    }
    else
    {
        mapTensorId(idx, tid);
        err = AERROR_CODE(NO_ERROR);
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
            VSILOGW("Got incorrect index %d, max size is %lu", order[i], order.size());
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
        Operation* operation, uint32_t operation_index) \
    {                                                   \
        int err = AERROR_CODE(NO_ERROR);                \
        err = addNode(VSI_NN_OP_##OVXLIB_OP, operation, nullptr, operation_index); \
        return err;                                     \
    }

int OvxlibDelegate::addNode_ADD(Model* model,
        Operation* operation, uint32_t operation_index)
{
    int err = AERROR_CODE(NO_ERROR);
    err = addNode(VSI_NN_OP_ADD, operation, nullptr, operation_index);
    return err;
}

int OvxlibDelegate::addNode_CONCATENATION(Model* model,
        Operation* operation, uint32_t operation_index)
{
    int err = AERROR_CODE(NO_ERROR);
    ConcatOperation* concat = reinterpret_cast<ConcatOperation*>(operation);
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_CONCAT, concat->inputs(), concat->outputs(),
            concat->fusedType(), &nodes, operation_index);
    std::vector<Operand*> outputs = model->getOperands(operation->outputs());
    int32_t dim = static_cast<int32_t>(outputs[0]->dimensions.size());
    nodes[0]->nn_param.concat.axis = static_cast<uint32_t>(convertAxis(concat->axis, dim));
    return err;
}

int OvxlibDelegate::addNode_CONV_2D(Model* model,
        Operation* operation, uint32_t operation_index)
{
    int err = AERROR_CODE(NO_ERROR);
    Conv2DOperation* conv2d = reinterpret_cast<Conv2DOperation*>(operation);
    std::vector<Operand*> inputs = model->getOperands(operation->inputs());
    Operand* weight = inputs[1];

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

int OvxlibDelegate::addNode_DEPTHWISE_CONV_2D(Model* model,
        Operation* operation, uint32_t operation_index)
{
    int err = AERROR_CODE(NO_ERROR);
    DepthwiseConv2DOperation* conv2d = reinterpret_cast<DepthwiseConv2DOperation*>(operation);
    std::vector<Operand*> inputs = model->getOperands(operation->inputs());
    Operand* weight = inputs[1];

    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_CONV2D, conv2d, &nodes, operation_index);
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
        Operation* operation, uint32_t operation_index)
{
    int err = AERROR_CODE(NO_ERROR);
    err = addNode(VSI_NN_OP_RELU, operation, nullptr, operation_index);
    return err;
}

int OvxlibDelegate::addNode_FULLY_CONNECTED(Model* model,
        Operation* operation, uint32_t operation_index)
{
    int err = AERROR_CODE(NO_ERROR);
    std::vector<Operand*> inputs = model->getOperands(operation->inputs());
    FullyConnectedOperation* fc = reinterpret_cast<FullyConnectedOperation*>(operation);

    Operand* weight = inputs[1];

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
        Operation* operation, uint32_t operation_index)
{
    int err = AERROR_CODE(NO_ERROR);
    ReshapeOperation* reshape = reinterpret_cast<ReshapeOperation*>(operation);
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_RESHAPE, operation, &nodes, operation_index);
    int32_t *shape = addParamPool(reshape->shape, true);
    nodes[0]->nn_param.reshape.size = reinterpret_cast<uint32_t*>(shape);
    nodes[0]->nn_param.reshape.dim_num = reshape->shape.size();
    return err;
}

int OvxlibDelegate::addNode_SOFTMAX(Model* model,
        Operation* operation, uint32_t operation_index)
{
    int err = AERROR_CODE(NO_ERROR);
    SoftmaxOperation* softmax = reinterpret_cast<SoftmaxOperation*>(operation);
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_SOFTMAX, softmax->inputs(), softmax->outputs(),
            softmax->fusedType(), &nodes, operation_index);
    nodes[0]->nn_param.softmax.beta = softmax->beta;
    nodes[0]->nn_param.softmax.axis = -1; // set a default value to sdk
    return err;
}

int OvxlibDelegate::addNode_TRANSPOSE(Model* model,
        Operation* operation, uint32_t operation_index)
{
    int err = AERROR_CODE(NO_ERROR);
    PermuteOperation* permute = reinterpret_cast<PermuteOperation*>(operation);
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
        Operation* operation, uint32_t operation_index)
{
    int err = AERROR_CODE(NO_ERROR);
    AveragePool2DOperation* pool = reinterpret_cast<AveragePool2DOperation*>(operation);
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
        Operation* operation, uint32_t operation_index)
{
    int err = AERROR_CODE(NO_ERROR);
    MaxPool2DOperation* pool = reinterpret_cast<MaxPool2DOperation*>(operation);
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
        Operation* operation, uint32_t operation_index)
{
    int err = AERROR_CODE(NO_ERROR);
    //SqueezeOperation* squeeze = reinterpret_cast<SqueezeOperation*>(operation);
    std::vector<Operand*> outputs = model->getOperands(operation->outputs());
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_RESHAPE, operation, &nodes, operation_index);
    // TODO: add squeeze node
    nodes[0]->nn_param.reshape.size = (uint32_t*)outputs[0]->dimensions.data();
    nodes[0]->nn_param.reshape.dim_num = outputs[0]->dimensions.size();
    return err;
}

int OvxlibDelegate::addNode_DATA_CONVERT(Model* model,
        Operation* operation, uint32_t operation_index)
{
    int err = AERROR_CODE(NO_ERROR);
    err = addNode(VSI_NN_OP_DATACONVERT, operation, nullptr, operation_index);
    return err;
}

int OvxlibDelegate::addNode_PAD(Model* model,
        Operation* operation, uint32_t operation_index)
{
    int err = AERROR_CODE(NO_ERROR);
    PadOperation* pad = reinterpret_cast<PadOperation*>(operation);
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
        Operation* operation, uint32_t operation_index)
{
    int err = AERROR_CODE(NO_ERROR);
    MulOperation* mul = reinterpret_cast<MulOperation*>(operation);
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_MULTIPLY, operation, &nodes, operation_index);
    nodes[0]->nn_param.multiply.scale = 1.0f;
    fillVxParam(&nodes[0]->vx_param, mul->vxParam());
    return err;
}

int OvxlibDelegate::addNode_DIV(Model* model,
        Operation* operation, uint32_t operation_index)
{
    int err = AERROR_CODE(NO_ERROR);
    DivOperation* div = reinterpret_cast<DivOperation*>(operation);
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_DIVIDE, operation, &nodes, operation_index);
    nodes[0]->nn_param.divide.scale = 1.0f;
    fillVxParam(&nodes[0]->vx_param, div->vxParam());
    return err;
}

int OvxlibDelegate::addNode_MEAN(Model* model,
        Operation* operation, uint32_t operation_index)
{
    int err = AERROR_CODE(NO_ERROR);
    MeanOperation* mean = reinterpret_cast<MeanOperation*>(operation);
    assert(mean->axes.size() > 0);
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_REDUCE, operation, &nodes, operation_index);
    std::vector<Operand*> inputs = model->getOperands(mean->inputs());
    std::vector<int32_t> convert_axes = convertAxes(mean->axes, inputs[0]->ndim());
    int32_t *axes = addParamPool(convert_axes, true);
    nodes[0]->nn_param.reduce.type = VSI_NN_REDUCE_MEAN;
    nodes[0]->nn_param.reduce.axis = reinterpret_cast<uint32_t*>(axes);
    nodes[0]->nn_param.reduce.axis_num = mean->axes.size();
    nodes[0]->nn_param.reduce.keep_dim = (uint32_t)mean->keepDim;
    return err;
}

int OvxlibDelegate::addNode_TANH(Model* model,
        Operation* operation, uint32_t operation_index)
{
    int err = AERROR_CODE(NO_ERROR);
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_TANH, operation, &nodes, operation_index);
    nodes[0]->nn_param.tanh.scale_a = 1.0f;
    nodes[0]->nn_param.tanh.scale_b = 1.0f;
    return err;
}

int OvxlibDelegate::addNode_SPACE_TO_DEPTH(Model* model,
        Operation* operation, uint32_t operation_index)
{
    int err = AERROR_CODE(NO_ERROR);
    SpaceToDepthOperation* sp_to_dp = reinterpret_cast<SpaceToDepthOperation*>(operation);
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_SPACE2DEPTH, operation, &nodes, operation_index);
    nodes[0]->nn_param.space2depth.block_size[0] = sp_to_dp->blockSize[0];
    nodes[0]->nn_param.space2depth.block_size[1] = sp_to_dp->blockSize[1];
    return err;
}

int OvxlibDelegate::addNode_DEPTH_TO_SPACE(Model* model,
        Operation* operation, uint32_t operation_index)
{
    int err = AERROR_CODE(NO_ERROR);
    DepthToSpaceOperation* dp_to_sp = reinterpret_cast<DepthToSpaceOperation*>(operation);
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_DEPTH2SPACE, operation, &nodes, operation_index);
    nodes[0]->nn_param.space2depth.block_size[0] = dp_to_sp->blockSize[0];
    nodes[0]->nn_param.space2depth.block_size[1] = dp_to_sp->blockSize[1];
    return err;
}

int OvxlibDelegate::addNode_BATCH_TO_SPACE_ND(Model* model,
        Operation* operation, uint32_t operation_index)
{
    int err = AERROR_CODE(NO_ERROR);
    BatchToSpaceNDOperation* bp_to_sp = reinterpret_cast<BatchToSpaceNDOperation*>(operation);
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
        Operation* operation, uint32_t operation_index)
{
    int err = AERROR_CODE(NO_ERROR);
    SpaceToBatchNDOperation* sp_to_batch = reinterpret_cast<SpaceToBatchNDOperation*>(operation);
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
        Operation* operation, uint32_t operation_index)
{
    int err = AERROR_CODE(NO_ERROR);
    ResizeBilinearOperation* resize = reinterpret_cast<ResizeBilinearOperation*>(operation);
    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_RESIZE, operation, &nodes, operation_index);
    nodes[0]->nn_param.resize.type = VSI_NN_INTERPOLATION_BILINEAR;
    nodes[0]->nn_param.resize.size[0] = resize->outputWidth;
    nodes[0]->nn_param.resize.size[1] = resize->outputHeight;
    return err;
}

int OvxlibDelegate::addNode_LOCAL_RESPONSE_NORM(Model* model,
        Operation* operation, uint32_t operation_index)
{
    int err = AERROR_CODE(NO_ERROR);
    LocalResponseNormOperation* lrn = reinterpret_cast<LocalResponseNormOperation*>(operation);
    std::vector<vsi_nn_node_t*> nodes;
    // TF uses LRN2
    err = addNode(VSI_NN_OP_LRN2, operation, &nodes, operation_index);
    nodes[0]->nn_param.lrn.type = VX_CONVOLUTIONAL_NETWORK_NORM_ACROSS_MAPS;
    nodes[0]->nn_param.lrn.size = lrn->radius * 2 + 1;
    nodes[0]->nn_param.lrn.alpha = lrn->scale;
    nodes[0]->nn_param.lrn.beta = lrn->exponent;
    nodes[0]->nn_param.lrn.bias = lrn->bias;
    return err;
}

int OvxlibDelegate::addNode_STRIDED_SLICE(Model* model,
        Operation* operation, uint32_t operation_index)
{
    int err = AERROR_CODE(NO_ERROR);
    StridedSliceOperation* stride_slice = reinterpret_cast<StridedSliceOperation*>(operation);
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
        Operation* operation, uint32_t operation_index)
{
    int err = AERROR_CODE(NO_ERROR);
    SvdfOperation* svdf = reinterpret_cast<SvdfOperation*>(operation);
    std::vector<Operand*> inputs = model->getOperands(operation->inputs());

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
        Operation* operation, uint32_t operation_index)
{
    int err = AERROR_CODE(NO_ERROR);
    LshProjectionOperation* lsh = reinterpret_cast<LshProjectionOperation*>(operation);

    std::vector<vsi_nn_node_t*> nodes;
    err = addNode(VSI_NN_OP_LSH_PROJECTION, operation, &nodes, operation_index);
    nodes[0]->nn_param.lsh_projection.type = mapLshProjectionType(lsh->type);

    return err;
}

int OvxlibDelegate::addNode_LSTM(Model* model,
        Operation* operation, uint32_t operation_index)
{
    int err = AERROR_CODE(NO_ERROR);
    LstmUnitOperation* lstm_unit = reinterpret_cast<LstmUnitOperation*>(operation);
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
        Operation* operation, uint32_t operation_index)
{
    int err = AERROR_CODE(NO_ERROR);
    L2Pool2DOperation* pool = reinterpret_cast<L2Pool2DOperation*>(operation);

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
        Operation* operation, uint32_t operation_index)
{
    int err = AERROR_CODE(NO_ERROR);
    RnnOperation* rnn = reinterpret_cast<RnnOperation*>(operation);

    std::vector<vsi_nn_node_t*> nodes;
    addNode(VSI_NN_OP_RNN, operation, &nodes, operation_index);
    nodes[0]->nn_param.rnn.activation = rnn->activation;

    return err;
}


DECLARE_SAMPLE_OP(RELU1, RELU1)
DECLARE_SAMPLE_OP(RELU6, RELU6)
DECLARE_SAMPLE_OP(SIGMOID, SIGMOID)
DECLARE_SAMPLE_OP(FLOOR, FLOOR)
DECLARE_SAMPLE_OP(SUB, SUBTRACT)
DECLARE_SAMPLE_OP(DEQUANTIZE, DATACONVERT)
DECLARE_SAMPLE_OP(L2_NORM, L2_NORMALIZE)
DECLARE_SAMPLE_OP(HASHTABLE_LOOKUP, HASHTABLE_LOOKUP)
DECLARE_SAMPLE_OP(EMBEDDING_LOOKUP, EMBEDDING_LOOKUP)
#undef DECLARE_SAMPLE_OP


}
