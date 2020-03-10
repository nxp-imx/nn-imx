#include <set>
#include <vector>
#include "error.hpp"
#include "model.hpp"
#include "model_transform/transformations.hpp"
#include "op/public.hpp"
using namespace nnrt::op;

namespace nnrt {
namespace {

static constexpr uint32_t MARK_DATA_DIRECTION_CONSUMER = 0;
static constexpr uint32_t MARK_DATA_DIRECTION_PRODUCER = 1;
}

int Fp32ToFp16::run(Model* model, bool* modified) {
    *modified = true;
    if (nullptr == model) {
        return NNA_ERROR_CODE(NO_ERROR);
    }

    auto insert_data_convert_node = [model](
        OperandPtr operand, OperationPtr operation, OperandType dst_type, uint32_t direction) {
        /*
         *  Operation(producer) --> Operand
         *  Operand --> Operation(consumer)
         *
         */
        int org_operand_index = model->getOperandIndex(operand);
        int new_operand_index = -1;
        OperandPtr new_operand = model->cloneOperand(operand, &new_operand_index);
        new_operand->type = dst_type;
        uint32_t inputs[1] = {0};
        uint32_t outputs[1] = {0};
        if (MARK_DATA_DIRECTION_PRODUCER == direction) {
            inputs[0] = new_operand_index;
            outputs[0] = org_operand_index;
            operation->replaceOutputs(org_operand_index, new_operand_index);
        } else {
            inputs[0] = org_operand_index;
            outputs[0] = new_operand_index;
            operation->replaceInputs(org_operand_index, new_operand_index);
        }
        model->addOperation(OperationType::DATA_CONVERT, inputs, 1, outputs, 1);
    };

    auto insert_data_convert_nodes = [model, insert_data_convert_node, modified](
        OperandPtr operand, OperandType dst_type, uint32_t direction) {
        if (operand->type != OperandType::TENSOR_FLOAT32) {
            return;
        }
        std::vector<uint32_t> operation_indexes;
        if (MARK_DATA_DIRECTION_CONSUMER == direction) {
            operation_indexes = model->getConsumers(operand);
        } else {
            operation_indexes = model->getProducers(operand);
        }
        for (auto operation_index : operation_indexes) {
            *modified = true;
            insert_data_convert_node(operand, model->operation(operation_index), dst_type, direction);
        }
    };

    // Mark half operands.
    std::set<uint32_t> fp16_operands;

    auto mark_operand_float_to_half = [model, &fp16_operands](uint32_t operand_index) {
        OperandPtr operand = model->operand(operand_index);
        if (nullptr != operand && operand->type == OperandType::TENSOR_FLOAT32) {
            fp16_operands.emplace(operand_index);
        }
    };

    auto mark_input_output_float_to_half = [mark_operand_float_to_half](OperationPtr operation) {
        for (uint32_t operand_index : operation->inputs()) {
            mark_operand_float_to_half(operand_index);
        }
        for (uint32_t operand_index : operation->outputs()) {
            mark_operand_float_to_half(operand_index);
        }
    };

    auto operations = model->operations();
    for (auto it = operations.begin(); it != operations.end(); ++it) {
        OperationPtr operation = it->second;
        switch (operation->type()) {
            case OperationType::CONV_2D:
            case OperationType::GROUPED_CONV_2D:
            case OperationType::DECONV_2D:
            case OperationType::DEPTHWISE_CONV_2D:
            case OperationType::FULLY_CONNECTED: {
                mark_operand_float_to_half(operation->input(0));
                mark_operand_float_to_half(operation->input(1));
                mark_operand_float_to_half(operation->output(0));
            } break;
            case OperationType::LSTM_UNIT: {
                mark_operand_float_to_half(operation->input(0));
                mark_operand_float_to_half(operation->input(1));
                mark_operand_float_to_half(operation->input(2));
                mark_operand_float_to_half(operation->input(3));
                mark_operand_float_to_half(operation->input(4));
                mark_operand_float_to_half(operation->input(5));
                mark_operand_float_to_half(operation->input(6));
                mark_operand_float_to_half(operation->input(7));
                mark_operand_float_to_half(operation->input(8));
                mark_operand_float_to_half(operation->input(9));
                mark_operand_float_to_half(operation->input(10));
                mark_operand_float_to_half(operation->input(11));
                mark_operand_float_to_half(operation->input(16));
                mark_operand_float_to_half(operation->input(18));
                mark_operand_float_to_half(operation->input(19));
                mark_operand_float_to_half(operation->output(0));
                mark_operand_float_to_half(operation->output(1));
                mark_operand_float_to_half(operation->output(2));
                mark_operand_float_to_half(operation->output(3));
            } break;
            case OperationType::UNIDIRECTIONAL_SEQUENCE_LSTM: {
                mark_operand_float_to_half(operation->input(0));
                mark_operand_float_to_half(operation->input(1));
                mark_operand_float_to_half(operation->input(2));
                mark_operand_float_to_half(operation->input(3));
                mark_operand_float_to_half(operation->input(4));
                mark_operand_float_to_half(operation->input(5));
                mark_operand_float_to_half(operation->input(6));
                mark_operand_float_to_half(operation->input(7));
                mark_operand_float_to_half(operation->input(8));
                mark_operand_float_to_half(operation->input(9));
                mark_operand_float_to_half(operation->input(10));
                mark_operand_float_to_half(operation->input(11));
                mark_operand_float_to_half(operation->input(16));
                mark_operand_float_to_half(operation->input(18));
                mark_operand_float_to_half(operation->input(19));
                mark_operand_float_to_half(operation->output(0));
            } break;
            case OperationType::LSH_PROJECTION:
                break;
            default:
                mark_input_output_float_to_half(operation);
                break;
        }
    }

    for (uint32_t operand_index : fp16_operands) {
        OperandPtr operand = model->operand(operand_index);
        if (operand->isConst()) {
            operand->type = OperandType::TENSOR_FLOAT16;
        } else if (model->isInput(operand_index)) {
            insert_data_convert_nodes(operand, OperandType::TENSOR_FLOAT16, MARK_DATA_DIRECTION_CONSUMER);
        } else if (model->isOutput(operand_index)) {
            insert_data_convert_nodes(operand, OperandType::TENSOR_FLOAT16, MARK_DATA_DIRECTION_PRODUCER);
        } else {
            operand->type = OperandType::TENSOR_FLOAT16;
        }
    }
    return NNA_ERROR_CODE(NO_ERROR);
}
}
