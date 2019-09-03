#include <vector>
#include <assert.h>
#include "model.h"
#include "error.h"
#include "operand.h"
#include "operation.h"
#include "graph_transformations/transformations.h"

namespace ovxlib
{
    static void insert_reshape(Model * model, Operand * operand,
        Operation * operation, std::vector<uint32_t> & new_dimension)
    {
        assert(model != nullptr && operand != nullptr && operation != nullptr);
        assert(new_dimension.size() > 0);

        int org_operand_index = model->getOperandIndex(operand);
        if (org_operand_index < 0)
        {
            assert(false);
        }

        // New Operand with new dimensions
        int new_operand_index = -1;
        Operand * new_operand = model->cloneOperand(operand, &new_operand_index);
        new_operand->dimensions = new_dimension;

        // new_dimension Operand
        int new_dimension_index = -1;
        Operand * new_dimension_operand = model->addOperand(nullptr, &new_dimension_index);
        new_dimension_operand->type = OperandType::TENSOR_INT32;
        new_dimension_operand->dimensions.push_back(new_dimension.size());
        model->setOperandValue(new_dimension_index, new_dimension.data(), sizeof(uint32_t) * new_dimension.size());

        // New Operation
        uint32_t inputs[1] = { (uint32_t)org_operand_index };
        uint32_t outputs[1] = { (uint32_t)new_operand_index };
        int reshape_index = -1;
        ReshapeOperation* reshape = new ReshapeOperation();
        reshape->shape.assign(new_dimension.begin(), new_dimension.end());
        reshape->setInputs(inputs, 1);
        reshape->setOutputs(outputs, 1);
        model->addOperation(reshape, &reshape_index);

        // update current operation
        operation->replaceInputs(org_operand_index, new_operand_index);
    }

    void fill_broadcast_operations(Model * model, Operation* operation,
        std::vector<Operand*> inputs)
    {
        int32_t dim = abs((int32_t)inputs[0]->ndim() - (int32_t)inputs[1]->ndim());
        if (dim == 0) {
            return;
        }

        uint32_t small_input_idx = inputs[1]->ndim() > inputs[0]->ndim() ? 0 : 1;
        Operand* operand = inputs[small_input_idx];

        if (operand->isConst()) {
            for (int32_t i = 0; i < dim; ++i) {
                operand->dimensions.insert(operand->dimensions.begin(), 1);
            }
        }
        else {
            // Add reshape layer
            std::vector<uint32_t> align_dimensions(operand->dimensions.begin(), operand->dimensions.end());
            for (int32_t i = 0; i < dim; ++i) {
                align_dimensions.insert(align_dimensions.begin(), 1);
            }
            insert_reshape(model, operand, operation, align_dimensions);
        }
    }

    int AlignBroadcastOp::run(Model * model, bool * modified)
    {
        auto operations = model->operations();
        for (auto it = operations.begin(); it != operations.end(); ++it)
        {
            Operation * operation = it->second;
            std::vector<Operand*> inputs = model->getOperands(operation->inputs());
            switch (operation->type())
            {
            case OperationType::MUL:
            case OperationType::ADD:
            case OperationType::DIV:
            case OperationType::SUB:
            {
                fill_broadcast_operations(model, operation, inputs);
                break;
            }
            default:
                break;
            }
        }

        return AERROR_CODE(NO_ERROR);
    }
}