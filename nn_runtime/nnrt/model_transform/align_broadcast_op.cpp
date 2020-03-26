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
#include <vector>
#include <cassert>
#include "nnrt/model.hpp"
#include "nnrt/error.hpp"
#include "nnrt/op/public.hpp"
#include "nnrt/model_transform/transformations.hpp"

using namespace nnrt::op;

namespace nnrt
{
    static void insert_reshape(Model * model, OperandPtr operand,
        OperationPtr operation, std::vector<uint32_t> & new_dimension)
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
        OperandPtr new_operand = model->cloneOperand(operand, &new_operand_index);
        new_operand->dimensions = new_dimension;

        // new_dimension Operand
        uint32_t new_dimension_index = -1;
        OperandPtr new_dimension_operand = model->addOperand(nullptr, &new_dimension_index);
        new_dimension_operand->type = OperandType::TENSOR_INT32;
        new_dimension_operand->dimensions.push_back(new_dimension.size());
        model->setOperandValue(new_dimension_index, new_dimension.data(), sizeof(uint32_t) * new_dimension.size());

        // New Operation
        uint32_t inputs[1] = { (uint32_t)org_operand_index };
        uint32_t outputs[1] = { (uint32_t)new_operand_index };
        uint32_t reshape_index = -1;
        std::shared_ptr<ReshapeOperation> reshape = std::make_shared<ReshapeOperation>();
        reshape->shape.assign(new_dimension.begin(), new_dimension.end());
        reshape->setInputs(inputs, 1);
        reshape->setOutputs(outputs, 1);
        model->addOperation(std::dynamic_pointer_cast<Operation>(reshape), &reshape_index);

        // update current operation
        operation->replaceInputs(org_operand_index, new_operand_index);
    }

    void fill_broadcast_operations(Model * model, OperationPtr operation,
        std::vector<OperandPtr> inputs)
    {
        int32_t dim = abs((int32_t)inputs[0]->ndim() - (int32_t)inputs[1]->ndim());
        if (dim == 0) {
            return;
        }

        uint32_t small_input_idx = inputs[1]->ndim() > inputs[0]->ndim() ? 0 : 1;
        OperandPtr operand = inputs[small_input_idx];

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
        (void)modified;
        auto operations = model->operations();
        for (auto it = operations.begin(); it != operations.end(); ++it)
        {
            OperationPtr operation = it->second;
            std::vector<OperandPtr> inputs = model->getOperands(operation->inputs());
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

        return NNA_ERROR_CODE(NO_ERROR);
    }
}
