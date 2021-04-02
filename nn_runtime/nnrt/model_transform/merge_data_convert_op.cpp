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
#include <cassert>
#include <vector>

#include "nnrt/error.hpp"
#include "nnrt/model.hpp"
#include "nnrt/model_transform/transformations.hpp"
#include "nnrt/op/public.hpp"

using namespace nnrt::op;

namespace nnrt {

bool is_remove_all_data_convert_op(Model* model, uint32_t oprand_in, uint32_t oprand_out) {
    OperandPtr oprand0 = model->operand(oprand_in);
    OperandPtr oprand1 = model->operand(oprand_out);
    std::vector<OperandType> type_table = {
        OperandType::TENSOR_QUANT8_ASYMM,
        OperandType::TENSOR_QUANT8_ASYMM_SIGNED,
        OperandType::TENSOR_QUANT8_SYMM,
        OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL,
    };
    if (std::find(type_table.begin(), type_table.end(), oprand0->type) != type_table.end() &&
        std::find(type_table.begin(), type_table.end(), oprand1->type) != type_table.end()) {
        return true;
    }
    return false;
}

int MergeDataConvertOp::run(Model* model, bool* modified) {
    (void)modified;
    auto operations = model->operations();
#if 0
    std::map<uint32_t /*operand id*/, std::vector<uint32_t> /*operation ids*/>
        index_by_operand_for_op_input;
    std::map<uint32_t /*operand id*/, std::vector<uint32_t> /*operation ids*/>
        index_by_operand_for_op_output;

    for (auto it = operations.begin(); it != operations.end(); ++it) {
        OperationPtr operation = it->second;
        for (auto index : operation->inputs()) {
            auto it1 = index_by_operand_for_op_input.find(index);
            if (it1 != index_by_operand_for_op_input.end()) {
                it1->second.push_back(it->first);
            } else {
                std::vector<uint32_t> operation_ids;
                operation_ids.push_back(it->first);
                index_by_operand_for_op_input.insert(
                    std::make_pair<uint32_t, std::vector<uint32_t>>(std::move(index),
                                                                    std::move(operation_ids)));
            }
        }
    }

    for (auto it = operations.begin(); it != operations.end(); ++it) {
        OperationPtr operation = it->second;
        for (auto index : operation->outputs()) {
            auto it1 = index_by_operand_for_op_output.find(index);
            if (it1 != index_by_operand_for_op_output.end()) {
                it1->second.push_back(it->first);
            } else {
                std::vector<uint32_t> operation_ids;
                operation_ids.push_back(it->first);
                index_by_operand_for_op_output.insert(
                    std::make_pair<uint32_t, std::vector<uint32_t>>(std::move(index),
                                                                    std::move(operation_ids)));
            }
        }
    }
#endif
    op::IndexByOperand index_by_operand_for_op_input;
    op::IndexByOperand index_by_operand_for_op_output;
    model->get_index_by_operand(index_by_operand_for_op_input,
                                index_by_operand_for_op_output);

    std::vector<uint32_t> data_convert_op_list;
    for (auto it = operations.begin(); it != operations.end(); ++it) {
        OperationPtr operation = it->second;
        switch (operation->type()) {
            case OperationType::DEQUANTIZE:
            case OperationType::QUANTIZE: {
                data_convert_op_list.push_back(it->first);
                break;
            }
            default:
                break;
        }
    }

    std::vector<std::vector<uint32_t>> path_list;
    for (int32_t i = 0; i < (int32_t)(data_convert_op_list.size()) - 1; ++i) {
        auto output_ids = operations[data_convert_op_list[i]]->outputs();
        for (int32_t j = i + 1; j < (int32_t)(data_convert_op_list.size()); ++j) {
            auto input_ids = operations[data_convert_op_list[j]]->inputs();
            for (auto output_id : output_ids) {
                if (std::find(input_ids.begin(), input_ids.end(), output_id) != input_ids.end()) {
                    std::vector<uint32_t> path;
                    path.push_back(data_convert_op_list[i]);
                    path.push_back(data_convert_op_list[j]);
                    path_list.push_back(std::move(path));
                }
            }
        }
    }

    // merge path
    for (int32_t i = 0; i < (int32_t)(path_list.size()) - 1; ++i) {
        for (int32_t j = i + 1; j < (int32_t)(path_list.size()); ++j) {
            if (path_list[i][0] == path_list[j][path_list[j].size() - 1]) {
                path_list[i].insert(
                    path_list[i].begin(), path_list[j].begin(), path_list[j].end() - 1);
                path_list.erase(path_list.begin() + j);
                j = i;
            } else if (path_list[j][0] == path_list[i][path_list[i].size() - 1]) {
                path_list[i].insert(
                    path_list[i].end(), path_list[j].begin() + 1, path_list[j].end());
                path_list.erase(path_list.begin() + j);
                j = i;
            }
        }
    }

    for (auto path : path_list) {
        OperationPtr operation0 = operations[path[0]];
        OperationPtr operation1 = operations[path[path.size() - 1]];
        auto oprand0 = operation0->inputs()[0];
        auto oprand1 = operation1->outputs()[0];
        if (is_remove_all_data_convert_op(model, oprand0, oprand1)) {
            auto oprand0_output = index_by_operand_for_op_output[oprand0];
            if (oprand0_output.size() > 0) {
                path.insert(path.begin(), oprand0_output[0]);
                operation0 = operations[path[0]];
            }
        }
        auto old_output0 = operation0->outputs()[0];
        operation0->replaceOutputs(old_output0, operation1->outputs()[0]);
        model->removeOperand(old_output0);
        for (int32_t i = 1; i < (int32_t)(path.size()) - 1; ++i) {
            OperationPtr operation2 = operations[path[i]];
            model->removeOperand(operation2->outputs()[0]);
            model->removeOperation(path[i]);
        }
        model->removeOperation(path[path.size() - 1]);
    }

    return NNA_ERROR_CODE(NO_ERROR);
}
}  // namespace nnrt