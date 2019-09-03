#include <vector>
#include "error.h"
#include "transformations.h"

namespace ovxlib
{
int ValidateQuantizedGraph::run(Model * model, bool * modified)
{
    *modified = false;
    if (nullptr == model) {
        return AERROR_CODE(NO_ERROR);
    }
    // Set bias scale and zero point
    auto operations = model->operations();
    for (auto it = operations.begin(); it != operations.end(); ++ it) {
        Operation* op = it->second;
        switch (op->type()) {
            case OperationType::CONV_2D:
            case OperationType::DEPTHWISE_CONV_2D:
            case OperationType::FULLY_CONNECTED:
                break;
            default:
                continue;
        }
        std::vector<Operand*> inputs = model->getOperands(op->inputs());
        if (inputs[0]->type == OperandType::TENSOR_QUANT8_ASYMM
                && inputs[2]->type == OperandType::TENSOR_INT32) {
            inputs[2]->quant.scalar.scale = \
                        inputs[0]->quant.scalar.scale * inputs[1]->quant.scalar.scale;
            inputs[2]->quant.scalar.zeroPoint = 0;
            inputs[2]->type = OperandType::TENSOR_QUANT32_SYMM;
            *modified = true;
        }
    }
    return AERROR_CODE(NO_ERROR);
}
}
