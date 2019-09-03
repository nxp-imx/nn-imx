#include <vector>
#include "vsi_nn_pub.h"
#include "operand.h"
#include "utils.h"

namespace ovxlib
{
size_t Operand::bytes() const {
    size_t bytes = static_cast<size_t>(operand_utils::GetTypeBytes(type));
    for (auto i : dimensions) {
        bytes *= i;
    }
    return bytes;
}

bool Operand::isTensor() const {
    return (type >= OperandType::TENSOR_INDEX_START);
}

bool Operand::isQuantized() const {
    switch(type) {
        case OperandType::TENSOR_QUANT8_ASYMM:
        case OperandType::TENSOR_QUANT8_SYMM:
        case OperandType::TENSOR_QUANT32_SYMM:
            return true;
        default: break;
    }
    return false;
}

Operand* Operand::clone() {
    Operand* operand = new Operand();
    operand->type = type;
    operand->dimensions = dimensions;
    //operand->scale = scale;
    //operand->zeroPoint = zeroPoint;
    operand->cloneQuantParams(this);
    operand->number_of_consumers_ = number_of_consumers_;
    return operand;
}

void Operand::cloneQuantParams(Operand* operand) {
    if (!operand) {
        return;
    }
    switch (operand->type) {
        case OperandType::TENSOR_QUANT32_SYMM:
        case OperandType::TENSOR_QUANT8_SYMM:
            quant.scalar.scale = operand->quant.scalar.scale;
            break;
        case OperandType::TENSOR_QUANT8_ASYMM:
            quant.scalar.scale = operand->quant.scalar.scale;
            quant.scalar.zeroPoint = operand->quant.scalar.zeroPoint;
            break;
#if 0
        case OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL:
            quant.vScale = operand->quant.vScale;
            quant.vZeroPoint = operand->quant.vZeroPoint;
            break;
        case OperandType::TENSOR_QUANT8_ASYMM_PER_CHANNEL:
            quant.vScale = operand->quant.vScale;
            quant.vZeroPoint = operand->quant.vZeroPoint;
            break;
#endif
        default:
            break;
    }
}

void Operand::echo(uint32_t index) const {
    char buf[256] = {0};
    size_t sz = 0;
    sz += snprintf(&buf[sz], 256 - sz, "%-4u [", index);
    for (uint32_t i = 0; i < dimensions.size(); i ++)
    {
    sz += snprintf(&buf[sz], 256 - sz, "%-5d", dimensions[i]);
    }
    sz += snprintf(&buf[sz], 256 - sz, " ]");
    VSILOGD("%s", buf);
}

}

