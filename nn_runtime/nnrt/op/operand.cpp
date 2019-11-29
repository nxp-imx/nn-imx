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
#include <vector>

#include "utils.hpp"
#include "logging.hpp"

#include "op/operand.hpp"

namespace nnrt {
namespace op {

namespace {
    const std::string tag = "Operand";
}

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
    switch (type) {
        case OperandType::TENSOR_QUANT8_ASYMM:
        case OperandType::TENSOR_QUANT8_SYMM:
        case OperandType::TENSOR_QUANT32_SYMM:
            return true;
        default:
            break;
    }
    return false;
}

OperandPtr Operand::clone() {
    OperandPtr operand = std::make_shared<Operand>();
    if (operand) {
        operand->type = type;
        operand->dimensions = dimensions;
        // operand->scale = scale;
        // operand->zeroPoint = zeroPoint;
        operand->cloneQuantParams(this);
        operand->number_of_consumers_ = number_of_consumers_;
    } else {
        assert(false);
        NNRT_LOGE(tag) << "Fatal Error: OOM cannot allocate Operand";
    }
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
    for (uint32_t i = 0; i < dimensions.size(); i++) {
        sz += snprintf(&buf[sz], 256 - sz, "%-5d", dimensions[i]);
    }
    sz += snprintf(&buf[sz], 256 - sz, " ]");
    NNRT_LOGD_PRINT("%s", buf);
}
}
}
