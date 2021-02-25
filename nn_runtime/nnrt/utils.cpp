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
#include <algorithm>
#include <string.h>
#ifdef __ANDROID__
#include <sys/system_properties.h>
#endif

#include "nnrt/utils.hpp"
#include "nnrt/model.hpp"
#include "nnrt/types.hpp"
#include "nnrt/error.hpp"
#include "nnrt/op/public.hpp"
namespace nnrt
{
namespace operand_utils
{

int Transpose(Model* model, op::Operand* src, op::Operand* dst,
        std::vector<uint32_t>& perm, DataLayout layout)
{
    if (!src || !dst) {
        return NNA_ERROR_CODE(UNEXPECTED_NULL);
    }
    if (src->ndim() != perm.size()) {
        return NNA_ERROR_CODE(BAD_DATA);
    }
    dst->dimensions.resize(src->ndim());
    // Convert whcn to nchw layout
    std::vector<uint32_t> src_dimensions = src->dimensions;
    if (layout == DataLayout::WHCN) {
        std::reverse(src_dimensions.begin(), src_dimensions.end());
    }
    for (uint32_t i = 0; i < src->ndim(); ++ i) {
        dst->dimensions[i] = src->dimensions[perm[i]];
    }
    std::vector<int32_t> src_strides(src->ndim());
    std::vector<int32_t> dst_strides(dst->ndim());

    auto _compute_strides = [](std::vector<uint32_t>& shape,
            std::vector<int>& strides) {

        int acc = 1;
        std::transform(shape.rbegin(),
                       shape.rend(),
                       std::back_inserter(strides),
                       [&strides, &acc](const uint32_t d) {
                           if (strides.empty()) {
                               return 1;  // Stride for last dimension
                           } else {
                               acc *= static_cast<int>(d);
                               return acc;
                           }
                       });
        std::reverse(strides.begin(), strides.end());
    };
    _compute_strides(src->dimensions, src_strides);
    _compute_strides(dst->dimensions, dst_strides);

    uint8_t* src_data = model->getBuffer<uint8_t>(src->weak_mem_ref.lock());
    uint8_t* dst_data = model->getBuffer<uint8_t>(dst->weak_mem_ref.lock());

    int type_bytes = GetTypeBytes(src->type);
    for (int i_dst = 0; i_dst < (int)dst->size(); ++ i_dst) {
        int i_org = 0;
        int i_t = i_dst;
        for (int i = 0; i < (int)perm.size(); ++ i) {
            i_org += (i_t / dst_strides[i]) * src_strides[perm[i]];
            i_t %= dst_strides[i];
        }
        memcpy( &dst_data[i_dst * type_bytes], &src_data[i_org * type_bytes], type_bytes );
    }

    // Convert back to whcn
    if (layout == DataLayout::WHCN) {
        std::reverse(dst->dimensions.begin(), dst->dimensions.end());
    }
    return NNA_ERROR_CODE(NO_ERROR);
}

int Reshape(op::Operand* src, op::Operand* dst, std::vector<int>& shape)
{
    if (!src || !dst) {
        return NNA_ERROR_CODE(UNEXPECTED_NULL);
    }
    int negative_index = -1;
    size_t shape_product = 1;
    dst->dimensions.resize(shape.size());
    for (int i = 0; i < (int)shape.size(); ++ i) {
        if (shape[i] == -1) {
            if (negative_index < 0) {
                negative_index = i;
            } else {
                //TODO: Log invalid shape
                assert(false);
                return NNA_ERROR_CODE(BAD_DATA);
            }
        } else if (shape[i] > 0) {
            shape_product *= static_cast<size_t>(shape[i]);
            dst->dimensions[i] = shape[i];
        } else {
            //TODO: Log invalid shape
            assert(false);
            return NNA_ERROR_CODE(BAD_DATA);
        }
    }
    if (negative_index >= 0) {
        dst->dimensions[negative_index] = static_cast<uint32_t>(src->size() / shape_product);
    } else if (src->size() != shape_product) {
        //TODO: Log invalid shape
        assert(false);
        return NNA_ERROR_CODE(BAD_DATA);
    }
    return NNA_ERROR_CODE(NO_ERROR);
}

int GetTypeBytes(OperandType type)
{
    int bytes = 0;
    switch (type) {
        case OperandType::TENSOR_FLOAT32:
        case OperandType::TENSOR_INT32:
        case OperandType::TENSOR_QUANT32_SYMM:
        case OperandType::INT32:
        case OperandType::UINT32:
            bytes = 4;
            break;
        case OperandType::TENSOR_FLOAT16:
        case OperandType::TENSOR_INT16:
        case OperandType::FLOAT16:
        case OperandType::INT16:
        case OperandType::UINT16:
            bytes = 2;
            break;
        case OperandType::TENSOR_QUANT8_ASYMM:
        case OperandType::TENSOR_QUANT8_SYMM:
        case OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL:
        case OperandType::INT8:
        case OperandType::UINT8:
            bytes = 1;
            break;
        default:
            break;
    }
    return bytes;
}

uint16_t Fp32toFp16(float in) {
    uint32_t fp32 = *((uint32_t*)&in);
    uint32_t t1 = (fp32 & 0x80000000u) >> 16; /* sign bit. */
    uint32_t t2 = (fp32 & 0x7F800000u) >> 13; /* Exponent bits */
    uint32_t t3 = (fp32 & 0x007FE000u) >> 13; /* Mantissa bits, no rounding */
    uint32_t fp16 = 0u;
    if (t2 >= 0x023c00u) {
        fp16 = t1 | 0x7BFF; /* Don't round to infinity. */
    } else if (t2 <= 0x01c000u) {
        fp16 = t1;
    } else {
        t2 -= 0x01c000u;
        fp16 = t1 | t2 | t3;
    }
    return (uint16_t)fp16;
}

float Fp16toFp32(uint16_t in) {
    int32_t t1, t2, t3;
    float out;

    t1 = in & 0x7fff;         // Non-sign bits
    t2 = in & 0x8000;         // Sign bit
    t3 = in & 0x7c00;         // Exponent
    t1 <<= 13;                // Align mantissa on MSB
    t2 <<= 16;                // Shift sign bit into position
    t1 += 0x38000000;         // Adjust bias
    t1 = (t3 == 0 ? 0 : t1);  // Denormals-as-zero
    t1 |= t2;                 // Re-insert sign bit
    *((uint32_t*)&out) = t1;
    return out;
}

bool IsDynamicShape(nnrt::op::OperandPtr operand) {
    if (operand->dimensions.empty()) {
        return true;
    }

    for (auto dim : operand->dimensions) {
        if (0 != dim) {
            return false;
        }
    }
    return true;
}

bool InsertFp16ToFp32LayerBeforeOperand(Model* model,
                                        op::OperationPtr operation,
                                        op::OperandPtr operand) {
    std::vector<uint32_t> convertInputs;
    std::vector<uint32_t> convertOutputs;
    int orgIdx = model->getOperandIndex(operand);
    int newIdx = -1;
    auto newOperand = model->cloneOperand(operand, &newIdx);
    newOperand->type = OperandType::TENSOR_FLOAT32;
    convertInputs.push_back(orgIdx);
    convertOutputs.push_back(newIdx);
    operation->replaceInputs(orgIdx, newIdx);
    if (model->addOperation(OperationType::DATA_CONVERT,
                            convertInputs.data(),
                            convertInputs.size(),
                            convertOutputs.data(),
                            convertOutputs.size())) {
        return true;
    } else {
        return false;
    }
}

bool InsertPermuteBeforeOperand(Model* model,
                                op::OperationPtr operation,
                                uint32_t operandId,
                                const std::vector<uint32_t>& permVal) {
    int newOutOperandId = -1;
    uint32_t permInputs[1];
    uint32_t permOutputs[1];
    auto inputOperandPtr = model->operand(operandId);
    op::OperandPtr newOutOperandPtr = model->cloneOperand(inputOperandPtr, &newOutOperandId);
    newOutOperandPtr->dimensions = operation->dimensionTrans(newOutOperandPtr->dimensions, permVal);
    permInputs[0] = operandId;
    permOutputs[0] = (uint32_t)newOutOperandId;
    operation->replaceInputs(operandId, newOutOperandId);

    std::shared_ptr<nnrt::op::PermuteOperation> permuteOp =
        std::make_shared<nnrt::op::PermuteOperation>();
    permuteOp->perm.assign(permVal.begin(), permVal.end());

    uint32_t permId = 0;
    permuteOp->setInputs(permInputs, 1);
    permuteOp->setOutputs(permOutputs, 1);
    if(inputOperandPtr->isPerChannel()){
        inputOperandPtr->quant.vec.channelDim =
            nnrt::op::utils::convertAxis(static_cast<int32_t>(inputOperandPtr->quant.vec.channelDim),
                                            static_cast<int32_t>(inputOperandPtr->ndim()));
    }

    if (model->addOperation(permuteOp, &permId)) {
        return true;
    } else {
        return false;
    }
}

bool InsertReshapeBeforeOperand(Model* model,
                                op::OperationPtr operation,
                                uint32_t operandId,
                                const std::vector<uint32_t>& shape) {
    int newOutOperandId = -1;
    uint32_t reshapeInputs[2];
    uint32_t reshapeOutputs[1];
    auto inputOperandPtr = model->operand(operandId);
    op::OperandPtr newOutOperandPtr = model->cloneOperand(inputOperandPtr, &newOutOperandId);
    newOutOperandPtr->dimensions = shape;
    reshapeInputs[0] = operandId;
    reshapeOutputs[0] = (uint32_t)newOutOperandId;
    operation->replaceInputs(operandId, newOutOperandId);

    uint32_t shapeId = 0;
    auto shapeOperandPtr = model->addOperand(nullptr, &shapeId);
    shapeOperandPtr->type = nnrt::OperandType::TENSOR_INT32;
    shapeOperandPtr->dimensions = {static_cast<uint32_t>(shape.size())};
    model->setOperandValue(shapeId, shape.data(), shapeOperandPtr->bytes());
    reshapeInputs[1] = shapeId;

    std::shared_ptr<nnrt::op::ReshapeOperation> reshapeOp =
        std::make_shared<nnrt::op::ReshapeOperation>();

    uint32_t reshapeId = 0;
    reshapeOp->setInputs(reshapeInputs, 2);
    reshapeOp->setOutputs(reshapeOutputs, 1);
    if (model->addOperation(reshapeOp, &reshapeId)) {
        return true;
    } else {
        return false;
    }
}
}

namespace OS {
int getEnv(std::string name, int& result) {
    int get_success = 0;
#ifdef __ANDROID__
    char env[10] = {0};
    #if ANDROID_SDK_VERSION >= 30
        name = "vendor." + name;
    #endif
    get_success = __system_property_get(name.c_str(), env);
    if (get_success) result = atoi(env);
#else
    char* env = getenv(name.c_str());
    if (env) {
        result = atoi(env);
        get_success = 1;
    }
#endif
    return get_success;
}
}  // namespace OS
}