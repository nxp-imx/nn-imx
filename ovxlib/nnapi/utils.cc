#include <assert.h>
#include <vector>
#include <algorithm>
#include "utils.h"
#include "model.h"
#include "types.h"
#include "error.h"

namespace ovxlib
{
namespace operand_utils
{

int Transpose(Model* model, Operand* src, Operand* dst,
        std::vector<uint32_t>& perm, OperandLayout layout)
{
    if (!src || !dst) {
        return AERROR_CODE(UNEXPECTED_NULL);
    }
    if (src->ndim() != perm.size()) {
        return AERROR_CODE(BAD_DATA);
    }
    dst->dimensions.resize(src->ndim());
    // Convert whcn to nchw layout
    std::vector<uint32_t> src_dimensions = src->dimensions;
    if (layout == OperandLayout::WHCN) {
        std::reverse(src_dimensions.begin(), src_dimensions.end());
    }
    for (uint32_t i = 0; i < src->ndim(); ++ i) {
        dst->dimensions[i] = src->dimensions[perm[i]];
    }
    std::vector<int32_t> src_strides(src->ndim());
    std::vector<int32_t> dst_strides(dst->ndim());

    auto _compute_strides = [](std::vector<uint32_t>& shape,
            std::vector<int>& strides) {
        int s = 1;
        for (uint32_t i = shape.size() - 1; i >= 0; -- i) {
            strides[i] = s;
            s *= static_cast<int>(shape[i]);
        }
    };
    _compute_strides(src->dimensions, src_strides);
    _compute_strides(dst->dimensions, dst_strides);

    uint8_t* src_data = model->getBuffer<uint8_t>(src->mem_ref);
    uint8_t* dst_data = model->getBuffer<uint8_t>(dst->mem_ref);

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
    if (layout == OperandLayout::WHCN) {
        std::reverse(dst->dimensions.begin(), dst->dimensions.end());
    }
    return AERROR_CODE(NO_ERROR);
}

int Reshape(Operand* src, Operand* dst, std::vector<int>& shape)
{
    if (!src || !dst) {
        return AERROR_CODE(UNEXPECTED_NULL);
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
                return AERROR_CODE(BAD_DATA);
            }
        } else if (shape[i] > 0) {
            shape_product *= static_cast<size_t>(shape[i]);
            dst->dimensions[i] = shape[i];
        } else {
            //TODO: Log invalid shape
            assert(false);
            return AERROR_CODE(BAD_DATA);
        }
    }
    if (negative_index >= 0) {
        dst->dimensions[negative_index] = static_cast<uint32_t>(src->size() / shape_product);
    } else if (src->size() != shape_product) {
        //TODO: Log invalid shape
        assert(false);
        return AERROR_CODE(BAD_DATA);
    }
    return AERROR_CODE(NO_ERROR);
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
        case OperandType::INT8:
        case OperandType::UINT8:
            bytes = 1;
            break;
        default:
            break;
    }
    return bytes;
}

}
}
