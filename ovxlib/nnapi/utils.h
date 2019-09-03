#ifndef __OVXLIB_UTIL_H__
#define __OVXLIB_UTIL_H__

#include <vector>
#include "model.h"
#include "types.h"

namespace ovxlib
{
namespace operand_utils
{

int GetTypeBytes(OperandType type);

int Transpose(Model* model, Operand* src, Operand* dst,
        std::vector<uint32_t>& perm, OperandLayout layout);

int Reshape(Operand* src, Operand* dst, std::vector<int>& shape);

}
}

#endif
