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

#ifndef __ANEURALNETWORKS_TOPK_V2_HPP__
#define __ANEURALNETWORKS_TOPK_V2_HPP__

#include "api_requirement/spec_macros.hpp"

/**
 * Finds values and indices of the k largest entries for the last dimension.
 *
 * Resulting values in each dimensions are sorted in descending order. If
 * two values are equal, the one with larger index appears first.
 *
 * Supported tensor {@link OperandCode}:
 * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
 * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
 * * {@link ANEURALNETWORKS_TENSOR_INT32}
 * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
 *
 * Supported tensor rank: from 1
 *
 * Inputs:
 * * 0: input, an n-D tensor specifying the input.
 * * 1: k, an {@link ANEURALNETWORKS_INT32} scalar, specifying the number of
 *      top elements to look for along the last dimension.
 *
 * Outputs:
 * * 0: An n-D tensor of the same type as the input, containing the k
 *      largest elements along each last dimensional slice.
 * * 1: An n-D tensor of type {@link ANEURALNETWORKS_TENSOR_INT32}
 *      containing the indices of values within the last dimension of input.
 *
 * Available since API level 29.
 */

#define OP_SPEC_NAME TopkV2Operation
OP_SPEC_BEGIN()
#define ARG_NAMES         \
    (input,                 \
     k)
#define ARGC BOOST_PP_TUPLE_SIZE(ARG_NAMES)

#define BOOST_PP_LOCAL_MACRO(n) OP_SPEC_ARG(BOOST_PP_TUPLE_ELEM(ARGC, n, ARG_NAMES))
#define BOOST_PP_LOCAL_LIMITS (0, ARGC)
#include BOOST_PP_LOCAL_ITERATE()
OP_SPEC_END()

// order of argument is important
MAKE_SPEC(topk_v2)
    .input_(nnrt::OperandType::TENSOR_FLOAT32)
    .k_(nnrt::OperandType::INT32));

    OVERRIDE_SPEC(topk_v2, float16)
    .input_(nnrt::OperandType::TENSOR_FLOAT16));

    OVERRIDE_SPEC(topk_v2, int32)
    .input_(nnrt::OperandType::TENSOR_INT32));

    OVERRIDE_SPEC(topk_v2, quant8_asymm)
    .input_(nnrt::OperandType::TENSOR_QUANT8_ASYMM));

#undef ARG_NAMES
#undef ARGC
#undef OP_SPEC_NAME

#endif
