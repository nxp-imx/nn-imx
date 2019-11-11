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

#ifndef __ANEURALNETWORKS_TILE_HPP__
#define __ANEURALNETWORKS_TILE_HPP__

#include "api_requirement/spec_macros.hpp"

/**
 * Constructs a tensor by tiling a given tensor.
 *
 * This operation creates a new tensor by replicating `input` `multiples`
 * times. The output tensor's i-th dimension has `input.dims(i) * multiples[i]`
 * elements, and the values of `input` are replicated `multiples[i]` times
 * along the i-th dimension.
 * For example, tiling `[a b c d]` by `[2]` produces `[a b c d a b c d]`.
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
 * * 1: multiples, a 1-D tensor of {@link ANEURALNETWORKS_TENSOR_INT32}.
 *      The length of multiples must be n.
 *
 * Outputs:
 * * 0: A tiled tensor of the same {@link OperandCode} and rank as `input`.
 *
 * Available since API level 29.
 */

#define OP_SPEC_NAME TileOperation
OP_SPEC_BEGIN()
#define ARG_NAMES         \
    (input,                 \
     multiples)
#define ARGC BOOST_PP_TUPLE_SIZE(ARG_NAMES)

#define BOOST_PP_LOCAL_MACRO(n) OP_SPEC_ARG(BOOST_PP_TUPLE_ELEM(ARGC, n, ARG_NAMES))
#define BOOST_PP_LOCAL_LIMITS (0, ARGC)
#include BOOST_PP_LOCAL_ITERATE()
OP_SPEC_END()

// order of argument is important
MAKE_SPEC(tile)
    .input_(nnrt::OperandType::TENSOR_FLOAT32)
    .multiples_(nnrt::OperandType::TENSOR_INT32));

    OVERRIDE_SPEC(tile, float16)
    .input_(nnrt::OperandType::TENSOR_FLOAT16));

    OVERRIDE_SPEC(tile, int32)
    .input_(nnrt::OperandType::TENSOR_INT32));

    OVERRIDE_SPEC(tile, quant8_asymm)
    .input_(nnrt::OperandType::TENSOR_QUANT8_ASYMM));

#undef ARG_NAMES
#undef ARGC
#undef OP_SPEC_NAME

#endif
