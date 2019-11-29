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

/**
     * Inserts a dimension of 1 into a tensor's shape.
     *
     * Given a tensor input, this operation inserts a dimension of 1 at the
     * given dimension index of input's shape. The dimension index starts at
     * zero; if you specify a negative dimension index, it is counted backward
     * from the end.
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
     * * 0: An n-D tensor.
     * * 1: An {@link ANEURALNETWORKS_INT32} scalar specifying the dimension
     *      index to expand. Must be in the range [-(n + 1), (n + 1)).
     *
     * Outputs:
     * * 0: An (n + 1)-D tensor with the same {@link OperandCode} and data as
     *      input0.
     *
     * Available since API level 29.
     */


#ifndef __ANEURALNETWORKS_EXPAND_DIMS_HPP__
#define __ANEURALNETWORKS_EXPAND_DIMS_HPP__

#include "api_requirement/spec_macros.hpp"

#define OP_SPEC_NAME ExpandDims
OP_SPEC_BEGIN()
#define ARG_NAMES         \
    (input,               \
     dimIndex)
#define ARGC BOOST_PP_TUPLE_SIZE(ARG_NAMES)

#define BOOST_PP_LOCAL_MACRO(n) OP_SPEC_ARG(BOOST_PP_TUPLE_ELEM(ARGC, n, ARG_NAMES))
#define BOOST_PP_LOCAL_LIMITS (0, ARGC)
#include BOOST_PP_LOCAL_ITERATE()
OP_SPEC_END()

// order of argument is important
MAKE_SPEC(base_signature)
    .input_(nnrt::OperandType::TENSOR_FLOAT32)
    .dimIndex_(nnrt::OperandType::INT32));

OVERRIDE_SPEC(base_signature, input_float16)
    .input_(nnrt::OperandType::TENSOR_FLOAT16));

OVERRIDE_SPEC(base_signature, input_int32)
    .input_(nnrt::OperandType::TENSOR_INT32));
OVERRIDE_SPEC(base_signature, input_quant8_asym)
    .input_(nnrt::OperandType::TENSOR_QUANT8_ASYMM));
#undef ARG_NAMES
#undef ARGC
#undef OP_SPEC_NAME

#endif