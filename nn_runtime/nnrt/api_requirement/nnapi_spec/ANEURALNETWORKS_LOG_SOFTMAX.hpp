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

#ifndef __ANEURALNETWORKS_LOG_SOFTMAX_HPP__
#define __ANEURALNETWORKS_LOG_SOFTMAX_HPP__

#include "api_requirement/spec_macros.hpp"

/**
 * Computes the log softmax activations given logits.
 *
 * The output is calculated using this formula:
 *
 *     output = logits * beta - log(reduce_sum(exp(logits * beta), axis))
 *
 * Supported tensor {@link OperandCode}:
 * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
 * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
 *
 * Supported tensor rank: from 1.
 *
 * Inputs:
 * * 0: A tensor specifying the input logits.
 * * 1: An {@link ANEURALNETWORKS_FLOAT32} scalar, specifying the positive
 *      scaling factor for the exponent, beta.
 * * 2: An {@link ANEURALNETWORKS_INT32} scalar specifying the axis to
 *      reduce across. Negative index is used to specify axis from the
 *      end (e.g. -1 for the last axis). Must be in the range [-n, n).
 *
 * Outputs:
 * * 0: The output tensor of the same {@link OperandCode} and shape as
 *      input0.
 *
 * Available since API level 29.
 */

#define OP_SPEC_NAME LogSoftmaxOperation
OP_SPEC_BEGIN()
#define ARG_NAMES         \
    (logits,                 \
     beta,           \
     axis)
#define ARGC BOOST_PP_TUPLE_SIZE(ARG_NAMES)

#define BOOST_PP_LOCAL_MACRO(n) OP_SPEC_ARG(BOOST_PP_TUPLE_ELEM(ARGC, n, ARG_NAMES))
#define BOOST_PP_LOCAL_LIMITS (0, ARGC)
#include BOOST_PP_LOCAL_ITERATE()
OP_SPEC_END()

// order of argument is important
MAKE_SPEC(log_softmax)
    .logits_(nnrt::OperandType::TENSOR_FLOAT32)
    .beta_(nnrt::OperandType::FLOAT32)
    .axis_(nnrt::OperandType::INT32));

    OVERRIDE_SPEC(log_softmax, float16)
    .logits_(nnrt::OperandType::TENSOR_FLOAT16));

#undef ARG_NAMES
#undef ARGC
#undef OP_SPEC_NAME

#endif
