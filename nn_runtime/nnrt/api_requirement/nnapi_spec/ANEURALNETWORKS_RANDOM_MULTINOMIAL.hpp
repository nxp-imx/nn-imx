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

#ifndef __ANEURALNETWORKS_RANDOM_MULTINOMIAL_HPP__
#define __ANEURALNETWORKS_RANDOM_MULTINOMIAL_HPP__

#include "api_requirement/spec_macros.hpp"

/**
 * Draws samples from a multinomial distribution.
 *
 * Supported tensor {@link OperandCode}:
 * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
 * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
 *
 * Inputs:
 * * 0: A 2-D tensor with shape [batches, classes], specifying the
 *      unnormalized log-probabilities for all classes.
 * * 1: A scalar {@link ANEURALNETWORKS_INT32}, specifying the number of
 *      independent samples to draw for each row slice.
 * * 2: A 1-D {@link ANEURALNETWORKS_TENSOR_INT32} tensor with shape [2],
 *      specifying seeds used to initialize the random distribution.
 * Outputs:
 * * 0: A 2-D {@link ANEURALNETWORKS_TENSOR_INT32} tensor with shape
 *      [batches, samples], containing the drawn samples.
 *
 * Available since API level 29.
 */

#define OP_SPEC_NAME RandomMultinomialOperation
OP_SPEC_BEGIN()
#define ARG_NAMES         \
    (input,                 \
     sample_num,           \
     seeds)
#define ARGC BOOST_PP_TUPLE_SIZE(ARG_NAMES)

#define BOOST_PP_LOCAL_MACRO(n) OP_SPEC_ARG(BOOST_PP_TUPLE_ELEM(ARGC, n, ARG_NAMES))
#define BOOST_PP_LOCAL_LIMITS (0, ARGC)
#include BOOST_PP_LOCAL_ITERATE()
OP_SPEC_END()

// order of argument is important
MAKE_SPEC(random_multinomial)
    .input_(nnrt::OperandType::TENSOR_FLOAT32)
    .sample_num_(nnrt::OperandType::INT32)
    .seeds_(nnrt::OperandType::TENSOR_INT32));

    OVERRIDE_SPEC(random_multinomial, float16)
    .input_(nnrt::OperandType::TENSOR_FLOAT16)
    .sample_num_(nnrt::OperandType::INT32)
    .seeds_(nnrt::OperandType::TENSOR_INT32));

#undef ARG_NAMES
#undef ARGC
#undef OP_SPEC_NAME

#endif
