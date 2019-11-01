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
     * Applies Local Response Normalization along the depth dimension.
     *
     * The 4-D input tensor is treated as a 3-D array of 1-D vectors (along the
     * last dimension), and each vector is normalized independently. Within a
     * given vector, each component is divided by the weighted, squared sum of
     * inputs within depth_radius.
     *
     * The output is calculated using this formula:
     *
     *     sqr_sum[a, b, c, d] = sum(
     *         pow(input[a, b, c, d - depth_radius : d + depth_radius + 1], 2))
     *     output = input / pow((bias + alpha * sqr_sum), beta)
     *
     * For input tensor with rank less than 4, independently normalizes each
     * 1-D slice along specified dimension.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16} (since API level 29)
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * Supported tensor rank: up to 4
     * Tensors with rank less than 4 are only supported since API level 29.
     *
     * Inputs:
     * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying
     *      the input.
     * * 1: An {@link ANEURALNETWORKS_INT32} scalar, specifying the radius of
     *      the normalization window.
     * * 2: An {@link ANEURALNETWORKS_FLOAT32} scalar, specifying the bias, must
     *      not be zero.
     * * 3: An {@link ANEURALNETWORKS_FLOAT32} scalar, specifying the scale
     *      factor, alpha.
     * * 4: An {@link ANEURALNETWORKS_FLOAT32} scalar, specifying the exponent,
     *      beta.
     * * 5: An optional {@link ANEURALNETWORKS_INT32} scalar, default to -1,
     *      specifying the dimension normalization would be performed on.
     *      Negative index is used to specify axis from the end (e.g. -1 for
     *      the last axis). Must be in the range [-n, n).
     *      Available since API level 29.
     *
     * Outputs:
     * * 0: The output tensor of same shape as input0.
     *
     * Available since API level 27.
     */

#ifndef __ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION_HPP__
#define __ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION_HPP__

#include "api_requirement/spec_macros.hpp"

#define OP_SPEC_NAME LocalResponseNormOperation
OP_SPEC_BEGIN()
#define ARG_NAMES         \
    (input,               \
     radius,              \
     bias,                \
     alpha,               \
     beta,                \
     axis)
#define ARGC BOOST_PP_TUPLE_SIZE(ARG_NAMES)

#define BOOST_PP_LOCAL_MACRO(n) OP_SPEC_ARG(BOOST_PP_TUPLE_ELEM(ARGC, n, ARG_NAMES))
#define BOOST_PP_LOCAL_LIMITS (0, ARGC)
#include BOOST_PP_LOCAL_ITERATE()
OP_SPEC_END()

// order of argument is important
MAKE_SPEC(local_response_norm)
    .input_(nnrt::OperandType::TENSOR_FLOAT32)
    .radius_(nnrt::OperandType::INT32)
    .bias_(nnrt::OperandType::FLOAT32)
    .alpha_(nnrt::OperandType::FLOAT32)
    .beta_(nnrt::OperandType::FLOAT32)
    .axis_(nnrt::OperandType::INT32, OPTIONAL));

    OVERRIDE_SPEC(local_response_norm, 0)
    .input_(nnrt::OperandType::TENSOR_FLOAT16));

#undef ARG_NAMES
#undef ARGC
#undef OP_SPEC_NAME

#endif