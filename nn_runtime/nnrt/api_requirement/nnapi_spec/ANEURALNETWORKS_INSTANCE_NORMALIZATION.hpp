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
     * Applies instance normalization to the input tensor.
     *
     * The values in the output tensor are computed as:
     *
     *     output[b, h, w, c] =
     *         (input[b, h, w, c] - mean[b, c]) * gamma /
     *         sqrt(var[b, c] + epsilon) + beta
     *
     * Where the mean and variance are computed across the spatial dimensions:
     *
     *     mean[b, c] =
     *         sum_{h, w}(input[b, h, w, c]) / sum(1)
     *
     *     var[b, c] =
     *         sum_{h, w}(pow(input[b, h, w, c] - mean[b, c], 2)) / sum(1)
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * Supported tensor rank: 4, with "NHWC" or "NCHW" data layout.
     * With the default data layout NHWC, the data is stored in the order of:
     * [batch, height, width, channels]. Alternatively, the data layout could
     * be NCHW, the data storage order of: [batch, channels, height, width].
     *
     * Inputs:
     * * 0: An n-D tensor, specifying the tensor to be normalized.
     * * 1: A scalar, specifying gamma, the scale applied to the normalized
     *      tensor. The scalar must be of {@link ANEURALNETWORKS_FLOAT16} if
     *      input0 is of {@link ANEURALNETWORKS_TENSOR_FLOAT16} and of {@link
     *      ANEURALNETWORKS_FLOAT32} if input0 is of {@link
     *      ANEURALNETWORKS_TENSOR_FLOAT32}.
     * * 2: A scalar, specifying beta, the offset applied to the normalized
     *      tensor. The scalar must be of {@link ANEURALNETWORKS_FLOAT16} if
     *      input0 is of {@link ANEURALNETWORKS_TENSOR_FLOAT16} and of {@link
     *      ANEURALNETWORKS_FLOAT32} if input0 is of {@link
     *      ANEURALNETWORKS_TENSOR_FLOAT32}.
     * * 3: A scalar, specifying epsilon, the small value added to variance to
     *      avoid dividing by zero. The scalar must be of {@link ANEURALNETWORKS_FLOAT16} if
     *      input0 is of {@link ANEURALNETWORKS_TENSOR_FLOAT16} and of {@link
     *      ANEURALNETWORKS_FLOAT32} if input0 is of {@link
     *      ANEURALNETWORKS_TENSOR_FLOAT32}.
     * * 4: An {@link ANEURALNETWORKS_BOOL} scalar, set to true to specify
     *      NCHW data layout for input0 and output0. Set to false for NHWC.
     *
     * Outputs:
     * * 0: A tensor of the same {@link OperandCode} and same shape as input0.
     *
     * Available since API level 29.
     */

#ifndef __ANEURALNETWORKS_INSTANCE_NORMALIZATION_HPP__
#define __ANEURALNETWORKS_INSTANCE_NORMALIZATION_HPP__

#include "api_requirement/spec_macros.hpp"

#define OP_SPEC_NAME InstanceNormOperation
OP_SPEC_BEGIN()
#define ARG_NAMES         \
    (input,               \
     gamma,               \
     beta,                \
     epsilon,             \
     data_layout)
#define ARGC BOOST_PP_TUPLE_SIZE(ARG_NAMES)

#define BOOST_PP_LOCAL_MACRO(n) OP_SPEC_ARG(BOOST_PP_TUPLE_ELEM(ARGC, n, ARG_NAMES))
#define BOOST_PP_LOCAL_LIMITS (0, ARGC)
#include BOOST_PP_LOCAL_ITERATE()
OP_SPEC_END()

// order of argument is important
MAKE_SPEC(instance_norm)
    .input_(nnrt::OperandType::TENSOR_FLOAT32)
    .gamma_(nnrt::OperandType::FLOAT32)
    .beta_(nnrt::OperandType::FLOAT32)
    .epsilon_(nnrt::OperandType::FLOAT32)
    .data_layout_(nnrt::OperandType::BOOL));

    OVERRIDE_SPEC(instance_norm, 0)
    .input_(nnrt::OperandType::TENSOR_FLOAT16)
    .gamma_(nnrt::OperandType::FLOAT16)
    .beta_(nnrt::OperandType::FLOAT16)
    .epsilon_(nnrt::OperandType::FLOAT16));

#undef ARG_NAMES
#undef ARGC
#undef OP_SPEC_NAME

#endif