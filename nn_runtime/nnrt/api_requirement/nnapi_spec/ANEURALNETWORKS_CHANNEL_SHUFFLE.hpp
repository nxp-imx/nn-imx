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
* Shuffle the channels of the input tensor.
*
* Given an input tensor and a integer value of num_groups, CHANNEL_SHUFFLE
* divide the channel dimension into num_groups groups, and reorganize the
* channels by grouping channels with the same index in each group.
*
* Along the channel dimension, the output is calculated using this formula:
*
*     output_channel[k * num_groups + g] = input_channel[g * group_size + k]
*
* where group_size = num_channels / num_groups
*
* The number of channels must be divisible by num_groups.
*
* Supported tensor {@link OperandCode}:
* * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
* * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
* * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
*
* Supported tensor rank: up to 4
*
* Inputs:
* * 0: An n-D tensor, specifying the tensor to be shuffled.
* * 1: An {@link ANEURALNETWORKS_INT32} scalar, specifying the number of
*      groups.
* * 2: An {@link ANEURALNETWORKS_INT32} scalar, specifying the dimension
*      channel shuffle would be performed on. Negative index is used to
*      specify axis from the end (e.g. -1 for the last axis). Must be in
*      the range [-n, n).
*
* Outputs:
* * 0: A tensor of the same {@link OperandCode} and same shape as input0.
*
* Available since API level 29.
*/

#ifndef __AANEURALNETWORKS_CHANNEL_SHUFFLE_HPP__
#define __AANEURALNETWORKS_CHANNEL_SHUFFLE_HPP__

#include "api_requirement/spec_macros.hpp"

#define OP_SPEC_NAME ChannelShuffleOperation
OP_SPEC_BEGIN()
#define ARG_NAMES         \
    (input,               \
     groups,              \
     axis)
#define ARGC BOOST_PP_TUPLE_SIZE(ARG_NAMES)

#define BOOST_PP_LOCAL_MACRO(n) OP_SPEC_ARG(BOOST_PP_TUPLE_ELEM(ARGC, n, ARG_NAMES))
#define BOOST_PP_LOCAL_LIMITS (0, ARGC)
#include BOOST_PP_LOCAL_ITERATE()
OP_SPEC_END()

// order of argument is important
MAKE_SPEC(channel_shuffle)
    .input_(nnrt::OperandType::TENSOR_FLOAT32)
    .groups_(nnrt::OperandType::INT32)
    .axis_(nnrt::OperandType::INT32));

    OVERRIDE_SPEC(channel_shuffle, 0)
    .input_(nnrt::OperandType::TENSOR_FLOAT16));

    OVERRIDE_SPEC(channel_shuffle, 1)
    .input_(nnrt::OperandType::TENSOR_QUANT8_ASYMM));

#undef ARG_NAMES
#undef ARGC
#undef OP_SPEC_NAME

#endif