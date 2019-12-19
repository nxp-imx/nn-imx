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

#ifndef __ANEURALNETWORKS_SPACE_TO_DEPTH_HPP__
#define __ANEURALNETWORKS_SPACE_TO_DEPTH_HPP__

#include "api_requirement/spec_macros.hpp"

/**
     * Rearranges blocks of spatial data, into depth.
     *
     * More specifically, this op outputs a copy of the input tensor where
     * values from the height and width dimensions are moved to the depth
     * dimension. The value block_size indicates the input block size and how
     * the data is moved.
     *
     * Chunks of data of size block_size * block_size from depth are rearranged
     * into non-overlapping blocks of size block_size x block_size.
     *
     * The depth of the output tensor is input_depth * block_size * block_size.
     * The input tensor's height and width must be divisible by block_size.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16} (since API level 29)
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: 4, with "NHWC" or "NCHW" data layout.
     * With the default data layout NHWC, the data is stored in the order of:
     * [batch, height, width, channels]. Alternatively, the data layout could
     * be NCHW, the data storage order of: [batch, channels, height, width].
     *
     * Inputs:
     * * 0: A 4-D tensor, of shape [batches, height, width, depth_in],
     *      specifying the input.
     * * 1: An {@link ANEURALNETWORKS_INT32} scalar, specifying the block_size.
     *      block_size must be >=1 and block_size must be a divisor of both the
     *      input height and width.
     * * 2: An optional {@link ANEURALNETWORKS_BOOL} scalar, default to false.
     *      Set to true to specify NCHW data layout for input0 and output0.
     *      Available since API level 29.
     *
     * Outputs:
     * * 0: The output 4-D tensor, of shape [batches, height/block_size,
     *      width/block_size, depth_in*block_size*block_size].
     *
     * Available since API level 27.
     */

#define OP_SPEC_NAME Space2DepthOperation
OP_SPEC_BEGIN()
#define ARG_NAMES           \
    (input,                 \
     block_size,            \
     data_layout)
#define ARGC BOOST_PP_TUPLE_SIZE(ARG_NAMES)

#define BOOST_PP_LOCAL_MACRO(n) OP_SPEC_ARG(BOOST_PP_TUPLE_ELEM(ARGC, n, ARG_NAMES))
#define BOOST_PP_LOCAL_LIMITS (0, ARGC)
#include BOOST_PP_LOCAL_ITERATE()
OP_SPEC_END()

// order of argument is important
MAKE_SPEC(space2depth)
    .input_(nnrt::OperandType::TENSOR_FLOAT32)
    .block_size_(nnrt::OperandType::INT32)
    .data_layout_(nnrt::OperandType::BOOL, OPTIONAL));

    OVERRIDE_SPEC(space2depth, float16)
    .input_(nnrt::OperandType::TENSOR_FLOAT16));

    OVERRIDE_SPEC(space2depth, asymm_u8)
    .input_(nnrt::OperandType::TENSOR_QUANT8_ASYMM));

#undef ARG_NAMES
#undef ARGC
#undef OP_SPEC_NAME

#endif
