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
     * Resizes images to given size using the nearest neighbor interpretation.
     *
     * Resized images must be distorted if their output aspect ratio is not the
     * same as input aspect ratio. The corner pixels of output may not be the
     * same as corner pixels of input.
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
     *
     * Supported tensor rank: 4, with "NHWC" or "NCHW" data layout.
     * With the default data layout NHWC, the data is stored in the order of:
     * [batch, height, width, channels]. Alternatively, the data layout could
     * be NCHW, the data storage order of: [batch, channels, height, width].
     *
     * Both resizing by shape and resizing by scale are supported.
     *
     * Inputs (resizing by shape):
     * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying
     *      the input. Zero batches is supported for this tensor.
     * * 1: An {@link ANEURALNETWORKS_INT32} scalar, specifying the output
     *      height of the output tensor.
     * * 2: An {@link ANEURALNETWORKS_INT32} scalar, specifying the output
     *      width of the output tensor.
     * * 3: An {@link ANEURALNETWORKS_BOOL} scalar, default to false.
     *      Set to true to specify NCHW data layout for input0 and output0.
     *
     * Inputs (resizing by scale):
     * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying
     *      the input. Zero batches is supported for this tensor.
     * * 1: A scalar, specifying height_scale, the scaling factor of the height
     *      dimension from the input tensor to the output tensor. The output
     *      height is calculated as new_height = floor(height * height_scale).
     *      The scalar must be of {@link ANEURALNETWORKS_FLOAT16} if input0 is
     *      of {@link ANEURALNETWORKS_TENSOR_FLOAT16} and of
     *      {@link ANEURALNETWORKS_FLOAT32} otherwise.
     * * 2: A scalar, specifying width_scale, the scaling factor of the width
     *      dimension from the input tensor to the output tensor. The output
     *      width is calculated as new_width = floor(width * width_scale).
     *      The scalar must be of {@link ANEURALNETWORKS_FLOAT16} if input0 is
     *      of {@link ANEURALNETWORKS_TENSOR_FLOAT16} and of
     *      {@link ANEURALNETWORKS_FLOAT32} otherwise.
     * * 3: An {@link ANEURALNETWORKS_BOOL} scalar, default to false.
     *      Set to true to specify NCHW data layout for input0 and output0.
     *
     * Outputs:
     * * 0: The output 4-D tensor, of shape
     *      [batches, new_height, new_width, depth].
     *
     * Available since API level 29.
     */

#ifndef __ANEURALNETWORKS_RESIZE_NEAREST_NEIGHBOR_HPP__
#define __ANEURALNETWORKS_RESIZE_NEAREST_NEIGHBOR_HPP__

#include "api_requirement/spec_macros.hpp"

#define OP_SPEC_NAME ResizeNearestNeighborOperation
OP_SPEC_BEGIN()
#define ARG_NAMES         \
    (input,               \
     output_height,       \
     output_width,        \
     height_scale,        \
     width_scale,         \
     data_layout)
#define ARGC BOOST_PP_TUPLE_SIZE(ARG_NAMES)

#define BOOST_PP_LOCAL_MACRO(n) OP_SPEC_ARG(BOOST_PP_TUPLE_ELEM(ARGC, n, ARG_NAMES))
#define BOOST_PP_LOCAL_LIMITS (0, ARGC)
#include BOOST_PP_LOCAL_ITERATE()
OP_SPEC_END()

// order of argument is important
MAKE_SPEC(height_width_base)
    .input_(nnrt::OperandType::TENSOR_FLOAT32)
    .output_height_(nnrt::OperandType::INT32)
    .output_width_(nnrt::OperandType::INT32)
    .data_layout_(nnrt::OperandType::BOOL));

    OVERRIDE_SPEC(height_width_base, 0)
    .input_(nnrt::OperandType::TENSOR_FLOAT16));

    OVERRIDE_SPEC(height_width_base, 1)
    .input_(nnrt::OperandType::TENSOR_QUANT8_ASYMM));

MAKE_SPEC(scale_base)
    .input_(nnrt::OperandType::TENSOR_FLOAT32)
    .height_scale_(nnrt::OperandType::FLOAT32)
    .width_scale_(nnrt::OperandType::FLOAT32)
    .data_layout_(nnrt::OperandType::BOOL));

    OVERRIDE_SPEC(scale_base, 0)
    .input_(nnrt::OperandType::TENSOR_FLOAT16)
    .height_scale_(nnrt::OperandType::FLOAT16)
    .width_scale_(nnrt::OperandType::FLOAT16));

    OVERRIDE_SPEC(scale_base, 1)
    .input_(nnrt::OperandType::TENSOR_QUANT8_ASYMM));

#undef ARG_NAMES
#undef ARGC
#undef OP_SPEC_NAME

#endif