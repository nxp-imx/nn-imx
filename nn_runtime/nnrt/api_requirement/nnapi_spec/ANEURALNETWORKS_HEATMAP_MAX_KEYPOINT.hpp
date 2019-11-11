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

#ifndef __ANEURALNETWORKS_HEATMAP_MAX_KEYPOINT_HPP__
#define __ANEURALNETWORKS_HEATMAP_MAX_KEYPOINT_HPP__

#include "api_requirement/spec_macros.hpp"

/**
 * Localize the maximum keypoints from heatmaps.
 *
 * This operation approximates the accurate maximum keypoint scores and
 * indices after bicubic upscaling by using Taylor expansion up to the
 * quadratic term.
 *
 * The bounding box is represented by its upper-left corner coordinate
 * (x1,y1) and lower-right corner coordinate (x2,y2) in the original image.
 * A valid bounding box should satisfy x1 <= x2 and y1 <= y2.
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
 * Inputs:
 * * 0: A 4-D Tensor of shape
 *      [num_boxes, heatmap_size, heatmap_size, num_keypoints],
 *      specifying the heatmaps, the height and width of heatmaps should
 *      be the same, and must be greater than or equal to 2.
 * * 1: A 2-D Tensor of shape [num_boxes, 4], specifying the bounding boxes,
 *      each with format [x1, y1, x2, y2]. For input0 of type
 *      {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}, this tensor should
 *      be of {@link ANEURALNETWORKS_TENSOR_QUANT16_ASYMM}, with zeroPoint
 *      of 0 and scale of 0.125.
 * * 2: An {@link ANEURALNETWORKS_BOOL} scalar, set to true to specify
 *      NCHW data layout for input0. Set to false for NHWC.
 *
 * Outputs:
 * * 0: A tensor of the same {@link OperandCode} as input0, with shape
 *      [num_boxes, num_keypoints], specifying score of the keypoints.
 * * 1: A tensor of the same {@link OperandCode} as input1, with shape
 *      [num_boxes, num_keypoints, 2], specifying the location of
 *      the keypoints, the second dimension is organized as
 *      [keypoint_x, keypoint_y].
 *
 * Available since API level 29.
 */

#define OP_SPEC_NAME HeatmapMaxKeypointOperation
OP_SPEC_BEGIN()
#define ARG_NAMES         \
    (input0,                 \
     input1,           \
     layout)
#define ARGC BOOST_PP_TUPLE_SIZE(ARG_NAMES)

#define BOOST_PP_LOCAL_MACRO(n) OP_SPEC_ARG(BOOST_PP_TUPLE_ELEM(ARGC, n, ARG_NAMES))
#define BOOST_PP_LOCAL_LIMITS (0, ARGC)
#include BOOST_PP_LOCAL_ITERATE()
OP_SPEC_END()

// order of argument is important
MAKE_SPEC(heatmap_max_keypoint)
    .input0_(nnrt::OperandType::TENSOR_FLOAT32)
    .input1_(nnrt::OperandType::TENSOR_FLOAT32)
    .layout_(nnrt::OperandType::BOOL));

    OVERRIDE_SPEC(heatmap_max_keypoint, float16)
    .input0_(nnrt::OperandType::TENSOR_FLOAT16)
    .input1_(nnrt::OperandType::TENSOR_FLOAT16));

    OVERRIDE_SPEC(heatmap_max_keypoint, quant8_asymm)
    .input0_(nnrt::OperandType::TENSOR_QUANT8_ASYMM)
    .input1_(nnrt::OperandType::TENSOR_QUANT16_ASYMM));

#undef ARG_NAMES
#undef ARGC
#undef OP_SPEC_NAME

#endif
