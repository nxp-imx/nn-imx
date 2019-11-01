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

#ifndef __ANEURALNETWORKS_AXIS_ALIGNED_BBOX_TRANSFORM_HPP__
#define __ANEURALNETWORKS_AXIS_ALIGNED_BBOX_TRANSFORM_HPP__

#include "api_requirement/spec_macros.hpp"

/**
 * Transform axis-aligned bounding box proposals using bounding box deltas.
 *
 * Given the positions of bounding box proposals and the corresponding
 * bounding box deltas for each class, return the refined bounding box
 * regions. The resulting bounding boxes are cliped against the edges of
 * the image.
 *
 * Supported tensor {@link OperandCode}:
 * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
 * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
 * * {@link ANEURALNETWORKS_TENSOR_QUANT16_ASYMM}
 *
 * Inputs:
 * * 0: A 2-D Tensor of shape [num_rois, 4], specifying the locations of the
 *      bounding box proposals, each line with format [x1, y1, x2, y2].
 *      For tensor of type {@link ANEURALNETWORKS_TENSOR_QUANT16_ASYMM},
 *      the zeroPoint must be 0 and the scale must be 0.125. Zero num_rois
 *      is supported for this tensor.
 * * 1: A 2-D Tensor of shape [num_rois, num_classes * 4], specifying the
 *      bounding box delta for each region of interest and each class. The
 *      bounding box deltas are organized in the following order
 *      [dx, dy, dw, dh], where dx and dy is the relative correction factor
 *      for the center position of the bounding box with respect to the width
 *      and height, dw and dh is the log-scale relative correction factor
 *      for the width and height. For input0 of type
 *      {@link ANEURALNETWORKS_TENSOR_QUANT16_ASYMM}, this tensor should be
 *      of {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}. Zero num_rois is
 *      supported for this tensor.
 * * 2: An 1-D {@link ANEURALNETWORKS_TENSOR_INT32} tensor, of shape
 *      [num_rois], specifying the batch index of each box. Boxes with
 *      the same batch index are grouped together. Zero num_rois is
 *      supported for this tensor.
 * * 3: A 2-D Tensor of shape [batches, 2], specifying the information of
 *      each image in the batch, each line with format
 *      [image_height, image_width].
 *
 * Outputs:
 * * 0: A tensor of the same {@link OperandCode} as input0, with shape
 *      [num_rois, num_classes * 4], specifying the coordinates of each
 *      output bounding box for each class, with format [x1, y1, x2, y2].
 *
 * Available since API level 29.
 */

#define OP_SPEC_NAME AxisAlignedBBoxTransformOperation
OP_SPEC_BEGIN()
#define ARG_NAMES         \
    (loc,                 \
     box_delta,           \
     box_index,           \
     image_meta)
#define ARGC BOOST_PP_TUPLE_SIZE(ARG_NAMES)

#define BOOST_PP_LOCAL_MACRO(n) OP_SPEC_ARG(BOOST_PP_TUPLE_ELEM(ARGC, n, ARG_NAMES))
#define BOOST_PP_LOCAL_LIMITS (0, ARGC)
#include BOOST_PP_LOCAL_ITERATE()
OP_SPEC_END()

// order of argument is important
MAKE_SPEC(axis_aligned_bbox_transform)
    .loc_(nnrt::OperandType::TENSOR_FLOAT32)
    .box_delta_(nnrt::OperandType::TENSOR_FLOAT32)
    .box_index_(nnrt::OperandType::TENSOR_INT32)
    .image_meta_(nnrt::OperandType::TENSOR_FLOAT32));

    OVERRIDE_SPEC(axis_aligned_bbox_transform, float16)
    .loc_(nnrt::OperandType::TENSOR_FLOAT16)
    .box_delta_(nnrt::OperandType::TENSOR_FLOAT16)
    .image_meta_(nnrt::OperandType::TENSOR_FLOAT32));

    OVERRIDE_SPEC(axis_aligned_bbox_transform, quant16)
    .loc_(nnrt::OperandType::TENSOR_QUANT16_ASYMM)
    .box_delta_(nnrt::OperandType::TENSOR_QUANT8_ASYMM)
    .image_meta_(nnrt::OperandType::TENSOR_QUANT16_ASYMM));

#undef ARG_NAMES
#undef ARGC
#undef OP_SPEC_NAME

#endif
