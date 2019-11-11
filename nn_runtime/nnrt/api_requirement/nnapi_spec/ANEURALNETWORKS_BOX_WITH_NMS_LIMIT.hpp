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

#ifndef __ANEURALNETWORKS_BOX_WITH_NMS_LIMIT_HPP__
#define __ANEURALNETWORKS_BOX_WITH_NMS_LIMIT_HPP__

#include "api_requirement/spec_macros.hpp"

/**
 * Greedily selects a subset of bounding boxes in descending order of score.
 *
 * This op applies NMS algorithm to each class. In each loop of execution,
 * the box with maximum score gets selected and removed from the pending set.
 * The scores of the rest of boxes are lowered according to the
 * intersection-over-union (IOU) overlapping with the previously selected
 * boxes and a specified NMS kernel method. Any boxes with score less
 * than a threshold are removed from the pending set.
 *
 * Three NMS kernels are supported:
 * * Hard:     score_new = score_old * (1 if IoU < threshold else 0)
 * * Linear:   score_new = score_old * (1 if IoU < threshold else 1 - IoU)
 * * Gaussian: score_new = score_old * exp(- IoU^2 / sigma)
 *
 * Axis-aligned bounding boxes are represented by its upper-left corner
 * coordinate (x1,y1) and lower-right corner coordinate (x2,y2). A valid
 * bounding box should satisfy x1 <= x2 and y1 <= y2.
 *
 * Supported tensor {@link OperandCode}:
 * * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
 * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
 * * {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}
 *
 * Inputs:
 * * 0: A 2-D Tensor of shape [num_rois, num_classes], specifying the score
 *      of each bounding box proposal. The boxes are grouped by batches in the
 *      first dimension. Zero num_rois is supported for this tensor.
 * * 1: A 2-D Tensor specifying the bounding boxes of shape
 *      [num_rois, num_classes * 4], organized in the order [x1, y1, x2, y2].
 *      The boxes are grouped by batches in the first dimension. The sequential
 *      order of the boxes corresponds with input0. For input0 of type
 *      {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM}, this tensor should be of
 *      {@link ANEURALNETWORKS_TENSOR_QUANT16_ASYMM}, with zeroPoint of 0 and
 *      scale of 0.125. Zero num_rois is supported for this tensor.
 * * 2: A 1-D {@link ANEURALNETWORKS_TENSOR_INT32} tensor, of shape
 *      [num_rois], specifying the batch index of each box. Boxes with
 *      the same batch index are grouped together.
 * * 3: An {@link ANEURALNETWORKS_FLOAT32} scalar, score_threshold. Boxes
 *      with scores lower than the threshold are filtered before sending
 *      to the NMS algorithm.
 * * 4: An {@link ANEURALNETWORKS_INT32} scalar, specifying the maximum
 *      number of selected bounding boxes for each image. Set to a negative
 *      value for unlimited number of output bounding boxes.
 * * 5: An {@link ANEURALNETWORKS_INT32} scalar, specifying the NMS
 *      kernel method, options are 0:hard, 1:linear, 2:gaussian.
 * * 6: An {@link ANEURALNETWORKS_FLOAT32} scalar, specifying the IoU
 *      threshold in hard and linear NMS kernel. This field is ignored if
 *      gaussian kernel is selected.
 * * 7: An {@link ANEURALNETWORKS_FLOAT32} scalar, specifying the sigma in
 *      gaussian NMS kernel. This field is ignored if gaussian kernel is
 *      not selected.
 * * 8: An {@link ANEURALNETWORKS_FLOAT32} scalar, nms_score_threshold.
 *      Boxes with scores lower than the threshold are dropped during the
 *      score updating phase in soft NMS.
 *
 * Outputs:
 * * 0: A 1-D Tensor of the same {@link OperandCode} as input0, with shape
 *      [num_output_rois], specifying the score of each output box. The boxes
 *      are grouped by batches, but the sequential order in each batch is not
 *      guaranteed. For type of {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM},
 *      the scale and zero point must be the same as input0.
 * * 1: A 2-D Tensor of the same {@link OperandCode} as input1, with shape
 *      [num_output_rois, 4], specifying the coordinates of each
 *      output bounding box with the same format as input1. The sequential
 *      order of the boxes corresponds with output0. For type of
 *      {@link ANEURALNETWORKS_TENSOR_QUANT16_ASYMM}, the scale must be
 *      0.125 and the zero point must be 0.
 * * 2: A 1-D {@link ANEURALNETWORKS_TENSOR_INT32} tensor, of shape
 *      [num_output_rois], specifying the class of each output box. The
 *      sequential order of the boxes corresponds with output0.
 * * 3: A 1-D {@link ANEURALNETWORKS_TENSOR_INT32} tensor, of shape
 *      [num_output_rois], specifying the batch index of each box. Boxes
 *      with the same batch index are grouped together.
 *
 * Available since API level 29.
 */

#define OP_SPEC_NAME BoxWithNmsLimitOperation
OP_SPEC_BEGIN()
#define ARG_NAMES         \
    (box_score,                 \
     boxes,           \
     batch_index,   \
     score_threshold, \
     max_boxes,   \
     nms_kernel_method, \
     iou_threshold, \
     nms_sigma, \
     nms_score_threshold)
#define ARGC BOOST_PP_TUPLE_SIZE(ARG_NAMES)

#define BOOST_PP_LOCAL_MACRO(n) OP_SPEC_ARG(BOOST_PP_TUPLE_ELEM(ARGC, n, ARG_NAMES))
#define BOOST_PP_LOCAL_LIMITS (0, ARGC)
#include BOOST_PP_LOCAL_ITERATE()
OP_SPEC_END()

// order of argument is important
MAKE_SPEC(box_with_nms_limit_operation)
    .box_score_(nnrt::OperandType::TENSOR_FLOAT32)
    .boxes_(nnrt::OperandType::TENSOR_FLOAT32)
    .batch_index_(nnrt::OperandType::TENSOR_INT32)
    .score_threshold_(nnrt::OperandType::FLOAT32)
    .max_boxes_(nnrt::OperandType::INT32)
    .nms_kernel_method_(nnrt::OperandType::INT32)
    .iou_threshold_(nnrt::OperandType::FLOAT32)
    .nms_sigma_(nnrt::OperandType::FLOAT32)
    .nms_score_threshold_(nnrt::OperandType::FLOAT32));

    OVERRIDE_SPEC(box_with_nms_limit_operation, float16)
    .box_score_(nnrt::OperandType::TENSOR_FLOAT16)
    .boxes_(nnrt::OperandType::TENSOR_FLOAT16));

    OVERRIDE_SPEC(box_with_nms_limit_operation, quant8_asymm)
    .box_score_(nnrt::OperandType::TENSOR_QUANT8_ASYMM)
    .boxes_(nnrt::OperandType::TENSOR_QUANT16_ASYMM));

#undef ARG_NAMES
#undef ARGC
#undef OP_SPEC_NAME

#endif
