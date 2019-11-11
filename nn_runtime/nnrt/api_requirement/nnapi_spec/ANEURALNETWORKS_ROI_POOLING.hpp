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
* Select and scale the feature map of each region of interest to a unified
* output size by average pooling sampling points from bilinear interpolation.
*
* The region of interest is represented by its upper-left corner coordinate
* (x1,y1) and lower-right corner coordinate (x2,y2) in the original image.
* A spatial scaling factor is applied to map into feature map coordinate.
* A valid region of interest should satisfy x1 <= x2 and y1 <= y2.
*
* No rounding is applied in this operation. The sampling points are unified
* distributed in the pooling bin and their values are calculated by bilinear
* interpolation.
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
* * 0: A 4-D tensor, specifying the feature map.
* * 1: A 2-D Tensor of shape [num_rois, 4], specifying the locations of
*      the regions of interest, each line with format [x1, y1, x2, y2].
*      For input0 of type {@link ANEURALNETWORKS_TENSOR_QUANT8_ASYMM},
*      this tensor should be of {@link ANEURALNETWORKS_TENSOR_QUANT16_ASYMM},
*      with zeroPoint of 0 and scale of 0.125. Zero num_rois is
*      supported for this tensor.
* * 2: An 1-D {@link ANEURALNETWORKS_TENSOR_INT32} tensor, of shape
*      [num_rois], specifying the batch index of each box. Boxes with
*      the same batch index are grouped together. Zero num_rois is
*      supported for this tensor.
* * 3: An {@link ANEURALNETWORKS_INT32} scalar, specifying the output
*      height of the output tensor.
* * 4: An {@link ANEURALNETWORKS_INT32} scalar, specifying the output
*      width of the output tensor.
* * 5: An {@link ANEURALNETWORKS_FLOAT32} scalar, specifying the ratio
*      from the height of original image to the height of feature map.
* * 6: An {@link ANEURALNETWORKS_FLOAT32} scalar, specifying the ratio
*      from the width of original image to the width of feature map.
* * 7: An {@link ANEURALNETWORKS_INT32} scalar, specifying the number of
*      sampling points in height dimension used to compute the output.
*      Set to 0 for adaptive value of ceil(roi_height/out_height).
* * 8: An {@link ANEURALNETWORKS_INT32} scalar, specifying the number of
*      sampling points in width dimension used to compute the output.
*      Set to 0 for adaptive value of ceil(roi_width/out_width).
* * 9: An {@link ANEURALNETWORKS_BOOL} scalar, set to true to specify
*      NCHW data layout for input0 and output0. Set to false for NHWC.
*
* Outputs:
* * 0: A tensor of the same {@link OperandCode} as input0. The output
*      shape is [num_rois, out_height, out_width, depth].
*
* Available since API level 29.
*/

#ifndef __AANEURALNETWORKS_ROI_POOLING_HPP__
#define __AANEURALNETWORKS_ROI_POOLING_HPP__

#include "api_requirement/spec_macros.hpp"

#define OP_SPEC_NAME ROIPoolingOperation
OP_SPEC_BEGIN()
#define ARG_NAMES         \
    (input,               \
     roi_location,        \
     batch_index,         \
     height,              \
     width,               \
     height_ratio,        \
     width_ratio,         \
     sampling_points_height,\
     sampling_points_width,\
     layout)
#define ARGC BOOST_PP_TUPLE_SIZE(ARG_NAMES)

#define BOOST_PP_LOCAL_MACRO(n) OP_SPEC_ARG(BOOST_PP_TUPLE_ELEM(ARGC, n, ARG_NAMES))
#define BOOST_PP_LOCAL_LIMITS (0, ARGC)
#include BOOST_PP_LOCAL_ITERATE()
OP_SPEC_END()

// order of argument is important
MAKE_SPEC(roi_pooling)
    .input_(nnrt::OperandType::TENSOR_FLOAT32)
    .roi_location_(nnrt::OperandType::TENSOR_FLOAT32)
    .batch_index_(nnrt::OperandType::TENSOR_INT32)
    .height_(nnrt::OperandType::INT32)
    .width_(nnrt::OperandType::INT32)
    .height_ratio_(nnrt::OperandType::FLOAT32)
    .width_ratio_(nnrt::OperandType::FLOAT32)
    .sampling_points_height_(nnrt::OperandType::INT32)
    .sampling_points_width_(nnrt::OperandType::INT32)
    .layout_(nnrt::OperandType::BOOL)
    );

    OVERRIDE_SPEC(roi_pooling, 0)
    .input_(nnrt::OperandType::TENSOR_FLOAT16)
    .roi_location_(nnrt::OperandType::TENSOR_FLOAT16)
    );

    OVERRIDE_SPEC(roi_pooling, 1)
    .input_(nnrt::OperandType::TENSOR_QUANT8_ASYMM)
    .roi_location_(nnrt::OperandType::TENSOR_QUANT8_ASYMM)
    );

#undef ARG_NAMES
#undef ARGC
#undef OP_SPEC_NAME

#endif
