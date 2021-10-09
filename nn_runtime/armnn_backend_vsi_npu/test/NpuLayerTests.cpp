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

#include "NpuWorkloadFactoryHelper.hpp"

#include "NpuWorkloadFactory.hpp"
#include <test/UnitTests.hpp>

// #include <backendsCommon/test/DetectionPostProcessLayerTestImpl.hpp>
// #include <backendsCommon/test/DebugTestImpl.hpp>
// #include <backendsCommon/test/DetectionPostProcessLayerTestImpl.hpp>
#include <backendsCommon/test/LayerTests.hpp>

// #include <backendsCommon/test/PermuteTestImpl.hpp>
// #include <backendsCommon/test/TransposeConvolution2dTestImpl.hpp>

TEST_SUITE("VSI_NPU")
{
using namespace armnn;

using FactoryType = NpuWorkloadFactory;
// ============================================================================
// UNIT tests

// Convolution
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2d3x5, SimpleConvolution2d3x5Test, true, DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2d3x5Uint8,
                     SimpleConvolution2d3x5Uint8Test,
                     true,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2d3x5Nhwc, SimpleConvolution2d3x5Test, true, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2d3x5Uint8Nhwc,
                     SimpleConvolution2d3x5Uint8Test,
                     true,
                     DataLayout::NHWC)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2d3x5QSymm16, SimpleConvolution2d3x5QSymm16Test, true,
// DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2d3x5QSymm16Nhwc, SimpleConvolution2d3x5QSymm16Test, true,
// DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedConvolution2d, SimpleConvolution2d3x5Test, false, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedConvolutionUint8,
                     SimpleConvolution2d3x5Uint8Test,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedConvolution2dNhwc, SimpleConvolution2d3x5Test, false, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedConvolutionUint8Nhwc,
                     SimpleConvolution2d3x5Uint8Test,
                     false,
                     DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution1d, Convolution1dTest, true)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution1dUint8, Convolution1dUint8Test, true)

ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2d3x3, SimpleConvolution2d3x3Test, true, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2d3x3Uint8,
                     SimpleConvolution2d3x3Uint8Test,
                     true,
                     DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2d3x3QSymm16, SimpleConvolution2d3x3QSymm16Test, true,
// DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2d3x3Nhwc, SimpleConvolution2d3x3Test, true, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2d3x3Uint8Nhwc,
                     SimpleConvolution2d3x3Uint8Test,
                     true,
                     DataLayout::NHWC)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2d3x3QSymm16Nhwc, SimpleConvolution2d3x3QSymm16Test, true,
//                      DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedConvolution2dSquare,
                     SimpleConvolution2d3x3Test,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedConvolution2dSquareNhwc,
                     SimpleConvolution2d3x3Test,
                     false,
                     DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedConvolution2dSquareStride2x2Nhwc,
                     SimpleConvolution2d3x3Stride2x2Test,
                     false,
                     DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2dAsymmetricPaddingLargerThanHalfKernelSize,
                     Convolution2dAsymmetricPaddingLargerThanHalfKernelSizeTest,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2dAsymmetricPadding,
                     Convolution2dAsymmetricPaddingTest,
                     DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2dAsymmetricPaddingLargerThanHalfKernelSizeNhwc,
                     Convolution2dAsymmetricPaddingLargerThanHalfKernelSizeTest,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2dAsymmetricPaddingNhwc,
                     Convolution2dAsymmetricPaddingTest,
                     DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2dSquareNhwc, SimpleConvolution2d3x3NhwcTest, false)

// ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d3x3Dilation3x3,
//                      Convolution2d3x3Dilation3x3Test<DataType::Float32, DataType::Float32>,
//                      false,
//                      DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d3x3Dilation3x3Nhwc,
//                      Convolution2d3x3Dilation3x3Test<DataType::Float32, DataType::Float32>,
//                      false,
//                      DataLayout::NHWC)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d3x3Dilation3x3Uint8,
//                      Convolution2d3x3Dilation3x3Test<DataType::QAsymmU8, DataType::Signed32>,
//                      false,
//                      DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d3x3Dilation3x3NhwcUint8,
//                      Convolution2d3x3Dilation3x3Test<DataType::QAsymmU8, DataType::Signed32>,
//                      false,
//                      DataLayout::NHWC)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d3x3Dilation3x3Int16,
//                      Convolution2d3x3Dilation3x3Test<DataType::QSymmS16, DataType::Signed32>,
//                      false,
//                      DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d3x3Dilation3x3NhwcInt16,
//                      Convolution2d3x3Dilation3x3Test<DataType::QSymmS16, DataType::Signed32>,
//                      false,
//                      DataLayout::NHWC)

// ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x3x3Dilation3x3,
//                      Convolution2d2x3x3Dilation3x3Test<DataType::Float32, DataType::Float32>,
//                      false,
//                      DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x3x3Dilation3x3Nhwc,
//                      Convolution2d2x3x3Dilation3x3Test<DataType::Float32, DataType::Float32>,
//                      false,
//                      DataLayout::NHWC)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x3x3Dilation3x3Uint8,
//                      Convolution2d2x3x3Dilation3x3Test<DataType::QAsymmU8, DataType::Signed32>,
//                      false,
//                      DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x3x3Dilation3x3NhwcUint8,
//                      Convolution2d2x3x3Dilation3x3Test<DataType::QAsymmU8, DataType::Signed32>,
//                      false,
//                      DataLayout::NHWC)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x3x3Dilation3x3Int16,
//                      Convolution2d2x3x3Dilation3x3Test<DataType::QSymmS16, DataType::Signed32>,
//                      false,
//                      DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x3x3Dilation3x3NhwcInt16,
//                      Convolution2d2x3x3Dilation3x3Test<DataType::QSymmS16, DataType::Signed32>,
//                      false,
//                      DataLayout::NHWC)

// ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x2Dilation2x2Padding2x2Stride3x3,
//                      Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test<DataType::Float32,
//                      DataType::Float32>,
//                      false,
//                      DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x2Dilation2x2Padding2x2Stride3x3Nhwc,
//                      Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test<DataType::Float32,
//                      DataType::Float32>,
//                      false,
//                      DataLayout::NHWC)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x2Dilation2x2Padding2x2Stride3x3Uint8,
//                      Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test<DataType::QAsymmU8,
//                      DataType::Signed32>,
//                      false,
//                      DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x2Dilation2x2Padding2x2Stride3x3NhwcUint8,
//                      Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test<DataType::QAsymmU8,
//                      DataType::Signed32>,
//                      false,
//                      DataLayout::NHWC)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x2Dilation2x2Padding2x2Stride3x3Int16,
//                      Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test<DataType::QSymmS16,
//                      DataType::Signed32>,
//                      false,
//                      DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x2Dilation2x2Padding2x2Stride3x3NhwcInt16,
//                      Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test<DataType::QSymmS16,
//                      DataType::Signed32>,
//                      false,
//                      DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2dPerAxisQuantTestNchw,
                     Convolution2dPerAxisQuantTest,
                     DataLayout::NCHW);
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2dPerAxisQuantTestNhwc,
                     Convolution2dPerAxisQuantTest,
                     DataLayout::NHWC);

// Depthwise Convolution
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2d, DepthwiseConvolution2dTest, true, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dUint8,
                     DepthwiseConvolution2dUint8Test,
                     true,
                     DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedDepthwiseConvolution2d,
                     DepthwiseConvolution2dTest,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedDepthwiseConvolution2dUint8,
                     DepthwiseConvolution2dUint8Test,
                     false,
                     DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dQSymm16, DepthwiseConvolution2dInt16Test, true,
// DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dNhwc, DepthwiseConvolution2dTest, true, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dUint8Nhwc,
                     DepthwiseConvolution2dUint8Test,
                     true,
                     DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedDepthwiseConvolution2dNhwc,
                     DepthwiseConvolution2dTest,
                     false,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedDepthwiseConvolution2dUint8Nhwc,
                     DepthwiseConvolution2dUint8Test,
                     false,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dDepthNhwc, DepthwiseConvolution2dDepthNhwcTest, false)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleDepthwiseConvolution2d3x3Dilation3x3Nhwc,
                     SimpleDepthwiseConvolution2d3x3Dilation3x3NhwcTest)

// ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2d3x3Dilation3x3,
//                      DepthwiseConvolution2d3x3Dilation3x3Test<DataType::Float32,
//                      DataType::Float32>,
//                      false,
//                      DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2d3x3Dilation3x3Nhwc,
//                      DepthwiseConvolution2d3x3Dilation3x3Test<DataType::Float32,
//                      DataType::Float32>,
//                      false,
//                      DataLayout::NHWC)
// ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2d3x3Dilation3x3Uint8,
//                      DepthwiseConvolution2d3x3Dilation3x3Test<DataType::QAsymmU8,
//                      DataType::Signed32>,
//                      false,
//                      DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2d3x3Dilation3x3NhwcUint8,
//                      DepthwiseConvolution2d3x3Dilation3x3Test<DataType::QAsymmU8,
//                      DataType::Signed32>,
//                      false,
//                      DataLayout::NHWC)
// ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2d3x3Dilation3x3Int16,
//                      DepthwiseConvolution2d3x3Dilation3x3Test<DataType::QSymmS16,
//                      DataType::Signed32>,
//                      false,
//                      DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2d3x3Dilation3x3NhwcInt16,
//                      DepthwiseConvolution2d3x3Dilation3x3Test<DataType::QSymmS16,
//                      DataType::Signed32>,
//                      false,
//                      DataLayout::NHWC)

// ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2d2x3x3Dilation3x3,
//                      DepthwiseConvolution2d2x3x3Dilation3x3Test<DataType::Float32,
//                      DataType::Float32>,
//                      false,
//                      DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2d2x3x3Dilation3x3Nhwc,
//                      DepthwiseConvolution2d2x3x3Dilation3x3Test<DataType::Float32,
//                      DataType::Float32>,
//                      false,
//                      DataLayout::NHWC)
// ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2d2x3x3Dilation3x3Uint8,
//                      DepthwiseConvolution2d2x3x3Dilation3x3Test<DataType::QAsymmU8,
//                      DataType::Signed32>,
//                      false,
//                      DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2d2x3x3Dilation3x3NhwcUint8,
//                      DepthwiseConvolution2d2x3x3Dilation3x3Test<DataType::QAsymmU8,
//                      DataType::Signed32>,
//                      false,
//                      DataLayout::NHWC)
// ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2d2x3x3Dilation3x3Int16,
//                      DepthwiseConvolution2d2x3x3Dilation3x3Test<DataType::QSymmS16,
//                      DataType::Signed32>,
//                      false,
//                      DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2d2x3x3Dilation3x3NhwcInt16,
//                      DepthwiseConvolution2d2x3x3Dilation3x3Test<DataType::QSymmS16,
//                      DataType::Signed32>,
//                      false,
//                      DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(
    DepthwiseConvolution2dMult4,
    DepthwiseConvolution2dMult4Test<armnn::DataType::Float32, armnn::DataType::Float32>,
    false,
    armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(
    DepthwiseConvolution2dMult2,
    DepthwiseConvolution2dMult2Test<armnn::DataType::Float32, armnn::DataType::Float32>,
    false,
    armnn::DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dDepthMul1,
                     DepthwiseConvolution2dDepthMul1Test,
                     true,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dDepthMul1Uint8,
                     DepthwiseConvolution2dDepthMul1Uint8Test,
                     true,
                     DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dDepthMul1Int16,
//                      DepthwiseConvolution2dDepthMul1Int16Test, true, DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedDepthwiseConvolution2dDepthMul1,
                     DepthwiseConvolution2dDepthMul1Test,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedDepthwiseConvolution2dDepthMul1Uint8,
                     DepthwiseConvolution2dDepthMul1Uint8Test,
                     false,
                     DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dDepthMul1Nhwc,
                     DepthwiseConvolution2dDepthMul1Test,
                     true,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dDepthMul1Uint8Nhwc,
                     DepthwiseConvolution2dDepthMul1Uint8Test,
                     true,
                     DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedDepthwiseConvolution2dDepthMul1Nhwc,
                     DepthwiseConvolution2dDepthMul1Test,
                     false,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedDepthwiseConvolution2dDepthMul1Uint8Nhwc,
                     DepthwiseConvolution2dDepthMul1Uint8Test,
                     false,
                     DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dAsymmetric,
                     DepthwiseConvolution2dAsymmetricTest,
                     true,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedDepthwiseConvolution2dAsymmetric,
                     DepthwiseConvolution2dAsymmetricTest,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dAsymmetricNhwc,
                     DepthwiseConvolution2dAsymmetricTest,
                     true,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedDepthwiseConvolution2dAsymmetricNhwc,
                     DepthwiseConvolution2dAsymmetricTest,
                     false,
                     DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dDepthMul64, DepthwiseConvolution2dDepthMul64Test);

ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dPerAxisQuantTestNchw,
                     DepthwiseConvolution2dPerAxisQuantTest,
                     DataLayout::NCHW);
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dPerAxisQuantTestNhwc,
                     DepthwiseConvolution2dPerAxisQuantTest,
                     DataLayout::NHWC);

// // Pooling
// //MaxPooling
// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleMaxPooling2dSize2x2Stride2x2, SimpleMaxPooling2dSize2x2Stride2x2Test,
// false)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleMaxPooling2dSize2x2Stride2x2Uint8,
// SimpleMaxPooling2dSize2x2Stride2x2Uint8Test, false)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleMaxPooling2dSize2x2Stride2x2Int16,
// SimpleMaxPooling2dSize2x2Stride2x2Int16Test, false)

// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleMaxPooling2dSize3x3Stride2x4, SimpleMaxPooling2dSize3x3Stride2x4Test,
// false)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleMaxPooling2dSize3x3Stride2x4Uint8,
// SimpleMaxPooling2dSize3x3Stride2x4Uint8Test, false)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleMaxPooling2dSize3x3Stride2x4Int16,
// SimpleMaxPooling2dSize3x3Stride2x4Int16Test, false)

ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleMaxPooling2d, SimpleMaxPooling2dTest, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleMaxPooling2dNhwc, SimpleMaxPooling2dTest, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleMaxPooling2dUint8, SimpleMaxPooling2dUint8Test, DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleMaxPooling2dInt16, SimpleMaxPooling2dInt16Test, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleMaxPooling2dUint8Nhwc, SimpleMaxPooling2dUint8Test, DataLayout::NHWC)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleMaxPooling2dInt16Nhwc, SimpleMaxPooling2dInt16Test, DataLayout::NHWC)

// ARMNN_AUTO_TEST_CASE_WITH_THF(IgnorePaddingSimpleMaxPooling2d, IgnorePaddingSimpleMaxPooling2dTest)
// ARMNN_AUTO_TEST_CASE_WITH_THF(IgnorePaddingSimpleMaxPooling2dUint8,
// IgnorePaddingSimpleMaxPooling2dUint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(IgnorePaddingSimpleMaxPooling2dInt16,
// IgnorePaddingSimpleMaxPooling2dInt16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(IgnorePaddingMaxPooling2dSize3, IgnorePaddingMaxPooling2dSize3Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(IgnorePaddingMaxPooling2dSize3Uint8,
// IgnorePaddingMaxPooling2dSize3Uint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(IgnorePaddingMaxPooling2dSize3Int16,
// IgnorePaddingMaxPooling2dSize3Int16Test)

// AveragePooling
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleAveragePooling2d, SimpleAveragePooling2dTest, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleAveragePooling2dNhwc, SimpleAveragePooling2dTest, DataLayout::NHWC)
// Not support zp = -1
// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleAveragePooling2dUint8, SimpleAveragePooling2dUint8Test, DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleAveragePooling2dInt16, SimpleAveragePooling2dInt16Test,
// DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleAveragePooling2dUint8Nhwc,
//                      SimpleAveragePooling2dUint8Test,
//                      DataLayout::NHWC)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleAveragePooling2dInt16Nhwc, SimpleAveragePooling2dInt16Test,
// DataLayout::NHWC)

// ARMNN_AUTO_TEST_CASE_WITH_THF(IgnorePaddingSimpleAveragePooling2d,
// IgnorePaddingSimpleAveragePooling2dTest)
// ARMNN_AUTO_TEST_CASE_WITH_THF(IgnorePaddingSimpleAveragePooling2dUint8,
// IgnorePaddingSimpleAveragePooling2dUint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(IgnorePaddingSimpleAveragePooling2dInt16,
// IgnorePaddingSimpleAveragePooling2dInt16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(IgnorePaddingSimpleAveragePooling2dNoPadding,
// IgnorePaddingSimpleAveragePooling2dNoPaddingTest)
// ARMNN_AUTO_TEST_CASE_WITH_THF(IgnorePaddingSimpleAveragePooling2dNoPaddingUint8,
//                      IgnorePaddingSimpleAveragePooling2dNoPaddingUint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(IgnorePaddingSimpleAveragePooling2dNoPaddingInt16,
//                      IgnorePaddingSimpleAveragePooling2dNoPaddingInt16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(IgnorePaddingAveragePooling2dSize3, IgnorePaddingAveragePooling2dSize3Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(IgnorePaddingAveragePooling2dSize3Uint8,
// IgnorePaddingAveragePooling2dSize3Uint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(IgnorePaddingAveragePooling2dSize3Int16,
// IgnorePaddingAveragePooling2dSize3Int16Test)

// ARMNN_AUTO_TEST_CASE_WITH_THF(IgnorePaddingAveragePooling2dSize3x2Stride2x2,
//                      IgnorePaddingAveragePooling2dSize3x2Stride2x2Test, false)
// ARMNN_AUTO_TEST_CASE_WITH_THF(IgnorePaddingAveragePooling2dSize3x2Stride2x2NoPadding,
//                      IgnorePaddingAveragePooling2dSize3x2Stride2x2Test, true)

ARMNN_AUTO_TEST_CASE_WITH_THF(LargeTensorsAveragePooling2d, LargeTensorsAveragePooling2dTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(LargeTensorsAveragePooling2dUint8, LargeTensorsAveragePooling2dUint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(LargeTensorsAveragePooling2dInt16, LargeTensorsAveragePooling2dInt16Test)

// //L2Pooling
// ARMNN_AUTO_TEST_CASE_WITH_THF(IgnorePaddingSimpleL2Pooling2d, IgnorePaddingSimpleL2Pooling2dTest)
// ARMNN_AUTO_TEST_CASE_WITH_THF(IgnorePaddingSimpleL2Pooling2dUint8,
// IgnorePaddingSimpleL2Pooling2dUint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(IgnorePaddingSimpleL2Pooling2dInt16,
// IgnorePaddingSimpleL2Pooling2dInt16Test)

// ARMNN_AUTO_TEST_CASE_WITH_THF(IgnorePaddingL2Pooling2dSize3, IgnorePaddingL2Pooling2dSize3Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(IgnorePaddingL2Pooling2dSize3Uint8, IgnorePaddingL2Pooling2dSize3Uint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(IgnorePaddingL2Pooling2dSize3Int16, IgnorePaddingL2Pooling2dSize3Int16Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleL2Pooling2d, SimpleL2Pooling2dTest, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleL2Pooling2dNhwc, SimpleL2Pooling2dTest, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleL2Pooling2dUint8, SimpleL2Pooling2dUint8Test, DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleL2Pooling2dInt16, SimpleL2Pooling2dInt16Test, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleL2Pooling2dNhwcUint8, SimpleL2Pooling2dUint8Test, DataLayout::NHWC)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleL2Pooling2dNhwcInt16, SimpleL2Pooling2dInt16Test, DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(L2Pooling2dSize7, L2Pooling2dSize7Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(L2Pooling2dSize7Uint8, L2Pooling2dSize7Uint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(L2Pooling2dSize7Int16, L2Pooling2dSize7Int16Test)

// //NonSquarePooling
// ARMNN_AUTO_TEST_CASE_WITH_THF(AsymmNonSquarePooling2d, AsymmetricNonSquarePooling2dTest)
// ARMNN_AUTO_TEST_CASE_WITH_THF(AsymmNonSquarePooling2dUint8, AsymmetricNonSquarePooling2dUint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(AsymmNonSquarePooling2dInt16, AsymmetricNonSquarePooling2dInt16Test)

// // Linear Activation
ARMNN_AUTO_TEST_CASE_WITH_THF(ConstantLinearActivation, ConstantLinearActivationTest)
// ARMNN_AUTO_TEST_CASE_WITH_THF(ConstantLinearActivationUint8, ConstantLinearActivationUint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(ConstantLinearActivationInt16, ConstantLinearActivationInt16Test)

// InstanceNormalization
ARMNN_AUTO_TEST_CASE_WITH_THF(InstanceNormFloat32Nchw, InstanceNormFloat32Test, DataLayout::NCHW);
ARMNN_AUTO_TEST_CASE_WITH_THF(InstanceNormFloat16Nchw, InstanceNormFloat16Test, DataLayout::NCHW);

ARMNN_AUTO_TEST_CASE_WITH_THF(InstanceNormFloat32Nhwc, InstanceNormFloat32Test, DataLayout::NHWC);
ARMNN_AUTO_TEST_CASE_WITH_THF(InstanceNormFloat16Nhwc, InstanceNormFloat16Test, DataLayout::NHWC);

ARMNN_AUTO_TEST_CASE_WITH_THF(InstanceNormFloat32Nchw2, InstanceNormFloat32Test2, DataLayout::NCHW);
ARMNN_AUTO_TEST_CASE_WITH_THF(InstanceNormFloat16Nchw2, InstanceNormFloat16Test2, DataLayout::NCHW);

ARMNN_AUTO_TEST_CASE_WITH_THF(InstanceNormFloat32Nhwc2, InstanceNormFloat32Test2, DataLayout::NHWC);
ARMNN_AUTO_TEST_CASE_WITH_THF(InstanceNormFloat16Nhwc2, InstanceNormFloat16Test2, DataLayout::NHWC);

// // Normalization
// Note: methodType = LocalBrightness in all 3 uts
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleNormalizationAcross, SimpleNormalizationAcrossTest)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleNormalizationWithin, SimpleNormalizationWithinTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleNormalizationAcrossNhwc, SimpleNormalizationAcrossNhwcTest)

// // Softmax
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleSoftmaxBeta1, SimpleSoftmaxTest, 1.0f)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleSoftmaxBeta2, SimpleSoftmaxTest, 2.0f)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleSoftmaxBeta1Uint8, SimpleSoftmaxUint8Test, 1.0f)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleSoftmaxBeta2Uint8, SimpleSoftmaxUint8Test, 2.0f)

ARMNN_AUTO_TEST_CASE_WITH_THF(Simple3dSoftmax, Simple3dSoftmaxTest, 1.0f)
ARMNN_AUTO_TEST_CASE_WITH_THF(Simple3dSoftmaxUint8, Simple3dSoftmaxUint8Test, 1.0f)

ARMNN_AUTO_TEST_CASE_WITH_THF(Simple4dSoftmax, Simple4dSoftmaxTest, 1.0f)
ARMNN_AUTO_TEST_CASE_WITH_THF(Simple4dSoftmaxUint8, Simple4dSoftmaxUint8Test, 1.0f)

// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleSoftmaxFloat16, SimpleSoftmaxFloat16Test, 1.0f)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Simple3dSoftmaxFloat16, Simple3dSoftmaxFloat16Test, 1.0f)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Simple4dSoftmaxFloat16, Simple4dSoftmaxFloat16Test, 1.0f)

// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleSoftmaxUint16, SimpleSoftmaxUint16Test, 1.0f)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Simple3dSoftmaxUint16, Simple3dSoftmaxUint16Test, 1.0f)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Simple4dSoftmaxUint16, Simple4dSoftmaxUint16Test, 1.0f)

ARMNN_AUTO_TEST_CASE_WITH_THF(Simple2dAxis0Softmax, SimpleAxisSoftmaxTest, 1.0f, 0)
ARMNN_AUTO_TEST_CASE_WITH_THF(Simple2dAxis1Softmax, SimpleAxisSoftmaxTest, 1.0f, 1)

ARMNN_AUTO_TEST_CASE_WITH_THF(Simple2dAxis0NegSoftmax, SimpleAxisSoftmaxTest, 1.0f, -2)
ARMNN_AUTO_TEST_CASE_WITH_THF(Simple2dAxis1NegSoftmax, SimpleAxisSoftmaxTest, 1.0f, -1)

ARMNN_AUTO_TEST_CASE_WITH_THF(Simple3dAxis0Softmax, Simple3dAxisSoftmaxTest, 1.0f, 0)
ARMNN_AUTO_TEST_CASE_WITH_THF(Simple3dAxis1Softmax, Simple3dAxisSoftmaxTest, 1.0f, 1)
ARMNN_AUTO_TEST_CASE_WITH_THF(Simple3dAxis2Softmax, Simple3dAxisSoftmaxTest, 1.0f, 2)

ARMNN_AUTO_TEST_CASE_WITH_THF(Simple3dAxis0NegSoftmax, Simple3dAxisSoftmaxTest, 1.0f, -3)
ARMNN_AUTO_TEST_CASE_WITH_THF(Simple3dAxis1NegSoftmax, Simple3dAxisSoftmaxTest, 1.0f, -2)
ARMNN_AUTO_TEST_CASE_WITH_THF(Simple3dAxis2NegSoftmax, Simple3dAxisSoftmaxTest, 1.0f, -1)

ARMNN_AUTO_TEST_CASE_WITH_THF(Simple4dAxis0Softmax, Simple4dAxisSoftmaxTest, 1.0f, 0)
ARMNN_AUTO_TEST_CASE_WITH_THF(Simple4dAxis1Softmax, Simple4dAxisSoftmaxTest, 1.0f, 1)
ARMNN_AUTO_TEST_CASE_WITH_THF(Simple4dAxis2Softmax, Simple4dAxisSoftmaxTest, 1.0f, 2)
ARMNN_AUTO_TEST_CASE_WITH_THF(Simple4dAxis3Softmax, Simple4dAxisSoftmaxTest, 1.0f, 3)

ARMNN_AUTO_TEST_CASE_WITH_THF(Simple4dAxis0NegSoftmax, Simple4dAxisSoftmaxTest, 1.0f, -4)
ARMNN_AUTO_TEST_CASE_WITH_THF(Simple4dAxis1NegSoftmax, Simple4dAxisSoftmaxTest, 1.0f, -3)
ARMNN_AUTO_TEST_CASE_WITH_THF(Simple4dAxis2NegSoftmax, Simple4dAxisSoftmaxTest, 1.0f, -2)
ARMNN_AUTO_TEST_CASE_WITH_THF(Simple4dAxis3NegSoftmax, Simple4dAxisSoftmaxTest, 1.0f, -1)

// // Sigmoid Activation
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleSigmoid, SimpleSigmoidTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleSigmoidUint8, SimpleSigmoidUint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleSigmoidInt16, SimpleSigmoidInt16Test)

// // BoundedReLU Activation
ARMNN_AUTO_TEST_CASE_WITH_THF(ReLu1, BoundedReLuUpperAndLowerBoundTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(ReLu6, BoundedReLuUpperBoundOnlyTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(ReLu1Uint8, BoundedReLuUint8UpperAndLowerBoundTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(ReLu6Uint8, BoundedReLuUint8UpperBoundOnlyTest)
// ARMNN_AUTO_TEST_CASE_WITH_THF(BoundedReLuInt16, BoundedReLuInt16Test)

// // ReLU Activation
ARMNN_AUTO_TEST_CASE_WITH_THF(ReLu, ReLuTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(ReLuUint8, ReLuUint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(ReLuInt16, ReLuInt16Test)

// // SoftReLU Activation
ARMNN_AUTO_TEST_CASE_WITH_THF(SoftReLu, SoftReLuTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(SoftReLuUint8, SoftReLuUint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SoftReLuInt16, SoftReLuInt16Test)

// // LeakyReLU Activation
ARMNN_AUTO_TEST_CASE_WITH_THF(LeakyReLu, LeakyReLuTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(LeakyReLuUint8, LeakyReLuUint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(LeakyReLuInt16, LeakyReLuInt16Test)

// // Abs Activation
ARMNN_AUTO_TEST_CASE_WITH_THF(Abs, AbsTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(AbsUint8, AbsUint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(AbsInt16, AbsInt16Test)

// // Sqrt Activation
ARMNN_AUTO_TEST_CASE_WITH_THF(Sqrt, SqrtTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(SqrtNN, SqrtNNTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(SqrtUint8, SqrtUint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SqrtInt16, SqrtInt16Test)

// // Square Activation
ARMNN_AUTO_TEST_CASE_WITH_THF(Square, SquareTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(SquareUint8, SquareUint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SquareInt16, SquareInt16Test)

// // Tanh Activation
ARMNN_AUTO_TEST_CASE_WITH_THF(Tanh, TanhTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(TanhUint8, TanhUint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(TanhInt16, TanhInt16Test)

// // Fully Connected
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleFullyConnected, FullyConnectedFloat32Test, false, false)
// ARMNN_AUTO_TEST_CASE_WITH_THF(FullyConnectedUint8, FullyConnectedTest<DataType::QAsymmU8>, false)
// ARMNN_AUTO_TEST_CASE_WITH_THF(FullyConnectedQSymm16, FullyConnectedTest<DataType::QSymmS16>, false)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleFullyConnectedWithBias, FullyConnectedFloat32Test, true, false)
// ARMNN_AUTO_TEST_CASE_WITH_THF(FullyConnectedBiasedUint8, FullyConnectedTest<DataType::QAsymmU8>, true)
// ARMNN_AUTO_TEST_CASE_WITH_THF(FullyConnectedBiasedQSymm16, FullyConnectedTest<DataType::QSymmS16>, true)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleFullyConnectedWithTranspose, FullyConnectedFloat32Test, false, true)

// ARMNN_AUTO_TEST_CASE_WITH_THF(FullyConnectedLarge, FullyConnectedLargeTest, false)
// ARMNN_AUTO_TEST_CASE_WITH_THF(FullyConnectedLargeTransposed, FullyConnectedLargeTest, true)

// // Splitter
// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleSplitterFloat32, SplitterFloat32Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleSplitterFloat16, SplitterFloat16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleSplitterUint8, SplitterUint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleSplitterInt16, SplitterInt16Test)

// ARMNN_AUTO_TEST_CASE_WITH_THF(CopyViaSplitterFloat32, CopyViaSplitterFloat32Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(CopyViaSplitterFloat16, CopyViaSplitterFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(CopyViaSplitterUint8, CopyViaSplitterUint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(CopyViaSplitterInt16, CopyViaSplitterInt16Test)

// // Concat
// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConcat, ConcatTest)
// ARMNN_AUTO_TEST_CASE_WITH_THF(ConcatFloat16, ConcatFloat16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(ConcatUint8, ConcatUint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(ConcatUint8DifferentQParams, ConcatUint8DifferentQParamsTest)
// ARMNN_AUTO_TEST_CASE_WITH_THF(ConcatUint16, ConcatUint16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(ConcatUint8DifferentInputOutputQParam,
//                      ConcatDifferentInputOutputQParamTest<DataType::QAsymmU8>, true)
// ARMNN_AUTO_TEST_CASE_WITH_THF(ConcatInt16DifferentInputOutputQParam,
//                      ConcatDifferentInputOutputQParamTest<DataType::QSymmS16>, true)

// Add
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleAdd, AdditionTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(Add5d, Addition5dTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(AddBroadcast1Element, AdditionBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(AddBroadcast, AdditionBroadcastTest)

ARMNN_AUTO_TEST_CASE_WITH_THF(AdditionUint8, AdditionUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(AddBroadcastUint8, AdditionBroadcastUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(AddBroadcast1ElementUint8, AdditionBroadcast1ElementUint8Test)

// ARMNN_AUTO_TEST_CASE_WITH_THF(AdditionInt16, AdditionInt16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(AddBroadcastInt16, AdditionBroadcastInt16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(AddBroadcast1ElementInt16, AdditionBroadcast1ElementInt16Test)

// Sub
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleSub, SubtractionTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(SubBroadcast1Element, SubtractionBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(SubBroadcast, SubtractionBroadcastTest)

ARMNN_AUTO_TEST_CASE_WITH_THF(SubtractionUint8, SubtractionUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SubBroadcastUint8, SubtractionBroadcastUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SubBroadcast1ElementUint8, SubtractionBroadcast1ElementUint8Test)

// ARMNN_AUTO_TEST_CASE_WITH_THF(SubtractionInt16, SubtractionInt16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SubBroadcastInt16, SubtractionBroadcastInt16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SubBroadcast1ElementInt16, SubtractionBroadcast1ElementInt16Test)

// // Div
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleDivision, DivisionTest)
// TODO: Shared implementation for Div not support Zero
// ARMNN_AUTO_TEST_CASE_WITH_THF(DivisionByZero, DivisionByZeroTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(DivisionBroadcast1Element, DivisionBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(DivisionBroadcast1DVector, DivisionBroadcast1DVectorTest)

// ARMNN_AUTO_TEST_CASE_WITH_THF(DivisionFloat16, DivisionFloat16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(DivisionFloat16Broadcast1Element, DivisionBroadcast1ElementFloat16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(DivisionFloat16Broadcast1DVector, DivisionBroadcast1DVectorFloat16Test)

// NOTE: division by zero for quantized div needs more attention
//       see IVGCVSW-1849
ARMNN_AUTO_TEST_CASE_WITH_THF(DivisionUint8, DivisionUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(DivisionUint8Broadcast1Element, DivisionBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(DivisionUint8Broadcast1DVector, DivisionBroadcast1DVectorUint8Test)

// ARMNN_AUTO_TEST_CASE_WITH_THF(DivisionInt16, DivisionInt16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(DivisionInt16Broadcast1Element, DivisionBroadcast1ElementInt16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(DivisionInt16Broadcast1DVector, DivisionBroadcast1DVectorInt16Test)

// Equal
ARMNN_AUTO_TEST_CASE_WITH_THF(EqualSimple,            EqualSimpleTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(EqualBroadcast1Element, EqualBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(EqualBroadcast1dVector, EqualBroadcast1dVectorTest)

ARMNN_AUTO_TEST_CASE_WITH_THF(EqualSimpleFloat16,            EqualSimpleFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(EqualBroadcast1ElementFloat16, EqualBroadcast1ElementFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(EqualBroadcast1dVectorFloat16, EqualBroadcast1dVectorFloat16Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(EqualSimpleUint8,            EqualSimpleUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(EqualBroadcast1ElementUint8, EqualBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(EqualBroadcast1dVectorUint8, EqualBroadcast1dVectorUint8Test)

// Greater
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterSimple,            GreaterSimpleTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterBroadcast1Element, GreaterBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterBroadcast1dVector, GreaterBroadcast1dVectorTest)

ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterSimpleFloat16,            GreaterSimpleFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterBroadcast1ElementFloat16, GreaterBroadcast1ElementFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterBroadcast1dVectorFloat16, GreaterBroadcast1dVectorFloat16Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterSimpleUint8,            GreaterSimpleUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterBroadcast1ElementUint8, GreaterBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterBroadcast1dVectorUint8, GreaterBroadcast1dVectorUint8Test)

// GreaterOrEqual
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterOrEqualSimple, GreaterOrEqualSimpleTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterOrEqualBroadcast1Element, GreaterOrEqualBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterOrEqualBroadcast1dVector, GreaterOrEqualBroadcast1dVectorTest)

ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterOrEqualSimpleFloat16, GreaterOrEqualSimpleFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterOrEqualBroadcast1ElementFloat16,
                     GreaterOrEqualBroadcast1ElementFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterOrEqualBroadcast1dVectorFloat16,
                     GreaterOrEqualBroadcast1dVectorFloat16Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterOrEqualSimpleUint8, GreaterOrEqualSimpleUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterOrEqualBroadcast1ElementUint8, GreaterOrEqualBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterOrEqualBroadcast1dVectorUint8, GreaterOrEqualBroadcast1dVectorUint8Test)

// Less
ARMNN_AUTO_TEST_CASE_WITH_THF(LessSimple, LessSimpleTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(LessBroadcast1Element, LessBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(LessBroadcast1dVector, LessBroadcast1dVectorTest)

ARMNN_AUTO_TEST_CASE_WITH_THF(LessSimpleFloat16, LessSimpleFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(LessBroadcast1ElementFloat16, LessBroadcast1ElementFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(LessBroadcast1dVectorFloat16, LessBroadcast1dVectorFloat16Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(LessSimpleUint8, LessSimpleUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(LessBroadcast1ElementUint8, LessBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(LessBroadcast1dVectorUint8, LessBroadcast1dVectorUint8Test)

// LessOrEqual
ARMNN_AUTO_TEST_CASE_WITH_THF(LessOrEqualSimple, LessOrEqualSimpleTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(LessOrEqualBroadcast1Element, LessOrEqualBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(LessOrEqualBroadcast1dVector, LessOrEqualBroadcast1dVectorTest)

ARMNN_AUTO_TEST_CASE_WITH_THF(LessOrEqualSimpleFloat16, LessOrEqualSimpleFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(LessOrEqualBroadcast1ElementFloat16, LessOrEqualBroadcast1ElementFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(LessOrEqualBroadcast1dVectorFloat16, LessOrEqualBroadcast1dVectorFloat16Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(LessOrEqualSimpleUint8, LessOrEqualSimpleUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(LessOrEqualBroadcast1ElementUint8, LessOrEqualBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(LessOrEqualBroadcast1dVectorUint8, LessOrEqualBroadcast1dVectorUint8Test)

// NotEqual
ARMNN_AUTO_TEST_CASE_WITH_THF(NotEqualSimple,            NotEqualSimpleTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(NotEqualBroadcast1Element, NotEqualBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(NotEqualBroadcast1dVector, NotEqualBroadcast1dVectorTest)

ARMNN_AUTO_TEST_CASE_WITH_THF(NotEqualSimpleFloat16,            NotEqualSimpleFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(NotEqualBroadcast1ElementFloat16, NotEqualBroadcast1ElementFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(NotEqualBroadcast1dVectorFloat16, NotEqualBroadcast1dVectorFloat16Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(NotEqualSimpleUint8,            NotEqualSimpleUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(NotEqualBroadcast1ElementUint8, NotEqualBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(NotEqualBroadcast1dVectorUint8, NotEqualBroadcast1dVectorUint8Test)

// Max
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleMaximum, MaximumSimpleTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(MaximumBroadcast1Element, MaximumBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(MaximumBroadcast1DVector, MaximumBroadcast1DVectorTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(MaximumFloat16, MaximumFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(MaximumBroadcast1ElementFloat16, MaximumBroadcast1ElementFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(MaximumBroadcast1DVectorFloat16, MaximumBroadcast1DVectorFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(MaximumUint8, MaximumUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(MaximumBroadcast1ElementUint8, MaximumBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(MaximumBroadcast1DVectorUint8, MaximumBroadcast1DVectorUint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(MaximumInt16, MaximumInt16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(MaximumBroadcast1ElementInt16, MaximumBroadcast1ElementInt16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(MaximumBroadcast1DVectorInt16, MaximumBroadcast1DVectorInt16Test)

// Min
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleMinimum1, MinimumBroadcast1ElementTest1)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleMinimum2, MinimumBroadcast1ElementTest2)
ARMNN_AUTO_TEST_CASE_WITH_THF(Minimum1DVectorUint8, MinimumBroadcast1DVectorUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(MinimumFloat16, MinimumFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(MinimumBroadcast1ElementFloat16, MinimumBroadcast1ElementFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(MinimumBroadcast1DVectorFloat16, MinimumBroadcast1DVectorFloat16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(MinimumInt16, MinimumInt16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(MinimumBroadcast1ElementInt16, MinimumBroadcast1ElementInt16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(MinimumBroadcast1DVectorInt16, MinimumBroadcast1DVectorInt16Test)

// Mul
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleMultiplication, MultiplicationTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(MultiplicationBroadcast1Element, MultiplicationBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(MultiplicationBroadcast1DVector, MultiplicationBroadcast1DVectorTest)
// ARMNN_AUTO_TEST_CASE_WITH_THF(MultiplicationUint8, MultiplicationUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(MultiplicationBroadcast1ElementUint8, MultiplicationBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(MultiplicationBroadcast1DVectorUint8, MultiplicationBroadcast1DVectorUint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(MultiplicationInt16, MultiplicationInt16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(MultiplicationBroadcast1ElementInt16,
// MultiplicationBroadcast1ElementInt16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(MultiplicationBroadcast1DVectorInt16,
// MultiplicationBroadcast1DVectorInt16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Multiplication5d, Multiplication5dTest)

// Batch Norm
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchNormFloat32, BatchNormFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchNormFloat32Nhwc, BatchNormFloat32NhwcTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchNormFloat16, BatchNormFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchNormFloat16Nhwc, BatchNormFloat16NhwcTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchNormUint8, BatchNormUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchNormUint8Nhwc, BatchNormUint8NhwcTest)
// ARMNN_AUTO_TEST_CASE_WITH_THF(BatchNormInt16, BatchNormInt16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(BatchNormInt16Nhwc, BatchNormInt16NhwcTest)

// Resize Bilinear - NCHW
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleResizeBilinear,
                     SimpleResizeBilinearTest<DataType::Float32>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleResizeBilinearFloat16,
                     SimpleResizeBilinearTest<DataType::Float16>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleResizeBilinearUint8,
                     SimpleResizeBilinearTest<DataType::QAsymmU8>,
                     DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleResizeBilinearUint16,
//                      SimpleResizeBilinearTest<DataType::QSymmS16>,
//                      DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearNop,
                     ResizeBilinearNopTest<DataType::Float32>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearNopFloat16,
                     ResizeBilinearNopTest<DataType::Float16>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearNopUint8,
                     ResizeBilinearNopTest<DataType::QAsymmU8>,
                     DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(esizeBilinearNopUint16,
//                      SimpleResizeBilinearTest<DataType::QSymmS16>,
//                      DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearSqMin,
                     ResizeBilinearSqMinTest<DataType::Float32>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearSqMinFloat16,
                     ResizeBilinearSqMinTest<DataType::Float16>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearSqMinUint8,
                     ResizeBilinearSqMinTest<DataType::QAsymmU8>,
                     DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearSqMinUint16,
//                      SimpleResizeBilinearTest<DataType::QSymmS16>,
//                      DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearMin,
                     ResizeBilinearMinTest<DataType::Float32>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearMinFloat16,
                     ResizeBilinearMinTest<DataType::Float16>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearMinUint8,
                     ResizeBilinearMinTest<DataType::QAsymmU8>,
                     DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearMinUint16,
//                      SimpleResizeBilinearTest<DataType::QSymmS16>,
//                      DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearMag,
                     ResizeBilinearMagTest<DataType::Float32>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearMagFloat16,
                     ResizeBilinearMagTest<DataType::Float16>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearMagUint8,
                     ResizeBilinearMagTest<DataType::QAsymmU8>,
                     DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearMagUint16,
//                      SimpleResizeBilinearTest<DataType::QSymmS16>,
//                      DataLayout::NCHW)

// Resize Bilinear - NHWC
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearNopNhwc,
                     ResizeBilinearNopTest<DataType::Float32>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearNopNhwcFloat16,
                     ResizeBilinearNopTest<DataType::Float16>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearNopUint8Nhwc,
                     ResizeBilinearNopTest<DataType::QAsymmU8>,
                     DataLayout::NHWC)
// ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearNopUint16Nhwc,
//                      ResizeBilinearNopTest<DataType::QSymmS16>,
//                      DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleResizeBilinearNhwc,
                     SimpleResizeBilinearTest<DataType::Float32>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleResizeBilinearNhwcFloat16,
                     SimpleResizeBilinearTest<DataType::Float16>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleResizeBilinearUint8Nhwc,
                     SimpleResizeBilinearTest<DataType::QAsymmU8>,
                     DataLayout::NHWC)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleResizeBilinearUint16Nhwc,
//                      ResizeBilinearNopTest<DataType::QSymmS16>,
//                      DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearSqMinNhwc,
                     ResizeBilinearSqMinTest<DataType::Float32>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearSqMinNhwcFloat16,
                     ResizeBilinearSqMinTest<DataType::Float16>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearSqMinUint8Nhwc,
                     ResizeBilinearSqMinTest<DataType::QAsymmU8>,
                     DataLayout::NHWC)
// ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearSqMinUint16Nhwc,
//                      ResizeBilinearNopTest<DataType::QSymmS16>,
//                      DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearMinNhwc,
                     ResizeBilinearMinTest<DataType::Float32>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearMinNhwcFloat16,
                     ResizeBilinearMinTest<DataType::Float16>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearMinUint8Nhwc,
                     ResizeBilinearMinTest<DataType::QAsymmU8>,
                     DataLayout::NHWC)
// ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearMinUint16Nhwc,
//                      ResizeBilinearNopTest<DataType::QSymmS16>,
//                      DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearMagNhwc,
                     ResizeBilinearMagTest<DataType::Float32>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearMagNhwcFloat16,
                     ResizeBilinearMagTest<DataType::Float16>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearMagUint8Nhwc,
                     ResizeBilinearMagTest<DataType::QAsymmU8>,
                     DataLayout::NHWC)
// ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearMagUint16Nhwc,
//                      ResizeBilinearNopTest<DataType::QSymmS16>,
//                      DataLayout::NHWC)

// Resize NearestNeighbor - NCHW
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleResizeNearestNeighbor,
                     SimpleResizeNearestNeighborTest<DataType::Float32>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleResizeNearestNeighborUint8,
                     SimpleResizeNearestNeighborTest<DataType::QAsymmU8>,
                     DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleResizeNearestNeighborUint16,
//                      SimpleResizeNearestNeighborTest<DataType::QSymmS16>,
//                      DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborNop,
                     ResizeNearestNeighborNopTest<DataType::Float32>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborNopUint8,
                     ResizeNearestNeighborNopTest<DataType::QAsymmU8>,
                     DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(esizeNearestNeighborNopUint16,
//                      SimpleResizeNearestNeighborTest<DataType::QSymmS16>,
//                      DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborSqMin,
                     ResizeNearestNeighborSqMinTest<DataType::Float32>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborSqMinUint8,
                     ResizeNearestNeighborSqMinTest<DataType::QAsymmU8>,
                     DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborSqMinUint16,
//                      SimpleResizeNearestNeighborTest<DataType::QSymmS16>,
//                      DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborMin,
                     ResizeNearestNeighborMinTest<DataType::Float32>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborMinUint8,
                     ResizeNearestNeighborMinTest<DataType::QAsymmU8>,
                     DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborMinUint16,
//                      SimpleResizeNearestNeighborTest<DataType::QSymmS16>,
//                      DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborMag,
                     ResizeNearestNeighborMagTest<DataType::Float32>,
                     DataLayout::NCHW, 0.10f, 50, 0.11f, 20)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborMagUint8,
                     ResizeNearestNeighborMagTest<DataType::QAsymmU8>,
                     DataLayout::NCHW, 0.10f, 50, 0.11f, 20)
// ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborMagUint16,
//                      SimpleResizeNearestNeighborTest<DataType::QSymmS16>,
//                      DataLayout::NCHW)

// Resize NearestNeighbor - NHWC
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborNopNhwc,
                     ResizeNearestNeighborNopTest<DataType::Float32>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborNopUint8Nhwc,
                     ResizeNearestNeighborNopTest<DataType::QAsymmU8>,
                     DataLayout::NHWC)
// ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborNopUint16Nhwc,
//                      ResizeNearestNeighborNopTest<DataType::QSymmS16>,
//                      DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleResizeNearestNeighborNhwc,
                     SimpleResizeNearestNeighborTest<DataType::Float32>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleResizeNearestNeighborUint8Nhwc,
                     SimpleResizeNearestNeighborTest<DataType::QAsymmU8>,
                     DataLayout::NHWC)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleResizeNearestNeighborUint16Nhwc,
//                      ResizeNearestNeighborNopTest<DataType::QSymmS16>,
//                      DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborSqMinNhwc,
                     ResizeNearestNeighborSqMinTest<DataType::Float32>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborSqMinUint8Nhwc,
                     ResizeNearestNeighborSqMinTest<DataType::QAsymmU8>,
                     DataLayout::NHWC)
// ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborSqMinUint16Nhwc,
//                      ResizeNearestNeighborNopTest<DataType::QSymmS16>,
//                      DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborMinNhwc,
                     ResizeNearestNeighborMinTest<DataType::Float32>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborMinUint8Nhwc,
                     ResizeNearestNeighborMinTest<DataType::QAsymmU8>,
                     DataLayout::NHWC)
// ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborMinUint16Nhwc,
//                      ResizeNearestNeighborNopTest<DataType::QSymmS16>,
//                      DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborMagNhwc,
                     ResizeNearestNeighborMagTest<DataType::Float32>,
                     DataLayout::NHWC, 0.10f, 50, 0.11f, 20)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborMagUint8Nhwc,
                     ResizeNearestNeighborMagTest<DataType::QAsymmU8>,
                     DataLayout::NHWC, 0.10f, 50, 0.11f, 20)
// ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborMagUint16Nhwc,
//                      ResizeNearestNeighborNopTest<DataType::QSymmS16>,
//                      DataLayout::NHWC)

// // Fake Quantization
// ARMNN_AUTO_TEST_CASE_WITH_THF(FakeQuantization, FakeQuantizationTest)

// L2 Normalization
ARMNN_AUTO_TEST_CASE_WITH_THF(L2Normalization1d, L2Normalization1dTest, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(L2Normalization2d, L2Normalization2dTest, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(L2Normalization3d, L2Normalization3dTest, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(L2Normalization4d, L2Normalization4dTest, DataLayout::NCHW)

// ARMNN_AUTO_TEST_CASE_WITH_THF(L2Normalization1dInt16, L2Normalization1dInt16Test, DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(L2Normalization2dInt16, L2Normalization2dInt16Test, DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(L2Normalization3dInt16, L2Normalization3dInt16Test, DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(L2Normalization4dInt16, L2Normalization4dInt16Test, DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE_WITH_THF(L2Normalization1dUint8, L2Normalization1dUint8Test, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(L2Normalization2dUint8, L2Normalization2dUint8Test, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(L2Normalization3dUint8, L2Normalization3dUint8Test, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(L2Normalization4dUint8, L2Normalization4dUint8Test, DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE_WITH_THF(L2Normalization1dNhwc, L2Normalization1dTest, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(L2Normalization2dNhwc, L2Normalization2dTest, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(L2Normalization3dNhwc, L2Normalization3dTest, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(L2Normalization4dNhwc, L2Normalization4dTest, DataLayout::NHWC)

// ARMNN_AUTO_TEST_CASE_WITH_THF(L2Normalization1dInt16Nhwc, L2Normalization1dInt16Test, DataLayout::NHWC)
// ARMNN_AUTO_TEST_CASE_WITH_THF(L2Normalization2dInt16Nhwc, L2Normalization2dInt16Test, DataLayout::NHWC)
// ARMNN_AUTO_TEST_CASE_WITH_THF(L2Normalization3dInt16Nhwc, L2Normalization3dInt16Test, DataLayout::NHWC)
// ARMNN_AUTO_TEST_CASE_WITH_THF(L2Normalization4dInt16Nhwc, L2Normalization4dInt16Test, DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(L2Normalization1dUint8Nhwc, L2Normalization1dUint8Test, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(L2Normalization2dUint8Nhwc, L2Normalization2dUint8Test, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(L2Normalization3dUint8Nhwc, L2Normalization3dUint8Test, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(L2Normalization4dUint8Nhwc, L2Normalization4dUint8Test, DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(L2Normalization2dShape, L2Normalization2dShapeTest);

// ARMNN_AUTO_TEST_CASE_WITH_THF(L2NormalizationDefaultEpsilon, L2NormalizationDefaultEpsilonTest,
// DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(L2NormalizationNonDefaultEpsilon, L2NormalizationNonDefaultEpsilonTest,
// DataLayout::NCHW)

// LogSoftmax
ARMNN_AUTO_TEST_CASE_WITH_THF(LogSoftmaxFloat32_1, LogSoftmaxTest1<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(LogSoftmaxFloat32_2, LogSoftmaxTest2<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(LogSoftmaxFloat32_3, LogSoftmaxTest3<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(LogSoftmaxFloat32_4, LogSoftmaxTest4<DataType::Float32>)

ARMNN_AUTO_TEST_CASE_WITH_THF(LogSoftmaxFloat16_1, LogSoftmaxTest1<DataType::Float16>)
ARMNN_AUTO_TEST_CASE_WITH_THF(LogSoftmaxFloat16_2, LogSoftmaxTest2<DataType::Float16>)
ARMNN_AUTO_TEST_CASE_WITH_THF(LogSoftmaxFloat16_3, LogSoftmaxTest3<DataType::Float16>)
ARMNN_AUTO_TEST_CASE_WITH_THF(LogSoftmaxFloat16_4, LogSoftmaxTest4<DataType::Float16>)

// Pad
// Note: Pad2d vsimulator can pass, but hardware cannot pass
// ARMNN_AUTO_TEST_CASE_WITH_THF(PadFloat322d, PadFloat322dTest)
// ARMNN_AUTO_TEST_CASE_WITH_THF(PadFloat322dCustomPadding, PadFloat322dCustomPaddingTest)
// ARMNN_AUTO_TEST_CASE_WITH_THF(PadFloat323d, PadFloat323dTest)
// TODO: Driver need fix batch issue
// ARMNN_AUTO_TEST_CASE_WITH_THF(PadFloat324d, PadFloat324dTest)

// ARMNN_AUTO_TEST_CASE_WITH_THF(PadUint82d, PadUint82dTest)
// ARMNN_AUTO_TEST_CASE_WITH_THF(PadUint82dCustomPadding, PadUint82dCustomPaddingTest)
// ARMNN_AUTO_TEST_CASE_WITH_THF(PadUint83d, PadUint83dTest)
// ARMNN_AUTO_TEST_CASE_WITH_THF(PadUint84d, PadUint84dTest)

// ARMNN_AUTO_TEST_CASE_WITH_THF(Pad2dQSymm16, Pad2dTestCommon<DataType::QSymmS16>, 2.0f, 0, 0.0f)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Pad2dQSymm16CustomPadding, Pad2dTestCommon<DataType::QSymmS16>, 2.0f, 0,
// 1.0f)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Pad3dQSymm16, Pad3dTestCommon<DataType::QSymmS16>, 2.0f, 0)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Pad4dQSymm16, Pad4dTestCommon<DataType::QSymmS16>, 2.0f, 0)

// // Constant
ARMNN_AUTO_TEST_CASE_WITH_THF(Constant, ConstantTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(ConstantUint8, ConstantUint8CustomQuantizationScaleAndOffsetTest)
// ARMNN_AUTO_TEST_CASE_WITH_THF(ConstantInt16, ConstantInt16CustomQuantizationScaleAndOffsetTest)

// Concat
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat1d, Concat1dTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat1dUint8, Concat1dUint8Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(Concat2dDim0, Concat2dDim0Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat2dDim0Uint8, Concat2dDim0Uint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat2dDim1, Concat2dDim1Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat2dDim1Uint8, Concat2dDim1Uint8Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(Concat2dDim0DiffInputDims, Concat2dDim0DiffInputDimsTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat2dDim0DiffInputDimsUint8, Concat2dDim0DiffInputDimsUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat2dDim1DiffInputDims, Concat2dDim1DiffInputDimsTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat2dDim1DiffInputDimsUint8, Concat2dDim1DiffInputDimsUint8Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(Concat3dDim0, Concat3dDim0Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat3dDim0Uint8, Concat3dDim0Uint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat3dDim1, Concat3dDim1Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat3dDim1Uint8, Concat3dDim1Uint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat3dDim2, Concat3dDim2Test, true)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat3dDim2Uint8, Concat3dDim2Uint8Test, true)

ARMNN_AUTO_TEST_CASE_WITH_THF(Concat3dDim0DiffInputDims, Concat3dDim0DiffInputDimsTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat3dDim0DiffInputDimsUint8, Concat3dDim0DiffInputDimsUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat3dDim1DiffInputDims, Concat3dDim1DiffInputDimsTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat3dDim1DiffInputDimsUint8, Concat3dDim1DiffInputDimsUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat3dDim2DiffInputDims, Concat3dDim2DiffInputDimsTest, true)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat3dDim2DiffInputDimsUint8, Concat3dDim2DiffInputDimsUint8Test, true)

ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDim0, Concat4dDim0Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDim1, Concat4dDim1Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDim2, Concat4dDim2Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDim3, Concat4dDim3Test, true)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDim0Uint8, Concat4dDim0Uint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDim1Uint8, Concat4dDim1Uint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDim2Uint8, Concat4dDim2Uint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDim3Uint8, Concat4dDim3Uint8Test, true)

ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDiffShapeDim0, Concat4dDiffShapeDim0Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDiffShapeDim1, Concat4dDiffShapeDim1Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDiffShapeDim2, Concat4dDiffShapeDim2Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDiffShapeDim3, Concat4dDiffShapeDim3Test, true)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDiffShapeDim0Uint8, Concat4dDiffShapeDim0Uint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDiffShapeDim1Uint8, Concat4dDiffShapeDim1Uint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDiffShapeDim2Uint8, Concat4dDiffShapeDim2Uint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDiffShapeDim3Uint8, Concat4dDiffShapeDim3Uint8Test, true)

// // Floor
// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleFloor, SimpleFloorTest<DataType::Float32>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleFloorFloat16, SimpleFloorTest<DataType::Float16>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleFloorQuantisedSymm16, SimpleFloorTest<DataType::QSymmS16>)

// // Reshape
// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleReshapeFloat32, SimpleReshapeTest<DataType::Float32>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleReshapeQuantisedAsymm8, SimpleReshapeTest<DataType::QAsymmU8>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleReshapeQuantisedSymm16, SimpleReshapeTest<DataType::QSymmS16>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Reshape5d, Reshape5dTest<DataType::Float32>)

// Rsqrt
ARMNN_AUTO_TEST_CASE_WITH_THF(Rsqrt2d, Rsqrt2dTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(Rsqrt3d, Rsqrt3dTest<DataType::Float32>)
// TODO: inf/-inf not support by shader kernel yet
// ARMNN_AUTO_TEST_CASE_WITH_THF(RsqrtZero, RsqrtZeroTest<DataType::Float32>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(RsqrtNegative, RsqrtNegativeTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(Rsqrt2dFloat16, Rsqrt2dTest<DataType::Float16>)
ARMNN_AUTO_TEST_CASE_WITH_THF(Rsqrt3dFloat16, Rsqrt3dTest<DataType::Float16>)
ARMNN_AUTO_TEST_CASE_WITH_THF(Rsqrt2dQuantisedAsymm8, Rsqrt2dTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(Rsqrt3dQuantisedAsymm8, Rsqrt3dTest<DataType::QAsymmU8>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Rsqrt2dQuantisedSymm16, Rsqrt2dTest<DataType::QSymmS16>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Rsqrt3dQuantisedSymm16, Rsqrt3dTest<DataType::QSymmS16>)

// Permute
ARMNN_AUTO_TEST_CASE_WITH_THF(SimplePermuteFloat32, SimplePermuteTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(PermuteFloat32ValueSet1Test, PermuteValueSet1Test<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(PermuteFloat32ValueSet2Test, PermuteValueSet2Test<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(PermuteFloat32ValueSet3Test, PermuteValueSet3Test<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimplePermuteQASymm8, SimplePermuteTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(PermuteQASymm8ValueSet1Test, PermuteValueSet1Test<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(PermuteQASymm8ValueSet2Test, PermuteValueSet2Test<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(PermuteQASymm8ValueSet3Test, PermuteValueSet3Test<DataType::QAsymmU8>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SimplePermuteQSymm16, SimplePermuteTest<DataType::QSymmS16>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(PermuteQSymm16ValueSet1Test, PermuteValueSet1Test<DataType::QSymmS16>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(PermuteQSymm16ValueSet2Test, PermuteValueSet2Test<DataType::QSymmS16>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(PermuteQSymm16ValueSet3Test, PermuteValueSet3Test<DataType::QSymmS16>)

// // Lstm
// BOOST_AUTO_TEST_CASE(LstmUtilsZeroVector) {
//                      LstmUtilsZeroVectorTest(); }
// BOOST_AUTO_TEST_CASE(LstmUtilsMeanStddevNormalization) {
//                      LstmUtilsMeanStddevNormalizationNoneZeroInputTest();
//                      LstmUtilsMeanStddevNormalizationAllZeroInputTest();
//                      LstmUtilsMeanStddevNormalizationMixedZeroInputTest(); }
// BOOST_AUTO_TEST_CASE(LstmUtilsVectorBatchVectorCwiseProduct) {
//                      LstmUtilsVectorBatchVectorCwiseProductTest(); }
// BOOST_AUTO_TEST_CASE(LstmUtilsVectorBatchVectorAdd) {
//                      LstmUtilsVectorBatchVectorAddTest(); }

// ARMNN_AUTO_TEST_CASE_WITH_THF(LstmLayerFloat32WithCifgWithPeepholeNoProjection,
//                      LstmLayerFloat32WithCifgWithPeepholeNoProjectionTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(LstmLayerFloat32NoCifgNoPeepholeNoProjection,
                     LstmLayerFloat32NoCifgNoPeepholeNoProjectionTest)
// ARMNN_AUTO_TEST_CASE_WITH_THF(LstmLayerFloat32NoCifgWithPeepholeWithProjection,
//                      LstmLayerFloat32NoCifgWithPeepholeWithProjectionTest)

// ARMNN_AUTO_TEST_CASE_WITH_THF(LstmLayerFloat32NoCifgWithPeepholeWithProjectionWithLayerNorm,
//                      LstmLayerFloat32NoCifgWithPeepholeWithProjectionWithLayerNormTest)

// ARMNN_AUTO_TEST_CASE_WITH_THF(LstmLayerInt16NoCifgNoPeepholeNoProjection,
//                      LstmLayerInt16NoCifgNoPeepholeNoProjectionTest)
// ARMNN_AUTO_TEST_CASE_WITH_THF(LstmLayerInt16WithCifgWithPeepholeNoProjection,
//                      LstmLayerInt16WithCifgWithPeepholeNoProjectionTest)
// ARMNN_AUTO_TEST_CASE_WITH_THF(LstmLayerInt16NoCifgWithPeepholeWithProjection,
//                      LstmLayerInt16NoCifgWithPeepholeWithProjectionTest)
// ARMNN_AUTO_TEST_CASE_WITH_THF(LstmLayerInt16NoCifgNoPeepholeNoProjectionInt16Constant,
//                      LstmLayerInt16NoCifgNoPeepholeNoProjectionInt16ConstantTest)

// Convert from Float16 to Float32
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvertFp16ToFp32, SimpleConvertFp16ToFp32Test)
// Convert from Float32 to Float16
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvertFp32ToFp16, SimpleConvertFp32ToFp16Test)

// Mean
ARMNN_AUTO_TEST_CASE_WITH_THF(MeanSimpleFloat32, MeanSimpleTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(MeanSimpleAxisFloat32, MeanSimpleAxisTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(MeanKeepDimsFloat32, MeanKeepDimsTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(MeanMultipleDimsFloat32, MeanMultipleDimsTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(MeanVts1Float32, MeanVts1Test<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(MeanVts2Float32, MeanVts2Test<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(MeanVts3Float32, MeanVts3Test<DataType::Float32>)

ARMNN_AUTO_TEST_CASE_WITH_THF(MeanSimpleQuantisedAsymm8, MeanSimpleTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(MeanSimpleAxisQuantisedAsymm8, MeanSimpleAxisTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(MeanKeepDimsQuantisedAsymm8, MeanKeepDimsTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(MeanMultipleDimsQuantisedAsymm8, MeanMultipleDimsTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(MeanVts1QuantisedAsymm8, MeanVts1Test<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(MeanVts2QuantisedAsymm8, MeanVts2Test<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(MeanVts3QuantisedAsymm8, MeanVts3Test<DataType::QAsymmU8>)

// ARMNN_AUTO_TEST_CASE_WITH_THF(MeanSimpleQuantisedSymm16, MeanSimpleTest<DataType::QSymmS16>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(MeanSimpleAxisQuantisedSymm16, MeanSimpleAxisTest<DataType::QSymmS16>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(MeanKeepDimsQuantisedSymm16, MeanKeepDimsTest<DataType::QSymmS16>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(MeanMultipleDimsQuantisedSymm16, MeanMultipleDimsTest<DataType::QSymmS16>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(MeanVts1QuantisedSymm16, MeanVts1Test<DataType::QSymmS16>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(MeanVts2QuantisedSymm16, MeanVts2Test<DataType::QSymmS16>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(MeanVts3QuantisedSymm16, MeanVts3Test<DataType::QSymmS16>)

// ARMNN_AUTO_TEST_CASE_WITH_THF(AdditionAfterMaxPool, AdditionAfterMaxPoolTest)

// ArgMinMax
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMaxFloat32, ArgMaxSimpleTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMinFloat32, ArgMinSimpleTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMinChannelFloat32, ArgMinChannelTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMaxChannelFloat32, ArgMaxChannelTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMaxHeightFloat32, ArgMaxHeightTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMinWidthFloat32, ArgMinWidthTest<DataType::Float32>)

// ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMaxSigned32, ArgMaxSimpleTest<DataType::Signed32>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMinSigned32, ArgMinSimpleTest<DataType::Signed32>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMinChannelSigned32, ArgMinChannelTest<DataType::Signed32>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMaxChannelSigned32, ArgMaxChannelTest<DataType::Signed32>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMaxHeightSigned32, ArgMaxHeightTest<DataType::Signed32>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMinWidthSigned32, ArgMinWidthTest<DataType::Signed32>)

ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMaxSimpleQuantisedAsymm8, ArgMaxSimpleTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMinSimpleQuantisedAsymm8, ArgMinSimpleTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMinChannelQuantisedAsymm8, ArgMinChannelTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMaxChannelQuantisedAsymm8, ArgMaxChannelTest<DataType::QAsymmU8>)

// ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMaxSimpleQuantisedSymm16, ArgMaxSimpleTest<DataType::QSymmS16>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMinSimpleQuantisedSymm16, ArgMinSimpleTest<DataType::QSymmS16>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMinChannelQuantisedSymm16, ArgMinChannelTest<DataType::QSymmS16>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMaxChannelQuantisedSymm16, ArgMaxChannelTest<DataType::QSymmS16>)

// Space To Batch Nd
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdSimpleFloat32, SpaceToBatchNdSimpleFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiChannelsFloat32, SpaceToBatchNdMultiChannelsFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiBlockFloat32, SpaceToBatchNdMultiBlockFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdPaddingFloat32, SpaceToBatchNdPaddingFloat32Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdSimpleFloat16, SpaceToBatchNdSimpleFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiChannelsFloat16, SpaceToBatchNdMultiChannelsFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiBlockFloat16, SpaceToBatchNdMultiBlockFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdPaddingFloat16, SpaceToBatchNdPaddingFloat16Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdSimpleUint8, SpaceToBatchNdSimpleUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiChannelsUint8, SpaceToBatchNdMultiChannelsUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiBlockUint8, SpaceToBatchNdMultiBlockUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdPaddingUint8, SpaceToBatchNdPaddingUint8Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdSimpleNhwcFloat32, SpaceToBatchNdSimpleNhwcFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiChannelsNhwcFloat32,
SpaceToBatchNdMultiChannelsNhwcFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiBlockNhwcFloat32,
SpaceToBatchNdMultiBlockNhwcFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdPaddingNhwcFloat32, SpaceToBatchNdPaddingNhwcFloat32Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdSimpleNhwcFloat16, SpaceToBatchNdSimpleNhwcFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiChannelsNhwcFloat16,
SpaceToBatchNdMultiChannelsNhwcFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiBlockNhwcFloat16,
SpaceToBatchNdMultiBlockNhwcFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdPaddingNhwcFloat16, SpaceToBatchNdPaddingNhwcFloat16Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdSimpleNhwcUint8, SpaceToBatchNdSimpleNhwcUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiChannelsNhwcUint8,
SpaceToBatchNdMultiChannelsNhwcUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiBlockNhwcUint8, SpaceToBatchNdMultiBlockNhwcUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdPaddingNhwcUint8, SpaceToBatchNdPaddingNhwcUint8Test)

// ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdSimpleUint16, SpaceToBatchNdSimpleUint16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiChannelsUint16, SpaceToBatchNdMultiChannelsUint16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiBlockUint16, SpaceToBatchNdMultiBlockUint16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdPaddingUint16, SpaceToBatchNdPaddingUint16Test)

// ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdSimpleNhwcUint16, SpaceToBatchNdSimpleNhwcUint16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiChannelsNhwcUint16,
// SpaceToBatchNdMultiChannelsNhwcUint16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiBlockNhwcUint16, SpaceToBatchNdMultiBlockNhwcUint16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdPaddingNhwcUint16, SpaceToBatchNdPaddingNhwcUint16Test)

// BatchToSpace
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNhwcFloat32_1, BatchToSpaceNdNhwcTest1<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNhwcFloat32_2, BatchToSpaceNdNhwcTest2<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNhwcFloat32_3, BatchToSpaceNdNhwcTest3<DataType::Float32>)
// TODO: Support output BatchSize > 1
// ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNhwcFloat32_4, BatchToSpaceNdNhwcTest4<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNhwcFloat32_5, BatchToSpaceNdNhwcTest5<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNhwcFloat32_6, BatchToSpaceNdNhwcTest6<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNhwcFloat32_7, BatchToSpaceNdNhwcTest7<DataType::Float32>)

ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNhwcFloat16_1, BatchToSpaceNdNhwcTest1<DataType::Float16>)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNhwcFloat16_2, BatchToSpaceNdNhwcTest2<DataType::Float16>)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNhwcFloat16_3, BatchToSpaceNdNhwcTest3<DataType::Float16>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNhwcFloat16_4, BatchToSpaceNdNhwcTest4<DataType::Float16>)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNhwcFloat16_5, BatchToSpaceNdNhwcTest5<DataType::Float16>)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNhwcFloat16_6, BatchToSpaceNdNhwcTest6<DataType::Float16>)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNhwcFloat16_7, BatchToSpaceNdNhwcTest7<DataType::Float16>)

ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNhwcUint1,  BatchToSpaceNdNhwcTest1<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNhwcUint2,  BatchToSpaceNdNhwcTest2<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNhwcUint3,  BatchToSpaceNdNhwcTest3<DataType::QAsymmU8>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNhwcUint4,  BatchToSpaceNdNhwcTest4<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNhwcUint5,  BatchToSpaceNdNhwcTest5<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNhwcUint6,  BatchToSpaceNdNhwcTest6<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNhwcUint7,  BatchToSpaceNdNhwcTest7<DataType::QAsymmU8>)

// ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNhwcQsymm16_1,  BatchToSpaceNdNhwcTest1<DataType::QSymmS16>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNhwcQsymm16_2,  BatchToSpaceNdNhwcTest2<DataType::QSymmS16>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNhwcQsymm16_3,  BatchToSpaceNdNhwcTest3<DataType::QSymmS16>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNhwcQsymm16_4,  BatchToSpaceNdNhwcTest4<DataType::QSymmS16>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNhwcQsymm16_5,  BatchToSpaceNdNhwcTest5<DataType::QSymmS16>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNhwcQsymm16_6,  BatchToSpaceNdNhwcTest6<DataType::QSymmS16>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNhwcQsymm16_7,  BatchToSpaceNdNhwcTest7<DataType::QSymmS16>)

ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNchwFloat16_1, BatchToSpaceNdNchwTest1<DataType::Float16>)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNchwFloat16_2, BatchToSpaceNdNchwTest2<DataType::Float16>)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNchwFloat16_3, BatchToSpaceNdNchwTest3<DataType::Float16>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNchwFloat16_4, BatchToSpaceNdNchwTest4<DataType::Float16>)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNchwFloat16_5, BatchToSpaceNdNchwTest5<DataType::Float16>)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNchwFloat16_6, BatchToSpaceNdNchwTest6<DataType::Float16>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNchwFloat16_7, BatchToSpaceNdNchwTest7<DataType::Float16>)

ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNchwUint1,  BatchToSpaceNdNchwTest1<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNchwUint2,  BatchToSpaceNdNchwTest2<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNchwUint3,  BatchToSpaceNdNchwTest3<DataType::QAsymmU8>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNchwUint4,  BatchToSpaceNdNchwTest4<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNchwUint5,  BatchToSpaceNdNchwTest5<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNchwUint6,  BatchToSpaceNdNchwTest6<DataType::QAsymmU8>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNchwUint7,  BatchToSpaceNdNchwTest7<DataType::QAsymmU8>)

// ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNchwQsymm16_1,  BatchToSpaceNdNchwTest1<DataType::QSymmS16>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNchwQsymm16_2,  BatchToSpaceNdNchwTest2<DataType::QSymmS16>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNchwQsymm16_3,  BatchToSpaceNdNchwTest3<DataType::QSymmS16>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNchwQsymm16_4,  BatchToSpaceNdNchwTest4<DataType::QSymmS16>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNchwQsymm16_5,  BatchToSpaceNdNchwTest5<DataType::QSymmS16>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNchwQsymm16_6,  BatchToSpaceNdNchwTest6<DataType::QSymmS16>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNchwQsymm16_7,  BatchToSpaceNdNchwTest7<DataType::QSymmS16>)

// DepthToSpace
ARMNN_AUTO_TEST_CASE(DepthToSpaceNchwFloat32_1, DepthToSpaceTest1<DataType::Float32>,
DataLayout::NCHW);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNchwFloat32_2, DepthToSpaceTest2<DataType::Float32>,
DataLayout::NCHW);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNchwFloat32_3, DepthToSpaceTest3<DataType::Float32>,
DataLayout::NCHW);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNchwFloat32_4, DepthToSpaceTest4<DataType::Float32>,
DataLayout::NCHW);

ARMNN_AUTO_TEST_CASE(DepthToSpaceNchwFloat16_1, DepthToSpaceTest1<DataType::Float16>,
DataLayout::NCHW);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNchwFloat16_2, DepthToSpaceTest2<DataType::Float16>,
DataLayout::NCHW);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNchwFloat16_3, DepthToSpaceTest3<DataType::Float16>,
DataLayout::NCHW);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNchwFloat16_4, DepthToSpaceTest4<DataType::Float16>,
DataLayout::NCHW);

ARMNN_AUTO_TEST_CASE(DepthToSpaceNchwUint8_1, DepthToSpaceTest1<DataType::QAsymmU8>,
DataLayout::NCHW);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNchwUint8_2, DepthToSpaceTest2<DataType::QAsymmU8>,
DataLayout::NCHW);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNchwUint8_3, DepthToSpaceTest3<DataType::QAsymmU8>,
DataLayout::NCHW);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNchwUint8_4, DepthToSpaceTest4<DataType::QAsymmU8>,
DataLayout::NCHW);

// ARMNN_AUTO_TEST_CASE(DepthToSpaceNchwInt16_1, DepthToSpaceTest1<DataType::QSymmS16>,
// DataLayout::NCHW);
// ARMNN_AUTO_TEST_CASE(DepthToSpaceNchwInt16_2, DepthToSpaceTest2<DataType::QSymmS16>,
// DataLayout::NCHW);
// ARMNN_AUTO_TEST_CASE(DepthToSpaceNchwInt16_3, DepthToSpaceTest3<DataType::QSymmS16>,
// DataLayout::NCHW);
// ARMNN_AUTO_TEST_CASE(DepthToSpaceNchwInt16_4, DepthToSpaceTest4<DataType::QSymmS16>,
// DataLayout::NCHW);

ARMNN_AUTO_TEST_CASE(DepthToSpaceNhwcFloat32_1, DepthToSpaceTest1<DataType::Float32>,
DataLayout::NHWC);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNhwcFloat32_2, DepthToSpaceTest2<DataType::Float32>,
DataLayout::NHWC);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNhwcFloat32_3, DepthToSpaceTest3<DataType::Float32>,
DataLayout::NHWC);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNhwcFloat32_4, DepthToSpaceTest4<DataType::Float32>,
DataLayout::NHWC);

ARMNN_AUTO_TEST_CASE(DepthToSpaceNhwcFloat16_1, DepthToSpaceTest1<DataType::Float16>,
DataLayout::NHWC);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNhwcFloat16_2, DepthToSpaceTest2<DataType::Float16>,
DataLayout::NHWC);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNhwcFloat16_3, DepthToSpaceTest3<DataType::Float16>,
DataLayout::NHWC);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNhwcFloat16_4, DepthToSpaceTest4<DataType::Float16>,
DataLayout::NHWC);

ARMNN_AUTO_TEST_CASE(DepthToSpaceNhwcUint8_1, DepthToSpaceTest1<DataType::QAsymmU8>,
DataLayout::NHWC);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNhwcUint8_2, DepthToSpaceTest2<DataType::QAsymmU8>,
DataLayout::NHWC);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNhwcUint8_3, DepthToSpaceTest3<DataType::QAsymmU8>,
DataLayout::NHWC);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNhwcUint8_4, DepthToSpaceTest4<DataType::QAsymmU8>,
DataLayout::NHWC);

// ARMNN_AUTO_TEST_CASE(DepthToSpaceNhwcInt16_1, DepthToSpaceTest1<DataType::QSymmS16>,
// DataLayout::NHWC);
// ARMNN_AUTO_TEST_CASE(DepthToSpaceNhwcInt16_2, DepthToSpaceTest2<DataType::QSymmS16>,
// DataLayout::NHWC);
// ARMNN_AUTO_TEST_CASE(DepthToSpaceNhwcInt16_3, DepthToSpaceTest3<DataType::QSymmS16>,
// DataLayout::NHWC);
// ARMNN_AUTO_TEST_CASE(DepthToSpaceNhwcInt16_4, DepthToSpaceTest4<DataType::QSymmS16>,
// DataLayout::NHWC);

// // SpaceToDepth
// ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToDepthNchwAsymmQ8, SpaceToDepthNchwAsymmQ8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToDepthNhwcAsymmQ8, SpaceToDepthNhwcAsymmQ8Test)

// ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToDepthNhwc1Float32, SpaceToDepthNhwcFloat32Test1)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToDepthNchw1Float32, SpaceToDepthNchwFloat32Test1)

// ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToDepthNhwc2Float32, SpaceToDepthNhwcFloat32Test2)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToDepthNchw2Float32, SpaceToDepthNchwFloat32Test2)

// ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToDepthNhwcQSymm16, SpaceToDepthNhwcQSymm16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToDepthNchwQSymm16, SpaceToDepthNchwQSymm16Test)

// Strided Slice
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice4dFloat32, StridedSlice4dFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice4dReverseFloat32, StridedSlice4dReverseFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceSimpleStrideFloat32, StridedSliceSimpleStrideFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceSimpleRangeMaskFloat32, StridedSliceSimpleRangeMaskFloat32Test)
// TODO: support ShrinkAxisMask
// ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceShrinkAxisMaskFloat32, StridedSliceShrinkAxisMaskFloat32Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceShrinkAxisMaskCTSFloat32,
// StridedSliceShrinkAxisMaskCTSFloat32Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceShrinkAxisMaskBitPosition0Dim3Float32,
//                      StridedSliceShrinkAxisMaskBitPosition0Dim3Float32Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceShrinkAxisMaskBitPosition0Float32,
// StridedSliceShrinkAxisMaskBitPosition0Float32Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceShrinkAxisMaskBitPosition1Float32,
// StridedSliceShrinkAxisMaskBitPosition1Float32Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceShrinkAxisMaskBitPosition2Float32,
// StridedSliceShrinkAxisMaskBitPosition2Float32Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceShrinkAxisMaskBitPosition3Float32,
// StridedSliceShrinkAxisMaskBitPosition3Float32Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceShrinkAxisMaskBitPosition0And1Float32,
//                      StridedSliceShrinkAxisMaskBitPosition0And1Float32Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceShrinkAxisMaskBitPosition0And2Float32,
//                      StridedSliceShrinkAxisMaskBitPosition0And2Float32Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceShrinkAxisMaskBitPosition0And3Float32,
//                      StridedSliceShrinkAxisMaskBitPosition0And3Float32Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceShrinkAxisMaskBitPosition0And1And3Float32,
//                      StridedSliceShrinkAxisMaskBitPosition0And1And3Float32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice3dFloat32, StridedSlice3dFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice3dReverseFloat32, StridedSlice3dReverseFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice2dFloat32, StridedSlice2dFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice2dReverseFloat32, StridedSlice2dReverseFloat32Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice4dUint8, StridedSlice4dUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice4dReverseUint8, StridedSlice4dReverseUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceSimpleStrideUint8, StridedSliceSimpleStrideUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceSimpleRangeMaskUint8, StridedSliceSimpleRangeMaskUint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceShrinkAxisMaskUint8, StridedSliceShrinkAxisMaskUint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceShrinkAxisMaskBitPosition0Dim3Uint8,
//                      StridedSliceShrinkAxisMaskBitPosition0Dim3Uint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceShrinkAxisMaskBitPosition0Uint8,
// StridedSliceShrinkAxisMaskBitPosition0Uint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceShrinkAxisMaskBitPosition1Uint8,
// StridedSliceShrinkAxisMaskBitPosition1Uint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceShrinkAxisMaskBitPosition2Uint8,
// StridedSliceShrinkAxisMaskBitPosition2Uint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceShrinkAxisMaskBitPosition3Uint8,
// StridedSliceShrinkAxisMaskBitPosition3Uint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceShrinkAxisMaskBitPosition0And1Uint8,
//                      StridedSliceShrinkAxisMaskBitPosition0And1Uint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceShrinkAxisMaskBitPosition0And2Uint8,
//                      StridedSliceShrinkAxisMaskBitPosition0And2Uint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceShrinkAxisMaskBitPosition0And3Uint8,
//                      StridedSliceShrinkAxisMaskBitPosition0And3Uint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceShrinkAxisMaskBitPosition0And1And3Uint8,
//                      StridedSliceShrinkAxisMaskBitPosition0And1And3Uint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice3dUint8, StridedSlice3dUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice3dReverseUint8, StridedSlice3dReverseUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice2dUint8, StridedSlice2dUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice2dReverseUint8, StridedSlice2dReverseUint8Test)

// ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice4dInt16, StridedSlice4dInt16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice4dReverseInt16, StridedSlice4dReverseInt16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceSimpleStrideInt16, StridedSliceSimpleStrideInt16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceSimpleRangeMaskInt16, StridedSliceSimpleRangeMaskInt16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceShrinkAxisMaskInt16, StridedSliceShrinkAxisMaskInt16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice3dInt16, StridedSlice3dInt16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice3dReverseInt16, StridedSlice3dReverseInt16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice2dInt16, StridedSlice2dInt16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice2dReverseInt16, StridedSlice2dReverseInt16Test)

// // Debug
// ARMNN_AUTO_TEST_CASE_WITH_THF(Debug4dFloat32, Debug4dFloat32Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Debug3dFloat32, Debug3dFloat32Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Debug2dFloat32, Debug2dFloat32Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Debug1dFloat32, Debug1dFloat32Test)

// ARMNN_AUTO_TEST_CASE_WITH_THF(Debug4dUint8, Debug4dUint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Debug3dUint8, Debug3dUint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Debug2dUint8, Debug2dUint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Debug1dUint8, Debug1dUint8Test)

// ARMNN_AUTO_TEST_CASE_WITH_THF(Debug4dQSymm16, Debug4dInt16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Debug3dQSymm16, Debug3dInt16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Debug2dQSymm16, Debug2dInt16Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Debug1dQSymm16, Debug1dInt16Test)

// // Gather
ARMNN_AUTO_TEST_CASE_WITH_THF(Gather1dParamsFloat32, Gather1dParamsFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Gather1dParamsFloat16, Gather1dParamsFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Gather1dParamsUint8, Gather1dParamsUint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Gather1dParamsInt16, Gather1dParamsInt16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(GatherMultiDimParamsFloat32, GatherMultiDimParamsFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(GatherMultiDimParamsFloat16, GatherMultiDimParamsFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(GatherMultiDimParamsUint8, GatherMultiDimParamsUint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(GatherMultiDimParamsInt16, GatherMultiDimParamsInt16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(GatherMultiDimParamsMultiDimIndicesFloat32,
GatherMultiDimParamsMultiDimIndicesFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(GatherMultiDimParamsMultiDimIndicesFloat16,
GatherMultiDimParamsMultiDimIndicesFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(GatherMultiDimParamsMultiDimIndicesUint8,
GatherMultiDimParamsMultiDimIndicesUint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(GatherMultiDimParamsMultiDimIndicesInt16,
// GatherMultiDimParamsMultiDimIndicesInt16Test)

// Abs
ARMNN_AUTO_TEST_CASE_WITH_THF(Abs2d, Abs2dTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(Abs3d, Abs3dTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(AbsZero, AbsZeroTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(Abs2dFloat16, Abs2dTest<DataType::Float16>)
ARMNN_AUTO_TEST_CASE_WITH_THF(Abs3dFloat16, Abs3dTest<DataType::Float16>)
ARMNN_AUTO_TEST_CASE_WITH_THF(Abs2dQuantisedAsymm8, Abs2dTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(Abs3dQuantisedAsymm8, Abs3dTest<DataType::QAsymmU8>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Abs2dQuantisedSymm16, Abs2dTest<DataType::QSymmS16>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(Abs3dQuantisedSymm16, Abs3dTest<DataType::QSymmS16>)

// Detection PostProcess
//BOOST_AUTO_TEST_CASE(DetectionPostProcessRegularNmsFloat)
//{
//    DetectionPostProcessRegularNmsFloatTest<NpuWorkloadFactory>();
//}
// BOOST_AUTO_TEST_CASE(DetectionPostProcessFastNmsFloat)
// {
//     DetectionPostProcessFastNmsFloatTest<RefWorkloadFactory>();
// }
// BOOST_AUTO_TEST_CASE(DetectionPostProcessRegularNmsUint8)
// {
//     DetectionPostProcessRegularNmsQuantizedTest<
//         RefWorkloadFactory, DataType::QAsymmU8>();
// }
// BOOST_AUTO_TEST_CASE(DetectionPostProcessFastNmsUint8)
// {
//     DetectionPostProcessRegularNmsQuantizedTest<
//         RefWorkloadFactory, DataType::QAsymmU8>();
// }
// BOOST_AUTO_TEST_CASE(DetectionPostProcessRegularNmsInt16)
// {
//     DetectionPostProcessRegularNmsQuantizedTest<
//         RefWorkloadFactory, DataType::QSymmS16>();
// }
// BOOST_AUTO_TEST_CASE(DetectionPostProcessFastNmsInt16)
// {
//     DetectionPostProcessFastNmsQuantizedTest<
//         RefWorkloadFactory, DataType::QSymmS16>();
// }

// // Dequantize
ARMNN_AUTO_TEST_CASE(DequantizeSimpleUint8, DequantizeSimpleUint8Test)
ARMNN_AUTO_TEST_CASE(DequantizeOffsetUint8, DequantizeOffsetUint8Test)
// ARMNN_AUTO_TEST_CASE(DequantizeSimpleAsymmInt8, DequantizeSimpleAsymmInt8Test)
// ARMNN_AUTO_TEST_CASE(DequantizeOffsetAsymmInt8, DequantizeOffsetAsymmInt8Test)
// ARMNN_AUTO_TEST_CASE(DequantizeSimpleInt8, DequantizeSimpleInt8Test)
// ARMNN_AUTO_TEST_CASE(DequantizeSimpleInt16, DequantizeSimpleInt16Test)
// ARMNN_AUTO_TEST_CASE(DequantizeSimpleUint8ToFp16, DequantizeSimpleUint8ToFp16Test)
// ARMNN_AUTO_TEST_CASE(DequantizeSimpleInt8ToFp16, DequantizeSimpleInt8ToFp16Test)
// ARMNN_AUTO_TEST_CASE(DequantizeSimpleInt16ToFp16, DequantizeSimpleInt16ToFp16Test)

// // Quantize
ARMNN_AUTO_TEST_CASE_WITH_THF(QuantizeSimpleUint8, QuantizeSimpleUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(QuantizeClampUint8, QuantizeClampUint8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(QuantizeClampAsymmInt8, QuantizeClampAsymmInt8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(QuantizeClampInt8, QuantizeClampInt8Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(QuantizeClampInt16, QuantizeClampInt16Test)

// // PReLU
// ARMNN_AUTO_TEST_CASE_WITH_THF(PreluFloat32, PreluTest<DataType::Float32>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(PreluFloat16, PreluTest<DataType::Float16>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(PreluUint8,   PreluTest<DataType::QAsymmU8>)
// ARMNN_AUTO_TEST_CASE_WITH_THF(PreluInt16,   PreluTest<DataType::QSymmS16>)

// Slice
ARMNN_AUTO_TEST_CASE(Slice4dFloat32, Slice4dFloat32Test)
ARMNN_AUTO_TEST_CASE(Slice3dFloat32, Slice3dFloat32Test)
ARMNN_AUTO_TEST_CASE(Slice2dFloat32, Slice2dFloat32Test)
ARMNN_AUTO_TEST_CASE(Slice1dFloat32, Slice1dFloat32Test)

ARMNN_AUTO_TEST_CASE(Slice4dUint8, Slice4dUint8Test)
ARMNN_AUTO_TEST_CASE(Slice3dUint8, Slice3dUint8Test)
ARMNN_AUTO_TEST_CASE(Slice2dUint8, Slice2dUint8Test)
ARMNN_AUTO_TEST_CASE(Slice1dUint8, Slice1dUint8Test)

// ARMNN_AUTO_TEST_CASE(Slice4dInt16, Slice4dInt16Test)
// ARMNN_AUTO_TEST_CASE(Slice3dInt16, Slice3dInt16Test)
// ARMNN_AUTO_TEST_CASE(Slice2dInt16, Slice2dInt16Test)
// ARMNN_AUTO_TEST_CASE(Slice1dInt16, Slice1dInt16Test)

// TransposeConvolution2d
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleTransposeConvolution2dFloatNchw,
                     SimpleTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                     true,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleTransposeConvolution2dFloatNhwc,
                     SimpleTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                     true,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleTransposeConvolution2dUint8Nchw,
                     SimpleTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                     true,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleTransposeConvolution2dUint8Nhwc,
                     SimpleTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                     true,
                     DataLayout::NHWC)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleTransposeConvolution2dInt16Nchw,
//                      SimpleTransposeConvolution2dTest<DataType::QSymmS16, DataType::Signed32>,
//                      true,
//                      DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleTransposeConvolution2dInt16Nhwc,
//                      SimpleTransposeConvolution2dTest<DataType::QSymmS16, DataType::Signed32>,
//                      true,
//                      DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedSimpleTransposeConvolution2dFloatNchw,
                     SimpleTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedSimpleTransposeConvolution2dFloatNhwc,
                     SimpleTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                     true,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedSimpleTransposeConvolution2dUint8Nchw,
                     SimpleTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                     true,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedSimpleTransposeConvolution2dUint8Nhwc,
                     SimpleTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                     true,
                     DataLayout::NHWC)
// ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedSimpleTransposeConvolution2dInt16Nchw,
//                      SimpleTransposeConvolution2dTest<DataType::QSymmS16, DataType::Signed32>,
//                      true,
//                      DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedSimpleTransposeConvolution2dInt16Nhwc,
//                      SimpleTransposeConvolution2dTest<DataType::QSymmS16, DataType::Signed32>,
//                      true,
//                      DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE_WITH_THF(PaddedTransposeConvolution2dFloatNchw,
                     PaddedTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                     true,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(PaddedTransposeConvolution2dFloatNhwc,
                     PaddedTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                     true,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(PaddedTransposeConvolution2dUint8Nchw,
                     PaddedTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                     true,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(PaddedTransposeConvolution2dUint8Nhwc,
                     PaddedTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                     true,
                     DataLayout::NHWC)
// ARMNN_AUTO_TEST_CASE_WITH_THF(PaddedTransposeConvolution2dInt16Nchw,
//                      PaddedTransposeConvolution2dTest<DataType::QSymmS16, DataType::Signed32>,
//                      true,
//                      DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(PaddedTransposeConvolution2dInt16Nhwc,
//                      PaddedTransposeConvolution2dTest<DataType::QSymmS16, DataType::Signed32>,
//                      true,
//                      DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedPaddedTransposeConvolution2dFloatNchw,
                     PaddedTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedPaddedTransposeConvolution2dFloatNhwc,
                     PaddedTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                     true,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedPaddedTransposeConvolution2dUint8Nchw,
                     PaddedTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                     true,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedPaddedTransposeConvolution2dUint8Nhwc,
                     PaddedTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                     true,
                     DataLayout::NHWC)
// ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedPaddedTransposeConvolution2dInt16Nchw,
//                      PaddedTransposeConvolution2dTest<DataType::QSymmS16, DataType::Signed32>,
//                      true,
//                      DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedPaddedTransposeConvolution2dInt16Nhwc,
//                      PaddedTransposeConvolution2dTest<DataType::QSymmS16, DataType::Signed32>,
//                      true,
//                      DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE_WITH_THF(StridedTransposeConvolution2dFloatNchw,
                     StridedTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                     true,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedTransposeConvolution2dFloatNhwc,
                     StridedTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                     true,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedTransposeConvolution2dUint8Nchw,
                     StridedTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                     true,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedTransposeConvolution2dUint8Nhwc,
                     StridedTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                     true,
                     DataLayout::NHWC)
// ARMNN_AUTO_TEST_CASE_WITH_THF(StridedTransposeConvolution2dInt16Nchw,
//                      StridedTransposeConvolution2dTest<DataType::QSymmS16, DataType::Signed32>,
//                      true,
//                      DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(StridedTransposeConvolution2dInt16Nhwc,
//                      StridedTransposeConvolution2dTest<DataType::QSymmS16, DataType::Signed32>,
//                      true,
//                      DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedStridedTransposeConvolution2dFloatNchw,
                     StridedTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedStridedTransposeConvolution2dFloatNhwc,
                     StridedTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                     true,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedStridedTransposeConvolution2dUint8Nchw,
                     StridedTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                     true,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedStridedTransposeConvolution2dUint8Nhwc,
                     StridedTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                     true,
                     DataLayout::NHWC)
// ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedStridedTransposeConvolution2dInt16Nchw,
//                      StridedTransposeConvolution2dTest<DataType::QSymmS16, DataType::Signed32>,
//                      true,
//                      DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedStridedTransposeConvolution2dInt16Nhwc,
//                      StridedTransposeConvolution2dTest<DataType::QSymmS16, DataType::Signed32>,
//                      true,
//                      DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE_WITH_THF(MultiChannelTransposeConvolution2dFloatNchw,
                     MultiChannelTransposeConvolution2dTest<DataType::Float32,
                     DataType::Float32>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(MultiChannelTransposeConvolution2dFloatNhwc,
                     MultiChannelTransposeConvolution2dTest<DataType::Float32,
                     DataType::Float32>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(MultiChannelTransposeConvolution2dUint8Nchw,
                     MultiChannelTransposeConvolution2dTest<DataType::QAsymmU8,
                     DataType::Signed32>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(MultiChannelTransposeConvolution2dUint8Nhwc,
                     MultiChannelTransposeConvolution2dTest<DataType::QAsymmU8,
                     DataType::Signed32>,
                     DataLayout::NHWC)
// ARMNN_AUTO_TEST_CASE_WITH_THF(MultiChannelTransposeConvolution2dInt16Nchw,
//                      MultiChannelTransposeConvolution2dTest<DataType::QSymmS16,
//                      DataType::Signed32>,
//                      DataLayout::NCHW)
// ARMNN_AUTO_TEST_CASE_WITH_THF(MultiChannelTransposeConvolution2dInt16Nhwc,
//                      MultiChannelTransposeConvolution2dTest<DataType::QSymmS16,
//                      DataType::Signed32>,
//                      DataLayout::NCHW)

// ARMNN_AUTO_TEST_CASE_WITH_THF(TransposeConvolution2dPerAxisQuantTestNchw,
//                      TransposeConvolution2dPerAxisQuantTest,
//                      DataLayout::NCHW);
// ARMNN_AUTO_TEST_CASE_WITH_THF(TransposeConvolution2dPerAxisQuantTestNhwc,
//                      TransposeConvolution2dPerAxisQuantTest,
//                      DataLayout::NHWC);

// // Stack
// ARMNN_AUTO_TEST_CASE_WITH_THF(Stack0Axis,           StackAxis0Float32Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(StackOutput4DAxis1,   StackOutput4DAxis1Float32Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(StackOutput4DAxis2,   StackOutput4DAxis2Float32Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(StackOutput4DAxis3,   StackOutput4DAxis3Float32Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(StackOutput3DInputs3, StackOutput3DInputs3Float32Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(StackOutput5D,        StackOutput5DFloat32Test)
// ARMNN_AUTO_TEST_CASE_WITH_THF(StackFloat16,         StackFloat16Test)

}