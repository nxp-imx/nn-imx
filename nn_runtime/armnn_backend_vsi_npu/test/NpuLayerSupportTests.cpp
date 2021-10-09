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

#include <layers/ConvertFp16ToFp32Layer.hpp>
#include <layers/ConvertFp32ToFp16Layer.hpp>
#include <test/TensorHelpers.hpp>

#include <backendsCommon/TensorHandle.hpp>
#include <backendsCommon/test/IsLayerSupportedTestImpl.hpp>
#include <backendsCommon/test/LayerTests.hpp>
#include "NpuLayerSupport.hpp"
#include "NpuWorkloadFactory.hpp"

#include <doctest/doctest.h>

#include <string>

namespace {

bool LayerTypeMatchesTest() {
    return LayerTypeMatchesTestImpl<armnn::LayerType::FirstLayer>(
        Tag<armnn::LayerType::FirstLayer>());
};

}  // anonymous namespace

TEST_SUITE("NpuLayerSupported"){

TEST_CASE("IsLayerSupportedLayerTypeMatches") {
    LayerTypeMatchesTest();
}
TEST_CASE("IsLayerSupportedNpuAddition") {
    armnn::TensorShape shape0 = {1, 1, 3, 4};
    armnn::TensorShape shape1 = {4};
    armnn::TensorShape outShape = {1, 1, 3, 4};
    armnn::TensorInfo in0(shape0, armnn::DataType::Float32);
    armnn::TensorInfo in1(shape1, armnn::DataType::Float32);
    armnn::TensorInfo out(outShape, armnn::DataType::Float32);

    armnn::NpuLayerSupport supportChecker;
    std::string reasonNotSupported;
    CHECK(supportChecker.IsAdditionSupported(in0, in1, out, reasonNotSupported));
}

// BOOST_AUTO_TEST_CASE(IsLayerSupportedFloat16Reference) {
//     armnn::NpuWorkloadFactory factory;
//     IsLayerSupportedTests<armnn::NpuWorkloadFactory, armnn::DataType::Float16>(&factory);
// }

// BOOST_AUTO_TEST_CASE(IsLayerSupportedFloat32Npu) {
//     armnn::NpuWorkloadFactory factory;
//     IsLayerSupportedTests<armnn::NpuWorkloadFactory, armnn::DataType::Float32>(&factory);
// }

// BOOST_AUTO_TEST_CASE(IsLayerSupportedUint8Npu) {
//     armnn::NpuWorkloadFactory factory;
//     IsLayerSupportedTests<armnn::NpuWorkloadFactory, armnn::DataType::QAsymmU8>(&factory);
// }

TEST_CASE("IsConvertFp16ToFp32SupportedNpu") {
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::NpuWorkloadFactory,
                                               armnn::ConvertFp16ToFp32Layer,
                                               armnn::DataType::Float16,
                                               armnn::DataType::Float32>(reasonIfUnsupported);

    CHECK(result);
}

TEST_CASE("IsConvertFp16ToFp32SupportedFp32InputNpu") {
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::NpuWorkloadFactory,
                                               armnn::ConvertFp16ToFp32Layer,
                                               armnn::DataType::Float32,
                                               armnn::DataType::Float32>(reasonIfUnsupported);

    CHECK(!result);
    CHECK_EQ(reasonIfUnsupported, "Layer is not supported with float32 data type input");
}

TEST_CASE("IsConvertFp16ToFp32SupportedFp16OutputNpu") {
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::NpuWorkloadFactory,
                                               armnn::ConvertFp16ToFp32Layer,
                                               armnn::DataType::Float16,
                                               armnn::DataType::Float16>(reasonIfUnsupported);

    CHECK(!result);
    CHECK_EQ(reasonIfUnsupported, "Layer is not supported with float16 data type output");
}

TEST_CASE("IsConvertFp32ToFp16SupportedNpu") {
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::NpuWorkloadFactory,
                                               armnn::ConvertFp32ToFp16Layer,
                                               armnn::DataType::Float32,
                                               armnn::DataType::Float16>(reasonIfUnsupported);

    CHECK(result);
}

TEST_CASE("IsConvertFp32ToFp16SupportedFp16InputNpu") {
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::NpuWorkloadFactory,
                                               armnn::ConvertFp32ToFp16Layer,
                                               armnn::DataType::Float16,
                                               armnn::DataType::Float16>(reasonIfUnsupported);

    CHECK(!result);
    CHECK_EQ(reasonIfUnsupported, "Layer is not supported with float16 data type input");
}

TEST_CASE("IsConvertFp32ToFp16SupportedFp32OutputNpu") {
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::NpuWorkloadFactory,
                                               armnn::ConvertFp32ToFp16Layer,
                                               armnn::DataType::Float32,
                                               armnn::DataType::Float32>(reasonIfUnsupported);

    CHECK(!result);
    CHECK_EQ(reasonIfUnsupported, "Layer is not supported with float32 data type output");
}

TEST_CASE("IsLayerSupportedMeanDimensionsNpu") {
    std::string reasonIfUnsupported;

    bool result = IsMeanLayerSupportedTests<armnn::NpuWorkloadFactory,
                                            armnn::DataType::Float32,
                                            armnn::DataType::Float32>(reasonIfUnsupported);

    CHECK(result);
}

TEST_CASE("IsLayerNotSupportedMeanDimensionsNpu") {
    std::string reasonIfUnsupported;

    bool result = IsMeanLayerNotSupportedTests<armnn::NpuWorkloadFactory,
                                               armnn::DataType::Float32,
                                               armnn::DataType::Float32>(reasonIfUnsupported);

    CHECK(!result);

    CHECK(reasonIfUnsupported.find(
        "Npu Mean: Expected 4 dimensions but got 2 dimensions instead, for the 'output' tensor.")
        != std::string::npos);
}

}
