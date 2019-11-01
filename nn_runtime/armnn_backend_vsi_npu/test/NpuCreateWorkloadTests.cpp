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

#include <test/CreateWorkload.hpp>
#include <NpuWorkloadFactory.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>

#include "workloads/NpuActivationWorkload.hpp"
#include "workloads/NpuBatchNormalizationWorkload.hpp"
#include "workloads/NpuConcatWorkload.hpp"
#include "workloads/NpuConvolution2dWorkload.hpp"
#include "workloads/NpuElementwiseWorkload.hpp"
#include "workloads/NpuFullyConnectedWorkload.hpp"
#include "workloads/NpuL2NormalizationWorkload.hpp"
#include "workloads/NpuNormalizationWorkload.hpp"
#include "workloads/NpuPooling2dWorkload.hpp"
#include "workloads/NpuReshapeWorkload.hpp"
#include "workloads/NpuSoftmaxWorkload.hpp"
#include "workloads/NpuDepthWiseConvolution2dWorkload.hpp"

using FactoryType = NpuWorkloadFactory;

namespace {

template <typename Workload>
void CheckInputOutput(std::unique_ptr<Workload> workload,
                      const TensorInfo& inputInfo,
                      const TensorInfo& outputInfo) {
    auto queueDescriptor = workload->GetData();
    auto inputHandle = boost::polymorphic_downcast<NpuTensorHandler*>(queueDescriptor.m_Inputs[0]);
    auto outputHandle =
        boost::polymorphic_downcast<NpuTensorHandler*>(queueDescriptor.m_Outputs[0]);
    BOOST_TEST((inputHandle->GetTensorInfo() == inputInfo));
    BOOST_TEST((outputHandle->GetTensorInfo() == outputInfo));
}

template <typename Workload>
void CheckInputsOutput(std::unique_ptr<Workload> workload,
                       const TensorInfo& inputInfo0,
                       const TensorInfo& inputInfo1,
                       const TensorInfo& outputInfo) {
    auto queueDescriptor = workload->GetData();
    auto inputHandle0 = boost::polymorphic_downcast<NpuTensorHandler*>(queueDescriptor.m_Inputs[0]);
    auto inputHandle1 = boost::polymorphic_downcast<NpuTensorHandler*>(queueDescriptor.m_Inputs[1]);
    auto outputHandle =
        boost::polymorphic_downcast<NpuTensorHandler*>(queueDescriptor.m_Outputs[0]);
    BOOST_TEST((inputHandle0->GetTensorInfo() == inputInfo0));
    BOOST_TEST((inputHandle1->GetTensorInfo() == inputInfo1));
    BOOST_TEST((outputHandle->GetTensorInfo() == outputInfo));
}
}

BOOST_AUTO_TEST_SUITE(VSI_Backend)

template <typename ActivationWorkloadType, armnn::DataType DataType>
static void NpuCreateActivationWorkloadTest() {
    Graph graph;
    NpuWorkloadFactory factory;
    auto workload = CreateActivationWorkloadTest<ActivationWorkloadType, DataType>(factory, graph);

    // Checks that outputs are as we expect them (see definition of CreateActivationWorkloadTest).
    CheckInputOutput(
        std::move(workload), TensorInfo({1, 1}, DataType), TensorInfo({1, 1}, DataType));
}

BOOST_AUTO_TEST_CASE(CreateActivationFloat32Workload) {
    NpuCreateActivationWorkloadTest<NpuActivationFloat32Workload, armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateActivationUint8Workload) {
    NpuCreateActivationWorkloadTest<NpuActivationUint8Workload, armnn::DataType::QuantisedAsymm8>();
}

template <typename WorkloadType,
          typename DescriptorType,
          typename LayerType,
          armnn::DataType DataType>
static void NpuCreateElementwiseWorkloadTest() {
    Graph graph;
    NpuWorkloadFactory factory;
    auto workload =
        CreateElementwiseWorkloadTest<WorkloadType, DescriptorType, LayerType, DataType>(factory,
                                                                                         graph);

    CheckInputsOutput(std::move(workload),
                      TensorInfo({2, 3}, DataType),
                      TensorInfo({2, 3}, DataType),
                      TensorInfo({2, 3}, DataType));
}

BOOST_AUTO_TEST_CASE(CreateAdditionFloatWorkload) {
    NpuCreateElementwiseWorkloadTest<NpuAdditionFloat32Workload,
                                     AdditionQueueDescriptor,
                                     AdditionLayer,
                                     armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateAdditionUint8Workload) {
    NpuCreateElementwiseWorkloadTest<NpuAdditionUint8Workload,
                                     AdditionQueueDescriptor,
                                     AdditionLayer,
                                     armnn::DataType::QuantisedAsymm8>();
}

// BOOST_AUTO_TEST_CASE(CreateAdditionInt16Workload)
// {
//     RefCreateElementwiseWorkloadTest<RefAdditionWorkload,
//         AdditionQueueDescriptor,
//         AdditionLayer,
//         armnn::DataType::QuantisedSymm16>();
// }

BOOST_AUTO_TEST_CASE(CreateSubtractionFloatWorkload) {
    NpuCreateElementwiseWorkloadTest<NpuSubtractionFloat32Workload,
                                     SubtractionQueueDescriptor,
                                     SubtractionLayer,
                                     armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateSubtractionUint8Workload) {
    NpuCreateElementwiseWorkloadTest<NpuSubtractionUint8Workload,
                                     SubtractionQueueDescriptor,
                                     SubtractionLayer,
                                     armnn::DataType::QuantisedAsymm8>();
}

// BOOST_AUTO_TEST_CASE(CreateSubtractionInt16Workload)
// {
//     RefCreateElementwiseWorkloadTest<RefSubtractionWorkload,
//         SubtractionQueueDescriptor,
//         SubtractionLayer,
//         armnn::DataType::QuantisedSymm16>();
// }

BOOST_AUTO_TEST_CASE(CreateMultiplicationFloatWorkload) {
    NpuCreateElementwiseWorkloadTest<NpuMultiplicationFloat32Workload,
                                     MultiplicationQueueDescriptor,
                                     MultiplicationLayer,
                                     armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateMultiplicationUint8Workload) {
    NpuCreateElementwiseWorkloadTest<NpuMultiplicationUint8Workload,
                                     MultiplicationQueueDescriptor,
                                     MultiplicationLayer,
                                     armnn::DataType::QuantisedAsymm8>();
}

// BOOST_AUTO_TEST_CASE(CreateMultiplicationInt16Workload)
// {
//     RefCreateElementwiseWorkloadTest<RefMultiplicationWorkload,
//         MultiplicationQueueDescriptor,
//         MultiplicationLayer,
//         armnn::DataType::QuantisedSymm16>();
// }

BOOST_AUTO_TEST_CASE(CreateDivisionFloatWorkload) {
    NpuCreateElementwiseWorkloadTest<NpuDivisionFloat32Workload,
                                     DivisionQueueDescriptor,
                                     DivisionLayer,
                                     armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateDivisionUint8Workload) {
    NpuCreateElementwiseWorkloadTest<NpuDivisionUint8Workload,
                                     DivisionQueueDescriptor,
                                     DivisionLayer,
                                     armnn::DataType::QuantisedAsymm8>();
}

// BOOST_AUTO_TEST_CASE(CreateDivisionInt16Workload)
// {
//     RefCreateElementwiseWorkloadTest<RefDivisionWorkload,
//         DivisionQueueDescriptor,
//         DivisionLayer,
//         armnn::DataType::QuantisedSymm16>();
// }

template <typename BatchNormalizationWorkloadType, armnn::DataType DataType>
static void NpuCreateBatchNormalizationWorkloadTest(DataLayout dataLayout) {
    Graph graph;
    NpuWorkloadFactory factory;
    auto workload = CreateBatchNormalizationWorkloadTest<BatchNormalizationWorkloadType, DataType>(
        factory, graph, dataLayout);

    TensorShape inputShape;
    TensorShape outputShape;

    switch (dataLayout) {
        case DataLayout::NHWC:
            inputShape = {2, 4, 4, 3};
            outputShape = {2, 4, 4, 3};
            break;
        case DataLayout::NCHW:
        default:
            inputShape = {2, 3, 4, 4};
            outputShape = {2, 3, 4, 4};
            break;
    }

    // Checks that outputs and inputs are as we expect them (see definition of
    // CreateBatchNormalizationWorkloadTest).
    CheckInputOutput(
        std::move(workload), TensorInfo(inputShape, DataType), TensorInfo(outputShape, DataType));
}

BOOST_AUTO_TEST_CASE(CreateBatchNormalizationFloat32Workload) {
    NpuCreateBatchNormalizationWorkloadTest<NpuBatchNormalizationFloat32Workload,
                                            armnn::DataType::Float32>(DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreateBatchNormalizationFloat32WorkloadNhwc) {
    NpuCreateBatchNormalizationWorkloadTest<NpuBatchNormalizationFloat32Workload,
                                            armnn::DataType::Float32>(DataLayout::NHWC);
}

BOOST_AUTO_TEST_CASE(CreateBatchNormalizationUint8Workload) {
    NpuCreateBatchNormalizationWorkloadTest<NpuBatchNormalizationUint8Workload,
                                            armnn::DataType::QuantisedAsymm8>(DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreateBatchNormalizationUint8WorkloadNhwc) {
    NpuCreateBatchNormalizationWorkloadTest<NpuBatchNormalizationUint8Workload,
                                            armnn::DataType::QuantisedAsymm8>(DataLayout::NHWC);
}

// BOOST_AUTO_TEST_CASE(CreateBatchNormalizationInt16Workload)
// {
//     RefCreateBatchNormalizationWorkloadTest<RefBatchNormalizationWorkload,
//     armnn::DataType::QuantisedSymm16>
//             (DataLayout::NCHW);
// }

// BOOST_AUTO_TEST_CASE(CreateBatchNormalizationInt16WorkloadNhwc)
// {
//     RefCreateBatchNormalizationWorkloadTest<RefBatchNormalizationWorkload,
//     armnn::DataType::QuantisedSymm16>
//             (DataLayout::NHWC);
// }

// BOOST_AUTO_TEST_CASE(CreateConvertFp16ToFp32Float32Workload)
// {
//     Graph                graph;
//     NpuWorkloadFactory factory;
//     auto workload = CreateConvertFp16ToFp32WorkloadTest<RefConvertFp16ToFp32Workload>(factory,
//     graph);

//     // Checks that outputs and inputs are as we expect them
//     CheckInputOutput(
//         std::move(workload), TensorInfo({1, 3, 2, 3}, DataType::Float16), TensorInfo({1, 3, 2,
//         3}, DataType::Float32));
// }

// BOOST_AUTO_TEST_CASE(CreateConvertFp32ToFp16Float16Workload)
// {
//     Graph                graph;
//     NpuWorkloadFactory factory;
//     auto workload = CreateConvertFp32ToFp16WorkloadTest<RefConvertFp32ToFp16Workload>(factory,
//     graph);

//     // Checks that outputs and inputs are as we expect them
//     CheckInputOutput(
//         std::move(workload), TensorInfo({1, 3, 2, 3}, DataType::Float32), TensorInfo({1, 3, 2,
//         3}, DataType::Float16));
// }

static void NpuCreateConvolution2dWorkloadTest(DataLayout dataLayout = DataLayout::NCHW) {
    Graph graph;
    NpuWorkloadFactory factory;
    auto workload =
        CreateConvolution2dWorkloadTest<NpuConvolution2dFloat32Workload, DataType::Float32>(
            factory, graph, dataLayout);

    TensorShape inputShape = (dataLayout == DataLayout::NCHW)
                                 ? std::initializer_list<unsigned int>({2, 3, 8, 16})
                                 : std::initializer_list<unsigned int>({2, 8, 16, 3});
    TensorShape outputShape = (dataLayout == DataLayout::NCHW)
                                  ? std::initializer_list<unsigned int>({2, 2, 2, 10})
                                  : std::initializer_list<unsigned int>({2, 2, 10, 2});

    // Checks that outputs and inputs are as we expect them (see definition of
    // CreateConvolution2dWorkloadTest).
    CheckInputOutput(std::move(workload),
                     TensorInfo(inputShape, DataType::Float32),
                     TensorInfo(outputShape, DataType::Float32));
}

BOOST_AUTO_TEST_CASE(CreateConvolution2dFloatNchwWorkload) {
    NpuCreateConvolution2dWorkloadTest(DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreateConvolution2dFloatNhwcWorkload) {
    NpuCreateConvolution2dWorkloadTest(DataLayout::NHWC);
}

static void NpuCreateDepthwiseConvolutionWorkloadTest(DataLayout dataLayout)
{
    Graph graph;
    NpuWorkloadFactory factory;
    auto workload =
    CreateDepthwiseConvolution2dWorkloadTest<NpuDepthWiseConvolution2dFloat32Workload,
    DataType::Float32>
            (factory, graph, dataLayout);

    TensorShape inputShape  = (dataLayout == DataLayout::NCHW) ? std::initializer_list<unsigned
    int>({ 2, 2, 5, 5 })
                                                               : std::initializer_list<unsigned
                                                               int>({ 2, 5, 5, 2 });
    TensorShape outputShape = (dataLayout == DataLayout::NCHW) ? std::initializer_list<unsigned
    int>({ 2, 2, 5, 5 })
                                                               : std::initializer_list<unsigned
                                                               int>({ 2, 5, 5, 2 });

    // Checks that inputs/outputs are as we expect them (see definition of
    // CreateDepthwiseConvolution2dWorkloadTest).
    CheckInputOutput(std::move(workload),
                     TensorInfo(inputShape, DataType::Float32),
                     TensorInfo(outputShape, DataType::Float32));
}

BOOST_AUTO_TEST_CASE(CreateDepthwiseConvolutionFloat32NhwcWorkload)
{
    NpuCreateDepthwiseConvolutionWorkloadTest(DataLayout::NHWC);
}

template <typename FullyConnectedWorkloadType, armnn::DataType DataType>
static void NpuCreateFullyConnectedWorkloadTest()
{
    Graph graph;
    NpuWorkloadFactory factory;
    auto workload = CreateFullyConnectedWorkloadTest<FullyConnectedWorkloadType,
    DataType>(factory, graph);

    // Checks that outputs and inputs are as we expect them (see definition of
    // CreateFullyConnectedWorkloadTest).
    float inputsQScale = DataType == armnn::DataType::QuantisedAsymm8 ? 1.0f : 0.0;
    float outputQScale = DataType == armnn::DataType::QuantisedAsymm8 ? 2.0f : 0.0;
    CheckInputOutput(std::move(workload),
        TensorInfo({ 3, 1, 4, 5 }, DataType, inputsQScale),
        TensorInfo({ 3, 7 }, DataType, outputQScale));
}

BOOST_AUTO_TEST_CASE(CreateFullyConnectedWorkloadFloat32)
{
    NpuCreateFullyConnectedWorkloadTest<NpuFullyConnectedFloat32Workload, armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateFullyConnectedWorkloadQuantisedAsymm8)
{
    NpuCreateFullyConnectedWorkloadTest<NpuFullyConnectedUint8Workload,
    armnn::DataType::QuantisedAsymm8>();
}

// BOOST_AUTO_TEST_CASE(CreateFullyConnectedWorkloadQuantisedAsymm16)
// {
//     RefCreateFullyConnectedWorkloadTest<RefFullyConnectedWorkload,
//     armnn::DataType::QuantisedSymm16>();
// }

template <typename NormalizationWorkloadType, armnn::DataType DataType>
static void NpuCreateNormalizationWorkloadTest(DataLayout dataLayout)
{
    Graph graph;
    NpuWorkloadFactory factory;
    auto workload = CreateNormalizationWorkloadTest<NormalizationWorkloadType, DataType>(factory,
    graph, dataLayout);

    TensorShape inputShape;
    TensorShape outputShape;

    switch (dataLayout)
    {
        case DataLayout::NHWC:
            inputShape  = { 3, 1, 5, 5 };
            outputShape = { 3, 1, 5, 5 };
            break;
        case DataLayout::NCHW:
        default:
            inputShape  = { 3, 5, 5, 1 };
            outputShape = { 3, 5, 5, 1 };
            break;
    }

    // Checks that outputs and inputs are as we expect them (see definition of
    // CreateNormalizationWorkloadTest).
    CheckInputOutput(std::move(workload), TensorInfo(inputShape, DataType),
    TensorInfo(outputShape, DataType));
}

BOOST_AUTO_TEST_CASE(CreateRefNormalizationFloat32NchwWorkload)
{
    NpuCreateNormalizationWorkloadTest<NpuNormalizationFloat32Workload,
    armnn::DataType::Float32>(DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreateRefNormalizationFloat32NhwcWorkload)
{
    NpuCreateNormalizationWorkloadTest<NpuNormalizationFloat32Workload,
    armnn::DataType::Float32>(DataLayout::NHWC);
}

BOOST_AUTO_TEST_CASE(CreateRefNormalizationUint8NchwWorkload)
{
    NpuCreateNormalizationWorkloadTest<NpuNormalizationUint8Workload,
    armnn::DataType::QuantisedAsymm8>(DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreateRefNormalizationUint8NhwcWorkload)
{
    NpuCreateNormalizationWorkloadTest<NpuNormalizationUint8Workload,
    armnn::DataType::QuantisedAsymm8>(DataLayout::NHWC);
}

// BOOST_AUTO_TEST_CASE(CreateRefNormalizationInt16NchwWorkload)
// {
//     RefCreateNormalizationWorkloadTest<RefNormalizationWorkload,
//     armnn::DataType::QuantisedSymm16>(DataLayout::NCHW);
// }

// BOOST_AUTO_TEST_CASE(CreateRefNormalizationInt16NhwcWorkload)
// {
//     RefCreateNormalizationWorkloadTest<RefNormalizationWorkload,
//     armnn::DataType::QuantisedSymm16>(DataLayout::NHWC);
// }

template <typename Pooling2dWorkloadType, armnn::DataType DataType>
static void NpuCreatePooling2dWorkloadTest(DataLayout dataLayout)
{
    Graph graph;
    NpuWorkloadFactory factory;
    auto workload = CreatePooling2dWorkloadTest<Pooling2dWorkloadType, DataType>(factory, graph,
    dataLayout);

    TensorShape inputShape;
    TensorShape outputShape;

    switch (dataLayout)
    {
        case DataLayout::NHWC:
            inputShape  = { 3, 5, 5, 2 };
            outputShape = { 3, 2, 4, 2 };
            break;
        case DataLayout::NCHW:
        default:
            inputShape =  { 3, 2, 5, 5 };
            outputShape = { 3, 2, 2, 4 };
    }

    // Checks that outputs and inputs are as we expect them (see definition of
    // CreatePooling2dWorkloadTest).
    CheckInputOutput(std::move(workload),
                     TensorInfo(inputShape, DataType),
                     TensorInfo(outputShape, DataType));
}

BOOST_AUTO_TEST_CASE(CreatePooling2dFloat32Workload)
{
    NpuCreatePooling2dWorkloadTest<NpuPooling2dFloat32Workload,
    armnn::DataType::Float32>(DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreatePooling2dFloat32NhwcWorkload)
{
    NpuCreatePooling2dWorkloadTest<NpuPooling2dFloat32Workload,
    armnn::DataType::Float32>(DataLayout::NHWC);
}

BOOST_AUTO_TEST_CASE(CreatePooling2dUint8Workload)
{
    NpuCreatePooling2dWorkloadTest<NpuPooling2dUint8Workload,
    armnn::DataType::QuantisedAsymm8>(DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreatePooling2dUint8NhwcWorkload)
{
    NpuCreatePooling2dWorkloadTest<NpuPooling2dUint8Workload,
    armnn::DataType::QuantisedAsymm8>(DataLayout::NHWC);
}

// BOOST_AUTO_TEST_CASE(CreatePooling2dInt16Workload)
// {
//     RefCreatePooling2dWorkloadTest<RefPooling2dWorkload,
//     armnn::DataType::QuantisedSymm16>(DataLayout::NCHW);
// }

// BOOST_AUTO_TEST_CASE(CreatePooling2dInt16NhwcWorkload)
// {
//     RefCreatePooling2dWorkloadTest<RefPooling2dWorkload,
//     armnn::DataType::QuantisedSymm16>(DataLayout::NHWC);
// }

template <typename SoftmaxWorkloadType, armnn::DataType DataType>
static void NpuCreateSoftmaxWorkloadTest()
{
    Graph graph;
    NpuWorkloadFactory factory;
    auto workload = CreateSoftmaxWorkloadTest<SoftmaxWorkloadType, DataType>(factory, graph);

    // Checks that outputs and inputs are as we expect them (see definition of
    // CreateSoftmaxWorkloadTest).
    CheckInputOutput(
        std::move(workload),
        TensorInfo({4, 1}, DataType),
        TensorInfo({4, 1}, DataType));
}

BOOST_AUTO_TEST_CASE(CreateSoftmaxFloat32Workload)
{
    NpuCreateSoftmaxWorkloadTest<NpuSoftmaxFloat32Workload, armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateSoftmaxQuantisedAsymm8Workload)
{
    NpuCreateSoftmaxWorkloadTest<NpuSoftmaxUint8Workload, armnn::DataType::QuantisedAsymm8>();
}

// BOOST_AUTO_TEST_CASE(CreateSoftmaxQuantisedSymm16Workload)
// {
//     RefCreateSoftmaxWorkloadTest<RefSoftmaxWorkload, armnn::DataType::QuantisedSymm16>();
// }

// template <typename SplitterWorkloadType, armnn::DataType DataType>
// static void RefCreateSplitterWorkloadTest()
// {
//     Graph graph;
//     NpuWorkloadFactory factory;
//     auto workload = CreateSplitterWorkloadTest<SplitterWorkloadType, DataType>(factory, graph);

//     // Checks that outputs are as we expect them (see definition of CreateSplitterWorkloadTest).
//     SplitterQueueDescriptor queueDescriptor = workload->GetData();
//     auto inputHandle =
//     boost::polymorphic_downcast<ConstCpuTensorHandle*>(queueDescriptor.m_Inputs[0]);
//     BOOST_TEST((inputHandle->GetTensorInfo() == TensorInfo({ 5, 7, 7 }, DataType)));

//     auto outputHandle0 =
//     boost::polymorphic_downcast<CpuTensorHandle*>(queueDescriptor.m_Outputs[0]);
//     BOOST_TEST((outputHandle0->GetTensorInfo() == TensorInfo({ 1, 7, 7 }, DataType)));

//     auto outputHandle1 =
//     boost::polymorphic_downcast<CpuTensorHandle*>(queueDescriptor.m_Outputs[1]);
//     BOOST_TEST((outputHandle1->GetTensorInfo() == TensorInfo({ 2, 7, 7 }, DataType)));

//     auto outputHandle2 =
//     boost::polymorphic_downcast<CpuTensorHandle*>(queueDescriptor.m_Outputs[2]);
//     BOOST_TEST((outputHandle2->GetTensorInfo() == TensorInfo({ 2, 7, 7 }, DataType)));
// }

// BOOST_AUTO_TEST_CASE(CreateSplitterFloat32Workload)
// {
//     RefCreateSplitterWorkloadTest<RefSplitterWorkload, armnn::DataType::Float32>();
// }

// BOOST_AUTO_TEST_CASE(CreateSplitterUint8Workload)
// {
//     RefCreateSplitterWorkloadTest<RefSplitterWorkload, armnn::DataType::QuantisedAsymm8>();
// }

// template <typename SplitterWorkloadType, typename ConcatWorkloadType, armnn::DataType DataType>
// static void RefCreateSplitterConcatWorkloadTest()
// {
//     // Tests that it is possible to decide which output of the splitter layer
//     // should be lined to which input of the concat layer.
//     // We tested that is is possible to specify 0th output
//     // of the splitter to be the 1st input to the concat and the 1st output of the splitter to be
//     0th input
//     // of the concat.

//     Graph graph;
//     NpuWorkloadFactory factory;
//     auto workloads = CreateSplitterConcatWorkloadTest<SplitterWorkloadType, ConcatWorkloadType,
//     DataType>
//             (factory, graph);

//     auto wlSplitter = std::move(workloads.first);
//     auto wlConcat = std::move(workloads.second);

//     //Checks that the index of inputs/outputs matches what we declared on InputDescriptor
//     construction.
//     armnn::CpuTensorHandle* sOut0 =
//     dynamic_cast<armnn::CpuTensorHandle*>(wlSplitter->GetData().m_Outputs[0]);
//     armnn::CpuTensorHandle* sOut1 =
//     dynamic_cast<armnn::CpuTensorHandle*>(wlSplitter->GetData().m_Outputs[1]);
//     armnn::CpuTensorHandle* mIn0 =
//     dynamic_cast<armnn::CpuTensorHandle*>(wlConcat->GetData().m_Inputs[0]);
//     armnn::CpuTensorHandle* mIn1 =
//     dynamic_cast<armnn::CpuTensorHandle*>(wlConcat->GetData().m_Inputs[1]);

//     BOOST_TEST(sOut0);
//     BOOST_TEST(sOut1);
//     BOOST_TEST(mIn0);
//     BOOST_TEST(mIn1);

//     bool validDataPointers = (sOut0 == mIn1) && (sOut1 == mIn0);

//     BOOST_TEST(validDataPointers);
// }

// BOOST_AUTO_TEST_CASE(CreateSplitterConcatFloat32)
// {
//     RefCreateSplitterConcatWorkloadTest<RefSplitterWorkload, RefConcatWorkload,
//     DataType::Float32>();
// }

// BOOST_AUTO_TEST_CASE(CreateSplitterConcatUint8)
// {
//     RefCreateSplitterConcatWorkloadTest<RefSplitterWorkload, RefConcatWorkload,
//     DataType::QuantisedAsymm8>();
// }

// template <typename SplitterWorkloadType, typename ActivationWorkloadType, armnn::DataType
// DataType>
// static void RefCreateSingleOutputMultipleInputsTest()
// {
//     // Tests that it is possible to assign multiple (two) different layers to each of the outputs
//     of a splitter layer.
//     // We created a splitter with two outputs. That each of those outputs is used by two
//     different activation layers.

//     Graph graph;
//     NpuWorkloadFactory factory;
//     std::unique_ptr<SplitterWorkloadType> wlSplitter;
//     std::unique_ptr<ActivationWorkloadType> wlActiv0_0;
//     std::unique_ptr<ActivationWorkloadType> wlActiv0_1;
//     std::unique_ptr<ActivationWorkloadType> wlActiv1_0;
//     std::unique_ptr<ActivationWorkloadType> wlActiv1_1;

//     CreateSplitterMultipleInputsOneOutputWorkloadTest<SplitterWorkloadType,
//         ActivationWorkloadType, DataType>(factory, graph, wlSplitter, wlActiv0_0, wlActiv0_1,
//         wlActiv1_0, wlActiv1_1);

//     armnn::CpuTensorHandle* sOut0 =
//     dynamic_cast<armnn::CpuTensorHandle*>(wlSplitter->GetData().m_Outputs[0]);
//     armnn::CpuTensorHandle* sOut1 =
//     dynamic_cast<armnn::CpuTensorHandle*>(wlSplitter->GetData().m_Outputs[1]);
//     armnn::CpuTensorHandle* activ0_0Im =
//     dynamic_cast<armnn::CpuTensorHandle*>(wlActiv0_0->GetData().m_Inputs[0]);
//     armnn::CpuTensorHandle* activ0_1Im =
//     dynamic_cast<armnn::CpuTensorHandle*>(wlActiv0_1->GetData().m_Inputs[0]);
//     armnn::CpuTensorHandle* activ1_0Im =
//     dynamic_cast<armnn::CpuTensorHandle*>(wlActiv1_0->GetData().m_Inputs[0]);
//     armnn::CpuTensorHandle* activ1_1Im =
//     dynamic_cast<armnn::CpuTensorHandle*>(wlActiv1_1->GetData().m_Inputs[0]);

//     BOOST_TEST(sOut0);
//     BOOST_TEST(sOut1);
//     BOOST_TEST(activ0_0Im);
//     BOOST_TEST(activ0_1Im);
//     BOOST_TEST(activ1_0Im);
//     BOOST_TEST(activ1_1Im);

//     bool validDataPointers = (sOut0 == activ0_0Im) && (sOut0 == activ0_1Im) &&
//                              (sOut1 == activ1_0Im) && (sOut1 == activ1_1Im);

//     BOOST_TEST(validDataPointers);
// }

// BOOST_AUTO_TEST_CASE(CreateSingleOutputMultipleInputsFloat32)
// {
//     RefCreateSingleOutputMultipleInputsTest<RefSplitterWorkload, RefActivationWorkload,
//         armnn::DataType::Float32>();
// }

// BOOST_AUTO_TEST_CASE(CreateSingleOutputMultipleInputsUint8)
// {
//     RefCreateSingleOutputMultipleInputsTest<RefSplitterWorkload, RefActivationWorkload,
//         armnn::DataType::QuantisedAsymm8>();
// }

// template <typename ResizeBilinearWorkloadType, armnn::DataType DataType>
// static void RefCreateResizeBilinearTest(DataLayout dataLayout)
// {
//     Graph graph;
//     NpuWorkloadFactory factory;
//     auto workload = CreateResizeBilinearWorkloadTest<ResizeBilinearWorkloadType,
//     DataType>(factory, graph, dataLayout);

//     TensorShape inputShape;
//     TensorShape outputShape;

//     switch (dataLayout)
//     {
//         case DataLayout::NHWC:
//             inputShape  = { 2, 4, 4, 3 };
//             outputShape = { 2, 2, 2, 3 };
//             break;
//         case DataLayout::NCHW:
//         default:
//             inputShape  = { 2, 3, 4, 4 };
//             outputShape = { 2, 3, 2, 2 };
//     }

//     // Checks that outputs and inputs are as we expect them (see definition of
//     CreateResizeBilinearWorkloadTest).
//     CheckInputOutput(std::move(workload),
//                      TensorInfo(inputShape, DataType),
//                      TensorInfo(outputShape, DataType));
// }

// BOOST_AUTO_TEST_CASE(CreateResizeBilinearFloat32)
// {
//     RefCreateResizeBilinearTest<RefResizeBilinearWorkload,
//     armnn::DataType::Float32>(DataLayout::NCHW);
// }

// BOOST_AUTO_TEST_CASE(CreateResizeBilinearUint8)
// {
//     RefCreateResizeBilinearTest<RefResizeBilinearWorkload,
//     armnn::DataType::QuantisedAsymm8>(DataLayout::NCHW);
// }

// BOOST_AUTO_TEST_CASE(CreateResizeBilinearQuantisedAsymm16)
// {
//     RefCreateResizeBilinearTest<RefResizeBilinearWorkload,
//     armnn::DataType::QuantisedSymm16>(DataLayout::NCHW);
// }

// BOOST_AUTO_TEST_CASE(CreateResizeBilinearFloat32Nhwc)
// {
//     RefCreateResizeBilinearTest<RefResizeBilinearWorkload,
//     armnn::DataType::Float32>(DataLayout::NHWC);
// }

// template <typename RsqrtWorkloadType, armnn::DataType DataType>
// static void NpuCreateRsqrtTest()
// {
//     Graph graph;
//     NpuWorkloadFactory factory;

//     auto workload = CreateRsqrtWorkloadTest<RsqrtWorkloadType, DataType>(factory, graph);

//     // Checks that outputs are as we expect them (see definition of CreateRsqrtWorkloadTest).
//     CheckInputOutput(std::move(workload),
//                      TensorInfo({ 1, 1 }, DataType),
//                      TensorInfo({ 1, 1 }, DataType));

// }

// BOOST_AUTO_TEST_CASE(CreateRsqrtFloat32)
// {
//     NpuCreateRsqrtTest<NpuRsqrtFloat32Workload, armnn::DataType::Float32>();
// }

// BOOST_AUTO_TEST_CASE(CreateRsqrtUint8)
// {
//     NpuCreateRsqrtTest<NpuRsqrtUint8Workload, armnn::DataType::QuantisedAsymm8>();
// }

// BOOST_AUTO_TEST_CASE(CreateRsqrtQsymm16)
// {
//     RefCreateRsqrtTest<RefRsqrtWorkload, armnn::DataType::QuantisedSymm16>();
// }

template <typename L2NormalizationWorkloadType, armnn::DataType DataType>
static void NpuCreateL2NormalizationTest(DataLayout dataLayout)
{
    Graph graph;
    NpuWorkloadFactory factory;
    auto workload =
            CreateL2NormalizationWorkloadTest<L2NormalizationWorkloadType, DataType>(factory,
            graph, dataLayout);

    TensorShape inputShape;
    TensorShape outputShape;

    switch (dataLayout)
    {
        case DataLayout::NHWC:
            inputShape  = { 5, 50, 67, 20 };
            outputShape = { 5, 50, 67, 20 };
            break;
        case DataLayout::NCHW:
        default:
            inputShape  = { 5, 20, 50, 67 };
            outputShape = { 5, 20, 50, 67 };
            break;
    }

    // Checks that outputs and inputs are as we expect them (see definition of
    // CreateL2NormalizationWorkloadTest).
    CheckInputOutput(std::move(workload), TensorInfo(inputShape, DataType),
    TensorInfo(outputShape, DataType));
}

BOOST_AUTO_TEST_CASE(CreateL2NormalizationFloat32)
{
    NpuCreateL2NormalizationTest<NpuL2NormalizationFloat32Workload,
    armnn::DataType::Float32>(DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreateL2NormalizationFloat32Nhwc)
{
    NpuCreateL2NormalizationTest<NpuL2NormalizationFloat32Workload,
    armnn::DataType::Float32>(DataLayout::NHWC);
}

BOOST_AUTO_TEST_CASE(CreateL2NormalizationUint8)
{
    NpuCreateL2NormalizationTest<NpuL2NormalizationUint8Workload,
    armnn::DataType::QuantisedAsymm8>(DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(CreateL2NormalizationUint8Nhwc)
{
    NpuCreateL2NormalizationTest<NpuL2NormalizationUint8Workload,
    armnn::DataType::QuantisedAsymm8>(DataLayout::NHWC);
}

// BOOST_AUTO_TEST_CASE(CreateL2NormalizationInt16)
// {
//     RefCreateL2NormalizationTest<RefL2NormalizationWorkload,
//     armnn::DataType::QuantisedSymm16>(DataLayout::NCHW);
// }

// BOOST_AUTO_TEST_CASE(CreateL2NormalizationInt16Nhwc)
// {
//     RefCreateL2NormalizationTest<RefL2NormalizationWorkload,
//     armnn::DataType::QuantisedSymm16>(DataLayout::NHWC);
// }

template <typename ReshapeWorkloadType, armnn::DataType DataType>
static void NpuCreateReshapeWorkloadTest()
{
    Graph graph;
    NpuWorkloadFactory factory;
    auto workload = CreateReshapeWorkloadTest<ReshapeWorkloadType, DataType>(factory, graph);

    // Checks that outputs and inputs are as we expect them (see definition of
    // CreateReshapeWorkloadTest).
    CheckInputOutput(
        std::move(workload),
        TensorInfo({ 4, 1 }, DataType),
        TensorInfo({ 1, 4 }, DataType));
}

BOOST_AUTO_TEST_CASE(CreateReshapeWorkloadFloat32)
{
    NpuCreateReshapeWorkloadTest<NpuReshapeFloat32Workload, armnn::DataType::Float32>();
}

BOOST_AUTO_TEST_CASE(CreateReshapeWorkloadQuantisedAsymm8)
{
    NpuCreateReshapeWorkloadTest<NpuReshapeUint8Workload, armnn::DataType::QuantisedAsymm8>();
}

// BOOST_AUTO_TEST_CASE(CreateReshapeWorkloadQuantisedSymm16)
// {
//     RefCreateReshapeWorkloadTest<RefReshapeWorkload, armnn::DataType::QuantisedSymm16>();
// }

template <typename ConcatWorkloadType, armnn::DataType DataType>
static void NpuCreateConcatWorkloadTest(const armnn::TensorShape& outputShape,
                                        unsigned int concatAxis)
{
    Graph graph;
    NpuWorkloadFactory factory;
    auto workload = CreateConcatWorkloadTest<ConcatWorkloadType, DataType>(factory, graph,
    outputShape, concatAxis);

    CheckInputsOutput(std::move(workload),
                      TensorInfo({ 2, 3, 2, 5 }, DataType),
                      TensorInfo({ 2, 3, 2, 5 }, DataType),
                      TensorInfo(outputShape, DataType));
}

BOOST_AUTO_TEST_CASE(CreateConcatDim0Float32Workload)
{
    NpuCreateConcatWorkloadTest<NpuConcatFloat32Workload, armnn::DataType::Float32>({ 4, 3, 2, 5 }, 0);
}

BOOST_AUTO_TEST_CASE(CreateConcatDim0Uint8Workload)
{
    NpuCreateConcatWorkloadTest<NpuConcatUint8Workload, armnn::DataType::QuantisedAsymm8>({ 4, 3, 2, 5
    }, 0);
}

// BOOST_AUTO_TEST_CASE(CreateConcatDim0Uint16Workload)
// {
//     RefCreateConcatWorkloadTest<RefConcatWorkload, armnn::DataType::QuantisedSymm16>({ 4, 3, 2, 5
//     }, 0);
// }

BOOST_AUTO_TEST_CASE(CreateConcatDim1Float32Workload)
{
    NpuCreateConcatWorkloadTest<NpuConcatFloat32Workload, armnn::DataType::Float32>({ 2, 6, 2, 5 }, 1);
}

BOOST_AUTO_TEST_CASE(CreateConcatDim1Uint8Workload)
{
    NpuCreateConcatWorkloadTest<NpuConcatUint8Workload, armnn::DataType::QuantisedAsymm8>({ 2, 6, 2, 5
    }, 1);
}

BOOST_AUTO_TEST_CASE(CreateConcatDim2Float32Workload)
{
    NpuCreateConcatWorkloadTest<NpuConcatFloat32Workload, armnn::DataType::Float32>({ 2, 3, 4, 5 }, 2);
}

BOOST_AUTO_TEST_CASE(CreateConcatDim2Uint8Workload)
{
    NpuCreateConcatWorkloadTest<NpuConcatUint8Workload, armnn::DataType::QuantisedAsymm8>({ 2, 3, 4, 5
    }, 2);
}

BOOST_AUTO_TEST_CASE(CreateConcatDim3Float32Workload)
{
    NpuCreateConcatWorkloadTest<NpuConcatFloat32Workload, armnn::DataType::Float32>({ 2, 3, 2, 10 }, 3);
}

BOOST_AUTO_TEST_CASE(CreateConcatDim3Uint8Workload)
{
    NpuCreateConcatWorkloadTest<NpuConcatUint8Workload, armnn::DataType::QuantisedAsymm8>({ 2, 3, 2,
    10 }, 3);
}

// template <typename ConstantWorkloadType, armnn::DataType DataType>
// static void RefCreateConstantWorkloadTest(const armnn::TensorShape& outputShape)
// {
//     armnn::Graph graph;
//     NpuWorkloadFactory factory;
//     auto workload = CreateConstantWorkloadTest<ConstantWorkloadType, DataType>(factory, graph,
//     outputShape);

//     // Check output is as expected
//     auto queueDescriptor = workload->GetData();
//     auto outputHandle =
//     boost::polymorphic_downcast<CpuTensorHandle*>(queueDescriptor.m_Outputs[0]);
//     BOOST_TEST((outputHandle->GetTensorInfo() == TensorInfo(outputShape, DataType)));
// }

// BOOST_AUTO_TEST_CASE(CreateConstantUint8Workload)
// {
//     RefCreateConstantWorkloadTest<RefConstantWorkload, armnn::DataType::QuantisedAsymm8>({ 2, 3,
//     2, 10 });
// }

// BOOST_AUTO_TEST_CASE(CreateConstantInt16Workload)
// {
//     RefCreateConstantWorkloadTest<RefConstantWorkload, armnn::DataType::QuantisedSymm16>({ 2, 3,
//     2, 10 });
// }

// BOOST_AUTO_TEST_CASE(CreateConstantFloat32Workload)
// {
//     RefCreateConstantWorkloadTest<RefConstantWorkload, armnn::DataType::Float32>({ 2, 3, 2, 10
//     });
// }

// BOOST_AUTO_TEST_CASE(CreateConstantSigned32Workload)
// {
//     RefCreateConstantWorkloadTest<RefConstantWorkload, armnn::DataType::Signed32>({ 2, 3, 2, 10
//     });
// }

// template <typename armnn::DataType DataType>
// static void NpuCreatePreluWorkloadTest(const armnn::TensorShape& outputShape)
// {
//     armnn::Graph graph;
//     NpuWorkloadFactory factory;
//     auto workload = CreatePreluWorkloadTest<NpuPreluWorkload, DataType>(factory, graph,
//     outputShape);

//     // Check output is as expected
//     auto queueDescriptor = workload->GetData();
//     auto outputHandle =
//     boost::polymorphic_downcast<CpuTensorHandle*>(queueDescriptor.m_Outputs[0]);
//     BOOST_TEST((outputHandle->GetTensorInfo() == TensorInfo(outputShape, DataType)));
// }

// BOOST_AUTO_TEST_CASE(CreatePreluFloat32Workload)
// {
//     NpuCreatePreluWorkloadTest<armnn::DataType::Float32>({ 5, 4, 3, 2 });
// }

// BOOST_AUTO_TEST_CASE(CreatePreluUint8Workload)
// {
//     NpuCreatePreluWorkloadTest<armnn::DataType::QuantisedAsymm8>({ 5, 4, 3, 2 });
// }

// BOOST_AUTO_TEST_CASE(CreatePreluInt16Workload)
// {
//     RefCreatePreluWorkloadTest<armnn::DataType::QuantisedSymm16>({ 5, 4, 3, 2 });
// }

BOOST_AUTO_TEST_SUITE_END()