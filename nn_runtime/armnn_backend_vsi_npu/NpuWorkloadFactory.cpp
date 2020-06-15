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

#include "NpuWorkloadFactory.hpp"
#include "NpuBackendId.hpp"

#include <backendsCommon/MakeWorkloadHelper.hpp>
#include <backendsCommon/MemCopyWorkload.hpp>

#include "NpuTensorHandler.hpp"
#include "workloads/TNpuWorkloads.hpp"
#include "workloads/NpuFullyConnectedWorkload.hpp"
#include "workloads/NpuPooling2dWorkload.hpp"
#include "workloads/NpuBatchNormalizationWorkload.hpp"
#include "workloads/NpuConvolution2dWorkload.hpp"
#include "workloads/NpuSoftmaxWorkload.hpp"
#include "workloads/NpuActivationWorkload.hpp"
#include "workloads/NpuDepthWiseConvolution2dWorkload.hpp"

#include "workloads/NpuElementwiseWorkload.hpp"
#include "workloads/NpuMeanWorkload.hpp"
#include "workloads/NpuElementwiseUnarytWorkload.hpp"
#include "workloads/NpuPadWorkload.hpp"
#include "workloads/NpuPermuteWorkload.hpp"
#include "workloads/NpuReshapeWorkload.hpp"
#include "workloads/NpuLstmWorkload.hpp"
#include "workloads/NpuStridedSliceWorkload.hpp"
#include "workloads/NpuSplitterWorkload.hpp"
//#include "workloads/NpuStackWorkload.hpp"
#include "workloads/NpuConcatWorkload.hpp"
#include "workloads/NpuSpaceToDepthWorkload.hpp"
#include "workloads/NpuPreluWorkload.hpp"
#include "workloads/NpuTensorCopyWorkload.hpp"
#include "workloads/NpuSpaceToBatchNdWorkload.hpp"
#include "workloads/NpuBatchToSpaceNdWorkload.hpp"
#include "workloads/NpuL2NormalizationWorkload.hpp"
#include "workloads/NpuTransposeConvolution2dWorkload.hpp"
#include "workloads/NpuNormalizationWorkload.hpp"
#include "workloads/NpuResizeWorkload.hpp"
#include "workloads/NpuConstantWorkload.hpp"
#include "workloads/NpuComparisonWorkload.hpp"
#include "workloads/NpuGatherWorkload.hpp"
#include "workloads/NpuDepthToSpaceWorkload.hpp"
#include "workloads/NpuInstanceNormWorkload.hpp"
#include "workloads/NpuDetectionPostProcessWorkload.hpp"
#include "workloads/NpuArgMinMaxWorkload.hpp"
#include "workloads/NpuLogSoftmaxWorkload.hpp"

#include <iostream>


namespace armnn {

namespace {
static const BackendId s_Id{NpuBackendId()};
}
template <typename Fp16Workload, typename F32Workload, typename U8Workload, typename QueueDescriptorType>
std::unique_ptr<IWorkload> NpuWorkloadFactory::MakeWorkload(const QueueDescriptorType& descriptor,
                                                            const WorkloadInfo& info) const {
    return armnn::MakeWorkloadHelper<Fp16Workload,
                                     F32Workload,
                                     U8Workload,
                                     NullWorkload,
                                     NullWorkload,
                                     NullWorkload>(descriptor, info);
}

NpuWorkloadFactory::NpuWorkloadFactory() {}

const BackendId& NpuWorkloadFactory::GetBackendId() const {
    return s_Id;
}

bool NpuWorkloadFactory::IsLayerSupported(const Layer& layer,
                                          Optional<DataType> dataType,
                                          std::string& outReasonIfUnsupported) {
    return IWorkloadFactory::IsLayerSupported(s_Id, layer, dataType, outReasonIfUnsupported);
}

std::unique_ptr<ITensorHandle> NpuWorkloadFactory::CreateTensorHandle(
    const TensorInfo& tensorInfo, const bool IsMemoryManaged) const {
    return CreateTensorHandle(tensorInfo, DataLayout::NHWC);
}

std::unique_ptr<ITensorHandle> NpuWorkloadFactory::CreateTensorHandle(
    const TensorInfo& tensorInfo, DataLayout dataLayout, const bool IsMemoryManaged) const {
    // TODO: add dataLayout for tensor
    return std::make_unique<NpuTensorHandler>(tensorInfo);
}

std::unique_ptr<IWorkload> NpuWorkloadFactory::CreateInput(const InputQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const {
    if (info.m_InputTensorInfos.empty()) {
        throw InvalidArgumentException("RefWorkloadFactory::CreateInput: Input cannot be zero length");
    }
    if (info.m_OutputTensorInfos.empty()) {
        throw InvalidArgumentException("RefWorkloadFactory::CreateInput: Output cannot be zero length");
    }

    if (info.m_InputTensorInfos[0].GetNumBytes() != info.m_OutputTensorInfos[0].GetNumBytes()) {
        throw InvalidArgumentException(
            "RefWorkloadFactory::CreateInput: data input and output differ in byte count.");
    }

    return std::make_unique<CopyMemGenericWorkload>(descriptor, info);

}

std::unique_ptr<IWorkload> NpuWorkloadFactory::CreateOutput(const OutputQueueDescriptor& descriptor,
                                                            const WorkloadInfo& info) const {
    if (info.m_InputTensorInfos.empty()) {
        throw InvalidArgumentException("RefWorkloadFactory::CreateOutput: Input cannot be zero length");
    }
    if (info.m_OutputTensorInfos.empty()) {
        throw InvalidArgumentException("RefWorkloadFactory::CreateOutput: Output cannot be zero length");
    }
    if (info.m_InputTensorInfos[0].GetNumBytes() != info.m_OutputTensorInfos[0].GetNumBytes()) {
        throw InvalidArgumentException(
            "RefWorkloadFactory::CreateOutput: data input and output differ in byte count.");
    }

    return std::make_unique<CopyMemGenericWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NpuWorkloadFactory::CreateActivation(const ActivationQueueDescriptor& descriptor,
                                                                const WorkloadInfo& info) const {
    return MakeWorkload<NpuActivationFloat16Workload, NpuActivationFloat32Workload, NpuActivationUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> NpuWorkloadFactory::CreateSoftmax(const SoftmaxQueueDescriptor& descriptor,
                                                             const WorkloadInfo& info) const {
    return MakeWorkload<NpuSoftmaxFloat16Workload, NpuSoftmaxFloat32Workload, NpuSoftmaxUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> NpuWorkloadFactory::CreateSplitter(
    const SplitterQueueDescriptor& descriptor, const WorkloadInfo& info) const {
    return MakeWorkload<NpuSplitterFloat16Workload,
                        NpuSplitterFloat32Workload,
                        NpuSplitterUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NpuWorkloadFactory::CreateMerger(const MergerQueueDescriptor& descriptor,
                                                                   const WorkloadInfo& info) const {
    return MakeWorkload<NullWorkload, NullWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NpuWorkloadFactory::CreateFullyConnected(
    const FullyConnectedQueueDescriptor& descriptor, const WorkloadInfo& info) const {
    return MakeWorkload<NpuFullyConnectedFloat16Workload,
                        NpuFullyConnectedFloat32Workload,
                        NpuFullyConnectedUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NpuWorkloadFactory::CreatePermute(
    const PermuteQueueDescriptor& descriptor, const WorkloadInfo& info) const {
    return MakeWorkload<NpuPermuteFloat16Workload,
                        NpuPermuteFloat32Workload,
                        NpuPermuteUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NpuWorkloadFactory::CreatePooling2d(
    const Pooling2dQueueDescriptor& descriptor, const WorkloadInfo& info) const {
    return MakeWorkload<NpuPooling2dFloat16Workload,
                        NpuPooling2dFloat32Workload,
                        NpuPooling2dUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NpuWorkloadFactory::CreateConvolution2d(
    const Convolution2dQueueDescriptor& descriptor, const WorkloadInfo& info) const {
    return MakeWorkload<NpuConvolution2dFloat16Workload,
                        NpuConvolution2dFloat32Workload,
                        NpuConvolution2dUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NpuWorkloadFactory::CreateDepthwiseConvolution2d(
    const DepthwiseConvolution2dQueueDescriptor& descriptor, const WorkloadInfo& info) const {
    return MakeWorkload<NpuDepthWiseConvolution2dFloat16Workload,
                        NpuDepthWiseConvolution2dFloat32Workload,
                        NpuDepthWiseConvolution2dUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> NpuWorkloadFactory::CreateDetectionPostProcess(
    const armnn::DetectionPostProcessQueueDescriptor& descriptor,
    const armnn::WorkloadInfo& info) const {
    return MakeWorkload<NullWorkload, NpuDetectionPostProcessFloat32Workload, NullWorkload>(
        descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NpuWorkloadFactory::CreateNormalization(
    const NormalizationQueueDescriptor& descriptor, const WorkloadInfo& info) const {
    return MakeWorkload<NpuNormalizationFloat16Workload,
                        NpuNormalizationFloat32Workload,
                        NpuNormalizationUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NpuWorkloadFactory::CreateAddition(
    const AdditionQueueDescriptor& descriptor, const WorkloadInfo& info) const {
    return MakeWorkload<NpuAdditionFloat16Workload,
                        NpuAdditionFloat32Workload,
                        NpuAdditionUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NpuWorkloadFactory::CreateMultiplication(
    const MultiplicationQueueDescriptor& descriptor, const WorkloadInfo& info) const {
    return MakeWorkload<NpuMultiplicationFloat16Workload,
                        NpuMultiplicationFloat32Workload,
                        NpuMultiplicationUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NpuWorkloadFactory::CreateBatchNormalization(
    const BatchNormalizationQueueDescriptor& descriptor, const WorkloadInfo& info) const {
    return MakeWorkload<NpuBatchNormalizationFloat16Workload,
                        NpuBatchNormalizationFloat32Workload,
                        NpuBatchNormalizationUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NpuWorkloadFactory::CreateMemCopy(
    const MemCopyQueueDescriptor& descriptor, const WorkloadInfo& info) const {
    return MakeWorkload<NpuMemCopyFloat16Workload,
                        NpuMemCopyFloat32Workload,
                        NpuMemCopyUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> NpuWorkloadFactory::CreateResize(const ResizeQueueDescriptor& descriptor,
                                                            const WorkloadInfo& info) const
{
    return MakeWorkload<NpuResizeFloat16Workload,
                        NpuResizeFloat32Workload,
                        NpuResizeUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> NpuWorkloadFactory::CreateResizeBilinear(const ResizeBilinearQueueDescriptor& descriptor,
                                                                    const WorkloadInfo& info) const
{
    ResizeQueueDescriptor resizeDescriptor;
    resizeDescriptor.m_Parameters.m_Method       = ResizeMethod::Bilinear;
    resizeDescriptor.m_Parameters.m_DataLayout   = descriptor.m_Parameters.m_DataLayout;
    resizeDescriptor.m_Parameters.m_TargetWidth  = descriptor.m_Parameters.m_TargetWidth;
    resizeDescriptor.m_Parameters.m_TargetHeight = descriptor.m_Parameters.m_TargetHeight;

    return CreateResize(resizeDescriptor, info);
}

std::unique_ptr<IWorkload> NpuWorkloadFactory::CreateFakeQuantization(
    const FakeQuantizationQueueDescriptor& descriptor, const WorkloadInfo& info) const {
    return MakeWorkload<NullWorkload, NullWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NpuWorkloadFactory::CreateL2Normalization(
    const L2NormalizationQueueDescriptor& descriptor, const WorkloadInfo& info) const {
    return MakeWorkload<NpuL2NormalizationFloat16Workload,
                        NpuL2NormalizationFloat32Workload,
                        NpuL2NormalizationUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NpuWorkloadFactory::CreateConcat(
    const ConcatQueueDescriptor& descriptor, const WorkloadInfo& info) const {
    return MakeWorkload<NpuConcatFloat16Workload, NpuConcatFloat32Workload, NpuConcatUint8Workload>(
        descriptor, info);
}

std::unique_ptr<IWorkload> NpuWorkloadFactory::CreateConstant(
    const ConstantQueueDescriptor& descriptor, const WorkloadInfo& info) const {
    return MakeWorkload<NpuConstantFloat16Workload,
                        NpuConstantFloat32Workload,
                        NpuConstantUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> NpuWorkloadFactory::CreateReshape(
    const ReshapeQueueDescriptor& descriptor, const WorkloadInfo& info) const {
    return MakeWorkload<NpuReshapeFloat16Workload,
                        NpuReshapeFloat32Workload,
                        NpuReshapeUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> NpuWorkloadFactory::CreateSpaceToBatchNd(
    const SpaceToBatchNdQueueDescriptor& descriptor, const WorkloadInfo& info) const {
    return MakeWorkload<NpuSpaceToBatchNDFloat16Workload,
                        NpuSpaceToBatchNDFloat32Workload,
                        NpuSpaceToBatchNDUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> NpuWorkloadFactory::CreateSpaceToDepth(
    const armnn::SpaceToDepthQueueDescriptor& descriptor, const armnn::WorkloadInfo& info) const {
    return MakeWorkload<NpuSpaceToDepthFloat16Workload,
                        NpuSpaceToDepthFloat32Workload,
                        NpuSpaceToDepthUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> NpuWorkloadFactory::CreateFloor(const FloorQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const {
    return MakeWorkload<NullWorkload, NullWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NpuWorkloadFactory::CreateLstm(const LstmQueueDescriptor& descriptor,
                                                          const WorkloadInfo& info) const {
    return MakeWorkload<NpuLstmFloat16Workload, NpuLstmFloat32Workload, NpuLstmUint8Workload>(
        descriptor, info);
}

std::unique_ptr<IWorkload> NpuWorkloadFactory::CreateConvertFp16ToFp32(
    const ConvertFp16ToFp32QueueDescriptor& descriptor, const WorkloadInfo& info) const {
    return MakeWorkload<NpuFp16ToFp32Workload, NullWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NpuWorkloadFactory::CreateConvertFp32ToFp16(
    const ConvertFp32ToFp16QueueDescriptor& descriptor, const WorkloadInfo& info) const {
    return MakeWorkload<NullWorkload, NpuFp32ToFp16Workload, NullWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NpuWorkloadFactory::CreateDivision(
    const DivisionQueueDescriptor& descriptor, const WorkloadInfo& info) const {
    return MakeWorkload<NpuDivisionFloat16Workload,
                        NpuDivisionFloat32Workload,
                        NpuDivisionUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NpuWorkloadFactory::CreateSubtraction(
    const SubtractionQueueDescriptor& descriptor, const WorkloadInfo& info) const {
    return MakeWorkload<NpuSubtractionFloat16Workload,
                        NpuSubtractionFloat32Workload,
                        NpuSubtractionUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NpuWorkloadFactory::CreateMaximum(
    const MaximumQueueDescriptor& descriptor, const WorkloadInfo& info) const {
    return MakeWorkload<NpuMaximumFloat16Workload,
                        NpuMaximumFloat32Workload,
                        NpuMaximumUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NpuWorkloadFactory::CreateMean(
    const MeanQueueDescriptor& descriptor, const WorkloadInfo& info) const {
    return MakeWorkload<NpuMeanFloat16Workload, NpuMeanFloat32Workload, NpuMeanUint8Workload>(
        descriptor, info);
}

std::unique_ptr<armnn::IWorkload> NpuWorkloadFactory::CreateMinimum(
    const MinimumQueueDescriptor& descriptor, const WorkloadInfo& info) const {
    return MakeWorkload<NpuMinimumFloat16Workload,
                        NpuMinimumFloat32Workload,
                        NpuMinimumUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> NpuWorkloadFactory::CreatePad(const PadQueueDescriptor& descriptor,
                                                         const WorkloadInfo& info) const {
    return MakeWorkload<NpuPadFloat16Workload, NpuPadFloat32Workload, NpuPadUint8Workload>(
        descriptor, info);
}

std::unique_ptr<IWorkload> NpuWorkloadFactory::CreateEqual(const EqualQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const {
    ComparisonQueueDescriptor comparisonDescriptor;
    comparisonDescriptor.m_Parameters.m_Operation = ComparisonOperation::Equal;

    return CreateComparison(comparisonDescriptor, info);
}

std::unique_ptr<IWorkload> NpuWorkloadFactory::CreateBatchToSpaceNd(
    const BatchToSpaceNdQueueDescriptor& descriptor, const WorkloadInfo& info) const {
    return MakeWorkload<NpuBatchToSpaceNdDFloat16Workload,
                        NpuBatchToSpaceNdDFloat32Workload,
                        NpuBatchToSpaceNdNDUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> NpuWorkloadFactory::CreateStridedSlice(
    const StridedSliceQueueDescriptor& descriptor, const WorkloadInfo& info) const {
    return MakeWorkload<NpuStridedSliceFloat16Workload,
                        NpuStridedSliceFloat32Workload,
                        NpuStridedSliceUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> NpuWorkloadFactory::CreateGreater(
    const GreaterQueueDescriptor& descriptor, const WorkloadInfo& info) const {
    ComparisonQueueDescriptor comparisonDescriptor;
    comparisonDescriptor.m_Parameters.m_Operation = ComparisonOperation::Greater;

    return CreateComparison(comparisonDescriptor, info);
}

std::unique_ptr<IWorkload> NpuWorkloadFactory::CreateDebug(const DebugQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const {
    return MakeWorkload<NullWorkload, NullWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NpuWorkloadFactory::CreateRsqrt(const RsqrtQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const {
    ElementwiseUnaryQueueDescriptor elementwiseUnaryDescriptor;
    elementwiseUnaryDescriptor.m_Parameters.m_Operation = UnaryOperation::Rsqrt;

    return CreateElementwiseUnary(elementwiseUnaryDescriptor, info);
}

std::unique_ptr<IWorkload> NpuWorkloadFactory::CreateGather(const armnn::GatherQueueDescriptor& descriptor,
                                                            const armnn::WorkloadInfo& info) const {
    return MakeWorkload<NpuGatherWorkloadFp16, NpuGatherWorkloadFp32, NpuGatherWorkloadU8>(descriptor, info);
}

std::unique_ptr<IWorkload> NpuWorkloadFactory::CreatePreCompiled(const PreCompiledQueueDescriptor& descriptor,
                                                                 const WorkloadInfo& info) const {
    return MakeWorkload<NullWorkload, NullWorkload, NullWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NpuWorkloadFactory::CreateQuantize(const QuantizeQueueDescriptor& descriptor,
                                                              const WorkloadInfo& info) const {
    return MakeWorkload<NullWorkload, NpuQuantizeFloat32Workload, NullWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> NpuWorkloadFactory::CreateDequantize(const DequantizeQueueDescriptor& descriptor,
                                                                const WorkloadInfo& info) const {
    return MakeWorkload<NullWorkload, NullWorkload, NpuDequantizeUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> NpuWorkloadFactory::CreatePrelu(const PreluQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const {
    return MakeWorkload<NpuPreluFloat16Workload, NpuPreluFloat32Workload, NpuPreluUint8Workload>(
        descriptor, info);
}

std::unique_ptr<IWorkload> NpuWorkloadFactory::CreateTransposeConvolution2d(
    const TransposeConvolution2dQueueDescriptor& descriptor, const WorkloadInfo& info) const {
    return MakeWorkload<NpuTransposeConvolution2dFloat16Workload,
                        NpuTransposeConvolution2dFloat32Workload,
                        NpuTransposeConvolution2dUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> NpuWorkloadFactory::CreateComparison(
    const ComparisonQueueDescriptor& descriptor, const WorkloadInfo& info) const {
    return MakeWorkload<NpuComparisonFloat16Workload,
                        NpuComparisonFloat32Workload,
                        NpuComparisonUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> NpuWorkloadFactory::CreateElementwiseUnary(
    const ElementwiseUnaryQueueDescriptor& descriptor, const WorkloadInfo& info) const {
    return MakeWorkload<NpuElementwiseUnarytFloat16Workload,
                        NpuElementwiseUnarytFloat32Workload,
                        NpuElementwiseUnarytUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> NpuWorkloadFactory::CreateDepthToSpace(
    const DepthToSpaceQueueDescriptor& descriptor, const WorkloadInfo& info) const {
    return MakeWorkload<NpuDepthToSpaceFloat16Workload,
                        NpuDepthToSpaceFloat32Workload,
                        NpuDepthToSpaceUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> NpuWorkloadFactory::CreateInstanceNormalization(
    const InstanceNormalizationQueueDescriptor& descriptor, const WorkloadInfo& info) const {
    return MakeWorkload<NpuInstanceNormFloat16Workload,
                        NpuInstanceNormFloat32Workload,
                        NpuInstanceNormUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> NpuWorkloadFactory::CreateArgMinMax(
    const ArgMinMaxQueueDescriptor& descriptor, const WorkloadInfo& info) const {
    return MakeWorkload<NpuArgMinMaxFloat16Workload,
                        NpuArgMinMaxFloat32Workload,
                        NpuArgMinMaxUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> NpuWorkloadFactory::CreateLogSoftmax(const LogSoftmaxQueueDescriptor& descriptor,
                                                                const WorkloadInfo& info) const
{
    return MakeWorkload<NpuLogSoftmaxFloat16Workload,
                        NpuLogSoftmaxFloat32Workload,
                        NpuLogSoftmaxUint8Workload>(descriptor, info);
}

}  // namespace armnn
