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

#include "NpuLayerSupport.hpp"
#include "NpuBackendId.hpp"

#include <InternalTypes.hpp>
#include <LayerSupportCommon.hpp>
#include <armnn/Descriptors.hpp>
#include <armnn/Types.hpp>

#include <armnn/BackendRegistry.hpp>
#include <backendsCommon/test/WorkloadTestUtils.hpp>

#include <boost/core/ignore_unused.hpp>

#include <algorithm>
#include <array>
#include <vector>

using namespace boost;

namespace armnn {

namespace {

template <typename Float32Func, typename Uint8Func, typename... Params>
bool IsSupportedForDataTypeRef(Optional<std::string&> reasonIfUnsupported,
                               DataType dataType,
                               Float32Func floatFuncPtr,
                               Uint8Func uint8FuncPtr,
                               Params&&... params) {
    return IsSupportedForDataTypeGeneric(reasonIfUnsupported,
                                         dataType,
                                         &FalseFunc<Params...>,
                                         floatFuncPtr,
                                         uint8FuncPtr,
                                         &FalseFunc<Params...>,
                                         &FalseFunc<Params...>,
                                         std::forward<Params>(params)...);
}

inline armnn::Optional<armnn::DataType> GetBiasTypeFromWeightsType(
    armnn::Optional<armnn::DataType> weightsType) {
    if (!weightsType) {
        return weightsType;
    }

    switch (weightsType.value()) {
        case armnn::DataType::Float16:
        case armnn::DataType::Float32:
            return weightsType;
        case armnn::DataType::QAsymmU8:
        case armnn::DataType::QAsymmS8:
            return armnn::DataType::Signed32;
        case armnn::DataType::QSymmS16:
            return armnn::DataType::Signed32;
        default:
            ARMNN_ASSERT_MSG(false, "GetBiasTypeFromWeightsType(): Unsupported data type.");
    }
    return armnn::EmptyOptional();
}

}  // anonymous namespace

namespace {

std::string CreateIncorrectDimensionsErrorMsg(unsigned int expected,
                                              unsigned int actual,
                                              std::string& layerStr,
                                              std::string& tensorName) {
    std::string errorMsg = "Npu " + layerStr + ": Expected " + std::to_string(expected) +
                           " dimensions but got" + " " + std::to_string(actual) +
                           " dimensions instead, for the '" + tensorName + "' tensor.";

    return errorMsg;
}

}  // anonymous namespace

namespace {
template <typename F>
bool CheckSupportRule(F rule, Optional<std::string&> reasonIfUnsupported, const char* reason) {
    bool supported = rule();
    if (!supported && reason) {
        reasonIfUnsupported.value() +=
            std::string(reason) + "\n";  // Append the reason on a new line
    }
    return supported;
}

struct Rule {
    bool operator()() const { return m_Res; }

    bool m_Res = true;
};

template <typename T>
bool AllTypesAreEqualImpl(T t) {
    ignore_unused(t);
    return true;
}

template <typename T, typename... Rest>
bool AllTypesAreEqualImpl(T t1, T t2, Rest... rest) {
    static_assert(std::is_same<T, TensorInfo>::value, "Type T must be a TensorInfo");

    return (t1.GetDataType() == t2.GetDataType()) && AllTypesAreEqualImpl(t2, rest...);
}

struct TypesAreEqual : public Rule {
    template <typename... Ts>
    TypesAreEqual(const Ts&... ts) {
        m_Res = AllTypesAreEqualImpl(ts...);
    }
};

struct QuantizationParametersAreEqual : public Rule {
    QuantizationParametersAreEqual(const TensorInfo& info0, const TensorInfo& info1) {
        m_Res = info0.GetQuantizationScale() == info1.GetQuantizationScale() &&
                info0.GetQuantizationOffset() == info1.GetQuantizationOffset();
    }
};

struct TypeAnyOf : public Rule {
    template <typename Container>
    TypeAnyOf(const TensorInfo& info, const Container& c) {
        m_Res = std::any_of(
            c.begin(), c.end(), [&info](DataType dt) { return dt == info.GetDataType(); });
    }
};

struct BiasAndWeightsTypesMatch : public Rule {
    BiasAndWeightsTypesMatch(const TensorInfo& biases, const TensorInfo& weights) {
        m_Res = biases.GetDataType() == GetBiasTypeFromWeightsType(weights.GetDataType()).value();
    }
};

struct BiasAndWeightsTypesCompatible : public Rule {
    template <typename Container>
    BiasAndWeightsTypesCompatible(const TensorInfo& info, const Container& c) {
        m_Res = std::any_of(c.begin(), c.end(), [&info](DataType dt) {
            return dt == GetBiasTypeFromWeightsType(info.GetDataType()).value();
        });
    }
};

struct ShapesAreSameRank : public Rule {
    ShapesAreSameRank(const TensorInfo& info0, const TensorInfo& info1) {
        m_Res = info0.GetShape().GetNumDimensions() == info1.GetShape().GetNumDimensions();
    }
};

struct ShapesAreSameTotalSize : public Rule {
    ShapesAreSameTotalSize(const TensorInfo& info0, const TensorInfo& info1) {
        m_Res = info0.GetNumElements() == info1.GetNumElements();
    }
};

struct ShapesAreBroadcastCompatible : public Rule {
    unsigned int CalcInputSize(const TensorShape& in, const TensorShape& out, unsigned int idx) {
        unsigned int offset = out.GetNumDimensions() - in.GetNumDimensions();
        unsigned int sizeIn = (idx < offset) ? 1 : in[idx - offset];
        return sizeIn;
    }

    ShapesAreBroadcastCompatible(const TensorInfo& in0,
                                 const TensorInfo& in1,
                                 const TensorInfo& out) {
        const TensorShape& shape0 = in0.GetShape();
        const TensorShape& shape1 = in1.GetShape();
        const TensorShape& outShape = out.GetShape();

        for (unsigned int i = 0; i < outShape.GetNumDimensions() && m_Res; i++) {
            unsigned int sizeOut = outShape[i];
            unsigned int sizeIn0 = CalcInputSize(shape0, outShape, i);
            unsigned int sizeIn1 = CalcInputSize(shape1, outShape, i);

            m_Res &= ((sizeIn0 == sizeOut) || (sizeIn0 == 1)) &&
                     ((sizeIn1 == sizeOut) || (sizeIn1 == 1));
        }
    }
};

struct TensorNumDimensionsAreCorrect : public Rule {
    TensorNumDimensionsAreCorrect(const TensorInfo& info, unsigned int expectedNumDimensions) {
        m_Res = info.GetNumDimensions() == expectedNumDimensions;
    }
};

struct TensorBatchSizeIsSupported : public Rule {
    TensorBatchSizeIsSupported(const TensorInfo& info, unsigned int expectedBatchSize) {
        m_Res = info.GetShape()[0] == expectedBatchSize;
    }
};

struct AxisIsSupported : public Rule {
    AxisIsSupported(int axis, std::vector<int> noSupportedAxises) {
        m_Res = std::find(noSupportedAxises.begin(), noSupportedAxises.end(), axis) ==
                noSupportedAxises.end();
    }
};

struct IsAlphaBetaSupported : public Rule {
    IsAlphaBetaSupported(const ActivationDescriptor& descriptor) {
        if (descriptor.m_Function == ActivationFunction::TanH) {
            m_Res = descriptor.m_A == 1.0f && descriptor.m_B == 1.0f;
        } else {
            m_Res = true;
        }
        if (descriptor.m_Function == ActivationFunction::BoundedReLu) {
            if (descriptor.m_A == 1.0f || descriptor.m_A == 6.0f) {
                m_Res = true;
            } else {
                m_Res = false;
            }
        }
    }
};

struct IsInputDimsSupported : public Rule {
    IsInputDimsSupported(const TensorInfo& input) {
        m_Res = (input.GetNumDimensions() <= 2);
    }
};

struct IsPadDescriptorSupported : public Rule {
    IsPadDescriptorSupported(const PadDescriptor& descriptor) {
        m_Res = false;
        // only support pad on W/H for 4D tensor
        if (descriptor.m_PadList.size() == 4) {
            if (descriptor.m_PadList[0].first == 0 &&
                descriptor.m_PadList[0].second == 0 &&
                descriptor.m_PadList[3].first == 0 &&
                descriptor.m_PadList[3].second == 0) {
                m_Res = true;
            }
        }
    }
};

struct IsUint8UnbiasedConvolution : public Rule {
    IsUint8UnbiasedConvolution(const Convolution2dDescriptor& descriptor, const TensorInfo& info) {
        m_Res = (info.GetDataType() == DataType::QAsymmU8 && !descriptor.m_BiasEnabled);
    }
};

struct IsNormChannelTypeSupported : public Rule {
    IsNormChannelTypeSupported(const NormalizationDescriptor& descriptor) {
        m_Res = (descriptor.m_NormChannelType != NormalizationAlgorithmChannel::Within);
    }
};

struct Conv2dWeightPerChannelSupported : public Rule {
    Conv2dWeightPerChannelSupported(const TensorInfo& weight) {
        m_Res = (weight.GetQuantizationDim().value() == 0);
    }
};

struct DepthwiseConvolutionWeightPerChannelSupported : public Rule {
    DepthwiseConvolutionWeightPerChannelSupported(const TensorInfo& weight) {
        m_Res = (weight.GetQuantizationDim().value() == 0);
    }
};

struct TypeIs : public Rule
{
    TypeIs(const TensorInfo& info, DataType dt)
    {
        m_Res = (dt == info.GetDataType());
    }
};

}  // namespace

bool NpuLayerSupport::IsActivationSupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            const ActivationDescriptor& descriptor,
                                            Optional<std::string&> reasonIfUnsupported) const {
    bool supported = true;

    // Define supported types.
    std::array<DataType, 4> supportedTypes = {
        DataType::Float32, DataType::QAsymmU8, DataType::Float16, DataType::QAsymmS8
        // DataType::QSymmS16
    };

    supported &= CheckSupportRule(IsAlphaBetaSupported(descriptor),
                                  reasonIfUnsupported,
                                  "Npu activation: alpha and beta not equal to 1.0");

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu activation: input type not supported.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu activation: output type not supported.");

    supported &= CheckSupportRule(TypesAreEqual(input, output),
                                  reasonIfUnsupported,
                                  "Npu activation: input and output types mismatched.");

    supported &= CheckSupportRule(ShapesAreSameRank(input, output),
                                  reasonIfUnsupported,
                                  "Npu activation: input and output shapes are of different rank.");

    struct ActivationFunctionSupported : public Rule {
        ActivationFunctionSupported(const ActivationDescriptor& desc, const TensorInfo& tensorInput) {
            switch (desc.m_Function) {
                case ActivationFunction::Abs:
                case ActivationFunction::BoundedReLu:
                case ActivationFunction::LeakyReLu:
                case ActivationFunction::ReLu:
                case ActivationFunction::Sigmoid:
                case ActivationFunction::SoftReLu:
                case ActivationFunction::Sqrt:
                case ActivationFunction::Square:
                case ActivationFunction::TanH: {
                    m_Res = true;
                    break;
                }
                case ActivationFunction::Linear:{
                    std::array<DataType, 1> supportedTypes = {DataType::Float32};
                    m_Res = TypeAnyOf(tensorInput, supportedTypes)();
                }
                break;
                default: {
                    m_Res = false;
                    break;
                }
            }
        }
    };

    // Function is supported
    supported &= CheckSupportRule(ActivationFunctionSupported(descriptor, input),
                                  reasonIfUnsupported,
                                  "Npu activation: function not supported.");

    return supported;
}

bool NpuLayerSupport::IsAdditionSupported(const TensorInfo& input0,
                                          const TensorInfo& input1,
                                          const TensorInfo& output,
                                          Optional<std::string&> reasonIfUnsupported) const {
    bool supported = true;

    std::array<DataType, 4> supportedTypes = {
        DataType::Float32, DataType::QAsymmU8, DataType::Float16, DataType::QAsymmS8,
        // DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(input0, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu addition: input 0 is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(input1, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu addition: input 1 is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu addition: output is not a supported type.");

    supported &= CheckSupportRule(TypesAreEqual(input0, input1),
                                  reasonIfUnsupported,
                                  "Npu addition: input 0 and Input 1 types are mismatched");

    supported &= CheckSupportRule(TypesAreEqual(input0, output),
                                  reasonIfUnsupported,
                                  "Npu addition: input and output types are mismatched");

    supported &= CheckSupportRule(ShapesAreBroadcastCompatible(input0, input1, output),
                                  reasonIfUnsupported,
                                  "Npu addition: shapes are not suitable for implicit broadcast.");

    return supported;
}

bool NpuLayerSupport::IsBatchNormalizationSupported(
    const TensorInfo& input,
    const TensorInfo& output,
    const TensorInfo& mean,
    const TensorInfo& variance,
    const TensorInfo& beta,
    const TensorInfo& gamma,
    const BatchNormalizationDescriptor& descriptor,
    Optional<std::string&> reasonIfUnsupported) const {
    ignore_unused(descriptor);

    std::array<DataType, 3> supportedTypes = {
        DataType::Float32, DataType::QAsymmU8, DataType::Float16
        // DataType::QSymmS16
    };

    bool supported = true;

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu batch normalization: input is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu batch normalization: output is not a supported type.");

    supported &= CheckSupportRule(TypesAreEqual(input, output),
                                  reasonIfUnsupported,
                                  "Npu batch normalization: input and output types are mismatched");

    supported &= CheckSupportRule(TypeAnyOf(mean, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu batch normalization: mean is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(variance, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu batch normalization: variance is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(beta, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu batch normalization: beta is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(gamma, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu batch normalization: gamma is not a supported type.");

    return supported;
}

bool NpuLayerSupport::IsBatchToSpaceNdSupported(const TensorInfo& input,
                                                const TensorInfo& output,
                                                const BatchToSpaceNdDescriptor& descriptor,
                                                Optional<std::string&> reasonIfUnsupported) const {
    ignore_unused(descriptor);
    bool support = true;
    support &= IsSupportedForDataTypeRef(
        reasonIfUnsupported, input.GetDataType(), &TrueFunc<>, &TrueFunc<>);
    support &= IsSupportedForDataTypeRef(
        reasonIfUnsupported, output.GetDataType(), &TrueFunc<>, &TrueFunc<>);
    return support;
}

bool NpuLayerSupport::IsConcatSupported(const std::vector<const TensorInfo*> inputs,
                                        const TensorInfo& output,
                                        const ConcatDescriptor& descriptor,
                                        Optional<std::string&> reasonIfUnsupported) const {
    ignore_unused(descriptor);

    bool supported = true;
    std::array<DataType, 4> supportedTypes = {
        DataType::Float32, DataType::QAsymmU8, DataType::Float16, DataType::QAsymmS8,
        // DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu concatenation: output type not supported");
    for (const TensorInfo* input : inputs) {
        supported &= CheckSupportRule(TypeAnyOf(*input, supportedTypes),
                                      reasonIfUnsupported,
                                      "Npu concatenation: input type not supported");

        supported &= CheckSupportRule(TypesAreEqual(*input, output),
                                      reasonIfUnsupported,
                                      "Npu concatenation: input and output types mismatched.");
    }

    return supported;
}

bool NpuLayerSupport::IsConstantSupported(const TensorInfo& output,
                                          Optional<std::string&> reasonIfUnsupported) const {
    std::array<DataType, 3> supportedTypes = {DataType::Float32,
                                              DataType::Float16,
                                              DataType::QAsymmU8};

    return CheckSupportRule(TypeAnyOf(output, supportedTypes),
                                     reasonIfUnsupported,
                                     "Npu constant: output is not a supported type.");
}

bool NpuLayerSupport::IsConvertFp16ToFp32Supported(
    const TensorInfo& input,
    const TensorInfo& output,
    Optional<std::string&> reasonIfUnsupported) const {
    return (IsSupportedForDataTypeGeneric(reasonIfUnsupported,
                                          input.GetDataType(),
                                          &TrueFunc<>,
                                          &FalseInputFuncF32<>,
                                          &FalseFuncU8<>,
                                          &FalseFuncI32<>,
                                          &FalseFuncU8<>) &&
            IsSupportedForDataTypeGeneric(reasonIfUnsupported,
                                          output.GetDataType(),
                                          &FalseOutputFuncF16<>,
                                          &TrueFunc<>,
                                          &FalseFuncU8<>,
                                          &FalseFuncI32<>,
                                          &FalseFuncU8<>));
}

bool NpuLayerSupport::IsConvertFp32ToFp16Supported(
    const TensorInfo& input,
    const TensorInfo& output,
    Optional<std::string&> reasonIfUnsupported) const {
    return (IsSupportedForDataTypeGeneric(reasonIfUnsupported,
                                          input.GetDataType(),
                                          &FalseInputFuncF16<>,
                                          &TrueFunc<>,
                                          &FalseFuncU8<>,
                                          &FalseFuncI32<>,
                                          &FalseFuncU8<>) &&
            IsSupportedForDataTypeGeneric(reasonIfUnsupported,
                                          output.GetDataType(),
                                          &TrueFunc<>,
                                          &FalseOutputFuncF32<>,
                                          &FalseFuncU8<>,
                                          &FalseFuncI32<>,
                                          &FalseFuncU8<>));
}

bool NpuLayerSupport::IsConvolution2dSupported(const TensorInfo& input,
                                               const TensorInfo& output,
                                               const Convolution2dDescriptor& descriptor,
                                               const TensorInfo& weights,
                                               const Optional<TensorInfo>& biases,
                                               Optional<std::string&> reasonIfUnsupported) const {
    bool supported = true;

    // Define supported types.
    std::array<DataType, 5> supportedTypes = {
        DataType::Float32, DataType::QAsymmU8, DataType::Float16,
        DataType::QAsymmS8, DataType::QSymmS8
        //DataType::QSymmS16
    };

    supported &= !CheckSupportRule(IsUint8UnbiasedConvolution(descriptor, input),
                                   reasonIfUnsupported,
                                   "Npu convolution2d: Uint8UnbiasedConvolution not supported.");

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu convolution2d: input is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu convolution2d: output is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(weights, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu convolution2d: weights is not a supported type.");

    supported &= CheckSupportRule(TypesAreEqual(input, output),
                                  reasonIfUnsupported,
                                  "Npu convolution2d: input and output types mismatched.");

    if (weights.HasPerAxisQuantization()) {
        supported &= CheckSupportRule(Conv2dWeightPerChannelSupported(weights),
                                      reasonIfUnsupported,
                                      "Npu convolution2d: only support per-channel quantize for weight.");
    } else {
        supported &= CheckSupportRule(TypesAreEqual(input, weights),
                                      reasonIfUnsupported,
                                      "Npu convolution2d: input and weights types mismatched.");
    }

    if (biases.has_value()) {
        std::array<DataType, 3> biasesSupportedTypes = {
            DataType::Float32, DataType::Signed32, DataType::Float16};
        supported &= CheckSupportRule(TypeAnyOf(biases.value(), biasesSupportedTypes),
                                      reasonIfUnsupported,
                                      "Npu convolution2d: biases is not a supported type.");
    }
    ignore_unused(descriptor);

    return supported;
}

bool NpuLayerSupport::IsDebugSupported(const TensorInfo& input,
                                       const TensorInfo& output,
                                       Optional<std::string&> reasonIfUnsupported) const {
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(reasonIfUnsupported);
    return false;
}

bool NpuLayerSupport::IsDepthwiseConvolutionSupported(
    const TensorInfo& input,
    const TensorInfo& output,
    const DepthwiseConvolution2dDescriptor& descriptor,
    const TensorInfo& weights,
    const Optional<TensorInfo>& biases,
    Optional<std::string&> reasonIfUnsupported) const {
    bool supported = true;

    // Define supported types.
    std::array<DataType, 5> supportedTypes = {
        DataType::Float32, DataType::QAsymmU8, DataType::Float16,
        DataType::QAsymmS8, DataType::QSymmS8
        };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu DepthwiseConvolution2d: input is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu DepthwiseConvolution2d: output is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(weights, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu DepthwiseConvolution2d: weights is not a supported type.");

    supported &= CheckSupportRule(TypesAreEqual(input, output),
                                  reasonIfUnsupported,
                                  "Npu DepthwiseConvolution2d: input and output types mismatched.");

    if (weights.HasPerAxisQuantization()) {
        supported &= CheckSupportRule(DepthwiseConvolutionWeightPerChannelSupported(weights),
                                      reasonIfUnsupported,
                                      "Npu DepthwiseConvolution2d: only support per-channel quantize for weight.");
    } else {
        supported &= CheckSupportRule(TypesAreEqual(input, weights),
                                      reasonIfUnsupported,
                                      "Npu DepthwiseConvolution2d: input and weights types mismatched.");
    }

    if (biases.has_value()) {
        std::array<DataType, 3> biasesSupportedTypes = {
            DataType::Float32, DataType::Signed32, DataType::Float16};
        supported &=
            CheckSupportRule(TypeAnyOf(biases.value(), biasesSupportedTypes),
                             reasonIfUnsupported,
                             "Npu DepthwiseConvolution2d: biases is not a supported type.");
    }
    ignore_unused(descriptor);
    return supported;
}

bool NpuLayerSupport::IsDequantizeSupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            Optional<std::string&> reasonIfUnsupported) const {
    bool supported = true;

    std::array<DataType, 2> supportedInputTypes = {
        DataType::QAsymmU8, DataType::QAsymmS8, //DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedInputTypes),
                                  reasonIfUnsupported,
                                  "Npu dequantize: input type not supported.");

    std::array<DataType, 1> supportedOutputTypes = {
        DataType::Float32,
    };

    supported &= CheckSupportRule(TypeAnyOf(output, supportedOutputTypes),
                                  reasonIfUnsupported,
                                  "Npu dequantize: output type not supported.");

    supported &= CheckSupportRule(
        ShapesAreSameTotalSize(input, output),
        reasonIfUnsupported,
        "Npu dequantize: input and output shapes have different num total elements.");

    return supported;
}

bool NpuLayerSupport::IsDetectionPostProcessSupported(
    const TensorInfo& boxEncodings,
    const TensorInfo& scores,
    const TensorInfo& anchors,
    const TensorInfo& detectionBoxes,
    const TensorInfo& detectionClasses,
    const TensorInfo& detectionScores,
    const TensorInfo& numDetections,
    const DetectionPostProcessDescriptor& descriptor,
    Optional<std::string&> reasonIfUnsupported) const {
    boost::ignore_unused(
        anchors, detectionBoxes, detectionClasses, detectionScores, numDetections, descriptor);

    bool supported = true;

    std::array<DataType, 2> supportedInputTypes = {DataType::Float32, DataType::QAsymmU8};

    supported &=
        CheckSupportRule(TypeAnyOf(boxEncodings, supportedInputTypes),
                         reasonIfUnsupported,
                         "Npu DetectionPostProcess: input 0 is not a supported type.");

    supported &=
        CheckSupportRule(TypeAnyOf(scores, supportedInputTypes),
                         reasonIfUnsupported,
                         "Npu DetectionPostProcess: input 1 is not a supported type.");

    return supported;
}

bool NpuLayerSupport::IsDilatedDepthwiseConvolutionSupported(
    const TensorInfo& input,
    const TensorInfo& output,
    const DepthwiseConvolution2dDescriptor& descriptor,
    const TensorInfo& weights,
    const Optional<TensorInfo>& biases,
    Optional<std::string&> reasonIfUnsupported) const {
    return false && IsDepthwiseConvolutionSupported(
                        input, output, descriptor, weights, biases, reasonIfUnsupported);
}

bool NpuLayerSupport::IsDivisionSupported(const TensorInfo& input0,
                                          const TensorInfo& input1,
                                          const TensorInfo& output,
                                          Optional<std::string&> reasonIfUnsupported) const {
    bool supported = true;

    std::array<DataType, 3> supportedTypes = {
        DataType::Float32, DataType::QAsymmU8, DataType::Float16
        // DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(input0, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu division: input 0 is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(input1, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu division: input 1 is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu division: output is not a supported type.");

    supported &= CheckSupportRule(TypesAreEqual(input0, input1),
                                  reasonIfUnsupported,
                                  "Npu division: input 0 and Input 1 types are mismatched");

    supported &= CheckSupportRule(TypesAreEqual(input0, output),
                                  reasonIfUnsupported,
                                  "Npu division: input and output types are mismatched");

    supported &= CheckSupportRule(ShapesAreBroadcastCompatible(input0, input1, output),
                                  reasonIfUnsupported,
                                  "Npu division: shapes are not suitable for implicit broadcast.");

    return supported;
}

bool NpuLayerSupport::IsEqualSupported(const TensorInfo& input0,
                                       const TensorInfo& input1,
                                       const TensorInfo& output,
                                       Optional<std::string&> reasonIfUnsupported) const {
    return IsComparisonSupported(input0,
                                 input1,
                                 output,
                                 ComparisonDescriptor(ComparisonOperation::Equal),
                                 reasonIfUnsupported);
}

bool NpuLayerSupport::IsFakeQuantizationSupported(
    const TensorInfo& input,
    const FakeQuantizationDescriptor& descriptor,
    Optional<std::string&> reasonIfUnsupported) const {
    ignore_unused(descriptor);
    return false && IsSupportedForDataTypeRef(
                        reasonIfUnsupported, input.GetDataType(), &TrueFunc<>, &FalseFuncU8<>);
}

bool NpuLayerSupport::IsFloorSupported(const TensorInfo& input,
                                       const TensorInfo& output,
                                       Optional<std::string&> reasonIfUnsupported) const {
    ignore_unused(output);
    bool supported = false;

    std::array<DataType, 2> supportedTypes = {DataType::Float32, DataType::QSymmS16};

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu Floor: input type not supported.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu Floor: output type not supported.");

    return supported;
}

bool NpuLayerSupport::IsFullyConnectedSupported(const TensorInfo& input,
                                                const TensorInfo& output,
                                                const TensorInfo& weights,
                                                const TensorInfo& biases,
                                                const FullyConnectedDescriptor& descriptor,
                                                Optional<std::string&> reasonIfUnsupported) const {
    bool supported = true;

    // Define supported types.
    std::array<DataType, 4> supportedTypes = {
        DataType::Float32, DataType::QAsymmU8, DataType::Float16, DataType::QAsymmS8,
        // DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu Fully Connected: input type not supported.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu Fully Connected: output type not supported.");

    supported &= CheckSupportRule(TypesAreEqual(input, output),
                                  reasonIfUnsupported,
                                  "Npu Fully Connected: input and output types mismatched.");

    supported &= CheckSupportRule(TypeAnyOf(weights, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu Fully Connected: weights type not supported.");

    supported &= CheckSupportRule(TypesAreEqual(input, weights),
                                  reasonIfUnsupported,
                                  "Npu Fully Connected: input and weight types mismatched.");

    if (descriptor.m_BiasEnabled) {
        // Defined supported types for bias
        std::array<DataType, 3> supportedBiasTypes = {
            DataType::Float32, DataType::Signed32, DataType::Float16};

        supported &= CheckSupportRule(TypeAnyOf(biases, supportedBiasTypes),
                                      reasonIfUnsupported,
                                      "Npu Fully Connected: bias type not supported.");

        supported &= CheckSupportRule(BiasAndWeightsTypesMatch(biases, weights),
                                      reasonIfUnsupported,
                                      "Npu Fully Connected: bias and weight types mismatch.");

        supported &= CheckSupportRule(
            BiasAndWeightsTypesCompatible(weights, supportedBiasTypes),
            reasonIfUnsupported,
            "Npu Fully Connected: bias type inferred from weights is incompatible.");
    }

    return supported;
}

bool NpuLayerSupport::IsGatherSupported(const armnn::TensorInfo& input0,
                                        const armnn::TensorInfo& input1,
                                        const armnn::TensorInfo& output,
                                        armnn::Optional<std::string&> reasonIfUnsupported) const {
    ignore_unused(input1);
    ignore_unused(output);
    return IsSupportedForDataTypeRef(
        reasonIfUnsupported, input0.GetDataType(), &TrueFunc<>, &TrueFunc<>);
}

bool NpuLayerSupport::IsGreaterSupported(const TensorInfo& input0,
                                         const TensorInfo& input1,
                                         const TensorInfo& output,
                                         Optional<std::string&> reasonIfUnsupported) const {
    return IsComparisonSupported(input0,
                                 input1,
                                 output,
                                 ComparisonDescriptor(ComparisonOperation::Greater),
                                 reasonIfUnsupported);
}

bool NpuLayerSupport::IsInputSupported(const TensorInfo& input,
                                       Optional<std::string&> reasonIfUnsupported) const {
    ignore_unused(input);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool NpuLayerSupport::IsL2NormalizationSupported(const TensorInfo& input,
                                                 const TensorInfo& output,
                                                 const L2NormalizationDescriptor& descriptor,
                                                 Optional<std::string&> reasonIfUnsupported) const {
    ignore_unused(descriptor);
    // Define supported types
    std::array<DataType, 3> supportedTypes = {
        DataType::Float32, DataType::QAsymmU8, DataType::Float16};

    bool supported = true;

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu L2normalization: input type not supported.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu L2normalization: output type not supported.");

    supported &= CheckSupportRule(TypesAreEqual(input, output),
                                  reasonIfUnsupported,
                                  "Npu L2normalization: input and output types mismatched.");

    supported &= CheckSupportRule(ShapesAreSameTotalSize(input, output),
                                  reasonIfUnsupported,
                                  "Npu L2normalization: input and output shapes have different "
                                  "num total elements.");

    return supported;
}
bool NpuLayerSupport::IsLstmSupported(const TensorInfo& input,
                         const TensorInfo& outputStateIn,
                         const TensorInfo& cellStateIn,
                         const TensorInfo& scratchBuffer,
                         const TensorInfo& outputStateOut,
                         const TensorInfo& cellStateOut,
                         const TensorInfo& output,
                         const LstmDescriptor& descriptor,
                         const LstmInputParamsInfo& paramsInfo,
                         Optional<std::string&> reasonIfUnsupported) const {
    bool supported = true;
    // TODO: {Sven} check data type matched
    ignore_unused(input);
    ignore_unused(outputStateIn);
    ignore_unused(cellStateIn);
    ignore_unused(scratchBuffer);
    ignore_unused(outputStateOut);
    ignore_unused(cellStateOut);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(paramsInfo);
    ignore_unused(reasonIfUnsupported);
    return supported;
    }
// bool NpuLayerSupport::IsLstmSupported(const TensorInfo& input,
//                                       const TensorInfo& outputStateIn,
//                                       const TensorInfo& cellStateIn,
//                                       const TensorInfo& scratchBuffer,
//                                       const TensorInfo& outputStateOut,
//                                       const TensorInfo& cellStateOut,
//                                       const TensorInfo& output,
//                                       const LstmDescriptor& descriptor,
//                                       const TensorInfo& inputToForgetWeights,
//                                       const TensorInfo& inputToCellWeights,
//                                       const TensorInfo& inputToOutputWeights,
//                                       const TensorInfo& recurrentToForgetWeights,
//                                       const TensorInfo& recurrentToCellWeights,
//                                       const TensorInfo& recurrentToOutputWeights,
//                                       const TensorInfo& forgetGateBias,
//                                       const TensorInfo& cellBias,
//                                       const TensorInfo& outputGateBias,
//                                       const TensorInfo* inputToInputWeights,
//                                       const TensorInfo* recurrentToInputWeights,
//                                       const TensorInfo* cellToInputWeights,
//                                       const TensorInfo* inputGateBias,
//                                       const TensorInfo* projectionWeights,
//                                       const TensorInfo* projectionBias,
//                                       const TensorInfo* cellToForgetWeights,
//                                       const TensorInfo* cellToOutputWeights,
//                                       Optional<std::string&> reasonIfUnsupported) const {
//     ignore_unused(descriptor);
//     ignore_unused(inputToForgetWeights);
//     ignore_unused(inputToCellWeights);
//     ignore_unused(inputToOutputWeights);
//     ignore_unused(recurrentToForgetWeights);
//     ignore_unused(recurrentToCellWeights);
//     ignore_unused(recurrentToOutputWeights);
//     ignore_unused(forgetGateBias);
//     ignore_unused(cellBias);
//     ignore_unused(outputGateBias);
//     ignore_unused(inputToInputWeights);
//     ignore_unused(recurrentToInputWeights);
//     ignore_unused(cellToInputWeights);
//     ignore_unused(inputGateBias);
//     ignore_unused(projectionWeights);
//     ignore_unused(projectionBias);
//     ignore_unused(cellToForgetWeights);
//     ignore_unused(cellToOutputWeights);

//     bool supported = true;

//     std::array<DataType, 2> supportedTypes = {
//         DataType::Float32, DataType::QAsymmU8,
//         // DataType::QSymmS16
//     };

//     supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes),
//                                   reasonIfUnsupported,
//                                   "Npu Lstm: input is not a supported type.");

//     supported &= CheckSupportRule(TypesAreEqual(input, outputStateIn),
//                                   reasonIfUnsupported,
//                                   "Npu Lstm: input and outputStateIn types are mismatched");

//     supported &= CheckSupportRule(TypesAreEqual(input, cellStateIn),
//                                   reasonIfUnsupported,
//                                   "Npu Lstm: input and cellStateIn types are mismatched");

//     supported &= CheckSupportRule(TypesAreEqual(input, scratchBuffer),
//                                   reasonIfUnsupported,
//                                   "Npu Lstm: input and scratchBuffer types are mismatched");

//     supported &= CheckSupportRule(TypesAreEqual(input, outputStateOut),
//                                   reasonIfUnsupported,
//                                   "Npu Lstm: input and outputStateOut types are mismatched");

//     supported &= CheckSupportRule(TypesAreEqual(input, cellStateOut),
//                                   reasonIfUnsupported,
//                                   "Npu Lstm: input and cellStateOut types are mismatched");

//     supported &= CheckSupportRule(TypesAreEqual(input, output),
//                                   reasonIfUnsupported,
//                                   "Npu Lstm: input and output types are mismatched");

//     supported &= !descriptor.m_PeepholeEnabled;

//     return supported;
// }

bool NpuLayerSupport::IsMaximumSupported(const TensorInfo& input0,
                                         const TensorInfo& input1,
                                         const TensorInfo& output,
                                         Optional<std::string&> reasonIfUnsupported) const {
    bool supported = true;

    std::array<DataType, 3> supportedTypes = {
        DataType::Float32, DataType::QAsymmU8, DataType::Float16
        // DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(input0, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu maximum: input 0 is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(input1, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu maximum: input 1 is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu maximum: output is not a supported type.");

    supported &= CheckSupportRule(TypesAreEqual(input0, input1),
                                  reasonIfUnsupported,
                                  "Npu maximum: input 0 and Input 1 types are mismatched");

    supported &= CheckSupportRule(TypesAreEqual(input0, output),
                                  reasonIfUnsupported,
                                  "Npu maximum: input and output types are mismatched");

    supported &= CheckSupportRule(ShapesAreBroadcastCompatible(input0, input1, output),
                                  reasonIfUnsupported,
                                  "Npu maximum: shapes are not suitable for implicit broadcast.");

    return supported;
}

bool NpuLayerSupport::IsMeanSupported(const TensorInfo& input,
                                      const TensorInfo& output,
                                      const MeanDescriptor& descriptor,
                                      Optional<std::string&> reasonIfUnsupported) const {
    bool supported = true;
    std::string meanLayerStr = "Mean";
    std::string outputTensorStr = "output";

    std::array<DataType, 4> supportedTypes = {
        DataType::Float32, DataType::QAsymmU8, DataType::Float16, DataType::QAsymmS8,
        // DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu Mean: input type not supported.");

    supported &= CheckSupportRule(TypesAreEqual(input, output),
                                  reasonIfUnsupported,
                                  "Npu Mean: input and output types are mismatched");

    if (descriptor.m_KeepDims) {
        supported &= CheckSupportRule(
            TensorNumDimensionsAreCorrect(output, input.GetNumDimensions()),
            reasonIfUnsupported,
            CreateIncorrectDimensionsErrorMsg(
                input.GetNumDimensions(), output.GetNumDimensions(), meanLayerStr, outputTensorStr)
                .data());
    } else if (descriptor.m_Axis.empty()) {
        supported &=
            CheckSupportRule(TensorNumDimensionsAreCorrect(output, 1),
                             reasonIfUnsupported,
                             CreateIncorrectDimensionsErrorMsg(
                                 1, output.GetNumDimensions(), meanLayerStr, outputTensorStr)
                                 .data());
    } else {
        auto outputDim =
            input.GetNumDimensions() - armnn::numeric_cast<unsigned int>(descriptor.m_Axis.size());

        if (outputDim > 0) {
            supported &= CheckSupportRule(
                TensorNumDimensionsAreCorrect(output, outputDim),
                reasonIfUnsupported,
                CreateIncorrectDimensionsErrorMsg(
                    outputDim, output.GetNumDimensions(), meanLayerStr, outputTensorStr)
                    .data());
        } else {
            supported &=
                CheckSupportRule(TensorNumDimensionsAreCorrect(output, 1),
                                 reasonIfUnsupported,
                                 CreateIncorrectDimensionsErrorMsg(
                                     1, output.GetNumDimensions(), meanLayerStr, outputTensorStr)
                                     .data());
        }
    }

    return supported;
}

bool NpuLayerSupport::IsMergerSupported(const std::vector<const TensorInfo*> inputs,
                                        const TensorInfo& output,
                                        const MergerDescriptor& descriptor,
                                        Optional<std::string&> reasonIfUnsupported) const {
    return false && IsConcatSupported(inputs, output, descriptor, reasonIfUnsupported);
}

bool NpuLayerSupport::IsMemCopySupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         Optional<std::string&> reasonIfUnsupported) const {
    ignore_unused(output);
    return IsSupportedForDataTypeGeneric(reasonIfUnsupported,
                                                  input.GetDataType(),
                                                  &TrueFunc<>,
                                                  &TrueFunc<>,
                                                  &TrueFunc<>,
                                                  &FalseFuncI32<>,
                                                  &TrueFunc<>);
}

bool NpuLayerSupport::IsMinimumSupported(const TensorInfo& input0,
                                         const TensorInfo& input1,
                                         const TensorInfo& output,
                                         Optional<std::string&> reasonIfUnsupported) const {
    bool supported = true;

    std::array<DataType, 3> supportedTypes = {
        DataType::Float32, DataType::QAsymmU8, DataType::Float16
        // DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(input0, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu minimum: input 0 is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(input1, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu minimum: input 1 is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu minimum: output is not a supported type.");

    supported &= CheckSupportRule(TypesAreEqual(input0, input1),
                                  reasonIfUnsupported,
                                  "Npu minimum: input 0 and Input 1 types are mismatched");

    supported &= CheckSupportRule(TypesAreEqual(input0, output),
                                  reasonIfUnsupported,
                                  "Npu minimum: input and output types are mismatched");

    supported &= CheckSupportRule(ShapesAreBroadcastCompatible(input0, input1, output),
                                  reasonIfUnsupported,
                                  "Npu minimum: shapes are not suitable for implicit broadcast.");

    return supported;
}

bool NpuLayerSupport::IsMultiplicationSupported(const TensorInfo& input0,
                                                const TensorInfo& input1,
                                                const TensorInfo& output,
                                                Optional<std::string&> reasonIfUnsupported) const {
    bool supported = true;

    std::array<DataType, 3> supportedTypes = {
        DataType::Float32, DataType::QAsymmU8, DataType::Float16
        // DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(input0, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu multiplication: input 0 is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(input1, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu multiplication: input 1 is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu multiplication: output is not a supported type.");

    supported &= CheckSupportRule(TypesAreEqual(input0, input1),
                                  reasonIfUnsupported,
                                  "Npu multiplication: input 0 and Input 1 types are mismatched");

    supported &= CheckSupportRule(TypesAreEqual(input0, output),
                                  reasonIfUnsupported,
                                  "Npu multiplication: input and output types are mismatched");

    supported &=
        CheckSupportRule(ShapesAreBroadcastCompatible(input0, input1, output),
                         reasonIfUnsupported,
                         "Npu multiplication: shapes are not suitable for implicit broadcast.");

    supported &= input0.GetQuantizationOffset() >= 0;
    supported &= input1.GetQuantizationOffset() >= 0;
    supported &= output.GetQuantizationOffset() >= 0;

    return supported;
}

bool NpuLayerSupport::IsNormalizationSupported(const TensorInfo& input,
                                               const TensorInfo& output,
                                               const NormalizationDescriptor& descriptor,
                                               Optional<std::string&> reasonIfUnsupported) const {
    // Define supported types
    std::array<DataType, 3> supportedTypes = {
        DataType::Float32, DataType::QAsymmU8, DataType::Float16};

    bool supported = true;

    supported &= CheckSupportRule(IsNormChannelTypeSupported(descriptor),
                                  reasonIfUnsupported,
                                  "Npu normalization: channel type unsupported.");

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu normalization: input type not supported.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu normalization: output type not supported.");

    supported &= CheckSupportRule(ShapesAreSameTotalSize(input, output),
                                  reasonIfUnsupported,
                                  "Npu normalization: input and output shapes have different "
                                  "num total elements.");

    return supported;
}

bool NpuLayerSupport::IsOutputSupported(const TensorInfo& output,
                                        Optional<std::string&> reasonIfUnsupported) const {
    ignore_unused(output);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool NpuLayerSupport::IsPadSupported(const TensorInfo& input,
                                     const TensorInfo& output,
                                     const PadDescriptor& descriptor,
                                     Optional<std::string&> reasonIfUnsupported) const {
    bool supported = true;
    ignore_unused(output);

    supported &= CheckSupportRule(
        IsPadDescriptorSupported(descriptor), reasonIfUnsupported, "Npu pad: input dimension not support.");
    supported &= IsSupportedForDataTypeRef(
        reasonIfUnsupported, input.GetDataType(), &TrueFunc<>, &TrueFunc<>);
    supported &= (descriptor.m_PadValue - 0.0f) < 1e-5;
    return supported;
}

bool NpuLayerSupport::IsPermuteSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         const PermuteDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported) const {
    ignore_unused(output);
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(
        reasonIfUnsupported, input.GetDataType(), &TrueFunc<>, &TrueFunc<>);
}

bool NpuLayerSupport::IsPreCompiledSupported(const TensorInfo& input,
                                             const PreCompiledDescriptor& descriptor,
                                             Optional<std::string&> reasonIfUnsupported) const {
    ignore_unused(input);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return false;
}

bool NpuLayerSupport::IsPooling2dSupported(const TensorInfo& input,
                                           const TensorInfo& output,
                                           const Pooling2dDescriptor& descriptor,
                                           Optional<std::string&> reasonIfUnsupported) const {
    ignore_unused(descriptor);
    bool supported = true;

    // Define supported output and inputs types.
    std::array<DataType, 4> supportedTypes = {
        DataType::Float32, DataType::QAsymmU8, DataType::Float16, DataType::QAsymmS8,
        // DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu poolind2d: input is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu poolind2d: output is not a supported type.");

    supported &= CheckSupportRule(TypesAreEqual(input, output),
                                  reasonIfUnsupported,
                                  "Npu poolind2d: input and output types are mismatched.");

    return supported;
}

bool NpuLayerSupport::IsQuantizeSupported(const TensorInfo& input,
                                          const TensorInfo& output,
                                          Optional<std::string&> reasonIfUnsupported) const {
    bool supported = true;

    // Define supported output types.
    std::array<DataType, 3> supportedInputTypes = {
        DataType::Float32, DataType::QAsymmU8, DataType::QAsymmS8,
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedInputTypes),
                                  reasonIfUnsupported,
                                  "Npu quantize: input type not supported.");

    // Define supported output types.
    std::array<DataType, 2> supportedOutputTypes = {
        DataType::QAsymmU8, DataType::QAsymmS8,
        //DataType::QSymmS16
    };
    supported &= CheckSupportRule(TypeAnyOf(output, supportedOutputTypes),
                                  reasonIfUnsupported,
                                  "Npu quantize: output type not supported.");

    supported &= CheckSupportRule(
        ShapesAreSameTotalSize(input, output),
        reasonIfUnsupported,
        "Npu quantize: input and output shapes have different num total elements.");

    return supported;
}

bool NpuLayerSupport::IsReshapeSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         const ReshapeDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported) const {
    ignore_unused(output);
    ignore_unused(descriptor);
    // Define supported output types.
    std::array<DataType, 3> supportedOutputTypes = {
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmU8,
        // DataType::QSymmS16
    };

    return CheckSupportRule(TypeAnyOf(input, supportedOutputTypes),
                            reasonIfUnsupported,
                            "Npu reshape: input type not supported.");
}

bool NpuLayerSupport::IsResizeSupported(const TensorInfo& input,
                                        const TensorInfo& output,
                                        const ResizeDescriptor& descriptor,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(descriptor);
    bool supported = true;
    std::array<DataType,3> supportedTypes =
    {
        DataType::Float32,
        DataType::QAsymmU8,
        DataType::Float16
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                  "Reference Resize: input type not supported");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Reference Resize: output type not supported");

    supported &= CheckSupportRule(TypesAreEqual(input, output), reasonIfUnsupported,
                                  "Reference Resize: input and output types not matching");

    return supported;
}

bool NpuLayerSupport::IsResizeBilinearSupported(const TensorInfo& input,
                                                const TensorInfo& output,
                                                Optional<std::string&> reasonIfUnsupported) const {
    bool supported = true;
    std::array<DataType, 3> supportedTypes = {
        DataType::Float32, DataType::QAsymmU8, DataType::Float16};

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu ResizeBilinear: input type not supported");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu ResizeBilinear: output type not supported");

    supported &= CheckSupportRule(TypesAreEqual(input, output),
                                  reasonIfUnsupported,
                                  "Npu ResizeBilinear: input and output types not matching");

    return supported;
}

bool NpuLayerSupport::IsRsqrtSupported(const TensorInfo& input,
                                       const TensorInfo& output,
                                       Optional<std::string&> reasonIfUnsupported) const {
    bool supported = true;
    std::array<DataType, 3> supportedTypes = {
        DataType::Float32, DataType::QAsymmU8, DataType::Float16};

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu rsqrt: input type not supported");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu rsqrt: output type not supported");

    supported &= CheckSupportRule(TypesAreEqual(input, output),
                                  reasonIfUnsupported,
                                  "Npu rsqrt: input and output types not matching");

    supported &= CheckSupportRule(
        ShapesAreSameTotalSize(input, output),
        reasonIfUnsupported,
        "Npu Rsqrt: input and output shapes have different number of total elements");

    return supported;
}

bool NpuLayerSupport::IsSoftmaxSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         const SoftmaxDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported) const {
    ignore_unused(output);
    bool supported = true;
    std::array<DataType, 4> supportedTypes = {
        DataType::Float32, DataType::QAsymmU8, DataType::Float16, DataType::QAsymmS8,
        // DataType::QSymmS16
    };

    std::vector<int> noSupportedAxises = {0, -4};

    supported &= CheckSupportRule(AxisIsSupported(descriptor.m_Axis, noSupportedAxises),
                                  reasonIfUnsupported,
                                  "NPU sotfmax: axis not supported");

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu softmax: output type not supported");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu softmax: input type not supported");

    supported &= CheckSupportRule(TypesAreEqual(input, output),
                                  reasonIfUnsupported,
                                  "Npu softmax: input type not supported");

    return supported;
}

bool NpuLayerSupport::IsSpaceToBatchNdSupported(const TensorInfo& input,
                                                const TensorInfo& output,
                                                const SpaceToBatchNdDescriptor& descriptor,
                                                Optional<std::string&> reasonIfUnsupported) const {
    ignore_unused(descriptor);
    bool supported = true;
    std::array<DataType, 3> supportedTypes = {
        DataType::Float32, DataType::QAsymmU8, DataType::Float16};

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu SpaceToBatchNd: input type not supported");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu SpaceToBatchNd: output type not supported");

    supported &= CheckSupportRule(TypesAreEqual(input, output),
                                  reasonIfUnsupported,
                                  "Npu SpaceToBatchNd: input and output types are mismatched");

    supported &= CheckSupportRule(TensorBatchSizeIsSupported(input, 1),
                                  reasonIfUnsupported,
                                  "Npu SpaceToBatchNd: input batch size > 1 not supported");

    return supported;
}

bool NpuLayerSupport::IsSpaceToDepthSupported(const TensorInfo& input,
                                              const TensorInfo& output,
                                              const SpaceToDepthDescriptor& descriptor,
                                              Optional<std::string&> reasonIfUnsupported) const {
    ignore_unused(descriptor);
    bool supported = true;

    std::array<DataType, 3> supportedTypes = {
        DataType::Float32, DataType::QAsymmU8, DataType::Float16
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu SpaceToDepth: input type not supported");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu SpaceToDepth: output type not supported");

    supported &= CheckSupportRule(TypesAreEqual(input, output),
                                  reasonIfUnsupported,
                                  "Npu SpaceToDepth: input and output types are mismatched");

    return supported;
}

bool NpuLayerSupport::IsSplitterSupported(
    const TensorInfo& input,
    const std::vector<std::reference_wrapper<TensorInfo>>& outputs,
    const ViewsDescriptor& descriptor,
    Optional<std::string&> reasonIfUnsupported) const {
    ignore_unused(outputs);
    ignore_unused(descriptor);
    bool supported = true;
    std::array<DataType, 3> supportedTypes = {
        DataType::Float32, DataType::QAsymmU8, DataType::Float16};

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu splitter: input type not supported");

    return supported;
}

bool NpuLayerSupport::IsStridedSliceSupported(const TensorInfo& input,
                                              const TensorInfo& output,
                                              const StridedSliceDescriptor& descriptor,
                                              Optional<std::string&> reasonIfUnsupported) const {
    bool supported = true;
    ignore_unused(output);
    ignore_unused(descriptor);
    supported &= IsSupportedForDataTypeRef(
        reasonIfUnsupported, input.GetDataType(), &TrueFunc<>, &TrueFunc<>);
    // TODO: Support the two params.
    supported &= descriptor.m_EllipsisMask == 0;
    supported &= descriptor.m_NewAxisMask == 0;
    return supported;
}

bool NpuLayerSupport::IsSubtractionSupported(const TensorInfo& input0,
                                             const TensorInfo& input1,
                                             const TensorInfo& output,
                                             Optional<std::string&> reasonIfUnsupported) const {
    bool supported = true;

    std::array<DataType, 3> supportedTypes = {
        DataType::Float32, DataType::QAsymmU8, DataType::Float16
        // DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(input0, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu subtraction: input 0 is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(input1, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu subtraction: input 1 is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes),
                                  reasonIfUnsupported,
                                  "Npu subtraction: output is not a supported type.");

    supported &= CheckSupportRule(TypesAreEqual(input0, input1),
                                  reasonIfUnsupported,
                                  "Npu subtraction: input 0 and Input 1 types are mismatched");

    supported &= CheckSupportRule(TypesAreEqual(input0, output),
                                  reasonIfUnsupported,
                                  "Npu subtraction: input and output types are mismatched");

    supported &=
        CheckSupportRule(ShapesAreBroadcastCompatible(input0, input1, output),
                         reasonIfUnsupported,
                         "Npu subtraction: shapes are not suitable for implicit broadcast.");

    return supported;
}

bool NpuLayerSupport::IsPreluSupported(const TensorInfo& input,
                                       const TensorInfo& alpha,
                                       const TensorInfo& output,
                                       Optional<std::string&> reasonIfUnsupported) const {
    bool supported = true;

    std::array<DataType, 3> supportedTypes{
        DataType::Float32, DataType::QAsymmU8, DataType::Float16
        // DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes),
                                  reasonIfUnsupported,
                                  "PReLU: input is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(alpha, supportedTypes),
                                  reasonIfUnsupported,
                                  "PReLU: alpha is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes),
                                  reasonIfUnsupported,
                                  "PReLU: output is not a supported type.");

    supported &= CheckSupportRule(TypesAreEqual(input, alpha, output),
                                  reasonIfUnsupported,
                                  "PReLU: input, alpha and output types are mismatched");

    supported &= CheckSupportRule(ShapesAreBroadcastCompatible(input, alpha, output),
                                  reasonIfUnsupported,
                                  "PReLU: shapes are not suitable for implicit broadcast");

    return supported;
}

bool NpuLayerSupport::IsTransposeConvolution2dSupported(const TensorInfo& input,
                                  const TensorInfo& output,
                                  const TransposeConvolution2dDescriptor& descriptor,
                                  const TensorInfo& weights,
                                  const Optional<TensorInfo>& biases,
                                  Optional<std::string&> reasonIfUnsupported) const {
    bool supported = true;

    // Define supported types.
    std::array<DataType, 3> supportedTypes = {
        DataType::Float32, DataType::QAsymmU8, DataType::Float16
        // DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes),
              reasonIfUnsupported,
              "Npu transpose_convolution2d: input is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes),
              reasonIfUnsupported,
              "Npu transpose_convolution2d: output is not a supported type.");

    supported &= CheckSupportRule(TypeAnyOf(weights, supportedTypes),
              reasonIfUnsupported,
              "Npu transpose_convolution2d: weights is not a supported type.");

    supported &= CheckSupportRule(TypesAreEqual(input, output),
              reasonIfUnsupported,
              "Npu transpose_convolution2d: input and output types mismatched.");

    supported &= CheckSupportRule(TypesAreEqual(input, weights),
              reasonIfUnsupported,
              "Npu transpose_convolution2d: input and weights types mismatched.");

    if (biases.has_value()) {
        std::array<DataType, 3> biasesSupportedTypes = {DataType::Float32, DataType::Signed32};
        supported &= CheckSupportRule(TypeAnyOf(biases.value(), biasesSupportedTypes),
                reasonIfUnsupported,
                "Npu transpose_convolution2d: biases is not a supported type.");
    }
    ignore_unused(descriptor);

    return supported;
}

bool NpuLayerSupport::IsComparisonSupported(const TensorInfo& input0,
                                            const TensorInfo& input1,
                                            const TensorInfo& output,
                                            const ComparisonDescriptor& descriptor,
                                            Optional<std::string&> reasonIfUnsupported) const
{
    boost::ignore_unused(descriptor);

    std::array<DataType, 3> supportedInputTypes =
    {
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmU8,
    };

    bool supported = true;
    supported &= CheckSupportRule(TypeAnyOf(input0, supportedInputTypes), reasonIfUnsupported,
                                  "Npu comparison: input 0 is not a supported type");

    supported &= CheckSupportRule(TypesAreEqual(input0, input1), reasonIfUnsupported,
                                  "Npu comparison: input 0 and Input 1 types are mismatched");

    supported &= CheckSupportRule(TypeIs(output, DataType::Boolean), reasonIfUnsupported,
                                  "Npu comparison: output is not of type Boolean");

    return supported;
}

bool NpuLayerSupport::IsDepthToSpaceSupported(const TensorInfo& input,
                                              const TensorInfo& output,
                                              const DepthToSpaceDescriptor& descriptor,
                                              Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(descriptor);
    bool supported = true;

    std::array<DataType,3> supportedTypes =
    {
        DataType::Float32,
        DataType::Float16,
        DataType::QAsymmU8,
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
        "NPU DepthToSpace: input type not supported");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
        "NPU DepthToSpace: output type not supported");

    supported &= CheckSupportRule(TypesAreEqual(input, output), reasonIfUnsupported,
        "NPU DepthToSpace: input and output types are mismatched");

    return supported;
}

bool NpuLayerSupport::IsInstanceNormalizationSupported(const TensorInfo& input,
                                                       const TensorInfo& output,
                                                       const InstanceNormalizationDescriptor& descriptor,
                                                       Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(descriptor);
    // Define supported types
    std::array<DataType, 2> supportedTypes =
        {
            DataType::Float32,
            DataType::Float16,
        };

    bool supported = true;

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                  "NPU Instance Normalization: input type not supported.");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "NPU Instance Normalization: output type not supported.");

    supported &= CheckSupportRule(TypesAreEqual(input, output), reasonIfUnsupported,
                                  "NPU Instance Normalization: input and output types mismatched.");

    supported &= CheckSupportRule(ShapesAreSameTotalSize(input, output), reasonIfUnsupported,
                                  "NPU Instance Normalization: input and output shapes have different "
                                  "num total elements.");

    return supported;
}

bool NpuLayerSupport::IsArgMinMaxSupported(const armnn::TensorInfo &input, const armnn::TensorInfo &output,
                                           const armnn::ArgMinMaxDescriptor &descriptor,
                                           armnn::Optional<std::string &> reasonIfUnsupported) const
{
    ignore_unused(descriptor);

    std::array<DataType, 4> supportedTypes =
    {
        DataType::Float32,
        DataType::QAsymmU8,
        DataType::Float16,
        DataType::Signed32
    };

    bool supported = true;

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                  "NPU ArgMinMax: input is not a supported type.");
    supported &= CheckSupportRule(TypeIs(output, DataType::Signed32), reasonIfUnsupported,
                                  "NPU ArgMinMax: output type not supported");

    return supported;
}

bool NpuLayerSupport::IsLogSoftmaxSupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            const LogSoftmaxDescriptor& descriptor,
                                            Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(descriptor);

    std::array<DataType, 3> supportedTypes =
    {
            DataType::Float32,
            DataType::Float16,
            DataType::QAsymmU8
    };

    bool supported = true;
    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                  "Npu LogSoftmax: input type not supported");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Npu LogSoftmax: output type not supported");

    supported &= CheckSupportRule(TypesAreEqual(input, output), reasonIfUnsupported,
                                  "Npu LogSoftmax: input and output types do not match");

    return supported;
}

bool NpuLayerSupport::IsSliceSupported(const TensorInfo& input,
                                       const TensorInfo& output,
                                       const SliceDescriptor& descriptor,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    boost::ignore_unused(descriptor);
    bool supported = true;

    std::array<DataType, 3> supportedTypes =
    {
        DataType::Float32,
        DataType::QAsymmU8,
        DataType::QSymmS16
    };

    supported &= CheckSupportRule(TypeAnyOf(input, supportedTypes), reasonIfUnsupported,
                                  "Npu Slice: input type not supported");

    supported &= CheckSupportRule(TypeAnyOf(output, supportedTypes), reasonIfUnsupported,
                                  "Npu Slice: output type not supported");

    supported &= CheckSupportRule(TypesAreEqual(input, output), reasonIfUnsupported,
                                  "Npu Slice: input and output types are mismatched");

    return supported;
}


}  // namespace armnn
