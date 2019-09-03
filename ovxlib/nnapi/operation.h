#ifndef __OVXLIB_OPERATION_H__
#define __OVXLIB_OPERATION_H__

#include <vector>
#include <iostream>
#include "operand.h"

namespace ovxlib
{

struct VxParam
{
    OverflowPolicy overflowPolicy;
    RoundingPolicy roundingPolicy;
    Rounding downScaleSizeRounding;
    uint32_t accumulatorBits;
};

class Operation
{
    public:
        Operation(OperationType type);

        OperationType type() {return type_;}

        void setType(OperationType type) {type_ = type;}

        std::vector<uint32_t>& inputs() {return inputs_;}

        std::vector<uint32_t>& outputs() {return outputs_;}

        size_t inputNum() {return inputs_.size();}

        size_t outputNum() {return outputs_.size();}

        uint32_t input(uint32_t index);

        uint32_t output(uint32_t index);

        void setInputs(const uint32_t* inputs, uint32_t input_size);

        void setOutputs(const uint32_t* outputs, uint32_t output_size);

        void setInputs(const std::vector<uint32_t>& inputs);

        void setOutputs(const std::vector<uint32_t>& outputs);

        bool replaceOutputs(uint32_t org_index, uint32_t new_index);

        bool replaceInputs(uint32_t org_index, uint32_t new_index);

        int find_position(std::vector<uint32_t> operands_indexes, uint32_t index);

        void echo(uint32_t index = 0);

        void setVxParam(OverflowPolicy overflow_policy = OverflowPolicy::WRAP,
            RoundingPolicy rounding_policy = RoundingPolicy::TO_ZERO,
            Rounding down_scale_size_rounding = Rounding::FLOOR,
            uint32_t accumulator_bits = 0
            );

        VxParam& vxParam() {return vx_param_;}

        void setFusedType(FusedType fuse_type) {fused_type_ = fuse_type;}

        FusedType fusedType() {return fused_type_;}

        void setOperandLayout(OperandLayout layout) {operand_layout_ = layout;}
        OperandLayout getOperandLayout() {return operand_layout_;}

    private:
        OperationType type_{OperationType::NONE};

        std::vector<uint32_t> inputs_;

        std::vector<uint32_t> outputs_;

        FusedType fused_type_{FusedType::NONE};

        VxParam vx_param_;

        OperandLayout operand_layout_{OperandLayout::NCHW};
};


/*
 * Examples of operation definitions.
 * TODO: Move to ovxlib cplus public header.
 */
#define OVXLIB_DECLARE_OPERATION(name, type)                    \
    struct name##Operation: Operation {                         \
        name##Operation(): Operation(OperationType::type) {}    \
    }

struct ReshapeOperation: Operation
{
    ReshapeOperation(): Operation(OperationType::RESHAPE) {}
    std::vector<int32_t> shape;
};

struct PermuteOperation: Operation
{
    PermuteOperation(): Operation(OperationType::TRANSPOSE) {}
    std::vector<int32_t> perm;
};

struct Conv2DOperation: Operation
{
    Conv2DOperation(): Operation(OperationType::CONV_2D) {
        strides.resize(2);
        dilations.resize(2);
        pad.resize(4);
    }
    std::vector<int32_t> strides;
    std::vector<int32_t> dilations;
    std::vector<int32_t> pad;
    PadType padType{PadType::AUTO};
};

struct DepthwiseConv2DOperation: Operation
{
    DepthwiseConv2DOperation(): Operation(OperationType::DEPTHWISE_CONV_2D) {
        strides.resize(2);
        dilations.resize(2);
        pad.resize(4);
    }
    std::vector<int32_t> strides;
    std::vector<int32_t> dilations;
    std::vector<int32_t> pad;
    PadType padType{PadType::AUTO};
    int32_t multiplier;
};

struct ConcatOperation: Operation
{
    ConcatOperation(): Operation(OperationType::CONCATENATION) {}
    int32_t axis;
};

struct SplitOperation: Operation
{
    SplitOperation(): Operation(OperationType::SPLIT) {}
    int32_t axis;
};

struct AveragePool2DOperation: Operation
{
    AveragePool2DOperation(): Operation(OperationType::AVERAGE_POOL_2D) {
        strides.resize(2);
        ksize.resize(2);
        pad.resize(4);
    }
    std::vector<int32_t> strides;
    std::vector<int32_t> ksize;
    std::vector<int32_t> pad;
    PadType padType{PadType::AUTO};
    Rounding roundType{Rounding::FLOOR};
    PoolMode poolMode{PoolMode::VALID};
};

struct MaxPool2DOperation: Operation
{
    MaxPool2DOperation(): Operation(OperationType::MAX_POOL_2D) {
        strides.resize(2);
        ksize.resize(2);
        pad.resize(4);
    }
    std::vector<int32_t> strides;
    std::vector<int32_t> ksize;
    std::vector<int32_t> pad;
    PadType padType{PadType::AUTO};
    Rounding roundType{Rounding::FLOOR};
};

struct L2Pool2DOperation: Operation
{
    L2Pool2DOperation(): Operation(OperationType::L2_POOL_2D) {
        strides.resize(2);
        ksize.resize(2);
        pad.resize(4);
    }
    std::vector<int32_t> strides;
    std::vector<int32_t> ksize;
    std::vector<int32_t> pad;
    PadType padType{PadType::AUTO};
    Rounding roundType{Rounding::FLOOR};
};

struct SqueezeOperation: Operation
{
    SqueezeOperation(): Operation(OperationType::SQUEEZE) {}
    std::vector<int32_t> axes;
};

struct SoftmaxOperation: Operation
{
    SoftmaxOperation(): Operation(OperationType::SOFTMAX) {}
    float beta{1.0f};
};

struct PadOperation: Operation
{
    PadOperation(): Operation(OperationType::PAD) {}
    std::vector<int32_t> padFront;
    std::vector<int32_t> padBack;
    float padValue{0.0f};
    PadMode padMode{PadMode::CONSTANT};
};

struct MeanOperation: Operation
{
    MeanOperation(): Operation(OperationType::MEAN) {}
    std::vector<int32_t> axes;
    bool keepDim{true};
};

struct Conv1DOperation: Operation
{
    Conv1DOperation(): Operation(OperationType::CONV_1D) {
        strides.resize(1);
        dilations.resize(1);
        pad.resize(2);
    }
    std::vector<int32_t> strides;
    std::vector<int32_t> dilations;
    std::vector<int32_t> pad;
    PadType padType{PadType::AUTO};
};

struct TanhOperation: Operation
{
    TanhOperation(): Operation(OperationType::TANH) {}
    float scaleA{1.0f};
    float scaleB{1.0f};
};

struct LeakyReluOperation: Operation
{
    LeakyReluOperation(): Operation(OperationType::LEAKY_RELU) {}
    float ratio{0.1f};
};

struct ResizeBilinearOperation: Operation
{
    ResizeBilinearOperation(): Operation(OperationType::RESIZE_BILINEAR) {}
    int outputHeight;
    int outputWidth;
};

struct ResizeNearestOperation: Operation
{
    ResizeNearestOperation(): Operation(OperationType::RESIZE_NEAREST) {}
    int outputHeight;
    int outputWidth;
};

struct Unpool2DOperation: Operation
{
    Unpool2DOperation(): Operation(OperationType::UNPOOL_2D) {}
    int output_height;
    int output_width;
};

struct MatrixMulOperation: Operation
{
    MatrixMulOperation(): Operation(OperationType::MATRIX_MUL) {}
    bool transpose[2];
};

struct BatchNormOperation: Operation
{
    BatchNormOperation(): Operation(OperationType::BATCH_NORM) {}
    float eps;
};

struct LocalResponseNormOperation: Operation
{
    LocalResponseNormOperation(): Operation(OperationType::LOCAL_RESPONSE_NORM) {}
    int32_t radius;
    float bias;
    float scale; //alpha
    float exponent; //beta
};

struct SvdfOperation: Operation
{
    SvdfOperation(): Operation(OperationType::SVDF) {}
    int32_t rank;
};

struct LstmUnitOperation: Operation
{
    LstmUnitOperation(): Operation(OperationType::LSTM_UNIT) {}
    FusedType activation{FusedType::TANH};
    float cellClip{0.0f};
    float projClip{0.0f};
    float forgetBias{0.0f};

    static const uint8_t INPUT_COUNT = 24;
    static const uint8_t OUTPUT_COUNT = 4;
};

struct LstmLayerOperation: Operation
{
    LstmLayerOperation(): Operation(OperationType::LSTM_LAYER) {}
    FusedType activation{FusedType::TANH};
    float cellClip{0.0f};
    float projClip{0.0f};
    float forgetBias{0.0f};
};

struct RnnOperation: Operation
{
    RnnOperation(): Operation(OperationType::RNN) {}

    int32_t activation;
};

struct DepthToSpaceOperation: Operation
{
    DepthToSpaceOperation(): Operation(OperationType::DEPTH_TO_SPACE) {}
    int32_t blockSize[2];
};

struct SpaceToDepthOperation: Operation
{
    SpaceToDepthOperation(): Operation(OperationType::SPACE_TO_DEPTH) {}
    int32_t blockSize[2];
};

struct BatchToSpaceNDOperation: Operation
{
    BatchToSpaceNDOperation(): Operation(OperationType::BATCH_TO_SPACE_ND) {}
    std::vector<int32_t> blockSize;
    std::vector<int32_t> cropStart;
    std::vector<int32_t> cropEnd;
};

struct SpaceToBatchNDOperation: Operation
{
    SpaceToBatchNDOperation(): Operation(OperationType::SPACE_TO_BATCH_ND) {}
    std::vector<int32_t> blockSize;
    std::vector<int32_t> padFront;
    std::vector<int32_t> padBack;
};

struct StridedSliceOperation: Operation
{
    StridedSliceOperation(): Operation(OperationType::STRIDED_SLICE) {}
    std::vector<int32_t> starts;
    std::vector<int32_t> ends;
    std::vector<int32_t> strides;
    int32_t beginMask;
    int32_t endMask;
    int32_t shrinkAxisMask;
};

struct ReverseOperation: Operation
{
    ReverseOperation(): Operation(OperationType::REVERSE) {}
    std::vector<int32_t> axes;
};

struct LshProjectionOperation: Operation
{
    LshProjectionOperation(): Operation(OperationType::LSH_PROJECTION){}
    LshProjectionType type{LshProjectionType::SPARSE};
};

OVXLIB_DECLARE_OPERATION(Add, ADD);
OVXLIB_DECLARE_OPERATION(Mul, MUL);
OVXLIB_DECLARE_OPERATION(Sub, SUB);
OVXLIB_DECLARE_OPERATION(Div, DIV);
OVXLIB_DECLARE_OPERATION(Minimum, MINIMUM);
OVXLIB_DECLARE_OPERATION(Maximum, MAXIMUM);
OVXLIB_DECLARE_OPERATION(Relu, RELU);
OVXLIB_DECLARE_OPERATION(Relu1, RELU1);
OVXLIB_DECLARE_OPERATION(Relu6, RELU6);
OVXLIB_DECLARE_OPERATION(PRelu, PRELU);
OVXLIB_DECLARE_OPERATION(Sigmoid, SIGMOID);
OVXLIB_DECLARE_OPERATION(SoftRelu, SOFT_RELU);
OVXLIB_DECLARE_OPERATION(Floor, FLOOR);
OVXLIB_DECLARE_OPERATION(Quantize, QUANTIZE);
OVXLIB_DECLARE_OPERATION(Dequantize, DEQUANTIZE);
OVXLIB_DECLARE_OPERATION(Noop, NOOP);
OVXLIB_DECLARE_OPERATION(Abs, ABS);
OVXLIB_DECLARE_OPERATION(Pow, POW);
OVXLIB_DECLARE_OPERATION(Sqrt, SQRT);
OVXLIB_DECLARE_OPERATION(RSqrt, RSQRT);
OVXLIB_DECLARE_OPERATION(HashtableLookup, HASHTABLE_LOOKUP);
OVXLIB_DECLARE_OPERATION(EmbeddingLookup, EMBEDDING_LOOKUP);
OVXLIB_DECLARE_OPERATION(ImageProcess, IMAGE_PROCESS);
OVXLIB_DECLARE_OPERATION(FullyConnected, FULLY_CONNECTED);
OVXLIB_DECLARE_OPERATION(L2Norm, L2_NORM);
OVXLIB_DECLARE_OPERATION(DataConvert, DATA_CONVERT);
#undef OVXLIB_DECLARE_OPERATION

}

#endif
