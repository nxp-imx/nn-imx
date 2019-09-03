#ifndef __OVXLIB_TYPES_H__
#define __OVXLIB_TYPES_H__

#include "stdint.h"
namespace ovxlib {

enum class OperationType: uint32_t {
    NONE,
    ADD,
    MUL,
    DIV,
    SUB,

    CONCATENATION,
    SPLIT,

    CONV_2D,
    DEPTHWISE_CONV_2D,
    FULLY_CONNECTED,
    CONV_1D,
    //FCL2,

    AVERAGE_POOL_2D,
    MAX_POOL_2D,
    L2_POOL_2D,
    //CONV_RELU_POOL,
    //DECONVOLUTION,
    //POOL_WITH_ARGMAX,

    MEAN,
    PAD,
    RELU,
    RELU1,
    RELU6,
    TANH,
    LEAKY_RELU,
    PRELU,
    SIGMOID, //LOGISTIC
    LOGISTIC = SIGMOID,
    SOFT_RELU,
    SOFTMAX,

    RESIZE_BILINEAR,
    RESIZE_NEAREST,
    UNPOOL_2D,

    L2_NORM,
    L2_NORMALIZATION = L2_NORM,
    LOCAL_RESPONSE_NORM,
    LOCAL_RESPONSE_NORMALIZATION = LOCAL_RESPONSE_NORM,
    BATCH_NORM,
    //LAYER_NORM,
    //LOCAL_RESPONSE_NORM2,
    //INSTANCE_NORM,

    DATA_CONVERT,

    RESHAPE,
    SQUEEZE,
    TRANSPOSE,
    REVERSE,
    SPACE_TO_DEPTH,
    DEPTH_TO_SPACE,
    SPACE_TO_BATCH_ND,
    BATCH_TO_SPACE_ND,
    //ROI_POOL
    //REORG

    FLOOR,

    RNN,
    SVDF,
    HASHTABLE_LOOKUP,
    EMBEDDING_LOOKUP,
    //LSTMUNIT_OVXLIB
    LSTM_UNIT,
    LSTM = LSTM_UNIT,
    LSTM_LAYER,

    LSH_PROJECTION,
    DEQUANTIZE,
    QUANTIZE,

    //SLICE,
    STRIDED_SLICE,

    //PROPOSAL,
    NOOP,

    //VARIABLE
    //CROP
    SQRT,
    RSQRT,
    //DROPOUT, //No need
    //SHUFFLE_CHANNEL
    //SCALE
    MATRIX_MUL,
    //ELU
    //TENSOR_STACK_CONCAT
    //SIGNAL_FRAME
    //A_TIMES_B_PLUS_C
    ABS,
    //NBG
    POW,
    //FLOOR_DIV
    MINIMUM,
    MAXIMUM,
    //SPATIAL_TRANSFORMER

    //SELECT

    IMAGE_PROCESS,
    //SYNC_HOST
    //CONCAT_SHIFT

    //TENSOR_ADD_MEAN_STDDEV_NORM
    //LSTMUNIT_ACTIVATION
};

enum class OperandType: uint8_t {
    NONE,
    INT8,
    INT16,
    INT32,
    UINT8,
    UINT16,
    UINT32,
    FLOAT16,
    FLOAT32,
    FLOAT64,
    TENSOR_INDEX_START,
    TENSOR_INT16 = TENSOR_INDEX_START,
    TENSOR_FLOAT16,
    TENSOR_FLOAT32,
    TENSOR_INT32,
    TENSOR_QUANT8_ASYMM,
    TENSOR_QUANT8_SYMM,
    TENSOR_QUANT32_SYMM,
};

enum TensorLifeTime
{
    VIRTUAL,
    NORMAL,
    CONST,
};

enum class FusedType: uint8_t {
    NONE,
    RELU,
    RELU1,
    RELU6,
    TANH,
    SIGMOID,
};

enum class QuantizerType: uint8_t {
    ASYMMETRIC,
    SYMMETRIC,
    ASYMMETRIC_PER_CHANNEL,
    SYMMETRIC_PER_CHANNEL
};

enum class PadType: uint8_t {
    AUTO,
    VALID,
    SAME,
};

enum class OverflowPolicy {
    WRAP,
    SATURATE,
};

enum class RoundingPolicy {
    TO_ZERO,
    RTNE,
};

enum class Rounding {
    FLOOR,
    CEILING,
    RTNE,
};

enum class PadMode {
    CONSTANT,
};

/*
 * In border mode, Caffe computes with full kernel size,
 * Android & Tensorflow computes with valid kernel size.
 */
enum class PoolMode {
    VALID,
    FULL,
};

enum class LshProjectionType {
    SPARSE = 1,
    DENSE = 2,
};

enum class OperandLayout {
    NHWC,
    NCHW,
    WHCN // openvx type
};

}
#endif
