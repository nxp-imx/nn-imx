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
* A recurrent neural network layer that applies a basic RNN cell to a
* sequence of inputs in forward and backward directions.
*
* This Op unrolls the input along the sequence dimension, and implements
* the following operation for each element in the sequence s =
* 1...sequence_length:
*   fw_outputs[s] = fw_state = activation(inputs[s] * fw_input_weights’ +
*          fw_state * fw_recurrent_weights’ + fw_bias)
*
* And for each element in sequence t = sequence_length : 1
*   bw_outputs[t] = bw_state = activation(inputs[t] * bw_input_weights’ +
*          bw_state * bw_recurrent_weights’ + bw_bias)
*
* Where:
* * “{fw,bw}_input_weights” is a weight matrix that multiplies the inputs;
* * “{fw,bw}_recurrent_weights” is a weight matrix that multiplies the
*    current “state” which itself is the output from the previous time step
*    computation;
* * “{fw,bw}_bias” is a bias vector (added to each output vector in the
*    batch);
* * “activation” is the function passed as the “fused_activation_function”
*   argument (if not “NONE”).
*
* The op also supports an auxiliary input. Regular cell feeds one input
* into the two RNN cells in the following way:
*
*       INPUT  (INPUT_REVERSED)
*         |         |
*    ---------------------
*    | FW_RNN     BW_RNN |
*    ---------------------
*         |         |
*      FW_OUT     BW_OUT
*
* An op with an auxiliary input takes two inputs and feeds them into the
* RNN cells in the following way:
*
*       AUX_INPUT   (AUX_INPUT_REVERSED)
*           |             |
*     INPUT | (INPUT_R'D.)|
*       |   |       |     |
*    -----------------------
*    |  \  /        \    / |
*    | FW_RNN       BW_RNN |
*    -----------------------
*         |           |
*      FW_OUT      BW_OUT
*
* While stacking this op on top of itself, this allows to connect both
* forward and backward outputs from previous cell to the next cell's
* inputs.
*
* Supported tensor {@link OperandCode}:
* * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
* * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
*
* The input tensors must all be the same type.
*
* Inputs:
* * 0: input.
*      A 3-D tensor. The shape is defined by the input 6 (timeMajor). If
*      it is set to true, then the input has a shape [maxTime, batchSize,
*      inputSize], otherwise the input has a shape [batchSize, maxTime,
*      inputSize].
* * 1: fwWeights.
*      A 2-D tensor of shape [fwNumUnits, inputSize].
* * 2: fwRecurrentWeights.
*      A 2-D tensor of shape [fwNumUnits, fwNumUnits].
* * 3: fwBias.
*      A 1-D tensor of shape [fwNumUnits].
* * 4: fwHiddenState.
*      A 2-D tensor of shape [batchSize, fwNumUnits]. Specifies a hidden
*      state input for the first time step of the computation.
* * 5: bwWeights.
*      A 2-D tensor of shape [bwNumUnits, inputSize].
* * 6: bwRecurrentWeights.
*      A 2-D tensor of shape [bwNumUnits, bwNumUnits].
* * 7: bwBias.
*      A 1-D tensor of shape [bwNumUnits].
* * 8: bwHiddenState
*      A 2-D tensor of shape [batchSize, bwNumUnits]. Specifies a hidden
*      state input for the first time step of the computation.
* * 9: auxInput.
*      A 3-D tensor. The shape is the same as of the input 0.
* * 10:fwAuxWeights.
*      A 2-D tensor of shape [fwNumUnits, inputSize].
* * 11:bwAuxWeights.
*      A 2-D tensor of shape [bwNumUnits, inputSize].
* * 12:fusedActivationFunction.
*      A {@link FuseCode} value indicating the activation function. If
*      “NONE” is specified then it results in a linear activation.
* * 13:timeMajor
*      An {@link ANEURALNETWORKS_BOOL} scalar specifying the shape format
*      of input and output tensors.
* * 14:mergeOutputs
*      An {@link ANEURALNETWORKS_BOOL} scalar specifying if the outputs
*      from forward and backward cells are separate (if set to false) or
*      concatenated (if set to true).
* Outputs:
* * 0: fwOutput.
*      A 3-D tensor. The first two dimensions of the shape are defined by
*      the input 6 (timeMajor) and the third dimension is defined by the
*      input 14 (mergeOutputs). If timeMajor is set to true, then the first
*      two dimensions are [maxTime, batchSize], otherwise they are set to
*      [batchSize, maxTime]. If mergeOutputs is set to true, then the third
*      dimension is equal to (fwNumUnits + bwNumUnits), otherwise it is set
*      to fwNumUnits.
* * 1: bwOutput.
*      A 3-D tensor. If the input 14 (mergeOutputs) is set to true, then
*      this tensor is not produced. The shape is defined by the input 6
*      (timeMajor). If it is set to true, then the shape is set to
*      [maxTime, batchSize, bwNumUnits], otherwise the shape is set to
*      [batchSize, maxTime, bwNumUnits].
*
* Available since API level 29.
*/

#ifndef __AANEURALNETWORKS_BIDIRECTIONAL_SEQUENCE_RNN_HPP__
#define __AANEURALNETWORKS_BIDIRECTIONAL_SEQUENCE_RNN_HPP__

#include "api_requirement/spec_macros.hpp"

#define OP_SPEC_NAME BidirectionalSequenceRnnOperation
OP_SPEC_BEGIN()
#define ARG_NAMES         \
    (input,               \
    fw_input_weight_i, \
    fw_input_weight_h, \
    fw_input_bias, \
    fw_input_h_state, \
    bw_input_weight_i, \
    bw_input_weight_h, \
    bw_input_bias, \
    bw_input_h_state, \
    aux_input, \
    fw_aux_input_weight, \
    bw_aux_input_weight, \
    activation, \
    timeMajor, \
    mergeOutputs)
#define ARGC BOOST_PP_TUPLE_SIZE(ARG_NAMES)

#define BOOST_PP_LOCAL_MACRO(n) OP_SPEC_ARG(BOOST_PP_TUPLE_ELEM(ARGC, n, ARG_NAMES))
#define BOOST_PP_LOCAL_LIMITS (0, ARGC)
#include BOOST_PP_LOCAL_ITERATE()
OP_SPEC_END()

// order of argument is important
MAKE_SPEC(bidirectional_sequence_rnn)
    .input_(nnrt::OperandType::TENSOR_FLOAT32)
    .fw_input_weight_i_(nnrt::OperandType::TENSOR_FLOAT32)
    .fw_input_weight_h_(nnrt::OperandType::TENSOR_FLOAT32)
    .fw_input_bias_(nnrt::OperandType::TENSOR_FLOAT32)
    .fw_input_h_state_(nnrt::OperandType::TENSOR_FLOAT32)
    .bw_input_weight_i_(nnrt::OperandType::TENSOR_FLOAT32)
    .bw_input_weight_h_(nnrt::OperandType::TENSOR_FLOAT32)
    .bw_input_bias_(nnrt::OperandType::TENSOR_FLOAT32)
    .bw_input_h_state_(nnrt::OperandType::TENSOR_FLOAT32)
    .aux_input_(nnrt::OperandType::TENSOR_FLOAT32)
    .fw_aux_input_weight_(nnrt::OperandType::TENSOR_FLOAT32)
    .bw_aux_input_weight_(nnrt::OperandType::TENSOR_FLOAT32)
    .activation_(nnrt::OperandType::INT32)
    .timeMajor_(nnrt::OperandType::BOOL)
    .mergeOutputs_(nnrt::OperandType::BOOL)
    );

    OVERRIDE_SPEC(bidirectional_sequence_rnn, 0)
    .input_(nnrt::OperandType::TENSOR_FLOAT16)
    .fw_input_weight_i_(nnrt::OperandType::TENSOR_FLOAT16)
    .fw_input_weight_h_(nnrt::OperandType::TENSOR_FLOAT16)
    .fw_input_bias_(nnrt::OperandType::TENSOR_FLOAT16)
    .fw_input_h_state_(nnrt::OperandType::TENSOR_FLOAT16)
    .bw_input_weight_i_(nnrt::OperandType::TENSOR_FLOAT16)
    .bw_input_weight_h_(nnrt::OperandType::TENSOR_FLOAT16)
    .bw_input_bias_(nnrt::OperandType::TENSOR_FLOAT16)
    .bw_input_h_state_(nnrt::OperandType::TENSOR_FLOAT16)
    .aux_input_(nnrt::OperandType::TENSOR_FLOAT16)
    .fw_aux_input_weight_(nnrt::OperandType::TENSOR_FLOAT16)
    .bw_aux_input_weight_(nnrt::OperandType::TENSOR_FLOAT16)
    );

#undef ARG_NAMES
#undef ARGC
#undef OP_SPEC_NAME

#endif
