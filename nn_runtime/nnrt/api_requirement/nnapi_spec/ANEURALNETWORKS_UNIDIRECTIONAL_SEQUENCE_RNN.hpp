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
* sequence of inputs.
*
* This layer unrolls the input along the sequence dimension, and implements
* the following operation
* for each element in the sequence s = 1...sequence_length:
*   outputs[s] = state = activation(inputs[s] * input_weights’ + state *
*   recurrent_weights’ + bias)
*
* Where:
* * “input_weights” is a weight matrix that multiplies the inputs;
* * “recurrent_weights” is a weight matrix that multiplies the current
*    “state” which itself is the output from the previous time step
*    computation;
* * “bias” is a bias vector (added to each output vector in the batch);
* * “activation” is the function passed as the “fused_activation_function”
*   argument (if not “NONE”).
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
*      it is set to 1, then the input has a shape [maxTime, batchSize,
*      inputSize], otherwise the input has a shape [batchSize, maxTime,
*      inputSize].
* * 1: weights.
*      A 2-D tensor of shape [numUnits, inputSize].
* * 2: recurrent_weights.
*      A 2-D tensor of shape [numUnits, numUnits].
* * 3: bias.
*      A 1-D tensor of shape [numUnits].
* * 4: hidden state
*      A 2-D tensor of shape [batchSize, numUnits]. Specifies a hidden
*      state input for the first time step of the computation.
* * 5: fusedActivationFunction.
*      A {@link FuseCode} value indicating the activation function. If
*      “NONE” is specified then it results in a linear activation.
* * 6: timeMajor
*      An {@link ANEURALNETWORKS_INT32} scalar specifying the shape format
*      of input and output tensors. Must be set to either 0 or 1.
* Outputs:
* * 0: output.
*      A 3-D tensor. The shape is defined by the input 6 (timeMajor). If
*      it is set to 1, then the output has a shape [maxTime, batchSize,
*      numUnits], otherwise the output has a shape [batchSize, maxTime,
*      numUnits].
*
* Available since API level 29.
*/

#ifndef __AANEURALNETWORKS_UNIDIRECTIONAL_SEQUENCE_RNN_HPP__
#define __AANEURALNETWORKS_UNIDIRECTIONAL_SEQUENCE_RNN_HPP__

#include "api_requirement/spec_macros.hpp"

#define OP_SPEC_NAME UnidirectionalSequenceRnnOperation
OP_SPEC_BEGIN()
#define ARG_NAMES         \
    (input,               \
     weights,        \
     recurrent_weights,         \
     bias,              \
     hidden_state,               \
     fusedActivationFunction,        \
     timeMajor)
#define ARGC BOOST_PP_TUPLE_SIZE(ARG_NAMES)

#define BOOST_PP_LOCAL_MACRO(n) OP_SPEC_ARG(BOOST_PP_TUPLE_ELEM(ARGC, n, ARG_NAMES))
#define BOOST_PP_LOCAL_LIMITS (0, ARGC)
#include BOOST_PP_LOCAL_ITERATE()
OP_SPEC_END()

// order of argument is important
MAKE_SPEC(unidirectional_sequence_rnn)
    .input_(nnrt::OperandType::TENSOR_FLOAT32)
    .weights_(nnrt::OperandType::TENSOR_FLOAT32)
    .recurrent_weights_(nnrt::OperandType::TENSOR_FLOAT32)
    .bias_(nnrt::OperandType::TENSOR_FLOAT32)
    .hidden_state_(nnrt::OperandType::TENSOR_FLOAT32)
    .fusedActivationFunction_(nnrt::OperandType::INT32)
    .timeMajor_(nnrt::OperandType::INT32)
    );

    OVERRIDE_SPEC(unidirectional_sequence_rnn, 0)
    .input_(nnrt::OperandType::TENSOR_FLOAT16)
    .weights_(nnrt::OperandType::TENSOR_FLOAT16)
    .recurrent_weights_(nnrt::OperandType::TENSOR_FLOAT16)
    .bias_(nnrt::OperandType::TENSOR_FLOAT16)
    .hidden_state_(nnrt::OperandType::TENSOR_FLOAT16)
    );

#undef ARG_NAMES
#undef ARGC
#undef OP_SPEC_NAME

#endif
