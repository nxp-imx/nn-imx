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
* A recurrent neural network specified by an LSTM cell.
*
* Performs (fully) dynamic unrolling of input.
*
* This Op unrolls the input along the time dimension, and implements the
* following operation for each element in the sequence
* s = 1...sequence_length:
*   outputs[s] = projection(state = activation(LSTMOp(inputs[s])))
*
* Where LSTMOp is the LSTM op as in {@link ANEURALNETWORKS_LSTM},
* the "projection" is an optional projection layer from state and output
* and the “activation” is the function passed as the
* “fused_activation_function” argument (if not “NONE”).
*
* Supported tensor {@link OperandCode}:
* * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
* * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
*
* Supported tensor rank: 3, either time-major or batch-major.
*
* All input and output tensors must be of the same type.
*
* Inputs:
* * 0: The input (\f$x_t\f$).
*      A 3-D tensor of shape:
*        If time-major: [max_time, batch_size, input_size]
*        If batch-major: [batch_size, max_time, input_size]
*      where “max_time” is the number of timesteps (sequence length),
*      “batch_size” corresponds to the batching dimension, and
*      “input_size” is the size of the input.
* * 1: The input-to-input weights (\f$W_{xi}\f$). Optional.
*      A 2-D tensor of shape [num_units, input_size], where “num_units”
*      corresponds to the number of cell units.
* * 2: The input-to-forget weights (\f$W_{xf}\f$).
*      A 2-D tensor of shape [num_units, input_size].
* * 3: The input-to-cell weights (\f$W_{xc}\f$).
*      A 2-D tensor of shape [num_units, input_size].
* * 4: The input-to-output weights (\f$W_{xo}\f$).
*      A 2-D tensor of shape [num_units, input_size].
* * 5: The recurrent-to-input weights (\f$W_{hi}\f$). Optional.
*      A 2-D tensor of shape [num_units, output_size], where “output_size”
*      corresponds to either the number of cell units (i.e., “num_units”),
*      or the second dimension of the “projection_weights”, if defined.
* * 6: The recurrent-to-forget weights (\f$W_{hf}\f$).
*      A 2-D tensor of shape [num_units, output_size].
* * 7: The recurrent-to-cell weights (\f$W_{hc}\f$).
*      A 2-D tensor of shape [num_units, output_size].
* * 8: The recurrent-to-output weights (\f$W_{ho}\f$).
*      A 2-D tensor of shape [num_units, output_size].
* * 9: The cell-to-input weights (\f$W_{ci}\f$). Optional.
*      A 1-D tensor of shape [num_units].
* * 10:The cell-to-forget weights (\f$W_{cf}\f$). Optional.
*      A 1-D tensor of shape [num_units].
* * 11:The cell-to-output weights (\f$W_{co}\f$). Optional.
*      A 1-D tensor of shape [num_units].
* * 12:The input gate bias (\f$b_i\f$). Optional.
*      A 1-D tensor of shape [num_units].
* * 13:The forget gate bias (\f$b_f\f$).
*      A 1-D tensor of shape [num_units].
* * 14:The cell bias (\f$b_c\f$).
*      A 1-D tensor of shape [num_units].
* * 15:The output gate bias (\f$b_o\f$).
*      A 1-D tensor of shape [num_units].
* * 16:The projection weights (\f$W_{proj}\f$). Optional.
*      A 2-D tensor of shape [output_size, num_units].
* * 17:The projection bias (\f$b_{proj}\f$). Optional.
*      A 1-D tensor of shape [output_size].
* * 18:The output state (in) (\f$h_{t-1}\f$).
*      A 2-D tensor of shape [batch_size, output_size].
* * 19:The cell state (in) (\f$C_{t-1}\f$).
*      A 2-D tensor of shape [batch_size, num_units].
* * 20:The activation function (\f$g\f$).
*      A value indicating the activation function:
*      <ul>
*      <li>0: None;
*      <li>1: Relu;
*      <li>3: Relu6;
*      <li>4: Tanh;
*      <li>6: Sigmoid.
*      </ul>
* * 21:The clipping threshold (\f$t_{cell}\f$) for the cell state, such
*      that values are bound within [-cell_clip, cell_clip]. If set to 0.0
*      then clipping is disabled.
* * 22:The clipping threshold (\f$t_{proj}\f$) for the output from the
*      projection layer, such that values are bound within
*      [-proj_clip, proj_clip]. If set to 0.0 then clipping is disabled.
* * 23:Time-major if true, batch-major if false.
* * 24:The input layer normalization weights. Optional.
*      A 1-D tensor of shape [num_units]. Used to rescale normalized inputs
*      to activation at input gate.
* * 25:The forget layer normalization weights. Optional.
*      A 1-D tensor of shape [num_units]. Used to rescale normalized inputs
*      to activation at forget gate.
* * 26:The cell layer normalization weights. Optional.
*      A 1-D tensor of shape [num_units]. Used to rescale normalized inputs
*      to activation at cell gate.
* * 27:The output layer normalization weights. Optional.
*      A 1-D tensor of shape [num_units]. Used to rescale normalized inputs
*      to activation at output gate.
*
* Outputs:
* * 0: The output (\f$o_t\f$).
*      A 3-D tensor of shape:
*        If time-major: [max_time, batch_size, output_size]
*        If batch-major: [batch_size, max_time, output_size]
*
* Available since API level 29.
*/

#ifndef __AANEURALNETWORKS_UNIDIRECTIONAL_SEQUENCE_LSTM_HPP__
#define __AANEURALNETWORKS_UNIDIRECTIONAL_SEQUENCE_LSTM_HPP__

#include "api_requirement/spec_macros.hpp"

#define OP_SPEC_NAME UnidirectionalSequenceLstmOperation
OP_SPEC_BEGIN()
#define ARG_NAMES         \
    (input,               \
    weight_i2i, \
    weight_i2f, \
    weight_i2c, \
    weight_i2o, \
    weight_r2i, \
    weight_r2f, \
    weight_r2c, \
    weight_r2o, \
    weight_c2i, \
    weight_c2f, \
    weight_c2o, \
    bias_i, \
    bias_f, \
    bias_c, \
    bias_o, \
    weight_proj, \
    bias_proj, \
    h_state, \
    c_state, \
    activation, \
    cell_clip, \
    proj_clip, \
    timeMajor, \
    layernorm_i, \
    layernorm_f, \
    layernorm_c, \
    layernorm_o)
#define ARGC BOOST_PP_TUPLE_SIZE(ARG_NAMES)

#define BOOST_PP_LOCAL_MACRO(n) OP_SPEC_ARG(BOOST_PP_TUPLE_ELEM(ARGC, n, ARG_NAMES))
#define BOOST_PP_LOCAL_LIMITS (0, ARGC)
#include BOOST_PP_LOCAL_ITERATE()
OP_SPEC_END()

// order of argument is important
MAKE_SPEC(unidirectional_sequence_lstm)
    .input_(nnrt::OperandType::TENSOR_FLOAT32)
    .weight_i2i_(nnrt::OperandType::TENSOR_FLOAT32)
    .weight_i2f_(nnrt::OperandType::TENSOR_FLOAT32)
    .weight_i2c_(nnrt::OperandType::TENSOR_FLOAT32)
    .weight_i2o_(nnrt::OperandType::TENSOR_FLOAT32)
    .weight_r2i_(nnrt::OperandType::TENSOR_FLOAT32)
    .weight_r2f_(nnrt::OperandType::TENSOR_FLOAT32)
    .weight_r2c_(nnrt::OperandType::TENSOR_FLOAT32)
    .weight_r2o_(nnrt::OperandType::TENSOR_FLOAT32)
    .weight_c2i_(nnrt::OperandType::TENSOR_FLOAT32)
    .weight_c2f_(nnrt::OperandType::TENSOR_FLOAT32)
    .weight_c2o_(nnrt::OperandType::TENSOR_FLOAT32)
    .bias_i_(nnrt::OperandType::TENSOR_FLOAT32)
    .bias_f_(nnrt::OperandType::TENSOR_FLOAT32)
    .bias_c_(nnrt::OperandType::TENSOR_FLOAT32)
    .bias_o_(nnrt::OperandType::TENSOR_FLOAT32)
    .weight_proj_(nnrt::OperandType::TENSOR_FLOAT32)
    .bias_proj_(nnrt::OperandType::TENSOR_FLOAT32)
    .h_state_(nnrt::OperandType::TENSOR_FLOAT32)
    .c_state_(nnrt::OperandType::TENSOR_FLOAT32)
    .activation_(nnrt::OperandType::INT32)
    .cell_clip_(nnrt::OperandType::FLOAT32)
    .proj_clip_(nnrt::OperandType::FLOAT32)
    .timeMajor_(nnrt::OperandType::BOOL)
    .layernorm_i_(nnrt::OperandType::TENSOR_FLOAT32)
    .layernorm_f_(nnrt::OperandType::TENSOR_FLOAT32)
    .layernorm_c_(nnrt::OperandType::TENSOR_FLOAT32)
    .layernorm_o_(nnrt::OperandType::TENSOR_FLOAT32)
    );

    OVERRIDE_SPEC(unidirectional_sequence_lstm, 0)
    .input_(nnrt::OperandType::TENSOR_FLOAT16)
    .weight_i2i_(nnrt::OperandType::TENSOR_FLOAT16)
    .weight_i2f_(nnrt::OperandType::TENSOR_FLOAT16)
    .weight_i2c_(nnrt::OperandType::TENSOR_FLOAT16)
    .weight_i2o_(nnrt::OperandType::TENSOR_FLOAT16)
    .weight_r2i_(nnrt::OperandType::TENSOR_FLOAT16)
    .weight_r2f_(nnrt::OperandType::TENSOR_FLOAT16)
    .weight_r2c_(nnrt::OperandType::TENSOR_FLOAT16)
    .weight_r2o_(nnrt::OperandType::TENSOR_FLOAT16)
    .weight_c2i_(nnrt::OperandType::TENSOR_FLOAT16)
    .weight_c2f_(nnrt::OperandType::TENSOR_FLOAT16)
    .weight_c2o_(nnrt::OperandType::TENSOR_FLOAT16)
    .bias_i_(nnrt::OperandType::TENSOR_FLOAT16)
    .bias_f_(nnrt::OperandType::TENSOR_FLOAT16)
    .bias_c_(nnrt::OperandType::TENSOR_FLOAT16)
    .bias_o_(nnrt::OperandType::TENSOR_FLOAT16)
    .weight_proj_(nnrt::OperandType::TENSOR_FLOAT16)
    .bias_proj_(nnrt::OperandType::TENSOR_FLOAT16)
    .h_state_(nnrt::OperandType::TENSOR_FLOAT16)
    .c_state_(nnrt::OperandType::TENSOR_FLOAT16)
    .cell_clip_(nnrt::OperandType::FLOAT16)
    .proj_clip_(nnrt::OperandType::FLOAT16)
    .layernorm_i_(nnrt::OperandType::TENSOR_FLOAT16)
    .layernorm_f_(nnrt::OperandType::TENSOR_FLOAT16)
    .layernorm_c_(nnrt::OperandType::TENSOR_FLOAT16)
    .layernorm_o_(nnrt::OperandType::TENSOR_FLOAT16)
    );
#undef ARG_NAMES
#undef ARGC
#undef OP_SPEC_NAME

#endif
