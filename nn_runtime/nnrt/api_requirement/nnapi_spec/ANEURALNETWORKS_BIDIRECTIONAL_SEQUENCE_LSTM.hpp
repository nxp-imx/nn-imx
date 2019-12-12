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
* Performs a forward LSTM on the input followed by a backward LSTM.
*
* Supported tensor {@link OperandCode}:
* * {@link ANEURALNETWORKS_TENSOR_FLOAT16}
* * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
*
* Supported tensor rank: 3, either time-major or batch-major.
*
* All input and output tensors must be of the same type.
*
*
* Inputs:
* * 0: The input.
*      A 3-D tensor of shape:
*        If time-major: [max_time, batch_size, input_size]
*        If batch-major: [batch_size, max_time, input_size]
*      where "max_time" is the number of timesteps (sequence length),
*      "batch_size" corresponds to the batching dimension, and
*      "input_size" is the size of the input.
* * 1: The forward input-to-input weights. Optional.
*      A 2-D tensor of shape [fw_num_units, input_size], where “fw_num_units”
*      corresponds to the number of forward cell units.
* * 2: The forward input-to-forget weights.
*      A 2-D tensor of shape [fw_num_units, input_size].
* * 3: The forward input-to-cell weights.
*      A 2-D tensor of shape [fw_num_units, input_size].
* * 4: The forward input-to-output weights.
*      A 2-D tensor of shape [fw_num_units, input_size].
* * 5: The forward recurrent-to-input weights. Optional.
*      A 2-D tensor of shape [fw_num_units, fw_output_size], where “fw_output_size”
*      corresponds to either the number of cell units (i.e., fw_num_units),
*      or the second dimension of the “fw_projection_weights”, if defined.
* * 6: The forward recurrent-to-forget weights.
*      A 2-D tensor of shape [fw_num_units, fw_output_size].
* * 7: The forward recurrent-to-cell weights.
*      A 2-D tensor of shape [fw_num_units, fw_output_size].
* * 8: The forward recurrent-to-output weights.
*      A 2-D tensor of shape [fw_num_units, fw_output_size].
* * 9: The forward cell-to-input weights. Optional.
*      A 1-D tensor of shape [fw_num_units].
* * 10: The forward cell-to-forget weights. Optional.
*       A 1-D tensor of shape [fw_num_units].
* * 11: The forward cell-to-output weights. Optional.
*       A 1-D tensor of shape [fw_num_units].
* * 12: The forward input gate bias. Optional.
*       A 1-D tensor of shape [fw_num_units].
* * 13: The forward forget gate bias.
*       A 1-D tensor of shape [fw_num_units].
* * 14: The forward cell gate bias.
*       A 1-D tensor of shape [fw_num_units].
* * 15: The forward output gate bias.
*       A 1-D tensor of shape [fw_num_units].
* * 16: The forward projection weights. Optional.
*       A 2-D tensor of shape [fw_output_size, fw_num_units].
* * 17: The forward projection bias. Optional.
*       A 1-D tensor of shape [fw_output_size].
* * 18: The backward input-to-input weights. Optional.
*       A 2-D tensor of shape [bw_num_units, input_size], where “bw_num_units”
*       corresponds to the number of backward cell units.
* * 19: The backward input-to-forget weights.
*       A 2-D tensor of shape [bw_num_units, input_size].
* * 20: The backward input-to-cell weights.
*       A 2-D tensor of shape [bw_num_units, input_size].
* * 21: The backward input-to-output weights.
*       A 2-D tensor of shape [bw_num_units, input_size].
* * 22: The backward recurrent-to-input weights. Optional.
*       A 2-D tensor of shape [bw_num_units, bw_output_size], where “bw_output_size”
*       corresponds to either the number of cell units (i.e., “bw_num_units”),
*       or the second dimension of the “bw_projection_weights”, if defined.
* * 23: The backward recurrent-to-forget weights.
*       A 2-D tensor of shape [bw_num_units, bw_output_size].
* * 24: The backward recurrent-to-cell weights.
*       A 2-D tensor of shape [bw_num_units, bw_output_size].
* * 25: The backward recurrent-to-output weights.
*       A 2-D tensor of shape [bw_num_units, bw_output_size].
* * 26: The backward cell-to-input weights. Optional.
*       A 1-D tensor of shape [bw_num_units].
* * 27: The backward cell-to-forget weights. Optional.
*       A 1-D tensor of shape [bw_num_units].
* * 28: The backward cell-to-output weights. Optional.
*       A 1-D tensor of shape [bw_num_units].
* * 29: The backward input gate bias. Optional.
*       A 1-D tensor of shape [bw_num_units].
* * 30: The backward forget gate bias.
*       A 1-D tensor of shape [bw_num_units].
* * 31: The backward cell gate bias.
*       A 1-D tensor of shape [bw_num_units].
* * 32: The backward output gate bias.
*       A 1-D tensor of shape [bw_num_units].
* * 33: The backward projection weights. Optional.
*       A 2-D tensor of shape [bw_output_size, bw_num_units].
* * 34: The backward projection bias. Optional.
*       A 1-D tensor of shape [bw_output_size].
* * 35: The forward input activation state.
*       A 2-D tensor of shape [batch_size, bw_output_size].
* * 36: The forward input cell state.
*       A 2-D tensor of shape [batch_size, bw_num_units].
* * 37: The backward input activation state.
*       A 2-D tensor of shape [batch_size, bw_output_size].
* * 38: The backward input cell state.
*       A 2-D tensor of shape [batch_size, bw_num_units].
* * 39: The auxiliary input. Optional.
*       A 3-D tensor of shape [max_time, batch_size, input_size], where “batch_size”
*       corresponds to the batching dimension, and “input_size” is the size
*       of the input.
* * 40: The forward auxiliary input-to-input weights. Optional.
*       A 2-D tensor of shape [fw_num_units, input_size].
* * 41: The forward auxiliary input-to-forget weights. Optional.
*       A 2-D tensor of shape [fw_num_units, input_size].
* * 42: The forward auxiliary input-to-cell weights. Optional.
*       A 2-D tensor of shape [fw_num_units, input_size].
* * 43: The forward auxiliary input-to-output weights. Optional.
*       A 2-D tensor of shape [fw_num_units, input_size].
* * 44: The backward auxiliary input-to-input weights. Optional.
*       A 2-D tensor of shape [bw_num_units, input_size].
* * 45: The backward auxiliary input-to-forget weights. Optional.
*       A 2-D tensor of shape [bw_num_units, input_size].
* * 46: The backward auxiliary input-to-cell weights. Optional.
*       A 2-D tensor of shape [bw_num_units, input_size].
* * 47: The backward auxiliary input-to-output weights. Optional.
*       A 2-D tensor of shape [bw_num_units, input_size].
* * 48: The activation function.
*       A value indicating the activation function:
*       <ul>
*       <li>0: None;
*       <li>1: Relu;
*       <li>3: Relu6;
*       <li>4: Tanh;
*       <li>6: Sigmoid.
*       </ul>
* * 49: The clipping threshold for the cell state, such
*       that values are bound within [-cell_clip, cell_clip]. If set to 0.0
*       then clipping is disabled.
*       If all the input tensors have type {@link ANEURALNETWORKS_TENSOR_FLOAT32},
*       this scalar must be of the type {@link ANEURALNETWORKS_FLOAT32},
*       otherwise if all the input tensors have the type {@link
*       ANEURALNETWORKS_TENSOR_FLOAT16}, this scalar must be of type {@link
*       ANEURALNETWORKS_FLOAT16}.
* * 50: The clipping threshold for the output from the
*       projection layer, such that values are bound within
*       [-proj_clip, proj_clip]. If set to 0.0 then clipping is disabled.
*       If all the input tensors have type {@link ANEURALNETWORKS_TENSOR_FLOAT32},
*       this scalar must be of the type {@link ANEURALNETWORKS_FLOAT32},
*       otherwise if all the input tensors have the type {@link
*       ANEURALNETWORKS_TENSOR_FLOAT16}, this scalar must be of type {@link
*       ANEURALNETWORKS_FLOAT16}.
* * 51: merge_outputs
*       An {@link ANEURALNETWORKS_BOOL} scalar specifying if the outputs
*       from forward and backward cells should be merged.
* * 52: time_major
*       An {@link ANEURALNETWORKS_BOOL} scalar specifying the shape format
*       of input and output tensors.
* * 53: The forward input layer normalization weights. Optional.
*       A 1-D tensor of shape [fw_num_units]. Used to rescale normalized inputs
*       to activation at input gate.
* * 54: The forward forget layer normalization weights. Optional.
*       A 1-D tensor of shape [fw_num_units]. Used to rescale normalized inputs
*       to activation at forget gate.
* * 55: The forward cell layer normalization weights. Optional.
*       A 1-D tensor of shape [fw_num_units]. Used to rescale normalized inputs
*       to activation at cell gate.
* * 56: The forward output layer normalization weights. Optional.
*       A 1-D tensor of shape [fw_num_units]. Used to rescale normalized inputs
*       to activation at output gate.
* * 57: The backward input layer normalization weights. Optional.
*       A 1-D tensor of shape [bw_num_units]. Used to rescale normalized inputs
*       to activation at input gate.
* * 58: The backward forget layer normalization weights. Optional.
*       A 1-D tensor of shape [bw_num_units]. Used to rescale normalized inputs
*       to activation at forget gate.
* * 59: The backward cell layer normalization weights. Optional.
*       A 1-D tensor of shape [bw_num_units]. Used to rescale normalized inputs
*       to activation at cell gate.
* * 60: The backward output layer normalization weights. Optional.
*       A 1-D tensor of shape [bw_num_units]. Used to rescale normalized inputs
*       to activation at output gate.
*
* Outputs:
* * 0: The forward output.
*      A 3-D tensor of shape:
*        If time-major and not merge_outputs:
*          [max_time, batch_size, fw_output_size]
*        If time-major and merge_outputs:
*          [max_time, batch_size, fw_output_size + bw_output_size]
*        If batch-major and not merge_outputs:
*          [batch_size, max_time, fw_output_size]
*        If batch-major and merge_outputs:
*          [batch_size, max_time, fw_output_size + bw_output_size]
* * 1: The backward output.  Unused if merge_outputs is true.
*      A 3-D tensor of shape:
*        If time-major: [max_time, batch_size, bw_output_size]
*        If batch-major: [batch_size, max_time, bw_output_size]
*
* Available since API level 29.
*/

#ifndef __AANEURALNETWORKS_BIDIRECTIONAL_SEQUENCE_LSTM_HPP__
#define __AANEURALNETWORKS_BIDIRECTIONAL_SEQUENCE_LSTM_HPP__

#include "api_requirement/spec_macros.hpp"

#define OP_SPEC_NAME BidirectionalSequenceLstmOperation
OP_SPEC_BEGIN()
#define ARG_NAMES         \
    (input,               \
    fw_input_weight_i2i, \
    fw_input_weight_i2f, \
    fw_input_weight_i2c, \
    fw_input_weight_i2o, \
    fw_input_weight_r2i, \
    fw_input_weight_r2f, \
    fw_input_weight_r2c, \
    fw_input_weight_r2o, \
    fw_input_weight_c2i, \
    fw_input_weight_c2f, \
    fw_input_weight_c2o, \
    fw_input_bias_i, \
    fw_input_bias_f, \
    fw_input_bias_c, \
    fw_input_bias_o, \
    fw_input_weight_proj, \
    fw_input_bias_proj, \
    bw_input_weight_i2i, \
    bw_input_weight_i2f, \
    bw_input_weight_i2c, \
    bw_input_weight_i2o, \
    bw_input_weight_r2i, \
    bw_input_weight_r2f, \
    bw_input_weight_r2c, \
    bw_input_weight_r2o, \
    bw_input_weight_c2i, \
    bw_input_weight_c2f, \
    bw_input_weight_c2o, \
    bw_input_bias_i, \
    bw_input_bias_f, \
    bw_input_bias_c, \
    bw_input_bias_o, \
    bw_input_weight_proj, \
    bw_input_bias_proj, \
    fw_input_h_state, \
    fw_input_c_state, \
    bw_input_h_state, \
    bw_input_c_state, \
    aux_input, \
    fw_aux_input_weight_i2i, \
    fw_aux_input_weight_i2f, \
    fw_aux_input_weight_i2c, \
    fw_aux_input_weight_i2o, \
    bw_aux_input_weight_i2i, \
    bw_aux_input_weight_i2f, \
    bw_aux_input_weight_i2c, \
    bw_aux_input_weight_i2o, \
    activation, \
    cell_clip, \
    proj_clip, \
    merge_outputs, \
    time_major, \
    fw_input_layernorm_i, \
    fw_input_layernorm_f, \
    fw_input_layernorm_c, \
    fw_input_layernorm_o, \
    bw_input_layernorm_i, \
    bw_input_layernorm_f, \
    bw_input_layernorm_c, \
    bw_input_layernorm_o)
#define ARGC BOOST_PP_TUPLE_SIZE(ARG_NAMES)

#define BOOST_PP_LOCAL_MACRO(n) OP_SPEC_ARG(BOOST_PP_TUPLE_ELEM(ARGC, n, ARG_NAMES))
#define BOOST_PP_LOCAL_LIMITS (0, ARGC)
#include BOOST_PP_LOCAL_ITERATE()
OP_SPEC_END()

// order of argument is important
MAKE_SPEC(bidirectional_sequence_lstm)
    .input_(nnrt::OperandType::TENSOR_FLOAT32)
    .fw_input_weight_i2i_(nnrt::OperandType::TENSOR_FLOAT32)
    .fw_input_weight_i2f_(nnrt::OperandType::TENSOR_FLOAT32)
    .fw_input_weight_i2c_(nnrt::OperandType::TENSOR_FLOAT32)
    .fw_input_weight_i2o_(nnrt::OperandType::TENSOR_FLOAT32)
    .fw_input_weight_r2i_(nnrt::OperandType::TENSOR_FLOAT32)
    .fw_input_weight_r2f_(nnrt::OperandType::TENSOR_FLOAT32)
    .fw_input_weight_r2c_(nnrt::OperandType::TENSOR_FLOAT32)
    .fw_input_weight_r2o_(nnrt::OperandType::TENSOR_FLOAT32)
    .fw_input_weight_c2i_(nnrt::OperandType::TENSOR_FLOAT32)
    .fw_input_weight_c2f_(nnrt::OperandType::TENSOR_FLOAT32)
    .fw_input_weight_c2o_(nnrt::OperandType::TENSOR_FLOAT32)
    .fw_input_bias_i_(nnrt::OperandType::TENSOR_FLOAT32)
    .fw_input_bias_f_(nnrt::OperandType::TENSOR_FLOAT32)
    .fw_input_bias_c_(nnrt::OperandType::TENSOR_FLOAT32)
    .fw_input_bias_o_(nnrt::OperandType::TENSOR_FLOAT32)
    .fw_input_weight_proj_(nnrt::OperandType::TENSOR_FLOAT32)
    .fw_input_bias_proj_(nnrt::OperandType::TENSOR_FLOAT32)
    .bw_input_weight_i2i_(nnrt::OperandType::TENSOR_FLOAT32)
    .bw_input_weight_i2f_(nnrt::OperandType::TENSOR_FLOAT32)
    .bw_input_weight_i2c_(nnrt::OperandType::TENSOR_FLOAT32)
    .bw_input_weight_i2o_(nnrt::OperandType::TENSOR_FLOAT32)
    .bw_input_weight_r2i_(nnrt::OperandType::TENSOR_FLOAT32)
    .bw_input_weight_r2f_(nnrt::OperandType::TENSOR_FLOAT32)
    .bw_input_weight_r2c_(nnrt::OperandType::TENSOR_FLOAT32)
    .bw_input_weight_r2o_(nnrt::OperandType::TENSOR_FLOAT32)
    .bw_input_weight_c2i_(nnrt::OperandType::TENSOR_FLOAT32)
    .bw_input_weight_c2f_(nnrt::OperandType::TENSOR_FLOAT32)
    .bw_input_weight_c2o_(nnrt::OperandType::TENSOR_FLOAT32)
    .bw_input_bias_i_(nnrt::OperandType::TENSOR_FLOAT32)
    .bw_input_bias_f_(nnrt::OperandType::TENSOR_FLOAT32)
    .bw_input_bias_c_(nnrt::OperandType::TENSOR_FLOAT32)
    .bw_input_bias_o_(nnrt::OperandType::TENSOR_FLOAT32)
    .bw_input_weight_proj_(nnrt::OperandType::TENSOR_FLOAT32)
    .bw_input_bias_proj_(nnrt::OperandType::TENSOR_FLOAT32)
    .fw_input_h_state_(nnrt::OperandType::TENSOR_FLOAT32)
    .fw_input_c_state_(nnrt::OperandType::TENSOR_FLOAT32)
    .bw_input_h_state_(nnrt::OperandType::TENSOR_FLOAT32)
    .bw_input_c_state_(nnrt::OperandType::TENSOR_FLOAT32)
    .aux_input_(nnrt::OperandType::TENSOR_FLOAT32)
    .fw_aux_input_weight_i2i_(nnrt::OperandType::TENSOR_FLOAT32)
    .fw_aux_input_weight_i2f_(nnrt::OperandType::TENSOR_FLOAT32)
    .fw_aux_input_weight_i2c_(nnrt::OperandType::TENSOR_FLOAT32)
    .fw_aux_input_weight_i2o_(nnrt::OperandType::TENSOR_FLOAT32)
    .bw_aux_input_weight_i2i_(nnrt::OperandType::TENSOR_FLOAT32)
    .bw_aux_input_weight_i2f_(nnrt::OperandType::TENSOR_FLOAT32)
    .bw_aux_input_weight_i2c_(nnrt::OperandType::TENSOR_FLOAT32)
    .bw_aux_input_weight_i2o_(nnrt::OperandType::TENSOR_FLOAT32)
    .activation_(nnrt::OperandType::INT32)
    .cell_clip_(nnrt::OperandType::FLOAT32)
    .proj_clip_(nnrt::OperandType::FLOAT32)
    .merge_outputs_(nnrt::OperandType::BOOL)
    .time_major_(nnrt::OperandType::BOOL)
    .fw_input_layernorm_i_(nnrt::OperandType::TENSOR_FLOAT32)
    .fw_input_layernorm_f_(nnrt::OperandType::TENSOR_FLOAT32)
    .fw_input_layernorm_c_(nnrt::OperandType::TENSOR_FLOAT32)
    .fw_input_layernorm_o_(nnrt::OperandType::TENSOR_FLOAT32)
    .bw_input_layernorm_i_(nnrt::OperandType::TENSOR_FLOAT32)
    .bw_input_layernorm_f_(nnrt::OperandType::TENSOR_FLOAT32)
    .bw_input_layernorm_c_(nnrt::OperandType::TENSOR_FLOAT32)
    .bw_input_layernorm_o_(nnrt::OperandType::TENSOR_FLOAT32)
    );

    OVERRIDE_SPEC(bidirectional_sequence_lstm, 0)
    .input_(nnrt::OperandType::TENSOR_FLOAT16)
    .fw_input_weight_i2i_(nnrt::OperandType::TENSOR_FLOAT16)
    .fw_input_weight_i2f_(nnrt::OperandType::TENSOR_FLOAT16)
    .fw_input_weight_i2c_(nnrt::OperandType::TENSOR_FLOAT16)
    .fw_input_weight_i2o_(nnrt::OperandType::TENSOR_FLOAT16)
    .fw_input_weight_r2i_(nnrt::OperandType::TENSOR_FLOAT16)
    .fw_input_weight_r2f_(nnrt::OperandType::TENSOR_FLOAT16)
    .fw_input_weight_r2c_(nnrt::OperandType::TENSOR_FLOAT16)
    .fw_input_weight_r2o_(nnrt::OperandType::TENSOR_FLOAT16)
    .fw_input_weight_c2i_(nnrt::OperandType::TENSOR_FLOAT16)
    .fw_input_weight_c2f_(nnrt::OperandType::TENSOR_FLOAT16)
    .fw_input_weight_c2o_(nnrt::OperandType::TENSOR_FLOAT16)
    .fw_input_bias_i_(nnrt::OperandType::TENSOR_FLOAT16)
    .fw_input_bias_f_(nnrt::OperandType::TENSOR_FLOAT16)
    .fw_input_bias_c_(nnrt::OperandType::TENSOR_FLOAT16)
    .fw_input_bias_o_(nnrt::OperandType::TENSOR_FLOAT16)
    .fw_input_weight_proj_(nnrt::OperandType::TENSOR_FLOAT16)
    .fw_input_bias_proj_(nnrt::OperandType::TENSOR_FLOAT16)
    .bw_input_weight_i2i_(nnrt::OperandType::TENSOR_FLOAT16)
    .bw_input_weight_i2f_(nnrt::OperandType::TENSOR_FLOAT16)
    .bw_input_weight_i2c_(nnrt::OperandType::TENSOR_FLOAT16)
    .bw_input_weight_i2o_(nnrt::OperandType::TENSOR_FLOAT16)
    .bw_input_weight_r2i_(nnrt::OperandType::TENSOR_FLOAT16)
    .bw_input_weight_r2f_(nnrt::OperandType::TENSOR_FLOAT16)
    .bw_input_weight_r2c_(nnrt::OperandType::TENSOR_FLOAT16)
    .bw_input_weight_r2o_(nnrt::OperandType::TENSOR_FLOAT16)
    .bw_input_weight_c2i_(nnrt::OperandType::TENSOR_FLOAT16)
    .bw_input_weight_c2f_(nnrt::OperandType::TENSOR_FLOAT16)
    .bw_input_weight_c2o_(nnrt::OperandType::TENSOR_FLOAT16)
    .bw_input_bias_i_(nnrt::OperandType::TENSOR_FLOAT16)
    .bw_input_bias_f_(nnrt::OperandType::TENSOR_FLOAT16)
    .bw_input_bias_c_(nnrt::OperandType::TENSOR_FLOAT16)
    .bw_input_bias_o_(nnrt::OperandType::TENSOR_FLOAT16)
    .bw_input_weight_proj_(nnrt::OperandType::TENSOR_FLOAT16)
    .bw_input_bias_proj_(nnrt::OperandType::TENSOR_FLOAT16)
    .fw_input_h_state_(nnrt::OperandType::TENSOR_FLOAT16)
    .fw_input_c_state_(nnrt::OperandType::TENSOR_FLOAT16)
    .bw_input_h_state_(nnrt::OperandType::TENSOR_FLOAT16)
    .bw_input_c_state_(nnrt::OperandType::TENSOR_FLOAT16)
    .aux_input_(nnrt::OperandType::TENSOR_FLOAT16)
    .fw_aux_input_weight_i2i_(nnrt::OperandType::TENSOR_FLOAT16)
    .fw_aux_input_weight_i2f_(nnrt::OperandType::TENSOR_FLOAT16)
    .fw_aux_input_weight_i2c_(nnrt::OperandType::TENSOR_FLOAT16)
    .fw_aux_input_weight_i2o_(nnrt::OperandType::TENSOR_FLOAT16)
    .bw_aux_input_weight_i2i_(nnrt::OperandType::TENSOR_FLOAT16)
    .bw_aux_input_weight_i2f_(nnrt::OperandType::TENSOR_FLOAT16)
    .bw_aux_input_weight_i2c_(nnrt::OperandType::TENSOR_FLOAT16)
    .bw_aux_input_weight_i2o_(nnrt::OperandType::TENSOR_FLOAT16)
    .cell_clip_(nnrt::OperandType::FLOAT16)
    .proj_clip_(nnrt::OperandType::FLOAT16)
    .fw_input_layernorm_i_(nnrt::OperandType::TENSOR_FLOAT16)
    .fw_input_layernorm_f_(nnrt::OperandType::TENSOR_FLOAT16)
    .fw_input_layernorm_c_(nnrt::OperandType::TENSOR_FLOAT16)
    .fw_input_layernorm_o_(nnrt::OperandType::TENSOR_FLOAT16)
    .bw_input_layernorm_i_(nnrt::OperandType::TENSOR_FLOAT16)
    .bw_input_layernorm_f_(nnrt::OperandType::TENSOR_FLOAT16)
    .bw_input_layernorm_c_(nnrt::OperandType::TENSOR_FLOAT16)
    .bw_input_layernorm_o_(nnrt::OperandType::TENSOR_FLOAT16)
    );

#undef ARG_NAMES
#undef ARGC
#undef OP_SPEC_NAME

#endif
