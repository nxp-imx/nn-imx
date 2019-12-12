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

#ifndef __ANEURALNETWORKS_LSTM_HPP__
#define __ANEURALNETWORKS_LSTM_HPP__

#include "api_requirement/spec_macros.hpp"
    /**
     * Performs a single time step in a Long Short-Term Memory (LSTM) layer
     *
     * The LSTM operation is described by the following equations.
     *
     * \f{eqnarray*}{
     * i_t =& \sigma(W_{xi}x_t+W_{hi}h_{t-1}+W_{ci}C_{t-1}+b_i) & \\
     * f_t =& \sigma(W_{xf}x_t+W_{hf}h_{t-1}+W_{cf}C_{t-1}+b_f) & \\
     * C_t =& clip(f_t \odot C_{t-1} + i_t \odot
     *        g(W_{xc}x_t+W_{hc}h_{t-1}+b_c),\ t_{cell}) & \\
     * o_t =& \sigma(W_{xo}x_t+W_{ho}h_{t-1}+W_{co}C_t+b_o) & \\
     *      & & \\
     *      & clip(W_{proj}(o_t \odot g(C_t))+b_{proj},\ t_{proj})
     *      & if\ there\ is\ a\ projection; \\
     * h_t =& & \\
     *      & o_t \odot g(C_t) & otherwise. \\
     * \f}
     * Where:
     * * \f$x_t\f$ is the input,
     * * \f$i_t\f$ is the input gate,
     * * \f$f_t\f$ is the forget gate,
     * * \f$C_t\f$ is the cell state,
     * * \f$o_t\f$ is the output,
     * * \f$h_t\f$ is the output state,
     * * \f$\sigma\f$ is the logistic sigmoid function,
     * * \f$g\f$ is the cell input and cell output activation function, usually
     *   \f$tahn\f$,
     * * \f$W_{xi}\f$ is the input-to-input weight matrix,
     * * \f$W_{hi}\f$ is the recurrent to input weight matrix,
     * * \f$W_{ci}\f$ is the cell-to-input weight matrix,
     * * \f$b_i\f$ is the input gate bias,
     * * \f$W_{xf}\f$ is the input-to-forget weight matrix,
     * * \f$W_{hf}\f$ is the recurrent-to-forget weight matrix,
     * * \f$W_{cf}\f$ is the cell-to-forget weight matrix,
     * * \f$b_f\f$ is the forget gate bias,
     * * \f$W_{xc}\f$ is the input-to-cell weight matrix,
     * * \f$W_{hc}\f$ is the recurrent-to-cell weight matrix,
     * * \f$b_c\f$ is the cell bias,
     * * \f$W_{xo}\f$ is the input-to-output weight matrix,
     * * \f$W_{ho}\f$ is the recurrent-to-output weight matrix,
     * * \f$W_{co}\f$ is the cell-to-output weight matrix,
     * * \f$b_o\f$ is the output gate bias,
     * * \f$W_{proj}\f$ is the projection weight matrix,
     * * \f$b_{proj}\f$ is the projection bias,
     * * \f$t_{cell}\f$ is the threshold for clipping the cell state, and
     * * \f$t_{proj}\f$ is the threshold for clipping the projected output.
     * * \f$\odot\f$ is the
     *   <a href="https://en.wikipedia.org/wiki/Hadamard_product_(matrices)">
     *   Hadamard product</a> that takes two matrices and produces another
     *   matrix, each element of which is the product of the corresponding
     *   elements of the input matrices.
     *
     * Since API level 29 LSTM supports layer normalization.
     * In case layer normalization is used, the inputs to internal activation
     * functions (sigmoid and \f$g\f$) are normalized, rescaled and recentered
     * following an approach from section 3.1 from
     * https://arxiv.org/pdf/1607.06450.pdf
     *
     * The operation has the following independently optional inputs:
     * * The input-to-input weights (\f$W_{xi}\f$), recurrent-to-input weights
     *   (\f$W_{hi}\f$), cell-to-input (\f$W_{ci}\f$) weights, and input gate
     *   bias (\f$b_i\f$) either all have values, or none of them have values
     *   (i.e., all set to null). If they have no values, coupling of input and
     *   forget gates (CIFG) is used, in which case the input gate (\f$i_t\f$)
     *   is calculated using the following equation instead.
     *   \f{eqnarray*}{
     *   i_t = 1 - f_t
     *   \f}
     * * The cell-to-forget weights (\f$W_{cf}\f$) and cell-to-output weights
     *   (\f$W_{co}\f$) either both have values or neither of them have values.
     *   If they have values, the peephole optimization is used. Additionally,
     *   if CIFG is not used, cell-to-input weights (\f$W_{ci}\f$) is also
     *   required to have values for peephole optimization.
     * * The projection weights (\f$W_{proj}\f$) is required only for the
     *   recurrent projection layer, and should otherwise have no value.
     * * The projection bias (\f$b_{proj}\f$) may (but not required to) have a
     *   value if the recurrent projection layer exists, and should otherwise
     *   have no value.
     * * (API level >= 29) The four layer normalization weights either all have
     *   values or none of them have values. Additionally, if CIFG is used,
     *   input layer normalization weights tensor is omitted and the other layer
     *   normalization weights either all have values or none of them have
     *   values. Layer normalization is used when the values of all the layer
     *   normalization weights are present.
     *
     * References:
     *
     * The default non-peephole non-CIFG implementation is based on:
     * http://www.bioinf.jku.at/publications/older/2604.pdf
     * S. Hochreiter and J. Schmidhuber. "Long Short-Term Memory". Neural
     * Computation, 9(8):1735-1780, 1997.
     *
     * The peephole implementation and projection layer is based on:
     * https://research.google.com/pubs/archive/43905.pdf
     * Hasim Sak, Andrew Senior, and Francoise Beaufays. "Long short-term memory
     * recurrent neural network architectures for large scale acoustic
     * modeling." INTERSPEECH, 2014.
     * (However, the concept of peephole optimization was introduced in work
     * prior to this paper.)
     *
     * The coupling of input and forget gate (CIFG) is based on:
     * http://arxiv.org/pdf/1503.04069.pdf
     * Greff et al. "LSTM: A Search Space Odyssey"
     *
     * The layer normalization is based on:
     * https://arxiv.org/pdf/1607.06450.pdf
     * Jimmy Ba et al. "Layer Normalization"
     *
     * Supported tensor {@link OperandCode}:
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT16} (since API level 29)
     * * {@link ANEURALNETWORKS_TENSOR_FLOAT32}
     *
     * All input and output tensors must be of the same type.
     *
     * Inputs:
     * * 0: The input (\f$x_t\f$).
     *      A 2-D tensor of shape [batch_size, input_size], where “batch_size”
     *      corresponds to the batching dimension, and “input_size” is the size
     *      of the input.
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
     *      Until API level 29 this scalar must be of type {@link
     *      ANEURALNETWORKS_FLOAT32}. Since API level 29, if all the input
     *      tensors have type {@link ANEURALNETWORKS_TENSOR_FLOAT32}, this
     *      scalar must be of the type {@link ANEURALNETWORKS_FLOAT32},
     *      otherwise if all the input tensors have the type {@link
     *      ANEURALNETWORKS_TENSOR_FLOAT16}, this scalar must be of type {@link
     *      ANEURALNETWORKS_FLOAT16}.
     * * 22:The clipping threshold (\f$t_{proj}\f$) for the output from the
     *      projection layer, such that values are bound within
     *      [-proj_clip, proj_clip]. If set to 0.0 then clipping is disabled.
     *      Until API level 29 this scalar must be of type {@link
     *      ANEURALNETWORKS_FLOAT32}. Since API level 29, if all the input
     *      tensors have type {@link ANEURALNETWORKS_TENSOR_FLOAT32}, this
     *      scalar must be of the type {@link ANEURALNETWORKS_FLOAT32},
     *      otherwise if all the input tensors have the type {@link
     *      ANEURALNETWORKS_TENSOR_FLOAT16}, this scalar must be of type {@link
     *      ANEURALNETWORKS_FLOAT16}.
     * Since API level 29 there are additional inputs to this op:
     * * 23:The input layer normalization weights.
     *      A 1-D tensor of shape [num_units]. Used to rescale normalized inputs
     *      to activation at input gate.
     * * 24:The forget layer normalization weights.
     *      A 1-D tensor of shape [num_units]. Used to rescale normalized inputs
     *      to activation at forget gate.
     * * 25:The cell layer normalization weights.
     *      A 1-D tensor of shape [num_units]. Used to rescale normalized inputs
     *      to activation at cell gate.
     * * 26:The output layer normalization weights.
     *      A 1-D tensor of shape [num_units]. Used to rescale normalized inputs
     *      to activation at output gate.
     *
     * Outputs:
     * * 0: The scratch buffer.
     *      A 2-D tensor of shape [batch_size, num_units * 3] with CIFG, or
     *      [batch_size, num_units * 4] without CIFG.
     * * 1: The output state (out) (\f$h_t\f$).
     *      A 2-D tensor of shape [batch_size, output_size].
     * * 2: The cell state (out) (\f$C_t\f$).
     *      A 2-D tensor of shape [batch_size, num_units].
     * * 3: The output (\f$o_t\f$).
     *      A 2-D tensor of shape [batch_size, output_size]. This is effectively
     *      the same as the current “output state (out)” value.
     *
     * Available since API level 27.
     */

#define OP_SPEC_NAME LstmUnit
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
    weight_norm_input, \
    weight_norm_forget,\
    weight_norm_cell, \
    weight_norm_out)
#define ARGC BOOST_PP_TUPLE_SIZE(ARG_NAMES)

#define BOOST_PP_LOCAL_MACRO(n) OP_SPEC_ARG(BOOST_PP_TUPLE_ELEM(ARGC, n, ARG_NAMES))
#define BOOST_PP_LOCAL_LIMITS (0, ARGC)
#include BOOST_PP_LOCAL_ITERATE()
OP_SPEC_END()

// Layer normalization supported since API 29

// order of argument is important
// order of argument is important
MAKE_SPEC(unidirectional_sequence_lstm)
    .input_(nnrt::OperandType::TENSOR_FLOAT32)
    .weight_i2i_(nnrt::OperandType::TENSOR_FLOAT32)
    .weight_i2f_(nnrt::OperandType::TENSOR_FLOAT32)
    .weight_i2c_(nnrt::OperandType::TENSOR_FLOAT32)
    .weight_i2o_(nnrt::OperandType::TENSOR_FLOAT32)
    .weight_r2i_(nnrt::OperandType::TENSOR_FLOAT32, OPTIONAL)
    .weight_r2f_(nnrt::OperandType::TENSOR_FLOAT32)
    .weight_r2c_(nnrt::OperandType::TENSOR_FLOAT32)
    .weight_r2o_(nnrt::OperandType::TENSOR_FLOAT32)
    .weight_c2i_(nnrt::OperandType::TENSOR_FLOAT32, OPTIONAL)
    .weight_c2f_(nnrt::OperandType::TENSOR_FLOAT32, OPTIONAL)
    .weight_c2o_(nnrt::OperandType::TENSOR_FLOAT32, OPTIONAL)
    .bias_i_(nnrt::OperandType::TENSOR_FLOAT32, OPTIONAL)
    .bias_f_(nnrt::OperandType::TENSOR_FLOAT32)
    .bias_c_(nnrt::OperandType::TENSOR_FLOAT32)
    .bias_o_(nnrt::OperandType::TENSOR_FLOAT32)
    .weight_proj_(nnrt::OperandType::TENSOR_FLOAT32, OPTIONAL)
    .bias_proj_(nnrt::OperandType::TENSOR_FLOAT32, OPTIONAL)
    .h_state_(nnrt::OperandType::TENSOR_FLOAT32)
    .c_state_(nnrt::OperandType::TENSOR_FLOAT32)
    .activation_(nnrt::OperandType::INT32)
    .cell_clip_(nnrt::OperandType::FLOAT32)
    .proj_clip_(nnrt::OperandType::FLOAT32)
    .weight_norm_input_(nnrt::OperandType::TENSOR_FLOAT32, OPTIONAL)
    .weight_norm_forget_(nnrt::OperandType::TENSOR_FLOAT32, OPTIONAL)
    .weight_norm_cell_(nnrt::OperandType::TENSOR_FLOAT32, OPTIONAL)
    .weight_norm_out_(nnrt::OperandType::TENSOR_FLOAT32, OPTIONAL)
    );
    // set Parameter added in API level 29 as OPTINAL to support old-fasion usage

MAKE_SPEC(unidirectional_sequence_lstm_FP16)
    .input_(nnrt::OperandType::TENSOR_FLOAT16)
    .weight_i2i_(nnrt::OperandType::TENSOR_FLOAT16)
    .weight_i2f_(nnrt::OperandType::TENSOR_FLOAT16)
    .weight_i2c_(nnrt::OperandType::TENSOR_FLOAT16)
    .weight_i2o_(nnrt::OperandType::TENSOR_FLOAT16)
    .weight_r2i_(nnrt::OperandType::TENSOR_FLOAT16, OPTIONAL)
    .weight_r2f_(nnrt::OperandType::TENSOR_FLOAT16)
    .weight_r2c_(nnrt::OperandType::TENSOR_FLOAT16)
    .weight_r2o_(nnrt::OperandType::TENSOR_FLOAT16)
    .weight_c2i_(nnrt::OperandType::TENSOR_FLOAT16, OPTIONAL)
    .weight_c2f_(nnrt::OperandType::TENSOR_FLOAT16, OPTIONAL)
    .weight_c2o_(nnrt::OperandType::TENSOR_FLOAT16, OPTIONAL)
    .bias_i_(nnrt::OperandType::TENSOR_FLOAT16, OPTIONAL)
    .bias_f_(nnrt::OperandType::TENSOR_FLOAT16)
    .bias_c_(nnrt::OperandType::TENSOR_FLOAT16)
    .bias_o_(nnrt::OperandType::TENSOR_FLOAT16)
    .weight_proj_(nnrt::OperandType::TENSOR_FLOAT16, OPTIONAL)
    .bias_proj_(nnrt::OperandType::TENSOR_FLOAT16, OPTIONAL)
    .h_state_(nnrt::OperandType::TENSOR_FLOAT16)
    .c_state_(nnrt::OperandType::TENSOR_FLOAT16)
    .activation_(nnrt::OperandType::INT32)
    .cell_clip_(nnrt::OperandType::FLOAT16)
    .proj_clip_(nnrt::OperandType::FLOAT16)
    .weight_norm_input_(nnrt::OperandType::TENSOR_FLOAT16)
    .weight_norm_forget_(nnrt::OperandType::TENSOR_FLOAT16)
    .weight_norm_cell_(nnrt::OperandType::TENSOR_FLOAT16)
    .weight_norm_out_(nnrt::OperandType::TENSOR_FLOAT16)
    );

#undef ARG_NAMES
#undef ARGC
#undef OP_SPEC_NAME

#endif
