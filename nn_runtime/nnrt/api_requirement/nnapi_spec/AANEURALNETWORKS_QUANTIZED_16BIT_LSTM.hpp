/****************************************************************************
*
*    Copyright (c) 2020 Vivante Corporation
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

#ifndef __ANEURALNETWORKS_QUANTIZED_16BIT_LSTM_HPP__
#define __ANEURALNETWORKS_QUANTIZED_16BIT_LSTM_HPP__


#define OP_SPEC_NAME Quantized16BitLstmOperation
OP_SPEC_BEGIN()
#define ARG_NAMES           \
    (input,                 \
    weight_i2i,             \
    weight_i2f,             \
    weight_i2c,             \
    weight_i2o,             \
    weight_r2i,             \
    weight_r2f,             \
    weight_r2c,             \
    weight_r2o,             \
    bias_i,                 \
    bias_f,                 \
    bias_c,                 \
    bias_o,                 \
    cell_state,             \
    output_cell)
#define ARGC BOOST_PP_TUPLE_SIZE(ARG_NAMES)

#define BOOST_PP_LOCAL_MACRO(n) OP_SPEC_ARG(BOOST_PP_TUPLE_ELEM(ARGC, n, ARG_NAMES))
#define BOOST_PP_LOCAL_LIMITS (0, ARGC)
#include BOOST_PP_LOCAL_ITERATE()
OP_SPEC_END()

// order of argument is important
MAKE_SPEC(quantized_16bit_lstm)
    .input_(nnrt::OperandType::TENSOR_QUANT8_ASYMM)
    .weight_i2i_(nnrt::OperandType::TENSOR_QUANT8_ASYMM)
    .weight_i2f_(nnrt::OperandType::TENSOR_QUANT8_ASYMM)
    .weight_i2c_(nnrt::OperandType::TENSOR_QUANT8_ASYMM)
    .weight_i2o_(nnrt::OperandType::TENSOR_QUANT8_ASYMM)
    .weight_r2i_(nnrt::OperandType::TENSOR_QUANT8_ASYMM)
    .weight_r2f_(nnrt::OperandType::TENSOR_QUANT8_ASYMM)
    .weight_r2c_(nnrt::OperandType::TENSOR_QUANT8_ASYMM)
    .weight_r2o_(nnrt::OperandType::TENSOR_QUANT8_ASYMM)
    .bias_i_(nnrt::OperandType::TENSOR_INT32)
    .bias_f_(nnrt::OperandType::TENSOR_INT32)
    .bias_c_(nnrt::OperandType::TENSOR_INT32)
    .bias_o_(nnrt::OperandType::TENSOR_INT32)
    .cell_state_(nnrt::OperandType::TENSOR_QUANT16_SYMM)
    .output_cell_(nnrt::OperandType::TENSOR_QUANT8_ASYMM));

#undef ARG_NAMES
#undef ARGC
#undef OP_SPEC_NAME

#endif
