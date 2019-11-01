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

#ifndef __FAKE_BIAS_TYPE_SELECTOR__
#define __FAKE_BIAS_TYPE_SELECTOR__

namespace armnn{

template <armnn::DataType D>
struct FakeBiasTypeSelector {
    using type = void;
};

template <>
struct FakeBiasTypeSelector<DataType::Float32> {
    using type = float;
    static constexpr armnn::DataType value = DataType::Float32;
};

template <>
struct FakeBiasTypeSelector<DataType::Float16> {
    using type = float;
    static constexpr armnn::DataType value = DataType::Float32;
};

template <>
struct FakeBiasTypeSelector<DataType::QuantisedAsymm8> {
    using type = uint8_t;
    static constexpr armnn::DataType value = DataType::Signed32;
};

}

#endif
