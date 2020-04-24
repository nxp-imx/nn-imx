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

#ifndef _GROUPED_CONV2D_VALIDATE_HPP_
#define _GROUPED_CONV2D_VALIDATE_HPP_

#include "OperationValidate.hpp"

namespace android {
namespace nn {
namespace op_validate {
template <typename T_model, typename T_Operation>
class GroupedConv2DValidate : public OperationValidate<T_model, T_Operation> {
   public:
    GroupedConv2DValidate(const T_model& model, const T_Operation& operation)
        : OperationValidate<T_model, T_Operation>(model, operation) {}
    bool SignatureCheck() override {
        uint32_t groupNumber = 0;
        uint32_t inputChannel = 0;
        uint32_t outputChannel = 0;
        // Default layout = NHWC
        bool layout = false;
        bool support = true;
        auto model = this->ModelForRead();
        auto operation = this->OperationForRead();
        auto input = model.operands[operation.inputs[0]];
        auto output = model.operands[operation.outputs[0]];
        if (12 == operation.inputs.size()) {
            auto& groupNumberOperand = model.operands[operation.inputs[9]];
            auto& layoutOperand = model.operands[operation.inputs[11]];
            groupNumber = get_buffer::getScalarData<uint32_t>(model, groupNumberOperand);
            layout = get_buffer::getScalarData<bool>(model, layoutOperand);
        }
        else if (9 == operation.inputs.size()) {
            auto& groupNumberOperand = model.operands[operation.inputs[6]];
            auto& layoutOperand = model.operands[operation.inputs[8]];
            groupNumber = get_buffer::getScalarData<uint32_t>(model, groupNumberOperand);
            layout = get_buffer::getScalarData<bool>(model, layoutOperand);
        }
        else {
            LOG(ERROR) << "GROUPED_CONV_2D: inputs size mismatched";
            assert(false);
        }
        if (layout) {
            // NCHW
            inputChannel = input.dimensions[1];
            outputChannel = output.dimensions[1];
        } else {
            // NHWC
            inputChannel = input.dimensions[3];
            outputChannel = output.dimensions[3];
        }
        if (groupNumber == inputChannel / outputChannel) {
            support = true;
        } else {
            support = false;
        }

        return support &&
               hal::limitation::nnapi::match("GroupedConv2DInput", this->InputArgTypes()) &&
               hal::limitation::nnapi::match("GroupedConv2DOutput", this->OutputArgTypes());
    };
};

}  // end of op_validate
}
}

#endif