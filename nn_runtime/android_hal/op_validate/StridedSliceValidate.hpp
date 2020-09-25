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

#ifndef _STRIDED_SLICE_VALIDATE_HPP_
#define _STRIDED_SLICE_VALIDATE_HPP_

#include "OperationValidate.hpp"

namespace android {
namespace nn {
namespace op_validate {
template <typename T_model, typename T_Operation>
class StridedSliceValidate : public OperationValidate<T_model, T_Operation> {
   public:
    StridedSliceValidate(const T_model& model, const T_Operation& operation)
        : OperationValidate<T_model, T_Operation>(model, operation){};

    bool SignatureCheck(std::string& reason) override {
        if (::hal::limitation::nnapi::match("StridedSlice_Inputs", this->InputArgTypes()) &&
            ::hal::limitation::nnapi::match("StridedSlice_Outputs", this->OutputArgTypes())) {
            bool is_support = true;
            for (auto input_idx = 1; input_idx < this->m_Operation.inputs.size(); ++input_idx) {
                if (this->IsInput(input_idx)) {
                    is_support = false;
                    reason +=
                        "StridedSlice: not supported because only input(0) can be model_input";
                    break;
                }
            }
            auto operation = this->OperationForRead();
            auto model = this->ModelForRead();
            vsi_driver::VsiRTInfo vsiMemory;
            auto stride_op = model.operands[operation.inputs[3]];
            auto ptr = (int32_t *)get_buffer::getOperandDataPtr(model, stride_op, vsiMemory);
            for( auto i  = 0; i < stride_op.dimensions.size(); i++){
                if(ptr[i] < 0){
                    is_support = false;
                    reason +=
                    "StridedSilce: not supported stride parameter less than 0";
                    break;
                }
            }
            return is_support;
        }
        else {
            reason += "StridedSlice: signature matching failed";
            return false;
        }
    }
};

}  // namespace op_validate
}  // namespace nn
}  // namespace android

#endif
