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

#ifndef __ARM_NN_INTERPRETER_H__
#define __ARM_NN_INTERPRETER_H__

#include <set>
#include <vector>
#include "nnrt/interpreter.hpp"
#include "nnrt/model.hpp"
#include "nnrt/op/public.hpp"

namespace armnn {
class Armnn_Interpreter : public nnrt::Interpreter {
   public:
    Armnn_Interpreter();
    virtual ~Armnn_Interpreter();

    const char* name() override { return "Armnn_Interpreter"; }

    int run(nnrt::Model* model, bool* modified) override;

    nnrt::FusedType mapFusedType(int fused_code);

    nnrt::PadType mapPadType(int code);

    nnrt::FusedType mapLstmActivationType(int value);

    nnrt::LshProjectionType mapLshProjectionType(int value);

    template <typename T>
    std::vector<T> reverseArray(T* data, size_t length) {
        std::vector<T> array(length);
        for (size_t i = 0; i < length; ++i) {
            array[i] = data[length - i - 1];
        }
        return array;
    }

    template <typename T>
    std::vector<T> reverseArray(std::vector<T>& data) {
        std::vector<T> array(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            array[i] = data[data.size() - i - 1];
        }
        return array;
    }

    int32_t reverseMask(int32_t mask, size_t dim_num);

    inline std::vector<int32_t> convertPermute(std::vector<int32_t>& perm) {
        return convertAxes(perm, perm.size());
    }

    inline std::vector<int32_t> convertPermute(int32_t* perm_buffer, size_t length) {
        return convertAxes(perm_buffer, length, length);
    }

    std::vector<int32_t> convertAxes(int32_t* axes_buffer, size_t length, size_t dim_num);

    std::vector<int32_t> convertAxes(std::vector<int32_t>& axes, size_t dim_num);

    inline int32_t convertAxis(int32_t axis, int32_t dim_num) {
        return (dim_num - computeAxis(axis, dim_num) - 1);
    }

    void fillIntArray(nnrt::Model* model,
                      nnrt::op::OperationPtr operation,
                      std::vector<int32_t>& array,
                      int32_t op_index,
                      bool reverse,
                      bool is_axis);

    inline int32_t computeAxis(int32_t axis, int32_t dim_num) {
        if (axis >= 0) {
            return axis;
        } else {
            return dim_num + axis;
        }
    }

    inline void truncateOperationIOs(nnrt::Model* model,
                                     nnrt::op::OperationPtr operation,
                                     int32_t input_num,
                                     int32_t output_num);

    inline void resetFusedType(nnrt::Model* model, nnrt::op::OperationPtr operation, int32_t input_index) {
        nnrt::op::OperandPtr operand = model->operand(operation->input(input_index));
        operation->setFusedType(mapFusedType(operand->scalar.int32));
    }

    void replaceOperation(nnrt::Model* model, uint32_t op_index, nnrt::op::OperationPtr new_operation);

#define REGISTER_OP(NAME) \
    nnrt::op::OperationPtr map_##NAME(nnrt::Model* model, nnrt::op::OperationPtr operation, uint32_t)
    REGISTER_OP(ADD);
    REGISTER_OP(CONV_2D);
    REGISTER_OP(DEPTHWISE_CONV_2D);
    REGISTER_OP(RELU);
    REGISTER_OP(RESHAPE);
    REGISTER_OP(FULLY_CONNECTED);
    REGISTER_OP(TRANSPOSE);
    REGISTER_OP(CONCATENATION);
    REGISTER_OP(AVERAGE_POOL_2D);
    REGISTER_OP(SQUEEZE);
    REGISTER_OP(SOFTMAX);
    REGISTER_OP(MAX_POOL_2D);
    REGISTER_OP(PAD);
    REGISTER_OP(MUL);
    REGISTER_OP(MEAN);
    REGISTER_OP(RELU1);
    REGISTER_OP(RELU6);
    REGISTER_OP(ABS);
    REGISTER_OP(SIGMOID);
    REGISTER_OP(TANH);
    REGISTER_OP(LEAKY_RELU);
    REGISTER_OP(SOFT_RELU);
    REGISTER_OP(SQRT);
    REGISTER_OP(SQUARE);
    REGISTER_OP(FLOOR);
    REGISTER_OP(DIV);
    REGISTER_OP(SUB);
    REGISTER_OP(DEQUANTIZE);
    REGISTER_OP(SPACE_TO_DEPTH);
    REGISTER_OP(DEPTH_TO_SPACE);
    REGISTER_OP(SPACE_TO_BATCH_ND);
    REGISTER_OP(BATCH_TO_SPACE_ND);
    REGISTER_OP(L2_NORMALIZATION);
    REGISTER_OP(RESIZE_BILINEAR);
    REGISTER_OP(LOCAL_RESPONSE_NORMALIZATION);
    REGISTER_OP(EMBEDDING_LOOKUP);
    REGISTER_OP(RNN);
    REGISTER_OP(HASHTABLE_LOOKUP);
    REGISTER_OP(LSTM);
    REGISTER_OP(SVDF);
    REGISTER_OP(LSH_PROJECTION);
    REGISTER_OP(L2_POOL_2D);
    REGISTER_OP(STRIDED_SLICE);
    REGISTER_OP(BATCH_NORM);
    REGISTER_OP(MAXIMUM);
    REGISTER_OP(MINIMUM);
    REGISTER_OP(RSQRT);
    REGISTER_OP(PRELU);
    REGISTER_OP(DECONV_2D);
    REGISTER_OP(DATA_CONVERT);
#undef REGISTER_OP
   protected:
    // TODO: Add a parent interpreter class and move this function to it.
    std::vector<uint32_t> reorderOperands(std::vector<uint32_t>& operands, std::vector<int> order);

   private:
    typedef nnrt::op::OperationPtr (Armnn_Interpreter::*AddNodeFunc)(nnrt::Model*,
                                                                    nnrt::op::OperationPtr,
                                                                    uint32_t);
    std::map<nnrt::OperationType, AddNodeFunc> op_container_;
    std::set<uint32_t> operands_to_remove_;
};
};
#endif
