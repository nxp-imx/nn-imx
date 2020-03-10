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
#ifndef __DUMP_JSON_MODEL_HPP__
#define __DUMP_JSON_MODEL_HPP__

/* demo code:
*   class Dump dump(model);
*   dump.getInputsData(execution); // it is optional
*   dump.dump2json();
*/
#include "json.h"
#include "op/public.hpp"
#include "execution.hpp"

namespace nnrt {
    class Dump {
    public:
        explicit Dump(Model *model): model_(model){
            convertRTmodel();
        };
        ~Dump() {};

        void getInputsData(const Execution *exec);

        void dump2json();

    private:
        void convertRTmodel();

        void dumpModelData();

        int dumpOperand(Json::Value &jOperand,
            const op::OperandPtr& operand);

        int dumpOperandValue(uint32_t index,
            Json::Value &jOperands,
            const op::OperandPtr& operand);

        int dumpOperandValue(Json::Value &jOperands,
            void *buffer, size_t length);

        int dumpOpration(Json::Value &jOperation,
            const op::OperationPtr& opration);

    private:
        Json::Value rtModel_;
        std::vector<uint8_t> modelData_;
        const nnrt::Model* model_;
    };
}
#endif
