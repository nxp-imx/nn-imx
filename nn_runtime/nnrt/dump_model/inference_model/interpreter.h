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

#pragma once

#include "json.h"
#include "nnrt/model.hpp"
#include "nnrt/compilation.hpp"
#include "nnrt/execution.hpp"
#include "nnrt/op/public.hpp"
#include <assert.h>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <algorithm>
#include <string>

/*it is a interpreter, which interprete the json file dumped from nnrt*/
class Interpreter{
public:
    explicit Interpreter(std::string modelFile,
        std::string dataFile):
        rtModelPath(modelFile), modelDataPath(dataFile)
    {
    };

    ~Interpreter(){}

    int interpreterRT();
    int run();

private:
    int readData(const std::string dataFile);

    void addOprandFromJsonFile(Json::Value &operands);

    void addOperationFromJsonFile(Json::Value &operations);

    void getModelIO(Json::Value &modelIn, Json::Value &modelOut);

    void setModelIO();

    void  setInputValue();

    void setOutputBuffer();

private:
    std::vector<std::vector<uint8_t>> mInputData;
    std::vector<std::vector<uint8_t>> mOutputData;
    std::vector<std::vector<uint8_t>> mGoldenData;

    /* all of model data */
    std::vector <uint8_t> mData;
    Json::Value jModel;

    /*the indexes of input and output*/
    std::vector <uint32_t> mInputs;
    std::vector <uint32_t> mOutputs;

    nnrt::Model *mModel;
    nnrt::Compilation *mCompilator;
    nnrt::Execution *mExecution;

    std::string rtModelPath;
    std::string modelDataPath;
};

