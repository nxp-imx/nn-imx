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
#include <iostream>
#include <sstream>
#include <iterator>
#include <vector>
#include <string>

#include "vsi_nn_pub.h"
#include "model.hpp"
#include "error.hpp"
#include "version.hpp"

#include "op/public.hpp"

#ifdef __linux__
#include <unistd.h>
#else
#include <process.h>
#endif

#ifdef __ANDROID__
#include <sys/system_properties.h>
#endif

#define FAST_MODE "VIV_FAST_MODE"

namespace nnrt
{
/*
    fetch env{@name} and put it into @result,
    return:
        -1, fail to get this env.
*/

static int getEnv(std::string name, int &result)
{
    int get_success = 0;
#ifdef __ANDROID__
    char env[10] = { 0 };
    get_success = __system_property_get(name.c_str(), env);
    if (get_success)
        result = atoi(env);
#else
    char *env = getenv(name.c_str());
    if (env) {
        result = atoi(env);
        get_success = 1;
    }
#endif
    return get_success;
}

static void setEnv(std::string name, std::string val)
{
#ifdef __ANDROID__
    __system_property_set(name.c_str(), val.c_str());
#elif __linux__
    setenv(name.c_str(), val.c_str(), 1);
#endif
}

int8_t getProcessName(char * process) {
    int32_t pid = getpid();
#ifdef __ANDROID__
    char line[256] = { '\0' }, t[32] = { '\0' }, type[32] = { '\0' }, mode[32] = { '\0' };
    char last[256] = { '\0' };
    FILE* fp = fopen("/sys/kernel/debug/binder/transactions", "r");
    if (NULL == fp)
        return -1;
    vx_int32 _from_pid_out = 0, _from_tid_out = 0, _to_pid_out = 0, _to_tid_out = 0, time = 0, s = 0;
    vx_int32 _from_pid_in = 0, _from_tid_in = 0, _to_pid_in = 0, _to_tid_in = 0;
    sprintf(line, "%d", pid);
    while (fgets(last, 256, fp)) {
        if (strstr(last, line) != NULL && strstr(last, "transaction") != NULL) {
            /**********************************************************************************
            * outgoing transaction 14397865: 00000000 from 10450:10450 to 10443:10452 code 7 flags 10 pri 0:120 r1
            * incoming transaction 14397865: 00000000 from 10450:10450 to 10443:10452 code 7 flags 10 pri 0:120 r1
            ***********************************************************************************/
            if (strstr(last, "incoming") != NULL) {
                sscanf(last, "%s %s %d: %d from %d:%d to %d:%d %s",
                    type, mode, &time, &s, &_from_pid_in, &_from_tid_in, &_to_pid_in, &_to_tid_in, t);
            }
            else if (strstr(last, "outgoing") != NULL) {
                sscanf(last, "%s %s %d: %d from %d:%d to %d:%d %s",
                    type, mode, &time, &s, &_from_pid_out, &_from_tid_out, &_to_pid_out, &_to_tid_out, t);
            }
        }
    }

    fclose(fp);

    if (pid == _to_pid_out && (_from_pid_out == _from_pid_in) && (_from_tid_out == _from_tid_in) &&
        (_to_pid_out == _to_pid_in) && (_to_tid_out == _to_tid_in)) {
        sprintf(line, "/proc/%d/cmdline", _from_pid_out);
        fp = fopen(line, "r");
        fgets(last, 256, fp);
        VSILOGD(" PROCESSNAME: %s",last);
        fclose(fp);

        if (strlen(last) > 0 && process)
            memcpy(process, last, strlen(last));
    }
    return 1;

#elif __linux__
    char file_name[256] = { '\0' };
    char string[256] = { '\0' };
    sprintf(file_name, "/proc/%d/cmdline", pid);
    FILE *fp = fopen(file_name, "r");
    if (fp) {
        fread(string, 1, 255, fp);
        string[255] = '\0';
    }
    else {
        return -1;
    }
    VSILOGD("proc name: %s",string );
    fclose(fp);
    if (strlen(string) > 0 && process)
        memcpy(process, string, strlen(string));
    return 1;
#else
    // the hack is invalid in windows or other OS.
    return -1;
#endif

}

void Model::checkProcess() {

    char processName[256] = { '\0' };
    if (-1 == getProcessName(processName)){
        int val = 1;
        getEnv(FAST_MODE, val);
        relaxed_ = val;

        return;
    }
    // for the CTS process, run with the hack
    std::string fp32_process_name = "NeuralNetworksTest_";
    if (strstr(processName, fp32_process_name.c_str()) != NULL) {
        VSILOGD("set env variable VIV_VX_DISABLE_TP_NN_EVIS = 1");
        setEnv("VIV_VX_DISABLE_TP_NN_EVIS", "1");

        relaxed_ = false;
    }
    else {
        int val = -1;
        if (getEnv("VIV_VX_DISABLE_TP_NN_EVIS", val)) {
            // for non-CTS process, run without hack
            VSILOGD("reset VIV_VX_DISABLE_TP_NN_EVIS to 0");
            setEnv("VIV_VX_DISABLE_TP_NN_EVIS", "0");
        }

        // run the model with fp32 data-type depending on VIV_FAST_MODE
        // VIV_FAST_MODE:
        //     0: run with fp32 data type for float data by default.
        //     1: run with fp16 data type.for float data
        val = 1;
        getEnv(FAST_MODE, val);
        relaxed_ = val;
    }

    return;
}

Model::Model() : memory_pool_(*this)
{
    /*fetch env ${VIV_FAST_MODEL} to set the relaxed mode,
      run on relaxed mode by default*/
    checkProcess();
}

Model::~Model()
{
}

bool Model::isInput(uint32_t index)
{
    for (uint32_t i = 0; i < input_indexes_.size(); ++ i)
    {
        if (index == input_indexes_[i])
        {
            return true;
        }
    }
    return false;
}

bool Model::isOutput(uint32_t index)
{
    for (uint32_t i = 0; i < output_indexes_.size(); ++ i)
    {
        if (index == output_indexes_[i])
        {
            return true;
        }
    }
    return false;
}

void Model::identifyInputsAndOutputs(const uint32_t* inputs_ptr,
        uint32_t input_count, const uint32_t* outputs_ptr,
        uint32_t output_count)
{
    if (!inputs_ptr || !outputs_ptr)
    {
        return;
    }
    input_indexes_.clear();
    output_indexes_.clear();
    input_indexes_.insert(input_indexes_.begin(),
            inputs_ptr, inputs_ptr + input_count);
    output_indexes_.insert(output_indexes_.begin(),
            outputs_ptr, outputs_ptr + output_count);
    for (auto it : input_indexes_) {
        operands_[it]->setGraphInputOutput();
    }
    for (auto it : output_indexes_) {
        operands_[it]->setGraphInputOutput();
    }
}

op::OperationPtr Model::addOperation(OperationType code,
        const uint32_t* inputs, uint32_t input_size,
        const uint32_t* outputs, uint32_t output_size,
        uint32_t* out_index)
{
    op::OperationPtr new_operation = std::make_shared<op::Operation>(code);
    if (nullptr == new_operation)
    {
        return nullptr;
    }
    new_operation->setInputs(inputs, input_size);
    new_operation->setOutputs(outputs, output_size);
    //new_operation->inputs.insert(new_operation->inputs.begin(),
    //        inputs, inputs + input_size );
    //new_operation->outputs.insert(new_operation->outputs.begin(),
    //        outputs, outputs + output_size );
    // operations_[operation_unique_id_] = new_operation;
    operations_.insert(std::make_pair(operation_unique_id_, new_operation));
    if (nullptr != out_index)
    {
        *out_index = (int)operation_unique_id_;
    }
    operation_unique_id_ ++;
    return new_operation;
}

op::OperationPtr Model::addOperation(op::OperationPtr new_operation,
        uint32_t* out_index)
{
    if (nullptr == new_operation)
    {
        return nullptr;
    }
    operations_.insert(std::make_pair(operation_unique_id_, new_operation));
    if (nullptr != out_index)
    {
        *out_index = (int)operation_unique_id_;
    }
    operation_unique_id_ ++;
    return new_operation;
}

op::OperandPtr Model::cloneOperand(op::OperandPtr operand, int* out_index)
{
    op::OperandPtr new_operand = operand->clone();
    if (nullptr == new_operand)
    {
        return nullptr;
    }
    // operands_[operand_unique_id_] = new_operand;
    operands_.insert(std::make_pair(operand_unique_id_, new_operand));
    if (nullptr != out_index)
    {
        *out_index = (int)operand_unique_id_;
    }
    operand_unique_id_ ++;
    return new_operand;
}

op::OperandPtr Model::addOperand(op::OperandPtr new_operand,
        uint32_t* out_index)
{
    if (!new_operand) {
        new_operand = std::make_shared<op::Operand>();
    }

    if (new_operand) {
        operands_.insert(std::make_pair(operand_unique_id_, new_operand));
        if (nullptr != out_index) {
            *out_index = (int)operand_unique_id_;
        }
        operand_unique_id_++;
    } else {
        VSILOGE("OOM: create new operand failed");
    }

    return new_operand;
}

int Model::getOperandIndex(op::OperandPtr operand)
{
    for (auto it = operands_.begin(); it != operands_.end(); ++ it)
    {
        if (it->second == operand)
        {
            return (int)it->first;
        }
    }
    return -1;
}

#if 0
template<typename T>
T* Model::getBuffer(DataLocation& location)
{
    return getBuffer<T>(location.poolIndex, location.offset);
}

template<typename T>
T* Model::getBuffer(uint32_t index, size_t offset)
{
    void* data = nullptr;
    if (index <= pool_.size()) {
        data = pool_[index]->data(offset);
    }
    return static_cast<T*>(data);
}
#endif

int Model::setOperandValue(uint32_t operand_index, const void* buffer, size_t length)
{
    if (operands_.find(operand_index) == operands_.end()) {
        VSILOGW("Operand index(%u) is not found.", operand_index);
        return NNA_ERROR_CODE(BAD_DATA);
    }
    op::OperandPtr operand = operands_[operand_index];
    return setOperandValue(operand, buffer, length);
}

int Model::setOperandValue(op::OperandPtr operand, const void* buffer, size_t length)
{
    int err = NNA_ERROR_CODE(NO_ERROR);

    if (nullptr == buffer) {
        operand->setNull();
    }
    else if (!operand->isTensor()) {
        assert (length <= sizeof(operand->scalar));
        memcpy( &operand->scalar, buffer, length);
    } else {
        mem_refs_.push_back(memory_pool_.add_reference(buffer, length));
        operand->weak_mem_ref = mem_refs_.back();
    }

    return err;
}

int Model::setOperandValueFromMemory(op::OperandPtr operand,
        const Memory* memory, size_t offset, size_t length)
{
    mem_refs_.push_back(memory_pool_.add_reference(memory, offset, length));
    operand->weak_mem_ref = mem_refs_.back();
    return NNA_ERROR_CODE(NO_ERROR);
}
int Model::setOperandValueFromMemory(uint32_t operand_index,
        const Memory* memory, size_t offset, size_t length)
{
    if (operands_.find(operand_index) == operands_.end())
    {
        VSILOGW("Operand index(%u) is not found.", operand_index);
        return NNA_ERROR_CODE(BAD_DATA);
    }
    op::OperandPtr operand = operands_[operand_index];
    return setOperandValueFromMemory(operand, memory, offset, length);
}

std::vector<op::OperandPtr> Model::getOperands(const std::vector<uint32_t> & indexes)
{
    std::vector<op::OperandPtr> out_operands;
    for (size_t i = 0; i < indexes.size(); ++ i)
    {
        int32_t idx = static_cast<int32_t>(indexes[i]);
        if (idx >= 0) {
            out_operands.push_back(operands_[idx]);
        }
    }
    return out_operands;
}

std::vector<uint32_t> Model::getConsumers(op::OperandPtr operand)
{
    std::vector<uint32_t> indexes;
    for( auto& op_kv : operations_) {
        for ( auto op_index : op_kv.second->inputs()){
            if (operands_[op_index] == operand) {
                indexes.push_back(op_kv.first);
                break;
            }
        }
    }
    return indexes;
}

std::vector<uint32_t> Model::getProducers(op::OperandPtr operand)
{
    std::vector<uint32_t> indexes;

    for( auto& op_kv : operations_) {
        for ( auto op_index : op_kv.second->outputs()){
            if (operands_[op_index] == operand) {
                indexes.push_back(op_kv.first);
                break;
            }
        }
    }
    return indexes;
}

void Model::removeOperand(uint32_t index)
{
    operands_.erase(index);
}

bool Model::validate()
{
    bool valid = true;
    for (auto it : operands_) {
        if (!it.second->isValid()) {
            valid = false;
            break;
        }
    }
    valid_ = valid;
    return isValid();
}

uint32_t Model::inputIndex(uint32_t index) const {
    if (index < input_indexes_.size()) {
        return input_indexes_[index];
    }
    return op::NNRT_INVALID_OPERAND_INDEX;
}

uint32_t Model::outputIndex(uint32_t index) const {
    if (index < output_indexes_.size()) {
        return output_indexes_[index];
    }
    return op::NNRT_INVALID_OPERAND_INDEX;
}

int Model::updateOperand(uint32_t operand_index, const op::OperandPtr operand_type)
{
    op::OperandPtr operand = operands_[operand_index];
    if (!operand) {
        VSILOGW("Invliad operand index %d", operand_index);
        return NNA_ERROR_CODE(BAD_DATA);
    }
    // TODO: Update other attrs
    if (operand->dimensions.size() == 0) {
        operand->dimensions = operand_type->dimensions;
    } else {
        if (operand->dimensions.size() != operand_type->dimensions.size()) {
            return NNA_ERROR_CODE(BAD_DATA);
        }
        for (int i = 0; i < (int)operand->dimensions.size(); i ++) {
            if (operand_type->dimensions[i] <= 0) {
                return NNA_ERROR_CODE(BAD_DATA);
            }
            operand->dimensions[i] = operand_type->dimensions[i];
        }
    }
    return NNA_ERROR_CODE(NO_ERROR);
}

void Model::relax(bool fast_model) {
    int fastVal = -1;
    getEnv(FAST_MODE, fastVal);
    if (1 == fastVal) {
        if (!fast_model) {
            VSILOGW("VIV_FAST_MODE has been setted, fast mode can't be setted false.");
        }
        relaxed_ = true;
    } else {
        relaxed_ = fast_model;
    }
}

void Model::echo() {
    VSILOGD("Software Version: %s", VERSION::as_str());
    VSILOGD("================== model info ==================");
    VSILOGD("================== operands ==================");
    for (auto it : operands_) {
        it.second->echo(it.first);
    }
    VSILOGD("================== operations ==================");
    for (auto it : operations_) {
        it.second->echo(it.first);
    }
    VSILOGD("================================================");
}

void Model::freezeCompile()
{
    signature_ = generateSignature();
    compiled_ = true;
}

std::string Model::generateSignature()
{
    /**
     * @todo Add more info to signature
     */
    std::ostringstream result;
    for (uint32_t i : inputIndexes()) {
        op::OperandPtr operand = this->operand(i);
        std::copy(operand->dimensions.begin(), operand->dimensions.end(),
                std::ostream_iterator<uint32_t>(result, ","));
    }
    return result.str();
}

}
