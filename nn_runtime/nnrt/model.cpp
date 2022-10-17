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
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>
/* Include string.h for strlen() */
#include <string.h>

#include "nnrt/error.hpp"
#include "nnrt/logging.hpp"
#include "nnrt/model.hpp"
#include "nnrt/op/public.hpp"
#include "nnrt/version.hpp"
#include "nnrt/file_map_memory.hpp"

#ifdef __linux__
#include <unistd.h>
#else
#include <process.h>
#endif

#define FAST_MODE "VIV_FAST_MODE"

namespace nnrt {

Model::Model() : memory_pool_(*this) {
    /*fetch env ${VIV_FAST_MODEL} to set the relaxed mode,
      run on relaxed mode by default*/
    int val = 1;
    if (OS::getEnv(FAST_MODE, val)) {
        NNRT_LOGD_PRINT("%s = %d", FAST_MODE, val);
        relaxed_ = val;
    }
}

Model::~Model() {
#ifdef __linux__
    if (cache_memory_ != nullptr) munmap(cache_memory_, cache_size_);
    if (cache_handle_ != -1) {
        close(cache_handle_);
        cache_handle_ = -1;
    }
#endif
}

bool Model::isInput(uint32_t index) {
    for (uint32_t i = 0; i < input_indexes_.size(); ++i) {
        if (index == input_indexes_[i]) {
            return true;
        }
    }
    return false;
}

bool Model::isOutput(uint32_t index) {
    for (uint32_t i = 0; i < output_indexes_.size(); ++i) {
        if (index == output_indexes_[i]) {
            return true;
        }
    }
    return false;
}

void Model::identifyInputsAndOutputs(const uint32_t* inputs_ptr,
                                     uint32_t input_count,
                                     const uint32_t* outputs_ptr,
                                     uint32_t output_count) {
    if (!inputs_ptr || !outputs_ptr) {
        return;
    }
    input_indexes_.clear();
    output_indexes_.clear();
    input_indexes_.insert(input_indexes_.begin(), inputs_ptr, inputs_ptr + input_count);
    output_indexes_.insert(output_indexes_.begin(), outputs_ptr, outputs_ptr + output_count);
    for (auto it : input_indexes_) {
        operands_[it]->setGraphInputOutput();
    }
    for (auto it : output_indexes_) {
        operands_[it]->setGraphInputOutput();
    }
}

op::OperationPtr Model::addOperation(OperationType code,
                                     const uint32_t* inputs,
                                     uint32_t input_size,
                                     const uint32_t* outputs,
                                     uint32_t output_size,
                                     uint32_t* out_index) {
    op::OperationPtr new_operation = std::make_shared<op::Operation>(code);
    if (nullptr == new_operation) {
        return nullptr;
    }
    new_operation->setInputs(inputs, input_size);
    new_operation->setOutputs(outputs, output_size);
    // new_operation->inputs.insert(new_operation->inputs.begin(),
    //        inputs, inputs + input_size );
    // new_operation->outputs.insert(new_operation->outputs.begin(),
    //        outputs, outputs + output_size );
    // operations_[operation_unique_id_] = new_operation;
    operations_.insert(std::make_pair(operation_unique_id_, new_operation));
    if (nullptr != out_index) {
        *out_index = (int)operation_unique_id_;
    }
    operation_unique_id_++;
    return new_operation;
}

op::OperationPtr Model::addOperation(op::OperationPtr new_operation, uint32_t* out_index) {
    if (nullptr == new_operation) {
        return nullptr;
    }
    operations_.insert(std::make_pair(operation_unique_id_, new_operation));
    if (nullptr != out_index) {
        *out_index = (int)operation_unique_id_;
    }
    operation_unique_id_++;
    return new_operation;
}

op::OperandPtr Model::cloneOperand(op::OperandPtr operand, int* out_index) {
    op::OperandPtr new_operand = operand->clone();
    if (nullptr == new_operand) {
        return nullptr;
    }
    // operands_[operand_unique_id_] = new_operand;
    operands_.insert(std::make_pair(operand_unique_id_, new_operand));
    if (nullptr != out_index) {
        *out_index = (int)operand_unique_id_;
    }
    operand_unique_id_++;
    return new_operand;
}

op::OperandPtr Model::addOperand(op::OperandPtr new_operand, uint32_t* out_index) {
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
        NNRT_LOGE_PRINT("OOM: create new operand failed");
    }

    return new_operand;
}

int Model::getOperandIndex(op::OperandPtr operand) {
    for (auto it = operands_.begin(); it != operands_.end(); ++it) {
        if (it->second == operand) {
            return (int)it->first;
        }
    }
    return -1;
}

void Model::get_index_by_operand(op::IndexByOperand& index_for_op_input,
                                 op::IndexByOperand& index_for_op_output) {
    auto operations = this->operations();
    for (auto it = operations.begin(); it != operations.end(); ++it) {
        op::OperationPtr operation = it->second;
        for (auto index : operation->inputs()) {
            auto it1 = index_for_op_input.find(index);
            if (it1 != index_for_op_input.end()) {
                it1->second.push_back(it->first);
            } else {
                std::vector<uint32_t> operation_ids;
                operation_ids.push_back(it->first);
                index_for_op_input.insert(
                    std::make_pair<uint32_t, std::vector<uint32_t>>(std::move(index),
                                                                    std::move(operation_ids)));
            }
        }
        for (auto index : operation->outputs()) {
            auto it1 = index_for_op_output.find(index);
            if (it1 != index_for_op_output.end()) {
                it1->second.push_back(it->first);
            } else {
                std::vector<uint32_t> operation_ids;
                operation_ids.push_back(it->first);
                index_for_op_output.insert(
                    std::make_pair<uint32_t, std::vector<uint32_t>>(std::move(index),
                                                                    std::move(operation_ids)));
            }
        }
    }
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

int Model::setOperandValue(uint32_t operand_index, const void* buffer, size_t length) {
    if (operands_.find(operand_index) == operands_.end()) {
        NNRT_LOGW_PRINT("Operand index(%u) is not found.", operand_index);
        return NNA_ERROR_CODE(BAD_DATA);
    }
    op::OperandPtr operand = operands_[operand_index];
    return setOperandValue(operand, buffer, length);
}

int Model::setOperandValue(op::OperandPtr operand, const void* buffer, size_t length) {
    int err = NNA_ERROR_CODE(NO_ERROR);

    if (nullptr == buffer) {
        operand->setNull();
    } else if (!operand->isTensor()) {
        assert(length <= sizeof(operand->scalar));
        memcpy(&operand->scalar, buffer, length);
    } else {
        mem_refs_.push_back(memory_pool_.add_reference(buffer, length));
        operand->weak_mem_ref = mem_refs_.back();
    }

    return err;
}

int Model::setOperandValueFromMemory(op::OperandPtr operand,
                                     const Memory* memory,
                                     size_t offset,
                                     size_t length) {
    if (operand->isTensor()) {
        mem_refs_.push_back(memory_pool_.add_reference(memory, offset, length));
        operand->weak_mem_ref = mem_refs_.back();
    } else {
        assert(length <= sizeof(operand->scalar));
        memcpy(&operand->scalar, memory->data(offset), length);
    }
    return NNA_ERROR_CODE(NO_ERROR);
}
int Model::setOperandValueFromMemory(uint32_t operand_index,
                                     const Memory* memory,
                                     size_t offset,
                                     size_t length) {
    if (operands_.find(operand_index) == operands_.end()) {
        NNRT_LOGW_PRINT("Operand index(%u) is not found.", operand_index);
        return NNA_ERROR_CODE(BAD_DATA);
    }
    op::OperandPtr operand = operands_[operand_index];
    return setOperandValueFromMemory(operand, memory, offset, length);
}

std::vector<op::OperandPtr> Model::getOperands(const std::vector<uint32_t>& indexes) {
    std::vector<op::OperandPtr> out_operands;
    for (size_t i = 0; i < indexes.size(); ++i) {
        int32_t idx = static_cast<int32_t>(indexes[i]);
        if (idx >= 0) {
            out_operands.push_back(operands_[idx]);
        }
    }
    return out_operands;
}

std::vector<uint32_t> Model::getConsumers(const op::OperandPtr& operd) {
    std::vector<uint32_t> indexes;
    for (auto& op_kv : operations_) {
        for (auto op_index : op_kv.second->inputs()) {
            if (operand(op_index) == operd) {
                indexes.push_back(op_kv.first);
                break;
            }
        }
    }
    return indexes;
}

std::vector<uint32_t> Model::getProducers(const op::OperandPtr& operd) {
    std::vector<uint32_t> indexes;

    for (auto& op_kv : operations_) {
        for (auto op_index : op_kv.second->outputs()) {
            if (operand(op_index) == operd) {
                indexes.push_back(op_kv.first);
                break;
            }
        }
    }
    return indexes;
}

void Model::removeOperand(uint32_t index) {
    operands_.erase(index);
}

void Model::removeOperation(uint32_t index) {
    operations_.erase(index);
}

bool Model::validate() {
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

int Model::updateOperand(uint32_t operand_index, const op::OperandPtr operand_type) {
    op::OperandPtr operand = operands_[operand_index];
    if (!operand) {
        NNRT_LOGW_PRINT("Invliad operand index %d", operand_index);
        return NNA_ERROR_CODE(BAD_DATA);
    }
    // TODO: Update other attrs
    if (operand->dimensions.size() == 0) {
        operand->dimensions = operand_type->dimensions;
    } else {
        if (operand->dimensions.size() != operand_type->dimensions.size()) {
            return NNA_ERROR_CODE(BAD_DATA);
        }
        for (int i = 0; i < (int)operand->dimensions.size(); i++) {
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
    OS::getEnv(FAST_MODE, fastVal);
    if (1 == fastVal) {
        if (!fast_model) {
            NNRT_LOGW_PRINT("VIV_FAST_MODE has been setted, fast mode can't be setted false.");
        }
        relaxed_ = true;
    } else {
        relaxed_ = fast_model;
    }
}

void Model::echo() {
    NNRT_LOGD_PRINT("Software Version: %s", VERSION::as_str());
    NNRT_LOGD_PRINT("================== model info ==================");
    NNRT_LOGD_PRINT("================== operands ==================");
    for (auto it : operands_) {
        it.second->echo(it.first);
    }
    NNRT_LOGD_PRINT("================== operations ==================");
    for (auto it : operations_) {
        it.second->echo(it.first);
    }
    NNRT_LOGD_PRINT("================================================");
}

void Model::freezeCompile() {
    signature_ = generateSignature();
    compiled_ = true;
}

std::string Model::generateSignature() {
    /**
     * @todo Add more info to signature
     */
    std::ostringstream result;
    for (uint32_t i : inputIndexes()) {
        op::OperandPtr operand = this->operand(i);
        std::copy(operand->dimensions.begin(),
                  operand->dimensions.end(),
                  std::ostream_iterator<uint32_t>(result, ","));
    }
    return result.str();
}

bool Model::replace_model_with_nbg() {
    if (cache_handle_ == -1 || cache_size_ == 0) return true;
    operations_.clear();
    operation_unique_id_ = 0;

    for (auto i = operands_.begin(); i != operands_.end();) {
        if (std::find(input_indexes_.begin(), input_indexes_.end(), i->first) ==
                input_indexes_.end() &&
            std::find(output_indexes_.begin(), output_indexes_.end(), i->first) ==
                output_indexes_.end()) {
            i = operands_.erase(i);
        } else {
            ++i;
        }
    }

    const uint32_t* inputs = input_indexes_.data();
    uint32_t input_size = input_indexes_.size();
    const uint32_t* outputs = output_indexes_.data();
    uint32_t output_size = output_indexes_.size();
    uint32_t* out_index = nullptr;
    addOperation(OperationType::NBG, inputs, input_size, outputs, output_size, out_index);
    return allocate_cache_memory(cache_size_);
}

bool Model::set_cache_handle(int handle) {
    if (-1 == handle) return false;
    bool status = false;

#ifdef __linux__
    if (cache_handle_ != -1) {
        NNRT_LOGD_PRINT(
            "Close previous model cache, it's safe because different compilation should have "
            "some model structure\n");
        close(cache_handle_);
        cache_handle_ = -1;
    }

    auto is_writeable = [handle]() {
        int test_data = 0x5A5A5A5A;
        size_t length = write(handle, &test_data, sizeof(test_data));
        lseek(handle, 0, SEEK_SET);
        ftruncate(handle, 0);  // reset file content

        return (length == sizeof(test_data));
    };

    cache_size_ = lseek(handle, 0, SEEK_END);
    lseek(handle, 0, SEEK_SET);

    if (cache_size_ || is_writeable()) {
        cache_handle_ = dup(handle);
        status = (cache_handle_ != -1);
    } else {
        NNRT_LOGD_PRINT("Set cache handle failed");
        cache_handle_ = -1;
        cache_size_ = 0;
    }
#endif
    return status;
}

bool Model::allocate_cache_memory(int size) {
#ifdef __linux__
    auto flag = (-1 == cache_handle_) ? (MAP_SHARED | MAP_ANONYMOUS) : (MAP_SHARED);
    cache_memory_ = mmap(nullptr, size, PROT_READ, flag, cache_handle_, 0);
    return (cache_memory_ == nullptr) ? false : true;
#endif
    return false;
}

}  // namespace nnrt
