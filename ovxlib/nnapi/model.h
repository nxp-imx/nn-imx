#ifndef __OVXLIB_MODEL_H__
#define __OVXLIB_MODEL_H__

#include <vector>
#include <map>

#include "memory.h"
#include "operation.h"
#include "operand.h"

namespace ovxlib
{
class Model
{
    public:
        Model();
        ~Model();

        std::vector<uint32_t>& inputIndexes() {return input_indexes_;}

        const std::vector<uint32_t>& outputIndexes() {return output_indexes_;}

        uint32_t inputIndex(uint32_t index) {
            if (index < input_indexes_.size()) {
                return input_indexes_[index];
            }
            return OVXLIB_INVALID_OPERAND_INDEX;
        }

        uint32_t outputIndex(uint32_t index) {
            if (index < output_indexes_.size()) {
                return output_indexes_[index];
            }
            return OVXLIB_INVALID_OPERAND_INDEX;
        }

        Operand* operand(uint32_t index) {
            if (operands_.find(index) == operands_.end()) {
                return nullptr;
            }
            return operands_[index];
        }

        Operation* operation(uint32_t index) {
            if (operations_.find(index) == operations_.end()) {
                return nullptr;
            }
            return operations_[index];
        }

        std::map<uint32_t, Operand*>& operands() {return operands_;}

        std::map<uint32_t, Operation*>& operations() {return operations_;}

        size_t getOperandSize() {return operands_.size();}

        size_t getOperationSize() {return operations_.size();}

        bool isInput(uint32_t index);

        bool isOutput(uint32_t index);

        void identifyInputsAndOutputs(const uint32_t* inputs_ptr,
                uint32_t input_count, const uint32_t* outputs_ptr,
                uint32_t output_count);

        Operation* addOperation(OperationType code,
                const uint32_t* inputs, uint32_t input_size,
                const uint32_t* outputs, uint32_t output_size,
                int* out_index = nullptr);

        Operation* addOperation(Operation* new_operation,
                int* out_index = nullptr);

        Operand* cloneOperand(Operand* operand, int* out_index = nullptr);

        Operand* addOperand(Operand* new_operand = nullptr,
                int * out_index = nullptr);

        int getOperandIndex(Operand* operand);

        template<typename T>
        T* getBuffer(const mem_pool::shared_ref& ref) {
            void* data = nullptr;
            if (ref) {
                data = const_cast<void*>(ref->address_);
                VSILOGI("Read from shared reference");
            }
            else {
                VSILOGE("Error while getBuffer <<<<<<<<<<<<<<,");
            }
            return static_cast<T*>(data);
        }

        template<typename T>
        T* getModifiableBuffer(Operand* operand) {
            // void* data = nullptr;
            // uint32_t index = operand->location.poolIndex;
            // uint32_t offset = operand->location.offset;
            // size_t length = operand->location.length;
            // if (index < pool_.size()) {
            //     if (!pool_[index]->modifiable()) {
            //         //resetModifiableBuffer(operand);
            //         //TODO: Decounter buffer reference.
            //         index = pool_.addBuffer(length);
            //         operand->location.poolIndex = index;
            //         operand->location.offset = 0;
            //     }
            //     data = pool_[index]->data(offset);
            // }
            // return static_cast<T*>(data);
        }

        bool addBuffer(const void* buffer, size_t length, uint32_t index);

        int setOperandValue(Operand* operand, const void* buffer, size_t length);

        int setOperandValue(uint32_t operand_index, const void* buffer, size_t length);

        int setOperandValueFromMemory(Operand* operand, const Memory * memory,
                size_t offset, size_t length);

        int setOperandValueFromMemory(uint32_t operand_index, const Memory * memory,
                size_t offset, size_t length);

        void relax(bool fast_model) {
            relaxed_ = fast_model;
        }

        void finish() { finalized_ = true; }

        void freezeCompile();

        bool isRelaxed() {
            return relaxed_;
        }

        bool isFinished() {
            return finalized_;
        }

        bool isCompiled() {
            return compiled_;
        }

        bool validate();

        bool isValid() {
            return valid_;
        }

        std::vector<Operand *> getOperands(const std::vector<uint32_t> & indexes);

        std::vector<uint32_t> getConsumers(Operand* operand);

        std::vector<uint32_t> getProducers(Operand* operand);

        void removeOperand(uint32_t index);

        int updateOperand(uint32_t index, const Operand* operand_type);

        void echo();

        std::string signature() {
            if (!isCompiled()) {
                VSILOGW("Uncompiled model doesn't have the signature.");
                return "Not Finished";
            }
            return signature_;
        }

        std::string generateSignature();

    private:
        uint32_t operand_unique_id_ = 0;
        uint32_t operation_unique_id_ = 0;
        bool relaxed_;          /* the flag to run fp16 data,instead of fp32*/
        bool finalized_{false};
        bool compiled_{false};
        bool valid_{false};
        std::string signature_;
        std::map<uint32_t, Operation*> operations_;
        std::map<uint32_t, Operand*> operands_;
        std::vector<uint32_t> input_indexes_;
        std::vector<uint32_t> output_indexes_;
        void checkProcess();

};
}

#endif
