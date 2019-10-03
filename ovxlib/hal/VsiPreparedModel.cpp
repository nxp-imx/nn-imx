#include "VsiPreparedModel.h"

#include <sys/mman.h>
#include <sys/system_properties.h>

#include "../nnapi/event.h"
#include "OperationsUtils.h"
#include "Utils.h"
#if ANDROID_SDK_VERSION > 27
#include "ValidateHal.h"
#endif

namespace android {
namespace nn {
namespace vsi_driver {

static ovxlib::OperationType op_code_mapping(OperationType op) {  // Android O 8.1 API LEVEL 27
    switch (op) {
#define MAP_OP(op)                             \
    case OperationType::op: {                  \
        LOG(INFO) << "add operation: " << #op; \
        return ovxlib::OperationType::op;      \
    }
        MAP_OP(ADD);
        MAP_OP(CONV_2D);
        MAP_OP(DEPTHWISE_CONV_2D);
        MAP_OP(RELU);
        MAP_OP(RESHAPE);
        MAP_OP(FULLY_CONNECTED);
        MAP_OP(SOFTMAX);
        MAP_OP(CONCATENATION);
        MAP_OP(AVERAGE_POOL_2D);
        MAP_OP(MAX_POOL_2D);
        MAP_OP(MUL);
        MAP_OP(RELU1);
        MAP_OP(RELU6);
        MAP_OP(TANH);
        MAP_OP(LOGISTIC);
        MAP_OP(FLOOR);
        MAP_OP(DEQUANTIZE);
        MAP_OP(SPACE_TO_DEPTH);
        MAP_OP(DEPTH_TO_SPACE);
        MAP_OP(L2_NORMALIZATION);
        MAP_OP(RESIZE_BILINEAR);
        MAP_OP(LOCAL_RESPONSE_NORMALIZATION);
        MAP_OP(EMBEDDING_LOOKUP);
        MAP_OP(RNN);
        MAP_OP(HASHTABLE_LOOKUP);
        MAP_OP(LSTM);
        MAP_OP(SVDF);
        MAP_OP(LSH_PROJECTION);
        MAP_OP(L2_POOL_2D);
#if ANDROID_SDK_VERSION > 27
        MAP_OP(BATCH_TO_SPACE_ND);
        MAP_OP(DIV);
        MAP_OP(MEAN);
        MAP_OP(PAD);
        MAP_OP(SPACE_TO_BATCH_ND);
        MAP_OP(SQUEEZE);
        MAP_OP(STRIDED_SLICE);
        MAP_OP(SUB);
        MAP_OP(TRANSPOSE);
#endif
#undef MAP_OP

        default:
            LOG(ERROR) << "Unknown operation code:" << static_cast<int32_t>(op);
            break;
    }

    return ovxlib::OperationType::NONE;
};

static ovxlib::OperandType operand_mapping(OperandType code) {
#define MAP_OPERAND(code)                      \
    case OperandType::code: {                  \
        LOG(INFO) << "add operand: " << #code; \
        return ovxlib::OperandType::code;      \
    }

    switch (code) {
        MAP_OPERAND(FLOAT32);
        MAP_OPERAND(INT32);
        MAP_OPERAND(UINT32);
        MAP_OPERAND(TENSOR_FLOAT32);
        MAP_OPERAND(TENSOR_INT32);
        MAP_OPERAND(TENSOR_QUANT8_ASYMM);
        default:
            break;
    }

#undef MAP_OPERAND

    return ovxlib::OperandType::NONE;
}

void VsiPreparedModel::release_rtinfo(std::vector<VsiRTInfo>& rtInfos){
    while(!rtInfos.empty()){
        auto &rt = rtInfos.back();
        if("mmap_fd" == rt.mem_type)
            rt.vsi_mem.reset();
        rtInfos.pop_back();
    }
}

int VsiPreparedModel::map_rtinfo_from_hidl_memory(const hidl_vec<hidl_memory>& pools,
            std::vector<VsiRTInfo>& rtInfos){
        rtInfos.clear();
        rtInfos.resize(pools.size());

    for(size_t i = 0; i < pools.size(); i++){
        auto & hidl_memory = pools[i];
        auto & rt = rtInfos[i];

        std::shared_ptr<ovxlib::Memory>  vsi_mem = nullptr;
        sp<IMemory> shared_mem = nullptr;
        uint8_t *buffer = nullptr;

        if ("ashmem" == hidl_memory.name()) {
                shared_mem = mapMemory(hidl_memory);
                assert(shared_mem);
                shared_mem->read();
                buffer =
                    reinterpret_cast<uint8_t*>(static_cast<void*>(shared_mem->getPointer()));
        }else if ("mmap_fd" == hidl_memory.name()) {
                size_t size = hidl_memory.size();
                int fd = hidl_memory.handle()->data[0];
                int mode = hidl_memory.handle()->data[1];
                size_t offset = getSizeFromInts(hidl_memory.handle()->data[2], hidl_memory.handle()->data[3]);

                vsi_mem = std::make_shared<ovxlib::Memory>();
                vsi_mem ->readFromFd(size, mode, fd, offset);
        }else{
                LOG(ERROR) << "invalid hidl_memory";
                return ANEURALNETWORKS_BAD_DATA;
        }

        rt.shared_mem = shared_mem;
        rt.mem_type = std::string(hidl_memory.name());
        rt.ptr = buffer;
        rt.vsi_mem = vsi_mem;
    }
    return ANEURALNETWORKS_NO_ERROR;
}

void VsiPreparedModel::fill_operand_value(ovxlib::Operand* ovx_operand, const Operand& hal_operand) {
    switch (hal_operand.lifetime) {
        case OperandLifeTime::MODEL_INPUT:
        case OperandLifeTime::MODEL_OUTPUT:
        case OperandLifeTime::TEMPORARY_VARIABLE:
            // Skip lifetime is TEMPORARY_VARIABLE, MODEL_INPUT, MODEL_OUTPUT, or NO_VALUE
            break;
        case OperandLifeTime::NO_VALUE: {
            native_model_->setOperandValue(ovx_operand, nullptr, 0);
        } break;
        case OperandLifeTime::CONSTANT_COPY: {
            const auto& location = hal_operand.location;
            native_model_->setOperandValue(
                ovx_operand, model_.operandValues.data() + location.offset, location.length);
            break;
        } break;
        case OperandLifeTime::CONSTANT_REFERENCE: {

            const auto& location = hal_operand.location;
            auto &rt_info = const_buffer_[location.poolIndex];
            if("ashmem" == rt_info.mem_type){
                const uint8_t* buffer = rt_info.ptr;
                native_model_->setOperandValue(ovx_operand, buffer + location.offset, location.length);
            }else if ("mmap_fd" == rt_info.mem_type) {
                native_model_->setOperandValueFromMemory(
                    ovx_operand, rt_info.vsi_mem.get(), location.offset, location.length);
            }
        } break;
    }
}

void VsiPreparedModel::construct_ovx_operand(ovxlib::Operand* ovx_operand, const Operand& hal_operand) {
    ovx_operand->type = operand_mapping(hal_operand.type);
    ovx_operand->quant.scalar.scale = hal_operand.scale;
    ovx_operand->quant.scalar.zeroPoint = hal_operand.zeroPoint;
    ovx_operand->dimensions = hal_operand.dimensions;
    // tensor shape with zero should set as Null, ovx won't create concrete tensor
    // for this operand
//    for (auto d : ovx_operand->dimensions) {
//        if (0 == d) {
//            ovx_operand->setNull();
//            break;
//        }
//    }

    // TODO: add check error
    switch (ovx_operand->type) {
        case ovxlib::OperandType::FLOAT32:
        case ovxlib::OperandType::INT32:
        case ovxlib::OperandType::UINT32:
            break;
        case ovxlib::OperandType::TENSOR_FLOAT32:
        case ovxlib::OperandType::TENSOR_INT32:
        case ovxlib::OperandType::TENSOR_QUANT8_ASYMM: {
            break;
        }
        default:
            break;
    }
}

Return<ErrorStatus> VsiPreparedModel::Create(const Model& model) {
    // [0] validate HAL::Model, return ErrorCode if validate failed
    // For scalar operand, dimension must be 0
    // [1] create async procedure to prepare model
    // [1.0] convert HAL model to ovxlib::Model
    LOG(INFO) << __FUNCTION__;

    map_rtinfo_from_hidl_memory(model.pools, const_buffer_);

    // add operand and set its value
    for (const auto& hal_operand : model.operands) {
        int registered_idx = 0;
        ovxlib::Operand* ovx_operand = native_model_->addOperand(nullptr, &registered_idx);

        construct_ovx_operand(ovx_operand, hal_operand);
        fill_operand_value(ovx_operand, hal_operand);
    }

    for (const auto& hal_op : model.operations) {
        ovxlib::Operation* ovx_op =
            native_model_->addOperation(op_code_mapping(hal_op.type) /* Operation Type*/,
                                        &hal_op.inputs[0],    /*inputs */
                                        hal_op.inputs.size(), /*num of inputs */
                                        &hal_op.outputs[0],   /*outputs */
                                        hal_op.outputs.size() /*num of outputs */
                                        );
        ovx_op->setOperandLayout(ovxlib::OperandLayout::NHWC);
    }

    native_model_->finish();
    std::vector<uint32_t> inputs = model.inputIndexes;
    std::vector<uint32_t> outputs = model.outputIndexes;
    native_model_->identifyInputsAndOutputs(inputs.data(), inputs.size(), outputs.data(), outputs.size());
    native_compile_ = std::make_shared<ovxlib::Compilation>(native_model_.get());

    return ErrorStatus::NONE;
}

Return<ErrorStatus> VsiPreparedModel::execute(const Request& request,
                                              const sp<IExecutionCallback>& callback) {
    LOG(INFO) << __FUNCTION__;

    map_rtinfo_from_hidl_memory(request.pools, io_buffer_);

    if (!native_exec_) native_exec_ = std::make_shared<ovxlib::Execution>(native_compile_.get());

    std::vector<hidl_memory> io_pools = request.pools;
    std::vector<RequestArgument> input_args = request.inputs;
    std::vector<RequestArgument> output_args = request.outputs;

    enum IO { INPUT = 0, OUTPUT };
    // Adjust the runtime info for the arguments passed to the model,
    // modifying the buffer location, and possibly the dimensions.
    auto update_operand = [this](const std::vector<uint32_t>& indexes,
                                 const hidl_vec<RequestArgument>& arguments) {
        nnAssert(indexes.size() == arguments.size());
        auto ovx_operands = native_model_->getOperands(indexes);
        for (size_t i = 0; i < indexes.size(); i++) {
            const RequestArgument& request_arg = arguments[i];
            auto& ovx_operand = ovx_operands[i];
            if (request_arg.dimensions.size() > 0) {
                // It's the responsibility of the caller to validate that
                // arg.dimensions only modifies the dimensions that were
                // unspecified in the model.  That's the case in SampleDriver.cpp
                // with the call to validateRequest().
                // TODO: after revaluing the dimension, model should be re-compiled
                ovx_operand->dimensions = request_arg.dimensions;
            }
        }
    };

    // Adjust the runtime info for the arguments passed to the model,
    // modifying the buffer location, and possibly the dimensions.
    auto updateForArguments = [this, &request](
        const std::vector<uint32_t>& indexes, const hidl_vec<RequestArgument>& arguments, IO flag) {
        nnAssert(indexes.size() == arguments.size());
        auto ovx_operands = native_model_->getOperands(indexes);
        for (size_t i = 0; i < indexes.size(); i++) {
            const RequestArgument& request_arg = arguments[i];
            auto& ovx_operand = ovx_operands[i];
            if (request_arg.hasNoValue) {
                if (flag == IO::INPUT)
                    native_exec_->setInput(i, nullptr /*operand */, nullptr, 0);
                else
                    native_exec_->setOutput(i, nullptr, nullptr, 0);
            } else {
                auto location = request_arg.location;
                auto poolIndex = location.poolIndex;
                nnAssert(poolIndex < request.pools.size());

                auto &rt_info = io_buffer_[poolIndex];
                if("ashmem" == rt_info.mem_type){
                    uint8_t* buffer = rt_info.ptr;
                    if (flag == IO::INPUT)
                        native_exec_->setInput(i, ovx_operand, buffer + location.offset, location.length);
                    else
                        native_exec_->setOutput(i, ovx_operand, buffer + location.offset, location.length);
                }else if ("mmap_fd" == rt_info.mem_type) {
                    auto &vsi_mem = rt_info.vsi_mem;
                    if (flag == IO::INPUT)
                        native_exec_->setInputFromMemory(
                            i, ovx_operand, vsi_mem.get(), location.offset, location.length);
                    else
                        native_exec_->setOutputFromMemory(
                            i, ovx_operand, vsi_mem.get(), location.offset, location.length);
                }
             }
        }
    };

    update_operand(model_.inputIndexes, input_args);
    update_operand(model_.outputIndexes, output_args);
    if (!native_model_->isCompiled()) {
        native_compile_->run();
    }

    updateForArguments(model_.inputIndexes, input_args, IO::INPUT);
    updateForArguments(model_.outputIndexes, output_args, IO::OUTPUT);

    int error = native_exec_->compute();
    native_exec_.reset();
    release_rtinfo(io_buffer_);

    if(error != ANEURALNETWORKS_NO_ERROR){
        callback->notify(ErrorStatus::INVALID_ARGUMENT);
        return ErrorStatus::INVALID_ARGUMENT;
    }

    callback->notify(ErrorStatus::NONE);
    return ErrorStatus::NONE;
}
}
}
}
