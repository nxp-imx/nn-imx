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

#define LOG_TAG "VsiDriver"

#if ANDROID_SDK_VERSION > 28
#include "VsiDevice1_2.h"
#elif ANDROID_SDK_VERSION > 27
#include "VsiDevice.h"
#endif

#include "HalInterfaces.h"
#include "Utils.h"

#if ANDROID_SDK_VERSION > 27
#include "ValidateHal.h"
#endif

#include <sys/system_properties.h>
#include <android-base/logging.h>
#include <hidl/LegacySupport.h>
#include <thread>

namespace android {
namespace nn {
namespace vsi_driver {

class VsiDriver : public VsiDevice {
   public:
    VsiDriver() : VsiDevice("vsi-npu") {initalizeEnv();}
    Return<void> getCapabilities(getCapabilities_cb _hidl_cb) ;
    Return<void> getSupportedOperations(const V1_0::Model& model, getSupportedOperations_cb cb) ;

#if ANDROID_SDK_VERSION > 27
    Return<void> getCapabilities_1_1(getCapabilities_1_1_cb _hidl_cb) ;
    Return<void> getSupportedOperations_1_1(const V1_1::Model& model,
                                            getSupportedOperations_1_1_cb cb) ;
#endif

#if ANDROID_SDK_VERSION > 28
    Return<void> getCapabilities_1_2(V1_2::IDevice::getCapabilities_1_2_cb _hidl_cb) override;
    Return<void> getSupportedOperations_1_2(const V1_2::Model& model,
                                            V1_2::IDevice::getSupportedOperations_1_2_cb cb) ;

    Return<void>
        getVersionString(getVersionString_cb _hidl_cb) override{
        _hidl_cb(ErrorStatus::NONE, "android hal vsi npu 1.2 alpha");
        return Void();
    };

    Return<void>
        getType(getType_cb _hidl_cb) override{
        _hidl_cb(ErrorStatus::NONE, V1_2::DeviceType::ACCELERATOR);
        return Void();
    };

    Return<void>
        getSupportedExtensions(getSupportedExtensions_cb _hidl_cb)override{
        _hidl_cb(ErrorStatus::NONE, {/* No extensions. */});
        return Void();
    };

    Return<void>
        getNumberOfCacheFilesNeeded(getNumberOfCacheFilesNeeded_cb _hidl_cb)override{
        // Set both numbers to be 0 for cache not supported.
        _hidl_cb(ErrorStatus::NONE, /*numModelCache=*/0, /*numDataCache=*/0);
        return Void();
    };
#endif

   private:
   int32_t disable_float_feature_; // switch that float-type running on hal
   private:
    void initalizeEnv()
    {
        disable_float_feature_ = 0;
        char env[100] = {0};
        int ireturn = __system_property_get("DISABLE_FLOAT_FEATURE", env);
        if(ireturn)
        {
            disable_float_feature_= atoi(env);
            if(disable_float_feature_)
                LOG(INFO)<< "float-type model will not running on hal";
        }
    }
    template <typename T_model, typename T_getSupportOperationsCallback>
    Return<void> getSupportedOperationsBase(const T_model& model,
                                            T_getSupportOperationsCallback cb) {
           LOG(INFO)<< "getSupportedOperations";
          if(validateModel(model)) {
            const size_t count = model.operations.size();
            std::vector<bool> supported(count, true);
            for (size_t i = 0; i < count; i++) {
                 supported[i] = getSupportedOperation(i, model);
            }
            cb(ErrorStatus::NONE, supported);
        } else {
            LOG(ERROR)<< "invalid model";
            std::vector<bool> supported;
            cb(ErrorStatus::INVALID_ARGUMENT, supported);
        }
        LOG(INFO)<< "getSupportedOperations exit";
        return Void();
    };

    template<typename T_Model>
    bool getSupportedOperation(const size_t operation_index,
                                               const T_Model& model);
};

Return<void> VsiDriver::getCapabilities(getCapabilities_cb cb) {
    V1_0::Capabilities capabilities;
    if(disable_float_feature_){
        capabilities.float32Performance = {.execTime = 1.9f, .powerUsage = 1.9f};
        capabilities.quantized8Performance = {.execTime = 0.9f, .powerUsage = 0.9f};
    }else{
        capabilities.float32Performance = {.execTime = 0.9f, .powerUsage = 0.9f};
        capabilities.quantized8Performance = {.execTime = 0.9f, .powerUsage = 0.9f};
    }
    cb(ErrorStatus::NONE, capabilities);
    return Void();
}

Return<void> VsiDriver::getSupportedOperations(const V1_0::Model& model, getSupportedOperations_cb cb) {
    return getSupportedOperationsBase(model, cb);
}

#if ANDROID_SDK_VERSION > 27
Return<void> VsiDriver::getCapabilities_1_1(getCapabilities_1_1_cb cb) {
    V1_1::Capabilities capabilities;
    if(disable_float_feature_){
        capabilities.float32Performance = {.execTime = 1.9f, .powerUsage = 1.9f};
        capabilities.quantized8Performance = {.execTime = 0.9f, .powerUsage = 0.9f};
        capabilities.relaxedFloat32toFloat16Performance = {.execTime = 1.5f, .powerUsage = 1.5f};
    }else{
        capabilities.float32Performance = {.execTime = 0.9f, .powerUsage = 0.9f};
        capabilities.quantized8Performance = {.execTime = 0.9f, .powerUsage = 0.9f};
        capabilities.relaxedFloat32toFloat16Performance = {.execTime = 0.5f, .powerUsage = 0.5f};
    }
    cb(ErrorStatus::NONE, capabilities);
    return Void();
}

Return<void> VsiDriver::getSupportedOperations_1_1(const V1_1::Model& model,
                                                   getSupportedOperations_1_1_cb cb) {
    return getSupportedOperationsBase(model, cb);
}
#endif

    template<typename T_Model>
    bool VsiDriver::getSupportedOperation(const size_t operation_index, const T_Model& model){
#if ANDROID_SDK_VERSION > 28
        const auto &model_1_2 = convertToV1_2(model);
        const auto &operation = model_1_2.operations[operation_index];

        auto checkSupportedOperand = [](auto &operand)->bool{
             bool isSupported = true;
             switch(operand.type){
                    //API 29 newly added operand
                    case OperandType::BOOL:
                    case OperandType::TENSOR_QUANT16_SYMM:
                    case OperandType::TENSOR_FLOAT16:
                    case OperandType::TENSOR_BOOL8:
                    case OperandType::FLOAT16:
                    case OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL:
                    case OperandType::TENSOR_QUANT16_ASYMM:
                    case OperandType::TENSOR_QUANT8_SYMM:
                        isSupported = false;
                        break;
                    default:
                        break;
                 }
             return isSupported;
         };

         auto getOpeandPtr = [&model_1_2](auto &operand)->auto{
            auto& location = operand.location;
            return model_1_2.operandValues.data() + location.offset;
         };

        // TODO: [NNRT-1] Support static shape inference for NNAPI 1.2
         for(size_t i = 0; i < operation.outputs.size(); i++){
            auto &dims = model_1_2.operands[operation.outputs[i]].dimensions;
            for(auto dimIndex: dims)
                if(dimIndex == 0)
                    return false;
        }

         // TODO: nnapi 1.2 new operand type
         for(size_t i = 0; i < operation.inputs.size(); i++){
              auto & operand = model_1_2.operands[operation.inputs[i]];
              if(false == checkSupportedOperand(operand))
                return false;
         }
        for(size_t i = 0; i < operation.outputs.size(); i++){
              auto & operand = model_1_2.operands[operation.outputs[i]];
              if(false == checkSupportedOperand(operand))
                return false;
         }
        switch (operation.type)
        {
            //TODO: check API 28 op new feature
            case OperationType::LSH_PROJECTION:{
                auto typePtr = getOpeandPtr(model_1_2.operands[operation.inputs[3]]);
                if(3 == *(int32_t*)typePtr)
                    return false;
                break;
                }
            case OperationType::TANH:{
                if(OperandType::TENSOR_FLOAT32 != model_1_2.operands[operation.inputs[0]].type)
                    return false;
                break;
                }
            case OperationType::LSTM:{
                if(operation.inputs.size()>23)
                    return false;
                break;
                }
            case OperationType::RESIZE_BILINEAR:{
                auto & scalarOperand= model_1_2.operands[operation.inputs[1]];
                if(OperandType::INT32 != scalarOperand.type)
                    return false;
                break;
                }
            //to-do: check operand with operation
            //API 29 newly added operataion
            case OperationType::ABS:
            case OperationType::ARGMAX:
            case OperationType::ARGMIN:
            case OperationType::AXIS_ALIGNED_BBOX_TRANSFORM:
            case OperationType::BIDIRECTIONAL_SEQUENCE_LSTM:
            case OperationType::BIDIRECTIONAL_SEQUENCE_RNN:
            case OperationType::BOX_WITH_NMS_LIMIT:
            case OperationType::CAST:
            case OperationType::CHANNEL_SHUFFLE:
            case OperationType::DETECTION_POSTPROCESSING:
            case OperationType::EQUAL:
            case OperationType::EXP:
            case OperationType::EXPAND_DIMS:
            case OperationType::GATHER:
            case OperationType::GENERATE_PROPOSALS:
            case OperationType::GREATER:
            case OperationType::GREATER_EQUAL:
            case OperationType::GROUPED_CONV_2D:
            case OperationType::HEATMAP_MAX_KEYPOINT:
            case OperationType::INSTANCE_NORMALIZATION:
            case OperationType::LESS:
            case OperationType::LESS_EQUAL:
            case OperationType::LOGICAL_AND:
            case OperationType::LOGICAL_NOT:
            case OperationType::LOGICAL_OR:
            case OperationType::LOG_SOFTMAX:
            case OperationType::LOG:
            case OperationType::MAXIMUM:
            case OperationType::MINIMUM:
            case OperationType::NEG:
            case OperationType::NOT_EQUAL:
            case OperationType::PAD_V2:
            case OperationType::POW:
            case OperationType::PRELU:
            case OperationType::QUANTIZE:
            case OperationType::QUANTIZED_16BIT_LSTM:
            case OperationType::RANDOM_MULTINOMIAL:
            case OperationType::REDUCE_ALL:
            case OperationType::REDUCE_ANY:
            case OperationType::REDUCE_MAX:
            case OperationType::REDUCE_MIN:
            case OperationType::REDUCE_PROD:
            case OperationType::REDUCE_SUM:
            case OperationType::ROI_ALIGN:
            case OperationType::ROI_POOLING:
            case OperationType::RSQRT:
            case OperationType::SELECT:
            case OperationType::SIN:
            case OperationType::SLICE:
            case OperationType::SPLIT:
            case OperationType::SQRT:
            case OperationType::TILE:
            case OperationType::TOPK_V2:
            case OperationType::TRANSPOSE_CONV_2D:
            case OperationType::UNIDIRECTIONAL_SEQUENCE_LSTM:
            case OperationType::UNIDIRECTIONAL_SEQUENCE_RNN:
            case OperationType::RESIZE_NEAREST_NEIGHBOR:
                return false;
                break;
            default:
                return true;
        }
#endif
        return true;
    }

#if ANDROID_SDK_VERSION > 28
    Return<void> VsiDriver::getCapabilities_1_2(V1_2::IDevice::getCapabilities_1_2_cb _hidl_cb){
        static const PerformanceInfo kPerf = {.execTime = 0.9f, .powerUsage = 0.9f};
        V1_2::Capabilities capabilities;

        // Set the base value for all operand types
        capabilities.operandPerformance = nonExtensionOperandPerformance({FLT_MAX, FLT_MAX});

        // Load supported operand types
        update(&capabilities.operandPerformance, OperandType::TENSOR_QUANT8_ASYMM, kPerf);
        if(!disable_float_feature_){
            update(&capabilities.operandPerformance, OperandType::TENSOR_FLOAT32, kPerf);
            update(&capabilities.operandPerformance, OperandType::TENSOR_FLOAT16, kPerf);
        }
        _hidl_cb(ErrorStatus::NONE, capabilities);
        return Void();
    }

    Return<void> VsiDriver::getSupportedOperations_1_2(const V1_2::Model& model,
                                            V1_2::IDevice::getSupportedOperations_1_2_cb _hidl_cb){
        return getSupportedOperationsBase(model, _hidl_cb);
    }
 #endif
}  // namespace vsi_driver
}  // namespace nn
}  // namespace android

using android::nn::vsi_driver::VsiDriver;
using android::sp;

int main() {
    sp<VsiDriver> driver(new VsiDriver());
    return driver->run();
}
