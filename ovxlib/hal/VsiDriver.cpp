/****************************************************************************
 *
 *    Copyright (c) 2005 - 2018 by Vivante Corp.  All rights reserved.
 *
 *    The material in this file is confidential and contains trade secrets
 *    of Vivante Corporation. This is proprietary information owned by
 *    Vivante Corporation. No part of this work may be disclosed,
 *    reproduced, copied, transmitted, or used in any way for any purpose,
 *    without the express written permission of Vivante Corporation.
 *
 *****************************************************************************/

#define LOG_TAG "VsiDriver"

#include "VsiDevice.h"

#include "HalInterfaces.h"
#include "Utils.h"

#if ANDROID_SDK_VERSION > 27
#include "ValidateHal.h"
#endif

#include <android-base/logging.h>
#include <hidl/LegacySupport.h>
#include <thread>

namespace android {
namespace nn {
namespace vsi_driver {

class VsiDriver : public VsiDevice {
   public:
    VsiDriver() : VsiDevice("vsi-npu") {}
    Return<void> getCapabilities(getCapabilities_cb _hidl_cb) ;
    Return<void> getSupportedOperations(const V1_0::Model& model, getSupportedOperations_cb cb) ;

#if ANDROID_SDK_VERSION > 27
    Return<void> getCapabilities_1_1(getCapabilities_1_1_cb _hidl_cb) ;
    Return<void> getSupportedOperations_1_1(const V1_1::Model& model,
                                            getSupportedOperations_1_1_cb cb) ;
#else
    Return<void> getCapabilities_1_1(getCapabilities_cb _hidl_cb) ;
    Return<void> getSupportedOperations_1_1(const Model& model,
                                            getSupportedOperations_cb cb) ;
#endif

};

Return<void> VsiDriver::getCapabilities(getCapabilities_cb cb) {
#if ANDROID_SDK_VERSION > 27
    return getCapabilities_1_1([&](ErrorStatus error, const V1_1::Capabilities& capabilities) {
        cb(error, convertToV1_0(capabilities));
    });
#else
    return getCapabilities_1_1([&](ErrorStatus error, const V1_0::Capabilities& capabilities) {
        cb(error, capabilities);
    });
#endif
}
#if ANDROID_SDK_VERSION > 27
Return<void> VsiDriver::getCapabilities_1_1(getCapabilities_1_1_cb cb) {
#else
Return<void> VsiDriver::getCapabilities_1_1(getCapabilities_cb cb) {
#endif
    android::nn::initVLogMask();
    VLOG(DRIVER) << "getCapabilities()";
    Capabilities capabilities = {
        .float32Performance = {.execTime = 0.9f, .powerUsage = 0.9f},
        .quantized8Performance = {.execTime = 0.9f, .powerUsage = 0.9f},
#if ANDROID_SDK_VERSION > 27
        .relaxedFloat32toFloat16Performance = {.execTime = 0.5f, .powerUsage = 0.5f}
#endif
         };
    cb(ErrorStatus::NONE, capabilities);
    return Void();
}

Return<void> VsiDriver::getSupportedOperations(const V1_0::Model& model, getSupportedOperations_cb cb) {
#if ANDROID_SDK_VERSION > 27
    return getSupportedOperations_1_1(convertToV1_1(model), cb);
#else
    return getSupportedOperations_1_1(model, cb);
#endif
}

#if ANDROID_SDK_VERSION > 27
Return<void> VsiDriver::getSupportedOperations_1_1(const Model& model,
                                                   getSupportedOperations_1_1_cb cb) {
#else
Return<void> VsiDriver::getSupportedOperations_1_1(const Model& model,
                                                   getSupportedOperations_cb cb) {
#endif
    VLOG(DRIVER) << "getSupportedOperations()";
    if (validateModel(model)) {
        const size_t count = model.operations.size();
        std::vector<bool> supported(count, true);
        cb(ErrorStatus::NONE, supported);
    } else {
        std::vector<bool> supported;
        cb(ErrorStatus::INVALID_ARGUMENT, supported);
    }
    return Void();
}

}  // namespace ovx_driver

}  // namespace nn
}  // namespace android

using android::nn::vsi_driver::VsiDriver;
using android::sp;

int main() {
    sp<VsiDriver> driver(new VsiDriver());
    return driver->run();
}
