
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

#define LOG_TAG "VsiDevice"

#include "VsiDevice.h"
#include "VsiPreparedModel.h"

#include "HalInterfaces.h"

#if ANDROID_SDK_VERSION > 27
#include "ValidateHal.h"
#endif

#include <android-base/logging.h>
#include <hidl/LegacySupport.h>
#include <thread>

namespace android {
namespace nn {
namespace vsi_driver {

Return<ErrorStatus> VsiDevice::prepareModel_1_1(const Model& model,
#if ANDROID_SDK_VERSION > 27
                                                ExecutionPreference preference,
#endif
                                                const sp<IPreparedModelCallback>& callback) {
    if (VLOG_IS_ON(DRIVER)) {
        VLOG(DRIVER) << "prepareModel";
        logModelToInfo(model);
    }
    if (callback.get() == nullptr) {
        LOG(ERROR) << "invalid callback passed to prepareModel";
        return ErrorStatus::INVALID_ARGUMENT;
    }
    if (!validateModel(model)) {
        callback->notify(ErrorStatus::INVALID_ARGUMENT, nullptr);
        return ErrorStatus::INVALID_ARGUMENT;
    }

    // TODO: make asynchronous later
    sp<VsiPreparedModel> preparedModel = new VsiPreparedModel(model);
    if (!preparedModel.get()) {
        callback->notify(ErrorStatus::INVALID_ARGUMENT, nullptr);
        return ErrorStatus::INVALID_ARGUMENT;
    }
    Return<void> returned = callback->notify(ErrorStatus::NONE, preparedModel);
    if (!returned.isOk()) {
        LOG(ERROR) << " hidl callback failed to return properly: " << returned.description();
    }
    return ErrorStatus::NONE;
}

Return<DeviceStatus> VsiDevice::getStatus() {
    VLOG(DRIVER) << "getStatus()";
    return DeviceStatus::AVAILABLE;
}

int VsiDevice::run() {
    // TODO: Increase ThreadPool to 4 ?
    android::hardware::configureRpcThreadpool(1, true);
    if (registerAsService(name_) != android::OK) {
        LOG(ERROR) << "Could not register service";
        return 1;
    }
    android::hardware::joinRpcThreadpool();

    return 1;
}

}  // namespace ovx_driver
}  // namespace nn
}  // namespace android
